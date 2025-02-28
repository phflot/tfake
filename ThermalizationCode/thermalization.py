import os
import torch.nn as nn
import functools
import torch.utils.data
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import os
from pathlib import Path
from os.path import join
import geomloss
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import kornia
import cv2
import torchvision
import segmentation_models_pytorch as smp
import scipy
import time
import math

from datasets import *
from augmentor import DenseAugmentor, ResizeAugmentor, additional_augmentation
from wrapper import Pix2PixDatasets
from thermalization_utils import *


class patch_extractor2:
    """Class to extract (sampled) patches from images"""
    def __init__(self, pz=10, sample_num=5000, resizer=torchvision.transforms.Resize(256), center=False, device="cuda:1"):
        self.unfolder = torch.nn.Unfold(pz)
        self.pz=pz
        self.sample_num = sample_num
        # Step 1: Initialize model with the best available weights
        weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(weights=weights).to(device)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.resizer = resizer
        self.center = center

    def manual_filter_background_old(self, im_batch_th, im_batch_rgb):
        """remove background"""
        im_batch = self.preprocess(im_batch_rgb)
        div_batch = 8
        if im_batch.shape[0] > div_batch:
            batch_size = im_batch.shape[0]
            prediction_ls = [self.model(im_batch[i * div_batch: (i+1) * div_batch])["out"] for i in range(0, int(batch_size/div_batch))]
            prediction = torch.cat(prediction_ls, dim=0)
        else:
            prediction = self.model(im_batch)["out"]
        normalized_masks = prediction.softmax(dim=1)
        re_mask = self.resizer(normalized_masks)
        re_mask = (re_mask[:, 0].detach().cpu().numpy() > .3).astype(bool)
        im_batch_th[:, 0, :, :][re_mask] = np.nan 
        return im_batch_th

    def manual_filter_background(self, im_batch_th):
        """remove background"""
        im_batch_th[im_batch_th == 0.] = np.nan 
        return im_batch_th
        
    def preseg_filter_background(self, im_batch_th, batch_seg):
        batch_seg = batch_seg.detach().cpu().numpy()
        re_mask = (batch_seg[:, :1, :, :]).astype(bool)
        im_batch_th[re_mask] = np.nan 
        return im_batch_th

    def extract(self, im_batch, batch_seg=None):
        """return patches of images in im_batch"""
        im_batch = im_batch.contiguous()
        with torch.no_grad():
            if batch_seg is None:
                im_batch = self.manual_filter_background(im_batch.clone())
            if batch_seg is not None:
                batch_seg = batch_seg.contiguous()
                im_batch = self.preseg_filter_background(im_batch.clone(), batch_seg.clone())
                #print(im_batch)
        patches = self.unfolder(im_batch.clone())
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape(-1, self.pz**2)
        patches = patches[~patches.isnan().all(axis=1)] # remove all black or all background patches
        patches[patches.isnan()] = 0.
        if self.sample_num > 0:
            patches = patches[torch.randperm(patches.shape[0])[:self.sample_num]]
        if self.center:
            patches = patches - patches.mean(dim=1).unsqueeze(-1)
        return patches

def get_mean_dict_dict():
    """
    reference values for temperatures in face segmentations
    see datasets.py for class labels
    """
    mean_dict_cold = {0: 0, 1: 13/20 ,2: 11.5/20, 3: 14/20,4: 14/20, 5: 11/20, 
                     6: 11/20, 7: 12/20, 8: 12/20, 
                     9: 15/20, 10: 12.5/20, 11: 12.5/20, 12: 14/20, 13: 10/20, 
                     14: 11/20 , 15: 10/20, 16: 0./20, 17: 8/20, 18: 8/20}

    mean_dict_warm = {0: 0, 1: 15/20 ,2: 15/20, 3: 15/20,4: 15/20, 5: 14/20, 
                         6: 14/20, 7: 15/20, 8: 15/20, 
                         9: 15/20, 10: 15/20, 11: 15/20, 12: 15/20, 13: 10/20, 
                         14: 12/20 , 15: 12/20, 16: 0./20, 17: 8/20, 18: 8/20}
    
    mean_dict_dict = {"cold": mean_dict_cold, "warm": mean_dict_warm}
    return mean_dict_dict

def calc_pdiv(pe, div, batch_th, batch_semi2th, batch_semi_mask, regdiv, shrinker_ls, device="cuda:1"):
    """
    pe: patch extractor
    div: Divergence function that takes samples as input
    batch_th: True thermal images
    batch_semi2th: thermal predictions
    batch_semi_mask: segmentations of predictions
    regdiv: regularization strength for Sinkhorn
    shrinker_ls: list of torchvision resizing transforms for multiple resolutions
    Approximate the Wasserstein distance betweeon patch distributions on multiple resolutions
    """
    pdiv = torch.tensor(0.).to(device)
    if regdiv > 0:
        for shrinker in shrinker_ls:
            patches_thermal = pe.extract(shrinker(batch_th)) # , im_batch_rgb=shrinker(batch_rgb))
            patches_semi2th = pe.extract(shrinker(batch_semi2th) , batch_seg=shrinker(batch_semi_mask))
            pdiv += div(patches_thermal, patches_semi2th)
    return pdiv

def calc_segreg(batch_semi2th, batch_semi, batch_semi_mask, mean_dict, HEIGHT, WIDTH, device="cuda:1"):
    """
    batch_semi: True RGB images
    batch_semi2th: thermal predictions
    batch_semi_mask: segmentations of predictions
    mean_dict: reference temperatures per segmentation areas
    HEIGHT, WIDTH: resolution
    Estimates the deviation from reference temperatures
    """
    sum_sqsegdev = torch.tensor(0.).to(device)
    pix_num = batch_semi.detach().flatten().size()[0]
    for key, val in mean_dict.items():
        val_norm = val 
        for bnum in range(batch_semi_mask.shape[0]):
            maskshape = batch_semi_mask[bnum].detach().shape
            batch_semi_seg = batch_semi2th[bnum][batch_semi_mask[bnum] == key]
            seg_size = batch_semi_seg.detach().flatten().size()[0]
            if seg_size > 0:
                sqsegdev = torch.abs(batch_semi_seg.mean() - torch.tensor(val_norm).to(device)) **2
                sum_sqsegdev += sqsegdev * torch.tensor(seg_size/(HEIGHT * WIDTH)).to(device)
                
    sum_sqsegdev = sum_sqsegdev/batch_semi_mask.shape[0]
    return sum_sqsegdev

def thermalization_main(fake_folder, sj_folder):
    """
    fake_folder: directory to FAKE dataset
    sj_folder: directory to DATASET with paired thermal/RGB images
    """
    print("Start Training")
    torch.manual_seed(43)
    np.random.seed(43)
    WIDTH = 256
    HEIGHT = 256
    epochs = 10
    resize_augmentor = ResizeAugmentor(output_shape=(WIDTH, HEIGHT))
    augmentor = DenseAugmentor(output_shape=(WIDTH, HEIGHT), scale=(1e7, 1e8))
    augmentor2 = DenseAugmentor(output_shape=(WIDTH, HEIGHT), scale=(1e7, 1e8))

    batch_size = 64
    heat_setting = "cold"
    mean_dict_dict = get_mean_dict_dict()
    mean_dict = mean_dict_dict[heat_setting]
    device = "cuda:1"
    show_images_bool = True
    show_sejong_bool = False # data privacy communicated to us by SEJONG authors
    regtemp = 1.
    regdiv = 0.01
    translator = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
        activation="sigmoid"
    )
    translator = translator.to(device)
    opt = torch.optim.Adam(translator.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    div = geomloss.SamplesLoss(loss="sinkhorn", p=2, debias=False, blur=.001)

    
    shrink_size_ls = [256, 128, 64, 32, 16]  
    shrinker_ls = [v2.Resize(red_size) for red_size in shrink_size_ls]
    jitter = v2.ColorJitter(brightness=.5, hue=.3, contrast=.5, saturation=.5)
    cinv = v2.RandomInvert(p=.9)
    gray = v2.RandomGrayscale(p=.1)
    kernelsize = 1 + math.ceil(HEIGHT/10 *.5) * 2
    blurrer = torchvision.transforms.GaussianBlur(int(kernelsize), sigma=(1.0, 10.0))
    
    pe = patch_extractor2(8, 512, center=False, device=device)
    it_num = 0
    for epoch in range(epochs):
    
        dataset_train = AnnotatedThermalDatasetDenseSejong(ds_folder=sj_folder, augmentor=augmentor2)
        dataset_train = Pix2PixDatasets(dataset_train)
        
        dataset_semi = FakeThermalizationDataset(
            ds_folder=fake_folder, width=WIDTH, height=HEIGHT,
            augmentor=augmentor
        )
        generator = torch.Generator().manual_seed(42)
        dataset_train, dataset_test =  torch.utils.data.random_split(dataset_train, [.8, .2], generator=generator)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=5)
        semi_dataloader = torch.utils.data.DataLoader(dataset_semi, batch_size=batch_size, shuffle=True, num_workers=5)
        semi_iterator = iter(semi_dataloader)
        batch_semi_org = next(semi_iterator)
        test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
        test_iterator = iter(test_dataloader)
        test_batch_viz = next(test_iterator)
        pbar = tqdm(train_dataloader)
        
        for batch in pbar:
            opt.zero_grad()
            
            batch_semi_org = next_batch_semi(semi_iterator, dataset_semi, batch_size)
            batch_semi_mask = batch_semi_org[0].clone().detach().to(device)
            batch_semi = batch_semi_org[1].to(device)/255 # Not normalized before
            batch_rgb = batch["A"].to(device)
            batch_rgb = additional_augmentation(batch_rgb, blurrer, gray, cinv, jitter, HEIGHT, WIDTH)
            batch_th = batch["B"].to(device)
            batch_rgb2th = translator(batch_rgb)
            
            l2_error = criterion(batch_th, batch_rgb2th)
            batch_semi2th = translator(batch_semi)
            pdiv = calc_pdiv(pe, div, batch_th, batch_semi2th, batch_semi_mask, regdiv, shrinker_ls, device=device)
            batch_semi_mask = fill_glasses_batch(batch_semi_mask, batch_semi, HEIGHT, WIDTH, device=device)
            sum_sqsegdev = calc_segreg(batch_semi2th, batch_semi, batch_semi_mask, mean_dict, HEIGHT, WIDTH, device=device)
            reg =  regtemp * sum_sqsegdev
            pdiv =  regdiv * pdiv/(0.5 * len(shrinker_ls) * pe.pz**2) # adjust for patch dimension
            loss = l2_error + reg +  pdiv 
            loss.backward()
            opt.step()
            if it_num%10 == 0:
                descr = f"DF: {l2_error.item():.4f}, PatchDiv: {pdiv.item():.4f}, SegReg: {reg.item():.4f}."
                pbar.set_description(descr)
            it_num +=1        
        if show_images_bool:
            show_images(translator, batch_semi, batch_semi2th, test_batch_viz, show_sejong=show_sejong_bool)
        opt = adjust_lr(opt, epoch)





if __name__ == "__main__":
    sj_folder = "XXXXX/Sejong Face Database"
    fake_folder = "XXXX/fake/"
    thermalization_main(fake_folder, sj_folder)
