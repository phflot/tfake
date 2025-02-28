import torch
import matplotlib.pyplot as plt
import torchvision
import scipy
import numpy as np
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

def show_images(translator, batch_semi, batch_semi2th, test_batch_viz, show_sejong=True):
    translator.eval()
    batch_semi_viz = batch_semi[:8, 0]
    batch_semi2th_viz = batch_semi2th[:8, 0]
    grid = make_grid(batch_semi_viz, nrow=1).detach().cpu()
    grid2 = make_grid(batch_semi2th_viz, nrow=1).detach().cpu()
    f, ax = plt.subplots(2, 8, figsize=(2*10, 2*3))
    for num in range(8):
        ax[0, num].imshow(grid[num], cmap="gray")
        ax[0, num].axis("off")
        ax[1, num].imshow(grid2[num], cmap="gray")
        ax[1, num].axis("off")
    plt.show()
    plt.close()
    batch_rgb_viz = test_batch_viz["A"][:8].to("cuda:1")
    batch_rgb2th_viz = translator(batch_rgb_viz)
    grid = make_grid(batch_rgb_viz[:, 0], nrow=1).detach().cpu()#.permute(1, 2, 0)
    grid2 = make_grid(batch_rgb2th_viz.squeeze(), nrow=1).detach().cpu()
    f, ax = plt.subplots(2, 8, figsize=(2*10, 2*3))
    for num in range(8):
        ax[0, num].imshow(grid[num], cmap="gray")
        ax[0, num].axis("off")
        ax[1, num].imshow(grid2[num], cmap="gray")
        ax[1, num].axis("off")
    if show_sejong:
        plt.show()
    plt.close()
    translator.train()

def next_batch_semi(semi_iterator, dataset_semi, batch_size):
    try:
        batch_semi_org = next(semi_iterator)
    except:
        semi_dataloader = torch.utils.data.DataLoader(dataset_semi, batch_size=batch_size, shuffle=True)
        semi_iterator = iter(semi_dataloader)
        batch_semi_org = next(semi_iterator)
    return batch_semi_org

def adjust_lr(opt, epoch):
    if epoch == 3:
        for g in opt.param_groups:
            g['lr'] = 1e-4  
    return opt

def fill_glasses(mask):
    if (mask[0] == 16).any():
        return scipy.ndimage.binary_fill_holes((mask[0] == 16).detach().cpu().numpy().astype(int)).astype(bool) 
    else:
        return torch.zeros_like(mask.detach().cpu()).numpy()[0].astype(bool) 

def fill_glasses_batch(batch_semi_mask, batch_semi, HEIGHT, WIDTH, device="cuda:1"):
    glas_mask_ls = [fill_glasses(mask) for mask in batch_semi_mask] 
    glas_mask_ls = np.array(glas_mask_ls)
    glas_mask = torch.tensor(glas_mask_ls).to(device)
    current_batch_size = batch_semi.shape[0]
    glas_mask = glas_mask.reshape(current_batch_size, 1, HEIGHT, WIDTH)
    batch_semi_mask[glas_mask] = 16

    return batch_semi_mask