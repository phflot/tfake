import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from glob import glob
import h5py
import warnings
import pickle
import mediapipe as mp
import os
from os import getenv
from os.path import isdir, join
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


WIDTH = 224
HEIGHT = 224

# segmentation class ids:
BACKGROUND = 0
SKIN = 1
NOSE = 2
RIGHT_EYE = 3
LEFT_EYE = 4
RIGHT_BROW = 5
LEFT_BROW = 6
RIGHT_EAR = 7
LEFT_EAR = 8
MOUTH_INTERIOR = 9
TOP_LIP = 10
BOTTOM_LIP = 11
NECK = 12
HAIR = 13
BEARD = 14
CLOTHING = 15
GLASSES = 16
HEADWEAR = 17
FACEWEAR = 18
IGNORE = 255


class FakeSegmentationDataset:
    """Dataset for the segmentation with the fake ground truth.

    """
    def __init__(self, ds_folder="C:\\data\\fake", width=WIDTH, height=HEIGHT, augmentor=None):
        with open(join(ds_folder, "annotations.pkl"), 'rb') as f:
            self.landmarks = pickle.load(f)
        self.ds_folder = ds_folder
        self.file_names = glob(join(ds_folder, "*_seg.png"))
        self.file_names = [f.split(os.sep)[-1].split("_")[0] for f in self.file_names]
        self.augmentor = augmentor
        self.width = width
        self.height = height

    def __getitem__(self, item):
        segmentation_masks = cv2.imread(join(self.ds_folder, f"{self.file_names[item]}_seg.png"),
                             cv2.IMREAD_UNCHANGED).astype(np.float32)

        rgb = cv2.imread(join(self.ds_folder, f"{self.file_names[item]}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        segmentation_masks = cv2.resize(segmentation_masks,
                                        (self.width, self.height),
                                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        segmentation_masks[segmentation_masks == IGNORE] = BACKGROUND
        segmentation_masks = torch.from_numpy(segmentation_masks).unsqueeze(0)

        if self.augmentor is not None:
            segmentation_masks, rgb = self.augmentor(segmentation_masks, rgb)
            if len(segmentation_masks.shape) == 2:
                segmentation_masks = segmentation_masks.unsqueeze(0)

        # convert segmentation mask to multichannel image, where each channel is a
        # binary image with based on the segmentation class id:
        segmentation_masks = segmentation_masks.squeeze().long()
        segmentation_masks = torch.nn.functional.one_hot(
            segmentation_masks, num_classes=19).permute(2, 0, 1)

        segmentation_masks = segmentation_masks.to(torch.float32)
        rgb = rgb.to(torch.float32)

        return segmentation_masks, rgb

    def _fill_glasses(self, segmentation_masks):
        """Fills the glasses that are opaque in the segmentation mask."""
        idx = segmentation_masks == GLASSES
        if idx.sum() == 0:
            return segmentation_masks
        glass_idx = np.where(idx)
        glass_points = np.column_stack(glass_idx[::-1]).astype(np.int32)

        hull = ConvexHull(glass_points)
        hull_points = glass_points[hull.vertices]
        hull_points = hull_points.reshape(-1, 1, 2).astype(np.int32)

        mask = np.zeros_like(segmentation_masks)
        cv2.fillConvexPoly(mask, hull_points, 1)
        segmentation_masks[mask == 1] = GLASSES
        return segmentation_masks

    def __len__(self):
        return len(self.file_names)


class FakeDenseThermalRGB:
    def __init__(self, ds_folder="C:\\data\\fake", width=WIDTH, height=HEIGHT, augmentor=None):
        self.ds_folder = ds_folder
        self.file_names = glob(join(ds_folder, "*thermal.png"))
        self.file_names = [f.split(os.sep)[-1].split("_")[0] for f in self.file_names]
        self.augmentor = augmentor
        self.width = width
        self.height = height

    def __getitem__(self, item):
        thermal = cv2.imread(join(self.ds_folder, f"{self.file_names[item]}_thermal.png"),
                             cv2.IMREAD_UNCHANGED).astype(np.float32)
        thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min())
        thermal = torch.from_numpy(thermal).unsqueeze(0)

        rgb = cv2.imread(join(self.ds_folder, f"{self.file_names[item]}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

        if self.augmentor is not None:
            thermal, rgb = self.augmentor(thermal, rgb)
            if len(thermal.shape) == 2:
                thermal = thermal.unsqueeze(0)
        thermal = thermal.to(torch.float32)
        rgb = rgb.to(torch.float32)

        return thermal, rgb

    def __len__(self):
        return len(self.file_names)


class FakeThermalizationDataset:
    """Dataset Class for FAKE"""
    def __init__(self, ds_folder="C:\\data\\fake", width=WIDTH, height=HEIGHT, augmentor=None):
        with open(join(ds_folder, "annotations.pkl"), 'rb') as f:
            self.landmarks = pickle.load(f)
        self.ds_folder = ds_folder
        self.file_names = glob(join(ds_folder, "*_seg.png"))
        self.file_names = [f.split(os.sep)[-1].split("_")[0] for f in self.file_names]
        self.augmentor = augmentor
        self.width = width
        self.height = height
        self.gaussian_blur = (
            transforms.GaussianBlur(kernel_size=5, sigma=0.5))

    def __getitem__(self, item):
        segmentation_masks = cv2.imread(join(self.ds_folder, f"{self.file_names[item]}_seg.png"),
                             cv2.IMREAD_UNCHANGED).astype(np.float32)

        rgb = cv2.imread(join(self.ds_folder, f"{self.file_names[item]}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        segmentation_masks = cv2.resize(segmentation_masks,
                                        (self.width, self.height),
                                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        segmentation_masks[segmentation_masks == IGNORE] = BACKGROUND
        segmentation_masks = torch.from_numpy(segmentation_masks).unsqueeze(0)

        if self.augmentor is not None:
            segmentation_masks, rgb = self.augmentor(segmentation_masks, rgb)
            if len(segmentation_masks.shape) == 2:
                segmentation_masks = segmentation_masks.unsqueeze(0)

        rgb = rgb.to(torch.float32)
        # remove id BACKGROUND and IGNORE from rgb:
        rgb[segmentation_masks.repeat((3, 1, 1)) == BACKGROUND] = 0
        rgb = self.gaussian_blur(rgb)

        # convert segmentation mask to multichannel image, where each channel is a
        # binary image with based on the segmentation class id:
        #segmentation_masks = segmentation_masks.squeeze().long()
        #segmentation_masks = torch.nn.functional.one_hot(
        #    segmentation_masks, num_classes=19).permute(2, 0, 1)

        segmentation_masks = segmentation_masks.to(torch.float32)

        return segmentation_masks, rgb

    def __len__(self):
        return len(self.file_names)



class AnnotatedThermalDatasetDenseSejong(Dataset):
    """Sejong Dataset for the thermalizer"""
    def __init__(self, ds_folder="E:\\data\\landmark_project\\Sejong Face Database\\",
                 augmentor=None):

        # original shape of the used thermal camera:
        self.shape_thermal = (384, 288)

        self.ds_folder = ds_folder
        self.augmentor = augmentor
        self.file_names = []

        for dirpath, dirnames, filenames in os.walk(ds_folder, topdown=False):
            for folder in dirnames:
                images = glob(join(dirpath, folder, "*_t.jpg"))
                for image in images:
                    rgb_image = image.replace("_t.jpg", "_v.jpg")
                    self.file_names.append((rgb_image, image))
        self.file_names = self.file_names

    def __getitem__(self, item):
        rgb_path, thermal_path = self.file_names[item]
        thermal = cv2.resize(cv2.imread(thermal_path)[..., 0], self.shape_thermal)
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(rgb_path), self.shape_thermal), cv2.COLOR_BGR2RGB).astype(float)

        thermal = torch.from_numpy(thermal).unsqueeze(0).float()
        thermal -= thermal.min()
        mx = thermal.max()
        thermal /= mx if mx > 0 else 1
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        if self.augmentor is not None:
            thermal, rgb = self.augmentor(thermal, rgb)
            if len(thermal.shape) == 2:
                thermal = thermal.unsqueeze(0)
        thermal = thermal.float()
        rgb = rgb.float()
        return thermal, rgb

    def __len__(self):
        return len(self.file_names)

    def __mul__(self, other):
        self.file_names *= other
        return self


class AnnotatedThermalDatasetDenseSejongPreprocessed(Dataset):
    """Sejong Dataset for the thermalizer"""
    def __init__(self, ds_folder="E:\\data\\landmark_project\\sejong_preprocessed\\",
                 augmentor=None,
                 segmented=False):

        self.segmented = segmented
        self.ds_folder = ds_folder
        self.augmentor = augmentor
        thermal_files = glob(join(ds_folder, "*_thermal.png"))
        rgb_files = glob(join(ds_folder, "*_rgb.png"))
        self.file_names = [(rgb, thermal) for rgb, thermal in zip(rgb_files, thermal_files)]

    def __getitem__(self, item):
        rgb_path, thermal_path = self.file_names[item]
        if self.segmented:
            base_name = os.path.basename(rgb_path)
            file_name, ext = os.path.splitext(base_name)
            rgb_path = join(os.path.dirname(rgb_path),
                            "segmented", file_name + "_segmented" + ext)
        thermal = cv2.resize(cv2.imread(thermal_path)[..., 0], self.shape_thermal)
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB).astype(float)

        if self.segmented:
            idx = (rgb == 0).all(axis=2)
            thermal = thermal.astype(float)
            thermal[idx] = 0

        thermal = torch.from_numpy(thermal).unsqueeze(0).float()
        thermal -= thermal.min()
        mx = thermal.max()
        thermal /= mx if mx > 0 else 1

        thermal = (thermal - 0.1) / 0.9
        thermal = torch.clamp(thermal, 0, 1)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        if self.augmentor is not None:
            thermal, rgb = self.augmentor(thermal, rgb)
            if len(thermal.shape) == 2:
                thermal = thermal.unsqueeze(0)
        thermal = thermal.float()
        rgb = rgb.float()
        return thermal, rgb

    def __len__(self):
        return len(self.file_names)

    def __mul__(self, other):
        self.file_names *= other
        return self

def _transform(x):
    return cv2.normalize(np.array(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)).astype(np.float32),
                         None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)










