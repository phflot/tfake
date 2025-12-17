#!/usr/bin/env python

import os
from os.path import join
import torch
from torch.utils.data import Dataset

from core.augmentations import GeometricLandmarkAugmentor, ImageAugmentor, ThermalAugmentor
from core.training import JointLandmarkTrainer
from core.datasets import FakeLandmarks70ThermalRGB, TextureDataset, ThermalTextureDataset

from torchvision.transforms import RandomGrayscale, Normalize


import numpy as np


class TextureMixed(Dataset):
    def __init__(self, ds_normal, ds_thermal, thermal_augmentor=None, image_augmentor=None,
                 gray_augmentor=None, p_thermal=0.4):
        self.ds_normal = ds_normal
        self.ds_thermal = ds_thermal
        self.thermal_augmentor = thermal_augmentor
        self.image_augmentor = image_augmentor
        self.gray_augmentor = gray_augmentor
        self.p_thermal = p_thermal

    def __getitem__(self, item):
        thermal, pose = self.ds_thermal[item]
        rgb, _ = self.ds_normal[item]

        thermal = self.thermal_augmentor(thermal / 255) * 255
        thermal = thermal.repeat((3, 1, 1))

        rgb = self.image_augmentor(rgb.to(torch.uint8)).float()
        rgb = self.gray_augmentor(rgb)

        output = thermal if np.random.rand() < self.p_thermal else rgb

        return output, pose

    def __len__(self):
        return len(self.ds_normal)


class MobileNetDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.transform = Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                                   std=torch.tensor([0.2290, 0.2240, 0.2250]))

    def __getitem__(self, item):
        img, pose = self.datasets[item]
        return self.transform(img / 255), pose

    def __len__(self):
        return len(self.datasets)


if __name__ == "__main__":
    input_dir = "/local/landmark_project/datasets"

    thermal_ds = [(join(input_dir, "fake_thermal_test2_weights_warm_e9.pth"), 0.5),
                  (join(input_dir, "fake_thermal_test2_weights_cold_e9.pth"), 0.5)]

    output_dir = join("/local/landmark_project/training", "240728_70_22_refine_w+c_joint0.4_2024")

    print(f"outputdir: {output_dir}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    gpus = list(range(torch.cuda.device_count()))
    # gpus = list(range(6))
    print(f"cuda gpu count: {torch.cuda.device_count()}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"Using gpus {gpus}")

    augmentor = GeometricLandmarkAugmentor(shear=7)
    thermal_augmentor = ThermalAugmentor()
    image_augmentor = ImageAugmentor()
    gray_augmentor = RandomGrayscale(p=0.1)

    tmp = ThermalTextureDataset(ds_folder=join(input_dir, "texture"), thermal_channels=1,
                                augmentor=augmentor, lm_shape=(70, 2))

    dataset = FakeLandmarks70ThermalRGB(ds_folder=join(input_dir, "fake"),
                                         geometric_augmentor=augmentor, thermal_channels=3,
                                         thermal_augmentor=thermal_augmentor,
                                         image_augmentor=image_augmentor,
                                         gray_augmentor=gray_augmentor,
                                         texture_ds=tmp,
                                         thermal_folder=thermal_ds, p_thermal=0.4)

    dataset += TextureMixed(TextureDataset(ds_folder=join(input_dir, "texture"), augmentor=augmentor,
                                           lm_shape=(70, 2)), tmp,
                            thermal_augmentor=thermal_augmentor, image_augmentor=image_augmentor,
                            gray_augmentor=gray_augmentor, p_thermal=0.4)

    dataset = MobileNetDataset(dataset)

    print(f"Size of the dataset: {len(dataset)}")

    n_epochs = 4000

    # checkpoint = "joint_70_rgb_gray.pt"

    trainer = JointLandmarkTrainer(dataset=dataset, gpus=gpus, batch_size=512, display_step=100,
                                   learning_rate=0.001, progress_folder=join(output_dir, "progress_landmarker"),
                                   model_path=join(output_dir, "models_landmarker"),
                                   n_landmarks=70, num_workers=80, checkpoint=None)
    trainer.train(n_epochs)
