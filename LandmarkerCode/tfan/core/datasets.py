import os
from glob import glob
from os.path import join
import warnings
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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


class AnnotatedThermalDatasetDenseSejong(Dataset):
    """Sejong Dataset for rgb2thermal translation."""
    def __init__(self, ds_folder="Sejong Face Database\\",
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


class TextureDataset(Dataset):
    """Dataset for texture images with random landmarks."""
    def __init__(self, ds_folder="ElBa\\texture", lm_shape=(70, 2), augmentor=None):
        self.file_names = glob(join(ds_folder, '*.jpeg'))
        self.lm_shape = lm_shape
        self.augmentor = augmentor

    def __getitem__(self, item):
        path = self.file_names[item]
        img = np.array((cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        pose = torch.rand(self.lm_shape).to(torch.float64)

        if self.augmentor is not None:
            img, pose = self.augmentor(img, pose)

        return img, pose

    def __len__(self):
        return len(self.file_names)


def _transform(x):
    """Transforms the image to a normalized grayscale image.
    Args:
        x: The image to transform."""
    return cv2.normalize(np.array(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)).astype(np.float32),
                         None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


class ThermalTextureDataset(TextureDataset):
    """Dataset for texture images with random landmarks that returns normalized grayscale images."""
    def __init__(self, ds_folder="ElBa\\texture", lm_shape=(70, 2),
                 thermal_channels=1, augmentor=None):
        super().__init__(ds_folder, lm_shape, augmentor)

        self.transforms = _transform
        self.thermal_channels = thermal_channels

    def __getitem__(self, item):
        path = self.file_names[item]
        img = self.transforms(cv2.imread(path))
        img = torch.from_numpy(img).unsqueeze(0)
        pose = torch.rand(self.lm_shape).to(torch.float64)

        if self.augmentor is not None:
            img, pose = self.augmentor(img, pose)
        if self.thermal_channels > 1:
            img = img.repeat((self.thermal_channels, 1, 1))

        img *= 255
        return img.float(), pose


class FakeLandmarks478(Dataset):
    """FAKE base class that reads the 478 annotations and returns the RGB image + pose."""
    def __init__(self, ds_folder="fake", augmentor=None):
        with open(join(ds_folder, "annotations.pkl"), 'rb') as f:
            self.landmarks = pickle.load(f)

        self.ds_folder = ds_folder
        self.augmentor = augmentor
        self.file_names = list(self.landmarks.keys())

    def __getitem__(self, item):
        pose = self.landmarks[self.file_names[item]]
        path = join(self.ds_folder, self.file_names[item])
        pose = torch.from_numpy(pose)
        img = np.array((cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, pose

    def __len__(self):
        return len(self.file_names)


class FakeLandmarks478ThermalRGB(FakeLandmarks478):
    """Dataset for RGB and thermal images with 478-point landmarks."""
    def __init__(self, p_thermal=0.4, thermal_channels=1, n_samples=None,
                 geometric_augmentor=None, thermal_augmentor=None,
                 image_augmentor=None, gray_augmentor=None, texture_ds=None, background_texture_prob=0.25,
                 thermal_folder=None, **kwargs):
        super().__init__(**kwargs)
        if n_samples is not None:
            self.file_names = self.file_names[:n_samples]
        self.thermal_file_names = glob(join(self.ds_folder, "*thermal.png"))
        thermal_ids = np.array([int(os.path.basename(f).split("_")[0]) for f in self.thermal_file_names])
        all_ids = np.array([int(f.split(".")[0]) for f in self.file_names])
        _, idx, idx_thermal = np.intersect1d(all_ids, thermal_ids, assume_unique=False, return_indices=True)
        self.file_names = np.array(self.file_names)[idx]
        self.thermal_file_names = np.array(self.thermal_file_names)[idx_thermal]
        self.p_thermal = p_thermal
        self.thermal_channels = thermal_channels
        self.geometric_augmentor = geometric_augmentor
        self.thermal_augmentor = thermal_augmentor
        self.image_augmentor = image_augmentor
        self.gray_augmentor = gray_augmentor
        self.texture_ds = texture_ds
        self.background_texture_prob = background_texture_prob
        self.thermal_folder = thermal_folder

    def __getitem__(self, item):
        if np.random.rand() > self.p_thermal:
            img, pose = super().__getitem__(item)
            if self.geometric_augmentor is not None:
                img, pose = self.geometric_augmentor(img, pose[:, :2])
            if self.image_augmentor is not None:
                img = self.image_augmentor(img.to(torch.uint8)).float()
            if self.gray_augmentor is not None:
                img = self.gray_augmentor(img)
            return img, pose
        else:
            pose = self.landmarks[self.file_names[item]]
            pose = torch.from_numpy(pose)

            path = join(self.ds_folder, self.file_names[item])
            base_folder = os.path.dirname(path)
            file_name = os.path.splitext(os.path.basename(path))[0]
            segmentation_masks = cv2.imread(
                join(base_folder, os.path.basename(file_name) + "_seg.png"),
                cv2.IMREAD_UNCHANGED
            ).astype(np.float32)
            segmentation_masks[segmentation_masks == IGNORE] = BACKGROUND

            if self.thermal_folder is not None:
                if isinstance(self.thermal_folder, list):
                    prob = [x[1] for x in self.thermal_folder]
                    random_idx = np.random.choice(len(self.thermal_folder), p=prob)
                    thermal_folder = self.thermal_folder[random_idx][0]
                else:
                    thermal_folder = self.thermal_folder
                thermal = cv2.imread(join(thermal_folder, os.path.basename(self.thermal_file_names[item])),
                                     cv2.IMREAD_UNCHANGED)
            else:
                thermal = cv2.imread(self.thermal_file_names[item], cv2.IMREAD_UNCHANGED)
                warnings.warn("The thermal folder is not set, using the thermal file names from the dataset folder.")
            thermal = thermal.astype(float) / (2 ** 16 - 1)
            thermal = torch.from_numpy(thermal).unsqueeze(0)

            thermal_min = thermal.min()
            thermal -= thermal_min
            thermal_max = thermal.max()
            thermal /= thermal_max

            segmentation_masks = torch.from_numpy(segmentation_masks).unsqueeze(0)

            if np.random.rand() < self.background_texture_prob and self.texture_ds is not None:
                rand_idx = np.random.randint(0, len(self.texture_ds))
                background, _ = self.texture_ds[rand_idx]

                background = background.to(thermal.dtype) / 255
                background = transforms.functional.resize(background, thermal.shape[1:])
                thermal[segmentation_masks == BACKGROUND] = background[segmentation_masks == BACKGROUND]
            else:
                if np.random.rand() > 0.8:
                    thermal[segmentation_masks == BACKGROUND] = 0

            if self.geometric_augmentor is not None:
                thermal, pose = self.geometric_augmentor(thermal, pose[:, :2])
            if self.thermal_augmentor is not None:
                thermal = self.thermal_augmentor(thermal)

            if self.thermal_channels > 1:
                thermal = thermal.repeat((self.thermal_channels, 1, 1))
            thermal *= 255
            return thermal.float(), pose


class FakeLandmarks70(Dataset):
    """FAKE base class that reads the 70 annotations and returns the RGB image + pose."""
    def __init__(self, ds_folder="fake", augmentor=None):
        annotations = glob(join(ds_folder, '*.txt'))
        self.augmentor = augmentor

        self.ds_folder = ds_folder

        self.files = []

        for annotation in annotations:
            with open(annotation) as f:
                tmp = f.readlines()
                landmarks = np.array([[float(x) for x in line.split()] for line in tmp])
                file_name = os.path.basename(annotation).split("_")[0] + ".png"
                self.files.append((join(os.path.dirname(annotation), file_name), landmarks / 512))

    def __getitem__(self, item):
        path, pose = self.files[item]
        pose = torch.from_numpy(pose)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        if self.augmentor is not None:
            img, pose = self.augmentor(img, pose)
        return img, pose

    def __len__(self):
        return len(self.files)


class FakeLandmarks70ThermalRGB(FakeLandmarks70):
    """Dataset for RGB and thermal images with 70-point landmarks."""
    def __init__(self, p_thermal=0.4, thermal_channels=1, n_samples=None,
                 geometric_augmentor=None, thermal_augmentor=None,
                 image_augmentor=None, gray_augmentor=None, texture_ds=None, background_texture_prob=0.25,
                 thermal_folder=None, **kwargs):
        super().__init__(**kwargs)
        self.thermal_file_names = glob(join(self.ds_folder, "*thermal.png"))
        thermal_ids = np.array([int(os.path.basename(f).split("_")[0]) for f in self.thermal_file_names])
        all_ids = np.array([int(os.path.basename(f).split(".")[0]) for f, _ in self.files])
        _, idx, idx_thermal = np.intersect1d(all_ids, thermal_ids, assume_unique=False, return_indices=True)

        self.files = [self.files[i] for i in idx]
        self.thermal_file_names = np.array(self.thermal_file_names)[idx_thermal]
        self.p_thermal = p_thermal
        self.thermal_channels = thermal_channels
        self.geometric_augmentor = geometric_augmentor
        self.thermal_augmentor = thermal_augmentor
        self.image_augmentor = image_augmentor
        self.gray_augmentor = gray_augmentor
        self.texture_ds = texture_ds
        self.background_texture_prob = background_texture_prob
        self.thermal_folder = thermal_folder

    def __getitem__(self, item):
        if np.random.rand() > self.p_thermal:
            img, pose = super().__getitem__(item)
            if self.geometric_augmentor is not None:
                img, pose = self.geometric_augmentor(img, pose[:, :2])
            if self.image_augmentor is not None:
                img = self.image_augmentor(img.to(torch.uint8)).float()
            if self.gray_augmentor is not None:
                img = self.gray_augmentor(img)
            return img, pose
        else:
            path, pose = self.files[item]
            pose = torch.from_numpy(pose)

            base_folder = os.path.dirname(path)
            file_name = os.path.splitext(os.path.basename(path))[0]
            segmentation_masks = cv2.imread(
                join(base_folder, os.path.basename(file_name) + "_seg.png"),
                cv2.IMREAD_UNCHANGED
            ).astype(np.float32)
            segmentation_masks[segmentation_masks == IGNORE] = BACKGROUND

            if self.thermal_folder is not None:
                if isinstance(self.thermal_folder, list):
                    prob = [x[1] for x in self.thermal_folder]
                    random_idx = np.random.choice(len(self.thermal_folder), p=prob)
                    thermal_folder = self.thermal_folder[random_idx][0]
                else:
                    thermal_folder = self.thermal_folder
                thermal = cv2.imread(join(thermal_folder, os.path.basename(self.thermal_file_names[item])),
                                     cv2.IMREAD_UNCHANGED)
            else:
                thermal = cv2.imread(self.thermal_file_names[item], cv2.IMREAD_UNCHANGED)
                warnings.warn("The thermal folder is not set, using the thermal file names from the dataset folder.")
            thermal = thermal.astype(float) / (2 ** 16 - 1)
            thermal = torch.from_numpy(thermal).unsqueeze(0)

            thermal_min = thermal.min()
            thermal -= thermal_min
            thermal_max = thermal.max()
            thermal /= thermal_max

            segmentation_masks = torch.from_numpy(segmentation_masks).unsqueeze(0)

            if np.random.rand() < self.background_texture_prob and self.texture_ds is not None:
                rand_idx = np.random.randint(0, len(self.texture_ds))
                background, _ = self.texture_ds[rand_idx]

                background = background.to(thermal.dtype) / 255
                background = transforms.functional.resize(background, thermal.shape[1:])
                thermal[segmentation_masks == BACKGROUND] = background[segmentation_masks == BACKGROUND]
            else:
                if np.random.rand() > 0.8:
                    thermal[segmentation_masks == BACKGROUND] = 0

            if self.geometric_augmentor is not None:
                thermal, pose = self.geometric_augmentor(thermal, pose[:, :2])
            if self.thermal_augmentor is not None:
                thermal = self.thermal_augmentor(thermal)

            if self.thermal_channels > 1:
                thermal = thermal.repeat((self.thermal_channels, 1, 1))
            thermal *= 255
            return thermal.float(), pose


class FakeLandmarks70ThermalRGBJoint(FakeLandmarks70):
    """Dataset for RGB and thermal images with 70-point landmarks.
    Modification for the ablation study: This dataset returns all ablation configurations,
    i.e. RGB only, RGB + gray augmentation, thermal."""
    def __init__(self, p_thermal=0.2, thermal_channels=1, geometric_augmentor=None, thermal_augmentor=None,
                 image_augmentor=None, gray_augmentor=None, texture_ds=None, background_texture_prob=0.25,
                 thermal_folder=None, buffer_ds=False, **kwargs):
        super().__init__(**kwargs)
        self.thermal_file_names = glob(join(self.ds_folder, "*thermal.png"))
        thermal_ids = np.array([int(os.path.basename(f).split("_")[0]) for f in self.thermal_file_names])
        all_ids = np.array([int(os.path.basename(f).split(".")[0]) for f, _ in self.files])
        _, idx, idx_thermal = np.intersect1d(all_ids, thermal_ids, assume_unique=False, return_indices=True)
        self.files = [self.files[i] for i in idx]
        self.thermal_file_names = np.array(self.thermal_file_names)[idx_thermal]
        self.p_thermal = p_thermal
        self.thermal_channels = thermal_channels
        self.geometric_augmentor = geometric_augmentor
        self.thermal_augmentor = thermal_augmentor
        self.image_augmentor = image_augmentor
        self.gray_augmentor = gray_augmentor
        self.texture_ds = texture_ds
        self.background_texture_prob = background_texture_prob
        self.thermal_folder = thermal_folder
        self.buffer_ds = buffer_ds
        self.img_buffer = {}
        self.segmentation_buffer = {}
        self.thermal_buffer = {}

    def __getitem__(self, item):
        path, pose = self.files[item]
        pose = torch.from_numpy(pose)

        if (not self.buffer_ds) or (path not in self.img_buffer):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
            img = torch.from_numpy(img).permute(2, 0, 1)
            if self.buffer_ds:
                self.img_buffer[path] = img
        if self.buffer_ds:
            img = self.img_buffer[path]

        # get base folder and filename:
        base_folder = os.path.dirname(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        segmentation_path = join(base_folder, os.path.basename(file_name) + "_seg.png")

        if (not self.buffer_ds) or (segmentation_path not in self.segmentation_buffer):
            segmentation_masks = cv2.imread(
                segmentation_path,
                cv2.IMREAD_UNCHANGED
            ).astype(np.float32)
            segmentation_masks[segmentation_masks == IGNORE] = BACKGROUND
            segmentation_masks = torch.from_numpy(segmentation_masks).unsqueeze(0)
            if self.buffer_ds:
                self.segmentation_buffer[segmentation_path] = segmentation_masks

        if self.buffer_ds:
            segmentation_masks = self.segmentation_buffer[segmentation_path]

        if self.thermal_folder is not None:
            thermal_filename = join(self.thermal_folder, os.path.basename(self.thermal_file_names[item]))
        else:
            thermal_filename = self.thermal_file_names[item]
            warnings.warn("The thermal folder is not set, using the thermal file names from the dataset folder.")

        if (not self.buffer_ds) or (thermal_filename not in self.thermal_buffer):
            thermal = cv2.imread(thermal_filename, cv2.IMREAD_UNCHANGED)
            thermal = thermal.astype(float) / (2 ** 16 - 1)
            thermal = torch.from_numpy(thermal).unsqueeze(0)

            thermal_min = thermal.min()
            thermal -= thermal_min
            thermal_max = thermal.max()
            thermal /= thermal_max

            if self.buffer_ds:
                self.thermal_buffer[thermal_filename] = thermal

        if self.buffer_ds:
            thermal = self.thermal_buffer[thermal_filename]


        if np.random.rand() < self.background_texture_prob and self.texture_ds is not None:
            rand_idx = np.random.randint(0, len(self.texture_ds))
            background, _ = self.texture_ds[rand_idx]

            background = background.to(thermal.dtype) / 255
            background = transforms.functional.resize(background, thermal.shape[1:])
            thermal[segmentation_masks == BACKGROUND] = background[segmentation_masks == BACKGROUND]
        else:
            if np.random.rand() > 0.8:
                thermal[segmentation_masks == BACKGROUND] = 0

        tmp = torch.concat([img, thermal], 0)

        if self.geometric_augmentor is not None:
            tmp, pose = self.geometric_augmentor(tmp, pose[:, :2])

        thermal = tmp[-1].unsqueeze(0)
        img = tmp[:-1]

        if self.thermal_augmentor is not None:
            thermal = self.thermal_augmentor(thermal)
        if self.image_augmentor is not None:
            img = self.image_augmentor(img.to(torch.uint8)).float()

        if self.gray_augmentor is not None:
            gray = self.gray_augmentor(img)
        else:
            gray = img

        if self.thermal_channels > 1:
            thermal = thermal.repeat((self.thermal_channels, 1, 1))
        thermal *= 255

        return img, gray, thermal.float(), pose


class FakeThermalizationDataset:
    """Fake dataset return segmentation masks and rgb images."""
    def __init__(self, ds_folder="fake", width=WIDTH, height=HEIGHT, augmentor=None):
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
        # segmentation_masks = self._fill_glasses(segmentation_masks)
        segmentation_masks = torch.from_numpy(segmentation_masks).unsqueeze(0)

        if self.augmentor is not None:
            segmentation_masks, rgb = self.augmentor(segmentation_masks, rgb)
            if len(segmentation_masks.shape) == 2:
                segmentation_masks = segmentation_masks.unsqueeze(0)

        rgb = rgb.to(torch.float32)
        # remove id BACKGROUND and IGNORE from rgb:
        rgb[segmentation_masks.repeat((3, 1, 1)) == BACKGROUND] = 0
        rgb = self.gaussian_blur(rgb)

        segmentation_masks = segmentation_masks.to(torch.float32)

        return segmentation_masks, rgb

    def __len__(self):
        return len(self.file_names)


class Pix2PixDatasets(Dataset):
    """Pix2Pix / Cyclegan dataset wrapper. """
    def __init__(self, dataset, normalize=True):
        self.dataset = dataset
        self.normalize = normalize
        self.rgb_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, item):
        thermal, rgb = self.dataset[item]

        if self.normalize:
            rgb /= 255
            thermal -= thermal.min()
            thermal /= thermal.max()

        ret = {
            "A": rgb,
            "B": thermal,
            "A_paths": f"{item}.png",
            "B_paths": f"{item}.png"
        }
        return ret

    def __len__(self):
        return len(self.dataset)
