""" Code in the inherited augmentations is modified from torchvision 0.13.1+cu113 """

# from datasets import HEIGHT, WIDTH
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, \
    RandomAdjustSharpness, RandomAutocontrast, RandomEqualize, GaussianBlur, RandomGrayscale
from torchvision.transforms.functional import InterpolationMode
import math
import numbers
from typing import List
import torchvision
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import cv2

class RandomHorizontalLandmarkFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, img, lm):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), lm * torch.Tensor([-1, 1]).unsqueeze(0) + torch.Tensor([1, 0]).unsqueeze(0)
        return img, lm


class RandomNoise(torch.nn.Module):
    def __init__(self, sigma=(0.05, 7), p=0.7):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            return torch.clamp(img + torch.randn(img.shape).to(img.device) * sigma, 0, 255)
        return img


class RandomResizedCropLandmark(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.4, 2), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.NEAREST):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)

    def forward(self, img, lm):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        _, height, width = F.get_dimensions(img)

        lm_aug = (lm - torch.tensor([j / width, i / height]).unsqueeze(0)) * \
                 torch.tensor([width / w, height / h]).unsqueeze(0)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), lm_aug


def _get_inverse_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float],
        inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation,
    # protected method taken from torchvision.functional

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


class RandomShearLandmark(T.RandomAffine):
    def __init__(self, shear=None):
        super().__init__(0, shear=shear)

    def forward(self, img, lm):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear,
                                                            [width, height])
        translate_f = [1.0 * t for t in translations]

        if isinstance(angle, int):
            angle = float(angle)

        if isinstance(translations, tuple):
            translations = list(translations)

        if isinstance(shear, numbers.Number):
            shear = [shear, 0.0]

        if isinstance(shear, tuple):
            shear = list(shear)

        if len(shear) == 1:
            shear = [shear[0], shear[0]]

        if len(shear) != 2:
            raise ValueError("Shear should be a sequence containing two values. Got {}".format(shear))

        dims = torch.tensor([width, height]).unsqueeze(0).to(lm.device)
        lm = lm * dims
        matrix = _get_inverse_affine_matrix([0.5 * height, 0.5 * width], angle, translate_f, scale, [-i for i in shear][::-1])

        m = torch.tensor(matrix).reshape(2, 3).to(torch.float64)
        lm_tmp = torch.flip(lm, (1,))
        lm_aug = (m[:, :2] @ lm_tmp.T + m[:, 2].unsqueeze(1)).T
        lm_aug = torch.flip(lm_aug, (1,))
        lm_aug /= dims

        img_aug = F.affine(img, angle, translations, scale, shear, interpolation=self.interpolation, fill=fill)

        return img_aug, lm_aug


class RandomRotLandmark(T.RandomAffine):
    def __init__(self, degrees):
        super().__init__(degrees)

    def forward(self, img, lm):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear,
                                                            [width, height])

        translate_f = [1.0 * t for t in translations]

        if isinstance(angle, int):
            angle = float(angle)

        if isinstance(translations, tuple):
            translations = list(translations)

        if isinstance(shear, numbers.Number):
            shear = [shear, 0.0]

        if isinstance(shear, tuple):
            shear = list(shear)

        if len(shear) == 1:
            shear = [shear[0], shear[0]]

        if len(shear) != 2:
            raise ValueError("Shear should be a sequence containing two values. Got {}".format(shear))

        dims = torch.tensor([width, height]).unsqueeze(0).to(lm.device)
        lm = lm * dims
        matrix = _get_inverse_affine_matrix([0.5 * height, 0.5 * width], angle, translate_f, scale, [-i for i in shear])
        m = torch.tensor(matrix).reshape(2, 3).to(torch.float64)
        lm_aug = torch.flip((m[:, :2] @ torch.flip(lm, (1,)).T + m[:, 2].unsqueeze(1)).T, (1,))
        lm_aug /= dims
        # lm_aug = (m[:, :2] @ lm.T + m[:, 2].unsqueeze(1)).T

        img_aug = F.affine(img, angle, translations, scale, shear, interpolation=self.interpolation, fill=fill)

        return img_aug, lm_aug


class LandmarkAugmentor:
    def __init__(self):
        self.resized_crop = RandomResizedCropLandmark(size=(224, 224))
        self.rot = RandomRotLandmark(degrees=45)
        self.shear = RandomShearLandmark(shear=10)
        self.flip = RandomHorizontalLandmarkFlip()

        self.sharp = RandomAdjustSharpness(2, p=0)
        self.contr = RandomAutocontrast()
        self.equ = RandomEqualize(p=0.1)
        self.jit = ColorJitter(0.1, 0.1, 0.1, (-0.05, 0.05))
        self.blur = GaussianBlur(9, (0.01, 2))
        self.noise = RandomNoise()
        self.gray = RandomGrayscale(p=0.1)

    def __call__(self, img, lm):
        img_aug, lm_aug = self.rot(img, lm)
        # TODO: fix left and right landmarks!
        # img_aug, lm_aug = self.flip(img_aug, lm_aug)
        img_aug, lm_aug = self.shear(img_aug, lm_aug)
        img_aug, lm_aug = self.resized_crop(img_aug, lm_aug)
        img_aug = self.sharp(img_aug.to(torch.uint8))
        img_aug = self.contr(img_aug)
        img_aug = self.jit(img_aug)
        if np.random.rand() < 0.3:
            img_aug = self.blur(img_aug)
        img_aug = self.equ(img_aug).to(torch.float32)
        img_aug = self.noise(img_aug)
        img_aug = self.gray(img_aug)

        return img_aug, lm_aug


class ThermalRandomJitter(torch.nn.Module):
    def __init__(self, p=0.8, jitter_magnitude=2):
        super().__init__()
        self.p = p
        self.jitter_magnitude = jitter_magnitude / 20

    def forward(self, x):
        x += (np.random.rand() - 0.5) * 2 * self.jitter_magnitude if np.random.rand() < self.p else 0
        return torch.clip(x, 0, 1)


class ThermalRandomInversion(torch.nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if np.random.rand() < self.p:
            x[x > 0] = 1 - x[x > 0]
        return x


class ThermalLandmarkAugmentor:
    def __init__(self, sigma=(0.01, 3), degrees=45, shear=10, scale=(0.4, 2), ratio=(3. / 4., 4. / 3.)):
        self.dual_aug = torch.nn.Sequential(
            RandomRotLandmark(degrees=degrees),
            RandomShearLandmark(shear=shear)
        )
        self.crop = RandomResizedCropLandmark(size=(224, 224), scale=scale, ratio=ratio)
        self.thermal_aug = torch.nn.Sequential(
            ThermalRandomJitter(jitter_magnitude=2),
            GaussianBlur(9, sigma),
            RandomNoise(sigma=(0.00001, 0.0001), p=0.2)
        )

    def __call__(self, thermal, lm):
        for f in self.dual_aug:
            thermal, lm = f(thermal, lm)
        thermal, lm = self.crop(thermal, lm)
        return self.thermal_aug(thermal), lm


class ImageAugmentor:
    def __init__(self):
        self.image_aug1 = torch.nn.Sequential(
            RandomAdjustSharpness(2, p=0),
            RandomAutocontrast(),
            ColorJitter(0.1, 0.1, 0.1, (-0.05, 0.05))
        )
        self.blur = GaussianBlur(9, (0.01, 2))
        self.image_aug2 = torch.nn.Sequential(
            RandomEqualize(p=0.1),
            RandomNoise()
        )

    def __call__(self, img):
        img = self.image_aug1(img)
        if np.random.rand() < 0.3:
            img = self.blur(img)
        img = self.image_aug2(img)

        return img


class ThermalAugmentor:
    def __init__(self, jitter_magnitude=0.2, sigma=(0.01, 5)):
        self.thermal_aug = torch.nn.Sequential(
            ThermalRandomJitter(jitter_magnitude=jitter_magnitude),
            GaussianBlur(21, sigma),
            ThermalRandomInversion(p=0.1),
            RandomNoise(sigma=(0.00001, 0.0001), p=0.2)
        )

    def __call__(self, thermal):
        return self.thermal_aug(thermal)


class GeometricLandmarkAugmentor:
    def __init__(self, degrees=45, shear=10,
                 scale=(0.4, 2), ratio=(3. / 4., 4. / 3.)):
        self.dual_aug = torch.nn.Sequential(
            RandomRotLandmark(degrees=degrees),
            RandomShearLandmark(shear=shear)
        )
        self.crop = RandomResizedCropLandmark(size=(224, 224),
                                              scale=scale,
                                              ratio=ratio)

    def __call__(self, img, lm):
        for f in self.dual_aug:
            img, lm = f(img, lm)
        img, lm = self.crop(img, lm)
        return img, lm


class PersonSegmenter(torch.nn.Module):
    """Augmentation class to matte out the person in an image.

    author: Moritz Piening
    """
    def __init__(self, device=torch.device("cpu"), **kwargs):
        super().__init__()
        # Step 1: Initialize model with the best available weights
        self.weights = FCN_ResNet50_Weights.DEFAULT
        self.device = device
        self.model = fcn_resnet50(weights=self.weights)
        self.model = self.model.to(device)
        self.model.eval()

        # Step 2: Initialize the inference transforms
        self.preprocess = self.weights.transforms()
        self.class_to_idx = {cls: idx for (idx, cls) in enumerate(self.weights.meta["categories"])}

    def segment_person(self, img):# Step 3: Apply inference preprocessing transforms
        ref_size = img.shape[-2:]
        batch = self.preprocess(img)
        if len(img.shape) == 3:
            batch = batch.unsqueeze(0)

        # Step 4: Use the model and visualize the prediction
        prediction = self.model(batch)["out"]
        normalized_masks = prediction.softmax(dim=1)

        mask_background = normalized_masks[0, self.class_to_idx["__background__"]]
        mask_person = normalized_masks[0, self.class_to_idx["person"]]
        mask_person = mask_person[4:-4, 4:-4]
        mask_background = mask_background[4:-4, 4:-4]
        resize = torchvision.transforms.Resize(ref_size)
        mask_person = resize(mask_person.unsqueeze(0)).squeeze(0)
        mask_background = resize(mask_background.unsqueeze(0)).squeeze(0)
        return mask_background, mask_person

    def __call__(self, img):
        person_mask = self.segment_person(img)[1]
        return img * person_mask.unsqueeze(0)


class TemperatureClamper:
    """Augmentation class to clamp the temperature of the image to a certain range.

    """
    def __init__(self, min_temp=0.0, max_temp=1.0):
        self.min_temp = min_temp
        self.max_temp = max_temp

    def __call__(self, img):
        return torch.clip(img, self.min_temp, self.max_temp)


class DenseAugmentor:
    """Augmentor for the thermalization task"""
    def __init__(self, output_shape=(256, 256), scale=(0.25, 1.2),
                 device=torch.device("cpu")):
        self.dual_aug = torch.nn.Sequential(
            T.RandomAffine(45, [0, 0], [0.9, 2], 15),
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=output_shape, scale=scale)
        )
        self.rgb_aug = torch.nn.Sequential(
            # PersonSegmenter(device=device),
            RandomAutocontrast(),
            RandomEqualize(p=0.05),
            ColorJitter(0.05, 0.05, 0.05, (-0.05, 0.05)),
            # GaussianBlur(9, (0.01, 2))
            GaussianBlur(9, (0.01, 0.5))
        )

    def __call__(self, thermal, rgb):
        tmp = torch.cat([thermal, rgb], 0)
        tmp = self.dual_aug(tmp)
        try:
            result = self.rgb_aug(tmp[1:].to(torch.uint8)).to(torch.float)
        except:
            result = self.rgb_aug(tmp[1:]).to(torch.float)
        return tmp[0], result


class JointLandmarkAugmentor:
    def __init__(self):
        self.thermal_augmentor = ThermalLandmarkAugmentor()
        self.rgb_augmentor = LandmarkAugmentor()

    def __call__(self, img, lm):
        if img.shape[0] == 1:
            out = self.thermal_augmentor
        else:
            out = self.rgb_augmentor
        return out(img, lm)


class ResizeAugmentor:
    def __init__(self, output_shape=(224, 224)):
        self.resize = T.Resize(output_shape)

    def __call__(self, thermal, rgb):
        return self.resize(thermal), self.resize(rgb)


# based on https://github.com/OsamaMazhar/Random-Shadows-Highlights/blob/master/RandomShadowsHighlights.py

def sample_shadow_masks(w=128, h=128, left_high_factor=50, left_low_factor=50, right_high_factor=50, right_low_factor=50):
    tl = (0, left_high_factor)
    bl = (0, left_high_factor+left_low_factor)
    
    tr = (w, right_high_factor)
    br = (w, right_high_factor+right_low_factor)
    
    contour = np.array([tl, tr, br, bl], dtype=np.int32)
    
    mask = np.zeros([h, w, 3],np.uint8)
    cv2.fillPoly(mask,[contour],(255,255,255))
    if np.random.rand() > 0.5:
        mask = np.transpose(mask, axes=[1, 0, 2])
    return mask # cv2.bitwise_not(mask)

def shadow_maker(im, b=0.8, w=128, h=128, left_high_factor=50, left_low_factor=50, right_high_factor=50, right_low_factor=50):
    mask = sample_shadow_masks(w, h, left_high_factor, left_low_factor, right_high_factor, right_low_factor)
    mask = torch.tensor(mask).permute(2, 0, 1).to(torch.bool)
    im[mask] = b * im[mask]
    return im

def random_shadow_maker(im, factor_range=(5, 256-5), b_range=(.1, .9), w=256, h=256):
    b = np.random.uniform(low=b_range[0], high=b_range[1])
    left_high_factor = np.random.randint(low=factor_range[0], high=factor_range[1])
    right_high_factor = np.random.randint(low=factor_range[0], high=factor_range[1])
    left_low_factor = np.random.randint(low=factor_range[0], high=factor_range[1])
    right_low_factor = np.random.randint(low=factor_range[0], high=factor_range[1])
    return shadow_maker(im, b, w, h, left_high_factor, left_low_factor, right_high_factor, right_low_factor)



def additional_augmentation(batch_rgb, blurrer, gray, cinv, jitter, HEIGHT, WIDTH):
    cbatch_size = batch_rgb.shape[0]
    tidx = torch.randperm(cbatch_size)[:int(cbatch_size/4)]
    batch_rgb[tidx] = gray(cinv(batch_rgb[tidx]))
    tidx = torch.randperm(cbatch_size)[:int(cbatch_size/4)]
    batch_rgb[tidx] = jitter(batch_rgb[tidx])
    tidx = torch.randperm(cbatch_size)[:int(cbatch_size/4)]
    batch_rgb[tidx] = blurrer(batch_rgb[tidx])
    tidx = torch.randperm(cbatch_size)[:int(cbatch_size/4)]
    for tidx_it in tidx:
        batch_rgb[tidx_it] = random_shadow_maker(batch_rgb[tidx_it], factor_range=(5, HEIGHT-5), 
                                                 w=HEIGHT, h=WIDTH)
    return batch_rgb
