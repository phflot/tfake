import timm
import torch
import torch.nn as nn
import torchvision.models
import cv2

try:
    pass
except:
    from .cvt import create_model as cvt_create_model

import numpy as np

from os.path import join

from einops import rearrange


MOBILENET = "mobilenetv2_100"
RESNET = "resnet101"
CVT = "cvt"

DEVICE = "cuda"


def create_model(model):
    """Creates and configures a model based on the specified model type.
    
    Args:
        model (str): The type of model to create. Expects "cvt", "mobilenetv2_100", or "resnet101".

    Returns:
        tuple: The configured model and a reference to the forward features layers.
    """
    if model == "cvt":
        model = cvt_create_model(model_file="CvT-21-224x224-IN-1k.pth",
                                 cfg="lm_model_224.yaml")
        model.head.requires_grad = False
        forward_features = model.forward_features
        model.global_pool = lambda x: x
    else:
        model = timm.create_model(model, pretrained=True)
        model.classifier.requires_grad = False

        def forward_features(x):
            x = model.forward_features(x)
            return model.global_pool(x)
    return model, forward_features


def convert_model(model_path_in, model_path_out, mode="RGB", n_landmarks=70):
    """Converts a model saved with as DataParallel instance.

    Args:
        model_path_in (str): Path to the input model file.
        model_path_out (str): Path to save the converted model file.
        mode (str): Mode of the model, options are "RGB", "THERMAL", or "JOINT".
        n_landmarks (int): The number of landmarks to predict.
    """
    if mode == "RGB":
        model = DMM(n_landmarks=n_landmarks)
    elif mode == "THERMAL":
        model = DMMThermal(n_landmarks=n_landmarks)
    elif mode == "JOINT":
        model = DMMv2(n_landmarks=n_landmarks)
    dmm = nn.DataParallel(model, device_ids=[0])
    dmm.load_state_dict(
        torch.load(model_path_in, map_location=torch.device('cpu')),
        strict=False)
    torch.save(dmm.module.state_dict(), model_path_out)


def _warping_depth(eta, levels, m, n):
    """Computes the max pyramid depth based on image size and eta for 224x224 images."""
    min_dim = min(m, n)
    warping_depth = 0
    d = warping_depth

    for i in range(levels):
        warping_depth += 1
        min_dim = min_dim * eta
        if round(min_dim) < 224:
            break
        d = warping_depth
    return d


class DenseLandmarks:
    """Class for processing images to extract dense landmarks using different models."""
    def __init__(self, model_path="landmark_model.pt", device="cpu",
                 gpus=[0, 1], eta=0.75, max_lvl=0, stride=100, n_landmarks=70, mode="RGB", refine_landmarks=True):
        if mode == "RGB":
            dmm = DMM(n_landmarks=n_landmarks)
        elif mode == "THERMAL":
            dmm = DMMThermal(n_landmarks=n_landmarks)
        elif mode == "JOINT":
            dmm = DMMv2(n_landmarks=n_landmarks)
        elif mode == "XROMM":
            dmm = DMMThermal(n_landmarks=n_landmarks)

        self.refine_landmarks = refine_landmarks
        self.mode = mode
        dmm.load_state_dict(torch.load(model_path), strict=False)

        if device == "cuda":
            dmm = nn.DataParallel(dmm, device_ids=gpus)
        dmm.eval()
        dmm.to(device)
        self.device = device
        self.dmm = dmm
        self.img_shape = None
        self.eta = eta
        self.max_lvl = max_lvl
        self.stride = stride
        #self.transform = Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]),
        #                           std=torch.tensor([0.2290, 0.2240, 0.2250]))
        self.transform = lambda x: x

    def process(self, image, sliding_window=True):
        """Compute the landmarks for an image.

        Args:
            image (np.ndarray): The input image to process.
            sliding_window (bool): If True, uses a sliding window approach to extract landmarks.

        Returns:
            tuple: The extracted landmarks and the associated confidence scores.
        """
        if self.img_shape is None:
            img_shape = image.shape[:2]
            wp = _warping_depth(self.eta, 100, *img_shape)

        if self.mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.mode == "THERMAL":
            image = np.clip(image, 20, 40)
            image -= image.min()
            image /= image.max()
            image = np.expand_dims(image, 2)
        elif self.mode == "JOINT":
            if len(image.shape) == 2:
                image = np.clip(image, 20, 40)
                image -= image.min()
                image /= image.max()
                image = np.stack([image, image, image], 2) * 255
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.mode == "XROMM":
            image = image.astype(float)
            image = (image - image.min()) / (image.max() - image.min())
            if len(image.shape) > 2:
                image = image[:, :, 0]

        # image = image / 255
        if not sliding_window:
            return self.get_landmarks_single(image, self.refine_landmarks)

        best_score = np.inf
        lm = None
        best_scores = None
        for i in range(wp, self.max_lvl-1, -1):
            lvl_factor = self.eta ** i
            size = (int(round(img_shape[1] * lvl_factor)), int(round(img_shape[0] * lvl_factor)))
            img = cv2.resize(image, size)
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            hx = img_shape[1] / size[0]
            hy = img_shape[0] / size[1]
            lm_lvl, score, scores = self.get_landmarks(img.astype(np.float32), stride=self.stride)
            if score < best_score:
                best_score = score
                best_scores = scores
                lm = lm_lvl * np.expand_dims(np.array([hx, hy]), 0)

        return lm, best_scores

    def _refine_landmarks(self, img, lm_scaled):
        x_coords = lm_scaled[:, 0]
        y_coords = lm_scaled[:, 1]

        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)

        largest_side = max(y_max - y_min, x_max - x_min) * 2
        y_center = (y_max + y_min) / 2.2
        x_center = (x_max + x_min) / 2

        padding = int(largest_side // 2)

        padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                        value=[0, 0, 0])

        x_start = int(x_center - largest_side / 2 + padding)
        y_start = int(y_center - largest_side / 2 + padding)
        x_end = int(x_center + largest_side / 2 + padding)
        y_end = int(y_center + largest_side / 2 + padding)

        patch = padded_img[y_start:y_end, x_start:x_end]

        x = cv2.resize(patch, (224, 224))
        x = torch.from_numpy(x).to(torch.float32).to(self.device).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            cropped_transformed = self.transform(x)
            refined_lm = self.dmm(cropped_transformed)

        refined_lm_scaled = (refined_lm[..., :-1] * largest_side).cpu().detach().squeeze().numpy()
        refined_lm_scaled += np.array([[x_start - padding, y_start - padding]])
        confidences = refined_lm[..., -1].cpu().detach().squeeze().numpy()
        return refined_lm_scaled, confidences

    def get_landmarks_single(self, img, refine=True):
        """Extracts landmarks by resizing the image.

        Args:
            img (np.ndarray): The input image.
            refine (bool): If True, refines the extracted landmarks in a bbox around the initial detection.

        Returns:
            tuple: The extracted landmarks and the associated confidence scores.
        """
        shape = torch.tensor(img.shape[1::-1]).unsqueeze(0).unsqueeze(0).to(self.device)
        x = cv2.resize(img, (224, 224))
        with torch.no_grad():
            x = torch.from_numpy(x).to(torch.float32).to(self.device).permute(2, 0, 1).unsqueeze(0)
            x = self.transform(x)
            lm = self.dmm(x)

        lm_scaled = (lm[..., :-1] * shape).cpu().detach().squeeze().numpy()
        confidences = lm[..., -1].cpu().detach().squeeze().numpy()

        if refine:
            lm_scaled, confidences = self._refine_landmarks(img, lm_scaled)

        return lm_scaled, confidences

    def get_landmarks(self, img, stride=50):
        """Sliding window implementation for the landmarks using torch unfold.

        Args:
            img (np.ndarray): The input image.
            stride (int): The stride of the sliding window.

        Returns:
            tuple: The best landmarks found, the best score, and the associated confidence scores.
        """

        img_dims = img.shape
        y_pad = 224 - img_dims[0] % 224
        x_pad = 224 - img_dims[1] % 224

        y_pad_l = y_pad // 2
        y_pad_r = y_pad // 2 + y_pad % 2

        x_pad_l = x_pad // 2
        x_pad_r = x_pad // 2 + y_pad % 2

        pad = (0, 0, x_pad_l, x_pad_r, y_pad_l, y_pad_r)

        with torch.no_grad():
            x = torch.nn.functional.pad(torch.from_numpy(img).to(torch.float32), pad).to(self.device)
            img_unfold = x.unfold(0, 224, stride).unfold(1, 224, stride)
            s = img_unfold.shape
            img_unfold = img_unfold.reshape((s[0] * s[1],) + s[2:])
            lm = self.dmm(self.transform(img_unfold))
            best_scores = lm[..., -1].mean(1)
            best_score_idx = best_scores.argmin().item()
            best_score = best_scores[best_score_idx].item()
            offset = torch.Tensor([stride * (best_score_idx % s[1]), stride * (best_score_idx // s[1])]).unsqueeze(0).to(self.device)
            lm = lm[best_score_idx]
            lm_out = lm[:, :-1] * 224 + offset
            lm_out = lm_out - torch.tensor([x_pad_l, y_pad_l]).unsqueeze(0).to(self.device)

        lm_scaled = lm_out.cpu().detach().numpy()
        confidences = lm[..., -1].cpu().detach().numpy()

        return lm_scaled, best_score, confidences


class DMMThermal(nn.Module):
    """Network for single channel thermal images. """
    def __init__(self, n_landmarks, use_depth=False, backend="mobilenetv2_100"):
        super().__init__()
        self.use_depth = False
        if backend == "mobilenetv2_100":
            self.feature_network = timm.create_model("mobilenetv2_100", pretrained=True, num_classes=0)
            self.feature_network.classifier.requires_grad = False
            self.feature_network.conv_stem = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        elif backend == "resnet":
            self.feature_network = timm.create_model("resnet101", pretrained=True, num_classes=0)
            self.feature_network.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.feature_network.fc.requires_grad = False
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.n_landmarks = n_landmarks
        self.fc = nn.Linear(1280, n_landmarks * (self.use_depth + 3))
        # self.fc = nn.Linear(1024, n_landmarks * (self.use_depth + 3))
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.feature_network.conv_stem.weight)

        self.fc.requires_grad_(True)
        self.act = nn.ReLU()
        self.act_leaky = nn.LeakyReLU()

        torchvision.models.mobilenet_v3_large()

    def forward(self, x):
        x = self.feature_network.forward_features(x)
        x = self.feature_network.global_pool(x)
        x = self.fc(x)
        x = x.reshape((x.shape[0], self.n_landmarks, self.use_depth + 3))
        x = self.act_leaky(x)

        return x


class DMMv2(nn.Module):
    """Network for 3-channel images. """
    def __init__(self, n_landmarks, use_depth=False):
        super().__init__()
        self.use_depth = False
        self.feature_network = timm.create_model("mobilenetv2_100", pretrained=True, num_classes=0)
        self.feature_network.classifier.requires_grad = False

        self.n_landmarks = n_landmarks
        self.fc = nn.Linear(1280, n_landmarks * (self.use_depth + 3))
        nn.init.xavier_uniform(self.fc.weight)

        self.fc.requires_grad_(True)
        self.act = nn.ReLU()
        self.act_leaky = nn.LeakyReLU()

    def forward(self, x):
        x = self.feature_network.forward_features(x)
        x = self.feature_network.global_pool(x)
        x = self.fc(x)
        x = x.reshape((x.shape[0], self.n_landmarks, self.use_depth + 3))
        x = self.act_leaky(x)

        return x


class DMM(nn.Module):
    """Legacy network for 3-channel images. """
    def __init__(self, model=MOBILENET, n_landmarks=468, use_depth=False, train_head=True):
        super().__init__()
        self.feature_network, self.forward_features = create_model(model)
        self.transform = lambda x: x
        self.use_depth = int(use_depth)
        self.n_landmarks = n_landmarks
        self.fc = nn.Linear(1280, n_landmarks * (self.use_depth + 3))
        nn.init.xavier_uniform(self.fc.weight)
        self.fc.requires_grad_(True)
        self.act = nn.ReLU()
        self.act_leaky = nn.LeakyReLU()
        self.train_head = train_head

    def forward(self, x):
        if self.train_head:
            x = self.feature_network.forward_features(self.transform(x))
            x = self.feature_network.global_pool(x)
        else:
            with torch.no_grad():
                x = self.feature_network.forward_features(self.transform(x))
                x = self.feature_network.global_pool(x)
        x = self.fc(x)
        x = x.reshape((x.shape[0], self.n_landmarks, self.use_depth + 3))
        x = self.act_leaky(x)
        return x
