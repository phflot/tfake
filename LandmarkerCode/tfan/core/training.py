from os import getenv
from multiprocessing import cpu_count

import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from .datasets import *
from .models import *
from .loss import GaussianNLLLoss
from ..neurovc.util import draw_landmarks, normalize_color


def invert_normalize(tensor):
    mean = torch.tensor([0.4850, 0.4560, 0.4060])
    std = torch.tensor([0.2290, 0.2240, 0.2250])

    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    return tensor * std[:, None, None] + mean[:, None, None]


class JointLandmarkTrainerAblation:
    """Joint Trainer for all ablation models."""
    class Model:
        def __init__(self, n_landmarks=70, gpus=[0, 1], learning_rate=0.0004, checkpoint=None,
                     epoch_offset=0):
            self.model = nn.DataParallel(DMMv2(n_landmarks=n_landmarks), device_ids=gpus)
            self.epoch_offset = epoch_offset
            if checkpoint is not None:
                self.model.load_state_dict(
                    torch.load(checkpoint))
            self.model.cuda()
            self.model.train()
            self.gnll = GaussianNLLLoss()
            self.learning_rate = learning_rate

            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate,
                                         weight_decay=.00005, eps=1e-8)
            print(
                f"Number of parameters in model = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
            self.scaler = GradScaler(enabled=True)
            self.scheduler = None

        def set_scheduler(self, n_epochs, dataloader):
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.learning_rate,
                                                           total_steps=n_epochs * len(dataloader) + 100,
                                                           epochs=n_epochs,
                                                           pct_start=0.3, cycle_momentum=True,
                                                           anneal_strategy='cos')
            for i in range(self.epoch_offset * len(dataloader)):
                self.scheduler.step()

        def set_scheduler_stepLR(self, n_epochs=None, dataloader=None):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.5)
            for i in range(self.epoch_offset * len(dataloader)):
                self.scheduler.step()

    def __init__(self, dataset=None, gpus=[0], batch_size=4, checkpoints=None, learning_rate=0.002,
                 progress_folder="landmarks/lm_test", display_step=100, model_path="landmarks/lm_models",
                 n_landmarks=70, num_workers=50, thermal_prob=0.4):

        if checkpoints is None:
            checkpoints = {
                "rgb_gray": None,
                "thermal": None,
                "joint": None
            }

        self.models = {
            # "rgb": self.Model(n_landmarks=n_landmarks, gpus=gpus, learning_rate=learning_rate),
            "rgb_gray": self.Model(n_landmarks=n_landmarks, gpus=gpus, learning_rate=learning_rate, checkpoint=checkpoints["rgb_gray"]),
            "thermal": self.Model(n_landmarks=n_landmarks, gpus=gpus, learning_rate=learning_rate, checkpoint=checkpoints["thermal"]),
            "joint": self.Model(n_landmarks=n_landmarks, gpus=gpus, learning_rate=learning_rate, checkpoint=checkpoints["joint"])
        }

        if num_workers == 0:
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                          num_workers=num_workers, pin_memory=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                          num_workers=num_workers, pin_memory=True,
                                                          persistent_workers=True, prefetch_factor=2)
        self.display_step = display_step

        self.results_folder = progress_folder
        self.model_path = model_path

        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        self.mse = nn.MSELoss()
        self.thermal_prob = thermal_prob

    def _display_progress(self, img, cond, fake, iteration="test"):

        img = invert_normalize(img)
        tmp = img[0].detach().cpu().permute((1, 2, 0)).numpy()[..., ::-1]

        fake = torch.nan_to_num(fake, nan=0.0)
        cond = torch.nan_to_num(cond, nan=0.0)

        img_cond = draw_landmarks(tmp, cond[0, :, :2].detach().cpu().numpy() * 224)
        img_fake = draw_landmarks(tmp, fake[0, :, :2].detach().cpu().numpy() * 224)

        cv2.imwrite(join(self.results_folder, iteration + ".png"),
                    (np.concatenate([img_cond, img_fake], 1) * 255).astype(np.uint8))

    def train(self, n_epochs):

        for key in self.models.keys():
            self.models[key].set_scheduler_stepLR(n_epochs, self.dataloader)

        iterations = 0
        image_counter = 0
        for epoch in range(n_epochs):
            for i, data in enumerate(self.dataloader):
                for key in self.models.keys():
                    self.models[key].optimizer.zero_grad()

                image, img_gray, thermal, pose = [x.cuda() for x in data]

                images = {# "rgb": image,
                          "rgb_gray": img_gray,
                          "thermal": thermal,
                          "joint": thermal if np.random.rand() < self.thermal_prob else img_gray}
                iterations += 1
                for key, image in images.items():
                    model = self.models[key]
                    predicted = model.model(image)
                    loss = model.gnll(predicted[..., :-1], pose, predicted[..., -1])

                    model.scaler.scale(loss).backward()
                    model.scaler.unscale_(model.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), True)
                    model.scaler.step(model.optimizer)
                    model.scheduler.step()
                    model.scaler.update()

                    if iterations % 1000 == 0:
                        torch.save(model.model.state_dict(), os.path.join(self.model_path, f'joint_lm_{iterations:05d}_{key}.pt'))
                    if iterations % self.display_step == 0:
                        self._display_progress(image, pose, predicted, iteration=f'{image_counter:06}_{key}')
                        with torch.no_grad():
                            mse = self.mse(predicted[..., :-1], pose)
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}, nme: {mse} {key}')

                if iterations % self.display_step == 0:
                    image_counter += 1
            torch.save(model.model.state_dict(), os.path.join(self.model_path, f'joint_lm_{epoch+1:03d}_{iterations:05d}_{key}.pt'))
        for key in self.models.keys():
            model = self.models[key]
            torch.save(model.model.state_dict(), os.path.join(self.model_path, f'joint_70pt_{key}.pt'))


class JointLandmarkTrainer:
    """Trainer for the joint RGB + thermal multimodal model."""
    def __init__(self, dataset=None, gpus=[0], batch_size=4, checkpoint=None, learning_rate=0.002,
                 progress_folder="landmarks/lm_test", display_step=100, model_path="landmarks/lm_models",
                 n_landmarks=70, num_workers=50):

        self.model = nn.DataParallel(DMMv2(n_landmarks=n_landmarks), device_ids=gpus)

        if num_workers > 0:
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                          num_workers=num_workers,
                                                          pin_memory=True, persistent_workers=True, )
        else:
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                          num_workers=num_workers, pin_memory=True)
        self.display_step = display_step
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint), strict=False)
            print(f"loaded model {checkpoint}")

        self.model.cuda()
        self.model.train()

        self.results_folder = progress_folder
        self.model_path = model_path

        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        self.gnll = GaussianNLLLoss()
        self.mse = nn.MSELoss()
        self.learning_rate = learning_rate

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate,
                                     weight_decay=.00005, eps=1e-8)
        print(f"Number of parameters in model = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        self.scaler = GradScaler(enabled=True)
        self.scheduler = None

    def _display_progress(self, img, cond, fake, iteration="test"):

        img = invert_normalize(img)
        tmp = img[0].detach().cpu().permute((1, 2, 0)).numpy()[..., ::-1]

        fake = torch.nan_to_num(fake, nan=0.0)
        cond = torch.nan_to_num(cond, nan=0.0)

        img_cond = draw_landmarks(tmp, cond[0, :, :2].detach().cpu().numpy() * 224)
        img_fake = draw_landmarks(tmp, fake[0, :, :2].detach().cpu().numpy() * 224)

        cv2.imwrite(join(self.results_folder, iteration + ".png"),
                    (np.concatenate([img_cond, img_fake], 1)).astype(np.uint8))

    def train(self, n_epochs):
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.learning_rate,
                                                       total_steps=n_epochs * len(self.dataloader) + 100,
                                                       epochs=n_epochs,
                                                       pct_start=0.3, cycle_momentum=True,
                                                       anneal_strategy='linear')
        
        iterations = 0
        image_counter = 0
        for epoch in range(n_epochs):
            for i, data in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                condition, pose = [x.cuda() for x in data]

                predicted = self.model(condition)
                loss = self.gnll(predicted[..., :-1], pose, predicted[..., -1])
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), True)
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                iterations += 1
                if iterations % 1000 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, f'joint_70pt_{iterations:05d}.pt'))
                if iterations % self.display_step == 0:
                    self._display_progress(condition, pose, predicted, iteration=f'{image_counter:06}_epoch{epoch + 1}_iter{i + 1:5d}])_loss{loss}')
                    with torch.no_grad():
                        mse = self.mse(predicted[..., :-1], pose)
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}, nme: {mse}')
                    image_counter += 1
                if i % 100 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
            torch.save(self.model.state_dict(), os.path.join(self.model_path, f'joint_70pt_epoch{epoch:05d}_iter_{iterations:05d}.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'joint_70pt.pt'))
