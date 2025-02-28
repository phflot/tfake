from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Pix2PixDatasets(Dataset):
    """Dataset wrapper to allow for image translation"""
    def __init__(self, dataset, normalize=True):
        self.dataset = dataset
        self.normalize = normalize
        self.rgb_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    def __getitem__(self, item):
        thermal, rgb = self.dataset[item]
        if self.normalize:
            rgb /= 255
            thermal -= thermal.min() # if thermal.min() > 0 else 1
            thermal /= thermal.max() # if thermal.max() > 0 else 1

        ret = {
            "A": rgb,
            "B": thermal,
            "A_paths": f"{item}.png",
            "B_paths": f"{item}.png"
        }
        return ret

    def __len__(self):
        return len(self.dataset)