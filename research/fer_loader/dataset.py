import os
import cv2
import torch
from torch.utils.data import Dataset
from research.fer_loader.utils import set_seed, ensure_dir

class FERImageDataset(Dataset):
    def __init__(self, samples, transform=None, grayscale=True, convert_to_3ch=False):
        self.samples = samples
        self.transform = transform
        self.grayscale = grayscale
        self.convert_to_3ch = convert_to_3ch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image {path}")

        if len(img.shape) == 2 or self.convert_to_3ch:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img.float() / 255.0, label
