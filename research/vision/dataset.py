import os
import random
import shutil
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ml.vision.config import IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD


@dataclass
class Config:
    root_dir: str
    image_size: tuple = (IMG_SIZE, IMG_SIZE)
    val_ratio: float = 0.1
    random_state: int = 42
    grayscale: bool = False  # Training mostly in RGB (3 channels) to match ImageNet stats
    convert_to_3ch: bool = True  # Force 3 channels even if source is gray
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    force_rebuild_val: bool = False


def get_train_augmentations(image_size: tuple) -> A.Compose:
    """
    Strong augmentations for training.
    Includes Normalization to match Production Inference.
    """
    height, width = image_size
    return A.Compose(
        [
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03, scale_limit=0.05, rotate_limit=0, border_mode=cv2.BORDER_REFLECT_101, p=0.5
            ),
            A.RandomScale(scale_limit=0.05, p=0.3),
            # Noise and Blur
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.6),  # Fixed param name
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
                ],
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            # Geometry
            A.Resize(height, width),
            # Critical: Normalize using same stats as Inference
            A.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
            ToTensorV2(),
        ]
    )


def get_val_augmentations(image_size: tuple) -> A.Compose:
    """
    Validation transforms (Resize + Normalize only).
    """
    height, width = image_size
    return A.Compose([A.Resize(height, width), A.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD), ToTensorV2()])


class FERImageDataset(Dataset):
    """
    PyTorch Dataset for Facial Emotion Recognition.
    Expects list of (image_path, label_idx).
    """

    def __init__(self, samples: list[tuple[str, int]], transform=None, convert_to_3ch=True) -> None:
        self.samples = samples
        self.transform = transform
        self.convert_to_3ch = convert_to_3ch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Read image
        img = cv2.imread(path)
        if img is None:
            # Fallback for corrupted images
            # Create a black image to avoid crashing training
            img = np.zeros((48, 48, 3), dtype=np.uint8)
            logger.warning(f'Failed to read image: {path}')
        else:
            # OpenCV loads as BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            # Manual fallback if no transform
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


# --- UTILS FOR FILE HANDLING ---


def ensure_dir(path: str, remove_if_exists: bool = False) -> None:
    if os.path.exists(path) and remove_if_exists:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_val_split(root_dir: str, val_ratio: float, seed: int, force: bool = False) -> None:
    """
    Splits 'train' folder into 'train' and 'val'.
    """
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    if not os.path.exists(train_dir):
        logger.error(f'Train directory not found at {train_dir}')
        return

    if os.path.exists(val_dir) and not force:
        logger.info('Val split already exists — skipping creation.')
        return

    ensure_dir(val_dir, remove_if_exists=True)
    set_seed(seed)

    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    for cls in classes:
        class_path = os.path.join(train_dir, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Calculate split
        val_count = int(len(images) * val_ratio)
        if val_count == 0:
            continue

        val_imgs = random.sample(images, val_count)

        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        for img_name in val_imgs:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(val_dir, cls, img_name)
            shutil.move(src, dst)

    logger.info(f'Validation split created at {val_dir}')


def list_image_samples(root_split_dir: str, label_map: dict[str, int]) -> list[tuple[str, int]]:
    samples = []
    for cls_name, idx in label_map.items():
        class_dir = os.path.join(root_split_dir, cls_name)
        # Handle case sensitivity mapping if needed (e.g. "Happy" vs "happy")
        # Trying to find the directory case-insensitively
        if not os.path.isdir(class_dir):
            found = False
            for d in os.listdir(root_split_dir):
                if d.lower() == cls_name.lower():
                    class_dir = os.path.join(root_split_dir, d)
                    found = True
                    break
            if not found:
                continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((os.path.join(class_dir, fname), idx))
    return samples


def build_dataloaders(cfg: Config):
    """
    Main entry point to prepare DataLoaders.
    """
    set_seed(cfg.random_state)

    # Needs to match EMOTION_CLASSES keys in ml/vision/config.py
    # But usually dataset folders are named string: "angry", "happy"
    # We map folder names to Indices 0..4
    # Expected folders: "angry", "fear", "happy", "neutral", "sad"
    # (Mapping based on ml.vision.config order)

    # 0: Angry_Disgust, 1: Fear_Surprise, 2: Happy, 3: Sad, 4: Neutral
    # Note: Check your dataset folder names!
    label_map = {
        'angry': 0,
        'disgust': 0,
        'angry_disgust': 0,
        'fear': 1,
        'surprise': 1,
        'fear_surprise': 1,
        'happy': 2,
        'sad': 3,
        'neutral': 4,
    }

    create_val_split(cfg.root_dir, cfg.val_ratio, cfg.random_state, cfg.force_rebuild_val)

    train_dir = os.path.join(cfg.root_dir, 'train')
    val_dir = os.path.join(cfg.root_dir, 'val')
    test_dir = os.path.join(cfg.root_dir, 'test')  # Optional

    train_samples = list_image_samples(train_dir, label_map)
    val_samples = list_image_samples(val_dir, label_map)

    if not train_samples:
        logger.error(f'No training samples found in {train_dir}. Check folder structure!')
        return None, None, None

    train_aug = get_train_augmentations(cfg.image_size)
    val_aug = get_val_augmentations(cfg.image_size)

    train_ds = FERImageDataset(train_samples, transform=train_aug, convert_to_3ch=cfg.convert_to_3ch)
    val_ds = FERImageDataset(val_samples, transform=val_aug, convert_to_3ch=cfg.convert_to_3ch)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    test_loader = None
    if os.path.exists(test_dir):
        test_samples = list_image_samples(test_dir, label_map)
        if test_samples:
            test_ds = FERImageDataset(test_samples, transform=val_aug, convert_to_3ch=cfg.convert_to_3ch)
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )

    logger.info(f'Dataset prepared: Train={len(train_ds)}, Val={len(val_ds)}')
    return train_loader, val_loader, test_loader
