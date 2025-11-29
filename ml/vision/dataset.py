import os
import random
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from ml.vision.models import EmotionCNN
from ml.vision.config import IMG_SIZE
import torch
import logging

logger = logging.getLogger("ml.vision.dataset")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

@dataclass
class Config:
    root_dir: str
    image_size: tuple = (IMG_SIZE, IMG_SIZE)
    val_ratio: float = 0.1
    random_state: int = 42
    grayscale: bool = True
    convert_to_3ch: bool = False
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    force_rebuild_val: bool = False

# Аугментации — берём логичные, консервативные (из train_model.py). :contentReference[oaicite:6]{index=6}
def get_train_augmentations(image_size: tuple):
    h, w = image_size
    return A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=0, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomScale(scale_limit=0.05, p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.2), per_channel=False, noise_scale_factor=0.5, p=0.6),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4)
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.Resize(h, w),
        ToTensorV2()
    ])

def get_val_augmentations(image_size: tuple):
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        ToTensorV2()
    ])

class FERImageDataset(Dataset):
    """Dataset that expects list of (path, label)"""
    def __init__(self, samples: List[Tuple[str,int]], transform=None, grayscale=True, convert_to_3ch=False):
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
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = img.float() / 255.0
        return img, label

# Helpers для формирования списков и split'ов — адаптированы из train_model.py. :contentReference[oaicite:7]{index=7}
def ensure_dir(path: str, remove_if_exists: bool = False):
    if os.path.exists(path):
        if remove_if_exists:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_val_split(root_dir: str, val_ratio: float, seed: int, force: bool = False):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    if os.path.exists(val_dir) and not force:
        logger.info("Val split already exists — skipping creation.")
        return
    ensure_dir(val_dir, remove_if_exists=True)
    set_seed(seed)
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    for cls in classes:
        class_path = os.path.join(train_dir, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        val_count = int(len(images) * val_ratio)
        val_imgs = random.sample(images, val_count)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        for img_name in val_imgs:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(val_dir, cls, img_name)
            shutil.move(src, dst)
    logger.info(f"Validation split created at {val_dir}")

def list_image_samples(root_split_dir: str, label_map: Dict[str,int]):
    samples = []
    for cls, idx in label_map.items():
        class_dir = os.path.join(root_split_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(class_dir, fname), idx))
    return samples

# Утилита создания аугментированного набора на диск (используется в build_dataloaders)
def get_train_augmentations_for_saving(image_size: tuple):
    # как в train_model.py, но без ToTensorV2, сохраняем в файлы
    h, w = image_size
    return A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=0, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomScale(scale_limit=0.05, p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.2), per_channel=False, noise_scale_factor=0.5, p=0.6),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4)
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.OneOf([A.MotionBlur(blur_limit=3, p=0.2), A.MedianBlur(blur_limit=3, p=0.1)], p=0.2),
        A.Resize(h, w),
    ])

def create_augmented_dataset(source_dir, target_dir, num_augmentations, image_size, random_state):
    import cv2
    if os.path.exists(target_dir):
        logger.info(f"Augmented dataset already exists at {target_dir}")
        return
    os.makedirs(target_dir, exist_ok=True)
    augmentation_transform = get_train_augmentations_for_saving(image_size)
    for label in os.listdir(source_dir):
        label_source_dir = os.path.join(source_dir, label)
        label_target_dir = os.path.join(target_dir, label)
        os.makedirs(label_target_dir, exist_ok=True)
        if not os.path.isdir(label_source_dir):
            continue
        for img_file in os.listdir(label_source_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(label_source_dir, img_file)
                target_path = os.path.join(label_target_dir, img_file)
                shutil.copy2(source_path, target_path)
                image = cv2.imread(source_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for i in range(num_augmentations):
                    augmented = augmentation_transform(image=image)
                    augmented_image = augmented["image"]
                    aug_filename = f"{os.path.splitext(img_file)[0]}_aug_{i+1}.jpg"
                    aug_path = os.path.join(label_target_dir, aug_filename)
                    aug_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(aug_path, aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info(f"Created augmented dataset with {num_augmentations} augmentations per image")

# Собираем даталоадеры (основная точка входа для тренировки)
def build_dataloaders(cfg: Config):
    set_seed(cfg.random_state)
    label_names = ["angry_disgust", "fear_surprise", "happy", "neutral", "sad"]
    label_map = {name: i for i, name in enumerate(label_names)}

    create_val_split(cfg.root_dir, cfg.val_ratio, cfg.random_state, cfg.force_rebuild_val)

    train_dir = os.path.join(cfg.root_dir, "train")
    val_dir = os.path.join(cfg.root_dir, "val")
    test_dir = os.path.join(cfg.root_dir, "test")

    augmented_train_dir = os.path.join(cfg.root_dir, "train_augmented")
    create_augmented_dataset(train_dir, augmented_train_dir, 5, cfg.image_size, cfg.random_state)

    train_samples = list_image_samples(augmented_train_dir, label_map)
    val_samples = list_image_samples(val_dir, label_map)
    test_samples = list_image_samples(test_dir, label_map)

    train_aug = get_train_augmentations(cfg.image_size)
    val_aug = get_val_augmentations(cfg.image_size)

    train_ds = FERImageDataset(train_samples, transform=train_aug, grayscale=cfg.grayscale, convert_to_3ch=cfg.convert_to_3ch)
    val_ds = FERImageDataset(val_samples, transform=val_aug, grayscale=cfg.grayscale, convert_to_3ch=cfg.convert_to_3ch)
    test_ds = FERImageDataset(test_samples, transform=val_aug, grayscale=cfg.grayscale, convert_to_3ch=cfg.convert_to_3ch)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    logger.info(f"Dataset ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader
