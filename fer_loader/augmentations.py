import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentations(image_size: tuple[int, int]) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.03, 0.03),
                 rotate=(0, 0), shear=(-3, -3), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
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

def get_val_augmentations(image_size: tuple[int, int]) -> A.Compose:
    h, w = image_size
    return A.Compose([A.Resize(h, w), ToTensorV2()])
