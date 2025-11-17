import os
import random
from torch.utils.data import DataLoader
from .dataset import FERImageDataset
from .augmentations import get_train_augmentations, get_val_augmentations
from .utils import set_seed, ensure_dir
from .logger import get_logger

logger = get_logger()

def create_val_split(root_dir, val_ratio, seed, force=False):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    if os.path.exists(val_dir) and not force:
        logger.info("Val split already exists — skipping creation.")
        return

    ensure_dir(val_dir, remove_if_exists=True)
    set_seed(seed)

    classes = sorted(os.listdir(train_dir))
    for cls in classes:
        class_path = os.path.join(train_dir, cls)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg"))]
        val_count = int(len(images) * val_ratio)
        val_imgs = random.sample(images, val_count)

        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        for img_name in val_imgs:
            os.rename(os.path.join(class_path, img_name),
                      os.path.join(val_dir, cls, img_name))
    logger.info(f"Validation split created at {val_dir}")

def list_image_samples(root_split_dir, label_map):
    samples = []
    for cls, idx in label_map.items():
        class_dir = os.path.join(root_split_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg")):
                samples.append((os.path.join(class_dir, fname), idx))
    return samples

def build_dataloaders(cfg):
    set_seed(cfg.random_state)
    label_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    label_map = {name: i for i, name in enumerate(label_names)}

    create_val_split(cfg.root_dir, cfg.val_ratio, cfg.random_state, cfg.force_rebuild_val)

    train_samples = list_image_samples(os.path.join(cfg.root_dir, "train"), label_map)
    val_samples = list_image_samples(os.path.join(cfg.root_dir, "val"), label_map)
    test_samples = list_image_samples(os.path.join(cfg.root_dir, "test"), label_map)

    train_aug = get_train_augmentations(cfg.image_size)
    val_aug = get_val_augmentations(cfg.image_size)

    train_ds = FERImageDataset(train_samples, transform=train_aug,
                               grayscale=cfg.grayscale, convert_to_3ch=cfg.convert_to_3ch)
    val_ds = FERImageDataset(val_samples, transform=val_aug,
                             grayscale=cfg.grayscale, convert_to_3ch=cfg.convert_to_3ch)
    test_ds = FERImageDataset(test_samples, transform=val_aug,
                              grayscale=cfg.grayscale, convert_to_3ch=cfg.convert_to_3ch)

    logger.info(f"Dataset ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                   num_workers=cfg.num_workers, pin_memory=cfg.pin_memory),
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                   num_workers=cfg.num_workers, pin_memory=cfg.pin_memory),
        DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                   num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    )
