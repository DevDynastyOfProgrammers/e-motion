import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.vision.config import DEFAULT_DEVICE, IMG_SIZE
from ml.vision.models import EmotionCNN
from ml.vision.utils import get_device
from research.vision.dataset import Config, build_dataloaders

# Hyperparameters
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch_idx}', unit='batch')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def main():
    # Setup Paths
    # Assumption: dataset is in 'research/vision/dataset/data' or similar
    # Adjust 'root_dir' to point to your actual FER-2013 extracted folder
    dataset_root = 'research/vision/data'

    if not os.path.exists(dataset_root):
        logger.error(f'Dataset not found at {dataset_root}. Please set correct path in train.py')
        logger.info('Structure should be: research/vision/data/train/[class_folders]')
        return

    # Create config
    cfg = Config(root_dir=dataset_root, image_size=(IMG_SIZE, IMG_SIZE), batch_size=64, force_rebuild_val=False)

    device = get_device('cuda')  # Try to force CUDA for training
    logger.info(f'Using device: {device}')

    # Data
    train_loader, val_loader, _ = build_dataloaders(cfg)
    if not train_loader:
        return

    # Model
    # 5 classes: Angry_Disgust, Fear_Surprise, Happy, Sad, Neutral
    model = EmotionCNN(num_classes=5, in_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # Weights Folder
    weights_dir = Path('research/vision/weights')
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    logger.info('🚀 Starting training...')

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(
            f'Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc:.2f}%'
        )

        scheduler.step(val_acc)

        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = weights_dir / 'best_emotion_model.pth'
            torch.save(model.state_dict(), save_path)
            logger.success(f'New best model saved! ({val_acc:.2f}%)')

    logger.info(f'🏁 Training complete. Best Accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
