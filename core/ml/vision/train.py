import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .dataset import build_dataloaders, Config
from .models import EmotionCNN
from .utils import get_device
from .config import IMG_SIZE

EPOCHS = 50
LEARNING_RATE = 1e-4

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
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

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%; Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return train_losses, val_losses, train_accs, val_accs

def main(cfg: Config):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    model = EmotionCNN(num_classes=5, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS)

    # Сохраняем model_state_dict (без оптимизатора, чтобы файл был компактнее)
    torch.save({'model_state_dict': model.state_dict()}, 'weights/emotion_model.pth')
    print("Saved weights to weights/emotion_model.pth")

if __name__ == "__main__":
    # example default config: замените root_dir при запуске
    cfg = Config(root_dir="dataset", image_size=(IMG_SIZE, IMG_SIZE), batch_size=64)
    main(cfg)
