import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from .models import EmotionCNN
from .config import EMOTION_CLASSES, IMG_SIZE
from .utils import load_checkpoint, get_device

class EmotionRecognizer:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = get_device(device)
        # Инициализация модели (кол-во классов = len(EMOTION_CLASSES))
        self.model = EmotionCNN(num_classes=len(EMOTION_CLASSES))
        self.model = load_checkpoint(self.model, model_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        # Трансформации — те же, что при валидации
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, frame: np.ndarray):
        """
        Принимает кадр (numpy array от OpenCV BGR), возвращает словарь с вероятностями и предсказанием.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame).resize((IMG_SIZE, IMG_SIZE))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        result = {}
        for i, emotion_name in EMOTION_CLASSES.items():
            result[f"prob_{emotion_name.lower()}"] = float(probs[0][i])
        top_prob, top_class_idx = torch.max(probs, 1)
        result['predicted_class'] = EMOTION_CLASSES[top_class_idx.item()]
        result['confidence'] = float(top_prob)
        return result
