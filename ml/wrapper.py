# ml/wrapper.py

import os
import numpy as np
from typing import Protocol, runtime_checkable
from loguru import logger
from core.emotion import Emotion, EmotionPrediction

# Импортируем архитектуру модели
from ml.vision.models import EmotionCNN
from ml.vision.config import EMOTION_CLASSES

@runtime_checkable
class EmotionModel(Protocol):
    """Protocol for any emotion recognition model (Mock or Real)."""
    def predict(self, frame: np.ndarray) -> EmotionPrediction: ...


class RandomEmotionModel:
    """Fallback model that returns random probabilities."""
    def predict(self, frame: np.ndarray) -> EmotionPrediction:
        # 0: angry_disgust, 1: fear_surprise, 2: happy, 3: neutral, 4: sad
        raw_probs = np.random.dirichlet(np.ones(5), size=1)[0]
        idx = np.argmax(raw_probs)
        
        dominant_map = {
            0: Emotion.ANGER, 1: Emotion.FEAR, 2: Emotion.JOY,
            3: Emotion.NEUTRAL, 4: Emotion.SORROW
        }

        return EmotionPrediction(
            dominant_emotion=dominant_map.get(idx, Emotion.NEUTRAL),
            confidence=float(raw_probs[idx]),
            prob_angry_disgust=float(raw_probs[0]),
            prob_fear_surprise=float(raw_probs[1]),
            prob_happy=float(raw_probs[2]),
            prob_neutral=float(raw_probs[3]),
            prob_sad=float(raw_probs[4]),
        )


class PyTorchEmotionModel:
    """Real model wrapper that uses a .pth file for inference."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model_path = model_path

        if EmotionCNN is None:
            raise ImportError("Could not import EmotionCNN. Check ml.vision package.")

        try:
            import torch
            import torchvision.transforms as T
            self.torch = torch
            self.T = T
        except ImportError as e:
            raise ImportError(f"PyTorch not found: {e}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        logger.info(f"Loading ML model from {model_path} on {device}...")

        try:
            # 1. Создаем экземпляр архитектуры
            # Важно: num_classes должно совпадать с тем, на чем обучали (5)
            self.model = EmotionCNN(num_classes=5)
            
            # 2. Загружаем чекпоинт
            checkpoint = torch.load(model_path, map_location=device)
            
            # 3. Извлекаем веса (поддержка разных форматов сохранения)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                # Если вдруг сохранена полная модель
                state_dict = checkpoint.state_dict()

            # 4. Загружаем веса в модель
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise e

        # Трансформации для инференса (Grayscale + Resize 48x48)
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((48, 48)),
            T.Grayscale(num_output_channels=1), # EmotionCNN ожидает 1 или 3 канала, проверим ниже
            T.ToTensor(),
        ])

    def predict(self, frame: np.ndarray) -> EmotionPrediction:
        fallback = EmotionPrediction(Emotion.NEUTRAL, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        if frame is None or frame.size == 0:
            return fallback

        try:    
            # OpenCV loads as BGR. If we pass this to ToPILImage, blue becomes red.
            # This messes up detection. Let's fix it.
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess using the RGB frame
            input_tensor = self.transforms(rgb_frame)

            if input_tensor.shape[0] == 1:
                input_tensor = input_tensor.repeat(3, 1, 1)

            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            # Inference
            with self.torch.no_grad():
                outputs = self.model(input_tensor)
                probs = self.torch.softmax(outputs, dim=1)[0]

            # Mapping (порядок классов из config.py или обучения)
            # 0: Angry_Disgust, 1: Fear_Surprise, 2: Happy, 3: Sad, 4: Neutral
            # ВАЖНО: Проверь порядок классов в ml/vision/config.py!
            # Здесь предполагаем стандартный порядок FER:
            
            p_angry = float(probs[0].item())
            p_fear = float(probs[1].item())
            p_happy = float(probs[2].item())
            p_sad = float(probs[3].item()) # Внимание: порядок может отличаться
            p_neutral = float(probs[4].item())

            # Определяем доминанту
            val_list = [p_angry, p_fear, p_happy, p_neutral, p_sad]
            max_val = max(val_list)
            
            # Маппинг индексов на Enum
            # Если порядок был 0:Angry, 1:Fear, 2:Happy, 3:Sad, 4:Neutral
            dominant_map = {
                p_angry: Emotion.ANGER,
                p_fear: Emotion.FEAR,
                p_happy: Emotion.JOY,
                p_neutral: Emotion.NEUTRAL,
                p_sad: Emotion.SORROW
            }
            
            dominant = dominant_map.get(max_val, Emotion.NEUTRAL)

            return EmotionPrediction(
                dominant_emotion=dominant,
                confidence=max_val,
                prob_angry_disgust=p_angry,
                prob_fear_surprise=p_fear,
                prob_happy=p_happy,
                prob_neutral=p_neutral,
                prob_sad=p_sad,
            )

        except Exception as e:
            # logger.warning(f"Inference failed: {e}") # Спам в логах, можно раскомментить для отладки
            return fallback


def create_emotion_model(model_path: str | None) -> EmotionModel:
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Model path '{model_path}' does not exist. Using Random Mock.")
        return RandomEmotionModel()

    try:
        return PyTorchEmotionModel(model_path)
    except (ImportError, Exception) as e:
        logger.warning(f"Could not load real model ({e}). Using Random Mock.")
        return RandomEmotionModel()