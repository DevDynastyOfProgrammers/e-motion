import os
import random
import numpy as np
from typing import Protocol, runtime_checkable
from loguru import logger
from core.emotion import Emotion

# --- Interface ---

@runtime_checkable
class EmotionModel(Protocol):
    """Protocol for any emotion recognition model (Mock or Real)."""
    def predict(self, frame: np.ndarray) -> Emotion:
        ...

# --- Implementations ---

class RandomEmotionModel:
    """Fallback model that returns random emotions. Used when PyTorch is missing."""
    def predict(self, frame: np.ndarray) -> Emotion:
        # Frame is ignored
        return random.choice(list(Emotion))

class PyTorchEmotionModel:
    """Real model wrapper that uses a .pt file for inference."""
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model_path = model_path
        
        # Lazy imports to avoid crashing if torch is not installed
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
        
        # Load the model (entire model or state_dict)
        try:
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise e

        # TODO: Adjust transforms based on model's expected input
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((48, 48)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            # T.Normalize(mean=[0.5], std=[0.5]) 
        ])

        # TODO: Compare with model's expected output classes
        # Define mapping from index to Emotion Enum
        self.label_map = {
            0: Emotion.NEUTRAL,
            1: Emotion.JOY,
            2: Emotion.ANGER,
            3: Emotion.SORROW,
            4: Emotion.FEAR
        }
        logger.success("ML Model loaded successfully.")

    def predict(self, frame: np.ndarray) -> Emotion:
        if frame is None or frame.size == 0:
            return Emotion.NEUTRAL

        try:
            # Preprocess
            input_tensor = self.transforms(frame).unsqueeze(0).to(self.device)
            
            # Inference
            with self.torch.no_grad():
                outputs = self.model(input_tensor)
                _, predicted_idx = self.torch.max(outputs, 1)
                idx = int(predicted_idx.item())
            
            return self.label_map.get(idx, Emotion.NEUTRAL)
            
        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            return Emotion.NEUTRAL

# --- Factory ---

def create_emotion_model(model_path: str | None) -> EmotionModel:
    """
    Factory that attempts to create a Real model, falling back to Random if fails.
    """
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Model path '{model_path}' does not exist. Using Random Mock.")
        return RandomEmotionModel()

    try:
        return PyTorchEmotionModel(model_path)
    except (ImportError, Exception) as e:
        logger.warning(f"Could not load real model ({e}). Using Random Mock.")
        return RandomEmotionModel()