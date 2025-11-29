import os
import numpy as np
from typing import Protocol, runtime_checkable
from loguru import logger
from core.emotion import Emotion, EmotionPrediction


@runtime_checkable
class EmotionModel(Protocol):
    """Protocol for any emotion recognition model (Mock or Real)."""

    def predict(self, frame: np.ndarray) -> EmotionPrediction: ...


class RandomEmotionModel:
    """Fallback model that returns random probabilities."""

    def predict(self, frame: np.ndarray) -> EmotionPrediction:
        # Simulate a probability distribution
        raw_probs = np.random.dirichlet(np.ones(5), size=1)[0]

        # Map to our dataclass fields (Arbitrary order for mock)
        # 0: angry_disgust, 1: fear_surprise, 2: happy, 3: neutral, 4: sad

        # Find dominant to match our Game Enum
        idx = np.argmax(raw_probs)
        dominant_map = {
            0: Emotion.ANGER,
            1: Emotion.FEAR,
            2: Emotion.JOY,
            3: Emotion.NEUTRAL,
            4: Emotion.SORROW,
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
    """Real model wrapper that uses a .pt file for inference."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model_path = model_path

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
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise e

        # IMPORTANT: Kolya needs to confirm these transforms!
        self.transforms = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((48, 48)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )

    def predict(self, frame: np.ndarray) -> EmotionPrediction:
        # Default fallback
        fallback = EmotionPrediction(Emotion.NEUTRAL, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        if frame is None or frame.size == 0:
            return fallback

        try:
            # Preprocess
            input_tensor = self.transforms(frame).unsqueeze(0).to(self.device)

            # Inference
            with self.torch.no_grad():
                outputs = self.model(input_tensor)
                # Assuming outputs are logits, apply softmax to get probabilities
                probs = self.torch.softmax(outputs, dim=1)[0]

            # Map output tensor indices to specific emotion fields.
            # WARNING: This index mapping MUST match emotion model's training order.
            # Assuming alphabetical order of groups or specific training order (example):
            # 0: angry_disgust
            # 1: fear_surprise
            # 2: happy
            # 3: neutral
            # 4: sad

            p_angry = float(probs[0].item())
            p_fear = float(probs[1].item())
            p_happy = float(probs[2].item())
            p_neutral = float(probs[3].item())
            p_sad = float(probs[4].item())

            # Determine dominant for Game Director (Visualization)
            val_list = [p_angry, p_fear, p_happy, p_neutral, p_sad]
            max_val = max(val_list)
            max_idx = val_list.index(max_val)

            dominant_map = {
                0: Emotion.ANGER,
                1: Emotion.FEAR,
                2: Emotion.JOY,
                3: Emotion.NEUTRAL,
                4: Emotion.SORROW,
            }

            return EmotionPrediction(
                dominant_emotion=dominant_map.get(max_idx, Emotion.NEUTRAL),
                confidence=max_val,
                prob_angry_disgust=p_angry,
                prob_fear_surprise=p_fear,
                prob_happy=p_happy,
                prob_neutral=p_neutral,
                prob_sad=p_sad,
            )

        except Exception as e:
            logger.warning(f"Inference failed: {e}")
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
