import os
import numpy as np
from typing import Protocol, runtime_checkable, Optional
from loguru import logger

from core.emotion import Emotion, EmotionPrediction
from ml.vision.inference import EmotionRecognizer

@runtime_checkable
class EmotionModel(Protocol):
    """Protocol for any emotion recognition model (Mock or Real)."""
    def predict(self, frame: np.ndarray) -> EmotionPrediction: ...


class RandomEmotionModel:
    """Fallback model that returns random probabilities."""
    def predict(self, frame: np.ndarray) -> EmotionPrediction:
        # Mocking random output
        probs = np.random.dirichlet(np.ones(5), size=1)[0]
        idx = np.argmax(probs)
        
        map_idx_to_enum = {
            0: Emotion.ANGER, 1: Emotion.FEAR, 2: Emotion.JOY,
            3: Emotion.NEUTRAL, 4: Emotion.SORROW
        }
        
        return EmotionPrediction(
            dominant_emotion=map_idx_to_enum.get(idx, Emotion.NEUTRAL),
            confidence=float(probs[idx]),
            prob_angry_disgust=float(probs[0]),
            prob_fear_surprise=float(probs[1]),
            prob_happy=float(probs[2]),
            prob_neutral=float(probs[3]),
            prob_sad=float(probs[4]),
        )


class PyTorchEmotionModel:
    """
    Wrapper around ml.vision.inference.EmotionRecognizer.
    Adapts the Dictionary output to the Game's EmotionPrediction DataClass.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        logger.info(f"Initializing PyTorchEmotionModel from {model_path}...")
        try:
            self.recognizer = EmotionRecognizer(model_path, device=device)
            logger.success("EmotionRecognizer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize internal EmotionRecognizer: {e}")
            raise e

        # Mapping string class names (from config.py) to Game Enums
        # Keys must match values in ml.vision.config.EMOTION_CLASSES
        self._str_to_enum = {
            'Angry_Disgust': Emotion.ANGER,
            'Fear_Surprise': Emotion.FEAR,
            'Happy': Emotion.JOY,
            'Sad': Emotion.SORROW,
            'Neutral': Emotion.NEUTRAL
        }

    def predict(self, frame: np.ndarray) -> EmotionPrediction:
        fallback = EmotionPrediction(Emotion.NEUTRAL, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        
        if frame is None:
            return fallback

        try:
            # Delegate to the specialized Inference class
            result = self.recognizer.predict(frame)
            
            # Convert string label to Enum
            predicted_class_str = result.get('predicted_class', 'Neutral')
            dominant_enum = self._str_to_enum.get(predicted_class_str, Emotion.NEUTRAL)
            
            return EmotionPrediction(
                dominant_emotion=dominant_enum,
                confidence=result.get('confidence', 0.0),
                prob_angry_disgust=result.get('prob_angry_disgust', 0.0),
                prob_fear_surprise=result.get('prob_fear_surprise', 0.0),
                prob_happy=result.get('prob_happy', 0.0),
                prob_neutral=result.get('prob_neutral', 0.0),
                prob_sad=result.get('prob_sad', 0.0)
            )
            
        except Exception as e:
            # logger.warning(f"Inference Error: {e}") 
            return fallback


def create_emotion_model(model_path: str | None) -> EmotionModel:
    """Factory method to create the best available model."""
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"⚠️ Model not found at '{model_path}'. Using Random Mock.")
        return RandomEmotionModel()

    try:
        return PyTorchEmotionModel(model_path)
    except Exception as e:
        logger.error(f"❌ Error loading real model ({e}). Fallback to Mock.")
        return RandomEmotionModel()