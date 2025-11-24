from enum import Enum, auto
from dataclasses import dataclass

class Emotion(Enum):
    """
    Defines the high-level emotional states for Game Logic (Director).
    """
    NEUTRAL = auto()
    JOY = auto()
    ANGER = auto()
    SORROW = auto()
    FEAR = auto()

@dataclass
class EmotionPrediction:
    """
    Raw output from the Vision Model.
    Passed directly to Core Game Director Model.
    """
    dominant_emotion: Emotion
    confidence: float
    
    # Specific probabilities from the model
    prob_angry_disgust: float
    prob_fear_surprise: float
    prob_happy: float
    prob_neutral: float
    prob_sad: float

    def to_vector(self) -> list[float]:
        """Returns probabilities as a list for the regression model."""
        return [
            self.prob_angry_disgust,
            self.prob_fear_surprise,
            self.prob_happy,
            self.prob_neutral,
            self.prob_sad
        ]