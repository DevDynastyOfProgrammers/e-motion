"""
Runtime configuration for the Emotion State Model.
"""

from typing import Final, List

# Features expected by the model
# 1. Confidence
# 2-6. Basic Emotions
# 7+. Advanced Features (calculated at runtime)
BASIC_EMOTION_COLUMNS: Final[list[str]] = [
    'prob_angry_disgust',
    'prob_fear_surprise',
    'prob_happy',
    'prob_neutral',
    'prob_sad',
]

# Math Constants
EPSILON: Final[float] = 1e-8
CONFIDENCE_WEIGHT: Final[float] = 0.3
SIMILARITY_THRESHOLD: Final[float] = 0.7

# Model weights for distance calculation (Euclidean vs Cosine)
WEIGHT_EUCLIDEAN: Final[float] = 0.7
WEIGHT_COSINE: Final[float] = 0.3
