from typing import Dict, Final, List

# Image Parameters
IMG_SIZE: Final[int] = 64
# ImageNet Standard Normalization (RGB)
# Use these values for pre-processing in inference to match training
NORMALIZATION_MEAN: Final[list[float]] = [0.485, 0.456, 0.406]
NORMALIZATION_STD: Final[list[float]] = [0.229, 0.224, 0.225]

# Model Configuration
# 0: Angry_Disgust, 1: Fear_Surprise, 2: Happy, 3: Sad, 4: Neutral
# Make sure this order matches exactly the alphabetical order of folders used during training!
EMOTION_CLASSES: Final[dict[int, str]] = {0: 'Angry_Disgust', 1: 'Fear_Surprise', 2: 'Happy', 3: 'Sad', 4: 'Neutral'}

DEFAULT_DEVICE: Final[str] = 'cpu'  # Force CPU for inference to save GPU for rendering if needed
