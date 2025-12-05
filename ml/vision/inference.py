from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.vision.config import EMOTION_CLASSES, IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD
from ml.vision.models import EmotionCNN
from ml.vision.utils import get_device, load_checkpoint


class EmotionRecognizer:
    """
    High-level Interface for Emotion Recognition Inference.
    Handles image preprocessing (BGR -> RGB, Resize, Normalize) and model execution.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = get_device(device)

        # Initialize Architecture
        num_classes = len(EMOTION_CLASSES)
        self.model = EmotionCNN(num_classes=num_classes)

        # Load Weights
        self.model = load_checkpoint(self.model, model_path, self.device)

        # Define Transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
            ]
        )

    def predict(self, frame: np.ndarray) -> dict[str, str | float]:
        """
        Predicts emotion from a raw OpenCV frame (BGR).
        Returns a dictionary with probabilities and the dominant class.
        """
        # 1. Preprocess: BGR (OpenCV) -> RGB (PIL)
        # OpenCV frames are numpy arrays in BGR format
        if frame is None or frame.size == 0:
            raise ValueError('Empty frame provided for inference')

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 2. Transform & Batch
        tensor = self.transform(pil_img)
        tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        # 4. Format Result
        result = {}
        probs_cpu = probs[0].cpu().numpy()

        # Add individual probabilities
        for idx, emotion_name in EMOTION_CLASSES.items():
            key_name = f'prob_{emotion_name.lower()}'  # e.g. prob_angry_disgust
            result[key_name] = float(probs_cpu[idx])

        # Find dominant
        top_idx = int(torch.argmax(probs, dim=1).item())
        top_prob = float(probs_cpu[top_idx])

        result['predicted_class'] = EMOTION_CLASSES[top_idx]
        result['confidence'] = top_prob

        return result
