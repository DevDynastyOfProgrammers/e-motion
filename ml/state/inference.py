import os
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger

from ml.state.core.preset_mapping import PresetMapping
from ml.state.model.classifier import AdvancedEmotionClassifier, AdvancedPresetAnalyzer

@dataclass
class StatePrediction:
    """Prediction result from the State Model intended for the game engine."""
    preset_name: str
    confidence: float
    multipliers: Dict[str, float]

class StateDirector:
    """
    Interface class for using the 'Game Director' model.
    Loads weights (prototypes) and converts emotion probabilities into game multipliers.
    """

    def __init__(self, prototypes_path: str):
        self.classifier: Optional[AdvancedEmotionClassifier] = None
        self._load_model(prototypes_path)

    def _load_model(self, path: str) -> None:
        """Loads weights (prototypes) from a .npy file."""
        if not os.path.exists(path):
            logger.warning(f"⚠️ StateDirector: Prototypes file not found at '{path}'. Game will use defaults.")
            return

        try:
            logger.info(f"📂 StateDirector: Loading prototypes from {path}...")
            
            # 1. Load prototype dictionary from .npy
            # allow_pickle=True is required because it contains dictionaries
            loaded_prototypes = np.load(path, allow_pickle=True).item()
            
            # 2. Initialize Analyzer
            analyzer = AdvancedPresetAnalyzer()
            analyzer.preset_prototypes = loaded_prototypes
            
            # 3. Initialize Classifier
            self.classifier = AdvancedEmotionClassifier(analyzer)
            
            count = len(loaded_prototypes)
            logger.success(f"✅ StateDirector initialized. Loaded {count} prototypes.")
            
        except Exception as e:
            logger.error(f"❌ StateDirector: Failed to load model: {e}")
            self.classifier = None

    def predict(self, emotion_probs: Dict[str, float], confidence: float = 1.0) -> StatePrediction:
        """
        Main inference method.
        """
        # 1. Prepare input vector
        input_vector = self._build_input_vector(emotion_probs, confidence)

        # 2. Inference
        preset_name = "standard"
        pred_conf = 0.0

        if self.classifier:
            try:
                preset_name, pred_conf, _ = self.classifier.predict(input_vector)
                
                # Log only if confidence is high, or use trace for per-frame debug
                # logger.trace(f"State Inference: {preset_name} ({pred_conf:.2f})") 
            except Exception as e:
                logger.error(f"StateDirector Inference Error: {e}")

        # 3. Convert to multipliers
        multipliers_list = PresetMapping.get_preset_multipliers(preset_name)
        
        multipliers_dict = {
            "spawn_rate_multiplier": multipliers_list[0],
            "enemy_speed_multiplier": multipliers_list[1],
            "enemy_health_multiplier": multipliers_list[2],
            "enemy_damage_multiplier": multipliers_list[3],
            "player_speed_multiplier": multipliers_list[4],
            "player_damage_multiplier": multipliers_list[5],
            "item_drop_chance_modifier": multipliers_list[6]
        }
        
        # 4. Convert offsets to multipliers
        final_multipliers = {}
        for k, v in multipliers_dict.items():
            final_val = 1.0 + v
            # Safety limits (Clamping)
            if final_val < 0.1: final_val = 0.1
            if final_val > 5.0: final_val = 5.0
            final_multipliers[k] = final_val

        return StatePrediction(
            preset_name=preset_name,
            confidence=pred_conf,
            multipliers=final_multipliers
        )

    def _build_input_vector(self, probs: Dict[str, float], confidence: float) -> np.ndarray:
        """Assembles a Numpy vector in strict order."""
        keys_map = {
            'angry_disgust': ['prob_angry_disgust', 'angry_disgust', 'angry'],
            'fear_surprise': ['prob_fear_surprise', 'fear_surprise', 'fear'],
            'happy': ['prob_happy', 'happy', 'joy'],
            'neutral': ['prob_neutral', 'neutral'],
            'sad': ['prob_sad', 'sad', 'sorrow']
        }
        
        vector = [confidence]
        
        for emotion_key in ['angry_disgust', 'fear_surprise', 'happy', 'neutral', 'sad']:
            val = 0.0
            possible_keys = keys_map[emotion_key]
            for key in possible_keys:
                if key in probs:
                    val = probs[key]
                    break
            vector.append(float(val))
            
        return np.array(vector, dtype=np.float32)