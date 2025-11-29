import os
import numpy as np
import logging
from typing import Dict, Any
from dataclasses import dataclass

from ml.state.core.preset_mapping import PresetMapping
from ml.state.model.classifier import AdvancedEmotionClassifier
from ml.state.model.analyzer import AdvancedPresetAnalyzer

# Настраиваем логгер для этого модуля
logger = logging.getLogger("ml.state")

@dataclass
class StatePrediction:
    """Результат работы State Model для игрового движка."""
    preset_name: str
    confidence: float
    multipliers: Dict[str, float]

class StateDirector:
    """
    Класс-интерфейс для использования модели 'Game Director'.
    Загружает веса при инициализации и предоставляет метод predict().
    """

    def __init__(self, prototypes_path: str):
        self.classifier = None
        self._load_model(prototypes_path)

    def _load_model(self, path: str) -> None:
        """Загрузка весов (прототипов) из .npy файла."""
        if not os.path.exists(path):
            logger.warning(f"⚠️ Prototypes file not found at: {path}. Using fallback defaults.")
            return

        try:
            # Инициализация анализатора и загрузка весов
            analyzer = AdvancedPresetAnalyzer()
            
            # allow_pickle=True нужен, так как мы сохраняем словари/объекты numpy
            loaded_prototypes = np.load(path, allow_pickle=True).item()
            analyzer.preset_prototypes = loaded_prototypes
            
            # Создаем классификатор на основе загруженных прототипов
            self.classifier = AdvancedEmotionClassifier(analyzer)
            logger.success(f"StateDirector model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load StateDirector model: {e}")
            self.classifier = None

    def predict(self, emotion_probs: Dict[str, float], confidence: float = 1.0) -> StatePrediction:
        """
        Принимает вероятности эмоций и возвращает игровые множители.
        
        Args:
            emotion_probs: словарь вида {'angry_disgust': 0.1, ...} или {'prob_angry...': 0.1}
            confidence: уровень уверенности Vision модели (0.0 - 1.0)
        """
        # 1. Подготовка вектора признаков для модели
        # Ожидаемый порядок в classifier.py: [confidence, angry, fear, happy, neutral, sad]
        
        # Карта нормализации ключей (на случай если Vision отдает без префикса prob_)
        keys_map = {
            'angry_disgust': 'prob_angry_disgust',
            'fear_surprise': 'prob_fear_surprise',
            'happy': 'prob_happy',
            'neutral': 'prob_neutral',
            'sad': 'prob_sad'
        }

        # Собираем вектор
        input_vector = [confidence]
        for key in ['angry_disgust', 'fear_surprise', 'happy', 'neutral', 'sad']:
            # Ищем ключ с префиксом или без
            lookup_key = keys_map[key]
            # Если ключа нет, берем 0.0 (или пробуем искать без префикса)
            val = emotion_probs.get(lookup_key, emotion_probs.get(key, 0.0))
            input_vector.append(val)
        
        input_np = np.array(input_vector)

        # 2. Инференс
        preset_name = "standard"
        pred_conf = 0.0

        if self.classifier:
            try:
                # classifier возвращает: (preset_name, confidence_score, details_dict)
                preset_name, pred_conf, _ = self.classifier.predict(input_np)
            except Exception as e:
                logger.error(f"Inference error: {e}. Falling back to 'standard'.")

        # 3. Маппинг Пресета в Числовые Множители
        # Получаем список значений из PresetMapping
        multipliers_list = PresetMapping.get_preset_multipliers(preset_name)
        
        # PresetParameters содержит поля в определенном порядке:
        # spawn_rate, enemy_speed, enemy_health, enemy_damage, player_speed, player_damage, item_drop
        
        multipliers_dict = {
            "spawn_rate_multiplier": multipliers_list[0],
            "enemy_speed_multiplier": multipliers_list[1],
            "enemy_health_multiplier": multipliers_list[2],
            "enemy_damage_multiplier": multipliers_list[3],
            "player_speed_multiplier": multipliers_list[4],
            "player_damage_multiplier": multipliers_list[5],
            "item_drop_chance_modifier": multipliers_list[6]
        }

        # 4. Преобразование offset -> multiplier
        # В PresetMapping значения часто смещения (0.2 = +20%, -0.5 = -50%).
        # Движку нужны множители (1.2 и 0.5 соответственно).
        for k, v in multipliers_dict.items():
            multipliers_dict[k] = 1.0 + v
            # Защита от отрицательных значений (кроме специальных случаев)
            if multipliers_dict[k] < 0.1: 
                multipliers_dict[k] = 0.1

        return StatePrediction(
            preset_name=preset_name,
            confidence=pred_conf,
            multipliers=multipliers_dict
        )