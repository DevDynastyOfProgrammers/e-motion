import unittest
import queue
import time
from unittest.mock import MagicMock

# Фиктивные импорты для теста
from core.director import GameDirector
from core.event_manager import EventManager
from core.emotion import Emotion, EmotionPrediction
from core.ecs.systems.biofeedback import BiofeedbackSystem
from ml.state.inference import StatePrediction

class TestFullIntegration(unittest.TestCase):
    def setUp(self):
        self.event_manager = EventManager()
        # Используем более реалистичный фактор (например, 5.0)
        # При dt=0.1 фактор будет 0.5 (половина пути к цели за кадр)
        self.director = GameDirector(smoothing_factor=5.0) 
        
        self.system = BiofeedbackSystem(self.event_manager, self.director)
        self.system.shutdown()
        
        self.system.result_queue = queue.Queue()

    def test_data_flow_from_ml_to_director(self):
        """Тест прохождения данных от ML-очереди до изменения множителей в Director."""
        
        # 1. Создаем фейковые данные (как будто от нейросети)
        fake_emotion = EmotionPrediction(
            dominant_emotion=Emotion.ANGER,
            confidence=0.95,
            prob_angry_disgust=0.8,
            prob_fear_surprise=0.0,
            prob_happy=0.0,
            prob_neutral=0.1,
            prob_sad=0.1
        )
        
        # Фейковый ответ от State Director
        fake_state_pred = StatePrediction(
            preset_name="hardcore",
            confidence=0.8,
            multipliers={
                "spawn_rate_multiplier": 2.5,  # Значительно больше 1.0
                "enemy_speed_multiplier": 1.5
            }
        )

        # 2. Кладем в очередь (симуляция работы потока)
        self.system.result_queue.put({
            'emotion': fake_emotion,
            'state': fake_state_pred
        })

        # 3. Проверяем начальное состояние Директора
        self.assertEqual(self.director.state.spawn_rate_multiplier, 1.0)

        # 4. Выполняем update (Main Thread logic)
        self.system.update(0.016) # 1 кадр

        # 5. Проверяем Target State (должен мгновенно обновиться)
        self.assertEqual(self.director.target_state.spawn_rate_multiplier, 2.5)
        self.assertEqual(self.director.target_state.enemy_speed_multiplier, 1.5)
        # Параметр, которого не было в словаре, должен остаться 1.0
        self.assertEqual(self.director.target_state.player_speed_multiplier, 1.0)

        # 6. Проверяем интерполяцию
        # Делаем update директора, он должен сдвинуть current state в сторону target
        self.director.update(0.1)
        
        current_spawn = self.director.state.spawn_rate_multiplier
        self.assertGreater(current_spawn, 1.0)
        self.assertLess(current_spawn, 2.6) # Не должен перелететь

        print(f"Integration Success: Spawn Rate moved from 1.0 -> {current_spawn:.2f} (Target: 2.5)")

    def tearDown(self):
        self.system.shutdown()

if __name__ == '__main__':
    unittest.main()