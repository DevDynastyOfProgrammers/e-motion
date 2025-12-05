import os
import queue
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.director import GameDirector
from core.ecs.systems.biofeedback import BiofeedbackSystem
from core.emotion import Emotion, EmotionPrediction
from core.event_manager import EventManager
from ml.state.inference import StatePrediction


class TestFullIntegration(unittest.TestCase):
    def setUp(self):
        self.event_manager = EventManager()
        # Use a more realistic factor (e.g., 5.0)
        # With dt=0.1 the factor will be 0.5 (halfway to target per frame)
        self.director = GameDirector(smoothing_factor=5.0)

        # Initialize system and immediately stop worker to prevent threading noise during tests
        self.system = BiofeedbackSystem(self.event_manager, self.director)
        self.system.shutdown()

        # Mock the queue to manually feed data
        self.system.result_queue = queue.Queue()

    def test_data_flow_from_ml_to_director(self):
        """Test data flow from ML queue to multiplier updates in Director."""

        # 1. Create fake data (simulating Neural Network output)
        fake_emotion = EmotionPrediction(
            dominant_emotion=Emotion.ANGER,
            confidence=0.95,
            prob_angry_disgust=0.8,
            prob_fear_surprise=0.0,
            prob_happy=0.0,
            prob_neutral=0.1,
            prob_sad=0.1,
        )

        # Fake response from State Director
        fake_state_pred = StatePrediction(
            preset_name='hardcore',
            confidence=0.8,
            multipliers={
                'spawn_rate_multiplier': 2.5,  # Significantly higher than 1.0
                'enemy_speed_multiplier': 1.5,
            },
        )

        # 2. Put into queue (simulate worker thread output)
        # Added 'frame': None because BiofeedbackSystem expects it
        self.system.result_queue.put({'emotion': fake_emotion, 'state': fake_state_pred, 'frame': None})

        # 3. Check initial Director state
        self.assertEqual(self.director.state.spawn_rate_multiplier, 1.0)

        # 4. Execute update (Main Thread logic)
        self.system.update(0.016)  # Simulate 1 frame (approx. 60 FPS)

        # 5. Check Target State (should update instantly)
        self.assertEqual(self.director.target_state.spawn_rate_multiplier, 2.5)
        self.assertEqual(self.director.target_state.enemy_speed_multiplier, 1.5)
        # Parameter missing from dict should remain default 1.0
        self.assertEqual(self.director.target_state.player_speed_multiplier, 1.0)

        # 6. Check interpolation
        # Update director, it should shift current state towards target
        self.director.update(0.1)

        current_spawn = self.director.state.spawn_rate_multiplier
        self.assertGreater(current_spawn, 1.0)
        self.assertLess(current_spawn, 2.6)  # Should not overshoot

        print(f'Integration Success: Spawn Rate moved from 1.0 -> {current_spawn:.2f} (Target: 2.5)')

    def tearDown(self):
        self.system.shutdown()


if __name__ == '__main__':
    unittest.main()
