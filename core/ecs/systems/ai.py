import random
from core.emotion import Emotion
from core.director import GameDirector, GameStateVector
from core.event_manager import EventManager
from core.events import EmotionStateChangedEvent

class EmotionRecognitionSystem:
    """
    Simulates an ML model that recognizes the player's emotion.
    Periodically posts an EmotionStateChangedEvent.
    """
    RECOGNITION_INTERVAL = 5.0  # seconds

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self._time_since_last_recognition = 0.0

    def update(self, delta_time: float):
        self._time_since_last_recognition += delta_time
        if self._time_since_last_recognition >= self.RECOGNITION_INTERVAL:
            self._time_since_last_recognition = 0.0
            
            # Simulate the model's output
            new_emotion = random.choice(list(Emotion))
            print(f"[EMOTION_MODEL] Detected new emotion: {new_emotion.name}")
            self.event_manager.post(EmotionStateChangedEvent(new_emotion))


class GameplayMappingSystem:
    """
    Simulates an ML model that maps an emotional state to a gameplay vector.
    """
    MAPPING_INTERVAL = 60.0  # seconds

    def __init__(self, event_manager: EventManager, director: GameDirector):
        self.event_manager = event_manager
        self.director = director
        self._time_since_last_mapping = 0.0
        self._current_emotion: Emotion = Emotion.NEUTRAL

        # Data-driven mapping from emotion to gameplay vector
        self._emotion_to_vector_map = {
            Emotion.NEUTRAL: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            Emotion.JOY:     [0.8, 0.9, 0.8, 0.8, 1.3, 1.2, 2.0],
            Emotion.ANGER:   [2.5, 1.5, 1.2, 1.5, 1.1, 1.5, 0.8],
            Emotion.SORROW:  [0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0],
            Emotion.FEAR:    [3.0, 1.8, 1.0, 1.2, 1.2, 1.0, 1.2],
        }

        self.event_manager.subscribe(EmotionStateChangedEvent, self._on_emotion_changed)

    def _on_emotion_changed(self, event: EmotionStateChangedEvent):
        self._current_emotion = event.emotion
        
    def update(self, delta_time: float):
        self._time_since_last_mapping += delta_time
        if self._time_since_last_mapping >= self.MAPPING_INTERVAL:
            self._time_since_last_mapping = 0.0
            
            target_vector = self._emotion_to_vector_map.get(
                self._current_emotion, 
                self._emotion_to_vector_map[Emotion.NEUTRAL]
            )
            
            print(f"[MAPPING_MODEL] Applying new game state for emotion '{self._current_emotion.name}'")
            self.director.set_new_target_vector(target_vector)

    def get_current_emotion_name(self) -> str:
        return self._current_emotion.name
        
    def get_time_to_next_mapping(self) -> float:
        return self.MAPPING_INTERVAL - self._time_since_last_mapping