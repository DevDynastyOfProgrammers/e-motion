import random
import time
import numpy as np
from loguru import logger
from core.emotion import Emotion
from core.director import GameDirector
from core.event_manager import EventManager
from core.events import EmotionStateChangedEvent
from core.ml.wrapper import create_emotion_model, EmotionModel, RandomEmotionModel
from settings import EMOTION_MODEL_PATH

try:
    import cv2
except ImportError:
    cv2 = None

class EmotionRecognitionSystem:
    """
    Captures video from webcam, runs inference via EmotionModel,
    and broadcasts EmotionStateChangedEvent.
    """
    RECOGNITION_INTERVAL = 1.0 # seconds

    def __init__(self, event_manager: EventManager) -> None:
        self.event_manager = event_manager
        self._time_since_last_recognition = 0.0
        
        logger.info("Initializing Emotion Recognition System...")
        
        # 1. Initialize Model
        # This is fast because create_emotion_model checks file existence first
        self.model: EmotionModel = create_emotion_model(EMOTION_MODEL_PATH)
        
        # 2. Initialize Camera ONLY if we have a real model and cv2 is installed
        self.cap = None
        
        if isinstance(self.model, RandomEmotionModel):
            logger.info("Using RandomMock model. Webcam will NOT be initialized.")
            return
        elif cv2 is None:
            logger.warning("OpenCV not installed. Webcam will NOT be initialized.")
            return
        
        logger.info("Real model loaded. Attempting to open webcam...")
        start_time = time.time()
        try:
            # Use CAP_DSHOW on Windows for faster startup if possible, else default
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
            if not self.cap.isOpened():
                # Fallback to default backend if DSHOW fails
                self.cap = cv2.VideoCapture(0)
            
            if self.cap.isOpened():
                elapsed = time.time() - start_time
                logger.success(f"Webcam initialized in {elapsed:.2f}s")
            else:
                logger.warning("Could not open webcam (index 0).")
                self.cap = None
        except Exception as e:
            logger.warning(f"OpenCV error during init: {e}")

    def __del__(self) -> None:
        if self.cap:
            self.cap.release()

    def update(self, delta_time: float) -> None:
        self._time_since_last_recognition += delta_time
        if self._time_since_last_recognition >= self.RECOGNITION_INTERVAL:
            self._time_since_last_recognition = 0.0
            
            new_emotion = Emotion.NEUTRAL
            
            # 1. Capture Frame (Only if camera exists)
            frame = None 
            if self.cap:
                ret, raw_frame = self.cap.read()
                if ret:
                    frame = raw_frame
            
            # 2. Predict
            if frame is not None:
                new_emotion = self.model.predict(frame)
            else:
                # If no camera, model handles None (usually returns Mock/Neutral)
                # Or we explicitly fallback to mock behavior here if needed
                new_emotion = self.model.predict(np.zeros((1, 1)))

            # 3. Broadcast
            # logger.debug(f"Detected emotion: {new_emotion.name}")
            self.event_manager.post(EmotionStateChangedEvent(new_emotion))


class GameplayMappingSystem:
    """
    Simulates an ML model that maps an emotional state to a gameplay vector.
    """

    MAPPING_INTERVAL = 5.0  # seconds

    def __init__(self, event_manager: EventManager, director: GameDirector) -> None:
        self.event_manager = event_manager
        self.director = director
        self._time_since_last_mapping = 0.0
        self._current_emotion: Emotion = Emotion.NEUTRAL

        self._emotion_to_vector_map = {
            Emotion.NEUTRAL: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            Emotion.JOY: [0.8, 0.9, 0.8, 0.8, 1.3, 1.2, 2.0],
            Emotion.ANGER: [2.5, 1.5, 1.2, 1.5, 1.1, 1.5, 0.8],
            Emotion.SORROW: [0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0],
            Emotion.FEAR: [3.0, 1.8, 1.0, 1.2, 1.2, 1.0, 1.2],
        }

        self.event_manager.subscribe(EmotionStateChangedEvent, self._on_emotion_changed)

    def _on_emotion_changed(self, event: EmotionStateChangedEvent) -> None:
        self._current_emotion = event.emotion

    def update(self, delta_time: float) -> None:
        self._time_since_last_mapping += delta_time
        if self._time_since_last_mapping >= self.MAPPING_INTERVAL:
            self._time_since_last_mapping = 0.0

            target_vector = self._emotion_to_vector_map.get(
                self._current_emotion, 
                self._emotion_to_vector_map[Emotion.NEUTRAL]
            )
            
            logger.info(f"[MAPPING] Emotion '{self._current_emotion.name}' -> Updating Game State")
            self.director.set_new_target_vector(target_vector)

    def get_current_emotion_name(self) -> str:
        return self._current_emotion.name

    def get_time_to_next_mapping(self) -> float:
        return self.MAPPING_INTERVAL - self._time_since_last_mapping