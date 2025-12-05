import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from core.director import GameDirector, GameStateVector
from core.emotion import Emotion, EmotionPrediction
from core.event_manager import EventManager
from core.events import EmotionStateChangedEvent
from ml.state.inference import StateDirector, StatePrediction

# ML Imports (We assume wrapper interface will be finalized in Block 2)
from ml.wrapper import EmotionModel, RandomEmotionModel, create_emotion_model
from settings import EMOTION_MODEL_PATH, STATE_PROTOTYPES_PATH


class BiofeedbackWorker(threading.Thread):
    """
    Dedicated worker thread for Heavy ML Inference.
    Captures camera, runs Vision Model, runs State Model, and puts results in Queue.
    """

    # Constants for configuration
    EMA_ALPHA: float = 0.2
    TARGET_FPS: float = 5.0
    DEBUG_WIDTH: int = 160
    DEBUG_HEIGHT: int = 120

    def __init__(self, result_queue: queue.Queue, stop_event: threading.Event) -> None:
        super().__init__(name='BiofeedbackWorker', daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event

        # Resources
        self.vision_model: EmotionModel | None = None
        self.state_director: StateDirector | None = None
        self.cap: cv2.VideoCapture | None = None

        # State Persistence (for smoothing)
        self.current_probs: dict[str, float] = {
            'angry_disgust': 0.0,
            'fear_surprise': 0.0,
            'happy': 0.0,
            'neutral': 1.0,
            'sad': 0.0,
        }
        self.last_preset: str = ''

    def run(self) -> None:
        """Main thread loop."""
        try:
            # Prevent PyTorch from hogging all CPU cores (leaves room for PyGame)
            torch.set_num_threads(1)

            logger.info('🧵 Biofeedback worker initialized.')
            self._init_resources()

            frame_interval = 1.0 / self.TARGET_FPS

            while not self.stop_event.is_set():
                loop_start = time.perf_counter()

                try:
                    self._process_single_cycle()
                except Exception as e:
                    logger.error(f'⚠️ Error inside biofeedback loop: {e}')
                    # Throttle error loop to prevent log spam
                    time.sleep(1.0)

                # Maintain Target FPS
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, frame_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            logger.critical(f'🔥 BiofeedbackWorker CRASHED: {e}')
        finally:
            self._cleanup()

    def _init_resources(self) -> None:
        """Safe loading of models and camera."""
        # 1. Vision Model
        try:
            self.vision_model = create_emotion_model(EMOTION_MODEL_PATH)
            logger.info(f'Vision Model: {type(self.vision_model).__name__}')
        except Exception as e:
            logger.error(f'Failed to create vision model: {e}')
            self.vision_model = RandomEmotionModel()

        # 2. State Director
        try:
            self.state_director = StateDirector(STATE_PROTOTYPES_PATH)
        except Exception as e:
            logger.error(f'Failed to create StateDirector: {e}')

        # 3. Camera (Only if we have a real model, otherwise no point capturing)
        if not isinstance(self.vision_model, RandomEmotionModel):
            self._open_camera()

    def _open_camera(self) -> None:
        """Attempts to open the webcam using multiple backends."""
        logger.debug('Opening webcam...')
        # Try default backend first
        self.cap = cv2.VideoCapture(0)

        # Fallback to DirectShow on Windows if default fails
        if not self.cap or not self.cap.isOpened():
            logger.warning('VideoCapture(0) failed. Trying CAP_DSHOW...')
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if self.cap and self.cap.isOpened():
            logger.success('Webcam initialized.')
        else:
            logger.warning('❌ Could not open webcam. Running on black frames.')

    def _process_single_cycle(self) -> None:
        """One iteration of Capture -> Predict -> Smooth -> Enqueue."""
        # 1. Capture
        frame = self._capture_frame()

        # 2. Vision Inference
        prediction = self._run_vision_inference(frame)

        # 3. Prepare Debug Frame (Resize for UI)
        debug_frame = self._prepare_debug_frame(frame)

        # 4. Smoothing (EMA)
        smoothed_prediction = self._apply_ema_smoothing(prediction)

        # 5. State Director (Game Logic)
        state_result = self._run_state_inference(smoothed_prediction)

        # 6. Send to Main Thread
        payload = {'emotion': smoothed_prediction, 'state': state_result, 'frame': debug_frame}

        try:
            self.result_queue.put_nowait(payload)
        except queue.Full:
            # Drop frame if main thread is too slow
            pass

    def _capture_frame(self) -> np.ndarray | None:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def _run_vision_inference(self, frame: np.ndarray | None) -> EmotionPrediction:
        if self.vision_model:
            return self.vision_model.predict(frame)
        # Fallback empty prediction
        return EmotionPrediction(Emotion.NEUTRAL, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def _prepare_debug_frame(self, frame: np.ndarray | None) -> np.ndarray | None:
        if frame is None:
            return None
        try:
            # Resize -> RGB -> Transpose (for PyGame Surface)
            small = cv2.resize(frame, (self.DEBUG_WIDTH, self.DEBUG_HEIGHT))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            return np.transpose(rgb, (1, 0, 2))
        except Exception:
            return None

    def _apply_ema_smoothing(self, pred: EmotionPrediction) -> EmotionPrediction:
        """Applies Exponential Moving Average to probabilities."""
        raw_probs = {
            'angry_disgust': pred.prob_angry_disgust,
            'fear_surprise': pred.prob_fear_surprise,
            'happy': pred.prob_happy,
            'neutral': pred.prob_neutral,
            'sad': pred.prob_sad,
        }

        # Update internal state
        for k, v in raw_probs.items():
            self.current_probs[k] = (self.EMA_ALPHA * v) + ((1.0 - self.EMA_ALPHA) * self.current_probs[k])

        return EmotionPrediction(
            dominant_emotion=pred.dominant_emotion,
            confidence=pred.confidence,
            prob_angry_disgust=self.current_probs['angry_disgust'],
            prob_fear_surprise=self.current_probs['fear_surprise'],
            prob_happy=self.current_probs['happy'],
            prob_neutral=self.current_probs['neutral'],
            prob_sad=self.current_probs['sad'],
        )

    def _run_state_inference(self, emotion_pred: EmotionPrediction) -> StatePrediction:
        if not self.state_director:
            # Return neutral/default state if director failed to load
            return StatePrediction('default', 1.0, {})

        result = self.state_director.predict(self.current_probs, emotion_pred.confidence)

        # Log state changes
        if result.preset_name != self.last_preset:
            logger.info(f'🎭 Director changed state: {self.last_preset} -> {result.preset_name}')
            self.last_preset = result.preset_name

        return result

    def _cleanup(self) -> None:
        if self.cap:
            self.cap.release()
        logger.info('Biofeedback worker stopped.')


class BiofeedbackSystem:
    """
    ECS System that consumes data from the BiofeedbackWorker
    and applies it to the GameDirector.
    Runs on the Main Thread.
    """

    def __init__(self, event_manager: EventManager, director: GameDirector):
        self.event_manager = event_manager
        self.director = director

        self.result_queue: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        self.worker = BiofeedbackWorker(self.result_queue, self.stop_event)
        self.worker.start()

        self.current_debug_frame: np.ndarray | None = None

    def update(self, delta_time: float) -> None:
        try:
            # Non-blocking get
            data = self.result_queue.get_nowait()
            self._apply_game_data(data)
        except queue.Empty:
            pass

    def _apply_game_data(self, data: dict[str, Any]) -> None:
        # 1. Update Emotion Event (for UI/Player Entity)
        emotion_pred: EmotionPrediction = data['emotion']
        self.event_manager.post(EmotionStateChangedEvent(emotion_pred))

        # 2. Update Debug Frame (for UI)
        self.current_debug_frame = data['frame']

        # 3. Update Game Difficulty
        state_pred: StatePrediction = data['state']
        mults = state_pred.multipliers

        target = GameStateVector(
            spawn_rate_multiplier=mults.get('spawn_rate_multiplier', 1.0),
            enemy_speed_multiplier=mults.get('enemy_speed_multiplier', 1.0),
            enemy_health_multiplier=mults.get('enemy_health_multiplier', 1.0),
            enemy_damage_multiplier=mults.get('enemy_damage_multiplier', 1.0),
            player_speed_multiplier=mults.get('player_speed_multiplier', 1.0),
            player_damage_multiplier=mults.get('player_damage_multiplier', 1.0),
            item_drop_chance_modifier=mults.get('item_drop_chance_modifier', 1.0),
        )
        self.director.set_new_target_vector(target)

    def shutdown(self) -> None:
        """Safely stops the worker thread."""
        logger.info('Shutting down Biofeedback System...')
        self.stop_event.set()
        self.worker.join(timeout=1.0)
