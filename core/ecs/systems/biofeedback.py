import threading
import queue
import time
import cv2
import torch
import numpy as np
from typing import Optional, Dict
from loguru import logger

from core.event_manager import EventManager
from core.events import EmotionStateChangedEvent
from core.director import GameDirector, GameStateVector
from core.emotion import Emotion, EmotionPrediction

from ml.wrapper import create_emotion_model, EmotionModel, RandomEmotionModel
from ml.state.inference import StateDirector
from settings import EMOTION_MODEL_PATH, STATE_PROTOTYPES_PATH


class BiofeedbackWorker(threading.Thread):
    def __init__(self, result_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="BiofeedbackWorker", daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        
        self.vision_model: Optional[EmotionModel] = None
        self.state_director: Optional[StateDirector] = None
        self.cap = None

        # Smoothing state
        self.ema_alpha = 0.2
        self.current_probs = {
            'angry_disgust': 0.0, 'fear_surprise': 0.0, 'happy': 0.0,
            'neutral': 1.0, 'sad': 0.0
        }
        self.last_preset = ""

    def run(self):
        try:
            torch.set_num_threads(1) 
            logger.info("🧵 Biofeedback worker thread started.")
            
            # Инициализация с подробными логами
            self._init_resources()
            logger.info("✅ Resources initialized. Starting inference loop.")

            target_fps = 5.0 
            frame_interval = 1.0 / target_fps

            while not self.stop_event.is_set():
                loop_start = time.time()
                
                try:
                    # --- 1. Capture ---
                    frame = None
                    if self.cap and self.cap.isOpened():
                        if self.cap.grab():
                            ret, raw_frame = self.cap.retrieve()
                            if ret:
                                frame = raw_frame
                    
                    # --- 2. Inference ---
                    # (Внутри _run_vision уже есть защита от None)
                    prediction = self._run_vision(frame)
                    
                    # --- 3. Smoothing (EMA) ---
                    raw_probs = {
                        'angry_disgust': prediction.prob_angry_disgust,
                        'fear_surprise': prediction.prob_fear_surprise,
                        'happy': prediction.prob_happy,
                        'neutral': prediction.prob_neutral,
                        'sad': prediction.prob_sad
                    }

                    for key in self.current_probs:
                        raw_val = raw_probs.get(key, 0.0)
                        self.current_probs[key] = (self.ema_alpha * raw_val) + ((1.0 - self.ema_alpha) * self.current_probs[key])

                    smoothed_prediction = EmotionPrediction(
                        dominant_emotion=prediction.dominant_emotion,
                        confidence=prediction.confidence,
                        prob_angry_disgust=self.current_probs['angry_disgust'],
                        prob_fear_surprise=self.current_probs['fear_surprise'],
                        prob_happy=self.current_probs['happy'],
                        prob_neutral=self.current_probs['neutral'],
                        prob_sad=self.current_probs['sad']
                    )

                    # --- 4. State Director Inference ---
                    state_result = self.state_director.predict(self.current_probs, prediction.confidence)

                    if state_result.preset_name != self.last_preset:
                        logger.info(f"🎭 Game State Changed: {self.last_preset} -> {state_result.preset_name}")
                        self.last_preset = state_result.preset_name

                    # --- 5. Send Results ---
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.result_queue.put({
                        'emotion': smoothed_prediction,
                        'state': state_result
                    })

                except Exception as e:
                    logger.error(f"⚠️ Error inside worker loop: {e}")
                    # Не падаем, пробуем продолжить (например, если камера лагнула один кадр)
                    time.sleep(0.5)

                # --- 6. Sleep ---
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.01)

        except Exception as e:
            logger.critical(f"🔥 BiofeedbackWorker CRASHED: {e}")
        finally:
            self._cleanup()

    def _init_resources(self):
        logger.debug("Loading Vision Model...")
        try:
            self.vision_model = create_emotion_model(EMOTION_MODEL_PATH)
            logger.debug(f"Vision Model loaded: {type(self.vision_model)}")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")

        logger.debug("Loading State Director...")
        try:
            self.state_director = StateDirector(STATE_PROTOTYPES_PATH)
            logger.debug("State Director loaded.")
        except Exception as e:
            logger.error(f"Failed to load State Director: {e}")

        if not isinstance(self.vision_model, RandomEmotionModel):
            logger.debug("Opening Camera...")
            try:
                # Попробуем сначала стандартный API, так как DSHOW иногда виснет
                logger.debug("Attempting cv2.VideoCapture(0)...")
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    logger.warning("VideoCapture(0) failed. Trying DSHOW backend...")
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
                if self.cap.isOpened():
                    logger.success("Webcam initialized successfully.")
                else:
                    logger.warning("❌ Could not open webcam. Inference will run on black frames.")
            except Exception as e:
                logger.error(f"Camera init crashed: {e}")

    def _run_vision(self, frame) -> EmotionPrediction:
        if self.vision_model:
            return self.vision_model.predict(frame)
        return EmotionPrediction(Emotion.NEUTRAL, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        logger.info("Biofeedback worker stopped.")


class BiofeedbackSystem:
    def __init__(self, event_manager: EventManager, director: GameDirector):
        self.event_manager = event_manager
        self.director = director
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = BiofeedbackWorker(self.result_queue, self.stop_event)
        self.worker.start()

    def update(self, delta_time: float):
        try:
            data = self.result_queue.get_nowait()
            self._apply_data(data)
        except queue.Empty:
            pass

    def _apply_data(self, data):
        self.event_manager.post(EmotionStateChangedEvent(data['emotion']))
        mults = data['state'].multipliers
        target = GameStateVector(
            spawn_rate_multiplier=mults.get("spawn_rate_multiplier", 1.0),
            enemy_speed_multiplier=mults.get("enemy_speed_multiplier", 1.0),
            enemy_health_multiplier=mults.get("enemy_health_multiplier", 1.0),
            enemy_damage_multiplier=mults.get("enemy_damage_multiplier", 1.0),
            player_speed_multiplier=mults.get("player_speed_multiplier", 1.0),
            player_damage_multiplier=mults.get("player_damage_multiplier", 1.0),
            item_drop_chance_modifier=mults.get("item_drop_chance_modifier", 1.0)
        )
        self.director.set_new_target_vector(target)

    def shutdown(self):
        self.stop_event.set()
        # Таймаут меньше, чтобы игра не висла при выходе, если поток завис
        self.worker.join(timeout=0.5)