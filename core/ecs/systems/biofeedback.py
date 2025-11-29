import threading
import queue
import time
import cv2
import numpy as np
from typing import Optional, Dict
from loguru import logger

from core.event_manager import EventManager
from core.events import EmotionStateChangedEvent
from core.director import GameDirector, GameStateVector
from core.emotion import Emotion, EmotionPrediction

# ML Imports
from ml.wrapper import create_emotion_model, EmotionModel, RandomEmotionModel
from ml.state.inference import StateDirector
from settings import EMOTION_MODEL_PATH, STATE_PROTOTYPES_PATH

class BiofeedbackWorker(threading.Thread):
    """
    Фоновый воркер:
    1. Захватывает кадр веб-камеры.
    2. Vision Model -> Эмоции.
    3. State Director -> Игровой пресет.
    """
    def __init__(self, result_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="BiofeedbackWorker", daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        
        self.vision_model: Optional[EmotionModel] = None
        self.state_director: Optional[StateDirector] = None
        self.cap = None

    def run(self):
        logger.info("Biofeedback worker started.")
        self._init_resources()

        # Ограничение FPS инференса (не нужно гнать на 60 FPS, достаточно 5-10)
        target_inference_fps = 10
        frame_time = 1.0 / target_inference_fps

        while not self.stop_event.is_set():
            start_time = time.time()
            
            # --- 1. Capture ---
            frame = None
            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret:
                    frame = raw_frame
            
            # --- 2. Vision Inference ---
            # Даже если нет камеры, модель вернет заглушку (если это Mock)
            prediction: EmotionPrediction = self._run_vision(frame)
            
            # --- 3. State Inference ---
            # Конвертируем dataclass в dict
            emotion_dict = {
                'angry_disgust': prediction.prob_angry_disgust,
                'fear_surprise': prediction.prob_fear_surprise,
                'happy': prediction.prob_happy,
                'neutral': prediction.prob_neutral,
                'sad': prediction.prob_sad
            }
            
            # Получаем новый игровой стейт
            state_result = self.state_director.predict(emotion_dict, prediction.confidence)

            # --- 4. Отправка результата в Main Thread ---
            try:
                # Если очередь полна, выбрасываем старый результат (нам нужен только свежий)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.result_queue.put({
                    'emotion': prediction,
                    'state': state_result
                })
            except Exception as e:
                logger.error(f"Queue error in worker: {e}")

            # --- 5. Sleep ---
            elapsed = time.time() - start_time
            sleep_time = max(0.01, frame_time - elapsed)
            time.sleep(sleep_time)

        self._cleanup()

    def _init_resources(self):
        # Vision
        self.vision_model = create_emotion_model(EMOTION_MODEL_PATH)
        
        # State
        self.state_director = StateDirector(STATE_PROTOTYPES_PATH)

        # Camera (инициализируем только если модель реальная)
        if not isinstance(self.vision_model, RandomEmotionModel):
            try:
                # DSHOW быстрее на Windows, но нужен фолбек
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
                
                if self.cap.isOpened():
                    logger.success("Webcam initialized.")
                else:
                    logger.warning("Webcam NOT found. Inference will use empty frames.")
            except Exception as e:
                logger.error(f"Camera init failed: {e}")

    def _run_vision(self, frame) -> EmotionPrediction:
        """Безопасный запуск модели зрения."""
        if self.vision_model:
            # Обработка None внутри модели (wrapper.py) уже реализована
            return self.vision_model.predict(frame)
        return EmotionPrediction(Emotion.NEUTRAL, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        logger.info("Biofeedback worker resources released.")


class BiofeedbackSystem:
    """
    ECS-система. Живет в Main Thread.
    Читает очередь результатов из Worker Thread и обновляет GameDirector.
    """
    def __init__(self, event_manager: EventManager, director: GameDirector):
        self.event_manager = event_manager
        self.director = director
        
        # Очередь размером 1, чтобы не накапливать лаг
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        
        self.worker = BiofeedbackWorker(self.result_queue, self.stop_event)
        self.worker.start()

    def update(self, delta_time: float):
        """Вызывается в каждом кадре игрового цикла."""
        try:
            # get_nowait не блокирует игру
            data = self.result_queue.get_nowait()
            self._apply_data(data)
        except queue.Empty:
            pass

    def _apply_data(self, data):
        emotion_pred = data['emotion']
        state_pred = data['state']

        # 1. Рассылаем эмоцию (для UI, Debug Render)
        self.event_manager.post(EmotionStateChangedEvent(emotion_pred))

        # 2. Обновляем GameDirector
        # Получаем dict множителей: {'spawn_rate_multiplier': 1.2, ...}
        mults = state_pred.multipliers
        
        target_vector = GameStateVector(
            spawn_rate_multiplier=mults.get("spawn_rate_multiplier", 1.0),
            enemy_speed_multiplier=mults.get("enemy_speed_multiplier", 1.0),
            enemy_health_multiplier=mults.get("enemy_health_multiplier", 1.0),
            enemy_damage_multiplier=mults.get("enemy_damage_multiplier", 1.0),
            player_speed_multiplier=mults.get("player_speed_multiplier", 1.0),
            player_damage_multiplier=mults.get("player_damage_multiplier", 1.0),
            item_drop_chance_modifier=mults.get("item_drop_chance_modifier", 1.0)
        )
        
        # GameDirector сам плавно интерполирует текущее состояние к этому target_vector
        self.director.set_new_target_vector(target_vector)

    def shutdown(self):
        """Обязательно вызвать при выходе из игры/стейта."""
        logger.info("Shutting down Biofeedback System...")
        self.stop_event.set()
        # Ждем завершения потока (с таймаутом, чтобы не зависнуть навсегда)
        self.worker.join(timeout=1.0)