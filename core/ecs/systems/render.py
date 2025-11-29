import pygame
from dataclasses import fields
from core.ecs.entity import EntityManager
from core.ecs.component import TransformComponent, RenderComponent
from core.director import GameDirector
from core.event_manager import EventManager
from core.events import EmotionStateChangedEvent

class RenderSystem:
    """Renders all entities with TransformComponent and RenderComponent to the screen."""
    def draw(self, entity_manager: EntityManager, screen: pygame.Surface) -> None:
        screen.fill((20, 20, 20)) # Dark background

        entities = entity_manager.get_entities_with_components(TransformComponent, RenderComponent)
        for entity, (transform, render) in entities:
            rect = pygame.Rect(transform.x, transform.y, transform.width, transform.height)
            pygame.draw.rect(screen, render.color, rect)


class DebugRenderSystem:
    """
    Renders real-time debug information about ML and Game State.
    Listens to EventManager to get the latest emotion.
    """

    def __init__(self, director: GameDirector, event_manager: EventManager) -> None:
        self.director = director
        self.event_manager = event_manager
        
        self.font = pygame.font.Font(None, 24)
        self.color = (255, 255, 255)
        self.x_offset = 10
        self.y_offset = 10
        self.line_height = 20
        
        # Храним последнюю эмоцию локально для отрисовки
        self.last_emotion_name = "Waiting..."
        self.last_confidence = 0.0
        
        # Подписываемся на обновление эмоций
        self.event_manager.subscribe(EmotionStateChangedEvent, self._on_emotion_update)

    def _on_emotion_update(self, event: EmotionStateChangedEvent):
        self.last_emotion_name = event.prediction.dominant_emotion.name
        self.last_confidence = event.prediction.confidence

    def draw(self, screen: pygame.Surface) -> None:
        lines = []

        # 1. ML Info
        lines.append(f"--- BIOFEEDBACK ---")
        lines.append(f"Emotion: {self.last_emotion_name} ({self.last_confidence:.2f})")
        lines.append("")

        # 2. Director Info (Interpolation)
        current_state = self.director.state
        target_state = self.director.target_state

        lines.append(f"--- GAME STATE (Current -> Target) ---")
        
        # Динамически проходим по полям датакласса
        for field in fields(current_state):
            name = field.name
            cur_val = getattr(current_state, name)
            tgt_val = getattr(target_state, name)
            
            # Форматируем имя: enemy_speed_multiplier -> Enemy Speed
            pretty_name = name.replace("_multiplier", "").replace("_modifier", "").replace("_", " ").title()
            
            # Красим строку, если значение отличается от 1.0 (активный модификатор)
            prefix = " "
            if abs(cur_val - 1.0) > 0.01:
                prefix = "*" 
            
            lines.append(f"{prefix} {pretty_name:<18}: {cur_val:5.2f} -> {tgt_val:5.2f}")

        # Рендеринг текста
        y = self.y_offset
        for line in lines:
            text_surf = self.font.render(line, True, self.color)
            # Добавляем черную обводку для читаемости
            outline_surf = self.font.render(line, True, (0,0,0))
            screen.blit(outline_surf, (self.x_offset + 1, y + 1))
            screen.blit(text_surf, (self.x_offset, y))
            y += self.line_height