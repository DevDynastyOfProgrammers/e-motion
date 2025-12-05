from dataclasses import fields
from typing import TYPE_CHECKING

import pygame

from core.director import GameDirector
from core.ecs.component import RenderComponent, TransformComponent
from core.ecs.entity import EntityManager
from core.event_manager import EventManager
from core.events import EmotionStateChangedEvent

if TYPE_CHECKING:
    from core.ecs.systems.biofeedback import BiofeedbackSystem


class RenderSystem:
    """Renders all entities with TransformComponent and RenderComponent to the screen."""

    def draw(self, entity_manager: EntityManager, screen: pygame.Surface) -> None:
        screen.fill((20, 20, 20))  # Dark background

        entities = entity_manager.get_entities_with_components(TransformComponent, RenderComponent)
        for entity, (transform, render) in entities:
            rect = pygame.Rect(transform.x, transform.y, transform.width, transform.height)
            pygame.draw.rect(screen, render.color, rect)


class DebugRenderSystem:
    """
    Renders real-time debug information about ML and Game State.
    Visualizes emotion probabilities as bars and shows the active Game Preset.
    """

    def __init__(self, director: GameDirector, event_manager: EventManager, bio_system: 'BiofeedbackSystem') -> None:
        self.director = director
        self.event_manager = event_manager
        self.bio_system = bio_system

        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 28)
        self.color = (255, 255, 255)

        # UI Layout
        self.x_offset = 10
        self.y_offset = 10
        self.bar_width = 100
        self.bar_height = 10

        # Data storage
        self.last_prediction = None
        self.current_preset = 'Unknown'

        # Subscribe
        self.event_manager.subscribe(EmotionStateChangedEvent, self._on_emotion_update)

    def _on_emotion_update(self, event: EmotionStateChangedEvent):
        self.last_prediction = event.prediction
        # We assume the last computed multipliers reflect the current preset
        # Ideally, we would pass the preset name in an event, but for debug we can infer or wait
        pass

    def draw(self, screen: pygame.Surface) -> None:
        # Background
        bg_rect = pygame.Rect(5, 5, 250, 550)  # Увеличили высоту, чтобы влезла камера
        s = pygame.Surface((bg_rect.width, bg_rect.height))
        s.set_alpha(180)
        s.fill((0, 0, 0))
        screen.blit(s, (bg_rect.x, bg_rect.y))

        y = self.y_offset

        # --- 0. CAMERA FEED ---
        # Берем кадр из bio_system
        if self.bio_system.current_debug_frame is not None:
            try:
                # numpy array -> pygame surface
                frame_surf = pygame.surfarray.make_surface(self.bio_system.current_debug_frame)
                screen.blit(frame_surf, (self.x_offset, y))
                y += 130  # Смещаем текст вниз (высота картинки 120 + 10 отступ)
            except Exception:
                pass
        else:
            self._draw_text(screen, '[No Camera Feed]', y)
            y += 30

        # --- 1. Vision Model Outputs ---
        self._draw_text(screen, 'VISION MODEL (Smoothed)', y, self.title_font)
        y += 25

        if self.last_prediction:
            probs = [
                ('Angry', self.last_prediction.prob_angry_disgust, (255, 50, 50)),
                ('Fear', self.last_prediction.prob_fear_surprise, (200, 50, 200)),
                ('Happy', self.last_prediction.prob_happy, (50, 255, 50)),
                ('Neutral', self.last_prediction.prob_neutral, (200, 200, 200)),
                ('Sad', self.last_prediction.prob_sad, (50, 100, 255)),
            ]

            for name, val, color in probs:
                self._draw_bar(screen, self.x_offset, y, name, val, color)
                y += 18
        else:
            self._draw_text(screen, 'Waiting for camera...', y)
            y += 30

        y += 10

        # --- 2. State Director ---
        # Get target multipliers to see what we are aiming for
        target = self.director.target_state
        current = self.director.state

        self._draw_text(screen, 'STATE DIRECTOR', y, self.title_font)
        y += 25

        # Interpolation Progress (Visual guess based on one param)
        diff = abs(target.spawn_rate_multiplier - current.spawn_rate_multiplier)
        status = 'Morphing...' if diff > 0.05 else 'Stable'
        self._draw_text(screen, f'Status: {status}', y)
        y += 20

        # Draw Multipliers
        params = [
            ('Spawn Rate', current.spawn_rate_multiplier, target.spawn_rate_multiplier),
            ('Enemy Speed', current.enemy_speed_multiplier, target.enemy_speed_multiplier),
            ('Enemy HP', current.enemy_health_multiplier, target.enemy_health_multiplier),
            ('Player Dmg', current.player_damage_multiplier, target.player_damage_multiplier),
        ]

        for name, cur, tgt in params:
            # Color logic: Red if getting harder (spawn up), Green if easier
            color = (200, 200, 200)
            if tgt > 1.05:
                color = (255, 100, 100)  # Harder
            if tgt < 0.95:
                color = (100, 255, 100)  # Easier

            text = f'{name}: {cur:.2f} -> {tgt:.2f}'
            self._draw_text(screen, text, y, color=color)
            y += 18

    def _draw_bar(self, screen, x, y, label, value, color):
        # Draw Label
        text_surf = self.font.render(f'{label}', True, (255, 255, 255))
        screen.blit(text_surf, (x, y))

        # Draw Bar Background
        bar_x = x + 60
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, y + 2, self.bar_width, self.bar_height))

        # Draw Filled Bar
        fill_width = int(value * self.bar_width)
        pygame.draw.rect(screen, color, (bar_x, y + 2, fill_width, self.bar_height))

        # Draw Value Text
        val_surf = self.font.render(f'{value:.2f}', True, (200, 200, 200))
        screen.blit(val_surf, (bar_x + self.bar_width + 5, y))

    def _draw_text(self, screen, text, y, font=None, color=None):
        if font is None:
            font = self.font
        if color is None:
            color = self.color
        surf = font.render(text, True, color)
        screen.blit(surf, (self.x_offset, y))
