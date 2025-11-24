import pygame
import math
from core.ecs.entity import EntityManager
from core.ecs.component import PlayerInputComponent, TransformComponent
from core.event_manager import EventManager
from core.events import PlayerMoveIntentEvent


class PlayerInputSystem:
    """Processes player input and posts movement intent events."""

    def __init__(self, event_manager: EventManager) -> None:
        self.event_manager = event_manager

    def update(self, entity_manager: EntityManager) -> None:
        # Find the player entity
        entities = entity_manager.get_entities_with_components(
            PlayerInputComponent, TransformComponent
        )
        for entity, (input_comp, transform) in entities:
            keys = pygame.key.get_pressed()
            dx, dy = 0.0, 0.0

            if keys[pygame.K_a]:
                dx -= 1.0
            if keys[pygame.K_d]:
                dx += 1.0
            if keys[pygame.K_w]:
                dy -= 1.0
            if keys[pygame.K_s]:
                dy += 1.0

            # Normalize vector
            if dx != 0 or dy != 0:
                length = math.sqrt(dx**2 + dy**2)
                dx /= length
                dy /= length

                self.event_manager.post(PlayerMoveIntentEvent(entity, (dx, dy)))
