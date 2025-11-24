import pygame
from dataclasses import fields
from core.ecs.entity import EntityManager
from core.ecs.component import TransformComponent, RenderComponent
from core.director import GameDirector

# from .ai import GameplayMappingSystem (assuming relative import works)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ai import GameplayMappingSystem


class RenderSystem:
    """Renders all entities with TransformComponent and RenderComponent to the screen."""

    def draw(self, entity_manager: EntityManager, screen: pygame.Surface) -> None:
        screen.fill((20, 20, 20))

        entities = entity_manager.get_entities_with_components(TransformComponent, RenderComponent)
        for entity, (transform, render) in entities:
            rect = pygame.Rect(transform.x, transform.y, transform.width, transform.height)
            pygame.draw.rect(screen, render.color, rect)


class DebugRenderSystem:
    """
    Renders real-time debug information about the ML simulation and
    GameDirector state onto the screen.
    """

    def __init__(self, director: GameDirector, mapping_system: "GameplayMappingSystem") -> None:
        self.director = director
        self.mapping_system = mapping_system
        self.font = pygame.font.Font(None, 26)
        self.color = (255, 255, 255)
        self.x_offset = 10
        self.y_offset = 10
        self.line_height = 28

    def draw(self, screen: pygame.Surface) -> None:
        lines_to_render = []

        # ML System Info
        emotion = self.mapping_system.get_current_emotion_name()
        countdown = self.mapping_system.get_time_to_next_mapping()

        lines_to_render.append("--- MAPPING SYSTEM ---")
        lines_to_render.append(f"Current Emotion: {emotion}")
        lines_to_render.append(f"Next Vector In: {countdown:.1f}s")
        lines_to_render.append("")

        # Director Info
        # Using the new Dataclass getters
        target_state = self.director.target_state
        current_state = self.director.state

        lines_to_render.append("--- GAME DIRECTOR (Current -> Target) ---")

        # Iterate over dataclass fields dynamically
        for field in fields(current_state):
            name = field.name
            current_val = getattr(current_state, name)
            target_val = getattr(target_state, name)

            # Formatting name: enemy_speed_multiplier -> Enemy Speed
            clean_name = (
                name.replace("_multiplier", "").replace("_modifier", "").replace("_", " ").title()
            )

            line = f"{clean_name:<20}: {current_val:>5.2f} -> {target_val:.2f}"
            lines_to_render.append(line)

        current_y = self.y_offset
        for line in lines_to_render:
            text_surface = self.font.render(line, True, self.color)
            screen.blit(text_surface, (self.x_offset, current_y))
            current_y += self.line_height
