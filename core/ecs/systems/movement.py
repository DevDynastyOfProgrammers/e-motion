import math

from core.director import GameDirector
from core.ecs.component import (
    AIComponent,
    PlayerInputComponent,
    ProjectileComponent,
    TransformComponent,
)
from core.ecs.entity import EntityManager
from core.event_manager import EventManager
from core.events import PlayerMoveIntentEvent


class MovementSystem:
    """
    Handles movement requests and updates TransformComponents.
    """

    def __init__(self, event_manager: EventManager, entity_manager: EntityManager, director: GameDirector):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.director = director

        self.event_manager.subscribe(PlayerMoveIntentEvent, self.on_player_move)
        self.movement_requests: dict[int, tuple[float, float]] = {}

    def on_player_move(self, event: PlayerMoveIntentEvent) -> None:
        self.movement_requests[event.entity_id] = event.direction

    def update(self, delta_time: float) -> None:
        for entity_id, direction in self.movement_requests.items():
            transform = self.entity_manager.get_component(entity_id, TransformComponent)
            if transform:
                speed_multiplier = 1.0
                if self.entity_manager.get_component(entity_id, PlayerInputComponent):
                    speed_multiplier = self.director.state.player_speed_multiplier

                transform.x += direction[0] * transform.velocity * speed_multiplier * delta_time
                transform.y += direction[1] * transform.velocity * speed_multiplier * delta_time

        self.movement_requests.clear()


class EnemyChaseSystem:
    """Processes AI-controlled enemies to chase the player."""

    def __init__(self, director: GameDirector) -> None:
        self.director = director

    def update(self, entity_manager: EntityManager, player_transform: TransformComponent, delta_time: float) -> None:
        if not player_transform:
            return

        speed_multiplier = self.director.state.enemy_speed_multiplier

        enemies = entity_manager.get_entities_with_components(AIComponent, TransformComponent)
        for entity, (ai_comp, transform) in enemies:
            dx = player_transform.x - transform.x
            dy = player_transform.y - transform.y

            dist = math.hypot(dx, dy)
            if dist > 0:
                dx, dy = dx / dist, dy / dist

            transform.x += dx * transform.velocity * speed_multiplier * delta_time
            transform.y += dy * transform.velocity * speed_multiplier * delta_time


class ProjectileMovementSystem:
    """Moves all entities with a ProjectileComponent."""

    def update(self, entity_manager: EntityManager, delta_time: float) -> None:
        projectiles = entity_manager.get_entities_with_components(ProjectileComponent, TransformComponent)
        for entity, (proj, transform) in projectiles:
            transform.x += proj.dx * transform.velocity * delta_time
            transform.y += proj.dy * transform.velocity * delta_time
