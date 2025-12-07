import math
from dataclasses import dataclass, field

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


@dataclass
class MovementSystem:
    """
    Handles movement requests and updates TransformComponents.
    """

    event_manager: EventManager
    entity_manager: EntityManager
    director: GameDirector

    movement_requests: dict[int, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_manager.subscribe(PlayerMoveIntentEvent, self.on_player_move)

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


@dataclass
class EnemyChaseSystem:
    """Processes AI-controlled enemies to chase the player."""

    director: GameDirector

    def update(self, entity_manager: EntityManager, player_transform: TransformComponent, delta_time: float) -> None:
        if not player_transform:
            return

        speed_multiplier = self.director.state.enemy_speed_multiplier

        enemies = entity_manager.get_entities_with_components(AIComponent, TransformComponent)
        for _, (_, transform) in enemies:
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
        for _, (proj, transform) in projectiles:
            transform.x += proj.dx * transform.velocity * delta_time
            transform.y += proj.dy * transform.velocity * delta_time
