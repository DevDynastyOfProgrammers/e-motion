import random
import math
from loguru import logger
from typing import Dict
from core.ecs.entity import EntityManager
from core.ecs.factory import EntityFactory
from core.ecs.component import TransformComponent, AIComponent
from core.event_manager import EventManager
from core.director import GameDirector
from core.events import SpawnProjectileEvent
from core.skill_data import ProjectileData, SpawnProjectileEffectData
from settings import SCREEN_WIDTH, SCREEN_HEIGHT


class EnemySpawningSystem:
    """Manages the spawning of enemy waves over time."""

    def __init__(self, entity_factory: EntityFactory, director: GameDirector) -> None:
        self.factory = entity_factory
        self.director = director

        self.time_since_last_single_spawn = 0.0
        self.base_single_spawn_interval = 3.0

        self.time_since_last_group_spawn = 0.0
        self.base_group_spawn_interval = 10.0
        self.group_size = 5
        self.group_spawn_radius = 100.0

    def update(self, delta_time: float) -> None:
        spawn_rate_multiplier = self.director.state.spawn_rate_multiplier
        if spawn_rate_multiplier <= 0:
            spawn_rate_multiplier = 0.001

        current_single_interval = self.base_single_spawn_interval / spawn_rate_multiplier
        current_group_interval = self.base_group_spawn_interval / spawn_rate_multiplier

        self.time_since_last_single_spawn += delta_time
        if self.time_since_last_single_spawn >= current_single_interval:
            self.time_since_last_single_spawn = 0.0
            self._spawn_single_enemy()

        self.time_since_last_group_spawn += delta_time
        if self.time_since_last_group_spawn >= current_group_interval:
            self.time_since_last_group_spawn = 0.0
            self._spawn_enemy_group()

    def _get_random_offscreen_position(self):
        side = random.randint(0, 3)
        if side == 0:
            x, y = random.randint(0, SCREEN_WIDTH), -50
        elif side == 1:
            x, y = SCREEN_WIDTH + 50, random.randint(0, SCREEN_HEIGHT)
        elif side == 2:
            x, y = random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT + 50
        else:
            x, y = -50, random.randint(0, SCREEN_HEIGHT)
        return x, y

    def _spawn_single_enemy(self) -> None:
        x, y = self._get_random_offscreen_position()
        logger.debug(f"Spawning single enemy at ({x}, {y})")
        self.factory.create_enemy(x, y)

    def _spawn_enemy_group(self) -> None:
        center_x, center_y = self._get_random_offscreen_position()
        logger.debug(f"Spawning GROUP of {self.group_size} enemies around ({center_x}, {center_y})")
        for _ in range(self.group_size):
            offset_x = random.uniform(-self.group_spawn_radius, self.group_spawn_radius)
            offset_y = random.uniform(-self.group_spawn_radius, self.group_spawn_radius)
            self.factory.create_enemy(center_x + offset_x, center_y + offset_y)


class ProjectileSpawningSystem:
    """Listens for SpawnProjectileEvent and creates projectiles."""

    def __init__(
        self,
        event_manager: EventManager,
        entity_manager: EntityManager,
        projectile_definitions: Dict[str, ProjectileData],
        factory: EntityFactory,
    ) -> None:
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.projectile_definitions = projectile_definitions
        self.factory = factory
        self.event_manager.subscribe(SpawnProjectileEvent, self.on_spawn_projectile)

    def on_spawn_projectile(self, event: SpawnProjectileEvent) -> None:
        if not isinstance(event.effect_data, SpawnProjectileEffectData):
            return

        caster_transform = self.entity_manager.get_component(event.caster_id, TransformComponent)
        if not caster_transform: return
        
        projectile_data = self.projectile_definitions.get(event.effect_data.projectile_id)
        if not projectile_data:
            logger.error(f"Unknown projectile_id '{event.effect_data.projectile_id}'")
            return

        direction = (1.0, 0.0)
        if event.effect_data.target_logic == "nearest_enemy":
            target = self._find_nearest_enemy(caster_transform.x, caster_transform.y)
            if target:
                target_transform = self.entity_manager.get_component(target, TransformComponent)
                dx = target_transform.x - caster_transform.x
                dy = target_transform.y - caster_transform.y
                dist = math.hypot(dx, dy)
                if dist > 0:
                    direction = (dx / dist, dy / dist)

        self.factory.create_projectile(
            event.caster_id, caster_transform.x, caster_transform.y, direction, projectile_data
        )

    def _find_nearest_enemy(self, x: float, y: float) -> int | None:
        enemies = self.entity_manager.get_entities_with_components(AIComponent, TransformComponent)
        closest_enemy = None
        min_dist_sq = float("inf")

        for enemy_id, (ai, transform) in enemies:
            dist_sq = (transform.x - x) ** 2 + (transform.y - y) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_enemy = enemy_id
        return closest_enemy
