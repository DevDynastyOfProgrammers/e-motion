from typing import Type

from loguru import logger

from core.director import GameDirector
from core.ecs.component import (
    AIComponent,
    Component,
    DamageOnCollisionComponent,
    HealthComponent,
    LifetimeComponent,
    PlayerInputComponent,
    ProjectileComponent,
    RenderComponent,
    SkillSetComponent,
    TransformComponent,
)
from core.ecs.entity import EntityManager
from core.entity_data import EntityData
from core.skill_data import ProjectileData


class EntityFactory:
    """
    A factory for creating pre-configured game entities using Data-Driven design.
    """

    def __init__(
        self,
        entity_manager: EntityManager,
        director: GameDirector,
        entity_definitions: dict[str, EntityData],
    ) -> None:
        self.entity_manager = entity_manager
        self.director = director
        self.entity_definitions = entity_definitions

        self.component_map: dict[str, Type[Component]] = {
            'Transform': TransformComponent,
            'Render': RenderComponent,
            'DamageOnCollision': DamageOnCollisionComponent,
            'Lifetime': LifetimeComponent,
        }

    def _assemble_entity(self, x: float, y: float, config_key: str, is_enemy: bool = False) -> int:
        """
        Generic builder that assembles an entity from a configuration key.
        """
        data = self.entity_definitions.get(config_key)
        if not data:
            logger.error(f"CRITICAL: Entity definition '{config_key}' not found!")
            return self.entity_manager.create_entity()

        entity_id = self.entity_manager.create_entity()

        # 1. Transform & Velocity
        velocity = data.transform.velocity
        self.entity_manager.add_component(
            entity_id,
            TransformComponent(x, y, data.transform.width, data.transform.height, velocity),
        )

        self.entity_manager.add_component(
            entity_id,
            TransformComponent(x, y, data.transform.width, data.transform.height, velocity),
        )

        # 2. Render
        self.entity_manager.add_component(entity_id, RenderComponent(color=data.color))

        # 3. Health
        max_hp = data.max_hp
        if is_enemy:
            health_multiplier = self.director.state.enemy_health_multiplier
            max_hp = int(max_hp * health_multiplier)

        self.entity_manager.add_component(entity_id, HealthComponent(max_hp, max_hp))

        # 4. Tags - REMOVED

        # 5. Marker Components (AI, PlayerInput)
        for comp_name in data.components:
            if comp_name == 'PlayerInput':
                self.entity_manager.add_component(entity_id, PlayerInputComponent())
            elif comp_name == 'AI':
                self.entity_manager.add_component(entity_id, AIComponent())

        # 6. Skills
        if data.skills:
            self.entity_manager.add_component(entity_id, SkillSetComponent(skill_ids=data.skills))

        return entity_id

    def create_player(self, x: float, y: float) -> int:
        return self._assemble_entity(x, y, 'Player', is_enemy=False)

    def create_enemy(self, x: float, y: float) -> int:
        return self._assemble_entity(x, y, 'Enemy', is_enemy=True)

    def create_projectile(
        self,
        caster_id: int,
        x: float,
        y: float,
        direction: tuple[float, float],
        projectile_data: ProjectileData,
    ) -> int:
        """Creates a projectile entity based on its data definition."""
        proj_id = self.entity_manager.create_entity()

        # Add the base projectile marker component with caster_id
        self.entity_manager.add_component(proj_id, ProjectileComponent(direction[0], direction[1], caster_id))

        # Add components from the YAML definition
        for comp_name, comp_args in projectile_data.components.items():
            comp_class = self.component_map.get(comp_name)
            if comp_class:
                if comp_class == TransformComponent:
                    comp_args['x'], comp_args['y'] = x, y

                component = comp_class(**comp_args)
                self.entity_manager.add_component(proj_id, component)
            else:
                logger.warning(f"Unknown component type '{comp_name}' in projectile '{projectile_data.projectile_id}'")

        return proj_id
