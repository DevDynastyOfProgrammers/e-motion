# core/ecs/factory.py

from core.ecs.entity import EntityManager
from core.director import GameDirector
from core.ecs.component import (
    TransformComponent, RenderComponent, PlayerInputComponent, 
    AIComponent, HealthComponent, TagComponent, DamageOnCollisionComponent, 
    LifetimeComponent, SkillSetComponent, ProjectileComponent, Component
)
from core.skill_data import ProjectileData
from typing import Type

class EntityFactory:
    """
    A factory for creating pre-configured game entities.
    """
    def __init__(self, entity_manager: EntityManager, director: GameDirector) -> None:
        self.entity_manager = entity_manager
        self.director = director # Injected director dependency
        self.component_map: dict[str, Type[Component]] = {
            "Transform": TransformComponent,
            "Render": RenderComponent,
            "DamageOnCollision": DamageOnCollisionComponent,
            "Lifetime": LifetimeComponent,
        }

    def create_player(self, x: float, y: float) -> int:
        player_id = self.entity_manager.create_entity()
        
        self.entity_manager.add_component(player_id, TransformComponent(x, y, 30, 30, velocity=200))
        self.entity_manager.add_component(player_id, RenderComponent(color=(0, 150, 255)))
        self.entity_manager.add_component(player_id, PlayerInputComponent())
        self.entity_manager.add_component(player_id, HealthComponent(100, 100))
        self.entity_manager.add_component(player_id, TagComponent(tag="player"))
        self.entity_manager.add_component(player_id, SkillSetComponent(skill_ids=["PlayerAura", "Fireball", "Iceball"]))
        
        return player_id

    def create_enemy(self, x: float, y: float) -> int:
        enemy_id = self.entity_manager.create_entity()
        
        # Apply health multiplier from director
        base_health = 50
        health_multiplier = self.director.get_enemy_health_multiplier()
        final_health = int(base_health * health_multiplier)

        self.entity_manager.add_component(enemy_id, TransformComponent(x, y, 25, 25, velocity=100))
        self.entity_manager.add_component(enemy_id, RenderComponent(color=(255, 50, 50)))
        self.entity_manager.add_component(enemy_id, AIComponent())
        self.entity_manager.add_component(enemy_id, HealthComponent(final_health, final_health))
        self.entity_manager.add_component(enemy_id, TagComponent(tag="enemy"))
        self.entity_manager.add_component(enemy_id, SkillSetComponent(skill_ids=["EnemyAura"]))

        return enemy_id

    def create_projectile(
        self, 
        caster_id: int, 
        x: float, 
        y: float, 
        direction: tuple[float, float], 
        projectile_data: ProjectileData
    ) -> int:
        """Creates a projectile entity based on its data definition."""
        proj_id = self.entity_manager.create_entity()
        
        # Add the base projectile marker component with caster_id
        self.entity_manager.add_component(proj_id, ProjectileComponent(direction[0], direction[1], caster_id))
        
        # Add components from the YAML definition
        for comp_name, comp_args in projectile_data.components.items():
            comp_class = self.component_map.get(comp_name)
            if comp_class:
                # We need to type-ignore or cast args here because kwargs unpacking 
                # into dynamic classes is hard for static analysis
                if comp_class == TransformComponent:
                    comp_args['x'], comp_args['y'] = x, y
                
                # Using type: ignore because we trust YAML structure matches Component __init__
                component = comp_class(**comp_args) # type: ignore
                self.entity_manager.add_component(proj_id, component)
            else:
                print(f"WARNING: Unknown component type '{comp_name}' in projectile '{projectile_data.projectile_id}'")

        return proj_id