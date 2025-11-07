from .component import TransformComponent, RenderComponent, PlayerInputComponent, \
    AIComponent, HealthComponent, TagComponent, DamageOnCollisionComponent, \
    LifetimeComponent, SkillSetComponent, ProjectileComponent
from core.skill_data import ProjectileData

class EntityFactory:
    """
    A factory for creating pre-configured game entities.
    """
    def __init__(self, entity_manager):
        self.entity_manager = entity_manager
        # Mapping from string in YAML to component class
        self.component_map = {
            "Transform": TransformComponent,
            "Render": RenderComponent,
            "DamageOnCollision": DamageOnCollisionComponent,
            "Lifetime": LifetimeComponent,
        }

    def create_player(self, x, y):
        player_id = self.entity_manager.create_entity()
        
        self.entity_manager.add_component(player_id, TransformComponent(x, y, 30, 30, velocity=200))
        self.entity_manager.add_component(player_id, RenderComponent(color=(0, 150, 255)))
        self.entity_manager.add_component(player_id, PlayerInputComponent())
        self.entity_manager.add_component(player_id, HealthComponent(100, 100))
        self.entity_manager.add_component(player_id, TagComponent(tag="player"))
        self.entity_manager.add_component(player_id, SkillSetComponent(skill_ids=["PlayerAura", "Fireball"]))
        
        return player_id

    def create_enemy(self, x, y):
        enemy_id = self.entity_manager.create_entity()
        
        self.entity_manager.add_component(enemy_id, TransformComponent(x, y, 25, 25, velocity=100))
        self.entity_manager.add_component(enemy_id, RenderComponent(color=(255, 50, 50)))
        self.entity_manager.add_component(enemy_id, AIComponent())
        self.entity_manager.add_component(enemy_id, HealthComponent(50, 50))
        self.entity_manager.add_component(enemy_id, TagComponent(tag="enemy"))
        self.entity_manager.add_component(enemy_id, SkillSetComponent(skill_ids=["EnemyAura"]))

        return enemy_id

    def create_projectile(self, x, y, direction, projectile_data: ProjectileData):
        """Creates a projectile entity based on its data definition."""
        proj_id = self.entity_manager.create_entity()
        
        # Add the base projectile marker component
        self.entity_manager.add_component(proj_id, ProjectileComponent(direction[0], direction[1]))
        
        # Add components from the YAML definition
        for comp_name, comp_args in projectile_data.components.items():
            comp_class = self.component_map.get(comp_name)
            if comp_class:
                # Special handling for Transform to set initial position
                if comp_class == TransformComponent:
                    comp_args['x'], comp_args['y'] = x, y
                
                component = comp_class(**comp_args)
                self.entity_manager.add_component(proj_id, component)
            else:
                print(f"WARNING: Unknown component type '{comp_name}' in projectile '{projectile_data.projectile_id}'")

        return proj_id