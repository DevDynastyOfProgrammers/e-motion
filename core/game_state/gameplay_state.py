from .base_state import BaseState
from core.ecs.entity import EntityManager
from core.ecs.factory import EntityFactory
from core.ecs.system import RenderSystem, PlayerInputSystem, MovementSystem, EnemySpawningSystem, EnemyChaseSystem, DeathSystem, \
    SkillSystem, SkillExecutionSystem, DamageSystem, ProjectileSpawningSystem, ProjectileMovementSystem, ProjectileImpactSystem, LifetimeSystem
from core.ecs.component import TransformComponent
from core.event_manager import EventManager
from core.data_loader import DataLoader

class GameplayState(BaseState):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        
        self.entity_manager = EntityManager()
        self.event_manager = EventManager()
        data_loader = DataLoader()
        self.skill_definitions, self.projectile_definitions = data_loader.load_game_data("skills.yaml")
        self.entity_factory = EntityFactory(self.entity_manager)
        
        # --- System Initialization ---
        self.render_system = RenderSystem()
        self.player_input_system = PlayerInputSystem(self.event_manager)
        self.movement_system = MovementSystem(self.event_manager, self.entity_manager)
        self.enemy_spawning_system = EnemySpawningSystem(self.entity_factory)
        self.enemy_chase_system = EnemyChaseSystem()
        self.death_system = DeathSystem(self.event_manager, self.entity_manager)
        self.skill_system = SkillSystem(self.event_manager, self.entity_manager, self.skill_definitions)
        self.skill_execution_system = SkillExecutionSystem(self.event_manager, self.entity_manager, self.skill_definitions)
        self.damage_system = DamageSystem(self.event_manager, self.entity_manager)
        self.projectile_spawning_system = ProjectileSpawningSystem(self.event_manager, self.entity_manager, self.projectile_definitions, self.entity_factory)
        self.projectile_movement_system = ProjectileMovementSystem()
        self.projectile_impact_system = ProjectileImpactSystem(self.event_manager, self.entity_manager)
        self.lifetime_system = LifetimeSystem(self.event_manager, self.entity_manager)
        
        self.player = self.entity_factory.create_player(300, 300)

    def handle_events(self, events):
        """event handling plug"""
        pass

    def update(self, delta_time):
        self.player_input_system.update(self.entity_manager)
        self.skill_system.update(delta_time)

        player_transform = self.entity_manager.get_component(self.player, TransformComponent)
        self.enemy_chase_system.update(self.entity_manager, player_transform, delta_time)
        self.enemy_spawning_system.update(delta_time)
        self.projectile_movement_system.update(self.entity_manager, delta_time)
        self.projectile_impact_system.update()
        self.lifetime_system.update(delta_time)
        self.event_manager.process_events()
        self.movement_system.update(delta_time)
        self.death_system.update()
        
    def draw(self, screen):
        self.render_system.update(self.entity_manager, screen)