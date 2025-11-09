from .base_state import BaseState
from core.ecs.entity import EntityManager
from core.ecs.factory import EntityFactory
from core.ecs.system import RenderSystem, PlayerInputSystem, MovementSystem, EnemySpawningSystem, EnemyChaseSystem, DeathSystem, \
    SkillSystem, SkillExecutionSystem, DamageSystem, ProjectileSpawningSystem, ProjectileMovementSystem, ProjectileImpactSystem, LifetimeSystem
from core.ecs.component import TransformComponent, PlayerInputComponent
from core.event_manager import EventManager
from core.data_loader import DataLoader
from core.director import GameDirector

class GameplayState(BaseState):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        
        self.entity_manager = EntityManager()
        self.event_manager = EventManager()
        self.director = GameDirector()
        data_loader = DataLoader()
        self.skill_definitions, self.projectile_definitions = data_loader.load_game_data("skills.yaml")
        # Pass director to factory
        self.entity_factory = EntityFactory(self.entity_manager, self.director) 
        
        # --- System Initialization (with Director injection) ---
        self.render_system = RenderSystem()
        self.player_input_system = PlayerInputSystem(self.event_manager)
        self.movement_system = MovementSystem(self.event_manager, self.entity_manager, self.director)
        self.enemy_spawning_system = EnemySpawningSystem(self.entity_factory, self.director)
        self.enemy_chase_system = EnemyChaseSystem(self.director)
        self.death_system = DeathSystem(self.event_manager, self.entity_manager)
        self.skill_system = SkillSystem(self.event_manager, self.entity_manager, self.skill_definitions)
        self.skill_execution_system = SkillExecutionSystem(self.event_manager, self.entity_manager, self.skill_definitions)
        # Pass director to DamageSystem
        self.damage_system = DamageSystem(self.event_manager, self.entity_manager, self.director)
        self.projectile_spawning_system = ProjectileSpawningSystem(self.event_manager, self.entity_manager, self.projectile_definitions, self.entity_factory)
        self.projectile_movement_system = ProjectileMovementSystem()
        self.projectile_impact_system = ProjectileImpactSystem(self.event_manager, self.entity_manager)
        self.lifetime_system = LifetimeSystem(self.event_manager, self.entity_manager)
        
        self.player = self.entity_factory.create_player(300, 300)
        
        # --- Proof of Concept: Simulate receiving a full MVP vector ---
        # TODO : Change this temporary code to model integration
        # [spawn, enemy_spd, enemy_hp, enemy_dmg, player_spd, player_dmg, item_drop]
        self.director.set_new_target_vector([2.0, 1.5, 2.5, 2.0, 1.4, 1.5, 1.0])

    def handle_events(self, events):
        """event handling plug"""
        pass

    def update(self, delta_time):
        self.director.update(delta_time)
        
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