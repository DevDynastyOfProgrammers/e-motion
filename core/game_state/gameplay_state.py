import pygame
from core.ecs.entity import EntityManager
from core.ecs.factory import EntityFactory
from core.event_manager import EventManager
from core.data_loader import DataLoader
from core.director import GameDirector
from core.ecs.component import TransformComponent

from core.ecs.systems import (
    RenderSystem,
    PlayerInputSystem,
    MovementSystem,
    EnemySpawningSystem,
    EnemyChaseSystem,
    DeathSystem,
    SkillSystem,
    SkillExecutionSystem,
    DamageSystem,
    ProjectileSpawningSystem,
    ProjectileMovementSystem,
    ProjectileImpactSystem,
    LifetimeSystem,
    EmotionRecognitionSystem,
    GameplayMappingSystem,
    DebugRenderSystem,
)


class GameplayState:
    """
    Main gameplay state implementing the GameState Protocol.
    """

    def __init__(self, state_manager) -> None:
        self.state_manager = state_manager

        self.entity_manager = EntityManager()
        self.event_manager = EventManager()
        self.director = GameDirector()

        data_loader = DataLoader()
        self.skill_definitions, self.projectile_definitions = data_loader.load_game_data("skills.yaml")
        self.entity_definitions = data_loader.load_entities("entities.yaml")
        
        # --- FACTORY INJECTION ---
        self.entity_factory = EntityFactory(
            self.entity_manager, 
            self.director, 
            self.entity_definitions # Pass the loaded data
        )

        # --- System Initialization ---
        self.render_system = RenderSystem()
        self.player_input_system = PlayerInputSystem(self.event_manager)
        self.movement_system = MovementSystem(
            self.event_manager, self.entity_manager, self.director
        )
        self.enemy_spawning_system = EnemySpawningSystem(self.entity_factory, self.director)
        self.enemy_chase_system = EnemyChaseSystem(self.director)
        self.death_system = DeathSystem(self.event_manager, self.entity_manager)
        self.skill_system = SkillSystem(
            self.event_manager, self.entity_manager, self.skill_definitions
        )
        self.skill_execution_system = SkillExecutionSystem(
            self.event_manager, self.entity_manager, self.skill_definitions
        )
        self.damage_system = DamageSystem(self.event_manager, self.entity_manager, self.director)
        self.projectile_spawning_system = ProjectileSpawningSystem(
            self.event_manager,
            self.entity_manager,
            self.projectile_definitions,
            self.entity_factory,
        )
        self.projectile_movement_system = ProjectileMovementSystem()
        self.projectile_impact_system = ProjectileImpactSystem(
            self.event_manager, self.entity_manager
        )
        self.lifetime_system = LifetimeSystem(self.event_manager, self.entity_manager)

        # --- ML Simulation Systems Initialization ---
        self.emotion_recognition_system = EmotionRecognitionSystem(self.event_manager)
        self.gameplay_mapping_system = GameplayMappingSystem(self.event_manager, self.director)
        self.debbug_render_system = DebugRenderSystem(self.director, self.gameplay_mapping_system)

        self.player = self.entity_factory.create_player(300, 300)

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Process events (Protocol implementation)"""
        pass

    def update(self, delta_time: float) -> None:
        """Update logic (Protocol implementation)"""
        # Update ML simulators first
        self.emotion_recognition_system.update(delta_time)
        self.gameplay_mapping_system.update(delta_time)

        # Update the director's state interpolation
        self.director.update(delta_time)

        # --- Regular game loop ---
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

    def draw(self, screen: pygame.Surface) -> None:
        """Render logic (Protocol implementation)"""
        self.render_system.draw(self.entity_manager, screen)
        self.debbug_render_system.draw(screen)
