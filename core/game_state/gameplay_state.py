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
    Manages the Entity-Component-System architecture and the Game Director.
    """
    def __init__(self, state_manager) -> None:
        self.state_manager = state_manager
        
        # 1. Core Services Initialization
        self.entity_manager = EntityManager()
        self.event_manager = EventManager()
        self.director = GameDirector()
        
        # 2. Data Loading
        data_loader = DataLoader()
        self.skill_definitions, self.projectile_definitions = data_loader.load_game_data("skills.yaml")
        self.entity_definitions = data_loader.load_entities("entities.yaml")
        
        # 3. Factory Setup
        self.entity_factory = EntityFactory(
            self.entity_manager, 
            self.director,
            self.entity_definitions
        )
        
        # 4. Systems Initialization
        self._init_systems()

        # 5. Game World Setup
        self.player = self.entity_factory.create_player(300, 300)

    def _init_systems(self) -> None:
        """Initialize all ECS systems and organize them into execution groups."""
        
        # Input & AI (ML)
        self.player_input_system = PlayerInputSystem(self.event_manager)
        self.emotion_recognition_system = EmotionRecognitionSystem(self.event_manager)
        self.gameplay_mapping_system = GameplayMappingSystem(self.event_manager, self.director)
        
        # Logic & Mechanics
        self.skill_system = SkillSystem(self.event_manager, self.entity_manager, self.skill_definitions)
        self.skill_execution_system = SkillExecutionSystem(self.event_manager, self.entity_manager, self.skill_definitions)
        self.enemy_spawning_system = EnemySpawningSystem(self.entity_factory, self.director)
        self.lifetime_system = LifetimeSystem(self.event_manager, self.entity_manager)
        
        # Movement & Physics
        self.movement_system = MovementSystem(self.event_manager, self.entity_manager, self.director)
        self.enemy_chase_system = EnemyChaseSystem(self.director)
        self.projectile_movement_system = ProjectileMovementSystem()
        self.projectile_impact_system = ProjectileImpactSystem(self.event_manager, self.entity_manager)
        self.projectile_spawning_system = ProjectileSpawningSystem(
            self.event_manager, self.entity_manager, self.projectile_definitions, self.entity_factory
        )
        
        # Combat & Cleanup
        self.damage_system = DamageSystem(self.event_manager, self.entity_manager, self.director)
        self.death_system = DeathSystem(self.event_manager, self.entity_manager)
        
        # Rendering
        self.render_system = RenderSystem()
        self.debug_render_system = DebugRenderSystem(self.director, self.gameplay_mapping_system)

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Process raw PyGame events."""
        pass

    def update(self, delta_time: float) -> None:
        """Update the state logic."""
        
        # --- 1. Update ML & Director ---
        self.emotion_recognition_system.update(delta_time)
        self.gameplay_mapping_system.update(delta_time)
        self.director.update(delta_time)
        
        # --- 2. Update Input & Logic ---
        self.player_input_system.update(self.entity_manager)
        self.skill_system.update(delta_time)

        # --- 3. Update AI & Spawning ---
        # Specific update signature for EnemyChaseSystem (needs player position)
        player_transform = self.entity_manager.get_component(self.player, TransformComponent)
        self.enemy_chase_system.update(self.entity_manager, player_transform, delta_time)
        self.enemy_spawning_system.update(delta_time)
        
        # --- 4. Update Physics & Projectiles ---
        self.projectile_movement_system.update(self.entity_manager, delta_time)
        self.projectile_impact_system.update()
        self.movement_system.update(delta_time)
        
        # --- 5. Cleanup ---
        self.lifetime_system.update(delta_time)
        self.death_system.update()
        
        # --- 6. Process Events (Deferred calls) ---
        self.event_manager.process_events()
        
    def draw(self, screen: pygame.Surface) -> None:
        """Render the state content to the screen."""
        self.render_system.draw(self.entity_manager, screen)
        self.debug_render_system.draw(screen)