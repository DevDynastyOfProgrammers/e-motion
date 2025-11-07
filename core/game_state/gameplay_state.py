# from .base_state import BaseState
# from core.ecs.entity import EntityManager
# from core.ecs.factory import EntityFactory
# from core.ecs.component import *
# from core.ecs.system import *

# class GameplayState(BaseState):
#     def __init__(self, state_manager):
#         super().__init__(state_manager)
#         self.entity_manager = EntityManager()

#         self.entity_factory = EntityFactory(self.entity_manager)
        
#         # Initialize systems
#         self.render_system = RenderSystem()
#         self.player_input_system = PlayerInputSystem()
#         self.enemy_spawning_system = EnemySpawningSystem(self.entity_factory)
#         self.enemy_chase_system = EnemyChaseSystem()
#         self.spell_aura_system = SpellAuraSystem()
#         self.death_system = DeathSystem()

#         # Create player and enemy entities
#         # TODO : use Factory pattern for entity creation
#         self.player = self.entity_factory.create_player(100, 100)

#     def handle_events(self, events):
#         """event handling plug"""
#         pass

#     def update(self, delta_time):
#         # Update all systems in a specific order
#         self.player_input_system.update(self.entity_manager, delta_time)

#         player_transform = self.entity_manager.get_component(self.player, TransformComponent)
#         self.enemy_chase_system.update(self.entity_manager, player_transform, delta_time)

#         self.spell_aura_system.update(self.entity_manager, delta_time)

#         self.death_system.update(self.entity_manager)

#         self.enemy_spawning_system.update(delta_time)
        
#     def draw(self, screen):
#         self.render_system.update(self.entity_manager, screen)


# e-motion/core/game_state/gameplay_state.py

from .base_state import BaseState
from core.ecs.entity import EntityManager
from core.ecs.factory import EntityFactory
from core.ecs.system import RenderSystem, PlayerInputSystem, MovementSystem, EnemySpawningSystem, EnemyChaseSystem, SpellAuraSystem, DeathSystem
from core.ecs.component import TransformComponent
from core.event_manager import EventManager

class GameplayState(BaseState):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        self.entity_manager = EntityManager()
        self.event_manager = EventManager()

        self.entity_factory = EntityFactory(self.entity_manager)
        
        # Initialize systems and pass them the event manager
        self.render_system = RenderSystem()
        self.player_input_system = PlayerInputSystem(self.event_manager)
        self.movement_system = MovementSystem(self.event_manager, self.entity_manager)
        self.enemy_spawning_system = EnemySpawningSystem(self.entity_factory)
        self.enemy_chase_system = EnemyChaseSystem()
        self.spell_aura_system = SpellAuraSystem()
        self.death_system = DeathSystem(self.event_manager)

        # Create player and enemy entities
        self.player = self.entity_factory.create_player(100, 100)

    def handle_events(self, events):
        """event handling plug"""
        pass

    def update(self, delta_time):
        # 1. Input and AI systems post events about their intentions
        self.player_input_system.update(self.entity_manager)

        player_transform = self.entity_manager.get_component(self.player, TransformComponent)
        self.enemy_chase_system.update(self.entity_manager, player_transform, delta_time)

        # 2. Process game logic and state changes
        self.spell_aura_system.update(self.entity_manager, delta_time)
        self.death_system.update(self.entity_manager)
        self.enemy_spawning_system.update(delta_time)
        
        # 3. Process all queued events (like movement)
        self.event_manager.process_events()

        # 4. Apply physics and movement based on processed events and logic
        self.movement_system.update(delta_time)
        
    def draw(self, screen):
        self.render_system.update(self.entity_manager, screen)