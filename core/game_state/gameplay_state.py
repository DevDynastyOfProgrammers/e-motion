from .base_state import BaseState
from core.ecs.entity import EntityManager
from core.ecs.component import *
from core.ecs.system import *

class GameplayState(BaseState):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        self.entity_manager = EntityManager()
        
        # Initialize systems
        self.render_system = RenderSystem()
        self.player_input_system = PlayerInputSystem()
        self.enemy_chase_system = EnemyChaseSystem()
        self.spell_aura_system = SpellAuraSystem()
        self.death_system = DeathSystem()

        # Create player and enemy entities
        # TODO : use Factory pattern for entity creation
        self.create_player(100, 100)
        self.create_enemy(400, 300)

    def create_player(self, x, y):
        self.player = self.entity_manager.create_entity()
        #TODO : change hardcode pars to constants from settings.py
        self.entity_manager.add_component(self.player, TransformComponent(x, y, 30, 30, velocity=200))
        self.entity_manager.add_component(self.player, RenderComponent(color=(0, 150, 255)))
        self.entity_manager.add_component(self.player, PlayerInputComponent())
        self.entity_manager.add_component(self.player, HealthComponent(100, 100))
        self.entity_manager.add_component(self.player, TagComponent(tag="player"))
        self.entity_manager.add_component(self.player, SpellAuraComponent(radius=300.0, damage=20, tick_rate=1.0, target_tag="enemy"))

    def create_enemy(self, x, y):
        enemy = self.entity_manager.create_entity()
        #TODO : change hardcode pars to constants from settings.py
        self.entity_manager.add_component(enemy, TransformComponent(x, y, 25, 25, velocity=100))
        self.entity_manager.add_component(enemy, RenderComponent(color=(255, 50, 50)))
        self.entity_manager.add_component(enemy, AIComponent())
        self.entity_manager.add_component(enemy, HealthComponent(50, 50))
        self.entity_manager.add_component(enemy, TagComponent(tag="enemy"))
        self.entity_manager.add_component(enemy, SpellAuraComponent(radius=50.0, damage=10, tick_rate=1.0, target_tag="player"))

    def handle_events(self, events):
        """event handling plug"""
        pass

    def update(self, delta_time):
        # Update all systems in a specific order
        self.player_input_system.update(self.entity_manager, delta_time)

        player_transform = self.entity_manager.get_component(self.player, TransformComponent)
        self.enemy_chase_system.update(self.entity_manager, player_transform, delta_time)

        self.spell_aura_system.update(self.entity_manager, delta_time)

        self.death_system.update(self.entity_manager)
        
    def draw(self, screen):
        self.render_system.update(self.entity_manager, screen)