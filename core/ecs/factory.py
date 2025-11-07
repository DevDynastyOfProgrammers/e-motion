from .component import TransformComponent, RenderComponent, PlayerInputComponent, AIComponent, HealthComponent, TagComponent, SpellAuraComponent

class EntityFactory:
    """
    A factory for creating pre-configured game entities.
    This encapsulates the complexity of adding multiple components to an entity.
    """
    def __init__(self, entity_manager):
        self.entity_manager = entity_manager

    def create_player(self, x, y):
        player_id = self.entity_manager.create_entity()
        
        #TODO: change hardcode pars to constants from settings.py
        self.entity_manager.add_component(player_id, TransformComponent(x, y, 30, 30, velocity=200))
        self.entity_manager.add_component(player_id, RenderComponent(color=(0, 150, 255)))
        self.entity_manager.add_component(player_id, PlayerInputComponent())
        self.entity_manager.add_component(player_id, HealthComponent(100, 100))
        self.entity_manager.add_component(player_id, TagComponent(tag="player"))
        self.entity_manager.add_component(player_id, SpellAuraComponent(radius=300.0, damage=20, tick_rate=1.0, target_tag="enemy"))
        
        return player_id

    def create_enemy(self, x, y):
        enemy_id = self.entity_manager.create_entity()
        
        #TODO: change hardcode pars to constants from settings.py
        self.entity_manager.add_component(enemy_id, TransformComponent(x, y, 25, 25, velocity=100))
        self.entity_manager.add_component(enemy_id, RenderComponent(color=(255, 50, 50)))
        self.entity_manager.add_component(enemy_id, AIComponent())
        self.entity_manager.add_component(enemy_id, HealthComponent(50, 50))
        self.entity_manager.add_component(enemy_id, TagComponent(tag="enemy"))
        self.entity_manager.add_component(enemy_id, SpellAuraComponent(radius=50.0, damage=10, tick_rate=1.0, target_tag="player"))

        return enemy_id