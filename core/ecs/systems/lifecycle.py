from core.ecs.entity import EntityManager
from core.ecs.component import HealthComponent, LifetimeComponent
from core.event_manager import EventManager
from core.events import RequestEntityRemovalEvent, EntityDeathEvent

class DeathSystem:
    """Manages entity death and removal based on health or requests."""
    def __init__(self, event_manager: EventManager, entity_manager: EntityManager):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.event_manager.subscribe(RequestEntityRemovalEvent, self.on_removal_request)
    
    def on_removal_request(self, event: RequestEntityRemovalEvent):
        self.entity_manager.remove_entity(event.entity_id)

    def update(self):
        entities_to_remove = []
        for entity, (health,) in self.entity_manager.get_entities_with_components(HealthComponent):
            if health.current_hp <= 0:
                entities_to_remove.append(entity)
        
        for entity in entities_to_remove:
            if self.entity_manager.remove_entity(entity):
                self.event_manager.post(EntityDeathEvent(entity))
                print(f"Entity {entity} has DIED and been removed.")


class LifetimeSystem:
    """Decrements lifetime on components and removes entities when it expires."""
    def __init__(self, event_manager: EventManager, entity_manager: EntityManager):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        
    def update(self, delta_time: float):
        entities = self.entity_manager.get_entities_with_components(LifetimeComponent)
        for entity, (lifetime,) in entities:
            lifetime.time_remaining -= delta_time
            if lifetime.time_remaining <= 0:
                self.event_manager.post(RequestEntityRemovalEvent(entity))