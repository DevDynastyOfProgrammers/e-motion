from dataclasses import dataclass

from loguru import logger

from core.ecs.component import HealthComponent, LifetimeComponent, PlayerInputComponent
from core.ecs.entity import EntityManager
from core.event_manager import EventManager
from core.events import EntityDeathEvent, RequestEntityRemovalEvent


@dataclass
class DeathSystem:
    """Manages entity death and removal based on health or requests."""

    event_manager: EventManager
    entity_manager: EntityManager

    def __post_init__(self) -> None:
        self.event_manager.subscribe(RequestEntityRemovalEvent, self.on_removal_request)

    def on_removal_request(self, event: RequestEntityRemovalEvent) -> None:
        """Обрабатывает запросы на удаление сущностей."""
        self.entity_manager.remove_entity(event.entity_id)

    def update(self) -> None:
        """Проверяет здоровье сущностей и удаляет тех, у кого здоровье меньше или равно 0."""
        entities_to_remove = self._get_entities_to_remove()

        for entity in entities_to_remove:
            if self.entity_manager.get_component(entity, PlayerInputComponent):
                logger.info(f'GAME OVER: Player (Entity {entity}) has died!')

        for entity in entities_to_remove:
            if self.entity_manager.remove_entity(entity):
                self.event_manager.post(EntityDeathEvent(entity))
                logger.debug(f'Entity {entity} has DIED and been removed.')

    def _get_entities_to_remove(self) -> list[int]:
        """Возвращает список сущностей, которые должны быть удалены (здоровье <= 0)."""
        entities_to_remove = []
        for entity, (health,) in self.entity_manager.get_entities_with_components(HealthComponent):
            if health.current_hp <= 0:
                entities_to_remove.append(entity)
        return entities_to_remove


@dataclass
class LifetimeSystem:
    """Decrements lifetime on components and removes entities when it expires."""

    event_manager: EventManager
    entity_manager: EntityManager

    def update(self, delta_time: float) -> None:
        """Обновляет время жизни сущностей и удаляет их по истечении времени."""
        entities = self.entity_manager.get_entities_with_components(LifetimeComponent)
        for entity, (lifetime,) in entities:
            lifetime.duration -= delta_time
            if lifetime.duration <= 0:
                self.event_manager.post(RequestEntityRemovalEvent(entity))
                logger.debug(f'Entity {entity} has expired and is scheduled for removal.')
