from typing import Dict

from loguru import logger

from core.ecs.component import SkillSetComponent, TransformComponent
from core.ecs.entity import EntityManager
from core.event_manager import EventManager
from core.events import ApplyAreaDamageEvent, RequestSkillActivationEvent, SpawnProjectileEvent
from core.skill_data import (
    AreaDamageEffectData,
    AutoOnCooldownTriggerData,
    PeriodicTriggerData,
    SkillData,
    SpawnProjectileEffectData,
)


class SkillSystem:
    """Manages skill state (cooldowns, timers) and checks triggers."""

    def __init__(
        self,
        event_manager: EventManager,
        entity_manager: EntityManager,
        skill_definitions: dict[str, SkillData],
    ) -> None:
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.skill_definitions = skill_definitions

    def update(self, delta_time: float) -> None:
        entities = self.entity_manager.get_entities_with_components(SkillSetComponent)
        for entity, (skill_set,) in entities:
            # Update timers
            for skill_id in skill_set.skills:
                if skill_set.cooldowns[skill_id] > 0:
                    skill_set.cooldowns[skill_id] -= delta_time
                skill_set.periodic_timers[skill_id] += delta_time

            # Check triggers
            for skill_id in skill_set.skills:
                skill_data = self.skill_definitions.get(skill_id)
                if not skill_data or not skill_data.trigger:
                    continue

                if skill_set.cooldowns[skill_id] > 0:
                    continue

                trigger_data = skill_data.trigger
                should_activate = False

                if isinstance(trigger_data, AutoOnCooldownTriggerData):
                    should_activate = True
                elif isinstance(trigger_data, PeriodicTriggerData):
                    if skill_set.periodic_timers[skill_id] >= trigger_data.interval:
                        should_activate = True

                if should_activate:
                    self.event_manager.post(RequestSkillActivationEvent(entity, skill_id))
                    if isinstance(trigger_data, PeriodicTriggerData):
                        skill_set.periodic_timers[skill_id] = 0.0


class SkillExecutionSystem:
    """Listens for skill activation requests and executes their effects."""

    def __init__(
        self,
        event_manager: EventManager,
        entity_manager: EntityManager,
        skill_definitions: dict[str, SkillData],
    ) -> None:
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.skill_definitions = skill_definitions
        self.event_manager.subscribe(RequestSkillActivationEvent, self.on_skill_request)

    def on_skill_request(self, event: RequestSkillActivationEvent) -> None:
        skill_set = self.entity_manager.get_component(event.entity_id, SkillSetComponent)
        transform = self.entity_manager.get_component(event.entity_id, TransformComponent)
        skill_data = self.skill_definitions.get(event.skill_id)

        if skill_set is None or transform is None or skill_data is None:
            return

        if skill_set.cooldowns.get(event.skill_id, 0) > 0:
            return

        skill_set.cooldowns[event.skill_id] = skill_data.cooldown
        logger.debug(
            f"Entity {event.entity_id} activated skill '{event.skill_id}' with cooldown {skill_data.cooldown}."
        )

        for effect in skill_data.effects:
            if isinstance(effect, AreaDamageEffectData):
                self.event_manager.post(
                    ApplyAreaDamageEvent(
                        caster_id=event.entity_id,
                        caster_pos=(transform.x, transform.y),
                        effect_data=effect,
                    )
                )
            elif isinstance(effect, SpawnProjectileEffectData):
                self.event_manager.post(SpawnProjectileEvent(caster_id=event.entity_id, effect_data=effect))
