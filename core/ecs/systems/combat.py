import math
from loguru import logger
from core.ecs.entity import EntityManager
from core.ecs.component import (
    HealthComponent,
    TransformComponent,
    PlayerInputComponent,
    AIComponent,
    ProjectileComponent,
    DamageOnCollisionComponent,
)
from core.event_manager import EventManager
from core.director import GameDirector
from core.events import ApplyAreaDamageEvent, ApplyDirectDamageEvent, RequestEntityRemovalEvent
from core.skill_data import AreaDamageEffectData


class DamageSystem:
    """Listens for damage events and applies them, considering multipliers."""

    def __init__(
        self, event_manager: EventManager, entity_manager: EntityManager, director: GameDirector
    ) -> None:
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.director = director
        self.event_manager.subscribe(ApplyAreaDamageEvent, self.on_area_damage)
        self.event_manager.subscribe(ApplyDirectDamageEvent, self.on_direct_damage)

    def _get_damage_multiplier(self, caster_id: int) -> float:
        if self.entity_manager.get_component(caster_id, PlayerInputComponent):
            return self.director.state.player_damage_multiplier
        elif self.entity_manager.get_component(caster_id, AIComponent):
            return self.director.state.enemy_damage_multiplier
        return 1.0
    
    def _is_valid_target(self, entity_id: int, target_group: str) -> bool:
        """
        Determines if an entity matches the target group based on its components.
        Replaces TagComponent logic.
        """
        if target_group == "enemy":
            # Target is enemy if it has AIComponent
            return self.entity_manager.get_component(entity_id, AIComponent) is not None
        elif target_group == "player":
            # Target is player if it has PlayerInputComponent
            return self.entity_manager.get_component(entity_id, PlayerInputComponent) is not None
        return False

    def on_area_damage(self, event: ApplyAreaDamageEvent) -> None:
        effect_data = event.effect_data
        
        if not isinstance(effect_data, AreaDamageEffectData):
            return 
            
        caster_pos = event.caster_pos
        
        base_damage = effect_data.damage
        multiplier = self._get_damage_multiplier(event.caster_id)
        final_damage = int(base_damage * multiplier)
        if final_damage <= 0: return

        targets = self.entity_manager.get_entities_with_components(HealthComponent, TransformComponent)
        for target_id, (health, transform) in targets:
            if target_id == event.caster_id: continue
            
            if self._is_valid_target(target_id, effect_data.target_group):
                distance = math.hypot(caster_pos[0] - transform.x, caster_pos[1] - transform.y)
                if distance <= effect_data.radius:
                    health.current_hp -= final_damage
                    logger.debug(f"Entity {target_id} took {final_damage} AREA damage! HP: {health.current_hp}") 

    def on_direct_damage(self, event: ApplyDirectDamageEvent) -> None:
        health = self.entity_manager.get_component(event.target_id, HealthComponent)
        if health:
            base_damage = event.damage
            multiplier = self._get_damage_multiplier(event.caster_id)
            final_damage = int(base_damage * multiplier)
            if final_damage <= 0:
                return

            health.current_hp -= final_damage
            logger.debug(f"Entity {event.target_id} took {final_damage} DIRECT damage! HP: {health.current_hp}")


class ProjectileImpactSystem:
    """Checks for projectile collisions and posts damage events."""

    def __init__(self, event_manager: EventManager, entity_manager: EntityManager) -> None:
        self.event_manager = event_manager
        self.entity_manager = entity_manager

    def _is_valid_target(self, entity_id: int, target_group: str) -> bool:
        """Duplicated logic for now, could be moved to a static helper or Entity utils."""
        if target_group == "enemy":
            return self.entity_manager.get_component(entity_id, AIComponent) is not None
        elif target_group == "player":
            return self.entity_manager.get_component(entity_id, PlayerInputComponent) is not None
        return False
    
    def update(self) -> None:
        projectiles = list(
            self.entity_manager.get_entities_with_components(
                ProjectileComponent, TransformComponent, DamageOnCollisionComponent
            )
        )
        targets = list(
            self.entity_manager.get_entities_with_components(
                HealthComponent, TransformComponent
            )
        )

        for proj_id, (proj, proj_trans, proj_damage) in projectiles:
            for target_id, (health, target_trans) in targets:
                
                if not self._is_valid_target(target_id, proj_damage.target_group):
                    continue

                # Simple collision check
                if math.hypot(proj_trans.x - target_trans.x, proj_trans.y - target_trans.y) < target_trans.width:
                    self.event_manager.post(ApplyDirectDamageEvent(proj.caster_id, target_id, proj_damage.damage))
                    self.event_manager.post(RequestEntityRemovalEvent(proj_id))
                    break