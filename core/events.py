from dataclasses import dataclass

from core.emotion import EmotionPrediction
from core.skill_data import AnyEffectData


@dataclass(frozen=True)
class Event:
    """Base immutable class for all events."""


# --- ML & Input Events ---


@dataclass(frozen=True)
class EmotionStateChangedEvent(Event):
    """Broadcast when the vision model produces a new prediction."""

    prediction: EmotionPrediction


@dataclass(frozen=True)
class PlayerMoveIntentEvent(Event):
    """Broadcast when the player presses movement keys."""

    entity_id: int
    direction: tuple[float, float]


@dataclass(frozen=True)
class RequestSkillActivationEvent(Event):
    """Broadcast when a skill trigger condition is met."""

    entity_id: int
    skill_id: str


# --- Combat & Physics Events ---


@dataclass(frozen=True)
class ApplyAreaDamageEvent(Event):
    """Request to apply damage in a radius."""

    caster_id: int
    caster_pos: tuple[float, float]
    effect_data: AnyEffectData


@dataclass(frozen=True)
class ApplyDirectDamageEvent(Event):
    """Request to apply damage to a specific target."""

    caster_id: int
    target_id: int
    damage: int


@dataclass(frozen=True)
class SpawnProjectileEvent(Event):
    """Request to spawn a projectile entity."""

    caster_id: int
    effect_data: AnyEffectData


# --- Lifecycle Events ---


@dataclass(frozen=True)
class RequestEntityRemovalEvent(Event):
    """Request to safely remove an entity from the manager."""

    entity_id: int


@dataclass(frozen=True)
class EntityDeathEvent(Event):
    """Notification that an entity has been removed due to death/destruction."""

    entity_id: int
