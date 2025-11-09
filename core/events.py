from core.skill_data import AnyEffectData
from core.emotion import Emotion

class Event:
    """
    Base class for all events.
    """
    pass

class EmotionStateChangedEvent(Event):
    """
    Broadcast by the emotion recognition model simulator when a new
    emotional state is detected.
    """
    def __init__(self, new_emotion: Emotion):
        self.emotion = new_emotion

class EntityDeathEvent(Event):
    """
    Event broadcast when an entity's health reaches zero.
    """
    def __init__(self, entity_id):
        self.entity_id = entity_id

class PlayerMoveIntentEvent(Event):
    """
    Event broadcast by the input system when the player intends to move.
    Carries a direction vector.
    """
    def __init__(self, entity_id, direction):
        self.entity_id = entity_id
        self.direction = direction

class RequestSkillActivationEvent(Event):
    """Broadcast when an entity wants to activate a skill."""
    def __init__(self, entity_id, skill_id):
        self.entity_id = entity_id
        self.skill_id = skill_id

# --- Skill Effect Events ---

class ApplyAreaDamageEvent(Event):
    """Broadcast to apply area damage."""
    def __init__(self, caster_id, caster_pos, effect_data):
        self.caster_id = caster_id
        self.caster_pos = caster_pos
        self.effect_data: AnyEffectData = effect_data

class SpawnProjectileEvent(Event):
    """Broadcast to spawn a projectile."""
    def __init__(self, caster_id, effect_data):
        self.caster_id = caster_id
        self.effect_data: AnyEffectData = effect_data

# --- Damage & Removal Events ---

class ApplyDirectDamageEvent(Event):
    """Broadcast to apply damage to a single, specific entity."""
    def __init__(self, caster_id, target_id, damage):
        self.caster_id = caster_id # ID to identify the damage source
        self.target_id = target_id
        self.damage = damage

class RequestEntityRemovalEvent(Event):
    """Broadcast to request the removal of an entity (e.g., a projectile)."""
    def __init__(self, entity_id):
        self.entity_id = entity_id