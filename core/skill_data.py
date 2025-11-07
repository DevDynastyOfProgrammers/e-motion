from dataclasses import dataclass, field
from typing import List, Union, Dict, Any, Optional

@dataclass
class TriggerData:
    """Base class for skill activation triggers."""
    pass

@dataclass
class AutoOnCooldownTriggerData(TriggerData):
    """Trigger type: AUTO_ON_COOLDOWN"""
    pass

@dataclass
class PeriodicTriggerData(TriggerData):
    """Trigger type: PERIODIC"""
    interval: float

AnyTriggerData = Union[AutoOnCooldownTriggerData, PeriodicTriggerData]

@dataclass
class EffectData:
    """
    Base dataclass for all skill effect data.
    Acts as an interface to ensure all effects can be treated polymorphically.
    Every effect 'type' defined in skills.yaml must have a corresponding
    subclass inheriting from this.
    """
    pass

@dataclass
class AreaDamageEffectData(EffectData):
    """
    Data for an effect that deals damage in a radius around the caster.
    Maps to 'type: AREA_DAMAGE' in YAML.
    """
    radius: float
    damage: int
    # TODO: Target by a component type, not by string tag
    # e.g., Target by PlayerInputComponent or AIComponent
    target_tag: str

@dataclass
class SpawnProjectileEffectData(EffectData):
    """
    Data for an effect that spawns a projectile.
    Maps to 'type: SPAWN_PROJECTILE' in YAML.
    """
    projectile_id: str
    target_logic: str

# A type hint for clarity, allowing any of the defined effect data classes.
AnyEffectData = Union[AreaDamageEffectData, SpawnProjectileEffectData]

@dataclass
class ProjectileData:
    """
    Represents the static configuration for a single projectile,
    loaded from the ProjectileDefinitions section of skills.yaml.
    """
    projectile_id: str
    components: Dict[str, Dict[str, Any]] # e.g., {"Transform": {"width": 10, ...}}

@dataclass
class SkillData:
    """
    Represents the static, configuration data for a single skill.
    """
    skill_id: str
    cooldown: float = 0.0
    trigger: Optional[AnyTriggerData] = None
    effects: List[AnyEffectData] = field(default_factory=list)