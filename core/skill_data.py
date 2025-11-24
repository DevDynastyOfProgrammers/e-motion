from dataclasses import dataclass, field


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


AnyTriggerData = AutoOnCooldownTriggerData | PeriodicTriggerData


@dataclass
class EffectData:
    """
    Base dataclass for all skill effect data.
    """

    pass


@dataclass
class AreaDamageEffectData(EffectData):
    """
    Data for an effect that deals damage in a radius around the caster.
    """

    radius: float
    damage: int
    target_tag: str


@dataclass
class SpawnProjectileEffectData(EffectData):
    """
    Data for an effect that spawns a projectile.
    """

    projectile_id: str
    target_logic: str


AnyEffectData = AreaDamageEffectData | SpawnProjectileEffectData


@dataclass
class ProjectileData:
    """
    Represents the static configuration for a single projectile.
    """

    projectile_id: str
    components: dict[str, dict[str, object]]


@dataclass
class SkillData:
    """
    Represents the static, configuration data for a single skill.
    """

    skill_id: str
    cooldown: float = 0.0
    trigger: AnyTriggerData | None = None
    effects: list[AnyEffectData] = field(default_factory=list)
