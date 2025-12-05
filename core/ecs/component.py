from dataclasses import dataclass


# Base class for all components
class Component:
    """Base class for all components"""


@dataclass
class TransformComponent(Component):
    """Holds position, size, and velocity of an entity"""

    x: float
    y: float
    width: float
    height: float
    velocity: float = 0.0


@dataclass
class RenderComponent(Component):
    """Holds rendering information for an entity"""

    color: str
    layer: int = 0


@dataclass
class PlayerInputComponent(Component):
    """Marker component for player-controlled entities"""


@dataclass
class AIComponent(Component):
    """Marker component for AI-controlled entities"""

    ai_type: str = 'chase_player'


@dataclass
class HealthComponent(Component):
    """Holds health information for an entity"""

    current_hp: int
    max_hp: int


class SkillSetComponent(Component):
    """Holds the runtime state of an entity's skills."""

    def __init__(self, skill_ids: list) -> None:
        self.skills = skill_ids

        # Tracks the current cooldown for each skill.
        self.cooldowns = {skill_id: 0.0 for skill_id in skill_ids}

        # Tracks individual timers for periodic triggers.
        self.periodic_timers = {skill_id: 0.0 for skill_id in skill_ids}


@dataclass
class ProjectileComponent(Component):
    """
    Marker component for projectiles. Stores its movement direction and caster.
    """

    dx: float
    dy: float
    caster_id: int  # ID to trace damage source


@dataclass
class DamageOnCollisionComponent(Component):
    """
    Data component for entities that deal damage on collision (like projectiles).
    """

    damage: int
    target_group: str


@dataclass
class LifetimeComponent(Component):
    """
    Gives an entity a limited time to live.
    """

    duration: float


COMPOMENTS = (
    TransformComponent
    | RenderComponent
    | PlayerInputComponent
    | AIComponent
    | HealthComponent
    | SkillSetComponent
    | ProjectileComponent
    | DamageOnCollisionComponent
    | LifetimeComponent
)
