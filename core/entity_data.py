from dataclasses import dataclass, field


@dataclass
class EntityTransformData:
    width: float
    height: float
    velocity: float


@dataclass
class EntityData:
    """
    Represents the static configuration for a game entity (Player, Enemy, etc.)
    """

    id: str
    transform: EntityTransformData
    max_hp: int
    color: tuple[int, int, int]

    # List of extra marker components like 'PlayerInput' or 'AI'
    components: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
