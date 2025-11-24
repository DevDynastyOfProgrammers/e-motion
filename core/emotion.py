from enum import Enum, auto


class Emotion(Enum):
    """
    Defines the possible emotional states for the simulation.
    """

    NEUTRAL = auto()
    JOY = auto()
    ANGER = auto()
    SORROW = auto()
    FEAR = auto()
