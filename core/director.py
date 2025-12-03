import logging
from dataclasses import dataclass, fields
from typing import Dict, Any

from core.utils.lerp import lerp

logger = logging.getLogger("core.director")

@dataclass
class GameStateVector:
    """
    Data Object representing the dynamic difficulty parameters.
    Values are multipliers (1.0 = standard gameplay).
    """
    spawn_rate_multiplier: float = 1.0
    enemy_speed_multiplier: float = 1.0
    enemy_health_multiplier: float = 1.0
    enemy_damage_multiplier: float = 1.0
    player_speed_multiplier: float = 1.0
    player_damage_multiplier: float = 1.0
    item_drop_chance_modifier: float = 1.0

    def clamp(self, min_val: float = 0.1, max_val: float = 5.0) -> None:
        """Safeguard to prevent game breaking values."""
        for field in fields(self):
            val = getattr(self, field.name)
            clamped = max(min_val, min(val, max_val))
            setattr(self, field.name, clamped)


class GameDirector:
    """
    Manages the game difficulty by interpolating between the current state
    and a target state determined by the ML Engine.
    """

    def __init__(self, smoothing_factor: float = 0.5) -> None:
        self._current_state = GameStateVector()
        self._target_state = GameStateVector()
        self._smoothing_factor = smoothing_factor

    @property
    def state(self) -> GameStateVector:
        """Read-only access to the current interpolated state."""
        return self._current_state

    @property
    def target_state(self) -> GameStateVector:
        """Read-only access to the target state (for debugging)."""
        return self._target_state

    def update(self, delta_time: float) -> None:
        """
        Smoothly interpolates the current state towards the target state using Lerp.
        Should be called every frame.
        """
        # Clamp delta_time factor to avoid instant jumps on lag spikes
        raw_factor = self._smoothing_factor * delta_time
        factor = max(0.0, min(raw_factor, 1.0))

        for field in fields(GameStateVector):
            name = field.name
            current_val = getattr(self._current_state, name)
            target_val = getattr(self._target_state, name)
            
            # Linear Interpolation
            new_val = lerp(current_val, target_val, factor)
            setattr(self._current_state, name, new_val)

    def set_new_target_vector(self, vector: GameStateVector) -> None:
        """Updates the target state. The update() loop will interpolate towards this."""
        if not isinstance(vector, GameStateVector):
            logger.warning(f"Invalid state object received: {type(vector)}")
            return
            
        # Ensure the model didn't output crazy values (e.g. negative health)
        vector.clamp()
        self._target_state = vector

    def get_debug_info(self) -> Dict[str, float]:
        """Returns current multipliers for UI debugging."""
        return {f.name: getattr(self._current_state, f.name) for f in fields(GameStateVector)}