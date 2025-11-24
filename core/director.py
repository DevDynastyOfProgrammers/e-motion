from dataclasses import dataclass, fields
from core.utils.lerp import lerp

@dataclass
class GameStateVector:
    """
    Defines the game state parameters controlled by the Director.
    All values are multipliers (1.0 = normal).
    """
    spawn_rate_multiplier: float = 1.0
    enemy_speed_multiplier: float = 1.0
    enemy_health_multiplier: float = 1.0
    enemy_damage_multiplier: float = 1.0
    player_speed_multiplier: float = 1.0
    player_damage_multiplier: float = 1.0
    item_drop_chance_modifier: float = 1.0

class GameDirector:
    """
    Manages the dynamic state of the game.
    Interpolates between GameStateVector objects.
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
        """Read-only access to the target state (for debug)."""
        return self._target_state

    def update(self, delta_time: float) -> None:
        """Smoothly interpolates the current state towards the target state."""
        factor = self._smoothing_factor * delta_time
        
        for field in fields(GameStateVector):
            name = field.name
            current_val = getattr(self._current_state, name)
            target_val = getattr(self._target_state, name)
            
            # Interpolate
            setattr(self._current_state, name, lerp(current_val, target_val, factor))

    def set_new_target_vector(self, vector: GameStateVector) -> None:
        if isinstance(vector, GameStateVector):
            self._target_state = vector
        else:
            print(f"WARNING: Director received invalid state object: {type(vector)}")