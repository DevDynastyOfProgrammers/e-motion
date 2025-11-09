# core/director.py

from enum import IntEnum
from typing import List
from core.utils.lerp import lerp

class GameStateVector(IntEnum):
    """
    Defines the fixed indices for parameters within the game state vector.
    This ensures type-safe and readable access to vector elements.
    """
    SPAWN_RATE_MULTIPLIER = 0
    ENEMY_SPEED_MULTIPLIER = 1
    PLAYER_SPEED_MULTIPLIER = 2
    # --- Future MVP parameters can be added here ---
    # ENEMY_HEALTH_MULTIPLIER = 3
    # ENEMY_DAMAGE_MULTIPLIER = 4
    # etc.

class GameDirector:
    """
    Manages the dynamic state of the game, influenced by ML model predictions.
    It holds the current and target state vectors and smoothly interpolates
    between them over time.
    """
    def __init__(self, smoothing_factor: float = 0.5):
        """
        Initializes the GameDirector.

        :param smoothing_factor: How quickly the current state approaches the
                                 target state. Higher is faster.
        """
        # For the PoC, the vector has a fixed size of 3.
        # [spawn_rate, enemy_speed, player_speed]
        self._poc_vector_size = 3
        
        # The state currently being used by game systems.
        # Defaults to a neutral state (all multipliers are 1.0).
        self._current_state_vector: List[float] = [1.0] * self._poc_vector_size
        
        # The target state received from the ML model.
        # Also defaults to neutral.
        self._target_state_vector: List[float] = [1.0] * self._poc_vector_size
        
        self._smoothing_factor = smoothing_factor

    def update(self, delta_time: float):
        """
        Smoothly interpolates the current state vector towards the target vector.
        This method should be called once per frame.
        """
        for i in range(len(self._current_state_vector)):
            self._current_state_vector[i] = lerp(
                self._current_state_vector[i],
                self._target_state_vector[i],
                self._smoothing_factor * delta_time
            )

    def set_new_target_vector(self, vector: List[float]):
        """
        Sets a new target state vector, typically received from the ML model.
        Performs validation to ensure the vector has the correct dimensions.
        """
        if len(vector) == self._poc_vector_size:
            print(f"Director received new target vector: {vector}")
            self._target_state_vector = vector
        else:
            print(f"WARNING: GameDirector received a vector of invalid size. "
                  f"Expected {self._poc_vector_size}, got {len(vector)}.")

    # --- Public Getters for Systems ---

    def get_spawn_rate_multiplier(self) -> float:
        """Returns the current, interpolated spawn rate multiplier."""
        return self._current_state_vector[GameStateVector.SPAWN_RATE_MULTIPLIER]

    def get_enemy_speed_multiplier(self) -> float:
        """Returns the current, interpolated enemy speed multiplier."""
        return self._current_state_vector[GameStateVector.ENEMY_SPEED_MULTIPLIER]

    def get_player_speed_multiplier(self) -> float:
        """Returns the current, interpolated player speed multiplier."""
        return self._current_state_vector[GameStateVector.PLAYER_SPEED_MULTIPLIER]