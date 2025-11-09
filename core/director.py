from enum import IntEnum
from typing import List
from core.utils.lerp import lerp

class GameStateVector(IntEnum):
    """
    Defines the fixed indices for parameters within the game state vector.
    This ensures type-safe and readable access to vector elements.
    The order matches the MVP specification.
    """
    SPAWN_RATE_MULTIPLIER = 0
    ENEMY_SPEED_MULTIPLIER = 1
    ENEMY_HEALTH_MULTIPLIER = 2
    ENEMY_DAMAGE_MULTIPLIER = 3
    PLAYER_SPEED_MULTIPLIER = 4
    PLAYER_DAMAGE_MULTIPLIER = 5
    ITEM_DROP_CHANCE_MODIFIER = 6

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
        # The vector now has a fixed size of 7 for the MVP.
        self._vector_size = len(GameStateVector)
        
        # The state currently being used by game systems.
        # Defaults to a neutral state (all multipliers are 1.0).
        self._current_state_vector: List[float] = [1.0] * self._vector_size
        
        # The target state received from the ML model.
        self._target_state_vector: List[float] = [1.0] * self._vector_size
        
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
        if len(vector) == self._vector_size:
            print(f"Director received new target vector: {vector}")
            self._target_state_vector = vector
        else:
            print(f"WARNING: GameDirector received a vector of invalid size. "
                  f"Expected {self._vector_size}, got {len(vector)}.")

    # --- Public Getters for Systems ---

    def get_spawn_rate_multiplier(self) -> float:
        return self._current_state_vector[GameStateVector.SPAWN_RATE_MULTIPLIER]

    def get_enemy_speed_multiplier(self) -> float:
        return self._current_state_vector[GameStateVector.ENEMY_SPEED_MULTIPLIER]

    def get_enemy_health_multiplier(self) -> float:
        return self._current_state_vector[GameStateVector.ENEMY_HEALTH_MULTIPLIER]
    
    def get_enemy_damage_multiplier(self) -> float:
        return self._current_state_vector[GameStateVector.ENEMY_DAMAGE_MULTIPLIER]

    def get_player_speed_multiplier(self) -> float:
        return self._current_state_vector[GameStateVector.PLAYER_SPEED_MULTIPLIER]

    def get_player_damage_multiplier(self) -> float:
        return self._current_state_vector[GameStateVector.PLAYER_DAMAGE_MULTIPLIER]

    def get_item_drop_chance_modifier(self) -> float:
        return self._current_state_vector[GameStateVector.ITEM_DROP_CHANCE_MODIFIER]
    
    # --- Getters for Debug UI ---
    
    def get_current_vector(self) -> List[float]:
        return self._current_state_vector
    
    def get_target_vector(self) -> List[float]:
        return self._target_state_vector