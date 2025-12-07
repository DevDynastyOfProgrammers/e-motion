from core.game_state.gameplay_state import GameplayState


class GameStateManager:
    """
    Manages the stack of game states.
    Allows switching between different screens (menu, gameplay, pause)
    """

    def __init__(self) -> None:
        # Stack of game states
        self.states: list[GameplayState] = []

    def push_state(self, state: list[GameplayState]) -> None:
        """Add new state to the top of the stack (makes it active)"""
        self.states.append(state)

    def pop_state(self) -> None:
        """Delete the top state from the stack"""
        if self.states:
            self.states.pop()

    def get_current_state(self) -> GameplayState | None:
        """Returns the current active state (top of the stack)"""
        return self.states[-1] if self.states else None
