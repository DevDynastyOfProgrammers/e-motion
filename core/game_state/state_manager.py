class GameStateManager:
    """
    Manages the stack of game states.
    Allows switching between different screens (menu, gameplay, pause)
    """
    def __init__(self):
        # Stack of game states
        self.states = []

    def push_state(self, state):
        """Add new state to the top of the stack (makes it active)"""
        self.states.append(state)

    def pop_state(self):
        """Delete the top state from the stack"""
        if self.states:
            self.states.pop()

    def get_current_state(self):
        """Returns the current active state (top of the stack)"""
        if self.states:
            return self.states[-1]
        return None