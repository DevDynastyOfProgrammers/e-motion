class BaseState:
    """
    Base class for all game states (scenes).
    Defines the interface that each child class must implement.
    """
    def __init__(self, state_manager):
        self.state_manager = state_manager

    def handle_events(self, events):
        """Processes events"""
        raise NotImplementedError

    def update(self, delta_time):
        """Updates the state logic"""
        raise NotImplementedError

    def draw(self, screen):
        """Renders the state to the screen"""
        raise NotImplementedError