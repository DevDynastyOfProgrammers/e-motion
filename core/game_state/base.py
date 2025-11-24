from typing import Protocol, List, Any, runtime_checkable
import pygame

@runtime_checkable
class GameState(Protocol):
    """
    Protocol defining the strict interface for all game states.
    Using Protocol allows for structural subtyping and better static analysis
    compared to abstract base classes.
    """
    
    def handle_events(self, events: List[pygame.event.Event]) -> None:
        """
        Process raw PyGame events.
        """
        ...

    def update(self, delta_time: float) -> None:
        """
        Update the state logic.
        :param delta_time: Time elapsed since the last frame in seconds.
        """
        ...

    def draw(self, screen: pygame.Surface) -> None:
        """
        Render the state content to the screen.
        :param screen: The main PyGame display surface.
        """
        ...