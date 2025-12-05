from collections import defaultdict
from typing import Any, Callable, Type

from core.events import Event

# Define a type alias for event handlers
# Handler receives any Event (or subclass) and returns None
HandlerType = Callable[[Any], None]


class EventManager:
    """
    Manages event broadcasting and subscription.
    """

    def __init__(self) -> None:
        # Modern typing for dict and list
        self.subscribers: dict[Type[Event], list[HandlerType]] = defaultdict(list)
        self.event_queue: list[Event] = []

    def subscribe(self, event_type: Type[Event], handler: HandlerType) -> None:
        """
        Subscribe a handler function to a specific event type.
        """
        self.subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[Event], handler: HandlerType) -> None:
        """
        Unsubscribe a handler function from an event type.
        """
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)

    def post(self, event: Event) -> None:
        """
        Add an event to the processing queue.
        """
        self.event_queue.append(event)

    def process_events(self) -> None:
        """
        Dispatch all events in the queue to their subscribers.
        """
        queue_to_process = self.event_queue[:]
        self.event_queue.clear()

        for event in queue_to_process:
            event_type = type(event)
            # Find subscribers for this exact event type
            if event_type in self.subscribers:
                for handler in self.subscribers[event_type]:
                    handler(event)
