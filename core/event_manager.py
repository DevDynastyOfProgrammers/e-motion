from collections import defaultdict

class EventManager:
    """
    Manages event broadcasting and subscription.
    Uses a queue to process events at the end of a game loop cycle.
    """
    def __init__(self):
        # A dictionary mapping event types to a list of handler functions
        self.subscribers = defaultdict(list)
        # A queue of events to be processed in the current frame
        self.event_queue = []

    def subscribe(self, event_type, handler):
        """
        Subscribe a handler function to a specific event type.
        
        :param event_type: The class of the event to subscribe to.
        :param handler: The function or method to be called when the event is posted.
        """
        self.subscribers[event_type].append(handler)

    def unsubscribe(self, event_type, handler):
        """
        Unsubscribe a handler function from an event type.
        """
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)

    def post(self, event):
        """
        Add an event to the processing queue.
        The event will be dispatched to all subscribers during process_events().
        
        :param event: An instance of an Event class.
        """
        self.event_queue.append(event)

    def process_events(self):
        """
        Dispatch all events in the queue to their subscribers.
        This should be called once per game loop iteration.
        """
        # Process a copy of the queue, in case handlers post new events
        queue_to_process = self.event_queue[:]
        self.event_queue.clear()
        
        for event in queue_to_process:
            event_type = type(event)
            if event_type in self.subscribers:
                for handler in self.subscribers[event_type]:
                    handler(event)