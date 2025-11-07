class Event:
    """
    Base class for all events.
    """
    pass

class EntityDeathEvent(Event):
    """
    Event broadcast when an entity's health reaches zero.
    """
    def __init__(self, entity_id):
        self.entity_id = entity_id

class PlayerMoveIntentEvent(Event):
    """
    Event broadcast by the input system when the player intends to move.
    Carries a direction vector.
    """
    def __init__(self, entity_id, direction):
        self.entity_id = entity_id
        # A tuple like (dx, dy), e.g., (1, 0) for right
        self.direction = direction