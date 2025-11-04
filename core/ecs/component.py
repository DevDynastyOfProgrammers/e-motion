class Component:
    """Base class for all components."""
    pass

class TransformComponent(Component):
    """Holds position, size, and velocity of an entity."""
    def __init__(self, x, y, width, height, velocity=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = velocity

class RenderComponent(Component):
    """Holds rendering information for an entity."""
    def __init__(self, color, layer=0):
        self.color = color
        self.layer = layer

class PlayerInputComponent(Component):
    """Marker component for player-controlled entities"""
    pass

class AIComponent(Component):
    """Marker component for AI-controlled entities"""
    def __init__(self, ai_type="chase_player"):
        self.ai_type = ai_type

class HealthComponent(Component):
    """Holds health information for an entity."""
    def __init__(self, current_hp, max_hp):
        self.current_hp = current_hp
        self.max_hp = max_hp