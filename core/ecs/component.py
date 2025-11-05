class Component:
    """Base class for all components"""
    pass

class TransformComponent(Component):
    """Holds position, size, and velocity of an entity"""
    def __init__(self, x, y, width, height, velocity=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = velocity

class RenderComponent(Component):
    """Holds rendering information for an entity"""
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
    """Holds health information for an entity"""
    def __init__(self, current_hp, max_hp):
        self.current_hp = current_hp
        self.max_hp = max_hp

class TagComponent(Component):
    """Marker component for tagging entities (e.g., 'enemy', 'player')"""
    def __init__(self, tag):
        self.tag = tag

class SpellAuraComponent(Component):
    """Holds aura spell information for an entity"""
    def __init__(self, radius: float, damage: int, tick_rate: float, target_tag: str):
        self.radius = radius   
        self.damage = damage
        self.tick_rate = tick_rate
        self.target_tag = target_tag # Target to attack
        self.time_since_last_tick = 0.0