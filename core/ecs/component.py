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

class SkillSetComponent(Component):
    """Holds the runtime state of an entity's skills."""
    def __init__(self, skill_ids: list):
        self.skills = skill_ids
        
        # Tracks the current cooldown for each skill.
        self.cooldowns = {skill_id: 0.0 for skill_id in skill_ids}

        # Tracks individual timers for periodic triggers.
        self.periodic_timers = {skill_id: 0.0 for skill_id in skill_ids}

class ProjectileComponent(Component):
    """
    Marker component for projectiles. Stores its movement direction.
    """
    def __init__(self, direction_x, direction_y):
        self.dx = direction_x
        self.dy = direction_y

class DamageOnCollisionComponent(Component):
    """
    Data component for entities that deal damage on collision (like projectiles).
    """
    def __init__(self, damage: int, target_tag: str):
        self.damage = damage
        self.target_tag = target_tag

class LifetimeComponent(Component):
    """
    Gives an entity a limited time to live.
    """
    def __init__(self, duration: float):
        self.time_remaining = duration