"""
Game Design Data: Maps Logic States ("TENSION") to Gameplay Multipliers.
"""
from dataclasses import dataclass
from typing import Dict, List, Final

@dataclass(frozen=True)
class PresetParameters:
    """Detailed gameplay multipliers for a specific state."""
    spawn_rate_multiplier: float
    enemy_speed_multiplier: float
    enemy_health_multiplier: float
    enemy_damage_multiplier: float
    player_speed_multiplier: float
    player_damage_multiplier: float
    item_drop_chance_modifier: float
    
    # Metadata for UI/Debug
    difficulty_level: str
    description: str

class PresetMapping:
    """Static repository of Game Balance Presets."""

    # 1. Group Mappings (Granular -> Broad)
    PRESET_GROUPS: Final[Dict[str, str]] = {
        "god_mode": "RELAX",
        "walk_in_the_park": "RELAX",
        "beginner": "RELAX",
        "standard": "FLOW",
        "challenge": "FLOW",
        "survival_horror": "TENSION",
        "nightmare": "TENSION",
        "hardcore": "OVERLOAD",
        "impossible": "OVERLOAD",
        "bullet_heaven": "ACTION",
    }

    # 2. The Core Definitions (The "Truth" of Game Balance)
    PRESET_DEFINITIONS: Final[Dict[str, PresetParameters]] = {
        "RELAX": PresetParameters(
            spawn_rate_multiplier=-0.5,
            enemy_speed_multiplier=-0.5,
            enemy_health_multiplier=-0.5,
            enemy_damage_multiplier=-0.5,
            player_speed_multiplier=0.3,
            player_damage_multiplier=0.5,
            item_drop_chance_modifier=0.3,
            difficulty_level="easy",
            description="Relaxed State: Fewer, weaker enemies."
        ),
        "FLOW": PresetParameters(
            spawn_rate_multiplier=0.0,
            enemy_speed_multiplier=0.0,
            enemy_health_multiplier=0.0,
            enemy_damage_multiplier=0.0,
            player_speed_multiplier=0.0,
            player_damage_multiplier=0.0,
            item_drop_chance_modifier=0.0,
            difficulty_level="medium",
            description="Flow State: Balanced gameplay."
        ),
        "TENSION": PresetParameters(
            spawn_rate_multiplier=0.2,
            enemy_speed_multiplier=0.0,
            enemy_health_multiplier=0.8,
            enemy_damage_multiplier=0.8,
            player_speed_multiplier=-0.2,
            player_damage_multiplier=-0.3,
            item_drop_chance_modifier=-0.5,
            difficulty_level="hard",
            description="Tension State: High stakes, tough enemies."
        ),
        "OVERLOAD": PresetParameters(
            spawn_rate_multiplier=0.3,
            enemy_speed_multiplier=0.6,
            enemy_health_multiplier=0.5,
            enemy_damage_multiplier=1.0,
            player_speed_multiplier=0.1,
            player_damage_multiplier=-0.5,
            item_drop_chance_modifier=-0.8,
            difficulty_level="extreme",
            description="Overload: Run for your life."
        ),
        "ACTION": PresetParameters(
            spawn_rate_multiplier=1.0,
            enemy_speed_multiplier=0.8,
            enemy_health_multiplier=0.9,
            enemy_damage_multiplier=0.9,
            player_speed_multiplier=0.5,
            player_damage_multiplier=0.8,
            item_drop_chance_modifier=0.7,
            difficulty_level="chaotic",
            description="Action: Pure chaos."
        ),
    }

    PRESETS_BY_DIFFICULTY: Final[List[str]] = [
        "RELAX", 
        "FLOW", 
        "TENSION", 
        "OVERLOAD", 
        "ACTION"
    ]

    @classmethod
    def remap_preset(cls, preset_name: str) -> str:
        """
        Maps granular names (e.g. 'nightmare') to group names (e.g. 'TENSION').
        Returns the original name if no mapping exists.
        """
        return cls.PRESET_GROUPS.get(preset_name, preset_name)

    @classmethod
    def get_preset_multipliers(cls, preset_name: str) -> List[float]:
        """
        Returns list of multipliers in specific order.
        Used by StateDirector to construct GameStateVector.
        """
        # Remap using the helper method
        group_name = cls.remap_preset(preset_name)
        
        # Fallback to FLOW if unknown
        params = cls.PRESET_DEFINITIONS.get(group_name, cls.PRESET_DEFINITIONS["FLOW"])
        
        return [
            params.spawn_rate_multiplier,
            params.enemy_speed_multiplier,
            params.enemy_health_multiplier,
            params.enemy_damage_multiplier,
            params.player_speed_multiplier,
            params.player_damage_multiplier,
            params.item_drop_chance_modifier,
        ]