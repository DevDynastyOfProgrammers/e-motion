"""
Comprehensive preset mapping and configuration for game balancing system.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class PresetParameters:
    """Detailed parameters for each game preset."""

    # Game balance multipliers
    spawn_rate_multiplier: float
    enemy_speed_multiplier: float
    enemy_health_multiplier: float
    enemy_damage_multiplier: float
    player_speed_multiplier: float
    player_damage_multiplier: float
    item_drop_chance_modifier: float

    # Metadata
    difficulty_level: str  # easy, medium, hard, etc.
    description: str
    tags: list[str] = field(default_factory=list)


class PresetMapping:
    """Comprehensive mapping between preset names and their configurations."""

    # Preset grouping definitions
    PRESET_GROUPS: ClassVar[dict[str, str]] = {
        # RELAX
        "god_mode": "RELAX",
        "walk_in_the_park": "RELAX",
        "beginner": "RELAX",
        # FLOW
        "standard": "FLOW",
        "challenge": "FLOW",
        # TENSION
        "survival_horror": "TENSION",
        "nightmare": "TENSION",
        # OVERLOAD
        "hardcore": "OVERLOAD",
        "impossible": "OVERLOAD",
        # ACTION
        "bullet_heaven": "ACTION",
    }

    # Complete preset definitions with all parameters
    PRESET_DEFINITIONS: ClassVar[dict[str, PresetParameters]] = {
        "RELAX": PresetParameters(
            spawn_rate_multiplier=-0.5,
            enemy_speed_multiplier=-0.5,
            enemy_health_multiplier=-0.5,
            enemy_damage_multiplier=-0.5,
            player_speed_multiplier=0.3,
            player_damage_multiplier=0.5,
            item_drop_chance_modifier=0.3,
            difficulty_level="easy",
            description="Combined Relaxed State",
            tags=["easy", "group"],
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
            description="Combined Flow State",
            tags=["medium", "group"],
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
            description="Combined Tension State",
            tags=["hard", "group"],
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
            description="Combined Overload State",
            tags=["extreme", "group"],
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
            description="Combined Action State",
            tags=["chaotic", "group"],
        ),
    }

    # Preset ordering by difficulty
    PRESETS_BY_DIFFICULTY: ClassVar[list[str]] = ["RELAX", "FLOW", "TENSION", "OVERLOAD", "ACTION"]

    # Mapping dictionaries for quick access
    PRESET_TO_IDX: ClassVar[dict[str, int]] = {preset: idx for idx, preset in enumerate(PRESETS_BY_DIFFICULTY)}
    IDX_TO_PRESET: ClassVar[dict[int, str]] = {idx: preset for preset, idx in PRESET_TO_IDX.items()}

    # Difficulty level mapping
    DIFFICULTY_LEVELS: ClassVar[dict[str, list[str]]] = {
        "very_easy": ["god_mode"],
        "easy": ["walk_in_the_park", "beginner"],
        "medium": ["standard"],
        "medium_hard": ["challenge"],
        "hard": ["survival_horprise"],
        "very_hard": ["nightmare"],
        "extreme": ["hardcore"],
        "chaotic": ["bullet_heaven"],
        "impossible": ["impossible"],
    }

    # Emotional profiles for each preset (placeholder - needs to be defined)
    EMOTIONAL_PROFILES: ClassVar[dict[str, dict[str, float]]] = {
        "RELAX": {
            "prob_angry_disgust": 0.1,
            "prob_fear_surprise": 0.1,
            "prob_happy": 0.6,
            "prob_neutral": 0.2,
            "prob_sad": 0.0,
        },
        "FLOW": {
            "prob_angry_disgust": 0.1,
            "prob_fear_surprise": 0.2,
            "prob_happy": 0.3,
            "prob_neutral": 0.3,
            "prob_sad": 0.1,
        },
        "TENSION": {
            "prob_angry_disgust": 0.2,
            "prob_fear_surprise": 0.4,
            "prob_happy": 0.1,
            "prob_neutral": 0.2,
            "prob_sad": 0.1,
        },
        "OVERLOAD": {
            "prob_angry_disgust": 0.4,
            "prob_fear_surprise": 0.3,
            "prob_happy": 0.1,
            "prob_neutral": 0.1,
            "prob_sad": 0.1,
        },
        "ACTION": {
            "prob_angry_disgust": 0.3,
            "prob_fear_surprise": 0.3,
            "prob_happy": 0.2,
            "prob_neutral": 0.1,
            "prob_sad": 0.1,
        },
    }

    @classmethod
    def remap_preset(cls, granular_preset: str) -> str:
        """Convert granular preset to group preset."""
        if granular_preset in cls.PRESETS_BY_DIFFICULTY:
            return granular_preset
        return cls.PRESET_GROUPS.get(granular_preset, "FLOW")  # Default fallback

    @classmethod
    def get_preset_parameters(cls, preset_name: str) -> PresetParameters:
        """Get parameters for a preset with fallback to FLOW."""
        remapped_name = cls.remap_preset(preset_name)
        return cls.PRESET_DEFINITIONS.get(remapped_name, cls.PRESET_DEFINITIONS["FLOW"])

    @classmethod
    def get_emotional_prototype(cls, preset_name: str) -> np.ndarray:
        """Get emotional prototype vector for a preset."""
        if preset_name not in cls.EMOTIONAL_PROFILES:
            raise ValueError(f"Unknown preset: {preset_name}")

        profile = cls.EMOTIONAL_PROFILES[preset_name]
        return np.array(
            [
                profile["prob_angry_disgust"],
                profile["prob_fear_surprise"],
                profile["prob_happy"],
                profile["prob_neutral"],
                profile["prob_sad"],
            ]
        )

    @classmethod
    def get_all_emotional_prototypes(cls) -> dict[str, np.ndarray]:
        """Get emotional prototypes for all presets."""
        return {preset: cls.get_emotional_prototype(preset) for preset in cls.PRESETS_BY_DIFFICULTY}

    @classmethod
    def get_preset_by_difficulty(cls, difficulty: str) -> list[str]:
        """Get all presets for a specific difficulty level."""
        return cls.DIFFICULTY_LEVELS.get(difficulty, [])

    @classmethod
    def get_difficulty_level(cls, preset_name: str) -> str:
        """Get difficulty level for a preset."""
        for difficulty, presets in cls.DIFFICULTY_LEVELS.items():
            if preset_name in presets:
                return difficulty
        raise ValueError(f"Unknown preset: {preset_name}")

    @classmethod
    def get_preset_multipliers(cls, preset_name: str) -> list[float]:
        """Get game balance multipliers as a list."""
        params = cls.get_preset_parameters(preset_name)
        return [
            params.spawn_rate_multiplier,
            params.enemy_speed_multiplier,
            params.enemy_health_multiplier,
            params.enemy_damage_multiplier,
            params.player_speed_multiplier,
            params.player_damage_multiplier,
            params.item_drop_chance_modifier,
        ]

    @classmethod
    def get_adjacent_presets(cls, preset_name: str) -> tuple[str | None, str | None]:
        """Get easier and harder adjacent presets."""
        idx = cls.PRESET_TO_IDX.get(preset_name)
        if idx is None:
            return None, None

        easier = cls.IDX_TO_PRESET.get(idx - 1)
        harder = cls.IDX_TO_PRESET.get(idx + 1)

        return easier, harder

    @classmethod
    def validate_emotional_vector(cls, emotions: list[float]) -> bool:
        """Validate that emotion probabilities are reasonable."""
        if len(emotions) != 5:
            return False

        # Check if probabilities sum to approximately 1.0
        total = sum(emotions)
        return 0.9 <= total <= 1.1 and all(0 <= e <= 1 for e in emotions)

    @classmethod
    def get_recommended_preset(cls, dominant_emotion: str, intensity: float) -> str:
        """Get recommended preset based on dominant emotion and intensity."""
        emotion_to_preset: dict[str, list[str]] = {
            "happy": ["god_mode", "walk_in_the_park", "beginner"],
            "neutral": ["standard", "challenge"],
            "sad": ["challenge", "survival_horror"],
            "fear_surprise": ["survival_horror", "nightmare"],
            "angry_disgust": ["hardcore", "bullet_heaven", "impossible"],
        }

        presets = emotion_to_preset.get(dominant_emotion, ["standard"])

        # Adjust based on intensity
        if intensity < 0.3:
            return presets[0]  # Easier preset
        elif intensity > 0.7:
            return presets[-1]  # Harder preset
        else:
            return presets[len(presets) // 2]  # Middle preset

    @classmethod
    def get_all_preset_names(cls) -> list[str]:
        """Get all available preset names."""
        return list(cls.PRESET_GROUPS.keys()) + cls.PRESETS_BY_DIFFICULTY

    @classmethod
    def is_valid_preset(cls, preset_name: str) -> bool:
        """Check if a preset name is valid."""
        return preset_name in cls.PRESET_GROUPS or preset_name in cls.PRESETS_BY_DIFFICULTY


# Utility functions for common operations
def get_preset_difficulty_score(preset_name: str) -> float:
    """Get normalized difficulty score (0-1) for a preset."""
    idx = PresetMapping.PRESET_TO_IDX.get(preset_name, 0)
    max_idx = len(PresetMapping.PRESETS_BY_DIFFICULTY) - 1
    return idx / max_idx if max_idx > 0 else 0.0


def calculate_emotional_distance(emotions1: list[float], emotions2: list[float]) -> float:
    """Calculate cosine distance between two emotional vectors."""
    vec1 = np.array(emotions1)
    vec2 = np.array(emotions2)

    # Normalize vectors
    norm1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    norm2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

    # Cosine similarity
    similarity = np.dot(norm1, norm2)
    return 1.0 - similarity  # Convert to distance


def find_closest_preset(emotional_vector: list[float]) -> tuple[str, float]:
    """Find the closest preset based on emotional similarity."""
    prototypes = PresetMapping.get_all_emotional_prototypes()

    best_preset = "FLOW"  # Default fallback
    best_distance = float("inf")

    for preset, prototype in prototypes.items():
        distance = calculate_emotional_distance(emotional_vector, prototype.tolist())
        if distance < best_distance:
            best_distance = distance
            best_preset = preset

    return best_preset, 1.0 - best_distance  # Return preset and similarity


def get_preset_difficulty_progression() -> list[tuple[str, float]]:
    """Get all presets with their difficulty scores in order."""
    return [(preset, get_preset_difficulty_score(preset)) for preset in PresetMapping.PRESETS_BY_DIFFICULTY]
