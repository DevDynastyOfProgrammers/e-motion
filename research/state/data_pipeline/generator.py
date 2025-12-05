"""
Synthetic data generator using emotional balance/seesaw principle.
Emotions exist on continuums and have natural oppositions.
"""

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ml.state.constants import data_config


class EmotionalBalanceGenerator:
    """
    Generator based on emotional balance theory:
    - Happy ⇄ Sad (Pleasure-Pain)
    - Fear ⇄ Anger (Approach-Avoidance)
    - Surprise ⇄ Disgust (Novelty-Familiarity)
    - Neutral as balancing point
    """

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        # Emotional balance axes (opposing pairs)
        self.emotional_axes = {
            'pleasure_pain': ('prob_happy', 'prob_sad'),  # Happy ⇄ Sad
            'approach_avoidance': ('prob_fear_surprise', 'prob_angry_disgust'),  # Fear ⇄ Anger
            'novelty_familiarity': ('prob_neutral', 'prob_fear_surprise'),  # Neutral ⇄ Surprise
        }

        # Preset emotional balance profiles
        self.preset_balances: dict[str, dict[str, Any]] = {
            # Easy presets - Positive emotional balance
            'god_mode': {
                'pleasure_pain': 0.8,  # Strong positive (happy)
                'approach_avoidance': 0.7,  # Approach dominant
                'novelty_familiarity': 0.6,  # Balanced novelty
                'confidence_range': (0.8, 0.95),
            },
            'walk_in_the_park': {
                'pleasure_pain': 0.7,
                'approach_avoidance': 0.6,
                'novelty_familiarity': 0.5,
                'confidence_range': (0.7, 0.9),
            },
            'beginner': {
                'pleasure_pain': 0.6,
                'approach_avoidance': 0.5,
                'novelty_familiarity': 0.4,
                'confidence_range': (0.6, 0.8),
            },
            # Medium presets - Balanced emotions
            'standard': {
                'pleasure_pain': 0.0,  # Perfect balance
                'approach_avoidance': 0.0,
                'novelty_familiarity': 0.0,
                'confidence_range': (0.5, 0.75),
            },
            'challenge': {
                'pleasure_pain': -0.2,  # Slight negative
                'approach_avoidance': -0.1,
                'novelty_familiarity': -0.1,
                'confidence_range': (0.5, 0.7),
            },
            # Hard presets - Negative emotional balance
            'survival_horror': {
                'pleasure_pain': -0.6,  # Strong negative (fear/sad)
                'approach_avoidance': -0.7,  # Avoidance dominant
                'novelty_familiarity': -0.5,  # High novelty (surprise)
                'confidence_range': (0.3, 0.6),
            },
            'nightmare': {
                'pleasure_pain': -0.7,
                'approach_avoidance': -0.6,
                'novelty_familiarity': -0.6,
                'confidence_range': (0.25, 0.55),
            },
            'hardcore': {
                'pleasure_pain': -0.5,  # Anger dominant
                'approach_avoidance': -0.8,  # Strong avoidance (anger)
                'novelty_familiarity': -0.3,
                'confidence_range': (0.2, 0.5),
            },
            'bullet_heaven': {
                'pleasure_pain': 0.1,  # Mixed but slightly positive
                'approach_avoidance': -0.3,  # Some avoidance
                'novelty_familiarity': 0.2,  # Some novelty
                'confidence_range': (0.4, 0.65),
            },
            'impossible': {
                'pleasure_pain': -0.9,  # Extreme negative
                'approach_avoidance': -0.9,  # Extreme avoidance
                'novelty_familiarity': -0.8,  # Extreme novelty (overwhelming)
                'confidence_range': (0.15, 0.45),
            },
        }

    def balance_to_emotions(
        self, pleasure_pain: float, approach_avoidance: float, novelty_familiarity: float
    ) -> np.ndarray:
        """
        Convert emotional balance coordinates to emotion probabilities.
        Uses the emotional seesaw principle.
        """
        # Base neutral emotion (foundation)
        neutral_base = 0.2
        epsilon = 1e-8  # Small constant to avoid division by zero

        # Calculate emotion pairs based on balance axes
        # Happy/Sad axis (Pleasure-Pain)
        if pleasure_pain >= 0:
            happy = neutral_base + abs(pleasure_pain) * 0.6
            sad = neutral_base * (1 - abs(pleasure_pain))
        else:
            happy = neutral_base * (1 - abs(pleasure_pain))
            sad = neutral_base + abs(pleasure_pain) * 0.6

        # Fear_Surprise/Anger_Disgust axis (Approach-Avoidance)
        if approach_avoidance >= 0:
            fear_surprise = neutral_base + abs(approach_avoidance) * 0.5
            angry_disgust = neutral_base * (1 - abs(approach_avoidance))
        else:
            fear_surprise = neutral_base * (1 - abs(approach_avoidance))
            angry_disgust = neutral_base + abs(approach_avoidance) * 0.5

        # Adjust neutral based on novelty-familiarity
        if novelty_familiarity >= 0:
            # More novelty = less neutral, more surprise
            neutral = neutral_base * (1 - novelty_familiarity * 0.5)
            fear_surprise += novelty_familiarity * 0.3
        else:
            # More familiarity = more neutral
            neutral = neutral_base + abs(novelty_familiarity) * 0.3
            fear_surprise *= 1 - abs(novelty_familiarity) * 0.5

        # Ensure values are within bounds
        emotions = np.array([angry_disgust, fear_surprise, happy, neutral, sad])
        emotions = np.clip(emotions, 0.05, 0.8)

        # Normalize to sum to 1.0
        emotions_sum = emotions.sum()
        if emotions_sum > epsilon:
            emotions = emotions / emotions_sum

        return emotions

    def generate_balanced_vector(self, preset: str, num_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Generate emotion vectors using emotional balance theory."""
        balance_profile = self.preset_balances[preset]

        confidence_values: list[float] = []
        emotion_vectors: list[np.ndarray] = []

        for _ in range(num_samples):
            # Add variation to balance points
            pleasure_pain = np.random.normal(
                balance_profile['pleasure_pain'], abs(balance_profile['pleasure_pain']) * 0.2 + 0.1
            )
            approach_avoidance = np.random.normal(
                balance_profile['approach_avoidance'], abs(balance_profile['approach_avoidance']) * 0.2 + 0.1
            )
            novelty_familiarity = np.random.normal(
                balance_profile['novelty_familiarity'], abs(balance_profile['novelty_familiarity']) * 0.2 + 0.1
            )

            # Clip to reasonable ranges
            pleasure_pain = np.clip(pleasure_pain, -0.95, 0.95)
            approach_avoidance = np.clip(approach_avoidance, -0.95, 0.95)
            novelty_familiarity = np.clip(novelty_familiarity, -0.95, 0.95)

            # Convert balance coordinates to emotions
            emotions = self.balance_to_emotions(pleasure_pain, approach_avoidance, novelty_familiarity)

            # Generate confidence based on emotional clarity and balance
            emotional_clarity = np.max(emotions) - np.min(emotions)
            balance_stability = 1.0 - (abs(pleasure_pain) + abs(approach_avoidance) + abs(novelty_familiarity)) / 3.0

            confidence = np.random.uniform(*balance_profile['confidence_range'])
            confidence *= 0.7 + 0.3 * emotional_clarity  # Boost for clear emotions
            confidence *= 0.8 + 0.2 * balance_stability  # Boost for balanced states

            confidence = np.clip(confidence, 0.1, 0.95)

            confidence_values.append(confidence)
            emotion_vectors.append(emotions)

        return np.array(confidence_values), np.array(emotion_vectors)

    def generate_dataset(self, total_samples: int = 25000) -> pd.DataFrame:
        """Generate dataset using emotional balance theory."""
        logger.info('🎭 Generating dataset with emotional balance theory...')

        # Realistic preset distribution
        preset_weights: dict[str, float] = {
            'standard': 0.15,
            'challenge': 0.14,
            'survival_horror': 0.15,
            'nightmare': 0.12,
            'beginner': 0.10,
            'hardcore': 0.09,
            'walk_in_the_park': 0.08,
            'bullet_heaven': 0.07,
            'god_mode': 0.04,
            'impossible': 0.02,
        }

        samples_per_preset = {preset: int(total_samples * weight) for preset, weight in preset_weights.items()}

        # Adjust total to match exactly
        total_allocated = sum(samples_per_preset.values())
        difference = total_samples - total_allocated
        if difference != 0:
            samples_per_preset['standard'] += difference
            logger.info(f'🔧 Adjusted sample count by {difference} for standard preset')

        # Generate data
        all_data: list[dict[str, Any]] = []

        for preset, num_samples in samples_per_preset.items():
            logger.info(f'🎯 Generating {num_samples} balanced samples for {preset}')

            confidence_values, emotion_vectors = self.generate_balanced_vector(preset, num_samples)

            for i in range(num_samples):
                confidence = confidence_values[i]
                emotions = emotion_vectors[i]

                row = {
                    'true_class': preset,
                    'predicted_class': preset,
                    'confidence': confidence,
                    'prob_angry_disgust': emotions[0],
                    'prob_fear_surprise': emotions[1],
                    'prob_happy': emotions[2],
                    'prob_neutral': emotions[3],
                    'prob_sad': emotions[4],
                    'preset': preset,
                }

                # Add balance coordinates for analysis
                row.update(self._get_balance_coordinates(emotions))
                row.update(self._generate_balance_parameters(preset))

                all_data.append(row)

        df = pd.DataFrame(all_data)
        self._validate_emotional_balance(df)

        logger.info(f'✅ Generated balanced dataset with {len(df):,} samples')
        return df

    def _get_balance_coordinates(self, emotions: np.ndarray) -> dict[str, float]:
        """Calculate balance coordinates from emotions for analysis."""
        angry_disgust, fear_surprise, happy, neutral, sad = emotions
        epsilon = 1e-8

        # Calculate balance coordinates
        pleasure_pain = (happy - sad) / (happy + sad + epsilon)
        approach_avoidance = (fear_surprise - angry_disgust) / (fear_surprise + angry_disgust + epsilon)
        novelty_familiarity = (fear_surprise - neutral) / (fear_surprise + neutral + epsilon)

        return {
            'pleasure_pain_balance': pleasure_pain,
            'approach_avoidance_balance': approach_avoidance,
            'novelty_familiarity_balance': novelty_familiarity,
        }

    def _generate_balance_parameters(self, preset: str) -> dict[str, float]:
        """Generate game parameters with balance-appropriate variations."""
        base_parameters: dict[str, list[float]] = {
            'god_mode': [-0.8, -0.9, -1.0, -1.0, 0.7, 1.0, 0.5],
            'walk_in_the_park': [-0.5, -0.5, -0.5, -0.5, 0.3, 0.5, 0.3],
            'beginner': [-0.2, -0.2, 0.0, -0.2, 0.0, 0.2, 0.1],
            'standard': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'challenge': [0.4, 0.2, 0.4, 0.3, 0.0, 0.0, 0.0],
            'survival_horror': [0.2, 0.0, 0.8, 0.8, -0.2, -0.3, -0.5],
            'nightmare': [0.8, 0.5, 0.7, 0.7, 0.3, 0.0, 0.4],
            'hardcore': [0.3, 0.6, 0.5, 1.0, 0.1, -0.5, -0.8],
            'bullet_heaven': [1.0, 0.8, 0.9, 0.9, 0.5, 0.8, 0.7],
            'impossible': [1.0, 1.0, 1.0, 1.0, -0.5, -1.0, -1.0],
        }

        base_params = base_parameters[preset]
        param_names = [
            'spawn_rate_multiplier',
            'enemy_speed_multiplier',
            'enemy_health_multiplier',
            'enemy_damage_multiplier',
            'player_speed_multiplier',
            'player_damage_multiplier',
            'item_drop_chance_modifier',
        ]

        # Add variations correlated with emotional balance
        varied_params: dict[str, float] = {}
        for i, (name, base_value) in enumerate(zip(param_names, base_params)):
            variation = np.random.normal(0, abs(base_value) * 0.05) if base_value != 0 else np.random.normal(0, 0.02)
            varied_params[name] = float(base_value + variation)

        return varied_params

    def _validate_emotional_balance(self, df: pd.DataFrame) -> None:
        """Validate that emotional balance principles are maintained."""
        logger.info('🔍 Validating emotional balance patterns...')

        # Check balance correlations with presets
        balance_correlations: dict[str, dict[str, float]] = {}
        for preset in self.preset_balances.keys():
            preset_data = df[df['preset'] == preset]
            if len(preset_data) > 0:
                avg_pleasure = preset_data['pleasure_pain_balance'].mean()
                avg_approach = preset_data['approach_avoidance_balance'].mean()
                avg_novelty = preset_data['novelty_familiarity_balance'].mean()

                balance_correlations[preset] = {
                    'pleasure_pain': avg_pleasure,
                    'approach_avoidance': avg_approach,
                    'novelty_familiarity': avg_novelty,
                }

        logger.info('📊 Emotional balance by preset:')
        for preset, balances in balance_correlations.items():
            logger.info(f'  {preset}:')
            for axis, value in balances.items():
                logger.info(f'    {axis}: {value:+.3f}')

        # Check emotional oppositions
        emotion_pairs = [('prob_happy', 'prob_sad'), ('prob_fear_surprise', 'prob_angry_disgust')]

        for emo1, emo2 in emotion_pairs:
            correlation = df[emo1].corr(df[emo2])
            logger.info(f'📈 Correlation {emo1} vs {emo2}: {correlation:+.3f}')

            if correlation > -0.3:  # Should be negatively correlated
                logger.warning(f'⚠️ Weak opposition between {emo1} and {emo2}')

    def analyze_emotional_space(self, df: pd.DataFrame) -> None:
        """Analyze the generated emotional space."""
        logger.info('🔬 Analyzing emotional space structure...')

        # Plot emotional distributions
        emotion_cols = ['prob_angry_disgust', 'prob_fear_surprise', 'prob_happy', 'prob_neutral', 'prob_sad']

        logger.info('📋 Emotion distributions:')
        for col in emotion_cols:
            logger.info(f'  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}')

        # Check emotional balance ranges
        balance_cols = ['pleasure_pain_balance', 'approach_avoidance_balance', 'novelty_familiarity_balance']
        for col in balance_cols:
            logger.info(f'  {col}: range=({df[col].min():.3f}, {df[col].max():.3f})')


def setup_logging() -> None:
    """Configure logging for the generator."""
    logger.remove()

    # Console logging with colors
    logger.add(
        sys.stdout,
        level='INFO',
        format='<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>',
        colorize=True,
    )

    # File logging
    logger.add(
        'emotional_balance_generation.log',
        rotation='10 MB',
        level='DEBUG',
        format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}',
    )


def main() -> None:
    """Generate dataset using emotional balance theory."""
    setup_logging()

    logger.info('🚀 Starting emotional balance dataset generation...')

    generator = EmotionalBalanceGenerator(random_state=data_config.random_state)
    df = generator.generate_dataset(total_samples=20000)

    # Analyze the emotional space
    generator.analyze_emotional_space(df)

    # Save dataset
    output_dir = Path('train')
    output_dir.mkdir(exist_ok=True)

    dataset_path = output_dir / 'emotional_balance_dataset.csv'
    df.to_csv(dataset_path, index=False, sep=';')
    logger.info(f'💾 Dataset saved to {dataset_path}')

    # Save sample for inspection
    sample_path = output_dir / 'emotional_balance_sample.csv'
    df.head(1000).to_csv(sample_path, index=False, sep=';')
    logger.info(f'📄 Sample saved to {sample_path}')

    logger.info('🎉 Emotional balance dataset generation completed!')


if __name__ == '__main__':
    main()
