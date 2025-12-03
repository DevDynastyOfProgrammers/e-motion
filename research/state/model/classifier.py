"""
Advanced similarity based model for emotion classification.
"""

import numpy as np

from ml.state.config import cosine_config
from research.state.model.analyzer import AdvancedPresetAnalyzer


class AdvancedEmotionClassifier:
    """Advanced classifier using multiple similarity metrics and feature engineering."""

    def __init__(self, preset_analyzer: AdvancedPresetAnalyzer) -> None:
        self.preset_analyzer = preset_analyzer
        self.config = cosine_config
        self._epsilon = 1e-8  # Small constant to avoid division by zero
        self._expected_features = 6  # Expected number of basic features

    def create_feature_vector(self, emotional_vector: np.ndarray) -> np.ndarray:
        """Create advanced feature vector from basic emotions."""
        self._validate_input_vector(emotional_vector)

        # Unpack emotional vector
        confidence, *basic_emotions = emotional_vector
        prob_angry_disgust, prob_fear_surprise, prob_happy, prob_neutral, prob_sad = basic_emotions

        # Calculate advanced features
        intensity_features = self._calculate_intensity_features(basic_emotions)
        entropy_features = self._calculate_entropy_features(basic_emotions)
        valence_features = self._calculate_valence_features(
            prob_angry_disgust, prob_fear_surprise, prob_happy, prob_neutral, prob_sad
        )
        dominance_features = self._calculate_dominance_features(basic_emotions, confidence)
        distribution_features = self._calculate_distribution_features(basic_emotions)

        # Combine all features into final vector
        feature_vector = [
            confidence,
            *basic_emotions,
            *intensity_features,
            *entropy_features,
            *valence_features,
            *dominance_features,
            *distribution_features,
        ]

        return np.array(feature_vector)

    def _validate_input_vector(self, emotional_vector: np.ndarray) -> None:
        """Validate input vector dimensions."""
        if len(emotional_vector) != self._expected_features:
            raise ValueError(f"Expected {self._expected_features} basic features, got {len(emotional_vector)}")

    def _calculate_intensity_features(self, basic_emotions: list[float]) -> list[float]:
        """Calculate emotional intensity related features."""
        emotional_intensity = max(basic_emotions)
        return [emotional_intensity]

    def _calculate_entropy_features(self, basic_emotions: list[float]) -> list[float]:
        """Calculate emotional complexity features."""
        emotional_entropy = -sum(p * np.log(p + self._epsilon) for p in basic_emotions if p > 0)
        return [emotional_entropy]

    def _calculate_valence_features(
        self,
        prob_angry_disgust: float,
        prob_fear_surprise: float,
        prob_happy: float,
        prob_neutral: float,
        prob_sad: float,
    ) -> list[float]:
        """Calculate emotional valence features."""
        negative_emotions = prob_angry_disgust + prob_fear_surprise + prob_sad
        positive_emotions = prob_happy + prob_neutral
        emotional_balance = positive_emotions - negative_emotions

        return [negative_emotions, positive_emotions, emotional_balance]

    def _calculate_dominance_features(self, basic_emotions: list[float], confidence: float) -> list[float]:
        """Calculate emotional dominance features."""
        sorted_probs = sorted(basic_emotions, reverse=True)

        if len(sorted_probs) > 1:
            dominance_ratio = sorted_probs[0] / (sorted_probs[1] + self._epsilon)
        else:
            dominance_ratio = 1.0

        emotional_intensity = max(basic_emotions)
        confidence_emotional_ratio = confidence / (emotional_intensity + self._epsilon)

        return [dominance_ratio, confidence_emotional_ratio]

    def _calculate_distribution_features(self, basic_emotions: list[float]) -> list[float]:
        """Calculate emotional distribution features."""
        emotional_variance = np.var(basic_emotions)
        emotional_range = max(basic_emotions) - min(basic_emotions)

        return [emotional_variance, emotional_range]

    def predict(self, emotional_vector: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """
        Predict preset using advanced similarity analysis.

        Args:
            emotional_vector: 6-dimensional vector [confidence, prob_angry_disgust,
                            prob_fear_surprise, prob_happy, prob_neutral, prob_sad]

        Returns:
            Tuple of (preset_name, confidence_score, all_similarities)
        """
        feature_vector = self.create_feature_vector(emotional_vector)

        preset, confidence_score, all_similarities = self.preset_analyzer.find_most_similar_preset(feature_vector)

        return preset, confidence_score, all_similarities

    def predict_batch(self, emotional_vectors: list[list[float]]) -> list[tuple[str, float]]:
        """
        Predict for multiple emotion vectors.

        Args:
            emotional_vectors: List of 6-dimensional emotion vectors

        Returns:
            List of tuples (preset_name, confidence_score)
        """
        return [self._predict_single_vector(vector) for vector in emotional_vectors]

    def _predict_single_vector(self, vector: list[float]) -> tuple[str, float]:
        """Predict for a single emotion vector."""
        preset, confidence, _ = self.predict(np.array(vector))
        return preset, confidence
