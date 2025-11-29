"""
Analyze preset patterns and create emotional prototypes with advanced features.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.preset_mapping import PresetMapping
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity


class AdvancedPresetAnalyzer:
    """Advanced analyzer with multiple similarity metrics and feature engineering."""

    def __init__(self) -> None:
        self.preset_prototypes: dict[str, np.ndarray] = {}
        self.preset_stats: dict[str, dict[str, Any]] = {}
        self.emotion_columns = ["prob_angry_disgust", "prob_fear_surprise", "prob_happy", "prob_neutral", "prob_sad"]
        self.all_columns = [
            "confidence",
            "prob_angry_disgust",
            "prob_fear_surprise",
            "prob_happy",
            "prob_neutral",
            "prob_sad",
        ]
        self._epsilon = 1e-8  # Small constant to avoid division by zero

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced emotional features with comprehensive metrics."""
        logger.info("Creating advanced emotional features...")

        df_enhanced = df.copy()

        # Emotional intensity and complexity metrics
        df_enhanced["emotional_intensity"] = df_enhanced[self.emotion_columns].max(axis=1)
        df_enhanced["emotional_entropy"] = self._calculate_emotional_entropy(df_enhanced[self.emotion_columns])

        # Emotional valence metrics
        df_enhanced["negative_emotions"] = (
            df_enhanced["prob_angry_disgust"] + df_enhanced["prob_fear_surprise"] + df_enhanced["prob_sad"]
        )
        df_enhanced["positive_emotions"] = df_enhanced["prob_happy"] + df_enhanced["prob_neutral"]
        df_enhanced["emotional_balance"] = df_enhanced["positive_emotions"] - df_enhanced["negative_emotions"]

        # Emotional dominance and confidence metrics
        df_enhanced["dominance_ratio"] = self._calculate_dominance_ratio(df_enhanced[self.emotion_columns])
        df_enhanced["confidence_emotional_ratio"] = df_enhanced["confidence"] / (
            df_enhanced["emotional_intensity"] + self._epsilon
        )

        # Emotional distribution metrics
        df_enhanced["emotional_variance"] = df_enhanced[self.emotion_columns].var(axis=1)
        df_enhanced["emotional_range"] = df_enhanced[self.emotion_columns].max(axis=1) - df_enhanced[
            self.emotion_columns
        ].min(axis=1)

        new_features_count = len([col for col in df_enhanced.columns if col not in self.all_columns])
        logger.info(f"Created {new_features_count} advanced features")

        return df_enhanced

    def _calculate_emotional_entropy(self, emotion_data: pd.DataFrame) -> pd.Series:
        """Calculate emotional entropy as a measure of emotional complexity."""
        return -np.sum(emotion_data * np.log(emotion_data + self._epsilon), axis=1)

    def _calculate_dominance_ratio(self, emotion_data: pd.DataFrame) -> pd.Series:
        """Calculate ratio between dominant and subdominant emotions."""
        sorted_probs = np.sort(emotion_data.values, axis=1)
        return sorted_probs[:, -1] / (sorted_probs[:, -2] + self._epsilon)

    def analyze_preset_patterns(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Calculate emotional prototypes for each preset using advanced features."""
        logger.info("Analyzing emotional patterns for each preset with advanced features...")

        df_advanced = self.create_advanced_features(df)

        feature_columns = self._get_feature_columns()

        for preset in PresetMapping.PRESETS_BY_DIFFICULTY:
            preset_data = df_advanced[df_advanced["preset"] == preset]

            if preset_data.empty:
                logger.warning(f"No data found for preset: {preset}")
                continue

            self._calculate_preset_prototype(preset, preset_data, feature_columns)

        return self.preset_prototypes

    def _get_feature_columns(self) -> list[str]:
        """Get list of all feature columns for prototype calculation."""
        return self.all_columns + [
            "emotional_intensity",
            "emotional_entropy",
            "negative_emotions",
            "positive_emotions",
            "emotional_balance",
            "dominance_ratio",
            "confidence_emotional_ratio",
            "emotional_variance",
            "emotional_range",
        ]

    def _calculate_preset_prototype(self, preset: str, preset_data: pd.DataFrame, feature_columns: list[str]) -> None:
        """Calculate and store prototype for a single preset."""
        feature_vectors = preset_data[feature_columns].values
        prototype = feature_vectors.mean(axis=0)

        self.preset_prototypes[preset] = prototype
        self.preset_stats[preset] = {
            "mean": prototype,
            "std": feature_vectors.std(axis=0),
            "count": len(preset_data),
            "features": feature_columns,
        }

        logger.info(f"Preset {preset}: {len(feature_columns)} features, {len(preset_data)} samples")

    def calculate_similarity_scores(self, emotional_vector: np.ndarray, method: str = "euclidean") -> dict[str, float]:
        """Calculate similarity scores using multiple methods."""
        if not self.preset_prototypes:
            raise ValueError("Preset prototypes not calculated. Call analyze_preset_patterns first.")

        similarity_methods = {
            "cosine": self._cosine_similarity,
            "euclidean": self._euclidean_similarity,
            "manhattan": self._manhattan_similarity,
        }

        if method not in similarity_methods:
            raise ValueError(f"Unknown similarity method: {method}. Available: {list(similarity_methods.keys())}")

        return {
            preset: similarity_methods[method](emotional_vector, prototype)
            for preset, prototype in self.preset_prototypes.items()
        }

    def _cosine_similarity(self, emotional_vector: np.ndarray, prototype: np.ndarray) -> float:
        """Calculate cosine similarity using basic emotions."""
        basic_emotions = emotional_vector[1:6]
        prototype_basic = prototype[1:6]

        input_norm = basic_emotions / (np.linalg.norm(basic_emotions) + self._epsilon)
        prototype_norm = prototype_basic / (np.linalg.norm(prototype_basic) + self._epsilon)

        return cosine_similarity([input_norm], [prototype_norm])[0][0]

    def _euclidean_similarity(self, emotional_vector: np.ndarray, prototype: np.ndarray) -> float:
        """Calculate similarity based on Euclidean distance."""
        distance = np.linalg.norm(emotional_vector - prototype)
        return 1 / (1 + distance)

    def _manhattan_similarity(self, emotional_vector: np.ndarray, prototype: np.ndarray) -> float:
        """Calculate similarity based on Manhattan distance."""
        distance = np.sum(np.abs(emotional_vector - prototype))
        return 1 / (1 + distance)

    def find_most_similar_preset(
        self, emotional_vector: np.ndarray, confidence: float | None = None
    ) -> tuple[str, float, dict[str, float]]:
        """Find most similar preset using weighted combination of methods."""
        confidence, emotional_6d = self._preprocess_input(emotional_vector, confidence)

        # Calculate similarities using multiple methods
        cosine_similarities = self.calculate_similarity_scores(emotional_6d, "cosine")
        euclidean_similarities = self.calculate_similarity_scores(emotional_6d, "euclidean")

        combined_scores = self._combine_similarity_scores(cosine_similarities, euclidean_similarities, confidence)

        best_preset, best_score = max(combined_scores.items(), key=lambda x: x[1])
        return best_preset, best_score, combined_scores

    def _preprocess_input(self, emotional_vector: np.ndarray, confidence: float | None) -> tuple[float, np.ndarray]:
        """Preprocess input vector and extract confidence."""
        if confidence is None and len(emotional_vector) == 6:
            return emotional_vector[0], emotional_vector
        return confidence or 1.0, emotional_vector

    def _combine_similarity_scores(
        self, cosine_scores: dict[str, float], euclidean_scores: dict[str, float], confidence: float
    ) -> dict[str, float]:
        """Combine similarity scores with weighting and confidence boost."""
        combined_scores = {}

        for preset in self.preset_prototypes:
            # Weighted combination favoring euclidean distance
            combined_score = 0.3 * cosine_scores[preset] + 0.7 * euclidean_scores[preset]

            # Apply confidence boost
            confidence_boosted_score = combined_score * (1 + 0.2 * confidence)
            combined_scores[preset] = min(confidence_boosted_score, 1.0)

        return combined_scores

    def plot_emotional_prototypes(self, save_path: str = "plots/cosine/emotional_prototypes.html") -> None:
        """Create comprehensive visualization of emotional prototypes."""
        if not self.preset_prototypes:
            raise ValueError("No prototypes to plot. Call analyze_preset_patterns first.")

        fig = self._create_visualization_layout()
        self._add_basic_emotions_plot(fig)
        self._add_advanced_features_plot(fig)
        self._add_similarity_heatmap(fig)
        self._add_sample_statistics(fig)

        self._finalize_plot(fig, save_path)

    def _create_visualization_layout(self) -> go.Figure:
        """Create subplot layout for visualization."""
        return make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Basic Emotional Prototypes",
                "Advanced Feature Comparison",
                "Similarity Matrix",
                "Preset Statistics",
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "heatmap"}, {"type": "bar"}]],
        )

    def _add_basic_emotions_plot(self, fig: go.Figure) -> None:
        """Add basic emotions bar chart to visualization."""
        presets = list(self.preset_prototypes.keys())

        for preset in presets:
            basic_values = self.preset_prototypes[preset][1:6]  # Basic emotions only
            fig.add_trace(
                go.Bar(
                    name=preset,
                    x=self.emotion_columns,
                    y=basic_values,
                    hovertemplate=f"Preset: {preset}<br>Emotion: %{{x}}<br>Value: %{{y:.3f}}",
                ),
                row=1,
                col=1,
            )

    def _add_advanced_features_plot(self, fig: go.Figure) -> None:
        """Add advanced features comparison to visualization."""
        advanced_features = ["confidence", "emotional_intensity", "emotional_balance"]

        for preset in self.preset_prototypes:
            prototype = self.preset_prototypes[preset]
            adv_values = [prototype[0], prototype[6], prototype[9]]  # confidence, intensity, balance

            fig.add_trace(
                go.Bar(
                    name=preset,
                    x=advanced_features,
                    y=adv_values,
                    hovertemplate=f"Preset: {preset}<br>Feature: %{{x}}<br>Value: %{{y:.3f}}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    def _add_similarity_heatmap(self, fig: go.Figure) -> None:
        """Add similarity heatmap to visualization."""
        presets = list(self.preset_prototypes.keys())
        similarity_matrix = self._calculate_similarity_matrix(presets)

        fig.add_trace(
            go.Heatmap(
                z=similarity_matrix,
                x=presets,
                y=presets,
                colorscale="Viridis",
                hovertemplate="Preset1: %{y}<br>Preset2: %{x}<br>Similarity: %{z:.3f}",
            ),
            row=2,
            col=1,
        )

    def _calculate_similarity_matrix(self, presets: list[str]) -> np.ndarray:
        """Calculate similarity matrix between all preset pairs."""
        n_presets = len(presets)
        similarity_matrix = np.zeros((n_presets, n_presets))

        for i, preset1 in enumerate(presets):
            for j, preset2 in enumerate(presets):
                distance = np.linalg.norm(self.preset_prototypes[preset1] - self.preset_prototypes[preset2])
                similarity_matrix[i, j] = 1 / (1 + distance)

        return similarity_matrix

    def _add_sample_statistics(self, fig: go.Figure) -> None:
        """Add sample count statistics to visualization."""
        presets = list(self.preset_prototypes.keys())
        samples = [self.preset_stats[preset]["count"] for preset in presets]

        fig.add_trace(
            go.Bar(
                x=presets,
                y=samples,
                name="Samples",
                hovertemplate="Preset: %{x}<br>Samples: %{y}",
                marker_color="lightgreen",
            ),
            row=2,
            col=2,
        )

    def _finalize_plot(self, fig: go.Figure, save_path: str) -> None:
        """Finalize plot layout and save to file."""
        fig.update_layout(title="Advanced Emotional Space Analysis", height=1000, showlegend=True, barmode="group")

        # Update axis labels
        fig.update_xaxes(title_text="Basic Emotions", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)
        fig.update_xaxes(title_text="Advanced Features", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        fig.update_xaxes(title_text="Preset", row=2, col=1)
        fig.update_yaxes(title_text="Preset", row=2, col=1)
        fig.update_xaxes(title_text="Preset", row=2, col=2)
        fig.update_yaxes(title_text="Sample Count", row=2, col=2)

        fig.write_html(save_path)
        logger.info(f"Advanced emotional prototypes plot saved to {save_path}")
