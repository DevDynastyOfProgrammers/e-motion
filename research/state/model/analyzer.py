"""
Analyze preset patterns and create emotional prototypes with advanced features.
LOCATED IN: research/state/model/analyzer.py
"""
import sys
import os
from typing import Any, List, Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.state.core.preset_mapping import PresetMapping


class AdvancedPresetAnalyzer:
    """Advanced analyzer with multiple similarity metrics, feature engineering, and visualization."""

    def __init__(self) -> None:
        self.preset_prototypes: Dict[str, np.ndarray] = {}
        self.preset_stats: Dict[str, Dict[str, Any]] = {}
        
        # Basic emotion columns expected in the dataset
        self.emotion_columns = [
            "prob_angry_disgust", 
            "prob_fear_surprise", 
            "prob_happy", 
            "prob_neutral", 
            "prob_sad"
        ]
        
        # All columns used for initial analysis
        self.all_columns = [
            "confidence",
            *self.emotion_columns
        ]
        self._epsilon = 1e-8  # Small constant to avoid division by zero

    # --- MATH & ANALYSIS SECTION ---

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced emotional features with comprehensive metrics."""
        # logger.info("Creating advanced emotional features...")

        df_enhanced = df.copy()

        # 1. Emotional intensity and complexity metrics
        df_enhanced["emotional_intensity"] = df_enhanced[self.emotion_columns].max(axis=1)
        df_enhanced["emotional_entropy"] = self._calculate_emotional_entropy(df_enhanced[self.emotion_columns])

        # 2. Emotional valence metrics
        df_enhanced["negative_emotions"] = (
            df_enhanced["prob_angry_disgust"] + df_enhanced["prob_fear_surprise"] + df_enhanced["prob_sad"]
        )
        df_enhanced["positive_emotions"] = df_enhanced["prob_happy"] + df_enhanced["prob_neutral"]
        df_enhanced["emotional_balance"] = df_enhanced["positive_emotions"] - df_enhanced["negative_emotions"]

        # 3. Emotional dominance and confidence metrics
        df_enhanced["dominance_ratio"] = self._calculate_dominance_ratio(df_enhanced[self.emotion_columns])
        df_enhanced["confidence_emotional_ratio"] = df_enhanced["confidence"] / (
            df_enhanced["emotional_intensity"] + self._epsilon
        )

        # 4. Emotional distribution metrics
        # ddof=1 ensures compatibility with sample variance
        df_enhanced["emotional_variance"] = df_enhanced[self.emotion_columns].var(axis=1, ddof=1)
        df_enhanced["emotional_range"] = df_enhanced[self.emotion_columns].max(axis=1) - df_enhanced[
            self.emotion_columns
        ].min(axis=1)

        return df_enhanced

    def _calculate_emotional_entropy(self, emotion_data: pd.DataFrame) -> pd.Series:
        """Calculate emotional entropy as a measure of emotional complexity."""
        # -sum(p * log(p))
        return -np.sum(emotion_data * np.log(emotion_data + self._epsilon), axis=1)

    def _calculate_dominance_ratio(self, emotion_data: pd.DataFrame) -> pd.Series:
        """Calculate ratio between dominant (1st) and subdominant (2nd) emotions."""
        sorted_probs = np.sort(emotion_data.values, axis=1)
        # Last element is max, second to last is second max
        return sorted_probs[:, -1] / (sorted_probs[:, -2] + self._epsilon)

    def analyze_preset_patterns(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate emotional prototypes (centroids) for each preset using advanced features."""
        logger.info("Analyzing emotional patterns for each preset...")

        df_advanced = self.create_advanced_features(df)
        feature_columns = self._get_feature_columns()

        for preset in PresetMapping.PRESETS_BY_DIFFICULTY:
            # Filter data by preset (using remapped groups)
            preset_data = df_advanced[df_advanced["preset"] == preset]

            if preset_data.empty:
                logger.warning(f"No data found for preset: {preset}")
                continue

            self._calculate_preset_prototype(preset, preset_data, feature_columns)

        return self.preset_prototypes

    def _get_feature_columns(self) -> List[str]:
        """Get list of all feature columns for prototype calculation."""
        # Must match the order in ml.state.model.classifier.AdvancedEmotionClassifier._create_advanced_features
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

    def _calculate_preset_prototype(self, preset: str, preset_data: pd.DataFrame, feature_columns: List[str]) -> None:
        """Calculate and store prototype (mean vector) for a single preset."""
        feature_vectors = preset_data[feature_columns].values
        prototype = feature_vectors.mean(axis=0)

        self.preset_prototypes[preset] = prototype
        self.preset_stats[preset] = {
            "mean": prototype,
            "std": feature_vectors.std(axis=0),
            "count": len(preset_data),
            "features": feature_columns,
        }
        logger.info(f"Analyzed {preset}: {len(preset_data)} samples.")

    # --- VISUALIZATION SECTION (RESTORED) ---

    def plot_emotional_prototypes(self, save_path: str = "plots/advanced_emotional_prototypes.html") -> None:
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
        # Ensure we use sorted presets for consistency
        presets = [p for p in PresetMapping.PRESETS_BY_DIFFICULTY if p in self.preset_prototypes]

        for preset in presets:
            # Indices 1 to 6 correspond to basic emotions in the feature vector
            basic_values = self.preset_prototypes[preset][1:6]  
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
        
        # Determine indices for these features based on _get_feature_columns
        # confidence is at 0
        # intensity is at 6
        # balance is at 9
        
        presets = [p for p in PresetMapping.PRESETS_BY_DIFFICULTY if p in self.preset_prototypes]

        for preset in presets:
            prototype = self.preset_prototypes[preset]
            adv_values = [prototype[0], prototype[6], prototype[9]]  

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
        presets = [p for p in PresetMapping.PRESETS_BY_DIFFICULTY if p in self.preset_prototypes]
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

    def _calculate_similarity_matrix(self, presets: List[str]) -> np.ndarray:
        """Calculate similarity matrix between all preset pairs."""
        n_presets = len(presets)
        similarity_matrix = np.zeros((n_presets, n_presets))

        for i, preset1 in enumerate(presets):
            for j, preset2 in enumerate(presets):
                # Using simple Euclidean distance based similarity for visualization
                distance = np.linalg.norm(self.preset_prototypes[preset1] - self.preset_prototypes[preset2])
                similarity_matrix[i, j] = 1 / (1 + distance)

        return similarity_matrix

    def _add_sample_statistics(self, fig: go.Figure) -> None:
        """Add sample count statistics to visualization."""
        presets = [p for p in PresetMapping.PRESETS_BY_DIFFICULTY if p in self.preset_prototypes]
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
        fig.write_html(save_path)