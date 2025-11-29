"""
Training module for advanced similarity models.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import data_config
from core.preset_mapping import PresetMapping
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, f1_score

from model.analyzer import AdvancedPresetAnalyzer
from model.classifier import AdvancedEmotionClassifier


class AdvancedCosineTrainer:
    """Trainer for advanced similarity based models."""

    def __init__(self) -> None:
        self.preset_analyzer = AdvancedPresetAnalyzer()
        self.model: AdvancedEmotionClassifier
        self._feature_columns = [
            "confidence",
            "prob_angry_disgust",
            "prob_fear_surprise",
            "prob_happy",
            "prob_neutral",
            "prob_sad",
        ]

    def train(self, df: pd.DataFrame) -> AdvancedEmotionClassifier:
        """Train and SAVE advanced similarity model."""
        logger.info("Training advanced similarity model...")

        # Analyze preset patterns and create prototypes
        self.preset_analyzer.analyze_preset_patterns(df)

        # Create and initialize model
        self.model = AdvancedEmotionClassifier(self.preset_analyzer)

        # SAVE trained prototypes for server
        logger.info("💾 Saving trained prototypes...")
        np.save(data_config.prototypes_save_path, self.preset_analyzer.preset_prototypes)

        # Save preset stats too
        stats_path = data_config.prototypes_save_path.replace(".npy", "_stats.npy")
        np.save(stats_path, self.preset_analyzer.preset_stats)

        # Evaluate performance
        results = self.evaluate_model(df)
        logger.info(f"Advanced model trained with accuracy: {results['accuracy']:.2f}%")

        return self.model

    def evaluate_model(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate model performance comprehensively."""
        logger.info("Evaluating advanced similarity model...")

        X, y_true = self._prepare_features_and_labels(df)

        predictions = self._collect_predictions(X, y_true)
        evaluation_results = self._calculate_evaluation_metrics(predictions, y_true)

        # Log detailed performance analysis
        self._log_performance_summary(evaluation_results)
        self.analyze_preset_performance(df)
        self.analyze_confusion_patterns(predictions)

        return evaluation_results

    def _prepare_features_and_labels(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and true labels from dataframe."""
        X = df[self._feature_columns].values
        y_true = df["preset"].values
        return X, y_true

    def _collect_predictions(self, X: np.ndarray, y_true: np.ndarray) -> list[tuple[str, str, float]]:
        """Collect predictions for all samples."""
        predictions = []

        for features, true_preset in zip(X, y_true):
            predicted_preset, confidence, _ = self.model.predict(features)
            predictions.append((true_preset, predicted_preset, confidence))

        return predictions

    def _calculate_evaluation_metrics(
        self, predictions: list[tuple[str, str, float]], y_true: np.ndarray
    ) -> dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        y_true_list = [true for true, _, _ in predictions]
        y_pred_list = [pred for _, pred, _ in predictions]

        correct_predictions = sum(1 for true, pred, _ in predictions if true == pred)
        total_samples = len(predictions)
        accuracy = 100.0 * correct_predictions / total_samples

        # Confidence scores for correct predictions only
        confidence_scores = [confidence for true, pred, confidence in predictions if true == pred]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        f1_weighted = f1_score(y_true_list, y_pred_list, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "avg_confidence": avg_confidence,
            "correct_predictions": correct_predictions,
            "total_samples": total_samples,
            "predictions": predictions,
        }

    def _log_performance_summary(self, results: dict[str, Any]) -> None:
        """Log comprehensive performance summary."""
        logger.info("📊 Model Performance Summary")
        logger.info(f"🎯 Accuracy: {results['accuracy']:.2f}%")
        logger.info(f"🏆 F1-Score (Weighted): {results['f1_weighted']:.4f}")
        logger.info(f"📈 Average Confidence: {results['avg_confidence']:.3f}")
        logger.info(f"✅ Correct Predictions: {results['correct_predictions']}/{results['total_samples']}")

    def analyze_preset_performance(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Analyze performance metrics for each preset."""
        logger.info("Analyzing performance by preset...")

        results = {}

        for preset in PresetMapping.PRESETS_BY_DIFFICULTY:
            preset_data = df[df["preset"] == preset]

            if preset_data.empty:
                logger.warning(f"No data found for preset: {preset}")
                continue

            preset_results = self._evaluate_preset_performance(preset_data, preset)
            results[preset] = preset_results

            self._log_preset_results(preset, preset_results)

        return results

    def _evaluate_preset_performance(self, preset_data: pd.DataFrame, preset: str) -> dict[str, float]:
        """Evaluate performance for a single preset."""
        X_preset = preset_data[self._feature_columns].values

        correct_count = 0
        confidence_scores = []

        for features in X_preset:
            predicted_preset, confidence, _ = self.model.predict(features)
            if predicted_preset == preset:
                correct_count += 1
                confidence_scores.append(confidence)

        accuracy = 100.0 * correct_count / len(X_preset)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {
            "accuracy": accuracy,
            "samples": len(X_preset),
            "correct": correct_count,
            "avg_confidence": avg_confidence,
        }

    def _log_preset_results(self, preset: str, results: dict[str, float]) -> None:
        """Log results for a single preset."""
        logger.info(
            f"  {preset}: {results['accuracy']:.2f}% "
            f"({results['correct']}/{results['samples']}) - "
            f"avg confidence: {results['avg_confidence']:.3f}"
        )

    def log_advanced_metrics(self, y_true: list[str], y_pred: list[str]) -> float:
        """Calculate and log detailed classification metrics."""
        logger.info("--- Detailed Classification Metrics ---")

        # Generate comprehensive classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        logger.info(f"\n{report}")

        # Calculate weighted F1 score
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        logger.info(f"🏆 F1-Score (Weighted Avg): {f1_weighted:.4f}")
        logger.info("---------------------------------------")

        return f1_weighted

    def analyze_confusion_patterns(self, predictions: list[tuple[str, str, float]]) -> None:
        """Analyze confusion patterns between presets."""
        logger.info("Analyzing confusion patterns...")

        confusion_counts = {}
        for true_preset, predicted_preset, confidence in predictions:
            if true_preset != predicted_preset:
                confusion_key = (true_preset, predicted_preset)
                confusion_counts[confusion_key] = confusion_counts.get(confusion_key, 0) + 1

        self._log_confusion_analysis(confusion_counts)

    def _log_confusion_analysis(self, confusion_counts: dict[tuple[str, str], int]) -> None:
        """Log confusion pattern analysis."""
        if not confusion_counts:
            logger.info("🎉 No confusion patterns found (perfect accuracy)")
            return

        # Show top confusion pairs
        top_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info("🔍 Top confusion patterns:")
        for (true_preset, predicted_preset), count in top_confusions:
            logger.info(f"  {true_preset} → {predicted_preset}: {count} times")

    def plot_performance_analysis(
        self, df: pd.DataFrame, save_path: str = "plots/cosine/advanced_performance.html"
    ) -> None:
        """Create comprehensive performance analysis visualization."""
        preset_results = self.analyze_preset_performance(df)

        if not preset_results:
            logger.warning("No preset results available for visualization")
            return

        fig = self._create_performance_plot(preset_results)
        self._finalize_plot(fig, save_path)

    def _create_performance_plot(self, preset_results: dict[str, dict[str, float]]) -> go.Figure:
        """Create performance visualization plot."""
        presets = list(preset_results.keys())
        accuracies = [preset_results[p]["accuracy"] for p in presets]
        samples = [preset_results[p]["samples"] for p in presets]
        confidences = [preset_results[p]["avg_confidence"] for p in presets]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Accuracy by Preset",
                "Sample Distribution",
                "Average Confidence by Preset",
                "Accuracy vs Confidence Correlation",
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
        )

        self._add_accuracy_plot(fig, presets, accuracies)
        self._add_sample_plot(fig, presets, samples)
        self._add_confidence_plot(fig, presets, confidences)
        self._add_correlation_plot(fig, accuracies, confidences, presets)

        return fig

    def _add_accuracy_plot(self, fig: go.Figure, presets: list[str], accuracies: list[float]) -> None:
        """Add accuracy subplot to figure."""
        fig.add_trace(
            go.Bar(
                name="Accuracy",
                x=presets,
                y=accuracies,
                hovertemplate="Preset: %{x}<br>Accuracy: %{y:.1f}%",
                marker_color="lightblue",
            ),
            row=1,
            col=1,
        )

    def _add_sample_plot(self, fig: go.Figure, presets: list[str], samples: list[int]) -> None:
        """Add sample distribution subplot to figure."""
        fig.add_trace(
            go.Bar(
                name="Samples",
                x=presets,
                y=samples,
                hovertemplate="Preset: %{x}<br>Samples: %{y}",
                marker_color="lightcoral",
            ),
            row=1,
            col=2,
        )

    def _add_confidence_plot(self, fig: go.Figure, presets: list[str], confidences: list[float]) -> None:
        """Add confidence subplot to figure."""
        fig.add_trace(
            go.Bar(
                name="Confidence",
                x=presets,
                y=confidences,
                hovertemplate="Preset: %{x}<br>Confidence: %{y:.3f}",
                marker_color="lightgreen",
            ),
            row=2,
            col=1,
        )

    def _add_correlation_plot(
        self, fig: go.Figure, accuracies: list[float], confidences: list[float], presets: list[str]
    ) -> None:
        """Add accuracy-confidence correlation subplot to figure."""
        fig.add_trace(
            go.Scatter(
                x=accuracies,
                y=confidences,
                text=presets,
                mode="markers+text",
                hovertemplate="Preset: %{text}<br>Accuracy: %{x:.1f}%<br>Confidence: %{y:.3f}",
                marker=dict(size=10, color="orange"),
            ),
            row=2,
            col=2,
        )

    def _finalize_plot(self, fig: go.Figure, save_path: str) -> None:
        """Finalize plot layout and save to file."""
        fig.update_layout(title="Advanced Similarity Model Performance Analysis", height=800, showlegend=False)

        # Update axis labels
        fig.update_xaxes(title_text="Preset", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_xaxes(title_text="Preset", row=1, col=2)
        fig.update_yaxes(title_text="Number of Samples", row=1, col=2)
        fig.update_xaxes(title_text="Preset", row=2, col=1)
        fig.update_yaxes(title_text="Average Confidence", row=2, col=1)
        fig.update_xaxes(title_text="Accuracy (%)", row=2, col=2)
        fig.update_yaxes(title_text="Average Confidence", row=2, col=2)

        fig.write_html(save_path)
        logger.info(f"📊 Advanced performance analysis plot saved to {save_path}")
