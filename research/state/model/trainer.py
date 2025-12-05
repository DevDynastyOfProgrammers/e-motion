import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ml.state.core.preset_mapping import PresetMapping
from ml.state.model.classifier import AdvancedEmotionClassifier
from ml.state.model.classifier import AdvancedPresetAnalyzer as RuntimeAnalyzer
from research.state.model.analyzer import AdvancedPresetAnalyzer as ResearchAnalyzer


@dataclass
class LocalDataConfig:
    data_path: str = 'research/state/data/emotional_balance_dataset.csv'
    prototypes_save_path: str = 'research/state/data/models/trainer_result_model.npy'


data_config = LocalDataConfig()


class AdvancedCosineTrainer:
    """Trainer class that orchestrates data loading, analysis, and saving."""

    def __init__(self) -> None:
        # Используем Research-версию для аналитики
        self.research_analyzer = ResearchAnalyzer()
        self.model: AdvancedEmotionClassifier
        self._feature_columns = [
            'confidence',
            'prob_angry_disgust',
            'prob_fear_surprise',
            'prob_happy',
            'prob_neutral',
            'prob_sad',
        ]

    def train(self, df: pd.DataFrame) -> AdvancedEmotionClassifier:
        """Train the model on the DataFrame and save artifacts."""
        logger.info('Training advanced similarity model...')

        # 1. Analyze preset patterns (Calculate Centroids/Prototypes)
        # Этап Research: считаем средние вектора из CSV
        self.research_analyzer.analyze_preset_patterns(df)

        # 2. Bridge Research -> Production
        # Создаем легкий Runtime-анализатор и передаем ему рассчитанные данные
        runtime_analyzer = RuntimeAnalyzer()
        runtime_analyzer.preset_prototypes = self.research_analyzer.preset_prototypes

        # 3. Initialize Production Model
        # Теперь классификатор получает правильный тип объекта
        self.model = AdvancedEmotionClassifier(runtime_analyzer)

        # 4. SAVE trained prototypes for the Game
        save_path = Path(data_config.prototypes_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f'💾 Saving trained prototypes to {save_path}...')
        np.save(str(save_path), self.research_analyzer.preset_prototypes)

        # 5. Evaluate
        results = self.evaluate_model(df)
        logger.info(f'Advanced model trained with accuracy: {results["accuracy"]:.2f}%')

        return self.model

    def evaluate_model(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run a self-test evaluation on the training set."""
        logger.info('Evaluating advanced similarity model...')

        X, y_true = self._prepare_features_and_labels(df)
        predictions = self._collect_predictions(X, y_true)
        evaluation_results = self._calculate_evaluation_metrics(predictions, y_true)

        self._log_performance_summary(evaluation_results)
        return evaluation_results

    def _prepare_features_and_labels(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[self._feature_columns].values
        y_true = df['preset'].values
        return X, y_true

    def _collect_predictions(self, X: np.ndarray, y_true: np.ndarray) -> list[tuple[str, str, float]]:
        predictions = []
        # Simulate Runtime Inference
        for features, true_preset in zip(X, y_true):
            # features: [conf, angry, fear, happy, neutral, sad]
            # model.predict uses the RuntimeAnalyzer internally now
            predicted_preset, confidence, _ = self.model.predict(features)
            predictions.append((true_preset, predicted_preset, confidence))
        return predictions

    def _calculate_evaluation_metrics(
        self, predictions: list[tuple[str, str, float]], y_true: np.ndarray
    ) -> dict[str, Any]:
        y_true_list = [true for true, _, _ in predictions]
        y_pred_list = [pred for _, pred, _ in predictions]

        correct_predictions = sum(1 for true, pred, _ in predictions if true == pred)
        total_samples = len(predictions)
        accuracy = 100.0 * correct_predictions / total_samples

        f1_weighted = f1_score(y_true_list, y_pred_list, average='weighted', zero_division=0)

        conf_scores = [c for t, p, c in predictions if t == p]
        avg_conf = np.mean(conf_scores) if conf_scores else 0.0

        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'avg_confidence': avg_conf,
            'correct_predictions': correct_predictions,
            'total_samples': total_samples,
        }

    def _log_performance_summary(self, results: dict[str, Any]) -> None:
        logger.info('📊 Model Performance Summary')
        logger.info(f'🎯 Accuracy: {results["accuracy"]:.2f}%')
        logger.info(f'🏆 F1-Score: {results["f1_weighted"]:.4f}')
        logger.info(f'✅ Correct: {results["correct_predictions"]}/{results["total_samples"]}')


if __name__ == '__main__':
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, format='<green>{time:HH:mm:ss}</green> | <level>{message}</level>', level='INFO')

    logger.info('🚀 Starting Research Training...')

    # Look for dataset
    possible_paths = [
        'research/state/data/emotional_balance_dataset.csv',
        'train/emotional_balance_dataset.csv',
        'emotional_balance_dataset.csv',
    ]

    csv_path = None
    for p in possible_paths:
        if Path(p).exists():
            csv_path = p
            break

    if not csv_path:
        logger.error('❌ Dataset not found!')
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, delimiter=';')

        # Remap presets using Game Logic
        df['preset'] = df['preset'].apply(PresetMapping.remap_preset)

        trainer = AdvancedCosineTrainer()
        trainer.train(df)

        logger.success('🎉 Training complete & Prototypes updated for the Game!')

    except Exception as e:
        logger.exception(f'💥 Training failed: {e}')
