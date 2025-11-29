"""
Main training script for advanced similarity emotion classification.
"""

import sys
from pathlib import Path

import pandas as pd
from config import data_config
from ml.state.core.preset_mapping import PresetMapping
from loguru import logger
from ml.state.model.trainer import AdvancedCosineTrainer


def setup_logging() -> None:
    """Configure logging with structured format."""
    logger.remove()

    # Console logging with rich format
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # File logging with rotation
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "advanced_training.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def analyze_emotional_patterns(df: pd.DataFrame) -> dict[str, float]:
    """
    Analyze emotional patterns and correlations in the dataset.

    Args:
        df: DataFrame with emotion data

    Returns:
        Dictionary of correlation coefficients
    """
    logger.info("🔍 Analyzing emotional patterns in dataset...")

    emotion_columns = ["prob_angry_disgust", "prob_fear_surprise", "prob_happy", "prob_neutral", "prob_sad"]

    # Basic statistics
    logger.info("📊 Overall emotion statistics:")
    for column in emotion_columns + ["confidence"]:
        mean_val = df[column].mean()
        std_val = df[column].std()
        logger.info(f"  {column}: mean={mean_val:.4f}, std={std_val:.4f}")

    # Correlation analysis with preset difficulty
    preset_codes = df["preset"].map(PresetMapping.PRESET_TO_IDX)

    correlation_results = {}
    for column in emotion_columns + ["confidence"]:
        correlation = df[column].corr(preset_codes)
        correlation_results[column] = correlation

    logger.info("📈 Correlation with preset difficulty:")
    for emotion, correlation in correlation_results.items():
        significance = "***" if abs(correlation) > 0.3 else "**" if abs(correlation) > 0.2 else "*"
        logger.info(f"  {emotion}: {correlation:+.4f} {significance}")

    return correlation_results


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load and preprocess the emotion dataset.

    Returns:
        Preprocessed DataFrame

    Raises:
        FileNotFoundError: If dataset file is not found
        Exception: For other loading errors
    """
    data_path = Path(data_config.data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    logger.info(f"📁 Loading dataset from {data_path}")
    df = pd.read_csv(data_path, delimiter=";")
    logger.info(f"✅ Loaded dataset with {len(df):,} samples")

    return df


def apply_preset_remapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preset remapping to compress granular presets into groups.

    Args:
        df: Original DataFrame with granular presets

    Returns:
        DataFrame with remapped presets
    """
    logger.info("🔄 Applying preset remapping (simplification)...")

    original_preset_count = len(df["preset"].unique())

    # Apply remapping to preset column
    df_remapped = df.copy()
    df_remapped["preset"] = df_remapped["preset"].apply(PresetMapping.remap_preset)

    new_preset_count = len(df_remapped["preset"].unique())
    new_distribution = df_remapped["preset"].value_counts()

    logger.info(f"✅ Compressed {original_preset_count} granular presets into {new_preset_count} groups")
    logger.info("📊 New preset distribution:")
    for preset, count in new_distribution.items():
        logger.info(f"  {preset}: {count:,} samples")

    return df_remapped


def create_necessary_directories() -> None:
    """Create all necessary directories for outputs."""
    directories = [
        Path(data_config.plots_dir),
        Path(data_config.model_save_path).parent,
        Path("logs"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"📁 Created directory: {directory}")


def train_and_evaluate_model(df: pd.DataFrame) -> AdvancedCosineTrainer:
    """
    Train and evaluate the advanced similarity model.

    Args:
        df: Preprocessed training data

    Returns:
        Trained trainer instance
    """
    logger.info("🎯 Training advanced similarity model...")

    trainer = AdvancedCosineTrainer()
    trainer.train(df)

    logger.info("✅ Model training completed successfully")
    return trainer


def create_visualizations(trainer: AdvancedCosineTrainer, df: pd.DataFrame) -> None:
    """
    Create and save model visualizations.

    Args:
        trainer: Trained model trainer
        df: Training data for visualization
    """
    logger.info("📊 Creating model visualizations...")

    try:
        # Emotional prototypes visualization
        prototypes_path = f"{data_config.plots_dir}/advanced_emotional_prototypes.html"
        trainer.preset_analyzer.plot_emotional_prototypes(prototypes_path)
        logger.info(f"✅ Emotional prototypes saved to: {prototypes_path}")

        # Performance analysis visualization
        performance_path = f"{data_config.plots_dir}/advanced_performance.html"
        trainer.plot_performance_analysis(df, performance_path)
        logger.info(f"✅ Performance analysis saved to: {performance_path}")

    except Exception as exc:
        logger.warning(f"⚠️ Could not create some visualizations: {exc}")
        logger.debug("Detailed visualization error:", exc_info=True)


def main() -> None:
    """Main training pipeline for advanced similarity emotion classification."""
    logger.info("🚀 Starting advanced similarity emotion classification pipeline")

    try:
        # Setup environment
        setup_logging()
        create_necessary_directories()

        # Load and preprocess data
        df = load_and_preprocess_data()
        df_processed = apply_preset_remapping(df)

        # Analyze data patterns
        correlation_results = analyze_emotional_patterns(df_processed)

        # Train model
        trainer = train_and_evaluate_model(df_processed)

        # Create visualizations
        create_visualizations(trainer, df_processed)

        logger.info("🎉 Advanced similarity model training completed successfully!")

    except FileNotFoundError as exc:
        logger.error(f"❌ Data file error: {exc}")
        logger.error("💡 Please run data_pipeline/generator.py first to create data!")
        sys.exit(1)

    except Exception as exc:
        logger.error(f"💥 Unexpected error during training: {exc}")
        logger.debug("Detailed error:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
