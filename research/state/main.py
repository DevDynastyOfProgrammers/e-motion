"""
Main training pipeline script.
Orchestrates data loading, analysis, training, and visualization.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.state.core.preset_mapping import PresetMapping
from research.state.model.trainer import AdvancedCosineTrainer, data_config


def setup_logging() -> None:
    """Configure logging with structured format."""
    logger.remove()

    # Console logging
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

    # File logging (store logs in research folder)
    log_dir = Path("research/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "training_pipeline.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def analyze_emotional_patterns(df: pd.DataFrame) -> dict[str, float]:
    """
    Analyze emotional patterns and correlations in the dataset.
    """
    logger.info("🔍 Analyzing emotional patterns in dataset...")

    emotion_columns = ["prob_angry_disgust", "prob_fear_surprise", "prob_happy", "prob_neutral", "prob_sad"]

    # Basic statistics
    logger.info("📊 Overall emotion statistics:")
    for column in emotion_columns + ["confidence"]:
        if column in df.columns:
            mean_val = df[column].mean()
            std_val = df[column].std()
            logger.info(f"  {column}: mean={mean_val:.4f}, std={std_val:.4f}")

    # Correlation analysis with preset difficulty (mapped to int)
    # We map presets to indices just to check correlation
    preset_codes = df["preset"].map(PresetMapping.PRESET_TO_IDX)

    correlation_results = {}
    for column in emotion_columns + ["confidence"]:
        if column in df.columns:
            # Drop NaN to avoid errors
            valid_df = df.dropna(subset=[column])
            valid_codes = preset_codes[valid_df.index]
            
            correlation = valid_df[column].corr(valid_codes)
            correlation_results[column] = correlation

    logger.info("📈 Correlation with preset difficulty:")
    for emotion, correlation in correlation_results.items():
        if pd.isna(correlation):
            continue
        significance = "***" if abs(correlation) > 0.3 else "**" if abs(correlation) > 0.2 else "*"
        logger.info(f"  {emotion}: {correlation:+.4f} {significance}")

    return correlation_results


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load the dataset using path from trainer config.
    """
    data_path = Path(data_config.data_path)

    if not data_path.exists():
        # Try relative path fallback
        fallback = Path("research/state/data/emotional_balance_dataset.csv")
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path} or {fallback}")

    logger.info(f"📁 Loading dataset from {data_path}")
    df = pd.read_csv(data_path, delimiter=";")
    logger.info(f"✅ Loaded dataset with {len(df):,} samples")

    return df


def apply_preset_remapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preset remapping to compress granular presets into groups.
    """
    logger.info("🔄 Applying preset remapping (simplification)...")

    original_preset_count = len(df["preset"].unique())

    df_remapped = df.copy()
    df_remapped["preset"] = df_remapped["preset"].apply(PresetMapping.remap_preset)

    new_preset_count = len(df_remapped["preset"].unique())
    new_distribution = df_remapped["preset"].value_counts()

    logger.info(f"✅ Compressed {original_preset_count} granular presets into {new_preset_count} groups")
    logger.info("📊 New preset distribution:")
    for preset, count in new_distribution.items():
        logger.info(f"  {preset}: {count:,} samples")

    return df_remapped


def train_and_evaluate_model(df: pd.DataFrame) -> AdvancedCosineTrainer:
    """
    Train and evaluate the advanced similarity model.
    """
    logger.info("🎯 Training advanced similarity model...")

    trainer = AdvancedCosineTrainer()
    trainer.train(df)

    logger.info("✅ Model training completed successfully")
    return trainer


def create_visualizations(trainer: AdvancedCosineTrainer, df: pd.DataFrame) -> None:
    """
    Create and save model visualizations.
    """
    logger.info("📊 Creating model visualizations...")

    # Ensure plot directory exists
    # We use 'plots_dir' if available in data_config, else default
    plots_dir = Path(getattr(data_config, "plots_dir", "research/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Emotional Prototypes
        # Note: In our new trainer, the analyzer is stored as 'research_analyzer'
        prototypes_path = plots_dir / "advanced_emotional_prototypes.html"
        
        # Check if the analyzer has the plotting method (it should if it's the original file)
        if hasattr(trainer.research_analyzer, "plot_emotional_prototypes"):
            trainer.research_analyzer.plot_emotional_prototypes(str(prototypes_path))
            logger.info(f"✅ Emotional prototypes saved to: {prototypes_path}")
        else:
            logger.warning("⚠️ Analyzer missing plot_emotional_prototypes method.")

        # 2. Performance Analysis
        # Note: The trainer should have this method if copied correctly
        if hasattr(trainer, "plot_performance_analysis"):
            performance_path = plots_dir / "advanced_performance.html"
            trainer.plot_performance_analysis(df, str(performance_path))
            logger.info(f"✅ Performance analysis saved to: {performance_path}")

    except Exception as exc:
        logger.warning(f"⚠️ Could not create some visualizations: {exc}")
        # logger.debug("Detailed visualization error:", exc_info=True)


def main() -> None:
    """Main execution entry point."""
    logger.info("🚀 Starting training pipeline...")

    try:
        setup_logging()

        # 1. Load Data
        df = load_and_preprocess_data()
        
        # 2. Preprocess (Remap presets)
        df_processed = apply_preset_remapping(df)

        # 3. Analyze Patterns
        try:
            analyze_emotional_patterns(df_processed)
        except Exception as e:
            logger.warning(f"Skipping pattern analysis due to error: {e}")

        # 4. Train
        trainer = train_and_evaluate_model(df_processed)

        # 5. Visualize
        create_visualizations(trainer, df_processed)

        logger.success("🎉 Pipeline finished successfully!")

    except FileNotFoundError as exc:
        logger.error(f"❌ Data file error: {exc}")
        sys.exit(1)

    except Exception as exc:
        logger.error(f"💥 Unexpected error: {exc}")
        # import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()