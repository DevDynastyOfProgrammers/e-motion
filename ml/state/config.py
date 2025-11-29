"""
Configuration settings for cosine similarity based emotion classification.
"""

from dataclasses import dataclass, field

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pydantic settings for API keys and external service configuration."""

    OPENROUTER_API_KEY: str = ""
    API_URL: str = "https://openrouter.ai/api/v1/chat/completions"

    class Config:
        env_file = ".env"
        case_sensitive = False


@dataclass(frozen=True)
class CosineModelConfig:
    """Configuration for cosine similarity model."""

    emotion_columns: list[str] = field(
        default_factory=lambda: ["prob_angry_disgust", "prob_fear_surprise", "prob_happy", "prob_neutral", "prob_sad"]
    )
    use_confidence: bool = True
    confidence_weight: float = 0.3
    similarity_threshold: float = 0.7
    epsilon: float = 1e-8  # Small constant for numerical stability


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data handling."""

    data_path: str = "data/train/emotional_balance_dataset.csv"
    model_save_path: str = "data/models/synthetic_cosine_emotion_classifier.pkl"
    prototypes_save_path: str = "data/models/synthetic_reset_prototypes.npy"
    plots_dir: str = "logs/plots"
    test_size: float = 0.2
    random_state: int = 42
    delimiter: str = ";"


@dataclass(frozen=True)
class ServiceConfig:
    """Configuration for API service."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    default_model_type: str = "cosine"
    max_batch_size: int = 100
    api_timeout: int = 30
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


# Configuration instances (preserved for backward compatibility)
settings = Settings()
cosine_config = CosineModelConfig()
data_config = DataConfig()
service_config = ServiceConfig()
