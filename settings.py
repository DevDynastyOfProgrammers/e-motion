import os
import sys
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


def get_env_int(key: str, default: int) -> int:
    """Helper to read an int from env with a default fallback."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        # We use a raw print here because logger might not be configured yet
        print(f"WARNING: Invalid integer for {key}={value}. Using default {default}.")
        return default


def get_env_str(key: str, default: str) -> str:
    """Helper to read a string from env with a default fallback."""
    return os.getenv(key, default)


# Window settings
SCREEN_WIDTH: int = get_env_int("SCREEN_WIDTH", 1280)
SCREEN_HEIGHT: int = get_env_int("SCREEN_HEIGHT", 720)
FPS: int = get_env_int("FPS", 60)

# Game settings
PLAYER_VELOCITY: int = get_env_int("PLAYER_VELOCITY", 250)
ENEMY_VELOCITY: int = get_env_int("ENEMY_VELOCITY", 150)

# ML Settings
EMOTION_MODEL_PATH: str = get_env_str("EMOTION_MODEL_PATH", "assets/models/emotion_model.pt")

# System settings
LOG_LEVEL: str = get_env_str("LOG_LEVEL", "INFO")

# --- Logger Configuration ---
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

logger.info(f"Config loaded: {SCREEN_WIDTH}x{SCREEN_HEIGHT} @ {FPS}FPS | Log Level: {LOG_LEVEL}")
