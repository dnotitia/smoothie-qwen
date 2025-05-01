import logging
import yaml

from pydantic import ValidationError

from config import AppConfig


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (default: INFO)

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("smoothie-qwen")
    logger.setLevel(log_level)

    return logger


def read_config(config_path: str) -> AppConfig:
    """
    Read and validate configuration from YAML file, returning structured config.

    Args:
        config_path: Path to configuration file

    Returns:
        AppConfig instance
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    try:
        config = AppConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration file:\n{e}")

    return config
