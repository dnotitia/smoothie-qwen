import argparse
import yaml
import os
import logging

from datetime import datetime

from identify import Identifier
# from analyze import Analyzer
# from adjust import Adjuster
from utils import setup_logging, read_config


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Smoothie-Qwen: Token weight smoothing for language suppression."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--cache", action="store_true", help="Use cached token analysis if available"
    )
    args = parser.parse_args()

    # Load configuration
    config = read_config(args.config)

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logger = setup_logging(log_level)

    logger.info("=" * 50)
    logger.info("Smoothie-Qwen: Token weight smoothing started")
    logger.info("=" * 50)

    # Extract configuration parameters
    model_name = config.model.name
    output_path = config.model.output_path
    model_dtype = config.model.dtype

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get analysis configuration
    analysis_method = config.analysis.method
    window_size = config.analysis.window_size
    sample_size = config.analysis.sample_size

    # Get weight adjustment parameters
    min_scale = config.adjustment.min_scale
    smoothness = config.adjustment.smoothness

    # Get Unicode ranges
    unicode_ranges = [
        (target.range[0], target.range[1]) for target in config.unicode_targets
    ]

    # Cache directory setup
    cache_dir = os.path.join(output_path, ".token_cache")

    # Generate timestamp for model output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_basename = os.path.basename(model_name.replace("/", "_"))
    model_output_dir = os.path.join(
        output_path, f"{model_basename}_min{min_scale}_smooth{smoothness}_{timestamp}"
    )

    # Step 1: Identify tokens in the target Unicode ranges
    logger.info("1. Starting token identification...")
    identifier = Identifier(
        model_name=model_name,
        unicode_ranges=unicode_ranges,
        cache_dir=cache_dir,
        model_dtype=model_dtype,
        verbose=args.verbose
    )

    # Load model and tokenizer
    identifier.load_model()

    # Use cache if available and requested
    cache_loaded = False
    if args.cache:
        cache_loaded = identifier.load_token_data()
    
    if not cache_loaded:
        # Identify target and broken tokens
        identifier.identify_tokens()
        identifier.save_token_data()
    

    # TODO: Step 2: Analyze token combinations

if __name__ == "__main__":
    main()
