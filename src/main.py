import argparse
import yaml
import os
import sys
import logging

from datetime import datetime

from identify import TokenIdentifier
from analyze import TokenAnalyzer
from adjust import WeightAdjuster
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging (default is verbose).",
    )
    parser.add_argument(
        "--cache", action="store_true", help="Use cached token analysis if available."
    )
    args = parser.parse_args()

    # Load configuration
    config = read_config(args.config)

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
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
    ngram_weights = config.analysis.ngram_weights

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
    identifier = TokenIdentifier(
        model_name=model_name,
        unicode_ranges=unicode_ranges,
        cache_dir=cache_dir,
        model_dtype=model_dtype,
        verbose=not args.quiet,
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

    # Step 2: Analyze token combinations
    logger.info("2. Starting token combination analysis...")
    analyzer = TokenAnalyzer(
        model_name=model_name,
        tokenizer=identifier.tokenizer,
        target_tokens=identifier.target_tokens,
        broken_tokens=identifier.broken_tokens,
        unicode_ranges=unicode_ranges,
        cache_dir=cache_dir,
        verbose=not args.quiet,
    )

    if not cache_loaded:
        if analysis_method == "ngram":
            analyzer.analyze_ngram_combinations(
                sample_size=sample_size, max_ngram=window_size
            )
            analyzer.save_token_data()
    else:
        analyzer.load_token_data()

    # Step 3: Adjust token weights
    logger.info("3. Starting weight adjustment...")
    adjuster = WeightAdjuster(
        model=identifier.model,
        tokenizer=identifier.tokenizer,
        target_tokens=identifier.target_tokens,
        token_analysis=analyzer.token_analysis,
        verbose=not args.quiet,
    )

    modified_count = None
    if analysis_method == "ngram":

        logger.info(
            f"Using ngram weights for window size {window_size}: {ngram_weights}"
        )

        modified_count = adjuster.modify_weights(
            ngram_weights=ngram_weights,
            target_scale_factor=min_scale,
            log_base=smoothness,
            minimum_scale_factor=min_scale,
        )
    else:
        # Force the use of n-gram analysis
        logger.error(f"Unsupported analysis method: {analysis_method}")
        logger.error("Only 'ngram' analysis method is currently supported.")
        return 1

    # Step 4: Save the modified model
    logger.info("4. Saving modified model...")
    adjuster.save_modified_model(model_output_dir)

    # Log results
    logger.info("=" * 50)
    logger.info("Token weight smoothing completed")
    logger.info(f"- Model: {model_name}")
    logger.info(f"- Output path: {model_output_dir}")
    logger.info(f"- Target tokens: {len(identifier.target_tokens):,}")
    logger.info(f"- Broken tokens: {len(identifier.broken_tokens):,}")
    logger.info(f"- Modified tokens: {modified_count:,}")
    logger.info(f"- Cache used: {'Yes' if cache_loaded else 'No'}")
    logger.info("=" * 50)

    # Create README
    with open(os.path.join(model_output_dir, "README.md"), "w") as f:
        f.write(f"# Modified Model: {model_name}\n\n")
        f.write("## Configuration\n")
        f.write(f"- Base model: {model_name}\n")
        f.write(f"- Minimum scale factor: {min_scale}\n")
        f.write(f"- Smoothness: {smoothness}\n")
        f.write(f"- Sample size: {sample_size}\n")
        f.write(f"- Window size: {window_size}\n")
        f.write(f"- N-gram weights: {ngram_weights}\n\n")
        f.write("## Unicode Ranges\n")
        for i, r in enumerate(unicode_ranges):
            f.write(f"- Range {i+1}: {hex(r[0])} - {hex(r[1])}\n")
        f.write("\n## Statistics\n")
        f.write(f"- Target tokens: {len(identifier.target_tokens):,}\n")
        f.write(f"- Broken tokens: {len(identifier.broken_tokens):,}\n")
        f.write(f"- Modified tokens: {modified_count:,}\n")

    logger.info(f"Configuration saved to {os.path.join(model_output_dir, 'README.md')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
