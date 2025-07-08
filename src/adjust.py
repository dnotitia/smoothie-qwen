import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any


class WeightAdjuster:
    """
    Adjusts token weights in the model's lm_head layer to reduce the probability
    of generating text in target languages.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        target_tokens: List[int],
        token_analysis: Dict[int, Dict[int, float]],
        verbose: bool = False,
    ):
        """
        Initialize the weight adjuster.

        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            target_tokens: List of token IDs containing target characters
            token_analysis: Dictionary of token analysis {token_id: {n: probability, ...}}
            verbose: Enable verbose logging
        """
        # Model and tokenizer
        self.model = model
        self.tokenizer = tokenizer

        # Token data
        self.target_tokens = target_tokens
        self.token_analysis = token_analysis

        # Set up logging
        self.logger = logging.getLogger("WeightAdjuster")
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def compute_scale_factor(
        self, weighted_prob: float, log_base: float, minimum_scale_factor: float
    ) -> float:
        """
        Compute scale factor for token weight adjustment using logarithmic scaling.

        Args:
            weighted_prob: Weighted probability of generating target language (0.0-1.0)
            log_base: Base of logarithmic scaling (higher values = more aggressive)
            minimum_scale_factor: Minimum scale factor to apply

        Returns:
            Scale factor to apply to token weight (minimum_scale_factor-1.0)
        """
        if weighted_prob <= 0:
            return 1.0
        elif weighted_prob >= 1.0:
            return minimum_scale_factor
        else:
            return 1.0 - (1.0 - minimum_scale_factor) * (
                np.log(1 + (log_base - 1) * weighted_prob) / np.log(log_base)
            )

    def modify_weights(
        self,
        ngram_weights: List[float] = [0.7, 0.2, 0.1],
        target_scale_factor: float = 0.01,
        log_base: float = 100,
        minimum_scale_factor: float = 0.01,
    ) -> int:
        """
        Modify token weights in the model's lm_head layer.

        Args:
            ngram_weights: Weights for n-gram probabilities [2-gram, 3-gram, 4-gram]
            target_scale_factor: Scale factor to apply to target tokens
            log_base: Base of logarithmic scaling for broken tokens
            minimum_scale_factor: Minimum scale factor to apply

        Returns:
            Number of modified tokens
        """
        self.logger.info("Starting weight modification...")
        self.logger.info(
            f"Settings: n-gram weights={ngram_weights}, log_base={log_base}, min_scale={minimum_scale_factor}"
        )

        # Normalize n-gram weights
        if sum(ngram_weights) == 0:
            self.logger.warning("All n-gram weights are 0. Using default [1.0, 0, 0]")
            normalized_weights = [1.0, 0, 0]
        else:
            total_weight = sum(ngram_weights)
            normalized_weights = [w / total_weight for w in ngram_weights]

        self.logger.info(
            f"Normalized n-gram weights: 2-gram={normalized_weights[0]:.2f}, "
            f"3-gram={normalized_weights[1]:.2f}, 4-gram={normalized_weights[2]:.2f}"
        )

        modified_count = 0

        with torch.no_grad():
            if not hasattr(self.model, "lm_head"):
                raise ValueError("Model does not have lm_head layer")

            # Get lm_head weights
            lm_head_weight = self.model.lm_head.weight
            self.logger.info(f"lm_head weight shape: {lm_head_weight.shape}")

            # 1. Adjust target token weights
            self.logger.info(
                f"Reducing weights of {len(self.target_tokens):,} target tokens by factor {target_scale_factor:.4f}"
            )

            for token_id in tqdm(self.target_tokens, desc="Processing target tokens"):
                lm_head_weight[token_id] *= target_scale_factor
                modified_count += 1
            # 2. Adjust broken token weights based on n-gram analysis
            self.logger.info(
                f"Processing {len(self.token_analysis):,} analyzed tokens..."
            )

            for token_id, probs in tqdm(
                self.token_analysis.items(), desc="Processing analyzed tokens"
            ):
                # Convert string token ID to integer if needed
                token_id = int(token_id) if isinstance(token_id, str) else token_id

                # Skip if already processed as target token
                if token_id in self.target_tokens:
                    continue

                # Calculate weighted average of n-gram probabilities
                bigram_prob = probs.get(2, 0)
                trigram_prob = probs.get(3, 0)
                fourgram_prob = probs.get(4, 0)

                weighted_prob = (
                    normalized_weights[0] * bigram_prob
                    + normalized_weights[1] * trigram_prob
                    + normalized_weights[2] * fourgram_prob
                )

                # Calculate scale factor
                scale_factor = self.compute_scale_factor(
                    weighted_prob, log_base, minimum_scale_factor
                )

                # Apply scale factor to token weight
                lm_head_weight[token_id] *= scale_factor
                modified_count += 1

        self.logger.info(
            f"Weight modification completed: {modified_count:,} tokens modified"
        )

        return modified_count

    def save_modified_model(self, output_path: str) -> None:
        """
        Save the modified model.

        Args:
            output_path: Directory path to save the model
        """
        self.logger.info(f"Saving modified model to: {output_path}")

        os.makedirs(output_path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        self.logger.info("Model saved successfully")
