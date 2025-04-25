# Smoothie Qwen
A lightweight adjustment tool for smoothing token probabilities in the Qwen2.5 models to encourage balanced multilingual generation. We have uploaded models adjusted using Smoothie Qwen. Explore the complete  collection at [Smoothie Qwen Collection on 🤗 Hugging Face](https://huggingface.co/collections/dnotitia/private-models-smoothie-qwen-68075260246ae00e76cb4f3a) for integration into your projects.
- [dnotitia/Smoothie-Qwen2.5-0.5-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-0.5B-Instruct)
- [dnotitia/Smoothie-Qwen2.5-1.5B-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-1.5B-Instruct)
- [dnotitia/Smoothie-Qwen2.5-3B-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-3B-Instruct)
- [dnotitia/Smoothie-Qwen2.5-7B-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-7B-Instruct)
- [dnotitia/Smoothie-Qwen2.5-14B-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-14B-Instruct)
- [dnotitia/Smoothie-Qwen2.5-32B-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-32B-Instruct)
- [dnotitia/Smoothie-Qwen2.5-72B-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-72B-Instruct)

## Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Output](#output)
- [How It Works](#how-it-works)
- [Notes](#notes)

## Overview

**Smoothie Qwen** is a post-processing tool designed to subtly refine the token distribution in Qwen2.5 models. By analyzing and adjusting token weights particularly those associated with specific Unicode ranges it helps mitigate unintended biases toward certain languages while preserving the model’s core capabilities.

This approach is especially useful for applications requiring balanced multilingual outputs, where overrepresentation of one language might skew results. The tool identifies target tokens through Unicode ranges, including subword tokenization (e.g., partial characters from BPE tokenization), and applies probabilistic smoothing to encourage diversity.

## Key Features

- Token identification based on Unicode ranges of the target language
- Detection of broken or malformed tokens (e.g. `�`) caused by subword tokenization
- Identification of token combinations that may probabilistically form the target language
- Flexible analysis strategies (e.g., N-gram analysis) to detect high-risk token patterns
- Configurable analysis methods with future support for additional techniques
- Adjustment of token weights in the `lm_head` layer to reduce generation likelihood
- Saving of modified models for reuse or deployment
- Automation of model generation across parameter variations (`min_scale`, `smoothness`)


## Installation

This project uses `uv` for dependency management:

```shell
$ uv venv
$ source .venv/bin/activate
$ uv pip install -r requirements.txt
```

## Usage

Run with a YAML configuration file:
```shell
$ python src/main.py --config config.yaml
```

## Parameters
- `model.name`: Name or path of the base model to modify
- `model.output_path`: Directory path to save the modified model (default: ./modified_model)
- `analysis.method`: Token analysis method (currently supports "ngram")
- `analysis.window_size`: Size of the window used for combining tokens during analysis. (default: 2, range: 2–4)
- `analysis.sample_size`: Number of samples to analyze per token (default: 1000)
- `adjustment.min_scale`: Minimum adjustment ratio to reduce token weights (0.0-1.0)
  - 1.0: No change to weights
  - 0.1: Target token weights multiplied by 0.1
  - Lower values reduce the probability of target tokens being generated
- `adjustment.smoothness`: Intensity of smoothing adjustment (>1)
  - Higher values lead to more aggressive weight reduction
  - Lower values produce more gradual, smoother adjustments
- `unicode_targets`: List of Unicode ranges specifying which language tokens to target


## Output
When running the script, the modified models are saved under:
- TODO : 모델 경로 작성


## How It Works
1. **Token Identification**: Identify tokens in the target Unicode ranges, including broken or malformed tokens from subword tokenization (e.g., BPE artifacts).
2. **Token Combination Analysis**: Analyze token sequences using N-gram methods to detect combinations that are likely to produce the target language.
3. **Weight Smoothing**: Adjust (down-weight) the probabilities of the identified tokens in the lm_head layer based on the specified min_scale and smoothness parameters.
4. **Model Saving**: Save the model with updated token weights to a new directory for later use or deployment.


## Notes
- This method modifies the model weights directly. It is recommended to validate the model’s performance after applying these changes.
- Unicode target ranges can be customized to suppress other languages or specific token patterns.
- Additional analysis methods beyond N-gram may be supported in future versions.
