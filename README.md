# Smoothie-Qwen
Smooths token weights in the lm_head layer to adjust the probability of specific language generation in LLMs.

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

This repository suppresses specific language generation in LLMs by smoothing token weights in the lm_head layer.

It identifies tokens in the target language’s Unicode range, including broken tokens from subword tokenization (e.g., BPE). Since some languages can be formed by combining neutral tokens, it uses techniques like N-gram analysis to detect and down-weight token patterns likely to produce the target language.

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

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

1. Create a YAML configuration file:
```yaml
# Model Configuration
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  output_path: "./modified_model"

# Analysis Configuration
analysis:
  method: "ngram"
  window_size: 2
  sample_size: 1000

# Weight Smoothing Configuration
adjustment:
  min_scale: 0.5
  smoothness: 10.0

# Target Unicode Ranges
unicode_targets:
  - name: CJK Unified Ideographs
    range: [0x4E00, 0x9FFF]
```

2. Run the script:
```bash
# Login to Hugging Face (if needed)
huggingface-cli login

# Run the main script
python src/main.py --config config.yaml
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

