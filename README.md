# Smoothie Qwen
<p align="center">
    <img src="asset/smoothie-qwen-logo.png" width="400">
</p>

**Smoothie Qwen** is a lightweight adjustment tool that smooths token probabilities in Qwen2.5 models, enhancing balanced multilingual generation capabilities.  We've uploaded adjusted models to our [Smoothie Qwen Collection on 🤗 Hugging Face](https://huggingface.co/collections/dnotitia/private-models-smoothie-qwen-68075260246ae00e76cb4f3a), making them readily available for integration into your projects.

- [dnotitia/Smoothie-Qwen2.5-0.5B-Instruct](https://huggingface.co/dnotitia/Smoothie-Qwen2.5-0.5B-Instruct)
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
- [How It Works](#how-it-works)
- [Notes](#notes)
- [References](#references)

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
- `model.name`: Name or path of the base model to modify.
- `model.output_path`: Directory path to save the modified model (default: ./modified_model).
- `analysis.method`: Token analysis method (currently supports "ngram").
- `analysis.window_size`: Size of the window used for combining tokens during analysis. (default: 2, range: 2–4)
- `analysis.sample_size`: Number of samples to analyze per token (default: 1000).
- `adjustment.min_scale`: Minimum scaling factor applied to target tokens when fully suppressed (range: 0.0–1.0).
  - 1.0: No weight adjustment.
  - 0.1: The most suppressed target tokens will have their weights multiplied by 0.1.
  - Lower `min_scale` values allow stronger suppression of identified tokens.
- `adjustment.smoothness`: Controls the steepness of weight suppression based on token generation probability (>1).
  - Higher values cause the weight to decrease more sharply even at lower probabilities.
  - Lower values result in a more gradual and smoother suppression curve.
- `unicode_targets`: List of Unicode ranges specifying which language tokens to target.

## How It Works
1. **Token Identification**: Identify tokens in the target Unicode ranges, including broken or malformed tokens from subword tokenization (e.g., BPE artifacts).
2. **Token Combination Analysis**: Analyze token sequences using N-gram methods to detect combinations that are likely to produce the target language.
3. **Weight Smoothing**: Adjust (down-weight) the probabilities of the identified tokens in the `lm_head` layer based on the specified min_scale and smoothness parameters.
4. **Model Saving**: Save the model with updated token weights to a new directory for later use or deployment.

### Weight Smoothing Formula

The scale factor **S** applied to each token's weight is calculated as:

<p align="center">
  <img src="asset/formula.png" width="850">
</p>

where:
- `min_scale` defines the minimum weight scaling allowed (between 0 and 1).
- `smoothness` controls the sharpness of suppression (higher values = more aggressive).
- `weighted_prob` is the estimated probability of generating the target language (between 0 and 1).

The original token weight is multiplied by **S** to smoothly adjust the generation probability.

## Notes
- This method modifies the model weights directly. It is recommended to validate the model’s performance after applying these changes.
- Unicode target ranges can be customized to suppress other languages or specific token patterns.
- Additional analysis methods beyond N-gram may be supported in future versions.

---

## Experiments
We conducted example experiments using **Qwen2.5-Coder-14B-Instruct** to demonstrate how **Smoothie Qwen** can be tuned to suppress unintended Chinese generation while maintaining core task performance. These settings are provided as a reference and can be freely adjusted according to user preferences and objectives.

> **Note:**
> These experiments were intended as simple tests, using a minimal n-gram window size of 2 to validate basic behavior with limited token combinations.

### Example Setup
- **Base Model**: `Qwen2.5-Coder-14B-Instruct`
- **Evaluation Tool**:  
  - Customized [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) was used for evaluation.

### Evaluation Metrics and Datasets

- **no_chinese_score**  
  Measures whether generated text contains Chinese characters.
  - **1.0**: No Chinese detected (successful suppression)
  - **0.0**: Chinese detected (suppression failed)

  This metric was evaluated using:
  - [**KMMLU**](https://huggingface.co/datasets/HAERAE-HUB/KMMLU):
    - Korean prompts without reference answers, freeform generation.
    - Evaluated on:
      - `kmmlu_generation_nochinese_cs` (Computer Science)
      - `kmmlu_generation_nochinese_ie` (Industrial Engineering)
  - [**Custom Chinese Prompts**](https://huggingface.co/datasets/dnotitia/chinese-prompts_v001):
    - Synthetic dataset combining Korean words/phrases with templates encouraging Chinese translation.
    - Covers diverse categories (numbers, colors, food, family, animals, places, etc.).

- **acc**  
  Standard accuracy on KMMLU multiple-choice tasks, measuring whether the model maintains core task performance.

> **Note:**
> Smoothie Qwen is designed for flexibility. Users can freely choose datasets and evaluation methods according to their goals.

---

## 1. Weight Adjustment Summary

- **Vocabulary size**: 151,643 tokens
- **Target tokens**: 26,153 tokens (17.25%)
- **Broken tokens**: 1,457 tokens (0.96%)

<p align="center">
  <img src="asset/token_weight_heatmap_2d.png" width="600">
</p>

<p align="center">
  <img src="asset/token_weight_heatmap_3d.png" width="600">
</p>

---

## 2. Experiment 1: min_scale Adjustment

The `min_scale` parameter controls the maximum reduction of token weights for identified Chinese tokens.

| Experiment | min_scale | chin_prom | chin_cs | chin_ie | acc_cs | acc_ie |
|:----------:|:---------:|:---------:|:-------:|:-------:|:------:|:------:|
| base (m10) | 1.0       | 0.190     | 0.995   | 0.990   | 0.715  | 0.385  |
| m09        | 0.9       | 0.250     | 0.995   | 1.000   | 0.710  | 0.395  |
| m08        | 0.8       | 0.375     | 0.995   | 1.000   | 0.710  | 0.395  |
| m07        | 0.7       | 0.605     | 0.995   | 1.000   | 0.710  | 0.395  |
| m06        | 0.6       | 0.875     | 0.995   | 1.000   | 0.710  | 0.395  |
| m05        | 0.5       | 0.950     | 0.995   | 1.000   | 0.710  | 0.395  |
| m04        | 0.4       | 0.965     | 0.995   | 1.000   | 0.710  | 0.395  |
| m03        | 0.3       | 0.980     | 0.995   | 1.000   | 0.710  | 0.395  |
| m02        | 0.2       | 0.985     | 1.000   | 1.000   | 0.710  | 0.395  |
| m01        | 0.1       | 0.990     | 1.000   | 1.000   | 0.710  | 0.395  |

- Lowering `min_scale` significantly improves Chinese suppression scores.
- Task performance (`acc_cs`, `acc_ie`) remains stable across all settings.
- **Reasonable trade-off**: `min_scale = 0.5`.

<p align="center">
  <img src="asset/exp1_min_scale.png" width="600">
</p>

---

## 3. Experiment 2: smoothness Adjustment

The `smoothness` parameter controls the curvature of the scaling applied to token weights during smoothing.

| Experiment | smoothness | chin_prom | chin_cs | chin_ie | acc_cs | acc_ie |
|:----------:|:----------:|:---------:|:-------:|:-------:|:------:|:------:|
| log1_1     | 1.1        | 0.920     | 0.995   | 1.000   | 0.710  | 0.395  |
| log10      | 10.0       | 0.950     | 0.995   | 1.000   | 0.710  | 0.395  |
| log100     | 100.0      | 0.990     | 1.000   | 1.000   | 0.710  | 0.395  |
| log1000    | 1000.0     | 1.000     | 1.000   | 1.000   | 0.710  | 0.395  |

- Increasing `smoothness` leads to stronger suppression performance.
- The selected value (`smoothness = 10.0`) provided satisfactory results.

<p align="center">
  <img src="asset/exp2_smoothness.png" width="600">
</p>

---

## Conclusion

Through targeted token weight adjustment using **Smoothie Qwen**,
we achieve **over 90% reduction** in unintended Chinese generation while maintaining the model’s original task capabilities.

- **Suggested configuration**:
  - `min_scale = 0.5`
  - `smoothness = 10.0`

These settings represent a practical balance between suppression effectiveness and task performance,
but users are encouraged to adjust them further depending on their specific goals.

Smoothie Qwen models are ready to be adapted into projects requiring more balanced and controlled multilingual generation.


## References
- Logo design with ❤️ by [J비주얼스쿨](https://www.jvisualschool.com/)
- [Qwen2.5 모델 확률 조정을 통해 중국어 안나오게 하기](https://www.linkedin.com/posts/jg-choi_github-workddllmforeignblock-llm-%EB%AA%A8%EB%8D%B8%EC%9D%98-activity-7306159255936540673-_RoZ), LinkedIn