from typing import Literal, List, Annotated, Tuple
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct", description="Target model to modify"
    )
    output_path: str = Field(
        default="./modified_model", description="Path to save the modified model"
    )

    dtype: str = Field(
        default="bfloat16",
        description="Data type for model weights.'",
    )


class AnalysisConfig(BaseModel):
    # TODO: Add support for other analysis methods
    method: Literal["ngram"] = "ngram"

    window_size: Annotated[
        int,
        Field(
            default=2,
            ge=2,
            le=4,
            description="Window size used for token combination analysis (default: 2, range: 2–4)",
        ),
    ]

    sample_size: Annotated[
        int,
        Field(
            default=1000, ge=1, description="Number of samples per token for analysis"
        ),
    ]

    ngram_weights: Annotated[
        List[float],
        Field(
            default=[0.6, 0.3, 0.1],
            description=(
                "Weights for n-gram analysis. For window_size=2, only the first weight is used. "
                "For window_size=3, the first two weights are used. For window_size=4, all three weights are used."
            ),
        ),
    ]

    @field_validator("ngram_weights", mode="after")
    @classmethod
    def validate_ngram_weights(cls, v: List[float], info) -> List[float]:
        """
        Adjust n-gram weights based on the window_size:
        - Use window_size - 1 weights (e.g., 3 weights for window_size=4)
        - Pad with zeros if there are not enough weights
        - Normalize the weights to sum to 1.0
        - If the total is 0 or negative, default to [1.0, 0.0, 0.0]
        - Always return a list of length 3 by padding with zeros if necessary
        """

        window_size = info.data.get("window_size", 2)
        required_len = window_size - 1

        if len(v) < required_len:
            default_weights = [0.0, 0.0, 0.0]
            v = v + default_weights[len(v) : required_len]

        v = v[:required_len]

        weight_sum = sum(v)
        if weight_sum <= 0:
            v = [1.0] + [0.0] * (required_len - 1)
        elif abs(weight_sum - 1.0) > 0.001:
            v = [w / weight_sum for w in v]

        weights = v + [0.0] * (3 - len(v))

        return weights


class AdjustmentConfig(BaseModel):
    min_scale: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Minimum scaling factor for target token weights (0.0–1.0)",
        ),
    ]

    smoothness: Annotated[
        float,
        Field(
            default=10.0,
            gt=1.0,
            description="Smoothing intensity (>1). Higher = more aggressive down-weighting",
        ),
    ]


class UnicodeTarget(BaseModel):
    name: str = Field(
        default="CJK Unified Ideographs",
        description="Name of the Unicode range (Optional)",
    )
    range: Tuple[int, int] = Field(
        default=(0x4E00, 0x9FFF),
        description="Start and end of Unicode range (inclusive)",
    )


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    adjustment: AdjustmentConfig = Field(default_factory=AdjustmentConfig)
    unicode_targets: List[UnicodeTarget] = Field(default_factory=list)
