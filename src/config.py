from pydantic import BaseModel, Field, conint, confloat
from typing import List, Tuple, Literal


class ModelConfig(BaseModel):
    name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct", description="Target model to modify"
    )
    output_path: str = Field(
        default="./modified_model", description="Path to save the modified model"
    )


class AnalysisConfig(BaseModel):
    # TODO : Add support for other analysis methods
    method: Literal["ngram"] = "ngram"
    window_size: conint(ge=2, le=4) = Field(
        default=2,
        description="Window size used for token combination analysis (default: 2, range: 2–4)",
    )
    sample_size: conint(ge=1) = Field(
        default=1000, description="Number of samples per token for analysis"
    )


class AdjustmentConfig(BaseModel):
    min_scale: confloat(ge=0.0, le=1.0) = Field(
        default=0.5,
        description="Minimum scaling factor for target token weights (0.0–1.0)",
    )
    smoothness: confloat(gt=1.0) = Field(
        default=10.0,
        description="Smoothing intensity (>1). Higher = more aggressive down-weighting",
    )


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
