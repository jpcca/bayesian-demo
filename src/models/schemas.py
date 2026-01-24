"""
Data models for the height/weight prediction evaluation project.
Adapted from the transcribe project's api/models.py
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DistributionParams(BaseModel):
    """Parameters for a probability distribution."""

    distribution_type: Literal["normal", "lognormal", "truncated_normal"]
    mu: float = Field(..., description="Mean parameter")
    sigma: float = Field(..., gt=0, description="Standard deviation (must be positive)")
    unit: Literal["cm", "kg"]

    @field_validator("sigma")
    @classmethod
    def validate_sigma_minimum(cls, v: float) -> float:
        """Ensure sigma is not too small (overconfident)."""
        if v < 1.0:
            raise ValueError("Sigma must be at least 1.0 to avoid overconfidence")
        return v


class PredictionResult(BaseModel):
    """Structured output from a prediction agent."""

    reasoning: str = Field(..., description="Explanation of the prediction process")
    web_searches_performed: List[str] = Field(
        default_factory=list, description="Search queries used (for web search conditions)"
    )
    height_distribution: Optional[DistributionParams] = Field(
        None, description="Height distribution parameters"
    )
    weight_distribution: Optional[DistributionParams] = Field(
        None, description="Weight distribution parameters"
    )
    pymc_code: Optional[str] = Field(
        None, description="PyMC model code (for probabilistic approach)"
    )
    error: Optional[str] = Field(None, description="Error message if prediction failed")

    @property
    def is_valid(self) -> bool:
        """Check if this is a valid prediction with distributions."""
        return (
            self.height_distribution is not None
            and self.weight_distribution is not None
            and self.error is None
        )


class GroundTruth(BaseModel):
    """Ground truth distribution for a subject."""

    subject_id: str
    height: DistributionParams
    weight: DistributionParams
    text_description: str = Field(..., description="The input paragraph")


class EvaluationMetrics(BaseModel):
    """Metrics for comparing predicted and ground truth distributions."""

    # KL divergence (lower is better, 0 = perfect match)
    kl_divergence_height: float
    kl_divergence_weight: float

    # Wasserstein distance (lower is better)
    wasserstein_distance_height: float
    wasserstein_distance_weight: float

    # Mean absolute error on mu (lower is better)
    mae_height_mu: float
    mae_weight_mu: float

    # Sigma error (absolute difference)
    sigma_error_height: float
    sigma_error_weight: float

    # Overall validity
    is_valid: bool


class ExperimentResult(BaseModel):
    """Result for a single subject in an experiment."""

    subject_id: str
    approach: Literal["baseline", "web_search", "probabilistic"]
    prediction: PredictionResult
    ground_truth: GroundTruth
    metrics: Optional[EvaluationMetrics] = None  # None if prediction is invalid

    @property
    def is_success(self) -> bool:
        """Whether this prediction was successful and valid."""
        return self.prediction.is_valid and self.metrics is not None


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across all subjects for one approach."""

    approach: Literal["baseline", "web_search", "probabilistic"]
    n_total: int
    n_valid: int
    n_invalid: int
    invalid_rate_percent: float

    # Mean metrics (only computed on valid predictions)
    mean_kl_divergence_height: Optional[float] = None
    mean_kl_divergence_weight: Optional[float] = None
    mean_wasserstein_distance_height: Optional[float] = None
    mean_wasserstein_distance_weight: Optional[float] = None
    mean_mae_height_mu: Optional[float] = None
    mean_mae_weight_mu: Optional[float] = None
    mean_sigma_error_height: Optional[float] = None
    mean_sigma_error_weight: Optional[float] = None

    # Standard deviations (for error bars)
    std_kl_divergence_height: Optional[float] = None
    std_kl_divergence_weight: Optional[float] = None


def sanitize_nulls(data: Any) -> Any:
    """
    Recursively replaces the string "null" with None in a dictionary or list.
    Reused from transcribe project for LLM output handling.
    """
    if isinstance(data, dict):
        return {k: sanitize_nulls(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_nulls(i) for i in data]
    elif data == "null":
        return None
    return data
