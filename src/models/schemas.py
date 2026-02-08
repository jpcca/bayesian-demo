"""
Data models for the height/weight prediction evaluation project.
Adapted from the transcribe project's api/models.py
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage statistics for a single query.

    All four token types are billable at different rates:
    - input_tokens: 1.0x base input price
    - output_tokens: model's output price
    - cache_creation_input_tokens: 1.25x base input price
    - cache_read_input_tokens: 0.1x base input price (90% discount, not free)

    For cost estimation, price each category separately — do not multiply
    total_tokens by a single rate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    total_tokens: int = 0  # All tokens processed (input + output + cache creation + cache read)
    num_turns: int = 0  # Number of unique message IDs (turns)


class DistributionParams(BaseModel):
    """Parameters for a probability distribution."""

    distribution_type: Literal["normal", "lognormal", "truncated_normal"]
    mu: float = Field(..., description="Mean parameter")
    sigma: float = Field(..., gt=0, description="Standard deviation (must be positive)")
    unit: Literal["cm", "kg"]


class BeliefUpdateStep(BaseModel):
    """A single step in the sequential Bayesian belief updating process."""

    step: int = Field(..., description="Step number in the sequence")
    evidence: str = Field(..., description="The piece of evidence being incorporated")
    prior_height: Optional[Dict[str, float]] = Field(None, description="Height distribution before update")
    update_rationale: str = Field(..., description="Explanation of how evidence changes belief")
    posterior_height: Optional[Dict[str, float]] = Field(None, description="Height distribution after update")
    prior_weight: Optional[Dict[str, float]] = Field(None, description="Weight distribution before update")
    posterior_weight: Optional[Dict[str, float]] = Field(None, description="Weight distribution after update")


class PredictionResult(BaseModel):
    """Structured output from a prediction agent."""

    reasoning: str = Field(..., description="Explanation of the prediction process")
    web_searches_performed: List[str] = Field(
        default_factory=list, description="Search queries used (for web search conditions)"
    )
    belief_updates: List[BeliefUpdateStep] = Field(
        default_factory=list, description="Sequential Bayesian belief updates (for probabilistic approach)"
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
    """Ground truth actual measurements for a subject."""

    subject_id: str
    height_cm: float = Field(..., gt=0, description="Actual measured height in centimeters")
    weight_kg: float = Field(..., gt=0, description="Actual measured weight in kilograms")
    text_description: str = Field(..., description="The input paragraph")


class EvaluationMetrics(BaseModel):
    """Metrics for comparing predicted distribution against actual ground truth value."""

    # Negative log-likelihood of true value under predicted distribution (lower is better)
    nll_height: float
    nll_weight: float

    # Absolute error: |predicted_mu - true_value| (lower is better)
    abs_error_height: float
    abs_error_weight: float

    # Z-score: (true_value - predicted_mu) / predicted_sigma
    # Ideally |z| < 2 for 95% of well-calibrated predictions
    z_score_height: float
    z_score_weight: float

    # Coverage: whether true value falls within 95% credible interval
    in_95ci_height: bool
    in_95ci_weight: bool

    # Overall validity
    is_valid: bool


class ExperimentResult(BaseModel):
    """Result for a single subject in an experiment."""

    subject_id: str
    approach: Literal["baseline", "web_search", "probabilistic"]
    prediction: PredictionResult
    ground_truth: GroundTruth
    metrics: Optional[EvaluationMetrics] = None  # None if prediction is invalid
    token_usage: Optional[TokenUsage] = None  # Token usage for this prediction

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
    mean_nll_height: Optional[float] = None
    mean_nll_weight: Optional[float] = None
    mean_abs_error_height: Optional[float] = None
    mean_abs_error_weight: Optional[float] = None
    mean_abs_z_score_height: Optional[float] = None  # Mean of |z-score|
    mean_abs_z_score_weight: Optional[float] = None

    # Coverage: percentage of true values within 95% CI (should be ~95% if well-calibrated)
    coverage_95ci_height_percent: Optional[float] = None
    coverage_95ci_weight_percent: Optional[float] = None

    # Standard deviations (for error bars)
    std_nll_height: Optional[float] = None
    std_nll_weight: Optional[float] = None
    std_abs_error_height: Optional[float] = None
    std_abs_error_weight: Optional[float] = None

    # Token usage statistics (averaged across all predictions, including invalid ones)
    mean_input_tokens: Optional[float] = None
    mean_output_tokens: Optional[float] = None
    mean_total_tokens: Optional[float] = None
    mean_num_turns: Optional[float] = None
    total_tokens_all_predictions: Optional[int] = None


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
