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

    @property
    def height_mu(self) -> Optional[float]:
        """Helper property for calibration module compatibility."""
        return self.height_distribution.mu if self.height_distribution else None

    @property
    def height_sigma(self) -> Optional[float]:
        """Helper property for calibration module compatibility."""
        return self.height_distribution.sigma if self.height_distribution else None

    @property
    def weight_mu(self) -> Optional[float]:
        """Helper property for calibration module compatibility."""
        return self.weight_distribution.mu if self.weight_distribution else None

    @property
    def weight_sigma(self) -> Optional[float]:
        """Helper property for calibration module compatibility."""
        return self.weight_distribution.sigma if self.weight_distribution else None


class GroundTruth(BaseModel):
    """Ground truth distribution for a subject (legacy model for synthetic data)."""

    subject_id: str
    height: DistributionParams
    weight: DistributionParams
    text_description: str = Field(..., description="The input paragraph")


class Subject(BaseModel):
    """A subject with actual measurements for calibration.

    Used with NHANES data where we have real height/weight measurements
    rather than synthetic ground truth distributions.
    """

    subject_id: str
    text_description: str = Field(..., description="The input paragraph describing the subject")
    actual_height_cm: float = Field(..., description="Actual measured height in centimeters")
    actual_weight_kg: float = Field(..., description="Actual measured weight in kilograms")
    # Metadata
    age: int = Field(..., description="Age of the subject in years")
    gender: str = Field(..., description="Gender of the subject")
    ethnicity: str = Field(..., description="Ethnicity of the subject")


class EvaluationMetrics(BaseModel):
    """Metrics for comparing predicted and ground truth distributions (legacy)."""

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


class CalibrationMetrics(BaseModel):
    """Calibration assessment for a single prediction.

    Measures how well the predicted uncertainty intervals capture the true value.
    A well-calibrated model should have X% of true values fall within X% intervals.

    Uses dict-based coverage for flexibility with different confidence levels.
    """

    # Coverage at various confidence levels: {0.50: True, 0.80: True, ...}
    height_coverage: Optional[Dict[float, bool]] = Field(
        default=None, description="Coverage at each confidence level for height"
    )
    weight_coverage: Optional[Dict[float, bool]] = Field(
        default=None, description="Coverage at each confidence level for weight"
    )

    # Interval widths (sharpness - narrower is better IF calibrated)
    height_interval_width_90: Optional[float] = Field(
        default=None, description="Width of the 90% prediction interval for height (cm)"
    )
    weight_interval_width_90: Optional[float] = Field(
        default=None, description="Width of the 90% prediction interval for weight (kg)"
    )

    # Interval scores (proper scoring rule - lower is better)
    height_interval_score: Optional[float] = Field(
        default=None, description="Interval score for height prediction"
    )
    weight_interval_score: Optional[float] = Field(
        default=None, description="Interval score for weight prediction"
    )

    # Point accuracy (MAE of predicted mean vs actual)
    height_mae: Optional[float] = Field(
        default=None, description="Mean absolute error of height prediction (cm)"
    )
    weight_mae: Optional[float] = Field(
        default=None, description="Mean absolute error of weight prediction (kg)"
    )


class AggregatedCalibrationMetrics(BaseModel):
    """Aggregated calibration metrics across subjects.

    Coverage rates should match nominal levels if model is well-calibrated:
    - 50% coverage should be ~0.50
    - 80% coverage should be ~0.80
    - 90% coverage should be ~0.90
    - 95% coverage should be ~0.95

    Uses dict-based coverage rates for flexibility with different confidence levels.
    """

    # Metadata
    approach: Optional[str] = Field(default=None, description="The prediction approach used")
    n_predictions: int = Field(default=0, description="Number of valid predictions")

    # Coverage rates: {0.50: 0.52, 0.80: 0.78, ...}
    height_coverage_rates: Dict[float, float] = Field(
        default_factory=dict, description="Observed coverage rate at each nominal level for height"
    )
    weight_coverage_rates: Dict[float, float] = Field(
        default_factory=dict, description="Observed coverage rate at each nominal level for weight"
    )

    # Calibration errors: {0.50: 0.02, 0.80: 0.02, ...}
    height_calibration_error: Dict[float, float] = Field(
        default_factory=dict,
        description="Absolute difference between observed and nominal coverage for height",
    )
    weight_calibration_error: Dict[float, float] = Field(
        default_factory=dict,
        description="Absolute difference between observed and nominal coverage for weight",
    )

    # Mean sharpness (lower is better IF calibrated)
    mean_height_interval_width_90: Optional[float] = Field(
        default=None, description="Mean 90% interval width for height (cm)"
    )
    mean_weight_interval_width_90: Optional[float] = Field(
        default=None, description="Mean 90% interval width for weight (kg)"
    )

    # Mean interval scores (lower is better)
    mean_height_interval_score: Optional[float] = Field(
        default=None, description="Mean interval score for height predictions"
    )
    mean_weight_interval_score: Optional[float] = Field(
        default=None, description="Mean interval score for weight predictions"
    )

    # Mean MAE
    mean_height_mae: Optional[float] = Field(
        default=None, description="Mean absolute error for height predictions (cm)"
    )
    mean_weight_mae: Optional[float] = Field(
        default=None, description="Mean absolute error for weight predictions (kg)"
    )


class ExperimentResult(BaseModel):
    """Result for a single subject in an experiment."""

    subject_id: str
    approach: Literal["baseline", "web_search", "probabilistic"]
    prediction: PredictionResult

    # Support both legacy (ground_truth) and new (subject) formats
    ground_truth: Optional[GroundTruth] = Field(
        default=None, description="Ground truth distribution (legacy, for synthetic data)"
    )
    subject: Optional[Subject] = Field(
        default=None, description="Subject with actual measurements (for calibration studies)"
    )

    # Legacy evaluation metrics (for synthetic ground truth comparison)
    metrics: Optional[EvaluationMetrics] = Field(
        default=None, description="Evaluation metrics (None if prediction is invalid)"
    )

    # Calibration metrics (for real measurement comparison)
    calibration_metrics: Optional[CalibrationMetrics] = Field(
        default=None, description="Calibration metrics (None if prediction is invalid)"
    )

    @property
    def is_success(self) -> bool:
        """Whether this prediction was successful and valid."""
        return self.prediction.is_valid and (
            self.metrics is not None or self.calibration_metrics is not None
        )


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across all subjects for one approach (legacy)."""

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
