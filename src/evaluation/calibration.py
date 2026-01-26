"""Calibration evaluation metrics for distributional predictions.

This module provides metrics for assessing how well-calibrated distributional
predictions are compared to actual measurements. Key metrics include:

- Coverage: Does the true value fall within predicted intervals?
- Interval Score: A proper scoring rule rewarding calibration and sharpness
- Calibration curves: Comparing expected vs observed coverage rates

References:
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. Journal of the American Statistical Association.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


class SubjectProtocol(Protocol):
    """Protocol for Subject types from either nhanes_loader or schemas."""

    actual_height_cm: float
    actual_weight_kg: float


if TYPE_CHECKING:
    from ..data.nhanes_loader import Subject as NHANESSubject
    from ..models.schemas import (
        AggregatedCalibrationMetrics,
        CalibrationMetrics,
        ExperimentResult,
        PredictionResult,
        Subject as SchemaSubject,
    )

    # Accept either Subject type
    Subject = Union[NHANESSubject, SchemaSubject]

# Default confidence levels for coverage evaluation
DEFAULT_COVERAGE_LEVELS = [0.50, 0.80, 0.90, 0.95]


def check_coverage(
    predicted_mu: float,
    predicted_sigma: float,
    true_value: float,
    levels: Optional[List[float]] = None,
) -> Dict[float, bool]:
    """Check if true value falls within predicted intervals at various confidence levels.

    For a Normal(mu, sigma) distribution, the interval at level p is:
    [mu - z * sigma, mu + z * sigma] where z = norm.ppf((1+p)/2)

    Args:
        predicted_mu: Mean of the predicted distribution
        predicted_sigma: Standard deviation of the predicted distribution
        true_value: The actual observed value
        levels: Confidence levels to check (default: [0.50, 0.80, 0.90, 0.95])

    Returns:
        Dictionary mapping each confidence level to whether the true value
        falls within that interval.
        Example: {0.50: True, 0.80: True, 0.90: False, 0.95: False}

    Raises:
        ValueError: If predicted_sigma <= 0

    Example:
        >>> result = check_coverage(170.0, 5.0, 172.0)
        >>> result[0.50]  # True if 172 is in 50% interval around 170
        True
    """
    if predicted_sigma <= 0:
        raise ValueError(f"predicted_sigma must be positive, got {predicted_sigma}")

    if levels is None:
        levels = DEFAULT_COVERAGE_LEVELS

    coverage: Dict[float, bool] = {}
    for level in levels:
        # z-score for two-tailed interval at this confidence level
        z = stats.norm.ppf((1 + level) / 2)
        lower = predicted_mu - z * predicted_sigma
        upper = predicted_mu + z * predicted_sigma
        coverage[level] = lower <= true_value <= upper

    return coverage


def interval_score(
    predicted_mu: float,
    predicted_sigma: float,
    true_value: float,
    alpha: float = 0.10,
) -> float:
    """Calculate interval score (Gneiting & Raftery, 2007).

    This is a proper scoring rule that rewards both calibration and sharpness.
    A well-calibrated, sharp prediction minimizes this score.

    For a (1-alpha) prediction interval [lower, upper]:
    IS = (upper - lower) + (2/alpha) * (lower - y) * 1{y < lower}
                        + (2/alpha) * (y - upper) * 1{y > upper}

    The score has three components:
    1. Interval width (sharpness penalty)
    2. Penalty for observations below the interval
    3. Penalty for observations above the interval

    Args:
        predicted_mu: Mean of the predicted distribution
        predicted_sigma: Standard deviation of the predicted distribution
        true_value: The actual observed value
        alpha: Significance level (default 0.10 for 90% interval)

    Returns:
        Interval score (lower is better)

    Raises:
        ValueError: If predicted_sigma <= 0 or alpha not in (0, 1)

    Example:
        >>> # Perfect prediction should have low score
        >>> score = interval_score(170.0, 5.0, 170.0)
        >>> score > 0  # Always positive
        True
    """
    if predicted_sigma <= 0:
        raise ValueError(f"predicted_sigma must be positive, got {predicted_sigma}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Compute interval bounds for (1-alpha) confidence level
    z = stats.norm.ppf(1 - alpha / 2)
    lower = predicted_mu - z * predicted_sigma
    upper = predicted_mu + z * predicted_sigma

    # Base score: interval width
    score = upper - lower

    # Penalty for observation below interval
    if true_value < lower:
        score += (2 / alpha) * (lower - true_value)

    # Penalty for observation above interval
    if true_value > upper:
        score += (2 / alpha) * (true_value - upper)

    return float(score)


def _compute_interval_width(
    predicted_mu: float,
    predicted_sigma: float,
    level: float = 0.90,
) -> float:
    """Compute the width of a prediction interval.

    Args:
        predicted_mu: Mean of the predicted distribution (unused, kept for API consistency)
        predicted_sigma: Standard deviation of the predicted distribution
        level: Confidence level (default 0.90)

    Returns:
        Width of the prediction interval
    """
    z = stats.norm.ppf((1 + level) / 2)
    return 2 * z * predicted_sigma


def evaluate_calibration(
    prediction: "PredictionResult",
    subject: SubjectProtocol,
) -> Optional["CalibrationMetrics"]:
    """Evaluate calibration of a single prediction.

    Args:
        prediction: The distributional prediction result
        subject: The subject with actual measurements

    Returns:
        CalibrationMetrics object with coverage, interval scores, and MAE.
        Returns None if prediction is invalid or missing required data.

    Example:
        >>> from models.schemas import PredictionResult, Subject, Sex
        >>> pred = PredictionResult(height_mu=170.0, height_sigma=5.0,
        ...                         weight_mu=70.0, weight_sigma=3.0)
        >>> subj = Subject(age_months=120, sex=Sex.MALE,
        ...                actual_height_cm=172.0, actual_weight_kg=68.0)
        >>> metrics = evaluate_calibration(pred, subj)
        >>> metrics is not None
        True
    """
    # Lazy import to avoid circular dependencies
    from ..models.schemas import CalibrationMetrics

    # Check for valid prediction
    if not prediction.is_valid:
        return None

    metrics = CalibrationMetrics()

    # Evaluate height calibration
    if (
        prediction.height_mu is not None
        and prediction.height_sigma is not None
        and prediction.height_sigma > 0
        and subject.actual_height_cm is not None
    ):
        metrics.height_coverage = check_coverage(
            prediction.height_mu,
            prediction.height_sigma,
            subject.actual_height_cm,
        )
        metrics.height_interval_score = interval_score(
            prediction.height_mu,
            prediction.height_sigma,
            subject.actual_height_cm,
        )
        metrics.height_mae = abs(prediction.height_mu - subject.actual_height_cm)
        metrics.height_interval_width_90 = _compute_interval_width(
            prediction.height_mu,
            prediction.height_sigma,
            level=0.90,
        )

    # Evaluate weight calibration
    if (
        prediction.weight_mu is not None
        and prediction.weight_sigma is not None
        and prediction.weight_sigma > 0
        and subject.actual_weight_kg is not None
    ):
        metrics.weight_coverage = check_coverage(
            prediction.weight_mu,
            prediction.weight_sigma,
            subject.actual_weight_kg,
        )
        metrics.weight_interval_score = interval_score(
            prediction.weight_mu,
            prediction.weight_sigma,
            subject.actual_weight_kg,
        )
        metrics.weight_mae = abs(prediction.weight_mu - subject.actual_weight_kg)
        metrics.weight_interval_width_90 = _compute_interval_width(
            prediction.weight_mu,
            prediction.weight_sigma,
            level=0.90,
        )

    return metrics


def aggregate_calibration_results(
    results: List["ExperimentResult"],
) -> "AggregatedCalibrationMetrics":
    """Aggregate calibration metrics across all subjects for one approach.

    Computes:
    - Coverage rates at each level
    - Calibration error = |observed_coverage - nominal_coverage|
    - Mean interval widths
    - Mean interval scores
    - Mean MAE

    Args:
        results: List of ExperimentResult objects from a single approach

    Returns:
        AggregatedCalibrationMetrics with summary statistics

    Example:
        >>> # After running experiments
        >>> agg = aggregate_calibration_results(experiment_results)
        >>> agg.height_coverage_rates[0.90]  # Should be close to 0.90 if well-calibrated
        0.88
    """
    # Lazy import to avoid circular dependencies
    from ..models.schemas import AggregatedCalibrationMetrics

    agg = AggregatedCalibrationMetrics()

    # Collect metrics from valid results
    height_coverages: Dict[float, List[bool]] = {level: [] for level in DEFAULT_COVERAGE_LEVELS}
    weight_coverages: Dict[float, List[bool]] = {level: [] for level in DEFAULT_COVERAGE_LEVELS}
    height_interval_scores: List[float] = []
    weight_interval_scores: List[float] = []
    height_maes: List[float] = []
    weight_maes: List[float] = []
    height_widths: List[float] = []
    weight_widths: List[float] = []

    for result in results:
        if result.calibration_metrics is None:
            continue

        cal = result.calibration_metrics

        # Collect height metrics
        if cal.height_coverage:
            for level, covered in cal.height_coverage.items():
                if level in height_coverages:
                    height_coverages[level].append(covered)
        if cal.height_interval_score is not None:
            height_interval_scores.append(cal.height_interval_score)
        if cal.height_mae is not None:
            height_maes.append(cal.height_mae)
        if cal.height_interval_width_90 is not None:
            height_widths.append(cal.height_interval_width_90)

        # Collect weight metrics
        if cal.weight_coverage:
            for level, covered in cal.weight_coverage.items():
                if level in weight_coverages:
                    weight_coverages[level].append(covered)
        if cal.weight_interval_score is not None:
            weight_interval_scores.append(cal.weight_interval_score)
        if cal.weight_mae is not None:
            weight_maes.append(cal.weight_mae)
        if cal.weight_interval_width_90 is not None:
            weight_widths.append(cal.weight_interval_width_90)

    # Compute aggregated metrics
    agg.n_predictions = max(
        len(height_interval_scores),
        len(weight_interval_scores),
    )

    # Coverage rates and calibration errors for height
    for level in DEFAULT_COVERAGE_LEVELS:
        if height_coverages[level]:
            observed_rate = float(np.mean(height_coverages[level]))
            agg.height_coverage_rates[level] = observed_rate
            agg.height_calibration_error[level] = abs(observed_rate - level)

    # Coverage rates and calibration errors for weight
    for level in DEFAULT_COVERAGE_LEVELS:
        if weight_coverages[level]:
            observed_rate = float(np.mean(weight_coverages[level]))
            agg.weight_coverage_rates[level] = observed_rate
            agg.weight_calibration_error[level] = abs(observed_rate - level)

    # Mean scores
    if height_interval_scores:
        agg.mean_height_interval_score = float(np.mean(height_interval_scores))
    if weight_interval_scores:
        agg.mean_weight_interval_score = float(np.mean(weight_interval_scores))

    # Mean MAE
    if height_maes:
        agg.mean_height_mae = float(np.mean(height_maes))
    if weight_maes:
        agg.mean_weight_mae = float(np.mean(weight_maes))

    # Mean interval widths
    if height_widths:
        agg.mean_height_interval_width_90 = float(np.mean(height_widths))
    if weight_widths:
        agg.mean_weight_interval_width_90 = float(np.mean(weight_widths))

    return agg


def calibration_curve(
    results: List["ExperimentResult"],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve for plotting.

    A well-calibrated model has observed coverage approximately equal to
    expected coverage at all levels (points lie on the diagonal).

    Args:
        results: List of ExperimentResult objects
        n_bins: Number of probability bins (default 10)

    Returns:
        Tuple of:
        - expected: nominal coverage levels (e.g., [0.1, 0.2, ..., 1.0])
        - observed_height: actual coverage at each level for height
        - observed_weight: actual coverage at each level for weight

    Example:
        >>> expected, obs_h, obs_w = calibration_curve(results)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
        >>> plt.plot(expected, obs_h, 'b-', label='Height')
        >>> plt.plot(expected, obs_w, 'r-', label='Weight')
    """
    # Expected coverage levels
    expected = np.linspace(1 / n_bins, 1.0, n_bins)

    # Compute coverage at each level
    observed_height = np.zeros(n_bins)
    observed_weight = np.zeros(n_bins)

    height_counts = np.zeros(n_bins, dtype=int)
    weight_counts = np.zeros(n_bins, dtype=int)

    for result in results:
        if result.calibration_metrics is None:
            continue
        if not result.prediction.is_valid:
            continue

        pred = result.prediction
        subj = result.subject

        # Skip if no subject data
        if subj is None:
            continue

        # Height calibration curve
        if (
            pred.height_mu is not None
            and pred.height_sigma is not None
            and pred.height_sigma > 0
            and subj.actual_height_cm is not None
        ):
            for i, level in enumerate(expected):
                z = stats.norm.ppf((1 + level) / 2)
                lower = pred.height_mu - z * pred.height_sigma
                upper = pred.height_mu + z * pred.height_sigma
                if lower <= subj.actual_height_cm <= upper:
                    observed_height[i] += 1
                height_counts[i] += 1

        # Weight calibration curve
        if (
            pred.weight_mu is not None
            and pred.weight_sigma is not None
            and pred.weight_sigma > 0
            and subj.actual_weight_kg is not None
        ):
            for i, level in enumerate(expected):
                z = stats.norm.ppf((1 + level) / 2)
                lower = pred.weight_mu - z * pred.weight_sigma
                upper = pred.weight_mu + z * pred.weight_sigma
                if lower <= subj.actual_weight_kg <= upper:
                    observed_weight[i] += 1
                weight_counts[i] += 1

    # Compute rates (avoid division by zero)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed_height = np.where(
            height_counts > 0,
            observed_height / height_counts,
            np.nan,
        )
        observed_weight = np.where(
            weight_counts > 0,
            observed_weight / weight_counts,
            np.nan,
        )

    return expected, observed_height, observed_weight
