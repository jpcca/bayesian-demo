"""Evaluation metrics for prediction quality.

This module provides tools for assessing the quality of distributional predictions:

- Calibration metrics: Coverage, interval scores, calibration curves
- Aggregation: Summary statistics across multiple predictions

Example:
    >>> from evaluation import evaluate_calibration, aggregate_calibration_results
    >>> metrics = evaluate_calibration(prediction, subject)
    >>> agg = aggregate_calibration_results(experiment_results)
"""

from .calibration import (
    DEFAULT_COVERAGE_LEVELS,
    aggregate_calibration_results,
    calibration_curve,
    check_coverage,
    evaluate_calibration,
    interval_score,
)

__all__ = [
    "check_coverage",
    "interval_score",
    "evaluate_calibration",
    "aggregate_calibration_results",
    "calibration_curve",
    "DEFAULT_COVERAGE_LEVELS",
]
