"""
Evaluation metrics for comparing predicted distributions against actual ground truth values.

Implements metrics for evaluating probabilistic predictions:
1. Negative log-likelihood (NLL) - How likely is the true value under the predicted distribution?
2. Absolute error - Distance from predicted mean to true value
3. Z-score - How many standard deviations away is the true value?
4. Coverage - Does the true value fall within the credible interval?
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from models.schemas import (
        AggregatedMetrics,
        EvaluationMetrics,
        ExperimentResult,
    )


def calculate_nll_normal(pred_mu: float, pred_sigma: float, true_value: float) -> float:
    """
    Calculate negative log-likelihood of true value under predicted normal distribution.

    NLL = 0.5 * log(2π) + log(σ) + (x - μ)² / (2σ²)

    Lower is better. Penalizes both:
    - Being far from the true value (large |x - μ|)
    - Being overconfident when wrong (small σ with large error)

    Args:
        pred_mu: Predicted mean
        pred_sigma: Predicted standard deviation
        true_value: Actual ground truth value

    Returns:
        Negative log-likelihood (lower is better)
    """
    return (
        0.5 * np.log(2 * np.pi)
        + np.log(pred_sigma)
        + (true_value - pred_mu) ** 2 / (2 * pred_sigma**2)
    )


def calculate_z_score(pred_mu: float, pred_sigma: float, true_value: float) -> float:
    """
    Calculate z-score: how many standard deviations the true value is from predicted mean.

    z = (true_value - pred_mu) / pred_sigma

    For well-calibrated predictions:
    - ~68% of z-scores should be in [-1, 1]
    - ~95% of z-scores should be in [-2, 2]
    - ~99.7% of z-scores should be in [-3, 3]

    Args:
        pred_mu: Predicted mean
        pred_sigma: Predicted standard deviation
        true_value: Actual ground truth value

    Returns:
        Z-score (positive if true > predicted mean, negative otherwise)
    """
    return (true_value - pred_mu) / pred_sigma


def is_in_95ci_normal(pred_mu: float, pred_sigma: float, true_value: float) -> bool:
    """
    Check if true value falls within 95% credible interval of predicted distribution.

    For normal distribution, 95% CI is approximately [μ - 1.96σ, μ + 1.96σ].

    Args:
        pred_mu: Predicted mean
        pred_sigma: Predicted standard deviation
        true_value: Actual ground truth value

    Returns:
        True if true_value is within 95% CI, False otherwise
    """
    z = abs(calculate_z_score(pred_mu, pred_sigma, true_value))
    return z <= 1.96  # 95% CI corresponds to |z| <= 1.96


def evaluate_prediction(prediction, ground_truth) -> Optional["EvaluationMetrics"]:
    """
    Evaluate a single prediction against actual ground truth measurements.

    Compares the predicted probability distributions against the actual
    measured height and weight values.

    Args:
        prediction: PredictionResult object with height_distribution and weight_distribution
        ground_truth: GroundTruth object with actual height_cm and weight_kg values

    Returns:
        EvaluationMetrics if prediction is valid, None otherwise
    """
    # Import here to avoid circular dependency
    from models.schemas import EvaluationMetrics

    if not prediction.is_valid:
        return None

    # Get predicted distribution parameters
    pred_height_mu = prediction.height_distribution.mu
    pred_height_sigma = prediction.height_distribution.sigma
    pred_weight_mu = prediction.weight_distribution.mu
    pred_weight_sigma = prediction.weight_distribution.sigma

    # Get actual ground truth values
    true_height = ground_truth.height_cm
    true_weight = ground_truth.weight_kg

    # Calculate height metrics
    nll_height = calculate_nll_normal(pred_height_mu, pred_height_sigma, true_height)
    abs_error_height = abs(pred_height_mu - true_height)
    z_score_height = calculate_z_score(pred_height_mu, pred_height_sigma, true_height)
    in_95ci_height = is_in_95ci_normal(pred_height_mu, pred_height_sigma, true_height)

    # Calculate weight metrics
    nll_weight = calculate_nll_normal(pred_weight_mu, pred_weight_sigma, true_weight)
    abs_error_weight = abs(pred_weight_mu - true_weight)
    z_score_weight = calculate_z_score(pred_weight_mu, pred_weight_sigma, true_weight)
    in_95ci_weight = is_in_95ci_normal(pred_weight_mu, pred_weight_sigma, true_weight)

    return EvaluationMetrics(
        nll_height=nll_height,
        nll_weight=nll_weight,
        abs_error_height=abs_error_height,
        abs_error_weight=abs_error_weight,
        z_score_height=z_score_height,
        z_score_weight=z_score_weight,
        in_95ci_height=in_95ci_height,
        in_95ci_weight=in_95ci_weight,
        is_valid=True,
    )


def aggregate_results(results: List["ExperimentResult"]) -> "AggregatedMetrics":
    """
    Aggregate metrics across all subjects for one approach.

    Args:
        results: List of ExperimentResult objects for one approach

    Returns:
        AggregatedMetrics object with mean metrics and coverage percentages
    """
    # Import here to avoid circular dependency
    from models.schemas import AggregatedMetrics

    if not results:
        raise ValueError("Cannot aggregate empty results list")

    approach = results[0].approach
    n_total = len(results)

    # Filter valid results
    valid_results = [r for r in results if r.is_success]
    n_valid = len(valid_results)
    n_invalid = n_total - n_valid
    invalid_rate = (n_invalid / n_total) * 100

    # If no valid results, return early
    if n_valid == 0:
        return AggregatedMetrics(
            approach=approach,
            n_total=n_total,
            n_valid=0,
            n_invalid=n_invalid,
            invalid_rate_percent=invalid_rate,
        )

    # Extract metrics from valid results
    nll_height = [r.metrics.nll_height for r in valid_results]
    nll_weight = [r.metrics.nll_weight for r in valid_results]
    abs_error_height = [r.metrics.abs_error_height for r in valid_results]
    abs_error_weight = [r.metrics.abs_error_weight for r in valid_results]
    abs_z_score_height = [abs(r.metrics.z_score_height) for r in valid_results]
    abs_z_score_weight = [abs(r.metrics.z_score_weight) for r in valid_results]
    in_95ci_height = [r.metrics.in_95ci_height for r in valid_results]
    in_95ci_weight = [r.metrics.in_95ci_weight for r in valid_results]

    # Calculate coverage percentages
    coverage_height = (sum(in_95ci_height) / n_valid) * 100
    coverage_weight = (sum(in_95ci_weight) / n_valid) * 100

    # Calculate token usage statistics (across ALL predictions, including invalid)
    results_with_tokens = [r for r in results if r.token_usage is not None]
    mean_input_tokens = None
    mean_output_tokens = None
    mean_total_tokens = None
    mean_num_turns = None
    total_tokens_all = None

    if results_with_tokens:
        input_tokens = [r.token_usage.input_tokens for r in results_with_tokens]
        output_tokens = [r.token_usage.output_tokens for r in results_with_tokens]
        total_tokens = [r.token_usage.total_tokens for r in results_with_tokens]
        num_turns = [r.token_usage.num_turns for r in results_with_tokens]

        mean_input_tokens = float(np.mean(input_tokens))
        mean_output_tokens = float(np.mean(output_tokens))
        mean_total_tokens = float(np.mean(total_tokens))
        mean_num_turns = float(np.mean(num_turns))
        total_tokens_all = int(np.sum(total_tokens))

    return AggregatedMetrics(
        approach=approach,
        n_total=n_total,
        n_valid=n_valid,
        n_invalid=n_invalid,
        invalid_rate_percent=invalid_rate,
        # Means
        mean_nll_height=float(np.mean(nll_height)),
        mean_nll_weight=float(np.mean(nll_weight)),
        mean_abs_error_height=float(np.mean(abs_error_height)),
        mean_abs_error_weight=float(np.mean(abs_error_weight)),
        mean_abs_z_score_height=float(np.mean(abs_z_score_height)),
        mean_abs_z_score_weight=float(np.mean(abs_z_score_weight)),
        # Coverage (should be ~95% for well-calibrated predictions)
        coverage_95ci_height_percent=coverage_height,
        coverage_95ci_weight_percent=coverage_weight,
        # Standard deviations (for error bars in plots)
        std_nll_height=float(np.std(nll_height)),
        std_nll_weight=float(np.std(nll_weight)),
        std_abs_error_height=float(np.std(abs_error_height)),
        std_abs_error_weight=float(np.std(abs_error_weight)),
        # Token usage statistics
        mean_input_tokens=mean_input_tokens,
        mean_output_tokens=mean_output_tokens,
        mean_total_tokens=mean_total_tokens,
        mean_num_turns=mean_num_turns,
        total_tokens_all_predictions=total_tokens_all,
    )


def format_results_table(aggregated_metrics: List["AggregatedMetrics"]) -> str:
    """
    Format aggregated metrics as a research paper style table.

    Args:
        aggregated_metrics: List of AggregatedMetrics for different approaches

    Returns:
        Markdown formatted table string
    """
    # Build header
    table = "| Approach | N Valid | Invalid % | NLL (H) | NLL (W) | Abs Err H (cm) | Abs Err W (kg) | Mean |z| H | Mean |z| W | 95% CI H | 95% CI W | Mean Tokens | Mean Turns |\n"
    table += "|----------|---------|-----------|---------|---------|----------------|----------------|----------|----------|----------|----------|-------------|------------|\n"

    # Build rows
    for metrics in aggregated_metrics:
        if metrics.n_valid == 0:
            # No valid predictions - but may still have token usage
            token_str = f"{metrics.mean_total_tokens:.0f}" if metrics.mean_total_tokens else "N/A"
            turns_str = f"{metrics.mean_num_turns:.1f}" if metrics.mean_num_turns else "N/A"
            table += f"| {metrics.approach} | 0 | {metrics.invalid_rate_percent:.1f} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {token_str} | {turns_str} |\n"
        else:
            token_str = f"{metrics.mean_total_tokens:.0f}" if metrics.mean_total_tokens else "N/A"
            turns_str = f"{metrics.mean_num_turns:.1f}" if metrics.mean_num_turns else "N/A"
            table += (
                f"| {metrics.approach} "
                f"| {metrics.n_valid}/{metrics.n_total} "
                f"| {metrics.invalid_rate_percent:.1f} "
                f"| {metrics.mean_nll_height:.2f} "
                f"| {metrics.mean_nll_weight:.2f} "
                f"| {metrics.mean_abs_error_height:.1f} "
                f"| {metrics.mean_abs_error_weight:.1f} "
                f"| {metrics.mean_abs_z_score_height:.2f} "
                f"| {metrics.mean_abs_z_score_weight:.2f} "
                f"| {metrics.coverage_95ci_height_percent:.0f}% "
                f"| {metrics.coverage_95ci_weight_percent:.0f}% "
                f"| {token_str} "
                f"| {turns_str} |\n"
            )

    return table


# Example usage
if __name__ == "__main__":
    # Test NLL calculation
    # Perfect prediction (true value = mean) should have low NLL
    nll = calculate_nll_normal(175, 6, 175)
    print(f"NLL (perfect mean): {nll:.4f}")

    # Prediction off by 1 sigma
    nll = calculate_nll_normal(175, 6, 181)  # 181 = 175 + 6
    print(f"NLL (1 sigma off): {nll:.4f}")

    # Z-score tests
    z = calculate_z_score(175, 6, 175)
    print(f"Z-score (exact): {z:.4f}")  # Should be 0

    z = calculate_z_score(175, 6, 181)
    print(f"Z-score (1 sigma high): {z:.4f}")  # Should be 1

    z = calculate_z_score(175, 6, 169)
    print(f"Z-score (1 sigma low): {z:.4f}")  # Should be -1

    # Coverage tests
    in_ci = is_in_95ci_normal(175, 6, 175)
    print(f"In 95% CI (exact match): {in_ci}")  # Should be True

    in_ci = is_in_95ci_normal(175, 6, 187)  # > 1.96 sigma away
    print(f"In 95% CI (2 sigma away): {in_ci}")  # Should be False
