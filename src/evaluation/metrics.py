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
from scipy import stats as scipy_stats
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from models.schemas import (
        AggregatedMetrics,
        DistributionMetrics,
        EvaluationMetrics,
        ExperimentResult,
        PopulationGroundTruth,
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
        # Standard deviations (for error bars in plots), using ddof=1 for sample std
        # Falls back to 0.0 when n_valid < 2 (ddof=1 requires at least 2 samples)
        std_nll_height=float(np.std(nll_height, ddof=1)) if n_valid >= 2 else 0.0,
        std_nll_weight=float(np.std(nll_weight, ddof=1)) if n_valid >= 2 else 0.0,
        std_abs_error_height=float(np.std(abs_error_height, ddof=1)) if n_valid >= 2 else 0.0,
        std_abs_error_weight=float(np.std(abs_error_weight, ddof=1)) if n_valid >= 2 else 0.0,
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
    table = "| Approach | N Valid | Invalid % | NLL (H) | NLL (W) | Abs Err H (cm) | Abs Err W (kg) | Mean |z| H | Mean |z| W | 95% CI H | 95% CI W | Input Tok | Output Tok | Total Tok | Turns |\n"
    table += "|----------|---------|-----------|---------|---------|----------------|----------------|----------|----------|----------|----------|-----------|------------|-----------|-------|\n"

    # Build rows
    for metrics in aggregated_metrics:
        in_tok = f"{metrics.mean_input_tokens:.0f}" if metrics.mean_input_tokens else "N/A"
        out_tok = f"{metrics.mean_output_tokens:.0f}" if metrics.mean_output_tokens else "N/A"
        total_tok = f"{metrics.mean_total_tokens:.0f}" if metrics.mean_total_tokens else "N/A"
        turns_str = f"{metrics.mean_num_turns:.1f}" if metrics.mean_num_turns else "N/A"

        if metrics.n_valid == 0:
            table += f"| {metrics.approach} | 0 | {metrics.invalid_rate_percent:.1f} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {in_tok} | {out_tok} | {total_tok} | {turns_str} |\n"
        else:
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
                f"| {in_tok} "
                f"| {out_tok} "
                f"| {total_tok} "
                f"| {turns_str} |\n"
            )

    return table


# ============================================================================
# Distribution-to-Distribution Metrics
# ============================================================================
# These functions compare two Normal distributions (predicted vs ground truth
# population) rather than a distribution vs a single point.


def kl_divergence_normal(
    mu_gt: float, sigma_gt: float, mu_pred: float, sigma_pred: float
) -> float:
    """
    KL divergence from ground truth to predicted distribution: KL(gt || pred).

    Measures information lost when using the predicted distribution to
    approximate the ground truth distribution. Lower is better, 0 = identical.

    KL(gt || pred) = log(σ_pred/σ_gt) + (σ_gt² + (μ_gt - μ_pred)²) / (2σ_pred²) - 0.5

    Args:
        mu_gt: Ground truth distribution mean
        sigma_gt: Ground truth distribution std
        mu_pred: Predicted distribution mean
        sigma_pred: Predicted distribution std

    Returns:
        KL divergence (non-negative, lower is better)
    """
    return (
        np.log(sigma_pred / sigma_gt)
        + (sigma_gt**2 + (mu_gt - mu_pred) ** 2) / (2 * sigma_pred**2)
        - 0.5
    )


def wasserstein2_normal(
    mu_gt: float, sigma_gt: float, mu_pred: float, sigma_pred: float
) -> float:
    """
    Squared Wasserstein-2 distance between two Normal distributions.

    Also known as Earth Mover's Distance squared. Measures the "cost" of
    transforming one distribution into the other. Lower is better, 0 = identical.

    W₂² = (μ_pred - μ_gt)² + (σ_pred - σ_gt)²

    Args:
        mu_gt: Ground truth distribution mean
        sigma_gt: Ground truth distribution std
        mu_pred: Predicted distribution mean
        sigma_pred: Predicted distribution std

    Returns:
        Squared Wasserstein-2 distance (non-negative, lower is better)
    """
    return (mu_pred - mu_gt) ** 2 + (sigma_pred - sigma_gt) ** 2


def compute_mean_shift(mu_gt: float, mu_pred: float) -> float:
    """
    Absolute difference between distribution means.

    Simple measure of location mismatch. Lower is better.

    Args:
        mu_gt: Ground truth mean
        mu_pred: Predicted mean

    Returns:
        |μ_pred - μ_gt| (non-negative, lower is better)
    """
    return abs(mu_pred - mu_gt)


def compute_sigma_ratio(sigma_gt: float, sigma_pred: float) -> float:
    """
    Ratio of predicted to ground truth standard deviations.

    Measures calibration of uncertainty. Ideal value is 1.0.
    - > 1.0: predicted distribution is wider (underconfident)
    - < 1.0: predicted distribution is narrower (overconfident)

    Args:
        sigma_gt: Ground truth std
        sigma_pred: Predicted std

    Returns:
        σ_pred / σ_gt (positive, ideal = 1.0)
    """
    return sigma_pred / sigma_gt


def overlap_coefficient_normal(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """
    Overlap coefficient (OVL) between two Normal distributions.

    Computed as the integral of min(f(x), g(x)) dx, which gives the
    area of overlap between the two PDFs. Range [0, 1].
    1.0 = identical distributions, 0.0 = no overlap.

    Uses numerical integration over a wide range for robustness.

    Args:
        mu1, sigma1: First distribution parameters
        mu2, sigma2: Second distribution parameters

    Returns:
        Overlap coefficient in [0, 1] (higher is better)
    """
    # Handle identical distributions
    if abs(mu1 - mu2) < 1e-10 and abs(sigma1 - sigma2) < 1e-10:
        return 1.0

    # Handle equal variances (single intersection point → closed-form)
    if abs(sigma1 - sigma2) < 1e-10:
        # When σ₁ = σ₂, OVL = 2Φ(-|μ₁-μ₂|/(2σ)) where Φ is std normal CDF
        d = abs(mu1 - mu2) / (2 * sigma1)
        return float(2 * scipy_stats.norm.cdf(-d))

    # General case: numerical integration of min(f, g)
    # Use a range that covers both distributions well (±6σ from each mean)
    max_sigma = max(sigma1, sigma2)
    lo = min(mu1, mu2) - 6 * max_sigma
    hi = max(mu1, mu2) + 6 * max_sigma
    n_points = 10000

    x = np.linspace(lo, hi, n_points)
    f1 = scipy_stats.norm.pdf(x, mu1, sigma1)
    f2 = scipy_stats.norm.pdf(x, mu2, sigma2)
    min_vals = np.minimum(f1, f2)
    dx = x[1] - x[0]
    overlap = float(np.sum(min_vals) * dx)

    return float(np.clip(overlap, 0.0, 1.0))


def evaluate_against_distribution(
    prediction, gt_distribution: "PopulationGroundTruth"
) -> Optional["DistributionMetrics"]:
    """
    Evaluate a predicted distribution against a ground truth population distribution.

    Computes distribution-to-distribution metrics for both height and weight.

    Args:
        prediction: PredictionResult with height_distribution and weight_distribution
        gt_distribution: PopulationGroundTruth with population-level stats

    Returns:
        DistributionMetrics if prediction is valid, None otherwise
    """
    from models.schemas import DistributionMetrics

    if not prediction.is_valid:
        return None

    pred_h_mu = prediction.height_distribution.mu
    pred_h_sigma = prediction.height_distribution.sigma
    pred_w_mu = prediction.weight_distribution.mu
    pred_w_sigma = prediction.weight_distribution.sigma

    gt_h_mu = gt_distribution.height_mean
    gt_h_sigma = gt_distribution.height_std
    gt_w_mu = gt_distribution.weight_mean
    gt_w_sigma = gt_distribution.weight_std

    return DistributionMetrics(
        kl_div_height=kl_divergence_normal(gt_h_mu, gt_h_sigma, pred_h_mu, pred_h_sigma),
        kl_div_weight=kl_divergence_normal(gt_w_mu, gt_w_sigma, pred_w_mu, pred_w_sigma),
        wasserstein_height=wasserstein2_normal(gt_h_mu, gt_h_sigma, pred_h_mu, pred_h_sigma),
        wasserstein_weight=wasserstein2_normal(gt_w_mu, gt_w_sigma, pred_w_mu, pred_w_sigma),
        mean_shift_height=compute_mean_shift(gt_h_mu, pred_h_mu),
        mean_shift_weight=compute_mean_shift(gt_w_mu, pred_w_mu),
        sigma_ratio_height=compute_sigma_ratio(gt_h_sigma, pred_h_sigma),
        sigma_ratio_weight=compute_sigma_ratio(gt_w_sigma, pred_w_sigma),
        overlap_height=overlap_coefficient_normal(gt_h_mu, gt_h_sigma, pred_h_mu, pred_h_sigma),
        overlap_weight=overlap_coefficient_normal(gt_w_mu, gt_w_sigma, pred_w_mu, pred_w_sigma),
    )


def aggregate_distribution_results(
    results: List["ExperimentResult"],
) -> Dict[str, Optional[float]]:
    """
    Aggregate distribution metrics across all subjects for one approach.

    Args:
        results: List of ExperimentResult objects for one approach

    Returns:
        Dict with aggregated metric names and values
    """
    valid = [r for r in results if r.distribution_metrics is not None]
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "n_with_dist_metrics": 0,
            "mean_kl_div_height": None,
            "mean_kl_div_weight": None,
            "mean_wasserstein_height": None,
            "mean_wasserstein_weight": None,
            "mean_mean_shift_height": None,
            "mean_mean_shift_weight": None,
            "mean_sigma_ratio_height": None,
            "mean_sigma_ratio_weight": None,
            "mean_overlap_height": None,
            "mean_overlap_weight": None,
        }

    return {
        "n_with_dist_metrics": n_valid,
        "mean_kl_div_height": float(np.mean([r.distribution_metrics.kl_div_height for r in valid])),
        "mean_kl_div_weight": float(np.mean([r.distribution_metrics.kl_div_weight for r in valid])),
        "mean_wasserstein_height": float(np.mean([r.distribution_metrics.wasserstein_height for r in valid])),
        "mean_wasserstein_weight": float(np.mean([r.distribution_metrics.wasserstein_weight for r in valid])),
        "mean_mean_shift_height": float(np.mean([r.distribution_metrics.mean_shift_height for r in valid])),
        "mean_mean_shift_weight": float(np.mean([r.distribution_metrics.mean_shift_weight for r in valid])),
        "mean_sigma_ratio_height": float(np.mean([r.distribution_metrics.sigma_ratio_height for r in valid])),
        "mean_sigma_ratio_weight": float(np.mean([r.distribution_metrics.sigma_ratio_weight for r in valid])),
        "mean_overlap_height": float(np.mean([r.distribution_metrics.overlap_height for r in valid])),
        "mean_overlap_weight": float(np.mean([r.distribution_metrics.overlap_weight for r in valid])),
        "std_kl_div_height": float(np.std([r.distribution_metrics.kl_div_height for r in valid], ddof=1)) if n_valid >= 2 else 0.0,
        "std_kl_div_weight": float(np.std([r.distribution_metrics.kl_div_weight for r in valid], ddof=1)) if n_valid >= 2 else 0.0,
        "std_wasserstein_height": float(np.std([r.distribution_metrics.wasserstein_height for r in valid], ddof=1)) if n_valid >= 2 else 0.0,
        "std_wasserstein_weight": float(np.std([r.distribution_metrics.wasserstein_weight for r in valid], ddof=1)) if n_valid >= 2 else 0.0,
    }


def format_distribution_results_table(
    approach_metrics: List[tuple],
) -> str:
    """
    Format distribution metrics as a markdown table.

    Args:
        approach_metrics: List of (approach_name, aggregated_dict) tuples

    Returns:
        Markdown formatted table string
    """
    table = "| Approach | N | KL Div (H) | KL Div (W) | W₂² (H) | W₂² (W) | Shift H (cm) | Shift W (kg) | σ Ratio H | σ Ratio W | Overlap H | Overlap W |\n"
    table += "|----------|---|------------|------------|----------|----------|--------------|--------------|-----------|-----------|-----------|----------|\n"

    for approach, metrics in approach_metrics:
        n = metrics.get("n_with_dist_metrics", 0)
        if n == 0:
            table += f"| {approach} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"
        else:
            table += (
                f"| {approach} "
                f"| {n} "
                f"| {metrics['mean_kl_div_height']:.3f} "
                f"| {metrics['mean_kl_div_weight']:.3f} "
                f"| {metrics['mean_wasserstein_height']:.1f} "
                f"| {metrics['mean_wasserstein_weight']:.1f} "
                f"| {metrics['mean_mean_shift_height']:.1f} "
                f"| {metrics['mean_mean_shift_weight']:.1f} "
                f"| {metrics['mean_sigma_ratio_height']:.2f} "
                f"| {metrics['mean_sigma_ratio_weight']:.2f} "
                f"| {metrics['mean_overlap_height']:.3f} "
                f"| {metrics['mean_overlap_weight']:.3f} |\n"
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
