"""
Evaluation metrics for comparing predicted and ground truth distributions.
Implements the two main metrics from the project requirements:
1. Distribution error (KL divergence, Wasserstein distance, MAE)
2. Invalid output rate
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


def calculate_kl_divergence_normal(
    pred_mu: float, pred_sigma: float, true_mu: float, true_sigma: float
) -> float:
    """
    Calculate KL divergence between two normal distributions.

    For two normal distributions N(μ₁, σ₁²) and N(μ₂, σ₂²):
    KL(P || Q) = log(σ₂/σ₁) + (σ₁² + (μ₁ - μ₂)²) / (2σ₂²) - 1/2

    Where P is predicted, Q is ground truth.

    Lower is better; 0 = perfect match.
    """
    return (
        np.log(true_sigma / pred_sigma)
        + (pred_sigma**2 + (pred_mu - true_mu) ** 2) / (2 * true_sigma**2)
        - 0.5
    )


def calculate_wasserstein_distance_normal(
    pred_mu: float, pred_sigma: float, true_mu: float, true_sigma: float
) -> float:
    """
    Calculate Wasserstein-2 distance between two normal distributions.

    For normal distributions, W₂² has closed form:
    W₂²(N(μ₁, σ₁²), N(μ₂, σ₂²)) = (μ₁ - μ₂)² + (σ₁ - σ₂)²

    Lower is better; 0 = perfect match.
    """
    return np.sqrt((pred_mu - true_mu) ** 2 + (pred_sigma - true_sigma) ** 2)


def calculate_distribution_error(
    pred_mu: float,
    pred_sigma: float,
    true_mu: float,
    true_sigma: float,
    distribution_type: str = "normal",
) -> dict:
    """
    Calculate multiple error metrics between predicted and ground truth distributions.

    Args:
        pred_mu: Predicted mean
        pred_sigma: Predicted standard deviation
        true_mu: Ground truth mean
        true_sigma: Ground truth standard deviation
        distribution_type: Type of distribution (currently only "normal" supported)

    Returns:
        Dictionary with error metrics
    """
    if distribution_type != "normal":
        raise NotImplementedError(
            f"Distribution type {distribution_type} not yet supported. Use 'normal'."
        )

    # KL divergence
    kl_div = calculate_kl_divergence_normal(pred_mu, pred_sigma, true_mu, true_sigma)

    # Wasserstein distance
    wasserstein = calculate_wasserstein_distance_normal(pred_mu, pred_sigma, true_mu, true_sigma)

    # Simple errors
    mae_mu = abs(pred_mu - true_mu)
    sigma_error = abs(pred_sigma - true_sigma)

    return {
        "kl_divergence": kl_div,
        "wasserstein_distance": wasserstein,
        "mae_mu": mae_mu,
        "sigma_error": sigma_error,
    }


def evaluate_prediction(prediction, ground_truth) -> Optional["EvaluationMetrics"]:
    """
    Evaluate a single prediction against ground truth.

    Args:
        prediction: PredictionResult object
        ground_truth: GroundTruth object

    Returns:
        EvaluationMetrics if prediction is valid, None otherwise
    """
    # Import here to avoid circular dependency
    from models.schemas import EvaluationMetrics

    if not prediction.is_valid:
        return None

    # Calculate height metrics
    height_metrics = calculate_distribution_error(
        pred_mu=prediction.height_distribution.mu,
        pred_sigma=prediction.height_distribution.sigma,
        true_mu=ground_truth.height.mu,
        true_sigma=ground_truth.height.sigma,
        distribution_type=prediction.height_distribution.distribution_type,
    )

    # Calculate weight metrics
    weight_metrics = calculate_distribution_error(
        pred_mu=prediction.weight_distribution.mu,
        pred_sigma=prediction.weight_distribution.sigma,
        true_mu=ground_truth.weight.mu,
        true_sigma=ground_truth.weight.sigma,
        distribution_type=prediction.weight_distribution.distribution_type,
    )

    return EvaluationMetrics(
        kl_divergence_height=height_metrics["kl_divergence"],
        kl_divergence_weight=weight_metrics["kl_divergence"],
        wasserstein_distance_height=height_metrics["wasserstein_distance"],
        wasserstein_distance_weight=weight_metrics["wasserstein_distance"],
        mae_height_mu=height_metrics["mae_mu"],
        mae_weight_mu=weight_metrics["mae_mu"],
        sigma_error_height=height_metrics["sigma_error"],
        sigma_error_weight=weight_metrics["sigma_error"],
        is_valid=True,
    )


def aggregate_results(results: List["ExperimentResult"]) -> "AggregatedMetrics":
    """
    Aggregate metrics across all subjects for one approach.

    Args:
        results: List of ExperimentResult objects for one approach

    Returns:
        AggregatedMetrics object
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
    kl_div_height = [r.metrics.kl_divergence_height for r in valid_results]
    kl_div_weight = [r.metrics.kl_divergence_weight for r in valid_results]
    wasserstein_height = [r.metrics.wasserstein_distance_height for r in valid_results]
    wasserstein_weight = [r.metrics.wasserstein_distance_weight for r in valid_results]
    mae_height = [r.metrics.mae_height_mu for r in valid_results]
    mae_weight = [r.metrics.mae_weight_mu for r in valid_results]
    sigma_err_height = [r.metrics.sigma_error_height for r in valid_results]
    sigma_err_weight = [r.metrics.sigma_error_weight for r in valid_results]

    return AggregatedMetrics(
        approach=approach,
        n_total=n_total,
        n_valid=n_valid,
        n_invalid=n_invalid,
        invalid_rate_percent=invalid_rate,
        # Means
        mean_kl_divergence_height=np.mean(kl_div_height),
        mean_kl_divergence_weight=np.mean(kl_div_weight),
        mean_wasserstein_distance_height=np.mean(wasserstein_height),
        mean_wasserstein_distance_weight=np.mean(wasserstein_weight),
        mean_mae_height_mu=np.mean(mae_height),
        mean_mae_weight_mu=np.mean(mae_weight),
        mean_sigma_error_height=np.mean(sigma_err_height),
        mean_sigma_error_weight=np.mean(sigma_err_weight),
        # Standard deviations (for error bars in plots)
        std_kl_divergence_height=np.std(kl_div_height),
        std_kl_divergence_weight=np.std(kl_div_weight),
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
    table = "| Approach | N Valid | Invalid Rate (%) | KL Div (Height) | KL Div (Weight) | Wasserstein (Height) | Wasserstein (Weight) | MAE Height (cm) | MAE Weight (kg) |\n"
    table += "|----------|---------|------------------|-----------------|-----------------|----------------------|----------------------|-----------------|------------------|\n"

    # Build rows
    for metrics in aggregated_metrics:
        if metrics.n_valid == 0:
            # No valid predictions
            table += f"| {metrics.approach} | 0 | {metrics.invalid_rate_percent:.1f} | N/A | N/A | N/A | N/A | N/A | N/A |\n"
        else:
            table += (
                f"| {metrics.approach} "
                f"| {metrics.n_valid}/{metrics.n_total} "
                f"| {metrics.invalid_rate_percent:.1f} "
                f"| {metrics.mean_kl_divergence_height:.3f} "
                f"| {metrics.mean_kl_divergence_weight:.3f} "
                f"| {metrics.mean_wasserstein_distance_height:.2f} "
                f"| {metrics.mean_wasserstein_distance_weight:.2f} "
                f"| {metrics.mean_mae_height_mu:.2f} "
                f"| {metrics.mean_mae_weight_mu:.2f} |\n"
            )

    return table


# Example usage
if __name__ == "__main__":
    # Test KL divergence calculation
    # Two identical distributions should have KL = 0
    kl = calculate_kl_divergence_normal(175, 6, 175, 6)
    print(f"KL divergence (identical): {kl:.6f}")  # Should be 0

    # Two different distributions
    kl = calculate_kl_divergence_normal(175, 6, 170, 8)
    print(f"KL divergence (different): {kl:.6f}")

    # Wasserstein distance
    w = calculate_wasserstein_distance_normal(175, 6, 175, 6)
    print(f"Wasserstein (identical): {w:.6f}")  # Should be 0

    w = calculate_wasserstein_distance_normal(175, 6, 170, 8)
    print(f"Wasserstein (different): {w:.6f}")
