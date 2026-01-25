"""
Unit tests for evaluation/metrics.py

Tests statistical metric calculations.
"""

import pytest
import math

import sys
sys.path.insert(0, "src")

from models.schemas import (
    DistributionParams,
    PredictionResult,
    GroundTruth,
    EvaluationMetrics,
    ExperimentResult,
)
from evaluation.metrics import (
    calculate_kl_divergence_normal,
    calculate_wasserstein_distance_normal,
    calculate_distribution_error,
    evaluate_prediction,
    aggregate_results,
    format_results_table,
)


class TestKLDivergence:
    """Tests for KL divergence calculation."""

    def test_identical_distributions(self):
        """KL divergence of identical distributions should be 0."""
        kl = calculate_kl_divergence_normal(
            pred_mu=175, pred_sigma=6, true_mu=175, true_sigma=6
        )
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_different_means(self):
        """KL divergence should increase with mean difference."""
        kl_small = calculate_kl_divergence_normal(
            pred_mu=175, pred_sigma=6, true_mu=176, true_sigma=6
        )
        kl_large = calculate_kl_divergence_normal(
            pred_mu=175, pred_sigma=6, true_mu=180, true_sigma=6
        )
        assert kl_large > kl_small > 0

    def test_different_sigmas(self):
        """KL divergence should be positive for different sigmas."""
        kl = calculate_kl_divergence_normal(
            pred_mu=175, pred_sigma=6, true_mu=175, true_sigma=10
        )
        assert kl > 0

    def test_kl_is_non_negative(self):
        """KL divergence should always be non-negative."""
        test_cases = [
            (175, 6, 180, 8),
            (160, 5, 170, 10),
            (180, 10, 175, 5),
        ]
        for pred_mu, pred_sigma, true_mu, true_sigma in test_cases:
            kl = calculate_kl_divergence_normal(pred_mu, pred_sigma, true_mu, true_sigma)
            assert kl >= 0


class TestWassersteinDistance:
    """Tests for Wasserstein distance calculation."""

    def test_identical_distributions(self):
        """Wasserstein distance of identical distributions should be 0."""
        dist = calculate_wasserstein_distance_normal(
            pred_mu=175, pred_sigma=6, true_mu=175, true_sigma=6
        )
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_mean_difference_only(self):
        """Wasserstein distance with same sigma equals mean difference."""
        dist = calculate_wasserstein_distance_normal(
            pred_mu=175, pred_sigma=6, true_mu=180, true_sigma=6
        )
        # W2 distance for same sigma is sqrt((mu1-mu2)^2 + 0) = |mu1-mu2|
        assert dist == pytest.approx(5.0, abs=0.01)

    def test_sigma_difference_only(self):
        """Wasserstein distance with same mean depends on sigma difference."""
        dist = calculate_wasserstein_distance_normal(
            pred_mu=175, pred_sigma=6, true_mu=175, true_sigma=10
        )
        # W2 distance for same mean is |sigma1 - sigma2|
        assert dist == pytest.approx(4.0, abs=0.01)

    def test_wasserstein_is_symmetric(self):
        """Wasserstein distance should be symmetric."""
        dist1 = calculate_wasserstein_distance_normal(
            pred_mu=175, pred_sigma=6, true_mu=180, true_sigma=8
        )
        dist2 = calculate_wasserstein_distance_normal(
            pred_mu=180, pred_sigma=8, true_mu=175, true_sigma=6
        )
        assert dist1 == pytest.approx(dist2, abs=1e-10)


class TestDistributionError:
    """Tests for comprehensive distribution error calculation."""

    def test_identical_distributions(self):
        """All errors should be 0 for identical distributions."""
        errors = calculate_distribution_error(
            pred_mu=175, pred_sigma=6, true_mu=175, true_sigma=6
        )

        assert errors["kl_divergence"] == pytest.approx(0.0, abs=1e-10)
        assert errors["wasserstein_distance"] == pytest.approx(0.0, abs=1e-10)
        assert errors["mae_mu"] == pytest.approx(0.0, abs=1e-10)
        assert errors["sigma_error"] == pytest.approx(0.0, abs=1e-10)

    def test_mae_calculation(self):
        """MAE should be absolute difference of means."""
        errors = calculate_distribution_error(
            pred_mu=180, pred_sigma=6, true_mu=175, true_sigma=6
        )
        assert errors["mae_mu"] == pytest.approx(5.0, abs=1e-10)

    def test_sigma_error_calculation(self):
        """Sigma error should be absolute difference of sigmas."""
        errors = calculate_distribution_error(
            pred_mu=175, pred_sigma=10, true_mu=175, true_sigma=6
        )
        assert errors["sigma_error"] == pytest.approx(4.0, abs=1e-10)


class TestEvaluatePrediction:
    """Tests for evaluate_prediction function."""

    def test_valid_prediction(self, sample_prediction_json, sample_ground_truth):
        """Test evaluation of valid prediction."""
        prediction = PredictionResult(**sample_prediction_json)
        ground_truth = GroundTruth(**sample_ground_truth)

        metrics = evaluate_prediction(prediction, ground_truth)

        assert metrics is not None
        assert metrics.is_valid is True
        assert metrics.kl_divergence_height is not None
        assert metrics.kl_divergence_weight is not None
        assert metrics.mae_height_mu is not None
        assert metrics.mae_weight_mu is not None

    def test_invalid_prediction(self, sample_ground_truth):
        """Test evaluation of invalid prediction."""
        prediction = PredictionResult(reasoning="Error", error="Failed")
        ground_truth = GroundTruth(**sample_ground_truth)

        metrics = evaluate_prediction(prediction, ground_truth)

        assert metrics is None

    def test_perfect_prediction(self, sample_ground_truth):
        """Test evaluation when prediction exactly matches ground truth."""
        # Create prediction that matches ground truth exactly
        prediction = PredictionResult(
            reasoning="Perfect match",
            height_distribution=DistributionParams(**sample_ground_truth["height"]),
            weight_distribution=DistributionParams(**sample_ground_truth["weight"]),
        )
        ground_truth = GroundTruth(**sample_ground_truth)

        metrics = evaluate_prediction(prediction, ground_truth)

        assert metrics is not None
        assert metrics.is_valid is True
        assert metrics.kl_divergence_height == pytest.approx(0.0, abs=1e-10)
        assert metrics.kl_divergence_weight == pytest.approx(0.0, abs=1e-10)
        assert metrics.mae_height_mu == pytest.approx(0.0, abs=1e-10)
        assert metrics.mae_weight_mu == pytest.approx(0.0, abs=1e-10)


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def _create_valid_metrics(self):
        """Helper to create valid EvaluationMetrics."""
        return EvaluationMetrics(
            kl_divergence_height=0.5,
            kl_divergence_weight=0.3,
            wasserstein_distance_height=2.0,
            wasserstein_distance_weight=3.0,
            mae_height_mu=5.0,
            mae_weight_mu=4.0,
            sigma_error_height=1.0,
            sigma_error_weight=1.5,
            is_valid=True
        )

    def test_aggregate_valid_results(self, sample_prediction_json, sample_ground_truth):
        """Test aggregation of valid results."""
        prediction = PredictionResult(**sample_prediction_json)
        ground_truth = GroundTruth(**sample_ground_truth)
        metrics = self._create_valid_metrics()

        results = [
            ExperimentResult(
                subject_id="001",
                approach="baseline",
                prediction=prediction,
                ground_truth=ground_truth,
                metrics=metrics
            ),
            ExperimentResult(
                subject_id="002",
                approach="baseline",
                prediction=prediction,
                ground_truth=ground_truth,
                metrics=metrics
            ),
        ]

        aggregated = aggregate_results(results)

        assert aggregated.approach == "baseline"
        assert aggregated.n_total == 2
        assert aggregated.n_valid == 2
        assert aggregated.n_invalid == 0
        assert aggregated.invalid_rate_percent == 0.0
        assert aggregated.mean_kl_divergence_height == pytest.approx(0.5, abs=0.01)

    def test_aggregate_mixed_results(self, sample_prediction_json, sample_ground_truth):
        """Test aggregation with both valid and invalid results."""
        prediction_valid = PredictionResult(**sample_prediction_json)
        prediction_invalid = PredictionResult(reasoning="Error", error="Failed")
        ground_truth = GroundTruth(**sample_ground_truth)

        valid_metrics = self._create_valid_metrics()

        results = [
            ExperimentResult(
                subject_id="001",
                approach="baseline",
                prediction=prediction_valid,
                ground_truth=ground_truth,
                metrics=valid_metrics
            ),
            ExperimentResult(
                subject_id="002",
                approach="baseline",
                prediction=prediction_invalid,
                ground_truth=ground_truth,
                metrics=None
            ),
        ]

        aggregated = aggregate_results(results)

        assert aggregated.n_total == 2
        assert aggregated.n_valid == 1
        assert aggregated.n_invalid == 1
        assert aggregated.invalid_rate_percent == 50.0

    def test_aggregate_empty_results_raises(self):
        """Test aggregation with no results raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty results"):
            aggregate_results([])


class TestFormatResultsTable:
    """Tests for format_results_table function."""

    def test_format_table_structure(self, sample_prediction_json, sample_ground_truth):
        """Test that table has correct markdown structure."""
        prediction = PredictionResult(**sample_prediction_json)
        ground_truth = GroundTruth(**sample_ground_truth)
        metrics = EvaluationMetrics(
            kl_divergence_height=0.5,
            kl_divergence_weight=0.3,
            wasserstein_distance_height=2.0,
            wasserstein_distance_weight=3.0,
            mae_height_mu=5.0,
            mae_weight_mu=4.0,
            sigma_error_height=1.0,
            sigma_error_weight=1.5,
            is_valid=True
        )

        results = [
            ExperimentResult(
                subject_id="001",
                approach="baseline",
                prediction=prediction,
                ground_truth=ground_truth,
                metrics=metrics
            ),
        ]

        aggregated = [aggregate_results(results)]
        table = format_results_table(aggregated)

        # Check markdown table structure
        assert "|" in table
        assert "Approach" in table or "approach" in table.lower()
        assert "baseline" in table
