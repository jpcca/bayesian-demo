"""
Unit tests for evaluation/metrics.py

Tests for metrics that compare predicted distributions against actual ground truth values.
"""

import pytest
import numpy as np

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
    calculate_nll_normal,
    calculate_z_score,
    is_in_95ci_normal,
    evaluate_prediction,
    aggregate_results,
    format_results_table,
)


class TestNLLCalculation:
    """Tests for negative log-likelihood calculation."""

    def test_nll_at_mean(self):
        """NLL should be lowest when true value equals predicted mean."""
        # When true value = mean, NLL = 0.5*log(2π) + log(σ)
        nll = calculate_nll_normal(pred_mu=175, pred_sigma=6, true_value=175)
        expected = 0.5 * np.log(2 * np.pi) + np.log(6)
        assert nll == pytest.approx(expected, abs=1e-10)

    def test_nll_increases_with_distance(self):
        """NLL should increase as true value moves away from mean."""
        nll_at_mean = calculate_nll_normal(pred_mu=175, pred_sigma=6, true_value=175)
        nll_1sigma = calculate_nll_normal(pred_mu=175, pred_sigma=6, true_value=181)
        nll_2sigma = calculate_nll_normal(pred_mu=175, pred_sigma=6, true_value=187)
        assert nll_at_mean < nll_1sigma < nll_2sigma

    def test_nll_symmetric(self):
        """NLL should be same for equal distances above and below mean."""
        nll_above = calculate_nll_normal(pred_mu=175, pred_sigma=6, true_value=181)
        nll_below = calculate_nll_normal(pred_mu=175, pred_sigma=6, true_value=169)
        assert nll_above == pytest.approx(nll_below, abs=1e-10)

    def test_nll_penalizes_overconfidence(self):
        """Smaller sigma should increase NLL when prediction is wrong."""
        # Same error, different sigmas
        nll_wide = calculate_nll_normal(pred_mu=175, pred_sigma=10, true_value=180)
        nll_narrow = calculate_nll_normal(pred_mu=175, pred_sigma=3, true_value=180)
        assert nll_narrow > nll_wide  # Narrow sigma penalized more for same error


class TestZScore:
    """Tests for z-score calculation."""

    def test_zscore_at_mean(self):
        """Z-score should be 0 when true value equals mean."""
        z = calculate_z_score(pred_mu=175, pred_sigma=6, true_value=175)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_zscore_one_sigma_above(self):
        """Z-score should be 1 when true value is one sigma above mean."""
        z = calculate_z_score(pred_mu=175, pred_sigma=6, true_value=181)
        assert z == pytest.approx(1.0, abs=1e-10)

    def test_zscore_one_sigma_below(self):
        """Z-score should be -1 when true value is one sigma below mean."""
        z = calculate_z_score(pred_mu=175, pred_sigma=6, true_value=169)
        assert z == pytest.approx(-1.0, abs=1e-10)

    def test_zscore_scales_with_sigma(self):
        """Z-score should scale inversely with sigma."""
        z_small_sigma = calculate_z_score(pred_mu=175, pred_sigma=3, true_value=178)
        z_large_sigma = calculate_z_score(pred_mu=175, pred_sigma=6, true_value=178)
        assert z_small_sigma == pytest.approx(1.0, abs=1e-10)
        assert z_large_sigma == pytest.approx(0.5, abs=1e-10)


class TestCoverageCheck:
    """Tests for 95% CI coverage check."""

    def test_at_mean_is_covered(self):
        """True value at mean should be within 95% CI."""
        assert is_in_95ci_normal(pred_mu=175, pred_sigma=6, true_value=175) is True

    def test_within_1_sigma_is_covered(self):
        """True value within 1 sigma should be within 95% CI."""
        assert is_in_95ci_normal(pred_mu=175, pred_sigma=6, true_value=181) is True

    def test_at_1_96_sigma_boundary(self):
        """True value exactly at 1.96 sigma should be within 95% CI."""
        boundary_value = 175 + 1.96 * 6
        assert is_in_95ci_normal(pred_mu=175, pred_sigma=6, true_value=boundary_value) is True

    def test_beyond_1_96_sigma_not_covered(self):
        """True value beyond 1.96 sigma should not be within 95% CI."""
        outside_value = 175 + 2.0 * 6  # 2 sigma away
        assert is_in_95ci_normal(pred_mu=175, pred_sigma=6, true_value=outside_value) is False

    def test_symmetric_coverage(self):
        """Coverage should be symmetric above and below mean."""
        # Both should be covered (within 1.96 sigma)
        assert is_in_95ci_normal(pred_mu=175, pred_sigma=6, true_value=186) is True
        assert is_in_95ci_normal(pred_mu=175, pred_sigma=6, true_value=164) is True


class TestEvaluatePrediction:
    """Tests for evaluate_prediction function."""

    def test_valid_prediction(self, sample_prediction_json, sample_ground_truth):
        """Test evaluation of valid prediction against actual ground truth."""
        prediction = PredictionResult(**sample_prediction_json)
        ground_truth = GroundTruth(**sample_ground_truth)

        metrics = evaluate_prediction(prediction, ground_truth)

        assert metrics is not None
        assert metrics.is_valid is True
        assert metrics.nll_height is not None
        assert metrics.nll_weight is not None
        assert metrics.abs_error_height is not None
        assert metrics.z_score_height is not None
        assert isinstance(metrics.in_95ci_height, bool)

    def test_invalid_prediction(self, sample_ground_truth):
        """Test evaluation of invalid prediction."""
        prediction = PredictionResult(reasoning="Error", error="Failed")
        ground_truth = GroundTruth(**sample_ground_truth)

        metrics = evaluate_prediction(prediction, ground_truth)

        assert metrics is None

    def test_perfect_prediction(self, sample_ground_truth):
        """Test evaluation when prediction mean exactly matches ground truth."""
        # Create prediction that matches ground truth exactly
        prediction = PredictionResult(
            reasoning="Perfect match",
            height_distribution=DistributionParams(
                distribution_type="normal", mu=178.0, sigma=5.0, unit="cm"
            ),
            weight_distribution=DistributionParams(
                distribution_type="normal", mu=72.0, sigma=7.0, unit="kg"
            ),
        )
        ground_truth = GroundTruth(**sample_ground_truth)

        metrics = evaluate_prediction(prediction, ground_truth)

        assert metrics is not None
        assert metrics.is_valid is True
        assert metrics.abs_error_height == pytest.approx(0.0, abs=1e-10)
        assert metrics.abs_error_weight == pytest.approx(0.0, abs=1e-10)
        assert metrics.z_score_height == pytest.approx(0.0, abs=1e-10)
        assert metrics.z_score_weight == pytest.approx(0.0, abs=1e-10)
        assert metrics.in_95ci_height is True
        assert metrics.in_95ci_weight is True


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def _create_valid_metrics(self):
        """Helper to create valid EvaluationMetrics."""
        return EvaluationMetrics(
            nll_height=2.5,
            nll_weight=2.8,
            abs_error_height=3.0,
            abs_error_weight=2.5,
            z_score_height=0.5,
            z_score_weight=-0.3,
            in_95ci_height=True,
            in_95ci_weight=True,
            is_valid=True,
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
                metrics=metrics,
            ),
            ExperimentResult(
                subject_id="002",
                approach="baseline",
                prediction=prediction,
                ground_truth=ground_truth,
                metrics=metrics,
            ),
        ]

        aggregated = aggregate_results(results)

        assert aggregated.approach == "baseline"
        assert aggregated.n_total == 2
        assert aggregated.n_valid == 2
        assert aggregated.n_invalid == 0
        assert aggregated.invalid_rate_percent == 0.0
        assert aggregated.mean_nll_height == pytest.approx(2.5, abs=0.01)
        assert aggregated.mean_abs_error_height == pytest.approx(3.0, abs=0.01)
        assert aggregated.coverage_95ci_height_percent == 100.0

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
                metrics=valid_metrics,
            ),
            ExperimentResult(
                subject_id="002",
                approach="baseline",
                prediction=prediction_invalid,
                ground_truth=ground_truth,
                metrics=None,
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

    def test_aggregate_coverage_calculation(self, sample_prediction_json, sample_ground_truth):
        """Test that coverage percentage is calculated correctly."""
        prediction = PredictionResult(**sample_prediction_json)
        ground_truth = GroundTruth(**sample_ground_truth)

        # Create metrics with different coverage
        metrics_covered = EvaluationMetrics(
            nll_height=2.5, nll_weight=2.8,
            abs_error_height=3.0, abs_error_weight=2.5,
            z_score_height=0.5, z_score_weight=-0.3,
            in_95ci_height=True, in_95ci_weight=True,
            is_valid=True,
        )
        metrics_not_covered = EvaluationMetrics(
            nll_height=5.0, nll_weight=5.5,
            abs_error_height=15.0, abs_error_weight=12.0,
            z_score_height=2.5, z_score_weight=2.0,
            in_95ci_height=False, in_95ci_weight=False,
            is_valid=True,
        )

        results = [
            ExperimentResult(
                subject_id="001", approach="baseline",
                prediction=prediction, ground_truth=ground_truth,
                metrics=metrics_covered,
            ),
            ExperimentResult(
                subject_id="002", approach="baseline",
                prediction=prediction, ground_truth=ground_truth,
                metrics=metrics_not_covered,
            ),
        ]

        aggregated = aggregate_results(results)

        # 1 out of 2 covered = 50%
        assert aggregated.coverage_95ci_height_percent == 50.0
        assert aggregated.coverage_95ci_weight_percent == 50.0


class TestFormatResultsTable:
    """Tests for format_results_table function."""

    def test_format_table_structure(self, sample_prediction_json, sample_ground_truth):
        """Test that table has correct markdown structure."""
        prediction = PredictionResult(**sample_prediction_json)
        ground_truth = GroundTruth(**sample_ground_truth)
        metrics = EvaluationMetrics(
            nll_height=2.5, nll_weight=2.8,
            abs_error_height=3.0, abs_error_weight=2.5,
            z_score_height=0.5, z_score_weight=-0.3,
            in_95ci_height=True, in_95ci_weight=True,
            is_valid=True,
        )

        results = [
            ExperimentResult(
                subject_id="001",
                approach="baseline",
                prediction=prediction,
                ground_truth=ground_truth,
                metrics=metrics,
            ),
        ]

        aggregated = [aggregate_results(results)]
        table = format_results_table(aggregated)

        # Check markdown table structure
        assert "|" in table
        assert "Approach" in table or "approach" in table.lower()
        assert "baseline" in table
        assert "NLL" in table
        assert "95% CI" in table
