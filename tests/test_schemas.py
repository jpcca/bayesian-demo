"""
Unit tests for models/schemas.py

Tests Pydantic model validation and serialization.
"""

import pytest
from pydantic import ValidationError

import sys

sys.path.insert(0, "src")

from models.schemas import (
    DistributionParams,
    PredictionResult,
    GroundTruth,
    EvaluationMetrics,
    ExperimentResult,
    sanitize_nulls,
)


class TestDistributionParams:
    """Tests for DistributionParams model."""

    def test_valid_normal_distribution(self):
        """Test creating a valid normal distribution."""
        dist = DistributionParams(distribution_type="normal", mu=175.0, sigma=6.0, unit="cm")
        assert dist.distribution_type == "normal"
        assert dist.mu == 175.0
        assert dist.sigma == 6.0
        assert dist.unit == "cm"

    def test_valid_distribution_types(self):
        """Test different valid distribution types."""
        for dist_type in ["normal", "lognormal", "truncated_normal"]:
            dist = DistributionParams(distribution_type=dist_type, mu=175.0, sigma=6.0, unit="cm")
            assert dist.distribution_type == dist_type


class TestPredictionResult:
    """Tests for PredictionResult model."""

    def test_valid_prediction(self, sample_prediction_json):
        """Test creating a valid prediction result."""
        result = PredictionResult(**sample_prediction_json)
        assert result.reasoning == sample_prediction_json["reasoning"]
        assert result.height_distribution is not None
        assert result.weight_distribution is not None
        assert result.is_valid is True

    def test_prediction_with_error(self):
        """Test prediction with error field."""
        result = PredictionResult(reasoning="Failed to process", error="API timeout")
        assert result.error == "API timeout"
        assert result.is_valid is False

    def test_prediction_missing_distributions(self):
        """Test prediction without distributions is invalid."""
        result = PredictionResult(reasoning="Some reasoning")
        assert result.height_distribution is None
        assert result.weight_distribution is None
        assert result.is_valid is False

    def test_prediction_partial_distributions(self):
        """Test prediction with only one distribution is invalid."""
        result = PredictionResult(
            reasoning="Partial result",
            height_distribution=DistributionParams(
                distribution_type="normal", mu=175.0, sigma=6.0, unit="cm"
            ),
        )
        assert result.is_valid is False


class TestGroundTruth:
    """Tests for GroundTruth model."""

    def test_valid_ground_truth(self, sample_ground_truth):
        """Test creating valid ground truth with actual measurements."""
        gt = GroundTruth(**sample_ground_truth)
        assert gt.subject_id == "001"
        assert gt.height_cm == 178.0
        assert gt.weight_kg == 72.0

    def test_ground_truth_requires_all_fields(self):
        """Test that ground truth requires all fields."""
        with pytest.raises(ValidationError):
            GroundTruth(subject_id="001")

    def test_ground_truth_positive_values(self):
        """Test that height and weight must be positive."""
        with pytest.raises(ValidationError):
            GroundTruth(
                subject_id="001",
                text_description="Test",
                height_cm=-175.0,
                weight_kg=70.0,
            )


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics model."""

    def test_valid_metrics(self):
        """Test creating valid evaluation metrics."""
        metrics = EvaluationMetrics(
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
        assert metrics.nll_height == 2.5
        assert metrics.abs_error_height == 3.0
        assert metrics.z_score_height == 0.5
        assert metrics.in_95ci_height is True
        assert metrics.is_valid is True

    def test_metrics_requires_all_fields(self):
        """Test that EvaluationMetrics requires all metric fields."""
        # All fields are required in EvaluationMetrics
        with pytest.raises(ValidationError):
            EvaluationMetrics(is_valid=False)  # Missing required fields


class TestExperimentResult:
    """Tests for ExperimentResult model."""

    def test_successful_experiment(self, sample_prediction_json, sample_ground_truth):
        """Test successful experiment result."""
        prediction = PredictionResult(**sample_prediction_json)
        ground_truth = GroundTruth(**sample_ground_truth)
        metrics = EvaluationMetrics(
            nll_height=2.5,
            nll_weight=2.8,
            abs_error_height=0.0,
            abs_error_weight=0.0,
            z_score_height=0.0,
            z_score_weight=0.0,
            in_95ci_height=True,
            in_95ci_weight=True,
            is_valid=True,
        )

        result = ExperimentResult(
            subject_id="001",
            approach="baseline",
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics,
        )
        assert result.is_success is True

    def test_failed_experiment(self, sample_ground_truth):
        """Test failed experiment result."""
        prediction = PredictionResult(reasoning="Error", error="Failed")
        ground_truth = GroundTruth(**sample_ground_truth)

        result = ExperimentResult(
            subject_id="001",
            approach="baseline",
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=None,
        )
        assert result.is_success is False


class TestSanitizeNulls:
    """Tests for sanitize_nulls utility function."""

    def test_string_null_to_none(self):
        """Test that string 'null' is converted to None."""
        data = {"key": "null", "other": "value"}
        result = sanitize_nulls(data)
        assert result["key"] is None
        assert result["other"] == "value"

    def test_nested_null_conversion(self):
        """Test null conversion in nested structures."""
        data = {"outer": {"inner": "null", "valid": 123}, "list": ["null", "value", "null"]}
        result = sanitize_nulls(data)
        assert result["outer"]["inner"] is None
        assert result["outer"]["valid"] == 123
        assert result["list"] == [None, "value", None]

    def test_preserves_actual_none(self):
        """Test that actual None values are preserved."""
        data = {"key": None}
        result = sanitize_nulls(data)
        assert result["key"] is None

    def test_preserves_non_null_strings(self):
        """Test that other strings are not affected."""
        data = {"key": "not null", "number": 42}
        result = sanitize_nulls(data)
        assert result["key"] == "not null"
        assert result["number"] == 42
