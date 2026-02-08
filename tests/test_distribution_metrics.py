"""
Tests for distribution-to-distribution metrics.

Tests the new functions in evaluation/metrics.py that compare
two Normal distributions (predicted vs ground truth population).
"""

import math
import pytest
import numpy as np

from evaluation.metrics import (
    kl_divergence_normal,
    wasserstein2_normal,
    compute_mean_shift,
    compute_sigma_ratio,
    overlap_coefficient_normal,
    evaluate_against_distribution,
)
from models.schemas import (
    DistributionMetrics,
    DistributionParams,
    PopulationGroundTruth,
    PredictionResult,
)


# ============================================================================
# KL Divergence Tests
# ============================================================================

class TestKLDivergence:
    def test_identical_distributions(self):
        """KL divergence of identical distributions should be 0."""
        kl = kl_divergence_normal(170, 8, 170, 8)
        assert abs(kl) < 1e-10

    def test_shifted_mean_positive(self):
        """KL divergence should be positive when means differ."""
        kl = kl_divergence_normal(170, 8, 175, 8)
        assert kl > 0

    def test_different_sigma_positive(self):
        """KL divergence should be positive when sigmas differ."""
        kl = kl_divergence_normal(170, 8, 170, 12)
        assert kl > 0

    def test_asymmetric(self):
        """KL(p||q) != KL(q||p) in general."""
        kl_forward = kl_divergence_normal(170, 8, 175, 10)
        kl_reverse = kl_divergence_normal(175, 10, 170, 8)
        assert kl_forward != pytest.approx(kl_reverse, abs=1e-6)

    def test_larger_shift_larger_kl(self):
        """Larger mean shift should produce larger KL divergence."""
        kl_small = kl_divergence_normal(170, 8, 172, 8)
        kl_large = kl_divergence_normal(170, 8, 180, 8)
        assert kl_large > kl_small

    def test_known_value(self):
        """Test against manually computed KL divergence.

        KL(N(0,1) || N(0,2)) = log(2/1) + (1 + 0)/(2*4) - 0.5
                               = log(2) + 0.125 - 0.5
                               ≈ 0.3182
        """
        kl = kl_divergence_normal(0, 1, 0, 2)
        expected = math.log(2) + 1 / 8 - 0.5
        assert kl == pytest.approx(expected, abs=1e-10)


# ============================================================================
# Wasserstein Distance Tests
# ============================================================================

class TestWasserstein:
    def test_identical_distributions(self):
        """Wasserstein distance of identical distributions should be 0."""
        w = wasserstein2_normal(170, 8, 170, 8)
        assert w == 0.0

    def test_shifted_mean(self):
        """W₂² with only mean shift = (Δμ)²."""
        w = wasserstein2_normal(170, 8, 175, 8)
        assert w == pytest.approx(25.0)  # (175-170)² + (8-8)² = 25

    def test_different_sigma(self):
        """W₂² with only sigma difference = (Δσ)²."""
        w = wasserstein2_normal(170, 8, 170, 12)
        assert w == pytest.approx(16.0)  # (170-170)² + (12-8)² = 16

    def test_combined(self):
        """W₂² = (Δμ)² + (Δσ)²."""
        w = wasserstein2_normal(170, 8, 175, 12)
        assert w == pytest.approx(41.0)  # 25 + 16


# ============================================================================
# Mean Shift Tests
# ============================================================================

class TestMeanShift:
    def test_identical(self):
        assert compute_mean_shift(170, 170) == 0.0

    def test_positive_shift(self):
        assert compute_mean_shift(170, 175) == 5.0

    def test_negative_shift(self):
        assert compute_mean_shift(175, 170) == 5.0

    def test_always_positive(self):
        assert compute_mean_shift(170, 160) > 0


# ============================================================================
# Sigma Ratio Tests
# ============================================================================

class TestSigmaRatio:
    def test_identical(self):
        assert compute_sigma_ratio(8, 8) == pytest.approx(1.0)

    def test_wider_prediction(self):
        """σ_pred > σ_gt → ratio > 1 (underconfident)."""
        assert compute_sigma_ratio(8, 16) == pytest.approx(2.0)

    def test_narrower_prediction(self):
        """σ_pred < σ_gt → ratio < 1 (overconfident)."""
        assert compute_sigma_ratio(8, 4) == pytest.approx(0.5)


# ============================================================================
# Overlap Coefficient Tests
# ============================================================================

class TestOverlapCoefficient:
    def test_identical_distributions(self):
        """Identical distributions should have overlap ≈ 1.0."""
        ovl = overlap_coefficient_normal(170, 8, 170, 8)
        assert ovl == pytest.approx(1.0, abs=1e-6)

    def test_far_apart_low_overlap(self):
        """Very far apart distributions should have overlap near 0."""
        ovl = overlap_coefficient_normal(100, 2, 200, 2)
        assert ovl < 0.01

    def test_range_0_to_1(self):
        """Overlap should always be in [0, 1]."""
        ovl = overlap_coefficient_normal(170, 8, 180, 10)
        assert 0.0 <= ovl <= 1.0

    def test_more_overlap_when_closer(self):
        """Closer means should produce higher overlap."""
        ovl_close = overlap_coefficient_normal(170, 8, 172, 8)
        ovl_far = overlap_coefficient_normal(170, 8, 185, 8)
        assert ovl_close > ovl_far

    def test_equal_sigma_case(self):
        """Test the equal-sigma special case."""
        ovl = overlap_coefficient_normal(170, 8, 175, 8)
        assert 0.0 < ovl < 1.0


# ============================================================================
# evaluate_against_distribution Tests
# ============================================================================

class TestEvaluateAgainstDistribution:
    def test_valid_prediction(self):
        """Should return DistributionMetrics for a valid prediction."""
        prediction = PredictionResult(
            reasoning="test",
            height_distribution=DistributionParams(
                distribution_type="normal", mu=170, sigma=8, unit="cm"
            ),
            weight_distribution=DistributionParams(
                distribution_type="normal", mu=80, sigma=15, unit="kg"
            ),
        )
        gt = PopulationGroundTruth(
            distribution_key="RIAGENDR=Female",
            height_mean=165,
            height_std=7,
            weight_mean=75,
            weight_std=18,
            n=5000,
            n_variables_matched=1,
        )

        metrics = evaluate_against_distribution(prediction, gt)

        assert metrics is not None
        assert isinstance(metrics, DistributionMetrics)
        assert metrics.kl_div_height > 0
        assert metrics.kl_div_weight > 0
        assert metrics.wasserstein_height > 0
        assert metrics.mean_shift_height == pytest.approx(5.0)
        assert metrics.mean_shift_weight == pytest.approx(5.0)
        assert metrics.sigma_ratio_height > 0
        assert 0 < metrics.overlap_height <= 1
        assert 0 < metrics.overlap_weight <= 1

    def test_invalid_prediction_returns_none(self):
        """Should return None for invalid prediction."""
        prediction = PredictionResult(
            reasoning="failed",
            error="Something went wrong",
        )
        gt = PopulationGroundTruth(
            distribution_key="Overall",
            height_mean=167,
            height_std=10,
            weight_mean=82,
            weight_std=22,
            n=75000,
            n_variables_matched=0,
        )

        metrics = evaluate_against_distribution(prediction, gt)
        assert metrics is None

    def test_perfect_prediction(self):
        """Perfect prediction should have KL≈0, W₂²≈0, ratio≈1, overlap≈1."""
        prediction = PredictionResult(
            reasoning="perfect",
            height_distribution=DistributionParams(
                distribution_type="normal", mu=170, sigma=8, unit="cm"
            ),
            weight_distribution=DistributionParams(
                distribution_type="normal", mu=80, sigma=15, unit="kg"
            ),
        )
        gt = PopulationGroundTruth(
            distribution_key="test",
            height_mean=170,
            height_std=8,
            weight_mean=80,
            weight_std=15,
            n=1000,
            n_variables_matched=3,
        )

        metrics = evaluate_against_distribution(prediction, gt)

        assert metrics.kl_div_height == pytest.approx(0, abs=1e-8)
        assert metrics.kl_div_weight == pytest.approx(0, abs=1e-8)
        assert metrics.wasserstein_height == pytest.approx(0, abs=1e-8)
        assert metrics.wasserstein_weight == pytest.approx(0, abs=1e-8)
        assert metrics.mean_shift_height == pytest.approx(0, abs=1e-8)
        assert metrics.sigma_ratio_height == pytest.approx(1.0)
        assert metrics.sigma_ratio_weight == pytest.approx(1.0)
        assert metrics.overlap_height == pytest.approx(1.0, abs=1e-6)
        assert metrics.overlap_weight == pytest.approx(1.0, abs=1e-6)
