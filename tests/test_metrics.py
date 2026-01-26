"""Unit tests for evaluation metrics.

Tests cover:
- KL divergence for normal distributions
- Wasserstein distance for normal distributions
- Calibration metrics (coverage, interval score)
- Statistical tests (Friedman, Wilcoxon)
"""

import numpy as np


class TestKLDivergence:
    """Test KL divergence calculation for normal distributions."""

    def test_identical_distributions_returns_zero(self) -> None:
        """KL(P || P) = 0."""
        from src.evaluation.metrics import calculate_kl_divergence_normal

        kl = calculate_kl_divergence_normal(175, 6, 175, 6)
        assert abs(kl) < 1e-10

    def test_different_means(self) -> None:
        """KL increases with mean difference."""
        from src.evaluation.metrics import calculate_kl_divergence_normal

        kl_small = calculate_kl_divergence_normal(175, 6, 170, 6)
        kl_large = calculate_kl_divergence_normal(175, 6, 160, 6)
        assert kl_large > kl_small > 0

    def test_different_sigmas(self) -> None:
        """KL is asymmetric in sigma."""
        from src.evaluation.metrics import calculate_kl_divergence_normal

        # Underconfident (pred sigma > true sigma)
        kl_under = calculate_kl_divergence_normal(175, 10, 175, 6)
        # Overconfident (pred sigma < true sigma)
        kl_over = calculate_kl_divergence_normal(175, 4, 175, 6)
        # Both should be positive
        assert kl_under > 0
        assert kl_over > 0


class TestWassersteinDistance:
    """Test Wasserstein-2 distance calculation for normal distributions."""

    def test_identical_distributions_returns_zero(self) -> None:
        """W2(P, P) = 0."""
        from src.evaluation.metrics import calculate_wasserstein_distance_normal

        w = calculate_wasserstein_distance_normal(175, 6, 175, 6)
        assert abs(w) < 1e-10

    def test_mean_difference_only(self) -> None:
        """W2 = |mu1 - mu2| when sigmas equal."""
        from src.evaluation.metrics import calculate_wasserstein_distance_normal

        w = calculate_wasserstein_distance_normal(180, 6, 170, 6)
        assert abs(w - 10) < 1e-10

    def test_sigma_difference_only(self) -> None:
        """W2 = |sigma1 - sigma2| when means equal."""
        from src.evaluation.metrics import calculate_wasserstein_distance_normal

        w = calculate_wasserstein_distance_normal(175, 10, 175, 6)
        # W2 = sqrt(0 + 16) = 4
        assert abs(w - 4) < 1e-10


class TestCalibrationMetrics:
    """Test calibration metrics (coverage and interval score)."""

    def test_check_coverage_inside_interval(self) -> None:
        """True value inside predicted interval returns True."""
        from src.evaluation.calibration import check_coverage

        # N(175, 6), true=175 (at mean)
        coverage = check_coverage(175, 6, 175)
        assert coverage[0.50] == True  # noqa: E712
        assert coverage[0.90] == True  # noqa: E712
        assert coverage[0.95] == True  # noqa: E712

    def test_check_coverage_outside_interval(self) -> None:
        """True value outside predicted interval returns False."""
        from src.evaluation.calibration import check_coverage

        # N(175, 6), true=200 (very far)
        coverage = check_coverage(175, 6, 200)
        assert coverage[0.50] == False  # noqa: E712
        assert coverage[0.90] == False  # noqa: E712
        assert coverage[0.95] == False  # noqa: E712

    def test_interval_score_perfect_prediction(self) -> None:
        """Interval score for true value at mean."""
        from src.evaluation.calibration import interval_score

        # If true value is within interval, score = interval width
        score = interval_score(175, 6, 175, alpha=0.1)
        # 90% interval width = 2 * 1.645 * 6 = 19.74
        assert score > 0
        assert score < 25  # Should be roughly the interval width

    def test_interval_score_penalizes_miss(self) -> None:
        """Interval score higher when true value outside."""
        from src.evaluation.calibration import interval_score

        score_hit = interval_score(175, 6, 175, alpha=0.1)
        score_miss = interval_score(175, 6, 200, alpha=0.1)
        assert score_miss > score_hit


class TestStatisticalTests:
    """Test statistical comparison functions."""

    def test_friedman_significant(self) -> None:
        """Friedman detects difference in shifted groups."""
        from src.evaluation.statistical_tests import friedman_test

        np.random.seed(42)
        baseline = np.random.normal(0, 1, 30).tolist()
        better = [x - 0.8 for x in baseline]  # Shifted down (better scores)
        results = {"baseline": baseline, "better": better, "same": baseline}
        result = friedman_test(results)
        assert result.p_value < 0.05

    def test_friedman_not_significant(self) -> None:
        """Friedman finds no difference in similar groups from same distribution."""
        from src.evaluation.statistical_tests import friedman_test

        # Use independent samples from the same distribution
        # These should not show significant differences
        np.random.seed(42)
        a = np.random.normal(0, 1, 30).tolist()
        b = np.random.normal(0, 1, 30).tolist()
        c = np.random.normal(0, 1, 30).tolist()
        results = {"a": a, "b": b, "c": c}
        result = friedman_test(results)
        assert result.p_value > 0.05

    def test_wilcoxon_effect_size(self) -> None:
        """Wilcoxon effect size in valid range."""
        from src.evaluation.statistical_tests import pairwise_wilcoxon

        np.random.seed(42)
        a = np.random.normal(0, 1, 30).tolist()
        b = [x + 0.5 for x in a]
        result = pairwise_wilcoxon(a, b)
        assert -1 <= result.effect_size <= 1
