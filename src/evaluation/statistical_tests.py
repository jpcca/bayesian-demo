"""Statistical tests for comparing prediction approaches.

Provides appropriate non-parametric statistical tests for comparing
k related samples (approaches) on calibration metrics.

Tests included:
- Friedman test: Overall comparison of k approaches
- Wilcoxon signed-rank test: Pairwise comparisons with Bonferroni correction
- McNemar's test: Comparing paired proportions (coverage rates)

Example:
    >>> from evaluation.statistical_tests import friedman_test, all_pairwise_comparisons
    >>> results = {'baseline': [0.5, 0.6], 'web_search': [0.7, 0.8], 'probabilistic': [0.9, 0.85]}
    >>> friedman = friedman_test(results)
    >>> pairwise = all_pairwise_comparisons(results)
"""

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Minimum sample size for reliable statistical tests
MIN_SAMPLE_SIZE = 5


@dataclass
class FriedmanResult:
    """Result of Friedman test for k related samples.

    Attributes:
        statistic: Friedman chi-square statistic
        p_value: p-value from chi-square distribution
        n_subjects: Number of subjects (questions/predictions)
        n_approaches: Number of approaches compared
        significant: Whether p < 0.05
    """

    statistic: float
    p_value: float
    n_subjects: int
    n_approaches: int
    significant: bool  # p < 0.05


@dataclass
class WilcoxonResult:
    """Result of Wilcoxon signed-rank test.

    Attributes:
        statistic: Wilcoxon W statistic
        p_value: Raw p-value
        p_value_corrected: Bonferroni corrected p-value
        n_subjects: Number of subjects compared
        significant: Whether p_corrected < 0.05
        effect_size: r = Z / sqrt(2N) effect size
    """

    statistic: float
    p_value: float
    p_value_corrected: float  # Bonferroni corrected
    n_subjects: int
    significant: bool  # p_corrected < 0.05
    effect_size: float  # r = Z / sqrt(2N)


@dataclass
class McNemarResult:
    """Result of McNemar's test for paired proportions.

    Attributes:
        statistic: McNemar chi-square statistic
        p_value: p-value from test
        n_subjects: Number of subjects compared
        coverage_a: Coverage rate for approach A
        coverage_b: Coverage rate for approach B
        significant: Whether p < 0.05
    """

    statistic: float
    p_value: float
    n_subjects: int
    coverage_a: float
    coverage_b: float
    significant: bool


def _validate_and_clean_data(data: List[float], name: str = "data") -> Tuple[np.ndarray, int]:
    """Validate and clean input data, filtering NaN values.

    Args:
        data: Input list of floats
        name: Name for error messages

    Returns:
        Tuple of (cleaned array, count of NaN values removed)

    Raises:
        ValueError: If data is empty or has too few samples after cleaning
    """
    arr = np.array(data, dtype=float)
    nan_mask = np.isnan(arr)
    nan_count = int(np.sum(nan_mask))

    if nan_count > 0:
        warnings.warn(
            f"Removed {nan_count} NaN values from {name}",
            UserWarning,
            stacklevel=3,
        )
        arr = arr[~nan_mask]

    if len(arr) == 0:
        raise ValueError(f"{name} is empty after removing NaN values")

    return arr, nan_count


def _check_aligned_lengths(results_by_approach: Dict[str, List[float]]) -> int:
    """Check that all approach lists have the same length.

    Args:
        results_by_approach: Dict mapping approach names to score lists

    Returns:
        The common length

    Raises:
        ValueError: If lengths differ or dict is empty
    """
    if not results_by_approach:
        raise ValueError("results_by_approach cannot be empty")

    lengths = {name: len(scores) for name, scores in results_by_approach.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) > 1:
        raise ValueError(f"All approach lists must have the same length. Got: {lengths}")

    return unique_lengths.pop()


def friedman_test(results_by_approach: Dict[str, List[float]]) -> FriedmanResult:
    """Friedman test for comparing k related samples.

    Tests whether there are differences among k approaches
    when each subject is measured under all conditions.

    Args:
        results_by_approach: {'baseline': [scores], 'web_search': [scores], 'probabilistic': [scores]}
            All lists must have same length (same subjects).

    Returns:
        FriedmanResult with test statistics

    Raises:
        ValueError: If input is invalid or has too few samples

    Example:
        >>> results = {
        ...     'baseline': [0.5, 0.6, 0.55, 0.7, 0.65],
        ...     'web_search': [0.7, 0.8, 0.75, 0.85, 0.8],
        ...     'probabilistic': [0.9, 0.85, 0.88, 0.92, 0.87]
        ... }
        >>> result = friedman_test(results)
        >>> print(f"Chi-square: {result.statistic:.2f}, p={result.p_value:.4f}")
    """
    n_subjects = _check_aligned_lengths(results_by_approach)
    n_approaches = len(results_by_approach)

    if n_approaches < 2:
        raise ValueError("Need at least 2 approaches for comparison")

    if n_subjects < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {MIN_SAMPLE_SIZE} subjects for reliable Friedman test, got {n_subjects}"
        )

    # Prepare data as list of arrays for scipy
    approach_names = list(results_by_approach.keys())
    groups = []

    for name in approach_names:
        arr, nan_count = _validate_and_clean_data(results_by_approach[name], f"approach '{name}'")
        groups.append(arr)

    # Check if all scores are identical (no variance)
    all_scores = np.concatenate(groups)
    if np.std(all_scores) < 1e-10:
        return FriedmanResult(
            statistic=0.0,
            p_value=1.0,
            n_subjects=n_subjects,
            n_approaches=n_approaches,
            significant=False,
        )

    # Run Friedman test
    statistic, p_value = stats.friedmanchisquare(*groups)

    return FriedmanResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_subjects=n_subjects,
        n_approaches=n_approaches,
        significant=p_value < 0.05,
    )


def pairwise_wilcoxon(
    scores_a: List[float],
    scores_b: List[float],
    n_comparisons: int = 3,  # For Bonferroni correction
) -> WilcoxonResult:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Tests whether
    the distribution of differences between paired samples is
    symmetric around zero.

    Args:
        scores_a: Scores from approach A
        scores_b: Scores from approach B
        n_comparisons: Number of comparisons for Bonferroni correction

    Returns:
        WilcoxonResult with corrected p-value and effect size

    Raises:
        ValueError: If inputs have different lengths or too few samples

    Example:
        >>> baseline = [0.5, 0.6, 0.55, 0.7, 0.65]
        >>> improved = [0.7, 0.8, 0.75, 0.85, 0.8]
        >>> result = pairwise_wilcoxon(baseline, improved)
        >>> print(f"W={result.statistic:.1f}, p_corrected={result.p_value_corrected:.4f}")
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have same length. Got {len(scores_a)} and {len(scores_b)}"
        )

    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)

    # Handle NaN values - remove pairs where either has NaN
    valid_mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    nan_count = int(np.sum(~valid_mask))

    if nan_count > 0:
        warnings.warn(f"Removed {nan_count} pairs with NaN values", UserWarning, stacklevel=2)
        arr_a = arr_a[valid_mask]
        arr_b = arr_b[valid_mask]

    n_subjects = len(arr_a)

    if n_subjects < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {MIN_SAMPLE_SIZE} valid pairs for Wilcoxon test, got {n_subjects}"
        )

    # Check if differences are all zero (identical scores)
    differences = arr_a - arr_b
    if np.all(np.abs(differences) < 1e-10):
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            n_subjects=n_subjects,
            significant=False,
            effect_size=0.0,
        )

    # Run Wilcoxon signed-rank test
    # zero_method='wilcox' excludes zero differences
    try:
        result = stats.wilcoxon(arr_a, arr_b, alternative="two-sided", zero_method="wilcox")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError:
        # All differences are zero after rounding
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            n_subjects=n_subjects,
            significant=False,
            effect_size=0.0,
        )

    # Bonferroni correction
    p_value_corrected = min(p_value * n_comparisons, 1.0)

    # Calculate effect size r = Z / sqrt(2N)
    # For Wilcoxon, we can use normal approximation to get Z
    # Z = (W - mean) / std where mean = n(n+1)/4 and std = sqrt(n(n+1)(2n+1)/24)
    n = n_subjects
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if std_w > 0:
        z_score = (statistic - mean_w) / std_w
        effect_size = abs(z_score) / np.sqrt(2 * n)
    else:
        effect_size = 0.0

    return WilcoxonResult(
        statistic=statistic,
        p_value=p_value,
        p_value_corrected=p_value_corrected,
        n_subjects=n_subjects,
        significant=p_value_corrected < 0.05,
        effect_size=float(effect_size),
    )


def mcnemar_test(coverage_a: List[bool], coverage_b: List[bool]) -> McNemarResult:
    """McNemar's test for comparing paired proportions.

    Tests whether two approaches have different coverage rates.
    Uses the exact binomial test for small samples.

    Args:
        coverage_a: List of bools (True = value in interval) for approach A
        coverage_b: List of bools for approach B

    Returns:
        McNemarResult with test statistics

    Raises:
        ValueError: If inputs have different lengths or too few samples

    Example:
        >>> coverage_baseline = [True, False, True, True, False, True, True, False, True, True]
        >>> coverage_improved = [True, True, True, True, True, True, True, False, True, True]
        >>> result = mcnemar_test(coverage_baseline, coverage_improved)
        >>> print(f"Coverage: {result.coverage_a:.1%} vs {result.coverage_b:.1%}")
    """
    if len(coverage_a) != len(coverage_b):
        raise ValueError(
            f"Coverage lists must have same length. Got {len(coverage_a)} and {len(coverage_b)}"
        )

    n_subjects = len(coverage_a)

    if n_subjects < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {MIN_SAMPLE_SIZE} samples for McNemar test, got {n_subjects}"
        )

    arr_a = np.array(coverage_a, dtype=bool)
    arr_b = np.array(coverage_b, dtype=bool)

    # Calculate coverage rates
    coverage_rate_a = float(np.mean(arr_a))
    coverage_rate_b = float(np.mean(arr_b))

    # Build 2x2 contingency table
    # Rows: A (True/False), Cols: B (True/False)
    # b = A covered, B not covered (discordant)
    # c = A not covered, B covered (discordant)
    b = int(np.sum(arr_a & ~arr_b))  # A yes, B no
    c = int(np.sum(~arr_a & arr_b))  # A no, B yes

    # McNemar test focuses on discordant pairs
    n_discordant = b + c

    if n_discordant == 0:
        # No discordant pairs - coverage is identical
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            n_subjects=n_subjects,
            coverage_a=coverage_rate_a,
            coverage_b=coverage_rate_b,
            significant=False,
        )

    # Use exact binomial test for small samples, chi-square for large
    if n_discordant < 25:
        # Exact binomial test: under null, b ~ Binomial(n_discordant, 0.5)
        # Two-sided p-value
        result = stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
        p_value = float(result.pvalue)
        statistic = float(b)  # Use count as statistic for exact test
    else:
        # Chi-square approximation with continuity correction
        statistic = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        n_subjects=n_subjects,
        coverage_a=coverage_rate_a,
        coverage_b=coverage_rate_b,
        significant=p_value < 0.05,
    )


def all_pairwise_comparisons(
    results_by_approach: Dict[str, List[float]],
) -> Dict[Tuple[str, str], WilcoxonResult]:
    """Run all pairwise Wilcoxon tests with Bonferroni correction.

    For 3 approaches: 3 comparisons (baseline vs web_search,
    baseline vs probabilistic, web_search vs probabilistic)

    Args:
        results_by_approach: Dict mapping approach names to score lists

    Returns:
        Dict mapping pairs to WilcoxonResult

    Example:
        >>> results = {
        ...     'baseline': [0.5, 0.6, 0.55, 0.7, 0.65],
        ...     'web_search': [0.7, 0.8, 0.75, 0.85, 0.8],
        ...     'probabilistic': [0.9, 0.85, 0.88, 0.92, 0.87]
        ... }
        >>> pairwise = all_pairwise_comparisons(results)
        >>> for (a, b), result in pairwise.items():
        ...     print(f"{a} vs {b}: p={result.p_value_corrected:.4f}")
    """
    _check_aligned_lengths(results_by_approach)

    approach_names = sorted(results_by_approach.keys())
    pairs = list(combinations(approach_names, 2))
    n_comparisons = len(pairs)

    if n_comparisons == 0:
        return {}

    results: Dict[Tuple[str, str], WilcoxonResult] = {}

    for name_a, name_b in pairs:
        result = pairwise_wilcoxon(
            results_by_approach[name_a],
            results_by_approach[name_b],
            n_comparisons=n_comparisons,
        )
        results[(name_a, name_b)] = result

    return results


def format_statistical_report(
    friedman: FriedmanResult,
    pairwise: Dict[Tuple[str, str], WilcoxonResult],
) -> str:
    """Format results as a markdown table for the research report.

    Args:
        friedman: Result from friedman_test
        pairwise: Results from all_pairwise_comparisons

    Returns:
        Formatted markdown string with tables

    Example:
        >>> # After running tests
        >>> report = format_statistical_report(friedman_result, pairwise_results)
        >>> print(report)
    """
    lines = [
        "## Statistical Analysis",
        "",
        "### Friedman Test (Overall Comparison)",
        "",
        f"- **Chi-square statistic**: {friedman.statistic:.3f}",
        f"- **p-value**: {friedman.p_value:.4f}",
        f"- **Subjects (N)**: {friedman.n_subjects}",
        f"- **Approaches (k)**: {friedman.n_approaches}",
        f"- **Significant (p < 0.05)**: {'Yes' if friedman.significant else 'No'}",
        "",
    ]

    if friedman.significant:
        lines.append("> The Friedman test indicates significant differences among approaches.")
    else:
        lines.append("> The Friedman test did not find significant differences among approaches.")

    lines.extend(
        [
            "",
            "### Pairwise Comparisons (Wilcoxon Signed-Rank with Bonferroni Correction)",
            "",
            "| Comparison | W | p-value | p-corrected | Effect Size (r) | Significant |",
            "|------------|---|---------|-------------|-----------------|-------------|",
        ]
    )

    for (name_a, name_b), result in sorted(pairwise.items()):
        sig_marker = "*" if result.significant else ""
        effect_interp = _interpret_effect_size(result.effect_size)
        lines.append(
            f"| {name_a} vs {name_b} | {result.statistic:.1f} | "
            f"{result.p_value:.4f} | {result.p_value_corrected:.4f}{sig_marker} | "
            f"{result.effect_size:.3f} ({effect_interp}) | "
            f"{'Yes' if result.significant else 'No'} |"
        )

    lines.extend(
        [
            "",
            "*Note: Effect size interpretation: small (r < 0.3), medium (0.3-0.5), large (r > 0.5)*",
            "",
            f"*Bonferroni correction applied for {len(pairwise)} comparisons*",
        ]
    )

    return "\n".join(lines)


def _interpret_effect_size(r: float) -> str:
    """Interpret effect size according to Cohen's conventions.

    Args:
        r: Effect size value

    Returns:
        Interpretation string (small/medium/large)
    """
    if r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"
