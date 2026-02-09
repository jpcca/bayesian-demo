"""
Tests for subject-to-distribution matching.

Tests the matching logic that connects test subjects to their
best-fitting ground truth population distribution.
"""

import pytest

from evaluation.matching import (
    build_distribution_key,
    match_subject_to_distribution,
    VARIABLE_PRIORITY,
)
from models.schemas import PopulationGroundTruth, SubjectDemographics


# ============================================================================
# build_distribution_key Tests
# ============================================================================

class TestBuildDistributionKey:
    def test_single_variable(self):
        key = build_distribution_key({"RIAGENDR": "Male"}, ["RIAGENDR"])
        assert key == "RIAGENDR=Male"

    def test_two_variables(self):
        demographics = {"RIAGENDR": "Male", "RIDAGEYR": "18-37"}
        key = build_distribution_key(demographics, ["RIAGENDR", "RIDAGEYR"])
        assert key == "RIAGENDR=Male__RIDAGEYR=18-37"

    def test_variable_order_matters(self):
        demographics = {"RIAGENDR": "Male", "RIDAGEYR": "18-37"}
        key1 = build_distribution_key(demographics, ["RIAGENDR", "RIDAGEYR"])
        key2 = build_distribution_key(demographics, ["RIDAGEYR", "RIAGENDR"])
        assert key1 != key2

    def test_missing_variable_raises(self):
        with pytest.raises(ValueError, match="not found"):
            build_distribution_key({"RIAGENDR": "Male"}, ["RIDAGEYR"])

    def test_none_value_raises(self):
        with pytest.raises(ValueError, match="not found"):
            build_distribution_key({"RIAGENDR": None}, ["RIAGENDR"])


# ============================================================================
# match_subject_to_distribution Tests
# ============================================================================

class TestMatchSubjectToDistribution:
    @pytest.fixture
    def sample_distributions(self):
        """Sample distribution data mimicking nhanes_ground_truth.json."""
        return {
            "Overall": {
                "height_mean": 167.0,
                "height_std": 10.0,
                "weight_mean": 82.0,
                "weight_std": 22.0,
                "n": 75000,
            },
            "RIAGENDR=Male": {
                "height_mean": 175.0,
                "height_std": 7.5,
                "weight_mean": 89.0,
                "weight_std": 21.0,
                "n": 33000,
            },
            "RIAGENDR=Female": {
                "height_mean": 161.0,
                "height_std": 7.0,
                "weight_mean": 77.0,
                "weight_std": 21.0,
                "n": 42000,
            },
            "RIAGENDR=Male__RIDAGEYR=18-37": {
                "height_mean": 177.0,
                "height_std": 7.0,
                "weight_mean": 85.0,
                "weight_std": 20.0,
                "n": 5000,
            },
            "RIAGENDR=Male__RIDAGEYR=18-37__RIDRETH1=White": {
                "height_mean": 179.0,
                "height_std": 6.5,
                "weight_mean": 87.0,
                "weight_std": 19.0,
                "n": 2000,
            },
        }

    def test_exact_match_3_vars(self, sample_distributions):
        """Should match on all 3 available variables."""
        demographics = SubjectDemographics(
            RIAGENDR="Male", RIDAGEYR="18-37", RIDRETH1="White"
        )
        result = match_subject_to_distribution(demographics, sample_distributions)

        assert isinstance(result, PopulationGroundTruth)
        assert result.distribution_key == "RIAGENDR=Male__RIDAGEYR=18-37__RIDRETH1=White"
        assert result.n_variables_matched == 3
        assert result.height_mean == 179.0
        assert result.n == 2000

    def test_fallback_to_fewer_vars(self, sample_distributions):
        """When full match not found, should fall back to fewer variables."""
        demographics = SubjectDemographics(
            RIAGENDR="Male", RIDAGEYR="18-37", RIDRETH1="Black"
        )
        result = match_subject_to_distribution(demographics, sample_distributions)

        # "Black" doesn't exist in 3-var combo, should fall back to 2-var
        assert result.distribution_key == "RIAGENDR=Male__RIDAGEYR=18-37"
        assert result.n_variables_matched == 2

    def test_fallback_to_single_var(self, sample_distributions):
        """Should fall back to single variable when needed."""
        demographics = SubjectDemographics(
            RIAGENDR="Female", RIDAGEYR="54-63"
        )
        result = match_subject_to_distribution(demographics, sample_distributions)

        # No 2-var combo for Female + 54-63, falls back to RIAGENDR=Female
        assert result.distribution_key == "RIAGENDR=Female"
        assert result.n_variables_matched == 1

    def test_fallback_to_overall(self, sample_distributions):
        """Should use Overall when no variable matches."""
        demographics = SubjectDemographics()  # All None
        result = match_subject_to_distribution(demographics, sample_distributions)

        assert result.distribution_key == "Overall"
        assert result.n_variables_matched == 0

    def test_priority_order_respected(self, sample_distributions):
        """Variables should be dropped from least to most informative."""
        # Gender is first priority, so even with lots of other vars,
        # a gender-only match should be found if nothing more specific exists
        demographics = SubjectDemographics(
            RIAGENDR="Male",
            RIDAGEYR="54-63",  # Not in sample distributions for Male
            RIDRETH1="Other",
            DMDEDUC2="CollegeGrad",
            OCD150="Sedentary",
            INDFMPIR="LowIncome",
            SMQ020="No",
        )
        result = match_subject_to_distribution(demographics, sample_distributions)

        # Should fall back progressively until it finds RIAGENDR=Male
        assert result.distribution_key == "RIAGENDR=Male"
        assert result.n_variables_matched == 1

    def test_returns_correct_stats(self, sample_distributions):
        """Returned PopulationGroundTruth should have correct values."""
        demographics = SubjectDemographics(RIAGENDR="Male")
        result = match_subject_to_distribution(demographics, sample_distributions)

        assert result.height_mean == 175.0
        assert result.height_std == 7.5
        assert result.weight_mean == 89.0
        assert result.weight_std == 21.0
        assert result.n == 33000

    def test_no_overall_raises(self):
        """Should raise ValueError when no distribution matches and no Overall."""
        demographics = SubjectDemographics(RIAGENDR="Unknown")
        with pytest.raises(ValueError, match="No matching distribution"):
            match_subject_to_distribution(demographics, {})
