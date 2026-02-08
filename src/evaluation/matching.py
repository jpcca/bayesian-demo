"""
Subject-to-distribution matching for ground truth population distributions.

Matches each test subject to their best-fitting demographic group in the
NHANES ground truth data, with progressive fallback when exact matches
are not available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from models.schemas import PopulationGroundTruth, SubjectDemographics


# Variables ordered from most to least informative for height/weight prediction.
# When an exact match isn't found, we drop variables from the END of this list first.
VARIABLE_PRIORITY = [
    "RIAGENDR",  # Gender — most informative for body size
    "RIDAGEYR",  # Age — strong effect on height/weight
    "RIDRETH1",  # Race/Ethnicity
    "DMDEDUC2",  # Education level
    "OCD150",    # Work activity level
    "INQ300",    # Household income
    "SMQ020",    # Smoking status — least informative for body size
]


def load_ground_truth_distributions(path: str | Path) -> Dict[str, dict]:
    """
    Load ground truth distributions from nhanes_ground_truth.json.

    Args:
        path: Path to the JSON file

    Returns:
        Dict mapping distribution keys to stats dicts
        (each with height_mean, height_std, weight_mean, weight_std, n)
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data["distributions"]


def build_distribution_key(demographics: Dict[str, str], variables: List[str]) -> str:
    """
    Build a distribution lookup key from demographic values.

    Args:
        demographics: Dict of variable_name -> value (e.g., {"RIAGENDR": "Male", ...})
        variables: Ordered list of variable names to include in the key

    Returns:
        Key string like "RIAGENDR=Male__RIDAGEYR=18-37"

    Raises:
        ValueError: If any requested variable is missing from demographics
    """
    parts = []
    for var in variables:
        if var not in demographics or demographics[var] is None:
            raise ValueError(f"Variable {var} not found in demographics")
        parts.append(f"{var}={demographics[var]}")
    return "__".join(parts)


def match_subject_to_distribution(
    demographics: "SubjectDemographics",
    distributions: Dict[str, dict],
) -> "PopulationGroundTruth":
    """
    Match a subject to their best ground truth population distribution.

    Tries the most specific match first (all available variables), then
    progressively drops the least informative variables until a match is found.

    Args:
        demographics: SubjectDemographics with categorical variable values
        distributions: Dict of distribution keys to stats from nhanes_ground_truth.json

    Returns:
        PopulationGroundTruth with the matched distribution parameters
    """
    from models.schemas import PopulationGroundTruth

    # Convert demographics model to dict, filtering out None values
    demo_dict = {
        k: v for k, v in demographics.model_dump().items()
        if v is not None
    }

    # Get available variables in priority order
    available_vars = [v for v in VARIABLE_PRIORITY if v in demo_dict]

    # Try progressively fewer variables, dropping from least informative
    for n_vars in range(len(available_vars), 0, -1):
        vars_to_try = available_vars[:n_vars]
        try:
            key = build_distribution_key(demo_dict, vars_to_try)
        except ValueError:
            continue

        if key in distributions:
            stats = distributions[key]
            return PopulationGroundTruth(
                distribution_key=key,
                height_mean=stats["height_mean"],
                height_std=stats["height_std"],
                weight_mean=stats["weight_mean"],
                weight_std=stats["weight_std"],
                n=stats["n"],
                n_variables_matched=n_vars,
            )

    # Final fallback: overall population distribution
    if "Overall" in distributions:
        stats = distributions["Overall"]
        return PopulationGroundTruth(
            distribution_key="Overall",
            height_mean=stats["height_mean"],
            height_std=stats["height_std"],
            weight_mean=stats["weight_mean"],
            weight_std=stats["weight_std"],
            n=stats["n"],
            n_variables_matched=0,
        )

    raise ValueError("No matching distribution found, not even 'Overall'")
