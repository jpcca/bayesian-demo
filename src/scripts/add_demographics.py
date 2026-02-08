"""
One-time script to enrich data/subjects.json with demographic variables.

Looks up each subject by SEQN in the raw NHANES data and applies the same
categorical mappings used in nhanes_joint_distribution.ipynb, so that each
subject can be matched to their ground truth population distribution.

Usage:
    cd src
    python scripts/add_demographics.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.nhanes import load_nhanes_data


# Same categorical mappings as nhanes_joint_distribution.ipynb
GENDER_MAP = {1.0: "Male", 2.0: "Female"}
RACE_MAP = {
    1.0: "MexicanAmerican",
    2.0: "OtherHispanic",
    3.0: "White",
    4.0: "Black",
    5.0: "Other",
}
EDUCATION_MAP = {
    1.0: "LessThan9th",
    2.0: "9thTo11th",
    3.0: "HighSchool",
    4.0: "SomeCollege",
    5.0: "CollegeGrad",
}
ACTIVITY_MAP = {
    1.0: "Sedentary",
    2.0: "Light",
    3.0: "Moderate",
    4.0: "Heavy",
}
SMOKING_MAP = {1.0: "Yes", 2.0: "No"}


def map_income(x):
    """Map INQ300 codes to income categories (matches nhanes_joint_distribution.ipynb)."""
    if pd.isna(x):
        return None
    elif x <= 4:
        return "Under20k"
    elif x <= 7:
        return "20kTo45k"
    elif x <= 10:
        return "45kTo75k"
    elif x <= 14:
        return "75kTo100k"
    else:
        return "Over100k"


def map_age_to_bin(age: float, gt_path: Path) -> str | None:
    """Map age to the same bins used in nhanes_ground_truth.json.

    Reads the actual age bin labels from the JSON metadata to ensure consistency.
    """
    with open(gt_path, "r") as f:
        gt = json.load(f)

    age_labels = gt["metadata"]["variables"]["RIDAGEYR"]["values"]
    # Parse bin edges from labels like ["18-37", "38-53", ...]
    bins = []
    for label in age_labels:
        lo, hi = label.split("-")
        bins.append((int(lo), int(hi), label))

    age_int = int(age)
    for lo, hi, label in bins:
        if lo <= age_int <= hi:
            return label
    return None


def main():
    print("Loading NHANES data...")
    df = load_nhanes_data()

    # Deduplicate by SEQN (food table creates duplicates)
    df_unique = df.drop_duplicates(subset=["SEQN"])
    print(f"Unique subjects: {len(df_unique)}")

    gt_path = project_root / "data" / "processed" / "nhanes_ground_truth.json"

    # Build demographics lookup by SEQN
    demographics_by_seqn = {}
    for _, row in df_unique.iterrows():
        seqn = int(row["SEQN"])
        demo = {}

        # Age bin — use the same bins as nhanes_ground_truth.json
        if not pd.isna(row.get("RIDAGEYR")):
            age_label = map_age_to_bin(row["RIDAGEYR"], gt_path)
            if age_label:
                demo["RIDAGEYR"] = age_label

        # Gender
        if not pd.isna(row.get("RIAGENDR")):
            demo["RIAGENDR"] = GENDER_MAP.get(row["RIAGENDR"])

        # Race
        if not pd.isna(row.get("RIDRETH1")):
            demo["RIDRETH1"] = RACE_MAP.get(row["RIDRETH1"])

        # Education
        if not pd.isna(row.get("DMDEDUC2")):
            demo["DMDEDUC2"] = EDUCATION_MAP.get(row["DMDEDUC2"])

        # Income
        if not pd.isna(row.get("INQ300")):
            demo["INQ300"] = map_income(row["INQ300"])

        # Work activity
        if not pd.isna(row.get("OCD150")):
            demo["OCD150"] = ACTIVITY_MAP.get(row["OCD150"])

        # Smoking
        if not pd.isna(row.get("SMQ020")):
            demo["SMQ020"] = SMOKING_MAP.get(row["SMQ020"])

        # Filter out None values
        demo = {k: v for k, v in demo.items() if v is not None}
        demographics_by_seqn[seqn] = demo

    # Load subjects.json
    subjects_path = project_root / "data" / "subjects.json"
    with open(subjects_path, "r") as f:
        subjects = json.load(f)

    # Enrich each subject
    matched = 0
    unmatched = []
    for subject in subjects:
        seqn = int(subject["subject_id"])
        if seqn in demographics_by_seqn:
            subject["demographics"] = demographics_by_seqn[seqn]
            matched += 1
        else:
            unmatched.append(seqn)
            subject["demographics"] = {}

    print(f"Matched: {matched}/{len(subjects)}")
    if unmatched:
        print(f"Unmatched SEQNs: {unmatched}")

    # Save
    with open(subjects_path, "w") as f:
        json.dump(subjects, f, indent=4)

    print(f"Updated {subjects_path}")

    # Print sample
    print("\nSample output:")
    for s in subjects[:3]:
        print(f"  {s['subject_id']}: {s.get('demographics', {})}")


if __name__ == "__main__":
    main()
