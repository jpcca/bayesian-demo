"""
Script to expand data/subjects.json from 50 to 200 subjects using NHANES data.

Follows the same methodology as src/notebooks/nhanes_data.ipynb (PR #4):
  1. Load and join NHANES tables
  2. Deduplicate by SEQN
  3. Exclude the existing 50 subjects
  4. Sample 150 new subjects (seed=43 for reproducibility)
  5. Generate natural-language descriptions using NHANES codebook mappings
  6. Add demographics using the same mappings as add_demographics.py
  7. Merge with existing subjects and save

Usage:
    cd /home/user/bayesian-demo
    uv run python src/scripts/expand_subjects.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# ---------------------------------------------------------------------------
# NHANES loading (mirrors nhanes_data.ipynb, including MCQ_L)
# ---------------------------------------------------------------------------

DATA_ROOT = project_root / "data" / "NHANES" / "August2021-August2023"


def load_nhanes_full() -> pd.DataFrame:
    """Load all NHANES survey components and inner-join on SEQN."""
    data = {}
    for f in DATA_ROOT.glob("*.xpt"):
        data[f.stem] = pd.read_sas(f)

    df_demographic = data["DEMO_L"][[
        "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1",
        "DMDEDUC2", "DMDMARTZ", "INDFMPIR",
    ]]
    df_food = data["DR1IFF_L"][[
        "SEQN", "DR1IFDCD", "DR1IGRMS", "DR1_030Z", "DR1_040Z", "DR1FS",
    ]]
    df_blood_pressure = data["BPXO_L"][[
        "SEQN",
        "BPXOSY1", "BPXOSY2", "BPXOSY3",
        "BPXODI1", "BPXODI2", "BPXODI3",
        "BPXOPLS1", "BPXOPLS2", "BPXOPLS3",
    ]]
    df_body_measure = data["BMX_L"][["SEQN", "BMXHT", "BMXWT"]]
    df_income = data["INQ_L"][["SEQN", "INQ300"]]
    df_medical = data["MCQ_L"][[
        "SEQN", "MCQ010", "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160F",
    ]]
    df_occupation = data["OCQ_L"][["SEQN", "OCD150"]]
    df_physical_activity = data["PAQ_L"][[
        "SEQN", "PAD790Q", "PAD790U", "PAD810Q", "PAD810U", "PAD820", "PAD680",
    ]]
    df_smoking = data["SMQ_L"][["SEQN", "SMQ020"]]

    df_nhanes = (
        df_demographic
        .merge(df_food, on="SEQN")
        .merge(df_blood_pressure, on="SEQN")
        .merge(df_body_measure, on="SEQN")
        .merge(df_income, on="SEQN")
        .merge(df_medical, on="SEQN")
        .merge(df_occupation, on="SEQN")
        .merge(df_physical_activity, on="SEQN")
        .merge(df_smoking, on="SEQN")
    )
    return df_nhanes


# ---------------------------------------------------------------------------
# Demographic mappings (same as add_demographics.py)
# ---------------------------------------------------------------------------

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
ACTIVITY_MAP = {1.0: "Sedentary", 2.0: "Light", 3.0: "Moderate", 4.0: "Heavy"}
SMOKING_MAP = {1.0: "Yes", 2.0: "No"}


def map_income_ratio(x):
    if pd.isna(x):
        return None
    elif x < 1:
        return "BelowPoverty"
    elif x < 2.5:
        return "LowIncome"
    elif x < 4:
        return "MiddleIncome"
    else:
        return "HighIncome"


def map_age_to_bin(age: float, gt_path: Path) -> str | None:
    with open(gt_path, "r") as f:
        gt = json.load(f)
    age_labels = gt["metadata"]["variables"]["RIDAGEYR"]["values"]
    bins = []
    for label in age_labels:
        lo, hi = label.split("-")
        bins.append((int(lo), int(hi), label))
    age_int = int(age)
    for lo, hi, label in bins:
        if lo <= age_int <= hi:
            return label
    return None


def build_demographics(row: pd.Series, gt_path: Path) -> dict:
    demo = {}
    if not pd.isna(row.get("RIDAGEYR")):
        label = map_age_to_bin(row["RIDAGEYR"], gt_path)
        if label:
            demo["RIDAGEYR"] = label
    if not pd.isna(row.get("RIAGENDR")):
        v = GENDER_MAP.get(row["RIAGENDR"])
        if v:
            demo["RIAGENDR"] = v
    if not pd.isna(row.get("RIDRETH1")):
        v = RACE_MAP.get(row["RIDRETH1"])
        if v:
            demo["RIDRETH1"] = v
    if not pd.isna(row.get("DMDEDUC2")):
        v = EDUCATION_MAP.get(row["DMDEDUC2"])
        if v:
            demo["DMDEDUC2"] = v
    if not pd.isna(row.get("INDFMPIR")):
        v = map_income_ratio(row["INDFMPIR"])
        if v:
            demo["INDFMPIR"] = v
    if not pd.isna(row.get("OCD150")):
        v = ACTIVITY_MAP.get(row["OCD150"])
        if v:
            demo["OCD150"] = v
    if not pd.isna(row.get("SMQ020")):
        v = SMOKING_MAP.get(row["SMQ020"])
        if v:
            demo["SMQ020"] = v
    return demo


# ---------------------------------------------------------------------------
# Template-based description generation (same info as original notebook,
# but deterministic — avoids needing a nested Claude session)
# ---------------------------------------------------------------------------

_RACE_TEXT = {
    1.0: "Mexican American",
    2.0: "other Hispanic",
    3.0: "non-Hispanic White",
    4.0: "non-Hispanic Black",
    5.0: "other race",
}
_EDUCATION_TEXT = {
    1.0: "less than a 9th-grade education",
    2.0: "some high school education (9th–11th grade)",
    3.0: "a high school degree or equivalent",
    4.0: "some college education",
    5.0: "a college degree or above",
}
_MARITAL_TEXT = {
    1.0: "married",
    2.0: "widowed",
    3.0: "divorced",
    4.0: "separated",
    5.0: "never married",
    6.0: "living with a partner",
}
_ACTIVITY_TEXT = {
    1.0: "sedentary",
    2.0: "light",
    3.0: "moderate",
    4.0: "heavy",
    5.0: None,  # not working
}
_OCCASION_TEXT = {
    1.0: "breakfast",
    2.0: "lunch",
    3.0: "dinner",
    4.0: "supper",
    5.0: "brunch",
    6.0: "a snack",
    7.0: "a drink",
}


def _fmt(val) -> float | None:
    """Return float or None for NaN/missing."""
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def generate_description(row: pd.Series) -> str:
    """Build a natural-language description for one NHANES subject row."""
    seqn = int(row["SEQN"])

    # --- Basic demographics ---
    age = _fmt(row.get("RIDAGEYR"))
    gender_code = _fmt(row.get("RIAGENDR"))
    race_code = _fmt(row.get("RIDRETH1"))
    educ_code = _fmt(row.get("DMDEDUC2"))
    marital_code = _fmt(row.get("DMDMARTZ"))
    indfmpir = _fmt(row.get("INDFMPIR"))

    pronoun = "He" if gender_code == 1.0 else "She"
    pronoun_obj = "him" if gender_code == 1.0 else "her"
    gender_noun = "man" if gender_code == 1.0 else "woman"

    race_text = _RACE_TEXT.get(race_code, "")
    age_text = f"{int(age)}-year-old" if age is not None else ""
    educ_text = _EDUCATION_TEXT.get(educ_code, "")
    marital_text = _MARITAL_TEXT.get(marital_code, "")

    # Sentence 1: age, race, gender, education
    sentence1_parts = [pronoun, "is"]
    if age_text:
        sentence1_parts.append(f"a {age_text}")
    if race_text:
        sentence1_parts.append(f"{race_text} {gender_noun}" if age_text else f"a {race_text} {gender_noun}")
    elif gender_noun:
        sentence1_parts.append(gender_noun if age_text else f"a {gender_noun}")

    if educ_text:
        sentence1_parts.append(f"with {educ_text}")
    if marital_text:
        sentence1_parts.append(f"who is {marital_text}")
    sentence1 = " ".join(sentence1_parts) + "."

    # Sentence 2: income
    sentence2 = ""
    if indfmpir is not None:
        pct = int(round(indfmpir * 100))
        if indfmpir < 1:
            sentence2 = f"{pronoun} has a family income below the poverty line."
        else:
            sentence2 = f"{pronoun} has a family income at {pct}% of the poverty line."

    # Sentence 3: work activity
    ocd150 = _fmt(row.get("OCD150"))
    activity_text = _ACTIVITY_TEXT.get(ocd150)
    sentence3 = ""
    if ocd150 == 5.0:
        sentence3 = f"{pronoun} is not currently working."
    elif activity_text:
        sentence3 = f"{pronoun} reports {activity_text}-intensity physical activity at work."

    # Sentence 4: sedentary minutes
    pad680 = _fmt(row.get("PAD680"))
    sentence4 = ""
    if pad680 is not None and pad680 > 0:
        sentence4 = f"{pronoun} spends approximately {int(pad680)} minutes per day in sedentary activity."

    # Sentence 5: vigorous leisure activity
    pad820 = _fmt(row.get("PAD820"))
    sentence5 = ""
    if pad820 is not None and pad820 > 0:
        sentence5 = (
            f"{pronoun} engages in {int(pad820)} minutes of vigorous-intensity "
            "leisure-time physical activity per session."
        )

    # Sentence 6: medical conditions
    conditions = []
    if _fmt(row.get("MCQ010")) == 1.0:
        conditions.append("asthma")
    if _fmt(row.get("MCQ160A")) == 1.0:
        conditions.append("arthritis")
    if _fmt(row.get("MCQ160B")) == 1.0:
        conditions.append("congestive heart failure")
    if _fmt(row.get("MCQ160C")) == 1.0:
        conditions.append("coronary heart disease")
    if _fmt(row.get("MCQ160F")) == 1.0:
        conditions.append("a stroke")
    sentence6 = ""
    if conditions:
        cond_str = ", ".join(conditions[:-1]) + (" and " if len(conditions) > 1 else "") + conditions[-1]
        sentence6 = f"{pronoun} has a history of {cond_str}."

    # Sentence 7: smoking
    smq020 = _fmt(row.get("SMQ020"))
    sentence7 = ""
    if smq020 == 1.0:
        sentence7 = f"{pronoun} has smoked at least 100 cigarettes in {pronoun_obj} lifetime."
    elif smq020 == 2.0:
        sentence7 = f"{pronoun} has never smoked."

    # Sentence 8: dietary recall (food occasion)
    dr1_030z = _fmt(row.get("DR1_030Z"))
    occasion = _OCCASION_TEXT.get(dr1_030z)
    sentence8 = ""
    if occasion:
        sentence8 = f"{pronoun} consumed {occasion} during the dietary recall."

    parts = [s for s in [sentence1, sentence2, sentence3, sentence4,
                         sentence5, sentence6, sentence7, sentence8] if s]
    return " ".join(parts)


def generate_descriptions_for_batch(df_batch: pd.DataFrame) -> dict[int, str]:
    """Generate descriptions for all subjects in a batch."""
    return {int(row["SEQN"]): generate_description(row) for _, row in df_batch.iterrows()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading NHANES data (this may take a moment)...")
    df_nhanes = load_nhanes_full()
    print(f"  Total rows (with food table duplicates): {len(df_nhanes)}")

    # Deduplicate by SEQN — one row per unique subject
    df_unique = df_nhanes.drop_duplicates(subset=["SEQN"]).reset_index(drop=True)
    print(f"  Unique subjects: {len(df_unique)}")

    # Load existing subjects
    subjects_path = project_root / "data" / "subjects.json"
    with open(subjects_path, "r") as f:
        existing_subjects = json.load(f)
    existing_seqns = {int(s["subject_id"]) for s in existing_subjects}
    print(f"  Existing subjects: {len(existing_seqns)}")

    # Filter out already-included subjects and those with missing body measurements
    df_candidates = df_unique[~df_unique["SEQN"].isin(existing_seqns)].reset_index(drop=True)
    df_candidates = df_candidates.dropna(subset=["BMXHT", "BMXWT"]).reset_index(drop=True)
    print(f"  Candidate new subjects (with valid measurements): {len(df_candidates)}")

    # Sample 150 new subjects (seed=43, distinct from original seed=42)
    n_new = 150
    df_new = df_candidates.sample(n=n_new, random_state=43).reset_index(drop=True)
    print(f"  Sampled {len(df_new)} new subjects")

    # Generate descriptions using NHANES codebook mappings
    print("  Generating descriptions from NHANES codebook mappings...")
    all_descriptions = generate_descriptions_for_batch(df_new)
    print(f"  Generated {len(all_descriptions)} descriptions")

    # Build new subject entries
    gt_path = project_root / "data" / "processed" / "nhanes_ground_truth.json"
    new_subjects = []
    missing_desc = []
    for _, row in df_new.iterrows():
        seqn = int(row["SEQN"])
        if seqn not in all_descriptions:
            missing_desc.append(seqn)
            continue
        demo = build_demographics(row, gt_path)
        new_subjects.append({
            "subject_id": str(seqn),
            "text_description": all_descriptions[seqn],
            "height_cm": float(row["BMXHT"]),
            "weight_kg": float(row["BMXWT"]),
            "demographics": demo,
        })

    if missing_desc:
        print(f"WARNING: Missing descriptions for SEQNs: {missing_desc}")

    print(f"New subjects created: {len(new_subjects)}")

    # Combine with existing and save
    combined = existing_subjects + new_subjects
    print(f"Total subjects: {len(combined)}")

    with open(subjects_path, "w") as f:
        json.dump(combined, f, indent=4)

    print(f"\nSaved {len(combined)} subjects to {subjects_path}")

    # Quick verification
    if new_subjects:
        import statistics
        heights = [s["height_cm"] for s in new_subjects]
        weights = [s["weight_kg"] for s in new_subjects]
        print(f"\nNew subjects stats:")
        print(f"  Height: mean={statistics.mean(heights):.1f} cm, std={statistics.stdev(heights):.1f} cm")
        print(f"  Weight: mean={statistics.mean(weights):.1f} kg, std={statistics.stdev(weights):.1f} kg")


if __name__ == "__main__":
    main()
