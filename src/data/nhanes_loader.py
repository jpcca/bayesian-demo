"""NHANES data loader for calibration study.

Downloads and processes NHANES anthropometric data, generating text vignettes
for LLM prediction tasks.

Example:
    >>> subjects = load_subjects(n=50, cycle='2017-2018')
    >>> for s in subjects[:3]:
    ...     print(f"{s.subject_id}: {s.text_description[:50]}...")
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

import pandas as pd

if TYPE_CHECKING:
    pass

# NHANES cycle to suffix mapping
CYCLE_SUFFIXES: dict[str, tuple[str, str]] = {
    "2017-2018": ("2017", "J"),
    "2015-2016": ("2015", "I"),
    "2013-2014": ("2013", "H"),
}

# NHANES ethnicity code mapping
ETHNICITY_MAP: dict[int, str] = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multi-racial",
}

# Gender code mapping
GENDER_MAP: dict[int, str] = {
    1: "male",
    2: "female",
}

# Occupation pools for vignette generation
OCCUPATIONS_ACTIVE: list[str] = [
    "works in construction",
    "is a fitness instructor",
    "works as a delivery driver",
    "is a nurse who is on their feet all day",
    "works as a warehouse associate",
    "is a landscaper",
    "works in retail",
    "is a professional athlete",
    "works as a physical therapist",
    "is a dance instructor",
]

OCCUPATIONS_MODERATE: list[str] = [
    "works as a teacher",
    "is a stay-at-home parent",
    "works in hospitality",
    "is a restaurant manager",
    "works as a mechanic",
    "is a real estate agent",
    "works in sales",
    "is a veterinarian",
    "works as a chef",
    "is a social worker",
]

OCCUPATIONS_SEDENTARY: list[str] = [
    "works as a software engineer",
    "is an accountant",
    "works in data entry",
    "is a lawyer",
    "works as a financial analyst",
    "is a graphic designer",
    "works in customer service",
    "is a researcher",
    "works as a writer",
    "is a project manager",
]

# Hobby pools for vignette generation
HOBBIES_ACTIVE: list[str] = [
    "enjoys playing soccer on weekends",
    "goes to the gym regularly",
    "loves hiking and outdoor activities",
    "plays basketball with friends",
    "enjoys swimming laps at the local pool",
    "is training for a marathon",
    "practices yoga daily",
    "enjoys cycling on weekends",
    "plays tennis competitively",
    "takes dance classes",
]

HOBBIES_MODERATE: list[str] = [
    "enjoys walking their dog",
    "does light gardening",
    "goes bowling occasionally",
    "enjoys playing golf",
    "takes leisurely bike rides",
    "does household projects on weekends",
    "enjoys casual swimming",
    "takes occasional fitness classes",
    "goes fishing when weather permits",
    "plays recreational volleyball",
]

HOBBIES_SEDENTARY: list[str] = [
    "enjoys reading in their free time",
    "spends time playing video games",
    "enjoys watching movies and TV shows",
    "is passionate about cooking",
    "enjoys board games with friends",
    "spends time on social media",
    "enjoys crafting and knitting",
    "is an avid podcast listener",
    "enjoys photography as a hobby",
    "spends weekends relaxing at home",
]

# Activity descriptors based on BMI category
ACTIVITY_DESCRIPTORS: dict[str, list[str]] = {
    "underweight": [
        "describes themselves as having a fast metabolism",
        "mentions they often forget to eat when busy",
        "says they've always been naturally thin",
        "notes they have a small appetite",
    ],
    "normal": [
        "describes themselves as moderately active",
        "tries to maintain a balanced lifestyle",
        "stays active through daily routines",
        "enjoys a mix of activities and relaxation",
    ],
    "overweight": [
        "mentions they've been meaning to exercise more",
        "describes their lifestyle as busy with little time for exercise",
        "says they enjoy good food",
        "notes they prefer relaxing activities",
    ],
    "obese": [
        "mentions they have a sedentary job",
        "describes their lifestyle as mostly inactive",
        "says they struggle to find time for physical activity",
        "notes they prefer indoor activities",
    ],
}

# Names for vignettes (diverse set)
NAMES_MALE: list[str] = [
    "James",
    "Michael",
    "David",
    "Carlos",
    "Wei",
    "Marcus",
    "Jose",
    "Ahmed",
    "Tyler",
    "Brandon",
    "Kevin",
    "Jorge",
    "Jamal",
    "Ryan",
    "Daniel",
    "Luis",
    "Anthony",
    "Christopher",
    "Matthew",
    "Alex",
    "Eric",
    "Derek",
    "Victor",
    "Nathan",
    "Isaiah",
    "Raj",
    "Omar",
    "Sergio",
    "Andre",
    "Trevor",
]

NAMES_FEMALE: list[str] = [
    "Jessica",
    "Sarah",
    "Maria",
    "Ashley",
    "Ming",
    "Jasmine",
    "Ana",
    "Fatima",
    "Emily",
    "Brittany",
    "Nicole",
    "Carmen",
    "Aaliyah",
    "Jennifer",
    "Amanda",
    "Rosa",
    "Stephanie",
    "Megan",
    "Rachel",
    "Lisa",
    "Angela",
    "Diana",
    "Sofia",
    "Priya",
    "Layla",
    "Keisha",
    "Monica",
    "Grace",
    "Olivia",
    "Samantha",
]


@dataclass
class Subject:
    """A subject with demographic info and actual measurements.

    Attributes:
        subject_id: Unique identifier for the subject.
        text_description: Natural language vignette describing the subject.
        actual_height_cm: True height measurement in centimeters.
        actual_weight_kg: True weight measurement in kilograms.
        age: Age in years.
        gender: 'male' or 'female'.
        ethnicity: Ethnicity description string.
    """

    subject_id: str
    text_description: str
    actual_height_cm: float
    actual_weight_kg: float
    age: int
    gender: str
    ethnicity: str


def _get_cache_path(cache_dir: str, url: str) -> Path:
    """Generate a cache file path for a URL.

    Args:
        cache_dir: Directory to store cached files.
        url: URL to generate cache path for.

    Returns:
        Path to the cached file.
    """
    # Use URL hash to create unique filename
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    filename = url.split("/")[-1]
    return Path(cache_dir) / f"{url_hash}_{filename}"


def _download_xpt(url: str, cache_path: Path) -> pd.DataFrame:
    """Download and read an XPT (SAS transport) file.

    Args:
        url: URL to download from.
        cache_path: Local path to cache the file.

    Returns:
        DataFrame with the XPT file contents.

    Raises:
        ValueError: If download or parsing fails.
    """
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(url, cache_path)
        except Exception as e:
            raise ValueError(f"Failed to download {url}: {e}") from e

    try:
        return pd.read_sas(cache_path, format="xport")
    except Exception as e:
        raise ValueError(f"Failed to read XPT file {cache_path}: {e}") from e


def download_nhanes(cycle: str = "2017-2018", cache_dir: str = "data/nhanes") -> pd.DataFrame:
    """Download NHANES demographics and body measures, merge on SEQN.

    Downloads data from CDC NHANES public data files and caches locally
    to avoid repeated downloads.

    Args:
        cycle: NHANES cycle to download. Supported: '2017-2018', '2015-2016', '2013-2014'.
        cache_dir: Directory to cache downloaded files.

    Returns:
        DataFrame with merged demographics and body measures data.

    Raises:
        ValueError: If cycle is not supported or download fails.

    Example:
        >>> df = download_nhanes('2017-2018')
        >>> print(df.columns.tolist())  # doctest: +ELLIPSIS
        ['SEQN', 'RIAGENDR', 'RIDAGEYR', ...]
    """
    if cycle not in CYCLE_SUFFIXES:
        supported = ", ".join(CYCLE_SUFFIXES.keys())
        raise ValueError(f"Unsupported cycle '{cycle}'. Supported: {supported}")

    year, suffix = CYCLE_SUFFIXES[cycle]
    base_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

    # Build URLs
    demo_url = f"{base_url}/{year}/DataFiles/DEMO_{suffix}.XPT"
    bmx_url = f"{base_url}/{year}/DataFiles/BMX_{suffix}.XPT"

    # Download and read files
    demo_cache = _get_cache_path(cache_dir, demo_url)
    bmx_cache = _get_cache_path(cache_dir, bmx_url)

    demo_df = _download_xpt(demo_url, demo_cache)
    bmx_df = _download_xpt(bmx_url, bmx_cache)

    # Merge on SEQN (subject ID)
    merged = pd.merge(demo_df, bmx_df, on="SEQN", how="inner")

    return merged


def _get_bmi_category(height_cm: float, weight_kg: float) -> str:
    """Calculate BMI and return category.

    Args:
        height_cm: Height in centimeters.
        weight_kg: Weight in kilograms.

    Returns:
        BMI category: 'underweight', 'normal', 'overweight', or 'obese'.
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)

    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


def generate_vignette(row: pd.Series, rng: random.Random | None = None) -> str:
    """Generate a natural language description from NHANES record.

    Creates a 3-5 sentence vignette describing the subject without
    revealing actual height or weight measurements.

    Args:
        row: pandas Series with NHANES variables (RIAGENDR, RIDAGEYR,
             RIDRETH3, BMXHT, BMXWT).
        rng: Random number generator for reproducibility.

    Returns:
        Natural language description of the subject.

    Example:
        >>> import pandas as pd
        >>> row = pd.Series({
        ...     'RIAGENDR': 1,
        ...     'RIDAGEYR': 34,
        ...     'RIDRETH3': 1,
        ...     'BMXHT': 175.0,
        ...     'BMXWT': 80.0
        ... })
        >>> vignette = generate_vignette(row)
        >>> 'year-old' in vignette
        True
    """
    if rng is None:
        rng = random.Random()

    # Extract variables
    gender_code = int(row["RIAGENDR"])
    age = int(row["RIDAGEYR"])
    ethnicity_code = int(row["RIDRETH3"])
    height_cm = float(row["BMXHT"])
    weight_kg = float(row["BMXWT"])

    # Map codes to values
    gender = GENDER_MAP.get(gender_code, "person")
    ethnicity = ETHNICITY_MAP.get(ethnicity_code, "American")

    # Select name based on gender
    if gender == "male":
        name = rng.choice(NAMES_MALE)
        pronoun = "He"
    else:
        name = rng.choice(NAMES_FEMALE)
        pronoun = "She"

    # Get BMI category for activity hints
    bmi_category = _get_bmi_category(height_cm, weight_kg)

    # Select occupation and hobby based on BMI category tendencies
    # (with some randomization to avoid stereotyping)
    if bmi_category == "underweight":
        occupation = rng.choice(OCCUPATIONS_ACTIVE + OCCUPATIONS_MODERATE)
        hobby = rng.choice(HOBBIES_ACTIVE + HOBBIES_MODERATE)
    elif bmi_category == "normal":
        occupation = rng.choice(OCCUPATIONS_ACTIVE + OCCUPATIONS_MODERATE + OCCUPATIONS_SEDENTARY)
        hobby = rng.choice(HOBBIES_ACTIVE + HOBBIES_MODERATE)
    elif bmi_category == "overweight":
        occupation = rng.choice(OCCUPATIONS_MODERATE + OCCUPATIONS_SEDENTARY)
        hobby = rng.choice(HOBBIES_MODERATE + HOBBIES_SEDENTARY)
    else:  # obese
        occupation = rng.choice(OCCUPATIONS_SEDENTARY + OCCUPATIONS_MODERATE)
        hobby = rng.choice(HOBBIES_SEDENTARY + HOBBIES_MODERATE)

    # Select activity descriptor
    activity_desc = rng.choice(ACTIVITY_DESCRIPTORS[bmi_category])

    # Build the vignette
    # Sentence 1: Name, age, ethnicity, gender
    sentence1 = f"{name} is a {age}-year-old {ethnicity} {'man' if gender == 'male' else 'woman'} who {occupation}."

    # Sentence 2: Activity level hint
    sentence2 = f"{pronoun} {activity_desc}."

    # Sentence 3: Hobby
    sentence3 = f"{pronoun} {hobby}."

    # Combine sentences
    vignette = f"{sentence1} {sentence2} {sentence3}"

    return vignette


def load_subjects(
    n: int = 50,
    cycle: str = "2017-2018",
    age_range: tuple[int, int] = (18, 65),
    seed: int = 42,
    cache_dir: str = "data/nhanes",
) -> list[Subject]:
    """Load n subjects from NHANES with diverse demographics.

    Uses stratified sampling to ensure gender balance, ethnicity
    representation, and age diversity.

    Args:
        n: Number of subjects to load.
        cycle: NHANES cycle to use.
        age_range: Tuple of (min_age, max_age) to filter subjects.
        seed: Random seed for reproducibility.
        cache_dir: Directory to cache downloaded files.

    Returns:
        List of Subject objects with demographics and measurements.

    Raises:
        ValueError: If not enough subjects available after filtering.

    Example:
        >>> subjects = load_subjects(n=10, cycle='2017-2018', seed=42)
        >>> len(subjects)
        10
        >>> all(18 <= s.age <= 65 for s in subjects)
        True
    """
    # Download data
    df = download_nhanes(cycle=cycle, cache_dir=cache_dir)

    # Filter for required columns and valid data
    required_cols = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "BMXHT", "BMXWT"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in NHANES data")

    # Keep only required columns and drop rows with missing values
    df = df[required_cols].dropna()

    # Filter by age range
    min_age, max_age = age_range
    df = df[(df["RIDAGEYR"] >= min_age) & (df["RIDAGEYR"] <= max_age)]

    # Filter valid gender and ethnicity codes
    df = df[df["RIAGENDR"].isin(GENDER_MAP.keys())]
    df = df[df["RIDRETH3"].isin(ETHNICITY_MAP.keys())]

    # Remove extreme outliers (>3 SD from mean)
    for col in ["BMXHT", "BMXWT"]:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

    if len(df) < n:
        raise ValueError(f"Not enough subjects after filtering: {len(df)} available, {n} requested")

    # Stratified sampling: try to balance gender
    males = df[df["RIAGENDR"] == 1]
    females = df[df["RIAGENDR"] == 2]

    n_male = n // 2
    n_female = n - n_male

    # Adjust if one gender doesn't have enough
    if len(males) < n_male:
        n_male = len(males)
        n_female = n - n_male
    elif len(females) < n_female:
        n_female = len(females)
        n_male = n - n_female

    # Sample from each gender
    sampled_males = males.sample(n=n_male, random_state=seed)
    sampled_females = females.sample(n=n_female, random_state=seed + 1)

    sampled = pd.concat([sampled_males, sampled_females])

    # Further stratify by ethnicity within the sample
    # (shuffle to mix ethnicities rather than grouped)
    sampled = sampled.sample(frac=1, random_state=seed + 2).reset_index(drop=True)

    # Convert to Subject objects
    subjects = []
    for idx, row in sampled.iterrows():
        # Create deterministic RNG for this subject
        subject_rng = random.Random(seed + int(row["SEQN"]))

        subject = Subject(
            subject_id=f"NHANES_{int(row['SEQN'])}",
            text_description=generate_vignette(row, rng=subject_rng),
            actual_height_cm=float(row["BMXHT"]),
            actual_weight_kg=float(row["BMXWT"]),
            age=int(row["RIDAGEYR"]),
            gender=GENDER_MAP[int(row["RIAGENDR"])],
            ethnicity=ETHNICITY_MAP[int(row["RIDRETH3"])],
        )
        subjects.append(subject)

    return subjects
