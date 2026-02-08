"""NHANES Data Loader"""

from pathlib import Path
import pandas as pd


DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "NHANES" / "August2021-August2023"

# Column definitions for each data file
COLUMN_DEFINITIONS = {
    "DEMO_L": {
        "description": "Demographics",
        "columns": {
            "SEQN": "Respondent sequence number (subject ID)",
            "RIDAGEYR": "Age in years at screening",
            "RIAGENDR": "Gender (1=Male, 2=Female)",
            "RIDRETH1": "Race/Hispanic origin",
            "DMDEDUC2": "Education level - Adults 20+",
            "DMDMARTZ": "Marital status",
            "INDFMPIR": "Family poverty income ratio",
        },
    },
    "DR1IFF_L": {
        "description": "Dietary Interview - Individual Foods",
        "columns": {
            "SEQN": "Subject ID",
            "DR1IFDCD": "USDA food code",
            "DR1IGRMS": "Grams consumed",
            "DR1_030Z": "Name of eating occasion (breakfast/lunch etc.)",
            "DR1_040Z": "Time of eating occasion",
            "DR1FS": "Food source",
        },
    },
    "BPXO_L": {
        "description": "Blood Pressure - Oscillometric",
        "columns": {
            "SEQN": "Subject ID",
            "BPXOSY1": "Systolic BP reading 1",
            "BPXOSY2": "Systolic BP reading 2",
            "BPXOSY3": "Systolic BP reading 3",
            "BPXODI1": "Diastolic BP reading 1",
            "BPXODI2": "Diastolic BP reading 2",
            "BPXODI3": "Diastolic BP reading 3",
            "BPXOPLS1": "Pulse reading 1",
            "BPXOPLS2": "Pulse reading 2",
            "BPXOPLS3": "Pulse reading 3",
        },
    },
    "BMX_L": {
        "description": "Body Measures",
        "columns": {
            "SEQN": "Subject ID",
            "BMXHT": "Standing height (cm)",
            "BMXWT": "Weight (kg)",
        },
    },
    "INQ_L": {
        "description": "Income",
        "columns": {
            "SEQN": "Subject ID",
            "INQ300": "Total family income",
        },
    },
    "OCQ_L": {
        "description": "Occupation",
        "columns": {
            "SEQN": "Subject ID",
            "OCD150": "Work activity level (typical physical activity at work)",
        },
    },
    "PAQ_L": {
        "description": "Physical Activity",
        "columns": {
            "SEQN": "Subject ID",
            "PAD790Q": "Vigorous work activity",
            "PAD790U": "Moderate work activity",
            "PAD810Q": "Moderate recreational activity",
            "PAD810U": "Sedentary activity (minutes)",
            "PAD820": "Minutes vigorous LTPA",
            "PAD680": "Minutes sedentary activity",
        },
    },
    "SMQ_L": {
        "description": "Smoking",
        "columns": {
            "SEQN": "Subject ID",
            "SMQ020": "Smoked at least 100 cigarettes in life",
        },
    },
}


def load_nhanes_data() -> pd.DataFrame:
    """Load and merge NHANES data from multiple survey components."""
    data = {}
    for f in DATA_ROOT.glob("*.xpt"):
        data[f.stem] = pd.read_sas(f)

    dfs = [data[key][list(defn["columns"].keys())] for key, defn in COLUMN_DEFINITIONS.items()]

    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="SEQN")

    return result
