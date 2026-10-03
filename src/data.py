"""Data loading and the preprocessing pipeline for the heart_disease dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "heart_disease.csv"

TARGET = "Heart Disease Status"

NUMERIC_COLS = [
    "Age",
    "Blood Pressure",
    "Cholesterol Level",
    "BMI",
    "Sleep Hours",
    "Triglyceride Level",
    "Fasting Blood Sugar",
    "CRP Level",
    "Homocysteine Level",
]

# Ordinal features: the order carries meaning, so encode them as increasing
# integers rather than one-hot.
ORDINAL_COLS = {
    "Exercise Habits": ["Low", "Medium", "High"],
    "Alcohol Consumption": ["None", "Low", "Medium", "High"],
    "Stress Level": ["Low", "Medium", "High"],
    "Sugar Consumption": ["Low", "Medium", "High"],
}

# Nominal features: no inherent order, so one-hot encode them.
NOMINAL_COLS = [
    "Gender",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
]

FEATURE_COLS = NUMERIC_COLS + list(ORDINAL_COLS) + NOMINAL_COLS


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV.

    Important: pandas treats the string "None" as a missing value by default,
    but in the Alcohol Consumption column "None" is a valid category meaning
    "does not drink". So we disable the default NA list and treat only empty
    cells as missing.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load_xy(path: Path | str = DATA_PATH):
    """Return (X, y) where y is the binary label, 1 = has heart disease."""
    df = load_raw(path)
    y = (df[TARGET] == "Yes").astype(int)
    X = df[FEATURE_COLS].copy()
    return X, y


def get_splits(test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split that preserves the 80/20 class balance."""
    X, y = load_xy()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Preprocessing pipeline: impute -> encode -> scale.

    Pass scale=False for tree-based models, which do not need feature scaling.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    ordinal_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=list(ORDINAL_COLS.values()))),
        ]
    )

    nominal_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_COLS),
            ("ord", ordinal_pipe, list(ORDINAL_COLS)),
            ("nom", nominal_pipe, NOMINAL_COLS),
        ]
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Transformed column names, used for importance charts."""
    return list(preprocessor.get_feature_names_out())
