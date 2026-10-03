"""Nạp dữ liệu và xây pipeline tiền xử lý cho bộ heart_disease."""
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

# Biến thứ bậc: thứ tự có ý nghĩa nên mã hoá bằng số tăng dần, không one-hot.
ORDINAL_COLS = {
    "Exercise Habits": ["Low", "Medium", "High"],
    "Alcohol Consumption": ["None", "Low", "Medium", "High"],
    "Stress Level": ["Low", "Medium", "High"],
    "Sugar Consumption": ["Low", "Medium", "High"],
}

# Biến danh mục không có thứ tự -> one-hot.
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
    """Đọc CSV.

    Quan trọng: pandas mặc định coi chuỗi "None" là giá trị thiếu, trong khi ở cột
    Alcohol Consumption thì "None" là một hạng mục hợp lệ (không uống rượu).
    Vì vậy phải tắt danh sách NA mặc định và chỉ coi ô rỗng là thiếu.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load_xy(path: Path | str = DATA_PATH):
    """Trả về (X, y) với y là nhãn nhị phân 1 = có bệnh."""
    df = load_raw(path)
    y = (df[TARGET] == "Yes").astype(int)
    X = df[FEATURE_COLS].copy()
    return X, y


def get_splits(test_size: float = 0.2, random_state: int = 42):
    """Chia train/test có phân tầng để giữ nguyên tỷ lệ 80/20 của nhãn."""
    X, y = load_xy()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Pipeline tiền xử lý: điền khuyết -> mã hoá -> chuẩn hoá.

    scale=False dùng cho các mô hình cây (không cần chuẩn hoá thang đo).
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
    """Tên cột sau khi biến đổi, dùng để vẽ biểu đồ độ quan trọng."""
    return list(preprocessor.get_feature_names_out())
