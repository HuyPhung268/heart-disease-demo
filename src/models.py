"""Danh mục mô hình đem ra so sánh."""
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data import build_preprocessor

SEED = 42

# (tên, bộ phân loại, có cần chuẩn hoá thang đo không, mô tả ngắn cho học sinh)
MODEL_SPECS = [
    (
        "Baseline (đoán lớp đa số)",
        DummyClassifier(strategy="most_frequent"),
        False,
        "Luôn đoán 'Không bệnh'. Mốc tối thiểu mà mọi mô hình phải vượt qua.",
    ),
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=SEED),
        True,
        "Mô hình tuyến tính, dễ diễn giải: mỗi biến có một hệ số ảnh hưởng.",
    ),
    (
        "Logistic Regression (cân bằng lớp)",
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        True,
        "Như trên nhưng phạt nặng hơn khi bỏ sót ca bệnh -> recall cao, accuracy giảm.",
    ),
    (
        "K-Nearest Neighbors",
        KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        True,
        "Dự đoán theo 25 hàng xóm gần nhất. Rất nhạy với thang đo nên bắt buộc chuẩn hoá.",
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(max_depth=5, random_state=SEED),
        False,
        "Cây quyết định giới hạn độ sâu 5 để tránh học vẹt.",
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1
        ),
        False,
        "Trung bình 300 cây độc lập (bagging) để giảm phương sai.",
    ),
    (
        "HistGradientBoosting",
        HistGradientBoostingClassifier(random_state=SEED),
        False,
        "Boosting: các cây nối tiếp nhau, cây sau sửa lỗi cây trước.",
    ),
    (
        "LightGBM",
        LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=SEED,
            n_jobs=-1, verbose=-1,
        ),
        False,
        "Boosting tối ưu tốc độ, thường là mô hình mạnh nhất trên dữ liệu bảng.",
    ),
]

MODEL_NOTES = {name: note for name, _, _, note in MODEL_SPECS}


def build_pipelines() -> dict[str, Pipeline]:
    """Ghép bộ tiền xử lý phù hợp với từng mô hình thành Pipeline hoàn chỉnh."""
    return {
        name: Pipeline(
            [("prep", build_preprocessor(scale=scale)), ("clf", clf)]
        )
        for name, clf, scale, _ in MODEL_SPECS
    }
