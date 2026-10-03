"""Tinh chỉnh siêu tham số bằng GridSearchCV và so sánh với mô hình mặc định.

Chạy:  python tune.py
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from src.data import build_preprocessor, get_splits
from src.models import SEED

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

# (tên, bộ phân loại, cần chuẩn hoá?, lưới siêu tham số)
# Tiền tố "clf__" là cách sklearn trỏ tham số vào bước tên "clf" trong Pipeline.
GRIDS = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=2000, random_state=SEED),
        True,
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear"],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(random_state=SEED),
        False,
        {
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [1, 20, 100],
            "clf__criterion": ["gini", "entropy"],
        },
    ),
    (
        "Random Forest",
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        False,
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [4, 8, None],
            "clf__min_samples_leaf": [1, 20],
        },
    ),
    (
        "LightGBM",
        LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1),
        False,
        {
            "clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__num_leaves": [15, 31],
        },
    ),
]


def main():
    X_train, X_test, y_train, y_test = get_splits()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, searches = [], {}
    for name, clf, scale, grid in GRIDS:
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"  -> {name}: {n_combos} tổ hợp × 5 fold = {n_combos * 5} lần fit")

        pipe = Pipeline([("prep", build_preprocessor(scale=scale)), ("clf", clf)])

        # Mô hình mặc định (không tinh chỉnh) để làm mốc so sánh.
        base = Pipeline([("prep", build_preprocessor(scale=scale)), ("clf", clf)])
        base.fit(X_train, y_train)
        base_auc = roc_auc_score(y_test, base.predict_proba(X_test)[:, 1])

        search = GridSearchCV(
            pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        tuned_auc = roc_auc_score(
            y_test, search.best_estimator_.predict_proba(X_test)[:, 1]
        )

        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        rows.append(
            {
                "Mô hình": name,
                "Số tổ hợp": n_combos,
                "CV ROC-AUC (mặc định)": search.cv_results_["mean_test_score"].mean(),
                "CV ROC-AUC (tốt nhất)": search.best_score_,
                "Độ lệch CV của tổ hợp tốt nhất": cv_std,
                "Test ROC-AUC (mặc định)": base_auc,
                "Test ROC-AUC (đã tinh chỉnh)": tuned_auc,
                "Cải thiện trên test": tuned_auc - base_auc,
                "Tham số tốt nhất": str(search.best_params_),
            }
        )
        searches[name] = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": pd.DataFrame(search.cv_results_)[
                [
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "mean_train_score",
                    "rank_test_score",
                ]
            ],
            "best_estimator": search.best_estimator_,
            "test_auc_tuned": tuned_auc,
            "test_auc_default": base_auc,
        }

    tuning = pd.DataFrame(rows)
    joblib.dump(
        {"tuning": tuning, "searches": searches},
        ARTIFACTS / "tuning.joblib",
        compress=3,
    )
    tuning.to_csv(ARTIFACTS / "tuning.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + "=" * 130)
    print(tuning.drop(columns=["Tham số tốt nhất"]).round(4).to_string(index=False))
    print("=" * 130)
    for r in rows:
        print(f"\n{r['Mô hình']}: {r['Tham số tốt nhất']}")
    print(f"\nĐã lưu vào {ARTIFACTS}")


if __name__ == "__main__":
    main()
