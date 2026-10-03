"""Huấn luyện và so sánh toàn bộ mô hình, lưu kết quả vào thư mục artifacts/.

Chạy:  python train.py
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import get_splits
from src.models import SEED, build_pipelines

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)


def evaluate(name, pipe, X_train, X_test, y_train, y_test, cv):
    """Huấn luyện một pipeline rồi đo trên cả cross-validation lẫn tập test."""
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    row = {
        "Mô hình": name,
        "CV ROC-AUC": cv_auc.mean(),
        "CV ROC-AUC (độ lệch)": cv_auc.std(),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    curves = {
        "roc": (fpr, tpr),
        "pr": (rec, prec),
        "proba": y_proba,
        "cm": confusion_matrix(y_test, y_pred),
    }
    return row, curves, pipe


def shuffled_control(pipe, X_train, X_test, y_train, y_test):
    """Thí nghiệm đối chứng: xáo trộn nhãn để phá huỷ mọi liên hệ feature-target.

    Nếu điểm số trên nhãn thật không cao hơn trên nhãn xáo trộn thì mô hình
    không học được gì cả.
    """
    rng = np.random.default_rng(SEED)
    y_shuf = pd.Series(rng.permutation(y_train.values), index=y_train.index)
    pipe.fit(X_train, y_shuf)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pipe.predict(X_test)),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def export_splits(X_train, X_test, y_train, y_test):
    """Ghi tập train/test ra CSV để xem được bằng mắt.

    Lưu ý: hai tệp này KHÔNG phải đầu vào của chương trình. Nguồn dữ liệu duy nhất
    vẫn là Dataset/heart_disease.csv; việc chia tách diễn ra trong bộ nhớ bằng
    train_test_split(). Xuất ra đây chỉ để minh hoạ cho người học.
    """
    train = X_train.copy()
    train["Heart Disease Status"] = y_train.map({0: "No", 1: "Yes"})
    test = X_test.copy()
    test["Heart Disease Status"] = y_test.map({0: "No", 1: "Yes"})
    train.to_csv(ARTIFACTS / "train_set.csv", index=False)
    test.to_csv(ARTIFACTS / "test_set.csv", index=False)
    print(f"Đã xuất train_set.csv ({len(train):,} dòng) "
          f"và test_set.csv ({len(test):,} dòng)")


def main():
    X_train, X_test, y_train, y_test = get_splits()
    export_splits(X_train, X_test, y_train, y_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipelines = build_pipelines()

    rows, curves, fitted = [], {}, {}
    for name, pipe in pipelines.items():
        print(f"  -> {name}")
        row, curve, model = evaluate(
            name, pipe, X_train, X_test, y_train, y_test, cv
        )
        rows.append(row)
        curves[name] = curve
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

    print("\nThí nghiệm đối chứng (nhãn xáo trộn)...")
    real = results[~results["Mô hình"].str.startswith("Baseline")]
    best = real.iloc[0]["Mô hình"]
    best_auc = float(real.iloc[0]["ROC-AUC"])
    control = shuffled_control(
        build_pipelines()[best], X_train, X_test, y_train, y_test
    )

    print("Tính permutation importance...")
    ref = fitted["LightGBM"]
    perm = permutation_importance(
        ref, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=SEED, n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "Biến": X_test.columns,
                "Độ quan trọng": perm.importances_mean,
                "Độ lệch": perm.importances_std,
            }
        )
        .sort_values("Độ quan trọng", ascending=False)
        .reset_index(drop=True)
    )

    control["ROC-AUC (nhãn thật)"] = best_auc

    joblib.dump(
        {
            "results": results,
            "curves": curves,
            "models": fitted,
            "control": control,
            "control_model": best,
            "importance": importance,
            "y_test": y_test,
        },
        ARTIFACTS / "results.joblib",
        compress=3,
    )
    results.to_csv(ARTIFACTS / "metrics.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(results.round(4).to_string(index=False))
    print("=" * 100)
    print(f"\nĐối chứng nhãn xáo trộn ({best}): "
          f"Accuracy={control['Accuracy']:.4f}  ROC-AUC={control['ROC-AUC']:.4f}")
    print(f"\nĐã lưu vào {ARTIFACTS}")


if __name__ == "__main__":
    main()
