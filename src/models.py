"""The set of models we compare."""
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data import build_preprocessor

SEED = 42

# (name, classifier, needs feature scaling, short note for students)
MODEL_SPECS = [
    (
        "Baseline (majority class)",
        DummyClassifier(strategy="most_frequent"),
        False,
        "Always predicts 'No disease'. The floor every model must beat.",
    ),
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=SEED),
        True,
        "A linear model that is easy to interpret: one coefficient per feature.",
    ),
    (
        "Logistic Regression (balanced)",
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        True,
        "Same model, but missing a disease case is penalised harder — higher "
        "recall, lower accuracy.",
    ),
    (
        "K-Nearest Neighbors",
        KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        True,
        "Predicts from the 25 nearest neighbours. Very sensitive to feature "
        "scale, so scaling is mandatory.",
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(max_depth=5, random_state=SEED),
        False,
        "A decision tree capped at depth 5 to limit memorisation.",
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1
        ),
        False,
        "Averages 300 independent trees (bagging) to reduce variance.",
    ),
    (
        "HistGradientBoosting",
        HistGradientBoostingClassifier(random_state=SEED),
        False,
        "Boosting: trees are built in sequence, each correcting the last.",
    ),
    (
        "LightGBM",
        LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=SEED,
            n_jobs=-1, verbose=-1,
        ),
        False,
        "Speed-optimised boosting, usually the strongest model on tabular data.",
    ),
]

MODEL_NOTES = {name: note for name, _, _, note in MODEL_SPECS}


def build_pipelines() -> dict[str, Pipeline]:
    """Attach the right preprocessor to each model to form a full Pipeline."""
    return {
        name: Pipeline(
            [("prep", build_preprocessor(scale=scale)), ("clf", clf)]
        )
        for name, clf, scale, _ in MODEL_SPECS
    }
