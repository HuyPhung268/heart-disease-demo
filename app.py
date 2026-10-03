"""Heart disease classification dashboard.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (
    FEATURE_COLS,
    NOMINAL_COLS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    TARGET,
    load_raw,
)
from src.theme import (
    VALUE_FONT, apply_theme, banner, bar_marker, kpi, labels, note, page_head,
)

ROOT = Path(__file__).resolve().parent

# Friendly labels for the binary features in the prediction form.
FIELD_LABELS = {
    "Smoking": "Smoking",
    "Family Heart Disease": "Family history of heart disease",
    "Diabetes": "Diabetes",
    "High Blood Pressure": "High blood pressure",
    "Low HDL Cholesterol": "Low HDL cholesterol",
    "High LDL Cholesterol": "High LDL cholesterol",
}

st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
P = apply_theme()


# ------------------------------------------------------------------- data load
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_raw()


def _load(name: str):
    """Load an artifact, returning (data, error). Never leaks a traceback."""
    path = ROOT / "artifacts" / name
    if not path.exists():
        return None, "missing"
    try:
        return joblib.load(path), None
    except Exception as exc:  # usually a sklearn/lightgbm version mismatch
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource
def get_artifacts():
    return _load("results.joblib")


@st.cache_resource
def get_tuning():
    return _load("tuning.joblib")


@st.cache_data
def compare_extremes() -> pd.DataFrame:
    """Every model's prediction for two patients at opposite extremes."""
    healthy = {
        "Age": 25, "Gender": "Female", "Blood Pressure": 120,
        "Cholesterol Level": 150, "Exercise Habits": "High", "Smoking": "No",
        "Family Heart Disease": "No", "Diabetes": "No", "BMI": 21.0,
        "High Blood Pressure": "No", "Low HDL Cholesterol": "No",
        "High LDL Cholesterol": "No", "Alcohol Consumption": "None",
        "Stress Level": "Low", "Sleep Hours": 8.0, "Sugar Consumption": "Low",
        "Triglyceride Level": 100, "Fasting Blood Sugar": 80,
        "CRP Level": 0.5, "Homocysteine Level": 5.0,
    }
    risky = {
        "Age": 80, "Gender": "Male", "Blood Pressure": 180,
        "Cholesterol Level": 300, "Exercise Habits": "Low", "Smoking": "Yes",
        "Family Heart Disease": "Yes", "Diabetes": "Yes", "BMI": 40.0,
        "High Blood Pressure": "Yes", "Low HDL Cholesterol": "Yes",
        "High LDL Cholesterol": "Yes", "Alcohol Consumption": "High",
        "Stress Level": "High", "Sleep Hours": 4.0, "Sugar Consumption": "High",
        "Triglyceride Level": 400, "Fasting Blood Sugar": 160,
        "CRP Level": 15.0, "Homocysteine Level": 20.0,
    }
    X = pd.DataFrame([{c: healthy[c] for c in FEATURE_COLS},
                      {c: risky[c] for c in FEATURE_COLS}])
    rows = []
    for name, model in get_artifacts()[0]["models"].items():
        p = model.predict_proba(X)[:, 1]
        rows.append({"Model": name, "Healthy": p[0],
                     "High risk": p[1], "Difference": p[1] - p[0]})
    return pd.DataFrame(rows)


df = get_data()
art, art_err = get_artifacts()
if art is None:
    if art_err == "missing":
        st.error("No training results yet. Run `python train.py` first.")
    else:
        st.error(
            "Could not load `artifacts/results.joblib`. The usual cause is a "
            "library version mismatch between training and runtime - install "
            "`requirements.txt` exactly, then re-run `python train.py`.\n\n"
            f"Details: `{art_err}`"
        )
    st.stop()

results: pd.DataFrame = art["results"]
curves: dict = art["curves"]
models: dict = art["models"]
real_models = [m for m in models if not m.startswith("Baseline")]
best = results[~results["Model"].str.startswith("Baseline")].iloc[0]
ctrl = art["control"]          # shuffled-label control experiment results

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="side-title">🫀 Heart Disease</div>'
                '<div class="side-sub">Analysis &amp; modelling dashboard</div>',
                unsafe_allow_html=True)
    rows = [
        ("Records", f"{len(df):,}"),
        ("Input features", f"{len(FEATURE_COLS)}"),
        ("Disease rate", f"{(df[TARGET] == 'Yes').mean():.1%}"),
        ("Training set", "8,000"),
        ("Test set", "2,000"),
        ("Models", f"{len(models)}"),
    ]
    st.markdown(
        "".join(f'<div class="side-row"><span>{k}</span><span>{v}</span></div>'
                for k, v in rows),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="verdict">'
        '<div class="vhead">Verdict</div>'
        f'<div class="vbig">{best["ROC-AUC"]:.3f}</div>'
        '<div class="vcap">Best ROC-AUC &mdash; level with random guessing (0.500)</div>'
        f'<div class="vrow"><span>Shuffled labels</span>'
        f'<span>{ctrl["ROC-AUC"]:.3f}</span></div>'
        f'<div class="vrow"><span>Real labels</span>'
        f'<span>{ctrl["ROC-AUC (real labels)"]:.3f}</span></div>'
        '<div class="vnote">Shuffled labels score as well as real ones &mdash; '
        'the data carries no predictive signal.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

tabs = st.tabs([
    "📊 Overview",
    "🔧 Preprocessing",
    "🤖 Model comparison",
    "⚙️ Tuning",
    "🔬 Data diagnostics",
    "🎯 Predict",
])

# ================================================================== OVERVIEW
with tabs[0]:
    page_head("Step 01", "Data overview",
              "Health indicators and cardiovascular risk factors.")

    k = st.columns(4)
    kpi(k[0], "Records", f"{len(df):,}", f"{df.shape[1]} columns", "accent")
    kpi(k[1], "Disease rate", f"{(df[TARGET] == 'Yes').mean():.1%}",
        f"{(df[TARGET] == 'Yes').sum():,} positive cases", "warn")
    kpi(k[2], "Rows with a gap", f"{df.isna().any(axis=1).mean():.1%}",
        f"{df.isna().any(axis=1).sum():,} rows")
    kpi(k[3], "Duplicate rows", f"{df.duplicated().sum()}", "none")

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.45], gap="large")

    with left:
        st.markdown("### Target distribution")
        counts = df[TARGET].value_counts()
        fig = go.Figure(go.Pie(
            labels=["No disease", "Has disease"],
            values=[counts.get("No", 0), counts.get("Yes", 0)],
            hole=0.62, sort=False,
            marker=dict(colors=[P.healthy, P.diseased], line=dict(width=0)),
            textinfo="percent", textfont=dict(size=13, color="white"),
        ))
        fig.add_annotation(text=f"<b>{len(df):,}</b><br><span style='font-size:11px'>records</span>",
                           showarrow=False, font=dict(size=20, color=P.accent))
        fig.update_layout(height=300, showlegend=True,
                          legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
        st.plotly_chart(fig, width="stretch")
        note("An <b>80/20</b> split. The data is imbalanced, so accuracy is not an "
             "appropriate evaluation metric.")

    with right:
        st.markdown("### Missing values by column")
        miss = (df.isna().mean() * 100).sort_values()
        miss = miss[miss > 0]
        fig = go.Figure(go.Bar(
            x=miss.values, y=miss.index, orientation="h",
            marker=bar_marker(P.accent),
            text=[f"{v:.2f}%" for v in miss.values],
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(height=450, xaxis_title="% missing",
                          xaxis=dict(range=[0, miss.max() * 1.18]),
                          yaxis=dict(tickfont=dict(size=10.5)))
        labels(fig)
        st.plotly_chart(fig, width="stretch")
        note("Every column is under <b>0.35%</b> missing and evenly spread: 500 cells "
             "out of 200,000. Scattered across 20 columns they touch <b>500 rows "
             "(5.0%)</b> &mdash; impute rather than drop.")

    st.markdown("### Distribution by feature")
    c1, c2 = st.columns([1, 3])
    col = c1.selectbox("Feature", FEATURE_COLS, label_visibility="collapsed")

    if col in NUMERIC_COLS:
        fig = go.Figure()
        for lbl, color in (("No", P.healthy), ("Yes", P.diseased)):
            fig.add_trace(go.Histogram(
                x=df.loc[df[TARGET] == lbl, col], nbinsx=38, opacity=0.72,
                name="No disease" if lbl == "No" else "Has disease",
                marker=dict(color=color, line=dict(width=0)),
            ))
        fig.update_layout(barmode="overlay", height=330, xaxis_title=col,
                          yaxis_title="Records")
    else:
        tmp = df.groupby([col, TARGET]).size().reset_index(name="n")
        fig = go.Figure()
        for lbl, color in (("No", P.healthy), ("Yes", P.diseased)):
            sub = tmp[tmp[TARGET] == lbl]
            fig.add_trace(go.Bar(
                x=sub[col], y=sub["n"], name="No disease" if lbl == "No" else "Has disease",
                marker=bar_marker(color), text=[f"{v:,}" for v in sub["n"]],
            ))
        fig.update_layout(barmode="group", height=360, xaxis_title=col,
                          yaxis_title="Records")
        labels(fig)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")

    if col in NUMERIC_COLS:
        grp = pd.qcut(df[col], 4, duplicates="drop")
    else:
        grp = df[col].fillna("(missing)")
    rate = df.groupby(grp, observed=True)[TARGET].apply(
        lambda s: (s == "Yes").mean() * 100).round(1)
    note("Disease rate per group: "
         + " &middot; ".join(f"<b>{k}</b> {v}%" for k, v in rate.items())
         + " &mdash; the dataset-wide base rate is <b>20.0%</b>.")

    with st.expander("View raw data"):
        st.dataframe(df.head(50), width="stretch", height=320)

# ============================================================= PREPROCESSING
with tabs[1]:
    page_head("Step 02", "Preprocessing",
              "Turn 20 raw columns into a numeric matrix, with no data leakage.")

    k = st.columns(4)
    kpi(k[0], "Numeric features", f"{len(NUMERIC_COLS)}", "median impute + scale")
    kpi(k[1], "Ordinal features", f"{len(ORDINAL_COLS)}", "OrdinalEncoder")
    kpi(k[2], "Nominal features", f"{len(NOMINAL_COLS)}", "OneHotEncoder")
    kpi(k[3], "Output features", f"{len(FEATURE_COLS)}", "complete numeric matrix", "accent")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    banner(
        "<b>A reading gotcha.</b> The <code>Alcohol Consumption</code> column has a "
        "<code>None</code> level meaning <i>does not drink</i>. Pandas treats the "
        "string <code>\"None\"</code> as missing by default, making the column look "
        "25.9% empty instead of its real 0.32%.", "info")

    a, b = st.columns(2, gap="large")
    a.markdown("**Default read**")
    a.code("pd.read_csv(path)\n# -> 2,586 NaN  (25.9%)", language="python")
    b.markdown("**Correct read**")
    b.code("pd.read_csv(\n    path,\n    keep_default_na=False,\n    na_values=[''],\n)"
           "\n# -> 32 NaN  (0.32%)", language="python")

    st.markdown("### Handling rules by feature group")
    spec = pd.DataFrame([
        {"Group": "Numeric", "Columns": len(NUMERIC_COLS),
         "Impute": "Median", "Encode": "-", "Scale": "StandardScaler",
         "Examples": ", ".join(NUMERIC_COLS[:3])},
        {"Group": "Ordinal", "Columns": len(ORDINAL_COLS),
         "Impute": "Mode", "Encode": "OrdinalEncoder", "Scale": "-",
         "Examples": ", ".join(list(ORDINAL_COLS)[:2])},
        {"Group": "Nominal", "Columns": len(NOMINAL_COLS),
         "Impute": "Mode", "Encode": "OneHotEncoder", "Scale": "-",
         "Examples": ", ".join(NOMINAL_COLS[:2])},
    ])
    st.dataframe(spec, width="stretch", hide_index=True)
    note("Ordinal features keep the <code>Low &lt; Medium &lt; High</code> relation, so "
         "they map to increasing integers. Nominal features have no order, so they "
         "must be one-hot encoded.")

    st.markdown("### The complete pipeline")
    st.code(
        """from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("prep", build_preprocessor(scale=True)),   # impute -> encode -> scale
    ("clf",  LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)    # statistics learned from X_train only
pipe.predict(X_test)          # applied unchanged to X_test""",
        language="python")
    note("Wrapping preprocessing in a <code>Pipeline</code> guarantees the median and "
         "the scaling parameters are computed on each fold's training part only, "
         "which is what prevents data leakage.")

# ========================================================= MODEL COMPARISON
with tabs[2]:
    page_head("Step 03", "Model comparison",
              "Stratified 80/20 split &middot; 5-fold cross-validation &middot; "
              "scored on 2,000 held-out test records.")

    base_acc = results.loc[results["Model"].str.startswith("Baseline"),
                           "Accuracy"].iloc[0]
    k = st.columns(4)
    kpi(k[0], "Best model", best["Model"].split(" (")[0],
        "by ROC-AUC", "accent")
    kpi(k[1], "ROC-AUC", f"{best['ROC-AUC']:.3f}",
        f"{best['ROC-AUC'] - 0.5:+.3f} vs random", "alert")
    kpi(k[2], "Accuracy", f"{best['Accuracy']:.1%}",
        f"{(best['Accuracy'] - base_acc) * 100:+.2f} pts vs baseline")
    kpi(k[3], "Recall", f"{best['Recall']:.1%}", "share of disease cases caught", "alert")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.markdown("### Full metrics table")
    num_cols = [c for c in results.columns if c != "Model"]
    st.dataframe(
        results.style
        .format({c: "{:.3f}" for c in num_cols})
        .background_gradient(subset=["ROC-AUC"], cmap="RdYlGn", vmin=0.42, vmax=0.58)
        .background_gradient(subset=["Accuracy"], cmap="Blues", vmin=0.4, vmax=1.0),
        width="stretch", hide_index=True,
    )
    n_tied = int((results["Accuracy"].round(3) == round(base_acc, 3)).sum())
    note(f"<b>{n_tied}</b> models score exactly <b>{base_acc:.3f}</b> accuracy, matching "
         "the always-predict-majority baseline. Every ROC-AUC sits near <b>0.50</b>.")

    st.markdown("### ROC curves")
    pick = st.multiselect("Models", list(curves),
                          default=["LightGBM", "Random Forest", "Logistic Regression"],
                          label_visibility="collapsed")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random (0.500)",
                             line=dict(dash="dash", color=P.muted, width=2)))
    for i, name in enumerate(pick):
        fpr, tpr = curves[name]["roc"]
        auc = results.loc[results["Model"] == name, "ROC-AUC"].iloc[0]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({auc:.3f})",
                                 line=dict(width=2.2, color=P.colorway[i % len(P.colorway)])))
    fig.update_layout(height=460, xaxis_title="False positive rate",
                      yaxis_title="True positive rate",
                      legend=dict(y=0.04, x=0.98, xanchor="right"))
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Confusion matrix")
    cm_pick = st.selectbox("Model", list(curves),
                           index=list(curves).index("LightGBM"),
                           label_visibility="collapsed")
    cm = curves[cm_pick]["cm"]
    tn, fp, fn, tp = cm.ravel()
    m1, m2 = st.columns([1.1, 1], gap="large")
    with m1:
        fig = go.Figure(go.Heatmap(
            z=cm, x=["No disease", "Has disease"], y=["No disease", "Has disease"],
            colorscale=[[0, "#EAF2FC"], [1, "#8FBBEE"]], showscale=False,
            text=cm, texttemplate="%{text}",
            textfont=dict(size=20, family="Inter", color="#12385C"),
            hovertemplate="Actual %{y}<br>Predicted %{x}<br><b>%{z}</b><extra></extra>",
        ))
        fig.update_layout(height=320, xaxis_title="Predicted", yaxis_title="Actual",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with m2:
        g = st.columns(2)
        kpi(g[0], "Caught disease", f"{tp}", "True Positive")
        kpi(g[1], "Missed disease", f"{fn}", "False Negative", "alert")
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        g = st.columns(2)
        kpi(g[0], "False alarm", f"{fp}", "False Positive", "warn")
        kpi(g[1], "Ruled out correctly", f"{tn}", "True Negative")
        if tp + fn:
            note(f"The model misses <b>{fn}/{tp + fn}</b> disease cases "
                 f"({fn / (tp + fn):.0%}). In medical screening a miss costs far more "
                 "than a false alarm.")

    st.markdown("### Feature importance")
    imp = art["importance"].head(12).sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=imp["Importance"], y=imp["Feature"], orientation="h",
        error_x=dict(array=imp["Std"], color=P.muted, thickness=1.2, width=3),
        marker=bar_marker([P.orange if v < 0 else P.accent
                           for v in imp["Importance"]]),
        text=[f"{v:+.4f}" for v in imp["Importance"]],
    ))
    fig.add_vline(x=0, line_color=P.axis, line_width=1.5)
    fig.update_layout(height=450, xaxis_title="ROC-AUC drop when the feature is shuffled",
                      yaxis=dict(tickfont=dict(size=10.5)))
    labels(fig)
    st.plotly_chart(fig, width="stretch")
    note("Permutation importance on LightGBM. Every value sits near 0 and the error "
         "bars cross zero &mdash; shuffling any feature leaves the model no worse.")

# ==================================================================== TUNING
def render_tuning():
    page_head("Step 04", "Hyperparameter tuning",
              "GridSearchCV sweeps the whole parameter grid, scoring each combo "
              "with 5-fold cross-validation.")

    tun, tun_err = get_tuning()
    if tun is None:
        if tun_err == "missing":
            st.warning("No tuning results yet. Run `python tune.py` first.")
        else:
            st.warning(f"Could not load `artifacts/tuning.joblib`: `{tun_err}`")
        return

    tuning: pd.DataFrame = tun["tuning"]
    searches: dict = tun["searches"]

    worst = tuning.loc[tuning["Test change"].idxmin()]
    k = st.columns(4)
    kpi(k[0], "Models tuned", f"{len(tuning)}", "4 parameter grids")
    kpi(k[1], "Total combos", f"{int(tuning['Combos'].sum())}",
        f"{int(tuning['Combos'].sum()) * 5} fits", "accent")
    kpi(k[2], "Best change", f"{tuning['Test change'].max():+.4f}",
        "test ROC-AUC")
    kpi(k[3], "Worst change", f"{worst['Test change']:+.4f}",
        worst["Model"], "alert")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.markdown("### Default versus tuned")
    disp = tuning.drop(columns=["Best params"])
    st.dataframe(
        disp.style.format(
            {c: "{:.4f}" for c in disp.columns if c not in ("Model", "Combos")}
        ).background_gradient(subset=["Test change"], cmap="RdYlGn",
                              vmin=-0.03, vmax=0.03),
        width="stretch", hide_index=True,
    )
    banner(
        f"<b>{worst['Model']}</b> improved its cross-validation score to "
        f"{worst['CV ROC-AUC (best)']:.4f}, yet its test score <i>fell</i> from "
        f"{worst['Test ROC-AUC (default)']:.4f} to "
        f"{worst['Test ROC-AUC (tuned)']:.4f}. Picking the highest-CV combo out of "
        "dozens selects the one that happens to suit that particular fold split, "
        "not the one that generalises best.", "bad")

    st.markdown("### Change versus the noise band")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tuning["Model"], y=tuning["Test change"], name="Test change",
        marker=bar_marker([P.orange if v < 0 else P.accent
                           for v in tuning["Test change"]]),
        width=0.45, text=[f"{v:+.4f}" for v in tuning["Test change"]],
        textposition="inside", insidetextanchor="middle", cliponaxis=False,
        textfont=dict(family="Inter", size=10.5, color="white"),
    ))
    for sign, show in ((1, True), (-1, False)):
        fig.add_trace(go.Scatter(
            x=tuning["Model"], y=sign * tuning["CV std of best combo"],
            mode="markers", name="Fold-to-fold noise band (±1σ)",
            showlegend=show,
            marker=dict(size=13, symbol="diamond-open", color=P.muted,
                        line=dict(width=2)),
        ))
    fig.add_hline(y=0, line_color=P.axis, line_width=1.5)
    fig.update_layout(height=430, yaxis_title="ROC-AUC change",
                      margin=dict(l=90, r=30, t=30, b=50),
                      yaxis=dict(range=[-0.042, 0.042]),
                      legend=dict(orientation="h", y=1.16, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")
    note("Every model's change is the same size as, or smaller than, the natural "
         "fold-to-fold variation. To claim tuning helped, the gain would have to be "
         "clearly larger than this band.")

    st.markdown("### Grid detail")
    mname = st.selectbox("Model", list(searches), key="tune_model",
                         label_visibility="collapsed")
    cvr = searches[mname]["cv_results"].copy()
    cvr["Combo"] = cvr["params"].apply(
        lambda d: ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in d.items()))
    top = cvr.sort_values("mean_test_score", ascending=False).head(14)[::-1]

    fig = go.Figure(go.Bar(
        x=top["mean_test_score"], y=top["Combo"], orientation="h",
        error_x=dict(array=top["std_test_score"], color=P.muted, thickness=1.2, width=3),
        marker=bar_marker(P.accent),
        text=[f"{v:.4f}" for v in top["mean_test_score"]],
        textposition="inside", insidetextanchor="start",
        textfont=dict(family="Inter", size=10.5, color="white"),
        cliponaxis=False,
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=P.critical, line_width=1.6,
                  annotation_text="random", annotation_position="top")
    fig.update_layout(height=540, xaxis_title="CV ROC-AUC",
                      xaxis=dict(range=[0, 0.60]),
                      margin=dict(l=24, r=60),
                      yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig, width="stretch")

    gap = (cvr["mean_train_score"] - cvr["mean_test_score"]).max()
    note(f"The 14 leading combos with fold-to-fold error bars. The bars overlap and "
         f"all cross the 0.5 line. The largest train-validation gap in the "
         f"<b>{mname}</b> grid is <b>{gap:.3f}</b>.")

    st.markdown("### Best parameters")
    st.dataframe(tuning[["Model", "Best params"]],
                 width="stretch", hide_index=True)


with tabs[3]:
    render_tuning()

# ========================================================== DATA DIAGNOSTICS
with tabs[4]:
    page_head("Step 05", "Data diagnostics",
              "Three independent tests of whether the data carries any predictive "
              "signal at all.")

    y = (df[TARGET] == "Yes").astype(int)
    corr = df[NUMERIC_COLS].corrwith(y)

    k = st.columns(4)
    kpi(k[0], "Baseline accuracy", f"{base_acc:.1%}", "always predicts the majority")
    kpi(k[1], "Best accuracy", f"{best['Accuracy']:.1%}",
        f"{(best['Accuracy'] - base_acc) * 100:+.2f} pts")
    kpi(k[2], "ROC-AUC, real labels", f"{ctrl['ROC-AUC (real labels)']:.4f}",
        art["control_model"], "alert")
    kpi(k[3], "ROC-AUC, shuffled", f"{ctrl['ROC-AUC']:.4f}",
        f"{ctrl['ROC-AUC'] - ctrl['ROC-AUC (real labels)']:+.4f} vs real labels",
        "alert")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    st.markdown("### Test 1 &mdash; accuracy against the baseline")
    comp = results.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp["Model"], y=comp["Accuracy"], name="Accuracy",
                         marker=bar_marker(P.accent), width=0.38, offset=-0.4,
                         text=[f"{v:.3f}" for v in comp["Accuracy"]]))
    fig.add_trace(go.Bar(x=comp["Model"], y=comp["ROC-AUC"], name="ROC-AUC",
                         marker=bar_marker(P.orange), width=0.38, offset=0.02,
                         text=[f"{v:.3f}" for v in comp["ROC-AUC"]]))
    fig.add_hline(y=0.8, line_dash="dash", line_color=P.accent, line_width=1.6,
                  annotation_text="baseline accuracy 0.80",
                  annotation_position="top left",
                  annotation_font=dict(size=10.5, color=P.accent))
    fig.add_hline(y=0.5, line_dash="dash", line_color=P.orange, line_width=1.6,
                  annotation_text="random 0.50",
                  annotation_position="bottom left",
                  annotation_font=dict(size=10.5, color=P.orange))
    labels(fig)
    fig.update_layout(height=440, yaxis_range=[0, 1.12], yaxis_title="Score",
                      xaxis=dict(tickangle=-25, tickfont=dict(size=10)),
                      legend=dict(orientation="h", y=1.14, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")
    note("Every model's accuracy hugs the baseline's 0.80 line, while every ROC-AUC "
         "hugs the 0.50 line of random guessing.")

    st.markdown("### Test 2 &mdash; training on shuffled labels")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Real labels", "Randomly shuffled labels"],
        y=[ctrl["ROC-AUC (real labels)"], ctrl["ROC-AUC"]],
        marker=bar_marker([P.accent, P.orange]), width=0.42,
        text=[f"{ctrl['ROC-AUC (real labels)']:.4f}", f"{ctrl['ROC-AUC']:.4f}"],
        textposition="outside", textfont=dict(size=15),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=P.muted, line_width=1.6,
                  annotation_text="random")
    fig.update_layout(height=380, yaxis_range=[0.40, 0.58],
                      yaxis_title="ROC-AUC on the test set")
    st.plotly_chart(fig, width="stretch")
    note(f"<b>{art['control_model']}</b> was retrained after every training label was "
         "shuffled. If the data carried signal, the score would drop sharply. Here "
         "shuffled labels score level with, or above, the real ones.")

    st.markdown("### Test 3 &mdash; feature-label correlation")
    corr = corr.sort_values()
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker=bar_marker([P.orange if v < 0 else P.accent for v in corr.values]),
        text=[f"{v:+.4f}" for v in corr.values],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    for v in (0.1, -0.1):
        fig.add_vline(x=v, line_dash="dot", line_color=P.critical, line_width=1.5)
    fig.add_vline(x=0, line_color=P.axis, line_width=1.5)
    labels(fig)
    fig.update_layout(height=420, xaxis_range=[-0.26, 0.26],
                      xaxis_title="Correlation with the disease label",
                      yaxis=dict(tickfont=dict(size=10.5)))
    st.plotly_chart(fig, width="stretch")
    note("Every coefficient falls within <b>±0.02</b>; the dashed red lines mark the "
         "<b>|r| = 0.1</b> threshold normally used for a weak association. Smokers "
         "have a 20.1% disease rate against 19.9% for non-smokers.")

    banner(
        "<b>Conclusion.</b> Three independent tests agree: the input features carry "
        "no predictive information about the label. This is randomly generated "
        "synthetic data &mdash; fine for demonstrating the workflow, but not a basis "
        "for any medical conclusion.", "bad")

# =================================================================== PREDICT
with tabs[5]:
    page_head("Step 06", "Predict for one patient",
              "Enter the values and pick a model to see the predicted probability.")

    model_name = st.selectbox("Model", real_models,
                              index=real_models.index("LightGBM"))

    with st.form("predict"):
        g1, g2, g3 = st.columns(3, gap="large")
        vals = {}
        with g1:
            st.markdown("**Basics**")
            vals["Age"] = st.slider("Age", 18, 80, 50)
            vals["Gender"] = st.radio("Gender", ["Male", "Female"], horizontal=True)
            vals["BMI"] = st.slider("BMI", 18.0, 40.0, 29.0, 0.1)
            vals["Sleep Hours"] = st.slider("Sleep hours per night", 4.0, 10.0, 7.0, 0.1)
            vals["Exercise Habits"] = st.select_slider(
                "Exercise habits", ["Low", "Medium", "High"], "Medium")
        with g2:
            st.markdown("**Lab results**")
            vals["Blood Pressure"] = st.slider("Systolic blood pressure", 120, 180, 150)
            vals["Cholesterol Level"] = st.slider("Total cholesterol", 150, 300, 225)
            vals["Triglyceride Level"] = st.slider("Triglycerides", 100, 400, 250)
            vals["Fasting Blood Sugar"] = st.slider("Fasting blood sugar", 80, 160, 120)
            vals["CRP Level"] = st.slider("CRP", 0.0, 15.0, 7.5, 0.1)
            vals["Homocysteine Level"] = st.slider("Homocysteine", 5.0, 20.0, 12.5, 0.1)
        with g3:
            st.markdown("**History and lifestyle**")
            for c in NOMINAL_COLS:
                if c == "Gender":
                    continue
                vals[c] = st.radio(FIELD_LABELS.get(c, c), ["No", "Yes"],
                                   horizontal=True, key=f"r_{c}")
            vals["Alcohol Consumption"] = st.select_slider(
                "Alcohol", ["None", "Low", "Medium", "High"], "Low")
            vals["Stress Level"] = st.select_slider(
                "Stress level", ["Low", "Medium", "High"], "Medium")
            vals["Sugar Consumption"] = st.select_slider(
                "Sugar intake", ["Low", "Medium", "High"], "Medium")

        submitted = st.form_submit_button("Predict", width="stretch", type="primary")

    if submitted:
        row = pd.DataFrame([{c: vals[c] for c in FEATURE_COLS}])
        model = models[model_name]
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(model.predict(row)[0])

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        r1, r2 = st.columns([1.15, 1], gap="large")
        with r1:
            gauge_color = P.critical if pred else P.accent
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                number={"suffix": "%", "font": {"size": 44, "color": gauge_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickcolor": P.axis, "tickfont": {"size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.7},
                    "bgcolor": P.soft, "borderwidth": 0,
                    "steps": [
                        {"range": [0, 20], "color": P.band_good},
                        {"range": [20, 50], "color": P.band_warn},
                        {"range": [50, 100], "color": P.band_bad},
                    ],
                    "threshold": {"line": {"color": P.axis, "width": 3},
                                  "thickness": 0.82, "value": 50},
                },
            ))
            fig.update_layout(height=290, margin=dict(t=25, b=10, l=30, r=30))
            st.plotly_chart(fig, width="stretch")
        with r2:
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
            kk = st.columns(2)
            kpi(kk[0], "Verdict", "At risk" if pred else "Not at risk",
                "50% threshold", "alert" if pred else "accent")
            kpi(kk[1], "Probability", f"{proba:.1%}", "dataset base rate 20%")
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            note(f"<b>{model_name}</b> &middot; test-set ROC-AUC "
                 f"<b>{results.loc[results['Model'] == model_name, 'ROC-AUC'].iloc[0]:.3f}</b>. "
                 "See the Data diagnostics tab for how much to trust this number.")

    st.markdown("### Two extreme profiles side by side")
    cmp_df = compare_extremes()
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cmp_df["Model"], y=cmp_df["Healthy"],
                             name="Healthy profile", marker=bar_marker(P.accent),
                             text=[f"{v:.0%}" for v in cmp_df["Healthy"]]))
        fig.add_trace(go.Bar(x=cmp_df["Model"], y=cmp_df["High risk"],
                             name="High-risk profile", marker=bar_marker(P.orange),
                             text=[f"{v:.0%}" for v in cmp_df["High risk"]]))
        labels(fig)
        fig.update_layout(barmode="group", height=430,
                          yaxis_title="Predicted probability", yaxis_tickformat=".0%",
                          xaxis=dict(tickangle=-25, tickfont=dict(size=9.5)),
                          legend=dict(orientation="h", y=1.14, x=1, xanchor="right"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(
            cmp_df.style.format({"Healthy": "{:.1%}", "High risk": "{:.1%}",
                                 "Difference": "{:+.1%}"})
            .background_gradient(subset=["Difference"], cmap="Oranges"),
            width="stretch", hide_index=True, height=330,
        )
    note("Healthy profile: age 25, BMI 21, non-smoker, every reading at its best. "
         "High-risk profile: age 80, BMI 40, smoker, diabetic, every reading at its "
         "worst. Both sit outside the training distribution, so the model is "
         "extrapolating.")
