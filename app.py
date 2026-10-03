"""Dashboard phân loại bệnh tim.

Chạy:  streamlit run app.py
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

# Nhãn tiếng Việt cho các biến nhị phân trong form dự đoán.
NHAN_VI = {
    "Smoking": "Hút thuốc",
    "Family Heart Disease": "Tiền sử gia đình mắc bệnh tim",
    "Diabetes": "Tiểu đường",
    "High Blood Pressure": "Cao huyết áp",
    "Low HDL Cholesterol": "HDL thấp",
    "High LDL Cholesterol": "LDL cao",
}

st.set_page_config(
    page_title="Phân loại bệnh tim",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
P = apply_theme()


# ----------------------------------------------------------------- nạp dữ liệu
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_raw()


def _load(name: str):
    """Nạp artifact, trả về (dữ liệu, lỗi). Không ném traceback ra giao diện."""
    path = ROOT / "artifacts" / name
    if not path.exists():
        return None, "missing"
    try:
        return joblib.load(path), None
    except Exception as exc:  # thường do lệch phiên bản sklearn/lightgbm
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource
def get_artifacts():
    return _load("results.joblib")


@st.cache_resource
def get_tuning():
    return _load("tuning.joblib")


@st.cache_data
def compare_extremes() -> pd.DataFrame:
    """Dự đoán của mọi mô hình cho hai bệnh nhân ở hai thái cực."""
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
        rows.append({"Mô hình": name, "Khoẻ mạnh": p[0],
                     "Nguy cơ cao": p[1], "Chênh lệch": p[1] - p[0]})
    return pd.DataFrame(rows)


df = get_data()
art, art_err = get_artifacts()
if art is None:
    if art_err == "missing":
        st.error("Chưa có kết quả huấn luyện. Chạy `python train.py` trước.")
    else:
        st.error(
            "Không nạp được `artifacts/results.joblib`. Nguyên nhân thường gặp là "
            "phiên bản thư viện lúc chạy khác với lúc huấn luyện — hãy cài đúng "
            "`requirements.txt` rồi chạy lại `python train.py`.\n\n"
            f"Chi tiết: `{art_err}`"
        )
    st.stop()

results: pd.DataFrame = art["results"]
curves: dict = art["curves"]
models: dict = art["models"]
real_models = [m for m in models if not m.startswith("Baseline")]
best = results[~results["Mô hình"].str.startswith("Baseline")].iloc[0]
ctrl = art["control"]          # kết quả thí nghiệm đối chứng nhãn xáo trộn

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="side-title">🫀 Phân loại bệnh tim</div>'
                '<div class="side-sub">Dashboard phân tích & mô hình</div>',
                unsafe_allow_html=True)
    rows = [
        ("Số bản ghi", f"{len(df):,}"),
        ("Biến đầu vào", f"{len(FEATURE_COLS)}"),
        ("Tỷ lệ mắc bệnh", f"{(df[TARGET] == 'Yes').mean():.1%}"),
        ("Tập huấn luyện", "8,000"),
        ("Tập kiểm tra", "2,000"),
        ("Số mô hình", f"{len(models)}"),
    ]
    st.markdown(
        "".join(f'<div class="side-row"><span>{k}</span><span>{v}</span></div>'
                for k, v in rows),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="verdict">'
        '<div class="vhead">Kết luận</div>'
        f'<div class="vbig">{best["ROC-AUC"]:.3f}</div>'
        '<div class="vcap">ROC-AUC tốt nhất — ngang đoán ngẫu nhiên (0.500)</div>'
        f'<div class="vrow"><span>Nhãn xáo trộn</span>'
        f'<span>{ctrl["ROC-AUC"]:.3f}</span></div>'
        f'<div class="vrow"><span>Nhãn thật</span>'
        f'<span>{ctrl["ROC-AUC (nhãn thật)"]:.3f}</span></div>'
        '<div class="vnote">Nhãn xáo trộn cho kết quả ngang nhãn thật — '
        'dữ liệu không chứa tín hiệu dự đoán.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

tabs = st.tabs([
    "📊 Tổng quan",
    "🔧 Tiền xử lý",
    "🤖 So sánh mô hình",
    "⚙️ Tinh chỉnh",
    "🔬 Chẩn đoán dữ liệu",
    "🎯 Dự đoán",
])

# ============================================================== TỔNG QUAN
with tabs[0]:
    page_head("Bước 01", "Tổng quan dữ liệu",
              "Bộ dữ liệu chỉ số sức khoẻ và yếu tố nguy cơ tim mạch.")

    k = st.columns(4)
    kpi(k[0], "Bản ghi", f"{len(df):,}", f"{df.shape[1]} cột", "accent")
    kpi(k[1], "Tỷ lệ mắc bệnh", f"{(df[TARGET] == 'Yes').mean():.1%}",
        f"{(df[TARGET] == 'Yes').sum():,} ca bệnh", "warn")
    kpi(k[2], "Dòng có ô thiếu", f"{df.isna().any(axis=1).mean():.1%}",
        f"{df.isna().any(axis=1).sum():,} dòng")
    kpi(k[3], "Bản ghi trùng", f"{df.duplicated().sum()}", "không có")

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.45], gap="large")

    with left:
        st.markdown("### Phân bố biến mục tiêu")
        counts = df[TARGET].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Không bệnh", "Có bệnh"],
            values=[counts.get("No", 0), counts.get("Yes", 0)],
            hole=0.62, sort=False,
            marker=dict(colors=[P.healthy, P.diseased], line=dict(width=0)),
            textinfo="percent", textfont=dict(size=13, color="white"),
        ))
        fig.add_annotation(text=f"<b>{len(df):,}</b><br><span style='font-size:11px'>bản ghi</span>",
                           showarrow=False, font=dict(size=20, color=P.accent))
        fig.update_layout(height=300, showlegend=True,
                          legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
        st.plotly_chart(fig, width="stretch")
        note("Tỷ lệ <b>80/20</b>. Dữ liệu mất cân bằng nên accuracy không phải "
             "chỉ số đánh giá phù hợp.")

    with right:
        st.markdown("### Tỷ lệ giá trị thiếu theo cột")
        miss = (df.isna().mean() * 100).sort_values()
        miss = miss[miss > 0]
        fig = go.Figure(go.Bar(
            x=miss.values, y=miss.index, orientation="h",
            marker=bar_marker(P.accent),
            text=[f"{v:.2f}%" for v in miss.values],
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(height=450, xaxis_title="% giá trị thiếu",
                          xaxis=dict(range=[0, miss.max() * 1.18]),
                          yaxis=dict(tickfont=dict(size=10.5)))
        labels(fig)
        st.plotly_chart(fig, width="stretch")
        note("Mọi cột thiếu dưới <b>0.35%</b> và phân bố đều, tổng cộng 500 ô trên "
             "200,000 ô. Rải trên 20 cột nên ảnh hưởng <b>500 dòng (5.0%)</b> — xử lý bằng điền khuyết thay vì xoá dòng.")

    st.markdown("### Phân phối theo từng biến")
    c1, c2 = st.columns([1, 3])
    col = c1.selectbox("Biến", FEATURE_COLS, label_visibility="collapsed")

    if col in NUMERIC_COLS:
        fig = go.Figure()
        for lbl, color in (("No", P.healthy), ("Yes", P.diseased)):
            fig.add_trace(go.Histogram(
                x=df.loc[df[TARGET] == lbl, col], nbinsx=38, opacity=0.72,
                name="Không bệnh" if lbl == "No" else "Có bệnh",
                marker=dict(color=color, line=dict(width=0)),
            ))
        fig.update_layout(barmode="overlay", height=330, xaxis_title=col,
                          yaxis_title="Số bản ghi")
    else:
        tmp = df.groupby([col, TARGET]).size().reset_index(name="n")
        fig = go.Figure()
        for lbl, color in (("No", P.healthy), ("Yes", P.diseased)):
            sub = tmp[tmp[TARGET] == lbl]
            fig.add_trace(go.Bar(
                x=sub[col], y=sub["n"], name="Không bệnh" if lbl == "No" else "Có bệnh",
                marker=bar_marker(color), text=[f"{v:,}" for v in sub["n"]],
            ))
        fig.update_layout(barmode="group", height=360, xaxis_title=col,
                          yaxis_title="Số bản ghi")
        labels(fig)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")

    if col in NUMERIC_COLS:
        grp = pd.qcut(df[col], 4, duplicates="drop")
    else:
        grp = df[col].fillna("(thiếu)")
    rate = df.groupby(grp, observed=True)[TARGET].apply(
        lambda s: (s == "Yes").mean() * 100).round(1)
    note("Tỷ lệ mắc bệnh theo nhóm: "
         + " · ".join(f"<b>{k}</b> {v}%" for k, v in rate.items())
         + " — mức nền toàn bộ dữ liệu là <b>20.0%</b>.")

    with st.expander("Xem dữ liệu thô"):
        st.dataframe(df.head(50), width="stretch", height=320)

# ============================================================= TIỀN XỬ LÝ
with tabs[1]:
    page_head("Bước 02", "Tiền xử lý",
              "Chuyển 20 cột thô thành ma trận số, không rò rỉ dữ liệu.")

    k = st.columns(4)
    kpi(k[0], "Biến liên tục", f"{len(NUMERIC_COLS)}", "điền trung vị + chuẩn hoá")
    kpi(k[1], "Biến thứ bậc", f"{len(ORDINAL_COLS)}", "OrdinalEncoder")
    kpi(k[2], "Biến danh mục", f"{len(NOMINAL_COLS)}", "OneHotEncoder")
    kpi(k[3], "Đặc trưng đầu ra", f"{len(FEATURE_COLS)}", "ma trận số đầy đủ", "accent")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    banner(
        "<b>Lưu ý khi đọc dữ liệu.</b> Cột <code>Alcohol Consumption</code> có mức "
        "<code>None</code> nghĩa là <i>không uống rượu</i>. Pandas mặc định coi chuỗi "
        "<code>\"None\"</code> là giá trị thiếu, khiến cột này trông như thiếu 25.9% "
        "thay vì 0.32% thật sự.", "info")

    a, b = st.columns(2, gap="large")
    a.markdown("**Đọc mặc định**")
    a.code("pd.read_csv(path)\n# -> 2,586 NaN  (25.9%)", language="python")
    b.markdown("**Đọc đúng**")
    b.code("pd.read_csv(\n    path,\n    keep_default_na=False,\n    na_values=[''],\n)"
           "\n# -> 32 NaN  (0.32%)", language="python")

    st.markdown("### Quy tắc xử lý theo nhóm biến")
    spec = pd.DataFrame([
        {"Nhóm": "Liên tục", "Số cột": len(NUMERIC_COLS),
         "Điền khuyết": "Trung vị", "Mã hoá": "—", "Chuẩn hoá": "StandardScaler",
         "Ví dụ": ", ".join(NUMERIC_COLS[:3])},
        {"Nhóm": "Thứ bậc", "Số cột": len(ORDINAL_COLS),
         "Điền khuyết": "Mốt", "Mã hoá": "OrdinalEncoder", "Chuẩn hoá": "—",
         "Ví dụ": ", ".join(list(ORDINAL_COLS)[:2])},
        {"Nhóm": "Danh mục", "Số cột": len(NOMINAL_COLS),
         "Điền khuyết": "Mốt", "Mã hoá": "OneHotEncoder", "Chuẩn hoá": "—",
         "Ví dụ": ", ".join(NOMINAL_COLS[:2])},
    ])
    st.dataframe(spec, width="stretch", hide_index=True)
    note("Biến thứ bậc giữ quan hệ <code>Low &lt; Medium &lt; High</code> nên mã hoá "
         "thành số tăng dần. Biến danh mục không có thứ tự nên phải one-hot.")

    st.markdown("### Pipeline hoàn chỉnh")
    st.code(
        """from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("prep", build_preprocessor(scale=True)),   # điền khuyết → mã hoá → chuẩn hoá
    ("clf",  LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)    # thống kê chỉ học từ X_train
pipe.predict(X_test)          # áp dụng nguyên vẹn lên X_test""",
        language="python")
    note("Bọc tiền xử lý trong <code>Pipeline</code> đảm bảo trung vị và tham số chuẩn "
         "hoá chỉ tính trên tập train của từng fold, tránh rò rỉ dữ liệu.")

# ========================================================= SO SÁNH MÔ HÌNH
with tabs[2]:
    page_head("Bước 03", "So sánh mô hình",
              "Chia 80/20 phân tầng · Cross-validation 5-fold · Đánh giá trên "
              "2,000 bản ghi tập kiểm tra.")

    base_acc = results.loc[results["Mô hình"].str.startswith("Baseline"),
                           "Accuracy"].iloc[0]
    k = st.columns(4)
    kpi(k[0], "Mô hình tốt nhất", best["Mô hình"].split(" (")[0],
        f"theo ROC-AUC", "accent")
    kpi(k[1], "ROC-AUC", f"{best['ROC-AUC']:.3f}",
        f"{best['ROC-AUC'] - 0.5:+.3f} so với ngẫu nhiên", "alert")
    kpi(k[2], "Accuracy", f"{best['Accuracy']:.1%}",
        f"{(best['Accuracy'] - base_acc) * 100:+.2f} điểm so với baseline")
    kpi(k[3], "Recall", f"{best['Recall']:.1%}", "tỷ lệ bắt được ca bệnh", "alert")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.markdown("### Bảng chỉ số đầy đủ")
    num_cols = [c for c in results.columns if c != "Mô hình"]
    st.dataframe(
        results.style
        .format({c: "{:.3f}" for c in num_cols})
        .background_gradient(subset=["ROC-AUC"], cmap="RdYlGn", vmin=0.42, vmax=0.58)
        .background_gradient(subset=["Accuracy"], cmap="Blues", vmin=0.4, vmax=1.0),
        width="stretch", hide_index=True,
    )
    n_tied = int((results["Accuracy"].round(3) == round(base_acc, 3)).sum())
    note(f"<b>{n_tied}</b> mô hình đạt accuracy đúng <b>{base_acc:.3f}</b>, bằng mức "
         "baseline luôn đoán lớp đa số. ROC-AUC của tất cả đều quanh <b>0.50</b>.")

    st.markdown("### Đường cong ROC")
    pick = st.multiselect("Mô hình", list(curves),
                          default=["LightGBM", "Random Forest", "Logistic Regression"],
                          label_visibility="collapsed")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Ngẫu nhiên (0.500)",
                             line=dict(dash="dash", color=P.muted, width=2)))
    for i, name in enumerate(pick):
        fpr, tpr = curves[name]["roc"]
        auc = results.loc[results["Mô hình"] == name, "ROC-AUC"].iloc[0]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({auc:.3f})",
                                 line=dict(width=2.2, color=P.colorway[i % len(P.colorway)])))
    fig.update_layout(height=460, xaxis_title="Tỷ lệ dương tính giả",
                      yaxis_title="Tỷ lệ dương tính thật",
                      legend=dict(y=0.04, x=0.98, xanchor="right"))
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Ma trận nhầm lẫn")
    cm_pick = st.selectbox("Mô hình", list(curves),
                           index=list(curves).index("LightGBM"),
                           label_visibility="collapsed")
    cm = curves[cm_pick]["cm"]
    tn, fp, fn, tp = cm.ravel()
    m1, m2 = st.columns([1.1, 1], gap="large")
    with m1:
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Không bệnh", "Có bệnh"], y=["Không bệnh", "Có bệnh"],
            colorscale=[[0, "#EAF2FC"], [1, "#8FBBEE"]], showscale=False,
            text=cm, texttemplate="%{text}",
            textfont=dict(size=20, family="Inter", color="#12385C"),
            hovertemplate="Thực tế %{y}<br>Dự đoán %{x}<br><b>%{z}</b><extra></extra>",
        ))
        fig.update_layout(height=320, xaxis_title="Dự đoán", yaxis_title="Thực tế",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with m2:
        g = st.columns(2)
        kpi(g[0], "Bắt đúng ca bệnh", f"{tp}", "True Positive")
        kpi(g[1], "Bỏ sót ca bệnh", f"{fn}", "False Negative", "alert")
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        g = st.columns(2)
        kpi(g[0], "Báo động nhầm", f"{fp}", "False Positive", "warn")
        kpi(g[1], "Loại đúng ca lành", f"{tn}", "True Negative")
        if tp + fn:
            note(f"Mô hình bỏ sót <b>{fn}/{tp + fn}</b> ca bệnh "
                 f"({fn / (tp + fn):.0%}). Trong sàng lọc y tế, bỏ sót tốn kém hơn "
                 "báo động nhầm.")

    st.markdown("### Độ quan trọng của biến")
    imp = art["importance"].head(12).sort_values("Độ quan trọng")
    fig = go.Figure(go.Bar(
        x=imp["Độ quan trọng"], y=imp["Biến"], orientation="h",
        error_x=dict(array=imp["Độ lệch"], color=P.muted, thickness=1.2, width=3),
        marker=bar_marker([P.orange if v < 0 else P.accent
                           for v in imp["Độ quan trọng"]]),
        text=[f"{v:+.4f}" for v in imp["Độ quan trọng"]],
    ))
    fig.add_vline(x=0, line_color=P.axis, line_width=1.5)
    fig.update_layout(height=450, xaxis_title="Mức giảm ROC-AUC khi xáo trộn biến",
                      yaxis=dict(tickfont=dict(size=10.5)))
    labels(fig)
    st.plotly_chart(fig, width="stretch")
    note("Permutation importance trên LightGBM. Mọi giá trị đều quanh 0 và thanh sai "
         "số cắt qua vạch 0 — xáo trộn bất kỳ biến nào cũng không làm mô hình tệ đi.")

# ============================================================== TINH CHỈNH
def render_tuning():
    page_head("Bước 04", "Tinh chỉnh siêu tham số",
              "GridSearchCV duyệt toàn bộ lưới tham số, chấm điểm bằng "
              "cross-validation 5-fold.")

    tun, tun_err = get_tuning()
    if tun is None:
        if tun_err == "missing":
            st.warning("Chưa có kết quả tinh chỉnh. Chạy `python tune.py` trước.")
        else:
            st.warning(f"Không nạp được `artifacts/tuning.joblib`: `{tun_err}`")
        return

    tuning: pd.DataFrame = tun["tuning"]
    searches: dict = tun["searches"]

    worst = tuning.loc[tuning["Cải thiện trên test"].idxmin()]
    k = st.columns(4)
    kpi(k[0], "Mô hình tinh chỉnh", f"{len(tuning)}", "4 lưới tham số")
    kpi(k[1], "Tổng tổ hợp", f"{int(tuning['Số tổ hợp'].sum())}",
        f"{int(tuning['Số tổ hợp'].sum()) * 5} lần huấn luyện", "accent")
    kpi(k[2], "Cải thiện tốt nhất", f"{tuning['Cải thiện trên test'].max():+.4f}",
        "ROC-AUC trên tập test")
    kpi(k[3], "Thay đổi xấu nhất", f"{worst['Cải thiện trên test']:+.4f}",
        worst["Mô hình"], "alert")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.markdown("### Mặc định so với đã tinh chỉnh")
    disp = tuning.drop(columns=["Tham số tốt nhất"])
    st.dataframe(
        disp.style.format(
            {c: "{:.4f}" for c in disp.columns if c not in ("Mô hình", "Số tổ hợp")}
        ).background_gradient(subset=["Cải thiện trên test"], cmap="RdYlGn",
                              vmin=-0.03, vmax=0.03),
        width="stretch", hide_index=True,
    )
    banner(
        f"<b>{worst['Mô hình']}</b> có điểm cross-validation tăng lên "
        f"{worst['CV ROC-AUC (tốt nhất)']:.4f} nhưng điểm trên tập kiểm tra lại giảm "
        f"{worst['Test ROC-AUC (mặc định)']:.4f} → "
        f"{worst['Test ROC-AUC (đã tinh chỉnh)']:.4f}. Chọn tổ hợp có điểm CV cao nhất "
        "trong hàng chục lựa chọn là chọn tổ hợp hợp với cách chia fold đó, "
        "không phải tổ hợp tổng quát hoá tốt nhất.", "bad")

    st.markdown("### Mức thay đổi so với biên độ nhiễu")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tuning["Mô hình"], y=tuning["Cải thiện trên test"], name="Thay đổi trên test",
        marker=bar_marker([P.orange if v < 0 else P.accent
                           for v in tuning["Cải thiện trên test"]]),
        width=0.45, text=[f"{v:+.4f}" for v in tuning["Cải thiện trên test"]],
        textposition="inside", insidetextanchor="middle", cliponaxis=False,
        textfont=dict(family="Inter", size=10.5, color="white"),
    ))
    for sign, show in ((1, True), (-1, False)):
        fig.add_trace(go.Scatter(
            x=tuning["Mô hình"], y=sign * tuning["Độ lệch CV của tổ hợp tốt nhất"],
            mode="markers", name="Biên độ nhiễu giữa các fold (±1σ)",
            showlegend=show,
            marker=dict(size=13, symbol="diamond-open", color=P.muted,
                        line=dict(width=2)),
        ))
    fig.add_hline(y=0, line_color=P.axis, line_width=1.5)
    fig.update_layout(height=430, yaxis_title="Thay đổi ROC-AUC",
                      margin=dict(l=90, r=30, t=30, b=50),
                      yaxis=dict(range=[-0.042, 0.042]),
                      legend=dict(orientation="h", y=1.16, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")
    note("Mức thay đổi của mọi mô hình đều cùng cỡ hoặc nhỏ hơn biên độ dao động tự "
         "nhiên giữa các fold. Để kết luận tinh chỉnh có ích, mức cải thiện phải lớn "
         "hơn hẳn biên độ này.")

    st.markdown("### Chi tiết lưới tham số")
    mname = st.selectbox("Mô hình", list(searches), key="tune_model",
                         label_visibility="collapsed")
    cvr = searches[mname]["cv_results"].copy()
    cvr["Tổ hợp"] = cvr["params"].apply(
        lambda d: ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in d.items()))
    top = cvr.sort_values("mean_test_score", ascending=False).head(14)[::-1]

    fig = go.Figure(go.Bar(
        x=top["mean_test_score"], y=top["Tổ hợp"], orientation="h",
        error_x=dict(array=top["std_test_score"], color=P.muted, thickness=1.2, width=3),
        marker=bar_marker(P.accent),
        text=[f"{v:.4f}" for v in top["mean_test_score"]],
        textposition="inside", insidetextanchor="start",
        textfont=dict(family="Inter", size=10.5, color="white"),
        cliponaxis=False,
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=P.critical, line_width=1.6,
                  annotation_text="ngẫu nhiên", annotation_position="top")
    fig.update_layout(height=540, xaxis_title="CV ROC-AUC",
                      xaxis=dict(range=[0, 0.60]),
                      margin=dict(l=24, r=60),
                      yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig, width="stretch")

    gap = (cvr["mean_train_score"] - cvr["mean_test_score"]).max()
    note(f"14 tổ hợp dẫn đầu, kèm sai số giữa các fold. Các thanh sai số chồng lấn và "
         f"đều cắt qua vạch 0.5. Chênh lệch train–validation lớn nhất trong lưới "
         f"<b>{mname}</b> là <b>{gap:.3f}</b>.")

    st.markdown("### Tham số tốt nhất")
    st.dataframe(tuning[["Mô hình", "Tham số tốt nhất"]],
                 width="stretch", hide_index=True)


with tabs[3]:
    render_tuning()

# ======================================================= CHẨN ĐOÁN DỮ LIỆU
with tabs[4]:
    page_head("Bước 05", "Chẩn đoán dữ liệu",
              "Ba kiểm định độc lập đánh giá xem dữ liệu có chứa tín hiệu dự đoán "
              "hay không.")

    y = (df[TARGET] == "Yes").astype(int)
    corr = df[NUMERIC_COLS].corrwith(y)

    k = st.columns(4)
    kpi(k[0], "Accuracy baseline", f"{base_acc:.1%}", "luôn đoán lớp đa số")
    kpi(k[1], "Accuracy tốt nhất", f"{best['Accuracy']:.1%}",
        f"{(best['Accuracy'] - base_acc) * 100:+.2f} điểm")
    kpi(k[2], "ROC-AUC nhãn thật", f"{ctrl['ROC-AUC (nhãn thật)']:.4f}",
        art["control_model"], "alert")
    kpi(k[3], "ROC-AUC nhãn xáo trộn", f"{ctrl['ROC-AUC']:.4f}",
        f"{ctrl['ROC-AUC'] - ctrl['ROC-AUC (nhãn thật)']:+.4f} so với nhãn thật",
        "alert")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    st.markdown("### Kiểm định 1 — Accuracy so với baseline")
    comp = results.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp["Mô hình"], y=comp["Accuracy"], name="Accuracy",
                         marker=bar_marker(P.accent), width=0.38, offset=-0.4,
                         text=[f"{v:.3f}" for v in comp["Accuracy"]]))
    fig.add_trace(go.Bar(x=comp["Mô hình"], y=comp["ROC-AUC"], name="ROC-AUC",
                         marker=bar_marker(P.orange), width=0.38, offset=0.02,
                         text=[f"{v:.3f}" for v in comp["ROC-AUC"]]))
    fig.add_hline(y=0.8, line_dash="dash", line_color=P.accent, line_width=1.6,
                  annotation_text="baseline accuracy 0.80",
                  annotation_position="top left",
                  annotation_font=dict(size=10.5, color=P.accent))
    fig.add_hline(y=0.5, line_dash="dash", line_color=P.orange, line_width=1.6,
                  annotation_text="ngẫu nhiên 0.50",
                  annotation_position="bottom left",
                  annotation_font=dict(size=10.5, color=P.orange))
    labels(fig)
    fig.update_layout(height=440, yaxis_range=[0, 1.12], yaxis_title="Điểm",
                      xaxis=dict(tickangle=-25, tickfont=dict(size=10)),
                      legend=dict(orientation="h", y=1.14, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")
    note("Accuracy của mọi mô hình bám sát mốc 0.80 của baseline, trong khi ROC-AUC "
         "bám sát mốc 0.50 của dự đoán ngẫu nhiên.")

    st.markdown("### Kiểm định 2 — Huấn luyện trên nhãn xáo trộn")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Nhãn thật", "Nhãn xáo trộn ngẫu nhiên"],
        y=[ctrl["ROC-AUC (nhãn thật)"], ctrl["ROC-AUC"]],
        marker=bar_marker([P.accent, P.orange]), width=0.42,
        text=[f"{ctrl['ROC-AUC (nhãn thật)']:.4f}", f"{ctrl['ROC-AUC']:.4f}"],
        textposition="outside", textfont=dict(size=15),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=P.muted, line_width=1.6,
                  annotation_text="ngẫu nhiên")
    fig.update_layout(height=380, yaxis_range=[0.40, 0.58],
                      yaxis_title="ROC-AUC trên tập kiểm tra")
    st.plotly_chart(fig, width="stretch")
    note(f"Mô hình <b>{art['control_model']}</b> được huấn luyện lại sau khi xáo trộn "
         "toàn bộ nhãn tập train. Nếu dữ liệu chứa tín hiệu, điểm số phải sụt rõ rệt. "
         "Ở đây nhãn xáo trộn cho kết quả ngang bằng hoặc cao hơn nhãn thật.")

    st.markdown("### Kiểm định 3 — Tương quan giữa biến và nhãn")
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
                      xaxis_title="Hệ số tương quan với nhãn bệnh",
                      yaxis=dict(tickfont=dict(size=10.5)))
    st.plotly_chart(fig, width="stretch")
    note("Mọi hệ số nằm trong khoảng <b>±0.02</b>; hai vạch đứt đỏ là ngưỡng "
         "<b>|r| = 0.1</b> thường dùng cho một liên hệ yếu. Tỷ lệ mắc bệnh ở nhóm hút "
         "thuốc là 20.1% so với 19.9% ở nhóm không hút.")

    banner(
        "<b>Kết luận.</b> Ba kiểm định độc lập đều cho cùng một kết quả: các biến đầu "
        "vào không mang thông tin dự đoán về nhãn. Bộ dữ liệu là dữ liệu nhân tạo sinh "
        "ngẫu nhiên, phù hợp để minh hoạ quy trình nhưng không dùng để rút ra kết luận "
        "y học.", "bad")

# ================================================================= DỰ ĐOÁN
with tabs[5]:
    page_head("Bước 06", "Dự đoán cho một bệnh nhân",
              "Nhập thông số và chọn mô hình để xem xác suất dự đoán.")

    model_name = st.selectbox("Mô hình", real_models,
                              index=real_models.index("LightGBM"))

    with st.form("predict"):
        g1, g2, g3 = st.columns(3, gap="large")
        vals = {}
        with g1:
            st.markdown("**Thông tin cơ bản**")
            vals["Age"] = st.slider("Tuổi", 18, 80, 50)
            vals["Gender"] = st.radio("Giới tính", ["Male", "Female"], horizontal=True)
            vals["BMI"] = st.slider("BMI", 18.0, 40.0, 29.0, 0.1)
            vals["Sleep Hours"] = st.slider("Giờ ngủ mỗi đêm", 4.0, 10.0, 7.0, 0.1)
            vals["Exercise Habits"] = st.select_slider(
                "Mức vận động", ["Low", "Medium", "High"], "Medium")
        with g2:
            st.markdown("**Chỉ số xét nghiệm**")
            vals["Blood Pressure"] = st.slider("Huyết áp tâm thu", 120, 180, 150)
            vals["Cholesterol Level"] = st.slider("Cholesterol tổng", 150, 300, 225)
            vals["Triglyceride Level"] = st.slider("Triglyceride", 100, 400, 250)
            vals["Fasting Blood Sugar"] = st.slider("Đường huyết đói", 80, 160, 120)
            vals["CRP Level"] = st.slider("CRP", 0.0, 15.0, 7.5, 0.1)
            vals["Homocysteine Level"] = st.slider("Homocysteine", 5.0, 20.0, 12.5, 0.1)
        with g3:
            st.markdown("**Tiền sử và lối sống**")
            for c in NOMINAL_COLS:
                if c == "Gender":
                    continue
                vals[c] = st.radio(NHAN_VI.get(c, c), ["No", "Yes"],
                                   horizontal=True, key=f"r_{c}")
            vals["Alcohol Consumption"] = st.select_slider(
                "Rượu bia", ["None", "Low", "Medium", "High"], "Low")
            vals["Stress Level"] = st.select_slider(
                "Căng thẳng", ["Low", "Medium", "High"], "Medium")
            vals["Sugar Consumption"] = st.select_slider(
                "Đường và đồ ngọt", ["Low", "Medium", "High"], "Medium")

        submitted = st.form_submit_button("Dự đoán", width="stretch", type="primary")

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
            kpi(kk[0], "Kết luận", "Có nguy cơ" if pred else "Không nguy cơ",
                f"ngưỡng 50%", "alert" if pred else "accent")
            kpi(kk[1], "Xác suất", f"{proba:.1%}", "mức nền dữ liệu 20%")
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            note(f"Mô hình <b>{model_name}</b> · ROC-AUC trên tập kiểm tra "
                 f"<b>{results.loc[results['Mô hình'] == model_name, 'ROC-AUC'].iloc[0]:.3f}</b>. "
                 "Xem tab Chẩn đoán dữ liệu để biết mức tin cậy của con số này.")

    st.markdown("### Đối chiếu hai hồ sơ cực đoan")
    cmp_df = compare_extremes()
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cmp_df["Mô hình"], y=cmp_df["Khoẻ mạnh"],
                             name="Hồ sơ khoẻ mạnh", marker=bar_marker(P.accent),
                             text=[f"{v:.0%}" for v in cmp_df["Khoẻ mạnh"]]))
        fig.add_trace(go.Bar(x=cmp_df["Mô hình"], y=cmp_df["Nguy cơ cao"],
                             name="Hồ sơ nguy cơ cao", marker=bar_marker(P.orange),
                             text=[f"{v:.0%}" for v in cmp_df["Nguy cơ cao"]]))
        labels(fig)
        fig.update_layout(barmode="group", height=430,
                          yaxis_title="Xác suất mắc bệnh", yaxis_tickformat=".0%",
                          xaxis=dict(tickangle=-25, tickfont=dict(size=9.5)),
                          legend=dict(orientation="h", y=1.14, x=1, xanchor="right"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(
            cmp_df.style.format({"Khoẻ mạnh": "{:.1%}", "Nguy cơ cao": "{:.1%}",
                                 "Chênh lệch": "{:+.1%}"})
            .background_gradient(subset=["Chênh lệch"], cmap="Oranges"),
            width="stretch", hide_index=True, height=330,
        )
    note("Hồ sơ khoẻ mạnh: 25 tuổi, BMI 21, không hút thuốc, mọi chỉ số ở mức tốt "
         "nhất. Hồ sơ nguy cơ cao: 80 tuổi, BMI 40, hút thuốc, tiểu đường, mọi chỉ số "
         "ở mức xấu nhất. Cả hai nằm ngoài vùng dữ liệu huấn luyện nên mô hình đang "
         "ngoại suy.")
