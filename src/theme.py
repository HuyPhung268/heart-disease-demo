"""Giao diện dùng chung: bảng màu, CSS và template biểu đồ.

Chế độ sáng/tối do Streamlit quản lý (menu ☰ → Settings → Appearance) và được
khai báo trong .streamlit/config.toml.

Nguyên tắc thiết kế: mọi thành phần tự vẽ ở đây đều **không phụ thuộc theme**.
Màu nền dùng rgba trung tính, màu chữ dùng ``inherit``, nền biểu đồ để trong
suốt. Nhờ vậy giao diện đổi theo theme ngay lập tức, không phải chạy lại script
và không lệ thuộc ``st.context.theme`` (API này báo sai đúng lúc người dùng
chuyển theme).
"""
from dataclasses import dataclass, field

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Sắc độ trung tính, đọc được trên cả nền trắng lẫn nền tối.
NEUTRAL = "138, 148, 166"


@dataclass(frozen=True)
class Palette:
    teal: str = "#14B8A6"
    amber: str = "#F59E0B"
    indigo: str = "#6366F1"
    rose: str = "#F43F5E"
    sky: str = "#0EA5E9"
    slate: str = "#94A3B8"

    ink: str = "#8A94A6"                       # chữ phụ trên biểu đồ
    muted: str = "#8A94A6"
    line: str = f"rgba({NEUTRAL}, .34)"        # viền, trục
    axis: str = f"rgba({NEUTRAL}, .45)"        # vạch mốc
    grid: str = f"rgba({NEUTRAL}, .16)"
    soft: str = f"rgba({NEUTRAL}, .10)"        # nền đồng hồ đo

    heat_lo: str = "#0E4F4A"                   # ô nhạt của ma trận nhầm lẫn
    band_good: str = "rgba(34, 197, 94, .24)"
    band_warn: str = "rgba(245, 158, 11, .24)"
    band_bad: str = "rgba(244, 63, 94, .24)"

    colorway: list = field(default_factory=lambda: [
        "#14B8A6", "#F59E0B", "#6366F1", "#F43F5E", "#0EA5E9", "#94A3B8",
    ])


P = Palette()

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stMetric {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
}}
footer, header [data-testid="stStatusWidget"] {{ visibility: hidden; }}

/* ---- Chiều rộng: dùng trọn màn hình ---- */
.block-container {{
    padding: 2.1rem 2.6rem 3rem 2.6rem;
    max-width: 100% !important;
}}
@media (min-width: 2200px) {{
    .block-container {{ padding-left: 4rem; padding-right: 4rem; }}
}}

h1 {{ font-weight: 700; letter-spacing: -.025em; }}
h2 {{ font-weight: 650; letter-spacing: -.02em;
     font-size: 1.5rem !important; margin-top: .2rem !important; }}
h3 {{ font-weight: 600; letter-spacing: -.01em;
     font-size: 1.1rem !important; margin-top: 1.7rem !important; }}

/* ---- Tiêu đề trang ---- */
.page-head {{ margin-bottom: 1.6rem; }}
.page-head .eyebrow {{
    font-size: .72rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: {P.teal}; margin-bottom: .35rem;
}}
.page-head .lede {{ font-size: .95rem; margin-top: .35rem;
                    color: inherit; opacity: .66; }}

/* ---- Thẻ số liệu ---- */
.kpi {{
    background: rgba({NEUTRAL}, .06);
    border: 1px solid rgba({NEUTRAL}, .26);
    border-radius: 14px; padding: 1.05rem 1.2rem; height: 100%;
    transition: background .15s ease, transform .15s ease, border-color .15s ease;
}}
.kpi:hover {{ background: rgba({NEUTRAL}, .11);
              border-color: rgba({NEUTRAL}, .4); transform: translateY(-1px); }}
.kpi .label {{
    font-size: .73rem; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: inherit; opacity: .6;
}}
.kpi .value {{
    font-size: 1.9rem; font-weight: 700; color: inherit;
    line-height: 1.15; margin-top: .3rem; font-variant-numeric: tabular-nums;
}}
.kpi .sub {{ font-size: .8rem; color: inherit; opacity: .52; margin-top: .22rem; }}
.kpi.accent {{ border-left: 3px solid {P.teal}; }}
.kpi.warn   {{ border-left: 3px solid {P.amber}; }}
.kpi.alert  {{ border-left: 3px solid {P.rose}; }}
.kpi.alert .value {{ color: {P.rose}; }}

/* ---- Tabs ----
   Streamlit <=1.5x dùng [data-baseweb="tab"], >=1.6x đổi sang
   [role="tab"] / [data-testid="stTab"]. Khai báo cả hai để không phụ
   thuộc phiên bản. */
.stTabs [data-baseweb="tab-list"],
.stTabs [role="tablist"] {{
    gap: .3rem; border-bottom: 1px solid rgba({NEUTRAL}, .26); padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"],
.stTabs [role="tab"],
.stTabs [data-testid="stTab"] {{
    height: 46px; padding: 0 1.05rem; background: transparent;
    border-radius: 9px 9px 0 0; font-weight: 550; font-size: .92rem;
    color: inherit; opacity: .62;
    display: flex; align-items: center;
}}
.stTabs [data-baseweb="tab"]:hover,
.stTabs [role="tab"]:hover {{ opacity: .9; }}
.stTabs [aria-selected="true"] {{
    background: rgba(20, 184, 166, .13) !important;
    color: {P.teal} !important; opacity: 1;
    border-bottom: 2px solid {P.teal};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid rgba({NEUTRAL}, .26); border-radius: 12px;
}}

/* ---- Chú thích dưới biểu đồ ---- */
.note {{
    font-size: .85rem; color: inherit; opacity: .68;
    border-left: 2px solid rgba({NEUTRAL}, .3);
    padding: .1rem 0 .1rem .8rem; margin: .1rem 0 1.1rem 0; line-height: 1.55;
}}
.note b {{ font-weight: 650; opacity: 1; }}
.note code, .banner code {{
    background: rgba({NEUTRAL}, .16); padding: .08rem .32rem;
    border-radius: 4px; font-size: .84em;
}}

/* ---- Sidebar ---- */
.side-title {{ font-size: 1.14rem; font-weight: 700; letter-spacing: -.02em; }}
.side-sub {{ font-size: .8rem; opacity: .6; margin: .1rem 0 1rem 0; }}
.side-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    padding: .42rem 0; border-bottom: 1px solid rgba({NEUTRAL}, .2);
    font-size: .86rem;
}}
.side-row span:first-child {{ opacity: .62; }}
.side-row span:last-child {{ font-weight: 650; font-variant-numeric: tabular-nums; }}
.side-hint {{
    font-size: .79rem; line-height: 1.55; opacity: .82;
    background: rgba({NEUTRAL}, .09); border: 1px solid rgba({NEUTRAL}, .22);
    border-radius: 9px; padding: .62rem .78rem; margin-top: 1rem;
}}
.side-hint code {{ background: rgba({NEUTRAL}, .18); padding: .05rem .28rem;
                   border-radius: 4px; font-size: .9em; }}

/* ---- Băng kết quả: sắc thái thể hiện bằng nền và viền trái ---- */
.banner {{
    border-radius: 12px; padding: 1rem 1.25rem; margin: .5rem 0 1.2rem 0;
    font-size: .92rem; line-height: 1.6; color: inherit;
}}
.banner.good {{ background: rgba(34, 197, 94, .11);
                border: 1px solid rgba(34, 197, 94, .3); border-left: 3px solid #22C55E; }}
.banner.bad  {{ background: rgba(244, 63, 94, .11);
                border: 1px solid rgba(244, 63, 94, .3); border-left: 3px solid {P.rose}; }}
.banner.info {{ background: rgba(14, 165, 233, .11);
                border: 1px solid rgba(14, 165, 233, .3); border-left: 3px solid {P.sky}; }}
.banner b {{ font-weight: 700; }}

div[data-testid="stForm"] {{
    border: 1px solid rgba({NEUTRAL}, .26); border-radius: 14px;
    padding: 1.3rem 1.4rem; background: rgba({NEUTRAL}, .04);
}}
</style>
"""

_TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=P.muted),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=P.colorway,
    xaxis=dict(gridcolor=P.grid, linecolor=P.line, ticks="outside",
               tickcolor=P.line, zeroline=False, title_font=dict(size=11.5)),
    yaxis=dict(gridcolor=P.grid, linecolor=P.line, ticks="outside",
               tickcolor=P.line, zeroline=False, title_font=dict(size=11.5)),
    margin=dict(t=30, r=24, b=44, l=24),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=P.line, borderwidth=1,
                font=dict(size=11, color=P.muted)),
    hoverlabel=dict(font=dict(family="Inter", size=12)),
))


def apply_theme() -> Palette:
    """Nạp CSS, đặt template Plotly mặc định và trả về bảng màu."""
    pio.templates["demo"] = _TEMPLATE
    pio.templates.default = "demo"
    st.markdown(CSS, unsafe_allow_html=True)
    return P


def page_head(eyebrow: str, title: str, lede: str = ""):
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{eyebrow}</div>'
        f"<h2>{title}</h2>"
        + (f'<div class="lede">{lede}</div>' if lede else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value: str, sub: str = "", tone: str = ""):
    col.markdown(
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def banner(text: str, tone: str = "info"):
    st.markdown(f'<div class="banner {tone}">{text}</div>', unsafe_allow_html=True)
