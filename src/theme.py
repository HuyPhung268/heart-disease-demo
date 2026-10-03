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
    """Bảng màu phân loại đã qua kiểm định của skill dataviz.

    Dùng cột "dark" của bảng tham chiếu: toàn bộ 8 slot đạt cả 5 phép kiểm
    (dải độ sáng, sàn chroma, tách màu cho người mù màu, sàn thị giác thường,
    tương phản >= 3:1) trên CẢ nền sáng #FFFFFF lẫn nền tối #0B1120. Nhờ vậy
    một bộ màu duy nhất dùng được cho cả hai chế độ, không cần đọc theme lúc
    chạy. Thứ tự slot là cơ chế an toàn mù màu — gán theo thứ tự, không xoay vòng.
    """
    # 8 slot phân loại, theo đúng thứ tự đã kiểm định
    blue: str = "#3987e5"       # slot 1 — màu nhấn chính của giao diện
    orange: str = "#d95926"     # slot 2
    aqua: str = "#199e70"       # slot 3
    yellow: str = "#c98500"     # slot 4
    magenta: str = "#d55181"    # slot 5
    green: str = "#008300"      # slot 6
    violet: str = "#9085e9"     # slot 7
    red: str = "#e66767"        # slot 8

    # Màu trạng thái — cố định, không bao giờ dùng làm màu chuỗi dữ liệu
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"

    # Mực và khung biểu đồ. #898781 là màu muted dùng chung cho cả hai chế độ.
    muted: str = "#898781"
    ink: str = "#898781"
    line: str = f"rgba({NEUTRAL}, .34)"
    axis: str = f"rgba({NEUTRAL}, .45)"
    grid: str = f"rgba({NEUTRAL}, .16)"
    soft: str = f"rgba({NEUTRAL}, .10)"

    heat_lo: str = "#0E3D63"            # đầu đậm của thang ma trận nhầm lẫn
    band_good: str = "rgba(12, 163, 12, .18)"
    band_warn: str = "rgba(250, 178, 25, .20)"
    band_bad: str = "rgba(208, 59, 59, .20)"

    colorway: list = field(default_factory=lambda: [
        "#3987e5", "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", "#9085e9", "#e66767",
    ])

    # Vai trò ngữ nghĩa trong dashboard này
    @property
    def healthy(self) -> str:      # "Không bệnh" — slot 1
        return self.blue

    @property
    def diseased(self) -> str:     # "Có bệnh" — slot 2
        return self.orange

    @property
    def accent(self) -> str:       # màu nhấn giao diện
        return self.blue


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
    /* Streamlit có header đục (nền trắng/tối, z-index rất cao) cao 60px phủ
       từ đỉnh trang. Đệm trên phải lớn hơn 60px, nếu không mép trên của thẻ
       tab đầu tiên sẽ bị header che mất. */
    padding: 4.6rem 2.6rem 3rem 2.6rem;
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
    text-transform: uppercase; color: {P.accent}; margin-bottom: .35rem;
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
.kpi.accent {{ border-left: 3px solid {P.accent}; }}
.kpi.warn   {{ border-left: 3px solid {P.yellow}; }}
.kpi.alert  {{ border-left: 3px solid {P.critical}; }}
.kpi.alert .value {{ color: {P.critical}; }}

/* ---- Tabs: bo tròn thành thẻ, đồng bộ với .kpi ----
   Streamlit <=1.5x dùng [data-baseweb="tab"], >=1.6x đổi sang
   [role="tab"] / [data-testid="stTab"]. Khai báo cả hai để không phụ
   thuộc phiên bản. */
.stTabs [data-baseweb="tab-list"],
.stTabs [role="tablist"] {{
    gap: .55rem; border-bottom: none; flex-wrap: wrap;
    /* Streamlit đặt overflow-y:hidden và chiều cao vừa khít thẻ tab, làm mép
       trên bị cắt. Chừa đệm dọc để viền và hiệu ứng nhấc khi hover không bị xén. */
    padding: 4px 0 5px 0; height: auto; align-items: center;
    overflow-y: visible;
    margin-bottom: 1.2rem;
}}
.stTabs [data-baseweb="tab"],
.stTabs [role="tab"],
.stTabs [data-testid="stTab"] {{
    box-sizing: border-box; flex: 0 0 auto;
    height: 46px; padding: 0 1.15rem;
    background: rgba({NEUTRAL}, .06);
    border: 1px solid rgba({NEUTRAL}, .26);
    border-radius: 14px;
    font-weight: 550; font-size: .92rem;
    color: inherit; opacity: .72;
    display: flex; align-items: center; white-space: nowrap;
    transition: background .15s ease, border-color .15s ease,
                transform .15s ease, opacity .15s ease;
}}
.stTabs [data-baseweb="tab"]:hover,
.stTabs [role="tab"]:hover {{
    background: rgba({NEUTRAL}, .12);
    border-color: rgba({NEUTRAL}, .4);
    opacity: 1; transform: translateY(-1px);
}}
.stTabs [aria-selected="true"] {{
    background: rgba(57, 135, 229, .13) !important;
    border-color: rgba(57, 135, 229, .6) !important;
    color: {P.accent} !important; opacity: 1;
    box-shadow: inset 0 0 0 1px rgba(57, 135, 229, .22);
}}
/* Gỡ gạch chân mặc định của Streamlit:
   <=1.5x dựng bằng phần tử riêng, >=1.6x dựng bằng ::after trên tablist. */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [role="tablist"]::after {{ content: none !important; }}
.stTabs .react-aria-SelectionIndicator {{ display: none !important; }}

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

/* ---- Thẻ kết luận ở sidebar ---- */
.verdict {{
    border: 1px solid rgba(208, 59, 59, .34);
    background: rgba(208, 59, 59, .08);
    border-radius: 14px; padding: .95rem 1rem; margin-top: 1.1rem;
}}
.verdict .vhead {{
    font-size: .68rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: {P.critical}; margin-bottom: .6rem;
}}
.verdict .vbig {{
    font-size: 1.75rem; font-weight: 700; color: {P.critical};
    line-height: 1.1; font-variant-numeric: tabular-nums;
}}
.verdict .vcap {{ font-size: .76rem; opacity: .62; margin-top: .1rem; }}
.verdict .vrow {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .8rem; padding: .3rem 0;
    border-top: 1px solid rgba(208, 59, 59, .2);
}}
.verdict .vrow:first-of-type {{ margin-top: .7rem; }}
.verdict .vrow span:first-child {{ opacity: .7; }}
.verdict .vrow span:last-child {{ font-weight: 650; font-variant-numeric: tabular-nums; }}
.verdict .vnote {{
    font-size: .79rem; line-height: 1.5; margin-top: .7rem;
    padding-top: .65rem; border-top: 1px solid rgba(208, 59, 59, .2);
    opacity: .85;
}}

/* ---- Băng kết quả: sắc thái thể hiện bằng nền và viền trái ---- */
.banner {{
    border-radius: 12px; padding: 1rem 1.25rem; margin: .5rem 0 1.2rem 0;
    font-size: .92rem; line-height: 1.6; color: inherit;
}}
.banner.good {{ background: rgba(12, 163, 12, .1);
                border: 1px solid rgba(12, 163, 12, .32); border-left: 3px solid {P.good}; }}
.banner.bad  {{ background: rgba(208, 59, 59, .1);
                border: 1px solid rgba(208, 59, 59, .32); border-left: 3px solid {P.critical}; }}
.banner.info {{ background: rgba(25, 158, 112, .1);
                border: 1px solid rgba(25, 158, 112, .32); border-left: 3px solid {P.aqua}; }}
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
    # Khe hở giữa các cột để hai mảng màu không chạm nhau
    bargap=0.28, bargroupgap=0.12,
))

# Kiểu nhãn giá trị: chữ luôn mang màu mực, không bao giờ mang màu chuỗi dữ liệu.
VALUE_FONT = dict(family="Inter", size=10.5, color=P.muted)
BAR_RADIUS = 4          # bo đầu cột
LINE_WIDTH = 2.2


def bar_marker(color, **kw):
    """Marker chuẩn cho cột: bo đầu 4px, không viền."""
    return dict(color=color, cornerradius=BAR_RADIUS, line=dict(width=0), **kw)


def labels(fig, position="outside"):
    """Bật nhãn giá trị cho mọi trace cột trong figure."""
    fig.update_traces(textposition=position, textfont=VALUE_FONT,
                      cliponaxis=False, selector=dict(type="bar"))
    return fig


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
