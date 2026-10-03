"""Shared look and feel: palette, CSS and the chart template.

Light/dark mode is owned by Streamlit (menu > Settings > Appearance) and
declared in .streamlit/config.toml.

Design rule: every component drawn here is **theme-independent**. Backgrounds
use neutral rgba, text uses ``inherit``, chart backgrounds are transparent.
That way the UI follows the theme instantly, with no rerun, and without relying
on ``st.context.theme`` (that API reports the wrong value at the exact moment
the user switches theme).
"""
from dataclasses import dataclass, field

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Neutral tone that reads on both a white and a dark surface.
NEUTRAL = "138, 148, 166"


@dataclass(frozen=True)
class Palette:
    """Validated categorical palette from the dataviz skill.

    Uses the reference palette's "dark" column: all 8 slots pass all five
    checks (lightness band, chroma floor, CVD separation, normal-vision floor,
    contrast >= 3:1) against BOTH the light surface #FFFFFF and the dark
    surface #0B1120. One palette therefore serves both modes with no need to
    read the theme at runtime. Slot order is the colour-blind safety
    mechanism - assign in order, never cycle.
    """
    # 8 categorical slots, in the validated order
    blue: str = "#3987e5"       # slot 1 - the primary UI accent
    orange: str = "#d95926"     # slot 2
    aqua: str = "#199e70"       # slot 3
    yellow: str = "#c98500"     # slot 4
    magenta: str = "#d55181"    # slot 5
    green: str = "#008300"      # slot 6
    violet: str = "#9085e9"     # slot 7
    red: str = "#e66767"        # slot 8

    # Status colours - fixed, never reused as a data series colour
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"

    # Ink and chart chrome. #898781 is the muted tone shared by both modes.
    muted: str = "#898781"
    ink: str = "#898781"
    line: str = f"rgba({NEUTRAL}, .34)"
    axis: str = f"rgba({NEUTRAL}, .45)"
    grid: str = f"rgba({NEUTRAL}, .16)"
    soft: str = f"rgba({NEUTRAL}, .10)"

    heat_lo: str = "#0E3D63"            # dark end of the confusion-matrix ramp
    band_good: str = "rgba(12, 163, 12, .18)"
    band_warn: str = "rgba(250, 178, 25, .20)"
    band_bad: str = "rgba(208, 59, 59, .20)"

    colorway: list = field(default_factory=lambda: [
        "#3987e5", "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", "#9085e9", "#e66767",
    ])

    # Semantic roles used across this dashboard
    @property
    def healthy(self) -> str:      # "No disease" - slot 1
        return self.blue

    @property
    def diseased(self) -> str:     # "Has disease" - slot 2
        return self.orange

    @property
    def accent(self) -> str:       # UI accent
        return self.blue


P = Palette()

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stMetric {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
}}
footer, header [data-testid="stStatusWidget"] {{ visibility: hidden; }}

/* ---- Width: use the full screen ---- */
.block-container {{
    /* Streamlit paints an opaque 60px header (very high z-index) across the
       top of the page. Top padding must exceed 60px, otherwise the header
       clips the top edge of the first row of tab cards. */
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

/* ---- Page heading ---- */
.page-head {{ margin-bottom: 1.6rem; }}
.page-head .eyebrow {{
    font-size: .72rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: {P.accent}; margin-bottom: .35rem;
}}
.page-head .lede {{ font-size: .95rem; margin-top: .35rem;
                    color: inherit; opacity: .66; }}

/* ---- Stat cards ---- */
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

/* ---- Tabs: rounded cards that match .kpi ----
   Streamlit <=1.5x uses [data-baseweb="tab"], >=1.6x switched to
   [role="tab"] / [data-testid="stTab"]. Declare both so the CSS does not
   depend on the version. */
.stTabs [data-baseweb="tab-list"],
.stTabs [role="tablist"] {{
    gap: .55rem; border-bottom: none; flex-wrap: wrap;
    /* Streamlit sets overflow-y:hidden with a height that exactly matches the
       tab card, clipping its top edge. Add vertical padding so the border and
       the hover lift are not cut off. */
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
/* Remove Streamlit's own tab underline:
   <=1.5x uses a dedicated element, >=1.6x uses ::after on the tablist. */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [role="tablist"]::after {{ content: none !important; }}
.stTabs .react-aria-SelectionIndicator {{ display: none !important; }}

[data-testid="stDataFrame"] {{
    border: 1px solid rgba({NEUTRAL}, .26); border-radius: 12px;
}}

/* ---- Caption under a chart ---- */
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

/* ---- Verdict card in the sidebar ---- */
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

/* ---- Result banner: tone carried by the tint and the left border ---- */
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
    # Gap between bars so two colour blocks never touch
    bargap=0.28, bargroupgap=0.12,
))

# Value-label style: text always wears ink, never the series colour.
VALUE_FONT = dict(family="Inter", size=10.5, color=P.muted)
BAR_RADIUS = 4          # rounded bar end
LINE_WIDTH = 2.2


# Outline for data marks. A neutral semi-transparent tone so the same value
# reads as an edge on a white page and on a dark one.
MARK_EDGE = f"rgb({NEUTRAL})"
MARK_EDGE_W = 1.6
MARKER_SIZE = 8          # >= 8px, per the dataviz mark spec
MARKER_EVERY = 14        # draw a marker on 1 point in N along a dense line


def bar_marker(color, **kw):
    """Standard bar marker: 4px rounded end plus a neutral outline."""
    return dict(
        color=color,
        cornerradius=BAR_RADIUS,
        line=dict(width=MARK_EDGE_W, color=MARK_EDGE),
        **kw,
    )


def line_markers(x, y, color, every: int = MARKER_EVERY):
    """Evenly spaced marker positions along a dense line.

    A line such as an ROC curve has hundreds of points; putting a marker on
    every one is unreadable. This picks roughly one point in ``every``, always
    keeping the first and last.
    """
    n = len(x)
    if n == 0:
        return [], []
    step = max(1, n // max(1, every))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [x[i] for i in idx], [y[i] for i in idx]


def labels(fig, position="outside"):
    """Turn on value labels for every bar trace in the figure."""
    fig.update_traces(textposition=position, textfont=VALUE_FONT,
                      cliponaxis=False, selector=dict(type="bar"))
    return fig


def apply_theme() -> Palette:
    """Inject the CSS, set the default Plotly template, return the palette."""
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
