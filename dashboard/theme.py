"""Design system for the IDX Scanner dashboard.

Single source of truth for the dashboard's visual language. Centralises the
colour tokens, spacing/typography scale and component styling so the UI reads
as one intentional product instead of a default Streamlit prototype.

Theme approach (stable, built-in first):
  1. ``.streamlit/config.toml`` ships a polished DARK theme as the first-paint
     default (so native widgets — dataframes, inputs, tabs — are correct before
     any Python runs).
  2. ``apply_theme(mode)`` switches Streamlit's *native* theme at runtime via
     ``st.config.set_option("theme.*", ...)``. On Streamlit 1.37 this re-themes
     the canvas-based ``st.dataframe`` correctly (verified in-browser), which a
     CSS-only override cannot do.
  3. A baked-per-mode CSS layer refines typography, spacing, cards, badges,
     tabs and chart containers on top of the native theme.
  4. ``render_theme_toggle()`` gives users an explicit in-app Light/Dark switch
     (session-state driven) in addition to Streamlit's ⋮ → Settings → Theme.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens — two coherent palettes sharing the same semantic slots.
# Neutral surfaces, ONE accent (blue), restrained semantic colours.
# ---------------------------------------------------------------------------
TOKENS: dict[str, dict[str, str]] = {
    "dark": {
        "bg":            "#0F1115",  # app canvas
        "surface":       "#161A20",  # cards, sidebar
        "surface_2":     "#1E232B",  # raised / hover
        "border":        "#272D36",
        "border_strong": "#39424E",
        "text":          "#E6E8EC",
        "muted":         "#9BA4B0",
        "faint":         "#6B7480",
        "accent":        "#3B82F6",
        "accent_text":   "#93C5FD",
        "accent_weak":   "rgba(59,130,246,0.16)",
        "success":       "#34D399",
        "danger":        "#F87171",
        "warning":       "#FBBF24",
        "info":          "#38BDF8",
        "success_bg":    "rgba(52,211,153,0.14)",
        "danger_bg":     "rgba(248,113,113,0.14)",
        "warning_bg":    "rgba(251,191,36,0.14)",
        "info_bg":       "rgba(56,189,248,0.14)",
        "neutral_bg":    "rgba(155,164,176,0.12)",
        # chart-specific
        "chart_paper":   "#0F1115",
        "chart_plot":    "#0F1115",
        "chart_surface": "#161A20",
        "grid":          "#222831",
        "up":            "#22C55E",
        "down":          "#EF4444",
        "template":      "plotly_dark",
    },
    "light": {
        "bg":            "#FFFFFF",
        "surface":       "#F6F7F9",
        "surface_2":     "#EEF1F4",
        "border":        "#E4E7EC",
        "border_strong": "#CDD3DC",
        "text":          "#1A1D23",
        "muted":         "#586170",
        "faint":         "#8A93A0",
        "accent":        "#2563EB",
        "accent_text":   "#1D4ED8",
        "accent_weak":   "rgba(37,99,235,0.10)",
        "success":       "#16A34A",
        "danger":        "#DC2626",
        "warning":       "#B45309",
        "info":          "#0369A1",
        "success_bg":    "rgba(22,163,74,0.12)",
        "danger_bg":     "rgba(220,38,38,0.10)",
        "warning_bg":    "rgba(180,83,9,0.10)",
        "info_bg":       "rgba(3,105,161,0.10)",
        "neutral_bg":    "rgba(88,97,112,0.10)",
        "chart_paper":   "#FFFFFF",
        "chart_plot":    "#FFFFFF",
        "chart_surface": "#F6F7F9",
        "grid":          "#EBEEF2",
        "up":            "#16A34A",
        "down":          "#DC2626",
        "template":      "plotly_white",
    },
}

# Native Streamlit theme options per mode (drives canvas widgets + base chrome).
_NATIVE = {
    "dark": {
        "base": "dark",
        "primaryColor": TOKENS["dark"]["accent"],
        "backgroundColor": TOKENS["dark"]["bg"],
        "secondaryBackgroundColor": TOKENS["dark"]["surface"],
        "textColor": TOKENS["dark"]["text"],
        "font": "sans serif",
    },
    "light": {
        "base": "light",
        "primaryColor": TOKENS["light"]["accent"],
        "backgroundColor": TOKENS["light"]["bg"],
        "secondaryBackgroundColor": TOKENS["light"]["surface"],
        "textColor": TOKENS["light"]["text"],
        "font": "sans serif",
    },
}


# The chosen mode is persisted in the URL query param ?theme=dark|light and the
# toggle does a full page reload on change. Why a reload instead of a plain
# rerun: Streamlit only re-pushes the *native* theme to the browser when the
# runtime config differs from config.toml, and rerun-driven set_option pushes
# are racy (the canvas st.dataframe can lag a rerun). A fresh session applies
# set_option deterministically, so reloading guarantees both modes — chrome AND
# canvas widgets — always match the toggle.
_THEME_LABELS = ["🌙 Gelap", "☀️ Terang"]


def get_mode() -> str:
    """Current UI theme mode from the ?theme= query param (default dark)."""
    try:
        val = st.query_params.get("theme")
    except Exception:  # noqa: BLE001
        val = None
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    return val if val in ("dark", "light") else "dark"


def palette(mode: str | None = None) -> dict[str, str]:
    """Return the active token palette (used by charts + Styler colouring)."""
    return TOKENS[mode or get_mode()]


def chart_palette(mode: str | None = None) -> dict[str, str]:
    """Subset of tokens charts need, with stable keys."""
    p = palette(mode)
    return {
        "template": p["template"],
        "paper":   p["chart_paper"],
        "plot":    p["chart_plot"],
        "surface": p["chart_surface"],
        "grid":    p["grid"],
        "text":    p["text"],
        "muted":   p["muted"],
        "up":      p["up"],
        "down":    p["down"],
        "accent":  p["accent"],
    }


def pos_neg_colors(mode: str | None = None) -> tuple[str, str, str]:
    """(+, -, 0) colours for numeric table cells, contrast-tuned per mode."""
    p = palette(mode)
    return p["success"], p["danger"], p["faint"]


# ---------------------------------------------------------------------------
# CSS — one baked stylesheet per mode. Uses an explicit spacing/type scale so
# rhythm is consistent. No gradients, restrained radii, single accent.
# ---------------------------------------------------------------------------
def _css(mode: str) -> str:
    t = TOKENS[mode]
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  /* spacing scale */
  --s1: 4px;  --s2: 8px;  --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px;
  /* radius scale (restrained) */
  --r-sm: 6px; --r-md: 8px; --r-lg: 10px;
  /* type scale */
  --fs-xs: 11px; --fs-sm: 12.5px; --fs-md: 14px; --fs-lg: 16px;
  --fs-xl: 20px; --fs-2xl: 26px;
  /* palette (active mode) */
  --c-bg: {t['bg']}; --c-surface: {t['surface']}; --c-surface-2: {t['surface_2']};
  --c-border: {t['border']}; --c-border-strong: {t['border_strong']};
  --c-text: {t['text']}; --c-muted: {t['muted']}; --c-faint: {t['faint']};
  --c-accent: {t['accent']}; --c-accent-weak: {t['accent_weak']};
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif;
}}

/* ---------- base typography + layout rhythm ---------- */
html, body, [class*="css"], .stMarkdown, [data-testid="stAppViewContainer"] {{
  font-family: var(--font-sans);
}}
/* CSS owns the core surfaces so the chrome always matches the toggle, even if
   the native theme push lags a rerun. Canvas widgets (st.dataframe) still
   follow the native theme, which a full reload applies deterministically. */
[data-testid="stAppViewContainer"], .stApp {{ background: var(--c-bg) !important; }}
[data-testid="stSidebar"] {{ background: var(--c-surface) !important; }}
[data-testid="stHeader"] {{ background: var(--c-bg) !important; }}
[data-testid="stAppViewContainer"] {{ color: var(--c-text); }}
.block-container {{
  padding: 1.6rem 2.4rem 4rem;
  max-width: 1480px;
}}
.stMarkdown p, .stMarkdown li {{ font-size: var(--fs-md); line-height: 1.62; }}

/* heading hierarchy — tight, intentional, single weight ramp */
h1, h2, h3, h4 {{ font-family: var(--font-sans); letter-spacing: -0.012em; color: var(--c-text); }}
h1 {{ font-size: var(--fs-2xl); font-weight: 700; margin: 0 0 var(--s3); }}
h2 {{ font-size: var(--fs-xl); font-weight: 650; margin: var(--s5) 0 var(--s3); }}
h3 {{ font-size: var(--fs-lg); font-weight: 600; margin: var(--s4) 0 var(--s2);
      padding-bottom: 0; }}
h4 {{ font-size: var(--fs-md); font-weight: 600; color: var(--c-muted); }}

[data-testid="stCaptionContainer"], .stCaption, small {{
  color: var(--c-muted) !important; font-size: var(--fs-sm) !important; line-height: 1.5;
}}

/* ---------- header bar ---------- */
[data-testid="stHeader"] {{
  background: transparent; border-bottom: 1px solid var(--c-border);
  height: 2.6rem;
}}
[data-testid="stToolbar"] {{ right: 0.6rem; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--c-border); }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: var(--s4); }}
[data-testid="stSidebar"] .block-container {{ padding-top: var(--s3); }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ margin-top: var(--s3); }}
[data-testid="stSidebar"] hr {{ margin: var(--s3) 0; }}

/* ---------- horizontal rule ---------- */
hr {{ border-color: var(--c-border); margin: var(--s4) 0; }}

/* ---------- app brand (sidebar header) ---------- */
.app-brand {{ display:flex; align-items:center; gap:10px; margin: 2px 0 12px; }}
.app-brand .mark {{
  width:30px; height:30px; border-radius:var(--r-md); flex:0 0 auto;
  background: var(--c-accent-weak); color: var(--c-accent);
  display:flex; align-items:center; justify-content:center; font-size:16px;
  border:1px solid var(--c-accent);
}}
.app-brand .name {{ font-size:15px; font-weight:700; color:var(--c-text); line-height:1.1; }}
.app-brand .sub {{ font-size:11px; color:var(--c-faint); letter-spacing:.02em; }}

/* page-level header in main area */
.page-head {{ margin: 0 0 var(--s4); }}
.page-head .title {{ font-size: var(--fs-2xl); font-weight:700; letter-spacing:-0.015em; }}
.page-head .desc {{ font-size: var(--fs-sm); color: var(--c-muted); margin-top: 2px; }}

/* ---------- metric cards (custom) ---------- */
.metric-card {{
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--r-md); padding: 12px 16px; text-align: left;
  transition: border-color .15s ease;
}}
.metric-card:hover {{ border-color: var(--c-border-strong); }}
.metric-card .label {{
  font-size: var(--fs-xs); color: var(--c-muted); margin-bottom: 4px;
  text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
}}
.metric-card .value {{ font-size: var(--fs-2xl); font-weight: 700; line-height: 1.1; color: var(--c-text); }}

/* native st.metric polish */
[data-testid="stMetric"] {{
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--r-md); padding: 12px 16px;
}}
[data-testid="stMetricLabel"] p {{ font-size: var(--fs-xs) !important;
  text-transform: uppercase; letter-spacing:.06em; color: var(--c-muted) !important; }}

/* ---------- signal badges (flat, tinted, ONE shape) ---------- */
.badge {{
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: var(--fs-xs); font-weight: 600; letter-spacing: .04em;
  border: 1px solid transparent;
}}
.badge-BREAKOUT   {{ background: {t['success_bg']}; color: {t['success']}; border-color: {t['success']}33; }}
.badge-PRE_MARKUP {{ background: {t['info_bg']};    color: {t['info']};    border-color: {t['info']}33; }}
.badge-WATCH      {{ background: {t['warning_bg']}; color: {t['warning']}; border-color: {t['warning']}33; }}
.badge-AVOID      {{ background: {t['danger_bg']};  color: {t['danger']};  border-color: {t['danger']}33; }}
.badge-NONE       {{ background: {t['neutral_bg']}; color: var(--c-faint); }}

/* ---------- status chips (news / fundamental / misc) ---------- */
.chip {{
  display:inline-block; padding:3px 9px; border-radius: var(--r-sm);
  font-size: var(--fs-sm); font-weight:500; line-height:1.4; margin: 2px 0;
  border:1px solid transparent;
}}
.chip-ok    {{ background: {t['success_bg']}; color: {t['success']}; border-color:{t['success']}26; }}
.chip-info  {{ background: {t['info_bg']};    color: {t['info']};    border-color:{t['info']}26; }}
.chip-warn  {{ background: {t['warning_bg']}; color: {t['warning']}; border-color:{t['warning']}26; }}
.chip-muted {{ background: {t['neutral_bg']}; color: var(--c-muted); }}

/* ---------- mode pill + mini stat (sidebar) ---------- */
.mode-pill {{ display:inline-block; border-radius: var(--r-sm); padding:4px 10px;
  font-size: var(--fs-xs); font-weight:600; margin-bottom: var(--s2);
  border:1px solid transparent; }}
.mode-online {{ background: {t['info_bg']};    color:{t['info']};    border-color:{t['info']}33; }}
.mode-local  {{ background: {t['success_bg']}; color:{t['success']}; border-color:{t['success']}33; }}

.mini-stat {{ text-align:center; background: var(--c-surface); border:1px solid var(--c-border);
  border-radius: var(--r-sm); padding: 6px 0; }}
.mini-stat .k {{ font-size:10px; color:var(--c-muted); text-transform:uppercase; letter-spacing:.05em; }}
.mini-stat .v {{ font-size: var(--fs-lg); font-weight:700; }}

/* ---------- theme toggle (segmented control of <a> links) ---------- */
.theme-toggle {{
  display:flex; gap:3px; padding:3px; margin: 2px 0 10px;
  border:1px solid var(--c-border); border-radius: var(--r-md);
  background: var(--c-surface);
}}
.theme-seg {{
  flex:1; text-align:center; text-decoration:none !important;
  font-size: var(--fs-sm); font-weight:600; color: var(--c-muted) !important;
  padding:5px 10px; border-radius: var(--r-sm); line-height:1.4;
  transition: background .12s ease, color .12s ease;
}}
.theme-seg:hover {{ color: var(--c-text) !important; background: var(--c-surface-2); }}
.theme-seg.active {{ background: var(--c-accent-weak); color: var(--c-accent) !important; }}

/* ---------- status panel (deploy widget) ---------- */
.status-panel {{
  background: var(--c-surface); border:1px solid var(--c-border);
  border-radius: var(--r-md); padding: 10px 14px; font-size: var(--fs-sm);
  line-height: 1.7; margin-bottom: var(--s3);
}}
.status-panel b {{ color: var(--c-text); }}
.status-panel .k {{ color: var(--c-muted); }}
.status-panel code {{ background: var(--c-surface-2); color: var(--c-text);
  padding:1px 6px; border-radius: 4px; font-size: 11.5px; }}

/* ---------- tabs: underline style, not chunky pills ---------- */
[data-testid="stTabs"] [role="tablist"] {{
  gap: 2px; border-bottom: 1px solid var(--c-border); margin-bottom: var(--s3);
}}
[data-testid="stTabs"] [role="tablist"] button {{
  font-size: 13.5px; font-weight: 500; color: var(--c-muted);
  padding: 8px 14px 10px; border-radius: 0;
}}
[data-testid="stTabs"] [role="tablist"] button:hover {{ color: var(--c-text); background: transparent; }}
[data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {{
  color: var(--c-text); font-weight: 600;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--c-accent); height: 2px; }}

/* ---------- inputs / selects: subtle borders + clear focus ring ---------- */
[data-baseweb="input"], [data-baseweb="select"] > div, .stTextInput input,
.stNumberInput input, textarea, [data-baseweb="base-input"] {{
  border-radius: var(--r-md) !important;
}}
[data-baseweb="select"] > div, .stTextInput > div > div, .stNumberInput > div > div {{
  border-color: var(--c-border) !important; background: var(--c-surface) !important;
}}
.stTextInput > div > div:focus-within, [data-baseweb="select"] > div:focus-within {{
  border-color: var(--c-accent) !important;
  box-shadow: 0 0 0 2px var(--c-accent-weak) !important;
}}
/* multiselect tags */
[data-baseweb="tag"] {{ background: var(--c-accent-weak) !important; color: var(--c-accent) !important;
  border-radius: var(--r-sm) !important; }}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {{
  border-radius: var(--r-md); border: 1px solid var(--c-border);
  font-weight: 600; font-size: var(--fs-sm); transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--c-accent); color: var(--c-accent); background: var(--c-accent-weak);
}}
.stButton > button[kind="primary"] {{ border-color: var(--c-accent); }}

/* radio group as a segmented control feel (used by theme toggle + view modes) */
[role="radiogroup"] {{ gap: 4px; }}

/* ---------- dataframe / table container ---------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--c-border); border-radius: var(--r-md); overflow: hidden;
}}

/* ---------- expander ---------- */
[data-testid="stExpander"] {{
  border: 1px solid var(--c-border); border-radius: var(--r-md);
  background: var(--c-surface); box-shadow: none;
}}
[data-testid="stExpander"] summary {{ font-size: var(--fs-sm); font-weight: 600; }}
[data-testid="stExpander"] summary:hover {{ color: var(--c-accent); }}

/* ---------- charts get a tidy card frame ---------- */
[data-testid="stPlotlyChart"] {{
  border: 1px solid var(--c-border); border-radius: var(--r-lg);
  padding: 6px 6px 2px; background: var(--c-surface);
}}
[data-testid="stPlotlyChart"] > div {{ border-radius: var(--r-lg); overflow: hidden; }}

/* ---------- alerts: flat, tinted, accent rail ---------- */
[data-testid="stAlert"] {{ border-radius: var(--r-md); border: 1px solid var(--c-border);
  box-shadow: none; }}

/* ---------- accessibility: visible focus everywhere ---------- */
:focus-visible {{ outline: 2px solid var(--c-accent); outline-offset: 2px; border-radius: 3px; }}

/* ---------- restrained scrollbars ---------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: var(--c-border-strong); border-radius: 6px;
  border: 2px solid var(--c-bg); }}
::-webkit-scrollbar-thumb:hover {{ background: var(--c-faint); }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* hide default Streamlit footer for a cleaner product feel */
footer {{ visibility: hidden; }}
</style>
"""


def apply_theme(mode: str | None = None) -> str:
    """Apply the native Streamlit theme + inject the CSS layer for ``mode``.

    Call once near the top of the app (after ``st.set_page_config``). Returns
    the resolved mode so callers can pass it on to charts.
    """
    mode = mode or get_mode()
    for key, val in _NATIVE[mode].items():
        try:
            st.config.set_option(f"theme.{key}", val)
        except Exception:  # noqa: BLE001  (never let theming break the app)
            pass
    st.markdown(_css(mode), unsafe_allow_html=True)
    return mode


def render_theme_toggle(label: str = "Tampilan") -> str:
    """Explicit in-app Light/Dark switch, rendered as a segmented control.

    Each option is a real ``<a href="?theme=...">`` link in the main document,
    so clicking it is an ordinary top-level navigation → a fresh Streamlit
    session reads the query param and applies the native theme deterministically
    (no rerun race, no sandboxed-iframe navigation). Returns the active mode.
    """
    current = get_mode()

    def _seg(mode: str, icon: str, text: str) -> str:
        active = " active" if mode == current else ""
        return (f'<a href="?theme={mode}" target="_self" '
                f'class="theme-seg{active}">{icon} {text}</a>')

    st.markdown(
        '<div class="theme-toggle">'
        + _seg("dark", "🌙", "Gelap")
        + _seg("light", "☀️", "Terang")
        + '</div>',
        unsafe_allow_html=True,
    )
    return current
