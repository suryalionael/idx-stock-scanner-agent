"""IDX Stock Scanner Agent — Streamlit Dashboard.

Jalankan dari root repo:
    streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

# Pastikan root repo ada di sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from dashboard.charts import (
    broker_chart,
    broker_net_flow_chart,
    fundamental_trend_chart,
    history_timeline,
    ihsg_benchmark_chart,
    monthly_holders_chart,
    price_chart,
    price_chart_longterm,
    score_radar,
    set_chart_theme,
    shareholder_pie,
)
from dashboard.theme import (
    apply_theme,
    chart_palette,
    get_mode,
    palette,
    pos_neg_colors,
    render_theme_toggle,
    style_change_table,
    style_perf_table,
    style_table,
)
from dashboard.data_loader import (
    apply_retail_filter,
    available_dates,
    available_dates_unified,
    classify_indexalpha_error,
    fetch_broker_latest,
    fetch_broker_range,
    broker_range_bounds,
    get_table_df,
    is_remote_mode,
    load_all_ranked,
    load_all_tickers_for_date,
    load_all_tickers_unified,
    load_broker_history,
    load_published_payload,
    load_raw,
    last_raw_diag,
    latest_ranked_date,
    load_fundamentals_for_date,
    get_fundamental_row,
    load_news_articles_for_ticker,
    load_ihsg_data,
    get_ihsg_session,
)
from dashboard.explain import explain_signal_llm
from dashboard.ai_lab_view import render_ai_lab_tab
from dashboard.daily_movers_view import render_daily_movers_tab
from dashboard.knowledge_base_view import render_knowledge_base_tab
from dashboard.stock_dictionary_view import render_stock_dictionary_tab
from dashboard.search import (
    format_ticker_option,
    get_search_universe,
    load_ticker_context,
    normalize_ticker,
)
from dashboard.shareholders import (
    get_monthly_shareholder_stats,
    get_shareholder_composition,
)
from stock_scanner.alerts.level_calculator import compute_trading_levels, enrich_df_with_levels
from stock_scanner.pipeline.scalping import enrich_df_with_scalping
from stock_scanner.pipeline.long_term import (
    compute_long_term_score,
    compare_financial_statements,
    classify_cyclicality,
    enrich_df_with_long_term,
)
from stock_scanner.pipeline.smart_money_screener import (
    screen_smart_money,
    load_smart_money_config,
)
from stock_scanner.reference.issuers import get_company_name, get_sector, ticker_display

# ---------------------------------------------------------------------------
# Internal config
# ---------------------------------------------------------------------------

_BROKER_DIR = Path(__file__).parent.parent / "data" / "broker"
_ROOT_PERF = Path(__file__).parent.parent / "data" / "performance"
_INDEXALPHA_HEALTH_PATH = Path(__file__).parent.parent / "data" / "published" / "indexalpha_health.json"


def show_df(data, **kwargs):
    """Theme-correct st.dataframe wrapper.

    Paints the canvas dataframe's cells with the active theme's colours via a
    pandas Styler so the table is light in Light mode and dark in Dark mode,
    regardless of Streamlit's native theme state (fixes the inverted-table bug).
    Falls back to a plain dataframe if styling is not applicable.
    """
    try:
        styled = style_table(data)
    except Exception:  # noqa: BLE001
        styled = data
    st.dataframe(styled, **kwargs)


def _chart_debug(ticker: str) -> None:
    """TEMP deployed diagnostic — shows which OHLCV source served the chart and,
    if empty, exactly why. Auto-expands on failure. Remove once charts verified."""
    d = last_raw_diag(ticker)
    if not d:
        return
    rows = d.get("rows", 0) or 0
    badge = "🟢" if rows else "🔴"
    with st.expander(f"🐞 Chart data debug — {badge} {d.get('source') or 'none'} "
                     f"({rows} rows)", expanded=(rows == 0)):
        st.write({
            "ui_ticker": ticker,
            "source_used": d.get("source"),
            "rows": rows,
            "last_date": d.get("last_date"),
            "columns": d.get("cols"),
            "error": d.get("error"),
            "empty_reason": d.get("reason"),
        })

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IDX Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme / design system
# ---------------------------------------------------------------------------
# Resolve the active mode (session-state driven), apply Streamlit's native
# theme for the canvas widgets + inject the baked CSS design layer, then point
# the Plotly charts at the matching palette. The in-app Light/Dark toggle lives
# in the sidebar (render_theme_toggle).
_THEME_MODE = apply_theme(get_mode())
set_chart_theme(chart_palette(_THEME_MODE))

st.markdown(
    "<style>.company-subtitle{font-size:13px;color:var(--c-faint);"
    "margin-top:-6px;margin-bottom:10px;}</style>",
    unsafe_allow_html=True,
)

_SIG_EMOJI = {
    "BREAKOUT":   "🟢",
    "PRE_MARKUP": "🔵",
    "WATCH":      "🟠",
    "AVOID":      "🔴",
    "NONE":       "⚪",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _badge(signal: str) -> str:
    cls = f"badge-{signal}" if signal in _SIG_EMOJI else "badge-NONE"
    return f'<span class="badge {cls}">{signal}</span>'


def _fmt_score(val) -> str:
    try:
        return f"{float(val):.1f}" if pd.notna(val) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def _color_net_lot(val) -> str:
    """Per-cell CSS for net_lot: green (+), red (-), grey (0/NaN).

    Colours follow the active theme so cells stay legible in light mode too.
    """
    pos, neg, zero = pos_neg_colors()
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return f"color: {pos}; font-weight: 600"
    if v < 0:
        return f"color: {neg}; font-weight: 600"
    return f"color: {zero}"


def _render_data_status_badges(row: pd.Series) -> None:
    """Show compact news + fundamental status badges above the explanation."""
    news_status = str(row.get("news_data_status", "")).lower()
    fund_status = str(row.get("fundamental_status", "")).lower()

    badges: list[str] = []

    # News badge
    if news_status == "ok":
        n = row.get("news_count_3d", 0)
        score = row.get("news_sentiment_score")
        try:
            score_str = f"{float(score):.1f}" if score is not None and pd.notna(score) else "?"
        except (TypeError, ValueError):
            score_str = "?"
        badges.append(
            f'<span class="chip chip-ok">📰 News: {n} artikel · skor {score_str}</span>'
        )
    elif news_status == "none":
        badges.append(
            '<span class="chip chip-muted">📰 News: tidak ada berita 3 hari ini</span>'
        )
    elif news_status == "failed":
        badges.append(
            '<span class="chip chip-warn">⚠️ News: data unavailable today</span>'
        )

    # Fundamental badge
    if fund_status == "ok":
        pe  = row.get("pe_ratio")
        roe = row.get("roe_pct")
        parts = []
        if pe is not None:
            try:
                parts.append(f"PE {float(pe):.1f}x")
            except (TypeError, ValueError):
                pass
        if roe is not None:
            try:
                parts.append(f"ROE {float(roe):.1f}%")
            except (TypeError, ValueError):
                pass
        summary = " · ".join(parts) if parts else "data ok"
        badges.append(
            f'<span class="chip chip-info">📊 Fundamental: {summary}</span>'
        )
    elif fund_status == "partial":
        badges.append(
            '<span class="chip chip-warn">📊 Fundamental: data parsial</span>'
        )
    elif fund_status == "missing":
        badges.append(
            '<span class="chip chip-muted">📊 Fundamental: belum tersedia</span>'
        )

    if badges:
        st.markdown(" &nbsp; ".join(badges), unsafe_allow_html=True)
        st.markdown("")


def _fmt_price(val) -> str:
    """Format a price int/float as a clean integer string, or '-' if zero/None."""
    try:
        v = int(val)
        return f"{v:,}" if v > 0 else "-"
    except (TypeError, ValueError):
        return "-"


def _fmt_price_range(low, high) -> str:
    """Format an entry/TP range as 'low – high' or single value if equal."""
    try:
        l, h = int(low), int(high)
        if l <= 0 and h <= 0:
            return "-"
        if l == h or h <= 0:
            return f"{l:,}" if l > 0 else "-"
        return f"{l:,} – {h:,}"
    except (TypeError, ValueError):
        return "-"


def _get_or_compute_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with trading level columns guaranteed present.

    If columns are already in the DataFrame (from pipeline output), use them.
    If not (e.g., old cached parquet), compute on-the-fly.
    """
    level_cols = ["entry_low", "entry_high", "tp_low", "tp_high", "cutloss", "trade_setup_status"]
    if all(c in df.columns for c in level_cols):
        return df
    # Compute on-the-fly for older data without pre-computed columns
    return enrich_df_with_levels(df.copy())


def render_trading_levels_section(df_all: pd.DataFrame, selected_date: str) -> None:
    """Render the compact Trading Levels table.

    Shows entry / TP / cutloss for actionable signals.
    Toggle allows filtering to BREAKOUT + PRE_MARKUP only, or all active setups.
    """
    st.markdown("### 🎯 Level Trading")

    # Ensure level columns exist
    df_levels = _get_or_compute_levels(df_all)

    # Controls row
    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_left:
        show_watch = st.toggle(
            "Tampilkan WATCH juga",
            value=False,
            help="BREAKOUT & PRE_MARKUP selalu ditampilkan. Toggle ini menambahkan WATCH.",
        )
    with ctrl_right:
        only_active = st.toggle(
            "Sembunyikan inactive",
            value=True,
            help="Sembunyikan AVOID / NONE yang tidak punya setup trading.",
        )

    # Filter
    if only_active:
        priority = ["BREAKOUT", "PRE_MARKUP"]
        if show_watch:
            priority.append("WATCH")
        df_show = df_levels[df_levels["signal"].isin(priority)].copy()
    else:
        df_show = df_levels.copy()

    if df_show.empty:
        st.info("Tidak ada data level trading untuk filter yang dipilih.")
        return

    # Sort: BREAKOUT first, then PRE_MARKUP, then WATCH; within each by total_score desc
    sig_order = {"BREAKOUT": 0, "PRE_MARKUP": 1, "WATCH": 2, "NONE": 3, "AVOID": 4}
    df_show["_sig_rank"] = df_show["signal"].map(sig_order).fillna(5)
    sort_by = ["_sig_rank"]
    if "total_score" in df_show.columns:
        sort_by.append("total_score")
    df_show = df_show.sort_values(sort_by, ascending=[True, False]).drop(columns=["_sig_rank"])

    # Build display DataFrame
    rows = []
    for _, row in df_show.iterrows():
        ticker_clean = str(row.get("ticker", "")).replace(".JK", "")
        signal = str(row.get("signal", ""))
        status = str(row.get("trade_setup_status", "inactive")).lower()

        entry  = _fmt_price_range(row.get("entry_low"),  row.get("entry_high"))
        tp     = _fmt_price_range(row.get("tp_low"),     row.get("tp_high"))
        cl     = _fmt_price(row.get("cutloss"))
        close  = _fmt_price(row.get("close"))
        score  = row.get("total_score")
        score_str = f"{float(score):.1f}" if pd.notna(score) else "-"

        # R:R calculation
        try:
            el = int(row.get("entry_low", 0))
            tl = int(row.get("tp_low",   0))
            cl_val = int(row.get("cutloss", 0))
            if el > cl_val > 0 and tl > el:
                rr = (tl - el) / (el - cl_val)
                rr_str = f"1:{rr:.1f}"
            else:
                rr_str = "-"
        except (TypeError, ValueError):
            rr_str = "-"

        rows.append({
            "Ticker":  ticker_clean,
            "Signal":  signal,
            "Close":   close,
            "Area Entry":     entry if status == "active" else "-",
            "Target Profit":  tp    if status == "active" else "-",
            "Cutloss":        cl    if status == "active" else "-",
            "R:R":            rr_str if status == "active" else "-",
            "Score":          score_str,
        })

    df_disp = pd.DataFrame(rows)

    if df_disp.empty:
        st.info("Tidak ada setup trading yang aktif.")
        return

    # Counts summary
    active_n  = (df_show.get("trade_setup_status") == "active").sum() if "trade_setup_status" in df_show.columns else 0
    brk_n     = (df_show["signal"] == "BREAKOUT").sum()
    pre_n     = (df_show["signal"] == "PRE_MARKUP").sum()
    watch_n   = (df_show["signal"] == "WATCH").sum() if show_watch else 0

    cnt_parts = [f"🟢 BREAKOUT: **{brk_n}**", f"🔵 PRE_MARKUP: **{pre_n}**"]
    if show_watch:
        cnt_parts.append(f"🟠 WATCH: **{watch_n}**")
    st.caption("  ·  ".join(cnt_parts) + f"  ·  Total active: **{active_n}**")

    # Render table
    show_df(
        df_disp,
        use_container_width=True,
        hide_index=True,
        height=min(38 * len(df_disp) + 40, 520),
        column_config={
            "Ticker":        st.column_config.TextColumn("Ticker",         width="small"),
            "Signal":        st.column_config.TextColumn("Signal",         width="small"),
            "Close":         st.column_config.TextColumn("Close (Rp)",     width="small"),
            "Area Entry":    st.column_config.TextColumn("Area Entry (Rp)", width="medium"),
            "Target Profit": st.column_config.TextColumn("Target Profit (Rp)", width="medium"),
            "Cutloss":       st.column_config.TextColumn("Cutloss (Rp)",   width="small"),
            "R:R":           st.column_config.TextColumn("R:R",            width="small"),
            "Score":         st.column_config.TextColumn("Skor",           width="small"),
        },
    )

    # Disclaimer
    st.caption(
        "⚠️ Level dihitung otomatis dari ATR14, MA20, MA50 — bukan rekomendasi investasi. "
        "Selalu verifikasi dengan chart dan lakukan manajemen risiko mandiri."
    )


def _style_broker_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply net_lot color styling with a pandas-version-safe shim.

    pandas ≥ 2.1  : Styler.map()     (applymap removed in 2.2+)
    pandas < 2.1  : Styler.applymap() (legacy)

    Raises AttributeError if neither method exists (should never happen).
    """
    styler = df.style
    if "net_lot" not in df.columns:
        return styler
    # Try new API first, fall back to legacy
    apply_fn = getattr(styler, "map", None) or getattr(styler, "applymap", None)
    if apply_fn is None:
        return styler
    return apply_fn(_color_net_lot, subset=["net_lot"])


# ---------------------------------------------------------------------------
# Shareholder section (reusable)
# ---------------------------------------------------------------------------

def render_shareholders_section(ticker: str, scan_date: str) -> None:
    """Render shareholder composition + monthly holders panel.

    Fully isolated: any exception inside is caught and shown as a friendly
    message — the rest of the detail panel continues to render normally.
    """
    # --- Composition ---
    try:
        sh_left, sh_right = st.columns([1, 1])
        df_comp = get_shareholder_composition(ticker, scan_date)

        with sh_left:
            st.markdown("**Komposisi Kepemilikan**")
            if df_comp.empty:
                st.caption("Data komposisi belum tersedia.")
            else:
                disp = df_comp.copy()
                disp["shares"] = disp["shares"].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
                )
                disp["percentage"] = disp["percentage"].apply(
                    lambda x: f"{float(x):.2f}%" if pd.notna(x) else "N/A"
                )
                show_df(
                    disp,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "category":   st.column_config.TextColumn("Kategori"),
                        "shares":     st.column_config.TextColumn("Saham (lbr)"),
                        "percentage": st.column_config.TextColumn("%"),
                    },
                )

        with sh_right:
            if not df_comp.empty:
                try:
                    st.plotly_chart(
                        shareholder_pie(df_comp, ticker),
                        use_container_width=True,
                        key=f"pie_{ticker}_{scan_date}",
                    )
                except Exception as e:
                    st.caption(f"Chart komposisi tidak dapat dimuat: {e}")

    except Exception as exc:
        st.warning(f"⚠️ Data komposisi pemegang saham tidak tersedia. ({exc})")

    st.markdown("")

    # --- Monthly holders ---
    st.markdown("**Jumlah Pemegang Saham per Bulan**")
    try:
        df_monthly = get_monthly_shareholder_stats(ticker)
        if df_monthly.empty:
            st.caption("Data shareholder bulanan belum tersedia.")
        else:
            # Display table with safe formatting
            disp_m = df_monthly.copy()
            disp_m["shareholder_count"] = disp_m["shareholder_count"].apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
            )
            disp_m["growth_pct"] = disp_m["growth_pct"].apply(
                lambda x: (f"{'+'  if float(x) >= 0 else ''}{float(x):.2f}%")
                if pd.notna(x) else "—"
            )
            show_df(
                disp_m,
                use_container_width=True,
                hide_index=True,
                height=200,
                column_config={
                    "month":             st.column_config.TextColumn("Bulan"),
                    "shareholder_count": st.column_config.TextColumn("Jumlah Holder"),
                    "growth_pct":        st.column_config.TextColumn("Growth MoM"),
                },
            )
            try:
                st.plotly_chart(
                    monthly_holders_chart(df_monthly, ticker),
                    use_container_width=True,
                    key=f"monthly_{ticker}_{scan_date}",
                )
            except Exception as e:
                st.caption(f"Chart bulanan tidak dapat dimuat: {e}")

    except Exception as exc:
        st.warning(f"⚠️ Data shareholder bulanan tidak tersedia. ({exc})")


# ---------------------------------------------------------------------------
# Broker section (reusable, with real Index Alpha data)
# ---------------------------------------------------------------------------

def _format_number(val: float | None, decimals: int = 0) -> str:
    """Format number dengan thousand separator rapi."""
    if val is None or pd.isna(val):
        return "—"
    if decimals == 0:
        return f"{int(val):,}"
    return f"{float(val):,.{decimals}f}"


def _format_rp(val) -> str:
    """Compact Rupiah formatter: T (triliun) / M (miliar) / Jt (juta)."""
    if val is None or pd.isna(val):
        return "—"
    v = float(val)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e12:
        return f"{sign}Rp {a / 1e12:.2f} T"
    if a >= 1e9:
        return f"{sign}Rp {a / 1e9:.2f} M"
    if a >= 1e6:
        return f"{sign}Rp {a / 1e6:.1f} Jt"
    return f"{sign}Rp {a:,.0f}"


def _broker_side_table(df: pd.DataFrame, side: str, view_mode: str, has_value: bool) -> pd.DataFrame:
    """Build a Stockbit-style top-7 buyers/sellers table.

    side      : "buy" | "sell"
    view_mode : "Net" (rank by net_lot) | "Gross" (rank by gross buy/sell lot)
    """
    d = df.copy()
    if view_mode == "Net":
        if side == "buy":
            d = d[d["net_lot"] > 0].sort_values("net_lot", ascending=False)
        else:
            d = d[d["net_lot"] < 0].sort_values("net_lot", ascending=True)
        lot_col, val_col, avg_col = "net_lot", "net_value", None
    else:  # Gross
        if side == "buy":
            d = d.sort_values("buy_lot", ascending=False)
            lot_col, val_col, avg_col = "buy_lot", "buy_value", "buy_avg_price"
        else:
            d = d.sort_values("sell_lot", ascending=False)
            lot_col, val_col, avg_col = "sell_lot", "sell_value", "sell_avg_price"

    d = d.head(7)
    if d.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Kode"] = d["broker_code"].astype(str)
    if "broker_name" in d.columns:
        out["Broker"] = d["broker_name"].astype(str)
    out["Lot"] = d[lot_col].apply(_format_number) if lot_col in d.columns else "—"
    if has_value and val_col in d.columns:
        out["Value"] = d[val_col].apply(_format_rp)
    if avg_col and has_value and avg_col in d.columns:
        out["Avg"] = d[avg_col].apply(lambda x: _format_number(x, decimals=0))
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_broker_latest(ticker: str):
    """Cached (30 min) latest-session fetch — avoids re-hitting the API on every
    Streamlit rerun. Cleared by the Refresh button."""
    return fetch_broker_latest(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_broker_range(ticker: str, from_date: str, to_date: str, investor: str, market: str):
    """Cached (60 min) historical range fetch. Key = all args (ticker+from+to+
    investor+market). Historical aggregates are stable, so a long TTL is fine."""
    return fetch_broker_range(ticker, from_date, to_date, investor=investor, market=market)


def _render_broker_summary_body(df: pd.DataFrame, key_suffix: str, period_label: str | None = None) -> None:
    """Shared Stockbit-style body: Net/Gross toggle, summary, top buyer/seller,
    detail table. Used by both Latest and Historical modes."""
    for col in ["buy_lot", "sell_lot", "net_lot", "buy_value", "sell_value",
                "net_value", "buy_avg_price", "sell_avg_price", "buy_freq", "sell_freq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "net_lot" not in df.columns and {"buy_lot", "sell_lot"} <= set(df.columns):
        df["net_lot"] = df["buy_lot"].fillna(0) - df["sell_lot"].fillna(0)

    has_value = "net_value" in df.columns and df["net_value"].notna().any()

    view_mode = st.radio(
        "Tampilan", ["Net", "Gross"], horizontal=True, key=f"brk_view_{key_suffix}",
        help="Net = net beli/jual per broker. Gross = total beli/jual per broker.",
    )

    total_net_lot = df["net_lot"].fillna(0).sum() if "net_lot" in df.columns else 0.0
    total_net_value = df["net_value"].fillna(0).sum() if has_value else None
    if total_net_lot > 0:
        indikasi = "🟢 Akumulasi"
    elif total_net_lot < 0:
        indikasi = "🔴 Distribusi"
    else:
        indikasi = "⚪ Netral"

    net_label = "Net Lot" + (f" · {period_label}" if period_label else "")
    m1, m2, m3 = st.columns(3)
    m1.metric(net_label, _format_number(total_net_lot))
    m2.metric("Net Value", _format_rp(total_net_value) if total_net_value is not None else "—")
    m3.metric("Indikasi", indikasi)

    st.markdown("")

    col_buy, col_sell = st.columns(2)
    with col_buy:
        st.markdown("**🟢 Top Buyer**")
        tb = _broker_side_table(df, "buy", view_mode, has_value)
        if tb.empty:
            st.caption("—")
        else:
            show_df(tb, use_container_width=True, hide_index=True)
    with col_sell:
        st.markdown("**🔴 Top Seller**")
        ts = _broker_side_table(df, "sell", view_mode, has_value)
        if ts.empty:
            st.caption("—")
        else:
            show_df(ts, use_container_width=True, hide_index=True)

    # Checkbox gate (not st.expander) so this component can be embedded inside an
    # outer expander — e.g. the Broker Summary panels on the Scalping / Smart
    # Money tabs — without triggering Streamlit's nested-expander exception.
    if st.checkbox("📋 Tampilkan semua broker (detail)", key=f"brk_all_{key_suffix}"):
        cols = [c for c in [
            "broker_code", "broker_name", "buy_lot", "sell_lot", "net_lot",
            "buy_value", "sell_value", "net_value", "buy_avg_price", "sell_avg_price",
            "buy_freq", "sell_freq",
        ] if c in df.columns]
        d = (
            df.assign(_abs=lambda x: x["net_lot"].abs())
            .sort_values("_abs", ascending=False)[cols]
            .copy()
        )
        for c in ["buy_lot", "sell_lot", "net_lot"]:
            if c in d.columns:
                d[c] = d[c].apply(_format_number)
        for c in ["buy_value", "sell_value", "net_value"]:
            if c in d.columns:
                d[c] = d[c].apply(_format_rp)
        for c in ["buy_avg_price", "sell_avg_price"]:
            if c in d.columns:
                d[c] = d[c].apply(lambda x: _format_number(x, decimals=0))
        for c in ["buy_freq", "sell_freq"]:
            if c in d.columns:
                d[c] = d[c].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
        show_df(
            d, use_container_width=True, hide_index=True,
            column_config={
                "broker_code": st.column_config.TextColumn("Kode", width="small"),
                "broker_name": st.column_config.TextColumn("Broker", width="medium"),
                "buy_lot": st.column_config.TextColumn("Beli (lot)", width="small"),
                "sell_lot": st.column_config.TextColumn("Jual (lot)", width="small"),
                "net_lot": st.column_config.TextColumn("Net (lot)", width="small"),
                "buy_value": st.column_config.TextColumn("Nilai Beli", width="small"),
                "sell_value": st.column_config.TextColumn("Nilai Jual", width="small"),
                "net_value": st.column_config.TextColumn("Net Value", width="small"),
                "buy_avg_price": st.column_config.TextColumn("Avg Beli", width="small"),
                "sell_avg_price": st.column_config.TextColumn("Avg Jual", width="small"),
                "buy_freq": st.column_config.TextColumn("Freq Beli", width="small"),
                "sell_freq": st.column_config.TextColumn("Freq Jual", width="small"),
            },
        )

    if not has_value:
        st.caption("ℹ️ Data versi ringkas (lot saja) — nilai beli/jual & harga rata-rata "
                   "tidak tersedia di cache ini. Klik **🔄 Refresh** untuk mengambil versi lengkap.")


def _classify_indexalpha_error(err_msg: str) -> str:
    """Rendering wrapper — shared by both Latest and Historical modes so
    they can't drift out of sync again (Historical previously didn't
    classify at all; Latest was missing a 403 branch entirely, silently
    falling through to "click Refresh" — actively bad advice for a
    permission/expired-key failure). The actual classification logic is
    pure and unit-tested in dashboard.data_loader.classify_indexalpha_error
    (this module has real top-level UI code and can't be imported outside a
    live Streamlit context, so the testable logic lives there instead).

    Returns the level ("error"/"warning"/"info") so callers can decide
    whether to lock out further retries — "error" (401/403) is the only
    level that means "retrying will just fail again," per
    fetch_indexalpha.py's own non-retryable-client-error classification."""
    level, message = classify_indexalpha_error(err_msg)
    {"error": st.error, "warning": st.warning, "info": st.info}[level](message)
    return level


def _render_broker_latest(ticker: str, scan_date: str, key_prefix: str = "") -> None:
    """Latest mode: last completed trading session, fresh-first (cache-safe).

    Loads automatically — Index Alpha is now on a paid plan (25,000 req/
    month), so the old free-plan (5 req/day) manual-load gate that used to
    sit here is gone. Quota protection now comes entirely from
    _cached_broker_latest's own 30-minute st.cache_data TTL, which already
    dedupes repeated calls for the same ticker across Streamlit reruns —
    this is what actually made the manual button redundant for that
    concern even before the plan upgrade, not something that needed to be
    rebuilt to allow removing it.

    A confirmed 401/403 still sets a permanent-failure flag (fail_key) that
    disables the Refresh button and skips calling the API again on every
    future rerun for this (ticker, scan_date, key_prefix) — retrying a
    permission failure can only fail again regardless of remaining quota,
    so this stays a real stop, not just advisory copy next to a still-
    clickable button."""
    fail_key = f"brk_permfail_{key_prefix}_{ticker}_{scan_date}"

    top = st.columns([3, 1])
    with top[1]:
        if st.session_state.get(fail_key):
            st.button("🔒 Refresh dinonaktifkan",
                     key=f"brk_lat_refresh_{key_prefix}_{ticker}_{scan_date}", disabled=True,
                     help="API key ditolak (401/403) — retry dinonaktifkan sampai key diperbaiki di secrets panel.")
        elif st.button("🔄 Refresh", key=f"brk_lat_refresh_{key_prefix}_{ticker}_{scan_date}",
                       help="Ambil ulang sesi terbaru dari Index Alpha (memakai 1 kuota)."):
            _cached_broker_latest.clear()

    if st.session_state.get(fail_key):
        # Permanent failure already confirmed this session — re-show the last
        # known error WITHOUT calling the API again.
        _classify_indexalpha_error(st.session_state.get(f"{fail_key}_msg", ""))
        return

    with st.spinner("Memuat sesi terbaru…"):
        df, note, info = _cached_broker_latest(ticker)

    if df is None or df.empty:
        err_msg = note or "Data broker belum tersedia."
        level = _classify_indexalpha_error(err_msg)
        if level == "error":
            st.session_state[fail_key] = True
            st.session_state[f"{fail_key}_msg"] = err_msg
        return

    src = info.get("source")
    badge = {
        "fresh": "🟢 Fresh (Index Alpha)",
        "cache": "🗂️ Cache (sesi terbaru)",
        "fallback": "⚠️ Fallback",
    }.get(src, "")
    st.caption(f"Sumber: Index Alpha · {ticker.replace('.JK', '')} · Sesi {info.get('date')} · {badge}")
    if note:
        st.warning(f"⚠️ {note}")

    # A "fresh" fetch just wrote a brand-new parquet to data/broker/ — but the
    # global Retail Filter (app.py, near st.checkbox("Hide Retail Accumulation"))
    # already ran earlier this script pass, from cache only, before this panel
    # ever executed. Force one rerun so its cache-only read picks up the file
    # that now exists on disk. Guarded per (ticker, session date) so it fires
    # once, not on every rerun — _cached_broker_latest is st.cache_data-memoized,
    # so the replay re-reads the same cached result instead of refetching.
    #
    # Two conditions gate this beyond "fresh", to avoid reruns that can't
    # possibly change anything the user sees:
    #   - hide_retail off → apply_retail_filter() is already a no-op (early
    #     return before touching any parquet), so there's nothing to sync.
    #   - fetched date > scan_date → enrich_df_with_top_brokers()'s fallback
    #     is bounded to dates <= scan_date (see _newest_broker_date_for_ticker),
    #     so a "Latest" fetch for a newer session than the one being viewed
    #     will never be picked up by the filter regardless of rerunning.
    fetched_date = info.get("date")
    hide_retail_on = bool(st.session_state.get("hide_retail_accumulation"))
    if src == "fresh" and hide_retail_on and fetched_date and fetched_date <= scan_date:
        rerun_key = f"brk_fresh_rerun_{key_prefix}_{ticker}_{fetched_date}"
        if not st.session_state.get(rerun_key):
            st.session_state[rerun_key] = True
            st.rerun()

    _render_broker_summary_body(df, key_suffix=f"latest_{key_prefix}_{ticker}_{scan_date}")


def _render_broker_historical(ticker: str, scan_date: str, key_prefix: str = "") -> None:
    """Historical mode: period-aggregated broker summary (1W…1Y / custom).

    Reached only after a user explicitly switches the Latest/Historical
    radio — that click is itself the "explicit user action" gate, so no
    additional load-button is needed here (unlike _render_broker_latest,
    which is the default mode and therefore needed one). A confirmed 401/403
    still disables further Refresh clicks for this (ticker, scan_date,
    key_prefix, range) for the same reason as Latest mode: retrying a
    permission failure can only fail again."""
    from datetime import date as _date

    sel = st.radio(
        "Range", ["1W", "1M", "3M", "6M", "1Y", "Custom"], index=1, horizontal=True,
        key=f"brk_hist_range_{key_prefix}_{ticker}_{scan_date}",
    )

    if sel == "Custom":
        def_from, def_to = broker_range_bounds("1M")
        cc1, cc2 = st.columns(2)
        with cc1:
            d_from = st.date_input("Dari", value=_date.fromisoformat(def_from),
                                   key=f"brk_hist_from_{key_prefix}_{ticker}_{scan_date}")
        with cc2:
            d_to = st.date_input("Sampai", value=_date.fromisoformat(def_to),
                                 key=f"brk_hist_to_{key_prefix}_{ticker}_{scan_date}")
        from_date, to_date = d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d")
        if from_date > to_date:
            st.error("❌ Tanggal 'Dari' harus lebih awal atau sama dengan 'Sampai'.")
            return
    else:
        from_date, to_date = broker_range_bounds(sel)

    hist_fail_key = f"brk_hist_permfail_{key_prefix}_{ticker}_{scan_date}_{sel}"

    cap, ref = st.columns([3, 1])
    with cap:
        st.caption(f"Periode: **{from_date} → {to_date}** · agregat Index Alpha (RG sejak Jun 2025)")
    with ref:
        if st.session_state.get(hist_fail_key):
            st.button("🔒 Refresh dinonaktifkan",
                     key=f"brk_hist_refresh_{key_prefix}_{ticker}_{scan_date}", disabled=True,
                     help="API key ditolak (401/403) — retry dinonaktifkan sampai key diperbaiki di secrets panel.")
        elif st.button("🔄 Refresh", key=f"brk_hist_refresh_{key_prefix}_{ticker}_{scan_date}",
                       help="Ambil ulang periode ini dari Index Alpha (memakai 1 kuota)."):
            _cached_broker_range.clear()

    if st.session_state.get(hist_fail_key):
        _classify_indexalpha_error(st.session_state.get(f"{hist_fail_key}_msg", ""))
        return

    with st.spinner("Memuat broker summary historical…"):
        df, err = _cached_broker_range(ticker, from_date, to_date, "all", "RG")

    if df is None or df.empty:
        err_msg = err or f"Belum ada Broker Summary untuk {ticker.replace('.JK','')} pada periode ini."
        level = _classify_indexalpha_error(err_msg)
        if level == "error":
            st.session_state[hist_fail_key] = True
            st.session_state[f"{hist_fail_key}_msg"] = err_msg
        return
    if err:
        st.warning(f"⚠️ {err}")

    _render_broker_summary_body(
        df, key_suffix=f"hist_{key_prefix}_{ticker}_{scan_date}",
        period_label=(sel if sel != "Custom" else "periode"),
    )


def _render_indexalpha_integration_badge() -> None:
    """Integration-level health (separate from the per-session fresh/cache/
    fallback badge already shown by _render_broker_latest). This answers
    "is the IndexAlpha connection itself trustworthy right now", not "is
    this specific session's data fresh" — both must be visible, neither
    should be inferred from the other. Reads the local health-state file
    written by fetch_indexalpha._get() on every real call; never makes a
    network call itself (zero quota cost)."""
    import json as _json
    import os as _os
    from datetime import datetime as _dt, timezone as _tz

    key_set = bool(_os.environ.get("INDEX_ALPHA_API_KEY", "").strip())
    if not key_set:
        try:
            import streamlit as _st_key
            _sk_val = _st_key.secrets.get("INDEX_ALPHA_API_KEY", "")
            key_set = bool(_sk_val)
        except Exception:
            pass
    state = {}
    if _INDEXALPHA_HEALTH_PATH.exists():
        try:
            state = _json.loads(_INDEXALPHA_HEALTH_PATH.read_text())
        except Exception:  # noqa: BLE001
            state = {}

    if not key_set and not state:
        st.warning(
            "⚠️ **IndexAlpha API belum pernah terhubung di environment ini** "
            "(API key tidak diset, belum ada riwayat panggilan tersimpan). "
            "Data broker di bawah — jika ada — berasal dari cache lama, bukan sesi live."
        )
        return

    consec_fail = state.get("consecutive_failures", 0)
    last_success = state.get("last_success_at")
    last_error_type = state.get("last_error_type")
    last_status_code = state.get("last_status_code")

    # ── Specific error messages ────────────────────────────────────────
    if last_error_type == "missing_key":
        st.warning(
            "⚠️ **IndexAlpha: API key tidak ditemukan.** "
            "Data broker hanya dari cache. "
            "Set INDEX_ALPHA_API_KEY di environment untuk fetch sesi baru."
        )
        return
    if last_error_type == "auth_error":
        st.error(
            "❌ **IndexAlpha: Autentikasi gagal (401).** "
            "Periksa INDEX_ALPHA_API_KEY di environment/Streamlit secrets."
        )
        return
    if last_error_type == "forbidden":
        st.error(
            "❌ **IndexAlpha: Akses ditolak (403).** "
            "API key mungkin tidak memiliki akses ke endpoint ini."
        )
        return
    if last_error_type in ("rate_limit", "rate_limit_exhausted"):
        st.warning(
            "⚠️ **IndexAlpha: Rate limit / kuota bulanan habis.** "
            "Tunggu beberapa saat lalu coba lagi. "
            "Data cache masih tersedia untuk sesi yang sudah di-fetch sebelumnya."
        )
        return
    if last_error_type == "timeout" and consec_fail >= 3:
        st.warning(
            "⚠️ **IndexAlpha: Timeout berulang.** "
            "Server lambat atau jaringan tidak stabil. "
            "Data broker dari cache akan ditampilkan dulu."
        )
        return
    if last_error_type == "connection_error" and consec_fail >= 3:
        st.warning(
            "⚠️ **IndexAlpha: Koneksi gagal.** "
            "api.indexalpha.id tidak dapat dijangkau. "
            "Data broker dari cache akan ditampilkan dulu."
        )
        return
    if last_error_type == "logical_failure":
        st.warning(
            f"⚠️ **IndexAlpha: Response tidak sesuai** (HTTP {last_status_code}). "
            "Endpoint mungkin berubah. Data cache akan ditampilkan."
        )
        return

    # ── Generic consecutive failure ────────────────────────────────────
    if consec_fail and consec_fail >= 3:
        code_str = f" (HTTP {last_status_code})" if last_status_code else ""
        err_str = f" — {last_error_type}" if last_error_type else ""
        st.warning(
            f"⚠️ **IndexAlpha: {consec_fail} kegagalan berturut-turut{err_str}{code_str}.** "
            f"Terakhir sukses: {last_success or 'tidak pernah'}. "
            "Data broker dari cache masih ditampilkan."
        )
        return

    if not key_set:
        st.caption("ℹ️ INDEX_ALPHA_API_KEY tidak diset di environment ini — "
                   "menampilkan cache yang ada, tidak akan fetch sesi baru.")
    elif last_success:
        try:
            age_h = (_dt.now(_tz.utc) - _dt.fromisoformat(last_success)).total_seconds() / 3600
            st.caption(f"✅ IndexAlpha terverifikasi — sukses terakhir {age_h:.0f} jam lalu.")
        except Exception:  # noqa: BLE001
            st.caption(f"✅ IndexAlpha sukses terakhir: {last_success}.")


def render_broker_section(ticker: str, scan_date: str, key_prefix: str = "",
                          show_header: bool = True) -> None:
    """Stockbit-style Broker Summary — inline on the stock's own detail page.

    Two modes (default Latest), both real data from the Index Alpha API:
      • Latest     — last completed trading session, fresh-first (cache-safe).
      • Historical — period aggregate (1W/1M/3M/6M/1Y/Custom) via from/to.

    ``key_prefix`` namespaces every widget key so the same Broker Summary can be
    rendered on several tabs (Swing / Scalping / Smart Money / Search) at once
    without duplicate-key collisions. ``show_header`` hides the section title when
    embedded inside an expander that already labels it.
    """
    if show_header:
        st.markdown("#### 🏦 Broker Summary")
        st.caption("Rincian kepemilikan & serapan broker (data real Index Alpha). "
                   "Harga, volume, dan fundamental saham ini ada di tab lain.")
    _render_indexalpha_integration_badge()
    mode = st.radio(
        "Mode broker", ["Latest", "Historical"], horizontal=True,
        key=f"brk_main_mode_{key_prefix}_{ticker}_{scan_date}", label_visibility="collapsed",
    )
    if mode == "Latest":
        _render_broker_latest(ticker, scan_date, key_prefix)
    else:
        _render_broker_historical(ticker, scan_date, key_prefix)


# ---------------------------------------------------------------------------
# Full ticker detail panel (used by Today Overview + Search)
# ---------------------------------------------------------------------------

def render_ticker_detail(
    row: pd.Series,
    scan_date: str,
    api_key: str | None,
    key_prefix: str = "",
) -> None:
    """Render the complete detail panel for one ticker.

    Uses inner tabs:  📈 Sinyal & Chart  |  🏢 Shareholders  |  📋 Broker
    """
    signal = str(row.get("signal", "NONE"))
    ticker = str(row.get("ticker", "?"))
    company_name = get_company_name(ticker)
    sector = get_sector(ticker)

    badge_cls = f"badge-{signal}" if signal in _SIG_EMOJI else "badge-NONE"
    name_extra = f" &nbsp;<small style='color:#64748b'>({sector})</small>" if sector else ""
    st.markdown(
        f'<h4>{ticker} &nbsp; <span class="badge {badge_cls}">{signal}</span></h4>'
        f'<div class="company-subtitle">{company_name}{name_extra}</div>',
        unsafe_allow_html=True,
    )

    tab_signals, tab_shareholders, tab_broker = st.tabs(
        ["📈 Sinyal & Chart", "🏢 Shareholders", "📋 Broker Activity"]
    )

    # ---- TAB: Sinyal & Chart ----
    with tab_signals:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("**Skor Komponen**")
            score_items = [
                ("Trend",         "trend_score"),
                ("Momentum",      "momentum_score"),
                ("Breakout",      "breakout_score"),
                ("Volume",        "volume_score"),
                ("Penalty",       "penalty_score"),
                ("**Total**",     "total_score"),
                ("**Enhanced**",  "enhanced_total_score"),
                ("News",          "news_score"),
                ("Foreign Flow",  "foreign_score"),
            ]
            for label, col in score_items:
                val = row.get(col)
                st.markdown(f"{label}: `{_fmt_score(val)}/10`")

            st.markdown("")
            st.markdown("**Metrik Teknikal**")
            num_metrics = [
                ("Close",            "close",              "Rp {:.0f}"),
                ("RSI14",            "rsi14",              "{:.1f}"),
                ("ADX",              "adx",                "{:.1f}"),
                ("Vol Ratio 20d",    "vol_ratio_20d",      "{:.2f}x"),
                ("% dari 52w High",  "pct_from_52w_high",  "{:.1f}%"),
                ("News Score",       "news_sentiment_score", "{:.2f}"),
                ("Berita 3d",        "news_count_3d",       "{:.0f}"),
            ]
            for label, col, fmt in num_metrics:
                val = row.get(col)
                try:
                    val_str = fmt.format(float(val)) if val is not None and pd.notna(val) else "N/A"
                except (ValueError, TypeError):
                    val_str = "N/A"
                st.text(f"{label}: {val_str}")

            for label, col in [
                ("Supertrend Bullish", "supertrend_bullish"),
                ("Squeeze On",         "squeeze_on"),
                ("ATR Breakout",       "atr_breakout"),
                ("Vol Spike",          "vol_spike"),
                ("OBV Trend Up",       "obv_trend"),
            ]:
                val = row.get(col)
                st.text(f"{label}: {'✅' if val else '—'}")

        with right:
            st.markdown("**Skor Visual**")
            st.plotly_chart(
                score_radar(row, ticker),
                use_container_width=True,
                key=f"{key_prefix}radar_{ticker}_{scan_date}",
            )

            # ── Trading levels mini card ──────────────────────────
            levels = compute_trading_levels(row)
            if levels.get("trade_setup_status") == "active":
                st.markdown("")
                st.markdown("**🎯 Level Trading**")
                lvl_data = {
                    "": ["Area Entry", "Target Profit", "Cutloss"],
                    "Harga (Rp)": [
                        _fmt_price_range(levels["entry_low"], levels["entry_high"]),
                        _fmt_price_range(levels["tp_low"],    levels["tp_high"]),
                        _fmt_price(levels["cutloss"]),
                    ],
                }
                show_df(
                    pd.DataFrame(lvl_data),
                    use_container_width=True,
                    hide_index=True,
                    height=145,
                )
                try:
                    el, cl_v = levels["entry_low"], levels["cutloss"]
                    tl = levels["tp_low"]
                    if el > cl_v > 0 and tl > el:
                        rr = (tl - el) / (el - cl_v)
                        st.caption(f"Risk/Reward ≈ 1:{rr:.1f}")
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

        st.markdown("**Chart Harga (120 hari terakhir)**")
        df_raw = load_raw(ticker)
        st.plotly_chart(
            price_chart(df_raw, ticker, signal_date=scan_date),
            use_container_width=True,
            key=f"{key_prefix}chart_{ticker}_{scan_date}",
        )
        _chart_debug(ticker)

        # ---- Status badges (news & fundamental) ----
        _render_data_status_badges(row)

        st.markdown("**Penjelasan Sinyal**")
        with st.spinner("Membuat penjelasan..."):
            # Load per-article data for narrative bullets (cached by scan_date)
            articles = load_news_articles_for_ticker(ticker, scan_date)
            explanation = explain_signal_llm(row, api_key=api_key, articles=articles)
        st.markdown(explanation)

    # ---- TAB: Shareholders ----
    with tab_shareholders:
        render_shareholders_section(ticker, scan_date)

    # ---- TAB: Broker ----
    with tab_broker:
        render_broker_section(ticker, scan_date)


# ---------------------------------------------------------------------------
# Data preparation helpers (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _prepare_scalping_df(scan_date: str, df_preloaded: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load signals and enrich with scalping scores. Cached 5 min.

    Jika df_preloaded sudah tersedia (misalnya sudah di-load dari published
    payload atau local file oleh caller), gunakan langsung tanpa I/O ulang.
    Ini memastikan Scalping tab bekerja di remote mode (Streamlit Cloud).
    """
    if df_preloaded is not None and not df_preloaded.empty:
        df = df_preloaded.copy()
    else:
        df = load_all_tickers_for_date(scan_date)
    if df.empty:
        return df
    return enrich_df_with_scalping(df)


@st.cache_data(ttl=300, show_spinner=False)
def _prepare_longterm_df(scan_date: str, df_preloaded: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load signals and enrich with long-term scores. Cached 5 min.

    Jika df_preloaded sudah tersedia (misalnya sudah di-load dari published
    payload atau local file oleh caller), gunakan langsung tanpa I/O ulang.
    Ini memastikan Long Term tab bekerja di remote mode (Streamlit Cloud).
    """
    if df_preloaded is not None and not df_preloaded.empty:
        df = df_preloaded.copy()
    else:
        df = load_all_tickers_for_date(scan_date)
    if df.empty:
        return df
    # Build sector map from issuers reference
    from stock_scanner.reference.issuers import get_sector as _get_sector
    sector_map = {row["ticker"]: _get_sector(row["ticker"])
                  for _, row in df.iterrows()}
    return enrich_df_with_long_term(df, sector_map=sector_map)


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_financial_comparison(ticker: str) -> dict:
    """Lazy-load multi-period financials. Cached 10 min."""
    return compare_financial_statements(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_broker_intelligence(ticker: str, scan_date: str = "") -> dict:
    """Load broker history and compute accumulation intelligence. Cached 5 min.

    Reads ONLY from real cache files (data/broker/{ticker}_{date}.parquet).
    If insufficient real cache data exists, returns inactive/zero result.
    No mock fallback — dashboard shows empty state when data unavailable.

    Returns dict with:
        broker_accumulation_label, broker_accumulation_score, foreign_net_buy_*,
        big_broker_net_buy_*, top_buyer_brokers, top_seller_brokers, strengths, red_flags
    """
    from stock_scanner.pipeline.broker_intelligence import compute_broker_intelligence

    # Load real multi-day broker history from cache only
    broker_df = load_broker_history(ticker, n_days=20)

    # compute_broker_intelligence returns _INACTIVE_RESULT (all zeros/empty) if broker_df empty
    return compute_broker_intelligence(ticker, broker_df)


# ---------------------------------------------------------------------------
# Scalping Tab
# ---------------------------------------------------------------------------

def render_scalping_tab(df_all: pd.DataFrame, scan_date: str, api_key: str | None) -> None:
    """Render the 📈 Scalping tab content."""
    st.markdown("### 📈 Scalping — Kandidat Momentum Harian")
    st.caption(
        "Filter saham berpotensi top-gainer berdasarkan volume spike, momentum harga, "
        "dan ATR breakout dari data harian. ⚠️ Tidak ada data intraday — semua berbasis OHLCV daily."
    )

    # Enrich with scalping scores.
    # df_all sudah berisi semua ticker (local: dari signals parquet;
    # remote: dari all_tickers section di published JSON).
    with st.spinner("Menghitung scalping score..."):
        df_scalp = _prepare_scalping_df(scan_date, df_preloaded=df_all)

    if df_scalp.empty or "scalping_score" not in df_scalp.columns:
        st.warning("Tidak ada data scalping tersedia.")
        return

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        show_label = st.multiselect(
            "Filter label",
            options=["SCALPING_HIGH", "SCALPING_WATCH", "NOT_SCALPING"],
            default=["SCALPING_HIGH", "SCALPING_WATCH"],
            key="scalp_label_filter",
        )
    with ctrl2:
        min_vol = st.slider("Min Volume Ratio", 0.0, 5.0, 1.5, 0.5, key="scalp_vol_filter")
    with ctrl3:
        top_n = st.number_input("Top N", 5, 50, 15, 5, key="scalp_top_n")

    # Filter + sort
    df_show = df_scalp.copy()
    if show_label:
        df_show = df_show[df_show["scalping_label"].isin(show_label)]
    if min_vol > 0 and "vol_ratio_20d" in df_show.columns:
        df_show = df_show[pd.to_numeric(df_show["vol_ratio_20d"], errors="coerce").fillna(0) >= min_vol]
    df_show = df_show.sort_values("scalping_score", ascending=False).head(int(top_n))

    # Summary counts
    high_n  = (df_scalp["scalping_label"] == "SCALPING_HIGH").sum()
    watch_n = (df_scalp["scalping_label"] == "SCALPING_WATCH").sum()
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("🔥 SCALPING_HIGH", high_n)
    with s2:
        st.metric("👀 SCALPING_WATCH", watch_n)
    with s3:
        st.metric("Ditampilkan", len(df_show))

    if df_show.empty:
        st.info("Tidak ada kandidat scalping dengan filter ini.")
        return

    # Table
    from dashboard.data_loader import enrich_df_with_top_brokers
    df_show = enrich_df_with_top_brokers(df_show, scan_date)
    tbl_cols = ["ticker", "top_buyer", "top_seller",
                "scalping_score", "scalping_label", "scalping_reason",
                "close", "vol_ratio_20d", "roc5", "rsi14", "momentum_score",
                "atr_breakout", "vol_spike", "entry_low", "entry_high", "tp_low", "cutloss"]
    disp_cols = [c for c in tbl_cols if c in df_show.columns]
    tbl = df_show[disp_cols].copy()

    # Flag extended / overheating momentum (ROC5 > threshold) — visible, NOT excluded.
    from stock_scanner.pipeline.scalping import ROC5_OVERHEATED_PCT
    _n_overheated = 0
    if "roc5" in df_show.columns:
        _over = df_show["roc5"].apply(
            lambda x: "🔥 Extended" if (pd.notna(x) and float(x) > ROC5_OVERHEATED_PCT) else "")
        _n_overheated = int((_over != "").sum())
        tbl.insert(1, "Status", _over)

    # Format
    for col in ["close", "entry_low", "entry_high", "tp_low", "cutloss"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "-")
    for col in ["vol_ratio_20d", "roc5", "rsi14", "momentum_score", "scalping_score"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "-")

    show_df(
        tbl,
        use_container_width=True,
        hide_index=True,
        height=min(38 * len(tbl) + 40, 500),
        column_config={
            "ticker":          st.column_config.TextColumn("Ticker",         width="small"),
            "top_buyer":       st.column_config.TextColumn("Top Buyer",      width="small"),
            "top_seller":      st.column_config.TextColumn("Top Seller",     width="small"),
            "Status":          st.column_config.TextColumn("Status",         width="small"),
            "scalping_score":  st.column_config.TextColumn("Scalp Score",    width="small"),
            "scalping_label":  st.column_config.TextColumn("Label",          width="medium"),
            "scalping_reason": st.column_config.TextColumn("Alasan",         width="large"),
            "close":           st.column_config.TextColumn("Close (Rp)",     width="small"),
            "vol_ratio_20d":   st.column_config.TextColumn("Vol Ratio",      width="small"),
            "roc5":            st.column_config.TextColumn("ROC5 (%)",        width="small"),
            "rsi14":           st.column_config.TextColumn("RSI14",          width="small"),
            "momentum_score":  st.column_config.TextColumn("Mom Score",      width="small"),
            "atr_breakout":    st.column_config.CheckboxColumn("ATR Brk"),
            "vol_spike":       st.column_config.CheckboxColumn("Vol Spike"),
            "entry_low":       st.column_config.TextColumn("Entry",          width="small"),
            "entry_high":      st.column_config.TextColumn("Entry Hi",       width="small"),
            "tp_low":          st.column_config.TextColumn("TP",             width="small"),
            "cutloss":         st.column_config.TextColumn("CL",             width="small"),
        },
    )

    if _n_overheated:
        st.warning(
            f"🔥 **{_n_overheated} saham Extended** (ROC5 > {int(ROC5_OVERHEATED_PCT)}%) — "
            "momentum sudah jauh/overheated. Tetap ditampilkan, tapi risiko entry telat tinggi.",
            icon="🔥",
        )

    st.caption(
        "⚠️ Level trading dihitung dari ATR daily — bukan untuk scalping tick-by-tick. "
        "Selalu gunakan real-time chart dan order flow sebelum entry."
    )

    # Expandable detail per ticker
    st.divider()
    st.markdown("#### Detail Ticker Scalping")
    ticker_list = df_show["ticker"].tolist()
    if not ticker_list:
        return

    sel_scalp = st.selectbox(
        "Pilih ticker untuk detail:",
        options=ticker_list,
        format_func=lambda t: f"{t.replace('.JK','')} — {get_company_name(t)}",
        key="scalp_detail_select",
    )
    if sel_scalp:
        row = df_show[df_show["ticker"] == sel_scalp].iloc[0]
        _render_scalping_detail(row, scan_date, api_key)


def _render_scalping_detail(row: pd.Series, scan_date: str, api_key: str | None) -> None:
    """Render detail panel for one scalping candidate."""
    ticker = str(row.get("ticker", "?"))
    scalp_score = row.get("scalping_score")
    scalp_reason = str(row.get("scalping_reason", "—"))

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        # Scalping-focused metrics
        st.markdown("**🔥 Scalping Metrics**")
        scalp_metrics = [
            ("Scalping Score",  f"{float(scalp_score):.1f}/10" if pd.notna(scalp_score) else "—"),
            ("Volume Ratio",    f"{float(row.get('vol_ratio_20d', 0)):.1f}×"
                                if pd.notna(row.get('vol_ratio_20d')) else "—"),
            ("ROC5 (5d return)", f"+{float(row.get('roc5', 0)):.1f}%"
                                 if pd.notna(row.get('roc5')) else "—"),
            ("RSI14",           f"{float(row.get('rsi14', 0)):.0f}"
                                if pd.notna(row.get('rsi14')) else "—"),
            ("Momentum Score",  f"{float(row.get('momentum_score', 0)):.1f}/10"
                                if pd.notna(row.get('momentum_score')) else "—"),
            ("ADX",             f"{float(row.get('adx', 0)):.1f}"
                                if pd.notna(row.get('adx')) else "—"),
            ("Close",           f"Rp{int(row.get('close', 0)):,}"
                                if pd.notna(row.get('close')) else "—"),
        ]
        for label, val in scalp_metrics:
            st.markdown(f"`{label}`: **{val}**")

        st.markdown(f"\n**Alasan**: {scalp_reason}")

        # Flags
        flags = []
        if _bool_val(row.get("vol_spike")):     flags.append("Volume Spike 🔥")
        if _bool_val(row.get("atr_breakout")):  flags.append("ATR Breakout 🚀")
        if _bool_val(row.get("supertrend_bullish")): flags.append("Supertrend ✅")
        if _bool_val(row.get("squeeze_on")):    flags.append("Squeeze On 🔋")
        if flags:
            st.markdown("**Konfirmasi**: " + "  |  ".join(flags))

        # Trading levels
        levels = compute_trading_levels(row)
        if levels.get("trade_setup_status") == "active":
            st.markdown("")
            st.markdown("**🎯 Level Trading (Daily Approximation)**")
            level_rows = {
                "": ["Area Entry", "Target Profit", "Cutloss"],
                "Rp": [
                    _fmt_price_range(levels["entry_low"], levels["entry_high"]),
                    _fmt_price_range(levels["tp_low"],    levels["tp_high"]),
                    _fmt_price(levels["cutloss"]),
                ],
            }
            show_df(pd.DataFrame(level_rows), use_container_width=True,
                         hide_index=True, height=145)

    with col_right:
        # Price chart
        df_raw = load_raw(ticker)
        st.plotly_chart(
            price_chart(df_raw, ticker, signal_date=scan_date, lookback_days=60),
            use_container_width=True,
            key=f"scalp_chart_{ticker}_{scan_date}",
        )
        _chart_debug(ticker)

    # News (catalyst) — compact
    news_status = str(row.get("news_data_status", "")).lower()
    if news_status == "ok":
        articles = load_news_articles_for_ticker(ticker, scan_date)
        if articles:
            st.markdown("**📰 Katalis Berita**")
            from stock_scanner.pipeline.news_summarizer import summarize_news_articles, format_news_bullets
            summary = summarize_news_articles(articles)
            bullets = format_news_bullets(summary, len(articles),
                                         sentiment_score=row.get("news_sentiment_score"),
                                         max_chars=500)
            st.markdown(bullets.replace("**", "**"), unsafe_allow_html=False)

    # Broker Summary (kepemilikan & serapan) — collapsed to keep the scalping
    # momentum view uncluttered; opens on demand.
    st.divider()
    with st.expander("🏦 Broker Summary — kepemilikan & serapan broker"):
        render_broker_section(ticker, scan_date, key_prefix="scalp", show_header=False)


# ---------------------------------------------------------------------------
# Swing Tab
# ---------------------------------------------------------------------------

def render_swing_tab(df_all: pd.DataFrame, scan_date: str, api_key: str | None,
                     signal_filter: list[str]) -> None:
    """Render the 🔄 Swing Trading tab content — refactored Today Overview."""
    st.markdown("### 🔄 Swing Trading — Setup Teknikal 3–10 Hari")

    # ── Production diagnostics (temporary) ─────────────────────────────
    _diag_branch, _diag_sha = _git_commit_info()
    _diag_broker_dir = Path(__file__).parent.parent / "data" / "broker"
    _diag_n_broker_files = len(list(_diag_broker_dir.glob("*.parquet"))) if _diag_broker_dir.exists() else -1
    _diag_broker_dir_tracked = _diag_broker_dir.exists() and bool(list(_diag_broker_dir.glob("*.parquet")))
    _diag_os_key = "✅ SET" if os.environ.get("INDEX_ALPHA_API_KEY", "").strip() else "❌ NOT SET"
    _diag_ss_key = "N/A"
    try:
        import streamlit as _st_diag
        _sk = _st_diag.secrets.get("INDEX_ALPHA_API_KEY", "")
        _diag_ss_key = "✅ SET" if _sk else "❌ NOT SET"
    except Exception:
        _diag_ss_key = "⚠️ st.secrets not available"
    _diag_health_path = Path(__file__).parent.parent / "data" / "published" / "indexalpha_health.json"
    _diag_health = "NOT FOUND"
    if _diag_health_path.exists():
        try:
            import json
            _h = json.loads(_diag_health_path.read_text())
            _diag_health = (
                f"consec_fail={_h.get('consecutive_failures', '?')} "
                f"last_err={_h.get('last_error_type', 'none')} "
                f"last_status={_h.get('last_status_code', '?')} "
                f"last_detail={_h.get('last_error_detail', '')} "
                f"last_success={_h.get('last_success_at', 'never')} "
                f"total_calls={_h.get('total_calls', 0)}"
            )
        except Exception as _e:
            _diag_health = f"read error: {_e}"
    with st.expander("🔍 DIAGNOSTIC (production deploy check)", expanded=False):
        st.code(
            f"commit={_diag_sha}  branch={_diag_branch}\n"
            f"scan_date={scan_date}  df_all.shape={df_all.shape}\n"
            f"os.environ[INDEX_ALPHA_API_KEY] = {_diag_os_key}\n"
            f"st.secrets[INDEX_ALPHA_API_KEY] = {_diag_ss_key}\n"
            f"data/broker/ dir_exists={_diag_broker_dir.exists()}  "
            f"parquet_count={_diag_n_broker_files}\n"
            f"indexalpha_health.json = {_diag_health}\n"
            f"df_all.columns[:10] = {list(df_all.columns[:10])}\n"
            f"top_buyer in df_all = {'top_buyer' in df_all.columns}\n"
            f"top_seller in df_all = {'top_seller' in df_all.columns}"
        )

    # Signal distribution
    sig_counts_all = df_all["signal"].value_counts() if "signal" in df_all.columns else pd.Series(dtype=int)
    total_tickers  = len(df_all)

    _pal = palette()
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, key, color in [
        (c1, "Total",      None,         _pal["muted"]),
        (c2, "BREAKOUT",   "BREAKOUT",   _pal["success"]),
        (c3, "PRE_MARKUP", "PRE_MARKUP", _pal["info"]),
        (c4, "WATCH",      "WATCH",      _pal["warning"]),
        (c5, "AVOID",      "AVOID",      _pal["danger"]),
    ]:
        count = total_tickers if key is None else int(sig_counts_all.get(key, 0))
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{count}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Filtered signal table
    if "signal" in df_all.columns and signal_filter:
        df_filtered = df_all[df_all["signal"].isin(signal_filter)].copy()
    else:
        df_filtered = df_all.copy()

    if not df_filtered.empty:
        df_table = get_table_df(df_filtered, scan_date=scan_date)
        display = df_table.copy()
        for col in ["total_score", "enhanced_total_score", "trend_score", "momentum_score",
                    "breakout_score", "volume_score", "penalty_score", "news_score", "foreign_score"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        for col in ["rsi14", "vol_ratio_20d", "pct_from_52w_high", "adx"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        show_df(
            display,
            use_container_width=True, hide_index=True, height=320,
            column_config={
                "ticker":               st.column_config.TextColumn("Ticker",      width="small"),
                "top_buyer":            st.column_config.TextColumn("Top Buyer",   width="small"),
                "top_seller":           st.column_config.TextColumn("Top Seller",  width="small"),
                "signal":               st.column_config.TextColumn("Signal",      width="small"),
                "total_score":          st.column_config.TextColumn("Score",       width="small"),
                "enhanced_total_score": st.column_config.TextColumn("Enh.Score",  width="small"),
                "close":                st.column_config.NumberColumn("Close",     format="%.0f"),
                "rsi14":                st.column_config.TextColumn("RSI14",       width="small"),
                "adx":                  st.column_config.TextColumn("ADX",         width="small"),
                "vol_ratio_20d":        st.column_config.TextColumn("Vol Ratio",   width="small"),
                "pct_from_52w_high":    st.column_config.TextColumn("52w High%",   width="small"),
                "supertrend_bullish":   st.column_config.CheckboxColumn("Supertrend"),
                "squeeze_on":           st.column_config.CheckboxColumn("Squeeze"),
                "atr_breakout":         st.column_config.CheckboxColumn("ATR Brk"),
                "vol_spike":            st.column_config.CheckboxColumn("Vol Spike"),
            },
        )
    else:
        st.info("Tidak ada ticker dengan signal yang dipilih.")

    # Trading levels
    st.divider()
    render_trading_levels_section(df_filtered if not df_filtered.empty else df_all, scan_date)


# ---------------------------------------------------------------------------
# Long Term Tab
# ---------------------------------------------------------------------------

def render_longterm_tab(df_all: pd.DataFrame, scan_date: str, api_key: str | None) -> None:
    """Render the 📊 Long Term Investment tab content."""
    st.markdown("### 📊 Long Term — Investasi Berbasis Kualitas Bisnis & Valuasi")
    st.caption(
        "Fokus pada ROE, DER, pertumbuhan laba, valuasi intrinsik, dan margin of safety. "
        "Fundamental dominan; teknikal hanya sebagai timing helper."
    )

    # Enrich with long-term scores.
    # df_all sudah berisi semua ticker (local: dari signals parquet;
    # remote: dari all_tickers section di published JSON).
    with st.spinner("Menghitung long-term score..."):
        df_lt = _prepare_longterm_df(scan_date, df_preloaded=df_all)

    if df_lt.empty or "long_term_score" not in df_lt.columns:
        st.warning("Tidak ada data long-term tersedia.")
        return

    # Controls
    lt_c1, lt_c2, lt_c3 = st.columns([2, 2, 1])
    with lt_c1:
        lt_label_filter = st.multiselect(
            "Filter label",
            options=["LONG_TERM_CORE", "LONG_TERM_WATCHLIST", "NOT_LONG_TERM"],
            default=["LONG_TERM_CORE", "LONG_TERM_WATCHLIST"],
            key="lt_label_filter",
        )
    with lt_c2:
        fund_status_filter = st.multiselect(
            "Filter data fundamental",
            options=["ok", "partial", "missing"],
            default=["ok"],
            key="lt_fund_filter",
        )
    with lt_c3:
        lt_top_n = st.number_input("Top N", 5, 100, 20, 5, key="lt_top_n")

    # Filter
    df_lt_show = df_lt.copy()
    if lt_label_filter:
        df_lt_show = df_lt_show[df_lt_show["long_term_label"].isin(lt_label_filter)]
    if fund_status_filter and "fundamental_status" in df_lt_show.columns:
        df_lt_show = df_lt_show[df_lt_show["fundamental_status"].isin(fund_status_filter)]
    df_lt_show = df_lt_show.sort_values("long_term_score", ascending=False).head(int(lt_top_n))

    # Summary
    core_n    = (df_lt["long_term_label"] == "LONG_TERM_CORE").sum()
    watchl_n  = (df_lt["long_term_label"] == "LONG_TERM_WATCHLIST").sum()
    underval_n = (df_lt.get("valuation_status", pd.Series()) == "UNDERVALUED").sum()
    ms1, ms2, ms3 = st.columns(3)
    with ms1: st.metric("💎 LONG_TERM_CORE",      core_n)
    with ms2: st.metric("📋 LONG_TERM_WATCHLIST",  watchl_n)
    with ms3: st.metric("🏷️ Undervalued",          underval_n)

    if df_lt_show.empty:
        st.info("Tidak ada kandidat long-term dengan filter ini.")
        return

    # Fundamental table
    from dashboard.data_loader import enrich_df_with_top_brokers
    df_lt_show = enrich_df_with_top_brokers(df_lt_show, scan_date)
    tbl_cols_lt = [
        "ticker", "top_buyer", "top_seller",
        "long_term_score", "long_term_label", "valuation_status",
        "close", "pe_ratio", "pbv", "roe_pct", "der",
        "revenue_growth_pct", "profit_growth_pct", "div_yield_pct",
        "intrinsic_value", "margin_of_safety",
        "cyclicality", "long_term_reason",
    ]
    disp_cols_lt = [c for c in tbl_cols_lt if c in df_lt_show.columns]
    tbl_lt = df_lt_show[disp_cols_lt].copy()

    # Format
    for col in ["pe_ratio", "pbv", "roe_pct", "der", "div_yield_pct",
                "revenue_growth_pct", "profit_growth_pct", "long_term_score"]:
        if col in tbl_lt.columns:
            tbl_lt[col] = tbl_lt[col].apply(
                lambda x: f"{float(x):.1f}" if pd.notna(x) else "-")
    for col in ["close", "intrinsic_value"]:
        if col in tbl_lt.columns:
            tbl_lt[col] = tbl_lt[col].apply(
                lambda x: f"{int(x):,}" if pd.notna(x) and x else "-")
    if "margin_of_safety" in tbl_lt.columns:
        tbl_lt["margin_of_safety"] = tbl_lt["margin_of_safety"].apply(
            lambda x: f"{float(x):+.0f}%" if pd.notna(x) else "-")

    show_df(
        tbl_lt,
        use_container_width=True, hide_index=True,
        height=min(38 * len(tbl_lt) + 40, 520),
        column_config={
            "ticker":              st.column_config.TextColumn("Ticker",        width="small"),
            "top_buyer":           st.column_config.TextColumn("Top Buyer",     width="small"),
            "top_seller":          st.column_config.TextColumn("Top Seller",    width="small"),
            "long_term_score":     st.column_config.TextColumn("LT Score",      width="small"),
            "long_term_label":     st.column_config.TextColumn("Label",         width="medium"),
            "valuation_status":    st.column_config.TextColumn("Valuasi",       width="small"),
            "close":               st.column_config.TextColumn("Close (Rp)",    width="small"),
            "pe_ratio":            st.column_config.TextColumn("PE",            width="small"),
            "pbv":                 st.column_config.TextColumn("PBV",           width="small"),
            "roe_pct":             st.column_config.TextColumn("ROE%",          width="small"),
            "der":                 st.column_config.TextColumn("DER",           width="small"),
            "revenue_growth_pct":  st.column_config.TextColumn("Rev Growth%",   width="small"),
            "profit_growth_pct":   st.column_config.TextColumn("Profit Growth%",width="small"),
            "div_yield_pct":       st.column_config.TextColumn("Div Yield%",    width="small"),
            "intrinsic_value":     st.column_config.TextColumn("Intrinsic (Rp)",width="small"),
            "margin_of_safety":    st.column_config.TextColumn("MoS",           width="small"),
            "cyclicality":         st.column_config.TextColumn("Siklikalitas",  width="small"),
            "long_term_reason":    st.column_config.TextColumn("Ringkasan",     width="large"),
        },
    )

    # Expandable ticker detail
    st.divider()
    st.markdown("#### Detail Fundamental Ticker")
    lt_ticker_list = df_lt_show["ticker"].tolist()
    sel_lt = st.selectbox(
        "Pilih ticker:",
        options=lt_ticker_list,
        format_func=lambda t: f"{t.replace('.JK','')} — {get_company_name(t)}",
        key="lt_detail_select",
    )
    if sel_lt:
        lt_row = df_lt_show[df_lt_show["ticker"] == sel_lt].iloc[0]
        _render_longterm_detail(lt_row, scan_date, api_key)


def _render_longterm_detail(row: pd.Series, scan_date: str, api_key: str | None) -> None:
    """Render full fundamental detail for one long-term candidate."""
    ticker  = str(row.get("ticker", "?"))
    sector  = get_sector(ticker) or ""
    cycl    = str(row.get("cyclicality", classify_cyclicality(sector)))
    lt_score = row.get("long_term_score")
    strengths_raw = str(row.get("strengths", ""))
    flags_raw     = str(row.get("red_flags", ""))

    # Header
    st.markdown(
        f"**{ticker.replace('.JK','')}** &nbsp;·&nbsp; {get_company_name(ticker)}"
        f"&nbsp;·&nbsp; *{sector}*"
    )

    # Cyclicality warning
    if cycl == "cyclical":
        st.warning(
            "⚠️ Saham **siklikal** — kinerja sangat bergantung pada siklus industri/komoditas. "
            "Tidak cocok untuk buy-and-forget. Evaluasi siklus sebelum masuk."
        )
    elif cycl == "defensive":
        st.info("🛡️ Saham **defensif** — lebih stabil, cocok untuk long-term buy-and-hold.")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("**📊 Fundamental Snapshot**")
        fund_items = [
            ("Long Term Score",      f"{float(lt_score):.1f}/10" if pd.notna(lt_score) else "—"),
            ("Valuation Status",     str(row.get("valuation_status", "—"))),
            ("Intrinsic Value (Rp)", f"{int(row.get('intrinsic_value', 0)):,}"
                                     if pd.notna(row.get("intrinsic_value")) and row.get("intrinsic_value") else "—"),
            ("Margin of Safety",     f"{float(row.get('margin_of_safety', 0)):+.0f}%"
                                     if pd.notna(row.get("margin_of_safety")) else "—"),
            ("PE Ratio",             f"{float(row.get('pe_ratio', 0)):.1f}×"
                                     if pd.notna(row.get("pe_ratio")) else "—"),
            ("PBV",                  f"{float(row.get('pbv', 0)):.2f}×"
                                     if pd.notna(row.get("pbv")) else "—"),
            ("ROE",                  f"{float(row.get('roe_pct', 0)):.1f}%"
                                     if pd.notna(row.get("roe_pct")) else "—"),
            ("DER",                  f"{float(row.get('der', 0)):.2f}×"
                                     if pd.notna(row.get("der")) else "—"),
            ("Revenue Growth",       f"{float(row.get('revenue_growth_pct', 0)):+.1f}%"
                                     if pd.notna(row.get("revenue_growth_pct")) else "—"),
            ("Profit Growth",        f"{float(row.get('profit_growth_pct', 0)):+.1f}%"
                                     if pd.notna(row.get("profit_growth_pct")) else "—"),
            ("Dividend Yield",       f"{float(row.get('div_yield_pct', 0)):.2f}%"
                                     if pd.notna(row.get("div_yield_pct")) and row.get("div_yield_pct") else "—"),
        ]
        for label, val in fund_items:
            st.markdown(f"`{label}`: **{val}**")

        if strengths_raw:
            st.markdown("**✅ Kekuatan:**")
            for s in strengths_raw.split(" | "):
                if s.strip():
                    st.markdown(f"- {s.strip()}")

        if flags_raw:
            st.markdown("**⚠️ Red Flags:**")
            for f in flags_raw.split(" | "):
                if f.strip():
                    st.markdown(f"- {f.strip()}")

    with col_r:
        # Long-term price chart
        df_raw = load_raw(ticker)
        st.plotly_chart(
            price_chart_longterm(df_raw, ticker),
            use_container_width=True,
            key=f"lt_chart_{ticker}_{scan_date}",
        )
        _chart_debug(ticker)

    # Multi-period financial statement comparison (lazy, on demand)
    st.markdown("")
    with st.expander("📑 Laporan Keuangan Multi-Periode (dari yfinance)", expanded=False):
        with st.spinner("Mengambil data laporan keuangan..."):
            fin_data = _fetch_financial_comparison(ticker)

        annual = fin_data.get("annual", {}) or {}
        _core = ("revenue", "net_income", "total_assets", "total_equity", "op_cash_flow")
        has_any = any(annual.get(k) for k in _core)

        if fin_data.get("status") != "ok" or not has_any:
            # Honest, useful empty state — not a wall of dashes.
            st.info(
                "📄 **Laporan keuangan belum tersedia** untuk saham ini pada sumber data saat ini.\n\n"
                "Data harga, volume, dan broker tetap tersedia."
            )
        else:
            periods = [r["year"] for r in annual.get("revenue", [])] or \
                      [r["year"] for r in annual.get("net_income", [])]
            src = "tersimpan dari scan" if fin_data.get("_cache_age_days") is not None else "yfinance (live)"
            if periods:
                st.caption(f"Periode tersedia: {' · '.join(periods)}  ·  Sumber: {src}")

            # YoY — explain WHY a value is missing instead of a bare dash.
            yoy = fin_data.get("yoy", {})
            _field = {"revenue_chg": "revenue", "net_income_chg": "net_income",
                      "asset_chg": "total_assets", "equity_chg": "total_equity",
                      "ocf_chg": "op_cash_flow"}
            st.markdown("**YoY (terbaru vs tahun sebelumnya):**")
            yoy_items = [("Revenue", "revenue_chg"), ("Net Income", "net_income_chg"),
                         ("Total Asset", "asset_chg"), ("Equity", "equity_chg"),
                         ("Op. Cash Flow", "ocf_chg")]
            yoy_cols = st.columns(len(yoy_items))
            for col, (label, key) in zip(yoy_cols, yoy_items):
                with col:
                    val = yoy.get(key)
                    series = annual.get(_field[key], [])
                    if val is not None:
                        _pos, _neg, _ = pos_neg_colors()
                        color = _pos if val >= 0 else _neg
                        sub = f'<div style="font-size:16px;font-weight:700;color:{color}">{val:+.1f}%</div>'
                    elif not series:
                        sub = '<div style="font-size:11px;color:var(--c-faint)">tidak ada data</div>'
                    else:
                        sub = '<div style="font-size:11px;color:var(--c-faint)">hanya 1 periode</div>'
                    st.markdown(
                        f'<div style="text-align:center"><div style="font-size:11px;color:var(--c-muted)">'
                        f'{label}</div>{sub}</div>', unsafe_allow_html=True,
                    )

            st.markdown("")

            # Charts only for metrics that actually have data.
            chart_metrics = [m for m in [
                ("revenue", "Revenue"), ("net_income", "Net Income"),
                ("total_equity", "Total Equity"), ("op_cash_flow", "Operating Cash Flow"),
            ] if annual.get(m[0])]
            ch_cols = st.columns(2)
            for i, (metric, label) in enumerate(chart_metrics):
                with ch_cols[i % 2]:
                    fig = fundamental_trend_chart(fin_data, ticker, metric=metric, label=label)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"lt_fin_{ticker}_{metric}_{scan_date}")

    # ── Broker Activity (single-day view — sama seperti Swing tab) ────────
    st.markdown("")
    st.markdown("**📋 Broker Activity**")
    render_broker_section(ticker, scan_date)

    # ── Broker Intelligence (multi-day accumulation analysis) ──────────────
    st.markdown("")
    st.markdown("**🏦 Broker Intelligence**")
    with st.spinner("Menganalisis broker flow..."):
        bi = _fetch_broker_intelligence(ticker, scan_date)

    acc_label = bi.get("broker_accumulation_label", "NO_SIGNAL")
    acc_score = bi.get("broker_accumulation_score", 0.0)
    has_broker_data = acc_score > 0 or bi.get("active_broker_count_5d", 0) > 0

    if not has_broker_data:
        st.caption("Belum ada sinyal broker yang signifikan untuk periode ini.")
    else:
        # Accumulation badge
        _acc_colors = {
            "ACCUMULATION_STRONG": ("#14532d", "#4ade80", "🟢"),
            "ACCUMULATION_WATCH":  ("#7c3f00", "#fb923c", "🟠"),
            "NO_SIGNAL":           ("#1e293b", "#64748b", "⚪"),
        }
        bg, fg, emoji = _acc_colors.get(acc_label, ("#1e293b", "#64748b", "⚪"))
        st.markdown(
            f'<div style="background:{bg};border-radius:8px;padding:8px 14px;display:inline-block;margin-bottom:8px">'
            f'<span style="color:{fg};font-weight:700;font-size:14px">'
            f'{emoji} {acc_label} &nbsp;·&nbsp; Score: {acc_score:.1f}/10</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Broker Intelligence alert (3-layer)
        from stock_scanner.pipeline.broker_intelligence import compute_longterm_broker_alert
        alert_input = {**dict(row), **bi}
        broker_alert = compute_longterm_broker_alert(alert_input)
        alert_status = broker_alert.get("alert_status", "NO_ALERT")
        alert_reason = broker_alert.get("alert_reason", "")
        conf_score   = broker_alert.get("confidence_score", 0.0)

        if alert_status == "LONGTERM_BROKER_ALERT_STRONG":
            st.success(
                f"🚨 **LONGTERM BROKER ALERT — STRONG** (confidence: {conf_score:.1f}/10)\n\n"
                f"{alert_reason}"
            )
        elif alert_status == "LONGTERM_BROKER_ALERT_WATCH":
            st.warning(
                f"⚠️ **LONGTERM BROKER ALERT — WATCH** (confidence: {conf_score:.1f}/10)\n\n"
                f"{alert_reason}"
            )

        # Key metrics
        _pos, _neg, _ = pos_neg_colors()  # theme-aware (readable on light + dark)
        bi_c1, bi_c2, bi_c3 = st.columns(3)
        with bi_c1:
            fn_1d = bi.get("foreign_net_buy_1d", 0)
            fn_5d = bi.get("foreign_net_buy_5d", 0)
            color_1d = _pos if fn_1d >= 0 else _neg
            color_5d = _pos if fn_5d >= 0 else _neg
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">🌍 Foreign Net (1d)</div>'
                f'<div class="value" style="color:{color_1d};font-size:20px">'
                f'{"+" if fn_1d >= 0 else ""}{fn_1d:,.0f} lot</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with bi_c2:
            color_fn5 = _pos if fn_5d >= 0 else _neg
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">🌍 Foreign Net (5d)</div>'
                f'<div class="value" style="color:{color_fn5};font-size:20px">'
                f'{"+" if fn_5d >= 0 else ""}{fn_5d:,.0f} lot</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with bi_c3:
            big_5d = bi.get("big_broker_net_buy_5d", 0)
            color_big = _pos if big_5d >= 0 else _neg
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">🏦 Big Local Net (5d)</div>'
                f'<div class="value" style="color:{color_big};font-size:20px">'
                f'{"+" if big_5d >= 0 else ""}{big_5d:,.0f} lot</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Top buyers / sellers table
        buyers  = bi.get("top_buyer_brokers", [])
        sellers = bi.get("top_seller_brokers", [])

        if buyers or sellers:
            tbl_col1, tbl_col2 = st.columns(2)
            with tbl_col1:
                st.markdown("**Top Buyer Brokers (5d)**")
                if buyers:
                    buyer_df = pd.DataFrame(buyers)
                    buyer_df["net_lot"] = buyer_df["net_lot"].apply(lambda x: f"+{x:,.0f}")
                    show_df(
                        buyer_df[["broker_code", "type_label", "net_lot"]].rename(
                            columns={"broker_code": "Broker", "type_label": "Tipe", "net_lot": "Net Lot"}
                        ),
                        use_container_width=True, hide_index=True, height=200,
                    )
                else:
                    st.caption("Tidak ada data.")

            with tbl_col2:
                st.markdown("**Top Seller Brokers (5d)**")
                if sellers:
                    seller_df = pd.DataFrame(sellers)
                    seller_df["net_lot"] = seller_df["net_lot"].apply(lambda x: f"{x:,.0f}")
                    show_df(
                        seller_df[["broker_code", "type_label", "net_lot"]].rename(
                            columns={"broker_code": "Broker", "type_label": "Tipe", "net_lot": "Net Lot"}
                        ),
                        use_container_width=True, hide_index=True, height=200,
                    )
                else:
                    st.caption("Tidak ada data.")

        # Net flow chart
        broker_history_df = load_broker_history(ticker, n_days=20)
        if not broker_history_df.empty and "date" in broker_history_df.columns:
            try:
                st.plotly_chart(
                    broker_net_flow_chart(broker_history_df, ticker),
                    use_container_width=True,
                    key=f"lt_broker_flow_{ticker}_{scan_date}",
                )
            except Exception as e:
                st.caption(f"Chart broker flow tidak dapat dimuat: {e}")

        # Strengths / red flags
        bi_strengths  = bi.get("strengths", [])
        bi_red_flags  = bi.get("red_flags", [])
        if bi_strengths:
            st.markdown("**✅ Broker Strengths:**")
            for s in bi_strengths:
                st.markdown(f"- {s}")
        if bi_red_flags:
            st.markdown("**⚠️ Broker Red Flags:**")
            for f in bi_red_flags:
                st.markdown(f"- {f}")

    # ── News sentiment (reputation/risk layer) ─────────────────────────────
    st.markdown("")
    news_status = str(row.get("news_data_status", "")).lower()
    if news_status in ("ok", "none"):
        st.markdown("**📰 Sentimen Berita (Risiko / Reputasi)**")
        if news_status == "ok":
            articles = load_news_articles_for_ticker(ticker, scan_date)
            if articles:
                from stock_scanner.pipeline.news_summarizer import summarize_news_articles, format_news_bullets
                summary = summarize_news_articles(articles)
                bullets = format_news_bullets(summary, len(articles),
                                             sentiment_score=row.get("news_sentiment_score"),
                                             max_chars=600)
                st.markdown(bullets)
        else:
            st.caption("Tidak ada berita relevan dalam 3 hari terakhir.")

    # ── AI Explanation (if API key available) ────────────────────────────
    st.markdown("**🤖 AI Explanation**")
    with st.spinner("Membuat narasi long-term..."):
        articles_lt = load_news_articles_for_ticker(ticker, scan_date)
        explanation = explain_signal_llm(row, api_key=api_key, articles=articles_lt)
    st.markdown(explanation)


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------

def _bool_val(val) -> bool:
    """Safe bool conversion for dashboard use."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Remote mode bootstrap (fetch payload once, share across sidebar + tabs)
# ---------------------------------------------------------------------------
_remote_mode = is_remote_mode()
_remote_payload: dict = {}

if _remote_mode:
    @st.cache_data(ttl=300, show_spinner="Memuat data dari server...")
    def _fetch_remote_payload() -> dict:
        return load_published_payload()

    _remote_payload = _fetch_remote_payload()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def _fmt_date_id(date_str: str) -> str:
    """YYYY-MM-DD → 'Senin, 08 Jun 2026' (Indonesian). Falls back to raw string."""
    from datetime import datetime as _dt
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    mons = ["", "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
            "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
    try:
        d = _dt.strptime(str(date_str)[:10], "%Y-%m-%d")
        return f"{days[d.weekday()]}, {d.day} {mons[d.month]} {d.year}"
    except (ValueError, IndexError):
        return str(date_str)


def _now_wib_label() -> str:
    """Current report/view time in WIB → 'Senin, 08 Jun 2026 06:00 WIB'."""
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    n = _dt.now(_tz(_td(hours=7)))
    return f"{_fmt_date_id(n.strftime('%Y-%m-%d'))} {n.strftime('%H:%M')} WIB"


@st.cache_data(ttl=900, show_spinner=False)
def _git_commit_info() -> tuple[str, str]:
    """(branch, short_sha) of the deployed checkout — robust on local & Cloud."""
    root = str(Path(__file__).parent.parent)
    try:
        import subprocess
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=4)
        br = subprocess.run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, timeout=4)
        if sha.returncode == 0 and sha.stdout.strip():
            return (br.stdout.strip() or "?"), sha.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    # Fallback: parse .git/HEAD directly (no git binary needed).
    try:
        gd = Path(__file__).parent.parent / ".git"
        head = (gd / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            branch = ref.rsplit("/", 1)[-1]
            rp = gd / ref
            if rp.exists():
                return branch, rp.read_text().strip()[:7]
            pr = gd / "packed-refs"
            if pr.exists():
                for ln in pr.read_text().splitlines():
                    if ln.strip().endswith(ref) and not ln.startswith("#"):
                        return branch, ln.split()[0][:7]
        else:
            return "detached", head[:7]
    except Exception:  # noqa: BLE001
        pass
    import os
    return os.environ.get("GIT_BRANCH", "?"), (os.environ.get("GIT_COMMIT", "unknown")[:7])


def _published_status_payload(remote_mode: bool, remote_payload: dict) -> dict:
    """The latest COMMITTED published payload (what the widget reports on)."""
    if remote_mode and remote_payload:
        return remote_payload
    import json
    p = Path(__file__).parent.parent / "data" / "published" / "latest_scan.json"
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def render_deployment_status(remote_mode: bool, remote_payload: dict) -> None:
    """Header widget: data source, branch@commit, published Market date + Last
    updated — reflects the latest COMMITTED published data so the morning
    auto-refresh is verifiable at a glance."""
    pub = _published_status_payload(remote_mode, remote_payload)
    scan_date = str(pub.get("scan_date") or "—")
    gen = str(pub.get("generated_at") or "—")
    branch, sha = _git_commit_info()
    source = "☁️ Remote (GitHub)" if remote_mode else "💻 Local"
    market = _fmt_date_id(scan_date) if scan_date != "—" else "—"
    updated = gen[:16].replace("T", " ") + " WIB" if gen and gen != "—" else "—"
    # Stale guard: compare the published session to the latest VALID IDX trading
    # day from the calendar (not a fixed day-count window). The published data is
    # "fresh" while it is at most one session behind the expected last trading day
    # (the morning scan normally publishes the prior session); anything older than
    # that is flagged, and we surface the trading day the data *should* show so the
    # gap is actionable rather than a silent stale value.
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now_wib = _dt.now(_tz(_td(hours=7)))
    expected_str = None
    stale = False
    try:
        from stock_scanner.utils.trading_calendar import (
            expected_market_date, previous_trading_day,
        )
        expected = expected_market_date(now_wib)
        expected_str = expected.strftime("%Y-%m-%d")
        pub_date = _dt.strptime(scan_date, "%Y-%m-%d").date()
        stale = pub_date < previous_trading_day(expected)
    except Exception:  # noqa: BLE001
        stale = False
    dot = "🟠" if stale else "🟢"
    market_line = f'<span class="k">Market date:</span> <b>{market}</b>'
    if stale and expected_str:
        market_line += (
            f'<br><span class="k">⚠️ Data tertinggal:</span> '
            f'sesi bursa terakhir <b>{_fmt_date_id(expected_str)}</b>'
        )
    st.markdown(
        f'<div class="status-panel">'
        f'<b>📡 Status Deploy</b> {dot}<br>'
        f'<span class="k">Sumber data:</span> {source}<br>'
        f'<span class="k">Branch/commit:</span> <code>{branch}@{sha}</code><br>'
        f'{market_line}<br>'
        f'<span class="k">Last updated:</span> {updated}'
        f'</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        '<div class="app-brand">'
        '<div class="mark">📈</div>'
        '<div><div class="name">IDX Scanner</div>'
        '<div class="sub">Signal & market intelligence</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Explicit in-app Light/Dark switch.
    render_theme_toggle()
    render_deployment_status(_remote_mode, _remote_payload)

    # Mode badge
    if _remote_mode:
        st.markdown(
            '<div class="mode-pill mode-online">☁️ Mode: <b>Online</b> — data dari GitHub</div>',
            unsafe_allow_html=True,
        )
        if _remote_payload:
            scan_date_str = _remote_payload.get("scan_date", "")
            exec_date_str = _remote_payload.get("execution_date", scan_date_str)
            is_live       = _remote_payload.get("is_live_scan", True)
            # (Last updated / Market date now shown in the 📡 Status Deploy widget.)
            if not is_live:
                st.warning(
                    f"⚠️ **Data bukan sesi live**\n\n"
                    f"Script jalan {exec_date_str} (hari libur bursa). "
                    f"Data market yang dipakai: **{scan_date_str}** (last trading day).",
                    icon="⚠️",
                )
        else:
            st.warning(
                "⚠️ Data tidak tersedia.\n\n"
                "Pastikan `data/published/latest_scan.json` sudah di-commit ke repo, "
                "atau set secret `REMOTE_DATA_URL` dengan URL yang benar."
            )
    else:
        st.markdown(
            '<div class="mode-pill mode-local">💻 Mode: <b>Lokal</b></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    all_dates = available_dates_unified(_remote_payload if _remote_mode else None)
    if not all_dates:
        if _remote_mode:
            st.error(
                "Tidak ada data dari server.\n\n"
                "Pastikan REMOTE_DATA_URL sudah benar dan scan sudah pernah dijalankan."
            )
        else:
            st.warning(
                "Belum ada data scan.\n\nJalankan dulu:\n"
                "```\npython -m stock_scanner.pipeline.run_daily_scan\n```"
            )
        st.stop()

    selected_date = st.selectbox("Sesi market (tanggal data)", options=all_dates, index=0)
    # Disambiguate: report/view time (now) vs the market session being analysed.
    st.caption(f"🕒 Dibuka: {_now_wib_label()}")
    st.caption(f"📅 Data market: {_fmt_date_id(selected_date)}")
    st.divider()

    # Load ALL tickers for date (local or remote)
    if _remote_mode:
        df_all = load_all_tickers_unified(selected_date, _remote_payload)
    else:
        df_all = load_all_tickers_for_date(selected_date)
    if df_all.empty:
        if _remote_mode:
            st.error(
                f"Data server untuk {selected_date} kosong atau belum tersedia.\n\n"
                "Scan mungkin belum selesai hari ini."
            )
        else:
            st.error(f"Tidak ada data untuk {selected_date}.")
        st.stop()

    # Exclude suspended / recently-unsuspended names from every candidate list,
    # table, and screener tab (cheap, from the per-ticker scan date).
    try:
        from stock_scanner.pipeline.suspension import filter_active
        _n_before = len(df_all)
        df_all = filter_active(df_all)
        _n_suspended = _n_before - len(df_all)
    except Exception:  # noqa: BLE001
        _n_suspended = 0
    if _n_suspended:
        st.caption(f"🚫 {_n_suspended} saham suspend / baru dibuka disembunyikan dari kandidat")

    # Global retail-accumulation filter — applied here, once, exactly like
    # filter_active() above, so every tab (which all receive this same
    # df_all) inherits it automatically. Default OFF: apply_retail_filter()
    # returns df_all completely unchanged (same object, no broker parquet
    # read) when the checkbox is off, so dashboard output stays
    # byte-identical to before this feature existed. See
    # dashboard/data_loader.py::apply_retail_filter — reuses the audited
    # broker classification (stock_scanner/configs/broker_config.yaml via
    # broker_analytics.py) and the existing top_buyer/top_seller broker
    # parquet loop; hides only stocks positively determined to be
    # retail-dominated, never ones with unknown/missing broker data.
    hide_retail = st.checkbox(
        "Hide Retail Accumulation",
        value=False,
        key="hide_retail_accumulation",
        help="Sembunyikan saham yang akumulasinya didominasi broker ritel "
             "(retail_ratio > threshold di broker_config.yaml). Saham tanpa "
             "data broker tetap ditampilkan.",
    )
    try:
        _n_before_retail = len(df_all)
        df_all = apply_retail_filter(df_all, selected_date, hide_retail)
        _n_retail_hidden = _n_before_retail - len(df_all)
    except Exception:  # noqa: BLE001
        _n_retail_hidden = 0
    if hide_retail and _n_retail_hidden:
        st.caption(f"🏪 {_n_retail_hidden} saham didominasi broker ritel disembunyikan")

    # Mini signal summary
    if "signal" in df_all.columns:
        sig_counts = df_all["signal"].value_counts()
        _pal = palette()
        sc1, sc2, sc3 = st.columns(3)
        for col, sig, color in [
            (sc1, "BREAKOUT",   _pal["success"]),
            (sc2, "PRE_MARKUP", _pal["info"]),
            (sc3, "WATCH",      _pal["warning"]),
        ]:
            cnt = int(sig_counts.get(sig, 0))
            with col:
                st.markdown(
                    f'<div class="mini-stat">'
                    f'<div class="k">{sig[:3]}</div>'
                    f'<div class="v" style="color:{color}">{cnt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Fitur tidak tersedia di remote mode
    if _remote_mode:
        st.divider()
        st.caption(
            "⚠️ Mode online: chart OHLCV, broker flow, news articles, "
            "dan history tidak tersedia. Hanya data sinyal & level trading."
        )
    else:
        # Local mode: detect scan_date vs market_data_date mismatch
        # If most tickers' last data row is older than selected_date, it's a stale scan.
        if not df_all.empty and "date" in df_all.columns:
            _max_market_date = str(pd.to_datetime(df_all["date"]).max().date())
            if _max_market_date < selected_date:
                st.divider()
                st.warning(
                    f"⚠️ **Data bukan sesi live** — "
                    f"scan_date: {selected_date}, "
                    f"market data terakhir: **{_max_market_date}** "
                    f"(kemungkinan hari libur bursa)."
                )

    st.divider()
    st.markdown("**Filter Signal**")
    signal_filter = st.multiselect(
        "Tampilkan",
        options=["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"],
        default=["BREAKOUT", "PRE_MARKUP", "WATCH"],
        label_visibility="collapsed",
    )

    # Filtered ticker list for sidebar
    if "signal" in df_all.columns and signal_filter:
        df_sidebar = df_all[df_all["signal"].isin(signal_filter)].copy()
    else:
        df_sidebar = df_all.copy()
    if "total_score" in df_sidebar.columns:
        df_sidebar = df_sidebar.sort_values("total_score", ascending=False)

    sidebar_tickers = df_sidebar["ticker"].tolist() if not df_sidebar.empty else []

    def _fmt_sidebar_ticker(t: str) -> str:
        rows = df_sidebar[df_sidebar["ticker"] == t] if not df_sidebar.empty else pd.DataFrame()
        if rows.empty:
            rows = df_all[df_all["ticker"] == t]
        sig = rows["signal"].values[0] if not rows.empty else "NONE"
        score = rows["total_score"].values[0] if (not rows.empty and "total_score" in rows.columns) else 0.0
        try:
            score_str = f"{float(score):.1f}"
        except (TypeError, ValueError):
            score_str = "—"
        name = get_company_name(t)
        short_name = name[:22] + "…" if len(name) > 24 else name
        return f"{_SIG_EMOJI.get(sig, '⚪')} {t}  {score_str}  |  {short_name}"

    st.divider()
    st.markdown(f"**Daftar Saham** ({len(sidebar_tickers)} ticker)")

    selected_ticker: str | None = None
    if not sidebar_tickers:
        st.caption("Tidak ada ticker dengan signal yang dipilih.")
    else:
        selected_ticker = st.selectbox(
            "Pilih ticker:",
            options=sidebar_tickers,
            format_func=_fmt_sidebar_ticker,
            label_visibility="collapsed",
        )

    st.divider()
    api_key_input = st.text_input(
        "Anthropic API Key (opsional)",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Jika diisi, penjelasan AI menggunakan Claude. Kosong = rule-based.",
    )
    active_api_key = api_key_input.strip() or None
    st.caption(f"Mode explain: {'🤖 Claude API' if active_api_key else '📋 Rule-based'}")



# ---------------------------------------------------------------------------
# Smart Money Screener tab
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def _run_smart_money_screen(scan_date: str, df_all: pd.DataFrame) -> pd.DataFrame:
    """Run the smart-money screener over all tickers (cached 10 min per date)."""
    cfg = load_smart_money_config()
    return screen_smart_money(df_all, scan_date, cfg=cfg)


def render_smart_money_tab(df_all: pd.DataFrame, scan_date: str) -> None:
    """🎯 Smart Money — accumulation-footprint screener across the whole universe."""
    st.markdown("### 🎯 Smart Money Screener")
    st.caption(
        "Mendeteksi saham yang mulai diakumulasi sebelum harga naik: volume naik "
        "duluan, kepemilikan/broker akumulasi, harga belum naik, fundamental sehat. "
        "Kepemilikan & serapan broker memakai data REAL Index Alpha — detail penuhnya "
        "ada di halaman detail tiap saham (Search/Swing)."
    )

    if df_all is None or df_all.empty:
        st.info("Belum ada data scan untuk tanggal ini.")
        return

    with st.spinner("Menjalankan screener…"):
        res = _run_smart_money_screen(scan_date, df_all)

    if res is None or res.empty:
        st.warning("Screener tidak menghasilkan kandidat.")
        return

    # ── Summary counts ───────────────────────────────────────────────────
    n_strong = int((res["smart_money_label"] == "🔥 Strong Candidate").sum())
    n_watch = int((res["smart_money_label"] == "👀 Watch").sum())
    n_hidden = int(res["hidden_accum"].isin(
        ["Hidden Accumulation", "Strong Accumulation"]).sum())
    n_absorb = int(res["broker_absorption"].isin(
        ["Strong Accumulation", "Moderate Accumulation"]).sum())
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🔥 Strong Candidate", n_strong)
    s2.metric("👀 Watch", n_watch)
    s3.metric("Hidden Accumulation", n_hidden)
    s4.metric("Broker Absorption", n_absorb)

    # ── Filters ──────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        labels = st.multiselect(
            "Kandidat", ["🔥 Strong Candidate", "👀 Watch", "—"],
            default=["🔥 Strong Candidate", "👀 Watch"], key="sm_label",
        )
    with f2:
        grades = st.multiselect(
            "Grade fundamental", ["A", "B", "C", "N/A"],
            default=["A", "B"], key="sm_grade",
        )
    with f3:
        min_pillars = st.number_input("Min pillar", 1, 5, 2, 1, key="sm_pillars")

    show = res.copy()
    if labels:
        show = show[show["smart_money_label"].isin(labels)]
    if grades:
        show = show[show["fundamental_grade"].isin(grades)]
    show = show[show["smart_money_pillars"] >= int(min_pillars)]

    st.caption(f"Menampilkan {len(show)} dari {len(res)} saham.")

    if show.empty:
        st.info("Tidak ada saham yang cocok dengan filter ini.")
        return

    # ── Table ────────────────────────────────────────────────────────────
    from dashboard.data_loader import enrich_df_with_top_brokers
    show = enrich_df_with_top_brokers(show, scan_date)
    disp = show.copy()
    disp["close"] = disp["close"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    disp["vol_ratio_20d"] = disp["vol_ratio_20d"].apply(
        lambda x: f"{float(x):.1f}×" if pd.notna(x) else "—")
    disp["roc20"] = disp["roc20"].apply(lambda x: f"{float(x):+.1f}%" if pd.notna(x) else "—")
    disp["absorb_share"] = disp["absorb_share"].apply(
        lambda x: f"{float(x) * 100:.0f}%" if pd.notna(x) else "—")
    disp["fundamental_score"] = disp["fundamental_score"].apply(
        lambda x: f"{int(x)}" if pd.notna(x) else "—")

    cols = ["ticker", "top_buyer", "top_seller",
            "smart_money_label", "smart_money_pillars", "close",
            "volume_accum", "vol_ratio_20d", "roc20", "ownership",
            "broker_absorption", "absorb_broker", "absorb_share",
            "hidden_accum", "fundamental_grade", "fundamental_score"]
    show_df(
        disp[cols], use_container_width=True, hide_index=True,
        height=min(40 * len(disp) + 44, 560),
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "top_buyer": st.column_config.TextColumn("Top Buyer", width="small"),
            "top_seller": st.column_config.TextColumn("Top Seller", width="small"),
            "smart_money_label": st.column_config.TextColumn("Kandidat", width="medium"),
            "smart_money_pillars": st.column_config.NumberColumn("Pillar", width="small"),
            "close": st.column_config.TextColumn("Close", width="small"),
            "volume_accum": st.column_config.TextColumn("Vol Accum", width="small"),
            "vol_ratio_20d": st.column_config.TextColumn("Vol×", width="small"),
            "roc20": st.column_config.TextColumn("ROC20", width="small"),
            "ownership": st.column_config.TextColumn("Ownership", width="small"),
            "broker_absorption": st.column_config.TextColumn("Broker Absorb", width="medium"),
            "absorb_broker": st.column_config.TextColumn("Top Brk", width="small"),
            "absorb_share": st.column_config.TextColumn("Share", width="small"),
            "hidden_accum": st.column_config.TextColumn("Hidden Accum", width="medium"),
            "fundamental_grade": st.column_config.TextColumn("Grade", width="small"),
            "fundamental_score": st.column_config.TextColumn("F.Score", width="small"),
        },
    )

    # ── Per-ticker detail: reason + price chart + Broker Summary ─────────
    st.divider()
    sel = st.selectbox(
        "Lihat detail kandidat:", options=show["ticker"].tolist(),
        format_func=lambda t: f"{t.replace('.JK', '')} — {get_company_name(t)}",
        key="sm_detail",
    )
    if sel:
        r = show[show["ticker"] == sel].iloc[0]
        d_left, d_right = st.columns([1, 1.1])
        with d_left:
            st.markdown(f"**{sel.replace('.JK','')}** — {r['smart_money_label']} "
                        f"({int(r['smart_money_pillars'])} pillar)")
            st.markdown(f"- Alasan: {r['reasons']}")
            st.markdown(f"- Ownership: **{r['ownership']}** · Broker absorption: "
                        f"**{r['broker_absorption']}**"
                        + (f" oleh {r['absorb_broker']}" if pd.notna(r.get('absorb_broker')) else ""))
            st.markdown(f"- Hidden accumulation: **{r['hidden_accum']}** · "
                        f"Fundamental: **{r['fundamental_grade']}** "
                        f"({r['fundamental_score'] if pd.notna(r['fundamental_score']) else '—'}/100)")
        with d_right:
            st.markdown("**📈 Chart Harga**")
            _sm_raw = load_raw(sel)
            st.plotly_chart(
                price_chart(_sm_raw, sel, signal_date=scan_date),
                use_container_width=True, key=f"sm_chart_{sel}_{scan_date}",
            )

        with st.expander("🏦 Broker Summary — kepemilikan & serapan broker"):
            render_broker_section(sel, scan_date, key_prefix="smart", show_header=False)

    st.caption(
        "Skor volume, harga, tren, dan fundamental dihitung untuk **seluruh universe**."
    )


# ---------------------------------------------------------------------------
# Consecutive up/down streak screener tab (bullish/sideways only; no bearish)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _cached_streak_screen() -> pd.DataFrame:
    """Cached scan of the OHLC bundle for up/down streaks (bearish excluded).
    min_len=1 so the tab can offer any selectable streak length 1–7."""
    from dashboard.streaks import screen_streaks
    return screen_streaks(min_len=1)


def _streak_price_df(df_raw: pd.DataFrame, n: int = 12) -> pd.DataFrame | None:
    """Recent daily price table (newest first): Tanggal, Close, Δ% (close-to-close).
    Lets the consecutive up/down days be read numerically next to the chart."""
    if df_raw is None or df_raw.empty or "close" not in df_raw.columns:
        return None
    d = df_raw.sort_values("date").copy()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["chg"] = d["close"].pct_change() * 100
    d = d.tail(n)
    out = pd.DataFrame({
        "Tanggal": pd.to_datetime(d["date"]).dt.strftime("%d %b"),
        "Close": d["close"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—"),
        # exact-zero stays neutral ("0.00%" — no +/- so it isn't tinted)
        "Δ%": d["chg"].apply(
            lambda x: "—" if pd.isna(x) else (f"{x:+.2f}%" if round(x, 2) != 0 else "0.00%")),
    })
    return out.iloc[::-1].reset_index(drop=True)   # newest at top


def _render_streak_card(r: pd.Series, scan_date: str) -> None:
    """One result: header + trend chips + reason + chart + price table + broker."""
    tk = str(r["ticker"])
    up = r["streak_dir"] == "up"
    arrow = "🟢 ▲" if up else "🔴 ▼"
    dir_id = "naik" if up else "turun"
    trend_chip = {
        "bullish":  ("chip-ok",    "🟢 Bullish"),
        "sideways": ("chip-muted", "⚪ Sideways"),
    }.get(str(r["trend_state"]), ("chip-muted", str(r["trend_state"])))

    st.markdown(f"#### {arrow} {tk.replace('.JK', '')} — {dir_id} "
                f"{int(r['streak_len'])} hari berturut-turut")
    st.markdown(
        f'<span class="chip {trend_chip[0]}">Struktur: {trend_chip[1]}</span>&nbsp;'
        f'<span class="chip chip-info">{get_company_name(tk)}</span>&nbsp;'
        f'<span class="chip chip-muted">Close {r["last_close"]:,.0f} · {r["last_date"]}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"📋 {r['reason']}")

    df_raw = load_raw(tk)
    c_chart, c_tbl = st.columns([1.7, 1])
    with c_chart:
        st.plotly_chart(
            price_chart(df_raw, tk, signal_date=scan_date),
            use_container_width=True, key=f"streak_chart_{tk}",
        )
    with c_tbl:
        st.markdown("**📋 Tabel Harga Harian**")
        ptbl = _streak_price_df(df_raw)
        if ptbl is None or ptbl.empty:
            st.caption("Data harga tidak tersedia.")
        else:
            st.dataframe(
                style_change_table(ptbl, pct_cols=("Δ%",)),
                use_container_width=True, hide_index=True, height=360,
                column_config={
                    "Tanggal": st.column_config.TextColumn("Tanggal", width="small"),
                    "Close":   st.column_config.TextColumn("Close", width="small"),
                    "Δ%":      st.column_config.TextColumn("Δ%", width="small"),
                },
            )
            st.caption("Δ% = perubahan close vs sesi sebelumnya (hijau naik / merah turun).")

    with st.expander("🏦 Broker Summary — kepemilikan & serapan broker"):
        render_broker_section(tk, scan_date, key_prefix="streak", show_header=False)
    st.divider()


def render_streak_tab(scan_date: str) -> None:
    """🔁 Stocks up/down several consecutive days, only if bullish/sideways."""
    st.markdown("### 🔁 Naik / Turun Beberapa Hari Berturut-turut")
    st.caption(
        "Saham yang ditutup **naik** atau **turun** beberapa hari berturut-turut "
        "(arah close-to-close), disaring hanya yang struktur besarnya **bullish** "
        "atau **sideways** — yang **bearish (tren turun, termasuk yang lemah) "
        "dikecualikan**. Sumber: bundle OHLC sesi terakhir yang dipublikasikan."
    )

    df = _cached_streak_screen()
    if df is None or df.empty:
        st.info("📭 Belum ada kandidat beruntun (atau bundle OHLC belum tersedia). "
                "Bundle dibuat ulang otomatis tiap scan harian.")
        return

    def _count(dirk: str, ln: int) -> int:
        return int(((df["streak_dir"] == dirk) & (df["streak_len"] >= ln)).sum())

    # ── Summary counts by streak bucket (bullish/sideways only) ──────────
    st.markdown("**Ringkasan — jumlah saham per panjang streak** "
                "<span style='color:var(--c-faint)'>(struktur bullish / sideways saja)</span>",
                unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Streak": [f"≥ {k} hari" for k in range(1, 8)],
        "📈 Naik": [_count("up", k) for k in range(1, 8)],
        "📉 Turun": [_count("down", k) for k in range(1, 8)],
    })
    st.dataframe(style_table(summary), use_container_width=True, hide_index=True, height=290)
    st.markdown("")

    # ── Filters ──────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([1.2, 1.3, 1])
    direction = f1.radio("Arah", ["📈 Naik", "📉 Turun"], horizontal=True, key="streak_dir_f")
    trend     = f2.radio("Struktur", ["Semua", "Bullish", "Sideways"], horizontal=True, key="streak_trend_f")
    topn      = f3.number_input("Tampilkan", min_value=3, max_value=20, value=6, step=1, key="streak_topn_f")
    # Selectable streak length 1–7 (minimum consecutive days, close-to-close).
    minln = st.slider("Minimal hari berturut-turut (1–7)", min_value=1, max_value=7,
                      value=3, key="streak_len_f")

    dirk = "up" if "Naik" in direction else "down"
    sub = df[(df["streak_dir"] == dirk) & (df["streak_len"] >= minln)]
    if trend == "Bullish":
        sub = sub[sub["trend_state"] == "bullish"]
    elif trend == "Sideways":
        sub = sub[sub["trend_state"] == "sideways"]

    if sub.empty:
        st.info(f"Tidak ada saham **{direction}** dengan streak **≥ {minln} hari** "
                f"& struktur **{trend.lower()}** pada sesi ini. Coba kurangi panjang "
                "streak atau pilih struktur *Semua*.")
        return

    st.caption(f"**{len(sub)}** saham cocok ({direction}, ≥ {minln} hari, {trend.lower()}) "
               f"— menampilkan {min(len(sub), int(topn))} teratas (streak terpanjang dulu).")
    for _, r in sub.head(int(topn)).iterrows():
        _render_streak_card(r, scan_date)


# ---------------------------------------------------------------------------
# MAIN PAGE HEADER (left-aligned, restrained — no centred hero)
# ---------------------------------------------------------------------------
_hc_left, _hc_right = st.columns([3, 1], gap="small")
with _hc_left:
    st.markdown(
        '<div class="page-head">'
        '<div class="title">Market Dashboard</div>'
        '<div class="desc">Sinyal teknikal, level trading & analitik emiten IDX</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with _hc_right:
    st.markdown(
        f'<div style="text-align:right;padding-top:8px">'
        f'<span class="chip chip-info">📅 Sesi {_fmt_date_id(selected_date)}</span></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------------
(tab_scalping, tab_swing, tab_longterm, tab_smart, tab_streak,
 tab_perf, tab_search, tab_history, tab_knowledge_base, tab_daily_movers, tab_ai_lab,
 tab_dictionary) = st.tabs(
    ["📈 Scalping", "🔄 Swing Trading", "📊 Long Term", "🎯 Smart Money",
     "🔁 Naik/Turun Beruntun", "📋 Signal Performance", "🔍 Search Emiten", "🕐 History",
     "🧠 Learning Agent", "🚀 Daily Movers >10%", "🧪 AI Lab", "📖 Stock Dictionary"]
)


# ===========================================================================
# TAB — SIGNAL LIST PERFORMANCE
# ===========================================================================
with tab_perf:
    st.markdown("### 📋 Signal List Performance")
    st.caption("Win-rate sinyal Swing & Scalping — referensi = Open (close sesi "
               "sinyal); High & Close diukur dari sesi bursa BERIKUTNYA — W/L High "
               "(High vs Open) dan W/L Close (Close vs Open) dinilai terpisah.")

    from stock_scanner.pipeline.performance import load_results
    _res = load_results()
    if _res is None or _res.empty:
        st.info("📭 Belum ada data performa sinyal. Akan terisi otomatis setelah scan "
                "harian mengevaluasi sesi berikutnya.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            _strat = st.radio("Strategi", ["Swing", "Scalping"], horizontal=True,
                              key="perf_strat").lower()
        _sd = _res[_res["strategy"] == _strat]
        # Review date = EVAL DATE (the market session reviewed), newest first.
        _dates = sorted(
            _sd.loc[_sd["status"] == "evaluated", "eval_date"].dropna().astype(str).unique(),
            reverse=True)
        with c2:
            _date = st.selectbox("Tanggal review (sesi market)", options=_dates,
                                 index=0 if _dates else None, key="perf_date")

        _day = _sd[(_sd["eval_date"].astype(str) == str(_date))
                   & (_sd["status"] == "evaluated")].copy()
        _n = len(_day)
        _w = int((_day["wl_close"] == "W").sum()) if _n else 0
        _wr = round(_w / _n * 100, 1) if _n else 0.0
        # Pending = signals still awaiting their next session (no eval_date yet).
        _pend = int((_sd["status"] == "pending").sum())

        _perf_pal = palette()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sinyal direview", _n)
        with m2:
            # W in green, L in red (colour + count + letter — not colour alone).
            st.markdown(
                f'<div class="metric-card"><div class="label">Win / Loss</div>'
                f'<div class="value" style="font-size:22px">'
                f'<span style="color:{_perf_pal["success"]};font-weight:700">{_w} W</span>'
                f'<span style="color:var(--c-faint)"> / </span>'
                f'<span style="color:{_perf_pal["danger"]};font-weight:700">{_n - _w} L</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        m3.metric("Win Rate", f"{_wr}%" if _n else "—")
        m4.metric("Pending (menunggu sesi)", _pend)
        if _pend:
            st.caption(f"⏳ {_pend} sinyal {_strat} menunggu sesi bursa berikutnya; "
                       "akan otomatis masuk review begitu OHLC-nya tersedia.")

        # -------------------------------------------------------------------
        # IHSG BENCHMARK — synced to the same review/market date as above so
        # signal performance and the overall market move are comparable at a
        # glance. Falls back to the last valid IDX trading day if `_date`
        # itself has no IHSG session (holiday/weekend), never to a stale
        # unrelated date.
        # -------------------------------------------------------------------
        st.markdown("#### 📈 Benchmark IHSG")
        _ihsg = get_ihsg_session(str(_date)) if _date else None
        if _ihsg is None:
            st.caption("Data IHSG tidak tersedia untuk sesi ini.")
        else:
            _status_icon = {"up": "🟢", "down": "🔴", "flat": "⚪"}[_ihsg["status"]]
            _status_label = {"up": "Naik", "down": "Turun", "flat": "Flat"}[_ihsg["status"]]
            _ihsg_color = {"up": _perf_pal["success"], "down": _perf_pal["danger"],
                           "flat": "var(--c-faint)"}[_ihsg["status"]]

            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.markdown(
                    f'<div class="metric-card"><div class="label">IHSG Close ({_ihsg["date"]})</div>'
                    f'<div class="value">{_ihsg["close"]:,.0f}</div></div>',
                    unsafe_allow_html=True,
                )
            with ic2:
                st.markdown(
                    f'<div class="metric-card"><div class="label">% Change Harian</div>'
                    f'<div class="value" style="color:{_ihsg_color}">'
                    f'{_status_icon} {_ihsg["pct_change"]:+.2f}% ({_status_label})</div></div>',
                    unsafe_allow_html=True,
                )
            with ic3:
                _avg_ret = _day["pct_close"].astype(float).mean() if _n else None
                if _avg_ret is not None:
                    _spread = _avg_ret - _ihsg["pct_change"]
                    _spread_label = ("Outperform" if _spread > 0
                                      else "Underperform" if _spread < 0 else "Setara")
                    _spread_color = (_perf_pal["success"] if _spread > 0
                                      else _perf_pal["danger"] if _spread < 0 else "var(--c-faint)")
                    st.markdown(
                        f'<div class="metric-card"><div class="label">Rata² Return Sinyal vs IHSG</div>'
                        f'<div class="value" style="color:{_spread_color};font-size:18px">'
                        f'{_spread_label} {_spread:+.2f}pp</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Belum ada sinyal evaluated untuk dibandingkan.")

            if _ihsg["date"] != str(_date):
                st.caption(f"⚠️ Tidak ada data IHSG untuk {_date} (libur/weekend) — "
                           f"menampilkan sesi bursa valid terakhir: {_ihsg['date']}.")

            _ihsg_hist = load_ihsg_data()
            if not _ihsg_hist.empty:
                _window = _ihsg_hist[_ihsg_hist["date"] <= pd.Timestamp(_ihsg["date"])].tail(30)
                st.plotly_chart(
                    ihsg_benchmark_chart(_window, highlight_date=_ihsg["date"]),
                    use_container_width=True, key="ihsg_benchmark_chart",
                )
        st.divider()

        # Table (matches the screenshot layout)
        _show = _day[["ticker", "signal", "prev", "close", "high", "pct_high", "pct_close",
                      "wl_high", "wl_close", "eval_date", "status"]].copy()
        for c in ("pct_high", "pct_close"):
            _show[c] = _show[c].apply(lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "—")
        for c in ("prev", "close", "high"):
            _show[c] = _show[c].apply(lambda x: f"{float(x):,.0f}" if pd.notna(x) else "—")
        # Theme-correct table with the W/L columns highlighted (green W / red L,
        # tinted cell + bold letter). style_perf_table is used directly instead of
        # show_df so the W/L colours win over the base cell colours.
        st.dataframe(
            style_perf_table(_show, wl_col=["wl_high", "wl_close"]),
            use_container_width=True, hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Signal", width="small"),
                "signal": st.column_config.TextColumn("Type", width="small"),
                "prev": st.column_config.TextColumn("Open"),
                "close": st.column_config.TextColumn("Close"),
                "high": st.column_config.TextColumn("High"),
                "pct_high": st.column_config.TextColumn("% High"),
                "pct_close": st.column_config.TextColumn("% Close"),
                "wl_high": st.column_config.TextColumn("W/L High", width="small"),
                "wl_close": st.column_config.TextColumn("W/L Close", width="small"),
                "eval_date": st.column_config.TextColumn("Eval Date", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
            },
        )

        # Download links (CSV always; Excel if the daily file exists)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ CSV", data=_day.to_csv(index=False).encode(),
                file_name=f"{_strat}_{_date}.csv", mime="text/csv", key="perf_dl_csv")
        with dl2:
            _xlsx = _ROOT_PERF / "daily" / f"signal_list_{_date}.xlsx"
            if _xlsx.exists():
                st.download_button(
                    "⬇️ Excel (Swing+Scalping)", data=_xlsx.read_bytes(),
                    file_name=_xlsx.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="perf_dl_xlsx")


# ===========================================================================
# TAB 1 — SCALPING
# ===========================================================================
with tab_scalping:
    render_scalping_tab(df_all, selected_date, active_api_key)


# ===========================================================================
# TAB 2 — SWING TRADING
# ===========================================================================
with tab_swing:
    render_swing_tab(df_all, selected_date, active_api_key, signal_filter)

    # Ticker detail (driven by sidebar selectbox)
    st.divider()
    st.markdown("### Detail Ticker")
    if not selected_ticker:
        st.info("Pilih ticker dari sidebar untuk melihat detail chart dan analisis.")
    else:
        ticker_rows = df_all[df_all["ticker"] == selected_ticker]
        if ticker_rows.empty:
            st.warning(f"Data untuk {selected_ticker} tidak ditemukan.")
        else:
            render_ticker_detail(ticker_rows.iloc[0], selected_date, active_api_key, key_prefix="sw_")


# ===========================================================================
# TAB 3 — LONG TERM
# ===========================================================================
with tab_longterm:
    render_longterm_tab(df_all, selected_date, active_api_key)


# ===========================================================================
# TAB — SMART MONEY SCREENER
# ===========================================================================
with tab_smart:
    render_smart_money_tab(df_all, selected_date)


# ===========================================================================
# TAB — NAIK / TURUN BERUNTUN (consecutive streaks, bullish/sideways only)
# ===========================================================================
with tab_streak:
    render_streak_tab(selected_date)


# ===========================================================================
# TAB 4 — SEARCH EMITEN
# ===========================================================================
with tab_search:
    st.markdown("### 🔍 Cari Emiten")

    universe = get_search_universe(selected_date)

    # Selectbox is natively searchable in Streamlit; add company names to options
    search_col, _spacer = st.columns([2, 3])
    with search_col:
        search_ticker = st.selectbox(
            "Ketik kode atau nama emiten:",
            options=[""] + universe,
            format_func=lambda t: "— pilih ticker —" if t == "" else format_ticker_option(t),
            key="search_box",
        )

    # Free-text fallback — normalize and try to match
    manual_input_col, _btn_col = st.columns([2, 1])
    with manual_input_col:
        manual_input = st.text_input(
            "Atau ketik langsung (mis. BBCA, BBCA.JK):",
            placeholder="BBCA",
            key="manual_search",
        )

    # Resolve final ticker
    final_ticker: str | None = None
    if search_ticker:
        final_ticker = search_ticker
    elif manual_input.strip():
        candidate = normalize_ticker(manual_input.strip())
        if candidate in universe:
            final_ticker = candidate
        else:
            st.warning(
                f"**{candidate}** tidak ditemukan di universe. "
                f"Pastikan ticker benar atau tambahkan ke `stock_scanner/configs/issuers.csv`."
            )

    if final_ticker:
        with st.spinner(f"Memuat data {final_ticker}…"):
            ctx = load_ticker_context(final_ticker, selected_date)

        company_name = get_company_name(final_ticker)
        sector = get_sector(final_ticker)
        sector_note = f" · {sector}" if sector else ""

        # Header
        st.markdown(
            f"## {final_ticker}"
            f"<div class='company-subtitle'>{company_name}{sector_note}</div>",
            unsafe_allow_html=True,
        )

        if not ctx["found_in_scan"]:
            st.info(
                f"**{final_ticker}** tidak ditemukan dalam hasil scan tanggal **{selected_date}**. "
                f"Menampilkan data yang tersedia (chart, broker, shareholder)."
            )

        # If signal data exists, use render_ticker_detail; otherwise show partial panel
        if ctx["signal_row"] is not None:
            render_ticker_detail(ctx["signal_row"], selected_date, active_api_key, key_prefix="srch_")
        else:
            # Partial panel: chart + shareholders + broker only
            srch_tabs = st.tabs(["📈 Chart", "🏢 Shareholders", "📋 Broker"])

            with srch_tabs[0]:
                df_raw = ctx["raw_ohlcv"]
                if df_raw.empty:
                    st.warning("Tidak ada data OHLCV tersedia. Jalankan incremental update terlebih dahulu.")
                else:
                    st.plotly_chart(
                        price_chart(df_raw, final_ticker, signal_date=selected_date),
                        use_container_width=True,
                        key=f"srch_chart_{final_ticker}",
                    )
                _chart_debug(final_ticker)

            with srch_tabs[1]:
                render_shareholders_section(final_ticker, selected_date)

            with srch_tabs[2]:
                render_broker_section(final_ticker, selected_date)


# ===========================================================================
# TAB 5 — HISTORY
# ===========================================================================
with tab_history:
    st.markdown("### Signal Terkuat (Lintas Tanggal)")

    h_col1, h_col2 = st.columns([2, 1])
    with h_col1:
        hist_signal_filter = st.multiselect(
            "Filter Signal",
            options=["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"],
            default=["BREAKOUT", "PRE_MARKUP", "WATCH"],
            key="hist_sig",
        )
    with h_col2:
        hist_ticker = st.text_input("Filter Ticker", placeholder="Mis: BBCA.JK", key="hist_tkr")

    df_hist = load_all_ranked(
        min_signal=hist_signal_filter if hist_signal_filter else None,
        ticker_filter=hist_ticker.strip() if hist_ticker.strip() else None,
        limit_rows=200,
    )

    if df_hist.empty:
        st.info("Belum ada history atau tidak ada data yang cocok dengan filter.")
    else:
        st.plotly_chart(history_timeline(df_hist), use_container_width=True, key="hist_timeline")

        st.markdown("**10 Signal Teratas**")
        top10 = df_hist.head(10).copy()
        if "date" in top10.columns:
            top10["date"] = pd.to_datetime(top10["date"]).dt.strftime("%Y-%m-%d")
        for col in ["total_score", "rsi14", "vol_ratio_20d", "pct_from_52w_high"]:
            if col in top10.columns:
                top10[col] = top10[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")

        # Add company names to history table
        if "ticker" in top10.columns:
            top10.insert(2, "company", top10["ticker"].apply(get_company_name))

        show_df(
            top10,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date":              st.column_config.TextColumn("Tanggal",       width="small"),
                "ticker":            st.column_config.TextColumn("Ticker",        width="small"),
                "company":           st.column_config.TextColumn("Nama Emiten"),
                "signal":            st.column_config.TextColumn("Signal",        width="small"),
                "total_score":       st.column_config.TextColumn("Score",         width="small"),
                "close":             st.column_config.NumberColumn("Close",       format="%.0f"),
                "rsi14":             st.column_config.TextColumn("RSI14",         width="small"),
                "vol_ratio_20d":     st.column_config.TextColumn("Vol Ratio",     width="small"),
                "pct_from_52w_high": st.column_config.TextColumn("52w High%",     width="small"),
            },
        )


# ===========================================================================
# TAB — LEARNING AGENT (read-only knowledge_base view)
# ===========================================================================
with tab_knowledge_base:
    render_knowledge_base_tab()


# ===========================================================================
# TAB — DAILY MOVERS >10% (read-only)
# ===========================================================================
with tab_daily_movers:
    render_daily_movers_tab()


# ===========================================================================
# TAB — AI LAB (experimental, read-only)
# ===========================================================================
with tab_ai_lab:
    render_ai_lab_tab()


# ===========================================================================
# TAB — STOCK DICTIONARY (reference, read-only)
# ===========================================================================
with tab_dictionary:
    render_stock_dictionary_tab()
