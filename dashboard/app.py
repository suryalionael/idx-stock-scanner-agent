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
    monthly_holders_chart,
    price_chart,
    price_chart_longterm,
    score_radar,
    shareholder_pie,
)
from dashboard.data_loader import (
    available_dates,
    available_dates_unified,
    get_table_df,
    is_remote_mode,
    load_all_ranked,
    load_all_tickers_for_date,
    load_all_tickers_unified,
    load_broker_for_ticker,
    load_broker_history,
    load_published_payload,
    load_raw,
    latest_ranked_date,
    load_fundamentals_for_date,
    get_fundamental_row,
    load_news_articles_for_ticker,
)
from dashboard.explain import explain_signal_llm
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
from stock_scanner.reference.issuers import get_company_name, get_sector, ticker_display

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
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #0f172a; }
  [data-testid="stSidebar"]          { background-color: #1e293b; }
  [data-testid="stHeader"]           { background-color: #0f172a; }
  .metric-card {
    background: #1e293b; border-radius: 10px;
    padding: 14px 18px; text-align: center;
  }
  .metric-card .label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
  .metric-card .value { font-size: 28px; font-weight: 700; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
  }
  .badge-BREAKOUT   { background:#166534; color:#4ade80; }
  .badge-PRE_MARKUP { background:#1e3a5f; color:#38bdf8; }
  .badge-WATCH      { background:#7c2d12; color:#fb923c; }
  .badge-AVOID      { background:#450a0a; color:#f87171; }
  .badge-NONE       { background:#1e293b; color:#64748b; }
  .company-subtitle { font-size: 13px; color: #64748b; margin-top: -6px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

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
    """Per-cell CSS for net_lot: green (+), red (-), grey (0/NaN)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "color: #4ade80; font-weight: 600"
    if v < 0:
        return "color: #ef4444; font-weight: 600"
    return "color: #94a3b8"


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
            f'<span style="background:#14532d;color:#4ade80;padding:2px 8px;'
            f'border-radius:8px;font-size:12px">📰 News: {n} artikel · skor {score_str}</span>'
        )
    elif news_status == "none":
        badges.append(
            '<span style="background:#1e293b;color:#94a3b8;padding:2px 8px;'
            'border-radius:8px;font-size:12px">📰 News: tidak ada berita 3 hari ini</span>'
        )
    elif news_status == "failed":
        badges.append(
            '<span style="background:#450a0a;color:#f87171;padding:2px 8px;'
            'border-radius:8px;font-size:12px">⚠️ News: data unavailable today</span>'
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
            f'<span style="background:#1e3a5f;color:#38bdf8;padding:2px 8px;'
            f'border-radius:8px;font-size:12px">📊 Fundamental: {summary}</span>'
        )
    elif fund_status == "partial":
        badges.append(
            '<span style="background:#292524;color:#fb923c;padding:2px 8px;'
            'border-radius:8px;font-size:12px">📊 Fundamental: data parsial</span>'
        )
    elif fund_status == "missing":
        badges.append(
            '<span style="background:#1e293b;color:#64748b;padding:2px 8px;'
            'border-radius:8px;font-size:12px">📊 Fundamental: belum tersedia</span>'
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
    st.dataframe(
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
                st.dataframe(
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
            st.dataframe(
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
# Broker section (reusable, with net_lot coloring)
# ---------------------------------------------------------------------------

def render_broker_section(ticker: str, scan_date: str) -> None:
    """Render broker activity panel with colored net_lot.

    Fully isolated: any exception shows a friendly message instead of crashing.
    """
    try:
        with st.spinner("Memuat data broker..."):
            df_broker = load_broker_for_ticker(ticker, scan_date, use_mock=True)
    except Exception as exc:
        st.warning(f"⚠️ Data broker tidak dapat dimuat. ({exc})")
        return

    if df_broker.empty:
        st.caption("Tidak ada data broker untuk ticker ini.")
        return

    # Net total summary badge
    if "net_lot" in df_broker.columns:
        net_total = pd.to_numeric(df_broker["net_lot"], errors="coerce").fillna(0).sum()
        color = "#4ade80" if net_total >= 0 else "#ef4444"
        sign = "+" if net_total >= 0 else ""
        st.markdown(
            f'<span style="color:{color};font-weight:600">'
            f'Net total: {sign}{net_total:,.0f} lot</span>',
            unsafe_allow_html=True,
        )

    # Mirror bar chart
    try:
        st.plotly_chart(
            broker_chart(df_broker, ticker),
            use_container_width=True,
            key=f"broker_{ticker}_{scan_date}",
        )
    except Exception as e:
        st.caption(f"Chart broker tidak dapat dimuat: {e}")

    # Detailed table with colored net_lot
    with st.expander("Tabel detail broker"):
        try:
            disp_cols = [c for c in ["broker_code", "broker_name", "buy_lot", "sell_lot", "net_lot"]
                         if c in df_broker.columns]
            tbl = df_broker[disp_cols].copy()

            # Coerce lot columns to numeric before any styling
            for col in ["buy_lot", "sell_lot", "net_lot"]:
                if col in tbl.columns:
                    tbl[col] = pd.to_numeric(tbl[col], errors="coerce")

            # Try styled render first; fall back to plain table on any Styler error
            try:
                styled = _style_broker_table(tbl)
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                # Fallback: add a formatted text column so colors are still implied
                tbl_plain = tbl.copy()
                if "net_lot" in tbl_plain.columns:
                    tbl_plain["net_lot"] = tbl_plain["net_lot"].apply(
                        lambda x: (f"+{x:,.0f}" if x > 0 else f"{x:,.0f}")
                        if pd.notna(x) else "—"
                    )
                st.dataframe(tbl_plain, use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"Tabel broker tidak dapat dimuat: {e}")


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
                st.dataframe(
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

    Ketika tidak ada file broker nyata di data/broker/, generates mock multi-day
    data menggunakan PlaceholderBrokerFetcher (deterministik per ticker+date)
    supaya Broker Intelligence section tetap bisa ditampilkan untuk demo.
    """
    from stock_scanner.pipeline.broker_intelligence import compute_broker_intelligence

    broker_df = load_broker_history(ticker, n_days=20)

    if broker_df.empty:
        # Tidak ada data nyata — generate mock untuk demo
        from stock_scanner.pipeline.broker_summary import PlaceholderBrokerFetcher
        fetcher = PlaceholderBrokerFetcher()
        base = pd.Timestamp(scan_date) if scan_date else pd.Timestamp.today()
        frames: list[pd.DataFrame] = []
        trading_days = 0
        for i in range(35):                    # ~5 weeks back
            d = base - pd.Timedelta(days=i)
            if d.weekday() >= 5:               # skip Sat / Sun
                continue
            day_str = d.strftime("%Y-%m-%d")
            try:
                df_day = fetcher.fetch(ticker, day_str).copy()
                df_day["date"] = day_str
                frames.append(df_day)
            except Exception:
                pass
            trading_days += 1
            if trading_days >= 20:
                break
        if frames:
            broker_df = pd.concat(frames, ignore_index=True)

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
    tbl_cols = ["ticker", "scalping_score", "scalping_label", "scalping_reason",
                "close", "vol_ratio_20d", "roc5", "rsi14", "momentum_score",
                "atr_breakout", "vol_spike", "entry_low", "entry_high", "tp_low", "cutloss"]
    disp_cols = [c for c in tbl_cols if c in df_show.columns]
    tbl = df_show[disp_cols].copy()

    # Format
    for col in ["close", "entry_low", "entry_high", "tp_low", "cutloss"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "-")
    for col in ["vol_ratio_20d", "roc5", "rsi14", "momentum_score", "scalping_score"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "-")

    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        height=min(38 * len(tbl) + 40, 500),
        column_config={
            "ticker":          st.column_config.TextColumn("Ticker",         width="small"),
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
            st.dataframe(pd.DataFrame(level_rows), use_container_width=True,
                         hide_index=True, height=145)

    with col_right:
        # Price chart
        df_raw = load_raw(ticker)
        st.plotly_chart(
            price_chart(df_raw, ticker, signal_date=scan_date, lookback_days=60),
            use_container_width=True,
            key=f"scalp_chart_{ticker}_{scan_date}",
        )

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


# ---------------------------------------------------------------------------
# Swing Tab
# ---------------------------------------------------------------------------

def render_swing_tab(df_all: pd.DataFrame, scan_date: str, api_key: str | None,
                     signal_filter: list[str]) -> None:
    """Render the 🔄 Swing Trading tab content — refactored Today Overview."""
    st.markdown("### 🔄 Swing Trading — Setup Teknikal 3–10 Hari")

    # Signal distribution
    sig_counts_all = df_all["signal"].value_counts() if "signal" in df_all.columns else pd.Series(dtype=int)
    total_tickers  = len(df_all)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, key, color in [
        (c1, "Total",      None,         "#94a3b8"),
        (c2, "BREAKOUT",   "BREAKOUT",   "#4ade80"),
        (c3, "PRE_MARKUP", "PRE_MARKUP", "#38bdf8"),
        (c4, "WATCH",      "WATCH",      "#fb923c"),
        (c5, "AVOID",      "AVOID",      "#f87171"),
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
        df_table = get_table_df(df_filtered)
        display = df_table.copy()
        for col in ["total_score", "enhanced_total_score", "trend_score", "momentum_score",
                    "breakout_score", "volume_score", "penalty_score", "news_score", "foreign_score"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        for col in ["rsi14", "vol_ratio_20d", "pct_from_52w_high", "adx"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        st.caption("💡 Klik baris untuk melihat Detail Ticker di bawah")
        tbl_event = st.dataframe(
            display,
            use_container_width=True, hide_index=True, height=320,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "ticker":               st.column_config.TextColumn("Ticker",      width="small"),
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
        # Table row click → queue ticker change (consumed next rerun before sidebar renders)
        sel_rows = tbl_event.selection.rows if tbl_event.selection else []
        if sel_rows:
            row_idx = sel_rows[0]
            if row_idx < len(df_table):
                clicked = df_table.iloc[row_idx]["ticker"]
                if clicked != st.session_state.get("active_ticker"):
                    st.session_state["_pending_ticker"] = clicked
                    st.rerun()
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
    tbl_cols_lt = [
        "ticker", "long_term_score", "long_term_label", "valuation_status",
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

    st.caption("💡 Klik baris untuk melihat Detail Fundamental di bawah")
    lt_event = st.dataframe(
        tbl_lt,
        use_container_width=True, hide_index=True,
        height=min(38 * len(tbl_lt) + 40, 520),
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "ticker":              st.column_config.TextColumn("Ticker",        width="small"),
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
    # LT table row click → queue detail ticker change
    lt_sel_rows = lt_event.selection.rows if lt_event.selection else []
    if lt_sel_rows:
        lt_row_idx = lt_sel_rows[0]
        if lt_row_idx < len(df_lt_show):
            lt_clicked = df_lt_show.iloc[lt_row_idx]["ticker"]
            if lt_clicked != st.session_state.get("lt_active_ticker"):
                st.session_state["_lt_pending_ticker"] = lt_clicked
                st.rerun()

    # Expandable ticker detail
    st.divider()
    st.markdown("#### Detail Fundamental Ticker")
    lt_ticker_list = df_lt_show["ticker"].tolist()

    # Consume any pending LT table click before selectbox renders
    _lt_pending = st.session_state.pop("_lt_pending_ticker", None)
    if _lt_pending and _lt_pending in lt_ticker_list:
        st.session_state["lt_active_ticker"] = _lt_pending

    _lt_active = st.session_state.get("lt_active_ticker")
    _lt_idx = lt_ticker_list.index(_lt_active) if (_lt_active and _lt_active in lt_ticker_list) else 0

    sel_lt = st.selectbox(
        "Pilih ticker:",
        options=lt_ticker_list,
        index=_lt_idx,
        format_func=lambda t: f"{t.replace('.JK','')} — {get_company_name(t)}",
        key="lt_detail_select",
    )
    st.session_state["lt_active_ticker"] = sel_lt
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

    # Multi-period financial statement comparison (lazy, on demand)
    st.markdown("")
    with st.expander("📑 Laporan Keuangan Multi-Periode (dari yfinance)", expanded=False):
        with st.spinner("Mengambil data laporan keuangan..."):
            fin_data = _fetch_financial_comparison(ticker)

        if fin_data["status"] == "failed":
            st.warning(f"Gagal mengambil data: {fin_data.get('error', '—')}")
        else:
            if fin_data["status"] == "partial":
                st.caption("⚠️ Data parsial — beberapa periode tidak tersedia.")

            # YoY summary
            yoy = fin_data.get("yoy", {})
            if yoy:
                st.markdown("**YoY Changes (terbaru vs tahun sebelumnya):**")
                yoy_items = [
                    ("Revenue",     yoy.get("revenue_chg")),
                    ("Net Income",  yoy.get("net_income_chg")),
                    ("Total Asset", yoy.get("asset_chg")),
                    ("Equity",      yoy.get("equity_chg")),
                    ("Op. Cash Flow", yoy.get("ocf_chg")),
                ]
                yoy_cols = st.columns(len(yoy_items))
                for col, (label, val) in zip(yoy_cols, yoy_items):
                    with col:
                        if val is not None:
                            color = "#4ade80" if val >= 0 else "#f87171"
                            st.markdown(
                                f'<div style="text-align:center">'
                                f'<div style="font-size:11px;color:#94a3b8">{label}</div>'
                                f'<div style="font-size:16px;font-weight:700;color:{color}">'
                                f'{val:+.1f}%</div></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption(f"{label}: —")

            st.markdown("")

            # Charts for key metrics
            chart_metrics = [
                ("revenue",      "Revenue"),
                ("net_income",   "Net Income"),
                ("total_equity", "Total Equity"),
                ("op_cash_flow", "Operating Cash Flow"),
            ]
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
        bi_c1, bi_c2, bi_c3 = st.columns(3)
        with bi_c1:
            fn_1d = bi.get("foreign_net_buy_1d", 0)
            fn_5d = bi.get("foreign_net_buy_5d", 0)
            color_1d = "#4ade80" if fn_1d >= 0 else "#ef4444"
            color_5d = "#4ade80" if fn_5d >= 0 else "#ef4444"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">🌍 Foreign Net (1d)</div>'
                f'<div class="value" style="color:{color_1d};font-size:20px">'
                f'{"+" if fn_1d >= 0 else ""}{fn_1d:,.0f} lot</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with bi_c2:
            color_fn5 = "#4ade80" if fn_5d >= 0 else "#ef4444"
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
            color_big = "#4ade80" if big_5d >= 0 else "#ef4444"
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
                    st.dataframe(
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
                    st.dataframe(
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
with st.sidebar:
    st.markdown("## 📈 IDX Scanner")

    # Mode badge
    if _remote_mode:
        st.markdown(
            '<div style="background:#1e3a5f;border-radius:6px;padding:4px 10px;'
            'font-size:11px;color:#38bdf8;margin-bottom:6px;">'
            '☁️ Mode: <b>Online</b> — data dari GitHub</div>',
            unsafe_allow_html=True,
        )
        if _remote_payload:
            gen_at = _remote_payload.get("generated_at", "—")
            # Distinguish: freshly fetched from remote vs loaded from local fallback
            scan_date_str = _remote_payload.get("scan_date", "")
            st.caption(f"Data: {scan_date_str} · {gen_at[:10] if gen_at != '—' else '—'}")
        else:
            st.warning(
                "⚠️ Data tidak tersedia.\n\n"
                "Pastikan `data/published/latest_scan.json` sudah di-commit ke repo, "
                "atau set secret `REMOTE_DATA_URL` dengan URL yang benar."
            )
    else:
        st.markdown(
            '<div style="background:#14532d;border-radius:6px;padding:4px 10px;'
            'font-size:11px;color:#4ade80;margin-bottom:6px;">'
            '💻 Mode: <b>Lokal</b></div>',
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

    selected_date = st.selectbox("Tanggal Scan", options=all_dates, index=0)
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

    # Mini signal summary
    if "signal" in df_all.columns:
        sig_counts = df_all["signal"].value_counts()
        sc1, sc2, sc3 = st.columns(3)
        for col, sig, color in [
            (sc1, "BREAKOUT",   "#4ade80"),
            (sc2, "PRE_MARKUP", "#38bdf8"),
            (sc3, "WATCH",      "#fb923c"),
        ]:
            cnt = int(sig_counts.get(sig, 0))
            with col:
                st.markdown(
                    f'<div style="text-align:center;background:#0f172a;border-radius:6px;padding:4px 0">'
                    f'<div style="font-size:10px;color:#94a3b8">{sig[:3]}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{color}">{cnt}</div>'
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

    # Consume pending table-click ticker BEFORE selectbox renders so the
    # selectbox reflects the row the user just clicked on.
    _pending = st.session_state.pop("_pending_ticker", None)
    if _pending and _pending in sidebar_tickers:
        st.session_state["active_ticker"] = _pending

    selected_ticker: str | None = None
    if not sidebar_tickers:
        st.caption("Tidak ada ticker dengan signal yang dipilih.")
    else:
        _active = st.session_state.get("active_ticker")
        _sb_idx = sidebar_tickers.index(_active) if (_active and _active in sidebar_tickers) else 0
        selected_ticker = st.selectbox(
            "Pilih ticker:",
            options=sidebar_tickers,
            index=_sb_idx,
            format_func=_fmt_sidebar_ticker,
            label_visibility="collapsed",
        )
        st.session_state["active_ticker"] = selected_ticker

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
# MAIN TABS
# ---------------------------------------------------------------------------
tab_scalping, tab_swing, tab_longterm, tab_search, tab_history = st.tabs(
    ["📈 Scalping", "🔄 Swing Trading", "📊 Long Term", "🔍 Search Emiten", "🕐 History"]
)


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

    # Ticker detail — driven by sidebar selectbox OR table row click.
    # session_state["active_ticker"] is the single source of truth:
    #   • updated by sidebar selectbox (see sidebar block above)
    #   • overridden by table row click via the _pending_ticker pattern
    st.divider()
    st.markdown("### Detail Ticker")
    effective_ticker = st.session_state.get("active_ticker") or selected_ticker
    if not effective_ticker:
        st.info("Pilih ticker dari sidebar atau klik baris pada tabel di atas.")
    else:
        ticker_rows = df_all[df_all["ticker"] == effective_ticker]
        if ticker_rows.empty:
            st.warning(f"Data untuk {effective_ticker} tidak ditemukan.")
        else:
            render_ticker_detail(ticker_rows.iloc[0], selected_date, active_api_key, key_prefix="sw_")


# ===========================================================================
# TAB 3 — LONG TERM
# ===========================================================================
with tab_longterm:
    render_longterm_tab(df_all, selected_date, active_api_key)


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

        st.dataframe(
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
