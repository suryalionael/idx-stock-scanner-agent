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

from dashboard.charts import history_timeline, price_chart, score_radar
from dashboard.data_loader import (
    get_table_df,
    list_ranked_dates,
    load_all_ranked,
    load_ranked,
    load_raw,
)
from dashboard.explain import explain_signal_llm

# ---------------------------------------------------------------------------
# Page config — harus baris pertama setelah import
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
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: render ticker detail panel
# ---------------------------------------------------------------------------
def render_ticker_detail(row: pd.Series, scan_date: str, api_key: str | None) -> None:
    signal = str(row.get("signal", "NONE"))
    ticker = str(row.get("ticker", "?"))

    badge_class = f"badge-{signal}" if signal in ("BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID") else "badge-NONE"
    st.markdown(
        f'<h4>{ticker} &nbsp; <span class="badge {badge_class}">{signal}</span></h4>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("**Skor Komponen**")
        score_items = [
            ("Trend", "trend_score"),
            ("Momentum", "momentum_score"),
            ("Breakout", "breakout_score"),
            ("Volume", "volume_score"),
            ("Penalty", "penalty_score"),
            ("**Total**", "total_score"),
        ]
        for label, col in score_items:
            val = row.get(col)
            val_str = f"{float(val):.1f}" if val is not None and pd.notna(val) else "N/A"
            st.markdown(f"{label}: `{val_str}/10`")

        st.markdown("")
        st.markdown("**Metrik Teknikal**")
        metrics = [
            ("Close",           "close",              "Rp {:.0f}"),
            ("MA20",            "ma20",               "Rp {:.0f}"),
            ("MA50",            "ma50",               "Rp {:.0f}"),
            ("MA200",           "ma200",              "Rp {:.0f}"),
            ("RSI14",           "rsi14",              "{:.1f}"),
            ("Vol Ratio 20d",   "vol_ratio_20d",      "{:.2f}x"),
            ("% dari 52w High", "pct_from_52w_high",  "{:.1f}%"),
        ]
        for label, col, fmt in metrics:
            val = row.get(col)
            try:
                val_str = fmt.format(float(val)) if val is not None and pd.notna(val) else "N/A"
            except (ValueError, TypeError):
                val_str = "N/A"
            st.text(f"{label}: {val_str}")

        for label, col in [("ATR Breakout", "atr_breakout"), ("Vol Spike", "vol_spike"), ("OBV Trend Up", "obv_trend")]:
            val = row.get(col)
            st.text(f"{label}: {'✅' if val else '—'}")

    with right:
        st.markdown("**Skor Visual**")
        st.plotly_chart(score_radar(row, ticker), use_container_width=True, key=f"radar_{ticker}_{scan_date}")

    st.markdown("**Chart Harga (120 hari terakhir)**")
    df_raw = load_raw(ticker)
    st.plotly_chart(
        price_chart(df_raw, ticker, signal_date=scan_date),
        use_container_width=True,
        key=f"chart_{ticker}_{scan_date}",
    )

    st.markdown("**Penjelasan Sinyal**")
    with st.spinner("Membuat penjelasan..."):
        explanation = explain_signal_llm(row, api_key=api_key)
    st.markdown(explanation)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 IDX Scanner")
    st.divider()

    available_dates = list_ranked_dates()
    if not available_dates:
        st.warning(
            "Belum ada data scan.\n\nJalankan dulu:\n"
            "```\npython -m stock_scanner.pipeline.run_daily_scan\n```"
        )
        st.stop()

    selected_date = st.selectbox(
        "Tanggal Scan",
        options=available_dates,
        index=0,
    )

    st.divider()
    st.markdown("**Filter Signal**")
    signal_filter = st.multiselect(
        "Tampilkan Signal",
        options=["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"],
        default=["BREAKOUT", "PRE_MARKUP", "WATCH"],
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
# Load data
# ---------------------------------------------------------------------------
df_ranked = load_ranked(selected_date)
if df_ranked.empty:
    st.error(f"Tidak ada data untuk tanggal **{selected_date}**.")
    st.stop()

df_filtered = df_ranked[df_ranked["signal"].isin(signal_filter)].copy() if signal_filter else df_ranked.copy()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_today, tab_history = st.tabs(["📊 Today Overview", "🕐 History"])

# ===========================================================================
# TAB 1 — TODAY OVERVIEW
# ===========================================================================
with tab_today:
    st.markdown(f"### Scan: {selected_date}")

    # Summary cards
    signal_counts = df_ranked["signal"].value_counts()
    total_tickers = len(df_ranked)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, key, color in [
        (c1, "Total",      None,         "#94a3b8"),
        (c2, "BREAKOUT",   "BREAKOUT",   "#4ade80"),
        (c3, "PRE_MARKUP", "PRE_MARKUP", "#38bdf8"),
        (c4, "WATCH",      "WATCH",      "#fb923c"),
        (c5, "AVOID",      "AVOID",      "#f87171"),
    ]:
        count = total_tickers if key is None else int(signal_counts.get(key, 0))
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{count}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacing

    # Signal table
    if df_filtered.empty:
        st.info("Tidak ada ticker yang cocok dengan filter signal yang dipilih.")
    else:
        df_table = get_table_df(df_filtered)
        display = df_table.copy()

        for col in ["total_score", "trend_score", "momentum_score",
                    "breakout_score", "volume_score", "penalty_score"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        for col in ["rsi14", "vol_ratio_20d", "pct_from_52w_high"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=320,
            column_config={
                "ticker":           st.column_config.TextColumn("Ticker",    width="small"),
                "signal":           st.column_config.TextColumn("Signal",    width="small"),
                "total_score":      st.column_config.TextColumn("Score",     width="small"),
                "close":            st.column_config.NumberColumn("Close",   format="%.0f"),
                "rsi14":            st.column_config.TextColumn("RSI14",     width="small"),
                "vol_ratio_20d":    st.column_config.TextColumn("Vol Ratio", width="small"),
                "pct_from_52w_high":st.column_config.TextColumn("52w High%", width="small"),
                "atr_breakout":     st.column_config.CheckboxColumn("ATR Break"),
                "vol_spike":        st.column_config.CheckboxColumn("Vol Spike"),
            },
        )

    # Ticker detail
    st.divider()
    st.markdown("### Detail Ticker")

    all_tickers = df_ranked["ticker"].tolist()
    if not all_tickers:
        st.info("Tidak ada ticker tersedia.")
    else:
        def _ticker_label(t: str) -> str:
            rows = df_ranked[df_ranked["ticker"] == t]
            sig = rows["signal"].values[0] if not rows.empty else ""
            return f"{t}  [{sig}]"

        selected_ticker = st.selectbox(
            "Pilih ticker untuk detail:",
            options=all_tickers,
            format_func=_ticker_label,
        )
        if selected_ticker:
            ticker_row = df_ranked[df_ranked["ticker"] == selected_ticker].iloc[0]
            render_ticker_detail(ticker_row, selected_date, active_api_key)

# ===========================================================================
# TAB 2 — HISTORY
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

        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date":             st.column_config.TextColumn("Tanggal",   width="small"),
                "ticker":           st.column_config.TextColumn("Ticker",    width="small"),
                "signal":           st.column_config.TextColumn("Signal",    width="small"),
                "total_score":      st.column_config.TextColumn("Score",     width="small"),
                "close":            st.column_config.NumberColumn("Close",   format="%.0f"),
                "rsi14":            st.column_config.TextColumn("RSI14",     width="small"),
                "vol_ratio_20d":    st.column_config.TextColumn("Vol Ratio", width="small"),
                "pct_from_52w_high":st.column_config.TextColumn("52w High%", width="small"),
            },
        )
