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

from dashboard.charts import broker_chart, history_timeline, price_chart, score_radar
from dashboard.data_loader import (
    available_dates,
    get_table_df,
    load_all_ranked,
    load_all_tickers_for_date,
    load_broker_for_ticker,
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
  .ticker-row {
    padding: 4px 8px; border-radius: 6px;
    font-size: 13px; cursor: pointer;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Signal color/emoji helper
# ---------------------------------------------------------------------------
_SIG_EMOJI = {
    "BREAKOUT":   "🟢",
    "PRE_MARKUP": "🔵",
    "WATCH":      "🟠",
    "AVOID":      "🔴",
    "NONE":       "⚪",
}


def _sig_label(sig: str) -> str:
    return _SIG_EMOJI.get(sig, "⚪") + " " + sig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 IDX Scanner")
    st.divider()

    # Date selector — union of ranked + signals dates
    all_dates = available_dates()
    if not all_dates:
        st.warning(
            "Belum ada data scan.\n\nJalankan dulu:\n"
            "```\npython -m stock_scanner.pipeline.run_daily_scan\n```"
        )
        st.stop()

    selected_date = st.selectbox("Tanggal Scan", options=all_dates, index=0)

    st.divider()

    # Load ALL tickers for selected date (signals file → fallback to ranked)
    df_all = load_all_tickers_for_date(selected_date)

    if df_all.empty:
        st.error(f"Tidak ada data untuk {selected_date}.")
        st.stop()

    # Signal distribution mini summary in sidebar
    if "signal" in df_all.columns:
        sig_counts = df_all["signal"].value_counts()
        cols_sig = st.columns(3)
        for i, (sig, color) in enumerate([
            ("BREAKOUT", "#4ade80"),
            ("PRE_MARKUP", "#38bdf8"),
            ("WATCH", "#fb923c"),
        ]):
            with cols_sig[i % 3]:
                cnt = int(sig_counts.get(sig, 0))
                st.markdown(
                    f'<div style="text-align:center;background:#0f172a;border-radius:6px;padding:4px 0">'
                    f'<div style="font-size:10px;color:#94a3b8">{sig[:3]}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{color}">{cnt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # Signal filter for sidebar ticker list
    st.markdown("**Filter Signal**")
    signal_filter = st.multiselect(
        "Tampilkan",
        options=["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"],
        default=["BREAKOUT", "PRE_MARKUP", "WATCH"],
        label_visibility="collapsed",
    )

    # Build filtered ticker list
    if "signal" in df_all.columns and signal_filter:
        df_sidebar = df_all[df_all["signal"].isin(signal_filter)].copy()
    else:
        df_sidebar = df_all.copy()

    # Sort by total_score desc
    if "total_score" in df_sidebar.columns:
        df_sidebar = df_sidebar.sort_values("total_score", ascending=False)

    sidebar_tickers = df_sidebar["ticker"].tolist() if not df_sidebar.empty else []

    # Ticker selector in sidebar — shows signal alongside name
    def _fmt_ticker(t: str) -> str:
        rows = df_sidebar[df_sidebar["ticker"] == t]
        if rows.empty:
            rows = df_all[df_all["ticker"] == t]
        sig = rows["signal"].values[0] if not rows.empty else "NONE"
        score = rows["total_score"].values[0] if (not rows.empty and "total_score" in rows.columns) else 0.0
        try:
            score_str = f"{float(score):.1f}"
        except (ValueError, TypeError):
            score_str = "—"
        return f"{_SIG_EMOJI.get(sig, '⚪')} {t}  ({score_str})"

    st.divider()
    st.markdown(f"**Daftar Saham** ({len(sidebar_tickers)} ticker)")

    if not sidebar_tickers:
        st.caption("Tidak ada ticker dengan signal yang dipilih.")
        selected_ticker = None
    else:
        selected_ticker = st.selectbox(
            "Pilih ticker:",
            options=sidebar_tickers,
            format_func=_fmt_ticker,
            label_visibility="collapsed",
        )

    st.divider()

    # API Key
    api_key_input = st.text_input(
        "Anthropic API Key (opsional)",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Jika diisi, penjelasan AI menggunakan Claude. Kosong = rule-based.",
    )
    active_api_key = api_key_input.strip() or None
    st.caption(f"Mode explain: {'🤖 Claude API' if active_api_key else '📋 Rule-based'}")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_today, tab_history = st.tabs(["📊 Today Overview", "🕐 History"])


# ===========================================================================
# Helper: render ticker detail panel
# ===========================================================================
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
            ("Trend",        "trend_score"),
            ("Momentum",     "momentum_score"),
            ("Breakout",     "breakout_score"),
            ("Volume",       "volume_score"),
            ("Penalty",      "penalty_score"),
            ("**Total**",    "total_score"),
            ("**Enhanced**", "enhanced_total_score"),
            ("News",         "news_score"),
            ("Foreign Flow", "foreign_score"),
        ]
        for label, col in score_items:
            val = row.get(col)
            val_str = f"{float(val):.1f}" if val is not None and pd.notna(val) else "N/A"
            st.markdown(f"{label}: `{val_str}/10`")

        st.markdown("")
        st.markdown("**Metrik Teknikal**")
        metrics = [
            ("Close",           "close",              "Rp {:.0f}"),
            ("RSI14",           "rsi14",              "{:.1f}"),
            ("ADX",             "adx",                "{:.1f}"),
            ("Vol Ratio 20d",   "vol_ratio_20d",      "{:.2f}x"),
            ("% dari 52w High", "pct_from_52w_high",  "{:.1f}%"),
            ("News Score",      "news_sentiment_score", "{:.2f}"),
            ("News (3d)",       "news_count_3d",       "{:.0f} berita"),
        ]
        for label, col, fmt in metrics:
            val = row.get(col)
            try:
                val_str = fmt.format(float(val)) if val is not None and pd.notna(val) else "N/A"
            except (ValueError, TypeError):
                val_str = "N/A"
            st.text(f"{label}: {val_str}")

        bool_metrics = [
            ("Supertrend Bullish", "supertrend_bullish"),
            ("Squeeze On",         "squeeze_on"),
            ("ATR Breakout",       "atr_breakout"),
            ("Vol Spike",          "vol_spike"),
            ("OBV Trend Up",       "obv_trend"),
        ]
        for label, col in bool_metrics:
            val = row.get(col)
            st.text(f"{label}: {'✅' if val else '—'}")

    with right:
        st.markdown("**Skor Visual**")
        st.plotly_chart(score_radar(row, ticker), use_container_width=True, key=f"radar_{ticker}_{scan_date}")

    # --- Price chart ---
    st.markdown("**Chart Harga (120 hari terakhir)**")
    df_raw = load_raw(ticker)
    st.plotly_chart(
        price_chart(df_raw, ticker, signal_date=scan_date),
        use_container_width=True,
        key=f"chart_{ticker}_{scan_date}",
    )

    # --- Broker activity panel ---
    st.markdown("**Aktivitas Broker (Top 10)**")
    with st.spinner("Memuat data broker..."):
        df_broker = load_broker_for_ticker(ticker, scan_date, use_mock=True)

    if df_broker.empty:
        st.caption("Tidak ada data broker untuk ticker ini.")
    else:
        # Summary net lot
        if "net_lot" in df_broker.columns:
            net_total = pd.to_numeric(df_broker["net_lot"], errors="coerce").fillna(0).sum()
            color = "#4ade80" if net_total >= 0 else "#f87171"
            sign = "+" if net_total >= 0 else ""
            st.markdown(
                f'<span style="color:{color};font-weight:600">Net foreign lot: {sign}{net_total:,.0f}</span>',
                unsafe_allow_html=True,
            )
        st.plotly_chart(
            broker_chart(df_broker, ticker),
            use_container_width=True,
            key=f"broker_{ticker}_{scan_date}",
        )
        with st.expander("Tabel detail broker"):
            disp_cols = [c for c in ["broker_code", "broker_name", "buy_lot", "sell_lot", "net_lot"]
                         if c in df_broker.columns]
            st.dataframe(df_broker[disp_cols], use_container_width=True, hide_index=True)

    # --- AI explanation ---
    st.markdown("**Penjelasan Sinyal**")
    with st.spinner("Membuat penjelasan..."):
        explanation = explain_signal_llm(row, api_key=api_key)
    st.markdown(explanation)


# ===========================================================================
# TAB 1 — TODAY OVERVIEW
# ===========================================================================
with tab_today:
    st.markdown(f"### Scan: {selected_date}")

    # Summary cards (based on ALL tickers from signals)
    signal_counts = df_all["signal"].value_counts() if "signal" in df_all.columns else pd.Series(dtype=int)
    total_tickers = len(df_all)

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

    # Signal table — filtered by sidebar signal_filter
    if "signal" in df_all.columns and signal_filter:
        df_filtered = df_all[df_all["signal"].isin(signal_filter)].copy()
    else:
        df_filtered = df_all.copy()

    if df_filtered.empty:
        st.info("Tidak ada ticker yang cocok dengan filter signal yang dipilih.")
    else:
        df_table = get_table_df(df_filtered)
        display = df_table.copy()

        for col in ["total_score", "enhanced_total_score", "trend_score", "momentum_score",
                    "breakout_score", "volume_score", "penalty_score",
                    "news_score", "foreign_score"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        for col in ["rsi14", "vol_ratio_20d", "pct_from_52w_high", "adx"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=320,
            column_config={
                "ticker":                st.column_config.TextColumn("Ticker",         width="small"),
                "signal":                st.column_config.TextColumn("Signal",         width="small"),
                "total_score":           st.column_config.TextColumn("Score",          width="small"),
                "enhanced_total_score":  st.column_config.TextColumn("Enh.Score",      width="small"),
                "news_score":            st.column_config.TextColumn("News",           width="small"),
                "foreign_score":         st.column_config.TextColumn("Foreign",        width="small"),
                "close":                 st.column_config.NumberColumn("Close",        format="%.0f"),
                "rsi14":                 st.column_config.TextColumn("RSI14",          width="small"),
                "adx":                   st.column_config.TextColumn("ADX",            width="small"),
                "vol_ratio_20d":         st.column_config.TextColumn("Vol Ratio",      width="small"),
                "pct_from_52w_high":     st.column_config.TextColumn("52w High%",      width="small"),
                "supertrend_bullish":    st.column_config.CheckboxColumn("Supertrend"),
                "squeeze_on":            st.column_config.CheckboxColumn("Squeeze"),
                "atr_breakout":          st.column_config.CheckboxColumn("ATR Break"),
                "vol_spike":             st.column_config.CheckboxColumn("Vol Spike"),
            },
        )

    # --- Ticker detail (driven by sidebar selectbox) ---
    st.divider()
    st.markdown("### Detail Ticker")

    if not selected_ticker:
        st.info("Pilih ticker dari sidebar untuk melihat detail chart dan analisis.")
    else:
        # Get the row from df_all (includes all signal types)
        ticker_rows = df_all[df_all["ticker"] == selected_ticker]
        if ticker_rows.empty:
            st.warning(f"Data untuk {selected_ticker} tidak ditemukan.")
        else:
            ticker_row = ticker_rows.iloc[0]
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
                "date":              st.column_config.TextColumn("Tanggal",   width="small"),
                "ticker":            st.column_config.TextColumn("Ticker",    width="small"),
                "signal":            st.column_config.TextColumn("Signal",    width="small"),
                "total_score":       st.column_config.TextColumn("Score",     width="small"),
                "close":             st.column_config.NumberColumn("Close",   format="%.0f"),
                "rsi14":             st.column_config.TextColumn("RSI14",     width="small"),
                "vol_ratio_20d":     st.column_config.TextColumn("Vol Ratio", width="small"),
                "pct_from_52w_high": st.column_config.TextColumn("52w High%", width="small"),
            },
        )
