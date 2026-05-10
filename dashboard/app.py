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
    history_timeline,
    monthly_holders_chart,
    price_chart,
    score_radar,
    shareholder_pie,
)
from dashboard.data_loader import (
    available_dates,
    get_table_df,
    load_all_ranked,
    load_all_tickers_for_date,
    load_broker_for_ticker,
    load_raw,
    latest_ranked_date,
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


def _color_net_lot(val):
    """Pandas Styler function: green for +, red for -, grey for 0/NaN."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "color: #4ade80; font-weight: 600"
    if v < 0:
        return "color: #ef4444; font-weight: 600"
    return "color: #94a3b8"


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
            for col in ["buy_lot", "sell_lot", "net_lot"]:
                if col in tbl.columns:
                    tbl[col] = pd.to_numeric(tbl[col], errors="coerce")

            if "net_lot" in tbl.columns:
                styled = tbl.style.applymap(_color_net_lot, subset=["net_lot"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(tbl, use_container_width=True, hide_index=True)
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

        st.markdown("**Chart Harga (120 hari terakhir)**")
        df_raw = load_raw(ticker)
        st.plotly_chart(
            price_chart(df_raw, ticker, signal_date=scan_date),
            use_container_width=True,
            key=f"{key_prefix}chart_{ticker}_{scan_date}",
        )

        st.markdown("**Penjelasan Sinyal**")
        with st.spinner("Membuat penjelasan..."):
            explanation = explain_signal_llm(row, api_key=api_key)
        st.markdown(explanation)

    # ---- TAB: Shareholders ----
    with tab_shareholders:
        render_shareholders_section(ticker, scan_date)

    # ---- TAB: Broker ----
    with tab_broker:
        render_broker_section(ticker, scan_date)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 IDX Scanner")
    st.divider()

    all_dates = available_dates()
    if not all_dates:
        st.warning(
            "Belum ada data scan.\n\nJalankan dulu:\n"
            "```\npython -m stock_scanner.pipeline.run_daily_scan\n```"
        )
        st.stop()

    selected_date = st.selectbox("Tanggal Scan", options=all_dates, index=0)
    st.divider()

    # Load ALL tickers for date
    df_all = load_all_tickers_for_date(selected_date)
    if df_all.empty:
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
# MAIN TABS
# ---------------------------------------------------------------------------
tab_today, tab_search, tab_history = st.tabs(
    ["📊 Today Overview", "🔍 Search Emiten", "🕐 History"]
)


# ===========================================================================
# TAB 1 — TODAY OVERVIEW
# ===========================================================================
with tab_today:
    st.markdown(f"### Scan: {selected_date}")

    # Summary cards
    sig_counts_all = df_all["signal"].value_counts() if "signal" in df_all.columns else pd.Series(dtype=int)
    total_tickers = len(df_all)

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

    # Signal table
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
                    "breakout_score", "volume_score", "penalty_score", "news_score", "foreign_score"]:
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
                "ticker":               st.column_config.TextColumn("Ticker",      width="small"),
                "signal":               st.column_config.TextColumn("Signal",      width="small"),
                "total_score":          st.column_config.TextColumn("Score",       width="small"),
                "enhanced_total_score": st.column_config.TextColumn("Enh.Score",   width="small"),
                "news_score":           st.column_config.TextColumn("News",        width="small"),
                "foreign_score":        st.column_config.TextColumn("Foreign",     width="small"),
                "close":                st.column_config.NumberColumn("Close",     format="%.0f"),
                "rsi14":                st.column_config.TextColumn("RSI14",       width="small"),
                "adx":                  st.column_config.TextColumn("ADX",         width="small"),
                "vol_ratio_20d":        st.column_config.TextColumn("Vol Ratio",   width="small"),
                "pct_from_52w_high":    st.column_config.TextColumn("52w High%",   width="small"),
                "supertrend_bullish":   st.column_config.CheckboxColumn("Supertrend"),
                "squeeze_on":           st.column_config.CheckboxColumn("Squeeze"),
                "atr_breakout":         st.column_config.CheckboxColumn("ATR Break"),
                "vol_spike":            st.column_config.CheckboxColumn("Vol Spike"),
            },
        )

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
            render_ticker_detail(ticker_rows.iloc[0], selected_date, active_api_key, key_prefix="ov_")


# ===========================================================================
# TAB 2 — SEARCH EMITEN
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
# TAB 3 — HISTORY
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
