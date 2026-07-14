"""Read-only dashboard view over the daily movers >10% mirror.

STRICTLY READ-ONLY: no writes, no refresh-from-dashboard fetch path, no path
into signal_engine.py / scanner_config.yaml / model promotion. This module
only ever reads the committed data/published/daily_movers.json mirror
(dashboard.data_loader.load_daily_movers_payload) — it never opens
data/db/signals.db, which is gitignored and doesn't exist on a deployed
instance anyway. See stock_scanner/pipeline/daily_movers.py and
scripts/build_daily_movers.py for how the mirror is produced.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_daily_movers_payload


@st.cache_data(ttl=300)
def _load_df() -> pd.DataFrame:
    payload = load_daily_movers_payload()
    rows = payload.get("rows", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values(["trade_date", "pct_change_close"], ascending=[False, False]).reset_index(drop=True)


def render_daily_movers_tab() -> None:
    """🚀 Daily Movers >10% tab — read-only. See module docstring."""
    st.markdown("### 🚀 Daily Movers >10%")
    st.caption(
        "Read-only view of sessions where a ticker moved >=10% against its previous close. "
        "**Intraday** catches names that spiked >10% even if they closed back below that "
        "level (`high / prev_close - 1 >= 0.10`); **Close** catches names that actually "
        "finished the session above 10% (`close / prev_close - 1 >= 0.10`). This table has "
        "no effect on production scoring, ranking, or model promotion — it is not read by "
        "signal_engine.py or the live morning scan."
    )

    df = _load_df()
    if df.empty:
        st.info(
            "📭 No daily movers recorded yet. `scripts/build_daily_movers.py` runs on its own "
            "daily schedule — see `.github/workflows/daily_movers.yml`."
        )
        return

    # ── Filters ──────────────────────────────────────────────────────────
    filt_col1, filt_col2 = st.columns([2, 1])
    with filt_col1:
        available_dates = sorted(df["trade_date"].dt.date.unique(), reverse=True)
        selected_date = st.selectbox(
            "Trade date", options=["All dates"] + [str(d) for d in available_dates],
            key="dm_date_filter",
        )
    with filt_col2:
        definition = st.radio(
            "Definition", options=["Either", "Intraday", "Close"],
            key="dm_definition_filter", horizontal=True,
        )

    filtered = df
    if selected_date != "All dates":
        filtered = filtered[filtered["trade_date"].dt.date.astype(str) == selected_date]
    if definition == "Intraday":
        filtered = filtered[filtered["hit_10pct_intraday"]]
    elif definition == "Close":
        filtered = filtered[filtered["hit_10pct_close"]]

    # ── Summary ──────────────────────────────────────────────────────────
    summary_cols = st.columns(4)
    summary_cols[0].metric("Rows shown", len(filtered))
    summary_cols[1].metric("Distinct tickers", filtered["ticker"].nunique())
    summary_cols[2].metric("Hit close >=10%", int(filtered["hit_10pct_close"].sum()))
    summary_cols[3].metric("Hit intraday >=10%", int(filtered["hit_10pct_intraday"].sum()))

    st.markdown("")

    if filtered.empty:
        st.info("No rows match the selected filters.")
        return

    # ── Table ────────────────────────────────────────────────────────────
    display = filtered.copy()
    display["trade_date"] = display["trade_date"].dt.strftime("%Y-%m-%d")
    for pct_col in ["pct_change_close", "pct_change_high"]:
        display[pct_col] = (display[pct_col] * 100).round(2)
    display_cols = [
        "trade_date", "ticker", "prev_close", "open", "high", "low", "close", "volume",
        "pct_change_close", "pct_change_high", "hit_10pct_close", "hit_10pct_intraday",
    ]
    display_cols = [c for c in display_cols if c in display.columns]
    st.dataframe(
        display[display_cols].rename(columns={
            "pct_change_close": "pct_change_close (%)", "pct_change_high": "pct_change_high (%)",
        }),
        use_container_width=True, hide_index=True,
    )
