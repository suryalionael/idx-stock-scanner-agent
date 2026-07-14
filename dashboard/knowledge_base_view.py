"""Read-only dashboard view over the Learning Agent's knowledge_base.

STRICTLY READ-ONLY: no status mutation, no edit buttons, no training hooks,
no path back into signal_engine.py / scanner_config.yaml / model_registry.
Reviewing and changing a hypothesis's status is a separate, deliberate,
terminal-only action — see scripts/review_knowledge_base.py and
docs/KNOWLEDGE_BASE_POLICY.md. This module only ever reads the committed
data/published/knowledge_base.json mirror (dashboard.data_loader.
load_knowledge_base_payload) — it never opens data/db/signals.db, which is
gitignored and doesn't exist on a deployed instance anyway.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_knowledge_base_payload

_STATUS_ORDER = ["candidate", "reviewed", "testing", "tested_passed", "tested_failed", "promoted", "archived"]


@st.cache_data(ttl=300)
def _load_df() -> pd.DataFrame:
    payload = load_knowledge_base_payload()
    rows = payload.get("knowledge_base", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "generated_at" in df.columns:
        df = df.sort_values("generated_at", ascending=False)
    return df.reset_index(drop=True)


def _extract_pattern(row: pd.Series) -> dict:
    try:
        return json.loads(row.get("pattern_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def render_knowledge_base_tab() -> None:
    """🧠 Learning Agent tab — read-only. See module docstring."""
    st.markdown("### 🧠 Learning Agent — Knowledge Base")
    st.caption(
        "Read-only research view. Every hypothesis here is an LLM-articulated description "
        "of a pattern that already passed a statistical gate (Fisher's exact test, "
        "Benjamini-Hochberg FDR correction, Wilson confidence interval). **This table has "
        "no effect on production scoring, ranking, or model promotion** — see "
        "[docs/KNOWLEDGE_BASE_POLICY.md](https://github.com/suryalionael/idx-stock-scanner-agent/"
        "blob/main/docs/KNOWLEDGE_BASE_POLICY.md) for the full policy. Status changes only "
        "happen via `scripts/review_knowledge_base.py` on the command line — nothing on this "
        "page writes anything."
    )

    df = _load_df()
    if df.empty:
        st.info(
            "📭 No hypotheses yet. The Learning Agent is run manually, not on a schedule — "
            "see docs/LEARNING_AGENT_RUNBOOK.md."
        )
        return

    # ── Summary ──────────────────────────────────────────────────────────
    counts = df["status"].value_counts()
    summary_cols = st.columns(len(_STATUS_ORDER))
    for col, status in zip(summary_cols, _STATUS_ORDER):
        col.metric(status, int(counts.get(status, 0)))

    st.markdown("")

    # ── Filter ───────────────────────────────────────────────────────────
    available_statuses = [s for s in _STATUS_ORDER if s in counts.index]
    selected = st.multiselect(
        "Filter by status", options=available_statuses, default=available_statuses,
        key="kb_status_filter",
    )
    filtered = df[df["status"].isin(selected)] if selected else df.iloc[0:0]
    st.caption(f"{len(filtered)} of {len(df)} hypotheses shown.")

    if filtered.empty:
        st.info("No hypotheses match the selected status filter.")
        return

    # ── List ─────────────────────────────────────────────────────────────
    display_cols = ["hypothesis_id", "status", "confidence", "supporting_trades",
                    "affected_sector", "hypothesis", "linked_model_version_id"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    # ── Detail inspector ─────────────────────────────────────────────────
    st.markdown("#### Inspect one hypothesis")
    options = filtered["hypothesis_id"].tolist()
    selected_id = st.selectbox("Hypothesis ID", options=options, key="kb_detail_select")
    if not selected_id:
        return

    row = filtered[filtered["hypothesis_id"] == selected_id].iloc[0]
    _render_detail(row)


def _render_detail(row: pd.Series) -> None:
    col_llm, col_stats = st.columns(2)

    with col_llm:
        st.markdown("**🤖 LLM-authored** — qualitative, not statistical proof")
        st.markdown(f"> {row.get('hypothesis', '')}")
        st.metric("Confidence (LLM-assigned)", f"{float(row.get('confidence') or 0):.2f}")
        st.caption(f"Expected effect: {row.get('expected_effect') or '—'}")
        st.caption(f"Affected sector: {row.get('affected_sector') or '— (market-wide)'}")

    with col_stats:
        st.markdown("**📊 Code-derived statistics** — from Phase 1 pattern mining")
        pattern = _extract_pattern(row)
        rep = pattern.get("representative", {})
        if not rep:
            st.caption("No backing pattern data available for this hypothesis.")
        else:
            st.metric("Sample size (n / n_success)", f"{rep.get('n', '?')} / {rep.get('n_success', '?')}")
            win_rate = rep.get("win_rate")
            shrunk = rep.get("win_rate_shrunk")
            st.metric(
                "Win rate (raw / shrunk)",
                f"{win_rate:.2%} / {shrunk:.2%}" if win_rate is not None and shrunk is not None else "—",
            )
            ci_lower = rep.get("ci_lower")
            st.metric("95% CI lower bound", f"{ci_lower:.2%}" if ci_lower is not None else "—")
            p_adj = rep.get("p_value_adjusted")
            st.metric("FDR-adjusted p-value", f"{p_adj:.4f}" if p_adj is not None else "—")
            flag = rep.get("ticker_concentration_flag")
            st.metric("Ticker concentration flag", "⚠️ Yes" if flag else "✅ No")
            stable = rep.get("time_split_stable")
            st.metric(
                "Time-split stable",
                "✅ Yes" if stable is True else ("❌ No" if stable is False else "⚠️ Inconclusive"),
            )
            st.caption(
                f"This is 1 representative of {pattern.get('member_count', 1)} near-duplicate "
                f"statistical finding(s) merged into this cluster."
            )

    st.markdown("---")
    meta = st.columns(4)
    meta[0].caption(f"**Status:** {row.get('status')}")
    meta[1].caption(f"**Generated:** {row.get('generated_at', '?')}")
    meta[2].caption(f"**Source run:** {row.get('source_run_id', '?')}")
    linked = row.get("linked_model_version_id")
    meta[3].caption(f"**Linked model:** {linked}" if linked else "**Linked model:** none yet")
    if row.get("reviewed_by"):
        st.caption(f"Reviewed by: {row.get('reviewed_by')}")
