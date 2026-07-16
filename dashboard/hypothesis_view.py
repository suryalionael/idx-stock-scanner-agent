"""Read-only dashboard view for AI Lab's Hypothesis Generator +
Statistical Validation.

STRICTLY READ-ONLY, same contract as reflection_view.py: no writes, no
refresh-from-dashboard fetch path, no live 9router calls, no path into
signal_engine.py / scanner_config.yaml / knowledge_base. Only reads the
committed data/published/hypotheses_report.json mirror via
dashboard.data_loader — never opens data/db/signals.db and never imports
stock_scanner.ai_lab.hypothesis_engine / statistical_validation / client /
agents.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_hypotheses_report_payload

_CONDITION_SEP = " AND "


@st.cache_data(ttl=300)
def _load_hypotheses_payload() -> dict:
    return load_hypotheses_report_payload()


def _conditions_str(conditions: list) -> str:
    return _CONDITION_SEP.join(f"{dim}={value}" for dim, value in conditions)


def _hypotheses_df(payload: dict) -> pd.DataFrame:
    rows = payload.get("hypotheses", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["conditions_str"] = df["conditions"].apply(_conditions_str)
    df["created_at"] = pd.to_datetime(df["created_at"])
    meta = pd.json_normalize(df["metadata_json"]).add_prefix("meta_")
    df = pd.concat([df.drop(columns=["metadata_json"]), meta], axis=1)
    return df


def render_hypothesis_section() -> None:
    """7. Hypothesis Generator — read-only. See module docstring."""
    st.markdown("#### 7. Hypothesis Generator")
    st.caption(
        "Multi-condition (order 2-3) hypotheses, seeded from Reflection Engine's own gated "
        "findings and validated with Wilson confidence intervals, Fisher's exact test, shrunk "
        "win rate, and Benjamini-Hochberg correction — never LLM-invented numbers. A candidate "
        "pool for a possible future Knowledge Base promotion, always via human review, never "
        "automatic. Read-only research findings; nothing here changes Production Scanner logic, "
        "thresholds, or the Knowledge Base. Generated manually via "
        "`scripts/run_hypothesis_engine.py` — see docs/AI_LAB_ARCHITECTURE.md."
    )

    payload = _load_hypotheses_payload()
    df = _hypotheses_df(payload)
    summary = payload.get("summary") or {}
    narrative = payload.get("narrative")

    if df.empty:
        st.info(
            "📭 No hypotheses yet — either `scripts/run_hypothesis_engine.py` hasn't been run, "
            "there are no gated Reflection Engine observations to seed candidates from, or no "
            "candidate cleared the statistical significance gate. An empty report is the "
            "correct, honest result when upstream evidence is still sparse."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Total Hypotheses", summary.get("total_hypotheses", len(df)))
    by_status = summary.get("by_status") or {}
    cols[1].metric("Validated", by_status.get("validated", 0))
    cols[2].metric("Rejected", by_status.get("rejected", 0))
    cols[3].metric("Resolved Trades Analyzed", summary.get("resolved_trade_count", "—"))

    if narrative and narrative.get("overall_summary"):
        st.markdown("**🧭 Overall Summary** _(LLM narration over the code-computed hypotheses below — never a source of new numbers)_")
        st.markdown(f"> {narrative['overall_summary']}")
        clusters = narrative.get("clusters") or []
        if clusters:
            st.markdown("**Clustered findings** _(narrative grouping only, not a statistical claim)_")
            for c in clusters:
                st.caption(f"**{c.get('label', '—')}**: {', '.join(c.get('hypothesis_ids', []))}")

    st.divider()
    _render_candidate_hypotheses(df)
    st.divider()
    _render_validated_hypotheses(df)
    st.divider()
    _render_rejected_hypotheses(df)
    st.divider()
    _render_statistical_evidence(df)
    st.divider()
    _render_strongest_predictors(df)
    st.divider()
    _render_emerging_patterns(df)
    st.divider()
    _render_timeline(df)


def _base_table(sub: pd.DataFrame, extra_cols: dict) -> None:
    if sub.empty:
        st.caption("None found.")
        return
    display_cols = {"conditions_str": "Conditions", "win_rate": "Win Rate", "sample_size": "Trades", **extra_cols}
    display = sub[list(display_cols)].copy()
    if "win_rate" in display.columns:
        display["win_rate"] = (display["win_rate"] * 100).round(1).astype(str) + "%"
    if "bh_adjusted_p" in display.columns:
        display["bh_adjusted_p"] = display["bh_adjusted_p"].round(4)
    st.dataframe(display.rename(columns=display_cols), use_container_width=True, hide_index=True)


def _render_candidate_hypotheses(df: pd.DataFrame) -> None:
    st.markdown("##### Candidate Hypotheses")
    st.caption(f"Every candidate hypothesis that entered statistical validation this run ({len(df)} total).")
    _base_table(df.sort_values("bh_adjusted_p"), {"status": "Status", "evidence_strength": "Evidence"})


def _render_validated_hypotheses(df: pd.DataFrame) -> None:
    st.markdown("##### Validated Hypotheses")
    sub = df[df["status"] == "validated"].sort_values("bh_adjusted_p")
    _base_table(sub, {"evidence_strength": "Evidence", "bh_adjusted_p": "BH-adjusted p"})


def _render_rejected_hypotheses(df: pd.DataFrame) -> None:
    st.markdown("##### Rejected Hypotheses")
    sub = df[df["status"] == "rejected"].sort_values("bh_adjusted_p")
    if sub.empty:
        st.caption("None found.")
        return
    display = sub[["conditions_str", "win_rate", "sample_size", "failed_gate", "rejection_reason"]].copy()
    display["win_rate"] = (display["win_rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(
        display.rename(columns={
            "conditions_str": "Conditions", "win_rate": "Win Rate", "sample_size": "Trades",
            "failed_gate": "Failed Gate", "rejection_reason": "Reason",
        }),
        use_container_width=True, hide_index=True,
    )


def _render_statistical_evidence(df: pd.DataFrame) -> None:
    st.markdown("##### Statistical Evidence")
    options = df["hypothesis_id"].tolist()
    labels = {row["hypothesis_id"]: f"{row['conditions_str']} ({row['status']})" for _, row in df.iterrows()}
    selected = st.selectbox(
        "Select a hypothesis to inspect", options=options, format_func=lambda i: labels.get(i, i),
        key="hyp_evidence_select",
    )
    if not selected:
        return
    row = df[df["hypothesis_id"] == selected].iloc[0]
    st.markdown(f"**{row['conditions_str']}**")
    st.caption(row["description"])
    cols = st.columns(5)
    cols[0].metric("Win Rate", f"{row['win_rate']:.1%}")
    cols[1].metric("Shrunk Win Rate", f"{row['shrunk_win_rate']:.1%}")
    cols[2].metric("95% Wilson CI", f"{row['wilson_lower']:.1%}–{row['wilson_upper']:.1%}")
    cols[3].metric("Fisher's exact p", f"{row['fisher_p']:.4f}")
    cols[4].metric("BH-adjusted p", f"{row['bh_adjusted_p']:.4f}")
    cols2 = st.columns(4)
    cols2[0].metric("Sample Size", int(row["sample_size"]))
    cols2[1].metric("Successes", int(row["successes"]))
    cols2[2].metric("Failures", int(row["failures"]))
    cols2[3].metric("Evidence Strength", row.get("evidence_strength") or "—")
    if row.get("llm_note"):
        st.caption(f"🧭 {row['llm_note']}")
    source_ids = row.get("source_reflection_ids") or []
    if source_ids:
        st.caption(f"Seeded from reflection observation(s): {', '.join(source_ids)}")


def _render_strongest_predictors(df: pd.DataFrame) -> None:
    st.markdown("##### Strongest Predictors")
    sub = df[(df["status"] == "validated") & (df["evidence_strength"] == "STRONG")].sort_values("bh_adjusted_p")
    _base_table(sub, {"bh_adjusted_p": "BH-adjusted p"})


def _render_emerging_patterns(df: pd.DataFrame) -> None:
    st.markdown("##### Emerging Patterns")
    st.caption("Most recently validated hypotheses.")
    sub = df[df["status"] == "validated"].sort_values("created_at", ascending=False).head(10)
    _base_table(sub, {"evidence_strength": "Evidence"})


def _render_timeline(df: pd.DataFrame) -> None:
    st.markdown("##### Timeline")
    recent = df.sort_values("created_at", ascending=False).head(20)
    for _, h in recent.iterrows():
        icon = "✅" if h["status"] == "validated" else "❌"
        note = f" — _{h['llm_note']}_" if pd.notna(h.get("llm_note")) and h.get("llm_note") else ""
        st.markdown(f"{icon} **{h['created_at']:%Y-%m-%d %H:%M UTC}** {h['conditions_str']}{note}")
        st.caption(h["description"])
