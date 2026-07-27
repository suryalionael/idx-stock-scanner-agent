"""Read-only dashboard view for AI Lab's Reflection Engine.

STRICTLY READ-ONLY, same contract as ai_lab_view.py: no writes, no
refresh-from-dashboard fetch path, no live 9router calls, no path into
signal_engine.py / scanner_config.yaml / knowledge_base. Only reads the
committed data/published/reflection_report.json mirror via
dashboard.data_loader — never opens data/db/signals.db and never imports
stock_scanner.ai_lab.reflection_engine / client / agents.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_reflection_report_payload

_CALIBRATION_CATEGORY = "confidence_calibration"


@st.cache_data(ttl=300)
def _load_reflection_payload() -> dict:
    return load_reflection_report_payload()


def _observations_df(payload: dict) -> pd.DataFrame:
    observations = payload.get("observations", [])
    if not observations:
        return pd.DataFrame()
    df = pd.DataFrame(observations)
    stats_df = pd.json_normalize(df["supporting_statistics"])
    df = pd.concat([df.drop(columns=["supporting_statistics"]), stats_df], axis=1)
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    return df


def render_reflection_section() -> None:
    """6. Reflection — read-only. See module docstring."""
    st.markdown("#### 6. Reflection")
    st.caption(
        "Statistically gated observations over RESOLVED AI Lab recommendations only — "
        "Wilson confidence intervals, Fisher's exact test, Benjamini-Hochberg correction, or a "
        "one-sample binomial calibration test, never LLM-invented numbers. Read-only research "
        "findings; nothing here changes Production Scanner logic, thresholds, or the Knowledge "
        "Base."
    )

    payload = _load_reflection_payload()
    df = _observations_df(payload)
    summary = payload.get("summary") or {}
    narrative = payload.get("narrative")

    if df.empty:
        st.info(
            "📭 Waiting for enough resolved recommendations. An empty report is the correct, "
            "honest result when there aren't yet enough resolved trades to clear the "
            "statistical significance gate — not an error."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Total Observations", summary.get("total_observations", len(df)))
    cols[1].metric("Resolved Trades Analyzed", summary.get("resolved_trade_count", "—"))
    cols[2].metric("Categories", len(summary.get("by_category") or {}) or df["category"].nunique())
    cols[3].metric("Last Run", (payload.get("generated_at") or "—")[:10])

    if narrative and narrative.get("overall_summary"):
        st.markdown("**🧭 Overall Summary** _(LLM narration over the code-computed observations below — never a source of new numbers)_")
        st.markdown(f"> {narrative['overall_summary']}")

    st.divider()
    _render_success_failure_patterns(df)
    st.divider()
    _render_model_comparison(df)
    st.divider()
    _render_sector_comparison(df)
    st.divider()
    _render_confidence_distribution(df)
    st.divider()
    _render_recommendation_distribution(df)
    st.divider()
    _render_recent_reflections(df)


def _render_pattern_table(sub: pd.DataFrame) -> None:
    if sub.empty:
        st.caption("None found.")
        return
    display = sub[["title", "win_rate", "baseline_rate", "n", "confidence"]].copy()
    display["win_rate"] = (display["win_rate"] * 100).round(1).astype(str) + "%"
    display["baseline_rate"] = (display["baseline_rate"] * 100).round(1).astype(str) + "%"
    display["confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(
        display.rename(columns={
            "title": "Observation", "win_rate": "Win Rate", "baseline_rate": "Baseline",
            "n": "Trades", "confidence": "Confidence",
        }),
        use_container_width=True, hide_index=True,
    )


def _render_success_failure_patterns(df: pd.DataFrame) -> None:
    st.markdown("##### Top Success Patterns / Top Failure Patterns")
    comparable = df[df["category"] != _CALIBRATION_CATEGORY].copy()
    if comparable.empty:
        st.caption("No win-rate-vs-baseline observations yet (only confidence calibration findings so far).")
        return
    comparable["is_success"] = comparable["win_rate"] > comparable["baseline_rate"]

    col_success, col_failure = st.columns(2)
    with col_success:
        st.markdown("**🟢 Top Success Patterns**")
        top = comparable[comparable["is_success"]].sort_values("confidence", ascending=False).head(10)
        _render_pattern_table(top)
    with col_failure:
        st.markdown("**🔴 Top Failure Patterns**")
        bottom = comparable[~comparable["is_success"]].sort_values("confidence", ascending=False).head(10)
        _render_pattern_table(bottom)


def _render_model_comparison(df: pd.DataFrame) -> None:
    st.markdown("##### Model Comparison")
    sub = df[df["category"] == "model_performance"]
    if sub.empty:
        st.caption("No AI-model-level observations yet.")
        return
    _render_pattern_table(sub.sort_values("win_rate", ascending=False))


def _render_sector_comparison(df: pd.DataFrame) -> None:
    st.markdown("##### Sector Comparison")
    sub = df[df["category"] == "sector_performance"]
    if sub.empty:
        st.caption("No sector-level observations yet.")
        return
    _render_pattern_table(sub.sort_values("win_rate", ascending=False))


def _render_confidence_distribution(df: pd.DataFrame) -> None:
    st.markdown("##### Confidence Distribution")
    sub = df[df["category"] == _CALIBRATION_CATEGORY]
    if sub.empty:
        st.caption("No confidence calibration issues detected yet.")
        return
    chart_df = sub[["value", "avg_confidence", "win_rate"]].set_index("value")
    chart_df.columns = ["Avg Stated Confidence", "Realized Win Rate"]
    st.bar_chart(chart_df)

    display = sub[["value", "calibration_issue", "avg_confidence", "win_rate", "n", "confidence"]].copy()
    display["avg_confidence"] = (display["avg_confidence"] * 100).round(1).astype(str) + "%"
    display["win_rate"] = (display["win_rate"] * 100).round(1).astype(str) + "%"
    display["confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(
        display.rename(columns={
            "value": "Confidence Bucket", "calibration_issue": "Issue", "avg_confidence": "Stated Confidence",
            "win_rate": "Realized Win Rate", "n": "Trades", "confidence": "Statistical Confidence",
        }),
        use_container_width=True, hide_index=True,
    )


def _render_recommendation_distribution(df: pd.DataFrame) -> None:
    st.markdown("##### Recommendation Distribution")
    sub = df[df["category"] == "recommendation_level_performance"]
    if sub.empty:
        st.caption("No recommendation-level observations yet.")
        return
    chart_df = sub[["value", "win_rate", "baseline_rate"]].set_index("value")
    chart_df.columns = ["Win Rate", "Baseline"]
    st.bar_chart(chart_df)


def _render_recent_reflections(df: pd.DataFrame) -> None:
    st.markdown("##### Recent Reflections")
    recent = df.sort_values("generated_at", ascending=False).head(20)
    for _, obs in recent.iterrows():
        note = f" — _{obs['llm_note']}_" if pd.notna(obs.get("llm_note")) and obs.get("llm_note") else ""
        st.markdown(f"**{obs['generated_at']:%Y-%m-%d %H:%M UTC}** [{obs['category']}] {obs['title']}{note}")
        st.caption(obs["description"])
