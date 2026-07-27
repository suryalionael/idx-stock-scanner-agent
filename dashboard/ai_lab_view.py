"""Read-only dashboard view for AI Lab — experimental, standalone AI
recommendation engine.

STRICTLY READ-ONLY, matching the pattern established by
knowledge_base_view.py / daily_movers_view.py: no writes, no
refresh-from-dashboard fetch path, no live 9router calls from here, no
path into signal_engine.py / scanner_config.yaml / model promotion. Only
reads the committed data/published/ai_recommendations.json and
data/published/ai_learning_events.json mirrors via dashboard.data_loader —
never opens data/db/signals.db (gitignored, absent on a deployed instance)
and never imports stock_scanner.ai_lab.client / agents. Section 6
(Reflection) delegates to dashboard/reflection_view.py, section 7
(Hypothesis Generator) delegates to dashboard/hypothesis_view.py, and
section 8 (Knowledge Base) delegates to dashboard/knowledge_entries_view.py
— same isolation contract, each its own committed
data/published/{reflection,hypotheses,knowledge}_report.json mirror.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data_loader import (
    load_ai_learning_events_payload,
    load_ai_pipeline_status_payload,
    load_ai_recommendations_payload,
)
from dashboard.hypothesis_view import render_hypothesis_section
from dashboard.knowledge_entries_view import render_knowledge_entries_section
from dashboard.reflection_view import render_reflection_section
from stock_scanner.ai_lab.performance import compute_performance

_EVENT_ICONS = {
    "pattern_learned": "🧩",
    "confidence_updated": "📈",
    "hypothesis_generated": "💡",
    "accuracy_improved": "🎯",
    "outcome_resolved": "📊",
    "reflection_generated": "🪞",
    "hypothesis_validated": "🔬",
    "knowledge_base_updated": "📚",
}


@st.cache_data(ttl=300)
def _load_recommendations_payload() -> dict:
    return load_ai_recommendations_payload()


@st.cache_data(ttl=300)
def _load_recommendations_df() -> pd.DataFrame:
    # Reads the cached payload rather than calling load_ai_recommendations_payload()
    # again — render_ai_lab_tab() also needs the raw payload (for
    # generated_at/summary), and this is what keeps both call sites to a
    # single cached fetch instead of two independent reads of the same file.
    payload = _load_recommendations_payload()
    rows = payload.get("rows", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=300)
def _load_events_df() -> pd.DataFrame:
    payload = load_ai_learning_events_payload()
    events = payload.get("events", [])
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=300)
def _load_pipeline_status_payload() -> dict:
    return load_ai_pipeline_status_payload()


def _render_pipeline_status_row() -> None:
    """Overall AI Automation Pipeline health — reads
    data/published/ai_pipeline_status.json via the one unified loader
    (dashboard.data_loader.load_ai_pipeline_status_payload); never computed
    here, purely a read of already-published execution metadata."""
    status_payload = _load_pipeline_status_payload()
    if not status_payload:
        st.caption("⏳ AI Automation Pipeline has not run yet.")
        return

    last_run = status_payload.get("last_run", "—")
    stage_names = ["generation", "resolution", "reflection", "hypothesis", "knowledge_base"]
    icon_map = {"ok": "🟢", "skipped": "🟡", "failed": "🔴"}
    badges = []
    for name in stage_names:
        stage = status_payload.get(name) or {}
        icon = icon_map.get(stage.get("status"), "⚪")
        badges.append(f"{icon} {name}")
    st.caption(f"⚙️ AI Automation Pipeline — last run {last_run} — " + " · ".join(badges))


def render_ai_lab_tab() -> None:
    """🧪 AI Lab tab — read-only. See module docstring."""
    st.markdown("### 🧪 AI Lab — Experimental AI Recommendation Engine")
    st.caption(
        "**Completely separate from the Production Scanner.** AI Lab is a standalone, "
        "experimental recommendation engine that proposes stock ideas and tracks its own "
        "forward-tested performance. It does not affect production scoring, ranking, signal "
        "classification, or model promotion. Only after demonstrating superior long-term "
        "performance would future integration into production even be considered — see "
        "docs/AI_LAB_ARCHITECTURE.md."
    )
    _render_pipeline_status_row()

    payload = _load_recommendations_payload()
    df = _load_recommendations_df()
    if df.empty:
        st.info("📭 No recommendations have been generated yet.")
    else:
        by_status = (payload.get("summary") or {}).get("by_status", {})
        status_cols = st.columns(4)
        status_cols[0].metric("Pending", by_status.get("PENDING", 0))
        status_cols[1].metric("Active", by_status.get("ACTIVE", 0))
        status_cols[2].metric("Resolved", by_status.get("CLOSED", 0))
        status_cols[3].metric("Expired", by_status.get("EXPIRED", 0))
        st.caption(f"Last generated: {payload.get('generated_at', '—')}")
        available_models = sorted(df["ai_model"].unique())
        _render_recommendations_section(df, available_models)
        st.divider()
        _render_performance_section(df, available_models)
        st.divider()
        _render_history_section(df)
        st.divider()
        _render_learning_timeline_section()

    st.divider()
    render_reflection_section()
    st.divider()
    render_hypothesis_section()
    st.divider()
    render_knowledge_entries_section()


# ---------------------------------------------------------------------------
# 1. AI Recommendations (+ 2. Recommendation Details side panel)
# ---------------------------------------------------------------------------

def _render_recommendations_section(df: pd.DataFrame, available_models: list[str]) -> None:
    st.markdown("#### 1. AI Recommendations")

    filt_col1, filt_col2, filt_col3 = st.columns([2, 2, 2])
    with filt_col1:
        model_filter = st.selectbox("AI Model", options=["All models"] + available_models, key="ailab_model_filter")
    with filt_col2:
        sort_choice = st.selectbox(
            "Sort by", options=["Highest AI Score", "Highest Confidence", "Newest Recommendation"],
            key="ailab_sort",
        )
    with filt_col3:
        status_filter = st.multiselect(
            "Status", options=sorted(df["status"].unique()), default=sorted(df["status"].unique()),
            key="ailab_status_filter",
        )

    filtered = df.copy()
    if model_filter != "All models":
        filtered = filtered[filtered["ai_model"] == model_filter]
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]
    else:
        filtered = filtered.iloc[0:0]

    sort_map = {
        "Highest AI Score": ("score", False),
        "Highest Confidence": ("confidence", False),
        "Newest Recommendation": ("created_at", False),
    }
    sort_col, ascending = sort_map[sort_choice]
    filtered = filtered.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    filtered.insert(0, "rank", range(1, len(filtered) + 1))

    st.caption(f"{len(filtered)} of {len(df)} recommendation(s) shown.")
    if filtered.empty:
        st.info("No recommendations match the selected filters.")
        return

    display = filtered.copy()
    display["confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
    display["expected_return"] = display["expected_return"].apply(
        lambda v: f"{v * 100:.2f}%" if v is not None and pd.notna(v) else "N/A (insufficient data)"
    )
    display["generated_time"] = display["created_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    display_cols = {
        "rank": "Rank", "ticker": "Ticker", "score": "AI Score", "confidence": "Confidence",
        "recommendation": "Recommendation", "expected_return": "Expected Return",
        "risk_level": "Risk Level", "generated_time": "Generated Time", "status": "Status",
    }
    st.dataframe(display[list(display_cols)].rename(columns=display_cols), use_container_width=True, hide_index=True)

    st.markdown("##### 2. Recommendation Details")
    options = filtered["id"].tolist()
    labels = {row["id"]: f"#{row['rank']} {row['ticker']} ({row['ai_model']})" for _, row in filtered.iterrows()}
    selected_id = st.selectbox(
        "Select a row to inspect", options=options, format_func=lambda i: labels.get(i, i), key="ailab_detail_select",
    )
    if selected_id:
        _render_detail(filtered[filtered["id"] == selected_id].iloc[0])


def _render_detail(row: pd.Series) -> None:
    reasoning = row.get("reasoning") or {}
    col_why, col_stats = st.columns(2)

    with col_why:
        st.markdown(f"**🤖 Why {row['ticker']} was selected ({row['ai_model']})**")
        st.markdown(f"> {reasoning.get('why', '—')}")
        st.caption(reasoning.get("reasoning_summary", "—"))

        strengths = reasoning.get("strengths") or []
        weaknesses = reasoning.get("weaknesses") or []
        risks = reasoning.get("risks") or []
        if strengths:
            st.markdown("**Strengths:**\n" + "\n".join(f"- {s}" for s in strengths))
        if weaknesses:
            st.markdown("**Weaknesses:**\n" + "\n".join(f"- {w}" for w in weaknesses))
        if risks:
            st.markdown("**Risks:**\n" + "\n".join(f"- {r}" for r in risks))

    with col_stats:
        st.markdown("**📊 Code-computed evidence** — never LLM-generated")
        indicators = reasoning.get("technical_indicators") or {}
        if indicators:
            st.markdown("**Technical indicators:**")
            st.table(pd.DataFrame([indicators]).T.rename(columns={0: "value"}))
        else:
            st.caption("No technical indicators recorded for this recommendation.")

        stats = reasoning.get("statistical_evidence") or []
        patterns = reasoning.get("similar_patterns") or []
        if stats:
            st.markdown("**Statistical evidence (validated knowledge_base patterns, exact matches):**")
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
        else:
            st.caption("No exactly-matching validated pattern found for this ticker at generation time.")
        if patterns:
            st.caption("Similar historical patterns: " + "; ".join(patterns))
        similarity = reasoning.get("best_pattern_similarity_pct")
        if similarity is not None:
            st.caption(f"Closest known pattern similarity (partial match, any pattern): {similarity:.1f}%")

    st.divider()
    _render_decision_trace(row)
    st.divider()
    _render_historical_comparison(row)


def _render_decision_trace(row: pd.Series) -> None:
    """Section 1 of the explainability upgrade: the AI Score/confidence are
    never shown as a single unexplained number — always alongside the
    component scores they were computed from (see
    stock_scanner.ai_lab.scoring)."""
    trace = row.get("decision_trace") or {}
    confidence = row.get("confidence_breakdown") or {}

    st.markdown("**🔍 Decision Trace** — how the AI Score was derived")
    if trace:
        cols = st.columns(5)
        cols[0].metric("Technical", f"{trace.get('technical_score', 0):.0f}")
        cols[1].metric("Statistical", f"{trace.get('statistical_score', 0):.0f}")
        cols[2].metric("Pattern Similarity", f"{trace.get('pattern_similarity_score', 0):.0f}")
        cols[3].metric("Risk", f"{trace.get('risk_score', 0):.0f}")
        cols[4].metric("Final Score", f"{trace.get('final_score', 0):.0f}")
    else:
        st.caption("No decision trace recorded for this recommendation.")

    st.markdown("**Confidence Breakdown**")
    if confidence:
        cols = st.columns(5)
        cols[0].metric("Technical", f"{confidence.get('technical', 0):.0%}")
        cols[1].metric("Statistical", f"{confidence.get('statistical', 0):.0%}")
        cols[2].metric("Pattern Similarity", f"{confidence.get('pattern_similarity', 0):.0%}")
        cols[3].metric("Risk Adjustment", f"{confidence.get('risk_adjustment', 0):+.0%}")
        cols[4].metric("Final Confidence", f"{confidence.get('final_confidence', 0):.0%}")
        st.caption((row.get("reasoning") or {}).get("confidence_explanation", "—"))
    else:
        st.caption("No confidence breakdown recorded for this recommendation.")


def _render_historical_comparison(row: pd.Series) -> None:
    """Section 7 of the explainability upgrade: compare the current setup
    against the closest validated historical pattern, code-computed stats
    + an LLM explanation constrained to those exact numbers."""
    comparison = row.get("historical_comparison") or {}
    st.markdown("**📐 Historical Pattern Comparison**")
    if not comparison or comparison.get("verdict") == "no_data" or comparison.get("sample_size") is None:
        st.caption("No validated historical pattern matches this ticker's current setup.")
        return

    cols = st.columns(4)
    cols[0].metric("Sample Size", comparison.get("sample_size"))
    win_rate = comparison.get("win_rate")
    cols[1].metric("Win Rate", f"{win_rate:.1%}" if win_rate is not None else "—")
    ci_lower, ci_upper = comparison.get("ci_lower"), comparison.get("ci_upper")
    cols[2].metric(
        "95% CI",
        f"{ci_lower:.1%}–{ci_upper:.1%}" if ci_lower is not None and ci_upper is not None else "—",
    )
    verdict_labels = {"stronger": "🟢 Stronger", "weaker": "🔴 Weaker", "similar": "🟡 Similar"}
    cols[3].metric("Verdict", verdict_labels.get(comparison.get("verdict"), "—"))
    if comparison.get("pattern_description"):
        st.caption(f"Matched pattern: {comparison['pattern_description']}")
    st.caption(comparison.get("explanation", "—"))


# ---------------------------------------------------------------------------
# 3. AI Performance
# ---------------------------------------------------------------------------

def _render_performance_section(df: pd.DataFrame, available_models: list[str]) -> None:
    st.markdown("#### 3. AI Performance")
    model_choice = st.selectbox(
        "Performance for", options=["All models"] + available_models, key="ailab_perf_model_filter",
    )
    metrics = compute_performance(df, ai_model=None if model_choice == "All models" else model_choice)

    def _pct(v: float | None) -> str:
        return f"{v:.1%}" if v is not None else "— (not enough data)"

    def _num(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "— (not enough data)"

    row1 = st.columns(5)
    row1[0].metric("Win Rate", _pct(metrics["win_rate"]))
    row1[1].metric("Average Return", _pct(metrics["average_return"]))
    row1[2].metric("Average Loss", _pct(metrics["average_loss"]))
    row1[3].metric("Profit Factor", _num(metrics["profit_factor"]))
    row1[4].metric("Expected Value", _pct(metrics["expected_value"]))

    row2 = st.columns(4)
    row2[0].metric("Sharpe Ratio", _num(metrics["sharpe_ratio"]))
    row2[1].metric("Number of Recommendations", metrics["number_of_recommendations"])
    row2[2].metric("Active Positions", metrics["active_positions"])
    row2[3].metric("Closed Positions", metrics["closed_positions"])
    st.caption("Metrics recompute automatically from every published recommendation batch — "
               "no manual refresh needed. Ratio metrics show \"not enough data\" (never 0 or NaN) "
               "until at least one recommendation has resolved (CLOSED or EXPIRED).")


# ---------------------------------------------------------------------------
# 4. Recommendation History
# ---------------------------------------------------------------------------

def _render_history_section(df: pd.DataFrame) -> None:
    st.markdown("#### 4. Recommendation History")
    st.caption(
        "Every recommendation ever generated — append-only, never overwritten. \"Current Price\" "
        "reflects the last price this read-only dashboard has on record (entry/exit price from the "
        "pipeline's own run), not a live quote — AI Lab never fetches live data from the dashboard."
    )
    history = df.sort_values("created_at", ascending=False).copy()
    history["current_price"] = history["exit_price"].where(history["exit_price"].notna(), history["entry_price"])
    history["confidence"] = (history["confidence"] * 100).round(1).astype(str) + "%"
    history["date"] = history["created_at"].dt.strftime("%Y-%m-%d")
    display_cols = {
        "date": "Date", "ticker": "Ticker", "entry_price": "Entry Price", "current_price": "Current Price",
        "return_percentage": "Return %", "status": "Status", "confidence": "Confidence", "ai_model": "AI Model",
    }
    st.dataframe(history[list(display_cols)].rename(columns=display_cols), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 5. AI Learning Timeline
# ---------------------------------------------------------------------------

def _render_learning_timeline_section() -> None:
    st.markdown("#### 5. AI Learning Timeline")
    events_df = _load_events_df()
    if events_df.empty:
        st.caption("No learning events logged yet.")
        return

    for _, ev in events_df.sort_values("created_at", ascending=False).head(20).iterrows():
        icon = _EVENT_ICONS.get(ev["event_type"], "•")
        model_tag = f" [{ev['ai_model']}]" if ev.get("ai_model") else ""
        st.markdown(f"{icon} **{ev['created_at']:%Y-%m-%d %H:%M UTC}**{model_tag} — {ev['description']}")
