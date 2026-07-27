"""Read-only dashboard view for AI Lab's Knowledge Base Engine.

STRICTLY READ-ONLY, same contract as reflection_view.py/hypothesis_view.py:
no writes, no refresh-from-dashboard fetch path, no live 9router calls, no
path into signal_engine.py / scanner_config.yaml / the production
knowledge_base table. Only reads the committed
data/published/knowledge_report.json mirror via dashboard.data_loader —
never opens data/db/signals.db and never imports
stock_scanner.ai_lab.knowledge_base_engine / client / agents.

Named knowledge_entries_view.py, not knowledge_base_view.py — that path
is already taken by the unrelated production knowledge_base table's view.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_knowledge_report_payload

_ACTIVE_STATUSES = {"confirmed", "strong"}


@st.cache_data(ttl=300)
def _load_knowledge_payload() -> dict:
    return load_knowledge_report_payload()


def _conditions_str(conditions: list) -> str:
    return " AND ".join(f"{dim}={value}" for dim, value in conditions)


def _entries_df(payload: dict) -> pd.DataFrame:
    rows = payload.get("entries", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["conditions_str"] = df["conditions"].apply(_conditions_str)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df["last_confirmed"] = pd.to_datetime(df["last_confirmed"])
    ci = pd.DataFrame(df["confidence_interval"].tolist(), columns=["wilson_lower", "wilson_upper"], index=df.index)
    df = pd.concat([df, ci], axis=1)
    df["status_changed"] = df["previous_lifecycle_status"].notna() & (
        df["previous_lifecycle_status"] != df["lifecycle_status"]
    )
    return df


def render_knowledge_entries_section() -> None:
    """8. Knowledge Base — read-only. See module docstring."""
    st.markdown("#### 8. Knowledge Base")
    st.caption(
        "Long-lived institutional knowledge curated from validated_hypotheses across ALL "
        "historical runs — grouped by exact normalized condition-set (deterministic "
        "similarity, never LLM clustering), with a deterministic lifecycle (Emerging → "
        "Confirmed → Strong, or → Weakening → Contradicted → Archived). Not another "
        "statistical engine — no new significance testing here, only curation of what "
        "Statistical Validation already decided. Read-only research findings; nothing here "
        "changes Production Scanner logic, thresholds, or the production Knowledge Base."
    )

    payload = _load_knowledge_payload()
    df = _entries_df(payload)
    summary = payload.get("summary") or {}
    narrative = payload.get("narrative")

    if df.empty:
        st.info(
            "📭 Waiting for enough resolved recommendations. An empty report is the correct, "
            "honest result when no condition-set has been validated by Statistical Validation "
            "yet — not an error."
        )
        return

    by_status = summary.get("by_lifecycle_status") or {}
    cols = st.columns(6)
    cols[0].metric("Total Knowledge Entries", summary.get("total_entries", len(df)))
    cols[1].metric("Emerging", by_status.get("emerging", 0))
    cols[2].metric("Confirmed", by_status.get("confirmed", 0))
    cols[3].metric("Strong", by_status.get("strong", 0))
    cols[4].metric("Archived", by_status.get("archived", 0))
    cols[5].metric("Last Run", (payload.get("generated_at") or "—")[:10])

    if narrative and narrative.get("overall_summary"):
        st.markdown("**🧭 Overall Summary** _(LLM narration over the code-computed knowledge entries below — never a source of new numbers, lifecycle changes, or confidence)_")
        st.markdown(f"> {narrative['overall_summary']}")
        groups = narrative.get("organized_groups") or []
        if groups:
            st.markdown("**Organized groups** _(narrative grouping only, not a merge)_")
            for g in groups:
                st.caption(f"**{g.get('label', '—')}**: {', '.join(g.get('knowledge_ids', []))}")

    st.divider()
    _render_active_knowledge(df)
    st.divider()
    _render_emerging_knowledge(df)
    st.divider()
    _render_weakening_knowledge(df)
    st.divider()
    _render_archived_knowledge(df)
    st.divider()
    _render_strongest_patterns(df)
    st.divider()
    _render_recent_changes(df)
    st.divider()
    _render_knowledge_timeline(df)


def _entry_table(sub: pd.DataFrame, extra_cols: dict) -> None:
    if sub.empty:
        st.caption("None found.")
        return
    display_cols = {
        "conditions_str": "Conditions", "average_win_rate": "Avg Win Rate",
        "confirmation_count": "Confirmations", **extra_cols,
    }
    display = sub[list(display_cols)].copy()
    if "average_win_rate" in display.columns:
        display["average_win_rate"] = (display["average_win_rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(display.rename(columns=display_cols), use_container_width=True, hide_index=True)


def _render_active_knowledge(df: pd.DataFrame) -> None:
    st.markdown("##### Active Knowledge")
    st.caption("Confirmed or Strong — evidence has independently reconfirmed this belief at least once.")
    sub = df[df["lifecycle_status"].isin(_ACTIVE_STATUSES)].sort_values("confirmation_count", ascending=False)
    _entry_table(sub, {"lifecycle_status": "Status", "contradiction_count": "Contradictions"})


def _render_emerging_knowledge(df: pd.DataFrame) -> None:
    st.markdown("##### Emerging Knowledge")
    st.caption("Just discovered — validated once, not yet independently reconfirmed.")
    sub = df[df["lifecycle_status"] == "emerging"].sort_values("last_confirmed", ascending=False)
    _entry_table(sub, {"first_seen": "First Seen"})


def _render_weakening_knowledge(df: pd.DataFrame) -> None:
    st.markdown("##### Weakening Knowledge")
    st.caption("At least one contradicting validation recorded, but confirmations still outnumber it.")
    sub = df[df["lifecycle_status"] == "weakening"].sort_values("contradiction_count", ascending=False)
    _entry_table(sub, {"contradiction_count": "Contradictions"})


def _render_archived_knowledge(df: pd.DataFrame) -> None:
    st.markdown("##### Archived Knowledge")
    st.caption("Contradicted or archived — contradicting evidence has caught up to or overtaken confirming evidence.")
    sub = df[df["lifecycle_status"].isin({"contradicted", "archived"})].sort_values("contradiction_count", ascending=False)
    _entry_table(sub, {"lifecycle_status": "Status", "contradiction_count": "Contradictions"})


def _render_strongest_patterns(df: pd.DataFrame) -> None:
    st.markdown("##### Strongest Patterns")
    sub = df[df["lifecycle_status"] == "strong"].sort_values("confirmation_count", ascending=False)
    _entry_table(sub, {"shrunk_win_rate": "Shrunk Win Rate", "wilson_lower": "CI Lower", "wilson_upper": "CI Upper"})


def _render_recent_changes(df: pd.DataFrame) -> None:
    st.markdown("##### Recent Changes")
    st.caption("Entries whose lifecycle status differs from the previous curation run.")
    sub = df[df["status_changed"]].sort_values("created_at", ascending=False)
    if sub.empty:
        st.caption("No lifecycle status changes in the most recent curation run.")
        return
    for _, e in sub.iterrows():
        st.markdown(
            f"**{e['conditions_str']}**: {e['previous_lifecycle_status']} → **{e['lifecycle_status']}**"
        )
        note = e.get("llm_note")
        if pd.notna(note) and note:
            st.caption(f"🧭 {note}")


def _render_knowledge_timeline(df: pd.DataFrame) -> None:
    st.markdown("##### Knowledge Timeline")
    recent = df.sort_values("created_at", ascending=False).head(20)
    for _, e in recent.iterrows():
        note = f" — _{e['llm_note']}_" if pd.notna(e.get("llm_note")) and e.get("llm_note") else ""
        st.markdown(f"**{e['created_at']:%Y-%m-%d %H:%M UTC}** [{e['lifecycle_status']}] {e['conditions_str']}{note}")
        st.caption(e["description"])
