"""AI Lab — Knowledge Base Engine orchestration (DB + LLM wiring around the
pure stock_scanner.ai_lab.knowledge_base_engine module).

Curates ALL historical validated_hypotheses rows (validated and rejected,
every run) into long-lived KnowledgeEntry rows with a deterministic
lifecycle (pure code, deterministic, no LLM, no new Fisher/Wilson/BH math,
curation only), then makes one best-effort LLM call to
summarize/explain/organize/highlight changes. The published report's
`entries` never depend on that call succeeding.

Completely standalone: reads ai_recommendations + validated_hypotheses
(both read-only) and writes only to knowledge_entries / ai_learning_events
— never touches signals/outcomes/model_registry/knowledge_base (the
unrelated production table) or any production scanner file.

Extracted from scripts/run_knowledge_base_engine.py so
stock_scanner.ai_lab.pipeline can call it in-process as part of the
automated AI pipeline. scripts/run_knowledge_base_engine.py remains the CLI
entry point, now a thin wrapper around run() below.
"""
import json

from loguru import logger

from stock_scanner.ai_lab.agents.knowledge_review_agent import generate_knowledge_narrative
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterClient, NineRouterConfigError
from stock_scanner.ai_lab.knowledge_base_engine import generate_knowledge_entries
from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus
from stock_scanner.db.ai_lab import import_ai_recommendations, load_recommendations, log_learning_event
from stock_scanner.db.hypotheses import import_hypotheses, load_hypotheses
from stock_scanner.db.init_db import create_schema, get_connection
from stock_scanner.db.knowledge_entries import export_knowledge_report, import_knowledge_entries, upsert_knowledge_entries

_RESOLVED_STATUSES = ("CLOSED", "EXPIRED")


async def run(use_mock: bool = False, strong_threshold: int = 5, archive_margin: int = 3) -> dict:
    """Curate all historical validated hypotheses into knowledge entries.

    Reports SKIPPED (not FAILED) when there are no validated hypotheses
    available to curate — generate_knowledge_entries() already returns []
    for that case; this surfaces it as an explicit status.
    """
    conn = get_connection()
    create_schema(conn)
    import_ai_recommendations(conn)
    import_hypotheses(conn)

    all_recs = load_recommendations(conn)
    resolved_trade_count = int(all_recs["status"].isin(_RESOLVED_STATUSES).sum()) if not all_recs.empty else 0

    hypotheses_df = load_hypotheses(conn)  # unfiltered — need validated AND rejected, all history
    hypotheses = hypotheses_df.to_dict("records")
    for row in hypotheses:
        row["conditions"] = json.loads(row["conditions"]) if isinstance(row["conditions"], str) else row["conditions"]
        row["metadata_json"] = json.loads(row["metadata_json"]) if isinstance(row["metadata_json"], str) else row["metadata_json"]
    logger.info(f"knowledge_base_engine: {len(hypotheses)} historical hypothesis row(s) available to curate")

    entries = generate_knowledge_entries(hypotheses, strong_threshold=strong_threshold, archive_margin=archive_margin)
    by_status: dict[str, int] = {}
    for e in entries:
        by_status[e.lifecycle_status.value] = by_status.get(e.lifecycle_status.value, 0) + 1
    logger.info(
        f"knowledge_base_engine: {len(entries)} knowledge entrie(s) curated "
        f"(strong_threshold={strong_threshold}, archive_margin={archive_margin}) — {by_status}"
    )

    narrative_dict = None
    if entries:
        try:
            client = MockNineRouterClient() if use_mock else NineRouterClient()
            narrative = await generate_knowledge_narrative(client, entries)
        except NineRouterConfigError as e:
            logger.warning(f"knowledge_base_engine: 9router not configured, skipping narrative: {e}")
            narrative = None

        if narrative is not None:
            notes_by_id = {n.knowledge_id: n.note for n in narrative.knowledge_notes}
            entries = [
                e.model_copy(update={"llm_note": notes_by_id[e.knowledge_id]})
                if e.knowledge_id in notes_by_id else e
                for e in entries
            ]
            narrative_dict = {
                "overall_summary": narrative.overall_summary,
                "organized_groups": [g.model_dump() for g in narrative.organized_groups],
                "highlighted_changes": [h.model_dump() for h in narrative.highlighted_changes],
            }

    import_knowledge_entries(conn)
    upsert_knowledge_entries(conn, entries)
    log_learning_event(
        conn, event_type="knowledge_base_updated",
        description=(
            narrative_dict["overall_summary"] if narrative_dict else
            f"Curated {len(entries)} knowledge entry(ies) from {len(hypotheses)} historical hypothesis row(s)."
        ),
        metadata={
            "entry_count": len(entries), "by_lifecycle_status": by_status,
            "hypothesis_row_count": len(hypotheses), "resolved_trade_count": resolved_trade_count,
            "strong_threshold": strong_threshold, "archive_margin": archive_margin,
        },
    )

    export_knowledge_report(conn, narrative=narrative_dict, resolved_trade_count=resolved_trade_count)
    logger.info("knowledge_base_engine: published data/published/knowledge_report.json")

    if not entries:
        return {
            "status": PipelineStageStatus.SKIPPED,
            "reason": "no validated hypotheses available to curate",
            "hypothesis_row_count": len(hypotheses),
        }
    return {
        "status": PipelineStageStatus.OK,
        "entry_count": len(entries),
        "by_lifecycle_status": by_status,
        "hypothesis_row_count": len(hypotheses),
    }
