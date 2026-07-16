#!/usr/bin/env python3
"""AI Lab — Knowledge Base Engine orchestrator (NOT wired into any GitHub
Actions schedule; see docs/AI_LAB_ARCHITECTURE.md "What's phased for
later").

Curates ALL historical validated_hypotheses rows (validated and rejected,
every run) into long-lived KnowledgeEntry rows with a deterministic
lifecycle (stock_scanner/ai_lab/knowledge_base_engine.py — pure code,
deterministic, no LLM, no new Fisher/Wilson/BH math, curation only), then
makes one best-effort LLM call to summarize/explain/organize/highlight
changes (stock_scanner/ai_lab/agents/knowledge_review_agent.py). The
published report's `entries` never depend on that call succeeding.

Completely standalone: reads ai_recommendations + validated_hypotheses
(both read-only) and writes only to knowledge_entries / ai_learning_events
— never touches signals/outcomes/model_registry/knowledge_base (the
unrelated production table) or any production scanner file.

Usage:
    python scripts/run_knowledge_base_engine.py --mock          # no live 9router call, deterministic
    python scripts/run_knowledge_base_engine.py                  # real 9router call (needs NINEROUTER_* env vars)
    python scripts/run_knowledge_base_engine.py --strong-threshold 3 --archive-margin 2
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger  # noqa: E402

from stock_scanner.ai_lab.agents.knowledge_review_agent import generate_knowledge_narrative  # noqa: E402
from stock_scanner.ai_lab.client import (  # noqa: E402
    MockNineRouterClient,
    NineRouterClient,
    NineRouterConfigError,
)
from stock_scanner.ai_lab.knowledge_base_engine import generate_knowledge_entries  # noqa: E402
from stock_scanner.db.ai_lab import import_ai_recommendations, load_recommendations, log_learning_event  # noqa: E402
from stock_scanner.db.hypotheses import import_hypotheses, load_hypotheses  # noqa: E402
from stock_scanner.db.init_db import create_schema, get_connection  # noqa: E402
from stock_scanner.db.knowledge_entries import (  # noqa: E402
    export_knowledge_report,
    import_knowledge_entries,
    upsert_knowledge_entries,
)

_RESOLVED_STATUSES = ("CLOSED", "EXPIRED")


async def main(use_mock: bool, strong_threshold: int, archive_margin: int) -> None:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="Use MockNineRouterClient instead of a live 9router call.")
    parser.add_argument("--strong-threshold", type=int, default=5)
    parser.add_argument("--archive-margin", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.mock, args.strong_threshold, args.archive_margin))
