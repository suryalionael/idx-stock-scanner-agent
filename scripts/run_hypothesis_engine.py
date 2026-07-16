#!/usr/bin/env python3
"""AI Lab — Hypothesis Generator + Statistical Validation orchestrator
(NOT wired into any GitHub Actions schedule; see
docs/AI_LAB_ARCHITECTURE.md "What's phased for later").

Refines Reflection Engine's single-dimension findings into multi-condition
(order 2-3) candidate hypotheses (stock_scanner/ai_lab/hypothesis_engine.py
— pure code, deterministic, no LLM, apriori-seeded from
reflection_observations), scores them (stock_scanner/ai_lab/
statistical_validation.py — Wilson CI / Fisher's exact / shrunk win rate /
Benjamini-Hochberg, reused from reflection_engine.py/pattern_miner.py, not
re-derived), then makes one best-effort LLM call to summarize/prioritize/
cluster the results (stock_scanner/ai_lab/agents/hypothesis_review_agent.py).
The published report's `hypotheses` never depend on that call succeeding.

Completely standalone: reads ai_recommendations + reflection_observations
(both read-only) and writes only to validated_hypotheses / ai_learning_events
— never touches signals/outcomes/model_registry/knowledge_base or any
production scanner file.

Usage:
    python scripts/run_hypothesis_engine.py --mock          # no live 9router call, deterministic
    python scripts/run_hypothesis_engine.py                  # real 9router call (needs NINEROUTER_* env vars)
    python scripts/run_hypothesis_engine.py --min-n-success 5 --alpha 0.1 --max-order 2
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger  # noqa: E402

from stock_scanner.ai_lab.agents.hypothesis_review_agent import generate_hypothesis_narrative  # noqa: E402
from stock_scanner.ai_lab.client import (  # noqa: E402
    MockNineRouterClient,
    NineRouterClient,
    NineRouterConfigError,
)
from stock_scanner.ai_lab.hypothesis_engine import generate_candidate_hypotheses  # noqa: E402
from stock_scanner.ai_lab.statistical_validation import validate_hypotheses  # noqa: E402
from stock_scanner.db.ai_lab import import_ai_recommendations, load_recommendations, log_learning_event  # noqa: E402
from stock_scanner.db.hypotheses import export_hypotheses_report, import_hypotheses, upsert_hypotheses  # noqa: E402
from stock_scanner.db.init_db import create_schema, get_connection  # noqa: E402
from stock_scanner.db.reflection import import_reflection_observations, load_observations  # noqa: E402

_RESOLVED_STATUSES = ("CLOSED", "EXPIRED")


async def main(use_mock: bool, min_n_success: int, alpha: float, max_order: int) -> None:
    conn = get_connection()
    create_schema(conn)
    import_ai_recommendations(conn)
    import_reflection_observations(conn)

    all_recs = load_recommendations(conn)
    df_resolved = all_recs[all_recs["status"].isin(_RESOLVED_STATUSES)] if not all_recs.empty else all_recs
    total_n = len(df_resolved)
    total_success = int((df_resolved["trade_outcome"] == "WIN").sum()) if total_n else 0
    baseline_rate = total_success / total_n if total_n else 0.0
    logger.info(f"hypothesis_engine: {total_n} resolved recommendation(s), baseline win rate {baseline_rate:.1%}")

    reflection_df = load_observations(conn)
    reflection_observations = reflection_df.to_dict("records")
    for row in reflection_observations:
        stats = row["supporting_statistics"]
        row["supporting_statistics"] = json.loads(stats) if isinstance(stats, str) else stats
    logger.info(f"hypothesis_engine: {len(reflection_observations)} reflection observation(s) available as seeds")

    candidates = generate_candidate_hypotheses(df_resolved, reflection_observations, max_order=max_order)
    logger.info(f"hypothesis_engine: {len(candidates)} candidate hypothesis(es) generated (max_order={max_order})")

    hypotheses = validate_hypotheses(candidates, baseline_rate, total_n, total_success,
                                      min_n_success=min_n_success, alpha=alpha)
    validated_count = sum(1 for h in hypotheses if h.status.value == "validated")
    rejected_count = len(hypotheses) - validated_count
    logger.info(
        f"statistical_validation: {len(hypotheses)} scored — {validated_count} validated, "
        f"{rejected_count} rejected (min_n_success={min_n_success}, alpha={alpha})"
    )

    narrative_dict = None
    if hypotheses:
        try:
            client = MockNineRouterClient() if use_mock else NineRouterClient()
            narrative = await generate_hypothesis_narrative(client, hypotheses)
        except NineRouterConfigError as e:
            logger.warning(f"hypothesis_engine: 9router not configured, skipping narrative: {e}")
            narrative = None

        if narrative is not None:
            notes_by_id = {n.hypothesis_id: n.note for n in narrative.hypothesis_notes}
            hypotheses = [
                h.model_copy(update={"llm_note": notes_by_id[h.hypothesis_id]})
                if h.hypothesis_id in notes_by_id else h
                for h in hypotheses
            ]
            narrative_dict = {
                "overall_summary": narrative.overall_summary,
                "prioritized_hypothesis_ids": narrative.prioritized_hypothesis_ids,
                "clusters": [c.model_dump() for c in narrative.clusters],
            }

    import_hypotheses(conn)
    upsert_hypotheses(conn, hypotheses)
    log_learning_event(
        conn, event_type="hypothesis_validated",
        description=(
            narrative_dict["overall_summary"] if narrative_dict else
            f"Validated {validated_count}/{len(hypotheses)} candidate hypothesis(es) from "
            f"{len(candidates)} generated, seeded by {len(reflection_observations)} reflection observation(s)."
        ),
        metadata={
            "candidate_count": len(candidates), "validated_count": validated_count,
            "rejected_count": rejected_count, "resolved_trade_count": total_n,
            "min_n_success": min_n_success, "alpha": alpha, "max_order": max_order,
        },
    )

    export_hypotheses_report(conn, narrative=narrative_dict, resolved_trade_count=total_n)
    logger.info("hypothesis_engine: published data/published/hypotheses_report.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="Use MockNineRouterClient instead of a live 9router call.")
    parser.add_argument("--min-n-success", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-order", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.mock, args.min_n_success, args.alpha, args.max_order))
