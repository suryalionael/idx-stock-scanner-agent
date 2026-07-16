#!/usr/bin/env python3
"""AI Lab — Reflection Engine orchestrator (NOT wired into any GitHub
Actions schedule; see docs/AI_LAB_ARCHITECTURE.md "What's phased for
later").

Reviews RESOLVED ai_recommendations (status IN ('CLOSED','EXPIRED')) and
produces statistically gated ReflectionObservation rows (see
stock_scanner/ai_lab/reflection_engine.py — pure code, deterministic, no
LLM), then makes one best-effort LLM call to summarize/prioritize them
(stock_scanner/ai_lab/agents/reflection_agent.py). The published report's
`observations` never depend on that call succeeding — a failed or
unconfigured LLM call just leaves `narrative: null`.

Completely standalone: reads ai_recommendations (read-only) and writes
only to reflection_observations / ai_learning_events — never touches
signals/outcomes/model_registry/knowledge_base or any production scanner
file.

Usage:
    python scripts/run_reflection_engine.py --mock          # no live 9router call, deterministic
    python scripts/run_reflection_engine.py                  # real 9router call (needs NINEROUTER_* env vars)
    python scripts/run_reflection_engine.py --min-n-success 5 --alpha 0.1
"""
import argparse
import asyncio
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger  # noqa: E402

from stock_scanner.ai_lab.agents.reflection_agent import generate_reflection_narrative  # noqa: E402
from stock_scanner.ai_lab.client import (  # noqa: E402
    MockNineRouterClient,
    NineRouterClient,
    NineRouterConfigError,
)
from stock_scanner.ai_lab.reflection_engine import generate_observations  # noqa: E402
from stock_scanner.db.ai_lab import (  # noqa: E402
    import_ai_recommendations,
    load_recommendations,
    log_learning_event,
)
from stock_scanner.db.init_db import create_schema, get_connection  # noqa: E402
from stock_scanner.db.reflection import (  # noqa: E402
    export_reflection_report,
    import_reflection_observations,
    upsert_observations,
)

_RESOLVED_STATUSES = ("CLOSED", "EXPIRED")


async def main(use_mock: bool, min_n_success: int, alpha: float) -> None:
    conn = get_connection()
    create_schema(conn)
    import_ai_recommendations(conn)

    all_recs = load_recommendations(conn)
    df_resolved = all_recs[all_recs["status"].isin(_RESOLVED_STATUSES)] if not all_recs.empty else all_recs
    logger.info(f"reflection_engine: {len(df_resolved)}/{len(all_recs)} recommendation(s) resolved")

    observations = generate_observations(df_resolved, min_n_success=min_n_success, alpha=alpha)
    logger.info(
        f"reflection_engine: {len(observations)} observation(s) gated "
        f"(min_n_success={min_n_success}, alpha={alpha})"
    )

    narrative_dict = None
    if observations:
        try:
            client = MockNineRouterClient() if use_mock else NineRouterClient()
            narrative = await generate_reflection_narrative(client, observations)
        except NineRouterConfigError as e:
            logger.warning(f"reflection_engine: 9router not configured, skipping narrative: {e}")
            narrative = None

        if narrative is not None:
            notes_by_id = {n.observation_id: n.note for n in narrative.observation_notes}
            observations = [
                o.model_copy(update={"llm_note": notes_by_id[o.observation_id]})
                if o.observation_id in notes_by_id else o
                for o in observations
            ]
            narrative_dict = {
                "overall_summary": narrative.overall_summary,
                "prioritized_observation_ids": narrative.prioritized_observation_ids,
            }

    import_reflection_observations(conn)
    upsert_observations(conn, observations)
    log_learning_event(
        conn, event_type="reflection_generated",
        description=(
            narrative_dict["overall_summary"] if narrative_dict else
            f"Generated {len(observations)} reflection observation(s) from "
            f"{len(df_resolved)} resolved recommendation(s)."
        ),
        metadata={
            "observation_count": len(observations), "resolved_trade_count": len(df_resolved),
            "min_n_success": min_n_success, "alpha": alpha,
        },
    )

    export_reflection_report(conn, narrative=narrative_dict, resolved_trade_count=len(df_resolved))
    logger.info("reflection_engine: published data/published/reflection_report.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="Use MockNineRouterClient instead of a live 9router call.")
    parser.add_argument("--min-n-success", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.mock, args.min_n_success, args.alpha))
