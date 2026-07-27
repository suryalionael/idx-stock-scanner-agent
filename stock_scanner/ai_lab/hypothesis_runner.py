"""AI Lab — Hypothesis Generator + Statistical Validation orchestration
(DB + LLM wiring around the pure stock_scanner.ai_lab.hypothesis_engine /
statistical_validation modules).

Refines Reflection Engine's single-dimension findings into multi-condition
(order 2-3) candidate hypotheses (pure code, deterministic, no LLM,
apriori-seeded from reflection_observations), scores them (Wilson CI /
Fisher's exact / shrunk win rate / Benjamini-Hochberg, reused from
reflection_engine.py/pattern_miner.py, not re-derived), then makes one
best-effort LLM call to summarize/prioritize/cluster the results. The
published report's `hypotheses` never depend on that call succeeding.

Completely standalone: reads ai_recommendations + reflection_observations
(both read-only) and writes only to validated_hypotheses / ai_learning_events
— never touches signals/outcomes/model_registry/knowledge_base or any
production scanner file.

Extracted from scripts/run_hypothesis_engine.py so
stock_scanner.ai_lab.pipeline can call it in-process as part of the
automated AI pipeline. This one call covers both "Hypothesis Generator"
and "Statistical Validation" in the automation diagram — they are not
separate stages. scripts/run_hypothesis_engine.py remains the CLI entry
point, now a thin wrapper around run() below.
"""
import json

from loguru import logger

from stock_scanner.ai_lab.agents.hypothesis_review_agent import generate_hypothesis_narrative
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterClient, NineRouterConfigError
from stock_scanner.ai_lab.hypothesis_engine import generate_candidate_hypotheses
from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus
from stock_scanner.ai_lab.statistical_validation import validate_hypotheses
from stock_scanner.db.ai_lab import import_ai_recommendations, load_recommendations, log_learning_event
from stock_scanner.db.hypotheses import export_hypotheses_report, import_hypotheses, upsert_hypotheses
from stock_scanner.db.init_db import create_schema, get_connection
from stock_scanner.db.reflection import import_reflection_observations, load_observations

_RESOLVED_STATUSES = ("CLOSED", "EXPIRED")


async def run(use_mock: bool = False, min_n_success: int = 3, alpha: float = 0.05, max_order: int = 3) -> dict:
    """Generate and statistically validate candidate hypotheses.

    Reports SKIPPED (not FAILED) when there are no gated reflection
    observations to seed from, or no candidate cleared the significance
    gate — generate_candidate_hypotheses()/validate_hypotheses() already
    return [] for that case; this surfaces it as an explicit status.
    """
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

    if not hypotheses:
        return {
            "status": PipelineStageStatus.SKIPPED,
            "reason": "no gated reflection observations to seed from, or no candidate cleared validation",
            "resolved_trade_count": total_n,
        }
    return {
        "status": PipelineStageStatus.OK,
        "candidate_count": len(candidates),
        "validated_count": validated_count,
        "rejected_count": rejected_count,
        "resolved_trade_count": total_n,
    }
