"""AI Lab — Reflection Engine orchestration (DB + LLM wiring around the pure
stock_scanner.ai_lab.reflection_engine module).

Reviews RESOLVED ai_recommendations (status IN ('CLOSED','EXPIRED')) and
produces statistically gated ReflectionObservation rows (pure code,
deterministic, no LLM), then makes one best-effort LLM call to
summarize/prioritize them. The published report's `observations` never
depend on that call succeeding — a failed or unconfigured LLM call just
leaves `narrative: null`.

Completely standalone: reads ai_recommendations (read-only) and writes
only to reflection_observations / ai_learning_events — never touches
signals/outcomes/model_registry/knowledge_base or any production scanner
file.

Extracted from scripts/run_reflection_engine.py so
stock_scanner.ai_lab.pipeline can call it in-process as part of the
automated AI pipeline. scripts/run_reflection_engine.py remains the CLI
entry point, now a thin wrapper around run() below.
"""
from loguru import logger

from stock_scanner.ai_lab.agents.reflection_agent import generate_reflection_narrative
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterClient, NineRouterConfigError
from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus
from stock_scanner.ai_lab.reflection_engine import generate_observations
from stock_scanner.db.ai_lab import import_ai_recommendations, load_recommendations, log_learning_event
from stock_scanner.db.init_db import create_schema, get_connection
from stock_scanner.db.reflection import export_reflection_report, import_reflection_observations, upsert_observations

_RESOLVED_STATUSES = ("CLOSED", "EXPIRED")


async def run(use_mock: bool = False, min_n_success: int = 3, alpha: float = 0.05) -> dict:
    """Generate statistically-gated reflection observations from resolved trades.

    Reports SKIPPED (not FAILED) when there simply isn't enough resolved,
    statistically significant evidence yet — generate_observations() already
    returns [] for that case; this just surfaces it as an explicit status
    instead of leaving callers to infer "empty list = nothing happened."
    """
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

    if not observations:
        return {
            "status": PipelineStageStatus.SKIPPED,
            "reason": "insufficient resolved recommendations",
            "resolved_trade_count": len(df_resolved),
        }
    return {
        "status": PipelineStageStatus.OK,
        "observation_count": len(observations),
        "resolved_trade_count": len(df_resolved),
    }
