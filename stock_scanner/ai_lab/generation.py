"""AI Lab — recommendation generation.

Runs the Evidence -> Hypothesis Agent -> Decision Agent -> AI Recommendation
pipeline for a small candidate universe (top-N by the production scanner's
own quality_adjusted_score, from the most recent ranked CSV) across every
registered AI model persona, then stores + publishes the results.

Completely standalone: reads data/ranked/ranked_{date}.csv and
knowledge_base (both read-only, both already produced by existing,
untouched pipelines), writes only to ai_recommendations / ai_learning_events
— never touches signals/outcomes/model_registry/knowledge_base or any
production scanner file.

Extracted from scripts/run_ai_lab.py so stock_scanner.ai_lab.pipeline can
call it in-process as part of the automated AI pipeline (see
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md). scripts/run_ai_lab.py
remains the CLI entry point, now a thin wrapper around run() below.
"""
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_scanner.ai_lab.agents.decision_agent import assemble_recommendation, generate_decision
from stock_scanner.ai_lab.agents.hypothesis_agent import build_evidence, generate_hypothesis
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterClient
from stock_scanner.ai_lab.models import AI_MODEL_REGISTRY
from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus
from stock_scanner.ai_lab.scoring import (
    compute_confidence_breakdown,
    compute_decision_trace,
    compute_historical_comparison,
    generate_evidence_highlights,
)
from stock_scanner.db.ai_lab import (
    export_ai_recommendations,
    export_learning_events,
    import_ai_recommendations,
    import_learning_events,
    log_learning_event,
    upsert_recommendations,
)
from stock_scanner.db.init_db import create_schema, get_connection
from stock_scanner.db.knowledge_base import import_knowledge_base, load_knowledge_base

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RANKED_DIR = _REPO_ROOT / "data" / "ranked"
_DEFAULT_TOP_N = 15


def _load_latest_ranked(ranked_dir: Path) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    files = sorted(ranked_dir.glob("ranked_*.csv"))
    if not files:
        return None, None
    latest = files[-1]
    signal_date = latest.stem.replace("ranked_", "")
    return pd.read_csv(latest), signal_date


async def _run_one(client, ticker: str, feature_row: pd.Series, kb_df: pd.DataFrame,
                    model_spec, generated_date: str, ninerouter_model: str):
    evidence = build_evidence(ticker, feature_row, kb_df, model_spec)

    # Deterministic, code-computed — before either LLM call, and passed
    # INTO both prompts as already-final context (see docs/AI_LAB_ARCHITECTURE.md).
    trace = compute_decision_trace(model_spec, feature_row, evidence)
    confidence = compute_confidence_breakdown(trace)
    comparison = compute_historical_comparison(evidence, trace)
    highlights = generate_evidence_highlights(feature_row, evidence, trace)

    hypothesis = await generate_hypothesis(client, evidence, model_spec, highlights)
    if hypothesis is None:
        return None
    decision = await generate_decision(client, evidence, hypothesis, model_spec, trace, confidence, comparison)
    if decision is None:
        return None
    return assemble_recommendation(
        evidence, hypothesis, decision, model_spec, generated_date, ninerouter_model,
        trace, confidence, comparison, highlights,
    )


async def run(
    scan_date: str | None = None,
    top_n: int = _DEFAULT_TOP_N,
    model_keys: list[str] | None = None,
    use_mock: bool = False,
    ranked_dir: Path | None = None,
) -> dict:
    """Generate AI Lab recommendations for the latest ranked candidates.

    scan_date threads the actual Daily Scan date through to `generated_date`
    instead of always using wall-clock date — matters when this runs
    in-process right after a scan for a non-live/backfilled date. Standalone
    CLI use (scan_date=None) falls back to date.today(), unchanged from the
    original script's behavior.
    """
    ranked_dir = ranked_dir or _DEFAULT_RANKED_DIR
    model_keys = model_keys or list(AI_MODEL_REGISTRY)

    ranked_df, signal_date = _load_latest_ranked(ranked_dir)
    if ranked_df is None:
        reason = f"no ranked_*.csv found in {ranked_dir}"
        logger.warning(f"ai_lab: {reason} — nothing to do.")
        return {"status": PipelineStageStatus.SKIPPED, "reason": reason}

    sort_col = "quality_adjusted_score" if "quality_adjusted_score" in ranked_df.columns else "total_score"
    candidates = ranked_df.sort_values(sort_col, ascending=False).head(top_n)
    logger.info(f"ai_lab: {len(candidates)} candidate ticker(s) from {signal_date} (sorted by {sort_col})")

    conn = get_connection()
    create_schema(conn)
    import_knowledge_base(conn)
    kb_df = load_knowledge_base(conn)
    logger.info(f"ai_lab: {len(kb_df)} knowledge_base pattern(s) available as evidence")

    if use_mock:
        client = MockNineRouterClient()
        ninerouter_model = "mock"
    else:
        client = NineRouterClient()
        ninerouter_model = client.model

    generated_date = scan_date or date.today().isoformat()
    models = [AI_MODEL_REGISTRY[k] for k in model_keys]

    recommendations = []
    for model_spec in models:
        for _, feature_row in candidates.iterrows():
            ticker = feature_row["ticker"]
            rec = await _run_one(client, ticker, feature_row, kb_df, model_spec, generated_date, ninerouter_model)
            if rec is not None:
                recommendations.append(rec)

    logger.info(f"ai_lab: {len(recommendations)} recommendation(s) generated "
                f"across {len(models)} model(s) for {generated_date}")

    import_ai_recommendations(conn)
    import_learning_events(conn)
    upsert_recommendations(conn, recommendations)
    log_learning_event(
        conn, event_type="hypothesis_generated",
        description=f"Generated {len(recommendations)} recommendation(s) for {generated_date} "
                    f"across models: {', '.join(m.key for m in models)}.",
        metadata={"generated_date": generated_date, "candidate_count": len(candidates),
                  "models": [m.key for m in models]},
    )

    export_ai_recommendations(conn)
    export_learning_events(conn)
    logger.info("ai_lab: published data/published/ai_recommendations.json + ai_learning_events.json")

    return {
        "status": PipelineStageStatus.OK,
        "candidate_count": len(candidates),
        "recommendations_generated": len(recommendations),
        "signal_date": signal_date,
    }
