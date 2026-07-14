#!/usr/bin/env python3
"""AI Lab — manual orchestrator (NOT wired into any GitHub Actions schedule).

Runs the Evidence -> Hypothesis Agent -> Decision Agent -> AI Recommendation
pipeline for a small candidate universe (top-N by the production scanner's
own quality_adjusted_score, from the most recent ranked CSV) across every
registered AI model persona, then stores + publishes the results.

Completely standalone: reads data/ranked/ranked_{date}.csv and
knowledge_base (both read-only, both already produced by existing,
untouched pipelines), writes only to ai_recommendations / ai_learning_events
— never touches signals/outcomes/model_registry/knowledge_base or any
production scanner file.

Usage:
    python scripts/run_ai_lab.py --mock                  # no live 9router call, deterministic
    python scripts/run_ai_lab.py                          # real 9router call (needs NINEROUTER_* env vars)
    python scripts/run_ai_lab.py --top-n 10 --models momentum_ai,breakout_ai
"""
import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from stock_scanner.ai_lab.agents.decision_agent import (  # noqa: E402
    assemble_recommendation,
    generate_decision,
)
from stock_scanner.ai_lab.agents.hypothesis_agent import (  # noqa: E402
    build_evidence,
    generate_hypothesis,
)
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterClient  # noqa: E402
from stock_scanner.ai_lab.models import AI_MODEL_REGISTRY  # noqa: E402
from stock_scanner.ai_lab.scoring import (  # noqa: E402
    compute_confidence_breakdown,
    compute_decision_trace,
    compute_historical_comparison,
    generate_evidence_highlights,
)
from stock_scanner.db.ai_lab import (  # noqa: E402
    export_ai_recommendations,
    export_learning_events,
    import_ai_recommendations,
    import_learning_events,
    log_learning_event,
    upsert_recommendations,
)
from stock_scanner.db.init_db import create_schema, get_connection  # noqa: E402
from stock_scanner.db.knowledge_base import import_knowledge_base, load_knowledge_base  # noqa: E402

_DEFAULT_RANKED_DIR = repo_root / "data" / "ranked"
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


async def main(top_n: int, model_keys: list[str], use_mock: bool, ranked_dir: Path) -> None:
    ranked_df, signal_date = _load_latest_ranked(ranked_dir)
    if ranked_df is None:
        logger.warning(f"ai_lab: no ranked_*.csv found in {ranked_dir} — nothing to do.")
        return

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

    generated_date = date.today().isoformat()
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N,
                        help="Number of top-ranked candidate tickers to evaluate.")
    parser.add_argument("--models", type=str, default=",".join(AI_MODEL_REGISTRY),
                        help="Comma-separated AI model keys to run.")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockNineRouterClient instead of a live 9router call.")
    parser.add_argument("--ranked-dir", type=Path, default=_DEFAULT_RANKED_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    asyncio.run(main(args.top_n, model_keys, args.mock, args.ranked_dir))
