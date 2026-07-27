#!/usr/bin/env python3
"""AI Lab — recommendation generation, CLI entry point.

Runs automatically as part of the Daily Scan's AI Automation Pipeline (see
the "AI Automation Pipeline" section of docs/AI_LAB_ARCHITECTURE.md and
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md) — this script remains
available for manual/standalone runs and debugging. The actual logic lives
in stock_scanner/ai_lab/generation.py::run().

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
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stock_scanner.ai_lab import generation  # noqa: E402
from stock_scanner.ai_lab.models import AI_MODEL_REGISTRY  # noqa: E402

_DEFAULT_RANKED_DIR = repo_root / "data" / "ranked"
_DEFAULT_TOP_N = 15


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
    asyncio.run(generation.run(top_n=args.top_n, model_keys=model_keys,
                                use_mock=args.mock, ranked_dir=args.ranked_dir))
