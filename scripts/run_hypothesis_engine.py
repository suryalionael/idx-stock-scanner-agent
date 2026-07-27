#!/usr/bin/env python3
"""AI Lab — Hypothesis Generator + Statistical Validation, CLI entry point.

Runs automatically as part of the Daily Scan's AI Automation Pipeline (see
the "AI Automation Pipeline" section of docs/AI_LAB_ARCHITECTURE.md and
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md) — this script remains
available for manual/standalone runs and debugging. The actual logic lives
in stock_scanner/ai_lab/hypothesis_runner.py::run() (one call covers both
"Hypothesis Generator" and "Statistical Validation" — they are not
separate stages/scripts).

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
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stock_scanner.ai_lab import hypothesis_runner  # noqa: E402


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
    asyncio.run(hypothesis_runner.run(use_mock=args.mock, min_n_success=args.min_n_success,
                                       alpha=args.alpha, max_order=args.max_order))
