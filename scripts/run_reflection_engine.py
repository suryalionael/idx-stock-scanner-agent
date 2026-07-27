#!/usr/bin/env python3
"""AI Lab — Reflection Engine, CLI entry point.

Runs automatically as part of the Daily Scan's AI Automation Pipeline (see
the "AI Automation Pipeline" section of docs/AI_LAB_ARCHITECTURE.md and
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md) — this script remains
available for manual/standalone runs and debugging. The actual logic lives
in stock_scanner/ai_lab/reflection_runner.py::run().

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

from stock_scanner.ai_lab import reflection_runner  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="Use MockNineRouterClient instead of a live 9router call.")
    parser.add_argument("--min-n-success", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(reflection_runner.run(use_mock=args.mock, min_n_success=args.min_n_success, alpha=args.alpha))
