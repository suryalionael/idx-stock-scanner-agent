#!/usr/bin/env python3
"""AI Lab — Knowledge Base Engine, CLI entry point.

Runs automatically as part of the Daily Scan's AI Automation Pipeline (see
the "AI Automation Pipeline" section of docs/AI_LAB_ARCHITECTURE.md and
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md) — this script remains
available for manual/standalone runs and debugging. The actual logic lives
in stock_scanner/ai_lab/knowledge_runner.py::run().

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
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stock_scanner.ai_lab import knowledge_runner  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="Use MockNineRouterClient instead of a live 9router call.")
    parser.add_argument("--strong-threshold", type=int, default=5)
    parser.add_argument("--archive-margin", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(knowledge_runner.run(use_mock=args.mock, strong_threshold=args.strong_threshold,
                                      archive_margin=args.archive_margin))
