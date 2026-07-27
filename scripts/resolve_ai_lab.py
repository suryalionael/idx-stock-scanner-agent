#!/usr/bin/env python3
"""AI Lab — recommendation resolver ("Performance Tracker"), CLI entry point.

Runs automatically as part of the Daily Scan's AI Automation Pipeline (see
the "AI Automation Pipeline" section of docs/AI_LAB_ARCHITECTURE.md and
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md) — this script remains
available for manual/standalone runs and debugging. The actual logic lives
in stock_scanner/ai_lab/resolution.py::run().

Advances every AI Lab recommendation's lifecycle against forward OHLCV
data: PENDING -> ACTIVE (entry_price set once generated_date's close is
available) and ACTIVE -> CLOSED | EXPIRED (TP/SL simulation, same fallback
exit policy as stock_scanner.pipeline.evaluator). Also refreshes running
highest_price/lowest_price/max_runup_pct/max_drawdown_pct/holding_days on
every ACTIVE row each run, even when it doesn't resolve this time — see
stock_scanner/ai_lab/resolver.py.

Completely standalone: reads data/raw/*.parquet (read-only, produced by the
existing scan pipeline) and writes only to ai_recommendations /
ai_learning_events — never touches signals/outcomes/model_registry or any
production scanner file.

Usage:
    python scripts/resolve_ai_lab.py                              # data/raw, horizon=10, risk_pct=3.0
    python scripts/resolve_ai_lab.py --horizon-days 15 --risk-pct 2.5
"""
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stock_scanner.ai_lab import resolution  # noqa: E402

_DEFAULT_RAW_DIR = repo_root / "data" / "raw"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=_DEFAULT_RAW_DIR)
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--risk-pct", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    resolution.run(raw_dir=args.raw_dir, horizon_days=args.horizon_days, risk_pct=args.risk_pct)
