#!/usr/bin/env python3
"""Learning Agent — Phase 1 (statistics-only pattern mining).

Answers one question: does the historical database contain enough
statistical signal to justify an LLM hypothesis stage (Phase 2, via
9router)? No LLM call happens here. No database writes happen here — this
is a read-only research tool. See docs/LEARNING_AGENT_ARCHITECTURE.md.

Usage:
    python scripts/run_learning_agent.py
    python scripts/run_learning_agent.py --min-n-success 8 --alpha 0.05
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from stock_scanner.db.init_db import get_connection  # noqa: E402
from stock_scanner.learning.pattern_miner import MiningResult, mine_patterns  # noqa: E402
from scripts.train_challenger import load_training_examples  # noqa: E402

_REPORTS_DIR = repo_root / "data" / "reports"


def load_sector_reference(conn) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT ticker, sector FROM sector_reference", conn)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"sector_reference load failed (sector dimension will be skipped): {e}")
        return pd.DataFrame(columns=["ticker", "sector"])


def _distinct_dimensions(candidates: list) -> set:
    dims: set = set()
    for c in candidates:
        dims.update(c.dimensions)
    return dims


def _verdict(result: MiningResult) -> str:
    primary = result.candidates_by_order.get(1, []) + result.candidates_by_order.get(2, [])
    passed = [c for c in primary if c.passed_gate]
    n_passed = len(passed)
    n_core_dims = len(_distinct_dimensions(passed))
    if n_passed == 0:
        return (
            f"NOT ENOUGH SIGNAL YET — no single- or pairwise-feature pattern cleared the "
            f"statistical gate (n_success>=8, FDR-adjusted p<0.05, Wilson CI lower bound > "
            f"baseline) at current volume (n={result.total_n}, positives={result.total_success}, "
            f"baseline={result.baseline_win_rate:.2%}). Recommend accumulating more labeled "
            f"outcomes before building the LLM hypothesis stage (Phase 2)."
        )
    if n_passed < 3:
        return (
            f"MARGINAL SIGNAL — {n_passed} pattern(s) cleared the gate. Worth reviewing "
            f"individually, but too few and too provisional to justify automating hypothesis "
            f"generation yet. Re-run after more data accumulates."
        )
    redundancy_note = (
        f" NOTE: these {n_passed} rows involve only {n_core_dims} distinct underlying "
        f"dimensions — technical features that are mechanically correlated (e.g. volume/"
        f"volatility expansion co-occurs) will each pass individually and then recombine "
        f"pairwise, inflating the row count without adding {n_passed} independent "
        f"discoveries. Read the single-feature table as the core findings; treat the "
        f"pairwise table as showing how those same core signals combine, not as that many "
        f"separate discoveries. De-duplicating correlated findings before handing them to "
        f"an LLM (Phase 2) is real future work, not yet built."
        if n_passed >= 10 and n_core_dims < n_passed / 2 else ""
    )
    return (
        f"{n_passed} patterns cleared the gate ({n_core_dims} distinct underlying "
        f"dimensions involved) — plausible candidates exist.{redundancy_note} LLM-based "
        f"hypothesis articulation (Phase 2, via 9router) is worth pursuing, but review the "
        f"ticker-concentration and time-split-stability flags on each before trusting it."
    )


def _write_json(result: MiningResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_n": result.total_n,
        "total_success": result.total_success,
        "baseline_win_rate": round(result.baseline_win_rate, 4),
        "verdict": _verdict(result),
        "candidates_by_order": {
            str(order): [c.to_dict() for c in candidates]
            for order, candidates in result.candidates_by_order.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"JSON report written -> {out_path}")


def _fmt_slice(c) -> str:
    return ", ".join(f"{k}={v}" for k, v in c.slice_definition.items())


def _write_markdown(result: MiningResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pattern Miner Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Verdict:** {_verdict(result)}",
        "",
        f"- Total evaluated signals: {result.total_n}",
        f"- Total positive labels: {result.total_success}",
        f"- Baseline win rate: {result.baseline_win_rate:.2%}",
        "",
    ]

    for order, title in ((1, "Single-feature patterns"), (2, "Pairwise-feature patterns")):
        candidates = sorted(
            [c for c in result.candidates_by_order.get(order, []) if c.passed_gate],
            key=lambda c: c.p_value_adjusted,
        )
        n_core_dims = len(_distinct_dimensions(candidates))
        lines.append(
            f"## {title} (gated, FDR-corrected) — {len(candidates)} passed "
            f"({n_core_dims} distinct underlying dimensions involved)"
        )
        lines.append("")
        if order == 2 and candidates:
            lines.append(
                "> A pairwise row passing the gate does not mean it's an independent "
                "discovery from the single-feature table above — mechanically correlated "
                "features (e.g. volume/volatility expansion measures) each pass alone and "
                "then recombine here. Read this table as *how the core single-feature "
                "signals combine*, not as that many separate findings."
            )
            lines.append("")
        if not candidates:
            lines.append("_None cleared the gate at current data volume._")
            lines.append("")
            continue
        lines.append("| Slice | n | n_success | win_rate | shrunk | CI lower | q-value | ticker_conc | time_stable |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in candidates:
            lines.append(
                f"| {_fmt_slice(c)} | {c.n} | {c.n_success} | {c.win_rate:.2%} | "
                f"{c.win_rate_shrunk:.2%} | {c.ci_lower:.2%} | {c.p_value_adjusted:.4f} | "
                f"{'⚠ ' + f'{c.ticker_concentration:.0%}' if c.ticker_concentration_flag else '—'} | "
                f"{c.time_split_stable if c.time_split_stable is not None else 'inconclusive'} |"
            )
        lines.append("")

    triple = sorted(
        result.candidates_by_order.get(3, []),
        key=lambda c: c.p_value_adjusted if c.p_value_adjusted is not None else 1.0,
    )[:20]
    lines.append(
        f"## Triple-feature patterns (EXPLORATORY — {len(result.candidates_by_order.get(3, []))} tested, "
        f"top 20 by adjusted p-value shown; do NOT treat as validated findings, see note below)"
    )
    lines.append("")
    lines.append(
        "> At this database's current volume, a 5x5x5 quintile cross averages only a handful "
        "of positives per cell — even a technically-passing p-value here is much more likely "
        "to be noise than a genuine pattern. Treat this section as leads to revisit once more "
        "data accumulates, not as findings."
    )
    lines.append("")
    if triple:
        lines.append("| Slice | n | n_success | win_rate | q-value | passed_gate (informational only) |")
        lines.append("|---|---|---|---|---|---|")
        for c in triple:
            lines.append(
                f"| {_fmt_slice(c)} | {c.n} | {c.n_success} | {c.win_rate:.2%} | "
                f"{c.p_value_adjusted:.4f} | {c.passed_gate} |"
            )
    out_path.write_text("\n".join(lines))
    logger.info(f"Markdown report written -> {out_path}")


def main(min_n_success: int, alpha: float, max_order: int) -> None:
    if max_order >= 3:
        logger.warning(
            "max-order=3 requested — triple-feature search is C(n_dims,3) combinations "
            "(thousands at current dimension count) and can take several minutes; it's "
            "exploratory/uncorrected in the report either way (see "
            "docs/LEARNING_AGENT_ARCHITECTURE.md). Use --max-order 2 for the fast, primary "
            "single+pairwise result."
        )

    conn = get_connection()
    df = load_training_examples(conn)
    if df.empty:
        logger.warning("No training examples in DB — nothing to mine.")
        conn.close()
        return
    sector_df = load_sector_reference(conn)
    conn.close()

    logger.info(
        f"Loaded {len(df)} evaluated signals, {int(df['label_success'].sum())} positive "
        f"({df['label_success'].mean():.2%}). Sector coverage: "
        f"{df['ticker'].isin(sector_df['ticker']).mean():.1%} of rows."
    )

    result = mine_patterns(df, sector_df=sector_df, min_n_success=min_n_success, alpha=alpha, max_order=max_order)

    logger.info(_verdict(result))

    today = date.today().strftime("%Y-%m-%d")
    _write_json(result, _REPORTS_DIR / f"pattern_miner_{today}.json")
    _write_markdown(result, _REPORTS_DIR / f"pattern_miner_{today}.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Agent Phase 1 — statistics-only pattern miner")
    parser.add_argument("--min-n-success", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--max-order", type=int, default=2, choices=[1, 2, 3],
        help="Default 2 (single+pairwise, ~seconds). 3 adds exploratory triple-feature "
             "search, which can take several minutes at current dimension count.",
    )
    args = parser.parse_args()
    main(args.min_n_success, args.alpha, args.max_order)
