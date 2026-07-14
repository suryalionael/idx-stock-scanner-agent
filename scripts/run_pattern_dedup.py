#!/usr/bin/env python3
"""De-duplicate a pattern_miner report's pairwise findings before they reach
an LLM. Read-only, no DB access, no LLM call — see
docs/LEARNING_AGENT_ARCHITECTURE.md.

Usage:
    python scripts/run_pattern_dedup.py
    python scripts/run_pattern_dedup.py --report data/reports/pattern_miner_2026-07-13.json
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger  # noqa: E402

from stock_scanner.learning.pattern_dedup import cluster_patterns, load_candidates_from_report  # noqa: E402

_REPORTS_DIR = repo_root / "data" / "reports"


def _write_json(clusters, out_path: Path, source_report: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(source_report),
        "n_clusters": len(clusters),
        "clusters": [c.to_dict() for c in clusters],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"JSON report written -> {out_path}")


def _fmt_slice(c) -> str:
    return ", ".join(f"{k}={v}" for k, v in c.slice_definition.items())


def _write_markdown(clusters, out_path: Path, n_input: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pattern Dedup Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**{n_input} passed pairwise patterns collapsed to {len(clusters)} clusters.**",
        "",
    ]
    for c in clusters:
        r = c.representative
        lines.append(f"## Cluster `{c.cluster_id}` — {c.member_count} member(s)")
        lines.append("")
        lines.append(
            f"Representative: **{_fmt_slice(r)}** — n={r.n}, n_success={r.n_success}, "
            f"win_rate={r.win_rate:.2%}, q-value={r.p_value_adjusted:.4f}"
        )
        if c.member_count > 1:
            lines.append("")
            lines.append("All members absorbed into this cluster:")
            for m in c.members:
                lines.append(f"- {_fmt_slice(m)} (n={m.n}, n_success={m.n_success}, win_rate={m.win_rate:.2%})")
        lines.append("")
    out_path.write_text("\n".join(lines))
    logger.info(f"Markdown report written -> {out_path}")


def main(report_path: Path, jaccard_threshold: float, overlap_threshold: float) -> None:
    if not report_path.exists():
        logger.error(f"Report not found: {report_path}. Run scripts/run_learning_agent.py first.")
        return

    candidates = load_candidates_from_report(report_path, order=2, passed_only=True)
    logger.info(f"Loaded {len(candidates)} passed pairwise candidates from {report_path}")
    if not candidates:
        logger.warning("No passed pairwise candidates to cluster.")
        return

    clusters = cluster_patterns(candidates, jaccard_threshold=jaccard_threshold, overlap_threshold=overlap_threshold)
    logger.info(f"{len(candidates)} candidates -> {len(clusters)} clusters")

    today = date.today().strftime("%Y-%m-%d")
    _write_json(clusters, _REPORTS_DIR / f"pattern_dedup_{today}.json", report_path)
    _write_markdown(clusters, _REPORTS_DIR / f"pattern_dedup_{today}.md", len(candidates))


if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="De-duplicate pairwise pattern_miner findings")
    parser.add_argument("--report", type=Path, default=_REPORTS_DIR / f"pattern_miner_{today_str}.json")
    parser.add_argument("--jaccard-threshold", type=float, default=0.6)
    parser.add_argument("--overlap-threshold", type=float, default=0.8)
    args = parser.parse_args()
    main(args.report, args.jaccard_threshold, args.overlap_threshold)
