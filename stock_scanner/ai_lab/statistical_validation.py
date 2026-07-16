"""Statistical Validation — the ONLY place in the Hypothesis Generator
pipeline that calls Wilson CI / Fisher's exact test / shrunk win rate /
Benjamini-Hochberg correction. Reuses
stock_scanner.ai_lab.reflection_engine.score_group() and
passes_slice_gate() (already generic — take only n_success/n/baseline_rate/
total_n/total_success) rather than re-deriving the same math a third time
in this codebase (pattern_miner.py -> reflection_engine.py -> here). See
docs/AI_LAB_ARCHITECTURE.md "Hypothesis Generator + Statistical
Validation" for the full split rationale between hypothesis_engine.py
(pure candidate generation, no stats) and this module (stats only, no
candidate generation).

BH correction is applied separately per interaction order (order-2 pool,
order-3 pool) — same per-tier-separation rationale pattern_miner and
reflection_engine already established: order-3's larger test count
shouldn't dilute order-2's statistical power.

Every candidate becomes a Hypothesis row, validated or rejected — nothing
is silently dropped, so a rejected hypothesis's rejection_reason/
failed_gate is always traceable back to real numbers, never invented.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from stock_scanner.ai_lab.reflection_engine import passes_slice_gate, score_group
from stock_scanner.ai_lab.schemas import EvidenceStrength, Hypothesis, HypothesisCandidate, HypothesisStatus
from stock_scanner.learning.pattern_miner import benjamini_hochberg

_STRONG_THRESHOLD = 0.01  # bh_adjusted_p below this -> STRONG, else MODERATE (both already cleared alpha)


def _hypothesis_id(conditions: list[list[str]], created_at: str) -> str:
    conditions_repr = json.dumps(conditions, sort_keys=True)
    return hashlib.sha1(f"{conditions_repr}|{created_at}".encode()).hexdigest()[:16]


def _describe(conditions: list[list[str]], stats: dict, avg_return_percentage: float | None, avg_holding_days: float | None) -> str:
    cond_str = " AND ".join(f"{dim}={value}" for dim, value in conditions)
    direction = "outperforms" if stats["win_rate"] > stats["baseline_rate"] else "underperforms"
    return (
        f"Recommendations where {cond_str} realized a {stats['win_rate']:.1%} win rate "
        f"({stats['n_success']}/{stats['n']} wins, shrunk estimate {stats['win_rate_shrunk']:.1%}) vs a "
        f"{stats['baseline_rate']:.1%} baseline — {direction} baseline (95% Wilson CI "
        f"{stats['ci_lower']:.1%}-{stats['ci_upper']:.1%}, Fisher's exact BH-adjusted "
        f"p={stats['p_value_adjusted']:.4f}). Average return {avg_return_percentage}%, "
        f"average holding period {avg_holding_days} trading days."
    )


def validate_hypotheses(
    candidates: list[HypothesisCandidate],
    baseline_rate: float,
    total_n: int,
    total_success: int,
    min_n_success: int = 3,
    alpha: float = 0.05,
) -> list[Hypothesis]:
    """Score every candidate (Wilson CI / Fisher's exact / shrunk win rate,
    BH-corrected per interaction order) and classify validated vs.
    rejected. Returns [] on empty input — not an error."""
    if not candidates:
        return []

    created_at = datetime.now(timezone.utc).isoformat()
    results: list[Hypothesis] = []

    by_order: dict[int, list[HypothesisCandidate]] = {}
    for c in candidates:
        by_order.setdefault(c.order, []).append(c)

    for order in sorted(by_order):
        group = by_order[order]
        scored = [score_group(c.n_success, c.n, baseline_rate, total_n, total_success) for c in group]
        q_values = benjamini_hochberg([s["p_value"] for s in scored])

        for candidate, stats, q in zip(group, scored, q_values):
            stats["p_value_adjusted"] = q
            passed = q < alpha and passes_slice_gate(stats, min_n_success)
            hid = _hypothesis_id(candidate.conditions, created_at)
            description = _describe(candidate.conditions, stats, candidate.avg_return_percentage, candidate.avg_holding_days)

            if passed:
                status = HypothesisStatus.VALIDATED
                evidence_strength = EvidenceStrength.STRONG if q < _STRONG_THRESHOLD else EvidenceStrength.MODERATE
                rejection_reason = None
                failed_gate = None
            else:
                status = HypothesisStatus.REJECTED
                evidence_strength = None
                if q >= alpha:
                    failed_gate = "not_significant"
                    rejection_reason = f"BH-adjusted p-value {q:.4f} does not clear alpha={alpha}."
                else:
                    failed_gate = "no_directional_lift"
                    rejection_reason = (
                        f"Statistically significant (BH-adjusted p={q:.4f}) but the 95% Wilson CI "
                        f"({stats['ci_lower']:.1%}-{stats['ci_upper']:.1%}) doesn't clearly separate from the "
                        f"{baseline_rate:.1%} baseline with enough supporting trades "
                        f"(min_n_success={min_n_success})."
                    )

            results.append(
                Hypothesis(
                    hypothesis_id=hid, created_at=created_at, description=description,
                    conditions=candidate.conditions, sample_size=candidate.n, successes=candidate.n_success,
                    failures=candidate.n_failure, win_rate=stats["win_rate"], shrunk_win_rate=stats["win_rate_shrunk"],
                    wilson_lower=stats["ci_lower"], wilson_upper=stats["ci_upper"], fisher_p=stats["p_value"],
                    bh_adjusted_p=q, evidence_strength=evidence_strength, status=status,
                    rejection_reason=rejection_reason, failed_gate=failed_gate,
                    source_reflection_ids=candidate.source_reflection_ids,
                    metadata_json={
                        "avg_return_percentage": candidate.avg_return_percentage,
                        "avg_holding_days": candidate.avg_holding_days,
                        "interaction_order": order,
                        "baseline_rate": baseline_rate,
                    },
                )
            )

    return sorted(results, key=lambda h: (h.status != HypothesisStatus.VALIDATED, h.bh_adjusted_p, h.hypothesis_id))
