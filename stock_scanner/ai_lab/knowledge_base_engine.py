"""Knowledge Base Engine — curates validated_hypotheses across ALL
historical runs into long-lived KnowledgeEntry rows with a deterministic
lifecycle (Emerging/Confirmed/Strong/Weakening/Contradicted/Archived).
Fourth/fifth components of the closed learning loop: Performance Tracker
-> Reflection Engine -> Hypothesis Generator -> Statistical Validation ->
Knowledge Base Engine (this module).

NOT another statistical engine — no new Fisher/Wilson/BH math here. This
is curation over what stock_scanner.ai_lab.statistical_validation already
decided. Deterministic, pure Python, no LLM, no DB writes, no side
effects.

"Deterministic similarity based on normalized conditions" (not LLM
clustering): stock_scanner.ai_lab.hypothesis_engine already normalizes
every hypothesis's `conditions` into a canonical sorted list at
generation time, so unlike stock_scanner.learning.pattern_dedup's problem
(raw signal-id-set overlap producing genuinely different-looking
candidate rows for the same underlying event), there is no "RSI>70 vs RSI
High vs RSI Top Quartile" divergence to reconcile here — every run that
rediscovers the same idea produces the EXACT SAME normalized
condition-set. So similarity is exact equality on the normalized
(sorted) condition-set — the simplest, strictest form, no threshold
tuning, no reuse of pattern_dedup's Jaccard/overlap machinery (same scope
boundary already established in the Hypothesis Generator phase).

"Never strengthen knowledge from duplicate evidence": each Hypothesis
Generator run re-scores a condition-set against the ENTIRE current
resolved-recommendations population (not an incremental slice) — summing
sample_size/successes across every historical row for a condition-set
would count the same underlying trades many times over. So
cumulative_sample_size/successes/failures, confidence_interval,
shrunk_win_rate, and evidence_strength all come from the LATEST
CONFIRMING row only (a snapshot, not a sum); average_win_rate IS a
genuine average of win_rate across every independent confirming run (a
ratio, not a raw count, so averaging doesn't double-count);
confirmation_count/contradiction_count count distinct validation runs,
never raw trades.

Contradiction direction: derived from each row's `metadata_json.baseline_rate`
(added in stock_scanner.ai_lab.statistical_validation) vs. its `win_rate`.
A group's `established_direction` is set once, by the first validated row
with a determinable direction, and never flips — a contradicted belief
stays framed around its original claim (matches the spec's own "Energy +
Breakout = Positive [then] = Negative" example: the entry doesn't become
a "Negative" entry, it becomes an increasingly contradicted "Positive"
one).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from stock_scanner.ai_lab.schemas import (
    KnowledgeEntry,
    KnowledgeLifecycleStatus,
    KnowledgePromotionStatus,
)

_DEFAULT_STRONG_THRESHOLD = 5  # matches the spec's own worked example: "Validation #5 -> Strong Knowledge"
_DEFAULT_ARCHIVE_MARGIN = 3  # contradictions must outnumber confirmations by this much to archive, not just tie


def _normalize_conditions(conditions: list[list[str]]) -> frozenset:
    return frozenset(tuple(c) for c in conditions)


def _direction(row: dict) -> str | None:
    """"success" if this row's win_rate beat its own run's baseline_rate,
    "failure" if it fell short, None if undeterminable (rejected row, or
    a legacy row missing metadata_json.baseline_rate — skipped, not
    crashed on)."""
    if row.get("status") != "validated":
        return None
    metadata = row.get("metadata_json") or {}
    baseline_rate = metadata.get("baseline_rate")
    win_rate = row.get("win_rate")
    if baseline_rate is None or win_rate is None or win_rate == baseline_rate:
        return None
    return "success" if win_rate > baseline_rate else "failure"


def _lifecycle_status(
    confirmation_count: int, contradiction_count: int, strong_threshold: int, archive_margin: int,
) -> KnowledgeLifecycleStatus:
    if contradiction_count >= confirmation_count + archive_margin:
        return KnowledgeLifecycleStatus.ARCHIVED
    if contradiction_count >= confirmation_count and contradiction_count > 0:
        return KnowledgeLifecycleStatus.CONTRADICTED
    if contradiction_count > 0:
        return KnowledgeLifecycleStatus.WEAKENING
    if confirmation_count >= strong_threshold:
        return KnowledgeLifecycleStatus.STRONG
    if confirmation_count >= 2:
        return KnowledgeLifecycleStatus.CONFIRMED
    return KnowledgeLifecycleStatus.EMERGING


def _process_group(rows: list[dict], strong_threshold: int, archive_margin: int) -> dict | None:
    """Walk one condition-set's full history in chronological order,
    replaying the lifecycle ladder at every directional event so the
    final status AND the status one step before it (previous_lifecycle_status)
    are both real, code-computed values. Returns None if this
    condition-set was never validated (nothing to know yet)."""
    rows_sorted = sorted(rows, key=lambda r: r["created_at"])

    established_direction: str | None = None
    confirmation_count = 0
    contradiction_count = 0
    first_seen: str | None = None
    last_confirmed: str | None = None
    latest_confirming_row: dict | None = None
    confirming_win_rates: list[float] = []
    originating_hypotheses: list[str] = []
    status_sequence: list[KnowledgeLifecycleStatus] = []

    for row in rows_sorted:
        originating_hypotheses.append(row["hypothesis_id"])
        direction = _direction(row)
        if direction is None:
            continue
        if first_seen is None:
            first_seen = row["created_at"]

        if established_direction is None or direction == established_direction:
            established_direction = direction
            confirmation_count += 1
            last_confirmed = row["created_at"]
            latest_confirming_row = row
            confirming_win_rates.append(row["win_rate"])
        else:
            contradiction_count += 1

        status_sequence.append(
            _lifecycle_status(confirmation_count, contradiction_count, strong_threshold, archive_margin)
        )

    if established_direction is None or latest_confirming_row is None:
        return None

    return {
        "originating_hypotheses": originating_hypotheses,
        "confirmation_count": confirmation_count,
        "contradiction_count": contradiction_count,
        "average_win_rate": round(sum(confirming_win_rates) / len(confirming_win_rates), 4),
        "cumulative_sample_size": latest_confirming_row["sample_size"],
        "cumulative_successes": latest_confirming_row["successes"],
        "cumulative_failures": latest_confirming_row["failures"],
        "shrunk_win_rate": latest_confirming_row["shrunk_win_rate"],
        "confidence_interval": [latest_confirming_row["wilson_lower"], latest_confirming_row["wilson_upper"]],
        "evidence_strength": latest_confirming_row.get("evidence_strength"),
        "first_seen": first_seen,
        "last_confirmed": last_confirmed,
        "lifecycle_status": status_sequence[-1],
        "previous_lifecycle_status": status_sequence[-2] if len(status_sequence) >= 2 else None,
    }


def _describe(conditions: list[list[str]], processed: dict) -> tuple[str, str]:
    cond_str = " AND ".join(f"{dim}={value}" for dim, value in conditions)
    title = f"{cond_str}: {processed['lifecycle_status'].value}"
    ci_lower, ci_upper = processed["confidence_interval"]
    description = (
        f"First observed {processed['first_seen']}, independently confirmed "
        f"{processed['confirmation_count']} time(s) (most recently {processed['last_confirmed']}) with an "
        f"average win rate of {processed['average_win_rate']:.1%} across confirming validations — latest "
        f"snapshot: {processed['cumulative_successes']}/{processed['cumulative_sample_size']} "
        f"(shrunk estimate {processed['shrunk_win_rate']:.1%}, 95% Wilson CI {ci_lower:.1%}-{ci_upper:.1%}). "
        f"{processed['contradiction_count']} contradicting validation(s) recorded."
    )
    return title, description


def generate_knowledge_entries(
    hypotheses: list[dict],
    strong_threshold: int = _DEFAULT_STRONG_THRESHOLD,
    archive_margin: int = _DEFAULT_ARCHIVE_MARGIN,
) -> list[KnowledgeEntry]:
    """Group ALL historical validated_hypotheses rows (validated and
    rejected, every run) by exact normalized condition-set, and curate
    each group into one KnowledgeEntry. Recomputes every entry from
    scratch each call — not an incremental update against a prior
    snapshot — which is what makes this trivially deterministic and
    reproducible: identical `hypotheses` input always produces an
    identical entry sequence. Returns [] on empty input, or if no
    condition-set has ever been validated — not an error, the correct
    honest result."""
    if not hypotheses:
        return []

    groups: dict[frozenset, list[dict]] = {}
    for row in hypotheses:
        key = _normalize_conditions(row["conditions"])
        groups.setdefault(key, []).append(row)

    created_at = datetime.now(timezone.utc).isoformat()
    entries = []
    for key, rows in groups.items():
        processed = _process_group(rows, strong_threshold, archive_margin)
        if processed is None:
            continue

        conditions = sorted(list(c) for c in key)
        id_src = f"{json.dumps(conditions, sort_keys=True)}|{created_at}"
        knowledge_id = hashlib.sha1(id_src.encode()).hexdigest()[:16]
        title, description = _describe(conditions, processed)

        entries.append(
            KnowledgeEntry(
                knowledge_id=knowledge_id, created_at=created_at, title=title, description=description,
                conditions=conditions, originating_hypotheses=processed["originating_hypotheses"],
                evidence_count=processed["confirmation_count"] + processed["contradiction_count"],
                cumulative_sample_size=processed["cumulative_sample_size"],
                cumulative_successes=processed["cumulative_successes"],
                cumulative_failures=processed["cumulative_failures"],
                average_win_rate=processed["average_win_rate"],
                shrunk_win_rate=processed["shrunk_win_rate"],
                confidence_interval=processed["confidence_interval"],
                first_seen=processed["first_seen"], last_confirmed=processed["last_confirmed"],
                confirmation_count=processed["confirmation_count"],
                contradiction_count=processed["contradiction_count"],
                evidence_strength=processed["evidence_strength"],
                lifecycle_status=processed["lifecycle_status"],
                previous_lifecycle_status=processed["previous_lifecycle_status"],
                # Deployment gate always starts at CANDIDATE — no automatic
                # promotion exists anywhere in this engine or anywhere else.
                # Only a future human-run promotion tool may advance it.
                promotion_status=KnowledgePromotionStatus.CANDIDATE,
            )
        )

    return sorted(entries, key=lambda e: (-e.confirmation_count, e.contradiction_count, e.knowledge_id))
