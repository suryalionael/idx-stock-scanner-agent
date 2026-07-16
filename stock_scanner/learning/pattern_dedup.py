"""De-duplicate pairwise pattern candidates before anything reaches an LLM.

Phase 1's own report showed why this stage is required, not optional: 224
"passed" pairwise rows reduced to 26 distinct underlying dimensions once
traced by hand. Mechanically-correlated technical features (volume/
volatility expansion co-occurs) each pass the statistical gate individually
and then recombine pairwise — those recombinations are the same underlying
event described from different angles, not independent discoveries.

Clustering key: Jaccard similarity of the signal_id sets behind each
candidate's n_success — two patterns whose positive examples overlap
heavily ARE the same underlying event, regardless of which dimension pair
produced them. Jaccard alone misses one real case: a broad pattern with
many supporters and a narrower one that is a strict subset of it have a low
Jaccard score purely from the size mismatch (e.g. 8/50 = 0.16) even though
the narrower one is fully contained in the broader one. The overlap
coefficient (Szymkiewicz-Simpson: |A∩B| / min(|A|,|B|)) catches exactly
this containment case (8/min(50,8) = 1.0) — a candidate is merged into a
cluster if EITHER measure clears its threshold.

Read-only, research-only: no LLM, no database access. See
docs/LEARNING_AGENT_ARCHITECTURE.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from stock_scanner.learning.pattern_miner import PatternCandidate

# NOTE on clustering method: an earlier version of this module used union-find
# (single-linkage) — merge i and j whenever are_duplicates(i,j) holds, then take
# connected components. On the real Phase 1 report this collapsed 224 candidates
# into ONE 223-member cluster: single-linkage chains transitively (A~B, B~C implies
# A merges with C even if A and C individually share almost nothing), and with
# only ~58 total positives spread across correlated technical dimensions, long
# chains connecting most of the population are common, not a corner case. Fixed
# below with greedy complete-linkage: a candidate joins an existing cluster only
# if it satisfies are_duplicates() against EVERY current member of that cluster,
# not just one connecting edge.

_DEFAULT_JACCARD_THRESHOLD = 0.6
_DEFAULT_OVERLAP_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Similarity primitives
# ---------------------------------------------------------------------------


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def overlap_coefficient(a: frozenset, b: frozenset) -> float:
    """Szymkiewicz-Simpson coefficient — |A∩B| / min(|A|,|B|). Catches
    containment (a narrower pattern fully inside a broader one) that a
    Jaccard score alone would understate when the two sets are very
    different sizes."""
    smaller = min(len(a), len(b))
    if smaller == 0:
        return 0.0
    return len(a & b) / smaller


def are_duplicates(
    a: frozenset,
    b: frozenset,
    jaccard_threshold: float = _DEFAULT_JACCARD_THRESHOLD,
    overlap_threshold: float = _DEFAULT_OVERLAP_THRESHOLD,
) -> bool:
    return jaccard(a, b) >= jaccard_threshold or overlap_coefficient(a, b) >= overlap_threshold


# ---------------------------------------------------------------------------
# Clustered pattern
# ---------------------------------------------------------------------------


@dataclass
class ClusteredPattern:
    cluster_id: str
    representative: PatternCandidate
    member_count: int
    members: list = field(default_factory=list)  # list[PatternCandidate] — full traceability

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "representative": self.representative.to_dict(),
            "member_count": self.member_count,
            "members": [m.to_dict() for m in self.members],
        }


def _cluster_id(members: list[PatternCandidate]) -> str:
    """Deterministic — same member set always produces the same id, so
    re-running dedup on unchanged input is idempotent and reproducible."""
    keys = sorted(json.dumps(m.slice_definition, sort_keys=True) for m in members)
    return hashlib.sha1("|".join(keys).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def cluster_patterns(
    candidates: list[PatternCandidate],
    jaccard_threshold: float = _DEFAULT_JACCARD_THRESHOLD,
    overlap_threshold: float = _DEFAULT_OVERLAP_THRESHOLD,
) -> list[ClusteredPattern]:
    """Cluster pairwise (or any single-order) candidates by positive-example
    overlap, using greedy complete-linkage (see module docstring for why
    single-linkage/union-find was tried and rejected — it chains). Scoped by
    the caller to order-2 candidates in this Phase — see
    docs/LEARNING_AGENT_ARCHITECTURE.md for why order-1 stays untouched and
    order-3 stays out of this pipeline entirely.

    Deterministic: candidates are processed most-significant-first (lowest
    adjusted p-value), so each cluster's representative is naturally its
    first (strongest) member, and re-running on identical input always
    produces the same clusters."""
    n = len(candidates)
    if n == 0:
        return []

    order = sorted(
        range(n),
        key=lambda i: (
            candidates[i].p_value_adjusted if candidates[i].p_value_adjusted is not None else 1.0,
            -candidates[i].n_success,
        ),
    )

    clusters_idx: list[list[int]] = []
    for i in order:
        placed = False
        for cluster in clusters_idx:
            if all(
                are_duplicates(
                    candidates[i].signal_ids,
                    candidates[m].signal_ids,
                    jaccard_threshold,
                    overlap_threshold,
                )
                for m in cluster
            ):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters_idx.append([i])

    clusters: list[ClusteredPattern] = []
    for indices in clusters_idx:
        members = [candidates[i] for i in indices]
        representative = members[0]  # first = most significant, by construction (see `order` above)
        clusters.append(
            ClusteredPattern(
                cluster_id=_cluster_id(members),
                representative=representative,
                member_count=len(members),
                members=members,
            )
        )

    clusters.sort(
        key=lambda c: (
            c.representative.p_value_adjusted
            if c.representative.p_value_adjusted is not None
            else 1.0
        )
    )
    logger.info(
        f"pattern_dedup: {n} candidates -> {len(clusters)} clusters "
        f"(complete-linkage, jaccard>={jaccard_threshold} or overlap>={overlap_threshold})"
    )
    return clusters


def load_candidates_from_report(
    json_path: Path, order: int = 2, passed_only: bool = True
) -> list[PatternCandidate]:
    """Reconstruct PatternCandidate objects from a pattern_miner_{date}.json
    report — lets dedup run standalone against a committed report without
    re-running the miner."""
    data = json.loads(Path(json_path).read_text())
    rows = data.get("candidates_by_order", {}).get(str(order), [])
    candidates = [PatternCandidate.from_dict(r) for r in rows]
    if passed_only:
        candidates = [c for c in candidates if c.passed_gate]
    return candidates
