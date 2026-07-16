"""Hypothesis Generator — pure candidate discovery over RESOLVED AI Lab
recommendations, seeded by Reflection Engine's own gated findings (see
docs/AI_LAB_ARCHITECTURE.md "Hypothesis Generator + Statistical
Validation"). Second component of the closed learning loop: Performance
Tracker -> Reflection Engine -> Hypothesis Generator (this module) ->
Statistical Validation -> (future) Knowledge Base.

Deterministic, pure Python, no LLM, no DB writes, no side effects. This
module deliberately does NOT call Wilson CI / Fisher's exact /
Benjamini-Hochberg — those are stock_scanner.ai_lab.statistical_validation's
job alone, per this component's spec-literal split. Everything here works
with raw counts (n / n_success / win_rate) only.

Apriori-seeded, not brute-force ("prune intelligently"): starting from a
blank slate and cross-producting every ai_model x sector x recommendation
x verdict x indicator value up to order 3 would be a combinatorial
explosion. Instead, every gated ReflectionObservation IS a seed —
Reflection already spent a full statistical search finding which single
dimensions (or indicator pairs) are interesting; this module's job is to
refine those by one or two more conditions, not re-discover them from
scratch. Bounded to at most 2 expansion rounds (order 1/2 seed -> order 2
-> order 3), hard-capped at max_order.

Numeric technical indicators (not just booleans) are auto-detected and
bucketed into Low/Mid/High terciles via pd.qcut — this is what makes
"RSI is High AND ATR is High" a discoverable atomic condition, a
percentile-bucketed generalization rather than a hardcoded per-indicator
threshold (matches reflection_engine.py's own no-hardcoded-feature-list
convention).
"""
from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from stock_scanner.ai_lab.reflection_engine import prepare_dataframe
from stock_scanner.ai_lab.schemas import HypothesisCandidate, ObservationCategory

_MIN_SLICE_N = 5  # mirrors reflection_engine._MIN_SLICE_N / pattern_miner._MIN_SLICE_N
_MAX_ORDER = 3
_MIN_LIFT = 0.10  # raw win-rate deviation from baseline required before an order-2
# candidate is expanded further to order-3 — a cheap PRUNING heuristic, not a
# significance claim. Real significance testing happens once, later, in
# statistical_validation.py, on every order-2 AND order-3 candidate alike; this
# heuristic only decides what's worth spending an extra combination on.
_NUM_BUCKETS = 3
_BUCKET_LABELS = ["Low", "Mid", "High"]

_CATEGORICAL_DIMENSIONS = ["ai_model", "sector", "recommendation", "historical_verdict"]


class _HypothesisSeed(NamedTuple):
    conditions: frozenset
    source_reflection_ids: list[str]


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(clean.mean()), 3) if not clean.empty else None


def _detect_indicator_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Auto-detect boolean vs. numeric keys across every row's
    technical_indicators dict (not a hardcoded feature list). Returns
    (boolean_keys, numeric_keys), both sorted for determinism."""
    bool_keys: set[str] = set()
    numeric_keys: set[str] = set()
    for indicators in df["technical_indicators"]:
        for k, v in indicators.items():
            if isinstance(v, bool):
                bool_keys.add(k)
            elif isinstance(v, (int, float)):
                numeric_keys.add(k)
    return sorted(bool_keys), sorted(numeric_keys)


def _materialize_indicator_columns(df: pd.DataFrame, boolean_keys: list[str], numeric_keys: list[str]) -> pd.DataFrame:
    """Adds one categorical `__ind_{key}` column per detected indicator:
    True/False for booleans, Low/Mid/High tercile buckets for numerics."""
    df = df.copy()
    for k in boolean_keys:
        df[f"__ind_{k}"] = df["technical_indicators"].apply(lambda d, k=k: bool(d.get(k, False)))
    for k in numeric_keys:
        raw = df["technical_indicators"].apply(lambda d, k=k: d.get(k))
        numeric = pd.to_numeric(raw, errors="coerce")
        try:
            df[f"__ind_{k}"] = pd.qcut(numeric, q=_NUM_BUCKETS, labels=_BUCKET_LABELS, duplicates="drop")
        except (ValueError, IndexError):
            continue  # not enough distinct values to bucket this run — skip
    return df


def _atomic_conditions(df: pd.DataFrame, boolean_keys: list[str], numeric_keys: list[str]) -> list[tuple[str, str]]:
    """Every (dimension, value) pair available to combine into a candidate
    hypothesis's condition set."""
    conditions: list[tuple[str, str]] = []
    for dim in _CATEGORICAL_DIMENSIONS:
        if dim not in df.columns:
            continue
        for value in df[dim].dropna().unique():
            if value:
                conditions.append((dim, str(value)))
    for k in boolean_keys:
        conditions.append((k, "True"))
    for k in numeric_keys:
        col = f"__ind_{k}"
        if col in df.columns:
            for label in df[col].dropna().unique():
                conditions.append((k, str(label)))
    return conditions


def _seed_from_reflection(observations: list[dict]) -> list[_HypothesisSeed]:
    """Parse gated ReflectionObservation rows (dicts, as loaded from
    stock_scanner.db.reflection.load_observations or the exported JSON)
    into seeds. Excludes CONFIDENCE_CALIBRATION — that category describes
    confidence buckets, not a recommendation-attribute condition."""
    seeds = []
    for obs in observations:
        category = obs.get("category")
        if category == ObservationCategory.CONFIDENCE_CALIBRATION.value:
            continue
        stats = obs.get("supporting_statistics") or {}
        obs_id = obs.get("observation_id")
        source_ids = [obs_id] if obs_id else []

        if category == ObservationCategory.TECHNICAL_PATTERN.value:
            indicators = [i for i in str(stats.get("dimension", "")).split("+") if i]
            conditions = frozenset((ind, "True") for ind in indicators)
        else:
            dim, value = stats.get("dimension"), stats.get("value")
            if not dim or value is None:
                continue
            conditions = frozenset({(dim, str(value))})

        if conditions:
            seeds.append(_HypothesisSeed(conditions, source_ids))
    return seeds


def _mask_for(df: pd.DataFrame, conditions: frozenset) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for dim, value in conditions:
        col = dim if dim in df.columns else f"__ind_{dim}"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df[col].astype(str) == value
    return mask


def _score_candidate(df: pd.DataFrame, conditions: frozenset, source_reflection_ids: list[str]) -> HypothesisCandidate | None:
    grp = df[_mask_for(df, conditions)]
    n = len(grp)
    if n < _MIN_SLICE_N:
        return None
    n_success = int(grp["label_success"].sum())
    return HypothesisCandidate(
        conditions=sorted(list(c) for c in conditions),
        order=len(conditions),
        n=n, n_success=n_success, n_failure=n - n_success,
        win_rate=n_success / n,
        avg_return_percentage=_safe_mean(grp["return_percentage"]),
        avg_holding_days=_safe_mean(grp["holding_days"]),
        source_reflection_ids=sorted(set(source_reflection_ids)),
    )


def generate_candidate_hypotheses(
    df_resolved: pd.DataFrame,
    reflection_observations: list[dict],
    max_order: int = _MAX_ORDER,
) -> list[HypothesisCandidate]:
    """Apriori-seeded candidate generation. Returns [] if there are no
    gated reflection observations to seed from (no upstream evidence yet)
    or no resolved recommendations — not an error, the correct honest
    result. Deterministic: identical (df_resolved, reflection_observations)
    always produces the same candidate set in the same sorted order."""
    if df_resolved.empty or not reflection_observations:
        return []

    df = prepare_dataframe(df_resolved)
    total_n = len(df)
    total_success = int(df["label_success"].sum())
    baseline_rate = total_success / total_n if total_n else 0.0

    boolean_keys, numeric_keys = _detect_indicator_types(df)
    df = _materialize_indicator_columns(df, boolean_keys, numeric_keys)
    universe = _atomic_conditions(df, boolean_keys, numeric_keys)

    seeds = _seed_from_reflection(reflection_observations)
    seen: set[frozenset] = {s.conditions for s in seeds}
    candidates: dict[frozenset, HypothesisCandidate] = {}

    frontier = seeds
    round_num = 0
    while frontier and round_num < max_order - 1:  # at most 2 rounds for max_order=3
        round_num += 1
        next_frontier: list[_HypothesisSeed] = []
        for conditions, source_ids in frontier:
            used_dims = {dim for dim, _ in conditions}
            for dim, value in universe:
                if dim in used_dims:
                    continue
                new_conditions = conditions | {(dim, value)}
                if new_conditions in seen or len(new_conditions) > max_order:
                    continue
                seen.add(new_conditions)

                candidate = _score_candidate(df, new_conditions, source_ids)
                if candidate is None:
                    continue
                candidates[new_conditions] = candidate

                if len(new_conditions) < max_order and abs(candidate.win_rate - baseline_rate) >= _MIN_LIFT:
                    next_frontier.append(_HypothesisSeed(new_conditions, source_ids))
        frontier = next_frontier

    return sorted(candidates.values(), key=lambda c: (c.order, c.conditions))
