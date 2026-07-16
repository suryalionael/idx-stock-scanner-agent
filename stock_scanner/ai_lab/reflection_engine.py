"""Reflection Engine — statistically gated observations over RESOLVED AI
Lab recommendations (status IN ('CLOSED','EXPIRED')). First component of
the closed learning loop (see docs/AI_LAB_ARCHITECTURE.md "Reflection
Engine"): Performance Tracker -> Reflection Engine -> (future) Calibration
Engine / Hypothesis Generator / Statistical Validation.

Reuses stock_scanner.learning.pattern_miner's pure math primitives
(wilson_ci, benjamini_hochberg, shrunk_win_rate) rather than re-deriving
them — pattern_miner.py lives under stock_scanner/learning/, not
stock_scanner/pipeline/ or stock_scanner/alerts/, so this is the same
one-way reuse AI Lab's Hypothesis Agent already does for Pattern
Miner/Statistical Validation. No LLM, no database writes, no import of
stock_scanner.ai_lab.client/agents — pure pandas + scipy, deterministic
given the same input DataFrame.

Every ReflectionObservation is one of three independently
Benjamini-Hochberg-corrected tiers (mirrors pattern_miner's per-order-tier
separation, so a noisy tier can't drown a real signal in another):

  1. Categorical dimensions (ai_model, sector, recommendation level,
     historical_comparison.verdict) — Wilson CI + Fisher's exact vs. rest
     of population, same shape as pattern_miner._score_slice.
  2. Technical indicator combinations (order <= 2) — same scorer, applied
     to boolean keys auto-detected from each row's
     reasoning.technical_indicators (not a hardcoded feature list — AI Lab
     personas expose different indicator sets than the production
     scanner).
  3. Confidence calibration — quartile buckets of stated `confidence`,
     tested against realized win rate via a one-sample binomial test
     (scipy.stats.binomtest), not a win-rate-vs-baseline slice test.

Gate (tiers 1-2): BH-adjusted q < alpha AND (Wilson CI lower bound above
baseline with enough *successes* to trust it — a success pattern; OR
Wilson CI upper bound below baseline with enough *non-successes* to trust
it — a failure pattern). Using n_success as the sample-size floor for
success patterns but (n - n_success) for failure patterns is deliberate:
a slice with hardly any wins can still have plenty of losses to reliably
show it underperforms, and gating failure patterns on n_success would
make them almost undiscoverable by construction.

label_success = 1 if trade_outcome == "WIN" else 0 — LOSS and BREAKEVEN
both count as non-success (standard win-rate framing, matches
stock_scanner.pipeline.evaluator's hit_tp being binary too).
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import binomtest, fisher_exact

from stock_scanner.ai_lab.schemas import ObservationCategory, ReflectionObservation
from stock_scanner.learning.pattern_miner import benjamini_hochberg, shrunk_win_rate, wilson_ci
from stock_scanner.reference.issuers import get_sector

_MIN_SLICE_N = 5  # skip trivially tiny slices before spending a stats call — mirrors pattern_miner._MIN_SLICE_N
_DEFAULT_MIN_N_SUCCESS = 3
_DEFAULT_ALPHA = 0.05
_MAX_TECHNICAL_ORDER = 2

_CATEGORICAL_DIMENSIONS = ["ai_model", "sector", "recommendation", "historical_verdict"]
_CATEGORY_BY_DIMENSION = {
    "ai_model": ObservationCategory.MODEL_PERFORMANCE,
    "sector": ObservationCategory.SECTOR_PERFORMANCE,
    "recommendation": ObservationCategory.RECOMMENDATION_LEVEL_PERFORMANCE,
    "historical_verdict": ObservationCategory.HISTORICAL_VERDICT_ACCURACY,
}
_DIMENSION_LABELS = {
    "ai_model": "AI model",
    "sector": "Sector",
    "recommendation": "Recommendation level",
    "historical_verdict": "Historical verdict",
}


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(clean.mean()), 3) if not clean.empty else None


def score_group(n_success: int, n: int, baseline_rate: float, total_n: int, total_success: int) -> dict:
    """Wilson CI + Fisher's exact (this group vs. rest of population) +
    shrunk win rate — the shared primitive tiers 1-2 build on."""
    win_rate = n_success / n if n else 0.0
    ci_lower, ci_upper = wilson_ci(n_success, n)
    win_rate_shrunk = shrunk_win_rate(n_success, n, baseline_rate)

    rest_n = total_n - n
    rest_success = total_success - n_success
    table = [[n_success, n - n_success], [rest_success, max(rest_n - rest_success, 0)]]
    _, p_value = fisher_exact(table, alternative="two-sided")

    return {
        "n": n,
        "n_success": n_success,
        "win_rate": win_rate,
        "win_rate_shrunk": win_rate_shrunk,
        "baseline_rate": baseline_rate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(p_value),
    }


def passes_slice_gate(c: dict, min_n_success: int) -> bool:
    success_pattern = c["ci_lower"] > c["baseline_rate"] and c["n_success"] >= min_n_success
    failure_pattern = c["ci_upper"] < c["baseline_rate"] and (c["n"] - c["n_success"]) >= min_n_success
    return success_pattern or failure_pattern


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_success"] = (df["trade_outcome"] == "WIN").astype(int)
    df["sector"] = df["ticker"].apply(get_sector)

    def _parse_json_col(col: str) -> pd.Series:
        return df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else (v or {}))

    reasoning = _parse_json_col("reasoning")
    historical = _parse_json_col("historical_comparison")
    df["technical_indicators"] = reasoning.apply(lambda r: r.get("technical_indicators") or {})
    df["historical_verdict"] = historical.apply(lambda r: r.get("verdict"))
    return df


# ---------------------------------------------------------------------------
# Tier 1 — categorical dimensions
# ---------------------------------------------------------------------------


def _score_categorical_dimensions(df: pd.DataFrame, min_n_success: int, alpha: float) -> list[dict]:
    total_n = len(df)
    total_success = int(df["label_success"].sum())
    baseline_rate = total_success / total_n if total_n else 0.0

    candidates = []
    for dim in _CATEGORICAL_DIMENSIONS:
        if dim not in df.columns:
            continue
        for value, grp in df.groupby(dim, dropna=True):
            if not value or len(grp) < _MIN_SLICE_N:
                continue
            n = len(grp)
            n_success = int(grp["label_success"].sum())
            stats = score_group(n_success, n, baseline_rate, total_n, total_success)
            candidates.append(
                {
                    "dimension": dim,
                    "value": str(value),
                    "category": _CATEGORY_BY_DIMENSION[dim],
                    "avg_return_percentage": _safe_mean(grp["return_percentage"]),
                    "avg_holding_days": _safe_mean(grp["holding_days"]),
                    **stats,
                }
            )

    q_values = benjamini_hochberg([c["p_value"] for c in candidates])
    gated = []
    for c, q in zip(candidates, q_values):
        c["p_value_adjusted"] = q
        if q < alpha and passes_slice_gate(c, min_n_success):
            gated.append(c)
    return gated


# ---------------------------------------------------------------------------
# Tier 2 — technical indicator combinations
# ---------------------------------------------------------------------------


def _detect_boolean_indicators(df: pd.DataFrame) -> list[str]:
    keys: set[str] = set()
    for indicators in df["technical_indicators"]:
        for k, v in indicators.items():
            if isinstance(v, bool):
                keys.add(k)
    return sorted(keys)


def _score_technical_indicators(df: pd.DataFrame, min_n_success: int, alpha: float) -> list[dict]:
    total_n = len(df)
    total_success = int(df["label_success"].sum())
    baseline_rate = total_success / total_n if total_n else 0.0

    keys = _detect_boolean_indicators(df)
    if not keys:
        return []

    df = df.copy()
    for k in keys:
        df[f"__ind_{k}"] = df["technical_indicators"].apply(lambda d, k=k: bool(d.get(k, False)))

    candidates = []
    for order in range(1, _MAX_TECHNICAL_ORDER + 1):
        for combo in itertools.combinations(keys, order):
            mask = pd.Series(True, index=df.index)
            for k in combo:
                mask &= df[f"__ind_{k}"]
            grp = df[mask]
            n = len(grp)
            if n < _MIN_SLICE_N:
                continue
            n_success = int(grp["label_success"].sum())
            stats = score_group(n_success, n, baseline_rate, total_n, total_success)
            candidates.append(
                {
                    "dimension": "+".join(combo),
                    "value": "True",
                    "category": ObservationCategory.TECHNICAL_PATTERN,
                    "avg_return_percentage": _safe_mean(grp["return_percentage"]),
                    "avg_holding_days": _safe_mean(grp["holding_days"]),
                    **stats,
                }
            )

    q_values = benjamini_hochberg([c["p_value"] for c in candidates])
    gated = []
    for c, q in zip(candidates, q_values):
        c["p_value_adjusted"] = q
        if q < alpha and passes_slice_gate(c, min_n_success):
            gated.append(c)
    return gated


# ---------------------------------------------------------------------------
# Tier 3 — confidence calibration
# ---------------------------------------------------------------------------


def _score_confidence_calibration(df: pd.DataFrame, min_n_success: int, alpha: float, n_buckets: int = 4) -> list[dict]:
    total_n = len(df)
    total_success = int(df["label_success"].sum())
    baseline_rate = total_success / total_n if total_n else 0.0

    if df.empty or df["confidence"].nunique() < 2:
        return []

    working = df.copy()
    try:
        working["__confidence_bucket"] = pd.qcut(working["confidence"], q=n_buckets, duplicates="drop")
    except ValueError:
        return []

    candidates = []
    for bucket, grp in working.groupby("__confidence_bucket", observed=True):
        n = len(grp)
        if n < _MIN_SLICE_N:
            continue
        n_success = int(grp["label_success"].sum())
        win_rate = n_success / n
        avg_confidence = float(grp["confidence"].mean())
        ci_lower, ci_upper = wilson_ci(n_success, n)
        win_rate_shrunk = shrunk_win_rate(n_success, n, baseline_rate)
        p_value = float(binomtest(n_success, n, avg_confidence, alternative="two-sided").pvalue)
        candidates.append(
            {
                "dimension": "confidence_bucket",
                "value": str(bucket),
                "category": ObservationCategory.CONFIDENCE_CALIBRATION,
                "n": n,
                "n_success": n_success,
                "win_rate": win_rate,
                "win_rate_shrunk": win_rate_shrunk,
                "baseline_rate": baseline_rate,
                "avg_confidence": avg_confidence,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "avg_return_percentage": _safe_mean(grp["return_percentage"]),
                "avg_holding_days": _safe_mean(grp["holding_days"]),
            }
        )

    q_values = benjamini_hochberg([c["p_value"] for c in candidates])
    gated = []
    for c, q in zip(candidates, q_values):
        c["p_value_adjusted"] = q
        overconfident = c["avg_confidence"] > c["ci_upper"]
        underconfident = c["avg_confidence"] < c["ci_lower"]
        if q < alpha and c["n"] >= min_n_success and (overconfident or underconfident):
            c["calibration_issue"] = "overconfident" if overconfident else "underconfident"
            gated.append(c)
    return gated


# ---------------------------------------------------------------------------
# Title / description — code-only, factual, every number sourced from `c`
# ---------------------------------------------------------------------------


def _build_title_description(c: dict) -> tuple[str, str]:
    category = c["category"]
    n, n_success = c["n"], c["n_success"]
    win_rate, baseline_rate = c["win_rate"], c["baseline_rate"]
    ci_lower, ci_upper = c["ci_lower"], c["ci_upper"]
    p_adj = c["p_value_adjusted"]
    avg_ret, avg_hold = c["avg_return_percentage"], c["avg_holding_days"]

    if category == ObservationCategory.CONFIDENCE_CALIBRATION:
        issue = c["calibration_issue"]
        title = f"Confidence bucket {c['value']}: {issue}"
        description = (
            f"{n} resolved recommendations with stated confidence averaging "
            f"{c['avg_confidence']:.1%} realized a {win_rate:.1%} win rate "
            f"({n_success}/{n} wins, 95% Wilson CI {ci_lower:.1%}-{ci_upper:.1%}) — "
            f"stated confidence is {issue} relative to what actually happened "
            f"(one-sample binomial test, BH-adjusted p={p_adj:.4f}). "
            f"Average return {avg_ret}%, average holding period {avg_hold} trading days."
        )
        return title, description

    direction = "consistently succeeds" if win_rate > baseline_rate else "consistently fails"
    dim_label = _DIMENSION_LABELS.get(c["dimension"], "Technical pattern")
    if category == ObservationCategory.TECHNICAL_PATTERN:
        title = f"Technical pattern '{c['dimension']}': {direction}"
        subject = f"recommendations where {c['dimension']} all held true"
    else:
        title = f"{dim_label} '{c['value']}': {direction}"
        subject = f"recommendations where {dim_label.lower()} = '{c['value']}'"

    description = (
        f"{n} {subject} realized a {win_rate:.1%} win rate ({n_success}/{n} wins, "
        f"shrunk estimate {c['win_rate_shrunk']:.1%}) vs a {baseline_rate:.1%} baseline "
        f"across all resolved recommendations (95% Wilson CI {ci_lower:.1%}-{ci_upper:.1%}, "
        f"Fisher's exact BH-adjusted p={p_adj:.4f}). Average return {avg_ret}%, "
        f"average holding period {avg_hold} trading days."
    )
    return title, description


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_observations(
    df_resolved: pd.DataFrame,
    min_n_success: int = _DEFAULT_MIN_N_SUCCESS,
    alpha: float = _DEFAULT_ALPHA,
) -> list[ReflectionObservation]:
    """Score all three tiers over already-resolved ai_recommendations rows
    (caller filters to status IN ('CLOSED','EXPIRED') before calling — see
    scripts/run_reflection_engine.py) and return the gated survivors as
    ReflectionObservation objects. Returns [] on insufficient data — not
    an error; an empty report is the correct, honest result when there
    aren't enough resolved trades yet."""
    if df_resolved.empty:
        return []

    df = prepare_dataframe(df_resolved)
    generated_at = datetime.now(timezone.utc).isoformat()

    raw: list[dict] = []
    raw += _score_categorical_dimensions(df, min_n_success, alpha)
    raw += _score_technical_indicators(df, min_n_success, alpha)
    raw += _score_confidence_calibration(df, min_n_success, alpha)

    observations = []
    for c in raw:
        obs_id_src = f"{c['category'].value}|{c['dimension']}|{c['value']}|{generated_at}"
        observation_id = hashlib.sha1(obs_id_src.encode()).hexdigest()[:16]
        title, description = _build_title_description(c)
        confidence = max(0.0, min(1.0, 1 - c["p_value_adjusted"]))
        # Keep `dimension`/`value` in supporting_statistics (not just baked
        # into the title string) so dashboard code can group/chart by them
        # directly instead of parsing title text.
        stats = {k: v for k, v in c.items() if k != "category"}
        observations.append(
            ReflectionObservation(
                observation_id=observation_id,
                category=c["category"],
                title=title,
                description=description,
                supporting_statistics=stats,
                affected_trade_count=c["n"],
                confidence=confidence,
                generated_at=generated_at,
            )
        )
    return observations
