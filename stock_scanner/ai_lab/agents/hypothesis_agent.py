"""Hypothesis Agent — code-computed Evidence -> HypothesisOutput via 9router.

Responsibilities: analyze statistically validated patterns and, given a
code-generated pool of grounded candidate observations (see
scoring.generate_evidence_highlights), select/rephrase them into a
narrative ("why") plus strengths/weaknesses/risks. Confidence and scoring
are NOT produced here anymore — they're fully code-computed in
scoring.py; this agent is narrative-only. Reads ONLY already-validated
data (stock_scanner.db.knowledge_base rows, current technical-indicator
snapshots) — never computes statistics itself, never queries
stock_scanner/pipeline/ directly, never writes to knowledge_base (that
table belongs to Learning Agent Phase 1; this agent only reads it).
"""
from __future__ import annotations

import json

import pandas as pd
from loguru import logger

from stock_scanner.ai_lab.client import NineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.models import AIModelSpec
from stock_scanner.ai_lab.prompts import HYPOTHESIS_SYSTEM_PROMPT, build_hypothesis_prompt
from stock_scanner.ai_lab.schemas import Evidence, HypothesisOutput


def build_evidence(
    ticker: str,
    feature_row: pd.Series,
    knowledge_base_df: pd.DataFrame | None,
    model_spec: AIModelSpec,
) -> Evidence:
    """Build the code-computed Evidence for one ticker: current technical
    indicator values restricted to this AI model's focus features, plus
    any validated knowledge_base pattern clusters whose slice_definition
    exactly matches the ticker's current feature values.

    Matching is a plain equality check on each pattern's slice_definition
    keys against feature_row — the pattern-mining dimensions in this repo
    are boolean signal columns (ma_full_alignment, vol_spike, ...), so
    exact equality is the correct comparison here, not a fuzzy/threshold
    match. `statistical_evidence`/`similar_patterns` stay restricted to
    100%-exact matches (claiming a pattern applies when it doesn't fully
    hold would be misleading); `best_pattern_similarity_pct` separately
    tracks the closest PARTIAL match across every pattern, exact or not,
    for DecisionTrace.pattern_similarity_score — a distinct, continuous
    "how close to any known setup" measure.
    """
    indicators: dict[str, float | bool | None] = {}
    for f in model_spec.focus_features:
        if f not in feature_row.index:
            continue
        val = feature_row[f]
        if pd.isna(val):
            indicators[f] = None
        elif isinstance(val, (bool,)):
            indicators[f] = bool(val)
        else:
            try:
                indicators[f] = float(val)
            except (TypeError, ValueError):
                indicators[f] = None

    statistical_evidence: list[dict] = []
    similar_patterns: list[str] = []
    best_similarity_pct = 0.0
    if knowledge_base_df is not None and not knowledge_base_df.empty:
        for _, kb_row in knowledge_base_df.iterrows():
            try:
                pattern = json.loads(kb_row.get("pattern_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            rep = pattern.get("representative", {})
            slice_def = rep.get("slice_definition") or {}
            if not slice_def:
                continue

            matched = sum(1 for k, v in slice_def.items() if feature_row.get(k) == v)
            similarity_pct = matched / len(slice_def) * 100
            best_similarity_pct = max(best_similarity_pct, similarity_pct)

            if matched == len(slice_def):
                statistical_evidence.append({
                    "n": rep.get("n"), "n_success": rep.get("n_success"),
                    "win_rate": rep.get("win_rate"), "win_rate_shrunk": rep.get("win_rate_shrunk"),
                    "ci_lower": rep.get("ci_lower"), "ci_upper": rep.get("ci_upper"),
                    "p_value_adjusted": rep.get("p_value_adjusted"),
                })
                similar_patterns.append(", ".join(f"{k}={v}" for k, v in slice_def.items()))

    return Evidence(
        ticker=ticker, technical_indicators=indicators,
        statistical_evidence=statistical_evidence, similar_patterns=similar_patterns,
        best_pattern_similarity_pct=round(best_similarity_pct, 2),
    )


async def generate_hypothesis(
    client: NineRouterClient, evidence: Evidence, model_spec: AIModelSpec, highlights: dict,
) -> HypothesisOutput | None:
    """One 9router call. A failure (NineRouterResponseError, after the
    client's own retries) is logged and returns None rather than raising —
    the caller decides whether to skip this ticker or abort the batch,
    mirroring stock_scanner.learning.hypothesis_agent's per-item failure
    handling (a single bad response never aborts the whole run)."""
    prompt = build_hypothesis_prompt(evidence, model_spec, highlights)
    try:
        return await client.complete_structured(prompt, HypothesisOutput, system=HYPOTHESIS_SYSTEM_PROMPT)
    except NineRouterResponseError as e:
        logger.warning(f"hypothesis_agent: {evidence.ticker}/{model_spec.key}: {e}")
        return None
