"""Decision Agent — narrates already-computed scoring into the final
storable AIRecommendation.

Responsibilities: explain (not decide) the score, confidence, recommendation
level, risk level, and historical comparison — all of which are computed
deterministically in stock_scanner.ai_lab.scoring BEFORE this agent's LLM
call happens. Does NOT edit production code, does NOT write to
signal_engine.py/scanner_config.yaml, and does NOT decide promotion — see
docs/AI_LAB_ARCHITECTURE.md's "Future Ready" section for the (unimplemented)
Auto Promotion Engine this schema is designed to support later.
"""
from __future__ import annotations

import hashlib

from loguru import logger

from stock_scanner.ai_lab.client import NineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.models import AIModelSpec
from stock_scanner.ai_lab.prompts import DECISION_SYSTEM_PROMPT, build_decision_prompt
from stock_scanner.ai_lab.schemas import (
    AIRecommendation,
    ConfidenceBreakdown,
    DecisionOutput,
    DecisionTrace,
    Evidence,
    HistoricalComparison,
    HypothesisOutput,
    RecommendationStatus,
)
from stock_scanner.ai_lab.scoring import (
    classify_recommendation_level,
    classify_risk_level,
    compute_expected_return,
)


def recommendation_id(ticker: str, ai_model: str, generated_date: str) -> str:
    """Deterministic — same (ticker, ai_model, day) always yields the same
    id, so re-running the pipeline the same day UPSERTs in place rather
    than appending duplicate rows (see stock_scanner.db.ai_lab.upsert_recommendations)."""
    raw = f"{ticker}|{ai_model}|{generated_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


async def generate_decision(
    client: NineRouterClient,
    evidence: Evidence,
    hypothesis: HypothesisOutput,
    model_spec: AIModelSpec,
    trace: DecisionTrace,
    confidence: ConfidenceBreakdown,
    comparison: HistoricalComparison,
) -> DecisionOutput | None:
    """One 9router call, purely narrative — the trace/confidence/comparison
    arguments are already final by this point. A failure is logged and
    returns None — mirrors hypothesis_agent.generate_hypothesis's per-item
    failure handling."""
    prompt = build_decision_prompt(evidence, hypothesis.why, model_spec, trace, confidence, comparison)
    try:
        return await client.complete_structured(prompt, DecisionOutput, system=DECISION_SYSTEM_PROMPT)
    except NineRouterResponseError as e:
        logger.warning(f"decision_agent: {evidence.ticker}/{model_spec.key}: {e}")
        return None


def assemble_recommendation(
    evidence: Evidence,
    hypothesis: HypothesisOutput,
    decision: DecisionOutput,
    model_spec: AIModelSpec,
    generated_date: str,
    ninerouter_model: str,
    trace: DecisionTrace,
    confidence: ConfidenceBreakdown,
    comparison: HistoricalComparison,
    highlights: dict,
) -> AIRecommendation:
    """Merge code-computed DecisionTrace/ConfidenceBreakdown/
    HistoricalComparison with the LLM's narrative fields into one storable
    row. strengths/weaknesses/risks fall back to the code-generated
    candidate highlights if the LLM returned an empty list for any of them
    — guarantees "richer" observations even when the model under-delivers,
    without ever inventing a fact the code didn't already generate.
    recommendation_level/risk_level/expected_return are 100% code-computed
    here, never taken from the LLM (see scoring.py)."""
    reasoning = {
        "why": hypothesis.why,
        "technical_indicators": evidence.technical_indicators,
        "statistical_evidence": evidence.statistical_evidence,
        "similar_patterns": evidence.similar_patterns,
        "best_pattern_similarity_pct": evidence.best_pattern_similarity_pct,
        "strengths": hypothesis.strengths or highlights.get("strengths", []),
        "weaknesses": hypothesis.weaknesses or highlights.get("weaknesses", []),
        "risks": hypothesis.risks or highlights.get("risks", []),
        "reasoning_summary": decision.reasoning_summary,
        "historical_comparison_explanation": decision.historical_comparison_explanation,
        "confidence_explanation": decision.confidence_explanation,
    }
    comparison_with_explanation = comparison.model_copy(update={"explanation": decision.historical_comparison_explanation})

    return AIRecommendation(
        id=recommendation_id(evidence.ticker, model_spec.key, generated_date),
        ticker=evidence.ticker,
        ai_model=model_spec.key,
        score=trace.final_score,
        confidence=confidence.final_confidence,
        recommendation=classify_recommendation_level(trace, confidence),
        reasoning=reasoning,
        decision_trace=trace,
        confidence_breakdown=confidence,
        historical_comparison=comparison_with_explanation,
        expected_return=compute_expected_return(evidence),
        risk_level=classify_risk_level(trace),
        generated_date=generated_date,
        status=RecommendationStatus.PENDING,
        model=ninerouter_model,
    )
