"""Decision Agent — Evidence + HypothesisOutput -> DecisionOutput -> AIRecommendation.

Responsibilities: evaluate the hypothesis, produce a final score/confidence/
recommendation/expected-return/risk-level judgment, and assemble the
storable AIRecommendation row. Does NOT edit production code, does NOT
write to signal_engine.py/scanner_config.yaml, and does NOT decide
promotion — see docs/AI_LAB_ARCHITECTURE.md's "Future Ready" section for
the (unimplemented) Auto Promotion Engine this schema is designed to
support later.
"""
from __future__ import annotations

import hashlib

from loguru import logger

from stock_scanner.ai_lab.client import NineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.models import AIModelSpec
from stock_scanner.ai_lab.prompts import DECISION_SYSTEM_PROMPT, build_decision_prompt
from stock_scanner.ai_lab.schemas import (
    AIRecommendation,
    DecisionOutput,
    Evidence,
    HypothesisOutput,
    RecommendationStatus,
)


def recommendation_id(ticker: str, ai_model: str, generated_date: str) -> str:
    """Deterministic — same (ticker, ai_model, day) always yields the same
    id, so re-running the pipeline the same day UPSERTs in place rather
    than appending duplicate rows (see stock_scanner.db.ai_lab.upsert_recommendations)."""
    raw = f"{ticker}|{ai_model}|{generated_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


async def generate_decision(
    client: NineRouterClient, evidence: Evidence, hypothesis: HypothesisOutput, model_spec: AIModelSpec,
) -> DecisionOutput | None:
    """One 9router call. A failure is logged and returns None — mirrors
    hypothesis_agent.generate_hypothesis's per-item failure handling."""
    prompt = build_decision_prompt(evidence, hypothesis.why, hypothesis.confidence, model_spec)
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
) -> AIRecommendation:
    """Merge code-computed Evidence + LLM HypothesisOutput + LLM
    DecisionOutput into one storable row. `reasoning` embeds the evidence
    verbatim (never re-derived from the LLM's own words) alongside the
    LLM's narrative fields — see schemas.py's Evidence docstring."""
    reasoning = {
        "why": hypothesis.why,
        "technical_indicators": evidence.technical_indicators,
        "statistical_evidence": evidence.statistical_evidence,
        "similar_patterns": evidence.similar_patterns,
        "confidence_explanation": hypothesis.confidence_explanation,
        "strengths": hypothesis.strengths,
        "weaknesses": hypothesis.weaknesses,
        "risks": hypothesis.risks,
        "reasoning_summary": decision.reasoning_summary,
    }
    return AIRecommendation(
        id=recommendation_id(evidence.ticker, model_spec.key, generated_date),
        ticker=evidence.ticker,
        ai_model=model_spec.key,
        score=decision.score,
        confidence=decision.confidence,
        recommendation=decision.recommendation,
        reasoning=reasoning,
        expected_return=decision.expected_return,
        risk_level=decision.risk_level,
        generated_date=generated_date,
        status=RecommendationStatus.PENDING,
        model=ninerouter_model,
    )
