"""Prompt builders for AI Lab agents.

Rule enforced by construction, not just instruction text: every number the
model is allowed to reason about is embedded in the prompt FROM `Evidence`
(code-computed). The model is asked to narrate/explain/judge over given
numbers — never to state new ones as fact. See schemas.py's module
docstring for how this is enforced again on the response side.
"""
from __future__ import annotations

from stock_scanner.ai_lab.models import AIModelSpec
from stock_scanner.ai_lab.schemas import Evidence

HYPOTHESIS_SYSTEM_PROMPT = """You are a quantitative research assistant for an experimental, \
non-production stock recommendation system (IDX / Indonesian stock market). You are given \
ONLY validated, code-computed evidence below — you must not invent, estimate, or restate any \
number that was not given to you. Your job is to explain, in plain language, why this evidence \
is (or is not) interesting, and to give your own qualitative confidence in how actionable it is. \
Respond with ONLY a JSON object matching the requested schema, no other text."""

DECISION_SYSTEM_PROMPT = """You are a decision-making assistant for an experimental, non-production \
stock recommendation system (IDX / Indonesian stock market). You are given a ticker's validated \
evidence and a prior qualitative hypothesis about it. Decide on a final score, confidence, \
recommendation action, expected return, and risk level. Do not invent statistics — base your \
judgment only on what is given. Respond with ONLY a JSON object matching the requested schema, \
no other text."""


def build_hypothesis_prompt(evidence: Evidence, model_spec: AIModelSpec) -> str:
    indicators = "\n".join(f"  - {k}: {v}" for k, v in evidence.technical_indicators.items())
    stats = "\n".join(
        f"  - n={s.get('n')}, n_success={s.get('n_success')}, win_rate={s.get('win_rate')}, "
        f"win_rate_shrunk={s.get('win_rate_shrunk')}, ci_lower={s.get('ci_lower')}, "
        f"p_value_adjusted={s.get('p_value_adjusted')}"
        for s in evidence.statistical_evidence
    ) or "  (none available for this ticker yet)"
    patterns = "\n".join(f"  - {p}" for p in evidence.similar_patterns) or "  (none)"

    return f"""Persona: {model_spec.display_name} — {model_spec.description}
{model_spec.persona_instructions}

Ticker: {evidence.ticker}

Technical indicators (code-computed, current):
{indicators}

Statistically validated pattern evidence (from Learning Agent Phase 1's gated pattern clusters):
{stats}

Similar historical patterns matched:
{patterns}

Respond with ONLY a JSON object with exactly these keys:
  "why": one or two sentences on why this ticker is interesting given the evidence above
  "confidence": a number from 0 to 1 — your own qualitative judgment of how actionable this looks
  "confidence_explanation": one sentence explaining the confidence number
  "strengths": a list of short strings (supporting factors from the evidence above)
  "weaknesses": a list of short strings (gaps or caveats in the evidence above)
  "risks": a list of short strings (things that could invalidate this)

Do not include any other fields or any numbers not present in the evidence above."""


def build_decision_prompt(evidence: Evidence, hypothesis_why: str, hypothesis_confidence: float,
                           model_spec: AIModelSpec) -> str:
    return f"""Persona: {model_spec.display_name} — {model_spec.description}

Ticker: {evidence.ticker}

Prior hypothesis: {hypothesis_why}
Hypothesis confidence: {hypothesis_confidence:.2f}

Technical indicators (code-computed, current):
{chr(10).join(f"  - {k}: {v}" for k, v in evidence.technical_indicators.items())}

Based on the above, respond with ONLY a JSON object with exactly these keys:
  "score": a number from 0 to 100 — overall opportunity quality
  "confidence": a number from 0 to 1
  "recommendation": one of "BUY", "WATCH", "SELL", "AVOID"
  "expected_return": expected forward return as a fraction, e.g. 0.08 for 8% (can be negative)
  "risk_level": one of "LOW", "MEDIUM", "HIGH"
  "reasoning_summary": one or two sentences summarizing the decision

Do not include any other fields."""
