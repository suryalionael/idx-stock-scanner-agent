"""Prompt builders for AI Lab agents.

Rule enforced by construction, not just instruction text: every number and
every candidate observation the model is allowed to reference is embedded
in the prompt FROM code-computed data (Evidence, DecisionTrace,
ConfidenceBreakdown, HistoricalComparison, and the rule-based candidate
highlights from scoring.generate_evidence_highlights). The model is asked
to narrate/select/explain over given facts — never to state new ones. See
schemas.py's module docstring for how this is enforced again on the
response side (HypothesisOutput/DecisionOutput carry no numeric scoring
fields at all anymore).
"""
from __future__ import annotations

from stock_scanner.ai_lab.models import AIModelSpec
from stock_scanner.ai_lab.schemas import (
    ConfidenceBreakdown,
    DecisionTrace,
    Evidence,
    HistoricalComparison,
)

HYPOTHESIS_SYSTEM_PROMPT = """You are a quantitative research assistant for an experimental, \
non-production stock recommendation system (IDX / Indonesian stock market). You are given \
ONLY validated, code-computed evidence and a pool of pre-generated candidate observations below \
— you must not invent, estimate, or restate any fact that was not given to you. Your job is to \
write a short rationale and select/lightly rephrase (never invent new ones) the most relevant \
candidate strengths, weaknesses, and risks. Respond with ONLY a JSON object matching the \
requested schema, no other text."""

DECISION_SYSTEM_PROMPT = """You are a narration assistant for an experimental, non-production \
stock recommendation system (IDX / Indonesian stock market). All scoring, confidence, \
recommendation level, risk level, and historical-comparison verdict have ALREADY been computed \
by code and are given to you below — you cannot change them and must not contradict them. Your \
only job is to explain, in plain language, why those numbers came out the way they did. Do not \
invent statistics or restate numbers incorrectly. Respond with ONLY a JSON object matching the \
requested schema, no other text."""


def build_hypothesis_prompt(evidence: Evidence, model_spec: AIModelSpec, highlights: dict) -> str:
    indicators = "\n".join(f"  - {k}: {v}" for k, v in evidence.technical_indicators.items())
    stats = "\n".join(
        f"  - n={s.get('n')}, n_success={s.get('n_success')}, win_rate={s.get('win_rate')}, "
        f"win_rate_shrunk={s.get('win_rate_shrunk')}, ci_lower={s.get('ci_lower')}, "
        f"ci_upper={s.get('ci_upper')}, p_value_adjusted={s.get('p_value_adjusted')}"
        for s in evidence.statistical_evidence
    ) or "  (none available for this ticker yet)"
    patterns = "\n".join(f"  - {p}" for p in evidence.similar_patterns) or "  (none)"

    candidate_strengths = "\n".join(f"  - {s}" for s in highlights.get("strengths", [])) or "  (none generated)"
    candidate_weaknesses = "\n".join(f"  - {w}" for w in highlights.get("weaknesses", [])) or "  (none generated)"
    candidate_risks = "\n".join(f"  - {r}" for r in highlights.get("risks", [])) or "  (none generated)"

    return f"""Persona: {model_spec.display_name} — {model_spec.description}
{model_spec.persona_instructions}

Ticker: {evidence.ticker}

Technical indicators (code-computed, current):
{indicators}

Statistically validated pattern evidence (from Learning Agent Phase 1's gated pattern clusters,
exact matches only):
{stats}

Similar historical patterns matched (exact):
{patterns}

Closest known pattern similarity (partial match across ALL patterns, not just exact ones):
{evidence.best_pattern_similarity_pct:.1f}%

Candidate strengths (code-generated from the evidence above — choose the most relevant, you may
rephrase for fluency but must not invent new ones or add facts not listed):
{candidate_strengths}

Candidate weaknesses (same rule):
{candidate_weaknesses}

Candidate risks (same rule):
{candidate_risks}

Respond with ONLY a JSON object with exactly these keys:
  "why": one or two sentences on why this ticker is interesting given the evidence above
  "strengths": up to 5 strings, chosen/rephrased ONLY from the candidate strengths above
  "weaknesses": up to 5 strings, chosen/rephrased ONLY from the candidate weaknesses above
  "risks": up to 5 strings, chosen/rephrased ONLY from the candidate risks above

If a candidate list above is "(none generated)", return an empty list for that key — do not
invent an entry. Do not include any other fields or any numbers not present in the evidence
above."""


def build_decision_prompt(
    evidence: Evidence,
    hypothesis_why: str,
    model_spec: AIModelSpec,
    trace: DecisionTrace,
    confidence: ConfidenceBreakdown,
    comparison: HistoricalComparison,
) -> str:
    comparison_stats = (
        f"sample_size={comparison.sample_size}, win_rate={comparison.win_rate}, "
        f"ci_lower={comparison.ci_lower}, ci_upper={comparison.ci_upper}, verdict={comparison.verdict.value}"
        if comparison.sample_size is not None
        else "no validated historical pattern matches this ticker (verdict=no_data)"
    )

    return f"""Persona: {model_spec.display_name} — {model_spec.description}

Ticker: {evidence.ticker}
Prior hypothesis rationale: {hypothesis_why}

Code-computed decision trace (already final, do not alter):
  technical_score={trace.technical_score}, statistical_score={trace.statistical_score},
  pattern_similarity_score={trace.pattern_similarity_score}, risk_score={trace.risk_score},
  final_score={trace.final_score}

Code-computed confidence breakdown (already final, do not alter):
  technical={confidence.technical}, statistical={confidence.statistical},
  pattern_similarity={confidence.pattern_similarity}, risk_adjustment={confidence.risk_adjustment},
  final_confidence={confidence.final_confidence}

Code-computed historical comparison (already final, do not alter):
  {comparison_stats}

Based ONLY on the numbers above, respond with ONLY a JSON object with exactly these keys:
  "reasoning_summary": 1-3 sentences explaining why the technical indicators are bullish or
    bearish, why the historical evidence supports or contradicts them, and why the final
    score/confidence make this recommendation conservative or aggressive
  "historical_comparison_explanation": 1-2 sentences explaining the historical_comparison
    verdict above ({comparison.verdict.value}) using only the given sample_size/win_rate/CI —
    if verdict is no_data, say so plainly rather than inventing a comparison
  "confidence_explanation": 1 sentence explaining the final_confidence value above in terms of
    its technical/statistical/pattern_similarity/risk_adjustment components

Do not include any other fields, and do not state any score/confidence/rate different from the
numbers given above."""
