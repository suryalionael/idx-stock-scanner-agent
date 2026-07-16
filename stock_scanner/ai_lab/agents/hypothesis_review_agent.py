"""LLM narration over already-validated/rejected Hypothesis objects — see
stock_scanner.ai_lab.hypothesis_engine (candidate generation) and
stock_scanner.ai_lab.statistical_validation (validation) for how those
Hypothesis objects are computed (pure code, no LLM involved). This
module's only job is summarize/explain/prioritize/cluster, never decide
whether a hypothesis is true or invent a new one (enforced by
prompts.HYPOTHESIS_REVIEW_SYSTEM_PROMPT and HypothesisNarrativeOutput's
all-string/list fields).

Named hypothesis_review_agent.py, not hypothesis_agent.py — that path is
already taken by stock_scanner.ai_lab.agents.hypothesis_agent, the
unrelated per-recommendation narrator used by the Decision pipeline.

Same code-only-fallback contract as reflection_agent.py: the statistical
output (Hypothesis rows) is already complete and deterministic before
this call ever happens, so a failed or skipped narrative call degrades to
`narrative: null` in the published report, never discards the
code-computed hypotheses. See scripts/run_hypothesis_engine.py.
"""
from __future__ import annotations

from loguru import logger

from stock_scanner.ai_lab.client import NineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.prompts import HYPOTHESIS_REVIEW_SYSTEM_PROMPT, build_hypothesis_review_prompt
from stock_scanner.ai_lab.schemas import Hypothesis, HypothesisNarrativeOutput


async def generate_hypothesis_narrative(
    client: NineRouterClient, hypotheses: list[Hypothesis],
) -> HypothesisNarrativeOutput | None:
    """One 9router call summarizing/prioritizing/clustering the given
    hypotheses. Returns None immediately (no LLM call) if there's nothing
    to summarize, and returns None (logged, not raised) if the call
    itself fails after the client's own retries — mirrors
    reflection_agent.generate_reflection_narrative's failure handling."""
    if not hypotheses:
        return None
    prompt = build_hypothesis_review_prompt(hypotheses)
    try:
        return await client.complete_structured(prompt, HypothesisNarrativeOutput, system=HYPOTHESIS_REVIEW_SYSTEM_PROMPT)
    except NineRouterResponseError as e:
        logger.warning(f"hypothesis_review_agent: narrative call failed for {len(hypotheses)} hypothesis(es): {e}")
        return None
