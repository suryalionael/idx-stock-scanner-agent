"""LLM narration over already-gated ReflectionObservation objects — see
stock_scanner.ai_lab.reflection_engine for how those observations are
computed (pure code, no LLM involved). This module's only job is
summarize/explain/prioritize, never invent a new observation or number
(enforced by prompts.REFLECTION_SYSTEM_PROMPT and
ReflectionNarrativeOutput's all-string/list fields).

Deliberately does not follow scripts/run_ai_lab.py's "drop the whole item
if the LLM call fails" precedent: the report's observations are already
complete and deterministic before this call ever happens, so a failed or
skipped narrative call should degrade to `narrative: null` in the
published report, never discard the (real, code-computed) observations.
See scripts/run_reflection_engine.py for how that fallback is applied.
"""
from __future__ import annotations

from loguru import logger

from stock_scanner.ai_lab.client import NineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.prompts import REFLECTION_SYSTEM_PROMPT, build_reflection_prompt
from stock_scanner.ai_lab.schemas import ReflectionNarrativeOutput, ReflectionObservation


async def generate_reflection_narrative(
    client: NineRouterClient, observations: list[ReflectionObservation],
) -> ReflectionNarrativeOutput | None:
    """One 9router call summarizing/prioritizing the given observations.
    Returns None immediately (no LLM call) if there's nothing to
    summarize, and returns None (logged, not raised) if the call itself
    fails after the client's own retries — mirrors
    hypothesis_agent.generate_hypothesis/decision_agent.generate_decision's
    per-item failure handling."""
    if not observations:
        return None
    prompt = build_reflection_prompt(observations)
    try:
        return await client.complete_structured(prompt, ReflectionNarrativeOutput, system=REFLECTION_SYSTEM_PROMPT)
    except NineRouterResponseError as e:
        logger.warning(f"reflection_agent: narrative call failed for {len(observations)} observation(s): {e}")
        return None
