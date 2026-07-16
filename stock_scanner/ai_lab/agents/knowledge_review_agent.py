"""LLM narration over already-curated KnowledgeEntry objects — see
stock_scanner.ai_lab.knowledge_base_engine for how those entries are
computed (pure code, deterministic, no LLM involved). This module's only
job is summarize/explain/organize/highlight, never decide whether a
pattern is true, change a lifecycle_status, compute confidence, or merge
entries (enforced by prompts.KNOWLEDGE_REVIEW_SYSTEM_PROMPT and
KnowledgeNarrativeOutput's all-string/list fields).

Same code-only-fallback contract as reflection_agent.py/
hypothesis_review_agent.py: the curated entries are already complete and
deterministic before this call ever happens, so a failed or skipped
narrative call degrades to `narrative: null` in the published report,
never discards the code-computed entries. See
scripts/run_knowledge_base_engine.py.
"""
from __future__ import annotations

from loguru import logger

from stock_scanner.ai_lab.client import NineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.prompts import KNOWLEDGE_REVIEW_SYSTEM_PROMPT, build_knowledge_review_prompt
from stock_scanner.ai_lab.schemas import KnowledgeEntry, KnowledgeNarrativeOutput


async def generate_knowledge_narrative(
    client: NineRouterClient, entries: list[KnowledgeEntry],
) -> KnowledgeNarrativeOutput | None:
    """One 9router call summarizing/explaining/organizing/highlighting
    changes across the given knowledge entries. Returns None immediately
    (no LLM call) if there's nothing to summarize, and returns None
    (logged, not raised) if the call itself fails after the client's own
    retries — mirrors reflection_agent.generate_reflection_narrative /
    hypothesis_review_agent.generate_hypothesis_narrative's failure
    handling."""
    if not entries:
        return None
    prompt = build_knowledge_review_prompt(entries)
    try:
        return await client.complete_structured(prompt, KnowledgeNarrativeOutput, system=KNOWLEDGE_REVIEW_SYSTEM_PROMPT)
    except NineRouterResponseError as e:
        logger.warning(f"knowledge_review_agent: narrative call failed for {len(entries)} entry(ies): {e}")
        return None
