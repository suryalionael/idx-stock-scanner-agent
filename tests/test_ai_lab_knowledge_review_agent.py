"""Tests for stock_scanner/ai_lab/agents/knowledge_review_agent.py — LLM
narration over already-curated KnowledgeEntry objects. Follows
tests/test_ai_lab_hypothesis_review_agent.py's convention of exercising
the async agent via MockNineRouterClient (happy path) and a minimal
failing stub (error path)."""
import asyncio

from stock_scanner.ai_lab.agents.knowledge_review_agent import generate_knowledge_narrative
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.schemas import KnowledgeEntry, KnowledgeLifecycleStatus


class _FailingClient:
    async def complete_structured(self, prompt, response_model, system=None):
        raise NineRouterResponseError("boom")


def _entry(**overrides) -> KnowledgeEntry:
    base = dict(
        knowledge_id="k1", created_at="2026-07-15T00:00:00+00:00",
        title="sector=Technology AND rsi14=High: strong",
        description="First observed 2026-07-10T00:00:00+00:00, independently confirmed 5 time(s)...",
        conditions=[["sector", "Technology"], ["rsi14", "High"]], originating_hypotheses=["h1", "h2"],
        evidence_count=5, cumulative_sample_size=20, cumulative_successes=16, cumulative_failures=4,
        average_win_rate=0.78, shrunk_win_rate=0.7, confidence_interval=[0.6, 0.9],
        first_seen="2026-07-10T00:00:00+00:00", last_confirmed="2026-07-14T00:00:00+00:00",
        confirmation_count=5, contradiction_count=0, lifecycle_status=KnowledgeLifecycleStatus.STRONG,
        previous_lifecycle_status=KnowledgeLifecycleStatus.CONFIRMED,
    )
    base.update(overrides)
    return KnowledgeEntry(**base)


def test_generate_knowledge_narrative_returns_output_from_mock_client():
    client = MockNineRouterClient()
    result = asyncio.run(generate_knowledge_narrative(client, [_entry()]))
    assert result is not None
    assert result.overall_summary


def test_generate_knowledge_narrative_returns_none_on_empty_input_without_calling_llm():
    calls = []

    class _TrackingClient:
        async def complete_structured(self, prompt, response_model, system=None):
            calls.append(1)
            raise AssertionError("should never be called for an empty entry list")

    result = asyncio.run(generate_knowledge_narrative(_TrackingClient(), []))
    assert result is None
    assert calls == []


def test_generate_knowledge_narrative_returns_none_on_llm_failure():
    result = asyncio.run(generate_knowledge_narrative(_FailingClient(), [_entry()]))
    assert result is None


def test_prompt_only_references_given_knowledge_ids_and_shows_status_change():
    from stock_scanner.ai_lab.prompts import build_knowledge_review_prompt

    entries = [_entry(knowledge_id="k1"), _entry(knowledge_id="k2", previous_lifecycle_status=None)]
    prompt = build_knowledge_review_prompt(entries)
    assert "k1" in prompt
    assert "k2" in prompt
    assert "confirmed" in prompt  # k1's previous status shown
    assert "N/A (first time seen)" in prompt  # k2's previous status shown
    assert "never invent" in prompt.lower() or "do not invent" in prompt.lower() or "must not" in prompt.lower()


def test_custom_mock_response_via_responses_override():
    from stock_scanner.ai_lab.schemas import KnowledgeNarrativeOutput

    custom = KnowledgeNarrativeOutput(
        overall_summary="custom summary", knowledge_notes=[], organized_groups=[], highlighted_changes=[],
    )
    client = MockNineRouterClient(responses={"KnowledgeNarrativeOutput": custom})
    result = asyncio.run(generate_knowledge_narrative(client, [_entry()]))
    assert result is custom
