"""Tests for stock_scanner/ai_lab/agents/hypothesis_review_agent.py — LLM
narration over already-validated/rejected Hypothesis objects. Follows
tests/test_ai_lab_reflection_agent.py's convention of exercising the async
agent via MockNineRouterClient (happy path) and a minimal failing stub
(error path)."""
import asyncio

from stock_scanner.ai_lab.agents.hypothesis_review_agent import generate_hypothesis_narrative
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.schemas import Hypothesis, HypothesisStatus


class _FailingClient:
    async def complete_structured(self, prompt, response_model, system=None):
        raise NineRouterResponseError("boom")


def _hypothesis(**overrides) -> Hypothesis:
    base = dict(
        hypothesis_id="h1", created_at="2026-07-15T00:00:00+00:00",
        description="Recommendations where sector=Technology AND rsi14=High realized a 20.0% win rate...",
        conditions=[["sector", "Technology"], ["rsi14", "High"]],
        sample_size=20, successes=4, failures=16, win_rate=0.2, shrunk_win_rate=0.3,
        wilson_lower=0.08, wilson_upper=0.42, fisher_p=0.001, bh_adjusted_p=0.002,
        status=HypothesisStatus.VALIDATED, source_reflection_ids=["r1"],
    )
    base.update(overrides)
    return Hypothesis(**base)


def test_generate_hypothesis_narrative_returns_output_from_mock_client():
    client = MockNineRouterClient()
    result = asyncio.run(generate_hypothesis_narrative(client, [_hypothesis()]))
    assert result is not None
    assert result.overall_summary


def test_generate_hypothesis_narrative_returns_none_on_empty_input_without_calling_llm():
    calls = []

    class _TrackingClient:
        async def complete_structured(self, prompt, response_model, system=None):
            calls.append(1)
            raise AssertionError("should never be called for an empty hypothesis list")

    result = asyncio.run(generate_hypothesis_narrative(_TrackingClient(), []))
    assert result is None
    assert calls == []


def test_generate_hypothesis_narrative_returns_none_on_llm_failure():
    result = asyncio.run(generate_hypothesis_narrative(_FailingClient(), [_hypothesis()]))
    assert result is None


def test_prompt_only_references_given_hypothesis_ids():
    from stock_scanner.ai_lab.prompts import build_hypothesis_review_prompt

    hyps = [_hypothesis(hypothesis_id="h1"), _hypothesis(hypothesis_id="h2", status=HypothesisStatus.REJECTED,
                                                          rejection_reason="not significant")]
    prompt = build_hypothesis_review_prompt(hyps)
    assert "h1" in prompt
    assert "h2" in prompt
    assert "never invent" in prompt.lower() or "do not invent" in prompt.lower()


def test_custom_mock_response_via_responses_override():
    from stock_scanner.ai_lab.schemas import HypothesisNarrativeOutput

    custom = HypothesisNarrativeOutput(
        overall_summary="custom summary", prioritized_hypothesis_ids=["h1"],
        hypothesis_notes=[], clusters=[],
    )
    client = MockNineRouterClient(responses={"HypothesisNarrativeOutput": custom})
    result = asyncio.run(generate_hypothesis_narrative(client, [_hypothesis()]))
    assert result is custom
