"""Tests for stock_scanner/ai_lab/agents/reflection_agent.py — LLM
narration over already-gated ReflectionObservation objects. Follows
tests/test_ai_lab_agents.py's convention of exercising the async agent via
MockNineRouterClient (happy path) and a minimal failing stub (error path)."""
import asyncio

from stock_scanner.ai_lab.agents.reflection_agent import generate_reflection_narrative
from stock_scanner.ai_lab.client import MockNineRouterClient, NineRouterResponseError
from stock_scanner.ai_lab.schemas import ObservationCategory, ReflectionObservation


class _FailingClient:
    async def complete_structured(self, prompt, response_model, system=None):
        raise NineRouterResponseError("boom")


def _observation(**overrides) -> ReflectionObservation:
    base = dict(
        observation_id="obs1", category=ObservationCategory.MODEL_PERFORMANCE,
        title="AI model 'momentum_ai': consistently succeeds",
        description="20 recommendations ... 80.0% win rate vs 50.0% baseline ...",
        supporting_statistics={"n": 20, "n_success": 16, "win_rate": 0.8, "baseline_rate": 0.5},
        affected_trade_count=20, confidence=0.95, generated_at="2026-07-15T00:00:00+00:00",
    )
    base.update(overrides)
    return ReflectionObservation(**base)


def test_generate_reflection_narrative_returns_output_from_mock_client():
    client = MockNineRouterClient()
    result = asyncio.run(generate_reflection_narrative(client, [_observation()]))
    assert result is not None
    assert result.overall_summary


def test_generate_reflection_narrative_returns_none_on_empty_observations_without_calling_llm():
    calls = []

    class _TrackingClient:
        async def complete_structured(self, prompt, response_model, system=None):
            calls.append(1)
            raise AssertionError("should never be called for an empty observation list")

    result = asyncio.run(generate_reflection_narrative(_TrackingClient(), []))
    assert result is None
    assert calls == []


def test_generate_reflection_narrative_returns_none_on_llm_failure():
    result = asyncio.run(generate_reflection_narrative(_FailingClient(), [_observation()]))
    assert result is None


def test_generate_reflection_narrative_prompt_only_references_given_ids():
    from stock_scanner.ai_lab.prompts import build_reflection_prompt

    obs = [_observation(observation_id="obs1"), _observation(observation_id="obs2", title="Sector 'Financials': consistently succeeds")]
    prompt = build_reflection_prompt(obs)
    assert "obs1" in prompt
    assert "obs2" in prompt
    assert "never invent" in prompt.lower() or "do not invent" in prompt.lower()


def test_custom_mock_response_via_responses_override():
    from stock_scanner.ai_lab.schemas import ReflectionNarrativeOutput

    custom = ReflectionNarrativeOutput(
        overall_summary="custom summary", prioritized_observation_ids=["obs1"], observation_notes=[],
    )
    client = MockNineRouterClient(responses={"ReflectionNarrativeOutput": custom})
    result = asyncio.run(generate_reflection_narrative(client, [_observation()]))
    assert result is custom
