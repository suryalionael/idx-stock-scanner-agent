"""Tests for the async 9router client (stock_scanner/ai_lab/client.py).

No real network calls anywhere — NineRouterClient._call is monkeypatched
with AsyncMock, and one test verifies the actual HTTP request shape via a
patched httpx.AsyncClient. Async calls are driven with asyncio.run() inside
plain `def test_...` functions so no pytest-asyncio/anyio plugin config is
required.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from stock_scanner.ai_lab.client import (
    MockNineRouterClient,
    NineRouterClient,
    NineRouterConfigError,
    NineRouterResponseError,
)
from stock_scanner.ai_lab.schemas import (
    DecisionOutput,
    HypothesisOutput,
    RecommendationAction,
    RiskLevel,
)

_FAST_RETRY = (0.001, 0.005)  # keep test runtime negligible


def _client(**kwargs) -> NineRouterClient:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("model", "deepseek-v4-flash-free")
    kwargs.setdefault("base_url", "https://example.invalid/v1")
    kwargs.setdefault("retry_wait_seconds", _FAST_RETRY)
    return NineRouterClient(**kwargs)


def _valid_hypothesis_json() -> str:
    return json.dumps({
        "why": "test", "confidence": 0.8, "confidence_explanation": "test",
        "strengths": ["a"], "weaknesses": ["b"], "risks": ["c"],
    })


# ---------------------------------------------------------------------------
# Configuration — never guesses, always fails loudly
# ---------------------------------------------------------------------------

def test_missing_all_config_raises(monkeypatch):
    monkeypatch.delenv("NINEROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NINEROUTER_MODEL", raising=False)
    monkeypatch.delenv("NINEROUTER_BASE_URL", raising=False)
    with pytest.raises(NineRouterConfigError, match="NINEROUTER_API_KEY"):
        NineRouterClient()


def test_missing_base_url_only_raises(monkeypatch):
    monkeypatch.delenv("NINEROUTER_BASE_URL", raising=False)
    with pytest.raises(NineRouterConfigError, match="NINEROUTER_BASE_URL"):
        NineRouterClient(api_key="k", model="m")


def test_reads_from_env_vars(monkeypatch):
    monkeypatch.setenv("NINEROUTER_API_KEY", "env-key")
    monkeypatch.setenv("NINEROUTER_MODEL", "deepseek-v4-flash-free")
    monkeypatch.setenv("NINEROUTER_BASE_URL", "https://example.invalid/v1")
    client = NineRouterClient()
    assert client.api_key == "env-key"
    assert client.model == "deepseek-v4-flash-free"
    assert client.base_url == "https://example.invalid/v1"


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("NINEROUTER_API_KEY", "env-key")
    client = _client(api_key="explicit-key")
    assert client.api_key == "explicit-key"


# ---------------------------------------------------------------------------
# complete_structured — success, retry, and exhaustion paths
# ---------------------------------------------------------------------------

def test_complete_structured_success():
    client = _client()
    client._call = AsyncMock(return_value=_valid_hypothesis_json())
    result = asyncio.run(client.complete_structured("prompt", HypothesisOutput))
    assert isinstance(result, HypothesisOutput)
    assert result.confidence == 0.8


def test_complete_structured_retries_on_bad_json_then_succeeds():
    client = _client(max_retries=3)
    client._call = AsyncMock(side_effect=["not json", _valid_hypothesis_json()])
    result = asyncio.run(client.complete_structured("prompt", HypothesisOutput))
    assert isinstance(result, HypothesisOutput)
    assert client._call.await_count == 2


def test_complete_structured_raises_after_exhausting_retries():
    client = _client(max_retries=2)
    client._call = AsyncMock(return_value="not json at all")
    with pytest.raises(NineRouterResponseError):
        asyncio.run(client.complete_structured("prompt", HypothesisOutput))
    assert client._call.await_count == 2


def test_complete_structured_retries_on_schema_validation_failure():
    client = _client(max_retries=2)
    bad_schema = json.dumps({"why": "test"})  # missing required fields
    client._call = AsyncMock(side_effect=[bad_schema, _valid_hypothesis_json()])
    result = asyncio.run(client.complete_structured("prompt", HypothesisOutput))
    assert isinstance(result, HypothesisOutput)


def test_complete_structured_retries_on_http_error():
    client = _client(max_retries=2)
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    client._call = AsyncMock(side_effect=[httpx.ConnectError("boom", request=request), _valid_hypothesis_json()])
    result = asyncio.run(client.complete_structured("prompt", HypothesisOutput))
    assert isinstance(result, HypothesisOutput)


# ---------------------------------------------------------------------------
# _call — actual HTTP request shape (mocked transport, no real network)
# ---------------------------------------------------------------------------

def test_call_posts_openai_compatible_payload_and_auth_header():
    client = _client(api_key="secret-key", model="deepseek-v4-flash-free",
                     base_url="https://example.invalid/v1")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={
        "choices": [{"message": {"content": _valid_hypothesis_json()}}],
    })

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        content = asyncio.run(client._call("hello", system="sys"))

    assert content == _valid_hypothesis_json()
    call_kwargs = mock_async_client.post.call_args
    assert call_kwargs.args[0] == "https://example.invalid/v1/chat/completions"
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert call_kwargs.kwargs["json"]["model"] == "deepseek-v4-flash-free"
    assert call_kwargs.kwargs["json"]["messages"][0] == {"role": "system", "content": "sys"}


def test_call_raises_on_unexpected_response_shape():
    client = _client()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"unexpected": "shape"})

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        with pytest.raises(NineRouterResponseError):
            asyncio.run(client._call("hello", system=None))


# ---------------------------------------------------------------------------
# MockNineRouterClient — deterministic, no network
# ---------------------------------------------------------------------------

def test_mock_client_default_hypothesis_output():
    client = MockNineRouterClient()
    result = asyncio.run(client.complete_structured("prompt", HypothesisOutput))
    assert isinstance(result, HypothesisOutput)


def test_mock_client_default_decision_output():
    client = MockNineRouterClient()
    result = asyncio.run(client.complete_structured("prompt", DecisionOutput))
    assert isinstance(result, DecisionOutput)
    assert result.recommendation == RecommendationAction.WATCH


def test_mock_client_custom_response():
    custom = DecisionOutput(
        score=99.0, confidence=0.99, recommendation=RecommendationAction.BUY,
        expected_return=0.2, risk_level=RiskLevel.LOW, reasoning_summary="custom",
    )
    client = MockNineRouterClient(responses={"DecisionOutput": custom})
    result = asyncio.run(client.complete_structured("prompt", DecisionOutput))
    assert result is custom
