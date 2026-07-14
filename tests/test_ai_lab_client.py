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
from stock_scanner.ai_lab.schemas import DecisionOutput, HypothesisOutput

_FAST_RETRY = (0.001, 0.005)  # keep test runtime negligible


def _client(**kwargs) -> NineRouterClient:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("model", "deepseek-v4-flash-free")
    kwargs.setdefault("base_url", "https://example.invalid/v1")
    kwargs.setdefault("retry_wait_seconds", _FAST_RETRY)
    return NineRouterClient(**kwargs)


def _valid_hypothesis_json() -> str:
    return json.dumps({
        "why": "test", "strengths": ["a"], "weaknesses": ["b"], "risks": ["c"],
    })


# ---------------------------------------------------------------------------
# Configuration — never guesses, always fails loudly
# ---------------------------------------------------------------------------

def _disable_real_dotenv(monkeypatch):
    # These two tests assert "missing config raises" — a real .env at the
    # repo root (gitignored, developer-local) must never silently
    # repopulate a var this test just deleted. load_dotenv() itself is
    # tested separately below (test_init_calls_load_dotenv,
    # test_env_file_values_are_loaded_via_dotenv).
    monkeypatch.setattr("stock_scanner.ai_lab.client.load_dotenv", lambda *a, **k: False)


def test_missing_all_config_raises(monkeypatch):
    _disable_real_dotenv(monkeypatch)
    monkeypatch.delenv("NINEROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NINEROUTER_MODEL", raising=False)
    monkeypatch.delenv("NINEROUTER_BASE_URL", raising=False)
    with pytest.raises(NineRouterConfigError, match="NINEROUTER_API_KEY"):
        NineRouterClient()


def test_missing_base_url_only_raises(monkeypatch):
    _disable_real_dotenv(monkeypatch)
    monkeypatch.delenv("NINEROUTER_BASE_URL", raising=False)
    with pytest.raises(NineRouterConfigError, match="NINEROUTER_BASE_URL"):
        NineRouterClient(api_key="k", model="m")


def test_init_calls_load_dotenv(monkeypatch):
    mock_load_dotenv = MagicMock(return_value=False)
    monkeypatch.setattr("stock_scanner.ai_lab.client.load_dotenv", mock_load_dotenv)
    NineRouterClient(api_key="k", model="m", base_url="https://example.invalid/v1")
    mock_load_dotenv.assert_called_once()


def _write_temp_dotenv(tmp_path, monkeypatch, contents: str) -> None:
    # python-dotenv's load_dotenv()/find_dotenv() with no explicit path
    # resolve relative to the CALLER'S FILE location via frame
    # introspection (default usecwd=False) — NOT the process cwd. Since
    # client.py's file location is fixed inside this repo, a plain
    # load_dotenv() call would always find the real repo-root .env
    # regardless of monkeypatch.chdir(). To actually isolate the test, we
    # replace stock_scanner.ai_lab.client.load_dotenv with a thin wrapper
    # around the REAL dotenv.load_dotenv(), pinned to an explicit temp
    # path — this still exercises real python-dotenv parsing/env-setting,
    # just without depending on frame-based auto-discovery.
    from dotenv import load_dotenv as real_load_dotenv

    env_path = tmp_path / ".env"
    env_path.write_text(contents)
    monkeypatch.setattr(
        "stock_scanner.ai_lab.client.load_dotenv",
        lambda *a, **k: real_load_dotenv(dotenv_path=env_path),
    )


def test_env_file_values_are_loaded_via_dotenv(tmp_path, monkeypatch):
    # Proves .env values actually reach the client — not just that
    # load_dotenv() gets called (see test_init_calls_load_dotenv above).
    monkeypatch.delenv("NINEROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NINEROUTER_MODEL", raising=False)
    monkeypatch.delenv("NINEROUTER_BASE_URL", raising=False)
    _write_temp_dotenv(
        tmp_path, monkeypatch,
        "NINEROUTER_API_KEY=from-dotenv-key\n"
        "NINEROUTER_MODEL=oc/deepseek-v4-flash-free\n"
        "NINEROUTER_BASE_URL=https://from-dotenv.invalid/v1\n",
    )

    client = NineRouterClient()
    assert client.api_key == "from-dotenv-key"
    assert client.model == "oc/deepseek-v4-flash-free"
    assert client.base_url == "https://from-dotenv.invalid/v1"


def test_process_env_wins_over_dotenv_file(tmp_path, monkeypatch):
    # load_dotenv()'s default override=False must hold: a real env var
    # (e.g. injected by CI) always wins over a stray local .env.
    monkeypatch.setenv("NINEROUTER_API_KEY", "real-process-env-key")
    monkeypatch.delenv("NINEROUTER_MODEL", raising=False)
    monkeypatch.delenv("NINEROUTER_BASE_URL", raising=False)
    _write_temp_dotenv(
        tmp_path, monkeypatch,
        "NINEROUTER_API_KEY=from-dotenv-key\n"
        "NINEROUTER_MODEL=oc/deepseek-v4-flash-free\n"
        "NINEROUTER_BASE_URL=https://from-dotenv.invalid/v1\n",
    )

    client = NineRouterClient()
    assert client.api_key == "real-process-env-key"   # NOT overridden by .env
    assert client.model == "oc/deepseek-v4-flash-free"  # .env fills in what's missing


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
    assert result.why == "test"


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
    mock_response.text = json.dumps({
        "choices": [{"message": {"content": _valid_hypothesis_json()}}],
    })

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        content = asyncio.run(client._call("hello", HypothesisOutput, system="sys"))

    assert content == _valid_hypothesis_json()
    call_kwargs = mock_async_client.post.call_args
    assert call_kwargs.args[0] == "https://example.invalid/v1/chat/completions"
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert call_kwargs.kwargs["json"]["model"] == "deepseek-v4-flash-free"
    assert call_kwargs.kwargs["json"]["messages"][0] == {"role": "system", "content": "sys"}
    response_format = call_kwargs.kwargs["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "HypothesisOutput"
    assert response_format["json_schema"]["schema"] == HypothesisOutput.model_json_schema()


def test_call_raises_on_unexpected_response_shape():
    client = _client()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = json.dumps({"unexpected": "shape"})

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        with pytest.raises(NineRouterResponseError):
            asyncio.run(client._call("hello", HypothesisOutput, system=None))


def test_call_raises_on_completely_invalid_json_body():
    client = _client()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = "not json at all, not even close"

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        with pytest.raises(NineRouterResponseError):
            asyncio.run(client._call("hello", HypothesisOutput, system=None))


def test_call_ignores_trailing_extra_data_after_valid_json():
    # Regression test: verified live against the 9router endpoint that a
    # long/complex response body can be a complete, valid JSON object
    # followed by trailing extra bytes. Must not crash with an uncaught
    # json.JSONDecodeError ("Extra data") — the first complete object is
    # the correct response and trailing bytes are ignored.
    client = _client()
    valid_body = json.dumps({"choices": [{"message": {"content": _valid_hypothesis_json()}}]})
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = valid_body + '{"duplicate": "trailing garbage"}'

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        content = asyncio.run(client._call("hello", HypothesisOutput, system=None))

    assert content == _valid_hypothesis_json()


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
    assert result.reasoning_summary


def test_mock_client_custom_response():
    custom = DecisionOutput(
        reasoning_summary="custom", historical_comparison_explanation="custom",
        confidence_explanation="custom",
    )
    client = MockNineRouterClient(responses={"DecisionOutput": custom})
    result = asyncio.run(client.complete_structured("prompt", DecisionOutput))
    assert result is custom
