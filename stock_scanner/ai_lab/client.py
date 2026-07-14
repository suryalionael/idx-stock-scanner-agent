"""Async client for the 9router LLM API.

9router is a real product decision already recorded in this repo (see
stock_scanner/learning/hypothesis_agent.py's NineRouterClient stub and
docs/LEARNING_AGENT_ARCHITECTURE.md, "Provider: 9router"). The endpoint is
now confirmed: an OpenAI-compatible `/chat/completions` route under the
`/v1` base path (e.g. NINEROUTER_BASE_URL=https://<tunnel-host>/v1 ->
POST https://<tunnel-host>/v1/chat/completions). Still, every piece of
configuration — base URL, model, API key — is read from environment
variables with no hardcoded fallback: a tunnel host is expected to change
between environments/sessions, so hardcoding any one value here would
silently break the moment the tunnel is re-issued. If any of
NINEROUTER_API_KEY/_MODEL/_BASE_URL is unset this client raises
immediately and loudly rather than guessing, the same fail-fast contract
as the existing Learning Agent stub.

`.env` (repo root, gitignored — see .gitignore's `.env`/`*.env` entries) is
the single source of truth for local runs: `NineRouterClient.__init__`
calls `load_dotenv()` before reading any NINEROUTER_* variable, so
`scripts/run_ai_lab.py` (or any other caller) never needs to remember to
`source .env` first. `load_dotenv()` never overrides a variable that's
already set in the process environment (its default `override=False`), so
real CI secrets (e.g. injected via GitHub Actions) still win over a stray
local `.env` if both happen to be present.

Never parses free-form text: every call must produce a JSON object that
validates against a Pydantic schema (see schemas.py), or the call is
retried and ultimately raised as NineRouterResponseError — there is no
fallback path that accepts an unvalidated response.
"""
from __future__ import annotations

import json
import os
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 3


class NineRouterConfigError(RuntimeError):
    """Required 9router configuration (API key / model / base URL) is
    missing. Never falls back to a guessed value — see module docstring."""


class NineRouterResponseError(RuntimeError):
    """The model's response never produced valid JSON matching the
    expected schema, after all retry attempts were exhausted."""


class NineRouterClient:
    """Async 9router client — see module docstring for the configuration
    and compatibility-assumption caveats.

    Configuration (env vars only; `Do not hardcode the API key or model
    anywhere in the code` per project policy — this applies to base_url
    too, since a wrong hardcoded host is a real credential-leak risk).
    `.env` at the repo root (gitignored) is loaded automatically via
    load_dotenv() before these are read, so a local run needs nothing more
    than a `.env` file with the three lines below — no `source .env` step:
        NINEROUTER_API_KEY   — required
        NINEROUTER_MODEL     — required
        NINEROUTER_BASE_URL  — required, e.g. "https://<host>/v1"
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_wait_seconds: tuple[float, float] = (1.0, 10.0),
    ):
        # .env is the single source of truth for local runs — load it before
        # reading any NINEROUTER_* var. Never overrides a variable already
        # set in the process environment (default override=False), so real
        # CI secrets still win over a stray local .env.
        load_dotenv()

        self.api_key = api_key or os.environ.get("NINEROUTER_API_KEY")
        self.model = model or os.environ.get("NINEROUTER_MODEL")
        self.base_url = (base_url or os.environ.get("NINEROUTER_BASE_URL") or "").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        # (min, max) seconds for exponential backoff between retries —
        # overridable so tests don't have to sleep through real backoff delays.
        self.retry_wait_seconds = retry_wait_seconds

        missing = [
            name for name, val in
            [("NINEROUTER_API_KEY", self.api_key), ("NINEROUTER_MODEL", self.model),
             ("NINEROUTER_BASE_URL", self.base_url)]
            if not val
        ]
        if missing:
            raise NineRouterConfigError(
                f"Missing required 9router configuration: {', '.join(missing)}. "
                "These must be set as environment variables — see "
                "docs/AI_LAB_ARCHITECTURE.md for what's known/unknown about 9router's API contract."
            )

    async def complete_structured(
        self, prompt: str, response_model: type[T], system: str | None = None,
    ) -> T:
        """Send `prompt`, retrying on network errors and on responses that
        fail JSON parsing or schema validation. Returns a validated
        instance of `response_model`, or raises NineRouterResponseError
        once all attempts are exhausted."""
        last_error: Exception | None = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(
                    multiplier=1, min=self.retry_wait_seconds[0], max=self.retry_wait_seconds[1],
                ),
                retry=retry_if_exception_type((httpx.HTTPError, NineRouterResponseError)),
                reraise=True,
            ):
                with attempt:
                    raw_content = await self._call(prompt, response_model, system)
                    return self._parse(raw_content, response_model)
        except (httpx.HTTPError, NineRouterResponseError) as exc:
            last_error = exc
        raise NineRouterResponseError(
            f"9router call failed after {self.max_retries} attempt(s): {last_error}"
        ) from last_error

    async def _call(self, prompt: str, response_model: type[T], system: str | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            # json_schema, not json_object: at least one 9router-routed
            # provider (Novita, serving tencent/hy3) rejects json_object
            # with "does not support 'json_object' response format.
            # Supported formats: json_schema" — verified against the live
            # endpoint. json_schema is also the stricter, more portable
            # OpenAI-compatible option, built directly from the Pydantic
            # model so the two can never drift apart.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers,
            )
            response.raise_for_status()
            # Not response.json(): verified live against this endpoint that
            # a long/complex response body can contain a complete, valid
            # top-level JSON object followed by trailing extra bytes (json
            # raised "Extra data" at a later offset — a server/proxy-side
            # quirk, not something this client can prevent). raw_decode
            # parses only the first complete JSON value and deliberately
            # ignores whatever comes after it, which is the correct
            # response either way. A body that isn't valid JSON at all
            # still raises — surfaced as NineRouterResponseError below,
            # never propagated as an uncaught crash.
            try:
                data, _ = json.JSONDecoder().raw_decode(response.text)
            except json.JSONDecodeError as exc:
                raise NineRouterResponseError(f"9router response body was not valid JSON: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise NineRouterResponseError(
                f"Unexpected 9router response shape (expected OpenAI-compatible "
                f"choices[0].message.content): {data!r}"
            ) from exc

    def _parse(self, raw_content: str, response_model: type[T]) -> T:
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise NineRouterResponseError(f"9router response was not valid JSON: {exc}") from exc

        try:
            return response_model.model_validate(data)
        except ValidationError as exc:
            logger.warning(f"9router response failed schema validation ({response_model.__name__}): {exc}")
            raise NineRouterResponseError(
                f"9router response did not match {response_model.__name__}: {exc}"
            ) from exc


class MockNineRouterClient:
    """Deterministic, no network — same role as
    stock_scanner.learning.hypothesis_agent.MockLLMClient: lets the whole
    Evidence -> Hypothesis -> Decision -> AIRecommendation pipeline be
    built and verified end to end before 9router's real API contract is
    confirmed. Exposes the same `complete_structured` signature as
    NineRouterClient so agents code never has to branch on which client it
    was given.
    """

    def __init__(self, responses: dict[str, BaseModel] | None = None):
        # Optional: map response_model class name -> a fixed instance to
        # return every time. Without one, a bland-but-valid default
        # instance is constructed from the model's own defaults/required
        # fields via _default_for.
        self._responses = responses or {}

    async def complete_structured(
        self, prompt: str, response_model: type[T], system: str | None = None,
    ) -> T:
        if response_model.__name__ in self._responses:
            return self._responses[response_model.__name__]  # type: ignore[return-value]
        return _default_mock_instance(response_model)


def _default_mock_instance(response_model: type[T]) -> T:
    from stock_scanner.ai_lab.schemas import (
        DecisionOutput,
        HypothesisOutput,
        RecommendationAction,
        RiskLevel,
    )

    if response_model is HypothesisOutput:
        return HypothesisOutput(  # type: ignore[return-value]
            why="Mock hypothesis — replace with a real 9router response.",
            confidence=0.5,
            confidence_explanation="Mock confidence explanation.",
            strengths=["Mock strength"], weaknesses=["Mock weakness"], risks=["Mock risk"],
        )
    if response_model is DecisionOutput:
        return DecisionOutput(  # type: ignore[return-value]
            score=50.0, confidence=0.5, recommendation=RecommendationAction.WATCH,
            expected_return=0.0, risk_level=RiskLevel.MEDIUM,
            reasoning_summary="Mock decision — replace with a real 9router response.",
        )
    raise NotImplementedError(f"MockNineRouterClient has no default for {response_model.__name__}")
