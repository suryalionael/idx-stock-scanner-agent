"""LLM articulation layer over de-duplicated pattern clusters.

Turns a ClusteredPattern (stock_scanner.learning.pattern_dedup) into a
structured hypothesis via an LLM call. This is the ONLY module in the
Learning Agent that talks to an LLM — everything upstream (pattern_miner,
pattern_dedup) is pure statistics. See docs/LEARNING_AGENT_ARCHITECTURE.md.

Guardrails enforced here, not just documented:
  - The prompt builder (_build_prompt) reads ONLY the cluster's aggregated
    stats. It never touches ClusteredPattern.members[*].signal_ids or any
    ticker — those exist on the cluster for traceability, not for the LLM
    to see. See test_hypothesis_agent.py's prompt-content assertions.
  - The response parser (_parse_response) never trusts the LLM for status,
    supporting_trades, or source_cluster_id, even if the model includes
    them — those three fields are always overwritten from code. This is
    what makes "no auto-promotion" a structural property of the parser,
    not a prompt instruction a model could ignore.
  - A malformed/unparseable response is logged and skipped — never
    retried into a looser shape, never raised past the batch.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

from stock_scanner.learning.pattern_dedup import ClusteredPattern

_REQUIRED_KEYS = {"hypothesis", "confidence", "affected_sector", "expected_effect"}


# ---------------------------------------------------------------------------
# LLM client interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...


class NineRouterClient(LLMClient):
    """Real 9router integration — not yet wired. Raises rather than
    returning fake data, so a misconfigured run fails loudly instead of
    silently producing hypotheses that look real but aren't. Fill in once
    9router's base URL / auth scheme / request-response shape are known —
    see docs/LEARNING_AGENT_ARCHITECTURE.md, Phase 2."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            "9router API details (base URL, auth scheme, request/response shape) are "
            "not yet configured. See docs/LEARNING_AGENT_ARCHITECTURE.md, Phase 2. "
            "Use MockLLMClient to exercise the rest of the pipeline in the meantime."
        )


class MockLLMClient(LLMClient):
    """Deterministic, no network — lets pattern_dedup -> hypothesis_agent ->
    knowledge_base be built and verified end to end before 9router is wired."""

    def __init__(self, response: str | None = None):
        self._response = response

    def complete(self, prompt: str) -> str:
        if self._response is not None:
            return self._response
        return json.dumps(
            {
                "hypothesis": "Mock hypothesis — replace with a real LLM response.",
                "confidence": 0.5,
                "affected_sector": None,
                "expected_effect": "Higher win rate (mock)",
            }
        )


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    hypothesis: str
    confidence: float
    supporting_trades: int
    affected_sector: str | None
    expected_effect: str
    status: str
    source_cluster_id: str

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "supporting_trades": self.supporting_trades,
            "affected_sector": self.affected_sector,
            "expected_effect": self.expected_effect,
            "status": self.status,
            "source_cluster_id": self.source_cluster_id,
        }


# ---------------------------------------------------------------------------
# Prompt construction — aggregated stats only, see module docstring
# ---------------------------------------------------------------------------


def _build_prompt(cluster: ClusteredPattern) -> str:
    r = cluster.representative
    slice_desc = ", ".join(f"{k}={v}" for k, v in r.slice_definition.items())
    return f"""You are a quantitative research assistant reviewing a statistically pre-validated
pattern from an Indonesian stock market (IDX) signal-screening system. The pattern below
already passed a rigorous statistical gate (Fisher's exact test, Benjamini-Hochberg FDR
correction, Wilson confidence interval lower bound above baseline) — your job is ONLY to
describe it in plain English, not to judge whether it is statistically real.

Pattern (representative of {cluster.member_count} near-duplicate statistical findings):
  Feature combination: {slice_desc}
  Sample size: {r.n} signals, {r.n_success} winners
  Win rate: {r.win_rate:.2%} (shrunk estimate: {r.win_rate_shrunk:.2%}), baseline: {r.baseline_win_rate:.2%}
  95% CI lower bound on win rate: {r.ci_lower:.2%}
  FDR-adjusted p-value: {r.p_value_adjusted:.4f}
  Ticker concentration flag: {r.ticker_concentration_flag}
  Time-split direction stability: {r.time_split_stable}

Respond with ONLY a JSON object, no other text, with exactly these keys:
  "hypothesis": one or two sentences describing the pattern in trading terms
  "confidence": a number from 0 to 1 reflecting how actionable this looks (qualitative
    judgment — the statistical significance is already established above, this is about
    whether the effect size and stability flags make it worth testing further)
  "affected_sector": a sector name if the pattern is sector-specific, or null otherwise
  "expected_effect": a short phrase, e.g. "Higher win rate" or "Lower drawdown risk"

Do not include any other fields. Do not speculate about individual tickers or trades —
you were not given any."""


# ---------------------------------------------------------------------------
# Response parsing — strict, code-enforced fields, never trusts the model
# ---------------------------------------------------------------------------


def _parse_response(raw: str, cluster: ClusteredPattern) -> Hypothesis | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(
            f"hypothesis_agent: unparseable response for cluster {cluster.cluster_id} (skip): {e}"
        )
        return None

    if not isinstance(data, dict) or not _REQUIRED_KEYS.issubset(data.keys()):
        logger.warning(
            f"hypothesis_agent: response missing required keys for cluster {cluster.cluster_id} (skip)"
        )
        return None

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        logger.warning(
            f"hypothesis_agent: non-numeric confidence for cluster {cluster.cluster_id} (skip)"
        )
        return None
    if not (0.0 <= confidence <= 1.0):
        logger.warning(
            f"hypothesis_agent: confidence {confidence} out of [0,1] for cluster {cluster.cluster_id} (skip)"
        )
        return None

    hypothesis_text = data.get("hypothesis")
    expected_effect = data.get("expected_effect")
    if not isinstance(hypothesis_text, str) or not hypothesis_text.strip():
        logger.warning(
            f"hypothesis_agent: empty/invalid hypothesis text for cluster {cluster.cluster_id} (skip)"
        )
        return None

    affected_sector = data.get("affected_sector")
    if affected_sector is not None and not isinstance(affected_sector, str):
        affected_sector = None

    return Hypothesis(
        hypothesis=hypothesis_text.strip(),
        confidence=confidence,
        # Always code-supplied, never trusted from the model's output even if present:
        supporting_trades=cluster.representative.n_success,
        affected_sector=affected_sector,
        expected_effect=expected_effect if isinstance(expected_effect, str) else "",
        status="candidate",
        source_cluster_id=cluster.cluster_id,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def generate_hypotheses(clusters: list[ClusteredPattern], client: LLMClient) -> list[Hypothesis]:
    """One LLM call per cluster. A single bad RESPONSE (malformed JSON,
    missing keys, out-of-range confidence) is logged and skipped — never
    aborts the batch. A client-level NotImplementedError (e.g.
    NineRouterClient before it's wired) is NOT caught here — it should fail
    the whole run loudly and immediately, not get silently swallowed into
    N identical per-cluster warnings that look like "no interesting
    clusters" instead of "the LLM client isn't configured yet"."""
    hypotheses: list[Hypothesis] = []
    for cluster in clusters:
        prompt = _build_prompt(cluster)
        try:
            raw = client.complete(prompt)
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001 — a transient per-call failure must not abort the batch
            logger.warning(
                f"hypothesis_agent: LLM call failed for cluster {cluster.cluster_id} (skip): {e}"
            )
            continue
        hyp = _parse_response(raw, cluster)
        if hyp is not None:
            hypotheses.append(hyp)
    logger.info(
        f"hypothesis_agent: {len(hypotheses)}/{len(clusters)} clusters produced a valid hypothesis"
    )
    return hypotheses
