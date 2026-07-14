"""Tests for the LLM articulation layer — the guardrail tests here are the
ones that matter most: they verify "no raw DB rows to the LLM" and
"no auto-promotion" are structural properties of this code, not just
documented intentions. See docs/LEARNING_AGENT_ARCHITECTURE.md.
"""
import json

import pytest

from stock_scanner.learning.hypothesis_agent import (
    MockLLMClient,
    NineRouterClient,
    _build_prompt,
    _parse_response,
    generate_hypotheses,
)
from stock_scanner.learning.pattern_dedup import ClusteredPattern
from stock_scanner.learning.pattern_miner import PatternCandidate


def _candidate(signal_ids: set, ticker_suffix: str = "BBCA.JK") -> PatternCandidate:
    return PatternCandidate(
        dimensions=("vol_ratio_20d", "atr_breakout"),
        slice_definition={"vol_ratio_20d": "Q5", "atr_breakout": True},
        interaction_order=2, n=97, n_success=18, win_rate=0.1856, win_rate_shrunk=0.1602,
        baseline_win_rate=0.0372, ci_lower=0.1207, ci_upper=0.26, p_value=0.00001,
        p_value_adjusted=0.00001, ticker_concentration=0.3, ticker_concentration_flag=False,
        time_split_stable=True, passed_gate=True, signal_ids=frozenset(signal_ids),
    )


def _cluster(candidate: PatternCandidate, cluster_id: str = "abc123", member_count: int = 3) -> ClusteredPattern:
    return ClusteredPattern(cluster_id=cluster_id, representative=candidate, member_count=member_count,
                            members=[candidate])


# ---------------------------------------------------------------------------
# Guardrail: prompt never contains raw signal_ids or tickers
# ---------------------------------------------------------------------------

def test_prompt_never_contains_ticker_suffix():
    candidate = _candidate({"deadbeef01234567", "cafebabe89abcdef"})
    cluster = _cluster(candidate)
    prompt = _build_prompt(cluster)
    assert ".JK" not in prompt


def test_prompt_never_contains_raw_signal_ids():
    signal_ids = {"deadbeef01234567", "cafebabe89abcdef"}
    candidate = _candidate(signal_ids)
    cluster = _cluster(candidate)
    prompt = _build_prompt(cluster)
    for sid in signal_ids:
        assert sid not in prompt


def test_prompt_contains_only_aggregated_stats():
    candidate = _candidate({"s1", "s2"})
    cluster = _cluster(candidate)
    prompt = _build_prompt(cluster)
    # The aggregated numbers ARE expected in the prompt:
    assert "97" in prompt or "0.1856" in prompt.replace(",", "")  # sample size or win rate present in some form
    assert str(cluster.member_count) in prompt


# ---------------------------------------------------------------------------
# Guardrail: response parser never trusts the LLM for structural fields
# ---------------------------------------------------------------------------

def test_parse_response_forces_status_candidate_even_if_llm_says_otherwise():
    candidate = _candidate({"s1"})
    cluster = _cluster(candidate)
    raw = json.dumps({
        "hypothesis": "Test", "confidence": 0.7, "affected_sector": None,
        "expected_effect": "Higher win rate",
        "status": "promoted",   # adversarial/malformed — must be ignored
    })
    hyp = _parse_response(raw, cluster)
    assert hyp is not None
    assert hyp.status == "candidate"


def test_parse_response_forces_supporting_trades_from_data_not_llm():
    candidate = _candidate({"s1"})   # n_success = 18, hardcoded in _candidate()
    cluster = _cluster(candidate)
    raw = json.dumps({
        "hypothesis": "Test", "confidence": 0.7, "affected_sector": None,
        "expected_effect": "Higher win rate",
        "supporting_trades": 999999,   # adversarial — must be ignored
    })
    hyp = _parse_response(raw, cluster)
    assert hyp is not None
    assert hyp.supporting_trades == 18


def test_parse_response_forces_source_cluster_id_from_data():
    candidate = _candidate({"s1"})
    cluster = _cluster(candidate, cluster_id="real-cluster-id")
    raw = json.dumps({
        "hypothesis": "Test", "confidence": 0.7, "affected_sector": None,
        "expected_effect": "Higher win rate",
        "source_cluster_id": "fabricated-id",   # adversarial — must be ignored
    })
    hyp = _parse_response(raw, cluster)
    assert hyp is not None
    assert hyp.source_cluster_id == "real-cluster-id"


# ---------------------------------------------------------------------------
# Malformed responses — never raise, always skip
# ---------------------------------------------------------------------------

def test_parse_response_malformed_json_returns_none():
    candidate = _candidate({"s1"})
    cluster = _cluster(candidate)
    assert _parse_response("{not valid json", cluster) is None


def test_parse_response_missing_required_keys_returns_none():
    candidate = _candidate({"s1"})
    cluster = _cluster(candidate)
    raw = json.dumps({"hypothesis": "Test"})  # missing confidence/affected_sector/expected_effect
    assert _parse_response(raw, cluster) is None


def test_parse_response_confidence_out_of_range_returns_none():
    candidate = _candidate({"s1"})
    cluster = _cluster(candidate)
    raw = json.dumps({
        "hypothesis": "Test", "confidence": 1.5, "affected_sector": None,
        "expected_effect": "Higher win rate",
    })
    assert _parse_response(raw, cluster) is None


def test_parse_response_non_numeric_confidence_returns_none():
    candidate = _candidate({"s1"})
    cluster = _cluster(candidate)
    raw = json.dumps({
        "hypothesis": "Test", "confidence": "very high", "affected_sector": None,
        "expected_effect": "Higher win rate",
    })
    assert _parse_response(raw, cluster) is None


# ---------------------------------------------------------------------------
# generate_hypotheses — batch behavior
# ---------------------------------------------------------------------------

def test_generate_hypotheses_end_to_end_with_mock_client():
    candidate = _candidate({"s1", "s2"})
    cluster = _cluster(candidate)
    client = MockLLMClient()
    hypotheses = generate_hypotheses([cluster], client)
    assert len(hypotheses) == 1
    assert hypotheses[0].status == "candidate"
    assert hypotheses[0].source_cluster_id == cluster.cluster_id


def test_generate_hypotheses_skips_bad_response_without_aborting_batch():
    cluster_1 = _cluster(_candidate({"s1"}), cluster_id="c1")
    cluster_2 = _cluster(_candidate({"s2"}), cluster_id="c2")

    # A client that always returns malformed JSON — proves a bad response
    # is skipped for every cluster without raising or aborting the batch.
    client = MockLLMClient(response="{not valid json")
    hypotheses = generate_hypotheses([cluster_1, cluster_2], client)
    assert hypotheses == []


def test_generate_hypotheses_propagates_not_implemented_immediately():
    candidate = _candidate({"s1"})
    cluster = _cluster(candidate)
    client = NineRouterClient()
    with pytest.raises(NotImplementedError):
        generate_hypotheses([cluster], client)
