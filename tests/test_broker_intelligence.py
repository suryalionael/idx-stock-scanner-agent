"""Tests for stock_scanner/pipeline/broker_intelligence.py::classify_broker().

This module powers the live Smart Money tab (via smart_money_screener.py)
and single-ticker broker-detail panels — no test coverage existed for it
before the broker classification audit refactored classify_broker() to
read from stock_scanner/configs/broker_config.yaml instead of two hardcoded
frozensets. These tests lock in the exact four-value return contract
("foreign" | "big_local" | "local" | "unknown") that downstream code
depends on.
"""
from stock_scanner.pipeline.broker_intelligence import classify_broker


def test_classify_broker_foreign():
    assert classify_broker("GS") == "foreign"
    assert classify_broker("cs") == "foreign"  # case-insensitive


def test_classify_broker_big_local():
    assert classify_broker("CC") == "big_local"
    assert classify_broker("PD") == "big_local"


def test_classify_broker_local_fallback_for_well_formed_unrecognized_code():
    # XL/XC are canonically "retail" in the new classification, but this
    # module has no retail bucket — they fall back to "local", exactly as
    # they did before the audit (this is the whole point of legacy_type).
    assert classify_broker("XL") == "local"
    assert classify_broker("XC") == "local"


def test_classify_broker_unknown_for_empty_or_malformed_code():
    assert classify_broker("") == "unknown"
    assert classify_broker("ABC") == "unknown"
    assert classify_broker("1") == "unknown"


def test_classify_broker_disputed_code_still_returns_a_legacy_bucket():
    # OD is canonically "mixed_unknown" post-audit (a genuine 3-way identity
    # conflict — see docs/BROKER_CLASSIFICATION_AUDIT.md), but
    # classify_broker()'s contract has no such value — it must still return
    # one of the original four so smart_money_screener.py's aggregation
    # never breaks.
    assert classify_broker("OD") in {"foreign", "big_local", "local", "unknown"}
