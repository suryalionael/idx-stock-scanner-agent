"""Tests for stock_scanner/pipeline/broker_analytics.py — the single source
of truth for broker classification (see docs/BROKER_CLASSIFICATION_AUDIT.md
and stock_scanner/configs/broker_config.yaml). load_broker_config() /
get_broker_name() / get_broker_type() / get_broker_legacy_type() are the
functions every other broker-related module (broker_intelligence.py,
fetch_indexalpha.py, dashboard/data_loader.py) now reads through instead of
maintaining their own hardcoded mapping.
"""
import pandas as pd
import pytest

from stock_scanner.pipeline.broker_analytics import (
    VALID_BROKER_TYPES,
    classify_brokers_in_df,
    get_broker_legacy_type,
    get_broker_name,
    get_broker_type,
    load_broker_config,
    validate_broker_config,
)

# Pre-audit hardcoded sets, kept here ONLY as the reference for the
# backward-compatibility regression test below — never re-imported from
# broker_intelligence.py (which no longer defines them at all).
_OLD_FOREIGN_BROKER_CODES = frozenset({
    "CS", "ML", "MS", "JP", "DB", "YU", "ZP", "RB", "GS", "UX", "AK", "DP",
    "SB", "BK", "QA", "LS", "MK", "KK", "OX", "SA",
})
_OLD_BIG_LOCAL_BROKER_CODES = frozenset({
    "YP", "OD", "DH", "YJ", "ZH", "BW", "DR", "CC", "ID", "YB", "GI", "LG",
    "ZU", "HD", "PD", "EP", "FZ", "KI", "AD", "NI",
})


def _old_classify(code: str) -> str:
    if code in _OLD_FOREIGN_BROKER_CODES:
        return "foreign"
    if code in _OLD_BIG_LOCAL_BROKER_CODES:
        return "big_local"
    if len(code) == 2 and code.isalpha():
        return "local"
    return "unknown"


# ---------------------------------------------------------------------------
# load_broker_config
# ---------------------------------------------------------------------------

def test_load_broker_config_reads_the_real_file():
    cfg = load_broker_config()
    assert "brokers" in cfg
    assert len(cfg["brokers"]) > 0
    assert "metrics" in cfg  # unchanged section from the pre-audit config


def test_load_broker_config_missing_file_returns_empty_dict(tmp_path):
    cfg = load_broker_config(tmp_path / "does_not_exist.yaml")
    assert cfg == {}


def test_load_broker_config_malformed_file_returns_empty_dict(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("brokers: [this is not: valid: yaml: at all")
    cfg = load_broker_config(path)
    assert cfg == {}


# ---------------------------------------------------------------------------
# validate_broker_config — schema validation (architecture review
# recommendation #1: enum validation only, no schema changes). The first
# test here is the actual regression guard: it's what makes a future
# hand-edit that types "retial" or "hihg" fail CI immediately instead of
# silently shipping a broken classification.
# ---------------------------------------------------------------------------

def test_real_broker_config_has_no_validation_errors():
    """The permanent regression guard. If this ever fails, someone edited
    broker_config.yaml with a value outside the documented enums — fix the
    entry, don't loosen this test."""
    errors = validate_broker_config(load_broker_config())
    assert errors == []


@pytest.mark.parametrize("field,bad_value", [
    ("type", "retial"),            # typo
    ("type", "RETAIL"),            # wrong case — enums are case-sensitive
    ("type", "big_local"),         # a legacy_type value, not a type value
    ("legacy_type", "hihg"),       # typo
    ("legacy_type", "institutional"),  # a type value, not a legacy_type value
    ("type_confidence", "very_high"),  # not one of the four levels
    ("name_confidence", "unknown"),    # close to a real value, but not one
])
def test_validate_broker_config_catches_invalid_enum_value(field, bad_value):
    entry = {
        "name": "Test Broker", "name_confidence": "high",
        "type": "retail", "type_confidence": "high",
        "legacy_type": "local", "notes": "synthetic test entry",
    }
    entry[field] = bad_value
    errors = validate_broker_config({"brokers": {"ZZ": entry}})
    assert len(errors) == 1
    assert "ZZ" in errors[0]
    assert field in errors[0]


def test_validate_broker_config_catches_missing_enum_field():
    entry = {
        "name": "Test Broker", "name_confidence": "high",
        "type_confidence": "high", "legacy_type": "local", "notes": "x",
        # 'type' omitted entirely
    }
    errors = validate_broker_config({"brokers": {"ZZ": entry}})
    assert any("ZZ.type" in e for e in errors)


def test_validate_broker_config_catches_multiple_bad_fields_in_one_entry():
    entry = {
        "name": "Test Broker", "name_confidence": "bogus",
        "type": "bogus", "type_confidence": "high",
        "legacy_type": "local", "notes": "x",
    }
    errors = validate_broker_config({"brokers": {"ZZ": entry}})
    assert len(errors) == 2


def test_validate_broker_config_non_dict_entry():
    errors = validate_broker_config({"brokers": {"ZZ": "not a mapping"}})
    assert len(errors) == 1
    assert "ZZ" in errors[0]


def test_validate_broker_config_non_dict_brokers_section():
    errors = validate_broker_config({"brokers": ["not", "a", "mapping"]})
    assert len(errors) == 1


def test_validate_broker_config_valid_entry_produces_no_errors():
    entry = {
        "name": "Test Broker", "name_confidence": "none",
        "type": "mixed_unknown", "type_confidence": "none",
        "legacy_type": "unknown", "notes": "x",
    }
    assert validate_broker_config({"brokers": {"ZZ": entry}}) == []


def test_validate_broker_config_empty_config():
    assert validate_broker_config({}) == []


def test_load_broker_config_does_not_raise_on_invalid_entries(tmp_path):
    """The fail-open half of the contract: a config with bad enum values
    must still load (with a logged warning, not tested here — loguru
    doesn't route through pytest's caplog by default) rather than crash
    the dashboard or the daily scan."""
    path = tmp_path / "broker_config.yaml"
    path.write_text(
        "brokers:\n"
        "  ZZ:\n"
        "    name: Test\n"
        "    name_confidence: high\n"
        "    type: not_a_real_type\n"
        "    type_confidence: high\n"
        "    legacy_type: local\n"
        "    notes: x\n"
    )
    cfg = load_broker_config(path)
    assert cfg["brokers"]["ZZ"]["type"] == "not_a_real_type"  # loaded as-is, not stripped


def test_get_broker_type_falls_back_safely_on_invalid_stored_value():
    """Defense in depth: even if a bad value slips past the load-time
    warning, the getter must never hand callers a value outside its
    documented four-value contract."""
    bad_cfg = {"brokers": {"ZZ": {"type": "not_a_real_type"}}}
    assert get_broker_type("ZZ", bad_cfg) == "mixed_unknown"


def test_get_broker_legacy_type_falls_back_safely_on_invalid_stored_value():
    bad_cfg = {"brokers": {"ZZ": {"legacy_type": "not_a_real_type"}}}
    assert get_broker_legacy_type("ZZ", bad_cfg) == "local"  # ZZ is a well-formed 2-letter code


# ---------------------------------------------------------------------------
# get_broker_name / get_broker_type — real config
# ---------------------------------------------------------------------------

def test_get_broker_name_known_code():
    assert get_broker_name("YP") == "Mirae Asset Sekuritas Indonesia"


def test_get_broker_name_unrecorded_code_is_unknown():
    assert get_broker_name("ZZ") == "Unknown"


def test_get_broker_name_case_and_whitespace_insensitive():
    assert get_broker_name(" yp ") == get_broker_name("YP")


def test_get_broker_type_known_retail_code():
    assert get_broker_type("XL") == "retail"
    assert get_broker_type("XC") == "retail"


def test_get_broker_type_disputed_code_is_mixed_unknown_not_a_guess():
    # OD has a genuine 3-way identity conflict in the audit — must never
    # resolve to a confident type.
    assert get_broker_type("OD") == "mixed_unknown"


def test_get_broker_type_unrecorded_code_is_mixed_unknown():
    assert get_broker_type("ZZ") == "mixed_unknown"


# ---------------------------------------------------------------------------
# get_broker_legacy_type — the backward-compatibility contract
# ---------------------------------------------------------------------------

def test_legacy_type_matches_pre_audit_classification_for_every_configured_code():
    """The single most important test in this file: every broker currently
    in broker_config.yaml must produce EXACTLY the type the old hardcoded
    FOREIGN_BROKER_CODES/BIG_LOCAL_BROKER_CODES sets would have produced —
    this is what makes the refactor backward-compatible for
    stock_scanner.pipeline.smart_money_screener, which depends on
    classify_broker()'s output. A future edit to broker_config.yaml that
    breaks this is a real regression, not a config typo to shrug off."""
    cfg = load_broker_config()
    mismatches = [
        (code, _old_classify(code), get_broker_legacy_type(code, cfg))
        for code in cfg["brokers"]
        if _old_classify(code) != get_broker_legacy_type(code, cfg)
    ]
    assert mismatches == []


def test_legacy_type_unrecorded_well_formed_code_falls_back_to_local():
    assert get_broker_legacy_type("ZZ") == "local"


def test_legacy_type_unrecorded_malformed_code_falls_back_to_unknown():
    assert get_broker_legacy_type("ABC") == "unknown"
    assert get_broker_legacy_type("") == "unknown"


# ---------------------------------------------------------------------------
# classify_brokers_in_df
# ---------------------------------------------------------------------------

def test_classify_brokers_in_df_uses_canonical_type():
    df = pd.DataFrame({"broker_code": ["XL", "GS", "CC", "ZZ"]})
    result = classify_brokers_in_df(df, load_broker_config())
    assert list(result["broker_type"]) == ["retail", "foreign", "institution", "local"]
    assert set(result["broker_type"]).issubset(VALID_BROKER_TYPES)


def test_classify_brokers_in_df_mixed_unknown_folds_into_local():
    # OD is canonically mixed_unknown (disputed identity) — the internal
    # vocabulary these metric formulas expect has no such bucket, so it
    # must fold into "local", the same "not confidently classified"
    # fallback the pre-audit code always used.
    df = pd.DataFrame({"broker_code": ["OD"]})
    result = classify_brokers_in_df(df, load_broker_config())
    assert result["broker_type"].iloc[0] == "local"


def test_classify_brokers_in_df_empty_df():
    result = classify_brokers_in_df(pd.DataFrame(), load_broker_config())
    assert result.empty


def test_classify_brokers_in_df_missing_broker_code_column():
    df = pd.DataFrame({"other_col": [1, 2]})
    result = classify_brokers_in_df(df, load_broker_config())
    assert (result["broker_type"] == "unknown").all()


def test_classify_brokers_in_df_nan_code_defaults_to_local():
    df = pd.DataFrame({"broker_code": [float("nan")]})
    result = classify_brokers_in_df(df, load_broker_config())
    assert result["broker_type"].iloc[0] == "local"
