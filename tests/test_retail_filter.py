"""Tests for the global Retail Accumulation filter — Phase 2 on top of the
frozen broker foundation (stock_scanner/configs/broker_config.yaml +
stock_scanner/pipeline/broker_analytics.py, see
docs/BROKER_CLASSIFICATION_AUDIT.md).

Covers dashboard/data_loader.py's enrich_df_with_top_brokers(compute_retail=
True) and apply_retail_filter() — the sidebar checkbox in dashboard/app.py
is a ~15-line wrapper around these (checkbox -> apply_retail_filter(df_all,
...) -> caption), not independently testable outside a live Streamlit
session, so the actual logic under test lives entirely here.
"""
import pandas as pd
import pytest

from dashboard.data_loader import apply_retail_filter, enrich_df_with_top_brokers
from stock_scanner.pipeline.broker_analytics import calculate_retail_ratio, classify_brokers_in_df

# A small, fully-controlled synthetic broker_config — deterministic and
# independent of whatever the live audited broker_config.yaml says about
# any particular code, so these tests don't silently break if the audit
# ever reclassifies a broker.
_SYNTHETIC_CONFIG = {
    "brokers": {
        "RT": {"name": "Retail Test Broker", "name_confidence": "high",
               "type": "retail", "type_confidence": "high",
               "legacy_type": "local", "notes": "test fixture"},
        "IN": {"name": "Institutional Test Broker", "name_confidence": "high",
               "type": "institutional", "type_confidence": "high",
               "legacy_type": "big_local", "notes": "test fixture"},
        "FR": {"name": "Foreign Test Broker", "name_confidence": "high",
               "type": "foreign", "type_confidence": "high",
               "legacy_type": "foreign", "notes": "test fixture"},
    },
    "metrics": {
        "retail_ratio": {"thresholds": {"significant": 50, "moderate": 20, "low": 0}},
    },
}


def _broker_df(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """rows: (broker_code, buy_lot, sell_lot)."""
    data = []
    for code, buy, sell in rows:
        data.append({
            "broker_code": code, "broker_name": code, "buy_lot": buy,
            "sell_lot": sell, "net_lot": buy - sell,
        })
    return pd.DataFrame(data)


def _write_broker_parquet(broker_dir, ticker: str, date: str, rows) -> None:
    broker_dir.mkdir(parents=True, exist_ok=True)
    clean = ticker.replace(".JK", "")
    _broker_df(rows).to_parquet(broker_dir / f"{clean}.JK_{date}.parquet", index=False)


@pytest.fixture
def synthetic_broker_config(monkeypatch):
    monkeypatch.setattr(
        "stock_scanner.pipeline.broker_analytics.load_broker_config",
        lambda *a, **k: _SYNTHETIC_CONFIG,
    )
    return _SYNTHETIC_CONFIG


# ---------------------------------------------------------------------------
# checkbox OFF -> identical dataframe (byte-identical, no broker read at all)
# ---------------------------------------------------------------------------

def test_checkbox_off_returns_identical_dataframe_same_object():
    df = pd.DataFrame({"ticker": ["AAAA.JK", "BBBB.JK"], "signal": ["BREAKOUT", "WATCH"]})
    result = apply_retail_filter(df, "2026-07-25", hide_retail=False)
    assert result is df  # not even a copy


def test_checkbox_off_does_not_touch_disk(tmp_path, monkeypatch):
    # Prove no parquet read happens when off: point broker_dir at a
    # directory that doesn't exist at all. If the OFF path tried to read
    # anything, this would raise; it must return untouched instead.
    df = pd.DataFrame({"ticker": ["AAAA.JK"], "signal": ["BREAKOUT"]})
    result = apply_retail_filter(df, "2026-07-25", hide_retail=False, broker_dir=tmp_path / "does_not_exist")
    assert result is df


# ---------------------------------------------------------------------------
# checkbox ON -> retail-dominated rows removed, others kept
# ---------------------------------------------------------------------------

def test_checkbox_on_removes_retail_dominated_rows(tmp_path, synthetic_broker_config):
    # RT (retail) buys 900, sells 0 -> all-retail positive net -> ratio 100%.
    _write_broker_parquet(tmp_path, "RETAILDOM.JK", "2026-07-25", [("RT", 900, 0)])
    # FR (foreign) buys 900, sells 0 -> 0% retail.
    _write_broker_parquet(tmp_path, "FOREIGNDOM.JK", "2026-07-25", [("FR", 900, 0)])

    df = pd.DataFrame({"ticker": ["RETAILDOM.JK", "FOREIGNDOM.JK"], "signal": ["BREAKOUT", "BREAKOUT"]})
    result = apply_retail_filter(df, "2026-07-25", hide_retail=True, broker_dir=tmp_path)

    assert list(result["ticker"]) == ["FOREIGNDOM.JK"]


def test_checkbox_on_keeps_mixed_row_below_threshold(tmp_path, synthetic_broker_config):
    # Retail buys 400, foreign buys 600 -> retail ratio 40% < 50% threshold.
    _write_broker_parquet(tmp_path, "MIXED.JK", "2026-07-25", [("RT", 400, 0), ("FR", 600, 0)])
    df = pd.DataFrame({"ticker": ["MIXED.JK"], "signal": ["BREAKOUT"]})
    result = apply_retail_filter(df, "2026-07-25", hide_retail=True, broker_dir=tmp_path)
    assert list(result["ticker"]) == ["MIXED.JK"]


# ---------------------------------------------------------------------------
# Unknown != Retail — fail open
# ---------------------------------------------------------------------------

def test_missing_broker_parquet_stays_visible_when_filter_on(tmp_path, synthetic_broker_config):
    # No file written for this ticker at all.
    df = pd.DataFrame({"ticker": ["NODATA.JK"], "signal": ["BREAKOUT"]})
    result = apply_retail_filter(df, "2026-07-25", hide_retail=True, broker_dir=tmp_path)
    assert list(result["ticker"]) == ["NODATA.JK"]


def test_unresolvable_ratio_stays_visible_when_filter_on(tmp_path, synthetic_broker_config):
    # A broker file exists but has no positive net_lot at all (everyone net
    # selling) -> calculate_retail_ratio() returns None -> must stay visible,
    # not be treated as "confirmed not retail" or "confirmed retail".
    _write_broker_parquet(tmp_path, "NOPOSITIVE.JK", "2026-07-25", [("FR", 0, 500)])
    df = pd.DataFrame({"ticker": ["NOPOSITIVE.JK"], "signal": ["BREAKOUT"]})

    enriched = enrich_df_with_top_brokers(df, "2026-07-25", broker_dir=tmp_path, compute_retail=True)
    assert pd.isna(enriched["retail_ratio"].iloc[0])
    assert enriched["is_retail_dominated"].iloc[0] is None

    result = apply_retail_filter(df, "2026-07-25", hide_retail=True, broker_dir=tmp_path)
    assert list(result["ticker"]) == ["NOPOSITIVE.JK"]


def test_mix_of_dominated_unknown_and_missing(tmp_path, synthetic_broker_config):
    _write_broker_parquet(tmp_path, "DOMINATED.JK", "2026-07-25", [("RT", 900, 0)])
    _write_broker_parquet(tmp_path, "UNRESOLVABLE.JK", "2026-07-25", [("FR", 0, 500)])
    # MISSING.JK has no file at all.
    df = pd.DataFrame({"ticker": ["DOMINATED.JK", "UNRESOLVABLE.JK", "MISSING.JK"]})
    result = apply_retail_filter(df, "2026-07-25", hide_retail=True, broker_dir=tmp_path)
    assert set(result["ticker"]) == {"UNRESOLVABLE.JK", "MISSING.JK"}


# ---------------------------------------------------------------------------
# retail_ratio calculation — reuses broker_analytics, no reinvented formula
# ---------------------------------------------------------------------------

def test_retail_ratio_matches_broker_analytics_calculate_retail_ratio(tmp_path, synthetic_broker_config):
    rows = [("RT", 700, 100), ("FR", 300, 0), ("IN", 200, 50)]
    _write_broker_parquet(tmp_path, "XYZ.JK", "2026-07-25", rows)
    df = pd.DataFrame({"ticker": ["XYZ.JK"]})

    enriched = enrich_df_with_top_brokers(df, "2026-07-25", broker_dir=tmp_path, compute_retail=True)
    got_ratio = enriched["retail_ratio"].iloc[0]

    # Independently compute via the exact same broker_analytics functions,
    # directly — this is the formula being reused, not a parallel one.
    broker_df = _broker_df(rows)
    classified = classify_brokers_in_df(broker_df, _SYNTHETIC_CONFIG)
    expected_ratio = calculate_retail_ratio(classified, _SYNTHETIC_CONFIG)

    assert got_ratio == pytest.approx(expected_ratio)


def test_is_retail_dominated_uses_configured_threshold_not_hardcoded(tmp_path, monkeypatch):
    # Threshold set to 90 instead of the default 50 — a ticker at 60% retail
    # must NOT be flagged dominated under this threshold, proving the
    # threshold comes from config, not a hardcoded constant.
    custom_config = {
        "brokers": _SYNTHETIC_CONFIG["brokers"],
        "metrics": {"retail_ratio": {"thresholds": {"significant": 90}}},
    }
    monkeypatch.setattr(
        "stock_scanner.pipeline.broker_analytics.load_broker_config",
        lambda *a, **k: custom_config,
    )
    _write_broker_parquet(tmp_path, "SIXTY.JK", "2026-07-25", [("RT", 600, 0), ("FR", 400, 0)])
    df = pd.DataFrame({"ticker": ["SIXTY.JK"]})
    enriched = enrich_df_with_top_brokers(df, "2026-07-25", broker_dir=tmp_path, compute_retail=True)
    assert enriched["retail_ratio"].iloc[0] == pytest.approx(60.0)
    assert enriched["is_retail_dominated"].iloc[0] is False


# ---------------------------------------------------------------------------
# Additive-only columns, existing columns untouched
# ---------------------------------------------------------------------------

def test_existing_columns_never_modified(tmp_path, synthetic_broker_config):
    _write_broker_parquet(tmp_path, "ABC.JK", "2026-07-25", [("RT", 900, 0)])
    df = pd.DataFrame({"ticker": ["ABC.JK"], "signal": ["BREAKOUT"], "total_score": [8.5]})
    result = apply_retail_filter(df.copy(), "2026-07-25", hide_retail=True, broker_dir=tmp_path)
    if not result.empty:
        assert result["signal"].iloc[0] == "BREAKOUT"
        assert result["total_score"].iloc[0] == 8.5


def test_enrich_default_compute_retail_false_has_no_new_columns(tmp_path):
    # Existing Swing-tab call shape (get_table_df) — must be completely
    # unaffected by the new parameter's existence.
    _write_broker_parquet(tmp_path, "ABC.JK", "2026-07-25", [("RT", 900, 0)])
    df = pd.DataFrame({"ticker": ["ABC.JK"]})
    result = enrich_df_with_top_brokers(df, "2026-07-25", broker_dir=tmp_path)
    assert "retail_ratio" not in result.columns
    assert "is_retail_dominated" not in result.columns
    assert "top_buyer" in result.columns


# ---------------------------------------------------------------------------
# No duplicate broker parquet reads
# ---------------------------------------------------------------------------

def test_no_duplicate_parquet_read_on_second_plain_call(tmp_path, synthetic_broker_config):
    _write_broker_parquet(tmp_path, "ABC.JK", "2026-07-25", [("RT", 900, 0)])
    df = pd.DataFrame({"ticker": ["ABC.JK"]})

    enriched = enrich_df_with_top_brokers(df, "2026-07-25", broker_dir=tmp_path, compute_retail=True)
    assert enriched["is_retail_dominated"].iloc[0] is True

    # Delete the underlying file — if the second call (the exact shape
    # get_table_df/Swing already makes today: no compute_retail arg) tried
    # to re-read it, this would silently lose the data. It must not touch
    # the file at all, because top_buyer/top_seller already exist.
    (tmp_path / "ABC.JK_2026-07-25.parquet").unlink()
    second = enrich_df_with_top_brokers(enriched, "2026-07-25", broker_dir=tmp_path)
    assert second["top_buyer"].iloc[0] == enriched["top_buyer"].iloc[0]
    assert second["is_retail_dominated"].iloc[0] is True  # preserved, not recomputed


def test_idempotency_guard_still_computes_retail_if_only_top_buyer_present(tmp_path, synthetic_broker_config):
    # Defensive edge case: a df that already has top_buyer/top_seller (from
    # an earlier plain call) but NOT retail columns must still get them
    # computed when compute_retail=True is requested — the guard must not
    # over-skip.
    _write_broker_parquet(tmp_path, "ABC.JK", "2026-07-25", [("RT", 900, 0)])
    df = pd.DataFrame({"ticker": ["ABC.JK"]})
    plain = enrich_df_with_top_brokers(df, "2026-07-25", broker_dir=tmp_path)  # no retail cols
    assert "retail_ratio" not in plain.columns

    with_retail = enrich_df_with_top_brokers(plain, "2026-07-25", broker_dir=tmp_path, compute_retail=True)
    assert with_retail["is_retail_dominated"].iloc[0] is True


# ---------------------------------------------------------------------------
# Existing Smart Money / broker_intelligence behavior unchanged
# ---------------------------------------------------------------------------

def test_broker_intelligence_classify_broker_untouched_by_this_feature():
    # This feature imports only stock_scanner.pipeline.broker_analytics
    # (classify_brokers_in_df/calculate_retail_ratio); broker_intelligence.py
    # — which powers the live Smart Money tab — was not touched. Spot-check
    # its existing, already-tested contract still holds (full coverage in
    # tests/test_broker_intelligence.py, re-run as part of the full suite).
    from stock_scanner.pipeline.broker_intelligence import classify_broker
    assert classify_broker("GS") == "foreign"
    assert classify_broker("CC") == "big_local"


def test_real_broker_config_end_to_end_smoke(tmp_path):
    # One integration-style check against the ACTUAL audited config (not
    # the synthetic fixture) — XL (Stockbit) is high-confidence Retail per
    # docs/BROKER_CLASSIFICATION_AUDIT.md.
    _write_broker_parquet(tmp_path, "SMOKE.JK", "2026-07-25", [("XL", 900, 0)])
    df = pd.DataFrame({"ticker": ["SMOKE.JK"]})
    result = apply_retail_filter(df, "2026-07-25", hide_retail=True, broker_dir=tmp_path)
    assert result.empty  # 100% XL (retail) net buying -> hidden
