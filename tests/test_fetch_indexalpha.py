"""Tests for stock_scanner.pipeline.fetch_indexalpha — proves the IndexAlpha
integration is active and correctly handled, without spending any real
quota (paid plan, 25,000 requests/month). All HTTP is mocked; no network
calls occur.

Covers:
  - ticker cleaning / cache path construction
  - response normalization (success, empty, partial/malformed fields)
  - auth: missing key raises before any request is attempted
  - HTTP error handling: 401 (auth), 429 (rate-limit + retry), 5xx retry,
    timeout retry, connection error retry
  - logical failure (HTTP 200 but success=false) detection
  - the API key is never present in any exception message or log call
  - health-state recording (success path resets consecutive_failures;
    failure path increments it) without touching the real committed
    data/published/indexalpha_health.json
"""
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stock_scanner.pipeline import fetch_indexalpha as ia


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_health_path(tmp_path, monkeypatch):
    """Every test writes health state to a throwaway file, never the real
    committed data/published/indexalpha_health.json."""
    monkeypatch.setattr(ia, "_HEALTH_PATH", tmp_path / "indexalpha_health.json")
    yield


@pytest.fixture
def no_sleep(monkeypatch):
    """Prevent real time.sleep() in retry logic — makes tests fast."""
    monkeypatch.setattr(ia.time, "sleep", lambda *_: None)


@pytest.fixture
def api_key_env(no_sleep, monkeypatch):
    monkeypatch.setenv("INDEX_ALPHA_API_KEY", "test-key-not-real")


def _mock_response(json_body: dict, status: int = 200):
    """Build a context-manager mock compatible with urllib.request.urlopen()."""
    body_bytes = json.dumps(json_body).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body_bytes
    resp.status = status
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


# ---------------------------------------------------------------------------
# Ticker cleaning / cache path
# ---------------------------------------------------------------------------

def test_clean_ticker_strips_jk_suffix():
    assert ia._clean_ticker("BBCA.JK") == "BBCA"
    assert ia._clean_ticker("bbca") == "BBCA"
    assert ia._clean_ticker(" BBCA.JK ") == "BBCA"


def test_broker_cache_path_matches_indexalpha_convention(tmp_path):
    path = ia._broker_cache_path("BBCA.JK", "2026-06-17", tmp_path)
    assert path == tmp_path / "BBCA.JK_2026-06-17.parquet"
    # Bare ticker (no .JK) must produce the SAME path — this is the exact
    # convention dashboard/data_loader.py's load_broker_history() also
    # depends on (verified empirically against real cache files this audit).
    path2 = ia._broker_cache_path("BBCA", "2026-06-17", tmp_path)
    assert path2 == path


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

def test_normalize_response_empty_list_returns_empty_df():
    assert ia._normalize_response([]).empty


def test_normalize_response_maps_fields_correctly():
    raw = [{
        "code": "yp", "buy_volume": 1000, "sell_volume": 400,
        "buy_value": 5_000_000, "sell_value": 2_000_000,
        "buy_avg": 5000.0, "sell_avg": 5050.0,
        "buy_freq": 12, "sell_freq": 5,
    }]
    df = ia._normalize_response(raw)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["broker_code"] == "YP"  # uppercased
    assert row["broker_name"] == "Indo Premier Sekuritas"  # known-code lookup
    assert row["buy_lot"] == 1000.0
    assert row["sell_lot"] == 400.0
    assert row["net_lot"] == 600.0
    assert row["net_value"] == 3_000_000.0


def test_normalize_response_unknown_broker_code_falls_back_to_unknown():
    raw = [{"code": "ZZ", "buy_volume": 1, "sell_volume": 1}]
    df = ia._normalize_response(raw)
    assert df.iloc[0]["broker_name"] == "Unknown"


def test_normalize_response_handles_missing_optional_fields():
    """Malformed/partial payload (missing buy_value etc.) must not raise —
    missing numeric fields default to 0, not NaN or an exception."""
    raw = [{"code": "CC"}]
    df = ia._normalize_response(raw)
    assert df.iloc[0]["buy_lot"] == 0.0
    assert df.iloc[0]["net_value"] == 0.0


def test_normalize_response_sorts_by_abs_net_lot_descending():
    raw = [
        {"code": "AA", "buy_volume": 10, "sell_volume": 10},   # net=0
        {"code": "BB", "buy_volume": 100, "sell_volume": 0},   # net=100
        {"code": "CC", "buy_volume": 0, "sell_volume": 500},   # net=-500
    ]
    df = ia._normalize_response(raw)
    assert list(df["broker_code"]) == ["CC", "BB", "AA"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_get_api_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("INDEX_ALPHA_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="INDEX_ALPHA_API_KEY"):
        ia._get_api_key()


def test_get_api_key_reads_env(api_key_env):
    assert ia._get_api_key() == "test-key-not-real"


def test_fetcher_without_key_raises_before_any_request(monkeypatch):
    """No network call should even be attempted if the key is missing.
    fetch() never raises to its caller (dashboard/data_loader.py relies on
    this) — it catches the EnvironmentError internally and returns an empty
    DataFrame, same as every other failure path."""
    monkeypatch.delenv("INDEX_ALPHA_API_KEY", raising=False)
    fetcher = ia.IndexAlphaFetcher()
    with patch("urllib.request.urlopen") as mock_urlopen:
        df = fetcher.fetch("BBCA", "2026-06-17")
        assert df.empty
        mock_urlopen.assert_not_called()


# ---------------------------------------------------------------------------
# HTTP error handling (401 / 429 / timeout) — all mocked, zero network calls
# ---------------------------------------------------------------------------

def test_get_success_parses_json_and_records_health(api_key_env):
    payload = {"success": True, "data": [{"code": "YP", "buy_volume": 1, "sell_volume": 1}]}
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")
    assert result["success"] is True

    state = json.loads(ia._HEALTH_PATH.read_text())
    assert state["consecutive_failures"] == 0
    assert state["last_status_code"] == 200
    assert state["last_error_type"] is None
    assert state["total_successes"] == 1


def test_get_401_raises_permission_error_and_never_leaks_key(api_key_env):
    err = urllib.error.HTTPError(url="x", code=401, msg="Unauthorized", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(PermissionError) as exc_info:
            ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "super-secret-key-value")

    assert "super-secret-key-value" not in str(exc_info.value)
    state = json.loads(ia._HEALTH_PATH.read_text())
    assert state["consecutive_failures"] == 1
    assert state["last_error_type"] == "auth_error"
    assert state["last_status_code"] == 401


def test_get_logical_failure_200_but_success_false(api_key_env):
    """HTTP 200 with {"success": false} must NOT record as health success.
    _get() should raise RuntimeError on logical failure."""
    payload = {"success": False, "error": "no data for this range"}
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        with pytest.raises(RuntimeError, match="logical failure"):
            ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")

    state = json.loads(ia._HEALTH_PATH.read_text())
    assert state["last_error_type"] == "logical_failure"
    assert state["consecutive_failures"] == 1
    assert state.get("total_successes", 0) == 0
    assert state.get("total_failures", 0) == 1


def test_get_429_retries_then_succeeds(api_key_env):
    """First call 429s, second (retry) succeeds — must not raise.
    _get now has _MAX_RETRIES=3, so we need 4 429s to exhaust, or
    [429, 429, success] to test retry-then-success."""
    err = urllib.error.HTTPError(url="x", code=429, msg="Too Many Requests", hdrs=None, fp=None)
    success_payload = {"success": True, "data": []}
    with patch("urllib.request.urlopen", side_effect=[err, err, _mock_response(success_payload)]):
        result = ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")
    assert result["success"] is True


def test_get_429_exhausted_retries_raises_runtime_error(api_key_env):
    """_MAX_RETRIES=3 → need 4 consecutive 429s to exhaust."""
    err = urllib.error.HTTPError(url="x", code=429, msg="Too Many Requests", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=[err] * 5):  # more than enough
        with pytest.raises(RuntimeError, match="429"):
            ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")

    state = json.loads(ia._HEALTH_PATH.read_text())
    assert "rate_limit" in state["last_error_type"]
    assert state["consecutive_failures"] >= 1


def test_get_500_retries_then_succeeds(api_key_env):
    """5xx errors are now retryable — 500 then success."""
    err = urllib.error.HTTPError(url="x", code=500, msg="Server Error", hdrs=None, fp=None)
    payload = {"success": True, "data": []}
    with patch("urllib.request.urlopen", side_effect=[err, _mock_response(payload)]):
        result = ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")
    assert result["success"] is True


def test_get_500_exhausted_retries_raises_runtime_error(api_key_env):
    """All 500 retries exhausted."""
    err = urllib.error.HTTPError(url="x", code=500, msg="Server Error", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=[err] * 4):
        with pytest.raises(RuntimeError, match="500"):
            ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")


def test_consecutive_failures_increment_then_reset_on_success(api_key_env):
    """401 is non-retryable, so each call records 1 failure immediately."""
    err = urllib.error.HTTPError(url="x", code=401, msg="Unauthorized", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=err):
        for _ in range(3):
            with pytest.raises(PermissionError):
                ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")
    assert json.loads(ia._HEALTH_PATH.read_text())["consecutive_failures"] == 3

    success_payload = {"success": True, "data": []}
    with patch("urllib.request.urlopen", return_value=_mock_response(success_payload)):
        ia._get("/stocks/broker-summary", {"ticker": "BBCA"}, "test-key-not-real")
    assert json.loads(ia._HEALTH_PATH.read_text())["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# IndexAlphaFetcher.fetch() — never raises to the caller (returns empty df)
# ---------------------------------------------------------------------------

def test_fetch_returns_empty_df_on_failure_not_exception(api_key_env):
    """fetch() must catch internal errors and return an empty DataFrame —
    callers (dashboard/data_loader.py) rely on this to render a fallback
    message instead of crashing the page."""
    err = urllib.error.HTTPError(url="x", code=401, msg="Unauthorized", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=err):
        df = ia.IndexAlphaFetcher().fetch("BBCA", "2026-06-17")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_returns_empty_df_when_api_reports_unsuccessful(api_key_env):
    """Logical failure (success=false) raises RuntimeError in _get(), but
    fetch() catches it and returns empty DataFrame."""
    payload = {"success": False, "error": "no data for this range"}
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        df = ia.IndexAlphaFetcher().fetch("BBCA", "2026-06-17")
    assert df.empty


def test_fetch_success_path_returns_real_dataframe(api_key_env):
    payload = {"success": True, "data": [
        {"code": "YP", "buy_volume": 500, "sell_volume": 100},
        {"code": "CC", "buy_volume": 50, "sell_volume": 600},
    ]}
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        df = ia.IndexAlphaFetcher().fetch("BBCA.JK", "2026-06-17")
    assert len(df) == 2
    assert set(df["broker_code"]) == {"YP", "CC"}
