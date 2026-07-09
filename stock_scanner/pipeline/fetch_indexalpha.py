"""Index Alpha API — broker summary fetcher dengan error handling robust.

Source: https://indexalpha.id / https://api.indexalpha.id

Endpoint
--------
  GET https://api.indexalpha.id/stocks/broker-summary
  Params:
    ticker   (str)  — "BBCA" (tanpa .JK)
    from     (date) — YYYY-MM-DD
    to       (date) — YYYY-MM-DD
    investor (enum) — "all" | "f" (foreign) | "or" | "d" (domestic)
    market   (enum) — "RG" (Regular, default) | "NG" (Negotiated) | "ALL"

  GET https://api.indexalpha.id/usage

Free plan: 5 req/day, 10 req/minute → cache-first, retry cuma utk transient error.

Health state dicatat utk SETIAP percobaan (sukses & gagal), termasuk saat key
tidak diset. Bedakan HTTP-level error vs logical error (JSON success: false).
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_BASE_URL       = "https://api.indexalpha.id"
_TIMEOUT        = 30       # seconds — total timeout (urllib doesn't support separate connect/read timeout)
_MAX_RETRIES     = 3        # max retry attempts (total attempts = _MAX_RETRIES + 1)
_BASE_BACKOFF    = 1.0      # seconds — initial backoff
_MAX_BACKOFF     = 30.0     # seconds — cap exponential backoff

# Status codes yang aman di-retry (transient)
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Known IDX broker codes → names (subset; extend sesuai kebutuhan)
IDX_BROKER_NAMES: dict[str, str] = {
    "YP": "Indo Premier Sekuritas",
    "CC": "Mandiri Sekuritas",
    "ZP": "Macquarie Sekuritas",
    "AK": "UBS Securities Indonesia",
    "BQ": "BNI Sekuritas",
    "ML": "Merrill Lynch / BofA",
    "MS": "Morgan Stanley Sekuritas",
    "JP": "JPMorgan Securities",
    "DB": "Deutsche Bank",
    "YU": "CLSA Securities Indonesia",
    "CS": "Credit Suisse / UBS",
    "RB": "CGS-CIMB Securities",
    "GS": "Goldman Sachs",
    "QA": "Citigroup Securities",
    "DP": "DBS Vickers Securities",
    "SB": "Nomura Securities",
    "PD": "Indo Premier Online",
    "CP": "Valbury Asia Securities",
    "FZ": "Waterfront Sekuritas",
    "OD": "Mirae Asset Sekuritas",
    "DR": "OSO Sekuritas",
    "LG": "Trimegah Sekuritas",
    "MQ": "Ciptadana Sekuritas",
    "IN": "Investindo Nusantara",
    "KI": "Ciptadana Sekuritas Asia",
    "NI": "BCA Sekuritas",
    "HD": "HD Capital",
    "MU": "Minna Padi Investama",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _clean_ticker(ticker: str) -> str:
    """Strip .JK suffix if present — Index Alpha expects bare codes."""
    return ticker.upper().replace(".JK", "").strip()


def _compute_backoff(attempt: int, rate_limited: bool) -> float:
    """Exponential backoff + jitter.

    Args:
        attempt: current attempt number (1-based).
        rate_limited: True jika karena 429 — backoff lebih agresif.

    Returns:
        Sleep time in seconds.
    """
    base = _BASE_BACKOFF * (2 ** (attempt - 1))
    if rate_limited:
        base *= 2  # double backoff for rate limits
    capped = min(base, _MAX_BACKOFF)
    jitter = random.uniform(0, capped * 0.25)  # up to 25% jitter
    return capped + jitter


# ---------------------------------------------------------------------------
# Health state — persisted record of every real call (success or failure).
# Read by scripts/check_indexalpha_health.py. Never contains the API key or
# response payload content — only status code / latency / error type, which
# are safe to commit and inspect.
# ---------------------------------------------------------------------------

_HEALTH_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "indexalpha_health.json"


def _record_health(success: bool, status_code: int | None, latency_ms: float | None,
                    error_type: str | None) -> None:
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    state = {}
    if _HEALTH_PATH.exists():
        try:
            state = json.loads(_HEALTH_PATH.read_text())
        except Exception:  # noqa: BLE001
            state = {}

    state["last_attempt_at"] = now
    state["last_status_code"] = status_code
    state["last_latency_ms"] = round(latency_ms, 1) if latency_ms is not None else None
    state["last_error_type"] = error_type
    state["total_calls"] = state.get("total_calls", 0) + 1

    if success:
        state["last_success_at"] = now
        state["consecutive_failures"] = 0
        state["total_successes"] = state.get("total_successes", 0) + 1
    else:
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        state["total_failures"] = state.get("total_failures", 0) + 1

    try:
        _HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HEALTH_PATH.write_text(json.dumps(state, indent=2))
    except Exception as exc:  # noqa: BLE001
        # Health tracking must never break the actual data fetch.
        logger.debug("IndexAlpha health-state write failed (non-fatal): {}", exc)


def _get_api_key() -> str:
    key = os.environ.get("INDEX_ALPHA_API_KEY", "").strip()
    if not key:
        # Fallback: check Streamlit secrets (Streamlit Cloud deployment)
        try:
            import streamlit as _st
            key = str(_st.secrets.get("INDEX_ALPHA_API_KEY", "")).strip()
        except Exception:
            pass
    if not key:
        raise EnvironmentError(
            "INDEX_ALPHA_API_KEY tidak ditemukan di environment. "
            "Set dengan: export INDEX_ALPHA_API_KEY='your_token_here' "
            "atau di Streamlit secrets sebagai INDEX_ALPHA_API_KEY"
        )
    return key


def _broker_cache_path(ticker: str, trade_date: str, broker_dir: Path) -> Path:
    """Canonical cache path — consistent dengan broker_summary.py."""
    clean = _clean_ticker(ticker)
    return broker_dir / f"{clean}.JK_{trade_date}.parquet"


# ---------------------------------------------------------------------------
# Normalize response → DataFrame
# ---------------------------------------------------------------------------

def _normalize_response(data: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw API response list → normalized DataFrame.

    Backward-compatible dengan broker_summary.py schema:
      broker_code, broker_name, buy_lot, sell_lot, net_lot

    Plus enriched columns only available from real data:
      buy_value, sell_value, net_value,
      buy_avg_price, sell_avg_price, buy_freq, sell_freq
    """
    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        code     = str(item.get("code", "")).upper().strip()
        buy_vol  = float(item.get("buy_volume",  0) or 0)
        sell_vol = float(item.get("sell_volume", 0) or 0)
        buy_val  = float(item.get("buy_value",   0) or 0)
        sell_val = float(item.get("sell_value",  0) or 0)

        rows.append({
            # ── backward-compat columns ──────────────────────────────────
            "broker_code":     code,
            "broker_name":     IDX_BROKER_NAMES.get(code, "Unknown"),
            "buy_lot":         buy_vol,
            "sell_lot":        sell_vol,
            "net_lot":         buy_vol - sell_vol,
            # ── enriched columns (not in mock data) ─────────────────────
            "buy_value":       buy_val,
            "sell_value":      sell_val,
            "net_value":       buy_val  - sell_val,
            "buy_avg_price":   float(item.get("buy_avg",   0) or 0),
            "sell_avg_price":  float(item.get("sell_avg",  0) or 0),
            "buy_freq":        int(item.get("buy_freq",    0) or 0),
            "sell_freq":       int(item.get("sell_freq",   0) or 0),
        })

    df = pd.DataFrame(rows)
    # Sort by absolute net_lot (largest mover first) — consistent dengan mock
    df = df.sort_values("net_lot", key=abs, ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Low-level HTTP call
# ---------------------------------------------------------------------------

def _get(path: str, params: dict, api_key: str) -> dict:
    """Make a GET request to the Index Alpha API with retry + exponential backoff.

    Retry policy:
      - 401 (auth)       → langsung raise, TIDAK di-retry
      - 403 (forbidden)  → langsung raise, TIDAK di-retry
      - 400, 404, 422    → langsung raise (client error, bukan transient)
      - 429 (rate limit) → retry dengan exponential backoff + jitter
      - 5xx              → retry dengan exponential backoff + jitter
      - timeout          → retry dengan exponential backoff + jitter
      - connection error → retry dengan exponential backoff + jitter

    Health state dicatat SETIAP attempt (sukses & gagal) — termasuk logical
    failure (JSON success=false) yang TIDAK dicatat sebagai sukses.

    Returns:
      Parsed JSON response dict (sudah diverifikasi `success=True` di body).
    """
    url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "application/json",
    }

    timeout = _TIMEOUT

    for attempt in range(1, _MAX_RETRIES + 2):
        start = time.monotonic()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                status = resp.status
            latency_ms = (time.monotonic() - start) * 1000
            parsed = json.loads(body)

            # ── Logical check: HTTP 200 !== success ─────────────────────
            if not parsed.get("success"):
                err_msg = parsed.get("error") or "unknown error (logical failure)"
                n_items = len(parsed.get("data", [])) if isinstance(parsed.get("data"), list) else 0
                logger.warning(
                    "IndexAlpha {} {} -> 200 logical_fail (items={}, error={})",
                    path, params.get("ticker", ""), n_items, err_msg,
                )
                _record_health(
                    success=False, status_code=status, latency_ms=latency_ms,
                    error_type="logical_failure",
                )
                raise RuntimeError(
                    f"Index Alpha API — logical failure: {err_msg}"
                )

            # ── Success path ────────────────────────────────────────────
            n_items = len(parsed.get("data", [])) if isinstance(parsed.get("data"), list) else 0
            logger.info(
                "IndexAlpha {} {} -> {} in {:.0f}ms ({} items)",
                path, params.get("ticker", ""), status, latency_ms, n_items,
            )
            _record_health(success=True, status_code=status, latency_ms=latency_ms, error_type=None)
            return parsed

        except urllib.error.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            code = exc.code

            # ── Non-retryable client errors ──────────────────────────
            if code in (401, 403):
                err_type = "auth_error" if code == 401 else "forbidden"
                _record_health(success=False, status_code=code, latency_ms=latency_ms, error_type=err_type)
                raise PermissionError(
                    f"Index Alpha API — {code}. Periksa INDEX_ALPHA_API_KEY."
                ) from exc

            if code in (400, 404, 405, 422):
                _record_health(success=False, status_code=code, latency_ms=latency_ms, error_type="client_error")
                raise RuntimeError(
                    f"Index Alpha API — HTTP {code} (client error). Ticker/parameter mungkin invalid."
                ) from exc

            # ── Retryable: 429, 5xx ──────────────────────────────────
            if code in _RETRYABLE_STATUSES:
                if attempt <= _MAX_RETRIES:
                    backoff = _compute_backoff(attempt, code == 429)
                    err_type = "rate_limit" if code == 429 else f"http_{code}"
                    _record_health(success=False, status_code=code, latency_ms=latency_ms, error_type=err_type)
                    logger.warning(
                        "IndexAlpha {} (attempt {}/{}, {}ms). "
                        "Retry in {:.1f}s ...",
                        err_type, attempt, _MAX_RETRIES + 1, latency_ms, backoff,
                    )
                    time.sleep(backoff)
                    continue
                _record_health(success=False, status_code=code, latency_ms=latency_ms, error_type=f"{err_type}_exhausted")
                raise RuntimeError(
                    f"Index Alpha API — {code} setelah {_MAX_RETRIES}x retry. "
                    "Server sibuk atau kuota habis."
                ) from exc

            # ── Other HTTP errors ────────────────────────────────────
            _record_health(success=False, status_code=code, latency_ms=latency_ms, error_type="http_error")
            raise RuntimeError(
                f"Index Alpha API — HTTP {code} pada {url.split('?')[0]}"
            ) from exc

        except (urllib.error.URLError, OSError, ConnectionError) as exc:
            latency_ms = (time.monotonic() - start) * 1000
            if attempt <= _MAX_RETRIES:
                backoff = _compute_backoff(attempt, rate_limited=False)
                _record_health(success=False, status_code=None, latency_ms=latency_ms, error_type="connection_error")
                logger.warning(
                    "IndexAlpha connection error (attempt {}/{}, {}ms): {}. "
                    "Retry in {:.1f}s ...",
                    attempt, _MAX_RETRIES + 1, latency_ms, exc, backoff,
                )
                time.sleep(backoff)
                continue
            _record_health(success=False, status_code=None, latency_ms=latency_ms, error_type="connection_exhausted")
            raise ConnectionError(
                f"Index Alpha API — koneksi gagal setelah {_MAX_RETRIES}x retry: {exc}"
            ) from exc

        except TimeoutError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            if attempt <= _MAX_RETRIES:
                backoff = _compute_backoff(attempt, rate_limited=False)
                _record_health(success=False, status_code=None, latency_ms=latency_ms, error_type="timeout")
                logger.warning(
                    "IndexAlpha timeout (attempt {}/{}, {}ms). Retry in {:.1f}s ...",
                    attempt, _MAX_RETRIES + 1, latency_ms, backoff,
                )
                time.sleep(backoff)
                continue
            _record_health(success=False, status_code=None, latency_ms=latency_ms, error_type="timeout_exhausted")
            raise TimeoutError(
                f"Index Alpha API — timeout setelah {_MAX_RETRIES}x retry "
                f"(timeout={_TIMEOUT}s)."
            ) from exc

        except json.JSONDecodeError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            _record_health(success=False, status_code=None, latency_ms=latency_ms, error_type="parse_error")
            raise RuntimeError(
                f"Index Alpha API — response tidak valid JSON: {exc}"
            ) from exc

    # ── All retries exhausted ────────────────────────────────────────────
    _record_health(success=False, status_code=None, latency_ms=None, error_type="exhausted_retries")
    raise RuntimeError("Index Alpha API — semua retries gagal.")


# ---------------------------------------------------------------------------
# Public fetcher class
# ---------------------------------------------------------------------------

class IndexAlphaFetcher:
    """Fetcher nyata untuk broker summary via Index Alpha API.

    Extends implicit interface dari BaseBrokerFetcher (tidak perlu import).
    Compatible dengan get_broker_summary() di broker_summary.py.

    Usage:
        fetcher = IndexAlphaFetcher()           # reads INDEX_ALPHA_API_KEY
        df      = fetcher.fetch("BBCA.JK", "2026-05-29")
        # → DataFrame with broker_code, buy_lot, sell_lot, net_lot, ...

    Quota-safe usage:
        df_range = fetcher.fetch_range("BBCA.JK", "2026-05-12", "2026-05-29")
        # → 1 request, returns ~15 trading days aggregated per broker
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Args:
            api_key: API key string. If None, reads INDEX_ALPHA_API_KEY env var.
        """
        self._api_key: str | None = api_key  # lazy-resolved on first call

    def _key(self) -> str:
        if self._api_key:
            return self._api_key
        self._api_key = _get_api_key()
        return self._api_key

    # ── single-day fetch (backward-compatible interface) ─────────────────

    def fetch(
        self,
        ticker: str,
        trade_date: str,
        investor: str = "all",
        market: str   = "RG",
    ) -> pd.DataFrame:
        """Fetch broker summary untuk satu ticker pada satu tanggal.

        Consumes 1 API request.

        Args:
            ticker     : "BBCA.JK" atau "BBCA" — keduanya diterima.
            trade_date : "YYYY-MM-DD"
            investor   : "all" | "f" | "or" | "d"
            market     : "RG" (Regular) | "NG" | "ALL"

        Returns:
            DataFrame dengan kolom broker_code … sell_freq.
            Empty DataFrame jika tidak ada data.
        """
        clean = _clean_ticker(ticker)
        params = {
            "ticker":   clean,
            "from":     trade_date,
            "to":       trade_date,
            "investor": investor,
            "market":   market,
        }

        logger.info(
            "IndexAlpha fetch: {} @ {} (investor={}, market={})",
            clean, trade_date, investor, market,
        )

        # ── Resolve API key — record health even when key is missing ───
        try:
            key = self._key()
        except EnvironmentError as exc:
            _record_health(success=False, status_code=None, latency_ms=None, error_type="missing_key")
            logger.error("IndexAlpha fetch gagal untuk {} @ {}: {}", clean, trade_date, exc)
            return pd.DataFrame()

        try:
            resp = _get("/stocks/broker-summary", params, key)
        except PermissionError as exc:
            # 401/403 already recorded by _get() — just log and return empty
            logger.error("IndexAlpha auth gagal untuk {} @ {}: {}", clean, trade_date, exc)
            return pd.DataFrame()
        except (RuntimeError, ConnectionError, TimeoutError) as exc:
            logger.error("IndexAlpha fetch gagal untuk {} @ {}: {}", clean, trade_date, exc)
            return pd.DataFrame()

        # Note: logical failure (success=false) already raises RuntimeError
        # inside _get(), so this path is only reached on success.
        data = resp.get("data") or []
        df   = _normalize_response(data)

        if df.empty:
            logger.info("IndexAlpha: tidak ada data broker untuk {} @ {}", clean, trade_date)
        else:
            logger.info(
                "IndexAlpha: {} broker records untuk {} @ {}  "
                "(net_lot range: {:.0f} … {:.0f})",
                len(df), clean, trade_date,
                df["net_lot"].min(), df["net_lot"].max(),
            )

        return df

    # ── date-range fetch (1 request for multiple days) ───────────────────

    def fetch_range(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        investor: str = "all",
        market: str   = "RG",
    ) -> pd.DataFrame:
        """Fetch broker summary aggregated over a date range.

        Consumes 1 API request (very quota-efficient for multi-day intel).
        The API aggregates across the requested period — no per-day breakdown.

        Args:
            ticker    : IDX ticker (with or without .JK)
            from_date : start date "YYYY-MM-DD"
            to_date   : end date   "YYYY-MM-DD"
            investor  : "all" | "f" | "or" | "d"
            market    : "RG" | "NG" | "ALL"

        Returns:
            DataFrame with broker activity aggregated over [from_date, to_date].
            Column "date" is not present (range aggregate).
        """
        clean = _clean_ticker(ticker)
        params = {
            "ticker":   clean,
            "from":     from_date,
            "to":       to_date,
            "investor": investor,
            "market":   market,
        }

        logger.info(
            "IndexAlpha fetch_range: {} [{} → {}] (investor={}, market={})",
            clean, from_date, to_date, investor, market,
        )

        # ── Resolve API key — record health even when key is missing ───
        try:
            key = self._key()
        except EnvironmentError as exc:
            _record_health(success=False, status_code=None, latency_ms=None, error_type="missing_key")
            logger.error(
                "IndexAlpha fetch_range gagal untuk {} [{} → {}]: {}",
                clean, from_date, to_date, exc,
            )
            return pd.DataFrame()

        try:
            resp = _get("/stocks/broker-summary", params, key)
        except PermissionError as exc:
            logger.error(
                "IndexAlpha auth gagal untuk {} [{} → {}]: {}",
                clean, from_date, to_date, exc,
            )
            return pd.DataFrame()
        except (RuntimeError, ConnectionError, TimeoutError) as exc:
            logger.error(
                "IndexAlpha fetch_range gagal untuk {} [{} → {}]: {}",
                clean, from_date, to_date, exc,
            )
            return pd.DataFrame()

        # Note: logical failure already raises RuntimeError in _get()
        data = resp.get("data") or []
        df   = _normalize_response(data)

        if not df.empty:
            df["from_date"] = from_date
            df["to_date"]   = to_date
            logger.info(
                "IndexAlpha: {} broker records untuk {} [{} → {}]",
                len(df), clean, from_date, to_date,
            )

        return df

    # ── quota check ─────────────────────────────────────────────────────

    def check_usage(self) -> dict:
        """Cek sisa kuota API.  Consumes 1 request.

        Returns:
            {
              "api_key":       str,
              "monthly_limit": int,
              "current_usage": int,
              "remaining":     int,
              "reset_date":    str,
            }
            Empty dict jika gagal.
        """
        try:
            key = self._key()
        except EnvironmentError:
            return {}

        try:
            resp = _get("/usage", {}, key)
            usage = resp.get("data", {})
            logger.info(
                "IndexAlpha usage: {}/{} used, {} remaining (reset: {})",
                usage.get("current_usage"),
                usage.get("monthly_limit"),
                usage.get("remaining"),
                usage.get("reset_date"),
            )
            return usage
        except Exception as exc:
            logger.error("IndexAlpha check_usage gagal: {}", exc)
        return {}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def fetch_with_cache(
    ticker: str,
    trade_date: str,
    broker_dir: Path,
    fetcher: IndexAlphaFetcher | None = None,
    investor: str = "all",
    market: str   = "RG",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch broker summary dengan cache parquet harian.

    Jika cache sudah ada dan force_refresh=False → TIDAK hit API.
    Hemat kuota: 1 ticker = 1 file cache = 0 request berikutnya.

    Cache path: data/broker/{TICKER}.JK_{YYYY-MM-DD}.parquet
    Compatible dengan load_broker_summary() di broker_summary.py.

    Args:
        ticker       : IDX ticker
        trade_date   : "YYYY-MM-DD"
        broker_dir   : Path ke data/broker/
        fetcher      : IndexAlphaFetcher instance (dibuat baru jika None)
        investor     : "all" | "f" | "or" | "d"
        market       : "RG" | "NG" | "ALL"
        force_refresh: True = selalu hit API meskipun cache ada

    Returns:
        DataFrame atau empty DataFrame jika API gagal dan tidak ada cache.
    """
    cache_path = _broker_cache_path(ticker, trade_date, broker_dir)

    if cache_path.exists() and not force_refresh:
        logger.debug(
            "Cache hit: {} (skipping API call)", cache_path.name
        )
        return pd.read_parquet(cache_path)

    if fetcher is None:
        fetcher = IndexAlphaFetcher()

    df = fetcher.fetch(ticker, trade_date, investor=investor, market=market)

    if not df.empty:
        broker_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.debug("Cache saved: {}", cache_path.name)

    return df


def fetch_batch_top_tickers(
    tickers: list[str],
    trade_date: str,
    broker_dir: Path,
    fetcher: IndexAlphaFetcher | None = None,
    max_tickers: int = 5,
    request_delay: float = 6.5,
    investor: str = "all",
    market: str   = "RG",
) -> dict[str, pd.DataFrame]:
    """Fetch broker summary untuk batch kecil top tickers dengan cache.

    Rancangan hemat kuota:
      - max_tickers default 5 (= free plan daily limit)
      - skip ticker yang sudah ada di cache
      - request_delay 6.5s antar call (aman di bawah 10 req/min free plan)

    Panggil hanya untuk tickers BREAKOUT / PRE_MARKUP terpilih, bukan seluruh universe.

    Args:
        tickers       : list ticker (mis. ["BBCA.JK", "TLKM.JK"])
        trade_date    : "YYYY-MM-DD"
        broker_dir    : Path ke data/broker/
        fetcher       : IndexAlphaFetcher instance
        max_tickers   : batas ticker yang di-fetch (default 5)
        request_delay : detik antar request (default 6.5s → ~9 req/min, aman)
        investor      : "all" | "f" | "or" | "d"
        market        : "RG" | "NG" | "ALL"

    Returns:
        {ticker: DataFrame}  — hanya ticker yang berhasil di-fetch / di-cache
    """
    if fetcher is None:
        fetcher = IndexAlphaFetcher()

    results: dict[str, pd.DataFrame] = {}
    api_calls = 0

    for ticker in tickers:
        if len(results) >= max_tickers:
            logger.info(
                "fetch_batch: batas {} ticker tercapai — stop.", max_tickers
            )
            break

        cache_path = _broker_cache_path(ticker, trade_date, broker_dir)

        # Cache hit — no API call needed
        if cache_path.exists():
            logger.debug("Cache hit: {}", cache_path.name)
            results[ticker] = pd.read_parquet(cache_path)
            continue

        # Rate-limit guard (kecuali request pertama)
        if api_calls > 0:
            logger.debug("Rate-limit delay: {:.1f}s ...", request_delay)
            time.sleep(request_delay)

        df = fetcher.fetch(ticker, trade_date, investor=investor, market=market)
        api_calls += 1

        if not df.empty:
            broker_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            results[ticker] = df
            logger.info(
                "Batch fetched: {}  ({} broker records)", ticker, len(df)
            )
        else:
            logger.warning("Batch: tidak ada data untuk {}", ticker)

    logger.info(
        "fetch_batch selesai: {} ticker, {} API call(s), {} cache hit(s).",
        len(results),
        api_calls,
        len(results) - api_calls,
    )
    return results


# ---------------------------------------------------------------------------
# Quick manual test (jalankan langsung untuk verifikasi API key)
# ---------------------------------------------------------------------------

def _manual_test(ticker: str = "BBCA", trade_date: str | None = None) -> None:
    """Test satu request. Jalankan dengan:
        python -m stock_scanner.pipeline.fetch_indexalpha
    atau:
        python stock_scanner/pipeline/fetch_indexalpha.py
    """
    if trade_date is None:
        # Pakai hari kerja terakhir sebagai default
        from stock_scanner.utils.trading_calendar import last_trading_day
        trade_date = last_trading_day(date.today()).strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print("Index Alpha API — Manual Test")
    print(f"Ticker: {ticker}  Date: {trade_date}")
    print(f"{'='*55}")

    try:
        fetcher = IndexAlphaFetcher()
        print("\n[1] Cek usage...")
        usage = fetcher.check_usage()
        if usage:
            print(f"  Remaining requests: {usage.get('remaining', '?')}")
            print(f"  Reset date:         {usage.get('reset_date', '?')}")
        else:
            print("  Usage check failed (non-fatal)")

        print(f"\n[2] Fetch broker summary: {ticker} @ {trade_date}")
        df = fetcher.fetch(ticker, trade_date)
        if df.empty:
            print("  No data returned.")
        else:
            print(f"  {len(df)} broker records:\n")
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 120)
            pd.set_option("display.float_format", "{:,.0f}".format)
            print(df[["broker_code", "broker_name", "buy_lot", "sell_lot",
                       "net_lot", "buy_value", "sell_value", "net_value"]].head(10).to_string(index=False))

    except EnvironmentError as exc:
        print(f"\n  ❌ {exc}")
        print("\n  Set your API key:")
        print("    export INDEX_ALPHA_API_KEY='your_token_here'")
        print("  Then re-run.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Index Alpha API manual test")
    p.add_argument("--ticker", default="BBCA", help="IDX ticker (default: BBCA)")
    p.add_argument("--date",   default=None,   help="Trade date YYYY-MM-DD (default: last trading day)")
    args = p.parse_args()
    _manual_test(args.ticker, args.date)
