#!/usr/bin/env python3
"""IndexAlpha integration health check.

Quota-safe by default: every check below is read-only against local state
(env var presence, the persisted health-state file written by
fetch_indexalpha._get() on every real call, and on-disk cache freshness) —
zero network calls, zero quota cost. The free plan is 5 requests/day; this
script must never spend one without the caller explicitly asking for it.

--live makes EXACTLY ONE real API call (check_usage()) to prove end-to-end
connectivity. Off by default — never run automatically in CI/cron. A human
decides when it's worth spending 1 of the 5 daily requests on verification.

Exit code: 0 = healthy, 1 = degraded (see printed reasons), 2 = misconfigured.

Usage:
    python scripts/check_indexalpha_health.py
    python scripts/check_indexalpha_health.py --live
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

_HEALTH_PATH = repo_root / "data" / "published" / "indexalpha_health.json"
_BROKER_DIR = repo_root / "data" / "broker"
_STALE_AFTER_DAYS = 3          # broker data older than this is flagged stale
_FAILURE_ALERT_THRESHOLD = 3   # consecutive failures before "FAIL" verdict


def check_key_configured() -> bool:
    """Bool only — never read or print the actual key value."""
    return bool(os.environ.get("INDEX_ALPHA_API_KEY", "").strip())


def read_health_state() -> dict:
    if not _HEALTH_PATH.exists():
        return {}
    try:
        return json.loads(_HEALTH_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def check_cache_freshness() -> dict:
    """Newest broker cache file's embedded date and age in days."""
    if not _BROKER_DIR.exists():
        return {"newest_date": None, "age_days": None, "n_files": 0}
    files = list(_BROKER_DIR.glob("*.parquet"))
    dates = []
    for f in files:
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                dates.append(datetime.strptime(parts[1], "%Y-%m-%d").date())
            except ValueError:
                continue
    if not dates:
        return {"newest_date": None, "age_days": None, "n_files": len(files)}
    newest = max(dates)
    age_days = (datetime.now(timezone.utc).date() - newest).days
    return {"newest_date": str(newest), "age_days": age_days, "n_files": len(files)}


def run_live_check() -> dict:
    """Spends exactly 1 of the 5 daily quota requests. Caller must opt in."""
    from stock_scanner.pipeline.fetch_indexalpha import IndexAlphaFetcher
    try:
        usage = IndexAlphaFetcher().check_usage()
        return {"ok": bool(usage), "usage": usage}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main(live: bool) -> int:
    print("=" * 60)
    print("IndexAlpha Integration Health Check")
    print("=" * 60)

    key_configured = check_key_configured()
    print(f"\n[1] API key configured: {'YES' if key_configured else 'NO'}")
    if not key_configured:
        print("    INDEX_ALPHA_API_KEY not set in this environment.")
        print("    Note: a GitHub repo secret of this name exists (configured")
        print("    2026-06-04 per `gh secret list`), but NO workflow currently")
        print("    passes it into env — so it has no effect in CI. Whether the")
        print("    deployed Streamlit Cloud app has it set cannot be checked")
        print("    from here; verify in the Streamlit Cloud secrets panel.")

    state = read_health_state()
    print(f"\n[2] Recorded call history ({_HEALTH_PATH.relative_to(repo_root)}):")
    if not state:
        print("    No recorded calls yet — fetch_indexalpha._get() has never")
        print("    been invoked successfully or unsuccessfully in this checkout.")
    else:
        print(f"    last_attempt_at:       {state.get('last_attempt_at')}")
        print(f"    last_success_at:       {state.get('last_success_at')}")
        print(f"    consecutive_failures:  {state.get('consecutive_failures')}")
        print(f"    last_status_code:      {state.get('last_status_code')}")
        print(f"    last_latency_ms:       {state.get('last_latency_ms')}")
        print(f"    last_error_type:       {state.get('last_error_type')}")
        total = state.get("total_calls", 0)
        succ = state.get("total_successes", 0)
        print(f"    lifetime success rate: {succ}/{total} "
              f"({succ/total*100:.0f}%)" if total else "    lifetime success rate: n/a")

    cache = check_cache_freshness()
    print(f"\n[3] Broker cache freshness ({_BROKER_DIR.relative_to(repo_root)}):")
    print(f"    files cached:  {cache['n_files']}")
    print(f"    newest date:   {cache['newest_date']}")
    print(f"    age (days):    {cache['age_days']}")
    cache_stale = cache["age_days"] is not None and cache["age_days"] > _STALE_AFTER_DAYS

    live_result = None
    if live:
        print(f"\n[4] LIVE check requested — spending 1 of 5 daily quota requests...")
        live_result = run_live_check()
        if live_result["ok"]:
            print(f"    OK — usage: {live_result['usage']}")
        else:
            print(f"    FAILED — {live_result.get('error')}")

    # --- Verdict ---
    print("\n" + "=" * 60)
    reasons = []
    if not key_configured:
        reasons.append("API key not configured in this environment")
    if state.get("consecutive_failures", 0) >= _FAILURE_ALERT_THRESHOLD:
        reasons.append(f"{state['consecutive_failures']} consecutive failures recorded")
    if cache_stale:
        reasons.append(f"broker cache is {cache['age_days']} days old (>{_STALE_AFTER_DAYS}d threshold)")
    if live_result is not None and not live_result["ok"]:
        reasons.append("live connectivity check failed")
    if not state and not live_result:
        reasons.append("no recorded successful call, ever, in this checkout")

    if not key_configured and not state:
        print("VERDICT: MISCONFIGURED — integration has likely never run successfully here.")
        code = 2
    elif reasons:
        print("VERDICT: DEGRADED")
        for r in reasons:
            print(f"  - {r}")
        code = 1
    else:
        print("VERDICT: HEALTHY")
        code = 0
    print("=" * 60)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Spend 1 of the 5 daily quota requests on a real connectivity check.")
    args = parser.parse_args()
    sys.exit(main(args.live))
