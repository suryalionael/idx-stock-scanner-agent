# Daily Automation — IDX Scanner → Telegram

Every weekday morning the scanner runs automatically, generates signals, and
posts a summary to the Telegram group. This doc covers the entrypoints, the
GitHub Actions schedule, the required secrets, and how to verify it works.

---

## TL;DR

- **Scheduler:** GitHub Actions, `.github/workflows/morning-alert.yml`
- **When:** 08:00 WIB (01:00 UTC), Monday–Friday
- **Target group:** `-1003764018733` (default fallback if the chat-id secret is unset)
- **Local one-command run:** `python scripts/run_daily_signal.py`

---

## Entrypoints

There are two non-UI entrypoints. Both are safe to run from a terminal or CI.

| Command | What it does | Use when |
|---|---|---|
| `python -m stock_scanner.alerts.telegram_alert` | Build + send the morning message from **existing** scan data | The scan already ran (this is what the GitHub workflow calls after its scan step) |
| `python scripts/run_daily_signal.py` | **Scan + send** in one command | You want a single local command that does everything |

Both share the same message builder and sender (`telegram_alert.py`), which uses
only `pandas` + the standard-library `urllib` — no heavy dependencies, so the
daily job has minimal failure surface.

### Useful flags

```bash
# Print the message instead of sending (no Telegram call)
python -m stock_scanner.alerts.telegram_alert --dry-run

# Report a specific date
python -m stock_scanner.alerts.telegram_alert --date 2026-06-02

# One-command scan + send, but don't send if there are no priority picks
python scripts/run_daily_signal.py --skip-if-empty

# Send using cached data only (skip the scan)
python scripts/run_daily_signal.py --skip-scan
```

### Exit codes

`run_daily_signal.py`:

| Code | Meaning |
|---|---|
| 0 | Sent, dry-run, or intentionally skipped |
| 1 | Failed to send (e.g. `TELEGRAM_BOT_TOKEN` missing, Telegram API error) |
| 2 | No scan data available at all |

A scan failure is **non-fatal**: the alert still goes out using the most recent
cached scan (up to 7 days back), so you always get a morning message.

---

## Environment variables

| Variable | Required? | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | **Yes** | The actual secret. From @BotFather. |
| `TELEGRAM_CHAT_ID` | No | Target chat/group. If unset, falls back to the default group `-1003764018733`. |
| `ANTHROPIC_API_KEY` | No | Only used if the explain agent is enabled in `scanner_config.yaml`. |

### Local setup (`.env.alerts`)

A gitignored `.env.alerts` file at the repo root holds your credentials:

```bash
TELEGRAM_BOT_TOKEN=123456:AA...your-token...
TELEGRAM_CHAT_ID=-1003764018733     # group; use 7585400125 for your personal DM
```

Load it before running locally:

```bash
set -a && source .env.alerts && set +a
python scripts/run_daily_signal.py --dry-run
```

> `.env.alerts` is in `.gitignore` — never commit it.

---

## GitHub Actions setup

### 1. Add repository secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | your bot token (required) |
| `TELEGRAM_CHAT_ID` | `-1003764018733` (optional — code falls back to this group if omitted) |
| `ANTHROPIC_API_KEY` | optional |

### 2. Schedule

`.github/workflows/morning-alert.yml` runs on:

```yaml
on:
  schedule:
    - cron: "0 1 * * 1-5"   # 08:00 WIB (01:00 UTC), Mon–Fri
  workflow_dispatch:        # manual trigger from the Actions tab
```

GitHub cron is always **UTC**. WIB = UTC+7, so 08:00 WIB = 01:00 UTC.
The alert lands before the 09:00 WIB market open.

### Job flow

1. Checkout + Python 3.11
2. `pip install -e ".[scanner]"` — installs yfinance / ta / scikit-learn /
   xgboost (needed by the scan). The dashboard extra (streamlit/plotly) is
   **not** installed to keep the job lean.
3. Restore `data/` cache (keyed on the universe file hash)
4. Run `python -m stock_scanner.pipeline.run_daily_scan`
5. Save `data/` cache
6. Send alert (`if: always()` — sent even if the scan failed, using cached data)
7. Upload the ranked CSV as an artifact (7-day retention)

---

## Verification

### A. Local verification

```bash
# 1. Dry-run (no send) — confirms data + message build
python scripts/run_daily_signal.py --skip-scan --dry-run

# 2. Real send to the group — confirms the bot can post
set -a && source .env.alerts && set +a
python -m stock_scanner.alerts.telegram_alert --date <latest-date>
# → look for: "Telegram: message sent (chat_id=-1003764018733)"
# → check the Telegram group for the message
```

### B. Workflow verification (before relying on the schedule)

1. Push these changes to `main`.
2. Add the secrets (above).
3. Go to **Actions → Morning Alert (IDX Scanner) → Run workflow**.
   - First do a **dry run** (set the `dry_run` input to `true`) — this runs the
     scan and prints the message in the logs without sending.
   - Then run it again with `dry_run=false` to confirm a real message arrives.
4. Once a manual run posts to the group, the daily 08:00 WIB schedule will do
   the same automatically.

> Note: scheduled workflows only run from the **default branch** (`main`), and
> GitHub may delay scheduled jobs by a few minutes under load.

---

## Behaviour when there are no signals

By default the job **always sends** a message — even with no BREAKOUT/PRE_MARKUP
picks it posts the signal distribution plus a watchlist, so you get daily
confirmation the job ran. To skip sending on empty days, pass `--skip-if-empty`
(only on `run_daily_signal.py` / `telegram_alert`). The scheduled workflow uses
the default (always send).

---

## Limitations

- **First scheduled run is slow.** A cold `data/` cache downloads full history
  for ~950 tickers via yfinance; the job allows up to 120 min. Subsequent runs
  are incremental and fast.
- **Single summary message.** This automation sends the compact morning summary
  (distribution + top picks). The richer multi-message deep-dive path
  (`stock_scanner.alerts.runner` → `daily_alert`) is intentionally not wired
  into the schedule to keep the daily job reliable.
- **No per-signal dedupe across runs.** Each run sends one message. Re-running
  the workflow the same day posts again (expected for manual re-runs).
- **Broker data is not part of the morning alert.** Broker Analytics remains a
  dashboard feature.
