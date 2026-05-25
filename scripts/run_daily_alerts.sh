#!/usr/bin/env bash
# =============================================================================
# IDX Stock Scanner — Daily Alert Runner
# =============================================================================
# Called by launchd (macOS scheduler) every trading day.
# Can also be run manually for testing.
#
# Usage:
#   ./scripts/run_daily_alerts.sh              # normal run
#   ./scripts/run_daily_alerts.sh --dry-run    # test without sending Telegram
#   ./scripts/run_daily_alerts.sh --skip-scan  # alert only (use existing data)
#
# Requirements:
#   - .env.alerts file in REPO_DIR with TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
#   - Python venv at .venv/
#   - All pipeline deps installed in venv
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

# Absolute path to the repo (resolve from this script's location)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
ENV_FILE="$REPO_DIR/.env.alerts"

DATE="$(date +%Y-%m-%d)"
LOG_DIR="$REPO_DIR/logs/daily"
LOG_FILE="$LOG_DIR/$DATE.log"
LOCK_FILE="/tmp/idx_scanner_daily_${DATE}.lock"

# ── Colours for terminal output ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

# ── Logging helpers ────────────────────────────────────────────────────────────

_log() { echo "[$(date '+%H:%M:%S')] $*"; }
_info()  { _log "INFO  $*"; }
_warn()  { _log "WARN  $*" >&2; }
_error() { _log "ERROR $*" >&2; }

# ── Lock file (anti double-run) ────────────────────────────────────────────────

_acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            _error "Job already running (PID $OLD_PID). Lock: $LOCK_FILE"
            _error "If this is stale, remove it with: rm $LOCK_FILE"
            exit 1
        else
            _warn "Stale lock found (PID $OLD_PID not running). Removing."
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
    _info "Lock acquired (PID $$): $LOCK_FILE"
}

_release_lock() {
    rm -f "$LOCK_FILE"
    _info "Lock released."
}

# Always clean up lock on exit (normal or error)
trap '_release_lock' EXIT

# ── Error handler ──────────────────────────────────────────────────────────────

_on_error() {
    local EXIT_CODE=$?
    local LINE=$1
    _error "Script failed at line $LINE (exit code $EXIT_CODE)."

    # Send Telegram failure alert if credentials available
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        "$VENV_PYTHON" - <<PYEOF 2>/dev/null || true
import os, urllib.request, urllib.parse, json
token = os.environ['TELEGRAM_BOT_TOKEN']
chat  = os.environ['TELEGRAM_CHAT_ID']
msg   = (
    "🚨 <b>IDX Scanner — Job Gagal</b>\n"
    f"Tanggal : $DATE\n"
    f"Baris   : $LINE\n"
    f"Exit    : $EXIT_CODE\n"
    "<i>Cek: $LOG_FILE</i>"
)
payload = json.dumps({
    "chat_id": chat, "text": msg,
    "parse_mode": "HTML", "disable_web_page_preview": True,
}).encode()
req = urllib.request.Request(
    f"https://api.telegram.org/bot{token}/sendMessage",
    data=payload, headers={"Content-Type": "application/json"}, method="POST"
)
urllib.request.urlopen(req, timeout=10)
PYEOF
    fi
}

trap '_on_error $LINENO' ERR

# ── Parse arguments ────────────────────────────────────────────────────────────

DRY_RUN=""
SKIP_SCAN=""
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN="--dry-run" ;;
        --skip-scan)  SKIP_SCAN="--skip-scan" ;;
        *)            EXTRA_ARGS+=("$arg") ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect all output to log file AND terminal
exec > >(tee -a "$LOG_FILE") 2>&1

_info "=================================================="
_info "IDX Scanner Daily Runner — $DATE"
[ -n "$DRY_RUN"   ] && _info "  Mode: DRY-RUN (no Telegram)"
[ -n "$SKIP_SCAN" ] && _info "  Mode: SKIP-SCAN (alert-only)"
_info "  Repo   : $REPO_DIR"
_info "  Log    : $LOG_FILE"
_info "  Lock   : $LOCK_FILE"
_info "=================================================="

# ── Lock ──────────────────────────────────────────────────────────────────────

_acquire_lock

# ── Environment variables ──────────────────────────────────────────────────────

if [ -f "$ENV_FILE" ]; then
    _info "Loading env vars from $ENV_FILE..."
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
    _info "  Loaded OK."
else
    _warn ".env.alerts not found at $ENV_FILE"
    _warn "Expecting TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment."
fi

# Quick sanity check for required vars
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
    _error "TELEGRAM_BOT_TOKEN is not set. Aborting."
    exit 3
fi
if [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
    _error "TELEGRAM_CHAT_ID is not set. Aborting."
    exit 3
fi

# Warn (but don't abort) if AI key missing
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    _warn "ANTHROPIC_API_KEY not set — AI summary will use rule-based fallback."
fi

# ── Python / venv check ────────────────────────────────────────────────────────

if [ ! -f "$VENV_PYTHON" ]; then
    _error "Python not found at $VENV_PYTHON"
    _error "Create venv with: python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 3
fi

PYTHON_VERSION=$("$VENV_PYTHON" --version 2>&1)
_info "Python: $PYTHON_VERSION ($VENV_PYTHON)"

# ── Run pipeline ───────────────────────────────────────────────────────────────

START_TS=$(date +%s)

_info "--------------------------------------------------"
_info "Running IDX Scanner pipeline..."
_info "--------------------------------------------------"

CMD=("$VENV_PYTHON" "-m" "stock_scanner.alerts.runner"
     "--date" "$DATE"
     ${DRY_RUN:+$DRY_RUN}
     ${SKIP_SCAN:+$SKIP_SCAN}
     "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

_info "Command: ${CMD[*]}"

# Run from repo root
cd "$REPO_DIR"
"${CMD[@]}"
EXIT_CODE=$?

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

_info "--------------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    _info "✅ Pipeline completed successfully in ${ELAPSED}s"
else
    _error "❌ Pipeline failed (exit code $EXIT_CODE) after ${ELAPSED}s"
fi
_info "Log saved: $LOG_FILE"
_info "=================================================="

exit $EXIT_CODE
