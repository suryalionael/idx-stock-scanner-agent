#!/usr/bin/env bash
# =============================================================================
# IDX Scanner — macOS Scheduler Install / Uninstall
# =============================================================================
# Installs (or removes) the launchd job that runs daily alerts automatically.
#
# Usage:
#   ./scripts/install_scheduler.sh install    # register job with launchd
#   ./scripts/install_scheduler.sh uninstall  # remove job
#   ./scripts/install_scheduler.sh status     # check if job is loaded
#   ./scripts/install_scheduler.sh run-now    # trigger immediately (for testing)
#   ./scripts/install_scheduler.sh logs       # tail today's log
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_TEMPLATE="$REPO_DIR/scripts/com.idxscanner.dailyalerts.plist"
PLIST_TARGET="$HOME/Library/LaunchAgents/com.idxscanner.dailyalerts.plist"
LABEL="com.idxscanner.dailyalerts"
DATE="$(date +%Y-%m-%d)"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠️  $*${RESET}"; }
err()  { echo -e "${RED}❌ $*${RESET}" >&2; }
info() { echo "   $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

_check_env_file() {
    local env_file="$REPO_DIR/.env.alerts"
    if [ ! -f "$env_file" ]; then
        warn ".env.alerts not found — creating template at $env_file"
        cat > "$env_file" <<'ENVEOF'
# IDX Scanner — environment variables for daily automation
# Fill in your actual credentials before running.

TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Optional: enables Claude AI summary in daily report
# ANTHROPIC_API_KEY=your_key_here
ENVEOF
        warn "Edit $env_file with your credentials before installing."
        warn "Then re-run: ./scripts/install_scheduler.sh install"
        return 1
    fi

    # Check for placeholder values
    if grep -q "your_bot_token_here\|your_chat_id_here" "$env_file"; then
        err ".env.alerts still has placeholder values. Edit it first."
        return 1
    fi

    ok ".env.alerts found and configured."
    return 0
}

_install() {
    echo ""
    echo "=== IDX Scanner — Install Scheduler ==="
    echo ""

    # 1. Check env file
    _check_env_file || exit 1

    # 2. Create required directories
    mkdir -p "$REPO_DIR/logs/daily"
    mkdir -p "$REPO_DIR/logs"
    ok "Log directories ready."

    # 3. Stamp script executable
    chmod +x "$REPO_DIR/scripts/run_daily_alerts.sh"
    ok "Shell runner is executable."

    # 4. Build plist from template (replace REPO_DIR placeholder)
    sed "s|REPO_DIR|$REPO_DIR|g" "$PLIST_TEMPLATE" > "$PLIST_TARGET"
    ok "Plist installed: $PLIST_TARGET"

    # 5. Validate plist syntax
    if ! plutil -lint "$PLIST_TARGET" > /dev/null 2>&1; then
        err "Plist syntax error. Check $PLIST_TARGET"
        exit 1
    fi
    ok "Plist syntax valid."

    # 6. Unload if already loaded (idempotent)
    if launchctl list | grep -q "$LABEL" 2>/dev/null; then
        launchctl unload "$PLIST_TARGET" 2>/dev/null || true
        info "Previous job unloaded."
    fi

    # 7. Load job
    launchctl load "$PLIST_TARGET"
    ok "Job loaded into launchd."

    echo ""
    echo "=== Installation Complete ==="
    info "Job label   : $LABEL"
    info "Schedule    : Mon–Fri at 06:00 WIB (fallback 07:00)"
    info "Script      : $REPO_DIR/scripts/run_daily_alerts.sh"
    info "Logs        : $REPO_DIR/logs/daily/YYYY-MM-DD.log"
    info "Plist       : $PLIST_TARGET"
    echo ""
    warn "The job will NOT run at load. It will start tomorrow at 08:00 WIB."
    info "To test immediately: ./scripts/install_scheduler.sh run-now"
    echo ""
}

_uninstall() {
    echo ""
    echo "=== IDX Scanner — Uninstall Scheduler ==="

    if [ ! -f "$PLIST_TARGET" ]; then
        warn "Plist not found at $PLIST_TARGET — nothing to uninstall."
        return 0
    fi

    launchctl unload "$PLIST_TARGET" 2>/dev/null || true
    rm -f "$PLIST_TARGET"

    ok "Job unloaded and plist removed."
    info "Logs at $REPO_DIR/logs/ are preserved."
    echo ""
}

_status() {
    echo ""
    echo "=== IDX Scanner — Scheduler Status ==="
    echo ""

    # launchctl list shows: PID, LastExit, Label
    if launchctl list | grep -q "$LABEL" 2>/dev/null; then
        ENTRY=$(launchctl list | grep "$LABEL")
        PID=$(echo "$ENTRY" | awk '{print $1}')
        LAST_EXIT=$(echo "$ENTRY" | awk '{print $2}')

        ok "Job is LOADED."
        info "Label       : $LABEL"
        info "PID now     : ${PID:--} (- means not currently running)"
        info "Last exit   : ${LAST_EXIT:--} (0 = success)"

        if [ -f "$PLIST_TARGET" ]; then
            info "Plist       : $PLIST_TARGET"
        fi
    else
        warn "Job is NOT loaded."
        info "Install with: ./scripts/install_scheduler.sh install"
    fi

    echo ""
    echo "--- Recent log files ---"
    if ls "$REPO_DIR/logs/daily/" 2>/dev/null | tail -5 | while read f; do
        size=$(wc -c < "$REPO_DIR/logs/daily/$f" 2>/dev/null || echo "?")
        echo "   $f  ($size bytes)"
    done; then : ; else
        info "(no log files yet)"
    fi

    echo ""
    if [ -f "/tmp/idx_scanner_daily_${DATE}.lock" ]; then
        warn "Lock file exists: /tmp/idx_scanner_daily_${DATE}.lock"
        warn "Job is currently running (or crashed — remove lock if stale)."
    fi
    echo ""
}

_run_now() {
    echo ""
    echo "=== IDX Scanner — Manual Test Run ==="
    echo ""
    warn "Running pipeline NOW (dry-run by default for safety)."
    warn "To send actual Telegram messages: add --no-dry-run flag."
    echo ""

    DRY_RUN_FLAG="--dry-run"
    if [ "${1:-}" == "--no-dry-run" ]; then
        DRY_RUN_FLAG=""
        ok "Sending real Telegram messages."
    fi

    exec "$REPO_DIR/scripts/run_daily_alerts.sh" $DRY_RUN_FLAG
}

_show_logs() {
    LOG_FILE="$REPO_DIR/logs/daily/$DATE.log"
    if [ -f "$LOG_FILE" ]; then
        echo "=== Tail: $LOG_FILE ==="
        tail -50 "$LOG_FILE"
    else
        warn "No log for today ($DATE)."
        echo "Available logs:"
        ls "$REPO_DIR/logs/daily/" 2>/dev/null | tail -10 | while read f; do
            echo "  $REPO_DIR/logs/daily/$f"
        done
    fi
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "${1:-help}" in
    install)   _install ;;
    uninstall) _uninstall ;;
    status)    _status ;;
    run-now)   _run_now "${2:-}" ;;
    logs)      _show_logs ;;
    *)
        echo "Usage: $0 {install|uninstall|status|run-now|logs}"
        echo ""
        echo "  install    Register the daily alert job with macOS launchd"
        echo "  uninstall  Remove the job from launchd"
        echo "  status     Check if job is loaded and show last exit code"
        echo "  run-now    Run the pipeline immediately (--no-dry-run to send real alerts)"
        echo "  logs       Show today's log (last 50 lines)"
        exit 1
        ;;
esac
