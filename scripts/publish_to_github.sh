#!/usr/bin/env bash
# publish_to_github.sh — Commit dan push data/published/latest_scan.json ke GitHub.
#
# Dipanggil oleh pipeline setelah scan harian berhasil.
# Aman: jika tidak ada perubahan, skip commit (tidak buat empty commit).
# Jika push gagal, log error tapi keluar dengan kode 0 agar scan tidak dianggap gagal.
#
# Usage:
#   bash scripts/publish_to_github.sh
#   bash scripts/publish_to_github.sh --dry-run       # print saja, tidak commit/push
#
# Environment:
#   GIT_AUTHOR_NAME   (opsional, default: "IDX Scanner Bot")
#   GIT_AUTHOR_EMAIL  (opsional, default: "scanner@idx-local")

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLISH_FILE="data/published/latest_scan.json"
DRY_RUN=false

# Parse args
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
  esac
done

echo "[publish] Root: $REPO_ROOT"
echo "[publish] Target: $PUBLISH_FILE"
echo "[publish] Dry-run: $DRY_RUN"

cd "$REPO_ROOT"

# ── Cek file ada ─────────────────────────────────────────────────────────────
if [[ ! -f "$PUBLISH_FILE" ]]; then
  echo "[publish] ERROR: $PUBLISH_FILE tidak ditemukan. Scan mungkin belum selesai." >&2
  exit 1
fi

# ── Git status — cek apakah ada perubahan ────────────────────────────────────
# git diff --quiet returns 0 if no changes, 1 if changes
if git diff --quiet HEAD -- "$PUBLISH_FILE" 2>/dev/null && \
   ! git ls-files --others --exclude-standard -- "$PUBLISH_FILE" | grep -q .; then
  echo "[publish] Tidak ada perubahan pada $PUBLISH_FILE — skip commit."
  exit 0
fi

echo "[publish] Perubahan terdeteksi, menyiapkan commit..."

# ── Set git identity jika belum ada ──────────────────────────────────────────
GIT_NAME="${GIT_AUTHOR_NAME:-IDX Scanner Bot}"
GIT_EMAIL="${GIT_AUTHOR_EMAIL:-scanner@idx-local}"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[publish] DRY-RUN — perintah yang akan dijalankan:"
  echo "  git add $PUBLISH_FILE"
  echo "  git commit -m 'data: update latest_scan.json ($(date +%Y-%m-%d))' --author='$GIT_NAME <$GIT_EMAIL>'"
  echo "  git push origin HEAD"
  exit 0
fi

# ── Stage file ────────────────────────────────────────────────────────────────
git add "$PUBLISH_FILE"

# ── Commit ───────────────────────────────────────────────────────────────────
COMMIT_DATE="$(date +%Y-%m-%d)"
git -c "user.name=$GIT_NAME" -c "user.email=$GIT_EMAIL" \
  commit -m "data: update latest_scan.json ($COMMIT_DATE)" \
  --no-gpg-sign || {
    echo "[publish] ERROR: git commit gagal." >&2
    exit 1
}
echo "[publish] Commit berhasil."

# ── Push ─────────────────────────────────────────────────────────────────────
# Push dengan timeout implisit — jika gagal, log tapi keluar 0 agar pipeline tetap OK
if git push origin HEAD 2>&1; then
  echo "[publish] Push berhasil ke GitHub."
else
  echo "[publish] WARNING: git push gagal. Dashboard akan tetap menampilkan data lama." >&2
  # Keluar dengan 0 — push failure tidak boleh blokir pipeline
  exit 0
fi

echo "[publish] Selesai."
exit 0
