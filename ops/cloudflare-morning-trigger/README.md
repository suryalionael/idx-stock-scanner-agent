# Morning trigger — external precise scheduler

Fires at **exactly 06:00 WIB** and triggers `send-alert.yml` on GitHub via a
`repository_dispatch` event. This replaces GitHub's best-effort cron for the
time-critical send. The heavy scan stays on GitHub (`scan.yml`, 05:00 WIB).

The repo stays **public**; the only credential off GitHub is a tightly scoped
PAT held by the scheduler. All app secrets (Telegram, Index Alpha, universe)
stay in GitHub Secrets.

---

## 1. Mint the PAT (minimal scope)

GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate new:

- **Resource owner:** your account
- **Repository access:** *Only select repositories* → `idx-stock-scanner-agent`
- **Repository permissions → Contents: Read and write**  ← the documented minimum
  for `repository_dispatch`. Leave everything else **No access**.
- **Expiration:** 90 days (rotate).

> `POST /repos/{owner}/{repo}/dispatches` requires `Contents: write` on a
> fine-grained PAT (or `repo`/`public_repo` on a classic PAT). There is no
> narrower permission for dispatch; single-repo scope limits the blast radius.

## 2. Test the dispatch with curl (no scheduler yet)

```bash
curl -L -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GH_PAT" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/suryalionael/idx-stock-scanner-agent/dispatches \
  -d '{"event_type":"morning-alert"}'
# Expect: HTTP 204 No Content → check the Actions tab: "Morning Alert (send only)" runs.
```

---

## Option 1 — cron-job.org (simplest, no code, free)

1. Sign up at https://cron-job.org → **Create cronjob**.
2. **URL:** `https://api.github.com/repos/suryalionael/idx-stock-scanner-agent/dispatches`
3. **Method:** `POST`
4. **Schedule:** Time `06:00`, **Timezone `Asia/Jakarta`**, days **Mon–Fri**.
5. **Headers:**
   - `Accept: application/vnd.github+json`
   - `Authorization: Bearer <YOUR_PAT>`
   - `X-GitHub-Api-Version: 2022-11-28`
   - `Content-Type: application/json`
6. **Request body:** `{"event_type":"morning-alert"}`
7. Save. cron-job.org treats HTTP 204 as success; enable failure notifications.

## Option 2 — Cloudflare Worker (free, code, this folder)

```bash
npm i -g wrangler
cd ops/cloudflare-morning-trigger
wrangler login
wrangler secret put GH_PAT      # paste the fine-grained PAT
wrangler deploy                 # cron trigger "0 23 * * 0-4" = 06:00 WIB
wrangler tail                   # watch logs; visit the worker URL to test-fire
```

`wrangler.toml` holds the non-secret owner/repo/event; `GH_PAT` is a Worker
secret (never committed).

---

## How timing & dedup work

- **05:00 WIB** `scan.yml` refreshes data and uploads the `scan-latest` artifact
  (60-min buffer absorbs cold-cache scans and any cron delay).
- **06:00 WIB** this trigger → `send-alert.yml` downloads the latest scan
  artifact, runs `telegram_alert --require-fresh`, and sends. Tiny job → arrives
  ~06:00–06:01.
- **Stale data:** the sender posts the "Data Belum Update" notice (never stale
  signals) and does **not** write the sent-marker.
- **Dedup:** a per-day `sent-<date>` marker artifact means the **06:12 WIB backup
  schedule** only sends if the 06:00 dispatch didn't already deliver a real alert.
