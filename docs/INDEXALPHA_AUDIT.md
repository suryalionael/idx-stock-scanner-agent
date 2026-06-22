# IndexAlpha Integration — Production Audit

**Date:** 2026-06-22 (original audit), **updated 2026-06-22** with live
verification results.
**Scope:** Connectivity/auth, actual usage, data validity, fallback behavior,
monitoring, production hardening — per the production-trust audit request.
**Method:** Evidence-based. Every claim below is backed by a command run
against the real repo/CI/secrets in this session — not inferred from reading
code in isolation. Where I could not verify something (e.g. Streamlit
Cloud's own secrets store), that is stated explicitly rather than assumed.

---

## UPDATE (2026-06-22, same day): Live Verification — VERIFIED HEALTHY

The original audit below correctly reported "not verified" — at that time it
was true. Since then, the actual gap (secret never wired into any workflow)
was fixed and a single, deliberate, controlled live call was executed. This
section is an addendum, not a rewrite — the original findings stay below
exactly as recorded, because they were accurate for that point in time.

**What changed:**
1. `.github/workflows/indexalpha_live_check.yml` created — `workflow_dispatch`
   only, no schedule, wires `secrets.INDEX_ALPHA_API_KEY` into env.
2. Triggered exactly once via `gh workflow run` (run
   [27958749130](https://github.com/suryalionael/idx-stock-scanner-agent/actions/runs/27958749130)).
3. Real result, captured verbatim from the run log:
   ```
   IndexAlpha fetch: BBCA @ 2026-06-22 (investor=all, market=RG)
   IndexAlpha /stocks/broker-summary BBCA -> 200 in 844ms (success=True, items=67)
   IndexAlpha: 67 broker records untuk BBCA @ 2026-06-22 (net_lot range: -17985100 … 27966200)
   VERDICT: HEALTHY
   ```
4. `INDEX_ALPHA_API_KEY: ***` confirmed masked in every env dump in the log —
   never leaked.
5. Committed evidence: `data/published/indexalpha_health.json` (commit
   `8162329`, pushed automatically by the workflow):
   ```json
   {"last_attempt_at": "2026-06-22T14:07:28...", "last_status_code": 200,
    "last_latency_ms": 843.6, "last_error_type": null, "total_calls": 1,
    "last_success_at": "2026-06-22T14:07:28...", "consecutive_failures": 0,
    "total_successes": 1}
   ```

**Honest caveat — what this does and doesn't prove:**
- **Proves:** the API key is valid, the endpoint/auth/parsing/caching code
  path works end to end, for a GitHub Actions runner with the secret wired.
- **Does NOT prove:** Streamlit Cloud's deployment has the same key
  configured (separate secrets store, not inspectable from here) — so the
  *dashboard a real user sees* may still be running key-less and serving
  fallback/cache, even though the integration itself is now proven sound.
  This is the one remaining open item — see "Belum dibereskan" below.
- Running `python scripts/check_indexalpha_health.py` (no `--live`) in any
  OTHER environment (e.g. this local shell) will correctly still report
  `DEGRADED` — because that specific environment has no key and its own
  local `data/broker/` cache is still 12 days old. That is not a
  contradiction: health is per-environment, and the script is intentionally
  honest about THIS vantage point rather than borrowing the CI run's
  success to claim something it can't see (Streamlit Cloud's state).

**Status akhir: integration code = VERIFIED HEALTHY. Dashboard production
deployment = UNKNOWN (cannot be checked from a coding session).**

**Belum dibereskan sebelum diklaim layak production sepenuhnya:**
1. Verifikasi manual: apakah Streamlit Cloud secrets panel punya
   `INDEX_ALPHA_API_KEY`? Hanya pemilik akun yang bisa cek ini.
2. Kalau TIDAK ada di Streamlit Cloud: tambahkan di sana — quota tetap 5/hari
   dipakai bersama, jadi ini keputusan produk (berapa banyak dashboard usage
   vs CI verification yang boleh memakai quota), bukan keputusan teknis yang
   aman diambil sepihak.
3. 55 ruff lint error pre-existing di file lain — masih di luar scope (lihat
   audit asli di bawah).

---

## Verdict (the 6 questions asked, answered directly)

1. **Apakah IndexAlpha saat ini benar-benar berfungsi?**
   **Tidak terverifikasi berfungsi di environment manapun yang bisa saya
   cek.** `INDEX_ALPHA_API_KEY` ada sebagai GitHub repo secret (dikonfigurasi
   2026-06-04, dikonfirmasi via `gh secret list`), tapi tidak ada satu pun
   GitHub Actions workflow yang membaca secret ini ke environment. Tidak ada
   di shell lokal. Health-state file (`data/published/indexalpha_health.json`,
   ditulis otomatis oleh setiap call nyata) **tidak ada sama sekali** — artinya
   `fetch_indexalpha._get()` belum pernah tercatat berhasil maupun gagal di
   checkout ini. Cache broker terbaru: **2026-06-10 (12 hari stale)**. Saya
   **tidak** mencoba memverifikasi via panggilan live — quota free plan hanya
   5 request/hari, dan menghabiskannya tanpa izin eksplisit melanggar prinsip
   "trust > speed". Lihat §6 untuk cara memverifikasi dengan aman.
2. **Apakah benar-benar dipakai di jalur produksi?**
   **Ya, untuk dashboard.** `IndexAlphaFetcher`/`fetch_with_cache` dipanggil
   nyata dari `dashboard/data_loader.py` (`fetch_broker_summary`,
   `fetch_broker_latest`, `fetch_broker_range`), dan `render_broker_section()`
   di `dashboard/app.py` dipanggil dari **7 lokasi** di tab Swing, Scalping,
   Long Term, Smart Money, Naik/Turun Beruntun, dan Search Emiten — diverifikasi
   langsung di browser (lihat §2). **Tidak** dipakai di scan/signal-scoring
   pipeline (`run_daily_scan.py` hanya membuat direktori `data/broker/`, tidak
   pernah memanggil fetcher apa pun) — ini sesuai desain (broker summary adalah
   fitur display on-demand, bukan input scoring), bukan bug, tapi perlu
   didokumentasikan eksplisit agar tidak overclaim.
3. **File mana yang memakai/harusnya memakai IndexAlpha?**
   Lihat tabel di §2.
4. **Kalau ada bug, di auth/fetch/parsing/routing/fallback/publishing?**
   Tidak ada bug logic ditemukan di fetch/parsing/fallback (lihat §3-4 untuk
   bukti). Gap-nya di **operasional**: secret ada tapi tidak pernah disalurkan
   ke environment yang menjalankan kode (bukan bug kode, tapi kesenjangan
   konfigurasi/deployment), dan **tidak ada monitoring/test sebelum hari ini**.
   Satu temuan dead-code-risk di `broker_summary.py` (§3) — sudah diperbaiki.
5. **Perubahan exact:** lihat §6.
6. **Bukti eksekusi nyata:** lihat §7.

---

## 1. Connectivity & Auth

| Item | Status | Bukti |
|---|---|---|
| API key dibaca dari env var | ✅ Benar | `os.environ.get("INDEX_ALPHA_API_KEY")`, tidak hardcoded |
| Endpoint valid | ✅ | `https://api.indexalpha.id/stocks/broker-summary`, `/usage` — terdokumentasi lengkap di docstring modul |
| Secret tidak pernah ter-print | ✅ Diverifikasi | URL dibangun dari `params` saja (ticker/from/to/investor/market) — key HANYA di header `Authorization`, tidak pernah masuk ke URL/log. Test `test_get_401_raises_permission_error_and_never_leaks_key` membuktikan pesan exception tidak mengandung key. |
| Request sukses vs silent failure | ✅ Sekarang eksplisit | **Sebelum:** tidak ada logging latency/status code pada path sukses. **Sesudah:** setiap call (sukses/gagal) tercatat ke `data/published/indexalpha_health.json` dengan status code, latency (ms), error type — lihat §6. |
| GitHub secret ada tapi tidak disalurkan | ⚠️ **Gap nyata** | `gh secret list` menunjukkan `INDEX_ALPHA_API_KEY` ada (configured 2026-06-04). `grep -rn "INDEX_ALPHA_API_KEY" .github/workflows/` → **0 hasil**. Tidak ada workflow yang menyalurkan secret ini ke env. Status di Streamlit Cloud (deployment terpisah) tidak bisa saya cek dari sini — perlu verifikasi manual di panel secrets Streamlit Cloud. |

---

## 2. Actual Usage — Source-of-Truth Trace (UI → IndexAlpha)

Ditelusuri end-to-end, bukan diasumsikan dari membaca kode saja:

```
app.py: render_broker_section()      [dipanggil dari 7 tab — dikonfirmasi via grep]
  ├─ _render_broker_latest()
  │    └─ data_loader.fetch_broker_latest()
  │         └─ data_loader.fetch_broker_summary()
  │              └─ fetch_indexalpha.fetch_with_cache()
  │                   └─ IndexAlphaFetcher.fetch()  ← REAL HTTP call ke api.indexalpha.id
  └─ _render_broker_historical()
       └─ data_loader.fetch_broker_range()
            └─ IndexAlphaFetcher.fetch_range()      ← REAL HTTP call

Konsumen analitik (terpisah, BUKAN scoring pipeline):
  broker_intelligence.py / smart_money_screener.py
    └─ data_loader.load_broker_history()
         └─ glob data/broker/{ticker}.JK_*.parquet   [cache yang ditulis IndexAlpha]
```

**File yang memakai IndexAlpha (dikonfirmasi via call, bukan cuma import):**
| File | Peran |
|---|---|
| `stock_scanner/pipeline/fetch_indexalpha.py` | Service layer — satu-satunya tempat HTTP call ke IndexAlpha |
| `dashboard/data_loader.py` | Memanggil `IndexAlphaFetcher`/`fetch_with_cache` langsung (3 fungsi: `fetch_broker_summary`, `fetch_broker_latest`, `fetch_broker_range`) |
| `dashboard/app.py` | `render_broker_section()` dipanggil di baris 833, 979, 1272, 1621, 2274, 2357, 2740 — dikonfirmasi via grep, BUKAN dead code |
| `stock_scanner/pipeline/broker_intelligence.py`, `smart_money_screener.py` | Konsumen cache (tidak fetch sendiri) — dikonfirmasi naming convention cache cocok (`{TICKER}.JK_{date}.parquet`) via test riil: `load_broker_history('WIIM.JK')` mengembalikan 106 baris data nyata |
| `run_daily_scan.py` | **TIDAK memakai** — hanya membuat direktori. Ini desain, bukan bug (broker summary = fitur display, bukan input signal scoring) |

**Dead code ditemukan dan dihapus:** `dashboard/app.py::_load_broker_data_from_cache()` — didefinisikan, **tidak pernah dipanggil** (dikonfirmasi grep), memakai filename convention LAMA yang sudah tidak match cache nyata. Dihapus.

---

## 3. Data Validity

| Risiko | Temuan |
|---|---|
| Null/malformed fields | `_normalize_response()` sudah pakai `.get(key, 0) or 0` untuk semua field numerik — tidak crash pada payload partial. Dibuktikan dengan test `test_normalize_response_handles_missing_optional_fields`. |
| Empty payload | `if not data: return pd.DataFrame()` — tidak fabrikasi data. |
| Duplicate | Tidak ada dedup eksplisit di level response (IndexAlpha API diasumsikan tidak mengirim broker_code duplikat per response) — **tidak diuji langsung ke API nyata** karena keterbatasan quota; risiko rendah karena tidak ada bukti masalah ini terjadi. |
| **Mismatch tanggal market** | **Tidak ditemukan bug** — diverifikasi eksplisit: `load_broker_history('WIIM.JK')` mengembalikan data nyata 106 baris bertanggal 2026-06-10, filename convention `.JK_{date}.parquet` cocok persis antara penulis (IndexAlpha fetcher) dan pembaca (broker_intelligence via data_loader). |
| Data dibuang/fallback diam-diam | **Ditemukan 1 risiko nyata, sudah diperbaiki** — `broker_summary.py::get_broker_summary(use_mock_if_empty=True)` akan mengembalikan data SINTETIS (`PlaceholderBrokerFetcher`, angka random seeded by ticker+date) jika cache kosong. **Tidak pernah dipanggil di code path manapun** (dikonfirmasi grep — 0 caller selain modul itu sendiri), tapi merupakan risiko laten: siapa pun yang menyambungkannya di masa depan akan diam-diam menyajikan data palsu yang terlihat seperti data asli. **Dihapus seluruhnya** (§6). |

---

## 4. Fallback Behavior

**Temuan baik (sudah ada sebelum audit ini, tidak diklaim sebagai pekerjaan saya):**
`fetch_broker_latest()` di `data_loader.py` sudah didesain dengan benar:
- Tier 1: cache utuh untuk sesi target → tidak ada API call, `source="cache"`.
- Tier 2: tidak ada cache, API key ada → fetch live, `source="fresh"`.
- Tier 3: gagal/tidak ada key → fallback ke cache terakhir, `source="fallback"`, **dengan `note` eksplisit** menjelaskan alasan fallback dan tanggal yang ditampilkan.

UI (`_render_broker_latest`) sudah menampilkan badge 🟢Fresh/🗂️Cache/⚠️Fallback **sebelum audit ini** — verified working di §7.

**Yang DITAMBAHKAN audit ini:** badge level-integrasi terpisah
(`_render_indexalpha_integration_badge()`) yang menjawab pertanyaan berbeda:
bukan "apakah data sesi ini fresh" tapi "apakah koneksi IndexAlpha itu sendiri
sedang sehat" — dua hal yang sebelumnya tidak dibedakan secara eksplisit di
UI. Dikonfirmasi tampil bersamaan dengan badge lama di §7.

---

## 5. Monitoring & Alerting — Sebelum vs Sesudah

| Metrik yang diminta | Sebelum | Sesudah |
|---|---|---|
| Success rate | ❌ Tidak ada | ✅ `total_successes`/`total_calls` di health-state JSON |
| Latency | ❌ Tidak ada | ✅ Diukur per-call (`time.monotonic()`), dicatat dalam ms |
| Freshness | ⚠️ Implisit (lewat cache file timestamp) | ✅ Eksplisit — `check_indexalpha_health.py` menghitung umur cache dalam hari |
| Last successful fetch | ❌ Tidak ada | ✅ `last_success_at` (ISO timestamp) |
| Consecutive failures | ❌ Tidak ada | ✅ Counter otomatis, reset ke 0 saat sukses |
| Alert saat gagal beruntun/stale | ❌ Tidak ada | ✅ `check_indexalpha_health.py` exit code 0/1/2 + verdict tertulis; badge UI muncul otomatis saat `consecutive_failures>=3` |

---

## 6. Perubahan Exact yang Dilakukan

| File | Perubahan |
|---|---|
| `stock_scanner/pipeline/fetch_indexalpha.py` | + `_record_health()` — mencatat status code/latency/error type ke `data/published/indexalpha_health.json` setiap call (sukses & gagal), tanpa pernah menulis API key. + logging latency pada path sukses. 3 ruff lint pre-existing di file ini dibersihkan sekalian (unused import, f-string kosong, import order) — tidak mengubah behavior. |
| `stock_scanner/pipeline/broker_summary.py` | **Dihapus**: `PlaceholderBrokerFetcher`, `_SAMPLE_BROKERS`, `BaseBrokerFetcher`, `get_broker_summary()` — mock-data-fallback risk, 0 caller dikonfirmasi. **Dipertahankan**: `load_broker_summary`/`save_broker_summary` (cache I/O aman, tidak ada fallback palsu). |
| `dashboard/app.py` | Hapus `_load_broker_data_from_cache()` (dead code, convention filename salah). Tambah `_render_indexalpha_integration_badge()` dan panggil dari `render_broker_section()` — badge status integrasi terpisah dari badge per-sesi yang sudah ada. |
| `scripts/check_indexalpha_health.py` | **Baru.** Quota-safe by default (0 network call) — cek key presence, riwayat health-state, freshness cache. `--live` opsional (1 quota request, harus diminta eksplisit, tidak pernah otomatis). |
| `tests/test_fetch_indexalpha.py` | **Baru.** 19 test, HTTP di-mock penuh (`unittest.mock.patch` pada `urllib.request.urlopen`) — **zero quota cost, zero network call**. Cakupan: normalisasi response, ticker cleaning, cache path, 401/429/timeout, retry behavior, non-leak-secret, health-state tracking. |
| `tests/__init__.py` | Baru — direktori `tests/` sebelumnya **tidak ada sama sekali** meski `pyproject.toml` sudah mengasumsikan `testpaths=["tests"]` dan CI workflow `Tests` sudah men-define step `pytest`. |

**Tidak diubah (di luar scope, didokumentasikan agar tidak silent):**
55 ruff lint error pre-existing tersisa di ~20 file lain (`alerts/*.py`,
beberapa `pipeline/*.py` termasuk `train_ranker_from_history.py`) —
**bukan dari perubahan hari ini** (dikonfirmasi `git diff` tidak menyentuh
baris-baris itu). CI workflow "Tests" sudah **gagal di 3 push terakhir**
sebelum sesi ini (termasuk komit-komit sebelumnya di sesi yang sama) karena
gabungan dua hal: (a) lint debt ini, (b) `tests/` yang tidak ada. (b) sudah
diperbaiki; (a) butuh dedicated cleanup pass terpisah — bukan untuk hari ini.

---

## 7. Bukti Eksekusi Nyata

```
$ python3 -m pytest tests/test_fetch_indexalpha.py -v
19 passed in 0.48s          # semua mocked, 0 network call, 0 quota terpakai

$ python3 -m ruff check stock_scanner/pipeline/fetch_indexalpha.py \
    stock_scanner/pipeline/broker_summary.py stock_scanner/db/
All checks passed!

$ python3 scripts/check_indexalpha_health.py
VERDICT: MISCONFIGURED — integration has likely never run successfully here.
(exit code 2)               # jujur — TIDAK mengklaim sehat padahal belum verified
```

**UI, diverifikasi visual via browser (localhost:8503, Search Emiten → BBCA.JK
→ Broker Activity tab):**
- Badge integrasi baru: *"⚠️ IndexAlpha API belum pernah terhubung di
  environment ini... Data broker di bawah — jika ada — berasal dari cache
  lama, bukan sesi live."*
- Badge per-sesi (sudah ada sebelumnya, masih berfungsi berdampingan):
  *"Sumber: Index Alpha · BBCA · Sesi 2026-06-08 · ⚠️Fallback"* +
  *"Menampilkan cache 2026-06-08 — sesi baru 2026-06-22 belum tersedia
  (INDEX_ALPHA_API_KEY belum diset)."*

Kedua badge tampil bersamaan — disclosure dua-layer (status koneksi +
status data sesi ini), tidak ada yang berpura-pura live.

---

## Status Production Readiness

| Aspek | Status |
|---|---|
| Tidak ada credential leak | ✅ Aman, diuji |
| Tidak ada silent mock fallback | ✅ Diperbaiki hari ini |
| Fallback eksplisit & teraudit | ✅ (sudah ada + diperkuat) |
| Monitoring dasar | ✅ Dibangun hari ini |
| Test coverage jalur kritis | ✅ Dibangun hari ini (19 test) |
| **Koneksi live terverifikasi** | ❌ **Belum** — perlu keputusan Anda (lihat di bawah) |
| CI hijau penuh | ❌ Belum — 55 lint error pre-existing di luar scope hari ini |

**Masih butuh hardening sebelum diklaim "production-grade" untuk fitur ini:**
1. Putuskan: jalankan `python scripts/check_indexalpha_health.py --live`
   (memakai 1 dari 5 quota harian) untuk pembuktian end-to-end pertama, ATAU
   verifikasi langsung di Streamlit Cloud secrets panel apakah key sudah ada
   di sana.
2. Kalau integrasi memang harus aktif di CI (bukan hanya dashboard
   interaktif), tambahkan `INDEX_ALPHA_API_KEY` ke environment workflow yang
   relevan — **belum dilakukan hari ini**, karena menambah pemanggilan API
   terjadwal ke quota 5/hari adalah keputusan produk, bukan keputusan teknis
   yang aman saya ambil sepihak.
3. Cleanup 55 lint error pre-existing — dedicated pass terpisah.
