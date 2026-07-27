# ADR — Automated AI Pipeline + Stock Dictionary

**Status:** accepted, implemented.

## Context

The AI Lab subsystem (generation → resolution → reflection → hypothesis generation
+ statistical validation → knowledge base) was built and shipped deliberately
manual-only — see the "Scheduled execution" note under "What's phased for later" in
`docs/AI_LAB_ARCHITECTURE.md` — pending a first reviewed live run. That review has
happened. Separately, the dashboard has accumulated dozens of metrics/scores/terms
across the production scanner, broker analytics, and AI Lab with no centralized,
user-facing explanation.

## Current architecture (before this change)

- `stock_scanner/pipeline/run_daily_scan.py::main()` is the repo's single production
  orchestrator — fetch → features → signal engine → quality filters → ML ranking →
  knowledge-application ranking → explain → publish. Every optional step already
  follows the same try/except/log/continue isolation shape (see its existing
  production performance-tracking step).
- The five AI Lab CLI scripts (`scripts/run_ai_lab.py`, `scripts/resolve_ai_lab.py`,
  `scripts/run_reflection_engine.py`, `scripts/run_hypothesis_engine.py`,
  `scripts/run_knowledge_base_engine.py`) were thin wrappers whose real logic already
  lived in importable `stock_scanner/ai_lab/` and `stock_scanner/db/` modules — no
  business logic was inline, only DB/LLM orchestration boilerplate.
- No metric-glossary infrastructure existed beyond the broker config's narrow
  per-metric `description` block (four entries: FAR, Retail Ratio, RIDR, Smart Money
  Score).

## Problems identified

1. Manual, multi-script execution was required for the AI Lab chain to produce any
   value at all — nothing advanced it automatically.
2. Dashboard copy leaked implementation details ("run `scripts/run_x.py` manually"),
   surfacing internal script paths to end users.
3. Dozens of user-facing metrics across the scanner, broker analytics, and AI Lab had
   no centralized, growable, verifiably-complete, alias-searchable explanation.

## Options considered

### Part 1 — AI automation

| Option | Verdict |
|---|---|
| Subprocess-per-script chain (`subprocess.run(["python", "scripts/run_ai_lab.py"])`, etc.) | Rejected — duplicates isolation handling per call site; spawns child processes for no benefit since the logic was already importable. |
| New GitHub Actions workflow step running each script standalone | Rejected — `data/raw/*.parquet` and `data/ranked/*.csv` are gitignored/ephemeral, only present within the same process/job that produced them; a separate workflow step would need artifact/cache handoff for no reason when the same job already has them in memory. |
| **In-process async orchestrator iterating a name→function stage list (chosen)** | Reuses the orchestrator's existing isolation pattern; no new workflow; a future stage is one adapter function + one appended tuple. A class-based stage hierarchy was also considered and rejected — nothing here needs polymorphism, and a plain list of callables delivers the same one-line extensibility with far less ceremony. |

### Part 3 — Stock Dictionary

| Option | Verdict |
|---|---|
| Hardcode explanations per dashboard page | Rejected — violates single-source-of-truth; the same term explained differently (or not at all) on different tabs. |
| Database-backed dictionary | Rejected — over-engineered for static reference content with no query/write pattern beyond "look up by id or search term." |
| **Directory of versioned YAML configs (entries/aliases/categories) + loader + dedicated tab + automated coverage checker (chosen)** | Mirrors the one existing precedent (the broker config's per-metric `description` block); splitting entries/aliases/categories keeps each concern independently editable; coverage is verified by a script, not asserted by an entry count. |

## Chosen design

### Part 1 — Automated AI Pipeline

```
Daily Scan                  → run_daily_scan.py::main() invoked (scheduled workflow, or manually)
Production Scanner          → existing steps: fetch → features → signal engine → filters → ranking
Publish Production Results  → existing step: save signals/ranked, publish dashboard JSON + OHLC/IHSG bundles
AI Automation Pipeline       → Step 12 (new): stock_scanner.ai_lab.pipeline.run_ai_pipeline()
  ├─ generation        (stock_scanner/ai_lab/generation.py)
  ├─ resolution         (stock_scanner/ai_lab/resolution.py — "Performance Tracker")
  ├─ reflection         (stock_scanner/ai_lab/reflection_runner.py)
  ├─ hypothesis         (stock_scanner/ai_lab/hypothesis_runner.py — covers both
  │                       Hypothesis Generator and Statistical Validation)
  └─ knowledge_base     (stock_scanner/ai_lab/knowledge_runner.py)
Dashboard                    → reads committed data/published/*.json (unchanged read-only contract)
```

- Each script's business logic was extracted into a same-named importable module
  with a `run()` function; the CLI scripts remain thin wrappers, unchanged flags,
  fully independently runnable.
- `stock_scanner/ai_lab/pipeline.py::PIPELINE_STAGES` is a plain
  `list[tuple[str, async function]]` — the orchestrator (`run_ai_pipeline()`) loops
  over it, wraps each call in try/except, and records start/finish/duration. Adding a
  future stage (Calibration Engine, Decision Agent, Model Promotion) is one adapter
  function plus one appended tuple; no orchestrator change.
- Stage status is an enum (`PipelineStageStatus.OK` / `SKIPPED` / `FAILED`, defined in
  `stock_scanner/ai_lab/pipeline_status.py` — a separate module purely to avoid a
  circular import between the stage modules and the orchestrator). Detail lives in a
  `reason` (skip) or `error` (failure) field, never a new status value.
- `data/published/ai_pipeline_status.json` is published after every run — strictly
  execution metadata (`schema_version`, `last_run`, `scan_date`, and per-stage
  `status`/`started_at`/`finished_at`/`duration_ms`/`reason`|`error`). No business
  counts — those stay exclusively in each stage's own published payload
  (`ai_recommendations.json`, `reflection_report.json`, etc.).
- Gated by `ai_lab.enabled` (new top-level key in `scanner_config.yaml`, default
  `true`), following the same top-level-key pattern the existing
  `knowledge_application:` block already uses in that file.
- Wired into `run_daily_scan.py::main()` as a new final step, strictly after the
  existing publish and production-performance-tracking steps — this ordering is what
  makes "AI Lab never influences production scoring/ranking" true by construction,
  not a runtime check: by the time Step 12 runs, every scoring/ranking/filtering/
  publishing step has already read its inputs and wr itten its outputs.

### Part 2 — Dashboard status

- `dashboard/data_loader.py::load_ai_pipeline_status_payload()` is the **only** place
  any dashboard code reads `ai_pipeline_status.json`, following the same remote-then-
  local pattern every sibling `load_*_payload()` already uses.
- The four AI Lab payloads that already carry `generated_at`/`summary` are read
  directly by their views — no client-side `value_counts()`/`max()` computation. The
  one payload missing this (`knowledge_base.json`, the production Learning Agent
  table) gains it at export time, mirroring its sibling's shape.
- Manual-script-execution copy is replaced with status-aware, implementation-free
  text.

### Part 3 — Stock Dictionary

```
stock_scanner/configs/dictionary/
  stock_dictionary.yaml   # {schema_version, entries: [...]}
  aliases.yaml            # {schema_version, aliases: {entry_id: [alias, ...]}}
  categories.yaml         # {schema_version, categories: {category_id: {display_name, description}}}
```

- `categories.yaml`'s keys are the authoritative, enumerable category set — entries'
  `category` values are validated against it at load time, so there's no second,
  hardcoded enum that could drift out of sync.
- Each entry's `references` field is typed (`source_code` / `documentation` /
  `external`), not a flat string list.
- `dashboard/data_loader.py::load_stock_dictionary()` merges all three files, builds
  an alias→entry-id search index, and fails softly (warns, doesn't crash) on a
  missing file, a schema-version mismatch, or a dangling category reference —
  mirroring the existing broker-config loader's posture.
- `scripts/check_dictionary_coverage.py` discovers candidate metrics in priority
  order — dashboard table schemas, Streamlit column-config metadata, published JSON
  schemas, scanner/broker/smart-money config keys (validated even when not currently
  rendered anywhere), dashboard tab titles, and only then AST/string inspection as a
  fallback — and reports discovered/covered/allowlisted/missing counts.
  `tests/test_stock_dictionary_coverage.py` runs the same check as a regression test.

## Trade-offs

- The `ai_lab.enabled` toggle can silence a broken chain silently if nobody checks
  `ai_pipeline_status.json` — mitigated by that status report existing at all (there
  was no equivalent visibility before this change).
- The generation script's LLM client construction has no surrounding try/except,
  unlike its sibling scripts — a deliberate choice preserved during extraction, not
  routed around. Until `NINEROUTER_API_KEY`/`NINEROUTER_MODEL`/`NINEROUTER_BASE_URL`
  are added as scan-workflow secrets, the generation stage reports `"status": "failed"`
  on every scheduled run. The pipeline's per-stage isolation means this doesn't block
  resolution/reflection/hypothesis/knowledge, but it does mean zero new
  recommendations are produced until those secrets exist — an operational
  prerequisite, not a code gap.
- The Dictionary coverage checker's fallback tier (AST/string inspection) can both
  miss dynamically-built labels and over-flag incidental UI chrome; a small, reviewed
  allowlist keeps it practical rather than chasing every literal string match.
- Splitting Dictionary config into three files adds a small cross-file consistency
  requirement (an entry's `id` must exist consistently across `stock_dictionary.yaml`
  and any `aliases.yaml`/`categories.yaml` reference to it) — mitigated by loader-time
  validation warnings.

## Future extension points

- New pipeline stages: one adapter function + one `PIPELINE_STAGES` entry.
- Per-stage tuning knobs (`top_n`, `model_keys`, statistical thresholds) currently
  stay as CLI-flag-style defaults on each `run()`; they could move into
  `scanner_config.yaml`'s `ai_lab:` key later without changing the orchestrator.
- `schema_version` on both `ai_pipeline_status.json` and the Dictionary configs gives
  a documented seam for future shape changes without breaking existing readers.
- The deferred ⓘ column-header tooltip integration (`st.column_config.Column(...,
  help=...)`) would reuse the same alias-aware Dictionary index built for search.
