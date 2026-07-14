# Learning Agent — Architecture

**Status:** Phase 1 implemented (statistics-only). Phase 2 (LLM hypothesis articulation)
not yet built — gated on Phase 1's own verdict about data sufficiency. See
`docs/SELF_IMPROVING_ARCHITECTURE.md` for the separate, already-closed champion/challenger
promotion loop this system explicitly does not duplicate or bypass.

---

## Why this exists, and what it is not

The Learning Agent is a **research tool, not a trading model**. It cannot change
production behavior, cannot modify `scanner_config.yaml` or `signal_engine.py`, cannot
write to `model_registry`, and cannot promote anything. Its only job is to discover
recurring statistical patterns in historical signals/outcomes and, eventually (Phase 2),
articulate them as human-readable hypotheses. Any hypothesis that looks worth testing
still has to go through the **existing, unmodified** `scripts/train_challenger.py` +
`scripts/promote_challenger.py` pipeline — this system does not get its own promotion
path.

```
Historical Data → Pattern Miner (Phase 1, statistics only) → Report
                                                              ↓ (human review)
                          Hypothesis Agent (Phase 2, LLM via 9router) → Knowledge Base
                                                              ↓ (human translates a
                                                                 hypothesis into a
                                                                 train_challenger.py
                                                                 feature/threshold change)
                          scripts/train_challenger.py (existing, unmodified)
                                                              ↓
                          scripts/promote_challenger.py (existing, unmodified)
```

## Phase 1 — statistics-only Pattern Miner

**Question it answers:** does the historical database contain enough statistical signal
to justify building an LLM hypothesis stage at all?

**Package:** `stock_scanner/learning/` — deliberately outside `stock_scanner/pipeline/`.
Nothing the morning scan imports can reach this code; the package boundary makes "cannot
affect production" a structural property, not a convention to remember.

**Data source:** `scripts/train_challenger.py::load_training_examples()`, reused
unchanged (three other scripts already depend on its exact shape — this module adds its
own `sector_reference` join on top rather than modifying a shared function).

**Method:** for each candidate slice (single feature, pairwise, or triple feature-
dimension combination), compute win rate, a Wilson 95% confidence interval, a
Beta-Binomial shrinkage estimate (weak prior toward the global baseline, so tiny slices
don't display misleadingly extreme rates), and a Fisher's exact test p-value against the
rest of the population. Benjamini-Hochberg FDR correction is applied **separately within
each interaction-order tier** — single, pairwise, and triple never share one correction
pool, so the triple tier's much larger test count (thousands of combinations) can't drown
out real single-feature signal.

**Confidence gate — a pattern is only reported as a primary finding if all of:**

| Check | Rule | Why |
|---|---|---|
| Sample floor | `n_success >= 8` | Independent of p-value — a handful of positive examples isn't operationally trustworthy no matter how extreme the p-value looks |
| Significance | Fisher's exact test, FDR-adjusted p < 0.05 | Exact test, appropriate for the small counts this database actually has |
| Confidence interval | Wilson CI lower bound > baseline win rate | The conservative bound, not the point estimate — same discipline `promote_challenger.py` already applies to challenger-vs-champion comparisons |

Triple-feature (order-3) results are always computed but reported **separately, as
exploratory** — at current data volume a 5×5×5 quintile cross averages only a handful of
positives per cell, so a technically-passing result there is much more likely to be noise
than signal. Two report-only cross-checks (never a hard reject) accompany every gated
finding: ticker-concentration (flagged if one ticker contributes >40% of a slice's
positives) and time-split direction stability, reusing the pattern already proven in
`train_challenger.py::sensitivity_battery()`.

**Dimensions searched:** derived/normalized technical features (`rsi14`,
`vol_ratio_20d`, `atr_pct`, `pct_from_52w_high`, ... — see `NUMERIC_FEATURES` /
`BOOLEAN_FEATURES` in `pattern_miner.py`), plus `sector`, `regime` (derived from
`market_context.ihsg_pct_change_eval`), `strategy`, and `signal_label`. Redundant raw
inputs already summarized by a derived column (e.g. `ma20`/`ma50`/`ma200` behind
`price_vs_ma200`) are deliberately excluded — including both would inflate the
multiple-testing burden without adding independent information.

**Entrypoint:** `python scripts/run_learning_agent.py` — read-only, no schedule yet
(manual only). Writes `data/reports/pattern_miner_{date}.json` (machine-readable, full
detail) and `data/reports/pattern_miner_{date}.md` (human-readable, with an explicit
"is there enough signal yet" verdict at the top) under the existing gitignored
`data/reports/` convention. No database writes.

**Known data-volume reality (checked 2026-07-13):** `load_training_examples()` currently
returns ~1,558 evaluated rows; the broader `outcomes` table has 74 positive labels out of
1,940 evaluated (base rate ≈ 3.8%); `sector_reference` covers 87 of ~950 universe tickers.
Single- and pairwise-feature search may find a few gated patterns; three-way interactions
are very likely infeasible at this volume. The report's top-line verdict states this
explicitly rather than implying more statistical depth than the data currently supports.

## Phase 2 — not yet built

Status: **complete, verdict positive** (single-feature tier rediscovered the validated
SAFE set; report at `data/reports/pattern_miner_2026-07-13.md`). Phase 2 is cleared to
proceed. Provider: **9router** (product decision — not yet integrated anywhere in this
codebase; both existing unused stub integrations, `pipeline/explain_agent.py` and
`dashboard/explain.py`, assume direct Anthropic and are unaffected by this new path).

### Scope of `hypothesis_agent.py`

One job: turn Phase 1's already-gated, already-de-duplicated pattern clusters into
structured hypothesis text via an LLM call. It does **not** compute statistics, decide
significance, query the database directly, or call anything in
`stock_scanner/pipeline/`. It is read-only with respect to everything except its own
`knowledge_base` rows. Runs manually only (no cron) until several batches have been
hand-reviewed.

### Minimum files/tables

| File | Purpose |
|---|---|
| `stock_scanner/learning/pattern_dedup.py` | De-duplication stage — Phase 1 output in, clustered patterns out. Build and verify this **before** touching the LLM (see "smallest safe first step" below). |
| `stock_scanner/learning/hypothesis_agent.py` | The LLM call + strict response validation. |
| `stock_scanner/db/knowledge_base.py` | Schema bootstrap + `write_hypotheses()` / `load_knowledge_base()` / `update_status()`. |
| `scripts/run_hypothesis_agent.py` | CLI entrypoint, manual only, mirrors `scripts/run_learning_agent.py`'s shape. |
| `knowledge_base` table | Added to `stock_scanner/db/schema.sql` (see §2 of `SELF_IMPROVING_ARCHITECTURE.md` for the sibling `model_registry` table this mirrors in spirit). |
| `data/published/knowledge_base.json` | Committed JSON mirror, same durability pattern as `registry_io.py` (survives ephemeral CI runners; not needed until a workflow exists). |

No workflow/cron file yet — see smallest-first-step.

### Input/output contract

**Input** — a list of `ClusteredPattern` objects (post-dedup), never raw DataFrame rows,
never a DB connection, never anything the LLM could mistake for ticker-level detail:

```
{dimensions, slice_definition, n, n_success, win_rate, win_rate_shrunk,
 ci_lower, p_value_adjusted, ticker_concentration_flag, time_split_stable,
 member_count}   # member_count = how many raw pairwise rows this cluster represents
```

**Output** — strict JSON per hypothesis, schema-validated before being trusted; a
malformed response is logged and skipped, never retried into a looser shape:

```json
{
  "hypothesis": "string",
  "confidence": 0.0,
  "supporting_trades": 0,
  "affected_sector": null,
  "expected_effect": "string",
  "status": "candidate",
  "source_cluster_id": "string"
}
```

`confidence` is the LLM's qualitative framing — explicitly labeled as such, never
conflated with the p-value Phase 1 already computed. `supporting_trades` is copied from
the cluster's `n_success`, never invented by the model. `status` is always `"candidate"`
at creation. `source_cluster_id` traces every hypothesis back to the exact Phase 1
pattern(s) that produced it.

### Guardrails (non-negotiable)

- **No production changes** — `hypothesis_agent.py` and everything it imports stays
  inside `stock_scanner/learning/`; nothing in `stock_scanner/pipeline/` can reach it.
- **No DB writes to live systems** — the only table this stage ever writes is
  `knowledge_base`. Never `signals`, `feature_snapshots`, `outcomes`, `model_registry`,
  or `promotion_decisions`.
- **No raw DB rows to the LLM** — input is exclusively the aggregated, already-gated
  statistical summaries described above. No ticker, no signal_id, no trade-level row ever
  enters a prompt.
- **No auto-promotion** — `knowledge_base.status` starts and stays `"candidate"` until a
  human manually translates it into a `scripts/train_challenger.py` run.
  `hypothesis_agent.py` never calls `train_challenger.py` or `promote_challenger.py`.
- **No schedule by default** — manual invocation only until multiple runs have been
  hand-reviewed end to end.

### De-duplication rule (pairwise patterns → clusters)

Phase 1's own run showed why this is required, not optional: 224 "passed" pairwise rows
reduced to 26 distinct underlying dimensions. The rule: cluster pairwise candidates by
**Jaccard similarity of the `signal_id` sets behind each candidate's `n_success`** — two
patterns whose positive examples overlap heavily are the same underlying event, not
independent findings, regardless of which dimension pair produced them.

1. Extend `PatternCandidate` (Phase 1) with the member `signal_id` set for its
   `n_success` rows — a small, additive field; the aggregate stats it already carries are
   unaffected.
2. For each passed pairwise candidate, compute pairwise Jaccard similarity of these sets;
   union-find/single-linkage cluster at a similarity threshold (start at 0.6).
3. From each cluster, keep one representative (lowest adjusted p-value) plus
   `member_count` (how many raw rows the cluster absorbed) — this is the number
   `hypothesis_agent.py` actually sees, not the raw 224.
4. Cheaper fallback if Jaccard clustering proves unnecessary: group by shared dimension
   membership with an already-passing single-order dimension (e.g. every pattern
   involving `vol_ratio_20d`, which independently passed, is "explained by" it unless the
   paired dimension adds lift beyond what combining two correlated signals would predict
   anyway). Less rigorous, cheaper to compute — a fallback, not the default.

### Smallest safe first step

Build `pattern_dedup.py` **alone**, no LLM involved yet. Run it against the Phase 1
report already on disk (`data/reports/pattern_miner_2026-07-13.json`) and manually verify
the 224 pairwise rows collapse to roughly the ~26-dimension count Phase 1's report
already flagged. Only once that's confirmed by hand does `hypothesis_agent.py` get built
on top of trustworthy input — same "prove the input is right before adding the
expensive/risky next stage" discipline Phase 1 itself used (statistics before LLM),
applied one level deeper (de-duplication before LLM-within-Phase-2).
