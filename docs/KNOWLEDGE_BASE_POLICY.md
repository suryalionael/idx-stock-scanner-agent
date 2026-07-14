# Knowledge Base Policy

Operational policy for `knowledge_base` — who may set what, what evidence is required,
and what each status does and does not mean. Companion to
`docs/LEARNING_AGENT_RUNBOOK.md` (the "how to run the tools" doc) and
`docs/LEARNING_AGENT_ARCHITECTURE.md` (the design). This doc is the "what are the rules"
reference. Practical, not aspirational — every rule here maps to an actual constraint
already enforced in code, or an actual gap a human must fill in by judgment.

---

## Production behavior is unchanged by this table, unconditionally

No status value, no row, no field in `knowledge_base` is read by
`stock_scanner/pipeline/`, `signal_engine.py`, `scanner_config.yaml`, `run_daily_scan.py`,
or any GitHub Actions workflow. There is no code path from this table into production
scoring, classification, ranking, or model promotion. The only way a `knowledge_base`
finding ever affects production is the fully manual, human-driven path described below —
and even that path ends at `scripts/train_challenger.py` / `scripts/promote_challenger.py`,
which are themselves unmodified by anything in this system.

---

## Status values — meaning, and who/what may set each one

| Status | Meaning | Who/what may set it |
|---|---|---|
| `candidate` | Fresh output of `hypothesis_agent.py` — an LLM has articulated a statistically-gated pattern, nobody has looked at it yet. | **Only** `scripts/run_hypothesis_agent.py` (via `write_hypotheses()`), which a human runs manually. Never set by anything else. |
| `reviewed` | A human has read the hypothesis and its backing statistics (via `--show`) and judges it plausible, but is not yet acting on it. | **Only** a human, via `scripts/review_knowledge_base.py --decide`. |
| `archived` | A human has read it and decided it is not worth pursuing — weak stats, high ticker concentration, unstable across time, or just not interesting. | **Only** a human, via the review CLI. |
| `testing` | A human has manually translated the hypothesis into a concrete change to `scripts/train_challenger.py` (a feature or threshold to test) and is running or about to run it. | **Only** a human, via the review CLI — and only after they've actually made that edit themselves. Nothing automates this translation. |
| `tested_passed` / `tested_failed` | The real `train_challenger.py` run above produced a result. | **Only** a human, via the review CLI, **required** to also pass `--linked-model-version-id` pointing at the resulting `model_registry` row. The database's foreign-key constraint rejects a `model_version_id` that doesn't actually exist. |
| `promoted` | Record-keeping annotation: the linked model was, separately, actually promoted. | **Only** a human, via the review CLI, and **only after independently confirming** `model_registry.status = 'promoted'` for that `model_version_id` — see the explicit warning below. |

There is no status meaning "auto-approved" or "system-promoted." Every non-`candidate`
status requires a human action through `scripts/review_knowledge_base.py`.

---

## `promoted` is record-keeping, never execution

Setting `knowledge_base.status = 'promoted'` **does nothing except record a fact you have
already independently verified.** It does not call `promote_challenger.py`. It does not
change `model_registry`. It does not affect what the morning scan does. The actual,
consequential act of promotion is exclusively `scripts/promote_challenger.py` flipping
`model_registry.status`. Before ever setting `promoted` here, check that directly:

```bash
python3 -c "
from stock_scanner.db.init_db import get_connection
conn = get_connection()
print(conn.execute(\"SELECT model_version_id, status FROM model_registry WHERE model_version_id=?\",
                   ('<the linked model_version_id>',)).fetchone())
"
```

If that doesn't show `status='promoted'`, do not set it here yet — you'd be recording
something that hasn't happened.

---

## Required evidence before `candidate` → `reviewed`

Do not move a hypothesis to `reviewed` on the strength of its `hypothesis` text or
`confidence` score alone — both are LLM-authored and qualitative. Before marking
`reviewed`, run `--show` and actually look at:

1. **`ci_lower`** — the conservative bound on win rate, not the point estimate. This is
   the number the statistical gate actually required to exceed baseline.
2. **`p_value_adjusted`** — already FDR-corrected; a small number here is doing real work,
   don't re-derive significance from `confidence` instead.
3. **`ticker_concentration_flag`** — if true, a large share of the pattern's supporting
   trades come from one ticker. Treat this as a reason to lean toward `archived` unless
   you have an independent reason to think the ticker isn't an outlier.
4. **`time_split_stable`** — `None` means inconclusive (too little data to check), not
   "yes." Only treat `True` as actual evidence of stability.
5. **`member_count`** on the source cluster — a representative of a large cluster is
   backed by more corroborating (if redundant) findings than a `member_count=1` cluster.

If you can't articulate, in one sentence, why the `ci_lower`/`p_value_adjusted`/flags
support the hypothesis, it isn't ready for `reviewed` — send it to `archived` or leave it
at `candidate` until more data accumulates.

---

## Sector-specific hypotheses — when they're allowed despite thin coverage

`sector_reference` covers roughly 11% of the trading universe (as of the last check —
verify current coverage with `SELECT COUNT(*) FROM sector_reference` before relying on
this number). A hypothesis with `affected_sector` set is not automatically suspect — it
already passed the same statistical gate as everything else — but it carries an
**additional, distinct risk**: the covered subset of tickers may not represent that
sector's true behavior (it's a curated list, not a random or complete sample).

**Rule:** a sector-specific hypothesis may be reviewed and archived like any other
without restriction. It may **not** move to `testing` (i.e., must not inform an actual
`train_challenger.py` change) unless at least one of:

- **(a)** You manually check `sector_reference` coverage for that specific sector
  (`SELECT COUNT(*) FROM sector_reference WHERE sector = '<X>'`) and judge it large and
  diverse enough to generalize — there is no single universal threshold since coverage
  varies by sector; this is a judgment call to make explicitly, not skip.
- **(b)** The same underlying feature combination *also* clears the statistical gate
  without the sector restriction (check the single/pairwise report for the non-sector
  version of the same dimensions) — if the effect only appears when sliced by this
  thin sector sample, treat it as more likely to be sampling noise than a real
  sector-specific effect, and archive it instead.

If neither holds, mark it `reviewed` at most, with a note (via `--reviewed-by`, or your
own separate tracking) that it's pending better sector coverage — do not act on it yet.

---

## What this policy deliberately does not add

No automatic status transitions, no required approval chain, no schema changes beyond
what already exists, and no enforcement of a strict status state-machine in code (e.g.
"testing" cannot be reached except from "reviewed") — `scripts/review_knowledge_base.py`
trusts the single operator's judgment on ordering. If more than one person starts using
this table, revisit that decision; it is not a safe assumption at that point.
