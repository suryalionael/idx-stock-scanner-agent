# Learning Agent — Operator Runbook

Manual, on-demand research tool. **No cron, no scheduled workflow, no production
integration.** Every step here is run by a human, when a human decides to run it. See
`docs/LEARNING_AGENT_ARCHITECTURE.md` for the design; this doc is the "how do I actually
use it" companion.

---

## TL;DR

- **What it produces:** candidate hypotheses in the `knowledge_base` table — plain-English
  descriptions of statistically-gated patterns, for a human to read and decide what (if
  anything) to do next.
- **What it can never do:** change `signal_engine.py`, `scanner_config.yaml`, or
  `model_registry`; promote anything; run on a schedule. Every write stays inside
  `knowledge_base`.
- **Cadence:** whenever you decide to run it. There is no expectation of a fixed rhythm.

---

## The pipeline, step by step

Run in this order — each step reads the previous step's output file, so re-running an
earlier step and forgetting to re-run the later ones will use stale input.

```bash
# 1. Statistics only — no LLM. Writes data/reports/pattern_miner_{date}.json/.md
python scripts/run_learning_agent.py

# 2. De-duplicate pairwise findings. Writes data/reports/pattern_dedup_{date}.json/.md
python scripts/run_pattern_dedup.py

# 3. LLM articulation. Writes knowledge_base rows + data/published/knowledge_base.json
python scripts/run_hypothesis_agent.py --mock   # 9router not wired yet — see below
```

**On `--mock`:** `scripts/run_hypothesis_agent.py` without `--mock` uses `NineRouterClient`,
which currently raises `NotImplementedError` — 9router's API contract isn't configured
yet. This is intentional: it fails loudly rather than silently producing text that looks
like a real hypothesis. Use `--mock` until that's wired.

---

## Reviewing candidates

```bash
# See everything waiting for review
python scripts/review_knowledge_base.py --list

# Filter by status
python scripts/review_knowledge_base.py --list --status candidate

# Full detail on one — including the actual statistics behind it, not just the LLM text
python scripts/review_knowledge_base.py --show <hypothesis_id>

# Record your decision
python scripts/review_knowledge_base.py --decide <hypothesis_id> --status reviewed --reviewed-by "your_name"
```

**Always check `--show` before trusting a hypothesis, not just the one-line summary from
`--list`.** The confidence score is the LLM's qualitative framing — it is not a p-value.
The numbers that actually matter are in the "Source statistical pattern" block: `n` /
`n_success`, `ci_lower` (the conservative bound), `p_value_adjusted`, and especially
`ticker_concentration_flag` (⚠ if one ticker dominates the supporting trades) and
`time_split_stable` (whether the effect holds across both halves of the time range —
`None` means inconclusive, not "yes").

---

## Status vocabulary — what each one means and what to do

| Status | Meaning | Your action |
|---|---|---|
| `candidate` | Fresh from `hypothesis_agent.py`, unread. | Read it via `--show`. |
| `reviewed` | You looked at it; plausible, but not yet worth testing. | Leave it, or come back later. |
| `archived` | You decided it's not worth pursuing (weak stats, high ticker concentration, unstable across time). | Nothing further. |
| `testing` | You've manually translated it into a `train_challenger.py` feature/threshold change and are running that. | Run the real training script (see below). |
| `tested_passed` / `tested_failed` | Outcome of that real training run. | Set `--linked-model-version-id` to the resulting `model_registry` row. |
| `promoted` | Record-keeping only — the linked model eventually got promoted. | Nothing further; see the guardrail below. |

**Translating a hypothesis into an actual test is a manual step this tool does not do.**
Read the hypothesis and its source pattern, decide whether it suggests a concrete change
to `_FEATURE_LIST` or the threshold search space in `scripts/train_challenger.py`, edit
that file yourself, and run:

```bash
python scripts/train_challenger.py
```

unchanged, exactly as it already runs for the existing champion/challenger loop. Only
after that produces a real `model_registry` row do you come back and set
`tested_passed`/`tested_failed` with `--linked-model-version-id`.

---

## Guardrails — what this tool will not let you pretend

- **`knowledge_base.status='promoted'` is not a promotion.** The only thing that actually
  promotes a model is `scripts/promote_challenger.py` changing `model_registry.status`.
  Setting `promoted` here is you annotating, after the fact, that the linked model got
  promoted through that real pipeline — it has zero effect on production either way.
  Check `model_registry.status` directly if you need the actual truth.
- **`--linked-model-version-id` is checked against reality.** The database's own
  foreign-key constraint rejects a `model_version_id` that doesn't exist in
  `model_registry` — you cannot link a hypothesis to a model that was never actually
  trained.
- **`--decide` on a nonexistent `hypothesis_id` tells you so.** It does not silently
  report success — an `UPDATE` matching zero rows is reported as "nothing updated," not
  swallowed.

---

## Known limitations (carried over from Phase 1/2, unchanged)

- 9router isn't wired — every hypothesis today comes from `--mock`, i.e. is a canned
  placeholder string, not real LLM output. Nothing in `knowledge_base` right now should
  be read as an actual research finding until the real client is configured.
- Data volume is thin (roughly 60 total positive labels at last count) — expect few
  hypotheses per run, and treat `confidence` and `supporting_trades` with the same
  caution the Phase 1 report already flagged.
- Sector coverage is ~11% of the universe — sector-specific hypotheses are the least
  trustworthy category until `sector_reference` grows.
- Triple-feature (order-3) patterns never reach this pipeline at all — still
  exploratory-only, per Phase 1.
