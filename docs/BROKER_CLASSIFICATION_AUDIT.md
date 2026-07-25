# Broker Classification Audit

Phase 1 of the Retail Filter initiative — see `docs/AI_LAB_ARCHITECTURE.md` for
unrelated AI Lab context; this document is standalone.

## Why this exists

Before adding a "hide retail-dominated stocks" filter, an architecture review
found **two independent, non-integrated broker classification systems** in
this repo, and several broker-code mappings that conflict with each other
*within the codebase itself* — before any external verification was even
attempted. This document is the full audit: every broker code compared across
every source that claims to classify it, every conflict found, and a
canonical classification for each — defaulting to **Unknown** wherever the
evidence doesn't clearly support a confident answer. Nothing here was guessed.

## Sources compared

1. **`stock_scanner/configs/broker_config.yaml`** (pre-audit) — `broker_groups`
   with inline `#` comments, mostly uncorroborated.
2. **`stock_scanner/pipeline/broker_intelligence.py`** — hardcoded
   `FOREIGN_BROKER_CODES` / `BIG_LOCAL_BROKER_CODES` frozensets. This is the
   system actually powering production (`smart_money_screener.py`, the Smart
   Money tab, and single-ticker broker-detail panels).
3. **`stock_scanner/pipeline/fetch_indexalpha.py`**'s `IDX_BROKER_NAMES` dict —
   the *only* source of `broker_name` actually written into cached data
   (`data/broker/*.parquet`); the Index Alpha API itself returns only a bare
   `code`, never a name.
4. **Ground truth**: every `(broker_code, broker_name)` pair actually observed
   across all 101 files in `data/broker/` (88 distinct codes). Note this is
   *not* independent evidence — `broker_name` in the cache is always whatever
   `IDX_BROKER_NAMES` said *at fetch time*, so it mainly reveals how that dict
   has drifted over time (see ZP below), and which codes are covered at all
   (60 of 88 observed codes currently resolve to `"Unknown"` — i.e. most
   codes that actually trade are unnamed by any source today).
5. **External research** (this session and the prior architecture-review
   session): Indonesian financial-content sites, cross-checked against each
   other and against well-established market knowledge. Treated as
   low-to-medium authority — multiple sources were found to directly
   contradict each other, and one called Credit Suisse "popular among retail
   investors," which is simply wrong. No source claiming to be the Scribd
   document's content was accessible (paywalled beyond its title/metadata).

## Methodology for confidence and conflict resolution

- **Ownership vs. client base are different axes.** A foreign-owned broker
  (e.g. Mirae Asset, Korean) can still have an overwhelmingly retail client
  base; a 100%-domestic broker (e.g. Mandiri Sekuritas) can be
  institution-focused. `type` below reflects client-base composition where
  determinable — the actual thing a "retail accumulation" filter cares about
  — with ownership noted separately when it's informative.
- **A claim repeated across files is not independent corroboration** if the
  files share a common origin. `broker_config.yaml` and
  `broker_intelligence.py`'s comments are near-identical in most places —
  they were evidently generated together (or one copied from the other), so
  agreement between them counts as *one* source, not two.
- **A systemic pattern emerged while auditing, not just isolated errors**:
  the same real company name is claimed for *multiple different* 2-letter
  codes in several places — "Indo Premier" for both YP and PD; "BNI
  Sekuritas" for BW, BQ, *and* (externally) NI; "Trimegah Sekuritas" for both
  YJ and LG; a "BRI Danareksa / Danareksa" family claimed across GI, OD,
  *and* ZU. A single real IDX-registered company has exactly one code, so
  each of these clusters means at least one member is wrong — and I have no
  reliable way to determine *which* one without an authoritative IDX source.
  Every code inside a collision cluster is marked **Unknown**, not just
  "conflicting" — picking a winner from inside the codebase's own
  contradictions would be guessing.
- **Confidence tiers**: **High** = independently corroborated by 2+ sources
  that don't share an origin, or an unambiguous globally-known institution
  name (Goldman Sachs, Morgan Stanley, etc.) where confusion is implausible.
  **Medium** = one internal source plus supporting (but not fully
  independent) external corroboration, or a single internal source with no
  contradiction found. **Low** = single, uncorroborated source, or
  plausible-sounding but unverified. **None** = genuine conflict between
  sources with no way to adjudicate — always resolves to **Unknown**.

## Audit table

`legacy_type` is what `broker_intelligence.classify_broker()` returns
*today* and continues to return after the refactor (`foreign` / `big_local` /
`local` / `unknown`) — preserved unchanged to avoid altering
`smart_money_screener.py`'s live output. `type` is the new, audited
canonical classification (`retail` / `institutional` / `foreign` /
`mixed_unknown`) that a future retail filter will actually use. These
intentionally diverge in several rows — see Notes.

| Code | broker_config.yaml | broker_intelligence.py | IDX_BROKER_NAMES (live) | External research | Conflict? | Canonical name | Type | Confidence | Notes |
|---|---|---|---|---|---|---|---|---|---|
| XL | retail, no name | — | — | Stockbit Sekuritas Digital | No — just previously unclassified | Stockbit Sekuritas Digital | **Retail** | High | #1 by transaction frequency nationally (CNBC Indonesia); was in `retail` group with no name at all |
| XC | retail, no name | — | — | Ajaib Sekuritas Asia | No | Ajaib Sekuritas Asia | **Retail** | High | Top-4 by frequency (emitennews); same gap as XL |
| YP | big_local, "Indo Premier Sekuritas / Mirae Asset" (hedged) | big_local, same hedge | "Indo Premier Sekuritas" | Mirae Asset Sekuritas Indonesia | **Yes** — internally hedges between 2 names, one of which (Indo Premier) duplicates PD's claim | Mirae Asset Sekuritas Indonesia | **Retail** (client base) / Foreign (ownership) | Medium-High | The "Indo Premier" claim for YP collides with PD's own, independently-better-supported claim to that same name — a real company can't have two codes. #2 by tx frequency nationally, consistent with retail-app dominance |
| PD | big_local *and* institution, "Henan Putihrai" | big_local, "Henan Putihrai" | "Indo Premier Online" | Indo Premier Sekuritas | **Yes** — internal "Henan Putihrai" claim (from 1 shared origin) vs. live dict + external evidence | Indo Premier Sekuritas | **Retail** | Medium | IDX_BROKER_NAMES (the live, production-used source) and external research agree; "Henan Putihrai" is a different real firm and appears to be a labeling error. Also affected by the group-overlap bug (listed in both `big_local` and `institution`) |
| CC | big_local, "Mirae Asset Sekuritas" | big_local, "Mirae Asset Sekuritas" | "Mandiri Sekuritas" | Mandiri Sekuritas | **Yes**, but resolvable | Mandiri Sekuritas | **Institutional / Mixed** | High | Live dict + strong external corroboration (¬Rp651T trading value, #1 nationally, Bank Mandiri's IB arm) both agree; the config comment appears to be a copy-paste mix-up with YP |
| NI | big_local *and* institution, "BCA Sekuritas" | big_local, "BCA Sekuritas" | "BCA Sekuritas" | BNI Sekuritas | **Yes** | — | **Unknown** | None | All 3 internal sources agree (though not independently), directly contradicted by external research — and BW/BQ separately both claim "BNI Sekuritas" too (3-way collision on that name). Cannot adjudicate |
| AK | foreign, "UBS Securities Indonesia" | foreign, "UBS Securities Indonesia" | "UBS Securities Indonesia" | UBS Sekuritas Indonesia (EN/ID variant only) | No — trivial language variant | UBS Sekuritas Indonesia | **Foreign / Institutional** | High | Fully agreed everywhere |
| RX | not in config | not in config | not in config | Macquarie Sekuritas Indonesia | No, but a gap | Macquarie Sekuritas Indonesia | **Foreign / Institutional** | Medium-High | Ground-truth cache shows "Macquarie Sekuritas Indonesia" was actually written for RX in some files (historical `IDX_BROKER_NAMES` state), consistent with external research; simply missing from the *current* dict and both config files |
| MG | not in any source | not in any source | not in any source | Semesta Indovest Sekuritas (single low-quality mention) | — | — | **Unknown** | None | No corroborated evidence found anywhere, internal or external |
| AZ | not in any source | not in any source | not in any source | Sucor Sekuritas (single mention, domestic per one source) | — | — | **Unknown** | Low | Domestic ownership claimed by one source; nothing on client-base composition |
| GS, MS, DB, ML, DP, SB | foreign | foreign | matching names | Goldman Sachs / Morgan Stanley / Deutsche Bank / Merrill Lynch / DBS Vickers / Nomura | No | (as listed) | **Foreign / Institutional** | High | Globally famous institutions; no plausible confusion |
| JP | foreign, "JPMorgan Securities" | foreign, "JP Securities / Barclays" (hedged) | "JPMorgan Securities" | — | Internal hedge only | — | **Foreign / Institutional** | Medium | Type is solid (any of the hedged names is a foreign bank); exact identity between JPMorgan and Barclays not resolved |
| BK | foreign, "JP Securities / Barclays" (hedged) | foreign, same hedge | not in dict | JP Morgan Sekuritas Indonesia (external, consistent with common "BK = JPMorgan" bandarmology usage) | Internal hedge; possible JP/BK identity swap with the row above | — | **Foreign / Institutional** | Medium | Same situation as JP, mirrored — type confidence high, exact name uncertain, and JP vs BK may have the JPMorgan label on the wrong code |
| CS | foreign, "Credit Suisse / UBS" | foreign, "Credit Suisse / UBS Securities (post-merger)" | "Credit Suisse / UBS" | one low-quality source incorrectly called this "retail-popular" | No real conflict — the one contradicting source is unreliable | Credit Suisse / UBS Securities | **Foreign / Institutional** | High | Global institutional bank; the retail claim is contradicted by everything else and not credible |
| LS | foreign, "Credit Suisse International" | not present | not in dict | — | Possible duplicate of CS | — | **Unknown** | Low | Same parent name as CS under a different code — plausibly a duplicate/error, not independently verified |
| ZP | foreign, "Macquarie Sekuritas" | foreign, "Macquarie Sekuritas" | "Macquarie Sekuritas" | Maybank Sekuritas Indonesia (external, multiple sources) | **Yes** | — | **Foreign / Institutional** (type only) | Medium (type) / Low (name) | Ground-truth cache shows ZP was **historically** "Kim Eng Sekuritas" before being overwritten — Kim Eng → Maybank Kim Eng → Maybank Sekuritas is a real, well-known acquisition lineage, supporting the external Maybank claim over the current internal "Macquarie" one (which more likely belongs on RX — see above). Type stays confidently Foreign/Institutional either way |
| RB | foreign, "CGS-CIMB Securities" | foreign, "CGS-CIMB Securities" | not in dict | Two external sources gave two *different* other companies (Nikko Sekuritas vs. INA Sekuritas) | **Yes — 3-way** | — | **Unknown** | None | No two sources agree; CGS-CIMB's real code is more likely YU (see below) |
| YU | foreign, "CLSA Securities Indonesia" | foreign, "CLSA Securities Indonesia" | "CLSA Securities Indonesia" | CGS International / CGS-CIMB Sekuritas Indonesia (external, multiple sources); CLSA's real code identified externally as KZ | **Yes** | — | **Foreign / Institutional** (type only) | Medium (type) / Low (name) | Same pattern as ZP/RX — plausible code mix-up between YU and KZ for the CLSA label; type unaffected since CGS-CIMB and CLSA are both foreign institutional |
| KZ | not in any config source | not in any config source | not in dict | CLSA Securities Indonesia (external) | — | CLSA Securities Indonesia | **Foreign / Institutional** | Low-Medium | Complete gap in all 3 internal sources despite being the more likely real CLSA code per external research |
| QA | foreign, no independent comment | foreign, no independent comment | "Citigroup Securities" | Tuntun Sekuritas Indonesia (external, single source) | **Yes** | — | **Unknown** | None | Citigroup (bulge-bracket foreign bank) vs. Tuntun (small domestic broker) are wildly different classifications; 1-vs-1 with no way to adjudicate, and the stakes of guessing wrong are high |
| KK | foreign, "Maybank Securities" | foreign, "Maybank Securities" | not in dict | Phillip Sekuritas Indonesia (external, single source) | **Yes** | — | **Unknown** | None | Maybank (foreign bank) vs. Phillip Capital (large Indonesian retail branch network) would classify very differently |
| HD | big_local *and* institution, "HD Capital" | big_local, "HD Capital Sekuritas" | "HD Capital" | Name agreed; ownership disputed between two external sources | Name: No. Type: **Yes** | HD Capital Sekuritas | **Unknown** (type only) | High (name) / None (type) | Identity is the most solid in this table (3 independent-ish confirmations); but external sources gave opposite foreign/domestic classifications, so type stays Unknown even with a confirmed name |
| OD | big_local, "Mandiri Sekuritas" | big_local, "Mandiri Sekuritas" | "Mirae Asset Sekuritas" | BRI Danareksa Sekuritas (external) | **Yes — 3-way, every source disagrees** | — | **Unknown** | None | The clearest 3-way conflict in the whole audit — internal config, live dict, and external research each name a *different* company |
| BW, BQ | big_local / — , "BNI Sekuritas" both | big_local, "BNI Sekuritas" (BW only) | "BNI Sekuritas" (BQ only) | NI externally claimed as BNI too | **Yes — 3-way collision with NI** | — | **Unknown** (both) | None | Three different codes (BW, BQ, NI) each have a claim to "BNI Sekuritas" — a single company, one code. Cannot determine which, if any, is correct without IDX's registry |
| YJ, LG | big_local, both "Trimegah Sekuritas" | big_local, both "Trimegah Sekuritas" | — | — | **Yes — internal collision** | — | **Unknown** (both) | None | Same name claimed for two different codes within the same file |
| GI, ZU | big_local, "BRI Danareksa Sekuritas" (GI) / "Danareksa Sekuritas" (ZU) | same | — | (OD also externally claimed as BRI Danareksa — see above) | **Yes — collides with OD too** | — | **Unknown** (both) | None | BRI Danareksa Sekuritas is the post-2021-merger name for the entity that used to be Danareksa Sekuritas — plausible these codes track a real historical identity change, but combined with OD's separate claim to the same name, I can't confidently assign a single code |
| KI, MQ | institution/big_local, "Ciptadana Sekuritas Asia" (KI) / "Ciptadana Sekuritas" (MQ) | KI only | — | — | Possible dual-entity, lower severity | Ciptadana Sekuritas (KI), Ciptadana Sekuritas Asia (MQ) | **Mixed / Unknown** (both) | Low | Could legitimately be two distinct registered Ciptadana entities rather than an error — not confident enough to assert either way |
| CP | institution, "Valbury Asia Securities / institutional desk" | not present | "Valbury Asia Securities" | KB Valbury Sekuritas (external — post Korean-acquisition name) | No — consistent lineage, just an ownership update over time | KB Valbury Sekuritas | **Mixed / Unknown** | Low-Medium | Plausible that this is simply an outdated name (pre-acquisition), not a real conflict |
| CD | institution, "(Corporate direct)" | not present | not in dict | — | — | — | **Unknown** | None | The config's own comment admits this isn't a real company name |
| EM, IX | retail, "(Retail segment)" / "(Retail platform)" | not present | not in dict | — | — | — | **Unknown** (name) / plausibly Retail (unverified) | None | The config's own author didn't commit to real names for these; neither code appears anywhere in the 88 codes actually observed in cached trading data — may not be active/valid codes at all |
| ID, EP, AD, DH | big_local, single unverified names each | same | — | — | — | (as listed, unverified) | **Mixed / Unknown** | Low | Single-sourced internal claims, no independent cross-check attempted; plausible but not verified |
| ZH | big_local, "CIMB Niaga Sekuritas" | same | — | — | — | CIMB Niaga Sekuritas | **Foreign / Institutional** (tentative) | Low | CIMB is a known Malaysian banking group; single-sourced, not independently verified |
| MK, OX, SA, UX | foreign, single unverified names each | same | — | — | — | (as listed) | **Foreign / Institutional** (tentative) | Low-Medium | Plausible real foreign institutions (Daiwa, CIMB, Societe Generale, UOB Kay Hian); single-sourced, not independently re-verified this pass |
| FZ | big_local, "Waterfront Sekuritas" | same | "Waterfront Sekuritas" | — | No | Waterfront Sekuritas | **Mixed / Unknown** | Medium | Name agreed by 2 sources including the live dict; no evidence on client-base composition |
| DR | big_local, "OSO Sekuritas" | same | "OSO Sekuritas" | RHB Sekuritas Indonesia (external, multiple sources) | **Yes** | — | **Unknown** | None | Live dict + config agree with each other, but multiple independent external sources say DR = RHB, a completely different (and foreign-owned) company |
| IN | big_local, "Investindo Nusantara" | same | "Investindo Nusantara" | — | No | Investindo Nusantara Sekuritas | **Mixed / Unknown** | Medium | 3 internal sources agree (name only); no client-base evidence |
| MU | big_local, "Minna Padi Investama" | same | "Minna Padi Investama" | — | No | Minna Padi Investama | **Mixed / Unknown** | Medium | Same as IN — name well-attested internally, type unverified |
| YB | big_local, "Panin Sekuritas" *and* retail, "(retail desk)" | big_local, "Panin Sekuritas" | — | — | **Yes — internal group-overlap bug** | Panin Sekuritas | **Mixed / Unknown** | Low | Concrete example of the multi-group-membership bug in the pre-audit config — `classify_brokers_in_df()` would have silently picked whichever group came later in YAML order, with no warning |

### Codes observed in real trading data with zero classification anywhere

61 of the 88 codes actually seen in `data/broker/*.parquet` never appear in
*any* of the three config sources at all (beyond the handful already covered
above — RX, KZ, MG, AZ). These resolve to `broker_name = "Unknown"` in the
live cache today and would fall to the `mixed_unknown` default under the new
schema. Full list: `AF, AG, AH, AI, AO, AP, AR, AT, BB, BF, BR, BS, CD*, DD,
DU, DX, EL, ES, FS, GA, GR, HP, IC, ID*, IF, IH, II, IT, IU, KZ*, OK, PC, PF,
PG, PI, PO, PP, RF, RG, RO, RS, SF, SH, SQ, SS, TF, TP, TS, XA, YO, ZR` (`*`
already covered above, others genuinely blank). No individual research was
attempted on these 51 — there's no realistic way to responsibly classify 51
codes with zero starting evidence in one pass. They're recorded here as an
honest inventory gap, not silently left out.

## What changed vs. the pre-audit config

- **12 codes moved to a confident, evidence-backed classification for the
  first time**: XL, XC (Retail — previously in `retail` with no name at
  all), CC (Institutional — corrected from a mislabeled comment), YP, PD
  (both corrected from a hedged/wrong internal name to their better-supported
  identity), RX, KZ (gap-filled from external + historical-cache evidence).
- **9 codes moved to Unknown from a previously-confident classification**
  (NI, OD, RB, QA, KK, DR, and the collision clusters BW/BQ, YJ/LG, GI/ZU) —
  each because auditing surfaced a real conflict the pre-audit config had
  simply never checked for. This is the audit doing its job, not a
  regression: a wrong confident answer is worse than an honest "unknown" for
  a filtering feature.
- **`legacy_type`** (feeding `broker_intelligence.classify_broker()`) is
  preserved exactly as today's hardcoded sets for every code, regardless of
  what the audit found — Smart Money's live output does not change as a
  result of this document. Reconciling `legacy_type` with the new `type`
  where they disagree is a deliberate follow-up decision, not made here.
