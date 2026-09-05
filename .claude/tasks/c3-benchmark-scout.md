# c3-benchmark-scout — Track-A baseline conventions + Track-B admissibility screen

**Campaign 3, wave 1 (THE REAL-DATA WAVE). Agent:** web-scout. **ZERO worktrees. ZERO branches. ZERO
repo edits.** You are read-only on the repo and write-only under `.claude/`.
**Runs FIRST — you have no precondition and nothing waits on the harness spoke.**
Writes `.claude/outputs/c3-benchmark-scout.md` + artifacts to `.claude/outputs/c3-benchmark-scout/`.
**Budget:** ≈ 1.5 days.

**Binding documents, read first, in this order:**
1. `.claude/advisor-head-c3-charter.md` **IN FULL** (it is short) — especially **§3 (venues + the
   admissibility rule)** and **§2 (claim architecture)**.
2. `.claude/advisor-head-intervention.md` **§6 IN FULL** — the five criteria you score against, and
   **§8 prohibitions** (1–4). These are still in force across the campaign boundary.
3. `.claude/AGENT_PROTOCOL.md` §5 (output format), §7 (dial declaration).

---

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result

- **Dial / pillar:** **none — recon/scout.** You produce **no** performance number of our own, **no**
  admissibility verdict that criterion 4 depends on, and **no** venue adoption. You produce a *cited
  brief* and a *scorecard*, and the Hub + Advisor adopt or reject.
- **Laundering control:** n/a to you directly — but **every rival number you report must be reported
  next to the naive/linear/classical baseline number from the same source** (that pairing is what
  criterion 2 is scored on; a SOTA number alone is not usable).
- **Falsifies the brief:** a shortlist entry whose criterion-2 pairing you could not find, or whose
  criterion-4 tripwire you could not specify concretely enough to run.
- **Does NOT falsify the brief:** finding that **zero** Track-B venues survive the screen. A clean
  "none of these are admissible, here is the measured reason each dies" is a **finding and a
  deliverable**, not a failure. ⛔ Do not manufacture a survivor.

---

## ⛔⛔ THE FOUR HARD RULES — a report violating any one of these is sent back

1. **⛔ CRITERION 4 IS MEASURED, NEVER ARGUED.** You do **not** score criterion 4 pass/fail for any
   venue. Ever. For each shortlisted venue you instead **specify the tripwire run** (§2.3) precisely
   enough that the engineer can execute it without a second round-trip. The theorem has **SIX
   confirmations**; ⛔ **we are not shopping for a seventh metric-native venue** — if a venue smells
   metric-native, say so, spec the tripwire anyway, and rank it below the ones that do not.
2. **Every number carries a citation to the PRIMARY source** — arXiv id / venue + **the specific
   table or section number** it came from. ⛔ **Never relay a number out of a secondary source, a
   blog, a leaderboard aggregator, or another paper's comparison table** without going to the
   originating paper and confirming it. (Program precedent: four Advisor errata all came from
   relaying numbers out of text instead of re-deriving from source.)
3. **A missing published baseline is a NOT-RUN, never a null and never a win.** If no rival has a
   published number at our weight class on a venue, write **"NOT PUBLISHED at this weight class"**
   and state what it would cost *us* to train it. ⛔ Do not substitute a number from a different
   parameter count, different tokenizer, or different context length and call it comparable — say
   explicitly that it is not.
4. **⛔ No CAFE C-MAPSS number is externally comparable** (the banked label-bug report). If C-MAPSS or
   any CAFE-derived prognostics set appears in your sweep, it carries that embargo in-line, every
   time. Do not quote a CAFE number as evidence for or against anything.

---

## 1. TRACK A — the ready spine (bounded-state LM at 26–47 M)

Our weight class is **26–47 M params, from scratch, matched-param AND matched-state-byte**. The
primary is **enwik8** (byte-level, bpc, vocab 256, canonical 90/5/5 positional split — already built
at `chlu/data/enwik8.py`); **WikiText-103** second (`chlu/data/wikitext.py`, byte + word modes both
built); **PG-19** is the long-horizon candidate and is **not built**.

### 1.1 What to return, per rival × per venue

Rivals: **Mamba-2** · **Gated DeltaNet-2 (GDN-2)** · **TTT / Titans** (the TTT-class cell is our
**system-level swap**, the two-sided control — its numbers matter most) · **sliding-window
attention** · plus **vanilla Transformer** as the reference point the field quotes against.

For each cell of the (rival × venue) grid:

| field | what it must contain |
|---|---|
| number | bpc (enwik8) / ppl (WT-103, PG-19), **at the closest published param count**, with that count stated |
| provenance | arXiv id + **table/section**, and whether it is the authors' own or a third-party reproduction |
| comparability | ⚠ tokenizer, context/sequence length, train-token budget, and whether the split is canonical. Any mismatch to our setup gets an explicit **"NOT COMPARABLE because …"** |
| state bytes | the rival's **recurrent/KV state size at inference** in bytes, derived from its own hyperparameters (the matched-state-byte control needs this; if the paper does not state it, derive it and **show the arithmetic**) |
| gap to us | NOT PUBLISHED / comparable / adjacent-only |

⚠ **Expected and important:** most modern SSM/linear-attention papers report on **The Pile /
SlimPajama / LongBench**, not enwik8, and often at 130 M+. If the 26–47 M enwik8 cell is empty for a
rival, that is the finding — say so, and answer §1.3.

### 1.2 Conventions (this is as valuable as the numbers)

- **The dynamic-evaluation substitute column.** Every LM table we ship carries one (invariant, C3
  charter §5). What is the field's current dyn-eval protocol at this scale, what does it buy on
  enwik8/WT-103 in bpc/ppl, and which papers report it? This column is a *standing obligation* — we
  need its convention pinned, not improvised.
- **Within-document retention / revisit slices on real text.** Who computes them, how are they
  defined (byte distance? token distance? binned how?), and is there any published convention we
  should match rather than invent? ⚠ **We are building this instrument in the same wave**
  (`c3-csf3-harness` §2) — if a convention exists, we adopt it; if the field has none, say so
  explicitly so we can declare ours as ours.
- **Reporting norms:** seeds (how many do papers actually run?), error bars, whether bpc is quoted at
  a fixed eval context or at the training context, and the standard eval-context ablation.

### 1.3 PG-19 feasibility at our weight class

Answer mechanically, not impressionistically: corpus size on disk, tokenization/vocab convention,
the published protocol (eval context length, chunking), **what a 26–47 M from-scratch run costs** in
A100-hours under a **2×A100 / 4-day per-job envelope**, and whether the published numbers at any
weight class are close enough to ours to be a reference at all. Deliver a **GO / GO-WITH-CAVEAT /
NO-GO** with the arithmetic shown. A NO-GO here is a perfectly good answer.

---

## 2. TRACK B — the Head's direction, screened hard (long-horizon multivariate time series)

**Direction (binding):** physical AI / world models — **long-horizon multivariate time series first**
(forecasting · multi-step rollout · regime re-identification · prognostics); long video later.

### 2.1 The screen: intervention §6's five criteria, restated so you score them

1. **Strong baselines that do well** — attention/SSM/GRU/MLP are competitive on it. A task the
   competition fails *by construction* is inadmissible as a primary claim.
2. **Real headroom** — nothing saturated, no trivial method at ceiling.
3. **Memory management over time is the difficulty** — retention, interference, capacity pressure,
   selective recall under load. ⛔ Not single-shot lookup.
4. **NOT metric-native to the store** — ⛔ **YOU DO NOT SCORE THIS.** See §2.3.
5. **Every lever can be active** — learned φ/ψ, a live controller, lifetimes, and retry all matter at
   once on this task.

### 2.2 The two known hazards — check them FIRST, they kill most candidates

- ⚠ **Criterion 2 is where long-horizon forecasting suites die.** On the standard LTSF suites, simple
  linear/naive baselines sit at or near the frontier. **For every forecasting candidate you must
  report the best simple-baseline number beside the best deep number, from primary sources.** If the
  gap is not clear and large, the venue **fails criterion 2** and you say so.
- ⚠ **Criterion 4 is where classic streaming venues die** — six confirmations on the record.

### 2.3 The criterion-4 tripwire spec (the deliverable, not the verdict)

For each shortlisted venue, specify the **matched-bytes exemplar-store probe** (the instrument C2W10
built) concretely enough to run:

- the **query and key representations**, and whether they plausibly share a metric space (state the
  structural reason, as a hypothesis to be *tested*, not a conclusion);
- the **exemplar/kNN baseline form** the tripwire uses, and its **state-byte budget** — matched to
  what our block would carry;
- the **numeric outcome that means "dies"**: at what margin does the classical store beating the
  learned block at matched bytes declare the venue inadmissible;
- what data/preprocessing the probe needs, and roughly what it costs (minutes? a job?).

Point at `.claude/outputs/` for the C2W10 drift-map + embargo methodology and reuse its shape rather
than inventing one — locate it yourself and cite the file path you used.

### 2.4 Per-venue scorecard (the shortlist's required schema)

One row per candidate, ranked. Emit as **`.claude/outputs/c3-benchmark-scout/trackB-scorecard.json`**
as well as a table in the report, with these fields:

```
venue, modality, task_form, size_on_disk, license, splits_canonical,
crit1_baselines: {verdict, best_deep, best_naive, source},
crit2_headroom:  {verdict, evidence},
crit3_memory_is_the_difficulty: {verdict, evidence},
crit4_tripwire:  {SPEC_ONLY, query_repr, key_repr, exemplar_baseline,
                  state_byte_budget, dies_if, cost_estimate},   # ⛔ no verdict field
crit5_all_levers: {verdict, evidence},
csf3: {available, how_obtained, staging_notes},
overall: {RECOMMEND | RECOMMEND-IF-TRIPWIRE-CLEARS | REJECT, one_line_reason}
```

⛔ `crit4_tripwire` **has no verdict field by design.** If you add one, the row is invalid.

**Deliver: one primary recommendation + one fallback**, or an explicit "no admissible candidate
found" with the per-venue cause of death.

---

## 3. CSF3 data availability & licensing (both tracks)

For every dataset that reaches your shortlist (incl. PG-19 if it survives §1.3): where it is fetched
from, its **licence and any redistribution restriction**, size on disk, and whether it can be staged
on CSF3 given our conventions — **download-once, serial staging before a sweep, concurrency-safe
atomic-rename fetch** (the pattern in `chlu/data/enwik8.py`'s docstring and
`chlu/data/industrial/base.py`'s `download_file`). Flag anything requiring credentials, a click-
through EULA, or a login — those are blockers the Head must clear, not things an engineer can solve.

---

## 4. Stop conditions

- If a Track-A cell requires paid access or a source you cannot reach: mark **NOT OBTAINED**, name
  the source, move on. ⛔ Do not guess the number.
- If your sweep suggests the *entire* Track-B modality dies on criterion 2 or 3: **stop widening the
  net, write that up as the finding**, and give the Hub the evidence. That is a legitimate outcome.
- ⛔ Do not propose a venue outside the Head's ratified direction (§3 of the charter) as a primary.
  Adjacent modalities may appear in a clearly-labelled **"noted, not recommended"** appendix only.

## 5. Acceptance criterion (one line)

`.claude/outputs/c3-benchmark-scout.md` + `trackB-scorecard.json` exist; every Track-A grid cell is
either a primary-sourced number with table-level provenance and a comparability verdict or an
explicit NOT-PUBLISHED; §1.3 returns GO/CAVEAT/NO-GO with arithmetic; every Track-B row carries a
runnable criterion-4 tripwire spec **and no criterion-4 verdict**; a primary + fallback (or a
reasoned none) is named.

## 6. Report format

Protocol §5. Open with the dial declaration, then a **≤ 10-line executive answer** (the two
recommendations, the PG-19 verdict, and the single biggest risk), then the sections above.
If your report contains a downstream reconciliation list, say so **in the first 10 lines** (§5
corollary) — the Hub converts it into an owned task at review.
