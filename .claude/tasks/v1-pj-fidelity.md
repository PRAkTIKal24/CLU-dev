# V1 — fidelity audit of the Head's `pj_sub.tex` condensation

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-26.** The Head asks: **"make sure no information was lost or misrepresented during the editing."**

**Agent:** `paper-referee` (Bash-capable — the byte-identity criterion needs `md5`).
**Output:** `.claude/outputs/v1-pj-fidelity.md` · **Writes that one file and nothing else.**

---

## 0. ⛔⛔ THE EDIT BAR — absolute

**`pj_sub.tex` is the Head's own text and is EDIT-BARRED.** You **report** defects; you **never** fix them, not a typo, not a number, not a heading.
⇒ **Print `md5 pj_sub.tex` at the start and at the end of your pass. They must be identical.** If you believe an edit is needed, that belief goes in the report.

---

## 1. The two objects

| file | role | md5 at scoping | size |
|---|---|---|---|
| `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` | ⭐ **THE OBJECT.** The Head's condensation. | `301ecdf5ed117544cfb12d346fbb7d91` | 240 lines · 3,981 tex-words |
| `.claude/NIPSsubmission/v1-ttcl/submission.tex` | **THE SOURCE OF TRUTH.** The Advisor-accepted revision-1 base. | `caef2272f9dc96d349b46486563d24ee` | 555 lines · 12,331 tex-words |

⇒ **Survival ≈ 32 %.** ⭐ **That number sets the shape of this audit.** At this ratio the dominant risk is **not misquotation — it is OMISSION of mandatory content.** A paper that keeps every number it prints and drops the qualifier beside it is *"a materially less-qualified paper without a single number having been edited"* — a defect **no diff can see**, measured on a sibling short once already.

⚠ The base is itself only ~1 hour old (revision pass 1, Advisor-verified: 9 hunks, zero unenumerated diffs). Restored appendices and a three-result fold both landed in it. **Anything present in the base and absent here left during this condensation.**

---

## PART A — the Head's question: are the numbers intact?

For **every numeric token** in `pj_sub.tex`, find its ancestor in `submission.tex` and match on **all six axes**: value · precision · units · ± · seed count · scope clause.

⛔ **A number with NO ancestor in the base is the most serious finding available in this pass.** Report it first, whatever else you find.

Also report: numbers whose **precision changed** (a bound becoming a point estimate is a claim change, not a rounding), and any number whose **± or seed count was dropped**.

---

## PART B — ⭐ the companion question the Head did not ask, and the one that matters at 32 %

**Walk this do-not-cut list item by item.** For each: **PRESENT / PARTIAL / ABSENT** in `pj_sub.tex`, and — where absent — ⛔ **name the specific claim that now stands unqualified**, ranked by consequence.

| # | mandatory object | why it cannot be dropped |
|---|---|---|
| **B1** | The **C-2 grade label** — *"Reporting grade: verification… oracle channel placement, dim 2/4, 5 seeds, laptop-CPU, γ=0"* | Under C-2 this is the **grade**, not a caveat. Detached, §3 — two of six contributions and the entire headline — reads as an empirical result rather than verification of an exact theory on a designed testbed. |
| **B2** | ⛔⛔ **MF-C**: *"at a **400-epoch** budget"* **+** *"at convergence… the gate's accuracy **matches** full-budget… the payoff is **rationing, not accuracy**"* | **The single highest-risk relocation in the document.** Drop it and the paper re-asserts an accuracy gain it explicitly retracted. |
| **B3** | ⛔ **THE NOISE WALL** — 0/6 close; gate `0.36` vs Hopfield `0.71` at σ=0.6/kv32, **despite fidelity ≈1.0** | **CM-8 states verbatim that it *"travels with every reversal claim."*** It is not a negative that can take a brief mention — **it is the scope of the paper's only external positive.** |
| **B4** | ⛔ *"**intra-CLU**… never a cost win over Hopfield"* | The base's own instruction is *"we state this next to every accuracy-improvement claim."* Detached, the savings figure reads as a Hopfield saving — **a named forbidden form.** |
| **B5** | ⛔ *"energy-**gating** it **loses** to a 449-param physics-free router"* | **CM-7.** Keep the one-hop edge and demote the loss, and a reader infers the gated edge is the win. ⚠ The base's §4.2 is *titled* with its own concession — that title is load-bearing. |
| **B6** | ⛔ **C-6's three fine prints** — Prop-12's matched-quadratic scope · *"volume alone is not the latch receipt"* · *"a free ledger does not buy BIBO"* | **C-6 requires main-text adjacency by name.** These are also the paper's inversion-proofing — the one hostile attack the base currently closes outright. |
| **B7** | **LTT exchangeability + ECE ≈ `0.100 ± 0.021`** | **C-6**, and the coverage certificate is the surviving CLU-side asset if the escalatability claim is scoped down. |
| **B8** | The **measured score sentence** (*external benchmarks won on their own headline metric = ZERO*) | **Head-ruled IN**, 2026-08-26. Added by revision pass 1. |
| **B9** | The **§A20.5 substrate-scope sentence** | **Head-ruled IN**, same session. Added by revision pass 1. |
| **B10** | The **fold**: N103's tie (**CM-23(r)** — *"ties"*, never *"wins"*) · the trilemma's third corner (**CM-23(y)** + N119's *"neither fix may be described as available"*) · N90's mechanism attribution (**CM-23(g)**) with **N95's same-section obligation** | Head-approved and folded in by revision pass 1. |

⚠ **Advisor pre-flight measurements — these are CLAIMS, and your job is to verify or refute them, not inherit them:** B8, B9 and all of B10 measured **0 hits** in `pj_sub.tex` (`external benchmark`, `headline metric`, `govern the store`, `measured separately`, `no mask oracle`, `compute-adaptive`, `dead flat`, `directed`, `1.40`, `anytime` — all zero; the nine `ties` hits are false friends inside *"properties"*, *"capacities"*, *"quantities"*). B3, B4 and the 400-ep rider appear to **survive**. ⛔ **Re-measure everything. If I am wrong, that is a finding and it outranks the rest of your report.**

⛔ **You do NOT judge whether a cut was intended.** The Head cuts deliberately and often — three such overrides are on the record for a sibling short. **Report present/absent and the consequence; the ruling is the Head's.**

---

## PART C — the drift modes specific to THIS paper

For every surviving claim, quote it beside its base form and rule **IDENTICAL / NARROWER (safe) / ⛔ WIDER / CHANGED IN KIND**. Watch these six named modes:

1. ⛔ **a designed-testbed result reading as general** — V1's entire §3/§4 architecture rests on the C-2 designed-vs-learned split being scrupulous.
2. ⛔ **an intra-CLU saving reading as a win over Hopfield.**
3. ⛔ **a 400-ep property reading as converged** (MF-C again, from the other side).
4. ⛔ **the squeeze "priced" reverting to "cannot reach"** — MF-B was falsified and retired; the base's §3.2 heading was repaired one pass ago. ⚠ Check the heading and the abstract.
5. ⛔ **decision/transport conflation** — MF-A: the `det J = 0` object is a **state-replacing map**, never "the router"; §4.2's learned router is a *decision head* that inherits the receipt.
6. ⛔ **a tie reading as a win** — if any anytime/retry material survives, **CM-23(r) binds the word to "ties."**

**Also sweep the forbidden forms** (positive-controlled, zero-hit list printed): *"beats feedforward via test-time compute"* · the anytime curve as a **uniqueness** claim · *"the anytime read wins"* · *"9–10× savings vs Hopfield"* · any **energy-as-superior-confidence/routing** claim.

---

## Deliverables

1. **Part A table** — every numeric token, its ancestor, the six-axis match. No-ancestor findings first.
2. **Part B table** — B1…B10 with PRESENT/PARTIAL/ABSENT and, for each absent item, the claim now standing unqualified, ranked.
3. **Part C table** — surviving claims ruled IDENTICAL / NARROWER / WIDER / CHANGED IN KIND, quoted both ways.
4. **The forbidden-form sweep**, positive-controlled.
5. **`md5 pj_sub.tex` at start and end — identical.**
6. **A structural note:** which of the base's sections and appendices no longer exist. The base ran §1–§5 + Appendices A–F; this file has 5 main sections + 2 appendices + references. ⛔ Report the map; do not judge it.

## Acceptance criteria

- ⛔ **`pj_sub.tex` byte-identical** — md5 printed twice, matching.
- ⛔ `submission.tex` and `papers/v1-short/**` byte-untouched (manifest printed).
- Every registry citation checked **on disk at the moment of use**, never quoted from this task file.
- Every negative positive-controlled.

## ⛔ Prohibitions

1. **No edits to any paper file.** Findings go in the report.
2. ⛔ **Do not propose page cuts or restorations.** Report; the Head rules. *(Context you may use but must not act on: TTCL is **4–9 pp excluding references and appendices** — Head-verified 2026-08-26 — and this build measures **main text pp. 1–8**, appendix p. 9, references p. 10.)*
3. ⛔ **C-8 hermetic:** do not read V2's or V5's drafts. Any precedent you need is stated here.
4. ⛔ **Treat every C3-era number as PENDING** — quotable only from `claims_matrix.md` or a filed charter addendum.

## ⚠ Grep hazards on this machine

⛔ `grep` here is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long `.tex` lines it either **errors *"exceeds complexity limits"* and exits 0** — a silent false negative that looks like success — or **hangs**. ⇒ use **`/usr/bin/grep`** for context patterns; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — **sweep per-file.** ⚠ zsh does not word-split: quote any `--include=*.tex` glob or omit it. ⚠ **Beware false friends**: the nine `ties` hits in this file are all inside longer words. **Read every hit in context.**

## DIAL DECLARATION
**Dials touched: NONE.** This pass reads, greps and writes one report. It runs no experiment, edits no paper file, and produces no new measurement.
