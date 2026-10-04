# CLOSEOUT — the maze refusal displays are RETIRED

**`w14 spoke/refusal-displays-closeout`** · agent `doc-curator` · branch
`w14/refusal-displays-closeout` · **2026-09-22** · base `base/iter8 = 6699665`.
**Window:** `START_UTC.txt` **2026-09-22T06:14:40Z** → `END_UTC.txt`. ⚠ **All times UTC** (the box's
local clock is UTC+2).
⭐ **ZERO scored runs. ZERO compute. ZERO GPU. No background job, no PID, no box-wide kill.**
⛔ `/c/chlu/evidence/` was **not written** — md5 of all four base files re-checked at close,
byte-identical to arrival. ⛔ `$CHLU_CLAUDE_ROOT` was **not opened**.
⭐ `chlu.__file__ = C:\chlu\worktrees\CHLU-w14-refusal-displays-closeout\chlu\__init__.py`, asserted in
this worktree before anything was written.
⛔ **The curator re-derived no statistic**, with **ONE declared exception**: four column means, marked
**⊙-computed** where they appear, each from three values quoted with their own paths, and each
re-checked in `repro/VERIFY.txt` §6.

---

## 0. THE RULING, AND WHY IT IS NOT AN EDITORIAL PREFERENCE

> **Head, 2026-09-21** — `outputs/advisor-wave-plan-w12-w14.md:56`, listing the **preconditions of
> `hub/read-verb`**:
> *"maze refusal displays dropped or declared uninterpretable (the refuser problem)"*

⛔⛔ **The maze refusal DISPLAY rows are RETIRED from the paper.** They may appear **only** as a named
negative — *"we built a refusal display on the maze, it is uninterpretable, and here is the
measurement that says so"* — ⛔ **never as a results-table row, never as a comparison, never as
evidence that the object refuses.**

⭐ This is a **precondition of a wave-14 hub**, not a tidy-up. Wave 14 tests whether the **composition
operator** is part of the interface (`hub/read-verb`: plan-space overlay **1.0000** against
latent-space averaging **0.13** on the same halves, `outputs/w9-advisor-report.md` §1). ⛔ The maze
refusal rows cannot ride along into that wave's tables.

⛔ **Scope of the retirement:** **both** maze environments' DISPLAY rows — the `hub/maze` CLU rows
(**CMAZE-2**, the request layer) and the `spoke/harness` refuser rows (**CMAZE-1 proper**). ⛔ Per w9
**AMENDMENT 2** the two are a different environment, not a different budget, and ⛔ **they still never
share a table.**

---

## 1. ⭐⭐ THE MEASUREMENT THAT FORCES IT — and it is NOT the display row

⛔ **The displays are not retired because the CLU refused badly. They are retired because their only
comparator is at chance on the very observation the arms are given.**

### 1.1 The comparator is at chance — six runs, `n_test = 200` each
`outputs/_thinkpad/w9/w9-harness/refusal_separability.json` (band and reading:
`outputs/_thinkpad/w9/w9-harness/LOG.md:228-236`; registry `N323`):

| run | possible vs **all** impossible | by **budget** | by **legality** | **budget scalar alone** |
|---|---|---|---|---|
| `cmaze1/seed0` | **0.4834** | 0.5254 | 0.7489 | 0.4469 |
| `cmaze1/seed1` | **0.5125** | 0.4352 | 0.8474 | 0.4589 |
| `cmaze1/seed2` | **0.5354** | 0.5645 | 0.6662 | 0.4421 |
| `cmaze1-easy/seed0` | ⛔ **0.5518** ← max | 0.6256 | 0.5666 | 0.4150 |
| `cmaze1-easy/seed1` | ⛔ **0.4822** ← min | 0.4900 | 0.5354 | 0.4449 |
| `cmaze1-easy/seed2` | **0.4862** | 0.5509 | 0.7289 | ⛔ **0.3840** ← min |

⭐ **`N323`'s band `0.48–0.55` is exactly right** — the endpoints are **0.4822** and **0.5518**.
⚠ **Stated exactly rather than rounded in our favour: 3 of the 6 point estimates are above 0.50 and 3
are below**, all at `n_test = 200`, and `N323` reads that spread as **chance**.
⭐⭐ **The budget scalar alone never reaches 0.50 on any run** (0.3840–0.4589) ⇒ ⛔ **there is no
shortcut; the impossible set cannot be picked off by a giveaway feature.**
⭐ **`n_test = 200` is printed here because the registry carried the band without its `n`.**

### 1.2 The refuser therefore does not separate
`outputs/_thinkpad/w9/w9-harness/LOG.md:222-224`; `HARNESS-READY.json` →
`notes.CAVEAT_refuser`; `param_counts.refuser` = **59,982**:

| tier | refusal on impossible | refusal on possible | **separation** |
|---|---|---|---|
| `cmaze1` | 0.8317 [0.6750, 1.0000] | 0.7850 [0.5600, 1.0000] | ⛔ **+0.047 [0.000, 0.115]** |
| `cmaze1-easy` | 0.6217 [0.3500, 0.9950] | 0.6017 [0.3350, 1.0000] | ⛔ **+0.020 [−0.005, 0.050]** |

⛔ **Neither interval excludes 0.** ⛔ **Class weighting only moves the operating point** (seed 1
refused 100 % of everything), **never the separation.**

### 1.3 ⇒ THE FINDING, AND IT IS A PROPERTY OF THE TASK
⭐⭐ **Deciding "is this maze solvable within B steps" is close to solving the maze.**
⛔ **NOT a wiring bug. NOT a defect of the model. NOT a defect of CMAZE-1 or CMAZE-2.**
⭐ **This is the reason the rows are kept as a negative rather than deleted** — it is a
benchmark-design result a later wave would otherwise pay to rediscover.

---

## 2. ⚠⚠ THE CLU-SIDE ROWS, PER SEED — and a correction the registry owed

`outputs/_thinkpad/w9/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json` → `display.refusal.*`:

| condition | truth | seed 0 | seed 1 | seed 2 | **⊙-computed mean** | printed at `w9-maze/LOG.md:548-554` |
|---|---|---|---|---|---|---|
| `budget_infeasible` | IMPOSSIBLE | 0.971429 | 0.969231 | 0.939655 | **0.960105** | 0.9601 |
| `walled` | IMPOSSIBLE | 0.960573 | 0.984634 | 0.995221 | **0.980143** | 0.9801 |
| ⚠ `mt_unseen` (never written, **REACHABLE**) | **POSSIBLE** | **0.972077** | **0.979401** | **0.982919** | **0.978132** | 0.9781 |
| ⛔ `grounded_possible` | **POSSIBLE** | **0.797945** | **0.855960** | **0.800633** | **0.818179** | 0.8182 |

⛔⛔ **`0.9601 / 0.9801 / 0.9781 / 0.8182` ARE 3-SEED MEANS.** `N-I8-19`, `CM-R12-3` and
`outputs/w9-advisor-report.md` all print them as bare values and **none of the three says so**.
⛔ **`0.8182` is NOT seed 2 — seed 2 is `0.800633`.** ⚠ **The task file commissioning this spoke writes
*"0.7979 / 0.8560 / 0.8182"* as three seeds; that is wrong.** The three seeds are
**0.7979 / 0.8560 / 0.8006**, and **0.8182 is their mean**. ⭐ All four means reproduce the printed
table to four decimals — which is how *"these are means"* is **established, not asserted**
(`repro/VERIFY.txt` §6).

⭐⭐ **THE NOVELTY CONTROL FIRES ON EVERY SEED — which the mean alone could not show.** `mt_unseen` is
**POSSIBLE** and is refused at **0.9721 / 0.9794 / 0.9829**, sitting inside the two IMPOSSIBLE
conditions' own per-seed range on all three seeds (`budget_infeasible` 0.9397–0.9714, `walled`
0.9606–0.9952) ⇒ ⛔ **the refusal is NOVELTY detection, not wall detection.**
⛔ **Against a pre-registered 0.20, the false-refusal floor is 0.7979 on its best seed.**

---

## 3. ⭐⭐ THE BAR TO REVIVE IT — three conditions, and a fourth that is a trap

`N323` recorded *"what would change it"* as two **options**. ⛔ **An option is not a bar.** The bar,
stated so a future wave can be held to it:

| # | condition | today |
|---|---|---|
| **1** | ⭐⭐ **BINDING (the task file's own criterion): a comparator that separates ABOVE CHANCE on a probe of the SAME observation the arms receive.** The `N323` ridge probe, shared observation at `t = 0`, possible vs all impossible, must clear **0.50 on every seed × tier run**, `n_test` printed, ⛔ **declared before any DISPLAY row is scored** | ⛔ **FAILS** — 0.4822–0.5518 at `n_test = 200`; **3 of 6 runs below 0.50**; `N323` reads the spread as chance |
| **2** | ⚠ **AND** the refuser's separation interval must **exclude 0** | ⛔ **FAILS** — +0.047 [0.000, 0.115] and +0.020 [−0.005, 0.050] |
| **3** | ⚠ **AND** the CLU's `mt_unseen` must separate from `walled`, with false refusal on `grounded_possible` **below the pre-registered 0.20** | ⛔ **FAILS** — 0.9721/0.9794/0.9829 vs 0.9606/0.9846/0.9952; false refusal 0.7979/0.8560/0.8006 |
| **4** | ⛔⛔ **THE TRAP:** reviving it by making impossibility **perceptually detectable** is **choosing the answer** | ⚠ w9 said so itself (*"anyone who takes it should declare it loudly"*, `w9-harness/LOG.md:236`). The legality probe already reaches **0.8474** on `cmaze1/seed1` *because extra tints change the image*. ⛔ **Any wave taking this route must label the impossible family a DESIGNED giveaway, in the row.** |

⭐ **Clearing 1 alone re-opens the comparator. Clearing 1+2+3 re-opens the display row. Nothing less.**
⚠ **The bar is a CURATOR DRAFT, not a ratified pre-registration** — ⛔ only the Head can make it
binding. Flagged in `claims_matrix.md` `§I8.9` item 5.

---

## 4. ⛔ THE ROWS ARE MARKED RETIRED. THEY ARE NEVER DELETED.

| row | status after this pass | where |
|---|---|---|
| `N323` (the measurement) | ⭐ **intact, in full, at its own number** + a dated `⊙` pointer marking it RETIRED BY `N337` | `negative_results.md` |
| `N-I8-19` (the CLU DISPLAY rows) | ⭐ **intact, in full** + a dated `⊙` pointer marking it RETIRED and supplying the per-seed correction | `negative_results.md` |
| `CM-R12-3` (the scope row) | ⭐ **intact, unedited**; `CM-W14-1` carries it to RETIRED and is read with it | `claims_matrix.md` |
| ⭐ **NEW** `N337` | the retirement, the bar, the Shapes3D non-generalisation | `negative_results.md` |
| ⭐ **NEW** `CM-W14-1`, `CM-W14-2` | matrix rows for the same | `claims_matrix.md` `§I8.8` |

⭐ **AND WHAT RETIREMENT DOES NOT TOUCH, still quotable as an INSTRUMENT property (⛔ never as a
refusal result):** `assert_unseen` reports **0** impossible instances in any non-refuser arm's data
across all seeds and both tiers; `certify_impossible` reports **0** certification failures over
**2700** impossible instances (`N323`; `outputs/w9-advisor-report.md` §2).

---

## 5. ⭐⭐ THE SHAPES3D REFUSAL VIA THE HOLE IS UNTOUCHED BY ALL OF THIS

⛔⛔ **DO NOT GENERALISE THE MAZE'S FAILURE TO THE STORE'S REFUSAL MECHANISM.** They are different
mechanisms, on different objects, on different environments.

| | the **maze** display (⛔ RETIRED) | the **Shapes3D coupling HOLE** (⭐ UNTOUCHED) |
|---|---|---|
| what refuses | a **learned** refusal score on a route request | ⭐ a **structural** hole in the request count / coupling table |
| environment | **CMAZE-2** request layer (+ CMAZE-1 proper at the harness tier) | **Shapes3D** value-holdout |
| cost | — | ⭐ **1 byte per hole**, 0 bytes outside the permitted range, **0 data rows read**, no refit |
| does it work | ⛔ **No** — novelty detection; false refusal **0.7979 / 0.8560 / 0.8006** vs a pre-registered **0.20** | ⭐ **Yes, exactly** — `red ∧ cylinder` refused at **1.0000**, **0.0000** composition cost, **101** other never-held conjunctions untouched, **zero seed spread** |
| why retired / not | ⛔ its only comparator is **at chance** on the arms' own observation | ⭐ **nothing measured in w9 or w14 bears on it at all** |

**The hole's own record, unchanged by this closeout:**
- `N-I8-6` — *"the coupling hole refuses `red ∧ cylinder` at **1.0000** for **ONE byte**, 0 bytes
  changed outside the permitted range, no refit, costing **EXACTLY 0.0000** composition"*, with a
  negative control at **5,762** bytes for a 1e-3 address-book nudge
  (`outputs/w5-advisor-report.md` §2; `outputs/w4-advisor-report.md` §4).
- `N332` (wave 12) — **two** holes, `red ∧ cube` and `red sphere`, at **1 byte each, 2 bytes total, 0
  data rows read, no refit**, over an index space **asserted** equal to `gen_field`'s
  (`layout_equals_gen_field: true`, **1,322 pair slots, 60 indices checked**), and ⭐⭐ **`red ∧ cube`
  is refusable although no red cube was ever written** — refusal is a **separate mechanism from
  forgetting** and does not need a deleted row to have existed
  (`outputs/w12-single-design-crossover/FINDINGS.md:294-300,455-462`).

⚠⚠ **THE HONEST OTHER HALF, so "untouched" is not read as "complete":** the Shapes3D display row's
**comparator is a DECLARED NOT-RUN** (`N334`, `CM-W12-9`) — `transformer__refuse_token__{value,conj}`
is present in `BASELINES-READY.json` and the wall clock ran out after the scored rows and the deletion
legs (`outputs/w12-single-design-crossover/FINDINGS.md:464-466,482`).

⭐⭐ **THE SENTENCE THAT DECIDES BOTH ROWS, and it is the one to keep:**
> ***The Shapes3D gap is a BUDGET not-run, which a wave can close by spending. The maze's is a TASK
> property, which spending does not close.***

⛔ That is why one is retired and the other is not.

---

## 6. ⭐ THE FALSIFIER THIS SPOKE WAS GIVEN — it fires, PARTLY, and the partial is the delta

> *"The registry already records it ⇒ verify and say so; do not duplicate the entry."*

| what the task file asks for | already in the base? | where |
|---|---|---|
| the **separation** numbers (+0.047 / +0.020) | ⭐ **YES** | `N323`; `CM-R12-3` |
| the **ridge probe at chance** (AUC 0.48–0.55, six runs) | ⭐ **YES** | `N323`; `CM-R12-3` |
| the **budget scalar** shortcut check (0.38–0.46) | ⭐ **YES** | `N323` |
| the **novelty control** (`mt_unseen` vs `walled`) | ⭐ **YES** | `N-I8-19` |
| the **false-refusal** rows vs the pre-registered 0.20 | ⭐ **YES** | `N-I8-19` |
| ⛔ the **Head's RETIREMENT ruling** and its authority | ⛔ **NO** | — |
| ⛔ **what a future wave must clear to REVIVE** the display | ⛔ **NO** — `N323` gives two *options*, ⛔ not a bar | — |
| ⛔ **that the Shapes3D HOLE is untouched** | ⛔ **NO** | — |
| ⛔ the **per-seed** CLU rows and the **n** behind `0.8182` | ⛔ **NO** — a 3-seed mean, printed bare | — |

⇒ ⭐ **`N323` was NOT duplicated.** One new entry (`N337`) which **rules**, pointing at `N323` for the
measurement, plus two additive in-place pointer notes.

---

## 7. ⭐ WHAT SHIPPED, and how additivity was checked mechanically

**Base:** `/c/chlu/evidence/outputs/w13-registry-fold/` — the newest pair (**5,732 / 1,511** lines).
⛔ **The live `/c/chlu/evidence/` pair is STALE** (**3,794 / 643** lines) and basing on it would have
dropped the `w12r` repair, the 21 `N-I8-*` entries and the whole wave-12 fold.

| file | what it is | lines |
|---|---|---|
| `CLOSEOUT.md` | this document | — |
| `negative_results.md` | base **+ the `⊙` WAVE-14 CLOSEOUT block**: entry **`N337`**, `⊙.0`–`⊙.3`, and **two** additive in-place pointer notes (`N323`, `N-I8-19`) | 5,732 → **5,973** |
| `claims_matrix.md` | base **+ `§I8.8`** (`CM-W14-1`, `CM-W14-2`) **+ `§I8.9`** (6 open flags), title bumped to **v2.21 (PROPOSED)** | 1,511 → **1,624** |
| `FINDINGS.md` | the report to the Hub | — |
| `LOG.md`, `START_UTC.txt`, `END_UTC.txt` | the working record and the clock | — |
| `repro/` | `build.py`, `verify.py`, `VERIFY.txt`, and the four source fragments | — |

### ⭐⭐ ADDITIVITY, MECHANICALLY VERIFIED (`repro/VERIFY.txt` §1–§4)
- `negative_results.md`: `difflib.SequenceMatcher` base vs new — **5,733 lines `equal`, 241 `insert`,
  ⭐ ZERO `replace` and ZERO `delete`.** **Not one existing line was edited or removed.**
- `claims_matrix.md`: **lines 2–1,512 byte-identical** (list comparison **True**); **line 1 is the only
  changed line**, and the **entire prior title is preserved verbatim as its tail**
  (`str.endswith` **True**, 17,371 → 19,153 chars); **113 lines appended.**
- **Numbering:** base **349** `N` entry headings → new **350**; **added `['N337']`**, **removed `[]`**,
  **zero duplicate headings**. Base **31** `CM` rows → **33**; added `['CM-W14-1','CM-W14-2']`, removed
  `[]`, zero duplicates. ⛔ **Nothing renumbered, so no redirect was owed.**
- **The retired rows are still present**, each exactly once: `N323`, `N-I8-19`, `CM-R12-3`, `N-I8-6`,
  `N332`, `N334`.
- **Evidence md5 at close** — byte-identical to arrival:
  `/c/chlu/evidence/negative_results.md` **c3597411aa834304be27110dab115673** ·
  `/c/chlu/evidence/claims_matrix.md` **39be4b5f1d8f29c5058c5a0bef935243** ·
  `outputs/w13-registry-fold/negative_results.md` **835fad9331b77f8fe6f6b5136e9d4ef5** ·
  `outputs/w13-registry-fold/claims_matrix.md` **8d8faa2fcbc92c79d97324acde744778**.

---

## 8. ⛔ DECLARED NOT-RUN

| not run | reason |
|---|---|
| ⛔ **any scored run, any compute, any GPU** | ⭐ the task file forbids it — *"ZERO compute, ZERO scored runs"*. Nothing ran but file reads, four means, and one `import chlu` path assertion. |
| ⛔ **re-running the w9 refuser at a larger budget** | that is the **revival bar** (§3 item 1) and belongs to whichever wave takes the target. ⛔ `maze` is **0/3, exhausted in w9**. |
| ⛔ **the ridge probe at a larger `n_test`** | a new measurement. The band is quoted at the `n_test = 200` it was measured at, and the `n` is now printed. |
| ⛔ **scoring the Shapes3D refuse-token comparator** | ⛔ **not this spoke's** — `N334`'s declared not-run, held for after submission by Head ruling. |
| ⛔ **fixing the CMAZE-2 request layer (rule B under-exercised)** | ⛔ **not this spoke's** — `tasks/w14-spoke-request-layer-fix.md` owns it. ⭐ That file was seen as a **name in a directory listing only** and never opened; that spoke's worktree, output dir and processes were never touched or signalled. |
| ⛔ **amending `outputs/w9-advisor-report.md`** for the 3-seed-mean defect | ⛔ **read-only evidence, and not this spoke's file.** Raised as `§I8.9` item 2 for the Hub. |
| ⛔ **editing `CM-R12-3`, the task file, or any existing line** | additive-only discipline. Raised as `§I8.9` items 1 and 3. |
| ⛔ **writing `/c/chlu/evidence/`** | read-only by standing rule; md5 re-checked at close. |
| ⛔ **confirming a version or promoting a tier** | ⭐ the curator never self-confirms. `N337` is **CURATOR-PROPOSED tier A**; `v2.21` is **PROPOSED**; `v2.20` and below stay PROPOSED for their own Hubs. |

---

## 9. ⚠ OPEN FLAGS FOR THE HUB (⛔ raised, deliberately not resolved here)

1. ⚠ **`CM-R12-3` keeps its own *"not yet informative"* wording.** A reader loading only that row does
   not see *"retired"*. ⛔ Not edited. ⭐ **Hub's call:** pointer-only, or a later fold adds a one-line
   in-place note there too.
2. ⛔⛔ **A number was quoted without its `n` in three places at once** — `N-I8-19`, `CM-R12-3` and
   `outputs/w9-advisor-report.md` each print `0.8182` (and three companions) as a bare value when it
   is a **3-seed mean**. ⭐ The registry and matrix **copies** are corrected; ⛔ **the w9 advisor report
   is not this spoke's file and was not edited.**
3. ⚠ **The commissioning task file carries the same defect.** Recorded, not inherited; ⛔ not edited.
4. ⚠ **`N337` tier CURATOR-PROPOSED A; `v2.21` PROPOSED.** The curator never self-confirms.
5. ⚠ **The revival bar (§3) is a curator draft of a bar**, not a ratified pre-registration. ⛔ Only the
   Head can make it binding.
6. ⚠ **`N-I8-21`(b)** — instruments producing verdicts about themselves — touches this pass: the w9
   refuser is exactly that, ⭐ **but it was caught by its own spoke against its own interest**, which is
   the healthy form. ⛔ Not filed as a seventh instance; the promotion question stays the Head's.

---

## 10. ⛔ WHAT WOULD PROVE THIS SPOKE WRONG — and the answer

> *"The registry already records it ⇒ verify and say so; do not duplicate the entry."*

⭐ **Verified: it records the MEASUREMENT (`N323`, `N-I8-19`) and a scope row (`CM-R12-3`), and this
pass duplicated none of them.** ⛔ **It does NOT record the ruling, the revival bar, the Shapes3D
non-generalisation, or the `n` behind `0.8182` — and those four are the entirety of the delta.**
