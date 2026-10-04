# w13 spoke/registry-fold — FINDINGS (report to the Hub)

**Agent:** `doc-curator`. **Branch:** `w13/registry-fold`. **Base:** `base/iter8 = aa91785`.
**Window:** `START_UTC.txt` **2026-09-21T17:03:11Z** → `END_UTC.txt`.
**Scored runs: ⛔ NONE. Compute: ⛔ NONE. GPU: ⛔ NONE. Background jobs / PIDs launched: ⛔ NONE.
Box-wide kill: ⛔ never issued, never possible.** ⭐ No other agent's session, worktree, output
directory or process was touched, listed or signalled.
⛔ **`$CHLU_CLAUDE_ROOT` was NOT opened at any point.**
⭐ **`chlu.__file__ = C:\chlu\worktrees\CHLU-w13-registry-fold\chlu\__init__.py`** — asserted in this
worktree before anything was written.
⛔ **`/c/chlu/evidence/` was NOT written.** md5 re-checked at the close of the session, byte-identical
to arrival **and identical to the values `w12r spoke/registry-repair` recorded**:
`negative_results.md` **c3597411aa834304be27110dab115673** ·
`claims_matrix.md` **39be4b5f1d8f29c5058c5a0bef935243** ·
`outputs/w12r-registry-repair/negative_results.md` **f6b675d73ed2897742924f83436d35b7** ·
`outputs/w12r-registry-repair/claims_matrix.md` **fdd7b3a3865bae6d17c256d4ef29b0fe**.
⛔ **The curator re-derived no statistic.** Every number below is quoted from a wave-12 artefact with
the path that holds it. **The only arithmetic this pass performed is four subtractions of two quoted
`normalised_mean` values, in `N326`, marked in place.**

> ⚠ **ONE DISCREPANCY IN THE TASK FILE, RESOLVED AND DECLARED.** The standing-rules block carries a
> templating glitch — *"write ONLY to `/c/chlu/outputs/w13-w13-spoke-registry-fold/`"*, branch
> *`w13/w13-spoke-registry-fold`*. The job section and the launch both say
> **`/c/chlu/outputs/w13-registry-fold/`** and branch **`w13/registry-fold`**. ⭐ **Those were used.**

---

## 0. WHAT SHIPPED

| file | what it is | size |
|---|---|---|
| `negative_results.md` | the registry **+ the `⊛` WAVE-12 FOLD**: entries **`N326`–`N336`**, a file-header note, three additive in-place pointer notes, the `⊛.1` w8 verification table, the `⊛.3` fold-in ledger, a 17-row DECLARED NOT-RUN table, and **`⊛.4` — the `N-I8-21`(b) answer** | 5268 → **5732 lines** |
| `claims_matrix.md` | the matrix **+ a new `§I8.6` / `§I8.7`**, title bumped to **v2.20 (PROPOSED)**: rows **`CM-W12-1` … `CM-W12-10`** | 1235 → **1511 lines** |
| `LOG.md`, `FINDINGS.md`, `START_UTC.txt`, `END_UTC.txt` | the working record and the clock | — |

### ⭐ BASE, as the task file directs

`/c/chlu/evidence/outputs/w12r-registry-repair/` — the current repaired pair (**5268 / 1235** lines),
carrying `N316`–`N325` and `§I8.4`/`§I8.5`. ⛔ Basing on the live `/c/chlu/evidence/` pair would have
dropped the w12r repair *and* the 21 `N-I8-*` entries.

### ⭐⭐ ADDITIVITY, MECHANICALLY VERIFIED, NOT ASSERTED

- `negative_results.md`: `difflib.SequenceMatcher` over base vs. new — **5,269 lines `equal`, 464
  `insert`, and ⭐ ZERO `replace` and ZERO `delete` blocks.** **Not one existing line was edited or
  removed.** The three pointer notes and the header note **add** lines and edit none.
- `claims_matrix.md`: **lines 2–1236 byte-identical** to the base copy (list comparison **True**).
  **Line 1 is the only changed line**, and the **entire prior title is preserved verbatim as its tail**
  (`str.endswith` **True**, 16,270 → 17,371 chars).
- ⛔ **Nothing was renumbered**, so **no redirect line was owed.**

---

## 1. ⭐ THE W8 VERIFICATION — the falsifier this spoke was given. ⛔ IT DOES NOT FIRE.

> *"A w8 item you were told is already folded turns out to be missing or wrong ⇒ that is the finding;
> fix it and report the gap in the earlier fold."*

⭐ **All five are present and correct in the base copies. No gap. Nothing re-folded, nothing repaired.**

| the plan's w8 item | where it lives | what was checked |
|---|---|---|
| encoder-loop **WASH**, decomposition is the finding | `N-I8-4` + `N321` + `N322` | body read: A′ **5/6 negative** at p ≤ .0013 · A″ **4/6 positive** at p ≤ .0003 · GPU swap loss **+0.0100 / +0.0129 / +0.0075** · the "separate table, never merged rows" rule present |
| `K ≥ 36` collapse | `N-I8-5` + the in-place `⟲ w12r` qualifier on link 3 + `N325` | body read: K=40/A=7, **519,794 B = 0.991×**, valid mean **0.0648** vs **0.8918** at K=5; w12r's "1 of 5 pin-fitting configs, dtype-conditional" qualifier present |
| TensorFaces / MLC-VAE **STOP** | `N-I8-18` | body read: Tenenbaum & Freeman 2000, Vasilescu & Terzopoulos 2002, Vlasic 2005, MLC-VAE arXiv:2009.04075; **and** the crawl's narrowing of STOP #2 |
| MiniGrid **composition-blind** | `N-I8-16` | body read: `MiniGrid-DoorKey-*` rejected at **1.0000** blind by construction, `ENV.md:117` |
| the Advisor's Mexican-hat error | `N-I8-14` | body read: *"interpolation"* **0** occurrences, *"Mexican hat"* **1**, the nearest real object **INVERTS** it |

⭐⭐ **AND A MECHANICAL CHECK THAT ALSO RE-VALIDATES w12r's REPAIR:** the registry now holds **349
entry headings** and ⭐ **ZERO duplicate `N` numbers anywhere in the file** — the `N89`/`N90` collision
`N-I8-21`(a) registered is genuinely gone, not merely annotated.

---

## 2. ⭐⭐ THE WAVE-12 FOLD — `N326`–`N336`, every entry with a number and a path

**Acceptance criterion, checked mechanically: each of the eleven entries is exactly one `###` heading
and carries between 2 and 8 distinct artefact path references; `CM-W12-1` … `CM-W12-10` are each
exactly one heading.**

| task item | entry | matrix row | the number that decides it |
|---|---|---|---|
| 1 | **`N326`** | `CM-W12-1` | store − `learned_1x`, of ceiling: **−0.077 / +0.059 / −0.073 / −0.054 / −0.056** at n = 2…6 |
| 2 | **`N327`** | `CM-W12-2` | **0.848302 → 0.898881**; and for n ≥ 3, r = **−0.9841** against held-out rows |
| 3 | **`N328`** | `CM-W12-3` | identical at every rung; **23.886×–24.875×** vs **1.358×–1.372×** |
| 4 | **`N329`** | `CM-W12-4` | **+7.08…+9.65 → +14.99…+22.43** against **−4.57…−1.57 / −5.78…−1.84** |
| 5 | **`N330`** | `CM-W12-5` | rank/width **0.9844**; four controls **exactly 0.000**, **109/109** corners |
| 6 | **`N331`** | `CM-W12-6` | **7.4815 → 2.0663** (parts 1+2+4) / **7.854 → 4.088** (honest four-part) |
| 7 | **`N332`** | `CM-W12-8` | red ∧ cube **0 / 0 rows**; red ∧ sphere **5,507 / 5,511**; sphere excess **−0.3310** |
| 8 | **`N333`** | `CM-W12-7` | `LADDER.json:K_of_the_store_arms` = **5** vs `rungs` keyed **"2"…"6"** |
| 9 | **`N334`** | `CM-W12-9` | NegGrad+ **9.4×–20.8×** over; fine-tune **0.20–1.00×** under; **296,960** matched row-visits |
| ⊛a | **`N335`** | `CM-W12-10` | **32.25 → 32.25** on 20 deleted corners; uncarried **0.610 vs 0.614** |
| ⊛b | **`N336`** | — | **0 of 164** certifiably impossible, all three seeds |

### The five things in this fold the Hub should read before anything else

**(i) ⛔⛔ `N331` CORRECTS A NUMBER THE PROGRAMME IS ABOUT TO QUOTE, and this is the most likely
wave-12 figure to reach a draft in the wrong form.** The task file's *"the honest total FALLS 7.48× →
2.07×"* is the **`times_reference_pin_parts_1_2_4`** column of
`outputs/w12-single-design-crossover/LEDGER-ACROSS-K.json` — it **EXCLUDES the address book (part 3)**,
and ⚠ **that file's own first key is `"SMOKE_NOT_A_CLAIM": true`.** The **honest four-part total** in
the same file falls **7.854× → 4.088×** and is **U-SHAPED**, bottoming at **3.994× (2,094,130 B) at
K = 24**. ⇒ ⭐ **the direction survives on both columns — finer partitions are cheaper — but the
mechanism sentence *"the covariance shrinks faster than the address book grows"* is TRUE from K = 5 to
K = 24 and FALSE from K = 24 to K = 32**, where the address book grows **827,904 → 1,059,824 B**
against a covariance falling **735,328 → 560,608 B**. **Quote the direction, not the mechanism
unconditionally.** ⛔ A fourth fact must travel with it: **the object never reaches the 524,288 B pin
at rank 64 under ANY swept K** — minimum four-part total **3.994×**.

**(ii) ⭐ `N326`'s n = 3 row is recorded as UNEXPLAINED, and the entry says why in the arm's own
numbers.** At n = 3 the store sits at **its own lowest rung (0.829861)** while the learner drops to
**0.770949** from a tight **0.9256 / 0.9443 / 0.9530 / 0.9530** elsewhere. **The gap opens because the
opponent fell off, not because the arm improved**, and ⛔ **no wave-12 artefact records a mechanism for
that dip — the registry does not invent one.** It also **fails the pre-declared bar anyway**: **2 of 3
seeds** (+4.8167 / **−0.0333** / +7.9417) against `AMENDMENT-2` §3's every-seed rule, frozen at
**11:46:30Z before any arm number existed**.

**(iii) ⛔⛔ TWO ENTRIES THAT MUST NEVER BE PRINTED APART — `N326`/`CM-W12-1` and `N329`/`CM-W12-4`.**
The same object on the same rungs **wins at every rung** against the 524,288 B rivals (**+7.08…+9.65**
at n=2 **rising** to **+14.99…+22.43** at n=6) and **loses at both ends** against a learner matched to
its own 24.4× total (**all six CI95 upper bounds below zero**). ⭐ **The pin does not shift the level —
it flips the sign of the n-trend's interpretation.** The ladder hub itself **recommends the 1× column
lead** *"because the 524,288 B column is measured to flatter the arm"*; ⛔ **which column leads is a Hub
decision and this pass does not make it.**

**(iv) ⭐⭐ CURATOR-ADDED BEYOND THE TASK'S NINE, AND DECLARED AS SUCH — `N335` / `CM-W12-10`.** At
K\* = 32, **deleting all 11,044 red rows leaves the 20 red-requesting corners at 32.25 → 32.25** (judge
accuracy **0.8958 → 0.8958**), and the **uncarried** basis change is **0.610 on red-requesting corners
against 0.614 on all others** ⇒ ⛔ **the basis change is GENERIC, not targeted, and the per-well content
carries no value-specific information there.** At **K = 5, rank/width 0.156** the same deletion **was**
concentrated (`cos_min = 0.1999`, 93.8 % of directions above 0.99, vs **0.9976** for a size-matched
random deletion). ⇒ ⭐⭐ **the targeting of a deletion is a function of rank/width and disappears exactly
where the byte ledger is cheapest — the compose-and-forget trade-off did not vanish, it MOVED from
bytes to targeting.** ⛔ **"The store has forgotten red" and "the object no longer answers red" are
different statements, and at K\* only the first is demonstrated.** ⭐ **Why it is in the fold although
the task's nine do not name it:** its own hub calls it the wave's most consequential result and it
**bounds the programme's lead claim** — a negatives registry that folds a wave and omits its sharpest
negative is not a negatives registry. ⚠ **Struck at the Hub's word; nothing else depends on it.**
(`N336`, the second addition, is the item-deletion NOT-RUN that is itself a finding: **0 of 164**.)

**(v) ⚠ `N334` CORRECTS THE TASK FILE'S OWN PHRASING, in the direction of MORE disclosure.**
*"Built but deliberately NOT run"* under-reports what is on disk: `unlearn_finetune` and
`unlearn_neggrad` **were measured on the fit pool** — ⛔ **0 scored runs, no held-out test corner** —
and the numbers are real (**NegGrad+ over-forgets 9.4×–20.8×; fine-tune under-forgets on three arms of
four; at matched gradient row-visits 296,960, asserted equal across all eight arm × method combinations
on all three seeds**, so ⛔ **not a starved-comparator artefact**). What is genuinely not run is
**scoring them in the deletion legs** and the **refuse-token DISPLAY comparator** — wall clock, both
declared. **Sharded ensembles (SISA) and the voting CLU ensemble are UNBUILT and stay so**
(`outputs/advisor-wave-plan-w12-w14.md:32`). ⭐ **The standing form is recorded as asked: availability
is not authorisation.**

---

## 3. ⛔⛔ `N-I8-21`(b) — THE ANSWER THE TASK FILE ASKED FOR. REPORTED, NOT PROMOTED.

⚠ **First, a disambiguation the Hub needs before it rules, because the task file's parenthesis names
the wrong limb.** `N-I8-21` has two:
- **(a) the duplicate-number allocation** (`N89`/`N90`). ⭐ **CLOSED** — executed by w12r, redirect at
  five sites plus `CM-R12-6`; the registry's own annotation says *"(a) is now FIXED … **(b) stays OPEN
  and is not this pass's**"*. ⭐ **Re-verified this pass: 349 entry headings, ZERO duplicate numbers.**
  ⛔ **It is not on a third instance.**
- **(b) *instruments that manufactured their own verdicts*.** ⭐ **That is the escalating limb**, and
  the one w12r flagged and declined to promote (`w12r-registry-repair/FINDINGS.md` §5 item 6).

⭐⭐ **RECOMMENDATION: YES — PROMOTE (b) TO A STANDING RULE. ⛔ NOT PROMOTED HERE.** The grounds are a
count, and **wave 12 raised it from three to six, across five waves and at least four codebases:**

| # | instance | wave | path |
|---|---|---|---|
| 1 | BAR D void — the instrument could not produce the verdict it was built for | w8 | `N321` |
| 2 | the `alone` control measured its own variance | w8, replicated w9 | `N321` |
| 3 | the equal-seconds control under-funded itself **6–12 %** in the treatment's favour | w8 | `N322` |
| 4 | ⭐ **the ladder hub's aggregator compared an arm WITH ITSELF and scored the tie as a failure**, so `learned_1x` read *"NEVER CLEARED THE BAR"* | **w12** | `outputs/w12-composition-ladder/FINDINGS.md` §7.2 |
| 5 | a **tautological field** reading as a perfect pass — `d14_confinement` exactly `0.0` on 60 cells × 3 seeds | w6, filed w12r | `N320` |
| 6 | ⭐⭐ **a file flagged `"SMOKE_NOT_A_CLAIM": true` supplying the wave's headline byte ratio**, from a column that excludes part 3 of its own ledger | **w12** | `N331` |

⭐ **Instances 4, 5 and 6 are three different faces of one disease** — a self-comparison, a tautology,
and a smoke artefact promoted to a claim — and **4 and 6 were caught by their own authors**, which is
the programme working.
⭐ **The rule this pass would propose if asked to draft one (⛔ drafted, NOT adopted):** ***before a
number leaves an instrument, state what the instrument would print if the effect were absent. If that
answer is the same number, the instrument is not measuring the effect.*** It covers a self-comparison
(prints a tie), a tautology (prints 0.0), an under-funded control (prints the treatment's advantage)
and a smoke artefact (prints whatever it was seeded with) **in one sentence.**
⛔ **Three decisions that are the Hub's and not the curator's:** (i) standing or wave-scoped;
(ii) filed in the registry, the matrix `§0`, or the charter; (iii) whether `N-I8-21` is **split**, since
⚠ **one entry number currently carries one CLOSED limb and one ESCALATING limb — itself a small
registry hazard.**

---

## 4. ⛔ DECLARED NOT-RUNs — every one with its reason. **Nothing was silently dropped.**

**This pass, as a `doc-curator` spoke with ZERO scored runs by its task file:**

| not run | reason |
|---|---|
| **Any scored run, compute, GPU, background job or PID** | ⛔ the task file specifies **ZERO scored runs** |
| **Re-deriving any wave-12 statistic** | ⛔ the curator re-derives nothing. **Only the four subtractions in `N326`, marked in place** |
| **Writing `/c/chlu/evidence/`** | ⛔ **READ-ONLY** to this spoke; the Hub folds the copies back |
| **Re-folding the five w8 negatives** | ⛔ forbidden by the task file. ⭐ **Verified instead** — §1 |
| **Promoting `N-I8-21`(b)** | ⛔ **the Hub's call.** ⭐ Reported with its count and a draft rule — §3 |
| **Allocating main-series aliases for the w5/w6 findings** | ⛔ w12r's declared refusal on `N315`'s authority stands; this pass did not re-open it |
| **Editing `tests/test_gen_value_holdout.py`** (`N324`) | ⛔ still **UNOWNED**; ⚠ wave 12 hit the same three red tests independently and also did not touch them |

**Carried forward from wave 12 itself, restated so they are not lost** (full 17-row table in
`negative_results.md` → the `⊛` block): **seed 3** (cache gate **passes** — `array_shape
[480000,128,4,4]`, 59 blocks — but **the judge does not exist**: `fw.classifier_dir(3)` **False**,
`decoder_v1_s3.pkl` absent, `Judge(3)` raises) · **seed 4** (fails the cache rule: `array_shape: null`,
**0** blocks) · **the TEST number at K = 5** (only K\* = 32 was scored on TEST ⇒ the val/test gap is
untestable; **`single-design` has 2 of 3 runs left**) · **a 1×-matched LSTM / transformer / continuous
MLP / table** (wall clock) · **the 1× baseline column on the scored row** (`param_target = 131,072`;
refitting at **535,784** params unaffordable ⇒ the **524,288 B reference column**, labelled) · **the
refuse-token DISPLAY comparator** (wall clock) · **both unlearners in the deletion legs** (wall clock;
⭐ and held as the rebuttal kit regardless) · **SISA and the voting CLU ensemble** (⛔ unbuilt by
ruling) · **`generation-table` run 3** (⭐ **unspent**) · **the coverage-matched ladder control** ·
**dSprites** (⛔ no number produced for any other dataset and none implied).

---

## 5. ⚠ OPEN FLAGS FOR THE HUB

1. ⚠ **Tiers are CURATOR-PROPOSED** (`N326` A · `N327` A · `N328` A · `N329` A · `N330` A · `N331` A ·
   `N332` A · `N333` B · `N334` A · `N335` A · `N336` B), as are all ten `CM-W12-n` rows.
   ⛔ **The curator never self-confirms.**
2. ⚠ **`v2.20` is PROPOSED.** ⛔ This pass does **not** confirm `v2.19`, v2.18, v2.17 or below —
   **each is its own Hub's.**
3. ⚠ **`CM-W12-6` supersedes the FRAMING of `CM-R12-4`** while keeping every one of its qualifiers.
   ⛔ `CM-R12-4`'s text is **left in place, not edited**. **Whether the rows merge is the Hub's call.**
4. ⚠ **`N335` and `N336` are CURATOR-ADDED beyond the task's nine**, flagged in their own first lines.
5. ⚠ **Still open and untouched:** `§I8.3` 3–6 and `§I8.5` 4–6 — the `N-I8-1` exactness-band
   attribution, the "9/9 seeds" phrasing, the unspent budgets, the pre-ratification spot-checks,
   `N318`/`N319`'s artefact trails, `N324`'s red tests, the alias-vs-crosswalk question.
   ⛔ **Restated as still open, not dropped.**
6. ⚠ **A Hub decision this fold surfaces but does not take:** the ladder hub asks whether the
   composition rows should quote **`pca_tensor`** (1.37×) and let the single design carry the deletion
   rows alone, given `N328`. ⭐ Recorded in `CM-W12-3`; **not decided here.**

---

## 6. ⛔ WHAT WOULD HAVE PROVED THIS SPOKE WRONG — checked against the result

> *"A w8 item you were told is already folded turns out to be missing or wrong ⇒ that is the finding;
> fix it and report the gap in the earlier fold."*

⭐ **It does not fire.** All five w8 items were located, their bodies read, and their deciding numbers
checked against the entries — §1. **No gap in the w10 fold or the w12r repair was found**, and the
duplicate-number check that `N-I8-21`(a) exists for now returns **ZERO duplicates over 349 entry
headings**. ⛔ **Nothing was re-folded and nothing was repaired**, exactly as the task file requires —
the effort went to **wave 12, which was unrecorded**, and which is now `N326`–`N336` and
`CM-W12-1`…`CM-W12-10`.

**The acceptance criterion, checked mechanically:**
- ⭐ **Every entry carries a number and a path** — `N326`–`N336` each appear as exactly one `###`
  heading and each carries **2–8 distinct artefact path references**; each names its wave-12 source
  file in a `**Source:**` line.
- ⭐ **Renumbered entries carry a redirect line** — ⛔ **nothing was renumbered by this pass**, so none
  was owed. **Three additive in-place pointer notes** were written where a wave-12 entry changes how an
  older one must be read: `N-I8-1` → `N335`, `N-I8-5` → `N331`, `N325` → `N331`.
- ⭐ **`N-I8-21`(b) reported, not promoted** — §3, with the limb disambiguation, six instances, a draft
  rule and the three decisions left to the Hub.

## 7. COST

⛔ **Zero scored runs. Zero compute. Zero GPU. No cluster leg. No background job. No PID launched.
No box-wide kill possible or issued.** One session, reading and writing documents. Deliverables are in
`/c/chlu/outputs/w13-registry-fold/`, committed on branch `w13/registry-fold` under
`w13_registry_fold/`, and bundled to `w13-registry-fold.bundle` — **that branch only**, verified,
`list-heads` showing exactly one ref.
