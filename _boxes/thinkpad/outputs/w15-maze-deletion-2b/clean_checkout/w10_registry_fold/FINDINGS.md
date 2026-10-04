# w10 spoke/registry-fold — FINDINGS (report to the Hub)

**Agent:** `doc-curator`. **Branch:** `w10/registry-fold`. **Scored runs: ⛔ NONE** (task-declared).
**Window:** `START_UTC.txt` 2026-09-21T08:26:07Z → `END_UTC.txt`. Local, no compute, no GPU, no background jobs.
⛔ **The curator re-derived no statistic.** Every figure below is quoted from an agent's own artefact with
the path that holds it. ⛔ **`$CHLU_CLAUDE_ROOT` was never opened.**

---

## 0. WHAT SHIPPED

| file | what it is |
|---|---|
| `claims_matrix.md` | the live matrix **+ a new top-level ICLR-LONG DIVISION**, title bumped to **v2.18 (PROPOSED)**. 643 → **1039 lines** |
| `negative_results.md` | the live registry **+ a dated `⟲⟲ 2026-09-21` ITER8 FOLD**: **21 entries `N-I8-1…21`**, a **17-row DECLARED NOT-RUN table**, and 6 provenance flags. 3794 → **4739 lines** |
| `LOG.md` | the working record |
| `START_UTC.txt` / `END_UTC.txt` | the clock |

⛔ **`/c/chlu/evidence/` was NOT written.** Verified by md5 at the end of the session: `claims_matrix.md`
**39be4b5f1d8f29c5058c5a0bef935243**, `negative_results.md` **c3597411aa834304be27110dab115673** — both
unchanged from the values taken on arrival. **The Hub folds these copies back.**

⭐ **ADDITIVE-ONLY, MECHANICALLY VERIFIED, NOT ASSERTED:**
- `claims_matrix.md` lines **2–643 are byte-identical** to the evidence copy (`diff` empty).
  Line 1 is the only edited line, and the **entire prior title line is preserved verbatim as its tail**
  (checked with `str.endswith`, 12,031 → 14,099 chars).
- `negative_results.md` lines **1–3794 are byte-identical** to the evidence copy (`diff` empty); every new
  byte is appended from line 3795.

⛔ **No version but this pass's was confirmed.** v2.17 / v2.16 / v2.15 stay PROPOSED for their own Hubs.

---

## 1. THE NUMBERING DECISION, AND WHY IT IS NOT `N316`

On arrival the registry carried **`N89` and `N90` allocated TWICE** — the main series, and the
`⟲ 2026-09-11 — GEN PIVOT, PHASE 1` addendum (`negative_results.md:3787+`) which re-used both numbers
immediately below `N87`/`N88` (`:3780-3781`). That is the exact hazard **`N315` (`:3376`)** self-filed at
the C2/C3 boundary.

⇒ This fold uses a **separate, greppable namespace**: registry **`N-I8-n`**, matrix **`CM-I8-n`**.
⛔ **It allocates no number in either existing series, and it does not renumber anything.** The duplicate is
**FLAGGED as `N-I8-21`(a) and left for the Hub.**

---

## 2. THE SEVEN COMMISSIONED ITEMS — every one landed, with its deciding number and path

| task item | where it landed | the number that decided it | path |
|---|---|---|---|
| **1** exact deletion by downdate | `N-I8-1`, `CM-I8-1` | statistic **7.522e-16 / 8.826e-16 / 8.2e-16** on CMAZE, **subspace projector 0.000e+00**, `refit False`, 0 passes, items never kept; **1e-16…1e-14** on Shapes3D (`single-design`); **2.842e-14…1.126e-09** (`downdate-build`, independent code) | `outputs/_thinkpad/w9/w9-maze/LOG.md:181,208,220,531-534`; `w8/w8-single-design/LOG.md:472,481,590`; `w8/w8-downdate/LOG.md:1010-1024` |
| **2** no transfer to a route task | `N-I8-2`, `CM-I8-2` | BAR 1 **−0.0560 / −0.0209 / −0.0434**; lookup **0.1565 vs 0.1279** (r2), **0.1122 vs 0.0996** (r4); `landscape_shuffled` **0.9509**, `random_atoms` **0.9601**; vs transformer **0 corners ahead, 8 behind, mean −0.0304** | `outputs/_thinkpad/w9/w9-maze/LOG.md:420-457` |
| **2b** the isolated mechanism | `N-I8-3`, `CM-I8-3` | **PLAN space 1.0000 / 1.0000 / 1.0000** vs **LATENT space 0.1649 / 0.0950 / 0.1449**; plan coverage **1.000**; factorisation coverage **0.997**, complete for **99.1 %** | `w9-maze/LOG.md:288-346,489-527` |
| **3** store ≠ representation | `N-I8-4`, `CM-I8-4` | BAR B **fired on row 3**. A′ **5/6 negative** (p ≤ .0013); A″ **4/6 positive** (p ≤ .0003); net straddles zero | `outputs/_thinkpad/w8/w8-encoder-loop/LOG.md:798-846` |
| **4** compose/forget at matched bytes | `N-I8-5`, `CM-I8-5` | **7.48×** (3,922,450 B) at K=5 → **0.8918** vs incumbent **0.8941**; **0.991×** (519,794 B) at K=40/A=7 → **0.0648**; covariance forces **K ≥ 36** | `outputs/_thinkpad/w8/w8-single-design/LOG.md:580-620` |
| **5** the four closures + the absence | `N-I8-6`…`N-I8-10`, `N-I8-14`; `CM-I8-6`, `CM-I8-7`, `CM-I8-9` | refusal **(1.0000, 1.0000)** tie at **59,392×** bytes · lifetimes **+0.0562/+0.0337/+0.0112** vs margin 1.0, `t_pay(50 %)` **7.992/5.994/7.992** · placement **mean −3.068**, spread **14.918** · row 2 **−0.0192±0.0028** and **−0.0170±0.0024**, 3/3 seeds · Mexican-hat: *"interpolation"* **0 occurrences**, *"Mexican hat"* **1** (a round-off testbed) | `outputs/w5-advisor-report.md` §3 · `w6-…` §4 · `w4-…` §3 · `w4-…` §2 · `w6-…` §2 |
| **6** gen hygiene + the gym laws | `N-I8-15`, `N-I8-16`, `N-I8-17`; `CM-I8-10`, `CM-I8-11`, `CM-I8-12` | gen-1 **51 / 81 / 59 ULP** against a **1-ULP** bound · GYM-1 lookup **0.9797 vs 0.3700 / 0.4173** · GYM-2 **TW-B 0.9825 vs prereg 0.60** · CMAZE-1 blind **0.4980**, frag **0.0000**, rule-B bend **81.6 %** · **1242 of 1449 routes use one key** | `outputs/w4-advisor-report.md` §5(a) · `w5-…` §5 · `w8/w8-env-scout/ENV.md:18-68` · `w9-maze/LOG.md:413-419` |
| **7** the novelty STOP | `N-I8-18`, `CM-I8-2`(a) of §I8.1 | **0 of 3,038** citing works combine frozen-decoder + tensor-head + label-index; **highest atom count 4 of 6**; ≈**4,050** works screened; six orthogonal sweeps return **zero** | `outputs/_thinkpad/w8/w8-novelty/NOVELTY.md:14-38,238-258` |

⭐ **Both files were read, as instructed.** `NOVELTY.md` **and** its `LOG.md`. The LOG's carry-forward
correction (`w8-novelty/LOG.md:206-210`) is what stops this registry being more pessimistic than the
evidence supports — see §4 below.

---

## 3. ⭐ THE ONE PLACE THE EVIDENCE DOES NOT SUPPORT THE COMMISSIONING SENTENCE

⛔ **This is the "a row you cannot source ⇒ say so in the row" case, and it is the most important thing in
this report.**

The task file states item 1 as *"1e-16…1e-14 on Shapes3D (**9/9 seeds**, two independent
implementations)"*. **The property is real. That phrasing is not.**

1. ⛔ **"9/9 seeds" is 3 CONFIGURATIONS × 3 seeds inside ONE implementation** — `hub/single-design`'s
   matched-bytes candidates K=40 / K=44 / K=48 — **not nine seeds** (`w8-single-design/LOG.md:481,503`).
2. ⛔ **The band is not shared by both implementations.** `spoke/downdate-build`'s own four legs run
   **2.842e-14 · 4.547e-13 · 1.164e-10 · 1.126e-09** — five orders coarser
   (`w8-downdate/LOG.md:1010-1024`). ⚠ And its worst leg (test seed 1, 1.126e-09) sits **below its own
   measured reproducibility floor of 4.282e-09**, so at that leg the deviation is not resolvable at all.
3. ⛔ **It is not unconditional. It failed first.** Assertion 1 **FAILED on test seed 2 at 3.921e-06 =
   32.90 float32 ulp** on the incumbent's coarse imbalanced partition at A=64
   (`w8-downdate/LOG.md:618-640`). The fix is a **variance-share admission cut with no free parameter**,
   `tau = 4.941056e-16 / 1.1920929e-07 = 4.144858e-09`, **declared before the re-run and not moved**.
4. ⭐⭐ **And the strongest form of the two-implementation claim is the FAILURE agreement, not the
   successes:** `hub/single-design` reproduced the same seed-2 failure **independently, with different code
   and its own sign convention** — 14.377 ulp against the spoke's 32.90 / 35.46
   (`w8-single-design/LOG.md:481`).

⭐ **What IS unambiguously in the 1e-16 band, and is the number the paper should lead with: CMAZE, all three
scored seeds, statistic 7.5e-16…8.8e-16 and subspace projector EXACTLY 0.000e+00, replicated from a clean
checkout with ZERO numeric differences.**

⇒ **`CM-I8-1(b)` prints the three implementations as three separate rows** so no draft can quote one
implementation's band as both. ⛔ **Any paper sentence saying "9/9 seeds" is wrong as written.**

---

## 4. ⭐ THE CORRECTION THAT GIVES GROUND BACK (why reading only `NOVELTY.md` would have been worse)

`NOVELTY.md` §3 told the Hub not to claim covariance-downdate exactness, citing **Hall–Marshall–Martin**
among the owners. The addendum's §D **withdraws that attribution** after a full-text read:

> *"Subtracting eigenspaces yields a result **approximating** that which would obtain…"* · *"The subtraction
> operator **tends to instability**…"* · *"best used to remove a **small fraction** of the data points"* ·
> *"We have not performed any rigorous error analysis… **anecdotal in character**."*

The reason is structural: **they subtract DEFLATED eigenspaces.** Once the tail is truncated the exact
covariance of the remainder is unrecoverable (`NOVELTY.md:286-316`).

⇒ **The registry records BOTH halves:** ⛔ do not claim the primitive (Cao & Yang 2015 owns it in one line),
⭐⭐ **and do not concede exactness to the decremental-PCA line — it does not have it.**
⭐ **The sentence the paper should carry: *"downdate the statistic, not the truncated factors."***
`N-I8-1`(F), `CM-I8-1`(c), never-quote §I8.1(a)4.

---

## 5. WHAT THE HUB SHOULD DECIDE (⛔ raised, deliberately not decided here)

1. ⛔ **The `N89`/`N90` duplication** — renumbering is the Hub's call. This fold touched neither series.
2. ⚠ **The `⟲ 2026-09-11` gen-pivot negatives still have NO claims-matrix row at all.** Outside this
   spoke's item list; flagged, not written.
3. ⚠ **Two wave-8 Advisor questions carry no ruling in the evidence base** — whether the compose/forget
   crossover is framed as *"a budget result"* (`outputs/w8-advisor-report.md` §5.1) and whether the
   encoder-loop decomposition earns a paragraph (§5.3). ⭐ **`CM-I8-5` and `CM-I8-4` give approved wording
   for both answers, so the paper-writer is unblocked either way.**
4. ⚠ **Live budget, not a null:** `generation-table` run 3 and `single-design` 3/3 remain unspent.
5. ⚠ **Spot-checks before ratifying tiers:** `N-I8-2`'s RESULT table against
   `outputs/_thinkpad/w9/w9-maze/runs/*.json`; `CM-I8-1(b)` / `N-I8-5`'s byte ledger against
   `outputs/_thinkpad/w8/w8-downdate/DOWNDATE-READY*.json`.
6. ⚠ **Tiers are CURATOR-PROPOSED** (A ×10, B ×7, C ×4). The curator never self-confirms.

---

## 6. WHAT WOULD PROVE THIS SPOKE WRONG — checked against my own result

The task's falsifier is *"a row you cannot source ⇒ say so in the row rather than paraphrasing a report."*

- ⭐ **Every one of the 21 registry entries and 13 matrix rows carries at least one number and the file path
  that holds it**, and the 32 never-quotes each carry the number that forbids them.
- ⭐ **The one item I could not source as commissioned is §3 above**, and it is written into the row itself
  (`N-I8-1` §A, `CM-I8-1`(e)(i)) and into the Hub flags — **not paraphrased away**.
- ⚠ **Two further places where I recorded a narrower claim than the report's headline**, rather than
  copying it: (a) `N-I8-8` — the Advisor ruled placement "CLOSED on the label-MI partition", but the
  **later w8 measurement (0.8918 vs 0.8941) narrows that: what is closed is the DIAL, not the criterion**;
  (b) `N-I8-17` — w9's BLIND-SEQUENTIAL of 0.9037 is recorded as a **REQUEST-LAYER** weakness against
  CMAZE-2, **not** as a defect in the adopted CMAZE-1, because that is what the measurement shows.
- ⛔ **What I did NOT do:** re-derive any statistic, resolve the `N89`/`N90` collision, write a row for the
  gen-pivot negatives, or confirm any version — each is declared above rather than silently skipped.

---

## 7. COST

⛔ **Zero scored runs, zero compute, zero GPU, no background jobs, no PIDs launched, no box-wide kill
possible or issued.** One session, reading and writing documents only. Deliverables are in this directory
and are committed on branch `w10/registry-fold` under `projects/w10-registry-fold/`, bundled to
`w10-registry-fold.bundle` (that branch only, verified).
