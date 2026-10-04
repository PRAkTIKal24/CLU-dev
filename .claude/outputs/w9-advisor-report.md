# WAVE 9 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-21. Two agents, FORGIS_14, `base/iter8 = fa60b15`.**
Launch 2026-09-21 03:57Z. Both finished; `hub/maze` 03:57→06:46Z (3 h 15 m), `spoke/harness` 3 h 20 m.
**Zero GPU spent. Local CPU only.** Record: `.claude/outputs/_thinkpad/w9/`.
⚠ Every number is quoted from an agent's own report. ⛔ **The Hub has re-derived no statistic.**

---

## 0. The scoreboard

| agent | verdict | scored |
|---|---|---|
| ⛔ `hub/maze` | **the composition result does NOT transfer to a route task** — and the reason is isolated | **3/3 spent** |
| ⭐ `spoke/harness` | **exact judge delivered, 3 seeds, both tiers** — ⚠ and it disclosed three defects in itself | — |

**Budget after wave 9:** `maze` **0 left (3/3 spent)** · `single-design` 3/3 intact ·
`generation-table` run 3 unspent · `composition` 2/3 · `deletion` 2/3 · `encoder-loop` 2.

---

## 1. ⛔⛔ `hub/maze` — the answer, and the mechanism behind it

> **The Shapes3D composition result does not transfer to CMAZE-1. Three bars fail, on all three
> seeds, in the same direction. Under its own pre-registration the hub claims NO ROW for the arm.**

**RESULT row 2 (unseen pair) — the composition row, validity under the exact judge:**

| seed | clu | landscape_shuffled | random_atoms | lookup_matched | transformer | lstm |
|---|---|---|---|---|---|---|
| 0 | 0.1590 | **0.1590** | **0.1692** | 0.1590 | **0.1897** | **0.1897** |
| 1 | 0.0597 | **0.1045** | **0.0945** | **0.1592** | **0.1741** | 0.0846 |
| 2 | 0.1651 | 0.1514 | 0.1606 | 0.1514 | **0.1881** | 0.1606 |

1. ⛔ **BAR 1 fails.** A matched-parameter transformer beats the CLU on row 2 on **all three seeds**.
2. ⛔ **BAR 2 fails.** The CLU **ties the matched-bytes lookup exactly** on seed 0 (0.1590 = 0.1590)
   and is beaten by it on seed 1.
3. ⛔⛔ **BAR 3 fails, and it is the important one.** `landscape_shuffled` and `random_atoms`
   **do not collapse** — 0.9396/0.9222 (s0), 0.9725/0.9825 (s1), 0.9405/0.9757 (s2) on the seen row
   against the CLU's 1.0000. Only `landscape_removed` (atoms zeroed) collapses to 0.0000.
   ⭐ The hub names this as **wave 8's saturation trap arriving on schedule**: at `rank/d_k = 0.938`
   any full-rank atom block spans the same subspace, so the **content carries no information and the
   address book does all the work.**
4. ⛔ The non-saturated store (`clu_lowrank`, `rank/d_k = 0.219` — ⭐ **the regime the Shapes3D
   incumbent lives in**) scores 0.0240 / 0.0067 / ~0. **It cannot do the task at all.**

### ⭐⭐ The failure is LOCALISED, and this is the wave's most useful output
- Every piece of every held-out answer is already in the store: **plan coverage 1.000**.
- The factorisation the address book assumes is correct: same-agent ∪ same-target training plans
  cover **0.997** of a held-out route.
- ⭐⭐ **Overlaying the two halves IN PLAN SPACE scores 1.0000 — optimal on every pair, all three
  seeds. Averaging the identical halves IN LATENT SPACE scores 0.13**, the band every arm occupies.

> **The CLU is short of neither content, nor rank, nor factorisation. Its read is LINEAR in the
> request design, so composing two addresses composes their latents — while an exact-judge route
> task composes in the PLAN.** ⭐ **The lever is the READ VERB, not the partition.**

### ⭐ What transferred
**Exact deletion, on all three seeds:** downdate vs refit **7.522e-16 / 8.826e-16 / 8.2e-16**,
subspace projector **0.000e+00**, 216/226 rows removed, **0 passes over the data, items never kept**.
⭐ **Independently replicated** by a separate spoke from a clean checkout at `7db71f6`: a full
recursive diff of the result JSONs found **exactly four differing leaves, all provenance, and ZERO
numeric differences**; the independently retrained encoder was byte-identical.

### DISPLAY rows — reported, ⛔ not claimed
Refusal: **0.9714 on budget-infeasible** but **0.7979 false refusal** on grounded-possible (s0),
0.9692/0.8560 (s1) — far outside BAR 6's 0.20. ⭐ **And the control fires against the hub's own
story:** `mt_unseen` (POSSIBLE, ungrounded) is refused at 0.9721 against `walled` (IMPOSSIBLE) at
0.9606, so **the refusal is novelty detection, not wall detection**, and the hub says so rather than
quoting the favourable reading.

### ⚠ AMENDMENT 1 (val only; the scored budget is untouched)
Post-deletion composition on the surviving targets fell **0.1569 → 0.0000** as pre-registered, and
**BAR 6's post-deletion clause FAILS as registered.** ⭐ The hub then found the cause was **its own
basis-change carry**, not the object: wave 8's carry is stated for a read in the orthonormal basis,
while this read is expressed in the deployed modes. The head's own retained state is a pair of
sufficient statistics that are additive exactly as the store's are, so deletion subtracts from those
too. **Re-measured on VAL: −0.0120, inside BAR 6's ±0.05, against −0.1569 for the carry.**
⚠ Both numbers are reported; the scored-leg DISPLAY row is **not restated**.

---

## 2. ⭐ `spoke/harness` — the instrument, and three defects it found in itself

**`HARNESS-READY.json` = ready:** exact judge, **`tolerance_used: null`**, 3 seeds, both tiers, all
five baselines, both reported columns. Audits: **0/750 route disagreements** with an independent
replay, **0 stepper disagreements**, 0 impossible-set certification failures, and ⭐ **0 impossible
examples ever seen by a non-refuser arm — a checked property of the code, not a claim in prose.**

⭐⭐ **Three defects in its own instrument, found, fixed and disclosed:**
1. The action head **was not one head** — `d_model` tracked trunk width, so arms differed.
2. The observation **hid the absolute budget** (`t/budget` is 0 at t=0 for every instance), making
   budget-impossible mazes **undetectable by construction**.
3. The encoder **is not reliably trained in one run**: seed 1 stalled at BCE 0.080 / probe AUC 0.61
   where seeds 0 and 2 reached 0.005 / 0.92. Fix verified across all six combinations
   (worst-case BCE 0.101→0.010, AUC 0.441→0.754) **at no extra cost**.

⛔ **DECLARED NOT-RUN:** re-running the arm sweep on the fixed encoder — wall clock, the box was
running `hub/maze`'s three-process bank build throughout. **Seed 1's harness arm numbers inherit the
degraded encoder and its tables say so.** Also not run: a larger refuser budget, any RL arm, any
scale above `std`.

⛔ **The refuser does not work, and the spoke showed why it is not the wiring.** Separation is
**+0.047** (cmaze1) and +0.020 (easy). A ridge probe on the same observation at t=0 gets
**AUC 0.48–0.55 — chance — across all six seed×tier runs**, and the budget scalar alone gets
0.38–0.46, so there is no shortcut either. ⭐ **Deciding solvability-within-budget is close to
solving the maze.** DISPLAY comparisons against this refuser are **not yet informative**, and that
is written into the readiness file as a caveat.

⚠ **Learned arms sit at the floor under zero tolerance** — validity 0.0050 (MLP), 0.0017 (LSTM),
0.0000 (transformer), 0.0000 (table floor), against oracle 1.0000. Per-step accuracy is 0.52–0.75
and **0.75⁴³ ≈ 4e-6 — arithmetic, not a bug.** The spoke added a declared `cmaze1-easy` tier (same
generator, same judge, **never averaged with the main tier**) so the hub has dynamic range.

---

## 3. ⚠ THE CAVEAT THE HANDOVER CARRIED — resolved, and it lands on the maze, not the environment

Wave 8 adopted CMAZE-1 on **BLIND-SEQUENTIAL 0.4980** and **FRAGMENT-LOOKUP 0.0000** at both radii.

- ⭐ **The environment is fine as adopted.** On the harness's own instances at env-scout's defaults:
  **blind-sequential 0.5317**, **fragment-lookup r=1 0.0000 and r=2 0.0000** — both reproduce.
- ⛔ **The maze hub's split does not.** Blind-sequential reads **0.9077 / 0.8905 / 0.9130** there.
  ⭐ The hub diagnosed it against itself: **1242 of 1449 routes use exactly one key, so there is no
  key order for rule B to trap.** That is a **weakness of the maze hub's request layer**, logged as a
  correction to the spec — ⛔ not a property of CMAZE-1.
- ⭐ **Fragment-lookup is NOT a leak:** cross-maze reproduces **0.0006 / 0.0000**. The same-maze
  **0.5105** is a different quantity and is named as such.
- ⚠ **Consequence for the paper:** every CMAZE-1 composition claim must carry the blind-sequential
  number **measured on its own split**, ⛔ never the adopted 0.4980.
- ⚠ **And one reporting defect the replication spoke caught:** the n-way row is scored as a
  **conjunction** (n≈183–192) while its ceilings are **per route** (n≈389–414), so 0.0284 is not
  comparable with a 0.5255 ceiling. Both figures are now reported for that row.

---

## 4. Budget and cost
- **Scored runs: 3 of 3 on `maze`, all pre-registered before execution** (`PREREG-w9-maze-{1,2,3}.md`),
  all three reported. ⭐ **The best is not reported alone.**
- ⛔⛔ **LEDGER DEFECT, Hub-owned:** the three rows were written to the hub's **own** `LEDGER.md` and
  **are absent from `/c/chlu/SCORED-LEDGER.md`**. This is the wrong-path ledger failure in the
  policy's own mistakes register. **The Hub is appending them; no spend is unrecorded, but it was
  not visible in the shared ledger at the time it was made.**
- **Cost: local CPU only. No GPU, no cluster leg.** CSF3 unreachable by any hub — a **declared
  NOT-RUN, not a null.** Principal compute: 6 maze banks ≈27 min; encoder pool ≈2 min; 5 shared
  encoders at 12,000 steps, 20–35 min each; 3 DEV legs; a 45-config store sweep; 3 scored legs;
  1 val amendment leg; 3 post-hoc analyses; 1 full independent replication.
- ⭐ **Process discipline held:** every PID captured at launch in `logs/PIDS.txt`; **only the hub's
  own PIDs were ever killed** (two encoder jobs to stop three JAX processes thrashing, one sizing
  run); ⭐ **no box-wide kill was ever issued.** One sub-agent used — the replication spoke, which
  finished before its parent.

## 5. Limits
Everything is 3 seeds · the Hub re-derived nothing · **`maze`'s budget is exhausted**, so no
remedy can be tested this wave · Amendment 1's repair is **val, not held-out** · the harness's
**seed 1 arm numbers inherit a degraded encoder** and the arm sweep on the fixed encoder is a
declared NOT-RUN · the refuser is not informative, so **every DISPLAY refusal row is uninterpretable
against it** · `cmaze1-easy` is a declared separate tier and ⛔ must never be averaged with the main
tier · the maze's request layer under-exercises rule B and that is unfixed.

## 6. What the Advisor is asked to rule on
1. ⭐⭐ **The negative is sharp and localised: the read composes latents, the task composes plans,
   and plan-space overlay of the SAME halves scores 1.0000 against 0.13.** ⛔ **Is "the lever is the
   read verb" the right next question — and is it a NEW TARGET?** The Hub has not picked it.
2. ⭐ **Exact deletion has now transferred to a second environment and been independently replicated
   with zero numeric differences.** It is the programme's strongest and most robust claim.
3. ⚠ **`maze` is 0/3.** Any follow-up needs a budget decision that is not the Hub's.
4. ⚠ **The saturation trap is now measured twice** (w8, w9): a store either saturates and is
   decorative, or does not saturate and cannot do the task. ⛔ **Is that a result the paper states?**
5. ⚠ **CMAZE-1's rule B is under-exercised by single-key routes.** Fixing the request layer is a spec
   correction someone must own before this environment is used again.

---

## AMENDMENT 1 — 2026-09-21, appended by the Hub (never an edit)

**Trigger:** wave 10's `spoke/registry-fold` audited the deletion claim against the files and found
the programme's **phrasing** overstated, though the property itself holds. This lands on §1 and on
§6's item 2 above, and on the w8 report, so it is recorded here where the Advisor will meet it.

⛔ **"1e-16…1e-14 on Shapes3D, 9/9 seeds, two independent implementations" is wrong as written.**
1. **"9/9 seeds" is 3 CONFIGURATIONS × 3 seeds inside ONE implementation** (`hub/single-design`'s
   K=40/44/48 matched-bytes candidates) — not nine seeds.
2. **The band is not shared.** `spoke/downdate-build`'s four legs run **2.842e-14 · 4.547e-13 ·
   1.164e-10 · 1.126e-09** — five orders coarser. ⚠ Its worst leg sits **below its own measured
   reproducibility floor of 4.282e-09**, so there the deviation is not resolvable at all.
3. ⛔ **It was not unconditional — it FAILED first.** Assertion 1 failed on test seed 2 at
   **3.921e-06 = 32.90 float32 ulp** on the incumbent's coarse imbalanced partition at A=64. The fix
   is a variance-share admission cut with **no free parameter**, `tau = 4.144858e-09`, ⭐ **declared
   before the re-run and not moved.**
4. ⭐⭐ **The strongest form of the two-implementation claim is the FAILURE agreement, not the
   successes:** `hub/single-design` reproduced the same seed-2 failure independently, with different
   code and its own sign convention — 14.377 ulp against the spoke's 32.90.

⭐ **What IS unambiguously in the 1e-16 band — and what the paper should lead with — is wave 9's own
number:** CMAZE-1, all three scored seeds, statistic **7.5e-16…8.8e-16**, subspace projector
**exactly 0.000e+00**, replicated from a clean checkout with **zero numeric differences**.

⚠ The registry now prints the three implementations as **three separate rows** so no draft can quote
one implementation's band as another's.
⭐ **Unchanged by this amendment:** the property is real, it is the programme's strongest claim, and
the Hall–Marshall–Martin attribution is still withdrawn — the decremental-PCA line is approximate in
its own words, so ⛔ do not concede exactness to it.

⚠ **Second defect, unfixed and Hub-flagged:** **N89/N90 are allocated twice** in the negatives
registry (a 2026-09-11 gen-pivot addendum re-used them). The spoke flagged it and **allocated no
number in either series rather than compounding it**. ⛔ The gen-pivot negatives still have **no
matrix row at all.**

---

## AMENDMENT 2 — 2026-09-21, appended by the Hub (never an edit)

⛔⛔ **THE ENVIRONMENT IS MISLABELLED THROUGHOUT THIS REPORT. `hub/maze` scored CMAZE-**2**, not
CMAZE-1.** Caught by wave 10's `spoke/referee` (MF10) and corroborated by `spoke/registry-fold`.

**The maze hub's own FINAL REPORT §1 says so in its own words:**
> *"**CMAZE-2** — a REQUEST layer over CMAZE-1. The engine, the solver and the exact integer judge
> are `w8_envscout.maze_colour`, imported unmodified … The layer adds named agents and named targets
> inside one maze."*

**What this changes — and what it does NOT:**
- ⛔ **Every table and row in §1 above is a CMAZE-2 result** and must be relabelled. ⚠ The Hub wrote
  "CMAZE-1" throughout and that is wrong.
- ⭐ **§3's substantive conclusion SURVIVES and is in fact strengthened.** The three numbers are
  **0.4980** (published CMAZE-1) · **0.5317** (CMAZE-1 proper, at the harness tier) · **0.9037**
  (CMAZE-2, the request layer). ⭐ The blind-sequential weakness is a **REQUEST-LAYER** property of
  **CMAZE-2**, ⛔ not a defect in the adopted CMAZE-1 — which is what §3 argued, but could not be read
  coherently while all three numbers sat under one label.
- ⛔ **Consequence for the paper:** the maze section is **CMAZE-2**, defined in one line as a request
  layer over CMAZE-1 with engine, solver and exact judge imported unmodified. ⭐ **The harness tier
  (§2) IS CMAZE-1 proper.** ⛔ **The two must never be merged into one table** — they are a different
  environment, not merely a different budget.
- ⚠ **Also corrected (MF11):** "FRAGMENT-LOOKUP 0.0000 everywhere" is **false on the hub's own
  split** — same-maze reads **0.0741 / 0.5105**. The 0.0000 figures are the **cross-maze** quantity.
  §1's transfer claim is unaffected; the caveat column's phrasing is not.

⭐ **Unchanged:** the verdict (bars 1/2/3 fail on all three seeds), the isolated mechanism
(plan-space 1.0000 vs latent-space 0.13), and the deletion transfer (7.5e-16…8.8e-16, projector
0.000e+00, zero-difference replication).
