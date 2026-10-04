# w10 spoke/seeds-and-tables — FINDINGS (for the Hub)

**Agent:** results-analyst. **Window:** 2026-09-21T08:26:07Z → close (see `END_UTC.txt`).
**Scored runs spent: ZERO**, as chartered. This spoke re-read artefacts and produced no new number
from a model.
**Deliverables:** `TABLES.md` · `TABLES.json` · `figures/` (5) · `build_tables.py` ·
`verify_tables.py` · `LOG.md` · this file.
**Branch:** `w10/seeds-and-tables`, bundled as `w10-seeds-and-tables.bundle`.

---

## 0. The acceptance criterion, discharged mechanically

> *"`TABLES.md` contains one table where every cell traces to a file path recorded in `TABLES.json`,
> and a reader can regenerate every figure from that manifest alone."*

`verify_tables.py` does not import the builder. It re-reads `TABLES.json`, re-opens every source at
the path recorded there, re-resolves every `provenance` pointer and asserts the manifest value equals
the value on disk. Output:

```
  sources registered .............. 27
  source hashes re-verified ....... 27
  cells found in TABLES.json ...... 240
  per-seed values re-resolved ..... 720  (of which 3 via a declared reduction)
  figures ......................... 5  (5 hashes re-verified)
  declared NOT-RUN rows ........... 5
  corrections ..................... 4
  PASS
```

⭐ **Figures are byte-identical across rebuilds** (`sha256sum figures/*.png` before and after a second
`python build_tables.py` — zero diff), so "regenerate from the manifest alone" is a checked property,
not a promise.

⭐ **And the whole thing regenerates from the BUNDLE.** `git clone --branch w10/seeds-and-tables`
from `w10-seeds-and-tables.bundle` into an empty directory, then
`W10_OUT_DIR=<tmp> python scripts/w10_build_tables.py`, produces **byte-identical figures** (sha256
diff: zero) and a **byte-identical `TABLES.md`** (modulo the build timestamp line). Nothing in this
spoke's output depends on state that is not either in the bundle or in the source register.

⭐ **The verifier earned its keep: it failed twice on real defects of mine** — an unregistered source
(`w9-maze/replication/REPLICATION.md`) and a path shorthand `scored{1,2,3}_seed{0,1,2}_test.json`
that expands to 9 files when only 3 exist. Both fixed; the check is what found them.

---

## 1. ⛔ The tables had to be SPLIT FOUR WAYS. One scoreboard was not available.

The charter forbids rows at different compute budgets sharing a table. Applying that rule to what is
actually on disk produces **four tables, not one**:

| table | cell | pin | seeds | source of the pin |
|---|---|---|---|---|
| **T1 (MAIN)** | CMAZE-1, w9 hub/maze, TEST | K=8, A=31, ridge=0.01, AE 12,000, 410,688 read floats; baselines matched at 405,776–410,920 params | **3/3 scored** | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/ledger`, `/baseline_params` |
| **T2** | Shapes3D, w8 hub/encoder-loop | `w8_v1_short` decoder, `joint_steps=400`, `n_train=8192`, `dec_refit_steps=400` | 3 | `/c/chlu/outputs/w8-encoder-loop/run/encoder_loop_report.json` `/pins` |
| **T3** | CMAZE-1, w9 harness tier | ~60k-param arms, 500 train / 200 eval, enc 1,200 / bc 2,000 | 3 | `/c/chlu/outputs/w9-harness/harness_seed0.json` `/config` |
| **T4** | exact deletion, two implementations | w9/maze `CluRead` + w8/downdate covariance store | 3 each | `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` |

⚠ **T1 and T3 are the same environment at different budgets** — 410k-param arms over 96 mazes /
1,208 requests versus 60k-param arms over 500/200 instances. Merging them would have been the easiest
mistake available and it would have been void.

⛔ **T2 cannot be merged with the incumbent's Shapes3D row-3/row-4 values** and I did not try: the
metric there is a COUNT (`distinct_valid_novel_tuples`, 36/corner ceiling on row 3, 72 on row 4), the
decoder is the shortened `w8_v1_short`, and the incumbent's cell is a validity rate on `eval_b`.

---

## 2. ⛔ The main scoreboard's verdict is NEGATIVE, at 3 seeds, and it is clean

T1, mean of scored seeds 0/1/2, validity under the exact judge
(`/c/chlu/outputs/w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` `/table/<row>/<arm>/validity`):

| row | CLU | best learned baseline | lookup (matched bytes) | `random_atoms` (C.2) | pipeline ceiling |
|---|---|---|---|---|---|
| 1 seen | **1.0000** | 0.9865 (transformer) | 1.0000 | 0.9601 | 1.0000 |
| 2 unseen pair | 0.1279 | **0.1840** (transformer) | 0.1565 | 0.1414 | 0.6319 |
| 3 n-way, CONJUNCTION | 0.0284 | **0.0493** (lstm) | 0.0159 | 0.0195 | 0.5255 (per route) |
| 3 n-way, PER ROUTE | 0.0852 | **0.1246** (transformer) | 0.0946 | 0.0693 | 0.5255 |
| 4 budget interpolation | 0.0996 | **0.1430** (transformer) | 0.1122 | 0.0905 | 0.5573 |

- ⛔ **BAR 1 fails on all three composition rows** (−0.0560 / −0.0209 / −0.0434 against the best
  learned baseline; `/c/chlu/outputs/w9-maze/LOG.md` §5).
- ⛔ **BAR 3 fails and it is the mechanism.** `landscape_shuffled` 0.9509 and `random_atoms` 0.9601
  against the CLU's 1.0000 on row 1; on row 2 `random_atoms` 0.1414 sits inside the CLU's own seed
  spread (0.1590 / 0.0597 / 0.1651). Only `landscape_removed` collapses (0.0000 everywhere).
  ⛔ **Under the hub's own pre-registration no RESULT row is claimed for this arm.**
- ⭐ The row-1 result is real but uninformative: the matched-bytes lookup also reads 1.0000.
- ⭐ Every arm on every row sits far below its ceiling, so the rows measure composition, not
  rendering (`/pipeline_ceiling_rows`, `/span_ceiling_rows` — at this rank the store span tracks the
  pipeline ceiling to within 0.0138 on every row and every seed — exactly equal on 7 of the 12
  row×seed cells — so the atoms are not the binding constraint; the address book is).

---

## 3. ⭐ The three charter traps, each caught with its number

### 3.1 The n-way row is a CONJUNCTION; its ceiling is PER ROUTE — both are printed

Scored n-way row `n` = **183 / 192 / 191** (`/table/nway/clu/n`).
Its printed ceiling's `n` = **389 / 403 / 414** (`/pipeline_ceiling_rows/nway/n`).
Different denominators, so `0.0284 vs 0.5255` is **not** a fraction-of-ceiling. T1 carries row 3
twice and figure 2 draws both. The per-route reading is the one comparable with the ceiling, and the
CLU loses there too (0.0852 vs 0.1246).

### 3.2 BLIND-SEQUENTIAL on the hub's split, never the published 0.4980

| split | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| row 2, unseen pair (`/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity`) | 0.9077 | 0.8905 | **0.9128** | 0.9037 |
| row 3, n-way (`/env_columns/nway/...`) | 0.9229 | 0.8933 | 0.8841 | 0.9001 |
| published CMAZE-1 (`/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt`) | — | — | — | 0.4980 |

### 3.3 w8/encoder-loop absolutes are quarantined in T2

They are in their own table with their own pins in the caption, and the caption says in terms that
`w8_v1_short` is a different instrument.

---

## 4. ⚠ FOUR CORRECTIONS the paper must absorb — narrative vs artefact

Full text and provenance in `TABLES.md` §CORRECTIONS. None changes a verdict; all change a printed number.

- **C1.** The task file's band *"BLIND-SEQUENTIAL 0.9037–0.9130"* is not a band in the artefact.
  0.9037 is the 3-seed MEAN and 0.9128 (not 0.9130) is seed 2's value; the per-seed minimum is
  **0.8905**. The w9-maze LOG's 09:40Z bullet is where the 0.9130 comes from; the JSON says 0.9128
  (`scored3_seed2_test.json` and its `.txt` companion both print 0.9128).
- **C2.** ⛔ **The exact-deletion range quoted to the Advisor is wrong by ~5 orders of magnitude.**
  The w8 advisor report says *"downdate == refit … to 1e-16…1e-14"*. The artefact says
  `worst_max_abs_dev_over_seeds = 1.1255e-09` on the TEST leg (seed 1, well 3) and `1.1642e-10` on
  val (`DOWNDATE-READY-test.json` / `DOWNDATE-READY.json` `/downdate_equals_refit`). **The claim
  survives** — the artefact's own bar is `worst_in_float32_ulp_at_unit_scale = 0.009442`, i.e. under
  one hundredth of a float32 ulp, and `passed_every_seed = true` with the negative control firing on
  every seed — **but the paper must quote the ulp figure or the true absolute, not `1e-14`.**
- **C3.** The conjunction/per-route denominator mismatch above, now traced rather than asserted.
- **C4.** ⭐ *"Wave 9 inherits the 0.4980 blind-sequential column as a caveat"* mis-attributes it.
  At the **harness tier** BLIND-SEQUENTIAL reads **0.5317 (0.5550 / 0.5300 / 0.5100)** and
  FRAGMENT-LOOKUP r=1/r=2 are **0.0000** on every seed — **the published tripwire reproduces intact
  there**. It is destroyed only by the hub's *unseen pair on a SEEN maze* request layer, where 1,242
  of 1,449 optimal routes pick up exactly one key so there is no key ORDER for rule B to trap.
  ⇒ **This is a property of the REQUEST LAYER, not of CMAZE-1, and the two numbers belong to two
  different splits.** The spec correction for the next wave is already in w9-maze's LOG §3: a request
  layer must force pairs whose routes cross two or more doors.

---

## 5. ⛔ FIVE DECLARED NOT-RUNS, each with a launch line (never a blank, never a zero)

| id | row | status | cost to close |
|---|---|---|---|
| **NR1** | Shapes3D incumbent cell (K=64, A=1,216, `eval_b`) at 3 seeds | **seed 0 only exists** (arm 0.8495 / 51 tuples at 454,336 B vs `knn_proto` 0.8588 / 57 at 475,136 B; Δ −0.0093 = TIE under R1) | ~40 min CPU for seeds 1–2, but each spends one of the remaining `generation-table` runs (**1 left**) |
| **NR2** | Shapes3D at the GPU pin (`joint_steps=4000`, `n_train=32768`, `dec_steps=1800`) | deferred; blocked only on the Docker daemon on the Head's Mac | **$26.46** for 3 seeds, A100-40 SPOT, sequential (concurrency is ONE); `kill -9` resumability proved |
| **NR3** | w8 single-design RESULT rows 3/4 + post-deletion rows | **0/3 scored, budget intact** — falsified on val before a scored run was justified | no object on disk can carry the row at the byte pin |
| **NR4** | hub/maze-encoder-loop | contingent on a positive w8 encoder-loop result; **the contingency did not fire** | zero budget drawn |
| **NR5** | w9/maze post-deletion composition with the REPAIRED deletion, on TEST | measured on **VAL, seed 0 only** (−0.0120, inside BAR 6's ±0.05) | ~240 s/seed, but it is a held-out number and the maze budget is **3/3 spent** |

⚠ **NR5 is the one the Hub should look at hardest.** The scored DISPLAY row says post-deletion
composition on surviving targets goes **0.1024 → 0.0000** (per seed 0.1569/0.0702/0.0802 → 0.0000).
That is the as-pre-registered number and it FAILS. The cause is diagnosed and is the hub's own carry
being wrong for this read path (w8's `W_k ← W_k R_k` is stated for the orthonormal basis `U_k`; this
read is expressed in the deployed modes `G_k`). With the repaired mechanism the VAL seed-0 delta is
**−0.0120**. ⛔ One VAL measurement is not three TEST measurements, so it is NR5 and not a row.

---

## 6. What the evidence base will and will not support

⭐ **Supported at 3 seeds, traceable end to end:**
1. **Exact deletion** — two independent implementations, TEST condition, 3 seeds each, 0 refits, 0
   passes over retained data, items never kept, negative control firing on every seed (T4, figure 4).
   This is the strongest claim on disk, and the novelty scout found it unclaimed.
2. **The encoder-loop DECOMPOSITION** — unfreezing with the store on the gradient path hurts
   (A′ negative on 5 of 6 seed-rows, two at p ≤ 0.0013), the swap loss repairs it (A″ positive on 4
   of 6, two at p ≤ 0.0003), and the net is a wash because two real effects cancel (T2, figure 3).
3. **The CMAZE-1 negative result** — a matched-parameter transformer beats the CLU on every
   composition row, and the C.2 controls do not collapse (T1, figures 1–2).
4. **The environment's own columns** at both tiers, including the request-layer finding in C4.

⛔ **NOT supported, and the paper must not print it:**
1. **Any Shapes3D composition scoreboard at 3 seeds.** One scored seed exists (NR1).
2. **"The store shapes the representation."** BAR B fired on row 3
   (`CROSS-ARM.json` `/VERDICT/r3_comp_2way/BAR_B_FALSIFIED = true`).
3. **A RESULT row for the CLU on CMAZE-1.** BAR 3 fails; the hub's own pre-registration forbids it.
4. **An object that composes and forgets inside 524,288 B.** Falsified on val (NR3).
5. **Post-deletion composition holding on TEST.** Only the failing as-pre-registered number is
   held-out (NR5).

⚠ **The honest summary of the artefact base's thickness:** the programme has **one** environment with
three scored seeds and a clean instrument (CMAZE-1), and that environment returns a negative result;
it has **one** claim that is positive, replicated across two implementations and three seeds, and
unclaimed in the literature (exact deletion); and its flagship generative cell has **one seed**. Two
of the four tables in this file exist only because a second compute pin was never run.

---

## 7. What I did not do

- ⛔ No scored run, no model fitted, no number produced that was not already on disk.
- ⛔ `$CHLU_CLAUDE_ROOT` was **not read at any point**. The source register in `TABLES.json` is the
  complete list of files opened.
- ⛔ Nothing written outside `/c/chlu/outputs/w10-seeds-and-tables/` and my own worktree
  (`scripts/w10_build_tables.py`, `scripts/w10_verify_tables.py`, both additive).
- ⛔ No background job was left running; no box-wide kill at any point (this spoke launched no
  long-running process at all — the builder runs in ~1 s in the foreground).
