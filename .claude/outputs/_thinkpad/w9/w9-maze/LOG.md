# w9 hub/maze — LOG

TARGET: does the object transfer off Shapes3D. Base `base/iter8 = fa60b15`.
Worktree `/c/chlu/worktrees/CHLU-w9-maze`, branch `w9/maze`. Outputs `/c/chlu/outputs/w9-maze/`.
Budget: **3 scored runs**. Scoring gated on `/c/chlu/outputs/w9-harness/HARNESS-READY.json`. DEV starts now.

## 2026-09-21

- 03:57Z START_UTC.txt written. Booted charter → `w9-hub-common.md` → `w9-hub-maze.md`, nothing else.
- 03:58Z Env verified: `chlu.__file__ = C:\chlu\worktrees\CHLU-w9-maze\chlu\__init__.py` (inside worktree),
  jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4 / optax 0.2.6 / scipy 1.17.0. System python, no venv. 3 threads.
- 04:00Z Gate `/c/chlu/outputs/w9-harness/HARNESS-READY.json` ABSENT (w9-harness has only START_UTC.txt).
  DEV proceeds; no scored number is taken until the gate file exists.
- 04:05Z Read the three objects this hub must port:
  * `w8_envscout/maze_colour.py` — CMAZE-1: grid+tint+door+keys, exact integer judge, BFS solver over
    (cell, keystate), adversarial tint poisoning, `generate()` defaults = the ADOPTED configuration.
  * `chlu/experiments/exp_w8_single_design.py` — `CovContentStore`/`WellStats`: per-well sufficient
    statistic `S_k, t_k, n_k`, atoms = `[mu ; leading eigenvectors]`, deletion = rank-m downdate, no refit,
    items never kept. This is the CLU content that must transfer.
  * `chlu/core/gen_request_head.py` — the read `x_hat = Phi(request) W_k G_k`, factorial design
    (bias | main | pair), `one_hot_interpolated` (this is what makes a BUDGET BETWEEN trained values
    well-defined), closed-form ridge `solve_head`, `assert_head_request_only` (B.3 by signature).

### Design, fixed before any number (04:10-04:30Z)

- **CMAZE-2 = a request layer over CMAZE-1, engine and judge untouched.** `w9_maze/task.py`
  calls `generate()` with ITS OWN DEFAULTS and adds named agents and named targets inside one maze.
  Agents and targets are placed in TWO DIFFERENT door-free components and only CROSS pairs are kept,
  so "the doors are load-bearing" holds for EVERY pair, not just the one `generate()` guarantees.
  Adversarial tint poisoning is generalised: a tint is kept only if every kept pair stays feasible.
- **The request is four words** `(maze, agent, target, budget)`; budget is an ABSOLUTE step budget
  from the grid {30,45,60}, interpolated at {38,53} for row 4 by `one_hot_interpolated`, unchanged.
  A pair is written only at budget words its optimum fits: **no impossible example is ever written.**
- **The item is a picture**: the maze (floor/tint/door/key/agent/target, 15 ch) plus the answer drawn
  as a PLAN MAP -- for each (cell, phase) the action the optimal route takes, phase =
  2*(keys used)+carrying, 10 phases x 6 classes = 60 ch. Total 15x15x75.
- **The shared action head is parameterless**: read the class at the cell you stand on in the phase
  you are in, take that action, under the environment's own dynamics. Every arm emits a plan map and
  is executed identically, so a margin between arms is a margin on the READ.
- **Arms**: CLU = `exp_w8_single_design.CovContentStore` (per-well sufficient statistic, atoms =
  [mu; leading eigenvectors], exact rank-m downdate) + `gen_request_head` (factorial design,
  closed-form ridge, `assert_head_request_only`). Controls: landscape-removed, landscape-shuffled,
  random-atoms, no-pair-block. Floor: matched-bytes lookup + full-bytes lookup. Baselines: MLP,
  LSTM, transformer at matched parameters (matched to the CLU's DEPLOYED read floats).
- Rows: 1 seen (train pairs), 2 unseen pair on a seen maze (both words still grounded in that maze),
  3 n-way = two agents to one target, scored as the CONJUNCTION, 4 budget interpolation at 38/53.

### Measured before any arm existed

- 04:12Z **Representation ceiling is EXACT.** Draw the optimum as a plan map, execute the drawing,
  judge it: **144/144 valid, score 1.0000, 0 (cell,phase) collisions** on 8 sites; 1.0000 on all four
  rows of the 12-site smoke. The (cell, phase) index is sufficient -- the representation costs nothing.
- 04:15Z Banks built, 3 seeds x 96 mazes, 18 pairs each: acceptance 0.200/0.209/0.229 of generator
  seeds, 281s/272s/261s. `opt` mean 34.9 (CMAZE-1's one-pair distribution: 42.4), **rule B bends
  ~0.75-0.92 of pairs** (CMAZE-1: 0.816).
- 04:35Z **The pipeline runs end to end** (12-site smoke, 300 AE steps): ledger, matched parameters
  (mlp 19,807 / lstm 19,646 / transformer 20,864 against a target of 19,968), all rows, all controls,
  env columns. Every arm 0.0000 -- the encoder at 300 steps is the bottleneck, as expected.

### 04:36Z ⚠ FIRST REAL FINDING, and it is against the task file's expectation

- **FRAGMENT-LOOKUP is NOT 0.0000 on this split: r=1 0.0429, r=2 0.6571** (12-site smoke, unseen-pair
  and n-way rows). This is NOT train/test contamination -- no held-out item is in the table. It is
  structural: the rows are *unseen pairs on a SEEN maze*, so the fragment table contains routes
  through the very same maze, and a 5x5 colour-canonicalised patch is close to a position index
  inside a 15x15 grid. CMAZE-1 measured 0.0000 because ITS eval mazes were unseen.
  Consequence, and it is adopted: the fragment column is split into **FRAG-same-maze** (the honest
  floor for this hub's rows, printed beside every row) and **FRAG-cross-maze** (the CMAZE-1
  comparison). An arm that does not beat the same-maze fragment table has not composed anything.
- **BLIND-SEQUENTIAL on this split is 0.7500, not 0.4980** (12-site smoke) -- the cross-component
  pairs are shorter than CMAZE-1's. Both numbers are printed on every row, and every row is broken
  out by whether rule B bends that route, exactly as the target file requires.

### 06:10Z DEV leg 1 (val, seed 0, 96 mazes) -- four findings, two of them blockers

- ⭐ **EXACT DELETION TRANSFERS, first time off Shapes3D.** Deleting every item written under
  the word `target=green` (216 rows) by rank-m downdate of the sufficient statistic:
  **statistic max relative residual 4.901e-16, subspace projector residual 0.000e+00**,
  refit False, passes over retained data 0, items kept False. The wave-8 property holds verbatim
  on maze content. (This is a property of the store, not a composition number.)
- ⛔ **BLOCKER 1 -- the shared decoder memorises.** Pipeline ceiling (encode->decode->execute->judge)
  is **1.0000 on TRAIN items but 0.4087 / 0.2041 / 0.2810** on the unseen-pair / n-way /
  budget-interpolation rows. The encoder was trained on 864 items and simply learned them. No arm can
  exceed that, so no row below it means anything. FIX: an ENCODER POOL of mazes disjoint from the
  evaluation bank, so the shared encoder learns a generic maze+route code and still sees no held-out
  item. Re-measured before any arm is believed.
- ⛔ **BLOCKER 2 -- the latents are unstandardised**, so the learned baselines were fitting targets
  with MSE ~1900 and scored 0.0000 across the board. That is not a fair floor. FIX: per-dimension
  standardisation computed on TRAIN latents, applied to every arm identically, inverted before decode.
- ⚠ **DESIGN WEAKNESS in CMAZE-2, reported as a correction to the spec (common file rule 5).**
  BLIND-SEQUENTIAL scores **0.8750** on my unseen-pair rows and **1.0000** when measured on all pairs
  at slack 10 %. Cause, measured: **1242 of 1449 optimal routes pick up exactly ONE key**
  (198 use two, 9 use three), because a cross-component pair needs exactly the one door that separates
  its two components -- and with one key there is no key ORDER for rule B to trap. Only 8 of 30 mazes
  have three door-free components of >= 4 cells, so the door graph is nearly always a single cut.
  Stronger poisoning does nothing: adversarial=2/4/6 with max_poison=6/12/16 all leave
  blind-sequential at 1.0000. ⛔ This is a property of the REQUEST LAYER, not of CMAZE-1.
  Consequence, adopted: the CMAZE-1 number (0.4980) and MY number are BOTH printed on every RESULT
  row, and every row is broken out by whether rule B bends the route. What this hub's rows test is
  REQUEST composition (maze x agent x target x budget), not rule A x rule B composition.
- ⚠ FRAGMENT-LOOKUP on this split: r=1 0.0255, r=2 **0.4586**. Same-maze fragments are a strong
  floor here (see the 04:36Z bullet); it is printed beside every row.

### 06:45Z Harness gate

- `/c/chlu/outputs/w9-harness/HARNESS-READY.json` EXISTS as of its stamp 2026-09-21T04:36:37Z, with
  `"status": "partial"` -- seeds [0] of [0,1,2], baselines `mlp true / lstm false / transformer false`,
  judge `w8_envscout.maze_colour.judge` re-exported unwrapped (the same function this hub calls),
  columns fragment-lookup and composition-blind present as REPORTED COLUMNS.
  ⛔ Nothing in `/c/chlu/outputs/w9-harness/` is read further, written or edited by this hub
  (charter A.4: a hub that edits the harness voids its numbers).
  The gate is a readiness signal for SCORED runs; its state is re-read and recorded at each scored run.

### 08:05Z DEV leg 3 -- the store-configuration sweep on VAL (D3). ⭐ THE MECHANISM IS MEASURED.

Identity that governs the whole table: **span fraction = K x n_atoms / D = n_atoms / d_k = the
SATURATION RATIO.** A store cannot span more of the latent space without approaching the wave-8
saturation trap. `span:*` = a TRUE latent projected onto the deployed atom span, decoded, executed,
judged -- the hard bound on the CLU whatever its address book. Ridge 0.01 dominates everywhere;
ridge 100 zeroes every cell. Selection is on VAL; the test split is untouched.

| K | atoms | rank/d_k | span:seen | span:unseen | clu:seen | clu:unseen | read floats |
|---|---|---|---|---|---|---|---|
| 8 | 4  | 0.094 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 52,992 |
| 8 | 8  | 0.219 | 0.0257 | 0.0337 | 0.0240 | 0.0192 | 105,984 |
| 8 | 16 | 0.469 | 0.5099 | 0.2885 | 0.4445 | 0.0673 | 211,968 |
| 8 | 24 | 0.719 | 0.9843 | 0.5529 | 0.9735 | 0.1154 | 317,952 |
| 8 | 31 | 0.938 | 1.0000 | 0.6106 | 1.0000 | **0.1394** | 410,688 |
| 16| 12 | 0.688 | 0.9040 | 0.4952 | 0.8212 | 0.0913 | 314,880 |
| 16| 15 | 0.875 | 1.0000 | 0.5673 | 0.9975 | 0.1106 | 393,600 |
| 32| 6  | 0.625 | 0.7591 | 0.4135 | 0.4851 | 0.0721 | 313,344 |
| 32| 7  | 0.750 | 0.9818 | 0.4904 | 0.8129 | 0.0721 | 365,568 |

- ⭐⭐ **The content that wins on Shapes3D is LOW-RANK PER WELL, and maze-route latents are NOT
  low-rank.** At rank/d_k = 0.219 -- the regime the incumbent lives in (Shapes3D deployed A=7 atoms
  in wells of 200-900 channels, ~1-3 % span) -- the store cannot render even a TRAINING item
  (span:seen 0.0257). It needs **rank/d_k >= ~0.72** to render training items and **0.938** to reach
  its best held-out number. 543 training items in 256 dimensions are in general position; there is
  no 6-factor manifold for a principal subspace to find.
- ⭐ The store limit and the address-book limit are now SEPARATED: at K=8/A=31 the span bound is
  0.6106 on unseen pairs (the pipeline ceiling is 0.6202 -- the store is no longer the binding
  constraint) but the CLU reads 0.1394. **The remaining gap is the ADDRESS BOOK**, not the atoms.
- Selected on VAL for the scored legs: **K=8, n_atoms=31, ridge=0.01**, with rank/d_k = 0.938
  declared on every row, plus a second pre-registered **non-saturated** arm `clu_lowrank`
  (K=8, n_atoms=8, rank/d_k 0.219) reported beside it and never instead of it.

### 08:26Z ⭐⭐ SCORED RUN 1 of 3 — seed 0, TEST leg (PREREG-w9-maze-1). **THE DIAL IS FALSIFIED.**

Ledger: read floats 410,688 (atoms 7,936 + address book 402,752) = 1,642,752 B deployed; float64
statistic 35,904 B. K=8, A=31, D=256, design width 1,624, rank/d_k **0.938**.
Matched parameters: mlp 410,920 · lstm 409,846 · transformer 405,776 (all within 2 %).
Ceilings: representation 1.0000 all rows; pipeline 1.0000 (train) / 0.6205 / 0.5090 / 0.5438;
store span 1.0000 / 0.6205 / 0.5090 / 0.5461 (the store is NOT the binding constraint at this rank).
**Caveat columns on THIS split: BLIND-SEQUENTIAL 0.9077 (published CMAZE-1 0.4980); FRAGMENT-LOOKUP
r1 0.0599 / r2 0.5339 (published 0.0000 / 0.0000); rule B bends 0.513 of these routes (published 0.816).**

| RESULT row | clu | clu_lowrank | mlp | lstm | transformer | lookup(matched) | best baseline − clu |
|---|---|---|---|---|---|---|---|
| 1 seen | **1.0000** | 0.0240 | 0.9048 | 0.6507 | 0.9892 | 1.0000 | −0.0108 |
| 2 unseen pair | 0.1590 | 0.0000 | 0.1385 | **0.1897** | **0.1897** | 0.1590 | **+0.0307** |
| 3 n-way (conjunction) | 0.0383 | 0.0000 | 0.0273 | 0.0383 | **0.0492** | 0.0164 | **+0.0109** |
| 4 budget interpolation | 0.1313 | 0.0046 | 0.1290 | **0.1498** | 0.1452 | 0.1198 | **+0.0185** |

- ⛔ **BAR 1 FAILS.** A matched-parameter LSTM and transformer BEAT the CLU on row 2 (0.1897 vs
  0.1590), the transformer beats it on row 3, the LSTM on row 4. **The Shapes3D result does not
  transfer. That is the wave answer and it is reported as such.**
- ⛔ **BAR 2 FAILS.** The CLU ties the matched-bytes lookup exactly on row 2 (0.1590 = 0.1590) and is
  far below the same-maze fragment floor (r2 0.5339). Charter C.1: the store is decorative here.
- ⛔⛔ **BAR 3 FAILS, and this is the finding.** `landscape_shuffled` scores **0.9396** and
  `random_atoms` **0.9222** on the seen row against the CLU's 1.0000; on row 2 `random_atoms`
  (0.1692) BEATS the CLU (0.1590). Only `landscape_removed` (zeroed atoms) collapses. This is
  exactly the wave-8 SATURATION TRAP, arriving on schedule: at rank/d_k = 0.938 any full-rank atom
  block spans the same subspace, so the CONTENT carries no information and the address book does all
  the work. ⛔ **Under charter C.2/C.4 no row is claimed for this arm.**
- ⛔ The non-saturated store (`clu_lowrank`, rank/d_k 0.219 — the regime the Shapes3D incumbent lives
  in) scores 0.0240 / 0.0000 / 0.0000 / 0.0046. It cannot do the task at all.
- ⚠ **The rows break out the same way in both bend classes** (row 2: clu 0.1200 bend / 0.2000 no-bend;
  transformer 0.1700 / 0.2105), so the loss is not an artefact of the easy half — the CLU loses in
  both, and BAR 4's question does not arise because there is no margin to attribute.
- ⭐ **Exact deletion transfers again on the scored leg**: downdate vs refit statistic **7.522e-16**,
  subspace projector **0.000e+00**, 216 rows removed, refit False, 0 passes, items never kept.
- ⛔ DISPLAY refusal, as pre-registered: refusal 0.9714 on budget-infeasible but **0.7979 false
  refusal** on grounded-possible — far outside BAR 6's 0.20. And `mt_unseen` (POSSIBLE, ungrounded)
  is refused at 0.9721 against `walled` (IMPOSSIBLE) at 0.9606: **the control fires — the walled
  refusal is novelty detection, not wall detection**, and the hub says so rather than quoting it.
  The supervised refuser, the only permitted comparator, separates better (0.5099 / 0.3357 impossible
  vs 0.1140 / 0.2384 possible).
- ⚠ Post-deletion composition on the surviving targets fell 0.1569 → 0.0000. See the 08:45Z
  amendment: that number is my basis-change carry being wrong for THIS read path, not a property of
  the object; both the as-pre-registered number and the corrected one are reported.

### 09:12Z SCORED RUN 2 of 3 — seed 1, TEST leg (PREREG-w9-maze-2). Same configuration.

Ceilings: pipeline 1.0000 / 0.5871 / 0.5409 / 0.5485; store span 1.0000 / 0.5871 / 0.5360 / 0.5440.
Caveats on THIS split: BLIND-SEQUENTIAL 0.8905 · FRAG r1 0.0960 / r2 0.5349 · rule B bends 0.473.

| RESULT row | clu | clu_lowrank | shuffled | random_atoms | mlp | lstm | transformer | lookup |
|---|---|---|---|---|---|---|---|---|
| 1 seen | 1.0000 | 0.0067 | 0.9725 | 0.9825 | 0.9192 | 0.7344 | 0.9867 | 1.0000 |
| 2 unseen pair | 0.0597 | 0.0000 | **0.1045** | **0.0945** | 0.1095 | 0.0846 | **0.1741** | **0.1592** |
| 3 n-way | 0.0469 | 0.0000 | 0.0469 | 0.0312 | 0.0156 | **0.0677** | 0.0417 | 0.0052 |
| 4 budget interp | 0.0722 | 0.0000 | 0.1016 | 0.0813 | 0.1151 | 0.1084 | **0.1445** | 0.1129 |

- ⛔ Seed 1 is WORSE for the CLU than seed 0: the transformer beats it on rows 2 and 4 by
  +0.1144 and +0.0723, and **the CLU is beaten on row 2 by its own charter-C.2 controls**
  (`landscape_shuffled` 0.1045, `random_atoms` 0.0945 vs clu 0.0597) and by the matched-bytes
  lookup (0.1592). BAR 1, BAR 2 and BAR 3 all fail again, in the same direction.
- ⭐ Exact deletion again: statistic **8.826e-16**, subspace projector **0.000e+00**, 226 rows,
  0 passes, items never kept.
- DISPLAY refusal: 0.9692 impossible / **0.8560 false refusal**; `mt_unseen` 0.9794 vs `walled`
  0.9846 — the novelty control fires again.

### 09:40Z SCORED RUN 3 of 3 — seed 2, TEST leg (PREREG-w9-maze-3). Same configuration.

Ceilings: pipeline 1.0000 / 0.7018(span) row values in the run JSON; caveats on this split:
BLIND-SEQUENTIAL 0.9130, FRAG same-maze r2 0.4626, cross-maze r2 0.0000, rule B bends 0.469.
Rows (validity): clu 1.0000 / 0.1651 / 0.0000 / 0.0951; transformer 0.9836 / 0.1881 / 0.0419 / 0.1394;
lstm 0.6938 / 0.1606 / 0.0419 / 0.1106; lookup_matched 1.0000 / 0.1514 / 0.0262 / 0.1040;
landscape_shuffled 0.9405 / 0.1514 / 0.0105 / 0.0929; random_atoms 0.9757 / 0.1606 / 0.0000 / 0.0796.
Exact deletion again: statistic 8.2e-16, subspace projector 0.000e+00.

### 09:55Z ⚠ AMENDMENT 1 (appended, not edited) — the deletion mechanism, repaired and re-measured on VAL

The as-pre-registered DISPLAY number stands: on the scored legs, deleting the green target with the
wave-8 **basis-change carry** (`W_k <- W_k R_k`) took the surviving targets from 0.1569 to 0.0000
(seed 0) and 0.0702 to 0.0000 (seed 1). BAR 6's post-deletion-composition clause FAILS as registered.

The cause is mine, not the object's. The wave-8 carry is stated for a read expressed in the
ORTHONORMAL basis `U_k`; this hub's read is `x_hat_k = Phi W_k G_k`, expressed in the deployed MODES
`G_k`, so `R_k = U_k^T U'_k` is not the right change of coordinates here.

⭐ The repair is not a refit. The head's own retained state is a pair of sufficient statistics,
`A = Phi^T Phi` and `Mx = Phi^T X`, additive exactly as the store's `S, t, n` are. So the deletion
subtracts the deleted rows' contribution from them too and re-solves the same closed form
(`arms.CluRead.delete_rows_exact`). No pass over the pool, no items kept, nothing read that the
caller did not hand in.

Re-measured on VAL, seed 0 (⛔ NOT a scored run; the scored budget is untouched at 3/3):
**surviving targets 0.0944 → 0.0823, delta −0.0120**, inside BAR 6's ±0.05, against −0.1569 for the
carry. Deletion exactness unchanged: statistic 7.522e-16, subspace projector 0.000e+00.
⚠ Reported as an amendment with both numbers; the scored-leg DISPLAY row is not restated.

### 10:05Z INDEPENDENT REPLICATION (charter A.5) — scored run 1 replicates BIT-IDENTICALLY

A separate spoke cloned `w9/maze` at `7db71f64b0a34ca3147ff3a0910e5e77c03c96fc` from a bundle into a
clean checkout, modified no code, rebuilt maze bank site 0 from scratch (grid, tint, door, pairs and
routes all equal — the data pipeline is deterministic), retrained the shared encoder from scratch
(2127 s; the hub's leg had loaded a cache) and re-ran PREREG-w9-maze-1's configuration on the test leg.

A full recursive diff of the two result JSONs found **exactly four differing leaves, all provenance**
(`seconds`, `tag`, `ae_cache` path, `ae_loss_history`). **Zero numeric differences.** The
independently retrained encoder is byte-identical (md5 `a2c96e817f58ce5e7f4c5e2965b0b27b`). All three
claims replicate: (a) a matched LSTM/transformer beats the CLU on unseen pairs, 0.1897 vs 0.1590;
(b) the C.2 controls do not collapse, `landscape_shuffled` 0.9396 / `random_atoms` 0.9222 on seen;
(c) exact deletion, statistic 7.5217e-16 and subspace projector 0.0000e+00.

⭐ The spoke also caught a reporting defect of mine, adopted here: **the n-way row is scored as a
CONJUNCTION (n≈183-192 bundles) while its pipeline and span ceilings are per ROUTE (n≈389-414)**, so
0.0284 is not comparable with a 0.5255 ceiling. Both figures are now reported for that row.


### 10:20Z Post-hoc measurement on VAL (⛔ no held-out number involved) — WHY the rank is needed

Per-well explained variance of the top-A principal directions, seed 0 val latents, and what the
store can actually render at that A:

| n_atoms | rank/d_k | explained variance | store span renders a TRAINING item |
|---|---|---|---|
| 4  | 0.094 | 0.3144 | 0.0000 |
| 8  | 0.219 | 0.5438 | 0.0257 |
| 16 | 0.469 | 0.8050 | 0.5099 |
| 24 | 0.719 | 0.9372 | 0.9843 |
| 31 | 0.938 | 0.9922 | 1.0000 |

Global latent spectrum: 50 % / 90 % / 99 % of variance in **17 / 50 / 90** of 256 dimensions;
participation ratio (effective dimension) **46.0**.

⭐⭐ This refines the 08:05Z reading and is the better statement of the mechanism. The maze latent is
NOT especially high-dimensional in variance — its effective dimension is 46 of 256. What is
unforgiving is the JUDGE. A principal subspace holding **0.544** of each well's variance renders
2.6 % of training routes; it takes **0.937** to render 98 %. A route is a sequence of exact discrete
choices and the judge has no tolerance anywhere, so the residual variance a variance-optimal
compressor throws away is exactly what decides whether a route is legal. ⛔ On Shapes3D the read is
graded by a classifier over an image, which tolerates blur; here one wrong action is a zero.
**That is why the low-rank content transfers to one and not the other.**


### 10:55Z ⭐⭐⭐ THE MECHANISM, ISOLATED — composition in the PLAN vs composition in the LATENT

Three analyses of the ALREADY-REPORTED scored legs (no new arm, no new held-out number, no bar moved).

**(i) Every piece of every answer is already written.** PLAN COVERAGE of a held-out unseen-pair route
= the fraction of its (cell, phase, action) triples that already appear in the union of the TRAINING
routes of the SAME maze: **mean 1.000, median 1.000; 194 of 195 pairs at >= 0.95** (seed 0). Nothing
is missing. The row asks for a combination of already-written local decisions, which is exactly what
compositional generalisation is supposed to mean, and it explains why a same-maze fragment table
reaches 0.5105 while every parametric arm sits at 0.13-0.18.

**(ii) The factorisation the CLU assumes is CORRECT.** Coverage of a held-out route by the training
plans that share its AGENT, by those that share its TARGET, and by their union (mean of 3 seeds):

| same-agent | same-target | **UNION** | union complete |
|---|---|---|---|
| 0.838 | 0.885 | **0.997** | **99.1 % of pairs** |

So `(maze x agent) + (maze x target)` -- precisely the pair blocks of the address book -- carries the
whole answer. The CLU is not short of information and is not short of the right factorisation.

**(iii) ⭐⭐ The composition is destroyed by WHERE it is performed.** Same two halves, composed in two
different spaces, then executed and graded by the same exact judge:

| composition performed in ... | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **PLAN space** (overlay the two half-plans, target's plan wins conflicts) | **1.0000** | **1.0000** | **1.0000** |
| **LATENT space** (average the two halves' latents, decode) | 0.1649 | 0.0950 | 0.1449 |
| the same-agent half alone | 0.2835 | 0.1900 | 0.2523 |
| the same-target half alone | 0.1701 | 0.1750 | 0.2523 |

⭐ In plan space the overlay is not merely valid, it is **OPTIMAL** (score 1.0000 = opt/len on every
pair). In latent space the identical decomposition lands at 0.13 — **no better than one half alone**,
and exactly the band every arm in §4 occupies.

Controls on that overlay (it is not a tie-break artefact, and both halves are load-bearing):

| overlay rule | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| target's plan wins conflicts | 1.0000 | 1.0000 | 1.0000 |
| agent's plan wins conflicts | 0.5722 | 0.4250 | 0.4252 |
| target's plan + a RANDOM training plan instead of the agent's half | 0.3557 | 0.3700 | 0.4065 |

⭐⭐ **THE FINDING: the CLU's read is linear in the request design, so composing two addresses
composes their LATENTS; an exact-judge route task composes in the PLAN. The information is present
(i), the factorisation is right (ii), and the operator is available and perfect (iii) — what fails is
that a point-in-latent-space read cannot express "overlay these two plans, the second winning".**
⚠ This subsumes the rank story of 08:05Z rather than replacing it: rank decides whether a single plan
can be RENDERED at all (0.937 explained variance needed), and even with rendering perfect — at the
scored configuration the store span EQUALS the pipeline ceiling — latent superposition still caps the
row at ~0.15.
⚠ It travels with its own caveat: that the overlay is perfect is itself a statement that CMAZE-2's
routes decompose additively, which is the same property that leaves BLIND-SEQUENTIAL at 0.9037 here.
In an environment whose routes need two or more keys the overlay would not be perfect, and the
correction to the spec in §3 of the report applies to this result too.
⛔ The plan-space overlay is a REFERENCE, not an arm: it keeps whole written items and is not matched
to the CLU's byte count. It is reported to locate the failure, never as a win.

---
# FINAL REPORT — w9 hub/maze — TARGET: the world-model audience

**THE WAVE ANSWER: the object does NOT transfer off Shapes3D.** At matched parameters a transformer
beats the CLU on every held-out RESULT row (scored 3/3, three seeds, replicated bit-identically from
a clean checkout); the charter C.2 controls do not collapse, so under this hub's own pre-registration
no row may be claimed for the arm at all.

⭐⭐ **And the failure is located, not merely reported (§6b).** On these rows every piece of every
answer is already written (plan coverage 1.000), and the factorisation the address book assumes is
correct (the same-agent plus same-target training plans cover 0.997 of a held-out route). Composing
those two halves **in PLAN space scores 1.0000 — optimal on every pair, all three seeds**. Composing
the identical two halves **in LATENT space, which is what a read linear in the request design does,
scores 0.13** — no better than one half alone, and exactly the band every arm occupies. The CLU is
not short of content, rank or factorisation; **it composes in the wrong space for an exact judge.**

Two properties DO transfer, exactly and replicated: the request-only read and exact deletion.

## 1. What was built

**CMAZE-2** — a REQUEST layer over CMAZE-1. The engine, the solver and the exact integer judge are
`w8_envscout.maze_colour`, imported unmodified; `generate()` is called with its own defaults. The
layer adds named agents and named targets inside one maze, placed across two door-free components so
that the adoption condition "the doors are load-bearing" holds for EVERY pair, generalises the
adversarial tint placement to all pairs at once (a tint is kept only if every pair stays feasible),
and seals one target behind walls per maze.

A request is four words `(maze, agent, target, budget)`; an item is a maze picture with the answer
drawn on it as a plan map over (cell, phase); the shared action head is parameterless — read the
class at the cell you stand on in the phase you are in and take that action, under the environment's
own dynamics. **Every arm emits a plan map, is rendered by the same frozen decoder through the same
latent standardisation, and is graded by the same exact judge**, so a margin between arms is a margin
on the READ. A pair is written only at budget words its optimum fits: **no impossible example is ever
written.**

The CLU is the incumbent, imported not re-implemented: `exp_w8_single_design.CovContentStore` +
`chlu.core.gen_request_head`. Configuration pinned on VAL: K=8, n_atoms=31, ridge=0.01;
read = 410,688 floats (atoms 7,936 + address book 402,752) = 1,642,752 B deployed, float64
sufficient statistic 35,904 B declared beside it, never netted in.

## 2. Ceilings — measured before any arm existed, printed beside every row

| ceiling | seen | unseen pair | n-way | budget interp |
|---|---|---|---|---|
| representation (draw the optimum, execute the drawing, judge it) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pipeline (encode → decode → execute → judge) | 1.0000 | 0.6319 | 0.5255* | 0.5573 |
| store span (a TRUE latent projected on the deployed atoms) | 1.0000 | 0.6365 | 0.5230* | 0.5566 |

\* per ROUTE; the n-way row itself is scored as a conjunction (see §4).
The plan map costs nothing — 0 (cell, phase) collisions. At the pinned configuration the store span
equals the pipeline ceiling, so **the atoms are not the binding constraint on the scored legs; the
address book is.**

## 3. The caveat columns that travel with every RESULT row

| column | published CMAZE-1 | THIS hub's split (mean of 3 scored seeds) |
|---|---|---|
| BLIND-SEQUENTIAL | **0.4980** | **0.9037** |
| FRAGMENT-LOOKUP r=1 / r=2, cross-maze | 0.0000 / 0.0000 | **0.0006 / 0.0000** |
| FRAGMENT-LOOKUP r=1 / r=2, same-maze | (not measured there) | **0.0741 / 0.5105** |
| rule B bends the route | 0.816 | 0.484 |

⭐ **The split does NOT leak.** The published tripwire reproduces exactly where it was measured:
cross-maze fragment lookup is 0.0006 / 0.0000. The same-maze number is a different quantity — this
hub's rows are unseen PAIRS on SEEN mazes, so a fragment table contains routes through the very same
maze and a 5×5 colour-canonicalised patch is close to a position index in a 15×15 grid. ⚠ It is also
not a like-for-like arm: FRAG reads the true maze at every step, which charter B.3 forbids the CLU.
⚠ **BLIND-SEQUENTIAL is 0.9037 here, not 0.4980, and the reason is measured**: 1242 of 1449 optimal
routes pick up exactly ONE key, because a cross-component pair needs exactly the one door that
separates its two components, and with one key there is no key ORDER for rule B to trap. Stronger
poisoning does not move it (adversarial 2/4/6 at max_poison 6/12/16 all leave it at 1.0000). **This is
a weakness of the REQUEST LAYER, not of CMAZE-1, and it is a correction to the spec for the next wave:
a request layer must force pairs whose routes cross two or more doors.**

## 4. The RESULT rows — 3 scored runs, all reported, the best never alone

Validity under the exact judge; mean of seeds 0/1/2 (per-seed values in the LOG bullets above and in
`runs/*.json`). Matched parameters: mlp 410,920 · lstm 409,846 · transformer 405,776 vs the CLU's
410,688 read floats.

| row | clu | clu_lowrank | mlp | lstm | **transformer** | lookup(matched) | landscape_shuffled | random_atoms | landscape_removed |
|---|---|---|---|---|---|---|---|---|---|
| 1 seen | **1.0000** | 0.0157 | 0.9118 | 0.6930 | 0.9865 | **1.0000** | 0.9509 | 0.9601 | 0.0000 |
| 2 unseen pair | 0.1279 | 0.0000 | 0.1300 | 0.1450 | **0.1840** | 0.1565 | 0.1383 | 0.1414 | 0.0000 |
| 3 n-way, conjunction | 0.0284 | 0.0000 | 0.0196 | **0.0493** | 0.0442 | 0.0159 | 0.0319 | 0.0195 | 0.0000 |
| 3 n-way, per route | 0.0852 | 0.0026 | 0.1120 | 0.1140 | **0.1246** | 0.0946 | 0.0908 | 0.0693 | — |
| 4 budget interpolation | 0.0996 | 0.0015 | 0.1183 | 0.1229 | **0.1430** | 0.1122 | 0.1071 | 0.0905 | 0.0000 |

**The estimand** (PREREG §3): paired difference in validity, CLU minus comparator, paired by
(row, seed, bend class). **120 corners** (≥ 49 required), margin 0.05, direction CLU > comparator.
CLU ahead by the margin in **15**, behind in **21**, within it in 84. Mean paired delta +0.0071,
median −0.0046. Against the transformer: **0 corners ahead, 8 behind, mean −0.0304.**

## 5. The bars, as pre-registered

- ⛔ **BAR 1 — THE DIAL — FAILS on all three rows.** unseen pair −0.0560, n-way −0.0209, budget
  interpolation −0.0434 against the best learned baseline. **A matched-parameter transformer beats
  the CLU. The Shapes3D result does not transfer, and that is the wave answer.**
- ⛔ **BAR 2 — the C.1 floor — FAILS.** The CLU is below the matched-bytes lookup on rows 2 and 4 and
  far below the same-maze fragment table everywhere.
- ⛔⛔ **BAR 3 — the C.2 controls — FAILS, and this is the mechanism.** `landscape_shuffled` 0.9509
  and `random_atoms` 0.9601 against the CLU's 1.0000 on the seen row; on row 2 `random_atoms`
  (0.1414) is within noise of the CLU (0.1279). Only `landscape_removed` collapses. **Under this
  hub's own pre-registration no row is claimed for this arm.**
- **BAR 4 — the easy half.** There is no margin to attribute, but the breakout is reported: mean
  paired delta +0.0195 where rule B bends the route, −0.0053 where it does not. The CLU is not
  winning the easy half; it is not winning either half.
- **BAR 5 — ceiling honesty.** Rows 3 and 4 have pipeline ceilings of 0.5255 and 0.5573 and are
  labelled CEILING-BOUND-adjacent; row 2's ceiling is 0.6319. Every arm on every row sits far below
  its ceiling, so the rows measure composition, not rendering.
- ⭐/⛔ **BAR 6 — DISPLAY.** Deletion exactness PASSES (§6). Refusal FAILS: 0.9601 on
  budget-infeasible but **0.8182 false refusal** on grounded-possible.

## 6. WHY it fails — the mechanism, measured on VAL

The identity that governs this port:

    span fraction = K · n_atoms / D = n_atoms / d_k = THE SATURATION RATIO

A store cannot span more of the latent space without moving toward saturation, where `rank == d_k`,
the projection is the identity and the content carries no information. The val sweep, and the
per-well explained variance measured beside it:

| rank/d_k | 0.094 | 0.219 | 0.469 | 0.719 | **0.938** |
|---|---|---|---|---|---|
| per-well explained variance | 0.3144 | 0.5438 | 0.8050 | 0.9372 | **0.9922** |
| store span renders a TRAINING item | 0.0000 | 0.0257 | 0.5099 | 0.9843 | **1.0000** |
| CLU on unseen pairs (val) | 0.0000 | 0.0192 | 0.0673 | 0.1154 | **0.1394** |

⭐⭐ **It is the exact judge that breaks the low-rank store, not the dimensionality of the data.** The
maze latent's effective dimension is 46 of 256 (90 % of variance in 50 dimensions), so it is not
especially high-rank. But a principal subspace holding 0.544 of each well's variance renders 2.6 % of
training routes, and it takes 0.937 to render 98 %: a route is a sequence of exact discrete choices,
the judge has no tolerance anywhere, and the residual a variance-optimal compressor discards is
exactly what decides legality. On Shapes3D the read is graded by a classifier over an image, which
tolerates blur; the incumbent deploys ~7 atoms in wells of 200–900 channels — 1–3 % span — and wins.
⛔ Here the store must sit at rank/d_k ≥ ~0.72 to render anything, and at the 0.938 it needs to
compete **its own random-atom control does as well as it does**. The only configuration in which this
store can represent maze content is one in which the store is decorative. The non-saturated arm
(`clu_lowrank`, rank/d_k 0.219 — the incumbent's own regime) scores 0.0157 / 0.0000 / 0.0000 / 0.0015.
**That is the transfer answer, and it is a property of the content meeting an exact judge, not of the
tuning.**

## 6b. WHERE the composition is lost — the sharpest statement this hub can make

Three analyses of the already-reported scored legs (no new arm, no new held-out number):

1. **Every piece of every answer is already written.** Plan coverage of a held-out route by the
   TRAINING routes of the same maze: **mean 1.000**. The row asks for a combination of
   already-written local decisions.
2. **The factorisation the CLU assumes is correct.** The union of the same-agent and same-target
   training plans covers **0.997** of a held-out route and is complete for **99.1 %** of pairs —
   `(maze × agent) + (maze × target)` is exactly the address book's pair blocks.
3. **The composition dies in the latent.** The same two halves, composed in two different spaces:

| composition performed in ... | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **PLAN space** (overlay the half-plans, the target's winning conflicts) | **1.0000** | **1.0000** | **1.0000** |
| **LATENT space** (average the halves' latents, decode) | 0.1649 | 0.0950 | 0.1449 |
| one half alone (agent / target) | 0.28 / 0.17 | 0.19 / 0.18 | 0.25 / 0.25 |

Controls: with the agent's plan winning conflicts the overlay falls to 0.42–0.57, and replacing the
agent half with a RANDOM training plan falls to 0.36–0.41 — both halves are load-bearing and the
operator matters.

⭐⭐ **The CLU's read is linear in the request design, so composing two addresses composes their
LATENTS; an exact-judge route task composes in the PLAN.** The information is present, the
factorisation is right, and the operator is available and perfect — what fails is that a
point-in-latent-space read cannot express "overlay these two plans, the second winning". This
subsumes §6's rank story rather than replacing it: rank decides whether ONE plan can be rendered at
all, and even with rendering perfect (at the scored configuration the store span equals the pipeline
ceiling) latent superposition still caps the row at ~0.15.
⚠ That the overlay is perfect is itself a statement that CMAZE-2's routes decompose additively — the
same property that leaves BLIND-SEQUENTIAL at 0.9037 (§3). ⛔ The overlay is a REFERENCE, not an arm:
it keeps whole written items and is not byte-matched. It is reported to locate the failure, never as
a win.

**For the next wave, concretely:** the thing to change is not the partition, the rank or the encoder
— it is the READ. A read that returns a SET of plan fragments to be overlaid, addressed by the
request, would compose where this task composes. That is a change to the read verb, which
`gen_request_head`'s own docstring already records as the largest measured effect in the programme.

## 7. What DOES transfer, exactly and replicated

- ⭐ **Exact deletion.** Downdating the sufficient statistic by the deleted rows and re-deriving the
  atoms reproduces a from-scratch refit to **7.5e-16 … 8.8e-16** (statistic, relative) and
  **0.000e+00** (subspace projector) on all three scored seeds, with `refit False`, **0 passes over
  the retained data** and the items never kept. The wave-8 property holds verbatim on maze content
  and was re-derived bit-identically by the replication spoke.
- ⭐ **The request-only read (charter B.3)** holds by signature and numerically on every arm of every
  row: `head_read` has no feature parameter and the read is bit-stable under feature perturbation.
- ⭐ **The address book's budget interpolation** is structural: an unseen budget word costs the CLU
  nothing relative to the trained ones that the baselines do not also pay.
- ⭐ **AMENDMENT 1's repair**: the head's Gram `A, Mx` is itself a retained sufficient statistic, so
  the deletion can be applied to it too — no refit, no pool pass. Surviving-target composition then
  moves only −0.0120 (val) instead of −0.1569 (scored, with the wave-8 carry).

## 8. The DISPLAY rows — inductive bias, never quoted as RESULTs

Only comparator: a supervised transformer refuser at matched parameters, trained on impossible
examples from half the bank and evaluated on the other half. Our line: **no impossible example was
ever written.**

| condition | truth | CLU refusal | supervised refuser |
|---|---|---|---|
| budget below the shortest legal route | IMPOSSIBLE | 0.9601 | 0.5099 / 0.5702 |
| target sealed behind walls | IMPOSSIBLE | 0.9801 | 0.3357 / 0.3050 |
| (maze,target) never written but REACHABLE | POSSIBLE | **0.9781** | 0.1140 / 0.2545 |
| grounded held-out pairs | POSSIBLE | **0.8182** | 0.2384 / 0.2412 |

⛔ **The control fires.** The object refuses the reachable-but-unwritten combination (0.9781) as often
as the walled one (0.9801): **its refusal is novelty detection, not wall detection**, and its
false-refusal rate on grounded-possible requests is 0.8182 against a pre-registered 0.20. The
supervised refuser, which has seen impossible examples, separates better. ⛔ No refusal win is claimed.
Post-deletion refusal behaves in the declared direction (green requests 0.7642 → 1.0000 after the
content deletion; after the INTERFACE deletion of move-left, 0.9740 refusal on routes that need it vs
0.8696 on routes that do not) — but with a false-refusal floor of 0.80 those margins are not a result,
and the move-left deletion is an INTERFACE deletion in which no atom is touched, so it carries none of
§7's exactness claim.

## 9. The charter C floor checks, one line each

- **C.1 matched-bytes lookup** — ⛔ FAILED: the CLU does not beat it on rows 2 and 4 (0.1279 vs 0.1565;
  0.0996 vs 0.1122). The store is reported as decorative on those rows.
- **C.2 landscape-removed control** — ⭐ collapses to 0.0000 on every row and every seed.
- **C.2 landscape-shuffled control** — ⛔ does NOT collapse (0.9509 on seen): the saturation trap.
- **C.3 the tie clause** — the tie with the lookup on row 2 would count only if the same object also
  delivered refusal-with-accept-unseen-valid, exact deletion, or novel valid tuples on the same
  artefacts. Exact deletion IS delivered (§7); refusal is NOT (§8). Reported, not claimed.
- **C.4 nearest-atom audit** — ⭐ ratio 0.2139 of the typical inter-item distance: the read does not
  settle onto a written item; it is not a lookup table by C.4.

## 10. What would have proved this hub wrong, point by point (task file §"What would prove this hub wrong")

1. *A matched-parameter LSTM or transformer beats the CLU on the RESULT rows* → **YES, it does.**
   Reported as the wave answer.
2. *The CLU wins only where BLIND-SEQUENTIAL also wins* → no win exists; the bend breakout is
   reported anyway (+0.0195 bend / −0.0053 no-bend).
3. *Post-deletion composition needs retraining* → NO retraining is needed, but the wave-8 basis-change
   carry is the wrong change of coordinates for this read; with the head statistic downdated too the
   cost is −0.0120 (val, Amendment 1).
4. *Fragment-lookup rises above 0.0000 on YOUR split ⇒ your split leaks* → **cross-maze reproduces
   0.0006 / 0.0000; the split does not leak.** The same-maze value (0.5105) is a different quantity
   and is named as such.

## 11. Pre-registered runs, all of them

| # | PREREG | leg | seed | result |
|---|---|---|---|---|
| S1 | PREREG-w9-maze-1.md | test | 0 | BAR 1/2/3/6-refusal FAIL; deletion exactness PASS |
| S2 | PREREG-w9-maze-2.md | test | 1 | identical verdicts; CLU weakest of the three seeds |
| S3 | PREREG-w9-maze-3.md | test | 2 | identical verdicts |

**Scored 3/3. The best is not reported alone: every seed is in §4 and in the LOG.**
Best replicable configuration: K=8, n_atoms=31, ridge=0.01, AE 12,000 steps with the encoder pool —
replicated bit-identically from a clean checkout at `7db71f6`.

## 12. Cost

Local CPU only on the Windows hub box; **no GPU was used and no cluster leg was submitted** (CSF3 is
unreachable by any hub; that leg is NOT RUN, and a declared NOT-RUN is not a null). One hub process
plus at most three of its own child python processes at a time; **every PID captured at launch in
`logs/PIDS.txt`; only my own PIDs were ever killed (two encoder jobs, to stop three JAX processes
thrashing, and one sizing run whose answer had arrived); no box-wide kill was ever issued.**

**Wall clock: `START_UTC.txt` 2026-09-21T03:57:46Z → `END_UTC.txt`, about 3 h 15 m, well inside the
charter's 10 h.** Principal compute: 6 maze banks (3 evaluation, then 3 rebuilt to add the walled
target) ≈ 27 min; the encoder pool, 270 disjoint mazes ≈ 2 min; 5 shared encoders at 12,000 steps,
20–35 min each (one of them retrained independently by the replication spoke); 3 DEV legs; 1 store
sweep of 45 configurations; **3 scored legs — the whole budget, 3 of 3**; 1 amendment leg on val;
3 post-hoc analyses; 1 full independent replication. One sub-agent was used, the replication spoke;
it had finished before this hub did.
