# orgdiv-null-arms — experiment-engineer report

**Task + acceptance criterion (one line):** build N1–N5 in their strongest registered forms against
`FROZEN-interfaces.md` with full grids and ledgers, and answer the Hub's re-scoped question —
**does ANY matched-capacity organizer clear `chance + 0.05` on the rule-4-valid unseen split?**
**Status: done.** All five arms landed, the full registered grid ran (584 configs × 3 tune seeds),
`null*` was **computed** over the entire grid × 5 score seeds, every ledger identity check passed,
**17 new tests green, full suite 1 255 passed / 0 failed** on the branch.

> ## ⛔ THE ONE-LINE VERDICT
> **NO arm clears. `null*` = 0.00117 (N5) against a bar of 0.05039, computed over 584 registered
> configurations × 5 seeds — a factor 43 below the bar and 3 measurement grains above chance.**
> **But the family is NOT what failed.** The same launch points that every arm sees carry enough
> information for a combinatorial decoder to recover `A(x)` **exactly on 27.2 % ± 1.3 % of unseen
> queries** — while the frozen `P = 4` launch protocol puts those 4 particles into **2.20 distinct
> wells**, with `≥ F = 4` distinct wells reachable on only **5.0 %** of queries and exact-set
> occupancy measured at **0.0000 / 2 560 queries**. ⭐ **The information is in the launch; the
> `P`-particle occupancy read cannot express the answer. This is a READ-PROTOCOL refutation, not a
> family refutation — and it was the pre-registered §7.3 outcome, committed before the harness ran.**

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. **`FROZEN-interfaces.md`'s matched-capacity ledger row is STALE and internally inconsistent with
   the cat-test's own registered deviation D2.** It states `store parameters = n_atoms·(dim+2) =
   384·14 = 21 504 B` and `byte ratio 3.20× (a = 12)`; the cell that actually ran used `a = 32`
   (`n_atoms = 1024`, **57 344 B**, ratio 9.67×). ⛔ I did **not** improvise a match: `a ∈ {12, 32,
   64}` is the capacity axis and **both readings are measured** (identical result at every `a`).
   Someone must correct the frozen row before it is quoted again.
2. **`FROZEN-interfaces.md` §(ii)'s reader parameter counts do not match the shipped code.** Doc:
   `well_table = 2m = 16`, `mlp_small = 88`. Measured from `fit_readers`: **72** and **92**
   (`sum_linear` 104 and `knn` 0 agree). All four remain below the `N_a·m = 256` bound, so no verdict
   moves — but the ledger numbers in the doc are wrong.
3. ⭐ **The verdict re-labelling.** The Hub's re-scope offered "none clears ⇒ **the family** is
   refuted". Measured, that inference does **not** follow: the family is decodable through the frozen
   `φ` at 0.272 and the *read protocol* is what caps every arm at 0. Every downstream artifact that
   would have said "the compositional family is refuted" must say **"the `P`-particle occupancy read
   protocol is refuted at `P = 4`"** instead. This needs an owner.
4. **`orgdiv-cat-test` §13 open question 1 is ANSWERED and should be closed:** *"measure how many
   DISTINCT wells the P particles occupy per query"* → **2.20 ± 0.02** (raw launch), **1.70** (the
   settled physics read). The settle *reduces* it further.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** **TIER ii — the organizer swap's NULL SIDE**, re-scoped by the Hub addendum to
  the **family-solvability audit**. A hobbled null is the same referee attack as a hobbled rival, so
  every arm got the F3-grade standard: full registered grid, held-out selection, honest power.
- **Laundering control:** I *am* the control; my own guards are (i) byte-compared `φ` and a
  bit-identical launch protocol, (ii) the two-sided byte ledger per arm, (iii) the **shuffle-`φ`**
  launder run beside every arm's score, (iv) the L1–L4 internal-validity anchors.
- **Falsifies:** nothing of mine — I produce `null*`. ⛔ Selection on `Q_unseen` anywhere would
  invalidate the wave; §6 documents the mechanical guard that prevents it.
- **Does NOT falsify:** an arm beating the physics arm is a legitimate outcome, not a defect.

## 0. File ownership (zero-conflict rule) + compute declaration
**Created (mine, and only these):** `chlu/core/null_arms.py` (871 L) ·
`chlu/experiments/exp_null_arms.py` (847 L) · `tests/test_null_arms.py` (280 L).
**Edited (minimal, in-scope):** `chlu/cli/experiment_cmd.py` (**+46/−0**: one new subcommand block
inserted before `exp-paid-access`, one new handler; **no existing line altered** — the whole branch
diff is `4 files changed, 2 044 insertions(+)`, zero deletions).
⛔ **Not touched:** `chlu/core/factored_store.py`, `chlu/experiments/exp_cat_test.py`,
`chlu/core/monitors.py`, `tests/test_factored_store.py` (all `orgdiv-cat-test`'s), and every file
`pilot-placement-probe` declares (`blocks.py`, `exp_cluformer_pilot.py`, `exp_placement_probe.py`,
`train_cluformer.py`, `scripts/csf3/`). `chlu/config.py` untouched.
**Compute:** ONE JAX process at a time in worktree `../CHLU-null-arms` (worktree 3 of ≤ 3; a
concurrent `../CHLU-lane-parallel-controller` worktree exists — **zero file overlap**, verified).
Spent ≈ **75 min** wall: grid 1 060 s · score 90 s · gridmax 1 453 s · mechanism 470 s · ceiling 60 s
· oracle 337 s · tests 27 s + full suite.

---

# 1. ⭐ THE AUDIT, IN ONE TABLE (5 seeds, unseen exact-set accuracy, `tol = 0.478`)

`chance = 3.906e-4` (constant predictor, reproduced exactly) · **bar = chance + 0.05 = 0.05039** ·
clears iff `mean − 2 SE > bar` (prereg §3's convention).

| arm | selected config | params (B) | **unseen, best reader** | ± 2 SE | **SEEN (in-sample)** | shuffle-φ launder | clears? |
|---|---|---|---|---|---|---|---|
| **N1** gradient-placed atoms ⭐ | `lr 0.1, a=64, τ=1.0, init=written, soft` | 28 672 (114 688 B) | **0.0000** | 0.0000 | ⭐ **1.0000** | 0.0000 | ⛔ **no** |
| **N2** VQ (k-means++ ×10) | `n_codes=32, payload fitted` | 384 (1 536 B) | **0.00039** | 0.00078 | 0.0047 | 0.00039 | ⛔ no |
| **N3** static-geometric (F5's null) | `level=b, payload fitted, lr 1e-3` | 288 (1 152 B) | **0.0000** | 0.0000 | 0.0031 | 0.0000 | ⛔ no |
| **N4** kNN | `k=10, uniform, launch_mean` | 0 (+6 144 B state) | **0.00039** | 0.00078 | 0.0000 | 0.0000 | ⛔ no |
| **N5** Titans fast weights | `h=64, θ=1e-3, η=0, α=0.01, gate=surprise, chunk=32` | 840 (+840 state) | **0.00039** | 0.00078 | 0.0000 | 0.00039 | ⛔ no |
| — | *physics arm, quoted from `orgdiv-cat-test` §6.1, never re-adjudicated* | 1 024 atoms (57 344 B) | *0.0008* | *0.0008* | *0.0109* | — | *(died at K5)* |

### 1.1 `null*` — **computed over the ENTIRE registered grid**, not estimated (prereg §4.3)
Every one of the **584 configurations** was refit on all of SEEN and scored on `Q_unseen`, 5 seeds
(⛔ an **oracle-selected upper bound**, reported so the verdict reads *"no configuration clears"*
rather than *"the one we picked didn't"* — it is never any arm's score):

| arm | configs | **grid-max (mean over 5 seeds)** | best single seed | argmax |
|---|---|---|---|---|
| N1 | 180 | 0.00039 | 0.00195 | `lr 1e-3, a=12, τ=1.0, written, soft` |
| N2 | 84 | 0.00039 | 0.00195 | `product_vq, 32 codes` |
| N3 | 60 | 0.00078 | 0.00391 | `lr 1e-2, csb, written, τ=0.2` |
| N4 | 20 | 0.00078 | 0.00195 | `set_code, k=2, idw` (⚠ noiseless-key variant) |
| N5 | 240 | **0.00117** | 0.00391 | `lr 3e-3, h=64, η=0.9, α=0.01, gate=none, chunk=1` |
| ⭐ | **584** | ### **`null*` = 0.00117** | — | **N5**; bar 0.05039 ⇒ **43× short** |

**Capacity is not the binding constraint** (registered anchor L4): N1's grid-max is **identical
(0.00039) at `a = 12`, `a = 32` and `a = 64`** — i.e. at 21 504 B, 57 344 B and 114 688 B of store.
Soft and hard reads are also identical. N5 is flat in width (0.00117 / 0.00039 / 0.00078 at
`h = 64 / 413 / 1024`).

### 1.2 ⭐ The internal-validity anchors — this is why "no arm clears" is a statement about the
**problem** and not about my optimiser (all four registered in `PREREG.md` §3)

| # | anchor | registered | **measured** | verdict |
|---|---|---|---|---|
| **L1** | N1 fits its own training items | ≥ 0.50 | ⭐ **1.0000** (96/96, and 1.0000 in-sample on all 128) | ✅✅ |
| **L2** | N4 `k=1` memorises SEEN | ≥ 0.95 | **1.0000** | ✅ |
| **L3** | shuffle-φ launder ≤ chance + 0.005 | — | **0.0000 – 0.00039 on every arm** | ✅ |
| **L4** | N1 capacity flatness ≤ 0.02 | — | **0.0000** (identical at 12/32/64) | ✅ |
| — | N2 / N5 train-fit liveness (unregistered, reported) | — | **0.9931** (product-VQ) / **0.9097** | live |
| — | N3 train-fit | — | **0.0799** — ⚠ the one arm that is capacity-starved *in-sample* (32 units) | noted |

> ⭐ **N1 is the sharpest single fact in this report.** With **28 672 free floats** — 5× the physics
> arm's store — and the *identical* store parameterisation, N1 fits **100 %** of the items it was
> trained on and scores **0.0000** on rule-4-valid held-out combinations, both inside SEEN
> (validation) and on `Q_unseen`. It memorises perfectly and composes not at all.

**Per the Hub addendum (b):** N1 does **not** clear `chance + 0.05`, so **no revival target is set**
by this wave. Recorded per reader × seed: N1 = `0.0000` on all 4 readers × 5 seeds (20/20 cells).

---

# 2. Flag provenance (mandatory — every quantitative result above and below)

Commits `0d482db` · `bf8ca9d` · `6aed207` on `agent/experiment-engineer/orgdiv-null-arms`, base local
`main @ eaecc91`. Worktree `../CHLU-null-arms`, **main venv reused** (protocol §4 w6 lesson):
`jax 0.9.0`, `equinox 0.13.4`, `optax`, `numpy 2.4.1`, float32. **Seeds 0–4** scored; **0–2** tuned.

| flag | value | note |
|---|---|---|
| `n_wells / f_subset / n_items / n_unseen` | 32 / 4 / 128 / 512 | FROZEN §(iii), unchanged |
| **`atoms_per_well`** | **32** (config) · axis **{12, 32, 64}** | cat-test deviation D2; covers the stale frozen row too (recon 1) |
| `addr_dim d` / **`payload_dim m`** | 4 / **8** | cat-test deviation D1 |
| `payload_radius` / `atom_payload_init_radius` | 1.0 / 1.0 | cat-test D3 / D4, applied identically to every arm |
| `ball_radius` / `launch_radius` / `query_sigma` | 2.0 / 0.6 / 0.15 | ⛔ never altered — the frozen launch |
| `n_particles P` | 4 | ⭐ the binding constraint (§3) |
| `s_measured` / `target_ds` / `depth_ratio` | 0.318 / 2.7 / 3.0 | anchors placed identically to the physics arm |
| `tol` | `0.25 × RMS‖y − ȳ_seen‖` = **0.478** | chance **3.906e-4** |
| φ | `build_phi(cfg, phi_seed=20260801)`, **576 B** | byte-compared across arms + seeds (test) |
| launch keys | `PRNGKey(2000+seed)` (SEEN) / `fold_in(·,1)` (unseen) | **the physics arm's own keys** |
| grid | 5 lr × 3 capacity × 3 seeds/arm; **584 configs**; steps 400 | prereg §4.3 |
| validation | 32 of 128 SEEN, **rule-4-valid vs every training row (100 % on all 5 seeds)** | §6 |
| selection statistic | val exact-set accuracy, tie-broken on val MSE | reader-independent, declared |
| γ | **not an axis** — no arm has a rollout ⇒ γ-independent by construction | declared, §7 NOT-RUN 1 |
| N5 divergences | **81 / 240** configs (momentum × lr corners) | recorded, cannot win selection (test) |

---

# 3. ⭐⭐ THE MECHANISM — why every arm reads 0, measured against the tightest control

The metric needs `y(x) = Σ_{j∈A(x)} v_j`, a sum over **`F = 4` distinct** wells. So the read must
*occupy* 4 distinct wells. It does not:

| statistic, unseen queries, 5 seeds (2 560 queries) | **measured** |
|---|---|
| distinct wells occupied by the `P = 4` particles — **raw launch geometry** | **2.202 ± 0.02** |
| … fraction of queries with **≥ F = 4** distinct wells | **0.050** |
| … occupancy precision (particles landing in a well of `A(x)`) | **0.4106** (reproduces cat-test's 0.4061) |
| … **exact-set occupancy** (occupied set **==** `A(x)`) | ⛔ **0.0000 / 2 560** |
| the same, after N3's fitted static-geometric assignment | 2.207 · 0.050 · 0.408 · **0.0000** |
| the same, after N2's fitted codebook | 2.224 · 0.053 · **0.111** · **0.0000** |
| the same, after the **physics settle** (cat-test's arm, rebuilt here) | ⛔ **1.70 distinct** |

> ⭐ **The `P = 4` occupancy read is capped at 0.05 before any organizer exists**, because ≥ 4
> distinct wells are reachable on only 5 % of queries — and measured exact-set occupancy is
> **0.0000**. No reader in the frozen class can repair this: `sum_linear`, `well_table`, `knn` and
> `mlp` all consume the same `(P, d+m)` object, which contains at most ~2.2 distinct payloads. **You
> cannot sum four vectors you never visited.**
> ⭐ **And the physics settle makes it worse, not better** (1.70 vs 2.20) — the same sign as
> cat-test §1.2's `−0.109` occupancy dividend, now with the mechanism named: the settle *merges*
> particles into shared basins.

## 3.1 ⛔ DECLARED OUT-OF-PROTOCOL diagnostic — does more fan-out fix it?
Changing `P` re-draws the launch offsets, so these are **not matched arms and never a score.**

| `P` | distinct wells | ≥ F distinct | exact-set occupancy | **combinatorial ceiling** |
|---|---|---|---|---|
| **4** (frozen) | 2.19 | 0.047 | 0.0000 | **0.288** |
| 8 | 2.74 | 0.201 | 0.0004 | 0.501 |
| 16 | 3.30 | 0.412 | 0.0016 | 0.720 |
| 32 | 3.88 | 0.607 | 0.0016 | 0.861 |
| 64 | 4.34 | 0.741 | 0.0016 | **0.938** |

⚠ **Read this carefully, because it says two different things.** More particles fixes the
*addressability* (2.2 → 4.3 distinct wells) and the *information* (ceiling 0.29 → 0.94), but
**nearest-well occupancy still never recovers the set** (0.0016 at `P = 64`): the extra particles
land in *wrong* wells at the same 0.40 precision. ⇒ **`P` alone is not the fix; the assignment rule
is also wrong.**

---

# 4. ⛔ THE OUT-OF-CLASS `φ`-DECODABILITY CEILING (declared, never scored as an arm — SP-1 precedent)

A combinatorial matched filter that enumerates all **35 960** set codes, returns the nearest, and
reads the written `v_j`:

| condition | registered prediction | **measured (5 seeds)** | verdict |
|---|---|---|---|
| **noiseless** (exact `φ(x)`) | ≥ 0.99 | **1.0000 ± 0.0000** | ✅ confirmed |
| **as-launched** (`P = 4` mean, `σ_q = 0.15`) | **0.20**, band [0.05, 0.60] | **0.2719 ± 0.0126** | ✅ **confirmed, 36 % low** |

⭐ **This is the number that turns "none clears" into an attributable finding.** `φ` at `d = 4` is
*not* the destroyer: **27.2 % of unseen queries are exactly decodable from the very launch points
every arm sees**, and 100 % from the noiseless code. What the arms lack is not information — it is a
read that can (i) address `F` distinct wells and (ii) decode a *continuous* code rather than
quantise it to one codebook entry per particle. ⚠ Quantisation is the destructive step: N1/N2/N3 all
replace the launch point with a codebook center, discarding exactly the continuous coordinate the
ceiling's decoder uses.

---

# 5. The oracle-imitation row (T5.2 rider (i)) and F5's input — 3 seeds

| quantity | registered | **measured** | verdict |
|---|---|---|---|
| N3 fitted **on the physics arm's own assignments**, agreement on unseen | 0.45, band [0.25, 0.70] | ⭐ **0.8888** (0.884 / 0.898 / 0.884) | ⛔ **REFUTED — far higher** |
| … the same, on SEEN | — | **0.9512** | — |
| … that arm's unseen score (native / 4 readers) | 0.001 | **0.0000** / ≤ 0.00195 | ✅ |
| N3 fitted on the **read objective**, agreement (F5's registered null) | 0.22, band [0.15, 0.30] | **0.2576** (0.273/0.229/0.271) | ✅ **in band; reproduces cat-test's 0.211–0.233 independently** |
| **F5 fires** (≥ 0.99)? | NO | **NO**, at either fitting | ✅ |

> ⭐ **The finding hiding in that first row:** the physics arm's assignment is **89 % reproducible by
> a static power diagram** *when you fit the diagram to the assignments directly* — but only **26 %**
> reproducible when you fit it to the read objective, which is what cat-test measured and what F5
> registers. **The gap between 0.89 and 0.26 is not "a structurally non-VQ channel"; it is an
> optimisation gap in fitting the diagram.** F5 still does not fire, but its "does not fire"
> now means *"the read objective is a bad way to recover the assignment"*, not *"the physics
> organizer is non-VQ"*. ⚠ Anyone quoting cat-test's F5 result should carry this sentence.

---

# 6. ⛔ THE SELECTION GUARD (the wave-invalidating condition)

`Q_unseen` is constructed in `seed_setup` and **read in exactly three places**: `stage_score`,
`stage_gridmax` and the two declared diagnostics. No fit, no hyperparameter and no arm ever sees it.
Selection runs on a **rule-4-valid** slice of SEEN — 32 rows held out such that
`|A_val ∩ A_train| ≤ F−2` against **every** retained training row (**achieved 32/32 on all 5 seeds**;
asserted in `tests/test_null_arms.py`). ⭐ This is stronger than the registered protocol asked for:
the naive seen-holdout is *not* the same problem as `Q_unseen`, because two written items may share
`F−1` wells, and selection on such a split would reward near-neighbour interpolation.
⚠ **Honest limitation:** with 32 validation rows the accuracy metric has grain `1/32 = 0.031`, so
**every** config scored `val_acc = 0.0000` and selection ran entirely on the registered MSE
tie-break. That is precisely why §1.1's grid-max over `Q_unseen` (grain `1/2560`) was run: the
verdict does not depend on the selection statistic having resolved anything.

---

# 7. Ledgers — matched capacity, bytes and per-query read compute (prereg §1)

| arm | learned-init rule | params | **param bytes** | state bytes | φ bytes | total | read mult-adds/query |
|---|---|---|---|---|---|---|---|
| **N1** (`a=32`, matched) | all PARAMETERS | 14 336 | **57 344** | 0 | 576 | 57 920 | 20 480 |
| N1 (`a=12` / `a=64`) | " | 5 376 / 28 672 | 21 504 / 114 688 | 0 | 576 | — | 7 680 / 40 960 |
| N2 | codebook = PARAMETERS | 384 | 1 536 | 0 | 576 | 2 112 | 640 |
| N3 | `(c,σ,b)` + payloads = PARAMETERS | 288 | 1 152 | 0 | 576 | 1 728 | 640 |
| N4 | none; raw rows = **STATE** | 0 | 0 | 6 144 | 576 | 6 720 | 1 536 |
| N5 | ⭐ `M_0` = **PARAMETERS**, `M_t−M_0` = **STATE** | 840 | 3 360 | 3 360 | 576 | 7 296 | 768 |
| *physics arm (context)* | — | 14 336 | 57 344 | 0 | 576 | 57 920 | **6.88e7** |

- **φ is byte-identical on every arm** (576 B, byte-compared; test `test_phi_is_frozen_…`), and the
  launch points are **bit-identical** to `multi_particle_read`'s (test
  `test_launch_points_are_bit_identical_…`).
- **N1's parameter count equals `FactoredStore.n_bytes()` exactly** (asserted, not asserted-by-eye).
- ⚠ **Per-query compute rule (prereg §1):** the physics read costs `P × 1200` Verlet steps over
  `n_atoms × dim` = `6.88e7` mult-adds — **3 360× N1's matched-capacity static read** (20 480). The rule ("if arms differ by > 2×, the cheaper arm
  additionally runs at the richer arm's budget") binds a *comparison*; after the re-scope there is no
  physics arm in the comparison, so the column is **reported**. ⛔ If a future wave revives the
  physics arm and compares, this row must be honoured — and note the direction: the nulls are the
  *cheap* side by three orders of magnitude and still tie.

---

# 8. PREREG SCORECARD (`.claude/outputs/orgdiv-null-arms/PREREG.md`, filed before `null_arms.py` existed)

| # | registered | measured | verdict |
|---|---|---|---|
| **HEADLINE** | no arm clears; `max_arm ≤ 0.03`, point **0.010** | **0.00117**, none clears | ✅✅ |
| N1 hard / soft | 0.006 / 0.008, band [0, 0.03] | **0.00039 / 0.00039** | ✅ (over-predicted 15×) |
| N2 | 0.006 | **0.00039** | ✅ |
| N3 | 0.006 | **0.00078** | ✅ |
| N4 | 0.000, band [0, 0.002] | **0.00078** | ◐ just outside, on 1 of 2 560 queries |
| N5 | 0.000, band [0, 0.005] | **0.00117** | ✅ in band |
| **L1** N1 fits its train split | ≥ 0.50 | **1.0000** | ✅✅ |
| **L2** N4 `k=1` on seen | ≥ 0.95 | **1.0000** | ✅ |
| **L3** shuffle-φ launder | ≤ chance + 0.005 | **≤ 0.00039** | ✅ |
| **L4** capacity flatness | ≤ 0.02 | **0.0000** | ✅ |
| **ceiling, noiseless** | ≥ 0.99 | **1.0000** | ✅ |
| **ceiling, as-launched** | 0.20, band [0.05, 0.60] | **0.2719 ± 0.0126** | ✅ |
| oracle-imitation agreement | 0.45, band [0.25, 0.70] | **0.8888** | ⛔ **REFUTED** |
| F5 read-objective agreement | 0.22, band [0.15, 0.30] | **0.2576** | ✅ |
| F5 fires? | NO | NO | ✅ |
| §7.1 "N1 clears" (`P = 0.10`) | — | did not | — |
| §7.2 "L1 fails ⇒ audit void" | — | did not | — |
| ⭐ §7.3 **"ceiling high + arms at zero ⇒ PROTOCOL refutation, not family refutation"** | committed in advance | **this is the outcome** | ⭐ **the registered branch fired** |

**Score: 13 ✅ · 1 ◐ · 1 ⛔.** The one refutation (oracle imitation) is a *finding*, §5.

---

# 9. How I verified (commands + observed output)

```
git worktree add ../CHLU-null-arms -b agent/experiment-engineer/orgdiv-null-arms main
/Users/user/Desktop/CHLU/.venv/bin/python -m ruff check chlu/ tests/   # All checks passed!
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-null-arms --quick        # all stages, 40 s
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-null-arms --out-dir .../results
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-null-arms --stages mechanism gridmax --out-dir .../results
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_null_arms.py -q   # 17 passed in 13.18s
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q
#   run 1: 1 failed, 1253 passed  (the x64 leak below)
#   run 2 after the fix: ⭐ 1255 passed, 24 warnings in 1331.23s (0:22:11) — zero failures
```
Artifacts: `.claude/outputs/orgdiv-null-arms/results/{stage_grid, stage_score, stage_gridmax,
stage_mechanism, stage_ceiling, stage_oracle, null_arms_summary}.json` (the grid records carry
**every** config × seed) · `null_arms_summary.png` (4 panels; ⚠ panel (a) is log-scaled and clamps
exact zeros to 1e-5) · `run.log`, `run2.log`, `pytest_full{,2}.log` · `PREREG.md`.

**Tests: `tests/test_null_arms.py` — 17 passed (new); full suite on my branch 1255 passed, 0
failed** (the C2W4-close baseline was 1190; cat-test took it to 1216 with 26; +17 are mine, the rest
arrived with the two merged C2W5 branches). ⚠ **Two failures happened and both are reported:**
1. `test_fit_code_payloads_recovers_…` failed on first write because at the small fixture
   `K = 12 < N_a = 16` makes the count design matrix rank-deficient — **a real identifiability
   condition, not a bug.** The test now asserts recovery at `K ≥ N_a` **and** asserts the
   under-determined case reproduces `y` *without* recovering the payloads. (The registered cell has
   `K = 128 > N_a = 32`.)
2. ⭐ `test_n5_declares_init_as_parameters…` passed alone and **failed in the full suite**: another
   test enables `jax_enable_x64` process-wide, `jax.random.normal` then handed N5 a float64 `M_0`
   against float32 data, and `lax.scan`'s carry types stopped matching. Fixed in `6a2bd1f` by
   pinning `_mlp_init` to float32 and taking the carry dtype from `jax.eval_shape`; **bit-identical
   under the default flag, so no measured number moves.** Regression test added. ⚠ *This is a
   general hazard for this repo: a module that mixes explicitly-cast float32 data with
   flag-following `jax.random` initialisers is silently x64-dependent, and a per-file test run will
   not catch it.*

---

# 10. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)

1. **The γ axis.** Not run, and not applicable: no null arm has a rollout, so all five are
   γ-independent **by construction**. Reported as one column, not three identical ones.
2. **A tuned physics arm.** Out of scope (Hub ruling 1 defers it). The physics numbers here are
   *quoted* from `orgdiv-cat-test` or rebuilt only to supply the oracle-imitation target; **no
   K-verdict was re-adjudicated.**
3. **`OD` / `OD_min`.** Not computed. There is no physics arm to swap against (the re-scope's
   premise), so the tier-ii dividend statistic does not exist in this report.
4. **The settle-deleted launder column.** Not needed per the addendum (no physics arm).
5. **Readers on the full 584-config grid.** `stage_gridmax` scores the arm's *own* read only
   (fitting 4 readers 2 920 times is unaffordable). On the selected configs, where all four readers
   do run, native and reader scores agree to the last digit — but this is an assumption on the grid.
6. **`P > 4` as an ARM.** The `P`-sweep re-draws the launch offsets and is declared out-of-protocol.
   A matched `P = 16` arm would need a new frozen φ and a new cat-test cell.
7. **N5 with a learned key projection / deeper memory**, and **N2 with a learned encoder.** Both
   would break the "identical launch protocol" match; not run.

---

# 11. Git footprint

- **Branch:** `agent/experiment-engineer/orgdiv-null-arms` (off local `main @ eaecc91`), in worktree
  `../CHLU-null-arms`. ⛔ Not pushed, no PR, no merge — left for review. `clu-dev` untouched.
- **Commits** (verified **from the MAIN repo** before finishing, protocol §3.2 — the wave-4 lesson):
  - `0d482db` `[experiment-engineer] the matched-capacity null arms N1-N5 (C2W5 organizer audit)`
  - `bf8ca9d` `[experiment-engineer] the null-arm audit harness + chlu exp-null-arms CLI hook`
  - `6aed207` `[experiment-engineer] tests: the null arms' matching obligations`
  - `6a2bd1f` `[experiment-engineer] pin N5's fast weights to float32 (full-suite x64 leak)`
  **The worktree is left in place** (not removed) for Hub review; working tree clean.
- **Files touched:** `chlu/core/null_arms.py` (new, 871 L) · `chlu/experiments/exp_null_arms.py`
  (new, 847 L) · `tests/test_null_arms.py` (new, 280 L) · `chlu/cli/experiment_cmd.py` (+46/−0).
  `git diff --stat main..HEAD` = **4 files changed, 2 044 insertions(+), 0 deletions**.
- **Rebase:** onto local `main` (⚠ **not** `origin/main`) — **no-op, base unmoved**.
- **Concurrent worktree** `../CHLU-lane-parallel-controller` exists; **zero file overlap** (verified
  by diffing its branch against `main`). No conflicts; no foreign uncommitted work was present.

---

# 12. Open questions / follow-ups / risks

1. ⭐ **The read protocol is the named, measured target for any revival — not the write.** Two
   concrete requirements fall straight out of §3: the read must **address `F` distinct wells**
   (currently 2.2 of 4, and the settle *lowers* it to 1.7), and it must **not quantise away the
   continuous launch coordinate** (that is what the 0.272 ceiling decodes). ⚠ Neither is a physics
   question; both are protocol questions, and both bind the physics arm exactly as hard as they bind
   my nulls.
2. **The 0.05 structural cap should be a pre-condition, not a discovery.** `P(≥ F distinct wells
   occupied) = 0.050` is computable from the launch geometry alone, before any store is written, in
   seconds. It belongs beside K1–K5 as a **K0** for any future cat-test cell.
3. **Is the cat-test's `d = 4` the real culprit, or the `P = 4`?** §3.1 says `P` fixes addressability
   but not precision (0.40 at every `P`); cat-test §7.2 says `d = 32` collapses occupancy to 0.188.
   The joint `(d, P, launch_radius)` sweep is unrun and is the cheapest next instrument in the
   program — it needs no store at all.
4. **Risk to how this gets quoted.** The headline "no matched-capacity organizer clears" is *true*
   and is the acceptance criterion, but standing alone it invites the family-refutation reading that
   §4 refutes. ⛔ It should never be quoted without the 0.272 ceiling in the same sentence.
5. **N3's in-sample fit (0.08) is the weakest arm-side number here.** It is capacity-starved by
   construction (`N_a = 32` units). If a future wave wants F5's null to be strong on its own terms,
   it needs a fitting budget that reaches a real optimum — the oracle-imitation form got to 0.95
   agreement, so the capacity is not the whole story.

---

## Proposed handover updates (for the Hub)

- **§3 config / CLI — NEW:** `chlu exp-null-arms [--stages grid score gridmax mechanism ceiling
  oracle] [--arms N1..N5] [--quick]`; new module `chlu/core/null_arms.py` with `NullArmGrid`
  (registered tuning budget, lives next to its code per the `CatTestConfig` precedent, **not** in
  `chlu/config.py`).
- **§7 Known Issues — NEW (open, and it is a MEASUREMENT constraint, not a bug):** *the `P`-particle
  occupancy read cannot express an `F`-term sum at `P = F`.* Measured: `P = 4` particles occupy
  **2.20** distinct wells; `≥ F` distinct wells on **5.0 %** of queries; exact-set occupancy
  **0.0000 / 2 560**. Any experiment whose target is a sum over `F` wells must check this **before**
  writing a store (proposed **K0**).
- **§7 Known Issues — NEW (open):** `FROZEN-interfaces.md`'s matched-capacity ledger row (`384·14 =
  21 504 B`, `a = 12`, ratio 3.20×) contradicts the cat-test's registered deviation D2 (`a = 32`,
  57 344 B, 9.67×); its reader parameter counts (16, 88) contradict the shipped code (72, 92).
  Reconciliation items 1–2.
- **Registry/doctrine candidates:** (i) ⭐ *an organizer audit needs an in-class fit anchor* — N1 at
  **1.0000 train / 0.0000 held-out** is what makes "no arm clears" a statement about the problem
  rather than about the optimiser; make the L1-style anchor mandatory for any "nothing works"
  verdict. (ii) ⭐ *quote the decodability ceiling beside every null-arm null* — measuring what the
  read-in still carries (here 0.272) is what separates "the arms are weak", "the family is dead" and
  "the read protocol is dead"; it cost 60 s. (iii) *a seen-validation split must inherit the family's
  own held-out rule* (here rule 4), or hyperparameter selection is run on an easier problem than the
  one being scored.
- **`PREREG-TierII.md` errata:** §4.3's "`null*` = max over the entire registered grid" is now
  **computed** for the first time (584 configs × 5 seeds ⇒ 0.00117); §3.5's F5 discussion should
  carry §5's 0.89-vs-0.26 distinction between *imitability* and *read-objective recoverability*.
- **`orgdiv-cat-test` §13 open question 1: CLOSE** — answered at 2.20 (launch) / 1.70 (settled).
