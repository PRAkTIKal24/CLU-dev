# gen-navigation — experiment-engineer report

**Task + acceptance criterion:** traverse from one store mode to another, decode at every step, read every
intermediate image with the **gated external classifier**, and say whether the factor readings change smoothly and
monotonically — with **straight-line interpolation between the endpoint READ vectors (store bypassed)** and a
**deleted store** run on *every* number. **Pre-registered summary statistic `NAV`** = fraction of qualifying pairs
where the store path traverses and neither control does.
**Status: done.** ⛔ **VERDICT: `NAV = 0.0000 ± 0.0000`, 3/3 seeds. The store's landscape does NOT out-traverse
straight-line interpolation. ⭐ The task's own named falsifier FIRED.** Per the task and R29, a clean negative is a
result and routes to the Head, ⛔ not to re-tuning.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary, first 10 lines)
> 1. ⭐⭐ **The navigation dial's answer is NO, and the laundering control is what says so.** On the *fairest*
>    reading for the store (GRID, endpoints reading correctly, the stability clause dropped) the **linear control
>    beats the store path 0.697 ± 0.144 vs 0.550 ± 0.147, Δ = −0.162 ± 0.027, sign-consistent 3/3.** ⛔ Write it
>    as "interpolation does it at least as well", never as "navigation works". (§3)
> 2. ⭐⭐ **R31's mechanism REPLICATES on an independent instrument and GENERALISES beyond the corner.** The
>    external classifier (⛔ not the probe, ⛔ not the same harness) reads the 30-bit read's endpoint fidelity as
>    `scale` **0.021** / `orientation` **0.158** in the corner — **below chance** — reproducing
>    `gen-commitment-collapse`'s probe numbers (0.019 / 0.194) to within noise. ⭐ **NEW:** on the *spread* grid it
>    is `scale` **0.229** (chance 0.125) and `orientation` **0.236** (chance 0.067) — so **the read drops scale and
>    orientation everywhere, not only inside the corner.** R31's "the corner collapses" is about *distinctness*;
>    this is about *fidelity*, and the fidelity loss is global. (§5)
> 3. ⚠ **`visits_intermediate` is NOT evidence on its own, and a spoke could have shipped it as the headline.**
>    The `scramble`-deleted store — content destroyed — visits an intermediate factor value on **0.483 ± 0.012** of
>    GRID pairs and is weakly monotone on 0.350. ⛔ Any future "smooth interpolation in latent space" claim needs
>    this control. (§4)
> 4. ⚠ **The store path is a STAIRCASE, not a manifold walk** — 5.92 of 8 steps are exactly zero-length,
>    `jump_ratio` 5.63 ± 0.17 against the control's 1.55 ± 0.03, and 3.08 of 9 points carry a distinct code.
>    ⭐ It is nonetheless **more direct** in image space (`path_efficiency` 0.841 vs 0.747). Both facts belong in
>    any write-up; quoting either alone is misleading. (§4)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *navigation* — whether the store's landscape supports meaningful movement **between** wells/modes.
- **Laundering control:** ⭐ **straight-line interpolation between the two endpoint READ vectors, store bypassed at
  every interior point**, on **every** number (3 pools×factors × 4 arms × 3 seeds; 88 pairs/seed). Second control:
  the identical path read through a **`mean`-deleted** store (the fair deletion) and a **`scramble`-deleted** one.
- **Falsifies:** linear interpolation traverses as well or better ⇒ the store's landscape adds nothing.
  ⛔ **IT FIRED.**
- **Does NOT falsify** (declared in `PREREG.md` §DIAL before measurement): (a) losing to the linear control *on
  level* while that control uses a wider per-point channel; (b) failing **inside Montero's corner** — R30/R31
  already established the corner collapses, so a corner-only failure is a known confirmation; (c) the
  `mean`-deleted arm rendering a constant image (arithmetic).

## ⭐ Pre-registration
`PREREG.md` filed **21:30:27**, before `chlu/core/gen_navigate.py` (21:33), `exp_gen_navigate.py` (21:34), the
first smoke (21:34) and the first production output (21:38) — mtime-verifiable, in that order.
**Addendum 1 at 21:37**, after seed 0's first production run, declaring (a) a **prereg-compliance fix**
(`shape` is categorical and must be switch-only; the first build wrongly let it into the monotone denominator —
⭐ its reading was **0.000 either way**, recorded so the exclusion cannot be a way of dropping an inconvenient row)
and (b) **three post-hoc attribution diagnostics** with their own predictions (P9–P11).
⛔ **`TRAVERSES` and `NAV` are measured exactly as filed at 21:30; nothing in §3/§4 was amended.**

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | **`c46dfd8`** (branch tip; every number below produced at this tree) |
| branch / base | `agent/experiment-engineer/gen-navigation` off **`main @ be27713`** (unmoved; `git rebase main` = no-op) |
| worktree / venv | `../CHLU-gen-nav`, **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0**, equinox 0.13.4, numpy 2.4.1, `JAX_PLATFORMS=cpu` |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero) |
| seeds | **0, 1, 2** — all three complete, ⛔ none NOT RUN |
| **store** | ⭐ `gen-read-compose`'s per-seed checkpoints, **loaded, never retrained** — `3e3424a1ea40…` / `76827ad534ee…` / `8449e6057dac…`; `K=5`, `M=64`, `D=2048`, **30 bits/read** |
| **decoder** | `decoder_source="winner"`, recipe **`v1_capacity_schedule`** asserted by **name and hash** per seed: `01ef91f75eea…` / `001011bc603a…` / `2d7eea75add7…` (⛔ `dec_recipe` still defaults to `"baseline"`, the recipe that FAILED the gate) |
| **instrument** | ⭐ the **GATED external classifier**, `classifier_gate.json = PASS` asserted before any number; hashes `c8741ac417d8…` / `aa69b436989b…` / `ad16c4e53273…`, each asserted against its own `classifier_manifest.json`. ⛔ **Never the linear probe** (it shares the decoder's lineage) |
| address book / grouping | `gen-commitment-collapse`'s own per-seed `commit_cache_s*.npz` (cache **HIT** on all 3 seeds ⇒ ⛔ no re-encode, no re-derived grouping) — so the endpoints are the **same committed codes R31 reasoned about** |
| traversal | `n_steps = 8` ⇒ **9 points**; arms `store_snap` · `linear` · `deleted_mean` · `deleted_scramble`; `scramble_seed = 20260913 + seed` |
| endpoint rule | pools `CORNER` (72 held-out B requests) + `GRID` (lines through the 512 seen `eval_a` anchors; **0 of 512 dropped as in-corner**); `n_pairs_per_factor = 8`, `min_delta = 2`, draw `default_rng(seed·7919 + 20260913 + 101·pool + factor)`; `categorical_factors = ["shape"]` |
| success rule | `stability_floor = 0.9`; `design_constant_bits = 30.0`; `eval_batch = 256` |
| rows / pairs | **88 pairs per seed** (CORNER 40 + GRID 48), of which **56 qualify** for the monotone statistic (CORNER 16 + GRID 40). ⛔ `CORNER/shape` = **NOT APPLICABLE** (`n_values_realised = 1`, 0 single-factor pairs), reported as such, never as a zero |
| images scored | 88 pairs × 9 points × 4 arms = **3,168 renders per seed**, 9,504 total, each read by the external classifier |
| ⛔ NOT used | `soft_read`, `address_codes`, any feature tensor on any arm, `DreamLandscape`, `dream_rollout`, `clamp_from_posterior`, the linear probe |
| wall clock | **43.8 / 40.2 / 40.0 s** per seed, CPU only |
| ⛔ CSF3 | not used; not needed (the whole run is 2 minutes) |

---

## 1. ⭐⭐ THE HEADLINE — the pre-registered statistic, and the control that killed it

| pool | qualifying pairs/seed | n_values_realised | **NAV** | `store_snap` | **`linear`** (control 1) | `deleted_mean` (control 2) | `deleted_scramble` |
|---|---|---|---|---|---|---|---|
| **CORNER** | 16 | 1, 2, 3 | **0.0000 ± 0.0000** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 |
| **GRID** | 40 | 4, 8, 10, 15 | **0.0000 ± 0.0000** | **0.0083 ± 0.0118** | **0.0083 ± 0.0118** | 0.0000 | 0.0000 |
| **ALL** | 56 | 1–15 | **0.0000 ± 0.0000** | 0.0060 ± 0.0084 | 0.0060 ± 0.0084 | 0.0000 | 0.0000 |

⭐ **The store arm and the laundering control are not merely close — on the pre-registered statistic they are
identical to the last digit, on every pool and every seed.** One pair in 120 traverses — **seed 1, `GRID/orientation`, record 82** — and it is the **same record** for both
arms (verified from `records_s1.json`).

⭐ **PREREG P5 (`NAV ≤ 0.20`, point 0.07) HELD** — and the prereg says so in advance: *"measuring `NAV ≈ 0` is NOT
a surprise and will not be dressed as one."*

## 2. ⭐⭐ WHY it is zero — the four clauses decomposed (the number a re-tune would have been aimed at)

Fractions over qualifying pairs; a pair needs **all four**. Full table: `tables.md` §2.

| pool | arm | endpoints_read_ok | monotone | visits_intermediate | off-target stability (pre-reg) | …≥ 0.9 | **TRAVERSES** |
|---|---|---|---|---|---|---|---|
| CORNER | `store_snap` | **0.000** | 0.979 | **0.021** | 0.681 ± 0.024 | 0.000 | **0.0000** |
| CORNER | `linear` | **0.000** | 0.979 | **0.021** | 0.681 ± 0.024 | 0.000 | **0.0000** |
| GRID | `store_snap` | **0.517 ± 0.024** | 0.850 | 0.683 ± 0.066 | 0.625 ± 0.020 | 0.050 | **0.0083** |
| GRID | `linear` | **0.517 ± 0.024** | 0.833 | **0.775 ± 0.054** | 0.638 ± 0.015 | 0.050 | **0.0083** |
| GRID | `deleted_mean` | 0.000 | 1.000 | 0.350 | 0.108 | 0.000 | **0.0000** |
| GRID | `deleted_scramble` | 0.017 | 0.350 | **0.483** | 0.131 | 0.000 | **0.0000** |

⭐⭐ **The binding clause is not the geometry — it is `endpoints_read_ok`.** In the corner it is **exactly 0.000 on
all three seeds**: the decoded endpoint image *never* shows the requested `scale`/`orientation`. ⛔ **You cannot
measure navigation between two points that are not the points you asked for**, and that failure is inherited from
the READ (§5), not produced by the traversal.

## 3. ⭐⭐ THE FAIREST POSSIBLE READING FOR THE STORE — and the control still wins

The pre-registered conjunction is relaxed step by step (⛔ post-hoc, declared; `tables.md` §8):

| pool | arm | (a) **pre-registered** | (b) drop the stability clause | (c) **(b) ∧ condition on the endpoints reading correctly** | n in (c)/seed |
|---|---|---|---|---|---|
| GRID | `store_snap` | 0.0083 ± 0.0118 | 0.283 ± 0.072 | **0.550 ± 0.147** | 20.7 |
| GRID | **`linear`** | 0.0083 ± 0.0118 | **0.358 ± 0.066** | ⭐ **0.697 ± 0.144** | 20.7 |
| GRID | `deleted_scramble` | 0.0000 | 0.000 | 0.000 | 0.7 |
| CORNER | all arms | 0.0000 | 0.000 | ⛔ **undefined — 0 pairs have readable endpoints** | 0.0 |

**Head-to-head on that conditional set (GRID, `tables.md` §8b):**

| metric | `store_snap` | `linear` | **Δ (store − linear)** |
|---|---|---|---|
| `visits_intermediate` | 0.550 ± 0.147 | 0.712 ± 0.134 | ⛔ **−0.162 ± 0.027** (3/3 seeds) |
| `monotone` | 0.726 ± 0.061 | 0.759 ± 0.105 | −0.033 ± 0.047 |
| `off_target_constancy` | 0.828 ± 0.030 | 0.800 ± 0.040 | ⭐ **+0.028 ± 0.016** (3/3 seeds) |

⛔ **On the single most generous reading available to it, the store's discrete landscape traverses WORSE than a
straight line drawn between its own two endpoint reads.** ⭐ The one axis where it leads — perturbing the
non-target factors slightly less — is **+0.028**, an order of magnitude below what I pre-registered.

## 4. Smoothness, and the control that makes "smooth" meaningless on its own

| pool | arm | mean step RMSE | `jump_ratio` (1 = even) | `path_efficiency` (1 = direct) | distinct codes on path (of 9) | zero-length steps (of 8) |
|---|---|---|---|---|---|---|
| GRID | `store_snap` | 0.032 ± 0.001 | **5.63 ± 0.17** | ⭐ **0.841 ± 0.007** | 3.08 ± 0.11 | **5.92 ± 0.11** |
| GRID | `linear` | 0.036 ± 0.002 | **1.55 ± 0.03** | 0.747 ± 0.010 | — (continuous) | 0.17 ± 0.24 |
| GRID | `deleted_mean` | **0.000** | — (undefined) | — | 3.08 | 8.00 |
| GRID | `deleted_scramble` | 0.067 ± 0.007 | 5.06 ± 0.16 | 0.665 ± 0.022 | 3.08 | 5.92 |

- ⭐ **The store path is a staircase**: it sits still for ~6 of 8 steps and then jumps. That is the honest picture
  of a 64-mode-per-well codebook — ⛔ it is **not** a smooth manifold walk and should never be drawn as one.
- ⭐ **But it is more *direct*** (0.841 vs 0.747): the linear interpolant passes through off-manifold blends the
  decoder renders far from either endpoint, so it wastes image-space distance. **Both facts, or neither.**
- ⛔ **The laundering result of this section:** `deleted_scramble` — the store's content *destroyed* — still
  "visits an intermediate value" on **0.483** of GRID pairs and is weakly monotone on 0.350. ⚠ **A latent-space
  interpolation demo that reports only "it moves smoothly through intermediate values" is reporting something a
  destroyed store also does.**

## 5. ⭐⭐ THE MECHANISM — an independent replication of R31 that also extends it

Per-factor **endpoint fidelity** of the 30-bit read, read by the **external** classifier (`tables.md` §9):

| pool | factor | **n_values_realised** | chance | fidelity (3 seeds) | |
|---|---|---|---|---|---|
| CORNER | floor_hue | 2 | 0.500 | **0.996 ± 0.006** | above chance |
| CORNER | wall_hue | 2 | 0.500 | **1.000 ± 0.000** | above chance |
| CORNER | object_hue | 2 | 0.500 | 0.692 ± 0.224 | above chance |
| CORNER | **scale** | 3 | 0.333 | ⛔ **0.021 ± 0.029** | **BELOW chance** |
| CORNER | **orientation** | 3 | 0.333 | ⛔ **0.158 ± 0.133** | **BELOW chance** |
| CORNER | shape | **1** | — | 0.688 ± 0.237 | ⚠ **n/a — one realised value ⇒ a detection rate, not discrimination** |
| GRID | floor_hue | 10 | 0.100 | 0.997 ± 0.005 | above chance |
| GRID | wall_hue | 10 | 0.100 | 0.990 ± 0.009 | above chance |
| GRID | object_hue | 10 | 0.100 | 0.701 ± 0.038 | above chance |
| GRID | **scale** | 8 | 0.125 | ⚠ **0.229 ± 0.029** | above chance, far below 1 |
| GRID | shape | 4 | 0.250 | 0.583 ± 0.074 | above chance |
| GRID | **orientation** | 15 | 0.067 | ⚠ **0.236 ± 0.081** | above chance, far below 1 |

1. ⭐ **Cross-instrument replication.** `gen-commitment-collapse` §7 measured, with the **probe**, on B:
   floor 0.991 · wall 1.000 · object 0.671 · **scale 0.019** · **orientation 0.194**. This spoke measures, with the
   **external classifier**, on a different harness: 0.996 · 1.000 · 0.692 · **0.021** · **0.158**. ⛔ The collapse
   mechanism is **not** an artefact of measuring the decoder with its own family.
2. ⭐⭐ **And it extends R31.** On the *spread* grid — where R31 reports the code **does** move with `scale` (flip
   0.71) and `orientation` (flip 0.83) — the rendered value is right only **0.229** / **0.236** of the time.
   ⇒ ⛔ **"the code moves with the factor" and "the read carries the factor" are different statements, and only
   the first is true outside the corner.** R31's "the corner collapses, not the hold-out" is about *distinctness*;
   this fidelity loss is **global**.
3. ⭐ **In the corner, `scale` fidelity is BELOW chance (0.021 vs 0.333)** — the read does not merely lose the
   value, it systematically renders a *different* one (the cell's modal scale). ⛔ A future spoke must not read
   "0.02" as "noise".

## 6. Endpoint-code geometry — why the corner has nothing to traverse

| | CORNER/scale | CORNER/orientation | GRID (all factors) |
|---|---|---|---|
| **endpoint codes identical** | **0.750 ± 0.000** | **0.792 ± 0.295** | 0.021 ± 0.030 |
| interior codes off both endpoints | 0.04 | **0.00** | 0.5–2.8 |

⭐ **PREREG P2 HELD, almost exactly:** predicted 0.78 (derived as `1 − flip_rate` from R31's `scale 0.20 /
orientation 0.13`, shaded for the extreme `|Δ| = 2` draws), measured **0.771** pooled. ⭐ **PREREG P1 HELD:**
CORNER traverse rate ≤ 0.10, measured **0.000**.
⇒ **In the corner, three quarters of the pre-registered endpoint pairs receive the *same* code, so the "traversal"
is a single point.** ⛔ This is the commitment collapse inherited, exactly as the task's second falsifier named.

## 7. ⭐ The pre-registered prediction scorecard (⛔ failures reported as findings)

| # | prediction | measured | verdict |
|---|---|---|---|
| **P1** | CORNER store traverse ≤ 0.10 | **0.000** | ⭐ **HELD** |
| **P2** | identical endpoint codes ≈ 0.78 [0.55, 0.92] | **0.771** | ⭐ **HELD** (point estimate to 0.01) |
| **P3** | GRID store traverse 0.15 [0.05, 0.35] | **0.008** | ⛔ **FALSIFIED** — far below; the store traverses *less* than I expected |
| **P4** | GRID linear traverse 0.25 [0.10, 0.45] | **0.008** | ⛔ **FALSIFIED** — the *statistic* is harsher than I expected for **both** arms, which is why §3's relaxations exist |
| **P5** | **`NAV` ≤ 0.20** (point 0.07) | **0.0000** | ⭐ **HELD** |
| **P6** | `deleted_mean` traverse = **exactly** 0 | **0.0000**, all 8 steps zero-length, constancy 1.000 | ⭐ **HELD** (it was a plumbing check, and it passed) |
| **P7** | ⭐ store − linear off-target **stability** = +0.15 [+0.03, +0.35] | **−0.013** | ⛔ **FALSIFIED, wrong sign.** The store does **not** perturb the non-target factors less on the pre-registered measure |
| **P8** | CORNER stability ≥ GRID's (declared in advance as a known confirmation) | 0.681 vs 0.625 | ⭐ held — ⛔ and **not** reportable as a win, exactly as pre-declared |
| **P9** | constancy − stability ≥ +0.15 (the conflation diagnosis) | **+0.236** | ⭐ **HELD** — most "instability" is a *constant misread*, not drift |
| **P10** | store − linear off-target **constancy** ≥ +0.10 | **+0.020** on the qualifying pairs (0.861 vs 0.841); **+0.028** on §3's conditional set; sign-consistent **3/3** on both | ⛔ **FALSIFIED** — right sign, an order of magnitude too small |
| **P11** | GRID endpoint fidelity ≈ 0.75 [0.60, 0.90] | **0.625** | ⭐ **HELD**, at the bottom of the interval |

⭐ **P7 and P10 are the informative failures.** I pre-registered that the store's *discrete* path would perturb the
rest of the image less than an interpolant that moves all 2,048 coordinates. It does not — +0.028 at best, and
**worse** on the pre-registered measure. ⛔ There is no surviving axis on which the store's landscape beats the
straight line.

## 8. ⛔ The guarantee — the realised channel at every intermediate point

| arm | per-point width | vs R20's 30-bit design constant | informative bits | path information |
|---|---|---|---|---|
| `store_snap` | **30.0 bits** | ⭐ **within** | 30.0 | 63.17 bits |
| `linear` | 65,536 bits (a real vector) | ⛔ EXCEEDS | 65,536 | **63.17 bits** |
| `deleted_mean` / `deleted_scramble` | 30.0 bits | ⭐ within | **0.0** | 63.17 bits |

- ⭐ **Every intermediate point of every store arm is one `K`-tuple = `5·log2(64)` = 30 bits**, R20's design
  constant, unchanged. ⛔ **No widening anywhere on the claim path.**
- ⚠ **Two accountings are reported deliberately.** The linear control's *per-point object* is a real vector, but
  its interpolant is a deterministic function of the two endpoint codes and `t`, so the **path** it can carry is
  `2·K·log2 M + log2(S+1) = 63.17` bits — identical to the store arm's. ⛔ **The control is a wider pipe, not a
  bypass, and it must not be dismissed on channel-width grounds.**
- `assert_no_feature_tensor` passes on every seed: **by signature** (`snap_path_codes(store, code_a, code_b,
  n_steps)`, `read_from_codes(store, codes)`), **by name** (⛔ `soft_read` *and* `address_codes` — both
  feature-taking — are not imported), **by provenance** (the path re-derives bit-identically) and **numerically**
  (bit-identical with a decoy feature tensor present). `assert_request_only` is re-run on the endpoint path.
  ⭐ Endpoints are **exact**: `c(0) = code_a`, `c(1) = code_b`, asserted.

## 9. How I verified (commands + observed output)

```
# targeted tests, main venv, worktree, ⭐ WITH CHLU_GEN_DATA_ROOT so the real-data path executes:
CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen … pytest -q --no-cov -rs \
  tests/test_gen_navigate.py tests/test_config.py tests/test_gen_compose.py \
  tests/test_gen_classifier.py tests/test_gen_store.py tests/test_gen_data_splits.py
  -> 157 passed, 0 skipped, 0 failed, 1 warning in 170.23s   (⭐ 34 of them new; ⛔ NO full suite started)
ruff check chlu/core/gen_navigate.py chlu/experiments/exp_gen_navigate.py chlu/config.py tests/… -> All checks passed!
# the run, per seed:
  commit cache HIT commit_cache_s0.npz (⭐ no feature gather, no grouping re-derive)
  ⭐ store loaded … (hash 3e3424a1ea40017e…, K=5, M=64, 30 bits/read)
  ⭐ decoder: winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b… (hash + recipe asserted)
  ⭐ external classifier: gate=PASS hash=c8741ac417d88ebf… (⛔ not the probe)
  ⭐ traversal path asserted feature-free (4 checks) · endpoints exact=True · 30 bits/read
    channel [store_snap]: 30.0 bits at EVERY intermediate point ⭐ within
    channel [linear]:  65536.0 bits at EVERY intermediate point ⛔ EXCEEDS
  ⚠ CORNER/floor_hue (n_values_realised=2, n_values_realised < 3 ⇒ SWITCH-ONLY, ⛔ excluded from the headline)
  ⛔ CORNER/shape: NOT APPLICABLE — n_values_realised=1, 0 single-factor pairs exist
  seed 0 done: 88 pairs, NAV=0.0000 ⇒ ⛔ NEITHER ARM TRAVERSES — no navigable structure found
```
⚠ **Two of my own failures, recorded because a run that never failed was never tested.**
1. The first smoke died with `FileNotFoundError: …/gen/3dshapes.h5` — I had forwarded `$CHLU_GEN_DATA_ROOT` into
   `config.gen_data.data_root`, but `shapes3d_root(root)` treats an explicit root as the **dataset** dir while the
   env var is its **parent**. ⭐ This is the exact §7 item `gen-external-classifier` filed; it cost me one run and
   is now a comment at the site.
2. Seed 0's first production run returned `NAV = 0` with **every** arm at zero and the record schema could not say
   *which* clause failed. I stopped the run, filed Addendum 1, added the three attribution diagnostics and re-ran
   all three seeds from scratch (2 minutes). ⛔ The pre-registered statistic was not touched.

## 10. ⛔ Declared NOT RUN
- **dSprites / the wide condition / the `cls` pretext — NOT RUN.** One dataset, one condition, one pretext.
- **A traversal in *request* space** (interpolating the request, not the code) — ⛔ not run, ⛔ not claimed. It is a
  different experiment and would need its own prereg.
- **Uniform-weight or `topk` endpoints** — the endpoints are the **shipped `argmax`** commitment only; R31 showed
  `topk k=8` reaches 70 distinct codes at 160 bits, and traversal between *those* is a declared-channel-change
  experiment, not this one.
- ⛔ **No kill-rule verdict, no C set, no refusal number, no cell table** — out of scope.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-navigation`, worktree `../CHLU-gen-nav`, base **`main @ be27713`**.
  Rebase onto `main`: **no-op** (`main` has not moved; ⛔ I did not rebase onto the stale `origin/main`, §7.21).
- **Commit (1):** **`c46dfd8`** — verified **from the MAIN repo**:
  `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-navigation` lists it.
- **Files (4, all inside my declared ownership):** `chlu/core/gen_navigate.py` (new, +572) ·
  `chlu/experiments/exp_gen_navigate.py` (new, +738) · `tests/test_gen_navigate.py` (new, +426) ·
  `chlu/config.py` (**+100 / −0**, ⛔ **0 deletions**, ONE block at EOF marked `BEGIN/END gen-navigation`,
  **no `CHLUConfig` field, no `load_config`/`save_config` site** — reached through `gen_navigate_config()` so the
  tail anchor stays one contiguous liftable region).
- ⛔ **Untouched:** `gen_store.py`, `gen_compose.py`, `gen_bridge.py`, `gen_certificate.py`, `gen_request.py`,
  `gen_dream.py`, `chlu/eval/`, `chlu/data/`, `scripts/csf3/`, `chlu/cli/`.
- ⚠ **`chlu/config.py` merge note:** `gen-cell-harness` also appends at EOF this wave. If the anchor conflicts,
  ⛔ do **not** merge hunk-by-hunk: take the peer's whole file, re-apply my delimited block, assert **0 deletions**.
- ⛔ **No CLI hook added.** `chlu/cli/experiment_cmd.py` is not in my ownership list and is a high-collision shared
  file; the module ships its own `argparse main()` like `exp_gen_compose`/`exp_gen_classifier`. Hub decision.
- **The worktree is left in place** for Hub review. **No push, no PR, no merge. No conflicts.**

## Artefacts (all under `.claude/outputs/gen-navigation/`)
`PREREG.md` (+ the dated Addendum 1) · `tables.md` + **`tables.py`** (⭐ every table above is rebuilt **from disk**
by that script; no model is loaded, so a table cannot disagree with the artefacts) · `navigation_headline.png` ·
`runs/shapes3d_recon/summary.json` · `runs/shapes3d_recon/navigation.png` ·
`runs/shapes3d_recon/seed_{0,1,2}/{navigate_manifest_s*.json, records_s*.json}` (⭐ `records_*` carries the **full
`(9, 6)` classifier reading matrix** for every pair × arm, so every statistic here is recomputable) · `run.log`.

## Open questions / follow-ups / risks
1. ⭐⭐ **The actionable finding, and it is upstream of navigation — the same place `gen-commitment-collapse`
   pointed.** Navigation cannot be measured above the read's **endpoint fidelity** (0.59 CORNER / 0.63 GRID). ⛔ No
   traversal rule can fix that; the wells do not carry `scale`/`orientation` well enough to render them **anywhere
   in the factor space**, not just in the corner (§5.2). **This is a store-fitting problem, not a path problem.**
2. ⚠ **The negative is about THIS traversal (`snap`) on THIS store.** Two alternatives are unrun and unclaimed:
   (a) a **graph** walk over modes (nearest-neighbour hops in each well's codebook) rather than a snapped segment;
   (b) traversal at `topk k>1` endpoints, which is a **declared channel change**. ⛔ Neither is a rescue attempt —
   they are different experiments and each needs its own prereg.
3. ⚠ **`GRID/orientation` is the one factor where a single pair traversed** (0.042 ± 0.059, i.e. 1 pair in 24,
   one seed). ⛔ Do not quote it; it is one pair and the linear control traversed the same one.
4. ⚠ **The `endpoints_read_ok` conditioning in §3 selects on the outcome of one arm.** I used each arm's **own**
   endpoint success, which is the most generous version for the store; conditioning on the store's endpoints only
   would change the linear column. Both arms have identical endpoint rates (0.517), so the comparison is fair, but
   ⛔ it is a post-hoc conditional and is labelled as one.
5. ⚠ **`CORNER` has zero pairs with readable endpoints**, so every CORNER cell in §3 is `undefined`, ⛔ not zero.
6. ⚠ **`deleted_mean`'s `off_target_constancy` = 1.000 is arithmetic** (a constant image is perfectly constant).
   ⛔ It must never be read as the deletion control "doing well" on stability.

## Proposed handover updates (for the Hub)
- **§7 NEW (result, binding):** ⛔ **The navigation dial's answer is NO.** `NAV = 0.0000 ± 0.0000` (3/3 seeds), and
  on the fairest conditional reading **straight-line interpolation between the endpoint reads beats the store's
  snapped path by 0.162 ± 0.027, 3/3 seeds.** ⭐ The pre-registered laundering control fired. Store-landscape
  traversal is **not** a claim the programme can make on this store.
- **§7 NEW (mechanism, extends R31):** ⛔ **"The read drops `scale` and `orientation` INSIDE THE CORNER" is too
  weak.** Measured on the spread grid with the external classifier: `scale` **0.229** (chance 0.125),
  `orientation` **0.236** (chance 0.067) — **the fidelity loss is global**; only the *distinctness* collapse is
  corner-specific. In the corner `scale` fidelity is **below chance** (0.021 vs 0.333) — a systematic wrong value,
  not noise.
- **§7 NEW (instrument, standing):** ⭐ **R31's collapse mechanism replicates cross-instrument** — external
  classifier 0.996/1.000/0.692/0.021/0.158 vs the probe's 0.991/1.000/0.671/0.019/0.194. ⛔ It is not an artefact
  of measuring the decoder with its own family.
- **§7 NEW (standing control, fifth strike candidate):** ⛔ **"It moves smoothly through intermediate values" is
  not evidence.** A `scramble`-deleted store — content destroyed — visits an intermediate factor value on **0.483**
  of pairs and is weakly monotone on 0.350. Any future latent-interpolation claim carries this control.
- **§7 NEW (trap):** ⚠ **A weakly-monotone path over a 2-valued factor is a SWITCH, not a traversal**, and scores
  as a confident success. The guard is `min_delta ≥ 2` + `n_values_realised` beside every number; it is now a test
  (`test_a_two_valued_factor_would_score_a_vacuous_monotone_success`).
- **§7 CONFIRMED (env, already filed by `gen-external-classifier`):** `shapes3d_root(root)` treats an explicit root
  as the **dataset** dir while `$CHLU_GEN_DATA_ROOT` is its **parent**. It cost this spoke one run too. ⭐ Worth
  promoting from "will cost the next spoke one red test" to a named §7 item — it has now cost two spokes.
- **§7 UNCHANGED (standing trap, third mention):** `GenBridgeConfig.dec_recipe` still defaults to `"baseline"`.
  This is the **fourth** module to carry a private guard against it. Either flip the default or record it as
  permanent.
- **Config note:** `GenNavigateConfig` lives at the tail of `chlu/config.py`, reached via
  `gen_navigate_config(config)`; ⛔ deliberately **not** a `CHLUConfig` field, 0 house wiring sites, 0 deletions.
- **Ledger:** navigation **FAILS** its dial · both controls ran on every number · the store path is a **staircase**
  (5.92 of 8 steps zero-length, `jump_ratio` 5.63 vs 1.55) but **more direct** (`path_efficiency` 0.841 vs 0.747) ·
  30 bits at every intermediate point, **no widening anywhere** · 157 passed / 0 skipped / 0 failed with
  `CHLU_GEN_DATA_ROOT` set.
