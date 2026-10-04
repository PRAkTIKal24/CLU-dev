# PREREG — `gen-navigation`
**Filed 2026-09-13 21:30 (local), BEFORE any traversal harness existed and before any traversal number was
measured.** Base `main @ be27713`, branch `agent/experiment-engineer/gen-navigation`, worktree `../CHLU-gen-nav`.
Verify by mtime: this file predates `chlu/core/gen_navigate.py`, `chlu/experiments/exp_gen_navigate.py` and every
file under `.claude/outputs/gen-navigation/runs/`.

## DIAL DECLARATION (echoed from the task)
- **Dial:** *navigation* — does the store's landscape support meaningful movement **between** wells/modes?
- **Laundering control:** straight-line interpolation between the two **endpoint READ vectors** with the store
  bypassed at every intermediate point — run on **every** number.
- **Second control:** the identical path read through a **deleted store** (`delete_store`, `mean` — the fair
  deletion), plus `scramble` as a harder second deletion.
- **Falsifies the claim:** the store's landscape carries no traversable structure — operationally, linear
  interpolation traverses **as well or better** than the store path, or the store path never visits an
  intermediate factor value.
- **Does NOT falsify:** (a) losing to the linear control **on level** while the linear control uses a wider
  per-point channel (R23.5's standing reading); (b) failing in Montero's corner *specifically* — R30/R31 already
  established the corner collapses, so a corner failure that is **absent on the spread grid** is a confirmation
  of a standing result, not news; (c) the `mean`-deleted arm producing a constant image (that is arithmetic).

---

## 1. What is measured

A **traversal** is a path of `S+1 = 9` points (`t = 0, 1/8, …, 1`) between two endpoint requests. At each point:
decode → read with the **gated external classifier** (`classifier_gate.json = PASS`) → record all six factor
readings. Four arms, identical endpoints, identical decoder, identical classifier:

| arm | intermediate point | store consulted at intermediate points? | per-point channel |
|---|---|---|---|
| `store_snap` (**the claim**) | per well, the **nearest store mode** to the interpolant between the two endpoint modes → `read_from_codes` | **yes** — the codebook is what snaps the path | `K·log2(M) = 30 bits` |
| `linear` (**control 1**) | `(1−t)·read(c_a) + t·read(c_b)`, decoded directly | **no** | a real vector, `D·32 = 65,536 bits` |
| `deleted_mean` (**control 2**) | `store_snap`'s codes read through `delete_store(store,"mean")` | content deleted, addressing kept | 30 bits, **0 of them informative** |
| `deleted_scramble` (control 2b) | same codes through `delete_store(store,"scramble")` | content destroyed, marginals kept | 30 bits |

⭐ **Both controls run on every number.** ⛔ **No feature tensor appears on any arm's path**: the endpoints are
committed codes from `request_posterior → compose_codes` (request-only, already guarded by `assert_request_only`),
and every intermediate object is a function of `(store.modes, c_a, c_b, t)` alone. `gen_store.soft_read` and
`gen_store.address_codes` are **not imported** into `gen_navigate` — both take FEATURES.

⭐ **Two channel accountings are reported, because only reporting one would flatter somebody.**
*Per-point symbol width* (the 30-bit design constant, R20) and *path-derived information content*
`2·K·log2(M) + log2(S+1) = 60 + 3.17 = 63.17 bits` — which the `linear` control also obeys, since its
interpolant is a deterministic function of the two endpoint codes and `t`.

## 2. Endpoint-selection rule (⛔ fixed here, before any measurement)

Two pools, both mechanically enumerated; ⛔ no pair is hand-picked.

- **`CORNER`** — the 72 held-out (B) requests, i.e. Montero's cell. Realised values (from
  `chlu/data/shapes3d.py` + R31, a static data fact, not a measurement):
  `floor_hue 2 · wall_hue 2 · object_hue 2 · scale 3 · shape 1 · orientation 3` (2·2·2·3·1·3 = 72).
- **`GRID`** — anchors drawn from the **A** set (the 512 seen `eval_a` configurations, spread over the factor
  space), ⛔ excluding any anchor inside Montero's corner. Full grid cardinalities `(10, 10, 10, 8, 4, 15)`.

For each pool and each of the **six** factors `f`:
1. Enumerate **all** unordered pairs of requests that differ in **exactly** `f` and agree on all other five.
2. Prefer the **`|Δ index| ≥ 2`** subset (so at least one intermediate grid value *exists* to be visited). If it
   is empty (CORNER hues: 2 values ⇒ `|Δ| = 1` only; `shape`: categorical) fall back to `|Δ| = 1` and mark the
   pair `qualifies_monotone = false`.
3. Sort the candidate list canonically and draw `n_pairs_per_factor = 8` **without replacement** with
   `numpy.default_rng(seed·7_919 + 20_260_913)`.

⇒ per seed: `CORNER` 5 factors × 8 = 40 pairs (⛔ `shape` has **1** realised value in the corner and yields
**zero** pairs — reported as `NOT APPLICABLE`, never as a zero) + `GRID` 6 × 8 = 48 ⇒ **88 pairs**, of which
**56 qualify** for the monotonicity statistic (`CORNER` scale + orientation = 16; `GRID` 5 ordinal factors = 40).
⭐ `n_values_realised` is reported **beside every per-factor number** (R31's fourth-strike rule).
Seeds **0, 1, 2**.

## 3. Definitions (⛔ fixed here)

Let `R[s, f]` be the classifier's top-1 reading of factor `f` at step `s`; `v_a`, `v_b` the target factor's true
endpoint indices; `d = sign(v_b − v_a)`.

- **`endpoints_read_ok`** — `R[0, tgt] == v_a` **and** `R[S, tgt] == v_b`.
- **`monotone`** — `d·(R[s+1, tgt] − R[s, tgt]) ≥ 0` for every `s` (**weak** monotonicity).
  ⚠ A step function is weakly monotone; that is why criterion 3 below exists and why `|Δ| = 1` pairs are excluded.
- **`visits_intermediate`** — at least one `s` with `R[s, tgt]` strictly between `v_a` and `v_b`.
- **`off_target_stability`** — mean over the five non-target factors of the fraction of the 9 steps whose reading
  equals the shared (true, identical at both endpoints) value.
- ⭐ **`TRAVERSES`** (the per-pair success predicate) — `qualifies_monotone` **and** `endpoints_read_ok` **and**
  `monotone` **and** `visits_intermediate` **and** `off_target_stability ≥ 0.9`.
- **Smoothness** — per-step image RMSE; `jump_ratio = max_step / mean_step` (1.0 = perfectly even);
  `path_efficiency = ‖img_S − img_0‖ / Σ_s ‖img_{s+1} − img_s‖` (1.0 = perfectly direct).
- **Switch statistic** (for `|Δ| = 1` and `shape` pairs, ⛔ never in the headline) — `n_switches` in the target
  trace; `clean_switch` = exactly one switch with correct endpoints.

## 4. ⭐ THE SUMMARY STATISTIC (pre-registered, primary)

> **`NAV` = the fraction of the qualifying pre-registered pairs for which `store_snap.TRAVERSES` is true
> AND `linear.TRAVERSES` is false AND `deleted_mean.TRAVERSES` is false** — i.e. a strict win over both controls.

Reported alongside the three **marginal** traverse rates (`store_snap`, `linear`, `deleted_mean`), by pool and by
factor, each with `n_values_realised` and `n_qualifying_pairs`.

⭐ **Verdict rule, fixed now:** if `linear` traverse rate **≥** `store_snap` traverse rate on the `GRID` pool,
the headline is **"the store's landscape adds nothing over interpolation"** and it is written that way.

## 5. PREDICTIONS (⛔ committed before measurement, with derivations)

| # | prediction | point | interval | derivation |
|---|---|---|---|---|
| **P1** | `CORNER` `store_snap` traverse rate on `scale`/`orientation` | **0.05** | ≤ 0.10 | R31: inside the corner the committed code is ≈ `f(hue triple, shape)`; measured `code_flip_rate` on B is `scale 0.20`, `orientation 0.13`. Most corner pairs differing only in scale/orientation therefore receive the **same** code tuple ⇒ the path is a single point ⇒ `visits_intermediate` is false. |
| **P2** | fraction of `CORNER` scale/orientation pairs whose **two endpoint codes are identical** | **0.78** | 0.55–0.92 | 1 − flip rate, averaged over the two factors ((1−0.20 + 1−0.13)/2 = 0.835), shaded down because the drawn pairs are the extreme `|Δ| = 2` ones, which flip more often than the average edit. |
| **P3** | `GRID` `store_snap` traverse rate | **0.15** | 0.05–0.35 | The snap is a Voronoi assignment: the straight segment between two codebook vectors lies almost entirely inside the two endpoint cells (they share a bisector), so intermediate modes are the exception. R31's companion number agrees — on A, editing one factor alone reaches only 1.7–10.0 distinct codes. |
| **P4** | `GRID` `linear` traverse rate | **0.25** | 0.10–0.45 | The linear control has a continuous per-point object, so an intermediate *rendered* value is available to it whenever the decoder is locally linear in that factor. ⭐ **I predict `linear > store_snap` on `GRID`** — i.e. the task's "most likely outcome". |
| **P5** | ⭐ **`NAV` (the headline)** | **0.07** | ≤ 0.20 | P3 ∧ ¬P4 with the two arms positively correlated (they share endpoints, decoder and classifier), so the strict-win fraction is well below P3. |
| **P6** | `deleted_mean` traverse rate | **0.00** | exactly 0 | Arithmetic: every mode of a `mean`-deleted well is that well's mean mode, so `read_from_codes` returns the **same** vector for every code ⇒ the path is constant ⇒ endpoints cannot both read correctly. ⛔ A non-zero value here is a plumbing bug, not a finding. |
| **P7** | ⭐ **the one axis on which I predict the store WINS:** `off_target_stability`, `store_snap` − `linear`, on `GRID` | **+0.15** | +0.03 … +0.35 | The store path changes only the wells whose snapped mode changes (K = 5, typically 1–2 move), whereas the linear interpolant perturbs **all** D = 2048 coordinates at every intermediate point, so the decoder's rendering of the non-target factors drifts. **A traversal that changes everything is not navigation.** |
| **P8** | `CORNER` `store_snap` `off_target_stability` ≥ `GRID`'s | — | — | Trivially, because in the corner the code barely moves at all. ⛔ **Recorded in advance as a known confirmation, so that it cannot be reported as a win.** |

⛔ **A P-number that fails is a finding and is reported as one.** ⛔ **`NAV ≤ 0.20` is my own prediction, so
measuring `NAV ≈ 0` is NOT a surprise and will not be dressed as one; the informative content is then
P7 (does the store at least move *less* of the image?) and P1/P2 (does navigation inherit the collapse?).**

## 6. What would prove this spoke wrong (task's own list, restated)
1. `linear` traverses as well as or better than `store_snap` ⇒ the store's landscape is not doing the work.
2. Traversal works only on the hue axes and fails on `scale`/`orientation` ⇒ navigation inherits the commitment
   collapse rather than escaping it.
3. The pairs were hand-picked — §2's enumeration + seeded draw is the guard.

---

# ⭐ ADDENDUM 1 — 2026-09-13 21:37, written AFTER seed 0's first production run and
# BEFORE any number in the report. ⛔ Declared as post-hoc; verify by mtime.

**⛔ Nothing in §3's `TRAVERSES` definition or §4's `NAV` statistic changes.** The
pre-registered headline is measured exactly as filed. Two changes, both declared:

### (a) ⛔ A PREREG-COMPLIANCE FIX, not a new choice
§2 states `shape` is **categorical** and therefore switch-only. The first harness
build nevertheless let `GRID/shape` into the `|Δ| ≥ 2` (monotone-qualifying) set,
because it has 4 ordinal-looking indices. That **contradicts the filed prereg** and
dilutes the monotone denominator with a factor for which "monotone" is meaningless.
Fixed by a declared `categorical_factors = ["shape"]` config field. ⭐ This restores
§2; it does not amend it. `GRID/shape`'s first-run reading (`store_snap` traverse
rate **0.00** on 8 pairs) is recorded here so that the exclusion cannot be a way of
dropping an inconvenient row — **it was zero either way.**

### (b) ⭐ THREE DIAGNOSTICS ADDED, because the first run's zero could not be
### attributed without them — ⛔ declared post-hoc, ⛔ never the headline
Seed 0's first run scored `NAV = 0.0000` with **every** arm at zero, and the record
schema could not say *which* clause failed. The added fields are attribution only:

1. **`readings`** — the full `(S+1, 6)` classifier matrix per pair per arm, so every
   statistic below is recomputable from disk.
2. **`off_target_constancy`** — the fraction of steps whose non-target readings equal
   the **step-0 reading** (rather than the *requested* value). ⚠ §3's pre-registered
   `off_target_stability` measures against the requested value and therefore
   **conflates two different failures**: the path drifting, and the store's read
   never having rendered the non-target factors correctly *at either endpoint*.
   ⭐ The conflation is itself a finding; the split is reported beside the
   pre-registered number, ⛔ never in place of it.
3. **`endpoint_fidelity`** — the fraction of all six factors the classifier reads as
   requested at `t = 0` and `t = 1`. ⭐ This is the **ceiling** on any traversal
   claim: if the endpoints do not render the request, nothing between them can.

**Predicted before recomputation (addendum predictions, same rules apply):**

| # | prediction | point | derivation |
|---|---|---|---|
| **P9** | `GRID` `off_target_constancy` for `store_snap` exceeds its pre-registered `off_target_stability` by ≥ 0.15 | +0.25 | the first run's `stab` of 0.60 sits beside an `endpoints_read_ok` of only 0.48, so most of the "instability" must be a constant misread, not drift. |
| **P10** | ⭐ `store_snap` `off_target_constancy` > `linear`'s on `GRID` (P7 restated on the diagnostic that isolates drift) | +0.10 | K = 5 wells: the snapped path perturbs only the wells whose mode changes, the interpolant perturbs all D = 2048 coordinates at every point. |
| **P11** | `GRID` `endpoint_fidelity` ≈ 0.75 | 0.60–0.90 | four of six Shapes3D factors sit near 1.0 for any competent reader (R16), while the 30-bit read is known to drop `scale` and `orientation` inside the corner and to be imperfect outside it. |
