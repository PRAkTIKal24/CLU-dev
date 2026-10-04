# bprime-s-remeasure — results-analyst report
Task + acceptance criterion: re-estimate `bprime-c6`'s `s` under the registered `α‖q‖²`-subtracted
convention on the SAME banked cells, report both conventions side by side, and state whether every
downstream `d/s` claim moves in the strengthening direction. **Status: done** (all 21 banked cells
re-measured; two independent estimators × two conventions).

## ⛔ HEADLINE — the pre-registered direction is FALSIFIED, in the harmless direction (task §4)
**The correction is a NO-OP on `bprime-c6`: its `s` was ALREADY confinement-subtracted.** My
re-measurement under the registered convention reproduces the banked `s_fitted_well`
**bitwise on 21/21 cells (max |Δ| = 0.000e+00)**. So `s` does **not** get smaller, `d/s` does **not** get
larger, and no draft language is proposed on the strength of a correction that does not exist.
⭐ The flag is nevertheless **discharged, not merely dropped**: the un-subtracted ruler *is* biased on
this rig, by a **measured 1.093 ± 0.023 (E1) / 1.067 ± 0.036 (E2), n = 18 admissible cells — not 1.44×**,
and `bprime-c6` never used it. ⚠ **Sign note for the Hub: the bias runs the other way from the flag's
premise for c6** — had c6 used the un-subtracted ruler, `s` would be *larger* (0.4371 vs 0.4006 sweep-mean)
and `d/s` *smaller* (3.255 vs 3.588 at the shipped cell), i.e. the un-corrected reading would have
**weakened**, not strengthened, the suppression claim.

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. **N224 / claims-matrix §0.9 (xxx-adjacent instrument clause) must be re-worded**: `s = 0.40` is
   **CHECKED AND CLEARED**, not "flagged". Proposed wording in §7.
2. ⛔ **`bprime-c6` §1's table, shipped row, `d/s (fitted s)` = 3.72 is CONTRADICTED by c6's own
   summary artifact**: `c6_summary.json → rows[R=1.00].ds_fit = **3.5902**` (and the raw cells agree).
   **3.7131 is `sep̄/s̄`** — spacing, not mean third-party distance. **All five other rows in that column
   match the artifact to the digit** (1.1019 / 1.7055 / 2.0483 / 2.6781 / 4.4088), so this is an isolated
   **transcription/convention slip on one cell**, in the *overstating* direction (+3.5 %). Owner needed
   (paper-writer / doc-curator) — see §5.
3. ⛔⛔ **ENVIRONMENT INCIDENT, program-wide, not mine to fix (see §6):** the working tree, the venv and
   part of `.git` are **iCloud-evicted and unrecoverable** (`dataless`, `errno 60`). **`uv run chlu` cannot
   start**, 18 tracked `chlu/*.py` files, **3 490 of 16 996 venv files**, **552 loose git objects** and
   `.claude/outputs/bprime-c6/c6_summary.json` currently **cannot be read at all**.

## ⭐ DIAL DECLARATION (echoed from the task file, protocol §7)
- **Dial:** none — instrument correction on banked artifacts. One measured quantity (`s`) re-estimated;
  no arm retrained, no dividend re-scored.
- **Laundering control:** not applicable (no performance number); the substitute-control role is played by
  the **independent second estimator** (`orgdiv-cat-test`'s own `effective_s`) run on `bprime-c6`'s store.
- **Falsifies:** the pre-registered direction failing. **It failed** — reported as the headline above.
- **Does NOT falsify:** any c6 verdict; nothing here re-scores T5.4/T5.5, the table launder, or the gate.

---

## 1. Setup — what was run, exactly

**Nothing was retrained.** The banked cells were **rebuilt deterministically** (same family/arm, same
seed, same config, same commit-state code) and identity with the banked store was **asserted before any
convention was read**: `sep` and `s_fitted_well` reproduce **digit-for-digit on 21/21 cells**.

| item | value |
|---|---|
| banked artifacts read | `.claude/outputs/bprime-c6/results/exp_route3_thirdparty.json` (18 cells) · `.../thirdparty_topup_R042.json` (3 cells) · `.claude/outputs/bprime-c6.md` §1/§1.1/§7 |
| banked artifact, initially unreadable → **recovered** | `.claude/outputs/bprime-c6/c6_summary.json` — iCloud-`dataless` (`TimeoutError errno 60`) for most of this session; it **materialised late** (§6) and was then read. ⭐ **It confirms my independent re-derivation digit-for-digit** (§2.3) and it is the file that settles reconciliation item 2. Nothing in this report depended on it: every number was first derived from the two raw cell dumps. |
| code state measured | **`be995ca`** = `bprime-c6`'s own branch tip (`agent/experiment-engineer/bprime-c6`, an ancestor of `main`), 128 of its 134 tracked `chlu/*` blobs; **7 blobs unreadable in `.git`** (`chlu/chlu.py`, `core/blocks.py`, `core/clu_controller.py`, `core/monitors.py`, `core/regularization.py`, `data/__init__.py`, `training/train_baselines.py`) were filled from the **HEAD worktree** copies. ⚠ Declared hybrid; validated by exact reproduction of every banked number. |
| repo HEAD at report time | `7fcef50` |
| estimator provenance | `chlu/core/clu_system.py::_well_fit` is **byte-identical at `d4f56c8` (c6's base), `be995ca` (c6's tip) and HEAD `7fcef50`** — verified by extracting the function from `git show <sha>:…` and string-comparing (1 993 chars, 3/3 equal) |
| env | ⚠ **fresh venv at `/tmp/rmvenv`** (the repo venv is destroyed, §6), pinned to the c6-era versions read off the dead venv's `dist-info` names: **JAX 0.9.0 · jaxlib 0.9.0 · numpy 2.4.1 · scipy 1.17.0 · equinox 0.13.4 · optax 0.2.6 · diffrax 0.7.0 · matplotlib 3.10.8** — same JAX as `bprime-c6`'s run (its §7 flag table: "JAX 0.9.0") |
| family / arm | `overload/load1x_shipped`, `atoms_per_item=341`, `n_offer=capacity=budget=6`, `stage_admission=True` |
| swept axis | `ball_radius ∈ {0.42, 0.55, 0.64, 0.80, 1.00, 1.20}` with `d_safe_override = 0.58·R` (`_sweep_overrides`), **nothing else changed** |
| seeds | **{0,1,2} at every radius; +{3,4,5} at `R = 0.42`** (c6's own top-up) = 21 cells, 18 admissible |
| store / read flags | `addr_dim=4, payload_dim=1, dim=6, atom_width=0.30, confine α=0.05, dt=0.05, gamma_address=0.05, gamma_read=0.02, address_steps=400, read_steps=800, query_sigma σ_q=0.15, kinetic_mode=newtonian_learned`; write `300 steps, lr 3e-3, wd 1e-4, σ_addr 0.25, σ_pay 0.6, margin 0.15, barrier 0.2, masked_write=True` (identical to c6 §7) |
| wall-clock | 18-cell sweep **204.0 s**, top-up **41.0 s**, on CPU |
| commands | `cd /tmp/chlu-c6 && PYTHONPATH=/tmp/chlu-c6 /tmp/rmvenv/bin/python remeasure_s.py --radii 0.42 0.55 0.64 0.80 1.00 1.20 --seeds 0 1 2 --out /tmp/rm_main.json` · `… --radii 0.42 --seeds 3 4 5 --cancellation-cell 9 9 --out /tmp/rm_topup.json` · `python analyse.py` · `python t55_table.py` |
| artifacts | `.claude/outputs/bprime-s-remeasure/{PREREG.md, rm_main.json, rm_topup.json, rm_main.log, rm_topup.log, remeasure_summary.json, analysis_stdout.txt, s_conventions.png, remeasure_s.py, analyse.py, t55_table.py, plot.py}` |
| **PREREG** | `.claude/outputs/bprime-s-remeasure/PREREG.md`, written **before** the harness ran; both hypotheses (registry's and mine) with numbers. Scorecard in §3. |

### 1.1 The two estimators, spelled out
- **E1 — the estimator that produced `bprime-c6`'s `s`.** `CluSystem._well_fit`: fit
  `V(z+ru) − α(‖z+ru‖² − ‖z‖²) − V(z) ≈ D(1 − e^{−r²/2s²})` over **8 random directions × 12 radii ∈
  [0.15, 1.5]**, `s` by grid search over **120 points in [0.05, 1.2]** (⇒ `s` is quantised at
  **0.00966**, i.e. ±2.7 % at `s ≈ 0.36`). Per cell: **median over the 6 live items**
  (`exp_route3_attribution.py` L565), then **mean over admissible seeds** per radius.
  My variant is that function **verbatim**, with the confinement coefficient exposed as an argument;
  with the coefficient set to `α` it reproduces the shipped `well_fits()` **bitwise on 21/21 cells**.
- **E2 — the `orgdiv-cat-test` estimator, i.e. N224's own ruler.**
  `chlu/core/factored_store.py::effective_s`: slope of `ln(profile)` vs `r²`, **16 rays × 24 radii out to
  `4·s_hint`**, `s_hint = atom_width = 0.30` and `confine = α` — exactly how `exp_cat_test.py` L193/L201
  calls it. Median over items.
- **E3 — the T5.5 coupling-law fit** (`bprime-c6` §1.1's `s = 0.3979`, `R² = 0.9953`):
  `ln κ − ln(d/σ_q)` regressed on `d² − σ_q²`, slope `= −1/2s²`, on the 6 per-radius points.
- **Conventions:** **(S)** subtract `α‖q‖²` (α = `confine` = 0.05, the store's exact term:
  `memory_potentials.py` → `return v + self.confine * jnp.sum(q**2)`); **(U)** do not.

---

## 2. Results

### 2.1 ⭐ Per-cell, both conventions, both estimators (21 cells; `adm` = c6's own admissibility)
| R | seed | adm | banked `s` | **E1 (S) corrected** | E1 (U) | U/S | **E2 (S)** | E2 (U) | U/S | Δ(E1(S) − banked) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.42 | 0 | Y | 0.494538 | **0.494538** | 0.523529 | 1.0586 | 0.449827 | 0.458991 | 1.0204 | **0.0e+00** |
| 0.42 | 1 | n | 0.451050 | **0.451050** | 0.470378 | 1.0429 | 0.444385 | 0.454075 | 1.0218 | **0.0e+00** |
| 0.42 | 2 | Y | 0.460714 | **0.460714** | 0.494538 | 1.0734 | 0.445883 | 0.451926 | 1.0136 | **0.0e+00** |
| 0.42 | 3 | n | 0.518697 | **0.518697** | 0.538025 | 1.0373 | 0.455058 | 0.461570 | 1.0143 | **0.0e+00** |
| 0.42 | 4 | n | 0.470378 | **0.470378** | 0.494538 | 1.0514 | 0.451871 | 0.456552 | 1.0104 | **0.0e+00** |
| 0.42 | 5 | Y | 0.489706 | **0.489706** | 0.518697 | 1.0592 | 0.453112 | 0.462154 | 1.0200 | **0.0e+00** |
| 0.55 | 0 | Y | 0.436555 | **0.436555** | 0.465546 | 1.0664 | 0.434095 | 0.439055 | 1.0114 | **0.0e+00** |
| 0.55 | 1 | Y | 0.393067 | **0.393067** | 0.426891 | 1.0861 | 0.416822 | 0.441003 | 1.0580 | **0.0e+00** |
| 0.55 | 2 | Y | 0.407563 | **0.407563** | 0.446218 | 1.0948 | 0.428352 | 0.445697 | 1.0405 | **0.0e+00** |
| 0.64 | 0 | Y | 0.397899 | **0.397899** | 0.431723 | 1.0850 | 0.409756 | 0.433960 | 1.0591 | **0.0e+00** |
| 0.64 | 1 | Y | 0.393067 | **0.393067** | 0.426891 | 1.0861 | 0.413179 | 0.437635 | 1.0592 | **0.0e+00** |
| 0.64 | 2 | Y | 0.407563 | **0.407563** | 0.446218 | 1.0948 | 0.421165 | 0.442066 | 1.0496 | **0.0e+00** |
| 0.80 | 0 | Y | 0.378571 | **0.378571** | 0.417227 | 1.1021 | 0.385920 | 0.419075 | 1.0859 | **0.0e+00** |
| 0.80 | 1 | Y | 0.393067 | **0.393067** | 0.426891 | 1.0861 | 0.403403 | 0.434300 | 1.0766 | **0.0e+00** |
| 0.80 | 2 | Y | 0.383403 | **0.383403** | 0.431723 | 1.1260 | 0.393272 | 0.431983 | 1.0984 | **0.0e+00** |
| **1.00** | 0 | Y | 0.359244 | **0.359244** | 0.402731 | 1.1211 | 0.385107 | 0.425386 | 1.1046 | **0.0e+00** |
| **1.00** | 1 | Y | 0.368908 | **0.368908** | 0.412395 | 1.1179 | 0.386638 | 0.425587 | 1.1007 | **0.0e+00** |
| **1.00** | 2 | Y | 0.359244 | **0.359244** | 0.383403 | 1.0673 | 0.390036 | 0.418387 | 1.0727 | **0.0e+00** |
| 1.20 | 0 | Y | 0.378571 | **0.378571** | 0.422059 | 1.1149 | 0.378082 | 0.422367 | 1.1171 | **0.0e+00** |
| 1.20 | 1 | Y | 0.330252 | **0.330252** | 0.373739 | 1.1317 | 0.366162 | 0.412596 | 1.1268 | **0.0e+00** |
| 1.20 | 2 | Y | 0.378571 | **0.378571** | 0.417227 | 1.1021 | 0.383780 | 0.419295 | 1.0925 | **0.0e+00** |

- **bitwise-identical to the shipped `well_fits()`: 21/21.** `max |Δ(E1(S) − banked)| = 0.000e+00`;
  `max |sep_new − sep_banked| = 0.000e+00` ⇒ **the rebuilt store IS the banked store.**
- **The hazard, measured on this rig (18 admissible cells):**
  **E1 U/S = 1.0930 ± 0.0229 sd** (SE 0.0054; range **1.0586 – 1.1317**; **18/18 > 1**)
  **E2 U/S = 1.0671 ± 0.0362 sd** (SE 0.0085; range **1.0114 – 1.1268**; **18/18 > 1**).
  ⇒ the bias is **real and one-signed**, and **N224's 1.44× does not transfer**: 1.44 sits
  **(1.44 − 1.093)/0.0054 ≈ 64 SE** above the measured E1 factor. ⚠ It also **grows with the geometry**
  (1.05 at `R = 0.42` → 1.13 at `R = 1.20`), which is the one number a future ruler-user should carry.
- Well **depth** is inflated by the same defect (shipped cell: `D` 0.566 → 0.654, **+15.5 %**).

### 2.2 Per radius — the corrected `d/s`, beside every alternative ruler
(`d̄` = mean third-party distance over admissible seeds, from the banked artifact, unchanged by anything here)

| R | n | `d̄` | **`s` corrected (= banked)** | **`d/s` corrected** | `s` UNsub | `d/s` UNsub | `s` E2 (S) | `d/s` E2 | `d/s` atom-width 0.30 |
|---|---|---|---|---|---|---|---|---|---|
| 0.42 | 3 | 0.5300 | **0.4817** | **1.100** | 0.5123 | 1.035 | 0.4496 | 1.179 | 1.767 |
| 0.55 | 3 | 0.7034 | **0.4124** | **1.706** | 0.4462 | 1.576 | 0.4264 | 1.650 | 2.345 |
| 0.64 | 3 | 0.8183 | **0.3995** | **2.048** | 0.4349 | 1.881 | 0.4147 | 1.973 | 2.728 |
| 0.80 | 3 | 1.0302 | **0.3850** | **2.676** | 0.4253 | 2.422 | 0.3942 | 2.613 | 3.434 |
| **1.00 (shipped)** | 3 | 1.3006 | **0.3625** | **3.588** | 0.3995 | 3.255 | 0.3873 | 3.358 | 4.335 |
| 1.20 | 3 | 1.5956 | **0.3625** | **4.402** | 0.4043 | 3.946 | 0.3760 | 4.243 | 5.319 |

**Sweep-mean `s`: corrected 0.4006 · un-subtracted 0.4371 · E2-corrected 0.4080.**
⭐ **`bprime-c6`'s headline "`s = 0.40`" IS the corrected number** (0.4006 is exactly the sweep-mean of
the per-radius corrected `s`; §1.1's "well_fits() (independent) 0.4006" is that mean), **and N224's own
estimator, run on this store with the subtraction on, independently returns 0.4080 — a 1.8 % agreement.**

### 2.3 The suppression fit — invariant to the convention, by construction
Recomputed from the banked raw cells (merged with the top-up), reproducing `bprime-c6` §1.1 **to the digit**:

| channel | slope | implied `s` | **R²** | decades |
|---|---|---|---|---|
| static ∇V ratio | **−3.157582** | **0.397931** | **0.995279** | 2.719 |
| slot coupling at `t = 1` | **−3.155629** | **0.398054** | **0.995239** | 2.717 |

(c6 §1.1 published −3.158 / 0.3979 / 0.9953 and −3.155 / 0.3981 / 0.9952.)
⭐ **Independently confirmed against c6's own merged summary once it materialised** — `c6_summary.json →
fits` carries `slope −3.1575816989210934`, `s_implied 0.39793093654915646`, `r2 0.995278728468345`,
`prefactor 0.3788438654075226` (and −3.1556290057633323 / 0.39805403670538725 / 0.9952389315579215 for
the slot-coupling channel): **my re-derivation reproduces every digit**, so §1.1's fit is sound and
reproducible from the raw cells alone.
⛔ **The `R² = 0.995` claim cannot move under any `s` convention**: `coupling_law_fit`
(`chlu/eval/attribution.py` L572–605) regresses `ln κ − ln(d/σ_q)` on `d² − σ_q²`; **`s` is the OUTPUT of
that regression and enters nowhere in it.** The only thing a convention change can move is the *x-axis
labels* of the plotted curve (column 5 vs 7 vs 10 of §2.2).

### 2.4 ⭐ Why E3 could never have been contaminated (the mechanism, measured)
`bprime-c6`'s coupling κ is built from `‖∇V_full − ∇V_{−k}‖` (`exp_route3_attribution.py` L490–L500):
**a difference of two potentials whose confinement term is identical and static**, so `α` cancels
analytically. Measured on the shipped cell (seed 0, 24 query points): recomputing both gradients with the
confinement analytically removed changes the difference by
**max |Δ| = 5.96e-08 against a mean difference-norm of 0.2134 (2.8e-7 relative)** — float32 rounding, not
a bias. (`rm_main.json → cells[12].gradient_confinement_cancellation`.)

### 2.5 The T5.5 closed form under each ruler (context for §2.2, not a claim)
| R | `d̄` | measured κ ± 2 SE | T5.5 @ `s = 0.30` | **T5.5 @ corrected `s`** | T5.5 @ UNsub `s` | meas / T5.5(corrected) |
|---|---|---|---|---|---|---|
| 0.42 | 0.5300 | 8.139e-01 ± 3.1e-01 | 8.410e-01 | 2.025e+00 | 2.160e+00 | 0.402 |
| 0.55 | 0.7034 | 3.441e-01 ± 1.8e-01 | 3.402e-01 | 1.170e+00 | 1.432e+00 | 0.294 |
| 0.64 | 0.8183 | 2.262e-01 ± 3.9e-02 | 1.498e-01 | 7.184e-01 | 9.862e-01 | 0.315 |
| 0.80 | 1.0302 | 9.702e-02 ± 2.3e-02 | 2.140e-02 | 2.066e-01 | 3.887e-01 | 0.470 |
| **1.00** | 1.3006 | 1.534e-02 ± 6.8e-03 | 8.149e-04 | **1.512e-02** | 4.649e-02 | **1.015** |
| 1.20 | 1.5956 | 1.553e-03 ± 1.9e-03 | 8.680e-06 | 7.180e-04 | 4.735e-03 | 2.162 |
⚠ **Do not quote the shipped row's 1.5 % agreement as a validation.** The ratio is **not constant**
(0.29 → 2.16 across the sweep) — the corrected-`s` closed form *crosses* the measurement near the shipped
cell. That non-constancy is precisely why §1.1 fits the **slope** and reports a **0.379 prefactor**
instead of quoting the closed form pointwise.

*(Figure: `.claude/outputs/bprime-s-remeasure/s_conventions.png` — `s` vs `R` for all four
estimator×convention combinations with per-cell points; `d/s` vs `R`; and the inflation-factor histogram
against N224's 1.44×.)*

---

## 3. PREREG scorecard (`.claude/outputs/bprime-s-remeasure/PREREG.md`, written before the harness ran)
| # | prediction | measured | verdict |
|---|---|---|---|
| **P-R1** (registry) | corrected `s` at shipped cell **0.3625 → ≈0.25** (≥20 % drop) | **0.3625, Δ = 0.0 %** | ⛔ **REFUTED** |
| **P-R2** (registry) | `d/s` shipped **3.59 → ≈5.2** | **3.588, unchanged** | ⛔ **REFUTED** |
| **P-R3** (registry) | headline `s = 0.40 → ≈0.28` | **0.4006, unchanged** | ⛔ **REFUTED** |
| **P-M1** (mine) | E1(S) reproduces banked `s` to all digits (Δ = 0) | **bitwise, 21/21** | ✅ |
| **P-M2** (mine) | `d/s` unchanged | unchanged | ✅ |
| **P-M3** (mine) | inflation **1.44× point, band [1.10, 1.80]** | **E1 1.0930 ± 0.0229; E2 1.0671** | ⛔ **MY OWN POINT AND BAND ARE REFUTED** — the true factor is ~2× smaller than N224's, and **11 of 18 cells sit below my band's floor**. Recorded as a failure of my transfer assumption, not as a c6 finding. |
| **P-M4** (mine) | E3 `s_implied` and `R²` invariant to convention | invariant **by construction** (§2.3) + gradient cancellation 5.96e-08 (§2.4) | ✅ |
| **P-M5** (mine) | E2(S) ∈ [0.25, 0.40] at shipped cell; E2(U)/E2(S) ∈ [1.2, 2.0] | **E2(S) = 0.3873 ✅ in band; ratio 1.0927 ⛔ below band** | ◐ **half-refuted** — and the informative half is the ✅: E2 and E1 **agree to 1.8 % on the sweep mean**, so `s ≈ 0.40` is **not** estimator-dependent on this store. |

---

## 4. Interpretation — tied to the specific CHLU claim under test
1. **The claim "`bprime-c6`'s `d/s` rides an inflated ruler" is dead.** The ruler behind every number in
   c6 §1/§1.1 — `_well_fit` — has carried the exact analytic subtraction `α(‖z+ru‖² − ‖z‖²)` since before
   c6 ran (byte-identical at `d4f56c8`, `be995ca`, HEAD), and c6's *second* estimator (the coupling law)
   is confinement-free by an algebraic cancellation, not by a choice. Two independent channels, both clean.
2. **The 21 % gap between `orgdiv-cat-test`'s `s = 0.318` and c6's `s = 0.40` is a STORE difference, not
   a RULER difference.** N224's own estimator, run here with N224's own convention, returns **0.4080**
   on c6's store — 1.8 % from c6's 0.4006 and **28 % above** the cat-test's 0.318. The two rigs simply
   have different well widths (different family, `a = 32` atoms/well vs `atoms_per_item = 341`,
   different geometry). N224's closing sentence — *"`s` was already known not to be a constant of the
   architecture (0.482 → 0.362)"* — is the correct reading, and this measurement supplies the receipt:
   **within c6's own sweep `s` moves 0.4817 → 0.3625 (−25 %) purely by moving `ball_radius`.**
3. **Direction, stated plainly as the task requires:** every downstream `d/s` claim **does not move** —
   `d/s` corrected == `d/s` banked, exactly, at all six radii. ⛔ Nothing strengthens; nothing weakens.
   The *counterfactual* un-corrected reading would have moved `d/s` **down** (3.588 → 3.255 at the shipped
   cell), i.e. the registry's premise that the correction is claim-strengthening is inverted for c6.
4. **What the flag was actually worth.** It was worth **9.3 %** on this rig — a real, one-signed,
   geometry-dependent bias that a future `s`-measurement *would* suffer if it used a bare log-fit. The
   flag was correct as an instrument warning and wrong as a transferred magnitude.

## 5. Limitations, confounds, and declared NOT-RUNs
- **Hybrid code state (§1).** 7 of 134 `chlu/*` blobs at `be995ca` are unreadable in `.git`; HEAD copies
  were used for them. ⚠ Mitigation and its strength: every banked quantity I could compare against
  (`sep`, `s_fitted_well`, all 21 cells) reproduces **bit-for-bit**, which is a far stronger identity test
  than a file diff. Residual risk is confined to code paths that touch **none** of those quantities.
- **E1's grid quantisation** (0.00966 ⇒ ±2.7 % per cell) puts a floor on per-cell U/S ratios; this is why
  E2 (continuous) is reported beside it. Both agree: 1.093 vs 1.067.
- **Aggregation order.** §2.2 uses `d̄/s̄`; c6's artifact uses `mean(d/s per cell)`. They differ in the 3rd
  decimal (shipped cell 3.5882 vs 3.5902). ⛔ Neither is 3.72 (reconciliation item 2).
- ⛔ **NOT RUN — the cat-test rig re-measured with c6's estimator (the mirror of this task).** N224's
  0.438 / 0.304 / 0.318 are **not challenged and not reproduced here**; I only measured whether the factor
  *transfers* (it does not). Cost ≈ one `exp-cat-test` cell per seed; declare it if the Hub wants the
  1.44× itself audited.
- ⛔ **NOT RUN — any re-run of c6's third-party coupling probe.** The κ values in §2.3/§2.5 are the banked
  ones; only `s` was re-estimated. No verdict, no dividend, no gate was re-scored.
- ⛔ **NOT RUN — a mechanism study of why the inflation is 1.09 here and 1.44 there.** The measured
  covariate is that the factor **grows with `ball_radius`** (wells further from the origin, shallower:
  depth 1.40 → 0.57); that is a **hypothesis with a trend behind it, not an adjudication**.
- **`c6_summary.json` was unreadable for most of the session** and every c6 quantity here was therefore
  re-derived from the two raw cell dumps first; when the file finally materialised it **agreed to the last
  digit** on the fits (§2.3) and on all six per-radius rows, and it supplied the decisive evidence for
  reconciliation item 2. ⇒ **no result in this report rests on an artifact I could not open.**

## 6. ⛔⛔ Environment incident (blocking, program-wide — for the Head/Hub, not fixable by me)
`~/Desktop/CHLU` sits inside iCloud Drive's Desktop sync, and the sync daemon is erroring
(`brctl status`: `CKErrorDomain:7 / CKInternalErrorDomain:2061`, `needs-sync-up`, last container reset
2026-08-10). Files have been **evicted to the cloud and fault back in only pathologically slowly** — a
normal read aborts with `TimeoutError errno 60` after **~6 s**, and `brctl download` does not help.
⚠⚠ **CORRECTED LATE IN THE SESSION, and it changes the diagnosis:** this is a **fetch-latency fault, not
(mostly) data loss.** Two greps that I had abandoned as hung stayed blocked on dataless files and
**completed successfully after ~40 min**, and re-checking the flags afterwards showed
`chlu/eval/attribution.py`, `chlu/experiments/__init__.py`, `chlu/training/train_memory.py` and
`chlu/data/figure8.py` are now **materialised** (`blocks` 0 → 80/8/48/8). ⇒ **a blocking read of ~tens of
minutes per file does eventually win**; a 6-second read never does. ⭐ **And the one file I had declared lost came back**: `.claude/outputs/bprime-c6/c6_summary.json`
read cleanly (63 430 B) on a patient retry loop after ~2 h of accumulated attempts ⇒ **nothing in this
task's evidence base was actually lost.** ⚠ Still worth the Head's attention: `brctl status` reports a
**`needs-sync-up` upload error on `.claude` paths**, i.e. some `.claude/**` artifacts may **never have been
uploaded**, so for those the cloud copy cannot be assumed to be a recovery path.
Inventory at the time of the run:
- **venv: 3 490 of 16 996 files dataless.** `import jax` fails at
  `dlopen(jaxlib/_profile_data.so) mmap(...) errno=60` ⇒ **`uv run chlu …` cannot start at all.**
- **working tree: 18 tracked `chlu/*.py` dataless**, incl. `chlu/eval/attribution.py`,
  `chlu/experiments/__init__.py`, `chlu/training/train_memory.py` ⇒ **`import chlu` fails** even with a
  working venv. (All 18 were recoverable from `.git`.)
- **`.git`: 552 loose objects dataless** ⇒ `git archive`, `git log -- <path>`, `git log -S` abort with
  `fatal: mmap failed`. ⚠ **`git log --oneline A..B -- <path>` returns EMPTY-and-exit-0 under this fault**
  — a silent wrong answer; I hit it and caught it only by re-running without `2>/dev/null`.
- **`.claude` artifacts:** at least `outputs/bprime-c6/c6_summary.json`, `sweep.log`,
  `results/c6_curve.png` are dataless (the raw `results/*.json` survived).
- **Tool-by-tool failure signatures, measured this session (they differ, and one is silent):**
  `grep`/`ugrep` over a tree containing dataless files **hangs 2–40 min, then exits non-zero (2)** — but
  the matches it did print are **complete for the readable files** (verified: the recursive
  `def well_fits` sweep returned the true `clu_system.py:1308` hit *and* exit 2). ⛔ **`git log -- <path>`
  is the dangerous one: EMPTY output, exit 0** — I was briefly misled into "0 commits since `be995ca`" for
  seven files before re-running without `2>/dev/null` exposed `fatal: mmap failed`. `python open().read()`
  and `cp`/`head`/`dd` fail loudly (`TimeoutError errno 60` / `fcopyfile failed`) in ~6 s.
- ⭐ **Cold-start is NOT intrinsically 20 min:** in the clean `/tmp` venv, `import jax` takes **16.5 s**
  and the full 18-cell sweep **204 s**. Protocol §4's "JAX cold-start ~20 min here" is very likely this
  same eviction fault, not JAX. **Worth re-testing after the storage is repaired.**
- ⚠ **Recommended to the Head (revised by the late correction above):** treat this as **(a) a severe
  latency fault** that makes the repo unusable in-place until storage is repaired — the `/tmp`
  reconstruction in §1 is the workaround, and it is *faster* than waiting — and **(b) a bounded
  data-loss risk confined to `.claude/**`**, whose gitignored artifacts exist only in this tree and are
  the ones `brctl` reports as failing to upload. Tracked code is safe: all 18 dataless `chlu/*.py` were
  recovered from `.git` in seconds. Nothing was deleted by me.

## 7. Proposed N224 discharge wording (the disposition is the Hub's; I propose only)
> **N224 — ⟲ DISCHARGED BY MEASUREMENT (`bprime-s-remeasure`, 2026-08-18), instrument clause RETAINED.**
> The instrument half stands: a bare radial log-fit that does not subtract `α‖q‖²` **is** biased upward,
> **measured 1.093 ± 0.023 (E1) / 1.067 ± 0.036 (E2) on 18 admissible cells of the `overload/load1x_shipped`
> sweep**, one-signed on 18/18 and **growing with geometry** (1.05 at `ball_radius` 0.42 → 1.13 at 1.20).
> ⛔ **The transfer half is REFUTED: `bprime-c6`'s `s = 0.40` is CHECKED AND CLEARED, not flagged.** Its
> estimator (`CluSystem._well_fit`, byte-identical at `d4f56c8`/`be995ca`/HEAD) **already subtracts the
> confinement analytically**, and its second channel (the coupling law) is confinement-free by exact
> cancellation of a static term in `‖∇V_full − ∇V_{−k}‖` (residual 5.96e-08 vs difference-norm 0.2134).
> Re-measurement under the registered convention reproduces the banked `s` **bitwise on 21/21 cells
> (max |Δ| = 0.000e+00)** ⇒ **`d/s` does not move at any radius**, and `R² = 0.995` **cannot** move
> (`s` is the fit's output, not its input). ⛔ **`1.44×` is a CAT-TEST-STORE number and must never be
> applied to another rig's `s`** — on c6's store the same estimator with the same convention returns
> **0.4080** vs c6's **0.4006** (1.8 % apart), so the 0.318-vs-0.40 gap is a **store** difference
> (`s` moves 0.4817 → 0.3625 across c6's own `ball_radius` sweep), **not a ruler** difference.
> ⚠ **The standing clause that survives:** *any `d/s` must still travel with its subtraction convention
> AND its rig* — but the convention question is now **answered** for `bprime-c6`.

### 7.1 Exact numbers for r7's MF-3 rider, if the Head wants it updated beyond the flag
Drop-in replacement for a "flagged" rider (all figures traceable to §2):
> *The well width `s` is measured on the learned store with the confinement term `α‖q‖²` subtracted
> analytically (α = 0.05); without that subtraction the same fit reads **9.3 % high** (1.093 ± 0.023 over
> 18 cells, rising to 1.13 at the widest geometry). The subtracted values are used everywhere:
> `s = 0.4817 / 0.4124 / 0.3995 / 0.3850 / 0.3625 / 0.3625` at `ball_radius = 0.42 … 1.20`
> (sweep-mean **0.4006**), and an independent estimator of a different functional form returns
> **0.4080** on the same store. The suppression fit itself is convention-free: `s` is its output
> (`slope = −3.1576`, `s_implied = 0.3979`, `R² = 0.9953`).*
⚠ If the Head prefers the rider to stay a one-line flag, the **only** required edit is the word
**"flagged" → "checked"**; no number in r6 changes.

## 8. Git footprint
**None.** No tracked file was created, modified, or committed; no branch, no worktree in the repo. All
work is under `.claude/outputs/bprime-s-remeasure/`, `.claude/scratch/bprime-s-remeasure/` and the
throwaway `/tmp/{chlu-c6, chlu-head, chlu-rm, rmvenv}`. (⚠ The `/tmp` trees hold a reconstructed copy of
the package; they are outside the repo and will vanish on reboot.)

## Open questions / follow-ups / risks
1. **Storage repair is now on the critical path for every compute spoke.** Until it is fixed, `uv run
   chlu` is dead; the `/tmp` reconstruction recipe in §1/§6 is the workaround (≈4 min to rebuild).
2. **Is N224's own 1.44× reproducible on the cat-test store?** Untested here. If the Hub wants a symmetric
   audit, it is one `exp-cat-test`-style cell per seed.
3. **Should `s` ever be quoted as a scalar for the program?** The evidence says no: 0.3625 – 0.4817 inside
   one sweep, 0.318 on another rig. Recommend the Hub require **`s` + rig + `ball_radius`** on every quote,
   which subsumes the subtraction-convention clause.
4. **Reconciliation item 2 (the 3.72)** needs an owner before r7 lands; it is the only number I found that
   the c6 artifacts do not support.

---

## Proposed handover updates (for the Hub)
**§1.6 / experiments (add a line):**
> **`bprime-s-remeasure` (2026-08-18, results-analyst).** N224's re-measurement is **RUN and CLOSED**.
> `bprime-c6`'s `s` is **already `α‖q‖²`-subtracted**: 21/21 banked cells reproduce **bitwise**
> (max |Δ| = 0.000e+00), so **`s = 0.40` stands and no `d/s` moves**. The un-subtracted ruler's bias on
> this rig is **1.093 ± 0.023 (n = 18)**, **not 1.44×**; N224's own estimator returns **0.4080** on c6's
> store (vs c6's 0.4006, 1.8 % apart) ⇒ the 0.318-vs-0.40 gap is a **store** difference. `R² = 0.995` is
> convention-invariant by construction. The pre-registered "s smaller / d/s larger" direction is
> **falsified**; the *counterfactual* un-corrected reading would have made `d/s` **smaller** (3.588 → 3.255).
**§5 / provenance:** the re-measurement rode a **reconstructed** environment (`/tmp/rmvenv`, JAX 0.9.0
pinned to c6's own version; package tree = `be995ca` + 7 HEAD-filled blobs), validated by bitwise
reproduction; artifacts + scripts at `.claude/outputs/bprime-s-remeasure/`.
**§8 / risks — NEW, escalate to the Head:** iCloud eviction has made **3 490 venv files, 18 tracked
`chlu/*.py`, 552 `.git` objects and several `.claude/outputs/**` artifacts unreadable** (`errno 60`);
`import jax` and `import chlu` both fail in the repo; `git log -- <path>` can return **empty with exit 0**
under this fault (grep, by contrast, hangs then exits non-zero but does not lie). ⭐⭐ And the "**JAX cold-start ≈ 20 min**" environment fact is now **very likely explained, not just
suspect**: on clean local storage `import jax` is **16.5 s**, and the fault's measured signature is
exactly *tens of minutes of blocking per evicted file* across a venv with **3 490** of them. Recommend
the Hub re-test §4's cold-start claim after storage repair rather than carrying it as a property of JAX.
**Registry:** N224 disposition wording proposed verbatim in §7; **`c2w5` never-quote §0.9's clause
"`s = 0.40` FLAGGED, not refuted" needs the word "flagged" retired** (it is now CHECKED AND CLEARED).
**Reconciliation owner needed** for the c6 §1 shipped-row `d/s = 3.72` vs its own
`c6_summary.json` **3.5902** (all five other rows in that column are exact; 3.72 is `sep̄/s̄`).

## Flags
- ⛔ **For `experiment-engineer` (code, low priority, no bug in the physics):** nothing is broken in
  `_well_fit` — but its `s` is **grid-quantised at 0.00966** over `[0.05, 1.2]` and its confinement
  subtraction is **silent** (no flag in the returned value, no record in any artifact of *which*
  convention produced a stored `s`). One cheap hardening: have `well_fits()`/the thirdparty cell dump
  `"s_convention": "alpha_subtracted", "alpha": <confine>` into the JSON, so no future wave has to
  reconstruct a dead venv to answer this question. `effective_s` already documents the hazard in its
  docstring; `_well_fit`'s docstring states the subtracted form in its fit equation but never names it as
  a convention.
- ⚠ **For the Head:** the storage incident (§6) is the highest-severity item in this report, but it is
  **latency, not loss** on the evidence gathered — every file I needed, including the one I had written
  off, eventually read back. The residual risk is `.claude/**`, which is gitignored *and* shows upload
  errors, so it has no verified second copy.
