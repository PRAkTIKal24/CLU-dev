# c2w8p2-compact-atoms — experiment-engineer report (ARM A)

**Task + acceptance criterion (one line):** make C2W8's over-dug store *capture* by bounding how far
each atom's influence reaches, with **K7 (the capture instrument proven able to report a positive)
green and reported before any arm number**, and pass 1's frozen census re-run unchanged on ≥3 seeds.
**Status: DONE.** ⭐ **K7 green; the gate PASSES on all three legs on all three seeds.**

⚠ **DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — needs an owner at the review that accepts
this report), stated in the first 10 lines as required:**
1. **`well_lifecycle.own_foreign_site_depth` hard-codes the Gaussian atom kernel** (lines 114–121), so
   it is *kernel-mismatched* under any compact-atom arm. I did **not** touch it (declared read-only);
   arm A's own/foreign is reported *through it*, labelled. **The Hub owns whether/where to fix it.**
2. **K7 found two properties of the capture instrument that change how a positive G-CAP must be read**
   (a floor of `tol / expansion-rate`, and a false positive at the confinement minimum). They are
   pytest-pinned in my file, but they belong in the census's own documentation, which I may not edit.
3. **A compliance question for the Hub (§3.3): the companion lever `atom_site_local_init`.** The
   compact kernel is *exactly dead* without it. I believe it is compliant with the binding
   prohibition; **the task says to ask rather than decide, so I am asking.**
4. **I appended `ERRATA-C2W8-PASS2.md` §2, §3 and §4** (dated 2026-08-07, each filed *before* the cells
   it governs). They are part of this arm's registration — **§4 in particular records that my own
   attribution prediction was falsified and why the ablation was promoted to census strength.**
5. ⭐ **The lever-separation result (§7.1) is a decision input for the Hub's arm-A-vs-arm-B review:**
   a **Gaussian** at the same co-scaled width **also clears the gate 3/3**. Compactness sharpens
   G-CAP/G-DRIFT; it is not what makes the store capture.

---

## ⭐ DIAL DECLARATION (echoed before the first result)
- **Dial:** none as a new claim — **instrument/mechanism repair on the write side.** ⛔ No paper
  number, no tier-ii verdict, no full-CLU verdict, no I2 verdict, **no performance claim.**
- **Laundering control:** kNN-in-φ launder carried with the byte ledger on every reading
  (`clu_total_bytes` **360 960** vs `knn_launder_bytes` **288** = **1 253×**, unchanged from pass 1 —
  this arm buys **zero** bytes). Per the Head ruling (`ERRATA-C2W8-PASS2.md` §1 Q3) **the pass-2 gate
  is BYTE-BLIND**: no gate leg reads bytes and no performance number is quoted at that ratio.
- **Falsifies:** the gate fails on ≥3 seeds ⇒ compact atom influence does not make this store capture.
  *(It did not fail; see §4.)*
- **Does NOT falsify:** losing to the launder (no performance claim); a high foreign contribution
  (diagnostic); failing to beat arm B.
- ⛔ Depth is **not** quotable as feature importance (§A23.5, ACTIVE).

### Acceptance checklist (task §Acceptance, mechanical)
| # | requirement | where | status |
|---|---|---|---|
| 1 | K7 green **and reported before any arm number**, two-sided, pytest-asserted | §2 | ✅ |
| 2 | K6 OFF bit-identical **and** parameter-count-identical, pytest-asserted | §2b | ✅ |
| 3 | census re-run **unmodified**, all three legs per seed, ≥3 seeds | §4 | ✅ (3 seeds) |
| 4 | own/foreign reported, **explicitly diagnostic-not-target**, both aggregations | §4.1 | ✅ |
| 5 | byte ledger on every arm incl. launder; `(d, atom budget)` as ONE joint dial; γ_φ holes counted | §4.2 | ✅ |
| 6 | full suite green on the branch with the count arithmetic stated | §9 | ✅ 1458 passed / 0 failed |
| 7 | reconciliation list named in the **first 10 lines** | top | ✅ |
| 8 | declared NOT-RUNs listed as NOT-RUNs, never as nulls | §8 | ✅ |

---

## 1. What I did (bullets)

1. Wrote **`PREREG.md`** (`.claude/outputs/c2w8p2-compact-atoms/PREREG.md`) **before** any measured
   cell: K7's predicted values + tolerances, K6, the mechanism prediction **M1**, the config-selection
   protocol, and the Hub's P1–P5 restated (not re-derived).
2. **Built K7 first and ran it first.** `tests/test_compact_atoms.py`, two-sided, on the **read-only**
   shipped instrument.
3. Built the mechanism: a selectable **atom influence profile**
   (`chlu/core/memory_potentials.py::atom_profile`) with two **compact** kernels, behind a flag whose
   OFF path is bit-identical and parameter-count-identical (K6).
4. Measured **M1** (compact atoms on the shipped scattered init are *exactly* dead) and, because it
   held, built + flagged the companion lever `atom_site_local_init`.
5. Ran the **frozen** census (`exp_well_lifecycle.run_census_cell`, unmodified, on unmodified
   `chlu/core/well_lifecycle.py`) on a 4-point pilot grid (seed 7, disjoint) and then on **seeds
   0/1/2** at the selected configuration.
6. Ran two lever-separation ablations (registered in `ERRATA-C2W8-PASS2.md` §3 before running), one of
   which falsified my own prediction and was therefore promoted to **census strength** (§4, also
   registered first).

---

## 2. ⭐ K7 FIRST — the capture instrument CAN report a positive (reported before any arm number)

`pytest tests/test_compact_atoms.py -q` → **13 passed in 27.10 s** (commit `009d49b`).

| leg | construction | **pre-registered** | **measured** | verdict |
|---|---|---|---|---|
| K7-1 | synthetic `relax_fn`, basin `R_true = 0.37`, `r_hi=1, steps=12, tol=0.01` | 0.37 ± 1 bisection cell (2.44e-4) | **0.369873** (err **1.27e-4**) | ✅ |
| K7-3 ⭐ | **planted store**, real `CluSystem`; two identical wells at `(±0.30, 0)`, `D=1.0`, `s=0.15`, unused groups flattened; `n_dirs=64, steps=10, tol=σ_q=0.15` | **0.30** (analytic: symmetry puts the separatrix exactly on `x=0`), band **[0.20, 0.40]** | **0.30859** (settled point 0.0009 from the planted site) | ✅ |
| K7-4 ⭐ | **planted flat** site at `‖z‖=0.6`, `depth=1e-9` ⇒ `V = α‖q‖²` only | **exactly 0.0** | **0.0** | ✅ |
| K7-2b | synthetic escape-everywhere map | exactly 0.0 | **0.0** | ✅ |
| K7-2 | synthetic *expanding* map `z + 5(x−z)`, `tol=0.01` | I predicted **exactly 0.0** | **0.001953** = `tol/λ` to one bisection cell | ❌ **my prediction FALSIFIED — and it is a finding** |
| K7-5 | planted flat site **at the origin** (= the confinement minimum) | ≥ 0.9·`r_hi` (declared false positive) | **0.99902** | ✅ (limitation confirmed) |

⭐ **K7 verdict: the instrument fires on a planted basin and returns exact zero on a planted flat
site. A positive `capture_radius` is now evidence.** Pass 1's `0.000 on 47/48` was a measurement, not
an instrument failure.

⚠ **Two instrument properties nobody had (reconciliation items 2):**
- **The reading has a FLOOR of `tol / expansion-rate`, not 0** (K7-2). The bisection asks only whether
  the relaxed point lands within `tol` of the site, so **a site whose relaxation barely moves reports a
  positive radius with no basin at all.** At the census operating point `tol = σ_q = 0.15`. ⇒ a
  majority-positive G-CAP must always be read beside `λ_min` at the relaxed site and the *magnitude*
  of the radius. (For arm A both are unambiguous: `λ_min` at the relaxed site is **9.85–55.75** (pass 1:
  0.79–8.87) and the **median** radius is **0.293–0.430 = 2.0–2.9 × σ_q**; the per-well spread of the
  non-zero radii is 0.020–0.551. Not a floor artefact.)
- **A flat site at the confinement minimum is a false positive of ≈ `r_hi`** (K7-5). Benign in this
  census only because real φ-sites sit at ‖z‖ ≈ 0.5–1.0.

## 2b. K6 — OFF is bit-identical AND parameter-count-identical
`atom_kernel="gaussian"` returns the literal pre-pass-2 expression: asserted **bitwise** on `V(q)`
(`np.asarray(a(q)) == np.asarray(ref)`), on all store leaves and on the settled read `q_star` of a
full admit+write+read with the flags explicitly OFF, and `CluSystemConfig().as_flag_table() == {}`.
Parameter count identical across all three kernels and across `atom_site_local_init` on/off (both new
fields are `eqx.field(static=True)` / plain floats ⇒ **0 parameters, 0 bytes**).

---

## 3. The mechanism

### 3.1 The kernel (the designed lever)
`chlu/core/memory_potentials.py::atom_profile(d2, s, kernel, cutoff)`, `R = cutoff · s`, `t = r/R`:

| kernel | form | support | smoothness | write gradient outside `R` |
|---|---|---|---|---|
| `gaussian` (default, shipped) | `exp(−r²/2s²)` | **global** | `C^∞` | never zero (the tail) |
| **`wendland`** (the arm) | `(1−t)⁴(1+4t)` | **compact, `r ≤ R`** | `C²` (`φ'(R)=φ''(R)=0` ⇒ **force continuous**) | **exactly 0** |
| `truncated_gaussian` | `(e^{−r²/2s²} − e_R)/(1 − e_R)` | **compact** | `C⁰` only (**force jumps at R**) | **exactly 0** |

All satisfy `φ(0)=1`, so the depth parameterisation is untouched. Measured and pytest-pinned: value
**and** gradient are **exactly** `0.0` beyond `R` (not 1e-30); `|φ'|` at `R−1e-4` is `<1e-3` for
wendland and `>0.5` for truncated_gaussian. ⚠ **What the truncation does to the write:** an atom
further than `R` from every sampled point has **identically zero** gradient w.r.t. `amp`, `centers`
and `log_width` — it is inert *permanently* and can never be recruited. That is not a hypothesis; it
is M1 below.

### 3.2 Width co-scaling — measured, never hardcoded
`s = atom_width_frac_spacing × median_nn_task1` **of that seed's own run**, recovered inside the cell
as `d_safe / d_safe_frac` (the census computes `d_safe` from its own task-1 φ keys). Census values:
**s = 0.2111 / 0.2063 / 0.2202** for spacings **0.1407 / 0.1375 / 0.1468** (frac = 1.5), support
`R = 2.5 s` = **0.528 / 0.516 / 0.551** ≈ **3.75 key spacings, then exactly zero.** Pass 1's Gaussian
sat at `s = 0.3` fixed, with an infinite tail (1e-3 of depth still at `r = 1.11` ≈ 7.9 spacings).

### 3.3 ⚠ The companion lever, and the compliance question for the Hub
**M1 (pre-registered, then measured, `.claude/scratch/c2w8p2-armA/m1probe.py`, 3 designed-site writes
at `addr_dim=8`):**

| store | nearest own atom to the site | support `R` | own atoms in support | fitted depth |
|---|---|---|---|---|
| pass-1 Gaussian, `s = 0.3` | 0.0638 | — (global) | — | 0.4765 / 0.3605 / 0.0164 |
| **wendland, `s = 0.0703`, scattered init** | **1.088** | 0.176 | **0 / 512** | **0.000 / 0.000 / 0.000** |
| wendland, `s = 0.0703`, site-local init | 0.0281 | 0.176 | 512 / 512 | 0.1622 / 0.1682 / 0.1712 |
| wendland, `s = 0.1407`, site-local init | 0.0523 | 0.352 | 512 / 512 | 0.2914 / 0.3890 / 0.3910 |

⇒ **A co-scaled compact atom on the shipped scattered init is EXACTLY dead** (`ERRATA-C2W8.md` §3
measured the nearest of *all* 8 192 atoms at 0.738; a co-scaled support is 1.4–4.2× smaller). So the
arm is **only runnable** with `atom_site_local_init`: at **admission**, the admitted slot's atom
**centers** are re-drawn in a ball of radius `s` around the item's own site — the N98 localized init
moved from build time (where a φ-addressed stream cannot use it, `ERRATA-C2W8.md` §3) to the only
moment the address is known.

**Why I hold it compliant with the binding prohibition** (and why I am asking anyway):
it initialises **atom parameters**, not the attractor; the write objective is *already* handed the
target site `c_i` (`DesignFreedomPotential` honesty note 2), so it hands the optimizer **nothing it
did not already have**; the write then learns depth/width/center; the settled point is whatever the
landscape does (measured drift is **not** identically 0 — see §4, seeds 1/2 carry wells at 0.14–0.44);
basins stay free to interact (foreign contribution is non-zero: 0.12–0.35). Nothing pins, snaps or
regularizes the attractor toward `φ(item)`. It ships behind **its own flag**, separate from the
kernel, so the Hub can score the two levers apart — and §7's ablations do exactly that.

---

## 4. ⭐ THE GATE — pass 1's census, re-run UNCHANGED, 3 seeds

**The census is frozen and I did not touch it.** `chlu/core/well_lifecycle.py` and
`chlu/experiments/usage_telemetry.py` are **unmodified on my branch** (`git diff main --stat` in §9).
Every cell **is** `exp_well_lifecycle.run_census_cell`, called unmodified; the *only* substitution is
the store-config factory (`ewl.store_config`), bound once at import, scoped by `try/finally`, and
recorded in the artifact. That is what an arm is allowed to change.

| leg | criterion | **seed 0** | **seed 1** | **seed 2** | pass 1 |
|---|---|---|---|---|---|
| **G-CAP** | `capture_radius > 0` on a **majority** of live wells | **16/16 = 1.000** ✅ | **15/16 = 0.938** ✅ | **15/16 = 0.938** ✅ | **1/48** |
| — strict leg (`≥ σ_q = 0.15`) | reported beside it (K7 floor) | 13/16 | 11/16 | 13/16 | 0/48 |
| — median radius | | 0.4297 | 0.2930 | 0.3633 | 0.000 |
| **G-DEC** | self-probe `decode` > chance (0.0625) by 2 SE (SE = 0.0214, `n_probed`=128) | **0.1484 = +4.02 SE** ✅ | **0.1641 = +4.75 SE** ✅ | **0.1406 = +3.65 SE** ✅ | **exactly at chance, 3/3** |
| **G-DRIFT** | median `site_drift` < measured key spacing | **0.00100** vs 0.1407 (ratio **0.007**) ✅ | **0.00652** vs 0.1375 (**0.047**) ✅ | **0.00150** vs 0.1468 (**0.010**) ✅ | 0.665–0.925 vs 0.14 |
| **all three, same seed** | | ✅ | ✅ | ✅ | ✗ |

⭐ **`gate_pass = true`, 3/3 seeds, computed mechanically by `exp_capture_armA.gate_legs`**
(artifact: `.claude/outputs/c2w8p2-compact-atoms/capture_armA.json`).

**Against the Hub's registered priors (P1–P5, restated not re-derived):**

| # | Hub's prior (arm A) | measured | outcome |
|---|---|---|---|
| P1 | G-CAP fraction **0.35–0.75**; P(majority) = 0.45 | **0.938 – 1.000** | **above the registered band** — cleared |
| P2 | `decode` **0.08–0.20**; P(>chance @2SE) = 0.50 | **0.141 / 0.164 / 0.141** | **inside the band**, cleared at 3.6–4.8 SE |
| P3 | median drift **0.05–0.15**; P(<spacing) = 0.55 | **0.0010 / 0.0065 / 0.0015** | **below the band** (better) — cleared |
| P4 | all three legs, same arm, ≥3 seeds: **P = 0.35** | **3/3 seeds** | cleared |
| P5 | own/foreign (**diagnostic**): foreign>own on **< 24/48** | **4 of 48** | cleared (pass 1: **45/48**) |

Supporting numbers (same cells): self-probe `acq`/`strict` **0.484 / 0.414 / 0.492** (pass 1:
0.086 / 0.117 / 0.078); `λ_min` at the relaxed site **9.85 – 55.75** (pass 1: 0.791–8.873);
`depth_raw_median` **0.606 / 0.683 / 0.581**, netted 0.716 / 0.843 / 0.673 (B1). Census `P` =
0.125 / 0.0625 / 0.3125 and `M` = 0.183 / 0.200 / 0.225; monitor **#3 `vacuous_gate` still trips 3/3**
— that is pass 1's K1/K9 defect, untouched and **not mine to fix**. ⛔ I state **no** stage-2 unlock
verdict.

### 4.1 own/foreign — ⛔ DIAGNOSTIC, never a target, never a gate leg
Both aggregations, each labelled (`ERRATA-C2W8-PASS2.md` §1 Q2; the **median** is canonical):

| seed | own **median** | foreign **median** | own **mean** | foreign **mean** | foreign > own |
|---|---|---|---|---|---|
| 0 | **0.7703** | **0.1233** | 0.7631 | 0.1944 | **0 / 16** |
| 1 | **0.7979** | **0.3532** | 0.8535 | 0.3603 | **1 / 16** |
| 2 | **0.7311** | **0.3407** | 0.7325 | 0.3351 | **3 / 16** |
| *pass 1 (0/1/2)* | *0.518 / 0.282 / 0.123* | *1.261 / 0.947 / 0.611* | *0.472 / 0.327 / 0.169* | *1.189 / 0.929 / 0.562* | *16 / 15 / 14* |

The item's own well went from a **minority** of the landscape at its own site to a **majority**
(**45/48 → 4/48** wells with foreign > own). ⛔ **This was not tuned on and no gate leg reads it**;
under a factored store a high foreign contribution would be the *signal*, so it is reported and not
optimised. ⚠ **The estimator is kernel-mismatched** (reconciliation item 1): the frozen census
computes it with a hard-coded Gaussian sum, so under a compact kernel it **over-reads** both legs.

### 4.2 Byte ledger — every arm, launder included; **(d, atom budget) as ONE joint dial**

| column | arm A (all 3 seeds) | pass 1 |
|---|---|---|
| `clu_store_bytes` | 360 448 | 360 448 |
| `clu_codebook_bytes` | 512 | 512 |
| **`clu_total_bytes`** | **360 960** | **360 960** |
| `knn_launder_bytes` | 288 | 288 |
| ratio | 1 253× | 1 253× |
| `gamma_phi_hole_bytes` (γ_φ holes) | **0** (`gamma_phi = False`; no hole placed) | 0 |

**Joint dial:** `addr_dim = 8` ⇒ `n_atoms = max(32·16, 384, round(512·√2⁸)) = 8192`, 44 bytes/atom.
**Arm A buys exactly zero bytes** — the kernel and the init flag are static/parameter-free. ⛔ The gate
does not read this column and **no performance number is quoted at 1 253×**.

---

## 5. The pilot (config selection — declared, on seeds disjoint from the census)

Grid registered in `PREREG.md` §5 as `frac ∈ {0.5, 1.0}` and extended once to `{1.5, 2.0}` in
`ERRATA-C2W8-PASS2.md` §2, **filed before those cells ran**. All on **pilot seed 7 only** (declared
seed cut; spacing 0.1081).

| frac | `s` | `R = 2.5s` | cap>0 | cap ≥ σ_q | median cap | `decode` (SE) | median drift | wells with drift<0.01 | wells at `λ=2α` (bare bowl) | depth | own med | foreign med | G-CAP | G-DEC | G-DRIFT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.0541 | 0.135 | 16/16 | 0 | 0.129 | 0.0625 (+0.00) | 0.6307 | 1 | 15 | 0.153 | 0.193 | 1.3e-10 | ✅ | ✗ | ✗ |
| 1.0 | 0.1081 | 0.270 | 16/16 | 16 | 0.250 | 0.0938 (+1.46) | 0.6307 | 6 | 10 | 0.282 | 0.358 | 0.0020 | ✅ | ✗ | ✗ |
| **1.5 (selected)** | 0.1622 | 0.405 | 16/16 | 16 | 0.332 | **0.1797 (+5.48)** | **0.0007** | 11 | 0 | 0.419 | 0.542 | 0.0529 | ✅ | ✅ | ✅ |
| 2.0 | 0.2162 | 0.540 | 16/16 | 14 | 0.244 | **0.2031 (+6.57)** | **0.0011** | 14 | 0 | 0.489 | 0.643 | 0.2170 | ✅ | ✅ | ✅ |

**Selection rule, declared:** both 1.5 and 2.0 clear all three legs, so the tie-break is (i) the
**strict** G-CAP leg (`capture ≥ σ_q`: **16/16 at 1.5** vs 14/16 at 2.0) and (ii) the arm's own thesis
— **the tightest reach that clears the gate**. ⛔ Nothing was selected on own/foreign.

⭐ **The pilot's own finding, and it is the sharpest mechanism result in this report — a compact atom
turns the saddle-reach condition into a hard wall.** At frac ≤ 1.0 `site_drift` was **bimodal**: some
wells at exactly 0.0 with `λ_min ≈ 42–50`, the rest at 0.45–1.18 with `λ_min = 0.1000 = 2α` **exactly**
— the bare confinement bowl, i.e. *no well at all at the relaxed point*. The split is **not** erosion:
the census launches `_relaxed_sites` from the address with the **payload channel at 0**, while the
well sits at payload `a_i = (label−4.5)/9 ∈ [−0.5, +0.5]`. With a **compact** atom the force at that
launch point is **exactly zero** whenever `|a_i| > R`, so the read cannot feel its own well at all.
`R` crosses `max|a_i| = 0.5` between frac 1.5 and 2.0, and that is exactly where the bimodality
disappears (10 bare-bowl wells → 0). **G-CAP can pass while G-DRIFT fails, for a reason that has
nothing to do with the well and everything to do with the launch manifold.** Registered in
`ERRATA-C2W8-PASS2.md` §2 *before* the 1.5/2.0 cells ran; the registered predictions came out at
frac 2.0 = drift-0 ≥14/16 → **14** ✅, decode ≥0.15 → **0.203** ✅, foreign 0.02–0.20 → **0.217**
(just above the band, honest miss); at frac 1.5 = drift-0 12–14 → **11** (one below, honest miss),
decode 0.13–0.22 → **0.180** ✅, foreign 0.005–0.05 → **0.053** (just above, honest miss).

---

## 6. How I verified (commands + observed output)

All runs: main venv (`/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0**) with `PYTHONPATH` set to the
worktree `/Users/user/Desktop/CHLU-c2w8a` (protocol §4's preferred worktree recipe — **no worktree
`uv sync`**, so no package drift).

```
python -m pytest tests/test_compact_atoms.py -q      -> 13 passed in 27.10s
python .claude/scratch/c2w8p2-armA/k7probe.py        -> K7 numbers in §2
python .claude/scratch/c2w8p2-armA/m1probe.py        -> M1 table in §3.3
python -m chlu.experiments.exp_capture_armA --quick  -> smoke, 40s, artifact written
python -m chlu.experiments.exp_capture_armA --seeds 7 --width-frac {0.5,1.0,1.5,2.0}   (pilot)
python -m chlu.experiments.exp_capture_armA --seeds {0,1,2} --width-frac 1.5           (census)
python .claude/scratch/c2w8p2-armA/merge_census.py ... -> gate_pass True CAP 3 DEC 3 DRIFT 3 of 3
python -m ruff check chlu/ tests/test_compact_atoms.py -> All checks passed!
python -m pytest -q -p no:randomly                   -> 1458 passed, 0 failed (3113.29s)
```
The three census seeds were run as three parallel processes (wall clock; the cells are independent and
each carries its own seed) and merged with `merge_census.py`, which re-emits the arm-level aggregation
**using the shipped `exp_capture_armA` functions**, so the merged artifact is the arithmetic a single
`--seeds 0,1,2` run would have produced. Per-seed wall time **874 / 876 / 874 s**.

**Flag-provenance table (every non-default flag in effect; identical across the three census seeds
except where the measured spacing enters):**

| flag | value |
|---|---|
| commit | `ebe0592` (branch `agent/experiment-engineer/c2w8p2-compact-atoms`, base `main @ 80d7d4b`) |
| seeds | 0, 1, 2 (census) · 7 (pilot + ablations) |
| `atom_kernel` | **`wendland`** (compact, `C²`) |
| `atom_kernel_cutoff` | **2.5** (⇒ `R = 2.5 s`) |
| `atom_width` | **0.21107 / 0.20631 / 0.22025** = `1.5 × median_nn_task1` (measured per seed) |
| `atom_site_local_init` / `atom_site_local_radius` | **True** / = `atom_width` |
| `addr_dim` (= φ dim) / `payload_dim` | 8 / 1 · `n_atoms` **8192** · `capacity` 16 · `well_budget` 8 |
| `d_safe_override` | 0.12383 / 0.12103 / 0.12921 (= `0.88 × median_nn`) |
| `leak` / `stage_lifetimes` | 0.02 / True · `permanent_per_task` 1 |
| `write_steps` / `read_steps` / `address_steps` | 300 / 800 / 400 · `masked_write` True |
| `query_sigma` (σ_q) / `payload_tol` | 0.15 / 0.1 · `n_query_per_item` 8 |
| `kinetic_mode` / `dt` / `gamma_address` / `gamma_read` | `newtonian_learned` / 0.05 / 0.05 / 0.02 |
| `gamma_phi` / `soft_certificate` | False / False (defaults) |
| φ | `pca`, regime `task1_only`, dataset mnist, scale fixed on the fit pool |
| capture instrument | `capture_dirs` 16, `capture_bisect_steps` 8, `measure_capture` True (pass-1 values, untouched) |
| `promotable` | **False** — `phi_dim = addr_dim = 8` is below the CL entry's binding `phi_dim ≥ 16` (`ERRATA-C2W8.md` §3); inherited from pass 1 and unchanged |

---

## 7. ⭐ Lever separation — and it FALSIFIED one of my own registered predictions

Arm A bundles two levers (compact kernel · site-local init). Both ablations were registered with
predictions in `ERRATA-C2W8-PASS2.md` §3 **before** they ran, on pilot seed 7 (spacing 0.1081, same
`s = 0.1622` throughout, so the only differences are the two flags):

| cell | kernel | site-local init | cap>0 | cap ≥ σ_q | `decode` (SE) | median drift | wells at `λ=2α` | depth | own med | foreign med | f>o | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **arm A** | wendland | ✔ | 16/16 | **16** | 0.1797 (+5.48) | 0.0007 | 0 | 0.419 | 0.542 | 0.0529 | 0/16 | ✅✅✅ |
| **AB1** | **gaussian** | ✔ | 16/16 | 13 | **0.2344 (+8.03)** | 0.0041 | 0 | 0.564 | 0.540 | **0.0296** | 0/16 | ✅✅✅ |
| **AB2** | wendland | ✘ | **0/16** | 0 | 0.0625 (**= chance**) | 0.6307 | **16** | **0.000** | **0.000** | 0.000 | 15/16 | ✗✗✗ |

- **AB2 confirms M1 at the selected width** (registered P = 0.9): a compact atom on the shipped
  scattered init is **exactly dead** — depth 0.000, capture 0 on every well, `λ_min = 0.1000 = 2α`
  (bare confinement bowl) on all 16, `decode` exactly at chance. **Without the companion lever this
  arm does not exist.**
- ⚠ **AB1 FALSIFIES my registered prediction, in the direction that matters.** I registered
  `foreign ≥ 0.5 × own`, `foreign > own on ≥ 8/16` and `decode` **below** arm A's. Measured: foreign
  **0.0296** = 0.055 × own, foreign > own on **0/16**, and `decode` **0.2344 (+8.03 SE) — better than
  the compact arm's 0.1797** on the same seed. **A plain Gaussian at the same co-scaled width, with
  the site-local init, clears all three legs too.**

⭐ **Honest attribution: the reach thesis survives, my attribution of it does not.** What closes the
gate is the **co-scaled width plus the site-local init** — both of which bound reach — and **not** the
exactly-zero support. At this operating point compactness is **not necessary and costs a little
`decode`** (plausibly because a force that is exactly zero outside `R` also removes the pull on a read
launched off its well — the same mechanism as §5's payload-launch wall). ⚠ **One pilot seed is not a
verdict**, so this was registered in `ERRATA-C2W8-PASS2.md` §4 and promoted to census strength — where
it comes out **more balanced than the pilot suggested**:

### 7.1 AB1 at census strength (same 3 seeds, same width; only the kernel differs)
Artifact `ab1_gaussian_census.json`; predictions registered in `ERRATA-C2W8-PASS2.md` §4 first.

| | **arm A (wendland)** s0 / s1 / s2 | **AB1 (gaussian)** s0 / s1 / s2 | who wins |
|---|---|---|---|
| **G-CAP** fraction | **1.000 / 0.938 / 0.938** | 0.875 / 0.750 / 0.813 | **compact** |
| — `cap ≥ σ_q` | **13 / 11 / 13** of 16 | 12 / 8 / 10 | **compact** |
| — median radius | **0.430 / 0.293 / 0.363** | 0.258 / 0.123 / 0.207 | **compact** |
| **G-DEC** `decode` | 0.148 / 0.164 / 0.141 | **0.195 / 0.164 / 0.211** | **gaussian** |
| — margin | +4.02 / +4.75 / +3.65 SE | **+6.21 / +4.75 / +6.94 SE** | **gaussian** |
| **G-DRIFT** median | **0.0010 / 0.0065 / 0.0015** | 0.0146 / 0.0412 / 0.0308 | **compact** (10–20×) |
| **gate (3 legs, 3 seeds)** | ✅ **PASS** | ✅ **PASS** | tie |
| own/foreign median (⚠ *the estimator is Gaussian, so it is exact for AB1 and OVER-reads arm A*) | 0.770/0.798/0.731 vs 0.123/0.353/0.341 | 0.709/0.727/0.589 vs 0.058/0.147/0.114 | not comparable |
| foreign > own | 0 / 1 / 3 (of 48: **4**) | 0 / 2 / 2 (of 48: **4**) | tie |

⇒ **Refined, and this is the version to carry (the one-seed pilot over-called it):** at the selected
operating point **both variants clear the gate 3/3**. The **compact** kernel is better on **G-CAP**
(fraction, strict count and radius) and **10–20× better on G-DRIFT**; the **Gaussian** is better on
**G-DEC**. So **compact support is not *necessary* for this store to capture — the co-scaled width and
the site-local init are — but it does measurably sharpen the basin and pin the site.**
⚠ The own/foreign columns are **not comparable across the two**: the frozen estimator assumes a
Gaussian kernel, so it is exact for AB1 and **over-reads** arm A (reconciliation item 1, and the
directional half of my registered M2).
*My §4 AB1-census predictions: G-CAP 3/3 ✅ · `decode` 0.18–0.30 → 0.195/0.164/0.211 (**seed 1 below
the band**, all three still clear 2 SE) · drift < 0.02 on 3/3 → **only seed 0** (0.0146); seeds 1/2 at
0.0412/0.0308 **missed my band** while still passing the leg · foreign 0.02–0.15 ✅ · foreign>own ≤4/48
→ exactly **4** ✅.*

---

## 8. Declared NOT-RUNs (⛔ never nulls)

- **merge, prune, depth restoration, every §2.7 claim cell** — not built (they were gated on this
  gate; the gate has only just passed and building them was **not** in my scope).
- **The pass-1 Gaussian baseline was NOT re-run on this branch** — declared cut. K6 asserts the OFF
  path is bit-identical and parameter-count-identical to `main @ 80d7d4b`, so pass 1's `census.json`
  *is* the baseline; re-running it would have cost ~50 min for a bit-identical answer. Every "pass 1"
  row above is quoted from `.claude/outputs/c2w8-well-lifecycle/census.json`.
- **Pilot seed 8** — declared **seed cut** (the task's "cut seeds before cutting a cell"): the pilot
  ran on seed 7 only, 4 grid points instead of 2 seeds × 2 grid points.
- The **factored store / shared well vocabulary**; **I2**; **cross-stream criterion**; **wormholes /
  learned p₀**; **CSF3**; any **tier-ii / full-CLU / I2 verdict**; any **performance claim**; any
  **stage-2 unlock verdict** (I report `P`/`M`, not K1).
- **A CLI hook** for `exp_capture_armA` — deliberately **not** added: `chlu/cli/experiment_cmd.py` is
  outside my declared file ownership and both concurrent pass-2 spokes would collide in it. The
  experiment runs as `python -m chlu.experiments.exp_capture_armA`. Trivial for the Hub to add once.

---

## 9. Git footprint

- **Branch:** `agent/experiment-engineer/c2w8p2-compact-atoms` (protocol §3 naming), **worktree
  `../CHLU-c2w8a`**, base **`main @ 80d7d4b`**. ⚠ The task file names the branch
  `c2w8p2-compact-atoms`; I used the protocol form and **also point a plain ref `c2w8p2-compact-atoms`
  at the same commit** so either name resolves. Nothing pushed; no PR.
- ⚠ **Collision avoided:** the shared main checkout was on another agent's branch
  (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` @ `7fcef50`) for the whole session. I never
  edited it and worked exclusively in the worktree.

| commit | subject |
|---|---|
| `5f2611d` | `[experiment-engineer] ARM A: bound how far each atom's influence reaches` |
| `009d49b` | `[experiment-engineer] K7: prove the capture instrument can report a positive` |
| `ebe0592` | `[experiment-engineer] exp_capture_armA: pass 1's census, arm A's store` |

**Files touched (`git diff --stat main..HEAD`):**
```
 chlu/config.py                       |  53 +++++      (additive: ExperimentCaptureArmAConfig + 3 wiring lines)
 chlu/core/clu_system.py              |  59 ++++++      (4 config flags + 1 wiring call + _localize_slot_atoms)
 chlu/core/memory_potentials.py       | 119 ++++++-     (atom_profile, localize_group_atoms, 2 forwarded kwargs)
 chlu/experiments/exp_capture_armA.py | 385 ++++++++++  (new)
 tests/test_compact_atoms.py          | 311 ++++++++++  (new)
 5 files changed, 926 insertions(+), 1 deletion(-)
```

**Files I did NOT touch, verified:** `chlu/core/well_lifecycle.py`, `chlu/experiments/usage_telemetry.py`,
`chlu/core/friction_field.py`, `chlu/core/soft_certificate.py`, `chlu/core/emission_head.py`,
`chlu/experiments/cl_baselines.py`, `chlu/experiments/exp_well_lifecycle.py`, the C2W6/C2W7 files.

**Full suite (on the branch, main venv, worktree `PYTHONPATH`):** **`1458 passed, 0 failed, 36 warnings in 3113.29 s`**.
**Count arithmetic, verified by collection diff rather than asserted:** base `main @ 80d7d4b` collects
**1445**; my branch collects **1458**; `diff` of the two `--collect-only -q` lists shows **exactly the 13
new `tests/test_compact_atoms.py` node-ids added and nothing removed or renamed** (1445 + 13 = 1458).
⚠ The handover's standing figure is "1443 passed" — the base **collects 1445** here, so that line is
stale by 2 (I did not spend 52 min re-running the base to confirm which 2; flagged, not claimed).

### 9.1 Artifacts (all under `.claude/outputs/c2w8p2-compact-atoms/`)
`PREREG.md` (filed first) · **`capture_armA.json`** (the arm, 3 census seeds, merged) ·
**`ab1_gaussian_census.json`** (the §7.1 ablation, same 3 seeds) · `census_s{0,1,2}/`, `ab1_s{0,1,2}/`,
`ab1_gauss_local/`, `ab2_compact_nolocal/`, `pilot_f{05,10,15,20}/` (raw per-cell artifacts + logs) ·
`k7probe.py`, `m1probe.py`, `summarize.py`, `merge_census.py` (the scripts behind every table here).

---

## 10. Open questions / follow-ups / risks

1. ⭐ **The compliance question (§3.3).** `atom_site_local_init` is load-bearing — without it the arm
   is *exactly* dead. I hold it compliant (it initialises atom parameters, not the attractor) and I
   have shipped it behind its own flag with the ablations to price it, **but the Head's prohibition is
   binding and the task says to ask.** ⛔ **If the Hub rules it non-compliant, arm A's gate pass does
   not stand and must be withdrawn**, because the compact kernel alone is dead (M1).
2. **The gate passing is not retrieval.** `decode` 0.141–0.164 is 3.6–4.8 SE above chance and *far*
   from 1.0; `acq/strict` ≈ 0.41–0.49. The store now **has basins and holds its sites**; it is not yet
   a good memory. ⛔ No performance claim is made and the launder comparison stays off the table.
3. **Two levers, one arm — now priced at census strength (§7.1).** Both variants clear the gate 3/3;
   compact wins G-CAP and G-DRIFT, Gaussian wins G-DEC. ⇒ **the wave should not carry "compact atoms"
   forward as *the* fix**; the fix is bounded reach = co-scaled width + site-local init, and
   compactness is an optional sharpener whose price is a little `decode`. The Hub may prefer the
   simpler Gaussian variant for arm A's race against arm B; both artifacts are in the output dir.
4. **Reach vs. crowding is now the live trade-off.** The selected `R = 3.75` key spacings is set by the
   **payload** launch gap (`R > max|a_i|`), not by the address geometry; foreign contribution rises
   monotonically with it (1.3e-10 → 0.002 → 0.053 → 0.217 at fracs 0.5/1.0/1.5/2.0). An **anisotropic**
   support (tight on the address axes, wide on the payload axis) would decouple them; the machinery
   already exists (`AtomDictionaryPotential.axis_width_scale`, w26) but is **not wired** into
   `LearnedVStore`. That is the natural next mechanism and I did not build it.
5. **`theta_att` is now high** (0.597–0.970) because it is the max fitted depth among *non*-capturing
   wells and the non-capturing minority is now deep, so `is_attractor` (and hence `P`) is gated by a
   floor that rises with the arm's own success. That interaction did not matter when nothing captured;
   it does now, and it is worth a look before any prune verb is priced off `P`.
6. `well_lifecycle.own_foreign_site_depth`'s hard-coded Gaussian (reconciliation item 1) will also
   mis-read **arm B** if arm B changes the well's functional form.

---

## Proposed handover updates (for the Hub)

- **§7 (Known Issues), new:** *"`chlu/core/well_lifecycle.own_foreign_site_depth` hard-codes the
  Gaussian atom sum, so the census's own/foreign diagnostic is kernel-mismatched for any arm that
  changes the atom profile (C2W8 pass-2 arm A ships `wendland`/`truncated_gaussian`). Diagnostic only
  — no gate leg reads it — but it over-reads both legs under a compact kernel."*
- **§7 (Known Issues), new:** *"`soft_certificate.capture_radius` has a **floor of `tol /
  expansion-rate`**, not 0: a site whose relaxation barely moves reports a positive radius with no
  basin, and a flat site sitting at the confinement minimum reports ≈`r_hi`. Both are pytest-pinned in
  `tests/test_compact_atoms.py`. Read every `capture_radius > 0` beside `λ_min` at the relaxed site and
  the radius magnitude."*
- **§7.24 update:** the dead `atom_local_radius` lever now has a live counterpart —
  `CluSystemConfig.atom_site_local_init` performs the N98 localization **at admission time**, which is
  the form a φ-addressed stream can actually use (`ERRATA-C2W8.md` §3 recorded why the build-time form
  cannot be). N111/N211's warning (a localized init is actively harmful in a streaming block at a stale
  target) is *not* contradicted here — this init is at the item's own live address.
- **§3 (Config) additions, all defaults preserving current behaviour:**
  `CluSystemConfig.atom_kernel = "gaussian"`, `atom_kernel_cutoff = 2.5`,
  `atom_site_local_init = False`, `atom_site_local_radius = 0.0`; new
  `ExperimentCaptureArmAConfig` (`config.experiment_capture_arm_a`, YAML block
  `experiment_capture_arm_a`).
- **§8 (Open Directions) / arm-A follow-up:** the *anisotropic* support (tight on the address axes,
  wide on the payload axis) is the natural next mechanism — `AtomDictionaryPotential.axis_width_scale`
  exists (w26) but is **not wired** through `DesignFreedomPotential` / `LearnedVStore`. It would
  decouple the reach the read needs (payload) from the reach crowding forbids (address), which is the
  trade-off §5 and §7.1 both bottom out on.
- **§10 running log, one line:** *C2W8 pass 2 arm A — **K7 green** (planted basin 0.30 recovered as
  0.3086; planted flat site exactly 0.0) and the **capture gate passes 3/3 seeds on all three legs**
  (G-CAP 0.94–1.00 vs 1/48 at pass 1; `decode` 0.141–0.164 = +3.6…+4.8 SE vs exactly-at-chance;
  median site drift 0.001–0.007 vs a 0.14 key spacing). foreign>own fell **45/48 → 4/48**
  (diagnostic). Zero bytes bought. ⚠ **Its own registered ablation shows a plain Gaussian at the same
  co-scaled width + site-local init ALSO passes 3/3** (better `decode`, worse capture/drift), so the
  load-bearing lever is **bounded reach (width + localized init), not compact support**; and the arm
  **carries an open compliance question on the companion lever**.*
