# c2w8p3-capture-strong-phi — experiment-engineer report

**Task + acceptance criterion:** re-run the **frozen census + the COMPLETED gate**
(G-CAP·G-DEC·G-DRIFT·**G-ADDR**) with arm A's **co-scaled-width** store on the **strong-φ
Split-CIFAR-10** rig at **d = 12 / 32 768 atoms**, with an **internal PCA-φ reference arm at the same
d in the same run**, and state which pre-registered branch fired. **Status: DONE.** ⭐ **Branch (b)
NO DAYLIGHT, 3/3 arms, 0/9 cells** — the pre-registered outcome (Hub Q6 = 0.70), reported as a
**finding**. ⛔ **The completed gate FAILS on all three arms.** ⭐ **The store is NOT inert at d = 12
on CIFAR φ (9/9 cells).**

> ## ⚠⚠ DOWNSTREAM RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary). Six items.
> 1. ⛔⛔ **THE STRONG FITTED ENCODER IS THE WORST ARM, BEYOND 2 SE, ON EVERY LEG.** Paired on the
>    bit-identical stream: `simclr − randconv` **A1 = −0.1406 ± 0.0508 (0/3 seeds simclr better)**,
>    `simclr − pca` **= −0.1276 ± 0.0589 (0/3)**, while `randconv − pca` = +0.0130 ± 0.1017 (**tied**).
>    ⇒ the banked CL-accuracy gain of `simclr` (**0.31912** vs `randconv` **0.213** = **+0.106**, vs
>    `pca` **0.16080** = **+0.158**; ⚠ that banked `pca` row is at **`phi_dim = 32`**, not at my
>    reference arm's `d = 12`) does **not** transport to addressability — **it inverts.** This
>    **upgrades wt2's item 2** from "address geometry and task
>    accuracy are DECOUPLED" to **"on the census they are ANTI-correlated"**, and it is measured on
>    the store, not on the geometry. **Every description of "strong φ" in this wave needs this
>    qualifier.**
> 2. ⛔⛔ **THE COMPLETED GATE'S SCORE IS ALMOST A MONOTONE FUNCTION OF SETTLE-COLLAPSE.** Across all
>    **9 cells**: Spearman **ρ(A1, G-DRIFT ratio) = −0.967**, **ρ(A1, settle↔launder agreement) =
>    +0.933** (Pearson −0.877 / +0.907). **That is the D2a signature at the level of the whole
>    design, not one cell** (§A29.6). ⚠ The per-arm mechanical flag `best_is_also_lowest_drift` is
>    **False on all three arms** — it is decided by ties of 0.004 in drift and **hides the effect**.
>    ⇒ **the flag as specified is too weak; the gate needs a two-sided drift leg or a drift floor.**
>    Owner needed.
> 3. ⛔ **A1's THRESHOLD IS DISCRETE AND THE BEST ARM SITS ON IT.** `randconv` scores **31/31/29 of
>    128** against a threshold of **32/128** — it fails **by ONE read** on 2 of 3 seeds. A *legal,
>    scale-covariant* rescale (`a = 0.8`, payload co-scaled) moves A1 by **+0.0469 (≤ the registered
>    0.05 guard, which HOLDS)** and yet **flips the leg's verdict False → True.** ⇒ **the §4 guard
>    passes on its registered metric while the gate's verdict is not stable to a rescale it declares
>    legal.** Hub call.
> 4. ⚠ **THE CUE IS NOT EQUALLY HARD ACROSS THE ARMS IT COMPARES.** `κ_q` is dimensionless against
>    the rig's **sizing** spacing, but the spacing a read must beat is the **codebook** spacing, and
>    the two differ by an **arm-dependent** factor (1.08 `pca` / 1.14 `simclr` / **1.41** `randconv`)
>    ⇒ `cue_sigma / codebook_spacing` = **0.927 / 0.875 / 0.710**, a **30 % spread**. This qualifies
>    every cross-arm A1/A3a comparison in this report (**it does not rescue `simclr`: `simclr` has an
>    EASIER cue than `pca` and still scores below it**). Filed against my own PREREG §1 in `ERRATA.md` §2.
> 5. ⚠ **wt1's item 3 reproduces on CIFAR:** banked `frac_never_read = 1.0000` on **9/9** cells while
>    the settle-side **A2 = 0.125–1.000**. The launch-point statistic is again a constant.
> 6. ⚠ **wt2's item 3 reproduces on the census rig:** monitor #3's refusal rate is **0.000–0.111**
>    (mean **0.0385**, **5/9** cells non-zero), **not identically 0.000**.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result

- **Dial:** **none as a new claim — a COMPONENT BUILD** measuring whether the physics adds anything
  once the encoder is not the bottleneck. ⛔ **No paper number. No tier-ii verdict** (no organizer
  swap exists here). ⛔ **No full-CLU verdict** (§A28.4). ⛔ **No arm-race adjudication** (§A30.1: the
  arm A/arm B race is VOID as a comparison and stays unadjudicated — nothing below ranks them).
- **Laundering control:** the **same-keys kNN-in-φ launder in the SAME projected φ**, on **every one
  of the 9 cells**, asserted bit-identical in code (9/9), with the byte ledger beside it (φ params +
  projection params on **every** arm **including the launder**). **Every quotation below is
  matched-ITEMS; matched-bytes is NOT met** — ratios in §6, and pass 1's **1 253×** stream-launder
  caveat travels unchanged.
- **Falsifies:** nothing — **both branches were pre-registered as reportable**; the non-compliant act
  would be *tuning away* branch (b). It was not tuned away: it is the headline.
- **Does NOT falsify:** losing to the kNN launder on the **metric-native cue protocol** — 1-NN over
  the stored keys is the Bayes rule there (metric-native-ceiling theorem, not news).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94: **every reading here is
  NON-PROMOTABLE**; no CL benchmark entry is claimed.

---

## 1. Both preconditions verified by me, on disk, before any cell — and the label they carry

| file | required field | **read by me** |
|---|---|---|
| `.claude/outputs/c2w8p3-gate-addr/GATE-ADDR-VALIDATED.json` | `gate_addr_validated == true` | ✅ **true**, and all **9** validation checks true, incl. `N1_armB_banked_config_fails`, `N2_planted_permutation_scores_zero_and_fails`, the three scale checks, and **`R3_attractor_can_move_off_the_key = true`** |
| `.claude/outputs/c2w8p3-phi-geometry/results/PHI-GEOMETRY.json` | `geometry_go` (either value) | ✅ **`geometry_go = true`**; `d_favoured_by_geometry = 16`; **`d_recommended_operational = 12`**; `store_inert_by_d = {8: false, 12: false, **16: true**}` |

⭐ **`geometry_go = true` ⇒ Head ruling R4's NO-GO re-label does NOT apply.** The spine ran on a
substrate **measured in advance to have improved separability** (σ_q/spacing 0.334 → 0.210 at d = 12,
3/3 seeds). **That makes this report's null *attributable*, not confounded** — and it makes the
inversion in reconciliation item 1 sharper, not weaker.

**The joint dial, declared as ONE dial:** **`d = 12` ⇔ 32 768 atoms** (`n_atoms = round(512·√2^d)`),
**achieved on 9/9 cells** (`budget_honoured = true`). ⛔ **`d = 16` is a declared NOT-RUN**, not a
null: wt2 measured the store **inert** there (median fitted depth **5.44e-7** at a fully honoured
131 072-atom budget), and an inert store makes a census vacuous **for a reason that is not the gate's
reason** (Head ruling R1's attached risk; R3 resolved by measurement).

## 2. ⚠ FIRST-10-LINES REQUIREMENT (Head ruling R1): the store's WELL DEPTH, reported FIRST

**`CluSystem` had never been run on CIFAR φ. It digs — on every cell.**

| arm | `depth_raw_median` per seed (0/1/2) | mean | **INERT (< 1e-6)?** | `n_atoms` |
|---|---|---|---|---|
| `simclr` | 0.94214 / 0.88514 / 0.73249 | 0.853 | **no** | 32 768 ✅ |
| `randconv` | 0.96536 / 0.76919 / 0.85656 | 0.864 | **no** | 32 768 ✅ |
| `pca` | 0.74911 / 0.47025 / 0.60117 | 0.607 | **no** | 32 768 ✅ |

⛔ **"Not inert" is not "addressable", and this run is the sharpest available demonstration:**
`simclr` seed 0 digs to **0.942** and has **15 of 16 items with capture radius exactly 0.0**,
`any_basin_rate = 0.0000`, **A1 = 0.0000**. **Deep wells, no basins** — the `N1′ narrow wells`
designed negative that `c2w8p3-gate-addr` had to *plant*, occurring here **spontaneously on the
primary strong arm**.

## Flag-provenance table (governs every number in this report)

| item | value |
|---|---|
| branch / base | **`c2w8p3-capture-strong-phi`**, base **local `main @ 18b4205`** (the integrated base carrying both preconditions' merges), worktree `../CHLU-c2w8p3c` |
| commits | `0f057a0` the spine · `d7eabdd` the tests (**`chlu/` is byte-identical between them**; `d7eabdd` touches `tests/` only) |
| **measuring commit** | **`0f057a0` for every cell** (all 10 runs launched at that tree); `d7eabdd` is the shipped HEAD and changes no number |
| venv | **main venv reused** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`), **no worktree `uv sync`**, no JAX drift. `jax 0.9.0` (recorded in `flags.jax` of the artifact) |
| ⛔ cwd discipline | every run executed with **cwd = the worktree** and provenance printed (`PROVENANCE chlu from /Users/user/Desktop/CHLU-c2w8p3c/chlu/__init__.py`). ⚠ One launch from the main checkout died instantly with `No module named …exp_capture_strong_phi` — `python -m` puts cwd on `sys.path[0]`; recorded because it is a live trap for the next spoke |
| dataset / protocol | **Split-CIFAR-10, reduced protocol** (`apply_cifar10`), 5 tasks × 2 classes, Class-IL, 1000 train / 500 test per task, then `n_fit_region = 25000`, `n_fit_pool = 6000` applied **after** the preset (the `cl-encoder` §10 trap, pinned by a test) |
| ⭐ stream identity | seed 0/1/2 fingerprints **`318a2b8590716f68` / `e7505727a7a2be65` / `41945861de7ae566`**, **identical on all three arms** ⇒ every cross-arm contrast is **PAIRED**. Seed 0 is **bit-identical to wt2's and to `c2w8-cifar-strong-phi`'s** seed-0 stream |
| ⭐ encoder identity | `simclr` `loss_first = 5.512242317199707`, `loss_final = 4.074422359466553`, `steps = 8000`, `phi_param_floats = 225 536` — **bit-identical to the banked `encoder_price.json`** ⇒ **this is the same built-and-priced encoder**, not a refit |
| φ arms | **`simclr`** (`phi_dim = 256`, `enc_steps = 8000`, PRIMARY) · **`randconv`** (`phi_dim = 256`, 0 fit steps, CONTROL) · ⛔ **`pca` at `phi_dim = 12`** — the **INTERNAL weak-φ REFERENCE at matching `d`, measured in THIS run** (Head ruling R1: no pass-1/2 number is a baseline here) |
| φ regime | **`task1_only` only** (binding). `generic_frozen` NOT RUN (declared) |
| the map | **`PhiProjection(form="pca")`** (wt2's), fitted on the `task1_only` pool, frozen; **`identity`** on the `pca` arm (provably neutral: PCA-of-PCA-`d` = PCA-`d`). `assert_no_truncation` raises otherwise |
| **the store arm** | **arm A CO-SCALED WIDTH**, declared not inherited: `atom_width_frac_spacing = **1.5**` (⭐ the **BANKED census value**; the shipped `ExperimentCaptureArmAConfig` default is **0.5**, which `c2w8p3-gate-addr` measured does **not** clear the pass-2 gate), `atom_kernel = wendland` (**DECLARED SECONDARY axis**, §A29.4(i)), `atom_kernel_cutoff = 2.5`, `atom_site_local_init = True`, `site_local_radius_frac = 1.0`. Achieved `atom_width` = 0.621–0.856, co-scaled to **each seed's own measured spacing**, never hardcoded |
| census knobs | `addr_dim = 12`, `payload_dim = 1`, `capacity = 16`, `well_budget = 8`, `overdig_target = 2.0` (**achieved 2.00 on 9/9**), `n_offer_per_task = 8`, `read_batch = 16`, `read_every = 4`, `leak = 0.02`, `permanent_per_task = 1`, `d_safe_frac = 0.88`, `write_steps = 300`, `read_steps = 800`, `address_steps = 400`, `n_query_per_item = 8`, `payload_scale = 9.0`, `capture_dirs = 16`, `capture_bisect_steps = 8`, `run_gate_addr = True`, `gaddr_kappa_q = 1.0`, `gaddr_n_query_per_item = 8` (⇒ **128 cue queries/cell**), `addr_scale_mult = 1.0` (except the §7 control) |
| seeds | **0, 1, 2** on **all three arms** — **9 census cells**, plus 1 scale-control cell |
| wall | **27 043 s** of census cells (7.5 h) + **2 373 s** scale control; measured **1 864 s/cell** for the pilot cell (no other JAX process), 2 373–4 184 s under 2-way parallelism (2 workers, 8 cores) |
| PREREG | `.claude/outputs/c2w8p3-capture-strong-phi/PREREG.md`, filed **before the first CIFAR cell** (only a synthetic `d = 4` smoke preceded it). Additions in `ERRATA.md` §1/§2, both marked as additions |
| artifacts | `results/CAPTURE-STRONG-PHI.json` (the merged 9 cells) · per-cell runs under `results/<arm>_s<seed>/` · `results/render_output.txt` · `results/analyse_output.txt` · `results/scale_control_output.txt` · `results/full_suite.log` · `render.py`, `analyse.py`, `merge.py` — **every number below is re-derived by those scripts** |

---

## 3. ⭐ THE COMPLETED GATE — all four legs, per seed, three arms (9 cells)

*(`G-CAP` = frac `capture_radius > 0`; `G-DEC` = self-probe decode vs chance 0.0625 beyond 2 SE;
`G-DRIFT` = median site drift ÷ measured key spacing, **reported TWO-SIDED**; A1/A2/A3 = G-ADDR.)*

| arm | seed | G-CAP | G-DEC | G-DRIFT ratio | **A1** | A2 | **A3a** (±2 SE) | **A3b** (±2 SE) | **ALL FOUR** |
|---|---|---|---|---|---|---|---|---|---|
| **`simclr`** | 0 | 0.062 ❌ | 0.0625 = chance ❌ | **2.014** ❌ | **0.0000** ❌ | 1.000 ❌ | −0.3438 ± 0.1148 ❌ | −0.0469 ± 0.1755 ✅ | ❌ |
| | 1 | 0.375 ❌ | 0.2422 ✅ | 0.636 ✅ | 0.1484 ❌ | 0.688 ❌ | −0.2031 ± 0.1060 ❌ | −0.0625 ± 0.1757 ✅ | ❌ |
| | 2 | 0.500 ❌ | 0.1875 ✅ | 0.286 ✅ | 0.1406 ❌ | 0.688 ❌ | −0.2109 ± 0.1000 ❌ | −0.2031 ± 0.1729 ❌ | ❌ |
| **`randconv`** | 0 | 0.688 ✅ | 0.1797 ✅ | 0.143 ✅ | **0.2422** ❌ | 0.375 ✅ | −0.0781 ± 0.0797 ✅ | −0.0312 ± 0.1745 ✅ | ❌ |
| | 1 | 0.812 ✅ | 0.1875 ✅ | 0.106 ✅ | **0.2422** ❌ | 0.438 ✅ | −0.1328 ± 0.0812 ❌ | **+0.0312** ± 0.1767 ✅ | ❌ |
| | 2 | 0.562 ✅ | 0.2109 ✅ | 0.199 ✅ | 0.2266 ❌ | 0.500 ✅ | −0.2188 ± 0.1105 ❌ | **+0.0469** ± 0.1717 ✅ | ❌ |
| ⛔ **`pca`** (reference) | 0 | 0.250 ❌ | 0.1094 ✅ | 1.777 ❌ | 0.0469 ❌ | 0.875 ❌ | −0.1953 ± 0.0950 ❌ | −0.0469 ± 0.1755 ✅ | ❌ |
| | 1 | 0.812 ✅ | 0.2734 ✅ | 0.042 ✅ | 0.2422 ❌ | 0.438 ✅ | −0.0859 ± 0.0950 ✅ | −0.0156 ± 0.1767 ✅ | ❌ |
| | 2 | **1.000** ✅ | 0.2656 ✅ | 0.046 ✅ | **0.3828** ✅ | 0.125 ✅ | −0.0859 ± 0.0898 ✅ | −0.0156 ± 0.1757 ✅ | ⭐ **✅** |

**⛔ The completed gate FAILS on every arm** (`gate_pass = false` 3/3; it requires all four legs on
all three seeds). `all_four_same_seed` = **0/3 (`simclr`) · 0/3 (`randconv`) · 1/3 (`pca`)**.

### 3.1 ⭐ The headline legs A1/A2/A3, aggregated — **and the arm ordering is the finding**

| arm | **A1 mean ± SE** | A1 as counts / 128 | A2 mean | A3a mean ± SE | A3b mean ± SE | items with **zero basin** / 16 |
|---|---|---|---|---|---|---|
| **`simclr`** (PRIMARY strong) | **0.0964 ± 0.0482** | 0 / 19 / 18 | 0.792 | −0.2526 ± 0.0456 | −0.1042 ± 0.0497 | **15 / 10 / 8** |
| **`randconv`** (unfitted control) | **0.2370 ± 0.0052** | **31 / 31 / 29** | 0.438 | −0.1432 ± 0.0409 | **+0.0156 ± 0.0239** | 5 / 3 / 7 |
| ⛔ **`pca`** (weak-φ reference) | **0.2240 ± 0.0974** | 6 / 31 / **49** | 0.479 | −0.1224 ± 0.0365 | −0.0260 ± 0.0104 | 12 / 3 / **0** |

⛔ **A1's threshold is `max(4·chance, chance + 2 SE)` = 0.2500 = 32/128.** `randconv` scores
**31, 31, 29** — it misses by **one read** on two seeds (reconciliation item 3).

**Paired contrasts (same seed, bit-identical stream):**

| contrast | mean ± SE | 2 SE | per seed | seeds better |
|---|---|---|---|---|
| **`simclr` − `randconv`** | **−0.1406 ± 0.0508** | 0.1017 | −0.2422 / −0.0938 / −0.0859 | **0/3** |
| **`simclr` − `pca`** | **−0.1276 ± 0.0589** | 0.1177 | −0.0469 / −0.0938 / −0.2422 | **0/3** |
| `randconv` − `pca` | +0.0130 ± 0.1017 | 0.2034 | +0.1953 / 0.0000 / −0.1562 | 1/3 (**tied**) |

⭐ **The answer to the wave's question, stated plainly.** *"Does the physics add anything once the
encoder is not the bottleneck?"* — **on this rig the encoder was never the binding constraint for
addressing.** Making φ stronger **by the CL-accuracy metric that defines "strong" here** (banked
`c2w8-cifar-strong-phi` ACC: `simclr` **0.31912 ± 0.0050** vs `randconv` **0.21296 ± 0.0063** =
**+0.106**, vs `pca` **0.16080 ± 0.0038** = **+0.158**; ⚠ the banked `pca` row is at `phi_dim = 32`,
not at this run's `d = 12`) made the store **strictly worse at addressing**, and
the **weak PCA-φ reference produced the single best cell in the run** (A1 = 0.3828, the only cell to
clear all four legs). ⛔ This is a **component-build measurement, not a verdict**: it does not
adjudicate anything, and it makes **no** claim about arm B.

### 3.2 ⛔ The "some basin" non-leg, beside the leg — the defect class, measured on the spine

| arm | **A1 (correct basin)** | `voronoi_only` | ⛔ `any_basin` (**NOT the leg**) | launder rate (same cue) |
|---|---|---|---|---|
| `simclr` | 0.000 / 0.148 / 0.141 | 0.063 / 0.156 / 0.211 | 0.000 / **0.984** / 0.289 | 0.406 / 0.359 / 0.422 |
| `randconv` | 0.242 / 0.242 / 0.227 | 0.328 / 0.266 / 0.227 | 0.516 / **0.977** / **1.000** | 0.406 / 0.398 / 0.445 |
| `pca` | 0.047 / 0.242 / 0.383 | 0.078 / 0.281 / 0.383 | 0.445 / 0.875 / **1.000** | 0.273 / 0.367 / 0.469 |

⭐ **On four of nine cells `any_basin` ≥ 0.98 while A1 ≤ 0.24.** Nearly **every** read lands in *a*
basin; roughly **one in four** lands in the **right** one. **A pass-2-style gate reading
`any_basin` would have scored these cells as a triumph.** That is precisely the defect G-ADDR was
built to close, now demonstrated on the spine rather than on a planted control.

## 4. ⭐⭐ THE BRANCH — **(b) NO DAYLIGHT**, computed by `daylight_verdict`, never argued

**Registered rule, quoted from the code that computed it:**
> *(a) DAYLIGHT iff a launder margin is **POSITIVE beyond 2 SE on ≥ 3 seeds** — cue (A3a, McNemar SE)
> or stream (A3b, pooled binomial SE); **(b) NO DAYLIGHT** otherwise.*

| arm | branch | A3a positive beyond 2 SE | A3b positive beyond 2 SE |
|---|---|---|---|
| `simclr` | **(b) NO DAYLIGHT** | 0/3 | 0/3 |
| `randconv` | **(b) NO DAYLIGHT** | 0/3 | 0/3 |
| `pca` | **(b) NO DAYLIGHT** | 0/3 | 0/3 |

⛔ **BRANCH (b) IS THE PRE-REGISTERED, REPORTABLE FINDING — NOT A SHORTFALL.** Quoting the
registration it fired against, filed before the first cell (`PREREG-C2W8-PASS3` §6/§8, echoed in the
artifact's own `daylight.prereg` block):

> *"**(b) NO DAYLIGHT** — the CIFAR spoke's ±0.0007 result reproduced on the census rig ⇒ **the
> tier-i thesis measured at the CL substrate. A REPORTABLE FINDING, not a failure to be tuned
> away.**" — Hub prior **Q6 = 0.70**, against **Q5 = 0.15** for daylight.*

**Nothing was tuned.** The store arm, the map, the dimension and the seeds were all fixed by ruling
or by wt2's measurement **before** the first cell; no knob was moved after a number was seen.

⭐ **And the ±0.0007-class narrowing is visible in the raw stream reads.** `randconv` seed 0's four
read events give store `[0.6875, 0.5000, 0.2500, 0.1875]` against launder
`[0.6875, 0.5000, 0.3125, 0.2500]` — **identical on the first two events**, `A3b = −0.0312`. On
`randconv` seeds 1 and 2 the stream margin is **positive** (+0.0312, +0.0469) and on `pca` it is
−0.0156 twice. ⇒ **the store and a 832-byte table are statistically indistinguishable on held-out
stream reads** (|A3b| ≤ 0.047 on **7 of 9** cells; **8 of 9** are inside 2 SE ≈ 0.175, the exception being `simclr` seed 2 at −0.2031 ± 0.1729). That is the
tier-i thesis, measured at the CL substrate, on the census rig. ⛔ **Neither branch is a tier-ii
verdict; no paper number is produced.**

## 5. ⚠⚠ THE D2a DIAGNOSTIC (§A29.6) — TWO-SIDED, and it is the sharpest thing in this run

⛔ **`G-DRIFT → 0` means the settled point approaches a deterministic function of the stored key =
D2a = TABLE-EXPRESSIBLE**, which the configuration intervention §8.2 prohibits. **No leg, objective
or tuning choice here treated drift → 0 as a target.**

| arm | settle↔launder **agreement** (chance 0.0625) | median ‖settle − launder key‖ ÷ spacing | G-DRIFT ratio |
|---|---|---|---|
| `simclr` | 0.055 / 0.164 / 0.250 | 2.005 / 1.447 / 1.587 | 2.014 / 0.636 / 0.286 |
| `randconv` | **0.516 / 0.484 / 0.266** | 0.435 / 1.242 / 1.601 | 0.143 / 0.106 / 0.199 |
| `pca` | 0.070 / 0.484 / **0.547** | 1.964 / 0.653 / **0.143** | 1.777 / 0.042 / 0.046 |

### ⛔⛔ SAY IT PROMINENTLY: across ALL NINE CELLS THE GATE'S SCORE TRACKS SETTLE-COLLAPSE

| statistic (n = 9 cells) | value |
|---|---|
| Spearman **ρ(A1, G-DRIFT ratio)** | **−0.9667** |
| Spearman **ρ(A1, settle↔launder agreement)** | **+0.9333** |
| Pearson r(A1, drift) / r(A1, agreement) | −0.8774 / +0.9072 |

Cells ordered by A1 (best first): `pca:2` (A1 0.383, drift 0.046, agree 0.547) · `randconv:0`
(0.242, 0.143, 0.516) · `randconv:1` (0.242, 0.106, 0.484) · `pca:1` (0.242, 0.042, 0.484) ·
`randconv:2` (0.227, 0.199, 0.266) · `simclr:1` (0.148, 0.636, 0.164) · `simclr:2` (0.141, 0.286,
0.250) · `pca:0` (0.047, 1.777, 0.070) · `simclr:0` (0.000, 2.014, 0.055).

⛔ **The one cell that cleared the completed gate (`pca` seed 2) is also the cell whose settle agrees
with the same-keys launder most often (0.547) and whose drift is essentially the lowest (0.046 vs a
minimum of 0.042).** The mechanical per-arm flag `best_is_also_lowest_drift` reports **False on all
three arms** — decided by a **0.004** tie-break on `pca`. ⚠ **Reporting only that boolean would have
concealed a ρ = −0.97 monotone relationship.** Both are in the artifact; reconciliation item 2 asks
the Hub to strengthen the flag. **This co-occurrence is the D2a signature, not a success.**

## 6. The byte ledger — φ **and** the map on **every** arm, launder included (§A4.3 / Head ruling R2)

| arm | encoder floats | **map floats** | CLU bytes | launder bytes (census ring, `n = 8`) | ratio | launder bytes (G-ADDR cue, `n = 16`) | ratio | **ratio with φ on BOTH sides** |
|---|---|---|---|---|---|---|---|---|
| `simclr` | **225 536** | **3 328** | 1 966 848 | 416 | **4 728×** | 832 | **2 364×** | **3.147×** |
| `randconv` | **225 536** | **3 328** | 1 966 848 | 416 | **4 728×** | 832 | **2 364×** | **3.147×** |
| `pca` | 39 936 | **0** (identity) | 1 966 848 | 416 | **4 728×** | 832 | **2 364×** | 13.278× |

⛔ **The φ term is the SAME number on the store row and on the launder row** (asserted in
`render.py`, which raises otherwise) — because both read the *same object*. ⛔ **Matching is
matched-ITEMS on every quotation in this report; matched-bytes is NOT met.** Pass 1's **1 253×**
stream-launder byte-ratio caveat travels unchanged. ⛔ **No performance claim is made at any ratio,
and no gate leg reads bytes.** `(d, atom budget) = (12, 32 768)` is carried as **one joint dial** on
every ledger row, with `budget_honoured = true` on 9/9.

**Head ruling R2(b), asserted rather than intended:** `launder_audit` re-derives the census launder's
keys from the projected φ and **raises** unless they are bit-identical. **9/9 cells**:
`launder_key_dim = store_address_dim = 12`, `launder_reads_projected_phi = true`,
`bit_identical_to_store_addresses = true`, `phi_dim_before_map = 256` (`simclr`/`randconv`) / `12`
(`pca`). `tests/test_capture_strong_phi.py` additionally asserts it **raises** on the handicap match
(a 16-dim launder against a 6-dim store) and on a same-width-but-different-φ launder.

## 7. ⭐ The §4 SCALE-ONLY control (wt1's, carried) — the guard HOLDS; the **verdict** does not

`randconv`, seed 0, identical φ, **address scale × 0.8 with the payload co-scaled** (`payload_scale`
9 → 11.25 — an *address-only* rescale is **not** covariant on this rig; `c2w8p3-gate-addr` §6(c)).

| quantity | `a = 1.0` | `a = 0.8` | Δ |
|---|---|---|---|
| `phi_scale` | 1.05915 | 0.84732 | ×0.8 **exactly** ✓ |
| measured spacing · `d_safe` · `atom_width` | 0.51746 · 0.45537 · 0.77619 | 0.41397 · 0.36429 · 0.62095 | all **×0.8 exactly** ✓ |
| `cue_displacement / codebook_spacing` | **2.46017** | **2.46017** | **0.00000** ✓ (the leg is exactly scale-covariant) |
| A3a launder rate | 0.40625 | 0.40625 | **0.00000** ✓ |
| A2 | 0.37500 | 0.37500 | **0.00000** ✓ |
| **A1** | 0.24219 | 0.28906 | **+0.04688** |
| A3a / A3b | −0.07812 / −0.03125 | −0.11719 / +0.01562 | −0.039 / +0.047 |
| median site drift / depth | 0.08121 / 0.96536 | 0.04608 / 0.69796 | −0.035 / −0.267 |

**Registered guard** (`PREREG-C2W8-PASS3` §4; Hub **Q8** prior 0.90; operationalised by wt1 as
`|ΔA1| ≤ 0.05`): **|ΔA1| = 0.0469 ⇒ the guard HOLDS.** ⚠⚠ **But the leg's VERDICT flips
`False → True`**, because the baseline sits **one read** below the 0.25 threshold. ⚠ **Declared
limitation (`ERRATA.md` §1):** the rescaled cell is a full re-run (store re-written, geometry
re-measured), so ΔA1 is an **upper bound** on scale-dependence, not a decomposition. Reconciliation
item 3 asks the Hub whether a leg whose *metric* is scale-stable but whose *verdict* is not may ship.

## 8. Registered predictions vs measurement (my `PREREG.md` §2) — 11 hits, 5 misses

| # | prediction | measured | |
|---|---|---|---|
| **S1** | store **NOT inert** at `d = 12`, ≥ 2/3 seeds | **9/9 cells not inert** (0.470–0.965) | ✅ **HIT** |
| S1b | `depth_raw_median` 0.25, band 0.02–0.70 | 0.607 (`pca`) / **0.853, 0.864** (`simclr`, `randconv`) | ❌ **MISS** — above the band on 2 arms; the store digs **deeper** than I predicted |
| S2 | no arm is inert | none inert | ✅ HIT |
| **G1** (= Hub **Q4** 0.15–0.45) | A1 best arm 0.32, band 0.10–0.60 | **0.2370** (`randconv`) | ✅ **HIT** — and **Q4 HITS** |
| **G2** | ordering `simclr ≥ randconv > pca` (prior 0.55) | **`randconv` ≈ `pca` > `simclr`** | ❌ **FALSIFIED** — reconciliation item 1 |
| **G3** | A3a launder rate 0.93, band 0.80–1.00 | **0.273–0.469** | ❌ **BADLY MISSED** — at `d = 12` the cue displaces **2.46–3.21×** the codebook spacing, so 1-NN itself is only ~40 % correct. My §1.2 registered the *mechanism* and I still got the *level* wrong |
| G4 | G-CAP passes, best arm | `randconv` **3/3**, `pca` 2/3 | ✅ HIT |
| G5 | G-DEC above chance beyond 2 SE | **8/9 cells** (only `simclr` seed 0 exactly at chance) | ✅ HIT |
| G6 | G-DRIFT ratio best arm 0.02, band 0.001–0.30 | `randconv` 0.106–0.199 | ✅ HIT (in band) |
| **G7** | completed gate passes on **any** arm (prior 0.15 yes) | **no arm passes** | ✅ HIT |
| **B1** | branch **(b)** (my prior 0.88; Hub Q6 0.70) | **(b), 3/3 arms, 0/9 cells** | ✅ **HIT** |
| B2 | A3a best arm −0.30, band −0.75…+0.05 | −0.143 (`randconv`) / −0.122 (`pca`) | ✅ HIT |
| **B3** | A3b best arm −0.30, band −0.70…0.00 | **+0.0156** (`randconv`) | ❌ **MISS (above band)** — the best arm's stream margin is **positive on average**, though far inside 2 SE |
| D1 | agreement 0.72, band 0.30–0.99 | 0.422 (`randconv`), 0.367 (`pca`) | ✅ HIT |
| D2 | ‖settle−launder‖÷spacing 0.35, band 0.01–1.5 | 0.435 / 1.242 / **1.601** (`randconv`) | ◐ 2/3 in band, 1 above |
| **D3** | `best_is_also_lowest_drift` **true** (prior 0.60) | **false on 3/3 arms** — *but* ρ(A1, drift) = **−0.967** | ❌ **MISS at the boolean, and the boolean is the wrong instrument** (recon. item 2) |
| L1 | `phi_param_floats` = 225 536 | **225 536** | ✅ HIT |
| L2 | `map_param_floats` = 3 328 at `d = 12` | **3 328** | ✅ HIT |
| L3 | φ term identical on store and launder rows | asserted, 9/9 | ✅ HIT |
| L4 | `ratio_clu_over_knn_launder` > 100× | **4 728×** (census ring) / **2 364×** (cue) | ✅ HIT |
| **P1** | 2 550 s per cell; abort at 3× | **1 864 s solo**, 2 373–4 184 s at 2-way parallelism | ✅ HIT — **no cut was needed, no cell was cut** |
| P2 | `simclr` fit 1 300–2 500 s/seed | simclr cells 4 003/4 172/4 184 s total, i.e. ≈ 1 400–1 600 s of fit above a `randconv`-class census | ✅ HIT |
| P3 | ≈ 8 h total | **7.5 h** of census cells | ✅ HIT |

## 9. How I verified (commands + observed output)

```
# the build, before any real cell — synthetic CIFAR-shaped data, no download
PYTHONPATH=<wt> .venv/bin/python .claude/scratch/.../smoke.py <dir>     -> 2 arms, d=4, green
# targeted tests
PYTHONPATH=<wt> .venv/bin/python -m pytest tests/test_capture_strong_phi.py -q -p no:randomly --no-cov
  -> 25 passed in 3.94s
# the 9 census cells + the scale control (cwd = the WORKTREE; provenance printed per cell)
python -m chlu.experiments.exp_capture_strong_phi --seeds <s> --arms <arm> --save-dir <dir>
python .claude/scratch/.../scale_control.py <dir> 0.8 randconv 0
# merge -> render -> analyse (every reported number is re-derived from the JSON)
python merge.py results/CAPTURE-STRONG-PHI.json results/*/capture_strong_phi.json
python render.py  results/CAPTURE-STRONG-PHI.json  > results/render_output.txt
python analyse.py results/CAPTURE-STRONG-PHI.json  > results/analyse_output.txt
ruff check chlu/ tests/test_capture_strong_phi.py  -> All checks passed!
```

### Suite — **1 564 passed / 0 failed (2 513 s = 41 m 53 s)**

```
PYTHONPATH=/Users/user/Desktop/CHLU-c2w8p3c .venv/bin/python -m pytest -q -p no:randomly --no-cov
  -> 1564 passed, 36 warnings in 2513.04s (0:41:53)
```

**Count arithmetic, with the checkout named — MEASURED, not quoted:**

| item | value |
|---|---|
| checkout | worktree **`/Users/user/Desktop/CHLU-c2w8p3c`** at **`d7eabdd`**, base **local `main @ 18b4205`** |
| base collected on this branch, **excluding** my new file (`--ignore=tests/test_capture_strong_phi.py`) | **1 539** |
| this branch adds `tests/test_capture_strong_phi.py` (`--collect-only`) | **25** |
| **expected** | 1 539 + 25 = **1 564** |
| **observed** | **1 564 passed / 0 failed** ✅ |

**HEAD stability** (a suite result on a moving HEAD is worthless): `HEAD = d7eabdd` and
`main = 18b4205` **identical before and after** the run — both printed at launch and at completion.
⚠ The suite ran **concurrently with the scale-control cell**, which is why it took 41 min rather than
wt2's 33; no test outcome depends on wall time.

⚠ **Reconciling the base count with the two precondition spokes:** wt2 measured **1 504** at
`main @ 1eda6a0` and shipped **+18**; wt1 shipped **+17**. `1 504 + 18 + 17 = 1 539`, which is
**exactly** what I collect at the integrated base `18b4205`. ⇒ **wt2's flagged 9-test discrepancy
against `PREREG-C2W8-PASS3`'s "1495/0" is confirmed to be in the prereg's figure, not in wt2's.**

## 10. Declared NOT-RUNs (never reported as nulls)

- ⛔ **`d = 16`, the geometry-favoured dimension** — the store is **INERT** there (wt2, censused cell
  at a fully honoured 131 072-atom budget). A **declared NOT-RUN**, and its own reason is recorded
  separately from the gate's reason.
- ⛔ **merge / prune / restoration verbs and every §2.7 claim cell** — still deferred (no population;
  monitor #3 defect open).
- ⛔ **the arm A vs arm B race** — VOID as a comparison, **stays UNADJUDICATED** (§A30.1). Arm B is
  not run, not scored, not mentioned as a comparator anywhere above.
- ⛔ **any tier-ii verdict, any full-CLU verdict (§A28.4), any I2 verdict, any paper number, any
  performance/ACC claim.**
- **`convae`** and every φ arm beyond `simclr`/`randconv`/`pca`; **`generic_frozen`** φ (the leaking
  reference regime); **MNIST** (excluded by Head ruling R1, not merely unrun); **`truncate`/`gaussian`
  map controls** (wt2 measured them; not re-run here); the scale control at seeds 1/2 and at
  non-`randconv` arms; any third read iteration (§A26.6).

## Git footprint

- **Branch:** `c2w8p3-capture-strong-phi` (worktree `../CHLU-c2w8p3c`), base **local `main @
  18b4205`**, named explicitly per the task file. **Not pushed. No PR. No merge. Left for Hub review.**
- **Commits (2):**
  - `0f057a0` `[experiment-engineer] the spine: the completed gate at strong phi, on CIFAR-10`
  - `d7eabdd` `[experiment-engineer] tests: the frozen census, R2(b), the declared store, the branch`
- **Files touched (3 + 1 new test):**
  - `chlu/experiments/exp_capture_strong_phi.py` — **new** (mine, declared).
  - `tests/test_capture_strong_phi.py` — **new** (mine, declared), 25 tests.
  - `chlu/config.py` — **additive only**: `ExperimentCaptureStrongPhiConfig` + its three registration
    sites, all inside `# --- BEGIN/END c2w8p3-capture-strong-phi ---` banners. **No existing default
    changed.**
  - `chlu/cli/experiment_cmd.py` — **additive only**: the `exp-capture-strong-phi` hook, same banners.
- ⛔ **Not modified, as declared read-only:** `chlu/core/well_lifecycle.py`,
  `chlu/experiments/exp_well_lifecycle.py` (wt1's instrument — the arms are measured on **one**
  arithmetic), `chlu/experiments/{exp_phi_read_in,phi_encoders,exp_phi_geometry}.py` (wt2's mapping),
  `chlu/core/{emission_head,memory_potentials,clu_system}.py`, `exp_capture_armA/B.py`, and the live
  `pilot-ttt-nan-and-d5-wiring` territory (`scripts/csf3/`, `train_cluformer.py`, `blocks.py`,
  `exp_cluformer_pilot.py`).
- **Rebase:** `git rebase main` from the worktree is a **no-op** — `main` has not moved from `18b4205`
  since the branch was cut (verified before and after the suite). ⚠ Protocol §3.5 / §7.21:
  `origin/main` is stale and was **not** used.
- **Collision note:** the shared checkout had another agent's branch
  (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` @ `7fcef50`) checked out for the whole
  session; **all work was done in the dedicated worktree** and the shared checkout was never written.

## Open questions / follow-ups / risks

1. ⚠ **Attribution of the `simclr` inversion is NOT established.** I measured *that* the fitted
   encoder is worse; I did **not** measure *why*. Two live candidates, both cheap: (i) `simclr`'s
   codebook spacing is 0.652 vs `randconv`'s 0.729 while the co-scaled atom width is *larger*
   (0.856 vs 0.776) ⇒ the compact supports may be over-overlapping; (ii) the l2-normalised
   contrastive φ concentrates the address cloud on a shell, which is a different geometry from the
   one `atom_site_local_init` was tuned for. **One width sweep at fixed φ decides it.**
2. ⚠ **`randconv`'s A1 sits one read below the threshold on 2/3 seeds.** More seeds (5–7) would
   settle whether the arm is genuinely at 0.24 or straddling 0.25. **This is a seed-count question,
   not a tuning question**, and I did not run it (declared).
3. ⛔ **The gate cannot presently distinguish "addresses well" from "collapses onto the key".**
   ρ(A1, drift) = −0.967 means the four legs are close to one degree of freedom. Until a two-sided
   drift leg exists, a *high* completed-gate score should be read with the D2a caveat attached.
4. ⚠ The `pca` arm's address scale is **~20× smaller** than the conv arms' (`phi_scale` 0.054 vs
   1.06–1.19) because the raw PCA-of-pixels features are large; the `1/r95` normalisation absorbs it
   exactly (all reported geometry is a dimensionless ratio), but any
   future cell quoting an **absolute** φ quantity across these arms will be comparing different units.

---

## Proposed handover updates (for the Hub)

- **§7-CURRENT, new entry (config, RESOLVED-BY-DECLARATION):** `ExperimentCaptureArmAConfig.atom_width_frac_spacing`
  defaults to **0.5** (the pilot cell) while the **banked** pass-2 arm-A census ran at **1.5**, and
  the default does **not** clear the pass-2 gate (`c2w8p3-gate-addr` item 2). The spine does **not**
  inherit it: `ExperimentCaptureStrongPhiConfig.atom_width_frac_spacing = 1.5` is declared in the new
  group and pinned by `tests/test_capture_strong_phi.py`. **The underlying default is still the
  trap** — any *other* caller of `exp_capture_armA` silently scores the pilot store.
- **§7-CURRENT, new entry (ENV, standing):** running `python -m chlu.…` from the **main checkout**
  while `PYTHONPATH` points at a worktree silently imports the **main repo's** `chlu` package
  (`sys.path[0]` = cwd). It cost one launch here (`No module named …exp_capture_strong_phi`). The
  safe recipe is **cwd = the worktree** plus a printed `chlu.__file__` provenance line per cell.
- **§3 config table:** new group **`experiment_capture_strong_phi`** (additive; `d = 12`,
  `arms = [randconv, simclr, pca]`, `atom_width_frac_spacing = 1.5`, `projection = pca`,
  `d2a_probe = True`). New CLI command **`chlu exp-capture-strong-phi`**.
- **Test count:** base at `main @ 18b4205` is **1 539** (measured); this branch takes it to **1 564**.
  `PREREG-C2W8-PASS3`'s "1495/0" is **wrong by 9** — confirmed independently of wt2's flag
  (1 504 + 18 + 17 = 1 539 closes exactly). Worth correcting wherever the next spoke will read it.
- **Reconciliation items 1–6 above need an owner** (item 1 = how "strong φ" may be described anywhere
  in this wave; item 2 = the gate's D2a blind spot; item 3 = the threshold/scale-verdict instability).
