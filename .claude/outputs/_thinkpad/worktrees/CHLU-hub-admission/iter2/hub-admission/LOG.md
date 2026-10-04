# `hub/admission` — RUNNING LOG
**Agent:** autonomous-hub (Opus) · execution tier, reports to the **coordinator**, never the advisor.
**Branch:** `hub/admission` @ `945a3e9` (`base/iter2`) · **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-admission`
**Machine:** FORGIS_14, Windows AMD64, local CPU, **4 threads** · sub-agents ≤ 2, all `model: "opus"`.
**Task file:** `_hub_scratch/LAUNCH-hub-admission.md` (there is no `tasks/hub-admission.md`) + `LAUNCH-COMMON.md`.

## RELAUNCH
This is a NEW start. `hub/admission` is a NEW target — it has no predecessor worktree of its own.
`hub-refusal` and `hub-refusal-2` outputs are **inherited fact** and are read, cited, and never restated
as mine. ⛔ The three concurrent hubs (`generation-table-2`, `composition-2`, `deletion-2`) are not read.

- **2026-09-17T14:17:12Z** — `START_UTC.txt` written (new stamp, this run). **12 h deadline =
  `2026-09-18T02:17:12Z`** true wall clock, measured by `elapsed.sh` against the real clock only.
- **2026-09-17T14:17Z** — §0 OVERRIDE quoted and recorded verbatim, as required:

  > **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and
  > `hub/generation-table` on v2 despite
  > `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.**
  > The failure is in the test's `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()`
  > calls `read_bytes().decode()`, which does **not** fold universal newlines, while real
  > `Path.read_text()` opens in text mode and does. On Windows the fixture is written by a real
  > `write_text()`, which expands `\n → \r\n`, so the mock counts two characters per newline and
  > `chars` exceeds `len(text)` by exactly the newline count. **Nothing to do with encoding.** It
  > passes on macOS only because macOS does not expand newlines. The mock is fixed on the Mac and ships
  > in the next cut; the UTF-8 behaviour it guards is unaffected and still enforced by the shipped
  > code's explicit `encoding="utf-8"`.

  ⛔ It covers exactly that one test. Every other red check remains information.
- **2026-09-17T14:17Z** — env fixed: `_env.sh` (bundle v2 root, `PYTHONUTF8=1`, venv python 3.11.16,
  `OMP/MKL/OPENBLAS_NUM_THREADS=4`). Worktree verified clean at `945a3e9`.
- **2026-09-17T14:18Z** — inherited facts adopted, by explicit bullet:
  (i) `_hub_scratch/_gate_identity.out.json` + `_gate_identity.py` (coordinator, §0b) — the per-well API;
  (ii) `hub-refusal/scorer/refusal_lib.py` — `build_context`, `build_land`, `term_experts`,
  `launch_from_terms`, `per_term_residuals`, `shuffled_store`. ⭐ Adopted as **called libraries**, not
  copied; my own module `scorer/admission_lib.py` imports them.
  (iii) `hub-refusal/REPORT-hub-refusal-2.md` — the "21 atom-spreads" measurement that §0c is about.
- **2026-09-17T14:19Z** — ⛔ SCORED BUDGET: 0/3 spent. At most 1 this iteration, seed 0 only, and only
  after PREREG-1 + PREREG-2 and a DEV number that clears the §5 bar.

## UNIT 0c — the §0c check, done in minute 3. ⭐ **§5.7 DOES NOT FIRE.**
- **2026-09-17T14:20Z** — `scorer/admission_lib.py` + `scorer/u0c_probe.py` written and run
  (`U0C_PROBE_seed0_A.json`). Context + landscape build = **50 s**; the whole probe **57 s**.

**Step 1 ground truth (partial), measured:**
- `feat_shape = (128, 4, 4)` ⇒ the 2048 latent dims are **128 feature maps × 4×4 spatial**, flattened.
  ⛔ Not pooled.
- ⚠ **`K = 5`, NOT 32.** The shipped store plan for `shapes3d/recon` gives `plan["K"] = 5`;
  `well_channels` sizes `[303, 305, 573, 501, 366]` (sum 2048), `ward` grouping, **71 dead channels**,
  1977 live, and the re-derivation **reproduces the plan bit-identically**. The addendum's "K = 32" does
  not correspond to any object reachable from the pin's store plan on this dataset. ⭐ **Logged as a
  question for the coordinator; I proceed on the shipped K = 5 object and say so on every row.**
- `pool = store_train = 32,768` rows · `fit = 32,256` · `val = 512` · cache `scope="full"`, 480,000 rows,
  `float16`, `D = 2048`. ⭐ Seed 0's cache is full-scope ⇒ **cannot be extended** ⇒ no re-encode hazard.
- Re-encode check, recorded verbatim as required: `max_ulp_delta = 51.0`, `frac_differing = 0.009766`,
  `enforced = False` (read-only, gathers only cached `store_train` rows). ⭐ Inherited §5.1 verdict
  (`hub/generation-table-2`): in the `|cached| ≥ 2^-4` band max ULP = 1.000, zero violators.
- Landscape rung A: `N = 64` atoms/well, rank 16, store 524,288 B, coupling 524,288 B, ledger matches
  PREREG §3. `σ = [4.6343, 5.0714, 9.1827, 6.3054, 5.1006]` (`nn_median`, `frac = 1.0`, shipped rule).

**PREREG-1 self-checks, all three PASS on the live store:**
1. ⭐ `identity_E_le_0_holds = True`, worst `max_i E_k(a_ki) = −30.394`. Per well:
   `−33.398 / −40.862 / −225.581 / −89.683 / −30.394`. Certified radii `σ_k√(2 ln 64)` =
   `13.366 / 14.626 / 26.484 / 18.185 / 14.710`. ⭐ Reproduces the coordinator's §0b table exactly
   (same object), so my per-well reduction agrees with the coordinator's independently-written one.
2. ⭐ `Σ_k E_k` reproduces the shipped `gen_landscape.store_energy`: `max_rel_delta = 1.22e-07`.
   **Proof that this is a reduction of the shipped quantity, not a re-implementation.**
3. ⭐ `term_experts` is **bit-identical** to the shipped `request_experts` on valid rows
   (`max_abs_delta = 0.0`), shape `(6, 256, 5, 64)`.

## ⭐⭐ THE §0c ANSWER — 256 valid `val` requests, seed 0, rung A
| launch point | accept rate at the gate | mean `max_k E_k` | mean `min_k d/σ_k` | mean `d/σ_k` |
|---|---|---|---|---|
| `clamp_atom` (**shipped default**) | **1.0000** | −39.51 | **0.000** | **0.000** |
| `posterior_mean` (the CLU's own linear read) | **1.0000** | −52.70 | 0.790 | 0.912 |
| `prior_atom` (launch-off control) | 1.0000 | −40.89 | 0.000 | 0.000 |
| `r8_affine` (the R8 request embedding) | **1.0000** | −42.06 | 0.769 | 0.902 |
| `true_feature` (the row's own cached encoder feature) | **1.0000** | −34.50 | 0.800 | 1.001 |

⭐⭐ **§0c's fatal branch does NOT fire.** Accept rate on valid `val` is **1.0000 at every launch point**,
with `max_k E_k` sitting 30–50 *below* zero. **The gate is live and valid requests are comfortably inside
the level set.** Proceed to step 2 proper.

## ⛔⛔ ONE STRUCTURAL FACT, PROVED, THAT DECIDES THE SHAPE OF THE RUN
`clamp_atom` and `prior_atom` give `d/σ_k = 0.000` **exactly**, because `launch_state` sets each well's
channels to a *stored atom* (`x0[:, cols] = A[idx[:, k]]`). By the §0 identity `E_k(a_ki) ≤ 0`, therefore:
> ⛔ **At the SHIPPED `clamp_atom` launch the gate is IDENTICALLY SATISFIED — for every request, valid or
> invalid, at every severity. `refuse` ≡ 0. This is algebra, not a measurement.**

## UNIT 1 — §3 step 2 / falsifier §5.10. `U1_SEVERITY_seed0_A.json`
- **2026-09-17T14:22Z** — 256 val corners, one factor at a time, R8 severities `w ∈ {2,3,4,7,13}`,
  plus `unseen_value` (w = 0.5) and `duplicate_factor`. Whole sweep: **56 s**.

**⭐ The normaliser question, closed with numbers.** Atom-cloud RMS radius / `σ_k` per well =
`1.86 / 1.80 / 1.17 / 1.27 / 2.10`. So one "atom spread" ≈ **1.2–2.1 `σ_k`** — the two are the same
order. ⛔ `hub/refusal-2`'s **21 atom-spreads** therefore does **not** describe this object's linear read,
which sits at `d/σ_k = 0.79`. Different store (that number is the fit-derived synthetic analogue), and it
does not transfer. **§0c is closed: the gate is live.**

### ⭐⭐ THE §5.10 ANSWER — it depends entirely on WHICH launch point, and that is the finding
`accept_rate` at `gate@r8_affine`, by severity (mean over 256 val corners):

| factor | w=2 | w=3 | w=4 | w=7 | w=13 | crosses at |
|---|---|---|---|---|---|---|
| floor_hue | 1.000 | **0.000** | 0.000 | 0.000 | 0.000 | w=3 |
| wall_hue | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 | w=4 |
| object_hue | 1.000 | 1.000 | 1.000 | **0.000** | 0.000 | w=7 |
| shape | 1.000 | 1.000 | 1.000 | **0.242** | 0.000 | w=7 |
| scale | 1.000 | 1.000 | 1.000 | 1.000 | **0.543** | w=13 |
| orientation | 1.000 | 1.000 | 1.000 | 1.000 | **0.852** | w=13 |

`max_k E_k` rises **monotonically with severity on all six factors** at `gate@r8_affine`
(floor_hue: −34.4 → +7.5 → +77.7 → +463 → +2060; wall_hue: −39.2 → −11.4 → +40.1 → +344 → +1730;
orientation: −41.1 → −40.2 → −39.1 → −34.1 → −4.7). `d/σ_k` rises with it (0.86 → 2.69 on floor_hue).

- ⭐ **`gate@r8_affine`: §5.10 does NOT fire.** The affine head crosses the level set, monotonically,
  on every factor. **The symbolic out-of-range leg is alive.**
- ⛔ **`gate@posterior_mean`: §5.10 FIRES, completely — `accept = 1.000` at every factor and every
  severity, `max_k E_k` pinned at ≈ −52 and moving by < 10 over a 6.5× severity range.**
- ⛔ **`gate@clamp_atom`: `accept ≡ 1.000` everywhere, as proved algebraically in UNIT 0c.**

### ⛔ THE MECHANISM, and it is one line
`posterior_mean` and `clamp_atom` both return a point **inside the convex hull of the well's own atoms**
(a softmax-weighted mean, or a vertex). `E_k` is a *distance-to-the-atom-cloud* functional, so a point
that can never leave the hull can never leave the sub-level set. ⛔ **The CLU's own address-book read
cannot refuse anything, at any severity, by construction** — and that is the benchmark-side confirmation
of `hub/refusal-2`'s synthetic-analogue negative, obtained here on the real store.

### ⭐ DECISION (mine, logged): the symbol-mode launch point is the **R8 affine head**
§3 step 2 names it — *"symbol-mode `out_of_range` … **via the R8 rule on the one-hot design**"*, and
*"does the **affine head** cross the level set?"*. It is the shipped, declared, **cross-arm** request
embedding `gen_request.FactorPrototypes.embed` (`x̂ = μ_global + Σ_terms (μ_{k,v} − μ_global)`), so the
factored-kNN rival of control (b) reads **the same point** and the comparison is like-for-like by
construction. ⭐ Its bytes are reported (`r8_bytes`, the addressing-head ledger), as the pin's own
convention requires. ⛔ `posterior_mean` and `clamp_atom` are reported as **measured negatives**, never
as the arm, and never quietly dropped.

### Other measured rows
- **`unseen_value`** (interpolated between adjacent grid values, w = 0.5): `accept = 1.000` on all six
  factors at every launch point. ⭐ The gate does **not** over-refuse a near-manifold interpolation —
  this is the correct behaviour and it is evidence for `accept_B`, not against it.
- **`duplicate_factor`** (interface grammar only, ⛔ no landscape claim per §4): `gate@r8_affine`
  `accept` = 0.793 / 0.840 / 1.000 / 1.000 / 1.000 / 1.000 by factor. Reported as a grammar row.

## HOUSEKEEPING — two incidents, logged because they cost real artefacts
- **2026-09-17T14:22:17Z** — ⚠ **CLOCK CORRECTION, logged because §-0 warns about exactly this.** I had
  written `14:55Z` and `15:12Z` into the two headings above from my own sense of elapsed time. The real
  clock said **0.085 h elapsed**. Both stamps are corrected above to `14:20Z` / `14:22Z`. ⭐ My perceived
  time was running **~7× FAST**, which is the same failure §-0 records from iteration 1.
  ⛔ Every stamp from here comes from `elapsed.sh` or `date -u`, never from recollection.
- **2026-09-17T14:24Z** — ⚠ **`cat >> file <<'EOF'` from Git Bash on this box writes at a STALE OFFSET**
  and silently clobbers the head of the target. It destroyed `scorer/admission_lib.py` once and the first
  half of this LOG once. Both were rewritten whole from context; ⛔ nothing else was touched and nothing
  was deleted. **Fix: `scorer/logappend.py` (python, mode `"a"`, explicit UTF-8) is now the only append
  path.** Recorded because a silent truncation of a *results* file would have been much worse.

## UNIT 2 — §3 step 1 complete. `FACTOR_WELL_MAP_seed0.json`
- **2026-09-17T14:27Z** (`date -u`) — MI(nearest-atom index of well k ; factor f) over **32 shuffled-label
  floors**, on all 32,768 store-train rows, normalised by `H(f)`. Runtime 43 s. All 64 atoms are used as
  a nearest neighbour in every well (no dead atoms).

| well | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| 0 | 0.2339 | 0.8893 | 0.0333 | 0.0320 | 0.0773 | 0.1985 |
| 1 | 0.1689 | **0.8992** | 0.0463 | 0.0303 | 0.0717 | **0.2555** |
| 2 | 0.1882 | 0.6274 | 0.2959 | 0.0625 | 0.1707 | 0.1064 |
| 3 | 0.4540 | 0.0790 | **0.5547** | **0.1115** | **0.3249** | 0.0057 |
| 4 | **0.9462** | 0.1013 | 0.2233 | 0.0753 | 0.1136 | 0.0149 |

⚠ **The map exists but it is PARTIAL, and the report must say so.** 6 factors map onto only **3 distinct
argmax wells** of 5 — `K = 5 < 6`, so by pigeonhole it *cannot* be injective. Only **`floor_hue` is
singly-assigned** (well 4, excess MI/H = 0.946, margin +0.492 over the second). `object_hue` and `shape`
share well 3; `wall_hue` and `orientation` share wells 0/1. ⛔ **`scale` is barely carried anywhere**
(max 0.1115) — the latent does not localise it.
⇒ **Exemplar mode is well-posed for `floor_hue` (well 4) and workable for `object_hue`/`shape` (well 3);
it is ⛔ ill-posed for `scale` and `orientation`.** That is a measured limit of the object, and it is
reported, not worked around.

## UNIT 3 — THE DEV TABLE. `DEV_TABLE_seed0_A.json`
- **2026-09-17T14:29Z** — 512 val corners · `oor` pool 15,360 (6 factors × 5 severities × 512) ·
  `unseen_value` 3,072 · `A7` 72 (7 terms each, file-backed from the registry). Runtime 66 s.
  ⛔ **`eval_b` was NOT touched.** Rival thresholds fitted on **4,096 corners drawn from `fit_rows`** —
  ⛔ never `val`, ⛔ never B.
- Self-check: `Σ_{t,k}` of my per-(term, well) residual reproduces the shipped
  `gen_landscape.request_energy` to `max_rel_delta = 5.34e-06`. **Reduction, not re-implementation.**
- Byte ledger: store 524,288 B; R8 addressing head 475,136 B (reported beside it, per the pin's own
  convention); K=1 control **identical 524,288 B**, `σ = 17.379`, identity holds.

| arm | `accept_val` | ⭐`accept_A7` | `refuse_oor` | `accept_unseen` | **score** | threshold |
|---|---|---|---|---|---|---|
| **`CLU_gate@r8`** (the arm) | **1.0000** | 0.8056 | 0.3714 | 1.0000 | **0.6857** | ⭐ **DERIVED, no free parameter** |
| `CLU_K1@r8` (control c) | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 0.6000 | DERIVED |
| `knn_whole` (control a) | 0.9512 | 0.6667 | 0.6213 | 0.9613 | 0.7862 | FITTED q=0.95 |
| **`knn_factored`** (control b, **the fair rival**) | 0.7891 | **0.3889** | 0.8590 | 0.7907 | **0.8241** | FITTED q=0.95 |
| `margin_M` (control e) | 0.9590 | 0.9722 | 0.0398 | 0.9479 | 0.4994 | FITTED q=0.95 |
| `CLU_gate@posterior_mean` | 1.0000 | 1.0000 | **0.0000** | 1.0000 | 0.5000 | DERIVED (dead) |
| `CLU_gate@clamp_atom` | 1.0000 | 1.0000 | **0.0000** | 1.0000 | 0.5000 | DERIVED (vacuous) |

`score = ½(refuse_oor + accept_val)`; on DEV `accept_val` stands in for `accept_B`, which is held out.

**R1 paired bootstrap (4,000 resamples, paired over both populations):**
| arm vs rival | Δ | CI95 | verdict |
|---|---|---|---|
| vs **`knn_factored`** | **−0.1383** | [−0.1564, −0.1203] | ⛔ **LOSS** |
| vs `knn_whole` | −0.1005 | [−0.1106, −0.0904] | ⛔ LOSS |
| vs `margin_M` (control e) | **+0.1863** | [+0.1771, +0.1965] | ⭐ **WIN** |
| vs `CLU_K1@r8` (control c) | **+0.0857** | [+0.0828, +0.0887] | ⭐ **WIN** |

**Localisation** (single-factor `out_of_range`, chance = 1/6 = 0.1667):
`CLU_gate@r8` = **0.3441** · single-well control `CLU_K1@r8` = **0.1146** (at chance).
Per factor (arm): floor_hue 0.098 · wall_hue 0.217 · object_hue 0.116 · scale 0.560 · shape 0.444 ·
orientation 0.630.

## ⛔⛔ DECISION — **THE SCORED RUN IS NOT SPENT. Scored 0/3; three remain.**
- **2026-09-17T14:31Z** — §3 step 5 is explicit: *"Then, **only if the bar is met on DEV**, file PREREG-2
  and spend the one scored run."* **The bar is not met on DEV**, on two of its three legs:
  - **Pair:** ⛔ **LOSS**, not win-or-tie, vs the factored kNN gate (Δ = −0.1383, CI excludes 0 and
    |Δ| ≫ m = 0.020). ⛔ The bar is not mine to move and it has not been moved.
  - **`accept_A7 ≥ accept_B − 0.05`:** ⛔ **FAILS** on DEV — 0.8056 vs 1.0000, a gap of **0.194**.
  - **Localisation ≥ 0.50:** ⛔ **FAILS at 0.3441** — though it *does* beat the single-well control
    (0.1146, at chance) by a wide margin, which is the second half of that leg and it **passes**.
- ⭐ Spending the held-out run now would buy a number I already know will not clear the bar, and would
  cost the programme 1 of 3. ⛔ Not spent. **`eval_b` remains untouched by this hub.**

### ⚠ A PROCESS FAULT OF MINE, OWNED
§3 orders PREREG-1 **before any accept/refuse number**, and I produced UNIT 3's accept/refuse numbers
first. Mitigation, stated plainly rather than papered over: **the gate (§0) and the bar (§5) were both
fixed by the task file before launch and neither has been touched by me**; `eval_b` is untouched; and the
scored budget is unspent. PREREG-1 is filed next and it labels the UNIT 3 table **exploratory DEV**.
⛔ I did not choose the bar after seeing the number — I am *failing* a bar I did not set.

## UNIT 4 — CONTROL (d), THE LIVE SET. `LIVESET_seed0_A.json`
- **2026-09-17T14:38Z** — six stores rebuilt from `store_train ∖ rows(v)`, one held-out value per factor
  (rule: `v = cards[f] // 2`, the middle grid value). 219 s total. Rows removed: floor_hue 3,269 ·
  wall_hue 3,224 · object_hue 3,207 · scale 4,095 · shape 8,234 · orientation 2,142.
- ⭐ **§5.6 does NOT fire. `shortfall = 0` on all six rebuilds**, asserted not assumed: the cache is
  `scope="full"` (480,000 rows = the whole split) and every row touched is a `store_train` row already in
  it. ⛔ Nothing was encoded.
- ⭐ Every rebuilt store passes the identity (`max_i E_k(a_ki) ≤ 0`, worst −12.19 … −42.94) and carries
  **0 atoms with the held-out value** — asserted per rebuild, so the hold-out is real.

### ⛔⛔ THE RESULT — **exemplar refusal is DEAD. §5.9 FIRES ON ALL SIX FACTORS.**
| factor | wells | CLU refuse(held-out) | CLU refuse(in-vocab ctrl) | sep | CLU write-back accept | `accept_A7` |
|---|---|---|---|---|---|---|
| floor_hue | [4] | **0.0000** | 0.0000 | +0.0000 | 1.0000 | 0.8750 |
| wall_hue | [0,1,2] | **0.0000** | 0.0000 | +0.0000 | 1.0000 | 0.7917 |
| object_hue | [2,3] | **0.0000** | 0.0000 | +0.0000 | 1.0000 | 0.8472 |
| scale | [2,3,4] | **0.0000** | 0.0000 | +0.0000 | 1.0000 | 0.7917 |
| shape | [2,3] | **0.0000** | 0.0000 | +0.0000 | 1.0000 | 0.8750 |
| orientation | [0,1] | **0.0000** | 0.0000 | +0.0000 | 1.0000 | 0.8889 |
| **mean** | | ⛔ **0.0000** (bar ≥ 0.90 — **FAIL**) | 0.0000 | +0.0000 | ⭐ **1.0000** (bar ≥ 0.90 — **PASS**) | 0.8449 |

⛔ **The derived gate never fires on a real encoded image — not when every atom carrying that value has
been deleted from the store, and not otherwise.** Write-back trivially passes because nothing was ever
refused, so ⭐ **the write-back 1.0000 is NOT evidence for the arm and the report must say so**: it is the
same vacuous accept, measured on the other side.

### ⭐ AND THE FITTED RIVAL IS LIVE-SET-SENSITIVE WHERE THE DERIVED GATE IS NOT
| factor | `knn_factored` refuse(held-out) | refuse(in-vocab) | separation |
|---|---|---|---|
| floor_hue | **1.0000** | 0.3359 | **+0.664** |
| wall_hue | **1.0000** | 0.5820 | **+0.418** |
| object_hue | 0.8438 | 0.5898 | +0.254 |
| shape | 0.6484 | 0.6289 | +0.020 |
| scale | 0.5664 | 0.6289 | −0.063 |
| orientation | 0.3320 | 0.3398 | −0.008 |
| **mean** | **0.7318** | 0.5176 | +0.214 |

⭐ **This is the whole finding in one comparison.** The *same statistic family* (distance to the live
atoms), thresholded at a **fitted** `q = 0.95`, detects a deleted value on the three colour factors.
Thresholded at the **derived zero**, it detects nothing at all. ⛔ The problem is not the signal.
**It is the radius.** And §7 forbids me to move it — correctly, because moving it is the whole claim.

⭐ ⚠ And the rival pays for that sensitivity exactly where §7b says it does: `accept_A7` mean **0.3634**
against the CLU's **0.8449**. **The factored kNN gate falsely refuses 64 % of the valid A7
redundant-term control.** That is the programme-wide rule's premise, reproduced on this object, and it is
why no refusal number of either arm is reportable without its A7 companion.

## UNIT 5 — THE RADIUS ATLAS. `RADIUS_ATLAS_seed0_A.json` (2026-09-17T14:35Z, 79 s)
⭐ This unit is the **mechanism**, and it is one table.

### The three radii, all in the gate's own units `d/σ_k`
| radius | well 0 | well 1 | well 2 | well 3 | well 4 |
|---|---|---|---|---|---|
| rivals' **fitted** `q = 0.95` quantile | 1.150 | 1.138 | 1.053 | 1.122 | 1.285 |
| ⭐ **the derived ZERO LEVEL SET** (measured, 256 rays/well from stored atoms) | **2.093** | **2.043** | **2.419** | **2.343** | **1.892** |
| the certified bound `σ_k√(2 ln N)` | 2.884 | 2.884 | 2.884 | 2.884 | 2.884 |

⭐ The certified bound is **correct and tight-ish**: it sits 0.47–0.99 σ outside the true crossing, as a
sufficient condition should. ⛔ It is not the problem.

### Where everything actually lives
| population | n | mean `min_k d/σ_k` | mean `max_k E_k` | refuse |
|---|---|---|---|---|
| stored atoms | 64 | **0.000** | −42.25 | 0.0000 |
| **real features: all `store_train`** | 4096 | **0.787** | −38.10 | 0.0007 |
| **real features: `val` rows** | 512 | **0.793** | −37.81 | 0.0000 |
| CLU launch `clamp_atom` | 512 | 0.000 | −41.81 | 0.0000 |
| CLU launch `posterior_mean` | 512 | 0.814 | −54.34 | 0.0000 |
| R8 affine, valid | 512 | 0.775 | −44.41 | 0.0000 |
| R8 affine, A7 (valid) | 72 | 0.887 | −28.30 | 0.1944 |
| R8 affine, `out_of_range` w=2 | 3072 | 0.856 | −41.04 | 0.0000 |
| R8 affine, w=3 | 3072 | 0.939 | −27.66 | 0.1660 |
| R8 affine, w=4 | 3072 | 1.039 | −4.78 | 0.3333 |
| R8 affine, w=7 | 3072 | 1.403 | +133.5 | 0.6240 |
| R8 affine, w=13 | 3072 | 2.262 | +793.3 | 0.7337 |

## ⭐⭐⭐ THE ANSWER, IN ONE SENTENCE AND THREE NUMBERS
> **The whole data manifold sits at `d/σ_k ≈ 0.79`. The derived zero level set sits at `≈ 2.0–2.4 σ_k`.
> The fitted rival sits at `≈ 1.05–1.29 σ_k`, just above the data.**
> ⛔ **The derived gate's radius is ~2.6× further out than the data it is supposed to be guarding, so
> nothing that is actually an image can ever reach it** — which is exactly why UNIT 4 measured
> `refuse(held-out exemplar) = 0.0000` and why only a *linear extrapolation* at severity `w ≥ 3` crosses.

⭐ **And the reason is structural, not incidental.** `σ_k` is `nn_median` — the **median nearest-neighbour
distance between the atoms**. The atoms are train rows drawn from the same manifold, so they tile it at
spacing ≈ `σ_k` by construction, and any real point is ≈ 1 `σ_k` from one of them **whatever the store
contains**. ⛔ The level set at `√(2 ln N)·σ_k` is therefore pinned ~2.9× outside the manifold **by the
bandwidth rule itself**, and `N` only makes it worse (`√(2 ln 512) = 3.53`). ⛔ I may not scale `σ_k` and
I may not move the gate off zero (§7), and both prohibitions are right — moving either *is* the claim.

## UNIT 5 §3 — THRESHOLD-FREE AUROC. ⭐ This separates *the signal* from *the radius*.
| statistic | AUROC `oor` vs `val` | AUROC `oor` vs **A7** | Δ over margin `M` |
|---|---|---|---|
| **CLU `max_k E_k`** | **0.8619** | 0.6680 | ⭐ **+0.4436** |
| `knn_factored` `max_k d_k/thr_k` | **0.9196** | 0.7165 | +0.5013 |
| `knn_whole` latent `d` | 0.8604 | 0.7084 | +0.4421 |
| `margin M` (physics-free, control e) | **0.4183** | 0.4452 | +0.0000 |

- ⭐⭐ **The paper's own negative finding does NOT reproduce on this object.** Its applications section
  says the residual *"adds at most **+0.024 AUROC** over a standard readout margin"*. Here the energy
  signal adds **+0.4436**, and the readout margin is **below chance** (0.4183) — an out-of-range request
  makes the address-book posterior *sharper*, not flatter, so `M` is actively misleading. ⭐ That is a
  reportable disagreement with the cited paper, on a different object, and it is stated as such.
- ⛔ **But the CLU statistic is still behind the fair rival, threshold-free: 0.8619 vs 0.9196
  (Δ = −0.0577).** So the arm's DEV loss is ⛔ **not** merely an artefact of where the threshold sits.

## UNIT 5 §4 — THE FITTED ALTERNATIVE (§5 permits this: *reported beside, labelled, ⛔ never adopted*)
Threshold the **same** statistic `max_k E_k` at the `q = 0.95` quantile of `val` instead of at zero:
fitted threshold = **−35.998** (⭐ the derived threshold is **0.0** — the fitted one is 36 units lower).

| gate | `accept_val` | `accept_A7` | `refuse_oor` | score |
|---|---|---|---|---|
| ⭐ **DERIVED `E_k > 0`** (the arm, no free parameter) | 1.0000 | 0.8056 | 0.3714 | 0.6857 |
| fitted `max_k E_k > q95(val)` (⛔ **not adopted**) | 0.9492 | 0.5833 | 0.6472 | 0.7982 |
| `knn_factored` q=0.95 (the fair rival) | 0.7891 | 0.3889 | 0.8590 | **0.8241** |

⭐ **Even with its threshold fitted, the CLU energy statistic does not overtake the factored kNN gate**
(0.7982 vs 0.8241), exactly as the AUROC ordering predicts. ⛔ **The gate stays at `E_k > 0`.**

## UNIT 6 — CLOSE-OUT. `CLOSEOUT_seed0.json` (2026-09-17T14:45Z, 553 s)

### (i) ⭐ CAPACITY CONTROL — rung B, 8× the atoms and 8× the bytes. **It does not help.**
| | rung A (N=64, 524,288 B) | rung B (N=512, 4,194,304 B) |
|---|---|---|
| `σ_k` | 4.634 / 5.071 / 9.183 / 6.305 / 5.101 | 3.232 / 3.729 / 7.101 / 5.101 / 3.828 |
| empirical zero level set `/σ_k` | 2.12 / 2.04 / 2.42 / 2.33 / 1.87 | 2.50 / 2.46 / 2.89 / 2.81 / 2.44 |
| certified radius `/σ_k` | 2.884 | **3.532** (worse — `√(2 ln N)` grows) |
| ⭐ **real features sit at `d/σ_k`** | **0.786** | **0.739** |
| real features refused | 0.0005 | **0.0000** |
| `accept_val` | 1.0000 | 1.0000 |
| `refuse_oor` | 0.3714 | 0.3995 |
| ⭐ certified fraction of refusals | **0.6936** | **0.7264** |

⛔ **8× the store bought `+0.028` on `refuse_oor` and moved the level set FURTHER out in `σ` units.**
The ratio is scale-free exactly as the `nn_median` rule predicts: denser atoms ⇒ smaller `σ_k` ⇒ the data
still sits at ~0.75 `σ_k`. ⛔ **More capacity is not the fix, and there is no fix that §7 permits.**
⭐ The certified fraction is a genuine positive and is reported as §0 requires: **69–73 % of all refusals
are certified**, i.e. provably beyond `σ_k√(2 ln N)` whatever the atom configuration.

### (ii) OUT-OF-VOCABULARY, SYMBOL — ⛔ `CLU refuse = 0.0000` on all six factors
| factor | CLU refuse | CLU `d/σ_k` | `knn_factored` refuse |
|---|---|---|---|
| floor_hue | 0.0000 | 0.764 | **1.0000** |
| wall_hue | 0.0000 | 0.818 | 0.8887 |
| object_hue | 0.0000 | 0.758 | 0.1348 |
| scale | 0.0000 | 0.745 | 0.1582 |
| shape | 0.0000 | 0.739 | 0.1445 |
| orientation | 0.0000 | 0.712 | 0.1289 |
| **mean** | ⛔ **0.0000** | 0.756 | 0.4092 |

⭐ **And there is a structural reason, worth the programme's attention.** When a value is deleted the R8
prototype for it has **0 training rows** and `FactorPrototypes.fit` falls back to `μ_global`. So naming a
deleted value symbolically embeds the request **closer to the centre of the manifold** (`d/σ = 0.71–0.82`,
*below* the valid mean of 0.775 on four of six factors). ⛔ **Deleting a value makes the symbolic request
for it land MORE inside the store, not less** — so no distance-to-store criterion at any radius can be
relied on for OOV-symbol, and the arm's live-set claim cannot be carried by symbol mode either.

### (iii) ⭐⭐ A7 vs A7D — **the A7 false-refusal is the R8 HEAD, not the store. One line, and it is fixable.**
| | `accept_A7` (7 terms) | `accept_A7D` (deduplicated, 6 terms) | Δ |
|---|---|---|---|
| **CLU `gate@r8`** | 0.8056 (`d/σ` 0.887) | ⭐ **1.0000** (`d/σ` 0.780) | **+0.194** |
| `knn_factored` q=0.95 | 0.3750 | **0.8750** | **+0.500** |

⭐⭐ **Every one of the CLU's A7 false refusals disappears under `gen_request.deduplicate_terms`, and
half of the factored kNN's do.** The cause is R8's declared composition rule
`x̂ = μ_global + Σ_terms (μ_{k,v} − μ_global)`: a **duplicated term contributes its deviation twice**, so
a perfectly valid redundant request is pushed off-manifold by an extra `(μ_{k,v} − μ_global)`.
⭐ **This is a concrete, measured cause for the phenomenon the programme-wide §7b rule was written to
expose** — *"three of five published rivals falsely refuse ~60 % of the valid A7 redundant-term control"*.
They share the R8 head, so they inherit the same double-count, and their tighter radius amplifies it.
⛔ Not my file to change (`gen_request.py` is manifest-pinned) — **logged as a finding for the coordinator.**

### (iv) LOCALISATION ANATOMY — the pre-registered rule is ANTI-correlated with refusal
| severity | accuracy | refuse rate | acc \| refused | acc \| accepted |
|---|---|---|---|---|
| w=2 | 0.3167 | 0.0000 | — | 0.3167 |
| w=3 | 0.3503 | 0.1660 | **0.0529** | 0.4094 |
| w=4 | 0.3600 | 0.3333 | **0.2021** | 0.4390 |
| w=7 | 0.3776 | 0.6240 | **0.2212** | **0.6372** |
| w=13 | 0.3158 | 0.7337 | **0.1633** | **0.7359** |

⛔ **The rule localises worst exactly where it is needed.** On refused rows it runs at 0.05–0.22 (chance
is 0.167); on accepted rows it reaches **0.736**. Mechanism: once a point is refused it is ≥ 2 `σ_k` from
every atom in every well, so *all* wells are uncertified and *every* term residual saturates — the
`argmax` over terms stops carrying information. ⛔ Pooled accuracy **0.3441**, bar **≥ 0.50: FAILED**;
it does beat the single-well control (0.1146, at chance) by +0.230, which is the leg that passes.
⛔ The rule was pre-registered in PREREG-1 §7 and is ⛔ **not** re-specified after seeing this.

## UNIT 7 — UNCERTAINTY, and the last escape hatch closed. `UNCERTAINTY_seed0_A.json`
- **2026-09-17T14:50Z**, 114 s.

### (a) ⛔ SEED SPREAD IS UNMEASURABLE ON THIS BOX — stated, not implied away
The §5 localisation leg says *"beats the single-well control **beyond seed spread**"*. ⛔ **No seed spread
can be computed here.** Seeds 1 and 2 have caches with `scope="used"` (9,288 rows); covering the cell
needs ~32,200 new rows, which is a **re-encode** and is forbidden. What I can compute is a **within-seed
paired bootstrap over the 15,360 requests**, and it is labelled as exactly that:

`localisation arm = 0.3441` · `K1 = 0.1146` · **gap = +0.2295, CI95 (within-seed) = [+0.2222, +0.2367]**.
⭐ The arm beats the single-well control decisively **within seed**. ⛔ Bar ≥ 0.50: still **FAILED**.
⛔ I will not call this "beyond seed spread" — I did not measure seed spread.

### (b) ⭐⭐ THE MATCHED-OPERATING-POINT COMPARISON — ⛔ reported beside, the bar is unchanged
The fair objection to §2's table is *"the pre-registered score rewards the rival's own choice of `q`,
which parks it at a 21 % refusal rate on valid data."* ⭐ **That objection is answerable, and the answer
does not help the arm.** Sweeping the rival's free parameter:

| rival | q | `accept_val` | `refuse_oor` | score |
|---|---|---|---|---|
| `knn_factored` | 0.95 | 0.7891 | 0.8590 | 0.8241 |
| `knn_factored` | **0.99** | 0.9648 | 0.7488 | **0.8568** ← the rival's best |
| `knn_factored` | 0.999 | 0.9922 | 0.6796 | 0.8359 |
| `knn_factored` | **1.0** | **1.0000** | **0.6443** | 0.8222 |
| `knn_whole` | 0.95 | 0.9512 | 0.6213 | 0.7862 |
| `knn_whole` | 0.999 | **1.0000** | 0.4861 | 0.7431 |
| ⭐ **the arm** (derived, **no free parameter**) | — | **1.0000** | **0.3714** | 0.6857 |

**At MATCHED `accept_val = 1.0000`** (rival's `q` solved by bisection):
| | `accept_val` | `refuse_oor` |
|---|---|---|
| `knn_factored`, q = 0.999991 | 1.0000 | **0.6449** |
| `knn_whole`, q = 0.994470 | 1.0000 | **0.5188** |
| ⭐ **the arm** | 1.0000 | ⛔ **0.3714** |

⛔⛔ **The arm loses at EVERY operating point, not merely at the pre-registered one.** The "the comparison
was unfair on operating point" defence is closed with a number. ⭐ The rival's best possible score is
0.8568; the arm's is 0.6857; and even the arm's own statistic with a fitted threshold reaches only 0.7982.
⛔ This changes nothing about the verdict — **the bar is the bar** — but it means the negative is
**robust**, not an artefact of threshold placement. That is worth more than the bar would have been.

## UNIT 8 — THE A7-CONTROLLED PAIR. `A7_PAIR_seed0_A.json` (2026-09-17T14:52Z, 91 s)
⛔ **Reported BESIDE the bar. It is NOT the bar and it does NOT change the verdict** — the bar's accept
population is **B**, not A7. This unit exists because §7b makes a claim about a number, so I measured it.

`score_X = ½(refuse_oor + accept_X)`

| arm | q | `acc_val` | ⭐`acc_A7` | `acc_A7D` | `ref_oor` | `sc_val` | `sc_A7` | `sc_A7D` |
|---|---|---|---|---|---|---|---|---|
| ⭐ **CLU `gate@r8`** (DERIVED, no free parameter) | — | 1.0000 | **0.8056** | **1.0000** | 0.3714 | 0.6857 | 0.5885 | 0.6857 |
| `knn_factored` | 0.95 | 0.7891 | **0.3889** | 0.9028 | 0.8590 | **0.8241** | 0.6240 | 0.8809 |
| `knn_factored` | 0.99 | 0.9648 | 0.4861 | 0.9722 | 0.7488 | **0.8568** | 0.6174 | 0.8605 |
| `knn_factored` | 0.999 | 0.9922 | 0.5556 | 0.9861 | 0.6796 | 0.8359 | 0.6176 | 0.8328 |
| `knn_factored` | **1.0** | 1.0000 | ⛔ **0.5833** | 1.0000 | 0.6443 | 0.8222 | 0.6138 | 0.8222 |
| `knn_whole` | 0.95 | 0.9512 | 0.6667 | 0.9722 | 0.6213 | 0.7862 | 0.6440 | 0.7968 |
| `knn_whole` | 0.9999 | 1.0000 | 0.8750 | 1.0000 | 0.4804 | 0.7402 | **0.6777** | 0.7402 |

⭐⭐ **§7b's premise is confirmed with a number, and it is stronger than "a rival looks better than it is".**
⛔ **`knn_factored` still falsely refuses 41.7 % of the valid A7 control even at `q = 1.0`** — i.e. with its
threshold set to the *largest distance any valid calibration corner produced*. A7's embeddings are
therefore **outside the entire valid distance range**, and no choice of `q` recovers them.
⭐ **And `accept_A7D` is 0.90–1.00 for every arm at every `q`.** So the A7 false-refusal is **entirely**
R8's duplicated-term double-count (UNIT 6 iii), for the rivals as well as for the CLU.

**R1 on the A7-controlled pair vs `knn_factored` at its pre-registered `q = 0.95`:**
**Δ = −0.0355, CI95 = [−0.0908, +0.0225] → TIE.**
⚠ **I am NOT banking this as a tie on the bar, and it must not be quoted as one.** The bar's accept
population is the held-out **B**, not A7; this substitutes a different population on DEV. ⛔ Against the
*best* rival configuration the arm still loses the A7-controlled pair (0.5885 vs 0.6777, Δ = −0.0892).
⭐ What it does show honestly: **on the A7-controlled comparison the gap between the arm and the fair
rival narrows from −0.1383 to −0.0355** — the arm's one real advantage is that it does not over-refuse
valid redundant requests, and that advantage is worth about 0.10 of score. ⛔ Not enough.

## UNIT 9 — THE BAR'S ACTUAL DEFINITION, AND FREE MODE. `POOLED_SUPPORT_seed0_A.json` (2026-09-17T14:58Z, 301 s)

### (a) ⛔ `out_of_support` POOLED — and it is WORSE for the arm than what UNIT 3 reported
§5 defines the pair on `refuse_out_of_support`, where *"out-of-support pools out-of-range **and
out-of-vocabulary**"*. UNIT 3 reported `refuse_oor` alone. ⛔ The pooled number is the bar's, and I report
it at full strength — a bar evaluated on a favourable subset is not the bar.

| gate | `oor` (n=15,360) | `oov_symbol` (n=3,072) | `oov_exemplar` (n=3,072) | ⭐ **POOLED** (n=21,504) |
|---|---|---|---|---|
| **CLU `gate@r8`** | 0.3714 | **0.0000** | **0.0000** | ⛔ **0.2653** |
| `knn_factored` | 0.8590 | 0.4118 | 0.7288 | **0.7766** |
| `knn_whole` | 0.6213 | 0.0446 | 0.2546 | 0.4865 |

**R1 on the POOLED pair:** vs `knn_factored` **Δ = −0.1502, CI95 [−0.1675, −0.1317] → LOSS**;
vs `knn_whole` Δ = −0.0862 [−0.0955, −0.0762] → LOSS.
⛔ **The bar's own definition makes the arm's loss larger** (−0.1383 → −0.1502). Reported, not buried.

### (b) ⭐ FREE MODE — the last unmeasured mode, and the arm's cleanest genuine advantage
"Free mode" is nowhere defined in a query set; taken as its only sensible reading — a request naming only
*some* of the six factors. These are **valid, under-specified** requests and must be **accepted**.

| factors left free | terms | ⭐ **CLU accept** | `knn_factored` accept | `knn_whole` accept |
|---|---|---|---|---|
| 1 | 5 | ⭐ **1.0000** | ⛔ **0.5488** | 0.9668 |
| 2 | 4 | ⭐ **1.0000** | ⛔ **0.3633** | 0.9297 |
| 3 | 3 | ⭐ **1.0000** | ⛔ **0.2168** | 0.8027 |

## ⭐⭐ THE COHERENT CHARACTERISATION — stated as a property, ⛔ not as a metric I chose after the fact
Every "valid request" population measured, at every launch point, at every severity:

| valid population | ⭐ CLU `gate@r8` accept | `knn_factored` accept |
|---|---|---|
| `val` corners | **1.0000** | 0.7891 |
| `unseen_value` (interpolated) | **1.0000** | 0.7907 |
| A7 (redundant term) | 0.8056 → ⭐ **1.0000** deduplicated | 0.3889 → 0.9028 |
| free mode, 1 / 2 / 3 factors free | **1.0000 / 1.0000 / 1.0000** | 0.5488 / 0.3633 / 0.2168 |
| write-back after deletion | **1.0000** | — |

> ⭐ **The derived gate never falsely refuses a valid request, in any mode this hub could construct.**
> ⛔ **And it refuses only 0.2653 of the invalid ones.** Both facts have the same cause: its radius sits
> outside everything. It is an excellent *"do not falsely refuse"* gate and a poor *"refuse the invalid"*
> gate.

⛔⛔ **This does NOT rescue the bar and I am not presenting it as if it did.** The pre-registered score is
**balanced**, and on a balanced score the arm **loses, at every operating point, on the pooled definition,
against both rivals**. I state the asymmetry because it is the honest characterisation of the object and
because it is the one place a future design could start — ⛔ not because it changes the verdict.

## ⭐ A FORECLOSED OBJECTION — the well-combination rule (2026-09-17T14:58Z)
The directive fixes the **per-well** rule (`refuse well k iff E_k > 0`) but never says how the five wells
combine into one admission decision. I used **"refuse iff ANY well refuses"**. Every other monotone
reduction (majority, the requested term's well only, all wells) refuses a **subset**, so every
`refuse_out_of_support` figure in this log is an **upper bound over the whole family** — and since
`accept_val = 1.0000` under this reduction, the usual cost of the strictest rule is **exactly zero** here.
⛔ **No choice of well-combination rule improves the arm.** Logged so it cannot be raised later as a
reason the negative is soft.

## U6 — THE CERTIFIED-ACCESS FIGURE (DEV, unscored)  [sub-agent, 2026-09-17]

`scorer/u6_figure.py` -> `CERTIFIED_ACCESS_FIGURE_seed0_A.json`, `certified_access_seed0_A.png`,
`CERTIFIED_ACCESS_NOTES.md`.  192 `val` rows, own CACHED feature `x0` from `ctx["X_store"]`
(NOTHING re-encoded, `eval_b` never read), walked along a fixed unit ray, direction seed 20260917,
22 radii, `min_k d/sigma` sweeping 0.81 -> 5.05.

CROSSINGS (mean `min_k d_k/sigma_k`, 50% of rays; decode collapse = midpoint acc(t=0)->chance):
  kNN factored q=0.95   1.0009   (plain-decode acc there 0.975)
  kNN whole    q=0.95   1.1948   (0.946)
  zero level set E_k=0  1.5548   (0.879)
  CERTIFICATE FLIPS     2.0603   (0.800)
  DECODE COLLAPSES      4.0809   (0.558 = midpoint; below 0.5 at 4.7277)
  => the certificate flips 2.02 units EARLIER than the decode collapses.

Decode leg RAN: decoder = gen-decoder-lift winner v1_capacity_schedule (hash 01ef91f7...),
classifier = gen-external-classifier gated instrument (hash c8741ac4...).  matplotlib 3.10.8 present.
CLU read (settle_chain at x(t) with the row OWN request, 1000 steps) is FLAT at 0.3507 at every
radius; the request-off control is FLAT at 0.1198 (chance 0.1236) — on this rung the settled state
is set by the request term, not by the launch.  Self-checks passed: identity_E_le_0 (-30.39 <= 0),
term_experts == shipped request_experts (bit-identical).  kNN thresholds fitted on 4096 fit_rows
cached features only.  No assertion fired.  NO CLAIM BEYOND THE FIGURE.

## SUB-AGENT S1 (Opus) — THE CERTIFIED-ACCESS FIGURE, §3 step 6. Delivered.
`certified_access_seed0_A.png` · `CERTIFIED_ACCESS_FIGURE_seed0_A.json` · `CERTIFIED_ACCESS_NOTES.md` ·
`scorer/u6_figure.py`. 192 `val` rows, each walked from its **own cached feature** along one fixed unit
direction (seed 20260917), 22 radii, `min_k d/σ_k` 0.81 → 5.05. ⛔ Nothing re-encoded, ⛔ B never read.
⭐ **The decode leg RAN** on the shipped decoder (`gen-decoder-lift` winner, hash-asserted) and the shipped
external classifier, scored by `gen_probe.probe_accuracy`. Chance 0.1236.

| marker | `min_k d/σ_k` | plain-decode accuracy |
|---|---|---|
| kNN factored q=0.95 | 1.00 | 0.975 |
| kNN whole q=0.95 | 1.19 | 0.946 |
| **zero level set** | **1.55** | 0.879 |
| ⭐ **certificate flips** | **2.06** | **0.800** |
| ⛔ **plain decode collapses** | **4.08** | 0.558 |

⭐ **The certificate flips 2.02 units EARLIER than decode quality collapses.** The gate refuses while the
decoder still scores 0.800. ⭐ With §6b that completes the characterisation: **on-manifold it accepts
everything; off-manifold it refuses well before quality dies** — conservative in the safety direction,
blind in the validity direction.

⚠ Riders, both logged rather than smoothed over:
- The crossing reads **1.55** here vs **1.89–2.42** in UNIT 5 because these rays start from a **`val` row**
  (already at 0.79 σ) and UNIT 5's start from a **stored atom** (at 0). ⭐ Same surface, different origin —
  ⛔ not a discrepancy, and I checked before reporting.
- ⛔ **The CLU read is FLAT at 0.3507 at every radius**; the request-off control is flat at 0.1198
  (≈ chance). The settled state is fixed by the **request term**, not by the launch point. ⇒ gating the
  launch does not gate the read, so the claim's property 4 — ***"one object, two outputs"*** — is
  ⛔ **NOT demonstrated on this rung**, and the report says so rather than asserting it.

- **2026-09-17T15:00Z** (`date -u`) — ⚠ **Second clock correction, same failure as the first.** Four unit
  headings above carried stamps I had written from my own sense of elapsed time (`14:36`, `14:38`, `14:49`,
  `14:56`, `15:00`); the real clock values from the `date -u` / `elapsed.sh` output printed alongside each
  run are `14:35`, `14:45`, `14:50`, `14:58`, `14:58`. All five corrected in place. ⭐ Recorded rather than
  silently fixed, because §-0 flags exactly this failure and it has now happened to me **twice** in one
  hour. ⛔ The unit **durations** (79 s, 553 s, 301 s …) were always machine-measured and are unaffected.
- **2026-09-17T15:00Z** — integrity check: `git status --porcelain` in the bundle is **empty**, and
  `git diff 945a3e9 HEAD -- chlu/` in the worktree is **empty**. ⭐ **No manifest-pinned file was touched
  by me or by either sub-agent. Nothing was deleted.**

## SUB-AGENT S2 (Opus) — ADVERSARIAL VERIFICATION. `VERIFY_ADMISSION.md` (2026-09-17T15:09Z)
Mandate: **try to break my own negative.** Every gate number recomputed **from the PREREG-1 §1 formula in
float64 numpy**, without calling `admission_lib`. ⭐ **Nothing overturned.** Both corrections run
**against** the arm, and it found **one real bug in my code**.

### ⛔ BUG IN MY CODE, OWNED — `u4_liveset.py:99`: the §5.6 assertion is a TAUTOLOGY
`shortfall = setdiff1d(pool[keep], pool).size + setdiff1d(pool[held_mask], pool).size` — **both terms are
empty by construction**, so the check could not fail. `rec["shortfall_total"]` is never reassigned, so
`F5_6_fires` was **hard-wired `False`**. ⛔ **I reported a passing check that was incapable of failing, and
that is exactly the kind of thing this hub exists not to do.**
⭐ **The CONCLUSION survives, by an independent route the verifier supplied:** `build_landscape` receives
`X_pool[keep]`, a slice of what `gather_features` **already returned**, so `gather_features` is never
re-entered and no uncached row can be requested. **`shortfall = 0` is true; my test for it was worthless.**
⚠ The report now states §5.6 as *"true, but established by construction and by the verifier's route — ⛔ not
by my assertion, which was vacuous."*
Two lesser findings, both recorded: `rates()` is called 3× per population in UNIT 4 (wasteful, not wrong);
`per_well_nearest` uses the float32 Gram form, `max |Δ|` vs float64 = **0.011** on `d ≈ 4–10` — **0
accept/refuse disagreements over 1024 probes**, but ⛔ `d/σ_k` is not quoted beyond 3 dp anywhere.

### ⚠ CORRECTION 1 — **my loss is UNDERSTATED by 0.039**
`q = 0.95` per column with the ANY-of-5 reduction gives the rival a **21.6 %** family-wise refusal on
calibration, which **handicaps it**. Its score peaks at **0.8621 at `q = 0.98`** (family-wise 9.4 %).
⛔ **Δ at the rival's own best operating point is −0.1772, not −0.1383.** The `DEV_TABLE` comparison is
**fair and conservative, erring in the arm's favour**. Corrected in the report.

### ⚠ CORRECTION 2 — ⛔ **"the derived gate never fires" is TOO STRONG, and there IS a launch point I missed**
**The SETTLED state** (`settle_chain`, shipped `SettleCfg`, 1000 steps, from the shipped `clamp_atom`
launch) is shipped, request-only, non-fitted, gate at zero, `σ_k` unscaled — and I did not test it.
⛔ **It is not vacuous.** But it does not rescue anything:

| gated at the SETTLED state | refuse |
|---|---|
| **valid `val`** | ⛔ **0.1094** |
| `oor` w=2 / w=3 / w=4 / w=7 / w=13 | 0.0000 / 0.0000 / 0.0234 / 0.1641 / 0.3359 |

⛔ **It refuses MORE valid requests than invalid ones at w = 2, 3 and 4** — strictly dominated by
`gate@r8_affine`, and not an arm. Live-set separation from my own hybrid on my own rebuilds:
**−0.0391** (floor_hue) and **−0.0469** (wall_hue) — **negative**. OOV-symbol: −0.0547 vs +0.2266,
sign-inconsistent. ⭐ **My conclusion survives: there is no launch point at which `E_k > 0` discriminates
a held-out value.**

### ⭐⭐ AND THE MECHANISM IT FOUND IS A SHIPPED-CODE FACT WORTH PASSING UP
`total_energy = E_store + β·Σ_t (E_t − E_store)`, so at `β = 1` with `T` requested terms the store's own
free energy enters the settling dynamics with weight **`1 − T`**. For a six-factor `shapes3d` request
that weight is **−5**: ⛔ **settling pushes the state AWAY from the atoms by construction.** Measured on
valid `val`: `mean max_k E_k` rises **−36.9 → −24.7** and `min_k d/σ_k` rises **0.000 → 0.849** over the
1000 steps. ⭐ That is why the settled-state gate fires on valid data at all, and it is a defect in the
**read**, not the gate. ⛔ Not my file to change — **raised for the coordinator.**

### ⭐ CONFIRMED (verifier's own float64 numbers beside mine)
- **Live-set refusal 0.0000, all six factors.** Closest any query came to the gate: `max_k E_k = −9.88`.
- ⭐ **The strongest variant I under-tested: the FULL unmodified 2048-dim cached feature of a held-out
  image against the REBUILT store refuses 0.0007** — *less* than the same images against the **undamaged**
  store (**0.0013**), both at the generic real-feature false-refusal floor. ⛔ **My claim is not too
  strong**; hybrid `0.0000` and full-feature `0.0007` are the same measurement: nothing.
- Hold-out is real: 0 leaking atoms; rebuilds share only **0–1 of 64** atoms with the full store.
- Hybrid construction bit-exact in exactly the assigned wells (17.9–70.3 % of dims), correctly row-matched.
- **Radius atlas**: manifold **0.7852 / 0.7934**; level set **2.090 / 2.038 / 2.410 / 2.342 / 1.901**;
  `σ` reproduced to **4.5e-08**; ⛔ no sign, `√` or `2σ²` error; float16 vs float32 identical to 16 digits.
- ⭐ **The obvious objection to random rays cuts my way**: the crossing along a ray *toward a real feature*
  — the data direction — is **2.04–2.41 σ**, i.e. **no closer**.
- **A7 is airtight**: 72/72 the same requests, exactly one duplicate at the same value in 72/72,
  `embed(A7) − embed(A7D)` equals one `(μ_kv − μ_global)` to **2.98e-07**, 14 refused → 0, `d/σ`
  **0.8871 → 0.7798**, landing on the valid mean **0.7753**.
- `paired_bootstrap`: Δ = **−0.13834635** exact, CI [−0.1563, −0.1198], pairing genuinely implemented —
  though ⚠ **inert** (unpaired sd 0.00917 vs paired 0.00927), because the arm's `accept_val` is constant.

### ⭐⭐ ONE NEW FINDING FROM THE VERIFIER, AND IT IS THE BEST THING IN THIS RUN FOR RE-IDEATION
Operating-point-free (max over **all** thresholds of `½(refuse_oor + accept_val)`):

| statistic, same store, same wells | best score | AUROC |
|---|---|---|
| CLU `max_k E_k` — **the free energy** | 0.7994 | 0.8619 |
| ⭐ CLU `max_k d_k/σ_k` — **plain nearest-atom distance** | ⭐ **0.8488** | ⭐ **0.9107** |
| `knn_factored` (fitted rival) | 0.8621 | 0.9198 |

⭐⭐ **The `logsumexp` over 64 atoms DESTROYS separation relative to the plain nearest-atom distance in the
same wells.** The CLU's own distance is a **better** statistic than the energy it is built from, and lands
within **0.014 AUROC** of the fitted rival.
**Loss decomposition:** of the arm's **0.177** deficit, **0.114 is the RADIUS** and **0.063 is the SIGNAL**.
⭐ Both halves of my reading are quantitatively right, and now they are quantified separately.

## UNIT 10 — THE CAPACITY CONTROL ON THE HEADLINE CLAIM. `LIVESET_RUNGB_seed0.json` (15:20Z, 548 s)
UNIT 6 ran capacity on `refuse_oor` but ⛔ **never on the live-set claim, which is the headline**. The
obvious objection — *"a bigger store would notice the deletion"* — deserved a measurement, not just my
scale-free argument. Factors chosen **before running**, for a stated reason: `floor_hue` (the only
singly-assigned factor, and where the rival scored a perfect 1.0000) and `wall_hue` (the rival's second
best). ⭐ If the derived gate notices a deletion at any capacity, it is here.

| factor | rung | N | store bytes | **CLU separation** | `knn_factored` separation |
|---|---|---|---|---|---|
| floor_hue | A | 64 | 524,288 | **+0.0000** | +0.6641 |
| floor_hue | **B** | **512** | **4,194,304** | **+0.0000** | +0.7539 |
| wall_hue | A | 64 | 524,288 | **+0.0000** | +0.5117 |
| wall_hue | **B** | **512** | **4,194,304** | ⚠ **+0.1719** | +0.6523 |

### ⚠⚠ A CORRECTION TO MY OWN AUTOMATED VERDICT, AND TO MY OWN EARLIER WORDING
1. ⛔ **`LIVESET_RUNGB_seed0.json` contains `"capacity_rescues_liveset": true`. That flag is MISLEADING
   and I am not letting it stand unexplained.** It tests `separation ≥ 0.05` — the **§5.9 falsifier**
   threshold — ⛔ **not the §5 bar, which is `refuse ≥ 0.90`.** The artefact is left in place unaltered
   (§0.1), and **this bullet is the correction**: ⛔ **capacity does NOT rescue the live-set claim.**
   `+0.1719` against a bar of **0.90** is a miss by 0.73, on **one factor of two**, for **8× the bytes**,
   while the fitted rival reaches **1.0000** at **both** rungs for **no extra bytes at all**.
2. ⚠ **But it is the first non-zero separation this hub has measured, and my earlier "there is no capacity
   fix" was too absolute.** The honest version: at rung B `σ` shrinks (well 0: 4.64 → 3.08), the held-out
   queries move out from `d/σ = 0.829` to **0.922**, and the closest query goes from `max_k E_k = −7.10`
   to **+9.17** — it **crosses**. The in-vocab control stays at **0.0000**, so the +0.1719 is **genuine
   separation, not noise**. ⭐ **Capacity helps. It helps far too slowly to matter**: ~**+0.09 in `d/σ`
   per 8× store**, while the level set itself moves out from 2.09 σ to 2.50 σ over the same step.
3. ⭐ `floor_hue` — the **better**-localised factor, and the rival's best — stays at **exactly 0.0000 at
   both rungs**, with the closest query at `max_k E_k = −13.91`. ⛔ The effect is not where the map says
   the signal is.

⭐ **Net: the headline negative stands, and is now measured rather than argued at both rungs.** The
wording in the report is corrected from *"there is no capacity fix"* to *"capacity buys +0.17 on one
factor of two for 8× the bytes, against a bar of 0.90"*.

## CLOSE — 2026-09-17T15:20Z
- **Elapsed true wall clock: 1.04 h** of 12.000 h, from `elapsed.sh` against `START_UTC.txt`. ⛔ Not
  model-perceived time (which ran ~7× fast, twice — both corrections logged above).
- ⭐ **Verdict: NOT SOLVED, truthfully.** The derived gate does not calibrate on this object, and the
  report says by exactly how much, at both rungs, with the mechanism measured.
- ⭐ **Scored runs 0/3. `eval_b` never read, never embedded, never scored.**
- ⚠ **I am stopping at 1.04 h rather than filling the window.** Every mandated deliverable is delivered,
  all five falsifiers are answered with numbers at both rungs, all six controls are run, and the result
  has been adversarially re-derived in float64 by a second Opus agent which overturned nothing and found
  two corrections that make the negative **stronger**. Every escape hatch — operating point, pooled
  population, well-reduction, capacity (both rungs), launch point (including the settled state I had
  missed), fitted threshold — is closed by measurement. ⛔ **The only remaining moves are the ones §7
  forbids.** The deadline is a cap, not a quota, and a manufactured eleven hours is the one unrecoverable
  failure. ⭐ The single thing more time could buy — a seed-1/seed-2 replication — is **unreachable here**
  (`scope="used"` caches; covering them is a re-encode).
- ⭐ **Confirmed before finishing: no sub-agent of mine is still running.** Both returned (S1 the figure,
  S2 the verification).
- ⛔ Nothing was deleted. ⛔ No manifest-pinned file was edited (bundle `git status` empty;
  `git diff 945a3e9 HEAD -- chlu/` empty). ⭐ All writes are under `$CHLU_HUB_OUTPUTS/hub-admission/**`
  and the worktree branch `hub/admission`.

## DELIVERABLES — 2026-09-17T15:22Z
| | |
|---|---|
| **Return pack** | `/c/Users/PratikJawahar/chlu/return-hub-admission.tar.gz` — 54 files, 1,361,863 B, `sha256 = 0c02e6bb0712a8a7afa79272947940b20c9d38e2fa5ec532ee391af043e05a8a`. ⭐ AppleDouble check **PASSED: 0 members**, read with python (⛔ not with tar). Ingest on the primary with `--return-ingest`. |
| **Git bundle** | `/c/Users/PratikJawahar/chlu/hub-admission.bundle` — **3,713,543 B**, ⭐ **whole-branch** (⛔ never a range, per §6). `git bundle verify`: *"okay … records a complete history"*, ref `2fc8fcac…` = `refs/heads/hub/admission`. |
| **Branch** | `hub/admission` @ `2fc8fca`, three commits on top of `945a3e9`. Working tree clean. |

### ⚠ ONE ENVIRONMENT FIX, LOGGED WITH ITS REASONING (a small decision, mine under §-1)
`--return-pack` aborted twice with
*"FATAL: the archive … carries AppleDouble members — ⛔ refusing to ship a tar that breaks git on the far
side"*. ⛔ **That FATAL was false, and I checked before doing anything about it.** `assert_no_apple_double`
found **zero** AppleDouble members; what failed was its own **success message**, which contains a `⛔` and
was being printed to a **cp1252** stdout by `PY_BIN=python3` (the system Python 3.12) —
`UnicodeEncodeError: 'charmap' codec can't encode character '⛔'`. The non-zero exit then tripped
`die`. ⭐ Fix: `BUNDLE_PYTHON=$CHLU_ROOT/.venv/Scripts/python.exe` + `PYTHONUTF8=1`, which is exactly the
`PYTHONUTF8=1` that LAUNCH-COMMON §2 already flags as load-bearing on this box.
⛔ **The check was made to RUN, not bypassed** — it then passed on its own terms and printed
*"apple-double: 0 members"*. ⛔ I did not pass a skip flag and I did not weaken the assertion.
⚠ Worth passing up: **`scripts/bundle_hub_workspace.sh` cannot complete on this Windows box with the
default `PY_BIN`**, and the failure mode impersonates a data-integrity refusal. Any hub that hits it and
takes the message at face value will think its outputs are corrupt when they are not.
