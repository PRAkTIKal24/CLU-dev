# c2w11-physics-organizer — experiment-engineer report

Task + acceptance criterion: build the C2W11 physics organizer (label-free loss package (a)–(e), the
DeepSets set-level ψ, the graded-novelty channel, the anytime curve), **re-run K5 on the organized arm
FIRST and BLOCKING**, then the three false-positive guards, then M3/M7/M8 and the traversal trigger.
**Status: done** — with the registered stop taken.

## ⛔⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
| # | site | what must change | owner |
|---|---|---|---|
| **R1** ⛔⛔ | the wave's read protocol / `PREREG-C2W11.md` §3 | **The binding cap is the LAUNCH HEAD, and it is now bounded, not just observed.** A matched filter over `N_a = 32` unit codes in `d_addr = 4` recovers **0.2439 ± 0.0227** of the needed channels, and the **best bijective re-placement of wells onto cue sites — the ceiling on ANY placement organizer — is 0.3290 ± 0.0074** (row-max, non-bijective, 0.3818 ± 0.0075). Exact-set needs ~4 of 4 wells (one missing payload contributes `‖v_j‖ = 0.60 > tol = 0.287`), so `0.3290 × 4 = 1.32 of 4` **cannot** clear it. ⇒ the organizer-swap question is **unaskable at `d_addr = 4`** and the next wave's dial is `d_addr` / the launch head, not the organizer | Hub → C2W12 scoping |
| **R2** ⛔ | `PREREG-C2W11.md` §5 V2's designed negative | *"permuted payloads ⇒ AUROC ≈ 0.5"* is **structurally non-discriminating** for a **depth-keyed** novelty channel (permuting `v_j` leaves every well written and every depth intact). Measured **0.7903 ± 0.0397** while blank-store gives **0.5029** and shuffled-labels **0.4972**. Pytest-asserted **as measured**, with the two biting negatives shipped beside it | curator + Hub |
| **R3** ⛔ | `PREREG-C2W11.md` §5 V3-MECHANICS / N199's flat-curve rule | **A flat anytime curve does NOT imply the store carries nothing.** Same store, same physics, same budget grid: shipped read **flat at 0.0004**; **oracle-addressed read 0.0223 → 0.8219 → 0.8711 → plateau**. Flatness must be **attributed** (addressing vs emptiness) before it is read as N199's signature | Hub |
| **R4** | charter §A34.9(c) / M7 | The spectral-shape defender is **measured harmful at this operating point**: it softens `λ_min` 5.0352 → 0.7196 and raises the participation ratio 3.889 → 7.576, but produces **0.0125** genuine soft directions and destroys addressability (M3 **0.9402 → 0.0810**, `any_basin` 0.9835 → 0.0803). §A4.2's refutation reproduces on a **different instantiation** | Hub → C2W12 |
| **R5** | `.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md` | **§2 appended (2026-08-11): the traversal trigger FIRED**, 0.7715 ± 0.0081 vs the registered 0.20 | done, FYI |
| **R6** | this spoke's own `PREREG.md` | **AMENDMENT §A1 filed before the claim cells** (term (a) gains the `reach` instantiation; the jig bound moves from `0.75 σ_q` to `0.75 ×` the measured capture radius). Both instantiations were raced | FYI |

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** **TIER ii — the physics organizer.** ⛔ No `OD`, no `OD_min`, no swap verdict, no
  paper number, no tier-ii verdict, no full-CLU verdict. Everything below is the **physics side**.
- **Laundering control:** ⛔ no launder margin is a pass condition anywhere. Reported as **DIAGNOSTIC**:
  the launch-only launder (address-leak dividend **−0.0854 ± 0.0043**), the settle-deleted launder, the
  oracle-addressing ceiling, and a byte ledger beside every reading.
- **Falsifies:** a MECHANICS leg that cannot fail its designed negative does not ship — all six
  designed negatives are pytest-asserted and **all six bite or are asserted as non-discriminating**.
- **Does NOT falsify:** a table-like inference read (permitted on both arms, §A13) · losing to a table
  on SEEN queries · a dividend ≈ 0 on the tier-i launder.
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ Every γ statement is read-budget-scoped.
  ⛔ **Wells are never named semantically** (`PREREG-TierII.md` §2.6) — obeyed in code, comments and
  every table below; wells/channels/features carry integer indices and nothing else.

## 0. Flag provenance (every number in this report)
| item | value |
|---|---|
| branch / base | `c2w11-physics-organizer` @ `d64868d`, base **`main @ 168a892`** (spoke A's merge), worktree `/Users/user/Desktop/CHLU-c2w11b` |
| venv | the **main** venv `/Users/user/Desktop/CHLU/.venv` reused via `PYTHONPATH` (w6 worktree-JAX hazard avoided); **JAX 0.9.0**, float32 |
| substrate (all from `FROZEN-INTERFACES-C2W11.json`, asserted at import by `load_frozen`) | `N_a=32, F=4, K=128, m=8, a=12, d_addr=4` · **`payload_radius = atom_payload_init_radius = 0.60`** · **`tol = 0.286960063782279`** · `chance_per_seed = [0.0, 0.001953125, 0.0]` · **`atom_width_frac_spacing = 0.37` (SELECTED; the harness refuses any other)** · placing write · feature-factored launches `k = 4`, `σ_q = 0.15`, shell `R = 2.0`, payload block pinned 0 · φ `phi_seed = 20260801`, 576 B |
| read | 400 (`γ=0.05`) + 800 (`γ=0.02`) Verlet steps, `dt = 0.05`; V3 grid `[50,100,200,400,800,1200]`, split `address = round(b/3)` |
| organizer | term (a) `λ_org=1.0` (`org_mode = reach`, AMENDMENT §A1) + `λ_band=0.1`, 300 Adam steps @ lr 0.02, batch 64, jig bound `0.75 × 0.896484375 = 0.6724`; (c) `λ_shape=1.0`, 120 steps @ 3e-3, capture margin 0.05; (d)/(e) `λ_read=λ_cal=1.0`, ψ 600 steps @ 3e-3, novelty head 400 steps @ 3e-3, `p_drop=0.25`, 3 train episodes. ⛔ `λ_share = 0` and **(f) kinetics NOT BUILT** |
| arms | `coef0` (all coefficients 0 — bit-identical control) · `org_only` (a) alone · `shape_only` (c) alone · `phys` (a)+(c)+(d)+(e) · `weak` (+`λ_weak=1.0`, the wave's single ablation) |
| seeds | **0,1,2,3,4** on every claim cell (5 seeds, `ddof=1`, `mean ± 2 SE`) |
| artifacts | `.claude/outputs/c2w11-physics-organizer/run1/{stage_launch_cap,stage_k5_organized,stage_guards,stage_m3,stage_m7_m8,stage_traversal,stage_v2}.json`, `run1/{coef0,org_only,phys}/stage_v3.json`, `run1/ablations/` |
| PREREG | `.claude/outputs/c2w11-physics-organizer/PREREG.md` (+ AMENDMENT §A1, both filed **before** the cells they govern); scorecard in §12 — **11 hit · 5 missed · 2 n/a** |
| wall | ≈ 45 min of compute total; a K5 cell is **11.3 s** |

---

# 1. ⛔ THE THREE MECHANICAL PRECONDITIONS — verified on disk, by me, before any code

| # | file | field | value | verdict |
|---|---|---|---|---|
| 1 | `.claude/outputs/c2w11/GATE-SEMANTICS-C2W11.json` | `kills_all_discriminating_passed` | **`true`** | ✅ |
| 2 | `.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json` | present, 29 940 B | data read, verdict **not** taken from its superseded `kills_all_passed` | ✅ |
| 3 | `.claude/outputs/c2w11-loss-package/LOSS-PACKAGE-DONE.json` | `loss_package_complete` | **`true`** (6/6 formalized, 4 deployment blockers) | ✅ |

**(1)'s per-leg table, quoted:** K0 **passed/discriminating** (0.9967 vs the same harness reproducing
0.0378 on C2W5's offsets) · K1 **passed/discriminating** (fails 3/3 at `w_frac = 1.5`) · K2
**passed/discriminating** (0.5 % at `m=1` → 100 % at `m=8`) · K3 **ABSTAINS** (`vacuous` 3/3) · K4
**ABSTAINS** (`vacuous` 3/3 — *"the store-only leak control is therefore uninformative"*) · K5
**ABSTAINS** (all cells vacuous **and the gate was circular**) · K6 reported precondition (0.0007) ·
K7-CAP **passed/discriminating** (SP-1 scores 1.0000 with `‖v̂−v‖∞ ≈ 1.5e-15`) · K8
**passed/discriminating** (1.0000 → 0.0625 across `K > N_a` → `K < N_a`).
**Discriminating legs = {K0, K1, K2, K7-CAP, K8}, all passed ⇒ the gate is TRUE.**

**(3)'s six-term table, quoted:** (a) `L_org` DIRECT ALGEBRAIC through the placing write — **the only
term live before wells exist** · (b) `L_share` direct on θ · (c) `L_shape` direct via Hellmann–Feynman
· (d) `L_read` implicit-at-settle to θ, **exactly 0** to φ at settle · (e) `L_cal` direct to `g`,
direct+implicit to θ · (f) `L_kin` trajectory/finite-budget only — ⛔ **NOT BUILT this wave**.

---

# 2. ⛔⛔ THE BLOCKING FIRST ACT — K5 RE-RUN ON THE ORGANIZED ARM

> **K5 ABSTAINS AGAIN, on a trained arm, on 5/5 seeds. Per the registered rule I STOPPED BEFORE THE
> VALUE LEGS.** This is the wave's **first** K5 reading on an arm carrying its training signal, and it
> is a **finding**, not a licence to proceed.

| statistic | value (5 seeds, mean ± 2 SE) | rule |
|---|---|---|
| best **read** score (frozen reader class **+** the added `deepsets_psi`) | **0.000391 ± 0.000781** (per seed 0/0/0/0/0.001953) | `≤ chance + δ` |
| best **per-item table** launder, through the same reader class | **0.000391 ± 0.000781** (0/0.001953/0/0/0) | `≤ chance + δ` |
| best margin (read − table) | **0.000391 ± 0.000781** | bar `> 0.10` |
| `chance` per seed | 0.0 / 0.001953 / 0.0 / 0.0 / 0.0 | `δ = 0.01` (5.12 grains) |
| **verdict** | **ABSTAIN, 5/5 seeds** — both halves of the conjunction hold | `K5_PASS = null` |

⛔ **Read the label correctly, per K5's own banked lesson:** every arm sits at ≈ 0, so this is a
**"not expressible at all"** reading, **not** a "table-expressible" reading. §3–§4 say *why*, and the
why is the point of this spoke.

**What was trained before this was scored** (so "untrained arm" cannot be offered as an explanation):
term (a) ran 300 steps and moved the jig to a median norm of **0.4790** of its **0.6724** bound (i.e.
the organizer used 71 % of the freedom it was given), term (c) ran 120 steps, ψ (1 224 params) trained
from loss 1.6845 → 0.7211, and the novelty head trained from 0.6931 → its floor. **Both instantiations
of term (a) were raced** (§4), and the weak-supervision ablation with them.

---

# 3. ⛔⛔ THE THREE FALSE-POSITIVE GUARDS — ONE SECTION, TOGETHER (never scattered)

With the G-DRIFT floor demoted to DIAGNOSTIC and K5 moved here, these three carry the **entire
false-positive load of the wave**. ⚠ K4's store-only form **ABSTAINED** (vacuous 3/3), so this is the
**FIRST informative run of that control in this wave** and family soundness rests on it.

### 3.1 K4-at-full-ψ — the `k4_full_psi_obligation`, discharged
| ψ capacity | params | blank store | query-only | permuted payloads | max leg | bar (`chance + 0.05`) | clears |
|---|---|---|---|---|---|---|---|
| `hidden = 8` | 424 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0500–0.0520 | ✅ |
| `hidden = 16` | 1 224 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | " | ✅ |
| **`hidden = 32`** | **3 976** | 0.0000 | 0.0000 | 0.0000 | **0.0000** | " | ✅ |

Max leg across **all** capacities × 5 seeds = **0.001953125** (one grain, seed 1). ⇒ **K4-at-full-ψ
PASSES 5/5. The family is NOT void.** ⭐ Per the registered rule P1 the ψ budget is **set by the
measured leak**: the largest clearing capacity is **`hidden = 32` on 5/5 seeds** — ⛔ ψ was *not* capped
by fiat (intervention Error 2), and the deployed claim cells nevertheless ran at `hidden = 16`
(a conservative choice, stated: the selection permits 32).
**Leg 4, the address-leak probe (DIAGNOSTIC):** full **0.1543** vs launder **0.2397**, dividend
**−0.0854 ± 0.0043** on the `phys` arm — the settle *loses* address precision on that arm, which §5
attributes to term (c), not to the launch head.

### 3.2 K7-CAP — the SP-1 parameter bound
`N_a·m = 256`. Measured reader params: `sum_linear` **104** · `well_table` **72** · `knn` **0** ·
`mlp` **92** · `zero_parameter_identity` **0** ⇒ **every member of the frozen class is under the
bound, 5/5 seeds.** ⛔ **ψ is NOT a member of that class and exceeds the bound by construction**
(424 / 1 224 / 3 976 params) — which is exactly why K4-at-full-ψ and K8 exist, and it is pinned by
`test_psi_exceeds_the_sp1_bound_and_that_is_why_k4_at_full_psi_exists`.

### 3.3 K8 — the structural cell
Spoke A's frozen facts stand (`K = 24 < N_a = 32`, design-matrix rank 24/24/24, rank-deficiency
asserted, SP-1 exact-set 0.0625 ± 0.0158). ⛔ **Scoring V1 on the K8 cell is a VALUE number and is
therefore a declared NOT-RUN under the K5 stop** — the harness is landed (`k8_structural_split` is in
`FROZEN`) and is one config away. **Sign agreement is consequently NOT ESTABLISHED, and no V1 clear
exists to be an artifact of.**

> ⭐ **Bearing on the Hub's Q11 (`P(ψ does the work | V1 clears)` = 0.15 with both guards).** The
> conditional never fired: **V1 did not clear** (§9 — the read is at the floor at every budget). What
> the guards *did* establish is the stronger, unconditional statement — **at this operating point ψ
> does NO work at all**: a blank store with ψ at 3 976 parameters scores **0.0000**.

---

# 4. ⭐⭐ THE SPOKE'S CENTRAL QUESTION, ANSWERED WITH A CEILING — can term (a) move launch-head precision?

The task file names the blocker: *"occupancy precision 0.2303, correct-and-distinct 0.92 of 4 … term
(a) is the named mechanism that could move that number."* **It cannot, and the reason is now a
bound rather than an attempt.**

| quantity | value (5 seeds, ± 2 SE) | what it is |
|---|---|---|
| identity precision (the shipped placement) | **0.2439 ± 0.0227** | reproduces spoke A's banked 0.2303 |
| ⭐ **best bijective re-assignment of wells to cue sites** | **0.3290 ± 0.0074** | ⛔ **THE CEILING on any placement organizer** — a linear assignment problem on the label-free co-occurrence matrix `P[j,c] = E[#picks of code c ∣ feature j present]`, solved exactly |
| row-max (non-bijective, unachievable) bound | 0.3818 ± 0.0075 | the loosest conceivable bound |
| ⇒ expected correct wells of `F = 4` **at the ceiling** | **1.32 of 4** | exact-set needs ~4 of 4 |

**Why the ceiling is a ceiling.** The organizer moves **wells**; it cannot move **φ**. A particle
launched at cue `R·e_{j_c}` settles into the nearest well, so any placement organizer is choosing an
**assignment of wells to cue sites** — and the optimum of that assignment is computed above. The
launch head's own pick distribution is the constraint: a matched filter deflating a sum of `F = 4`
unit codes drawn from `N_a = 32` in `d_addr = 4` cannot invert the sum.

### 4.1 Both term-(a) instantiations raced, plus the ablation — and every organizer LOSES to no organizer
| arm | organizer | settle occupancy precision (5 seeds ± 2 SE) |
|---|---|---|
| `coef0` | ⛔ none (bit-identical control) | **0.2329 ± 0.0087** |
| `org_only` | (a) **reach** (AMENDMENT §A1) | 0.2125 ± 0.0078 |
| `org_only` | (a) **nt_xent** (the theorist's spec) | **0.1985** |
| ⭐ `weak` | (a) reach **+ `λ_weak`** (the wave's single ablation) | **0.1199** |
| `phys` | (a)+(c)+(d)+(e) | 0.1543 ± 0.0033 |

⭐⭐ **THE A31.4 INVERSION REPRODUCES AT THE ORGANIZER LEVEL, and more sharply than at the encoder.**
Label information entering the **organizer** is **harmful** to address geometry: 0.1199 vs the
label-free 0.2125 and the no-organizer 0.2329. PREREG **B16** put `P(harmful beyond 2 SE) = 0.30` and
`P(tie) = 0.45`; measured **harmful**, ≈ 12 SE below the label-free arm. ⛔ This is a MECHANICS
statistic (occupancy), **not** a V1 swap number.

### 4.2 ⛔ DIAGNOSTIC — the oracle-addressing ceiling, and it is the wave's most load-bearing datum
Launch the same `k = 4` particles at the **needed** wells' address anchors (+`σ_q`), same store, same
400+800 settle, and read with the **zero-parameter raw sum** of the settled payload blocks:

| statistic | value |
|---|---|
| ⭐ **exact-set on UNSEEN queries @ `tol`** | **0.8621 ± 0.0036** |
| at 2× / 4× `tol` | 0.8621 / 0.9961 |
| occupancy precision | 0.9673 |
| per-particle `‖pay(q*) − v_occupied‖` | median **0.0002** (`‖v_j‖ = 0.60`) |

⇒ **the store, the settle and the payload composition WORK.** The read is not inert, not
table-collapsed and not capacity-starved; it composes four payloads to 0.86 exact-set the moment
addressing is correct. ⛔ **DIAGNOSTIC — it hands the read the addressing answer and is never a claim.**

---

# 5. M3 — per-feature G-ADDR (⛔ MECHANICS-ONLY, permanently barred from VALUE duty, §A34.8)

Bar = `max(4·chance, chance + 2 SE)` with `chance = 1/N_a = 0.03125` ⇒ **0.125**.

| arm | **M3** (5 seeds ± 2 SE) | `margin_in_SE` (seed 0) | `any_basin` ⛔ *reported, NOT the leg* | verdict |
|---|---|---|---|---|
| `coef0` | **0.9402 ± 0.0124** | **+82.5** | 0.9835 | **PASS** |
| `org_only` | 0.6912 ± 0.0545 | +31.9 | 0.8752 | PASS |
| `shape_only` | **0.0810 ± 0.0472** | **−3.5** | 0.0803 | **FAIL** |
| `phys` | **0.0697 ± 0.0445** | **−9.8** | 0.0749 | **FAIL** |

⭐⭐ **The decomposition this leg buys, and it is the finding that makes §4 interpretable.** On the
shipped store, **when a launch channel asserts feature `f`, the particle lands in `f`'s well and
inside its measured SC-6 capture radius 94.0 % of the time.** The addressing *physics* is nearly
perfect; the launch head's **assertion** is what is wrong (§4). ⛔ Note `any_basin` (0.9835) sits
above the leg (0.9402) — the banked pattern, and the reason it is reported and is not the leg.

**DIAGNOSTIC column, per ADDENDUM 13 ruling 2 (§A13):** measured site drift / spacing = `coef0`
**0.000428**, `org_only` 0.001553, `shape_only` 0.227, `phys` 0.229. The two organizer-free arms sit
**below** C2W8-close's `0.01 × spacing` floor ⇒ `fails_low_D2a_table_expressible`. ⛔ **Reported as a
DIAGNOSTIC column; it did not block or fail any leg scored here** — §A13 explicitly permits table-like
inference reads on both arms, and that permission is the reframe the organizer swap exists for. The
floor remains BLOCKING where it was built (C2W8's capture gate).

**Designed negatives, pytest-asserted** (`tests/test_c2w11_organizer.py`): (a) planted permutation ⇒
M3 fails its own bar while `any_basin` is **bit-identically unchanged** (it is target-blind);
(b) narrow-wells rig ⇒ **0.0000**; (c) ⚠⚠ **the D2a trap** — a planted near-zero-drift table-like
store (every particle collapsed onto one site) scores **≤ 0.10 with `any_basin` = 1.0000**, i.e. the
leg **cannot** be bought by settle-collapse.

---

# 6. M7 / M8 — the curvature-shape term and its end-of-training spectrum

`2α = 0.1000` (the floor, marked on every axis below). A **soft direction** is defined as
`λ_min ∈ (2α, 2α + 0.02]` **AND** `depth ≥ D_min = 0.15` **AND** measured capture `≥ σ_q` — i.e. *at
or near the floor but not the floor itself, at a site that is actually dug and still captures.*

| arm | median `λ_min` | median participation ratio | **soft-direction fraction** | median depth |
|---|---|---|---|---|
| `coef0` | **5.0352 ± 0.0226** | 3.889 ± 0.186 | **0.0000** | 0.600 |
| `org_only` | 4.7176 ± 0.2284 | 4.005 ± 0.262 | **0.0000** | 0.600 |
| `shape_only` | **0.7196 ± 0.0238** | **7.576 ± 0.615** | **0.0125** | 0.316 |
| `phys` | 0.6585 ± 0.0979 | 6.727 ± 0.690 | **0.0000** | 0.316 |

**M7 verdict, two-sided and measured (never assumed):** term (c) **does move the spectrum** — `λ_min`
falls **7.0×** (5.04 → 0.72) and the softest mode **delocalizes** (PR 3.89 → 7.58, M8 excess
**+3.69**) — but it **stops 7.2× above the `2α` floor** and yields **1.25 %** genuine soft directions,
while **halving the depth** (0.600 → 0.316) and **destroying addressability** (M3 0.9402 → 0.0810,
`any_basin` → 0.0803). ⇒ **a within-well soft direction does NOT survive superposition here.** This
is §A4.2's refutation reproduced on a **different instantiation** (a spectral-shape defender, not the
refuted tilt), which is worth more than a repeat of the same experiment.
**M8's designed negative holds:** the coefficient-zero arm shows **no excess** soft directions
(0.0000, and its `λ_min` sits 50× above the floor — nothing is masquerading as softness).
**Lifetime scoping that travels with this:** `τ_max = Γ/2α = 205.2 steps` at `γ = 0.05`; the shipped
read is **11.85 τ_max**, so even the softest permitted mode is erased to **3.49e-5** — ⛔ **term (c)
has no read-side consumer at the shipped budget, and M8 is its only consumer this wave** (exactly why
M8 was made mandatory).

---

# 7. ⛔ THE C2W9 TRAVERSAL TRIGGER — **FIRED**

Threshold registered **before** the run (PREREG §P6, identical to spoke A's coverage threshold).

| statistic | value |
|---|---|
| `mean_frac_needed_wells_unreachable` | **0.7715 ± 0.0081** (0.7725/0.7827/0.7773/0.7627/0.7622) |
| threshold | 0.20 ⇒ **fired by 3.86×** |
| queries with ≥ 1 unreachable needed well | **0.9996** |
| median distance to an unvisited needed well | **1.7483 = 2.75 × reach = 2.03 × well spacing** |
| per-slot breakdown | 0.8066 / 0.7852 / 0.7559 / 0.7422 — **flat**: every slot, not one bad slot |

⇒ **`.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md` §2 appended** (dated 2026-08-11, with mode,
fraction, per-feature breakdown, reach radii, seeds and provenance). ~~⛔ Spoke A's **coverage** half did
**not** fire; this is the **in-flight** mode.~~ ⛔⛔ **[STRUCK — HUB CORRECTION, 2026-08-12, charter
Add.15 §A43.4. BOTH HALVES FIRED: coverage 0.7546 ± 0.0116 (3.77×, and §1b records it PERSISTING
BIT-IDENTICALLY post-repair) and traversal 0.7715 ± 0.0081 (3.86×). The sentence generalised "the
PAYLOAD half of coverage is closed" — which the reach repair did achieve (`‖v_j‖/capture` 1.172 →
0.692) — into "coverage did not fire", which the same file's §1/§1b contradict. Full correction at
§2a of `TRAVERSAL-FAILURE-SIGNATURE.md`; registry N310. ⛔ No measurement in this report changes.]**
⭐ New scoping input for C2W9, and it survives the correction unchanged: the evidence is outside the
diamond **by ~2 well spacings**, so a wormhole/learned-`p₀` fix has a *quantified* distance to cross —
⚠ but both halves share ONE mechanism (the launch head asserts the wrong wells), so C2W9 stays
**DEFERRED behind the launch-head work** (Add.15 §A43.3).

---

# 8. V2 — the graded-novelty channel (⭐ the Head-flagged novelty piece)

⛔ **V2a's `> 0.60` floor is MECHANICS** (`PREREG-C2W11.md` §10's own leg-label index) and is what is
reported. ⛔ **V2's SWAP and V2b's ECE are VALUE and are declared NOT-RUN under the K5 stop.**

| statistic | value (5 seeds ± 2 SE) | bar |
|---|---|---|
| ⭐ **V2a per-feature novelty AUROC** | **0.7254 ± 0.0126** | floor **> 0.60** ⇒ **PASS**; ≤ 0.55 would be a null |
| graded response (mean logit by # novel channels 0/1/2) | **−1.740 → −1.415 → −1.158** | monotone — the *graded* in graded-novelty |
| collapse statistic (overlap-as-confidence) | **3.873 of `F = 4`** unique wells | banked shape: confident ⇒ collapse to `F` |
| **designed negative — blank store** | **0.5029 ± 0.0209** | ✅ bites |
| **designed negative — shuffled labels** | **0.4972 ± 0.0125** | ✅ bites |
| ⚠ **registered negative — permuted payloads** | **0.7903 ± 0.0397** | ⛔ **does NOT bite** — see R2 |

⭐ **The channel exists and it is store information, by construction rather than by measurement**
(`N-e3`): the dropout mask is drawn **independently of the query** and acts on the **WRITE**, so
`n_f ⟂ query`; a query-only novelty head is *provably* at the base rate. The head keys on **depth,
`λ_2nd` and the participation ratio** and **never on `λ_min`** — the (c)/(e) cross-term contract, and
it is enforced in code (`NOVELTY_FEATURES`), pinned by a test.
**Ledger:** novelty head **129 params / 516 B**; objective = **log-loss (strictly proper)**;
⛔ **AUROC is REPORTED and is never trained against** — recorded in the arm's own ledger dict.

---

# 9. V3 — the anytime curve at the FROZEN budget grid

⛔ **V3-PRIMARY (the swap difference) is VALUE and is NOT computed here.** The curve is emitted at the
frozen grid `[50,100,200,400,800,1200]` so `OD_V3(b)` is computable at the review from this artifact
and spoke C's. ⛔ **Quote the curve, not the endpoint.**

| budget `b` (total Verlet steps; particle-steps = 4 b) | 50 | 100 | 200 | 400 | 800 | 1200 | spread |
|---|---|---|---|---|---|---|---|
| `coef0`, shipped read | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | **0.0000** |
| `org_only`, shipped read | 0.0008 | 0.0004 | 0.0012 | 0.0004 | 0.0008 | 0.0012 | 0.0020 |
| `phys`, shipped read | 0.0008 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0008 |
| ⛔ **DIAGNOSTIC — `coef0`, ORACLE-addressed** | **0.0223** | **0.8219** | **0.8711** | 0.8621 | 0.8621 | 0.8621 | **0.8488** |
| ⛔ DIAGNOSTIC — `org_only`, oracle-addressed | 0.0258 | 0.7602 | 0.7863 | 0.7812 | 0.7812 | 0.7812 | 0.7605 |
| ⛔ DIAGNOSTIC — `phys`, oracle-addressed | 0.0012 | 0.0180 | 0.0277 | 0.0262 | 0.0266 | 0.0266 | 0.0266 |

**V3-MECHANICS verdict: the shipped curve is FLAT ⇒ a MECHANICS FAILURE, and it is DIAGNOSED, not
reported on top of** (the prereg's own instruction). ⭐⭐ **And the diagnosis overturns the naive
reading of N199** (R3): the *same store*, read by the *same physics* at the *same budgets*, produces a
**monotone, strongly non-flat curve** the moment addressing is correct — rising from 0.0223 at `b=50`
to 0.8711 at `b=200` and plateauing. **A flat curve therefore does not license "the memory carries
nothing"; it licenses "either it carries nothing OR it cannot be addressed", and those are separable
by exactly this control.** It also locates the navigation dial: **all of the resolvable structure is
resolved between `b = 50` and `b = 200`**; beyond 200 steps the curve is flat because it is *done*.

**V3-REPORTED (mandatory, never primary) — ⛔ a READ-COMPUTE RATIO (mult-adds), not wall-clock, not
training cost:** the physics read at the full budget costs **4.42e7 mult-adds** (4 particles × 1 200
steps × 384 atoms × 12 dims × 2) against a matched static read's ~2.05e4 ⇒ **≈ 2 160×**, in family
with the banked 3 360× at a tie. ⚠ Positioning carried: the anytime curve is a **SHAPE** claim, not a
uniqueness claim (§A3); "we have a curve and they don't" is §8.3-barred and stays in the "and also"
position.

---

# 10. Byte ledger (every arm, and the launders)

| item | bytes | note |
|---|---|---|
| store (`N_a·a` atoms) | **21 504** | identical on every arm |
| φ (frozen, byte-hashed `a2713a0f…`) | **576** | identical on every arm; a mismatch raises |
| launch head | **0** | reuses φ's codes (512 B already ledgered) |
| ⭐ **the organizer's jig** (`N_a × d × 4`) | **512** | the arm's **only** organizer parameter |
| ψ (deployed / grid) | **4 896** (`hidden=16`) / 1 696 · 4 896 · 15 904 | ⛔ exceeds the `N_a·m = 256` SP-1 bound by construction — guarded by K4-at-full-ψ + K8 |
| novelty head | **516** | 129 params |
| frozen reader class | 416 / 288 / 0 / 368 / **0** | all under the 256-param bound |
| byte ratio | `ratio_corrected` **3.833**, `A_atoms_per_item` 3.0, `D` 12 | ⛔ DIAGNOSTIC |

---

# 11. ⛔ DECLARED NOT-RUNs (never reported as nulls)

1. **V1** (leg i), **V2's swap and V2b's ECE** (leg ii), **V3-PRIMARY** (leg iii) — gated off by the
   registered **K5 stop**. The harness for all three is landed and each is one flag away
   (`stage_v2(..., value_legs=True)`, `stage_v3`, and V1 inside `stage_k5_organized`'s score pack).
2. **V1 at the K8 cell** — a VALUE number; NOT-RUN for the same reason. **Sign agreement is therefore
   NOT ESTABLISHED** and no V1 clear exists that could be a ψ-capacity artifact.
3. **Loss term (f) kinetics** — not built (charter's own instruction).
4. **Loss term (b)'s `λ_share`** — implemented, unit-tested (the structural refresh is monotone **by
   parameterisation**, asserted), but run at coefficient **0**: M4 is spoke A's leg and the theorist's
   R1 (`d/s` band vs depth heterogeneity) is an open reconciliation the Hub owns.
5. **Attention-ψ** — quarantined for trajectory input; DeepSets only.
6. **The store-side implicit channel of term (d)** — ψ was trained on its **direct** channel; the
   implicit-at-settle channel to θ is declared, not deployed, this wave.
7. **`OD`, `OD_min`, the swap verdict, any tier-ii or full-CLU verdict, any paper number.**

---

# 12. ⭐ PREREG SCORECARD (11 hit · 5 missed · 2 n/a)

| # | prediction | outcome | verdict |
|---|---|---|---|
| B1 | K5 **scores** (P = 0.60); best read 0.05 | **ABSTAINS**, best read 0.000391 | ❌ **MISS** (and B20's amendment had already moved this to P(abstain) = 0.85 — ✅) |
| B2 | K5 passes (P = 0.20) | abstained | n/a |
| B3 | K4-at-full-ψ clears (P = 0.75); blank leg 0.02 | **clears 5/5**, blank leg **0.0000** | ✅ HIT |
| B4 | ψ budget = 16 (P(32 survives) = 0.35) | **32 survives 5/5** | ❌ miss (point), ✅ direction |
| B5 | V1 best reader 0.06 | NOT-RUN (K5 stop) | n/a |
| B7 | V2a AUROC 0.82, P(> 0.60) = 0.80 | **0.7254 ± 0.0126**, floor **PASSED** | ✅ HIT (band [0.55,0.99]) |
| B9 | permuted-payload negative ≈ 0.50 | **0.7903** — non-discriminating | ❌ **MISS**, and it is R2 |
| B10 | curve non-flat, spread 0.04 (P = 0.55) | shipped **FLAT** (0.0000–0.0020) | ❌ miss — but see B10′ |
| B11 | read-compute ratio > 1000× | **≈ 2 160×** | ✅ HIT |
| B12 | M3 = 0.22, band [0.05, 0.55], P(pass) = 0.65 | **0.9402** on `coef0` | ❌ **MISS — under-predicted by 4×** |
| B13 | `any_basin` 0.95 | **0.9835** | ✅ HIT |
| B14 | soft-direction fraction 0.35; P(M7 positive) = 0.30 | **0.0125**; M7 **negative** | ✅ HIT (the P call) |
| B15 | PR excess +0.6, band [−0.5,+3.0]; P(coef-0 shows no excess) = 0.85 | **+3.69** (just outside), coef-0 excess **0.0000** | ✅ HIT (the P call), ❌ point |
| B16 | weak arm harmful (P = 0.30) / tie (0.45) | **harmful, ≈ 12 SE** | ✅ HIT (the minority call) |
| B17 | traversal fires (P = 0.40), 0.12 | **FIRED at 0.7715** | ✅ HIT (direction), ❌ point |
| B18 | accidental φ leak structurally absent (P = 0.85) | asserted bitwise 0 with the placement cut | ✅ HIT |
| **B19** | reach organizer achieves 0.30; P(> ceiling) = 0.05 | **0.2125** — *below* the un-organized arm | ❌ miss (point), ✅ the P call |
| ⭐ **B20** | **K5 abstains again (P = 0.85)**, from the assignment ceiling | **ABSTAINED 5/5** | ✅ **HIT — the amendment's own falsifiable call** |

⭐ **The scorecard's shape, stated because the program banks these:** my two *directional* calls that
carried the wave (B20's abstain and B16's inversion) both landed, and my two worst misses are both
**under-estimates of the physics** (M3 0.22 → 0.94; the oracle read, which I did not predict at all,
at 0.86). **The systematic bias in this spoke's priors is the mirror image of C2W5's**: I
under-predicted the *store's* competence and over-predicted the *organizer's*.

---

## How I verified (commands + observed output)
* `uv`-free but venv-exact: `PYTHONPATH=/Users/user/Desktop/CHLU-c2w11b /Users/user/Desktop/CHLU/.venv/bin/python …`
  (main venv reused per protocol §4; **JAX 0.9.0**).
* `python -m pytest tests/test_c2w11_organizer.py -q` → **23 passed** (two initial failures were real
  and are reported: the M3 permutation bar was absolute rather than chance-relative, and the raw depth
  feature's **sign** was inverted in the V2 negatives — both fixed, and the sign error is why the
  permuted-payload leg is now asserted *as measured*).
* `python -m ruff check` on all five touched files → **All checks passed** (spoke A's files are also
  clean; I matched that standard).
* **Full suite on my branch:** `python -m pytest -q --no-cov -x -p no:randomly` in the worktree
  `/Users/user/Desktop/CHLU-c2w11b` at `d64868d` → **1635 passed, 0 failed, 36 warnings (37 min)**.
  **Count arithmetic, checkout named:** `--collect-only` in that same checkout gives **1635**, and
  **1612** with `--ignore=tests/test_c2w11_organizer.py` ⇒ my file contributes exactly **+23** and the
  inherited 1612 are unchanged. ⚠ `-p no:randomly` was used so the count is reproducible.
* Stage runs: `run_c2w11_organizer(seeds=(0,1,2,3,4), stages=(…), arms=(…))`, logs in
  `.claude/scratch/c2w11-physics-organizer/run1_{a,b,c,d}.log`, JSON in `run1/`.

## Git footprint
* **Branch `c2w11-physics-organizer`** off **`main @ 168a892`**, in worktree
  `/Users/user/Desktop/CHLU-c2w11b`. ⚠ **Deviation, declared:** the task named wt1
  (`../CHLU-c2w11a`), but that worktree was **still checked out on spoke A's branch**
  (`c2w11-substrate-and-kills @ 4324002`) when I started, so per protocol §3.2 I took my own
  worktree rather than disturb another agent's checkout. `../CHLU-c2w11a` is untouched.
* Commits (3): `5144384` land the organizer (loss package, ψ, novelty read, stages) · `6c53626`
  pytest the designed negatives · `d64868d` lint + loop-variable binding.
* Files touched — **all within my declared ownership**: `chlu/training/losses.py` (additive),
  `chlu/core/psi_readout.py` (additive), **new** `chlu/core/novelty_read.py`, **new**
  `chlu/experiments/exp_c2w11_organizer.py`, **new** `tests/test_c2w11_organizer.py`.
  ⛔ `chlu/cli/experiment_cmd.py`, `chlu/config.py`, spoke A's and spoke C's files, the CSF3 pilot's
  and C2W8-close's territory: **not touched** (`git show --stat` confirms).
* Rebase onto local `main`: no-op (base has not moved). ⛔ Nothing pushed.

## Open questions / follow-ups / risks
1. ⛔⛔ **The wave's question is unaskable at `d_addr = 4`** (R1). The organizer swap compares two ways
   of placing wells that a launch head reaches 24 % of the time; both arms are floored by the same
   cap. **C2W12's dial is the launch head / `d_addr`, and the cheap next cell is the assignment
   ceiling as a function of `d_addr`** — it is a `scipy` call on a co-occurrence matrix, needs no
   store, and would have cost minutes in spoke A.
2. **`k = F` may be the wrong protocol.** The head launches exactly `F` particles and gets ~1 right.
   `k > F` (launch the top-`k` deflation picks and let ψ's capture weighting discard the scattered
   ones) is a **launch-protocol** change and therefore frozen this wave — but the collapse statistic
   (3.87 of 4 distinct) says the particles are distinct and merely *wrong*, which is the regime where
   over-launching helps.
3. **Term (c) should not ship as-is** (R4): it buys spectral softness with depth and capture. If
   C2W12 wants it, the capture guard must be the **measured** SC-6 radius inside the loss, not the
   differentiable inward-force surrogate this spoke declared and used.
4. **ψ was never given a chance to leak** — every leak leg is exactly 0.0000 because the read carries
   no information to leak *through*. ⚠ The K4-at-full-ψ pass is therefore **sound but weak evidence**:
   it discharges the obligation at this operating point, and it should be **re-run at any future
   operating point where the read is off the floor** before a V1 clear is believed.

## Proposed handover updates (for the Hub)
1. **§7 Known Issues — add:** *the tier-ii read is capped by the launch head, and the cap is bounded:*
   identity precision **0.2439 ± 0.0227**, best-bijective-re-placement ceiling **0.3290 ± 0.0074**,
   needed 4 of 4 ⇒ **the organizer swap cannot be decided at `d_addr = 4`.**
2. **§7 — add the oracle-addressing datum as the counterweight:** the same store scores **0.8621 ±
   0.0036** exact-set on unseen queries under oracle addressing, so *"the store is inert"* is refuted
   for this substrate and must not be carried forward.
3. **§7 — amend N199's flat-curve rule:** flatness is *addressing-or-emptiness* and the two are
   separable by the oracle control (R3).
4. **§7 — record K5's second abstain**, this time on a **trained** arm with two organizer
   instantiations raced, and record that **K4-at-full-ψ passed** (the wave's first informative K4)
   while **K8's V1 sign check is NOT-RUN**.
5. **§3 config defaults:** no `chlu/config.py` change was made or is needed. New knobs live in
   `OrganizerConfig` (in `chlu/experiments/exp_c2w11_organizer.py`, next to the code that reads them,
   per the `CatTestConfig` precedent) and in `C2W11LossCoeffs` (all coefficients default **0.0** ⇒ the
   shipped objective, structurally).
6. **Charter §A34.4/§A34.9(e):** the registered permuted-payload negative needs re-specification for a
   depth-keyed channel (R2); the shipped substitutes are blank-store and label-shuffle.
7. **The A31.4 inversion now has an organizer-level datum** (§4.1): label information in the organizer
   is **harmful** to address geometry (0.1199 vs 0.2125 label-free vs 0.2329 no-organizer).
