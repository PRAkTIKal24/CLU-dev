# c2w11-substrate-and-kills — experiment-engineer report (C2W11 spoke A)

**Task + acceptance criterion (one line):** build the repaired substrate (placing write · RE-SELECTED
co-scaled widths · feature-factored launches) on the factored store, run **every kill-condition
K0–K8 FIRST** with M1/M2/M4/M5/M6 and the coverage trigger, and freeze the interfaces the other two
spokes wait on — *a clean kill is a full acceptance*.
**Status: done.** All nine kill-conditions stated with bar and measured value, multi-seed; designed
negatives pytest-asserted for M1/M2/M4/M5; the frozen interfaces published at the exact required
path; the coverage trigger **fired** and its signature file is written.
⛔ **Verdict: `kills_all_passed = FALSE`. K5 fails, and it fails VACUOUSLY — read `0.0007`, table
`0.0007`, chance `0.0007`.** The wave's structural cap is **unmoved** despite three *measured*
substrate changes. Per `PREREG-C2W11.md`'s registered run-order note this is **the fifth convergent
datum on write-side organization, this time with the substrate repairs CONTROLLED FOR** — a result,
not a wasted wave.

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. ⭐⭐ **The placing write's site drift is `2.0e-6 × spacing`, i.e. `fails_low_D2a_table_expressible`
   against C2W8-close's new two-sided G-DRIFT floor (`0.01 × measured spacing`).** The C2W11 charter
   substrate **is** a placing write (§A34.10), so the ratified instrument and the ratified substrate
   are in tension. ⚠ I **report, and do not adjudicate** — `well_lifecycle.py` is C2W8-close's
   territory and read-only this wave. **Needs an owner and a ruling.**
2. ⭐ **`atom_width_frac_spacing = 1.5` measures `d/s = 0.57` here — below the `2.01` merger floor,
   K1 FAILS 3/3 seeds.** Repair (b)'s trap is confirmed by measurement: inheriting the banked value
   would have merged every well. Anywhere `1.5` is quoted as a *selected* width needs a curator note.
3. ⭐ **C2W5 deviation D2 (`a = 32`) is superseded at the placing write:** K1 passes at `a = 4`, `12`
   **and** `32`, so the byte ratio falls **9.67× → 1.50×**. ⚠ With the caveat in §4.2: depth is now a
   *placed* parameter, so this is a design choice made explicit, not free capacity.
4. **`PREREG-TierII.md` §3.4's K1 prediction ("passes at `a ≥ 12`, fails at `a ≤ 4`") is REFUTED in
   the other direction** — it now passes at `a = 4`. C2W5 refuted it downward; the repaired write
   refutes it upward.
5. ⭐⭐ **K6 is `0.0007` at the registered `d = 4` but `0.5273` at `d = 16` and `0.9844` at `d = 32`**
   (declared out-of-protocol d-diagnostic). At `d ≥ 16` the **launch head itself answers the family**.
   Any future `d`-sweep must report K6 **per `d`** or its fitted-reader scores are uninterpretable.
6. **The Hub's Q3 is REFUTED**: `P(M6 dividend ≥ 0) = 0.50` predicted; measured **−0.1567 ± 0.0052**,
   3/3 seeds, *more* negative than C2W5's −0.1094. My own registered 0.35 was also on the wrong side.
7. **My own PREREG's M6 launch-precision prediction (0.62) is REFUTED at 0.2308** — mechanism named
   in §4.3. Registered predictions that fail are findings; this one is.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** **TIER ii substrate + kill-conditions.** ⛔ **Every leg below is MECHANICS.** No
  VALUE leg, no `OD`/`OD_min`, no organizer swap, no paper number, no tier-ii verdict, no full-CLU
  verdict. **I do not score the physics arm's value; I decide whether it is worth scoring.**
- **Laundering control:** ⛔ **every launder margin here is DIAGNOSTIC and can never fail a leg**
  (§A33.1). The launch-only launder, the settle-deleted launder and the byte ledger are reported
  beside every reading, all labelled DIAGNOSTIC.
- **Falsifies:** K0 below its bar · K1 unsatisfiable at any affordable `a` · K2/K3/K4 failing — any
  stops the wave here. A leg that cannot fail its designed negative does not ship.
- **Does NOT falsify:** losing to a table on SEEN queries (Thm O1/D2a); a dividend ≈ 0 on the
  inherited tier-i launder (CM-27(b) by design at tier ii).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94 discipline on every reading.
- ⛔ **Wells are never named semantically**, verbatim from `PREREG-TierII.md` §2.6:
  > *"Wells `{j}` are co-activated by queries whose ground-truth factor set contains factor `f`, with
  > co-activation correlation `ρ = …` (95 % CI …), measured against a permutation null. No well is
  > identified with any factor; the claim is a correlation between co-activation/wormhole/
  > shell-position statistics and task structure."*

---

## §1 — PRECONDITION RE-VERIFIED BY ME, ON DISK, BEFORE ANY CELL

`.claude/outputs/c2w8-close/GATE-HARDENING-DONE.json` → **`gate_hardening_done = true`**, base commit
`9e0bb25`, branch merged at `main @ 2e1cdb2`. **All twelve items `true`**, read from the artifact's
*content*:

| item | done | item | done |
|---|---|---|---|
| i two-sided drift leg/floor | ✅ | vi.2 `theta_att`/`P` needs `n_non_capturing` | ✅ |
| ii A1 margin-in-SE beside the boolean | ✅ | vi.3 errata pass-2 numbering collision | ✅ |
| iii full-state co-scaling scale guard asserts the VERDICT | ✅ | vi.4 stale x64 comment | ✅ |
| iv `covered`/`n_never_read` split launch vs settle | ✅ | vi.5 census refuses a non-selected width | ✅ |
| v `d_safe` population fix | ✅ | vi.6 cue-difficulty arm dependence | ✅ |
| vi.1 own/foreign site-depth kernel | ✅ | acceptance 3: A3 → DIAGNOSTIC, out of the pass condition | ✅ |

**My read agrees with the Hub's account.** Proceeded.

---

## ⭐⭐ §2 — FIRST SCREEN: K0 AND M6, BEFORE ANYTHING ELSE

*(The two cheapest kill signals in the wave, reported first as the run order binds.)*

### 2.1 ⭐ K0 — launch expressivity, **NO STORE**, 3 seeds × 512 unseen rule-4-valid queries

| arm | `≥ F` distinct fraction (**bar ≥ 0.80**) | mean distinct (**bar ≥ 3.5**) | ⭐ **mean CORRECT-and-distinct** | occupancy precision | exact-set occupancy | K0 |
|---|---|---|---|---|---|---|
| ⭐ **feature-factored (`k = F = 4`)** | **0.9967 ± 0.0026** | **3.9967 ± 0.0026** | **0.9225 ± 0.0339** | 0.2308 ± 0.0085 | 0.0007 ± 0.0013 | ✅ **PASS 3/3** |
| designed offsets (C2W5, reproduction) | 0.0378 ± 0.0069 | 2.1680 ± 0.0475 | 0.8600 ± 0.0203 | 0.4211 ± 0.0036 | 0.0000 | ⛔ FAIL |
| ⛔ collapsed (M1's designed negative) | **0.0000** | **1.0573 ± 0.0138** | 0.4538 | 0.4326 | 0.0000 | ⛔ **FAILS, as designed** |

**Reproduction check:** the designed-offset arm reproduces the banked `orgdiv-null-arms` §3 numbers
(**0.050 / 2.202** over 5 seeds) at **0.0378 / 2.168** over 3 — same instrument, same story. My
instrument is not inventing the move.

**Full distribution, not just the mean** (seed 0, 512 queries), as the task requires:

| distinct wells reached | 0 | 1 | 2 | 3 | **4** |
|---|---|---|---|---|---|
| feature-factored | 0 | 0 | 0 | 3 | **509** |
| designed offsets | 0 | 107 | 254 | 131 | 20 |
| ⭐ **CORRECT-and-distinct, feature-factored** | 146 | 253 | 105 | 8 | **0** |
| ⭐ **CORRECT-and-distinct, designed offsets** | 173 | 251 | 84 | 4 | **0** |

**Per feature channel** (feature-factored, precision by deflation order — ⛔ channel indices carry no
semantic content): **0.4277 · 0.2188 · 0.1504 · 0.1543**. Chance is `F/N_a = 0.125`.

> ## ⭐⭐ THE FINDING, AND IT IS THE WAVE'S CENTRAL ONE
> **K0 clears by a factor of 26× — and the axis that actually matters did not move at all.**
> The feature-factored launch reaches `≥ F` distinct wells on **0.9967** of queries against C2W5's
> **0.0378**. But a read that must express `y = Σ_{j∈A} v_j` needs neither "distinct" nor "precise"
> on its own — it needs wells that are **distinct AND correct**, and that statistic goes
> **0.860 → 0.922 of 4**. The two histograms above are nearly the same distribution, and **neither
> arm ever reaches 4 correct wells, on any of 512 × 3 queries.**
> **Mechanism, measured:** only the **first** deflation channel carries signal (precision 0.428 —
> as good as the entire C2W5 launch); channels 2–4 sit at **0.15 against a chance of 0.125**. At
> `d = 4`, 32 codes in `R^4` are so collinear that the second-best matched-filter alignment is
> already noise. **The launch head converted a concentration cap into a precision cap; it did not
> add information.**

### 2.2 ⭐ M6 — ⛔ **DIAGNOSTIC, cannot fail a gate** (3 seeds, 512 unseen, written store)

| statistic | **launch (raw geometry)** | **after the settle** | **dividend** |
|---|---|---|---|
| occupancy precision | **0.2303 ± 0.0090** | **0.0736 ± 0.0040** | ⛔ **−0.1567 ± 0.0052** |
| distinct wells occupied | 3.998 | 3.807 | −0.191 |
| exact-set occupancy | 0.0007 | 0.0000 | — |
| M5 wells-visited `W/N_a` | 1.000 | 0.885 | −0.115 |
| *banked C2W5, for reference* | *0.4061* | *0.2967* | *−0.1094* |

⛔ **Scored against the BLANK STORE / raw launch geometry, never against `F/N_a`** (C2W5
reconciliation 4). 2 SEs do not overlap on any seed.

> **The settle destroys 68 % of the correct-well information the frozen launch geometry already
> contained** (correct-and-distinct `0.969 → 0.297` on seed 0), and the dividend is **more** negative
> than C2W5's on a substrate whose capture was repaired. ⭐ **Both the Hub's Q3 (0.50) and my own
> registered 0.35 are refuted, in the same direction.** §4.3 names the mechanism, and it is not
> capture: it is **reach**.

---

## §3 — THE K-TABLE (every kill-condition, bar and measured value, multi-seed)

**Seeds 0/1/2** (instrument cells, 3 seeds as registered) · 512 unseen rule-4-valid queries per seed.

| id | bar | **measured** | verdict |
|---|---|---|---|
| ⭐ **K0** | `≥0.80` distinct-`F`; mean distinct `≥ F−0.5` | **0.9967 ± 0.0026** · **3.9967 ± 0.0026** | ✅ **PASS** |
| **K1** | loss ≤ 0.05 · `λ_min>0` ≥ 90 % · capture ≥ σ_q ≥ 90 % | `a=4`: 0.0005–0.0023 / 1.00 / 1.00 · **`a=12`: 0.0007–0.0027 / 1.00 / 1.00** · `a=32`: 0.0007–0.0027 / 1.00 / 1.00 | ✅ **PASS at 4, 12 AND 32** |
| **K2** | 100 % of held-out queries, **both halves** | set half **100 %** (max overlap 2 = `F−2`) · payload half **100 % at `m=8`**, **0.52 % at `m=1`** | ✅ **PASS at `m = 8`** |
| **K3** | ≤ 0.60 of metric range | nearest-item table **0.0000** · strongest +0 B substitute **0.0013** | ✅ PASS — ⚠ **VACUOUS** |
| **K4** | all legs ≤ chance + 0.05 (**store-only form**) | blank **≤0.00195** · query-only **≤0.00195** · permuted **≤0.00195** · address-leak dividend **−0.1623** | ✅ PASS — ⚠ **VACUOUS** |
| ⛔ **K5** | read beats the per-item table by **> 0.10** on ≥ 1 reader | **best margin 0.0000 on all 3 seeds** (read 0.0007, table 0.0007) | ⛔⛔ **FAIL — and VACUOUSLY** |
| ⭐ **K6** | reported precondition, never a bar | **0.0007 ± 0.0013** (`0/512 · 0/512 · 1/512`) | ✅ reported |
| **K7-CAP** | every reader `< N_a·m = 256` | `sum_linear` **104** · `well_table` **72** · `knn` **0** · `mlp` **92** · **zero-parameter 0** | ✅ **PASS** |
| **K7-CAP** SP-1 probe (⛔ declared out-of-class diagnostic) | — | exact-set **1.0000** 3/3 · `‖v̂−v‖∞` = **1.50e-15 / 3.00e-15 / 4.25e-15** | reproduces banked **4.25e-15** |
| ⭐ **K8** | rule-4 split exists at `K<N_a` **and** SP-1 is rank-deficient | rank **24 < 32** 3/3 · rule 4 ✅ both halves · SP-1 unseen **0.0469/0.0684/0.0723**, `‖v̂−v‖∞ = **0.69–0.73** vs the headline cell's **4e-15`** | ✅ **PASS** |
| | ⭐ **`kills_all_passed` (mechanical AND over K0–K7-CAP)** | | ⛔ **FALSE** |

### 3.1 ⚠⚠ THE VACUITY, STATED WHERE IT CANNOT BE MISSED
**Computed mechanically** (`top_physics_score ≤ chance + 0.01`), `vacuous = True` on **3/3 seeds**.

| seed | chance | best physics reader | K5 table | K3 | best K4 leg |
|---|---|---|---|---|---|
| 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00195 |
| 1 | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |
| 2 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00195 |

⛔ **A reader of this report must not quote "K3 ✅ K4 ✅" as evidence the family is sound.** They are
upper-bound controls and they pass because **every number in the cell is ≈ 0**. And ⛔ **K5's failure
is a "not expressible at all" finding, NOT a "table-expressible" finding** — the `K`-row table scores
exactly what the read scores.

### 3.2 The MECHANICS legs, with their designed negatives

| leg | bar | measured | designed negative (pytest-asserted) | fires? |
|---|---|---|---|---|
| **M1** (=K0) | ≥ 0.80 | **0.9967** ✅ | collapsed-to-one-channel launch set | ✅ **0.0000 / 1.057 distinct** |
| **M2** (=K1) | §4 bars | ✅ **PASS** | harness REFUSES a non-selected width | ✅ `UnselectedAtomWidth` raised, both forms |
| **M4** sharing/refresh | ≥ 90 % non-decreasing depth on rewrite | **1.000 3/3** ✅ | a private well per item cannot deepen anything | ✅ **0.000** |
| **M5** anti-collapse | declared band `W/N_a ≥ 0.75`, TWO-SIDED | launch **1.000** → settle **0.885** ✅ **OK, not collapsed** | one-well launch ⇒ COLLAPSED; uniform ⇒ OK | ✅ both |
| ⭐ **M6** | ⛔ DIAGNOSTIC | −0.1567 ± 0.0052 | — (cannot fail a gate) | — |
| **M3** per-feature G-ADDR | — | ⛔ **DECLARED NOT-RUN** — `well_lifecycle.py`/`test_gate_addr.py` are C2W8-close's territory, READ-ONLY all wave | — | — |
| **M7 / M8** | — | ⛔ **DECLARED NOT-RUN** — loss term (c) is the loss-package spoke's | — | — |

⭐ **M5 is a genuine improvement over the banked cell:** C2W5 reported **15 / 10 / 14 of 32** wells
ever occupied (`W/N_a = 0.47 / 0.31 / 0.44`) and was reported **COLLAPSED**. Here the settled read
visits **0.88 of the vocabulary** and is **not** collapsed. ⚠ The `S_eff ∈ [8,16]` band is retired
(§A26.4) and is not used; "COLLAPSED" is reserved for concentration.

---

## §4 — THE THREE SUBSTRATE REPAIRS, EACH MEASURED

### 4.1 ⭐⭐ The width, RE-SELECTED against the STORE population (repair (b))

**Protocol registered in my `PREREG.md` §1 before the sweep ran; no free choices.** Selection seeds
**100/101/102**, disjoint from the claim seeds. `store_population_spacing` = median NN over the
`N_a = 32` anchors = **0.8590** (the store population **is** the anchors in a factored store — stated
rather than worked around; `d_safe_population = "sizing"` has no analogue here).

| `w_frac` | atom width | **measured `s`** (α-subtracted) | fit `R²` | **measured `d/s`** | in band [2.5,2.9] | K1 3/3 |
|---|---|---|---|---|---|---|
| 0.20 | 0.1718 | 0.1730 | 1.0000 | 4.97 | ✗ (≥4.0 ⇒ provably zero dividend) | ✅ |
| 0.25 | 0.2148 | 0.2162 | 1.0000 | 3.97 | ✗ | ✅ |
| 0.30 | 0.2577 | 0.2595 | 1.0000 | 3.31 | ✗ | ✅ |
| ⭐ **0.37** | **0.3178** | **0.3202** | **1.0000** | **2.68** | ✅ | ✅ |
| 0.50 | 0.4295 | 0.4341 | 1.0000 | 1.98 | ✗ (≤2.01 ⇒ merger) | ✅ |
| 0.75 | 0.6443 | 0.6834 | 0.9998 | 1.26 | ✗ | ⛔ |
| 1.00 | 0.8590 | 0.9614 | 0.9992 | 0.89 | ✗ | ⛔ |
| ⛔ **1.50 (the banked value, NOT inherited)** | 1.2885 | 1.5086 | 0.9998 | ⛔ **0.57** | ✗ | ⛔ **FAIL** |

> ⭐ **SELECTED: `atom_width_frac_spacing = 0.37`** — the **only** qualified point (in band *and*
> K1-passing). ⛔ **And the trap is confirmed by measurement: the banked `1.5` gives `d/s = 0.57`, a
> factor 3.5 below the `2.01` merger floor, and K1 fails there 3/3.** Inheriting it would have merged
> every well and every downstream number would have been scored against a different instrument.
> **Registered prediction W1 = 0.37 [0.30, 0.45] — HIT exactly. W2 = 0.67 [0.45, 0.95] — measured
> 0.57, HIT in band.**

The harness now **REFUSES** any other width (`UnselectedAtomWidth`), and the refusal is
pytest-asserted in both forms (no declaration at all; declaration ≠ request).

### 4.2 ⭐ The placing write (repair from §A29.4(ii))

| statistic (`a = 12`, selected width, 3 seeds) | **placing write** | *banked gradient write* |
|---|---|---|
| endpoint write loss (bar ≤ 0.05) | **0.0007 / 0.0010 / 0.0027** | *0.039 / 0.036 / 0.071 at `a=12`* |
| `‖∇V‖` at the written targets | 1.2e-4 | — |
| `λ_min > 0` | 1.00 | 1.00 |
| SC-6 capture ≥ σ_q | **1.00** | *0.88 / 0.88 / 0.69* ⛔ |
| median capture radius | **0.8536** (= 5.7 σ_q) | — |
| K1 passes at | **`a` = 4, 12 AND 32** | *only `a = 32`* |
| byte ratio at the smallest passing `a` | **1.50×** (`a=4`, store 7 168 B) | *9.67× (`a=32`, 57 344 B)* |
| median site drift / spacing | **2.0e-6** | *0.0010–0.0065 (arm A census)* |

⭐ **The write is no longer the binding constraint.** ⚠ **And the mechanism is stated rather than
sold:** under a placing write the well **depth is a placed parameter** (`place_depth = 0.30`), so
"K1 passes at `a = 4`" says the atoms are no longer required to *dig* the depth — it is not free
capacity, it is a design choice made explicit and ledgered. The **stationarity shift** is load-bearing
and pytest-asserted: without it `grad_norm_at_targets` and the endpoint loss are both strictly worse,
because an atom cloud placed exactly at the target is *not* stationary there (`2α‖q‖` alone puts
`l_grad` at the 0.05 bar).

⛔ **Reconciliation 1, restated because it matters:** drift `2.0e-6 × spacing` is **below** the new
two-sided G-DRIFT floor of `0.01 × spacing` ⇒ `fails_low_D2a_table_expressible`. **Reported, not
adjudicated.**

### 4.3 ⭐⭐ Feature-factored launches — and the reach wall they exposed

The head decomposes φ's set-code against **φ's own frozen codes** by greedy matched-filter deflation
(§A34.1: `k` structured by the encoder's decomposition, not free). It reads `φ(x)` and `{e_j}` and
**nothing else** — never `A(x)`, never a payload, never the store — and costs **0 parameters** (it
*replaces* C2W5's designed offsets, which were parameters, so the swap is byte-negative).

> ### ⛔⛔ THE READ LAUNCHES OUTSIDE EVERY BASIN, BY ARITHMETIC
> | quantity | value |
> |---|---|
> | distance from launch to a well's **full** target (`= ‖v_j‖`; the launch pins the payload block to 0 — the anti-decoration guard) | **1.0000** |
> | **measured** SC-6 capture radius, median | **0.8536** |
> | ⛔ **ratio** | **1.172 > 1** ⇒ `needed_well_inside_basin = FALSE` |
> | full-space coverage: needed wells uncovered by any launch diamond | ⛔ **1.0000 on 3/3 seeds** |
>
> **This is the mechanism for M6's negative dividend and for K5's vacuous failure, and it is not a
> capture failure.** Capture is repaired: 100 % of wells have a basin of radius 5.7 σ_q around their
> *full* target. The read simply starts **outside** it, because the address half of the launch is
> solved and the payload half cannot be (solving it would hand the read the answer). ⛔ **The binding
> constraint is REACH.**

⚠ **This cannot be fixed by widening the atoms:** every `w_frac` large enough to close 1.0 of payload
distance drives measured `d/s` below the 2.01 merger floor (§4.1).

---

## §5 — THE C2W9 COVERAGE TRIGGER: ⛔ **FIRED**

Threshold **0.20**, registered in my `PREREG.md` §2 before the run. Measured mean fraction of needed
wells uncovered: **0.7546 ± 0.0116** (0.7461 / 0.7520 / 0.7656). **Zero queries fully covered on any
seed.** ⇒ **`.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md` §1 is written**, with mode,
fraction, per-channel breakdown, reach radii and seeds. ⛔ The **traversal** half is spoke B's and is
a declared NOT-RUN here — the absence of a §2 in that file is not a measurement that traversal is fine.

---

## §6 — ⭐⭐ THE DELIVERABLE

> **`.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json`** — written at exactly that path.

All twelve required blocks present and **every ledger number emitted from the code that computes it**:
`kills_all_passed` (mechanically `false`, each kill with its own boolean/measured/bar/reason) ·
the frozen family `(N_a=32, F=4, K=128, m=8, a=12, d=4)` with `sep = 0.8638`, **measured `s = 0.3199`,
`d/s = 2.684`**, `tol = 0.4783`, chance **per seed** `[0.0, 0.00195, 0.0]`, depth ratio 3.0,
`γ_address = 0.05` / `γ_read = 0.02`, read budget 400 + 800 @ `dt = 0.05` · the frozen launch protocol
(rule, `k = 4`, launch keys, the bit-identical-launch assertion) · the frozen φ (`phi_bytes = 576`,
hash `a2713a0fb155e09f965cb6808720dbb1`) · the frozen reader class **including the mandatory
zero-parameter member**, measured params and the rule-4 seen-validation split rule · the two-sided
byte-ledger template (store **21 504 B** at `a=12`, φ 576 B, head **0 B**, `ratio_corrected = 3.83`) ·
`k4_full_psi_obligation` · `coverage_trigger_fired: true` · `k8_structural_split` ·
`selected_atom_width` **with the store-population spacing it was selected against** ·
⭐ **`v3_budget_grid`** (6 points: total Verlet steps `[50, 100, 200, 400, 800, 1200]`; particle-steps
`[200, 400, 800, 1600, 3200, 4800]` at `k = 4`; the address/read split rule frozen) ·
`store_population_spacing` per seed with **`σ_q / spacing = 0.1747`**.

### 6.1 ⭐ The file was CHECKED for self-sufficiency, not assumed to be sufficient
*"Anything the other spokes need and cannot re-derive from this file is a defect in this file."* I
wrote a **spoke-B/C simulation** that reads **only the JSON** and rebuilds the protocol
(`.claude/scratch/c2w11-substrate-and-kills/verify_frozen.py`):

```
phi:  bytes + byte-hash MATCH -> a2713a0fb155e09f965cb6808720dbb1
launches: BIT-IDENTICAL across an independent rebuild -> (512, 4, 12)
tol from JSON 0.478267 vs rebuilt 0.478267
v3 grid: [50,100,200,400,800,1200]  particle-steps: [200,400,800,1600,3200,4800]
ALL RE-DERIVATION CHECKS PASSED
```

⚠ **I did not repeat C2W5's `FROZEN-interfaces.md` failure.** Its ledger row (`21 504 B`) was quoted
against an `a = 32` cell; here 21 504 B is emitted for the `a = 12` cell that actually ran, and
57 344 B for `a = 32`. Its reader counts (16 / 88) contradicted the shipped code; mine are read out of
`reader_bytes()` at run time: **104 / 72 / 0 / 92 / 0**.

---

## §7 — FLAG PROVENANCE (governs every number in this report)

Commit **`5db2496`** (tree of record; the settle-bearing cells ran at `352ac46`, the SP-1-bearing
stages K6/K7-CAP/K8 and the frozen JSON were re-run at `5db2496` after the §10.1 intercept fix) · branch **`c2w11-substrate-and-kills`** · base **`main @ 2e1cdb2`** ·
worktree `../CHLU-c2w11a` · **main venv** (protocol §4; `jax 0.9.0`, float32) · **seeds 0/1/2**
(claim/instrument), **100/101/102** (width selection only).

| flag | value | note |
|---|---|---|
| `write_mode` | **`placing`** | ⭐ repair 1; `gradient` is the bit-identical OFF path |
| `launch_mode` / `n_channels` `k` / `n_particles` | **`feature_factored`** / `None ⇒ F` / **4** | ⭐ repair 3 |
| `atom_width_frac_spacing` / `atom_width_selected_frac` | **0.37 / 0.37** | ⭐ repair 2, **SELECTED**, not inherited |
| resolved atom width | **0.3178** | `= 0.37 × 0.8590` (store-population spacing) |
| `place_depth` / `place_jitter_frac_s` / `place_stationarity_shift` | 0.30 / 0.5 / **True** (40 iters) | shift median **0.0728** |
| `n_wells / f_subset / n_items / n_unseen` | 32 / 4 / 128 / 512 | registered design point |
| `atoms_per_well a` | **12** (K1 also at 4 and 32) | ⭐ **D2 superseded**: `a=32` no longer required |
| `payload_dim m` / `payload_radius` / `atom_payload_init_radius` | **8** / 1.0 / **1.0** | C2W5 D1/D3/D4, **re-verified** (§3, K2) |
| `addr_dim d` | 4 (+ declared out-of-protocol diagnostic at 8/16/32) | |
| `s_measured` (ruler) / `target_ds` / achieved | 0.318 → **0.3199 measured** / 2.7 / **2.684** | α‖q‖² subtracted; fit `R² = 1.0000` |
| `store_population_spacing` (median NN, 3 seeds) | **0.8586 / 0.8586 / 0.8586** | `σ_q/spacing = **0.1747**` |
| `query_sigma σ_q` / `confine α` / `2α` | 0.15 / 0.05 / 0.1000 | |
| `gamma_address` / `gamma_read` / read budget | 0.05 / 0.02 / **400 + 800**, `dt = 0.05` | ⚠ every γ statement is read-budget-scoped |
| `kinetic_mode` / `p₀` / `lambda_traj` | `newtonian_learned` / 0 / **0.0** | trajectory write term NOT deployed (declared) |
| `reach_radius_frac_s` / `coverage_trigger_threshold` | **2.0 / 0.20** | both registered before the run |
| `depth_ratio` / `soft_cert_B` | 3.0 / 0.542 | `bprime-c6`'s re-located edge |
| organizer | ⛔ **NOT RUN** (spoke B's) | this spoke writes and reads; it does not organize |
| bytes | store **21 504 B** (`a=12`), φ **576 B**, launch head **0 B**, ratio **3.83×** | ⛔ reported, never claimed |

---

## §8 — ⭐ PREREG SCORECARD (`.claude/outputs/c2w11-substrate-and-kills/PREREG.md`, filed before any cell)

| # | registered | measured | verdict |
|---|---|---|---|
| **K0** frac ≥F | 0.97, band [0.85, 1.00] | **0.9967** | ✅ **HIT** |
| **K0** mean distinct | 3.95, [3.70, 4.00] | **3.9967** | ✅ HIT |
| **K0** baseline reproduction | 0.050 / 2.20 ± 0.02 | **0.0378 / 2.168** | ◐ same story, `≥F` fraction 0.012 below the band (3 seeds vs 5) |
| **K1** | loss 0.030 [0.005,0.20]; capture 0.95 [0.60,1.00]; P(pass)=0.60 | **0.0007–0.0027**; **1.00** | ✅ **HIT**, loss at the band edge (better than predicted) |
| **K2** `m=8` / `m=1` | 1.000 / 0.005 [0,0.02] | **1.000 / 0.0052** | ✅✅ HIT (independent reproduction of D1) |
| **K3** | 0.000 / 0.001; vacuous with P=0.80 | **0.0000 / 0.0013**; **vacuous** | ✅ HIT |
| **K4** | all ≈0.000; vacuous with P=0.75 | **≤0.00195**; **vacuous** | ✅ HIT |
| **K5** | margin 0.00 [0.00,0.10]; P(pass)=0.20 | **0.0000**, FAIL | ✅ HIT (the pessimistic branch) |
| **K6** | 0.010, band [0.002, 0.08] | **0.0007** | ⛔ **MISS — below my band** (3.5× below the floor) |
| **K7-CAP** params | 104/72/0/92 + 0 vs 256 | **exact** | ✅✅ HIT |
| **K7-CAP** SP-1 | 1.0000, `‖v̂−v‖∞ < 1e-10` | **1.0000**, **4.25e-15** | ✅✅ **HIT** (and it reproduces the banked 4.25e-15 to the digit — ⚠ only after the intercept defect of §10.1 was fixed) |
| **K8** rank / probe | rank 24; exact-set < 0.05; `‖v̂−v‖∞ > 0.1` | **24**; **0.0488/0.0684/0.0723**; **0.69–0.73** | ◐ rank+recovery HIT; **exact-set exceeds 0.05 on 2/3 seeds** — reported, and it is *why* K8 is a structural not a numeric argument |
| **M1** neg | 0.000 / 1.00 | **0.0000 / 1.057** | ✅ HIT |
| **M2** neg | refuses | **raises, both forms** | ✅ HIT |
| **M4** | 1.000 [0.90,1.00] | **1.000** | ✅ HIT |
| **M5** launch / settle | 1.00 / 0.60 [0.30,1.00]; P(COLLAPSED)=0.55 | **1.000 / 0.885**, **not collapsed** | ✅ HIT (upper half of my band) |
| ⛔ **M6** launch precision | **0.62** [0.45, 0.80] | **0.2308** | ⛔⛔ **REFUTED** — badly, and §4.3 names why |
| ⛔ **M6** dividend | **−0.040** [−0.20, +0.06]; P(≥0)=0.35 | **−0.1567 ± 0.0052** | ◐ inside my band; **the sign prediction beat the Hub's Q3 (0.50)** |
| **M6** distinct pair | 3.95 → 3.20 | **4.00 → 3.81** | ◐ settle side better than predicted |
| **W1** selected `w_frac` | **0.37** [0.30, 0.45] | **0.37** | ✅✅ **HIT exactly** |
| **W2** `d/s` at `w=1.5` / K1 fails | 0.67 [0.45,0.95]; P(fail)=0.85 | **0.57**; **FAILS** | ✅ HIT |
| **W3** store spacing | 0.86 [0.78, 0.96] | **0.8586** | ✅ HIT |
| **W4** `σ_q`/spacing | 0.175 [0.15, 0.21] | **0.1747** | ✅✅ HIT |
| **W5** estimator `R²` | 0.995 [0.95, 1.000] | **1.0000** | ✅ HIT |
| **C1** uncovered fraction | 0.45 [0.20, 0.70] | **0.7546** | ⛔ **MISS — above my band** (worse than I predicted) |
| **C2** trigger fires | FIRES, P = 0.75 | **FIRES 3/3** | ✅ HIT |

**Score: 21 hits, 2 partial, 3 misses (K6 low, M6 launch precision, C1 high).** ⚠ **Every one of my
three misses is in the pessimistic direction** — the launch head is less precise, the coverage worse
and the asserted sets less often right than I registered. That is the *opposite* of C2W5's banked
scorecard shape ("systematic over-prediction of the store's capability"), and only because I
registered a launch-head prediction that was too optimistic for `d = 4`.

---

## §9 — DECLARED OUT-OF-PROTOCOL DIAGNOSTIC (⛔ never a score)

Changing `d` **re-draws φ**, so these are **not matched arms**. Run because spoke B must not discover
the `d`-dependence mid-run. Seed 0, no store, 512 unseen queries:

| `d` | `≥F` distinct | **correct-and-distinct** | precision | exact-set occupancy | ⭐ **K6** |
|---|---|---|---|---|---|
| **4 (registered)** | 0.994 | **0.951** | 0.238 | 0.0000 | **0.0000** |
| 8 | 1.000 | 1.637 | 0.409 | 0.0352 | 0.0352 |
| 16 | 1.000 | 3.178 | 0.794 | 0.5273 | ⛔ **0.5273** |
| 32 | 1.000 | 3.984 | 0.996 | 0.9844 | ⛔ **0.9844** |

⛔ **At `d ≥ 16` the launch head answers the family before any store or reader exists** (K6 = 0.53 /
0.98). That is reconciliation item 5 and it is a trap for anyone reading C2W5's `d`-sweep as an
invitation to raise `d`: the *store's* score would rise for a reason that has nothing to do with the
store. `d = 4` is the only cell in this table where K6 ≈ 0 — which is exactly why it is the registered
design point, and why every fitted-reader number in §3 is interpretable.

---

## §10 — HOW I VERIFIED (commands + observed output)

```bash
# worktree, off the named base (never origin/main)
git worktree add ../CHLU-c2w11a -b c2w11-substrate-and-kills 2e1cdb2

# the harness, main venv, cwd = the worktree (protocol §4)
PYTHONPATH=/Users/user/Desktop/CHLU-c2w11a /Users/user/Desktop/CHLU/.venv/bin/python \
  -m chlu exp-c2w11-substrate --seeds 0 1 2 --out-dir .../run1
# -> K0 0.9941/0.9980/0.9980 ; C2W5 offsets 0.0391/0.0312/0.0430
# -> width: only w=0.37 qualifies (d/s 2.68) ; w=1.50 -> d/s 0.57, K1 FAIL
# -> M6 dividend -0.1606/-0.1577/-0.1519 ; K1 PASS at a=4,12,32
# -> K5 FAIL (margin 0.0000, vacuous) ; K8 rank 24 < 32
# -> coverage fired 0.7461/0.7520/0.7656 ; kills_all_passed = False

# targeted tests
python -m pytest -q --no-cov tests/test_c2w11_substrate.py    # -> 25 passed
python -m ruff check <the five files>                          # -> All checks passed!
```

⚠ **The clean run was taken with NOTHING else running** (HEAD verified identical before and after:
`5db2496`), after the two defects of §10.1 were fixed.

### 10.1 ⚠⚠ SUITE HONESTY NOTE — **the suite caught two real defects in my own code, and my first diagnosis of them was WRONG**

The full suite failed **6 tests** while `tests/test_c2w11_substrate.py` passed **25/25 in isolation**.

⚠ **My first hypothesis was process contention** (I had run other JAX/pytest processes alongside the
suite, which this program's environment note forbids). **That hypothesis was wrong**, and I record it
because acting on it would have meant re-running until green and shipping a real bug. Reproducing
deterministically —
`pytest tests/ -k "c2w11 or test_a_stability or admission"` → **6 failed, 22 passed, in 14 s** — gave
the actual cause:

> ⛔ **An earlier test module enables `jax_enable_x64` GLOBALLY.** `place_write` was then reached with
> a **float64** jitter draw and a **float32** target, and `lax.scan` requires the carry's dtype to be
> invariant ⇒ `TypeError`. **It fails only in-suite and passes alone** — the defect class that a
> "re-run it clean" reflex hides.

**Fixed** (commit `5db2496`): every array is promoted to one dtype before the scan, and `place_write`
follows the **store's** dtype rather than pinning float32. Pytest-pinned by
`test_the_stationarity_shift_survives_a_MIXED_dtype_call` and
`test_place_write_follows_the_stores_dtype_rather_than_pinning_float32`.

⭐ **A second defect was found the other way — by reading a number, not by a crash.** The SP-1
out-of-class probe appended an intercept column that is **exactly collinear** with the indicator block
(every row sums to `F`). `lstsq` then returns a minimum-norm solution that splits weight into the
intercept: `y` stays exact while `v̂` is shifted by a constant, so `‖v̂−v‖∞` was measuring **the fit's
gauge, not payload recovery** — 0.055 / 0.110 / 0.085, against a banked 4.25e-15 that I should have
noticed it contradicted. Removing the intercept gives **1.50e-15 / 3.00e-15 / 4.25e-15**, reproducing
the banked value essentially exactly, and sharpens K8's structural separation from ~7× to **fourteen
orders of magnitude**. Pytest-pinned by
`test_SP1_probe_takes_NO_intercept_because_it_is_collinear_with_the_indicator`.

| checkout | commit | **collected** | result |
|---|---|---|---|
| base, fresh detached worktree `../CHLU-c2w11base` | `2e1cdb2` | **1 579** | *collect-only; greenness at this commit is C2W8-close's own measurement (1 579 passed / 0 failed at `70b11ae`, the branch merged here)* |
| **my branch**, worktree `../CHLU-c2w11a` | `5db2496` | **1 607** | ✅ **1 607 passed, 0 failed in 2 949 s (49 m 09 s)** |

**Count arithmetic (both measured in MY worktree with the same interpreter):** `1 607 − 1 579 = +28`,
and **28 is exactly the number of tests in `tests/test_c2w11_substrate.py`** (25 as first landed, plus
the 3 regression tests of §10.1) — I added no tests elsewhere and removed none. ⚠ Counts are comparable only within one checkout; both numbers above were
taken with the main venv on this machine.

---

## §11 — DECLARED NOT-RUNs (⛔ never reported as nulls)

1. **ψ, the novelty head, the organization loss, every null arm, the organizer swap** — not mine.
2. **K4 at full ψ capacity** — I do not own ψ. The store-only form is landed and **blocking**; the
   frozen obligation is named in the interfaces JSON as `k4_full_psi_obligation` with its legs, bar
   and the assertion baked in, for spoke B.
3. **M3 (per-feature G-ADDR)** — `well_lifecycle.py` / `test_gate_addr.py` are C2W8-close's territory
   and READ-ONLY all wave. Owner named, not measured.
4. **M7 / M8** (curvature-shape term and its end-of-training spectrum) — the loss-package spoke's.
5. **V1 / V2 / V3, `OD`, `OD_min`, any tier-ii or full-CLU verdict** — VALUE, wave level. I froze the
   V3 budget grid only.
6. **`k = 8` channels** — priced but not run as a cell; the `d`-diagnostic (§9) was the cheaper and
   more informative of the two out-of-protocol sweeps and it is the one I bought.
7. **5-seed cells** — every cell here is a 3-seed **instrument** cell, as the standing discipline
   permits; ⛔ no cell in this report is a claim cell and none may be quoted as one.
8. **Attention-ψ · `d = 16` as an operating point · wormholes/learned `p₀` · `lambda_traj > 0`** —
   inherited NOT-RUNs, unchanged.

---

## §12 — Git footprint

**Branch** `c2w11-substrate-and-kills` (worktree `../CHLU-c2w11a`), base **`main @ 2e1cdb2`**, 5
commits, **not pushed** — left for Hub review:

| commit | subject | files |
|---|---|---|
| `2a582e9` | land all THREE C2W11 CLI subcommands in one commit | `chlu/cli/experiment_cmd.py` (+59/−0) |
| `dec0c40` | the placing write and the re-selected co-scaled width | `chlu/core/factored_store.py` (additive) |
| `158ddc0` | feature-factored launches: one particle per phi channel | `chlu/core/feature_launch.py` (new) |
| `352ac46` | K0–K8 and M1/M2/M4/M5/M6, in the pre-registered run order | `chlu/experiments/exp_c2w11_substrate.py` (new), `tests/test_c2w11_substrate.py` (new) |
| `5964bf1` | emit chance PER SEED and make the freeze stage re-loadable | `chlu/experiments/exp_c2w11_substrate.py` |
| `5db2496` | two defects found by RUNNING it: an x64 carry and a collinear intercept | `chlu/core/factored_store.py`, `chlu/experiments/exp_c2w11_substrate.py`, `tests/test_c2w11_substrate.py` |

⛔ **Not touched:** `chlu/config.py` (zero C2W11 spoke touches it) · `chlu/core/well_lifecycle.py` ·
`chlu/experiments/exp_well_lifecycle.py` · `chlu/core/clu_system.py` · `chlu/core/soft_certificate.py`
· `tests/test_gate_addr.py` · `tests/test_well_lifecycle.py` · `tests/test_cifar_strong_phi.py` ·
`chlu/core/null_arms.py` · `chlu/experiments/exp_null_arms.py` · `chlu/core/psi_readout.py` ·
`chlu/core/emission_head.py` · `chlu/experiments/exp_capture_strong_phi.py` · `scripts/csf3/` ·
`chlu/training/train_cluformer.py` · `chlu/core/blocks.py` · `chlu/experiments/exp_cluformer_pilot.py`.
⚠ **`tests/test_factored_store.py` was NOT edited** — my additive tests all landed in the new
`tests/test_c2w11_substrate.py`, which keeps the C2W5 file bit-identical for anyone diffing it.
**No conflicts.** Rebase onto `main`: no-op (base has not moved).

**Artifacts:** `.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json` ·
`.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md` ·
`.claude/outputs/c2w11-substrate-and-kills/PREREG.md` ·
`.claude/outputs/c2w11-substrate-and-kills/run1/{stage_k0,stage_width_selection,stage_m6,stage_k6_k7cap,stage_k1,stage_k2,stage_k3_k4_k5,stage_k8,stage_m4,stage_coverage}.json`
+ `c2w11_substrate_summary.json`.

---

## §13 — Open questions / follow-ups / risks

1. ⭐⭐ **The one that decides the wave: is `d = 4` survivable?** The launch head's precision is capped
   by code collinearity at `d = 4` (channels 2–4 at 0.15 vs chance 0.125), and raising `d` fixes it
   *and* switches on the K6 leak (0.53 at `d=16`). **SP-2's window looks empty again, now measured on
   the launch side as well as the reader side.** If the Hub wants a non-vacuous cell, the honest lever
   is a **different code geometry at `d = 4`** (near-orthogonal / equiangular codes), not a bigger `d`
   — and that is a φ change, which is nobody's territory this wave.
2. ⭐ **Reach is now quantified: `1.172 × capture radius`.** The gap is the payload radius, exactly
   1.0 by construction. Closing it needs either learned `p₀` (C2W9) or a read that is allowed to
   start off the payload-zero manifold without being handed the answer. **Nothing in this wave's
   budget can close it.**
3. **Should spoke B run at all?** `kills_all_passed = false` on K5, vacuously. My reading (⛔ not an
   adjudication — that is the Hub's): the VALUE legs cannot be scored on a cell where every arm reads
   chance, and **V3's swap-differenced curve in particular would be a difference of two zeros.**
4. **The drift tension (reconciliation 1) needs a ruling before any census number is quoted on a
   placed store.**
5. **Risk in how §2.1 gets quoted.** "K0 moved 0.038 → 0.997" is true and is the headline of the
   substrate repair — ⛔ **it must never be quoted without "and correct-and-distinct moved
   0.860 → 0.922, and neither arm ever reached 4"** in the same sentence.

## Proposed handover updates (for the Hub)

- **§3 config / CLI — NEW:** `chlu exp-c2w11-substrate [--seeds …] [--stages k0 m6 width k1 k2 k3 k4
  k5 k6 k7cap k8 m4 m5 coverage freeze] [--quick] [--out-dir]`; plus **registered-but-gated**
  `exp-c2w11-organizer` and `exp-c2w11-nulls` stubs (spoke A owns `experiment_cmd.py` this wave).
  New module `chlu/core/feature_launch.py`. New `CatTestConfig` fields, all defaulting to C2W5
  behaviour (`CatTestConfig().as_flag_table() == {}` still holds): `write_mode`, `place_depth`,
  `place_jitter_frac_s`, `place_stationarity_shift`, `place_shift_iters`, `atom_width_frac_spacing`,
  `atom_width_selected_frac`, `width_guard`, `launch_mode`, `n_channels`, `reach_radius_frac_s`,
  `coverage_trigger_threshold`.
- **§7 Known Issues — NEW (open, needs a ruling):** *a placing write's site drift (`2e-6 × spacing`)
  fails the LOW side of C2W8-close's two-sided G-DRIFT leg*, while the placing write is the charter's
  carried substrate (§A34.10). Instrument and substrate are in tension. **Reported, unadjudicated.**
- **§7 Known Issues — NEW (open):** *the feature-factored launch head answers the family by itself at
  `d ≥ 16`* (K6 = 0.527 at `d=16`, 0.984 at `d=32`). Any `d`-sweep must report **K6 per `d`**.
- **§7 Known Issues — RESOLVED/SUPERSEDED:** C2W5's D2 (`a = 32` forced by K1) — the placing write
  passes K1 at `a = 4` (byte ratio 9.67× → 1.50×). And *"the `P`-particle occupancy read cannot
  express an `F`-term sum at `P = F`"* is **half-resolved**: `≥F` distinct wells now 0.997, but
  **correct-and-distinct is unmoved (0.860 → 0.922)** and exact-set occupancy is still 0.0000.
  ⛔ The Known Issue should be **rewritten, not closed**.
- **Registry/doctrine candidates:** (i) ⭐⭐ *quote `correct-and-distinct` beside every
  addressability number* — "distinct wells reached" and "occupancy precision" can each move a lot
  while the statistic a compositional read needs moves not at all; this wave would have mis-read its
  own headline without it. (ii) ⭐ *K6 is a launch-side statistic and must be swept over any axis that
  changes φ* — a launch head that is good enough to answer the question makes every fitted-reader
  score uninterpretable. (iii) ⭐⭐ *"it passes in isolation and fails in the suite" is a BUG REPORT, not a flake report*
  (§10.1, self-filed): my first diagnosis was contention, and acting on it would have meant re-running
  until green and shipping a live dtype bug. The suite's global `jax_enable_x64` flip is a standing
  hazard for any new `lax.scan` carry in this repo. (iv) ⭐ *a diagnostic that contradicts a banked
  number is wrong until proven otherwise* — SP-1 read 0.055-0.110 against a banked 4.25e-15 and the
  gap was the diagnostic's own collinear intercept, not the physics.
- **`PREREG-C2W11.md` scoring input:** **Q1 (K0 clears 0.80, 0.78) — CONFIRMED at 0.9967.**
  **Q3 (M6 dividend non-negative, 0.50) — REFUTED at −0.1567 ± 0.0052.** Q9 (the C2W9 trigger fires,
  0.55) — **CONFIRMED (coverage half).** Q2 (K1 at `a ≤ 12`, 0.70) — **CONFIRMED, at `a = 4`.**

---

# ⭐⭐ §14 — DATED ADDENDUM (2026-08-11): **THE PAYLOAD-REACH REPAIR**

> ⚠ **Numbering note (the C2W8-close vi.3 collision precedent).** The task file asked for "a dated
> §12 addendum". **§12 (git footprint) and §13 (open questions) already exist above and the body above
> this line is UNTOUCHED**, as instructed — so the addendum is filed as **§14** rather than
> overwriting a live section. Nothing in §§1–13 has been edited.

**Task:** `.claude/tasks/c2w11-payload-reach-repair.md` — close the ONE measured arithmetic blocker
(`‖v_j‖ / capture = 1.172 > 1`) to a **pre-registered ratio**, then re-run the kill set and score K5
**once** at the repaired operating point. **Status: done.**
**Full report:** `.claude/outputs/c2w11-payload-reach-repair.md`. Only the headline is repeated here.

## 14.1 What moved

| | §1–13 (broken reach, `‖v‖ = 1.0`) | **§14 (repaired, `‖v‖ = 0.60`)** |
|---|---|---|
| ⛔ `‖v_j‖ / measured SC-6 capture` | **1.172** | ⭐ **0.692 / 0.692 / 0.681** (claim seeds) |
| wells with `‖v_j‖ ≤ capture_j` (**per well**) | 0.000 | **1.000 / 0.938 / 0.938** (45/48) |
| payload-direction reach (the direction the read crosses) | **0.855 < 1.0** | **0.932 > 0.60** |
| full-space coverage: needed wells uncovered | **1.0000** | **0.9299 ± 0.0038** |
| ⭐ **M6 dividend** (DIAGNOSTIC) | **−0.1567 ± 0.0052** | ⭐ **−0.0015 ± 0.0026** (2 SE spans 0) |
| M6 launch → settle precision | 0.2303 → **0.0736** | 0.2303 → **0.2288** |
| M6 distinct wells launch → settle | 3.998 → 3.807 | 3.998 → **3.990** |
| K4 address-leak dividend | −0.1623 | **−0.0028** |
| endpoint write loss (`a = 12`) | 0.0007–0.0027 | 0.0077–0.0121 (bar ≤ 0.05) |
| site drift / spacing | 2.0e-6 | **4.3e-4** (still below the 0.01 floor) |

## 14.2 What did NOT move — and it is the wave's result

⛔ **K5 still FAILS, still VACUOUSLY** (best margin **0.0000 / 0.00195 / 0.0000**; read, table and
chance all ≈ 0.001), so **`kills_all_passed` remains `false`**. **K0 is bit-identical**
(0.9967 / 3.9967 — it has no store and no payload), **K6 = 0.0007**, **K7-CAP params 104/72/0/92/0**,
**K1 passes at `a` = 4, 12 and 32**, **K2 = 100 % at `m = 8`** with its scale-invariance **measured**
(every m-sweep value bit-identical to §3's while `tol` scaled by exactly 0.6), **K3/K4 green (vacuous)**,
**M4 = 1.000**, **M5 `W/N_a` = 1.000 → 1.000, OK**. **The registered address-space coverage statistic
is IDENTICAL at 0.7546 uncovered and the trigger still fires.**

> ### ⭐⭐ THE FINDING
> **Two failures were entangled; the repair separated them.** The **reach** failure was arithmetic and
> is **closed** (1.172 → 0.692), and closing it **repaired the settle**: the occupancy dividend went
> from **−0.1567** (the settle destroying 68 % of the launch's correct-well information) to
> **−0.0015 ± 0.0026**, i.e. **indistinguishable from zero, with one seed positive**. But the read is
> **still not expressible**: the binding constraint is now the **launch head's precision**
> (`occupancy precision 0.2303`, `correct-and-distinct 0.92 of 4`, address-space coverage 0.7546
> uncovered) — an **address-side** quantity that no payload radius can touch.
> ⛔ **"The read is not expressible with reach controlled for" is a stronger result than §3's**, and it
> was **pre-registered** (`PREREG.md` ADDENDUM §A2: K5 margin 0.000, P(pass) = 0.10; M6 dividend −0.04,
> P(≥ 0) = 0.20).

## 14.3 ⭐ The family-construction law, now measured on BOTH sides

Registered before the sweep as `‖v_j‖ < capture_radius ≲ min-well-spacing`. The sweep measured a
**floor** as well: at `‖v‖ ≤ 0.20` the **placing write itself fails K1** (endpoint loss 0.058–0.062 >
0.05, capture ≥ σ_q on only 0.84–0.91 of wells), because neighbouring wells' placed clouds crowd in
the payload block. ⇒

> **`k1_write_floor ≲ ‖v_j‖ < capture_radius ≲ min-well-spacing`** — an admissible **interval**,
> measured here as `‖v_j‖ ∈ [≈0.25, ≈0.64]` at (`N_a = 32`, `F = 4`, `m = 8`, `d = 4`, `w_frac = 0.37`).
> C2W5's D3 picked **1.0**, which is **above** the interval; the repair picks **0.60**, inside it.

## 14.4 Superseding notes for anyone quoting §§1–13

1. ⛔ **The operating point MOVED:** `payload_radius = atom_payload_init_radius = 0.60`. `tol` is
   **0.2870 / 0.2828 / 0.2796** (was 0.4783 / 0.4714 / 0.4660) — read it from the re-emitted
   `FROZEN-INTERFACES-C2W11.json`, never from §7's flag table.
2. `v3_budget_grid`, `k8_structural_split`, the reader class, φ, the launch protocol and the selected
   width (`w_frac = 0.37`) are **byte-identical** to the frozen file spokes B and C were gated on.
3. §13 item 2 ("Nothing in this wave's budget can close [reach]") is **superseded**: it was closed,
   inside the wave, by lowering the payload radius — and closing it did **not** move K5.
4. **§5 and `TRAVERSAL-FAILURE-SIGNATURE.md` §1 stand**; the post-repair coverage numbers are in that
   file's new **§1b**, which records that the coverage mode **PERSISTS**.
