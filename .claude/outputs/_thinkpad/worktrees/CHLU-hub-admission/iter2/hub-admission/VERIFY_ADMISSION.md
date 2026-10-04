# `VERIFY_ADMISSION` — adversarial verification of `hub/admission`, seed 0, rung A

**Verifier:** sub-agent of `hub/admission` (Opus), FORGIS_14, 4 threads, venv py 3.11.16.
**Mandate:** try to break a NEGATIVE result. Every gate number below was recomputed **from the
formula in PREREG-1 §1 in float64 numpy**, without calling `admission_lib`; `admission_lib` and
`admission_ctrl` appear only as objects to *disagree with*, and `admission_ctrl.KnnGate` only
because it is the rival.
**Scripts:** `scorer/verify_core.py`, `scorer/verify_liveset.py`, `scorer/verify_launch.py`,
`scorer/verify_fairness.py`. **Data:** `VERIFY_CORE.json`, `VERIFY_LIVESET.json`,
`VERIFY_LAUNCH.json`, `VERIFY_FAIRNESS.json`.
⛔ Nothing was deleted, nothing under `chlu/` was touched, no row was re-encoded, `ctx["eval_b"]`
was never read.

---

## LEAD — the three things that are wrong, in order of how much they matter

### ⛔ 1. There IS a launch point you missed at which the gate is not vacuous — the **SETTLED STATE** — but it does **not** rescue the live-set claim, and on its own legs it is **worse** than the R8 head.

`gen_landscape.settle_chain` at the shipped `SettleCfg` (1000 steps, `step_size = 0.2`,
`sigma_scale 4 → 1`, `T = 0`, metropolis) launched from the shipped `clamp_atom` state is
**shipped, request-only, non-fitted**, the gate stays at `E_k > 0`, and `σ_k` is not scaled.
You tested `clamp_atom`, `prior_atom`, `posterior_mean`, `r8_affine` and raw features — not the
point the read actually produces. Measured (n = 128/group, full store, floor_hue perturbations):

| population, gated at the SETTLED state | refuse | mean `max_k E_k` | mean `min_k d/σ_k` |
|---|---|---|---|
| **valid `val`** | ⛔ **0.1094** | −24.7 | 0.849 |
| `out_of_range` w=2 | 0.0000 | −33.3 | 0.793 |
| w=3 | 0.0000 | −31.1 | 0.806 |
| w=4 | 0.0234 | −27.9 | 0.826 |
| w=7 | 0.1641 | −19.3 | 0.855 |
| w=13 | 0.3359 | −9.2 | 0.934 |

The gate is **live** here — `refuse ≢ 0`, so "the derived gate never fires" is too strong as a
sentence about the object. But it refuses **more valid requests (0.109) than invalid ones at
w = 2, 3 and 4**, so it is strictly dominated by `gate@r8_affine` and it is not an arm.

**And it is not live-set sensitive.** Settling from *your own* hybrid exemplar launch, on *your
own* rebuilt held-out-value stores:

| rebuilt store | refuse(held-out) | refuse(in-vocab) | separation |
|---|---|---|---|
| floor_hue, exemplar (P2) | 0.2266 | 0.2656 | ⛔ **−0.0391** |
| wall_hue, exemplar (P2) | 0.1875 | 0.2344 | ⛔ **−0.0469** |
| floor_hue, OOV-symbol (P3) | 0.1953 | 0.2500 | −0.0547 |
| wall_hue, OOV-symbol (P3) | 0.3906 | 0.1641 | +0.2266 |

Both exemplar separations are **negative**; the symbol pair disagrees in sign between two
factors. ⛔ **Your conclusion survives. There is no launch point at which `E_k > 0` discriminates
a held-out value.**

**⭐ The mechanism, and it is a shipped-code fact worth passing to the coordinator.**
`total_energy = E_store + β·Σ_t (E_t − E_store)`, so at `β = 1` with `T` requested terms the
store's own free energy enters the settling dynamics with weight **`1 − T`**. With the six
factor terms of a `shapes3d` request that weight is **−5**: settling *pushes the state away from
the atoms by construction*. Measured on VALID `val`: `mean max_k E_k` rises **−36.9 → −24.7** and
`min_k d/σ_k` rises **0.000 → 0.849** over the 1000 steps. That is why the settled-state gate
fires on valid data, and it is the same sign error of intent that makes the level set unusable
from the other direction. (`read_settled(..., "atom_mixture")` on the settled state refuses
**0.0000** — convex hull of the atoms, dead as expected.)

### ⚠ 2. `DEV_TABLE`'s pair comparison is **fair but conservative — it understates your loss by 0.039.**
Not an error in your favour that needs retracting; an error *against* you that should be stated.
See claim 4 below: at `q = 0.95` per column the factored rival scores **0.8241**; its best over
`q` is **0.8621** at `q = 0.98`. `Δ` at the rival's own best operating point is **−0.1772**, not
**−0.1383**.

### ⚠ 3. One vacuous assertion in `u4_liveset.py` (right conclusion, test that cannot fail).
`u4_liveset.py:99` — `shortfall = setdiff1d(pool[keep], pool).size + setdiff1d(pool[held_mask],
pool).size` is **identically 0**: both arguments are subsets of `pool` by construction, so
falsifier §5.6 was never actually tested. `rec["shortfall_total"]` (line 72) is also never
re-assigned, so `rec["F5_6_fires"]` (line 206) is hard-wired `False`.
⭐ The **conclusion is nonetheless correct** and I verified it by a different route: the rebuilt
landscape is built from `X_pool[keep]`, a slice of the array `gather_features` already returned,
so `gather_features` is never re-entered and no row can be encoded. Vacuous check, true claim.

---

## THE VERDICTS

### CLAIM 1 — "`E_k > 0` refuses **0.0000** of real encoded images, even after every atom carrying that value is deleted" — ⭐ **CONFIRMED**
*(`verify_liveset.py`; my own float64 gate, my own row draws — a different 256 images per factor
from yours, so this is not a re-run.)*

**(a) The hold-out is real, and the rebuild is not a no-op.** Re-derived independently
(`store_r.atom_rows → Pk → grid.factor_indices_of`): **0** atoms carry the held-out value on all
six rebuilds. Additionally — which your check does not test — the rebuilt stores share only
**0–1 of 64** atoms with the full store, and the closest rebuilt atom is **2.72 – 12.56** in L2
from *any* deleted row. The store really changed.

**(b) The hybrid is exactly what it claims to be.** For all six factors,
`X_held_h[:, assigned] == X_pool[held_idx][:, assigned]` **bit-exactly** and
`X_held_h[:, other] == R8_embed[:, other]` **bit-exactly**. The real image occupies
366 / 1181 / 1074 / 1440 / 1074 / 608 channels = **17.9 % – 70.3 % of the 2048 dims**. Request
and image are the same row (`held_idx` indexes both). ⛔ No mis-indexing, no overwrite.

**(c) The refusal really is 0 — and not marginally.**

| factor | my refuse(held-out hybrid) | yours | my refuse(in-vocab) | worst single query `max_k E_k` |
|---|---|---|---|---|
| floor_hue | **0.0000** | 0.0000 | 0.0000 | −23.50 |
| wall_hue | **0.0000** | 0.0000 | 0.0000 | −10.41 |
| object_hue | **0.0000** | 0.0000 | 0.0000 | −37.84 |
| scale | **0.0000** | 0.0000 | 0.0000 | −9.88 |
| shape | **0.0000** | 0.0000 | 0.0000 | −35.88 |
| orientation | **0.0000** | 0.0000 | 0.0000 | −43.79 |

The *closest any held-out exemplar query anywhere came to the gate* is `max_k E_k = −9.88`. This
is not a threshold that was narrowly missed.

**(d) ⭐ THE STRONGEST VARIANT YOU ASKED FOR — the FULL, unmodified 2048-dim cached feature, no
hybrid, against the rebuilt store. It does NOT refuse at a materially non-zero rate.**

| | mean over the six factors |
|---|---|
| **full feature, held-out images, REBUILT store** | ⛔ **0.0007** (1 query of 1536; wall_hue only, grazing at `max_k E_k = +0.04`) |
| full feature, in-vocab images, rebuilt store | **0.0000** |
| **the same held-out images against the FULL store** | **0.0013** |
| `min_k d/σ_k`: held-out **0.819** vs in-vocab **0.782** | |
| `knn_factored` on the same full features | refuse(held) 0.787 vs refuse(in-vocab) 0.687 |

⭐ The held-out images refuse **less** against the damaged store (0.0007) than they do against the
**undamaged** one (0.0013), and both sit at the generic real-feature false-refusal floor you
already report (0.0005 – 0.0007). ⛔ **Your claim is not too strong. `0.0000` (hybrid) and
`0.0007` (full feature) are the same measurement: nothing.**

### CLAIM 2 — "the data manifold sits at `d/σ_k ≈ 0.79`, the zero level set at `≈ 2.0–2.4 σ_k`" — ⭐ **CONFIRMED**
*(`verify_core.py`; my own `nn_median`, my own energy, my own bisection — no grid search.)*

| quantity | yours | **mine** |
|---|---|---|
| `σ_k` | 4.6343 / 5.0714 / 9.1827 / 6.3054 / 5.1006 | identical, max rel delta **4.5e-08** |
| real features, all `store_train`, mean `min_k d/σ_k` | 0.787 | **0.7852** |
| real features, `val` rows | 0.793 | **0.7934** |
| R8 affine, valid | 0.775 | **0.7753** |
| zero level set `/σ_k`, per well | 2.093 / 2.043 / 2.419 / 2.343 / 1.892 | **2.090 / 2.038 / 2.410 / 2.342 / 1.901** |
| certified `σ_k√(2 ln 64)` `/σ_k` | 2.884 | **2.884** |
| worst `max_i E_k(a_ki)` (the identity) | −30.394 | **−30.3937** |

**No sign error, no missing `√`, no factor of 2.** Writing the kernel in its stable form makes the
whole chain auditable in one line: with `m = min_i d²` and `S = Σ_i exp(−(d²_i − m)/2σ²) ≥ 1`,

> `E_k = m/2 − σ_k² · log S`, so `E_k > 0 ⟺ m/(2σ_k²) > log S`, and `log S ≤ log N` gives
> `d ≥ σ_k√(2 log N)` as the certified sufficient condition — exactly the radius you report.

**No float16/float32 artefact.** Gating the `val` features cast from the raw cache as float16 and
as float32 gives `min_k d/σ_k = 0.7933935315067281` **in both cases**, refuse 0.0000 in both.

**And I checked the obvious objection to your estimator, which you do not address.** Your 256
rays/well are **random directions** in a 303–573-dim well; a random direction is not where the
data is. So I re-measured the crossing along the ray from the nearest atom **towards a real `val`
feature** — the data direction:

| | well 0 | well 1 | well 2 | well 3 | well 4 |
|---|---|---|---|---|---|
| zero crossing `/σ_k`, **data direction** | **2.191** | **2.153** | **2.408** | **2.370** | **2.036** |

⭐ Slightly **further out** than your random-ray number, i.e. the objection cuts your way. The
2.6× gap between the data (0.79) and the level set (2.0–2.4) is real in the direction that
matters.

### CLAIM 3 — "`accept_A7` = 0.8056, `accept_A7D` = 1.0000; the A7 false-refusal is R8's additive double-count of a duplicated term" — ⭐ **CONFIRMED**, and it is airtight
*(`verify_core.py`. This is the one you said would be quoted; it survives every check I could
think of.)*

| check | result |
|---|---|
| `n_A7` / `n_A7D` | 72 / 72 — **the same 72 requests** |
| terms per request | A7: **{7}**, A7D: **{6}** |
| ids | `A7D[i].id == A7[i].id + "-dedup"` for **72/72**; `row` matches for 72/72 |
| requests with **exactly one** duplicated factor term | **72/72** |
| the duplicate is at the **same value** as the first term | **72/72** |
| declared `redundant_factor` agrees with the duplicate found | **72/72** |
| duplicated factors present | all six |
| `embed(A7) − embed(A7D)` vs **one** `(μ_{k,v} − μ_global)` | **max abs delta 2.98e-07** (float32 ε); rel 1.4e-07 |
| ‖that extra deviation‖ vs ‖the embedding‖ | **6.47** vs 33.46 |
| `accept_A7` (mine) | **0.8055555556** — 14 refused of 72 |
| `accept_A7D` (mine) | **1.0000** — 0 refused |
| refused(A7) ⊇ refused(A7D) | **True** (14 ⊃ ∅) |
| `min_k d/σ_k` | A7 **0.8871** → A7D **0.7798**; the valid `val` mean is **0.7753** |

⭐ Deduplication puts A7 back **exactly on the valid manifold** (0.7798 vs 0.7753). The mechanism
is confirmed as stated: one extra `(μ_{k,v} − μ_global)` of norm 6.47 added to a 33.5-norm
embedding moves it 0.107 σ further out and that is the whole 0.194 of `accept_A7` deficit.

### CLAIM 4 — fairness of the factored rival — ⚠ **the rival is DISADVANTAGED; your comparison errs in the ARM's favour by 0.039**
*(`verify_fairness.py`.)* The "refuse if ANY of `K = 5` columns exceeds its own `q = 0.95`
quantile" reduction gives a **family-wise refusal of 21.6 % on the calibration set**, not 5 %.
That is not a free gift to the rival — it is a handicap:

| per-column `q` | family-wise refuse on cal | `accept_val` | `refuse_oor` | `accept_A7` | **score** |
|---|---|---|---|---|---|
| 0.90 | 0.3745 | 0.6523 | 0.9126 | 0.3056 | 0.7825 |
| **0.95 — your choice** | **0.2161** | 0.7891 | 0.8590 | 0.3889 | **0.8241** |
| **0.98 — the rival's best** | 0.0935 | 0.9316 | 0.7926 | 0.4583 | ⭐ **0.8621** |
| 0.98979 = `0.95^(1/5)` (family-wise 5 %) | 0.0498 | 0.9648 | 0.7493 | 0.4861 | 0.8571 |
| 0.999 | 0.0059 | 0.9922 | 0.6796 | 0.5556 | 0.8359 |

⭐ **Say it plainly: the comparison in `DEV_TABLE_seed0_A.json` is FAIR — and conservative.** The
rival is run 0.038 of score below its own optimum. At the rival's best operating point
`Δ = 0.6857 − 0.8621 = −0.1772`, against the **−0.1383** you report. ⛔ Your LOSS is **understated
by 0.0389**; it is not manufactured.

**Operating-point-free decomposition (max over ALL thresholds of `½(refuse_oor + accept_val)`):**

| statistic | best achievable score | at |
|---|---|---|
| **CLU `max_k E_k`** (the arm's statistic) | **0.7994** | accept_val 0.9355 / refuse_oor 0.6633 |
| **CLU `max_k d_k/σ_k`** (same store, distance not energy) | ⭐ **0.8488** | 0.9043 / 0.7934 |
| `knn_factored max_k d_k/thr_k` (the rival) | **0.8629** | 0.9355 / 0.7902 |
| `knn_whole` latent `d` | 0.8013 | 0.9102 / 0.6924 |
| the ARM as shipped, at the DERIVED zero | 0.6857 | 1.0000 / 0.3714 |

⇒ of the arm's **0.177** deficit against the rival at its best, **0.114 is the RADIUS** (where the
derived zero sits) and **0.063 is the SIGNAL**. Both of your conclusions — "the problem is the
radius" *and* "it is not *only* the radius" — are quantitatively right.

⭐ **A new finding you can have:** on this object the CLU's own per-well **distance** `max_k d_k/σ_k`
is a **better** statistic than the **energy** `max_k E_k` it is built from (0.8488 vs 0.7994 best
score; AUROC 0.9107 vs 0.8619), and it lands within 0.014 of the fitted rival's AUROC (0.9198).
The `logsumexp` over 64 atoms is *destroying* separation relative to the plain nearest-atom
distance in the same wells. That is a property of the functional, not of the threshold.

### CLAIM 5 — `paired_bootstrap` — ⭐ **CONFIRMED** (correct, genuinely paired), with one inertness note
*(`verify_core.py`, my own re-implementation, my own seed.)*

| | yours | mine |
|---|---|---|
| `Δ` | **−0.13834635** | **−0.13834635** (exact) |
| CI95 | [−0.15612, −0.12044] | [−0.15635, −0.11979] |
| bootstrap sd | — | 0.00927 |
| verdict | LOSS | LOSS |

`score = ½(refuse + accept)` is computed right. The pairing is genuinely implemented — one
`ia` and one `ir` shared by both arms (`admission_ctrl.py:240-242`).
⚠ **But the pairing is inert on this data, and that is worth knowing rather than claiming as
rigour:** an *unpaired* bootstrap gives sd **0.00917** and CI **[−0.15632, −0.12028]** — the same
interval. The reason is structural: the arm's `accept_val` is constant at 1.0 (zero variance), and
the dominant variance term is the **rival's** `accept_val` over only 512 `val` rows
(sd ≈ 0.0091 of the half-score), which no pairing can cancel. The CI is correct; it is just not
tighter for being paired.

### ITEM 6 — "is there a launch point at which the gate IS live-set sensitive?" — ⛔ **NO**
See the LEAD. The settled state is the one shipped, request-only, non-fitted point you had not
tested, and it is **live but not live-set sensitive** (exemplar separation −0.039 / −0.047) and it
**false-refuses 10.9 % of valid requests**. `atom_mixture` reads are dead by construction. Your
conclusion stands.

---

## OTHER THINGS FOUND, none of them load-bearing

1. **`u4_liveset.py:99` + `:72` + `:206`** — the §5.6 shortfall assertion is a tautology and
   `F5_6_fires` is hard-wired `False`. Right answer, untested. (Detailed in the LEAD.)
2. **`u4_liveset.py:161-168`** — `rates()` is invoked three times per population (twice for the
   scalar keys, once for `detail_*`), so every gate in UNIT 4 is computed 3×. Wasteful, not wrong.
3. **`admission_lib.per_well_nearest`** runs the Gram form in **float32**: max abs delta vs my
   float64 `d` is **0.011** on `d ≈ 4–10` (≈ 0.2 %). `E` is fine (max rel delta **1.8e-06**) and
   there were **0 accept/refuse disagreements** over 1024 probes, so no reported number moves —
   but `d/σ_k` should not be quoted past 3 decimals.
4. **`LOG.md` UNIT 0c wording** — "accept rate on valid `val` is 1.0000 **at every launch point**"
   is now false as written: at the settled state it is **0.8906**. Suggest "at every launch point
   tested", with the settled state added as a measured negative.
5. `Σ_k E_k` (mine, float64) vs the shipped `gen_landscape.store_energy`: max rel delta
   **2.6e-07**. Your §2.1 self-check reproduces.

## THE HEADLINE, UNBROKEN

I could not break the negative. `refuse(held-out exemplar) = 0.0000` survives an independent
float64 re-derivation, different query rows, an independent verification that the atoms really
were deleted, and the strongest variant you asked for (the full 2048-dim cached feature, which
refuses **0.0007** — *less* than the same images against the undamaged store). The radius
arithmetic is right to 3 significant figures and right in the data direction as well as in random
directions. The A7/A7D finding is exact to float32 epsilon. The one thing that changes is that
the loss to the factored rival is **larger** than you reported, not smaller.
