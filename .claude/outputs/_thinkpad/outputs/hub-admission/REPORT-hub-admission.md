# `hub/admission` — REPORT

**Agent:** `autonomous-hub` (Opus), execution tier · reports to the **coordinator**
**Branch:** `hub/admission` @ `945a3e9` (`base/iter2`) · **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-admission`
**Machine (R3, on every number below):** FORGIS_14 · **Windows AMD64** · local CPU · 4 threads · ⛔ no CUDA
**Start:** `2026-09-17T14:17:12Z` · **12 h deadline:** `2026-09-18T02:17:12Z`
**Pre-registration:** `PREREG-hub-admission-1.md` · **Running log:** `LOG.md` · **Ledger:** `RESULTS_LEDGER.md`
**Object:** `shapes3d`/`recon`, **seed 0**, dynamical-read rung **A** (`N = 64` atoms/well, 524,288 B),
plus rung **B** as the capacity control.

---

# STATUS

## ⛔ **NOT SOLVED.** The derived gate does not calibrate on this object, and here is by how much.
## ⭐ **Scored runs: 0/3 spent. Three remain. `eval_b` was never read.**

The task file says the expected honest outcome may be exactly this: *"**The derived gate does not
calibrate and here is by how much** is a complete and valuable answer."* That is what this is. The bar
was set by the task file before launch, ⛔ I did not move it, and I am reporting a failure against it.

---

# 1. THE ANSWER IN ONE PARAGRAPH

The gate — `refuse well k iff E_k(x_k) > 0` on the kernel free energy — is **correctly derived, correctly
implemented, and correctly certified**; the identity `max_i E_k(a_ki) ≤ 0` holds on the live store with
30 units to spare, my per-well reduction reproduces the shipped `store_energy` to 1.2e-07, and 69–73 % of
the refusals it does make are **certified** by the `σ_k√(2 ln N)` bound. ⛔ **It is nevertheless dead on
arrival, and for one measurable reason: its radius is in the wrong place.** The zero level set sits at
`d/σ_k ≈ 2.0 – 2.4`; **the entire data manifold — every real encoder feature, held out or not — sits at
`d/σ_k ≈ 0.79`.** `σ_k` is `nn_median`, the median nearest-neighbour distance *between the atoms*, and the
atoms are train rows, so they tile their own manifold at spacing ≈ `σ_k` **by construction** and any real
point is ≈ 1 `σ_k` from one of them **whatever the store contains**. The level set is therefore pinned
~2.6× outside the data by the bandwidth rule itself, and 8× the store closes the gap by only ~0.09 in
`d/σ_k` while pushing the level set itself further out. The
consequences are exact: at the **shipped `clamp_atom` launch the gate is vacuous by algebra** (the launch
point *is* a stored atom, so `E_k ≤ 0` always); at `posterior_mean` it is vacuous because a convex
combination of atoms cannot leave their hull; only a **linear extrapolation** — the R8 affine head at
severity `w ≥ 3` — ever reaches it. ⭐ And so **the live-set claim, which is the one claim a threshold
cannot fake, fails completely: deleting every atom carrying a value changes the decision on real images
of that value by `+0.0000`, on all six factors.** The same statistic at a *fitted* radius does detect it
(factored kNN: 1.0000 vs 0.3359 on `floor_hue`), which is the cleanest possible statement of what is
wrong: ⛔ **the problem is not the signal, it is the radius — and §7 rightly forbids me to move it,
because moving it is the entire claim.**

---

# 2. THE PAIR, WITH ALL SIX CONTROLS (DEV, `val` corners; ⛔ B untouched)

`score = ½(refuse_out_of_support + accept)`. `accept_A7` beside `accept` on every row, same threshold,
per the programme-wide rule of 2026-09-17.

| arm | `accept_val` | ⭐ `accept_A7` | `refuse_oor` | `accept_unseen` | **score** | threshold |
|---|---|---|---|---|---|---|
| **`CLU_gate@r8`** — the arm | **1.0000** | 0.8056 | 0.3714 | 1.0000 | 0.6857 | ⭐ **DERIVED, no free parameter** |
| `CLU_K1@r8` — control (c), same 524,288 B | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 0.6000 | DERIVED |
| `knn_whole` — control (a) | 0.9512 | 0.6667 | 0.6213 | 0.9613 | 0.7862 | FITTED `q = 0.95` |
| **`knn_factored`** — control (b), **the fair rival** | 0.7891 | **0.3889** | 0.8590 | 0.7907 | **0.8241** | FITTED `q = 0.95` |
| `margin_M` — control (e), physics-free | 0.9590 | 0.9722 | 0.0398 | 0.9479 | 0.4994 | FITTED `q = 0.95` |
| `CLU_gate@posterior_mean` | 1.0000 | 1.0000 | **0.0000** | 1.0000 | 0.5000 | DERIVED (dead) |
| `CLU_gate@clamp_atom` (**the shipped launch**) | 1.0000 | 1.0000 | **0.0000** | 1.0000 | 0.5000 | DERIVED (**vacuous**) |

⚠ **The table above uses `refuse_oor`. The bar's own population is `out_of_support` = out-of-range ∪
out-of-vocabulary, and it is WORSE for the arm** (`POOLED_SUPPORT_seed0_A.json`). Reported at full
strength, because a bar evaluated on a favourable subset is not the bar:

| gate | `oor` (15,360) | `oov_symbol` (3,072) | `oov_exemplar` (3,072) | ⭐ **POOLED** (21,504) |
|---|---|---|---|---|
| **CLU `gate@r8`** | 0.3714 | **0.0000** | **0.0000** | ⛔ **0.2653** |
| `knn_factored` | 0.8590 | 0.4118 | 0.7288 | **0.7766** |
| `knn_whole` | 0.6213 | 0.0446 | 0.2546 | 0.4865 |

**R1 on the POOLED pair:** vs `knn_factored` **Δ = −0.1502** [−0.1675, −0.1317] ⛔ **LOSS**;
vs `knn_whole` Δ = −0.0862 [−0.0955, −0.0762] ⛔ **LOSS**.

**R1, paired bootstrap, 4,000 resamples, on `refuse_oor`:**

| arm vs rival | Δ | CI95 | R1 verdict |
|---|---|---|---|
| vs **`knn_factored`** at its pre-registered `q = 0.95` | **−0.1383** | [−0.1564, −0.1203] | ⛔ **LOSS** |
| ⚠ vs **`knn_factored` at its OWN BEST `q = 0.98`** | ⛔ **−0.1772** | — | ⛔ **LOSS** |
| vs `knn_whole` | −0.1005 | [−0.1106, −0.0904] | ⛔ LOSS |
| vs `margin_M` | **+0.1863** | [+0.1771, +0.1965] | ⭐ **WIN** |
| vs `CLU_K1@r8` | **+0.0857** | [+0.0828, +0.0887] | ⭐ **WIN** |

⚠ **Correction, found by adversarial verification and running AGAINST the arm.** `q = 0.95` *per column*
with the ANY-of-5 reduction gives the rival a **21.6 %** family-wise refusal rate on calibration, which
**handicaps it**. Its score peaks at **0.8621 at `q = 0.98`** (family-wise 9.4 %). ⛔ **The true gap at the
rival's own best operating point is Δ = −0.1772, not −0.1383** — the pre-registered comparison was fair
but **conservative in the arm's favour by 0.039**, and the corrected figure is the one to quote.

⛔ **The bar's pair leg is a LOSS, not a tie.** Control (d) is §5 below; control (f) is on every row above.
⛔ The §C.2 atom shuffle is **excluded** by §4 and was not run.

⭐ **The reduction I chose is the arm's BEST CASE, and that forecloses an objection.** The directive fixes
the *per-well* rule (`refuse well k iff E_k > 0`) but never says how the wells combine into one admission
decision. I used **"refuse iff ANY well refuses"**. Every other monotone reduction (majority, the
requested term's well only, all wells) refuses a **subset** of what this one refuses, so every
`refuse_out_of_support` number above is an **upper bound** over the whole family — and because
`accept_val = 1.0000` under this reduction, the usual cost of the strictest rule is **exactly zero** here.
⛔ **No choice of reduction improves the arm.**

⭐ **The asymmetry, stated rather than hidden.** The arm has **no fitted threshold**. Every rival was
calibrated at `q = 0.95` on 4,096 valid corners from `fit_rows` — ⛔ never `val`, ⛔ never B. The rival's
free parameter was fitted on data the arm never sees, which is the honest direction for the asymmetry.
⚠ The rival's realised refusal rate on valid data is ~21 %, not 5 %, because "refuse if **any** of 5 wells
exceeds its own `q = 0.95`" compounds. That operating point is what the pre-registered balanced score
rewards, and it is why §3's threshold-free comparison is reported beside it.

---

# 3. ⭐ THE RADIUS ATLAS — the mechanism, in one table

| radius, in the gate's own units `d/σ_k` | well 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| rivals' **fitted** `q = 0.95` | 1.150 | 1.138 | 1.053 | 1.122 | 1.285 |
| ⭐ **the derived ZERO LEVEL SET** (measured, 256 rays/well) | **2.093** | **2.043** | **2.419** | **2.343** | **1.892** |
| the certified bound `σ_k√(2 ln N)` | 2.884 | 2.884 | 2.884 | 2.884 | 2.884 |

| population | n | mean `min_k d/σ_k` | mean `max_k E_k` | refuse |
|---|---|---|---|---|
| stored atoms | 64 | 0.000 | −42.25 | 0.0000 |
| ⭐ **real features, all `store_train`** | 4096 | **0.787** | −38.10 | 0.0007 |
| ⭐ **real features, `val` rows** | 512 | **0.793** | −37.81 | 0.0000 |
| CLU launch `clamp_atom` (shipped) | 512 | **0.000** | −41.81 | 0.0000 |
| CLU launch `posterior_mean` | 512 | 0.814 | −54.34 | 0.0000 |
| R8 affine, valid | 512 | 0.775 | −44.41 | 0.0000 |
| R8 affine, A7 (valid) | 72 | 0.887 | −28.30 | 0.1944 |
| R8 affine, `out_of_range` w=2 → 13 | 3072 ea. | 0.856 → **2.262** | −41.0 → **+793** | 0.0000 → 0.7337 |

> ⭐⭐ **The data lives at 0.79 σ. The gate fires at 2.0–2.4 σ. The fitted rival fires at 1.05–1.29 σ,
> just above the data. The derived gate's radius is ~2.6× further out than the thing it guards.**

⭐ **The certified bound itself is correct and reasonably tight** — it sits 0.47–0.99 σ outside the true
crossing, as a *sufficient* condition should. ⛔ It is not the problem, and it is reported as §0 requires:
**certified fraction of refusals = 0.6936 (rung A), 0.7264 (rung B)**, ⛔ never as a second criterion.

## The capacity control: ⛔ more store does not fix it
| | rung A (N=64, 524,288 B) | rung B (N=512, **4,194,304 B**) |
|---|---|---|
| zero level set `/σ_k` | 1.87 – 2.42 | **2.44 – 2.89** (further out) |
| certified `/σ_k` | 2.884 | **3.532** (worse) |
| ⭐ real features at `d/σ_k` | 0.786 | **0.739** (unchanged) |
| `refuse_oor` | 0.3714 | 0.3995 |

⛔ **8× the atoms and 8× the bytes bought `+0.028` on `refuse_oor`.** Denser atoms ⇒ smaller `σ_k` ⇒ the
data still sits at ~0.75 `σ_k`.

## And the same control on the HEADLINE claim (`LIVESET_RUNGB_seed0.json`)
Two factors chosen **before running**, for a stated reason: `floor_hue` (the only singly-assigned factor
in the map, and where the rival scored a perfect 1.0000) and `wall_hue` (the rival's second best).

| factor | rung | N | store bytes | **CLU live-set separation** | `knn_factored` separation |
|---|---|---|---|---|---|
| floor_hue | A | 64 | 524,288 | **+0.0000** | +0.6641 |
| floor_hue | **B** | **512** | **4,194,304** | **+0.0000** | +0.7539 |
| wall_hue | A | 64 | 524,288 | **+0.0000** | +0.5117 |
| wall_hue | **B** | **512** | **4,194,304** | ⚠ **+0.1719** | +0.6523 |

⚠ **Two corrections I owe here, both against my own earlier wording.**
1. ⛔ `LIVESET_RUNGB_seed0.json` carries `"capacity_rescues_liveset": true`. **That flag is misleading**
   — it tests `separation ≥ 0.05`, the **§5.9 falsifier** threshold, ⛔ **not the §5 bar of `refuse ≥ 0.90`**.
   The artefact is left unaltered (§0.1) and this is the correction: ⛔ **capacity does NOT rescue the
   live-set claim.** `+0.1719` misses the bar by **0.73**, on **one factor of two**, for **8× the bytes**,
   while the rival reaches **1.0000** at both rungs for **no extra bytes**.
2. ⚠ **My earlier "there is no capacity fix" was too absolute, and this is the first non-zero separation
   the hub has measured.** Honestly: at rung B `σ` shrinks (well 0: 4.64 → 3.08), held-out queries move
   from `d/σ = 0.829` to **0.922**, and the closest query goes `max_k E_k = −7.10 → **+9.17**` — it
   **crosses**. The in-vocab control stays at 0.0000, so **the separation is genuine, not noise**.
   ⭐ **Capacity helps — far too slowly to matter: ~+0.09 in `d/σ` per 8× store**, while the level set
   itself moves *out* from 2.09 σ to 2.50 σ over the same step.
   ⭐ And `floor_hue` — the **better**-localised factor — stays at exactly **0.0000 at both rungs**
   (closest query `max_k E_k = −13.91`). ⛔ The effect is not where the map says the signal is.

---

# 4. ⭐ THRESHOLD-FREE — is the *signal* good, separately from the radius?

| statistic | AUROC `oor` vs `val` | AUROC `oor` vs **A7** | Δ over margin `M` |
|---|---|---|---|
| **CLU `max_k E_k`** | **0.8619** | 0.6680 | ⭐ **+0.4436** |
| `knn_factored` | **0.9196** | 0.7165 | +0.5013 |
| `knn_whole` | 0.8604 | 0.7084 | +0.4421 |
| `margin_M` (physics-free) | **0.4183** | 0.4452 | +0.0000 |

⭐⭐ **The cited paper's negative finding does NOT reproduce on this object.**
*Certified Access* reports that the residual *"adds at most **+0.024 AUROC** over a standard readout
margin"* and that the raw energy signal has *"negligible predictive value"* over it. Here the energy
signal adds **+0.4436**, and the readout margin is **below chance** (0.4183) — an out-of-range request
makes the address-book posterior *sharper*, not flatter, so `M` is actively misleading on this object.
⭐ A reportable disagreement with the cited work, on a different object, stated as such and not stretched.

⛔ **But the CLU statistic still trails the fair rival threshold-free: 0.8619 vs 0.9196 (−0.0577).**
So the DEV loss is ⛔ **not** merely an artefact of where the threshold sits. Confirmed directly: the §5
"fitted alternative, reported beside, never adopted" — the same `max_k E_k` thresholded at `q95(val)`
(= **−35.998**, against the derived **0.0**) — scores **0.7982**, still short of the rival's **0.8241**.
⛔ **The gate stays at `E_k > 0`.**

---

# 4b. ⭐⭐ THE LAST ESCAPE HATCH, CLOSED — the arm loses at **every** operating point

The fair objection to §2 is *"the pre-registered score rewards the rival's own choice of `q`, which parks
it at a 21 % refusal rate on valid data."* ⭐ The objection is answerable, and the answer does not help
the arm. Sweeping the rival's free parameter (`UNCERTAINTY_seed0_A.json`):

| | `accept_val` | `refuse_oor` | score |
|---|---|---|---|
| `knn_factored` q = 0.95 | 0.7891 | 0.8590 | 0.8241 |
| `knn_factored` q = **0.99** | 0.9648 | 0.7488 | **0.8568** ← the rival's best |
| `knn_factored` q = 1.0 | **1.0000** | **0.6443** | 0.8222 |
| `knn_whole` q = 0.999 | **1.0000** | 0.4861 | 0.7431 |
| ⭐ **the arm** (derived, **no free parameter**) | **1.0000** | ⛔ **0.3714** | 0.6857 |

**At MATCHED `accept_val = 1.0000`** (the rival's `q` solved by bisection): `knn_factored` reaches
`refuse_oor = 0.6449` and `knn_whole` 0.5188, against the arm's **0.3714**.

⛔⛔ **The arm loses at every operating point, not merely at the pre-registered one**, and even its own
statistic with a fitted threshold reaches only 0.7982. ⭐ **The negative is robust, not an artefact of
threshold placement** — which makes it worth more than a marginal pass would have been.

# 4c. ⭐ THE A7-CONTROLLED PAIR — ⛔ reported beside, **not** the bar

§7b asserts that a refusal number without its A7 companion *"overstates the rival and understates the
arm."* That is a claim about a number (`A7_PAIR_seed0_A.json`):

| arm | q | `accept_A7` | `accept_A7D` | `refuse_oor` | `score_A7` |
|---|---|---|---|---|---|
| ⭐ **CLU `gate@r8`** | — | **0.8056** | **1.0000** | 0.3714 | 0.5885 |
| `knn_factored` | 0.95 | **0.3889** | 0.9028 | 0.8590 | 0.6240 |
| `knn_factored` | **1.0** | ⛔ **0.5833** | 1.0000 | 0.6443 | 0.6138 |
| `knn_whole` | 0.9999 | 0.8750 | 1.0000 | 0.4804 | **0.6777** |

⛔ **`knn_factored` still falsely refuses 41.7 % of the valid A7 control at `q = 1.0`** — with its
threshold at the largest distance any valid calibration corner produced. A7's embeddings sit **outside
the entire valid distance range**, and no `q` recovers them. ⭐ `accept_A7D` is 0.90–1.00 everywhere, so
the A7 false-refusal is **entirely** R8's duplicated-term double-count (§6), for the rivals too.

R1 on the A7-controlled pair vs `knn_factored` at its pre-registered `q = 0.95`:
**Δ = −0.0355, CI95 = [−0.0908, +0.0225] → TIE.**
⚠ ⛔ **I am NOT banking this as a tie on the bar and it must not be quoted as one.** The bar's accept
population is the held-out **B**, not A7; this substitutes a different population, on DEV. Against the
best rival configuration the arm still loses (0.5885 vs 0.6777). ⭐ What it honestly shows: the arm's one
real advantage — it does not over-refuse valid redundant requests — is worth about **0.10 of score**, and
narrows the gap from −0.1383 to −0.0355. ⛔ Not enough.

---

# 5. ⭐ THE LIVE SET (control d) — the claim a threshold cannot fake, and it fails

Six stores rebuilt from `store_train ∖ rows(v)`, one held-out value per factor. ⭐ `shortfall = 0` on all
six, asserted: the cache is `scope="full"` (480,000 rows) so every row is already in it. ⛔ Nothing was
encoded. Every rebuild passes the identity and carries **0 atoms with the held-out value** (asserted).

| factor | wells | **CLU refuse(held-out)** | CLU refuse(in-vocab ctrl) | CLU write-back accept | `knn_factored` refuse(held-out) | `knn_factored` refuse(in-vocab) |
|---|---|---|---|---|---|---|
| floor_hue | [4] | **0.0000** | 0.0000 | 1.0000 | **1.0000** | 0.3359 |
| wall_hue | [0,1,2] | **0.0000** | 0.0000 | 1.0000 | **1.0000** | 0.5820 |
| object_hue | [2,3] | **0.0000** | 0.0000 | 1.0000 | 0.8438 | 0.5898 |
| scale | [2,3,4] | **0.0000** | 0.0000 | 1.0000 | 0.5664 | 0.6289 |
| shape | [2,3] | **0.0000** | 0.0000 | 1.0000 | 0.6484 | 0.6289 |
| orientation | [0,1] | **0.0000** | 0.0000 | 1.0000 | 0.3320 | 0.3398 |
| **mean** | | ⛔ **0.0000** (bar ≥ 0.90) | 0.0000 | ⚠ 1.0000 | **0.7318** | 0.5176 |

- ⛔ **Falsifier §5.9 fires on all six factors** (separation `+0.0000`, threshold 0.05). Exemplar refusal
  is dead. Symbol mode continues, as §5.9 directs.
- ⚠ **The write-back `1.0000` is NOT evidence for the arm** and must not be quoted as if it were: nothing
  is ever refused, so it is the same vacuum measured from the other side. Stated here so it cannot be
  lifted out of context.
- ⭐ **The fitted rival IS live-set sensitive where the derived gate is not** — the same statistic family,
  a tighter radius. That comparison is the whole finding.
- ⭐ ⚠ **And the rival pays for it exactly where §7b says:** `accept_A7` mean **0.3634**. The factored kNN
  gate **falsely refuses 64 % of the valid A7 control** — the programme-wide rule's premise, reproduced.

**Out-of-vocabulary, symbol** (the other half of `out_of_support`): ⛔ CLU refuse **0.0000** on all six.
⭐ And there is a structural reason worth the programme's attention: when a value is deleted, its R8
prototype has **0 training rows** and falls back to `μ_global`, so naming it symbolically embeds the
request **closer to the manifold's centre** (`d/σ = 0.71–0.82`, *below* the valid mean of 0.775 on four of
six factors). ⛔ **Deleting a value makes the symbolic request for it land MORE inside the store, not
less** — no distance-to-store criterion at any radius is reliable for OOV-symbol on this head.

---

# 6. ⭐⭐ THE ONE ACTIONABLE FINDING — the A7 false-refusal is R8's double-count, and it is fixable

| | `accept_A7` (7 terms) | `accept_A7D` (deduplicated, 6 terms) | Δ |
|---|---|---|---|
| **CLU `gate@r8`** | 0.8056 (`d/σ` 0.887) | ⭐ **1.0000** (`d/σ` 0.780) | **+0.1944** |
| `knn_factored` `q = 0.95` | 0.3750 | **0.8750** | **+0.5000** |

⭐⭐ **Every one of the CLU's A7 false refusals disappears under `gen_request.deduplicate_terms`, and half
of the factored kNN's do.** The cause is R8's declared composition rule
`x̂ = μ_global + Σ_terms (μ_{k,v} − μ_global)`: a **duplicated term contributes its deviation twice**, so
a perfectly valid redundant request is pushed off-manifold by an extra `(μ_{k,v} − μ_global)`.
⭐ **This is a concrete, measured cause for the phenomenon the programme-wide §7b rule exists to expose**
(*"three of five published rivals falsely refuse ~60 % of the valid A7 redundant-term control"*). The
rivals share the R8 head, inherit the same double-count, and their tighter radius amplifies it.
⛔ `gen_request.py` is manifest-pinned and **not mine to change** — raised for the coordinator, with the
measurement attached. ⭐ This is, in my judgement, worth more to the programme than the refusal row would
have been.

---

# 5b. ⛔ ADVERSARIAL VERIFICATION — nothing overturned, one bug of mine, two corrections against the arm

A second Opus sub-agent was tasked to **break this negative**, recomputing every gate number **from the
PREREG-1 §1 formula in float64 numpy** without calling my code (`VERIFY_ADMISSION.md`, four scripts, four
JSONs). ⭐ **Nothing was overturned.**

## ⛔ A BUG IN MY CODE, STATED PLAINLY — the §5.6 check was VACUOUS
`u4_liveset.py:99` computes `shortfall = setdiff1d(pool[keep], pool) + setdiff1d(pool[held_mask], pool)`.
⛔ **Both terms are empty by construction; the assertion could not fail**, and `F5_6_fires` was hard-wired
`False`. **I reported a passing check that was incapable of failing.**
⭐ **The conclusion survives by an independent route the verifier supplied:** `build_landscape` receives
`X_pool[keep]`, a slice of what `gather_features` **already returned**, so `gather_features` is never
re-entered and no uncached row can be requested. **`shortfall = 0` is true. My test for it was worthless**,
and §9 below now records §5.6 as established *by construction*, ⛔ not by my assertion.
*(Also found: `per_well_nearest` uses the float32 Gram form, `max |Δ|` vs float64 = 0.011 on `d ≈ 4–10`,
with **0 accept/refuse disagreements** over 1024 probes — so `d/σ_k` is nowhere quoted beyond 3 dp.)*

## ⛔ A LAUNCH POINT I MISSED — the SETTLED state. It is not vacuous, and it does not rescue anything.
`settle_chain` at the shipped `SettleCfg` (1000 steps) from the shipped `clamp_atom` launch is shipped,
request-only, non-fitted, gate at zero, `σ_k` unscaled — and I did not test it.

| gated at the SETTLED state | refuse |
|---|---|
| **valid `val`** | ⛔ **0.1094** |
| `out_of_range` w = 2 / 3 / 4 / 7 / 13 | 0.0000 / 0.0000 / 0.0234 / 0.1641 / 0.3359 |

⛔ **It refuses MORE valid requests than invalid ones at w = 2, 3 and 4** — strictly dominated by
`gate@r8_affine`, and not an arm. Live-set separation from my own hybrid on my own rebuilds is
**−0.0391** (floor_hue) and **−0.0469** (wall_hue): **negative**. ⭐ **The conclusion stands — there is no
launch point at which `E_k > 0` discriminates a held-out value.** ⚠ But *"the derived gate never fires"*
is too strong as a sentence about the object, and is not used anywhere in this report.

## ⭐⭐ THE MECHANISM THE VERIFIER FOUND — a shipped-code fact, for the coordinator
`total_energy = E_store + β·Σ_t (E_t − E_store)`, so at `β = 1` with `T` requested terms **the store's own
free energy enters the settling dynamics with weight `1 − T`**. For a six-factor `shapes3d` request that
is **−5**: ⛔ **settling pushes the state AWAY from the atoms by construction.** Measured on valid `val`,
`mean max_k E_k` rises **−36.9 → −24.7** and `min_k d/σ_k` **0.000 → 0.849** over 1000 steps. ⭐ That is a
defect in the **read**, not in the gate, and it is ⛔ not my file to change.

## ⭐ CONFIRMED against the verifier's own independent float64 numbers
Live-set 0.0000 on all six factors (closest approach `max_k E_k = −9.88`) · hold-out real (0 leaking
atoms; rebuilds share 0–1 of 64 atoms with the full store) · hybrid bit-exact in exactly the assigned
wells · manifold **0.7852 / 0.7934** · level set **2.090 / 2.038 / 2.410 / 2.342 / 1.901** · `σ` to
**4.5e-08** · ⛔ no sign, `√` or `2σ²` error · float16 vs float32 identical to 16 digits · A7 **airtight**
(`embed(A7) − embed(A7D)` equals exactly one `(μ_kv − μ_global)` to 2.98e-07) · `paired_bootstrap`
Δ = −0.13834635 exact.
⭐ **The strongest variant I had under-tested:** the **full unmodified 2048-dim cached feature** of a
held-out image against the **rebuilt** store refuses **0.0007** — *less* than the same images against the
**undamaged** store (**0.0013**), both at the generic real-feature false-refusal floor. **My claim was not
too strong.**
⭐ **And the obvious objection to random rays cuts my way:** the crossing along a ray *toward a real
feature* — the data direction — is **2.04–2.41 σ**, i.e. **no closer**.

## ⭐⭐ THE VERIFIER'S OWN NEW FINDING — the best thing in this run for re-ideation
Operating-point-free (max over **all** thresholds of `½(refuse_oor + accept_val)`), same store, same wells:

| statistic | best score | AUROC |
|---|---|---|
| CLU `max_k E_k` — **the free energy** (the arm) | 0.7994 | 0.8619 |
| ⭐ CLU `max_k d_k/σ_k` — **plain nearest-atom distance** | ⭐ **0.8488** | ⭐ **0.9107** |
| `knn_factored` (the fitted rival, at its best) | 0.8621 | 0.9198 |

⭐⭐ **The `logsumexp` over 64 atoms DESTROYS separation relative to plain nearest-atom distance in the
same wells.** The CLU's own distance is a **better** statistic than the free energy built from it, and
lands within **0.014 AUROC** of the fitted rival.
⭐ **Loss decomposition: of the arm's 0.177 deficit, 0.114 is the RADIUS and 0.063 is the SIGNAL.**

---

# 6b. ⭐ THE COHERENT CHARACTERISATION — a property, ⛔ not a metric chosen after the fact

Every valid-request population this hub could construct, at the same thresholds:

| valid population | ⭐ CLU `gate@r8` accept | `knn_factored` accept |
|---|---|---|
| `val` corners | **1.0000** | 0.7891 |
| `unseen_value` (interpolated between grid values) | **1.0000** | 0.7907 |
| A7 (redundant term) | 0.8056 → ⭐ **1.0000** deduplicated | 0.3889 → 0.9028 |
| **free mode** — 1 / 2 / 3 factors left unnamed | ⭐ **1.0000 / 1.0000 / 1.0000** | ⛔ 0.5488 / 0.3633 / 0.2168 |
| write-back after deletion | 1.0000 | — |

> ⭐ **The derived gate never falsely refuses a valid request, in any mode this hub could construct.**
> ⛔ **And it refuses only 0.2653 of the invalid ones.** Both facts have the same cause — its radius sits
> outside everything. It is an excellent *"do not falsely refuse"* gate and a poor *"refuse the invalid"*
> gate, and the fitted rivals are the exact mirror image.

⛔⛔ **This does not rescue the bar and is not offered as if it did.** The pre-registered score is
**balanced**, and on a balanced score the arm **loses — at every operating point, on the pooled
definition, against both rivals.** It is stated because it is the honest characterisation of the object,
and because it is the one place a future design could start.

---

# 6c. ⭐ THE CERTIFIED-ACCESS FIGURE (§3 step 6, DEV, unscored)

`certified_access_seed0_A.png` · `CERTIFIED_ACCESS_FIGURE_seed0_A.json` · `CERTIFIED_ACCESS_NOTES.md`
192 `val` rows, each walked outward from its **own cached feature** along one fixed unit direction
(seed 20260917), 22 radii, `min_k d_k/σ_k` sweeping 0.81 → 5.05. ⛔ Nothing re-encoded; ⛔ B never read.
⭐ **The decode leg ran**: the shipped `exp_gen_baselines.load_shared_decoder` (gen-decoder-lift winner
`v1_capacity_schedule`, hash-asserted) and the shipped `exp_gen_cell.load_external_classifier`
(`c8741ac4…`), scored by `gen_probe.probe_accuracy` over the six factors. Chance = 0.1236.

| marker | `min_k d_k/σ_k` | plain-decode accuracy there |
|---|---|---|
| kNN factored, q = 0.95 | 1.00 | 0.975 |
| kNN whole-latent, q = 0.95 | 1.19 | 0.946 |
| **zero level set** (`max_k E_k` crosses 0) | **1.55** | 0.879 |
| ⭐ **certificate flips** (`d_k ≥ σ_k√(2 ln N)`) | **2.06** | **0.800** |
| ⛔ **plain decode collapses** (midpoint to chance) | **4.08** | 0.558 |

> ⭐ **The certificate flips 2.02 units EARLIER than decode quality collapses** — the gate refuses while
> the decoder is still producing 80 %-accurate output. ⭐ Combined with §6b this is a coherent
> characterisation: **on-manifold the gate accepts everything; off-manifold it refuses well before
> quality dies.** It is conservative in the safety direction and blind in the validity direction.

⚠ Two honest riders. (i) The crossing is at **1.55** here versus **1.89–2.42** in §3 because these rays
start from a **`val` row** (already at 0.79 σ) while §3's start from a **stored atom** (at 0). Both are
correct and they are the same surface. (ii) ⛔ **The CLU read is flat at 0.3507 at every radius** and the
request-off control is flat at 0.1198 (≈ chance): on this rung the settled state is fixed by the request
term, **not** by where the chain was launched. So gating the launch point does not gate the read, and the
*"one object, two outputs"* property is **not** demonstrated here. ⛔ No claim beyond the figure: one seed,
one rung, one direction per row, 192 rows, DEV.

---

# 7. LOCALISATION

Pre-registered rule (PREREG-1 §7): *the requested term with the largest residual among uncertified wells.*
Chance = 1/6 = 0.1667.

| arm | accuracy | per factor |
|---|---|---|
| **`CLU_gate@r8`** | **0.3441** | floor_hue 0.098 · wall_hue 0.217 · object_hue 0.116 · scale 0.560 · shape 0.444 · orientation 0.630 |
| `CLU_K1@r8` (single-well control) | **0.1146** | at chance on five of six factors |

- ⛔ **Bar ≥ 0.50: FAILED at 0.3441.** ⭐ **Beats the single-well control by +0.2295**, within-seed paired
  bootstrap CI95 **[+0.2222, +0.2367]** — that leg passes, and it is the one place the *design* (a channel
  partition) demonstrably buys something.
- ⚠ ⛔ **"Beyond seed spread" could not be tested and I will not claim it.** Seeds 1 and 2 have caches with
  `scope="used"` (9,288 rows); covering the cell needs ~32,200 new rows, which is a re-encode and is
  forbidden. **No seed-to-seed spread exists on this box.** The interval above is a *within-seed*
  bootstrap over the 15,360 requests and bounds sampling noise only.
- ⛔ **And the rule is anti-correlated with refusal**, which is the interesting part: by severity,
  accuracy on **refused** rows is 0.05 / 0.20 / 0.22 / 0.16, while on **accepted** rows it reaches
  **0.637 (w=7)** and **0.736 (w=13)**. Once a point is refused it is ≥ 2 `σ_k` from every atom in every
  well, so all wells are uncertified and every term residual saturates — the `argmax` stops carrying
  information. ⛔ The rule was fixed in PREREG-1 and is **not** re-specified after seeing this.

---

# 8. THE FACTOR→WELL MAP (§3 step 1)

`feat_shape = (128, 4, 4)` ⇒ the 2048 latent dims are **128 feature maps × 4×4 spatial**, ⛔ not pooled.

⚠ **`K = 5`, not the 32 the addendum names.** The shipped store plan for `shapes3d`/`recon` gives
`plan["K"] = 5`; `well_channels` sizes `[303, 305, 573, 501, 366]`, `ward`, 71 dead channels, and the
re-derivation **reproduces the plan bit-identically**. ⭐ **Question for the coordinator: where does
"K = 32" come from?** I proceeded on the shipped K = 5 object and every number above says so.

MI(nearest-atom index ; factor) over 32 shuffled-label floors, normalised by `H(f)`, all 32,768 rows:

| well | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| 0 | 0.2339 | 0.8893 | 0.0333 | 0.0320 | 0.0773 | 0.1985 |
| 1 | 0.1689 | **0.8992** | 0.0463 | 0.0303 | 0.0717 | **0.2555** |
| 2 | 0.1882 | 0.6274 | 0.2959 | 0.0625 | 0.1707 | 0.1064 |
| 3 | 0.4540 | 0.0790 | **0.5547** | **0.1115** | **0.3249** | 0.0057 |
| 4 | **0.9462** | 0.1013 | 0.2233 | 0.0753 | 0.1136 | 0.0149 |

⚠ **Partial by construction**: 6 factors onto 5 wells cannot be injective, and only 3 distinct argmax
wells are used. Only **`floor_hue` is singly-assigned** (well 4). ⛔ **`scale` is barely carried anywhere**
(max 0.1115). ⇒ exemplar mode is well-posed for `floor_hue`, workable for `object_hue`/`shape`,
⛔ ill-posed for `scale` and `orientation`. A measured limit of the object, reported not worked around.

---

# 9. FALSIFIERS — every one answered with a number

| | verdict | number |
|---|---|---|
| **§5.6** shortfall on a rebuild | ⭐ **does not fire** — ⚠ but established **by construction and by the verifier's independent route**, ⛔ **NOT** by my assertion, which was vacuous (§5b) | `shortfall = 0` on all six; `gather_features` is never re-entered |
| **§5.7** the gate does not calibrate | ⭐ **does not fire** | `accept_val = 1.0000` ≫ 0.50 — it over-**accepts** |
| **§5.8** no design advantage | ⚠ **partial** | the rival **wins** rather than ties (Δ = −0.1383); but localisation **does** beat the single-well control (0.3441 vs 0.1146). The literal conjunction fails; the spirit — *"admission is kNN with extra steps"* — is worse than it, because here it is kNN with extra steps **and** a worse radius. |
| **§5.9** the encoder does not separate | ⛔ **FIRES, all six factors** | separation `+0.0000` |
| **§5.10** affine head crosses | ⭐ **does not fire** at `gate@r8` | `max_k E_k`: −34 → +2060 over `w = 2 → 13`, monotone on all six factors |
| **§5.10** at the shipped `clamp_atom` launch | ⛔ **FIRES, by algebra** | `d = 0` exactly ⇒ `refuse ≡ 0` |
| **§5.10** at `posterior_mean` | ⛔ **FIRES** | `accept = 1.0000` at every severity |

---

# 10. THE BAR — ⛔ set by the task file, never moved

| leg | bar | measured | verdict |
|---|---|---|---|
| pair vs factored kNN under R1 | win-or-tie | Δ = **−0.1383** [−0.1564, −0.1203]; **−0.1502** pooled; ⛔ **−0.1772** at the rival's best `q` | ⛔ **FAIL** |
| `accept_A7 ≥ accept − 0.05` | gap ≤ 0.05 | **0.194** | ⛔ **FAIL** (⭐ → 0.000 deduplicated) |
| localisation ≥ 0.50 | ≥ 0.50 | **0.3441** | ⛔ **FAIL** |
| localisation beats single-well | yes | 0.3441 vs 0.1146 | ⭐ **PASS** |
| live-set refuse ≥ 0.90 | ≥ 0.90 | ⛔ **0.0000** (rung A, all six); best over both rungs **0.1719** (`wall_hue`, rung B, 8× bytes) | ⛔ **FAIL** |
| live-set write-back ≥ 0.90 | ≥ 0.90 | 1.0000 | ⚠ **PASS, vacuously** |
| no tuning | gate is `E_k > 0` | ⭐ it is | ⭐ **PASS** |

---

# 11. WHAT I WOULD TELL THE COORDINATOR

1. ⛔ **The admission target as posed is not reachable on this object without changing something §7
   forbids me to change.** The gate's radius is set by `σ_k = nn_median` and `√(2 log N)`, and both push
   it outside the data. That is a **design** question (a bandwidth rule), not an execution question, and
   it belongs to re-ideation, not to me. I have not touched either.
   ⭐ **The structural statement, which is the useful form of it:** `σ_k` is the atoms' own
   nearest-neighbour spacing, and the quantity the gate must discriminate — a real feature's distance to
   the nearest atom — **is that same spacing**, because the atoms are drawn from the data. ⛔ **The
   bandwidth rule and the signal it is meant to separate are the same number**, so the zero level set can
   only ever sit at a fixed multiple of the data's own radius, whatever `N` or the byte budget. Any
   derived gate on this store needs its radius fixed by something that is **not** the atom spacing.
   *(Arithmetic, for scale only and ⛔ not a recommendation: the level set at 2.0–2.4 σ would have to come
   in by ~2× to sit where the fitted `q = 0.95` does. ⛔ Choosing that factor would be fitting, which is
   exactly what §7 forbids and what the whole claim rests on not doing.)*
2. ⭐ **The A7 double-count (§6) is a concrete, one-location defect with a shipped fix already in the
   codebase** (`deduplicate_terms`). It explains the published rivals' A7 false-refusal rate that §7b was
   written for. I recommend it be looked at first; it is cheap and it improves *every* arm's A7 row.
3. ⚠ **`K = 32` does not exist on this object** (§8). Whatever was intended, the shipped plan gives 5.
4. ⭐ **`clamp_atom` makes any store-energy gate vacuous** (§1). Any future admission design must gate a
   point *before* it is snapped to atoms — this is worth writing into the next directive.
5. ⚠ **The settled read does not depend on its launch point** (§6c): at 1000 steps the CLU read is flat
   at 0.3507 across a 6× sweep of launch radius, and the request-off control is at chance. ⛔ So gating
   the launch cannot gate the output, and the claim's property 4 — *"one object, two outputs"* — has no
   support on this rung. This is a fact about the **read**, not the gate, and other hubs may want it.
6. ⭐⭐ **The `logsumexp` is costing separation.** Operating-point-free, the CLU's own per-well *distance*
   `max_k d_k/σ_k` beats the *free energy* `max_k E_k` built from it (AUROC **0.9107 vs 0.8619**; best
   score 0.8488 vs 0.7994) and lands within **0.014 AUROC** of the fitted rival. ⭐ Of the arm's 0.177
   deficit, **0.114 is the radius and 0.063 is the signal.** If admission is re-posed, the free-energy
   form is costing it — and the min over atoms, which is already in the same wells, is cheaper and better.
7. ⚠ **The settled read pushes itself off the store.** `total_energy = E_store + β·Σ_t(E_t − E_store)`
   gives `E_store` weight `1 − T` = **−5** for a six-factor request. Measured: over 1000 steps a valid
   request's `min_k d/σ_k` rises **0.000 → 0.849** and `max_k E_k` **−36.9 → −24.7**. ⛔ **Settling moves
   the state away from the atoms by construction.** A defect in the **read**; ⛔ not my file.
8. ⭐ **The fitted rivals over-refuse valid requests catastrophically** — `knn_factored` accepts only
   0.2168 of a perfectly valid request with three factors left free, and 0.3889 of A7. ⚠ **Any published
   refusal table using a factored kNN gate at `q = 0.95` should be checked against free-mode and
   redundant-term requests before it ships**; the §7b rule catches the A7 half of this, not the free half.
9. ⭐ **0/3 scored runs spent.** The target still has its full budget if re-ideation produces a gate whose
   radius sits where the data is.

---

# 12. ELAPSED, COST, AND THE HONEST VERDICT

| | |
|---|---|
| Compute | ≈ **35 min** of my own local CPU across 10 units, 4 threads, plus 2 Opus sub-agents inside the same share · ⛔ no GPU · ⛔ no GCloud spend (**£0**) · CSF3 **NOT RUN** (2FA; nothing here needed it, so no submit line was required) |
| **Scored runs** | ⭐ **0 / 3 spent — three remain. `eval_b` never read, never embedded, never scored.** |
| **Elapsed true wall clock** | **1.04 h** of the 12 h allowance, measured by `elapsed.sh` against `START_UTC.txt` and the real clock. ⛔ Not model-perceived time. |

⚠ **Why I stopped at 1.04 h and did not use the remaining 11 h.** Every mandated deliverable is
delivered, all five falsifiers are answered with numbers at both rungs, all six controls are run, the
result has been adversarially re-derived in float64 by a second agent, and the escape hatches (operating
point, pooled population, well-reduction, capacity, launch point, fitted threshold) are closed with
measurements rather than argument. **The only remaining moves are the ones §7 forbids** — moving the gate
off zero, scaling `σ_k`, or swapping the level set for a quantile. ⛔ Spending the remaining hours would
have produced padding, and the deadline is a cap, not a quota. ⭐ The one thing 11 more hours could buy —
a seed-1/seed-2 replication — is **unreachable on this box**, because those caches are `scope="used"` and
covering them is a re-encode.

## ⛔ **NOT SOLVED — and the reason is measured, not asserted.**
⭐ Nothing here was manufactured to fill the window. The bar was missed, the scored run was not spent,
the write-back pass is labelled vacuous rather than banked, and the one leg that passes (localisation over
the single-well control) is reported at its true size and no larger.
