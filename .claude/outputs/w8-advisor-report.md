# WAVE 8 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-21. Three spokes + two hubs, FORGIS_14,
`base/iter7 = ec2ef7c` (pin + 69 additive files, 0 deletions, frozen harness byte-identical).**
Launch 2026-09-20 16:36Z; all five finished. **Zero GPU spent.** Record: `.claude/outputs/_thinkpad/w8/`.
⚠ Every number is quoted from an agent's own report. ⛔ **The Hub has re-derived no statistic.**

---

## 0. The scoreboard

| agent | verdict | scored |
|---|---|---|
| ⭐ `spoke/downdate-build` | **exact deletion WORKS** — downdate == refit, 9/9 seeds | — |
| ⛔ `hub/single-design` | **the two properties are INCOMPATIBLE AT MATCHED BYTES** — and why | **0/3, intact** |
| ⛔ `hub/encoder-loop` | **NO — the store does not force better features** | 1/3 |
| ⚠ `spoke/novelty-scout` | **ADJACENT** — one claim stopped; exact deletion **unclaimed** | — |
| ⭐ `spoke/env-scout` | **CMAZE-1 adopted** — the first environment to pass its gates | — |

**Budget after wave 8:** `single-design` **3/3 intact** · `encoder-loop` 2 left · `generation-table`
run 3 **still unspent** · deletion 2/3 · composition 2/3.

---

## 1. ⛔⛔ `hub/single-design` — the answer, with the mechanism

> **The two properties are incompatible at matched bytes. Not because the deletion mechanism fails —
> it works exactly — and not because of the partition criterion, which turns out to be free.
> Because the byte budget forces a granularity at which the read stops composing.**

Each link measured:
1. ⭐ **Exact per-item deletion works.** Retain `S_k = Σxxᵀ`, `t_k = Σx`, `n_k`; atoms are the leading
   eigenvectors; delete by rank-1 subtract. **Downdate == refit-without-those-rows to 1e-16…1e-14**,
   ranks agreeing in every well, **0 rows refitted, 0 passes over the data, items never kept**,
   **9/9 across three seeds** on generation 2. ⭐ Independently corroborated by the spoke on its own
   leg (1.164e-10) **with different code**.
2. **The covariance costs `Σ_k d_k²`.** Fitting it inside the 524,288 B pin **forces K ≥ 36** — below
   that the statistic alone exceeds the budget with **zero atoms left for content**.
3. ⛔ **At K ≥ 36 composition collapses.** K=40/A=7, honest four-part ledger **519,794 B = 0.991×**:
   valid mean **0.0648**.
4. ⭐ **At K = 5 composition holds — 0.8918 against the incumbent's 0.8941 — but the honest ledger is
   7.48×** (label-free cut) or 24.5× (factor partition). The byte pin is primary at tolerance 0, so
   ⛔ that object cannot carry a RESULT row.

⭐⭐ **Scored 0/3, no ledger row written, full budget intact.** The object was falsified on `val`
before a single irreplaceable run was spent. **This is exactly what the val condition exists to do.**

⭐ **The unscored val observation worth carrying (explicitly NOT a RESULT row):** the
`ward_balanced K=5` object composes at **0.8918** (incumbent 0.8941), **deletes exactly**, and uses
⭐⭐ **NO labels in its partition** — a label-free correlation-dendrogram cut. ⚠ val, seed 0,
24 corners, one arm, no learned-baseline comparison.
⭐ Closing the ALS confound also showed the **label-free partition is 1.6× cheaper to make
downdatable** than the label-MI one — so label-free wins on two axes at once.

## 2. ⛔ `hub/encoder-loop` — no, and the decomposition is the finding
> **An encoder trained WITH THE STORE IN THE LOOP does not produce better features than an encoder
> trained ALONE, at equal compute.** BAR B, written before any number existed, **FIRED on row 3**.
> Row 4 resolved nothing — deltas straddle zero, between-seed spread exceeds the effect.
⛔ **The claim the programme has carried — that the store shapes the representation — is not supported.**

| column | isolates | row 4 | row 3 |
|---|---|---|---|
| **A′** `loop_nosw − alone` | unfreezing **without** the swap loss | +0.65 / **−4.84** (p=.0013) / −3.45 | **−5.97** (p=.0000) / −2.62 / −2.28 |
| **A″** `loop_swap − loop_nosw` | what the **swap loss** bought | −1.37 / **+5.69** (p=.0003) / **+7.39** (p=.0002) | −0.80 / **+2.37** / **+3.98** |
| **A** net | | −0.71 / +0.86 / +3.94 | −6.77 / −0.25 / +1.70 |

1. ⭐ **Unfreezing with the store on the gradient path HURTS** — 5/6 seed-rows negative, two at
   p ≤ 0.0013. ⭐ Independently replicates the programme's standing finding, now **with a control at
   identical steps from a bit-identical warm start**.
2. ⭐⭐ **The swap loss is a real ingredient and it REPAIRS that damage** — 4/6 seed-rows positive,
   two at p ≤ 0.0003. ⇒ **The net is a wash because two real effects cancel**, which is a very
   different statement from "nothing happened", and the paper should say the decomposed version.

⭐ **`kill -9` resumability PROVED** before any spend: killed mid-training, the resumed leg reached
**bit-identical parameters**, `resumed_from_step = 2`, and a foreign config fingerprint was **refused
by name**. ⭐ **Cost: 7 h 22 m, local CPU, zero GPU.** A launch spec is ready; the GPU leg answers a
**secondary** bar only — ⛔ BAR A's verdict does not depend on it, since both sides share the renderer.

## 3. ⚠ `spoke/novelty-scout` — one claim stopped, the strongest one survives
⛔ **STOP, as phrased:** *"multilinear/bilinear factored generation with a request-indexed tensor
head"* is **Tenenbaum & Freeman (2000)** and **TensorFaces (2002)**, re-run by Vlasic (2005) and
MLC-VAE (ICML 2020). ⛔ Also do not claim tensor-product label conditioning (MLC-VAE), nor the
**covariance-downdate primitive** (Cao & Yang 2015 owns the exactness in one line).
⭐⭐ **But: ~4,050 works screened in three directions; highest atom count reached by any single work
is 4 of 6; zero works combine frozen-decoder + tensor-head + label-index; six orthogonal sweeps
return zero. NONE does exact deletion on this object.**
⭐ **And a correction that gives ground back:** Hall–Marshall–Martin is **explicitly approximate** in
its own words (*"tends to instability"*, error analysis *"anecdotal in character"*) because it
subtracts **deflated** eigenspaces. ⛔ Do not concede exactness to the decremental-PCA line — it does
not have it. ⭐ *"Downdate the statistic, not the truncated factors"* is a sentence the paper should carry.
**Surviving claims, ranked by the scout:** 1) exact deletion of a compositional generative model;
2) the interface split; 3) wells — *"claim the use, not the idea"*.

## 4. ⭐ `spoke/env-scout` — the first environment to pass
**CMAZE-1 adopted** (adversarial placement, no acceptance filter, n=500, exact judge, no tolerance):
ORACLE **1.0000** · BLIND-B 0.0000 · BLIND-COLOUR 0.0840 · BLIND-A 0.1280 · BLIND-FACTOR 0.2600 ·
BLIND-SEQUENTIAL **0.4980** · RANDOM 0.0000 · ⭐⭐ **FRAGMENT-LOOKUP r=1 and r=2 both 0.0000.**
⭐ **Rule B bends the route in 81.6 % of episodes** (+14.8 steps; 42.4 vs 27.6 optimal) — the two-gym
law is satisfied for the first time. ⭐ It rejected MiniGrid **from the generator code**: DoorKey is
**1.0000 composition-blind**. ⚠ The uniform unfiltered generator's **0.9100** stays printed.

---

## 5. What the Advisor is asked to rule on
1. ⭐⭐ **The compose/forget question now has a quantitative answer, not a stalemate:** both properties
   hold in one object **at 7.48× the byte pin**, and fail together at 1.0×, because the covariance
   forces K ≥ 36 and composition dies above K ≈ 5. ⛔ **Is the honest framing "a budget result", and
   may the paper state the crossover?**
2. ⭐ **Exact deletion is confirmed and unclaimed in the literature** — two independent
   implementations, 9/9 seeds, 1e-16…1e-14. ⭐ This is the strongest claim the programme owns.
3. ⛔ **The store-shapes-representation claim is dead** — but the swap loss is real and repairs what
   unfreezing breaks. **Does that decomposition earn a paragraph?**
4. ⚠ **`generation-table` run 3 is still unspent** and `single-design` is 3/3 intact. **What are they
   for now?** Wave 9's maze has its own budget.
5. ⭐ **CMAZE-1 is ready.** ⛔ Wave 9 inherits the 0.4980 blind-sequential column as a caveat.

## 6. Limits
Everything 3 seeds on generation 2 · the Hub re-derived nothing · `single-design`'s key numbers are
**val, not held-out** — by design, and it spent nothing · `encoder-loop`'s decoder is shortened, so
BAR D is open and the GPU leg (**$26.46**, spec ready, `kill -9` proved) would close it ·
⚠ `single-design` logged against itself that it twice ran detached work under a standing instruction
not to — both polled to completion, neither stranded, **and its own fix is "size work before
launching, not poll better."**
