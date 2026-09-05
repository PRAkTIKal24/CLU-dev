# PREREG — headroom-retry-benchmark (w24)

**Written BEFORE any measurement harness was run on the new regimes.** Base: local `main @ 5e466c0`;
branch `agent/experiment-engineer/headroom-retry-benchmark`; worktree `../CHLU-headroom-retry`.
Author: experiment-engineer. Timestamp: start of the w24 task, before implementing `block_query` /
`select_store(crowded)` and before the Item-2 headroom gate was executed.

---

## 0. What is being predicted

The task's acceptance criterion is a **measured verdict** ("does a regime exist where CLU-gated retry
beats the feedforward-NN floor at matched compute?") plus two measured quantities per regime (the
**first-pass accuracy** and the **NN floor**). Both are pre-registered below with numbers.

## 1. The regimes I will build (fixed here, not tuned later)

| id | store | query | levels swept | ambiguity source |
|---|---|---|---|---|
| **R-BLOCK** | iid (`pool[:M]`) | **contiguous square occlusion**, area fraction `f`, random position per pattern, survivors scaled `1/(1-f)` (torch.dropout convention, inherited) | `f ∈ {0.4, 0.5, 0.6, 0.7}` (gate picks 2) | *correlated erasure* — surviving evidence is a contiguous crop consistent with several stored digits |
| **R-CROWD** | **crowded**: the `M` mutual-nearest patterns of an anchor in the pool (a tight cluster, so median-NN spacing collapses) | iid dropout mask `p` (the w22/UHop protocol, unchanged) | `p ∈ {0.2, 0.3, 0.5}` (gate picks 2) | *geometry* — inter-pattern spacing falls below `Δ_req ≈ 3.1·max(s, σ_q)`, so basins genuinely overlap |

Loads `M ∈ {128, 256}` in both. Everything else (six lines, ladder `k∈{0,1,2,4,8}`, `τ=0.99`,
`retry_boost=1.5`, `retry_step_frac=0.1`, `clu_steps=150`, `s=0.3·median-NN`, the
"generous-to-the-baselines" compute placement) is **inherited unchanged** from `retry-compute-study`.

Explicitly NOT used as the headroom lever: **full-field Gaussian noise** (the w23 σ≥0.4 cliff —
ambiguity, not destruction, per the task's ⭐ design constraint).

## 2. Pre-registered numbers — the Item-2 headroom gate

Gate = *first-pass CLU accuracy ∈ [0.5, 0.75]* **AND** *feedforward-NN floor NOT ≥0.95*.
Anchor for calibration: w23 iid-mask `M=128, p=0.5` gave **first 0.570 (already in band) but NN 0.996**
⇒ **the binding constraint is the NN floor, not first-pass.**

| prediction | quantity | registered value |
|---|---|---|
| **G1** | NN floor, R-BLOCK `f=0.5`, M=128 | **0.85 ± 0.10** (off ceiling, gate PASSES on the NN half) |
| **G2** | NN floor, R-BLOCK `f=0.7`, M=128 | **0.55 ± 0.15** |
| **G3** | first-pass CLU, R-BLOCK `f=0.5`, M=128 | **0.40 ± 0.15** (below the [0.5,0.75] band ⇒ I expect to need `f=0.4` for the *first-pass* half of the gate) |
| **G4** | NN floor, R-CROWD `p=0.3`, M=128 | **0.55 ± 0.20** (crowding is the stronger NN-killer of the two) |
| **G5** | first-pass CLU, R-CROWD `p=0.3`, M=128 | **0.35 ± 0.20** |
| **G6** | at least one (regime, M, level) cell passes BOTH halves of the gate | **YES**, and I expect it to be an **R-CROWD** cell at mild mask (`p=0.2–0.3`) |
| **G7** | measured packing slack `median_NN / (3.1·max(s, σ_q))` in R-CROWD | **< 1.0** (i.e. past the packing bound; w23 iid ran at ≡1.08) |

## 3. Pre-registered numbers — Item 4, the verdict that matters

**Primary hypothesis H-A (my honest prior, ~70%): NO — the feedforward-NN floor still dominates
CLU-gated retry in every ambiguity cell (gap ≤ 0 pp everywhere).**

*Derivation (why I predict this).* Under masked/occluded-pixel degradation with an iid-pixel corruption
model, `argmin_i ‖q − ξ_i‖²` over the store is (up to the survivor rescaling) the **maximum-likelihood
identity decision given the observed evidence** — it already uses all surviving evidence optimally. The
CLU settle is a *noisy, dynamical approximation to the same computation* (the Gaussian memory potential's
wells are placed at the same `ξ_i` under the same Euclidean metric). Making the evidence ambiguous removes
information from **both** decision rules by the same amount; it cannot open a gap in CLU's favour, because
the boost's aim point **is the ambiguous query**. Retry can only re-rank among basins already consistent
with `q` — exactly the set NN is ranking. ⇒ ambiguity buys **headroom** (both lines fall) but not
**dominance**. Predicted best gap: **−5 to −25 pp** in R-BLOCK, **−3 to −20 pp** in R-CROWD.

**Alternative hypothesis H-B (~30%): YES, in R-CROWD at high load.** The one asymmetry between the two
decision rules is that CLU's well width is **store-adaptive** (`s = 0.3·median-NN`, recomputed per cell)
while plain NN has no scale parameter; under crowding `s` shrinks with the cluster, so the CLU landscape
sharpens exactly where NN's margins collapse. If this matters, the crossover appears at
**M=256, mask p ∈ [0.3, 0.5]**, with gated ≥ NN by **+3 to +15 pp** at compute multiplier **×1.3–1.8**.
Registered discriminator: **H-B is confirmed only if `gated_best − feedforward_nn_best > 0` on ≥3 seeds
with the mean margin exceeding 1 seed-sd.**

**Where the margin appears if it appears (registered):** at **first-pass ∈ [0.45, 0.70]** *and* **NN floor
≤ 0.70** — the gate needs a wrong-enriched low-confidence tail to fix, and the NN must be off ceiling.
A margin at NN ≥ 0.9 is *not* predicted by either hypothesis.

## 4. Pre-registered mechanism controls (carried from w23; these are the ones that carry the novelty)

| # | registered |
|---|---|
| M1 | `random_kick` stays **dead flat** (within ±3 pp of k=0) in every new ambiguity cell — as in all 8 w23 cells |
| M2 | `ensemble` stays **dead flat** in every new ambiguity cell |
| M3 | `ungated_all` **collapses or under-performs gated** at ≥3× compute wherever first-pass > 0.3 |
| M4 | `clu_gated` still **rises monotonically and auto-stops** at multiplier **×1.2–1.9** |
| M5 | ⚠ **Ambiguity-specific risk, registered as a possible falsifier of M1/M2:** if the query is ambiguous, the directed boost's advantage over a random kick should *shrink* (the direction carries less information). I predict the gated−kick gap **falls from w23's +35…+76 pp to +10…+40 pp** but stays **> +5 pp**. A gap ≤ +3 pp would mean the directed mechanism itself is regime-limited — that would be a **new negative**, and I will report it as one. |

## 5. Falsification rules (stated in advance)

- If G6 fails for **all** candidate levels in both regimes ⇒ report the regimes as **failed their purpose**
  (Item 2 says: do not run the ladder into a saturated cell) and iterate levels **once**, on the gate only.
- If H-A holds ⇒ that is the **decision-grade NO** for R3's leaderboard claim; report it as plainly as w23
  reported the mask result, and do **not** retune the regime to manufacture a win (task ⚠ standing trap).
- If H-B holds ⇒ it must survive **≥3 seeds** and the margin must exceed one seed-sd, else it is reported
  as within-noise, not a win.
