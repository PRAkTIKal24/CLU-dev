# PREREG — potential-function-class (w21, Task D)

Written **before** any harness run. Base `main` @ `31c3e15`, branch
`agent/experiment-engineer/potential-function-class`, worktree `../CHLU-potential-class`.
Every number below is a prediction, with its derivation. Ranges are 5-seed means.

---

## 0. The two hypotheses as the task states them

- **H-EXPR** — `PotentialMLP` is too weak; a higher-capacity class (transformer/attention) fixes the w20 failure.
- **H-SUPP** — the failure is *global parameter support*: any weight update moves every stored item;
  attention is *more* global than an MLP so it should fail at least as badly; only atom writes are local.

**I pre-register a third possibility as my primary expectation (H-MIX), because it is what the algebra says:**

> **H-MIX.** *Fidelity* (can K items be held at all) is set by **expressivity** — so the transformer and
> atom arms both improve on the MLP. *Cross-write interference* is set by the **support of the write in
> configuration space**, which is a property of the function class **and its temperature**, not of capacity —
> so corruption orders as `designed ≈ atoms_local < atoms ≈ hopfield(β large) < hopfield(β small) ≈ attn < mlp`.
> The two metrics dissociate, and **neither H-EXPR nor H-SUPP as stated is the whole answer.**

**Derivation of the key clause (why I expect H-SUPP's "attention is more global" to be FALSE).**
The modern-Hopfield energy is `V(q) = −β⁻¹ logsumexp_i(β⟨q,k_i⟩) + α‖q‖²`. Its parameter sensitivity is
`∂V/∂k_i = −softmax_i(β⟨q,k_j⟩)·q`, i.e. the influence of memory `i` at a point `q` is its **softmax
weight**. On the unit ring with `k_i ≈ z_i`, a neighbour at angular separation `Δθ` carries weight
`≈ exp(−β(1−cosΔθ))`. At K=4 (Δθ=π/2) this is `exp(−β)`: **1.4e-1 at β=2, 3.4e-4 at β=8.**
So attention's support is *exponentially local in the inner-product metric* and is a **tunable knob**,
not "more global than an MLP". An MLP has no such knob at all. I therefore expect the transformer arm
to sit **between** designed and MLP, and to move with β — which is itself the discriminating measurement.

---

## 1. Matched-parameter table (pure arithmetic, must hold exactly)

| arm | learned parameterisation | count | Δ vs 4481 |
|---|---|---|---|
| `mlp` | `PotentialMLP(dim=3, hidden=64)` = 3·64+64 + 64·64+64 + 64+1 | **4481** | — |
| `hopfield` / `hopfield_sharp` | `n_mem·(d_head + 1)` = 1120·(3+1) | **4480** | −0.02% |
| `attn` | `d_head·dim + n_mem·d_head + n_mem` = 8·3 + 495·8 + 495 | **4479** | −0.04% |
| `atoms` / `atoms_local` | `n_atoms·(dim + 1 + 1)` = 896·5 | **4480** | −0.02% |

**P0.** All four learned classes are within **0.05%** of 4481 (tolerance 5%). If any arm's measured
`n_learned_params` differs from the table, the run is invalid and must be re-sized before reporting.

---

## 2. Fidelity — strict retrieval, 5 seeds, K ∈ {4, 8}

`designed` and `mlp` are **replications** of w20 through the same code path; if they do not reproduce,
the harness (not the physics) is at fault and nothing else in the run may be reported.

| arm | K=4 (predicted) | K=8 (predicted) | basis |
|---|---|---|---|
| `designed` | **1.000 ± 0.00** | **0.986 ± 0.01** | w20 measured 1.000±0.000 / 0.986±0.003 |
| `mlp` | **0.85 ± 0.10** | **0.60 ± 0.10** | w20 measured 0.853±0.095 / 0.599±0.059 |
| `hopfield` (β=2) | 0.70 – 0.95 | 0.45 – 0.75 | soft, overlapping memories; neighbour weight 1.4e-1 |
| `hopfield_sharp` (β=8) | **0.90 – 1.00** | 0.75 – 0.95 | at large β, `k_i = 2α z_i` makes each `z_i` an exact stationary point of the energy; the writer only has to find that solution |
| `attn` | 0.60 – 0.90 | 0.40 – 0.75 | value head must *construct* wells; no closed-form solution as above |
| `atoms` | **0.85 – 1.00** | 0.70 – 0.95 | a Gaussian well of width ≈ σ_addr at `z_i` is an exact minimiser of the write loss; 896 atoms is 112–224 per item, vastly over-complete |
| `atoms_local` | 0.85 – 1.00 | 0.70 – 0.95 | masking should not hurt: ≥112 atoms per item |

**P1.** At least one learned class clears **0.9 at both K=4 and K=8** (w20: none did). Confidence 60%.
**P1′ (the falsifier).** If no learned class clears it, that is the strong program-level result the task
names: the function class is not the problem.

**P2 (why w20's `local_rbf` failed).** w20's RBF rung used 24 atoms with `depth_raw ~ N(0,0.1)`
⇒ each atom has depth `softplus(0) ≈ 0.69` at initialisation, i.e. the landscape *starts* rugged on the
0.3 length-scale that retrieval has to traverse. This arm initialises depths at `softplus(−8) ≈ 3.4e-4`
(flat start, writer digs). I predict `atoms` ≫ w20's `local_rbf` (0.623±0.330 / 0.348±0.117) and that
**initialisation, not the atom basis, is what failed in w20.** Confidence 65%.

---

## 3. Blank controls (mandatory on every cell)

**P3.** The **classification** blank (strongest read = max of codebook / nearest-centroid) **fails
(≥0.9, i.e. ≫ chance)** for *every* learned class, exactly as in w20: the anti-decoration guard makes the
payload channel deterministic given the address, so any coupling of `q2` to `(q0,q1)` — which every
learned class has generically — is a perfect item code. The **value** blank (blank strict ≤ 0.1) **passes**
for every class. `designed` passes both. Consequence: **all headline scoring is the leak-immune value
criterion**; classification cells are reported as "not a measurement".
Confidence 80%. If a learned class's classification blank *passes*, that is a genuine finding
(the class is payload-separable) and must be checked before being reported.

---

## 4. ⭐ Cross-write interference (the discriminator): write A, write B, re-read A

Reference points: designed **0.000**, MLP **2.9e-2 … 5.0e-1** (w20 `free_mlp` = 3.53e-1).

| arm | predicted corruption of A | derivation |
|---|---|---|
| `designed` | **0.000** | exact additive separability (w20/w19) |
| `mlp` | 1e-1 – 5e-1 | replication of w20 |
| `hopfield` (β=2) | 5e-2 – 3e-1 | new key's softmax weight at an old site ≈ `exp(−2) = 1.4e-1` |
| `hopfield_sharp` (β=8) | **1e-3 – 3e-2** | same, `exp(−8) = 3.4e-4`, times an O(1) depth and an amplification by the write's own optimisation |
| `attn` | 1e-1 – 5e-1 | learned projection destroys the metric locality of the softmax |
| `atoms` (global write) | 1e-2 – 2e-1 | Gaussian atoms are local, but the *gradient write* moves every atom |
| `atoms_local` | **1e-5 – 1e-3, and < 1e-3 with high confidence** | A-atoms are bit-identical after B's write; the only channel left is B's own Gaussian tails at the site separation `2f·sin(π/K) = 1.414` (K=4) with `s ≈ 0.3`: `exp(−1.414²/(2·0.3²)) = 1.5e-5` × O(1) depth |

**P4 (the sharp one).** `atoms_local` corruption **< 1e-3**, and **at least 100× smaller than `mlp`**.
**P5.** `hopfield_sharp` corruption **< `hopfield`(β=2) corruption**, i.e. interference is a *monotone
function of the attention temperature*. This is the clean test that support, not capacity, drives
interference — capacity is identical across the two β arms (same parameter count, same class).
**P6.** `attn` and `mlp` corruption are both **> 1e-2** (no learned global-support class is clean).

---

## 5. Support radius, measured (`‖δ∇V(q)‖` vs distance `r` from the write site)

Primary statistic: `rms‖∇V_after(q) − ∇V_before(q)‖` over probe points on a sphere of radius `r`
around the new item's site, normalised by its value at the smallest probed radius. (The force, not the
raw `V`, because `V` is defined only up to a constant and only `∇V` enters the dynamics; raw
`std(δV)` is reported as a secondary column.)

Define **r₁₀** = the radius at which the normalised curve first falls below 0.10.

| arm | predicted decay law | predicted r₁₀ |
|---|---|---|
| `mlp` | none within the probed range (global support) | **> 2.0 (unresolved)**, normalised value at r=2 **≥ 0.3** |
| `attn` | slow / non-monotone | > 1.0 |
| `hopfield` (β=2) | `exp(−βr)`-like | 1.0 – 2.5 |
| `hopfield_sharp` (β=8) | `exp(−βr)`-like | 0.3 – 0.8 |
| `atoms` | Gaussian `exp(−r²/2s²)`, `s≈0.3` | 0.4 – 0.8 |
| `atoms_local` | Gaussian, identical to `atoms` | 0.4 – 0.8 |
| `designed` | `δV ≡ 0` (zero learned parameters) | n/a — must be **exactly 0 at every r**; if not, the harness is wrong |

**P7.** The **rank order of r₁₀ across arms matches the rank order of interference corruption** (§4).
This is the mechanism claim and is the most transferable number in the task. Falsified if the Spearman
correlation over the 6 learned arms is < 0.6.

---

## 6. Does the design-freedom curve move? (task Item 3)

w20 headline: *minimum designed structure that preserves the loop = essentially ALL of it*
(no learned rung cleared 0.9 at both K=4 and K=8; `skeleton_residual`, freedom 1, was closest at
0.903±0.101 / 0.959±0.043).

**P8 (primary, 55% confidence).** Re-running the ladder with the best class as the *learned family*
moves the minimum-viable-design point from **freedom 1 (marginal)** to **freedom 4** — the fully free
potential family with only coercivity + designed write sites — i.e. **w20's headline was a property of
the MLP function class, not of the loop.**
**P8′ (alternative, 45%).** The point does not move: no family clears 0.9 at both K. Then the headline
stands and strengthens, and the correct program statement is *"structure, not function class"*.
I commit to reporting whichever occurs, and to reporting P8's failure as the headline if it fails.

---

## 7. Cost (task Item 4)

**P9.** Wall-clock per write orders `designed (0) < mlp < atoms ≈ hopfield < attn`, within a factor 3
across the learned arms (all are one small dense op per probe point at dim=3; the atom/memory arms are
a single 896×3 or 1120×3 matmul, comparable to the MLP's 64×64). **No arm is 10× another.**
Confidence 60%. FLOPs/eval reported analytically: mlp ≈ 2·4481, atoms ≈ 896·(3 mul + 3 add + exp),
hopfield ≈ 1120·3 + logsumexp, attn ≈ 8·3 + 495·8 + softmax.

---

## 8. Which hypothesis I expect the data to support

**H-MIX** (§0): expressivity fixes *fidelity*, support locality fixes *interference*, and they dissociate.
Ranked expectation of the headline sentence:
1. (50%) "Both H-EXPR and H-SUPP are partly right and the task's dichotomy is false: fidelity ← class, interference ← support."
2. (25%) "H-SUPP: no learned class clears the fidelity bar; only the *local write operator* removes interference."
3. (15%) "Neither: every learned class fails both, the function class is not the problem, structure is."
4. (10%) "H-EXPR: the transformer arm fixes everything."

**Adversarial check committed in advance (task's ⚠).** If `hopfield`/`attn` beats `atoms_local` on
**interference**, I will (i) verify the atom write mask actually froze the A-atoms (bit-identical
parameter check, not a loss check), (ii) verify the hopfield arm's blank control, (iii) re-run at 2 extra
seeds, and (iv) report it as a contradiction of theorist C3/CM-6 explicitly rather than as a result.
