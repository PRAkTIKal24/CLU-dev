# PREREG — learned-landscape-write-read (w20)

Written **before** any harness was executed. Predictions + derivation. Scored in the report.

## Protocol being pre-registered

Five **design-freedom rungs** for the memory landscape `V`, identical geometry (K sites on a
ring of radius f=1, payload channel q2, anti-decoration guard q2(0)=p2(0)=0), identical
read (linear codebook read on the rollout tail), identical query jitter:

| rung | freedom | designed terms | learned terms |
|---|---|---|---|
| `designed` | 0 | ring vacuum + K angular wells + payload spring w/ true payloads (w19) | none |
| `skeleton_residual` | 1 | same as above | MLP residual × 0.1 |
| `sites_learned_payload` | 2 | ring vacuum + K angular wells (address geometry only) | MLP (full) |
| `local_rbf` | 3 | coercivity only | RBF atoms (centres/widths/depths learned) |
| `free_mlp` | 4 | coercivity only (0.05‖q‖², required for BIBO) | MLP (full) |

Training = a **write objective** on the learned part only: each target state
`z_i = (c_i, a_i)` is made a local minimum (‖∇V(z_i)‖² + a contrastive margin against
Gaussian perturbations incl. the q2=0 query manifold) with an inter-site barrier term.
No BPTT through the rollout (Prop 5 says the damped rollout is address-gradient-opaque;
the θ-gradient is not, but the static objective is cheaper and sufficient to make the
question well posed).

Retrieval is **two-phase**: relax the query at γ_address → address; roll out from the
address at γ_read → trajectory → linear read.

## Predictions

**P1 — write→relax consistency (K=4, per rung).** Basin-level / strict:
- `designed` ≈ 1.00 / ≈ 1.00 (w19 replication; if this misses, the harness is wrong, not the physics)
- `skeleton_residual` ≥ 0.95 / ≥ 0.85
- `sites_learned_payload` ≈ 1.00 basin (sites are designed) / 0.60–0.90 strict
- `local_rbf` 0.50–0.90 / 0.20–0.60
- `free_mlp` ≤ 0.50 / ≤ 0.20
**Derivation:** basin-level success is controlled by whether the *address geometry* exists;
strict success additionally needs the payload well to be accurate at the ~0.1 codebook scale.
Rungs 0–2 have the geometry designed in ⇒ basin-level should be near-perfect; rungs 3–4 must
*create* the geometry from the write loss, and D3 (exact structure is measure-zero) + N46
(emergent has no register) say a free family lands near, not on, the intended structure.
I predict the basin/strict gap **widens monotonically with design freedom** — this is the
w19 / theorist-Toy-D pattern (9/18 basin vs 2/18 strict) reappearing as a function of design.

**P2 — retrieval fidelity vs the w19 baseline.** Codebook read @K=2 / @K=8:
- `designed` 1.000 / ≥0.98 (w19: 1.000 / 0.992)
- `skeleton_residual` ≥0.95 / ≥0.85
- `sites_learned_payload` ≥0.90 / 0.50–0.85
- `local_rbf` 0.60–0.95 / ≤0.60
- `free_mlp` ≤0.70 @K=2, **≈ chance at K=8**
**Minimum-viable-design point (the deliverable):** I predict it is **rung 2
(`sites_learned_payload`) — the designed *address geometry* (site skeleton) is the minimum
required structure**, and that imposed *locality alone* (rung 3) is NOT sufficient because
locality does not fix where the atoms go.

**P3 — blank controls.** Every blank cell at chance (≤ 1/K + 0.15). Any blank above that
invalidates its cell. If a learned rung's blank control is *high*, the most likely cause is
the learned V having encoded site identity into the address plane that the read can see —
I commit in advance to treating such a cell as **not a measurement**, not as a success.

**P4 — durability.** `designed` and `skeleton_residual` flat (≤0.02 drop) out to 1200 steps.
`free_mlp`/`local_rbf`: I predict **decay** — a learned minimum is generically not exactly a
fixed point of the *discrete* Verlet map at γ_read=0, so residual energy circulates; a drop
≥0.10 from step 100 to step 1200 for at least one free rung.

**P5 — cross-write interference (Item 3).** Write A (K−1 items), then write B at a fresh
site by continuing training, then re-read A. Corruption = |Δ payload readout of A|:
- `designed` ≈ 0 (by construction — w19 measured 4.17e-7; nothing is trained)
- `skeleton_residual` ≤ 1e-2
- `sites_learned_payload` 1e-2 … 1e-1
- `local_rbf` ≤ 5e-2 (locality is the point)
- `free_mlp` **≥ 0.1** (i.e. of order the codebook spacing ⇒ destructive)
**Derivation:** an MLP has global support; every write moves V everywhere (CM-6 erosion is
the measured instance). **I predict additive separability does NOT survive learning for the
global families, and that w19's 4.17e-7 is therefore an artifact of the design.**

**P6 — the 2-D γ map (Item 4).** The Hub predicts the good region is off-diagonal:
γ_address > 0, γ_read ≈ 0. I split this into two claims and predict them differently:
- (a) **γ_address > 0 is required**: fidelity at γ_address = 0 is poor (≤0.7) for every rung,
  and rises to its maximum by γ_address ≈ 0.02–0.05. **PREDICT TRUE.**
- (b) **γ_read ≈ 0 is required / γ_read > 0 hurts**: **PREDICT FALSE.** Once phase 1 has
  relaxed the query to a minimum with p ≈ 0, that point is a fixed point of the damped *and*
  the conservative map, so phase 2 is nearly γ_read-independent. I predict the fidelity
  variation across γ_read at fixed good γ_address is **< 0.05**, i.e. the map is essentially
  a function of γ_address alone.
- Consequence if both fire: the w19 tension (γ=0 → 0.813) is **a single-phase artifact**;
  two-phase retrieval with γ_address>0, γ_read=0 recovers ≈1.0, which is what makes the
  Prop-4 gradient-safe conservative read usable. The "good region is off-diagonal" claim is
  then **true in the weak sense (γ_read=0 is permitted) and false in the strong sense
  (γ_read=0 is not required)**.

**P7 — the extraordinary-claim guard.** If `free_mlp` produces a working loop (codebook read
≥0.9 at K=4 with a passing blank control), that contradicts N46/N7/CM-16a/D1/D3 and I will
report it as an anomaly requiring checking, not as a success — first check being whether the
write objective has effectively hand-placed the structure (it does supply the site locations
`c_i` as targets: **the writer chooses where to write in every rung**, so "free" here means
free *potential family*, not free structure — stated up front so the result is not oversold).

## Committed interpretation rules
- If P1's `designed` rung does not reproduce w19 (≥0.95 basin & strict), I report a harness
  failure and do not interpret the other rungs.
- If every learned rung fails at K=2, the headline is **"the loop does not survive learning"**
  and I report it as such rather than tuning until one passes.
- Any rung that passes only after post-hoc hyperparameter search will be labelled TUNED with
  the number of configurations tried.
