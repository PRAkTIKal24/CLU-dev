# Task: designed-mechanism-learned-content — is the K=8 wall GEOMETRY or LEARNING? The primitive-decider. (w22)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/designed-mechanism-learned-content.md` · **Branch:** `agent/experiment-engineer/designed-mechanism-learned-content`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/potential-function-class.md` (the `atoms` arm that reached 0.98 @K=4 and broke @K=8) · `.claude/outputs/address-space-dimension-scaling.md` (the `4·2^d` designed capacity + the `d_eff`/`Δ_req` law) · `.claude/outputs/relaxation-fiber-capacity.md` §4.3 (**the parameter ceiling `B_total ≤ P·b_θ` — load-bearing here**) · `.claude/outputs/mass-visible-objective.md` (per-launch `mass_override`, `exp_zeromean`) · `chlu/core/memory_potentials.py` (`AtomDictionaryPotential` = the learned dictionary)

## Why — this is the exact configuration ICLR referees will attack
Every "primitive" is a **designed mechanism with learned content**: attention's `softmax(QK)V` is fixed, `W_{Q,K,V}` are learned. Our w19/w20 "designed" experiments designed the *content too* (hand-placed centers AND payloads) — more than a primitive should. The one time we tested the fair configuration — the `atoms` arm in `potential-function-class`, a fixed atom-dictionary mechanism with **learned** amplitudes/centers/widths — it scored **0.980 @K=4** (≈ the designed ceiling) and **0.741 @K=8** (failed the 0.9 bar).

**And K=8 sits right at the 2-D ring's own capacity ceiling (`K_max ≈ 8.4`).** So we cannot currently tell whether the wall is:
- **H-GEOMETRY:** the ring ran out of room; learned content is fine, the geometry is small ⇒ **the wall moves up with `d`** (designed capacity is `4·2^d`), and **the primitive claim is alive**.
- **H-LEARNING:** gradient descent cannot fill a landscape past ~8 items regardless of room ⇒ the wall **stays near 8 at every `d`**, and the primitive claim is in serious trouble.

**This one experiment discriminates them.** Per the Head: **no retreat is pre-registered** — a plateau is a finding to route around, not a stop sign.

## Item 1 — the discriminator: learned-content fidelity vs `d`
Fixed **designed mechanism** = `AtomDictionaryPotential` (the learned dictionary: `α‖q‖² + Σ_j −A_j exp(−‖q−c_j‖²/2s_j²)`, amplitudes/centers/widths **learned**, group-masked writes available). **Learned content** = those parameters, trained by the static write objective (`train_memory.py`).

Sweep **`d ∈ {2, 3, 4, 6, 8}`** (extend if cheap). At each `d`, push `K` up the ladder and find **`K_learned`** = the largest item count clearing **strict 0.9** (leak-immune value criterion; blank control over the strongest read on **every** cell — w20/w21 method finding). Overlay the **designed** ceiling `K_designed = 4·2^d` measured by `address-space-dimension-scaling` at the same `d`.

**Deliverable: `K_learned` vs `d` and `K_designed` vs `d` on one axis.**
- If `K_learned` **tracks `4·2^d`** (or any strong growth) ⇒ **H-GEOMETRY, primitive claim alive.**
- If `K_learned` **plateaus near 8** while `K_designed` climbs ⇒ **H-LEARNING, the wall is real.**
State which, quantitatively, with the fitted growth of `K_learned`.

⚠ **The parameter ceiling is a confound you MUST control (theorist §4.3).** `B_total ≤ P·b_θ`: a learned dictionary with too few atoms *cannot* represent many wells regardless of geometry. **Scale the atom count / parameter budget with `K` (and report `P` per cell)** so that a plateau is a *learning* failure, not a *capacity-of-the-parameterization* failure. If you cannot separate these, say so — but the whole point is to isolate learning from both geometry and parameter budget.

## Item 2 — does learned MASS help? (folds in `mass-visible-objective`)
The vision makes mass an access key, and `mass-visible-objective` made the spectrum functional (`τ ∝ M^0.79`) and shipped per-launch `mass_override`. Test whether **per-item learned masses** (each atom/item gets its own learned `M`, `exp_zeromean` parameterized) raise `K_learned` or fidelity at fixed `d`:
- arm (a) uniform mass (baseline), arm (b) per-item learned mass.
- ⚠ Respect `relaxation-fiber-capacity` Prop F1: **mass is address-side and only discriminates when the landscape COUPLES coordinates** (`∂_i∂_j V ≠ 0`); it is worth ~0 bits in a separable well. So report whether the atom wells are coupled at the address, and expect mass to help **only** if they are. A null here is a *prediction confirmed*, not a failure.

## Item 3 — the interference axis at scale
`potential-function-class` found the **write operator**, not the class, governs interference (masked write 70×). Confirm this holds **across `d`**: at each `d`, cross-write interference (write A, write B, re-read A) for masked vs global writes. Does the local-write advantage survive higher dimensions, and does it interact with `K_learned`?

## Item 4 — the honest performance statement
For the best configuration found, state: at what `(d, K)` does a **learned-content** CLU match the **designed** ceiling, and where does it fall away? This is the number the paper's "can CLU be learned as a primitive" claim rests on. Report it as a performance frontier, not a single point.

## Acceptance
The `K_learned`-vs-`d` discriminator with `K_designed` overlaid and the parameter budget controlled, the mass arm with its coupling check, the interference-across-`d` confirmation, and the performance frontier. ≥5 seeds on the discriminator cells (w20/w21 showed single seeds mislead here). Tests green.

⚠ **Pre-register H-GEOMETRY vs H-LEARNING and your predicted `K_learned(d)` before running.** ⚠ **Report the honest wall.** If learned content plateaus regardless of `d` and budget, that is the most consequential result the program could produce right now — state it loudly; do not tune around it. Equally, if it scales, do not overstate — a growth slower than `4·2^d` still needs its exponent named.
