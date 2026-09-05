# Task A: clu-autoencoder — does a landscape-plus-pointer beat a static latent vector at matched d? (w22)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/clu-autoencoder.md` · **Branch:** `agent/experiment-engineer/clu-autoencoder`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/relaxation-addressing-theory.md` **Item 5 (Prop 11, the fiber payload — this task is its scaled-up test)** · `.claude/outputs/gamma-read-sweep.md` and `.claude/outputs/potential-function-class.md` (**both prerequisites — take γ and the potential class from their results**)
- ⚠ **DEPENDENCY: run AFTER tasks C and D.** C decides the read γ; D decides the potential family. Running before them means guessing both.

## Why — the flagship experiment for the program's thesis
This closes the **architectural gap** standing since w19 (read-in `φ` ≈ identity, read-out `ψ` handcrafted, all representational burden on `V_θ`) and tests the Head's framing against a standard, unambiguous baseline.

**The framing, stated precisely (this wording matters — an earlier, looser version was wrong):**
- **Autoencoder:** capacity lives in the **latent vector**; the decoder is shared. Expressivity ≈ `d`.
- **CLU:** capacity lives in the **landscape `V_θ`**; the address `(m, q₀, p₀)` is a **pointer**; the rollout is the mechanism that reads the landscape's local structure out **through time**.
⚠ **Do NOT claim "the trajectory carries more bits than the endpoint."** With `V_θ` fixed, the trajectory is a *deterministic function* of the initial condition and adds no information. **Prop 11's actual content:** two items at the **same location** with payload written in **curvature** gave identical endpoints (**2.2e-16**), an endpoint read **at chance (0.425)**, and a **trajectory read at 100%**. The payload is in `V_θ`; the rollout makes it **linearly decodable**. That is the claim to test.

## Item 1 — the architecture
```
x → [encoder φ] → (q₀, p₀)   ← LEARNED (this is the w19 gap being closed)
(q₀,p₀) → [CLU rollout in V_θ] → trajectory (T × d)
trajectory → [decoder ψ] → x̂   ← LEARNED sequence decoder, NOT handcrafted scalars
```
Two decode modes, both required:
- **(a) endpoint decode** — ψ sees only the settled state (the AE-equivalent read).
- **(b) trajectory decode** — ψ sees the rollout (subsampled; report the subsample budget).
⚠ Use the **γ per read mode** from task C: endpoint decode wants γ>0 (settle), trajectory decode wants γ≈0 (the oscillation is the signal). Do not use one γ for both.

## Item 2 — the decisive comparison
**Baseline: a standard autoencoder at matched latent dimension `d` and matched total parameters.** Sweep `d ∈ {2, 4, 8, 16, 32}`.

**Deliverable: reconstruction error vs `d`, four curves — AE · CLU-endpoint · CLU-trajectory · (control).** The registered hypothesis: **AE and CLU-endpoint track each other; CLU-trajectory beats both at small `d`**, because it reads the landscape rather than only the pointer.

⚠ **The control is load-bearing and must be run:** a CLU with a **frozen/blank `V_θ`** (no landscape structure) and the same trajectory decoder. **If the trajectory decode wins with a blank landscape, the gain is from the decoder's extra capacity, not from the physics** — that is the AE-side analogue of the blank-landscape control, and without it this task proves nothing. Match ψ's parameter count across all arms.

## ⭐ Item 2b — THE PAYLOAD CHANNELS (Head, 2026-07-21). The fiber is richer than Prop 11 tested.
Prop 11 wrote the payload into **curvature** and read it as frequency. **That is one channel out of at least six**, and it is not the best one. `ω = √(k/M)`, so **mass reaches the same frequency channel — and mass is *per-launch* while curvature is not.**

| channel | side | per-item at one location? | learnable |
|---|---|---|---|
| ⭐ **mass `M`** | **address** | **yes** | **yes — Prop 6, ratios exact to 2.2e-14** |
| **`p₀`** (amplitude/phase) | **address** | yes | yes |
| curvature `k` | landscape | shared | via `θ` |
| anharmonic coefficients | landscape | shared | via `θ` |
| `μ²` (half-life) | landscape | per coset direction | write-mode selector |
| `γ` | global / `γ_φ` field | **no** (not per-item at a fixed location) | — |
| temperature | **unbuilt** (t-lever) and **stochastic** — a noise channel, not a deterministic code | — | — |

**Measure the per-channel capacity at a SINGLE location** — how many items are distinguishable via each channel alone, at a fixed read length — and then **whether channels multiplex** (are `M` and `k` independent codes, or one code?).

⚠ **The degeneracy that must be checked first: `ω = √(k/M)` makes a stiff-well/heavy-particle pair indistinguishable from a soft-well/light-particle pair IF you read only the period.** This is the theorist's OQ1 V↔M gauge. **Hub's proposed resolution, to confirm or refute: mass is a pure time-reparameterization; landscape shape is not** — varying `M` rescales time uniformly, while anharmonicity makes frequency amplitude-dependent and changes the orbit's *waveform*. ⇒ **read the waveform, not just the period.** If `relaxation-fiber-capacity` (theorist) has landed, take its verdict; otherwise measure the separability directly and report it.

⭐ **Consequence for the capacity law, and it is the reason this item exists:** `address-space-dimension-scaling` measured `K_max = 4·2^d` by **spatial packing alone**. If each location carries a fiber of its own, **total capacity is spatial × fiber — multiplicative.** Report the measured fiber capacity in bits/location so the two can be composed. ⚠ **Do not multiply them into a headline number until the channels are shown independent** — a degenerate channel adds nothing.

## Item 3 — where the capacity actually lives
Test the pointer-into-a-landscape claim directly:
1. **Fix `d` and scale `V_θ`'s parameters.** If capacity is in the landscape, reconstruction should improve with `|θ_V|` **at fixed `d`** — something an AE structurally cannot do (its bottleneck is `d`).
2. **Fix `V_θ` and scale `d`.** The pointer's precision should bound retrieval, giving the complementary curve.
3. **Fix both and scale the read length `T`.** Per Item 2b the fiber is read *through time*, so reconstruction should improve with `T` up to the point where the jet is exhausted — the third axis, and the one that has no autoencoder analogue at all.
Report both. **This 2-D picture is the strongest available evidence for the framing, and it is the figure I would build the short around.**

## Item 4 — write modes ↔ read modes
The vision assigns different memory modes to different read types: **SSB/latched items → endpoint decode; dissipating/trajectory items → trajectory decode.** With a mixed dataset (some items reconstructable from a point, some needing the rollout), does routing by mode beat using one mode for everything? ⚠ **Exploratory and permitted to be inconclusive** — say so if it is.

## Item 5 — dataset
Start with a standard, uncontroversial reconstruction benchmark (MNIST/CIFAR-scale is fine; state the choice and why). **The baseline's credibility matters more than the dataset's difficulty** — tune the AE at least as hard as CLU and report both tuning budgets, per the `primitive-harness` symmetric-rescue protocol.

## Acceptance
The learned φ/ψ architecture, the reconstruction-vs-`d` curves with the **blank-landscape control**, the 2-D capacity picture (Item 3), and the mode-routing probe. Matched parameters and matched tuning budgets throughout, both stated. Tests green.

⚠ **This task is permitted — and reasonably likely — to return "CLU-trajectory does not beat a matched AE."** That would say the fiber payload does not scale beyond the 2-item toy, and it is a decisive, publishable result that would redirect the program. **Report it plainly. Do not add capacity to ψ until CLU wins**; if you change ψ, change it for every arm.
