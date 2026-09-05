# PREREG — primitive-harness (w20)

Written **before** any harness run. Author: experiment-engineer. Base `main` @ `089cc6e`.

The acceptance criterion includes measured accuracy curves (recall vs #distractors) and a
matched-budget table, so protocol §5's pre-registration rule applies. I commit to numbers and
to the *reasoning* below, then measure.

---

## 0. What is being predicted

Five primitives — `mlp` (token-wise, no mixing; control), `gru`, `ssm` (S4D-style diagonal
SSM, optionally selective/Mamba-lite), `attention` (causal MHA), `clu` (input-driven
CHLU recurrence) — dropped into the **same** slot of the same 2-layer model
(embedding + learned positional embedding → 2×[block + residual + LayerNorm] → linear head),
matched on parameter count, same optimizer/schedule/data/LR grid.

Three task families, reported separately, never averaged:
1. **MQAR** (associative recall) — headline sweep: accuracy vs sequence length at fixed
   `num_kv_pairs`, i.e. vs number of distractor/pad positions.
2. **Adding problem** at T=128 (long-range integration; the HiPPO/S4-native family).
3. **Parity** at T=64 (state tracking; the family where recurrence beats attention).

---

## 1. The task file's structural prediction — I predict it FAILS as stated

The task says: *"attention degrades from interference and CLU does not, because of barrier
confinement (Prop 2)."* **I pre-register that this will not reproduce in this harness**, and
I want that on the record before I measure, because it is the load-bearing claim of the
headline figure.

Two independent reasons:

- **Prop 2 is a statement about a hand-designed landscape, not a trained block.**
  `clu-retrieval-demo` obtained hard read isolation from `RingRegisterPotential` — an
  *engineered* additively-separable potential with designed barriers, with **no training
  anywhere**. The drop-in CLU block here learns `V_θ` by backprop. Nothing forces the learned
  `V_θ` to have separated basins, and that report's own §6 shows the address selector is the
  stage that fails. Isolation is a property of the design, not of the primitive.
- **On MQAR, attention is the strong baseline, not the weak one.** In Zoology
  (Arora et al. 2023, arXiv:2312.04927) — the source of this task — the finding is the
  *opposite* of the one quoted: exact attention solves MQAR essentially perfectly because it
  has an O(T) KV cache, while fixed-state gated-convolution/SSM models degrade as KV pairs
  grow. A softmax over T keys is not obviously "interference-limited" at these scales; the
  recall-gap literature is about *recurrent* models running out of state.

**Information-theoretic core of my prediction:** the CLU block carries a fixed-size carry
`(q, p) ∈ R^{2·d_clu}`. Storing `kv` key→value bindings drawn from a vocab of size `V`
requires ≥ `kv · log2(V/2)` bits in that carry. Attention's cache grows with T; CLU's does
not. So **CLU must degrade with `num_kv_pairs`** whatever its potential looks like. Barrier
confinement can protect an *already-stored* item; it cannot create capacity.

### P1 — MQAR, accuracy vs sequence length (fixed kv=4, T ∈ {32,64,128,256})
| primitive | T=32 | T=256 | predicted shape |
|---|---|---|---|
| attention | ≥0.90 | **≥0.85** | ~flat; degrades ≤0.10 across the sweep |
| gru | ≥0.70 | 0.30–0.70 | decays with T (gradient path length) |
| ssm | ≥0.70 | 0.35–0.75 | decays, but less than GRU |
| **clu** | 0.35–0.85 | **0.15–0.55** | **decays**; ≥0.15 absolute drop T=32→256 |
| mlp (control) | ≤0.15 | ≤0.15 | at/near chance — it cannot mix tokens at all |

**Ranking prediction: attention > {ssm, gru} > clu > mlp at T=256.**
Chance ≈ 1/(V/2) ≈ 0.008 at V=256; the `mlp` control pins the floor (it can only learn the
unigram marginal, so ≤0.15 allows for it exploiting value-token frequency).

**Falsifier that would make the task's claim survive:** CLU flat (≤0.10 drop) across the T
sweep *and* attention dropping ≥0.15. I consider this unlikely (<15%).

### P1b — MQAR, accuracy vs number of KV pairs (kv ∈ {2,4,8,16} at T=128)
Predicted: attention ~flat (≥0.80 at kv=16); **clu degrades monotonically**, ≥0.25 absolute
drop from kv=2 to kv=16. This is the capacity claim above and is the honest version of the
"recall vs distractors" figure. If CLU's kv-curve is flat, my capacity argument is wrong.

### P2 — Adding problem, T=128 (MSE; predict-the-sum-of-2-marked-entries)
CLU is a *lossless oscillator* when γ→0: a symplectic integrator with a near-quadratic learned
`V` is a linear SSM with purely imaginary eigenvalues, i.e. structurally the same object S4
approximates. So this is the family where I expect CLU to be genuinely competitive.
- ssm: MSE ≤ 0.02 (solves it)
- **clu: MSE ≤ 0.05** — within 3× of the SSM
- gru: MSE ≤ 0.10
- attention: MSE ≤ 0.05 (fine at T=128)
- mlp control: MSE ≈ 0.17 (variance of the target; cannot mix)
**Ranking: ssm ≈ attention ≤ clu ≤ gru << mlp.**

### P3 — Parity, T=64
Attention (finite depth, no recurrence) is known not to represent parity robustly;
recurrent state tracking does.
- gru: ≥0.95
- **clu: ≥0.80** (nonlinear `V_θ` + recurrence gives it the mechanism)
- ssm: 0.60–0.95 (diagonal-linear recurrence + inter-layer nonlinearity: partial)
- attention: ≤0.75
- mlp control: ≈0.50 (chance)
**Ranking: gru > clu ≈ ssm > attention > mlp.**

### P4 — Compute cost (Item 2; stated, not competed on)
CLU runs a Verlet step (⇒ a `∇V` evaluation, i.e. a backward pass through the potential MLP)
per token, sequentially. Predicted **wall-clock per training step: 3–15× the GRU** and
**≥5× attention** at T=128. Predicted FLOPs/token: within 3× of the matched-parameter GRU
(the params are matched; the overhead is the grad-of-MLP and the sequential scan, which costs
wall-clock more than FLOPs). I predict the honest headline multiple is **wall-clock-dominated,
not FLOPs-dominated**.

### P5 — Drop-in-ability (Item 1)
I predict CLU **can** be made drop-in `(T,d)→(T,d)` with **three** concessions, none of which
require special-casing the harness: (i) input enters as a momentum impulse `p += W_in x_t`, so
the block is **not** energy-conserving (a driven, not autonomous, Hamiltonian); (ii) γ>0 is
required for a readable state (`clu-retrieval-demo` §6: γ=0 gives 0.813 vs 1.000), so the
block is **dissipative, not symplectic**; (iii) the carry is `2·d_clu` wide, so param matching
must solve for `d_clu` separately. I predict **no** fourth concession is needed. If a fourth
is needed — in particular if the rollout is numerically unstable (NaN/overflow) and needs
clipping or a bounded input kick — **that is a finding against the primitive claim** and I
will report it as such rather than quietly adding the clip.

---

## 2. Capacity coordination (task ⚠)

I will **not** quote the "8-item ceiling"; it is a 2-D-ring artifact of the designed landscape.
`address-space-dimension-scaling` is measuring the real capacity concurrently. My MQAR kv sweep
tops out at kv=16 with `d_clu` set by parameter matching (expected 24–48), i.e. an address
space of dimension ≫2, so the ring law does not bound this harness. If that task returns a
capacity law, P1b's kv-curve is the quantity to reconcile against it.

## 3. Tuning-budget commitment (Item 4)

The LR grid is **identical and fixed in advance** for every primitive: `{3e-4, 1e-3, 3e-3}`,
same steps, same batch, same seed for selection; best-LR selected on a held-out eval batch;
final numbers at 3 seeds. Baselines therefore receive *exactly* the same budget as CLU, by
construction, and I will report the count. **I will not add a CLU-only knob after seeing
results.** Any post-hoc change to CLU's config (dt, γ, kinetic mode) will be reported as a
separate, labelled, post-hoc row — never folded into the headline table.
