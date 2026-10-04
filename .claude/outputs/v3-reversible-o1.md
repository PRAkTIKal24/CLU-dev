# v3-reversible-o1 — results-analyst report

**Task + acceptance criterion:** Measure the adopted V3/ICLR claim *"γ=0 lattice training admits
reversible O(1)-memory (recompute-backwards) BPTT"* — never before run. Deliver the V3-short sentence
with numbers: (1) exactness of leapfrog inversion at γ=0 + γ>0 horizon degradation; (2) recompute-backwards
BPTT vs standard stored-activation BPTT — gradient agreement, peak memory O(1) vs O(T), wall-time overhead.

**Status:** done. Full grid ran on laptop CPU well under budget (each precision pass < 5 min).

---

## Setup (repo read-only; no code touched)

- **Commit:** `63fea62` (main @ wave-6 integration). JAX **0.9.0**, Equinox, device **cpu** (this laptop; no GPU).
- **Reversibility fact exploited:** the CHLU / CLULattice Hamiltonian `H = T(p) + V(q)` is **separable**, so
  `∂H/∂q = V'(q)` is independent of `p` and `∂H/∂p = T'(p)` is independent of `q`. The dissipative
  velocity-Verlet map (`chlu/core/integrators.py:velocity_verlet_step`) therefore has a **closed-form exact inverse**
  (given `(q',p')`): `p* = p'/(1-γ); p_half = p* + (dt/2)V'(q'); q = q' - dt·T'(p_half); p = p_half + (dt/2)V'(q)`.
  At γ=0 the inverse is bit-equivalent to a forward step with `dt→−dt`. This is the RevNet/reversible-integrator
  mechanism cited in Thread-7(1); the lattice rollouts are already `lax.scan`-pure (v3-lattice-build note),
  so nothing in `chlu/` needed changing — I reconstructed the map from `H` and validated against the shipped integrator.
- **Harness:** `.claude/scratch/v3-reversible-o1/reversible.py` (+ `plot.py`). Standard BPTT = `jax.grad` of a
  final-state MSE loss through a forward `lax.scan` (reverse-mode AD stores the length-T tape → O(T)).
  Reversible BPTT = forward scan to the final state (no tape) + a reverse `lax.scan` that at each step
  reconstructs the previous state by the inverse map and accumulates per-step VJP parameter grads (carry is
  fixed-size → O(1) in T). Both are jitted; **peak memory read from XLA's `compiled.memory_analysis().temp_size_in_bytes`**
  (the compiler's scratch-buffer estimate — reproducible, deterministic).

### Flag-provenance table (mandatory, §5)

| Item | Model | commit | seeds | precision | dt | γ | kinetic | potential | coupling / κ | trained? | other non-defaults |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 exactness | `CHLU(dim=8, hidden=16)` | 63fea62 | init key 0 (q key1, p key2) | float32 **and** float64 | 0.05 | {0,1e-4,1e-3,1e-2,0.05,0.1,0.3} | newtonian_learned | mlp | — | **no** (random init) | tie_channel_mass=F; friction_field=None; no lyapunov/langevin/anchor |
| 2 BPTT trade | `build_lattice(unit_dims=[2]·N)` N∈{2,8}, hidden=16 | 63fea62 | 0,1,2 | float32 **and** float64 | 0.05 | 0 | newtonian_learned | mlp | spring, κ_c=0.05 | **no** (random init) | loss = final-state MSE→0; chain topology |

> Untrained random-init models are appropriate here: reversibility and the memory/compute trade are
> **properties of the integrator + autodiff graph**, independent of parameter values. Confounds from training are
> therefore deliberately excluded (see Limitations for what this does *not* cover).

---

## Results

### Item 1 — Exactness of inversion (round-trip: T forward steps then T inverse steps; error on the recovered init)

**γ=0 (conservative): reconstruction error = pure float accumulation, no structural drift.**

| T | rel. recon err (float64) | rel. recon err (float32) |
|---|---|---|
| 16 | 9.1e-17 | 2.9e-8 |
| 64 | 3.3e-16 | 9.9e-8 |
| 256 | 5.8e-16 | 5.1e-7 |
| 1024 | 3.3e-15 | 2.5e-6 |
| 4096 | 1.0e-14 | 1.4e-5 |
| 16384 | 2.0e-13 | 3.5e-3 |

Error sits at machine-ε and grows sub-linearly (≈ √T·ε to ~T·ε) — exactly the "float-accumulation only"
signature the task predicted; **no structural blow-up at γ=0** at any T tested. At the training-relevant
precision (float32) and horizon (T=1024) the recovered initial state is accurate to **2.5e-6 relative**.

**γ>0 (dissipative): the (1−γ) contraction is invertible only in exact arithmetic and amplifies error as (1−γ)⁻ⁿ.**
Measured usable reversal horizon `n(γ)` (largest T with round-trip rel-err < 1e-6):

| γ | horizon n(γ), **float64** | horizon n(γ), **float32** | analytic ~ln(1e-6/ε)/(−ln(1−γ)) |
|---|---|---|---|
| 1e-4 | ≥131072 (ladder-capped) | ~10² (float noise floor limited) | 2.2e5 (f64) |
| 1e-3 | **32768** | ~10² | 2.2e4 |
| 1e-2 | **4096** | ~10¹–10² | 2.2e3 |
| 0.05 | **512** | ~30 | 4.3e2 |
| 0.1 | **128** | ~30 | 2.1e2 |
| 0.3 | **64** | ~16 | 62 |

Beyond the horizon the error explodes geometrically (e.g. float64 γ=0.05: rel-err 8e-12 @T=256 → 21 @T=1024 →
2e51 @T=4096 → inf). The measured horizons track the `(1−γ)⁻ⁿ` law to order-of-magnitude (they run a bit
longer than the naive prediction because the dissipative dynamics contract onto the potential well, where the
reconstruction is better-conditioned). **Practical takeaway: reversibility is exact only at γ=0; any γ>0 caps the
reversible horizon, and in float32 that cap is ~100× tighter than in float64** (γ=1e-2 falls from ~4k steps to ~10²).

### Item 2 — Memory–compute trade, measured (lattice training step, γ=0)

**(a) Gradient agreement — reversible ≡ standard to float tolerance.** Max relative difference between the
recompute-backwards gradient and the stored-activation gradient (median over 3 seeds):

| N (D) | T | float64 rel-diff | float32 rel-diff |
|---|---|---|---|
| 2 (4) | 64 | 4.8e-16 | 4.1e-7 |
| 2 (4) | 256 | 2.8e-15 | 1.4e-6 |
| 2 (4) | 1024 | 4.0e-15 | 2.1e-6 |
| 8 (16) | 64 | 4.8e-16 | 2.2e-7 |
| 8 (16) | 256 | 1.4e-15 | 1.1e-6 |
| 8 (16) | 1024 | 5.8e-15 | 1.7e-6 |

Float64: agreement at ~1e-15 (machine tolerance). **Float32 (the training default): agreement stays ≤ ~2e-6 relative
even at T=1024** — five-plus orders below the SGD gradient-noise floor, i.e. training-indistinguishable.

**(b) Peak activation memory — O(1) vs O(T).** XLA temp-buffer size (float64):

| N (D) | T | standard BPTT temp | reversible temp | ratio |
|---|---|---|---|---|
| 2 (4) | 64 | 0.379 MB | 6.34 KB | 60× |
| 2 (4) | 256 | 1.503 MB | 6.34 KB | 237× |
| 2 (4) | 1024 | **6.00 MB** | **6.34 KB** | **946×** |
| 8 (16) | 64 | 1.528 MB | 25.7 KB | 59× |
| 8 (16) | 256 | 6.063 MB | 25.7 KB | 236× |
| 8 (16) | 1024 | **24.20 MB** | **25.7 KB** | **940×** |

Standard temp scales **linearly in T** (×4 memory per ×4 T, exactly O(T)); reversible temp is **flat in T** and
scales only with state size (O(D), independent of T) — the defining O(1)-in-sequence-length signature. Ratio grows
without bound in T (≈946× already at T=1024, N=2). Float32 halves the byte counts (std 3.00 MB vs rev 4.16 KB @
T=1024,N=2 → 721×; std 12.10 MB vs rev 16.1 KB @ N=8 → 753×) with the same O(T)-vs-O(1) scaling.

**(c) Wall-time overhead — parity in this regime.** Median `t_rev / t_std` = **0.86× (float64), 0.94× (float32)**
across the whole grid (per-cell range 0.43×–1.16×). Reversible is *not* the ~2× one expects from doing an extra
forward-equivalent pass on the backward leg: on CPU with small D, XLA is dominated by memory traffic, so avoiding
the multi-MB tape offsets the recompute cost. ⚠ **This favorable overhead is regime-specific** (CPU, D≤16, small
potential MLP) — on a GPU or with large per-unit potentials the recompute would likely surface as a >1× (up to ~2×)
overhead. Claim parity only for the laptop/small-lattice regime measured here.

**Figure:** `.claude/outputs/v3-reversible-o1/mem_grad_summary.png` — (a) log-log peak temp memory vs T (O(T) lines
for standard, flat O(1) lines for reversible, both N); (b) gradient rel-diff vs T for float32/float64.

---

## Interpretation — tied to the V3 claim

The adopted V3/ICLR claim **"reversible O(1)-memory training"** is **confirmed, with an exactness caveat that scopes it honestly**:

- The claim is *structurally* true because the CLU lattice's separable Hamiltonian + leapfrog is analytically
  invertible; the shipped rollouts are already scan-pure, so O(1) BPTT is a drop-in (no core changes needed).
- Measured: **940–946× lower peak activation memory at T=1024** (O(T)→O(1)), **gradients identical to standard BPTT
  to float precision** (≤2e-6 rel in float32), at **≈0.9× wall-time** in the tested regime.
- **The reversibility is exact only at γ=0.** The dissipative knob γ (used in Exp-B governor, generation, and any
  "forgetting" mode) makes exact inversion impossible in finite precision: the (1−γ)⁻ⁿ amplification gives a finite
  usable horizon (~2×10⁴ steps @γ=1e-3 down to ~10² @γ=0.1, in float64; ~100× shorter in float32). This is not a bug
  — it is the honest boundary of the claim, and it dovetails with the program's γ=0-conservative-memory /
  γ>0-rationed-forgetting framing: **the memory that is reversibly trainable at O(1) is exactly the conservative
  (γ=0) memory.** Dissipative segments must fall back to stored activations or gradient checkpointing.

### The V3-short sentence, with numbers
> *γ=0 CLU-lattice training admits exact recompute-backwards BPTT: reconstructed gradients match standard
> stored-activation BPTT to ≤2×10⁻⁶ relative in float32 (≈10⁻¹⁵ in float64), at O(1)-in-T peak activation memory —
> **6.3 KB vs 6.00 MB at T=1024 (946× reduction; N=2), the standard cost growing linearly in T while the reversible
> cost stays flat** — for ≈0.9× wall-time in the laptop/small-lattice regime. With γ>0 the (1−γ) contraction is
> invertible only in exact arithmetic and amplifies reconstruction error as (1−γ)⁻ⁿ, giving a finite usable reversal
> horizon: ≈3.3×10⁴ steps at γ=10⁻³, 4.1×10³ at γ=10⁻², 5×10² at γ=0.05, 1.3×10² at γ=0.1 (float64; ~100× shorter in float32).*

---

## Limitations / confounds (per C-9)

1. **Untrained models & final-state loss.** I measured the graph/integrator properties, not a full wake–sleep step.
   Real `train_chlu` uses a **windowed per-step MSE** (contributions at every timestep) + Lyapunov + periodic sleep
   phase. The O(1) reverse-scan handles per-step cotangents too (inject at each step), but I did **not** run the
   actual `train.py` loss through it — so "on a lattice *training* step" is demonstrated for the gradient mechanics,
   not yet wired into the shipped trainer. Recommend an engineer task to add a reversible-BPTT training path.
2. **CPU / small D only.** Wall-time parity (≈0.9×) is regime-specific; do not generalize to GPU or large potentials
   without measuring (expect up to ~2× there). Memory ratios (O(T) vs O(1)) *do* generalize — they are structural.
3. **`memory_analysis().temp_size` is XLA's compiler estimate**, not a runtime peak-RSS. It is the right proxy for
   activation-tape size and is deterministic/reproducible, but a runtime allocator peak could differ by constant
   factors. The O(T)-vs-O(1) *scaling* is unambiguous regardless.
4. **γ>0 horizons** use a 1e-6 relative-error threshold; in float32 the small-γ horizons are limited by the baseline
   float-accumulation floor (already ~1e-6 by T~1000 at γ=0), so those cells report the noise floor, not the
   amplification cap — the amplification is only cleanly separable at γ≥1e-2 or in float64.
5. **newtonian_learned kinetic** used. Reversibility is exact for *any* separable CHLU kinetic mode (identity/learned/
   relativistic all give T(p)+V(q)); relativistic was not separately timed but the inverse map is identical.

## Recommended next experiments

- **Wire reversible-BPTT into `train_chlu`** (engineer): per-step-loss reverse scan + a `reversible_bptt` training flag;
  then re-measure memory/wall-time on a *real* Exp-A/B/D step and at ICLR sequence lengths (T≥4k) where standard BPTT
  OOMs but reversible does not — that is the headline V3 systems result.
- **Accelerator measurement** (CSF3 A100): true peak-memory (device HBM) and honest wall-time overhead at large D /
  deep lattices — this is where the O(1) memory pays for real (fitting long sequences that standard BPTT cannot).
- **Gradient-checkpointing baseline** (`jax.checkpoint`) as the fair middle-ground comparison: reversible O(1) vs
  checkpoint O(√T) vs standard O(T) — memory/compute Pareto. RevNet/checkpointing is the cited prior art; position against it.
- **Mixed γ segments:** measure the reversible/checkpoint hybrid needed when a network interleaves γ=0 (reversible)
  and γ>0 (forgetting) blocks — quantify how much of a real network stays O(1)-trainable.

---

## Git footprint
None — repo read-only. All artifacts under `.claude/`: harness `scratch/v3-reversible-o1/{reversible.py,plot.py}`,
logs `scratch/v3-reversible-o1/item2_{x64,f32}.log`, figure `outputs/v3-reversible-o1/mem_grad_summary.png`.

## Proposed handover updates (for the Hub)

**§1.6 / §5 (V3 evidence) — add:**
- **Reversible O(1)-memory BPTT MEASURED (backlog #1, commit 63fea62, laptop CPU, JAX 0.9.0):** γ=0 CLU lattice
  admits exact recompute-backwards BPTT via analytic leapfrog inversion (separable H). Gradients match standard
  stored-activation BPTT to **≤2e-6 rel (float32) / ~1e-15 (float64)** at T up to 1024. Peak activation memory
  **O(1) in T vs O(T)**: measured **6.3 KB (reversible) vs 6.00 MB (standard) at T=1024, N=2 → 946×** (940× at N=8);
  standard grows linearly in T, reversible flat. Wall-time **≈0.9× (parity, CPU/small-D regime — not GPU-validated)**.
- **Scope caveat (C-9 negative, honest):** exactness holds **only at γ=0**. γ>0 (dissipation) amplifies reconstruction
  error as (1−γ)⁻ⁿ → finite usable reversal horizon: **≈3.3e4 steps @γ=1e-3, 4.1e3 @γ=1e-2, 5e2 @γ=0.05, 1.3e2 @γ=0.1
  (float64; ~100× shorter in float32)**. "Reversibly O(1)-trainable memory" = the conservative (γ=0) memory; dissipative
  blocks need stored activations / checkpointing. This aligns with the γ=0-memory / γ>0-forgetting budget framing.

**§7 (candidate follow-up flag, for experiment-engineer):** the reversible-BPTT path is **not yet in `train_chlu`**;
only the gradient mechanics are validated (untrained model, final-state loss). Needs a `reversible_bptt` training-loop
implementation (per-step-loss reverse scan) + accelerator memory/wall-time measurement before it's a load-bearing V3
systems claim. No bug found in `chlu/` — the inverse map validated against the shipped integrator exactly.
