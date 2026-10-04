# Task: v3-reversible-o1 — measure reversible O(1)-memory BPTT on the lattice (w7; backlog #1, adopted Thread-7(1), never yet run)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v3-reversible-o1.md`
- **Read first:** protocol · roadmap v0.3 backlog #1 · `v3-lattice-build.md` (scan-pure rollouts = reversibility preserved by design) · brainstorm Thread 7(1) (RevNet/momentum-net lineage cited there).
- **Why now:** the V3 short is expected to GO at this review; "reversible O(1)-memory training" is an adopted V3/ICLR claim that has never been measured. Repo read-only; scratch in `.claude/scratch/v3-reversible-o1/`.

## Items
1. **Exactness:** at γ=0, reconstruct the forward trajectory by integrating backwards from the final state (invert the leapfrog exactly); measure reconstruction error vs trajectory length (expect float-accumulation only, no structural drift). Include a γ>0 arm to show where/how reversibility degrades (known: (1−γ) contraction is invertible in exact arithmetic but amplifies float error as (1−γ)^{-n} — measure the usable horizon vs γ).
2. **The memory–compute trade, measured:** recompute-backwards BPTT gradient vs standard (stored-activations) BPTT gradient on a lattice training step — (a) gradients agree to float tolerance; (b) peak memory vs sequence length (expect O(1) vs O(T)); (c) wall-time overhead factor. N ∈ {2, 8}, T ∈ {64, 256, 1024}. 3 seeds.
3. Deliverable: the V3-short sentence with numbers — *"γ=0 lattice training admits exact recompute-backwards BPTT: gradients match to [tol], peak memory O(1) vs O(T) (measured X vs Y at T=1024), at Z× wall-time overhead; with γ>0 the usable reversal horizon is [n(γ)]."* Negatives (if the overhead is ugly or the γ>0 horizon is short) fully written per C-9 — they scope the claim honestly.

Flag-provenance per §5. Laptop-scale; if T=1024 at N=8 exceeds ~1h, report the reduced grid rather than waiting.
