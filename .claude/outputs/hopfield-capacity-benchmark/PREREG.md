# PREREG — hopfield-capacity-benchmark (written BEFORE running the harness)

Commit at prereg time: base `main @ 8519df6`. Branch `agent/experiment-engineer/hopfield-capacity-benchmark`.
Author: experiment-engineer. Seeds to be used: 0 (primary), plus 1,2 if compute allows.

## Protocol I am matching (from the actual repos, verbatim)
- **U-Hop** `MAGICS-LAB/UHop @ cdac754` (`memory_retrieval.py`, `functions.py`, `data.py`).
  - Store `m_size` images reshaped to `(m_size, D)`, pixels in `[0,1]` (torchvision `ToTensor`).
  - Query per stored pattern `x`: `q = torch.dropout(x, p=0.5, train=True)` — **randomly zero 50% of pixels AND scale survivors by 1/(1-p)=2×** (this is the exact "half-mask", NOT mask-to-zero).
  - Update: `score = beta * activation(overlap(Xi, x)); x = Xi @ score`, default `beta=1, steps=1, overlap=dot_product`.
  - Activations available: `softmax` (dense MHN = Ramsauer), `sparsemax`, `entmax15`, top-k, poly. Dense=softmax; the sparse SOTA line = sparsemax / entmax15.
  - **Success metric = `sqdiff` = Σ(clamp(x,0,1) − clamp(x_new,0,1))², mean over all stored-as-query patterns.** LOWER is better. (⚠ the scout's "cosine>0.9" is NOT what the repo computes — the repo reports mean squared pixel error. I match the repo metric and ALSO report cosine + identity-accuracy for legibility.)
  - Noise sweep (`memory_retrieval_noise.py`): `q = clamp(|x + N(0,noise_level)|, 0, 1)`, report mean sqdiff.
- **Ramsauer** `ml-jku/hopfield-layers @ f56f929`: energy `-lse(β Xᵀx)+½|x|²+...`, update `x=X softmax(βXᵀx)` — identical to U-Hop MHN-softmax. Confirms the dense arm.

## Arms
1. Dense modern-Hopfield (softmax), repo-verbatim (β=1, 1 step) AND a tuned/iterated variant (β swept, steps>1) so I do not strawman it.
2. Sparse SOTA line: sparsemax (and entmax15 if the JAX port lands), same update.
3. CLU designed register: `GaussianMemoryPotential` (patterns = Gaussian well centers in D-dim), retrieved by the **actual damped velocity-Verlet CHLU rollout**, read the settled q. NOTHING learned (well centers = the stored images; width s set by a fixed data-driven rule, NOT tuned per load).
4. Nearest-neighbour in pixel space — the floor.

## Predictions (committed; will be scored in the report)
Primary axis: **identity-retrieval accuracy** (argmin over stored patterns of ‖output−ξ_i‖ equals the true index) vs number stored M; and **mean sqdiff** (repo metric).

P1. **Dense softmax-Hopfield at repo-default β=1 in D≈784/3072 is near-degenerate**: βXᵀq with β=1 and inner products of order Σx² gives an almost-flat softmax → output ≈ mean pattern → identity-acc collapses fast (predict acc < 0.3 by M=100 on MNIST). *This is the repo's literal default and I report it as such, but flag that it under-serves Hopfield.*
P2. **A tuned Hopfield (β≈1/⟨x,x⟩ scale, or the Ramsauer 1/√D convention) recovers**: predict tuned-dense-Hopfield identity-acc > 0.9 at M=100 MNIST, degrading with M.
P3. **CLU Gaussian register ≈ nearest-neighbour floor**: a settled particle in a localized Gaussian well is (to leading order) the argmin well, so predict **|acc_CLU − acc_NN| ≤ 0.05** across the MNIST load sweep. ⇒ CLU **AT** the NN floor.
P4. **Ordering verdict (the deliverable), MNIST identity-acc at high load (M≈500):** `NN ≈ CLU ≳ sparsemax ≳ tuned-dense-Hopfield ≫ dense-β1-Hopfield`. I predict CLU is **AT OR ABOVE** dense modern-Hopfield and **AT** the NN floor; I predict CLU is **NOT above** a well-tuned sparse Hopfield by more than noise. So the honest expected headline: **a designed CLU register matches the sparse-Hopfield SOTA on capacity and beats the dense β=1 line, but does not exceed the SOTA — its novelty is the mechanism (settling in a Hamiltonian landscape = spherical-code packing, `Δ_req`), not the exponential.**
P5. **Retry differentiator:** a second boosted relaxation pass recovers a SMALL fraction of first-pass CLU misses — predict **+2 to +8 pp** identity-acc at the load where CLU first dips below 0.9, at ~2× compute. Keep the arm only if the lift ≥ +2pp and exceeds its blank (retry-on-already-correct doesn't flip them).
P6. **Cross-over (acc<0.9 criterion), MNIST:** predict dense-β1 crosses first (M≲32), then tuned-dense/sparse/CLU/NN all cross together in a band around M≈ a few×D-independent value set by pattern overlap.
P7. **CIFAR-10:** same ordering but all methods cross earlier (CIFAR patterns are less separable in pixel space) — predict CLU still AT the NN floor.

## Kill rules (drop a differentiator if it shows no performance edge)
- Fiber payload: keep only if CLU recovers per-item payload bits at an accuracy the Hopfield fixed point provably cannot (report the bit-count with (σ_read,N,launches)); else drop to "capability note".
- Retention control: this is a *capability* Hopfield has no time-axis for; report as a demonstrated capability, not an accuracy number, unless it changes a benchmark curve.
- Retry: kill unless P5 holds.

## Honesty commitments
- Report per-dataset, never averaged. Report where CLU wins/ties/loses plainly.
- Do NOT claim the exponential capacity as novel (Demircigil 2017; Hu-Wu-Liu 2024 = optimal spherical codes).
- Do NOT hand-tune the CLU landscape per load; s fixed by one data-driven rule; if I tune, show untuned alongside.
- Report the exact protocol + repo hashes and every non-default flag in a provenance table.
