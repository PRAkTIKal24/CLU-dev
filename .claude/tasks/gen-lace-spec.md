# Task: gen-lace-spec — pin LACE's implementation constants and audit its transfer to conv features

- **Agent:** `web-scout` · **Output:** `.claude/outputs/gen-lace-spec.md` + **`.claude/outputs/gen-lace-spec/lace_spec.json`**
- **Read first:** `.claude/AGENT_PROTOCOL.md` **§5 and §6** only. ⛔ Read nothing else from the archive.
- **Repo:** read-only. **Zero worktrees, zero branches, zero tracked-code edits.**
- **Precondition:** none. Spawnable now.
- **Scope note (2026-09-09):** the Advisor has already read LACE and settled what it *is* and what it means for
  positioning. ⛔ **Do not re-derive any of that.** It is given to you as fact below. Your job is only the part
  the read did not cover: the numbers you need to *build* it, and whether it survives the move to our features.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument / spec recon for a competitor arm.*
- **Laundering control:** N/A (no measurement).
- **Falsifies:** N/A.

## Given (Advisor read, treat as established — do not re-verify, do not contradict without hard primary evidence)
LACE = attribute classifiers trained on the latents of a **frozen StyleGAN**; the energy is **classifier logits plus a
Gaussian prior**; sampling is an **ODE** on that energy; **AND/OR/NOT compose classifier energies**. It requires **full
attribute labels** (≈50k labelled pairs on CIFAR, ≈10k on FFHQ). It has **no refusal, no invalid-combination detection,
no deletion, and no persistent or enumerable concept store.** Its reported unseen-combination accuracy is **0.94–0.98**
at up to 1024². Paper: Nie, Vahdat, Anandkumar, NeurIPS 2021, **arXiv:2110.10873**.

## The one job
Return the constants and the transfer verdict needed to implement LACE as a **strong** baseline arm on **our frozen
conv feature maps** — not on a StyleGAN latent space.

## Why, and the discipline that governs it
⭐ **Anti-hobbling is binding.** This is the arm nearest our own slot (frozen backbone + composed energy). A weak LACE
that loses is worthless evidence; a LACE that beats us is the control working and gets reported unsoftened.

## What the JSON must contain
1. **Classifier head.** Architecture, width/depth/activation, how attribute labels are consumed, training loss,
   optimiser and schedule, as published.
2. **The energy.** Exact form combining logits and the Gaussian prior — verbatim equation with a locator — including
   every weighting/temperature constant.
3. **The sampler.** ODE formulation verbatim; solver, tolerances, step count, step size, noise schedule if any.
   Record the Langevin variant if given, and **state explicitly whether any Metropolis/MH correction is applied.**
   Also record Du et al. ICML 2023 (arXiv:2302.11552)'s precise claim that the *sampler*, not the model, is what
   makes composed-energy generation fail, and the conditions under which an uncorrected sampler is defensible.
4. **Composition operators.** AND / OR / NOT as implemented, with every α / temperature constant.
5. **Official code.** Repo URL, **commit hash**, files implementing the energy and the sampler, and published
   hyperparameters as they appear in config — each marked `PAPER:` or `OFFICIAL IMPLEMENTATION:`.
   ⛔ Never inherit a library default and present it as the paper's value.
6. ⭐ **The transfer audit — the deliverable that matters most.** LACE's Gaussian prior is meaningful over a trained
   StyleGAN latent with a known base distribution. **We have frozen conv feature maps with no such prior.** For each
   component state whether it is **latent-space-agnostic** or **assumes the StyleGAN prior**, and for everything in the
   second class name the **least-hobbling substitute** you can defend (e.g. a fitted Gaussian / whitening over cached
   train features — say which, and what it costs). Be pessimistic; this is where the arm gets quietly crippled.
7. **Supervision ledger.** State plainly what LACE must be given that our store is not: full factor labels over the
   train split, and how many. We will declare this asymmetry in the prereg rather than hide it.

## Acceptance
- Items 1–7 answered from primary, or **NOT OBTAINED** with one line on what you tried. Marked partial coverage is a
  success; a confident guess is a failure.
- Every equation verbatim with a locator; every hyperparameter carries `PAPER:` or `OFFICIAL IMPLEMENTATION:`.
- The transfer audit names each deviation and its strongest defensible substitute.

## What would prove this spoke wrong
- You specified an arm that cannot run without a StyleGAN and did not say so.
- ⛔ **You treated 0.94–0.98 as a bar our numbers must clear.** It is not: it is StyleGAN at 1024² with full attribute
  labels on their splits. Only **our** re-implementation, on **our** features and **our** splits, is comparable.
  If you find yourself transferring their number, stop.
