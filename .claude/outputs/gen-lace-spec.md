# gen-lace-spec — web-scout report

Task + acceptance criterion: pin LACE's implementation constants from primary sources and audit whether the method transfers to frozen conv feature maps; deliver `lace_spec.json` with items 1–7 answered from primary or marked NOT OBTAINED.
Status: **done** (items 1–7 all answered from primary; 3 minor NOT-OBTAINEDs logged, all low-impact).

**⭐ DIAL DECLARATION (echoed):** Dial: *none — instrument / spec recon for a competitor arm.* Laundering control: N/A (no measurement). Falsifies: N/A.

**⚠ RECONCILIATION LIST — needs an owner (protocol §5 corollary, stated in first 10 lines):** three items below require a Hub decision or a downstream task, they are **not** self-resolving: **(R1)** the Hub must choose feature-space vs image-space evaluation, because LACE **cannot render images from conv features without a decoder** (transfer audit, generator row); **(R2)** the prereg must declare two concessions granted to LACE (dense factor labels + a validation split for tuning OR/NOT constants); **(R3)** the LD arm must be run at ≥2 temperatures because the published constants are ~200× colder than the target posterior (derived below).

---

## Answer first

LACE ports to frozen conv features **almost entirely intact** — the classifier head is literally a 1×1-conv stack, and the energy and the AND/OR/NOT algebra are pure functions of logits. There is **exactly one** hard StyleGAN dependency in the maths: the `½‖z‖²` term is not a modelling choice but the *exact* log-density of StyleGAN's z-space (N(0,I) by construction), and it is what licenses the ODE drift to omit the prior score entirely. The least-hobbling substitute is to **fit a shrinkage Gaussian over cached train features and run the whole method in whitened coordinates**, which preserves LACE's VP-SDE invariance argument exactly; a bare `½‖x‖²` on raw post-ReLU features would be a silent crippling and must be refused. Separately, LACE **needs a StyleGAN to produce images at all** — the arm is generator-free only if our protocol is scored in feature space.

Full spec: **`.claude/outputs/gen-lace-spec/lace_spec.json`** (every constant tagged `PAPER:` / `OFFICIAL_IMPLEMENTATION:` / `DERIVED:`).

---

## Sources pinned

- Nie, Vahdat, Anandkumar (2021), "Controllable and Compositional Generation with Latent-Space Energy-Based Models", **NeurIPS 2021** (peer-reviewed), arXiv:2110.10873. OpenReview `kcI3T5qe1jr`.
- Official code: **`https://github.com/NVlabs/LACE` @ `dcfd4a57bee40fcb17ac8304413f1e35a64323cd`** (2021-12-01, "update readme"). Repo has **5 commits total**, none since 2021-12-01 — HEAD *is* the paper release. NVIDIA non-commercial license.
- Du, Durkan, Strudel, Tenenbaum, Dieleman, Fergus, Sohl-Dickstein, Doucet, Grathwohl (2023), "Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC", **ICML 2023**, arXiv:2302.11552.

---

## Findings that change how we build the arm

### F1 — The prior is exact in LACE and absent in the ODE drift (the crux)
`VPODE.forward` calls `ccf.get_cond_energy(...)`, **not** `ccf(...)`. The `½‖z‖²` term therefore never appears in the ODE vector field:
```
beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
cond_energy_neg = self.ccf.get_cond_energy(z, self.ys, ...)
cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
dz_dt = -0.5 * beta_t * cond_f_prime
```
`OFFICIAL_IMPLEMENTATION:` FFHQ/sampling.py. This omission is licensed *solely* by Appendix A.2: `PAPER:` "distribution of latent z in most generative models satisfies p₀(z)=N(0,I), diffusing it with VP-SDE will not change distribution at time t … p_t(z(t))=N(0,I)". Note this holds even for the "w-space" variant, because CCF **samples in z and maps through `g: z→w`** inside `classify()` — so the prior is exact there too. The **SGLD** arm, by contrast, calls `ccf.forward` and *does* include `− ‖z‖²·0.5`. **Consequence:** in non-whitened feature coordinates the prior score does not cancel and the shipped drift is the *wrong* vector field. Whitening is not a nicety; it is what keeps the ODE arm valid.

### F2 — `DERIVED:` The published LACE-LD is ~200× colder than its own target
`PAPER:` Eq. (2) is `x_{t+1} = x_t − (η/2)∇E + ε, ε~N(0,ηI)`. `OFFICIAL_IMPLEMENTATION:` the code does `z_k.data += sgld_lr * f_prime + sgld_std * randn_like(z_k)` with `--sgld_lr 1e-2 --sgld_std 1e-2 --n_steps 100`. Matching drift gives η = 0.02, hence the Langevin-consistent noise is √0.02 ≈ **0.1414**, but the shipped std is **0.01** — 14.1× too small. Effective temperature `T_eff = std²/(2·lr) = 1e-4/0.02 = 0.005 ≈ 1/200`. So LACE-LD is near-deterministic noisy gradient ascent on the energy and will not cover modes. **This is my arithmetic, not an authors' claim** — but both inputs are primary and the algebra is elementary. Anti-hobbling: run (a) as-published, (b) `std = √(2·lr)`, (c) optionally annealed MALA; report LACE's best.

### F3 — No Metropolis correction anywhere, and that is *defensible for the ODE arm*
No occurrence of "Metropolis"/"Metropolis-Hastings" in arXiv:2110.10873; no accept/reject logic in either `sampling.py`. Du et al. 2023 state `PAPER:` "*We conclude that the sampler (not the model) is responsible for this failure and propose new samplers, inspired by MCMC, which enable successful compositional generation*", with the mechanism being that for a product of **diffused** models, "*For t>0, this quantity is not equal to the sum of the scores of the two models*" (§4). **Scope note (my reading, flagged):** LACE structurally sidesteps that failure — it never diffuses the classifiers, only the N(0,I) prior, which is VP-SDE-invariant, so composition is applied to *undiffused* log-densities at every t. A deterministic probability-flow ODE derives validity from that invariance, not from detailed balance, so "no MH" is not per se a defect here. Du et al.'s critique *does* bite if we (i) run the mistuned LD sampler, or (ii) break the N(0,I) invariance. On unadjusted samplers generally, Du et al. §4.2: "*While they can perform well in practice, there exist many distributions from which they cannot generate decent samples*."

### F4 — Paper↔code reconciliations (all AGREE; two are non-obvious)
- **Architecture.** `PAPER:` App. A.3 "four-layer MLP … Linear 384 / 256 / 128 / #logits". `OFFICIAL_IMPLEMENTATION:` `DenseEmbedder(input_dim=512, up_dim=128, depth=4)` builds `np.linspace(512,128,4).astype(int) = [512,384,256,128]` of **1×1 Conv2d** + LeakyReLU(0.2). A 1×1 conv on a 1×1 map *is* a Linear. **They are the same network.**
- **Composition constants.** `logical_comb` operates on *negative* energies; the OR branch is `log(exp(a₁)*20 + exp(a₂))`, and negating gives exactly `PAPER:` `E_OR = −log(e^{β−E₁} + e^{−E₂})` with **β = ln 20**. NOT is `clamp(0.1/|·|, 0, 1)`, matching **α = min(0.1/|E(z,c₂)|, 1)**. Exact match both ways.

### F5 — Composition operators have three properties the paper doesn't advertise
All `DERIVED:` from `FFHQ/utils.py::logical_comb` (verbatim in the JSON):
1. **OR is not commutative** — `cof=20` multiplies only the left operand, so `'0*1' ≠ '1*0'`, and chained ORs re-apply ×20 to the running accumulator each fold.
2. **Operands are single characters** (`c.isnumeric()`), so **at most 10 operands** per expression. Hard structural limit.
3. **α is scale-coupled** — `min(0.1/|E₂|,1)` is calibrated to *their* energy magnitudes. On our features, if |E| ≫ 0.1 the NOT operator attenuates to ≈0.1/|E| and becomes weak. The authors' own comments are literally `# these values work well` (cof) and `# this value works well` (σ=0.1). These are tuned constants, not laws.

### F6 — Hyperparameter conflicts worth knowing before we copy anything
- **Epochs:** shipped `run_clf.sh` uses `--n_epochs 50` (both CIFAR-10 and FFHQ); App. A.3 as rendered says "100 epochs"; argparse default is 200. Use **50**.
- **The "staircase decay schedule" never fires.** `--decay_epochs [160,180] --decay_rate .3` are argparse defaults; at 50 epochs the milestones are unreachable. Effective schedule = **1000-iter linear warmup then constant lr=1e-4**.
- **Batch size:** scripts use 128 (FFHQ) / 512 (CIFAR-10); the argparse default is 64 and is *not* what was run. ⛔ This is precisely the "library default presented as the paper's value" trap.
- **σ inconsistency:** the continuous *energy* divides by σ²=0.01 (100× sharpening) but the continuous *training loss* (`utils.get_loss`) is plain `0.5·L2` with no σ.
- **`dis_temp` is not a clean temperature:** `logits/dis_temp` is applied to the gathered target logit but **not** inside `logits.logsumexp(1)`. Default 1.0 everywhere, so it never bit them — but don't sweep it.

### F7 — Supervision ledger, stated honestly
LACE needs a **dense factor-label matrix** (one label per attribute per sample) over a **closed vocabulary fixed at classifier-training time**. Counts, `PAPER:` — CIFAR-10: "*50k (z,c) pairs and 50k (w,c) pairs*", labelled by "*pre-trained DenseNet with error rate 4.54%*"; FFHQ: "*10k (w,c) pairs created by StyleFlow paper*", 12 attributes (`yaw smile gender age glasses light3 pitch haircolor beard bald light0 width`, from `run_clf.sh`); MetFaces: FFHQ-pretrained image classifier as labeller; AFHQ-Cats: CLIP prompts. Validation split `n_valid=500`.
⚠ **Do not write "LACE needs 50k human labels."** Every dataset uses *machine* labels from a teacher, so a labelling-cost argument is weak and would be an overstatement. The load-bearing asymmetry is **structural**: closed vocabulary, new concept ⇒ new labelling pass + new head, and **no deletion, no refusal, no invalid-combination detection, no enumerable store, no admission decision** — every query is answered.

---

## Transfer audit — verdict per component

| Component | Class | Least-hobbling substitute |
|---|---|---|
| DenseEmbedder classifier | agnostic | Runs **natively** on C×H×W (it's already 1×1 convs). Only change: `squeeze(-1).squeeze(-1)` → `mean(dim=(-2,-1))`; `input_dim=C`. Cost: none. Also report a flattened-`C·H·W` variant in case pooling kills the signal. |
| Conditional energy (log-softmax / scaled L2) | agnostic | Port verbatim incl. σ=0.1 and the sign convention. |
| AND/OR/NOT algebra | agnostic in form, **scale-coupled in constants** | Port operators verbatim; **tune `cof` and the `0.1` in α on a val split** (e.g. cof∈{5,20,80}, α-num∈{0.03,0.1,0.3,1.0}). Freezing them at 20/0.1 is the easiest way to hobble OR/NOT. Declare as a concession. |
| **`½‖z‖²` Gaussian prior** | **ASSUMES StyleGAN prior** | **Fit μ, Σ (Ledoit-Wolf / ridge shrinkage) over cached train features; sample in `u = Σ^{-1/2}(x−μ)`; decode before the classifier.** Then p₀(u)=N(0,I) *exactly* and App. A.2's invariance — hence the ODE drift — survives. Cost: one cached forward pass + one D×D eigendecomposition; **shrinkage mandatory if D>N**; infeasible above D≈2e4 → fall back to **channel whitening** (C×C over space+batch) and declare it. ⛔ Rejected: bare `½‖x‖²` on raw features (post-ReLU, non-negative, anisotropic → pulls off-manifold, derivation false); a learned flow prior (breaks the invariance identity, stops being LACE). |
| VP-SDE / probability-flow ODE | **conditionally** agnostic | Valid **only** in whitened coordinates. Then port `β∈[0.1,20]`, `T=1`, `dopri5`, `atol=rtol=1e-3`, `integration_times=linspace(1,1e-3,2)`, `use_adjoint=True` verbatim. Log realised NFE (`vpode.n_evals` is already printed) for an honest compute comparison. `PAPER:` reports ≈41.8 NFE at their ACC≥95%/FID≤10 point. |
| Langevin `sample_q_sgld` | conditionally agnostic + **temperature defect** | Whitened coords; run all three temperatures (F2). |
| **Generator `g` + truncation 0.7** | **REQUIRES a StyleGAN** | **(A) feature-space scoring ⇒ `g := identity`, drop truncation, arm runs end-to-end on cached features, no generator needed.** **(B) image-space scoring (FID / visual) ⇒ a feature→pixel decoder is a hard requirement**; LACE cannot render from conv features without one, and the LACE arm then acquires a decoder our arm may not have. **Hub must choose (R1).** Note: the repo's `x='i'` "pixel space" flag only relocates the *classifier* onto `g(z)`'s output — sampling is still in StyleGAN z-space. It is **not** a generator-free mode. |

**Runnability verdict:** LACE **is** runnable on frozen conv feature maps with no StyleGAN, *iff* (a) sampling happens in whitened feature coordinates from a fitted Gaussian, and (b) evaluation is feature-space. Under image-space evaluation it is **not** generator-free.

**Minimum deviation list** (what the prereg must record): fitted-and-whitened Gaussian prior (required, not optional) · global-avg-pool before the head · `g:=identity`, no truncation · OR/NOT constants re-tuned on val (concession to LACE) · LD reported at published + corrected temperature (concession to LACE).

---

## Numbers I am NOT transferring
Recorded in the JSON under an explicit `DO_NOT_USE_AS_A_BAR` key so nobody re-searches them: CIFAR-10 LACE-ODE FID 6.63±0.06 / ACC 0.972±0.001; FFHQ-glasses FID 20.93 / ACC 0.998; unseen-combination 0.94–0.98. **These are StyleGAN at up to 1024² on their splits with their labels and are not a target for our re-implementation.** Only our re-implementation on our features and our splits is comparable.

---

## Confidence & gaps

**Verified (two independent endpoints — paper text *and* official source):** energy form and sign convention; β=ln20 / α=min(0.1/|E|,1); β_min=0.1, β_max=20, T=1; dopri5 + atol/rtol 1e-3; the four-layer 384/256/128 head; absence of MH; the labelled-pair counts.
**Single-sourced (ar5iv rendering of arXiv:2110.10873):** equation *numbers* (Eq. 5 energy, Eq. 7 ODE, Eq. 2 Langevin, §3.3 composition) — content cross-verified, **numbering not**; the ≈41.8 NFE figure; the "100 epochs" appendix claim (contradicted by the shipped script's 50).
**My derivations, flagged as such:** the ~200× LD temperature deficit (F2); OR non-commutativity and the 10-operand limit (F5); the prior-absent-from-ODE-drift consequence (F1); the Du-et-al.-doesn't-directly-apply-to-LACE's-ODE scope note (F3).

**NOT OBTAINED (3, all low-impact, logged in JSON §9):** (i) verbatim App. A.3 block to settle 50-vs-100 epochs — `arxiv.org/pdf/…v2` exceeded the fetch size limit and OpenReview served a browser-verification interstitial; the shipped script is authoritative anyway. (ii) `CIFAR10/latent_model.py`'s `x_space=='cifar10_i'` branch and the generator wrapper — the retrieved excerpt held only `DenseEmbedder`/`get_norm`; branching is likely in `main.py`/`networks.py`; not needed for our arm. (iii) equation numbers cross-checked against the published PDF — **re-verify before citing an equation number in a paper.**

**What to search next, if the Hub wants it:** (1) whether any 2023–2026 follow-up runs LACE-style latent EBMs on *non-generative* frozen backbones (would give us a precedent for the whitening substitute rather than us inventing it); (2) Du et al.'s annealed-MALA hyperparameters, if we decide to grant LACE the corrected sampler.

---

## Bibtex-ready refs

```bibtex
@inproceedings{nie2021lace,
  title     = {Controllable and Compositional Generation with Latent-Space Energy-Based Models},
  author    = {Nie, Weili and Vahdat, Arash and Anandkumar, Anima},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021},
  eprint    = {2110.10873},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note      = {Code: https://github.com/NVlabs/LACE, commit dcfd4a57bee40fcb17ac8304413f1e35a64323cd}
}

@inproceedings{du2023reduce,
  title     = {Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and {MCMC}},
  author    = {Du, Yilun and Durkan, Conor and Strudel, Robin and Tenenbaum, Joshua B. and Dieleman, Sander and Fergus, Rob and Sohl-Dickstein, Jascha and Doucet, Arnaud and Grathwohl, Will},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  eprint    = {2302.11552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

Git footprint: **none** — read-only spoke, zero branches, zero worktrees, zero tracked-file edits. Only `.claude/outputs/gen-lace-spec.md` and `.claude/outputs/gen-lace-spec/lace_spec.json` written.

---

## Open questions / follow-ups / risks
1. **(R1, blocking arm design)** Feature-space or image-space evaluation? Image-space forces a decoder into the LACE arm and changes what the comparison means.
2. **(R3)** Confirm we will run LACE-LD at ≥2 temperatures. Refusing this leaves us open to "you ran their sampler at 1/200 the correct temperature and it under-covered."
3. Is D = C·H·W small enough for a full-covariance whitening fit? If not, the arm drops to channel whitening and that approximation must appear in the prereg, not in a footnote.
4. σ=0.1 in the energy vs no σ in the training loss: do we reproduce their inconsistency (faithful) or fix it (charitable)? Recommend faithful-plus-ablation.
5. Do any of our concepts need >10 operands in one expression? If so `logical_comb` must be extended — a re-implementation choice we should state rather than silently make.

## Proposed handover updates (for the Hub)
- LACE baseline arm is **specified and runnable** on frozen conv features under feature-space evaluation; spec at `.claude/outputs/gen-lace-spec/lace_spec.json`, code pinned to `NVlabs/LACE@dcfd4a5`.
- Record the **single hard StyleGAN dependency**: the `½‖z‖²` prior ⇒ mandatory fitted-Gaussian whitening of cached train features; and the **decoder blocker** if images are needed.
- Record the two **declared concessions to LACE** (dense factor labels over the train split; val split for OR/NOT constant tuning) and the **LD temperature caveat** for the prereg.
- Reported LACE numbers (0.94–0.98, FID 6.63, etc.) are quarantined in the JSON as *not a bar*; ensure no downstream doc adopts them as a target.
