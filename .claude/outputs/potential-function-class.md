# potential-function-class — experiment-engineer report

**Task + acceptance criterion:** is the w20 learned-landscape failure **EXPRESSIVITY** (H-EXPR) or **SUPPORT STRUCTURE** (H-SUPP)? — four function classes at matched parameters, ≥5 seeds, strict retrieval + blank + cross-write interference + measured support decay, the design-freedom re-run, and cost.
**Status: done.**

> ⚠ **HEADLINE, stated as the task demands: the data supports NEITHER H-EXPR NOR H-SUPP as stated.**
> **No learned class clears strict 0.9 at both K=4 and K=8** (7 classes, 5 seeds) — so the task's "if all learned classes fail, that is a strong program-level result" is the outcome: **the function class is not the problem.** But the class is *not* irrelevant either: it moves fidelity a lot (K=4 0.853→0.998) and interference by **340×**. And **H-SUPP's specific prediction that attention fails at least as badly as an MLP is REFUTED** — a sharp modern-Hopfield potential corrupts **27× less** than the MLP at identical parameter count. The decisive lever is not the class and not capacity: it is **the write operator** (see §3, 70× at identical class, parameters and budget).

> **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary).** Three items:
> 1. **w20's "the loop needs essentially ALL of the designed structure" survives, but its evidential status changes.** With a better learned family, rung 1 (`skeleton_residual`) goes from **0.903 ± 0.101 / 0.959 ± 0.043 (failing)** to **1.000 ± 0.000 / 0.986 ± 0.005 (passing, 20× less seed variance)**. The *point* barely moves; the *curve* moves a lot. Any text quoting w20's ladder numbers as "what a learned landscape can do" must now say "what an **MLP** learned landscape can do".
> 2. **w20's `local_rbf` failure (0.623 ± 0.330 / 0.348 ± 0.117) is an INITIALISATION artifact, not a property of atom dictionaries.** Same basis, fixed amplitude parameterisation and flat start → **0.980 ± 0.024 / 0.741 ± 0.019**. Do not cite w20's rung 3 as evidence about locality.
> 3. **The theorist's C3 bound is now measured across 6 function classes and 4.6 decades** (§4) and predicts drift to within a factor 1.6 for 5/6 arms. It also needs a scope correction: it must be evaluated at the **relaxed fixed point**, not the read's launch point.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| commit | `e06d69c` (branch `agent/experiment-engineer/potential-function-class`, base local `main` @ `31c3e15`) |
| worktree | `../CHLU-potential-class`; **main venv reused** (protocol §4 w6 lesson), **JAX 0.9.0**, no worktree `uv sync` |
| harness | `chlu/experiments/exp_potential_class.py`, run as `python -m chlu.experiments.exp_potential_class` (defaults, no overrides), exit 0, **~24 min** |
| seeds | item 1 & ladder **0–4 (5 seeds)**; interference **0–6 (7 seeds)**; support/C3 **0–2 (3 seeds)** |
| kinetic mode | **`newtonian_learned`**, inertia = 1 (inherited from w20 `model_for`) |
| potential | `DesignFreedomPotential`, rung `free_mlp` (freedom 4) with `learned_family ∈ {mlp, hopfield, attn, atoms}`; rung `designed` = the w19/w20 ceiling |
| matched params | **4481 / 4480 / 4480 / 4479 / 4480** (target 4481 = `PotentialMLP(3, hidden=64)`), max deviation **0.045 %** ≤ 5 % |
| write objective | `training/train_memory.py`, unchanged from w20: 600 Adam(w) steps, lr 3e-3, wd 1e-4, n_perturb 32, σ_addr 0.25, σ_pay 0.6, margin 0.15, barrier 0.2. `atoms_local` = **K masked single-item writes**, 600 steps each |
| retrieval | two-phase, γ_address **0.05** × 400 steps → γ_read **0.0** × 800 steps, dt 0.05 (w20 values) |
| queries | 32/item, σ_q = f·0.15, σ_p 0.05, **q2(0)=p2(0)=0 always** (anti-decoration guard) |
| geometry | λ=1.0, f=1.0, barrier 0.2, payload_kappa 1.0, bump_width 0.05, payload_seed 0, dim=3 — **identical to w19/w20** |
| hopfield | β = **2.0** (`hopfield`) and **8.0** (`hopfield_sharp`), α = **0.5** (⇒ exactly the Ramsauer energy), keys init 0.1, n_mem 1120, d_head = dim (W = I) |
| attn | β 1.0, d_head 8, n_mem 495, α 0.05, keys init 0.5 |
| atoms | n_atoms 896, s₀ 0.3, centers ~ N(0,1)³, **depth init 1e-4 with A = amp²**, α 0.05, K contiguous blocks |
| langevin_noise | **N/A** — deterministic Verlet, no Langevin, no temperature |
| artifacts | `.claude/outputs/potential-function-class/{PREREG.md, exp_potential_class_metrics.json, summary.txt, potential_class_fig1_*.png}` |

**Honesty note inherited from w20 and still binding:** in *every* arm the **writer supplies the target sites `c_i`**, and even the freest arm carries a coercivity term. "Free" means free *potential family*, not free structure. Nothing here tests whether item sites emerge.

---

## 0. Headline, in four numbers

1. **Fidelity: the class matters, and it is not enough.** K=4 strict goes **0.853 (MLP) → 0.998 (`hopfield_sharp`) / 0.980 (`atoms`)**; K=8 goes **0.599 → 0.774 / 0.741**. **No learned class reaches 0.9 at K=8.** The `designed` ceiling is 1.000 / 0.986.
2. ⭐ **Interference: the class matters enormously, and H-SUPP's attention prediction is refuted.** Corruption of item A by writing B: `designed` **0.000** · `atoms_local` **2.28e-3** · `hopfield_sharp` **2.91e-2** · `attn` **1.55e-1** · `atoms` **1.61e-1** · `hopfield(β=2)` **5.17e-1** · `mlp` **7.75e-1**. Attention is **not** more global than an MLP; its support is a *tunable* β.
3. ⭐⭐ **The single cleanest result: the WRITE OPERATOR, not the basis, buys locality.** `atoms` vs `atoms_local` are the same class, the same 4480 parameters, the same step budget — only the write differs. Corruption **1.61e-1 → 2.28e-3 (70×)**; strict-A across the second write **0.985→0.899** vs **0.976→0.979 (no loss at all, −0.003)**; fraction of θ moved by write B **1.00 → 0.2500** (= exactly 1/K, verified bit-level, 7/7 seeds).
4. **The mechanism is the C3 ratio, not the support radius.** My pre-registered decay statistic (`r₁₀` from a shell average around the new site) is **unresolvable for 4 of 6 learned arms** and correlates with interference at Spearman **0.143** — falsified. The statistic that works is `‖∇δV(q*)‖ / λ_min(Hess V(q*))` **at the stored items**, which predicts the measured drift to within a factor **1.6 for 5/6 arms across 4.6 decades**.

---

## 1. Matched-parameter table (built and asserted BEFORE any measurement)

| arm | family / write | learned parameterisation | params | dev. vs 4481 | FLOPs (MAC) / `V(q)` eval | write s (K=4 / K=8) |
|---|---|---|---|---|---|---|
| `designed` | — / — | none (w19 hand-built) | **0** | n/a | 0 | 0.0 / 0.0 |
| `mlp` | mlp / global | `PotentialMLP(3, h=64)` | **4481** | 0.0000 % | 4 352 | 1.7 / 1.8 |
| `hopfield` | hopfield(β=2) / global | `n_mem·(d_head+1)` = 1120·4 | **4480** | 0.0223 % | 5 600 | 1.9 / 2.5 |
| `hopfield_sharp` | hopfield(β=8) / global | identical | **4480** | 0.0223 % | 5 600 | 1.4 / 2.3 |
| `attn` | attn / global | `8·3 + 495·8 + 495` | **4479** | 0.0446 % | 5 469 | 1.6 / 2.0 |
| `atoms` | atoms / **global** | `n_atoms·(dim+1+1)` = 896·5 | **4480** | 0.0223 % | 7 168 | 2.6 / 3.1 |
| `atoms_local` | atoms / **local** | identical | **4480** | 0.0223 % | 7 168 | 3.4 / 6.6 |

**P0 confirmed exactly as pre-registered.** Cost (task item 4): no arm is within **1.65×** of another on FLOPs/eval; the only real cost gap is `atoms_local`'s wall-clock at K=8 (**3.7× the MLP**), because a local write is K sequential writes. **No class wins on fidelity at 10× the cost — the expensive arm is also not the best one.**

---

## 2. Item 1 — fidelity, 5 seeds, with a blank control on every cell

`bStrict` = blank strict (leak-immune **value** control); `bClsMax` = blank score under the **strongest** classification read (max of codebook / nearest-centroid) — w20's method finding, enforced here.

| arm | K | **strict** | basin | payload err | bStrict | **bClsMax** | value blank ✓ | class blank ✓ |
|---|---|---|---|---|---|---|---|---|
| designed | 4 | **1.000 ± 0.000** | 1.000 | 6.66e-4 | 0.000 | 0.234 | 5/5 | **5/5** |
| designed | 8 | **0.986 ± 0.003** | 0.986 | 1.63e-2 | 0.000 | 0.144 | 5/5 | **5/5** |
| mlp | 4 | 0.853 ± 0.095 | 0.853 | 1.94e-1 | 0.000 | 0.863 | 5/5 | 0/5 |
| mlp | 8 | 0.599 ± 0.059 | 0.715 | 4.03e-1 | 0.000 | 0.902 | 5/5 | 0/5 |
| hopfield (β=2) | 4 | 0.459 ± 0.017 | 0.966 | 1.43e-1 | 0.000 | 0.384 | 5/5 | 3/5 |
| hopfield (β=2) | 8 | 0.282 ± 0.014 | 0.602 | 4.06e-1 | 0.000 | 0.491 | 5/5 | 0/5 |
| **hopfield_sharp (β=8)** | 4 | **0.998 ± 0.003** | 0.998 | 3.25e-2 | 0.000 | 1.000 | 5/5 | 0/5 |
| **hopfield_sharp (β=8)** | 8 | **0.774 ± 0.029** | 0.774 | 3.22e-1 | 0.000 | 0.933 | 5/5 | 0/5 |
| attn | 4 | 0.509 ± 0.012 | 0.509 | 5.69e-1 | 0.000 | 1.000 | 5/5 | 0/5 |
| attn | 8 | 0.577 ± 0.036 | 0.588 | 4.78e-1 | 0.031 | 0.456 | 5/5 | 1/5 |
| **atoms** | 4 | **0.980 ± 0.024** | 0.980 | 2.00e-2 | 0.000 | 1.000 | 5/5 | 0/5 |
| **atoms** | 8 | 0.741 ± 0.019 | 0.741 | 3.25e-1 | 0.000 | 0.930 | 5/5 | 0/5 |
| atoms_local | 4 | 0.938 ± 0.057 | 0.938 | 5.84e-2 | 0.000 | 1.000 | 5/5 | 0/5 |
| atoms_local | 8 | 0.729 ± 0.048 | 0.754 | 3.39e-1 | 0.000 | 0.848 | 5/5 | 0/5 |

- **The w20 baselines replicate exactly** through the new harness: `designed` 1.000 / 0.986 and `mlp` **0.853 ± 0.095 / 0.599 ± 0.059** — identical to w20's reported values to three decimals. (Pre-registered as the gate: if they had not replicated, the harness would have been at fault and nothing else reportable.)
- **`ANY learned class clears the bar` = False.** The best learned arm is `hopfield_sharp` at min-over-K **0.774**; the designed ceiling is 0.986. **P1 falsified, P1′ (the strong program-level result) confirmed.**
- **Blank controls, as pre-registered (P3 ✓):** the **value** blank passes on **70/70** learned cells (all ≤ 0.031 against a 0.1 bar), so every headline number is leak-immune. The **classification** blank fails on **65/70** learned cells (≈1.000 for four arms) and passes 10/10 for `designed`. **Under a learned V, every classification-based retrieval number remains uninterpretable — w20's method finding replicates across four new function classes and is now a property of learned landscapes generally, not of the MLP.**
- **P2 confirmed (and it is a retraction of a w20 number).** `atoms` = 0.980 ± 0.024 / 0.741 ± 0.019 vs w20's `local_rbf` = 0.623 ± 0.330 / 0.348 ± 0.117. Same Gaussian-atom basis; what changed is the amplitude parameterisation and the flat start (§6). **w20's rung-3 failure was initialisation, not locality.**
- **β is a fidelity knob too, at fixed capacity:** β 2 → 8 takes K=4 from **0.459 → 0.998**. At β=2 the arm addresses well (basin 0.966) but returns the wrong value (payload err 1.43e-1): the memories overlap, so the settled payload is a softmax *blend* of stored values. That is the modern-Hopfield metastable-state regime, visible directly in our loop.

---

## 3. ⭐ Item 2 — cross-write interference: the discriminator (7 seeds)

Write A (3 items on a 4-site ring), re-read A; write B at the fresh site; re-read A.

| arm | **corruption of A** (mean ± std) | median | min … max | strict A before → after | frac. of θ moved by B |
|---|---|---|---|---|---|
| `designed` | **0.000e+00 ± 0.00** | 0.000 | 0.000 … 0.000 | 1.000 → 1.000 | 0.0000 |
| **`atoms_local`** | **2.28e-3 ± 3.52e-3** | **2.72e-5** | 1.03e-6 … 8.32e-3 | 0.976 → **0.979** (**−0.003**) | **0.2500** |
| `hopfield_sharp` | **2.91e-2 ± 7.24e-3** | 2.63e-2 | 2.01e-2 … 4.27e-2 | 0.999 → 0.952 | 1.0000 |
| `attn` | 1.55e-1 ± 1.13e-1 | — | 5.88e-3 … 2.83e-1 | 0.674 → 0.329 | 1.0000 |
| `atoms` | 1.61e-1 ± 1.13e-1 | 1.09e-1 | 5.09e-2 … 3.94e-1 | 0.985 → 0.899 | 1.0000 |
| `hopfield` (β=2) | 5.17e-1 ± 1.84e-2 | — | 4.87e-1 … 5.44e-1 | 0.999 → **0.000** | 1.0000 |
| `mlp` | **7.75e-1 ± 2.53e-1** | — | 4.11e-1 … 1.26e+0 | 0.921 → 0.153 | 1.0000 |

Codebook spacing is 0.667, so corruption ≥ 0.35 is *of order the whole code*: destructive.

**What this settles.**
- **H-SUPP's stated prediction for the transformer arm is REFUTED.** It predicted attention would fail "possibly worse" than the MLP. Measured: `hopfield_sharp` corrupts **26.7× less** than the MLP and is the best *global-write* arm by 5×; even the soft β=2 arm corrupts 1.5× less. **Reason, pre-registered before the run and confirmed:** the modern-Hopfield energy's parameter sensitivity is `∂V/∂k_i = −softmax_i(β⟨q,k_j⟩)·q`, so a memory's influence at a point decays like `exp(−β·gap)` — attention has an explicit *locality knob* that an MLP has no analogue of.
- **The β pair is the clean capacity-vs-support separation** (P5 ✓): identical class, identical 4480 parameters, identical write budget; β 2→8 drops corruption **5.17e-1 → 2.91e-2 (17.8×)**. **Interference is a function of support/temperature, not capacity.**
- ⭐⭐ **But the biggest lever is the WRITE OPERATOR, not the function class.** `atoms` → `atoms_local` is a 70× drop at *everything else identical*, and it is the only arm whose strict retrieval does **not degrade at all** across a subsequent write (−0.003, i.e. within noise of zero). **A local basis is necessary but not sufficient; the write must also be local.** This sharpens the theorist's C3 clause (a): the win comes from *structural/masked* atom writes, not from the atom basis under a free gradient write.
- **P4 half-confirmed, half narrowly failed, reported honestly.** I pre-registered `atoms_local` corruption **< 1e-3**. The **mean is 1.48e-3–2.28e-3** — above my threshold. The **median is 2.72e-5** and 5 of 7 seeds are ≤ 2.7e-4; two seeds (3 and 5) sit at 7.3e-3 / 8.3e-3 and drive the mean. So the *scale* prediction (Gaussian tails at site separation 1.414 with s≈0.3 ⇒ ~1e-5) matches the **median** and under-predicts the tail. The second clause — **≥100× smaller than the MLP** — is confirmed at **340×**.

### Adversarial check (pre-registered, triggered)
`hopfield_sharp` (2.91e-2) beats **`atoms`** (1.61e-1). My PREREG committed to four checks before reporting this:
1. **Did the local mask really freeze the other atoms?** `n_changed = 1120` of 4480 = **exactly 1/4**, on **7/7 seeds**, with a bit-level parameter comparison (not a loss check) — and `tests/test_potential_class.py::test_local_atom_write_leaves_other_items_bit_identical` asserts per-row bit-identity, with a control test that the *global* write does move other blocks. ✅
2. **Is the hopfield arm's blank control valid?** Its **classification** blank fails (1.000) — but the corruption metric is a **value-recovery** metric, whose blank passes 5/5 (bStrict 0.000). The number is leak-immune. ✅
3. **Two extra seeds:** interference re-run at **7** seeds (0–6). `hopfield_sharp` spread is 2.01e-2 … 4.27e-2 — tight, no seed near `atoms`. ✅
4. **Is it a contradiction of theorist C3 / CM-6?** **No, and this matters.** C3's clause (a) is about *dictionary/atom **writes*** — i.e. structural, masked ops — which is `atoms_local` (2.28e-3), and that still beats `hopfield_sharp` by **12.8×**. `atoms` is the atom *basis* under a **free gradient write**, which C3 does not claim to be safe. **The genuinely new statement is: a sharp attention memory is more write-local than a gradient-written atom dictionary, and only a masked write beats it.**

---

## 4. ⭐ Item 2.4 — support, measured: the pre-registered statistic FAILED and found the right one

### 4a. What I pre-registered (shell decay around the new write site) — **falsified**

`rms‖∇V_after(q) − ∇V_before(q)‖` on spheres of radius r about site B, normalised by the curve's peak (⚠ *not* by r→0: a localized write has a bump-shaped influence function with `∇δV = 0` at its own centre; normalising at r→0 gave 10³ ratios in the smoke run and was corrected before the production run).

| arm | r=0.05 | 0.2 | 0.5 | 1.0 | 1.5 | 2.0 | **3.0** | r₁₀ |
|---|---|---|---|---|---|---|---|---|
| designed | 0 | 0 | 0 | 0 | 0 | 0 | **0** | n/a (δV ≡ 0 exactly ✓) |
| mlp | 0.696 | 0.743 | 0.888 | 0.842 | 0.729 | 0.654 | **0.509** | **unresolved** |
| hopfield (β=2) | 1.000 | 0.994 | 0.972 | 0.886 | 0.849 | 0.844 | **0.844** | **unresolved** |
| hopfield_sharp | 0.987 | 0.973 | 0.970 | 0.961 | 0.967 | 0.954 | **0.904** | **unresolved** |
| attn | 0.290 | 0.523 | 0.882 | 0.882 | 0.892 | 0.917 | **0.780** | **unresolved** |
| atoms | 0.189 | 0.279 | 0.743 | 0.974 | 0.938 | 0.212 | **0.024** | 3.00 |
| atoms_local | 0.265 | 0.343 | 0.740 | 0.933 | 0.568 | 0.102 | **3.3e-4** | 2.67 |

**P7 falsified.** `r₁₀` is unresolved for 4 of 6 learned arms (no decay below 10 % within r ≤ 3, which spans the whole ring). Substituting the value at the largest probed radius, Spearman(support, corruption) over the six learned arms = **0.143** — no relationship. **`hopfield_sharp` has the FLATTEST δV profile of any arm (0.904 at r=3) and the second-LOWEST interference.** The shell statistic is not the mechanism, and reporting it alone would have produced a wrong story.

Two facts do survive from it and are worth carrying: `designed` perturbs **exactly nothing** (the zero-learned-parameter control behaves correctly), and the two atom arms are the only ones with a genuinely compact influence function — `atoms_local` falls to **3.3e-4** at r=3, i.e. **2 700× below** the flattest arm.

### 4b. What actually explains item 2 — the theorist's C3 law, measured at the STORED items

Evaluated at the **relaxed fixed point** `q*_i = R_γ(z_i)` under V_A (⚠ *not* at the read's launch point: that sits on the `q2 = 0` query manifold, where λ_min is **negative even for the fully designed landscape** — −16.76 at the site whose payload is −1, −4.93 site-averaged — which would make the bound vacuous for a landscape that retrieves at 1.000. At the fixed point the designed rung is a clean Morse minimum, λ_min = **+1.000**. Pinned by a test.)

| arm | ‖∇δV(q\*)‖ | λ_min(Hess V(q\*)) | **predicted drift** ‖∇δV‖/λ_min | **measured drift** | meas/pred |
|---|---|---|---|---|---|
| designed | 0.000 | 1.000 | **0.000** | **0.000** | — |
| mlp | 1.162 | 1.942 | 0.658 | 1.493 | 2.27 |
| hopfield (β=2) | 0.561 | 0.557 | 1.027 | 1.051 | **1.02** |
| hopfield_sharp | **0.195** | 0.875 | 0.221 | 0.214 | **0.97** |
| attn | 1.412 | 0.599 | 2.190 | 1.598 | **0.73** |
| atoms | 0.826 | **25.41** | 0.0359 | 0.0585 | **1.63** |
| atoms_local | **7.73e-4** | **25.93** | 3.55e-5 | 3.47e-5 | **0.98** |

**The C3 first-order law holds across six function classes and 4.6 decades of drift** — measured/predicted ∈ [0.73, 1.63] for 5 of 6 arms. The one outlier is the MLP at 2.27, whose drift (1.49) exceeds the ring radius, i.e. first-order theory should not apply there at all. This is a substantial strengthening of the theorist's C3 status ("bound proven + verified at the 1.0002-ratio level").

**And it decomposes the mechanism into two independent factors, which the shell statistic could not see:**
- **`atoms` is protected by STIFFNESS, not by a small perturbation.** Its `‖∇δV‖ = 0.826` is 71 % of the MLP's — the global gradient write moves its atoms just as much — but `λ_min = 25.4` is **13× stiffer** than the MLP's, because the write digs *narrow deep* wells. Locality of the *basis* buys curvature, not immunity.
- **`atoms_local` is the only arm protected by a SMALL PERTURBATION:** `‖∇δV‖ = 7.7e-4`, **1 070× smaller than `atoms`** at identical class and parameters. That is the write operator, isolated.
- **`hopfield_sharp` wins on both factors over `hopfield`:** numerator 0.195 vs 0.561 (2.9×, the `exp(−β)` support) and stiffness 0.875 vs 0.557 (1.6×) — product 4.6×, measured drift ratio **4.9×**.

> **The transferable statement (the task asked for "the most transferable number").** Interference is not governed by how far a write reaches; it is governed by **`‖∇δV(q*)‖ / λ_min(q*)` at the already-stored items**, and there are exactly two ways to make it small: **shrink the numerator with a local write operator** (`atoms_local`, 1 070×) or **inflate the denominator with stiff wells** (`atoms`/`hopfield_sharp`, 13–29×). The atom dictionary as a *basis* only does the second; only the masked write does the first.

---

## 5. Item 3 — does the design-freedom curve move? (5 seeds, both best families)

w20 headline: *minimum designed structure = essentially all of it*; **no learned rung cleared 0.9 at both K** and rung 1 was closest at 0.903 ± 0.101 / 0.959 ± 0.043.

| freedom | rung | family | K=4 | K=8 | passes? |
|---|---|---|---|---|---|
| 1 | `skeleton_residual` | **atoms** | **1.000 ± 0.000** | **0.986 ± 0.005** | ✅ |
| 1 | `skeleton_residual` | **hopfield** | **1.000 ± 0.000** | **0.986 ± 0.003** | ✅ |
| 1 | `skeleton_residual` | *mlp (w20)* | *0.903 ± 0.101* | *0.959 ± 0.043* | ❌ |
| 2 | `sites_learned_payload` | atoms | 1.000 ± 0.000 | 0.886 ± 0.042 | ❌ (−0.014) |
| 2 | `sites_learned_payload` | hopfield | 1.000 ± 0.000 | 0.889 ± 0.048 | ❌ (−0.011) |
| 2 | `sites_learned_payload` | *mlp (w20)* | *0.989 ± 0.014* | *0.739 ± 0.079* | ❌ |
| 4 | free | atoms | 0.980 ± 0.024 | 0.741 ± 0.019 | ❌ |
| 4 | free | hopfield | 0.998 ± 0.003 | 0.774 ± 0.029 | ❌ |
| 4 | free | *mlp (w20)* | *0.853 ± 0.095* | *0.599 ± 0.059* | ❌ |

**The answer, stated both ways (P8 falsified, P8′ not right either).** I pre-registered at 55 % that a better class would move the point all the way to freedom 4. It does not. But it is not true that nothing moves:

- **The minimum-viable-design point moves from "no learned rung passes at all" to "freedom 1 passes, robustly."** `skeleton_residual` goes from a straddling 0.903 ± 0.101 (individual seeds down to 0.766) to **1.000 ± 0.000 / 0.986 ± 0.005 — a 20× reduction in seed variance and equal to the designed ceiling (0.986) at K=8.** "Learning survives only as a small residual, and even that is marginal" (w20) becomes "**learning survives as a small residual, and with the right class it is no longer marginal**."
- **Freedom 2 now misses by 0.011–0.014** (0.886–0.889 vs the 0.9 bar) where the MLP missed by 0.161. Freedom 4 improves by 0.14 but is still 0.21 below the bar.
- **Two independent families agree to three decimals at every rung** (atoms vs hopfield: 1.000/0.986 vs 1.000/0.986; 1.000/0.886 vs 1.000/0.889), which is strong evidence that what remains is *not* a function-class limitation.

> **So w20's headline stands and hardens: the minimum designed structure that preserves the loop is essentially the whole designed landscape — and this is now shown to be a property of the LOOP, not of the MLP,** because three very different function classes at matched capacity all stop at the same rung with the same K=8 shortfall.

---

## 6. Two harness bugs I fixed, each of which would have produced a wrong published conclusion

1. ⭐ **The atom-dictionary arm was silently no-oping.** My first `AtomDictionaryPotential` used `A = softplus(depth_raw)` with a flat start at `depth_raw = −8`. `d softplus/dx = sigmoid(−8) = 3.4e-4`, and Adam moves a raw parameter by at most `lr × steps = 1.8`, so the deepest reachable well in a 600-step write is `softplus(−6.2) = 2e-3` — a landscape that is still flat. **Measured: strict 0.062 at K=4, write loss stuck at 0.12 (vs 0.003 for the MLP).** I would have reported "atom dictionaries fail" — the exact opposite of the truth — and it would have looked like clean support for H-EXPR. Fixed by `A = amp²` (non-negative, smooth, O(1) gradient at small amplitude): **strict 0.062 → 0.992 at seed 0.** Pinned by a regression test.
   **This is also the retraction of w20's rung-3 number** (§reconciliation list item 2): `RBFAtoms` initialises at `softplus(N(0, 0.1)) ≈ 0.69` per atom, so with a large dictionary the landscape *starts* rugged on the 0.3 length scale the relaxation must cross. Neither extreme works; the amplitude parameterisation is load-bearing.
2. **The support-decay normalisation.** Normalising the influence curve at r→0 divides by a near-zero for any localized write (a Gaussian atom has `∇δV = 0` at its own centre) and printed ratios up to **6 896×**. Caught in the smoke run, changed to peak-normalisation with `r₁₀` measured beyond the peak, before the production run.
3. **The C3 Hessian location** (§4b): at the read's launch point λ_min is negative even for the designed landscape, which would have reported the C3 bound as vacuous (`inf`) for all seven arms. Fixed to the relaxed fixed point; both the trap and the fix are pinned by a test.

---

## 7. PREREG scorecard (honest; `PREREG.md` written before any harness ran)

| # | prediction | measured | verdict |
|---|---|---|---|
| P0 | all arms ≤ 0.05 % from 4481 | 4481/4480/4480/4479/4480 | ✅ exact |
| — | *replication gate:* designed 1.000/0.986, mlp 0.853/0.599 | **identical to 3 d.p.** | ✅ |
| P1 | ≥1 learned class clears 0.9 at both K (60 %) | best is 0.774 at K=8 | ❌ **falsified — and P1′ (the strong program-level result) holds** |
| P2 | `atoms` ≫ w20 `local_rbf`; the w20 failure was initialisation (65 %) | 0.980/0.741 vs 0.623/0.348 | ✅ |
| P2 | hopfield_sharp 0.90–1.00 / 0.75–0.95 | **0.998 / 0.774** | ✅ both |
| P2 | hopfield(β=2) 0.70–0.95 / 0.45–0.75 | 0.459 / 0.282 | ❌ **too optimistic** (metastable blending) |
| P2 | attn 0.60–0.90 / 0.40–0.75 | 0.509 / 0.577 | ◐ K=8 ✅, K=4 below range |
| P2 | atoms 0.85–1.00 / 0.70–0.95 | 0.980 / 0.741 | ✅ |
| P3 | classification blank fails for every learned class; value blank passes (80 %) | 65/70 fail; 70/70 pass | ✅ |
| P4 | atoms_local corruption **< 1e-3** | mean 2.28e-3, **median 2.72e-5** | ❌ on the mean, ✅ on the median/scale |
| P4 | atoms_local ≥ 100× below mlp | **340×** | ✅ |
| P5 | hopfield_sharp < hopfield(β=2): interference monotone in attention temperature | **17.8×** at identical capacity | ✅ **the cleanest single confirmation** |
| P6 | attn and mlp both > 1e-2 | 1.55e-1, 7.75e-1 | ✅ |
| P7 | Spearman(support radius, corruption) ≥ 0.6 | r₁₀ **unresolved for 4/6 arms**; ρ = **0.143** | ❌ **falsified — and the falsification located the real statistic (C3 ratio, §4b)** |
| P8 | min-viable-design moves to freedom 4 (55 %) | stops at freedom 1 | ❌ |
| P8′ | it does not move at all (45 %) | freedom 1 goes from failing to passing at 20× lower variance | ◐ **neither branch; the honest answer is in between** |
| P9 | no arm 10× another in cost | ≤1.65× FLOPs; 3.7× wall-clock for atoms_local | ✅ |
| — | *H-MIX (my primary expectation):* fidelity ← class, interference ← support, the two dissociate | confirmed, **plus** a third factor (write operator) and a stiffness/perturbation decomposition neither hypothesis named | ◐ ✅ in direction, incomplete in content |

**Five falsifications, and the two most valuable results in this report came out of two of them** (P7 → the C3 ratio; P1 → the program-level "the function class is not the problem").

---

## 8. Verdict: H-EXPR or H-SUPP?

**Neither, as stated. Explicitly:**

- **H-EXPR is falsified as a fix and confirmed as a partial effect.** The MLP *is* too weak — a matched-parameter modern-Hopfield potential nearly doubles K=4 fidelity (0.853 → 0.998) and cuts interference 27× — but no learned class, in three families over five seeds, clears the bar at K=8, and two independent families agree to three decimals on where the ladder stops. **More capacity in a better class does not fix it.**
- **H-SUPP is confirmed in mechanism and falsified in its stated prediction.** Interference *is* about support, exactly as it claims — but its specific claim that "attention is more global than an MLP" is **wrong**: attention's support is `exp(−β·gap)`, a tunable knob an MLP does not have, and the β=2/β=8 pair changes interference 17.8× at identical capacity.
- **The factor neither hypothesis named is the one that dominates: the WRITE OPERATOR.** Same class, same 4480 parameters, same budget: masking the write to the item's own atom block drops interference 70× and removes the strict-retrieval loss entirely (−0.003). **Function class chooses the *stiffness* term of the C3 ratio; the write operator chooses the *perturbation* term, and it is worth 1 070× where the class is worth 13×.**
- **The program-level statement the task asked for, if all learned classes fail:** they do. **The function class is not the problem; structure is** — and this wave adds *which* structure: not the basis (locality of `atoms` alone buys 4.8× on interference and nothing at all on K=8 fidelity), but the **write operator** for interference and something still-unidentified for capacity, since even a perfectly local write leaves K=8 at 0.729.

---

## 9. How I verified

- Environment: worktree `../CHLU-potential-class`, **main venv reused** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`), **JAX 0.9.0**, no worktree `uv sync` (protocol §4 w6 lesson).
- **Full suite: `532 passed, 7 warnings in 631.27s`** (`pytest tests/ -q`), at the final commit. Reported honestly: the *previous* whole-suite run was **`1 failed, 531 passed`** — one of my own new tests asserted a negative λ_min at a site where it is +0.979 (the negative one is a *different* site: the K=4 launch points give +0.979 / **−16.76** / +0.979, mean −4.93). The assertion, not the harness, was wrong; it is corrected and now pins the site-averaged form. Log: `.claude/scratch/potential-function-class/pytest_full2.log`.
- `ruff check chlu/ tests/` → **All checks passed**. `ruff format` applied **only** to my two new files (`chlu/config.py` and `chlu/core/memory_potentials.py` are already unformatted on `main`; reformatting them would have been an out-of-scope diff).
- Production run: `python -m chlu.experiments.exp_potential_class` (no overrides), **exit 0**, ~24 min. `--quick` exit 0 in ~90 s. CLI parser verified: `exp-potential-class --quick --classes designed atoms_local` parses to `cmd_exp_potential_class`.
- Every number above is read out of the committed `results/exp_potential_class_metrics.json`, copied to `.claude/outputs/potential-function-class/` together with the figure and a rendered `summary.txt`.

## 10. Git footprint

- **Branch** `agent/experiment-engineer/potential-function-class`, off local `main` @ **`31c3e15`**. Rebase onto `main` = no-op (base has not moved). **Did not touch `origin/main`** (§7.21).
- **Worktree** `../CHLU-potential-class`, created because the main checkout was on `main` and other spokes may be running. Branch ref verified from the main repo (see below). Worktree left in place for review; remove with `git worktree remove ../CHLU-potential-class`.
- **Commits (4):**
  - `6dca248` Hopfield/attention/atom-dictionary potentials + local (masked) writes
  - `b8b4515` `exp-potential-class`: function-class sweep at matched parameters
  - `2b97d2b` tests (16; a 17th added in `e06d69c` — 17 total in `tests/test_potential_class.py`)
  - `e06d69c` C3 drift law at the stored sites; multi-family ladder; 7 interference seeds
- **Files: +** `chlu/experiments/exp_potential_class.py`, `tests/test_potential_class.py`; **M** `chlu/core/memory_potentials.py` (appended `AtomDictionaryPotential`, `HopfieldPotential`, `AttentionPotential`, `atom_write_mask_fn`, `LEARNED_FAMILIES`; added the `learned_family` axis to `DesignFreedomPotential` — **defaults to `"mlp"`, so every w20 path is byte-identical**; existing classes otherwise untouched), `chlu/config.py` (new `ExperimentPotentialClassConfig` + all **four** registration sites), `chlu/cli/experiment_cmd.py` (import, parser, `cmd_exp_potential_class`), `chlu/training/train_memory.py` (**+11 lines**: optional `update_mask_fn`, default `None` = unchanged behavior).
- Did **not** touch `utils/plotting.py` (shared) — the figure is local, per the w19/w20/`exp_paid_access` precedent. `results/` deliberately not committed. No unresolved conflicts.

## 11. Open questions / follow-ups / risks

1. ⭐ **What actually breaks at K=8?** Every learned class lands at 0.73–0.77 while `designed` holds 0.986, and it is **not** the function class (three families agree) and **not** the write locality (`atoms_local` is no better). w19's ring-capacity ceiling is ≈8.4 items, so K=8 is close to the designed limit — but the designed rung still works there and the learned ones do not. **This is the single most valuable follow-up in this thread**, and the w19 open question about *d*-dimensional address spaces is the obvious probe.
2. **The write objective is static (no BPTT) and untuned** — inherited unchanged from w20 so the comparison holds, but it contains no term rewarding *basin reach*: a deep narrow well satisfies the loss while having no gradient at the `q2 = 0` launch manifold. This bites localized bases hardest, and is a plausible part of the K=8 gap. A reach term (`relu(margin − ⟨−∇V(query), (z_i − query)/‖·‖⟩)`) is cheap and class-agnostic — but it would change the MLP baseline too, so it needs its own w20 re-run.
3. **`atoms_local`'s two bad seeds** (7.3e-3, 8.3e-3 out of 7) are unexplained. The mask is provably exact, so the leakage must be through the *confinement* term shared across blocks or through B's Gaussian tails with a learned width larger than 0.3. Worth 20 minutes with the per-atom widths.
4. **β was not swept**, only sampled at 2 and 8. The `exp(−β·gap)` support law predicts a smooth interference curve in β and a fidelity optimum (β too small ⇒ metastable blending, measured; β too large ⇒ vanishing gradients for far keys, untested). **A β sweep would turn §3's two points into a law** and is the cheapest high-value follow-up here.
5. **Construction (a) of the transformer arm (d coordinates as d tokens, self-attend) was not run** — at dim = 3 it is degenerate. If the Hub wants the "we tested real self-attention" claim in a paper, it needs the higher-*d* address space of follow-up 1 first.
6. **`atoms_local`'s per-item block allocation is fixed at write time and sized `n_atoms/K`.** Real use needs allocation, which is the controller's job (MVC-0 §3.B) and is out of scope here — but note that the 70× result depends on *knowing which block belongs to which item*, i.e. on the codebook the controller maintains.
7. **Single geometry (2-D ring + 1 payload channel), dim = 3, one write-hyperparameter configuration per family.** The atom and hopfield families each got one shape hyperparameter chosen a priori (`depth_init`, β); the MLP has none left once parameters are matched. The learned arms are therefore only lightly tuned, which is the honest baseline but means their K=8 failures are not proven irreducible.

---

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry.** The w20 learned-landscape failure has been decomposed. **Fidelity:** no learned function class (MLP / modern-Hopfield β∈{2,8} / cross-attention / atom dictionary, all at 4480±1 parameters, 5 seeds) clears strict 0.9 at both K=4 and K=8; best is 0.998/0.774. **Interference:** spans 340× across classes (mlp 7.75e-1 → atoms_local 2.28e-3) and is governed by the **write operator**, not the class. **Verdict: neither H-EXPR nor H-SUPP as stated.**
2. **§7 — new entry (retraction).** ⚠ **w20's `local_rbf` rung (0.623 ± 0.330 / 0.348 ± 0.117) must not be cited as evidence about locality or atom dictionaries.** It is an initialisation artifact: `RBFAtoms` starts every atom at depth `softplus(0) ≈ 0.69`, so a large dictionary begins rugged on the retrieval length scale. The same basis with a flat start and a squared amplitude gives **0.980 ± 0.024 / 0.741 ± 0.019**. Sibling trap, also pinned: a *softplus* flat start (`raw = −8`) has a 3.4e-4 gradient and makes the write a silent no-op (strict 0.062).
3. **§1/§8 — the transferable law.** Cross-write interference is governed by the C3 ratio **`‖∇δV(q*)‖ / λ_min(Hess V(q*))` at the already-stored items**, measured across 6 function classes and 4.6 decades with measured/predicted ∈ [0.73, 1.63] (5/6 arms). Two independent routes to safety: **shrink the numerator with a local write operator** (1 070×) or **inflate the denominator with stiff wells** (13–29×). *Support radius (shell decay around the new write site) is NOT the mechanism — Spearman 0.143 — and should not be used.*
4. **For the theorist.** (i) C3's bound is now numerically validated far beyond its "1.0002-ratio" status, **with a scope correction: it must be evaluated at the relaxed fixed point, not at the read's launch point** (λ_min is negative on the `q2 = 0` query manifold even for the designed landscape). (ii) C3 clause (a) should be split: the atom **basis** buys stiffness (13×); only the masked atom **write** buys locality (1 070×). (iii) **New in-framework result on Ramsauer:** the modern-Hopfield energy's parameter support is `exp(−β·gap)`, so attention-as-memory is *tunably local*, and at β=8 it is the best global-write class in the program — a materially better answer to the Ramsauer objection than "attention is global".
5. **§1 — the metastable-blend regime is now visible in our loop:** at β=2 the Hopfield arm addresses correctly (basin 0.966) but returns a softmax *blend* of stored payloads (strict 0.459, err 1.4e-1). This is Ramsauer's metastable-state regime observed through a value-recovery metric, and it is a concrete referee-facing contrast with CLU's exact sub-barrier isolation.
6. **New CLI/config surface:** `chlu exp-potential-class [--quick] [--classes …]`, `ExperimentPotentialClassConfig` (load-bearing: `param_target`/`param_tol`, `hopfield_beta_soft|sharp`, `hopfield_confine=0.5`, `atom_depth_init=1e-4`, `local_write_steps`, `ladder_families`). New core classes `AtomDictionaryPotential` / `HopfieldPotential` / `AttentionPotential` / `atom_write_mask_fn`; `DesignFreedomPotential` gains a `learned_family` axis **orthogonal to `rung`** (default `"mlp"` ⇒ w20 unchanged). `train_memory_landscape` gains `update_mask_fn` — **the third training path now supports parameter-local writes**, which is the MVC-0 prerequisite.
7. **Method note (reinforces w20's, now at 4 more classes):** the classification blank fails on **65/70** learned cells across four function classes (≈1.000 for four arms) while the value blank passes **70/70**. The address leak is a property of *learned landscapes*, not of the MLP. Value-recovery metrics remain the only valid primary scoring.
