# PREREG — `cl-encoder` (the CL-capable read-in φ)

**Written 2026-07-28, BEFORE any gate harness ran** (protocol §5 pre-registration rule; the
acceptance criterion is a measured ratio/threshold, so predictions are committed here first).
Branch `agent/experiment-engineer/cl-encoder`, base local `main @ ff85573`, worktree
`/Users/user/Desktop/CHLU-cl-encoder`, main venv reused (JAX 0.9.0 / equinox 0.13.4).

Binding upstream: `PREREG_CL_PHI.md` (`phi_dim ≥ 16`; `task1_only` = PRIMARY; `generic_frozen`
= declared upper bound only; φ fit objective **unsupervised only** — no label, no retrieval
loss, no store gradient; φ frozen at end of task 1, never refit, never trained through the
store [w20 law]).

---

## 0. Dial declaration (echoed from the task file)
- **Dial:** none directly — **enabler** for R4 (the CL entry). An encoder improvement is NOT a
  store result.
- **Laundering control:** kNN-in-φ over the same new features at matched memory — and note the
  trap **in advance**: *the gate metric IS the launder.* Clearing the gate raises the launder
  by construction; it buys the entry a scope, not a win.
- **Falsifies:** kNN-in-φ over the new features fails the gate ⇒ the encoder is not the fix and
  the CIFAR null has a second cause.
- **Does NOT falsify:** the store losing to iCaRL/replay (CM-23(n)); the launder firing on ACC.

## 1. The gate (registered exactly)

**Metric.** Class-IL end-of-stream ACC of **kNN-in-φ over a class-balanced ring buffer of 200
φ-keys**, Split-CIFAR-10 **reduced protocol** (`apply_cifar10`: 1000 train / 500 test per task,
5 tasks × 2 classes), `task1_only` φ, no task identity at test time, `ACC = mean_i A[T−1, i]`.
This is the same object as w25's `knn_phi_ringbuffer_task1_only` row (measured **0.219 ± 0.014**
at PCA-32; the same-keys variant was 0.207 ± 0.014 — the ring buffer is the stronger launder, so
it is the gate line).

**Gate value: kNN-in-φ ≥ 0.35.** Reference points under the same reduced protocol (w25):
LwF 0.162 · GDumb 0.301 · ER 0.369 · iCaRL 0.419 · joint upper bound 0.480.

**Decision rule (registered):**
- gate **cleared** (≥0.35, seed 0, then confirmed on 3 seeds) ⇒ run the full entry with the new
  φ (3 seeds, mandatory baseline table, both laundering lines).
- gate **missed** ⇒ STOP. Report the gate as the finding + name candidate second causes. No
  entry run over addresses that cannot separate the classes.
- **Borderline band 0.30–0.35** (registered in advance so it is not decided after the fact):
  treat as a MISS for the entry-run decision, but report it as "the encoder moved the gate by
  ≥0.09 without clearing it" and quote the extrapolation cost.

**Seeds.** Arm selection on **seed 0** (compute contention: 4 engineer worktrees). The winning
arm and the PCA-32 control are then re-measured on **seeds 0,1,2** before any gate verdict is
declared. A one-seed number never decides the gate.

## 2. Predicted values (committed before measurement)

| # | arm (all `task1_only`, CIFAR reduced protocol, 200-item buffer) | predicted kNN-in-φ ACC | P(clears 0.35) |
|---|---|---|---|
| P0 | **PCA-32 (control, reproduction of w25)** | 0.21 ± 0.02 | — (harness validity check: must land in 0.19–0.24 or the harness is wrong) |
| P1 | PCA-64 / PCA-128 | 0.21 – 0.25 | <0.02 |
| P2 | AE-32 / AE-64 (existing MLP AE arm) | 0.19 – 0.26 | <0.05 |
| P3 | **conv AE** (new, reconstruction MSE) | 0.22 – 0.32, point **0.26** | ≈0.20 |
| P4 | **SimCLR-lite conv trunk** (new, NT-Xent) | 0.28 – 0.45, point **0.36** | ≈0.45 |
| P5 | random-weight conv trunk (sees no data at all) | 0.20 – 0.30 | ≈0.10 |

**Derivations (why these numbers, not vibes):**
- **P1.** The CIFAR pixel covariance spectrum decays fast; PCA-32 already captures ~90 % of
  pixel variance, so components 33–128 add ~a few % of variance in high-frequency directions
  that contribute *distance noise* to a 1-NN read-out. The w24 `phi_dim` sweep showed retrieval
  saturating by 32 on MNIST for the same reason. Expect +0.01 ± 0.02 over PCA-32.
- **P2.** The existing AE is a 1-hidden-layer tanh MLP trained ≤400 Adam steps with
  reconstruction MSE — the same objective as PCA and, at this budget, close to its linear
  solution. On MNIST (w24) the AE tracked PCA to ≤0.03. Reconstruction of CIFAR is dominated by
  the same low-frequency colour content that already fails.
- **P3.** Convolution + spatial pooling removes the *pixel-alignment* sensitivity that cripples
  raw-pixel and PCA distances (a translated object is far in pixel space, near in pooled-conv
  space). That is the single biggest known deficiency of the current φ, so a real move is
  expected — but a reconstruction objective still spends capacity on colour/texture, which is
  why the point estimate stays below the gate.
- **P4.** Full-scale SimCLR (50 k images, all 10 classes, ResNet-18, hundreds of epochs) reaches
  kNN ≈0.88 on CIFAR-10. Our arm reduces unlabelled data ~10×, pretraining classes 5× (task-1
  classes only), trunk capacity ~20× and training length ~10×. Published SimCLR ablations lose
  10–20 pp per such axis in this regime ⇒ a linear-probe-equivalent of ~0.45–0.60; a
  20-exemplar-per-class 1-NN read-out typically retains 70–80 % of the linear probe ⇒
  0.32–0.48, widened downward for the 2-class pretraining pool ⇒ **0.28–0.45**.
- **P5.** Random conv features + pooling are a known non-trivial CIFAR baseline (Coates & Ng
  patch-feature lineage), but without any fitting they sit well below a trained SSL trunk.

## 3. Second-order predictions (also registered)

- **P6 — strict-φ cost** (`generic_frozen − task1_only`, same arm, same seeds): **≈0.00 ± 0.01
  for PCA/AE** (both are broken, so strictness cannot bite — w25's +0.001 was on a broken φ and
  proves nothing) and **+0.02 … +0.10 for a working SSL/conv arm** (SSL features fit on a
  2-class pool are measurably less transferable than ones fit on all 10). ⭐ *A working φ is the
  first setting in which the strict-φ cost can bite at all; if it is still ≈0 at a working arm,
  that is a genuine (and favourable) finding for the entry's defensibility.*
- **P7 — the launder trap, registered as an expectation, not a surprise.** Because the gate
  metric *is* the laundering control, any arm that clears the gate hands CLU-in-φ and kNN-in-φ
  the *same* improved addresses. Registered prediction: with a working φ the entry's CLU ACC
  will land **within ±0.05 of the kNN-in-φ line and most likely below it** (5 waves of
  precedent). If it does, the wave has bought R4 a scope, **not** a win, and the report says so
  in those words.
- **P8 — geometry at a working arm:** corrected packing slack stays **< 1** (0.25–0.50 band;
  w25 CIFAR was 0.337–0.345, MNIST ≈0.33). Crowding is intrinsic to classification streams and
  is *not* predicted to be fixed by a better φ. (⛔ never quote the retracted 1.08.)
- **P9 — `phi_dim`:** the working arm's gate number is predicted to be **monotone
  non-decreasing** in `phi_dim` over {32, 64, 128} with a difference ≤0.03 between 64 and 128.
  Every number is quoted with its `phi_dim` (binding since w24).

## 4. What would make me say the encoder is NOT the fix (second causes, named in advance)
1. **200 prototypes is the binding constraint, not the features** — testable in one line: kNN
   over the *same* φ with the *full* stream (10 000 keys) instead of 200. If that is also ≈0.2,
   features are the cause; if it is ≫0.35, the memory budget is the cause and the entry's
   problem is compression, not representation.
2. **The reduced protocol's ceiling** — joint is only 0.480; a gate of 0.35 is 73 % of joint.
3. **Class-IL read-out geometry** — a nearest-*address* decode over class labels may be the
   limiter rather than the address space itself.

## 5. Compute plan (cheapest decisive first, as tasked)
Stage A (minutes): PCA-32 control → PCA-64/128, AE-32/64. Stage B: conv AE, SimCLR-lite (seed 0).
Stage C: 3-seed confirmation of the best arm + PCA-32 control. Stage D (**only if the gate
clears**): full entry run. NOT-RUN ≠ null: anything unrun is reported as unrun.
