# PREREG — phi-read-in (written BEFORE any harness ran)

Committed predictions for the learned read-in `φ` around a DESIGNED key–value store,
re-fighting the w22 Hopfield/U-Hop protocol in **feature (φ) space**. Metric = mean
`sqdiff` in **pixel space** on the returned payload (keeps w22 comparability); we also
track identity-retrieval accuracy (payload index == true index).

Four lines: **CLU-in-φ** (settle φ(query) in a designed GaussianMemory over φ(store),
read payload of the well it lands in) · **kNN-in-φ** (the trivial baseline, now fair:
argmin‖φ(query)−φ_i‖ → payload) · **closed-form Hopfield-in-φ** (softmax update over
stored φ, decode to nearest stored φ → payload) · **raw-space CLU** (the w22 pixel-space
line, continuity control). Two φ arms: **φ-A = PCA-k (frozen)** and **φ-B = small AE
trained on a DISJOINT data-distribution pool, reconstruction loss only** (never sees the
store / wells / any retrieval loss).

## Central hypothesis (the one the program needs to test)
The phase doctrine says a designed store + learned interface should beat the trivial
feature baseline. The honest prior (w20 "learning destroys design"; w22 "a trivial NN
floor beats CLU and Hopfield on the image protocol") is that **kNN-in-φ is extremely
strong** and **CLU-in-φ, which settles to the nearest well, will closely track it**.
⇒ I pre-register the **laundering-control-firing** outcome as the most likely: CLU-in-φ
≈ kNN-in-φ. A CLU margin over kNN that appears would, per the doctrine, be the result the
program needs; I predict it will NOT appear on capacity and is at best marginal on noise.

## Win/tie/lose predictions (per axis, per arm) — CLU-in-φ vs the other lines
Tie band: |Δ identity-acc| ≤ 0.03. "win/lose" = outside that band.

| axis | arm | CLU-in-φ vs kNN-in-φ | CLU-in-φ vs Hopfield-in-φ | CLU-in-φ vs raw-space CLU |
|---|---|---|---|---|
| capacity (M sweep) | φ-A PCA | **TIE** (laundering fires) | **WIN** (nearest-well beats 1-step blend) | **WIN** (φ lifts CLU above raw pixels) |
| capacity (M sweep) | φ-B AE | **TIE** (laundering fires) | **WIN** | **WIN** |
| noise (σ sweep) | φ-A PCA | **TIE/LOSE** (kNN dominates noise, w22) | **TIE/WIN** (attractor cleanup) | **WIN** |
| noise (σ sweep) | φ-B AE | **TIE/LOSE** | **TIE/WIN** | **WIN** |

## Other registered predictions
- **P-φ (does φ fix the CIFAR chance-collapse?):** raw-pixel closed-form Hopfield was at
  chance on CIFAR (w22, DC-dominated inner products). **Prediction: Hopfield-in-φ recovers
  well above chance on CIFAR** (both PCA and AE decorrelate/whiten the DC mode). This is
  the "a φ fixes closed-form Hopfield for *everyone*" claim (task Why).
- **P-lift:** feature space lifts CLU capacity vs its w22 raw-pixel line at every M (MNIST
  raw CLU was 0.56 at M=128; predict CLU-in-φ > 0.75 at M=128 for at least one arm).
- **P-arm:** φ-B (trained AE) ≥ φ-A (PCA) on identity-acc at high load (a learned manifold
  packs the store better than a linear projection). Predict a small AE advantage (≤0.05).
- **P-Item4 (retry hook survives φ):** distance-to-nearest-well at settle **still
  separates** correct vs incorrect first-pass reads in φ-space (AUROC ≥ 0.65). The
  confidence signal is geometric, not pixel-specific, so it should transfer.
- **P-laundering (explicit):** on capacity, for BOTH arms, kNN-in-φ is within the tie band
  of CLU-in-φ at ≥ half the M grid points. If TRUE ⇒ report in the task's exact words:
  *"the win is φ's, not ours."* If CLU-in-φ beats kNN-in-φ outside the band anywhere with
  the designed store, that cell is the decision-grade positive.

## What each outcome means
- CLU-in-φ **beats** kNN-in-φ somewhere with the designed store ⇒ the store adds value
  the feature map alone does not: the result the program needs.
- CLU-in-φ **ties** kNN-in-φ everywhere ⇒ laundering: the value is φ's. Honest negative
  about the store, no longer excusable by a missing embedding (task ⚠).
- CLU-in-φ still **loses** to kNN-in-φ on capacity ⇒ reported plainly per the task's
  final ⚠ — a decision-grade negative about the store.

## Fairness categories (declared before running)
- φ-A PCA: fit on a **disjoint** data-distribution pool, unsupervised, frozen at retrieval — FAIR.
- φ-B AE: trained on the **disjoint** pool, reconstruction MSE only, frozen at retrieval,
  never sees store/wells/retrieval loss — FAIR (the w20 rule: learning off the CLU side).
- store: centers = φ(store patterns), well width s = 0.3·median-NN(φ) — DESIGNED, one fixed
  rule, not tuned per load.
- kNN-in-φ / Hopfield-in-φ: same φ, same masked/noisy queries — FAIR controls.
- Metric: mean sqdiff in pixel space on the payload — identical to w22 (comparability).
