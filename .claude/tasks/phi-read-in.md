# Task: phi-read-in — the learned read-in `φ` around a DESIGNED store (the phase-doctrine flagship) (w23)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/phi-read-in.md` · **Branch:** `agent/experiment-engineer/phi-read-in`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (§3.2 worktrees — **this wave has 4 parallel engineer tasks touching `chlu/`**) · `.claude/research_roadmap.md` v0.6 · `.claude/outputs/hopfield-capacity-benchmark.md` (the protocol you re-run, incl. the `sqdiff` metric + `torch.dropout(0.5)` mask pinned from source, and the x64 crash fix) · `.claude/outputs/clu-latent-io-audit.md` (the structural point: with `φ = identity` the latent IS raw space) · `.claude/claims_matrix.md` v2.1 CM-22/CM-23
- **⭐ Phase doctrine (Head, 2026-07-23, binding):** *learn around a designed core* — fixed relaxation physics, designed wells, **learned interfaces**. Attention is the precedent: `softmax(QKᵀ)V` is fixed; the projections are learned. Every "CLU on raw data" loss traces to the missing `φ`. This task builds it and re-fights the one external benchmark we already ran, in feature space.

## Why
w22 (`hopfield-capacity-benchmark`): CLU loses the capacity axis to a **trivial nearest-neighbour pixel baseline** because masked-image retrieval in *pixel space* doesn't need memory. In a feature space the trivial baseline becomes kNN-in-φ-space — the comparison we can actually contest — and closed-form Hopfield's CIFAR chance-collapse (DC-dominated inner products) is exactly what a `φ` fixes for *everyone*. First fair fight.

## Item 1 — two `φ` arms (never trained through the store)
- **φ-A (frozen/cheap):** PCA-k and/or a small frozen encoder. Zero learning on the CLU side of the interface.
- **φ-B (trained embedder):** a small AE or contrastive encoder trained **separately on the data distribution only** — it never sees the store, the wells, or a retrieval loss. (w20's law: learning destroys what design provides — the store stays designed.)
Store = **key–value**: address `φ(x)` written as a designed well; payload = the raw `x` (or label) carried in the well record, so read-out `ψ` = settle → return payload. State `d`, well width, and the packing-law occupancy (`Δ_req ≈ 3.1·max(w, σ_query)`, matrix v2.1 §1) for your chosen feature dim.

## Item 2 — re-run the Hopfield protocol in φ-space (MNIST + CIFAR-10)
Capacity axis (M sweep) + noise axis (σ sweep), masked queries per the pinned protocol, success = mean `sqdiff` in **pixel space** (payload comparison — keeps w22 comparability). Lines: CLU-in-φ · **kNN-in-φ (the trivial baseline, now fair)** · closed-form Hopfield-in-φ · the w22 raw-space CLU line (continuity control).

## Item 3 — ⛔ THE LAUNDERING CONTROL (mandatory, pre-registered)
The C17-3 lesson: γ "won" on C-MAPSS by turning the physics off (CLU's margin over a trivial feature was +0.003). So: **same `φ`, trivial store swap** — if kNN-in-φ matches CLU-in-φ everywhere, the win is `φ`'s, not ours, and the report must say so in those words. A CLU margin that exists ONLY with the designed store is the result the program needs.

## Item 4 — does the retry hook survive `φ`?
Report (briefly — the full study is `retry-compute-study`) whether the confidence signal (distance-to-nearest-well at settle) still separates correct/incorrect reads in φ-space. This gates the retry thread's feature-space extension.

## Acceptance
The capacity + noise tables/curves in φ-space with all four lines, both φ arms, the laundering control, and the Item 4 note. Pre-register win/tie/lose per axis per arm **before running**. State every knob's fairness category. Tests green; reuse the w22 x64-safe rollout. ⚠ If CLU-in-φ still loses to kNN-in-φ on capacity, report it plainly — that is a decision-grade negative about the store, no longer excusable by the missing embedding.
