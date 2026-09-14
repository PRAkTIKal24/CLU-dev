# Task: regime-remap-2000ep — the compute-parity CLU-vs-Hopfield map (resolves CM-8; highest-leverage w7 item)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/regime-remap-2000ep.md` (+ figures in `.claude/outputs/regime-remap-2000ep/`)
- **Read first:** protocol · claims matrix **CM-8 (frozen PROVISIONAL — this task unfreezes it)** · `anchor-robustness.md` item 2 (the 3-cell evidence + its recommended follow-ups 1–2) · `v1-hopfield-stress.md` (the original 26-cell map + apparatus).
- **Why:** anchor-robustness showed the "Hopfield dominant 26/26" map was under-trained — at 2000 ep, 3/3 sampled losing cells close entirely (fidelity→1.00, 9–10× savings). If this holds grid-wide, V1's regime story inverts from "Hopfield everywhere" to "CLU competitive-or-winning at compute parity with a calibrated rationing gate." **V1 drafting is HELD on this result.** Repo read-only; scratch in `.claude/scratch/regime-remap-2000ep/`.

## Items
1. **Full 26-cell grid re-map at 2000 ep**, ≥5 seeds × ≥5 episodes/cell, paired with the 500-ep baseline (same seeds/episodes) so the delta is attributable. Metrics per cell: CLU-EBM fidelity, gated accuracy, full-budget accuracy, Hopfield accuracy, compute savings at matched accuracy. Deliverable: the corrected regime map + the honest CM-8 rewrite sentence.
2. **Epoch-scaling frontier:** fidelity vs epochs {500, 1000, 2000, 4000} at kv ∈ {32, 64, **96, 128**} (the capacity-wall question — does a true wall reappear beyond the tested kv64?). 3 seeds. Deliverable: the compute–fidelity frontier figure + "the wall is at kv≈X for budget Y" or "no wall found at laptop scale."
3. **Compute-parity fairness check:** report Hopfield's cost curve too (its β/iteration knobs at equivalent wall-time) so "9–10× savings at parity" survives a reviewer asking "did you give Hopfield the same budget?" — the P8 no-strawman rule applied in reverse.
4. Negatives (cells that do NOT close; wall location) fully written per C-9.

**Report:** per-cell tables + error bars + the one-paragraph CM-8 replacement wording (Hub will splice it into the matrix). Flag-provenance per §5 (train_epochs is THE flag here — table it everywhere).
