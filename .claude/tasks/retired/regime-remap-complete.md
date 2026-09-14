# Task: regime-remap-complete — finish the CM-8 grid (Items 1/2/4) so V1's accuracy story can ship (w8)

- **Agent:** `results-analyst` · **Output:** append to `.claude/outputs/regime-remap-2000ep.md` (same file — it is idempotent/resumable) + a `## w8 completion` section; figures in `.claude/outputs/regime-remap-2000ep/`.
- **Read first:** protocol · `.claude/outputs/regime-remap-2000ep.md` (your own partial — Item 3 done, Items 1/2/4 stubbed; the driver is idempotent/resumable, `batch.py`) · `.claude/claims_matrix.md` **CM-8** (the cost story is FINAL: Hopfield cheaper at 1 matvec; this task settles the ACCURACY story only) · `.claude/outputs/anchor-robustness.md` item 2.
- **Why:** V1 cannot draft its regime-map section until the full grid confirms (or refutes) that 2000-ep training closes the Hopfield accuracy gap broadly. Item 3 already fixed the cost framing; do NOT re-open it. Repo read-only.

## Items (finish the stubs)
1. **Item 1 — full 26-cell grid at 2000 ep vs the 500-ep baseline**, ≥5 seeds × ≥5 episodes, paired. Per cell: CLU fidelity, gated acc, full-budget acc, Hopfield acc, and the **intra-CLU** savings (gate vs full-budget — label it that way, never "vs Hopfield"). Deliverable: corrected regime map + the final CM-8 accuracy sentence (how many cells close; which do not).
2. **Item 2 — epoch-scaling frontier** {500,1000,2000,4000} × kv∈{32,64,96,128}: does a capacity wall reappear beyond kv64? 3 seeds. The compute–fidelity frontier figure.
3. **Item 4 — negatives** (cells that do NOT close; wall location if any) per C-9.
4. Restate, verbatim from your own Item 3, the cost caveat next to every accuracy-improvement claim (the P8-in-reverse discipline): Hopfield is the cheaper retriever; the CLU number is intra-CLU compute rationing.

**Acceptance:** the full 26-cell map + frontier + the one-paragraph CM-8 accuracy replacement the Hub splices into the matrix. Flag-provenance (`train_epochs` tabled). If any cell exceeds ~1h, report the reduced grid rather than blocking.
