# Task: sleep-erosion-study — characterize CD/PCD landscape erosion (ICLR-grade; Head decision 2c)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/sleep-erosion-study.md` + figures in `.claude/outputs/sleep-erosion-study/`
- **Read first:** protocol · `.claude/outputs/v2-full-runs.md` §2 (Finding 0 — the discovery + the 2×2 attribution already done) · `.claude/outputs/generative-studies.md` (Study B mechanism analysis — γ=0 isoenergetic walk) · F5 §2.4. Requires fix-pack-3 item 1 (`sleep_mode` switch) — coordinate: if not yet merged, replicate via `sleep_frequency` override as v2-full-runs did.
- **Head framing:** move fast; ICLR-level depth; worst case = V2-short appendix, best case = its own short about long-run contrastive divergence.

**The finding to characterize:** wake–sleep training *destroys* a designed degenerate vacuum over epochs (ring depth +0.060@150 → −0.047@1000, inversion 300–600; wake-only deepens and data-pins instead). Mechanism hypothesis (from the code): PCD negatives thermalize into the low-V ring; CD keeps raising V there; nothing anchors V's *value* at data (wake is trajectory-MSE only).

## Questions (each = a figure/table)
1. **Erosion rate vs sleep hyperparameters:** ring depth & r* vs epoch for a grid over `sleep_frequency ∈ {1,5,20}`, `sleep_temperature ∈ {0, 0.5, 2}`, `sleep_steps ∈ {50, 500}`, `persistent_sleep_buffer ∈ {F,T}` (2-3 seeds; exp-d setting; checkpoint every 100 epochs). Where's the erosion horizon as a function of total sleep updates × step count? Does persistence change it (in exp-d's γ=0-sleep regime the generative-studies null may not apply — negatives here CAN reach the ring)?
2. **Mechanism test:** track the sleep-negatives' energy/position distribution over training — do they measurably concentrate in the ring before inversion (the hypothesized thermalize-then-raise cycle)? One scatter/density evolution figure.
3. **Anchoring cures (small ablations):** (a) energy-anchor term pinning mean V(data) (like train_generative's magnitude regularizer — does it stop erosion?); (b) sleep-negatives energy-gated (reuse gamma-field's persistent-hallucination gate concept: only raise V on ABOVE-band negatives); (c) wake-only control. Verdict: which cheapest cure preserves both the vacuum AND sleep's noise-rejection benefits (check the trained model still rejects noise à la Exp-B)?
4. **Generality probe (bounded):** does the same erosion hit the standard Exp-B sine setting (non-degenerate vacuum) — i.e., is this specific to degenerate/flat vacua or generic? One 2-seed comparison.
5. **Literature hook (no deep dive):** note the connection to known CD/PCD landscape-distortion results (Tieleman PCD, "CD doesn't follow the gradient of any function" line) — 3–4 citations for the write-up; flag if our *measured inversion of a designed vacuum* appears novel.

**Report:** the erosion phase diagram + mechanism evidence + cure verdict + an honest novelty assessment. This feeds: V2-short methods/appendix, possibly a standalone short, and the ICLR training-methodology section.
