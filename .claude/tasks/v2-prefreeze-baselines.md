# Task: v2-prefreeze-baselines — learned-architecture baselines for V2 + queued S1 extras (critique P8/V2.3)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v2-prefreeze-baselines.md` (+ figures/npz in `.claude/outputs/v2-prefreeze-baselines/`)
- **Read first:** protocol · `.claude/critique_register.md` (V2.3 — now BLOCKING, not optional) · `.claude/outputs/v2-full-runs.md` (the retention protocol + numbers to compare against) · `.claude/outputs/mo-deep-read.md` (Mo's S¹ task spec + protocol) · `.claude/outputs/gamma-field-build.md` (S1 arms + adaptive-K/compact-gate follow-ups).
- **Why:** V2 currently compares CLU against theory and against Mo's protocol — never against any trainable memory. "Interesting physics, but is it better than an LSTM at remembering?" is a guaranteed review. Repo **read-only**; scratch scripts in `.claude/scratch/v2-prefreeze-baselines/`; laptop-CPU.

## Items
1. **Learned baselines on the retention protocol (the blocking item):** implement **coRNN, LEM, LSTM** as small JAX scratch modules (dims matched to the CLU testbed; report param counts). Train each on the same task family, then run the retention-vs-perturbation protocol + the write/read latch test. **5 seeds.** Deliverable: retention curves overlaid on CLU's (+ the budget-table prediction), and the honest sentence: where does CLU win, lose, tie — and is the *structure* of its retention (slope −1, floor, latch) qualitatively absent in the baselines?
2. **Mo-S¹ head-to-head:** run our trained mode on Mo's S¹ task per the mo-deep-read spec (his protocol, our system) — completes the "his law is our overdamped face" figure from both directions. 3+ seeds.
3. **S1 re-run with the queued gamma-field extras:** adaptive-K spawn + compact-support gates. Try scratch-level first (the γ_φ module is importable); **if it requires `chlu/` edits, STOP that item and report "defer to engineer" with the exact edit needed** — do not modify the repo.
4. **(cheap, if an emergent checkpoint now exists on disk)** the deferred kick-size decomposition from seed-sweeps item 4; skip silently if no checkpoint.

**Report:** per-item numbers with error bars + one-line quotable verdicts; frozen-manifest discipline (config/seed/commit per figure); flag-provenance table per result (protocol §5). If a baseline needs tuning to be fair, do one modest lr sweep and report it — no untuned strawmen (register P8: this baseline must survive review).
