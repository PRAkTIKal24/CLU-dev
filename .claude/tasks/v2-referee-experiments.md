# Task: v2-referee-experiments — the two small runs the referee's SHOULD-FIXes need (w8)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v2-referee-experiments.md`
- **Read first:** protocol · `.claude/outputs/v2-referee.md` (SF-1, SF-2, SF-3 + the missing-experiment list items 3/4/5) · `.claude/outputs/v2-full-runs/` (check whether Mo's estimator sweep already exists before running it) · `.claude/outputs/v2-prefreeze-baselines.md` (baseline apparatus for the FLOP count).
- **Why:** three referee SHOULD-FIXes need data the writer can't fabricate. All laptop-scale, cheap. Repo read-only; scratch in `.claude/scratch/v2-referee-experiments/`.

## Items
1. **SF-1 — Mo's own λ̂(T=128) estimator across ALL regimes on the trained models.** The referee notes the 44% deviation figure implies partial data exists in `v2-full-runs` — **first check if the full-regime estimator sweep was already run**; if yes, this is a plotting/extraction job (produce the curve beside the exact-gap prediction for Fig 2). If not, run Mo's estimator (his finite-horizon protocol, per `mo-deep-read`) on the same trained checkpoints across overdamped→EP→underdamped. Deliverable: the estimator-vs-exact-gap-vs-measurement overlay so "Mo's law is the overdamped face" rests on Mo's *own* estimator, not a substituted predictor.
2. **SF-3 — GMOR tilt-sweep + EP onset on an ANCHORED 3000-ep checkpoint.** The headline laws (§3.1/3.2) ran on 150-ep checkpoints (pre-erosion). Re-verify GMOR μ²∝δ + EP √(h−h*) onset on a checkpoint trained to 3000 ep WITH the anchor (λ=100, the bulletproof setting). Confirms the headline laws survive past the erosion horizon under the shipped cure. 3 seeds. (Use `chlu exp-d --anchor-lambda 100` — the fix-pack-4 flag now ships this.)
3. **SF-2 — per-step FLOP/wall-time ratio: one CLU dissipative-Verlet step (hidden-64 MLP potential, KDK) vs one LSTM cell vs one LEM step.** So the "263 vs 69 map-steps ≈4× longer retention" claim can be stated (or retired) in compute-normalized terms. Trivial to measure. Deliverable: the ratio + the compute-normalized retention statement (if the 4× survives) or a recommendation to lead with the qualitative triad only.

**Report:** the three deliverables with numbers + which draft section each feeds (v2-revision consumes them). Flag-provenance per §5. If SF-1's estimator sweep turns out already-run, say so and just extract.
