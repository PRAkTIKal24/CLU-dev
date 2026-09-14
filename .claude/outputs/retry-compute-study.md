# retry-compute-study — experiment-engineer report

**Task + acceptance:** promote w22's single retry point into a defensible accuracy-vs-compute *curve* on the
Hopfield/U-Hop retrieval protocol — retry ladder k∈{0,1,2,4,8}, confidence-gated (threshold swept), with five
pre-registered controls (ungated-all, ensemble-of-k, random-kick, feedforward-matched, Hopfield-k-steps), at
≥2 loads (M) × ≥2 degradation levels, plus a created-benchmark spec page and pre-registrations. Tests green, x64-safe rollout reused.

**Status: done.** Curve is real and monotone; **mechanism attribution SURVIVES all controls** (the novelty-with-teeth);
but the trivial feedforward-NN floor **dominates in absolute terms** — reported plainly below. 7 new tests green, ruff clean, config round-trip green.

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). 2 sites.**
> 1. **CM-23's "an accuracy-vs-compute curve feedforward memories cannot draw" must be SPLIT into two claims.** MEASURED: the feedforward-NN floor sits at **0.99–1.00 (flat at ceiling)** on masked-pixel retrieval and beats CLU-gated in **every** cell (gap −3.5 to −42.2 pp). What survives is only the *shape* statement: NN is flat — it **cannot rise** with compute because it is already saturated; CLU draws a genuine *rising* curve. The *absolute-dominance* half of CM-23 is **FALSIFIED** on this benchmark (same uncomfortable truth as `hopfield-capacity-benchmark` §3: the NN floor dominates masked-image retrieval). Any paper text asserting CLU "beats feedforward via test-time compute" must be corrected to "draws a rising curve feedforward cannot, while remaining below the saturated NN floor."
> 2. **The novelty claim to keep** (survives with teeth): the directed **Lorentz-boost-style re-launch is the mechanism** — random-kick (equal energy) and ensemble-of-k (k independent restarts) are **dead flat** in every cell, so the lift is NOT stochastic-restart and NOT "just k tries." This is registry-grade and should anchor the retry positioning instead of an absolute-benchmark-win claim.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / commit | `agent/experiment-engineer/retry-compute-study`; base local `main @ 7ff0651`; commit **`60a020c`** |
| worktree | `../CHLU-retry-compute-study` (isolated; 3 concurrent engineer worktrees this wave — dimbudget, phi-read-in, this) |
| dataset | MNIST (`mnist_784` openml, cached), pixels `[0,1]` (ToTensor convention) |
| seed | 0 (single seed) |
| query protocols | **mask** = `torch.dropout(x,p)` (w22 capacity protocol); **noise** = `clamp(\|x+N(0,σ)\|,0,1)` (UHop noise harness) |
| loads M | 128, 256 · degradation: mask p∈{0.5,0.7}, noise σ∈{0.2,0.3} (headroom cells: first-pass ∈ [0.05, 0.92]) |
| CLU register | `GaussianMemoryPotential`, `s=0.3·median-NN`, `b=1`, `alpha=1e-3`, `gamma=0.1`, **clu_steps=150** (=1 compute unit), `dt=0.5·s/√b` (auto), Newtonian-identity, read = mean of last 10% of the damped Verlet rollout |
| retry ladder | k∈{0,1,2,4,8}; **retry_step_frac=0.1** (lowest-conf 10%/round); **retry_boost=1.5**; lock-on-retry |
| gate | confidence = cosine to nearest stored well; eligible = `(not locked) ∧ (cos < τ)`; **main τ=0.99**; τ swept ∈ {0.95,0.97,0.99,1.0} |
| controls | ungated-all (no gate/lock), ensemble-of-k (k+1 random-momentum starts, best-cosine), random-kick (equal-energy random dir), feedforward-NN (k+1 TTA-augmented votes, `ff_aug_sigma=0.1`), Hopfield-k-steps (β auto-sharpened, iterated k+1) |
| compute unit | total relaxation steps / (Nq·clu_steps). CLU methods at MEASURED multiplier (gated/kick auto-stop → sub-linear); feedforward-NN & Hopfield placed at matched *budget* multiplier (k+1) — **generous to the baselines** (one NN/Hopfield read ≪ one CLU settle) |
| JAX | main venv reused (protocol §4), float32 (x64 OFF for reported numerics; code is x64-safe) |
| designed vs learned | everything designed/closed-form on ALL six lines. Nothing learned (N46-admissible) |

---

## 1. The headline curve (Item 1) — flagship cell MASK, M=256, p=0.5 (first-pass 0.359)

Identity accuracy @ compute-multiplier, per k:

| line | k0 | k1 | k2 | k4 | k8 | shape |
|---|---|---|---|---|---|---|
| **clu_gated** | 0.359@1.0× | 0.461@1.1× | 0.562@1.2× | 0.738@1.4× | **0.961@1.64×** | ⭐ monotone rise, **auto-stops** |
| ungated_all | 0.359@1.0× | 0.684@2.0× | 0.059@3.0× | 0.004@5.0× | 0.004@9.0× | **collapses** (no gate) |
| random_kick | 0.359@1.0× | 0.359@1.1× | 0.359@1.2× | 0.355@1.4× | 0.348@1.64× | **dead flat** |
| ensemble | 0.359@1.0× | 0.359@2.0× | 0.359@3.0× | 0.359@5.0× | 0.359@9.0× | **dead flat** |
| feedforward_nn | 0.996@1.0× | 0.949@2.0× | 0.922@3.0× | 0.930@5.0× | 0.926@9.0× | flat-at-ceiling (TTA noise nudges it *down*) |
| hopfield_ksteps | 0.719@1.0× | 0.672@2.0× | 0.652@3.0× | 0.633@5.0× | 0.629@9.0× | declines with iteration |

**Reading:** CLU-gated climbs 0.36→0.96 and *stops spending* at ×1.64 (the low-confidence tail exhausts). Every CLU control is flat or collapses. The feedforward-NN line is above everything but **does not rise** — it is already saturated.

## 2. Item 1 — the full (M, level) grid: gated lift and control gaps (best-over-ladder)

**MASK protocol** (the boost's native regime):

| M | p | first | gated best (@compute) | lift | ungated gap | kick gap | ensemble gap | **NN gap** | hopfield gap |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.5 | 0.570 | 0.938 @1.41× | +36.7 | +36.7 | +35.2 | +36.7 | **−5.5** | +8.6 |
| 128 | 0.7 | 0.102 | 0.859 @1.81× | +75.8 | −1.6 | +75.8 | +75.8 | **−13.3** | +3.9 |
| 256 | 0.5 | 0.359 | 0.961 @1.64× | +60.2 | +27.7 | +60.2 | +60.2 | **−3.5** | +24.2 |
| 256 | 0.7 | 0.047 | 0.809 @1.81× | +76.2 | −10.2 | +76.2 | +76.2 | **−17.6** | +12.9 |

**NOISE protocol** (the boost is weaker; a genuine partial-null):

| M | σ | first | gated best (@compute) | lift | kick gap | ensemble gap | **NN gap** | hopfield gap |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.2 | 0.922 | 0.922 @1.0× | +0.0 | +0.0 | +0.0 | −7.8 | +47.7 |
| 128 | 0.3 | 0.602 | 0.734 @1.38× | +13.3 | +13.3 | +13.3 | −26.6 | +49.2 |
| 256 | 0.2 | 0.820 | 0.887 @1.19× | +6.6 | +6.6 | +6.6 | −11.3 | +54.3 |
| 256 | 0.3 | 0.230 | 0.578 @1.77× | +34.8 | +34.8 | +34.8 | −42.2 | +35.9 |

Positive gap = gated beats that control. **kick/ensemble gaps ≈ full lift in every cell** (they never move) → mechanism survives.
**NN gap negative in every cell** → the feedforward floor dominates absolutely.

## 3. Item 2 — the threshold sweep (MASK M=256 p=0.5) — the gate quantified

| τ | k0 | k2 | k4 | k8 (final @compute) |
|---|---|---|---|---|
| 0.95 (strict) | 0.359 | 0.562 | 0.738 | 0.828 @1.50× (under-retries, stops early) |
| 0.97 | 0.359 | 0.562 | 0.738 | 0.957 @1.64× |
| **0.99 (sweet spot)** | 0.359 | 0.562 | 0.738 | **0.961 @1.64×** |
| 1.00 (no gate) | 0.359 | 0.562 | 0.738 | **0.828 @1.81×** (over-retries → *lower* acc at *higher* compute) |

The gate has a clear optimum: too strict leaves recoverable misses; **no gate corrupts already-correct reads AND costs more** (0.828@1.81× vs 0.961@1.64×). This is the w22 −38pp blank-guard, now a continuous curve.

## 4. Pre-registration scorecard (`PREREG.md`, written before the harness)

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | gated rises, monotone, **dominant at equal compute** | rises & monotone (+6.6…+76.2pp, auto-stops) but **NOT dominant** — NN floor is above it | ◐ shape right, **dominance FALSE** |
| P2 | ungated below gated, may dip (blank-guard) | **collapses** at moderate first-pass (0.96→0.004); at near-chance first-pass it helps (few correct to corrupt) and can beat gated's *peak* but at 9× compute | ✅-in-spirit (gate load-bearing on the compute axis) |
| P3 | ensemble below gated; if it matches, mechanism dies | **dead flat** in all 8 cells | ✅ mechanism survives vs ensemble |
| P4 | random-kick below gated; ±3pp match = falsified | **dead flat** in all 8 cells (gap = full lift) | ✅ **mechanism survives — boost ≫ kick** |
| P5 | feedforward flat, not rising | **flat at ceiling 0.99–1.0** (TTA even nudges down) | ✅ shape confirmed… |
| P5b | (implicit) CLU beats feedforward | **FALSE** — NN dominates every cell (−3.5…−42.2pp) | ❌ absolute-dominance falsified |
| P6 | Hopfield flat after step 1 | **declines** with iteration; below gated in all cells | ✅ |

**Honest summary:** my central optimistic hypothesis (P1 dominance / P5b beat-feedforward) is **falsified** — the NN floor dominates. The *mechanism-attribution* pre-registrations (P3, P4) — the ones that actually carry the novelty — **survive decisively**: the directed boost is doing work no equal-energy kick or k-restart ensemble reproduces.

## 5. What this means (the teeth)

- ✅ **A real, novel capability:** CLU retrieval has a *test-time-compute knob* — a monotone, auto-stopping accuracy-vs-compute curve — that a single-shot associative memory structurally lacks. It is the physics (a directed symplectic re-launch), not a heuristic restart: **kick and ensemble are flat**.
- ✅ **The gate is load-bearing and self-limiting:** confidence-gating (τ≈0.99) both prevents corruption (vs ungated collapse) and **auto-caps compute** once the low-confidence tail is spent.
- ⛔ **It does not win the benchmark:** on masked-pixel MNIST the trivial NN floor is at ceiling and beats CLU everywhere. The curve is a *capability*, not a leaderboard result — exactly the honest framing `hopfield-capacity-benchmark` reached.
- ◐ **Protocol-dependent:** the boost recovers **mask** queries strongly (+36…+76pp) and **Gaussian-noise** queries weakly (+6.6…+34.8pp, 0 at ceiling). At σ≥0.4 (past the basin-capture cliff, `hopfield-capacity` §2.2) every arm is at chance and no retry can recover a query that has left every well.

---

## 6. Item 3 — CREATED-BENCHMARK SPEC (one page)

**Name.** *Retrieval-under-degradation at a compute budget* (RUD-C). An **anchored** benchmark — every choice is inherited from the existing w22 Hopfield/U-Hop protocol (`MAGICS-LAB/UHop @ cdac754`), not invented, so it inherits that protocol's discount-resistance.

**Task.** Store `M` patterns (MNIST images, `[0,1]`, `D=784`) as a designed/closed-form associative memory. Present one degraded query per stored pattern: **mask** = `torch.dropout(x,p)` (zero fraction `p`, scale survivors `1/(1-p)`) or **noise** = `clamp(|x+N(0,σ)|,0,1)`. Recover the stored pattern.

**The metric (the novel axis).** **Accuracy at compute budget c** — identity-retrieval accuracy (`argmin_i ‖x̂−ξ_i‖ == true i`) as a function of **total relaxation-step count** normalised by the first-pass budget `Nq·(steps/query)`. A method is scored by its **accuracy-vs-compute curve**, and compared by **curve dominance** (does line A lie above line B at every compute budget?). Wall-clock is a secondary axis.

**Cells.** A grid of ≥2 loads `M` × ≥2 degradation levels, run for BOTH query protocols (the mask/noise contrast is diagnostic — a method that only helps one is protocol-specific, and the spec makes that visible).

**Mandatory baselines (a submission is invalid without them).**
1. **Feedforward-NN matched-compute** (Item 2.4) — nearest-neighbour over the store with `k+1` test-time augmentations, majority-voted, plotted at the same budget. *This is the honesty floor:* on masked-pixel retrieval it sits at ceiling, so any "test-time compute wins" claim must show the curve **relative to it**, not in a vacuum.
2. **Stochastic-restart control** (Item 2.2) — `k` independent restarts, best-confidence answer. Separates a genuine directed-compute mechanism from "just try k times."
3. **Equal-energy random-perturbation control** (Item 2.3) — replace the method's directed step with an equal-energy random one. Separates the *mechanism* from *any* perturbation.
4. **No-gate control** (Item 2.1) — spend the extra compute on every query, not just the uncertain ones. Quantifies the value of the confidence gate.

**How an external method plugs in.** Implement two callables: `read(store, queries) -> reads` (the first pass, 1 compute unit) and `retry(store, queries, reads, confidence, budget_k) -> reads'` (the extra-compute pass; may inspect a label-free confidence). The harness supplies the store, queries, ground-truth indices, and the four baselines; it returns each method's accuracy-vs-compute curve and the pairwise dominance verdict. A modern-Hopfield / attention memory plugs in with `read` = one softmax update and `retry` = additional update steps (the `hopfield_ksteps` line is the reference implementation).

**Scoring rule.** Report (a) the curve, (b) dominance over each of the four mandatory baselines at matched compute, (c) the compute multiplier at which accuracy saturates (the *adaptive-compute efficiency*). A method "passes" the mechanism bar only if it dominates the stochastic-restart AND equal-energy controls; it "wins" the benchmark only if it also dominates the feedforward-NN floor.

**Provenance discount note.** Every knob (mask/noise forms, pixel range, store construction, identity-accuracy metric) is taken verbatim from the U-Hop harness; the ONE addition is the compute axis and the dominance metric. Nothing is tuned to flatter CLU (`s=0.3·median-NN` is the fixed w22 rule; `retry_step_frac`, `retry_boost`, `τ` are the only new knobs and their sweeps are reported).

---

## How I verified

- `pytest tests/test_retry_compute.py` → **7 passed**; `tests/test_config.py tests/test_retry_compute.py` → **14 passed** (exhaustive config round-trip incl. the new dataclass green). `ruff check` clean on all four touched files.
- Full experiment `python -m chlu.experiments.exp_retry_compute --seed 0` → completed exit 0, ~35 min wall (8 cells × 6 lines × ladder + threshold sweep; the ungated/ensemble full-population settles dominate cost). Wrote `results/exp_retry_compute_metrics.json` (48 KB) + `retry_compute_grid_mnist_{mask,noise}.png`. All numbers above re-derived from that JSON (copied to `.claude/outputs/retry-compute-study/results/`).
- CLI `chlu exp-retry-compute [--project N] [--seed I] [--quick] [--dataset …]` parses to `cmd_exp_retry_compute` (verified via `setup_experiment_parsers`); module runnable as `python -m chlu.experiments.exp_retry_compute --quick`.
- Diagnostics (in-report): confirmed cosine-to-nearest-well is a *ranking* signal (correct 0.998 vs wrong 0.949) but NOT an *acceptance* signal, motivating the lock-on-retry + threshold design; confirmed the boost recovers mask queries (+40pp round-1) and not noise queries; confirmed no per-item label-free signal (cosine/distance-to-query/energy) detects a good flip.

## Findings/results — see §1–§5. One-line: **the curve is real and the mechanism survives every control, but the feedforward-NN floor dominates in absolute terms; keep the mechanism claim, retract the beat-feedforward claim.**

## Git footprint

- **Branch** `agent/experiment-engineer/retry-compute-study`, base local `main @ 7ff0651`. Rebase onto local `main` = up-to-date (no-op; base unmoved). **Not pushed, no PR** (per protocol/task). Verified from main repo: `main..branch` shows `60a020c`.
- **Commit (1):** `60a020c` — retry-compute study.
- **Files:** **+** `chlu/experiments/exp_retry_compute.py`, `tests/test_retry_compute.py`; **M** `chlu/config.py` (+`ExperimentRetryComputeConfig` + load/save/master reg), `chlu/cli/experiment_cmd.py` (+import, parser, handler). No shared files reformatted beyond my additive hunks; `utils/plotting.py` untouched (figures local, per `exp_hopfield_capacity`/`exp_retrieval` precedent); `results/` not committed.
- **Isolation:** worked in a dedicated worktree `../CHLU-retry-compute-study` (protocol §3.2 — 3 concurrent engineer worktrees). Main venv reused (`PYTHONPATH` + `/Users/user/Desktop/CHLU/.venv/bin/python`), so no worktree-venv JAX drift (§4). No collision: my `config.py`/`experiment_cmd.py` edits are additive registrations; if a concurrent engineer also touched these files, the Hub merges branch-by-branch (my hunks don't overlap theirs — new dataclass appended after `ExperimentHopfieldCapacityConfig`, new CLI block after `exp-hopfield-capacity`).

## Open questions / follow-ups / risks

- **Single seed (0), MNIST only.** CIFAR needs the local parquet (see `hopfield-capacity` §6); φ-space stretch untouched (coordinate with `phi-read-in` — I did not share its worktree). Multi-seed error bars are the obvious next step.
- **Feedforward floor caveat.** The NN floor's dominance is a property of *masked-pixel* retrieval (surviving pixels uniquely identify the pattern). A protocol where the query lives in a space the store is NOT metric-native to (e.g. cross-modal, or φ-latent) could flip this — that is where the mechanism claim would gain teeth. Flagged for the Hub.
- **The mask/noise gap is a lead, not just a null.** The boost recovers mask (structured erasure) far better than Gaussian noise. Worth a theory pass: the boost aims from a wrong well toward the (partially-observed) query; for a mask the observed pixels pull toward the true basin, for full-field noise they do not.
- **Compute-axis honesty.** feedforward-NN and Hopfield are plotted at their *budget* multiplier (k+1); one such read is far cheaper than a CLU settle. This is generous to the baselines and stated everywhere; a wall-clock axis would move them left (further favouring them). The relaxation-step axis is the CLU-native honest unit per the task.

## Proposed handover updates (for the Hub)

1. **⭐ §6 ground truth — retry is a curve with a surviving mechanism, NOT a benchmark win.** On the anchored RUD-C protocol (MNIST, seed 0): CLU-gated retry draws a monotone, **auto-stopping** accuracy-vs-compute curve (+6.6…+76.2pp, saturating ×1.2–1.8). **Mechanism attribution SURVIVES:** equal-energy random-kick and ensemble-of-k-restarts are **dead flat** — the directed boost is the mechanism. **The confidence gate is load-bearing:** ungated retry-all collapses (0.96→0.004) at 9× compute; the τ-sweep shows over-retry corrupts. **BUT the feedforward-NN floor dominates in absolute terms (0.99–1.0) in every cell** — CLU's curve does not beat it on masked-pixel retrieval.
2. **⛔ §7 / claims-matrix — CM-23 must be split (reconciliation, owner needed).** "An accuracy-vs-compute curve feedforward memories cannot draw" is TRUE only as a *shape* claim (NN is saturated → cannot rise); the *absolute-dominance* reading is **FALSIFIED** (NN beats CLU-gated by 3.5–42.2pp). Retag: the retry claim is a **mechanism/capability** claim, not a leaderboard claim. This mirrors the w22 hopfield NN-floor finding — same physics, same honesty.
3. **New CLI/config/experiment surface:** `chlu exp-retry-compute`; `ExperimentRetryComputeConfig` (registered in load/save/master); `chlu/experiments/exp_retry_compute.py`; `tests/test_retry_compute.py`. `retry_step_frac=0.1`, `main_threshold=0.99`, `retry_boost=1.5` are the only new knobs (measurement-motivated: cos|correct≈0.998 vs cos|wrong≈0.949 → τ=0.99 gates the wrong tail cleanly).
4. **Design note for future retry work (avoid re-deriving):** cosine-to-nearest-well is a good *ranking* signal but a **useless acceptance signal** post-settle (a boost into the right well does not raise it above the wrong well it left). Gate + lock-on-retry, do NOT rely on a per-item accept rule. Recorded so nobody re-hits the "zero-lift" trap I hit first.
5. **Created-benchmark spec (§6 above) is ready to lift** into a paper appendix — it is anchored to U-Hop and specifies the four mandatory baselines that make a "test-time-compute wins" claim honest.
