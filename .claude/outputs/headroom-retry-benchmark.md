# headroom-retry-benchmark — experiment-engineer report

**Task + acceptance:** build ≥2 **ambiguity** (not noise) regimes where all methods land ~0.6–0.7, gate the
headroom **before** the full grid, then run the RUD-C ladder (k∈{0,1,2,4,8} + 5 controls, ≥3 seeds) and answer
Item 4: *does a regime exist where CLU-gated retry beats the feedforward-NN floor at matched compute?*
**Status: done.**

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). 2 sites.**
> 1. **R3 leaderboard verdict is decision-grade NO.** In **both** ambiguity regimes, across all 8 cells × 3
>    seeds, CLU-gated retry does **not** beat the correct feedforward floor. Against the **ML-optimal
>    observed-dimensions-only NN** (`feedforward_nn_masked`, the right rule under erasure) CLU loses by
>    **−3.5 … −38.0 pp in every cell** — that floor sits at **1.000** everywhere. The single apparent win vs the
>    *naïve full-vector* NN (crowded:mask M=256 p=0.5: **+0.65 ± 0.49 pp**) is marginal AND evaporates against the
>    masked floor (−22.9 pp). **CM-23(b)'s absolute-dominance half stays retracted; the shape/mechanism half
>    survives.** This is the same uncomfortable truth as w23/N90, now measured with headroom present so it cannot
>    be blamed on saturation.
> 2. **The mechanism claim survives with teeth (registry-grade).** random-kick and ensemble are flat-or-declining
>    in all 8 cells; ungated-all collapses at high compute; only the **directed** boost draws a monotone,
>    auto-stopping rising curve (+2.3 … +36.2 pp). ⚠ **New nuance to record:** under *geometric* crowding the
>    gated−kick peak margin shrinks to **+2.3 … +5.5 pp** (kick still actively *declines* while gated rises), i.e.
>    the boost's *direction* buys less as the store gets more ambiguous — partial trigger of pre-registered M5.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / commit | `agent/experiment-engineer/headroom-retry-benchmark`; base local `main @ 5e466c0`; commit **`eac7bb0`** |
| worktree | `../CHLU-headroom-retry` (isolated; main venv reused via `PYTHONPATH` + `.venv/bin/python`) |
| dataset | MNIST (`mnist_784` openml, cached), pixels `[0,1]` |
| seeds | **0,1,2** (3 seeds; every headline cell). τ-sweep on seed 0 only (`sweep_seeds=1`) |
| regimes | **`iid:block`** (contiguous occlusion, `block_rescale=True`, f∈{0.2,0.3}) · **`crowded:mask`** (NN cluster contracted `crowd_rho=0.25`, iid dropout p∈{0.3,0.5}) |
| loads M | 128, 256 |
| CLU register | `GaussianMemoryPotential`, `s=0.3·median-NN` (store-adaptive), `b=1`, `alpha=1e-3`, `gamma=0.1`, `clu_steps=150`, `dt=0.5·s/√b`, Newtonian-identity, read=mean last 10% of damped Verlet |
| retry ladder | k∈{0,1,2,4,8}; `retry_step_frac=0.1`; `retry_boost=1.5`; lock-on-retry; main τ=0.99 (swept {0.95,0.97,0.99,1.0}) |
| controls (7 lines) | ungated-all, ensemble-of-k, random-kick, feedforward-NN (TTA k+1 votes), **feedforward-NN-masked (NEW: ML-optimal, observed-dims-only)**, Hopfield-k-steps |
| compute unit | relaxation steps / (Nq·clu_steps); CLU at MEASURED multiplier (auto-stop → sub-linear); NN/Hopfield/masked at budget k+1 — **generous to baselines**, stated |
| JAX | main venv reused, float32 (x64 OFF for numerics; code x64-safe) |
| designed vs learned | everything designed/closed-form; nothing learned (N46-admissible) |
| wall time | full 3-seed grid ≈ 48 min; headroom gate (per arm) ≈ 20–28 s |

Pre-registrations: `PREREG.md` (regimes + Item-4, before any harness) and `PREREG_ADDENDUM.md` (gate iteration 2
amendments, written before those arms were run). Raw: `results/exp_retry_compute_metrics.json`,
`results/gate2_seed0.json`; figures `results/retry_compute_grid_mnist_{iid-block,crowded-mask}.png`.

---

## Item 2 — the headroom gate (run BEFORE the grid; this is the crux the task asked for first)

**Gate iteration 1 (as pre-registered in `PREREG.md`): 0/14 cells passed** — and it diagnosed *why*, cheaply:

- **iid:block with the inherited `1/(1-f)` survivor rescaling = DESTRUCTION, not ambiguity.** Under *contiguous*
  erasure the rescaling amplifies a surviving crop 2–3.3× and throws the query off-manifold: at f=0.4/M=128,
  first-pass **0.086**, NN floor **0.383** — both lines destroyed (the σ≥0.4-cliff failure the task forbids).
- **The plain NN-cluster crowded store creates NO ambiguity:** NN floor **1.000** at every load/level, because
  `s = 0.3·median-NN` is **store-adaptive** — a k× tighter cluster gets k× tighter wells and the NN rule is
  scale-free. The reported packing slack was **1.075 in all 14 cells** = the tautology `1/(3.1·0.3)`; the metric
  used a per-element σ_q against a vector median-NN (a √D=28 unit mismatch), so it measured `clu_s_frac`, not the
  store. (This retracts the "w23 iid ran at slack ≈1.08" claim — corrected value **0.23**, see A1 below.)

**Two amendments (declared in `PREREG_ADDENDUM.md` before measuring), then gate iteration 2 → 4/28 cells passed:**

| arm | selected cells (first-pass, naïve NN floor, slack) | gate |
|---|---|---|
| `iid:block` rescale=True | **f=0.2 M=128** (0.64, 0.945, 0.34) · **f=0.2 M=256** (0.53, 0.902, 0.33) | PASS×2 |
| `iid:block` rescale=False | f=0.3–0.7: first 0.30→0.004, NN 0.55→0.008 — total destruction | 0/6 (rejected) |
| `crowded:mask` rho=1.0 | NN floor 0.99–1.00 everywhere | 0/4 (rejected) |
| `crowded:mask` rho=0.5 | NN floor 0.98–1.00 | 0/4 (rejected) |
| `crowded:mask` rho=0.25 | **p=0.5 M=128** (0.63, 0.734, 0.06) · **p=0.5 M=256** (0.68, 0.738, 0.06) | PASS×2 |

The **`apply_ambiguity` preset ships the gate-selected levels** (block f∈{0.2,0.3}, crowd_rho=0.25 p∈{0.3,0.5}) —
the second level per regime is the neighbouring off-ceiling cell, included so each regime has a load×level grid.
**A regime was selected for HEADROOM, never for whether CLU wins in it** (task ⚠ standing trap).

---

## Item 1 + 3 — the full curves (3-seed mean ± sd; best-over-ladder)

Every cell now has real headroom (first-pass 0.26–0.93; naïve NN 0.74–0.96 — off ceiling). Full per-rung tables:
`.claude/outputs/headroom-retry-benchmark/rendered.md`. Best-over-ladder summary:

**iid:block (contiguous occlusion — the boost's native regime):**

| M | f | first | gated best @compute | lift | kick gap | ens gap | ungated gap | **naïve-NN gap** | **masked-NN gap** | hopf gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.2 | 0.612 | 0.800 @1.39× | +18.8 | +18.5 | +18.8 | +18.8 | **−11.7** | **−20.1** | +5.2 |
| 128 | 0.3 | 0.349 | 0.695 @1.64× | +34.6 | +34.1 | +34.6 | +32.3 | **−8.9** | **−30.5** | +5.5 |
| 256 | 0.2 | 0.516 | 0.751 @1.48× | +23.6 | +23.6 | +23.6 | +23.6 | **−13.8** | **−24.9** | +14.8 |
| 256 | 0.3 | 0.258 | 0.620 @1.74× | +36.2 | +36.1 | +36.2 | +24.2 | **−12.5** | **−38.0** | +9.0 |

**crowded:mask (geometric ambiguity, contracted store):**

| M | p | first | gated best @compute | lift | kick gap | ens gap | ungated gap | **naïve-NN gap** | **masked-NN gap** | hopf gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.3 | 0.932 | 0.956 @1.20× | +2.3 | +2.3 | +2.3 | +0.5 | **−1.6** | **−4.4** | +91.9 |
| 128 | 0.5 | 0.737 | 0.786 @1.81× | +4.9 | +4.9 | +4.7 | +0.5 | **−1.6** | **−21.3** | +74.7 |
| 256 | 0.3 | 0.926 | 0.963 @1.41× | +3.8 | +3.8 | +3.7 | +0.1 | **−0.6** | **−3.7** | +93.2 |
| 256 | 0.5 | 0.715 | 0.770 @1.81× | +5.5 | +5.5 | +4.7 | +1.0 | **+0.5** | **−23.1** | +73.6 |

**Reading.** Positive kick/ens/ungated gaps = mechanism survives. **Every naïve-NN gap is ≤ +0.5 pp; every
masked-NN gap is negative.** The masked-NN oracle is **1.000 in all 8 cells** — knowing *which* coordinates were
erased makes identity trivial on a designed MNIST store at any ambiguity level, and neither CLU nor the naïve NN
exploits that. That is the honest floor, and CLU is below it everywhere.

## Item 4 — the verdict that matters

**`item4_any_regime_benchmark_win = False`.** No regime — of the two built, at any load or level, on 3 seeds —
has CLU-gated retry beating the feedforward floor at matched compute once the floor is the correct ML-optimal
erasure rule. `item4_any_regime_beats_nn = True` reflects **one marginal cell vs the *naïve* NN**
(crowded:mask M=256 p=0.5, **+0.65 ± 0.49 pp**), which (i) is within ~1.3 sd of zero and (ii) loses to the masked
oracle by −22.9 pp. **The decision-grade answer to R3's leaderboard question is NO.**

## Mechanism controls (the claim that DOES survive)

- **random_kick:** flat or *declining* in all 8 cells — in crowded:mask it monotonically *falls* (0.932→0.839)
  while gated rises. Gap = +2.3 … +36.1 pp.
- **ensemble-of-k:** dead flat in all 8 cells (±≤0.01). Gap = +2.3 … +36.2 pp.
- **ungated_all:** collapses at high compute (e.g. 0.932→0.000 at 9× in crowded M=128 p=0.3) — the gate is
  load-bearing and self-limiting (gated auto-stops at ×1.2–1.8; the τ-sweep confirms τ=1.0 over-retries and drops
  accuracy, e.g. block M=128 f=0.2: 0.828@1.35× at τ=0.99 vs 0.453@1.81× at τ=1.0).
- **The directed boost is the mechanism** — no equal-energy kick and no k-restart ensemble reproduces the rise.

## Pre-registration scorecard

| # | registered | measured | verdict |
|---|---|---|---|
| A1 | corrected slack, w23 iid mask p=0.5 M=128 = 0.25±0.15 (retract "1.08") | **0.227** | ✅ |
| A2 | block rescale=True passes near f≈0.2, NN 0.90±0.08, ~40% | f=0.2/M=128: first 0.64, NN **0.945**, PASS | ✅ |
| A3/A4 | block rescale=False f=0.5/0.7 mildly degrading | **total destruction** (first 0.016/0.008) | ❌ (rescale=False unusable on MNIST) |
| A5 | crowd rho=0.25 p=0.3 passes ~45% | p=0.3 fails (NN 0.953, first 0.914); **p=0.5 passes** | ◐ |
| A6 | ≥1 cell passes both gate halves in iter 2, likely an amendment cell | **4 cells** pass (block f=0.2, crowd rho0.25 p=0.5) | ✅ |
| A7 | NN floor monotone ↓ in f and in 1/ρ | block 1.0→0.945→0.773; crowd 1.0→1.0→0.953 | ✅ |
| **H-A** | **NO — NN floor dominates every ambiguity cell (~70% prior)** | **confirmed vs masked-NN in 8/8 (−3.5…−38pp)** | ✅ **primary hypothesis holds** |
| H-B | YES in R-CROWD high load, ≥3 seeds, margin > 1 sd | only +0.65±0.49 vs naïve NN, loses to masked oracle | ❌ rejected |
| fairness rider | if gated beats naïve NN, masked-NN restores dominance (~80%) | the one +0.65pp cell → masked gap −22.9pp | ✅ **rider vindicated** |
| M1/M2 | kick & ensemble flat in every new cell | flat/declining, 8/8 | ✅ |
| M3 | ungated collapses/underperforms at ≥3× where first>0.3 | collapses to ~0 at 9× | ✅ |
| M4 | gated rises monotone, auto-stops ×1.2–1.9 | rises, stops ×1.20–1.81 | ✅ |
| M5 | gated−kick gap falls to +10…+40 but stays > +5 | iid:block +18–36 ✅; **crowded +2.3–5.5 (soft-falsified)** | ◐ **new finding** |

## How I verified
- `pytest tests/test_retry_compute.py tests/test_config.py` → **27 passed** (7 new/updated w24 tests + config
  round-trip incl. the new knobs + a regression guard that the defaults reproduce the w23 grid). `ruff check` clean
  on all 4 touched files.
- Headroom gate: `python -m chlu.experiments.exp_retry_compute --headroom` over the pre-registered candidate arms
  (`gate.py`/`gate2.py`), 0/14 then 4/28 cells — numbers above are from `results/gate2_seed0.json`.
- Full grid: `python -m chlu.experiments.exp_retry_compute --ambiguity --seed 0` (3 seeds via `n_seeds=3`),
  exit 0, ≈48 min, wrote `exp_retry_compute_metrics.json` (183 KB) + 2 figures. All numbers re-derived from that
  JSON (`render.py`, copied to `.claude/outputs/headroom-retry-benchmark/results/`).
- CLI: `chlu exp-retry-compute --ambiguity --headroom` parses to `cmd_exp_retry_compute` with both flags; preset
  resolves to the gate-selected regimes/levels (verified by direct parser+config introspection).

## Findings/results — see the tables. One-line: **headroom was successfully built out of ambiguity (both
regimes land 0.26–0.93 with the NN off ceiling), the CLU-gated curve is real and every mechanism control survives,
but NO ambiguity regime lets CLU beat the ML-optimal feedforward floor at matched compute — the leaderboard claim
for R3 is a decision-grade NO; the mechanism claim stands.**

## Git footprint
- **Branch** `agent/experiment-engineer/headroom-retry-benchmark`, base local `main @ 5e466c0`. Rebase onto local
  `main` = up-to-date (no-op). **Not pushed, no PR.** Verified from main repo: `main..branch` shows only `eac7bb0`.
- **Commit (1):** `eac7bb0` — w24 ambiguity/headroom regimes + Item-2 gate + masked-NN oracle line.
- **Files:** **M** `chlu/experiments/exp_retry_compute.py` (block/crowd/gate/masked-NN/multi-seed/aggregate),
  `chlu/config.py` (+`block_rescale`, `crowd_rho`; both default = current behaviour), `chlu/cli/experiment_cmd.py`
  (`--ambiguity`/`--headroom` already present from a prior in-worktree edit; handler wired), `tests/test_retry_compute.py`
  (+7 w24 tests). Config registered at **all three sites** (dataclass, load/save round-trip via the exhaustive test,
  `save_config` params list in the experiment) — the task's "three sites plus `save_config`" requirement.
- **Isolation:** dedicated worktree, main venv reused (no worktree-venv JAX drift). Additive-only edits to
  `config.py`/`experiment_cmd.py` (new fields appended to `ExperimentRetryComputeConfig`, new CLI flags on the
  existing `exp-retry-compute` parser) — no overlap with concurrent engineer branches.

## Open questions / follow-ups / risks
- **Two regimes, MNIST, 3 seeds.** Both are erasure-type (block occlusion + crowded-store-under-dropout). A
  regime where the query lives in a space the store is **not metric-native to** (cross-modal, φ-latent) is the only
  place left where the masked-NN oracle would stop being trivially 1.000 — flagged for the Hub as the one untested
  route to a leaderboard win (coordinate with `phi-read-in`/`phi-stream-discipline`).
- **The masked-NN oracle being exactly 1.000 everywhere is itself the finding:** on a designed associative store,
  identity retrieval under *known* erasure is information-theoretically trivial regardless of how much you erase —
  so "ambiguity" only bites methods that ignore the erasure mask. This bounds what *any* retrieval method can claim
  on erasure benchmarks and is worth stating in the RUD-C spec (baseline #1 should be the masked oracle, not the
  naïve NN).
- **M5 soft-falsification (crowded regime, gated−kick peak +2.3–5.5 pp)** is a genuine physics lead: the boost's
  *direction* carries less information as geometric ambiguity rises, even though the boost still helps and the kick
  actively hurts. A theory pass on "directed-boost value vs basin overlap" is warranted.

## Proposed handover updates (for the Hub)
1. **§6 / claims-matrix — R3 leaderboard verdict is a decision-grade NO (owner needed).** Built the headroom out of
   **ambiguity** (block occlusion + contracted crowded store; 3 seeds); confirmed both regimes land off-ceiling
   (first 0.26–0.93). **CLU-gated retry does not beat the ML-optimal `feedforward_nn_masked` floor (=1.000) in any
   of 8 cells (−3.5…−38 pp).** CM-23(b): the *shape/mechanism* half survives (kick/ensemble flat, boost real,
   monotone auto-stopping curve, +2.3…+36.2 pp); the *absolute-dominance* half stays retracted, now proven with
   headroom present. This is a **new negative** (candidate N-registry entry: "ambiguity-headroom does not yield a
   CLU leaderboard win over the correct erasure oracle").
2. **New config/CLI surface (defaults preserve w23 exactly):** `block_rescale` (bool, True), `crowd_rho` (float,
   1.0) added to `ExperimentRetryComputeConfig`; query_type `block`, store_mode `crowded`, `regimes`/`levels_for`,
   `--ambiguity`/`--headroom` on `chlu exp-retry-compute`; 7th line `feedforward_nn_masked` (ML-optimal
   observed-dims-only NN) reported on all erasure cells. A regression test pins the shipped default to the w23 grid.
3. **Retract "w23 iid store ran at packing slack ≈1.08."** That number was a **unit artifact** — a per-element σ_q
   against a vector median-NN pinned the slack at `1/(3.1·clu_s_frac)` for every store. Corrected slack for w23 iid
   mask p=0.5 M=128 = **0.227** (i.e. w23 already ran *past* the packing bound). Any text quoting 1.08 as a
   measured store property should be fixed (mirrors the §7.19 "never quote 2.6" precedent).
4. **RUD-C spec amendment (for the paper appendix):** mandatory baseline #1 should be the **masked (observed-dims)
   NN oracle**, not the naïve full-vector NN — on erasure protocols the naïve NN is beatable but the oracle is the
   real ML floor and is trivially 1.000. Any "test-time compute wins" claim must be relative to the oracle.
