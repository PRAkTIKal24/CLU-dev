# PREREG — `full-clu-harness` (C2W1)

**Written 2026-07-29, BEFORE any measured harness run.** Author: `experiment-engineer`.
Base `main @ 082d095`; branch `agent/experiment-engineer/full-clu-harness`; worktree `../CHLU-fullclu`.
Env: main venv reused (`/Users/user/Desktop/CHLU/.venv`), **JAX 0.9.0** (no worktree `uv sync` — w6 lesson).

At the time of writing, the only code that exists on the branch is the **API-FREEZE commit**
(`e776afe`, signatures + `NotImplementedError`). Nothing has been measured.

---

## 0. Dial declaration (echoed)
- **Dial:** none — **instrument**. No performance claim, no leaderboard. Acceptance = *"does not collapse."*
- **Laundering control:** built, not claimed — settle-deleted launder + same-keys null + blank/empty store,
  all harness-native, plus the doctrine's second (best-shared-metric) launder.
- **Falsifies the acceptance criterion:** a monitor trips and cannot be cleared by returning the responsible
  lever to its known productive band.
- **⛔ Hard falsifier:** if the only non-tripping configuration is the degenerate one (explicit per-item
  arrays / engineered separation / settled-point-only read), §8.2 has fired and that is the headline.
- **Does NOT falsify:** ≤0 dividend · slow wall-clock · monitors #7/#13 not being runtime trips ·
  a trip on #9 at large payload excursion or on #2 (declared **known-uncleanable-by-any-verb**, doctrine I-14).

## 1. Lever activation order (registered)

Stages are cumulative; each frees exactly one lever **inside the running full system** (charter §3.1).

| stage | lever freed | starting band (provenance) |
|---|---|---|
| S0 | *baseline*: learned `V_θ` store, derived addressing, masked write, two-phase read, settled-point ψ | d=4, K=8, atoms/item=8, width 0.15, γ=0.2, 400+200 steps @ dt 0.05 (doctrine row 1: γ≈0.2 optimum; w26 d=4) |
| S1 | **per-item lifetimes** (physical amplitude decay of the item's own atoms) | leak ∈ (0, 0.05], amp_floor 0.05 (w22/w25) |
| S2 | **admission gate + capacity pressure** (budget < offered, eviction) | `d_safe = 2 s_max + κ′σ_q` (doctrine R5/I-13), NOT `4.4 s` |
| S3 | **deletion demand** (≥1 explicit delete mid-stream) | canonical/set-function placement (PGCP) |
| S4 | **basin interaction** (address ball shrunk / K raised until wells overlap) | intervention §3.3: separability is the thing we are removing |
| S5 | **confidence-gated retry** (compute dial; confidence = settle residual, never energy — N97) | τ=0.5, ≤2 rounds |
| S6 | **trajectory read** (ψ sees the strided buffer, not only `q*`) | reported with its 3-way launder (doctrine I-2) |

## 2. Predicted trip-set, in the order I expect it (the scoreable part)

Scoring: ✅ = fires as predicted at the predicted stage; ⛔ = does not; ◐ = fires at a different stage.

| # | monitor | prediction | reasoning |
|---|---|---|---|
| 8 | certificates (N2 `sep/σ_q ≥ 5.15`) | **TRIPS FIRST, at S0**, and is **cleared by `expand`** (ball radius ↑) | at R=1.0, d=4, K=8 the farthest-point spacing is ≈ 2R·K^(−1/d) = 1.19, and 5.15·σ_q = 1.236 > 1.19 ⇒ the certificate cannot hold at the default radius. This is a *derived* prediction, not a guess |
| 11 | reach | **TRIPS at S0 for the 1–2 largest-\|a\| items** | w26 measured `a_U = 1.06` at d=4, K=32 on the shipped `V`; payloads on a ±1 grid put the extreme items within ~6 % of the threshold |
| 2 | settle→arg-min (`ρ_ex < 0.10`) | **TRIPS at S0 and stays tripped through S3**; may clear at S4 | Prop D2a: near-homogeneous wells ⇒ `D ≈ 0`. Basin interaction (S4) is the only registered lever that can create `B_i \ Vor_i`. **Declared known-uncleanable-by-verb** |
| 9 | lifetimes (`Δ_ret > 0.10`) | **TRIPS at S1** | w25 r=−0.85; doctrine measured the knee at excursion ≈ 0.45·R. Fix is C1W27's gated stiffness, which C2 must not build ⇒ reported as scope |
| 3 | vacuous gate | **TRIPS at S2 on the fire-rate leg first** (f=0 or f=1 on the first calibration), cleared by moving `d_safe` inside `[2s_max+κ′σ_q, sep]` | N74 is exactly this failure |
| 12 | starvation | **TRIPS at S2/S3** (fairness `min D/max D < 0.5` under eviction+rewrite) | w26: masked/sequential writes give each item atoms/K |
| 5 | addressing (`acq < 0.90`) | **TRIPS at S4** (basin interaction), cleared by `anneal` | N109: the annealed read *is* address acquisition |
| 4 | blank | **PASSES S0–S5, TRIPS at S6** (raw-trajectory ψ) | the trajectory contains `q0 = φ(x)` (doctrine I-2 / N68 at 100 % strength). Cleared by the store-relative form |
| 1 | overdamping | **NEVER TRIPS** (γ=0.2, N=400 is inside the measured band) | doctrine S2b table |
| 6 | objective divergence | **NEVER TRIPS** at these K | w25/w26 saw it at K≫ |
| 10 | dead axis | **TRIPS at startup on the first full run** (≥1 declared knob unread), cleared before the reported run | my own config has stage flags that are only read in later stages |
| 7 | mass gauge | **NOT a runtime trip; PASSES in `pytest`** (Newtonian exact) | Prop F1; relativistic is a scope, not a pass |
| 13 | maturity | **NEVER TRIPS** (provenance field) | N94 |
| M14 | guard liveness | **TRIPS at S0/S1** (no guard has fired yet), cleared once the canary stream runs | by construction |

**Aggregate prediction: 9 of 14 fire at least once; 5 never fire and are therefore reported as UNTESTED, not green.**

## 3. Predicted v0 dividend

**Sign: NEGATIVE.** Point prediction on the harness's own value-decode metric, at S0, K=8, d=4:

`dividend = full − settle_deleted_launder ∈ [−0.30, 0.00]`, **point estimate −0.10**.

Derivation, not vibes: (a) Prop D2 — with `D ≈ 0` the dividend is bounded above by ~0 and below by the
read error; (b) w26 measured the settle at ≈ −1 pp on *clean classification* against the same-keys launder,
and this harness's read is harder (the payload is recovered from a learned `V_θ`, not from a designed
payload spring), so the loss should be larger than 1 pp; (c) the launder is given the exact stored payload
by construction. **A positive dividend at v0 would be suspicious** and would go through all three controls
plus a seed re-run before being written down.

Also registered: **blank ≈ chance** (value read is leak-immune) and **same-keys null ≈ chance** on the
value metric; both would *falsify the instrument* if they scored high.

## 4. Compute order (declare, do not silently drop)

1. `pytest tests/test_monitors.py tests/test_clu_controller.py tests/test_clu_system.py` (cheap, no training).
2. `--quick` smoke of `exp-clu-system` (K=4, 60 write steps, 1 seed) — plumbing only, **not a reported number**.
3. **The reported run**: staged S0→S6, K=8, d=4, 300 write steps/item, **seed 0**.
4. Dividend + all three controls at S0 and at the last non-tripping stage.
5. Knob-liveness sweep (monitor #10 tier (b)) over the declared dials.
6. *If time allows*: seeds 1,2 for the dividend only (multi-seed is required before any **paper** number; this
   is an instrument report, so a single seed is acceptable **and must be labelled as such**).

Anything not reached is reported as **NOT RUN**, never as a null. Worktree count: **1** (C2's claim);
C1W27 holds `../CHLU-r2dsweep`. No background sweeps without telling the Hub.

## 5. What I will report even if it is ugly

The full trip table including cleared trips (that table *is* the ablation reviewers will demand), the
stage at which each trip fired, the lever it was attributed to, and — if the only clean configuration
turns out to be the degenerate one — that finding as the headline.
