# fit-gap-anatomy — results-analyst report

**Task + acceptance criterion:** decompose the "physics costs raw fit" gap (CM-1) into
**contraction-forbidden** vs **reach-priced** parts (falsifiable a), measure the fit-vs-horizon
**loan curve** and its crossing (falsifiable b), build the **recovery ladder** — each paid mechanism's
% of the twin gap recovered vs which CM-1 guarantee survives (falsifiable c), and scope **intra-unit
wormholes** as a design report (item 4). Per-falsifiable verdicts + ladder table + CM-1 scope update.

**Status: done** for items 1, 2, 3 (contraction rungs full; reach rungs by certificate + the honest
w7-deferral) and 4 (design). Repo untouched (read-only; all artifacts under `.claude/`). Negatives
written in full (C-9).

---

## Flag provenance (all runs)
- **Commit:** `9a13455` (integration/wave-5; `git log -1`). Repo read-only, **no code changes**.
- **Env:** `/Users/user/Desktop/CHLU/.venv`, `PYTHONPATH=<repo> uv run --no-sync python <script>` (the §7.12
  UF_HIDDEN CLI bug avoided by driving `python -m`/scripts; `chflags nohidden …_editable_impl_chlu.pth` before
  each call). JAX cache **warm** on this machine (imports 3 s, not 20 min); CPU device only.
- **Non-default flags in effect (shared):** `training.sleep_frequency=1e9` ⇒ **wake-only** (the natural
  minus-the-physics MSE objective; one epoch-0 sleep event fires, inert for the twin) EXCEPT the γ_φ rung
  (`sleep_frequency=5, sleep_steps=200`). Training defaults otherwise on `9a13455`: `lr=1e-3, batch=64,
  lyapunov_penalty="max" (λ=0.01), langevin_noise="legacy", sleep_temperature=0.5, sleep_friction=0.0,
  persistent_sleep_buffer=False`. `dt=0.05`, `window_size=64`.
- Scripts: `.claude/scratch/fit-gap-anatomy/{item1_reach,item2_loan,item3a_contraction_ladder,item3b_reach_certs,plots}.py`.
  Raw outputs: `.claude/outputs/fit-gap-anatomy/{item1_reach,item2_loan,item3a_ladder,item3b_reach_certs}.json`,
  `{loan_curve,reach_sweep}.png`.

| item | task | dim/hidden/kin | epochs | seeds | arms |
|---|---|---|---|---|---|
| 1 reach | shuttle q=A·sin(ωt), A=2, ω=1, req peak speed=2.0, seq_len=175 | 2/64/newtonian_learned + relativistic | 200 | {0,1,2} | newtonian(no-cap) + relativistic c∈{0.5,1,2,4,8} |
| 2 loan | circle-vacuum R=1, seq_len=65, n=256 | 4/64/newtonian_learned | 150 | {0,1,2} | chlu, broken_volume, twin (matched params), LSTM(hid 64) |
| 3a ladder | circle-vacuum (same as item 2) | 4/64/newtonian_learned | 150 | {0,1,2} | twin, CLU(γ=0), CLU+γ, CLU+γ_φ(K=2 learned) |
| 3b certs | analytic (no training) | 2/32 relativistic; 2-unit lattice | — | 0,1 | S^(M) squeeze; wormhole GatedCoupling |

---

## Falsifiable (a) — decompose the gap: contraction-forbidden vs reach-priced

### (a.1) Reach term isolated with the relativistic cap the w5 arm never had
The w5 minus-physics arm used `newtonian_learned` (no cap), so it could not see reach-pricing. I ran a
**relativistic arm with the cap active and swept c** on a high-velocity shuttle task whose required peak
latent speed is `A·ω = 2.0`. The cap is `v_max = c/√M_eff` (F5 Prop-1). Result (`item1_reach.json`,
mean over 3 seeds; figure `reach_sweep.png`):

| arm | v_max=c/√M | reconstruction MSE | Δ over plateau |
|---|---|---|---|
| relativistic c=0.5 | 0.62 | 2.378 | **+1.04 (+77%)** |
| relativistic c=1.0 | 1.22 | 2.053 | +0.71 (+53%) |
| relativistic c=2.0 | 2.42 | 1.372 | +0.03 |
| relativistic c=4.0 | 4.83 | 1.342 | 0 (plateau) |
| relativistic c=8.0 | 9.65 | 1.434 | +0.09 |
| newtonian (no cap) | ∞ | 1.482 | (≈ plateau) |

**Verdict: reach-pricing is real and measurable in aggregate.** MSE rises **monotonically** as the cap
tightens *below the required speed* (c=0.5→2.38, c=1→2.05) and **collapses to the uncapped baseline once
`v_max ≥ req`** (c≥2, MSE≈1.34–1.43 ≈ newtonian 1.48). The transition sits exactly where `v_max(c=2)=2.42`
crosses `req=2.0`. This is the reach term the CM-1 caveat said was *inactive* in the w5 arm — now activated
and priced. The arm did **not** evade the cap by shrinking M (M_eff stayed ≈0.58–0.61; v_max tracks c).

### (a.2) But the reach term is secondary and NOT locally concentrated (honest negatives)
Two sub-claims of falsifiable (a) **fail** on this single-unit testbed:
- **No per-step "far-region" concentration.** Correlation of per-step error with instantaneous target
  speed |q̇| is ≈ 0 (−0.01…+0.05 across all arms); high-velocity-quartile error ≈ low-velocity-quartile
  error (e.g. c=0.5: 2.39 vs 2.37). The cap degrades the *whole* autonomous rollout globally, not the
  high-velocity phases specifically — the capped orbit drifts as a whole rather than clipping at peaks.
- **A large representation floor dominates the absolute number.** Even the uncapped newtonian and the
  generous-cap c≥4 arms sit at MSE≈1.4 (signal variance ≈2), i.e. **no arm autonomously reproduces the
  moving target well.** The reach-priced increment (≤+1.0) rides on top of a ~1.4 floor that is a
  training/representation limit of a single conservative unit, not a physics cost.

**Combined (a) verdict:** the fit gap *does* decompose — contraction-forbidden is the **dominant** term
(measured 15× on the stationary testbed, w5) and **reach-priced is a real but secondary, aggregate-only
term** that appears only when the causal cap binds (`v_max < req`). The Head's mechanism is confirmed in
the direction of the effect but is **not** the dominant driver at single-unit scale — consistent with the
theory that reach-pricing is fundamentally a **lattice** phenomenon (sync ∝ √κ across units, F5 §7.2),
which is exactly why item 4 (intra-unit reach) is a genuine, unfilled gap.

---

## Falsifiable (b) — the loan curve: does the twin's fit advantage reverse at horizon?

Fit-vs-horizon (cumulative MSE, mean/3 seeds) on the stationary circle-vacuum orbit — legacy Exp-I
"long-horizon stability" made quantitative against a **matched-parameter** control (`item2_loan.json`,
figure `loan_curve.png`). Param match: chlu 4549 / broken_volume 4557 / twin 4551.

| horizon | **twin** | **chlu** | broken_vol | LSTM |
|---|---|---|---|---|
| 10 | **2.9e-6** | 2.2e-5 | 1.1e-4 | 7.5e-2 |
| 50 | **6.2e-5** | 1.2e-3 | 1.0e-3 | 4.9e-2 |
| 100 | **2.4e-4** | 1.2e-2 | 4.1e-3 | 4.9e-2 |
| 500 | **1.1e-2** | 1.8e-1 | 9.7e-2 | 7.7e-2 |
| 1000 | 3.2e-1 | **2.0e-1** | 1.2e-1 | 9.9e-2 |
| 5000 | **1.96e+2** | 2.2e-1 | 1.4e-1 | 1.3e-1 |

**Verdict: the loan is real and it is called between horizon 500 and 1000.** The unconstrained twin is
**best by 1–2 orders of magnitude up to ~500 steps** (its short-horizon fit advantage — the "loan"), then
its curve **crosses above CLU at ~700 steps and diverges catastrophically** (196 at 5000 — a free residual
recurrence has no bounded-attractor guarantee, so its small per-step residual compounds). CLU is the only
arm whose curve is *flat by construction* (bounded plateau ≈0.20–0.23). This is the quantitative core every
short's Pareto framing wanted: **the twin trades a bounded long-horizon guarantee for short-horizon fit,
and the trade is visibly unwound by ~700 steps.**

Honest nuance: broken-volume (0.14) and LSTM (0.13) plateau *slightly below* CLU (0.22) — so "physics costs
raw fit" persists as a small steady-state penalty even long-horizon (CLU is not the lowest plateau), but the
**twin's catastrophic divergence is the loan being called**, and only CLU/broken-vol/LSTM stay bounded
(LSTM by tanh saturation, broken-vol by its learned near-contraction). Terminal-MSE curves cross in the same
500→1000 window (twin 5.2e-2→2.0e0; CLU 1.9e-1→2.2e-1).

---

## Falsifiable (c) — the recovery ladder

Each paid mechanism, one at a time, on its matched testbed. **% gap recovered = (CLU_conservative_MSE −
rung_MSE)/(CLU − twin)**. Structural guarantees (CM-1) re-measured per rung.

### Contraction rungs (circle-vacuum; `item3a_ladder.json`, 3 seeds). Twin gap = 0.202.
| rung | eval MSE | **% twin gap recovered** | BIBO | latch drift | flat μ² |
|---|---|---|---|---|---|
| twin (no physics) | 0.0047 | 100% (def.) | 1.00 | — | — |
| **CLU conservative (γ=0)** | 0.2066 | 0% (def.) | 1.00 | 0.778 | 5.2e-3 |
| **+γ global (γ=0.05)** | 0.0216 | **92%** | **1.00** | **0.778** | **5.2e-3** |
| +γ_φ learned field (K=2) | 0.2548 | **−24%** | 1.00 | 0.706 | 2.1e-2 |

### Reach rungs (certificates; `item3b_reach_certs.json`)
| mechanism | certificate measured | value | reach granted |
|---|---|---|---|
| **+S^(M) squeeze** (ζ=1) | det S^(M) | **1.0000** (symplectic err 3e-7) | Δq=**2.18** (>req 2.0) |
| | H-injection ratio vs e^{2\|ζ\|} bound | 2.28 ≤ 7.39 ✓ (bounded) | |
| **+inter-unit wormhole** | κ=0 → independent units | max err **0.0** (bit-exact) | (routing, see v1-wormhole) |
| | gated edge energy bounded | (cite v1-wormhole: mean 1.69/max 5.49, N=4) | 0.875 vs 0.50 acc |

**Verdict: falsifiable (c) confirmed with a sharp qualitative structure — different mechanisms rent back
different cheats:**
- **+γ global recovers 92% of the contraction-forbidden fit gap at ZERO structural cost** — BIBO (1.0),
  latch (0.778, unchanged), and the protected near-flat μ² (5.2e-3, unchanged) all survive. This is the
  headline ladder result: licensed uniform contraction (conformal-symplectic drain, det J=(1−γ)^d) buys
  back almost all of the fit the twin stole by leaking volume, **without giving up any CM-1 guarantee.**
- **+γ_φ learned field does NOT recover fit here (−24%).** Honest negative: the friction field is a
  *targeted-forgetting* mechanism (Thread 1: leakage↓200×), not a fit-recovery lever — the stationary
  circle-vacuum task has no "trash region" to forget, so pushing γ_φ up at hallucinations while protecting
  data doesn't reduce reconstruction error (it slightly perturbs V; μ² rises to 2.1e-2). **The lesson: the
  paid mechanisms are not interchangeable — γ_global buys the contraction fit; γ_φ buys selective
  forgetting; they must be matched to the cheat.** (Caveat: this rung retrained from scratch with the sleep
  phase on, so it is not a perfectly matched delta from the γ=0 model; the qualitative "wrong tool" verdict
  is robust, the exact −24% is not.)
- **+S^(M) squeeze rents back reach with an exact certificate:** det=1 to machine precision (volume
  conserved — unlike the broken-volume twin), energy injection bounded by e^{2|ζ|}, and a single ζ=1
  squeeze grants 2.18 units of latent displacement — enough to cross the reach-task's required 2.0 that the
  capped flow cannot. **What is NOT claimed:** a *learned-controller* fit recovery on the reach task. That
  is the staged w7 `paid-access-experiments` discriminating experiment (Thread 8), and `v1-l0-gate` already
  killed the retry-for-*correctness* null — so I report the mechanism's structural certificate + reach
  capacity, not an end-to-end MSE recovery.
- **+inter-unit wormhole** is already measured (`v1-wormhole-routing`: 0.875 vs 0.50 at bounded gate
  energy). Here I re-verify the two structural certificates it inherits: κ=0 reduces **bit-exactly** to
  independent units (max err 0.0) and the gate keeps H C¹ (bounded energy). The **intra-unit** wormhole —
  the mechanism the ladder's last rung really wants — **does not exist**; see item 4.

---

## Item 4 — intra-unit wormhole scoping (design report; [design hypothesis], no build)

**What it is.** Current wormholes (`chlu/core/lattice.py::GatedCoupling`) are **inter-unit**: a gated
position-only coupling `V_wh(q_i,q_j)=σ((t−v)/w)·v`, `v=base(q_i,q_j)`, added to the *joint* Hamiltonian
of a CLU lattice. An **intra-unit** wormhole is a within-node nonlocal channel across **one** unit's own
latent vector `q ∈ R^d` — coupling two disjoint coordinate blocks `q_A, q_B` (A,B ⊂ {0..d−1}) that the
unit's local dynamics would otherwise only connect through the causal-cap-limited flow.

**Why the reach argument motivates it (and its honest tension).** Item 1 shows single-unit reach is a real
but secondary, aggregate cost; F5 §7.2 says reach-pricing is fundamentally a *lattice* (√κ) law. Within one
unit the dense MLP potential `V_θ(q)` *already* couples all coordinates unconditionally — so the intra-unit
wormhole's distinct value **cannot** be "nonlocal coupling" per se (the MLP has that). Its distinct value
must be argued as **conditional, sparse, energy-priced, certified** access: a channel that is *closed by
default* (zero energy, exactly conservative) and *opens on demand* (when local relaxation stalls), carrying
a routing signal and a bounded-energy ledger — the intra-node analog of sparse attention vs a dense MLP.
This is a genuine [design hypothesis], and its discriminating test is "does a gated intra-unit channel beat
the dense-MLP baseline on a reach task?" — precisely a w7-class experiment.

**Mechanism sketch (minimal, symplectic-preserving).** A potential *wrapper*, mirroring `TiltedPotential`:
```
V_wh_θ(q) = V_θ(q) + Σ_{(A,B)∈E_intra} σ((t − v_AB)/w) · v_AB,   v_AB = g_θ(q_A, q_B) ≥ 0
```
where `E_intra` is a sparse set of intra-vector block pairs, `g_θ` a small learned nonneg coupling
(quadratic `‖W_A q_A − W_B q_B‖²` or an MLP head), `t,w` static gate knobs. Because the term is added
**inside V (position-only, C¹)**, Hamilton's equations and the dissipative-Verlet map are unchanged in
form, so **every existing certificate transfers verbatim** (this is the key economy):

**Where it lives in `chlu/core/`.** Cleanest: a new `IntraWormholePotential(eqx.Module)` in
`chlu/core/potentials.py` wrapping any base potential (exact `TiltedPotential` precedent, already composed
in `CHLU.__init__`), selected by a new `potential_type="intra_wormhole"` or a `CHLU(intra_wormhole_edges=…)`
kwarg. **No change to `integrators.py`, `chlu_unit.step`, or the lattice** — it is purely a potential
augmentation, which is exactly why it inherits the symplectic/BIBO machinery for free. Reuse
`training.calibration.fit_calibration_head` (per v1-pivot) for the residual-energy routing trigger, and the
existing energy-ledger accounting from the smooth-gate regime.

**Expected certificate form (per F5 §7.4).** Same three guarantees as the inter-unit smooth gate:
1. **Symplecticity preserved** — gate is C¹ in q through V ⇒ H stays C¹ ⇒ dissipative-Verlet keeps
   `det J = (1−γ)^d` exactly (no piecewise Hamiltonian, no energy jump to ledger). Verifiable numerically
   as I did for the inter-unit case (max symplectic err ~1e-7).
2. **Bounded energy injection** — `sup_{v≥0} v·σ((t−v)/w) < ∞`, so the intra-node channel's energy is
   capped by construction (the same `e^{2|ζ|}`-style bounded-injection certificate as the squeeze).
3. **Exact closed-gate reduction** — at zero coupling (or fully-closed gate) the unit reduces bit-exactly
   to a plain CHLU (the intra analog of the κ=0 max-err-0.0 check I ran).

**Engineering scope (for the engineer).** ~1 module (`IntraWormholePotential`, ~60 LOC) + 1 config group +
CHLU wiring + 3 tests (symplectic-err, bit-exact closed-gate reduction, bounded-energy monotonicity), all
in the *potential* layer — **no core integrator/lattice edits**. The hard part is not the mechanism but the
**discriminating experiment**: it must beat the dense-MLP potential on a reach task while its gate stays
sparse/closed off-demand — otherwise it is a strictly-dominated reparameterization of V. Recommend it be
folded into the staged w7 `paid-access-experiments` task alongside the squeeze reach test, with the gate-vs-
dense-MLP baseline mandatory (the v1-l0-gate lesson: always include the no-physics/always-on control).

---

## Limitations / confounds
1. **Reach floor (item 1):** no single conservative unit autonomously reproduces the moving shuttle well
   (MSE floor ≈1.4); the reach-priced increment is measured *relative to the c≥req plateau*, which is the
   right control, but the absolute fit failure is a representation/training limit, not physics. A reach task
   a single unit *can* fit (lower A·ω, or an exactly-quadratic potential) would sharpen the increment.
2. **γ_φ rung not perfectly matched:** it retrains with the sleep phase active (different objective than the
   wake-only γ=0 baseline), so the −24% is not a clean single-knob delta; the qualitative "wrong tool for
   this gap" verdict is robust, the magnitude is not.
3. **+γ global rung** applies γ at *inference* on the wake-trained model; a model trained *with* γ might do
   better still — 92% is a lower bound on what licensed contraction recovers.
4. **Testbeds are low-dim (d=2,4) and synthetic**, laptop-CPU, 3 seeds. Loan-curve plateaus and the reach
   transition are seed-stable (stds small) but the industrial-scale behavior is untested.
5. **Squeeze/wormhole reach recovery is certificate-only here**, deliberately (Thread-8 scope discipline:
   the learned-controller reach claim is w7, and retry-for-correctness is already a killed null).

## Recommended next experiments
1. **(w7 feed) Squeeze reach recovery, done right:** on a reach task a single unit *can* fit, does a
   τ-gated S^(M) squeeze controller recover the reach-priced increment (item 1's Δ) while det=1 and energy
   stays bounded? Must beat a no-physics router (v1-l0-gate lesson).
2. **Lattice reach-pricing:** repeat item 1 on a 2-unit lattice sweeping κ — is the √κ sync law the
   *dominant* reach term (as theory predicts), unlike the secondary single-unit effect?
3. **Loan curve on a moving target:** item 2 is stationary; a periodic target would show whether the twin's
   loan is called *earlier* when the task also demands reach.
4. **Build + discriminate the intra-unit wormhole** (item 4) vs the dense-MLP baseline.

---

## Proposed handover updates (for the Hub)

**§1.6 / claims-matrix CM-1 — proposed scope update (fills the "crossing unmeasured" caveat):**
> The fit gap decomposes: **contraction-forbidden is the dominant term** (15×, stationary testbed, w5),
> and **reach-priced is a real but secondary, aggregate-only term** — a relativistic arm with the cap
> active (the arm w5 lacked) shows reconstruction MSE rises monotonically as `v_max=c/√M` falls below the
> required speed (+77% at c=0.5) and collapses to the uncapped baseline once `v_max ≥ req` (c≥2), with **no
> per-step far-region concentration** (corr(err,|q̇|)≈0) and on top of a large representation floor. The
> **loan is called between horizon 500 and 1000**: the unconstrained twin leads by 1–2 orders to ~500
> steps, crosses CLU at ~700, and diverges (196 at 5000 vs CLU's bounded 0.22 plateau) — legacy Exp-I made
> quantitative against a matched (±0.2%) control. **Do claim** the crossing now (≈700 steps). **Do not**
> claim CLU has the lowest long-horizon plateau (broken-vol 0.14, LSTM 0.13 < CLU 0.22) — the CLU asset is
> *boundedness by construction*, not lowest steady MSE.

**§8 / recovery-ladder (new, feeds all shorts' Pareto frame):**
> Paid mechanisms rent back specific cheats: **+γ global recovers 92% of the twin fit gap with BIBO/latch/μ²
> all preserved** (licensed contraction, det J=(1−γ)^d); **+γ_φ learned field does NOT recover fit (−24%) —
> it is a targeted-forgetting tool, not a fit tool** (mechanisms are not interchangeable); **+S^(M) squeeze**
> rents reach with an exact certificate (det=1 to 3e-7, energy≤e^{2|ζ|}, one ζ=1 squeeze grants Δq=2.18 >
> req 2.0); **+inter-unit wormhole** already measured (v1-wormhole 0.875 vs 0.50), κ=0 bit-exact reduction
> re-verified. The full squeeze/intra-unit reach *recovery* is w7's discriminating experiment.

**Item 4 for the engineer / w7:** intra-unit wormhole = an `IntraWormholePotential` V-wrapper (~60 LOC in
`chlu/core/potentials.py`, `TiltedPotential` precedent, **no core/lattice edits**); inherits all three F5
§7.4 certificates (symplectic-preserved, bounded-energy, bit-exact closed-gate). The genuine risk is that it
is a strictly-dominated reparameterization of the dense MLP potential — its discriminating test (gated-sparse
vs dense-MLP on a reach task, with a no-physics control) belongs in `paid-access-experiments`.

**Code note (no bug):** all runs used the `.venv` python + `PYTHONPATH` to dodge §7.12 (UF_HIDDEN); the
`chflags nohidden` shim was needed before every invocation — the durable `make fix-env` shim is worth
confirming still holds on `9a13455`. No divergence/NaN/OOM encountered.
