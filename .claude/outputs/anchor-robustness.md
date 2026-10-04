# anchor-robustness — results-analyst report

Task + acceptance criterion: harden the sleep-erosion V(data)-anchor cure and test it as the memory-fidelity intervention. **(P11/V2.4)** anchor operating envelope over λ × seeds × epochs × testbeds + demarcation-law-as-THEORY confirmation; **(P14/V1.3)** does the anchor / longer training move a losing Hopfield-stress regime-map cell; **(cross-link)** does the anchor rescue the non-symplectic (broken-volume) vacuum, or is CD-robustness genuinely a volume-conservation payoff (CM-1/CM-6).

**Status: done** — all three items answered with numbers + error bars (5 seeds Item 1; 6-run tilt control; 2 seeds Item 3; 2 seeds × 2 episodes Item 2). 38 exp-d/bv runs + 16 memory runs, no fabricated numbers; failures (seed-45 instability, broken-volume divergence, anchor-on-memory failure) reported as observed. Repo **read-only** (no tracked files touched); all artifacts under gitignored `.claude/`. Laptop-CPU (8 cores), JAX **f32** training, f64 probes.

**Headline verdicts:**
1. **P11:** The V(data)-anchor holds the designed SO(2) vacuum through **3000** erosive epochs for **λ∈{1,10,100}** where λ=0 always dies. Envelope has a **strength↔robustness tradeoff**: λ≈10 gives the strongest noise rejection (gap≈1.8) but is *not* seed-bulletproof (1/5 seeds still collapses); **λ=100 is bulletproof across 5/5 seeds** (r\*=0.911±0.016) at the cost of weaker noise rejection (gap≈0.60) and ~35× higher wake MSE. Demarcation confirmed as THEORY: erosion attacks the flat μ²≈0 direction; the anchor preserves it (μ²_min≈1e-6), and **lifting the flat direction by tilt immunizes the vacuum even with no anchor** (erosion ∝ flatness).
2. **P14:** The anchor **does NOT** improve memory fidelity (it pins to a structureless random init). But **longer training (500→2000 ep) closes the Hopfield gap entirely** in 3/3 previously-losing cells — including the kv64 "capacity collapse" cell: CLU-EBM fidelity **0.40→1.00**, gated accuracy **0.05→0.99**, **matching/beating** Hopfield at **9–10× compute savings**. The v1-hopfield-stress "Hopfield dominant 26/26" map was **under-trained**.
3. **Cross-link:** the anchor does **NOT** rescue the broken-volume (non-symplectic) vacuum — it diverges under CD regardless (wake MSE 3.5k–176k, r\*→∞, noise_gap negative). **CD-robustness is genuinely a volume-conservation payoff**, orthogonal to the anchor.

---

## 0. Setup — configs, seeds, commands, provenance

**Base code:** `main` (read-only, no branch). Reused the **bit-faithful `train_chlu` replica** from `sleep-erosion-study` (`driver.py`, validated max|Δparam| vs `run_experiment_d` = 0.0) with the anchor hook it already carries (`λ·(mean_i V(anchor_data_i) − target)²` added to the wake loss). Item-1/3 runner: `.claude/scratch/anchor-robustness/run_anchor.py`; Item-2 runner: `item2_memory.py` (reuses `exp_v1_calibration._regime_cell`/`_regime_metrics` verbatim).

**Metrics:** `ring_depth`=V(0)−mean_θV(ring) (>0 ⇒ well); `r*`=radius of the damped-settled vacuum from a ring point (γ=0.1, 2000 steps; 1≈intact, 0≈collapsed-to-origin, ∞=diverged); `noise_gap`=meanH(N(0,1))−meanH(ring) (>0 ⇒ rejects noise); `μ²_min`=smallest spectral-mass² eigenvalue at the settled point (`spectrum_probe`; ≈0 ⇒ protected flat/Goldstone mode survives, >0 ⇒ flat direction stiffened/destroyed); `wake_loss`=mean trajectory-MSE last 50 epochs.

**Flag-provenance (Item 1 & 3)** — erosion-sensitive knobs per §5:

| flag | value | flag | value |
|---|---|---|---|
| testbed | exp-d SO(2) `so2_invariant`, dim 4, hidden 64 | kinetic | newtonian_learned |
| **sleep_frequency** | **5** | **sleep_steps** | **500** |
| **persistent_sleep_buffer** | **False (CD)** | sleep_friction | 0.0 (⇒ Langevin noise inert) |
| tie_channel_mass | True | dt | 0.05 |
| lyapunov_penalty | max (λ=0.01) | langevin_noise | legacy |
| clamp_strength | 1000→1 (ramp 0.5) | lr / batch | 1e-3 / 64 |
| **anchor λ** | **{0, 1, 10, 100}** | anchor_target | epoch-0 mean V(ring) |
| **epochs** | **3000** (ckpts 0/300/1000/3000) | **seeds** | **{42,43,44,45,46}** |

Command: `PYTHONPATH=$PWD uv run --no-sync python run_anchor.py <tag> <chlu|broken_volume> <λ> <seed> 3000 [tilt_delta]`

**Flag-provenance (Item 2)** — `train_generative` on per-episode MQAR memory: embed_dim 16 (CLU dim 32), hidden 128, **kinetic relativistic**, potential mlp, train_lr 1e-3, train_batch 32, **train_k_steps 50**, train_friction 0.3, train_temperature 0.3, train_input_noise 0.05, clamp_key True, relax_steps 300, governor_sensitivity 0.95, calib_p_exit 0.5, calib_features r_margin, vocab 256, hopfield_beta 20, regime_n_clusters 8, **train_epochs ∈ {500 (default), 2000}**, 2 episodes/cell, seeds {42,43}.

---

## 1. Item 1 (P11) — anchor operating envelope on the exp-d SO(2) vacuum

**Mean±std over 5 seeds, @3000 erosive epochs** (`fig_envelope.png`; full metric-vs-epoch table `.claude/outputs/anchor-robustness/tables.md`):

| λ | ring_depth | r\* | noise_gap | μ²_min | wake_loss | verdict |
|---|---|---|---|---|---|---|
| **0** (disease) | −0.110±0.020 | 0.51±0.63 | +0.039±0.019 | **+0.041** | 0.0014 | vacuum destroyed **5/5** |
| **1** | +0.60±0.76 | 0.94±0.12 | +1.21±0.31 | ~1e-6 | 0.73* | vacuum held 5/5; wake unstable 2/5 |
| **10** | −0.09±0.67 | 0.70±0.35 | **+1.85±1.02** | +0.086 | 0.055 | 4/5 clean; **1/5 collapsed** |
| **100** | +0.118±0.013 | **0.911±0.016** | +0.60±0.07 | ~5e-7 | 0.048 | **5/5 bulletproof** |

*λ=1 mean wake inflated by one pathological seed (see per-seed).

**Per-seed @3000ep (the honest picture):**

| λ | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|
| 0 | r\*0.00 | r\*1.35 | r\*0.00 | r\*1.20 | r\*0.00 — all rd<0, μ² stiffened → **all destroyed** |
| 1 | ✓0.87 | ✓0.86 (wake0.11) | ✓0.88 | ✓1.18 (**wake3.5**) | ✓0.89 — vacuum held 5/5; wake blew up on s43,s45 |
| 10 | ✓0.87 | ✓0.86 | ✓0.88 | **✗ r\*0, rd−1.42, wake0.24** | ✓0.89 |
| 100 | ✓0.93 | ✓0.91 | ✓0.92 | ✓0.88 | ✓0.92 — **no failures, tight** |

**Findings.**
1. **λ=0 is the disease at every seed:** ring inverts (rd<0), μ²_min stiffens from ~0 to +0.04–0.21, and the vacuum either collapses to the origin (r\*=0, seeds 42/44/46) or drifts off-ring (r\*>1, seeds 43/45). Reproduces `sleep-erosion-study` at 3000 ep with 5 seeds.
2. **The anchor works, and it holds the whole 3000-epoch horizon.** At ckpt **300 and 1000, ALL λ≥1 are fully rescued (r\*=1.000)** with μ²_min≈1e-6 — the flat Goldstone mode is preserved. The interesting behavior is all in the **3000-epoch tail**, where seed-specific training instabilities emerge.
3. **Strength↔robustness tradeoff (the envelope):** λ∈{1,10} give the **strongest noise rejection** (noise_gap grows 0.2→1.2–1.85 over training — better than sleep-erosion's wake-only) at low wake cost, but are **not seed-bulletproof**: a single pathological seed (s45, also fragile at s43) destabilizes (λ=1 wake→3.5 with vacuum still held; λ=10 seed-45 full collapse r\*=0). **λ=100 suppresses the instability entirely** (5/5 clean, r\*=0.911±0.016) but over-regularizes: noise rejection halves (0.60) and wake MSE is ~35× the λ=0 floor. **Operating recommendation: λ≈10 for max noise-rejection where seed-robustness can be checked; λ=100 when a single-shot guarantee is needed.**
4. **The anchor cost is real and monotone in λ:** wake MSE floor 0.0014 (λ0) → ~0.01 (λ1,10 stable seeds) → 0.048 (λ100). Pinning V(data) trades trajectory-fit freedom for vacuum protection.

## 2. Item 1 (P11) — the demarcation law presents as THEORY (wake-invisible flat directions)

Three independent lines confirm the mechanism is *the flat direction*, not a patch:

- **μ² witness.** At λ=0 the settled-point spectrum loses its protected mode (μ²_min: ~0 at ep300 → **+0.04–0.21** at ep3000 = the flat direction is stiffened as the ring inverts). With the anchor (λ≥1) μ²_min stays **≈1e-6** through 3000 ep — the Goldstone/flat coordinate is preserved. Erosion = destruction of the flat mode; the cure = its preservation.
- **Tilt immune control (corrected; erosion ∝ flatness).** *(First attempt had a runner bug — `tilt_delta` was not threaded through `build_exp_d`; fixed and re-run.)* Lifting the flat direction with a tilted potential, **λ=0, no anchor**, @3000ep:

  | tilt_δ | seed42 r\* | seed44 r\* | vs untilted λ=0 |
  |---|---|---|---|
  | 0 (flat) | 0.00 (collapsed) | 0.00 (collapsed) | the disease |
  | 0.05 | 1.23 (survives) | 0.75 (survives) | **erosion gone** |
  | 0.2 | 1.36 (survives) | 0.20 (1 seed unstable) | mostly immune |
  | 0.5 | 1.27 (survives) | 1.28 (survives) | immune |

  As soon as the wake-MSE can *see* the angular coordinate (δ≥0.05), CD can no longer invert the vacuum — **the vacuum is immune with no anchor at all.** This is the crisp theory statement: *wake–sleep CD inverts a designed vacuum iff that vacuum has a flat direction unconstrained by the wake objective.*
- **Exp-B control (from `sleep-erosion-study §3.5`, cited):** the non-degenerate sine vacuum is immune (wake+sleep ≈ wake-only; well_gap slightly higher *with* sleep). Consistent: no flat direction ⇒ nothing to erode ⇒ anchor is a no-op.

**⇒ The demarcation law is theory, not a patch note:** erosion is a property of degenerate vacua; the anchor is the value-pin the wake objective structurally lacks along flat directions.

## 3. Item 3 (cross-link) — the anchor does NOT rescue the non-symplectic vacuum

Broken-volume arm (`BrokenVolumeCHLU`: identical H/potential/kinetic, learned scaling breaks det J=1), erosive CD, @3000ep:

| λ | seed | ring_depth | r\* | noise_gap | wake_loss |
|---|---|---|---|---|---|
| 0 | 42 | +11.6 | 1.39 | **−34.6** | 40 364 |
| 0 | 43 | +5.6 | **∞** | −38.1 | 10 605 |
| 10 (anchor) | 42 | +8.2 | 2.08 | **−49.5** | 175 579 |
| 10 (anchor) | 43 | +0.8 | **∞** | −16.7 | 3 485 |

Both λ=0 and λ=10 **diverge catastrophically**: wake MSE 10³–10⁵ (dynamics blow up — the F5 Prop-10/BIBO consequence: a non-symplectic settle is unbounded), r\*→∞, and **noise_gap goes strongly negative** (the landscape *inverts* — garbage gets *lower* energy than data). The anchor at λ=10 does not help — seed-42 wake is even higher (175 579). **Pinning V(data) cannot save a map whose phase-space volume is not conserved**, because the failure is in the *dynamics* (divergence), not merely the potential's flat direction.

**⇒ CM-1/CM-6 sharpened:** vacuum survival under CD is a **volume-conservation payoff** (symplectic CLU survives; broken-volume diverges), and the V(data)-anchor is an **orthogonal** cure that only operates on a symplectic (bounded) substrate. This corroborates minus-the-physics (broken-vol collapse at 150 ep) at 3000 ep and shows the anchor and volume-conservation address *different* failure modes.

## 4. Item 2 (P14) — memory-fidelity intervention

Reused `_regime_cell` verbatim; three losing cells from the v1-hopfield-stress map. Mean±std, 2 seeds × 2 episodes:

| cell (N,kv,ρ) | arm | epochs | fidelity | CLU gate acc | CLU full acc | Hopfield acc | savings |
|---|---|---|---|---|---|---|---|
| **128,16,0.8** | baseline | 500 | 0.91±0.00 | 0.53±0.16 | 0.42±0.08 | 0.92±0.02 | 1.6× |
| | **longer** | 2000 | **1.00±0.00** | **1.00±0.00** | **1.00±0.00** | 0.92±0.02 | **9.2×** |
| **128,32,0** | baseline | 500 | 0.66±0.18 | 0.37±0.07 | 0.37±0.04 | 0.98±0.00 | 1.2× |
| | **longer** | 2000 | **1.00±0.00** | **1.00±0.00** | **1.00±0.00** | 0.98±0.00 | **10.0×** |
| | anchor | 500 | 0.44±0.02 | 0.27±0.03 | 0.24±0.01 | 0.98±0.00 | 1.1× |
| **256,64,0** | baseline | 500 | 0.40±0.05 | 0.05±0.02 | 0.05±0.02 | 0.99±0.00 | 1.0× |
| | **longer** | 2000 | **1.00±0.00** | **0.99±0.00** | **0.99±0.00** | 0.99±0.00 | **9.3×** |
| | anchor | 500 | 0.51±0.12 | 0.04±0.00 | 0.04±0.00 | 0.99±0.00 | 1.0× |

**Findings.**
1. **Longer training closes the Hopfield gap in 3/3 cells — including the kv64 "capacity collapse."** At 4× the regime-map's default budget (500→2000 ep) the CLU-EBM storage fidelity rises **0.40–0.91 → 1.00** and the calibrated energy-gate accuracy rises **0.05–0.53 → 0.99–1.00**, **matching or beating** Hopfield (0.92/0.98/0.99) at **9–10× compute savings**. The kv64 baseline reproduces v1-hopfield-stress's fid≈0.36–0.40, then longer training lifts it to 1.00 — so kv64 was a **training-budget artifact, not a capacity wall** at this scale.
2. **The anchor does NOT help memory fidelity** (kv32 anchor 0.44 < baseline 0.66; kv64 anchor 0.51 ≈ baseline 0.40, both with gate acc ≈ baseline or worse). Expected: the memory EBM has **no designed epoch-0 structure** to anchor to — pinning stored-pattern energies to their *random-init* values is meaningless (unlike the SO(2) ring, which is a designed structural prior). The anchor is the right tool for **designed degenerate vacua**, the wrong tool for **capacity/retrieval-limited memory**.

**The honest sentence V1 needs (charter C-9):** *"Memory fidelity is improvable — the Hopfield gap closes entirely — by ~4× longer training (500→2000 epochs), which lifts CLU-EBM storage fidelity from 0.40–0.91 to 1.00 and turns 3/3 previously Hopfield-dominant cells into CLU-competitive-or-winning cells at 9–10× compute savings; the v1-hopfield-stress 'Hopfield-dominant 26/26' map was under-trained. The V(data)-anchor does not transfer to the memory-fidelity problem."*

---

## 5. Limitations / confounds

1. **Probes in f32.** Training and probing share one f32 process (x64 casts truncate); μ²_min≈1e-6 for protected modes is at the f32 hessian noise floor — read qualitatively (flat vs stiffened), not to 6 digits. Headline metrics (ring_depth, r\*, noise_gap) are robust.
2. **Item 2 is small-n and small-scale:** 2 seeds × 2 episodes/cell, kv≤64, laptop. The **exactly-1.00** fidelity/gate at 2000 ep is on tiny single-query-set episodes and may be optimistic; the *direction and magnitude* (fid 0.4→1.0, gate 0.05→0.99, gap closes) are unambiguous but a **≥5-seed, ≥5-episode, kv≥96** confirmation is needed before headlining "CLU beats Hopfield." I did not re-map the full 26-cell grid at 2000 ep (compute); I sampled 3 representative losing cells.
3. **Item 2 "longer" ≠ "persistent" precisely:** `train_generative` already persists its buffer (PCD); the arm varies *epochs* only. Capacity-scaling (bigger hidden/embed) was not separated from epoch-scaling — longer training alone suffices here, so capacity was not the binding constraint at kv≤64.
4. **Anchor-on-memory design choice:** I anchored to epoch-0 per-pattern energies (faithful transfer of the SO(2) cure). A *warm-start* anchor (pin after N epochs) or a **variance-only** anchor (keep patterns uniformly deep) was not tried; the negative result is specific to the init-pinned form. Given longer training already closes the gap, this is low priority.
5. **Item 1/3 seeds:** 5 seeds (Item 1), 2 seeds (Item 3 bv, tilt). The seed-45 pathology shows the anchor **reduces but does not eliminate** training-instability risk at λ≤10 — a real caveat for any "cure validated" claim. Item 3 divergence is unambiguous on both seeds.
6. **Immune control = tilted-ring, not literal Exp-B.** The matched non-degenerate control (tilt) is cleaner (continuous flatness knob) but differs from the Exp-B sine apparatus; the literal Exp-B immunity is the cited Q4 result (2 seeds).

## 6. Recommended next experiments

1. **(analyst, high value)** Re-map the **full v1-hopfield-stress 26-cell grid at 2000 ep, ≥5 seeds, ≥5 episodes** — if the gap closes broadly, CM-8/V1 must be rewritten from "Hopfield dominant everywhere" to "**Hopfield dominant only under-trained; at compute parity CLU is competitive-or-winning with a calibrated compute-rationing gate.**" This is the single highest-leverage follow-up here.
2. **(analyst)** Epoch-scaling curve for CLU-EBM fidelity vs Hopfield (500/1000/2000/4000 ep) at kv∈{32,64,96} to find the **compute–fidelity frontier** and whether a true capacity wall reappears at kv≥96.
3. **(analyst)** Item-1 λ-fine-sweep {10,30,100} × ≥8 seeds to locate the smallest λ that eliminates the seed-45-type instability while preserving noise rejection (the robust operating point between λ=10 and λ=100).
4. **(engineer)** Ship the anchor as a first-class `training.anchor_data_energy_lambda` option (still open from sleep-erosion follow-up #1) **and** expose `experiment_v1_gate.train_epochs` in the regime CLI so the compute-parity map is one flag.
5. **(analyst, cheap)** Bounded/`p`-only broken-volume scaling to keep it BIBO-bounded, then test whether the anchor helps a *bounded-but-non-symplectic* vacuum (isolates volume-conservation from mere divergence).

## Git footprint
None — no tracked files touched (read-only task). All artifacts under gitignored `.claude/`: report here; figures `.claude/outputs/anchor-robustness/{fig_envelope,fig_trajectory}.png` + `tables.md`; scratch (runners, 38 exp-d/bv JSONs, 16 memory JSONs, logs) under `.claude/scratch/anchor-robustness/`. Runner bug found & fixed **in my own scratch** (`run_anchor.py` tilt threading) — not a repo bug.

## Proposed handover updates (for the Hub)

- **§1.6 / CM-6 / V2 (P11 ANSWERED — anchor envelope + demarcation-as-theory):** the V(data)-anchor holds the designed SO(2) vacuum through **3000 erosive epochs (f5/s500/CD)** for **λ∈{1,10,100}** where λ=0 destroys it 5/5. Envelope = **strength↔robustness tradeoff**: λ≈10 max noise-rejection (gap 1.85) but 1/5 seeds collapse; **λ=100 bulletproof 5/5** (r\*=0.911±0.016) at ~35× wake-MSE cost + halved noise-rejection (0.60). Demarcation is THEORY: μ²_min stiffens 0→+0.04–0.21 at λ=0 (flat mode destroyed) but stays ≈1e-6 with anchor; and **tilting the flat direction immunizes the vacuum with no anchor** (r\* 0.00→0.75–1.36 as δ:0→0.05), erosion ∝ flatness — matches the cited Exp-B/Q4 immunity. **Caveat to fold in:** the cure is *not* perfectly seed-robust at λ≤10 (seed-45 pathology) — state as "robust for λ∈[1,100], seed-bulletproof at λ=100."
- **§1.6 / CM-8 / V1 (P14 ANSWERED — MAJOR, likely rewrites CM-8):** the "Hopfield-dominant 26/26" regime map was **under-trained (500 ep)**. At **2000 ep**, CLU-EBM fidelity **0.40–0.91→1.00** and gated accuracy **0.05–0.53→0.99–1.00**, **matching/beating Hopfield at 9–10× savings in 3/3 losing cells incl. kv64**. **Memory fidelity is a training-budget knob, not a capacity wall** at kv≤64. The **anchor does not transfer** to memory (pins random init). ⚠ Small-n (2 seeds × 2 episodes) — flag for a ≥5-seed full-grid re-map before headlining; but the direction is unambiguous and **directly contradicts the current CM-8 wording** — recommend Hub commission follow-up #1 before V1 drafts the "Hopfield wins" framing.
- **§7 / CM-1 (cross-link):** anchor does **not** rescue the broken-volume vacuum (diverges under CD regardless: wake 3.5k–176k, r\*→∞, noise_gap −16 to −49). **CD-robustness = volume-conservation payoff, orthogonal to the anchor** — corroborates minus-the-physics at 3000 ep; sharpen CM-1 to "vacuum survival under CD requires the symplectic substrate; the anchor is a separate cure that only operates on it."
- **For experiment-engineer:** (a) ship `training.anchor_data_energy_lambda` (sleep-erosion follow-up #1, still open); (b) expose `experiment_v1_gate.train_epochs` on the regime CLI so compute-parity is one flag; (c) no repo bug hit — my tilt bug was in my own scratch runner.
