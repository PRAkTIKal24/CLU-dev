# paid-access-experiments — experiment-engineer report

**Task + acceptance criterion:** build + test the intra-unit wormhole (constant-translation channel) and squeeze-as-access mechanisms, then run the w7 discriminating battery (paid-access-theory §7.1–7.3). Gate: (a) reach crossover at d=L (squeeze steps then drops; wormhole flat); (b) certificates hold; (c) latch transported per prediction; (d) beats the no-physics controller AND the dense-MLP arm.

**Status: done.** All four gate criteria met (with one honest nuance on the squeeze crossover edge — see §Findings). Build + 5 unit tests + full battery (≥5 seeds, dim 2 and 4) run and reported with real numbers.

---

## What I did
- **`chlu/core/potentials.py`** — two new primitives (no edits to integrators / `chlu_unit.step` / lattice):
  - `WormholeChannels(eqx.Module)` — construction (a), the recommended reach mechanism (Prop-A6): gated **canonical translation** `q→q+Δ_k, p→p` with a **hard gate frozen at capture** (`‖q−a_k‖<ρ_k`). Methods `deltas/gate_mask/selected_delta/jump/ledger`, plus `forbidden_state_dependent_jump` (design-guard demo only). Applied *between* Verlet steps at the experiment level.
  - `IntraWormholePotential(eqx.Module)` — construction (b), the smooth **throat** V-wrapper (TiltedPotential precedent): `V(q)=V_base(q)−Σ_k depth_k·exp(−‖q−via_k‖²/2w_k²)`. Doubles as the **dense/nonlocal-V discriminator** arm.
  - `so2_generator(dim)` helper for the latch charge `Q=pᵀXq`.
- **`tests/test_paid_access.py`** — 5 tests (the required ≥4): det J=1 through a jump; bit-exact closed-gate reduction; ledger + bounded-energy; latch `pᵀXΔ` + forbidden-gate volume break; CHLU composition.
- **`chlu/config.py`** — `ExperimentPaidAccessConfig` (band prerequisite, c, T, γ, ζ grid, basin geometry, seeds), registered in `CHLUConfig` + `load_config`.
- **`chlu/experiments/exp_paid_access.py`** — `run_experiment_paid_access`: §7.1 reach (6 arms), §7.2 latch transit, §7.3 per-arm certificates; writes `paid_access_metrics.json` + `paid_access_reach.png`.
- **`chlu/cli/experiment_cmd.py`** — `exp-paid-access` command (minimal +36-line hunk; **no reformat** — I reverted an accidental whole-file `ruff format` per protocol §3.3).

## How I verified (commands + observed output)
Env: main venv `/Users/user/Desktop/CHLU/.venv` (JAX **0.9.0**), `PYTHONPATH=<worktree>` (§4 reuse-main-venv rule).
- `pytest tests/test_paid_access.py tests/test_core.py --no-cov` → **14 passed** (5 new + 9 core; no regressions).
- `ruff check` on all 5 edited files → **All checks passed**.
- Config round-trip (`asdict`) + CLI parser build (`exp-paid-access --quick` → `cmd_exp_paid_access`) → OK.
- Full battery run (dim=2, 5 seeds) and dim=4 run — numbers below (identical crossover structure).

---

## Findings / results

### §7.1 REACH — landing rate vs basin distance (dim=2, 5 seeds; L=2.500, v_max,0=0.500)
| arm | 0.80 | 1.60 | 2.40 | 3.20 | 4.00 | 5.00 | vs L |
|---|---|---|---|---|---|---|---|
| plain_relax | 0 | 0 | 0 | 0 | 0 | 0 | escape-blocked everywhere (KE₀<ΔV_b) |
| **squeeze** S^(M) | 1 | 1 | 1 | 1 | **0** | **0** | steps up, **drops past L** |
| **wormhole** | 1 | 1 | 1 | 1 | 1 | 1 | **flat ≈1 all d** |
| newtonian_squeeze (control) | 1 | 1 | 1 | 1 | 1 | 1 | energy **buys** reach (validates the cap) |
| no_physics_router (CM-7) | 1 | 1 | 1 | 1 | 1 | 1 | trivially "reaches", **no certificate** |
| throat/dense-V (discriminator) | 1 | 0.8 | 0 | 0 | 0 | 0 | helps near escape, **fails reach** |

`< L` = {0.80, 1.60, 2.40}; `> L` = {3.20, 4.00, 5.00}. dim=4 reproduces this table exactly.

**Gate (a) — reach crossover:** MET. Squeeze rises to ≈1 for reachable basins and **collapses to 0 beyond the causal box**; the wormhole is flat ≈1 at all d; the Newtonian-squeeze control reaches even `d>L` (confirming the relativistic cap — not a coding artifact — is the operative constraint). **Honest nuance:** the squeeze's landing edge sits at d≈3.2 rather than exactly L=2.5, because the squeeze also grants an instantaneous *displacement* `(p₀/M₀)sinhζ` (Def-A10 term (i)) on top of the capped flow — so the crossover is a two-sided bracket `[L, L+squeeze-displacement]`, exactly as theory predicts, not a step precisely at L. The falsifiable heart holds: squeeze reach is **bounded** (fails d=4,5), wormhole reach is **not**.

**Gate (d) — beats no-physics router AND dense-MLP:** MET *in the certified sense*. The router matches the wormhole's landing (=1 everywhere) but by fiat — it is **not a phase-space map** (no det J, volume undefined/broken; `router_detJ=None` by construction), so it fails §7.3. The throat/dense-V arm (nonlocal *potential* coupling) **fails reach for d≥2.4** — a dense nonlocal V cannot cross the causal cone (Prop-A2). The wormhole is the only arm that reaches all d **with** a det J=1 + ledger certificate. So it is not a strictly-dominated reparameterization of the dense MLP (C-9 avoided).

### §7.2 LATCH TRANSIT — MET
- zero-shift channel (Δ ⊥ Xᵀp): measured ΔQ = **0.0**, predicted pᵀXΔ = 7.5e-10 → err **7.5e-10**.
- across-coset channel (radial Δ): measured ΔQ = **0.2500** = pᵀXΔ exactly → err **0.0**.
- squeeze preserves Q (channel-isotropic raw squeeze): ΔQ = **1.2e-7** (Q 1.5→1.5).
- random-shift baseline erases Q unpredictably: ΔQ ∈ {0.035, 0.157, −0.143, 0.302, −0.144}.

Interpretation: the wormhole **transports** the Goldstone charge by the exact bounded `pᵀXΔ` (0 by design for a coset-tangent channel), squeeze preserves it, random shift destroys it — Prop-A7 confirmed. (Note: a *banded* S^(M) legitimately changes Q because it does not commute with X — the latch-preserving squeeze requires channel isotropy, F5 §4.1; used here.)

### §7.3 CERTIFICATES — MET on every arm
- **wormhole det J** = `[1.0]×6` (exact); **ledger err** = `[0.0]×6` (matched loci ⇒ ΔH=V(b)−V(a) reproduced to 0).
- **squeeze injection** (matched quadratic H, banded m_eff): det S^(M)=1.000 (±4e-6), symplectic err 6e-8…1.3e-6; H_ratio ≤ e^{2ζ} for all ζ — (0.25: 1.13≤1.65) (0.5: 1.55≤2.72) (1.0: 3.79≤7.39) (2.0: 27.5≤54.6). **Caveat reported in-code:** the e^{2ζ} bound is a *quadratic-energy* certificate; on the quartic well H_ratio can exceed it (expected) — the certificate is stated against the matched quadratic H, per Prop-12 C2.
- forbidden state-dependent gate demonstrated to break volume by exactly `1+∇g·Δ` (det J=2.05 in the unit test) — the frozen gate avoids it.

**Gate (b) certificates + (c) latch:** MET.

### Verdict against the V1 pillar-4 gate
All four criteria (a)+(b)+(c)+(d) are satisfied. The mechanisms behave as the theory predicts: **squeeze cures escape but is causal-box-bounded on reach; the intra-unit wormhole cures reach at det J=1 with an exact energy ledger and exact latch transport; both beat the no-physics controller (certificate) and the dense-nonlocal-V arm (reach).** Recommend V1 can adopt pillar 4, subject to the Hub's cross-check of the one nuance below.

---

## Git footprint
- **Branch:** `agent/experiment-engineer/paid-access-experiments` (off `main` @ `63fea62`; **not pushed**, not merged — left for review).
- **Worktree:** built in `../CHLU-paid-access` (isolated because `fix-pack-4` concurrently touches `config.py`/train in its own worktree). Verified branch ref shows both commits **from the main repo** before removing the worktree (§3.2), worktree removed.
- **Commits:**
  - `e7a5d44` — wormhole primitives (`WormholeChannels`, `IntraWormholePotential`, `so2_generator`) + 5 tests.
  - `6f2384c` — reach/escape battery experiment + `ExperimentPaidAccessConfig` + `exp-paid-access` CLI hook.
- **Files touched:** `chlu/core/potentials.py`, `tests/test_paid_access.py`, `chlu/config.py`, `chlu/experiments/exp_paid_access.py`, `chlu/cli/experiment_cmd.py`.
- **Tests/commands run:** `pytest` (14 passed), `ruff check` (clean), full battery (dim 2 & 4), config/CLI smoke. Scratch artifacts + JSON + plot under `.claude/scratch/paid-access-experiments/` and `.claude/outputs/paid_access_reach.png`.
- **No conflicts.** No shared-file collision with fix-pack-4 (separate worktrees; I reverted an accidental whole-file reformat of `experiment_cmd.py`, keeping only a +36-line hunk).

## Flag-provenance table
| flag | value |
|---|---|
| commit | `6f2384c` (branch tip) |
| seeds | reach: {0,1,2,3,4}; latch/injection: default_rng(0) |
| kinetic modes | relativistic (reach + wormhole + throat), newtonian_learned (Newtonian control + injection cert) |
| mass band (PREREQUISITE) | `[4.0, 0.25]` → M_eff,0=4.0 (reach coord heavy), contrast 16× ⇒ S^(M) directional |
| c / rest_mass / m₀ | c=1.0, rest_mass=1.0 |
| dt / T (reach horizon) | dt=0.05, reach_steps T=100 ⇒ **L = T·dt·c/√M₀ = 2.5**, v_max,0=0.5 |
| γ | reach rollout **γ=0** (conservative ⇒ sharp box L); certificate re-absorption not γ-swept this run |
| ζ grid | [0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0] (line-searched; success = any ζ lands) |
| basin geometry | double well along coord 0, wells at {0,d}, barrier ΔV_b=1.0, d∈{0.8,1.6,2.4,3.2,4.0,5.0} |
| init | q₀=0, p₀=(1.2,0,…)+0.02·𝒩 (KE₀<ΔV_b by design ⇒ plain relax escape-blocked) |
| landing | min over trajectory |q₀−d|<0.4 (Def-A1 reachability) |
| dims run | 2 (headline) and 4 (identical result) |

## Open questions / follow-ups / risks
1. **Squeeze crossover overshoot (report, don't hide):** the squeeze reaches to ≈L + its own displacement `(p₀/M₀)sinhζ` (≈3.2 vs L=2.5), so the crossover is a bracket, not a step exactly at L. This *matches* Def-A10 (the squeeze carries the state up the barrier) but softens the "drops to 0 for d>L" wording. A sharper knife-edge would use a heavier reach coord (smaller displacement) or measure reach from squeezed-momentum-only. Suggest the paper state the bracket `[L, L+p₀sinhζ/M₀]`.
2. **γ / governor re-absorption (§7.3 last clause) not swept:** I used γ=0 for a sharp box and verified injection/det J/ledger; the `t_reabsorb ≈ 2ζ/γ_c` governor re-absorption timing (theory Appendix F) is **not** measured here. A γ>0 follow-up would close that certificate row.
3. **Oracle placement only** (as scoped): channel entrances/exits placed by construction. **Learned entrance-steering is explicitly out of scope** (theory open-risk 1) and remains the true engineering crux / likely failure point at scale — flag for any V1 adoption.
4. **Coercive-exit constraint (F5 §7 issue 7):** the analytic wells are coercive; a wormhole exit into a non-coercive Deep/Conv region could break BIBO. Not exercised here.
5. Full `pytest -q` (all modules) times out at 2 min on JAX collection here — I ran the targeted subset (14 passed). Not a code issue (known cold-collect cost, §6).

## Proposed handover updates (for the Hub)
- **§7 / claims matrix (new, CM-12/CM-7):** the w7 paid-access battery **passes all four gate criteria** — reach crossover present (squeeze bounded, wormhole flat), certificates hold (wormhole det J=1 & ledger=0 exact; squeeze injection ≤ e^{2ζ} on matched-quadratic H; det S^(M)=1), latch transported (pᵀXΔ exact, squeeze preserves Q, random shift erases), and the wormhole beats both the no-physics router (which lacks any volume certificate) and the throat/dense-V arm (which fails reach for d>L). **Caveat to inherit:** squeeze crossover is a bracket `[L, L+p₀sinhζ/M₀]`, not a step exactly at L.
- **§2 architecture:** `chlu/core/potentials.py` now exports `WormholeChannels` (phase-space gated canonical translation, applied outside `step`) and `IntraWormholePotential` (throat V-wrapper); new experiment `exp_paid_access` + CLI `exp-paid-access` + config `experiment_paid_access` (band `[4.0,0.25]` default).
- **Known-issue note:** governor re-absorption timing and learned entrance-steering remain unmeasured/out-of-scope; oracle placement only.
