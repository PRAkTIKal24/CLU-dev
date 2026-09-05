# v1-certificate-payoff — experiment-engineer report

Task + acceptance criterion: turn referee-F3's definitional "wormhole beats the router" into a **measured** result — exhibit ≥1 guarantee the wormhole provably preserves and the no-physics router provably violates, with certificates measured on all arms.
Status: **done** — both items 1 and 2 delivered (not just one), item 3 done. No fallback to referee option (b) needed.

**Headline for V1:** the router's `det J = 0` is not an absent label — it is **measured information destruction**. A cloud of 16 incoming states inside the capture ball exits the wormhole with its Goldstone-charge spread **exactly preserved** (`std(Q_out) = std(Q_in) = 0.0803`, `ΔQ = pᵀXΔ` to 1.2e-7 for *every* state) and exits the router with **bit-identically zero spread** (`std(Q_out) = 0.0`, all 16 land on `Q = 1.475`). The latch the wormhole transports, the router erases.

---

## Check-first result (task instruction)

`paid-access-experiments.md` §7.2 logged a **random-shift** baseline that erases Q, but **no router-latch arm and no router-exit/BIBO arm**. §7.3 recorded `router_detJ = None` ("volume undefined/broken by construction"). Follow-up #4 states the coercive-exit constraint was "not exercised here." So this was **not** a wiring/plotting note — both arms needed new (cheap, analytic, no-training) runs. They now exist.

One correction to the prior report while I was there: **`router_detJ` is not `None`/undefined.** The map `(q,p) ↦ (b, p)` is differentiable with Jacobian `blockdiag(0_d, I_d)`, so `det J = 0` **exactly** — the router is *volume-annihilating and non-invertible*, a strictly stronger and more damning statement than "carries no certificate." Now measured by `jacfwd`, not asserted.

---

## What I did

- `chlu/config.py` — extended `ExperimentPaidAccessConfig` with two knob blocks (`payoff_latch_*`, `bibo_*`). Defaults leave the existing w7 reach/latch/injection battery **bit-identical** (verified: reach table + latch + injection numbers reproduce `paid-access-experiments.md` exactly).
- `chlu/experiments/exp_paid_access.py`
  - `NonCoerciveWell` (new analytic `eqx.Module`, local to the experiment like `DoubleWellReach`/`HarmonicPotential`): `V = ½k·q₀² − ε·q₀⁴ + ½·conf·‖q₁:‖²`. Coercive **only** inside the connected component `|q₀| < x_b = √(k/4ε)`; the Deep/Conv architectural non-coercivity of handover §7 issue 7 made analytic and controllable.
  - `_latch_payoff` (§7.2b, item 1) — 4 arms × a cloud of incoming states.
  - `_bibo_battery` (§7.4, item 2) — 3 arms × 6 requested exits, `r*` measured at `T` and `2T`.
  - `_plot_certificate_payoff` → `paid_access_certificate_payoff.png` (2 panels).
  - reach certificates: `router_detJ` now measured (`0.0`), was `None`.
- `tests/test_paid_access.py` — 2 new tests (5 → 7).

Deliverables: `.claude/outputs/v1-certificate-payoff/{paid_access_certificate_payoff.png, paid_access_metrics.json, paid_access_reach.png}`.

---

## How I verified (commands + observed output)

Env: main venv `/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0** (matches the w7 provenance; no worktree sync, so no 0.10.2 drift per protocol §4).

- `uv run --no-sync pytest tests/test_paid_access.py tests/test_core.py --no-cov -q` → **16 passed** (7 paid-access + 9 core; no regressions).
- `uv run --no-sync ruff check chlu/config.py chlu/experiments/exp_paid_access.py tests/test_paid_access.py` → **All checks passed**.
- YAML round-trip of the new knobs through `save_config`/`load_config` → OK (`bibo_quartic=0.03`, `payoff_latch_samples=7` survive; this is the gate the Hub's `37dc664` fixed).
- Full battery `quick=False` (5 seeds, 6 distances) + `quick=True` smoke → both run; §7.1/§7.2/§7.3 numbers unchanged from `paid-access-experiments.md`.
- Robustness sweep `dim ∈ {2,4} × seed0 ∈ {0,7}` → all four cells identical in structure (below).

**Two test failures found and fixed during development (reported per §6):**
1. First cut of the latch test sampled jitter as `0.2·𝒩(0,1)`, which puts some incoming states **outside** the capture radius 0.35 → the hard gate never fires → `dQ = 0` instead of `pᵀXΔ = −0.025`. **Test bug, not code bug** (the experiment samples uniformly in a ball of radius `payoff_capture_jitter = 0.3 < 0.35`). Fixed by sampling on a shell of radius 0.25 and asserting *all* states are captured.
2. My "b=4.0 has a cheaper ledger than b=3.0" assertion was **false**: `V(3.0) = V(4.0) = 2.88` exactly (numerical coincidence of `½x²−0.02x⁴`). The genuine free-ledger row is **b=5.0**, where `V = 0.0` exactly. Assertion rewritten around b=5.0 — which is a *stronger* fine-print guard (see below).

---

## Findings / results

### PAYOFF A — the router erases the latch the wormhole transports (§7.2b, item 1)

16 incoming states drawn uniformly in the capture ball (radius 0.3 < ρ = 0.35) around an entrance on the vacuum circle; fixed momentum `p`; `Q = pᵀXq`; `std(Q_in) = 0.0803`.

| arm | det J | ΔQ mean | ΔQ std | predicted `pᵀXΔ` | max err vs prediction | **std(Q_out)** |
|---|---|---|---|---|---|---|
| wormhole, coset-tangent | **1.0000** | −0.0000 | 0.0000 | 0.0000 | 1.2e-07 | **0.0803** |
| wormhole, across-coset | **1.0000** | 0.2500 | 0.0000 | 0.2500 | **0.0** | **0.0803** |
| random shift (det J=1, no channel) | 1.0000 | 0.0972 | 0.2465 | *(no receipt)* | — | 0.2793 |
| **no-physics router** | **0.0000** | 0.0379 | 0.0803 | *(no receipt)* | — | **0.00e+00** |

- **The guarantee:** the wormhole's canonical translation is **injective** → it shifts *every* incoming state's charge by the *same exact constant* `pᵀXΔ` (`ΔQ std = 0`, err ≤ 1.2e-7), so the stored spread survives: `std(Q_out) = std(Q_in)` to all printed digits. Reconstruction `q_in = q_out − Δ` is exact (max err **2.2e-08**).
- **The violation:** the router maps the whole capture ball onto one point → all 16 states exit with `Q = 1.475` **bit-identically** (`max|Q_out − Q_out[0]| = 0.0`). `std(Q_out) = 0` ⇒ the latched coset content is **irrecoverable**. This is `det J = 0` cashed out as a measured downstream consequence.
- **Honest nuance (must go in the paper):** the `random_shift` arm has `det J = 1` and *still* scrambles Q (`ΔQ std = 0.2465`). **Volume preservation alone is not the latch certificate** — the *matched channel* (Δ chosen coset-tangent, so `XΔ ⊥ p`) is what makes `ΔQ = 0`. The full receipt is `det J = 1` **and** the channel's `pᵀXΔ` ledger.

Replicates exactly across `dim ∈ {2,4} × seed0 ∈ {0,7}`: wormhole `det J = 1.0000`, router `det J = 0.0000`; `std(Q_out)_wormhole = std(Q_in)` in every cell (0.0803 / 0.0679 / 0.0533 / 0.0448); `std(Q_out)_router = 0.00e+00` in every cell.

### PAYOFF B — uncertified exit blows up BIBO; the receipt keeps it bounded (§7.4, item 2)

`V = ½q₀² − 0.02·q₀⁴` ⇒ coercive edge `x_b = 3.536`, barrier `V_b = 3.125`. γ = 0.02, T = 2000, relativistic, 5 seeds. `r* = max_t‖q_t‖`.

| arm | b=1.0 | 2.0 | 3.0 | 3.6 | 4.0 | 5.0 |
|---|---|---|---|---|---|---|
| wormhole + receipt (screened) | 1.01 | 2.01 | 3.01 | **0.09** | **0.09** | **0.09** |
| wormhole, receipt ignored (ablation) | 1.01 | 2.01 | 3.01 | **102.13** | **103.43** | **104.83** |
| no-physics router (no receipt) | 1.01 | 2.01 | 3.01 | **102.13** | **103.43** | **104.83** |
| — escape rate, certified | 0 | 0 | 0 | 0 | 0 | 0 |
| — escape rate, blind / router | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| **wormhole receipt** | ADMIT | ADMIT | ADMIT | REJECT | REJECT | REJECT |
| *energy-only sub-level test* | admit | admit | admit | reject | **admit** | **admit** |

- **BIBO diagnostic (not just a big number):** `r*(2T)/r*(T)` = **1.000** for every bounded arm/exit (saturated) vs **1.96 / 1.94 / 1.91** for the escaping ones (≈2 ⇒ `r* ∝ T` ⇒ genuinely unbounded, terminal velocity capped at `c/√M₀ = 0.5` by the relativistic governor — **the causal cap bounds speed, not excursion**).
- **The receipt predicts BIBO blow-up 6/6 exits.** Ledger `ΔH = V(b) − V(a)` is exact to **0.0** at every exit.
- **The killer row (b = 5.0):** `V(5.0) = 0.0` ⇒ the energy ledger says the jump is **FREE** (ΔH = 0, *cheaper* than the admissible b=3.0's ΔH = 2.88), and an energy-only sub-level test **admits** it — yet it escapes. Likewise b=4.0 (`V = 2.88 < V_b`, admitted by energy, escapes). **`det J = 1` + a bounded/free energy ledger is NOT sufficient for BIBO; coercive-component membership is the operative clause.** This is exactly the C-6 / F5-issue-7 fine print the referee wanted next to the claim, and it is now measured, with a regression test.

**Attribution — the honest part.** `wormhole_blind` and `no_physics_router` **coincide exactly** (both land at `b`, neither is screened). So what buys BIBO is **the receipt, not the jump mechanism**. The defensible claim is therefore:

> The wormhole *can* form the receipt (it has the unit's `V_θ`, an exact energy ledger, and a well-defined Jacobian); a no-physics router has none of the three and cannot screen its own exit even in principle. Bolting a coercivity check onto the router means handing it the potential and the energy accounting — i.e. the certificate machinery.

Payoff A, by contrast, is a **mechanism-level** violation requiring no screening argument: the router's `det J = 0` *is* the erasure. **If V1 embeds only one, embed A.** (I drew the coincident B curves as a thick translucent band + dashed overlay so neither is occluded — the referee's own F1 lesson.)

**Trade the certificate makes explicit:** on rejected exits the certified wormhole **does not reach** the target (`r* = 0.09`, it stays home). The receipt does not make an unsafe exit safe; it *refuses* it. That is the correct reading and should be stated, not hidden.

### Item 3 — certificates re-verified on all arms (apples-to-apples)

| arm | det J | ledger err | latch receipt |
|---|---|---|---|
| wormhole (reach, 6 distances) | `[1.0]×6` exact | `[0.0]×6` | `ΔQ = pᵀXΔ`, err ≤ 1.2e-7 |
| wormhole (BIBO, 6 exits) | 1.0 | 0.0 (all exits) | — |
| squeeze `S^(M)` | `det S = 1.000 ± 4e-6` | `H_ratio ≤ e^{2ζ}` ✓ (matched-quadratic H) | preserves Q (1.2e-7) |
| random shift | 1.0 | *(none)* | **scrambles Q** |
| **no-physics router** | **0.0** (measured) | *(none: no V)* | **erases Q** |

Reach table, latch transit, and squeeze injection all reproduce `paid-access-experiments.md` to the digit → the additions are non-invasive.

---

## Flag-provenance table

| flag | value |
|---|---|
| commit | `27f232f` (branch tip); payoff code `d9a9f38`; config `f2a85aa` |
| base | local `main` @ `37dc664` |
| JAX / venv | **0.9.0**, main venv (`--no-sync`, no worktree resolve) |
| seeds | reach/BIBO: `{0,1,2,3,4}` (n_seeds=5, seed0=0); latch cloud: `default_rng(0)`; robustness: seed0 ∈ {0,7} |
| dims run | **2** (headline) and **4** (identical structure) |
| kinetic mode | **relativistic** (reach, wormhole, throat, BIBO); `newtonian_learned` (Newtonian control, injection cert) |
| mass band (PREREQUISITE) | `[4.0, 0.25]` ⇒ `M_eff,0 = 4.0`, contrast 16× |
| c / rest_mass | `c = 1.0`, `m₀ = 1.0` ⇒ `v_max,0 = 0.5` |
| dt | 0.05 |
| γ | reach: **0** (sharp box `L = 2.5`); **BIBO: 0.02** (bounded arms must settle) |
| BIBO potential | `k = 1.0`, `ε = 0.02` ⇒ `x_b = 3.5355`, `V_b = 3.125`; transverse conf = 4.0 |
| BIBO exits / horizon / escape | `b ∈ {1.0,2.0,3.0,3.6,4.0,5.0}`; `T = 2000` (`r*` also at `2T = 4000`); escape radius 20.0; `p₀ = 0.3`; margin 1e-3 |
| latch payoff | 16 incoming states, ball radius `0.3 < ρ = 0.35`; `f = 3.0`, `p_latch = 0.5`; `‖Δ‖ = 0.5` |
| ζ grid (injection) | `[0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0]` |
| training | **none** — analytic potentials, oracle channel placement (learned entrance-steering out of scope) |

---

## Git footprint

- **Branch:** `agent/experiment-engineer/v1-certificate-payoff`, off local `main` @ `37dc664`. **Not pushed, not merged.**
- **Commits (3, atomic):**
  - `f2a85aa` — config knobs (`payoff_latch_*`, `bibo_*`), behavior-preserving defaults.
  - `d9a9f38` — `NonCoerciveWell` + `_latch_payoff` + `_bibo_battery` + payoff figure + measured `router_detJ`.
  - `27f232f` — 2 new tests.
- **Files touched (3):** `chlu/config.py` (+26), `chlu/experiments/exp_paid_access.py` (+445/−2), `tests/test_paid_access.py` (+120). No edits to `integrators.py`, `chlu_unit.step`, `potentials.py`, or the lattice.
- **Worktree:** none needed — `git status` was clean, no other worktrees (`git worktree list` = main only), no concurrent branch checked out.
- **Rebase / ⚠ REPO FINDING (needs the Hub's attention):** `git rebase origin/main` **must not be run on this repo right now.** `origin/main` is stale at **`40c2f31 "rm docs/"`** (2026‑07‑02); local `main` is **82 commits ahead** and has **never been pushed**. Rebasing onto `origin/main` tries to replay all 82 foreign commits (MQAR, Experiment D, squeeze, …) and conflicts in `chlu/data/__init__.py` — files outside my scope. Per §3.5 I **aborted** (`git rebase --abort`), verified the tree is clean and my 3 commits intact, and re-ran the suite (**16 passed**) after the abort. My true base, local `main`, has not moved, so rebase-onto-base is a no-op and my branch is a clean fast-forward. **No conflicts remain; nothing was clobbered.**

---

## Open questions / follow-ups / risks

1. **B's arms coincide by construction.** `wormhole_blind ≡ no_physics_router` for BIBO. The measured claim is "*the receipt* buys BIBO, and only the wormhole can form one" — not "the wormhole's jump is intrinsically safer." Paper must say this. **A has no such caveat** and is the stronger embed.
2. **Still the designed testbed.** Oracle channel placement, analytic potentials, no training. This does *not* claim a real-data or learned-memory win — it demonstrates the certificate's downstream consequence, which is exactly what F3 asked for. CM-12 scope wording stands.
3. **Coercive screening is oracle too.** `in_coercive_component` uses the analytic `x_b`. For a learned `V_θ` (Deep/Conv, non-coercive by §7 issue 7) the component boundary is not known in closed form — certifying exits on a *trained* potential is genuinely open and is the natural next experiment (a sub-level-set estimator, or restoring the `α‖q‖²` confinement that Deep/Conv drop).
4. **b = 3.6 is 0.06 outside `x_b`** — the nearest escaping exit is close to the edge, but `r*(2T) = 102` vs `0.09` and the growth ratio 1.96 make the classification unambiguous (no knife-edge, unlike the w7 squeeze-crossover bracket).
5. `pytest -q` over the whole suite still hits the known slow-JAX collect cost; I ran the targeted subset (16 passed). Not a code issue (§6).

---

## Proposed handover updates (for the Hub)

- **§7 Known Issues — new (repo hygiene, HIGH):** `origin/main` is stale at `40c2f31` (2026‑07‑02) while local `main` is **82 unpushed commits ahead**. **Any spoke that follows protocol §3.5's `git rebase origin/main` literally will hit conflicts in foreign commits.** Protocol §3.5 or the handover should say: *rebase onto your **named base** (local `main`), not `origin/main`, until the remote is synced.* This bit me; it will bit the next agent.
- **§7 issue 7 (coercive exits) — status change: PROVEN + MEASURED, no longer "not exercised."** A wormhole exit outside the coercive component escapes with `r* ∝ T` (growth 1.91–1.96), while the receipt-screened exit stays bounded (`r*(2T)/r*(T) = 1.000`). Critically: **`det J = 1` + a bounded (even *zero*) energy ledger is insufficient** — `b = 5.0` has `ΔH = 0.0` (free) and still escapes; only coercive-**component** membership catches it. `paid-access-experiments.md` follow-up #4 is now closed.
- **Correction to `paid-access-experiments.md` §7.3 / the claims matrix:** `router_detJ` is **`0.0` (measured), not `None`/"volume undefined"**. The router map `(q,p)↦(b,p)` has `det J = 0` exactly ⇒ volume-annihilating, non-invertible. Stronger claim; the draft's "carries no volume certificate" understates it.
- **CM-7 / CM-12 — F3 is dischargeable, and V1 should NOT take the referee's option (b).** Recommended §3.2 wording:
  > The wormhole reaches with a receipt the router cannot supply — and the receipt has a measured consequence: at `det J = 1` the channel *transports* a stored Goldstone charge by the exact `pᵀXΔ` (spread preserved, `std(Q_out)=std(Q_in)=0.0803`), whereas the `det J = 0` router maps the whole capture ball to one point and **erases** it (`std(Q_out)=0`). Volume preservation alone is not sufficient (a random `det J = 1` shift scrambles Q); nor is a bounded energy ledger sufficient for BIBO (a free `ΔH = 0` exit into a non-coercive region escapes with `r* ∝ T`).
- **Figure guidance (F1/F3):** `paid_access_certificate_payoff.png` panel A is the natural **headline replacement or companion** for the occluded reach figure — it is the one plot where wormhole and router are *visibly* different (slope-1 line vs flat line), which the landing-rate plot structurally cannot show. Panel B needs the "blind ≡ router overlap" caption.
- **§2 architecture:** `exp_paid_access` gains `NonCoerciveWell`, `_latch_payoff`, `_bibo_battery`, `_plot_certificate_payoff`; config gains `payoff_latch_*` + `bibo_*` (defaults preserve w7 behavior bit-for-bit).
