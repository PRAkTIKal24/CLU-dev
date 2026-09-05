# traj-write-objective — experiment-engineer report

**Task + acceptance criterion:** ask the write objective, explicitly, to put information into the
trajectory/path (`λ_traj`, `λ_path`, both default 0.0), then measure honestly whether it buys anything
— acceptance = half the C2W2 gate's evidence, on the frozen race card, with liveness + convergence +
four launders on every cell.

**Status: done** (D0–D6 all run; two families ABSTAIN after their one bounded escalation — reported as
such, not hidden).

> ⭐ **RACE+SEAM FREEZE COMMIT = `1ce02e4`** (reported to the Hub mid-flight; the other two engineer
> branches were unblocked on it). Branch `agent/experiment-engineer/traj-write-objective`, base local
> `main` `233fd9e`, worktree `../CHLU-traj-write-objective`. **Verified from the MAIN repo.**

## ⛔ RECONCILIATION LIST (protocol §5 corollary — needs an owner, in the first 10 lines)
1. **D6 / spike R-4: the trajectory launder FIRES for `AttentionPsi`.** `q0_only` **0.3515–0.4480**
   and `blank_store` **0.3713–0.4728** against a bar of **0.1902**, `blank_leak=1` at *every* stride.
   ⇒ **No `AttentionPsi` trajectory number anywhere is quotable** until re-run store-relative. Owner
   needed: `phi-particle-head` (owns `psi_readout.py`, running ψ families this wave).
2. **The gym's ridge-saddle `λ_min = −0.5946` is a SINGLE-SEED number.** Reproduced exactly at seed 0,
   but the 3-seed mean at `λ_path=0` is **+0.1770 ± 0.4694** (seeds −0.5946 / +1.026 / +0.0996). The
   charter §3.5 "a multi-target ridge write produces a saddle" framing needs a multi-seed qualifier.
3. **The recency family was a harness DEFECT** (not a null) — fixed, default-off; `negative_results.md`
   / the gym's F3 row need updating (pre-fix `0.3019 ± 0.0679` is a scoring-domain artefact).
4. **D5 falsifies "the store was never written."** At sub-shipped atom budgets the write loss is
   **flat in write steps** (0.206→0.203 at 6× budget) ⇒ the gym's sub-shipped cells are a genuine
   **capacity** result, not an optimisation artefact. The gym's §8.1 wording needs this.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** pillar 1 — expressive latents. KPI = `dividend ≡ (full CLU) − (its own
  settle-deleted launder)`, same store, same φ, same read budget.
- **Laundering control:** four on every cell — settle-deleted launder, same-keys null, blank store,
  **trajectory launder** (`full`/`q0_only`/`endpoints`/`blank_store`) on every ψ that sees the address
  block — plus the family's strongest **+0 B substitute** with its signed margin.
- **Falsifies:** coefficient-zero regression gate fails; the trajectory launder fires; liveness fails
  at every registered coefficient *with a perturbing anchor present*.
- **Does NOT falsify:** dividend ≤ 0 **with liveness passed** (the gate's own ≤0 branch, a result);
  17.1× ψ wall-clock; `matched=False` bytes (architectural ≥2.20×, gym PREREG-B1); losing to a
  classical method on a metric-native protocol.

---

## 1. HEADLINE — admissible-cell coverage per family (⛔ first-class, not buried)

| family | admissible / cells | coverage | after the **one bounded escalation** | gate role |
|---|---|---|---|---|
| **`overload`** @ shipped atom budget (478× anchor) | **25 / 30** | **83 %** | n/a (not needed) | **VOTES** |
| `aggregate` | **0 / 12** | **0 %** | 900 steps → 0/9 · 1800 steps → 0/9 | **ABSTAINS** |
| `manifold` | **0 / 12** | **0 %** | 900 steps → 0/9 · 1800 steps → 0/9 | **ABSTAINS** |
| `recency` | (D4: defect, fixed & re-baselined) | — | — | see §5 |

**Every excluded cell is listed with its reason** in `race_card.md` / `race_card_scored.json` (29
exclusions; nothing silently filtered). Exclusion reasons are exclusively *write-side*: endpoint write
loss > 0.05, or `λ_min < 0`. **`under_powered_grids: []`** — no arm was excluded for an under-powered
grid, because the perturbing anchor was found (§3).

**ABSTAIN means what the ruling says it means:** `aggregate` and `manifold` neither block B′ nor
support "proceed." Only `overload` votes.

## 2. THE GATE'S ARITHMETIC (the verdict is the Hub's)

`any_family_clears = False`. `cleared_two_se = []`. `weak_proceed = []`. **Nine `le_zero_vote` arms,
all on `overload`:**

| arm | dividend ± SE (3 seeds) | clears 2 SE? | +0 B substitute margin | admissible | grade |
|---|---|---|---|---|---|
| `endpoint_write` (**the control = today's objective**) | **−0.0278 ± 0.0139** | no | **+0.2639** | 3/3 | ≤0 vote |
| `traj_write@0.03` | −0.1111 ± 0.0278 | no | +0.1806 | 3/3 | ≤0 vote |
| `traj_write@0.3` | −0.1806 ± 0.0139 | no | +0.1111 | 3/3 | ≤0 vote |
| `traj_write@3` | −0.1944 ± 0.0278 | no | +0.0972 | 3/3 | ≤0 vote |
| `traj_write@30` | −0.2639 ± 0.0773 | no | +0.0278 | 3/3 | ≤0 vote |
| `path_write@0.03` | −0.2639 ± 0.0139 | no | +0.0278 | 3/3 | ≤0 vote |
| `path_write@0.3` | −0.2917 ± 0.0636 | no | +0.0000 | 3/3 | ≤0 vote |
| `path_write@3` | −0.7083 (1 seed) | no | −0.2917 | 1/3 | ≤0 vote |
| `path_write@30` | n/a | no | n/a | 0/3 | abstain |
| `traj+path@0.3` | −0.4722 ± 0.0501 | no | −0.1806 | 3/3 | ≤0 vote |

⭐ **The result is not "no effect" — it is a monotone COST.** Asking the write to put information in
the trajectory makes the dividend *monotonically more negative* in the coefficient, on the one family
where the store demonstrably works, across 3 seeds, at every coefficient in a 3-decade grid. The
`path` term is worse than the `traj` term, and the two together are worse than either.
⛔ Byte ratio **478.20×** on every `overload` cell — **architectural, never quotable as a byte-matched
dividend** (gym PREREG-B1).

**Note on the control:** `endpoint_write` reproduces C2W1's anchor (seed 0: full **1.0000**, launder
**1.0000**, dividend **exactly 0.0000**) and is the *only* arm whose +0 B substitute margin is
comfortably positive (**+0.2639**) — i.e. today's objective already beats its strongest +0 B
substitute on `overload/decode`, it just has **no dividend over its own settle-deleted launder**.

## 3. LIVENESS AND THE PERTURBING ANCHOR (gate ruling (ii) — the counterweight)

⭐ **The grid carried its perturbing anchor, so the ≤0 vote is legitimate** (not an under-powered grid).
Registered bar (PREREG §1.3): *loss ≥3× the λ=0 value, or strict recall drops ≥0.20 absolute.*
Measured on `overload/load1x_shipped`, seed 0 (`anchor_overload_load1x_shipped_traj_s0.json`):

| λ_traj | write loss | full (decode) | dividend | strict recall | decode | λ_min |
|---|---|---|---|---|---|---|
| 0 | 0.0005 | 1.0000 | **0.0000** | **1.000** | 1.000 | 3.2958 |
| 0.03 | 0.0165 | 0.9167 | −0.0833 | 1.000 | 1.000 | 3.3044 |
| 0.3 | 0.1650 | 0.7917 | −0.2083 | 0.792 | 0.792 | 3.3226 |
| **3.0** | 1.6500 | 0.7500 | −0.2500 | **0.333** | 0.500 | 2.6571 |
| 30 | 16.5000 | 0.5833 | −0.4167 | 0.333 | 0.333 | 2.8480 |
| 300 | 165.0000 | 0.5833 | −0.4167 | 0.333 | 0.333 | 2.8637 |

**Anchor: λ_traj = 3.0** — strict recall **1.000 → 0.333** (drop 0.667, 3.3× the registered 0.20 bar).
For `λ_path` the anchor is even sharper: at λ ≥ 3 the term drives the write itself **inadmissible**
(`λ_min` −0.7197 / −0.4793 / −1.3022). *The term was asked, not whispered at.*

⛔ **But the P-A liveness bar FAILED at every coefficient:** the trajectory-written store's trajectory
does **not** carry more decodable information than the `λ=0` store's (registered bar: +0.05 over the
capacity-matched `endpoints` baseline *and* the blank store). `live = no` on all 9 arms.

**The mechanism, measured** (`grad_probe.json`, `anchor_traj_aggregate.json`): the recorded write loss
is **exactly** `L_endpoint + 0.55·λ` at every λ from 0.03 to 300 — the penalty is a **constant** the
optimiser never reduces. The term drags θ without ever descending its own objective. Diagnosis
(`rollout_vs_read`): the write-side rollout is *correct* — at the shipped band it reproduces the read's
settled payload bit-for-bit (`0.3333 / −0.9997 / −0.3334 / 0.9988` vs the read's
`0.3333 / −0.9997 / −0.3334 / 0.9988`) — but a **60-step** path only reaches ~0.27/0.50 of a payload
the read needs **400 steps** to climb to, so the path read-out `m_i` is dominated by the near-zero
early path and is nearly θ-independent.

## 4. THE MANDATORY TRAJECTORY LAUNDER — **0 / 54 cells fired** (race card)

| family | `full` | `endpoints` (capacity-matched) | `q0_only` | `blank_store` | bar | fired |
|---|---|---|---|---|---|---|
| `overload` | **+0.6681** | **+0.6681** | +0.1667 | +0.1667 | +0.3949 | **0/30** |
| `aggregate` | −0.5554 | −0.5554 | −0.4220 | −0.4221 | −0.2299 | **0/12** |
| `manifold` | −0.1772 | −0.1772 | −0.1772 | −0.0001 | +0.0032 | **0/12** |

No leak on the race card's ψ. ⭐ **And `endpoints` equals `full` to 4 decimals on every family and
every arm** — the capacity-matched 3-point read `[q0, q_addr, q*]` is indistinguishable from the full
strided buffer. **Honest caveat:** the gym's trajectory ψ is the *handcrafted tail-mean*, which reads
mostly the tail by construction, so this equality is partly tautological for that ψ; it is evidence
about the gym's v0 ψ, not about learned ψs. The learned-ψ statement is §6.

## 5. D4 — the recency family was a **HARNESS DEFECT**, diagnosed, fixed, re-baselined

**Diagnosis (code + measurement).** `queries_recency` asks *"which of these TWO items is more recent"*
and `score_index` grades against a **2-way chance of 0.5** — but every CLU arm (`argmax` occupancy,
`point_assign`) and the frozen settle-deleted launder answer an unrestricted **K-way** question, while
only `order_aware_launder(k=2)` was restricted to the pair. Measured (seed 0, K=5, 9 pairs, 72
queries): **the CLU's answer falls outside its own pair 19.4 % of the time**. A 6-way answer was being
graded on a 2-way curve. `1 − 0.3019 = 0.698` is a coincidence, **not** a label inversion.

| | full | launder | blank | dividend | +0 B `order_aware` |
|---|---|---|---|---|---|
| **pre-fix** (reproduces the gym **exactly**) | **0.3019 ± 0.0679** | 0.4495 | **0.3065** | −0.1477 ± 0.0495 | 0.7764 |
| **post-fix** (re-baselined control) | **0.4769 ± 0.0699** | 0.4796 | 0.5463 | **−0.0028 ± 0.0619** | 0.7764 |

Fix shipped as `GymConfig.restrict_index_to_pair` (**default `False` = shipped behaviour**) +
`restrict_to_pair()`; the frozen launder is restricted too, so it is not a thumb on the scale. After
the fix both the CLU **and its blank store** sit at 2-way chance and the dividend is ≈0 — the family
is no longer nonsense, but it carries no dividend and is beaten by its **+0 B insertion-order
substitute by −0.30**.

## 6. D6 — ⛔ spike R-4 CLOSED, and it FIRES

`AttentionPsi`, K=8, 2000 fit steps, params matched 4609, `chance = 0.1386`, `bar = 0.1902`:

| stride | `full` | `q0_only` | `endpoints` | `blank_store` | leak |
|---|---|---|---|---|---|
| 1 | 0.6658 | **0.4134** | 0.3490 | **0.4059** | **YES** |
| 2 | 0.6460 | **0.4332** | 0.3168 | **0.4381** | **YES** |
| 4 | 0.6460 | **0.4480** | 0.3391 | **0.4728** | **YES** |
| 8 | 0.6510 | **0.4257** | 0.3045 | **0.4653** | **YES** |
| 16 | 0.6386 | **0.3515** | 0.2970 | **0.3762** | **YES** |
| 32 | 0.6658 | **0.3911** | 0.3465 | **0.3713** | **YES** |

**The untested hypothesis is CONFIRMED:** a **pooled** DeepSets ψ dilutes `q₀` to 1 of 150 points
(C2W1: `q0_only` 0.129 vs chance 0.125, no leak), while an **attention** ψ *selects* it — `q0_only`
jumps to ≈0.41, **3.0× chance and 2.2× the bar**, at every stride. A **blank store** read by an
attention ψ scores 0.42. Its internal traj-minus-point "dividend" (+0.0025, 1 seed, 359.2× bytes) is
therefore **NOT QUOTABLE**. Run *before* the headline cells, as the task required — which is why the
race card's numbers (§2) are unaffected: they use the gym's handcrafted ψ, whose launder did not fire.

## 7. D5 — Rider B: the longer-write curve is **FLAT** (pillar (a), finally tested)

`overload`, sub-shipped atom budgets, 3 seeds, the **curve** (never its endpoint):

| atoms/item | write steps | decode | λ_min | final write loss | byte ratio |
|---|---|---|---|---|---|
| 8 | 300 | 0.2222 ± 0.0556 | +0.0667 ± 0.0333 | 0.2064 ± 0.0106 | 12.0× |
| 8 | 900 | 0.2083 ± 0.0636 | +0.0673 ± 0.0327 | 0.2117 ± 0.0157 | 12.0× |
| 8 | **1800** | 0.2083 ± 0.0636 | +0.0674 ± 0.0326 | **0.2030 ± 0.0120** | 12.0× |
| 16 | 300 | 0.3333 ± 0.0962 | −0.1821 ± 0.2017 | 0.2163 ± 0.0102 | 23.2× |
| 16 | 900 | 0.3333 ± 0.0962 | −0.3090 ± 0.3232 | 0.2216 ± 0.0113 | 23.2× |
| 16 | **1800** | 0.3889 ± 0.0556 | −0.4289 ± 0.4401 | 0.2129 ± 0.0066 | 23.2× |

⭐ **6× the write budget moves nothing**: `Δdecode(1800−300)` = **−0.0139** (api 8) and **+0.0556**
(api 16), against a registered discriminator of ≥ +0.25. The write loss plateau at **0.20–0.22** is
**not an optimisation-budget artefact** — it is the atom budget's expressivity floor. (λ_min gets
*worse* with more steps at api 16: +0.18 → −0.43.)

**The escalation, same evidence** (36 cells = 2 families × {900, 1800} steps × 3 arms × 3 seeds):
`aggregate` and `manifold` stay **0/9 admissible at BOTH budgets**, and the **`endpoint_write` control
arm's** endpoint write loss is flat in budget — `aggregate` **0.2821 ± 0.0392 → 0.2818 ± 0.0391**,
`manifold` **0.2934 ± 0.0427 → 0.2932 ± 0.0427` (900 → 1800 steps). ⚠ Read those as *control-arm*
numbers: pooled over all three arms the same means are 0.3926 → 0.4555 (`aggregate`) and 0.3137 →
0.3371 (`manifold`), because the Route-1 arms raise the endpoint loss too — which is itself part of
the §2 cost finding, not evidence about budget. Escalation spent; both families **ABSTAIN**.

## 8. P4 — `λ_path` at the ridge item: **both registered hypotheses FAIL**

`manifold/ridge`, λ_min at the ridge item (site 0, 3 seeds). ⭐ **Seed 0 at λ_path=0 reproduces the
gym's `−0.5946` exactly.**

| λ_path | λ_min @ ridge item | λ_min (min over sites) | softest-mode spectator participation |
|---|---|---|---|
| 0 | **+0.1770 ± 0.4694** (seeds −0.5946 / +1.026 / +0.0996) | −0.2410 ± 0.2001 | 0.609 ± 0.280 |
| 0.03 | −1.5621 ± 1.9012 | −3.6710 ± 1.7410 | 0.414 ± 0.229 |
| 0.3 | −2.3258 ± 1.2265 | −4.6658 ± 1.1843 | 0.083 ± 0.078 |
| 3.0 | −3.1749 ± 1.7265 | −3.7191 ± 2.2671 | **0.006 ± 0.004** |
| 30 | −2.1958 ± 1.1205 | −2.2451 ± 1.0755 | 0.011 ± 0.009 |

- **H-path-B (the task's acceptance: `λ_min ≥ 0`) — FALSIFIED.** No coefficient reaches ≥0 over 3 seeds.
- **H-path-A (mine: plateau just below 0, −0.02 ± 0.03 at λ=30) — ALSO FALSIFIED**, by two orders of
  magnitude (measured −2.20 ± 1.12).
- ⭐ **But the designed half works:** the tangent penalty *does* remove the spectator direction from
  the soft mode — participation collapses **0.609 → 0.006**. It buys that by wrecking the conditioning
  everywhere else. (Informative, and explicitly a non-falsifier per task §6.)
- ⚠ The ridge write only sees `λ_path` because I wired the seam into `_ridge_write` — a ridge is not a
  controller verb and is applied directly; without that patch the very arm the term was designed for
  would have run with the term OFF.

## 9. PREREG SCORECARD (11 registered predictions: **4 survive, 6 fail, 1 partial**)

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | `λ_traj=λ_path=0` ⇒ written `V` bit-identical to `main` | bit-identical (2 blocking tests) | ✅ **SURVIVES** |
| P2 | `λ_traj` LIVE at {3, 30}, inert at {0.03, 0.3} | **inert at ALL** (penalty exactly constant) | ❌ **FAILS** — the registered *counter*-prediction (0.4) won |
| P3 | perturbing anchor at λ=30 (loss ≥3×, or recall −0.20) | anchor found at **λ=3** (recall 1.000→0.333) | ✅ **SURVIVES** (earlier than predicted) |
| P4 | `λ_min` at ridge plateaus just below 0 (H-path-A) | −2.20 ± 1.12 at λ=30 | ❌ **FAILS** (as does H-path-B) |
| P5 | no family clears 0 beyond 2 SE ⇒ B′ branch | `any_family_clears = False` | ✅ **SURVIVES** |
| P6 | `manifold` +0 B substitute margin ≈ −0.90 ± 0.10 | **−1.1772 ± 0.1053** (n=12; `echo` = 1.0 by construction) | ◐ **PARTIAL** — direction right, magnitude outside the registered band |
| P7 | trajectory launder does NOT fire (`q0_only ≈ 0.129`) | 0/54 on the race-card ψ; `q0_only` = 0.1667 = chance | ✅ **SURVIVES** (race card only — see P8) |
| P8 | attention ψ `q0_only ≈ 0.16 ± 0.04`, still below bar | **0.3515–0.4480, FIRES at every stride** | ❌ **FAILS** — the registered alternative (0.30) won |
| P9 | recency = label inversion; fixed → 0.70, dividend ≈0 | defect ✔ but it is a **K-way/2-way domain mismatch**; fixed → 0.4769 | ❌ **FAILS** (defect direction right, mechanism & number wrong) |
| P10 | longer-write knee ≈900; `Δdecode(1800−300) ≥ 0.25` | curve **FLAT**; Δ = −0.014 / +0.056 | ❌ **FAILS** decisively |
| P11 | `λ_traj` write ≤ 6× endpoint-write wall-clock | **2.60–3.00×** warm (n=3/cell); `λ_path` ≈ 0.70–0.75× (no measurable cost) | ✅ **SURVIVES** (the 10× falsifier never armed; no rollout reduction needed) |

*A prediction that survives is evidence; one that fails is a finding.* P2/P8/P10 are the substantive
findings; in each case the **registered alternative** is what happened, which is precisely why both
were registered.

## 10. Flag provenance (every quantitative result above)

| item | value |
|---|---|
| commits | `1ce02e4` (freeze) · `fed58c5` (D1) · `5a1ebd2` (D3/D4) · `507bbe8` (liveness/ridge seam) |
| base | local `main` `233fd9e`; branch `agent/experiment-engineer/traj-write-objective` (worktree) |
| seeds | **{0,1,2}** on every voting cell and every table above except where "1 seed" is stated |
| sd convention | **sample sd, `ddof=1`; SE = sd/√n**; "clears" ⇔ `mean − 2·SE > 0` |
| families / gym arms | `overload/load1x_shipped` (atoms_per_item=341, n_offer=capacity=budget=6) · `aggregate/base` · `manifold/base` (n_spectator=1) · `manifold/ridge` · `recency/base` (stage_lifetimes, leak=0.06) |
| store band | `addr_dim=4`, `payload_dim=1`, `capacity` per family, `atom_width=0.3`, `atom_init_scale=1.0`, `confine=0.05`, `stage_admission=True`; `d_safe_override=0.58` on `overload` only |
| write | `write_steps=300` (D5/escalation: 900, 1800), `lr=3e-3`, `weight_decay=1e-4`, `sigma_addr=0.25`, `sigma_pay=0.6`, `margin=0.15`, `barrier=0.2`, `barrier_pairs="nn"`, `masked_write=True` |
| read | `dt=0.05`, `gamma_address=0.05`, `gamma_read=0.02`, `address_steps=400`, `read_steps=800`, `traj_stride=8`, `kinetic_mode=newtonian_learned`, `query_sigma=0.15` |
| **Route-1 terms** | `λ_traj`/`λ_path` grid **{0, 0.03, 0.3, 3.0, 30.0}** (+300 on the anchor sweep); `traj_kwargs = {rollout_steps: 60, stride: 6, n_launch: 4}` (⇒ 10 strided points), `traj_gamma = gamma_address = 0.05`, `margin_traj = 0.15`; `path_kwargs = {n_interp: 7}`, `w_tangent = 1.0` |
| D4 | `restrict_index_to_pair` **False** (shipped) for pre-fix rows, **True** for post-fix rows |
| byte ledger | `overload` **478.20×** · `aggregate` 54.56× · `manifold` 52.00× — all `matched=False`, all **architectural** (≥2.20×), ⛔ none quotable as a byte-matched dividend |
| env | main venv reused (protocol §4), **JAX 0.9.0**, no worktree `uv sync`; `chlu.__file__` verified in the worktree |
| suite | **919 passed** (880 at scoping + 39 new), 858 s |
| measured wall-clock | ≈ **2.6 h** of measured runs (budget ≤6 h, hard stop 10 h). Warm per-write cost vs the `endpoint_write` control: `λ_traj` **2.60–3.00×**, `λ_path` **0.70–0.75×**, `traj+path` **2.49×** (endpoint mean 19.3 s, n=3/cell). ⚠ the sub-1× `λ_path` figure is residual-JIT-flattered — the control cells run first in each family; read it as "no measurable cost". The 10× falsifier never armed, so **no declared rollout reduction was needed**. |

## 11. Deliverables & how I verified

**D0 (freeze, `1ce02e4`)** — `chlu/eval/race.py` (the `RaceCell` schema + scorer both routes emit) and
two default-off seams in `clu_system.py`: **(a)** write-objective passthrough
(`CluSystem/build_system(write_objective=…)` → `train_memory_landscape`, unknown keys **raise** so a
mis-spelled coefficient can never masquerade as an inert term); **(b)** store-potential factory hook
(`store_potential_factory` import path + `store_potential_kwargs`) so `ssb-shell-atoms` registers shell
atoms without editing a file it does not own. Both "OFF ⇒ bit-identical" gates are **blocking tests**.
The scorer implements both Head counterweights in code: every excluded cell returned with its reason;
zero-admissible ⇒ `abstain` (no vote); no perturbing anchor ⇒ `under_powered_grid` (no vote); signed
+0 B margin ⇒ `weak_proceed` graded automatically. `gate_summary` carries the literal note
*"arithmetic only; the C2W2 gate is applied by the Hub."*

**D1** — `trajectory_margin_penalty` (triplet margin on a strided damped-Verlet read-out) and
`path_equal_depth_penalty` (equal depth + zero tangent gradient), both behind coefficients defaulting
to **0.0**. ⛔ The coefficient-zero regression gate **PASSES**: both terms sit behind a Python-level
`>0` branch and fold their own sub-key, so at 0 not one extra op is traced and the key stream is
untouched. `_damped_verlet_path` is asserted equal to the shipped `velocity_verlet_step` for
`H=|p|²/2+V(q)`.

**Commands run** (all `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`, cwd worktree):
```
-m pytest tests/test_race_card.py -q --no-cov              -> 17 passed in 3.25s
-m pytest tests/test_clu_system.py -q --no-cov             -> 27 passed in 35.98s
-m pytest tests/test_traj_write_objective.py -q --no-cov   -> 16 passed in 48.11s
-m pytest -q --no-cov                                      -> 919 passed in 858.09s
ruff check <all touched files>                             -> All checks passed!
-m chlu.experiments.exp_traj_write --families overload --seeds 0 1 2        (30 cells)
-m chlu.experiments.exp_traj_write --families aggregate manifold --coeffs 0.3 --seeds 0 1 2  (24 cells)
-m chlu.experiments.exp_trajectory_read --part b --family attention --seed 0  (2561.9 s)
+ scratch harnesses: anchor sweep, grad probe, rollout-vs-read, recency diag, D5/escalation, P4 ridge
```

**Artifacts** under `.claude/outputs/traj-write-objective/`: `PREREG.md` · `race_card_overload.json`,
`race_card_secondary.json` (54 cells, frozen schema) · `race_card_scored.json`, `race_card.md`
(verdicts + every exclusion) · `race_card.png` (3 panels: dividend-vs-coefficient **curve**, the cost
of asking, the four-way trajectory launder) · `d5_escalation.json` · `p4_ridge.json` ·
`recency_fixed.json` · `anchor_*.json`, `grad_probe.json`, `timing.json`.

## 12. Git footprint

Branch `agent/experiment-engineer/traj-write-objective` (worktree `../CHLU-traj-write-objective`),
base local `main` `233fd9e`. **Not pushed, no PR, left for Hub review.**

| commit | what |
|---|---|
| `1ce02e4` | **RACE+SEAM FREEZE** — `chlu/eval/race.py` (new), two `clu_system.py` seams, 23 tests |
| `fed58c5` | D1 — `λ_traj`/`λ_path` terms + the blocking coefficient-zero gate, 10 tests |
| `5a1ebd2` | D3/D4 — `exp_traj_write.py` (new), recency defect fix, mandatory trajectory launder, endpoint-only write loss, CLI hook, 6 tests |
| `507bbe8` | liveness/anchor annotation, ridge-write seam, per-site endpoint losses |

Files: `chlu/eval/race.py` (+754) · `chlu/experiments/exp_traj_write.py` (+413) ·
`chlu/training/train_memory.py` (+199) · `chlu/core/clu_system.py` (+177) ·
`chlu/experiments/exp_memory_gym.py` (+162) · `chlu/cli/experiment_cmd.py` (+73) ·
`chlu/experiments/memory_gym.py` (+42) · `tests/{test_traj_write_objective,test_race_card,test_clu_system}.py` (+695).
**No file outside my §7 ownership list was touched. No conflicts.** `psi_readout.py`, `monitors.py`,
`implicit_grad.py`, `shell_atoms.py`, `exp_trajectory_read.py`, `config.py` were imported/run, never
edited.

## 13. Open questions / risks

1. **The `λ_traj` term never descends its own objective** (penalty exactly constant at every λ). The
   registered rollout (60 steps) is 1/20 of the read's 1200. A follow-up should ask whether a
   **read-length** rollout (400–1200 steps, ~20–60× the write cost) makes the term descend — that is
   the one version of Route 1 this wave did **not** test, and my report should not be read as
   excluding it. Declared explicitly rather than buried: **the null is for `rollout_steps=60`.**
2. **`aggregate`/`manifold` abstain because their write never converges at ANY budget** (0.28–0.29
   endpoint loss, flat in steps). That is a store/expressivity fact, not a Route-1 fact, and it means
   the gate rests on **one** family. The Hub should weigh that.
3. The gym's trajectory ψ is handcrafted (tail-mean); `endpoints == full` is partly tautological for
   it. The learned-ψ version is `phi-particle-head`'s — and per §6 the attention family currently
   leaks.
4. **`path_write@3/30` drives `λ_min` strongly negative.** If C2W3 keeps the path term, it needs a
   conditioning guard.

## Proposed handover updates (for the Hub)

- **§2 architecture:** `chlu/eval/race.py` (new) = the **frozen C2W2 cross-branch surface** (both
  routes emit `RaceCell`; scorer implements the gate arithmetic + both Head counterweights).
  `chlu/experiments/exp_traj_write.py` (new) = Route 1's race-card runner.
- **§3 config/CLI:** `CluSystemConfig` gains `store_potential_factory: str|None = None` and
  `store_potential_kwargs: dict = {}`; `CluSystem`/`build_system` gain `write_objective: dict|None`;
  `write_loss` gains `lambda_traj`/`lambda_path`/`traj_kwargs`/`path_kwargs`/`payload_dim` (**all
  default-off, bit-identical, blocking regression tests**); `GymConfig` gains
  `restrict_index_to_pair: bool = False`. New CLI command **`chlu exp-traj-write`**.
- **§7 Known Issues — ADD:** *"the recency gym family scored a K-way answer against a 2-way chance"* —
  **RESOLVED** by `restrict_index_to_pair` (default off preserves the old numbers). And **ADD (OPEN):**
  *"`AttentionPsi` trajectory reads leak `φ(x)` — `q0_only` ≈ 0.41 vs bar 0.19; no attention-ψ
  trajectory number is quotable store-relative."*
- **Test count:** full suite **880 → 919**.
- **Registry lag:** the four reconciliation items at the top of this report are all C2W2-fresh and are
  in **no** registry.
