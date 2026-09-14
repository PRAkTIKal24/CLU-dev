# PREREG — `tierii-read-fix`: the multi-well read protocol (charter §A20.3(b)–(f))

**Filed 2026-08-01 by `experiment-engineer`, BEFORE `chlu/core/multiwell_read.py` existed and
before any scoring harness ran.** Binding. Supersedes `PREREG-TierII.md` only where §A20 does
(the read protocol); every other clause of `PREREG-TierII.md` (the `OD_min` metric, the reader
class, the organizer-swap control, the ledger rules, §2.6's claim form, the K1–K5 pre-conditions,
the 5-seed / `mean ± 2 SE` statistics convention) is carried unchanged.

⭐ **What is replaced, and what is not.** The `P = 4` frozen-offset occupancy read is REFUTED at
`P = 4` (§A20.1) and is replaced. The *family* (`N_a = 32`, `F = 4`, `K = 128`, `y = Σ_{j∈A} v_j`),
the *store vehicle* (the factored store at `a = 32`, `m = 8`), the *metric* and the *control*
(the organizer swap) are unchanged.

---

## 0. ⭐ K0 — THE PRE-CONDITION, ADJUDICATED FIRST (store-free, before the build)

**K0 (new, standing, §A20.3 DIAL):** `P(≥ F distinct wells reachable)` from the launch geometry
alone, store-free. **Bar: ≥ 0.90; below it, redesign before running anything.**
Measured on the rule-4-valid unseen split, 5 seeds, 2 560 queries
(`.claude/scratch/tierii-read-fix/k0_design{,2,3}.py`; artifacts `k0_design*.json`):

| protocol | `d` | `k` | distinct wells | **K0 = P(≥F)** | precision | **K0b = P(A ⊆ occ)** | verdict |
|---|---|---|---|---|---|---|---|
| **shipped, REFUTED** (`P=4`, frozen offsets) | 4 | 4 | **2.194** | **0.0449** | 0.408 | 0.0000 | ⛔ fails (instrument reproduces `orgdiv-null-arms` §3: 2.202 / 0.050 / 0.4106) |
| head, designed init | 4 | 12 | 7.68 | 1.000 | 0.296 | 0.050 | K0 ok, K0b starved |
| **head, designed init (REGISTERED)** | **8** | **12** | **11.94** | **1.000** | **0.275** | **0.391** | ✅ **K0 passes** |
| head, designed init | 8 | 16 | 15.91 | 1.000 | 0.221 | 0.581 | (declared compute-dial point) |

**⛔ K0 is adjudicated PASS at the registered operating point, and the adjudication forced one
registered deviation** (D6, argued in §1).

**The launch-information ceiling under the new launch model** (out-of-class combinatorial decoder
over all `C(32,4) = 35 960` set codes on the *same* noisy code; declared OUT-OF-CLASS, never an arm
bar, SP-1 precedent): `d=4`: **0.072** · `d=8`: **0.695** · `d=16`: **0.982** · `d=24`: **0.992**.
The rule-2 / K4 leak (OLS `R²` of `y` on the set code, unseen): `d=4`: 0.093 · **`d=8`: 0.159** ·
`d=16`: 0.325 · `d=24`: 0.654.

---

## 1. ⛔ REGISTERED DEVIATIONS (argued, never silent — the cat-test D1–D5 precedent)

| # | prereg says | I run | the measurement that forced it |
|---|---|---|---|
| **D6** | `d = 4` (registered, SP-2's swept axis) | **`d = 8`** | ⭐ At `d = 4` the **launch-information ceiling is 0.072** under a single query-noise draw — `1.4×` the arm bar of `chance + 0.05 = 0.0507`. No read, however good, has room to clear the bar by `> 0.05` there: the family is information-starved at `d = 4`, which is *why* the arms all read 0. At `d = 8` the ceiling is **0.695** while the query-only leak stays at `R² = 0.159` (`d = 16` would raise it to 0.325, `d = 24` to 0.654). `d = 8` is the smallest `d` whose ceiling exceeds the bar by an order of magnitude. ⚠ It is also, independently, where `orgdiv-cat-test` §7.2 measured **peak** occupancy precision (0.417). `d` is `SP-2`'s explicitly swept axis, so this is a move *along* a registered axis, not off one. **Every arm gets the same `d`.** |
| **D7** | launch noise: `σ_q` i.i.d. **per particle** (`P` independent noisy views) | **ONE draw per query**: `c̃ = φ(x) + σ_q ξ`, all `k` launches derived from `c̃` | ⛔ A `k`-particle head fed `k` i.i.d. noisy views could average the query noise away (`σ/√k`) and buy its result from the compute dial rather than from the store. Registered here as the conservative choice: the new protocol sees **strictly less launch information than the refuted one** (that is why the ceiling on my launch is 0.072 at `d=4` where `orgdiv-null-arms` measured 0.272 on the `P=4` mean). ⛔ The 0.272 figure is therefore a **reference line from a different, easier launch model** and appears in the report only as such — never as a bar. |
| **D8** | reader class = 4 members | **5**: the 4 shipped + **`soft_well_table`** | The shipped `well_table` reader hard-assigns (`occupancy`, an `argmax`) — it *is* the quantisation step §A20.3(b) forbids. `soft_well_table` (`ŷ = W·(Π @ payload_table) + b`, `Π` the soft occupancy, **72 fitted params, identical to `well_table`'s**, `< N_a·m = 256`) is its non-quantising twin. The hard reader is **kept in the class** so `OD_min` is scored over both. |
| **D9** | `k = P = 4` | **`k = 12`** | K0. `k` is on the byte ledger and is **matched across every arm** (guard 4). `k = 16` is declared as the compute-dial point and is a NOT-RUN unless the budget allows a labelled single-seed diagnostic. |

⛔ **`ψ` version pinning (Advisor amendment A2).** `psi-payload-residual` has **NOT** landed at filing
time (no `.claude/outputs/psi-payload-residual.md`). **Every cell in this task uses the SHIPPED ψ,
uniformly across ALL arms**, and there is **no mid-task hot-swap**: if the fix lands mid-run, this
task still reports shipped-ψ numbers and says so. ψ's marginal value is not measured here.

---

## 2. THE PROTOCOL BEING REGISTERED (what is built)

**(b) the multi-well read.** `k` particles; each settles under the store's own damped Verlet
dynamics; the latent is `z = (k, dim)` **continuous settled states** plus a **soft** occupancy
`Π ∈ R^{N_a}` against the *consolidated* well table. No `argmax` anywhere in the latent's
construction. Aggregation over particles is **`noisy_or` / max-pool** (a SET union), not a sum —
duplicated particles in one well contribute once (the "dedupe/evolve-unique" verb).

**(c) the k-particle learned-launch head.** `s_j = w⟨ĉ, e_j⟩ + b_j`; slot `i`:
`α^(i) = softmax((s − κ Σ_{l<i} α^(l))/τ)`, `q₀_i = Σ_j α^(i)_j u_j + ρR(ĉ − Σ_j α^(i)_j e_j)`.
**Overlap-as-confidence:** `conf_i = max_j α^(i)_j` drives per-particle **inertial mass**
(`mass_override`, Prop-6 per-address masses), **friction** `γ_i`, and **learned `p₀`** — an
unconfident particle is heavy/over-damped and *cannot reach the well bottom*, so its payload block
stays near 0 and it contributes nothing to the sum. ⭐ **This is where the continuous launch
coordinate survives**: the launch position is near-discrete by design (it must address a well), and
the continuous code is carried by the particle attributes and by the soft occupancy.

**(d) consolidate-to-budget + trash-region pruning.** The store's *actual* minima are found by
probe descent, merged at `merge_radius = 0.5 sep` (mechanical), ranked by depth, truncated to the
designed budget; sub-threshold/spurious wells are routed to the **trash region `γ_φ(q)`**
(`chlu.core.friction_field`, built in C1 and never used — first use). Spurious wells are **trashed,
never merged into meaningful ones**. `S_eff = K·F/#(wells ever occupied)`, band `[S/2, S] = [8, 16]`;
outside the band the run is labelled **COLLAPSED**, not null.

**(e) learned `p₀`** — a registered lever with an on/off ablation.

---

## 3. ⭐ THE FALSIFIERS (re-registered; sign, threshold, seeds — 5 seeds, `mean ± 2 SE`)

| # | statement | fires iff | clears iff |
|---|---|---|---|
| **K0** | the launch cannot address `F` wells | `P(≥F distinct) < 0.90` | ✅ **already adjudicated PASS (1.000)** |
| **R1** | ⛔ **the read still cannot express a multi-well answer** — the §A20 refutation is NOT repaired | exact-set **occupancy** (occupied set `== A(x)`) `≤ 0.001` on the settled read (the refuted protocol measured **0.0000/2560**) | `> 0.02` |
| **R2** | ⛔ **the settle destroys addressability** (the refuted protocol's 2.20 → 1.70) | settled distinct wells `<` launched distinct wells `− 2 SE` | settled ≥ launched |
| **F1** | the organizer swap shows no margin (**the standing bar, unchanged**) | `OD_min + 2SE < +0.05` | `OD_min − 2SE > +0.05` **and** `OD(R) > 0` on ≥ 4 of 5 readers; `\|OD_min\| ≤ 0.05` ⇒ **TIE** |
| **F2** | the margin is the reader's | `max_R OD − 2SE > +0.05` while `min_R OD + 2SE < 0` | — |
| **G1** | ⛔ **the store adds nothing over its own launches** (Advisor amendment A1) | full read `−` **live-recomputed** launch-only launder `≤ 0` | `> 0` |
| **G2** | the soft-occupancy signal is not the reason placement trains | `‖∇_hard‖ / ‖∇_soft‖ > 1e-3` | `< 1e-3` |
| **G3** | staging is unnecessary | joint-from-flat gradient norm `> 1e-6` | `< 1e-6` while staged is `> 1e-2` |
| **G4** | `k` is not a capacity | doubling `k` at fixed bytes does **not** raise the score | it does (⇒ `k` MUST be ledgered) |
| **F5** | imitability (carried) | fitted static-geometric null reproduces the physics assignment on ≥ 99 % | — ⚠ carry `orgdiv-null-arms` §5's 0.89-vs-0.26 caveat |
| **S_eff** | allocation collapse | `S_eff ∉ [8, 16]` ⇒ run labelled **COLLAPSED** | in band |

## 4. ⭐ MY NUMERIC PREDICTIONS (the w14 rule — committed before measuring)

| # | quantity | **point** | band |
|---|---|---|---|
| P1 | physics arm, best reader, unseen exact-set | **0.12** | [0.02, 0.35] |
| P2 | organizer-swap null (N1′, gradient-placed, static read), best reader | **0.10** | [0.02, 0.35] |
| P3 | **`OD_min`** | **−0.01** | [−0.10, +0.05] |
| P4 | live launch-only launder `L_a` (landscape deleted, payload table retained) | **0.09** | [0.02, 0.25] |
| P5 | full read `−` `L_a` (**G1's statistic**) | **+0.02** | [−0.05, +0.15] |
| P6 | ⭐ exact-set **occupancy** of the settled read (**R1**) | **0.06** | [0.01, 0.25] |
| P7 | settled distinct wells (launched = 11.94) | **10.0** | [6, 12] |
| P8 | occupancy precision, settled (refuted protocol: 0.297 settled / 0.406 launch) | **0.28** | [0.15, 0.45] |
| P9 | `S_eff` | **12** | [8, 22] |
| P10 | `‖∇_hard‖/‖∇_soft‖` (G2) | **0** (exactly, by construction) | [0, 1e-8] |
| P11 | learned-`p₀` lever, Δ score (on − off) | **+0.01** | [−0.03, +0.06] |

**Pre-registered priors on the wave verdict:** `P(F1 clears) = 0.15` · `P(TIE, |OD_min| ≤ 0.05) = 0.60`
· `P(physics loses by > 0.05) = 0.25`. **Reason:** the family's set information lives entirely in
the launch (every well is written on every query; `A(x)` selects which `F` to sum), so the organizer
can only help through *basin geometry as a fixed decoder of the launch point*. That is a real but
narrow channel, and the null arm optimises the same parameters through a static softmax.

**⛔ What would make me report a null honestly:** R1 firing (occupancy still cannot express the
answer) is a **read-protocol result** and is reported as such — it does not become a family or a
physics verdict. F1 firing after R1 clears IS a tier-ii datum.

## 5. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)
1. **`k = 16`** as a scored arm (compute; the K0 table's `k=16` row is store-free geometry only).
2. **A ψ A/B.** Shipped ψ everywhere (§1).
3. **N2/N4/N5** as organizer-swap arms — the swap is run against **N1′** (the strongest organizer
   in `orgdiv-null-arms`: it fits 100 % of its own training items) and **N3′** (F5's static-geometric
   rule). N2/N4/N5 were measured at `null* = 0.00117` under the refuted read and are not re-run.
4. **The γ axis** (`γ_address ∈ {0.02, 0.05, 0.2}`) — one claim cell at `γ = 0.05`, as registered.
5. **`d ∈ {16, 24}`** as scored cells (K0 geometry only; the leak grows).

## 6. Provenance
Branch `agent/experiment-engineer/tierii-read-fix`, worktree `../CHLU-tierii-read-fix`, base local
`main @ 9b2d4db`. Main venv reused (protocol §4 w6 lesson): **jax 0.9.0, equinox 0.13.4,
numpy 2.4.1, float32**. Seeds 0–4. `atoms_per_well = 32`, `payload_dim = 8`, `s_measured = 0.318`,
`target_ds = 2.7`, `γ_address = 0.05`, `γ_read = 0.02`, read budget 400 + 800 Verlet steps,
`dt = 0.05`, `kinetic_mode = newtonian_learned`, `query_sigma = 0.15`, `soft_cert_B = 0.542`.
