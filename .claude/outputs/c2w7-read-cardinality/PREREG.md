# PREREG — `c2w7-read-cardinality`: multiplicity-as-counting-code (charter §A21, C2W7)

**Filed 2026-08-05 by `experiment-engineer`, BEFORE `chlu/experiments/exp_tierii_cardinality.py`
scored a single arm and before `chlu/core/multiplicity_read.py` was committed.** Binding.
Supersedes `.claude/outputs/tierii-read-fix/PREREG.md` only where §A21's C2W7 row does (the read's
**cardinality** machinery and the reader class); every other clause of `PREREG-TierII.md` and of
iteration 1's prereg (the `OD_min` metric, the organizer-swap control, the cat-test construction,
the ledger rules, §2.6's claim form, the K1–K5 pre-conditions, the 5-seed / `mean ± 2 SE`
convention, the `ERRATA-TierII.md` corrections E-T1…E-T5) is carried unchanged.

⭐ **What is replaced, and what is not.** Iteration 1's **read latent** (binary-ish occupancy +
`noisy_or` set union) and its **launch head** (successive-suppression, `k` distinct wells, no
cardinality commitment) are replaced. The **family** (`N_a = 32`, `F = 4`, `K = 128`,
`y = Σ_{j∈A} v_j`), the **store vehicle** (factored store, `a = 32`, `m = 8`), the **metric**
(exact-set accuracy at `tol = 0.25 × RMS`), the **control** (the organizer swap) and the
**operating point** (`d = 8`, `d/s ∈ [2.5, 2.9]`, `γ_address = 0.05`, `γ_read = 0.02`, read budget
400 + 800) are unchanged.

---

## 0. ⭐ K0 — THE PRE-CONDITION, ADJUDICATED FIRST (store-free, before the build)

`P(≥ F distinct wells reachable)` from the launch geometry alone, **bar 0.90**, standing rule.
Measured on the **SEEN** split (3 seeds, tuning) for the design decision, and re-reported on the
unseen split by the harness's `k0` stage. ⛔ Re-checked on the **TRAINED** head inside `arms`
(a trained head can collapse K0 — that is what the launch-collapse monitor watches).

| head | `k` | `F_hat` | **K0** | precision | launch-code exact-set (SEEN) |
|---|---|---|---|---|---|
| iteration 1's designed head (`tierii-read-fix`) | 12 | — (none) | **1.0000** | 0.274 | 0.0026 |
| multiplicity head, `card_b = 0.0` | 12 | 4.03 | 0.789 ⛔ | 0.589 | 0.1432 |
| multiplicity head, `card_b = 0.3` | 12 | 4.32 | 0.880 ⛔ | 0.569 | 0.1354 |
| ⭐ **multiplicity head, `card_b = 0.5` (REGISTERED)** | **12** | **4.51** | ⭐ **0.9115** | **0.560** | **0.1224** |
| multiplicity head, `card_a = 1.0, card_b = 0.3` | 12 | 5.00 | 0.956 | 0.536 | 0.0781 |

⭐ **REGISTERED FINDING, BEFORE THE RUN — `K0`'s bar and the `F`-commitment are in DIRECT TENSION
BY CONSTRUCTION.** `K0` demands `≥ F` distinct wells be *reachable*; a cardinality commitment
demands *exactly* `F_hat`. Every query with `F_hat < F` fails `K0` automatically. The registered
cell buys `K0 = 0.9115 ≥ 0.90` and **pays 0.021 of launch-code exact-set** for it (0.143 → 0.122).
⛔ This is a property of the K0 statistic, not a defect of either mechanism, and it means **K0 can
no longer be read as "the read is healthy"** — it is a *reachability* statement only.

**The decoder study that fixed the head's stage (i)** (SEEN split, 3 seeds, store-free; the
measurement that forced deviation D12):

| set-recovery rule on the same noisy code | exact-set (SEEN) |
|---|---|
| top-`F` of the raw overlaps (iteration 1's ranking) | 0.023 |
| OMP / greedy matching pursuit (the first design tried) | **0.008** |
| ⭐ **non-negative ISTA, 200 unrolled steps (REGISTERED)** | ⭐ **0.19** |
| exhaustive `C(32,4)` decoder — ⛔ **OUT-OF-CLASS reference line, never a bar** | 0.72 |

⛔ Greedy pursuit **provably cannot** do this job at this cell: the code's mutual coherence is
`≈ 1/√d = 0.35` while OMP's recovery guarantee needs `< 1/(2F−1) = 0.14`. Non-negativity is not a
trick — every true coefficient is `+1/√F` by the construction of `φ`.

---

## 1. ⛔ REGISTERED DEVIATIONS (argued, never silent — D1–D11 carried from cat-test/iteration 1)

| # | prior registration says | I run | the measurement that forced it |
|---|---|---|---|
| **D12** | the launch head selects wells by (suppressed) overlap ranking | **an unrolled non-negative ISTA over the frozen codes** (200 steps, `lax.scan`, learned `λ`, `η`, per-well threshold `bias`) | the table above: 0.008 (OMP) / 0.023 (top-F) vs **0.19** (NN-ISTA). At coherence 0.35 with `F = 4`, greedy recovery is outside its own guarantee; the information the exhaustive decoder reaches (0.72) is unreachable greedily. **Head params stay tiny and ledgered** (`k·d + N_a + 14 = 142`, 568 B, vs the store's 73 728 B) and the head is **identical on every arm including the launder**. |
| **D13** | reader class = iteration 1's 5 (D8) | **7**: the 5 carried + **`count_table`** (72 params) + **`count_identity`** (**0** params) | deliverable 6, filed as `AMENDMENT-C2W7`. Multiplicity/overlap weighting changes what a reader consumes (a weighted counting code, not a set). ⭐ **`count_identity` is forced by a measurement, not by taste** — see §6. |
| **D14** | `conf_b = 0.25`, `conf_w = 0.05` (iteration 1) | **`conf_b = 0.10`, `conf_w = 0.15`** | at a multiplicity head a *hard* confidence gate deletes an entire well from the counting code, and the all-or-nothing metric then scores the query 0 **even when the set is right**. The registered gate is a gentle importance weight; the hard/soft difference is measured in the weight-mode ablation. |
| **D15** | 5 seeds, `n_unseen = 512` | 5 seeds, **`n_unseen = 256`** | compute (carried from iteration 1's D11). 1 280 scored queries/arm; grain 1/1280 = 7.8e-4. |

**Carried unchanged from iteration 1:** D6 (`d = 8`), D7 (**one** query-noise draw per query),
D9 (`k = 12`), D10 (`payload_radius = 0.5`), and the measured constraints `a = 32`, `m = 8`.

**ψ pinning (Advisor amendment A2).** ⛔ **This vehicle contains no ψ**: `factored_store`,
`multiwell_read`, `multiplicity_read` and `null_arms` contain zero references to
`chlu/core/psi_readout.py` (verified by grep at filing). Every arm therefore uses the **shipped**
path, uniformly, with **no mid-task hot-swap**; ψ's marginal value is **NOT** measured here and
belongs to the CSF3 run-1/run-2 ablation.

**depth_ratio (Hub pre-registration, carried).** The claim cell KEEPS the registered `≥ 3×` depth
heterogeneity. `depth_ratio = 1` runs as a **registered diagnostic axis, never a claim cell**.

---

## 2. THE PROTOCOL BEING REGISTERED (what is built — the five deliverables)

**(1) Multiplicity-as-counting-code.** `x = NN-ISTA(c̃/R)` (evidence) → `F_hat = clip(a·IPR(x)+b, 1, 8)`
with `IPR = (Σx)²/Σx²` (**exactly `F` for a flat `F`-sparse vector**) → a **soft top-`F_hat`
commitment** `mask_j = σ((x_j − θ_q)/ε)` at a rank-interpolated per-query threshold → multiplicity
`n_j = k·mask_j/Σmask` (**`Σ_j n_j = k` exactly**) → a **stick-breaking allocation**
`β_ij = relu(min(i+1, c_j+n_j) − max(i, c_j))` which is a **partition of unity by construction**
(no `argmax`, no sampling, no normalisation). ⛔ The commitment is **query-driven**: `IPR` is a
functional of the query's own coefficient profile and never of well depth — which is exactly what
iteration 1 §4.1 proved absent (`top-F(π) == A(x)` was 0.000 at every sharpening `β`).

**(2) Overlap-as-importance weighting.** The read returns `m_j = F_hat · cnt_j / Σ_l cnt_l` where
`cnt_j = Σ_i a_i π_ij` and `a_i = w_i · conf_i` (descent × overlap). ⛔ Binary occupancy +
`noisy_or` is **replaced** as the answer (it is retained as the carried `pi` latent so the
iteration-1 readers stay in class). All four weight modes (`none`/`descent`/`overlap`/`both`) and
the `noisy_or` aggregation are computed **from one read pass** and reported for every arm.

**(3) The batch-level anti-collapse regularizer.** `λ · (log N_a − H(p̄))` on the **MARGINAL**
usage `p̄_j = mean_batch m_j` (normalised). ⛔ **Per-query concentration is CONFIDENCE and is never
penalised.** Ships **OFF** (`lambda_anticollapse = 0.0`, doctrine §3.3: monitored first,
regularized second); the ON arm runs as a labelled second state and **both are reported**.

**(4) The launch-collapse monitor** — `chlu/core/monitors.py` row **#15** (`launch_collapse`,
severity class I, additive; `monitors.py` ceded to C2W7 on the §10 record). Statistic
`S_marg = exp(H(p̄))` (the effective number of wells used **across the batch**). **Band
`S_marg ≥ 0.5·N_a = 16`; trips below.** Designed negative: an input-independent allocation
(`S_marg → F`) and a one-well head (`S_marg → 1`), both pytest-asserted.

**(5) Launch-only launders recomputed LIVE** (amendment A1, strong form): each cell's **own trained
head**, landscape deleted (zero settle), written payload table retained, scored through the **same
re-registered reader class at the same `k`**, with the whole cardinality mechanism. ⛔ `0.272` and
`0.695` appear only as **labelled out-of-class reference lines with their `(d, draws)` noise
model**; the harness recomputes the ceiling at this cell (`k0` stage, `ceiling_out_of_class`).

**Staging (G3).** Store WRITTEN → launch head TRAINED on the written store → store ORGANIZED
through the settle with the head frozen → read. The swap's null gets the **same frozen head**.

---

## 3. ⭐ THE FALSIFIERS (re-registered; sign, threshold, seeds — 5 seeds, `mean ± 2 SE`)

| # | statement | fires iff | clears iff |
|---|---|---|---|
| **K0** | the launch cannot address `F` wells | `P(≥F distinct) < 0.90` | ✅ adjudicated **PASS at 0.9115** on SEEN before the build; re-reported unseen and **re-checked on the trained head** |
| **R1** | ⛔ the read still cannot express a multi-well answer | exact-set **occupancy** of the read's asserted set `{j : m_j ≥ 0.5}` `≤ 0.001` | **`mean − 2 SE > 0.02`** |
| **R2** | the settle destroys addressability | settled distinct wells `<` launched distinct wells `− 2 SE` | settled ≥ launched |
| **G1** | ⛔ **the store adds nothing over its own learned launches** (A1 strong) | `read − live launder` (worst reader) `mean − 2 SE ≤ 0` | `> 0` |
| **G2** | the soft signal is not why the head trains | `‖∇_hard‖/‖∇_soft‖ > 1e-3` | `< 1e-3` |
| **G3** | staging is unnecessary | blank-store gradient `> 1e-6` while written `> 1e-2` | staged ordering required |
| **G4** | `k` is not capacity | doubling `k` at matched bytes does not move the score | it does ⇒ `k` MUST be ledgered |
| **M15** | the learned head collapsed marginally | `S_marg < 0.5 N_a = 16` | in band |
| **S_eff** | allocation collapse | `S_eff ∉ [8, 16]` ⇒ **COLLAPSED** | in band |
| **F1/F2/F5** | the organizer swap (carried verbatim from `PREREG-TierII.md`, with E-T5's imitability caveat) | — | **only adjudicated if the gate fires** |

## 4. ⭐ MY NUMERIC PREDICTIONS (the w14 rule — committed before the harness ran)

| # | quantity | **point** | band |
|---|---|---|---|
| P1 | physics arm, best reader, unseen exact-set | **0.10** | [0.02, 0.25] |
| P2 | live launch-only launder `L_a`, best reader | **0.12** | [0.03, 0.28] |
| P3 | ⭐ `G1_min` = read − launder (worst reader) | **−0.02** | [−0.12, +0.03] |
| P4 | ⭐ **R1** exact-set occupancy of `{j : m_j ≥ 0.5}` (physics, unseen) | **0.12** | [0.02, 0.30] |
| P5 | `F_hat` mean (physics, unseen) | **4.5** | [4.0, 5.0] |
| P6 | fraction of queries whose asserted set has **exactly** `F` members | **0.55** | [0.30, 0.80] |
| P7 | settled distinct wells (raw) | **4.4** | [3.5, 5.5] |
| P8 | `S_eff` | **16.0** | [8, 16] |
| P9 | `S_marg` (marginal usage perplexity, physics) | **28** | [20, 32] — monitor **does not trip** |
| P10 | `‖∇_hard‖/‖∇_soft‖` (G2) | **0** exactly | [0, 1e-8] |
| P11 | G1's designed negative: 0-step read vs launder | **bit-identical** | max abs diff 0.0 |
| P12 | ⭐ `count_identity` − `count_table` on the SAME latent | **+0.10** | [+0.02, +0.25] |
| P13 | dedupe verb LIVE: \|exact(`sum`) − exact(`noisy_or`)\| | **> 0.05** | iteration 1 measured the two **bit-identical** |
| P14 | weight mode `none` − `both` | **+0.02** | [−0.02, +0.10] |
| P15 | K0 re-checked on the **trained** head (unseen) | **0.90** | [0.80, 1.00] |
| P16 | anti-collapse ON − OFF: `ΔS_marg` | **+1.0** | [−1, +6]; \|Δscore\| < 0.03 |
| P17 | the organizer swap | ⛔ **NOT RUN** (the gate is predicted to fail on G1) | — |

**Pre-registered priors on the wave verdict:** `P(R1 clears 0.02) = 0.70` · `P(G1 clears) = 0.15` ·
`P(the gate fires ⇒ the swap runs) = 0.12`.
**Reason (stated before measuring):** the family's set information lives *entirely* in the query
code; the landscape is written identically for every query, so the settle can only *re-quantise* an
address the launch already computed. The A1-strong launder retains the payload table and gets the
whole cardinality mechanism, so anything the head achieves, the launder achieves. ⭐ **A stronger
head therefore makes G1 harder to clear, by construction — and that is the honest test.**

**⛔ What I will report as a negative, honestly:** if R1 clears and G1 fires, the finding is
*"the read's expressivity is repaired and the expressivity is entirely launch-side"* — a
read-protocol result and a vehicle result, **not** a physics verdict and **not** a family verdict.

## 5. ⚖ THE GATE (mechanical, adjudicated on the claim cell, 5 seeds)

`SWAP_RUNS` iff **all four**: `R1 mean − 2 SE > 0.02` ∧ `G1_min mean − 2 SE > 0` ∧
`S_eff ∈ [8, 16]` ∧ the launch-collapse monitor did not trip on any claim seed.
Implemented in `exp_tierii_cardinality.adjudicate_gate`; ⛔ **reported, never interpreted by me —
the tier-ii verdict is the ADVISOR's, against raw artifacts.**

## 6. ⛔ READER-CLASS RE-REGISTRATION (deliverable 6 — BLOCKING, filed before any run)

Also filed as a dated `AMENDMENT-C2W7` block appended to
`.claude/outputs/orgdiv-prereg/ERRATA-TierII.md` (⛔ the prereg itself is never edited).

| reader | consumes | fitted params (`d=8, m=8`) | quantises? |
|---|---|---|---|
| `sum_linear` | `z` (continuous settled states) | 136 | no |
| `well_table` | `argmax(z)` | 72 | **YES** (kept deliberately) |
| `knn` | canonicalised `z` | 0 | no |
| `mlp` | `z` | 108 | no |
| `soft_well_table` (D8) | `π` (noisy-or soft occupancy) | 72 | no |
| ⭐ **`count_table`** (NEW) | `m` (weighted counting code) | 72 | no |
| ⭐ **`count_identity`** (NEW) | `m` | **0** | no |

⛔ **All seven are `< N_a·m = 256`** (SP-1's storeless bound), ≥ 4 architectures, ≥ 2
non-quantising twins. **Frozen before the first arm ran.**

⭐ **Why `count_identity` exists — a MEASUREMENT, filed before the arms ran (SEEN split, 3 seeds).**
Every other member is fitted by **least squares** while the metric is a **thresholded** exact-set
accuracy. On the launch counting code: the set is exactly right on **18.0 %** of queries and, on
those, the identity residual is **0.006** against `tol = 0.234`. The least-squares fit is dominated
by the other 82 %, shrinks its gain to `diag(W) ≈ 0.40`, and drives the residual on the **good**
queries to **0.537 > tol**. Measured unseen scores at the same latent: **`count_table` (72 params)
0.000 / 0.000 / 0.004** vs **`count_identity` (0 params) 0.172 / 0.227 / 0.168**. A 2-parameter
gain+bias reader is shrunk just as hard (`a ≈ 0.5`) and also scores 0.000.
⛔ **This re-scopes iteration 1 §13.3's question**: the thing that zeroes arms is not the
**capacity cap** but the **fitting criterion**. `count_identity` adds **zero** fitted parameters
(the `knn` precedent) and is applied identically to physics, null and launder.

## 7. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)
1. `k = 16` / `k = 24` as scored arms (`k = 24` appears only as guard 4's capacity probe).
2. A ψ A/B (this vehicle contains no ψ; §1).
3. N2 / N3′ / N4 / N5 as swap arms — the swap, if it runs, is against **N1′** only.
4. The γ axis — one claim cell at `γ_address = 0.05`, `γ_read = 0.02`.
5. `d ∈ {4, 16, 24}` as scored cells.
6. The consolidate/trash stage (iteration 1's (d)) — measured there, not re-run here; C2W8 owns it.
7. The organizer-swap **robustness arm** (N1′ with the head re-fitted) — funded only if the primary
   swap runs and is non-vacuous.

## 8. Provenance
Branch `c2w7-read-cardinality`, worktree `../CHLU-c2w7`, base local `main @ 104ca19`.
**Main venv reused** (protocol §4, w6 lesson): **jax 0.9.0, equinox 0.13.4, float32**. Seeds 0–4.
`n_wells = 32`, `f_subset = 4`, `n_items = 128`, `n_unseen = 256`, `atoms_per_well = 32`,
`payload_dim = 8`, `addr_dim = 8`, `payload_radius = 0.5`, `s_measured = 0.2879`, `target_ds = 2.7`,
`depth_ratio = 3.0`, `γ_address = 0.05`, `γ_read = 0.02`, read budget **400 + 800** Verlet steps,
`dt = 0.05`, `kinetic_mode = newtonian_learned`, `query_sigma = 0.15` (**one draw per query**),
`k_particles = 12`, head training budget **60 steps @ 3e-2**, settle **60 + 120** (⚠ reduced,
declared beside every number), organizer **60 steps @ 3e-3 through the settle**.
⛔ `@jax.checkpoint` on every settle body differentiated through (the silent exit-0 OOM hazard).
