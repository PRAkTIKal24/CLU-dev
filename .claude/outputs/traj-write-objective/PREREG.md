# PREREG — `traj-write-objective` (C2W2, Route 1)

**Written before any measured science run.** Base local `main` `233fd9e`; branch
`agent/experiment-engineer/traj-write-objective`; worktree `../CHLU-traj-write-objective`.
Protocol §5 pre-registration rule. *A prediction that survives is evidence; one that fails is a
finding.* Every number below is a **commitment**, with the derivation that produced it.

⭐ **DIAL DECLARATION** — pillar 1 (expressive latents). KPI = `dividend ≡ (full CLU) − (its own
settle-deleted launder)`, same store, same φ, same read budget. Laundering controls: settle-deleted,
same-keys null, blank store, **trajectory launder on every ψ that sees the address block**, plus the
family's strongest **+0 B substitute**. Falsifies: §6 of the task file. Does NOT falsify: dividend ≤ 0
*with liveness passed* (that is the gate's own branch); 17.1× ψ wall-clock; `matched=False` bytes.

---

## 0. What is already frozen (D0, landed before this file was written)

`chlu/eval/race.py` (the cell schema + scorer) and the two `clu_system.py` seams are the **RACE+SEAM
FREEZE** commit. They contain **no science** and no predictions; nothing in §1–§8 below was measured
when that commit landed. This section exists so the freeze cannot be back-dated.

---

## 1. The two objective terms — exact forms (D1)

Both are additive terms in `chlu/training/train_memory.py::write_loss`, each behind its own
coefficient, **both defaulting to `0.0`**, so `λ_traj = λ_path = 0` is the shipped objective
bit-for-bit (the ⛔ coefficient-zero regression gate, D1 — blocking).

### 1.1 `λ_traj` — PRIMARY: a **triplet margin on a strided-trajectory read-out**

Rationale (§A2.1, ratified): today's `write_loss` constrains only **isolated settled endpoints**
(`L_grad`, `L_min` at `z_i`; `L_bar` at midpoints). Nothing in it mentions the *path* the read
actually traverses, so every read that touches the in-between regions loses by construction. The
minimal way to *ask* is to put a discriminative read-out **of the path** into the write loss.

For the item being written, target `z_i = (c_i, a_i)`, launch from the **query manifold** (payload
channel pinned to 0, address jittered exactly as queries are, `sigma_addr`):

```
q(0) = (c_i + ξ, 0),  ξ ~ N(0, sigma_addr² I_d),   p(0) = 0
(q,p)(t+1) = DampedVerlet_γ_traj[V](q,p)(t),  dt = cfg.dt = 0.05, identity mass
m_i = (1/n_pts) Σ_{t ∈ strided} q(t)[payload channels]          # the path read-out
L_traj = mean_ξ relu( margin_traj + |m_i − a_i| − |m_i − a_nn(i)| )
```

`a_nn(i)` = payload of the **nearest** other stored site (from `crowd_targets`; the write is
sequential, so `crowd_targets` is the full live set). The term is `0` when `K < 2` (no competitor
exists) — declared, and it means the term is **structurally inert on the first write of a stream**.

**Registered constants** (chosen before running, from the read's own band, `CluSystemConfig`):
`traj_rollout_steps = 60`, `traj_rollout_stride = 6` (⇒ `n_pts = 10`), `traj_gamma = 0.05`
(= `gamma_address`), `traj_n_launch = 4` jittered launches, `margin_traj = 0.15` (= `write_margin`).

**NOT-RUN alternatives** (named, so the primary is not a post-hoc pick):
(a) path-vs-path contrastive between item `i`'s trajectory and its competitor's (2× rollout cost);
(b) an InfoNCE / MI surrogate on the strided buffer (needs a learned critic *inside* the write loop —
outside this wave's compute budget).

### 1.2 `λ_path` — the path / equal-depth term

Gym §3.5, measured: a multi-target ridge write yields a **saddle** (`λ_min = −0.5946`, spectator
participation `1.000`) because `write_loss` minimises `V` at each target **independently** and
`L_bar` actively *raises* the connecting midpoint. Constrain the connecting path to equal depth with
zero gradient along the tangent:

```
z(s) = (1−s) z_i + s z_j,  s ∈ {1/(S+1), …, S/(S+1)},  S = 7
t̂ = (z_j − z_i)/‖z_j − z_i‖
L_path = mean_s [ (V(z(s)) − ½(V(z_i)+V(z_j)))²  +  w_tan · (∇V(z(s))·t̂)² ]
```

`w_tan = 1.0` (registered). Pairs = each target and its nearest other stored site (`"nn"`, matching
the shipped `barrier_pairs`).

### 1.3 Coefficient grids — **declared here, never chosen after seeing a result** (§1(ii))

| term | grid | span | perturbing anchor (expected) | zero point |
|---|---|---|---|---|
| `λ_traj` | `{0, 0.03, 0.3, 3.0, 30.0}` | 0.03→3.0 = **2 decades** | **30.0** | `0` (regression gate) |
| `λ_path` | `{0, 0.03, 0.3, 3.0, 30.0}` | 0.03→3.0 = **2 decades** | **30.0** | `0` (regression gate) |

**Registered anchor prediction** (this is what makes an inert-everywhere result a *legitimate* ≤0
vote rather than an under-powered grid): at `λ = 30.0` the term **visibly perturbs the write** —
predicted **`final_write_loss` ≥ 3× the `λ=0` value** and/or **strict endpoint recall drops by ≥ 0.20
absolute**. If neither moves at 30.0, the grid is reported as **under-powered**, not as "the term does
nothing", and the top of the grid is extended once to `300.0` before any vote is cast.

---

## 2. Liveness bar (P-A) — the Stage-0 analogue, BLOCKING for D3's verdict

**Test.** Train a fixed probe (the C2W1 ψ family, unmodified) to decode item identity from the
strided read trajectory of a store written at `λ_traj = c`, and compare against the **same probe** on
a store written at `λ_traj = 0`.

**Baselines — both mandatory, and the bar is against the harder one:**
- **capacity-matched** `endpoints = [q₀, q_addr, q*, p*]` (⛔ never `q0_only` — beating a strictly
  smaller read is not evidence);
- **blank-store probe control** (spike §4.3: 31–63 % of the only replicating v0 effect was reproduced
  by a store with nothing in it).

**Registered bar and direction.** The term is **LIVE at coefficient `c`** iff, over 3 seeds,

```
decode_traj(λ_traj = c) − decode_traj(λ_traj = 0)  ≥  +0.05 absolute   (direction: positive)
AND decode_traj(λ_traj = c) > max(endpoints, blank) + 3·SE
```

`0.05` is derived as ~1.5× the largest C2W1 trajectory-channel dispersion (the gym's per-family
dividend SE was ≈ 0.035); anything smaller is inside C2W1's own noise floor.

**Registered prediction: the term is LIVE at `λ_traj ∈ {3.0, 30.0}` and INERT at `{0.03, 0.3}`.**
Reasoning: `L_traj`'s gradient enters through a 60-step damped rollout whose Jacobian decays as
`(1−γ dt)^N ≈ e^{−0.15}` per stride but whose *magnitude* is set by `|m_i − a|` differences of order
`0.1`, so at `λ ≤ 0.3` the term contributes < 1 % of the `L_grad + L_min` scale measured in C2W1
(write loss ~1e-3 at convergence, `L_min` dominating). Confidence: 0.6.

**Counter-prediction registered simultaneously (the alternative that must be able to win):** the term
is inert at **every** coefficient because the write's atom parameterisation cannot make the *path*
discriminative without also moving the endpoint (the atoms that shape the path are the same atoms
that shape the well) — i.e. path information and endpoint information are not independently
addressable at this weight class. If this wins, it **is** the headline and it votes ≤0 in the gate.
Confidence: 0.4.

---

## 3. `λ_min` at the ridge item under `λ_path` — point estimate + range

Today: **`λ_min = −0.5946`**, tangent participation `1.000` (gym §3.5).

Two hypotheses, both registered **before** the run (the `v5-gate` pattern):

- **H-path-A (mine, confidence 0.65).** A tangent-flatness penalty drives the tangent curvature to
  zero **from below** and cannot by itself make it positive; the equal-depth piece removes the
  *slope* but adds no positive curvature. Prediction: `λ_min` rises **monotonically** in `λ_path` and
  **plateaus just below zero** —
  `λ_min(λ_path=0.3) = −0.30 ± 0.15`, `λ_min(3.0) = −0.05 ± 0.05`, `λ_min(30.0) = −0.02 ± 0.03`,
  i.e. **it does NOT strictly clear 0** on the registered grid. Tangent stays the softest mode,
  participation ≥ 0.80.
- **H-path-B (the task's stated acceptance, confidence 0.35).** `λ_min ≥ 0` at some `λ_path` in the
  grid, with the tangent the softest mode — the term overshoots because `L_min`'s transverse pressure
  is applied at the same points.

**Which is falsified how.** H-path-A dies if any grid point gives `λ_min ≥ +0.01` over 3 seeds.
H-path-B dies if `λ_min < 0` at every registered coefficient **including the perturbing anchor**.
⚠ Under H-path-A the task's spectral acceptance criterion **fails** — and that is a *result* about
what a path term can buy, not a task failure; it will be reported as such.

---

## 4. Per-family dividend predictions (D3) — point + range, per arm

Inherited baseline (C2W1): dividend ≈ 0 or negative on **every** family; best `aggregate/tight`
**+0.008 ± 0.035**; the shipped anchor **exactly 0.0000 with `D = 0`**; substitute audit **0-for-4**.
Prop D1/D2 says `D = 0 ⇒ dividend ≤ 0` structurally, and the dividend can only live in `B_i \ Vor_i`
(created by *geometric heterogeneity*). Neither new term creates geometric heterogeneity between the
launch manifold and the Voronoi cell of the keys — that is the basis of every point estimate below.

| family | metric | `endpoint_write` (control) | `traj_write` | `path_write` | `traj+path` |
|---|---|---|---|---|---|
| `overload` (shipped atom budget) | `decode` | **0.000 ± 0.02** | **−0.01 ± 0.05** | **−0.02 ± 0.05** | **−0.03 ± 0.07** |
| `aggregate` | `neg_mae` | **+0.008 ± 0.035** | **+0.00 ± 0.05** | **−0.01 ± 0.05** | **−0.01 ± 0.06** |
| `manifold` | `r2` | **+0.05 ± 0.10** | **+0.08 ± 0.12** | **+0.10 ± 0.12** | **+0.10 ± 0.15** |
| `recency` (only if D4 clears it) | `acc` | **0.00 ± 0.05** | **0.00 ± 0.06** | — | — |

**Registered gate-level prediction: NO family clears 0 beyond 2 SE ⇒ the ≤0 branch fires and B′
activates** (confidence 0.7). The single most likely exception is `manifold`, whose **`echo`
substitute sits at `1.0000` by construction** — so even a clearing `manifold` grades at best a
**weak proceed** (charter §A6), and I register that its `+0 B substitute margin` will be
**negative (≈ −0.9 ± 0.1)**.

**Registered substitute margins** (signed, `full − substitute`): `overload` **−0.05 ± 0.10** (the
audit went 0-for-4); `aggregate` (2-NN IDW) **−0.10 ± 0.10**; `manifold` (echo) **−0.90 ± 0.10**.

---

## 5. Trajectory-launder predictions (mandatory on every ψ that sees the address block)

C2W1 measured, and it **refuted** both the harness's and the spike's leak predictions:
`q0_only = 0.129` vs chance `0.125`; blank `0.148`.

**Registered:** `q0_only = 0.129 ± 0.02`, `blank_store = 0.145 ± 0.03`, both **below** the
`chance + 3 SE` bar ⇒ **the trajectory launder does NOT fire**. `endpoints` (capacity-matched) sits
**within 0.02 of `full`** on every arm — i.e. the trajectory buys ≈ nothing over the four-point
capacity-matched read. Confidence 0.7.
A prediction of a *leak* would need a reason and I have none; if the launder fires, ⛔ **no ψ number
in the report is quotable** until re-run store-relative, and I will say exactly that.

**D6 / spike R-4 (`AttentionPsi`, run BEFORE D3's headline cells).** The untested hypothesis is that a
**pooled** DeepSets read-out dilutes `q₀` to 1 of 150 points while an **attention** ψ can *select* it.
Registered: attention `q0_only = 0.16 ± 0.04` — **higher than DeepSets' 0.129, but still below the
bar** (confidence 0.55); a fired attention launder (`q0_only > 0.20`) would be a genuine finding and
would invalidate every trajectory-ψ number in the wave. I register that outcome as *possible* at 0.30.

---

## 6. D4 — the recency diagnostic: competing hypotheses

Gym measured CLU **0.3019 ± 0.0679** vs its own blank store **0.3065**, both **below** the 2-way
chance of **0.5**. Sub-chance on a binary task is the signature of an **inverted label**, not of a
weak model: `1 − 0.3019 = 0.6981 > 0.5`.

- **H-rec-1 (defect: label inversion / off-by-one, confidence 0.7).** The recency target and the
  `order_aware_launder`'s `newest=True` convention disagree about which of the two nearest keys is
  "more recent" (stream index vs slot index vs re-used slot after eviction). **Prediction after the
  fix: CLU `0.70 ± 0.07`, blank `0.69 ± 0.07`, dividend `≈ 0.00 ± 0.05`.** Note this predicts the
  *defect* is real **and** that fixing it still yields no dividend.
- **H-rec-2 (defect: the pair set is degenerate, confidence 0.2).** `_pairs_within` admits pairs whose
  two keys are closer than `query_sigma`, so the query cannot identify which key it came from and the
  score is chance-with-bias. Prediction after the fix: both arms → `0.50 ± 0.05`.
- **H-rec-3 (not a defect, confidence 0.1).** The family is genuinely sub-chance because the read
  systematically settles into the **older** (deeper, longer-written) well. Signature: the inversion
  survives every harness fix and tracks write order. Then the family stays **out** of the gate.

Whichever wins, the `endpoint_write` control is **re-baselined on the fixed family** so both arms see
the same family, and the pre-fix numbers are reported as the defect's evidence.

---

## 7. D5 — the longer-write curve: the predicted knee

Every sub-shipped-budget gym cell ran the same **300** write steps as the shipped one, and measured
`final write loss` **0.20–0.24**, `λ_min` **−0.21…−1.20** — *"beyond-capacity compression does not
present as graceful read degradation; it presents as write failure."* So pillar (a) was never tested.

Arm: `write_steps ∈ {300, 900, 1800}` at **≥2 sub-shipped atom budgets** (`atoms_per_item ∈ {8, 16}`
vs shipped 32), 3 seeds, reporting the **curve** `decode / λ_min / final_loss` vs write steps (⛔ never
the endpoint without the curve).

**Registered predictions.** Adam on a smooth landscape with `lr = 3e-3`: the residual after `n` steps
falls roughly as `n^{-1}` once past the transient, so `0.22 → 0.07` at 900 and `→ 0.035` at 1800.
- `final_loss`: `0.22 ± 0.03` (300) → **`0.07 ± 0.03`** (900) → **`0.035 ± 0.02`** (1800).
- `λ_min`: `−0.6 ± 0.4` (300) → **`−0.10 ± 0.10`** (900) → **`−0.03 ± 0.05`** (1800); crosses 0 for
  **`atoms_per_item = 16` but NOT for `8`** (confidence 0.55).
- `decode`: **the knee is at ≈ 900 steps** — `0.45 ± 0.15` (300) → **`0.75 ± 0.15`** (900) →
  **`0.80 ± 0.15`** (1800), i.e. most of the gain is bought by 900 and 1800 adds < 0.10.
- **Registered discriminator:** if `decode(1800) − decode(300) ≥ +0.25` at either budget, the gym's
  sub-shipped cells were measuring **write failure**, not capacity — and every "beyond-capacity"
  reading below the shipped budget must be re-labelled. Confidence 0.6.

---

## 8. Compute budget and its falsifier

Budget **≤ 6 h** of measured runs; **hard stop and report at 10 h** (task §3).
- **Registered:** the `λ_traj` write costs **≤ 6×** the endpoint write's per-item wall-clock
  (derivation: `traj_n_launch × traj_rollout_steps = 4 × 60 = 240` extra `∇V` evaluations per write
  step against the shipped ≈ `2 × n_perturb + K = 65` `V` evals + 1 `∇V`, with reverse-mode over a
  60-step scan costing ≈ 3× forward ⇒ ≈ 5.5×).
- **Falsifier:** if measured > **10×**, the rollout is reduced to `traj_rollout_steps = 30`,
  `traj_rollout_stride = 3`, and the reduction is **declared in the report** with the measured
  numbers that forced it.
- ψ is fitted on **precomputed** reads wherever the arm allows (the trajectory ψ costs 17.1× the
  point ψ per training step — §A4.4 accepted this); every arm states whether it did.

---

## 9. Scorecard (filled in after the runs, in the report — one row per registered prediction)

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | `λ_traj = λ_path = 0` ⇒ written `V` bit-identical to `main` | | |
| P2 | `λ_traj` LIVE at `{3.0, 30.0}`, inert at `{0.03, 0.3}` | | |
| P3 | perturbing anchor at `λ = 30.0` (loss ≥3×, or recall −0.20) | | |
| P4 | `λ_min` at the ridge item plateaus **just below 0** (H-path-A) | | |
| P5 | no family clears 0 beyond 2 SE ⇒ B′ branch | | |
| P6 | `manifold` +0 B substitute margin ≈ −0.90 | | |
| P7 | trajectory launder does NOT fire (`q0_only ≈ 0.129`) | | |
| P8 | attention ψ `q0_only ≈ 0.16`, still below the bar | | |
| P9 | recency is a **label-inversion defect**; fixed → `0.70`, dividend ≈ 0 | | |
| P10 | longer-write `decode` knee at ≈ 900 steps; `Δdecode(1800−300) ≥ 0.25` | | |
| P11 | `λ_traj` write ≤ 6× endpoint-write wall-clock | | |
