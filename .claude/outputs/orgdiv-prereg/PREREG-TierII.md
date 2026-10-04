# PREREG — TIER ii: THE ORGANIZATION DIVIDEND AND THE CAT TEST

**Filed 2026-07-31 by `physics-theorist` (C2W4 task `orgdiv-prereg`, charter ADDENDUM 3 §A15 task 6,
specifying §A13 tier ii). BINDING on C2W5 unless the Head/Advisor amends it.**
⛔ **Nothing in this document is a result.** No tier-ii number has been measured. Every number below is
either (i) a derivation, (ii) a theory check on a 1-/2-D toy (`.claude/scratch/orgdiv-prereg/`,
listed in §11), or (iii) a **pre-registered prediction** that C2W5 will confirm or refute.

---

# 0. THE METRIC, IN THE FIRST SCREEN

> ## The organization dividend
> Let a **store family** be a pair `(O, θ)`: an **organizer** `O` that maps a written stream of items to
> stored parameters `θ`, and an induced **query→latent map** `z_θ: x ↦ z ∈ R^n` (what a reader sees).
> Let `R` be a **reader class** — a *fixed, byte-ledgered* set of decoders fitted **only on the SEEN
> split**. For a held-out query set `Q_unseen`,
>
> ### `OD(R) ≡ score( R ∘ z_{O_phys} ) − score( R ∘ z_{O_null} )` , evaluated on `Q_unseen`
> ### **primary statistic: `OD_min ≡ min_{R ∈ readers} OD(R)`** (the margin must survive the reader class)
>
> **The control is the ORGANIZER SWAP**: `O_null` is a **matched-capacity non-physics organizer**
> (§4), at **identical φ (φ-bytes ledgered on every arm), identical bytes, identical capacity,
> identical query set, identical reader class, identical launch protocol.** Only the organizer varies.
>
> ⭐ **Table-like inference reads are EXPLICITLY PERMITTED ON BOTH ARMS, and this is part of the
> metric's definition.** A CLU that organizes during writing and then reads by nearest-well assignment
> at inference is *in scope and unpenalised*; reducing the read to a table after training is a
> **computational win, not a defeat** (§A14.1). ⛔ **No rollout-vs-table advantage may ever be scored
> as an organization advantage** — if an arm's read is a rollout, the *other* arm's read is given the
> same per-query compute budget or the comparison is void.
> ⛔ **The settle-deleted / matched-bytes launder is TIER i's control and is the WRONG control here.**
> It is still *reported* (as an inherited diagnostic), never as tier ii's evidence.

**Why (b) — the reader-class sweep — is the registered definition, and the others are not.**
The task offered three operational definitions of "a better organization".
- **(a) generalisation to unseen combinations at a fixed reader** — adopted as the **unit of scoring**
  (that is what `score(·)` measures) but **rejected as the definition**, because a single reader can be
  co-adapted to one arm's geometry (FB4's lesson: our designed instruments measure our constructions
  back at us).
- ⭐ **(b) the reader-class sweep — ADOPTED as the definition.** `OD_min > 0` says the arrangement is
  better *for every decoder in a declared class*, which is a property of the arrangement rather than of
  a matched reader. It is also the only version that a referee cannot answer with "you tuned the
  read-out". This is the Hub's steer and I concur, for the additional reason in §7: I can prove that
  at the designed operating point the two arms' latent maps are **identical functions**, so any
  single-reader margin there would have to be an artifact — a reader-class sweep makes that
  impossible to miss.
- **(c) intrinsic geometric measures** (co-activation aligned to latent factors) — **admitted as
  corroboration only, never as evidence, and never alone** (§A4.5: wells are not named semantically).
  Its admissible claim form is fixed verbatim in §2.6.

---

# 1. WHAT MUST BE MATCHED, EXACTLY (the ledger)

| quantity | rule |
|---|---|
| **φ** | **identical instance, frozen, on every arm**; `phi_bytes` ledgered on every arm (`PhiMismatchError` precedent reused verbatim). ⭐ Freezing φ is deliberate — see §6 rule 1: it **discharges the `∂q*/∂q₀ = 0` constraint by construction** instead of fighting it. |
| **bytes** | two-sided ledger, corrected byte law **`ratio = [A(D+2)+d]/(d+m)`** (`A = N_at/K`). ⛔ The shipped `byte_ratio_law` is wrong for `n_spectator > 0` — use `harness-debt`'s fixed version or recompute. |
| **learned-initial-state rule** | an initialisation is **PARAMETERS**; only the per-sequence deviation is **STATE**. Both declared per arm. A Titans-style arm's initial fast-weights are parameters; its per-stream update is state. |
| **capacity** | same `K` items, same well/code vocabulary size `N_a`, same `F`, same address dim `d`, same payload dim `m`. |
| **launch protocol** | the **same `P` designed launch offsets** on every arm (a designed mechanism; the null assigns each launch by its own rule). Offsets are parameters (ledgered); the occupancy vector is a per-read transient (F4), not state. |
| **reader class** | identical architectures, identical fitting budget, fitted on the SEEN split only, on both arms. Reader parameters ledgered on both arms. |
| **compute** | per-query read compute reported per arm; if the arms differ by > 2×, the cheaper arm is additionally run at the richer arm's budget. |

---

# 2. THE CAT TEST — construction, and the rule-4 proof obligation

## 2.1 The store (the vehicle: the factored store, §A4.5, un-deferred)
`N_a` **shared wells** (the vocabulary). Each item is an **`F`-subset** of wells (`F` = 4 registered).
Each well `j` carries `a` atoms and a **payload vector `v_j ∈ R^m`, drawn at write time and existing
only in the store**. Items **share** wells: sharing factor `S = K·F/N_a` items per well.

## 2.2 The query family
For an item/query `x` with well-set `A(x)`, the target is
### `y(x) = Σ_{j ∈ A(x)} v_j`   (registered primary; the `mean` variant is the declared alternate)
**Seen split:** `K` written items. **Unseen split:** `F`-subsets **never written**.

## 2.3 ⭐ The four §A3.7 rules, each discharged by CONSTRUCTION (not by hope)
| rule | how it is discharged | the check that runs (blocking) |
|---|---|---|
| **1 — not recoverable from row order** | `y` depends on `A(x)` only; insertion order is **re-shuffled per seed** and `v_j` is order-independent | `corr(score, insertion rank) `; and the `order_aware_launder` (+0 B) must score ≤ chance + 0.05 |
| **2 — not the query, or a function of it alone** | `v_j` is **store content**, drawn at write time, never in `φ(x)`; `A(x)` alone does not determine `y` | a reader fitted on `φ(x)` **with a blank store** must score ≤ chance + 0.05 (§2.5) |
| **3 — not at the arg-min table's maximum** | the operating point is registered in §7 and the nearest-item table's score is **measured first** | pre-condition **K3**: nearest-item table score ≤ 0.60 of the metric range, else the cell is void |
| ⭐ **4 — the answer is provably NOT in the table** | ⭐ **combinatorial construction:** every unseen `A` satisfies **`|A ∩ B| ≤ F−2` for every stored `B`** ⇒ any single stored row differs from the truth in **≥ 2 of `F`** wells; and `min_B ‖y(A) − y(B)‖ ≥ tol` is asserted per query | pre-condition **K2**: assert both, per query, at construction; the split is **rejected**, not repaired, if either fails |

⭐ **Rule 4 is a construction and it is verifiable in `O(K·|Q|)` set intersections.** The `aggregate`
family survived two +0 B audits for exactly this property; the cat test inherits it in set form.

## 2.4 Feasibility of the split — CHECKED, with numbers (`s5_design.py`, `s6_frontier.py`)
One stored item blocks `B = F(N_a−F)+1` combinations; expected surviving held-out count
`≈ C(N_a,F)·exp(−K·B/C(N_a,F))`. Verified against an explicit greedy construction (F=4):

| `N_a` | `K` | total combos | **rule-4-valid held-out (measured)** |
|---|---|---|---|
| 16 | 128 | 1 820 | **0** ⛔ (the split does not exist) |
| 24 | 64 | 10 626 | 6 150 |
| **32** | **128** | **35 960** | **23 193** ✅ |
| 32 | 256 | 35 960 | 14 261 |

⇒ **registered wave-scale design point: `N_a = 32`, `F = 4`, `K = 128`** (23 193 valid held-out
combinations; sample 512 of them, 5 seeds). ⛔ **`N_a = 16` is registered as FORBIDDEN — at `K ≥ 128`
the rule-4 held-out set is EMPTY and the family is unbuildable.** This is the kill-condition built
before the thing it can kill.

## 2.5 ⚠ Blank / leak controls — registered up front (collapse mode #4)
A compositional family is **more** leak-prone, not less (N68: a 1e-4 address leak made classification
perfect; blanks 0.992–1.000). Four controls, all blocking, all run **before** the arms are compared:
1. **Blank store** (wells present, `v_j` deleted/zeroed): every reader ≤ chance + 0.05.
2. **Query-only reader** (`φ(x)` → `y`, no store): ≤ chance + 0.05. ⭐ This is the control rule 2 exists
   for, and it is the single most likely way this family dies.
3. **Permuted payloads** (same-keys null: same wells, `v_j` permuted): ≤ chance + 0.05.
4. **Address-leak probe:** item-identity decode from the read's address block, full vs launder — a
   dividend of exactly 0.000 there is the C2W3 signature of a pure-query channel and must be reported.

## 2.6 ⛔ The admissible claim form (verbatim — C2W5's engineer copies this sentence)
> *"Wells `{j}` are co-activated by queries whose ground-truth factor set contains factor `f`, with
> co-activation correlation `ρ = …` (95 % CI …), measured against a permutation null. No well is
> identified with any factor; the claim is a correlation between co-activation/wormhole/shell-position
> statistics and task structure."*
⛔ **"The fur well" and every semantic naming of a well is forbidden in every artifact, including
figure captions and code comments.**

---

# 3. ⭐⭐ THE FALSIFIERS (each with a sign, a threshold, a tolerance and a seed count)

**Statistics convention (registered):** 5 seeds; sample sd (`ddof = 1`); `SE = sd/√5`; a quantity
*clears* iff `mean − 2·SE > threshold`; a quantity *fires* iff `mean + 2·SE < threshold`.
**Metric:** held-out **exact-set accuracy** (fraction of unseen queries whose decoded payload is within
`tol` of `y(x)`), range `[0,1]`, chance registered per split and reported.

## 3.0 Pre-conditions — ⛔ RUN AND PASS BEFORE THE ARMS ARE EVER COMPARED
| id | check | bar | if it fails |
|---|---|---|---|
| **K1** | write admissibility per arm: endpoint write loss ≤ **0.05**, `λ_min > 0` at ≥ **90 %** of wells, **SC-6 capture radius ≥ σ_q** at ≥ 90 % of wells | all three | the cell **ABSTAINS** (route3 precedent: neither unlocks nor blocks); coverage is reported first-class |
| **K2** | rule-4 assertions (§2.3) | 100 % of held-out queries | the split is **rejected and rebuilt** |
| **K3** | nearest-item table + the strongest **+0 B** substitute on the raw item table | ≤ **0.60** of metric range | ⛔ **the family is protocol-invalid** (FB4 killed 3 of 4 families this way) |
| **K4** | the four §2.5 leak controls | all ≤ chance + 0.05 | ⛔ **family void** |
| ⭐ **K5** | ⛔ **the per-item table launder** (§A9.5 at a new level): a `K`-row table keyed by nearest stored item, reading through the **same** reader class | **the read must beat it by > 0.10** on ≥ 1 reader | ⛔ **the read is table-expressible ⇒ tier ii is dead by the C2W3 mechanism** (a contractive-within-item / separated-across-item flow map, Fisher 6.3 → 1.47e7). ⚠ **This is the falsifier §5's flow-map warning demands, and it must run in stage 1.** |

## 3.1 F1 — ⛔ "The organizer swap shows no margin" (the headline falsifier)
**Fires iff** `OD_min ≡ min_R [ s(physics,R) − s(null*,R) ] ` has `mean + 2·SE < +0.05` over 5 seeds,
with `null*` per §4.3. ⇒ **tier ii is dead at this weight class and §A13's reframe loses its vehicle.**
**Clears iff** `OD_min − 2·SE > +0.05` **and** `OD(R) > 0` for **≥ 3 of the 4** readers.
**Ambiguous band** `|OD_min| ≤ 0.05`: reported as **TIE**, which is itself the registered
most-likely outcome (§3.6) and is a *finding*, not a null.

## 3.2 F2 — ⛔ "The margin is the reader's"
**Fires iff** `max_R OD(R) − 2SE > +0.05` **while** `min_R OD(R) + 2SE < 0` (positive at one reader,
negative at another) ⇒ the margin is a reader artifact, not organization. Report the whole
per-reader curve either way (quote the curve, not the endpoint).

## 3.3 F3 — ⛔ "The answer was in the table after all"
**Fires iff K3 or K5 fails** (bars above). ⭐ **Both are pre-conditions, checked BEFORE the arms are
compared** — build the kill-condition before the thing it can kill (C2W3 standing doctrine).

## 3.4 F4 — ⛔ "Sharing is unaffordable" — and here is the arithmetic, in advance (§5)
**Fires iff** the write cannot be made admissible (K1) at any `a` (atoms per well) for which the byte
ratio is within the registered target band. Registered bands, from the exact byte law
`ratio = 1.4·(N_at/K) + 0.8` at `d=4, m=1, n_spec=0`:
- tier-ii-legal (arms matched to each other; **no byte claim needed**): any `a` — `a = 12` gives
  `ratio = 5.00` at `(N_a,K) = (32,128)`;
- byte-frontier claim (optional, tier-i flavour): needs `N_at/K ≤ 1/7 = 0.1429`.
**Registered prediction:** K1 **passes** at `a ≥ 12` and **fails** at `a ≤ 4`. If K1 fails at every `a`
with `ratio ≤ 5`, F4 fires and the vehicle is unaffordable at this scale.

## 3.5 F5 — ⛔ "The organization is a (weighted) Voronoi diagram" — the structural falsifier
This is the one my derivations say is most likely to fire, so it is registered with its own bar.
**Fires iff** the **fitted static-geometric null** (§4.2 arm N3: `argmin_j [‖z−c_j‖²/2σ_j² − b_j]`,
fitted on the SEEN split) reproduces the physics arm's **assignment** on ≥ **99 %** of held-out
queries ⇒ the physics organizer is a VQ with a different codebook, and the *only* admissible tier-ii
claim is a **placement** claim (which `null*` attacks directly).
**Registered prediction (derived, §7):** it **FIRES at `d/s ≥ 4`** (measured disagreement **0.0000**,
3 depth ratios × 4000 queries) and **does NOT fire at `d/s ∈ [2.5, 2.9]` with depth ratio ≥ 3 and
`γ ≤ 0.05`** (measured irreducible disagreement **0.193–0.203**).

## 3.6 ⭐ THE ONE-LINE ANSWER: what kills tier ii outright
> ⛔ **If, at an operating point that passes K1–K5, the physics-organized store does not beat
> `null*` by `> +0.05` on the WORST reader in the class, multi-seed (5), on rule-4-valid unseen
> combinations — tier ii is dead at this weight class**, and the honest report is that physics-trained
> organization is *a competitive organizer with byte-exact deletion and settable lifetimes*, i.e. an
> "and also" claim, not a dividend.

**Pre-registered prior (I commit to numbers, per the w14 rule).** `P(F1 clears) ≈ 0.20` ·
`P(TIE, |OD_min| ≤ 0.05) ≈ 0.55` · `P(physics loses by > 0.05) ≈ 0.25`. Reason: §7 shows the two arms'
latent maps are **provably identical** at the designed gate and differ only through an **inertial**
term that is 99 %-recoverable in-distribution by 1-NN; the entire hypothesis rests on that term
extrapolating better off-support, which is unmeasured.

---

# 4. THE ORGANIZER ARMS

## 4.1 The physics arm (`O_phys`)
Factored store; masked/local write per well-group; admission + canonical placement; **soft
certificate ON** (SC-1…SC-7, `B = 0.33`, ⚠ its outer edge is being re-located by `bprime-c6` — **use
their number; if it has not landed, this prereg's operating point §7 is used and the dependency is
declared in the report**); training signal through **implicit gradients at the settled point** (which
reach `θ` exactly) and/or **trajectory reads** (mandatory for anything upstream of `∇V`); Route 3's
training-time machinery (slotted/trajectory write objectives, `allocate`, flow-map shaping) is
available **as tooling** (§A14.1).

## 4.2 The non-physics arms (each **matched-capacity**, each with its **strongest** form)
| arm | mechanism | DOF | bytes (learned-initial-state rule) | strongest form (registered) |
|---|---|---|---|---|
| **N1 gradient-placed atoms** ⭐ | **identical store parameterisation** (`c_j, log s_j, amp_j`), trained by plain Adam on the read objective with a **static** assignment rule — no rollout anywhere | identical to the physics arm, exactly | identical by construction | ⭐ **the cleanest and most damaging swap: same landscape family, non-physics training signal.** Any margin is then attributable to the *dynamics as a training signal* and to nothing else |
| **N2 VQ** | `k`-means / VQ-STE codebook of `N_a` codes + per-code payload | `N_a·(d+m)` | codebook = parameters | best of {k-means++ ×10 restarts, VQ-STE with EMA, product-VQ}; commitment-cost swept ≥ 5 points |
| **N3 fitted static-geometric rule** ⭐ | `argmin_j [‖z−c_j‖²/2σ_j² − b_j]` — power/Apollonius cells; **the decision rule the physics arm provably has to leading order** (§7 Thm O2) | `N_a(d+2)` | parameters | fit `(c, σ, b)` jointly on the seen split; this is F5's null |
| **N4 kNN** | no training; raw item keys + payloads; `k` nearest with IDW | `K(d+m)` | state | `k` swept `{1,2,3,5,10}`, uniform and IDW (the C2W1 `knn2_idw` substitute was the arm that beat us) |
| **N5 Titans-style write** | fast-weight memory `M_t` updated by a surprise-gated online rule; "organization" = the learned map | matched param count | ⭐ **init = PARAMETERS; per-stream deviation = STATE** — both declared | chunk granularity matched to the physics arm's; momentum + weight-decay variant per the published rule |

## 4.3 ⭐ The honest null (theorist T5.2 applied)
> ### `null* = max over ALL non-physics arms AND their entire registered tuning grid` — **computed, not estimated.**
**Tuning budget, registered and committed:** each arm gets **≥ 5 learning-rate points × 3 capacity
points × 3 seeds** on the SEEN split, selected by a held-out-from-seen validation split (never on
`Q_unseen`). The physics arm gets **the same budget, no more**. ⛔ The weak null
(`null(one sampled config)`) may not carry a headline. ⚠ **`null*` includes N3 fitted on the physics
arm's own assignments** — the "oracle-imitation" null (T5.2 rider (i)); a physics arm that cannot beat
an imitation of itself has no organization claim.

---

# 5. SHARING AFFORDABILITY vs `S*`, AND THE DELETION CURVE

## 5.1 The arithmetic (exact; `s4_sharing_bytes.py`, rationals; reproduces T1.3 exactly)
`ratio(S) = A_tot(D+2)/[S(d+m)] + d/(d+m)`; matched bytes ⇔ `S* = A_tot(D+2)/m = 7·A_tot`
(`d=4, m=1, n_spec=0`). Equivalently, and more usefully:
### ⭐ `ratio = 1.4·(N_at/K) + 0.8` — the byte ratio depends ONLY on **effective atoms per live item**.
Sharing is *only* a way of lowering `N_at/K`. Matched bytes ⇔ `N_at/K ≤ m/(D+2) = 1/7 = 0.1429`.

## 5.2 What the cat test needs, checked against `S*` — **the answer is: it is reachable, but not at a
write budget anyone has ever converged**
| `a` (atoms/well) | `N_at` at `N_a=32` | `N_at/K` at `K=128` | **ratio** | `S` | `S*` needed | verdict |
|---|---|---|---|---|---|---|
| 1 | 32 | 0.25 | **1.15×** | 16 | 28 | bytes nearly matched; **write budget 1 atom/well** |
| 4 | 128 | 1.00 | 2.20× | 16 | 112 | — |
| **12** | **384** | **3.00** | **5.00×** | 16 | 336 | ⭐ **registered default: the write has a chance** |
| 32 | 1024 | 8.00 | 12.0× | 16 | 896 | — |
**Matched-bytes + rule-4-constructible frontier** (`s6_frontier.py`, F=4, `K = 7·N_a·a`):
`a=2 ⇒ (N_a,K) = (28, 392)` · `a=4 ⇒ (32, 896)` · `a=8 ⇒ (40, 2240)` · **`a=11.65 ⇒ (44, 3589)`** ·
`a=341 ⇒ (156, 372 372)`.
> ⭐⭐ **The registered affordability statement.** The shipped write **converges** at 341 atoms/item
> (loss 2e-4, `λ_min +3.24`) and **fails** at ≤ 11.65 (loss 0.20–0.24, `λ_min ∈ [−1.20,−0.21]`).
> Matched bytes needs `N_at/K ≤ 0.143`. ⇒ **a byte-matched cat test at the shipped write's known
> failure edge (11.65 atoms/well) needs ≈ 3 600 items and 44 wells; at the shipped converged budget it
> needs ≈ 372 000 items.** ⛔ **Therefore: C2W5 must NOT promise a byte-matched tier-ii result.** Tier
> ii's control is the organizer swap, which is byte-matched **across arms by construction**; the
> table-relative ratio is *reported*, never claimed. **If a byte-frontier claim is wanted later, its
> price is `K ≈ 3.6k` items — that is a wave of its own, and it is now costed in advance.**

## 5.3 The deletion curve (§A9.9 standing ruling — registered definition)
Private atoms per item `a_priv`; private parameter-mass fraction `p = a_priv/(a_priv + F·a)`, capped by
`p ≤ [(d+m)r − d]/[(D+2)A_tot]`. ⛔ **A single scalar "deletion still works" is inadmissible on a
shared substrate.** The curve, x-axis `p` (equivalently `r`), **two** series:
1. **exactness on the private fraction** — byte-equality (1.0 by construction), `AUC(z_hole)`;
2. **measured degradation on the shared fraction** — read error and MIA-AUROC after (i) leaving shared
   atoms, plus wall-clock after (ii) re-fitting them (which is a *write*, i.e. the retraining baseline).
Registered anchors (`s4`): at `A_tot = 4`: `p ≤ 0.036 (r=1)`, `0.214 (r=2)`, `0.286 (r=2.4)`,
`0.750 (r=5)`; at `A_tot = 32`: `0.0045 / 0.027 / 0.036 / 0.094`; at `A_tot = 341`: `4.19e-4` at `r=1`.

---

# 6. DESIGN-RULE COMPLIANCE (§A13 / task §5) — each rule: satisfied or waived-with-reason

| # | rule | status | how |
|---|---|---|---|
| 1 | ⛔ `∂q*/∂q₀ = 0` ⇒ organizer trains only through trajectory reads / implicit gradients | ⭐ **SATISFIED, by design choice** | **φ is FROZEN and identical on all arms**, so no gradient ever needs to reach `q₀` — the T3 zero is discharged rather than fought. Store parameters `θ` receive **implicit** gradients (`∂q*/∂θ = −(Hess V)⁻¹∂_θ∇V ≠ 0`, exact). Any *learned* launch/particle head is **trajectory-read only**. **Report `‖∂L/∂·‖` at init for every trainable group** (`0.0` implicit / `2.654e-9` unroll / `6.421e-3` trajectory are the reference scales) |
| 2 | ⛔ γ/M trainable only through that channel; γ is the ~14× stronger selector | **WAIVED with reason** | γ is a **designed, registered operating point** here (§7), swept as an axis, **not learned**. If C2W5 learns it, it is trajectory-read-only and the ratios (`2.6–4.9e5` friction vs `1.7–2.9e5` mass) are the sanity check |
| 3 | ⛔ allocation collapse is a gradient-flow **attractor** | **SATISFIED** | the factored analogue is **collapse to private wells (S→1) or to a single well (N_eff→1)**. **Registered:** initialise the well-assignment logits **away** from both corners (uniform + noise), report **`‖∂L/∂(assignment logits)‖` at init** beside a liveness anchor, and monitor `S_eff = K·F/#(used wells)` every consolidation; a run whose `S_eff` leaves `[S/2, S]` is reported as collapsed, not as a null |
| 4 | ⚠ the 2α coercivity floor (`τ_max = Γ/2α`; α is the ceiling) | **CARRIED** | every lifetime/manifold statement carries it. ⭐ New, and it binds §7: **at the merger edge the axial curvature is exactly `2α` (measured `+0.1000` at `d/s = 2.00`, `α = 0.05`)** — the coercivity floor *is* the merger edge |
| 5 | ⚠ soft certificate (SC-1…SC-7, `B = 0.33`) is the sharing precondition; its edge was located with a broken ruler | **DEPENDENCY DECLARED** | use `bprime-c6`'s re-located `B`. ⛔ If it has not landed at C2W5 scoping, use §7's `d/s` band **as the operating spec** and declare the dependency in the report — **do not quote the stale `B = 0.33` edge as certified** |
| 6 | ⭐ Route 3's training-time machinery is LIVE tooling; only the **inference-read** claim is closed | **SATISFIED, stated in the metric** | §0 states table-like reads are permitted on both arms and that post-training reduction to a table is a **computational win**. `allocate`, if used, is a **pure function of `(query, shared policy params)` — never a per-item table** — with the **allocation-shuffle test** (`full_bytes` bit-identical under any permutation) as a **blocking** check |
| 7 | ⚠ the flow-map warning (contractive-within-item + separated-across-item ⇒ table-expressible) | ⭐ **SATISFIED by a registered kill-condition** | **K5** (§3.0) is exactly that check, and it runs **before** the arms are compared. Additionally report the flow-map pair (within-item contraction, between-item separation, Fisher ratio) per arm — the C2W3 values (0.218 / +15 % / 6.3 → 1.47e7) are the reference |

---

# 7. ⭐⭐ THE OPERATING POINT — derived, and it is the most decision-relevant thing in this document

Four results (derivations in the report; toys in `.claude/scratch/orgdiv-prereg/`) fix the band in
which a tier-ii dividend can exist **at all**:

1. **Theorem O1 (the quantizer bound).** Under a settled-point read the image of `x ↦ q*` is exactly
   the **set of minima of `V_θ`** (`Fix(T_θ)` contains no query variable), so an `N_min`-row table
   reproduces the read **for every reader**. *Measured:* 2 distinct settled points from 4000 queries
   in a 2-well store; 3 on a 41-point sweep of the whole inter-well segment (the third is the
   separatrix itself, measure zero). ⇒ ⛔ **composition CANNOT live in a settled point.** The read
   **must** be a **multi-particle occupancy** read (image up to `N_min^P`), which is §A4.5's
   multi-particle read and is registered as mandatory.
2. **Theorem O2 (the VQ ceiling).** To leading order the physics partition **is a power diagram** with
   weights `w_j = 2s² ln A_j`; exactly, the boundary offset is `δ = ln(A_i/A_j)/(d/s² − 4/d)`, i.e. the
   power-diagram offset **amplified by `[1 − 4(s/d)²]⁻¹`**. *Measured:* the fitted amplification is
   **1.940 vs 1.907 predicted** at `d/s = 2.9` (1.7 %); the power diagram's relative error tracks
   `4(s/d)²` (0.138–0.151 vs 0.111 at `d/s=6`; 0.214 vs 0.207 at 4.4; 0.333 vs 0.327 at 3.5). At
   **uniform spacing a single fitted weight scale absorbs it entirely.** ⇒ ⛔ **at the designed gate
   `d_safe = 4.4 s` the settled-point organization is EXACTLY a nearest-centroid VQ: disagreement
   `D = 0.0000` at depth ratios 1.5, 3.0 and 6.0 (4000–6000 queries/cell).** **A tier-ii experiment run
   at the designed gate has a structurally zero dividend and must not be run there.**
3. **Prop O3 (the 3-body term is real and useless).** A power/Voronoi partition is **pairwise**; the
   CLU's is a level set of a sum over **all** wells, so a third well shifts the `i–j` boundary — a
   term provably outside the power-diagram class, with the sign **flipping with the third well's
   side** (so no weight assignment absorbs it). *Measured:* `shift ≈ 1.19·exp(−½(D_k/s)²)` in address
   units, `D_k` = third well's distance **to the boundary**, log-slope **−0.962 vs −1 predicted**;
   magnitudes `6.3e-4 … 2.3e-9` for `d_k/s = 2.5 … 5.0`, i.e. **≤ 7e-4 of `d_ij`**. ⛔ **Below every
   noise floor the harness has ⇒ tier ii may not be built on it.**
4. ⭐ **Prop O4 (the inertial term — the only structurally non-VQ channel with usable magnitude).**
   At `γ ≤ 0.05`, `d/s ∈ [2.5, 2.9]`, depth ratio ≥ 3, **19–20 % of query mass** is assigned to a well
   that **no fitted static-geometric rule** (power **or** Apollonius/GMM-MAP, both fitted) reproduces.
   It is **momentum-carried**: the irreducible disagreement runs `0.197 (γ=0.05) → 0.040 (0.1) →
   0.0003 (0.2) → 0.0003 (0.5)` — a **×600 collapse** — and at large γ the read degenerates to
   "assign the launch point", which is monitor #1 exactly. It is **coherent, not chaotic**: flip rate
   under a `0.1σ_q` launch perturbation is **0.0–2.0 %**, and leave-one-out **1-NN recovers the
   assignment at 0.989–1.000** *in distribution*.
   ⇒ ⭐⭐ **Two consequences, both binding on C2W5.** (i) The tier-ii dividend can only be measured
   **off the training support** — in-distribution, a kNN null recovers ~99 % of the only channel that
   distinguishes the arms; **this is precisely why the metric is scored on unseen combinations, and it
   is now a derivation rather than a preference.** (ii) **γ is the tier-ii operating dial**: at
   `γ ≥ 0.2` the physics organizer *is* a VQ (monitor #1's collapse mode, seen from the organization
   side).

> ## ⭐ REGISTERED OPERATING POINT (the cell C2W5 runs; deviations must be argued)
> `d/s ∈ [2.5, 2.9]` (⛔ never ≥ 4.0 — provably zero dividend; ⛔ never ≤ 2.01 — **merger**: the axial
> curvature at the midpoint is `2α + (2A/s²)e^{−u/2}(1−u)`, `u = (d/2s)²`, so two equal wells become
> ONE minimum below `d/s = 2.0005…2.0145` over the whole shipped `(A,s)` box, and at `d/s = 2.00` the
> axial curvature is exactly `2α = 0.1000`) · **depth heterogeneity ≥ 3×** between neighbouring wells
> (at ratio 1.5 the non-VQ mass is only 0.0013) · `γ_address = 0.05`, `γ_read = 0.02` (shipped), with
> **`γ ∈ {0.02, 0.05, 0.2}` as a registered axis** — `γ = 0.2` is the *internal* VQ-collapse control ·
> multi-particle occupancy read, `P ≥ 4` designed launches · `N_a = 32`, `F = 4`, `K = 128`,
> `a = 12` (ratio 5.00×, reported not claimed).
> ⚠ **Bracket, not a measurement:** `s` for a *learned multi-atom* well is an unsolved modelling
> question (`bprime-theory` §9.2). All `d/s` numbers use the single-atom width proxy. ⭐ **C2W5's
> first-day instrument therefore includes an effective-`s` estimator** (fit `A e^{−r²/2s²}` to the
> radial profile of each written well) and the operating point is set on **measured** `d/s`.

---

# 8. WHAT C2W5's LEAD MUST RESOLVE AT SCOPING (named open questions)

1. ⭐ **The effective `s` of a learned multi-atom well** — everything in §7 is expressed in `d/s`, and
   `s` is not measured on a learned store. **First-day instrument; blocking for the operating point.**
2. ⭐ **Does the write converge at `a ≈ 12` atoms per WELL in a factored store?** The shipped failure
   evidence is per *item*-well; a factored write digs `N_a` wells, not `K`. **K1 answers it; if it
   fails at every `a`, F4 fires.**
3. **Which reader class exactly** (4 registered: nearest-well table · kNN · linear · small MLP) — sizes
   and the fitting protocol must be frozen before the first arm runs.
4. **Does `bprime-c6`'s re-located `B` land before scoping?** If not, §7's band is the spec (declared).
5. **Payload composition: sum or mean?** Sum is registered; mean changes the rule-4 tolerance
   arithmetic and must be re-verified if adopted.
6. **Is `allocate` used at all?** If yes, C4/C5 apply (pure function of `(query, shared policy)`,
   allocation-shuffle test blocking, `‖∂L/∂logits‖` at init).
7. **Seeds and compute:** 5 seeds × 5 arms × 4 readers × 3 γ = 300 cells. Budget and stagger declared
   before the first run; ≤ 3 local worktrees.

---

# 9. ⛔ DECLARED **NOT DERIVED** (never to be presented as settled)

1. **That the inertial term (Prop O4) carries *useful* information off-support.** It is coherent and
   non-VQ **in distribution**; whether it extrapolates to unseen combinations **is the tier-ii
   hypothesis and is unmeasured**. Everything in §3.6's prior rests on this being open.
2. **Every geometric number here is a 1-/2-D toy with single-atom wells** (`s` fixed, equal widths,
   `p₀ = 0`, `T = 0`, Newtonian `M = I`, no training). Transfer to a learned multi-atom store is
   **bracketed, not measured** (carried from `bprime-theory` §9.2 / `doctrine-repairs` OQ-C).
3. **The list of structurally non-VQ channels is NOT proven exhaustive.** I proved the 3-body term is
   outside the power-diagram class and measured an inertial term outside every static-geometric rule I
   fitted; I did **not** prove no other channel exists.
4. **Multi-particle occupancy expressivity is bounded, not characterised.** `N_min^P` is an upper
   bound on the image; what fraction is reachable from `φ`-determined launches is unmeasured (the
   `3d+1` launch-manifold rank result says the *per-item* content is bounded — the cross-item/set
   content is not).
5. **No claim that the physics arm's placement differs from `null*`'s at all.** If both organizers
   converge to the same codebook, `OD = 0` identically — untested, and it is F1's most likely route.
6. **Nothing here licenses a tier-i or tier-iii claim**, and no gym family is a claim venue (§A14.8).

---

# 10. THE ORDER C2W5 RUNS IN (so day one is startable)

1. effective-`s` estimator + operating-point calibration (OQ-1) → **K1** write-admissibility sweep over
   `a ∈ {4, 12, 32}` → 2. build the family + **K2** rule-4 assertions → 3. **K3/K4/K5** kill-conditions
   on the physics arm alone → 4. only then: the arms, `null*` computed over the full tuning grid →
   5. `OD(R)` per reader, 5 seeds, the curve → 6. the deletion curve (§5.3) as the "and also" column.
**Steps 1–3 are the whole first half of the wave and they can kill the wave cheaply. That is the
point.**

---

# 11. Provenance

Scripts (pure numpy/scipy, float64, **no repo code imported**, main venv
`/Users/user/Desktop/CHLU/.venv`, numpy 2.4.1 / scipy 1.17.0): `.claude/scratch/orgdiv-prereg/`
`common.py` (shipped damped velocity-Verlet, copied verbatim from `bprime-theory`) · `s1_merger.py` ·
`s2_partition.py` · `s3_quantizer.py` · `s3b_gmm_null.py` · `s3c_gamma.py` · `s3d_coherence.py` ·
`s4_sharing_bytes.py` · `s5_design.py` · `s6_frontier.py`, each writing its own JSON.
Constants: `α = 0.05`, `s = 0.30`, `dt = 0.05`, two-phase read `(γ,N) = (0.05,400) → (0.02,800)`,
`σ_q = 0.15`, `M = I`, `p₀ = 0`, `T = 0`, seeds `default_rng(0)`. Repo read-only at `main @ d4f56c8`;
**zero tracked-code edits, no branch, no worktree.**
My own predictions were registered in `PREREG-theory-checks.md` **before any script existed**;
the scorecard is in `.claude/outputs/orgdiv-prereg.md` §8.
