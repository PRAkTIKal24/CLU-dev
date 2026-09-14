# PREREG — cl-entry-build (w25): the CL entry (R4 + R3-native + R1-survivor)

**Written BEFORE the harness ran** (protocol §5 pre-registration rule). Everything below is a
prediction with its derivation. Measured values go in the report, next to these.

**Dial declaration (protocol §7) — echoed from the task file.**
- **Dials:** all four (admission/isolation = the anti-forgetting mechanism; lifetimes = scheduled
  retention on the live stream; compute-adaptive reads = retry in its native regime).
- **Laundering control:** kNN-in-φ at matched memory on every claim.
- **Falsifies:** losing the replay-free class (EWC/SI/LwF-class) or being matched by the
  kNN-in-φ launder; a flat retry curve, or a lift matched by kick/ensemble.
- **Does NOT falsify:** sitting below replay/GDumb/iCaRL (CM-23(n) filing rule); any oracle
  comparison.

---

## 0. Protocol being pre-registered

- **Stream:** Split-MNIST (5 tasks × 2 classes), Class-IL (task id NOT given at test time),
  from scratch. Split-CIFAR-10 same shape as the hard rung.
- **φ:** `task1_only` (PRIMARY, per `PREREG_CL_PHI.md`), `generic_frozen` (declared upper bound),
  **`phi_dim = 32 ≥ 16`** (binding clause §7 of that document). φ fit pool disjoint from every
  stored item; frozen at end of task 1; never refit.
- **CLU entry:** designed store = Gaussian wells over φ(x) with per-well amplitude, address =
  φ(x) ∈ R^32, payload = the class label; read = damped-Verlet settle → landed well → its label.
  MVC-0 controller: admission (spacing gate `d_safe = d_safe_mult·s`, refuse-and-relocate is
  DISABLED in φ-space — a relocated address is not the item's address; refusal is the only
  legal admission outcome here), placement, class-balanced eviction under a global item budget,
  per-item scheduled decay.
- **Memory budget:** 200 items for the CLU store AND for ER / iCaRL / GDumb / kNN-in-φ
  (matched **items**; bytes reported in the same table — the CLU store holds 32 floats/item,
  a raw exemplar 784, so CLU is ~24× cheaper per item and this is reported, never claimed as a win).
- **Metrics:** GEM formulas — ACC = mean_i A[T,i]; BWT = mean_{i<T}(A[T,i] − A[i,i]);
  forgetting = mean_i (max_t A[t,i] − A[T,i]).
- **Seeds:** 0,1,2 on every headline.

## 1. Item 1+2 — the entry and the mandatory baseline table (Split-MNIST, ACC at end of stream)

Derivation of each prediction is in the right column. **These are the numbers to beat/compare.**

| method | class | predicted ACC | derivation |
|---|---|---|---|
| finetune (none) | — | **0.19 ± 0.02** | van de Ven Table (19.90) |
| EWC | regularization (known null) | **0.20 ± 0.03** | van de Ven 20.01 |
| SI | regularization (known null) | **0.20 ± 0.03** | van de Ven 19.99 |
| LwF | distillation | **0.23 ± 0.05** | van de Ven 23.85 |
| ER (200) | replay | **0.80 ± 0.06** | small-buffer ER on Split-MNIST, Mammoth-class |
| iCaRL (200) | exemplar + NME | **0.88 ± 0.04** | published 94.6 at larger memory; −6pp for 200 exemplars + our reduced training |
| GDumb (200) | balanced buffer + retrain | **0.85 ± 0.05** | GDumb is strong at small budgets |
| **kNN-in-φ (200 keys)** | **the launder** | **0.83 ± 0.05** | 1-NN over 200 PCA-32 prototypes ≈ 0.85 on MNIST; −2pp for the strict (task-1-only) basis |
| **CLU entry, `task1_only` φ** | rehearsal-free store | **0.82 ± 0.06** | the store approximates 1-NN in φ over the admitted keys; settle costs a little vs exact NN |
| CLU entry, `generic_frozen` φ | reference upper bound | **0.85 ± 0.05** | +0.03 strict-φ cost on MNIST (PREREG_CL_PHI §7: gap ≈ 0 at phi_dim 32; allow +0.03) |
| joint (offline) | upper bound | **0.97 ± 0.01** | MLP on all 10 classes |

**Registered directional claims (these are what pass/fail):**
- **C1 (the entry claim):** CLU entry (`task1_only`) **beats every rehearsal-free baseline**
  (finetune/EWC/SI/LwF) by **> +0.40 ACC**, on 3 seeds, non-overlapping error bars.
- **C2 (the filing rule):** CLU entry sits **below** ER/iCaRL/GDumb. Predicted deficit
  **−0.01 … −0.10**. *This is not a failure* (CM-23(n)) but it WILL be reported as a deficit.
- **C3 (the launder — the honest prior):** **kNN-in-φ at matched memory ties or beats the CLU
  entry**; predicted CLU − kNN ∈ **[−0.05, +0.01]**, i.e. **the laundering control FIRES**
  (three consecutive waves of the same pattern: N89/N90/N95). If CLU − kNN > +0.03 outside the
  tie band on 3 seeds, that is a *new* result and must be re-run before it is believed.
- **C4 (forgetting):** CLU BWT ∈ **[−0.20, −0.02]** (loss comes from added class confusability
  as the store fills, not from destruction of past wells); EWC/SI/finetune BWT ≤ **−0.85**
  (catastrophic); ER/iCaRL BWT ∈ [−0.25, −0.05].
- **C5 (controller):** admitted fraction per-offered ≈ budget/offered ≈ **0.02** (budget-bound);
  intervention (refuse+evict) rate rises above **0.90** by task 5; per-admitted class purity is
  NOT claimed as an accuracy number. Report BOTH per-offered admission and the admitted count.
- **C6 (sizing rule / packing):** with `clu_s_frac = 0.2` and `d_safe_mult = 4.4` the store is
  self-consistently spaced (`d_safe = 0.88 × median-NN`), and the **corrected packing slack**
  `median_NN / (3.1·max(s, σ_q))` with σ_q = RMS‖φ(test query) − nearest key‖ is predicted
  **< 1.0 (0.2–0.8)** — i.e. the CL store runs **past** the packing bound. That is intrinsic to
  classification (queries are *different images*, not corrupted copies) and is exactly why
  Item 3's ambiguity is geometric.

## 2. Item 3 — the R3-native retry ladder (crowded-store retrieval of PAST-task items)

Queries = pixel-space dropout (p=0.5) corruptions of **stored** items, embedded through the frozen
φ; target = the item's own well. **No oracle exists:** the erasure is applied in pixel space and
the store's metric is φ, so the masked-NN oracle that beat CLU in w24 (`headroom-retry-benchmark`)
**cannot be constructed** — there is no coordinate subset of the store's space that is "known
erased". The honest floor is kNN-in-φ (= the laundering control), reported at matched compute.

- **R1:** first-pass identity accuracy on stored items ∈ **[0.55, 0.90]** (off ceiling, so the
  ladder has headroom).
- **R2:** gated retry lift (best over k ∈ {0,1,2,4,8}) = **+2 … +12 pp**, monotone, **auto-stops
  at ≤ 2.0× compute**.
- **R3:** **random-kick and ensemble-of-k are flat or declining** (gap to gated ≥ +1 pp in every
  cell); **ungated-all declines at k = 8**. If the kick matches gated, the mechanism claim dies.
- **R4:** kNN-in-φ at matched compute is **within ±5 pp of, and predicted ≥, CLU-gated best** in
  ≥ half the cells (the laundering prior again). A CLU-gated win over kNN-in-φ here would be the
  first native-regime leaderboard win and must be re-run before belief.
- **R5 (task-age):** |accuracy difference between the oldest and the newest task-age| ≤ **10 pp**,
  with **no monotone age trend guaranteed** — nothing in the store distinguishes an old well from
  a new one except which sites were free at write time. If a trend exists, predicted direction:
  older = worse.

## 3. Item 4 — the R1 survivor: scheduled per-item retention on the live stream

Wording is fixed: **scheduled per-item retention / scheduled forgetting**. Never "certified",
never "unlearning", never "deletion by construction", never "exact deletion" (CM-22 m/n/o).

Three cohorts written into the running entry: **permanent** (`leak = 0`), **fast**
(half-life 2 ticks ⇒ leak = ln2/2 = 0.3466), **slow** (half-life 8 ticks ⇒ leak = 0.0866).
`ticks_per_task = 4` ⇒ 20 ticks over the 5-task stream; `amp_floor = 0.05`.

- **T1:** well amplitude follows `A(t) = A₀·exp(−leak·t)` to **< 1e-6 relative error** (it is
  implemented as exactly that recursion; this is a *consistency* check, not a discovery).
- **T2:** **permanent items retrieve at 1.000 at every tick**, through all 5 tasks.
- **T3:** fast-cohort items self-evict (amp < 0.05) at **t = ln(1/0.05)/0.3466 = 8.64 ⇒ tick 9**
  (predicted eviction tick 9 ± 1); slow-cohort at t = 34.6 ⇒ **never within the 20-tick stream**.
- **T4:** the fast cohort's *retrieval* retention holds at ≈1.0 while the well is alive and drops
  to 0 at eviction (a step, not a smooth decay) — the amplitude decay is smooth, the retrieval
  consequence is a threshold. Predicted step location = the eviction tick ± 2.
- **T5 (the cost):** the retention arm's end-of-stream ACC is **0.05–0.25 lower** than the
  no-decay entry (the fast cohort's coverage is gone). Reported as the price of the dial.

## 4. Item 5 — Split-CIFAR-10, the strict-φ cost band

- **F1 (the band the task asks for):** strict-φ cost `generic_frozen − task1_only` on CIFAR
  = **+0.02 … +0.15** (vs ≈ 0.00 ± 0.03 on MNIST). CIFAR's task-1 classes (airplane/automobile)
  span far less of the later-class pixel variance than MNIST's {0,1} do.
- **F2:** CLU entry ACC on Split-CIFAR-10 (`task1_only`, PCA-32) = **0.22 … 0.38**.
  1-NN over 200 PCA-32 CIFAR prototypes is weak; this is the honest expectation.
- **F3:** EWC/SI/finetune on Split-CIFAR-10 Class-IL ≈ **0.19 ± 0.03** (the same known null);
  ER/iCaRL/GDumb at 200 = **0.30 … 0.55** with our reduced from-scratch CNN training.
- **F4 (the falsifiable version of C1 on CIFAR):** the CLU entry still beats the rehearsal-free
  class, but by **> +0.05 only** — a far narrower margin than MNIST. **If CLU ≤ 0.20 on CIFAR
  (i.e. no better than the known null), that is the headline finding for the entry's scope and
  is reported as such.**

## 5. What would make me re-run before believing a number

Any of: (a) CLU − kNN-in-φ > +0.03 anywhere; (b) a retry lift > +20 pp; (c) EWC/SI above 0.30
(would mean my Class-IL evaluation is leaking task identity — the classic bug); (d) CLU ACC
above ER *and* iCaRL *and* GDumb; (e) BWT ≥ 0 for any parametric baseline.
