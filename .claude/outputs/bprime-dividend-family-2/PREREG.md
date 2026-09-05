# PREREG — B′ dividend family 2 (`triad`)
**Filed 2026-08-18 21:38 BST by `experiment-engineer` (task `bprime-dividend-family-2`), BEFORE any arm trains
and before any store is written.** Branch `agent/experiment-engineer/bprime-dividend-family-2`, base `main @ 3c1dbaa`.
⛔ Nothing in this file may be edited after the first cell runs; a falsifier written after a result is void.

## 0. DIAL DECLARATION (echoed from the task)
- **Dial:** none — audit width. Every performance number is governed by the B′ control battery.
- **Laundering control:** the protocol itself (matched-byte table launder · +0 B substitute audit · same-keys
  null · blank store · rescue gate), plus this family's own strongest +0 B reader, declared in §3 below.
- **Falsifies:** a dividend claimed without its launder column beside it.
- **Does NOT falsify:** the family failing protocol validation (that is the deliverable, task rule 2);
  losing to a classical +0 B reader on a metric-native question (metric-native-ceiling theorem).

## 1. The family: `triad` — three-way barycentric superposition

**One line.** The query sits at the *barycentre of three* neighbouring stored items and the answer is the
barycentric mixture of their three payloads; the target is constructed to lie **outside** the payload range
of the **two nearest** rows, so every reader that returns a convex combination of the two nearest rows —
arg-min, 2-NN mean, 2-NN inverse-distance — has an error bounded below by `payload_tol`.

**Why it is a second *opening*, not `aggregate` with a different λ.** `aggregate` (N.1) is *pairwise* linear
interpolation: its discriminating bound is against **arg-min** only, and its own strongest control (2-NN
mean/IDW) is a *two-row* reader. `triad` moves the bound up one level — it discriminates **2-row readers
from ≥3-row readers** — and it is the first family in the program whose construction rule is a statement
about the *cardinality of the read*, which is the property a superposing memory is supposed to have and a
pairwise interpolator is not. The store's operating point, atom budget, admission gate, byte ledger and
σ_q are **identical to the audited `aggregate@base` cell**; only the query law changes, so the two families
are directly attributable to each other.

**Configuration (identical to `aggregate@base` except the query law).**
`n_offer = 8`, `capacity = budget = 6`, `reference_capacity = 6`, `consolidate_every = 2`,
staged admission on, `atoms_per_item = 32` ⇒ 192 atoms, `d = 4`, `m = 1`, `n_spectator = 0`,
`ball_radius = 1.0`, `σ_q = 0.15`, `payload_tol = 0.1`, deletion / revisit / collision offer on,
≥ 5 consolidation windows. Metric `neg_mae`, exact maximum `M = 0.0`.

**The query law.**
1. Enumerate all triples `(i, j, k)` of **live** items whose maximum pairwise address distance is
   `≤ triad_dist_mult × sep` with `triad_dist_mult = 2.0` and `sep` the stream's minimum separation.
   If no triple qualifies, the single tightest triple is kept (so a cell is never empty by accident).
2. Per triple, attempt `n_query_per_triple = 8` queries. For each, draw weights
   `w ∝ U(0.25, 0.45)^3`, normalised (⇒ each `w_r ∈ [0.217, 0.474]`: all three items materially involved),
   by **rejection**, at most `triad_max_draws = 32` draws per query slot; a slot with no admissible draw is
   dropped and counted.
3. Target `t = Σ_r w_r a_r`; query address `Σ_r w_r c_r + jitter(σ_q = 0.15)`; launch `q₀ = (address, 0, …)`.

**The two construction rules (both enforced at construction, both asserted in the test suite).**
- **(C1) `t` is absent from the table** — dropped if `min_r |t − a_r| < payload_tol` over **all** stored
  payloads (N.5's subsuming rule; the same rule `aggregate` uses).
- **(C2) `t` is outside the two-nearest hull** — with `(r, s)` the two live items nearest the *jittered*
  query address, dropped unless `t` lies at least `payload_tol` outside `[min(a_r,a_s), max(a_r,a_s)]`.
  ⭐ **Therefore, provably: any reader that returns a convex combination of the two nearest rows has
  error ≥ 0.1 on every admissible query** (`neg_mae ≤ −0.1`). This covers `settle_deleted` (arg-min),
  `knn2_mean_+0B` and `knn2_idw_+0B`. It does **not** cover 3-row readers, which is the point.

**Check against N.5's four rules** (each necessary condition, argued in advance):
| N.5 rule | `triad` | why |
|---|---|---|
| answer not recoverable from the table's **row order** | ✅ | `t` is a symmetric function of the *set* of `(c, a)` rows and the query; permuting rows leaves it invariant. Nothing in the construction reads insertion order (contrast `recency`, struck). |
| not the query itself or a function of it alone | ✅ | `t` depends on three stored payloads drawn per seed; two streams with identical query addresses and different payloads have different targets (contrast `manifold`, struck). |
| the arg-min table is not at the metric's exact maximum | ✅ **provable** | (C1)+(C2) ⇒ arg-min error ≥ `payload_tol` = 0.1 ⇒ `neg_mae ≤ −0.1 < 0 = M` (contrast `overload@load1x_shipped`, frontier-only). |
| the target is constructed to be absent from the table | ✅ **by construction** | (C1), enforced at draw time and asserted in a test. |

## 2. The protocol-validation gate this family must pass BEFORE it is scored
The shipped FB4 gate (`chlu/eval/fb4_gate.py`, rule frozen 2026-07-31, constants **not** tunable):
`S(f) = (sub − blank)/(M − blank)`; `f` is **substitute-saturated iff `S ≥ 0.95` AND `sub ≥ attn − 2 SE`**
(paired SE). The family is **released for scoring iff it is NOT saturated**; if it saturates, it is
**STRUCK as protocol-invalid and the failure is the deliverable** (task rule 2 — ⛔ no silent swap).
Run at the same nine seeds as the audit (the rule's own registered convention is three seeds; the 3-seed
subset `{0,1,2}` is reported beside the 9-seed figure, and the **9-seed figure is the primary**).

## 3. The declared reader set (the +0 B substitute audit for this family)
Shipped value-family readers, unchanged: `settle_deleted` (arg-min), `same_keys_null`,
`knn2_mean_+0B`, `knn2_idw_+0B`, `raw_table_mean_+0B` (the constant predictor).
⭐ **Added for this family, declared in advance as its own strongest control** (the analogue of the 2-NN
mean that `aggregate` ships and that "is expected to win — it does"):
- `knn3_mean_+0B`, `knn3_idw_+0B` — three-row averages;
- **`bary3_+0B`** — the barycentric reader: take the 3 nearest rows, solve the least-squares barycentric
  coordinates of the query in the affine hull of their addresses (weights summing to 1), clip to the
  simplex, renormalise, return `Σ w a`. This **inverts the construction up to the query jitter** and is
  expected to be the strongest +0 B reader on this family.
⛔ These readers are added **only** for `triad`; the `aggregate` / `overload` code path is bit-for-bit
unchanged so no registered family-1 number moves (a regression test asserts it).
Attention (`+4 B`, one fitted temperature) enters through the FB4 gate as the attention leg.

## 4. Numeric predictions (registered before any run; derivations given)
Jitter algebra used throughout: an isotropic `σ_q = 0.15` displacement in R⁴ projects onto the triangle's
2-D affine coordinates with per-coordinate σ ≈ 0.15; the induced barycentric-weight error is `δw ≈ σ_q/ℓ`
with `ℓ ≈ 0.9` the typical inter-site distance ⇒ `δw ≈ 0.17`; typical payload gap `E|a_r − a_s| ≈ 0.7`
⇒ a jitter-limited reader error of `≈ 0.12` MAE. `E|t| ≈ 0.45` under (C2) ⇒ the blank floor.

| # | quantity | registered prediction | pass/fail meaning |
|---|---|---|---|
| P1 | `n_live` (modal), byte ledger | 5 live; 5456 B full / 100 B launder / **54.56×** (modal, ≥ 7 of 9 seeds) | identical to Appendix P.4 ⇒ same operating point |
| P2 | admissible coverage | **≥ 40 %** of attempted slots, **≥ 24 queries** at every seed | below ⇒ **K1 kill** |
| P3 | blank store | **−0.45 ± 0.15** | context for `S` |
| P4 | arg-min launder `settle_deleted` | **≤ −0.10 (provable)**; point estimate −0.30 ± 0.08 | a value > −0.10 ⇒ construction bug (**F4**) |
| P5 | `knn2_idw_+0B` | **≤ −0.10 (provable)**; point estimate −0.22 ± 0.06 | as P4 |
| P6 | best +0 B reader (name) | **`bary3_+0B`** | if a 2-row reader wins, (C2) is not doing what it claims |
| P7 | best +0 B reader (value) | **−0.12 ± 0.05** | feeds `S` |
| P8 | **FB4 `S(triad)`** | **0.70 ± 0.15 ⇒ NOT saturated (`< 0.95`)** ⇒ **family VALID** | `S ≥ 0.95` **and** `sub ≥ attn − 2 SE` ⇒ **K2 kill: family struck** |
| P9 | CLU `full` (measured, ⛔ not banked for this family) | **−0.42 ± 0.10** | — |
| P10 | CLU raw-table margin (`full − best raw +0 B reader`) | **−0.30 ± 0.10, i.e. < 0 by ≥ 2 SE** | > 0 by ≥ 2 SE ⇒ our own store beats the table at a second family (a finding, reported loudly) |
| P11 | rivals' raw-table margins | **0 of 6 positive**; every margin < 0 by ≥ 2 SE; range **[−0.55, −0.10]** | any margin > 0 by ≥ 2 SE ⇒ **F3: the headline is contradicted at a second family** |
| P12 | rescue gate (`full > blank + 2 SE`) | rescued: `{deltanet, gdn, gdn2, mamba2}`; **not** rescued: `ttt_mlp`; `ttt_linear` undeclared | ⛔ a margin against a non-rescued arm is not quotable |
| P13 | `S` of `aggregate` under the SAME extended reader set (declared **secondary**, §5) | **0.67 ± 0.12**, still `< 0.95` ⇒ family 1 stays valid, but its +0 B margins get **more negative** | `S(aggregate) ≥ 0.95` ⇒ ⚠ a paper-wide finding to hand to the Hub, not a family-2 verdict |

## 5. Declared secondary measurement (registered now, so it cannot be a post-hoc addition)
The same extended reader set (`knn3_*`, `bary2/3_+0B`) is measured on **`aggregate@base`, 9 seeds,
store-only (no rival fits)**, as an *instrument-consistency* check: a referee who accepts `bary3` as a
legitimate +0 B reader for family 2 will ask why family 1's reader set stops at two rows.
⛔ This is **not** a re-registration of any family-1 number and no family-1 number is restated from it;
it is reported to the Hub as a consequence list with an owner.

## 6. Kill conditions (any one ⇒ stop, report the failure as the deliverable)
- **K1** coverage below P2 at any seed.
- **K2** FB4 saturation (P8's fail branch).
- **K3** fewer than 3 live items at any seed (no triple exists).
- **K4** the identical-φ assertion or the byte-ledger integer identity fails on any cell.
⛔ A second family design after a kill requires its own dated prereg section, with this one kept.

## 7. Seed plan, aggregation, provenance
- **n = 9 on every arm including the CLU**: seeds `0…8`, the same seeds family 1's nine-seed column uses.
- Aggregation: sample sd (`ddof = 1`), `SE = sd/√9`; margins **paired per seed** before averaging.
- Tuning symmetry: the **full F3 grid** for every rival arm —
  `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`, TTT arms additionally `b ∈ {1, 16}`,
  400 outer steps, best-of-grid **on the fit split**, plus the held-out `f3_val` selection carried beside it.
  A 2000-step re-check on the winners' sub-grid is **declared optional** and labelled if run.
- Byte ledger per arm, **dtype declared per row** (`float32`, 4 B/float, for every arm and every table).
- Fit streams come from different seeds (the F2a guard), never the eval cell.
