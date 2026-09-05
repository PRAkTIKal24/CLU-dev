# bprime-dividend-family-2 — experiment-engineer report

**Task + acceptance criterion:** build, protocol-validate and audit a **second B′ dividend family** —
prereg-first, n = 9 on every arm including the CLU, full F3 grid, complete control battery — and report it
in registry-rider format. **Status: done** — family built, **protocol-validation gate CLEARED**, full audit run at n = 9 on all six rivals + the CLU + the complete control battery, plus the 5× budget re-check and the declared secondary. Suite green (1910 passed / 0 failed).

## ⚠ FIRST 10 LINES — things the Hub must act on
1. ⛔ **ENVIRONMENT INCIDENT, 2026-08-18 21:00–21:25 BST: the machine's file I/O intermittently returned
   `ETIMEDOUT` on cold reads**, which made `git worktree add` (and every `git checkout`, and reads of ~535
   loose objects including `main`'s own root tree) fail with `fatal: mmap failed: Operation timed out`.
   It **recovered on its own** by 21:25 and `main` is intact (`git rev-parse main^{tree}` = `73db1153…`).
   ⚠ Any other spoke that saw a "corrupt repo" in that window saw the same transient fault, not damage.
   My worktree was populated by hand during the outage and then **verified**: `git write-tree` reproduced
   `main`'s root tree hash exactly and my index is byte-identical to `main`'s index (evidence in §8).
2. **Reconciliation list (needs an owner):** the declared secondary in PREREG §5 — the same extended +0 B
   reader set measured on **`aggregate`** — bears on a *registered family-1 number* (`S = 0.5068`). See **§6**.

---

# 0. DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — **audit width**. No new dial; a second family for the existing audit.
- **Laundering control:** the B′ battery itself (matched-byte table launder · +0 B substitute audit ·
  same-keys null · blank store · rescue gate), plus this family's own declared three-row readers.
- **Falsifies the claim:** a dividend quoted without its launder column; a positive raw-table margin
  (falsifier F3) would contradict the paper's headline at a second family.
- **Does NOT falsify:** the family failing protocol validation (that is the deliverable, task rule 2);
  losing to a classical +0 B reader (metric-native-ceiling theorem).

---

# 1. THE PRE-REGISTRATION (filed before any arm trained; §1 of this file, per the task)

⭐ **Prereg-first is verifiable in git, not asserted:** commit `d1883df`
*"[experiment-engineer] pre-register B′'s second dividend family (`triad`)"* contains **only** the
registration (`TRIAD_PREREG` in `chlu/experiments/memory_gym.py`) — no query builder, no reader, no
wiring — and precedes the implementation commit `f783c43` and every artifact below.

The full text as filed is reproduced verbatim:

---

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


---


# 2. WHAT HAPPENED, IN ORDER (prereg → gate → audit → secondary)

| step | artifact | status |
|---|---|---|
| 1. prereg committed | commit `d1883df` (registration only, no implementation) | ✅ before any store was written |
| 2. family implemented + tests | commits `f783c43`, `50b5574`, `f6c9d03` | ✅ 23 new tests green |
| 3. **protocol-validation gate** | `.claude/outputs/bprime-dividend-family-2/fb4_triad/exp_fb4_gate_metrics.json` | ✅ **CLEARS** — family released for scoring |
| 4. the audit, 9 seeds, full F3 | `.claude/outputs/bprime-dividend-family-2/audit_triad_f3_400/exp_bprime_rivals_metrics.json` | ✅ 9/9 cells, 0 degenerate |
| 5. 2000-step re-check (declared optional) | `.claude/outputs/bprime-dividend-family-2/audit_triad_f3_2000/` | see §4.4 |
| 6. declared secondary (PREREG §5) | `.claude/outputs/bprime-dividend-family-2/aggregate_extended_readers.json` | ✅ 9 seeds |

---

# 3. ⭐ THE PROTOCOL-VALIDATION GATE — the family is VALID

`chlu/eval/fb4_gate.py`, rule frozen 2026-07-31, constants untouched. **9 seeds** (the rule's own registered
convention is 3; the 9-seed figure is primary).

| quantity | measured (n = 9, mean ± SE) |
|---|---|
| blank store | **−0.2576 ± 0.0177** |
| best **+0 B** reader (`knn3_mean_+0B`, modal 9/9 in the gate run) | **−0.1886 ± 0.0142** |
| attention arm (**+4 B**, one fitted temperature) | **−0.1738 ± 0.0181** |
| **`S(triad) = (sub − blank)/(M − blank)`** | **0.2676** |
| ceiling leg (`S ≥ 0.95`) | **False** |
| attention leg (`sub ≥ attn − 2 SE`, paired SE = 0.0130) | True |
| **verdict** | **`CLEARS` — NOT substitute-saturated ⇒ the family is protocol-VALID and released for scoring** |

⚠ **Registered prediction P8 was `S = 0.70 ± 0.15`; measured `0.2676` — a MISS, in the safe direction**
(the family is *further* from saturation than registered). The reason is measured, not guessed: I priced
`blank ≈ −0.45` and the best reader at `≈ −0.12`; the store's blank read is in fact much better
(**−0.2576**) and the best reader worse (**−0.1886**), so the numerator `sub − blank` is only `0.069` of a
`0.258` gap. ⛔ The *verdict* — the family is not substitute-saturated — is what the gate registers, and it
holds by a factor of 3.6 in `S`.

⚠ The attention arm reads **0.0148 better than the best +0 B reader (1.1 SE)** ⇒ under the paper's own
"reads no worse than" discipline this is a **tie**, not an attention win.

---

# 4. THE AUDIT — 9 seeds, full F3 grid, every arm, the complete control battery

## 4.0 The family's own signature: readers separate by the CARDINALITY of the read
All eight readers of the **same** table, same queries, same scorer, `neg_mae` (higher is better), n = 9:

| reader | rows used | bytes | score (mean ± SE) |
|---|---|---|---|
| `settle_deleted` (arg-min launder) | 1 | table | **−0.4759 ± 0.0381** |
| `knn2_mean_+0B` | 2 | table + 0 B | **−0.4837 ± 0.0335** |
| `knn2_idw_+0B` | 2 | table + 0 B | **−0.4824 ± 0.0330** |
| `knn3_mean_+0B` ⭐ the strongest | 3 | table + 0 B | **−0.1886 ± 0.0142** |
| `knn3_idw_+0B` | 3 | table + 0 B | −0.2237 ± 0.0167 |
| `bary3_+0B` (the registered favourite) | 3 | table + 0 B | −0.2861 ± 0.0219 |
| `raw_table_mean_+0B` (constant predictor) | all | table + 0 B | −0.2090 ± 0.0153 |
| `same_keys_null` (same keys, permuted payloads) | 1 | table | −0.6102 ± 0.0330 |

⭐ **The 2-row → 3-row step is +0.2951 ± 0.0362 (8.2 SE).** That is the instrument doing exactly what it was
built to do, and it is *measured*, not asserted: **every** two-row reader sits below the family's provable
`−0.1` bound (they sit at ≈ −0.48, far below it), and the three-row readers clear it by a wide margin.
⚠ **P6 MISS, and it is a finding:** the registered favourite `bary3_+0B` **loses to a plain 3-NN mean**
(−0.2861 vs −0.1886, 3.8 SE). The least-squares barycentric solve amplifies the query jitter it is inverting;
the unweighted mean is the more robust estimator at `σ_q = 0.15`. The paper should quote `knn3_mean_+0B`
as this family's strongest +0 B reader.

## 4.1 The audit table (registry-rider format)
Metric `neg_mae`; **RAW margin** = `full − (best +0 B reader of a RAW (address,payload) table at the same
bytes)` — the strictly stronger control, and the column the headline is read off. `dtype = float32` (4 B per
float) on every row of every ledger.

| arm | `d_head` | state B | matched table B | param B | **full** | own projected table | **+0 B margin (own table)** | **RAW-table margin** | blank | rescued? |
|---|---|---|---|---|---|---|---|---|---|---|
| `ttt_linear` | 29 | 5220 | 5104 | 5592 | −0.4289 ± 0.0453 | −0.2720 ± 0.0193 | −0.1798 ± 0.0465 | **−0.2475 ± 0.0422 (5.9 SE)** | −0.4735 | ⛔ **no** |
| `ttt_mlp` | 12 | 5376 | 5376 | 5736 | −0.4187 ± 0.0501 | −0.2717 ± 0.0194 | −0.1671 ± 0.0418 | **−0.2373 ± 0.0584 (4.1 SE)** | −0.3641 | ⛔ **no** |
| `deltanet` | 36 | 5184 | 5184 | 9956 | −0.2624 ± 0.0236 | −0.3698 ± 0.0339 | +0.0038 ± 0.0079 *(tie)* | **−0.0810 ± 0.0282 (2.9 SE)** | −0.3359 | ✅ yes |
| `gdn` | 36 | 5184 | 5184 | 9956 | −0.2684 ± 0.0166 | −0.5400 ± 0.0520 | −0.0232 ± 0.0229 *(tie)* | **−0.0869 ± 0.0262 (3.3 SE)** | −0.7455 | ✅ yes |
| `gdn2` | 36 | 5184 | 5184 | 9956 | −0.2710 ± 0.0206 | −0.5437 ± 0.1073 | **+0.0499 ± 0.0170 (2.9 SE)** | **−0.0896 ± 0.0275 (3.3 SE)** | −0.8236 | ✅ yes |
| `mamba2` | 36 | 5184 | 5184 | 8380 | −0.2766 ± 0.0269 | −0.6757 ± 0.1904 | −0.0004 ± 0.0271 *(tie)* | **−0.0952 ± 0.0343 (2.8 SE)** | −1.0648 | ✅ yes |
| **CLU (⛔ MEASURED, not banked)** | — | **5456** | **100** (54.56×) | — | **−0.4649 ± 0.0672** | −0.4759 ± 0.0381 | −0.2835 ± 0.0703 | **−0.2835 ± 0.0703 (4.0 SE)** | −0.2576 ± 0.0177 | ⛔ **no** |

**Selection stability.** The strongest raw reader is `knn3_mean_+0B` on 5 of 9 seeds and
`raw_table_mean_+0B` (the constant predictor) on 4 — the margin is taken **per seed against that seed's own
best reader**, so the column is a max over readers, never a favourable pick. Grid winners across all
(arm, seed) cells: `lr = 1e-2` 26×, `3.16e-3` 18×, `1e-3` 10× (⛔ **no cell selects an lr below 1e-3**, the
same finding F3 reported on family 1); `wd = 0` 33×, `wd = 0.1` 21×; TTT `b = 16` 50×, `b = 1` 4×.

## 4.2 ⭐ THE HEADLINE, REPLICATED AT A SECOND FAMILY
> **At byte-matched state on `triad`, 0 of 6 rivals beat a zero-extra-byte reader of a raw table holding the
> same bytes — every margin is negative by ≥ 2.8 SE (−0.0810 to −0.2475), under the full F3 grid at nine
> seeds. Our own store loses the same way, by 4.0 SE (−0.2835), and it is the one arm that also fails the
> rescue gate.**

⚠ **Two-sided, in our own voice.** The CLU sits **0.2073 ± 0.0608 BELOW its own blank store**
(`lift < 0` ⇒ **not rescued**) — the same below-blank pattern §4.1.1 reports for our store on `aggregate`,
now at a second family. ⛔ No margin *against* the CLU or against `ttt_linear`/`ttt_mlp` is quotable on this
family (rescue-gate discipline: those three arms are not informative here).

## 4.3 The asymmetry family 1 found, replicated and sharpened
Against its **own projected** table (the registered P5 control) `gdn2` is **positive at 2.9 SE**
(+0.0499 ± 0.0170) — on `aggregate` at n = 9 the same arm was +0.047 ± 0.028, i.e. < 2 SE. Against the
**raw** table at the same bytes it is **−0.0896 ± 0.0275**. The gap between the two controls is
`p5_vs_raw_gap = +0.3623 ± 0.1111` for `gdn2` (+0.0906 to +0.4943 across arms, positive on 6 of 6).
⭐ **This is the second independent measurement of the paper's methodological point:** a launder read
through the memory's own projections is a *weaker* control than the same bytes spent on raw rows, and an
audit that stops at the projected control can report a dividend that the raw control removes.

## 4.4 Every control fired, on every cell
- **matched-byte table launder** ✅ (integer identity green on 9/9: `full 5456 B`, `launder 100 B`, ratio
  **54.56×** on 8 seeds; on seed 8 the admission gate admits a sixth item ⇒ `120 B` and **45.60×** —
  ⚠ **the modal-value rule applies exactly as in Appendix P.4; no single figure is *the* nine-seed ledger**).
- **+0 B substitute audit** ✅ (8 readers, §4.0) · **same-keys null** ✅ (−0.6102 ± 0.0330; every arm above it)
- **blank store** ✅ (per arm, above) · **rescue gate** ✅ scored for **every** arm *including the CLU* (§4.1)
- **identical φ** ✅ asserted in code across all arms (`phi_id = 283f6ef36159080d`, `phi_bytes = 0`)
- **`table_is_lossless` = true** on all six rival arms (their matched tables hold the whole 10-token stream).

---

# 5. THE PRE-REGISTRATION SCORECARD — 7 HIT / 13, computed by the harness
(`triad_scorecard` in the run artifact; P8 from the gate run, P13 from the secondary.)

| # | quantity | registered (before any run) | measured | verdict |
|---|---|---|---|---|
| P1 | ledger + live items | 5 live, 5456/100 B, **54.56×** modal | 5 live on 8/9 (6 on s8); 54.56× on 8/9, 45.60× on s8 | **HIT** |
| P2 | coverage ≥ 0.40, ≥ 24 queries/seed | — | min coverage **0.662**, min queries **53** | **HIT** |
| P3 | blank store | −0.45 ± 0.15 | **−0.2576** | MISS (blank better than priced) |
| P4a | arg-min ≤ −0.10 (**provable**) | ≤ −0.10 | worst seed **−0.3311** | **HIT** |
| P4b | arg-min point estimate | −0.30 ± 0.08 | −0.4759 | MISS (C2 bites harder) |
| P5 | `knn2_idw_+0B` | −0.22 ± 0.06 | −0.4824 | MISS (same cause) |
| P6 | strongest +0 B reader | `bary3_+0B` | **`knn3_mean_+0B`** | MISS — **a finding** (§4.0) |
| P7 | strongest +0 B reader value | −0.12 ± 0.05 | −0.1814 | MISS |
| P8 | **FB4 `S`** | 0.70 ± 0.15, `< 0.95` ⇒ valid | **0.2676**, `< 0.95` ⇒ **valid** | MISS on the value, **HIT on the verdict** |
| P9 | CLU `full` | −0.42 ± 0.10 | −0.4649 | **HIT** |
| P10 | CLU raw-table margin | −0.30 ± 0.10 | −0.2835 | **HIT** |
| P11 | rivals with a positive RAW margin | **0 of 6**, range [−0.55, −0.10] | **0 of 6**, range [−0.2475, −0.0810] | **HIT** |
| P12 | rescue gate | rescued ⊇ {deltanet, gdn, gdn2, mamba2}; `ttt_mlp` not | exactly that (+ `ttt_linear` and the CLU also not rescued) | **HIT** |
| P13 | `S(aggregate)` under the extended reader set | 0.67 ± 0.12 | **0.6559** | **HIT** |

⭐ **Every structural prediction held (P1, P2, P4a, P9–P13); every miss is a calibration miss in the
direction of "the two-row readers are worse than I priced", plus the one reader-ordering finding (P6).**
⛔ **Falsifier F3 did not fire** (no positive raw-table margin). Kill conditions K1–K4: none fired.

---

# 6. THE DECLARED SECONDARY (PREREG §5) — ⚠ RECONCILIATION ITEM, needs an owner
The same extended reader set measured on **`aggregate@base`, 9 seeds, store-only** (`.claude/scratch/
aggregate_extended_readers.py`; ⛔ the shipped `aggregate` code path is untouched — the extra readers are
evaluated outside it, and a regression test asserts the shipped reader set is bit-identical).

| reader (n = 9, `aggregate@base`) | score |
|---|---|
| `settle_deleted` | −0.3810 ± 0.0345 |
| `knn2_mean_+0B` / `knn2_idw_+0B` | −0.1552 ± 0.0245 / **−0.1508 ± 0.0231** |
| `bary2_+0B` | −0.1540 ± 0.0232 |
| **`bary3_+0B`** | **−0.1454 ± 0.0164** |
| `knn3_idw_+0B` / `knn3_mean_+0B` | −0.2283 ± 0.0200 / −0.2946 ± 0.0251 |
| blank store | **−0.3906** ⭐ reproduces the paper's published nine-seed blank (−0.3906 ± 0.0124) digit-for-digit |

- **`S(aggregate)` with the shipped reader set, n = 9: 0.6228.** With the extended set: **0.6559.**
- The strongest +0 B reader on family 1 becomes **`bary3_+0B` on 5 of 9 seeds** (2-NN IDW on the other 4),
  improving the control by **+0.0129**.
- ⭐ **Family 1 remains protocol-VALID under the stronger reader set** (0.6559 ≪ 0.95). The consequence is in
  the paper's own favour: a *stronger* +0 B control makes every rival's family-1 `+0 B` margin ~0.013 **more
  negative**, it does not rescue anyone.
- ⚠ **What the Hub must decide (this is the reconciliation item):** `draft-r6` quotes `S = 0.5068` for
  `aggregate` (the **3-seed protocol-validation run**, Appendix A.2). My nine-seed recomputation with the
  **same shipped readers** gives **0.6228** — consistent with Appendix N.1's own warning that the two are
  not the same estimate at the same `n`, but a referee who accepts `bary3` for family 2 will ask why family
  1's reader set stops at two rows. **Owner needed:** paper-writer (a sentence in N.1/N.5 declaring the
  reader set is per-family and why) — ⛔ I have re-registered nothing.

---

# 7. FLAGS — flag-provenance table (mandatory; every number above traces to a row here)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/bprime-dividend-family-2` off **`main @ 3c1dbaa`** |
| code commit for every run below | **`f6c9d03`** (tree verified identical to `main` + my 4 commits) |
| venv | ⭐ **the main repo venv reused** (`/Users/user/Desktop/CHLU/.venv`, `PYTHONPATH=<worktree>`) — no worktree `uv sync`, so no package drift (w6 lesson) |
| family / arm | `triad` / `base` (⛔ not in either harness's default plan; reached with `--families triad`) |
| seeds | **0–8 (n = 9) on every arm, the CLU included**; blank store at `seed + 991`; fit streams `seed + 101, 102`; held-out val stream `seed + 103`; query rng `seed + 7717`; per-rival key offset `seed*1000 + 7(i+1)` |
| store (non-default vs `GymConfig`) | `family=triad`, `capacity=6`, `consolidate_every=2`, `clu_overrides={stage_admission: True}` — ⭐ **identical to `aggregate@base`**; `n_offer=8`, `budget=6`, `reference_capacity=6`, `atoms_per_item=32` ⇒ **192 atoms**, `d=4`, `m=1`, `n_spectator=0`, `ball_radius=1.0`, `σ_q=0.15`, `payload_tol=0.1`, `α=0.05`, atom width 0.3 |
| read | dissipative velocity-Verlet, `dt=0.05`, 400 steps at `γ_address=0.05` then 800 at `γ_read=0.02`, learned-Newtonian kinetic term, **T = 0** (no Langevin) |
| write | staged admission; masked local optimisation, 300 Adam steps @ `3e-3`, `wd=1e-4`, `σ_addr=0.25`, `σ_pay=0.6`, hinge 0.15, barrier 0.2 |
| query law (registered) | `triad_dist_mult=2.0`, `n_query_per_triple=8`, `triad_w_lo/hi=0.25/0.45`, `triad_max_draws=32`, `triad_outside_pair_hull=True` (rule C2 **ON** — the pre-registered fallback was never needed) |
| tuning grid (all 6 rivals) | **full F3**: `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`, TTT arms also `b ∈ {1, 16}` ⇒ 24 points/TTT arm, 12/delta arm; **400 outer steps**; best-of-grid **on the fit split**; `optax.adam` at `wd=0`, decoupled `optax.adamw` at `wd=0.1`; `f3_lite_control` and held-out `f3_val` selections carried beside the primary |
| iso-state budget | 1364 float32 = **5456 B**; head widths 29 / 12 / 36 / 36 / 36 / 36 |
| **dtype** | **`float32`, 4 B per float, on every ledger row of every arm** (state, matched table, params) |
| aggregation | sample sd (`ddof=1`), `SE = sd/√9`; margins **paired per seed** before averaging; reader chosen per seed by arg-max |
| gate run | `chlu.experiments.exp_fb4_gate --families triad --seeds 0…8`, thresholds `0.95` / 2 SE **unchanged** |
| declared NOT-RUN | the byte-frontier column (`--no-frontier`: `overload` is family-1's instrument, not this family's); `f3_val`-selected audit tables are computed but not quoted above |
| wall clock | gate ≈ 95 s total; audit ≈ 165 s/cell × 9 ≈ 25 min (machine load ≈ 18, three jobs concurrent) |

---

# 8. GIT FOOTPRINT

| commit | subject | files |
|---|---|---|
| `d1883df` | pre-register B′'s second dividend family (`triad`) | `chlu/experiments/memory_gym.py` (registration only) |
| `f783c43` | implement `triad`, B′'s second dividend family | `chlu/experiments/memory_gym.py`, `chlu/experiments/exp_memory_gym.py`, `chlu/eval/dividend.py`, `tests/test_bprime_triad_family.py` |
| `50b5574` | wire `triad` through the audit and the protocol-validation gate | `chlu/experiments/exp_bprime_rivals.py`, `chlu/experiments/exp_fb4_gate.py`, `tests/test_bprime_triad_family.py` |
| `f6c9d03` | declare `triad` in the gym's family-set test | `tests/test_memory_gym.py` |

`git diff --stat main..HEAD` = **7 files, +697 / −24**. ⛔ Not pushed, not merged; branch left for review.
**Worktree:** `/Users/user/Desktop/CHLU-bprime-family2` (declared, mine alone).
⚠ **One collision to disclose:** while the shell's cwd was silently reset to the shared checkout, one edit
(`tests/test_memory_gym.py`) landed in `/Users/user/Desktop/CHLU` instead of my worktree. It was caught the
same minute, reverted with `git checkout -- tests/test_memory_gym.py`, and the shared checkout verified
clean (`git status --short` empty, still on `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`).
No other file was touched there and nothing was committed there.

---

# 4.4bis THE 5× BUDGET RE-CHECK (declared optional in the prereg; it ran)
Sub-grid containing every 400-step winner (`lr ∈ {1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`, TTT also
`b ∈ {1,16}`), **2000 outer steps**, same nine seeds, same store path.
Artifact: `.claude/outputs/bprime-dividend-family-2/audit_triad_f3_2000/exp_bprime_rivals_metrics.json`.

| arm | `full` 400 → 2000 | **RAW-table margin** 400 → 2000 | fit-split loss 400 → 2000 | rescued at 2000? |
|---|---|---|---|---|
| `ttt_linear` | −0.4289 → −0.3545 ± 0.0533 | **−0.2475 → −0.1731 ± 0.0505 (3.4 SE)** | 0.1823 → 0.1754 (−3.8 %) | ⛔ no |
| `ttt_mlp` | −0.4187 → −0.3771 ± 0.0296 | **−0.2373 → −0.1957 ± 0.0380 (5.1 SE)** | 0.1632 → **0.0918 (−43.8 %)** | ⛔ no |
| `deltanet` | −0.2624 → −0.2659 ± 0.0235 | −0.0810 → **−0.0845 ± 0.0285 (3.0 SE)** | 0.1958 → 0.1957 | ✅ |
| `gdn` | −0.2684 → −0.2649 ± 0.0213 | −0.0869 → **−0.0835 ± 0.0288 (2.9 SE)** | 0.1959 → 0.1957 | ✅ |
| `gdn2` | −0.2710 → −0.2714 ± 0.0198 | −0.0896 → **−0.0900 ± 0.0268 (3.4 SE)** | 0.1958 → 0.1957 | ✅ |
| `mamba2` | −0.2766 → −0.2754 ± 0.0267 | −0.0952 → **−0.0940 ± 0.0339 (2.8 SE)** | 0.1986 → 0.1986 | ✅ |

⭐ **0 of 6 at 5× the outer budget as well; every margin still ≥ 2.8 SE below zero.**
⭐ **`ttt_mlp` cuts its fit-split loss by 43.8 % and moves its eval metric by +0.042** — the same
fit-to-eval generalisation gap the paper measures at family 1 and in the SSD block ablation, now at a third
place. ⚠ The two TTT arms improve most and remain the two arms that fail the rescue gate; ⛔ their margins
stay unquotable under the rescue-gate discipline, and the headline does not rest on them.

---

# 9. SUITE, AND WHAT ELSE I VERIFIED
- **Full suite on the branch: `1910 passed, 0 failed` (54 m 56 s, `f6c9d03`, main venv).** Log:
  `/tmp/f2_suite2.log`. ⚠ An earlier full run stopped at one **fixture** assertion
  (`test_memory_gym.py::test_families_and_arms_are_declared` hard-codes the family tuple) — fixed in
  `f6c9d03` and re-verified; that run's other 1133 tests had passed.
- **23 new tests** (`tests/test_bprime_triad_family.py`): C1 and C2 per query on four seeds; ⭐ the family's
  **bound asserted directly** (a 21-point convex sweep over the two nearest rows, per query, must never come
  within `payload_tol`); the coverage kill condition; the store operating point being identical to
  `aggregate@base`; the barycentric reader exact at a noiseless barycentre; ⛔ **the regression test that the
  `aggregate` reader set is bit-identical with and without the family flag**; both harness plans; and the
  prereg scorecard's HIT/MISS + F3-falsifier logic.
- `ruff check chlu tests`: clean.
- **Independent reproduction of a published number:** my `aggregate` blank at nine seeds is **−0.3906**,
  matching §4.1.1 / Appendix A.1e's **−0.3906 ± 0.0124** digit-for-digit — evidence the harness path I ran
  is the shipped one.

---

# 10. ONE-PAGE SUMMARY (verifiable against the registries)
1. **A second dividend family exists, is protocol-VALID, and is measured.** `triad` — the barycentric
   mixture of **three** neighbouring stored payloads, constructed to sit outside the two-nearest hull.
   FB4 `S = 0.2676` (`CLEARS`; `aggregate` = 0.5068 registered / 0.6228 recomputed at n = 9).
2. **It is a genuinely different opening, and the difference is measured, not asserted:** on the *same*
   table, three-row readers beat two-row readers by **+0.2951 ± 0.0362 (8.2 SE)**. `aggregate` bounds
   arg-min; `triad` bounds **every two-row convex reader** — provably, and the test suite asserts it.
3. **The headline replicates:** **0 of 6 rivals** beat a **+0 B** reader of a raw table at the same bytes;
   margins **−0.0810 to −0.2475**, every one **≥ 2.8 SE** below zero, at 400 steps **and** at 2000.
4. **Our own store loses too, harder, and fails its own rescue gate** (−0.2835 ± 0.0703; lift over its own
   blank store **−0.2073 ± 0.0608**). The audit is two-sided at the second family as well.
5. **The projected-vs-raw asymmetry replicates and sharpens:** `gdn2` beats its *own projected* table by
   **+0.0499 ± 0.0170 (2.9 SE)** and loses to the *raw* table by **−0.0896 ± 0.0275**.
6. **Byte ledger:** `5456 B` state vs a `100 B` launder ⇒ **54.56× (modal, 8 of 9 seeds; 45.60× on the ninth)**
   — the same integer identity, green on 9/9, `float32` on every row.
7. **The paper's thinness sentence can now be revised**: two rival families audited against **two** surviving
   synthetic families, with the second one's construction rule strictly stronger than the first's.
8. ⚠ **One reconciliation item with no owner yet:** the +0 B reader set is now **per family** (§6).

---

# 11. OPEN QUESTIONS / FOLLOW-UPS / RISKS
1. ⚠ **`bary3` lost to `knn3_mean` (P6 MISS).** The registered "strongest reader" is not the strongest; the
   least-squares barycentric solve amplifies query jitter. **Risk if unstated:** a referee builds
   `knn3_mean` themselves and finds our declared favourite was the weaker one. It is stated here and the
   audit already scores the max over readers per seed, so no number changes.
2. ⚠ **The CLU is below its own blank store on `triad`** (as on `aggregate`). This is now a **two-family**
   pattern for our store and deserves a sentence in Limitations rather than a footnote.
3. **`triad`'s bound is a reader bound, not a memory bound.** Nothing in the construction stops a *settling*
   memory from answering; the store simply does not. Whether a settle can ever land on a 3-well barycentre
   is a physics question this family now makes measurable (a natural next probe, not run here).
4. **Not run, declared:** the byte-frontier column on `triad` (family 1's instrument); a `triad` cell at the
   `tight` geometry; the GRU/SWA arms (outside the ruled arm set); an `S`-vs-`n_seeds` curve for the gate.
5. ⚠ **Environment:** the 21:00–21:25 I/O fault (§ first-10-lines) also stalled the suite and the runs to
   ~3× wall clock (load average 18 with three jobs). All numbers were produced after recovery.

---

# 12. PROPOSED HANDOVER UPDATES (for the Hub)
1. **New result to bank (registry-ready):** `triad`, B′'s second dividend family — FB4 `CLEARS` at
   `S = 0.2676`; **0 of 6 rivals** beat a raw-table `+0 B` reader at matched bytes (−0.0810 … −0.2475, all
   ≥ 2.8 SE, n = 9, replicated at 2000 steps); the CLU also loses (−0.2835 ± 0.0703) **and fails its own
   rescue gate**. Artifacts under `.claude/outputs/bprime-dividend-family-2/`.
2. **`draft-r6` edits this enables** (⛔ paper-writer's, not mine): Appendix N gains **N.6 `triad`** with its
   four-rule check; §6's thinness sentence goes from *"one surviving synthetic family"* to **two**; §4 gains
   the second family's audit table; Appendix I gains this prereg's scorecard (7 HIT / 13, with the misses).
3. **Reconciliation item needing an owner (§6):** the `+0 B` reader set is now **per family**. `S = 0.5068`
   for `aggregate` is a **3-seed** figure; the same shipped readers at nine seeds give **0.6228**, and with
   the extended (three-row) readers **0.6559** — family 1 stays valid either way, and its rival margins get
   *more* negative. Needs one declarative sentence in N.1/N.5. **Suggested owner: paper-writer.**
4. **Known-issue candidate:** the shell's cwd can silently reset to the shared checkout mid-session; one edit
   landed there and was reverted within the minute (§8). Worth a line in the protocol's §3 (always prefix
   `cd <worktree> &&`).
5. **Environment note for §7 Known Issues:** a machine-wide transient I/O fault (2026-08-18 21:00–21:25 BST)
   made ~7 % of cold file reads return `ETIMEDOUT`, which presents as `fatal: mmap failed: Operation timed
   out` from git and looks exactly like repository corruption. It is not: `main`'s tree verified intact
   afterwards. Recipe if it recurs: wait, re-read, and verify with `git rev-parse main^{tree}`.
