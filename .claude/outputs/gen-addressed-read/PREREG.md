# PREREG — `gen-addressed-read` · the arm `x0`
**Filed:** 2026-09-16, **BEFORE** any scored number of this spoke was produced.
**Agent:** experiment-engineer · **Pin:** `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e` ·
**Branch:** `agent/experiment-engineer/gen-addressed-read` · **Worktree:** `../CHLU-addrread`

---

## ⭐ DIAL DECLARATION (echoed from the task file, verbatim in substance)
- **Dial:** ⭐ **addressing alone** — a read that stops at the addressed launch point, spending **zero** settling steps.
- **Laundering control:** `store_removed` on every row, **plus** the matched-bytes lookup floor (charter §C).
- **Falsifies:** that addressing alone carries the generation table.
- **Does NOT falsify:** losing to `knn_proto`, an unbounded real-valued channel (metric-native-ceiling theorem).

## ⛔⛔ §0. WHAT THE ARM IS, AND THE NAME IT MUST NEVER BE GIVEN
`x0` = **request → address-book posterior argmax over the written atoms → decode**.
⭐ It is an **ADDRESSED READ**. ⛔ It is **NEVER** described as dynamics, as settling, as a landscape read, or as
"the dynamical read at zero steps". It spends **zero steps**, uses **no coupling**, and has **no energy trace**.
Mechanically: `gen_landscape.launch_state(land, book, requests, launch_mode="clamp_atom")` → `x0`, then
`read_settled(land, x0, "state")` → the frozen decoder → the hash-asserted external classifier. No `settle_chain`
call appears anywhere on this arm's path, and §4's assertion A4 proves it by construction.

---

## §1. The rows this spoke will score
Two byte points (rungs), four rows each. ⭐ **One arm; the other three rows per rung are the charter §C floors and
controls, ⛔ not second arms.**

| label | what it is | role |
|---|---|---|
| `x0` (rung A) | posterior-**argmax** atom per well, concatenated | ⭐⭐ **THE HEADLINE ROW — MATCHED BYTES** |
| `x0/A_atom_mean_lookup` | posterior-**mean** over the same atoms, same book | ⭐ charter §C.1 floor — **the Advisor's named rival** |
| `x0/A_row_lookup` | the single best-scoring **whole stored train row** (one atom index for ALL wells) | ⭐ charter §C.1 floor, the *harder* of the two: it removes the per-well recombination and nothing else |
| `x0/A_atoms_shuffled` | atoms permuted **independently per well**, book unchanged | charter §C.2 landscape-removed control |
| `x0/B`, `x0/B_atom_mean_lookup`, `x0/B_row_lookup`, `x0/B_atoms_shuffled` | the same four at rung B | ⛔ **DECLARED LARGER — 8× the bytes**, never quoted as matched |

⛔ **No ladder, no tuning, no settling, no temperature, no second mechanism.** Every knob is the pin's default
(`GenDynamicalReadConfig`), and the rungs are `gen-dynamical-read`'s own pre-registered A/B.

## §2. The bar — `dynread_verdict`'s vocabulary, ⛔ and no other
Computed by `exp_gen_cell.dynread_verdict(mans, arm="x0")` (the pinned function, called not edited), with
`m = DYNREAD_MARGIN = 0.020` fixed at the pin and ⛔ not re-derived here.
- **Headline metric:** `external composition-B aggregate` (unweighted mean over the 6 factors of top-1 agreement
  between the external classifier's reading of the decoded image and the **requested** factor value), 72 held-out
  requests, 3 seeds (0, 1, 2), the shipped stratified split, `n_fit` 32,256 / `n_val` 512 / `n_B` 72 / `n_MS` 198.
- **Ceiling (a TIE is available here and nowhere else):** `knn_proto` `0.8650 ± 0.0122`.
- **Like-for-like — each must be BEATEN OUTRIGHT (Δ > 0, 3/3 agreeing signs, mean±sd disjoint); a tie here is a LOSS:**
  `store_removed` `0.1204 ± 0.0131` · `ebm_compose` `0.1937 ± 0.0182` · `lace` **`0.3117 ± 0.0153`**
  *(⛔ the CELL harness row; ⛔ NOT the `0.3241 ± 0.0165` baselines-instrument row)* · ⭐ `clu` **`0.5949 ± 0.0082`**
  — **the codebook read, at the SAME 524,288 B. This is the matched-bytes comparison the spoke exists to score.**
- **Plus, from §1 and charter §C.1:** `x0/<rung>_atom_mean_lookup` and `x0/<rung>_row_lookup`, both at **exactly**
  the rung's own bytes, both like-for-like, both must be **BEATEN OUTRIGHT**.
- ⛔ **`WIN` needs `Δ > m`.** A `+0.001` with three agreeing signs is a **TIE**, never a WIN.

## §3. ⭐⭐ THE BYTE LEDGER — the crux, predicted here and ASSERTED IN CODE
⛔ Not assumed: `n_bytes()` is measured on the **live objects** and the run **aborts** on mismatch.

**The claim under test:** the addressed read touches `store` and `book` only. `launch_state` reads `land.store` and
`book`; it never reads `land.coupling`, `land.request_strength` or `land.coupling_strength`. **If that is true, the
arm's ledger is STORE ONLY — half the dynamical read's, and EXACTLY `clu`'s at rung A.**

| quantity | rung A (predicted) | rung B (predicted) | `clu` (published) |
|---|---|---|---|
| atoms / well `N` | 64 | 512 | `M` = 64 modes |
| **store bytes** `N·D·4`, `D = 2048` | **524,288** | 4,194,304 | **524,288** |
| addressing head `(K·N + Σ_f V_f·K·N)·4`, `K = 5`, `Σ_f V_f = 57` | **74,240** | 593,920 | **74,240** |
| coupling `r(K−1)D·4` | 524,288 (`r`=16) — ⭐ **PREDICTED NOT NEEDED** | 4,194,304 (`r`=128) — **NOT NEEDED** | n/a (163,840 excluded, R26.3) |
| **⭐ DEPLOYED TOTAL for `x0`** | **524,288 B store + 74,240 B head** | 4,194,304 + 593,920 | **524,288 B store + 74,240 B head** |
| multiple of the 524,288 B budget | ⭐ **1.00×** | **8.00×** | 1.00× |

⭐ **P1: at rung A the addressed read's ledger equals `clu`'s FIELD FOR FIELD** — same store bytes, same addressing-head
bytes, same formula. ⭐ **P2: the coupling is not needed and its 524,288 B leave the ledger** (the dynamical read
carried 1,048,576 B = 2.00× for the same rung).

## §4. The assertions that make §3 a fact rather than a claim
Each ⛔ **aborts the run** on failure; each is recorded in the manifest.
- **A1 — measured ledger.** `store.n_bytes()` == `N·D·4` == the rung's `budget_bytes`; at A, == `clu`'s measured
  `deployed_bytes` (524,288) **read from the same manifest**, ⛔ not from a literal.
- **A2 — addressing-head parity.** the atom book's measured bytes == `(K·N + Σ_f V_f·K·N)·4`; at A, == `clu`'s
  measured `addressing_head.bytes` (74,240) read from the same manifest.
- **A3 — ⭐⭐ THE COUPLING IS NOT NEEDED, PROVED BIT-FOR-BIT.** `x0` is recomputed on three landscapes that differ
  only in the coupling — (i) the fitted low-rank coupling at the certificate's λ, (ii) `coupling=None`,
  (iii) λ = 0 — and additionally at β = 0. **All four must be bit-identical** (`np.array_equal`). ⛔ If any differ,
  P2 is FALSE, the ledger is 2× at rung A, "matched bytes" is withdrawn, and the headline row is re-declared.
- **A4 — zero steps.** the row records `settling_steps = 0`, `energy_trace = None`, `coupling_used = False`;
  a guard asserts `settle_chain` is not on this arm's call path and that the arm's own config snapshot has no
  step budget. ⛔ A row that cannot prove it spent zero steps is NOT RUN.
- **A5 — same cell.** per seed, the decoder param hash + recipe (`v1_capacity_schedule`), external-classifier param
  hash, encoder param hash, feature-cache sha256, store checkpoint, split mode and `n_fit/n_val/n_B/n_MS` are
  asserted **bit-identical** to the rows already on the manifest. ⛔ An arm scored on a cell of its own is not on
  this table.
- **A6 — replication of the diagnostic.** `x0` is asserted **bit-identical** to the `x0` array
  `gen-read-cell-arm` persisted in `settled_states_s{S}_{rung}_B_full.npz`. ⛔ Reported either way; ⛔ a mismatch
  means the wiring differs from the diagnostic's and no number here may be compared with it.

## §5. ⭐ THE PREDICTIONS — committed before measuring
⭐ P3–P5 are **replication** predictions: `x0` is the *same quantity* `gen-read-cell-arm` measured as a wiring
diagnostic (its `launch_diagnostic.json`), so a correct wiring must reproduce it **to the bit**, not merely closely.
P6–P8 are **new** and are the reason this spoke can fail.

| # | quantity | prediction | derivation |
|---|---|---|---|
| P1 | rung-A ledger | store 524,288 B + head 74,240 B, **1.00×** | §3 arithmetic, `D`=2048, `K`=5, `N`=64, `Σ_f V_f`=57 |
| P2 | coupling needed? | **NO** — bit-identical `x0` without it | `launch_state`'s body reads `land.store` and `book` only |
| P3 | `x0` rung A, composition-B | **0.5008 ± 0.0498**, per-seed 0.4329 / 0.5185 / 0.5509 | bit-replication of the diagnostic |
| P4 | `x0` rung B, composition-B | **0.6852 ± 0.0173**, per-seed 0.6667 / 0.7083 / 0.6806 | bit-replication of the diagnostic |
| P5 | distinct factor tuples of 72 | A **18.67 ± 2.36** (17/17/22) · B **25.33 ± 2.62** (23/24/29) | bit-replication of the diagnostic |
| ⭐ P6 | ⭐⭐ **MATCHED-BYTES VERDICT: `x0`(A) vs `clu`** | **LOSS**, Δ = **−0.0941**, 3/3 negative (−0.157 / −0.088 / −0.037), intervals disjoint | 0.5008 − 0.5949; \|Δ\| > m = 0.020 |
| P7 | `x0/A_atom_mean_lookup` | ⭐ **≤ 0.20 aggregate and ≤ 5 distinct tuples of 72** ⇒ `x0` **BEATS** it outright | the naive-Bayes posterior over 64 atoms is near-flat (mean confidence 0.012–0.067 measured by `gen-read-cell-arm`), so a posterior-weighted mean is ≈ the well's global atom mean ⇒ a near-constant output, near chance (0.1236) |
| P8 | `x0/A_row_lookup` | **0.20 – 0.40 aggregate**; `x0` **BEATS** it outright | 64 stored rows against 480,000 configurations: the best whole row matches the request on some factors but cannot compose; the gap to `x0` **is** the per-well recombination |
| P9 | `x0/A_atoms_shuffled` | **collapses to ≈ chance (0.1236 ± 0.03)** | a permuted atom table makes the book's index point at unrelated content |
| P10 | other like-for-like at rung A | `store_removed`, `ebm_compose`, `lace` **all BEATEN OUTRIGHT**; `knn_proto` **LOSS** (Δ = −0.3642) | from P3 against §2's published rows |
| P11 | overall | **like-for-like all beaten: NO · PASS: NO** | P6 alone forces it |

## §6. ⛔⛔ WHAT WOULD PROVE THIS SPOKE WRONG — the falsifiers, committed in advance
- **F1 (⭐ the charter §C floor).** `x0`(A) does **not** beat a matched-bytes lookup — `atom_mean_lookup` **or**
  `row_lookup` — beyond seed spread. ⇒ **the store is decorative, the arm does not count, and it is reported as
  such.** ⛔ Reported either way, and ⛔ never re-described as a win.
- **F2.** `x0`(A) does not beat `store_removed`. ⇒ the arm is the null.
- **F3.** A3 fails: the coupling changes `x0`. ⇒ **P1/P2 are FALSE**, the rung-A ledger is 1,048,576 B = 2.00×,
  "matched bytes" is **withdrawn**, and the headline row is re-declared as a 2× row.
- **F4 (the prediction that would make this a win).** `x0`(A) **beats** `clu` at matched bytes beyond seed spread.
  ⇒ P6 is refuted and addressing alone carries the generation table. ⭐ Predicted **NOT** to happen.
- **F5.** A6 fails: `x0` differs from the diagnostic's persisted `x0`. ⇒ the wiring differs; ⛔ nothing here is
  comparable with `gen-read-cell-arm` and no verdict is written.
- **F6.** `x0/A_atoms_shuffled` does **not** collapse. ⇒ the output does not depend on what is written; the arm is
  not reading a store at all.

## §7. ⛔⛔ THE REPORTING RULES THIS SPOKE BINDS ITSELF TO
1. ⭐⭐ **The headline is the MATCHED-BYTES row, and it leads.** ⛔ `0.6852` is the **8× row**. Leading with it,
   or quoting it against `0.5949` without the byte multiple attached **on the same line**, is forbidden.
2. ⛔ The arm is an **addressed read**. ⛔ Never dynamics, never settling, never "zero-step dynamics".
3. ⛔ Losing to `knn_proto` is the **metric-native-ceiling theorem**, not news, and does not falsify the dial claim.
4. ⭐ Every accuracy ships with its **distinct-factor-tuple count** of 72.
5. ⛔ If it loses, **it loses, and it is reported as the FLOOR for target 1.**

## §8. Cost, declared in advance
⭐ CPU only (`JAX_PLATFORMS=cpu`), main venv. ⛔ **No GPU** — this is a floor measurement, not a headline run
(charter §D). ⛔ No full suite. Predicted ≈ 20–35 min wall clock for 3 seeds × 8 rows; the five rival rows cost
**zero** (they are copied from `gen-read-cell-arm`'s manifests, ⛔ not re-run).
