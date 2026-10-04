# PREREG — `hub/refusal`, scored run **1**

**Filed:** 2026-09-16, before any C-set, A7-set, A7D-set, A-set or B-set row was scored by
this hub. **Branch:** `hub/refusal` @ `8e36bc9` (tree = pin `4125fce7…`). **Machine:**
FORGIS_14, Windows AMD64, CPU, `OMP/MKL/OPENBLAS_NUM_THREADS=4`.
**Author:** `autonomous-hub` for `hub/refusal`. **Scored-run count:** this is **run 1 of
the hub's attempts**; ⛔ its position in the *target's* 3-run budget is **NOT KNOWN TO ME**
(COMMON §1c) and this file is not executed until a human states it.

⛔ **Amendments are appended and dated; ⛔ nothing above an AMENDMENT line is ever edited.**

---

## 0. DIAL DECLARATION (protocol §7)

- **Dial:** **admission** — refusal of an inadmissible request, from the landscape alone.
- **Laundering control:** **three**, all mandatory, all reported beside every number:
  1. **`zero_step`** — the same criterion evaluated at the **launch** `x0` instead of the
     settled state. If the launch refuses as well as the settled state, the dynamics are
     decorative and the claim is "the address book refuses", not "the landscape refuses".
  2. **`request_off`** (β = 0) — the same chain with the request energy removed. The
     criterion is *defined* from the request terms, so this is not a null by
     construction; what it measures is whether the settled state the criterion is
     evaluated **at** was reached by steering or by relaxing onto a mode.
  3. **`store_removed`** — §C.2's landscape-removed control (below, §6).
- **Falsifies the claim:** §5.
- **Does NOT falsify the claim:** (a) failing to beat a rival on the kind that rival was
  built for, *while beating it on the other*, is not the claim — the claim is **one
  criterion above both rivals on BOTH kinds**; (b) a chain that is NOT CONVERGED is a
  declared property of this arm (`still_descending` 0.72–0.79 at rung A, §E.4, inherited)
  and is handled by §3, not by moving a bar; (c) losing on the *generation* metric is the
  inherited `0.6852 → 0.2114` finding and is a different question (§E.4).

---

## 1. THE TARGET, restated as the number I will report

**One landscape-derived criterion that refuses BOTH `out_of_range` AND `duplicate_factor`
while accepting unseen-valid, at ONE threshold chosen on valid data only, with ZERO
invalid supervision, over 3 seeds.**

**The reported result is the PAIR at one threshold**, per kind:
`(reject-invalid rate, accept-unseen-valid rate)`, with AUROC as a **companion**.
⛔ AUROC alone is not the result.

### The bar, re-derived from `gen-val-split-fix/refusal_scores.json` (⛔ never the
`gen-arms-a7-wiring` file), 3-seed mean ± std, computed with the shipped `binary_auroc`:

| rival row | leg | pooled AUROC(C vs B) | A7-ctrl `out_of_range` (vs B) | A7-ctrl `duplicate_factor` (vs A7) |
|---|---|---|---|---|
| `knn` = `knn_proto` | historical_688128 | **0.845743 ± 0.0238** | 0.971193 ± 0.0070 | 0.310314 ± 0.0013 |
| `knn` = `knn_proto` | deployed_524288 | 0.835712 ± 0.0218 | **0.958333 ± 0.0048** | 0.309028 ± 0.0145 |
| `lace` = `ld_published` | historical_688128 | **0.809671 ± 0.0210** | 0.626672 ± 0.0441 | 0.784594 ± 0.0301 |
| `knn_compsel` (variant) | historical_688128 | 0.852173 ± 0.0221 | 0.973765 ± 0.0085 | 0.326132 ± 0.0373 |
| `lace/ode` (variant) | historical_688128 | 0.721965 ± 0.0627 | 0.444316 ± 0.1255 | **0.990612 ± 0.0033** |

The three numbers my task file names are the three bolded in columns 3–5 of the first
three rows. ⭐ **I bind myself to the STRICTEST reading**, which is stricter than the task
file's: to claim a win I must beat, per kind, the **best number any rival variant
achieves on that kind**, not just the named row —

- **`out_of_range`, A7-controlled:** beat **0.973765** (`knn_compsel`, historical leg).
  *(The task file's named 0.958333 and the `knn_proto` 0.971193 are both below it.)*
- **`duplicate_factor`, A7-controlled:** beat **0.990612** (`lace/ode`, historical leg).
  *(The named LACE row `ld_published` gives 0.784594; I will beat that too or say I did not.)*
- **pooled C-vs-B:** beat **0.852173** (`knn_compsel`).

⛔ I record now, in advance, that **`duplicate_factor` ≥ 0.9906 is a hard bar** and that a
result which beats kNN and `ld_published` on both kinds but does **not** clear `lace/ode`
on `duplicate_factor` is **a partial result, reported as such**, ⛔ never as "the refusal
win". ⭐ The unclaimed thing is *one* criterion above **both** rivals on **both** kinds;
`lace/ode` is above chance on one kind only (0.4443 on `out_of_range`), and `knn` on the
other (0.3103 on `duplicate_factor`) — ⚠ **both rivals are BELOW CHANCE on the kind they
do not cover**, which is the gap this hub exists to close.

---

## 2. THE CRITERION — declared in full, ⛔ no free parameter

For a request with terms `{(f_t, v_t)}`:

1. **Expert per term** (`scorer/refusal_lib.term_experts`):
   `W_t = logsoftmax_a[ (1−w)·log_cond[f_t][i] + w·log_cond[f_t][j] ] − log_prior`,
   with `(i, j, w) = gen_request._extrapolating_index(grid.values[f_t], v_t)` — **the
   shipped R8 rule, imported**. On-grid ⇒ `w = 0` ⇒ identity.
   ⭐ **MEASURED BEFORE FILING:** for a valid request this tensor is **bit-identical**
   (`max_abs_delta = 0.0`) to the shipped `gen_landscape.request_experts`'.
2. **Launch** `x0`: the address-book argmax clamp, `launch_mode="clamp_atom"` —
   `gen_landscape.launch_state` generalised from factors to terms, identical on valid.
3. **Settle**: `gen_landscape.settle_chain`, rung **A** (the byte-MATCHED rung: 64
   atoms/well, rank 16, store 524,288 B), **1000 steps** (`exp_gen_cell.DYNREAD_STEPS`),
   `step_size 0.2`, σ anneal 4.0 → 1.0, `T = 0`, Metropolis on, β = 1,
   λ = the coercivity certificate's `0.9 × λ*` (measured 0.04854, ⛔ never 1.0).
4. **Per-term residual at the settled state**, at the **reference** bandwidth
   `sigma_scale = 1.0` (⛔ the annealed trace is not differenceable — `settle_chain`'s own
   binding note): `r_t = E_t(x_settled) − E_store(x_settled)`, computed by calling the
   shipped `gen_landscape.request_energy` with a one-term tensor.

### ⭐⭐ PRIMARY CRITERION — the headline, and the only one that may be promoted

```
S_max(request) = max_t  r_t(x_settled)
```

**Higher = more refusable.** It is a function of the settled state's own energy
decomposition and nothing else: no label, no image, no feature, no invalid example, no
free parameter. ⛔ There is **nothing to tune**: `max` has no knob.

### Pre-registered COMPANIONS — reported always, ⛔ **never** promoted to headline

| # | statistic | why it is declared, in advance |
|---|---|---|
| C1 | `S_sum = Σ_t r_t` | ⛔ **A TERM COUNTER** — 7-term `duplicate_factor` vs 6-term valid. Declared rejected *now*, on structure, so that its number cannot later be adopted. |
| C2 | `S_mean = mean_t r_t` | term-count-normalised, but a violated term is diluted by five satisfied ones. |
| C3 | `E_total(x_settled)` | the plain settled energy — "the read's own settled energy", the most literal reading of the target. |
| C4 | `E_store(x_settled)` | the store term alone — does the state simply sit off the atoms? |
| C5 | `rel_tail_drop` | the **convergence** statistic, per request: exactly the `rel` that `convergence_report` thresholds, kept per chain. |
| C6 | `S_max` at the **launch** `x0` | ⭐ the `zero_step` laundering control. |
| C7 | `S_max` under **`request_off`** (β = 0) | the steering control. |

⛔ If a companion beats the primary on the scored sets, **the primary is still the
headline** and the companion is reported as an observation with "not pre-registered as the
headline" attached. That sentence is the difference between a result and a fishing trip.

---

## 3. THE THRESHOLD — the entire claim, so it is spelled out

- **Set:** the **`val`** split — 512 rows, the **repaired stratified** cut of
  `store_train` (`chlu.data.gen_rowsplit.stratified_fit_val`), realising every value of
  every factor (10/10/10/8/4/15, max per-value count ratio 1.091 at seed 0).
  ⛔ Not `A`, ⛔ not `B`, ⛔ never `C`, ⛔ never `A7`.
  ⭐ `val` is **seen-valid and not held out**, so choosing on it spends **zero** held-out
  budget, which is strictly more conservative than the rivals' own threshold set (`A`).
- **Rule:** `τ_seed = quantile(S_max over val, q)` with **`q = 0.95`**.
- **⭐ Provenance of `q`: it is NOT mine.** `q = 0.95` is the shipped
  `GenBaselinesConfig.refusal_valid_quantile` default — **the same constant, to the digit,
  that every rival's published threshold uses**, via `gen_knn.threshold_from_valid`.
  ⛔ I did not choose it, did not sweep it and will not change it.
- **Decision:** `refuse ⇔ S_max > τ_seed`, the shipped `gen_knn.refusal_rates` predicate
  (strict `>`), one threshold per seed, applied unchanged to **every** request set.
- **Convergence:** `still_descending` is expected on ~0.72–0.79 of rung-A chains (§E.4,
  inherited, ⛔ not re-derived). ⭐ **DECLARED NOW: a still-descending chain does NOT
  invalidate `S_max`, because `S_max` is evaluated at the state actually reached at the
  declared 1000-step budget, and that budget is the same for every request set and every
  arm.** `rel_tail_drop` is reported per set (C5) so a reader can see whether the two
  halves converged differently. ⛔ I will **not** add a convergence term to the primary
  criterion after seeing a C row; if convergence is used at all it is as companion C5.

**ZERO INVALID SUPERVISION, itemised:**
1. `q` is the harness's shipped constant, not chosen here.
2. `τ` is computed from `val`, which contains **no invalid record**.
3. The criterion's functional form was fixed and committed in this file, with its
   structural argument, before any invalid record was loaded.
4. `c_invalid.json` and `a7_redundant.json` have **not been opened** by this hub. They
   will be loaded only through the shipped `load_c_requests` /
   `load_registered_control_requests`, at scoring time, after this file exists.
5. The runner (`scorer/run_units.py`) **mechanically refuses** to touch `A/B/C/A7/A7D`
   without `--allow-scored` and a named `--prereg`, and every unit writes a ledger row.

---

## 4. WHAT WILL BE COMPUTED, and the protocol it is computed by

Per seed ∈ {0, 1, 2}, rung **A**, arms {`full`, `request_off`, `zero_step`}, sets
{`val`, `A`, `B`, `MS`, `C`, `A7`, `A7D`}. Scored **exactly** as the rivals are, by the
shipped `exp_gen_baselines.refusal_metrics` protocol, re-implemented nowhere:

- **AUROC**, `chlu.eval.gen_probe.binary_auroc` (rank form, ties averaged), positives = C.
- **A7-controlled per kind**, `control_map` read from `query_sets.json`:
  `out_of_range → base` (= set `B`, `CONTROL_BASE_SET`), `duplicate_factor → a7_redundant`
  (= set `A7`). ⛔ Reported **by kind**, never pooled into one AUROC (R27.4); the
  `n_c`-weighted mean ships as a companion only.
- **The pair at `τ`:** `refused_fraction` on `C` (by kind) and `1 − refused_fraction` on
  `B`; also on `A`, `MS`, `A7`, `A7D`, and on `val` (which must land at ≈ 1 − q = 0.05 by
  construction — a build check, not a result).
- **The rivals' pairs at their own published threshold** are recomputed from
  `gen-val-split-fix/refusal_scores.json`'s per-request scores, with `A` at `q = 0.95`,
  so the comparison is pair-to-pair and not AUROC-to-pair. ⭐ No new rival run: it is a
  JOIN on `ids`, which is what that file exists for.

**Seed spread:** every number is 3-seed mean ± std with the per-seed values printed.
"Beyond seed spread" is read as: **my mean − 1 std > rival mean + 1 std**, stated per kind.

---

## 5. FALSIFIERS — the measured outcomes that kill the claim

1. **F1.** `S_max` fails to beat **both** rivals on **both** kinds at one threshold over
   3 seeds, beyond seed spread ⇒ **no refusal win.** Reported as not solved.
2. **F2.** Accept-unseen-valid collapses: `B` accept rate at `τ` materially below the
   rivals' ⇒ the rejection was bought by refusing valid requests. ⭐ A refuser that
   refuses everything scores zero; the pair is one number.
3. **F3.** `A7` is refused at a rate like `C[duplicate_factor]`'s ⇒ the criterion is a
   term counter after all, the A7 control convicts it, and it is reported convicted.
4. **F4.** The `zero_step` (launch) control refuses as well as the settled state ⇒ **the
   dynamics are decorative**; the honest claim shrinks to "the address book refuses".
5. **F5 (§C.2 / task falsifier 4).** `store_removed` does **not** collapse ⇒ the
   landscape is decorative and the criterion is reported as such.
6. **F6.** The criterion is above chance on one kind and at/below chance on the other ⇒
   this is exactly the rivals' failure mode, and the "one criterion, both kinds" novelty
   is **absent**. ⛔ Two thresholds or two statistics, one per kind, is **not** a repair.

---

## 6. THE §C FLOOR CHECKS, declared with their controls

- **§C.1 matched-bytes lookup.** The kNN rival **is** the lookup (atom-mean / kNN on the
  same atoms at the same byte target). Beating it on both kinds is the floor check.
  Rung A's ledger is store 524,288 B — the deployed target — and is asserted on every run.
- **§C.2 landscape-removed.** `store_removed`: the criterion recomputed with the atom
  store's content destroyed. **Declared construction, fixed now:** the atoms of each well
  are **randomly permuted across wells' row assignments with a fixed seed**
  (`shuffle`, the charter's own wording "atoms zeroed or shuffled"), the address book and
  the request terms untouched. **Expected: the criterion collapses toward chance.**
  ⛔ If it does not, F5 fires.
- **§C.4 lookup-table test.** `settled_state_spread` / distance of the settled state to
  its nearest atom is reported. If the settled state sits ON its nearest atom, the arm is
  a lookup table and is reported as one.

---

## 7. PREDICTIONS — committed to before the measurement

| # | prediction | direction |
|---|---|---|
| P1 | `S_max` A7-controlled AUROC on `out_of_range` | **> 0.97** (beats every kNN variant) |
| P2 | `S_max` A7-controlled AUROC on `duplicate_factor` | **> 0.80** (beats `ld_published`'s 0.7846 and every kNN variant's ~0.31); ⚠ clearing `lace/ode`'s **0.9906** is the hard bar and I do **not** predict it — I predict `0.80–0.99` and will report exactly where it lands |
| P3 | `A7` refused fraction at `τ` | **within 2× of `B`'s** — the term-count invariance, measured |
| P4 | `S_max(x0)` (launch, C6) on `out_of_range` | above chance (the launch already sees the extrapolated expert) but **below** the settled value on `duplicate_factor` — the conflict is what the *dynamics* expose |
| P5 | `store_removed` | AUROC → 0.5 ± 0.1 on both kinds |
| P6 | `val` refused fraction at `τ` | **= 0.05 ± 0.01** by construction (a build check) |

⛔ P2's honesty note is load-bearing: **if `duplicate_factor` lands at 0.85, that is a
partial result and I will say the criterion does not beat `lace/ode` on that kind.**

---

## 8. COST LINE (declared in advance)

CPU, local, 4 threads, ~30 min of wall clock for the whole scored run (3 seeds × 3 arms ×
7 sets at rung A, ≈ 0.146 s/request/1000 steps measured). ⛔ No GPU. ⛔ No GCloud. ⛔ No
cluster. ⛔ Nothing that spends money. ⛔ Nothing re-encoded, nothing written to any
manifest-pinned path.

---

## AMENDMENTS
*(none yet — appended and dated below this line, never inserted above it)*

### AMENDMENT 1 — 2026-09-16, ⛔ **before any scored set was loaded or any result seen**
§6's `store_removed` construction, stated to the line so it cannot drift:
`scorer/refusal_lib.shuffled_store(store, seed)` permutes **each channel column of each
well's atom matrix independently** among the `N` atoms, RNG `default_rng(seed*9176 +
20260916)`. Consequences, all deliberate: `store_bytes` unchanged **to the byte** (the
control is byte-matched); every channel's marginal over the atoms is the **same multiset
of values**; ⛔ no atom is a train row any more, so `assert_atoms_are_train_rows` would
convict it — the landscape's joint content, and only that, is destroyed. The address
book, the bandwidths σ and the low-rank coupling are **untouched**: exactly ONE declared
quantity differs from the `full` arm. The divergence radius is measured against the
control's **own** atom radius, not the full arm's.

### AMENDMENT 2 — 2026-09-16, ⛔ before any scored set was loaded
The measured build facts, recorded so §2's configuration is not a claim but a reading —
seed 0, rung A: λ\* = `0.05393`, **λ used = `0.04854`** (capped by the coercivity
certificate at `0.9 λ*`, ⛔ never 1.0); byte ledger store `524,288` B + coupling
`524,288` B, **PREREG §3 match `True`**; σ per well `4.6343 5.0714 9.1827 6.3054 5.1006`;
grouping re-derivation reproduces the pinned plan **bit-identically**; `term_experts`
reproduces the shipped `request_experts` **bit-identically** on valid requests
(`max_abs_delta = 0.0`).
⚠ Also recorded: `_assert_cache_reencodes` **would have failed** on this box at
`max_ulp_delta = 51.0` (bound 1) on `0.977 %` of elements (bound 1 %) — charter declared
property #3 measured cross-ISA. It is called with `will_extend=False` and a written
reason through its own scoping API; ⛔ nothing is re-encoded and nothing is extended, and
`would_have_failed: true` is carried in every provenance file.

### AMENDMENT 3 — 2026-09-16T19:2x Z, ⛔ **before the probe it declares is run, and before any C / A7 / A / B row has been loaded**

**A FIT-DERIVED SYNTHETIC MECHANISM PROBE is declared here, with the rule that binds it.**

*Why.* Two blockers stand between this hub and its scored run: (i) the target-wide
scored-run count is not known to me (COMMON §1c), and (ii) **this box can only produce
seed 0** — seeds 1 and 2 would need ~23,480 rows re-encoded, which COMMON §4 forbids. The
mechanism question — *does one landscape criterion separate BOTH kinds of unsatisfiable
request from satisfiable ones?* — can still be answered at seed 0 without touching a
held-out row. Charter §A.1 permits exactly this: *"Iterate freely on MECHANISM. Every
SCORED run (**any number on the held-out sets A/B/C**) is pre-registered first."*

*Construction, fixed here.* Bases = the first **256 `fit` rows** of seed 0 (⛔ disjoint
from `val`, from `A`, from `B` and from the negatives by `select_baseline_rows`' own
assertion). ⛔ `c_invalid.json` and `a7_redundant.json` are **not opened**. Four sets,
each built from the SAME 256 bases with `default_rng(20260916)` choosing the factor:
| set | construction | what it stands in for |
|---|---|---|
| `syn_valid` | the base request, unmodified (6 terms) | valid |
| `syn_a7` | + a 7th term, same factor, **same value** | the `a7_redundant` control |
| `syn_dup` | + a 7th term, same factor, a **different grid value** | `duplicate_factor` |
| `syn_oor_{1,2,3}` | one term's value moved **1 / 2 / 3 grid steps beyond the end** of that factor's range | `out_of_range` at three unknown severities |
The `out_of_range` severity in the real C set is unknown to me, so **all three severities
are reported and none is selected**; there is no knob in `S_max` that a severity could
tune anyway.

*⛔⛔ THE BINDING RULE.* This probe is **mechanism only**. ⛔ **No threshold, no criterion,
no companion, no hyperparameter and no reported bar may be changed as a result of it.**
The threshold rule stays `quantile(S_max on val, q = 0.95)` with `q` the harness's shipped
constant; `S_max` stays the headline. ⭐ **If this hub subsequently changes the criterion
or the threshold rule after seeing this probe, the zero-supervision claim is VOID and
must be reported as void** — that sentence exists so the temptation has a stated price.
⭐ Its numbers are reported as **"fit-derived synthetic, seed 0, NOT the benchmark"** and
⛔ may never be quoted as the refusal result: fit rows are *seen*, so they make the valid
side easier than `B`, and a constructed negative is not a registered C record.

### AMENDMENT 4 — 2026-09-16T19:2x Z — the seed constraint, recorded against §1 and §5
Measured on disk: seed 0's feature cache is `scope="full"` (480,000 rows, full coverage,
⛔ nothing ever encoded); seeds 1 and 2 are `scope="used"` with **9,288** rows against the
**32,768** the landscape needs. On this box the encoder does not reproduce the cache
(81 float16 ULP on 0.488 % of elements at seed 1), so `gather_features`' enforced gate
aborts at the point of extension — correctly.
⛔ **Consequence, binding on §1 and §5:** falsifier F1's "3 seeds" **cannot be satisfied on
this machine**. Any run performed here is **single-seed** and is reported as single-seed;
⛔ it is never presented as the 3-seed result, and ⛔ no seed-spread claim is made from it.
⭐ The PREREG, the scorer and the ~30 min cost estimate transfer to the Mac unchanged.

### AMENDMENT 5 — 2026-09-16T19:36Z — ⚠ a SAMPLING confound in AMENDMENT 3's probe, fixed and declared before the re-run

AMENDMENT 3 fixed the probe's bases as "the first **256 `fit` rows**". That is wrong, and
the pin's own code says why: `stratified_fit_val` returns `fit = sort(pool \ val)`, and
`chlu/data/gen_rowsplit.py`'s module docstring states that on Shapes3D "a dataset row
index **is** a lexicographic factor-index tuple and a **sorted tail is a contiguous
corner of the factor grid**". **`fit_rows[:256]` is therefore a grid corner**, not a
sample of the fit half. Measured consequence: `AUROC(val vs syn_valid) = 0.7221` between
**two valid sets**, and mean `S_max` `1.9287` (val) against `0.6224` (the corner).

**The fix, declared before it runs.** The 256 bases become a **uniform random sample
without replacement** from `fit_rows`, `default_rng(20260916)` — the same RNG seed already
declared in AMENDMENT 3, drawn before the per-request factor choice so the factor draw is
unchanged in distribution. ⛔ Nothing else moves.

⛔ **This is NOT a change to the criterion, the threshold rule, the companions or the
bars** — AMENDMENT 3 forbids those four and only those four, and all four are untouched:
the headline is still `S_max`, the threshold is still `quantile(S_max on val, q = 0.95)`
with the harness's own `q`, the seven companions are unchanged, and the rival bars in §1
are unchanged. It is a defect in the *instrument* that generated a diagnostic, repaired by
citing the pin's own documented trap.
⭐ **Both readings are reported** — the corner-based one above and the repaired one — and
⛔ neither may be used to change `S_max`. The first probe's finding that **all four arms
agree to within ~0.05 AUROC** is independent of which bases were used and stands either way.

### AMENDMENT 6 — 2026-09-16T19:50Z — a RUNG-B diagnostic leg of the synthetic probe, declared before it runs

*Why.* The AMENDMENT-5 probe returns a near-null at rung **A** (64 atoms/well, σ per well
`4.63 … 9.18`). A bandwidth that broad over 64 atoms makes `E_p(x)[e^{W_t}]` close to its
prior value for most `x`, which would flatten every per-term residual regardless of
satisfiability. Rung **B** — 512 atoms/well, rank 128, 2000 steps — is a **shipped,
declared rung** of the same harness (`GenDynamicalReadConfig.budgets = ['A','B']`,
`exp_gen_cell.DYNREAD_STEPS`), ⛔ not a knob I am inventing. Whether the null is a property
of *the criterion* or of *rung A's coarseness* is a mechanism question and is answerable
with no held-out row.

*Scope, and the rule that binds it.* Arms `full` and `zero_step` only (the comparison that
matters), same bases, same RNG seed, same threshold rule, same statistic.
⛔⛔ **RUNG A REMAINS THE HEADLINE OF PREREG-1 WHATEVER RUNG B SAYS.** Rung A is the
**byte-MATCHED** rung (store 524,288 B = the deployed target); rung B is `DECLARED LARGER`
(4 MB budget) and is not byte-matched to the rivals, so a number from it could never be
the refusal result anyway. ⭐ If rung B looks better, that is a **recommendation for a
PREREG-2**, whose first line must state that its rung was chosen after seeing invalid
examples and whose **zero-supervision claim is therefore VOID**. ⛔ It is never a silent
promotion, and this paragraph exists so that it cannot become one.

### AMENDMENT 7 — 2026-09-16T19:52Z — THE LAUNDERING CURVE, declared before it runs

The AMENDMENT-5 probe gives two endpoints: `AUROC(syn_dup vs syn_a7)` = **0.7511** at the
launch and **0.5825** after 1000 settling steps. `scorer/laundering_curve.py` measures the
same statistic along a step ladder **fixed here before the run**:
`steps ∈ {0, 10, 25, 50, 100, 250, 500, 1000}`, rung A, same AMENDMENT-5 bases, same RNG
seed, same `S_max`, same threshold rule.
⛔ Only the step budget moves, and the ladder is declared, not searched.
⛔ MECHANISM ONLY: nothing in PREREG-1 changes because of it, and ⛔ a step budget other
than the shipped `DYNREAD_STEPS['A'] = 1000` may **not** become the headline — that would
be a hyperparameter chosen after seeing invalid examples, i.e. a **PREREG-2 with the
zero-supervision claim VOID**.
⭐ Its purpose is to turn "the settling launders the unsatisfiability" from an assertion
into a measured curve, and to say **where** the signal is lost.

### AMENDMENT 8 — 2026-09-16T20:20Z — the SIMPLEX-BLINDNESS measurement, declared before it runs

`scorer/simplex_blindness.py` turns §6.2's structural *argument* into a *measurement*. For
the same fit-derived request, on-grid and at extrapolation severities
`{0, 1, 2, 3, 6, 12}` grid steps beyond the end, it reports: the induced expert's
`max_a P(atom | term)` and entropy (does extrapolation blow the distribution up, or does it
stay a sane point on the simplex?); the **perturbed term's own residual** `r_t` at the
launch and at the settled state, and the fraction of those that are **negative, i.e.
SATISFIED**; and the distance from the settled state to that expert's **argmax atom**
against the distance to the nearest atom at all.
⭐ **The prediction it tests, committed here:** an off-grid term's own residual stays
**negative** (satisfied) and its expert stays a sane distribution with bounded entropy,
even at 12 grid steps out — because the extrapolated conditional is renormalised over
stored atoms and there is no direction in which it can leave them.
⛔ MECHANISM ONLY. FIT-DERIVED SYNTHETIC, seed 0. ⛔ Nothing in PREREG-1 changes because of
it, and ⛔ no number from it is the refusal result. ⭐ A result that contradicts the
prediction is a **finding**, which is why it is written down first.

### AMENDMENT 9 — 2026-09-16T20:20Z — the §C.2 floor check applied **at the launch**, declared before it runs

The AMENDMENT-5/6 probes put the only real signal at the **launch** (`AUROC(dup vs a7)`
`0.7511` rung A, `0.8124` rung B). §C.2 requires a landscape-removed control **for the
quantity being claimed**, and the control I had ran the full 1000/2000 steps — so it
floors the *settled* statistic, ⛔ not the launch one. `settle_unit`'s arms are made to
**compose**, and `store_removed_zero_step` is added: the shuffled-atom landscape evaluated
at the launch. `request_off_zero_step` is added on the same mechanism.
⭐ **The question it settles:** is the launch-time `duplicate_factor` signal coming from the
**stored atoms**, or is it an artefact of the address book alone? If
`store_removed_zero_step` matches `zero_step`, then even the launch-time signal is not
landscape-derived, and the PREREG-2 recommendation in the report must say so.
⛔ MECHANISM ONLY, FIT-DERIVED SYNTHETIC, seed 0. ⛔ Nothing in PREREG-1 changes.
