# PREREG-hub-composition-1 — the ONE scored run this hub would spend, if it spends any
**Written 2026-09-16, BEFORE any scored run. ⛔ 0 of 3 scored runs spent at the time of writing.**
Hub `hub/composition` · branch `hub/composition` · base `8e36bc9` (tree `345244bb…` = the bundle's
`pin_tree`, i.e. pin `4125fce7…`). Charter §A.1: *"Every SCORED run … is pre-registered first … BEFORE
it runs."*

> ⛔ **READ THIS FIRST: my standing RECOMMENDATION is that this run is NOT spent on this machine.**
> See §6. The file exists so that (a) the rule is obeyed if the Head overrules me, and (b) the Mac can
> execute it verbatim, at 3 seeds, which this box cannot.

---

## 1. DIAL DECLARATION (protocol §7)
- **Dial:** compositional generation, end to end, scored on IMAGES by the external classifier.
- **Laundering control:** `store_removed`, on every row. Plus, for the addressed read, its own §C.1
  lookup floors (`atom_mean_lookup`, `row_lookup`) and §C.2 landscape-removed control (`atoms_shuffled`).
- **Falsifies the claim:** the arm fails to beat `knn_proto`'s **14.33 ± 4.78** distinct valid novel
  tuples of 72, or fails to reach the target's bar of **≥ 60**.
- **Does NOT falsify the claim:** losing to an unbounded-channel arithmetic baseline on a metric-native
  protocol (metric-native-ceiling theorem, R27.1) — that is a known confirmation, not news. Absolute
  levels below the **decoder ceiling of 67.67 ± 2.49** likewise.

## 2. THE RUN
```
python -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon \
    --stage cell --seed 0 --seed 1 --seed 2 --arm x0 --out <DIR>
python -m chlu.experiments.exp_gen_cell --stage aggregate --out <DIR>
```
`x0` is the **addressed read** added to the pin by `gen-addressed-read`: per well, the address book's
posterior-**argmax** stored atom, concatenated; **zero settling steps**. It is on the pin, it is
`ARMS`-adjacent by design (`X0_ARM = "x0"`, deliberately not in `ARMS`), and — verified from
`gen-read-cell-arm/runs/cell_results.json` — ⭐ **it has never been run on this table.** It emits 8 rows:
2 rungs × {`full`, `atom_mean_lookup`, `row_lookup`, `atoms_shuffled`}.
⛔ No other arm is requested: every other row already exists, frozen and Mac-produced, and re-running one
would break comparability with the rival rows for nothing.

## 3. THE BARS — declared now, ⛔ never changed after a result is seen (§A.3)
The headline statistic is **distinct VALID NOVEL tuples of 72 on B**, which on B is the identity
`= 72 × exact_match` (proved, LOG 19:33Z: all 72 targets distinct, and all 72 novel against every one of
the 479,928 train rows).

| # | bar | value | direction predicted | what it would mean |
|---|---|---|---|---|
| B1 | ⭐ **the target** | `x0` ≥ **60** | ⛔ **I predict FAIL** | falsifier 2 met |
| B2 | beat the rival ceiling | `x0` > `knn_proto` = **14.33 ± 4.78** | ⛔ **I predict FAIL** | falsifier 1 met |
| B3 | beat the CLU codebook row | `x0` > `clu` = **0.00 ± 0.00** | ⭐ **I predict PASS at rung B, FAIL/TIE at rung A** | the addressed read is the better read |
| B4 | §C.2 landscape-removed | `x0/A_atoms_shuffled` collapses | ⭐ predict PASS (collapse) | the store is load-bearing |
| B5 | §C.1 lookup floor | `x0` > `x0/A_row_lookup` | ⭐ predict PASS | not a whole-row lookup |
| B6 | §C.1 mean floor | `x0` > `x0/A_atom_mean_lookup` | ⭐ predict PASS | not a posterior mean |

### ⭐ MY QUANTITATIVE PREDICTION, committed before the run
From the `val` measurements at LOG 19:42Z (⛔ `val` is the selection set; B never entered):
`x0` rung A **5/512 = 0.0098**, rung B **28/512 = 0.0547**, `clu` **8/512 = 0.0156**, ceiling **0.9590**.
B is HARDER than `val` (held-out combinations), so I predict on B, of 72:
- **`x0` rung A: 0–2** (point estimate **0**)
- **`x0` rung B: 0–6** (point estimate **2**)
- ⛔ **Both bars B1 and B2 FAIL.** ⭐ A pre-registered prediction that survives is evidence; one that
  fails is a finding. This one is made to be checkable either way.

## 4. WHY I PREDICT FAILURE — the mechanism, measured, before the run
1. ⛔ **The store cannot round-trip its own training data.** Given the TRUE feature and allowed to
   address itself, the deployed 524,288 B store reads back **10/512 = 1.95 %** exact on `val` against a
   **95.90 %** decoder ceiling (rel-err 0.3486). Any request-only read must read OUT of this store, so
   ~2 % is a **cap**, not an estimate.
2. ⛔ **Addressing is not the lever.** An oracle that sees the true feature and picks the best atom per
   well beats the address book by **+2 items of 512** (rung A) and **+1** (rung B). No address book can
   buy more than that.
3. ⛔ **The obstacle is representational**: even under oracle selection, rel-err is **0.43** (rung A) /
   **0.31** (rung B).
⇒ The bar of 60/72 = **83 %** is ~43× above a cap measured at ~2 %, on **seen** data.

## 5. FLAG PROVENANCE (protocol, mandatory)
| item | value |
|---|---|
| commit | `8e36bc9` (tree `345244bbd40a29a7842357342e32e528890e0670` = `pin_tree`) |
| seeds | 0, 1, 2 (⛔ on THIS box, seed 0 only — see §6) |
| arms | `x0` only |
| store | LOADED from `gen-read-compose/runs/shapes3d_recon/seed_{s}/store_s{s}.pkl`, ⛔ not re-trained |
| decoder | `v1_capacity_schedule`, hash `01ef91f75eeaae7b…`, asserted per arm per seed |
| classifier | `gen-external-classifier` seed 0 `c8741ac417d88ebf…` (978,873 params), gate `PASS` |
| grouping | `per_seed`, re-derived; logged *"reproduces the plan BIT-IDENTICALLY"* on seed 0 |
| `n_fit` / `n_val` | 32,256 / 512 (shipped; ⛔ not raised — that is the ladder's business) |
| landscape | `atom_selection=kmeanspp`, `bandwidth_frac=1.0`, `nn_median`, `address_smoothing=1.0` |
| x0 specifics | `launch_mode=clamp_atom`, `read_mode=state`, `clamp_threshold=0.5`, **settling steps = 0** |
| budgets | A = 64 atoms/well / rank 16 / 524,288 B · B = 512 / 128 / 4,194,304 B |
| re-encode gate | ⚠ **must run with cache coverage shortfall = 0**; on this box that is seed 0 ONLY |
| env | Windows AMD64, Python 3.11.16, jax/jaxlib 0.9.0 CPU, numpy 2.4.1, `PYTHONUTF8=1`, 5 threads |
| h5py | 3.16.0, installed `--target` into `$OUT/pylibs` with `--no-deps`; ⛔ shared venv unmodified |

## 6. ⛔⛔ MY RECOMMENDATION: **DO NOT SPEND THIS SCORED RUN ON THE WINDOWS BOX.**
1. ⛔ **Seeds 1 and 2 cannot run here at all.** Their feature caches are `scope="used"` (9,288 rows) and
   the cell needs 33,039; **32,395 / 32,437 rows are missing** and producing them means re-encoding on a
   machine whose encoder does not reproduce the Mac cache (51.0 ULP on 0.977 % of elements). That is the
   hard constraint in LAUNCH-COMMON §4 and I will not cross it.
2. ⇒ A scored run here is **1 seed**. It would carry no seed spread, and the charter requires spread on
   every claim. ⛔ **It would consume 1 of 3 runs that are shared across machines and return less than
   the Mac can get for the same cost** — where seed 0's cache is full and seeds 1–2's are native.
3. ⭐ **And the two falsifiers this hub exists to settle no longer need it** (LOG 19:33Z): falsifiers 1
   and 2 are answered NOT MET from frozen artefacts, and falsifier 4 (`store_removed` collapses to 1
   distinct tuple, balacc exactly 0.5000) is answered PASS the same way.
⭐ **Recommendation: hand this PREREG to the Mac and run it there at 3 seeds. On this box, spend zero.**

## 7. Amendments
*(appended, dated, never edited — §A.3)*

### AMENDMENT 1 — 2026-09-16, after the independent replication, ⛔ BEFORE any scored run
*(appended, not edited; §A.3. ⛔ No bar in §3 is changed — 0 scored runs have been spent, so no result
has been seen against any of them. This amendment adds an ARTEFACT requirement and one new bar.)*

**Why.** The replication spoke objected, correctly, that `exp_gen_cell._record` persists per-render
factor tuples **only for `dynread`** (`persist_to` is passed on that branch alone). ⇒ for `clu`,
`knn_proto`, `lace` and `ebm_compose` the **WEAK** reading of "valid" — *the produced tuple lands in the
held-out cell at all*, rather than *equals the request* — is **not computable after the fact**, and it is
the reading under which `knn_proto` might look very different (it scores four factors at 1.000 and
`shape` at 0.986, and fails almost only on `scale`).

**A1.1 — ARTEFACT REQUIREMENT.** The scored run MUST persist, for **every arm**, the external
classifier's per-render factor tuple on both request sets. ⭐ Cheapest compliant form: pass `persist_to`
on every `_record` branch, ⛔ not only `dynread`. ⚠ It changes **no number**; it only stops the run
throwing away the evidence. ⛔ It is an edit to the frozen scoring harness and therefore needs the same
ruling as QUESTION 2 — ⛔ **do not make it unilaterally.** If the ruling is no, the run still proceeds and
the weak reading is reported **NOT MEASURABLE**, ⛔ never as zero.

**A1.2 — NEW BAR B7 (weak reading), declared before any result.**
`x0` distinct tuples that **land in the held-out cell AND are novel**, of 72.
- ⭐ Measured already on the frozen `dynread` rows: **0/72 items land in the cell**, on 12/12 rows — so
  for the atom-read family the strict and weak readings **agree at zero**.
- ⭐ **Prediction for `x0`: 0–8** (point estimate **1**).
- ⛔ `knn_proto`'s weak count is **UNKNOWN** and may be large. ⭐ **If `x0`'s weak count beats its strict
  count but still loses to `knn_proto`'s, falsifier 1 is still NOT MET** — the weak reading is reported
  beside the strict one, ⛔ it does not replace it, and ⛔ the headline stays the strict one whichever is
  kinder. Declared here, before either number exists.

**A1.3 — corrected reference values** (replication, 25-draw re-runs; see LOG 20:46:38Z):
permuted-residual control **115.6 ± 7.7** (⛔ not my single draw of 95); Gaussian control **382.4 ± 11.1**
(⛔ not 393). Population vs sample sd: ceiling **67.67 ± 2.49** (pop) = **± 3.06** (sample);
`knn_proto` **14.33 ± 4.78** (pop) = **± 5.86** (sample). ⛔ None of §3's bars moves: B1 (≥ 60) and B2
(> 14.33) are unaffected at either spread convention.
