# RESULTS LEDGER — `hub/generation-table-2` (RELAUNCH, 2026-09-17)

**Machine:** FORGIS_14, Windows 11 Pro AMD64, CPU only (R3 provenance on every row).
**Branch:** `hub/generation-table-2` @ `2e65d96`, base `945a3e9` (`base/iter2`).
⛔ This file is the relaunch's ledger. Iteration 1's `RESULTS-LEDGER.md` is untouched beside it.

---

## SCORED RUNS — **1 of 3 spent.** *"scored 1/3 — seed 0 only, 3-seed pending full caches."*

| # | UTC | what | pre-registered in | result |
|---|---|---|---|---|
| **1** | 2026-09-17 15:44–16:09 | **seed 0, `eval_b`.** Leg 1 = the pinned `exp_gen_cell --stage cell --seed 0`, run **UNEDITED** (814 s, 5 default arms → 7 rows). Leg 2 = `relaunch/scored/run_arm.py` (277 s): the arm, 7 controls, `clu_linear`, and independent recomputations of `clu` and `knn_proto`. **Same rows, same decoder, same classifier, same sitting ⇒ ONE run.** | `PREREG-hub-generation-table-2.md`, body frozen 13:15Z, prediction frozen 15:45Z | ⭐ **TIE vs `knn_proto` under R1** — see below |
| — | 15:43 | ⛔ **An aborted launch that is NOT a scored run.** I hit `env_hub.sh`'s documented import trap (`python -m` puts CWD at `sys.path[0]`), imported the **bundle root's unpatched** `exp_gen_cell`, and the enforcing re-encode gate raised at `exp_gen_cell.py:1780`. That line is **before `gather`, before the decoder ceiling and before the arm loop**: **zero `eval_b` rows were read and zero held-out numbers were computed.** Declared, not hidden. | — | aborted at the preflight; **0 numbers** |

### The scored result, in one block
```
eval_b, 72 images, seed 0, FORGIS_14 (Windows 11 AMD64, CPU)
  knn_proto  0.8588   57 distinct tuples/72   475,136 B   (the ceiling)
  THE ARM    0.8495   51 distinct tuples/72   454,336 B   (0.956 x the budget)
  D = -0.0093 ;  |D| = 0.0093 <= m = 0.020   =>   TIE under R1.  BANKED.  NOT a win.
```
⚠ **One cell of 72 images resolves `1/(72*6) = 0.0023` per step: Δ = −0.0093 is FOUR IMAGES.**
⭐ The powered companion is the **42-corner held-out DEV CONFIRM: Δ = −0.0131, 95 % CI
[−0.0243, −0.0019], 19/42.** Both instruments agree in sign and magnitude; both land inside the tie band.

### ⭐ The parity row (the addendum's labelled ledger row — ⛔ a label, not a spend)
| | Mac-published, 3 seeds | this box, seed 0 | bound | verdict |
|---|---|---|---|---|
| `clu` composition-B | 0.5949 ± 0.0082 | **0.5903** | within the published sd | ⭐ **PASS** |
| `clu` distinct tuples | 9.00 ± 0.82 | **9** | ±1 | ⭐ **PASS** |
| `clu` probe companion (harness `rival_regression_check`) | 0.6609848484848484 | **0.6609848484848484** | tolerance 0.05 | ⭐⭐ **`bit_identical: true`, Δ = 0.0** |
| `knn_proto` probe companion | 0.8762626262626263 | 0.8787878787878788 | 0.05 | ⭐ PASS (+0.0025) |
| ⚠ `lace` probe companion | 0.7077020202020202 | 0.6136363636363636 | 0.05 | ⛔ **`regressed: true`, Δ −0.0941** — flagged by the harness, reported, ⛔ not absorbed, ⛔ not mine to fix (`lace_selection_source` still points at the broken-split selection; charter declared property #2) |

### ⛔ THE NO-ENCODE GUARANTEE, checkable
Both legs printed their coverage before gathering a row: **`shortfall = 0 of 33,038`** (leg 1) and
**`33,038 / 33,038 rows cached`** (leg 2), cache `scope="full"`, 480,000 rows.
⭐ The harness's own words: *"NOT ENFORCED (enforced=false): run_seed coverage: 0 of 33038 needed rows
uncached … THE RE-ENCODE CHECK WOULD HAVE ABORTED THIS RUN (ulp) and was NOT enforced"*, at 51 ULP against
a bound of 1. ⛔ `ULP_BOUND` and `FRAC_BOUND` untouched; ⛔ nothing encoded; ⛔ `will_extend=False` never
passed from my code with a shortfall > 0; ⛔ `SKIP_LOAD` / `SKIP_TEST` never used.

### ⭐ REPLICATION (charter §A.5) — *"unreplicated = not a result"*
Leg 2 was re-run from a **clean detached worktree of the committed branch**
(`CHLU-hub-generation-table-2-replica` @ `2e65d96`, `git status` clean, `chlu.__file__` asserted to be the
replica's), from the same `scored_spec.json`.
⭐⭐ **BIT-IDENTICAL on all 13 rows** — every aggregate to 6 dp, every per-factor accuracy, every
distinct-tuple count, the byte ledger (454,336) and the §C.4 ratio (3.382263). ⛔ 0 mismatches.

---

## UNSCORED WORK — all of it on DEV corners cut from inside `split.train_rows`
⭐ **SELECT = the first 6 admitted corners** (every mechanism choice) · **CONFIRM = the other 42**
(⛔ untouched until the configuration was frozen at 13:10Z). Split declared **11:08Z, before the first
sweep**. 49 corners admitted of 64 candidates; **15 REFUSED** for intersecting `eval_a`/`eval_b` or not
realising 72 rows — ⛔ refused, never trimmed.

| id | what | n | outcome |
|---|---|---|---|
| R1 | lever isolation: the predecessor's 6 untested v2 levers, one at a time | 6 SELECT | ⛔ **every one is noise** (≤ ±0.007; each would need 61–4,657 corners) |
| R2 | the **atoms-vs-rank frontier** at a fixed byte budget, K ∈ {32,64} × A ∈ {320…832} × 2 pair sets | 6 SELECT | ⭐ score **monotone increasing in A**, CIs excluding 0; **K=64 worse than K=32 at matched A** |
| R3 | ⛔ the disqualification screen: §C.2 at K ∈ {32, 64, 128} | 6 SELECT | ⛔ **K=128 DISQUALIFIED** (§C.2 = 0.5147, does not collapse below the 0.5278 floor) though it scored **higher** than the arm; K=64 clears at 0.3657 |
| R4 | the **shared-basis head** (interaction cost independent of A) | 6 SELECT | ⭐ **+0.0258 [+0.0051, +0.0466], 5/6** vs iteration 1's design — the first resolved positive |
| R5 | ⛔ **the low-K end**, K ∈ {1,2,4,8,16} — my own hypothesis | 6 SELECT | ⛔⛔ **REFUTED: 0.65 at K=1 against 0.89 at K=64**, monotone increasing in K |
| legal | NMF (charter-legal span objective) vs the R4 `pca_affine` control | 6 SELECT | ⭐ NMF recovers ~60–79 % of the control's advantage with decodable, deletable atoms |
| **CONFIRM** | **the frozen arm + 9 controls/references** | **42** | ⭐ the powered table — see `LOG-iter2.md` 15:42Z |
| R6 | ⭐ **the MATCHED-BYTES BOUND** — a storeless 52-prototype model + the shared-basis pair remap, 471,744 B. ⛔ Not an arm. | 43 | ⭐⭐ **`knn_proto` IS beatable at matched bytes: +0.0118 [−0.0001, +0.0237], 54.1 tuples** — a tie, one corner short of a resolved win. ⭐⭐ **The cost of being a store: +0.0270 [+0.0188, +0.0352], 39/43.** |
| R7 | the **PARTITION** (equal blocks vs the shipped ward grouping) and the **ALLOCATION** (uniform vs water-fill) | 6 → 43 | ⛔ my 6-corner call of "null" was **REVERSED** at n = 43: water-fill **+0.0105 [+0.0046, +0.0164]**; ward alone **−0.0016** (null); ward + water-fill **+0.0172** |
| R8 | ⭐ the best DEV configuration, **measured rather than assumed additive** (NMF + ward + water-fill) | 43 | ⭐ **0.8219 / 52.0 tuples, Δ vs `knn_proto` −0.0037 [−0.0167, +0.0093]** — level with the ceiling, **+0.0115 over the frozen arm**; ⛔ never scored (budget spent); ⚠ additivity had over-predicted it by +0.0057 |
| R9 | ⭐⭐ **the READ VERB on the SHIPPED store, powered** — argmax vs linear, byte-identical at 598,528 B | 43 | ⭐⭐⭐ **+0.1768 [+0.1512, +0.2024], 43 of 43 corners, tuples 12.8 → 37.1.** The scored `eval_b` value (+0.2014) sits INSIDE this CI. ⚠ DEV rows use the shipped store fitted on a pool containing these corners — both rows share that leak, so only the PAIRED difference is quoted |

⭐ **Free determinism check:** an operational error of mine (see LOG 11:28Z) ran 19 DEV units twice in
independent processes. **All 19 agree to 10 decimal places; 0 disagreements; 0 corrupt ledger lines.**

## PRE-REGISTRATION
`PREREG-hub-generation-table-2.md` — object, bar, controls and falsifiers frozen **13:15Z**; the
10-point prediction frozen **15:45Z**, before either leg ran. **8 hits, 2 misses**, scored line by line in
`LOG-iter2.md` at 16:14Z. ⛔ The body was not edited after a held-out number existed.
⭐ The prediction I care most about having made in writing: *"I predict falsifier 1 resolves as a TIE
under R1, not a WIN."* It did.
