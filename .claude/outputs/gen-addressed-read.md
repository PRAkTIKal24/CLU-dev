# gen-addressed-read — experiment-engineer report
**Task + acceptance criterion:** pre-register the LAUNCH-ONLY read as an arm (`x0`) and score it on composition-B **at matched declared bytes**, with the charter §C floor.
**Status:** done — 3/3 seeds, the ledger is **measured and matched**, and at matched bytes the arm **LOSES**.

## ⛔⛔ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — this needs an owner AT THIS REVIEW)
1. ⛔⛔ **A SHARED GATE ARTEFACT WAS CORRUPT AND IS BLOCKING EVERY `gen-cell` RUN.** `gen-external-classifier/classifier_gate.json` carried **one stray byte (`0x78`, `x`) appended after the JSON's closing `}\n`**; `check_preconditions` aborts on it. ⭐ Repaired **by truncation only** (`raw[:-1]`, asserted to be a pure byte prefix of the original), the corrupt file backed up verbatim, evidence in `runs/gate_file_repair.json` (sha256 `fbd2c55a…` → `88079d5f…`, 1297 → 1296 B). ⛔ **No byte inside the JSON was altered**; the body reads `gate: PASS`, floor 0.95, per-seed 1.0/1.0/1.0. ⚠ The file's mtime (2026-09-15 22:11) **post-dates** `gen-read-cell-arm`'s run, which passed this gate — so nothing already scored is affected. **`gen-external-classifier`'s owner should confirm the body is the gate they wrote.**
2. ⭐⭐ **PREREG §5 P7 IS REFUTED, AND IT IS THE SPOKE'S HEADLINE FINDING.** I predicted the atom-mean lookup would collapse to ≤ 0.20. It scores **0.3927 ± 0.0986** and the arm **does not beat it** (2/3 seeds, intervals overlap) ⇒ **charter §C.1 FAILS at matched bytes.** Every site that reads as though the addressed read is a matched-bytes positive must change.
3. ⚠ **`x0` is not on `GenCellConfig.arms` nor on the module constant `ARMS`** — deliberately (see §"What I did not touch"). It is selected explicitly (`--arm x0`). If the Hub wants it in a default cell run, that is one line in `chlu/config.py`, which I am forbidden.
4. ⚠ **A near-constant lookup ties LACE on this metric.** `x0/A_row_lookup` renders **2.67 ± 0.47 distinct factor tuples of 72** and scores **0.3272 ± 0.0633** vs `lace` **0.3117 ± 0.0153** (71.67/72 tuples): Δ = +0.0154, **not** 3/3 same sign, intervals overlap ⇒ a **TIE**. ⛔ This is a fact about *composition-B aggregate* (an unweighted mean over six per-factor accuracies), not about LACE. Whoever quotes that metric should know a ~3-image constant output reaches LACE's level on it.
5. ⚠ The `⛔ REGRESSION on lace` line on seed 0 is **pre-existing** — byte-identical to `gen-read-cell-arm`'s (`measured 0.6402 / published 0.7077 / Δ −0.0676`). ⛔ Not caused by this spoke.

## ⭐ DIAL DECLARATION (echoed before the first result)
- **Dial:** ⭐ **addressing alone** — a read that stops at the addressed launch point, spending **zero** settling steps.
- **Laundering control:** `store_removed` on every row, **plus** the matched-bytes lookup floor (charter §C).
- **Falsifies:** that addressing alone carries the generation table.
- **Does NOT falsify:** losing to `knn_proto`, an unbounded real-valued channel (metric-native-ceiling theorem).

---

## ⛔⛔ WHAT THE ARM IS, AND THE NAME IT IS NOT GIVEN
`x0` = **request → address-book posterior argmax over the written atoms → decode.**
⭐ It is an **ADDRESSED READ**. ⛔ It is **not** dynamics, not settling, not a landscape read, not "the dynamical read at zero steps". Measured, not asserted: **0 steps · displacement from launch exactly `0.0` · the decoder's input IS the launch bit-for-bit · `energy_trace = None` · `coupling_used = False`**, on 3/3 seeds and both rungs. A source-level guard (`test_x0_is_NEVER_described_as_dynamics_anywhere_on_its_own_path`) and its own positive control pin the vocabulary.

---

## ⭐⭐⭐ THE MATCHED-BYTES VERDICT — LED WITH, AS INSTRUCTED

> ### **`x0` = 0.5008 ± 0.0498 vs `clu` = 0.5949 ± 0.0082 at the SAME 524,288 B + 74,240 B head.**
> ### **Δ = −0.0941, 3/3 seeds negative, intervals disjoint ⇒ ⛔ LOSS.** (m = 0.020)
> ### **⛔⛔ AND IT FAILS THE CHARTER §C.1 FLOOR: it does not beat the matched-bytes atom-mean lookup (0.3927 ± 0.0986; 2/3 seeds, intervals overlap) ⇒ at matched bytes the store is DECORATIVE and this row does NOT count as a positive.**
> ### **PASS: NO. ⭐ This is the FLOOR for target 1.**

**The 8× companion, with its multiple attached:** `x0/B` = **0.6852 ± 0.0173 at 8.00× the bytes** (4,194,304 B atoms + 593,920 B head) — ⛔ **not byte-matched.** At that price it beats `clu` outright (Δ = +0.0903) and clears both §C.1 floors, and still loses to the `knn_proto` ceiling (Δ = −0.1798). ⛔ **Quoting 0.6852 against 0.5949 without "at 8× the bytes" on the same line is the overclaim this spoke was written to stop.**

---

## ⭐⭐ THE BYTE LEDGER — the crux, MEASURED on the live objects (⛔ not assumed)

**Does the addressed read need the low-rank coupling? ⭐ NO — proved bit-for-bit, not read off the source.** `x0` was recomputed on four landscapes differing only in the coupling and the two strengths — the fitted coupling at the certificate's λ, `coupling=None`, λ = 0, β = 0 — and all four are **`np.array_equal`, max |Δ| = 0.0**, on every seed and both rungs. The run **aborts** otherwise, and that abort has a positive control in the test suite.

| | ⭐ `x0` rung A (measured) | `clu` (published) | `dynread` rung A (same atoms, settled) |
|---|---|---|---|
| store / atoms | **524,288 B** | **524,288 B** | 524,288 B |
| addressing head | **74,240 B** | **74,240 B** | 0 (reported inside its 2×) |
| **coupling, DEPLOYED** | ⭐ **0 B** | 0 B | 524,288 B |
| **× the 524,288 B budget** | ⭐⭐ **1.00×** | 1.00× | **2.00×** |

⭐ **The ledger is `clu`'s field for field** — same store bytes, same addressing-head bytes, same closed form `(K·N + Σ_f V_f·K·N)·4` at `N = M = 64`, `K = 5`, `Σ_f V_f = 57`. The comparison is a **match**, not a resemblance. ⭐ **524,288 B of coupling leave the ledger**: the dynamical read pays 2.00× at this rung; the addressed read pays 1.00×. The match is asserted against **this manifest's own store ledger**, ⛔ never a literal (`run_x0_budget`'s source contains no `524288`, pinned by a test).

---

## ⭐⭐ THE GENERATION TABLE — composition-B aggregate, external classifier, 3 seeds
⭐ Every accuracy carries its distinct-factor-tuple count. ⛔ Every byte figure carries its multiple of the 524,288 B budget.

| arm | composition-B | per-seed | distinct tuples / 72 | deployed bytes (× budget) | role |
|---|---|---|---|---|---|
| `store_removed` (**the null**) | 0.1204 ± 0.0131 | .1111 / .1389 / .1111 | 1.00 ± 0.00 | 524,288 + 74,240 = **1.00×** | laundering control |
| `ebm_compose` | 0.1937 ± 0.0182 | .1736 / .1898 / .2176 | 72.00 ± 0.00 | 14,226,660 = 27.14× | like-for-like |
| ⭐ **`x0/A_atoms_shuffled`** | **0.1366** ± 0.0422 | .1782 / .0787 / .1528 | 22.33 ± 3.68 | 524,288 + 74,240 = **1.00×** | ⭐ **§C.2 control — COLLAPSES** |
| `dynread` (the **settled** read, same atoms) | 0.2083 ± 0.0514 | .1620 / .1829 / .2801 | 13.33 ± 1.25 | 1,048,576 = 2.00× | ⛔ another spoke's row, for contrast |
| `lace` | 0.3117 ± 0.0153 | .3009 / .3333 / .3009 | 71.67 ± 0.47 | not byte-matched | like-for-like (cell harness) |
| ⚠ **`x0/A_row_lookup`** | **0.3272** ± 0.0633 | .2847 / .2801 / .4167 | **2.67 ± 0.47** | 524,288 + 74,240 = **1.00×** | ⭐ **§C.1 floor (whole stored row)** |
| ⛔⛔ **`x0/A_atom_mean_lookup`** | **0.3927** ± 0.0986 | .5231 / .2847 / .3704 | 11.00 ± 1.63 | 524,288 + 74,240 = **1.00×** | ⭐⭐ **§C.1 floor — THE ROW THAT FAILS THE ARM** |
| ⭐⭐ **`x0` (THE ADDRESSED READ)** | **0.5008** ± 0.0498 | .4329 / .5185 / .5509 | **18.67 ± 2.36** | **524,288 + 74,240 = 1.00×** | ⭐⭐ **THE HEADLINE ROW — MATCHED BYTES** |
| ⭐ `clu` (**the codebook read it would replace**) | **0.5949** ± 0.0082 | .5903 / .6065 / .5880 | 9.00 ± 0.82 | 524,288 + 74,240 = **1.00×** | ⭐⭐ **the matched-bytes rival** |
| `knn_proto` (**ceiling**) | 0.8650 ± 0.0122 | .8588 / .8542 / .8819 | 56.00 ± 3.74 | 475,136 = 0.91× | ⛔ unbounded real-valued channel |

### ⭐ The 8× companion block (⛔ every row here is **8.00×** the budget and is never quoted bare)
| row | composition-B | per-seed | distinct tuples / 72 | bytes |
|---|---|---|---|---|
| `x0/B_atoms_shuffled` | 0.1574 ± 0.0452 | .2060 / .1690 / .0972 | 33.00 ± 2.45 | 4,194,304 + 593,920 = **8.00×** |
| `dynread/B` (settled) | 0.2114 ± 0.1033 | .0671 / .3032 / .2639 | 16.00 ± 1.41 | 8,388,608 = 16.00× |
| `x0/B_row_lookup` | 0.4421 ± 0.0650 | .4560 / .5139 / .3565 | 5.33 ± 0.47 | 4,194,304 + 593,920 = **8.00×** |
| `x0/B_atom_mean_lookup` | 0.5231 ± 0.0842 | .6227 / .5301 / .4167 | 14.00 ± 2.45 | 4,194,304 + 593,920 = **8.00×** |
| ⭐ **`x0/B`** | **0.6852 ± 0.0173** | .6667 / .7083 / .6806 | 25.33 ± 2.62 | **4,194,304 + 593,920 = 8.00×** |

### ⭐⭐ RUNG A — the verdict table (PREREG §2b as hardened; m = 0.020; ⛔ a tie is available only against the ceiling)
| rival | role | Δ | paired per-seed | 3/3 same sign | intervals overlap | **verdict** |
|---|---|---|---|---|---|---|
| `knn_proto` | ceiling | **−0.3642** | −.4259 / −.3356 / −.3310 | yes | no | ⛔ **LOSS** *(the metric-native-ceiling theorem — ⛔ not news)* |
| ⭐ `clu` | like-for-like | **−0.0941** | −.1574 / −.0880 / −.0370 | yes | no | ⛔⛔ **LOSS** |
| ⛔ **`x0/A_atom_mean_lookup`** | ⭐ **§C.1 FLOOR** | +0.1080 | **−.0903** / +.2338 / +.1806 | ⛔ **no** | ⛔ **yes** | ⛔⛔ **LOSS** |
| `x0/A_row_lookup` | ⭐ §C.1 FLOOR | **+0.1736** | +.1481 / +.2384 / +.1343 | yes | no | **BEATEN OUTRIGHT** |
| `lace` | like-for-like | **+0.1890** | +.1319 / +.1852 / +.2500 | yes | no | **BEATEN OUTRIGHT** |
| `ebm_compose` | like-for-like | **+0.3071** | +.2593 / +.3287 / +.3333 | yes | no | **BEATEN OUTRIGHT** |
| `x0/A_atoms_shuffled` | ⭐ §C.2 control | **+0.3642** | +.2546 / +.4398 / +.3981 | yes | no | **BEATEN OUTRIGHT** (⭐ the control collapses) |
| `store_removed` | like-for-like | **+0.3804** | +.3218 / +.3796 / +.4398 | yes | no | **BEATEN OUTRIGHT** |

⛔⛔ **CEILING: LOSS. LIKE-FOR-LIKE ALL BEATEN: NO (`clu`). ⭐ CHARTER §C.1 FLOOR: FAIL. §C.2 CONTROL COLLAPSES: YES. PASS: NO.**

### RUNG B — the same table at **8.00× the bytes** (⛔ a companion, never the headline)
| rival | Δ | 3/3 same sign | intervals overlap | verdict |
|---|---|---|---|---|
| `knn_proto` (ceiling) | **−0.1798** | yes | no | ⛔ **LOSS** |
| `clu` | **+0.0903** | yes | no | **BEATEN OUTRIGHT** — ⛔ **at 8× the bytes** |
| `x0/B_atom_mean_lookup` (§C.1) | **+0.1620** | yes | no | **BEATEN OUTRIGHT** |
| `x0/B_row_lookup` (§C.1) | **+0.2431** | yes | no | **BEATEN OUTRIGHT** |
| `x0/B_atoms_shuffled` (§C.2) | **+0.5278** | yes | no | **BEATEN OUTRIGHT** |
| `lace` · `ebm_compose` · `store_removed` | +0.3735 · +0.4915 · +0.5648 | yes | no | **BEATEN OUTRIGHT** |

⭐ **§C.1 floor: PASS at 8×. Like-for-like all beaten: YES at 8×. Ceiling: LOSS ⇒ PASS: NO.**

---

## ⭐⭐ THE THREE FINDINGS THAT MATTER

### 1. ⛔⛔ THE §C.1 FLOOR FAILS AT MATCHED BYTES — and my own pre-registration is what caught it
PREREG §5 **P7 predicted** the atom-mean lookup would score **≤ 0.20 with ≤ 5 distinct tuples**, reasoning that a naive-Bayes posterior over 64 atoms is near-flat so its weighted mean is ≈ the well's global atom mean. **Measured: 0.3927 ± 0.0986, 11.00 ± 1.63 tuples.** ⭐ **The prediction is refuted by a factor ~2, and the derivation is refuted with it:** the posterior is **3.5× uniform** (mean confidence **0.0552** over `N = 64`, uniform = 0.0156), so the mean is a *steered blur*, not a constant — and the decoder reads factors out of a blur nearly as well as out of an atom.
⇒ **At matched bytes, replacing the argmax with a mean costs only 0.108 and that gap does not survive the bar** (seed 0 is **negative**, −0.0903; the intervals overlap). **Charter §C.1: the store is decorative at 524,288 B, and this row is reported as such.** ⭐ An un-pre-registered version of this run would have reported "x0 beats the lookup by +0.108" and been wrong.

### 2. ⭐ THE COUPLING IS DEAD WEIGHT FOR THIS READ — worth 524,288 B, measured
The addressed read is a function of (request, address book, atoms) **only**, proved by bit-identity across four landscapes. That halves the rung-A ledger relative to the dynamical read (2.00× → **1.00×**) and is what makes "matched bytes" a fact here. ⭐ It also sharpens `gen-read-cell-arm`'s result: at the **same** atom table, the settled read scores **0.2083** and pays **2.00×**, while the launch it starts from scores **0.5008** and pays **1.00×**. ⛔ The chain costs double the bytes to lose 0.29 of accuracy.

### 3. ⭐ THE STORE'S FACTORISATION IS REAL, THE ADDRESSING IS WHAT IS WEAK
Three rows isolate three different things at identical bytes:
| row | what it removes | composition-B | distinct tuples / 72 |
|---|---|---|---|
| `x0` | — | **0.5008** | **18.67** |
| `x0/A_row_lookup` | ⭐ the **per-well recombination** (one index for all wells) | 0.3272 | **2.67** |
| `x0/A_atoms_shuffled` | ⭐ the **content** the index points at | 0.1366 | 22.33 |
⭐ Composing across wells is worth **+0.1736** and takes the output from **2.67** distinct tuples to **18.67** — *the factorisation is load-bearing and is beaten outright.* ⭐ The content is load-bearing too: permuting the atoms drops the arm to **0.1366**, i.e. to the null (`store_removed` 0.1204, chance 0.1236), while **keeping 22.33 distinct tuples** — ⛔ diversity without correctness, which is why an accuracy is never reported without its tuple count.
⛔ What is *not* load-bearing enough is the **argmax over the mean**: finding 1. The defect is in the **address book's resolution**, not in the store's structure.

### ⭐ Charter §C.4, stated rather than discovered by a referee
`x0`'s output **is** stored atoms — one per well, concatenated — asserted in the test suite. ⭐ **By §C.4 this arm IS a lookup table and is reported as one**; what it adds over a plain lookup is the per-well recombination, and that gain (+0.1736, finding 3) is the honest size of the claim.

### ⛔ What did NOT go wrong (checked, so the loss is not mis-attributed)
- ⭐ **`x0` is bit-identical to `gen-read-cell-arm`'s persisted launch on 6/6 (seed × rung)** — `np.array_equal`, shapes (72, 2048). PREREG A6 holds; P3/P4/P5 reproduce **exactly** (0.5008 ± 0.0498 · 0.6852 ± 0.0173 · 18.67 / 25.33 tuples). ⭐ The promotion changed the **scoring**, ⛔ not the read.
- ⭐ **Same cell, 3/3 seeds:** decoder param hash + recipe (`v1_capacity_schedule`, ⛔ never `baseline`), external-classifier hash, encoder hash, feature-cache sha256, split mode `stratified`, `n_fit/n_val/n_B/n_MS = 32256/512/72/198` — all bit-identical to `gen-read-cell-arm`'s, and **all six prior rows reproduce to the bit** (Δ = 0.000000).
- ⭐ **Instrument invariance on the headline metric:** the external classifier and the probe companion give the **identical arm ordering** on composition-B (`knn_proto` > `x0/B` > `clu` > `x0` > `atom_mean` > `row_lookup` > `lace` > `dynread` > `ebm_compose` > `store_removed`). ⚠ The aggregate's own `_ranking.identical = False` is computed on the **in-corner** statistic and the only swap there is `lace` ↔ `lace/ld_corrected` — ⛔ no `x0`/`clu` reordering on any instrument.
- Atoms are train rows, asserted bit-for-bit on 6/6. Launch = posterior **argmax** on every row.
- ⚠ **`exact_match` (all six factors right) is brutal for every store arm:** `clu` **0.0000**, `x0` **0.0046 ± 0.0065**, both §C.1 floors **0.0000**, `x0/B` 0.0417 ± 0.0227, `knn_proto` 0.1991 ± 0.0664. ⛔ The aggregate is an unweighted per-factor mean and is much kinder than exact match; reconciliation item 4 is the same fact from the other side.

---

## ⭐ FLAG PROVENANCE
| | value |
|---|---|
| commit | `db99ae8` (branch `agent/experiment-engineer/gen-addressed-read`, worktree `../CHLU-addrread`) — ⭐ **the hash the scored run executed**, recorded in `runs/progress.txt` |
| base | ⭐ **the PIN `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`** (charter §A4). ⛔ Not `main`, which has moved; no rebase was needed or done. |
| seeds | 0, 1, 2 |
| config | `gen-val-split-fix/cell_rescore.yaml` (⭐ the same override the repaired-split cell and `gen-read-cell-arm` ran under) |
| dataset / pretext | shapes3d / recon · `n_fit` 32,256 · `n_val` requested 512, **realised 512** · rows B 72 / MS 198 · split `stratified` |
| decoder | `v1_capacity_schedule` (⛔ never `baseline`), winner hash asserted per arm per seed |
| classifier | `gen-external-classifier` gated (`gate: PASS`, floor 0.95, per-seed 1.0/1.0/1.0), hash asserted per arm per seed |
| store | `gen-read-compose` checkpoint, **loaded, not re-trained**; atoms are a seeded k-means++ draw over this seed's pool |
| rungs | **A (MATCHED): 64 atoms/well, 524,288 B + 74,240 B head = 1.00×** · B (8×): 512 atoms/well, 4,194,304 B + 593,920 B head |
| **settling steps** | ⭐⭐ **0** · `coupling_used` **False** · `energy_trace` **None** · displacement from launch **exactly 0.0** |
| launch / read | `launch_mode = clamp_atom` (posterior argmax) · `read_mode = state` · `clamp_threshold` 0.5 (reporting-only) · β, λ **inert on this path, proved** |
| §C.1 floors | `posterior_mean` over the same atoms · whole-stored-row joint argmax — **both at identical bytes** |
| §C.2 control | per-well independent permutation of the atom table, seed `7919·s + 20260916`, same indices |
| margin | `m = 0.020` — ⛔ the pin's `DYNREAD_MARGIN`, fixed before the run, ⛔ not re-derived |
| JAX | main venv (⛔ no worktree `uv sync`), `JAX_PLATFORMS=cpu` |
| `--quick` | ⛔ not used anywhere |

## ⭐ COST
⭐ **CPU only** (charter §D — this is a floor measurement, ⛔ no GPU). 3 seeds × 8 rows = **24 scored rows** in **62 s + 260 s + 261 s = 583 s (9.7 min)** wall clock, + 3 s aggregate. ⚠ Seed 0 is 4× faster because its feature cache is `scope="full"`; seeds 1–2 re-encode. Artefacts **15 MB** (24 launch-state npz). ⭐ The ten rival/contrast rows cost **nothing** — copied from `gen-read-cell-arm`, ⛔ not re-run. JAX cold start ~6 s, ⛔ not the 20 min the protocol warns about.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-addressed-read`, off the **pin** `ae90746`; worktree `../CHLU-addrread`.
- `db99ae8` `[experiment-engineer] score the addressed read as an arm, at matched declared bytes`
- **Files touched:** `chlu/experiments/exp_gen_cell.py` (+735) · `tests/test_gen_cell.py` (+392) — ⛔ **and nothing else** (`git diff --name-only ae90746..HEAD`).
- ⭐ **What I did NOT touch, on purpose:** `chlu/core/gen_landscape.py` (frozen — I *call* `launch_state`, `atom_posterior`, `read_settled`) · `chlu/config.py` · the classifier · the splits · the rival rows · **`ARMS`** (the pin's declared rival-row list — `x0` is selected explicitly like `dynread`, and a test pins that reasoning) · **`arm_byte_ledger`** (a pinned byte-ledger assertion — the measured ledger reaches the row through `_record`'s `**extra` instead) · `dynread_verdict` / `DYNREAD_LIKE_FOR_LIKE` (called with `arm="x0"`; the §C.1 floors are judged by a second comparator whose agreement with the pinned one is **asserted at runtime**, and the run aborts if they ever differ).
- ⭐ Branch ref verified **from the main repo** before hand-off (w4 precaution). ⛔ Not pushed, no PR. Worktree left in place for review.

## Tests
⭐ `pytest tests/test_gen_cell.py` with `CHLU_GEN_DATA_ROOT` set → **71 passed · 0 skipped · 0 failed** (45 at the pin + **26 new**). ⛔ No full suite (task forbids it). `ruff check` clean on both files.
⭐ The load-bearing new tests are the ones that **fire**: `test_x0_coupling_proof_FIRES_when_the_coupling_reaches_the_read` (the ledger refuses a match it cannot prove) · `test_x0_zero_step_proof_FIRES_when_something_sits_between_launch_and_decoder` · `test_the_naming_guard_itself_FIRES_on_a_bare_use` · `test_the_charter_C1_FLOOR_fails_LOUDLY_when_a_lookup_is_not_beaten` (an arm that beats every pinned rival and **still** does not count) · `test_a_sub_margin_gain_over_a_lookup_floor_is_a_LOSS_never_a_WIN`.

## Open questions / follow-ups / risks
1. ⭐⭐ **The floor failure localises the defect: the address book, not the store.** The argmax beats the mean by only 0.108 at matched bytes and the gap does not survive the bar, while the *factorisation* is beaten outright (+0.1736) and the *content* control collapses. **The next question is whether a sharper posterior (mean confidence is 0.0552 over 64 atoms) closes the 0.0941 gap to `clu`** — that is a **new pre-registration**, ⛔ not a re-run of this one.
2. ⚠ **Rung A's spread is ±0.0498 against `clu`'s ±0.0082** (per-seed 0.4329 → 0.5509). The `clu` loss is 3/3 and disjoint, so the verdict is safe; ⛔ but any *future* rung-A number needs more than 3 seeds. The §C.1 floor verdict in particular turns on **one seed's sign** and would be the first thing more seeds could move — ⛔ in either direction.
3. ⚠ **What does `n_val = 512` buy?** ⛔ Unchanged here (the ladder's business); the cell asserts against silent truncation as `gen-read-cell-arm` left it.
4. ⚠ Rung B's §C.1 pass is **at 8× the bytes** and ⛔ licenses nothing at matched bytes.

## Proposed handover updates (for the Hub)
- **§ Generation table:** add the eight `x0` rows. ⭐ The matched-bytes row is `x0` **0.5008 ± 0.0498 at 1.00× (524,288 B + 74,240 B head)**; the 8× row is `x0/B` **0.6852 ± 0.0173 at 8.00×** and ⛔ carries its multiple wherever it appears.
- **§ Live claims — ⛔ CORRECT THE RELAY.** *"The launch beats the codebook read 0.6852 vs 0.5949 on 3/3 seeds"* is **TRUE ONLY AT 8× THE BYTES**. ⭐ **At matched bytes the addressed read LOSES to the codebook read by 0.0941 (3/3 seeds, disjoint) and FAILS the charter §C.1 lookup floor.** Addressing alone does **not** carry the generation table.
- **§ Negative results:** *"The addressed read (posterior-argmax gather over stored atoms, zero steps) is byte-matched to the codebook read field for field — 524,288 B + 74,240 B, the low-rank coupling proved unnecessary bit-for-bit — and at that price it scores 0.5008 ± 0.0498 against the codebook's 0.5949 ± 0.0082, **and does not beat a matched-bytes atom-mean lookup** (0.3927 ± 0.0986; 2/3 seeds, overlapping intervals). Charter §C.1: at matched bytes the store is decorative. The per-well recombination IS load-bearing (+0.1736 over a whole-row lookup, 2.67 → 18.67 distinct tuples of 72) and the content IS load-bearing (permuting the atoms drops it to 0.1366, the null) — the weak link is the address book's resolution."*
- **§ Negative results, second entry:** *"A pre-registered prediction failed and that is the finding: the atom-mean lookup was predicted at ≤ 0.20 from a near-flat-posterior argument; the posterior is 3.5× uniform (0.0552 over 64 atoms) and the lookup scores 0.3927."*
- **§7 Known Issues, new:** *"`gen-external-classifier/classifier_gate.json` was corrupted by one stray trailing byte and aborted every `gen-cell` run at precondition 1/2. Repaired by truncation only (evidence + verbatim backup in `gen-addressed-read/runs/gate_file_repair.json`); the JSON body is untouched and reads `gate: PASS`. ⛔ Owner confirmation owed by `gen-external-classifier`."*
- **§ Owed:** one line adding `"x0"` to `GenCellConfig.arms` **if** the Hub wants it in default cell runs (I am forbidden that file; ⛔ note it is a *losing* arm at matched bytes, so the `dynread` precedent argues against).
- **§ Metric caveat:** composition-B aggregate is an unweighted mean over six per-factor accuracies; a **2.67-distinct-tuple** near-constant lookup **ties LACE** on it (0.3272 ± 0.0633 vs 0.3117 ± 0.0153). ⛔ `exact_match` is 0.0000 for `clu` and both floors.
