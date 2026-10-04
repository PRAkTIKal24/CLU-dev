# gen-read-cell-arm — experiment-engineer report
**Task + acceptance criterion:** add the dynamical read as ONE new arm (`dynread`) to `exp_gen_cell.py` and score it on composition-B against the arms already on the generation table.
**Status:** done — the arm is wired, ran 3/3 seeds on the shipped cell, and **LOST**.

## ⛔⛔ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — this needs an owner AT THIS REVIEW)
1. **PREREG §5 falsifier 1 is answered NO.** `dynread` does not beat `store_removed` AND the codebook read; it is a **LOSS** against `clu`, `lace`, `ebm_compose` and the `knn_proto` ceiling. Every site that reads as though the dynamical read is the generation-table row must change.
2. **PREREG §2b's `lace` row (0.3241 ± 0.0165) is not what is on disk.** The repaired-split cell manifests say **0.3117 ± 0.0153**. One of the two is wrong; ⛔ I did not touch either. (The harder number, 0.3241, was used as the bar — the verdict is a LOSS under both.)
3. **The task's "`knn_proto` renders 72/72" is not measured.** The repaired-split cell measures **56.00 ± 3.74** of 72. `clu`'s 9.00 ± 0.82 is confirmed.
4. **`GenCellConfig.arms` still lists five arms.** `chlu/config.py` is held by a peer spoke and I am forbidden to touch it, so `dynread` is selected explicitly (`--arm dynread`). One line is owed to whoever owns that file.
5. **The task file carried no DIAL DECLARATION block** (w24 ruling, binding since w25). I echo one below, reconstructed from the task's own falsifier section; ⛔ it was not pre-approved.

## ⭐ DIAL DECLARATION (echoed before the first result)
- **Dial:** *compute-adaptive reads* — a read that spends 1000/2000 settling steps, scored against reads that spend none. (The host module's own declaration is "composition, end to end, on IMAGES"; ⛔ composition was RELEASED as a claim in R29 and nothing here reopens it.)
- **Laundering control:** `store_removed`, on every row (the harness's), **plus** this arm's own `request_off` (β = 0, same launch, same chain, same key).
- **Falsifies the claim:** the table itself — losing to the like-for-like rows beyond seed spread.
- **Does NOT falsify:** losing to `knn_proto` (an unbounded real-valued channel) is the metric-native-ceiling theorem, not news.

---

## What I did
- Added `dynread` as a sixth arm to `chlu/experiments/exp_gen_cell.py`. ⭐ **Wiring, not building:** the landscape is `exp_gen_dynamical_read.build_landscape` and the chain is its `run_arm`, called on **this cell's** store rows and well partition. ⛔ No second implementation.
- Four rows, not one: two pre-registered rungs (**A** = 64 atoms/well / rank 16 / 1000 steps; **B** = 512 / rank 128 / 2000 steps) × {`full`, `request_off`}.
- `dynread_preflight` refuses three traps **as assertions**: λ taken by fiat instead of from the coercivity certificate; a thresholded launch (which degenerates to the prior's argmax); and T > 0, which would stop `request_off` being a same-chain control.
- Asserted the **realised** `n_val` against the requested one and recorded both.
- Persisted every artefact refusal and dreaming need.
- Added 21 tests to `tests/test_gen_cell.py`; the verdict is **computed** by `dynread_verdict`, never written by hand.

## How I verified
- **Targeted tests:** `pytest tests/test_gen_cell.py` with `CHLU_GEN_DATA_ROOT` set → **45 passed, 0 skipped, 0 failed** (24 pre-existing + 21 new). ⛔ No full suite (task forbids it). `ruff check` clean on both files.
- **⭐ SAME-CELL PROOF (`cell_results.json.same_cell_proof.ok = true`).** The run was seeded by **copying** `gen-val-split-fix`'s repaired-split manifests, then adding only the arm. Per seed, these are bit-identical to that spoke's: decoder param hash + recipe (`v1_capacity_schedule`), external-classifier param hash, encoder param hash, feature-cache sha256, store checkpoint path, row-split mode (`stratified`), and `n_fit`/`n_val`/`n_B`/`n_MS` = 32256/512/72/198. **All five rival rows reproduce to the bit.** ⛔ The arm is on the shipped cell, not one of my own.
- **⛔ `--quick` was not used anywhere.** A plumbing smoke ran first at 8 steps into a throwaway directory; ⛔ none of its numbers appear here.

---

## ⭐⭐ THE GENERATION TABLE — composition-B aggregate, external classifier, 3 seeds
⛔ **Read the "still-descending" column before the accuracy column. See "NOT CONVERGED" below.**

| arm | composition-B | per-seed | distinct tuples / 72 | still-descending @ its budget (s0/s1/s2) | deployed bytes |
|---|---|---|---|---|---|
| `store_removed` (**the null**) | 0.1204 ± 0.0131 | .1111 / .1389 / .1111 | 1.00 ± 0.00 | — | 524,288 |
| `ebm_compose` | 0.1937 ± 0.0182 | .1736 / .1898 / .2176 | 72.00 ± 0.00 | — | 14,226,660 |
| `lace` | **0.3117** ± 0.0153 *(⛔ not §2b's 0.3241)* | .3009 / .3333 / .3009 | 71.67 ± 0.47 | — | not byte-matched |
| ⭐ `clu` (codebook read — what this arm replaces) | **0.5949 ± 0.0082** | .5903 / .6065 / .5880 | 9.00 ± 0.82 | — | 524,288 |
| ⭐ **`dynread` (rung A, matched atoms)** | **0.2083 ± 0.0514** | .1620 / .1829 / .2801 | 13.33 ± 1.25 | **0.75 / 0.79 / 0.72** | 1,048,576 |
| **`dynread/B`** (declared larger) | 0.2114 ± 0.1033 | .0671 / .3032 / .2639 | 16.00 ± 1.41 | **0.60 / 0.25 / 0.15** | 8,388,608 |
| ⭐ **`dynread/A_request_off`** (steering control) | 0.0463 ± 0.0346 | .0556 / .0833 / .0000 | **1.00 ± 0.00** | 1.00 / 1.00 / 1.00 | 1,048,576 |
| **`dynread/B_request_off`** | 0.0000 ± 0.0000 | .0000 / .0000 / .0000 | **1.00 ± 0.00** | 1.00 / 1.00 / 1.00 | 8,388,608 |
| ⭐⭐ `knn_proto` (**ceiling**) | **0.8650 ± 0.0122** | .8588 / .8542 / .8819 | 56.00 ± 3.74 *(⛔ not 72/72)* | — | 475,136 |

### ⭐⭐ VERDICT — PREREG §2b as hardened by `PREREG-addendum-0`. m = 0.020.
| rival | role | Δ | paired per-seed | 3/3 same sign | intervals overlap | **verdict** |
|---|---|---|---|---|---|---|
| `knn_proto` | ceiling | **−0.6566** | −.6968 / −.6713 / −.6019 | yes | no | ⛔ **LOSS** |
| `store_removed` | like-for-like | +0.0880 | +.0509 / +.0440 / +.1690 | yes | no | **BEATEN OUTRIGHT** |
| `ebm_compose` | like-for-like | +0.0147 | −.0116 / −.0069 / +.0625 | **no** | yes | ⛔ **LOSS** |
| `lace` | like-for-like | −0.1034 | −.1389 / −.1505 / −.0208 | yes | no | ⛔ **LOSS** |
| ⭐ `clu` | like-for-like | **−0.3866** | −.4282 / −.4236 / −.3079 | yes | no | ⛔ **LOSS** |

⛔⛔ **CEILING VERDICT: LOSS. Like-for-like all beaten: NO. PASS: NO.**
⭐ The only row it beats is the **null**. It is beaten by the codebook read it was built to replace, by **0.3866**.

### ⛔⛔ AND SEPARATELY: NOT CONVERGED — so none of the above is quotable
⛔ **No chain settled at its declared step budget.** Still-descending fraction on B, at each rung's own budget: rung A **0.72–0.79**, rung B **0.15–0.60**, both `request_off` rows **1.00**. `QUOTABLE = false` on all four rows, all three seeds.
⭐ More compute does help, and ⛔ does not finish the job — rung B at 1000 vs 2000 steps: seed 0 0.542 → 0.597 (**up**), seed 1 0.736 → 0.250, seed 2 0.653 → 0.153. ⛔ I did **not** extend the budget: A = 1000 / B = 2000 were declared before the run and extending them after seeing the result is tuning a pre-registered bar.
⚠ ⛔ Nothing diverged (`diverged = false`, settled-RMS / atom-RMS radius 1.19–1.62), so this is slow drift, ⛔ not blow-up.

---

## ⭐⭐ THE TWO FINDINGS THAT MATTER

### 1. Steering is real — and it is not enough
| rung | full | request-off (β = 0) | Δ | per-seed Δ | tuples full → off |
|---|---|---|---|---|---|
| A | 0.2083 ± 0.0514 | 0.0463 ± 0.0346 | **+0.1620** | +.1065 / +.0995 / +.2801 | 13.33 → **1.00** |
| B | 0.2114 ± 0.1033 | 0.0000 ± 0.0000 | **+0.2114** | +.0671 / +.3032 / +.2639 | 16.00 → **1.00** |

⭐ **The falsifier this spoke was told to watch for did NOT fire.** Request-off does *not* score the same: it collapses to **1 distinct factor tuple of 72** and (rung B) to **exactly 0.0000** on all three seeds — every chain lands in one basin. The request energy carries **100 % of the output's distinctness**, reproducing `gen-dynamical-read` PREREG §4's build-scale collapse at full scale. ⛔ But the steering it buys tops out at ~0.21, far under `clu`'s 0.5949. **The gain is attributable; it is just small.**

### 2. ⭐⭐ SETTLING IS WHAT DESTROYS THE READ (wiring diagnostic, ⛔ NOT a table row)
`x0` (the launch) is a pure function of (request, address book, atoms) and never touches the chain. Decoding it through the **same** decoder and **same** classifier:

| rung | launch `x0` | settled `x_settled` | Δ (settling) | tuples launch → settled |
|---|---|---|---|---|
| A | **0.5008 ± 0.0498** | 0.2083 ± 0.0514 | **−0.2925** | 17/17/22 → 12/15/13 |
| B | **0.6852 ± 0.0173** | 0.2114 ± 0.1033 | **−0.4738** | 23/24/29 → 15/18/15 |

⭐⭐ **Rung B's launch (0.6852 ± 0.0173) BEATS the codebook read (0.5949 ± 0.0082) on 3/3 seeds — and 2000 steps of settling take it down to 0.2114.** This is the load-bearing result of the spoke: the store, the address book, the request path and the decoder are all working; **the dynamics are the defect, not the wiring.**
⛔ **This is a diagnostic, not a row.** It is `gen-dynamical-read`'s `zero_step` control used as an instrument, that spoke owns it, and ⛔ it is NOT the atom-mean lookup row (PREREG §2a), which is a read-error measurement and was explicitly out of scope.

### ⛔ What did NOT go wrong (checked, so the loss is not mis-attributed)
- λ came from the **coercivity certificate** on every rung/seed — used **0.0485–0.0528**, λ* = 0.0539–0.0586, capped on 6/6. ⛔ Never 1.0.
- Byte ledger matched PREREG §3 **exactly** on 6/6 (A: 524,288 + 524,288 = 2.00×; B: 4,194,304 + 4,194,304 = 2.00×).
- Atoms are **train rows**, asserted bit-for-bit on 6/6.
- Launch = posterior **argmax**; `mean_n_clamped = 0.0/5` on every row (confidence 0.012–0.067) — exactly as the task said.
- ⚠ The `⛔ REGRESSION on lace` line in my logs (0.6402 vs published 0.7077) is **pre-existing** — byte-identical to the line in `gen-val-split-fix/runs/C_seed0.log`. ⛔ Not caused by this spoke.

---

## ⭐ FLAG PROVENANCE
| | value |
|---|---|
| commit | `6449915` (branch `agent/experiment-engineer/gen-read-cell-arm`) |
| **code the run executed** | ⚠ `a55c0e4` — the pre-rebase hash of the same tree. The rebase below was a pure replay onto three out-of-scope commits; ⛔ the scored numbers were produced by `a55c0e4`, which is what `runs/progress.txt` records. |
| base | `main` @ `8a404de` — ⚠ the task named `main @ 115d4b5`. `main` was already 4 commits ahead at spawn (`7295673`) and **advanced 3 more during the run** (a peer's CSF3 CPU-jobs merge). The owned files are bit-identical across all three points (`git diff 115d4b5 main -- <my two files>` is empty), so the rebase was clean and changed nothing of mine. |
| seeds | 0, 1, 2 |
| config | `gen-val-split-fix/cell_rescore.yaml` (⭐ the same override the repaired-split cell ran under) |
| dataset / pretext | shapes3d / recon · `n_fit` 32,256 · **`n_val` requested 512, realised 512** · rows B 72 / MS 198 · split `stratified` |
| decoder | `v1_capacity_schedule` (⛔ never `baseline`), winner hash asserted per arm per seed |
| classifier | `gen-external-classifier` gated, hash asserted per arm per seed |
| store | `gen-read-compose` checkpoint, **loaded, not re-trained** |
| rungs | A: 64 atoms/well, rank 16, **1000 steps** · B: 512, rank 128, **2000 steps** |
| λ | **from the certificate**: 0.0485–0.0528 used (λ* 0.0539–0.0586), `coupling_safety_frac` 0.9 |
| β | 1.0 (`full`) / **0.0** (`request_off`) |
| settling | `step_size` 0.2 · σ anneal 4.0 → 1.0 · **T = 0.0 both ends** (⇒ deterministic chain ⇒ same-key control by construction) · metropolis on · `descent_window` 100 · `descent_tol` 1e-3 · `read_mode` `state` · `launch_mode` `clamp_atom` · `request_chunk` 256 |
| JAX | main venv, `JAX_PLATFORMS=cpu` (⛔ no worktree `uv sync`) |
| `--quick` | ⛔ not used |

## ⭐ ARTEFACTS ON DISK (for the refusal and dreaming spokes)
Root: **`.claude/outputs/gen-read-cell-arm/runs/artefacts/shapes3d_recon/seed_{0,1,2}/`** — 26 files/seed, **206 MB** total.
| file | contents |
|---|---|
| `energy_traces_s{S}_{A,B}_{B,MS}_{full,request_off}.npz` | ⭐ per-chain energy **every step** (`energy` annealed **and** `energy_ref` at the fixed reference bandwidth), `accept_prob`, `accepted`, `displacement`, `sigma_scale`, `temperature`. Shapes `(1001, 72)` / `(2001, 198)` etc. ⛔ **Refusal's `(final energy, still-descending)` is computable from these without re-running the cell.** |
| `settled_states_s{S}_{A,B}_{B,MS}_{full,request_off}.npz` | `x_settled` (72, 2048), `x0` (the launch), `read` (72, 128, 4, 4), `requests` (72, 6) |
| `decoded_s{S}_{dynread,dynread__B,dynread__A_request_off,dynread__B_request_off}_{B,MS}.npz` | ⭐ `images` (72, 3, 64, 64), **`factor_tuples` — the external classifier's tuple per decoded image**, `features`, `requests`, `labels`, `factor_names` |
| `stored_atom_tuples_s{S}_{A,B}.npz` | ⭐ `atom_factor_tuples`, `distinct_atom_factor_tuples`, **`atom_rows_into_pool`** — dreaming's novelty test needs the stored atoms' tuples from **this** run (the atoms are a seeded k-means++ draw) |

Manifests: `.claude/outputs/gen-read-cell-arm/runs/cell/seed_{S}/seed_{S}/cell_manifest.json` · aggregate + verdict + same-cell proof: `runs/cell_results.json` · launch diagnostic: `runs/launch_diagnostic.json` · logs: `runs/dynread_seed{S}.log`, `runs/progress.txt` · scripts: `run_scored.sh`, `analyse.py`, `launch_diagnostic.py`, `smoke.py`.

## ⭐ COST
3 seeds × 4 rows × 2 request sets, CPU, main venv: **234 s + 421 s + 422 s = 1,077 s (18 min)** wall clock for the scored run; ~4 min for the aggregate and the launch diagnostic; 206 MB of artefacts. ⭐ The five rival arms cost **nothing** — they were copied, not re-run. (For scale, `gen-val-split-fix`'s own cell rescore took ~1,008 s **per seed**.) ⚠ JAX cold start was ~5 s, ⛔ not the 20 min the protocol warns about.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-read-cell-arm` (worktree `../CHLU-cellarm`), rebased onto `main` @ `8a404de`
- `b0a156f` wire the dynamical read onto the generation table as a sixth arm *(pre-rebase `445efb5`)*
- `6449915` a request-set suffix spelled with a slash reads as a rung label *(pre-rebase `a55c0e4` — **the hash the scored run executed**)*
- ⚠ **`main` moved under the run.** It was `7295673` at spawn and `8a404de` at hand-off; the three new commits touch `scripts/csf3/**` and `tests/test_job_cpu_gen.py` only. Rebased onto local `main` per protocol §3.5 (⛔ never `origin/main`, which is frozen at `40c2f31`); **no conflicts**, and `git diff --stat main..HEAD` after the rebase lists exactly my two files. Tests re-run green **after** the rebase (45 passed).
- **Files touched:** `chlu/experiments/exp_gen_cell.py` (+602), `tests/test_gen_cell.py` (+272) — ⛔ and nothing else. ⛔ `chlu/config.py`, `chlu/core/gen_landscape.py`, `chlu/eval/rivals/**` untouched (verified by name against `git diff --name-only main..HEAD`).
- ⭐ Branch ref verified **from the main repo** before hand-off (`git -C /Users/user/Desktop/CHLU log main..agent/experiment-engineer/gen-read-cell-arm` shows both commits) — the w4 lost-commits precaution. Worktree `../CHLU-cellarm` left in place for review.
- ⛔ Not pushed, no PR.

## Open questions / follow-ups / risks
1. ⭐⭐ **The launch beats the codebook read and settling destroys it.** The obvious next question — does a *much shorter* chain keep the launch's quality while adding continuity? — is a **new pre-registration**, ⛔ not a re-run of this one. Whoever picks it up must declare the step budget before measuring.
2. ⛔ **Nothing converged.** Either the budgets rise (with a fresh PREREG) or the arm is reported as a LOSS *and* NOT CONVERGED. ⛔ I did not choose between these.
3. ⚠ **`request_off` never settles at all** (still-descending 1.00 at every checkpoint) while collapsing to one basin — it is descending into a single minimum it never reaches. That is a fact about the landscape's β = 0 limit and may matter to the refusal threshold.
4. ⚠ `stratified_fit_val`'s clamp headroom is **exactly 512** at the shipped `n_fit` — the config sits *on* the boundary, so any `n_val` increase is silently truncated. ⛔ Not fixed here (shared file, peer wave owns it); the cell now asserts and a test pins the behaviour.
5. ⚠ `dynread/B`'s spread is **±0.1033** across seeds (0.0671 → 0.3032). ⛔ Any future rung-B number needs more than 3 seeds.

## Proposed handover updates (for the Hub)
- **§ Generation table:** add the four `dynread` rows above. ⛔ Every one carries `QUOTABLE = false`.
- **§ Live claims:** `gen-dynamical-read` PREREG §5 falsifier 1 → **answered NO**. The dynamical read is a **LOSS** against `clu` (−0.3866), `lace` (−0.1034), `ebm_compose` (tie ⇒ LOSS) and the `knn_proto` ceiling (−0.6566); it beats only the null. ⛔ The pass condition that was weakened from "beat" to "win or tie" before the run does not rescue it — it loses under both.
- **§ Negative results:** *"Continuous settling over stored atoms, at the pre-registered budgets, is worse than the codebook read it replaces — and the launch it starts from is BETTER than that codebook read. The chain is the defect."* (rung B launch 0.6852 ± 0.0173 vs `clu` 0.5949 ± 0.0082 vs settled 0.2114 ± 0.1033.)
- **§7 Known Issues, new:** `stratified_fit_val` clamps `n_val` to `pool.size − n_fit`; the shipped gen-cell config sits exactly on the boundary. `exp_gen_cell` now asserts; ⛔ the shared repair is unowned.
- **§ Owed:** one line adding `"dynread"` to `GenCellConfig.arms` (I am forbidden that file); reconciliation of §2b's `lace` 0.3241 against the on-disk 0.3117; correction of "`knn_proto` renders 72/72" to the measured 56.00 ± 3.74.
