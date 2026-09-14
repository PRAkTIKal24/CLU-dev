# gen-baselines — results-analyst report (consolidated from the runs that already exist)

**Task + acceptance criterion:** write the deliverable the `gen-baselines` spoke never wrote, from its on-disk
artefacts, with the C numbers **re-scored under A7** and the **byte position declared in the table row**.
**Status: PARTIAL — and the partial is the finding.** Every composition number, every byte-ledger number, both
LACE temperatures, both selection rules and the full prereg scorecard are recovered and reported below.
⛔ **The A7 re-score is NOT RUN and CANNOT be computed from the artefacts** — proof in §0 — so the controlled C
column ships empty with a filed prediction, never a number. The post-change suite run is also **NOT RUN**
(started, truncated at ~50 %, no result line).

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 5 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⭐ **THE A7 RE-SCORE IS BLOCKED ON CODE, NOT ON COMPUTE.** `exp_gen_baselines`, `gen_knn.py`,
>    `gen_lace.py`, `gen_request.py` contain **zero** references to `a7`/`redundant` on `main`, and no
>    per-request refusal score is persisted. ⇒ the controlled column needs a **wiring commit
>    (`experiment-engineer`) + a ~110 min 3-seed refusal re-run**, not a spreadsheet. This is
>    `gen-c-set-a7`'s own reconciliation item 1, still unowned. (§0, §4)
> 2. ⭐ **Half the re-score is already done and needs no run.** A7's registry leaves `out_of_range`'s control at
>    the ordinary 6-term valid pool, so **`out_of_range` AUROC is A7-clean as shipped**: kNN **0.8045 ± 0.1227**,
>    LACE best **0.4694 ± 0.1368**. Only the `duplicate_factor` half and the pooled number are contaminated. (§4)
> 3. ⭐ **THE TRIGGER FIRES on composition — and the mandatory re-run is provably a no-op for that cell.**
>    `knn_proto` **0.8884** beats `clu_wide` 0.7484 and `read` 0.6759. But `prototype_direct_read(embeddings)`
>    (source-verified) **never touches the byte-sized store**, and the prototype table is 475,136 B — inside
>    *both* budgets. ⇒ the composition cell is **byte-budget-invariant**; the 31 % advantage lives entirely in
>    the **refusal** cells, where kNN loses. (§3, §5)
> 4. ⚠ **R27.3's whitening gate fires on 2 of 3 seeds and the fallback is NOT RUN.** The pre-registered rule
>    selected a **non-exact** whitening on seeds 0 (`1e-3`, |diag−1| = 0.2382) and 2 (`ledoit_wolf`, 0.1438);
>    `prior_identity_near_exact = false` on both. The run predates R27. Nearest-exact fallback configs exist in
>    the trial grid with valid-only scores (§6c) but were never decoded. (§6c)
> 5. ⚠ **`include_coupling_bytes: bool = True` is now a WRONG default** in `chlu/config.py::GenBaselinesConfig`
>    (R26 ruling 3 removed the coupling from the deployed store). Any future `exp-gen-baselines` run inherits the
>    688,128 B budget silently. Needs `experiment-engineer`. (§3)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — competitor arms, consolidated.* ⛔ No CLU number is produced here. CLU rows appear only as
  **cited reference rows** from `gen-read-compose`, clearly marked.
- **Laundering control:** N/A — **these arms ARE the control.**
- **Falsifies:** N/A for a dial. This spoke's inherited falsifier is `gen-baselines`' own **P6** (§7).
- **Does NOT falsify:** a competitor beating us is the control working and is reported unsoftened (§5).

---

## 0. ⛔ PROOF that the A7 re-score is not derivable — established before any prediction was written

| check | command | result |
|---|---|---|
| A7 test in `main` | `git cat-file -e main:tests/test_gen_c_set_a7.py` | **exists** (`3737757`) — precondition met |
| A7 artefact on disk | `ls .claude/data/gen/shapes3d/a7_redundant.json` | **69,217 B, 72 records**, `kind: valid_redundant`, 7 terms each — precondition met |
| arms read A7? | `git show main:chlu/experiments/exp_gen_baselines.py \| grep -i "a7\|redundant"` | ⛔ **no match**; same for `gen_knn.py`, `gen_lace.py`, `gen_request.py` |
| per-request scores persisted? | lists > 10 elements in any manifest | only `tuning.trials` (36) and `or_not_calibration.grid` (16). Refusal ships as `score_summary.{mean,std,n}` + AUROCs only |
| ⇒ verdict | — | ⭐ **the arms never scored an A7 request and the raw scores are gone.** A re-score requires a wiring commit + a re-run. **NOT RUN.** |

Cost of the blocked run, derived from `wall_clock_s` in the manifests (not guessed): kNN 476 / 285 / 277 s,
LACE 1646 / 3494 / 1988 s, total 2190 / 3923 / 2390 s ⇒ **≈ 2.4 h for 3 seeds end-to-end**, or ≈ 18 min if only
the refusal path is re-scored on 72 extra requests. Plus JAX cold start (protocol §4).

**Pre-registered prediction for that run** — `.claude/outputs/gen-baselines-consolidate/PREREG.md`, written
before this report: kNN dup **0.30** [0.15–0.45], LACE/ode dup **0.95** [0.85–1.00], oor **exactly unchanged**,
pooled **= ½(dup+oor) exactly**. ⛔ These are predictions. They are not in any results table.

---

## FLAG-PROVENANCE TABLE (protocol §5, mandatory)

| | value |
|---|---|
| artefacts consolidated | `.claude/outputs/gen-baselines/runs/shapes3d_recon/{baselines_results.json, seed_{0,1,2}/baselines_manifest.json}`, `seed{0,1,2}.log`, `suite_before.log`, `suite_after.log`, `progress.txt`, `run_seed.sh` |
| producing spoke's branch | `agent/experiment-engineer/gen-baselines` — `4584ba6`, `58dfef9`, `0d4e60e`, `d050a4f`, `f2d6113`, `20c9ac9`, `c016823`, `145aacf` (8 commits) |
| producing spoke's base | `main @ 9a4457d`; **merged to `main` at `1bcfcb9`** |
| runs executed at | branch tip **`145aacf`** (from `suite.sh`'s HEAD echo and `run_seed.sh`'s worktree `../CHLU-gen-baselines`) |
| my base | ⚠ `main @ 5fee3f4`, **not** the `3faef27` named in my task — `main` had already moved (R28 wave-5 merge). ⛔ Read-only; no branch, no commit, no worktree. |
| seeds | **0, 1, 2** (all three, exit 0 at 15:20:58 / 16:27:35 / 17:10:06 on 2026-09-12, `progress.txt`) |
| command (verbatim, per seed) | `PYTHONPATH=/Users/user/Desktop/CHLU-gen-baselines CHLU_GEN_DATA_ROOT=…/.claude/data/gen CHLU_GEN_BRIDGE_ROOT=…/gen-feature-bridge/runs CHLU_GEN_DECLIFT_DIR=…/gen-decoder-lift/winner CHLU_GEN_STORE_RUNS=…/gen-store-grouping/runs /Users/user/Desktop/CHLU/.venv/bin/python -u -m chlu.experiments.exp_gen_baselines --dataset shapes3d --pretext recon --stage baselines --arm all --seed S --out …/runs/shapes3d_recon/seed_S` |
| CLI overrides | **none** beyond `--dataset --pretext --stage --arm --seed --out`. Every value below is the shipped `GenBaselinesConfig` default **in effect**, listed because the defaults *are* the declared choices. |
| dataset / pretext / condition | `shapes3d` / `recon` / `recomb2element` (Montero) |
| ⭐ decoder | `decoder_source="decoder_lift_v1"`, `recipe="v1_capacity_schedule"`, `assert_decoder_hash=True`, **param_hash `01ef91f7…ab196`**, `param_count` 1,516,227, `superseded=False` ⇒ ⭐ **the `dec_recipe="baseline"` trap was avoided; the run is not void** |
| encoder / cache | encoder hash `98d32602…29c0d`, cache sha256 `d6728f01…4f2e2`, 480,000 rows, `scope="full"`, `verify_cache_checksum=True`, re-encode check max\|Δ\| **0.0** |
| ⛔ byte budget in effect | **`include_coupling_bytes=True` ⇒ 688,128 B** = `store_bytes` 524,288 (from `store_bytes.json`) + coupling 163,840 (`K(K−1)/2·M²·4`, `K=5`, `M=64`, from `store_plan.json`). **Superseded by R26 ruling 3.** See §3. |
| rows | fit 32,256 · val 512 · A 1,024 · B 72 · matched-seen 198 (seed 0) / 199 (seeds 1,2) · C 72 (36 `duplicate_factor` + 36 `out_of_range`, read from `c_invalid.json`, never regenerated) |
| leakage check | `n_holdout_leaked = 0` of 72, over 32,768 checked rows, all three seeds |
| kNN grid (36) | `k∈{1,3,5}` × `metric∈{euclidean,cosine}` × `normaliser∈{none,zscore}` × `deployment∈{fp32,fp16,int8}`; `knn_store_method="kmeans"`, `knn_kmeans_iters=25`, `knn_refusal_reduce="min"` |
| LACE grid (36) | `σ∈{0.03,0.1,0.3,1.0}` × `form∈{categorical,continuous,both}` × `shrinkage∈{none,ledoit_wolf,1e-3}` (⚠ **A1+A2 amended**, train-only diagnostics) |
| LACE constants | `n_epochs=50` (`OFFICIAL IMPLEMENTATION:`), `batch=128`, `lr=1e-4`, `warmup_iters=1000`, `ld_steps=100`, tune schedule `tune_epochs=6`/`tune_n=256`/`tune_ld_steps=30`; ⛔ staircase decay not wired, ⛔ `dis_temp` not swept |
| LACE samplers | `ode` (VP prob-flow, dopri5, atol=rtol=1e-3) · `ld_published` (std=1e-2 ⇒ `T_eff=std²/2lr=0.005`) · `ld_corrected` (std=√(2·lr)=0.14142) |
| refusal threshold | `refusal_valid_quantile=0.95`, set on **A only** (valid, seen); ⛔ no C item, no held-out item |
| OR/NOT side budget | 16 trials (`cof∈{5,20,80,320}` × `not_num∈{0.03,0.1,0.3,1.0}`), ⛔ **not** in the 36; selected `cof=5.0`, `not_num=0.03` vs published 20.0 / 0.1. ⭐ No A/B/C request uses OR or NOT, so these enter **no reported number**. |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, no `uv sync` in the worktree (protocol §4). ⛔ CSF3 unreachable (2FA) — local CPU only. |
| ± convention | **population sd (ddof=0)** — the convention the shipped aggregate and the R26/R27 ledger use (`0.8373 ± 0.0109` reproduces). Sample sd (ddof=1) in `[…]` in `.claude/outputs/gen-baselines-consolidate/tables.md`. |

**Every number below traces to a file.** Full per-seed extraction:
`.claude/outputs/gen-baselines-consolidate/tables.md` + `tables.json` (generator: `extract.py`, read-only).

---

## 1. ⭐ COMPOSITION — in-corner `shape` balanced accuracy (R23 gate statistic; AUROC companion, never the gate)

Positives = the arm's renders of the 72 B requests; negatives = its renders of the matched-seen requests
(same corner, seen shape). **Chance 0.5.** ⭐ Both ceilings on every row, per R23.

| arm | **balanced accuracy** | AUROC | per seed | ⭐ **byte position** | read channel |
|---|---|---|---|---|---|
| **decoder ceiling** (real features) | **0.9414 ± 0.0179** | 0.9886 ± 0.0119 | 0.9223 / 0.9365 / 0.9653 | — | — |
| **STORE ceiling** (R22/R23) | **0.6866** | — | — | 524,288 B | 30 bits |
| ⛔ **`knn_proto`** (byte-matched prototype) | ⭐ **0.8884 ± 0.0088** | 0.9912 ± 0.0084 | 0.8763 / 0.8920 / 0.8970 | ⭐ **475,136 B table, inside BOTH the 524,288 B and 688,128 B budgets — this cell is budget-INVARIANT (see §3)** | **unbounded** (2,048 floats) |
| `knn` (channel-bottlenecked) | **0.5206 ± 0.0487** | 0.5441 ± 0.0536 | 0.4621 / 0.5182 / 0.5813 | ⚠ **OVER-BUDGET: fitted at 688,128 B, 31 % more than the deployed 524,288 B** | 41.8 / 25.1 / 41.8 bits |
| `knn_compsel` (2nd selection rule) | **0.5113 ± 0.0504** | 0.5534 ± 0.0609 | 0.4621 / 0.4912 / 0.5805 | ⚠ same, **over-budget** | 41.8 / 41.8 / 41.8 bits |
| ⭐ **`lace_ode`** — LACE's BEST composition | **0.6797 ± 0.0207** | 0.7844 ± 0.0103 | 0.7077 / 0.6732 / 0.6582 | n/a (classifier + whitening, not byte-matched) | unbounded |
| `lace_ld_corrected` (std = √(2·lr)) | **0.6401 ± 0.0193** | 0.7389 ± 0.0452 | 0.6446 / 0.6612 / 0.6146 | n/a | unbounded |
| `lace_ld_published` (std = 1e-2, T_eff 0.005) | **0.5928 ± 0.1059** | 0.7667 ± 0.0506 | 0.7418 / 0.5052 / 0.5314 | n/a | unbounded |
| ⛔ *cited reference (not mine):* `clu_wide` | *0.7484 ± 0.0307* | *0.7931* | — | 524,288 B | **320 real weights** (dof 315) |
| ⛔ *cited reference (not mine):* `read` | *0.6759 ± 0.0107* | *0.7429* | — | 524,288 B | **30 bits** |

CLU rows cited from `.claude/outputs/gen-read-compose.md` §3–§4. ⭐ R27's requested channel comparison,
in one place: `read` **30 bits** · `clu_wide` **320 real weights** · `knn_proto` **unbounded, 2,048 floats**
· `knn` **25–42 bits**. ⛔ `knn_proto` is **byte**-matched, never channel-matched.

**A and B probe accuracy** (⚠ `shape` always separate from the aggregate; four factors sit at ≈1.0):

| arm | A aggregate | B aggregate | B `shape` | B exact-match |
|---|---|---|---|---|
| decoder ceiling | **0.9902 ± 0.0030** | — | — | — |
| `knn_proto` | 0.8111 ± 0.0245 | 0.8410 ± 0.0404 | **1.0000 ± 0.0000** | 0.0694 / 0.1111 / 0.2639 |
| `knn` | 0.4976 ± 0.0127 | 0.4823 ± 0.0588 | 0.6435 ± 0.0985 | **0.0000** |
| `lace_ode` | 0.3330 ± 0.0136 | 0.3102 ± 0.0544 | 0.4954 ± 0.0925 | **0.0000** |
| `lace_ld_corrected` | 0.3041 ± 0.0211 | 0.2955 ± 0.0430 | 0.4213 ± 0.0578 | **0.0000** |
| `lace_ld_published` | 0.2979 ± 0.0068 | 0.2616 ± 0.0985 | 0.3148 ± 0.2881 | **0.0000** |

⚠ B `shape` is a **one-class detection rate** (R19 carry-forward) — ⛔ never read as a composition score; the
in-corner balanced accuracy above is the like-for-like number.

---

## 2. ⭐ REFUSAL — reported **BY KIND**, pooled as a companion only (R27.4)

⭐ **Structural fact, verified to machine precision:** with 36 C items per kind, pooled AUROC(B vs C) =
**exactly** ½(dup + oor) (`max |pooled − ½(dup+oor)| = 1.11e-16` over all arms × seeds). ⇒ **the pooled number
is 50 % driven by the contaminated half.** That is why it is a companion.

### 2a. `out_of_range` — ✅ **A7-CLEAN AS SHIPPED** (its declared control is the ordinary 6-term valid pool;
`gen-c-set-a7` measured the no-model structural detector at **0.5000** on exactly this pairing)

| arm | AUROC(B vs C·oor) | per seed |
|---|---|---|
| ⭐ **`knn` / `knn_proto`** (identical — same store-distance score) | **0.8045 ± 0.1227** | 0.6628 / 0.8754 / 0.8754 |
| `knn_compsel` | 0.7722 ± 0.1308 | 0.6628 / 0.7369 / 0.9171 |
| `lace_ld_corrected` — LACE's best on this kind | **0.4694 ± 0.1368** | 0.6130 / 0.4545 / 0.3407 |
| `lace_ld_published` | 0.4383 ± 0.1302 | 0.2882 / 0.5216 / 0.5050 |
| `lace_ode` | 0.3971 ± 0.1084 | 0.5135 / 0.3789 / 0.2990 |

⭐ **LACE is an anti-detector on `out_of_range`: all three samplers sit BELOW chance** (0.397–0.469). It assigns
*lower* composed energy to an out-of-range request than to a valid unseen one. ⛔ This **falsifies the spoke's
own P3b derivation** ("the continuous energy handles an off-grid value natively and blows up quadratically") —
see §7. The extrapolated prototype, by contrast, lands far off-manifold exactly as P3a's derivation said.

### 2b. `duplicate_factor` — ⛔ **CONTAMINATED. Two columns, never merged.**

| arm | ⛔ **CONTAMINATED** AUROC(B vs C·dup), control = 6-term B | per seed | ⭐ **A7-CONTROLLED**, control = `a7_redundant` |
|---|---|---|---|
| `knn` / `knn_proto` | **0.3907 ± 0.1670** | 0.2137 / 0.5455 / 0.4128 | ⛔ **NOT RUN** (§0) |
| `knn_compsel` | 0.3711 ± 0.1511 | 0.2137 / 0.3846 / 0.5150 | ⛔ **NOT RUN** |
| `lace_ode` | **0.9991 ± 0.0012** | 1.0000 / 0.9977 / 0.9996 | ⛔ **NOT RUN** |
| `lace_ld_published` | 0.9925 ± 0.0126 | 1.0000 / 0.9996 / 0.9780 | ⛔ **NOT RUN** |
| `lace_ld_corrected` | 0.9697 ± 0.0257 | 0.9402 / 0.9815 / 0.9873 | ⛔ **NOT RUN** |

⛔ **The contamination, stated so no one quotes the left column as clean:** all 36 `duplicate_factor` C records
carry **7 terms** against A/B's **6**, so a bare term counter scores **1.0000** on this half (and four ordering
features score 0.9167). `gen-c-set-a7` measured both, and measured **0.5000 on all 11 features after A7**.
⚠ A high left-column number is therefore **not attributable to the arm** without the A7 control — that applies
in full to LACE's 0.9991, which is the single most flattering competitor number in this report.

### 2c. Pooled — ⛔ companion only, ⛔ contaminated on the dup half

| arm | AUROC(B vs C) | AUROC(A vs C) | AUROC(A∪B vs C) | ⭐ **ORACLE** AUROC(A∪B vs C) |
|---|---|---|---|---|
| `knn` / `knn_proto` | 0.5976 ± 0.1419 | 0.8140 ± 0.0481 | **0.7998 ± 0.0443** | ⭐ **0.8373 ± 0.0109** |
| `knn_compsel` | 0.5717 ± 0.1392 | 0.7999 ± 0.0357 | 0.7849 ± 0.0424 | — |
| `lace_ld_corrected` | **0.7195 ± 0.0563** | 0.7184 ± 0.0181 | 0.7185 ± 0.0181 | **1.0000 ± 0.0000** |
| `lace_ld_published` | 0.7154 ± 0.0625 | 0.7997 ± 0.0447 | 0.7942 ± 0.0377 | **1.0000 ± 0.0000** |
| `lace_ode` | 0.6981 ± 0.0544 | 0.7391 ± 0.0132 | 0.7364 ± 0.0137 | **1.0000 ± 0.0000** |

⭐ **R27.2's two numbers reproduce exactly from the artefacts:** kNN **oracle 0.8373 ± 0.0109** vs
**deployable 0.7998 ± 0.0443** (A∪B vs C), and (iv) reads the oracle.
⚠ **Reporting artefact found in the shipped aggregate, fix it before quoting:** `baselines_results.json` files
LACE's oracle bound under the per-seed *best-composition sampler*, so it reads `n_seeds=1` for `ld_published`
and `n_seeds=2` for `ode`. The bound is **per-family and identical on all three seeds: 1.0000, n=3**
(`arms.lace.tuning.ORACLE_UPPER_BOUND_auroc_c_vs_b.value` = 1.0 on seeds 0, 1, 2). ⛔ Not a missing run.

### 2d. Operating point at the deployed threshold (A-only 95th percentile)

| arm | refused on C | ⚠ **falsely refused on B** (valid, unseen) | refused on A |
|---|---|---|---|
| `knn` / `knn_proto` | 0.5602 ± 0.0424 | ⚠ **0.4444 ± 0.0481** | 0.0508 |
| `lace_ld_corrected` | 0.4630 ± 0.0289 | **0.0324 ± 0.0212** | 0.0508 |
| `lace_ld_published` | 0.5648 ± 0.0488 | 0.1806 ± 0.1389 | 0.0508 |
| `lace_ode` | 0.5185 ± 0.0212 | 0.1250 ± 0.1368 | 0.0508 |

⚠ The kNN arm's threshold, calibrated on **seen** valid data, refuses **44 % of valid unseen requests**. A
distance-to-store rule cannot tell "unseen but satisfiable" from "unsatisfiable" — a mechanism statement the
AUROC hides, and it is the same statement in both directions (its `out_of_range` AUROC is its best cell).

---

## 3. ⭐ THE BYTE POSITION — declared, in the table row, not in a footnote

| structure | bytes | status |
|---|---|---|
| the deployed store (modes) | **524,288** | `n_modes·D·4 = 64·2048·4`, from `store_bytes.json` |
| inter-well PMI coupling | **163,840** | `K(K−1)/2·M²·4 = 10·64²·4`, from `store_plan.json`'s own `K=5`, `M=64` |
| ⛔ **the target these runs used (A5)** | **688,128** | = 524,288 + 163,840. **SUPERSEDED by R26 ruling 3**: with conditional dreaming dropped, the coupling leaves the deployed store |
| ⇒ ⚠ **the kNN arm's declared position** | ⭐ **OVER-BUDGET by 163,840 B = +31.3 %** | ⛔ **NOT byte-matched.** Reported as an **unbounded upper bound** (over-budget precedent) |

Realised entry counts at 688,128 B vs the correct 524,288 B: fp32 **84** vs 64 · fp16 **168** vs 128 ·
int8 **328** vs 248 (+31.2 % / +31.2 % / +32.3 %). Every deployment landed on the budget **exactly, 0 slack**
(`byte_ledger.*.slack_bytes = 0`, `within_budget = true`) — so P4 holds and the over-resourcing is exactly the
31 %, not more. The superseded store-only ledger ships alongside as `byte_ledger_store_only_SUPERSEDED`.

⭐ **Which cells the 31 % can actually touch — source-verified, not asserted.**
`chlu/eval/rivals/gen_knn.py::prototype_direct_read(embeddings)` takes **only the composed embeddings**; the
byte-sized k-means store is never consulted. The prototype table is `(Σ_k C_k + 1)·D·4 = 58·2048·4 =` **475,136 B**,
fixed by the factor grid, **inside both budgets**; the manifest records `channel.selection = "N/A — the
prototype read depends on NO tuning axis"`.

| cell | touches the budget-sized store? | consequence |
|---|---|---|
| `knn_proto` **composition** (0.8884) | ⛔ **no** | ⭐ **byte-budget-invariant**; a matched re-run at 524,288 B returns the same number |
| `knn` **composition** (0.5206) | ✅ yes (328 vs 248 entries) | over-resourced; it loses to `read` anyway, so the over-resourcing is conservative |
| **all refusal** cells (both variants) | ✅ yes (score = distance to that store) | ⚠ **over-resourced by 31 %**; the kNN refusal numbers are upper bounds |

⚠ **Head-ruling caveat carried forward:** this is *declared, not fixed*. ⛔ Nobody may describe the kNN arm as
byte-matched. Its **refusal** numbers are an over-resourced upper bound and must be quoted that way.
⚠ **A second byte fact neither ledger carries** (`gen-read-compose` §3): the CLU **addressing head** is
**74,240 B** while the prototype arm's head is **475,136 B — 6.4× larger** — both excluded by the same
`store_bytes.json` convention. The day anyone counts heads, that reverses the direction of the byte story.

### ⛔ THE TRIGGER, EVALUATED EXPLICITLY (my task item 3)
> *"if the kNN arm **beats or ties** CLU on any headline cell, a matched re-run at 524,288 B becomes
> **mandatory before that cell is quoted**."*

⭐ **THE TRIGGER FIRES.** On the composition headline, `knn_proto` **0.8884 ± 0.0088** beats every CLU row:
`clu_wide` 0.7484 (**+0.1400**, 0/3 seeds for us, 4.6× spread), `read` 0.6759 (**+0.2125**), the store ceiling
0.6866 (**+0.2018**). ⛔ It does **not** fire on refusal: kNN's deployable 0.7998 and oracle 0.8373 both sit
below the certificate's barrier 0.9146 (R26), and on `out_of_range` it is the *only* rival with power.

⭐ **What the mandatory re-run would return, and why I can say it without running it:** the fired cell is
`knn_proto` composition, and §3 shows that cell **never reads the budget-sized store**. A matched re-run at
524,288 B changes `n_entries` 328→248, which enters `knn_proto`'s composition through **no code path**.
⇒ the re-run is a **confirmation, not a correction**. ⚠ I still recommend executing it (cheap: prototype fit
only) rather than quoting an argument, because "the cell is invariant" is a source-reading and the trigger asks
for a measurement. ⛔ Until it runs, the honest phrasing is: *the prototype arm beats us at 0.8884 and its
advantage is not explained by the byte error.* That is the harder-for-us position and it is the correct one.

---

## 4. ⭐ THE C SET, IN BOTH FORMS — the summary table the task asked for

⛔ **Never one column.** ⛔ **No number here is quotable without its set label.**

| arm | kind | ⛔ CONTAMINATED (control = B, 6 terms) | ⭐ A7-CONTROLLED (control per A7 registry) |
|---|---|---|---|
| `knn`/`knn_proto` | `out_of_range` | 0.8045 ± 0.1227 | ✅ **0.8045 ± 0.1227** — *unchanged by construction: A7 leaves this kind's control at the 6-term valid pool* |
| `knn`/`knn_proto` | `duplicate_factor` | 0.3907 ± 0.1670 | ⛔ **NOT RUN** |
| `knn`/`knn_proto` | *pooled (companion)* | 0.5976 ± 0.1419 | ⛔ **NOT RUN** (= ½(dup_A7 + 0.8045) once dup lands) |
| LACE best-per-kind | `out_of_range` | 0.4694 ± 0.1368 (`ld_corrected`) | ✅ **0.4694 ± 0.1368** — unchanged, same reason |
| LACE best-per-kind | `duplicate_factor` | 0.9991 ± 0.0012 (`ode`) | ⛔ **NOT RUN** |
| LACE best-per-sampler | *pooled (companion)* | 0.7195 ± 0.0563 (`ld_corrected`) | ⛔ **NOT RUN** |
| no-model structural detector | `duplicate_factor` | **1.0000** | **0.5000** (measured by `gen-c-set-a7`) |
| no-model structural detector | `out_of_range` | 0.5000 | 0.5000 |
| no-model structural detector | *pooled* | **0.7500** | **0.5000** |

⭐ **R27.4's differentiator, stated with its set labels attached so it is auditable — and ⛔ still NOT
QUOTABLE, exactly as R27.4 says.** On the shipped (contaminated) scoring, kNN covers `out_of_range` **0.8045**
and fails `duplicate_factor` **0.3907**; LACE is the mirror, **0.3971–0.4694** / **0.9697–0.9991**.
⭐ **Neither rival covers both kinds.** ⛔ The `duplicate_factor` leg of that sentence rests on the contaminated
column for *both* rivals and becomes quotable only after the A7 re-score. The `out_of_range` leg is clean now,
and on its own it already says: **LACE cannot detect an off-grid value at all — it is below chance.**

---

## 5. Where the competitors actually stand (interpretation, ⛔ no kill-rule verdict — `gen-cell-run`'s job)

1. ⭐ **The prototype arm is the real competitor and it wins on composition, unsoftened.** `knn_proto`
   **0.8884 ± 0.0088** vs the store ceiling 0.6866 (**+0.2018**) and `clu_wide` 0.7484 (**+0.1400**, 0/3 seeds
   for us). It is reproduced independently by `gen-read-compose`'s own `proto` arm at **0.8817 ± 0.0023** — two
   spokes, different row draws, inside one seed spread. ⛔ This is not a measurement error and it is not the
   byte error. **Additive prototype means solve recombination-to-element on Shapes3D once the features are
   decent** — R27.1 already anticipated exactly this reading, and it costs 30 bits' worth of nothing: the arm
   ships 2,048 floats to the decoder.
2. ⭐ **The channel is where the rival is beaten, and only there.** At a *bounded* channel the same family
   collapses: `knn` at 25–42 bits reads **0.5206 ± 0.0487**, i.e. **chance**, against our 30-bit `read` at
   0.6759. ⛔ That is a **bits-efficiency statement, not a rescue** (R27.1's own words), and it must be
   presented as such.
3. ⭐ **The two rivals have disjoint, complementary refusal failures** — kNN blind to `duplicate_factor`
   (0.3907, **below chance**), LACE blind to `out_of_range` (0.3971–0.4694, **below chance**). Both failures are
   mechanical, and both derivations were pre-registered: the averaged same-factor prototypes land *near* the
   manifold; the summed composed energy has no vocabulary for an off-grid value.
   ⚠ Only LACE's half of that story is contaminated; kNN's `out_of_range` competence is clean.
4. ⚠ **LACE is the stronger baseline on paper and the concessions are why — stated as asymmetries, not
   caveats.** It gets **dense factor labels** over the train split (our store gets none), and it reaches
   **0.9521 / 0.9867 / 0.9893** mean per-factor validation accuracy on its own supervision (seeds 2 / 0 / 1) — an almost perfect factor
   classifier. It nevertheless composes at **0.6797**, below `knn_proto`'s 0.8884 and only +0.004 above our
   `read`. ⇒ **dense supervision does not buy composition on this cell.** ⛔ `DO_NOT_USE_AS_A_BAR`: the paper's
   0.94–0.98 / FID 6.63 are StyleGAN at up to 1024², their splits, their labels — not comparable.

---

## 6. Build fidelity: the things that could have hobbled a rival, checked

### 6a. Tuning-trial budget — ⭐ **36 per arm, equal** (Head constraint), verified from the manifests
`knn.tuning.budget = 36` (3 k × 2 metrics × 2 normalisers × 3 deployments) and
`lace.tuning.budget = 36` (4 σ × 3 forms × 3 shrinkages), all three seeds. Declared **outside** the 36 and not
folded in: LACE's **16**-point OR/NOT calibration (⭐ exercised by no reported number — every A/B/C request is a
pure conjunction) and kNN's **2**-point refusal-reduction sensitivity (`min` = declared headline **0.5976**;
`mean` = 0.5755 ± 0.1387, both ship).

### 6b. Both LACE temperatures ran, and **LACE's best is reported** (R9.4)

| sampler | T_eff | composition (in-corner balacc) | refusal pooled AUROC(B vs C) ⛔contaminated | `out_of_range` ✅clean |
|---|---|---|---|---|
| `ode` (VP prob-flow) | — | ⭐ **0.6797 ± 0.0207 (BEST)** | 0.6981 ± 0.0544 | 0.3971 ± 0.1084 |
| `ld_published` (std = 1e-2, as-published) | **0.005** | 0.5928 ± 0.1059 | 0.7154 ± 0.0625 | 0.4383 ± 0.1302 |
| `ld_corrected` (std = √(2·lr) = 0.14142) | ≈1 | 0.6401 ± 0.0193 | ⭐ **0.7195 ± 0.0563 (BEST)** | ⭐ **0.4694 ± 0.1368 (BEST)** |

⭐ **LACE's best = ODE 0.6797 on composition, `ld_corrected` 0.7195 on pooled refusal.** The published
temperature is ~200× colder than its own target and it is **the worst sampler on both axes** — the correction
is what makes LACE competitive, which is the direction the never-weaken-a-competitor rule demands.

### 6c. ⚠ R27.3's whitening gate — **it fires on 2 of 3 seeds, and the fallback is NOT RUN**
Whitening was mandatory and full (`lace_option = 1`, D = 2048 ≤ 20000, N = 32,256; ⛔ no PCA, ⛔ no channel-only,
⛔ no learned prior). `prior_identity_near_exact` = `mean|diag(cov(u))−1| ≤ 0.05`:

| seed | pre-registered selection | `mean|diag−1|` | exact? | nearest-exact fallback in the grid | its val_agg | its val_shape | its ORACLE-only AUROC(B vs C) |
|---|---|---|---|---|---|---|---|
| 0 | σ=0.1, `both`, **`1e-3`** | **0.2382** | ⛔ **no** | σ=0.1, `both`, `none` (\|diag−1\| 0.0570) | 0.2266 (sel. 0.2454) | 0.4492 (sel. 0.4609) | 0.7741 (sel. 0.9230) |
| 1 | σ=0.03, `both`, **`none`** | **0.0567** | ✅ yes | — | — | — | — |
| 2 | σ=0.03, `both`, **`ledoit_wolf`** | **0.1438** | ⛔ **no** | σ=0.03, `both`, `none` (\|diag−1\| 0.0510) | 0.2617 (sel. 0.2721) | 0.4883 (sel. 0.5430) | 0.6659 (sel. 0.6647) |

⛔ **The fallback was never decoded** — the run (2026-09-12) predates R27 (2026-09-13), and only the selected
configuration is passed through the decoder; non-selected trials carry validation scores and oracle-only AUROCs
and nothing else. ⇒ the R27.3 "both reported, the stronger quoted" requirement is **NOT RUN**.
⚠ What the valid-only evidence says, labelled as an inference and not a result: the nearest-exact config loses
on validation on both seeds (−0.019 and −0.010 aggregate), so "the stronger" would most likely remain the
shipped configuration — ⛔ but that is validation, not the reported metric, and R27.3 asks for the decode.
⚠ Also: on seeds 0 and 2, LACE ran with the prior-score omission **not licensed** (R9.1). It is our own
selection code that did it, which is exactly what R27.3 named.

### 6d. Whitening finding (A2) reproduced on all three seeds
`mean|diag(cov(u))−1|`: `1e-3` **0.2382 / 0.2291 / 0.2155** · Ledoit-Wolf (λ 3.49e-4 / 3.52e-4 / 4.01e-4)
**0.1502 / 0.1425 / 0.1438** · ⭐ `none` + rel. floor 1e-8 **0.0570 / 0.0567 / 0.0510**, flooring **73 / 72 / 72**
near-null directions. ⭐ **The shrinkage is itself what breaks `p₀(u) = N(0,I)`** — and the floored count
independently corroborates `gen-store-grouping`'s 71 dead post-ReLU coordinates.

### 6e. Anti-hobbling checks that passed
`no_leak`: 0 of 72 held-out rows leaked into the 32,768 fit rows, all seeds. Cache re-encode max\|Δ\| **0.0**.
Off-grid handling: **36 of 468** request terms are off-grid, **linearly extrapolated, max 17.14 distance units
outside range** — ⛔ never clipped to the nearest grid value (clipping would have handed the kNN arm a free win
on `out_of_range`, and that cell is its best one). `A, B, C` read from disk, never regenerated.

### 6f. ⭐ NEW: no valid-only statistic selects for refusal power — now with a number (supports R27.2)
Spearman ρ over the 36 trials between the selection criterion and ORACLE-only AUROC(B vs C)
(`.claude/outputs/gen-baselines-consolidate/selection_power.md`; ⛔ the AUROCs are contaminated on the dup half):

| family | ρ(val_aggregate, AUROC) | ρ(val_`shape`, AUROC) | validation rank of the best-AUROC trial |
|---|---|---|---|
| `knn` | **−0.084 ± 0.210** | +0.056 ± 0.090 | **34 / 33 / 29** of 36 |
| `lace` | ⭐ **−0.588 ± 0.091** | −0.363 ± 0.143 | 33 / 17 / 19 of 36 |

⭐ **For LACE the pre-registered criterion is strongly ANTI-correlated with detection power**, and for both
families the best refusal configuration sits in the bottom third of the validation ranking. ⭐ **Near-tie
sensitivity, seed 2:** the top two LACE trials differ by **0.0020** in validation aggregate and by **0.334** in
AUROC(B vs C) (0.6647 vs 0.9990). ⇒ R27.2's ruling that kill rule (iv) reads the **oracle** is not a courtesy;
it is forced by the measurement. ⛔ It also means LACE's *deployable* refusal number is close to arbitrary.

---

## 7. ⭐ THE SPOKE'S OWN PREREG, SCORED — including the falsifications

`.claude/outputs/gen-baselines/PREREG.md` + amendments A1–A5. Metric = in-corner `shape` balanced accuracy.

| # | prediction | point / interval | **realised** | verdict |
|---|---|---|---|---|
| **P1a** | `knn` in-corner balacc on B | 0.62 [0.55–0.70] | **0.5206 ± 0.0487** | ⛔ **FALSIFIED** (below the interval) |
| **P1b** | `knn_proto` in-corner balacc on B | 0.90 [0.85–0.95] | **0.8884 ± 0.0088** | ⭐ **HELD** |
| **P2** | LACE best in-corner balacc on B | 0.85 [0.75–0.92] | **0.6797 ± 0.0207** (`ode`) | ⛔ **FALSIFIED** (well below) |
| **P3a** | `knn` AUROC(B vs C) pooled | 0.75 [0.60–0.90] | **0.5976 ± 0.1419** | ⛔ **FALSIFIED** (marginally below 0.60) |
| P3a′ | *its derivation*: oor ≈ 1.0, dup ≈ 0.5 | — | oor **0.8045**, dup **0.3907** | ⚠ **structure held, both levels low** |
| **P3b** | LACE AUROC(B vs C) pooled | 0.90 [0.80–0.98] | **0.7195 ± 0.0563** | ⛔ **FALSIFIED** |
| P3b′ | *its derivation*: the continuous energy blows up on off-grid values | — | `out_of_range` **0.3971–0.4694, BELOW CHANCE** | ⛔ **FALSIFIED — and this is the finding** |
| **P4** | realised kNN store bytes = 688,128 exactly, all 3 deployments (A5 restatement) | exact | **688,128 / 688,128 / 688,128**, slack **0** | ⭐ **HELD** |
| **P5a** | composition ordering ODE ≥ LD-pub ≥ LD-corr | — | ODE **0.6797** ≥ LD-**corr** 0.6401 ≥ LD-**pub** 0.5928 | ⛔ **FALSIFIED** (the two LD arms invert) |
| **P5b** | refusal: LD-corr ≥ LD-pub by ≥ +0.02 | — | +0.0041 (0.7195 vs 0.7154); per seed **+0.133 / −0.043 / −0.078**, 1/3 | ⛔ **FALSIFIED** (right sign in the mean, far below the margin, not 3/3) |
| ⭐ **P6** | **the spoke's own falsifier**: max(`knn`,`knn_proto`) ≥ 0.70 | ≥ 0.70 | **0.8884** | ⭐ **DID NOT FIRE — the kNN family was not hobbled** |

⭐ **Reading.** The two predictions that held are exactly the two that matter for anti-hobbling (P1b, P6): the
prototype arm landed inside a pre-registered interval filed **before the harness existed**, and the self-falsifier
did not fire. Everything that failed, failed in the direction of *the rivals being weaker than their author
expected on refusal* — and one of those failures (P3b′, LACE below chance on `out_of_range`) is a mechanism
finding, not a miss. ⚠ **P3a/P3b are scored against the CONTAMINATED C set**; their verdicts are provisional
until the A7 re-score lands. ⛔ P1a/P1b/P2/P5a/P5b/P6 are composition-only and are unaffected by A7.

---

## 8. ⛔ NOT-RUN INVENTORY — never a zero, never a silent gap

| item | status | evidence |
|---|---|---|
| ⭐ **A7-controlled C re-score, `duplicate_factor`, both rivals** | ⛔ **NOT RUN** | §0: no A7 wiring on `main`, no per-request scores persisted. Needs a wiring commit + ≈2.4 h (or ≈18 min refusal-only), 3 seeds |
| ⭐ **post-change full suite** | ⛔ **NOT RUN (started, truncated)** | `suite_after.log`: `=== suite after START 2026-09-13T00:32:17+02:00 HEAD=145aacf ===`, progress dots stop at **~50 %**, **no `N passed` line, no END line**. ⛔ Not a failure and ⛔ not a pass — no result exists |
| pre-change suite | ✅ **RAN, GREEN** | `suite_before.log`: `2099 passed, 13 skipped, 45 warnings in 8749.11s (2:25:49)`, `HEAD=9a4457d`, worktree `../CHLU-gb-base` |
| R27.3 exact-whitening fallback decode (seeds 0, 2) | ⛔ **NOT RUN** | §6c — the run predates R27 |
| `CLU-wide` arm | ✅ **RAN — but by `gen-read-compose`, not here** | cited, not produced: 0.7484 ± 0.0307, channel 320 real weights |
| CSF3 / cluster anything | ⛔ **NOT RUN, permanently** | 2FA, unreachable from an agent session (standing constraint) |
| dSprites, `recomb2element_wide` | ⛔ **NOT RUN** | only `runs/shapes3d_recon/` exists |
| `knn_compsel` on seed 0 | ✅ **not missing — identical by construction** | `selection_rules_coincide = true` on seed 0, so no separate variant is emitted. ⚠ The shipped aggregate reports `n_seeds = 2`; my tables carry the correct **n = 3** series with seed 0 = `knn`'s value |
| LACE oracle bound, seeds | ✅ **n = 3, not 1 or 2** | §2c — an aggregate filing artefact, not a missing run |

---

## Git footprint
**None.** ⛔ No branch, no commit, no worktree, no tracked file touched — read-only analysis as the task
specified. Artefacts written only under `.claude/outputs/gen-baselines-consolidate/`:
`PREREG.md`, `extract.py`, `tables.md`, `tables.json`, `selection_power.py`, `selection_power.md`.
Verification commands are in the flag-provenance table and §0.

---

## Open questions / follow-ups / risks
1. ⭐ **Owner needed for the A7 wiring** (reconciliation item 1). Until it lands, **no `duplicate_factor` or
   pooled C number from any arm is quotable**, including LACE's 0.9991 — which is the number most likely to be
   quoted by accident because it is the most flattering competitor result in the phase.
2. ⭐ **The trigger fired; decide whether the invariance argument suffices.** My reading is that the
   `knn_proto` composition cell is byte-invariant by source, so the matched re-run is a confirmation. If the
   Hub wants the measurement, it is the cheapest run in the phase (prototype fit + decode, no k-means, no LACE).
3. ⚠ **`include_coupling_bytes=True` is a live wrong default** — the next `exp-gen-baselines` run silently
   inherits 688,128 B. One-line fix for `experiment-engineer`; it will change every kNN **refusal** number.
4. ⚠ **The head-byte asymmetry (74,240 vs 475,136 B) is not in any ledger** and reverses the byte story if
   heads are counted. Someone should decide the convention before the paper, not after.
5. ⚠ **LACE's deployable refusal number is near-arbitrary** (§6f: a 0.002 validation tie costs 0.334 AUROC).
   Recommend the paper quote LACE's **oracle 1.0000** as its refusal bar and say why — it is the harder bar.
6. ⚠ **Neither rival was run on dSprites or the wide condition.** Every competitor claim in this phase is
   currently **one dataset, one condition, 3 seeds.**

### Recommended next experiments, in cost order
| # | run | cost | what it buys |
|---|---|---|---|
| 1 | A7 re-score, refusal path only, both arms, 3 seeds, at **524,288 B** | ≈20–40 min + wiring | closes reconciliation items 1 **and** the byte error in one run; scores PREREG Q1–Q3 |
| 2 | `knn_proto` composition at 524,288 B | ≈10 min | discharges the fired trigger with a measurement instead of a source-reading |
| 3 | R27.3 exact-whitening fallback decode, seeds 0 and 2 | ≈1 h | discharges R27.3; the never-weaken-a-competitor rule applied to our own selection code |
| 4 | both arms on `recomb2element_wide` | ≈3 h | the competitor claims stop being single-condition |

---

## Proposed handover updates (for the Hub)

**§1.6 (experiments) — fold in as `gen-baselines`, 3 seeds, `main @ 1bcfcb9`, runs at branch tip `145aacf`:**
- Composition, in-corner `shape` balanced accuracy: **`knn_proto` 0.8884 ± 0.0088** · `knn` (25–42 bit channel)
  **0.5206 ± 0.0487** · **LACE best (ODE) 0.6797 ± 0.0207** · LACE `ld_corrected` 0.6401 ± 0.0193 ·
  LACE `ld_published` 0.5928 ± 0.1059. Ceilings: decoder **0.9414 ± 0.0179**, store **0.6866**.
- ⭐ **`knn_proto` beats every CLU composition row** (`clu_wide` +0.1400, `read` +0.2125, store ceiling +0.2018)
  and is independently reproduced by `gen-read-compose`'s `proto` at 0.8817 ± 0.0023.
- Refusal **BY KIND** (R27.4). ✅ A7-clean now: `out_of_range` — kNN **0.8045 ± 0.1227**, LACE best
  **0.4694 ± 0.1368** (⭐ **all three LACE samplers below chance**). ⛔ Contaminated, NOT quotable:
  `duplicate_factor` — kNN **0.3907 ± 0.1670**, LACE **0.9991 ± 0.0012**.
- Pooled (companion, contaminated on half): kNN A∪B-vs-C **deployable 0.7998 ± 0.0443**, **oracle
  0.8373 ± 0.0109** (⭐ both R27.2 figures reproduce exactly). ⭐ **LACE oracle = 1.0000 ± 0.0000, n = 3** —
  correct the shipped aggregate's `n_seeds = 1/2`, which is a filing artefact, not a missing run.
- ⚠ Operating point: the kNN threshold **falsely refuses 44.4 % ± 4.8 % of valid unseen requests**.
- Tuning budget **36 per arm, equal**; +16 OR/NOT (LACE, exercised by no reported number), +2 reduction
  sensitivity (kNN). LACE concessions: dense factor labels (its own val accuracy 0.9521–0.9893), validation
  split for OR/NOT constants — **stated as asymmetries; it is still beaten by a prototype mean.**

**§5 (provenance):**
- ⭐ **kNN arm byte position: OVER-BUDGET at 688,128 B, +31.3 % over the deployed 524,288 B.** Declared, not
  fixed (Head ruling). **Refusal** numbers are an over-resourced upper bound. ⭐ **Composition** of
  `knn_proto` is **byte-budget-invariant** (`prototype_direct_read` never touches the store; table 475,136 B,
  inside both budgets) — source-verified in `chlu/eval/rivals/gen_knn.py`.
- ⭐ **THE TRIGGER FIRED** on the composition headline; the mandatory matched re-run is a confirmation for
  that cell, not a correction — but ⛔ it has **NOT** been executed.
- ⚠ Head-byte asymmetry for the ledger: CLU addressing head **74,240 B** vs prototype head **475,136 B**
  (6.4×), both excluded by the same convention.
- ⛔ **`.claude/outputs/gen-baselines.md` now exists**; suite: **before 2099 passed / 13 skipped (`9a4457d`)**,
  **after = NOT RUN (truncated at ~50 %, `145aacf`)**.

**§8 / risks + engineer queue:**
- `experiment-engineer`: (a) wire `a7_redundant.json` into `exp_gen_baselines` (+ `exp_gen_certificate`) —
  `gen-c-set-a7`'s unowned reconciliation item 1; (b) flip
  `GenBaselinesConfig.include_coupling_bytes` to `False` (R26 ruling 3 made `True` wrong); (c) persist
  **per-request refusal scores** in the manifest so the next re-scoring ruling needs no re-run — this report
  exists only because they were not persisted.
- ⛔ **Quoting rule to carry:** every C number carries its set label. `duplicate_factor` and pooled are
  **contaminated**; only `out_of_range` is clean today.
- ⛔ No kill-rule verdict was rendered here (task acceptance 7).
