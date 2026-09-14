# gen-arms-a7-wiring — experiment-engineer report

**Task + acceptance criterion:** make the kNN and LACE arms **scorable under A7** (per-request
refusal scores persisted at the pinned path), reproduce `out_of_range` **bit-identically**, flip
`include_coupling_bytes` to `False` with the realised **524,288 B** asserted, and run R27.3's
whitening fallback — **all four done; 6/6 seed-runs of the two declared legs complete (+1
invariance run), nothing NOT RUN.**

**Status: done.**

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⭐ **R27.4 IS NOW QUOTABLE, AND IT SURVIVES THE CONTROL FOR LACE BUT NOT FOR kNN.**
>    LACE's `duplicate_factor` **0.9991 → 0.9936 ± 0.0057** under A7 (ode) — ⛔ it was **not** the
>    term-count giveaway, it is mechanism. kNN's **0.3907 → 0.3338 ± 0.0692**. ⇒ the sentence
>    *"neither rival covers both kinds"* is now **evidenced on a controlled metric**, not on a
>    contaminated one. (§3)
> 2. ⚠ **THE A7-CONTROLLED NUMBER MOVES TWO THINGS AT ONCE AND MUST NEVER BE QUOTED BARE.**
>    A7 is built from **train** rows by ruling, so switching the control B → A7 changes the term
>    count (6→7) **and** the seen-ness of the negative pool (unseen-valid → seen-valid). The
>    dedup set `A7D` separates them: for kNN the two legs are **+0.2653 (seen/unseen)** and
>    **−0.3221 (term count)** and they nearly cancel. ⛔ A quoted A7 number needs this
>    decomposition attached. (§4)
> 3. ⭐ **THE HEAD'S MANDATORY RE-RUN TRIGGER IS NOW MEASURED, NOT ARGUED, AND IT IS A NO-OP.**
>    `knn_proto`'s in-corner balanced accuracy is **bit-identical at both budgets**
>    (0.8763 / 0.8920 / 0.8970, Δ = 0.000000 on 3/3 seeds), so the 31 % byte error never touched
>    the composition headline. `gen-baselines` §3 recommended executing this rather than quoting
>    the argument; it is executed. (§5)
> 4. ⚠ **`gen-baselines` §2a's ± is SAMPLE sd (ddof=1) while its flag-provenance table declares
>    population sd (ddof=0).** Means are bit-identical; the spreads differ by √(3/2)
>    (kNN oor 0.1002 vs the published 0.1227). ⛔ Not a new run, a convention clash to fix once. (§2)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — competitor arms brought up to the rulings.* ⛔ No CLU number, no kill-rule verdict.
- **Laundering control:** N/A — **I am maintaining the control.**
- **Falsifies:** a rival that gets *weaker* because of a change I made. ⭐ **It did not fire** — see
  §2 (bit-identity, worst |Δ| = **0.000e+00**) and §5 (at the corrected budget the kNN arm's
  `out_of_range` AUROC goes **up** 0.8045 → 0.8281 and its false-refusal rate on B goes **down**
  0.4444 → 0.3426).
- **Does NOT falsify:** the bottlenecked `knn` variant losing 0.5206 → 0.4993 of composition at the
  corrected budget. That is the **R26.3-mandated removal of an over-resourcing** (328 → 248 int8
  entries), declared in advance, ⛔ not a hobbling — and it is reported in §5 rather than buried.

## FLAG-PROVENANCE TABLE (protocol §5, mandatory)

| | value |
|---|---|
| branch / commit runs executed at | `agent/experiment-engineer/gen-arms-a7-wiring` @ **`2610100`** (echoed into `runs/progress.txt` by the runner before the first seed) |
| base | **`main @ 5fee3f4`** (unchanged at finish; `git rebase main` = "up to date") |
| worktree / env | `../CHLU-gen-a7wire`, **main venv** `/Users/user/Desktop/CHLU/.venv` via `PYTHONPATH` — ⛔ no `uv sync`. **JAX 0.9.0**, equinox 0.13.4 |
| dataset / pretext / condition | `shapes3d` / `recon` / `recomb2element` |
| seeds | **0, 1, 2** — ⭐ all six runs exit 0 (`runs/progress.txt`); ⛔ nothing NOT RUN |
| **leg H** (the regression + the reported LACE numbers) | `--arm all --byte-budget historical` ⇒ `include_coupling_bytes=True`, `expect_byte_target_bytes=None`, realised **688,128 B** = 524,288 + 163,840. ⛔ The SUPERSEDED budget, run deliberately so the published configuration reproduces |
| **leg D** (the deployed budget) | `--arm knn`, shipped defaults ⇒ `include_coupling_bytes=False`, realised **524,288 B**, **asserted** (`expected_total_bytes: 524288`, `assertion: "realised == expect_byte_target_bytes"` in all three manifests) |
| **leg D-lace** (invariance measurement) | `--arm lace --seed 1`, deployed budget — see §5 |
| new flags in effect (all defaulted ON) | `score_registered_controls=True` · `query_set_registry_file="query_sets.json"` · `require_registered_controls=True` · `control_dedup_sensitivity=True` · `persist_per_request_refusal_scores=True` · `lace_whitening_exact_tol=0.10` · `lace_whitening_exactness_fallback=True` |
| ⛔ changed default | `include_coupling_bytes` **True → False**; **new** `expect_byte_target_bytes=524288` |
| everything else | shipped `GenBaselinesConfig` defaults, unchanged: 36-trial kNN grid (k×metric×normaliser×deployment), 36-trial LACE grid (σ×form×shrinkage), `knn_refusal_reduce="min"`, `knn_kmeans_iters=25`, `lace_n_epochs=50`, `lace_batch=128`, `lace_lr=1e-4`, `lace_ld_steps=100`, `refusal_valid_quantile=0.95`, `n_fit=32256`, `n_val=512`, `decoder_source="decoder_lift_v1"`, `assert_decoder_hash=True` |
| decoder / encoder | param hash asserted in-run; decoder ceiling re-measured **0.9223 / 0.9365 / 0.9653** in-corner = the published values |
| ± convention | **population sd (ddof=0)** everywhere below. ⚠ `gen-baselines` §2a used ddof=1 — see reconciliation item 4 |
| artefacts | `.claude/outputs/gen-arms-a7-wiring/` — **`refusal_scores.json` (pinned, 2.7 MB, `"status": "complete"`)**, `PREREG.md`, `tables.txt`, `runs/{historical_688128,deployed_524288,deployed_524288_lace}/seed_*/`, `runs/*.log`, `runs/progress.txt`, `run_legs.sh`, `assemble.py`, `analyse.py` |
| wall clock | leg H 26 + 20 + 28 min · leg D 2.5 + 3.5 + 3.7 min · leg D-lace 17 min · **100 min total for 7 runs** |

---

## 1. What I built (the four items)

| # | item | what shipped |
|---|---|---|
| 1 | **A7 wiring** | Both arms score every C kind against the valid control the **on-disk `query_sets.json` declares for it** — ⛔ no list of controls lives in the experiment. `duplicate_factor → a7_redundant`, `out_of_range → base` (instantiated by the arms' own **B** pool, ⛔ never a second copy). A kind with no declared control is emitted as an **`error`**, ⛔ never silently scored. Missing registry ⇒ `FileNotFoundError` with the fix in the message. |
| 1b | **per-request persistence** | Every variant's per-request score + **ids** + the C kind vector ship in `refusal_scores.json` beside each manifest (popped out of the manifest so it does not multiply in size), and are assembled into the **pinned** `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` with `"status": "complete"` **written last**. ⭐ The A-vs-C-by-kind column in §3 was computed **from that file, not from a re-run** — the acceptance criterion demonstrating itself. |
| 1c | **`A7D`, a declared sensitivity** | The A7 records with the redundant term removed. ⛔ Not a registered control; it exists because switching B → A7 moves **two** things (§4). |
| 2 | **byte default** | `include_coupling_bytes: True → False`; **new** `expect_byte_target_bytes = 524288`, asserted in `resolve_store_bytes` and recorded in the manifest; inheriting the old flag now **stops the run** with an R26.3 message; `--byte-budget historical` reproduces 688,128 B deliberately. |
| 3 | **R27.3 gate** | The whitening-exactness threshold is config-backed and now **gates the configuration**: when the pre-registered selection is non-exact, the **nearest exact** trial (same σ/form, shrinkage swapped) runs as a second leg `*_wexact`, on **its own PRNG stream**, and both are reported. |
| 4 | **C reporting** | Every refusal table is **by kind**, contaminated and controlled **side by side**; the pooled controlled number is an `n_c`-weighted mean explicitly labelled `pooled_COMPANION`. ⛔ Never merged. |

⭐ **The anti-collision design that makes §2 possible.** `CONTROL_SETS` is deliberately **outside**
`REQUEST_SETS` — that tuple drives the decode/probe path *and* LACE's `key` splitting, so appending
to it would have moved every historical number. A7 is never decoded, and its LACE sampling draws
from an independent `PRNGKey(seed·10009 + 20260911 + 4242)` folded per draw; R27.3's fallback
classifier trains from `…+7717`. ⛔ The pre-registered leg's key stream is byte-for-byte untouched.

---

## 2. ⭐ THE REGRESSION — `out_of_range` reproduces BIT-IDENTICALLY, 3/3 seeds

Leg H (the published configuration) vs `.claude/outputs/gen-baselines/runs/shapes3d_recon/seed_*`:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `knn` / `knn_proto` | 0.6628086419753086 | 0.8753858024691358 | 0.8753858024691358 |
| `knn_compsel` | — | 0.7368827160493827 | 0.9170524691358025 |
| `lace/ode` | 0.5135030864197531 | 0.3788580246913580 | 0.2989969135802469 |
| `lace/ld_published` | 0.2881944444444444 | 0.5216049382716049 | 0.5050154320987654 |
| `lace/ld_corrected` | 0.6130401234567902 | 0.4544753086419753 | 0.3406635802469136 |

⭐ **Every digit matches the published manifest.** And the check is wider than `out_of_range`:

> **worst absolute difference over EVERY shared refusal *and* composition number, all 3 seeds,
> all 5–8 variants per seed: `0.000e+00`.**

That covers `auroc_c_vs_b`, `auroc_c_vs_a`, `auroc_c_vs_a_union_b`, both by-kind AUROCs, and every
arm's in-corner balanced accuracy. ⇒ **the wiring moved nothing.** The selections also reproduce
(seed 0 kNN `k=5 cosine zscore int8`; LACE `σ=0.1 both 1e-3` / `σ=0.03 both none` / `σ=0.03 both
ledoit_wolf`), as do the decoder ceilings and the whitening diagnostics (0.2382 / 0.0567 / 0.1438).

⚠ **A convention clash found while doing this** (reconciliation item 4): means are bit-identical but
`gen-baselines` §2a's `±` is **sample** sd. kNN oor = 0.8045 ± **0.1002** (ddof=0) = 0.8045 ±
**0.1227** (ddof=1, as published). LACE `ld_corrected` oor = 0.4694 ± **0.1117** / ± **0.1368**.

---

## 3. ⭐ REFUSAL BY KIND — CONTAMINATED vs A7-CONTROLLED (⛔ never merged; R27.4)

**Leg H, 688,128 B** (the published configuration, so these are directly comparable to
`gen-baselines` §2a/§2b). ⛔ `pooled` is a companion and is **not** in this table; it is
`½(dup + oor)` to **1.1e-16** in all 12 seed×leg cases (P4 held).

| arm | kind | ⛔ CONTAMINATED (vs B) | ⭐ **A7-CONTROLLED** | `A7D` sensitivity | seen-valid A |
|---|---|---|---|---|---|
| `knn` = `knn_proto` | `out_of_range` | **0.8045 ± 0.1002** | ✅ **0.8045 ± 0.1002** *(unchanged by construction)* | 0.9525 ± 0.0210 | 0.9554 ± 0.0181 |
| `knn` = `knn_proto` | `duplicate_factor` | 0.3907 ± 0.1364 | ⭐ **0.3338 ± 0.0692** | 0.6560 ± 0.0689 | 0.6726 ± 0.0623 |
| `knn_compsel` (n=2) | `out_of_range` | 0.8270 ± 0.0901 | ✅ 0.8270 ± 0.0901 | 0.9556 ± 0.0120 | 0.9591 ± 0.0148 |
| `knn_compsel` (n=2) | `duplicate_factor` | 0.4498 ± 0.0652 | 0.3708 ± 0.1138 | 0.6590 ± 0.0135 | 0.6784 ± 0.0140 |
| `lace/ode` | `out_of_range` | 0.3971 ± 0.0885 | ✅ 0.3971 ± 0.0885 | 0.4599 ± 0.0375 | 0.4815 ± 0.0207 |
| `lace/ode` | `duplicate_factor` | 0.9991 ± 0.0010 | ⭐⭐ **0.9936 ± 0.0057** | 0.9981 ± 0.0025 | 0.9967 ± 0.0009 |
| `lace/ld_published` | `out_of_range` | 0.4383 ± 0.1063 | ✅ 0.4383 ± 0.1063 | 0.5773 ± 0.1111 | 0.6037 ± 0.0702 |
| `lace/ld_published` | `duplicate_factor` | 0.9925 ± 0.0103 | 0.8797 ± 0.0851 | 1.0000 ± 0.0000 | 0.9958 ± 0.0029 |
| `lace/ld_corrected` | `out_of_range` | **0.4694 ± 0.1117** | ✅ **0.4694 ± 0.1117** | 0.4655 ± 0.0667 | 0.4768 ± 0.0530 |
| `lace/ld_corrected` | `duplicate_factor` | 0.9697 ± 0.0210 | 0.8192 ± 0.0674 | 0.9547 ± 0.0378 | 0.9601 ± 0.0317 |
| `lace/ode_wexact` (R27.3, n=2) | `out_of_range` | 0.5183 ± 0.0114 | ✅ 0.5183 ± 0.0114 | 0.5150 ± 0.0448 | 0.4995 ± 0.0159 |
| `lace/ode_wexact` (R27.3, n=2) | `duplicate_factor` | 0.9994 ± 0.0006 | **0.9967 ± 0.0033** | 0.9963 ± 0.0033 | 0.9973 ± 0.0003 |
| `lace/ld_published_wexact` (n=2) | `out_of_range` | 0.4614 ± 0.1130 | ✅ 0.4614 ± 0.1130 | 0.6694 ± 0.0513 | 0.6208 ± 0.0426 |
| `lace/ld_published_wexact` (n=2) | `duplicate_factor` | 0.9994 ± 0.0006 | 0.9211 ± 0.0789 | 0.9990 ± 0.0010 | 0.9982 ± 0.0016 |
| `lace/ld_corrected_wexact` (n=2) | `out_of_range` | 0.4794 ± 0.0897 | ✅ 0.4794 ± 0.0897 | 0.5384 ± 0.0114 | 0.5166 ± 0.0000 |
| `lace/ld_corrected_wexact` (n=2) | `duplicate_factor` | 0.9522 ± 0.0459 | 0.9030 ± 0.0287 | 0.9770 ± 0.0218 | 0.9661 ± 0.0275 |

⭐⭐ **THE RESULT THE HUB IS WAITING FOR.** LACE's `duplicate_factor` — *"the single most flattering
competitor number in this report"* (`gen-baselines` §2b) — **survives its own control**:
**0.9991 → 0.9936 ± 0.0057** for the ODE sampler, 3/3 seeds ≥ 0.9861. ⛔ It was **not** the 7-term
giveaway. The mechanism is real and nameable: a `duplicate_factor` request asks one factor for two
**different** values, and LACE's categorical head is one-hot over a closed vocabulary, so one term
**must** be unsatisfied and the composed energy rises; an A7 record repeats a factor at the **same**
value, so both terms are satisfied and the control lands with the valid pool.
⇒ **R27.4's differentiator is now quotable on a controlled metric:** kNN covers `out_of_range`
(0.8045) and fails `duplicate_factor` (0.3338); LACE is the mirror (0.3971–0.5183 / 0.9936).
⛔ **Neither rival covers both kinds — and that sentence no longer rests on a contaminated column.**

⚠ The two weaker LACE samplers **do** lose ground under the control (ld_published 0.9925 → 0.8797,
ld_corrected 0.9697 → 0.8192), so the control is not inert — it has teeth on exactly the samplers
whose energies are noisier. ⭐ Reported per sampler, ⛔ never as one "LACE" number.

---

## 4. ⚠ THE DECOMPOSITION — an A7-controlled number moves TWO things, and for kNN they nearly cancel

A7 is built from **train** rows by ruling (*"held-out rows are never spent on a control"*), so
swapping the control B → A7 changes the **term count** (6 → 7) **and** the **seen-ness** of the
negative pool (unseen-valid → seen-valid). `A7D` — the same records, same rows, redundant term
removed — holds the rows fixed, so the two legs separate exactly. ⭐ **I pre-registered this
confound before measuring it** (`PREREG.md` P3).

| arm | B (unseen, 6 terms) | `A7D` (seen, 6) | **A7** (seen, 7) | **seen/unseen leg** | **term-count leg** |
|---|---|---|---|---|---|
| `knn` = `knn_proto` (n=3) | 0.3907 | 0.6560 | **0.3338** | **+0.2653** | **−0.3221** |
| `knn_compsel` (n=2) | 0.4498 | 0.6590 | 0.3708 | +0.2091 | −0.2882 |
| `lace/ode` (n=3) | 0.9991 | 0.9981 | 0.9936 | −0.0010 | −0.0045 |
| `lace/ld_published` (n=3) | 0.9925 | 1.0000 | 0.8797 | +0.0075 | −0.1203 |
| `lace/ld_corrected` (n=3) | 0.9697 | 0.9547 | 0.8192 | −0.0149 | −0.1355 |

⭐ **What it says about the kNN arm, and it is a mechanism statement, not a scoring artefact.**
R8's declared embedding composes **every** term, so a redundant term adds its deviation **twice** —
which pushes a perfectly satisfiable A7 request off the valid manifold. Measured at the deployed
threshold: the kNN arm **refuses 49.5 % of A7 requests** but only **5.6 % of the same requests with
the repeat removed** (`A7D`), against 5.1 % on A. ⇒ *a distance-to-store rule cannot tell "said
twice" from "impossible"*, which is the same statement as its 44 % false-refusal rate on B in a
second direction. ⛔ This is a property of the **declared, arm-agnostic** R8 rule applied
identically to the control and to C — ⛔ not a choice this spoke made to weaken kNN, and ⛔ not
something to patch silently (an arm that de-duplicates its own control is choosing its own exam).

⚠ **P3 was FALSIFIED and this is the finding.** I predicted the controlled kNN number would **rise**
to ~0.55 and that the **seen/unseen** leg would dominate. Realised: it **falls** to 0.3338, and the
**term-count** leg is the larger (|−0.3221| > |+0.2653|). ⛔ Whoever quotes kNN's A7 number quotes
this decomposition with it.

---

## 5. ⭐ THE BYTE BUDGET — default flipped, asserted, and the trigger measured as a no-op

**Leg D** runs on the shipped defaults and every manifest carries
`byte_target: {store_bytes: 524288, coupling_bytes: 0, total_bytes: 524288,
expected_total_bytes: 524288, assertion: "realised == expect_byte_target_bytes",
superseded_by_R26_ruling_3: false}`. Realised entries: fp32 **64** / fp16 **128** / int8 **248**
(vs 84 / 168 / 328 at 688,128 B), slack **0** on every deployment.

### 5a. ⭐ `knn_proto`'s composition is BYTE-BUDGET-INVARIANT — measured, 3/3 seeds

| seed | 688,128 B | 524,288 B | Δ |
|---|---|---|---|
| 0 | 0.876263 | 0.876263 | **+0.000000** |
| 1 | 0.891960 | 0.891960 | **+0.000000** |
| 2 | 0.896985 | 0.896985 | **+0.000000** |
| **mean** | **0.8884 ± 0.0088** | **0.8884 ± 0.0088** | **0** |

⭐ `gen-baselines` §3 argued this from source (`prototype_direct_read(embeddings)` never consults the
byte-sized store) and recommended measuring it anyway. **Measured.** ⇒ the Head's *"a matched re-run
becomes mandatory before that cell is quoted"* trigger is **discharged**: `knn_proto` **0.8884** is
the number at the correct budget too, and the 31 % byte error explains **none** of its +0.14 margin
over `clu_wide`. ⛔ The honest phrasing stands unchanged and is now evidenced.

### 5b. ⚠ What DOES move — declared, because it moves DOWN for one variant

| quantity | 688,128 B | 524,288 B | reading |
|---|---|---|---|
| `knn` in-corner balacc | 0.5206 ± 0.0487 | **0.4993 ± 0.0220** | ⚠ **down 0.021** — the R26.3-mandated de-resourcing (328 → 248 int8 entries). ⛔ Not a hobbling: it is the correction of a 31 % advantage the arm should never have had, and the arm still loses this cell to `read`/`clu_wide` either way |
| `knn` `out_of_range` AUROC | 0.8045 ± 0.1002 | ⭐ **0.8281 ± 0.1299** | **up 0.024** — the rival gets **better** at the correct budget |
| `knn` `duplicate_factor`, contaminated | 0.3907 ± 0.1364 | 0.4331 ± 0.1665 | up |
| `knn` `duplicate_factor`, A7-controlled | 0.3338 ± 0.0692 | 0.3453 ± 0.0737 | up |
| `knn` false-refusal on **B** at the deployed threshold | 0.4444 | ⭐ **0.3426** | **down** — fewer false refusals |
| `knn_proto` in-corner balacc | 0.8884 ± 0.0088 | 0.8884 ± 0.0088 | invariant |

⭐ **Net: the falsifier did not fire.** Four of the five cells that moved moved in the rival's
favour; the one that moved against it is the byte correction R26.3 ordered, on the **weak** family
member, and it is stated here rather than left silent.

### 5c. LACE is byte-invariant — **source-verified and then measured**
`ctx["byte_target"]` is consumed at exactly three sites, all inside `run_knn_arm`
(`exp_gen_baselines.py:731,732,902`); ⛔ `run_lace_arm` never reads it. ⭐ **And it is measured, not
just argued:** a LACE run at **524,288 B** on seed 1
(`runs/deployed_524288_lace/seed_1`, 1,005 s) reproduces the 688,128 B run with

> **worst |Δ| = `0.000e+00` over 30 numbers** (3 samplers × {in-corner balacc, A/B composition,
> 3 pooled AUROCs, both by-kind AUROCs, both A7-controlled AUROCs}), whitening diagnostic identical.

⇒ every LACE number in §3 and §6 is valid at **both** budgets, and ⛔ LACE never needs re-running for
a byte correction.

---

## 6. ⭐ R27.3 — the whitening gate, both legs run and both reported

The gate fires **exactly where `gen-baselines` §6c said it would**, and the fallback the code picks
is **exactly the configuration §6c decoded by hand** — the selection rule is now mechanical:

| seed | pre-registered selection | \|diag−1\| | exact? | **fallback leg run** | its \|diag−1\| | val_agg (fallback vs selected) |
|---|---|---|---|---|---|---|
| 0 | σ=0.1, `both`, **`1e-3`** | 0.2382 | ⛔ no | σ=0.1, `both`, **`none`** | 0.0570 | 0.2266 vs 0.2454 |
| 1 | σ=0.03, `both`, **`none`** | 0.0567 | ✅ yes | — (gate passes, ⛔ no fallback) | — | — |
| 2 | σ=0.03, `both`, **`ledoit_wolf`** | 0.1438 | ⛔ no | σ=0.03, `both`, **`none`** | 0.0510 | 0.2617 vs 0.2721 |

**Both legs, side by side (⛔ neither dropped; the stronger is quoted per axis, because no single
leg wins everything):**

| axis | seed 0 pre-reg | seed 0 `_wexact` | seed 2 pre-reg | seed 2 `_wexact` | ⭐ quoted |
|---|---|---|---|---|---|
| best in-corner balacc | **0.7418** | 0.6913 | **0.6582** | 0.6563 | **pre-registered** (both seeds) |
| best pooled AUROC(B vs C) ⛔contaminated | **0.7766** | 0.7649 | 0.7415 | **0.7872** | split |
| best A7-controlled `duplicate_factor` | 1.0000 | 1.0000 | 0.9946 | **1.0000** | `_wexact` (s2), tie (s0) |
| best `out_of_range` ✅A7-clean | **0.6130** | 0.5297 | 0.5050 | **0.5745** | split |

⭐ **P6 held, and the ruling earns its keep on exactly one axis.** LACE's **best `out_of_range`**,
taking the stronger leg per seed as R27.3 directs: **0.6130 / 0.5216 / 0.5745 = 0.5697 ± 0.0378**,
against the published best-of-family **0.4694 ± 0.1117**. ⇒ running the licensed whitening
**strengthens LACE on its weakest kind by ≈ +0.10**, which is what never-weaken-a-competitor exists
to produce. ⛔ It does **not** change the qualitative reading: LACE still sits near chance on
`out_of_range` and far below the kNN arm's 0.8045–0.8281 there.

⚠ **The pre-registered leg is stronger on composition on both gated seeds**, so ⛔ nobody may use
R27.3 to swap LACE's composition number: LACE's composition stays as published
(`ode` 0.6797 ± 0.0207, family means bit-identical, §2).
⚠ **The unrepaired residue, stated:** on seeds 0 and 2 the *pre-registered* leg still ran with the
prior-score omission unlicensed (that is why the fallback exists). The second route R27.3 offered —
*"run the general form with the prior score included"* — is **NOT RUN**: it needs a new drift term in
`sample_ode`, which is a sampler change, not a selection change. ⭐ The fallback route was taken
because it is the one the grid already contained; the general form is a clean follow-up if the Hub
wants the third data point.

---

## 7. PREREG scorecard (`.claude/outputs/gen-arms-a7-wiring/PREREG.md`, filed before any run number)

| # | prediction | realised | verdict |
|---|---|---|---|
| **P1** | `out_of_range` bit-identical, all arms, all seeds | worst \|Δ\| over **every** shared refusal + composition number = **0.000e+00** | ⭐ **HELD, exactly** |
| **P2** | LACE A7-controlled `duplicate_factor` **0.90** [0.70, 1.00] (H2a: real conflict detection) vs **H2b** (collapse to ≈0.5) | **ode 0.9936 ± 0.0057**; ld_pub 0.8797; ld_corr 0.8192 | ⭐ **HELD; H2b REJECTED** |
| **P3** | kNN A7-controlled dup **rises** to 0.55 [0.35, 0.80]; seen/unseen the larger leg | **falls to 0.3338 ± 0.0692**; term-count leg (−0.3221) larger than seen/unseen (+0.2653) | ⛔ **FALSIFIED, both parts — and it is the finding (§4)** |
| **P4** | pooled-controlled = ½(dup + oor) exactly; `A7D` embeddings = plain embeddings | max \|Δ\| **1.11e-16** over 12 seed×leg cases; pinned by a unit test | ⭐ **HELD** |
| **P5** | `knn_proto` composition bit-identical across budgets | Δ = **0.000000**, 3/3 seeds | ⭐ **HELD — the Head's trigger is a measured no-op** |
| **P6** | gate fires on seeds 0 and 2 only; fallback weaker on validation but **not uniformly** weaker on the reported metrics | fires 0 and 2; validation −0.019 / −0.010; reported metrics **split by axis**, +0.10 on LACE's best `out_of_range` | ⭐ **HELD** |

⚠ **Disclosure carried from the prereg:** P2 and P3 were written after a `--quick` plumbing smoke
(n_fit 2,048, 1-epoch classifier). The smoke pointed at H2b for LACE (0.53–0.57) and I predicted
against it, giving the reason (a 1-epoch classifier cannot score a conflict). The full run vindicates
the reasoning, ⛔ not the smoke.

---

## 8. How I verified

```
# tests (worktree, main venv)
pytest tests/test_gen_baselines.py                       -> 60 passed   (41 -> 54 test functions, +13)
pytest tests/test_gen_baselines.py tests/test_gen_c_set_a7.py \
       tests/test_gen_data_splits.py tests/test_gen_c_set_size.py -> 116 passed in 20.6 s
pytest (+ test_gen_compose, test_gen_certificate, test_gen_launcher)  -> 155 passed in 33.9 s
ruff check chlu/config.py chlu/core/gen_request.py \
           chlu/experiments/exp_gen_baselines.py tests/test_gen_baselines.py -> All checks passed!
⛔ ruff FORMAT: all four files were ALREADY non-conforming at `main @ 5fee3f4` (verified by
   formatting `git show 5fee3f4:<file>`), so I did not reformat — that would be a multi-hundred-line
   out-of-scope diff. Same precedent as `gen-c-set-a7` §6.
⛔ FULL SUITE NOT STARTED — wave-close policy (R28); base `main @ 5fee3f4`, branch tip `2610100`.

# runs (6 + 1, per-seed, resumable; runner echoes HEAD before the first seed)
.claude/outputs/gen-arms-a7-wiring/run_legs.sh            -> 6/6 rc=0, 83 min
  leg H: --arm all  --byte-budget historical   seeds 0,1,2
  leg D: --arm knn  (shipped defaults)         seeds 0,1,2
  leg D-lace: --arm lace --seed 1              (the invariance measurement)
assemble.py  -> refusal_scores.json  status=complete  not_run=[]  2.7 MB
analyse.py   -> tables.txt  (every table in this report)
```

⭐ **Smoke before the reported runs:** a `--quick` run exercised the whole path (A7 loaded, both
columns emitted, scores persisted, gate fired) in 122 s before an hour of compute was spent.

⛔ **CSF3: unreachable (2FA), local CPU only.** Submit line for whoever has the token, matching
`scripts/csf3/job_gpu_gen.sh`'s stage convention:
```
sbatch scripts/csf3/job_gpu_gen.sh --dataset shapes3d --pretext recon --stage baselines \
       --arm all --seed $SEED            # ⭐ deployed budget is now the DEFAULT (524,288 B)
sbatch scripts/csf3/job_gpu_gen.sh ... --byte-budget historical   # ⛔ only to reproduce the 688,128 B run
```
⚠ The launcher does not yet forward `--byte-budget` (it is not in my file scope — §10).

---

## 9. Git footprint

| | |
|---|---|
| branch | `agent/experiment-engineer/gen-arms-a7-wiring` |
| base | `main @ 5fee3f4` (⛔ unmoved; `git rebase main` → "up to date"; ⛔ never `origin/main`, §7.21) |
| worktree | `../CHLU-gen-a7wire` (fresh; ⛔ the shared checkout was on another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` and was **never touched**) |
| commit | **`2610100`** — *"make the rivals scorable under A7; stop three wrong defaults"* |
| files touched | `chlu/config.py` (+75/−6, `GenBaselinesConfig` block only) · `chlu/core/gen_request.py` (+111) · `chlu/experiments/exp_gen_baselines.py` (+558/−25) · `tests/test_gen_baselines.py` (+308/−?) — **1021 insertions, 31 deletions, 4 files** |
| ⛔ not touched | `gen_store.py`, `gen_bridge.py`, `gen_compose.py`, `gen_certificate.py`, `chlu/data/**`, `scripts/csf3/**`, `chlu/cli/experiment_cmd.py` (§10), the composition cell, the C records |
| ⚠ one commit, not three | the three items interleave inside `refusal_metrics` / `run_lace_arm` / one config block; split commits would not have run. The message names all three. **Flagging it rather than pretending it was atomic.** |
| unresolved conflicts | none |
| ⛔ not pushed, no PR | left for Hub/Head review |

⚠ **One pre-existing test was AMENDED, not deleted:**
`test_resolve_store_bytes_derives_the_coupling_from_the_plans_own_K_and_M` asserted the old default.
It now **opts in explicitly** (`include_coupling_bytes=True` + an explicit expectation) and still
pins the derivation `K(K−1)/2·M²·4`, plus the new default path. ⛔ Nothing was weakened to make a
test pass.

---

## 10. Open questions / follow-ups / risks

1. ⚠ **`chlu/cli/experiment_cmd.py` does not expose `--byte-budget`** — deliberate: it is outside my
   named file scope and is a shared, high-collision file. `python -m chlu.experiments.exp_gen_baselines`
   (how every reported run has been launched) has it. **One-line follow-up for whoever owns that file.**
   ⚠ Same for `scripts/csf3/job_gpu_gen.sh`. Until then a cluster run gets the **correct (deployed)**
   budget by default — the safe direction.
2. ⚠ **`exp_gen_baselines` reads `c_invalid.json`/`query_sets.json` from `<root>/<dataset>/`, ignoring
   `condition`** — pre-existing (I kept it), but the **wide** condition has its own registry at
   `<root>/shapes3d/recomb2element_wide/`. A wide-condition baselines run would silently score the
   Montero C set against the Montero A7. ⛔ Not mine to change mid-wave (it would move the shipped
   Montero paths); **flagging it as a trap before anyone runs the wide cell.**
3. ⭐ **R27.3's second route is NOT RUN** — the general form with the prior score in the drift (§6).
4. ⚠ **The kNN arm refuses ~50 % of A7 requests** (§4). If the Hub wants the rival's *strongest*
   reading, a de-duplicating request embedding would be a different declared R8 rule for **every**
   arm — ⛔ an arm-level fix would be the arm choosing its own exam, so it is a **Head/Advisor**
   decision, not an engineering one. The `A7D` column is there so the decision can be made on data.
5. ⛔ **No kill-rule verdict here** (`gen-cell-run`'s job). ⛔ No CLU number was produced; the CLU rows
   in `gen-baselines` are untouched by this spoke.

---

## Proposed handover updates (for the Hub)

- **§7 Known Issues — NEW:** *"`gen_baselines.include_coupling_bytes` defaulted `True` (688,128 B)
  after R26.3 removed the coupling from the deployed store."* ⇒ **RESOLVED** at
  `agent/experiment-engineer/gen-arms-a7-wiring@2610100`: default `False`, realised budget
  **asserted** against `expect_byte_target_bytes=524288`, historical run reachable only via
  `--byte-budget historical`.
- **§7 Known Issues — NEW, OPEN:** *"`exp_gen_baselines` resolves `c_invalid.json` and
  `query_sets.json` at `<data_root>/<dataset>/`, ignoring `gen_data.condition`. The wide condition
  has its own registry in a sub-directory; a wide-condition baselines run would score the wrong C
  set against the wrong control."* Owner: whoever scopes the wide cell. (§10.2)
- **§7 Known Issues — NEW, informational:** *"the kNN arm refuses 49.5 % of `a7_redundant` (valid)
  requests because R8's declared embedding adds a repeated term's deviation twice; 5.6 % with the
  repeat removed."* Not a defect — a measured property of the declared rule. (§4)
- **R27.4 status change:** the C-by-kind differentiator is **no longer blocked**. The A7-controlled
  numbers to carry forward: kNN `out_of_range` **0.8045 ± 0.1002** / `duplicate_factor`
  **0.3338 ± 0.0692**; LACE(ode) **0.3971 ± 0.0885** / **0.9936 ± 0.0057**. ⛔ Always with §4's
  decomposition for the kNN dup figure.
- **R27.2 note:** kNN's oracle A∪B-vs-C bound was re-measured bit-identically at the historical
  budget; at the **deployed** budget it is a different number and `gen-cell-run` must read the
  budget label off the manifest before quoting kill rule (iv).
- **`gen-baselines` §2a/§2b ± convention:** sample sd (ddof=1) there, population sd elsewhere in the
  same report. One-line fix at the next curator pass. (§2)
- **Wave-close suite:** ⛔ not started by me (R28 policy). Base `main @ 5fee3f4`, branch tip `2610100`,
  targeted 116 passed / 0 failed.
