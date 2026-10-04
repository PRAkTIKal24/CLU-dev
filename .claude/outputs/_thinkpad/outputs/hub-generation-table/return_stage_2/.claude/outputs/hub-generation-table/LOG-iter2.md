# hub/generation-table-2 — RUNNING LOG (ITERATION 2)
**Agent:** `autonomous-hub` · **Branch:** `hub/generation-table-2` · **Worktree:** `C:/Users/PratikJawahar/chlu/CHLU-hub-generation-table-2`
**Base:** `base/iter2` = `945a3e9` (COMMON §1a). ⛔ Not `ae907466`, ⛔ not `8e36bc9`, ⛔ not a peer branch.
**Machine:** FORGIS_14, Windows 11 Pro AMD64, 16 cores / 63 GB / Intel Arc Pro (⛔ no CUDA). **Core budget 4 threads** (4 hubs x 4 = 16).
**START_UTC:** `2026-09-17T10:49:21Z` → **12 h deadline `2026-09-17T22:49:21Z`** (COMMON §1b).
**⚠ This is a SEPARATE file from the predecessor's `LOG.md`.** ⛔ §5.1 "YOU NEVER DELETE": the iteration-1
log, PREREG, ledger and bundle are inherited facts and are left byte-untouched. My artefacts live in
`iter2/`, except the two files the launch file names by absolute path.

---

## 2026-09-17T10:49Z — BOOT
- Read in order: `LAUNCH-COMMON.md`, `LAUNCH-hub-generation-table-2.md`, `.claude/autonomous-hub-charter.md`,
  `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` §7, `.claude/tasks/hub-generation-table.md`.
- **COMMON §0 override, quoted as required:**
  > **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and `hub/generation-table`
  > on v2 despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.** The failure is in
  > the test's `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()` calls
  > `read_bytes().decode()`, which does **not** fold universal newlines, while real `Path.read_text()` opens in
  > text mode and does. On Windows the fixture is written by a real `write_text()`, which expands `\n → \r\n`,
  > so the mock counts two characters per newline and `chars` exceeds `len(text)` by exactly the newline count.
  > **Nothing to do with encoding.** It passes on macOS only because macOS does not expand newlines. The mock
  > is fixed on the Mac and ships in the next cut; the UTF-8 behaviour it guards is unaffected and still
  > enforced by the shipped code's explicit `encoding="utf-8"`.
  - ⛔ Covers **exactly that one test**. Every other red check is information.
- Env confirmed: `CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2` (v2, ⛔ not `chlu-hub-bundle`),
  `PYTHONUTF8=1`, `PYTHONPATH=<my worktree>`, `PY=$CHLU_ROOT/.venv/Scripts/python.exe` (3.11.16, jax 0.9.0,
  numpy 2.4.1, CPU), `OMP/MKL/OPENBLAS_NUM_THREADS=4`. Preamble at `iter2/hubenv.sh`.
- Worktree verified at `945a3e9` ("[head-authorised] base/iter2: reference read + run_seed gate scoped to the
  hazard"), parent `8e36bc9` (the pin tree verbatim).
- **Budget (COMMON §1c): AT MOST ONE scored run this iteration, SEED 0 ONLY.** 0/3 spent on this target on
  any machine. Any number I report will read *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.

## 2026-09-17T10:52Z — ⭐⭐ §5.1 FALSIFIER: **CLEAR.** The stop condition did NOT fire.
Ran first, before anything else, as instructed. Full verdict for the other three hubs:
**`.claude/outputs/hub-generation-table/REENCODE-VERDICT-iter2.md`**; data:
**`.claude/outputs/hub-generation-table/reencode_magnitude.json`**.

- 64 rows x 2048 = 131,072 elements. **1,280 differ (0.9766 %)**, `max_abs_delta = 1.953e-03`,
  `max_ulp_delta = 51.0`.
- ⭐ **argmax-ULP element has `|cached| = 1.7583e-04`** — its local `np.spacing` at float16 is `1.19e-07`
  (the subnormal gap), and its *absolute* move is `6.08e-06`. ULP counting is meaningless there, exactly as
  predicted at `_hub_scratch/reports/I-numerical-divergence.md:212`. ⭐ **Measured, not assumed.**
- ⭐⭐ **Band `|cached| >= 2^-4`: 63,655 elements, 575 differ, `max_ulp_delta` = EXACTLY 1.000**
  (mean over differing = 0.9991), `max_abs_delta` 1.953e-03. Every differing element there moved by exactly
  one float16 rounding step. **ZERO violators of the `> 2 ULP` stop rule.**
- Band `|cached| < 2^-4`: 67,417 elements, 705 differ, max 51.0 ULP, **max abs delta 3.05e-05**.
- **⇒ VERDICT: CLEAR. All four hubs may proceed with no-extension scored work.** Announced immediately.

### ⚠ TWO CAVEATS I am putting on the record with the CLEAR, because the verdict alone would mislead
1. ⛔ **An ENFORCED `_assert_cache_reencodes` ABORTS on this box** — `max_ulp_delta 51.0 > ULP_BOUND 1.0`,
   `would_have_failed = true`, key `ulp`. Scored work is possible here **only** under COMMON §4's ruling that
   the gate does not bind when `n_shortfall == 0`. ⛔ One uncached row and it binds, fails, and I STOP and
   report the shortfall. ⛔ I do not encode.
2. ⚠ **`frac_differing = 0.9766 %` against `FRAC_BOUND = 1.0 %` — a margin of 31 elements of 131,072.**
   ⛔ The fraction bound is NOT an independent second reassurance here. It is very nearly red as well.

**Next:** read the predecessor's ledger/PREREG (inherited facts, COMMON §3), restate the banked tie, and
pick the lever. ⛔ I do not re-derive the predecessor's DEV numbers.

---
---

# ⭐⭐ RELAUNCH — `hub/generation-table-2`, second start (the first was stopped by the Head)
**START_UTC (NEW, this run): `2026-09-17T11:03:06Z` → 12 h deadline `2026-09-17T23:03:06Z`** (COMMON §1b).
**Model:** Opus. Every sub-agent `model: "opus"`, cap 2, inside 4 threads.
⛔ §0.1: nothing above this line is mine and nothing above it is touched. Everything above is
**inherited fact**. Below this line is the relaunch.

## 11:03Z — BOOT (relaunch)
- Read in order: `LAUNCH-COMMON.md`, `LAUNCH-hub-generation-table-2.md`, `.claude/autonomous-hub-charter.md`,
  `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` §7, `.claude/tasks/hub-generation-table.md`.
- **COMMON §0 override, quoted as required (the one red check a human waived, in writing):**
  > **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and `hub/generation-table`
  > on v2 despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.** The failure is in
  > the test's `_WindowsFlavourPath` mock, not the bundle. ... **Nothing to do with encoding.** ... the UTF-8
  > behaviour it guards is unaffected and still enforced by the shipped code's explicit `encoding="utf-8"`.
  - ⛔ It covers **exactly that one test**. Every other red check stays information. ⛔ `SKIP_LOAD`/`SKIP_TEST`
    are never a pass.
- Env: `CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2` (v2; ⛔ never `chlu-hub-bundle`, the v1 pin
  `ae907466`). `PYTHONUTF8=1`. `PY=$CHLU_ROOT/.venv/Scripts/python.exe`. `OMP/MKL/OPENBLAS_NUM_THREADS=4`.
- ✅ **ADOPTED, by explicit dated bullet (2026-09-17, relaunch):** the stopped predecessor's
  `iter2/hubenv.sh` shell preamble. It is a shell wrapper, not a scientific artefact; I have read every
  line. ⛔ Nothing else under `iter2/` is adopted yet.
- **§5.1 falsifier: INHERITED AS FACT, CLEAR, ⛔ NOT re-run** (COMMON §-0 / addendum). Cited from
  `REENCODE-VERDICT-iter2.md` + `reencode_magnitude.json`: band `|cached| ≥ 2^-4` → **max ULP exactly
  1.000, 0 violators**; the 51-ULP figure is entirely sub-`2^-4` post-ReLU dust (`max_abs_delta` there
  3.05e-05). ⚠ Its two caveats travel with it: (1) an **enforced** gate still aborts on this box
  (51 > `ULP_BOUND` 1.0); scored work is legitimate here **only** under COMMON §4's no-extension ruling;
  (2) `frac_differing` 0.9766 % vs `FRAC_BOUND` 1.0 % = **31 elements of headroom** — ⛔ not a second
  reassurance.
- **Budget (COMMON §1c):** 0/3 spent on this target on any machine. **This iteration: AT MOST ONE scored
  run, SEED 0 ONLY**, and only after a PREREG naming object/bar/controls AND a DEV number clearing the bar
  on non-held-out corners with §C controls passing. It will be reported as
  *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.
- **Target (addendum, R1/R2):** WIN or TIE vs `knn_proto` on composition-B aggregate, matched bytes,
  external classifier, shipped cell, seed 0. R1 tie := paired 95 % CI on Δ(arm − rival) ∋ 0, **or**
  |Δ| ≤ m = 0.020.
- **Banked starting position (inherited, ⛔ not re-derived):** DEV 49 corners, arm **0.8135 / 49.9 tuples**
  vs `knn_proto` **0.8307 / 51.8**, Δ = **−0.0172**, CI **[−0.0347, +0.0003]**, 16/49.
  ⭐ Under R1 that is **already a TIE** (CI ∋ 0 and |Δ| ≤ 0.020). **Banked. Now I push for the win.**

## 11:35Z — the inherited iteration-2 code, READ IN FULL, and ✅ ADOPTED (dated bullet, COMMON §-0)
✅ **Adopted:** `iter2/dev/{ctx,dev,fast,arm2,exp30_levers}.py` and the untracked
`chlu/core/gen_dict_fit.py` (488 lines) left by the stopped predecessor. I have read every line of all six.
They are **DEV scaffolding + one new additive core module**; ⛔ no pinned file is touched by any of them.
Copies taken into my own `relaunch/dev/` so the inherited files stay byte-untouched (§5.1).
What `gen_dict_fit.py` adds over iteration 1: NMF (span-objective, non-negative ⇒ decodable/deletable
atoms, R4-legal) · **metric-correct** rank truncation (feature-space, not coefficient-space) · **reduced**
52-column additive design (the full 58 carried 6 structurally unidentifiable columns) · **separate ridges**
for the additive and interaction blocks · per-well atom **allocation** (water-filling).
⛔ **It had never been executed.** Two defects found on first run, both mine to fix:

## 11:40Z — ⛔ DEFECT 1 in the inherited code, FIXED ADDITIVELY: the ragged read cannot call the shipped verb
`RaggedHead.coefficients()` returns a **list** of `(N, M_k)`; the pinned
`gen_compose.clu_wide_vector` takes a **rectangular** `(N, K, M)` and refuses a non-uniform store
(`ValueError: posterior is (32, 72, 26) but the store is K=32, M=26`). ⇒ the whole `waterfill` allocation
lever is **incompatible with the shipped read as pinned**.
⛔ `gen_compose.py` is a pinned harness file (charter §A.4). **I did not edit it.** Instead, new additive
module **`chlu/core/gen_wide_read.py`**:
- uniform allocation ⇒ a **one-line pass-through to the shipped `clu_wide_vector`**;
- ragged ⇒ the same scatter written out, **DEV-only**, and every ragged row carries
  `read_verb = "ragged equivalent (DEV only) — NOT the shipped call"` in its own record;
- ⛔ `stack_coefficients` **refuses** to zero-pad a ragged allocation — a padded atom is still an atom in
  `store_bytes`' eyes and padding would silently misprice the store.
- `tests/test_gen_wide_read.py`: **4 tests, all passing** — uniform is **bit-identical** to the shipped
  call (`np.array_equal`, not `allclose`); ragged is refused by the stacker; `store_bytes` prices a ragged
  allocation honestly; the read takes no feature argument (§B.3, by signature).
⭐ **Every scored configuration will be UNIFORM**, so the scored read is the shipped read verb, verbatim.

## 11:42Z — ⚠ DEFECT 2 in the inherited scaffolding, FIXED: a silent cross-corner cache hazard
`fast.channel_geometry` keys its on-disk cache by `f"s{seed}_c{index}_n{pool_size}"`. The inherited
`$CHLU_OUT/dev/cache` (2.1 GB) was written against **iteration 1's corner list**; `exp30_levers.py` defines
a **different** list. Same index + same pool size ⇒ **the same filename for a different corner**, and the
wrong channel partition would be loaded in silence. ⛔ I am not reusing that cache. My copy writes to
`$CHLU_OUT3/cache` and is re-keyed. Cost of recomputing: **4.1 s per corner** (corr 1.0 s, ward 1.6 s) —
the hazard was never worth the saving. ⛔ Nothing in the inherited cache is deleted.

## 11:45Z — timing probe (`relaunch/dev/smoke.py`), and the first number
Context 0.6 s · gather (32,766 × 2048) 0.5 s · channel geometry 4.1 s · **one full variant end-to-end
(store fit + head fit + shipped read + frozen decoder + external classifier) 32.4 s** (first call, includes
the decoder JIT). ⇒ a 10-variant corner is ~3–5 min. The box can afford a real sweep.
- DEV corner `c0` (`hue_lo=0.2, scale_hi, shape=0, orient_hi`), `K=32 A=832 rank=22`, reduced+metric+
  ridge_pair=100, k-means: **aggregate 0.8310, 48 distinct tuples of 72, 473,344 B** (0.996 × knn_proto).
  ⚠ ONE corner, and c0 is the corner the predecessor's pair set was selected on ⇒ ⛔ **not evidence of
  anything yet.** It is a smoke test. The predecessor's own 10-vs-49-corner reversal is the standing warning.

## 11:08Z — ⭐⭐ THE SELECTION DISCIPLINE, declared BEFORE the first sweep
⛔ Iteration 1's worst moment was a 10-corner "TIE" (Δ = +0.0072, 6/10) that became a 49-corner **LOSS**
(Δ = −0.0172, 16/49) — maximum-over-a-search bias, and the predecessor said so itself. ⛔ I am not repeating
it by widening `n` only after I like a number. The split is fixed **now**, in code (`relaunch/dev/r1.py`):
- **49 corners ADMITTED** of 64 candidates (15 REFUSED for intersecting `eval_a`/`eval_b` or not realising
  72 rows — ⛔ refused, never trimmed). ⭐ The admitted count **reproduces the predecessor's 49 exactly**,
  which is the first cross-check that my corner enumeration is the same object as theirs.
- **SELECT = the first 6 admitted corners.** ⭐ Every mechanism choice — K, A, rank, ridge, pairs,
  dictionary method — is made here and ⛔ nowhere else.
- **CONFIRM = the other 43.** ⛔ Not looked at until a configuration is frozen. The confirm number is the
  one I will believe, and it is the one that decides whether a scored run happens at all.
- Resumable per **UNIT = (corner, variant)**, JSONL ledger `relaunch/runs/r1_ledger.jsonl`. A kill costs
  ~34 s, ⛔ never the run.

## 11:09Z — the byte arithmetic that defines the whole search space, written out before I search it
```
store = (D/K) · A · 4                      A = Σ_k M_k  (total atoms), D = 2048
head  = (n_add·A + n_pair·r + r·A) · 4     n_add = 58 (full) or 52 (reduced)
budget = knn_proto's deployed 475,136 B    ⇒ ⭐ r is NEVER chosen: it is what the budget leaves
```
⇒ **an atom costs `(D/K + n_add)·4` B and a rank unit costs `(n_pair + A)·4` B.** Two consequences the
predecessor never exploited, both of which are pure arithmetic and neither of which is a new idea about
the science:
1. ⭐ **A larger `K` makes atoms CHEAPER** (464 B at K=32 → 336 B at K=64 → 272 B at K=128) because
   `d = D/K` shrinks. ⚠ But §C.2 is measured to stop collapsing at **K ≳ 128** (a partition hands the read
   `K` free scalars), so **K = 64 is the largest rung I will consider as an arm**; K = 128 is run ONLY to
   place the disqualification boundary and ⛔ is disqualified *before* it is scored, never after.
2. ⭐⭐ **`A` was FIXED at 832 for the whole of iteration 1 and never swept.** Both the store *and* the
   additive head scale with `A`, so lowering it frees a lot: at K=32 the rank the budget affords goes
   **22 (A=832) → 55 (A=576) → ~136 (A=384)**, and at r = 136 against `n_pair = 160` the interaction block
   is **essentially unrestricted**. ⇒ ⭐ **the atoms-vs-rank frontier at fixed bytes is the main open lever**,
   and DEV-12's "`knn_proto` is merely ADDITIVE, and a pairwise model beats it by +0.1285 (6/6)" is the
   reason to believe the rank end of it is the valuable end.

## 11:09Z — ⛔ AN IDEA I AM KILLING BEFORE I SPEND ANYTHING ON IT, and why (recorded, charter §B.6)
The deployed feature for any request lies in the row space of `Φ·C·B`, so the atoms only ever need to span
the **request-conditional mean** subspace — at most `n_add + n_pair ≈ 218` directions, not the 2048-dim
activation cloud the dictionary is currently fitted to. Fitting the atoms to the *label-conditional means*
would therefore be strictly better for the read. ⛔ **It is FORBIDDEN.** Charter §B.6: *"labels ground the
words of a request at the interface only (the address book); never in the landscape, the read or the
write."* A dictionary fitted on label-conditional means puts labels in the **write**. ⛔ Not done, not
measured, recorded here so nobody re-derives it as a good idea.
⭐ **The legal version of the same intuition is the NMF lever** (span objective, label-free, non-negative ⇒
atoms stay decodable and deletable), and it is already in the table.

## 11:09Z — first live rows (SELECT corner c0 only — ⛔ ONE corner, evidence of nothing yet)
| row | aggregate | distinct tuples / 72 | bytes |
|---|---|---|---|
| `knn_proto` — the ceiling | 0.7801 | 53 | 475,136 |
| `base` (the predecessor's arm reproduced: K=32, A=832, 58-col, coeff-metric trunc, ridge 10/10, r=17) | **0.8843** | **58** | 473,472 |
⭐ **`knn_proto` reads 0.7801 on c0 — the predecessor's DEV-05 reads 0.7801 on c0.** Independent wiring,
same instrument, same number ⇒ the DEV instrument is the same object theirs was.
⚠⚠ ⛔ **c0 is a SELECT corner and it is the corner iteration 1's pair set was chosen on.** A `+0.104` here
is exactly the shape of the result that fooled the predecessor at n = 10. ⛔ **It means nothing until the
43 CONFIRM corners say it.**

## 11:18Z — SUB-AGENT 1 (read-only, `model: "opus"`): the scored-cell path, audited line by line
Spawned to keep 2,800 lines of pinned harness out of my own context. ⛔ Read-only; it wrote nothing.
Facts I will act on, each with the line it came from:
- ⭐ **`exp_gen_cell --stage cell --seed 0` with the DEFAULT arms costs ≈ 30–80 min (most likely 45–55).**
  Dominated by `ebm_compose`'s 36-trial SGLD tuning loop (`exp_gen_cell.py:706-724`) and LACE's
  12,600-step classifier fit. ⭐ **No GPU anywhere in the module**; no `jax.devices` selection, no CUDA
  string. ⇒ the scored run is affordable inside my window.
- ⭐ Resumable **per arm per seed** (`2017-2020: elif arm in manifest["arms"]: ... continue`), and the
  manifest is re-written the moment **each** arm row completes (1990). ⚠ Two caveats: a resume re-does the
  whole preamble (1.97 GB sha256 + two 32 k-row gathers ≈ 2–4 min), and **a resume silently blanks
  `sampler_noise_floor`** (2169 is unconditional) — so if leg 1 is killed I re-run it into a **fresh seed
  dir**, ⛔ never resume it.
- ⛔⛔ **The pinned `clu` arm is the HARD 30-bit ARGMAX read** — `compose_codes` = `argmax` (`gen_compose.py:99`)
  → `read_compose` → `read_from_codes` (`exp_gen_cell.py:2022`). ⭐ `clu_wide_vector` is **never called or
  imported** by `exp_gen_cell.py`. ⇒ ⛔ **my arm is an ADDITIONAL row, not "the `clu` row improved"**, and I
  will not write it up as one. ⭐ `assert_request_only` (`gen_compose.py:186-190`) whitelists
  `clu_wide_vector(store, posterior)` by signature, so my arm passes the *same* request-only guard the
  pinned row does.
- `composition_B.aggregate` = the **unweighted mean of six per-factor top-1 accuracies**
  (`gen_probe.py:87`), ⛔ not exact-match — which is exactly why the free floor is as high as 0.5278.
  `distinct_factor_tuples` = `np.unique` over the external classifier's 6-factor argmax on the **72** B
  images (`exp_gen_cell.py:346-377`). ⭐ **My DEV harness calls those same three functions.**
- `knn_proto` is **fitted on `row_sets["fit"]`** (32,256 stratified rows of the pool, `1840`) and evaluated
  on B(72)+MS(198). Its 475,136 B is `(Σ_k C_k + 1)·D·4 = 58·2048·4` (`gen_request.py:333-352`).
- `--arm` is **singular** and `action="append"`; ⛔ `--arms a,b` is rejected by argparse. `--quick`
  exists and ⛔ **a quick number is never reported** — I will not use it.
- ⚠⚠ **One recommendation from the sub-agent that I am REJECTING:** it suggested pointing
  `CHLU_CLAUDE_ROOT` at `chlu/chlu-hub-bundle/.claude`. ⛔ **That is the v1 bundle (pin `ae907466`), which
  COMMON §2 forbids me to read or write.** My `env_hub.sh` already resolves it to
  `chlu-bundle-v2/.claude`, which is correct. Recorded because an agent's suggestion is not a fact.

## 11:28Z — ⛔⛔ MY OWN ERROR, CAUGHT AND CORRECTED: I was over the 4-thread budget for ~18 minutes
⛔ **`pkill` DOES NOT EXIST in this Git-Bash** (`which pkill` → not found). My two "kill and restart"
commands were **silent no-ops**, so every restart ADDED workers instead of replacing them. A process census
(`Get-CimInstance Win32_Process`, which is also the only reliable way to see command lines here — `wmic` is
gone on Windows 11) found **5 live `r1.py` instances**: the original 4-thread run plus two copies each of
the two 2-thread workers ⇒ **~10 threads against my allotted 4**. ⛔ That is a COMMON §3 violation and it
was mine, not the box's.
- **Fixed at 11:28Z:** all 10 PIDs killed with `Stop-Process -Force`; census re-run; **0 remaining**.
  ⛔ For the rest of this run I kill only via PowerShell `Stop-Process` and ⛔ never assume a kill worked —
  I re-census afterwards. ⚠ Note each launch shows as **TWO** PIDs (the venv `python.exe` shim plus its
  `uv`-managed CPython child); counting PIDs, not instances, is how you get this wrong in the other
  direction.
- ⚠ I was over budget while three peer hubs were running. Recorded as a fault, not smoothed over.

### ⭐ The one good thing that fell out of my own mistake: a FREE determinism check
The duplicate workers recomputed **19 units twice, in independent processes**. Ledger audit
(53 lines, 3 files): **0 corrupt lines, 30 distinct units, 19 duplicated, and every duplicated unit agrees
on its aggregate to 10 decimal places — 0 disagreements.**
⇒ ⭐ **The DEV pipeline (k-means dictionary → head fit → shipped read → frozen decoder → external
classifier) is bit-reproducible across processes on this box.** I did not plan this measurement; I am
keeping it because it is the cheapest evidence I will get that a DEV number is a property of the
configuration and not of the run.

## 11:47Z — ⭐⭐ EXP-R1 (lever isolation, 6 SELECT corners, 48 units): **EVERY HYPERPARAMETER LEVER IS NOISE**
The predecessor's untested iteration-2 levers, one at a time, off a reproduction of its own arm
(K=32, A=832, k-means, 2 pairs). ⭐ Every row carries its distinct-tuple count, as required.

| variant (6 SELECT corners) | aggregate | sd | tuples / 72 | bytes | Δ vs `knn_proto` | 95 % CI | wins |
|---|---|---|---|---|---|---|---|
| ⭐ `knn_proto` — the ceiling | **0.8673** | 0.0596 | **53.5** | 475,136 | — | — | — |
| `base` (58-col, coeff-metric trunc, ridge 10/10, r=17) | 0.8588 | 0.0851 | 54.5 | 473,472 | −0.0085 | [−0.070, +0.053] | 3/6 |
| `reduced` (52-col design, r=22) | 0.8627 | 0.0872 | **56.8** | 473,344 | −0.0046 | [−0.071, +0.062] | 2/6 |
| `metric` (feature-space rank truncation) | 0.8522 | 0.0939 | 53.3 | 473,472 | −0.0150 | [−0.069, +0.038] | 4/6 |
| `ridgepair` (separate ridge 100 on the pair block) | 0.8607 | 0.0969 | 54.5 | 473,472 | −0.0066 | [−0.064, +0.051] | 4/6 |
| `nmf` (span-objective non-negative atoms) | 0.8603 | 0.0863 | 55.3 | 473,472 | −0.0069 | [−0.047, +0.033] | 3/6 |
| `all3` (reduced+metric+ridgepair) | 0.8588 | 0.0982 | 53.7 | 473,344 | −0.0085 | [−0.065, +0.048] | 4/6 |
| `all3_p3` (+ `shape × orientation`, the third pair) | 0.8619 | 0.0957 | 53.3 | 474,416 | −0.0054 | [−0.062, +0.051] | 3/6 |
⭐ best-constant floor on these corners: **0.5278** — ⭐ it lands on the structural `eval_b` floor exactly.

### ⭐⭐ THE STATISTIC THAT ACTUALLY DECIDES A LEVER — and I had it wrong until I computed it
⛔ Comparing each variant to `knn_proto` is the WRONG test for choosing a mechanism: two variants share
almost all of the corner-to-corner variance. **Paired variant-vs-variant** is the right one, and the
difference in power is enormous:

| paired against `base` | Δ | paired sd | n a 95 % test would need |
|---|---|---|---|
| `knn_proto` | +0.0085 | **0.0764** | 312 |
| `reduced` | +0.0039 | **0.0153** | 61 |
| `metric` | −0.0066 | 0.0331 | 98 |
| `ridgepair` | +0.0019 | 0.0307 | 973 |
| `nmf` | +0.0015 | 0.0537 | 4,657 |
| `all3_p3` | +0.0031 | 0.0376 | 572 |

> ⛔⛔ **VERDICT: not one of the six levers is worth more than ±0.007, and every one of them would need
> between 61 and 4,657 corners to resolve. There are 49. ⇒ THE HYPERPARAMETERS ARE NOT THE LEVER, and no
> amount of tuning them will turn the inherited TIE into a WIN.** I am not going to spend the window
> discovering that six more times.

### ⭐⭐ THE BAR, COMPUTED RATHER THAN HOPED FOR — what a WIN actually costs
Median **paired sd against `knn_proto` = 0.0703**. Under R1 a WIN needs Δ > 0 with the 95 % CI excluding 0:
| n | SE | Δ a WIN requires |
|---|---|---|
| 6 | 0.0287 | **> +0.0562** |
| 24 | 0.0143 | > +0.0281 |
| 43 (my CONFIRM set) | 0.0107 | ⭐ **> +0.0210** |
| 49 (every admitted corner) | 0.0100 | ⭐ **> +0.0197** |
⇒ **Starting from the inherited −0.0172, a WIN needs a swing of ≈ +0.037 in the mean.** ⛔ That is a
structural change, not a tuning. ⭐ I am writing the number down BEFORE I go looking, so that I cannot
later mistake a +0.01 for progress.

### ⛔ The first frontier point is already a large effect — in the WRONG direction
`P2_K32_A320` (only 320 atoms, but the freed bytes buy **rank 160 = `n_pair`, i.e. a COMPLETELY
UNRESTRICTED pairwise interaction block** at 455,680 B): **0.7431 / 35.5 tuples**, Δ vs `base`
**−0.1134**, 0/2 corners. ⇒ ⭐ **Starving the store to buy interaction rank is strongly negative.** The
span the atoms provide is worth far more than the rank the bytes buy. ⚠ n = 2 so far, but at −0.11 against
a paired sd of 0.14 this is a different order of magnitude from the tuning levers.
⇒ **The frontier is telling me to look at MORE atoms, not fewer** — which is the `K = 64` end of the grid
(an atom costs 336 B there instead of 464 B, so the same bytes buy **1.7×** the atoms).

## 11:49Z — ⭐⭐⭐ THE STRUCTURAL IDEA THE FRONTIER POINTED AT: an interaction that costs no atoms
**New additive module `chlu/core/gen_shared_basis_head.py` + `tests/test_gen_shared_basis_head.py`
(8 tests, all passing).** ⛔ No pinned file touched.

**The byte problem, stated exactly.** The head is
`a(request) = Φ_add·C_add + (Φ_pair·E)·U`, costing `(n_add·A + n_pair·r + r·A)·4`.
⛔ **The `r·A` term is the whole difficulty:** one unit of interaction rank costs `A` floats, so at the
atom counts that actually work it costs more than three atoms — and EXP-R2 measures the score as
**monotone increasing in `A`**. The budget therefore forces a choice between SPAN and INTERACTION, span
wins, and what is left is a **lossy factorisation of `knn_proto`'s own additive model class** — an object
that can tie the ceiling but is structurally unable to beat it. ⭐ That, not tuning, is why iteration 1
landed at −0.0172.

**The observation.** `U` is an `(r, A)` basis in atom-coefficient space, stored only so the interaction can
name directions there. ⭐ **But the head already stores an `(n_add, A)` basis in exactly that space:
`C_add` itself** — and its 52 rows are the atom-space directions the six factors move along, which is
precisely where a factor-pair interaction lives. ⇒ **take `U := C_add` and store nothing for it:**
```
a(request) = (Φ_add + Φ_pair·E) · C_add          E is (n_pair, n_add)
head bytes = (n_add·A + n_pair·n_add)·4          ⭐ the interaction term is INDEPENDENT of A
```
At `n_pair=160, n_add=52` the interaction costs **33,280 B whatever `A` is**, against **3,968 B per rank
unit** at `A=832`. `E` is fitted by least squares **in the feature-space metric** (whitened by each well's
Gram), so the error minimised is the error that is decoded.
⚠ **What it gives up, stated plainly:** the interaction can no longer point anywhere in the `A`-dimensional
coefficient space — only inside the 52 directions the additive block already spans. ⛔ That is a real
restriction and it is why this is a measurement, not an obvious improvement.

### What it buys, in atoms, priced from the closed form before anything is run
| configuration | K | **atoms A** | interaction | bytes | × knn budget |
|---|---|---|---|---|---|
| iteration 1's arm (low-rank head) | 32 | 832 | rank 22 | 473,344 | 0.996 |
| low-rank head at K=64 | 64 | 832 | rank 49 | 473,984 | 0.998 |
| ⭐ `SH_P2_K32` | 32 | **928** | **full (52)** | 463,872 | 0.976 |
| ⭐⭐ `SH_P2_K64` | 64 | ⭐ **1,280** | **full (52)** | 463,360 | 0.975 |
| `SH_P3_K64` (3 pairs) | 64 | 1,216 | full (52) | 454,336 | 0.956 |
| `SH_ADD_K64` — ⭐ the additive-only reference | 64 | **1,408** | **none** | 473,088 | 0.996 |
⇒ ⭐ **`SH_P2_K64` carries 1,280 atoms (63 % of D spanned) AND a full interaction, where the low-rank head
could afford 832 atoms (41 %) and rank 49.** ⭐ `SH_ADD_K64` is the paired control that makes the
interaction's contribution a measured difference rather than an assumption.
⛔ **Not yet run. ⛔ Not a result. It is a pre-priced hypothesis and the next thing I measure.**

## 12:16Z — ⭐⭐ EXP-R2 (the atoms-vs-rank frontier, 20 configurations × 6 SELECT corners, 120 units)
Every configuration priced to `knn_proto`'s 475,136 B from the closed form, rank set to whatever the
budget leaves. Paired against `P2_K32_A832` (= iteration 1's atom count under the better 52-column design).

| K | atoms A | rank the budget affords | Δ vs `A=832` (paired, n=6) | 95 % CI | wins |
|---|---|---|---|---|---|
| 32 | 320 | **160 = `n_pair`** (unrestricted interaction) | ⛔ **−0.1030** | **[−0.162, −0.045]** | 0/6 |
| 32 | 448 | 109 | ⛔ **−0.0733** | **[−0.112, −0.035]** | 0/6 |
| 32 | 576 | 70 | ⛔ **−0.0444** | **[−0.079, −0.010]** | 0/6 |
| 32 | 704 | 42 | −0.0247 | [−0.061, +0.012] | 2/6 |
| 32 | **832** | 22 | — (reference) | — | — |
| 64 | 320 | 160 | ⛔ −0.1049 | [−0.175, −0.035] | 1/6 |
| 64 | 576 | 95 | ⛔ −0.0772 | [−0.127, −0.028] | 0/6 |
| 64 | 704 | 69 | ⛔ −0.0529 | [−0.083, −0.023] | 0/6 |
| 64 | 832 | 49 | ⛔ **−0.0355** | **[−0.062, −0.009]** | 1/6 |

### ⭐⭐ Two findings, both with CIs that exclude zero — ⛔ unlike every lever in EXP-R1
1. ⭐⭐ **The arm's score is MONOTONE INCREASING in the number of atoms, at both K, with no sign of a
   peak inside the budget.** Trading atoms for interaction rank is strongly negative: the configuration
   with a **completely unrestricted pairwise block** (`A=320, r=160=n_pair`) is **−0.103** against the one
   with 832 atoms and rank 22. ⇒ ⛔ **The span the atoms provide is worth far more than the rank the
   freed bytes buy.** ⭐ This is the measurement that kills "just give it more interaction", and it kills
   it in the direction opposite to the one DEV-12's +0.1285 pairwise bound would have suggested.
2. ⚠ **`K = 64` is WORSE than `K = 32` at every matched atom count** (−0.0355 [−0.062, −0.009] at A=832,
   and worse still below). ⛔ My arithmetic that a finer partition buys cheaper atoms was right; the
   inference that cheaper atoms are therefore better was **wrong**, and this says so with a CI. A K=64
   configuration must now buy back **+0.035 with its extra atoms before it breaks even**.
3. ⭐ The third pair term (`shape × orientation`) is a small, consistent positive: `P3_K32_A832` vs
   `P2_K32_A832` = **+0.0031, paired sd 0.0104, 4/6** — the tightest paired comparison in the whole run
   (it would need n = 44 to resolve, and I have 43). ⭐ Kept; ⛔ not claimed.

⚠ **`A = 832` is the reference, not a demonstrated optimum — the frontier is still rising at the edge of
the table I ran.** That is exactly what makes the shared-basis head (12:16Z above) the right next
measurement rather than a speculative one: it is the only way to buy atoms past this point without giving
up the interaction entirely. ⛔ Launched, not yet read.

## 12:31Z — ⭐⭐⭐ EXP-R4: THE SHARED-BASIS HEAD WORKS, and it is the first resolved positive in this run
8 configurations × 6 SELECT corners. ⭐ Distinct-tuple count beside every aggregate, as required.

| row (6 SELECT corners) | aggregate | tuples / 72 | bytes | × budget | Δ vs `knn_proto` | 95 % CI | wins |
|---|---|---|---|---|---|---|---|
| `knn_proto` — the ceiling | 0.8673 | 53.5 | 475,136 | 1.000 | — | — | — |
| iteration 1's design `P3_K32_A832` (low-rank r=21) | 0.8619 | 53.3 | 474,416 | 0.998 | −0.0054 | [−0.062, +0.051] | 3/6 |
| `SH_P2_K32` (A=928, shared basis) | 0.8588 | 51.2 | 463,872 | 0.976 | −0.0085 | [−0.062, +0.045] | 4/6 |
| `SH_P3_K32` (A=896, 3 pairs) | 0.8650 | 51.8 | 461,504 | 0.971 | −0.0023 | [−0.058, +0.053] | 5/6 |
| ⭐⭐ **`SH_P2_K64` (A=1,280, shared basis)** | ⭐ **0.8877** | ⭐ **57.5** | **463,360** | **0.975** | ⭐ **+0.0204** | [−0.028, +0.069] | **5/6** |
| `SH_P3_K64` (A=1,216, 3 pairs) | 0.8769 | 55.7 | 454,336 | 0.956 | +0.0096 | [−0.039, +0.059] | 4/6 |
| ⛔ `SH_ADD_K32` (A=1,024, **no interaction**) | 0.8345 | 48.7 | 475,136 | 1.000 | ⛔ **−0.0328** | **[−0.064, −0.002]** | 1/6 |
| ⛔ `SH_ADD_K64` (A=1,408, **no interaction**) | 0.8468 | 50.7 | 473,088 | 0.996 | ⛔ **−0.0204** | **[−0.032, −0.009]** | 0/6 |
| `SH_P2_K64_A832` (the same head, atoms held at 832) | 0.8221 | 47.5 | 312,832 | 0.658 | −0.0451 | [−0.092, +0.001] | 1/6 |

### ⭐⭐ The paired comparisons, which is where the signal is
| paired | Δ | paired sd | 95 % CI | wins | n needed |
|---|---|---|---|---|---|
| ⭐⭐ **`SH_P2_K64` − iteration 1's design** | ⭐ **+0.0258** | 0.0259 | ⭐ **[+0.0051, +0.0466]** | **5/6** | **4** |
| ⭐⭐ **`SH_P2_K64` − `SH_ADD_K64`** (the interaction's own contribution) | ⭐⭐ **+0.0408** | — | — | — | — |
| ⭐⭐ **`SH_P2_K64` − `SH_P2_K64_A832`** (what the extra 448 atoms buy) | ⭐⭐ **+0.0655** | — | — | — | — |
| ⛔ `SH_P2_K64_A832` − iteration 1's design | −0.0397 | 0.0306 | [−0.064, −0.015] | 0/6 | 3 |

> ⭐⭐⭐ **THE MECHANISM, decomposed on the same corners rather than asserted.** At `K = 64` the arm is
> **−0.0397 against iteration 1 when its atoms are held at 832** — exactly the K-penalty EXP-R2 measured.
> The shared-basis head then spends the freed bytes on **448 more atoms (+0.0655)** while keeping a **full
> 52-dimensional interaction (+0.0408)**, and the net is **+0.0258 [+0.0051, +0.0466], 5/6** — ⭐ **the
> first paired CI in this run that excludes zero in the right direction.**
> ⇒ ⭐ **The interaction was never worthless; it was unaffordable.** Making its price independent of `A`
> is what let the arm buy span and interaction at the same time instead of trading one for the other.

⚠⚠ ⛔ **THIS IS A SELECT-SET NUMBER AND IT IS BIASED UPWARD.** `SH_P2_K64` is the best of 20 + 8
configurations looked at on these same 6 corners. **Δ = +0.0204 against a WIN bar of +0.0191 at n = 43 is
not a win; it is a hypothesis.** ⛔ The 43 CONFIRM corners have not been touched and will not be until the
configuration is frozen and pre-registered. Iteration 1's +0.0072-at-n=10 → −0.0172-at-n=49 is the
standing warning and I am holding to the split I declared at 11:08Z.

## 12:32Z — ⛔ THE DISQUALIFICATION SCREEN, run BEFORE spending ~2 h on a confirm run
⭐ A configuration whose `store_removed` control does not collapse is **disqualified before it is scored,
never after** (launch addendum). A linear read over `K` wells gets **`K` free scalars from the partition
alone**, and iteration 1 measured the §C.2 control **stops collapsing at `K ≳ 128`**. ⭐ My candidate is at
`K = 64` — inside the warning's shadow — so this is a disqualifier I have to clear, not a formality.
Running §C.2 (`mean`), §C.1 (`atoms_medoid`), the R4 `pca_affine` control and the D-matched `random` null
at **K = 32, 64 and 128**, all at the shared-basis head. ⭐ K = 128 is run **only to place the boundary**
and ⛔ is not a candidate whatever it scores.

## 12:44Z — ⛔ THE DISQUALIFICATION SCREEN: K=128 IS DISQUALIFIED, K=64 CLEARS §C.2. Measured, at the
### shared-basis head, on the 6 SELECT corners. ⭐ K is reported WITH §C.2, as the addendum requires.
| K | atoms | ⭐ **§C.2 `store_removed` (mean)** | tuples | vs the **0.5278** free floor | verdict |
|---|---|---|---|---|---|
| 32 | 928 | **0.2087** | 11.3 | ⭐ 0.319 **below** | ✅ collapses |
| ⭐ **64** | 1,280 | ⭐ **0.3657** | 22.5 | ⭐ **0.162 below** | ✅ **collapses** |
| ⛔ **128** | 1,536 | ⛔ **0.5147** | 22.2 | ⛔ **0.013 below — it does NOT collapse** | ⛔ **DISQUALIFIED** |
> ⭐⭐ **Iteration 1's warning reproduces exactly on my own instrument: the §C.2 control stops collapsing
> at `K = 128`.** ⛔ **`K = 128` is therefore disqualified BEFORE it is scored, not after** — ⭐ and I am
> recording that **it scored HIGHER than the configuration I am keeping** (`SCR_K128_kmeans` 0.8962,
> Δ vs `knn_proto` **+0.0289, 5/6**, against K=64's +0.0204). ⛔ **That number is not available to me and
> I am not taking it.** A partition hands a linear read `K` free scalars; at K=128 those scalars alone
> reach the free floor, so nothing above it can be attributed to the store.
> ⚠ K=64's margin (0.162 below the floor) is **less than half** of K=32's. §C.2 passes at K=64, and it
> passes by less than it used to. Both halves of that sentence travel with every number I report.

### The other §C controls at K = 64, paired (6 SELECT corners)
| control | its score | Δ (arm − control) | 95 % CI | arm wins |
|---|---|---|---|---|
| ⭐ **§C.2 `mean`** | 0.3657 | ⭐⭐ **+0.5220** | [+0.460, +0.584] | **6/6** |
| the D-matched Gaussian `random` null | 0.7431 | ⭐ **+0.1447** | [+0.075, +0.214] | **6/6** |
| ⚠ **§C.1 `atoms_medoid`** (the real row nearest each learned centre) | 0.8735 | ⚠ **+0.0143** | [−0.009, +0.038] | 4/6 |
| ⛔ **R4 `pca_affine` — a CONTROL, never an arm** | ⭐ **0.9097** | ⛔ **−0.0220** | [−0.049, +0.005] | 1/6 |
⛔⛔ **The R4 control BEATS the arm, and it beats `knn_proto` outright** (`pca_affine` Δ vs `knn_proto`
**+0.0424 [+0.017, +0.068], 6/6 — a WIN by R1**). ⭐ This reproduces iteration 1's DEV-28 retraction on a
different head and a different `K`. **R4 (Head, 2026-09-17) forbids adopting it as a store** — *"a
principal direction cannot be decoded or deleted, and the deletion target needs atoms"* — so it is
reported beside the arm and ⛔ never as the arm. ⭐ But it is the honest headline of the screen and I am
not burying it.

## 13:07Z — ⭐⭐ THE CHARTER-LEGAL ANSWER TO THAT CONTROL: **NMF recovers most of the PCA gap**
If a principal basis wins because of its **objective** (span, `min ‖X − AW‖`, which IS the read verb's own
objective) rather than because it is an unconstrained rotation, then **NMF is the legal way to chase it**:
the same span objective with the atoms confined to the data's own non-negative orthant, so every atom is
still a valid feature vector — **decodable** (it can be pushed through the frozen decoder and looked at)
and **deletable** (removing one removes identifiable content, which is what §B.4's exact deletion needs).
The bridge features are post-ReLU with 51 % of entries exactly zero, so this is the natural parts
decomposition of this representation, not a trick.

| paired vs `SH_P2_K64` (k-means atoms), 6 SELECT corners | Δ | paired sd | 95 % CI | wins |
|---|---|---|---|---|
| ⛔ `pca_affine` (R4 control, not adoptable) | +0.0220 | 0.0333 | [−0.005, +0.049] | 5/6 |
| ⭐ **`NMF_P2_K64`** | **+0.0135** | 0.0215 | [−0.004, +0.031] | **5/6** |
| ⭐⭐ **`NMF_P3_K64`** | ⭐ **+0.0174** | 0.0299 | [−0.007, +0.041] | **5/6** |
⇒ ⭐ **NMF recovers ~60–79 % of the R4 control's advantage with atoms that stay charter-legal.**
⚠ It costs **~190 s a unit against ~35 s**; that is a real budget decision and it is why the arm row is
the expensive one in the confirm run.

## 13:10Z — ⭐⭐⭐ **THE CONFIGURATION IS FROZEN. ⛔ Nothing in it is re-chosen after a CONFIRM number.**
```
ARM  =  K = 64 wells (balanced ward-leaf-order partition, 32 channels each)
        A = 1,216 atoms  (M = 19 per well), NMF dictionary, LABEL-FREE
        head = SHARED-BASIS: C_add (52 x 1216) + E (220 x 52),  U := C_add, stored once
        pairs = floor_hue x scale, object_hue x scale, shape x orientation
        ridge_add = 10, ridge_pair = 100, reduced 52-column additive design
        read = gen_compose.clu_wide_vector (uniform allocation => the SHIPPED verb)
        store 155,648 B + head 298,688 B = 454,336 B = 0.956 x knn_proto's 475,136 B
```
- ⭐ **Why `P3` and not `P2`, stated honestly:** with k-means atoms P2 > P3 by 0.011; with NMF atoms
  P3 > P2 by 0.004. ⛔ **Neither difference is resolved at n = 6 and I am not pretending otherwise.** P3
  is taken on the launch addendum's own prior (*"`hue × scale` and `shape × orientation` carried the
  0.9734 bound"*) and because it costs **fewer** bytes. `ARM_KMEANS` is carried through the confirm run at
  the identical configuration so the dictionary objective is a paired difference, not a story.
- ⭐ **CONFIRM launched 13:08Z: 10 rows × 43 untouched corners = 430 units**, two workers, 2 threads each.
  Rows: `ARM` · `ARM_KMEANS` · `ADD_ONLY` (same atom count, no pair block) · §C.1 `atoms_medoid` /
  `atoms_kpp` / `atoms_random` · §C.2 `mean` · R4 `pca_affine` · `random` null · `ITER1` (iteration 1's
  design) · `knn_proto` recomputed per corner.
- ⭐ **THE BAR, restated before the result exists:** a WIN under R1 needs **Δ > +0.018** at n = 43.
  SELECT says +0.038, and ⛔ **SELECT is the set I searched ~45 configurations on, so it is biased upward
  by an unknown amount.** ⛔ If CONFIRM says tie, I report a tie.

## 13:27Z — ⭐ LEG 2 OF THE SCORED RUN IS BUILT AND **DEBUGGED ON A DRY RUN**, ⛔ not on `eval_b`
`relaunch/scored/run_arm.py`. ⭐ `RUN_ARM_DRY=1` replaces `B` and `MS` with two **Montero-shaped
train-split corners** (asserted to realise 72 rows each and to intersect `eval_a`/`eval_b` in **zero**
rows) so every line executes without a held-out row being read. ⛔ The dry numbers are leaky by
construction — the shipped store was fitted on a pool that CONTAINS those corners — and are **never
reported as results.** ⭐ The point is that leg 2 is debugged **before** the one scored run, not during it.
**It found two real bugs, and both would have corrupted the scored run:**
1. ⛔⛔ **The arm was being built with the LOW-RANK head at `rank = 0`, not the shared-basis head** — so
   the interaction block was silently absent. Caught by the byte ledger: 408,576 B instead of 454,336 B.
   ⭐ Fixed, and a hard assertion added: `run_arm.py` now **aborts** unless the LIVE ledger equals the
   PREREG's `expect_total_bytes`. ⛔ A mispriced arm cannot now be reported.
2. ⚠ The shipped-store rows priced only the store (524,288 B) and not its **addressing head**, so
   `clu_linear` looked 74,240 B more expensive than `clu_argmax`. ⭐ Both now carry
   `store + address_book_bytes = 598,528 B` ⇒ **byte-identical in every column**, which is the whole point
   of that comparison.
- Runs end to end in **285 s**. Cache coverage checked first: **32,903 / 32,903 rows cached,
  `shortfall = 0`** ⇒ ⭐ the run provably cannot extend the cache, which is the only condition under which
  COMMON §4 lets it happen at all. ⛔ The script raises `SystemExit` if that is ever not true.
- ⭐ Structural guards all pass on the live objects: grouping re-derivation **reproduces the store plan
  bit-identically**, the store is **LOADED** (`store_s0.pkl`) and ⛔ never re-trained, `assert_no_bypass`
  passes, and the read is `clu_wide_vector` through `gen_wide_read.wide_vector` (uniform ⇒ the shipped call).
- ⭐ **§C.4 measured on the live arm: distance-to-nearest-atom / atom spread = 3.27.** ⛔ The settled read
  does **not** sit on its nearest atom ⇒ by charter §C.4 it is **not** a lookup table. (⚠ dry-corner value;
  the scored one will be recomputed.)
- ⭐ Free cross-check of iteration 1's headline recommendation, on independent wiring: on the dry corner
  `clu_argmax_shipped` **0.5579 / 12 tuples** vs `clu_linear` **0.7014 / 40 tuples** at a byte-identical
  ledger. ⚠ Leaky, ⛔ not a result — but it is the same sign and the same order as DEV-29's
  **+0.1192, 9.5 → 35.0 tuples**, produced by a different implementation.

## 14:16Z — ⛔⛔ EXP-R5: **I TESTED MY OWN BEST IDEA AND IT IS WRONG.** The low-K end is much worse.
⭐ **Why I looked there at all:** my own measured trend was `K=32 > K=64 > K=128`, i.e. *smaller is
better*, and I had never tested `K < 32`. And the arithmetic had a clean story: the deployed additive map
is `Φ_add·C_add·B`, a `(52, D)` matrix; unfactorised it costs `52·2048·4 = 425,984 B`, while the
store+head factorisation costs `(D/K)·A·4 + 52·A·4`. ⇒ **the factorisation only SAVES bytes while `A` is
small, and at `K = 1` an atom spans all `D` channels, so ~51 atoms already span the whole rank-52 additive
map and the factorisation loss should go to ZERO.** Every configuration below is priced at ≈ 0.98–0.998 ×
budget and carries a full shared-basis interaction.

| K | atoms A | span | aggregate (6 SELECT corners) | tuples / 72 | §C.2 `mean` |
|---|---|---|---|---|---|
| 1 | 51 | 51 global dims | ⛔ **0.6528** | 26 | 0.0556 (1 tuple) |
| 2 | 98 | 98 | ⛔ 0.6690 | 24 | 0.1019 |
| 4 | 188 | 188 | ⛔ 0.6968 | 30 | 0.1042 |
| 8 | 344 | 344 | 0.7338–0.8125 | 42 | 0.0694 |
| 16 | 592 | 592 | 0.8125–0.8333 | 50 | 0.0972 |
| 32 | 896–928 | 896–928 | ≈0.86 | ~51 | 0.2087 |
| ⭐ **64** | **1,216–1,280** | **1,216–1,280** | ⭐ **0.888** | **57.5** | 0.3657 |
> ⛔ **REFUTED, and by a mile: the score is monotone INCREASING in K from 1 to 64.** `K = 1` — the
> configuration my byte algebra said should have *zero* factorisation loss — scores **0.65**, more than
> **0.23 below** the arm. ⛔ I am recording this as a wrong prediction of mine, not quietly dropping it.

⭐⭐ **And the refutation is informative, which is why it was worth 20 minutes.** If a 51-atom global
dictionary could represent the rank-52 additive map, it would score like `knn_proto`. It does not — so the
binding constraint is **not** the dimension of the span, it is **whether a LABEL-FREE dictionary happens to
contain the label-relevant directions.** k-means and NMF fit the *variance* of the activation cloud; the
head needs the ~52 *label-conditional* directions, and it may only use what the dictionary already spans.
⇒ ⭐ **the dictionary must be GENEROUS to have a chance of containing them, and many atoms in many wells
is how you buy that generosity at a fixed byte price.** That is also exactly why §B.6 (label-free writes)
is expensive here, and it is the honest mechanistic statement of what the store costs and what it buys.
⭐ §C.2 collapses to **0.056–0.13** at every low `K` — as expected, the free-scalar effect is a large-`K`
problem only, and the boundary sits between 64 (passes) and 128 (fails).

## 14:18Z — LAUNCH-COMMON.md was updated on disk at 16:12 local; re-read. **§7b is NEW.**
> **§7b (programme-wide, binding): every refusal or admission row — arm or rival, DEV or scored — prints
> `accept_A7` beside `accept_B` at the same threshold. A row without it is not reportable.**
⭐ **Read and acknowledged. It does not bind any row this hub produces:** `hub/generation-table` reports
composition-B aggregates, distinct factor-tuple counts and byte ledgers. ⛔ **It produces no refusal row
and no admission row**, so there is no `accept_B` anywhere in my output for an `accept_A7` to sit beside.
⚠ Recorded explicitly so that "this hub has no A7 column" reads as a checked fact rather than an omission.
§§1c, 4, 5, 7 (R1–R5) are unchanged from the copy I booted on, and the addendum is byte-unchanged.

## 15:42Z — ⭐⭐⭐ THE CONFIRM RESULT — 42 corners that were never looked at during selection
430 units, 2 h 25 m, two workers. ⛔ The configuration was frozen at 13:10Z and **nothing in it was
re-chosen after a CONFIRM number existed.** ⭐ Distinct-tuple count beside every aggregate.

| row (42 CONFIRM corners, seed 0, FORGIS_14) | aggregate | sd | **tuples / 72** | bytes | × budget | Δ vs `knn_proto` | 95 % CI | wins | R1 |
|---|---|---|---|---|---|---|---|---|---|
| ⭐ `knn_proto` — the ceiling | **0.8250** | 0.0620 | **51.4** | 475,136 | 1.000 | — | — | — | — |
| ⛔ R4 `pca_affine` — **a CONTROL, not an arm** | **0.8230** | 0.0610 | **52.7** | 454,336 | 0.956 | −0.0020 | [−0.014, +0.010] | 17/42 | tie |
| ⭐⭐ **THE ARM** (K=64, A=1,216 NMF atoms, shared-basis head) | ⭐ **0.8119** | 0.0640 | ⭐ **51.4** | ⭐ **454,336** | **0.956** | ⭐ **−0.0131** | **[−0.0243, −0.0019]** | **19/42** | ⭐ **TIE** |
| `CTRL_atoms_kpp` — §C.1 adversarial | 0.8086 | 0.0776 | 50.8 | 454,336 | 0.956 | −0.0164 | [−0.032, −0.001] | 18/42 | tie |
| `ITER1` — iteration 1's design | 0.8012 | 0.0709 | 49.3 | 474,416 | 0.998 | −0.0241 | [−0.042, −0.006] | 9/41 | LOSS |
| `CTRL_atoms_random` — §C.1 easy | 0.8010 | 0.0690 | 48.4 | 454,336 | 0.956 | −0.0240 | [−0.036, −0.012] | 13/42 | LOSS |
| `ARM_KMEANS` — the same arm, k-means atoms | 0.7970 | 0.0655 | 50.1 | 454,336 | 0.956 | −0.0281 | [−0.041, −0.015] | 11/42 | LOSS |
| ⭐ `CTRL_atoms_medoid` — **§C.1, the sharp form** | **0.7858** | 0.0754 | 48.7 | 454,336 | 0.956 | −0.0392 | [−0.052, −0.026] | 7/42 | LOSS |
| `ADD_ONLY` — the arm with **no interaction** | 0.7864 | 0.0753 | **43.6** | 408,576 | 0.860 | −0.0386 | [−0.051, −0.026] | 7/42 | LOSS |
| `CTRL_random` — the D-matched Gaussian null | 0.7173 | 0.0953 | 42.1 | 454,336 | 0.956 | −0.1077 | [−0.142, −0.074] | 7/42 | LOSS |
| ⭐ **best constant output** (structural **and** realised) | **0.5278** | — | ⛔ **1** | 0 | — | — | — | — | — |
| ⭐ **§C.2 `store_removed`** (`mean`, **K = 64**) | ⭐ **0.4119** | 0.1171 | ⭐ **9.1** | 454,336 | 0.956 | −0.4131 | [−0.451, −0.375] | 0/42 | collapses |

### ⭐⭐ THE HEADLINE, stated exactly, with the rule applied rather than paraphrased
> **Δ(arm − `knn_proto`) = −0.0131, paired 95 % CI [−0.0243, −0.0019], 19 of 42 corners, at 0.956 × the
> byte budget, with the distinct-tuple count IDENTICAL to the ceiling's (51.4 vs 51.4 of 72).**
> R1: *loss* requires `Δ < −m = −0.020` **with** the CI excluding 0. **|Δ| = 0.0131 ≤ 0.020**, so this is
> ⭐ **a TIE, and it is BANKED.** ⛔ It is **not a win**, and I will not call it one: the CI does exclude
> zero, and it excludes it on the wrong side. ⭐ **The honest sentence is: the arm is inside the tie band,
> slightly below the ceiling, at 95.6 % of its bytes and with the same number of distinct tuples.**

⚠⚠ **THE SELECT SET OVERSTATED THIS BY A FACTOR OF ~2.9, exactly as I warned at 12:31Z.**
SELECT said **+0.0378**; CONFIRM says **−0.0131**. ⭐ The 6-corner SELECT estimate, after ~45 configurations
were compared on it, was optimistic by **+0.051**. ⛔ **Had I widened `n` only after liking a number, I
would be reporting a win right now, and it would be false.** The split declared at 11:08Z is the only
reason this log contains the true number instead.

### ⭐⭐ The four PAIRED comparisons that ARE resolved — and they are the science of this run
| paired, n = 42 | Δ | 95 % CI | wins | meaning |
|---|---|---|---|---|
| ⭐⭐ **arm − §C.1 `atoms_medoid`** | **+0.0261** | ⭐ **[+0.0161, +0.0361]** | **36/42** | ⭐⭐ **§C.1 PASSES.** Same partition, same count, same bytes — the only difference is whether a dictionary entry is a **fitted centre** or the **real stored row nearest it**. ⇒ **the store is NOT a lookup table.** |
| ⭐⭐ **arm − `ADD_ONLY`** (no pair block) | **+0.0255** | ⭐ **[+0.0113, +0.0397]** | **32/42** | ⭐ the **shared-basis interaction** is worth +0.026 and **43.6 → 51.4 distinct tuples**, at a cost of 45,760 B that ⛔ **does not scale with the atom count** |
| ⭐⭐ **arm − `ARM_KMEANS`** (NMF vs k-means atoms) | **+0.0149** | ⭐ **[+0.0080, +0.0219]** | **31/42** | ⭐ **the dictionary's OBJECTIVE matters**: fitting for the SPAN (the read verb's own objective) beats fitting for the VERTEX, with atoms that stay decodable and deletable |
| ⭐⭐ **arm − §C.2 `mean`** | **+0.4000** | [+0.362, +0.438] | **42/42** | ⭐ §C.2 passes at **K = 64**, collapsing to **0.4119 / 9.1 tuples**, 0.116 **below** the 0.5278 free floor |
| ⭐ arm − the `random` null | +0.0946 | [+0.063, +0.127] | 34/42 | the partition and the atoms are not "any 1,216-dim subspace" |
| ⚠ **arm − §C.1 `atoms_kpp`** (adversarial) | ⚠ **+0.0033** | ⚠ **[−0.0126, +0.0060]** | 18/42 | ⛔ **against MAXIMALLY-SPREAD real rows the §C.1 margin VANISHES.** Reported as it is; ⛔ I do **not** claim a clean §C.1 pass against every atom selector |
| ⛔ **R4 `pca_affine` − arm** | ⛔ **+0.0111** | ⛔ **[+0.0060, +0.0163]** | 28/42 | the **control beats the arm**, resolved. R4 forbids adopting it. Reported on the same line, ⛔ never as the arm |
| ⚠ arm − `ITER1` (iteration 1's design) | ⚠ **+0.0108** | ⚠ **[−0.0026, +0.0243]** | 30/41 | ⛔ **NOT RESOLVED.** I cannot claim to have beaten iteration 1's arm either; it would need n ≈ 64 |

⭐ **Falsifier 2 (the free floor): PASSED and not narrowly** — 0.8119 with **51.4 distinct tuples of 72**
against the constant output's **0.5278 with 1**, and against `ADD_ONLY`'s 43.6.
⭐ **Falsifier 3 (the store is load-bearing): PASSED on §C.2 (42/42) and on §C.1-medoid (36/42),
⚠ NOT against the adversarial `atoms_kpp` selector.** Both halves reported.
⭐ **§C.4** (dry-run measurement, recomputed in the scored run): distance-to-nearest-atom / atom spread =
**3.27** ⇒ ⛔ the read does **not** sit on its nearest atom; by charter §C.4 it is not a lookup table.

⇒ ⭐ **My PREREG's own gate is met** (LAUNCH-COMMON §1c(ii)): the bar is WIN-or-TIE under R1, the DEV
number is a **TIE**, and the §C controls pass with the one exception named above. **The scored run is
authorised by the pre-registration, and I am taking it.**

## 15:43Z — ⛔ **MY SECOND OPERATIONAL FAULT: I hit the documented IMPORT TRAP on the scored run itself.**
I launched leg 1 with `cd "$CHLU_ROOT"` and `python -m chlu.experiments.exp_gen_cell`. ⛔ **`python -m`
puts the CWD at `sys.path[0]`, ahead of `PYTHONPATH`**, so it imported the **bundle root's** `chlu`, not my
worktree's — i.e. the **unpatched pin**, whose `run_seed:1780` calls `_assert_cache_reencodes` with the
enforcing default. ⭐ `env_hub.sh` warns about exactly this in a comment headed "⛔⛔ THE IMPORT TRAP", and
I walked into it anyway.
- ⭐ **The gate did its job**: `ENFORCED (enforced=true): no caller opinion — the strict default applies`
  → `AssertionError`, 51 ULP against a bound of 1.
- ⭐⭐ **THIS DOES NOT SPEND A SCORED RUN, and here is why in checkable terms:** the abort happens at
  `exp_gen_cell.py:1780`, which is **before `gather`, before the decoder-ceiling block and before the arm
  loop**. **Zero rows of `eval_b` were read and zero held-out numbers were computed.** (Iteration 1 reached
  the same conclusion about the same abort and logged `0 of 3` spent.) ⛔ The budget still reads
  **0 of 3 spent at the moment leg 1 restarted.**
- ⚠ **A deletion I am declaring rather than burying (COMMON §5.1):** before relaunching I ran
  `rm -rf $CHLU_OUT3/scored/cell` on the aborted attempt's output directory. It was **my own** directory,
  created by me ~60 s earlier, and the aborted run died before `run_seed` writes its first file
  (`_write_json` at line 1923, well after the gate), so it held no artefact. ⛔ It was still a deletion and
  it should have been a rename. Recorded.
- ⭐ **Relaunched correctly from `$CHLU_WT`**, verified by asserting the import site first:
  `chlu.__file__ = C:\Users\PratikJawahar\chlu\CHLU-hub-generation-table-2\chlu\__init__.py`.

## 15:44Z — ⭐⭐ **LEG 1 IS RUNNING, AND THE GATE RULING IS WORKING EXACTLY AS COMMON §4 DESCRIBES**
The harness's own words, from its own log, on the patched tree:
```
⭐ cache re-encode check: 64 rows, 1280/131072 elements differ (0.977%),
   max |delta| = 0.00195312 = 51.000 float16 ULP (bounds: 1 ULP, 1.0% of elements)
⚠ NOT ENFORCED (enforced=false): run_seed coverage: 0 of 33038 needed rows uncached
⛔⛔ THE RE-ENCODE CHECK WOULD HAVE ABORTED THIS RUN (ulp) and was NOT enforced, because:
   run_seed coverage: 0 of 33038 needed rows uncached. ⭐ The run encodes nothing new, so no
   inconsistent feature can be written — but the encoder does NOT reproduce this cache on this
   machine, and any run that extends the cache here WILL abort.
cache coverage for this run: shortfall=0 (gate enforced=False)
cache 480000 rows (scope=full), D=2048
```
⭐ **`shortfall = 0 of 33,038`** — the run provably cannot extend the cache, which is the *only* condition
under which COMMON §4 permits it. ⛔ `ULP_BOUND` and `FRAC_BOUND` are untouched and the check still
**reports 51 ULP loudly** in the manifest. ⭐ Both preconditions PASS: `classifier_gate.json gate=PASS`
(floor 0.95, 3 seeds) and `refusal_scores.json status=complete`. Rows: `store_train=32,768, B=72, MS=198,
fit=32,256, val=512`.

## 16:12Z — ⭐⭐⭐ **THE SCORED RUN IS DONE. seed 0, `eval_b`, both legs, one sitting. scored 1/3.**
Leg 1 `exp_gen_cell --stage cell --seed 0` on the **unedited** harness (814 s) + leg 2 `run_arm.py`
(277 s). ⭐ Distinct-tuple count beside **every** aggregate, byte ledger beside every aggregate, and every
number produced on **FORGIS_14, Windows 11 AMD64, CPU** (R3).

| row (eval_b, 72 images, seed 0, FORGIS_14) | composition-B | **tuples / 72** | deployed bytes | × knn budget |
|---|---|---|---|---|
| the frozen DECODER CEILING (real features through the decoder) | 0.9861 | — | — | — |
| ⚠ `atoms_kpp` — **§C.1 adversarial control** | **0.8634** | 46 | 454,336 | 0.956 |
| ⭐⭐ **`knn_proto` — THE CEILING** | **0.8588** | **57** | **475,136** | 1.000 |
| ⭐⭐ **THE ARM** — K=64, A=1,216 NMF atoms, shared-basis head | ⭐ **0.8495** | ⭐ **51** | ⭐ **454,336** | **0.956** |
| ⚠ `atoms_medoid` — **§C.1 sharp control** | **0.8495** | 46 | 454,336 | 0.956 |
| ⛔ `pca_affine` — **R4 control, never an arm** | 0.8472 | 53 | 454,336 | 0.956 |
| the same arm with **k-means** atoms | 0.8310 | 47 | 454,336 | 0.956 |
| `atoms_random` — §C.1 easy control | 0.8194 | 46 | 454,336 | 0.956 |
| ⭐⭐ **`clu_linear` — THE SHIPPED STORE, only the READ VERB changed** | ⭐ **0.7917** | ⭐ **27** | **598,528** | — |
| the D-matched Gaussian `random` null | 0.6944 | 44 | 454,336 | 0.956 |
| ⛔ **`clu` — the shipped 30-bit ARGMAX read** | **0.5903** | ⛔ **9** | **598,528** | — |
| ⚠ **best constant output** (structural **and** realised) | **0.5278** | ⛔ **1** | 0 | — |
| **§C.2 `store_removed`** on the arm (`mean`, **K = 64**) | **0.3657** | **11** | 454,336 | 0.956 |
| `lace` (CELL instrument) | 0.3032 | 72 | — | — |
| `lace/ld_corrected` | 0.2662 | 70 | — | — |
| `lace/ld_published` | 0.2593 | 56 | — | — |
| `ebm_compose` | 0.2315 | 72 | ⛔ 14,226,660 | ⛔ 29.9 |
| `store_removed` (shipped argmax path) | 0.1111 | 1 | 524,288 | — |
| §C.2 on `clu_linear` | 0.1111 | 4 | 598,528 | — |

### ⭐⭐ TWO CROSS-CHECKS THAT VALIDATE LEG 2'S WIRING COMPLETELY
- ⭐ **`knn_proto`: leg 1 `0.8588 / 57 tuples` — leg 2's independent recomputation `0.8588 / 57`,
  identical to machine precision.**
- ⭐ **`clu`: leg 1 `0.5903 / 9` — leg 2's independent reimplementation of the pinned argmax read
  `0.5903 / 9`.**
⇒ ⭐ **Leg 2 is measuring the same object leg 1 is, through the same instrument.** ⛔ Without this the
arm's row would be a number from a different pipeline, and that is exactly how a wrong comparison ships.

### ⭐⭐⭐ THE HEADLINE — R1 applied, not paraphrased
> **Δ(arm − `knn_proto`) = 0.8495 − 0.8588 = −0.0093, at 0.956 × the byte budget (454,336 vs 475,136 B),
> with 51 distinct factor tuples of 72 against the ceiling's 57.**
> **|Δ| = 0.0093 ≤ m = 0.020 ⇒ ⭐ TIE under R1. BANKED.** ⛔ **Not a win, and I am not calling it one.**
> ⚠⚠ **And a tie on ONE cell of 72 images is a weak instrument: one factor-accuracy step is
> `1/72/6 = 0.0023`, so Δ = −0.0093 is FOUR IMAGES.** ⛔ A single scored cell cannot carry a CI. **The
> powered estimate is the 42-corner DEV CONFIRM: Δ = −0.0131, 95 % CI [−0.0243, −0.0019].** ⭐ Both
> instruments agree, in sign and in magnitude, and both land inside the tie band.

### ⭐⭐ WHERE THE −0.0093 ACTUALLY LIVES — per factor, which makes it concrete
| factor | ARM | `knn_proto` | Δ |
|---|---|---|---|
| floor_hue / wall_hue / object_hue | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0 |
| ⭐ **scale** | ⭐ **0.1806** | 0.1667 | ⭐ **+0.0139** (the arm is BETTER) |
| **shape** | 0.9306 | 0.9861 | **−0.0555** |
| orientation | 0.9861 | 1.0000 | −0.0139 |
> ⭐⭐ **The arm's entire deficit is `shape` — 0.9306 vs 0.9861, i.e. FOUR of 72 images — and the arm
> BEATS the ceiling on `scale`, the one factor both of them are near chance on (0.18 / 0.17 against a
> 1/3 realised floor).** ⭐ `scale` is also the factor the charter records the codebook as losing 90.5 % of.
> ⛔ Reporting "−0.0093" without this table would make a four-image difference sound like a mechanism.

### ⭐⭐ THE PARITY MEASUREMENT (the addendum's labelled ledger row, ⛔ not a spend) — **PASSES BOTH BOUNDS**
| | Mac-published (3 seeds) | ⭐ this box, seed 0 | bound | verdict |
|---|---|---|---|---|
| `clu` composition-B | **0.5949 ± 0.0082** | ⭐ **0.5903** | within the published sd | ⭐ **PASS** (\|Δ\| = 0.0046) |
| `clu` distinct tuples | **9.00 ± 0.82** | ⭐ **9** | ±1 | ⭐ **PASS** (exactly 9) |
⭐⭐ **And the harness's own `rival_regression_check` reports `clu: bit_identical = true, delta = 0.0`** —
the Windows probe-companion number reproduces the Mac-published `0.6609848484848484` **to every digit**.
⭐ `knn_proto` probe delta **+0.0025**, not regressed (tolerance 0.05); `knn_proto` composition-B **0.8588**
against its published **0.8650 ± 0.0122** — inside the sd.
⇒ ⭐⭐ **Windows-vs-Mac parity on this table is not a worry. It is measured, and on the one row designed to
test it, it is exact.**
⚠ ⛔ **One rival DID regress and the harness flagged it, not me:** `lace` measured 0.6136 vs published
0.7077, **Δ −0.0941 against a 0.05 tolerance — `regressed: true`.** ⛔ Reported, ⛔ not absorbed, ⛔ not
mine to fix: `GenCellConfig.lace_selection_source` still points at the broken-split LACE selection
(charter declared property #2, COMMON §5). ⭐ It does not touch the arm-vs-`knn_proto` comparison.

## 16:14Z — ⭐ THE PRE-REGISTERED PREDICTIONS, SCORED ONE BY ONE. ⛔ Nothing adjusted afterwards.
| # | prediction (committed 15:45Z, before either leg ran) | realised | verdict |
|---|---|---|---|
| 1 | arm `0.79 ± 0.07`, **42–62 tuples** | **0.8495, 51 tuples** | ⭐ **HIT** |
| 2 | Δ vs `knn_proto` negative and inside the tie band; point est. −0.013, 90 % `[−0.07, +0.03]`; **"I predict a TIE, not a WIN"** | **−0.0093** | ⭐⭐ **HIT**, and the direction was called in writing |
| 3 | beats the shipped `clu` by > +0.15; beats `lace`, `ebm_compose`, `store_removed` outright | **+0.2592** vs `clu`; +0.546 vs `lace`; +0.618 vs `ebm`; +0.738 vs `store_removed` | ⭐ **HIT** |
| 4 | §C.2 below 0.50 with ≤ 15 tuples | **0.3657, 11 tuples** | ⭐ **HIT** |
| 5 | beats §C.1 `atoms_medoid` by **+0.01 … +0.05**; ties `atoms_kpp` | ⛔ **medoid Δ = 0.0000 (exactly tied)**; `atoms_kpp` **−0.0139** | ⛔⛔ **MISS — and it is the important one** |
| 6 | R4 `pca_affine` at or slightly **above** the arm | arm **+0.0023 above** it | ⚠ **MISS** (in the arm's favour, which is why I am flagging it) |
| 7 | `clu_linear` beats `clu_argmax` by **+0.08 … +0.20**, tuples ≥ 25 | ⭐⭐ **+0.2014, tuples 9 → 27** | ⭐ **HIT**, at the very top of the interval |
| 8 | **PARITY**: `clu` within the published 3-seed sd, tuples within ±1 | **0.5903 vs 0.5949 ± 0.0082; 9 vs 9.00 ± 0.82** | ⭐⭐ **HIT** |
| 9 | `knn_proto` reproduces within `rival_regression_tolerance` 0.05 | probe delta **+0.0025** | ⭐ **HIT** |
| 10 | §C.4 ratio > 1.5 | **3.382** | ⭐ **HIT** |
**8 hits, 2 misses.** ⭐ A prediction that survives is evidence; one that fails is a finding. Both here:

### ⛔⛔ THE MISS THAT MATTERS: **§C.1 FAILS ON THE SCORED CELL.** Stated plainly, not softened.
> On `eval_b`, **the arm scores 0.8495 and a dictionary of REAL STORED ROWS nearest the learned centres
> scores 0.8495 — the same number to four decimals — and the maximally-spread real rows score 0.8634,
> i.e. BETTER than the arm.**
> ⛔ **On this cell, charter §C.1 does NOT pass, and by §C.1's own words the store is reported as
> decorative on the scored cell.** ⛔ I predicted +0.01…+0.05 and I was wrong.

⚠ **What the DEV evidence says beside it, because both belong in the record and neither replaces the
other:** on **42 held-out DEV corners** the same comparison is **+0.0261, 95 % CI [+0.0161, +0.0361],
36 of 42 corners** — resolved, and in the arm's favour. And at 51 distinct tuples the arm renders **5 more
than any atom control** (46 for all three).
⇒ ⭐ **The honest reading: the §C.1 margin is real but SMALL (≈ +0.026, ~2 images on a 72-image cell), and
a single cell of 72 images cannot resolve it — it landed at exactly 0.** ⛔ It is **not** evidence that
§C.1 passes on `eval_b`; it is evidence that **one cell is the wrong instrument for a +0.026 effect**, and
that is a fact about the generation table itself that the programme should carry forward.
⛔ I am **not** quoting the DEV margin as though it discharged the scored §C.1. It does not.

### ⭐⭐ THE FINDING THAT IS BIGGER THAN THE CEILING COMPARISON — on `eval_b`, at a byte-identical ledger
> **On the SHIPPED `K = 5` store, at 598,528 B in every column (store 524,288 + addressing head 74,240),
> changing the read from the 30-bit argmax to a request-addressed LINEAR combination of the same modes is
> worth `0.5903 → 0.7917 = +0.2014`, and takes the distinct factor tuples from `9` to `27` of 72.**
> ⭐ Its §C.2 control collapses to **0.1111 / 4 tuples**. ⛔ No new store, no new bytes, no new decoder —
> `gen_compose.clu_wide_vector` already exists and already takes no feature argument.
⭐ Iteration 1 measured **+0.1192 (6/6)** for this on DEV corners and recommended it. **On the held-out
cell it is +0.2014** — ⭐ **the recommendation is confirmed on `eval_b`, and it is bigger there than on DEV.**
⚠ n = 1 cell; the DEV estimate (6 corners, 6/6) is the powered companion.

## 16:22Z — ⭐⭐⭐ EXP-R6: **IS `knn_proto` BEATABLE AT MATCHED BYTES AT ALL?** Now measured — and it
### prices exactly what being a STORE costs.
The scored budget is spent, so this is DEV work for the programme, on the **same 43 CONFIRM corners** the
arm was measured on, so every comparison below is **paired**. Two rows, ⛔ **neither of which is an arm**:

> **The observation.** The deployed additive map of *any* factored read is `Φ_add·C_add·B`, a `(52, D)`
> matrix. Store it **directly** — 52 dense D-dimensional prototypes — plus the same shared-basis pair
> remap, and it costs `(52·2048 + 220·52)·4 = 471,744 B ≤ 475,136 B`. ⭐ **It fits under `knn_proto`'s own
> budget with NO factorisation loss whatever.**
> ⛔ **It is a BOUND, not an arm:** no store, no partition, no atoms, nothing to delete atom-wise — it is
> precisely the *"glorified lookup table"* the programme's goal statement rules out as a deliverable. Its
> value is that it answers a question this table has never had an answer to.

| row (43 CONFIRM corners, paired) | aggregate | tuples / 72 | bytes | Δ vs `knn_proto` | 95 % CI | wins |
|---|---|---|---|---|---|---|
| ⭐⭐ **`DENSE_PAIR`** — 52 prototypes + shared-basis interaction, **NO STORE** | ⭐ **0.8374** | ⭐ **54.1** | **471,744** | ⭐ **+0.0118** | **[−0.0001, +0.0237]** | **29/43** |
| `DENSE_ADD` — 52 prototypes, no interaction, **NO STORE** | 0.8289 | 51.0 | **425,984** | **+0.0033** | [+0.0004, +0.0061] | 29/43 |
| `knn_proto` — the ceiling | 0.8256 | 51.6 | 475,136 | — | — | — |
| ⭐ **THE ARM** | 0.8104 | 51.0 | 454,336 | −0.0152 | [−0.0268, −0.0035] | 19/43 |

| paired, n = 43 | Δ | 95 % CI | wins | what it prices |
|---|---|---|---|---|
| ⭐⭐⭐ **`DENSE_PAIR` − THE ARM** | ⭐⭐ **+0.0270** | ⭐ **[+0.0188, +0.0352]** | **39/43** | ⭐⭐ **THE COST OF BEING A STORE** at these bytes |
| `DENSE_ADD` − THE ARM | +0.0185 | [+0.0067, +0.0303] | 26/43 | the same, without the interaction |
| `DENSE_PAIR` − `DENSE_ADD` | **+0.0085** | — | — | what the shared-basis interaction adds to a *dense* model, for 45,760 B |
| `DENSE_ADD` − `knn_proto` | +0.0033 | [+0.0004, +0.0061] | 29/43 | ⭐ a **sanity check that passes**: my reduced-design ridge reproduces `knn_proto`'s own model class, at **11 % FEWER bytes** |

### ⭐⭐⭐ THE THREE SENTENCES THIS RUN EXISTS TO PRODUCE
1. ⭐ **`knn_proto` is beatable at matched bytes — but only just, and only by something that is not a
   store.** `DENSE_PAIR` reaches **+0.0118 [−0.0001, +0.0237], 29/43, with 54.1 tuples against 51.6** at
   **471,744 B**. The CI touches zero: at n = 43 a WIN needs **+0.0118** and it measures **+0.0118**. ⇒
   **a TIE, one corner short of a resolved win.** ⛔ I am not rounding that up.
2. ⭐⭐ **The factorisation into wells and atoms costs `+0.0270 [+0.019, +0.035]`, 39 of 43 corners — and
   that is LARGER than the arm's entire deficit to the ceiling (0.0152).** ⇒ ⭐ **if being a store were
   free, the arm would beat `knn_proto`.** The store is not losing because the request head is weak, the
   dictionary is badly fitted, or the read verb is wrong. It is losing because it is a store.
3. ⭐⭐⭐ **And at this byte scale the store is not even buying compression.** Storing `C_add` (52 × A) and
   `B` (A × D/K) costs `A·(52 + D/K)·4`; storing their product costs `52·D·4`. They break even at
   **`A* = 52·D/(52 + D/K)`** — **1,268 atoms at K = 64.** ⭐ **The arm runs at A = 1,216, i.e. 96 % of
   break-even: the product quantiser saves 4 % of bytes and costs 0.027 of accuracy.**
   ⇒ ⛔ **At 475 kB on this table, a product-quantised store cannot be justified as compression.** It has
   to be justified by what a dense prototype table cannot do — **exact atom-wise deletion, request-only
   addressing, and the memory modes** — which is exactly what the programme's own goal statement says the
   core is. ⭐ **That is a design finding, and it is worth more to the programme than the 0.015 I did not
   close.**
⚠ `A*` rises with `K` (592 at K=16, 918 at K=32, 1,268 at K=64, 1,566 at K=128) — so compression only
starts to pay at large `K`, and ⛔ **§C.2 disqualifies K=128**, which is measured at 15:42Z. The two
constraints close on each other, and that closure is the honest reason this table ties rather than wins.

## 16:33Z — ⭐ EXP-R7: the last two levers the addendum names — **the PARTITION and the ALLOCATION**.
### Both were built by the stopped predecessor, ⛔ neither had ever been measured. Both are NULL.
k-means atoms throughout (35 s a unit against NMF's 190 s): the question here is the partition, and the
NMF-vs-k-means effect is orthogonal and already measured at +0.0149 [+0.0080, +0.0219].

| paired vs the arm's partition (`equal` + `uniform`), 6 SELECT corners | Δ | paired sd | 95 % CI | wins | n needed |
|---|---|---|---|---|---|
| `equal` + **waterfill** allocation (spectrum water-filling per well) | **−0.0004** | 0.0271 | [−0.022, +0.021] | 2/6 | **18,887** |
| **`ward`** — the SHIPPED `gen_store.correlation_grouping`, UNEQUAL wells | **−0.0147** | 0.0448 | [−0.051, +0.021] | 1/6 | 36 |
| `ward` + waterfill | +0.0008 | 0.0408 | [−0.032, +0.033] | 3/6 | 10,742 |

- ⭐⭐ **The per-well atom ALLOCATION is exactly neutral: −0.0004, and it would take ~19,000 corners to
  resolve.** The predecessor's module calls water-filling *"free at fixed bytes, and nobody had spent
  it"* — ⭐ **it is free, and it buys nothing.** ⛔ Closed. ⚠ It is also the one lever that forces a
  **ragged** store, which the pinned rectangular `clu_wide_vector` cannot read — so it would have cost
  the shipped read verb for zero gain. Its rows carry a per-corner byte ledger (466,900–473,064 B,
  varying with the well widths) because the ledger is measured from the live objects, not a formula.
- ⚠ **The SHIPPED ward grouping is WORSE than equal blocks (−0.0147, 1/6), not better.** Not resolved at
  n = 6 (it needs ~36); **launched on the 42 CONFIRM corners to close it properly.**
- ⇒ ⭐ Together with iteration 1's null on *learned* well placement, the picture is consistent:
  **nothing about where the wells are, or how many atoms each gets, is the lever.** ⛔ Three independent
  ways of moving the partition, three nulls.

## 17:28Z — ⛔⛔ **EXP-R7 ON 43 CORNERS REVERSES MY OWN 6-CORNER CALL.** The allocation lever is REAL.
At 16:33Z I called the per-well atom allocation *"exactly neutral: −0.0004, and it would take ~19,000
corners to resolve"*, on 6 SELECT corners. ⛔ **On the 43 CONFIRM corners it is not neutral.**

| paired vs `equal` + `uniform` (the arm's partition), **43 CONFIRM corners** | Δ | paired sd | 95 % CI | wins |
|---|---|---|---|---|
| ⭐ `equal` + **waterfill** | ⭐ **+0.0105** | 0.0196 | ⭐ **[+0.0046, +0.0164]** | **29/43** |
| **`ward`** (the SHIPPED grouping) + `uniform` | **−0.0016** | 0.0260 | [−0.0094, +0.0062] | 20/43 |
| ⭐⭐ **`ward` + waterfill** | ⭐⭐ **+0.0172** | 0.0289 | ⭐ **[+0.0085, +0.0258]** | **32/43** |

- ⭐⭐ **Water-filling the atoms across wells is worth +0.011 to +0.017, resolved, and it is FREE in bytes.**
  The predecessor's module called it *"free at fixed bytes, and nobody had spent it"*. ⭐ It is free, and
  ⛔ **I was wrong to close it on 6 corners.**
- ⭐ **The SHIPPED ward grouping is exactly NEUTRAL on its own** (−0.0016, n needed ≈ 1,000) — so the
  equal-block partition I chose is neither better nor worse than the shipped one. ⭐ That closes the
  partition question properly: **three independent ways of moving the partition, three nulls** (learned
  placement in iteration 1, equal-vs-ward here, and `K` itself, which peaks broadly at 32–64).
  ⚠ But ward **combined with** water-filling is the best of the four — the unequal wells give the
  allocator more to work with.
- ⚠⚠ **THE SAME METHODOLOGICAL LESSON, IN THE OPPOSITE DIRECTION.** At 12:31Z a 6-corner set overstated
  an effect by +0.051; at 16:33Z the same 6 corners hid a +0.011 one. ⇒ ⭐ **six corners is not an
  instrument for anything smaller than ~0.035, in either direction**, and the only reason both errors are
  in this log as errors is that the CONFIRM set was declared before the first sweep and always re-run.
- ⛔ **This cannot and does not change the scored result.** The budget is spent, the arm was frozen at
  13:10Z and stays frozen, and the scored row stands as measured. ⭐ It is a **recommendation for the next
  iteration**, and it is worth stating what it would be worth *if* the effects were additive:
  −0.0131 + 0.0172 ≈ **+0.004 against `knn_proto`** — ⛔ **still inside the tie band** (a win at n = 43
  needs +0.0134), and ⛔ additivity is an assumption, which is why I am measuring the combination directly
  rather than quoting that sum.
- ⚠ ⛔ **An adoption cost that must travel with this recommendation:** a water-filled allocation is
  **RAGGED**, and the pinned `gen_compose.clu_wide_vector` takes a rectangular `(N, K, M)` posterior and
  refuses one. ⇒ **adopting it requires the harness owner to generalise that one function** (my
  `chlu/core/gen_wide_read.py` is the worked pattern, and it is a pass-through to the shipped call
  whenever the allocation is uniform). ⛔ Until then a water-filled arm cannot use the shipped read verb.

## 17:29Z — EXP-R8 launched: the best DEV configuration, **measured rather than assumed additive**
`NMF atoms + ward grouping + water-filled allocation`, K = 64, A = 1,216, 3 pairs, on all 43 CONFIRM
corners. ⛔ **Not an arm of this iteration** — the scored budget is spent and the frozen arm is frozen.
It exists so the next iteration is told exactly what to score, with a number rather than a sum.

## 18:19Z — ⭐ PER-RENDER TUPLES PERSISTED (the addendum's deliverable for `hub/composition-2`)
`relaunch/scored/arm/per_render_tuples.npz` — **104 arrays, 58,547 B**. For each of **13 rows** and both
evaluation sets (`B`, `MS`): the external classifier's **per-render 6-factor prediction**, the true
`labels`, the dataset `rows`, and the `requests` that produced them.
Rows: `arm` · `clu_argmax_shipped` · `clu_linear` · `control_{kmeans, atoms_medoid, atoms_kpp,
atoms_random, pca_affine, random, mean}` · `knn_proto_recheck` · the two §C.2 rows.
⭐ **`hub/composition-2` can read the distinct-valid-novel-tuple statistic off this file instead of
spending a scored run of its own** — e.g. the arm's B block gives 51 distinct predicted tuples and
**13 of 72 exact matches** directly.
⚠ ⛔ **What is NOT in it, declared rather than left as a gap:** per-render tuples for `lace`,
`lace/ld_*` and `ebm_compose`. The pinned harness persists decoded arrays only for `dynread`
(`_record(..., persist_to=…)` is passed for that arm alone), and re-running those arms to capture them
would be a **second scored run**. ⭐ Their aggregates and distinct-tuple COUNTS are in the cell manifest
(`scored/cell/seed_0/cell_manifest.json`); only the per-render arrays are absent.

## 19:13Z — ⭐⭐ EXP-R8: the best DEV configuration, **MEASURED**, and additivity over-predicted it
NMF atoms + the SHIPPED ward grouping + a water-filled allocation, K = 64, A = 1,216, 3 pairs, all 43
CONFIRM corners. ⛔ **Not an arm of this iteration** — the scored budget is spent and the frozen arm stays
frozen. This exists so the next iteration is told what to score with a **number**, not a sum.

| row (43 CONFIRM corners) | aggregate | tuples / 72 | bytes | Δ vs `knn_proto` | 95 % CI | wins | R1 |
|---|---|---|---|---|---|---|---|
| ⛔ `DENSE_PAIR` — **storeless bound, NOT an arm** | 0.8374 | 54.1 | 471,744 | +0.0118 | [−0.0001, +0.0237] | 29/43 | tie |
| ⭐ `knn_proto` — the ceiling | 0.8256 | 51.6 | 475,136 | — | — | — | — |
| ⭐⭐ **`BEST_nmf_ward_wf`** — NMF + ward + water-fill | ⭐ **0.8219** | ⭐ **52.0** | 469,600 | ⭐ **−0.0037** | [−0.0167, +0.0093] | **23/43** | ⭐ **tie** |
| `BEST_nmf_equal_wf` — NMF + equal + water-fill | 0.8192 | 51.6 | 454,336 | −0.0064 | [−0.0188, +0.0061] | 20/43 | tie |
| ⭐ **THE FROZEN, SCORED ARM** | 0.8104 | 51.0 | 454,336 | −0.0152 | [−0.0268, −0.0035] | 19/43 | tie |
| `ARM_KMEANS` | 0.7962 | 49.8 | 454,336 | −0.0294 | [−0.0424, −0.0164] | 11/43 | LOSS |

| paired vs the frozen arm, n = 43 | Δ | 95 % CI | wins |
|---|---|---|---|
| ⭐ **`BEST_nmf_ward_wf`** | ⭐ **+0.0115** | ⭐ **[+0.0063, +0.0167]** | **31/43** |
| `BEST_nmf_equal_wf` | +0.0088 | [+0.0044, +0.0133] | 29/43 |

### ⭐⭐ AND THE REASON I MEASURED IT INSTEAD OF ADDING IT UP
At 17:28Z I wrote that *if* the effects were additive the allocation lever would be worth **+0.0172** on
top of the arm. ⭐ **Measured directly it is +0.0115.** ⇒ ⛔ **additivity over-predicted by +0.0057 — a
third of the effect.** That is exactly the size of error that turns a tie into a claimed win on paper, and
it is why the sum was flagged as an assumption when I wrote it rather than quoted as a result.

> ⭐ **The best charter-legal store configuration this run found sits at `Δ = −0.0037 [−0.0167, +0.0093],
> 23 of 43 corners, with 52.0 distinct tuples against the ceiling's 51.6, at 469,600 B (0.988 × budget).**
> Statistically **level with `knn_proto`** — ⛔ **still not a WIN** (which needs Δ > +0.0122 at n = 43),
> and ⛔ it is a **DEV** number: it was never scored, and it cannot be, because the budget is spent.
> ⚠ It also requires a **ragged** allocation, so it cannot use the pinned rectangular `clu_wide_vector`
> until that one function is generalised.

⭐ **The complete ordering this run establishes, all on the same 43 held-out corners:**
`DENSE_PAIR` (storeless, +0.0118) > `knn_proto` (0) > `BEST` (−0.0037) > **the scored arm** (−0.0152) >
`ARM_KMEANS` (−0.0294) — with **the factorisation loss, `DENSE_PAIR` − `BEST`, at +0.0155**, down from
+0.0270 against the frozen arm. ⇒ ⭐ **the levers found in this run close about 43 % of the cost of being
a store, and the rest of it is still there.**

---

# ⭐ 19:17Z — CLOSING: verification, packaging, and the elapsed-time statement

## Final verification sweep
| check | result |
|---|---|
| any python of mine still running (COMMON §3) | ⭐ **none** — process census by `Get-CimInstance`, ⛔ not by `pkill`, which does not exist here |
| worktree clean | ✅ `git status --short` empty |
| branch ref visible **from the MAIN repo**, not only the worktree (AGENT_PROTOCOL §3.2) | ✅ `2e65d96` |
| the ten pinned harness files byte-identical **to the pin tree `8e36bc9`** | ✅ `exp_gen_store` · `gen_compose` · `gen_store` · `gen_dream` · `gen_request` · `gen_probe` · `gen_rowsplit` · `gen_landscape` · `gen_bridge` · `exp_gen_compose` |
| `exp_gen_cell.py` byte-identical to **`base/iter2`** (the gate patch is the Head's, not mine) | ✅ zero-line diff |
| the five Head-added reference-read files byte-identical to `base/iter2` | ✅ `gen_factored_read` · `gen_partition` · `gen_address` · `gen_sensitivity` · `test_gen_factored_read` |
| my whole diff against `base/iter2` | ⭐ **5 files, 895 insertions, 0 deletions, 0 modified** |
| tests | ✅ **12 passed** (`test_gen_wide_read` 4, `test_gen_shared_basis_head` 8) |
| DEV ledger integrity | ✅ **950 lines, 0 corrupt, 927 distinct units, 19 duplicated, 0 disagreeing** |
| wrote only to `$CHLU_HUB_OUTPUTS/hub-generation-table/**` + my declared report `hub-generation-table-2.md` | ✅ — plus the two git worktrees below, which are mine and are sanctioned by AGENT_PROTOCOL §3.2 |

⚠ **Two deletions I performed, both declared rather than buried (COMMON §5.1):**
1. `rm -rf $CHLU_OUT3/scored/cell` at 15:43Z — **my own** directory, created ~60 s earlier by the run that
   aborted at the import trap, which dies **before** `run_seed` writes its first file. It held no
   artefact. ⛔ It should still have been a rename.
2. `rm -f` on the first (mispriced) return tarball — ⛔ **not actually deleted in the end:** it is
   **renamed** to `hub-generation-table-2.FULL-3.7GB.return.tar.gz` and left in place, because it is a
   perfectly valid **full** pack of the whole outputs tree, just far too large to carry. ⭐ If the Head
   wants everything including both iterations' caches, that file is it.
⛔ **Nothing inherited was deleted. Iteration 1's `LOG.md`, `PREREG`, `RESULTS-LEDGER.md`, bundle, return
pack and 2.1 GB `dev/cache` are byte-untouched, as is the stopped predecessor's `iter2/**`.**

## Git worktrees I created (both left in place)
`CHLU-hub-generation-table-2` @ `2e65d96` (mine, pre-created for me) and
`CHLU-hub-generation-table-2-replica` @ `2e65d96` **(detached)** — ⭐ the clean checkout the charter §A.5
replication ran from, left in place as the evidence for it.

## The git bundle — ⭐ WHOLE BRANCH, ⛔ never a range
`git bundle create … hub/generation-table-2` → `hub-generation-table-2.branch.bundle`, **3,205,237 B**,
`sha256 daf6ed2796de4f8db0f1f43108ee7211c2cd8ce0b033fc03d9c369d309ad54ea`.
`git bundle verify` → *"The bundle records a complete history"*, ref
`2e65d9688fc9ce0438046310790e9b71f1fc1084 refs/heads/hub/generation-table-2`.

## The return pack — and the judgement call in it, made in the open
⛔ `--return-pack hub-generation-table` on the real outputs directory packs **651 files / 3,752,286,316 B
(3.5 GiB)**, of which **> 99 % is regenerable cache** — iteration 1's `dev/cache/**` (2.1 GB) and my
`relaunch/cache/**` (1.5 GB) of per-corner channel geometry, regenerable in **~4 s per corner**.
⛔ I did not delete any of it (COMMON §5.1). ⭐ Instead `relaunch/scripts/stage_return.py` **copies** the
results into a staging tree inside my own write prefix and packs that, leaving every original byte where
it is:
| | |
|---|---|
| returned | ⭐ **199 files, 14,981,366 B → `hub-generation-table-2.return.tar.gz`, 10,672,123 B** |
| `sha256` | `1ab1e8931b7fc322cb988466f2a4dcf097d92234a44fa23b39200a32ac6dcf1d` |
| left in place, **not** returned | **252 files, 3,722,284,725 B** — each listed with size, and the 40 largest with their sha256, in `RETURN-EXCLUSIONS-2.json`, with how to regenerate it |
| the full pack, if wanted | `hub-generation-table-2.FULL-3.7GB.return.tar.gz`, `sha256 e02db812d80a2a577697245766f8e6fe967e6fc7c8609381270e48e5fc04d55d` |
| ingest | `scripts/bundle_hub_workspace.sh --return-ingest <tar>` (⛔ diverts, never overwrites) |
⚠ **A trap worth passing on:** `--return-pack --src <staging>` does **not** re-root the outputs directory
if `CHLU_HUB_OUTPUTS` is already exported — `chlu_derive_roots` honours the existing environment variable,
and my first pack silently took the real 3.5 GiB tree. ⭐ Run it under `env -u CHLU_HUB_OUTPUTS -u
CHLU_CLAUDE_ROOT -u CHLU_ROOT`.

## ⏱ ELAPSED TRUE WALL-CLOCK
**START `2026-09-17T11:03:06Z` → `2026-09-17T19:17:18Z` = 8 h 14 m of the 12 h window**, measured against
`START_UTC.txt` and `date -u`, ⛔ not against my sense of elapsed time (which COMMON §-0 warns ran ~6×
fast for one hub, and which felt to me like far less than eight hours).

# ⭐⭐ FINAL VERDICT — `hub/generation-table-2` (RELAUNCH)

> ## ⭐ **TIE. BANKED. ⛔ NOT A WIN. Reported truthfully.**
> **Scored runs spent: 1 of 3** — *"scored 1/3 — seed 0 only, 3-seed pending full caches."*
>
> `eval_b`, seed 0, FORGIS_14: **the arm 0.8495 with 51 distinct factor tuples of 72 at 454,336 B**
> against **`knn_proto` 0.8588 with 57 tuples at 475,136 B**. **Δ = −0.0093; |Δ| ≤ m = 0.020 ⇒ TIE
> under R1.** On 42 held-out DEV corners: **Δ = −0.0131, 95 % CI [−0.0243, −0.0019], 19/42** — the same
> answer from a properly powered instrument.
>
> ⭐ **Falsifier 2 PASSES and not narrowly** — 0.8495 with **51 tuples** against a **0.5278** constant
> floor that six published rows still sit below.
> ⭐ **Falsifier 3 PASSES on §C.2 everywhere** (0.3657 / 11 tuples scored, +0.4000 42/42 on DEV) and on
> **§C.4** (nearest-atom / spread = 3.382 ⇒ not a lookup table) — ⛔ **and FAILS §C.1 on the scored cell**,
> where a dictionary of real stored rows ties the arm exactly and maximally-spread rows beat it.
> ⭐ **Falsifier 1 is PART-answered:** every rival beaten outright, `knn_proto` tied, ⛔ the matched-bytes
> lookup clause failed.

### ⭐ The five things this run adds that are not the ceiling comparison
1. ⭐⭐ **The cost of being a store is priced: +0.0270 [+0.0188, +0.0352], 39/43** — larger than the
   arm's whole deficit to the ceiling. **If being a store were free, the arm would win.**
2. ⭐⭐ **And the store is not buying compression at this scale:** break-even is `A* = 52·D/(52 + D/K)`
   = **1,268 atoms at K = 64**; the arm runs at **1,216 = 96 % of it**. ⇒ **4 % of bytes saved for
   0.027 of accuracy.** The store must be justified by deletion, addressing and the memory modes —
   ⛔ **not on this table.**
3. ⭐⭐ **The read verb, confirmed on `eval_b` and BIGGER there than on DEV: +0.2014, 9 → 27 tuples, at a
   byte-identical 598,528 B ledger.** One function call, no new bytes. **My clearest recommendation.**
4. ⭐ **A new byte-efficient head** (`gen_shared_basis_head.py`): an interaction whose price does not
   scale with the atom count, **+0.0255 [+0.0113, +0.0397], 32/42**. And **the dictionary's objective
   matters**: NMF over k-means, **+0.0149 [+0.0080, +0.0219], 31/42**.
5. ⭐ **Parity is measured and exact where it matters:** `clu` **0.5903 / 9 tuples** against the published
   **0.5949 ± 0.0082 / 9.00 ± 0.82**, and the harness's own check reads **`bit_identical: true`**.

### ⛔ And five things I got wrong, all of them in this log as mine
1. ⛔ I exceeded my 4-thread budget for ~18 minutes because `pkill` does not exist here (11:28Z).
2. ⛔ I hit the documented import trap on the first scored launch (15:43Z) — the gate caught it before any
   held-out row was read.
3. ⛔ **My best structural idea, K = 1, was refuted by 0.23** (14:16Z).
4. ⛔ I **closed the allocation lever as null on 6 corners**; 43 corners resolved it at **+0.0105** (17:28Z).
5. ⛔ I **pre-registered a §C.1 margin of +0.01…+0.05 and measured exactly 0.0000** on the scored cell.

⭐ **The reason the headline in this log is −0.0131 and not +0.0378 is a split declared at 11:08Z, before
the first sweep: SELECT = 6 corners, CONFIRM = 42. The selection bias was +0.051. Without it this report
would claim a win, and the win would be false.** That, more than the arm, is what I would want the next
hub to take.

⛔ **`hub/generation-table-2`: the target is TIED, not WON. Reported as such.**

---

# ⭐⭐⭐ 19:22Z — EXP-R9 (POST-VERDICT, added with the remaining window): **THE READ VERB, POWERED**
My clearest recommendation was carried by `n = 1` on `eval_b` (+0.2014) and `n = 6` from iteration 1
(+0.1192). ⭐ **43 held-out DEV corners now resolve it.** Same **SHIPPED** `K = 5` store (loaded from
`store_s0.pkl`, ⛔ never re-trained), same address book, and the linear head asserted **in code** to be
**74,240 B = `GenCellConfig.addressing_head_bytes_reference`, to the byte** — so both rows carry
**598,528 B in every column**. The ONLY difference is the verb.

| row (43 CONFIRM corners, byte-identical at 598,528 B) | aggregate | **tuples / 72** |
|---|---|---|
| ⭐⭐ **`linear`** — `clu_wide_vector(store, request-addressed real coefficients)` | ⭐ **0.7363** | ⭐ **37.1** |
| ⛔ **`argmax`** — the SHIPPED `read_from_codes(store, argmax)` | **0.5595** | ⛔ **12.8** |
| §C.2 on the argmax read | 0.1525 | 1.0 |

| paired, n = 43 | Δ | 95 % CI | wins | n needed |
|---|---|---|---|---|
| ⭐⭐⭐ **`linear` − `argmax`** | ⭐⭐ **+0.1768** | ⭐⭐ **[+0.1512, +0.2024]** | ⭐⭐ **43 / 43** | **1** |
| `argmax` − §C.2 | +0.4071 | [+0.381, +0.434] | 43/43 | 1 |

### ⭐⭐ THREE INDEPENDENT INSTRUMENTS, THREE AGREEING ANSWERS
| instrument | n | store | Δ (linear − argmax) | tuples |
|---|---|---|---|---|
| iteration 1, DEV (DEV-29) | 6 | a **freshly fitted** K=5 store | **+0.1192**, 6/6 | 9.5 → 35.0 |
| ⭐ **this hub, `eval_b`, SCORED** | **1 cell** | the **shipped** store | ⭐ **+0.2014** | **9 → 27** |
| ⭐ **this hub, DEV, powered** | **43** | the **shipped** store | ⭐ **+0.1768 [+0.1512, +0.2024]** | **12.8 → 37.1** |
⭐⭐ **The scored `eval_b` value sits INSIDE the 43-corner CI, and every one of the 43 corners goes the
same way.** ⇒ ⛔ this is no longer a recommendation resting on one cell.

⚠ **THE LIMITATION, DECLARED IN THE FILE ITSELF AND NOT ONLY HERE:** the store and address book are the
**shipped seed-0 objects, fitted on a pool that CONTAINS these DEV corners**, so **both** rows are
inflated by the same leak. ⛔ **The absolute levels (0.7363 / 0.5595) are NOT comparable with `eval_b`'s.**
⭐ The leak is identical on both rows, on the same rows, through the same store — **the PAIRED DIFFERENCE
is the quantity, and it is the only thing I quote.**
⚠ ⛔ **And it does not make the shipped store non-decorative.** `linear` at 0.7363 still loses to
`knn_proto` by −0.0893 [−0.1025, −0.0760], 0/43, at **1.26 × the bytes**. ⭐ The read-verb change is worth
taking **on its own merits — for free** — and ⛔ it is a separate claim from anything about the store.

## ⚠ 19:25Z — CORRECTION to the 19:17Z packaging entry (new bullet; ⛔ the entry above is not rewritten)
The return pack quoted at 19:17Z (`1ab1e893…`, 199 files) was built **before** the EXP-R9 read-verb work
and the closing entries existed, and a later rebuild briefly swept its own predecessor tarballs back in
(3.77 GB — packing a return tarball inside a return tarball). ⭐ **Both superseded packs are RENAMED, not
deleted**, and `stage_return.py` now excludes `*.return*.tar.gz` by name.
**The authoritative pack is `hub-generation-table-2.return.tar.gz`: 204 files, 11,735,113 B →
7,202,080 B.** Verified to contain the final `LOG-iter2.md` (sha256 identical to the copy on disk,
1,103 lines), the report, the ledger, the PREREG, the git bundle, the table figure, the per-render
tuples, the full `cell_manifest.json` and both inherited iteration-1 documents.
⚠ **A pack cannot contain its own sha256.** The value is printed by the packer, is in this log below, and
is written beside the tarball in **`RETURN-PACK-SHA256.txt`** — ⛔ which is therefore *not* inside the
pack, by construction. Files left on the box and not returned: **256 files, 11,242,070,653 B**, every one
listed in `RETURN-EXCLUSIONS-2.json` with its size, the 40 largest with their sha256, and how to
regenerate it. ⛔ **Nothing was deleted.**

| artefact | bytes | sha256 |
|---|---|---|
| `hub-generation-table-2.branch.bundle` (whole branch) | 3,205,237 | `daf6ed2796de4f8db0f1f43108ee7211c2cd8ce0b033fc03d9c369d309ad54ea` |
| ⭐ `hub-generation-table-2.return.tar.gz` (**the one to carry**) | 7,202,080 | see `RETURN-PACK-SHA256.txt` beside it |
| `hub-generation-table-2.FULL-3.7GB.return.tar.gz` (everything, incl. both caches) | 3,741,820,467 | `e02db812d80a2a577697245766f8e6fe967e6fc7c8609381270e48e5fc04d55d` |

---

# ⭐⭐ 19:31Z — EXP-R10 (POST-VERDICT): **SHOWING the property the report says justifies the store**
The report's central design claim is that at this byte scale a product-quantised store cannot be justified
as **compression** (4 % of bytes saved, 0.027 of accuracy paid) and must instead be justified by what a
dense prototype table cannot do: atoms that are **decodable** and **exactly deletable**. R4 refuses the
PCA dictionary for the mirror-image reason — *"a principal direction cannot be decoded or deleted"*.
⭐ **A claim of that weight should be shown, not asserted.** `relaunch/scored/atoms_are_atoms.py`, on the
arm's own fitted NMF dictionary, seed 0.

### 1. ⭐ Is a dictionary entry a VALID feature vector? (the bridge features are post-ReLU)
| dictionary | entries **outside** the data's orthant | max negative magnitude |
|---|---|---|
| ⭐ **NMF (the arm's)** | ⭐ **0.00 %** | 0.0000 |
| k-means | 0.00 % | 0.0000 |
| §C.1 `atoms_medoid` (real stored rows) | 0.00 % | 0.0000 |
| ⛔ **R4 `pca_affine`** | ⛔ **47.12 %** | ⛔ **0.9097** |
⇒ ⭐ **R4's stated reason is now a measured number**: nearly half of a principal dictionary's entries are
negative, and a negative entry cannot be a post-ReLU feature. ⛔ It is not a feature vector, so there is
nothing to push through the decoder and nothing to look at.

### 2. ⭐ DECODABLE — every atom pushed through the FROZEN decoder **on its own**
`relaunch/scored/atoms/atoms_decoded.png`: 12 single atoms, feature vector **zero on 2,016 of 2,048
channels** and the atom itself on its own 32. All 12 decode to **finite, structured, scene-like images**
(pixels span the full 0.000–1.000 range, floor/wall/object layout visible in every one).
⚠ **Stated precisely, because the picture invites over-reading:** a single atom is a **very
out-of-distribution** input to the decoder (98.4 % of its channels are zero). ⭐ The claim demonstrated is
the modest and sufficient one — **an atom is an object you can put through the decoder and look at** —
⛔ **not** that a single atom is a sample, and ⛔ not that its picture is a "concept".

### 3. ⭐⭐ EXACTLY DELETABLE — zero ONE atom (well 3, atom 1) and re-read, 512 requests
| quantity | measured |
|---|---|
| ⭐⭐ **max absolute change OUTSIDE that atom's own well** (2,016 channels) | ⭐⭐ **0.0 — exactly zero** |
| ⭐ max deviation from the predicted change `−a_m · atom_m` | ⭐ **3.24e-07** (float32 rounding) |
| store bytes before / after | 155,648 / 155,648 |
⇒ ⭐⭐ **Deletion is EXACT in the strong sense: the change is confined to the deleted atom's own 32
channels and equals `−a_m · atom_m` to float32 there. Nothing else in the store moves at all.**
⚠ ⛔ **One sub-check came back VACUOUS and I am recording that rather than dropping it:** I also wanted to
show that a request with a **zero** coefficient on the deleted atom returns **bit-identical**. ⛔ There are
**0 such requests of 512** — the shared-basis head produces a dense coefficient vector, so every request
touches every atom a little. ⭐ The exactness is therefore demonstrated by the *locality* result (zero
change outside the well) and the *algebraic* one (deviation 3e-07), ⛔ not by a bit-identity subset,
which does not exist for this head. ⚠ **That is itself a design fact worth carrying: a dense request head
makes deletion exact but not sparse** — every read changes a little when any atom is removed.

## 19:32Z — ⭐ THE FINAL ARTEFACTS (this supersedes every earlier hash in this log)
| artefact | bytes | sha256 |
|---|---|---|
| `hub-generation-table-2.branch.bundle` — ⭐ **whole branch**, `git bundle verify` = complete history, ref `2e65d968…` | 3,205,237 | `daf6ed2796de4f8db0f1f43108ee7211c2cd8ce0b033fc03d9c369d309ad54ea` |
| ⭐⭐ `hub-generation-table-2.return.tar.gz` — **THE ONE TO CARRY**, 209 files / 11,922,633 B staged | **7,367,677** | ⭐ `01c39ad2ad07d80679fd917abaadd8141329a514a9cc1e20fcd832134ca88a7e` |
| `hub-generation-table-2.FULL-3.7GB.return.tar.gz` — everything incl. both iterations' caches | 3,741,820,467 | `e02db812d80a2a577697245766f8e6fe967e6fc7c8609381270e48e5fc04d55d` |
| `RETURN-PACK-SHA256.txt` | 304 | (carries the line above; ⛔ deliberately NOT inside the pack) |
| `RETURN-EXCLUSIONS-2.json` | — | 259 files / 11,286,673,906 B left on the box, each with size, the 40 largest with sha256, and how to regenerate |
⛔ Four superseded packs are **RENAMED, never deleted**: `*.SUPERSEDED-1916Z/1924Z/1925Z/1926Z.tar.gz`.
⭐ Ingest: `scripts/bundle_hub_workspace.sh --return-ingest <tar>` (diverts, never overwrites).

## 19:32Z — ⛔ STOPPING HERE, at 8 h 29 m of 12 h, and why that is the right call
I have **3 h 31 m** of the window left and I am stopping. ⛔ Not because a number looks presentable:
1. ⛔ **The scored budget is spent.** COMMON §1c allows **one** scored run this iteration and it is taken.
   **Nothing I do now can change the scored table.**
2. ⛔ **The DEV search has converged, and it converged onto a structural reason rather than a missing
   trick.** The best charter-legal configuration is **−0.0037** from the ceiling, the storeless bound is
   **+0.0118** above it, and the gap between them — **the cost of being a store, +0.0155 to +0.0270** — is
   the whole remaining distance. ⛔ No hyperparameter closes it; six of them were each measured at ≤ 0.007.
3. ⛔ **Seeds 1 and 2 are impossible here** (`scope="used"` caches; covering the cell needs ~32,200 new
   rows, and that is a re-encode, which is forbidden).
4. ⭐ **Everything the addendum asks for exists**: the table with the 0.5278 floor drawn on it, a
   distinct-tuple count beside every aggregate, the byte-identical linear-vs-argmax finding (now at
   **43/43** as well as on `eval_b`), per-render tuples persisted, and every row carrying its §C.1 medoid
   (with `K`), §C.2, §C.4 and `n`.

⭐ **What I would rather leave behind than a tenth attempt at the ceiling:** a running log with every
measurement in it, **five of my own errors recorded as mine** (a thread-budget breach, the import trap, a
refuted structural hypothesis, a lever wrongly closed on 6 corners, and a pre-registered §C.1 margin that
measured 0.0000), a pre-registration whose central prediction — *"a TIE, not a WIN"* — was right, a ledger
that says **scored 1/3** and exactly what it bought, and **a price on what the store costs and a
demonstration of what it buys.**

⛔ **`hub/generation-table-2`: TIED, not WON. Banked and reported as such.**
