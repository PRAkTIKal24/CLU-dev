# RUN THIS ON THE MAC — `hub/refusal`'s scored run, one command, ~30 min CPU

**Why this file exists.** ⛔ **The Windows hub box can only produce seed 0.** Seeds 1 and 2
need ~23,480 rows encoded through the frozen encoder, because
`gen-feature-bridge` wrote seed 0 at `cache_scope="full"` (480,000 rows) and seeds 1–2 at
`"used"` (**9,288** rows) against the **32,768** the landscape needs. On the Mac the
encoder reproduces the cache to within 1 float16 ULP, so `gather_features` extends
coverage legally and all three seeds run. On Windows it does not (81 ULP on 0.488 % of
elements at seed 1 — the declared cross-ISA delta), so the enforced gate aborts at the
point of extension, exactly as designed. ⛔ COMMON §4 forbids re-encoding, so this hub did
not. ⭐ **Everything else transfers unchanged.**

The target needs **3 seeds**. The Mac can give them. Nothing below is new science — it is
the same pre-registered criterion, the same scorer, the same protocol.

---

## 0. What you are running

`PREREG-hub-refusal-1.md` (in this directory) is binding and was filed **before any
`A`/`B`/`C`/`A7`/`A7D` row was loaded**. In one line:

> **`S_max = max_t [ E_t(x_settled) − E_store(x_settled) ]`**, refuse iff
> `S_max > quantile(S_max on the repaired stratified **val** split, q = 0.95)`,
> with `q` the harness's own shipped `GenBaselinesConfig.refusal_valid_quantile`.

⛔ **Do not change the criterion, the threshold rule, `q`, the rung, the step budget or
the arm list.** If any of them must change, that is a new PREREG and a new scored run.

---

## 1. Environment

```bash
cd <CHLU repo root>                       # the pin 4125fce7…
export CHLU_ROOT="$PWD"
export CHLU_CLAUDE_ROOT="$CHLU_ROOT/.claude"
export CHLU_GEN_DATA_ROOT="$CHLU_CLAUDE_ROOT/data/gen"
export CHLU_GEN_BRIDGE_ROOT="$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs"
export CHLU_HUB_OUTPUTS="$CHLU_CLAUDE_ROOT/outputs"
export PYTHONPATH="$CHLU_ROOT"
export PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```
The three scripts read **only** those environment variables — there is no path in them
that names a machine.

⭐ **Expected difference on the Mac, and it is the point:** `_assert_cache_reencodes` should
report `would_have_failed: false` at preflight, and `gather_features` should log
`extending cache coverage: N row(s) encoded through the frozen encoder` for seeds 1–2
after an **ENFORCED** pass. ⛔ If the enforced check fails on the Mac too, **stop** — that
is a real defect, not a platform delta, and nothing below is quotable.

## 2. The free legs first (no held-out row, no budget)

```bash
python .claude/outputs/hub-refusal/scorer/run_units.py \
    --seeds 0 1 2 --budget A --sets val MS \
    --arms full request_off zero_step store_removed
```
Check `runs/LEDGER.jsonl`: every row must carry
`"experts_reproduction": {"bit_identical": true}` — that is the proof the accept half is
scored by the **shipped** `request_experts`, unchanged. ⛔ If any row is `false`, stop.

## 3. The scored run — ⛔ only after the Head states the remaining count

```bash
python .claude/outputs/hub-refusal/scorer/run_units.py \
    --seeds 0 1 2 --budget A --sets A B C A7 A7D \
    --arms full request_off zero_step store_removed \
    --allow-scored --prereg PREREG-hub-refusal-1.md
```
⛔ The runner **refuses** to touch `A/B/C/A7/A7D` without both flags. That gate is
mechanical on purpose.

## 4. The join and the verdict

```bash
python .claude/outputs/hub-refusal/scorer/score.py --budget A \
    --arms full request_off zero_step store_removed
```
Writes `RESULTS.json` and prints: the rival rows (a JOIN on
`gen-val-split-fix/refusal_scores.json` — ⛔ **never** `gen-arms-a7-wiring/`, the script
asserts the directory), our rows for every pre-registered statistic, and the verdict
against the **strictest** rival on each kind.

## 5. What to report, in this order

1. ⭐ **The PAIR at one threshold, per kind:** `(reject out_of_range, reject
   duplicate_factor)` **with `accept B` AND `accept A7` printed beside them**. A7 is a
   **valid** control — refusing it is a false refusal. ⚠ Every rival except `lace/ode`
   falsely refuses ~60 % of it (kNN accepts `B` at 1.0000 and `A7` at **0.4120**); a
   criterion that refuses both kinds while doing the same has moved the problem, not
   solved it.
2. AUROC as the **companion**, per kind, A7-controlled, ⛔ never pooled into one number
   (R27.4) and ⛔ never reported alone.
3. All seven pre-registered companions, with `S_max` as the headline **whatever they
   say** — `S_sum` is declared a term counter in advance and may not be promoted.
4. The §C floor line: `store_removed` must collapse; `zero_step` must not match `full`.
5. The bar, from `RIVAL_ROWS.json`: beat **0.973765** (`knn_compsel`) on `out_of_range`
   and **0.990612** (`lace/ode`) on `duplicate_factor`. PREREG P2 predicts `0.80–0.99` on
   `duplicate_factor` and says plainly that landing below `0.9906` is a **partial result**.

## 6. Cost

3 seeds × 4 arms × 7 sets at rung A ≈ **0.146 s per request per 1000 steps**, measured on
the Windows box at 4 threads; ~2,022 requests per seed per arm. Budget ~30–45 min of CPU.
⛔ No GPU. ⛔ No GCloud. ⛔ Nothing that spends money.

## 7. Files in this directory

| file | what it is |
|---|---|
| `PREREG-hub-refusal-1.md` | the binding pre-registration + AMENDMENTS 1–4 |
| `LOG.md` | the running log: what was tried, measured, and what it means |
| `RIVAL_ROWS.json` | the rivals' AUROCs **and pairs**, a JOIN on the good file |
| `SYNTHETIC_PROBE_seed0_A.json` | ⛔ fit-derived synthetic, MECHANISM ONLY, **not** the benchmark |
| `runs/LEDGER.jsonl` | one row per unit, ok **and** failed |
| `runs/seed_N/A_provenance.json` | certificate, byte ledger, re-encode verdict, row split |
| `scorer/refusal_lib.py` | the scorer — calls the shipped read, re-implements nothing |
| `scorer/run_units.py` | the resumable unit runner (a kill costs ONE UNIT) |
| `scorer/score.py` | the join and the verdict |
| `scorer/synthetic_probe.py` | the declared fit-derived mechanism probe |
| `scorer/plumbing_smoke.py` | branch-coverage smoke; asserts shape/finiteness only |
