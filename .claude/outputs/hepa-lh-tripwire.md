# hepa-lh-tripwire — experiment-engineer report

**⛔ THE FALSIFICATION IN TASK §8 FIRED. On FD001 at context 512, a matched-byte kNN exemplar store
reaches h-AUROC 0.9777 ± 0.0006 (3 seeds) — beating HEPA's own clone-actual FD001 number (0.918 ±
0.008) by +0.060 and the published Table-1 number (0.81 ± 0.03) by +0.168. It beats them at the
SMALLEST of the three byte budgets too (0.9646 at 32 768 B).**

Task + acceptance criterion: build the matched-byte tripwire (kNN store at HEPA's own state-byte
budget + the three trivial baselines) and land `code/tripwire.py` + `tripwire_smoke.json` for FD001 @
512 at 3 seeds. **Status: done** (the §3 precondition `.claude/data/hepa/cmapss/READY` was already on
disk when the harness was finished; nothing is unvalidated).

> ⚠ **RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **PREREG §4(b) cannot be applied to C-MAPSS as written.** The tripwire does not "close the gap"
>    on FD001 — it is 0.060 **above** HEPA's best reported FD001 number at the 512 rung itself. Any
>    C-MAPSS KEEP is dead before the context sweep starts. The Hub must decide whether §4(b) is read
>    as a gap-closing test (as drafted) or as an absolute-dominance test (what the data now demands).
> 2. **GECCO's headline margin has a no-model competitor.** A 24-bin daily-phase prevalence table
>    ("seasonal", period 1440 min) scores **0.8181** on GECCO @512 — equal to the best published
>    baseline (MAE .81) and within one HEPA std of HEPA (.88 ± .06). GECCO is one of PREREG §5's two
>    predicted KEEPs and carries the suite's largest margin (+.07).
> 3. The **byte ledger** (§2 below) is the Hub-review gate named in task §4.1; every KEEP verdict in
>    wave 2 inherits it.

**Dial:** none — instrument. **Laundering control:** this harness *is* the laundering control (a
same-byte lookup and three no-model references). **Falsifies the claim:** a tripwire that matches or
beats HEPA at 512 on FD001 falsifies "HEPA's encoder does something a same-byte lookup cannot" — it
did. **Does NOT falsify:** a tripwire that loses to HEPA on a rung the dataset cannot physically
reach; and losing to an oracle is the metric-native-ceiling theorem, not news.

---

## 1 · What I did
- Wrote the harness (7 modules, `~/.claude/outputs/hepa-longhorizon-triage/code/`). It reads through
  **HEPA's own loader** (`load_dataset` → `global_zscore_bundle` → `EventDataset`, same cut points,
  same strides 4/4/1, same `min_context=128`, same entity filter) and scores with **HEPA's own
  metric** (`hepa.evaluation.metrics.h_auroc`, imported, not reimplemented) on labels from **HEPA's
  own `build_label_surface`**. Nothing in the metric or label path is mine.
- Pre-registered my own point predictions in `code/PREREG_tripwire.md` **before scoring any arm**
  (the Hub's frozen `PREREG.md` predicts datasets, not tripwire values). Scored 2/5 predictions right.
- Ran FD001 @512, seeds {42,123,456}, all 3 byte-ledger variants × 4 descriptors × {compliant,
  diagnostic} payloads → `tripwire_smoke.json`. Plus two validation runs: FD001 @2048 and GECCO @512.

## 2 · THE BYTE LEDGER (the thing the Hub reviews)
Constants from the clone, not from memory: `P=16`, `d_model=256`, `n_layers=2` (`hepa/utils/config.py::
PROTOCOL`, `hepa/model/encoder.py::CausalEncoder`); dtype = torch default **fp32 = 4 B** (encoder.py
declares none). `n_tokens(L) = ceil(L/P)`.

| reading | arithmetic | L=512 | L=2048 | L=8192 |
|---|---|---|---|---|
| `token_only` — the patch-embedded token stream | `n·256·4` | 32 768 | 131 072 | 524 288 |
| **`residual_stream` (PRIMARY)** — stream at each of 2 blocks + final norm | `(2+1)·n·256·4` | **98 304** | **393 216** | **1 572 864** |
| `kv_cache` — tokens + K,V at every layer + pooled h_t | `(1+2·2)·n·256·4 + 256·4` | 164 864 | 656 384 | 2 622 464 |

(`n` = 32 / 128 / 512.) **Rejected reading:** the pooled `h_t` alone (1 024 B) — it is
context-independent, so it gives every rung the same budget and makes the 8192 rung a strawman
(task §4.3 forbids exactly this). All three are computed at every rung and all three are in the JSON,
so the Hub can re-rule the primary without a re-run. **Store sizing:**
`n_exemplars = floor(budget / ((feature_dim + payload_floats)·4))`, recomputed per rung. Not charged:
numpy headers and the scaler's `2·d` floats (a few hundred bytes; charging them only shrinks the
tripwire). Realised utilisation is 87–100 % of budget in every arm and is in the JSON.

## 3 · Results — FD001, context 512, seeds {42,123,456}
`tripwire_smoke.json`. HEPA references (not measured by me): paper **.81±.03**, clone-actual
**.918±.008**.

| arm | store | realised bytes | h-AUROC (test) | vs HEPA .918 |
|---|---|---|---|---|
| **kNN `chan_stats`, `residual_stream`** | 431/1167 exemplars, dim 56 | 98 268 / 98 304 | **0.9777 ± 0.0006** (range .0013) | **+0.060** |
| kNN `chan_stats`, `token_only` | 143/1167 | 32 604 / 32 768 | 0.9646 ± 0.0035 | +0.047 |
| kNN `chan_stats`, `kv_cache` | 723/1167 | 164 844 / 164 864 | 0.9798 ± 0.0006 | +0.062 |
| kNN `last_patch`, `residual_stream` | 109/1167, dim 224 | 98 100 | 0.9167 ± 0.0002 | −0.001 |
| kNN `patch_means`, `residual_stream` | 54/1167, dim 448 | 96 984 | 0.8164 ± 0.0533 | −0.102 |
| kNN `raw_window`, `residual_stream` | **3**/1167, dim 7168 | 86 028 | 0.4998 ± 0.0004 | −0.418 |
| *diagnostic* `chan_stats::surface` (⛔ not PREREG-compliant) | 119/1167 | 98 056 | 0.9727 ± 0.0041 | +0.055 |
| **trivial: prevalence** | — | 0 | **0.5000** `deterministic` | −0.418 |
| **trivial: seasonal** | — | — | **NOT_APPLICABLE** — 1 sample/flight-cycle on a run-to-failure stream; no calendar period exists | — |
| **trivial: last-event-age** (no model, no window) | — | 0 | **0.7135** `deterministic` | −0.205 |

`k` ∈ {1,4,16,64} selected on **ft_val**; test never used for selection. Descriptor also selected on
val — `chan_stats` wins there too, so the headline is not a test-set pick.

**Leakage controls** (`.claude/scratch/hepa-lh-tripwire/{controls,null}.py`): train/test engine-id
overlap = **∅**; feature-shuffled store **0.5172**; random scores **0.4991**; oracle `−tte` **1.0000**;
20-draw payload-permutation null **0.518 ± 0.088** vs real 0.977 → **z = 5.2**. My h-AUROC and a
hand-rolled sklearn per-horizon mean agree to 6 dp (0.977029), confirming the import is the metric.

**Pre-registration scorecard** (`code/PREREG_tripwire.md`): prevalence 0.500 ✓ exact; seasonal
NOT_APPLICABLE ✓; kNN best 0.9777 vs predicted 0.85 [0.78–0.92] ✗ **too low — the store is far
stronger than I predicted**; last-event-age 0.7135 vs predicted 0.80 [0.72–0.88] ✗ just below;
seed range 0.0013 vs predicted 0.02 [0.005–0.06] ✗ too high. **My central falsification call ("does
NOT beat 0.918") was wrong** — I record that plainly rather than reframing it.

## 4 · Two validation runs (not the acceptance artifact)
- **FD001 @2048** (`scratch/.../FD001_L2048.json`): effective context is *identical* to the 512 rung
  (mean 178, max 311, **0 %** of windows reach either rung) yet h-AUROC rises 0.9777 → **0.9816**,
  purely because a 4× budget buys 1167 exemplars instead of 431. That is the tripwire demonstrating
  its own thesis: **+0.004 h-AUROC from bytes alone, with zero extra history.** The store also crosses
  into the **`deterministic`** regime here (pool fully retained) and the JSON says so instead of
  faking ±0.00 across three runs.
- **GECCO @512** (`scratch/.../GECCO_L512.json`, RevIN path, 34 563 test windows, 98.9 % at full rung):
  seasonal **0.8181** `deterministic`, last-event-age 0.5182, kNN `chan_stats` 0.6004 ± 0.0217. See
  reconciliation item 2. This run also validates the non-C-MAPSS path end-to-end.

## 5 · Conventions the Hub must quote with these numbers
- **Degenerate horizons: the SHIPPED CODE's rule** — a horizon is dropped only if it is *exactly*
  degenerate (`n_pos == 0 or n_pos == n_total`). The paper/`PROTOCOL.md` threshold at 0.001/0.999
  (scout D4). Task §4 told me to use the code's; I did, and every JSON says so in
  `protocol.degenerate_horizon_convention`. All 150 FD001 / 200 GECCO horizons were non-degenerate,
  so on these two datasets the two conventions coincide anyway.
- **Horizon conditioning is scalar-concat everywhere** (PREREG §2.4): the store carries one scalar
  time-to-event and answers `p(t,Δt) = weighted frac(neighbour tte ≤ Δt)`. No per-horizon table, no
  264K embedding. The `::surface` arms *do* carry a K-vector, are flagged
  `prereg_compliant: false`, and are excluded from the tripwire number by construction.
- **Deterministic vs seeded store** (`--seeds` is honest, never cosmetic). Rule: deterministic iff
  `floor(budget(L)/((d_f+1)·4)) ≥ n_pool`. For the primary budget + `chan_stats`:

  | | FD001 | FD002 | FD003 | FD004 | PSM | MBA | GECCO | BATADAL | ETTm1 | Weather | BeijingAQ |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | n_pool (ft_train windows) | 1 167 | 2 972 | 1 792 | 4 343 | 13 144 | 1 120 | 17 414 | 595 | 10 420 | 9 190 | 73 248 |
  | @512 | seed | seed | seed | seed | seed | **det** | seed | seed | seed | seed | seed |
  | @2048 | **det** | seed | seed | seed | seed | **det** | seed | seed | seed | seed | seed |
  | @8192 | **det** | **det** | **det** | **det** | seed | **det** | seed | **det** | **det** | seed | seed |

  (`.claude/scratch/hepa-lh-tripwire/pools.json`; MBA needed `openpyxl`, which HEPA's `pyproject`
  omits — flagging for the audit spoke, not fixed in their tree.) **MBA is deterministic-store at
  every rung; report one number labelled `deterministic`, never a ±0.00.**

## 6 · The exact call signature wave 2 uses
```bash
cd /Users/user/Desktop/CHLU
PYTHONPATH=.claude/outputs/hepa-longhorizon-triage/code \
.claude/scratch/hepa-lh-tripwire/venv/bin/python \
  .claude/outputs/hepa-longhorizon-triage/code/tripwire.py \
  --dataset FD001 --context 512 --seeds 42 123 456 --out <path>.json
# optional: --budgets token_only residual_stream kv_cache
#           --descriptors chan_stats last_patch patch_means raw_window
#           --no-diagnostic     env: HEPA_SRC HEPA_DATA_DIR TRIPWIRE_OUT
```
Library form: `tripwire.run(dataset, context, seeds, budgets, descriptors)`; the kill rule consumes
`res["tripwire"]["tripwire_h_auroc"]`. Runtime: **43 s** (FD001, full grid), **47 s** (GECCO, 60 k
windows) on CPU. Full docs in `code/README.md`. **The acceptance run was re-executed from scratch
and is bit-identical** (`json` compare modulo timestamp/runtime), so wave 2 inherits a deterministic
harness given (dataset, context, seeds).

## 7 · Flag provenance (mandatory)
| item | value |
|---|---|
| CHLU repo HEAD | `7fcef501fb1aeae33a8d149b046c9b6126dcecd7` (= PREREG's frozen `7fcef50`); **tracked tree clean, zero tracked files touched, no branch, no commit** |
| HEPA clone | `.claude/hepa-src` @ `85f55f39a72a677d74c3371707baea67600020d6`, 2026-07-27, clean. Imported at run time; **never vendored** (CC BY-NC-SA 4.0) |
| protocol file | `hepa/utils/config.py` + `REPRODUCIBILITY.md` (clone-actual), per PREREG §1 |
| seeds | 42, 123, 456 |
| dataset / context | FD001 / 512 (also 2048); GECCO / 512 |
| norm policy | FD001 `none` → loader-side `global_zscore_bundle`; GECCO `revin` → per-window z-score at featurisation (encoder.py parity: mean over time, unbiased std, eps 1e-5) |
| horizons | K=150 FD001 (Δt 1…150), K=200 GECCO — HEPA's `get_horizons` |
| strides / min_context | 4 / 4 / 1, `min_context=128`, entity filter `len ≤ 129` dropped — HEPA's own |
| k grid / selection | {1,4,16,64}, selected on **ft_val** h-AUROC |
| descriptor alignment | right-aligned at the cut, zero-padded at the front (declared; favours the tripwire) |
| distance / vote | Euclidean on store-fitted z-scored features; inverse-distance-weighted vote |
| env | **separate venv** `.claude/scratch/hepa-lh-tripwire/venv`: Python 3.11.13, torch **2.14.0**, numpy **2.4.6**, sklearn **1.9.0**, pandas 3.0.5, + openpyxl. ⚠ NOT the CHLU venv (numpy 2.4.1, sklearn 1.8.0, **no torch**) — torch is needed only because `hepa/evaluation/surface.py` and `finetune.py` are torch modules; **no model is trained anywhere in this harness** |
| trained anything? | **no.** No CLU arm, no CLU number, no HEPA training run. |

## 8 · Open questions / risks
1. **Why is FD001 this easy?** `norm_mode="none"` keeps the absolute sensor drift level, and 56 floats
   of per-channel mean/std/last/slope over the window recover RUL almost exactly. This is the classic
   similarity-based RUL method at 98 KB. It suggests HEPA's C-MAPSS cells price the *loader's
   normalisation choice*, not the encoder. **Not my call to make — flagging.**
2. **`raw_window` is a strawman at these budgets** (1–5 exemplars → 0.50). Reported unsuppressed, but
   it should not be read as "raw windows don't work"; it is a budget artifact and the JSON shows why.
3. I did **not** re-run HEPA itself; every HEPA number here is quoted from the scout report. A wave-2
   HEPA-vs-tripwire cell should be measured in one harness on one machine.
4. My cross-dataset validation is 2 descriptors × 1 budget on GECCO. The GECCO seasonal number is
   complete and deterministic, but its exemplar-store column is **not** the full grid.
5. The scratch venv is regenerable in one command (`code/README.md`) but lives under `scratch/`; if
   wave 2 runs after a scratch clean, it must be recreated first.

## Proposed handover updates (for the Hub)
- **New known issue / finding:** *a matched-byte kNN exemplar store beats HEPA's own best reported
  FD001 h-AUROC (0.9777 vs 0.918) at every byte budget tested, and a zero-model daily-phase table ties
  the best published GECCO baseline (0.8181).* Two of the suite's fourteen cells have a no-learning
  competitor. This is a claim about **the benchmark**, not about CLU.
- **PREREG §4(b) needs an interpretation ruling** (reconciliation item 1) before wave 2 scores any
  KEEP; the clause as drafted assumes the tripwire sits *below* HEPA.
- **Byte-ledger primary = `residual_stream`** (98 304 B @512). If the Hub re-rules it, no re-run is
  needed — all three variants are already in every JSON.
- **Env fact worth recording:** the CHLU venv has **no torch**; HEPA work needs the separate scratch
  venv. Also `HEPA/pyproject.toml` omits `openpyxl`, without which the MBA loader raises.
- **Cross-check with the audit spoke:** `audit.json` gives FD001 `max_available_context = 100`,
  `DROP_ON_AUDIT`; my independent telemetry agrees in direction (0 % of FD001 windows reach even the
  512 rung; mean effective context 178, max 311). GECCO `PASSES_AUDIT`, ceiling 8192 — consistent with
  my 98.9 %-at-full-rung measurement at 512.
