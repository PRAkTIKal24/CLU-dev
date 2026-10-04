# hepa-lh-audit — results-analyst report
**Task + acceptance criterion:** measure, per HEPA-suite dataset, how much entity history exists and how
old the information that decides an event is; acceptance = `audit.json` carries all 11 rows with the four
measurements + a criterion-1 verdict, or an explicit `NOT_ACQUIRED` record. **Status: done.**
**DIAL DECLARATION** (echoed; ⚠ the task file carries none — protocol §7 non-compliance, flagged to the Hub):
*Dial:* **none — instrument/recon.** This prices datasets, not a model. *Laundering control:* n/a (no
performance number is produced; no CLU arm, no CLU number — PREREG §6 honoured). *Falsifies the idea under
test:* all 11 rows DROP on criterion 1. *Does NOT falsify:* a dataset passing the audit and later losing
the sweep — the audit is necessary, not sufficient.

## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner at the review that accepts this report)
1. **The 8192 rung is unreachable on 8 of 11 rows** (all C-MAPSS, MBA, BATADAL, Weather, Beijing-AQ). Only
   **PSM, GECCO, ETTm1** can run it. PREREG §3's rung set must be rewritten per dataset.
2. **Both pre-registered KEEP candidates are damaged by the audit**: Beijing-AQ **cannot reach 8192**
   (val/test segments are 5236 steps), and **GECCO's ft_test split contains exactly ONE event onset**
   (77 positive steps, one contiguous run). The kill rule's clause (a) — "beats 512 by more than the seed
   spread" — is not measurable on a 1-event test split.
3. **MBA's `train.xlsx` and `test.xlsx` are byte-identical** (sha256 `bf2de57b…`, both 161 971 B, verified
   `df.equals(e)==True`): HEPA's MBA pretraining stream *is* its finetune/eval stream. Any MBA number in
   the HEPA paper inherits this. Whole MBA dataset = 7 680 samples = **27.9 s of ECG**.
4. **C-MAPSS's shipped `max_context=512` is never binding** — 0.13 % (FD003) / 0.12 % (FD004) / 0.00 %
   (FD001-2) of positive evaluation cuts ever fill it. Scout discrepancy **D3** is settled: the code's 512
   is a cap, the paper's "full engine history" is what actually happens. But paper Table 3's "8–23 tokens"
   understates the range — true engine lengths are **128–543 cycles = 8–34 tokens**.

## What I did
- Cloned HEPA (`.claude/hepa-src` @ `85f55f39`, CC BY-NC-SA 4.0, never entering a tracked file) and staged
  **all nine in-scope datasets — zero NOT_ACQUIRED**. GECCO's `spotseven.de` URL is **404 (link rot)**;
  recovered from **Zenodo DOI 10.5281/zenodo.3884398, CC BY 4.0**. BATADAL's site is live (CC BY 4.0).
  One source substitution (Weather) is declared in `PROVENANCE.md` §1.
- Wrote `.claude/scratch/hepa-lh-audit/DEFINITIONS.md` (**frozen before `audit.py` existed**), then audited
  every row **through HEPA's own `hepa.data.load_dataset`** — no loader reimplemented.
- Discharged the sequencing obligation: `.claude/data/hepa/cmapss/READY` written **12:22 CEST**, immediately
  after FD001 returned data and before any auditing.
- Artifacts: `.claude/outputs/hepa-longhorizon-triage/{audit.json, audit_raw.json, PROVENANCE.md}`,
  `.claude/scratch/hepa-lh-audit/{DEFINITIONS.md, audit.py, finalize.py, sha256.txt}`.

## Flag provenance (every number below)
| Item | Value |
|---|---|
| CHLU HEAD | `7fcef50` (= PREREG freeze point). **No tracked file touched; no branch, no commit.** |
| HEPA clone | `85f55f39a72a677d74c3371707baea67600020d6` (2026-07-27) |
| Protocol file | `hepa/utils/config.py` (clone-actual, PREREG §1). Non-default flags in effect: **none** — shipped `CONTEXT_BY_DATASET`, `NORM_POLICY`, `HORIZONS_BY_DATASET` used as-is; nothing was trained. |
| Seeds | **42** — loader engine-id permutation (`load_cmapss(seed=42)`) and the τ window sample. Deterministic; no seed spread applies (the audit is a property of the data, not of a fit). |
| Env | isolated venv, CPython 3.11.13, torch 2.14.0, pandas 2.x, numpy 2.x (CHLU `.venv` untouched; it has no torch) |
| Command | `HEPA_DATA_DIR=.claude/data/hepa PYTHONPATH=.claude/hepa-src .claude/scratch/hepa-lh-audit/venv/bin/python .claude/scratch/hepa-lh-audit/audit.py` then `python3 .../finalize.py` |
| Runtime | 12.5 s total for all 11 rows (CPU), sum of per-row `runtime_s` in `audit.json` |

**Definitions (full text: `.claude/scratch/hepa-lh-audit/DEFINITIONS.md`, echoed in `audit.json._meta`).**
*Entity* = HEPA's own; operative unit = **(entity_id, split) segment**, because `EventDataset` slices
`x[max(0,t-max_context):t]` inside one entity array — history never crosses a split boundary, and short
contexts are **truncated + zero-padded**, never back-filled. *Event* = **onset** of a contiguous positive
run. *history(e)* = onset index inside its segment = history at the last positive cut (generous reading;
the conservative `e−K` variant is in `audit.json`). *cov_R* = fraction of events with history > R.
**DROP_ON_AUDIT iff cov_512 < 0.10** (declared before computing). *Similar regime*: W=32-step
(=2 patches) windows, feature = per-channel window mean z-scored over all windows of that dataset,
τ = 10th percentile of pairwise Euclidean distances (≤5000 windows, seed 42); ttlsr = e − end of the
**latest prior same-segment window within τ**; no match ⇒ **CENSORED**.

## Table A — sizes, coverage, verdict  (source: `audit.json`)
| Dataset | Entity (HEPA's def.) | n_ent / n_seg | segment len min/med/p90/max | events (tot / **ft_test**) | pos. steps | cov>512 | cov>2048 | cov>8192 | **verdict** |
|---|---|---|---|---|---|---|---|---|---|
| FD001 | engine id | 100 / 100 | 128 / 199 / 275 / **362** | 100 / 30 | 100 | **0.000** | 0.000 | 0.000 | **DROP_ON_AUDIT** |
| FD002 | engine id | 260 / 260 | 128 / 199 / 271 / **378** | 260 / 78 | 260 | **0.000** | 0.000 | 0.000 | **DROP_ON_AUDIT** |
| FD003 | engine id | 100 / 100 | 145 / 220 / 352 / **525** | 100 / 30 | 100 | **0.010** | 0.000 | 0.000 | **DROP_ON_AUDIT** |
| FD004 | engine id | 249 / 249 | 128 / 234 / 348 / **543** | 249 / 76 | 249 | **0.004** | 0.000 | 0.000 | **DROP_ON_AUDIT** |
| PSM | 1 series → 3 segments | 1 / 3 | 8 584 / 26 153 / 47 394 / 52 704 | 73 / 21 | 24 181 | 0.959 | 0.945 | 0.644 | PASSES_AUDIT |
| MBA | 1 series → 3 segments | 1 / 3 | 568 / 2 104 / 4 107 / 4 608 | 62 / 18 | 2 533 | 0.774 | 0.339 | **0.000** | PASSES_AUDIT ⚠ |
| GECCO | 1 series → 3 segments | 1 / 3 | 34 691 / 34 692 / 62 765 / 69 783 | 51 / **1** | 1 726 | 1.000 | 0.980 | 0.745 | PASSES_AUDIT ⚠ |
| BATADAL | 1 series → 3 segments | 1 / 3 | 369 / 1 206 / 2 246 / 2 506 | **5** / 2 | 219 | 0.800 | 0.200 | 0.000 | PASSES_AUDIT ⚠ |
| ETTm1 | 1 series → 3 segments | 1 / 3 | 13 736 / 13 736 / 36 194 / 41 808 | 417 / 92 | 2 452 | 0.964 | 0.849 | 0.595 | PASSES_AUDIT |
| Weather | 1 series → 3 segments | 1 / 3 | 7 704 / 7 705 / 31 051 / 36 887 | 27 / 7 | 296 | 1.000 | 0.963 | 0.444 | PASSES_AUDIT |
| Beijing-AQ | station (12) | 12 / 36 | 5 236 / 5 236 / 24 544 / 24 544 | 2 451 / 434 | 19 168 | 0.896 | 0.844 | 0.455 | PASSES_AUDIT |

## Table B — event timing and the context ceiling
| Dataset | native step | time-to-prev-event med / p90 | time-to-last-similar-regime med / p90 (censored) | shipped ctx | **max_available_context** | max_useful_context | full-ctx frac of positive cuts @512 / @2048 / @8192 |
|---|---|---|---|---|---|---|---|
| FD001 | 1 cycle | **undefined** (1 event/engine) | — (**100 % censored**, n=0) | 512 | **100** | 100 | 0.000 / 0 / 0 |
| FD002 | 1 cycle | **undefined** | 86 / 142 (79.2 % cens.) | 512 | **100** | 100 | 0.000 / 0 / 0 |
| FD003 | 1 cycle | **undefined** | — (**100 % censored**, n=0) | 512 | **100** | 100 | 0.0013 / 0 / 0 |
| FD004 | 1 cycle | **undefined** | 74 / 149 (79.9 % cens.) | 512 | **100** | 100 | 0.0012 / 0 / 0 |
| PSM | 1 min | 481 / 2 661 | **53 / 61** (1.4 %) | 100 | **8192** | 8192 | 0.986 / 0.972 / 0.662 |
| MBA | 3.64 ms (275 Hz) | 118 / 132 | **123 / 721** (9.7 %) | 100 | **512** | 2048 | 0.835 / 0.368 / **0.000** |
| GECCO | 1 min | 290 / 2 972 | **46 / 63** (3.9 %) | 512 | **8192** | 8192 | 1.000 / 0.980 / 0.745 |
| BATADAL | 1 hour | 420 / 569 (n=2) | 193 / 527 (40 %, n=3) | 512 | **100** | 2048 | 0.645 / 0.206 / 0.000 |
| ETTm1 | 15 min | **21.5** / 584 | **87 / 195** (1.2 %) | 512 | **8192** | 8192 | 0.956 / 0.846 / 0.592 |
| Weather | 10 min | 149 / 5 100 | **59 / 129** (22 %) | 512 | **2048** | 8192 | 1.000 / 0.963 / 0.444 |
| Beijing-AQ | 1 hour | **24** / 442 | **115 / 504** (7.0 %) | 512 | **2048** | 8192 | 0.947 / 0.876 / 0.481 |

`max_available_context` = largest rung R∈{100,512,2048,8192} such that **every** split holds a segment
longer than R (derivation: `EventDataset` needs one full-length context per split or that split's rung is
pure zero-padding). `max_useful_context` = largest R where ≥10 % of *positive* cuts reach t ≥ R (pooled
over splits, so it can exceed the ceiling — Weather/Beijing-AQ/MBA/BATADAL get their long contexts only
inside `ft_train`). **The sweep spoke must use `max_available_context` as the hard ceiling.**

## Findings
**1 · The PREREG's C-MAPSS prediction is CONFIRMED, on all four rows, decisively.** cov_512 = 0.000 /
0.000 / 0.010 / 0.004 versus a 0.10 threshold. C-MAPSS **DROPs on the audit alone** — four GPU sweeps
never bought. The PREREG's stated *reason* is off in detail and I report it plainly: engines do **not**
top out at 368 cycles — FD003 reaches **525** and FD004 **543**, so a handful of engines *do* exceed 512.
It changes nothing: 1 engine in 100 and 1 in 249.
**2 · C-MAPSS time-to-previous-event does not exist.** Exactly one event per engine, by loader
construction (`labels[-1]=1`). There is no "previous event" to be far away.
**3 · C-MAPSS's pre-failure regime is NOVEL, not recent.** FD001/FD003 (single operating condition):
**100 % censored** — no prior 32-cycle window in the engine's own history is within τ of the pre-failure
window. FD002/FD004 (six conditions): ~80 % censored, and where a match exists it is 74–86 cycles back —
i.e. it matches the *operating condition*, not the degradation state. Long history would not help here;
there is nothing analogous in it.
**4 · Everywhere else, the analogous regime is RECENT.** Median time-to-last-similar-regime is
**46–193 native steps** on all seven non-C-MAPSS rows, and ≤ 721 steps at p90 on six of seven. Under this
(one, pre-declared) definition of "similar regime", a 512-step context already contains the most recent
analogue for the median event on every dataset in the suite. This is an audit-level *prior* against the
long-horizon hypothesis, on data alone, before a single model is trained.
**5 · Time-to-previous-event is short on the derived-label datasets.** ETTm1 median **21.5 steps**
(≈5.4 h) and Beijing-AQ median **24 steps** (1 day) — their "events" are clusters, not isolated
incidents. PSM (481) and GECCO (290) are longer but still ≪ 2048.
**6 · Three rows pass criterion 1 but are statistically unusable for the kill rule** (advisory; the
mechanical verdict stays PASSES_AUDIT): **GECCO ft_test = 1 event**, **BATADAL = 5 events total / 2 in
test** (and its `ft_val` segment is **369 steps**, so BATADAL's own shipped 512 context cannot be filled
even once in validation — literal ceiling **100**), **Weather ft_test = 7 events**.
**7 · The idea is NOT falsified, but it is cornered.** Three rows carry real ≥8192 history on every split
(PSM, GECCO, ETTm1) and three more carry ≥2048 (MBA reaches only 512; Weather and Beijing-AQ reach 2048).
The falsification condition in §8 of my task ("all eleven rows DROP") did **not** fire.

## Which PREREG §5/§3 predictions this audit settles
| PREREG claim | Verdict |
|---|---|
| FD001/2/3/4 "**DROP on audit**", high conf. | ✅ **SETTLED — all four DROP.** Reasoning detail corrected (max 543 cycles, not 368). |
| §3 "BATADAL cannot reach 8192" | ✅ CONFIRMED (longest segment 2 506). |
| §3 "C-MAPSS cannot reach 2048" | ✅ CONFIRMED, and stronger: it cannot meaningfully reach **512**. |
| BATADAL "**HOLD** — audit decides" | ⚠ **Audit decides: criterion 1 passes 4/5 events, but n=5 events and a 369-step val segment.** Recommend HOLD→**DROP for power**, Hub's call. |
| GECCO "**KEEP**", med conf. | ⚠ Audit cannot settle it, but the *basis* is damaged: **1 test event**. |
| Beijing-AQ "**KEEP**", med-high | ⚠ Audit cannot settle it; **8192 rung is NOT RUNNABLE** (val/test = 5 236). Max clean rung 2048. |
| PSM / MBA / ETTm1 / Weather "DROP" | Not settleable by audit (sweep-level). MBA's premise is void: 8192 steps is **longer than the entire MBA dataset** (7 680). |
| §5 governing hypothesis (label-locality ⇒ no history need) | Untouched by the audit — it needs the sweep. Note the audit gives it *indirect* support: the two shortest time-to-prev-event rows are exactly the two rolling-window-labelled ones (ETTm1 21.5, and Beijing-AQ 24 which is absolute-thresholded — so the pattern is **not** clean). |

## NOT_ACQUIRED list
**EMPTY.** 9/9 datasets, 11/11 rows acquired and loaded. (GECCO required a source substitution to Zenodo
after a 404; Weather required a substitution to the canonical Informer/Autoformer `weather.csv` mirrored
by THUML, because `bgc-jena.mpg.de/wetter/` serves yearly raw archives, not the file the loader expects.
Both are documented in `PROVENANCE.md` §1 with sha256.)

## Limitations / confounds
- "Similar regime" is **one** mechanical definition (W=32, channel means, τ = 10th pct). It is blind to
  intra-window shape and to single-channel events — precisely the failure mode HEPA's own §5.2 blames for
  BATADAL/MBA. A different W or a shape-aware feature could move medians; the *censoring* result for
  C-MAPSS is robust in direction (no prior window matched at ANY distance ≤ τ).
- Coverage uses the **generous** history reading (history at the last positive cut). The conservative
  `e−K` column in `audit.json` moves nothing above the 10 % line and moves nothing below it.
- BATADAL (5 events), Weather (27), GECCO (51) have small n; their coverage fractions are 1/5-resolution
  quantities. Reported as such, not smoothed.
- `max_available_context` is a *structural* ceiling (one full context per split). A sweep could still run
  a higher rung with zero-padding; that would measure padding, not history. The `full-ctx frac` column is
  the number to quote if anyone does.
- No h-AUROC, no training, no CLU arm was run here (PREREG §6). Nothing in this report is a performance
  claim.

## Recommended next experiments (for the sweep spoke)
1. **Do not sweep C-MAPSS.** Four datasets removed on data alone.
2. **Per-dataset rung sets** (hard, from `max_available_context`): PSM {100,512,2048,8192} · GECCO
   {512,2048,8192} · ETTm1 {512,2048,8192} · Weather {512,2048} · Beijing-AQ {512,2048} · MBA {100,512} ·
   BATADAL {512 only, and it is padded in val}. Report 2048/8192 as **NOT RUN** where the ceiling forbids,
   never as a null.
3. **Sweep PSM and ETTm1 first** — they are the only two rows with both a runnable 8192 rung and ≥20 test
   events. If the long-horizon effect is absent there, it is absent, and the section closes with a clean
   negative on the strongest available evidence.
4. **Before spending on GECCO**, decide whether a 1-event test split can carry the suite's largest claimed
   margin (+.07). My recommendation: re-split GECCO (event-stratified) or report it as uninterpretable.
5. **Quantify the padding confound explicitly**: at every rung report the mean *realised* context length of
   positive cuts alongside token count (PREREG §2.2 already demands the token count).

## Git footprint
**None.** No tracked file was created, modified, or deleted; no branch, no commit, no worktree.
Everything written lives under `.claude/` (data, scratch, outputs) and is gitignored.
`git status` at start and end shows the same tracked tree at `7fcef50`.

## Open questions / risks
- **MBA train≡test** (finding 3) is a property of the upstream TranAD release, not of HEPA's code; but it
  means MBA's pretraining is not held out. Someone should decide whether the suite's MBA row is quotable.
- The concurrent `hepa-lh-tripwire` spoke's harness sees a `cmapss` dir whose **entire content is ≤543
  cycles per engine**; if its tripwire assumes a 512-step context is formable, it will silently test
  zero-padding. `READY` states the observed lengths for exactly this reason.
- **No code bug found in HEPA's loaders.** All 11 loaded first try. Two *environmental* traps for the
  engineer: `hepa/__init__.py` imports torch (so any loader-only use still needs torch installed), and
  `scripts/download_data.py`'s GECCO URL is dead.

## Proposed handover updates (for the Hub)
- **§1.6 (experiments) — new row.** *hepa-lh-audit (2026-09-07, HEPA @ `85f55f39`, CHLU HEAD `7fcef50`,
  seed 42, no training):* all 9 in-scope HEPA datasets acquired (**0 NOT_ACQUIRED**) and audited through
  HEPA's own loaders. **All four C-MAPSS rows DROP on pre-registered criterion 1** (cov_512 = 0.000 /
  0.000 / 0.010 / 0.004 vs a 0.10 bar) — the PREREG's high-confidence prediction is confirmed and four GPU
  sweeps are cancelled. Seven rows pass. **The 8192 rung is structurally unreachable on 8 of 11 rows**;
  only PSM, GECCO, ETTm1 can run it.
- **§1.6 / §8 — the cornered-KEEP problem.** Both pre-registered KEEPs are damaged: Beijing-AQ cannot run
  8192 (val/test segments 5 236 steps; clean ceiling 2048) and **GECCO's test split holds exactly one
  event onset** — the +.07 GECCO margin, the largest in the HEPA suite, rests on a 1-event evaluation.
  PREREG §5's "fewer than 2 KEEP ⇒ section dies" clause is now live and should be re-read.
- **§5 (provenance) — new entry.** `.claude/outputs/hepa-longhorizon-triage/PROVENANCE.md` is the backup of
  record for `.claude/data/hepa/**` (35 files, sha256 + bytes + URLs + licences). Licences resolved:
  BATADAL **CC BY 4.0**, GECCO **CC BY 4.0** (Zenodo 10.5281/zenodo.3884398 — the `spotseven.de` URL is
  **404**), Beijing-AQ **CC BY 4.0** (UCI), PSM **BSD**, MBA **BSD-3**, ETTm1 **CC BY-ND 4.0** (⚠
  NoDerivatives), C-MAPSS **NOT STATED**, Jena raw **NOT STATED**. HEPA itself **CC BY-NC-SA 4.0**, cloned
  under `.claude/hepa-src` only.
- **§6 (what works) — three facts to bank.** (i) `EventDataset` truncates+zero-pads: a context rung above a
  segment's length buys padding, not history — always report realised context length. (ii) HEPA's C-MAPSS
  `max_context=512` is never binding (≤0.13 % of positive cuts fill it) — settles scout **D3**, and paper
  Table 3's "8–23 tokens" should read **8–34** (128–543 cycles). (iii) **MBA `train.xlsx` ≡ `test.xlsx`,
  byte-identical**; the whole MBA dataset is 27.9 s of ECG.
- **For `experiment-engineer`:** no HEPA code bug. Two traps: `hepa/__init__.py` pulls in torch even for
  loader-only work, and `scripts/download_data.py`'s GECCO URL is dead (use Zenodo 3884398).
- **Protocol:** the task file for `hepa-lh-audit` shipped **without a §7 DIAL DECLARATION block**; I
  supplied one at the top of this report. Worth a Hub check on the other wave-1 task files.
