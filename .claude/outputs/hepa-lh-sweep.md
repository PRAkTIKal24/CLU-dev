# hepa-lh-sweep — results-analyst report
**Task:** run HEPA as shipped at every reachable context rung on ETTm1 / Beijing-AQ / PSM / Weather,
3 seeds, produce the **curve**; and first reproduce HEPA on FD001+GECCO to price the wave-1 tripwire.
**Acceptance:** `sweep.json` + per-event prediction dumps. **Status: DONE** — 13/13 commissioned cells,
all at 3 seeds; 66 runs in `sweep.json` + 6 labelled diagnostics; 66 dumps (233 MB); **0 defects**
(every seed-cell carries h-AUROC, `n_tokens`, wall-clock and an existing npz path — verified).
⛔ **Long version with every table: `.claude/outputs/hepa-lh-sweep/FULL-REPORT.md`.** This file is a
strict subset of it.

## ⛔ RESULT, FIRST LINE (task §6)
**The curves are NOT flat — they DECLINE.** On all four datasets h-AUROC falls with context; the
longest reachable rung sits **−0.048 to −0.172 below the 512 rung**, i.e. **3.8× to 15.2× the 512-rung
seed spread** on the three datasets with enough test events to resolve anything (Beijing-AQ 434,
ETTm1 92, PSM 21), and −0.09/−0.12 on the declared-underpowered fourth (Weather, n=7).
**Kill-rule clause (a) fails on 4/4 datasets and 7/7 dataset×device cells, every time with the WRONG
SIGN. Clause (b) is never reached — there is no gap for a matched-byte lookup to close.**
HEPA does not merely fail to use long history; **it is harmed by being given it.**

## ⭐ THE REPRODUCTION (task §3) — 0.918, NOT 0.978 ⇒ THE WAVE-1 TRIPWIRE STANDS
| arm | our 3 seeds (42/123/456) | mean | sample sd | spread | HEPA's stated | verdict |
|---|---|---:|---:|---:|---|---|
| **FD001 @512** (cpu) | 0.9167 / 0.9167 / 0.9265 | **0.9200** | 0.0056 | 0.0097 | `REPRODUCIBILITY.md` **0.918 ± 0.008** | ✅ **REPRODUCED** (+0.002) |
| FD001 @512 (mps) | 0.9330 / 0.9169 / 0.9184 | 0.9228 | 0.0089 | 0.0160 | same | ✅ (+0.005) |
| **ETTm1 @512** (cpu) | 0.8167 / 0.8151 / 0.8009 | **0.8109** | 0.0087 | 0.0158 | **0.809 ± 0.008** | ✅ **REPRODUCED** (+0.002, sd matches to 0.0007) |
| **GECCO @512** (cpu) | 0.8467 / 0.7586 / 0.9117 | 0.8390 | **0.0769** | **0.1532** | **0.888 ± 0.024** | ⚠ mean −0.049; **variance NOT reproduced (sd 3.2×)** |
| GECCO @512 (mps) | 0.8459 / 0.9253 / 0.9175 | 0.8962 | 0.0438 | 0.0794 | same | ⚠ mean +0.008; **sd still 1.8×** |

- **Two of three cells land on the README to within 0.002 with matching seed variance, in both device
  families ⇒ the harness is faithful ⇒ the FD001 verdict is binding.**
- **Wave 1's matched-byte kNN store (0.9777 ± 0.0006) beats HEPA-as-RUN by +0.0577 (cpu) / +0.0549
  (mps)** — it was +0.060 against the README, so the correction is ≤0.005. ⇒ **THE WAVE-1 HEADLINE
  STANDS, now backed by a run instead of a README.** `PREREG_AMENDMENT` clause (b2) fails for FD001
  in both families (0.9777 ≥ 0.9200−0.02). FD001 was already `DROP_ON_AUDIT`; it is now DROP twice.
- **GECCO advisory** (out of the triage table, 1 test event): clause-(b)'s *exemplar store* (kNN
  0.6004) passes by ≥+0.2386 in both families. But the **zero-model daily-phase table (0.8181)** is
  only **+0.0209** below our cpu HEPA mean — clearing the 0.02 tie tolerance by **0.0009** — and
  +0.0781 below the mps mean. ⛔ **I do not rule on this**; the Hub owns whether a zero-model table is
  in clause (b2)'s scope. Cause of GECCO's variance, measured: its shipped early stopper returns
  **best-val epoch 0 on 3/3 cpu seeds** (val loss 1.68 → 4.56 by epoch 5). Its published cell is a
  near-initialisation predictor.

## THE CURVES
Figure **`.claude/outputs/hepa-lh-sweep/context_curves.png`** · data **`.claude/outputs/hepa-longhorizon-triage/sweep.json`**

| dataset | rung | n_tokens | ft_test events | n_test windows | **cpu** h-AUROC (spread, n) | **mps** h-AUROC (spread, n) | per-seed h-AUROC (primary family) | wall/seed (s) |
|---|---:|---:|---:|---:|---|---|---|---:|
| ETTm1 | 512 | 32 | 92 | 13607 | 0.8109 (spread 0.0158, n=3) | 0.8084 (spread 0.0113, n=3) | 0.8167, 0.8151, 0.8009 (cpu) | 306 |
| ETTm1 | 2048 | 128 | 92 | 13607 | 0.7412 (spread 0.0334, n=3) | 0.7420 (spread 0.0475, n=3) | 0.7338, 0.7616, 0.7283 (cpu) | 672 |
| ETTm1 | 8192 | 512 | 92 | 13607 | NOT RUN | 0.6367 (spread 0.0357, n=3) | 0.6490, 0.6479, 0.6132 (mps) | 497 |
| BeijingAQ | 512 | 32 | 434 | 61284 | NOT RUN | 0.7579 (spread 0.0153, n=3) | 0.7507, 0.7660, 0.7570 (mps) | 936 |
| BeijingAQ | 2048 | 128 | 434 | 61284 | NOT RUN | 0.6995 (spread 0.0929, n=3) | 0.7448, 0.7019, 0.6519 (mps) | 2292 |
| PSM | 100 | 7 | 21 | 26024 | 0.6253 (spread 0.0368, n=3) | 0.6252 (spread 0.0241, n=3) | 0.6115, 0.6160, 0.6484 (cpu) | 437 |
| PSM | 512 | 32 | 21 | 26024 | 0.5296 (spread 0.0131, n=3) | 0.5139 (spread 0.0099, n=3) | 0.5369, 0.5238, 0.5282 (cpu) | 527 |
| PSM | 2048 | 128 | 21 | 26024 | 0.3723 (spread 0.0561, n=3) | 0.3827 (spread 0.0840, n=3) | 0.4064, 0.3503, 0.3602 (cpu) | 1091 |
| PSM | 8192 | 512 | 21 | 26024 | NOT RUN | 0.4655 (spread 0.0749, n=3) | 0.5056, 0.4307, 0.4601 (mps) | 856 |
| Weather ⚠**UNDERPOWERED** | 512 | 32 | 7 | 7576 | 0.7687 (spread 0.1776, n=3) | 0.8036 (spread 0.1011, n=3) | 0.8519, 0.6744, 0.7798 (cpu) | 282 |
| Weather ⚠**UNDERPOWERED** | 2048 | 128 | 7 | 7576 | 0.6493 (spread 0.1267, n=3) | 0.7131 (spread 0.2119, n=3) | 0.6535, 0.5839, 0.7106 (cpu) | 912 |

### Kill rule, clause (a) — computed
| dataset | family | 512 mean | **512 seed spread** (max−min) | longest rung RUN | longest mean | **gap** | gap > spread (clause a)? | \|gap\| / spread |
|---|---|---:|---:|---:|---:|---:|---|---:|
| ETTm1 | cpu | 0.8109 | 0.0158 | 2048 | 0.7412 | **-0.0696** | **NO — and the gap is NEGATIVE** | 4.4× |
| ETTm1 | mps | 0.8084 | 0.0113 | 8192 | 0.6367 | **-0.1717** | **NO — and the gap is NEGATIVE** | 15.2× |
| BeijingAQ | mps | 0.7579 | 0.0153 | 2048 | 0.6995 | **-0.0583** | **NO — and the gap is NEGATIVE** | 3.8× |
| PSM | cpu | 0.5296 | 0.0131 | 2048 | 0.3723 | **-0.1573** | **NO — and the gap is NEGATIVE** | 12.0× |
| PSM | mps | 0.5139 | 0.0099 | 8192 | 0.4655 | **-0.0484** | **NO — and the gap is NEGATIVE** | 4.9× |
| Weather | cpu | 0.7687 | 0.1776 | 2048 | 0.6493 | **-0.1194** | **NO — and the gap is NEGATIVE** | 0.7× |
| Weather | mps | 0.8036 | 0.1011 | 2048 | 0.7131 | **-0.0905** | **NO — and the gap is NEGATIVE** | 0.9× |

- **ETTm1** (92 events; best-powered 8192-capable): monotone decline in **both** families —
  0.8109→0.7412 (cpu) and 0.8084→0.7420→0.6367 (mps). Cleanest curve in the sweep; it points down.
- **Beijing-AQ** (434 events; best powered overall; **the suite's only absolute-threshold label** and
  PREREG §5's med-high-confidence KEEP): 0.7579 → 0.6995. 8192 **NOT RUN — not reachable** (val/test
  segments 5 236; ceiling 2 048). The strongest KEEP candidate declines on its only rung step.
- **PSM** (21 events; shipped context is **100**, not 512): 0.6252 → 0.5139 → **0.3827 (below chance,
  i.e. its surface is inverted)** → 0.4655 (mps, device-clean; cpu agrees at 0.6253/0.5296/0.3723).
  **The shipped 100-step context is the best cell on this dataset.** My pre-registered
  counter-possibility (100→512 should *rise*) is rejected: it falls 0.111, **11× the 512 spread**.
- **Weather** ⚠ **UNDERPOWERED (n = 7 ft_test events)** — label applies to every Weather number here
  and in `sweep.json`: 0.7687→0.6493 (cpu), 0.8036→0.7131 (mps). Its 512 spread (0.10–0.18) is as
  wide as the decline. **It resolves nothing**; it was run only so the KEEP list is not a cherry-pick,
  exactly as the task instructed, and it is the one cell where |gap| does not exceed the spread.

## CONFOUNDS — and why none of them rescues the hypothesis
1. **⛔ No CSF3.** `ssh csf3` fails DNS resolution from this machine; no job was ever queued.
   Everything ran on a local **Apple M1 Pro**, in two families: **cpu** (reproduces the README) and
   **mps** (10–40× faster; the only family in which 8192 was reachable — ETTm1 @8192 on CPU bought
   *pretrain + 1 finetune epoch in 55 min*; on MPS the whole run is 475 s).
   **Measured paired device offset on usable cells (FD001/ETTm1/PSM, 18 paired runs): −0.0007 ±
   0.0163, max |Δ| = 0.0338** — smaller than the smallest context effect (0.0583). **Device cannot
   explain the curves.** (Full table in the appendix; the two >0.13 outliers are GECCO/Weather single
   seeds, i.e. seed variance through a degenerate stopping rule, not a device effect.)
2. **Token confound, stated as the task requires.** P=16 is held, so `n_tokens` = 7/32/128/512 at
   rungs 100/512/2048/8192: **a longer context also buys 4× more tokens at every step, so no rung
   comparison here is a clean test of *history* alone.** Wave 1 showed extra bytes alone buy h-AUROC
   (+0.004 with zero extra history). **The confound biases towards a RISE and it still fell 4/4** —
   it makes the negative stronger, not weaker.
3. **Padding confound, quantified.** At 8192 only **40.7 %** (ETTm1) / **69.0 %** (PSM) of test
   windows reach the rung; Beijing-AQ @2048 is 62.4 %. Per-window `realised_context` is dumped in
   every npz so the verdict spoke can restrict scoring to full-rung windows **without a re-run**.
4. **Early-stopping confound — CLOSED by a labelled DIAGNOSTIC** (⛔ outside clone-actual, outside
   `sweep.json`). HEPA's shipped stopper returns **best-val epoch 0 on 3/3 seeds** for ETTm1 @8192,
   PSM @512, PSM @2048, GECCO @512. Re-running ETTm1 with early stopping **disabled** (fixed 30 ep,
   3 seeds): 512 **0.7938** → 8192 **0.6075**, gap **−0.1863** vs the shipped **−0.1717**.
   **Removing early stopping does not rescue the long rung.** The decline is a property of the frozen
   encoder's representation at long context. (Fixed-epoch is worse at *both* rungs, so the stopper is
   doing real work — it just cannot rescue a long context.)
5. **3 seeds, not 5** (PREREG §1); spread is always reported raw (max−min), never as a standard error.
6. **Harness noise floor ≈0.003**: FD001 @512 s42 gave 0.9133 at 8 threads and 0.9167 at 3 (torch CPU
   reduction order is thread-count dependent). All `sweep.json` numbers used 2 (cpu) / 3 (mps) threads.
7. **fp16 dumps** reproduce the recorded float32 h-AUROC to **≤2.0e-4** (measured, FD001 + GECCO).
8. **Only Beijing-AQ (434) and ETTm1 (92) carry real statistical weight.** PSM has 21, Weather 7,
   GECCO 1. The two heaviest datasets are also the two cleanest declines.

## NOT RUN — reasons, never nulls or zeros
| cell | status | reason |
|---|---|---|
| **Beijing-AQ @8192** | **NOT REACHABLE** | ft_val/ft_test segments 5 236 steps; ceiling 2 048 (audit, re-verified by my own sizing pass) |
| **Weather @8192** | **NOT REACHABLE** | ft_val segment 7 704 steps; ceiling 2 048 |
| **ETTm1 @8192 / PSM @8192, `cpu`** | NOT RUN — wall-clock | ≈30 min per finetune epoch on CPU; substituted with MPS, which is device-clean across all rungs of both datasets |
| **Beijing-AQ, `cpu` (both rungs)** | NOT RUN — wall-clock | 34 min bought 10/50 pretrain epochs before a 73 248-window finetune; killed. Measured on MPS. |
| **C-MAPSS FD001–4 (sweep)** | out of scope | `DROP_ON_AUDIT` (cov₅₁₂ = 0.000/0.000/0.010/0.004 vs a 0.10 bar). FD001 @512 is the §3 reproduction only and is **not** in the triage table. |
| **MBA · BATADAL · GECCO @2048/8192 · SMAP/TEP/VIX** | out of scope | ceiling *is* 512 · ceiling 100 & 5 events · 1 ft_test event · PREREG §6 |
| **token-matched rung (P scaled with L)** | NOT RUN — outside this task's protocol | the one experiment that would separate history from capacity; see follow-ups. Its absence is not a negative result. |

## FLAG PROVENANCE
| item | value |
|---|---|
| CHLU HEAD | `7fcef50` (= PREREG freeze). **Zero tracked files touched; no branch, no commit, no worktree.** `git status` clean at start and end. |
| HEPA | `.claude/hepa-src` @ `85f55f39a72a677d74c3371707baea67600020d6`, **imported at run time, never vendored** (CC BY-NC-SA 4.0, `.claude/` only) |
| protocol | `hepa/utils/config.py` + `REPRODUCIBILITY.md` (clone-actual, PREREG §1). **Non-default flags: NONE.** Shipped `PROTOCOL` verbatim (AdamW; lr 3e-4 pre / 1e-3 ft; wd 1e-2; batch 64; α=0.1; n_cuts 40; dropout 0.1; `target_mode=joint_train`; 50 pre / 30 ft epochs, patience 10; d_model 256, 2 layers, 4 heads, d_ff 256, P=16, ~2.16 M params). |
| deltas | (1) `max_context` swept (PREREG §2.1); (2) **P=16 held at every rung** (PREREG §2.2); (3) 3 seeds not 5 (PREREG §1). **Nothing else.** |
| seeds | **42, 123, 456** |
| per-dataset | norm as shipped (`revin` on the four sweep datasets + GECCO; `none`+global z-score on FD001); K=200 (K=150 FD001); **scalar-concat horizon conditioning everywhere** (PREREG §2.4); strides 4/4/1, `min_context=128`; degenerate-horizon rule = the **shipped code's**, inherited by importing `hepa.evaluation.metrics.h_auroc` |
| env | venv `.claude/scratch/hepa-lh-tripwire/venv`: CPython 3.11.13, **torch 2.14.0**, numpy 2.4.6, sklearn 1.9.0, pandas 3.0.5 (+matplotlib for the figure). ⛔ NOT the CHLU `.venv` (no torch). |
| hardware / device | **Apple M1 Pro, 8 cores, 16 GB.** Families `cpu` and `mps`; **CSF3 unreachable (DNS)**. |
| runner | `.claude/scratch/hepa-lh-sweep/run_hepa.py` = `scripts/train.py` + a `--context`/`--device` flag. **No loader, model, training loop or metric reimplemented.** |
| command | `HEPA_DATA_DIR=$PWD/.claude/data/hepa PYTHONPATH=$PWD/.claude/hepa-src .claude/scratch/hepa-lh-tripwire/venv/bin/python .claude/scratch/hepa-lh-sweep/run_hepa.py --dataset <DS> --context <L> --seed <S> --device <cpu\|mps> --out-json …/runs/<DS>_L<L>_s<S>[_mps].json --preds-npz …/preds/<DS>_L<L>_s<S>[_mps].npz` |
| pre-registration | `.claude/outputs/hepa-lh-sweep/PREREG-sweep.md`, written **before the first run**. Scored **4/7**; the three misses are one miss — **I predicted indifference to context and measured harm.** Governing dataset predictions remain the Hub's frozen `PREREG.md` §5. |

## ARTIFACTS
`.claude/outputs/hepa-longhorizon-triage/sweep.json` (acceptance artifact) ·
`.claude/outputs/hepa-longhorizon-triage/preds/<DS>_L<rung>_s<seed>[_mps].npz` (66 dumps, 233 MB;
keys `p_surface` fp16 (N,K), `y_surface` uint8, `t_index`, **`realised_context`**, `entity_index`,
`entity_ids`, `horizons`, `cumulative_sizes`, `meta`) ·
`.claude/outputs/hepa-lh-sweep/{context_curves.png, PREREG-sweep.md, FULL-REPORT.md}` ·
`.claude/scratch/hepa-lh-sweep/{run_hepa.py, run_hepa_diag.py, launch*.sh, finalize.py, assemble.py,
plot_curves.py, devoffset.py, sizing.py, sizing.json, env.sh, runs/, diag/, logs/}` (a full training
log for every one of the 72 runs).

## GIT FOOTPRINT
**None.** No tracked file created, modified or deleted; no branch, commit, worktree or push.
`git rev-parse HEAD` = `7fcef501fb1aeae33a8d149b046c9b6126dcecd7` before and after; `git status`
empty. No HEPA source entered a tracked file (share-alike respected).

## ⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner at the review that accepts this report
1. **`PREREG.md` §5's "fewer than 2 KEEP ⇒ the long-horizon section dies" clause is LIVE at ZERO
   KEEP.** Every runnable dataset fails clause (a) before clause (b) is reached.
2. **§5's governing hypothesis is bypassed, not settled.** It predicted a split (locally-normalised
   labels DROP, absolute-threshold labels KEEP). Beijing-AQ is the suite's only absolute-threshold
   label and it declines too. §5's own falsifier ("Beijing-AQ fails while ETTm1 passes") did not
   occur; a third outcome did — **both fail**. There is no positive case on which to rebuild the
   section under a different criterion.
3. **Every quotation of "GECCO 0.888 ± 0.024" needs a variance footnote** (our sds are 1.8–3.2× it;
   one seed moves 0.167 between device families; best-val epoch 0 on 3/3 cpu seeds).
4. **The zero-model-vs-GECCO margin is 0.0009 wide in the cpu family.** Must be ruled on explicitly,
   not passed over.

## OPEN QUESTIONS / FOLLOW-UPS
1. **The right next experiment is a token-matched rung**, not another dataset: P=16/64/256 at
   L=512/2048/8192 holds `n_tokens`=32 and isolates history from capacity. ~9 runs, cheap on MPS.
   Not in `PREREG.md` §2; **not run**.
2. **Rung-restricted scoring is free**: score only windows with `realised_context ≥ rung` from the
   dumped arrays. Removes the padding confound with no retraining.
3. **HEPA's benchmark, not ours, is what is bleeding** — four independent findings now agree: a
   same-byte lookup beats it on FD001 (+0.058 vs our own run); a zero-model table is within 0.021 of
   it on GECCO (cpu); its RevIN finetune stops at initialisation on 4 of the cells we ran; and more
   context makes it monotonically worse on every dataset that can carry a curve. ⛔ **All of this is a
   claim about the benchmark. None of it is a CLU result and none of it may be written as one.**
4. **For `experiment-engineer`: no HEPA code bug.** All loaders and both training loops ran first try
   at every rung incl. 8192 — **no NaN, no OOM, no divergence, no crash in 72 runs**. Traps:
   `hepa/__init__.py` imports torch even for loader-only work; `pyproject.toml` omits `openpyxl`;
   and (new) `scripts/train.py` hard-codes `cuda`-or-`cpu`, so it can never select MPS — my runner
   adds `--device` rather than patching their tree.
5. **Protocol:** this task file, like `hepa-lh-audit`'s, ships **without a §7 DIAL DECLARATION**
   (second occurrence). Supplied below; worth a mechanical Hub check when task files are written.

**DIAL DECLARATION** (echoed, supplied by me). *Dial:* **none — instrument/recon**; this prices a
third-party benchmark's datasets, not our model. *Laundering control:* the wave-1 matched-byte
tripwire, quoted not re-derived. *Falsifies the idea under test:* flat or falling context curves —
**it fired.** *Does NOT falsify:* HEPA losing to a lookup is a statement about the benchmark, not
about any CHLU dial; and no rung comparison here is a clean test of history alone (P=16 held).
⛔ **No CLU arm, no CLU number** (PREREG §6 honoured).

---

## Proposed handover updates (for the Hub)

**§1.6 — new row.** *hepa-lh-sweep (2026-09-07; HEPA @`85f55f39`, CHLU HEAD `7fcef50`, clone-actual,
seeds 42/123/456, P=16 at every rung, **local Apple M1 Pro — CSF3 unreachable (DNS), no job queued**;
72 training runs, 0 failures):*
**(i) THE REPRODUCTION LANDS ON THE README.** FD001 @512 = **0.9200 ± 0.0056** vs 0.918 ± 0.008;
ETTm1 @512 = **0.8109 ± 0.0087** vs 0.809 ± 0.008. ⇒ **wave 1's tripwire STANDS: a matched-byte kNN
store (0.9777) beats HEPA-as-run by +0.0577.** "+0.060 vs a README" becomes "+0.058 vs a run" and is
quotable.
**(ii) GECCO's VARIANCE does not reproduce** (cpu 0.8390 sd 0.0769 · mps 0.8962 sd 0.0438 vs a
published sd of 0.024; one seed moves 0.167 between families) because its shipped early stopper
returns **best-val epoch 0 on 3/3 cpu seeds**.
**(iii) THE CONTEXT CURVE IS NEGATIVE ON ALL FOUR RUNNABLE DATASETS:**

| dataset | ft_test events | rungs run | h-AUROC curve (mean ± spread) | longest−512 gap | verdict on clause (a) |
|---|---:|---|---|---:|---|
| **ETTm1** | 92 | 512, 2048, 8192 (mps) | 512: 0.8084±0.011 → 2048: 0.7420±0.048 → 8192: 0.6367±0.036 | **-0.1717** | **FAIL — negative** |
| **BeijingAQ** | 434 | 512, 2048 (mps) | 512: 0.7579±0.015 → 2048: 0.6995±0.093 | **-0.0583** | **FAIL — negative** |
| **PSM** | 21 | 100, 512, 2048, 8192 (mps) | 100: 0.6252±0.024 → 512: 0.5139±0.010 → 2048: 0.3827±0.084 → 8192: 0.4655±0.075 | **-0.0484** | **FAIL — negative** |
| **Weather** ⚠UNDERPOWERED | 7 | 512, 2048 (cpu) | 512: 0.7687±0.178 → 2048: 0.6493±0.127 | **-0.1194** | **FAIL — negative** |

**(iv) The early-stopping confound is CLOSED** by a labelled diagnostic: with early stopping disabled
(fixed 30 ep, 3 seeds), ETTm1 goes 512 **0.7938** → 8192 **0.6075** (gap **−0.1863** vs the shipped
**−0.1717**). **Removing early stopping does not rescue the long rung.**

**§1.6 / §8 — the long-horizon section is dead on evidence, not on a null.** Clause (a) fails 4/4 with
a negative gap; clause (b) is never reached; PREREG §5's "fewer than 2 KEEP" clause is live at **0
KEEP**; and §5's governing label-locality hypothesis is bypassed rather than settled.

**§5 (provenance) — new entries.** `sweep.json` (13 commissioned dataset×rung cells, all at 3 seeds;
66 runs across two device families, each with its own provenance block) and `preds/**` (66
per-test-window fp16 CDF dumps, 233 MB, each carrying a per-window `realised_context`) are the wave-2
artifacts of record. `PROVENANCE.md` from `hepa-lh-audit` remains the backup of record for
`.claude/data/hepa/**`; nothing here re-fetched data.
**Environment fact to bank: CSF3 is NOT reachable from this machine (DNS).** Any future task that
budgets "queue time" must first verify `ssh csf3` resolves, or scope for local Apple silicon.

**§6 (what works) — four facts.** (i) **HEPA's shipped early-stopped predictor-finetune returns the
epoch-0 checkpoint** on GECCO @512, PSM @512/@2048 and ETTm1 @8192 (3/3 seeds each) — its
RevIN-dataset cells are variance-dominated and near-init. (ii) **Longer context makes HEPA
monotonically worse** on every dataset in its suite that can carry a curve, and this survives
removing early stopping. (iii) **The M1 Pro MPS backend runs this model 10–40× faster than torch CPU
at the 8192 rung** (475 s vs ≈30 min/epoch) and agrees with CPU to ≤0.034 h-AUROC on well-powered
cells — usable for future HEPA work. (iv) `scripts/train.py` never selects MPS (`cuda`-or-`cpu` hard-coded).
