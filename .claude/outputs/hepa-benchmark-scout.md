# hepa-benchmark-scout — web-scout report

Task + acceptance criterion: Extract everything needed to price CLU-on-HEPA-encoder experiments across HEPA's full benchmark suite — per-dataset task/metric/protocol/baselines/labels/licence, plus encoder architecture, predictor I/O, training cost, and what is public.
Status: **done**, with three named NOT-FOUND items (§9).

**Dial:** none — instrument/recon. **Laundering control:** n/a (no performance number produced). **Falsifies:** n/a.

> ⚠ **RECONCILIATION LIST — needs an owner (first 10 lines, per protocol §5).**
> 1. **The HEPA code IS public and resolves the 0.918-vs-0.81 blocker.** `github.com/Forgis-Labs/HEPA` (CC BY-NC-SA 4.0) is live. `REPRODUCIBILITY.md` states the repo's C-MAPSS numbers "are higher than the first paper version, which reported an early-stopped operating point." **Both prior scout reports treated this as unresolved and recommended asking the Head. It is now answered from primary source.**
> 2. **The target encoder is NOT weight-shared.** `hepa/model/hepa.py` instantiates a *separate* `TargetEncoder` module, clone-initialised and jointly trained. The paper text ("the same encoder f_θ, applied bidirectionally") and `PROTOCOL.md` ("weight-shared") are both misleading. This is what makes the 2.16M figure close (§7.4) and it invalidates the prior scout's parameter narrative.
> 3. **There is no N-CMAPSS, no MSR, and no traffic dataset anywhere in HEPA.** The task file's expectation is wrong; the suite is 14 benchmarks over 11 domains, enumerated in §3.
> 4. **Appendices L (Preprocessing) and M (Hyperparameters) DO exist** (headings confirmed from the arXiv HTML table of contents) but their bodies are unreachable by every renderer available here. The prior scout's caution stands; the repo now substitutes for them.
> 5. **Version label correction:** the paper has **four** versions; v3 = 25 May 2026, **v4 = 3 Jun 2026**. Both prior reports cite "v3, 3 Jun 2026" — that date is v4. All paper citations below are **arXiv:2605.11130** as rendered on ar5iv (which serves the latest version).

---

## 1 · Answer first

HEPA's suite is **14 benchmarks / 11 domains**, all scored by a single metric, **h-AUROC** (mean of per-horizon AUROC over non-degenerate horizons), on a dense unit-step horizon grid that is a **conditioning input** to the model (K=150 for C-MAPSS/TEP, K=200 elsewhere). All four baselines were **re-run by the authors**, not reprinted, under a matched-capacity protocol in which the baselines actually get a *larger* downstream head (264K) than HEPA (198K). Ten of fourteen wins are real but thin: **only three datasets have a margin ≥0.05** (C-MAPSS-3 +.05, C-MAPSS-4 +.06, GECCO +.07); five of the ten are +0.01 or exact ties. The whole suite is pretrainable in **under two hours on one A10G** (5 seeds × 14 datasets), so the compute price of a CLU-encoder swap is negligible — the real cost is data acquisition, since HEPA redistributes nothing and three loaders (SMAP, TEP, VIX) ship an event definition its own authors call provisional.

---

## 2 · Prior-report discrepancies (explicit, per hard rule 4)

| Claim in prior report | Status now | Evidence |
|---|---|---|
| `scout-cafe-hepa`: 0.918 vs .81 "UNRESOLVED, ask the Head" | **RESOLVED** | `REPRODUCIBILITY.md` §"Note on the C-MAPSS numbers": paper = early-stopped operating point; repo = fixed finetune epoch budget |
| `scout-hepa-predictor`: "HEPA's optimizer/lr/epochs are UNKNOWN to us" | **RESOLVED (from repo, not paper)** | AdamW, lr 3e-4 pre / 1e-3 ft, wd 1e-2, batch 64 — §8 |
| `scout-hepa-predictor`: "no EMA, no separate target network — the same encoder f_θ, two read modes" | **WRONG** | `hepa.py`: separate `TargetEncoder` instance, clone-init, `requires_grad=True` under `joint_train`. Two *encoders*, not two read modes |
| `scout-hepa-predictor`: activations "not stated … engineer must pick" | **RESOLVED: GELU** | `predictor.py` and `_TransformerBlock` both use `nn.GELU()` |
| `scout-hepa-predictor`: "Open: whether Δt is normalized before concat" | **RESOLVED: NOT normalised** | `predictor.py` docstring: "`delta_t`: (B,) float horizon values **in native time units**" |
| `scout-hepa-predictor`: "L2 normalisation … SINGLE-SOURCED" | **CONFIRMED** | App. E table row: "Loss \| L1 on L2-normalised"; `PROTOCOL.md`: `(1-α)·L1(normalize(ĥ), normalize(h*))` |
| `scout-hepa-predictor`: "198K head … all methods share" | **WRONG for baselines** | App. C: baselines get a *different* 264K head with a **learned horizon embedding**, not scalar concat |
| `scout-cafe-hepa`: "HEPA does not use CoxPH; margin thin (FD001 .81 vs PatchTST .80)" | **CONFIRMED** | Table 1 |
| `scout-hepa-predictor`: "C-MAPSS uses full engine history" | **CONTRADICTED BY CODE** | `utils/config.py` sets FD001–4 context = **512**; `config_snapshot.py` comments "capped by max_context" |
| Both: `Forgis-Labs/HEPA-SP` 404 ⇒ code unavailable | **SUPERSEDED** | `Forgis-Labs/HEPA` is public; abstract page links it |

---

## 3 · Master dataset table (paper Table 3 + repo loaders)

Paper: arXiv:2605.11130, **App. D, Table 3** — *"Dataset overview (14 datasets, 11 domains). Default: patch size P=16, sliding window of 512 steps (32 tokens), except C-MAPSS which uses full engine history (8–23 tokens per cycle)."*

| # | Dataset | Domain | Target event (Table 3) | Sensors (Table 3) | Rate (Table 3) | K | Context (`utils/config.py`) |
|---|---|---|---|---|---|---|---|
| 1–4 | C-MAPSS FD001–FD004 | Turbofan | Engine failure | 14 | 1/cycle | **150** | 512 |
| 5 | SMAP | Spacecraft | Spacecraft fault | 25 | 1 Hz | 200 | 100 |
| 6 | PSM | Server | Server fault | 25 | 1/min | 200 | 100 |
| 7 | MBA (ECG) | Cardiac | Cardiac arrhythmia | 2 | 275 Hz | 200 | 100 |
| 8 | GECCO | Water quality | Water contamination | 9 | 1/min | 200 | 512 |
| 9 | BATADAL | ICS / water distribution | Cyber-attack on SCADA | 43 | 1/hour | 200 | 512 |
| 10 | TEP | Chemical | Process fault | 52 | 1/3 min | **150** | 512 |
| 11 | ETTm1 | Power | Transformer overheating | 7 | 15/min | 200 | 512 |
| 12 | Weather | Climate | Heat spike | 21 | 10/min | 200 | 512 |
| 13 | Beijing-AQ | Air | PM2.5 spike | 11 | 1/hour | 200 | 512 |
| 14 | VIX | Finance | Volatility regime | 6 | 1/day | 200 | 512 |

**NOT IN PAPER for every dataset:** number of series/entities, total length, licence, data-source URL, split ratios. (App. D contains *only* the four columns above — re-asked under an explicit "say if absent" guard.) Sources below come from `scripts/download_data.py`; splits/labels from the loader modules.

---

## 4 · Per-dataset detail

Headline metric is **h-AUROC for all 14**. "Margin" = HEPA − best baseline, paper Table 1, 100 % labels.

### C-MAPSS FD001 / FD002 / FD003 / FD004 (Turbo.)
| | |
|---|---|
| Task / target | Event prediction; **event = last cycle of a run-to-failure engine**, `labels[-1]=1`, time-to-event ≡ RUL |
| Horizon | **Conditioning input**, Δt ∈ {1…150} dense unit steps (App. B) |
| Label construction ⭐ | `hepa/data/cmapss.py`: reads `train_/test_/RUL_FD00X.txt` but **`test_df` and `rul` are discarded** (`train_df, _test_df, _rul = _load_raw(subset)`). Docstring: *"the official `test_FDxxx.txt` engines are truncated before failure, so labelling their last observed cycle as the event is incorrect and yields chance-level h-AUROC. We therefore split the TRAIN engines by id"* |
| Split | Engine-ID permutation, `seed=42`, `split_ratios=(0.6, 0.1, 0.3)`; engines with `len(seq) < 16` dropped |
| Channels | 14, `SELECTED_SENSORS=[2,3,4,7,8,9,11,12,13,14,15,17,20,21]` (drops s1,5,6,10,16,18,19) |
| Normalisation | **`norm_mode="none"` — RevIN OFF** (`utils/config.py::NORM_POLICY`). Rationale (`REPRODUCIBILITY.md`): *"remaining-useful-life lives in the absolute drift level of the sensors; per-window RevIN would normalize that level away (h-AUROC → chance)"* |
| Point-adjust | None. A secondary **RMSE↓** column is reported for comparability |
| Source / licence | NASA Prognostics Data Repository; licence **NOT STATED** |
| HEPA result & margin | FD001 **.81±.03** vs .80 PatchTST → **+.01**; FD002 **.57±.01** vs .56 MAE → **+.01**; FD003 **.84±.01** vs .79 PatchTST → **+.05**; FD004 **.63±.02** vs .57 MAE → **+.06**. Wins all four |
| Secondary RMSE (Table 1) | HEPA 28.5 / 40.8 / 34.7 / — vs published SOTA 12.2 / 20.0 / 12.7 / — (ref [2]). **HEPA loses RMSE badly on all three** |
| Repo-vs-paper | `REPRODUCIBILITY.md` "expected" values: FD001 0.918±.008, FD002 0.661±.007, FD003 0.960±.003, FD004 0.627±.008 |

⚠ **Contradiction:** Table 2 reports "C-MAPSS-1 (85 eng.)" and "C-MAPSS-3 (100 eng.)" as *training* engine counts. A 0.6 split of the 100 FD001 train engines gives 60, not 85. **Unresolved — the released loader cannot produce 85.**

### SMAP (Spacecraft)
Task: anomaly-precursor event prediction. Label: `test_label.npy` per-timestep binary; train sequences unlabeled. Split: `chronological_split(test, labels)` with a leakage gap. Norm: z-score on train. 25 channels, 55 entities. Source: `github.com/khundman/telemanom`. Licence NOT STATED.
**HEPA .59±.05 vs MAE .64 → −.05 LOSS.** Secondary PA-F1: HEPA .94 vs SOTA .96 [3].
⚠ `REPRODUCIBILITY.md`: *"The SMAP, TEP, and VIX loaders use a placeholder event definition and are being aligned to the benchmark protocol."*

### PSM (Server)
Label: `test_label.csv` last column / `test_labels.npy`. 25 channels after dropping near-constant. `chronological_split`, z-score on train. Source: `github.com/eBay/RANSynCoders`. Licence NOT STATED.
**HEPA .57±.02 vs MAE .56 → +.01 WIN.** PA-F1 .94 vs SOTA .98 [3].

### MBA / MIT-BIH ECG (Cardiac)
2 channels (MLII + V5), single continuous recording. Label: annotation indices dilated by ±`label_window`: `for offset in range(-label_window, label_window+1): labels[idx]=1`. Norm: **min-max** (not z-score). Source: `github.com/imperial-qore/TranAD`. Licence NOT STATED.
**HEPA .75±.03 vs iTransformer .84 → −.09 LOSS** (largest loss). Text: *"iTransformer's variate-attention mechanism excels on MBA (h-AUROC 0.84 vs. HEPA's 0.75)."* Secondary F1: HEPA .98 vs LSTM-AE .77.

### GECCO 2018 (Drinking water) — "water flow"
9 sensors `[Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2]`, ~122 K timesteps. Label: `y = df["EVENT"]` — **a supplied ground-truth column, no thresholding**. Split `(0.5, 0.25, 0.25)` chronological with `gap=200`. Norm: z-score fit on `x[:t1]`. Source: `spotseven.de/gecco-challenge/gecco-challenge-2018/`. Licence NOT STATED.
**HEPA .88±.06 vs MAE .81 → +.07 WIN** (largest margin in the suite). ⚠ At 10 % labels HEPA **collapses to .39±.13**, the worst cell in Table 1 — Chronos-2 wins there at .80.

### BATADAL (ICS / water distribution)
43 SCADA channels. Label: `_LABEL = "ATT_FLAG"`, with `y_te = np.where(y_te == 1, 1, 0)` — **`-999` (unknown) is mapped to 0, i.e. unknown treated as negative**. Split by `t1/t2` fractions with a `gap`. Norm: z-score using pre-attack statistics. Files: `BATADAL_dataset03.csv` (train) / `dataset04.csv` (test). Source: `batadal.net/data.html`. Licence NOT STATED.
**HEPA .57±.03 vs PatchTST .66 → −.09 LOSS.** Secondary F1: HEPA .77 vs AEED .74. Paper §5.2: *"BATADAL and MBA have sensor-localised events where channel-fusion tokenisation dilutes the relevant subset."*

### TEP (Chemical)
52 process variables, 21 fault scenarios + 1 nominal. Label: `labels = (test_df["faultNumber"].values > 0)`. ⚠ Docstring: *"The benchmark marks a single fault-onset event per simulation run; this loader's per-timestep labeling differs and is being aligned to that protocol."* K=150. Source: Harvard Dataverse `doi:10.7910/DVN/6C3JR1`. Licence NOT STATED (Dataverse release).
**HEPA 1.00±.00 vs PatchTST .99 → +.01 WIN** (saturated). Secondary F1 .95 vs XGBoost .93.

### ETTm1 (Power)
7 channels `[HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]`, 15-min. Label: **derived** — `delta = (ot - local_mean)/max(local_std,1e-3); y = (delta > threshold_sigma)` with **threshold = 2σ** on the OT (oil-temperature) channel. Split 60/20/20 chronological. Source: `github.com/zhouhaoyi/ETDataset`. Licence NOT STATED.
**HEPA .81±.00 vs MAE .87 → −.06 LOSS.** ⚠ App. G Table 5 lists HEPA ETTm1 at **0.87±.00** under the matched-head protocol — a different protocol, do not cross-quote.

### Weather / Jena (Climate)
21 channels, 10-min. Label: derived heat spike — `y_t = 1 iff (T_t − rolling_mean)/rolling_std > 3` (**3σ**, vs ETTm1's 2σ). Split 70/15/15 chronological. Source: `bgc-jena.mpg.de/wetter/`. Licence NOT STATED.
**HEPA .89±.01 vs MAE .88 → +.01 WIN.**

### Beijing-AQ (Air)
Label: `labels = (pm25 > _SEVERE)` with **`_SEVERE = 250.0` µg/m³** — an absolute, not relative, threshold. Split 70/15/15 with gap. 12 stations, hourly 2013-03-01…2017-02-28. Source: UCI `archive.ics.uci.edu`. Licence NOT STATED (UCI ≈ CC BY 4.0, **unverified**).
**HEPA .81±.01, tied with PatchTST† (.81) and MAE (.81); paper bolds all three → counted a WIN, margin 0.00.** The `†` marker on PatchTST is unexplained in the retrieved caption.
⚠ **Channel-count conflict:** Table 3 says **11** sensors; the loader says `"n_channels": len(_NUMERIC) + 2, # 13` (11 numeric + wind direction as sin/cos).

### VIX (Finance)
6 channels (VIX OHLC + S&P500 close + date-derived). Label: `threshold = np.percentile(test[:, close_idx], spike_percentile); labels = (test[:, close_idx] > threshold)` — **90th-percentile VIX-close spike days, threshold fit on the TEST split**. Split 60/40 then `chronological_split`. Norm: z-score on train. Source: CBOE / Yahoo Finance. Licence NOT STATED.
**HEPA .57±.01, tied with MAE .57 → WIN by bolding, margin 0.00.** Also on the "placeholder event definition" list.

---

## 5 · Baseline rows — name, number, citation, rerun-vs-reprint

**All four baselines were RE-RUN by the authors.** App. C, verbatim: *"The methods differ only in how the frozen encoder is obtained"* — PatchTST *"trained end-to-end (supervised, no self-supervised pretraining) on each dataset. Same patch size (P=16) and context length (512 steps) as HEPA. **2.26M trained parameters**… 5 seeds"*; iTransformer *"Trained end-to-end (supervised) on each dataset with the same context window and horizon set… 5 seeds"*; MAE *"uses the same causal Transformer encoder architecture as HEPA but pretrained with a masked reconstruction objective… 5 seeds"*; Chronos-2 *"foundation model (**119M parameters**) pretrained on a large external time-series corpus… 3 seeds."*

**The only REPRINTED numbers** are the right-hand "SOTA" column of Table 1 (domain-specific metric + `Ref`): C-MAPSS RMSE 12.2/20.0/12.7 [2]; SMAP & PSM PA-F1 .96/.98 [3]; MBA F1 .77 (LSTM-AE); BATADAL F1 .74 (AEED); TEP F1 .93 (XGBoost). ⚠ **References [2] and [3] resolve to bibliography entries I could not retrieve — NOT-FOUND.**

### Table 1 (paper, ar5iv), h-AUROC, 100 % labels — full baseline grid
Caption verbatim: *"Main results (mean ± std; 5 seeds for HEPA, PatchTST, iTransformer, MAE; 3 seeds for Chronos-2). All methods use matched-capacity downstream heads on frozen encoders (appendix C). Each dataset has two rows: 100% labels and 10% labels (gray). Bold = best mean per row."*

| Dataset | Chr-2 | PatchTST | iTransf | MAE | **HEPA** | Margin |
|---|---|---|---|---|---|---|
| C-MAPSS-1 | .66±.00 | .80±.04 | .70±.05 | .69±.02 | **.81±.03** | +.01 |
| C-MAPSS-2 | .45±.01 | .44±.03 | .43±.03 | .56±.01 | **.57±.01** | +.01 |
| C-MAPSS-3 | .73±.00 | .79±.01 | .76±.01 | .78±.02 | **.84±.01** | +.05 |
| C-MAPSS-4 | — | .52±.03 | .45±.02 | .57±.02 | **.63±.02** | +.06 |
| SMAP | .54±.02 | .49±.03 | .47±.05 | **.64±.04** | .59±.05 | −.05 |
| PSM | .48±.01 | .55±.02 | .52±.03 | .56±.02 | **.57±.02** | +.01 |
| MBA | .53±.10 | .68±.07 | **.84±.03** | .73±.03 | .75±.03 | −.09 |
| BATADAL | .56±.01 | **.66±.03** | .46±.13 | .60±.04 | .57±.03 | −.09 |
| TEP | — | .99±.02 | .93±.02 | .96±.02 | **1.00±.00** | +.01 |
| ETTm1 | .74±.01 | .53±.03 | .79±.03 | **.87±.00** | .81±.00 | −.06 |
| Weather | .72±.02 | .64±.08 | .83±.04 | .88±.02 | **.89±.01** | +.01 |
| Beijing-AQ | .53±.00 | **.81±.01**† | .75±.02 | **.81±.01** | **.81±.01** | 0.00 |
| VIX | .40±.04 | .48±.11 | .40±.14 | **.57±.03** | **.57±.01** | 0.00 |
| GECCO | .74±.02 | .65±.07 | .64±.15 | .81±.04 | **.88±.06** | +.07 |
| **Best (100 %)** | 0 | 2 | 1 | 4 | **10** | |

10 % labels (gray rows) reproduced in the source; §5.4: *"At 10% labels across all 14 datasets, HEPA wins on 6 out of 14, compared to 10 out of 14 at full labels."*

⚠ **Repo/paper conflict:** `papers/hepa-2026/results_table1.csv` disagrees with the paper on ≥3 cells — MBA/iTransf **0.65** (paper .84), ETTm1/HEPA **0.87** (paper .81), PSM/Chr-2 **0.55** (paper .48). The paper is self-consistent (§5.2 text quotes 0.84 for MBA-iTransformer; the 4 losses ⇒ 10 wins arithmetic only closes with the paper's values). **Treat the CSV as stale.** Verified by two independent guarded fetches.

### Table 5 (App. G) and Table 6 (App. H)
Both were fully extracted by `scout-hepa-predictor` §5 and I re-confirm the captions and protocol sentence; I did not re-extract every cell. **Do not merge Tables 1/5/6 — three different protocols** (Table 5 = matched-264K-vs-198K encoder-swap incl. foundation models; Table 6 = SSL-objective-only vs MTS-JEPA/SC-JEPA).

### Point-adjust / scoring
- **h-AUROC uses no point adjustment.** `hepa/evaluation/metrics.py::h_auroc` = `np.mean` of per-horizon `roc_auc_score`, skipping horizons where `n_pos == 0 or n_pos == n_total`.
- ⚠ **Code/paper conflict:** App. B and `PROTOCOL.md` both say degenerate horizons are those with *"prevalence < 0.001 or > 0.999"*; the released code skips only **exactly** degenerate horizons (0 or 1). Different metric.
- **PA-F1 is point-adjusted and the paper says so.** §4 verbatim: *"PA-F1 thresholds p(t,1) at the smallest horizon and credits entire anomaly segments from a single detection (**inflated**)."* Appendix **K** is titled *"PA-F1 Inflation"* (body unreachable). **No point-adjust function exists in `metrics.py`** — PA-F1 is reported for comparability only, computed elsewhere. NOT-FOUND: its implementation.
- Also implemented: pooled AUPRC/AUROC/best-F1, ECE (10 equal-width bins), Brier, `monotonicity_violation_rate` (fraction of `p[:,k+1]-p[:,k] < -1e-6`).
- §4 verbatim on why h-AUROC: per-horizon prevalence ranges *"0.5% at Δt=1 and 96% at Δt=150"*, so pooled AUPRC is dominated by prevalence.

---

## 6 · Horizon: is it a conditioning input?

**Yes, unambiguously.** `HorizonPredictor.forward(h, delta_t)` concatenates Δt as **one raw scalar in native time units** to the 256-d embedding → `Linear(257, 256)`. App. B verbatim: *"All methods (HEPA, PatchTST, iTransformer, MAE, Chronos-2) use dense unit-step horizons: K=150 for C-MAPSS and TEP (Δt∈{1,2,…,150}), and K=200 for all other datasets (Δt∈{1,2,…,200}). All methods share the same horizon set per dataset."*
Pretraining **samples** Δt ~ LogUniform[1, K] (App. E: `Horizon sampling | LogU[1,150]`); finetuning/eval **enumerates** all K.
⚠ Baselines condition on Δt differently: App. C gives them *"a learned horizon embedding (256 entries × 256 dims)"* — an embedding table, not a scalar. **The horizon-conditioning mechanism is not matched across arms.**

---

## 7 · The model

### 7.1 Encoder, layer by layer (`hepa/model/encoder.py`, verbatim source)
Input `(B, T, C)` →
1. **`RevIN`** — reversible per-context per-channel z-score, `eps=1e-5`, mask-aware. **Disabled (`norm_mode="none"`) for C-MAPSS only.**
2. **`PatchEmbedding`** — right-pads T to a multiple of P=16, reshapes to `(B, T/16, C·16)`, then `nn.Linear(n_channels*16, 256)`. For C-MAPSS: `Linear(224, 256)`.
3. **`+ sinusoidal_pe(arange(N), 256)`** — additive, non-learned.
4. **`_TransformerBlock` ×2** — pre-norm; `nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)` with `attn_mask = generate_square_subsequent_mask(N)` (**causal**) and `key_padding_mask`; FFN = `Linear(256,256) → GELU → Dropout(0.1) → Linear(256,256) → Dropout(0.1)`. **`d_ff = 256`, i.e. no FFN expansion** (not 4×d).
5. **`nn.LayerNorm(256)`**
6. **Pool = last valid token**: `h[:, last_idx]` (or `h[:, -1]` if unmasked).

**Shape/channel count of the last three layers before the flat embedding** (C-MAPSS, 512-step context ⇒ N=32 tokens):

| Layer | Input shape | Output shape | Channels |
|---|---|---|---|
| `_TransformerBlock[0]` | (B, 32, 256) | (B, 32, 256) | 256 |
| `_TransformerBlock[1]` | (B, 32, 256) | (B, 32, 256) | 256 |
| `nn.LayerNorm(256)` | (B, 32, 256) | (B, 32, 256) | 256 |
| last-token select | (B, 32, 256) | **(B, 256)** | 256 |

**Flat embedding width D = d_model = 256.** No projection or normalisation after it.

### 7.2 Target encoder (`target_encoder.py`)
Same stack **without** the causal mask, plus **attention pooling**: `pool_query = nn.Parameter(randn(1,1,256)*0.02)` and a second `nn.MultiheadAttention(256, 4, dropout=0.0)`; returns `pooled.squeeze(1)`.
**Target-encoder embedding width = 256.** ⭐ It is a **separate module** (`hepa.py` instantiates `TargetEncoder(...)`, then `_init_target_encoder()` clones matching-shape encoder weights once, then `p.requires_grad = (target_mode == "joint_train")`). **Weights are cloned at init, then diverge.**

### 7.3 Predictor + event head — confirmed against `.claude/HEPA_predictor.py`
`hepa/model/predictor.py` is **byte-identical** to the local file (same docstring, same `nn.Sequential`, same `dt = delta_t.float().unsqueeze(-1)`).
`Linear(d+1→256) → GELU → Linear(256→256) → GELU → Linear(256→d)`; **in = `[h_t (256); Δt (1)] = 257`, out = 256.**
`event_head.py`: `LayerNorm(256) → Linear(256, 1) → squeeze(-1)`; docstring: *"Accepts (B, d) → (B,) or (B, K, d) → (B, K)"* — **it can be applied to all K horizons in one batched call**, which the prior scout inferred as a per-horizon loop. Both work; batched is what the code supports.
Survival CDF `p(t,Δt) = 1 − ∏_{j≤Δt}(1−λ_j)` and `ℒ_FT = Σ w⁺·BCE`, `w⁺ = N_neg/N_pos` *"(auto-estimated)"* per `PROTOCOL.md`. Whether `w⁺` is global or per-horizon: **still NOT-FOUND**.
Label surface (`evaluation/surface.py`): `build_label_surface` → `(B, T, K)`, *"y(t, Delta_t_k) = 1 iff the next event arrives within Delta_t_k steps"*, i.e. `time_to_event <= horizon` and finite.

### 7.4 ⭐ Parameter count reconciles exactly at 2.16M — and only if the target encoder is separate
My arithmetic from the released source, `n_channels=14`:

| Module | Params |
|---|---|
| `patch_embed` `Linear(224→256)` | 57,600 |
| `_TransformerBlock` ×2 (attn 197,376 + out 65,792 + FFN 131,584 + 2×LN 1,024 = 395,776 each) | 791,552 |
| final `LayerNorm` | 512 |
| **CausalEncoder** | **849,664** |
| TargetEncoder body (identical stack) | 849,664 |
| `pool_query` + `pool_attn` (256 + 197,376 + 65,792) | 263,424 |
| **TargetEncoder** | **1,113,088** |
| `HorizonPredictor` (66,048 + 65,792 + 65,792) | **197,632** ("197.6K" ✓) |
| `EventHead` (512 + 257) | **769** ✓ |
| **TOTAL** | **2,161,153 = 2.16M ✓ exact** |

⇒ The paper's *"2.16 M-parameter architecture"* **counts a full second encoder**. A CLU replacement matched at 2.16M must decide whether to match the *context* encoder (850K) or the whole pretraining apparatus. **This is the single most consequential number for pricing a matched-capacity claim, and neither prior report has it.**

---

## 8 · Training cost, hardware, epochs, seeds

**Paper (§5.1, verbatim):** *"Pretraining takes under one minute per dataset on a single A10G GPU, with the full 14-dataset, 5-seed sweep completing in under two hours."* · *"All numbers are reported as mean ±std across 5 seeds (HEPA, PatchTST, iTransformer, MAE) or 3 seeds (Chronos-2)."* **Optimiser, lr, batch, epochs are ABSENT from the retrievable paper body** (they live in Appendix M / Table 7, unreachable).

**⚠ The repo ships TWO mutually inconsistent protocol files. Do not average them; pick one and cite it.**

| | `papers/hepa-2026/config_snapshot.py` + `PROTOCOL.md` (claims "matches paper Table 7 / Section M exactly") | `hepa/utils/config.py` + `REPRODUCIBILITY.md` (what a clone actually runs) |
|---|---|---|
| Pretrain epochs | **100** (patience 10) | **50** (patience 10) |
| Finetune epochs | **50** (patience 10) | **30** (patience 10) |
| Seeds | **{0,1,2,3,4}** | **{42,123,456,789,1337}** |
| C-MAPSS context | "full engine history (cycle-as-patch)" in `PROTOCOL.md`; **512** in the dict | **512** |
| C-MAPSS norm | no norm policy defined | **`none` (RevIN off)**; RevIN elsewhere |
| C-MAPSS h-AUROC | .81 / .57 / .84 / .63 (paper) | 0.918 / 0.661 / 0.960 / 0.627 |

Shared by both: AdamW; **lr 3e-4 pretrain, 1e-3 finetune; wd 1e-2; batch 64; α=0.1; `n_cuts=40`; `dropout=0.1`; `target_mode="joint_train"`; `max_context=512`.**
`REPRODUCIBILITY.md` also specifies per-dataset finetune stopping: **C-MAPSS = fixed epochs (deterministic); all others = val-loss early stopping**, *"because the encoder separates RUL almost perfectly, so val-loss early stopping is sensitive to the stop epoch."*
Env: *"Python ≥ 3.10, PyTorch ≥ 2.0, a single GPU (an A10G suffices; < a few minutes per dataset/seed)."*

---

## 9 · What is public

| Artifact | Status |
|---|---|
| **Code** | ✅ `github.com/Forgis-Labs/HEPA` — full training/eval/data stack (48 files; `hepa/{data,model,training,evaluation,utils}`, `scripts/train.py`, `tests/`) |
| **Licence** | **CC BY-NC-SA 4.0** — ⚠ **non-commercial + share-alike**; note before vendoring any of it |
| **Frozen paper protocol** | ✅ `papers/hepa-2026/{PROTOCOL.md, config_snapshot.py, results_table1.csv, README.md}` |
| **Checkpoints** | ❌ `papers/hepa-2026/README.md` shows a `torch.load("checkpoints/FD001_s0.pt")` example but **"model weights download links remain marked as TODO"** |
| **Cached latents** | ❌ NOT-FOUND — no such artifact in the tree |
| **Data** | ❌ *"HEPA does not redistribute data."* `scripts/download_data.py` prints per-dataset sources only |
| **Technical report** | `artifacts/HEPA-small-technical-report.pdf` — present in tree, **not read** (no PDF renderer here) |
| **cafe-bench** | Still gated — outside this task's scope; unchanged from `scout-cafe-hepa` |
| **Paper appendices J–O** | Headings confirmed (J *Legacy Metrics*, K *PA-F1 Inflation*, L *Preprocessing Details*, M *Hyperparameters*, N *Pairwise Significance Tests*, O *Calibration Analysis*); **bodies unreachable** |

---

## 10 · Gaps and discrepancies

**NOT-FOUND (never inferred):**
1. Appendix L (Preprocessing Details), M (Hyperparameters incl. Table 7), J, K, N, O bodies — every renderer truncates (ar5iv dies in Appendix I.1 at *"With all we"*; `arxiv.org/html` dies mid-Appendix-C). No PDF renderer on this machine.
2. Per-dataset **licences, sizes, series counts, split ratios** — absent from App. D; only the loaders' hard-coded ratios recovered.
3. Bibliography entries **[2]** and **[3]** (the reprinted RMSE / PA-F1 SOTA sources).
4. PA-F1 implementation (not in `metrics.py`).
5. Whether `w⁺` is global or per-horizon.
6. Appendix I.1 result (*"Does the Predictor Help at Inference?"*); only App. C's forward reference to **I.2** was retrieved: pretrained-vs-random predictor init is worth *"at most +0.003 h-AUROC"*.
7. `n_cuts=40` semantics (presumed evaluation cut points t per sequence) — **not documented anywhere I could reach.**

**Discrepancies to resolve before any number is quoted:**
- **D1** Table 2's "85 engines" for C-MAPSS-1 vs the loader's 0.6×100 = 60. Unexplained.
- **D2** `results_table1.csv` ≠ paper Table 1 in ≥3 cells (MBA/iTransf, ETTm1/HEPA, PSM/Chr-2). Paper is internally consistent; CSV is not.
- **D3** C-MAPSS context: Table 3 "full engine history (8–23 tokens per cycle)" vs `CONTEXT_BY_DATASET["FD001"] = 512` (=32 tokens).
- **D4** Degenerate-horizon threshold: paper/PROTOCOL 0.001/0.999 vs code exactly-0-or-1.
- **D5** Beijing-AQ channels: Table 3 says 11, loader says 13.
- **D6** Target encoder "weight-shared" (paper §3.1, App. E, PROTOCOL.md) vs separate clone-initialised jointly-trained module (`hepa.py`). The 2.16M arithmetic sides with the code.
- **D7** Two protocol files with different epochs/seeds (§8).
- **D8** Downstream head is **not** matched in mechanism: HEPA 198K scalar-concat vs baselines 264K learned-horizon-embedding.
- **D9** Three loaders (SMAP, TEP, VIX) carry an event definition the authors call placeholder/being-aligned, yet all three appear in Table 1.

**Confidence.** Verified by ≥2 independent fetches or by exact arithmetic: Table 1 (paper), h-AUROC definition and code, App. B horizons, App. C protocol + 264K/198K/2.26M/119M counts, encoder/target/predictor/head source, the 2.16M reconciliation, A10G cost sentence, repo licence and file tree, `predictor.py` ≡ local file. Single-sourced: per-dataset loader details (one guarded fetch each, targeted quotes — the code is unambiguous but I did not re-read whole files); App. F/Table 4; the ToC appendix titles J–O. Every appendix extraction used an explicit "say if absent" guard, and the guard fired correctly (Appendix L returned *"NOT IN RETRIEVED CONTENT"*) — no fabricated appendix content in this report.

---

## Bibtex-ready refs

```bibtex
@article{petersen2026hepa,
  title   = {HEPA: A Self-Supervised Horizon-Conditioned Event Predictive
             Architecture for Time Series},
  author  = {Petersen, Jonas and Lombardi, Gian-Alessandro and Maggioni, Riccardo and
             Mazzoleni, Camilla and Martelli, Federico and Petersen, Philipp},
  journal = {arXiv preprint arXiv:2605.11130},
  note    = {v1 11 May 2026; v2 13 May 2026; v3 25 May 2026; v4 3 Jun 2026.
             Spotlight, FMSD Workshop, ICML 2026.
             Code: https://github.com/Forgis-Labs/HEPA (CC BY-NC-SA 4.0)},
  year    = {2026}
}
```

---

## Proposed handover updates (for the Hub)

1. **Close the 0.918-vs-0.81 blocker** — cause is on record in `REPRODUCIBILITY.md` (early-stopped operating point in the paper vs fixed finetune epoch budget in the repo). Two waves' worth of "ask the Head" items can be struck.
2. **Record `github.com/Forgis-Labs/HEPA` as public, CC BY-NC-SA 4.0**, and retire the "HEPA code is 404" note. Flag the **NC + ShareAlike** terms to anyone planning to vendor code.
3. **Correct the architecture record: the target encoder is a separate, jointly-trained module, not a second read mode of one encoder.** Add the 2.16M reconciliation table (§7.4) — matched-capacity claims must state whether they match 850K (context encoder) or 2.16M (whole apparatus).
4. **Correct the matched-head record:** baselines get **264K** with a learned horizon embedding; HEPA gets **198K** with scalar-concat. "Only the encoder differs" is false at the conditioning mechanism.
5. **Record the C-MAPSS labelling answer:** train-engine-only, id-split 0.6/0.1/0.3 seed 42, official test set and `RUL_*.txt` discarded, `labels[-1]=1`. Add discrepancy **D1** (85 engines) as an open item.
6. **Add the margin profile:** only 3 of 14 datasets have a HEPA margin ≥0.05 (C-MAPSS-3, C-MAPSS-4, GECCO); 4 losses (SMAP, MBA, BATADAL, ETTm1); 2 exact ties counted as wins (Beijing-AQ, VIX).
7. **Update the task-scoping record: no N-CMAPSS, no MSR, no traffic dataset exists in HEPA.** Whoever scoped this task from memory should not re-seed that list.
8. **Cost line for planning:** full 14-dataset × 5-seed pretrain sweep < 2 h on one A10G; per dataset/seed "< a few minutes". Compute is not the constraint — **data acquisition and the D9 placeholder-label datasets are.**
