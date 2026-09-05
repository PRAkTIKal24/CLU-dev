# scout-cafe-hepa — web-scout report

Task + acceptance criterion: Map the CAFE benchmark, extract the `Model.encode()` registration contract verbatim, extract HEPA's method/params/mechanism, size the parametric peers, and give a positioning verdict for CLU's CAFE entry.
Status: **partial** — items 1–5 delivered, but **`cafe-bench` source was NOT obtainable** (no Bash tool on this agent; `github.com/forgislabs/cafe-bench` returns HTTP 404 unauthenticated). Everything about the harness comes from `.claude/reference/CAFE_README.md`; everything about HEPA comes from the public arXiv paper.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (read this first).**
> **The CAFE leaderboard's HEPA numbers do NOT match the published HEPA paper.** README: FD001 **0.918**, FD002 **0.661**, FD003 **0.960**, FD004 **0.627**. Paper (arXiv:2605.11130v3, Table 1, 100% labels): FD001 **.81±.03**, FD002 **.57±.01**, FD003 **.84±.01**, FD004 **.63±.02**. Only FD004 agrees. **The handover §2026-07-20 entry, `clu-cafe-integration.md`, and the ICLR headline-target ("beat 0.918") all currently cite the README number as if it were the paper's.** Until the discrepancy is explained, the engineer's target is ambiguous by ~0.11 h-AUROC on FD001. See §3.4.

---

## Answer first

CAFE is a three-task (Classification / Anomaly / Event-Prediction), 50+-dataset harness whose entire model-side contract is **one method: `encode(X: np.ndarray) -> np.ndarray`, mapping `(N, T, C) → (N, D)` frozen embeddings**, with default probes (LogisticRegression / kNN-distance / `lifelines.CoxPHFitter`) supplied by `BaseModel` and optionally overridable per task. HEPA — the model CLU "companions" — is a **2.16M-param 2-layer causal Transformer** (d=256, 4 heads, patch P=16) pretrained as a **horizon-conditioned JEPA** (predict future *representations*, not values, with SIGReg anti-collapse), then frozen while a predictor head is finetuned into a **monotonic discrete-time survival CDF over horizons**. The Hub's hypothesis is **confirmed**: Event-Prediction is CLU's most-winnable entry — h-AUROC is explicitly a *time-to-basin-exit* discrimination metric, C-MAPSS is fully public/auto-downloadable, and HEPA's own margin over baselines there is thin (FD001 .81 vs PatchTST .80). But the mechanism CLU must beat is *not* an energy landscape — it is **predictability of future latents**, and HEPA already ships the anti-collapse regularizer the Hub independently proposed this wave.

---

## 1 · The CAFE benchmark map

Source: `.claude/reference/CAFE_README.md` (Hub-pulled 2026-07-20 from private `Forgis-Labs/CAFE`). Pipeline: `raw data → Dataset.load() → Model.encode() → Evaluator → results/*.json → leaderboard.json`.

| Task | Primary metric | # datasets (README header) | Domains |
|---|---|---|---|
| Classification | **Macro-F1** ↑ | 36 | healthcare, wearable, speech, industrial, neuro |
| Anomaly Detection | **VUS-PR** ↑ | 16 | IT, ICS, healthcare, aerospace, climate |
| Event Prediction | **h-AUROC** ↑ | 6 | aerospace, healthcare |

**Count audit (flag for the engineer):** Anomaly 16 = 13 TSB-AD-M + NAB + Yahoo-S5 + KPI ✓. Event 6 = 4 C-MAPSS + 2 PhysioNet ✓. **Classification 36 does not reconcile**: the README lists UEA (30) + PTB-XL/UCI-HAR/Sleep-EDF/CWRU/TEP (5) = **35**, *plus* a separately-documented UCR archive of 128 univariate sets that the "36" evidently does not include. Confirm the registered-key count with `cafe-bench ls --task classification` before quoting "36" in the paper.

### Metric definitions
- **Macro-F1** — class-balanced multiclass F1.
- **VUS-PR** — Volume Under the (PR) Surface; integrates PR-AUC over anomaly buffer sizes. README: *"Robust to point-adjust gaming."* It is the TSB-AD standard metric; TSB-AD (Liu & Paparrizos, NeurIPS 2024 D&B) calls VUS-PR *"the most reliable and accurate measure"*, and the collection is 1070 series / 40 datasets (TSB-AD-M = the multivariate slice).
- **h-AUROC** — README: *"AUROC integrated across all prediction horizons Δt ∈ {1, …, H}"*. HEPA paper's definition is sharper and is the one to implement against: *"decompose the surface into independent per-horizon binary classification problems, each with a universal 0.5 baseline that does not depend on prevalence,"* then average AUROC across horizons. **Note the wording mismatch: README says "integrated", paper says "averaged" — likely the same thing up to a normalizing constant, but this is one candidate explanation for the 0.918-vs-.81 gap (§3.4).**

### Data accessibility (ICLR reproducibility gate)
| Tier | Datasets | Verdict |
|---|---|---|
| **Auto-download, no creds** (`scripts/download_all.py`) | NAB, UCI-HAR, PTB-XL, TSB-AD, KPI, Sleep-EDF, **C-MAPSS** | ✅ clean |
| **Auto via `aeon`** | UCR (128), UEA (30) | ✅ clean |
| **HuggingFace mirror** (`download_from_hf.py`) | CWRU, PhysioNet-2012, PhysioNet-2019 | ✅ clean-ish (mirror, not canonical) |
| **Free account required** | PTB-XL, PhysioNet-2012, PhysioNet-2019 (physionet.org credentialing) | ⚠ gated but free |
| **Licensed** | **Yahoo S5** (Webscope license) | ⛔ the only hard gate — **drop it from the ICLR table or mark N/A** |
| **Optional larger variant** | TEP full Rieth-2017 (Harvard Dataverse) | ⚠ note which variant you ran |

**Bottom line for reproducibility: the entire Event-Prediction task (C-MAPSS FD001–004 + PhysioNet) is freely reproducible.** That strengthens the case for leading there.

### Leaderboard state
**Every cell is `—` (unevaluated) except HEPA's Event-Prediction row.** Classification and Anomaly leaderboards are entirely empty; even HEPA has no numbers there. Two consequences: (a) CLU can be the *first* entry on Anomaly/Classification (cheap "first non-HEPA submission" framing, but no comparison to defend against), and (b) **there is no published MOMENT/UniTS CAFE number to compare against on any task** — the Hub's "real weight-class comparison" does not yet exist and would have to be *run by us*.

---

## 2 · The `Model.encode()` registration interface ⭐ (deliverable for the engineer)

Quoted **verbatim** from `CAFE_README.md` §"Adding Your Model":

```python
# my_model.py
import numpy as np
from cafe_bench.models.base import BaseModel
from cafe_bench.registry import register_model

class MyModel(BaseModel):
    name = "my_model"

    def encode(self, X: np.ndarray) -> np.ndarray:
        # X: (N, T, C)  →  return (N, D) embeddings
        ...

register_model("my_model", MyModel)
```

> *"The default probes are inherited:*
> - *Classification → `sklearn.LogisticRegression` on frozen embeddings*
> - *Anomaly → kNN distance in embedding space*
> - *Event → `lifelines.CoxPHFitter` on frozen embeddings"*

Optional per-task overrides, verbatim:

```python
class MyModel(BaseModel):
    name = "my_model"

    def encode(self, X):       ...   # required
    def classify(self, ...):   ...   # optional override
    def anomaly_score(self, ...): ... # optional override
    def event_predict(self, ...): ... # optional override
```

Invocation: `cafe-bench run uea_epilepsy my_model --data-root data/` · sweep: `cafe-bench run-all --task classification my_model` · results land as `results/<model>/*.json`, aggregated by `leaderboard.py`.

**Contract summary for the engineer:**
1. **`encode` is the only required method.** Input is a batched, already-windowed array `(N, T, C)`; output is a flat embedding matrix `(N, D)`. **`D` is unconstrained.** The harness handles splits, probing, and metrics.
2. **Embeddings are frozen** — the probe is fit downstream. So CLU must produce a *representation*, not a prediction, unless it overrides the probe.
3. **The default Event probe is CoxPH on frozen embeddings** — a linear proportional-hazards model. This is the harness's weakest link and CLU's biggest lever: **HEPA does not use it** (it overrides with its own survival-CDF head, §3.2). If CLU submits `encode`-only, it is handicapped vs HEPA. **Recommendation: CLU should override `event_predict`** — the interface explicitly permits it and HEPA sets the precedent.
4. **Signatures of the three override hooks are NOT specified in the README** (`...` placeholders). ⛔ **Unresolved — the engineer must read `cafe_bench/models/base.py` and `cafe_bench/models/hepa_model.py` in the private repo.** Do not guess the `event_predict` signature; the h-AUROC evaluator expects a per-horizon probability surface `p(t, Δt)` (inferable from HEPA's formulation, but unverified as the harness contract).
5. Repo file map (from README §Repository Structure) for the engineer: `cafe_bench/models/base.py` (BaseModel + default probes), `cafe_bench/models/hepa_model.py` (the reference override), `cafe_bench/evaluators/event.py` (h-AUROC, per-horizon + integrated), `cafe_bench/datasets/event/cmapss.py`, `cafe_bench/registry.py`, `cafe_bench/pipeline.py`.

---

## 3 · HEPA extraction

**Correct citation (README's is stale):** the README bibtex says `arXiv:2506.XXXXX, year=2025` — a **placeholder**. The real paper is **arXiv:2605.11130** (v1 2026-05-11, v3 2026-06-03), *"HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series"*, **Spotlight at FMSD, ICML 2026**. Author list matches the README bibtex exactly (Petersen, Lombardi, Maggioni, Mazzoleni, Martelli, Petersen). **Note the title differs** from the README's ("Self-Supervised Physical Encoding Enables System Understanding, Anomaly Detection, and Event Prediction") — the README bibtex is a pre-publication draft title; use the arXiv one. (The task file's `arXiv:2605.07675` is a *different* paper — Factorybench, cited *by* HEPA. Do not cite 2605.07675 as HEPA.)

### 3.1 Architecture (verbatim numbers)
- Encoder: **causal Transformer**, hidden dim **d=256**, **2 layers**, **4 attention heads**, patch size **P=16**.
- **Total: 2.16M params** — paper: *"a single 2.16 M-parameter architecture with fixed hyperparameters"*. ✅ matches the task file's "~2.16M per the sample script".
- Predictor: **2-layer MLP** taking (encoder output, horizon Δt) → predicted future representation.
- Preprocessing: *"per-context instance normalisation and sinusoidal positional encodings"*.

### 3.2 How it encodes, and the objective
- Causal encoder over the past context → summary embedding `h_t`. The **target** branch uses *"the same encoder f_θ, applied bidirectionally to x(t, t+Δt] with attention pooling, produces the target representation h*(t, t+Δt]"*. So: **causal for the query, bidirectional+attention-pooled for the target.** One shared encoder, two read modes.
- **Pretraining loss:** `ℒ = (1−α)‖ĥ − h*‖₁ + α·ℒ_SIGReg`, with **α = 0.1**. L1 in representation space + **SIGReg anti-collapse regularization**.
- **Finetune:** encoder **frozen**, only the predictor trained: `ℒ_FT = Σ w⁺ · BCE(p(t,Δt), y(t,Δt))` (positive-weighted BCE across horizons).
- **Head = discrete-time survival:** `p(t,Δt) = 1 − ∏_{j=1}^{Δt} (1 − λ_j(t))`, with sigmoid-bounded conditional hazards `λ_Δt` composed into a **monotonic CDF over horizons**. Monotonicity is architectural, not learned.
- Fair-comparison protocol (quotable): *"All comparison methods share the same 198K-param downstream MLP head, positive-weighted BCE loss, and evaluation protocol; only the frozen encoder differs."* **This is the protocol CLU should adopt** — it isolates the encoder, which is exactly the claim CLU wants to make.

### 3.3 Why it leads Event-Prediction — the mechanism
Predicting *future representations* rather than future *values* forces the encoder to retain only the **predictable** components of the dynamics and discard high-frequency sensor noise. On slow-degradation lifecycle data the degradation signal *is* the predictable low-frequency component, so a horizon-conditioned latent predictor is directly aligned with the label. Paper's own framing: on *"lifecycle datasets where failures unfold over hundreds of observation steps — bearing degradation, turbofan wear, sensor drift — the method excels because the future window contains diagnostic information that the target encoder can capture."* Plus: horizon-conditioning makes the representation Δt-aware, matching h-AUROC's per-horizon decomposition; and the monotone survival CDF removes the calibration failure modes of an unconstrained per-horizon classifier.

### 3.4 ⛔ The number discrepancy (load-bearing, unresolved)
| Dataset | CAFE README leaderboard | HEPA paper Table 1 (100% labels) | Best paper baseline |
|---|---|---|---|
| FD001 | **0.918** | **.81 ± .03** | .80 (PatchTST) |
| FD002 | **0.661** | **.57 ± .01** | .56 (MAE) |
| FD003 | **0.960** | **.84 ± .01** | .79 (PatchTST) |
| FD004 | **0.627** | **.63 ± .02** | .57 (MAE) |

Full paper Table 1 C-MAPSS block, verbatim (Chronos-2 / PatchTST / iTransformer / MAE / HEPA):
- C-MAPSS-1: `.66±.00` / `.80±.04` / `.70±.05` / `.69±.02` / **`.81±.03`**
- C-MAPSS-2: `.45±.01` / `.44±.03` / `.43±.03` / `.56±.01` / **`.57±.01`**
- C-MAPSS-3: `.73±.00` / `.79±.01` / `.76±.01` / `.78±.02` / **`.84±.01`**
- C-MAPSS-4: `—` / `.52±.03` / `.45±.02` / `.57±.02` / **`.63±.02`**

Candidate explanations (**none verified**): (a) different h-AUROC aggregation — README "integrated across horizons" vs paper "averaged per-horizon AUROC", possibly different horizon set H; (b) different window/split — README fixes horizon at 125 cycles, paper reports 85 engines for C-MAPSS-1 and may use a different windowing; (c) the leaderboard used a newer/larger checkpoint than the paper; (d) leaderboard entries are placeholder/aspirational (consistent with the README's `arXiv:2506.XXXXX` and every other cell being `—`). **Explanation (d) deserves serious weight** given the README's other placeholders.
**Action for the Hub:** the Head is a HEPA co-author (C. Mazzoleni) — this is a one-question resolution, faster than any reverse-engineering. **Do not lock the ICLR headline target until it is answered.**

---

## 4 · The peers and baselines

### Parametric weight-class (CLU's real comparison)
- **MOMENT** — Goswami et al. (2024), *"MOMENT: A Family of Open Time-series Foundation Models"*, **ICML 2024**, arXiv:2402.03885. T5-style masked-reconstruction encoder pretrained on the "Time Series Pile"; Small/Base/Large, **up to ~385M params** (CAFE uses `AutonLab/MOMENT-1-large`). Plugs in via `momentfm` (`pip install momentfm`); CAFE's wrapper is `moment_model.py` — *"(embedding + reconstruction)"*, i.e. embedding for classification/event, reconstruction error for anomaly. **~178× HEPA's params.**
- **UniTS** — Gao et al. (2024), *"UniTS: A Unified Multi-Task Time Series Model"*, **NeurIPS 2024**, arXiv:2403.00131. Single shared-parameter model, no task-specific modules; task-tokenization unifies predictive+generative tasks. **UniTS-SUP = 3.4M params** (paper: *"48× larger than UniTS-SUP (164.5M vs. 3.4M)"* re GPT4TS). Two regimes: UniTS-SUP (multi-task supervised) and UniTS-PMT (frozen pretrained + tuned task tokens). CAFE registers both `units` and `units_ft` → both point at `mims-harvard/UniTS-supervised-m`; wrapper `units_model.py` exposes *"classify + detect_anomaly"*. **Same weight class as HEPA (3.4M vs 2.16M)** — this is the honest parametric peer, not MOMENT.
- **PatchTST** (Nie et al., ICLR 2023, arXiv:2211.14730) and **iTransformer** (Liu et al., ICLR 2024, arXiv:2310.06625) — supervised Transformers, **classification-only** in CAFE, but note **PatchTST is HEPA's strongest C-MAPSS rival in the paper** (.80 on FD001).

### Specialized baselines (one line each)
- **DeepSVDD** — Ruff et al., *"Deep One-Class Classification"*, ICML 2018. Learns a network mapping normal data into a minimum-volume hypersphere; anomaly score = distance to centre. *Structurally the closest prior art to CLU's "energy valley" story* — a learned single-basin geometry.
- **LSTM-AE** — Malhotra et al. (2016), arXiv:1607.00148. LSTM encoder-decoder trained to reconstruct normal series; anomaly score = reconstruction error.
- **IsolationForest** — Liu, Ting & Zhou, ICDM 2008. Non-parametric random-partition isolation depth.
- **DeepHit** — Lee et al., AAAI 2018. Discrete-time deep survival model, direct distribution over event times, no proportional-hazards assumption. **This is the closest published relative of HEPA's head.**
- **CoxPH** — Cox (1972), JRSS-B. Semi-parametric proportional hazards; CAFE's *default* Event probe via `lifelines`.
*(These five refs are standard and were not re-verified against source this session — single-sourced from my own knowledge. Verify before they enter a paper.)*

**Explicit CAFE exclusion, quotable:** *"Forecasting-only models (Chronos, Moirai, TimesFM) are excluded — they have no classification head or reconstruction path and cannot be fairly compared on these tasks."* CLU has both an energy/reconstruction path and a rollout path, so it is admissible on all three tasks — worth saying out loud in the paper.

---

## 5 · Positioning verdict

**The Hub's hypothesis is confirmed, with one important correction to the mechanism story.**

**Lead: Event-Prediction / C-MAPSS.** Reasons, in order of strength:
1. **The metric is literally a basin-exit metric.** h-AUROC asks: at every horizon Δt, can you separate "system will leave the healthy regime within Δt" from "it will not"? That is CLU's `governed_rollout` / escape-time structure expressed as a scoring rule. No other CAFE task has this shape.
2. **Fully public data** — C-MAPSS auto-downloads with no credentials. Clean ICLR reproducibility.
3. **Thin margins.** HEPA beats the best baseline by **+0.01 on FD001 and +0.01 on FD002** (paper numbers). FD002 (.57) and FD004 (.63) are *near-chance* — 6-operating-condition, multi-fault-mode regimes where every existing method struggles. **FD002/FD004 are the most winnable cells in the entire benchmark**, and a win there is a *mechanism* claim (multi-regime = multi-basin, which is exactly the structure CLU has and a single-latent JEPA does not). Recommend pre-registering FD002/FD004 as the differentiating target and FD001/FD003 as the "comparable" target.
4. Only 6 datasets → a complete task sweep is cheap.

**Second: Anomaly.** Valley/EBM locality is real, and DeepSVDD/LSTM-AE are directly commensurable. **But** — the voraus result this wave (CLU 0.51–0.62 vs LOF 0.81 / kNN 0.75) is the *warning shot*: TSB-AD-style anomaly rewards local outlier detection, and CAFE's **default anomaly probe is literally kNN distance in embedding space**, i.e. the harness hands the task to CLU's measured weak axis unless CLU overrides `anomaly_score`. If you enter Anomaly, **override the probe with valley-aware energy scoring** or expect a repeat of voraus.

**Third: Classification.** Confirmed hardest — and additionally the *largest* task (30–36 registered datasets, plus 128 UCR), so it is also the most expensive to sweep. Macro-F1 on frozen embeddings + LogisticRegression rewards linearly-separable global structure, which is the property CLU currently lacks a mechanism for (wormholes/boost are unbuilt). Recommend: run it, report it honestly in an appendix as "comparable, not leading", and let it support the "one primitive, three task families" thesis rather than carry a claim.

**Correction to the mechanism narrative.** The Hub's framing positions CLU vs HEPA as *physics-structured dynamics vs. a Transformer*. That is not the actual contrast. HEPA is **also** a dynamics-predictability method — it just learns predictability in latent space instead of imposing it. Two consequences:
- **What to borrow:** (a) HEPA's **fair-comparison protocol** (identical 198K-param downstream head, only the frozen encoder differs) — adopt it verbatim, it is the cleanest way to make CLU's claim an *encoder* claim; (b) the **monotone survival-CDF head** `p(t,Δt)=1−∏(1−λ_j)` — it is architecture-level monotonicity, cheap, and CLU should not lose points to a calibration artifact; (c) **horizon-conditioning** the readout.
- **What to differentiate on:** HEPA's `h*` is an *empirical* future representation with **no conservation law, no causal speed limit, and no notion of a basin**. CLU's differentiator must be stated as: *predictability is enforced by the Hamiltonian structure at every horizon simultaneously, so extrapolation beyond the training horizon degrades gracefully.* **The sharpest testable claim: h-AUROC as a function of Δt, evaluated beyond the H seen in finetuning.** If CLU's per-horizon AUROC curve decays more slowly than HEPA's at long Δt, that is a structural result no parameter count explains — and it is exactly Exp-I (long-horizon stability) transplanted onto a real benchmark. **Recommend this as the headline figure**, not a single scalar comparison.
- ⚠ **Novelty collision — flag for the anti-collapse thread.** HEPA's pretraining loss already contains an explicit anti-collapse term (`ℒ_SIGReg`, α=0.1). The Hub's "VICReg for a dynamical latent" thread must therefore be framed as anti-collapse of **structural/dynamical levers** (mass spectrum, symmetry, mode-band coverage) — *not* representation collapse, which HEPA (and VICReg, and SIGReg) already own. This is a narrowing, and the `anti-collapse-characterization` theorist should be told before it writes its framing section.

---

## Confidence & gaps

**Verified (primary source, direct quote):** HEPA architecture (2.16M / d=256 / 2 layers / 4 heads / P=16), both loss equations, the survival-CDF equation, the 198K-head protocol, the full Table-1 C-MAPSS block, the 14-benchmark list, attention-pooled bidirectional target. Verified twice (two independent fetches of arXiv:2605.11130v3 returned consistent numbers). UniTS 3.4M and MOMENT ≤385M verified from primary/official sources. TSB-AD venue + VUS-PR rationale verified.

**Single-sourced (README only, unverifiable without repo access):** the entire `cafe-bench` interface, the leaderboard numbers, dataset counts, file layout. The README is a *pre-release draft* — it contains an unassigned arXiv placeholder, a superseded paper title, and an empty leaderboard. **Treat its numbers as provisional.**

**Gaps / what to search or ask next:**
1. ⛔ **The 0.918 vs .81 discrepancy** — ask the Head (HEPA co-author) directly. Blocks the ICLR headline target.
2. ⛔ **Exact signatures of `encode` / `classify` / `anomaly_score` / `event_predict`** in `cafe_bench/models/base.py`, and whether `encode` receives variable-length or pre-windowed `(N,T,C)` for C-MAPSS. **The engineer cannot finish `clu-cafe-integration` without reading `base.py` + `hepa_model.py`.** Needs a Bash-capable agent or the Head to paste them.
3. Whether `cafe-bench` will be public by ICLR submission (currently 404) — an anchor benchmark that reviewers cannot install is a reviewer-facing risk. Mitigation: also report against the *paper's* public protocol.
4. HEPA appendices **G and H** reportedly contain MOMENT/UniTS comparisons — not extracted; worth pulling if the ICLR table needs foundation-model numbers.
5. Whether HEPA's C-MAPSS labels/windowing (85 engines, 125-cycle horizon) match CAFE's `cmapss.py` — determines whether our number is comparable to *either* published set.

---

## Bibtex-ready refs

```bibtex
@article{petersen2026hepa,
  title  = {HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series},
  author = {Petersen, Jonas and Lombardi, Gian-Alessandro and Maggioni, Riccardo and
            Mazzoleni, Camilla and Martelli, Federico and Petersen, Philipp},
  journal = {arXiv preprint arXiv:2605.11130},
  note   = {Spotlight, FMSD Workshop, ICML 2026},
  year   = {2026}
}
@inproceedings{goswami2024moment,
  title     = {MOMENT: A Family of Open Time-series Foundation Models},
  author    = {Goswami, Mononito and Szafer, Konrad and Choudhry, Arjun and Cai, Yifu and Li, Shuo and Dubrawski, Artur},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  note      = {arXiv:2402.03885}
}
@inproceedings{gao2024units,
  title     = {UniTS: A Unified Multi-Task Time Series Model},
  author    = {Gao, Shanghua and Koker, Teddy and Queen, Owen and Hartvigsen, Thomas and Tsiligkaridis, Theodoros and Zitnik, Marinka},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2403.00131}
}
@inproceedings{liu2024tsbad,
  title     = {The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark},
  author    = {Liu, Qinghua and Paparrizos, John},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2024}
}
```
*(UniTS author list beyond first author is from the NeurIPS listing and was not re-verified line-by-line — check before submission.)*

---

## Proposed handover updates (for the Hub)

1. **§2026-07-20 PIVOT entry, the HEPA leaderboard line** — annotate: *"README leaderboard FD001 0.918 / FD002 0.661 / FD003 0.960 / FD004 0.627 **conflicts with the published HEPA paper (arXiv:2605.11130v3, Table 1): .81/.57/.84/.63**. Only FD004 agrees. Target number UNRESOLVED — ask the Head (co-author) before locking the ICLR headline."*
2. **Correct the HEPA citation everywhere**: it is **arXiv:2605.11130**, *"HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series"*, ICML 2026 FMSD spotlight. The task file's `arXiv:2605.07675` is Factorybench (a *cited* work), and the README bibtex's `2506.XXXXX`/2025/old-title is a placeholder.
3. **`clu-cafe-integration` is BLOCKED on repo read-access**, not on this scout: the README specifies `encode` fully but leaves the three override signatures as `...`. Add to that task: "first action = read `cafe_bench/models/base.py` and `cafe_bench/models/hepa_model.py`." Also add: **CLU should override `event_predict`, not rely on the default CoxPH probe** (HEPA does), and **adopt HEPA's identical-198K-head protocol**.
4. **Sharpen the Event-Prediction target**: FD002 (.57) and FD004 (.63) are near-chance multi-regime cells and are the most winnable + most mechanism-diagnostic; FD001/FD003 are "comparable" targets where HEPA's margin over PatchTST is only +0.01/+0.05. Pre-register accordingly.
5. **New headline-figure proposal**: per-horizon AUROC vs Δt *extrapolated beyond the finetuning horizon H* — turns Exp-I (long-horizon stability) into a real-benchmark claim that param count cannot explain.
6. **Anti-collapse thread narrowing (tell the theorist before it drafts):** HEPA already ships an anti-collapse regularizer (`ℒ_SIGReg`, α=0.1) in its pretraining loss. "VICReg for a dynamical latent" must be framed as **structural/dynamical-lever collapse** (mass spectrum, symmetry, mode-band), explicitly *not* representation collapse — that ground is taken.
7. **Reproducibility note for §ICLR risks:** `github.com/forgislabs/cafe-bench` is **404 unauthenticated** as of 2026-07-20 and the CAFE arXiv ID is unassigned. Anchoring on a benchmark reviewers cannot install is a live risk; mitigate by also reporting against HEPA's published public protocol.
8. **Yahoo S5 is the only hard-licensed CAFE dataset** — exclude or mark N/A in any ICLR table.
```
