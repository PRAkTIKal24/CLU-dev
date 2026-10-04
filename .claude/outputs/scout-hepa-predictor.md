# scout-hepa-predictor — web-scout report

Task + acceptance criterion: Extract HEPA's 198K dt-conditioned head + predictor spec exactly, answer whether the predictor can be iterated, define the encoder→head socket, settle C-MAPSS labelling, re-extract App. G/H, identify MTS-JEPA, disambiguate HEPA vs HEPA-SP.
Status: **partial** — items 1, 2, 3, 5, 6 delivered; item 4 (C-MAPSS labelling) **UNRESOLVED, and I explain exactly why**; item 7 (HEPA-SP) **BLOCKED**.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (read first, per protocol §5).**
> 1. **⛔ TWO NUMBERS CURRENTLY IN OUR NOTES CAME FROM A HALLUCINATING FETCH AND MUST NOT BE USED.** An early WebFetch confidently "quoted" Appendix L (*"uses the official RUL_FD00x.txt files"*, *"FD001 has 85 training engines"*) and Appendix M (*"Adam, lr 1e-3, 100 epochs, batch 32"* / *"lr 1e-4, 50 epochs"*). **I then proved Appendices L and M are NOT retrievable by any renderer I have.** Those strings are fabricated by the summarizer. **Do not propagate. HEPA's optimizer/lr/epochs are UNKNOWN to us.**
> 2. **⭐ HEPA does NOT use a 30-cycle window on C-MAPSS — it uses FULL ENGINE HISTORY.** CAFE uses `window=30`. This is a *protocol* difference far larger than any head difference and is a fresh candidate explanation for the 0.918/0.81/0.73 spread. Affects `cmapss-fd002-004-fetch` and any external comparison.
> 3. **TEP-exclusion reason in the handover is wrong.** Handover §App.H says *"TEP excluded (encoder fails to converge on 52-dim input)"*. Table 6's actual caption: *"TEP is excluded because the public MTS-JEPA release does not include a chemical-process benchmark."* Correct at that site.
> 4. **Table 6 error bar:** handover records C-MAPSS-1 as `.81±.03` (Table 1); **Table 6 prints `0.81 ± 0.04`**. Minor, but pick one per table and don't cross-quote.
> 5. **MTS-JEPA has been RENAMED.** arXiv:2602.04643 is now *"SC-JEPA"*, SDM 2026. Citing "MTS-JEPA" without the note will look stale to a reviewer.

---

## Answer first

**The 198K head is fully recovered and its parameter count verifies to the digit** (197,632 + 769 = 198,401): it is a 3-linear-layer MLP `[h_t ; Δt] : 257 → 256 → 256 → 256` plus a *shared* `LayerNorm+Linear→1` event head applied **once per horizon**. **Δt is conditioned by raw-scalar concatenation** — the 257 input width is what makes the arithmetic close, which is strong evidence against FiLM/embedding/sinusoidal. **The predictor CAN be iterated dimensionally — `ĥ, h, h* ∈ ℝ²⁵⁶` — but SHOULD NOT be naively, because `ĥ` is trained to match a *bidirectional, attention-pooled read of the future window*, not a *causal read of the past* that the predictor consumes as input**; the spaces are type-compatible and semantically mismatched. HEPA ships **no** multi-step/autoregressive path and **no** beyond-horizon evaluation, so our rollout experiment is not preempted. Training horizons are **enumerated dense unit steps, Δt ∈ {1..150} for C-MAPSS**, log-uniformly *sampled* during pretraining.

---

## 1 · The 198K dt-conditioned head — implementable spec

**[PAPER-TEXT, verbatim, ar5iv render of arXiv:2605.11130v3]**
> *"the pretrained predictor MLP (a 3-layer MLP mapping `[𝐡_t;Δt]→𝐡̂∈ℝ^256`; 197.6K params) plus a shared linear event head (LayerNorm + linear → logit; 769 params), totalling 198K finetuned parameters."*

**[MY ARITHMETIC — verifies exactly, this is the reconstruction the engineer can build]**

| layer | shape | weights | biases | total |
|---|---|---|---|---|
| `fc1` | 257 → 256 | 65,792 | 256 | 66,048 |
| `fc2` | 256 → 256 | 65,536 | 256 | 65,792 |
| `fc3` | 256 → 256 | 65,536 | 256 | 65,792 |
| **predictor** | | | | **197,632 = "197.6K"** ✓ |
| `LayerNorm(256)` | γ,β | 256 | 256 | 512 |
| `Linear(256→1)` | | 256 | 1 | 257 |
| **event head** | | | | **769** ✓ (exactly) |
| **TOTAL** | | | | **198,401 ≈ "198K"** ✓ |

Both sub-counts land on the printed figures exactly, which pins three things the paper never states outright:
- **input width = 257 ⇒ Δt enters as ONE raw scalar concatenated to `h_t`.** Not FiLM, not a learned embedding, not sinusoidal — any of those would change the first-layer width and break 197.6K. **[INFERENCE, strongly supported]** *(Open: whether Δt is normalized, e.g. Δt/K, before concat. Arithmetic cannot see this. Recommend `Δt/K` for conditioning stability and record the choice.)*
- **constant hidden width 256** (no bottleneck/expansion).
- **"3-layer" = 3 Linear layers**; the pretraining-time description *"a 2-layer MLP"* refers to the same module counted as 2 hidden layers. **Not two different modules — the 197.6K arithmetic only closes for 3 Linears.** **[INFERENCE]** Flagged because the two phrasings appear in different sections and read like a contradiction.
- Activations are **not stated** anywhere I could retrieve. **[GAP]** GELU is the PatchTST-lineage default; the engineer must pick and record it.

**Loss [PAPER-TEXT, verbatim]:**
```
ℒ_FT = Σ_{Δt=1}^{K} w⁺ · BCE( p(t,Δt), y(t,Δt) )        with   w⁺ = N_neg / N_pos
```
`w⁺` is a plain global class-balance ratio. **[GAP]** whether `w⁺` is computed *per horizon* (`N_neg(Δt)/N_pos(Δt)`) or once globally is not stated — and it materially matters, because prevalence sweeps from ~0 to ~1 across Δt∈{1..150}. **Per-horizon is the only choice consistent with the paper's own "universal 0.5 baseline that does not depend on prevalence" framing** — recommend per-horizon and pre-register it.

**Survival CDF [PAPER-TEXT, verbatim]:**
```
p(t,Δt) = 1 − ∏_{j=1}^{Δt} ( 1 − λ_j(t) )
```
> *"Because each factor `(1−λ_j)∈(0,1)`, the survival product is non-increasing in Δt, so `p(t,Δt)` increases monotonically"*

**Assembly [INFERENCE, forced by "shared linear event head → logit" (singular output) + K horizons]:** for each `Δt ∈ {1..K}`: `ĥ_Δt = g_φ([h_t ; Δt])` → `λ_Δt = σ(EventHead(ĥ_Δt))` → cumulative product. **The predictor is run K times per sample, the head shares weights across horizons.** Monotonicity is architectural (the cumprod), never a loss term or a constraint on weights.
⇒ **This is precisely why the matched head resurrects the killed headline figure:** the per-horizon ranking varies with Δt because `ĥ_Δt` varies with Δt — unlike CoxPH, whose single risk score gave the engineer rank-corr 1.0000 across all 125 horizons.

**Optimizer / lr / schedule / epochs / early-stopping: NOT OBTAINED.** Appendix M is unreachable (see §7). Anything you have seen claiming these is item-1 of the reconciliation list.

---

## 2 · ⭐ THE PREDICTOR, AND WHETHER IT CAN BE ITERATED

**Signature [PAPER-TEXT, verbatim]:**
```
h_t        = f_θ(x_≤t)        ∈ ℝ^d        (causal encoder, d = 256)
ĥ(t,t+Δt]  = g_φ(h_t, Δt)     ∈ ℝ^d        (2-layer/3-Linear MLP predictor)
h*(t,t+Δt] ∈ ℝ^d                            (target: SAME encoder f_θ applied
                                             bidirectionally to x(t,t+Δt] with
                                             attention pooling)
ℒ = (1−α)‖ĥ − h*‖₁ + α·ℒ_SIGReg ,   α = 0.1
```
`h` is a **pooled vector, not a sequence** (confirmed independently by repo source, §3).

### Can it be iterated? — **DIMENSIONALLY YES, SEMANTICALLY NO.** ⭐

- **Yes, the output space equals the input space in the narrow sense:** `ĥ, h_t, h*` are all `∈ ℝ²⁵⁶`. One fetch also reports **L2 normalisation** applied to encoder and target representations, which would make them share a unit sphere. **[SINGLE-SOURCED — verify the L2-norm claim before building on it.]** So `g_φ(ĥ, Δt)` is a **type-correct** operation; nothing crashes.
- **But `ĥ` and `h_t` are different objects.** `h_t` is a **causal summary of the past** `x_≤t`. `ĥ` is trained to regress `h*`, a **bidirectional, attention-pooled summary of the future window** `x(t, t+Δt]`. These are two *different read modes of the same encoder* with different receptive fields and different pooling. Feeding `ĥ` back into `g_φ` therefore feeds the predictor an input drawn from a distribution it was **never trained on** — the causal-past manifold and the bidirectional-future-window manifold coincide only if the encoder happens to map both to the same region, which the loss never asks for. **[MY INFERENCE from the verbatim definitions — I consider this the correct answer to the task's yes/no, and it is more useful than either bare answer.]**

**Consequence for experiment design (the thing the Hub actually needs):** an "HEPA-iterated" arm is **constructible but is a straw man** — it would fail for a reason that is an artifact of JEPA's asymmetric read modes, not a real statement about direct-vs-iterated prediction. Two honest alternatives:
1. **The fair arm is HEPA-extrapolated, not HEPA-iterated:** evaluate `g_φ(h_t, Δt)` at `Δt > K`. The predictor accepts any scalar Δt, so this is a **one-line** experiment and tests exactly the intended claim (direct horizon-conditioned map vs integrated dynamics, beyond the training horizon) with no distribution violation. **Recommend this as the primary contrast.**
2. If an iterated HEPA arm is wanted anyway, it must be built with a **causal-mode target** (train `ĥ` against `f_θ(x_≤t+Δt)` rather than the bidirectional pooled `h*`), and that is a *modification of HEPA*, which must be disclosed as such.

### Training horizon set `H` [PAPER-TEXT, verbatim]
- **Pretraining:** Δt **sampled** from *"a log-uniform distribution over `[1, Δt_max]`"*.
- **Finetuning / evaluation:** **enumerated, dense unit steps** — *"K=150 for C-MAPSS and TEP (Δt∈{1,2,…,150}), and K=200 for all other datasets (Δt∈{1,2,…,200})"*.
- ⚠ **`K=150` for C-MAPSS, but CAFE's `cmapss.py` hard-codes `horizon_max=125`** (repo source, §3). Our 125 horizons ≠ HEPA's 150. **"Beyond H" starts at Δt=151 on the paper protocol, Δt=126 on the CAFE protocol.** The rollout experiment must state which.

### Online/target asymmetry
- **No EMA, no separate target network** — *"the same encoder f_θ"*, two read modes. **[VERIFIED, two fetches]**
- **No stop-gradient reported**; anti-collapse is carried entirely by **SIGReg** (α=0.1), which *"constrains the predicted representations toward an isotropic Gaussian."* **[SINGLE-SOURCED on the no-stop-grad point — a JEPA with neither EMA nor stop-grad is unusual enough that I flag it as the single claim here most worth a second look.]**

### Does HEPA already do multi-step / beyond-horizon? — **NO.**
Each horizon is *"computed at each of K discrete horizons"* **independently**; there is no feedback loop and no evaluation outside `[1,K]`. **Our rollout experiment is not preempted** and can be presented as novel — with the honest caveat that the *reason* HEPA can't iterate is architectural (§ above), which is itself the interesting sentence for the paper.

---

## 3 · The encoder→head socket (what CLU must implement)

**[REPO SOURCE — `~/cafe-bench/cafe_bench/models/hepa_model.py`, verbatim]**
```python
# L38-50
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "hepa-sd" / "HEPA-SP"))
from hepa.model.encoder import CausalEncoder
ckpt = torch.load(self._ckpt, map_location=self._device)
cfg  = ckpt["config"]
self._encoder = CausalEncoder(
    n_channels=cfg["n_channels"], patch_size=cfg.get("patch_size", 16),
    d_model=cfg.get("d_model", 256), n_heads=cfg.get("n_heads", 4),
    n_layers=cfg.get("n_layers", 2),
)
# L54-63
def encode(self, X: np.ndarray) -> np.ndarray:
    """(N, T, C) → (N, D) embeddings."""
    ...  out.append(self._encoder(batch).cpu().numpy())
```
**Socket, settled:**
- `CausalEncoder.forward` returns a **single pooled vector per sample** — `encode` concatenates it straight to `(N, D)` with no pooling/projection applied downstream. **D = d_model = 256.** **[REPO SOURCE — this independently confirms the paper's `h_t ∈ ℝ²⁵⁶` and confirms "pooled vector, not sequence".]**
- **No projection or normalization sits between encoder and head** in the CAFE path. (Whether HEPA-SP's own finetune path inserts one is unverifiable — §7.)
- **Which read mode feeds the head: the CAUSAL one.** `h_t = f_θ(x_≤t)`. The bidirectional + attention-pooled mode exists *only* to manufacture the pretraining target `h*` and is never used downstream. **CLU must therefore expose a causal, pooled, 256-d embedding** — it does not need to replicate the dual read mode at all.
- The checkpoint carries its own `config` dict (`n_channels, patch_size, d_model, n_heads, n_layers`) — hyperparameters travel with weights.

**[REPO SOURCE — `models/base.py` L67-100]** the override the matched-head protocol needs:
```python
def event_predict(self, X_train, t_train, e_train, X_test, horizons) -> np.ndarray:
    """Return (N_test, len(horizons)) P(event by horizon h) for each h."""
```
⇒ the 198K head slots in here, returning `p(t,Δt)` of shape `(N_test, K)`. **[REPO SOURCE — `evaluators/event.py` L38-52]** confirms h-AUROC = `np.mean` of per-horizon `roc_auc_score(y_h, p_event[:, i])` with `y_h = ((t_test <= h) & (e_test == 1))`, degenerate horizons skipped. (Docstring says "integrated…normalized by H"; code is a plain mean — same thing, as the engineer found.)

---

## 4 · Data / protocol — and the C-MAPSS labelling question

**[PAPER-TEXT, verbatim]**
- *"tokenised into non-overlapping patches of size P=16 (following PatchTST) with per-context instance normalisation and sinusoidal positional encodings"* ⇒ **stride = P = 16** (non-overlapping).
- Context: ***"sliding window of 512 steps (32 tokens)"*, with the note *"except C-MAPSS which uses full engine history"***. (512/16 = 32 tokens ✓.)

**⭐ The protocol gap nobody had spotted:**

| | HEPA paper | CAFE `cmapss.py` **[repo source]** |
|---|---|---|
| context | **full engine history** | `window=30` cycles |
| normalization | **per-context instance norm** | **per-channel z-score using TRAIN statistics** (L86-89) |
| channels | not stated | 14 (drops `s1,s5,s6,s10,s16,s18,s19`) |
| horizons K | **150** | **125** (`horizon_max=125`) |

Three of four rows differ. **These are not the same benchmark**, and this is a stronger candidate for the 0.918/0.81/0.73 spread than the head or the aggregation ever were.

### Does HEPA use the true-RUL file? — **I COULD NOT DETERMINE THIS. Answering explicitly, as instructed.**
Appendix L (per-dataset preprocessing) is **not present in any renderer I can reach**: `arxiv.org/html` truncates mid-Appendix-C; ar5iv carries G and H but returned "Appendix L is not included" under an explicit anti-hallucination guard; the PDF cannot be rendered locally (**no `pdftoppm`/poppler on this machine**); and I have **no Bash**, so I cannot run a text extractor. **I therefore have no evidence either way, and I am not going to guess** — the one fetch that "answered" this question was demonstrably fabricating (reconciliation item 1).

**What IS established [REPO SOURCE, `cmapss.py` L67-72, verbatim]** — CAFE's side of the question, confirming the engineer:
```python
for i in range(self._window, max_cycle + 1):
    window = sensors[i - self._window:i]   # (W, C)
    rul    = max_cycle - i                  # remaining cycles
    X_list.append(window); t_list.append(rul); e_list.append(1)   # all observed
```
`e=1` for **every** window, `t = max_cycle − i` = **cycles remaining in the recording**, and **`RUL_FD00x.txt` is never opened anywhere in the file.** Since official C-MAPSS *test* sequences are truncated before failure, CAFE's test labels are systematically wrong as RUL. ✓ engineer confirmed verbatim.

**Cheapest resolution by far: ask the Head (C. Mazzoleni is a HEPA co-author).** One question — *"does HEPA's C-MAPSS loader read RUL_FD00x.txt, and does it really use full engine history rather than fixed windows?"* — settles both the labelling and the context-length question and is worth more than any further scraping. **I recommend this over another scout wave.**

---

## 5 · Appendix G / H — clean re-extraction ✅

Both tables retrieved in full from **ar5iv** under an explicit "quote only what is present" guard, after the arxiv.org renderer proved to truncate.

### Table 5 (App. G) — matched 198K head, encoder-isolated
**Caption, verbatim:** *"HEPA vs. four foundation models (matched 198K MLP head, 100% labels). HEPA: 5 seeds. Chronos-2, MOMENT, TFM-2.5, Moirai: 3 seeds each. Bold = best per row."*
**Protocol, verbatim:** *"All five encoders are frozen and feed an identical 198K-param dt-conditioned MLP head trained with positive-weighted BCE under the same labels, splits, and evaluation protocol; only the frozen encoder differs."*

| Dataset | HEPA (5s) | Chronos-2 | MOMENT | TFM-2.5 | Moirai |
|---|---|---|---|---|---|
| C-MAPSS-1 | **0.73±.02** | 0.66±.00 | 0.56±.01 | 0.53±.00 | 0.61±.00 |
| C-MAPSS-2 | 0.58±.01 | 0.50±.01 | **0.70±.00** | 0.60±.01 | 0.66±.00 |
| C-MAPSS-3 | **0.82±.02** | 0.72±.02 | 0.47±.01 | 0.62±.01 | 0.70±.00 |
| SMAP | **0.60±.03** | 0.53±.01 | — | 0.51±.03 | — |
| PSM | 0.55±.02 | 0.49±.00 | — | **0.57±.01** | 0.53±.01 |
| MBA | 0.75±.01 | 0.55±.01 | **0.79±.01** | 0.76±.01 | 0.57±.02 |
| GECCO | 0.81±.07 | 0.81±.01 | — | **0.93±.01** | 0.82±.01 |
| BATADAL | 0.64±.02 | 0.58±.01 | 0.54±.07 | **0.65±.01** | 0.36±.01 |
| ETTm1 | **0.87±.00** | 0.78±.01 | — | 0.59±.01 | 0.60±.00 |

**Rows the handover had:** C-MAPSS-1/2/3 ✓ all confirmed identical. **Rows newly recovered:** SMAP, PSM, MBA, GECCO, BATADAL, ETTm1. **No previously-recorded cell was wrong** — they were simply absent. Note **HEPA wins only 4 of 9** under its own matched-head protocol (C-MAPSS-1, C-MAPSS-3, SMAP, ETTm1), losing MBA and BATADAL and GECCO and C-MAPSS-2 — materially more modest than Table 1 suggests, and worth knowing before we frame "beat HEPA".

### Table 6 (App. H) — SSL-objective-only ablation
**Caption, verbatim:** *"HEPA vs. MTS-JEPA. Mean (±std) over available seeds. HEPA: 5 seeds. MTS-JEPA reproduction: 1–3 seeds."* · *"TEP is excluded because the public MTS-JEPA release does not include a chemical-process benchmark."*

| Dataset | HEPA | MTS-JEPA |
|---|---|---|
| C-MAPSS-1 | **0.81 ± 0.04** | 0.69 ± 0.02 |
| C-MAPSS-2 | **0.57 ± 0.01** | 0.53 ± 0.02 |
| C-MAPSS-3 | **0.84 ± 0.02** | 0.78 ± 0.00 |
| SMAP | **0.59 ± 0.06** | 0.49 ± 0.00 |
| PSM | **0.57 ± 0.02** | 0.48 ± 0.00 |
| MBA | 0.75 ± 0.04 | **0.88 ± 0.00** |
| GECCO | **0.88 ± 0.06** | 0.84 ± 0.00 |

**7 rows only** — ETTm1, BATADAL, TEP are **absent** from Table 6. HEPA wins **6 of 7** (handover said "8 of 9" — that count is wrong; correct it). MTS-JEPA's sole win is MBA (cardiac), consistent with Table 5 where MBA is also HEPA's weakest lifecycle-atypical row.
⚠ Table 6's HEPA numbers are the **survival-CDF full-system** numbers (0.81 on C-MAPSS-1), *not* the matched-head 0.73 of Table 5 — **Tables 5 and 6 are different protocols. Never place them in one table.**

---

## 6 · MTS-JEPA — identified, and it has been renamed ⚠

**[VERIFIED, arXiv listing]** arXiv:**2602.04643**, submitted 4 Feb 2026, revised 17 Jul 2026.
- **v1 title:** *"MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction"* — the name HEPA cites.
- **current title:** *"SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction"*.
- **Authors:** Yanan He, Yunshi Wen, Xin Wang, Tengfei Ma.
- **Venue: accepted at SDM 2026** (SIAM International Conference on Data Mining) — **peer-reviewed, not a preprint.**
- **Objective:** asymmetric teacher–student JEPA + **multi-resolution predictive objective** + **soft codebook bottleneck** — the codebook *"stabilizes optimization and encourages discriminative, prototype-anchored regime codes"* and *"explicitly decouples transient shocks from long-term trends"*; motivated by *"representation collapse and an inability to capture precursor signals across varying temporal scales."* Benchmarks MSL, SMAP, SWaT, PSM under an *"early-warning anomaly prediction protocol."*
- **Public release: CONFLICTING EVIDENCE.** HEPA's Table 6 caption asserts *"the public MTS-JEPA release"* exists; the arXiv abstract page shows **no code link**. **Unresolved — do not assert a public release.**

**Why this matters to us, twice over:** (a) it is the **published template** for the encoder-swap claim — same head, same context/patch, *"only the SSL objective differs"*; (b) its **soft codebook = discrete regime codes** is uncomfortably close to CLU's "multi-basin ⇒ multi-regime" story for FD002/FD004. **The multi-basin novelty claim must now be positioned against SC-JEPA's codebook explicitly**, and SC-JEPA *loses* to HEPA on C-MAPSS-2 (0.53 vs 0.57), which is a useful data point: discrete regime codes alone did **not** crack the multi-regime cell.

---

## 7 · HEPA vs HEPA-SP — ⛔ BLOCKED, stating plainly

**`https://github.com/Forgis-Labs/HEPA-SP` returns HTTP 404 unauthenticated.** I **cannot clone it: this agent has no Bash tool** — contrary to the task file's premise ("You have Bash — clone it"). My toolset is Read/Write/Grep/Glob/WebSearch/WebFetch only. **The single biggest gap of the last wave is therefore still open**, and I am not going to reconstruct a plausible head from the repo I cannot read.

**What repo source *does* prove about it [`hepa_model.py` L38-40]:** HEPA-SP is expected at a sibling path `hepa-sd/HEPA-SP` and exposes a package `hepa` with module `hepa.model.encoder` containing class `CausalEncoder`. **So the head/predictor, if released, lives under `hepa/model/` — most likely `hepa/model/predictor.py` or `hepa/model/head.py`.** That is a concrete path for whoever has credentials.

**On the SP disambiguation itself:** I found **no mention of "semantic priors" or an SP variant anywhere in the retrievable text of arXiv:2605.11130v3.** The paper presents a single model. **Best current reading [INFERENCE]: every published number in the paper — Tables 1, 5, 6 — is BASELINE HEPA**, and "SP" is a later/internal extension whose repo happens to host the baseline package. **I cannot verify this and it should not be asserted in a draft until someone reads the repo or asks the Head.**

**Note the tooling asymmetry to fix for next wave:** the *engineer* has Bash and clone authorization; this scout does not. **Either grant the scout Bash, or assign the HEPA-SP read to an engineer-type agent.** This is the second consecutive wave lost to it.

---

## What the engineer can now build without guessing

1. **The predictor + head, end to end**: `Linear(257→256) → act → Linear(256→256) → act → Linear(256→256)`, then shared `LayerNorm(256) → Linear(256→1) → sigmoid` = `λ_Δt`; run once per `Δt ∈ {1..K}`; `p(t,Δt) = 1 − cumprod(1−λ)`. Param count **198,401**, verified against both printed figures.
2. **Δt conditioning**: raw scalar concatenated (recommend `Δt/K`, record the choice).
3. **Loss**: `Σ_Δt w⁺·BCE(p,y)`, `w⁺ = N_neg/N_pos` (recommend per-horizon; pre-register).
4. **Monotonicity**: architectural via cumprod — nothing to add to the loss.
5. **The socket**: causal, pooled, **256-d** embedding, no intervening projection/norm. CLU does **not** need HEPA's dual read mode.
6. **The CAFE override point**: `event_predict(X_train, t_train, e_train, X_test, horizons) -> (N_test, len(horizons))`; h-AUROC = mean of per-horizon AUROC, degenerate horizons skipped.
7. **Horizon sets**: HEPA C-MAPSS `K=150`; CAFE `horizon_max=125`. Pick one, state it, and define "beyond H" accordingly.
8. **The clean App. G/H tables** (§5) — the peer column is free and now complete.
9. **The rollout experiment's correct form**: HEPA-**extrapolated** (`Δt > K`), not HEPA-iterated. One-line change, no distribution violation, tests the intended claim.

## What remains unknown (equally important)

1. **⛔ HEPA's optimizer, lr, schedule, epochs, batch size, early stopping** — Appendix M unreachable. *Any* claim of a matched protocol is incomplete until these are known; a 198K head is very sensitive to lr/epochs. **This is the top blocker for a genuine matched-head comparison.**
2. **⛔ Whether HEPA uses the true-RUL file on C-MAPSS** — Appendix L unreachable. Determines whether *any* of our numbers are comparable to the paper's.
3. **Activation function** in the head (unstated), and whether Δt is normalized before concat.
4. **Whether `w⁺` is global or per-horizon.**
5. **Whether L2 normalisation really is applied to `h`/`h*`** — single-sourced; decides the geometry of any iteration argument.
6. **Whether no-stop-gradient is genuinely correct** — unusual for a JEPA; single-sourced.
7. **Everything inside HEPA-SP**, incl. what "semantic priors" are and which variant produced which number.
8. **Whether SC-JEPA/MTS-JEPA has a public release.**
9. **Exact C-MAPSS engine counts / splits** used by HEPA (the "85 engines" figure in our notes is from the hallucinating fetch — treat as unsourced).

---

## Confidence & gaps

**Verified (primary source, ≥2 independent fetches or exact arithmetic):** the 198K/197.6K/769 decomposition (arithmetic closes to the digit); `p(t,Δt)` and `ℒ_FT` equations; `w⁺=N_neg/N_pos`; `K=150` C-MAPSS / `K=200` others; log-uniform Δt in pretraining; `d=256`/2 layers/4 heads/`P=16`; no EMA; no multi-step path; the socket and all CAFE repo quotes (read directly from `~/cafe-bench`, not the README); Tables 5 and 6 in full (guarded fetch).

**Single-sourced:** L2 normalisation of `h`/`h*`; absence of stop-gradient; "full engine history" for C-MAPSS; the Table 5/6 rows beyond C-MAPSS (one guarded ar5iv fetch, though the C-MAPSS rows within the same fetch match our independent record exactly, which raises my confidence in the rest).

**Fabricated by a tool and excluded:** Appendix L/M content (RUL-file usage, engine counts, Adam/lr/epochs). Caught by re-asking with an explicit "say if absent" guard — **recommend that guard become standard practice for appendix extraction in this program.**

**Method note:** `arxiv.org/html/<id>` **silently truncates** long papers (this one died mid-Appendix-C) while **`ar5iv.labs.arxiv.org/html/<id>` carried the full appendices**. Anyone extracting appendices should use ar5iv and always ask "where does the text end?" first.

---

## Bibtex-ready refs

```bibtex
@article{petersen2026hepa,
  title   = {HEPA: A Self-Supervised Horizon-Conditioned Event Predictive
             Architecture for Time Series},
  author  = {Petersen, Jonas and Lombardi, Gian-Alessandro and Maggioni, Riccardo and
             Mazzoleni, Camilla and Martelli, Federico and Petersen, Philipp},
  journal = {arXiv preprint arXiv:2605.11130},
  note    = {v1 11 May 2026; v3 3 Jun 2026. Spotlight, FMSD Workshop, ICML 2026},
  year    = {2026}
}
@inproceedings{he2026scjepa,
  title     = {SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series
               Anomaly Prediction},
  author    = {He, Yanan and Wen, Yunshi and Wang, Xin and Ma, Tengfei},
  booktitle = {Proceedings of the 2026 SIAM International Conference on Data Mining (SDM)},
  year      = {2026},
  note      = {arXiv:2602.04643; v1 titled ``MTS-JEPA: Multi-Resolution
               Joint-Embedding Predictive Architecture for Time-Series Anomaly
               Prediction'' --- the name cited by HEPA}
}
```

---

## Proposed handover updates (for the Hub)

1. **Add the 198K head spec (§1) verbatim to the handover** — it unblocks `clu-horizon-encoder`. Include the arithmetic table; it is the evidence that the reconstruction is right.
2. **⛔ Post a correction notice for the hallucinated Appendix L/M numbers** (RUL-file usage, "85 engines", Adam/lr/epochs). They exist only in this session's discarded fetch, but if any of them reached a note, they must be struck. **Adopt the "say if absent" fetch guard as program practice.**
3. **New protocol row for the four-protocol table:** HEPA uses **full engine history + per-context instance norm + K=150**; CAFE uses **window=30 + train-stat z-score + K=125**. Add to the table as the *context* dimension — it is probably larger than the head effect.
4. **Correct two App-H facts:** HEPA wins **6 of 7** (not 8 of 9); TEP excluded because *"the public MTS-JEPA release does not include a chemical-process benchmark"* (not encoder non-convergence).
5. **Record the iterability answer**: dimensionally yes (`ℝ²⁵⁶`), semantically no (bidirectional-future target vs causal-past input). **Retarget the rollout experiment from "HEPA-iterated" to "HEPA-extrapolated at Δt > K"** — cheaper, fairer, and not a straw man.
6. **HEPA wins only 4/9 in Table 5** — soften any "beat the leader" framing; the matched-head protocol is much less flattering to HEPA than Table 1.
7. **MTS-JEPA → SC-JEPA rename + SDM 2026** — update the citation, and flag the **soft-codebook / discrete-regime-code novelty collision** with CLU's multi-basin story to the theorist. Useful ammunition: SC-JEPA's codebook *loses* on C-MAPSS-2 (0.53 vs 0.57), so discrete regime codes alone don't crack multi-regime.
8. **⛔ Tooling:** this scout has **no Bash**, so HEPA-SP is still unread after two waves. **Assign the HEPA-SP read to an engineer-type agent** (has Bash + clone auth), targeting `hepa/model/` per the import path in `hepa_model.py`. Also: **no poppler on this machine** — PDF page rendering is unavailable to all agents; use ar5iv.
9. **Two questions for the Head (both one-liners, both unblock more than a scout wave could):** (a) *does HEPA's C-MAPSS loader read `RUL_FD00x.txt`, and does it truly use full engine history?* (b) *what are HEPA's finetuning optimizer/lr/epochs, and are all published numbers baseline HEPA rather than HEPA-SP?*
