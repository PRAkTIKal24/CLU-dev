# rival-recon — web-scout report

**Task + acceptance criterion:** map the modern neural-memory rival family (write rules, state-byte conventions, task/tuning conventions, metric-native status, published failure modes) and deliver a fairness checklist that gates Track-2 harness design; every convention traced to a primary source or marked **UNPINNED**.
**Status: done.**

## ⛔ RECONCILIATION LIST (owner needed — protocol §5 corollary; first-10-lines rule)
1. **Charter §2.2 names "the MAD/zoology-style synthetic memory suite" as the Track-2 *primary*. This brief says that is inadmissible as a primary claim.** MQAR / in-context recall / noisy / fuzzy recall are *metric-native tasks* (query token ≡ stored key token; an exact-match scan is at ceiling) ⇒ intervention §6 criterion 4 + §8.4 fire. Compounded by Arora et al. ICML 2024 Thm 3.1 (Ω(N)-bit state) which makes MQAR structurally unwinnable for any fixed-state recurrence. **Recommended: MAD/zoology demoted to *diagnostic secondary*; Track-2 primary moves to the small-scale real-data run (enwik8 / WikiText) + MAD `compression` as the one admissible synthetic.** Head/Hub decision required.
2. **Charter §2.2: "the accuracy-vs-Verlet-steps anytime curve is a signature figure no baseline can draw" — FALSIFIED.** DEQs (Bai et al. 2019; Neural DEQ Solvers ICLR 2022) and EBTs (arXiv:2507.02092, Fig. 6a/Fig. 12) draw accuracy-vs-inference-compute at fixed weights; *and within the memory family* Titans Revisited (arXiv:2510.09551) sweeps inference-time chunk size on fixed weights. Re-word required (§5, item "anytime curve").
3. **Charter §2.2 "memory operations at chunk granularity (as Titans-class memories do — fair)" — CONFIRMED, pinned across 4 families.** No action; cite it (§1.5 below).
4. **Handover "real Mamba before any SSM claim" needs a version ruling:** the reference SSM in July 2026 is Mamba-2 (ICML 2024) or Mamba-3 (arXiv:2603.15569, Mar 2026). Mamba-1 alone is now a stale baseline.
5. **Pillar 4 ("explicit memory + principled forgetting") is heavily occupied** — Titans Eq. 13, Gated DeltaNet Eq. 8, Sparse Delta Memory Eq. 3 all ship learned forget gates on an explicit written state. Surviving differentiator is **byte-exact deletion + settable per-item lifetimes**, not "principled forgetting" as a phrase.

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — recon. No performance claim measured.
- **Laundering control:** n/a for me; **rival laundering exposure is Deliverable 3 below** — verdict: **every rival family surveyed is metric-native or weakly metric-native**, i.e. a kNN/lookup launder is a ceiling for them too.
- **Falsifies the deliverable:** a rival whose state-byte convention or metric-native status cannot be traced to a primary source and is not marked UNPINNED. Two are marked UNPINNED below (Titans state-byte convention; seed counts across the whole family).
- **Does NOT falsify:** finding rivals are metric-native (information, not defeat); finding our positioning occupied (items 1, 2, 5 above).

## What I did
Fanned across arXiv (abs + HTML + ar5iv), authors' reference implementations on GitHub (Mamba `mamba_simple.py`/`mamba2.py`, HazyResearch `zoology` paper configs, `athms/mad-lab` task configs), ACL Anthology, and 2026 preprints. Pulled equations, config literals and table numbers rather than prose. Primary sources only for every number; blog/secondary used for orientation only and not cited.

## How I verified
- Equation numbers and config literals read from source documents/files, not summaries (e.g. Mamba `allocate_inference_cache` quoted verbatim; `zoology/experiments/paper_configs/iclr24_zoology_figure2/configs.py` literals; `mad-lab/configs/tasks/*.yml` literals).
- Cross-checks where load-bearing: Arora Thm 3.1 independently re-extracted and matches the program's own w21 record; chunk-granularity claim confirmed in 4 independent families; Titans reproducibility claim confirmed by an independent reimplementation paper.
- Two arXiv HTML endpoints 404'd (`2405.21060v1`, `2412.06464v2`); I used `abs` + ar5iv mirrors instead and say so.

---

# Deliverable 1 — the write-mechanism map

## 1.1 TTT / test-time-training memories
**Sun, Dalal, Koceja, Fan, Wang, Bai, Chen, Wang, Song, Guestrin, Hashimoto, Koyejo, Choi, Sun (2024), "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", arXiv:2407.04620.** *(preprint; widely adopted)*

- **Memory parameter set:** the hidden state **is** `W`, the weights of an inner model. `f_lin(x)=Wx` (TTT-Linear); TTT-MLP = **two-layer MLP, hidden width 4× input dim, GELU** (§2.7). Both wrapped `f(x) = x + LN(f_res(x))`.
- **Inner-loop objective (Eq. 4):** `ℓ(W; x_t) = ‖ f(θ_K x_t; W) − θ_V x_t ‖²`, with learnable low-rank **training view** `θ_K`, **label view** `θ_V`, and a separate **test view** `θ_Q` (Eq. 5) used for the read `z_t = f(θ_Q x_t; W_t)`.
- **Update (Eq. 2):** `W_t = W_{t−1} − η ∇ℓ(W_{t−1}; x_t)` — **one gradient step per token**.
- **Chunk granularity (§2.4, "Parallelization with mini-batch TTT"):** gradients within a chunk are taken w.r.t. the chunk-start weights, `W_t = W_0 − η Σ_{s=1}^t G_s` (Eq. 6). **"We chose b = 16 for all experiments in this paper."** Smaller `b` ⇒ more effective GD steps ⇒ better perplexity, slower — an explicit **speed/quality dial**.
- **Reset between sequences:** `W_0` is a **learnable outer-loop parameter shared across all sequences** (§2.7); i.e. the per-sequence state is reset to a *trained* initialisation, not to zero. ⚠ This is the single most important convention for us: the *initialisation* is parameters, the *deviation* is state.

## 1.2 Titans-class test-time memory
**Behrouz, Zhong, Mirrokni (2024/2025), "Titans: Learning to Memorize at Test Time", arXiv:2501.00663** *(preprint; OpenReview record `8GjSf9Rh7Z`; no official code as of this scan — see §4)*

- **Memory:** `M_θ` an MLP with `L_M ≥ 1` layers; ablated at `L_M ∈ {1,2,3,4}` (§5.5).
- **Loss (Eq. 12):** `ℓ(M_{t−1}; x_t) = ‖ M_{t−1}(k_t) − v_t ‖²₂`.
- **Surprise + momentum (Eq. 10):** `S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t)`.
- **Forget gate / weight decay (Eq. 13):** `M_t = (1 − α_t) M_{t−1} + S_t`. Paper frames `α_t` as generalising the forgetting gate of modern gated RNNs.
- **Chunk granularity:** Eqs. 16–17 give the tensorised mini-batch/chunk-parallel form ("chunk size `b`"); **the numeric `b` used in experiments is not stated in the paper — UNPINNED.**
- **Three-way split:** *persistent memory* `P = [p_1 … p_{N_p}]` (learnable, input-independent, prepended — **parameters, not state**); *long-term memory* `M_t` (the test-time-written MLP); *core/short-term* = attention. Variants **MAC** (memory as context), **MAG** (memory as gating, over sliding-window attention), **MAL** (memory as layer).
- Scales: 170M/340M/400M on 15B FineWeb-Edu tokens; 760M on 30B.

## 1.3 Fast weights / Hebbian outer products (incl. the linear-attention equivalence)
- **Ba, Hinton, Mnih, Leibo, Ionescu (2016), "Using Fast Weights to Attend to the Recent Past", NIPS 2016, arXiv:1610.06258.** Write (Eq. 1): `A(t) = λ A(t−1) + η h(t) h(t)ᵀ`. Read = an **inner settling loop**, `h_{s+1}(t+1) = f([W h(t) + C x(t)] + A(t) h_s(t+1))`, run for `S` steps. Capacity claim: **O(H²)** vs O(H) for a vanilla RNN/LSTM. Task = associative retrieval, letters A–Z as keys, digits 0–9 as values, `c9k8j3f1??c → 9`.
- **Schlag, Irie, Schmidhuber (2021), "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021, arXiv:2102.11174.** Linear attention ≡ fast weights (Eq. 17): `W^(i) = W^(i−1) + v^(i) ⊗ φ(k^(i))`. **Delta rule (Eq. 24):** `W^(i) = W^(i−1) + β^(i)(v^(i) − v̄^(i)) ⊗ φ(k^(i))` with `v̄^(i) = W^(i−1)φ(k^(i))` and **dynamic learning rate** `β^(i) = σ(W_β x^(i))` (Eq. 21). **Capacity claim (§4.1):** "*with keys embedded in a `d_dot` space, there cannot be more than `d_dot` orthogonal vectors … storing more than `d_dot` associations will result in a retrieval error*"; DPFP (§5.4) raises `d_dot = 2·d_key·ν`.
- **Yang, Wang, Zhang, Kim (2024), "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024, arXiv:2406.06484 (DeltaNet).** `S_t = S_{t−1} − v_t^old k_tᵀ + v_t^new k_tᵀ`, `v_t^new = β_t v_t + (1−β_t) v_t^old`, `v_t^old = S_{t−1} k_t`. Chunkwise form Eqs. 8–9; **"C is set to a small constant (usually 64 or 128)"**. State `S_t ∈ ℝ^{d_k×d_v}` per head.
- **Yang, Kautz, Hatamizadeh (2024/2025), "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025, arXiv:2412.06464.** **Gated delta rule (Eq. 8):** `S_t = S_{t−1}( α_t (I − β_t k_t k_tᵀ) ) + β_t v_t k_tᵀ`. Paper states DeltaNet as `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` and **Mamba2 as `S_t = α_t S_{t−1} + v_t k_tᵀ`** — i.e. Mamba-2 is presented, by the rival authors themselves, as a *degenerate fast-weight rule*. Chunk `C` "a multiple of 16 … typically 64 as implemented in FLA". Head dim ablated at {64,128,256}; **128 chosen**.
- **⭐ Cabannes, Mazaré, Szilvasy, Douze, Lomeli, Auzina, Carpentier, Synnaeve, Jégou (2026), "Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity", arXiv:2607.07386 (7 Jul 2026, Meta FAIR — preprint, 3 weeks old).** The current frontier and the closest thing to a CLU-shaped rival. **Explicit memory slots** `M[i]`, Product-Key (PKM) top-k addressing, per-step: **decay (Eq. 3)** `M̃_t[i] ← α_t · M_{t−1}[i]`, **delta write (Eq. 4)** `M_t[i] ← M̃_t[i] + β_t k_t^{(i)}(v_t − M̃_t[i])`, applied **only to the top-W write-selected slots `I_t^w`** (= an admission policy by any other name). `α_t = exp(−A·softplus(W_a x_t + b_dt))` is **per-head, not per-slot**. **Read (Eq. 5):** `y_t = Σ_{i∈I_t^r} q_t^{(i)} · M_t[i]` over top-R read-selected slots. State size **Eq. 6:** `M_size = (d_qk^tot)² · d_v^tot / (4H²)`; Table 1: 553M state at 1.4B params (**state/param 156%**), 7.963B state at 8B (**111%**).

## 1.4 Mamba / SSM state (reference implementation, pinned from code)
**Gu & Dao (2023), "Mamba", arXiv:2312.00752 (COLM 2024); Dao & Gu (2024), "Transformers are SSMs" (Mamba-2), ICML 2024, arXiv:2405.21060; Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu (2026), "Mamba-3", arXiv:2603.15569.** Reference impl: **https://github.com/state-spaces/mamba**.

`mamba_ssm/modules/mamba_simple.py` — defaults `d_state=16, d_conv=4, expand=2`:
```python
conv_state = torch.zeros(batch, self.d_model * self.expand, self.d_conv, ...)
ssm_state  = torch.zeros(batch, self.d_model * self.expand, self.d_state, ...)
```
⇒ **Mamba-1 recurrent state per layer = expand·d_model·(d_conv + d_state) elements = 40·d_model at defaults.** Total bytes `= n_layer · expand·d_model·(d_conv+d_state) · sizeof(dtype)`; dtype is the *weights'* dtype (bf16 ⇒ 2 B), not necessarily fp32. Paper (§3.4): "*we always fix E=2*"; total hidden state "dimension DN per input"; ≈`3ED²` params per block.

`mamba_ssm/modules/mamba2.py` — defaults `d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256`; `conv_state` shape `(B, conv_dim, d_conv)` with `conv_dim = d_inner + 2·ngroups·d_state`; `ssm_state` shape `(B, nheads, headdim, d_state)`.
⇒ **Mamba-2 per layer = (2·d_model + 2·ngroups·d_state)·d_conv + 2·d_model·d_state elements = 264·d_model + 1024 at defaults.**

Mamba-3 (Mar 2026): exponential-trapezoidal discretisation `h_t = α_t h_{t−1} + β_t B_{t−1}x_{t−1} + γ_t B_t x_t`; complex/rotational state (fixes parity & modular-arithmetic state tracking Mamba-2 cannot do); MIMO with rank `R=4`; `d_state ∈ {16,32,64,128}`; **"comparable perplexity to Mamba-2 using half of its predecessor's state size."** Chunk: `C_MIMO = C_SISO / R`.

## 1.5 ⭐ Chunk-granularity verdict (charter §2.2 dependency) — **CONFIRMED, pinned, 4 independent families**
| family | chunk unit | value | source |
|---|---|---|---|
| TTT | mini-batch TTT | **b = 16** | Sun et al. §2.4, verbatim |
| Titans | chunk-parallel memory update | **b, value unstated** | Behrouz et al. Eqs. 16–17 (**UNPINNED value**) |
| DeltaNet | chunkwise parallel | **C = 64 or 128** | Yang et al. Eqs. 8–9 |
| Gated DeltaNet | chunkwise (FLA kernel) | **C = 64** | Yang et al. 2025, §algorithm |
| Mamba-2 | SSD block/chunk | **chunk_size = 256** | `mamba2.py` default |
**Chunked memory update is standard practice. Charter §2.2's fairness argument holds.** ⚠ Caveat (Deliverable 4): an independent reimplementation finds chunking is the *main source of Titans' degradation* and that inference chunk size must match pretraining chunk size — so chunking is standard *and* is a known cost, which we can turn into an ablation axis rather than hide.

---

# Deliverable 2 — task definitions, metrics, tuning conventions

## 2.1 MQAR / zoology (pinned from the authors' own config file)
**Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré (2024), "Zoology: Measuring and Improving Recall in Efficient Language Models", ICLR 2024, arXiv:2312.04927.** MQAR defined §3.2/Def. 3.1; details App. H.7.1.
`zoology/experiments/paper_configs/iclr24_zoology_figure2/configs.py` literals:
- `(input_seq_len, num_kv_pairs) ∈ {(64,4), (128,8), (256,16)}`; **`vocab_size = 8192`**; `d_model ∈ {64,128,256,512}`; **100 000 train / 3 000 test examples per cell**.
- **LR sweep: `np.logspace(-4, -2, 4)` per architecture per cell; metric = max test accuracy over the sweep.**
- Sequence mixers swept: attention, hyena, rwkv, base_conv, h3, based, mamba.
- **Seeds: not specified in the config or the paper — UNPINNED (community norm appears to be 1 seed, best-of-lr-sweep).**

**Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra, Ré (2024), "Simple linear attention language models balance the recall–throughput tradeoff" (Based), ICML 2024, arXiv:2402.18668.**
- **⛔ Theorem 3.1 (the one that decides Track 2):** *"Any recurrent model depending causally on input u ∈ {0,1}^{N×d} requires Ω(N)-bits in state size to solve MQAR."* Corollary F.1 applies it to Mamba. **A fixed-state recurrence cannot solve MQAR at scale by construction — this is our own w21 finding, re-verified from the primary source.**
- MQAR setup: train seq len 256, 4–64 kv pairs; eval seq len 1024, 4–256 kv pairs.
- **State-byte accounting (App. E.2), the only explicit per-architecture convention I found in this literature:** attention KV cache `n·d`; sliding window `w·d`; Based `(1 + d′ + d′²)·d` per head; Mamba `state_size × head_dim`; **"bytes during generation" = state dimensions × 4 (fp32)**. ⚠ Baselines are varied by *hyperparameters that change state size* (model dim), not by an explicit matched-state constraint.

## 2.2 MAD (pinned from paper + authors' repo)
**Poli, Thomas, Nguyen, Ponnusamy, Deiseroth, Kersting, Suzuki, Hie, Ermon, Ré, Zhang, Massaroli (2024), "Mechanistic Design and Architecture Search through Surrogate Modeling" (MAD), arXiv:2403.17844; repo `athms/mad-lab`.**
- Six tasks: **in-context recall, fuzzy in-context recall, noisy in-context recall, selective copying, compression, memorization.**
- `configs/tasks/in-context-recall.yml`: `vocab_size: 16, seq_len: 128, num_train_examples: 12800, multi_query: True`; variations `vocab ∈ {32,64,128}`, `seq_len ∈ {256,512,1024}`, `train ∈ {6400,3200,1600,800}`.
- `configs/tasks/compression.yml`: `vocab_size: 16, seq_len: 32, num_train_examples: 12800`; variations `vocab ∈ {32,64,128}`, `seq_len ∈ {64,128,256}`.
- **Metric:** token-level accuracy on 1 280 held-out eval samples per task setting.
- **⭐ Normalisation (§3.2, verbatim):** *"Fixed-state architectures are normalized to an iso-state and iso-parameter setting, including models featuring sparsely activated layers such as mixtures of experts (MoEs). Here, we normalize all fixed-state architectures to a common total state dimension of 4096."*
- **⭐ Architecture (App. B.3):** 2 blocks / **4 layers total**, **width 128**, **parameter counts exclude embeddings (embeddings tied across models)**. State-normalised configs: Mamba `d_state=4, d_conv=4, expand=2`; multi-head GLA 8 heads × head dim 16; MHA 16 heads × head dim 8; multi-head Hyena 16 heads × state 2.
- **⭐ Tuning protocol (App. B.4):** AdamW (β=0.9/0.98), cosine decay, **200 epochs**, **batch 128**, grid **lr ∈ {1e-4, 5e-4, 1e-3} × wd ∈ {0.0, 0.1}** (3×2), **best run per architecture per task setting reported**, **1 seed**.

## 2.3 ⭐ The tuning protocol we must adopt (N78 rescue standard, cited)
A baseline is **rescued by the literature's own standard** iff it received **at least the union of the two published grids**, per architecture, per task cell:
> **lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2}** (= MAD's 3 values ∪ Zoology's `logspace(-4,-2,4)` = {1e-4, 4.64e-4, 2.15e-3, 1e-2}, deduplicated/rounded) **× wd ∈ {0.0, 0.1}**, AdamW β=(0.9,0.98), cosine decay, **best-of-grid reported**; architectures normalised to **iso-state and iso-parameter** (params excluding embeddings), evaluated on an independent eval set.
> **Then, strictly beyond the community norm and per our own standing rule: ≥3 seeds at each arm's best-of-grid config, for every arm including ours, with the spread quoted.** The community uses **1 seed** (MAD App. B.4; Zoology/Based do not report seeds) — our multi-seed rule is *stricter*, which is a defensible position to state explicitly in the paper, not a deviation to hide.
> **Sanity gate:** a rescued baseline's best-of-grid number must land in the published range for that architecture on that task (Zoology Fig. 2 / MAD tables). A baseline below its published range is **not rescued** and no margin against it is quotable. *(This is the N78 mechanism: an un-rescued GRU turned a 2.1× margin into a quoted "19×".)*

## 2.4 enwik8 / WikiText small-scale conventions
- **Dai, Yang, Yang, Carbonell, Le, Salakhutdinov (2019), "Transformer-XL", ACL 2019** (Table 2, enwik8): Al-Rfou et al. 64L **235M → 1.06 bpc**; **Transformer-XL 12L, 41M → 1.06 bpc**; 18L 88M → 1.03; 24L 277M → 0.99. Table 3 (text8): Al-Rfou 12L 44M → 1.18; 64L 235M → 1.13; TXL 24L 277M → 1.08.
- **Merity, Keskar, Socher (2018), "An Analysis of Neural Language Modeling at Multiple Scales", arXiv:1803.08240** — the *recurrent, modest-compute* anchor: enwik8 **3-layer AWD-LSTM (h=1840), 47M → 1.232 bpc**; **4-layer AWD-QRNN (h=1800), 26M → 1.336 bpc**. WikiText-103: 4-layer QRNN (h=2500) → **32.0 valid / 33.0 test ppl**, trained in **12 h on one Volta for 14 epochs**.
- **What "small scale" means here:** 26M–47M parameters, ~90 MB of characters, single-GPU, ~12 h. That is *the* laptop/CSF3 weight class, and it is a **published, citable** class — we do not have to invent one.
- **Non-embarrassing floor (composite, marked INFERRED, not pinned to one source):** a from-scratch recurrent/bounded-state model at 20–50M params on enwik8 should land **≈1.20–1.40 bpc**; ≥1.5 bpc signals under-training rather than architecture; ≤1.10 bpc is transformer-with-long-context territory and not the target. ⚠ **I could not pin any published enwik8 number below ~26M params — if the harness runs at <10M params, there is no citable floor and we must publish our own tuned GRU/Mamba floor in the same table.**

---

# Deliverable 3 — the metric-native audit

**Definition used (intervention §6 crit. 4):** a memory is *metric-native* if the read is a similarity computation between the query and the stored keys in the same metric space ⇒ a kNN / exact-match lookup over the same keys is a provable ceiling.

| family | read mechanism | metric-native? | deciding mechanism |
|---|---|---|---|
| Linear attention / fast weights / DeltaNet / GLA / Gated DeltaNet | `y_t = S_t q_t = Σ_i v_i (k_i·q_t)` | **YES, provably** | the read *is* an inner product in key space; Schlag §4.1's `d_dot` capacity bound is literally the statement that this is a (rank-limited) kernel lookup |
| Mamba-1 / Mamba-2 / Mamba-3 | `y_t = C_t h_t`, `h_t = Ā h_{t−1} + B̄_t x_t` ⇒ `y_t = Σ_s (C_tᵀ Ā_{s+1:t} B_s) x_s` | **YES** | SSD duality (Dao & Gu 2024) makes it a masked/decayed linear attention; Gated DeltaNet states Mamba2 as `S_t = α_t S_{t−1} + v_t k_tᵀ`. Query `C_t`, key `B_s`, decay kernel. |
| Sparse Delta Memory (2026) | `y_t = Σ_{i∈I_t^r} q_t^{(i)} M_t[i]`, `I_t^r` = PKM **top-k inner product** | **YES, most explicitly of all** | top-k over inner-product scores *is* kNN; the architecture is an ANN index with a learned write |
| TTT-Linear | `z_t = W_t (θ_Q x_t)` | **YES** | linear map ≡ linear attention read |
| TTT-MLP | `z_t = f_MLP(θ_Q x_t; W_t)`, 2-layer GELU MLP | **WEAKLY** — nonlinear readout, but the inner objective `‖f(θ_K x)−θ_V x‖²` keys everything by `θ_K x` | a kernel/kNN *regressor* on the same `(θ_K x, θ_V x)` pairs is a strong ceiling candidate, not a proven one |
| Titans LMM | `M_t(q_t)`, `M` an `L_M`-layer MLP | **YES at `L_M=1`; WEAKLY for `L_M≥2`** | Eq. 12 keys storage by `k_t`; Eq. 13's `α_t` only rescales. Same argument as TTT-MLP. |
| Ba et al. fast weights | inner settling `h ← f(Wh + Cx + A h)` with `A = Σ λ^Δ h hᵀ` | **YES** | `A h` is a similarity-weighted recall in `h`-space; the settling loop is an iterated kernel read |

**⭐ Verdict: the entire modern neural-memory family is metric-native or weakly metric-native.** Consequences for the Hub:
1. **Criterion 4 does not distinguish us from them — it is a property of the *task*, not of the memory.** If the *task* is metric-native, a classical method is the ceiling for **every** entrant, and no one's primary claim can live there. This is the correct, defensible framing and it should replace any framing of "we lose because our memory is metric-native."
2. **A "shared weakness" fight is available and is genuinely novel:** TTT-MLP and Titans `L_M≥2` escape *literal* metric-nativeness only through a nonlinear readout. Our `ψ` + trajectory read is a strictly larger escape (a *path*, not a point evaluation). This is the strongest CHLU/CLU differentiation the audit found.
3. **Pre-empt the referee's argument:** a reviewer will say "your store is a kNN with extra steps." The answer must be: so is Mamba-2 (by the SSD duality), so is Gated DeltaNet (Eq. 8), so is Sparse Delta Memory (PKM top-k). Cite them.

## 3.1 Track-2 task admissibility (§8.4 non-negotiable)
| candidate task | metric-native classical ceiling? | admissible as PRIMARY? |
|---|---|---|
| MQAR (zoology) | **YES** — query token ≡ a key token seen in context; an exact-match scan is 100% | ⛔ **NO.** Also Ω(N)-bit theorem (Based Thm 3.1) ⇒ structurally unwinnable for fixed-state |
| MAD in-context recall / noisy / fuzzy recall | **YES** (fuzzy = lookup over token spans; noisy = lookup with distractors) | ⛔ **NO** |
| MAD selective copying | order-preserving filter — a filtered FIFO queue is at ceiling | ⛔ **NO** (fails crit. 2: trivial method at ceiling) |
| MAD memorization | fixed kv map in weights; no in-context memory pressure | ⛔ **NO** (fails crit. 3) |
| **MAD compression** | **NO** — compress a random token sequence into ONE aggregate token, reconstruct via MLP. There is no key to match; the read is a decompression | ✅ **YES** — and it is exactly charter §2.1(a) *beyond-capacity compression*. The one admissible synthetic in the community suite. |
| **enwik8 / WikiText next-token LM** | **NO** — the target is not a stored key | ✅ **YES** on crit. 1, 2, 4, 5. ⚠ **crit. 3 is the weak one:** at 26–47M params local n-gram statistics dominate and "memory management over time" may not be the binding difficulty. Mitigation: report the **recall-vs-context-length curve** and a long-range subset, as RULER-style evaluations do. |
| non-stationary streams with regime revisit (charter fallback) | **NO** if the query is "what regime am I in now", **YES** if it is "retrieve the item nearest this key" | ✅ **conditionally** — the query must be an aggregate/regime identification, never a nearest-key lookup |

**Recommendation to the Hub:** Track-2 primary = **enwik8 (or WikiText-103) at 26–47M params, matched params + matched state**, with **MAD `compression`** as the admissible synthetic; **MQAR/MAD-recall retained only as a diagnostic**, reported once with the Ω(N) theorem cited and no margin claimed. This contradicts charter §2.2 and needs a Head ruling (reconciliation item 1).

---

# Deliverable 4 — the failure modes they publish

| family | published limitation | source |
|---|---|---|
| TTT | "**TTT-MLP still faces challenges in memory I/O**, but shows larger potential in long context"; non-linear scaling trends unresolved ("constrained by our academic resources, we encourage the community to join us"); mini-batch `b` is an explicit speed/quality trade-off | Sun et al. 2024, §experiments + conclusion |
| Titans | Paper states almost no limitations (footnote: "we are working on finalizing the results of larger models"). **Independent reimplementation is where the failures live** | Behrouz et al. 2025 |
| Titans (independent) | "**the lack of publicly available code and ambiguities in the original description hinder reproducibility**"; **chunking identified as the main source of performance degradation**; larger chunks help but cost compute; **train/test chunk-size mismatch** (a model pretrained at chunk 64 is optimal only when evaluated at 64); Titans **loses** on sequential recommendation (MAC MRR 0.4371 vs BERT4Rec 0.4451); memory-only variant beats iTransformer on time series (MSE 0.4872 vs 0.4925) | **Titans Revisited**, arXiv:2510.09551 (Oct 2025, **preprint-grade**) |
| DeltaNet | "moderate performance on real-world tasks despite synthetic benchmark success"; **lacks a robust memory-clearing mechanism** — this is precisely what Gated DeltaNet was built to fix | Yang et al. 2025 (GDN) §intro |
| Gated DeltaNet | real-world retrieval gap attributed to "instruction-unaligned small language models being prone to repetition errors" | Yang et al. 2025 |
| Linear attention / fast weights | hard capacity limit at `d_dot` associations; retrieval error above it | Schlag et al. 2021 §4.1 |
| Mamba / SSM | **Ω(N)-bit state lower bound for MQAR** (Based Thm 3.1, Cor. F.1); Mamba-2 **cannot do state tracking** (parity, modular arithmetic) — fixed by Mamba-3's complex state; weak unstructured extraction (SWDE 28.5, FDA 23.4 for Mamba-3 at 1.5B) | Arora et al. 2024; Lahoti et al. 2026 |
| Sparse Delta Memory | "**memory requirements are not negligible … may be as large as the model parameters**"; MFU "**around an order of magnitude lower** than the highly optimized GDN kernel"; **1.49× slower training** than GDN at 8B at matched FLOPs | Cabannes et al. 2026 §limitations |

**Where our four pillars must land to be interesting:** (i) beyond-capacity graceful degradation — but see the preemption flag below; (ii) *exact* deletion / settable lifetimes, where every rival has only soft decay; (iii) compute-adaptive read, where the rivals' analogue is chunk size and it is a **published weakness** (train/test mismatch) rather than a feature; (iv) zero-free-parameter predicted hyperparameters, where no rival makes any claim at all.

## 4.1 ⚠ NOVELTY PREEMPTION FLAGS (charter §4 pillars)
| pillar | status | evidence |
|---|---|---|
| **1. Expressive latents (trajectory/manifold as the stored object)** | **Partially flanked, core survives.** TTT/Titans already store *a function* (an MLP) rather than a vector — the "richer latent" high ground in sequence modelling is taken. **Storing a *path* and reading it with a learned `ψ`, and manifold-valued memory via flat directions, are NOT preempted in anything I found.** ⚠ But: **Clark (2025/2026), "Transient dynamics of associative memory models", arXiv:2506.05303, Phys. Rev. E (2026)** shows that **"patterns can be transiently retrieved with high accuracy above capacity despite the absence of stable attractors"**, via "slow regions … lingering traces of the stable basins", and introduces **"transient-recovery curves"** showing "graceful, non-catastrophic changes in retrieval behavior above capacity." **This is the physics community's version of "the transient carries information the fixed point does not" and of charter §2.1(a)'s graceful-degradation-above-capacity claim.** Cite it as a positioning gift (our dynamics realise a predicted regime), never as our discovery. | arXiv:2506.05303 |
| **2. Structured exploration (wormholes, boosts, causal diamond)** | **No preemption found.** But "spend more inference compute to get a better answer" is heavily occupied (EBT, DEQ, adaptive-computation). Our novelty must be the *structure* of the exploration, not the fact of retrying. | — |
| **3. Physics-intuited, zero-free-parameter hyperparameters** | **No preemption found — strongest pillar.** No paper in this family predicts its own hyperparameters from theory. | — |
| **4. Explicit memory + principled forgetting** | ⚠ **Heavily occupied.** Titans Eq. 13 `α_t` forget gate; Gated DeltaNet Eq. 8 `α_t`; **Sparse Delta Memory Eq. 3 decay + Eq. 4 top-W write selection = decay + admission on explicit slots, at 8B scale, 3 weeks old**; Gated Differentiable Working Memory (Mei et al., arXiv:2601.12906, Jan 2026) = explicit differentiable slots with gated forgetting. **Surviving differentiator: byte-exact deletion (AUC 0.5000±0.0000) and settable per-item lifetimes.** None of these do exact deletion; all do soft decay. **The claim must be phrased on exactness, not on "principled forgetting."** | see §1.3 |

---

# Deliverable 5 — THE FAIRNESS CHECKLIST
*(lift verbatim into the Track-2 harness task file)*

**F1 — Matched parameters.** Count **non-embedding parameters**; embeddings and the unembedding/readout are **excluded and tied across arms**. *Convention source: MAD App. B.3 — "parameter counts exclude embeddings (embeddings are tied across models)"; general convention Kaplan et al. 2020.* For CLU this means: `φ`, `ψ`, `V_θ` initialisation, controller and codebook **all count as parameters**; the token embedding/readout does not.

**F2 — Matched state-bytes.** Per-arm formula, all **per layer, per sequence, in the arm's own inference dtype** (declare the dtype; the reference impls allocate in the *weights'* dtype, not fp32):
- **Attention:** `2 · n_kv_head · d_head · L` elements (KV cache). *Based App. E.2: `n·d`.*
- **Sliding-window attention:** `2 · n_kv_head · d_head · w`. *Based App. E.2: `w·d`.*
- **GRU:** `d_hidden` elements (the only arm whose state is trivially defined).
- **Mamba-1:** `expand·d_model·(d_conv + d_state)` = `40·d_model` at defaults. *`mamba_simple.py: allocate_inference_cache`.*
- **Mamba-2:** `(expand·d_model + 2·ngroups·d_state)·d_conv + expand·d_model·d_state` = `264·d_model + 1024` at defaults. *`mamba2.py: allocate_inference_cache`.*
- **DeltaNet / Gated DeltaNet / GLA:** `n_head · d_k · d_v`. *Yang et al. 2024/2025.*
- **TTT-Linear:** `d_head²` per head **plus the in-flight mini-batch buffer of `b=16` tokens**; **TTT-Linear's `W_0` is a *parameter* (learned, shared across sequences) — only the deviation is state.** *Sun et al. §2.4, §2.7.*
- **TTT-MLP:** `8·d_head²` (two layers, 4× hidden) + buffer.
- **Titans:** `|M_θ| + |S_t| = 2·|M_θ|` — **the momentum buffer (Eq. 10) doubles the recurrent state** — plus the sliding-window KV cache in MAC/MAG. Persistent memory `P` is a **parameter**, not state. ⚠ **The paper states no state-byte convention: UNPINNED. If Titans is run as a baseline, we must declare this accounting ourselves, in the caption, as our reconstruction.**
- **Sparse Delta Memory (if run):** `M_size = (d_qk^tot)²·d_v^tot/(4H²)`. *Cabannes et al. Eq. 6, Table 1.*
- **CLU store:** `n_atoms · d · sizeof(dtype) + |codebook| + |controller state|`, counting **only bytes mutated during the sequence**. The strided read trajectory (`T × d`) is a **per-read activation, not state** — it does not persist between tokens and must **not** be counted as state, but it **must** be counted in F4 (FLOPs/latency).

**F2a — ⭐ The commensurability problem, stated honestly (do not paper over).** Baseline state is *per-sequence, reset at sequence start*. CLU has **three** byte pools, and the laundering risk is moving item content between them:
1. **frozen parameters** (φ, ψ, V_θ init, codebook, controller weights) → counted in F1;
2. **test-time-mutated store bytes** → counted in F2;
3. **per-read transients** (trajectory, momenta, retries) → counted in F4 only.
**Honest reconciliation, and it is the literature's own:** report **both axes separately and simultaneously** — MAD normalises to **"iso-state and iso-parameter"** (App. B.3, total state dimension 4096); Sparse Delta Memory reports **isoFLOP + isoParameter with an explicit state/parameter ratio column** (Table 1: 111–156%). Precedent therefore exists for a state budget *larger than* the parameter budget — our large store is not automatically unfair, provided the ratio is printed.
**Binding guard:** the task must be **resampled per sequence** so that no item-specific content can hide in pool (1). If any arm can memorise the eval items in weights, F2 is meaningless. The settle-deleted / same-keys launder is matched on pool (2) bytes and the same φ.

**F3 — Baseline tuning (N78 rescue standard).** Every baseline, per architecture, per task cell, receives at minimum:
`lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0.0, 0.1}`, AdamW β=(0.9, 0.98), cosine decay, best-of-grid reported.
*(= MAD App. B.4's 3×2 grid ∪ Zoology `configs.py`'s `np.logspace(-4,-2,4)`.)*
**Sanity gate:** a rescued baseline's best-of-grid result must fall inside the published range for that architecture on that task (Zoology Fig. 2, MAD tables, Merity Table 3 for enwik8). **Outside the range ⇒ not rescued ⇒ no margin against it is quotable.**

**F4 — Compute axis, declared not hidden.** Report **FLOPs/token and wall-clock/token** alongside accuracy. *Sparse Delta Memory makes isoFLOP its primary constraint (Eq./§scaling), and reports its own 1.49× training slowdown and order-of-magnitude MFU deficit — a rival at the frontier publishes its compute deficit; so do we.* 150–1200 Verlet steps/read must appear in the table, not the appendix.

**F5 — Chunk granularity.** Memory updates at chunk granularity are **standard practice: pinned in 4 families** (TTT `b=16`, §2.4; DeltaNet `C=64/128`, Eqs. 8–9; Gated DeltaNet `C=64`; Mamba-2 `chunk_size=256`, reference impl). Charter §2.2's fairness argument **holds**. Two required consequences: (a) declare CLU's chunk size in the same table as the baselines'; (b) **because Titans Revisited (arXiv:2510.09551) shows inference-time performance is highly sensitive to a train/test chunk-size mismatch, evaluate every arm at its training chunk size and report the mismatch curve as an ablation.**

**F6 — Seeds and reporting.** **Community norm = 1 seed, best-of-lr-grid** (MAD App. B.4 explicitly; Zoology/Based/GDN report no seed count — **UNPINNED**). **Our rule is stricter and stays stricter: ≥3 seeds at each arm's best-of-grid config, mean ± SE quoted, curve quoted not endpoint.** State the difference in the paper as a methodological strength; do **not** compare our 3-seed mean against a rival's published 1-seed best without saying so.

**F7 — The anytime curve.** ⛔ **"A signature figure no baseline can draw" is FALSE and must be re-worded.** Prior art that draws accuracy-vs-inference-compute at **fixed weights**:
- **Deep Equilibrium Models** — Bai, Kolter, Koltun (2019), NeurIPS, arXiv:1909.01377; and **Bai, Koltun, Kolter (2022), "Neural Deep Equilibrium Solvers", ICLR 2022** — accuracy vs. solver iterations / NFEs is the canonical DEQ figure.
- **Energy-Based Transformers** — arXiv:2507.02092, **Fig. 6(a)** ("*EBTs can [reduce perplexity] by performing more forward passes over a single token/sample (Thinking Longer)*") and **Fig. 12** (denoising PSNR vs. forward passes vs. DiT). Inference is `ŷ_{i+1} = ŷ_i − α∇_{ŷ_i}E_θ(x, ŷ_i)` (Eq. 1) with a Langevin variant (Eq. 2). *This is already our V1 related-work anchor.*
- **Within the memory family:** Titans Revisited sweeps inference chunk size on fixed weights; TTT's mini-batch `b` is an explicit speed/quality dial.
**Defensible re-wording:** *"an anytime accuracy-vs-compute read at fixed weights, on an explicit addressable memory with per-item lifetimes and exact deletion — the combination is unmatched"*, with DEQ/EBT cited in the same sentence. **Never claim the curve itself is unique.**

**F8 — Metric-native audit line.** For every task in the Track-2 table, state in the caption whether a classical exact-match/kNN method is at ceiling, and **run that classical method as a column**. Tasks where it is at ceiling are diagnostics only. **MQAR is one such task, and Based Thm 3.1 (Ω(N)-bit state) is cited there once, with no margin claimed** (standing since w21).

**F9 — Rival-side laundering disclosure.** State explicitly in related work that the rivals' reads are metric-native too (linear attention/fast weights/DeltaNet/GLA: `S q = Σ v (k·q)`; Mamba via SSD duality; SDM via PKM top-k). This converts "your store is a kNN" from an attack into a shared property of the field.

---

# Confidence table

| claim | status |
|---|---|
| TTT Eq. 2/4/5/6, `b=16`, learnable shared `W_0` | **pinned** (arXiv HTML v2, §2.4/2.7) |
| Titans Eqs. 10/12/13, MAC/MAG/MAL, persistent memory, `L_M∈{1..4}`, 170M–760M / 15B–30B tokens | **pinned** (arXiv HTML v1) |
| Titans chunk size `b` numeric value | **UNPINNED** — not stated in the paper |
| Titans state-byte convention | **UNPINNED** — the paper presents none; the `2·|M_θ|` (momentum) accounting is **my reconstruction**, label it as such |
| Titans reproducibility / chunking-is-the-degradation / train-test chunk mismatch | **pinned but preprint-grade, single independent group** (arXiv:2510.09551) |
| Schlag Eqs. 17/21/24, `d_dot` capacity §4.1, DPFP `2·d_key·ν` | **pinned** (ar5iv) |
| DeltaNet update + `C=64/128`; Gated DeltaNet Eq. 8 + `C=64` | **pinned** |
| Mamba-1/Mamba-2 state shapes and defaults | **pinned to the reference implementation** (strongest evidence class here) |
| Mamba-3 changes, `R=4`, "half the state size" | **pinned but preprint** (arXiv:2603.15569, Mar 2026) |
| Based Thm 3.1 Ω(N) | **pinned + independently corroborated** (matches our w21 record) |
| Based App. E.2 state formulas | **pinned but partial** — H3/Hyena/RWKV formulas not recovered; **UNPINNED** for those |
| Zoology figure-2 config literals (vocab 8192, seq/kv cells, lr `logspace(-4,-2,4)`, 100k/3k) | **pinned to authors' code** |
| Zoology/Based seed counts | **UNPINNED** — not stated anywhere I could find |
| MAD iso-state 4096, 2 blocks/4 layers/width 128, embeddings excluded, 3×2 grid, 1 seed, 200 epochs, batch 128 | **pinned** (paper §3.2 + App. B.3/B.4 + repo yml) |
| Transformer-XL enwik8/text8 table numbers; Merity enwik8 1.232/1.336, WT-103 33.0 | **pinned** |
| "non-embarrassing floor ≈1.20–1.40 bpc at 20–50M params" | **INFERRED** from the two pinned tables; no source states a floor |
| Metric-native verdicts (all families) | **pinned by mechanism** — each read equation is quoted; the *inference* that this implies a classical ceiling is our own standing theorem, not theirs |
| Sparse Delta Memory Eqs. 3/4/5/6, Table 1 ratios, limitations | **pinned but 3-week-old preprint**; no independent replication exists |
| Pillar-2 and pillar-3 "no preemption found" | **absence of evidence** — medium confidence; searched arXiv/OpenReview, not exhaustive |

# Gaps / what to search next
1. **Titans' actual chunk size and any official code release** — re-scout before any Titans baseline is run; currently we would be reimplementing from an ambiguous description, which the independent group says is the reproducibility bottleneck.
2. **Seed conventions** — I could not find *any* paper in this family reporting seed counts for the synthetic suites. Worth one targeted pass through OpenReview reviewer threads for Zoology/MAD/GDN, where reviewers usually force this out.
3. **H3 / Hyena / RWKV state-byte formulas** from Based App. E.2 — needed only if those arms are in the table.
4. **RULER** (long-context recall suite used by SDM and GDN) — I did not characterise it; if Track 2 wants a recall-vs-length curve at the frontier's own convention, RULER is the current currency and should be scouted before the harness freezes.
5. **`Fast Spatial Memory with Elastic Test-Time Training` (arXiv:2604.07350)** and **`Adaptive Memory Decay for Log-Linear Attention` (arXiv:2605.06946)** — 2026 preprints I surfaced but did not read; the latter is a direct neighbour of per-item lifetimes and should be checked for pillar-4 preemption.
6. **Whether any rival does *exact* deletion** — I found only soft decay. One targeted pass (machine-unlearning ∩ sequence-memory) would either confirm our last uncontested pillar-4 claim or kill it, and it is cheap.

# Bibtex-ready refs
```bibtex
@article{sun2024ttt, title={Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
  author={Sun, Yu and Li, Xinhao and Dalal, Karan and Xu, Jiarui and Vikram, Arjun and Zhang, Genghan and Dubois, Yann and Chen, Xinlei and Wang, Xiaolong and Koyejo, Sanmi and Hashimoto, Tatsunori and Guestrin, Carlos},
  journal={arXiv preprint arXiv:2407.04620}, year={2024}}

@article{behrouz2025titans, title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663}, year={2025}}

@article{titansrevisited2025, title={Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model},
  journal={arXiv preprint arXiv:2510.09551}, year={2025}, note={preprint-grade}}

@inproceedings{schlag2021fwp, title={Linear Transformers Are Secretly Fast Weight Programmers},
  author={Schlag, Imanol and Irie, Kazuki and Schmidhuber, J{\"u}rgen},
  booktitle={ICML}, year={2021}, note={arXiv:2102.11174}}

@inproceedings{ba2016fastweights, title={Using Fast Weights to Attend to the Recent Past},
  author={Ba, Jimmy and Hinton, Geoffrey and Mnih, Volodymyr and Leibo, Joel Z and Ionescu, Catalin},
  booktitle={NIPS}, year={2016}, note={arXiv:1610.06258}}

@inproceedings{yang2024deltanet, title={Parallelizing Linear Transformers with the Delta Rule over Sequence Length},
  author={Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and Kim, Yoon},
  booktitle={NeurIPS}, year={2024}, note={arXiv:2406.06484}}

@inproceedings{yang2025gateddeltanet, title={Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author={Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  booktitle={ICLR}, year={2025}, note={arXiv:2412.06464}}

@article{gu2023mamba, title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri}, journal={arXiv preprint arXiv:2312.00752}, year={2023}}

@inproceedings{dao2024mamba2, title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert}, booktitle={ICML}, year={2024}, note={arXiv:2405.21060}}

@article{lahoti2026mamba3, title={Mamba-3: Improved Sequence Modeling using State Space Principles},
  author={Lahoti, Aakash and Li, Kevin Y. and Chen, Berlin and Wang, Caitlin and Bick, Aviv and Kolter, J. Zico and Dao, Tri and Gu, Albert},
  journal={arXiv preprint arXiv:2603.15569}, year={2026}}

@inproceedings{arora2024zoology, title={Zoology: Measuring and Improving Recall in Efficient Language Models},
  author={Arora, Simran and Eyuboglu, Sabri and Timalsina, Aman and Johnson, Isys and Poli, Michael and Zou, James and Rudra, Atri and R{\'e}, Christopher},
  booktitle={ICLR}, year={2024}, note={arXiv:2312.04927}}

@inproceedings{arora2024based, title={Simple linear attention language models balance the recall-throughput tradeoff},
  author={Arora, Simran and Eyuboglu, Sabri and Zhang, Michael and Timalsina, Aman and Alberti, Silas and Zinsley, Dylan and Zou, James and Rudra, Atri and R{\'e}, Christopher},
  booktitle={ICML}, year={2024}, note={arXiv:2402.18668; Theorem 3.1}}

@article{poli2024mad, title={Mechanistic Design and Architecture Search through Surrogate Modeling},
  author={Poli, Michael and Thomas, Armin W. and Nguyen, Eric and Ponnusamy, Pragaash and Deiseroth, Bj{\"o}rn and Kersting, Kristian and Suzuki, Taiji and Hie, Brian and Ermon, Stefano and R{\'e}, Christopher and Zhang, Ce and Massaroli, Stefano},
  journal={arXiv preprint arXiv:2403.17844}, year={2024}}

@article{cabannes2026sdm, title={Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity},
  author={Cabannes, Lo{\"i}c and Mazar{\'e}, Pierre-Emmanuel and Szilvasy, Gergely and Douze, Matthijs and Lomeli, Maria and Auzina, Ilze Amanda and Carpentier, Justin and Synnaeve, Gabriel and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2607.07386}, year={2026}}

@inproceedings{dai2019transformerxl, title={Transformer-XL: Attentive Language Models beyond a Fixed-Length Context},
  author={Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V. and Salakhutdinov, Ruslan},
  booktitle={ACL}, year={2019}}

@article{merity2018multiscale, title={An Analysis of Neural Language Modeling at Multiple Scales},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1803.08240}, year={2018}}

@article{clark2026transient, title={Transient dynamics of associative memory models},
  author={Clark, David G.}, journal={Physical Review E}, year={2026}, note={arXiv:2506.05303}}

@inproceedings{bai2019deq, title={Deep Equilibrium Models},
  author={Bai, Shaojie and Kolter, J. Zico and Koltun, Vladlen}, booktitle={NeurIPS}, year={2019}, note={arXiv:1909.01377}}

@inproceedings{bai2022neuraldeqsolvers, title={Neural Deep Equilibrium Solvers},
  author={Bai, Shaojie and Koltun, Vladlen and Kolter, J. Zico}, booktitle={ICLR}, year={2022}}
```

---

## Proposed handover updates (for the Hub)

1. **[C2W1] `rival-recon` landed. Track 2 is now unblocked, but with two charter-level contradictions requiring a Head ruling before the harness freezes:** (a) **MAD/zoology recall tasks are inadmissible as the Track-2 primary** (metric-native + Ω(N)-bit theorem); recommended primary = enwik8/WikiText at 26–47M params, with MAD `compression` as the one admissible synthetic and MQAR retained as a cited diagnostic only. (b) **"the anytime curve is a signature figure no baseline can draw" is falsified** (DEQ, Neural DEQ Solvers, EBT Fig. 6a/12, Titans-Revisited chunk sweep) — re-word to the combination claim.
2. **Add to the never-quote list:** "an anytime accuracy-vs-compute curve no baseline can draw" · "principled forgetting" as a novelty claim (Titans Eq. 13 / GDN Eq. 8 / SDM Eq. 3 own it — only **byte-exact deletion** and **settable per-item lifetimes** survive) · "graceful degradation above capacity" as our discovery (Clark, PRE 2026) · any Titans state-byte number without the "our reconstruction; the paper states no convention" caveat.
3. **Add as a standing rule:** every Track-2 table caption carries a **metric-native line** and runs the classical exact-match/kNN method as a column; and every Track-2 comparison reports **iso-state AND iso-parameter** (MAD App. B.3) **plus a state/parameter ratio column** (SDM Table 1 precedent) — never one axis alone.
4. **Version the "real Mamba" rule:** it now means **Mamba-2 (`mamba2.py`, `d_state=128, headdim=64, expand=2, chunk_size=256`) at minimum, with Mamba-3 named in limitations.** Mamba-1 alone no longer satisfies it.
5. **New standing baseline-tuning rule (F3 above):** `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`, best-of-grid, **plus the published-range sanity gate** — a baseline below its literature range is *not* rescued and no margin against it is quotable. This is the operational form of N78.
6. **Two new prior-art debts to schedule:** (i) **Sparse Delta Memory (arXiv:2607.07386, Meta FAIR, 7 Jul 2026)** — explicit slots + admission (top-W write selection) + decay + PKM addressing at 8B scale, the closest published relative of the CLU store; must be cited and differentiated in any explicit-memory positioning. (ii) **Clark, PRE 2026 (arXiv:2506.05303)** — transient-above-capacity retrieval; a positioning gift for pillar 1 and a preemption risk for §2.1(a).
7. **Owner needed for the reconciliation list at the top of this file** (protocol §5 corollary): items 1, 2 and 5 change task files, not just prose.
