# track2-admissibility — web-scout report

> ## ⚠⚠ ERRATUM BANNER — appended 2026-07-31 (C2W4) by `doc-curator-c2w3-sync`. **The body below is UNEDITED; read this first.**
> **Two corrections, both found in C2W3, both against statements in this file. Neither retracts this report's verdicts.**
>
> **E1 — the byte law (§2, and the CLU row of the pillar table).** *"`ratio = 1.4·atoms_per_item + 0.8`, 1e−9"* holds in **24 of 28** cells, not 28. The four `manifold` (`n_spectator = 1`) cells measure **52.00×** vs a published **43.33×** (**+20 %**); the floor at `n_spec = 1` **RISES** to **2.40×** (printed 2.00×). ⭐ Corrected law, exact in all 28 in rational arithmetic (0 ulp): **`ratio = [A(D+2) + d]/(d+m)`**. ⭐ **The error is CONSERVATIVE** (the store costs *more* relative to the table than published) ⇒ **the ≥2.20× caveat at `n_spec = 0` is unchanged, the theorem STANDS, and `PREREG-Bprime.md` §7's reuse licence STANDS — `bprime-rivals` does NOT re-measure.**
>
> ⛔ **E2 — MUNKEY's venue (§1 item 2 · §3.3 · never-quote item 13 · the pillar table · §7 · §9 item 3).** **MUNKEY (arXiv:2603.15033) is an ICLR-2026 *workshop* paper (ORAL), NOT ICML 2026.** arXiv v3 (2 Apr 2026) carries an **empty comments field**; the authors' own group page says *"Oral at ICLR Workshop TTU, 2026"* and an independent secondary says *"ICLR 2026 Workshop RSI"* ⇒ ⛔ **the workshop's identity is QUARANTINED — cite as "an ICLR-2026 workshop paper (oral)" and name no workshop.** Its **v3 self-describes as "a memory-augmented transformer"**, not "a ViT classifier" (that description cited v2). ⭐ **The narrowing itself STANDS** — unlearning-by-design at **MIA-AUROC → 0.5 by design**, **not exact** (gap to retraining **0.56 ± 0.21**) — and it is now **weaker as a threat to us**, not stronger.
>
> **Full errata, replacement sentences and the corrected BibTeX: `.claude/outputs/track2-admissibility/ERRATA-Bprime.md`.** ⛔ **`PREREG-Bprime.md` §7/§8 are deliberately NOT edited** (Hub ruling — a revised pre-registration stops being one). Sources: `bprime-theory` R-BYTE; `bprime-fb1-recon` reconciliations 1–2. Registry: **N167 / N168**, and **N134 / N163**'s dated correction blocks.

**Task + acceptance criterion:** the Track-2 candidate list surviving criterion 4 **and** the substitute audit *ex ante*; the **B′ pre-registration filed before the wave review**; the Sparse Delta Memory positioning brief + the exact-deletion prior-art sweep. Every convention traced to a primary source or marked **UNPINNED**.
**Status: done.** (Rider 4 — seed conventions via OpenReview — **partial/blocked**, see §R4.)
**Companion artifact:** `.claude/outputs/track2-admissibility/PREREG-Bprime.md` (filed as a standalone, timestamped pre-registration).

## ⛔ RECONCILIATION LIST (owner needed — protocol §5 corollary, first-10-lines rule)
1. **MAD `compression` is INADMISSIBLE.** It fails the substitute audit *ex ante* by arithmetic: MAD's own iso-state normalisation (4096 dims) exceeds the task's maximum payload (224 B) by **73× at fp32 / 36× at bf16**. This **overturns `rival-recon`'s single admissible synthetic**. ⇒ **Zero synthetic candidates survive. Track 2 has exactly one admissible primary and it is real-data LM.** Owner: Hub (changes C2W5 scope).
2. ⚠ **Pillar-4 near-miss (not the §6 emergency, but adjacent).** **MUNKEY (Laguna et al., arXiv:2603.15033, ICML 2026)** publishes *unlearning-by-design*: memory-key deletion `M_u = M \ {(k_i,v_i)}`, evaluated with **MIA AUROC → 0.5** — our own instrument. It is **not exact** (their avg gap to retraining **0.56 ± 0.21 ≠ 0**) and it is a **ViT classifier, not a sequence memory** ⇒ our claim survives, **narrowed**. Phrasing must change. Owner: Hub + curator.
3. **Titans is NeurIPS 2025 (peer-reviewed), not a preprint.** Every internal citation saying "preprint" is wrong. Owner: curator.
4. **SDM has official code** — `github.com/facebookresearch/sparse-delta-memory` (CC-BY-NC 4.0, Triton+CUDA, **SM 80+ GPU required**) ⇒ **it cannot run on this machine**; and its "no independent replication" caveat is now the wrong caveat. Owner: Hub (C2W5 sizing).
5. ⚠ **C2W3's factored/shared store structurally endangers byte-exact deletion** — our last uncontested pillar-4 differentiator is a *consequence* of per-item atom groups (the same property §A2.3 blames for excluding compression). Needs a design ruling **before** C2W3. Owner: Head/Hub.
6. **SDM Table 1 state/param ratios are CONFLICTING** across two independent extractions (156 % vs 168 %, 111 % vs 98 %) — **do not quote any SDM ratio** until a human reads the PDF. Owner: whoever drafts related work.
7. **Never-quote list re-stated in full at §8** (the `rival-recon` additions were never filed; this is currently its only home). Owner: curator.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed)
- **Dial:** none — recon. No performance claim measured.
- **Laundering control:** n/a for me; **the ex-ante substitute audit is the instrument** and it is applied to every candidate below *before* any harness exists.
- **Falsifies the deliverable:** a convention or metric-native verdict not traceable to a primary source and not marked UNPINNED.
- **Does NOT falsify:** few/no candidates surviving (decision-grade, feeds B′); our positioning found further occupied; contradicting a prior recommendation with evidence.

## What I did
Fanned across arXiv abs/HTML/ar5iv, NeurIPS/PMLR/ACL proceedings, authors' reference repos (`facebookresearch/sparse-delta-memory`, `fla-org/flash-linear-attention`, `athms/mad-lab`, `test-time-training/ttt-lm-pytorch`), and the Large Text Compression Benchmark. Pulled equations, config literals and table numbers, not prose. Two load-bearing items were **adversarially re-extracted** and one **failed** (SDM Table 1 — reported as conflicting, item 6 above).

---

# ⭐ DELIVERABLE 1 — the candidate list, screened against criterion 4 **and** the substitute audit, ex ante

**The two screens, as applied.** (1) *Criterion 4*: is the query in the same metric space as the stored keys, so an exact-match/kNN scan is a provable ceiling? (2) ⭐ *Substitute audit, ex ante*: name the strongest **+0 B** classical read-out over what the store already holds (order · echo · aggregate · count · recency · nearest-key). If a plausible +0 B substitute sits at or near ceiling, the task is inadmissible **before a harness is written**.

## 1.1 The ranked table

| rank | candidate | crit. 4 (not metric-native) | ⭐ strongest **+0 B** substitute | substitute at ceiling? | crit. 1 / 2 / 3 / 5 | verdict |
|---|---|---|---|---|---|---|
| **1** | **enwik8 / WikiText-103 LM, 26–47 M params** | ✅ target is a *next* token, never a stored key | (a) pure kNN over the store's own (ctx-rep, next-token) pairs; (b) count-based n-gram/PPM; (c) recency cache / dynamic evaluation | ⛔ **NO** — pure kNN is *degenerate*: **λ=1 gives ∞ perplexity** (Xu et al. ICML 2023); count-based PPM on enwik8 is **1.52–1.71 bpc** vs **1.232 bpc** for a 47 M AWD-LSTM | 1 ✅ · 2 ✅ · **3 ⚠ WEAK** · 5 ✅ | ✅ **ADMISSIBLE — the only primary** |
| **2 (fallback)** | non-stationary stream, regime revisit (charter fallback) | ✅ *iff* the query is regime-ID/aggregate, ⛔ if nearest-key | recency cache / **dynamic evaluation** (gradient adaptation to recent history) | ⚠ **HAZARDOUS** — Krause et al. ICML 2018 get **1.08 bpc (Hutter Prize)** / **1.19 (text8)** purely by adapting to recent history; kNN-LM + continuous cache **15.79** beats kNN-LM **16.12** | 1 ✅ · 2 ✅ · 3 ✅ · 5 ✅ | ⚠ **FALLBACK ONLY**, cache substitute mandatory as a column |
| ⛔ | **MAD `compression`** | ✅ (no key to match) | a shift-register/verbatim buffer + the task's own MLP decoder | ⛔ **YES, by arithmetic** — see §1.2 | 2 ⛔ · 3 ⛔ | ⛔ **REJECT** (reverses `rival-recon`) |
| ⛔ | RULER — NIAH (S/MK/MQ/MV) | ⛔ **fails** — literal needle retrieval | exact-match scan | YES | — | ⛔ **REJECT** |
| ⛔ | RULER — variable tracking (VT) | ✅ multi-hop, not a single lookup | union-find / transitive closure over the stored assignment pairs | **YES** (closed-form classical algorithm) | 2 ⛔ | ⛔ **REJECT** |
| ⛔ | RULER — aggregation (CWE / FWE) | ✅ aggregate, not lookup | **a counter** over stored tokens | **YES** (the task *is* counting) | 2 ⛔ | ⛔ **REJECT** |
| ⛔ | BABILong | ✅ multi-fact reasoning under distractors | sentence-level retrieval + a bAbI symbolic solver | likely YES; and RAG-S already beats RAG-C in the literature | scale ⛔ (10 k–1 M tokens) | ⛔ **REJECT** (scale + substitutability) |
| ⛔ | MQAR / zoology · MAD in-context/fuzzy/noisy recall · selective copying · memorization | ⛔ | exact-match scan / filtered FIFO | YES | — | ⛔ **already ruled inadmissible, §A3 — restated, not re-litigated** |
| ⛔ | POPGym / Memory Gym (memory-RL) | ✅ for several families | FIFO buffer (Repeat-Previous), counter (Count-Recall) | YES on the diagnostic families | 1 ✅ (**GRU is the best memory model**) · compute ⛔ | ⛔ **REJECT as primary**; distant fallback only |

## 1.2 ⭐ Why MAD `compression` dies (the arithmetic, ex ante)
- Task config, pinned from the authors' repo (`configs/tasks/compression.yml`, read verbatim): `vocab_size: 16, seq_len: 32, num_train_examples: 12800`; variations `vocab ∈ {32,64,128}`, `seq_len ∈ {64,128,256}`.
- **Maximum payload** (hardest cell): `256 × log₂128 = 1792 bits = 224 B`. Baseline cell: `32 × 4 = 128 bits = 16 B`.
- **MAD's own normalisation** (App. B.3, verbatim): *"All evaluated architectures that do not include attention layers are normalized to a total state dimension of 4,096."* ⇒ **16 384 B at fp32 / 8 192 B at bf16.**
- ⇒ **state / payload ≥ 73× (fp32), ≥ 36× (bf16); ≥ 1024× at the baseline cell.** Even the single-token bottleneck (width **128**, App. B.3) is `4096 bits > 1792 bits`.
- ⇒ **Compression in MAD is never beyond-capacity.** The difficulty is *routing all positions into one position*, i.e. an architectural/optimisation difficulty — **not** capacity pressure (crit. 3 fails) — and a hand-constructed shift-register + the task's own MLP decoder is at ceiling in principle (crit. 2 fails).
- ⇒ **Charter §2.1(a) "beyond-capacity compression" has no home in the community synthetic suite at its own iso-state convention.** Combined with §A2.3 (per-item atom groups structurally exclude compression), **pillar (a) currently has no admissible public venue at this weight class.** That is a direct, unsoftened input to the B′ decision.

*Evidence grade:* the config literals and the iso-state sentence are **pinned**; the "a shift-register is at ceiling" step is **INFERRED (arithmetic)** — MAD's per-architecture compression accuracies could not be extracted numerically (best = multi-head Mamba, worst = attention; **numeric values UNPINNED**), so I do *not* claim a published architecture is already at ceiling.

## 1.3 ⭐ Why enwik8/WikiText survives — and where it is weak
**Survives the substitute audit, two independent ways:**
- **The +0 B kNN read is degenerate.** Xu, Alon, Neubig (ICML 2023): with the kNN component alone (λ=1) *"the probability of the target word will be zero"* whenever the search misses the true target ⇒ **infinite perplexity**; the tuned interpolation is **λ ≈ 0.25–0.27**, i.e. the retrieval component carries a *minority* of the mass. Khandelwal et al. (ICLR 2020): WT-103 base LM **17.96 / 18.65** → kNN-LM **16.06 / 16.12**, λ = 0.25, 103 M datastore entries, keys quantised to **64 bytes**.
- **The count-based classical model is not at ceiling.** Large Text Compression Benchmark, enwik8: `ppmd J1` **21 388 296 B ≈ 1.711 bpc**, `ppmonstr J` **19 055 092 B ≈ 1.524 bpc**, `xz 5.2.1` **24 703 772 B ≈ 1.978 bpc** — versus **1.232 bpc** (3-layer AWD-LSTM, 47 M) and **1.336 bpc** (4-layer AWD-QRNN, 26 M) from Merity et al. 2018. ⚠ **Caveat, stated:** LTCB numbers are single-pass adaptive compression of the *whole* 100 MB including the decompressor, not held-out bpc on the 90/5/5 LM split — the comparison is indicative, not like-for-like. Best context-mixing entries (`cmix v21` **≈1.170**, `nncp v3.2` **≈1.193**) themselves contain LSTMs/transformers and are not "classical".

**⚠ Criterion 3 is the weak leg, and it is weak for a *published* reason:**
- Khandelwal et al. (ACL 2018): an LSTM LM uses *"about 200 tokens of context on average"* but *"sharply distinguishes nearby context"* — word **order is ignored beyond ~50 tokens**, the distant past acting as *"a rough semantic field or topic."*
- Krause et al. (ICML 2018): **gradient adaptation to recent history alone** gives **1.08 bpc (Hutter Prize) / 1.19 (text8)** — i.e. a large share of the "manages memory over time" gain on real text is obtainable by a cheap recency mechanism.
⇒ **Binding design instruction for the Track-2 harness:** aggregate bpc/ppl is **not** where the dividend can be claimed. The dividend must be reported on a **long-range slice** (recall-vs-distance curve; context-ablation à la Khandelwal 2018), with **three mandatory substitute columns**: (i) smoothed pure-kNN over the store, (ii) count-based n-gram/PPM, (iii) recency cache / dynamic evaluation. If the dividend vanishes once (iii) is in the table, the primary is dead and B′ is the paper.

## 1.4 ⭐ The headline finding
**Zero synthetic candidates survive both screens. Exactly one primary survives, and it survives on a criterion-3 leg the literature has already shown to be weak.** The pattern is structural and worth stating in the paper: *any synthetic memory task with a closed-form classical algorithm has a +0 B substitute at ceiling by construction* — which is the entire synthetic suite. Real-data LM is admissible **precisely because no classical method is at ceiling on it**. This is not bad luck; it is the same theorem as criterion 4, one level up.

---

# ⭐ DELIVERABLE 2 — the B′ PRE-REGISTRATION
*(filed 2026-07-30, before the C2W2 gate is adjudicated; mirrored at `.claude/outputs/track2-admissibility/PREREG-Bprime.md`)*

## 2.1 The claim, one sentence
> **When does test-time dynamics buy anything over a table at matched bytes?** — we apply one protocol (matched-byte launder + two-sided byte ledger + a **+0 B** substitute audit + same-keys null) uniformly to the modern neural-memory family (Mamba-2, DeltaNet, Gated DeltaNet, TTT, Titans, Sparse Delta Memory) **and to CLU**, and report for each family the dividend of its learned dynamics over a byte-matched non-parametric store.

## 2.2 The table the paper is (rows = families, cols = instruments; every cell **have / need**)

| family | matched-byte table launder | **+0 B** substitute | two-sided byte ledger | same-keys null | metric-native verdict | deletion / lifetime probe | anytime shape curve |
|---|---|---|---|---|---|---|---|
| **CLU** | **have** (28 cells, ratio ≥ 2.20×, `matched=False`) | **have** (**0-for-4**) | **have** (`ratio = 1.4·atoms_per_item + 0.8`, 1e−9) | **have** | **have** | **have** (AUC **0.5000 ± 0.0000**, byte-equal **3072/3072**) | **need** (shape only, §A3) |
| GRU | need | need | **have** (`d_hidden`) | need | **have** (weak) | n/a (no explicit item) | need |
| sliding-window attn | need | need | **have** (`2·n_kv·d_head·w`) | need | **have** | n/a | need |
| Mamba-2 | need | need | **have** (`264·d_model + 1024`, from `mamba2.py`) | need | **have** (SSD duality) | need | need |
| DeltaNet / GDN | need | need | **have** (`n_head·d_k·d_v`) | need | **have** (Eq. 8) | need | need |
| TTT-Linear / MLP | need | need | **have** (`d_head²` / `8·d_head²` + `b=16` buffer; **W₀ = parameters**) | need | **have** (Lin: yes; MLP: weakly) | need | need |
| Titans (MAC) | need | need | ⚠ **UNPINNED** (`2·|M_θ|` is **our reconstruction; the paper states no convention**) | need | **have** (`L_M=1` yes, `≥2` weakly) | need | **have** (Titans-Revisited chunk sweep) |
| **Sparse Delta Memory** | need | need | **have** (Eq. 6) + ⚠ **learned `M₀` = parameters, deviation = state** | need | **have** (PKM top-k *is* kNN) | need | need |

## 2.3 Sizing — what is reimplementable at this weight class (all pinned 2026-07-30)

| rival | official code | runnable here? | caption required |
|---|---|---|---|
| Mamba-1/2 | `state-spaces/mamba`; also in **FLA** | ✅ (arch is portable) | none |
| DeltaNet / GDN / GLA | **FLA** (`fla-org/flash-linear-attention`, 41 models, GDN & Mamba2 & DeltaNet present) | ✅ | none |
| TTT | **official** `test-time-training/ttt-lm-pytorch` (+ JAX training repo, + kernels repo); authors: the PyTorch repo is *"pure PyTorch without systems optimization … we do not recommend training with this codebase"* | ✅ (use JAX repo for training) | none |
| **SDM** | **official** `facebookresearch/sparse-delta-memory` — model def + **Triton and CUDA kernels**, CC-BY-NC 4.0, configs `debug_sdm.yaml` (small, single-GPU) / `sdm_flagship.yaml` (1.4 B) / `sdm_7B.yaml`; **requires Torch ≥ 2.8, Triton ≥ 3.4, SM 80+ (Ampere/Hopper)** | ⚠ **NOT on this machine** (macOS/JAX, no CUDA) | *"our reimplementation of Eqs. 3–5; the authors' kernels were not used"* |
| **Titans** | ⛔ **none, still.** Paper (NeurIPS 2025): *"Titans are implemented in Pytorch and JAX and we intend to make the code we used to train and evaluate our models available soon."* **Not in FLA.** Chunk size `b` **never given a numeric value** | ⚠ reimplementation-from-description only | *"reimplemented from the paper's description; the chunk size `b` is not stated in the paper"* + cite Titans-Revisited on reproducibility |

## 2.4 Byte conventions B′ needs (extends `rival-recon` §F2 — additions only)
- ⭐ **New general rule — learned-initial-state memories.** SDM's abstract: *"by learning the initial state of the SDM memory and therefore using it as a parametric memory."* Same structure as TTT's shared learnable `W₀`. ⇒ **For any memory with a learned init: the initialisation is PARAMETERS (F1), only the per-sequence deviation is STATE (F2). Both must be declared in the ledger; a paper that counts the init as state is inflating, one that counts the deviation as parameters is laundering.**
- **SDM:** `M_size = (d_qk^tot)² · d_v^tot / (4H²)` (Eq. 6); slots `M[i]` are explicit; decay `α_t` is **per-head, not per-slot**; the top-W/top-R index sets are transients (F4, not F2). ⚠ **SDM's Table 1 state/param ratio column is CONFLICTING between two extractions — unusable until human-verified.** The only pinned qualitative statement is the limitation: *"SDM memory requirements are not negligible, as the memory footprint may be as large as the model parameters."*
- **Titans:** the `2·|M_θ|` momentum accounting **remains our reconstruction and must be captioned as such, every time.** Re-scouted; still UNPINNED.
- **CLU:** unchanged; every dividend/byte claim inherits the **≥ 2.20× ratio caveat** until the shared substrate lands (§A3).

## 2.5 ⭐ Pre-registered predictions (commit now, measure later)
- **P1.** On real-data LM, the **+0 B pure-kNN** substitute over the store's own (key, value) pairs is **degenerate** (unbounded loss). With add-λ/backoff smoothing it loses to the tuned neural arm by **≥ 0.3 bpc** (equivalently ≥ 20 % relative ppl). *Derivation:* Xu et al. 2023 λ=1 → ∞ ppl; kNN-LM's tuned λ = 0.25 ⇒ retrieval carries a minority of the probability mass.
- **P2.** Of the four families with an explicit (k,v)-shaped state (Mamba-2, DeltaNet, GDN, SDM), **≥ 3 lose to their own byte-matched table on a metric-native probe**, and **0 of 4 lose to it on real-data LM bpc.**
- **P3.** The two **function-valued** memories (TTT-MLP, Titans `L_M ≥ 2`) show the **largest positive dividend** over their matched-byte tables on real-data LM — because a nonlinear readout is the field's only current escape from literal metric-nativeness (`rival-recon` D3).
- **P4.** **No published rival paper runs a non-parametric matched-byte control.** *Current status:* none found (Zoology/Based App. E.2 report state bytes but vary state by hyperparameters, not against a table; MAD normalises iso-state across **neural** architectures only; SDM reports isoFLOP+isoParam and a state/param ratio, no table baseline). **Absence of evidence, medium confidence** — this is the single claim most worth a second pass before submission.
- **P5.** The launder transfers to **all five** rival state types (explicit (k,v) → table directly; weight-valued memories → a table of the `(θ_K x, θ_V x)` pairs at equal bytes). Predicted failures: **0 of 5**.

## 2.6 ⭐ The falsifier of B′ ITSELF (registered before the gate)
- **FB1 — "not news."** If ≥ 1 established paper in the family already runs a matched-byte non-parametric control and reports the same verdict, the audit is a replication, not a contribution. *(Kills P4; kills B′'s novelty. Cheap to test — one targeted pass.)*
- **FB2 — "not apples-to-apples."** If for **≥ 2 of 5** rival families no byte-matched table can be defined without an arbitrary modelling choice, the cross-family comparison is invalid and B′ collapses to a CLU-only negative result.
- **FB3 — "the finding inverts."** If every rival shows a large positive dividend and only CLU does not, B′ is no longer "test-time dynamics rarely pays"; it is "test-time dynamics pays, except for ours." That is still publishable but it is a **different paper**, and we pre-commit to saying so rather than re-framing.
- **FB4 — "the instrument is invalid."** If the +0 B substitute is at ceiling for **every** family *including full attention*, the protocol is measuring the task and not the memory. Testable cheaply on one family; **this is the first thing B′ should run.**
- **FB5 — the theoretical neighbour.** **Wang, Shi & Fox, "Test-time regression: a unifying framework…", arXiv:2501.12352** already unifies linear attention, SSMs, fast-weight programmers, online learners and softmax attention as *"test-time regression"* special cases. B′ must differentiate on the **empirical byte-matched audit** (they unify mechanisms; we price them), and if a later version of that line adds a non-parametric matched-byte baseline, FB1 fires.

## 2.7 What B′ reuses rather than re-derives (banked; do not re-measure)
matched-bytes launder record (28 cells, `matched=False` architectural) · byte-floor theorem **`ratio = 1.4·atoms_per_item + 0.8`**, verified to **1e−9** in all 28 cells, floor **2.20×**, measured min **2.28×** · substitute audit **0-for-4** (insertion order **0.776** vs 0.302; echo **1.0000** vs −0.180) · **Prop D2a**, three independent confirmations · byte-exact deletion **AUC 0.5000 ± 0.0000**, byte-equal **3072/3072** · `D` is the dividend's variance, not its magnitude · the accuracy-vs-bytes curve (`decode 0.972 → 0.097` as ratio falls `478× → 2.28×`).

---

# ⭐ DELIVERABLE 3 — Sparse Delta Memory positioning brief

**Cabannes, Mazaré, Szilvasy, Douze, Lomeli, Auzina, Carpentier, Synnaeve, Jégou (2026), "Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity", arXiv:2607.07386, Meta FAIR, submitted 8 Jul 2026.** Preprint, 3 weeks old, **official code released**. Mandatory in C2W5's baseline set.

## 3.1 The mechanism map, at equation level

| SDM | CLU | same thing? |
|---|---|---|
| explicit slots `M[i]`, `i ∈ 1..N` | atom groups (one group per item, masked write) | **Structurally yes.** Both are an explicit, addressable, test-time-written store. SDM's slots are shared across the sequence; ours are per-item — **which is why we get exact deletion and cannot compress (§A2.3), and why they can compress and cannot delete exactly.** This is the cleanest statement of the trade and it should be the paper's sentence. |
| **PKM top-k addressing** (product-key; `O(√N·d + W² + R²)`) | derived addresses + admission gate | **Not the same.** Theirs is an inner-product ANN index over learned keys; ours is a geometric address in the potential landscape. **But the referee's "yours is a kNN with extra steps" applies to theirs first** (`rival-recon` F9). |
| **Eq. 3 decay:** `M̃_t[i] ← α_t · M_{t−1}[i]`, `α_t = exp(−A·softplus(W_a x_t + b_dt))`, **per-head, not per-slot** | our decay law (exact, `6.8e-8`), **per-item settable** | **Same operation, different granularity and provenance.** Theirs is input-conditioned and head-global; ours is per-item, exogenously settable, and **predicted by theory rather than fit.** The *phrase* "principled forgetting" is theirs (⛔ never-quote); the *granularity* and the *prediction* are ours. |
| **Eq. 4 write:** `M_t[i] ← M̃_t[i] + β_t k_t^{(i)}(v_t − M̃_t[i])`, applied only to the **top-W** selected slots `I_t^w` | admission gate | **Yes — this is an admission policy.** Top-W write selection = "only these slots are written this step". Our admission gate is a *decision to admit an item*; theirs is a *decision which slots receive it*. Adjacent, not identical; **do not claim admission as novel.** |
| **Eq. 5 read:** `y_t = M_tᵀ q_t = Σ_{i∈I_t^r} q_t^{(i)} · M_t[i]` over **top-R** slots | our read (settle / trajectory-ψ) | **Not the same, and this is our only structural escape.** Eq. 5 is a sparse *linear* combination ⇒ **provably metric-native**. A trajectory read is a path functional, not a point evaluation (`rival-recon` D3.2). |
| **Eq. 6 state:** `M_size = (d_qk^tot)²·d_v^tot/(4H²)` | `n_atoms·d·sizeof(dtype) + codebook + controller` | different formulas, same ledger obligation |
| **learned initial state `M₀`** ("parametric memory") | frozen `V_θ` init | **Same laundering hazard**: init = parameters, deviation = state. Their own framing makes this explicit and we should adopt their vocabulary. |

## 3.2 What survives as ours after the collision
1. **Byte-exact deletion** — AUC **0.5000 ± 0.0000**, byte-equal **3072/3072**. SDM has **no deletion mechanism at all** (no mention of deletion/unlearning/removal anywhere in the paper). ⚠ **Narrowed by MUNKEY (§3.3).**
2. **Settable per-item lifetimes** — SDM's `α_t` is **per-head**; ⚠ **partially preempted** by `Adaptive Memory Decay for Log-Linear Attention` (arXiv:2605.06946): decay learned from input by a 2-layer MLP, **per-token and per-level**. What survives is *settable* (exogenous, honored to a measured tolerance) + the *predicted* decay law — not "per-item decay" as a mechanism.
3. **The physics-predicts-the-knobs spine** — **still no preemption found** across this scan (SDM fits `A`, `W_a`, `b_dt`; nobody predicts a hyperparameter from theory). Strongest pillar, unchanged.
4. ⛔ *"Principled forgetting"* as a novelty phrase: **dead** (Titans Eq. 13, GDN Eq. 8, SDM Eq. 3).

## 3.3 ⭐ The exact-deletion prior-art sweep (machine-unlearning ∩ sequence-memory)
**Result: our claim survives, materially narrowed. This is the §6 falsifier's near-miss and the Hub should treat it as such.**
- ⚠ **MUNKEY — Laguna, da Silva Gonçalves, Vandenhirtz, Ryser, Cannistraci, Vogt (ETH Zurich), "Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion", arXiv:2603.15033v2 (16→24 Mar 2026; the paper's own header states ICML acceptance — *single-sourced, arXiv abs shows no comments field*).** Abstract, verbatim: *"We propose **unlearning by design**, a novel paradigm in which models are directly trained to support forgetting as an inherent capability … unlearning corresponds to removing the instance-identifying key, enabling direct **zero-shot forgetting without weight updates** or access to the original samples or labels."* External exemplar memory `ℳ = {(k_i, v_i)}`; deletion is `ℳ_u = ℳ \ {(k_i,v_i) | i ∈ D_f}`; evaluated with **MIA AUROC near 0.5** on the forget set — **our exact instrument**.
  - **Why our claim survives:** (i) it is a **memory-augmented ViT image classifier**, not a sequence/streaming memory; (ii) it is **not exact** — their own headline metric is the average gap to gold-standard retraining, **0.56 ± 0.21** (CIFAR-10, 10 % forget), i.e. **non-zero residual**; (iii) no theorem or certificate is stated.
  - **Why it narrows us hard:** the *mechanism* ("delete the memory key ⇒ forget the item, no retraining") and the *framing* ("unlearning by design") are now published in a strong venue, with the same MIA-AUROC-0.5 evidence. **We can no longer present key-deletion-as-forgetting as our idea.** The survivable claim is exactness and its verification: *byte-equality of the store to the never-written counterfactual, with MIA AUC exactly 0.5000 ± 0.0000* — versus a published architectural analogue whose own gap to retraining is non-zero.
- **No sequence-memory rival does exact deletion.** SDM, Titans, GDN, DeltaNet, TTT, Mamba: **soft decay only**, no removal primitive (searched; absence of evidence, medium-high confidence).
- ⚠ **The honest asymmetry we must state ourselves before a referee does:** a **table deletes exactly by construction** — kNN-LM's datastore (103 M entries, keys quantised to 64 B) supports entry removal trivially; the paper simply never discusses it because it is uninteresting for a table. **Exact deletion is only a result for a *learned/superposed* store.** Which leads to:
- ⛔ **CROSS-WAVE RISK (reconciliation item 5).** Our byte-exact deletion is a *consequence of per-item atom groups* — the same property §A2.3 identifies as structurally excluding compression. **C2W3's factored/shared store (wells shared across items) is very likely to break byte-exactness.** Either the C2W3 spec preserves a deletion-exact regime as a declared mode, or pillar 4's last differentiator is spent in the same wave that fixes pillar (a). **This needs a Head/Hub ruling before C2W3 is scoped.**

## 3.4 Their published limitations = our fair-fight footing (pinned)
- *"The main limitation of SDM is also its strength: SDM memory requirements are not negligible, as the memory footprint may be as large as the model parameters."*
- MFU **"around an order of magnitude lower"** than the optimised GDN kernel; **1.49× slower training** than GDN at 8 B at matched FLOPs.
- Precedent we inherit: **a frontier lab publishes its compute deficit.** Our 150–1200 Verlet steps/read and ψ's **17.1×** belong in the main table, not the appendix (F4).
- Their reported RULER averages (repo README, 1.4 B): FullAttn **32.5** · Mamba2 **17.9** · GDN **20.0** · **SDM 31.2**; (8 B): FullAttn **61.2** · GDN **34.2** · **SDM 50.2**. ⚠ Read from the repo README, **not** the paper — treat as reference-implementation-grade, and re-pin from the PDF before citing.

---

# RIDERS

**R1 — RULER.** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg, *"RULER: What's the Real Context Size of Your Long-Context Language Models?"*, **arXiv:2404.06654, COLM 2024**. **13 tasks, four categories:** retrieval (S-NIAH, MK-NIAH, MQ-NIAH, MV-NIAH), **multi-hop tracing** (variable tracking), **aggregation** (common-word extraction CWE, frequent-word extraction FWE), QA. Configurable sequence length and task complexity; synthetic. **Verdict:** ⛔ **inadmissible as a Track-2 primary** — retrieval fails criterion 4 outright; VT and CWE/FWE pass criterion 4 but **fail the substitute audit** (union-find; a counter). **Keep it as the field's calibration currency only** — SDM and GDN both report it, and SDM's repo publishes per-task numbers, so it is the cheapest way to check our baselines are not broken. It is *not* where a dividend can be claimed.

**R2 — Titans chunk size + code.** Re-scouted 2026-07-30. **(a) Venue: NeurIPS 2025** (poster 119639; proceedings paper `a4ca07aa…`), OpenReview `8GjSf9Rh7Z` — **not a preprint**. **(b) Code: still none.** Paper, verbatim: *"Titans are implemented in Pytorch and JAX and we intend to make the code we used to train and evaluate our models available soon."* Not present in FLA's 41-model table. Only third-party reimplementations exist. **(c) Chunk size `b`: still UNPINNED** — the paper defines *"chunks of size b ≥ 1"* and never assigns a number. **(d) Seeds: not reported.** **(e) Pinned hyperparameters:** *"LLama 2 tokenizer with a vocabulary size of 32K … training length of 4K tokens. We employ AdamW optimizer with learning rate of 4e-4 with cosine annealing schedule with batch size of 0.5M tokens, and weight decay of 0.1."* ⚠ Extracted from arXiv **v1 HTML**; the NeurIPS camera-ready PDF was **not machine-readable** in this session, so I cannot exclude that `b` was pinned in camera-ready. **Any Titans arm ships with the caption: "reimplemented from the paper's description; the chunk size is not stated."**

**R3 — Pillar-4 preemption check on the two unread 2026 preprints.**
- **arXiv:2604.07350 — Ma, Yu, Zhen, Yang, Chai, Gan, "Fast Spatial Memory with Elastic Test-Time Training" (8 Apr 2026).** LaCT fast weights + an **elastic (EWC-style) Fisher-weighted prior around an EMA anchor**. Domain is **4D/3D reconstruction**, not memory management. **NOT a pillar-4 preemption** — it is stability regularisation, no lifetimes, no deletion. *Adjacent relevance:* an EMA "anchor state" for fast weights is a cheap stability mechanism worth knowing about if our consolidation phase needs one.
- **arXiv:2605.06946 — Amin, Li, Zhang, Ayhan, "Adaptive Memory Decay for Log-Linear Attention" (7 May 2026, preprint, no venue).** Decay λ learned from input by a **two-layer MLP**, giving **per-token, per-level** decay over the Fenwick-tree hierarchy of Log-Linear Attention (**Guo, Yang, Goel, Xing, Dao, Kim, arXiv:2506.04761, ICLR 2026**). ⚠ **This IS a partial pillar-4 preemption** — it is the closest published neighbour to per-item lifetimes. **What survives:** *settable* (exogenous) lifetimes honored to a measured tolerance, and the *theory-predicted* decay law. **Add "per-item decay rate is unoccupied" to the never-quote list.**

**R4 — Seed conventions. ⚠ PARTIAL / BLOCKED, reported honestly.** OpenReview forum and PDF endpoints returned a **browser bot-check** to this agent on both attempts (`openreview.net/forum?id=8GjSf9Rh7Z`, `openreview.net/pdf?id=HklBjCEKvH`) — I could not read a single reviewer thread. What I *can* pin from primary papers: **MAD = 1 seed** (App. B.4, best-of-3×2 grid); **Zoology/Based = seeds not reported**; **Titans (NeurIPS 2025) = seeds not reported**; **SDM = seeds not reported in the abstract/limitations I could read.** ⇒ **The field-wide convention is "1 seed, best-of-grid," and no paper in this family reports variance.** Our **≥3-seed** rule stays strictly stricter and is stated in the paper as a methodological strength. **Next-step (needs a human or an authenticated fetch): one OpenReview reviewer pass on GDN (ICLR 2025) and Titans (NeurIPS 2025) rebuttals, where seeds are usually forced out.**

**R5 — "Real Mamba" rule, versioned, one line for the handover:**
> **"Real Mamba" = Mamba-2 minimum** (`mamba_ssm/modules/mamba2.py` defaults: `d_state=128, headdim=64, expand=2, d_conv=4, ngroups=1, chunk_size=256`), **with Mamba-3 (arXiv:2603.15569) named in limitations.** Mamba-1 alone no longer satisfies it.

---

# Confidence table

| claim | status |
|---|---|
| MAD `compression.yml` literals (`vocab 16, seq_len 32`, variations to 128/256) | **pinned** (authors' repo, read verbatim) |
| MAD iso-state 4096 / width 128 / 4 layers / embeddings excluded | **pinned** (App. B.3, quoted) |
| MAD compression state-vs-payload ratio ≥73× fp32 / ≥36× bf16 | **INFERRED (arithmetic)** from the two pinned lines — the arithmetic is checkable, the "shift-register at ceiling" step is a construction argument, not a measurement |
| MAD per-architecture compression accuracies | **UNPINNED** — only "best = multi-head Mamba, worst = attention" recovered |
| kNN-LM WT-103 17.96/18.65 → 16.06/16.12, λ=0.25, 103 M entries, 64-B keys | **pinned** (ar5iv, Khandelwal et al. ICLR 2020) |
| **kNN alone (λ=1) ⇒ ∞ perplexity** | **pinned** (Xu, Alon, Neubig, ICML 2023) — the load-bearing substitute-audit fact for the LM candidate |
| enwik8 LTCB: ppmd 1.711 / ppmonstr 1.524 / xz 1.978 / cmix 1.170 / nncp 1.193 bpc | **pinned to the benchmark of record**, but ⚠ **not the LM protocol** (whole-file adaptive pass, self-extracting size) — comparison is indicative |
| Merity 2018 enwik8 47 M → 1.232 / 26 M → 1.336 bpc | **pinned** (inherited, `rival-recon`) |
| Khandelwal 2018 "≈200 tokens of context; order ignored beyond ~50" | **pinned** (ACL 2018) |
| Krause 2018 dynamic evaluation: 1.08 bpc Hutter Prize, 1.19 text8, PTB 51.1, WT-2 44.3 | **pinned** (PMLR v80) ⚠ static baselines not recovered from the landing page |
| SDM Eqs. 3/4/5/6, PKM top-W/top-R, per-head α_t, limitations quote | **pinned** (arXiv HTML) |
| **SDM Table 1 state/param ratios** | ⛔ **CONFLICTING** across two extractions (156 %/168 %, 111 %/98 %) — **do not quote** |
| SDM official code, kernels, CC-BY-NC 4.0, SM 80+, `debug_sdm.yaml` | **pinned** (repo README) |
| SDM RULER averages (1.4 B / 8 B) | **pinned to the repo README**, not the paper — re-pin before citing |
| **Titans = NeurIPS 2025**; no code; `b` unpinned; no seeds; LR 4e-4 / 4K len / 0.5 M batch | **pinned** (proceedings listing + arXiv v1 HTML) ⚠ camera-ready PDF unreadable this session |
| TTT official code (pytorch + JAX + kernels) | **pinned** (repo) |
| FLA contains GDN/DeltaNet/Mamba2, **not** Titans, **not** SDM | **pinned** (repo README, 41-model table) |
| **MUNKEY** arXiv:2603.15033, key-deletion unlearning, MIA AUROC≈0.5, avg gap 0.56±0.21, ViT classifier | **pinned** (arXiv HTML v2 + abs) · **ICML acceptance is single-sourced** (paper header only) |
| "no sequence-memory rival does exact deletion" | **absence of evidence, medium-high** — targeted sweep, not exhaustive |
| "no rival runs a matched-byte non-parametric control" (B′ P4) | **absence of evidence, medium** — the single most load-bearing gap in B′ |
| RULER 13 tasks / 4 categories, COLM 2024, arXiv:2404.06654 | **pinned** |
| Log-Linear Attention arXiv:2506.04761, **ICLR 2026** | **pinned** (PDF header) |
| arXiv:2605.06946 per-token per-level MLP decay | **pinned but low-profile preprint**, no venue, no independent replication |
| POPGym: 15 envs, 13 baselines, **GRU best general-purpose memory model** | **pinned** (ICLR 2023) |
| Seed conventions across the family | **UNPINNED / blocked** — OpenReview bot-check defeated two fetch attempts |

---

# §8. ⛔ NEVER-QUOTE LIST — restated in full (inherited + new; this file is currently its only home)

**Inherited (task §8):**
1. "An anytime accuracy-vs-compute curve no baseline can draw" — **falsified** (DEQ; Neural DEQ Solvers; EBT Fig. 6a/12; Titans-Revisited chunk sweep). The anytime curve is a **shape** claim (§A3).
2. **"Principled forgetting"** as a novelty claim — Titans Eq. 13, Gated DeltaNet Eq. 8, SDM Eq. 3 own it.
3. **"Graceful degradation above capacity"** as our discovery — Clark, *Phys. Rev. E* 2026, arXiv:2506.05303.
4. Any **Titans state-byte number** without *"our reconstruction; the paper states no convention."*
5. **"Guo et al. Def. 1 / Def. 2" — it does not exist** (ε-certified removal is §3 Eq. (1), inline).
6. **"certified" / "unlearning" / "deletion-compliant" / unqualified "exact deletion."**
7. A margin against an **un-rescued baseline** (N78: a baseline below its published range is not rescued; no margin against it is quotable).

**⭐ New, from this scan (add to the registry):**
8. **"MAD `compression` is the one admissible synthetic"** — retracted by §1.2 of this report.
9. **"Titans is a preprint"** — it is **NeurIPS 2025**.
10. **"SDM has no code / no replication available"** — official code exists (`facebookresearch/sparse-delta-memory`, CC-BY-NC 4.0).
11. **Any SDM state/parameter ratio number** (156 %, 168 %, 111 %) until a human re-reads Table 1 — two independent extractions disagree.
12. **"Per-item decay rates / lifetimes are unoccupied"** — arXiv:2605.06946 does per-token, per-level input-dependent decay. Only *settable* + *theory-predicted* survives.
13. **"We are the only architecture that deletes an item by removing it from memory"** / "key deletion as forgetting is our idea" — **MUNKEY (arXiv:2603.15033, ICML 2026)** publishes exactly that mechanism, with MIA-AUROC≈0.5 as its metric. Permitted phrasing: *"byte-exact deletion, verified by byte-equality to the never-written counterfactual (3072/3072) at MIA AUC 0.5000 ± 0.0000 — where the published key-deletion architecture reports a non-zero gap to retraining (0.56 ± 0.21)."*
14. **"Exact deletion is hard"** without the asymmetry: **a table deletes exactly by construction.** Exact deletion is a result only for a *learned/superposed* store — and it is a property of per-item atom groups, which C2W3 is scheduled to remove.

---

# Gaps / what to search next (ranked by value-per-hour)
1. ⭐ **B′-P4 second pass** — "does any rival run a matched-byte non-parametric control?" is B′'s novelty load-bearer and is currently *absence of evidence*. Target: `Test-time regression` (arXiv:2501.12352) full text, Zoology App. E, SDM appendix, Titans-Revisited.
2. ⭐ **SDM Table 1 re-read by a human** (state/param ratios conflict) and **SDM's RULER numbers re-pinned from the PDF** rather than the README.
3. **OpenReview reviewer threads** (GDN ICLR 2025, Titans NeurIPS 2025) for seed conventions — needs an authenticated/browser fetch; blocked for me.
4. **MUNKEY's actual venue** (ICML year) — single-sourced from the paper's own header.
5. **MAD per-architecture compression accuracies** — would convert §1.2's rejection from arithmetic to arithmetic + measurement.
6. **Titans camera-ready PDF** — machine-readable copy, to confirm `b` is still unpinned post-review.
7. **enwik8 long-range slice protocol** — is there a published context-ablation protocol for character-level enwik8 (Khandelwal 2018 is word-level PTB/WT-2)? Needed before the Track-2 harness defines the long-range slice.
8. **H3 / Hyena / RWKV state-byte formulas** (Based App. E.2) — only if those arms enter the table. *(carried from `rival-recon`)*

---

# BibTeX-ready refs (new in this report; `rival-recon`'s list still stands)
```bibtex
@article{laguna2026munkey, title={Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion},
  author={Laguna, Sonia and da Silva Gon{\c c}alves, Jorge and Vandenhirtz, Moritz and Ryser, Alain and Cannistraci, Irene and Vogt, Julia E.},
  journal={arXiv preprint arXiv:2603.15033}, year={2026},
  note={v2, 24 Mar 2026; paper header states ICML acceptance (single-sourced)}}

@inproceedings{xu2023whyknnlm, title={Why do Nearest Neighbor Language Models Work?},
  author={Xu, Frank F. and Alon, Uri and Neubig, Graham}, booktitle={ICML}, year={2023}, note={arXiv:2301.02828; PMLR v202}}

@inproceedings{khandelwal2020knnlm, title={Generalization through Memorization: Nearest Neighbor Language Models},
  author={Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle={ICLR}, year={2020}, note={arXiv:1911.00172}}

@inproceedings{khandelwal2018sharp, title={Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context},
  author={Khandelwal, Urvashi and He, He and Qi, Peng and Jurafsky, Dan}, booktitle={ACL}, year={2018},
  note={arXiv:1805.04623; ACL Anthology P18-1027}}

@inproceedings{krause2018dynamiceval, title={Dynamic Evaluation of Neural Sequence Models},
  author={Krause, Ben and Kahembwe, Emmanuel and Murray, Iain and Renals, Steve},
  booktitle={ICML}, year={2018}, note={PMLR v80; arXiv:1709.07432}}

@inproceedings{hsieh2024ruler, title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Hsieh, Cheng-Ping and Sun, Simeng and Kriman, Samuel and Acharya, Shantanu and Rekesh, Dima and Jia, Fei and Zhang, Yang and Ginsburg, Boris},
  booktitle={COLM}, year={2024}, note={arXiv:2404.06654}}

@inproceedings{guo2026loglinear, title={Log-Linear Attention},
  author={Guo, Han and Yang, Songlin and Goel, Tarushii and Xing, Eric P. and Dao, Tri and Kim, Yoon},
  booktitle={ICLR}, year={2026}, note={arXiv:2506.04761}}

@article{amin2026adaptivedecay, title={Adaptive Memory Decay for Log-Linear Attention},
  author={Amin, Yaxita and Li, Helen Zichen and Zhang, Mengfan and Ayhan, Samet},
  journal={arXiv preprint arXiv:2605.06946}, year={2026}, note={preprint, no venue}}

@article{ma2026fsm, title={Fast Spatial Memory with Elastic Test-Time Training},
  author={Ma, Ziqiao and Yu, Xueyang and Zhen, Haoyu and Yang, Yuncong and Chai, Joyce and Gan, Chuang},
  journal={arXiv preprint arXiv:2604.07350}, year={2026}}

@article{wang2025testtimeregression, title={Test-time regression: a unifying framework for designing sequence models with associative memory},
  author={Wang, Ke Alexander and Shi, Jiaxin and Fox, Emily B.},
  journal={arXiv preprint arXiv:2501.12352}, year={2025}}

@inproceedings{behrouz2025titans, title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={NeurIPS}, year={2025}, note={arXiv:2501.00663; no official code as of 2026-07-30}}

@inproceedings{morad2023popgym, title={POPGym: Benchmarking Partially Observable Reinforcement Learning},
  author={Morad, Steven and Kortvelesy, Ryan and Bettini, Matteo and Liwicki, Stephan and Prorok, Amanda},
  booktitle={ICLR}, year={2023}, note={arXiv:2303.01859}}

@misc{mahoney_ltcb, title={Large Text Compression Benchmark}, author={Mahoney, Matt},
  howpublished={\url{https://www.mattmahoney.net/dc/text.html}}, note={enwik8 entries; accessed 2026-07-30}}
```

---

## Proposed handover updates (for the Hub)

1. **[C2W2] `track2-admissibility` landed. Headline: zero synthetic Track-2 candidates survive the ex-ante substitute audit.** MAD `compression` — `rival-recon`'s one admissible synthetic — **fails by arithmetic** (MAD's own iso-state 4096 exceeds the task's max payload by ≥73× fp32). RULER's non-retrieval families (VT, CWE/FWE) pass criterion 4 but fail the substitute audit (union-find; a counter). **Track 2 has exactly one admissible primary: real-data LM (enwik8/WikiText-103) at 26–47 M params** — and it survives on a criterion-3 leg the literature has already shown to be weak (Khandelwal ACL 2018: ≈200-token effective context, order ignored beyond ~50; Krause ICML 2018: 1.08 bpc from recency adaptation alone). **Mandatory harness consequence:** the dividend is reported on a **long-range slice**, never on aggregate bpc, with three substitute columns (smoothed kNN-over-store · n-gram/PPM · recency cache/dynamic evaluation).
2. **B′ is pre-registered and filed** (`.claude/outputs/track2-admissibility/PREREG-Bprime.md` + §2 here) **before the gate**: claim, have/need table, per-rival sizing, byte conventions, five numbered predictions (P1–P5) and five registered falsifiers of B′ itself (FB1–FB5). **FB4 ("the +0 B substitute is at ceiling for every family including attention ⇒ the instrument is invalid") should be B′'s first experiment** — it is cheap and it validates the protocol before it is spent on six families.
3. ⚠ **Pillar-4 narrowing (near-miss on the §6 falsifier).** MUNKEY (arXiv:2603.15033, ICML 2026) publishes key-deletion-as-unlearning with MIA AUROC≈0.5 — same mechanism family, same instrument, **not exact** (avg gap 0.56 ± 0.21), and a ViT classifier not a sequence memory. Our claim survives *only* as verified byte-exactness. **Never-quote items 13 and 14 added.**
4. ⛔ **New standing risk for C2W3 scoping:** byte-exact deletion is a **consequence of per-item atom groups** — the property §A2.3 blames for excluding compression. **The factored/shared store is likely to break it.** Requests a Head/Hub ruling: does C2W3 preserve a declared deletion-exact mode, or do we spend pillar 4 to fix pillar (a)?
5. **Rival status corrections:** Titans = **NeurIPS 2025** (not preprint), **still no official code**, chunk `b` **still unpinned**, **no seeds reported**; SDM has **official code** (`facebookresearch/sparse-delta-memory`, CC-BY-NC 4.0) but **requires an SM 80+ CUDA GPU — it cannot run on this machine**; FLA carries GDN/DeltaNet/Mamba2 but **neither Titans nor SDM**. **SDM's Table 1 state/param ratios are unusable** (two extractions disagree) — quarantine until human-verified.
6. **New byte-ledger rule (extends F2), applies to every arm:** for any memory with a **learned initial state** (TTT `W₀`, SDM `M₀`, our `V_θ` init), the **init is parameters (F1)** and only the **per-sequence deviation is state (F2)**; both declared. SDM's own abstract makes this explicit ("using it as a parametric memory") — adopt their vocabulary.
7. **Rider 4 is unfinished and needs a human or an authenticated fetch:** OpenReview bot-checks defeated both attempts. What is pinned: MAD 1 seed; Zoology/Based/Titans/SDM report none ⇒ the field norm is 1 seed, best-of-grid, no variance. Our ≥3-seed rule stays stricter and is stated as a strength.
8. **Version the "real Mamba" rule (one line, ready to paste):** Mamba-2 minimum (`mamba2.py`: `d_state=128, headdim=64, expand=2, d_conv=4, ngroups=1, chunk_size=256`), Mamba-3 (arXiv:2603.15569) named in limitations.
