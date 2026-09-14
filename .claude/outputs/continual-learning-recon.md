# continual-learning-recon — web-scout report
Task + acceptance criterion: pick the HG1 continual-learning target where a designed store + learned φ (replay-free episodic memory) can actually WIN — benchmark map (protocols pinned), ranked winnability audit with ONE recommendation + entry sketch, memory-module prior-art delta, retry-collision paragraph. Flag it plainly if the family is "solved by replay at this scale."
Status: **done** (read-only; no git footprint).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Three items, all for the Hub to convert into Phase-2 scoping decisions at the review that accepts this.**
> 1. ⛔ **Class-IL at small scale is effectively SOLVED by replay/generative-replay, and even by a *dumb balanced buffer* (GDumb).** DGR 90.8% / iCaRL 94.6% vs EWC/SI ≈ 20% on Split-MNIST Class-IL (van de Ven, verified table below). **The ONLY published white space for a replay-free store is "beat the replay-free regularizers, approach replay without a raw buffer" — the win is "best rehearsal-free method," NOT "beats replay."** This must travel with any CL claim or a referee kills it in one line (the C18-2 / Hopfield-"trivial-NN" lesson, re-instantiated).
> 2. ⛔ **The episodic-memory-module slot is NOT a category error — it is an established sub-line (iCaRL, SQHN, FearNet, key-value/MbPA), BUT SQHN (Nature Comms 2024) already occupies "energy-based Hopfield-class store, online-continual, replay-free, parameter-isolation."** The novelty surface for CLU is narrow and specific (designed *continuous* landscape + per-item decay/eviction schedules + relaxation/retry read + learned φ) — enumerated in §3. Do not re-derive SQHN's claim and call it new.
> 3. ⚠ **Weight-class fork is decisive and must be chosen before building:** from-scratch tiny (Split-MNIST/CIFAR-10, single-GPU) is CLU's only affordable slot; the modern Split-CIFAR-100/ImageNet-R Class-IL SOTA (~86–88%) is prompt-tuning on a **pretrained ViT** and is NOT comparable to a from-scratch entry. Pick the from-scratch slot and say so up front, or the numbers are incomparable (the w21-scout weight-class finding, verbatim).

---

## Answer first
**Recommended PRIMARY target: rehearsal-free (replay-buffer-free) Class-Incremental Learning on Split-MNIST → Split-CIFAR-10, from scratch, reported under van de Ven's three-scenario taxonomy with average accuracy + forgetting/BWT, positioned as an episodic-memory-module entry (iCaRL/SQHN-class).** This is the one CL slot where (a) the headline metric (per-task retention / forgetting / BWT) is *natively* what CLU's designed store measures, (b) the weight class is laptop/CSF3-scale, (c) an episodic-memory-module has clear published precedent and a fair referee slot, and (d) the competition CLU must beat — gradient-regularization methods (EWC/SI/LwF) — *provably collapses to ≈chance* in Class-IL, giving a real, defensible target. **FALLBACK: strict online / single-pass Class-IL (average online accuracy), the SQHN home turf**, where CLU's per-item masked write is the native operation and no replay is allowed by construction. **The honest caveat the Head must hear: at this scale Class-IL is *solved by replay* (DGR 90.8%, iCaRL 94.6%) and even by a dumb balanced buffer (GDumb is SOTA in its own setups). CLU's only winnable claim is "best replay-FREE method + per-item control + a retry-read curve no feedforward memory can draw" — not "beats replay."** That is a valid, publishable niche, but it is a niche, and it must be scoped as one.

---

## Item 1 — the benchmark map (protocols pinned)

### 1.1 The taxonomy every CL referee expects (use it or get desk-rejected)
**van de Ven & Tolias (2019), "Three scenarios for continual learning", arXiv:1904.07734; extended as van de Ven, Tuytelaars & Tolias (2022), "Three types of incremental learning", *Nature Machine Intelligence* 4:1185–1197, doi:10.1038/s42256-022-00568-3** [VERIFIED — ar5iv full text + NMI DOI]:
- **Task-IL** — task identity given at test time; model picks the task's own output head. *Easiest.*
- **Domain-IL** — same label set every task, input distribution shifts (e.g. permutation); task id **not** given. *Medium.*
- **Class-IL** — new classes each task, must infer across all classes seen; task id **not** given. *Hardest — and the one that matters.*
- **⭐ Load-bearing finding (this is the whole audit):** regularization methods "**completely fail when task identity needs to be inferred**" (Class-IL). Only replay works there.

### 1.2 The canonical small-scale baseline table — VERIFIED from primary (ar5iv 1904.07734)
Split-MNIST = **5 tasks × 2 classes**; Permuted-MNIST = **10 tasks × 10 classes**. Accuracy (%):

| Method | SM Task-IL | SM Domain-IL | **SM Class-IL** | PM Task-IL | PM Domain-IL | **PM Class-IL** |
|---|---|---|---|---|---|---|
| None (finetune) | 87.19 | 59.21 | **19.90** | 81.79 | 78.51 | **17.26** |
| EWC | 98.64 | 63.95 | **20.01** | 94.74 | 94.31 | **25.04** |
| Online EWC | 99.12 | 64.32 | **19.96** | 95.96 | 94.42 | **33.88** |
| SI | 99.09 | 65.36 | **19.99** | 94.75 | 95.33 | **29.31** |
| LwF | 99.57 | 71.50 | **23.85** | 69.84 | 72.64 | **22.64** |
| DGR (gen. replay) | 99.50 | 95.72 | **90.79** | 92.52 | 95.09 | **92.19** |
| DGR+distill | 99.61 | 96.83 | **91.79** | 97.51 | 97.35 | **96.38** |
| iCaRL (exemplar mem.) | — | — | **94.57** | — | — | **94.85** |

**Read this table the way a referee will:** in Class-IL, EWC/SI ≈ **20%** (Split-MNIST) — i.e. ~chance for a 2-way head with no task id, a *complete* failure; replay/generative-replay and the **exemplar-memory** method iCaRL are the *only* things that work (90–95%). **This is simultaneously the opportunity (regularizers fail, a memory store is the answer) and the threat (replay already wins).**

### 1.3 Canonical baselines + primary refs (the competition, with tuning discipline note)
| method | class | ref | tuning note (N78 rescued-baseline discipline) |
|---|---|---|---|
| EWC | prior/regularization | Kirkpatrick et al., *PNAS* 2017, arXiv:1612.00796 | fails Class-IL by design; **do not present its collapse as a CLU win** — it is the known null |
| Online EWC | regularization | Schwarz et al., ICML 2018 | same |
| SI | regularization | Zenke, Poole & Ganguli, ICML 2017, arXiv:1703.04200 | same |
| LwF | distillation | Li & Hoiem, TPAMI 2017, arXiv:1606.09282 | weak in Class-IL |
| GEM | memory + constrained grad | Lopez-Paz & Ranzato, NeurIPS 2017, arXiv:1706.08840 | **BWT/FWT metrics originate here** |
| A-GEM | memory (cheaper GEM) | Chaudhry et al., ICLR 2019, arXiv:1812.00420 | — |
| ER / tiny episodic memory | replay buffer | Chaudhry et al. 2019, arXiv:1902.10486 | **strong, must be tuned hard or the win is fake (N78)** |
| iCaRL | exemplar memory + NME classifier | Rebuffi et al., CVPR 2017, arXiv:1611.07725 | **the direct "episodic-memory-module" incumbent** |
| DER / DER++ | logit-replay | Buzzega et al., NeurIPS 2020, arXiv:2004.07211 | current strong baseline; Mammoth's home method |
| DGR | generative replay | Shin et al., NeurIPS 2017, arXiv:1705.08690 | the Class-IL winner at small scale |
| **GDumb** | **dumb balanced buffer + retrain** | Prabhu et al., ECCV 2020 | ⛔ **the pathology check — SOTA "in almost all cases" at matched memory; any entry must beat GDumb or it is "solved by a buffer"** |

### 1.4 The harness the field actually uses
- **Mammoth** (aimagelab, official DER/DER++ codebase) — de-facto standard for rehearsal-based method comparison; implements EWC/SI/A-GEM/iCaRL/DER++/… on Sequential-MNIST/CIFAR-10/CIFAR-100/Tiny-ImageNet (Task-IL + Class-IL). **Recommend as the primary harness** — its baselines are the ones referees trust. github.com/aimagelab/mammoth.
- **Avalanche** (ContinualAI, CVPRW 2021, arXiv:2302.01766; MIT license) — modular benchmarks (Split/Permuted/Rotated-MNIST, Split-CIFAR-10/100/110, CORe50, CLEAR) + strategies (Naive/Replay/GDumb/LwF/GEM/A-GEM/EWC/SI/…). Its sibling **continual-learning-baselines** repo pins reproduced numbers for EWC/SI/GEM/AGEM/LwF/iCaRL/GDumb. **Use Avalanche for benchmark/stream plumbing + the online-CL fallback; cross-check numbers against Mammoth.**
- **⚠ Metrics — pin the formulas from GEM (arXiv:1706.08840):** ACC = (1/T)Σᵢ A_{T,i}; **BWT = (1/(T−1)) Σᵢ₌₁^{T−1} (A_{T,i} − A_{i,i})** (negative = catastrophic forgetting); FWT = (1/(T−1)) Σᵢ₌₂^T (A_{i−1,i} − Ā_i). "Forgetting" (Chaudhry) = mean over tasks of (max past acc − final acc). Report ACC + forgetting/BWT together — a referee reads both.

### 1.5 Other families, pinned (for the audit, then mostly ruled out)
- **Permuted-/Rotated-MNIST (Domain-IL)** — regularization already works (EWC ≈ 94%), so *no* white space for a memory module. Low priority.
- **CLEAR** (Lin et al., NeurIPS 2021 D&B, arXiv:2201.06289) — real-world temporal-evolution imagery 2004–2014; **streaming protocol** (test on near-future); semi-supervised. Interesting but domain-shift-flavoured, not interference-limited; larger images.
- **CLOC** (Cai et al., ICCV 2021, arXiv:2108.09020) — 39M images / 700+ classes / geolocation stream; metric = **Average Online Accuracy**. Online CL at *scale* — wrong weight class.
- **Online/streaming CL, single-pass, batch=1** — each sample learned-then-discarded; metric = average online accuracy. **This is the FALLBACK regime** (§2), native to CLU's per-item write. Supported in Avalanche; SQHN's home.

---

## Item 2 ⭐ — the winnability audit (the deliverable)

Ranked by (a) native-metric fit × (b) weight class × (c) memory-module legibility × (d) pathology risk.

### #1 — RECOMMENDED PRIMARY: rehearsal-free Class-IL, Split-MNIST → Split-CIFAR-10, from scratch
- **(a) native metric:** ⭐ **highest fit.** Forgetting / BWT / per-task retention is *literally* what `sequential-write-interference` measured (designed-store gate retention **1.000 vs 0.16** ungated at K=16). "Does a new write destroy stored items" **is** the Class-IL question.
- **(b) weight class:** tiny, from-scratch, single-GPU/CSF3. ✅ (Split-MNIST first; Split-CIFAR-10 as the harder rung once φ exists.)
- **(c) memory-module slot:** ✅ **fair and precedented** — iCaRL (exemplar memory + NME), SQHN (energy store), FearNet all occupy it. A designed-store entry is *not* a category error.
- **(d) pathology risk:** ⚠ **high and must be pre-empted.** GDumb (dumb balanced buffer) is SOTA in its own setups; DGR/iCaRL hit 90–95%. ⇒ **CLU cannot claim "solves Class-IL"; it can claim "best *replay-free* method + per-item retention control + retry-read curve," at matched or smaller memory than iCaRL/GDumb.**
- **Predicted loss modes:** (i) referee says "your designed store is just a fancy replay buffer" → mitigate by *not* storing raw exemplars (store in the landscape via φ) and reporting memory footprint vs iCaRL/GDumb; (ii) "GDumb beats you at matched memory" → the entry MUST include GDumb at matched budget as a mandatory baseline (N78); (iii) "learned-everything CLU is worst-of-four" (the K=64 result, `sequential-write-interference` §3.2) → **this is why the doctrine is designed-store + learned-φ, NOT learned-everything**; the entry must use the designed store, whose gate gives 1.000.

### #2 — RECOMMENDED FALLBACK: strict online / single-pass Class-IL (average online accuracy)
- **(a) native metric:** ⭐ very high — single-pass stream = **one masked write per item**, exactly CLU's operation; no replay allowed by construction ⇒ CLU's replay-free advantage is native, not a handicap.
- **(b) weight class:** small (MNIST/CIFAR-10 online). ✅
- **(c) slot:** ✅ SQHN (Nature Comms 2024) proves the slot exists and is peer-reviewed.
- **(d) pathology:** ⚠ SQHN already occupies it → novelty must be sharp (continuous designed landscape + per-item decay + relaxation/retry read). Referee community for online-CL is smaller/split vs offline Class-IL, so **legibility is lower** — hence fallback, not primary.
- **Predicted loss modes:** SQHN-preemption ("you re-did the energy-store online-CL story"); smaller referee pool discounts the venue value.

### #3 — Domain-IL (Permuted/Rotated-MNIST). **Not recommended.**
- Regularization already works (EWC ≈ 94%); no memory-module white space; the interference story doesn't bite. Keep only as a taxonomy-completeness row.

### ⛔ Ruled out for near-term (weight class): Split-CIFAR-100 / ImageNet-R Class-IL SOTA
- Modern SOTA (~86–88% CIFAR-100, ~69–72% ImageNet-R) is **prompt-tuning on a pretrained ViT** (L2P/DualPrompt/CODA-Prompt). A from-scratch CLU is not comparable; requires a host backbone. Cite as related work; do not enter.

### The ranking, plainly
`#1 rehearsal-free Class-IL (SM→SC10, from scratch) ≫ #2 online single-pass Class-IL ≫ #3 Domain-IL ≫⛔ CIFAR-100/ImageNet-R (pretrained, wrong weight class)`

**⭐ The honest recon headline the Head asked for:** *Continual learning at laptop scale is not "unsolved" — Class-IL is solved by replay (90–95%) and even by a dumb buffer (GDumb), while gradient-regularization fails at ≈chance.* **The winnable move is not to beat replay; it is to own the replay-FREE, bounded-memory, per-item-controllable episodic store — the slot iCaRL/SQHN opened and did not close — and to add the one thing none of them have: an accuracy-vs-compute *retry* read curve.** That is a real target with a fair referee slot, but it is a niche, and the entry lives or dies on the mandatory GDumb + tuned-ER baselines and the "not just a replay buffer" framing.

---

## Item 3 — memory-module CL prior art + CLU's novelty surface

**Who has run associative-memory / key-value / energy stores on CL — and what's left for a designed *continuous* landscape:**

- **SQHN — Sparse Quantized Hopfield Network** (Nkambou-adjacent; **Nature Communications 2024, s41467-024-46976-4; arXiv:2307.xxxx**) [VERIFIED — fetched author-page text]. Energy-based Hopfield-class store; **online, single-pass**; **replay-free**, avoids forgetting via **parameter isolation (sparse one-hot codes) + neuron growth + LR decay**; tasks = online-continual auto-association (class- & domain-incremental), noisy-encoding, episodic-recognition; baselines = MHN (SGD/Adam/**EWC++**/episodic-replay), predictive-coding (BayesPCN). ⭐ **The nearest published neighbour and the main preemption.** *Left for CLU:* it is **discrete/quantized one-hot** with hard parameter isolation; CLU is a **continuous designed landscape** with (i) a **fiber payload channel** separate from the address, (ii) **per-item decay/eviction schedules** (γ_φ friction field — no SQHN analogue), (iii) **relaxation + retry read** (accuracy-vs-compute — no SQHN analogue), (iv) **learned φ read-in**.
- **iCaRL** (Rebuffi et al., CVPR 2017, arXiv:1611.07725) — the canonical **episodic-memory-module** for Class-IL: herding-selected exemplars + nearest-mean-of-exemplars classifier. **Proof the slot is fair and refereed** (94.57% SM Class-IL). *Left for CLU:* iCaRL stores **raw exemplars** in a growing buffer; CLU stores in a **bounded parametric landscape** with controllable per-item retention — the memory-footprint and decay-control axes are open.
- **Modern-Hopfield CL** (Anon., "Continual Learning in Modern Hopfield Networks with an Application to Diffusion Models", arXiv:2605.27975, 2026) [SECONDARY — abstract only] — MHN + intrinsic-forgetting-as-energy-increase analysis; pairs with diffusion. Recent; cite to show the line is active.
- **Complementary Learning Systems / generative-memory** — Spens & Burgess (2024, VAE+MHN generative replay); HiCL / cortico-hippocampal hybrids (Nature Comms 2025, s41467-025-56405-9); Deep Generative Dual Memory (ICLR'18). These are **replay-via-a-memory** — CLU's replay-FREE store is the contrast.
- **Key-value / episodic memory in lifelong learning** — d'Autume et al., "Episodic Memory in Lifelong Language Learning", NeurIPS 2019, arXiv:1906.01076 (key-value store + sparse experience replay + local adaptation = **MbPA**, Sprechmann et al. ICLR 2018); Tyulmankov et al., "Biological learning in key-value memory networks", NeurIPS 2021. *Left for CLU:* these are **discrete slot** memories read by soft attention; CLU reads by **physical relaxation** and can **retry** — the adaptive-compute read is the delta.

**Novelty surface for CLU-as-episodic-memory, enumerated (what a designed-landscape store can own):**
1. **Per-item decay / eviction schedules** in one store (permanent + scheduled-fade memories, retrieved on their own schedule) — no exemplar-buffer or Hopfield analogue.
2. **Admission gating on a designed store** — the measured 1.000-vs-0.16 retention (designed+gated) is a real, replay-free anti-interference mechanism; maps onto the `controller-mvp` admission/placement/eviction.
3. **Relaxation + retry read** = an accuracy-vs-compute curve (Item-3 retry study) — feedforward exemplar/soft-attention memories cannot draw it.
4. **Learned φ read-in** decoupled from the fixed store (the Phase doctrine) — the differentiator from SQHN's fixed sparse code.

**Verdict:** the slot is real and fair; the novelty is narrow but non-empty and concentrated in (1)+(3). **Do not claim "energy store for replay-free CL" as new — SQHN owns it. Claim per-item-controllable retention + adaptive-compute retrieval on a continuous designed landscape with a learned φ.**

### Retry-collision paragraph (for `retry-compute-study` Item 3)
`retry-compute-study`'s created benchmark (accuracy-vs-compute for iterative **relaxation-based retrieval from a parametric store**, controls: retry-all / ensemble-of-k / random-kick / matched-compute feedforward / **Hopfield-k-steps**) does **not** collide with the mainstream "test-time-compute-for-retrieval" prior art, which is **RAG/LLM-external-KB** (test-time scaling for generative retrieval, FAIR-RAG iterative refinement, adaptive iterative RAG — all 2025, all about querying an external document index with an LLM, a different setting). **The genuine neighbours are two, both already anticipated as controls:** (i) **Energy-Based Transformers** (Gladstone et al., arXiv:2507.02092, 2025) — energy-as-verifier test-time "System-2" compute, the framing-level incumbent that must anchor related work; (ii) **modern-Hopfield multi-step retrieval** (more update iterations = the `Hopfield-k-steps` control). ⇒ **No spec change needed; the created benchmark is defensible provided it (a) frames itself as associative-memory read-refinement / anytime retrieval — NOT "test-time RAG" — and (b) cites EBT + Hopfield-k-steps as the baselines it already lists.** Single-sourced risk: EBT author list beyond first author unverified (carried from w21 scout).

---

## Confidence & gaps
**VERIFIED from primary this session:** van de Ven three-scenario Split-MNIST/Permuted-MNIST full table (ar5iv 1904.07734) incl. Class-IL EWC/SI ≈ 20% vs DGR 90.8% / iCaRL 94.6%; the three-scenario taxonomy + "regularization fails Class-IL" claim; GEM BWT/FWT/ACC formulas (search-verified against arXiv:1706.08840); SQHN model/protocol/baselines/replay-free claim (Nature Comms author-page fetch); Avalanche benchmark+strategy list (arXiv:2302.01766 + official docs); Mammoth as DER/DER++ codebase; GDumb "SOTA in almost all cases" claim (Oxford PDF search); Farquhar & Gal desiderata/prior-focused-bias critique (arXiv:1805.09733); CLEAR streaming protocol (arXiv:2201.06289); CLOC average-online-accuracy at scale (arXiv:2108.09020).
**SECONDARY / single-sourced — verify before printing:** SQHN exact arXiv id (Nature Comms DOI is solid; preprint id not pinned) · arXiv:2605.27975 (abstract only) · CIFAR-100/ImageNet-R prompt-SOTA 86–88%/69–72% (carried from w21 scout, search-summary) · EBT author list.
**Could not fetch:** SQHN and van de Ven PDFs are paywalled/binary-encoded respectively (used author-page + ar5iv instead — numbers are from ar5iv HTML, reliable).
**Search next (for the engineer building the entry):** (1) pin GDumb's exact Split-MNIST/CIFAR-10 numbers at the memory budget CLU will use — the mandatory bar; (2) confirm SQHN's exact online-continual auto-association accuracy vs MHN-replay, to size the fallback; (3) Mammoth's current DER++/ER Class-IL Split-CIFAR-10 from-scratch numbers as the head-to-head table; (4) whether any *continuous* (non-quantized) energy/Hamiltonian store has been run on standard Split-MNIST Class-IL (a negative strengthens novelty).

---

## Bibtex-ready refs
```bibtex
@article{vandeven2019three,
  title={Three scenarios for continual learning},
  author={van de Ven, Gido M. and Tolias, Andreas S.},
  journal={arXiv preprint arXiv:1904.07734}, year={2019}}

@article{vandeven2022three,
  title={Three types of incremental learning},
  author={van de Ven, Gido M. and Tuytelaars, Tinne and Tolias, Andreas S.},
  journal={Nature Machine Intelligence}, volume={4}, number={12}, pages={1185--1197}, year={2022},
  doi={10.1038/s42256-022-00568-3}}

@inproceedings{lopezpaz2017gem,
  title={Gradient Episodic Memory for Continual Learning},
  author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
  booktitle={NeurIPS}, year={2017}, note={arXiv:1706.08840; defines ACC/BWT/FWT}}

@inproceedings{chaudhry2019agem,
  title={Efficient Lifelong Learning with A-GEM},
  author={Chaudhry, Arslan and Ranzato, Marc'Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
  booktitle={ICLR}, year={2019}, note={arXiv:1812.00420}}

@inproceedings{rebuffi2017icarl,
  title={iCaRL: Incremental Classifier and Representation Learning},
  author={Rebuffi, Sylvestre-Alvise and Kolesnikov, Alexander and Sperl, Georg and Lampert, Christoph H.},
  booktitle={CVPR}, year={2017}, note={arXiv:1611.07725; exemplar episodic memory + NME}}

@inproceedings{buzzega2020der,
  title={Dark Experience for General Continual Learning: a Strong, Simple Baseline},
  author={Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Abati, Davide and Calderara, Simone},
  booktitle={NeurIPS}, year={2020}, note={arXiv:2004.07211; DER/DER++; Mammoth codebase}}

@inproceedings{shin2017dgr,
  title={Continual Learning with Deep Generative Replay},
  author={Shin, Hanul and Lee, Jung Kwon and Kim, Jaehong and Kim, Jiwon},
  booktitle={NeurIPS}, year={2017}, note={arXiv:1705.08690}}

@inproceedings{kirkpatrick2017ewc,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and others},
  journal={PNAS}, volume={114}, number={13}, pages={3521--3526}, year={2017},
  note={arXiv:1612.00796}}

@inproceedings{zenke2017si,
  title={Continual Learning Through Synaptic Intelligence},
  author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
  booktitle={ICML}, year={2017}, note={arXiv:1703.04200}}

@inproceedings{prabhu2020gdumb,
  title={GDumb: A Simple Approach that Questions Our Progress in Continual Learning},
  author={Prabhu, Ameya and Torr, Philip H. S. and Dokania, Puneet K.},
  booktitle={ECCV}, year={2020},
  note={dumb balanced buffer + retrain; SOTA in own setups — the pathology check}}

@article{farquhar2018robust,
  title={Towards Robust Evaluations of Continual Learning},
  author={Farquhar, Sebastian and Gal, Yarin},
  journal={arXiv preprint arXiv:1805.09733}, year={2018}}

@article{sqhn2024,
  title={A sparse quantized Hopfield network for online-continual memory},
  author={and others},
  journal={Nature Communications}, year={2024}, doi={10.1038/s41467-024-46976-4},
  note={energy store, online single-pass, replay-free via parameter isolation; nearest neighbour to CLU-as-CL-memory}}

@inproceedings{lomonaco2021avalanche,
  title={Avalanche: an End-to-End Library for Continual Learning},
  author={Lomonaco, Vincenzo and others},
  booktitle={CVPR Workshops (CLVision)}, year={2021}, note={arXiv:2302.01766; MIT}}

@inproceedings{dautume2019episodic,
  title={Episodic Memory in Lifelong Language Learning},
  author={d'Autume, Cyprien de Masson and Ruder, Sebastian and Kong, Lingpeng and Yogatama, Dani},
  booktitle={NeurIPS}, year={2019}, note={arXiv:1906.01076; key-value episodic memory + MbPA}}

@inproceedings{lin2021clear,
  title={The CLEAR Benchmark: Continual LEArning on Real-World Imagery},
  author={Lin, Zhiqiu and Shi, Jia and Pathak, Deepak and Ramanan, Deva},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2021}, note={arXiv:2201.06289}}

@article{cai2021cloc,
  title={Online Continual Learning with Natural Distribution Shifts},
  author={Cai, Zhipeng and Sener, Ozan and Koltun, Vladlen},
  booktitle={ICCV}, year={2021}, note={arXiv:2108.09020; CLOC, Average Online Accuracy}}
```

---

## Proposed handover updates (for the Hub)
1. **⭐ Phase-2 target picked (pending Hub/Head ratify): PRIMARY = rehearsal-free Class-IL, Split-MNIST → Split-CIFAR-10, from scratch, three-scenario taxonomy + forgetting/BWT, episodic-memory-module framing; FALLBACK = strict online single-pass Class-IL (average online accuracy).** Entry = **designed store (`AtomDictionaryPotential`) + learned φ + `controller-mvp` admission/placement/eviction**, replay-free, with **per-item decay + retry read** as the differentiators.
2. **⛔ Binding scope caveat for the CL claim (needs an owner):** *Class-IL at small scale is solved by replay (DGR 90.8% / iCaRL 94.6%) and by GDumb; CLU's winnable claim is "best replay-FREE method + per-item retention control + adaptive-compute retry read," NOT "beats replay."* **Mandatory baselines in any table: tuned ER + iCaRL + GDumb at matched memory (N78), plus EWC/SI as the known-null.** Without GDumb-at-matched-memory the entry is "solved by a buffer."
3. **Harness decision:** build the entry inside/against **Mammoth** (referee-trusted Class-IL baselines) with **Avalanche** for the online-CL fallback stream. Pin BWT/forgetting formulas from GEM (arXiv:1706.08840).
4. **Prior-art must-cites + delta:** **SQHN (Nature Comms 2024)** is the nearest neighbour (energy store, online, replay-free) — cite and differentiate on *continuous landscape + per-item decay + retry + learned φ*; iCaRL is the fair-slot precedent; MbPA/key-value + CLS/generative-replay bound the space.
5. **Retry-collision (for `retry-compute-study`): NO spec change.** The created benchmark doesn't collide with RAG test-time-retrieval prior art (different setting); its genuine neighbours (EBT, Hopfield-k-steps) are already its controls — just frame as associative-memory read-refinement and cite them.
6. **Weight-class guardrail:** do NOT chase Split-CIFAR-100/ImageNet-R Class-IL SOTA — it is pretrained-ViT prompt-tuning, incomparable to from-scratch CLU. Cite as related work only.
