# scout-benchmark-targeting — web-scout report

**Task + acceptance criterion:** which popular benchmarks actually match what CLU is — parametric/contextual verdict, capacity-per-parameter comparison + legitimacy, ranked benchmark shortlist with SOTA/weight class, retrieval-cost check, prior-art debts closed-or-flagged.
**Status: done.** All 5 items answered. **Both blocking prior-art debts are CLOSED** (Ramsauer's constant, UnICORNN, SRNN-ISO). No repo edits, no git footprint.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary).**
> 1. ⛔ **`K_max = 4·2^d` must NEVER be quoted as parameter-efficiency or as "exponential capacity" without its denominator.** The designed landscape's parameter count is `K·(d+1)` — it grows **linearly in the number of stored items** (verified by reading `BallRegisterPotential`: `payloads` (K,), `centers` (K,d)). The exponential is in *address dimension*, not in parameters. Per-parameter it is **O(1) bits — my derivation gives ≈1.1–1.6 bits/param, i.e. BELOW the transformer's measured 2 bits/param** (§2.3). A reviewer will do this division in ten seconds.
> 2. ⛔ **The MQAR 0-of-3 loss must be re-scoped BUT NOT as "we were tested on the wrong thing and would otherwise have won."** The Hub's parametric/contextual diagnosis is **correct and standard** (§1) — *and* Arora et al.'s **Theorem 3.1 (ICML 2024) proves any causal recurrent model needs Ω(N) bits of state to solve MQAR**. CLU is a fixed-state recurrence. **It was structurally unwinnable, which is a stronger and more citable statement than "unfair".** Both halves must travel together.
> 3. ⛔ **The retrieval-cost claim as written is FALSE for the shipped implementation** and **not novel in the form where it is true** (§4). Product-Key Memory (NeurIPS 2019) already engineers `O(√K)` lookup over 1M slots; Memory Layers at Scale (ICML 2025) does 128B memory params at **zero added FLOPs**. Do not present O(1)-in-items retrieval as a CLU contribution.
> 4. ⚠ **Prior-art table additions, one of them urgent: Titans (NeurIPS 2025, Google)** writes to a neural memory *at test time* by **gradient descent with momentum and weight decay** — i.e. discretized damped second-order dynamics on an associative-memory loss. That is the nearest published neighbour to CLU's "write into `V_θ` by relaxation", it is 6 months old, and it ships 2M-token results. Also: **Hu, Wu & Liu (NeurIPS 2024)** prove modern-Hopfield optimal capacity is achieved by **optimal spherical codes** — which is, term for term, the packing/`d_eff` law `address-space-dimension-scaling` measured empirically.

---

## Answer first

**The Hub's diagnosis is right and is a recognized distinction** — "knowledge in weights vs knowledge in context" is standard terminology (Chan et al. NeurIPS 2022 "in-weights vs in-context"; Lewis et al. NeurIPS 2020 "parametric vs non-parametric memory") and a reviewer will accept it without argument. **But it does not rescue the MQAR result**, because a theorem (Arora et al., ICML 2024, Thm 3.1) says CLU's model class provably cannot win there, and because the axis the Hub wants — **bits of knowledge per parameter, where the field's number is 2 bits/param (Allen-Zhu & Li, arXiv:2404.05405, verified) — is one CLU currently LOSES**: the designed landscape spends `K(d+1)` parameters to hold `K` items, giving ≈1.1–1.6 bits/param by my derivation, and it is not learned, so the comparison is not even admissible as-is. **The one benchmark family where CLU's actual measured strength is both competitive and legitimately comparable is the modern-Hopfield / dense-associative-memory capacity protocol** (MNIST/CIFAR-10 half-masked retrieval, capacity-vs-stored-memories curves) — because in *that* line nothing is learned either: patterns are written in closed form, parameters grow with the number of memories, and capacity-vs-dimension is the field's own axis. **That is the target I recommend, and it is the only one where the w20 "learning destroys everything" blocker does not bind.**

---

## Item 1 — parametric vs contextual: the framing is standard, not folk

**Verdict: [VERIFIED] a recognized distinction with multiple canonical anchors. The Hub is not reifying a folk taxonomy.** Three independent, primary-verified formulations:

- **"parametric memory" / "non-parametric memory"** — Lewis et al. (2020), *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*, NeurIPS 2020, arXiv:2005.11401. Abstract, fetched verbatim: *"We introduce RAG models where the **parametric memory** is a pre-trained seq2seq model and the **non-parametric memory** is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever."* [VERIFIED — primary abstract]
- **"in-weights learning" vs "in-context learning"** — Chan et al. (2022), *"Data Distributional Properties Drive Emergent In-Context Learning in Transformers"*, NeurIPS 2022, arXiv:2205.05055. Abstract, fetched verbatim: *"…how future work might encourage both **in-context and in-weights learning** in domains beyond language."* Also verbatim: *"in-context learning traded off against more conventional weight-based learning, and models were unable to achieve both simultaneously"* — i.e. **the two are treated as competing mechanisms, which is exactly the Hub's line.** [VERIFIED — primary abstract]
- **the mechanism on the weights side** — Geva et al. (2021), *"Transformer Feed-Forward Layers Are Key-Value Memories"*, EMNLP 2021, pp. 5484–5495, doi:10.18653/v1/2021.emnlp-main.446: FFN layers *operate as key–value memories*, keys = input patterns, values = output distributions. [VERIFIED — venue/pages/DOI; abstract via ACL Anthology listing, not full text]
- **the architectural statement of the split, current** — Behrouz, Zhong & Mirrokni (2025), *"Titans: Learning to Memorize at Test Time"*, NeurIPS 2025, arXiv:2501.00663: *"attention due to its limited context but accurate dependency modeling performs as a **short-term memory**, while neural memory due to its ability to memorize the data, acts as a **long-term, more persistent, memory**."* [VERIFIED — fetched from arXiv HTML]

**Where the line actually sits, and the caveat that matters for us.** The line is *where the write happens*, not *when*: parametric = information written into weights by an optimization step; contextual = information present in the activations/KV-cache of the current forward pass. **Titans deliberately blurs it** — it performs parametric writes *at inference time* — so "parametric ⇒ written during training" is **not** safe to assert. CLU's vision (write to `V_θ`, address with a small pointer) is a **parametric** memory in this taxonomy, and Titans is its nearest published relative.

⚠ **The honest consequence for the MQAR post-mortem.** The category error is real: MQAR presents items at inference and requires retrieval from the same sequence — a KV-cache task, and attention's flatness (0.996, drop 0.001) is exactly the O(T)-cache explanation the engineer already gave. **But re-framing does not convert the loss into a win.** See §4/§3.1: there is a published Ω(N)-state lower bound for MQAR. The correct write-up is *"we benchmarked a parametric primitive on a contextual task; a lower bound tells us the class cannot win it; here is the parametric benchmark instead"* — not *"the benchmark was unfair."*

---

## Item 2 ⭐ — capacity per parameter: the constant, and why our number is not yet admissible

### 2.1 The result the Hub half-remembered — it is real, and it is 2 bits/param

**Allen-Zhu & Li (2024), *"Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws"*, arXiv:2404.05405** [VERIFIED — abstract from arXiv abs page; Results 1/2/3 and the capacity-ratio formula from ar5iv full text]

- Abstract verbatim: *"we establish that language models **can and only can store 2 bits of knowledge per parameter**, even when quantized to int8… a 7B model can store 14B bits of knowledge, surpassing the English Wikipedia and textbooks combined."*
- **The measured quantity ("capacity ratio")**, verbatim from ar5iv:
  `R(F) = [ N·log₂(N₀/e^{p₁}) + N·K·log₂(D^c/e^{p₂}) + K·D·log₂(T^L/(D·e^{p₃})) ] / P`
  with `p₁,p₂,p₃` the cross-entropy losses on names / values / first-chunks and **`P` the model's parameter count**. For the biography dataset it reduces to `R(F) = [N log₂(N₀/e^{p₁}) + N log₂(S₀/e^{p₂})]/P`.
- **Result 1 verbatim:** *"When trained for 1000 exposures on bioS(N), with N ranging from 10K to 10M, GPT2 models with sizes from 1M to 0.5B parameters (irrespective of depth or width) demonstrate the following: (a) the peak capacity ratio R(F) consistently exceeds R(F)≥2."*
- **Scope conditions (all load-bearing):**
  | condition | effect on the 2 bits/param |
  |---|---|
  | **1000 exposures** per knowledge piece | `R ≥ 2` |
  | **100 exposures** | `R ≥ 1` — **halves** |
  | int8 quantization | unchanged (`≥2`) |
  | **int4** | **>2× reduction** (Result 8) |
  | GatedMLP (LLaMA/Mistral) @100 exposures | **1.3× lower** (Result 6–7); comparable at 1000 exposures (Result 5) |
  | MoE, 32 experts | only 1.3× lower, at 88% fewer active params (Result 9) |
  | junk data 7:1 @100 exposures | **20× loss**, mitigated to 2× by a domain token (Results 10–12) |
  | parameter counting | **unused embedding rows excluded** (bioS uses 3,275 of 50,256 tokens ⇒ embeddings counted as `3275×64h`) |
- **Independent second anchor, different protocol, different number:** Morris et al. (2025), *"How much do language models memorize?"*, arXiv:2505.24832 (Meta/Google DeepMind/NVIDIA/Cornell): *"our models consistently memorize between **3.5 and 3.6 bits per parameter**"*, **α = 3.64** (half precision), **3.51 bf16 vs 3.83 fp32**; protocol = GPT-style models 100K–20M params, 1–8 layers, hidden 32–512, trained on **uniformly random bitstrings** (V=2048, S=64). [VERIFIED — arXiv HTML v3]
  ⇒ **The field has two constants for two different notions**: ~2 bits/param for *extractable factual knowledge*, ~3.6 bits/param for *raw unintended memorization*. **Quote whichever you use with its protocol; they are not interchangeable.**
- **The physics anchor a CHLU paper should use:** Gardner's critical storage capacity for a perceptron, `α_c = 2` patterns per synapse (Gardner 1988, *J. Phys. A* **21**, 257, "The space of interactions in neural network models"), commonly stated as ≈2 bits/synapse; Cover (1965) function-counting gives the same 2N. ⚠ **[SECONDARY — IOPscience/search summaries only; I did not read Gardner's primary text this session, and one summary muddled it as "of the order of 1 bit/synapse".] Verify before printing the constant.** The rhetorical payoff if it holds is large: *the transformer's empirical 2 bits/param sits exactly at the classical perceptron bound.*

### 2.2 Can `K_max = 4·2^d` go on that axis? — **No, not as it stands. Three independent blockers.**

1. **The parameter count is the wrong shape.** I read `chlu/core/memory_potentials.py`: `BallRegisterPotential` carries `payloads: (K,)` and `centers: (K,d)`. **Storage parameters = `K(d+1)` and grow linearly in items.** Allen-Zhu's `P` is a *fixed* budget into which knowledge is compressed. `K_max = 4·2^d` is a statement about **address resolution**, not about compression, and the two axes are orthogonal. Putting `4·2^d` next to "2 bits/param" without this sentence is the single most likely way to earn a retraction.
2. **Nothing is learned.** The `dim-scaling` landscape is hand-designed end-to-end (its own flag-provenance table says so: *"Designed vs learned: everything… Nothing here is evidence of emergence"*). Allen-Zhu's `R(F)` measures what **training** put into weights, from **natural-language** text, with a **cross-entropy readout on a held-out extraction task**. A designed landscape with a hand-built codebook reader is not the same measurement.
3. **The denominator is not comparable even in principle.** `R(F)` divides by the **whole model**, including everything that parses the query and extracts the fact. CHLU would be dividing by the storage medium alone. **Like-for-like requires the reader to be counted.**

### 2.3 What the number would be if you did it anyway — ⭐ do this as a PRE-REGISTRATION, not as a result

Two defensible accountings of the *designed* landscape at its measured operating point (d=8, `K_max`=1024 codebook / ≥2048 addressing, `Δ_req ≈ 3.1·max(w,σ) ≈ 0.43`, `d_eff ≈ 0.72–0.83·d`), both **MY DERIVATION from the w20 report's own numbers, not a measurement**:

| accounting | stored bits | params | bits/param |
|---|---|---|---|
| **payload-only** (what the codebook read recovers): `K·log₂A` bits with alphabet `A=K` | 1024·10 = 10,240 | 1024·9 = 9,216 | **≈1.11** |
| **address-information** (each address coordinate resolves `2R/Δ_req ≈ 4.65` ⇒ `log₂ ≈ 2.22` bits): `log₂K_max = d_eff·2.22` bits per item over `d+1` params | — | — | **→ ≈1.6 asymptotically** (`1.6d/(d+1)`) |

⇒ **Predicted: CLU's designed landscape stores O(1) bits per parameter, ≈1.1–1.6, i.e. below the transformer's 2 and well below Morris's 3.6.** It is the *same order*, which is not embarrassing — but it is **not a win, and the exponential-in-`d` headline gives no advantage on this axis at all.** ⚠ Note the pleasing coincidence that `log₂(2R/Δ_req) ≈ 2.22 ≈ 2` bits per address parameter, matching Gardner — that is an algebraic restatement of the measured packing law, **not independent evidence**, and must be labelled as such if it ever appears.

### 2.4 The comparison that IS legitimate

**Capacity vs dimension of the associative space, with patterns/items stored explicitly — the modern-Hopfield axis.** Both sides then have parameters growing with stored items, both are written in closed form, and the quantity compared (`K_max(d)`) is identical.

| model | capacity | source |
|---|---|---|
| classical Hopfield (Hebbian) | `P_max ≈ 0.138 N` | Amit–Gutfreund–Sompolinsky 1985 [VERIFIED, standard] |
| Demircigil et al. 2017 (exponential interaction) | `C ≅ 2^{d/2}` | J. Stat. Phys., doi:10.1007/s10955-017-1806-y [VERIFIED via 2 independent restatements] |
| Ramsauer et al. 2021 (continuous, = attention) | `M ≥ √p · C^{(d−1)/4}` (lower bound) | see §5.1 — **debt now closed** |
| Hu, Wu & Liu 2024 (**tight**) | `M* ≍ c^{D_Φ}`, achieved by an **optimal spherical code** | NeurIPS 2024, arXiv:2410.23126 [VERIFIED, arXiv HTML] |
| **CLU designed register (measured)** | **`K_max = 4·2^d`** (R²=1.0000, d=2–8), `= (2R/Δ_req)^{d_eff}` | `address-space-dimension-scaling` |

⭐ **This is the finding with the most leverage in the whole report.** Hu, Wu & Liu prove that optimal modern-Hopfield capacity is attained when memories form an **optimal spherical code** (maximizing minimum angular separation), giving `c^{D_Φ}`. `address-space-dimension-scaling` *measured* `K_max = (2R/Δ_req)^{d_eff}` with a `d`-independent resolution floor and farthest-point packing — **the same theorem, arrived at empirically, in a Hamiltonian rather than a softmax-energy setting.** Consequences, both ways:
- **Positioning gift:** the program's capacity law is not idiosyncratic; it is the field's own packing law. Cite Hu–Wu–Liu and frame `4·2^d` as *"a Hamiltonian register realizes the spherical-code capacity scaling, with an explicitly measured resolution floor `Δ_req ≈ 3.1·max(w,σ_query)`"*. That is a **legible, defensible, physics-native claim.**
- **Novelty risk:** *"capacity is exponential in the address dimension"* is **not novel** — it is the headline of the entire modern-Hopfield line since 2017. The novel part is the **measured constant and the mechanism** (`Δ_req`, `d_eff ≈ 0.72–0.83·d` shell concentration, blank-controlled), not the exponential.

---

## Item 3 — the benchmark shortlist, ranked by (winnability × reviewer legibility)

### ⭐ #1 — Associative-memory capacity, the modern-Hopfield protocol. **RECOMMENDED TARGET.**
- **Benchmarks/protocol** [VERIFIED across Hu et al. NeurIPS 2023 (arXiv:2309.12673), Wu, Hu, Hsiao & Liu, *U-Hop*, **ICML 2024** (arXiv:2404.03827), Santos et al. *Hopfield–Fenchel–Young* (arXiv:2411.08590)]: **MNIST / CIFAR-10 / Tiny-ImageNet**; queries = target image with **50% of pixels randomly masked**; success = **cosine similarity > 0.9** to ground truth (also sum-of-squared-pixel error); sweep the **number of stored memories** and report the degradation curve; separate **Gaussian-noise robustness** sweep at multiple noise levels.
- **SOTA / competitors:** sparse & entmax/normmax Hopfield variants, U-Hop (ICML 2024, reports ~30% average retrieval-error margin over dense softmax Hopfield), Hopfield–Fenchel–Young. Retrieval-error bound in this line: `‖T(x) − ξ_μ‖ ≤ 2m(M−1)e^{−β(Δ_μ − 2mR)}` — **exponential in the separation `Δ_μ`**, which is *literally the same quantity CLU measures as `Δ_req`.**
- **Weight class:** tiny. No pretraining. Single-GPU/CPU. **This is the only shortlist entry CLU can enter at its current scale.**
- **Why it is winnable:** ⭐ **nothing is learned on either side.** The Hopfield line writes patterns in closed form; CLU designs a landscape. The w20 blocker ("learning destroys everything design provides") **does not bind here.** CLU brings three things the line does not have: (i) a *fiber* payload channel separate from the address (the theorist's answer to Ramsauer), (ii) **per-item retention control**, (iii) **retry / cross-basin capture** at 0.665 with a 1.38·h friction tax.
- **Honest difficulty:** medium. You must beat or match on the *same* curves, with the **strongest-read blank control** (the w20 method finding), and you must not claim the exponential itself as novel.

### #2 — Parametric knowledge capacity in bits/param (Allen-Zhu bioS protocol)
- **Benchmark:** bioS(N)/bioR synthetic biographies, `R(F)` capacity ratio. **SOTA = 2 bits/param at 1000 exposures.**
- **Weight class:** ⭐ **1M–0.5B GPT2** — their *smallest* trained model is 1M params. **Affordable.**
- **Legibility:** very high; "2 bits per parameter" is a number reviewers know.
- **Winnability:** **low-to-moderate and it requires learning.** The experiment is: drop a CLU memory layer into a small transformer, train, report `R(F)`. Given w20 (`learning destroys write locality`, 0/3 families), the realistic outcome is `R(F) < 2`. ⚠ **But a credible `R(F)` measurement — even a losing one — is worth more than another synthetic recall table**, because it puts CLU on the field's own capacity axis for the first time. Treat as the **pre-registered high-risk arm.**

### #3 — Sequential / lifelong writing without forgetting
Two sub-targets at very different weight classes:
- **(3a) Knowledge editing — the natural home for the admission gate.** Benchmarks: **CounterFact / zsRE** (ROME, Meng et al. NeurIPS 2022, arXiv:2202.05262; MEMIT, ICLR 2023, arXiv:2210.07229), **WikiBigEdit** (lifelong, arXiv:2503.05683). Metrics are *exactly* CLU's gated-drift axis: efficacy, generalization, **locality/specificity**, and degradation under sequential edits. Known numbers [SECONDARY, from search summaries — verify before quoting]: MEMIT's catastrophic phase begins ≈**1,400 edits**; **AlphaEdit** (null-space-constrained, ICLR 2025, arXiv:2410.02355) stable to ≈**3,000** and degrading by ≈**5,000**; BetaEdit claims 10,000. ⭐ **AlphaEdit's null-space projection is conceptually the same object as MVC-0's spacing/admission gate — "write only where you will not disturb what is stored."** This is both the strongest validation of the gate idea and its most direct preemption. ⛔ **Weight class is GPT-J-6B / Llama-3-8B — out of reach without a host LLM. Ruled out as a near-term target; mandatory as a related-work citation.**
- **(3b) Classic continual learning.** Canonical framing: **van de Ven, Tuytelaars & Tolias (2022), "Three types of incremental learning", *Nature Machine Intelligence* 4:1185–1197, doi:10.1038/s42256-022-00568-3** — task-IL / domain-IL / **class-IL** on Split-MNIST and Split-CIFAR-100. **Use this taxonomy or a CL reviewer will reject the setup on sight.** Current SOTA is prompt-based on **pretrained ViT** (≈86–88% avg acc Split-CIFAR-100; ≈69–72% Split-ImageNet-R) [SECONDARY, search summaries]. ⚠ **Weight-class mismatch is fatal:** the modern numbers assume a pretrained backbone; a from-scratch CLU entry is not comparable to any of them, and Farquhar & Gal (arXiv:1805.09733) already criticized toy CL protocols. **Credible entry = class-IL Split-MNIST/Split-CIFAR-10 from scratch, reported against the three-scenario taxonomy, with EWC/SI/replay baselines — a legible but low-impact result.**

### ⛔ #4 — MQAR / Zoology / Based. **RULED OUT as a headline. Keep as a reported diagnostic only.**
- Arora et al. (2023), *"Zoology"*, arXiv:2312.04927: **Theorem 4.4** — data-independent gated convolutions need model dimension `Õ(N log c)`, i.e. **at least linear in sequence length**; **Proposition 4.3** — attention solves MQAR with `Õ(c²)` params and **dimension independent of `N`**.
- Arora et al. (2024), *"Based"*, **ICML 2024**, arXiv:2402.18668, **Theorem 3.1** verbatim: *"Any recurrent model depending causally on input requires **Ω(N)-bits in state size** to solve MQAR."*
- ⇒ **CLU is a fixed-state causal recurrence. The theorem says the class loses. The 0.27-vs-0.996 result is the predicted outcome, not a tuning failure and not an artifact of unfairness.** Report it once, cite the theorem, move on.
- ✅ **What survives:** the **CLU-vs-GRU crossover at kv≈8 (0.154 vs 0.008 at kv=16)** is a legitimate *constant-factor state-utilization* claim at matched state size — "how efficiently a fixed state is used", never "we solved recall". Frame it against the recall–state-size tradeoff, not against attention. ⚠ And the stage-C rescue is still owed.

### ⛔ #5 — Long-horizon: LRA / needle-in-a-haystack / RULER / BABILong. **RULED OUT, on three independent grounds.**
- **Contextual, not parametric.** NIAH/RULER (Hsieh et al., arXiv:2404.06654, NVIDIA; GPT-4 + 9 open models, 4k–128k) and BABILong test retrieval from the *current context*. Wrong category for a parametric memory — **the same mistake as MQAR, one level larger.**
- **LRA is contested.** Path-X: S4 96.4%, S5 98.5%, MEGA avg 88.2 [SECONDARY]. But *"Never Train from Scratch"* (ICLR 2024) and *"On the locality bias and results in the Long Range Arena"* (arXiv:2501.14850) show the gaps close with pretraining/positional encodings and that gains trace to **locality bias, not long-range carryover.** A win here would be discounted; a loss would be terminal.
- **We have a measured failure on exactly this capability.** `primitive-harness`: adding problem at the **no-mixing control floor** (0.182 vs 0.183). Long-horizon retention is currently a *demonstrated negative*, not an untested claim.
- ⚠ If the program insists on a long-context artifact, **Titans is the template**: 170M–760M models, FineWeb-Edu perplexity, NIAH to 16K, BABILong. That is the price of entry, and it is above the program's current compute.

### ⛔ #6 — Reconstruction at matched latent dimension (`clu-autoencoder`). **RULED OUT as a headline; keep as an internal ablation.**
- The modern protocol is **image-tokenizer reconstruction**: rFID / PSNR / SSIM on **ImageNet-1k 256×256** (and MS-COCO) at matched compression, e.g. 8× downsampling with **16 latent channels**, or a fixed **token count** (32/64/128). Reference points: **MAETok rFID 0.48 @128 tokens**; **SoftVQ-S 46M params, rFID 1.03 @64 tokens** [SECONDARY, from 2025 CVPR/arXiv summaries]. Expected baselines: SD-VAE, VQGAN, TiTok/SoftVQ, and a **matched-latent-dim** ablation.
- **Weight class:** tens-to-hundreds of millions of params trained on ImageNet. **Winnability ≈ 0.** A matched-latent-dim MNIST/CIFAR AE comparison is a legitimate *ablation* inside a paper, not a benchmark result.

### The ranking, stated plainly
`#1 Hopfield-capacity ≫ #2 bits/param (bioS) > #3b classic class-IL ≫ #3a knowledge editing (blocked on weight class) ≫ #4 MQAR (theorem says no) ≈ #5 long-context (wrong category + measured failure) ≈ #6 reconstruction (weight class)`

⚠ **The honest headline the Head asked for and may not want:** *for a **learned** CLU there is currently no benchmark on which it is competitive, and one of the losses is backed by a lower-bound theorem.* **For a **designed** CLU there is exactly one — the associative-memory capacity protocol — and it is a real, published, peer-reviewed line with a small weight class.** That is the target. I did not manufacture a second one.

---

## Item 4 — the retrieval-cost claim, checked

**The Hub's assertion:** CLU retrieval is `O(steps)`, independent of the number of stored items; attention is `O(K)` per query.

**Verdict: three separate problems. Do not build on it as stated.**

1. **False for the shipped implementation.** `BallRegisterPotential`'s `V(q)` sums over `K` Gaussian wells, so each `∇V` is `O(K·d)` and retrieval is `O(steps·K·d)`. The engineer's measured **linear-in-K cost is the correct behaviour of the code, not an artifact.** (`dim-scaling` §5: *"Cost scales linearly in K at fixed query budget (per-step work is O(K·d) per query)"*.)
2. **True for a *parametric* landscape — but that is the definition of parametric memory, not a contribution.** A learned `V_θ` (MLP) has forward cost set by `|θ|`, independent of how many items were written. This is precisely why the field builds memory layers, and it is **already engineered explicitly and at scale**:
   - **Lample et al. (2019), "Large Memory Layers with Product Keys", NeurIPS 2019, arXiv:1907.05242** — product keys give **exact** nearest-neighbour search in `O(√K)` instead of `O(K)`, adding *"up to a billion parameters with a negligible computational overhead."* [VERIFIED — NeurIPS proceedings + abstract]
   - **Berges et al. (2024/2025), "Memory Layers at Scale", arXiv:2412.09764, ICML 2025** — *"a trainable key-value lookup mechanism to add extra parameters to a model **without increasing FLOPs**"*; scaled to **128B memory parameters / 1T tokens**, ≈2× compute-matched dense baselines on factual QA. [VERIFIED — abstract + repo]
   - Mixture-of-experts is the same trick with routing.
   ⇒ **"Retrieval cost independent of the number of stored items" is occupied ground, at 128B-parameter scale, by an ICML paper. Claiming it invites an immediate reject.**
3. **The comparison to attention is category-confused.** Attention's `O(K)` is over **in-context** items in the KV cache; a parametric memory's `O(1)` is over **in-weight** items. They are not the same `K`. The valid comparison is *parametric CLU vs an FFN/memory-layer/MoE*, and there CLU must beat `O(√K)` product keys and zero-added-FLOPs memory layers — not softmax attention.
4. ⚠ **Even modern Hopfield does not have the property:** `ξ_new = X softmax(β Xᵀ ξ)` sums over all `M` stored patterns, so its one-step retrieval is `O(M·d)`. "One update" ≠ "O(1) in stored items". [MY INFERENCE from the well-attested update rule; the rule itself is SECONDARY — I still have not read Ramsauer's primary text.]

**What is defensible:** *retrieval is a fixed-length rollout whose cost is set by the integrator, not by the query* — an **anytime / interruptible** retrieval with a measured accuracy-vs-steps curve, plus **codebook-gated retry** at a measured 1.38·h price. That is an adaptive-compute claim (novelty (c) from the w19 scout), not a complexity claim.

---

## Item 5 — prior-art debts: **2 closed, 1 closed-with-a-nuance**

### 5.1 ⭐ Ramsauer's exact capacity constant — **CLOSED (via a peer-reviewed restatement; primary text still unread)**
From **Hu, Wu & Liu (2024), "Provably Optimal Memory Capacity for Modern Hopfield Models", NeurIPS 2024, arXiv:2410.23126, Lemma 2.1** [VERIFIED — arXiv HTML, fetched]:

> `M_Φ ≥ √p · C^{(D_Φ − 1)/4}`, where `C = b / W₀(exp(a + ln b))`,
> `a = (4/(D_Φ−1))·( ln( (2√p − 2)/R_Φ ) + 1 )`, `b = 4β/(5(D_Φ−1))`,
> `W₀(·)` = principal branch of the Lambert W function, `β` = inverse temperature,
> `R_Φ = ½·min_{μ≠ν} ‖Φ(ξ_μ) − Φ(ξ_ν)‖`, `p` = success probability.
> Separation: `Δ_μ^Φ := 𝒦(ξ_μ,ξ_μ) − max_{ν≠μ} 𝒦(ξ_ν,ξ_μ)`. Storage (Def. 2.2): a pattern is stored when a neighbourhood of `Φ(ξ_μ)` contains a generalized fixed point of the update.

**Status upgrade:** the w19 scout flagged `√p·c^{(d−1)/4}` as **[SINGLE-SOURCED, secondary — must not enter a paper]**. It is now **restated with the full Lambert-W constant in a NeurIPS 2024 paper by an independent group**, and independently referenced as "Theorem A5" in the Ramsauer group's own DeepRC paper (arXiv:2007.13505) [search-level attestation only]. **I consider the form safe to cite as `(Ramsauer et al. 2021, Thm 3; as restated in Hu, Wu & Liu 2024, Lemma 2.1)`.** ⚠ **I still have not read Ramsauer's primary PDF** — ar5iv exceeds the 10 MB fetch cap and OpenReview serves a bot-verification wall. **If the paper prints the bare theorem number without the "as restated in" attribution, someone must open the PDF by hand.**

⭐ **Bonus, and it is the more important half:** the same paper's **Proposition 2.1** gives the *tight* answer — `M* ≍ c^{D_Φ}` for some `c>1`, matching upper and lower bounds, **achieved when memories form an optimal spherical code maximizing minimum angular distance** (Lemma 2.2). This supersedes the loose `c^{(d−1)/4}` lower bound as the thing to cite for "how much can an energy memory hold", and it is the direct theoretical counterpart to `address-space-dimension-scaling`'s measured packing law. See §2.4.

### 5.2 ⭐ SRNN "initial state optimization" — **CLOSED. It is NOT a preemption of learned addressing.**
From **Chen, Zhang, Arjovsky & Bottou (2020), "Symplectic Recurrent Neural Networks", ICLR 2020, arXiv:1909.13334, §4.3** [VERIFIED — ar5iv full text, fetched]:
- **What:** *"two new parameter vectors for each sample, `p̂₀` and `q̂₀`, interpreted as our estimate of the actual initial states"*, optimized jointly with the network.
- **Why:** *"When noise is present in observations… our dynamical models will start from these noisy states and remain biased as we advance in time."* — **it is a denoising / nuisance-parameter estimate, not an index.**
- **How:** *"after every epoch we perform ISO with the **L-BFGS-B** algorithm on the `p̂₀` and `q̂₀` parameters for every training trajectory"*; at test time the first 10 observations are used to run the same L-BFGS-B fit for `p̂₀`.

**Verdict:** ⭐ **The preemption risk is LOWER than the w19 scout feared.** SRNN optimizes an initial state **per sample, to remove observation noise, against a full-trajectory reconstruction loss**. CLU optimizes/derives an initial state **to select which stored item is retrieved**. Same mechanical object, **different function, different objective, different information source.** ✅ The learned/derived-address novelty **survives**, provided the paper (i) cites SRNN §4.3 explicitly and (ii) states the distinction in one sentence. ⚠ **Two live consequences:**
- SRNN's ISO is **a successful gradient-based optimization of `(q₀,p₀)` at test time** — which sits uncomfortably next to w19/w20's Prop 7 ("no useful gradient through a robust retrieval map"). **They do not contradict**: SRNN differentiates a *trajectory-matching* loss (rich signal at every timestep), CLU differentiates *through a retrieval map that is exactly robust on a neighbourhood* (zero first-order signal by construction). **This is a genuinely good paragraph for the theory section — and a reviewer who knows SRNN will ask exactly this question.**
- SRNN's use of **L-BFGS-B** (a quasi-Newton method) on the initial state is a *third* precedent for a derivative-free/second-order address search, alongside the w20 retry route.

### 5.3 UnICORNN's Hamiltonian claim — **CLOSED, and it VERIFIES**
From **Rusch & Mishra (2021), "UnICORNN: A recurrent model for learning very long time dependencies", ICML 2021, arXiv:2103.05487** [VERIFIED — ar5iv full text, fetched]:
- ODE: `y' = z`, `z' = −[σ(w⊙y + Vu + b) + αy]`.
- **Hamiltonian (time-dependent), given explicitly:** `H(y,z,t) = (α/2)‖y‖² + (1/2)‖z‖² + Σᵢ (1/wᵢ)·log(cosh(wᵢ yᵢ + …))`.
- **Integrator: symplectic Euler**, *"which preserves the Hamiltonian structure in the discrete setting"*; the network is **invertible in time**.
- **Gradient bound (Prop. 3.1):** `|∂ℰ/∂θ| ≤ [(1−Δt^L)/(1−Δt)]·T(1+2γT)·V̄·(Ȳ+F)·Δ`, growing **at most as `(NΔt)³`** in sequence length.

⚠ **This is the sharpest structural competitor to CHLU's core and it must be handled explicitly.** A published ICML paper already has: a Hamiltonian second-order oscillator RNN, a **symplectic integrator**, time-invertibility, **and a polynomial gradient bound theorem** — CHLU has the first three and **no theorem of comparable strength**. CHLU's differentiators reduce to: the **relativistic kinetic governor**, **learned mass spectrum** (`τ ∝ M^0.79`), **energy-based wake–sleep training**, and **addressability**. The bare "symplectic ⇒ long-horizon stability" claim is **occupied**, with a bound attached.

---

## Confidence & gaps

**VERIFIED from primary text (fetched and quoted this session):** Allen-Zhu & Li abstract + `R(F)` formula + Results 1/2/3/5 + model-size range 1M–0.5B + N=10K–20M · Morris et al. 3.51/3.64/3.83 bits/param + protocol · Chan et al. abstract ("in-weights") · Lewis et al. abstract ("parametric/non-parametric memory") · Titans memory update `S_t = η_t S_{t−1} − θ_t∇ℓ`, `M_t = (1−α_t)M_{t−1} + S_t`, the `‖M(k_t) − v_t‖²` associative loss, "momentum and weight decay", short-/long-term memory framing, model sizes 170M–760M · Zoology Thm 4.4 / Prop 4.3 · Based **Thm 3.1 (Ω(N) state)** + 360M/1.3B scales · Hu–Wu–Liu Lemma 2.1 (Ramsauer constant with Lambert W), Def 2.2, Lemma 2.2, Prop 2.1 · SRNN §4.3 ISO · UnICORNN Hamiltonian + symplectic Euler + Prop 3.1 · `BallRegisterPotential` parameter shapes (read from the repo).

**SECONDARY / single-sourced — do not print without a second read:** Gardner's `α_c = 2` and the "2 bits/synapse" gloss (⚠ one summary said "≈1 bit/synapse"; **resolve before use**) · Demircigil `C ≅ 2^{d/2}` (two restatements, no primary) · U-Hop's ~30% margin and the exact half-mask/cosine-0.9 protocol constants · all continual-learning SOTA numbers (86–88% Split-CIFAR-100, 69–72% Split-ImageNet-R) · all model-editing edit-count thresholds (1,400 / 3,000 / 5,000 / 10,000) · LRA Path-X numbers (S4 96.4, S5 98.5, MEGA 88.2) · all image-tokenizer rFID numbers · Geva et al. full text (venue/DOI verified, abstract not fetched) · Ramsauer's primary theorem text (**still unread — fetch wall**).

**Could not verify / open:** Ramsauer's own PDF (ar5iv >10 MB, OpenReview bot wall) · whether the bioS datasets are publicly released (**check before scoping the §3 #2 experiment**) · whether anyone has run a *Hamiltonian* system on the Hopfield half-mask retrieval protocol (**a negative from search would materially strengthen #1 — worth one dedicated citation-graph sweep**).

**Search next, in priority order:**
1. ⭐ **Citation-graph walk forward from Hu–Wu–Liu (2024) and U-Hop (ICML 2024)** — confirm no Hamiltonian/symplectic entrant on the associative-memory capacity protocol. This is the load-bearing negative for recommendation #1.
2. **Titans' full experimental table + its follow-ups** (Google's 2025–26 "neural memory" line) — the closest competitor to CLU's write mechanism; we need to know what it *cannot* do.
3. **Gardner 1988 primary** — one number, high rhetorical value, currently unsafe.
4. **bioS dataset availability + the smallest reported `R(F)` cell**, to size the #2 arm.
5. **AlphaEdit's null-space projection vs MVC-0's spacing gate** — read the actual update rule; this may be a near-identical mechanism and it is an ICLR 2025 paper.

---

## Bibtex-ready refs

```bibtex
@article{allenzhu2024capacity,
  title={Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws},
  author={Allen-Zhu, Zeyuan and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2404.05405}, year={2024}}

@article{morris2025memorize,
  title={How much do language models memorize?},
  author={Morris, John X. and others},
  journal={arXiv preprint arXiv:2505.24832}, year={2025},
  note={Meta FAIR / Google DeepMind / NVIDIA / Cornell; alpha = 3.64 bits/param}}

@inproceedings{lewis2020rag,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2020},
  note={arXiv:2005.11401}}

@inproceedings{chan2022data,
  title={Data Distributional Properties Drive Emergent In-Context Learning in Transformers},
  author={Chan, Stephanie C. Y. and Santoro, Adam and Lampinen, Andrew K. and Wang, Jane X. and Singh, Aaditya and Richemond, Pierre H. and McClelland, Jay and Hill, Felix},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2022},
  note={arXiv:2205.05055}}

@inproceedings{geva2021ffn,
  title={Transformer Feed-Forward Layers Are Key-Value Memories},
  author={Geva, Mor and Schuster, Roei and Berant, Jonathan and Levy, Omer},
  booktitle={Proceedings of EMNLP}, pages={5484--5495}, year={2021},
  doi={10.18653/v1/2021.emnlp-main.446}}

@inproceedings{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2025},
  note={arXiv:2501.00663}}

@inproceedings{arora2024based,
  title={Simple linear attention language models balance the recall-throughput tradeoff},
  author={Arora, Simran and Eyuboglu, Sabri and Zhang, Michael and Timalsina, Aman and Alberti, Silas and Zinsley, Dylan and Zou, James and Rudra, Atri and R{\'e}, Christopher},
  booktitle={International Conference on Machine Learning (ICML)}, year={2024},
  note={arXiv:2402.18668; Thm 3.1: Omega(N)-bit state lower bound for MQAR}}

@article{arora2023zoology,
  title={Zoology: Measuring and Improving Recall in Efficient Language Models},
  author={Arora, Simran and Eyuboglu, Sabri and Timalsina, Aman and Johnson, Isys and Poli, Michael and Zou, James and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2312.04927}, year={2023}}

@inproceedings{hu2024optimal,
  title={Provably Optimal Memory Capacity for Modern Hopfield Models: Transformer-Compatible Dense Associative Memories as Spherical Codes},
  author={Hu, Jerry Yao-Chieh and Wu, Dennis and Liu, Han},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2024},
  note={arXiv:2410.23126}}

@inproceedings{wu2024uhop,
  title={Uniform Memory Retrieval with Larger Capacity for Modern Hopfield Models},
  author={Wu, Dennis and Hu, Jerry Yao-Chieh and Hsiao, Teng-Yun and Liu, Han},
  booktitle={International Conference on Machine Learning (ICML)}, year={2024},
  note={arXiv:2404.03827}}

@article{demircigil2017huge,
  title={On a Model of Associative Memory with Huge Storage Capacity},
  author={Demircigil, Mete and Heusel, Judith and L{\"o}we, Matthias and Upgang, Sven and Vermet, Franck},
  journal={Journal of Statistical Physics}, volume={168}, pages={288--299}, year={2017},
  doi={10.1007/s10955-017-1806-y}}

@article{gardner1988space,
  title={The space of interactions in neural network models},
  author={Gardner, Elizabeth},
  journal={Journal of Physics A: Mathematical and General}, volume={21}, number={1}, pages={257--270}, year={1988},
  note={UNVERIFIED PRIMARY: alpha_c = 2 constant taken from secondary sources this session}}

@inproceedings{lample2019memory,
  title={Large Memory Layers with Product Keys},
  author={Lample, Guillaume and Sablayrolles, Alexandre and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2019},
  note={arXiv:1907.05242}}

@inproceedings{berges2025memorylayers,
  title={Memory Layers at Scale},
  author={Berges, Vincent-Pierre and Oguz, Barlas and Haziza, Daniel and Yih, Wen-tau and Zettlemoyer, Luke and Ghosh, Gargi},
  booktitle={International Conference on Machine Learning (ICML)}, year={2025},
  note={arXiv:2412.09764}}

@article{vandeven2022three,
  title={Three types of incremental learning},
  author={van de Ven, Gido M. and Tuytelaars, Tinne and Tolias, Andreas S.},
  journal={Nature Machine Intelligence}, volume={4}, number={12}, pages={1185--1197}, year={2022},
  doi={10.1038/s42256-022-00568-3}}

@inproceedings{meng2022rome,
  title={Locating and Editing Factual Associations in GPT},
  author={Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2022},
  note={arXiv:2202.05262}}

@inproceedings{meng2023memit,
  title={Mass-Editing Memory in a Transformer},
  author={Meng, Kevin and Sharma, Arnab Sen and Andonian, Alex and Belinkov, Yonatan and Bau, David},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2023},
  note={arXiv:2210.07229}}

@article{fang2025alphaedit,
  title={AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models},
  author={Fang, Junfeng and others},
  journal={arXiv preprint arXiv:2410.02355}, year={2024},
  note={ICLR 2025; venue and author list SECONDARY, verify}}

@article{hsieh2024ruler,
  title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Hsieh, Cheng-Ping and Sun, Simeng and Kriman, Samuel and Acharya, Shantanu and Rekesh, Dima and Jia, Fei and Zhang, Yang and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2404.06654}, year={2024}}

@inproceedings{chen2020symplectic,
  title={Symplectic Recurrent Neural Networks},
  author={Chen, Zhengdao and Zhang, Jianyu and Arjovsky, Martin and Bottou, L{\'e}on},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2020},
  note={arXiv:1909.13334; Sec 4.3 = initial state optimization via L-BFGS-B}}

@inproceedings{rusch2021unicornn,
  title={UnICORNN: A recurrent model for learning very long time dependencies},
  author={Rusch, T. Konstantin and Mishra, Siddhartha},
  booktitle={International Conference on Machine Learning (ICML)}, year={2021},
  note={arXiv:2103.05487; symplectic Euler on a time-dependent Hamiltonian, Prop 3.1 gradient bound}}

@article{gladstone2025ebt,
  title={Energy-Based Transformers are Scalable Learners and Thinkers},
  author={Gladstone, Alexi and others},
  journal={arXiv preprint arXiv:2507.02092}, year={2025},
  note={author list beyond first author UNVERIFIED}}
```

---

## Proposed handover updates (for the Hub)

1. **§6 / claims-matrix — a hard constraint on the capacity headline.** `K_max = 4·2^d` is **exponential in address dimension, not in parameters**; the designed landscape spends `K(d+1)` parameters on `K` items (`payloads` (K,), `centers` (K,d)). Derived bits/param **≈1.1–1.6, below the transformer's measured 2**. **Every quotation of the exponential must carry the denominator.**
2. **§6 — the MQAR loss gets a theorem, not an excuse.** Add: *Arora et al., ICML 2024, Thm 3.1 — any causal recurrent model needs Ω(N) bits of state to solve MQAR.* CLU's 0.27 vs attention's 0.996 is the **predicted** outcome for its model class. Keep the CLU/GRU crossover as a **constant-factor state-utilization** claim only.
3. **⭐ Recommended benchmark target, and it is the only one that clears the w20 blocker: the modern-Hopfield / dense-associative-memory capacity protocol** — MNIST/CIFAR-10, 50%-masked queries, cosine>0.9 success, capacity-vs-#stored curves, Gaussian-noise robustness. **Nothing is learned on either side of that comparison**, which is why the designed landscape is admissible there and nowhere else. SOTA line: Ramsauer 2021 → sparse/entmax Hopfield (NeurIPS 2023) → U-Hop (ICML 2024) → Hopfield–Fenchel–Young. Weight class: single-GPU.
4. **⭐ Second target, high-risk, pre-registerable: Allen-Zhu's bioS `R(F)` capacity ratio.** Their smallest trained model is **1M params** — affordable. Pre-register a predicted `R(F)` before running. **Recommend the Hub pre-register both the optimistic (`R ≥ 2`) and the realistic (`R < 1`) hypotheses**, per the v5-gate precedent.
5. **Prior-art table — four additions, one urgent.** ⚠ **Titans (NeurIPS 2025, Google): a neural long-term memory written at test time by gradient descent with momentum + weight decay on `‖M(k_t) − v_t‖²`** — i.e. damped second-order dynamics on an associative-memory loss. **This is the nearest published neighbour to CLU's write mechanism and it must be cited and differentiated.** Also: Hu–Wu–Liu NeurIPS 2024 (optimal capacity = spherical codes — the theoretical twin of our measured packing law); Lample et al. NeurIPS 2019 + Berges et al. ICML 2025 (retrieval cost independent of stored items is **occupied ground at 128B params**); AlphaEdit ICLR 2025 (null-space-constrained sequential editing ≈ the admission gate, in a different medium).
6. **Prior-art debts — status.** ✅ **Ramsauer's constant CLOSED**: `M ≥ √p·C^{(D−1)/4}` with `C = b/W₀(exp(a + ln b))`, `a = (4/(D−1))(ln((2√p−2)/R_Φ)+1)`, `b = 4β/(5(D−1))`, via NeurIPS 2024 Lemma 2.1 — **cite as "as restated in Hu, Wu & Liu (2024)"; the primary PDF is still unread.** ✅ **UnICORNN CLOSED and VERIFIED** — explicit Hamiltonian, symplectic Euler, `(NΔt)³` gradient bound. ⚠ **This means "symplectic ⇒ long-horizon stability" is occupied by an ICML 2021 paper *with a theorem we do not have*.** ✅ **SRNN ISO CLOSED — and it is NOT a preemption**: per-sample denoising of initial conditions via L-BFGS-B, not addressing. **The learned-address novelty survives with a one-sentence distinction.**
7. **New theory paragraph the program should write:** SRNN successfully optimizes `(q₀,p₀)` by L-BFGS-B, while Prop 7 says a robust retrieval map yields no first-order signal. **They are consistent** — SRNN differentiates a dense trajectory-matching loss; CLU differentiates through an exactly-robust read. A reviewer who knows SRNN *will* ask. Assign to the theorist.
8. **Do NOT quote (additions to the §832 list):** `4·2^d` as parameter-efficiency or as novel (exponential capacity has been the modern-Hopfield headline since 2017) · "retrieval cost independent of stored items" as a CLU contribution · LRA/NIAH/RULER as our long-horizon target (wrong category **and** we sit at the no-mixing floor on the adding problem) · "symplectic gives long-horizon stability" without citing UnICORNN's Prop 3.1.
