# unlearning-recon — web-scout report
Task + acceptance criterion: test the Head's "R1 is mostly framing" claim — pin the unlearning benchmark map, pin what **certified** formally requires, audit whether "deletion by construction" is already owned, deliver ONE recommendation or a plain ⛔, with predicted loss modes, mandatory baselines, and bibtex-ready refs.
Status: **done** (read-only; no git footprint).

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). FOUR items, all for the Hub, all wording-level and therefore cheap-but-mandatory.**
> 1. ⛔ **DELETE THE WORD "CERTIFIED" FROM R1.** In this literature *certified* has a single, formal, DP-derived meaning: **(ε,δ)-statistical indistinguishability from a model retrained without the datum** (Guo et al. Def. 1–2, verbatim below). It requires a *randomized* mechanism, an explicit noise injection, and a proof. `exp(−leak·t)` is deterministic, noiseless, and proves nothing. **"Certified per-item lifetimes" is a one-line referee kill exactly as the task feared.** Proposed honest wording: **"scheduled per-item retention"** / **"set-at-write-time memory lifetimes with an exact decay law."**
> 2. ⛔ **"DELETION BY CONSTRUCTION" IS OCCUPIED, ~~three~~ four times over** — SILO (ICLR'24 spotlight) sells exactly this sentence for a nonparametric datastore; SISA (IEEE S&P'21) owns exact unlearning by construction in weights; Ticketed Learning–Unlearning (COLT'23) formalizes per-example structural deletion; **PALL (ICLR 2025)** already does *exact task unlearning inside a continual-learning episodic-memory architecture* — i.e. it occupies the CL∩unlearning cell w25 was going to enter. **A kNN/vector-store row-delete is the mandatory baseline and it is trivially exact.**
> 3. ⛔ **NEW, INTERNAL, AND POSSIBLY FATAL TO "EXACT": the MVC-0 controller makes the store HISTORY-DEPENDENT.** `refuse-and-relocate` places item *j* as a function of item *i* being present; LRU/staleness eviction removes item *k* because *i* occupied budget (`controller-mvp` §1, N91). Therefore **removing item *i* from the store does NOT reproduce the store that would have existed had *i* never been written** — which is the definition of exact unlearning. R1 cannot claim exactness without either (a) re-running placement on the retained set (= retraining the store, i.e. the thing unlearning is trying to avoid) or (b) a placement rule proven order-independent. **This is a real technical gap, not a framing gap, and nobody owns it yet.** ← the single most decision-relevant sentence in this report.
> 4. ⚠ **TERMINOLOGY COLLISION inside our own family:** "unlearning" in the Hopfield/energy-based lineage means **Crick–Mitchison anti-Hebbian removal of spurious states during a sleep phase** (Hopfield, Feinstein & Palmer, *Nature* 1983) — and CLU *has a wake–sleep phase*. An associative-memory referee will read "CLU unlearning" as dreaming, not as GDPR. Use **"deletion" / "erasure" / "scheduled forgetting."**

---

## Answer first
**R1 is NOT "mostly framing" — it is one wrong word plus one unfixed technical gap.** The word *certified* is load-bearing and formally defined ((ε,δ)-indistinguishability from retraining; Guo et al. 2020), CLU cannot supply it, and claiming it invites a desk-level correction. Worse, our decay-and-evict machinery is **not even *exact*** in the field's sense, because the controller's refuse-and-relocate/eviction history makes the store's final configuration depend on items that were later deleted — a gap no citation fixes and that must be closed by design or by scope. **Recommended single entry: reframe R1 away from machine unlearning entirely and file it as a memory-control result — "scheduled forgetting: per-item retention set at write time, following an exact, measured amplitude law, with permanence and decay coexisting in one store"** — with unlearning cited as *motivation* and an honest "we do not claim a certified guarantee" sentence. If the program insists on entering the unlearning arena, the only defensible claim is **"structural deletion from a non-parametric store, conditional on an order-independent placement rule and a φ that never saw the item,"** which referees will (correctly) call a kNN datastore with extra physics.

---

## Item 2 ⭐ — what "certified" formally requires (do this first; it decides the framing)

### 2.1 The definition, verbatim [VERIFIED — ar5iv full text of arXiv:1911.03030]
**Guo, Goldstein, Hannun & van der Maaten (2020), "Certified Data Removal from Machine Learning Models", ICML 2020, arXiv:1911.03030.**

> **Def. 1 (ε-certified removal).** A removal mechanism `M` performs ε-certified removal for learning algorithm `A` if for all measurable `T ⊆ H`, datasets `D ⊆ X`, and `x ∈ D`:
> `e^(−ε) ≤ P(M(A(D),D,x) ∈ T) / P(A(D\x) ∈ T) ≤ e^ε`
>
> **Def. 2 ((ε,δ)-certified removal).**
> `P(M(A(D),D,x) ∈ T) ≤ e^ε · P(A(D\x) ∈ T) + δ` and the reverse inequality.

**What this demands, itemized (each of which CLU currently fails):**
1. A **probability distribution over outputs** — the guarantee is a max-divergence bound, so the mechanism must be *randomized*. CLU's write/decay/evict path is deterministic.
2. An **explicit noise/perturbation budget** — Guo's mechanism = Newton-step removal **+ a random linear loss perturbation** at training time. No noise, no certificate.
3. **Proof-supporting structure** — proven only for **L2-regularized linear classifiers with differentiable convex losses**, requiring strong convexity (λ>0), γ-Lipschitz ℓ″, bounded gradients, ‖xᵢ‖₂ ≤ 1. [VERIFIED]
4. A **reference object**: the model retrained on `D\x`. The certificate is *always* relative to retraining.

⛔ **"We measured a clean exponential decay" satisfies none of 1–4.** A measured schedule is an *empirical property of one system*; a certificate is a *worst-case bound over all datasets, all deleted points, and all measurable output sets*.

### 2.2 Exact vs approximate — the taxonomy a referee will apply [VERIFIED across ≥2 sources]
| class | definition | canonical work | cost |
|---|---|---|---|
| **Retrain from scratch** | the gold standard / reference object | — | O(full training) |
| **Exact unlearning** | output distribution **identical** to retraining on the retain set | **SISA** (Bourtoule et al., IEEE S&P 2021, arXiv:1912.03817) — Sharded, Isolated, Sliced, Aggregated + cached checkpoints; **Ginart et al.** NeurIPS 2019 (k-means, arXiv:1907.05012); **Ticketed learning–unlearning** (Ghazi et al., COLT 2023, arXiv:2306.15744) | retrain 1 shard |
| **Certified / approximate unlearning** | (ε,δ)-indistinguishable from retraining | **Guo et al.** ICML 2020; **Neel, Roth & Sharifi-Malvajerdi**, "Descent-to-Delete", ALT 2021, arXiv:2007.02923; **Sekhari et al.**, NeurIPS 2021, arXiv:2103.03279 (**deletion capacity** — for convex losses, O(n/d^{1/4}) samples deletable, vs O(n/d^{1/2}) for DP learning); **Chien et al., "Langevin Unlearning"**, NeurIPS 2024 (Spotlight), arXiv:2401.10371; **Zhang et al., "Towards Certified Unlearning for DNNs"**, ICML 2024, arXiv:2408.00920 | Newton/Hessian step + noise |
| **Heuristic / empirical unlearning** | no guarantee; measured by attacks | gradient ascent, SCRUB, SalUn, NegGrad+, competition entries | cheap |

⭐ **The most CHLU-adjacent certified line is Langevin unlearning** (Chien, Wang, Chen & Li, NeurIPS 2024): *"an unlearning framework based on noisy gradient descent with privacy guarantees … approximate certified unlearning for non-convex problems, complexity saving compared to retraining, sequential and batch unlearning"* [abstract VERIFIED]. **This is both an opportunity and a warning: the certificate comes from the NOISE (Langevin/DP), not from the geometry.** If CLU ever wants a real certificate, this — not the decay law — is the route: certify the *Langevin generation/relaxation* channel, not the well amplitude. Companion: Chien et al., "Certified Machine Unlearning via Noisy Stochastic Gradient Descent", NeurIPS 2024.

### 2.3 ⛔ The paper that makes even "exact" hard to claim [VERIFIED]
**Thudi, Jia, Shumailov & Papernot (2022), "On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning", USENIX Security 22, arXiv:2110.11891.** Two results, both fatal to loose usage:
- the approximate-unlearning definition (be close to a retrained model) is **ill-posed**, because *the same model is obtainable from different datasets*;
- for exact approaches, **"even for a given training trajectory one cannot formally prove the absence of certain data points used during training."**
- Conclusion: **unlearning is well-defined only at the *algorithmic* level** — the only auditable claim is "we ran algorithm X, audit it."
⇒ **This is actually mildly favourable to CLU**: an *algorithmic* claim ("the store is a function of the live item set only; here is the algorithm") is the one form of claim the field accepts — provided item 3 of the reconciliation list (history dependence) is fixed, because that is precisely the algorithm-level property being asserted.

### 2.4 Verdict on our wording
| candidate wording | verdict |
|---|---|
| "certified per-item lifetimes" | ⛔ **forbidden.** Formal term, we fail all four requirements. |
| "exact deletion / exact unlearning" | ⛔ **not currently true** (controller history dependence, reconciliation item 3) and, if made true, **not novel** (SILO/SISA/kNN). |
| "provable / guaranteed forgetting" | ⛔ same trap, softer words. |
| ✅ **"scheduled per-item retention"** / **"set-at-write-time memory lifetimes"** | ✅ accurate, unclaimed, and matches the four-dial thesis. |
| ✅ **"deterministic, auditable amplitude decay: retention follows `exp(−leak·t)` to measurement precision, half-life fixed at write time"** | ✅ this is what we actually have (`controller-mvp` §3(b): measured 0.705 = exp(−0.35), permanent item 1.000 through 8 ticks, `decayed_out = 6/6`). |
| ✅ **"we make no (ε,δ) claim; our guarantee is structural and algorithmic, not statistical"** | ✅ say this explicitly in the paper — it converts a referee kill into a scoping sentence. |

---

## Item 1 — the benchmark map (protocols pinned from primary sources)

### 1.1 Vision / classification — the closest weight class to CLU
**Deep Unlearn** (Cadet et al. 2024, arXiv:2410.01276) — the most complete recent protocol pin [VERIFIED via HTML fetch]:
- **Datasets:** MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, UTKFace. **Architectures:** ResNet-18, TinyViT.
- **Forget set:** *"the forget set by sampling 10% of 𝒟"* — **random 10% subset**, not class-wise.
- **Reference:** model **retrained from scratch on the retain set only**.
- **Methods (18):** Fine-tune, Gradient Ascent, Successive Random Labels; NeurIPS'23 competition entries (MSG, CT, FCS, CFW, PRMQ, KDE, RNI); SalUn, CF-k, EU-k, SCRUB, BT, FF, IU, NG+.
- **Metrics:** privacy = **U-MIA** (population MIA) and **U-LiRA** (per-example likelihood-ratio attack, shadow models trained *and unlearned*); accuracy = **RA / FA / TA** (retain / forget / test) plus ratios vs the retrained reference; efficiency = **runtime speedup vs full retraining**; consistency = **RetDev**, indiscernibility.
- **Headline:** MSG and CT are the most consistent across datasets/architectures/seeds under U-LiRA.

**⚠ The evaluation-integrity paper that every unlearning referee now cites:** Hayes, Shumailov, Triantafillou, Khalifa & Papernot (2024), *"Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy"*, arXiv:2403.01218 — **population-level MIA drastically overstates unlearning; per-example U-LiRA is the standard.** [VERIFIED via 2 independent sources]

**NeurIPS 2023 Machine Unlearning Challenge** (Google DeepMind; unlearning-challenge.github.io; eval code `google-deepmind/unlearning_evaluation`) — face-image age prediction; score = **per-example forget quality (a DP-style ε estimated from attack FPR/FNR) combined with utility and efficiency**. [protocol shape VERIFIED from the challenge site + Deep Unlearn's reuse of its entrants; ⚠ **could-not-fetch** the metric PDF (binary) — do not quote its exact formula without re-pulling.]

### 1.2 LLM / generative — where the field's attention actually is now
- **TOFU** (Maini, Feng, Schwarzschild, Lipton & Kolter, 2024, arXiv:2401.06121; COLM 2024): 200 fictitious authors, forget splits **1% / 5% / 10%**. Two axes: **Model Utility** = harmonic mean of metrics (token-wise probability, ROUGE recall, Truth Ratio) on retain/real-authors/world-facts; **Forget Quality** = **p-value of a two-sample Kolmogorov–Smirnov test comparing the Truth-Ratio distribution of the unlearned model vs the retain-only retrained model** (p > 0.05 ⇒ indistinguishable ⇒ success). ⚠ Many follow-ups plot **−log₁₀(p)** — [SECONDARY, two-source but conflicting sign convention; check the axis before quoting].
- **MUSE** (Shi, Lee, Huang, Malladi, Zhao, Holtzman, Liu, Zettlemoyer, Smith & Zhang, 2024, arXiv:2407.06460; ICLR 2025): **six criteria** [VERIFIED via HTML]: C1 **VerbMem** (ROUGE-L F1 of continuations), C2 **KnowMem** (ROUGE on QA from forget set), C3 **PrivLeak** = `(AUC_unlearn − AUC_retrain)/AUC_retrain` using **Min-K% Prob** MIA, target band **[−5%, +5%]**, C4 **utility preservation** (KnowMem on retain), C5 **scalability** in forget-set size, C6 **sustainability** over *sequential* requests. ⭐ **C5+C6 are the criteria a scheduled-decay store would naturally target.**
- **WMDP** (Li et al., ICML 2024, arXiv:2403.03218): 3,668 MCQs proxying hazardous bio/cyber/chem knowledge; unlearning applied to off-the-shelf base models.
- **Robustness checks now mandatory:** Łucki/Hu et al., *"Jogging the Memory of Unlearned LLMs via Benign Relearning"* (arXiv:2406.13356, ICLR 2025) — small benign finetuning **reactivates** supposedly unlearned knowledge; LoRA is especially vulnerable. ⇒ **any forgetting claim must survive a relearning attack.**
- ⚠ **The GDumb-equivalent pathology check for this field:** Thaker, Maurya, Hu, Wu & Smith (2024), *"Guardrail Baselines for Unlearning in LLMs"*, arXiv:2403.03329 — **prompting and output filtering match finetuning-based unlearning on these benchmarks.** Same shape as GDumb in CL: *a trivial baseline saturates the metric.* Companion position paper: Thaker et al., *"LLM Unlearning Benchmarks are Weak Measures of Progress."*

### 1.3 Agent / memory-store deletion — the newest and most CLU-adjacent sub-line (2026)
- **Memora** — Uddin, Shubham, Blanco, Baral & Wang (2026), *"From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents"*, arXiv:2604.20006. Metric: **FAMA (Forgetting-Aware Memory Accuracy)** — penalizes reliance on obsolete/invalidated memory. Finding: *"failures to forget outdated memory"* = **64%** of sampled recommendation errors. [VERIFIED via HTML fetch]
- **Agentic Unlearning** — Wang et al. (2026), arXiv:2602.17692: deletes from **both** weights and persistent agent memory; names the **"parameter–memory backflow"** problem (retrieval reactivates forgotten content); memory deletion by **dependency-aware pruning with blocklists + reference counting**; metrics = MIA score + forget/retain accuracy; claims 24.8% privacy improvement at >90% accuracy. [VERIFIED via HTML fetch] ⚠ **This is the closest live competitor to "delete from the store, not the weights."**
- Also live: MemLeak (arXiv:2606.29788), "Control-Plane Placement Shapes Forgetting" (arXiv:2606.15903). **The sub-line is ~6 months old and moving — re-scout before any freeze.**

---

## Item 3 ⭐ — the winnability audit (ranked, with predicted loss modes)

### ⛔ #0 — the preemption you must answer first: "a kNN datastore also deletes exactly"
Confirmed, and it is **published, spotlighted, and sold in exactly our words**:
- **SILO** — Min, Gururangan, Wallace, Shi, Hajishirzi, Smith & Zettlemoyer, **ICLR 2024 (spotlight)**, arXiv:2308.04430, abstract verbatim: *"…augmenting it with a more general and easily modifiable **nonparametric datastore** … The datastore allows use of high-risk data without training on it, supports sentence-level data attribution, and **enables data producers to opt out from the model by removing content from the store.**"* [VERIFIED]
- **PALL** — Özdenizci, Rueckert & Legenstein, **ICLR 2025**, "Privacy-Aware Lifelong Learning", arXiv:2505.10941: task-specific **sparse subnetworks** + episodic-memory rehearsal ⇒ **"exact task unlearning"** in a *continual-learning* setting. [VERIFIED via abstract fetch] ⛔ **This occupies the CL∩unlearning cell.**
- **Ticketed Learning–Unlearning** — Ghazi, Kamath, Kumar, Manurangsi, Sekhari & Zhang, COLT 2023, arXiv:2306.15744: each example holds an encrypted **ticket**; deletion = present ticket ⇒ exact retrained-equivalent predictor. [VERIFIED]
- **SISA** — Bourtoule et al., IEEE S&P 2021.
⇒ **"Deletion by construction" is not a novelty claim. It is a table row.**

**And the counter-fact that kills the naive privacy pitch:** Huang, Gupta, Zhong, Li & Chen (2023), *"Privacy Implications of Retrieval-Based Language Models"*, EMNLP 2023, arXiv:2305.14888 — **kNN-LMs leak private information from their datastore *more* than parametric models.** ⇒ *Holding data in a store is not privacy; it is a different attack surface.* [VERIFIED]

### The ranked audit

**#1 — ✅ RECOMMENDED: "scheduled forgetting" as a MEMORY-CONTROL result (NOT an unlearning result).**
- **Claim:** a store in which each item's retention is a *dial set at write time*, obeying a measured law (`amp(t) = exp(−leak·t)`, half-life 1.98 at leak 0.35, self-eviction below floor 0.05, permanent items at leak≡0 untouched through 8 ticks — `controller-mvp` §3(b) / N91), with **permanent and forgettable content coexisting in one store**, and a capacity–lifetime tradeoff curve.
- **Why it can win:** the unlearning field is **entirely request-driven** — every method above deletes *on demand, after the fact.* **Nobody schedules deletion at write time with a closed-form retention law.** That is a genuinely open cell.
- **Weight class:** already built (MVC-0). Referee community: memory/architecture (ICLR), not security.
- **Predicted loss modes:** (i) ⚠ **"this is Expire-Span"** — Sukhbaatar, Ju, Poff, Roller, Szlam, Weston & Fan, ICML 2021, arXiv:2105.06548, *learns per-memory expiration spans* in a Transformer. Differentiator: theirs is a *learned relevance* span for efficiency; ours is a *specified* lifetime for control, plus permanence-as-a-designed-coset. **Must be cited in ¶1 of related work.** (ii) ⚠ **"this is MemoryBank"** — Zhong, Guo, Gao, Ye & Wang, AAAI 2024, arXiv:2305.10250, uses an **Ebbinghaus-curve exponential decay of memory strength**. Differentiator: heuristic bookkeeping score vs a *physical amplitude* that changes what retrieval does. (iii) ⛔ **"this is Benna–Fusi"** — Benna & Fusi, *Nature Neuroscience* 19:1697–1706 (2016), engineers **memory lifetimes** with multi-timescale synaptic variables and derives power-law forgetting; Fusi & Abbott, *Nat. Neuro.* 10:485–493 (2007) already owns the **capacity–lifetime tradeoff law**. **This is the most dangerous preemption for a "laws of memory" pitch and it predates us by a decade** — cite both, and differentiate on *per-item programmability* (their lifetimes are a population property of the synaptic model; ours is set per item at write time). (iv) "so what — a TTL field in a dict does this." **This is the hardest question and it must be answered in the paper**: the answer can only be that decay is *continuous and physical* (graded retrieval degradation, a monotone amplitude→retention curve, interacts with admission/packing), not a boolean.

**#2 — ⚠ CONDITIONAL: "structural deletion from a non-parametric designed store," filed into the unlearning arena.**
- Only viable **after** reconciliation item 3 (order-independent placement) is fixed. Then the claim is *algorithm-level auditable deletion* in Thudi's sense.
- **Mandatory baselines** (see below) will make it look like a vector-store `DELETE`. **Predicted loss modes:** (i) "SILO/PALL/SISA did this"; (ii) "your φ saw the datum" (below); (iii) "you have no ε"; (iv) "kNN row-delete is simpler and exactly as exact."
- **Verdict: do not lead with this.** Use it as a *property* paragraph inside #1.

**#3 — ⛔ NOT WINNABLE: anything with "certified" in the sentence.** See Item 2.

### ⛔ The scoping issue the task asked me to flag: **does φ retain the deleted item?**
**Yes, unless φ is disjoint by construction — and this is a fatal scoping issue if unaddressed.**
- The w23 φ discipline already fits an encoder on a **disjoint pool, never through the store** — that is the *only* configuration in which "deletion from the store = deletion from the system" is even arguable. **Any online-φ or task-1-φ variant that trains on data containing the deleted item destroys the deletion claim** and a referee will find it immediately (this is precisely Agentic Unlearning's **"parameter–memory backflow"**, arXiv:2602.17692).
- Even with a disjoint φ, **Huang et al. (EMNLP 2023)** shows the *store itself* is a leak channel — and **partially decayed** wells are the worst case: an item at amp 0.06 is neither present nor absent. **Predicted referee question: "what does a U-LiRA adversary see at t = one half-life?"** We have no answer. **Recommended pre-registration: measure retention AND an MIA-style distinguishability score as a function of `leak·t`, and report where they cross.** That measurement would be new and is cheap on the existing harness.
- **Also unanswered: what does the payload retain after eviction?** `AtomStorePotential.evict` frees a slot; whether residual curvature/site coordinates remain recoverable is untested. **Until measured, do not write "the item is gone."**

---

## Item 4 — prior art on memory-module / KV-store deletion + CLU's remaining novelty surface

| work | what it occupies | what it leaves |
|---|---|---|
| **SILO** (ICLR'24) | opt-out by removing rows from a nonparametric datastore | no *schedule*; no permanence class; discrete rows |
| **PALL** (ICLR'25) | exact **task** unlearning in a CL architecture (sparse subnetworks + episodic rehearsal) | per-**item** (not per-task) control; no lifetimes |
| **Agentic Unlearning** (2026) | deletion from persistent agent memory + weights; blocklists, ref-counting, dependency pruning | request-driven only; bookkeeping, not physics |
| **Ticketed L–U** (COLT'23) | formal per-example structural deletion for restricted concept classes | not a continuous store; no lifetimes |
| **Expire-Span** (ICML'21) | **learned per-memory expiration** in a Transformer | expiry is learned-for-efficiency, not specified-for-control; no permanence coset |
| **MemoryBank** (AAAI'24) | Ebbinghaus **exponential decay of memory strength** in an LLM memory bank | heuristic score, not a retrieval-physics amplitude; no deletion guarantee |
| **Benna–Fusi** (Nat.Neuro'16) / **Fusi–Abbott** ('07) | engineered memory **lifetimes**; the **capacity–lifetime tradeoff law**; power-law forgetting | lifetimes are a population property of the synapse model, not per-item programmable |
| **SQHN** (Nat. Comms 2024) | energy store, online-continual, replay-free (from `continual-learning-recon`) | **no deletion/eviction claim found** — [could-not-verify: I did not find any SQHN deletion result; treat "SQHN does not address deletion" as **SECONDARY**] |
| **Hopfield/Feinstein/Palmer** (*Nature* 1983) | the word **"unlearning"** itself (anti-Hebbian spurious-state removal) | nothing — it is a naming hazard, not a competitor |

**CLU's remaining novelty surface, enumerated honestly (in descending strength):**
1. ⭐ **Deletion scheduled at write time** with a closed-form retention law — the whole unlearning field is request-driven; genuinely unoccupied.
2. ⭐ **Permanence as a designed coset (`leak ≡ 0`) coexisting with decay in one store**, with a **capacity alarm instead of silent overwrite** when full of permanent items (`controller-mvp` §3(b)) — no analogue found in either literature.
3. **Forgetting as a continuous physical amplitude** (graded degradation of the *retrieval dynamics*), not a bookkeeping flag or a learned span. Differentiates from Expire-Span/MemoryBank.
4. **Joint dials**: lifetime × admission × capacity in one mechanism (the packing bound N_pack interacts with decay — freed wells restore admission headroom). Untested, but it is the one thing that makes "dials" a *system* rather than a TTL field.
5. ⛔ **NOT on the list:** exactness, certification, deletion-by-construction, privacy.

---

## Item 5 — collision check on "memory with dials"
- **"Controllable memory" is emerging as a framing.** *"Tell Me What To Learn: Generalizing Neural Memory to be Controllable in Natural Language"* (arXiv:2602.23201, 2026) frames precisely the gap — *"users have no control over what the model remembers or ignores over time"* — but controls memory by **natural-language instruction**, not by a physical dial with a law. [SECONDARY — search-summary + abstract listing only]. 2026 agent-memory work (control-plane placement, arXiv:2606.15903) is converging on the same vocabulary. ⚠ **The word "control" is being taken; the *laws* are not.**
- **Capacity–lifetime as a *law* has an owner in neuroscience** (Fusi & Abbott 2007). ⇒ **Do not present "there is a capacity–lifetime tradeoff" as a discovery.** Present *our* version of the curve, cite theirs, and claim the per-item programmability.
- **Boundary with model/knowledge editing (distinct and crowded — pin it explicitly):** ROME (Meng et al., NeurIPS 2022), MEMIT (ICLR 2023), AlphaEdit; survey arXiv:2310.19704. **The field's own boundary statement:** *"Model editing essentially redirects knowledge mappings to the given outputs, while LLM unlearning focuses on removing such mappings"* (arXiv:2505.19855, "Editing as Unlearning: Are Knowledge Editing Methods Strong Baselines for LLM Unlearning?"). ⇒ **One sentence in related work suffices: we neither redirect (editing) nor certify removal (unlearning); we schedule retention.** Also note **sequential editing degrades catastrophically after ~1,400 edits** (arXiv:2401.07453) — a useful contrast point for a store whose per-item writes are isolated by an admission gate.

---

## Mandatory-baseline list (the N78/GDumb discipline, transposed)
Any R1 table must contain, or it is "solved by a dict":
1. **Vector-store / kNN datastore row-delete** at matched memory — trivially exact, structurally identical claim (SILO). **Non-negotiable.**
2. **Retrain-from-scratch on the retain set** — the field's reference object for *every* metric.
3. **A TTL/dict-with-timestamps store** with the same half-lives — the "is this just bookkeeping?" ablation. If CLU's only advantage over this is aesthetic, R1 is not a result.
4. **Expire-Span-style learned expiry** (or a cited justification for omitting it) — the nearest per-item-lifetime prior.
5. **PALL** if any CL framing is used (it already claims exact task unlearning in CL).
6. If *any* privacy/forgetting-quality language appears: **U-LiRA (per-example)**, not population MIA (Hayes et al. 2024), **plus a benign-relearning attack** (arXiv:2406.13356).
7. If *any* generative/LLM framing appears: **guardrail baselines** (prompt + output filter) — the GDumb of this field.

---

## Confidence & gaps
**VERIFIED from primary this session:** Guo et al. Def.1/Def.2 verbatim + assumption list (ar5iv 1911.03030); Thudi et al. both impossibility claims + "algorithmic level" conclusion (USENIX'22 listing + arXiv:2110.11891, 2 sources); SILO abstract verbatim incl. the opt-out sentence (arXiv:2308.04430); PALL abstract/claim (arXiv:2505.10941); Deep Unlearn full protocol (HTML, arXiv:2410.01276); MUSE six criteria + PrivLeak formula (HTML, arXiv:2407.06460v2); Langevin Unlearning abstract verbatim (arXiv:2401.10371); Agentic Unlearning + Memora/FAMA (HTML fetches, arXiv:2602.17692 / 2604.20006); Hopfield–Feinstein–Palmer 1983 "unlearning" (*Nature* 304:158–159); Expire-Span (ICML 2021, arXiv:2105.06548); MemoryBank (AAAI 2024, arXiv:2305.10250); Benna–Fusi (Nat.Neuro 19:1697–1706) and Fusi–Abbott (Nat.Neuro 10:485–493); Sekhari deletion-capacity rates; Ticketed L–U (COLT'23); Guardrail baselines (arXiv:2403.03329).
**SECONDARY / single-sourced — verify before printing:** TOFU Forget-Quality sign convention (p-value vs −log₁₀ p) and the exact utility-metric list; NeurIPS'23 challenge score formula (**could-not-fetch** — PDF returned binary); "SQHN does not address deletion" (absence of evidence, not evidence of absence); arXiv:2602.23201 (abstract listing only); SISA arXiv id 1912.03817 (search-consistent, not fetched).
**Could not fetch:** the challenge metric PDF; TOFU full text (arXiv abstract page only).
**Search next:** (1) whether any work proves **order-independence of admission/placement** in a capacity-constrained store — this is reconciliation item 3 and it may already exist in the caching/streaming-algorithms literature; (2) MIA/distinguishability against a **partially decayed** memory (I found nothing — likely genuinely open, and it is the cheapest new measurement we could make); (3) whether Memora/FAMA has a public harness we could run the store against; (4) re-scout the 2026 agent-memory-deletion sub-line ~2 weeks pre-freeze — it is moving fast.

---

## Bibtex-ready refs
```bibtex
@inproceedings{guo2020certified,
  title={Certified Data Removal from Machine Learning Models},
  author={Guo, Chuan and Goldstein, Tom and Hannun, Awni and van der Maaten, Laurens},
  booktitle={ICML}, year={2020}, note={arXiv:1911.03030; Def.1 eps-certified, Def.2 (eps,delta)-certified removal}}

@inproceedings{bourtoule2021sisa,
  title={Machine Unlearning},
  author={Bourtoule, Lucas and Chandrasekaran, Varun and Choquette-Choo, Christopher A. and Jia, Hengrui and Travers, Adelin and Zhang, Baiwu and Lie, David and Papernot, Nicolas},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)}, year={2021},
  note={SISA: Sharded, Isolated, Sliced, Aggregated; exact unlearning}}

@inproceedings{thudi2022auditable,
  title={On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning},
  author={Thudi, Anvith and Jia, Hengrui and Shumailov, Ilia and Papernot, Nicolas},
  booktitle={31st USENIX Security Symposium}, year={2022}, note={arXiv:2110.11891}}

@inproceedings{ginart2019forget,
  title={Making AI Forget You: Data Deletion in Machine Learning},
  author={Ginart, Antonio and Guan, Melody Y. and Valiant, Gregory and Zou, James},
  booktitle={NeurIPS}, year={2019}, note={arXiv:1907.05012}}

@inproceedings{neel2021descent,
  title={Descent-to-Delete: Gradient-Based Methods for Machine Unlearning},
  author={Neel, Seth and Roth, Aaron and Sharifi-Malvajerdi, Saeed},
  booktitle={ALT}, year={2021}, note={arXiv:2007.02923}}

@inproceedings{sekhari2021remember,
  title={Remember What You Want to Forget: Algorithms for Machine Unlearning},
  author={Sekhari, Ayush and Acharya, Jayadev and Kamath, Gautam and Suresh, Ananda Theertha},
  booktitle={NeurIPS}, year={2021}, note={arXiv:2103.03279; deletion capacity}}

@inproceedings{ghazi2023ticketed,
  title={Ticketed Learning-Unlearning Schemes},
  author={Ghazi, Badih and Kamath, Pritish and Kumar, Ravi and Manurangsi, Pasin and Sekhari, Ayush and Zhang, Chiyuan},
  booktitle={COLT}, year={2023}, note={arXiv:2306.15744; PMLR 195:5110--5139}}

@inproceedings{chien2024langevin,
  title={Langevin Unlearning: A New Perspective of Noisy Gradient Descent for Machine Unlearning},
  author={Chien, Eli and Wang, Haoyu and Chen, Ziang and Li, Pan},
  booktitle={NeurIPS (Spotlight)}, year={2024}, note={arXiv:2401.10371}}

@inproceedings{zhang2024certifieddnn,
  title={Towards Certified Unlearning for Deep Neural Networks},
  author={Zhang, Binchi and Dong, Yushun and Wang, Tianhao and Li, Jundong},
  booktitle={ICML}, year={2024}, note={PMLR 235:58800--58818; arXiv:2408.00920}}

@inproceedings{min2024silo,
  title={SILO Language Models: Isolating Legal Risk in a Nonparametric Datastore},
  author={Min, Sewon and Gururangan, Suchin and Wallace, Eric and Shi, Weijia and Hajishirzi, Hannaneh and Smith, Noah A. and Zettlemoyer, Luke},
  booktitle={ICLR (spotlight)}, year={2024}, note={arXiv:2308.04430; opt-out by removing content from the store}}

@inproceedings{ozdenizci2025pall,
  title={Privacy-Aware Lifelong Learning},
  author={{\"O}zdenizci, Ozan and Rueckert, Elmar and Legenstein, Robert},
  booktitle={ICLR}, year={2025}, note={arXiv:2505.10941; exact task unlearning via sparse subnetworks}}

@inproceedings{huang2023knnprivacy,
  title={Privacy Implications of Retrieval-Based Language Models},
  author={Huang, Yangsibo and Gupta, Samyak and Zhong, Zexuan and Li, Kai and Chen, Danqi},
  booktitle={EMNLP}, year={2023}, note={arXiv:2305.14888; kNN-LMs leak MORE than parametric models}}

@article{hayes2024inexact,
  title={Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy},
  author={Hayes, Jamie and Shumailov, Ilia and Triantafillou, Eleni and Khalifa, Amr and Papernot, Nicolas},
  journal={arXiv preprint arXiv:2403.01218}, year={2024}, note={U-LiRA; population MIA overstates unlearning}}

@article{maini2024tofu,
  title={TOFU: A Task of Fictitious Unlearning for LLMs},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary C. and Kolter, J. Zico},
  journal={arXiv preprint arXiv:2401.06121}, year={2024}, note={Forget Quality = KS-test p-value on Truth Ratio vs retain-only model}}

@inproceedings{shi2025muse,
  title={MUSE: Machine Unlearning Six-Way Evaluation for Language Models},
  author={Shi, Weijia and Lee, Jaechan and Huang, Yangsibo and Malladi, Sadhika and Zhao, Jieyu and Holtzman, Ari and Liu, Daogao and Zettlemoyer, Luke and Smith, Noah A. and Zhang, Chiyuan},
  booktitle={ICLR}, year={2025}, note={arXiv:2407.06460; VerbMem/KnowMem/PrivLeak/utility/scalability/sustainability}}

@inproceedings{li2024wmdp,
  title={The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning},
  author={Li, Nathaniel and others},
  booktitle={ICML}, year={2024}, note={arXiv:2403.03218}}

@article{cadet2024deepunlearn,
  title={Deep Unlearn: Benchmarking Machine Unlearning for Image Classification},
  journal={arXiv preprint arXiv:2410.01276}, year={2024},
  note={18 methods, 5 datasets, 2 architectures; U-MIA + U-LiRA + RA/FA/TA; 10\% random forget set}}

@article{thaker2024guardrail,
  title={Guardrail Baselines for Unlearning in LLMs},
  author={Thaker, Pratiksha and Maurya, Yash and Hu, Shengyuan and Wu, Zhiwei Steven and Smith, Virginia},
  journal={arXiv preprint arXiv:2403.03329}, year={2024},
  note={prompting/filtering match finetuning-based unlearning --- the GDumb of unlearning}}

@article{hu2025relearning,
  title={Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning},
  journal={arXiv preprint arXiv:2406.13356}, year={2024}, note={ICLR 2025; relearning attack}}

@inproceedings{sukhbaatar2021expirespan,
  title={Not All Memories are Created Equal: Learning to Forget by Expiring},
  author={Sukhbaatar, Sainbayar and Ju, Da and Poff, Spencer and Roller, Stephen and Szlam, Arthur and Weston, Jason and Fan, Angela},
  booktitle={ICML}, year={2021}, note={arXiv:2105.06548; learned per-memory expiration span}}

@inproceedings{zhong2024memorybank,
  title={MemoryBank: Enhancing Large Language Models with Long-Term Memory},
  author={Zhong, Wanjun and Guo, Lianghong and Gao, Qiqi and Ye, He and Wang, Yanlin},
  booktitle={AAAI}, year={2024}, note={arXiv:2305.10250; Ebbinghaus exponential decay of memory strength}}

@article{benna2016consolidation,
  title={Computational principles of synaptic memory consolidation},
  author={Benna, Marcus K. and Fusi, Stefano},
  journal={Nature Neuroscience}, volume={19}, number={12}, pages={1697--1706}, year={2016},
  note={engineered memory lifetimes; power-law forgetting; near-linear capacity scaling}}

@article{fusi2007bounded,
  title={Limits on the memory storage capacity of bounded synapses},
  author={Fusi, Stefano and Abbott, L. F.},
  journal={Nature Neuroscience}, volume={10}, pages={485--493}, year={2007},
  note={the capacity--lifetime tradeoff law --- predates any CLU "dials" claim}}

@article{hopfield1983unlearning,
  title={'Unlearning' has a stabilizing effect in collective memories},
  author={Hopfield, J. J. and Feinstein, D. I. and Palmer, R. G.},
  journal={Nature}, volume={304}, pages={158--159}, year={1983},
  note={TERMINOLOGY COLLISION: "unlearning" = anti-Hebbian spurious-state removal}}

@article{cooper2024unlearningdoesnt,
  title={Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy, Research, and Practice},
  author={Cooper, A. Feder and Choquette-Choo, Christopher A. and Bogen, Miranda and others},
  journal={arXiv preprint arXiv:2412.06966}, year={2024}}

@article{wang2026agenticunlearning,
  title={Agentic Unlearning: When LLM Agent Meets Machine Unlearning},
  author={Wang, Bin and Wang, Fan and Wang, Pingping and Cong, Jinyu and Yu, Yang and Yin, Yilong and Han, Zhongyi and Wei, Benzheng},
  journal={arXiv preprint arXiv:2602.17692}, year={2026},
  note={deletes from persistent agent memory + weights; "parameter--memory backflow"}}

@article{uddin2026memora,
  title={From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents},
  author={Uddin, Md Nayem and Shubham, Kumar and Blanco, Eduardo and Baral, Chitta and Wang, Gengyu},
  journal={arXiv preprint arXiv:2604.20006}, year={2026}, note={Memora; FAMA metric}}
```

---

## Proposed handover updates (for the Hub)
1. ⛔ **R1 must be RENAMED before it is written anywhere.** "Certified per-item lifetimes" → **"scheduled per-item retention / set-at-write-time memory lifetimes."** *Certified* is (ε,δ)-indistinguishability from retraining (Guo et al., ICML 2020, Def. 1–2); we satisfy none of its four requirements. This is a **quote-block-level rule**, not a preference.
2. ⛔ **NEW BLOCKING TECHNICAL GAP (candidate negative-registry entry / candidate w25 theorist task):** *the MVC-0 controller's refuse-and-relocate and LRU eviction make the store's configuration history-dependent, so removing an item does not reproduce the never-written store* ⇒ **CLU cannot currently claim even *exact* deletion.** Two exits: (a) prove/design an **order-independent placement rule**, (b) scope the claim to "retention scheduling," not deletion. **Recommend (b) now, (a) as a theory task.**
3. **R1 is NOT "mostly framing" — revise the result-set status.** It is: one forbidden word + one unfixed exactness gap + one occupied novelty claim ("deletion by construction": SILO ICLR'24, PALL ICLR'25, SISA, Ticketed L–U). What survives is **schedule-at-write-time + permanence-as-a-coset + decay-as-physical-amplitude**, which is real and unoccupied but is a *memory-control* claim, not a privacy claim.
4. ⛔ **PALL (ICLR 2025) is a direct hit on the w25 CL entry** (exact task unlearning inside a continual-learning architecture with episodic memory). Add it to the `continual-learning-recon` must-cite/differentiate list alongside SQHN.
5. **Mandatory baselines for any R1 table:** kNN/vector-store row-delete at matched memory · retrain-from-scratch · **a TTL-dict with the same half-lives** (the "is this bookkeeping?" ablation) · Expire-Span as the per-item-lifetime prior. Any privacy language additionally requires **U-LiRA (per-example, not population)** + a **benign-relearning attack**.
6. **Cheapest new measurement in the program right now (recommend as a w25 experiment, pre-registered):** *distinguishability of a **partially decayed** item as a function of `leak·t`* — retention curve and an MIA-style separability curve on the same axis, plus "what remains after `evict`". I found **no prior work** on adversarial distinguishability of a partially-decayed memory; it is a genuinely open cell and it directly converts a referee's hardest question into a figure.
7. **Naming hazard to record:** never use "unlearning" for CLU's mechanism — in the energy-based/Hopfield lineage it means Crick–Mitchison anti-Hebbian dream-phase removal (Hopfield, Feinstein & Palmer, *Nature* 1983), and CLU *has* a sleep phase. Use "deletion / erasure / scheduled forgetting."
8. **Framing boundary for Chapter 8:** cite **Fusi & Abbott (2007)** for the capacity–lifetime tradeoff and **Benna & Fusi (2016)** for engineered lifetimes — *do not present a capacity–lifetime tradeoff as a discovery.* Claim per-item programmability. And add one sentence separating us from **knowledge editing** (redirects mappings) and **unlearning** (removes them): *we schedule retention.*
