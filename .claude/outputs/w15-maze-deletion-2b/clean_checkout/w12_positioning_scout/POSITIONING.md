# POSITIONING — how a new primitive gets positioned, and what evidence it needs

**w12 `spoke/positioning-scout` · agent `web-scout` · read-only · 2026-09-21.**
**Scored runs: ⛔ NONE (none commissioned).** No code run, no `chlu` import, nothing measured here.
**Evidence base for our side:** `/c/chlu/evidence/outputs/` + `claims_matrix.md`.
⛔ `$CHLU_CLAUDE_ROOT` **not read.**

⭐ **Web access: CONFIRMED.** First fetch of the session: `https://arxiv.org/abs/2008.02217` →
title *"Hopfield Networks is All You Need"*, abstract opens *"We introduce a modern Hopfield network
with continuous states and a corresponding update rule."* `NO-WEB-ACCESS.txt` was **not** written.

⛔⛔ **ONE DECLARED NOT-RUN UP FRONT, AND IT SHAPES §3.** **openreview.net is unreachable from this
box.** Every route — `openreview.net/forum?id=…`, `openreview.net/pdf?id=…`,
`api.openreview.net/notes?forum=…`, `api2.openreview.net/notes?forum=…`, with and without a browser
User-Agent — returns `{"name":"ChallengeRequiredError","status":403}`. **No reviewer text is quoted
in this brief.** §3 is therefore built on **ICLR's own published Reviewer Guide**, which is primary
and authoritative and *states the bars directly*, rather than on individual reviews. Where I use a
fact that only OpenReview could confirm (the Mamba ICLR-2024 outcome), it is **flagged
SECONDARY-ONLY** and is not load-bearing for any recommendation.

---

## 0. THE ANSWER, BEFORE THE EVIDENCE

Four sentences, then the file paths.

1. **None of the four primitive papers claimed to be a better model.** Every one of them claimed to
   be **a different KIND of object** with a **named property**, and then discharged the property
   claim with evidence of a *categorically stronger type than a benchmark number*.
2. **The minimum dynamics evidence in the record is not a benchmark win.** It is one of four things:
   a **theorem about the update map** (Ramsauer), an **algebraic identity with an already-accepted
   mechanism** (Ramsauer, ET), a **matched-budget ablation that removes the dynamics prior and
   measures the drop** (S4), or a **mechanism-isolating synthetic that a named rival class provably
   cannot solve** (Mamba). Krotov & Hopfield introduced Dense Associative Memory on **XOR and
   MNIST**. Chen et al. introduced neural ODEs at **parity** with a ResNet (0.42 % vs 0.41 %).
3. ⭐ **ICLR says in writing that SOTA is not required** — *"this does not necessarily require
   state-of-the-art results"* (ICLR 2026 Reviewer Guide). So the uncomfortable answer this spoke was
   told to look for — *"reviewers require a benchmark win"* — **is false as stated.**
4. ⛔⛔ **But the true answer is worse for us than that, and §6 states it.** Every one of the four
   discharged its dynamics claim with a **comparative** object: something a rival fails, or an
   identity a rival already accepts. **Our CRR is a parameter-free prediction of our own store's own
   settle half-life. Nothing competes with it and nothing fails it** — w6's own charter says exactly
   this, in its own words: *"There is no lookup comparator for 'does a closed form predict a
   rollout', and this hub claims no row against any rival"*
   (`evidence/outputs/w10-physics-section-evidence/BENCHMARK-STATEMENT.md` §2 N4, sourced to
   `_thinkpad/w6/w6-potentials/LOG.md` §7). ⭐ **That is not a reason to cut the section. It is a
   reason to position it as a MEASURED LAW ABOUT OUR OWN OBJECT — the Edge-of-Stability grammar —
   and never as a primitive.** §5 is that section, drafted.

---

## 1. THE POSITIONING SENTENCE — what each claimed to BE, and what it did not claim

All quotes verbatim from the arXiv abstract or the numbered section named. URLs are the source.

| # | paper | the positioning sentence (verbatim) | ⛔ what it did NOT claim |
|---|---|---|---|
| **P1** | **Krotov & Hopfield, *Dense Associative Memory for Pattern Recognition*, NeurIPS 2016** — `arxiv.org/abs/1606.01164` | *"A model of associative memory is studied, which stores and reliably retrieves many more patterns than the number of neurons in the network. We propose a simple duality between this dense associative memory and neural networks commonly used in deep learning."* | Claims **a duality**, not a better classifier. Evidence named in the abstract is *"two test cases: the logical gate XOR and the recognition of handwritten digits from the MNIST data set."* ⛔ No competitive-vision claim, no scaling claim. |
| **P2** | **Ramsauer et al., *Hopfield Networks is All You Need*, ICLR 2021** — `arxiv.org/abs/2008.02217` | *"We introduce a modern Hopfield network with continuous states and a corresponding update rule."* … *"The new update rule is equivalent to the attention mechanism used in transformers."* … *"These Hopfield layers enable new ways of deep learning, beyond fully-connected, convolutional, or recurrent networks, and provide pooling, memory, association, and attention mechanisms."* | ⛔⛔ **It never claims to beat a transformer at what a transformer does.** It claims to **BE** the thing the transformer already does (*equivalent to*), and then claims wins only in domains where transformers are weak: *"improved state-of-the-art on three out of four considered multiple instance learning problems"*, *"immune repertoire classification"*, *"the UCI benchmark collections of small classification tasks, **where deep learning methods typically struggle***", *"two drug design datasets."* ⭐ **Note the explicit self-selection of the arena.** |
| **P3** | **Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State Spaces* (S4), ICLR 2022** — `arxiv.org/abs/2111.00396` | *"We propose the Structured State Space sequence model (S4) based on a new parameterization for the SSM, and show that it can be computed much more efficiently than prior approaches **while preserving their theoretical strengths**."* | ⛔ The novelty is claimed as an **efficiency/parameterization** result over an **existing** dynamics prior (HiPPO, `arxiv.org/abs/2008.07669`) — *not* as a new dynamics. §4.4 says so against its own interest: *"S4's effectiveness primarily comes from the HiPPO initialization, not the NPLR parameterization."* ⭐ **The paper hands the credit for the dynamics to the prior work and keeps only the tractability.** |
| **P4** | **Gu & Dao, *Mamba*, COLM 2024 (Outstanding Paper)** — `arxiv.org/abs/2312.00752`; listing at `colmweb.org/2024/AcceptedPapers.html` | *"We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements."* … *"As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics."* | ⚠ **This is the one that DOES claim capability**, and it is the only one of the six whose abstract leads with head-to-head numbers (*"5× higher throughput"*, *"Mamba-3B … outperforms Transformers of the same size and matches Transformers twice its size"*). ⛔ Note what it still does not claim: no claim to subsume attention, no claim of a guarantee. |
| **P5** | **Hoover et al., *Energy Transformer*, NeurIPS 2023** — `arxiv.org/abs/2302.07253` (venue from the arXiv comments field) | *"We propose a novel architecture, called the Energy Transformer (or ET for short), that uses a sequence of attention layers that are purposely designed to minimize a specifically engineered energy function."* … §6: *"the entire transformer block (including feed-forward MLP, layer normalization, and residual connections) can be viewed as a single large Hopfield Network, not just attention alone."* | ⛔⛔ **The paper states its own evidence asymmetry in its own Conclusions:** *"We have tested the ET network **qualitatively** on the image completion task, and **quantitatively** on node anomaly detection and graphs classification."* Its own §3 is titled *"**Qualitative Inspection** of the ET framework on ImageNet."* ⭐ **A NeurIPS 2023 paper introduced an energy-based primitive with NO quantitative result in its flagship domain, and said so in the section title.** |
| **P6** | **Chen, Rubanova, Bettencourt & Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018 (Best Paper)** — `arxiv.org/abs/1806.07366` | *"We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network."* … *"These continuous-depth models have **constant memory cost**, adapt their evaluation strategy to each input, and can explicitly trade numerical precision for speed."* | ⛔ **Accuracy is not claimed.** §3, verbatim: *"We find that ODE-Nets and RK-Nets can achieve **around the same performance** as the ResNet."* The claim is a **property** (O(1) memory), and the paper carries a §6 **"Scope and Limitations"** listing minibatching, uniqueness, tolerance choice and reverse-trajectory error against itself. ⚠ Best-Paper status is SECONDARY-sourced (`syncedreview.com/2018/12/03/…`; NeurIPS proceedings page `papers.nips.cc/paper/7892`); the arXiv record carries no award field. |

⭐ **The pattern, stated once.** Five of six positioning sentences are of the form
**"this is a DIFFERENT KIND OF OBJECT, and it has PROPERTY X"** — a duality, an equivalence, a
continuous-depth family, an energy-minimising block. Only Mamba leads with *"it performs better."*
⛔ **And Mamba is the only one of the six reported to have been rejected at a top venue** (see §3.4,
SECONDARY-ONLY).

---

## 2. THE MINIMUM EVIDENCE — not their best result, their *minimum*

⭐ The question asked was: **what would a reviewer have accepted as sufficient for the DYNAMICS
claim specifically** — stripped of everything the paper also did. Below, per paper, the smallest
object that carries the dynamics claim, with its numbers and where the number lives.

### 2.1 Ramsauer (modern Hopfield) — **the minimum is a theorem, not an experiment**
The dynamics claim is discharged **entirely in closed form**, in §2 of `arxiv.org/abs/2008.02217`:
- **Theorem 1 — global convergence.** *"The update rule Eq. (3) converges globally: for
  ξ^{t+1}=f(ξ^t), the energy E(ξ^t) → E(ξ*) for t→∞ and a fixed point ξ*."* Proof by CCCP.
- **Theorem 2** strengthens it to `‖ξ^{t+1} − ξ^t‖ → 0` and stationary-point convergence.
- **Theorem 3 — exponential storage capacity.** `N ≥ √p · c^{(d−1)/4}`, with the paper's own worked
  constants: *"proven for c ≥ 3.1546 with β=1, K=3, d=20 and p=0.001"* and
  *"proven for c ≥ 1.3718 with β=1, K=1, d=75, and p=0.001."*
- **Theorem 4 — retrieval with ONE update**, bound `‖J^m‖₂ ≤ 2βNM²(N−1)exp(−β(Δᵢ − 2max{…}M))`.
- **Theorem 5 — exponentially small retrieval error**, `‖xᵢ − xᵢ*‖ ≤ 2e(N−1)M exp(−βΔᵢ)`.

⭐⭐ **There is no retrieval benchmark in the main text at all.** The main-text figures are the
energy/fixed-point illustration (Fig. 1), the layer diagrams (Figs. 2–5) and the **BERT head
analysis** (Fig. 2 / App. Fig. A.3). ⛔ **The empirical section carries UTILITY in self-selected
low-stakes domains; it does not carry the dynamics claim.**
⇒ **MINIMUM = {a convergence theorem} + {a capacity theorem with worked constants} + {an identity
with a mechanism the reviewer already accepts}.** The identity is the load-bearing part: *"the new
update rule is equivalent to the attention mechanism used in transformers."*

### 2.2 S4 — **the minimum is one matched-budget ablation**
`arxiv.org/abs/2111.00396` §4.4, *"SSM Ablations: the Importance of HiPPO"*, verbatim:
- Setup: *"all experiments in this section were performed on the sequential CIFAR-10 task … Models
  were constrained to at most **100K trainable parameters** and trained with a simple plateau
  learning rate scheduler and **no regularization**."*
- The number: *"when A is trained, **all initializations are able to reach perfect training
  accuracy**. However, their validation accuracies are separated by **over 15 %**."*
- The control that kills the obvious confound: *"random NPLR matrices still do not perform well,
  which validates that S4's effectiveness primarily comes from the **HiPPO initialization, not the
  NPLR parameterization**."*
- With 0.1 dropout, *"the HiPPO initialization — in other words, the full S4 method — achieves
  **84.27 % test accuracy with just 100K parameters**."*
- Inherited prior evidence, quoted in its §2.2: the LSSL *"found that simply modifying an SSM from a
  random matrix A to equation (2) improved its performance on the sequential MNIST benchmark from
  **60 % to 98 %**."*

⇒ **MINIMUM = one swap-out of the dynamics prior at matched parameter budget, with the
generalization gap reported and the parameterization confound separately excluded.**
⚠ S4 also had the headline (*"SoTA on every task from the Long Range Arena … solving the challenging
Path-X task of length 16k that all prior work fails on"*, **88 % vs 50 % random guessing**, §1) — but
that is its **best** result, not the minimum, and it is a *capability* claim, not the dynamics claim.

### 2.3 Mamba — **the minimum is a synthetic that the named rival class cannot solve**
`arxiv.org/abs/2312.00752` §4.1:
- **Selective Copying.** The task exists specifically to remove the shortcut: *"LTI SSMs (linear
  recurrences and global convolutions) can easily solve this task by only keeping track of time
  instead of reasoning about the data … The Selective Copying task prevents this shortcut by
  randomizing the spacing between tokens."* Result: *"gated architectures such as H3 and Mamba only
  partially improve performance, while the selection mechanism (modifying S4 to S6) easily solves
  this task."*
- **Induction Heads.** *"We train a **2-layer** model … at sequence length **256**, with a vocab size
  of **16** … We use a model dimension D of **64** for Mamba and 128 for the other models,"*
  evaluated *"from 2⁶ = 64 up to 2²⁰ = 1048576 at test time."* Abstract §1: *"Mamba not only solves
  them easily but can extrapolate solutions indefinitely long (>1M tokens)."*

⇒ **MINIMUM = a 2-layer, width-64, vocab-16 toy — but one designed so that a NAMED RIVAL CLASS
provably fails it, with the closed shortcut stated.** ⭐ The toy is tiny; the **argument structure**
is not.

### 2.4 Energy Transformer — **the minimum is construction + a Lyapunov statement + pictures**
`arxiv.org/abs/2302.07253`:
- The dynamics claim is **by construction**, stated in the Fig. 2 caption: *"The operations inside
  the ET block are carefully engineered so that the entire network has a **global energy function,
  which decreases with time and is bounded from below**."* Appendix: *"This energy function will
  decrease as the forward pass progresses until it reaches a local minimum."*
- The empirical support in the flagship domain is **visual**: Fig. 3, ImageNet-1k validation, *"50 %
  of the patches are masked"*, *"output reconstructions after **12 time steps**"*; Fig. 4 tracks the
  ATTN and HN gradient contributions across time in the image plane — *"Early in time, almost all
  signal to the masked tokens comes from the ATTN block … Later in time … almost all signal to masked
  tokens comes from the HN module."*
- Its own summary of the alignment claim: *"The qualitative investigation reveals the perfect
  alignment between the theoretical design principles of our network and its empirical computation."*
- ⭐ The **quantitative** burden is discharged **in a different domain**: graph anomaly detection
  (Yelp / Amazon / T-Finance / T-Social, *"mean and standard deviation over 5 runs"*, macro-F1 and
  AUC, Table 1) and graph classification (TUDataset, 8 datasets, *"100 runs of 10-fold cross
  validation"*, Table 2).

⇒ **MINIMUM = {the energy decreases and is bounded below, by construction} + {a visualisation showing
the mechanism doing what the design says} + {a quantitative win, moved to a domain where the
comparison is winnable}.**

### 2.5 Neural ODEs — **the minimum is parity plus a property, and a toy for the latent dynamics**
`arxiv.org/abs/1806.07366`:
- MNIST, §3 Table 1: **ResNet 0.41 % / 0.60 M params / O(L) memory**; **RK-Net 0.47 % / 0.22 M**;
  **ODE-Net 0.42 % / 0.22 M / O(1) memory.** Text: *"around the same performance as the ResNet."*
- Instrument-level checks reported as results: error can be controlled by tolerance (Fig. 3a);
  forward time ∝ NFE (3b); *"the number of evaluations in the backward pass is roughly half of the
  forward pass"* (3c); NFE grows through training (3d).
- ⭐⭐ **The LATENT-DYNAMICS claim is carried by a synthetic.** §5.1: *"We generated a dataset of
  **1000 2-dimensional spirals** … sampled at 100 equally-spaced timesteps … half are clockwise while
  the other half counter-clockwise … we add gaussian noise."* Table 2, predictive RMSE on 100
  extrapolated time points: **RNN 0.3937 / 0.3202 / 0.1813** vs **Latent ODE 0.1642 / 0.1502 /
  0.1346** at 30/50/100 observations. Plus Figs. 9(c)/10: the latent trajectories separate into two
  clusters and vary smoothly with the initial state.
- §6 **"Scope and Limitations"** names minibatching cost, Picard uniqueness conditions, tolerance
  choice (*"the default value of 1.5e-8 was used"*), and reverse-trajectory error, of which it says
  *"We did not find this to be a practical problem, and we **informally checked** that reversing many
  layers of continuous normalizing flows with default tolerances recovered the initial states."*

⇒ **MINIMUM = parity on a real task + a new PROPERTY the incumbent lacks + a 2-D synthetic for the
dynamics + an explicit limitations section.** ⭐ That combination won a NeurIPS Best Paper.

### 2.6 ⭐ The two closest published analogues to what WE have — neither is a primitive paper
Our CRR is a **measured law with a parameter-free numerical prediction and no rival**. The published
shape that fits that is **not** a primitive paper; it is an **empirical-science** paper:

- **Cohen, Kaur, Li, Kolter & Talwalkar, *Gradient Descent on Neural Networks Typically Occurs at the
  Edge of Stability*, ICLR 2021** — `arxiv.org/abs/2103.00065`. Abstract, verbatim: *"**We
  empirically demonstrate** that full-batch gradient descent on neural network training objectives
  typically operates in a regime we call the Edge of Stability. In this regime, the maximum
  eigenvalue of the training loss Hessian **hovers just above the numerical value 2/(step size)**,
  and the training loss behaves non-monotonically over short timescales, yet consistently decreases
  over long timescales. Since this behavior is inconsistent with several widespread presumptions in
  the field of optimization, **our findings raise questions** as to whether these presumptions are
  relevant to neural network training."* ⭐⭐ **A sharp, parameter-free numerical prediction (2/η),
  zero benchmark numbers, zero rivals beaten, ICLR 2021.** The verb is *"we empirically demonstrate"*
  and the payoff sentence is *"raise questions as to whether these presumptions are relevant."*
- **Frankle & Carbin, *The Lottery Ticket Hypothesis*, ICLR 2019 (Best Paper)** —
  `arxiv.org/abs/1803.03635`. *"Based on these results, **we articulate the 'lottery ticket
  hypothesis'**: dense, randomly-initialized, feed-forward networks contain subnetworks ('winning
  tickets') that — when trained in isolation — reach test accuracy **comparable to** the original
  network in a similar number of iterations."* Evidence: *"We consistently find winning tickets that
  are less than 10-20 % of the size of several fully-connected and convolutional feed-forward
  architectures for **MNIST and CIFAR10**."* ⭐ A **hypothesis**, articulated as such, on two small
  datasets, at **parity**.

⛔ **Neither of these claims to be a primitive.** That is precisely why they got away with the
evidence they had. ⭐ **This is the single most useful fact in this brief for §5.**

---

## 3. MOTIVATION vs CLAIM at ICLR — the two bars, from ICLR's own instructions

⛔ Reminder: **no OpenReview text is used.** Source below is the **ICLR 2026 Reviewer Guide**,
`https://iclr.cc/Conferences/2026/ReviewerGuide`, fetched 2026-09-21 (stripped text kept in the
session scratchpad; not a deliverable). Quotes are verbatim.

### 3.1 The guide separates the two bars explicitly, in one numbered list
Reviewers are told to *"Answer four key questions for yourself to make a recommendation to Accept or
Reject"*:

| # | the question, verbatim | which bar |
|---|---|---|
| i | *"What is the specific question and/or problem tackled by the paper?"* | framing |
| ii | *"**Is the approach well motivated, including being well-placed in the literature?**"* | ⭐ **MOTIVATION** |
| iii | *"**Does the paper support the claims?** This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous."* | ⭐ **CLAIM** |
| iv | *"What is the significance of the work? Does it contribute new knowledge and sufficient value to the community?"* | significance |

⭐⭐ **Read (ii) and (iii) together and the whole distinction falls out.**
- **Motivation is satisfied by PLACEMENT — by literature, by argument, by a pointer.** The guide asks
  whether the approach is *well motivated* and *well-placed*, not whether the motivation is measured.
- **A claim is satisfied only by CORRECT, RIGOROUS RESULTS** — and the guide deliberately writes
  *"whether theoretical or empirical"*, i.e. **a theorem discharges a claim exactly as an experiment
  does.** This is the sentence that licenses §2.1's theorem-only route.

### 3.2 ⭐ The guide says outright that SOTA is not the bar
Question (iv), verbatim and in full:
> *"What is the significance of the work? Does it contribute new knowledge and sufficient value to
> the community? **Note, this does not necessarily require state-of-the-art results.** Submissions
> bring value to the ICLR community when they convincingly demonstrate new, relevant, impactful
> knowledge (incl., empirical, theoretical, for practitioners, etc)."*

⛔ **This is dispositive against the hypothesis "reviewers require a benchmark win."** It is written
into the reviewing instructions. ⚠ What it does **not** say is that a claim may be stated above its
evidence — note the verb: *"convincingly demonstrate."*

### 3.3 ⭐⭐ The guide's own worked REJECT example separates score-bearing from non-score-bearing
The guide ships two example reviews. The leaning-to-reject one lists its four reasons and then, at
the very end, a bucket headed **"Things to improve the paper that did not impact the score."**
The score-bearing complaints, verbatim:

- **A motivation failure, listed FIRST among the four reasons:** *"the paper never clearly
  demonstrates the existence of problem they are trying to solve (nor differentiates it from the
  usual problem of generalizing well)"*, expanded as *"There is **no experiment in this paper that
  clearly shows** how this temporal generalization problem is different from the need to generalize
  well with function approximation. The paper points to references to establish the existence of the
  problem, but … that is not work to build upon."*
  ⚠⚠ **Read this carefully: motivation CAN sink a paper — when the motivation is doing a claim's
  job.** The objection is that the paper *asserted a phenomenon exists* and then built on it. **That
  is the failure mode our outlook section is most exposed to.**
- **A justification failure spanning both routes:** *"the algorithm is not well justified **either by
  theory or practice**"* — and *"Can you provide a proof or any kind of evidence that the proposed
  approach is sound, or how it's fixed point relates to TD?"* ⭐ Again the theory-**or**-practice
  disjunction: either discharges it; neither being present sinks it.
- **An over-claim, itemised as its own bullet:** *"**Over-claim not supported by the results:** 'we
  see that HR-TD is able to find a representation that is better at keeping the target value separate
  than TC is'. **The results do not show this.**"* ⭐ Note the shape of the kill: the reviewer quotes
  **one sentence** of the paper and says the results do not support **that sentence**. ⛔ **Claims are
  audited sentence by sentence, not section by section.**

The leaning-to-**accept** example, for contrast, accepts a paper whose empirical support is *"an
empirical study in the Inverted Double Pendulum domain"* — one toy domain — with the summary *"this
paper could be a significant algorithmic contribution, **with the caveat for some clarifications** on
the theory and experiments."*

### 3.4 ⚠ SECONDARY-ONLY, and not load-bearing: the Mamba ICLR-2024 outcome
Mamba is published at **COLM 2024 with an Outstanding Paper Award** — **VERIFIED** directly from
`https://colmweb.org/2024/AcceptedPapers.html`, which lists *"🏆 Outstanding Paper Award · Mamba:
Linear-Time Sequence Modeling with Selective State Spaces · Albert Gu, Tri Dao."*
⚠ It is **widely reported** to have been rejected at ICLR 2024 first, with reviewer criticism about
missing Long Range Arena results and over-reliance on perplexity. ⛔ **I could not verify a single
word of that from the primary record** — openreview.net is blocked here (see the header). The
secondary sources found are a CSPaper forum thread (`forum.cspaper.org/topic/2/…`), a Hacker News
thread (`news.ycombinator.com/item?id=39588854`), a Medium post
(`medium.com/@jelkhoury880/why-mamba-was-rejected-9b2f05f2141c`) and a post by Wenhu Chen on X.
**These are blog-grade.** ⛔ **No recommendation in §5 rests on this.** It is recorded because, if
true, it is the one data point where *the biggest benchmark numbers of the six papers did not buy an
acceptance* — which would sharpen §3.2 rather than contradict it.

### 3.5 The operative distinction, in one table
| | **MOTIVATION** | **CLAIM** |
|---|---|---|
| **what satisfies it** | placement in the literature; an argument; a citation; a plausible mechanism | correct, rigorous results — *"whether theoretical or empirical"* |
| **what a reviewer does with it** | checks it is *well-placed*, not that it is measured | audits it **sentence by sentence** against the artefacts |
| **failure mode** | ⛔ it silently does a claim's job (§3.3, reject reason #2) | ⛔ over-claim: *"The results do not show this."* |
| **the grammar that signals which one you mean** | *"we are motivated by…"*, *"this suggests…"*, *"a natural question is…"*, *"we conjecture…"* | *"we show"*, *"we establish"*, *"our method achieves"*, *"X outperforms Y"* |
| **SOTA required?** | no | ⛔ **no** — *"does not necessarily require state-of-the-art results"* — but **support at the stated level IS required** |

⭐ **The whole operational lesson: the bar you must clear is set by YOUR OWN VERB.** Nothing forces a
paper to make a claim. A paper that says *"we empirically demonstrate that our store's settle
half-life follows this closed form"* owes a measurement — and we have one. A paper that says *"this
suggests wells are a usable read"* owes a citation and a plausible mechanism. ⛔ **A paper that says
"this makes our store a new class of dynamical memory" owes a rival that fails.**

---

## 4. THE FIVE ROUTES, AND WHICH ONE WE ARE ON

| route | what it is | who used it | ⭐ does it need a benchmark win? | do we have it? |
|---|---|---|---|---|
| **R-A · THEOREM** | a proof about the update map: convergence, capacity, error bound | Ramsauer Th. 1–5 | **no** | ⭐ **partly — and this is our strongest under-used asset.** The CRR *is* a closed form derived from the step matrix (D3/D4 closed with D5), reproduced exactly against the shorts' own printed numbers before any data was touched: **27.02681** against a printed **27.03**, **0.9989994995** against **0.998999499**, **692.4538** against **692.5** — `w10-physics-section-evidence/EVIDENCE.md` §2.1 |
| **R-B · IDENTITY** | show the new object **is** an object already accepted | Ramsauer (= attention); ET (= one big Hopfield net) | **no** | ⚠ **available and not yet written** — the zero-dynamics/settling-off limit is exactly this shape, and it is w12 `spoke/clue-theorist`'s commission (`advisor-wave-plan-w12-w14.md` §1), not mine. **Left as an open slot in §5.** |
| **R-C · MATCHED ABLATION** | remove the dynamics prior at matched budget, report the drop | S4 §4.4 (>15 % val gap at ≤100K params) | **no — the DELTA is the result** | ⛔ **we have the ablation and it runs the WRONG WAY** (§5.0(2)) |
| **R-D · MECHANISM SYNTHETIC** | a toy a named rival class provably fails | Mamba §4.1 | **no** | ⛔ **NOT HELD.** The CRR error is *identical* across `field` / `removed` / `shuffled` / `pca_tensor` (**0.0428–0.0435**) — `EVIDENCE.md` §2.8. **Nothing fails it.** |
| **R-E · PHENOMENON** | *"we empirically demonstrate"* a measured law; no rival, no benchmark | Edge of Stability (ICLR 2021); LTH (ICLR 2019) | **no** | ⭐⭐ **YES — this is the route our evidence actually licenses.** |

---

## 5. ⭐⭐ THE DELIVERABLE — the recommended shape of our latent-dynamics outlook section

### 5.0 The governing decision
> **Position the latent-dynamics material as R-E (a measured law about our own object), in the
> Edge-of-Stability grammar, under the claim ladder's rung 3 ("wells as a use").**
> ⛔ **Never as a primitive, never as a fifth model class, never as a scoreboard row.**

Three reasons, each with a file:

1. **Our own charter already ruled it out of the scoreboard, before any number was read:** *"There is
   no lookup comparator for 'does a closed form predict a rollout', and this hub claims no row
   against any rival."* — `BENCHMARK-STATEMENT.md` §2 N4 ← `_thinkpad/w6/w6-potentials/LOG.md` §7.
2. **The same dynamics COST composition, measured on the same object and the same seeds**: row-4 mean
   paired Δ vs `pca_tensor` = **−39.959 / −46.918 / −39.061** at median displacement 0.39;
   **−2.735 / −3.796 / −2.959** at one-step; **−0.531 / −0.510 / −0.388** at the mildest setting that
   moves the state at all; and the zero-step arm is bit-identical to the incumbent
   (`np.array_equal` asserted) at Δ = **0.000 / 0.000 / 0.000** — `EVIDENCE.md` §1, ←
   `_thinkpad/w6/SCORED-LEDGER.md` + `_thinkpad/w6/w6-compose-physics/LOG.md` §FINAL REPORT.
   ⭐ **This is R-C run backwards. It must be printed, not hidden — see §5.2.**
3. **CM-21 already retired the four positioning claims this section would otherwise drift into**, and
   its approved replacement wording claims *controllability*, not *capability*
   (`evidence/claims_matrix.md`, CM-21 row).

### 5.1 The paragraph, drafted — six sentences, in this order
> ⚠ Numbers below are quoted from `w10-physics-section-evidence/EVIDENCE.md` §§1, 2.2–2.6, which
> sources every cell to `_thinkpad/w6/w6-potentials/RESULTS-LEDGER.md` and
> `_thinkpad/w6/w6-potentials/test/s{0,1,2}/results.json`. **The Head should re-verify against those
> files before print; this spoke re-derived nothing (NR-4).**

1. **MOTIVATION — one sentence, motivation grammar, no claim verb.**
   *"The store described above is the zero-dynamics corner of a dissipative field model; a natural
   question is whether switching the settling on leaves the store's relaxation predictable."*
   — grammar per §3.5: *a natural question is*. ⛔ Do not write *"this shows the store is dynamical."*
2. **THE RESULT, at exactly its own level — the verb is "predicts", never "beats".**
   *"It does: the curvature–retention relationship predicts the field store's settle half-life
   parameter-free — nothing fitted — to a factor **1.105** at the median held-out corner (median
   |log₁₀ n_meas/n_pred| = **0.0433 / 0.0435 / 0.0431** against a pre-registered bar of **0.10**,
   seed spread **0.0004**; 49 held-out corners × 3 seeds, **192,975 rows**, store byte pin
   **524,288 B at tolerance 0**, matched on every leg)."* (`EVIDENCE.md` §2.3.)
3. **THE ONE COMPARATIVE SENTENCE WE ARE ENTITLED TO — with its exact scope.**
   *"It also beats a one-parameter rival **fitted with sight of the held-out measurements** on **49 of
   49** corners on every one of three seeds (mean paired Δ **−0.5968 / −0.5796 / −0.5982**; the
   rival's own median error is **0.640 / 0.623 / 0.641** in log₁₀)."* (`EVIDENCE.md` §2.3.)
   ⚠ **Say "a fitted rival", never "a baseline" and never "a competitor."** The rival is a functional
   form (`FIT1`: `n = C/(εμ)²`), not a model; calling it a baseline imports the scoreboard reading
   that §5.0(1) forbids.
4. **THE FALSIFIER THAT FAILED, in the same breath — this is the sentence that buys the section.**
   *"One of the six pre-registered bars **missed on all three seeds**: the V-curve argmin sits at
   **0.3362 / 0.3301 / 0.3520** of γ_crit against a factor-2 bar (**0.3782 / 0.3784 / 0.3834**
   against the exact inversion of h\*(γ)=h), and the miss was pre-registered with its cause in a dated
   amendment **before the run** — approaching the exceptional point the 2×2 map is defective, so the
   predicted dip fills in."* (`EVIDENCE.md` §2.6.)
   ⭐ Follow it immediately with what makes the miss a **result**: *"all of the error sits in the one
   h/h\* bucket 0.5–2 that straddles that point — median **0.0008** below h/h\* = 0.1 against
   **0.1482** inside it — so the theory predicts where its own simplest form stops."* (§2.5.)
   ⭐ And the instrument honesty: *"on an exact quadratic, where the law holds by construction, the
   same measurement chain recovers it only to median **0.0349**, so the 0.10 bar is ≈3× the
   instrument's own floor."* (§2.2.)
5. **THE COST SENTENCE — mandatory, in the main text, not a footnote.**
   *"The same dynamics, at every strength at which they move the state, **cost** composition on this
   object: row-4 mean paired Δ **−39.959 / −46.918 / −39.061** at median displacement 0.39, and
   **−0.531 / −0.510 / −0.388** at the mildest setting that moves it at all."* (`EVIDENCE.md` §1.)
6. **THE NON-CLAIM SENTENCE — explicit, one line, in the main text.**
   *"This is a dynamics result about our own store. It occupies no row of the evaluation, it beats no
   model, and we make no capability claim from it."*

⭐ **Optional seventh sentence, only if `spoke/clue-theorist` lands R-B this wave:** one line stating
the settling-off identity (the subspace store **is** the CLU at zero dynamics). ⛔ **Do not write it
from this brief** — I did not derive it and it is not my commission (NR-8).

### 5.2 ⭐ Why this shape and not a shorter one
§3.3's reject example kills a paper for **asserting a phenomenon and then building on it**. Sentences
4–6 are what stop that reading: they turn the section from *"look, dynamics"* into *"here is a
measured law, here is the bar it missed, here is what it costs, here is what it does not buy."*
⭐ **The referee has already told us this is the currency we trade in:** w10 `spoke/referee` R3 —
*"the negatives discipline and `verify_tables.py` are genuinely above the field"*
(`w10-advisor-report.md` §2). ⛔ **Cutting sentence 4 or 5 to make the section look stronger spends
the one asset the reviewers scored us above the field on.**

### 5.3 ⛔ THE FORBIDDEN LIST for this section — bind these to the draft
| ⛔ never write | why, with the file |
|---|---|
| *"multilinear/bilinear factored generation with a request-indexed tensor head"* | Tenenbaum & Freeman (2000) / TensorFaces (2002) own it — `advisor-wave-plan-w12-w14.md` §0, binding |
| *"a competitor to the transformer"* / *"an alternative sequence model"* | CM-21(d), RETIRED: *"indefensible at 1–1.6 bits/register with no win over a trivial baseline"* — `claims_matrix.md` CM-21 |
| *"attention/SSMs have no retention-guarantee analogue"* | CM-21(a), REFUTED: HiPPO-LegS has one (approximation error `O(tL/√N)`, gradient `Θ(1/t)` — polynomial, not exponential); *"a reviewer will know"* |
| *"retrieval is a rollout, not a weighted sum" as OUR novelty* | CM-21(b): published — Kong, Brewer & Lai, *Nature Communications* 15 (2024), doi:10.1038/s41467-024-49190-4. **Must be cited** |
| *"continuity is our NTM/DNC escape hatch"* | CM-21(c): FALSE — canonical NTM **and** DNC were already fully soft, continuous and end-to-end differentiable |
| *"tunable memory permanence"* in the broad form | narrowed by EDEN (NeurIPS 2025); only **permanence as a per-item design parameter chosen at write time, three modes, one symmetry-protected** survives |
| *"the CRR result supports a scoreboard claim"* | ⛔ refuted on the same object and the same seeds — K21, `w10-draft-v0/DRAFT-v0.md`; Advisor-ruled |
| *"the dynamics make the store better"* | §5.1(5): on row 4 they make it **−39.959 / −46.918 / −39.061** worse |
| *"9/9 seeds"*, or the *"1e-16…1e-14 on Shapes3D, two independent implementations"* band, anywhere near this section | wrong as written — `w10-advisor-report.md` §1. The clean number is **CMAZE-2, 7.5e-16…8.8e-16, subspace projector 0.000e+00, 3 seeds** |
| any **raw** `d log n½/d log T` or `d log n½/d log γ` exponent used as designed-vs-emergent evidence | CM-16b instrument warning: *"a matched designed control gives the same shallow slopes"* ⇒ **artifact**. Never quote `n½` without `Δ` and `ℓ_θ/Δ` |
| *"CMAZE-1"* for the w9 maze rows | mislabelled — it is **CMAZE-2** (w9 Amendment 2; `w10-advisor-report.md` §3) |
| a full-state read presented as retrieval | CM-18 qualifier (1): a full-state read scores **1.000 on a blank landscape** ⇒ *"any retrieval claim that reads the address plane is DECORATIVE"* |

### 5.4 ⭐ Where the section goes, and what the ladder says
- The claim ladder is **(1) exact deletion on a compositional generative store · (2) the interface
  split · (3) wells as a use** (`advisor-wave-plan-w12-w14.md` §0). ⭐ **The latent-dynamics section
  is rung 3 and must be visibly rung 3** — after both claims, framed as a *use*, ⛔ never as a third
  contribution bullet co-equal with the other two.
- ⚠ **Precedent from our own referees on dilution:** w10 `spoke/referee` R2's complaint was *"two
  main-text tables report the same arm in two metrics on two splits"* while it named §6.1
  plan-vs-latent *"the best thing in the paper, and exact"* (`w10-advisor-report.md` §2); and the v2
  referee's approving formula for a contributions list was that it *"reads as one claim plus its
  evidence and its perimeter, which is the right shape"* (`evidence/outputs/v2-referee-final.md`
  — ⚠ that referee was reading the **v2 CRR-led paper**, a different object from the ICLR-long draft;
  the formula transfers, the verdict does not).
- **Keep the deletion sentence exactly as ruled — *"downdate the statistic, not the truncated
  factors"*** — and ⛔ **keep it out of this section.** It is rung 1. Mixing rung 1's sentence into
  rung 3's paragraph invites a reviewer to audit rung 1's evidence against rung 3's prose.

### 5.5 ⭐ What would UPGRADE this from R-E to a claim — one experiment, named
From §4, **R-D is the cheapest missing route.** The CRR's error is **identical across `field`,
`removed`, `shuffled` and `pca_tensor` (0.0428–0.0435)** because *the law is about the dynamics* —
stated in advance — and the arms separate only in the curvature **spectrum** (`field` [0.2727,
1.0000] vs `removed` [1.0000, 1.0000], no non-trivial curvature at all) — `EVIDENCE.md` §2.8.
⇒ **No arm of ours fails the law, so the law distinguishes nothing.**
⭐ **A comparator predicted IN ADVANCE to be mispredicted by the closed form — and that is — would
convert the law from a description of our object into a claim about a class.** §2.5 already names
where to look: the `h/h*` ∈ [0.5, 2] bucket where the map is defective and the median error jumps
**0.0008 → 0.1482**. ⛔ **Not run, not scoped, not costed by this spoke** — a suggestion to the Hub,
not a result (NR-6).

---

## 6. ⛔⛔ WHAT WOULD PROVE THIS SPOKE WRONG — and the honest answer

The task set the falsifier as: *"reviewers require a benchmark win for a primitive claim."*

⭐ **On the evidence, that statement is FALSE, and I say so plainly.** ICLR writes the opposite into
its reviewer instructions — *"Note, this does not necessarily require state-of-the-art results"*
(§3.2) — and the record agrees: Krotov & Hopfield introduced Dense Associative Memory on **XOR and
MNIST**; Chen et al. won a NeurIPS Best Paper at **0.42 % vs 0.41 %** parity with an explicit *"Scope
and Limitations"* section; Ramsauer discharged the dynamics claim in **five theorems with no
retrieval benchmark in the main text at all**; Hoover et al. published an energy-based primitive at
NeurIPS 2023 with a section titled *"**Qualitative** Inspection"* and **no quantitative result in its
flagship domain**, saying so in their own Conclusions.

⛔⛔ **But the finding that IS uncomfortable — reported because it is more useful than the comfortable
one — is this:**

> **Every one of those papers discharged its dynamics claim with a COMPARATIVE or GUARANTEED object:
> a theorem, an identity with a mechanism the reviewer already accepts, a matched-budget ablation
> whose delta is the result, or a synthetic that a NAMED RIVAL CLASS fails. Our CRR is none of these.
> It is a parameter-free prediction of our own store's own settle half-life, and by our own
> measurement NOTHING FAILS IT — the error is identical across `field`, `removed`, `shuffled` and
> `pca_tensor` (0.0428–0.0435, `EVIDENCE.md` §2.8) — while the dynamics that produce it COST
> composition by −39.959 / −46.918 / −39.061 on row 4 (`EVIDENCE.md` §1).**

⭐ **So the correct reading is not "we need a benchmark." It is "we do not have a primitive claim,
and we should stop reaching for one."** What we have is an **R-E phenomenon result**, and the
published precedent for that — Edge of Stability, the Lottery Ticket Hypothesis — is strong, recent,
top-venue, and needed **no benchmark and no rival at all**, precisely because neither paper claimed
to be a primitive. ⛔ **The mistake available to us is not under-claiming. It is writing rung-3
material in rung-1 grammar and being audited at the harder bar for no gain.**

⚠ **Second honest caveat, against my own brief.** §3 is built entirely on ICLR's published guidance
because **openreview.net is blocked from this box (NR-1)**. Published guidance states the *intended*
bar; actual reviewer behaviour can be harsher, and the one data point suggesting so (Mamba, §3.4) is
blog-grade. I could not check that here and I will not assert it either way. ⛔ **If the Hub wants
the behavioural bar rather than the stated one, that needs a box with OpenReview access and is a
DECLARED NOT-RUN here.**

---

## 7. LIMITS AND DECLARED NOT-RUNs

| # | not run | reason |
|---|---|---|
| **NR-1** | ⛔ **Any OpenReview review text, rating, or meta-review, for any of the papers in §1** | `openreview.net` returns `{"name":"ChallengeRequiredError","status":403}` on `/forum`, `/pdf`, `api.` and `api2.` endpoints, with and without a browser User-Agent. Verified four ways. **No review is quoted anywhere in this brief.** |
| **NR-2** | The Mamba ICLR-2024 rejection as a *primary* fact | NR-1. Recorded in §3.4 as **SECONDARY-ONLY**, blog-grade sources named, **not load-bearing** for any recommendation. |
| **NR-3** | Venue/award confirmation via Semantic Scholar | `api.semanticscholar.org` returned HTTP **429 Too Many Requests**. Venues instead taken from the arXiv comments field (ET → NeurIPS 2023, **verified**) and the COLM accepted-paper list (Mamba → COLM 2024 Outstanding Paper, **verified**). Neural ODE Best Paper is **SECONDARY-sourced** and flagged as such in §1 P6. |
| **NR-4** | ⛔ Any re-derivation or re-measurement of a CHLU number | Out of scope: this spoke is read-only with **zero scored runs**. **Every CHLU number in this brief is quoted with the file it came from and was NOT re-computed by me.** |
| **NR-5** | The `chlu.__file__`-inside-my-worktree assertion | No `chlu` import occurred and **no number here comes from the package**, so the assertion has nothing to guard. Declared rather than silently skipped. |
| **NR-6** | The experiment named in §5.5 (a comparator the CRR is predicted in advance to mispredict) | A suggestion to the Hub. Not scoped, not costed, not run. |
| **NR-7** | A survey of whether a benchmark satisfying `BENCHMARK-STATEMENT.md` §3's ten requirements already exists in the literature | Not my commission, and the existing statement already declares *"no scout in this evidence base looked, at this wave, for one."* ⚠ **Still true after this wave.** |
| **NR-8** | The zero-dynamics / settling-off identity statement (R-B in §4, optional sentence 7 in §5.1) | Commissioned to w12 `spoke/clue-theorist`, not to me (`advisor-wave-plan-w12-w14.md` §1). §5 leaves the slot open rather than filling it from a scout's guess. |
| **NR-9** | Any check of ICLR 2025 / earlier reviewer guides for drift in the wording quoted in §3 | Only the **2026** Reviewer Guide and **2026** Area Chair Guide were fetched. The quoted sentences are 2026's. |

⚠ **Scope of §2.** The "minimum" per paper is **my reconstruction** of which object carries the
dynamics claim once everything else is stripped — argued from the papers' own section structure,
section titles and own words, but it is **an argument, not a measurement**. ⛔ It cannot be checked
against the reviews that actually decided those papers (NR-1).

⚠ **Scope of §5.** The drafted paragraph is a **shape and a wording**, not new evidence. Every number
in it already exists in `w10-physics-section-evidence/EVIDENCE.md`. ⛔ **If a number there changes,
the paragraph is wrong and must be rebuilt, not patched.**

---

## 8. SOURCES

**Primary papers** (abstracts verbatim from arXiv; full text read via ar5iv):
`arxiv.org/abs/1606.01164` Krotov & Hopfield, *Dense Associative Memory for Pattern Recognition* ·
`arxiv.org/abs/2008.02217` Ramsauer et al., *Hopfield Networks is All You Need* ·
`arxiv.org/abs/2008.07669` Gu et al., *HiPPO: Recurrent Memory with Optimal Polynomial Projections* ·
`arxiv.org/abs/2111.00396` Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State
Spaces* (S4) · `arxiv.org/abs/2312.00752` Gu & Dao, *Mamba* ·
`arxiv.org/abs/2302.07253` Hoover et al., *Energy Transformer* ·
`arxiv.org/abs/1806.07366` Chen et al., *Neural Ordinary Differential Equations* ·
`arxiv.org/abs/2103.00065` Cohen et al., *…Edge of Stability* ·
`arxiv.org/abs/1803.03635` Frankle & Carbin, *The Lottery Ticket Hypothesis*.

**Venue confirmations:** `colmweb.org/2024/AcceptedPapers.html` (Mamba, Outstanding Paper —
**verified**) · arXiv comments field of 2302.07253 (ET → NeurIPS 2023 — **verified**) ·
`papers.nips.cc/paper/7892-neural-ordinary-differential-equations` and
`syncedreview.com/2018/12/03/neurips-2018-opens-best-papers-announced/` (Neural ODE Best Paper —
**SECONDARY**).

**Reviewing standards:** `iclr.cc/Conferences/2026/ReviewerGuide` — source of every quote in §3 ·
`iclr.cc/Conferences/2026/AreaChairGuide` — fetched and checked; it adds nothing on the claim bar
beyond *"scientific claims and merits of a submission"*, recorded here so it is not re-fetched.

**Programme evidence (read-only, not re-derived — NR-4):**
`evidence/outputs/w9-advisor-report.md` (current state) ·
`evidence/outputs/w10-advisor-report.md` (current state) ·
`evidence/outputs/w10-physics-section-evidence/EVIDENCE.md` §§1, 2.1–2.8 ·
`evidence/outputs/w10-physics-section-evidence/BENCHMARK-STATEMENT.md` §§1–3 ·
`evidence/outputs/w10-draft-v0/DRAFT-v0.md` (K19 / K20 / K21) ·
`evidence/outputs/advisor-wave-plan-w12-w14.md` §0 and §1 ·
`evidence/claims_matrix.md` (CM-16a/b, CM-18, CM-21) ·
`evidence/outputs/v2-referee-final.md` (the "one claim plus its evidence and its perimeter" shape —
⚠ written about the v2 CRR-led paper, a different object).
