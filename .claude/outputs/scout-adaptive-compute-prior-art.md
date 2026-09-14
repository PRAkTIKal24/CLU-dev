# scout-adaptive-compute-prior-art — web-scout report

Task + acceptance criterion: verify whether V1's energy-gated boost-retry → shell-jump cascade is open territory across 4 prior-art fronts; deliver verdicts + dangerous-neighbor list + associative-recall task specs + bibtex.
Status: **done**

What I did:
- Ran targeted web searches + primary-source fetches across all 4 fronts (adaptive compute/early-exit/cascades; EBM confidence & calibration; symplectic/squeeze transforms in ML; hierarchical/escalating memory), plus the associative-recall task-spec sweep (Zoology/MQAR fetched in full HTML; Mamba synthetics; H3/Hyena single-query AR).
- Fetched primary sources for the highest-risk neighbors: EBT (arXiv:2507.02092 abstract), Zoology MQAR formal spec (arXiv HTML), CALM confidence measures, Liu et al. energy-OOD, SympNets, Neural Canonical Transformations, LBN, L-GATr, Titans surprise metric, JEM.

How I verified: every verdict below is grounded in fetched abstracts/HTML or corroborating multi-hit search results; from-memory arXiv IDs are explicitly flagged in the bibtex block. "OPEN" claims are absence-of-evidence after targeted queries — residual risk noted in Confidence & gaps.

---

## Answer first

The *composite mechanism* — residual-energy-of-a-relaxed-symplectic-state as a trained confidence signal, gating structure-preserving Sp(2d) squeeze retries, escalating to heavier/deeper shells — is **not taken anywhere we could find**. But every *component genre* is crowded, and one 2025 paper (**Energy-Based Transformers, arXiv:2507.02092**) already owns the headline "energy as a verifier for inference-time System-2 compute." V1 is publishable as open territory **only if framed as**: (a) escalation across an *architectural* hierarchy (boost → shell), not just more gradient steps; (b) retries *certified* structure-preserving (det J = 1, phase-volume conservation — "the retry that can't destabilize"); (c) calibration trained as an EBM margin objective on *relaxation residuals* of a conservative system. EBT must be cited and positioned in the first paragraph of related work, not buried.

## (1) Verdict table

| Front | Verdict | Closest neighbors | One-line distinction |
|---|---|---|---|
| 1. Adaptive compute / early exit / cascades | **PARTIALLY TAKEN** (genre saturated; energy-gated *architectural* escalation open) | ACT (Graves 2016, arXiv:1603.08983, learned halting unit); PonderNet (Banino et al. 2021, halting distribution); BranchyNet (entropy threshold, ICPR 2016); PABEE (Zhou et al. 2020, patience = consecutive-agreement); CALM (Schuster et al., NeurIPS 2022, arXiv:2207.07061 — softmax response / hidden-state saturation / learned exit classifier, with sequence-level statistical calibration); Mixture-of-Depths (Raposo et al. 2024, arXiv:2404.02258, learned top-k router); speculative decoding (Leviathan et al., ICML 2023 — target-model likelihood-ratio verification); LLM cascades: FrugalGPT (Chen et al. 2023, learned scorer), GATEKEEPER (arXiv:2502.19335, confidence tuning), UCCI (arXiv:2605.18796, calibrated uncertainty → escalation threshold via constrained cost minimization), Agreement-Based Cascading (arXiv:2407.02348) | All gates are softmax/entropy/agreement/learned-classifier signals bolted onto feedforward stacks; **none uses a physical energy residual native to the model's own dynamics**, and none has structure-preservation guarantees on the retry action. EBT (front 2) is the only energy-signal entrant, and it escalates *gradient steps*, not *architecture tiers*. |
| 2. Energy-based confidence & calibration | **PARTIALLY TAKEN** (energy-as-confidence established; trained-threshold-on-relaxation-residual open) | Liu et al. (NeurIPS 2020, arXiv:2010.03759): energy score > softmax for OOD, −18.03% avg FPR@95 on CIFAR-10 WideResNet; also usable as *trainable* cost shaping the energy surface — closest precedent for "calibration in the training objective." JEM (Grathwohl et al., ICLR 2020, arXiv:1912.03263): EBM training of p(x,y) improves calibration + OOD. LeCun et al. (2006) EBM tutorial: margin losses are textbook. EBT (Gladstone et al. 2025, arXiv:2507.02092): energy = learned verifier of input–prediction compatibility, any modality. Conformal+energy exists only in scattered applied work (no canonical paper found). | Nobody claims: *jointly-trained margin + learned threshold τ such that the residual energy of a settled dynamical state ranks answer correctness and generalizes off-distribution as a retrieval-quality gate*. Warning: the margin loss itself is 20-year-old EBM machinery — novelty must be claimed on the *dynamical residual + gate use*, never on the loss form. |
| 3. Squeeze / symplectic transforms in ML | **components TAKEN, mechanism OPEN** | SympNets (Jin et al., Neural Networks 2020, arXiv:2001.03750): universal approximation of symplectic maps incl. linear symplectic (squeeze) modules — proves our transform *family* is standard. Neural Canonical Transformations (Li et al., PRX 10, 021020 (2020), arXiv:1910.00024): learned symplectic flows separating slow/fast collective modes — spiritually near Thread 5's mass hierarchy. Sp(n)-equivariant layers characterized (Brauer-algebra line, arXiv:2212.08630). **Lorentz Boost Network** (Erdmann et al., JINST 14 P06006 2019, arXiv:1812.09722): learns boosts of four-vectors into learned rest frames as *trainable feature engineering* — the closest "boost as learned reframing" prior. LorentzNet (Gong et al. 2022); L-GATr (Spinner et al., NeurIPS 2024, arXiv:2405.14806): Lorentz-equivariant transformer, HEP tasks. | All of these use symplectic/Lorentz structure to *parameterize the model or its features at training time*. **No paper found using inference-time structure-preserving transforms (squeezes, boosts) as energy-guided retrieval retries on a frozen model** — L0's "zero new params, line-search over rapidity ζ" appears genuinely unclaimed. |
| 4. Hierarchical memory / escalating retrieval | **PARTIALLY TAKEN** | Modern Hopfield (Ramsauer et al., ICLR 2021): retrieval = energy descent, β knob exists but is tuned, not an escalation controller. Krotov, Hierarchical Associative Memory (2021, arXiv:2107.06446): multi-layer energy AM — the natural "shell stack" precedent, but retrieval is monolithic (one global energy descent, no gate, no cheap-first). Energy Transformer (Hoover et al. 2023). Fast Weight Programmers (Schlag et al., ICML 2021); Clockwork RNN (Koutník et al. 2014); HM-RNN (Chung et al. 2017) — multi-timescale but no gated escalation. **Titans** (Behrouz et al. 2025, arXiv:2501.00663): test-time memory gated by *surprise* (= gradient of loss) + momentum + adaptive forgetting. Retrieval-LM gates: SPALM (Yogatama et al. 2021, arXiv:2102.02557, learned gating of parametric-vs-retrieved), AdaptRet/kNN-LM "when to retrieve" (He et al. 2021; Mallen et al. 2022) — genuine "cheap store → gate → expensive store" precedent, but gate = learned classifier on query features. | The "cheap→expensive with a gate" *pattern* exists (retrieval-LM line); the *principled physical gate* does not: no one gates escalation on height-above-learned-energy-floor in an attractor system, and no associative-memory paper does escalate-on-failed-relaxation. Hopfield-line papers are our mandatory baseline, not our scoop risk. |

## (2) Five most dangerous "reviewer will say you're X" papers

1. **Gladstone et al. (2025), "Energy-Based Transformers are Scalable Learners and Thinkers," arXiv:2507.02092.** *"You're EBT with a Hamiltonian coat of paint: energy verifier + inference-time compute."* Honest differentiation: EBT's "thinking" = unconstrained gradient descent on the prediction in a transformer's learned energy; more compute = more steps of the *same* operation. Ours: (i) retries are elements of Sp(2d) — det J = 1, provably non-destabilizing, an action class EBT has no analog of; (ii) escalation is *architectural* (light→heavy sectors, small→large shells priced by mass), not iterative; (iii) our energy is a Hamiltonian with conservation semantics — residual energy has a floor interpretation (governor/target-energy machinery), not just an unnormalized score. **Risk level: highest; cite prominently.** Preprint, ~10 authors; venue status unverified as of 2026-07 — check before citing as peer-reviewed.
2. **Schuster et al. (2022), "Confident Adaptive Language Modeling," NeurIPS 2022, arXiv:2207.07061.** *"Your 'trained calibration of the exit threshold' is CALM."* Differentiation: CALM calibrates softmax/saturation/classifier confidences *post-hoc* via Learn-then-Test to guarantee sequence-level quality; our τ is co-trained inside the contrastive objective and reads a physical scalar. **Borrow, don't fight:** CALM's distribution-free calibration wrapper is exactly what we should layer on top of our trained τ for deployment guarantees — that combination would be a strength, and reviewers will expect the comparison.
3. **Behrouz et al. (2025), "Titans: Learning to Memorize at Test Time," arXiv:2501.00663.** *"Gated multi-store memory with a principled signal already exists — surprise."* Differentiation: Titans' surprise = loss-gradient magnitude (a heuristic on the optimizer), decides *what to store*; our residual energy decides *whether the answer is good and how much more compute to buy* — write-gating vs read-escalation. Titans has no retry mechanism and no structure guarantees. Still: reviewers will conflate "physically-flavored memory signal" — pre-empt with one paragraph.
4. **Erdmann et al. (2019), "Lorentz Boost Networks," JINST 14 P06006, arXiv:1812.09722** (with L-GATr/LorentzNet as the modern flank). *"Learned Lorentz boosts in ML are old news."* Differentiation: LBN learns boosts of *input four-vectors* into rest frames as feature engineering, fixed after training; ours are inference-time actions on *latent phase space* selected per-query by energy descent, with the mass matrix making a single global rapidity region-differential for free (L0). We must say this explicitly or HEP-literate reviewers (our own community!) will file us under LBN.
5. **Ramsauer et al. (2021) + Krotov (2021) + Hoover et al. (2023)** (modern Hopfield / Hierarchical AM / Energy Transformer). *"Energy-descent associative retrieval in a hierarchy = solved."* Differentiation: their dynamics are dissipative descent to fixed points with no confidence output, no failure detection, no escalation; a hierarchical Hopfield retrieves through all layers every time. **Warning: this is also our weakest differentiation empirically** until we show the gate + escalation beats "just run the big Hopfield" on a compute-matched curve — a modern-Hopfield baseline is non-optional in the gate experiment.

Lineage citations reviewers will expect regardless: ACT, PonderNet, BranchyNet, PABEE, MoD, speculative decoding, FrugalGPT/cascade-survey (arXiv:2603.04445).

## (3) Associative-recall / selective-copy task specs (verified from primary sources)

**MQAR** (Arora et al. 2023, "Zoology," arXiv:2312.04927 — spec extracted from paper HTML):
- Input `x = {x₀,…,x_{N−1}}`, tokens from vocab C, c = |C| = **8192** in main experiments. For every position i: if ∃ j < i with token match (key reoccurs), output the token that followed the first occurrence (`u_{j+1}`). Multiple queries per sequence, at varying positions (vs one fixed-position query in older AR).
- Sweeps: sequence length N ∈ **{64…512}**; model dim d swept — headline result: attention solves at constant **d = 64**; gated convolutions require **d ≥ N**. Two-layer models (seq-mixer + MLP), 4 LRs swept log-uniform 1e-4→1e-2. Metric: held-out accuracy.
- Difficulty knobs: N, vocab size, #KV pairs, key–query gap distribution (follow-up work "Based" uses power-law gaps), model width d.

**Single-query AR** (H3/Hyena lineage, as described in Zoology): sequence of key-value bigrams from a random dictionary, vocab < 50, single query token at sequence end.

**Selective copying** (Gu & Dao 2023, Mamba, arXiv:2312.00752): copy content tokens whose positions are randomized among noise/blank tokens — defeats time-invariant (LTI) conv/SSM models; solved by input-dependent (selective) gating.

**Induction head** (same paper): trigger token appears twice (middle + end); on second occurrence recall the token that followed the first; Mamba shows length extrapolation far beyond training length. (Exact Mamba table values — seq 4096 / 16 content tokens for selective copy; train length 256, test 2⁶–2²⁰ for induction — are from memory; verify against Mamba Table 1/2 before quoting in a paper.)

**Recommendation for our gate experiment:** use MQAR with (N, #KV, vocab, gap) knobs as the *difficulty axis*; report residual-energy-vs-correctness AUROC per difficulty level, plus a compute-matched escalation-cost curve vs (i) always-small, (ii) always-large, (iii) modern-Hopfield retrieval, (iv) entropy-gated cascade. That makes us directly legible to the SSM/attention synthetic-eval community.

## (4) Bibtex-ready refs

```bibtex
@article{raposo2024mixture,  title={Mixture-of-Depths: Dynamically allocating compute in transformer-based language models},  author={Raposo, David and Ritter, Sam and Richards, Blake and Lillicrap, Timothy and Humphreys, Peter Conway and Santoro, Adam},  journal={arXiv:2404.02258}, year={2024}}
@inproceedings{schuster2022calm,  title={Confident Adaptive Language Modeling},  author={Schuster, Tal and Fisch, Adam and Gupta, Jai and Dehghani, Mostafa and Bahri, Dara and Tran, Vinh Q. and Tay, Yi and Metzler, Donald},  booktitle={NeurIPS}, year={2022}, note={arXiv:2207.07061}}
@inproceedings{liu2020energy,  title={Energy-based Out-of-distribution Detection},  author={Liu, Weitang and Wang, Xiaoyun and Owens, John D. and Li, Yixuan},  booktitle={NeurIPS}, year={2020}, note={arXiv:2010.03759}}
@inproceedings{grathwohl2020jem,  title={Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One},  author={Grathwohl, Will and Wang, Kuan-Chieh and Jacobsen, J{\"o}rn-Henrik and Duvenaud, David and Norouzi, Mohammad and Swersky, Kevin},  booktitle={ICLR}, year={2020}, note={arXiv:1912.03263}}
@article{gladstone2025ebt,  title={Energy-Based Transformers are Scalable Learners and Thinkers},  author={Gladstone, Alexi and others},  journal={arXiv:2507.02092}, year={2025}, note={preprint; venue status unverified 2026-07}}
@article{jin2020sympnets,  title={SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems},  author={Jin, Pengzhan and Zhang, Zhen and Zhu, Aiqing and Tang, Yifa and Karniadakis, George Em},  journal={Neural Networks}, volume={132}, pages={166--179}, year={2020}, note={arXiv:2001.03750}}
@article{li2020neuralct,  title={Neural Canonical Transformation with Symplectic Flows},  author={Li, Shuo-Hui and Dong, Chen-Xiao and Zhang, Linfeng and Wang, Lei},  journal={Physical Review X}, volume={10}, pages={021020}, year={2020}, note={arXiv:1910.00024}}
@article{erdmann2019lbn,  title={Lorentz Boost Networks: autonomous physics-inspired feature engineering},  author={Erdmann, Martin and Geiser, Erik and Rath, Yannik and Rieger, Marcel},  journal={JINST}, volume={14}, pages={P06006}, year={2019}, note={arXiv:1812.09722}}
@inproceedings{spinner2024lgatr,  title={Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics},  author={Spinner, Jonas and Bres{\'o}, V{\'i}ctor and de Haan, Pim and Plehn, Tilman and Thaler, Jesse and Brehmer, Johann},  booktitle={NeurIPS}, year={2024}, note={arXiv:2405.14806}}
@article{krotov2021hierarchical,  title={Hierarchical Associative Memory},  author={Krotov, Dmitry},  journal={arXiv:2107.06446}, year={2021}}
@article{behrouz2025titans,  title={Titans: Learning to Memorize at Test Time},  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},  journal={arXiv:2501.00663}, year={2025}}
@article{arora2023zoology,  title={Zoology: Measuring and Improving Recall in Efficient Language Models},  author={Arora, Simran and others},  journal={arXiv:2312.04927}, year={2023}}
@article{gu2023mamba,  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},  author={Gu, Albert and Dao, Tri},  journal={arXiv:2312.00752}, year={2023}}
@article{yogatama2021spalm,  title={Adaptive Semiparametric Language Models},  author={Yogatama, Dani and de Masson d'Autume, Cyprien and Kong, Lingpeng},  journal={TACL}, year={2021}, note={arXiv:2102.02557}}
% From memory, IDs high-confidence but NOT re-fetched this session — verify before camera-ready:
% Graves ACT arXiv:1603.08983 · PonderNet arXiv:2107.05407 · PABEE arXiv:2006.04152 · Leviathan spec-decoding arXiv:2211.17192 · FrugalGPT arXiv:2305.05176 · Ramsauer Hopfield arXiv:2008.02217 · Hoover Energy Transformer arXiv:2302.07253 · LorentzNet arXiv:2201.08187 · Schlag FWP arXiv:2102.11174 · Clockwork RNN arXiv:1402.3511 · BranchyNet arXiv:1709.01686
```

## Confidence & gaps

- **Verified (primary source fetched):** EBT mechanism scope (abstract-level; §"thinking" internals not deep-read — worth a full-paper pass before the V1 short), MQAR full spec, CALM confidence measures, Liu energy-OOD numbers, SympNets/NeuralCT/LBN/L-GATr existence+scope, Titans surprise definition, JEM calibration claims.
- **Single-sourced / absence-of-evidence:** "no inference-time symplectic retries" and "no energy-floor escalation gate" are *not-found* claims after targeted queries — solid but not exhaustive; a Semantic-Scholar citation-graph crawl of EBT's 2025–26 citers is the cheapest way to close the gap (EBT spawned a fast-moving follow-up literature; someone may have added compute-tiering already).
- **Search next:** (i) EBT citers 2025–26 ("energy verifier" + "cascade"); (ii) "Based"/post-Zoology MQAR variants for the current canonical difficulty knobs; (iii) exact Mamba synthetic-task table values; (iv) test-time-training literature (Sun et al. TTT line) as a fifth neighbor family not swept this session.

## Open questions / follow-ups / risks

- EBT full-paper deep-read before V1 short freeze (does their §System-2 include any *energy-thresholded early stopping*? If yes, our claim (c) narrows to "trained margin+τ on relaxation residuals of a conservative system").
- One search family not swept: quantum-ML squeezing operators used classically (photonic-computing line) — low risk, but a 30-min check would close front 3 completely.
- Modern-Hopfield baseline in the gate experiment is non-optional (dangerous-neighbor #5).

## Proposed handover updates (for the Hub)

- §8/roadmap V1: record verdict — **V1 territory open conditional on EBT positioning**; add EBT (arXiv:2507.02092) as V1's primary related-work anchor and Aug-7-gate framing constraint ("architectural escalation + certified retries + trained residual calibration" = the three claims EBT doesn't own).
- V1 gate experiment: adopt MQAR (vocab 8192, N 64–512, 2-layer, accuracy) as the standard-legible task; mandatory baselines = modern-Hopfield retrieval + entropy-gated cascade; borrow CALM's Learn-then-Test wrapper on top of trained τ as a deployment-guarantee bonus.
- New risk-register entry: EBT follow-up literature is hot; re-scout its citation graph ~2 weeks before the V1 short freezes.
- LBN (Erdmann et al. 2019) added to the differentiation ledger alongside LorentzNet: HEP reviewers will reach for it first on any "learned boost" claim.

## Sources (fetched/corroborated this session)

- EBT: https://arxiv.org/abs/2507.02092 · https://energy-based-transformers.github.io/
- Mixture-of-Depths: https://arxiv.org/abs/2404.02258
- CALM: https://arxiv.org/abs/2207.07061 · https://papers.neurips.cc/paper_files/paper/2022/file/6fac9e316a4ae75ea244ddcef1982c71-Paper-Conference.pdf
- Liu energy-OOD: https://arxiv.org/abs/2010.03759
- JEM: https://openreview.net/forum?id=Hkxzx0NtDB
- Zoology/MQAR: https://arxiv.org/abs/2312.04927 · https://arxiv.org/html/2312.04927v1
- Mamba: https://arxiv.org/pdf/2312.00752
- Titans: https://arxiv.org/abs/2501.00663
- SympNets: https://arxiv.org/abs/2001.03750
- Neural Canonical Transformation: https://arxiv.org/abs/1910.00024 · https://dx.doi.org/10.1103/PhysRevX.10.021020
- LBN: https://ar5iv.labs.arxiv.org/html/1812.09722 · https://iopscience.iop.org/article/10.1088/1748-0221/14/06/P06006
- L-GATr: https://arxiv.org/abs/2405.14806
- Krotov HAM: https://arxiv.org/pdf/2107.06446
- SPALM: https://arxiv.org/pdf/2102.02557 · kNN-LM reliance: https://arxiv.org/pdf/2210.15859
- Cascades: UCCI https://arxiv.org/html/2605.18796 · GATEKEEPER https://arxiv.org/pdf/2502.19335 · survey https://arxiv.org/html/2603.04445v1 · Agreement-Based https://arxiv.org/pdf/2407.02348 · Cluster-Route-Escalate https://arxiv.org/html/2606.27457
- Sp(n)-equivariant: https://arxiv.org/abs/2212.08630
- Early-exit survey: https://dl.acm.org/doi/full/10.1145/3698767
- Modern Hopfield flank: https://openreview.net/pdf?id=zwqlV7HoaT · https://arxiv.org/html/2502.10122v1
