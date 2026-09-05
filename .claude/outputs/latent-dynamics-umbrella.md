# latent-dynamics-umbrella — web-scout report

**Task + acceptance criterion:** Head question (no task file on disk; slug chosen by me): *are there other algorithms that "do latent dynamics", and does that class already have an umbrella term?* Deliver the named families, their canonical terms, and CLU's position within/against them.
**Status:** done
**Dial:** none — literature recon / positioning. No claim, no laundering control, nothing falsifiable here.
**Interpretation chosen:** "latent dynamics" = *the model carries an internal state that is evolved by a learned (or fixed) dynamical law, and the useful content lives in that state's trajectory.* I read the question as a **naming/positioning** question for the paper, not a benchmark question.

⚠ **Reconciliation list for the Hub (in first 10 lines, per §5):** two naming hazards need an owner — (1) **"LDM" is taken** (Latent Diffusion Model) and is also already claimed in ROM (Farenga et al. 2024) and RL ("latent dynamics model" ≡ world model); (2) the phrase *"CLU is a latent dynamics system"* maps a reader onto **family A (model-of-the-data's-dynamics)**, which the standing memory rule explicitly forbids as CLU's framing. Recommend a curator pass over any draft text using "latent dynamics" unqualified.

---

## Answer first

Yes — a very large amount of prior art "does latent dynamics", but it splits into **two families that share the phrase and almost nothing else**, and **only family A has settled umbrella terms**. **Family A = latent dynamics as a *model of the observed system's dynamics*** (world models / DVAEs / latent ODE-SDE / Koopman / latent-space ROM); its accepted umbrella terms are, by community: **"dynamical variational autoencoders (DVAE)"** (Girin et al. 2021), **"world models" / "latent dynamics model"** (RL), **"latent dynamics models (LDMs)" / "latent space dynamics identification (LaSDI)"** (reduced-order modeling), and **"latent variable models / latent neural dynamics"** (neuroscience). **Family B = latent dynamics as a *computational carrier*** — the state is not a model of anything, it is where information is held and transported (deep SSMs/S4/Mamba, HiPPO/LMU, reservoir computing, continuous-depth nets, DEQ, symplectic/reversible nets, test-time-memory layers). **Family B has no single umbrella term**; the closest are "**structured state-space models**", "**continuous-depth models**", and the newest and most unifying, "**test-time regression / associative-memory sequence layers**" (Wang, Shi & Fox 2025).

**CLU is family B wearing family-A clothing.** That is precisely the positioning risk: the physics vocabulary (Hamiltonian, Verlet, Langevin) reads as family A to a referee, while the actual claim (a latent information carrier, peer to MLP/Mamba/attention) is family B. Every rival on the C3 ladder — Mamba-2, GDN-2, TTT-Linear, sliding-window — is family B, which confirms the program is already competing in the right family; the *language* is what lags.

---

## Evidence

### Family A — latent dynamics as a model of the data's dynamics (umbrella terms exist)

- **DVAE — the most formal umbrella.** Girin, Leglaive, Bie, Diard, Hueber, Alameda-Pineda (2021), *"Dynamical Variational Autoencoders: A Comprehensive Review"*, **Foundations and Trends in Machine Learning 15(1–2):1–175** (peer-reviewed monograph; preprint arXiv:2008.12595). It explicitly coins a *class*: "a general class of models called dynamical VAEs (DVAEs) that encompasses a large subset of temporal VAE extensions", and unifies seven models (DKF, STORN, VRNN, SRNN, RVAE, DSAE…) under one notation. This is the single most defensible "umbrella term" answer for the probabilistic sequential-latent family. Members it subsumes include **Deep Kalman Filters** (Krishnan, Shalit, Sontag, arXiv:1511.05121), **Deep Variational Bayes Filters** (Karl, Soelch, Bayer, van der Smagt, ICLR 2017, arXiv:1605.06432 — subtitle literally *"Unsupervised Learning of State Space Models from Raw Data"*), and **E2C** (Watter et al., NeurIPS 2015, arXiv:1506.07365).
- **RL / robotics: "world model" and "latent dynamics model" are used interchangeably.** Ha & Schmidhuber (2018), *"World Models"*, arXiv:1803.10122; Hafner et al. (2019), *"Learning Latent Dynamics for Planning from Pixels"* (PlaNet, ICML), arXiv:1811.04551 — the **RSSM** is described in the follow-on literature as "a latent dynamics model equipped with an expressive recurrent neural network"; Hafner et al. (2020), *"Dream to Control: Learning Behaviors by Latent Imagination"* (Dreamer, ICLR), arXiv:1912.01603. ⚠ In this community **"latent dynamics model" is already a taken, specific term** meaning *a learned surrogate of the environment*.
- **Continuous-time generative latents.** Chen, Rubanova, Bettencourt, Duvenaud (2018), *"Neural Ordinary Differential Equations"*, NeurIPS, arXiv:1806.07366; Rubanova, Chen, Duvenaud (2019), *"Latent ODEs for Irregularly-Sampled Time Series"*, NeurIPS, arXiv:1907.03907; Li, Wong, Chen, Duvenaud (2020), *"Scalable Gradients for Stochastic Differential Equations"*, AISTATS, arXiv:2001.01328 — verified: it does "gradient-based stochastic variational inference for **latent stochastic differential equations**".
- **Scientific ML / reduced-order modeling — two competing umbrellas, both explicit.**
  - **"Latent dynamics models (LDMs)"**: Farenga, Fresca, Brivio, Manzoni (2024/2025), *"On latent dynamics learning in nonlinear reduced order modeling"*, **Neural Networks** (publisher version S0893608025000255; preprint arXiv:2408.15183). Verbatim: *"we present the novel mathematical framework of latent dynamics models (LDMs) for reduced order modeling of parameterized nonlinear time-dependent PDEs … while constraining the latent state to evolve accordingly to an (unknown) dynamical system."* ⚠ Caveat: the fetched abstract calls the framework "novel", i.e. it is a **claimed coinage**, not a settled community-wide term. Earlier related coinage: Farenga, Fresca, Manzoni, *"Neural Latent Dynamics Models"* (NLDM), NeurIPS 2022 (workshop track; OpenReview `Yk_I37Ca8Q` — ⚠ the OpenReview PDF was behind a bot-verification wall, so venue-track is **single-sourced** from search metadata).
  - **"Latent Space Dynamics Identification (LaSDI)"**: Fries, He, Choi (2022), *"LaSDI: Parametric Latent Space Dynamics Identification"*, **CMAME**; review: Bonneville, He, Tran, Park, Fries, Messenger, Cheung, Shin, Bortz, Ghosh, Chen, Belof, Choi (2024), *"A Comprehensive Review of Latent Space Dynamics Identification Algorithms for Intrusive and Non-Intrusive Reduced-Order-Modeling"*, arXiv:2403.10748 — a whole named family (gLaSDI, WLaSDI, tLaSDI, GPLaSDI).
- **Operator-theoretic branch.** Lusch, Kutz, Brunton (2018), *"Deep learning for universal linear embeddings of nonlinear dynamics"*, **Nature Communications 9:4950** — the latent space is chosen so the dynamics become *linear* (Koopman/DMD). Umbrella term: "Koopman / operator-theoretic learning".
- **Neuroscience.** Pandarinath et al. (2018), *"Inferring single-trial neural population dynamics using sequential auto-encoders"* (LFADS), **Nature Methods 15:805–815**; Vyas, Golub, Sussillo, Shenoy (2020), *"Computation Through Neural Population Dynamics"*, **Annual Review of Neuroscience 43:249–275**, doi:10.1146/annurev-neuro-092619-094115. Current survey and its umbrella: Kong et al. (2026), *"Machine Learning Methods for Studying Latent Neural Activity Dynamics"*, arXiv:2606.10530 (submitted 2026-06-09) — verbatim: *"we provide a comprehensive survey that outlines the trajectory of **Latent Variable Models (LVMs)** from early state-space models to more recent deep generative models"*, organized as *"(1) Single-Region **Latent Dynamics** … (2) Multi-Region Communication … (3) Behavior-Aligned Modeling."* ⇒ in neuroscience the umbrella noun is **LVM**; "latent dynamics" is a *subdomain label*.

### Family B — latent dynamics as computational carrier (no single umbrella)

- **Structured state-space models.** Gu, Goel, Ré (2022), *"Efficiently Modeling Long Sequences with Structured State Spaces"* (S4), ICLR, arXiv:2111.00396; Gu & Dao (2023), *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*, arXiv:2312.00752. Survey/umbrella: *"From S4 to Mamba: A Comprehensive Survey on Structured State Space Models"*, arXiv:2503.18970 (2025). The latent `h_t` here is **not** a model of the data's generating process — it is a compressed carrier of history.
- **Memory-as-dynamics, derived from first principles.** Gu, Dao, Ermon, Rudra, Ré (2020), *"HiPPO: Recurrent Memory with Optimal Polynomial Projections"*, NeurIPS, arXiv:2008.07669 — the latent ODE `d/dt c(t) = A(t)c(t) + B(t)f(t)` *is* the memory; it "yields a short derivation of the recent **Legendre Memory Unit (LMU)** from first principles" (Voelker, Kajić, Eliasmith, NeurIPS 2019). ⭐ This is the closest published *philosophical* precedent for CLU's "latent state as information carrier, optimality of the carrier is the claim".
- **Reservoir computing** — the purest family-B statement: an *untrained* latent dynamical system whose trajectory carries the computation, with only a readout learned. Lukoševičius & Jaeger (2009), *"Reservoir computing approaches to recurrent neural network training"*, **Computer Science Review 3(3):127–149**, doi:10.1016/j.cosrev.2009.03.005.
- **Continuous-depth / dynamical-systems view of architectures.** Massaroli, Poli, Park, Yamashita, Asama (2020), *"Dissecting Neural ODEs"*, NeurIPS, arXiv:2002.08071 — uses **"continuous-depth models"** as the class noun. Related stability lineage that CLU is a direct descendant of: Haber & Ruthotto (2017), *"Stable architectures for deep neural networks"*, **Inverse Problems 34:014004**, arXiv:1705.03341 (Hamiltonian/antisymmetric forward propagation for well-posedness); Chen, Zhang, Arjovsky, Bottou (2020), *"Symplectic Recurrent Neural Networks"*, **ICLR 2020 (spotlight)**, arXiv:1909.13334 (verified title/authors/date); Greydanus, Dzamba, Yosinski (2019), *"Hamiltonian Neural Networks"*, NeurIPS, arXiv:1906.01563. ⚠ Note HNN/SRNN are **family A** in intent (they fit a physical system's Hamiltonian from trajectories) but **family B** in machinery — CLU inherits the machinery and rejects the intent.
- **Implicit/equilibrium models.** Bai, Kolter, Koltun (2019), *"Deep Equilibrium Models"*, NeurIPS, arXiv:1909.01377 — latent state defined by a fixed point rather than a trajectory; the "dynamics" is a solver.
- **⭐ The newest and best cross-cutting unifier for family B.** Wang, Shi, Fox (2025), *"Test-time regression: a unifying framework for designing sequence models with associative memory"*, arXiv:2501.12352 — verbatim: *"Prominent layers, including **linear attention, state-space models, fast-weight programmers, online learners, and softmax attention**, arise as special cases defined by three design choices: the regression weights, the regressor function class, and the test-time optimization algorithm."* This is the closest thing that exists to an umbrella covering "latent state that is *written into* at test time as a carrier of information" — i.e. **the family CLU actually belongs to.** Members: Sun et al. (2024), *"Learning to (Learn at Test Time): RNNs with Expressive Hidden States"* (TTT), arXiv:2407.04620; Yang, Kautz, Hatamizadeh (2024), *"Gated Delta Networks: Improving Mamba2 with Delta Rule"*, ICLR 2025, arXiv:2412.06464 (verified); Behrouz, Zhong, Mirrokni (2024), *"Titans: Learning to Memorize at Test Time"*, arXiv:2501.00663.

### Naming hazards (verified collisions)

- **"LDM" is overwhelmingly Latent Diffusion Model** — Rombach, Blattmann, Lorenz, Esser, Ommer (2022), *"High-Resolution Image Synthesis with Latent Diffusion Models"*, CVPR, arXiv:2112.10752. Do not adopt "LDM".
- **"Latent dynamics model" in RL = world model** (PlaNet/Dreamer lineage, above). Using it for CLU invites the reading *"CLU models the data's dynamics"* — the exact framing the program's standing rule forbids.
- **"Latent variable model"** is taken by the probabilistic/neuroscience sense (Kong et al. 2026), and by classical factor analysis.

---

## Relevance to CHLU/CLU

**1. The one-sentence positioning I'd recommend.** *"CLU is not a latent dynamics **model** — it is a latent dynamics **substrate**. Family A learns a dynamical law so the latent trajectory predicts the data; CLU fixes the dynamical law (relativistic Hamiltonian + symplectic Verlet) so the latent trajectory can **hold and transport** information under a conservation guarantee. The learned object is the potential/store, not the flow."* This distinction is the cleanest available novelty axis and it is **not** occupied by any umbrella term found.

**2. Novelty vs. prior — where CLU is genuinely uncrowded.**
- Family B members almost universally use **contractive/dissipative or projection-optimal** state updates (HiPPO's `A` is stable-by-construction; SSMs use `exp(ΔA)` with negative real parts; reservoirs need the echo-state property, i.e. *forgetting*). CLU's **volume-preserving, γ-tunable** carrier is the differentiator: forgetting is a *dial*, not a stability prerequisite. Frame against **HiPPO/LMU** specifically — they are the strongest prior claim to "the latent ODE *is* the memory", and they get their memory from an *optimal projection*, not from a *conservation law*.
- The **symplectic** lineage (Haber–Ruthotto, SRNN, HNN) is family-A-in-intent; nobody in that line has claimed the state as a *general-purpose addressable carrier* on non-physics data. That gap is exactly the C3 program.
- The **wake–sleep/EBM training + Langevin generation** half has *no counterpart at all* in either family's umbrella — DVAEs use amortized VI, SSMs use plain backprop. This is CLU's least-crowded component and the most likely to read as unfamiliar (⇒ needs the most exposition, not the least).

**3. What to borrow.**
- **Borrow the DVAE playbook rhetorically** (Girin et al.): a review that *defines a class and re-derives seven models in one notation* is how a family gets named. If CLU wants a family, the paper must name the axis (conservative vs. dissipative carriers), not just the unit.
- **Borrow Wang–Shi–Fox's three-design-choice frame** (regression weights / regressor class / test-time optimizer). It is the current lingua franca for exactly the rivals on the C3 ladder, and CLU can be *stated inside it* — "our test-time optimizer is symplectic Verlet on a learned Hamiltonian; our regressor class is the potential `V_θ`" — which converts CLU from an outsider into a labelled point in a framework referees already accept. ⭐ Strong recommendation: this is the cheapest legibility win available.
- **Borrow the ROM community's error/stability estimate posture** (Farenga et al. 2024 derive bounded approximation error for the latent flow). If CLU ever wants a theorem about the carrier, that literature has the template.

**4. What to differentiate from — the traps.**
- ⛔ Do **not** benchmark CLU against family A on family-A metrics (rollout MSE on a physical system). That is the "metric-native ceiling" pattern: a Koopman/LaSDI method built to model *that* PDE will win, and it isn't news. The C3 charter's venue logic (CAMELS, N-CMAPSS as *application*, enwik8 as primary) already respects this — the framing here just supplies the *reason*.
- ⛔ Do not let the paper's physics vocabulary put a referee into family A. Recommend an explicit early sentence: *"we do not claim CLU models the physics of the data."*

**5. Answer to the literal question.** The class exists; it does **not** have one umbrella term. It has **four community-local ones** (DVAE · world model · LaSDI/LDM · LVM) for family A, and for family B — CLU's family — the best available are **"structured state-space models"** (architecture-flavoured), **"continuous-depth models"** (analysis-flavoured), and **"test-time regression / associative memory layers"** (the current unifier). If the program wants a term, the *unoccupied* one is something like **"conservative latent carriers"** — no search hit found for that or a synonym.

---

## Confidence & gaps

- **Verified (fetched primary source or two independent hits):** DVAE monograph + FnTML venue; Farenga et al. LDM abstract verbatim + Neural Networks publication; Kong et al. 2026 abstract verbatim; LaSDI review authorship/date; SRNN title/authors/date; latent-SDE, DVBF, Gated DeltaNet titles/authors; HiPPO↔LMU derivation claim (quoted from the paper's own abstract); Wang–Shi–Fox unifying-scope quote; Vyas et al. journal/volume/pages/DOI; Lukoševičius–Jaeger journal/pages/DOI.
- **Single-sourced / not fetched (IDs from my own knowledge, corroborated only by search snippets — Hub should re-check before they enter a paper):** Ha & Schmidhuber 1803.10122; PlaNet 1811.04551; Dreamer 1912.01603; Neural ODE 1806.07366; Latent ODE 1907.03907; DKF 1511.05121; E2C 1506.07365; HNN 1906.01563; S4 2111.00396; Mamba 2312.00752; DEQ 1909.01377; Haber–Ruthotto 1705.03341; Titans 2501.00663; TTT 2407.04620; Rombach 2112.10752; Lusch et al. Nat. Comms. 9:4950; Pandarinath Nat. Methods 15:805.
- **⚠ Could not verify:** the NLDM (`Yk_I37Ca8Q`) **workshop track** at NeurIPS 2022 — OpenReview served a bot-check page. Venue is search-metadata only.
- **Negative claim, stated honestly:** *"no single accepted umbrella term spans both families"* is **my assessment from ~10 targeted searches**, not an exhaustive proof. I found no survey that unifies world models + deep SSMs + reservoirs under one name. Absence of evidence.
- **Not searched (next queries if the Hub wants depth):** (a) does any deep-SSM paper explicitly self-describe as "latent dynamics"? (b) "physical reservoir computing" (Tanaka et al. 2019) as an even closer family-B analogue; (c) whether **Liquid Time-Constant Networks / CfC** (Hasani et al.) claim a carrier framing; (d) prior use of "conservative memory"/"volume-preserving memory" as a term-of-art in RNN stability work (unitary/orthogonal RNNs — Arjovsky uRNN, Lezcano-Casado expRNN — which are arguably the *true* nearest neighbours to "conservation as a memory guarantee" and were **not** covered in this pass). ⭐ (d) is the most important gap: unitary/orthogonal RNNs are the one prior family that already sells norm-preservation as long-horizon memory, and CLU must have an answer to them.

---

## Bibtex-ready refs

```bibtex
@article{girin2021dvae,
  title={Dynamical Variational Autoencoders: A Comprehensive Review},
  author={Girin, Laurent and Leglaive, Simon and Bie, Xiaoyu and Diard, Julien and Hueber, Thomas and Alameda-Pineda, Xavier},
  journal={Foundations and Trends in Machine Learning}, volume={15}, number={1--2}, pages={1--175}, year={2021},
  doi={10.1561/2200000089}, note={arXiv:2008.12595}}

@article{farenga2025ldm,
  title={On latent dynamics learning in nonlinear reduced order modeling},
  author={Farenga, Nicola and Fresca, Stefania and Brivio, Simone and Manzoni, Andrea},
  journal={Neural Networks}, year={2025}, note={arXiv:2408.15183}}

@article{bonneville2024lasdi,
  title={A Comprehensive Review of Latent Space Dynamics Identification Algorithms for Intrusive and Non-Intrusive Reduced-Order-Modeling},
  author={Bonneville, Christophe and He, Xiaolong and Tran, April and Park, Jun Sur and Fries, William and Messenger, Daniel A. and Cheung, Siu Wun and Shin, Yeonjong and Bortz, David M. and Ghosh, Debojyoti and Chen, Jiun-Shyan and Belof, Jonathan and Choi, Youngsoo},
  journal={arXiv preprint arXiv:2403.10748}, year={2024}}

@article{kong2026lvm,
  title={Machine Learning Methods for Studying Latent Neural Activity Dynamics},
  author={Kong, Shufeng and Deng, Fumei and Dong, Xinyi and Liu, Caihua and Chen, Weiwei and Wang, Yingheng and Cao, Daniel and Oliva, Azahara and Fernandez-Ruiz, Antonio and Gomes, Carla},
  journal={arXiv preprint arXiv:2606.10530}, year={2026}}

@article{vyas2020computation,
  title={Computation Through Neural Population Dynamics},
  author={Vyas, Saurabh and Golub, Matthew D. and Sussillo, David and Shenoy, Krishna V.},
  journal={Annual Review of Neuroscience}, volume={43}, pages={249--275}, year={2020},
  doi={10.1146/annurev-neuro-092619-094115}}

@inproceedings{gu2020hippo,
  title={HiPPO: Recurrent Memory with Optimal Polynomial Projections},
  author={Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={NeurIPS}, year={2020}, note={arXiv:2008.07669}}

@article{lukosevicius2009reservoir,
  title={Reservoir computing approaches to recurrent neural network training},
  author={Luko{\v{s}}evi{\v{c}}ius, Mantas and Jaeger, Herbert},
  journal={Computer Science Review}, volume={3}, number={3}, pages={127--149}, year={2009},
  doi={10.1016/j.cosrev.2009.03.005}}

@inproceedings{massaroli2020dissecting,
  title={Dissecting Neural ODEs},
  author={Massaroli, Stefano and Poli, Michael and Park, Jinkyoo and Yamashita, Atsushi and Asama, Hajime},
  booktitle={NeurIPS}, year={2020}, note={arXiv:2002.08071}}

@article{haber2017stable,
  title={Stable architectures for deep neural networks},
  author={Haber, Eldad and Ruthotto, Lars},
  journal={Inverse Problems}, volume={34}, number={1}, pages={014004}, year={2017}, note={arXiv:1705.03341}}

@inproceedings{chen2020srnn,
  title={Symplectic Recurrent Neural Networks},
  author={Chen, Zhengdao and Zhang, Jianyu and Arjovsky, Martin and Bottou, L{\'e}on},
  booktitle={ICLR}, year={2020}, note={arXiv:1909.13334, spotlight}}

@inproceedings{karl2017dvbf,
  title={Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data},
  author={Karl, Maximilian and Soelch, Maximilian and Bayer, Justin and van der Smagt, Patrick},
  booktitle={ICLR}, year={2017}, note={arXiv:1605.06432}}

@inproceedings{hafner2019planet,
  title={Learning Latent Dynamics for Planning from Pixels},
  author={Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle={ICML}, year={2019}, note={arXiv:1811.04551}}

@article{li2020latentsde,
  title={Scalable Gradients for Stochastic Differential Equations},
  author={Li, Xuechen and Wong, Ting-Kam Leonard and Chen, Ricky T. Q. and Duvenaud, David},
  journal={AISTATS}, year={2020}, note={arXiv:2001.01328}}

@article{lusch2018koopman,
  title={Deep learning for universal linear embeddings of nonlinear dynamics},
  author={Lusch, Bethany and Kutz, J. Nathan and Brunton, Steven L.},
  journal={Nature Communications}, volume={9}, pages={4950}, year={2018}}

@article{wang2025testtimeregression,
  title={Test-time regression: a unifying framework for designing sequence models with associative memory},
  author={Wang, Ke Alexander and Shi, Jiaxin and Fox, Emily B.},
  journal={arXiv preprint arXiv:2501.12352}, year={2025}}

@inproceedings{yang2025gateddeltanet,
  title={Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author={Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  booktitle={ICLR}, year={2025}, note={arXiv:2412.06464}}

@article{s4tomamba2025survey,
  title={From S4 to Mamba: A Comprehensive Survey on Structured State Space Models},
  author={(see arXiv listing)}, journal={arXiv preprint arXiv:2503.18970}, year={2025},
  note={AUTHOR LIST NOT VERIFIED -- fetch before citing}}
```

---

## Proposed handover updates (for the Hub)

- **Positioning line for §1 / any draft:** CLU is a **latent-dynamics substrate (family B)**, not a **latent-dynamics model (family A)**. Family A has four settled umbrella terms (DVAE · world model · LaSDI/LDM · LVM); family B has none, and the nearest unifier is **test-time regression / associative-memory layers** (arXiv:2501.12352) — which already contains every C3 ladder rival (Mamba-2, GDN-2, TTT) as a special case.
- **Never-quote candidate:** ⛔ do not call CLU a "latent dynamics model" or use the acronym **LDM** — collides with Latent Diffusion Model (arXiv:2112.10752), with the RL world-model sense, and with Farenga et al.'s ROM coinage.
- **Owner needed (small):** a legibility pass expressing CLU inside Wang–Shi–Fox's three design choices (regression weights / regressor class / test-time optimizer). Cheapest available referee-legibility win; pure prose, no compute.
- **Open gap flagged for a follow-up scout task:** **unitary/orthogonal RNNs** (uRNN, expRNN, and the norm-preservation-as-memory line) were NOT covered in this pass and are the closest prior art to "conservation law ⇒ long-horizon memory". Recommend a scoped follow-up before any conservation-based novelty claim ships.
