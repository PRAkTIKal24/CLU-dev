# v2-cite-check — web-scout report

Task + acceptance criterion: (1) verify + BibTeX every inline citation in `.claude/papers/v2-short/draft.md` (MF-2) with usage-checks on content-leans; (2) run the erosion-novelty CONFIRM pass on `venue-follow-up.md` §3 claims (b) and (c) against continual-EBM + equilibrium-propagation (MF-5 residual).
Status: **done**
**DIAL DECLARATION (echoed): none — citation verification + literature confirmation; no performance claim; no laundering control applies.**

> ## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — `v2-revision-7` unless the Hub says otherwise)
> 1. **U1 (draft lines 43, 81, 456):** "Mo proves … has dim(G/H) zero Lyapunov exponents" → must read **"at least dim(G/H)"**. Mo explicitly says "not exactly that number". §4 (line 55) and M§4 (line 535) already say "at least"; three sites don't. **Misstates a cited theorem — highest-priority fix.**
> 2. **U13 (lines 55, 148, 545):** "short-run CD **distorts an energy landscape** … (Fischer & Igel 2011)" — F&I *2011* bounds the **gradient bias**; the landscape/log-likelihood **divergence** result is **Fischer & Igel ICANN 2010**. Add/swap the 2010 ref.
> 3. **U14 (lines 55, 545):** "**conformal-symplectic** structure … standard (Hairer, Lubich & Wanner)" — conformal symplecticity is **McLachlan & Perlmutter 2001**; HLW is the right cite for leapfrog/`h<2` and Ch. XII dissipative perturbation, not for the conformal property. Also HLW is cited **with no year**.
> 4. **MF-2 mechanics:** `Rusch & Mishra 2021` must split into **2021a (coRNN, ICLR)** / **2021b (UnICORNN, ICML/PMLR 139)**; **EDEN** and **Titans** are cited by system name only and need author-year in text.
> 5. **Part 2 (c):** the frequency-vs-chain-length horizon law needs a **scope clause** vs the Decelle-line RBM result that learning *does* depend on k (see §Part 2). Un-hedging without it invites a "contradicts known RBM results" referee hit.
> 6. **Program-record correction (not in the draft, but in `venue-follow-up.md` §3 basis for claim (a)):** arXiv:**2503.21536** is **Toledo-Marin, Maiti, Fox & Melko** (not "Décelle-line") and attributes symmetry breaking to **hierarchical feature learning, not CD**. The "CD induces spurious symmetry breaking is also documented" basis is **unsupported by that reference**.

What I did:
- Swept the full 566-line draft for inline author-year citations (per-file Grep, `\b(19|20)\d\d[a-z]?\b` + name tokens; positive controls returned hits, so negatives are real). Found **26 distinct works** + 1 anonymous placeholder. **No "Guo" citation exists in the draft** (task's conditional); **arXiv 2606.24945/24946 (Wang) does not appear in the draft** — nothing to patch here for the escalation-3 trap.
- Verified each against publisher / arXiv / proceedings primary (never an aggregator alone where a primary exists). Retrieval date for every record below: **2026-08-18**.
- Usage-checked every place the draft leans on a source's *content* (Mo §3.2/§4/App A; L.1–L.5; HiPPO; Kong; Csordás; Jelassi; EDEN; Titans; UnICORNN; Ramsauer; Fischer–Igel; Nijkamp; HLW).
- Ran the MF-5 confirm pass across CD/EBM-theory, continual-EBM, equilibrium-propagation, RBM-symmetry-breaking literatures, incl. anything posted since Jul 20 2026.

How I verified: WebFetch of arXiv abs/HTML, PMLR, NeurIPS virtual/proceedings, DBLP, AIMS, PMC, Semantic Scholar Graph API; two SSO/paywall blocks routed to labelled public mirrors (`nature.com` → PMC; `pubmed` cookie-wall → S2 API). Every number/quote below is copied from the fetched primary unless marked otherwise.

---

# PART 1 — the bibliography

Legend: **✔ verified** (publisher/arXiv/proceedings primary) · **△ single-sourced** · **⛔ trap** (never-copy note).

## 1. Mo (2026) — the head-to-head anchor
**Record ✔** Hanson Hanxuan Mo, "Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks", **arXiv:2605.03338** (5 May 2026), cs.NE + math.DS. **Preprint — no venue, no journal-ref.** Single author.
⛔ **Traps:** (i) author is a *single* author — never "Mo et al."; (ii) it is **not peer-reviewed**; the draft's "a published machine-learning lifetime law" (abstract, §3.2, contributions) should read *"a recently posted / preprint lifetime law"* or at minimum not imply peer review. (iii) the theorem is **"at least dim(G/H)"**.
**Usage-check (4 leans):**
| draft says | source says | verdict |
|---|---|---|
| "proves that an exactly equivariant recurrent flow has $\dim(G/\mathcal H)$ zero Lyapunov exponents" (43, 81, 456) | "The theorem gives **at least** dim(G/H) zero exponents, **not exactly that number**, because unrelated zero exponents can arise from time translation, additional symmetries, conservation laws, or finite-time numerical effects." | **FAIL (drops "at least")** → reconciliation item 1 |
| "his published median $1.013$" (13, 43, 391, 460) | paper reports "median measured/predicted lifetime ratio about **1.013**" | **PASS** |
| "Where Mo lists conservation laws among *confounding* extra zero exponents" (465) | same sentence as row 1 — conservation laws named as a source of unrelated zeros | **PASS (verbatim support)** |
| "his exact code-level protocol (phase $0.35$ rad, threshold $0.2$ rad, his censoring, cap $15000$ steps)" (43, 456) | `mo-deep-read` §2 (full-HTML read): $\phi_0=0.35$, threshold $0.2$ rad, max $t=1500$ at $dt=0.1$ ⇒ $1.5\times10^4$ steps, non-crossers censored | **PASS △** — my independent re-fetch of the HTML through a summarising reader did not surface the constants (reader truncation), so this row rests on the program's own full-text read of the same v1 HTML |
| Fig. 1 / §4 use of "his diagnostics (normalized equivariance error, group-tangent exponent $\hat\lambda_\xi(T)$)" | paper: $\hat\lambda_\xi(T)=\frac1T\log\frac{\|D\phi_T(x)\xi_M(x)\|}{\|\xi_M(x)\|}$; equivariance error normalized by $1+\|f(x)\|$ | **PASS** |
```bibtex
@article{mo2026symmetry,
  title={Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks},
  author={Mo, Hanson Hanxuan}, journal={arXiv preprint arXiv:2605.03338}, year={2026},
  note={Preprint, not peer-reviewed; single author (never ``Mo et al.''); theorem states AT LEAST dim(G/H) zero exponents. Retrieved 2026-08-18.}}
```

## 2. Kong, Brewer & Lai 2024 — L.2 retirement
**Record ✔** Ling-Wei Kong, Gene A. Brewer, Ying-Cheng Lai, "Reservoir-computing based associative memory and itinerancy for complex dynamical attractors", **Nature Communications 15, 4840 (2024)**, **DOI 10.1038/s41467-024-49190-4** (verified via PMC11156990 after nature.com SSO 303; the DOI/volume/article-number are the publisher's own record fields). **← this closes the "DOI owed" item.**
**Usage-check:** draft L.2 "capacity scaling $N_c\propto K^{1.08\pm0.01}$" — source: "*$N_c \propto K^\gamma$ that are close to a linear law: $\gamma = 1.08\pm0.01$ for both the one-hot coding and binary coding, and $\gamma=1.17\pm0.02$ for the 2D coding*". **PASS, but incomplete** — add "(one-hot / binary index coding; $1.17\pm0.02$ for 2D coding)". Draft "location-addressable retrieval … via an index channel" — source: "*In the 'location-addressable' or 'parameter-addressable' scenario, the stored memory states within the neural network are activated by a specific location address or an index parameter*". **PASS.** Draft "the address enters as a bias rather than as the initial condition of the state" — **UNVERIFIED detail** (plausible from the index-channel construction; I did not find a sentence licensing "bias"). Soften or verify.
```bibtex
@article{kong2024reservoir,
  title={Reservoir-computing based associative memory and itinerancy for complex dynamical attractors},
  author={Kong, Ling-Wei and Brewer, Gene A. and Lai, Ying-Cheng},
  journal={Nature Communications}, volume={15}, number={1}, pages={4840}, year={2024},
  doi={10.1038/s41467-024-49190-4},
  note={Exponent 1.08+-0.01 is for one-hot AND binary coding; 2D coding gives 1.17+-0.02. Retrieved 2026-08-18.}}
```

## 3. Gu, Dao, Ermon, Rudra & Ré 2020 — HiPPO (L.1, §4)
**Record ✔** "HiPPO: Recurrent Memory with Optimal Polynomial Projections", **NeurIPS 2020** (DBLP), **arXiv:2008.07669** (17 Aug 2020; v2 23 Oct 2020).
**Usage-check (3 leans, all PASS):** (i) "$O(tL/\sqrt N)$ approximation bound" — Prop. 6: "*If $f$ is $L$-Lipschitz then $\|f_{\le t}-g^{(t)}\|=O(tL/\sqrt N)$*", with the smooth case $O(t^kN^{-k+1/2})$ (the ar5iv text-extraction mangled the radical; the $k$-th-order form's $N^{-k+1/2}$ confirms $\sqrt N$ at $k=1$). (ii) "**$\Theta(1/t)$ gradient decay, polynomial not exponential**" — Prop. 5: "*$\|\partial c(t_1)/\partial f(t_0)\|=\Theta(1/t_1)$*". (iii) "*exact* timescale-equivariance proposition and no discretization step to tune" — Prop. 3: "*For any scalar $\alpha>0$, if $h(t)=f(\alpha t)$, then hippo$(h)(t)=$hippo$(f)(\alpha t)$*", plus "*the discrete recurrence is invariant to the discretization step size*" (LegS only; LegT/LagT do carry $\theta$/$\Delta t$). Draft correctly scopes to **HiPPO-LegS**.
```bibtex
@inproceedings{gu2020hippo,
  title={HiPPO: Recurrent Memory with Optimal Polynomial Projections},
  author={Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2020}, note={arXiv:2008.07669. Props. 3/5/6 = timescale equivariance / Theta(1/t) gradient / O(tL/sqrt N). Retrieved 2026-08-18.}}
```

## 4. Jelassi, Brandfonbrener, Kakade & Malach 2024 (L.4)
**Record ✔** "Repeat After Me: Transformers are Better than State Space Models at Copying", **ICML 2024, PMLR 235:21502–21521**; arXiv:2402.01032.
**Usage-check:** draft "formal results on the limits of fixed-size latent state for copying and retrieval … conservation prevents *decay*, not *capacity*" — source proves a two-layer transformer copies exponential-length strings "*while GSSMs are fundamentally limited by their fixed-size latent state*". **PASS.**
```bibtex
@inproceedings{jelassi2024repeat,
  title={Repeat After Me: Transformers are Better than State Space Models at Copying},
  author={Jelassi, Samy and Brandfonbrener, David and Kakade, Sham M. and Malach, Eran},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={235}, pages={21502--21521}, year={2024}, note={arXiv:2402.01032. Retrieved 2026-08-18.}}
```

## 5. EDEN — NeurIPS 2025 (§4, L.5(ii))
**Record ✔** Arjun Karuvally, Pichsinee Lertsaroj, Terrence J. Sejnowski, Hava T. Siegelmann, "Exponential Dynamic Energy Network for High Capacity Sequence Memory", **NeurIPS 2025** (neurips.cc/virtual/2025/poster/118920, main-conference poster), **arXiv:2510.24965** (28 Oct 2025).
⛔ **Trap:** "EDEN" is also a 2019 DRAM-inference paper (arXiv:1910.05340) — always give authors.
**Usage-check (all PASS):** "analytic escape times" + "static/dynamic phase transition" — abstract: short-timescale energy functions "*are used to analytically compute memory escape times, revealing a phase transition between static and dynamic regimes*"; "capacity $O(\gamma^N)$" — "*exponential sequence memory capacity $\mathcal O(\gamma^N)$, outperforming the linear capacity $\mathcal O(N)$*". **Draft currently cites it with no authors — fix.**
```bibtex
@inproceedings{karuvally2025eden,
  title={Exponential Dynamic Energy Network for High Capacity Sequence Memory},
  author={Karuvally, Arjun and Lertsaroj, Pichsinee and Sejnowski, Terrence J. and Siegelmann, Hava T.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2025}, note={arXiv:2510.24965. Distinct from the 2019 ``EDEN'' DRAM paper. Retrieved 2026-08-18.}}
```

## 6. Titans — NeurIPS 2025 (§4, L.5)
**Record ✔** Ali Behrouz, Peilin Zhong, Vahab Mirrokni, "Titans: Learning to Memorize at Test Time", **NeurIPS 2025** (proceedings.neurips.cc paper_files/paper/2025), arXiv:2501.00663 (31 Dec 2024).
**Usage-check — the strongest content-lean in the paper, and it PASSES verbatim:** draft "the nearest published neighbour to this paper's *write* is test-time gradient descent with momentum on an associative-memory loss". Source: loss $\ell(\mathcal M_{t-1};x_t)=\|\mathcal M_{t-1}(k_t)-v_t\|_2^2$; update $\mathcal M_t=\mathcal M_{t-1}+S_t$, $S_t=\eta_tS_{t-1}-\theta_t\nabla\ell(\mathcal M_{t-1},x_t)$, with decay $\mathcal M_t=(1-\alpha_t)\mathcal M_{t-1}+S_t$; and explicitly: "*this formulation is similar to gradient descent with momentum, where $S_t$ is the momentum element*". Draft's "damping and inertia are read out and priced ($\gamma,M,\mu$) rather than chosen as optimizer hyperparameters" is a fair contrast ($\alpha_t,\eta_t$ are learned gates). **PASS.**
```bibtex
@inproceedings{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2025}, note={arXiv:2501.00663. Eq. 13--14: M_t=(1-alpha_t)M_{t-1}+S_t, S_t=eta_t S_{t-1}-theta_t grad l; ``similar to gradient descent with momentum''. Retrieved 2026-08-18.}}
```

## 7. Csordás & Schmidhuber 2019 (§4, L.3)
**Record ✔** Róbert Csordás, Jürgen Schmidhuber, "Improving Differentiable Neural Computers Through Memory Masking, De-allocation, and Link Distribution Sharpness Control", **ICLR 2019** (OpenReview `HyGEM3C9KQ`), arXiv:1904.10278.
**Usage-check:** draft "diagnose three DNC failure modes … **key/value entanglement — which they rank as the most important**". Source: "***Most importantly**, the lack of key-value separation makes the address distribution resulting from content-based look-up noisy and flat*"; "*DNC's de-allocation of memory results in aliasing*"; "*chaining memory reads with the temporal linkage matrix exponentially degrades the quality of the address distribution*". **PASS on all three modes and on the ranking.**
```bibtex
@inproceedings{csordas2019improving,
  title={Improving Differentiable Neural Computers Through Memory Masking, De-allocation, and Link Distribution Sharpness Control},
  author={Csord{\'a}s, R{\'o}bert and Schmidhuber, J{\"u}rgen},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2019}, note={arXiv:1904.10278; OpenReview HyGEM3C9KQ. Retrieved 2026-08-18.}}
```

## 8. Iqbal, Keller, Song, Miyato & Welling 2026 (§4, M§4)
**Record ✔ — author order in the draft is CORRECT.** Nabil Iqbal, T. Anderson Keller, Yue Song, Takeru Miyato, Max Welling, "Spontaneous symmetry breaking and Goldstone modes for deep information propagation", **arXiv:2605.14685** (14 May 2026), 28 pp, code at github.com/nabiliqbal/ssb-goldstone-deep-info-prop. **Preprint, no venue.**
⛔ **Trap:** Welling's own announcement lists "Iqbal, Keller, Miyato and Song" — the **arXiv order is Iqbal, Keller, Song, Miyato, Welling**. Do not copy the tweet order.
**Usage-check:** draft "a parallel physics-grounded route to stable long-range propagation is SSB with gapless Goldstone carriers" — abstract: "*these degrees of freedom enable coherent signal propagation across depth and recurrent iterations, providing a mechanism for stable information flow without relying on architectural stabilizers such as residual connections or normalization*"; also "*in recurrent settings … valuable for long-term memory*". **PASS** — note this is a *closer* neighbour than the draft's one-line treatment implies (they also claim the recurrent/long-memory benefit); §4's single sentence is defensible but a referee may press for the retention-vs-propagation demarcation.
```bibtex
@article{iqbal2026ssb,
  title={Spontaneous symmetry breaking and Goldstone modes for deep information propagation},
  author={Iqbal, Nabil and Keller, T. Anderson and Song, Yue and Miyato, Takeru and Welling, Max},
  journal={arXiv preprint arXiv:2605.14685}, year={2026},
  note={Preprint. arXiv author order differs from the authors' own social-media listing. Retrieved 2026-08-18.}}
```

## 9/10. ⛔ Rusch & Mishra 2021 — THE COLLISION (coRNN vs UnICORNN)
**2021a ✔** T. Konstantin Rusch, Siddhartha Mishra, "Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies", **ICLR 2021** (DBLP record; OpenReview `rOGm97YR22N`), arXiv:2010.00951.
**2021b ✔** T. Konstantin Rusch, Siddhartha Mishra, "UnICORNN: A recurrent model for learning very long time dependencies", **ICML 2021, PMLR 139:9168–9178** (journal-ref on the arXiv record), arXiv:2103.05487.
**Usage-check (L.5(iii), §4):** draft "**UnICORNN** … Hamiltonian, symplectic-Euler, *with* a gradient-bound theorem". Abstract: "*based on a structure preserving discretization of a **Hamiltonian system** of second-order ordinary differential equations that models networks of oscillators*"; "*we derive **rigorous bounds on the hidden state gradients** to prove the mitigation of the exploding and vanishing gradient problem*". **PASS** on Hamiltonian + gradient bound; **"symplectic-Euler" is body-level, not in the abstract — △ single-sourced wording**, safe but say "structure-preserving (symplectic) discretization" if you want zero exposure.
```bibtex
@inproceedings{rusch2021cornn,
  title={Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies},
  author={Rusch, T. Konstantin and Mishra, Siddhartha},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021}, note={arXiv:2010.00951. CITE AS 2021a. Retrieved 2026-08-18.}}
@inproceedings{rusch2021unicornn,
  title={UnICORNN: A recurrent model for learning very long time dependencies},
  author={Rusch, T. Konstantin and Mishra, Siddhartha},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={139}, pages={9168--9178}, year={2021}, note={arXiv:2103.05487. CITE AS 2021b. Retrieved 2026-08-18.}}
```

## 11. Rusch, Mishra, Erichson & Mahoney 2022 — LEM
**Record ✔** "Long Expressive Memory for Sequence Modeling", **ICLR 2022** (DBLP; OpenReview `vwj6aUeocyf`), arXiv:2110.04744. Identity-only in the draft (comparator) — nothing to usage-check.
```bibtex
@inproceedings{rusch2022lem,
  title={Long Expressive Memory for Sequence Modeling},
  author={Rusch, T. Konstantin and Mishra, Siddhartha and Erichson, N. Benjamin and Mahoney, Michael W.},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2022}, note={arXiv:2110.04744. Retrieved 2026-08-18.}}
```

## 12. Hochreiter & Schmidhuber 1997 — LSTM
**Record ✔** "Long Short-Term Memory", **Neural Computation 9(8):1735–1780 (1997)**, DOI 10.1162/neco.1997.9.8.1735 (MIT Press record). Identity-only.
```bibtex
@article{hochreiter1997lstm,
  title={Long Short-Term Memory}, author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural Computation}, volume={9}, number={8}, pages={1735--1780}, year={1997},
  doi={10.1162/neco.1997.9.8.1735}, note={Retrieved 2026-08-18.}}
```

## 13. Golubitsky, Stewart & Schaeffer 1988
**Record ✔** *Singularities and Groups in Bifurcation Theory: Volume II*, Applied Mathematical Sciences **69**, Springer-Verlag, 1988; ISBN 978-0-387-96652-6; DOI 10.1007/978-1-4612-4574-2.
⛔ **Trap:** Volume I (1985) is Golubitsky & Schaeffer only (AMS 51). The symmetry material the draft leans on is **Volume II** — and the **1988** year only matches Vol. II. Usage is a general "neutrality of orbit directions is classical" attribution: **PASS (identity-class)**; a referee could ask for a chapter pointer (Ch. XIII/XIV, equivariant bifurcation).
```bibtex
@book{golubitsky1988singularities,
  title={Singularities and Groups in Bifurcation Theory: Volume II},
  author={Golubitsky, Martin and Stewart, Ian and Schaeffer, David G.},
  series={Applied Mathematical Sciences}, volume={69}, publisher={Springer-Verlag}, year={1988},
  doi={10.1007/978-1-4612-4574-2}, note={Vol. II (1988) is the 3-author symmetry volume; Vol. I (1985) is Golubitsky & Schaeffer. Retrieved 2026-08-18.}}
```

## 14. Krupa 1990
**Record ✔** Martin Krupa, "Bifurcations of relative equilibria", **SIAM J. Math. Anal. 21(6):1453–1486 (1990)**, DOI 10.1137/0521081. Identity-class use. **PASS.**
```bibtex
@article{krupa1990bifurcations,
  title={Bifurcations of relative equilibria}, author={Krupa, Martin},
  journal={SIAM Journal on Mathematical Analysis}, volume={21}, number={6}, pages={1453--1486},
  year={1990}, doi={10.1137/0521081}, note={Retrieved 2026-08-18.}}
```

## 15. Rumberger 2001
**Record ✔** Matthias Rumberger, "Lyapunov exponents on the orbit space", **Discrete and Continuous Dynamical Systems 7(1):91–113 (2001)**, DOI 10.3934/dcds.2001.7.91 (AIMS publisher page). Content: equivariant system reduced to orbit space; formulas relating reduced and full Lyapunov exponents; "*drifts along the group orbits disappear*" — **directly supports** the draft's "neutrality of group-orbit directions is classical". **PASS.**
```bibtex
@article{rumberger2001lyapunov,
  title={Lyapunov exponents on the orbit space}, author={Rumberger, Matthias},
  journal={Discrete and Continuous Dynamical Systems}, volume={7}, number={1}, pages={91--113},
  year={2001}, doi={10.3934/dcds.2001.7.91}, note={Retrieved 2026-08-18.}}
```

## 16. Di Bernardo, Valente, Mastrogiuseppe & Ostojic 2025
**Record ✔** "Shaping manifolds in equivariant recurrent neural networks", **arXiv:2511.04802** (6 Nov 2025; v2 13 Nov 2025), 46 pp, 7 figs. **Preprint, no venue** — draft says "Di Bernardo et al. (2025)" ✔.
**Usage-check:** draft "use group representation theory to link the symmetry of a group-convolutional RNN's connectivity to the symmetry, dimension, and stability of its fixed-point manifolds, showing several subgroup manifolds can coexist (stable and saddle)". Abstract: "*using group representation theory to formalize the relationship between the symmetries in recurrent connectivity and the resulting fixed-point manifolds*"; "*several manifolds with different symmetry subgroups can coexist, some stable and others consisting of saddle points*". **PASS — near-verbatim.**
```bibtex
@article{dibernardo2025shaping,
  title={Shaping manifolds in equivariant recurrent neural networks},
  author={Di Bernardo, Arianna and Valente, Adrian and Mastrogiuseppe, Francesca and Ostojic, Srdjan},
  journal={arXiv preprint arXiv:2511.04802}, year={2025}, note={Preprint (46 pp). Retrieved 2026-08-18.}}
```

## 17. Keller 2025
**Record ✔** T. Anderson Keller, "Flow Equivariant Recurrent Neural Networks", **NeurIPS 2025 (spotlight poster)**, arXiv:2507.14793. **Single author** — draft's "Keller (2025)" ✔ (do not write "Keller et al.").
**Usage-check:** draft "extends equivariance from static transforms to continuous flows" — source: extends equivariant theory to "*'flows' — one-parameter Lie subgroups capturing natural transformations over time*". **PASS.** Draft's "neither prices *time* — their manifolds carry no dissipation parameter, decay rate, or retention timescale" is an **absence claim** about a cited work; defensible but it is the kind of sentence a referee/author will contest — keep it as "we find no retention-timescale statement in either".
```bibtex
@inproceedings{keller2025flow,
  title={Flow Equivariant Recurrent Neural Networks}, author={Keller, T. Anderson},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2025},
  note={Spotlight. Single author. arXiv:2507.14793. Retrieved 2026-08-18.}}
```

## 18. Hairer, Lubich & Wanner — ⛔ the mis-attribution
**Record ✔** Ernst Hairer, Christian Lubich, Gerhard Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, Springer Series in Computational Mathematics **31**, 2nd ed., Springer, 2006. **The draft cites it with no year at all** (lines 55, 545).
**Usage-check:** draft "**conformal-symplectic structure** and the $h<2$ limit are standard for damped leapfrog integrators (Hairer, Lubich & Wanner)". HLW covers Störmer–Verlet/leapfrog and its linear stability, and has Ch. **XII "Dissipatively Perturbed Hamiltonian and Reversible Systems"** (ToC verified) — but **conformal symplecticity ($J^\top\Omega J=(1-\gamma)\Omega$) is McLachlan & Perlmutter (2001)**, with the second-order conformal-symplectic *schemes* due to Bhatt, Floyd & Moore (2016). **PARTIAL FAIL** — split the cite. (I did not obtain HLW full text; the negative "HLW does not define conformal symplecticity" is △ single-sourced on the ToC + the independent literature consensus that McLachlan–Perlmutter is the origin: "*The conformal Hamiltonian system with linear damping term was first proposed by McLachlan in 2001*".) An ML-adjacent option: França, Jordan & Vidal-type "Conformal Symplectic and Relativistic Optimization" (NeurIPS 2020) — **not verified this pass; do not cite without a check.**
```bibtex
@book{hairer2006geometric,
  title={Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations},
  author={Hairer, Ernst and Lubich, Christian and Wanner, Gerhard},
  series={Springer Series in Computational Mathematics}, volume={31}, edition={2}, publisher={Springer}, year={2006},
  note={Cite for leapfrog/Stormer-Verlet + linear stability and Ch. XII dissipative perturbation -- NOT for conformal symplecticity. Retrieved 2026-08-18.}}
@article{mclachlan2001conformal,
  title={Conformal Hamiltonian systems}, author={McLachlan, Robert I. and Perlmutter, Matthew},
  journal={Journal of Geometry and Physics}, volume={39}, number={4}, pages={276--300}, year={2001},
  note={Origin of the conformal-symplectic property J^T Omega J = e^{-gamma t} Omega. Retrieved 2026-08-18.}}
@article{bhatt2016conformal,
  title={Second order conformal symplectic schemes for damped Hamiltonian systems},
  author={Bhatt, Ashish and Floyd, Dwayne and Moore, Brian E.},
  journal={Journal of Scientific Computing}, volume={66}, number={3}, pages={1234--1259}, year={2016},
  note={Optional: the integrator-side companion to McLachlan-Perlmutter. Retrieved 2026-08-18.}}
```

## 19. Hinton, Dayan, Frey & Neal 1995 — wake–sleep
**Record ✔** "The 'wake-sleep' algorithm for unsupervised neural networks", **Science 268(5214):1158–1161 (1995)**, DOI 10.1126/science.7761831, PMID 7761831 (Semantic Scholar Graph API record, `pages: 1158-61`; science.org returned 403).
⛔ **Trap:** Hinton's own web listing says **1158–1160**; the publisher/PubMed record is **1158–1161**. Use 1158–1161.
```bibtex
@article{hinton1995wakesleep,
  title={The ``wake-sleep'' algorithm for unsupervised neural networks},
  author={Hinton, Geoffrey E. and Dayan, Peter and Frey, Brendan J. and Neal, Radford M.},
  journal={Science}, volume={268}, number={5214}, pages={1158--1161}, year={1995},
  doi={10.1126/science.7761831}, note={Author's own page lists 1158--1160; publisher record is 1158--1161. Retrieved 2026-08-18.}}
```

## 20. Tieleman 2008 — PCD
**Record ✔** Tijmen Tieleman, "Training restricted Boltzmann machines using approximations to the likelihood gradient", **ICML 2008, pp. 1064–1071**, DOI 10.1145/1390156.1390290.
⛔ **Trap:** the title does **not** contain "persistent contrastive divergence" (PCD is introduced inside); the 2009 "Using fast weights to improve persistent contrastive divergence" (Tieleman & Hinton, ICML 2009) is a **different** paper. Draft's "PCD (Tieleman 2008)" is correct.
```bibtex
@inproceedings{tieleman2008pcd,
  title={Training restricted Boltzmann machines using approximations to the likelihood gradient},
  author={Tieleman, Tijmen}, booktitle={Proceedings of the 25th International Conference on Machine Learning (ICML)},
  pages={1064--1071}, year={2008}, doi={10.1145/1390156.1390290},
  note={Introduces PCD; title contains no ``persistent contrastive divergence''. Retrieved 2026-08-18.}}
```

## 21. Fischer & Igel — ⛔ the year/claim mismatch
**Record ✔ (2011)** Asja Fischer, Christian Igel, "Bounding the Bias of Contrastive Divergence Learning", **Neural Computation 23(3):664–673 (2011)**, DOI 10.1162/NECO_a_00085.
**Record ✔ (2010)** Asja Fischer, Christian Igel, "Empirical Analysis of the Divergence of Gibbs Sampling Based Learning Algorithms for Restricted Boltzmann Machines", **ICANN 2010, LNCS 6354:208–217**, DOI 10.1007/978-3-642-15825-4_26.
**Usage-check (FAIL-ish, reconciliation item 2):** draft (§4 line 55; App C line 148; M§4 line 545) cites **F&I 2011** for "*that short-run / non-convergent contrastive divergence **distorts an energy landscape** is classical*". F&I **2011** bounds the **bias of the CD gradient estimate** — a statement about the update, not the landscape. The landscape/likelihood **divergence** result is **F&I 2010**. Fix by citing 2010 (or both, with 2011 for the bias bound). Nijkamp et al. 2020 carries the landscape half correctly (below).
```bibtex
@article{fischer2011bounding,
  title={Bounding the Bias of Contrastive Divergence Learning}, author={Fischer, Asja and Igel, Christian},
  journal={Neural Computation}, volume={23}, number={3}, pages={664--673}, year={2011},
  doi={10.1162/NECO_a_00085}, note={BIAS OF THE GRADIENT, not landscape distortion. Retrieved 2026-08-18.}}
@inproceedings{fischer2010empirical,
  title={Empirical Analysis of the Divergence of Gibbs Sampling Based Learning Algorithms for Restricted Boltzmann Machines},
  author={Fischer, Asja and Igel, Christian}, booktitle={Artificial Neural Networks (ICANN)},
  series={LNCS}, volume={6354}, pages={208--217}, year={2010}, doi={10.1007/978-3-642-15825-4_26},
  note={THIS is the divergence-of-learning/landscape reference. Retrieved 2026-08-18.}}
```

## 22. Nijkamp, Hill, Han, Zhu & Wu 2020
**Record ✔** "On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models", **AAAI 2020** (journal-ref on arXiv:1903.12370; v4 27 Nov 2019).
⛔ **Trap:** the program's earlier bib (`venue-follow-up` §8) lists **five** authors incl. **Tian Han** ✔; a *sibling* paper "Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model" (NeurIPS 2019, arXiv:1904.09770) has **four** authors (no Han) — do not merge them.
**Usage-check:** draft "short-run CD distorts an energy landscape is classical (… Nijkamp et al. 2020)". Source: "*ConvNet potentials learned with non-convergent MCMC do not have a valid steady-state and cannot be considered approximate unnormalized densities of the training data because long-run MCMC samples differ greatly from observed images*". **PASS** (this sentence carries the landscape claim that F&I 2011 does not).
```bibtex
@inproceedings{nijkamp2020anatomy,
  title={On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models},
  author={Nijkamp, Erik and Hill, Mitch and Han, Tian and Zhu, Song-Chun and Wu, Ying Nian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, year={2020},
  note={arXiv:1903.12370. Distinct from the 4-author NeurIPS-2019 short-run-MCMC paper (arXiv:1904.09770). Retrieved 2026-08-18.}}
```

## 23/24. Graves et al. — NTM & DNC
**NTM ✔** Alex Graves, Greg Wayne, Ivo Danihelka, "Neural Turing Machines", **arXiv:1410.5401** (20 Oct 2014; v2 10 Dec 2014). **Never published in a venue** — draft's "Graves et al. 2014" ✔ but should not be implied peer-reviewed.
**DNC ✔** Graves, Wayne, Reynolds, Harley, Danihelka, Grabska-Barwińska, Colmenarejo, Grefenstette, Ramalho, Agapiou, Badia, Hermann, Zwols, Ostrovski, Cain, King, Summerfield, Blunsom, Kavukcuoglu, Hassabis, "Hybrid computing using a neural network with dynamic external memory", **Nature 538(7626):471–476 (2016)**, DOI 10.1038/nature20101.
**Usage-check:** draft L.3 "Both … were already fully soft, continuous and end-to-end differentiable" — NTM abstract: "*analogous to a Turing Machine or Von Neumann architecture but is **differentiable end-to-end***". **PASS.**
```bibtex
@article{graves2014ntm,
  title={Neural Turing Machines}, author={Graves, Alex and Wayne, Greg and Danihelka, Ivo},
  journal={arXiv preprint arXiv:1410.5401}, year={2014}, note={Preprint only -- never formally published. Retrieved 2026-08-18.}}
@article{graves2016dnc,
  title={Hybrid computing using a neural network with dynamic external memory},
  author={Graves, Alex and Wayne, Greg and Reynolds, Malcolm and Harley, Tim and Danihelka, Ivo and Grabska-Barwi{\'n}ska, Agnieszka and Colmenarejo, Sergio G{\'o}mez and Grefenstette, Edward and Ramalho, Tiago and Agapiou, John and Badia, Adri{\`a} Puigdom{\`e}nech and Hermann, Karl Moritz and Zwols, Yori and Ostrovski, Georg and Cain, Adam and King, Helen and Summerfield, Christopher and Blunsom, Phil and Kavukcuoglu, Koray and Hassabis, Demis},
  journal={Nature}, volume={538}, number={7626}, pages={471--476}, year={2016}, doi={10.1038/nature20101},
  note={20 authors -- ``Graves et al. 2016''. Retrieved 2026-08-18.}}
```

## 25. Ramsauer et al. 2021 (L.5(i))
**Record ✔** Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlović, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter, "Hopfield Networks is All You Need", **ICLR 2021**, arXiv:2008.02217.
**Usage-check:** draft "the modern Hopfield update ***is*** the attention mechanism". Source: the CCCP-derived update "*matches the computation performed in the attention layer of a transformer* **with a single attention head, identity projection matrices, and $\beta=1/\sqrt D$**". **PASS with conditions** — the identity is exact only in that special case; add "(for a single head with identity projections and $\beta=1/\sqrt D$)" or soften "is" → "coincides with".
```bibtex
@inproceedings{ramsauer2021hopfield,
  title={Hopfield Networks is All You Need},
  author={Ramsauer, Hubert and Sch{\"a}fl, Bernhard and Lehner, Johannes and Seidl, Philipp and Widrich, Michael and Adler, Thomas and Gruber, Lukas and Holzleitner, Markus and Pavlovi{\'c}, Milena and Sandve, Geir Kjetil and Greiff, Victor and Kreil, David and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021},
  note={arXiv:2008.02217. Attention equivalence holds for single head, identity projections, beta=1/sqrt(D). Retrieved 2026-08-18.}}
```

## 26. Jawahar & Pierini (2026) — the self-cite
**Record ✔** Pratik Jawahar, Maurizio Pierini, "CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning", **arXiv:2603.01768** (2 Mar 2026; v2 15 Mar 2026). arXiv **comments field, verbatim: "Accepted as a short paper at ICLR 2026 (AI & PDE)"** — i.e. an ICLR-2026 *workshop* short, not the main track; label it accordingly.
⚠ **Anonymity flag (Hub decision, not mine):** the venue is **double-blind**, and the draft writes "Our reference unit is the **CLU … introduced as CHLU in Jawahar & Pierini (2026)**" (lines 19, 383) while §1 calls it "our reference unit". Third-person self-citation is normally permitted, but "our reference unit … introduced in <our names>" is a de-facto identification. Recommend the standard construction: "the CLU (introduced as CHLU by Jawahar & Pierini, 2026)" with no possessive, or an anonymised placeholder.
```bibtex
@article{jawahar2026chlu,
  title={CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning},
  author={Jawahar, Pratik and Pierini, Maurizio}, journal={arXiv preprint arXiv:2603.01768}, year={2026},
  note={arXiv comments: ``Accepted as a short paper at ICLR 2026 (AI \& PDE)'' (workshop short). Retrieved 2026-08-18.}}
```

## 27. "Anonymous, 2026 (*the theory note*)" — identity-only
No external record exists or should exist (double-blind companion). **Nothing to verify.** Note for the revision spoke: NeurReps-style anonymised companion citations usually need a `note={Anonymous companion note, under review}` and no arXiv id; the bib must not resolve to a named preprint.

## Sweep completeness
Inline author-year citations found in the full 566-line file: items 1–27 above. **Not present in the draft** (checked, so no action): "Guo" (any year); Wang arXiv:2606.24946/24945; Minami–Hidaka; Du & Mordatch; Décelle/Seoane refs; Hopfield 1982; Scellier & Bengio. Non-citation proper nouns (LSTM/LEM/coRNN/Adam/BFGS/JAX) excluded.

---

# PART 2 — erosion-novelty CONFIRM pass (MF-5 residual)

Claims as `venue-follow-up.md` §3 words them:
- **(b)** "degeneracy-specificity demarcation (flat direction unconstrained by wake ⇔ eroded; non-degenerate ⇔ immune)" — i.e. "*wake–sleep CD inverts a designed vacuum **iff** it has a flat direction the wake objective cannot see*".
- **(c)** "horizon law: erosion set by CD-update **frequency** racing the wake clamp schedule, **independent of chain length $k$**".

## Verdict (b): **CONFIRMED-NOVEL** (still absent; confidence medium-high)
No prior statement of the demarcation was found in any searched venue. Nearest neighbours, and why each differs:
- **Toledo-Marin, Maiti, Fox & Melko (2025), "Exploring the Energy Landscape of RBMs: Reciprocal Space Insights into Bosons, Hierarchical Learning and Symmetry Breaking", *Mach. Learn.: Sci. Technol.* 6, 035030; arXiv:2503.21536.** RBMs "*initialize at saddle points with rotational symmetry that breaks during training*", and — quoting the paper — "*During training, this rotational symmetry is broken due to **hierarchical learning**, where different degrees of freedom progressively capture features at multiple levels of abstraction*". **Differs twice over:** the symmetry is an *initialization* symmetry of the weight spectrum, broken by feature learning; CD/PCD appear only as sampling implementations and are **not** the causal mechanism. **This also corrects the program's own record** (see reconciliation item 6).
- **"Distributional simplicity bias and effective convexity in energy-based models", arXiv:2605.07844 (2026).** Contains the closest-sounding sentence — attractors "*may instead form a **degenerate manifold of fixed points** … every point … corresponding to the same effective model*" — but this is degeneracy in **parameter** space (unidentifiability of the learned model), not a flat direction of the *learned potential in state space* that the objective cannot see. **Not prior art.**
- **Continual-EBM line** (Li, Du, Mordatch et al., "Energy-Based Models for Continual Learning", arXiv:2011.12216; LSEBMCL, arXiv:2501.05495): uses the contrastive/negative phase as the *cure* for forgetting; contains no statement that the negative phase erodes structure the positive phase cannot constrain, and no degeneracy condition.
- **Equilibrium propagation line** (Scellier & Bengio, *Front. Comput. Neurosci.* 11:24, 2017; Kubo, Delanois & Bazhenov, "Toward Lifelong Learning in Equilibrium Propagation: Sleep-like and Awake Rehearsal for Enhanced Stability", arXiv:2508.14081, Aug 2025; DEEP, *Mathematics* 13(11):1866, 2025): EP contrasts two attractor states; the "sleep"-phase literature is about **rehearsal against catastrophic forgetting**, not about the contrastive phase destroying a degenerate/flat direction. Nothing matching (b).

## Verdict (c): **AMBIGUOUS — novel as stated, but it now has a live neighbour that must be addressed in print**
No prior statement of "erosion horizon = CD-update frequency racing the supervised clamp schedule" was found; the frequency-vs-supervision-schedule axis really is absent. **But the "independent of chain length $k$" half now collides with a documented result, and this is the actionable finding:**
- **Decelle, Furtlehner & Seoane, "Equilibrium and non-Equilibrium regimes in the learning of Restricted Boltzmann Machines", NeurIPS 2021; arXiv:2105.13889.** RBMs "*operate in two well-defined regimes, namely equilibrium and out-of-equilibrium, depending on the interplay between the mixing time of the model and the number of steps, $k$, used to approximate the gradient*"; the out-of-equilibrium regime encodes data statistics "*through a dynamical process*" rather than in the Gibbs measure.
- **Agoritsas, Catania, Decelle & Seoane, "Explaining the effects of non-convergent MCMC in the training of Energy-Based Models", ICML 2023, PMLR 202:322–336; arXiv:2301.09428** (arXiv title says "non-convergent *sampling*"): EBMs trained with non-persistent short runs "*can perfectly reproduce a set of empirical statistics of the data, not at the level of the equilibrium measure, but through a precise dynamical process*" — i.e. **what is learned is a function of $k$.**
⇒ **What survives:** nobody states a *frequency-vs-clamp-schedule* horizon law, and nobody states $k$-independence of a vacuum-erosion horizon. **What must be scoped:** the draft/V5 cannot say "chain length is irrelevant" flatly against a literature in which $k$ vs mixing time defines the learning regime. Required clause (my recommended wording): *"within our sweep both chain lengths ($k=50$ and $500$) sit on the same side of the model's mixing time, so the frequency-decisive/steps-irrelevant finding is a statement about that regime, not a contradiction of the $k$-dependence documented for RBM learning (Decelle et al. 2021; Agoritsas et al. 2023)."*

## Coverage statement (so the absence is scoped, not asserted)
**Searched (2026-08-18):** arXiv (cs.LG, cs.NE, cond-mat.dis-nn, stat.ML) full-text/abstract search incl. listings through Aug 2026; NeurIPS (2020–2025 proceedings + virtual), ICML/PMLR (v119, v139, v202, v235), ICLR/OpenReview, AAAI; journals *Neural Computation*, *Machine Learning: Science and Technology*, *Frontiers in Computational Neuroscience*, *Mathematics*, *Nature Communications*.
**Terms:** contrastive divergence + {degenerate minima, flat direction, symmetry restoration, order parameter, ring/continuous attractor, designed prior, vacuum}; negative-phase {frequency, schedule, update rate} vs {chain length, k, MCMC steps}; energy-based model + {continual learning, catastrophic forgetting, landscape distortion, non-convergent MCMC}; equilibrium propagation + {flat directions, degenerate minima, continuous attractor, sleep, forgetting}; RBM + {symmetry breaking, out-of-equilibrium, mixing time}; wake–sleep + {erosion, prior destruction, Hamiltonian EBM}.
**NOT searched (declare as residual):** paywalled full-text search of *Neural Computation* / *Neural Networks* archives; forward-citation crawls of Fischer–Igel 2010/2011 and Nijkamp 2020 (Scholar "cited by", ~thousands — the highest-yield next pass if the Hub wants belt-and-braces); non-English literature; NeurIPS/ICML **workshop** tracks (only main tracks were covered).
**Both verdicts remain novel-by-absence** — absence over the surfaces listed, not proof of none. They are now **two-instrument** (Jul-20 pass + this pass, different queries and different databases), which is the strongest form available short of a citation-graph crawl.

## Ship guidance for `v2-revision-7` (replaces the "pending a literature scout" markers at lines 49, 148, 523, 553)
Recommended print wording: *"Landscape distortion under short-run contrastive divergence is classical (Fischer & Igel 2010, 2011; Nijkamp et al. 2020); what we add is a quantified inversion of a **designed, symmetry-degenerate** vacuum on a symplectic EBM, the degeneracy-specificity demarcation, and a frequency-set erosion horizon. A targeted search of the CD/EBM, continual-EBM and equilibrium-propagation literatures (Aug 2026) found no prior statement of the latter two; the nearest results concern $k$-dependence of RBM learning regimes (Decelle et al. 2021; Agoritsas et al. 2023), which our sweep does not contradict (scope clause above)."*

```bibtex
@inproceedings{decelle2021equilibrium,
  title={Equilibrium and non-Equilibrium regimes in the learning of Restricted Boltzmann Machines},
  author={Decelle, Aur{\'e}lien and Furtlehner, Cyril and Seoane, Beatriz},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2021}, note={arXiv:2105.13889. Retrieved 2026-08-18.}}
@inproceedings{agoritsas2023explaining,
  title={Explaining the effects of non-convergent MCMC in the training of Energy-Based Models},
  author={Agoritsas, Elisabeth and Catania, Giovanni and Decelle, Aur{\'e}lien and Seoane, Beatriz},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={202}, pages={322--336}, year={2023},
  note={arXiv:2301.09428 titled ``...non-convergent sampling...''; PMLR title says ``MCMC''. Retrieved 2026-08-18.}}
@article{toledomarin2025rbm,
  title={Exploring the Energy Landscape of RBMs: Reciprocal Space Insights into Bosons, Hierarchical Learning and Symmetry Breaking},
  author={Toledo-Mar{\'i}n, J. Quetzalc{\'o}atl and Maiti, Anindita and Fox, Geoffrey C. and Melko, Roger G.},
  journal={Machine Learning: Science and Technology}, volume={6}, number={3}, pages={035030}, year={2025},
  note={arXiv:2503.21536. Attributes symmetry breaking to HIERARCHICAL LEARNING, not to CD bias; authors are NOT the Decelle group. Retrieved 2026-08-18.}}
@article{scellier2017equilibrium,
  title={Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation},
  author={Scellier, Benjamin and Bengio, Yoshua}, journal={Frontiers in Computational Neuroscience},
  volume={11}, pages={24}, year={2017}, note={arXiv:1602.05179. Optional EP anchor if V5 cites the EP line. Retrieved 2026-08-18.}}
```

---

Findings/results summary (dense):
- **26 works + 1 anonymous placeholder** verified; **all 26 have a primary-source record**; **0 fabricated/nonexistent citations** found in the draft.
- **3 usage FAILs / PARTIALs:** Mo "at least" (3 sites), Fischer–Igel 2011-vs-2010, HLW-for-conformal-symplectic. **1 conditions-needed PASS:** Ramsauer "is the attention mechanism". **2 unverified details:** Kong "address enters as a bias"; UnICORNN "symplectic-Euler" (body-only).
- **2 label risks:** Mo 2026 and Iqbal et al. 2026 are **preprints**, but the draft repeatedly says "a **published** machine-learning lifetime law" (abstract, §1, §3.2, contributions). At a venue that will check, "published" is the wrong word for arXiv:2605.03338.
- **1 anonymity flag:** the Jawahar & Pierini self-cite construction.
- **Part 2:** (b) CONFIRMED-NOVEL; (c) novel-as-stated but needs an explicit $k$-regime scope clause against Decelle et al. 2021 / Agoritsas et al. 2023. MF-5 can now close on the drafting side.

Git footprint: none (read-only; no tracked file touched).

Open questions / follow-ups / risks:
1. Does the Hub want the "published" → "preprint" relabel applied to Mo everywhere, or a single scoping sentence in §3.2? (It changes the rhetorical weight of "a *published* ML law is the overdamped face of our budget".)
2. Forward-citation crawl of Fischer–Igel/Nijkamp (Scholar) is the only remaining lever on the (b)/(c) absence claims — worth a half-spoke if the Hub wants "verified" rather than "two-instrument novel-by-absence".
3. `Anonymous, 2026` theory-note bib entry must be settled before the LaTeX bib is built (it cannot resolve to a named preprint under double-blind).
4. I did not obtain HLW full text; the "HLW does not cover conformal symplecticity" half of item 3 is ToC + consensus based.

## Proposed handover updates (for the Hub)
- **MF-2 status:** bibliography inputs are complete — 26 verified records + BibTeX in `.claude/outputs/v2-cite-check.md`; `v2-revision-7` can typeset directly. Two entries need author-year added *in the prose* (EDEN, Titans); one needs a year (HLW); one needs a/b split (Rusch & Mishra 2021).
- **MF-5 status:** erosion-novelty confirm pass **done, 2026-08-18**. (b) CONFIRMED-NOVEL; (c) novel-as-stated + mandatory scope clause vs Decelle et al. 2021 / Agoritsas et al. 2023. The four "pending scout confirmation" markers (draft lines 49, 148, 523, 553) can be replaced with the wording in §Ship-guidance. MF-5 no longer blocks.
- **Program-record correction:** `venue-follow-up.md` §3, claim-(a) basis — arXiv:2503.21536 is **Toledo-Marin, Maiti, Fox & Melko** (MLST 6:035030, 2025), **not** "Décelle-line", and it attributes symmetry breaking to hierarchical feature learning, **not** to CD. The "CD induces spurious symmetry breaking in Boltzmann machines is documented" basis does **not** survive; the defensible substrate cites for claim (a) are Fischer–Igel 2010/2011 + Nijkamp et al. 2020 (which the draft already uses).
- **New standing bib facts:** Kong DOI = 10.1038/s41467-024-49190-4 (closes "DOI owed"); Mo 2026 = single-author **preprint**; Iqbal et al. 2026 arXiv order = Iqbal, Keller, Song, Miyato, Welling; CHLU self-cite = arXiv:2603.01768, comments "Accepted as a short paper at ICLR 2026 (AI & PDE)".
- **Anonymity item for the Head/Hub:** the "our reference unit … introduced as CHLU in Jawahar & Pierini (2026)" construction under a double-blind venue.

## Flags
- ⚠ **Draft says "published" for two preprints** (Mo 2026, Iqbal et al. 2026) — repeated in the abstract and contributions. Editorial, but referee-visible.
- ⚠ **Mo's theorem is "at least dim(G/H)"** — three draft sites drop the qualifier while two state it correctly; internally inconsistent as well as inaccurate.
- ⚠ **Fischer & Igel 2011 is cited for a claim it does not make** (landscape distortion); F&I **2010** does.
- ⚠ **Conformal symplecticity attributed to HLW**; origin is McLachlan & Perlmutter 2001.
- ⚠ **Part 2 (c)** cannot be un-hedged without the $k$-regime scope clause.
- ℹ SSO/paywall routing used and labelled: nature.com (303 → PMC), pubmed (cookie-wall → Semantic Scholar Graph API), science.org (403 → S2/PubMed record), OpenReview forum pages (browser check → DBLP records).
- ℹ Per-file Greps used throughout (`.claude/` directory-level Grep is unreliable in this repo); every negative sweep in this report was positive-controlled on the same file.
