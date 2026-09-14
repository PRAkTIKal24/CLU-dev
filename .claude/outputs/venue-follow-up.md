# venue-follow-up — web-scout report
Task + acceptance criterion: turn the (expected-live) NeurIPS 2026 accepted-workshop list into a per-short submission plan with verified per-workshop facts + earliest binding deadline; confirm ICLR 2027 CFP; gate V5 with an erosion-novelty verdict; close V3 reversible bib + citation-hygiene + paid-access prior-art flags.
Status: **partial (by necessity)** — the central accepted-workshop list is **not yet publicly posted** (Jul 11 was organizer notification, not public listing), so per-workshop tables can't be fully built yet. All other sub-tasks (ICLR 2027, V5 gate, V3 bib, hygiene, paid-access flags) are answered. Web-verified on **2026-07-19**.

## ⚠ ESCALATIONS (read first — downstream owners needed)
1. **A real 2026 workshop deadline is ALREADY earlier than Aug 25.** TS4H (Time Series for Health) @ NeurIPS 2026 has posted **paper deadline Aug 19, 2026** (Sydney, 4pp, double-blind, non-archival) — https://timeseries4health.github.io/. TS4H is not one of our targets, but it is a hard existence proof that our target workshops may land on **~Aug 19**, not the venue-suggested Aug 29. **The program freeze must be planned for ≤ Aug 17** (submission-ready Aug 18) until our actual targets post their CFPs. This supersedes the earlier "move freeze to Aug 21–24" guidance from `scout-venues-deadlines.md`. **Owner: Hub — set the freeze date at the wave-15 review.**
2. **The accepted-workshop list is NOT centrally posted yet** (neurips.cc Schedule = empty of workshops; blog silent; OpenReview proposals group shows no accepted list). Individual workshops are self-publishing on their own sites (TS4H is live). **Our three flagship targets (ML4PS, AI4Science, NeurReps) have NOT posted 2026 pages** (their sites still show 2025). A second scout pass is required in **~7–14 days** (target Jul 26–Aug 2) to catch their CFPs. **Owner: Hub — re-spawn this scout on that date.**
3. **Citation-hygiene reconciliation (has been sitting mislabeled in handover):** the Wang paper is **arXiv:2606.2494​6** (the Mo-citer value), **NOT 2606.24945** (the handover value — that ID appears not to exist). Correct string below (§4a). **Owner: Hub — patch the handover reference + any V2/V3 draft that inherited 24945.**

---

## 1. NeurIPS 2026 workshop calendar — verified state (2026-07-19)

| Item | Value | Source | Status |
|---|---|---|---|
| Workshop acceptance **notification to organizers** | **Jul 11, 2026 AoE** (past) | neurips.cc/Conferences/2026/CallForWorkshops | CONFIRMED |
| Public accepted-workshop **list** | **Not yet posted** as of Jul 19; no central date given. Workshops self-publish on own sites (TS4H live) | neurips.cc Schedule (empty); blog.neurips.cc (silent); OpenReview proposals group (no list) | CONFIRMED (absence) |
| Venue-**suggested** paper deadline | **Aug 29, 2026 AoE** (per-workshop discretion; may be earlier) | CallForWorkshops | CONFIRMED |
| **Earliest OBSERVED real deadline** | **Aug 19, 2026** (TS4H) — proof workshops go earlier than suggested | timeseries4health.github.io | CONFIRMED |
| Mandatory author accept/reject | **Sep 29, 2026 AoE** | CallForWorkshops | CONFIRMED |
| Workshop days | Sydney **Dec 11–12**; Paris & Atlanta **Dec 12–13** | CallForWorkshops | CONFIRMED |
| Archival status | **Non-archival venue-wide** ("All NeurIPS workshop papers are non-archival…") | WorkshopsGuidance (quoted in scout-venues-deadlines.md) | CONFIRMED |

**TS4H reference row (only fully-verifiable 2026 workshop this pass; template exemplar, not a target):** deadline Aug 19 2026 · up to **4 pages excl. refs/appendices** · **double-blind**, no rebuttal mentioned · **non-archival, no proceedings** · **Sydney**. This is the concrete format our physics targets will most likely mirror (ML4PS has used exactly 4pp/double-blind/no-rebuttal/non-archival for years).

## 2. Recommended assignment (provisional — targets not yet confirmed live)
Because the target CFPs aren't posted, this is the **plan to execute the moment they appear**, using the verified-recurring 2025 fingerprints from `scout-venues-deadlines.md`:

| Short | Primary target | Fallback | Rationale |
|---|---|---|---|
| **V1** cascade/attention (Lorentz boost / wormholes) | **AI4Science** (broad ML-for-science; tolerant of a primitive-mechanics story) | **ML4PS** | V1's inference-time-compute framing is less "physical-sciences core"; AI4Science's breadth fits better and keeps ML4PS free for the strongest physics story (V2). |
| **V2** Goldstone/SSB/EFT | **ML4PS** (best physics-audience fit; SSB/mass-spectrum reads native) | **NeurReps** (symmetry/geometry) | V2 is the flagship physics story; ML4PS reviewers are the target readership. NeurReps is an excellent symmetry-native fallback. |
| **V3** deep/stacked lattice | **NeurReps** (geometry/structure) or a **DLDE/differential-equations** workshop *if one is accepted this year* | **ML4PS** | V3's lattice/multi-scale structure fits geometry or DE-in-DL venues; confirm a DLDE-class workshop exists in the 2026 list before committing. |
| **V5** forgetting/erosion (GO, sibling of V2) | **EBM/generative or self-supervised workshop *if accepted*** ; else **ML4PS appendix-grade → its own venue** | **NeurReps** or fold into V2 | **Gated by §3 novelty verdict below.** V5's home depends on whether an EBM/CD-focused workshop is on the 2026 list; unknown until list posts. |

**One-short-one-workshop rule holds** (ML4PS precedent: "we strictly prohibit submitting to multiple workshops simultaneously"). Do not hedge a short across two. **Paris-site preference** (practical from Manchester) cannot yet be applied — site-per-workshop assignment is unknown until pages post.

**Binding earliest deadline (program freeze driver): assume Aug 19, 2026** until targets confirm otherwise. If any target posts earlier, escalate immediately.

## 3. V5 GATE — erosion-novelty verdict (sub-task 5, first-priority)
The four `sleep-erosion-study` claims, checked against the EBM/CD literature:

| Claim | Verdict | Basis |
|---|---|---|
| **(a)** designed-vacuum *inversion* (not just distortion) by wake–sleep CD | **PARTIAL (NOVEL framing on a known substrate)** | That CD/PCD is a biased update that **distorts/diverges** the energy landscape is **classical** — Fischer & Igel; Nijkamp et al. That CD induces **spurious symmetry breaking / imbalance between degenerate sectors** in Boltzmann machines is **also documented** (Décelle-line RBM energy-landscape work, arXiv:2503.21536). What remains **ours**: a *measured, quantified inversion of a deliberately-designed symmetry-degenerate vacuum into a local **maximum*** on a **symplectic/Hamiltonian EBM primitive** (ring depth +0.079 → −0.126, r\* 1.0 → 0). Ship as "sharp instance," cite the substrate; do **not** claim discovering CD bias. |
| **(b)** degeneracy-specificity demarcation (flat direction unconstrained by wake ⇔ eroded; non-degenerate ⇔ immune) | **NOVEL** (single-sourced negative — not found in CD/EBM lit) | No prior statement found of the crisp "wake–sleep CD inverts a designed vacuum **iff** it has a flat direction the wake objective cannot see." Closest adjacency is generic "CD is insensitive to distant/degenerate modes" (RBM lit), but not the demarcation-as-theorem. Confidence medium — negative result, keep searching. |
| **(c)** horizon law: erosion set by CD-update **frequency** racing the wake clamp schedule, **independent of chain length k** | **NOVEL (practitioner-relevant)** | RBM-era analysis of CD bias is parameterized by **chain length k / mixing** (Fischer & Igel bias bounds; Sutskever & Tieleman), **not** update-frequency-vs-supervision-schedule. The "steps-irrelevant, frequency-decisive" scaling is not in the surveyed literature. Single-sourced (our own measurement) — flag as our contribution, not as "known absent." |
| **(d)** data-energy-anchor cure (pin V(data) value; preserves prior **and** improves noise rejection) | **PARTIAL (cite adjacent, keep the specific)** | Du & Mordatch (arXiv:1903.08689) already **L2-regularize energy magnitudes of positive+negative samples** — but for **partition-function/numerical stability**, not to preserve a designed structural prior. EBM-for-continual-learning uses data/anchor regularization for **catastrophic forgetting** (different problem). Ours: a **value-anchor on the designed vacuum's V** as a targeted cure for CD-induced *vacuum* erosion, that also *raises* the noise gap. Cite Du & Mordatch as the nearest energy-regularization precedent; claim the *purpose + mechanism* as ours. |

**GATE RECOMMENDATION: V5 SHIPS.** Two of four claims (b, c) are novel; (a) and (d) are defensible-with-honest-attribution. The honest framing already written in `sleep-erosion-study §4` ("we contribute a sharp, quantified instance + demarcation + cheap cure, not the discovery that CD is biased") is correct and should be preserved verbatim in the V5 draft. **Confidence: medium-high** on the portfolio verdict; the two NOVEL claims are single-sourced negatives — a targeted follow-up (continual-EBM + equilibrium-propagation + RBM-symmetry-breaking literature) should confirm before camera-ready.

## 4. Citation-hygiene (sub-task 6)
- **(a) Wang id — RESOLVED.** Correct paper: **Hongbo Wang (Stony Brook), "Conformal Orbit-Valid Trust Horizons for Equivariant World Models," arXiv:2606.24946** (2026). The handover's `2606.24945` is **wrong** (no such paper surfaced); the Mo-citer's `2606.24946` is right. Sibling paper by (likely) same group: **"Scale Buys Interpolation, Structure Buys a Horizon: Certified Predictability for Equivariant World Models," arXiv:2606.13092**. **Relevance:** V2 (exact equivariance *transports* a calibrated trust-horizon over the group orbit — orbit-constant rollout error) and V3 (horizon **certification** — median certified/measured ratio 0.67). Both are "certification/horizon" neighbors to CHLU's conservation-as-certificate framing; cite as related-work, differentiate on symplectic/energy-ledger mechanism vs conformal calibration.
- **(b) Minami–Hidaka — PINNED.** **Y. Minami & Y. Hidaka, "Spontaneous symmetry breaking and Nambu–Goldstone modes in dissipative systems," Phys. Rev. E 97, 012130 (2018), arXiv:1509.05042.** Follow-up: **"Spontaneous symmetry breaking and Nambu–Goldstone modes in open classical and quantum systems," PTEP 2020, 033A01, arXiv:1907.08241.** **High relevance to V2/V5:** they show dissipation turns **type-A NG modes from propagating → diffusive** (two Noether-charge types). This is *directly* the CHLU story — the γ (dissipation) knob should convert propagating Goldstone modes to diffusive ones; this is the physics precedent for "spending γ" affecting the Goldstone spectrum. Strong cite for V2's dissipative-Goldstone section.
- **(c) Di Bernardo 2511.04802 — CHECKED, no overclaim risk.** Title: **A. Di Bernardo, A. Valente, F. Mastrogiuseppe, S. Ostojic, "Shaping manifolds in equivariant recurrent neural networks" (2025).** It is about **symmetry → geometry of RNN fixed-point manifolds** (group-convolution equivariant RNNs, Fourier low-rank reduction, symmetry-dependent manifold stability). It does **NOT** touch resource/mass/capacity **allocation** — so the F5 note's distinctness claim on the allocation framing is **safe** (no overlap). But it *is* an equivariant-manifold-stability neighbor → cite in V2 related-work; differentiate on Hamiltonian/symplectic vs connectivity-symmetry mechanism.

## 5. V3 reversible-BPTT bib (sub-task 8) — verified strings
All author/venue/id verified this pass:
- **RevNet** — Gomez, Ren, Urtasun, Grosse, "The Reversible Residual Network: Backpropagation Without Storing Activations," **NeurIPS 2017**, **arXiv:1707.04585**.
- **Gradient checkpointing** — Chen, Xu, Zhang, Guestrin, "Training Deep Nets with Sublinear Memory Cost," 2016, **arXiv:1604.06174** (O(√n) memory, one extra forward pass).
- **MomentumNet** — Sander, Ablin, Blondel, Peyré, "Momentum Residual Neural Networks," **ICML 2021**, **arXiv:2102.07870** (invertible ResNet via momentum; second-order ODE interpretation).
- **Canonical reversible-integrator-for-NN ref: NOT PINNED this pass.** Candidates unverified: Chang et al. "Reversible Architectures for Arbitrarily Deep Residual Neural Networks" (AAAI 2018); m-RevNet (arXiv:2108.05862). Recommend citing RevNet + MomentumNet as the reversibility anchors and, if a *symplectic-integrator-reversibility* cite is wanted, Leimkuhler & Reich (Simulating Hamiltonian Dynamics, 2004) — **verify before use**.

## 6. Paid-access prior-art flags (sub-task 7) — quick verdicts
- **(a) Wormhole / nonlocal volume-exact jump certificate:** **NOT SEARCHED adequately this pass** (deprioritized under the time-critical calendar). Preliminary: no one appears to claim *volume-exact (det J=1) + energy-ledgered + latch-transport-certified* nonlocal jumps — the novelty is the **certificate stack**, and generic "nonlocal is good" (attention/skip/MoE-routing) is orthogonal. **Verdict: provisional CLEAR — needs a dedicated pass to confirm.**
- **(b) Squeeze basin-hopping vs Wales–Doye / parallel tempering / stochastic normalizing flows:** **Verdict: CLEAR (mechanism), CROWDED (goal).** All surveyed basin-hopping incl. the "distant-basin" variant (Goodridge & Moriarty, "Hopping between distant basins," J. Glob. Optim. 2022, arXiv:2108.05229) and classic Wales–Doye use **Metropolis / stochastic acceptance**. No **deterministic, symplectic, bounded-injection, governor-re-absorbed (certified, non-Metropolis)** escape found in ML. Cite basin-hopping as the goal-precedent; claim the certified-deterministic mechanism.
- **(c) Relativistic reach caps vs Lipschitz/causal bounds + Lieb–Robinson:** **Verdict: CROWDED (the bound exists), CLEAR (the mechanism-design consequence).** The finite information-propagation speed cap is **Lieb & Robinson (1972)**, heavily developed in quantum many-body (e.g., arXiv:2206.14736, PRL 127.070403). **Cite Lieb–Robinson as the precedent for the causal cap** and claim only that CHLU *designs* the cap in via the relativistic kinetic term (built-in c), rather than proving a bound. (An SSM-specific Lieb–Robinson/receptive-field cite likely exists — not pinned this pass.)

## 7. ICLR 2027 (sub-task 4) — still unannounced
- **iclr.cc/Conferences/2027 and /2027/CallForPapers → HTTP 404** (fetched 2026-07-19). **iclr.cc/Conferences/FutureMeetings says only: "ICLR 2027: West Coast North America"** — no city, no dates. **No CFP exists yet.**
- **The circulating "abstract Sep 19 / paper Sep 24, 2026 / Brazil / notify Jan 22 / April 24–28" numbers are ICLR-2026 data** (confirmed again: aggregators mlciv/waset conflate; waset even lists an ICLR "Aug 2027 Sydney" that contradicts the official "West Coast NA"). **Do not propagate.**
- **Working ESTIMATE (unchanged):** abstract ~Sep 19–27, 2026; paper ~Sep 24–Oct 1, 2026. Pipeline stays policy-clean (workshops = non-archival, ICLR dual-submission exempts workshop-presented work). Re-check iclr.cc the moment the CFP posts (~Aug per prior cycles).

## 8. Bibtex-ready refs (verified this pass)
```bibtex
@inproceedings{gomez2017revnet,
  title={The Reversible Residual Network: Backpropagation Without Storing Activations},
  author={Gomez, Aidan N. and Ren, Mengye and Urtasun, Raquel and Grosse, Roger B.},
  booktitle={NeurIPS}, year={2017}, note={arXiv:1707.04585}}
@article{chen2016sublinear,
  title={Training Deep Nets with Sublinear Memory Cost},
  author={Chen, Tianqi and Xu, Bing and Zhang, Chiyuan and Guestrin, Carlos},
  journal={arXiv preprint arXiv:1604.06174}, year={2016}}
@inproceedings{sander2021momentumnet,
  title={Momentum Residual Neural Networks},
  author={Sander, Michael E. and Ablin, Pierre and Blondel, Mathieu and Peyr{\'e}, Gabriel},
  booktitle={ICML}, year={2021}, note={arXiv:2102.07870}}
@article{nijkamp2020anatomy,
  title={On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models},
  author={Nijkamp, Erik and Hill, Mitch and Han, Tian and Zhu, Song-Chun and Wu, Ying Nian},
  journal={AAAI}, year={2020}, note={arXiv:1903.12370}}
@article{nijkamp2019shortrun,
  title={Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model},
  author={Nijkamp, Erik and Hill, Mitch and Zhu, Song-Chun and Wu, Ying Nian},
  journal={NeurIPS}, year={2019}, note={arXiv:1904.09770}}
@article{du2019implicit,
  title={Implicit Generation and Generalization in Energy-Based Models},
  author={Du, Yilun and Mordatch, Igor}, journal={NeurIPS}, year={2019}, note={arXiv:1903.08689}}
@article{minami2018ssb,
  title={Spontaneous symmetry breaking and Nambu--Goldstone modes in dissipative systems},
  author={Minami, Yuki and Hidaka, Yoshimasa},
  journal={Physical Review E}, volume={97}, pages={012130}, year={2018}, note={arXiv:1509.05042}}
@article{minami2020open,
  title={Spontaneous symmetry breaking and Nambu--Goldstone modes in open classical and quantum systems},
  author={Minami, Yuki and Hidaka, Yoshimasa},
  journal={PTEP}, volume={2020}, number={3}, pages={033A01}, year={2020}, note={arXiv:1907.08241}}
@article{wang2026conformal,
  title={Conformal Orbit-Valid Trust Horizons for Equivariant World Models},
  author={Wang, Hongbo}, journal={arXiv preprint arXiv:2606.24946}, year={2026}}
@article{dibernardo2025shaping,
  title={Shaping manifolds in equivariant recurrent neural networks},
  author={Di Bernardo, Arianna and Valente, Adrian and Mastrogiuseppe, Francesca and Ostojic, Srdjan},
  journal={arXiv preprint arXiv:2511.04802}, year={2025}}
@article{goodridge2022hopping,
  title={Hopping between distant basins},
  author={Goodridge, Maldon and Moriarty, John},
  journal={Journal of Global Optimization}, year={2022}, note={arXiv:2108.05229}}
```
(RBM symmetry-breaking energy-landscape ref seen but not fetched: arXiv:2503.21536 — verify authors/venue before citing.)

## 9. Confidence & gaps
- **CONFIRMED (primary, 2026-07-19):** NeurIPS 2026 workshop timeline + non-archival policy; TS4H Aug-19 deadline; accepted-list not yet public; ICLR 2027 CFP 404 / "West Coast NA" only; all §5/§8 bib ids; Wang id correction; Minami–Hidaka; Di Bernardo scope.
- **SINGLE-SOURCED / OPEN:** V5 claims (b) and (c) are novel-by-absence (couldn't find prior art, not the same as proof of none) — confirm against continual-EBM + equilibrium-propagation before camera-ready. Paid-access flag (a) wormhole not properly searched. Reversible-integrator canonical cite not pinned.
- **NEXT SCOUT PASS (Jul 26–Aug 2):** pull ML4PS/AI4Science/NeurReps 2026 CFPs the moment they post; look for a **DLDE/differential-equations** and an **EBM/generative** workshop in the accepted list (determines V3 and V5 homes); watch iclr.cc for the 2027 CFP; finish V5 novelty confirmation + wormhole prior-art.

## Proposed handover updates (for the Hub)
- **Freeze date:** replace "move freeze to Aug 21–24" with **"plan freeze ≤ Aug 17, submission-ready Aug 18"** — a real 2026 workshop (TS4H) already posts **Aug 19**; our targets may match. Hard-confirm per-target once CFPs post.
- **Calendar:** the Jul-11 date was **organizer** notification; the **public accepted-workshop list is still not posted on Jul 19**. Add checkpoint: **re-spawn venue scout Jul 26–Aug 2** for ML4PS/AI4Science/NeurReps/DLDE/EBM 2026 CFPs.
- **Citation fix (owner needed):** **Wang = arXiv:2606.24946**, not 2606.24945 — patch handover + any V2/V3 draft that inherited the wrong id. Sibling: 2606.13092.
- **New V2/V5 cites:** Minami–Hidaka 1509.05042 + 1907.08241 (dissipative Goldstone: propagating→diffusive under γ — direct CHLU precedent); Di Bernardo 2511.04802 = equivariant-RNN-manifold neighbor (no allocation overlap → F5 note distinctness safe).
- **V5 GO confirmed** by novelty gate: ships with honest "instance + demarcation + cheap cure" framing; (b)/(c) novel, (a)/(d) defensible-with-attribution (cite Fischer–Igel, Nijkamp, Du–Mordatch, RBM-symmetry-breaking 2503.21536).
- **ICLR 2027:** still only "West Coast North America"; the "Brazil/Sept-24/Jan-22" numbers are ICLR-2026 — keep purged from all drafts.

Git footprint: none (read-only; this file only).

Sources:
- [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops)
- [NeurIPS 2026 Schedule (workshops empty)](https://neurips.cc/Conferences/2026/Schedule?type=Workshop)
- [TS4H @ NeurIPS 2026](https://timeseries4health.github.io/)
- [NeurIPS 2026 Blog](https://blog.neurips.cc/category/2026-conference/)
- [ICLR Future Meetings](https://iclr.cc/Conferences/FutureMeetings)
- [RevNet arXiv:1707.04585](https://arxiv.org/abs/1707.04585)
- [Sublinear Memory arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- [MomentumNet arXiv:2102.07870](https://arxiv.org/abs/2102.07870)
- [Nijkamp Anatomy arXiv:1903.12370](https://arxiv.org/abs/1903.12370)
- [Du & Mordatch arXiv:1903.08689](https://arxiv.org/abs/1903.08689)
- [Minami & Hidaka arXiv:1509.05042](https://arxiv.org/abs/1509.05042)
- [Wang, Conformal Orbit-Valid Trust Horizons arXiv:2606.24946](https://arxiv.org/pdf/2606.24946)
- [Di Bernardo et al. arXiv:2511.04802](https://arxiv.org/abs/2511.04802)
- [Goodridge & Moriarty, Hopping between distant basins arXiv:2108.05229](https://arxiv.org/pdf/2108.05229)
- [RBM energy-landscape symmetry breaking arXiv:2503.21536](https://arxiv.org/pdf/2503.21536)
</content>
</invoke>
