# scout-modular-interference — web-scout report
Task + acceptance criterion: verified bib + novelty sniff + one lift-ready paragraph for V3-short §4's modular-vs-monolithic interference related-work (replace the `[·]` placeholders the writer refused to fabricate).
Status: done
Read: protocol · `v3-interference-ntk.md` (the measured claim: cross-unit CD-update basin displacement R; modular ≈2.3e-5 / R_far≡0 vs shared-V_θ ≈0.20 unstructured; ∝κ² slope 1.99; O(1)-vs-O(N)) · `v3-short-draft.md` §4 slot + editorial Q5 · `di-bernardo-skim.md` (guard-rail prose format).
Scope: half-thread, focused pass — NOT a deep sweep. Read-only (no repo/git changes).

---

## ANSWER FIRST
The **ingredients** of V3 §4 are all established and citable (catastrophic interference: McCloskey & Cohen 1989 / French 1999; modular nets & MoE: Jacobs et al. 1991 → Shazeer et al. 2017; parameter-isolation CL: Kirkpatrick et al. 2017 EWC, Mallya & Lazebnik 2018 PackNet; interference *measured as a kernel/gradient object*: Doan et al. 2021 NTK-overlap, Riemer et al. 2019, Yu et al. 2020). But the **specific claim CHLU wants to keep is CLEAR/unclaimed**: nobody I found measures a **cross-module interference kernel with an explicit coupling-strength power law (∝κ²) and an O(1)-vs-O(N)-in-width architectural separation, exactly zero beyond the coupling graph, on a physics-structured lattice**. Verdict: **CLEAR at the specific-claim level, CROWDED at the neighbourhood level (cite the neighbours).** The defensible sentence is exactly the task's: *the firewall is **measured** as a kernel, is O(1)-in-width, decays as κ², and is an exact zero beyond the coupling edges* — prior work either **prevents** interference by construction (modular/MoE/parameter-isolation) or **measures** it only as a diffuse monolithic property to be **mitigated** (NTK-overlap, gradient-conflict); none prices it as a coupling-law.

---

## EVIDENCE (all verified this session: arXiv/DOI + venue)

### Catastrophic interference (the phenomenon)
- **McCloskey & Cohen (1989)**, "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem", *Psychology of Learning and Motivation* **24**:109–165, Elsevier. DOI 10.1016/S0079-7421(08)60536-8. [pre-arXiv; verified via ScienceDirect/Wikipedia lineage] — origin of the term.
- **French (1999)**, "Catastrophic forgetting in connectionist networks", *Trends in Cognitive Sciences* **3(4)**:128–135. DOI 10.1016/S1364-6613(99)01294-2. [verified: ScienceDirect/Cell abstract] — the canonical review; the shared-distributed-representation diagnosis is exactly the "monolithic V_θ moves everywhere" failure V3 measures.

### Modular networks / mixture-of-experts (the classic remedy: don't share)
- **Jacobs, Jordan, Nowlan & Hinton (1991)**, "Adaptive Mixtures of Local Experts", *Neural Computation* **3(1)**:79–87. DOI 10.1162/neco.1991.3.1.79. [verified] — founds modular/MoE; each expert handles a subset ⇒ reduced interference by architecture. Framed there as reducing "interference".
- **Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton & Dean (2017)**, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", **ICLR 2017**, arXiv:1701.06538. [verified: dblp, OpenReview B1ckMDqlg] — the sparse-MoE modern anchor (conditional computation, per-example active sub-network).

### Parameter-isolation continual learning (isolate params ⇒ no cross-task overwrite)
- **Kirkpatrick et al. (2017)**, "Overcoming catastrophic forgetting in neural networks" (EWC), *PNAS* **114(13)**:3521–3526, arXiv:1612.00796. [verified: PNAS + arXiv] — regularization (not strict isolation) that *slows* interference on important weights; contrast: soft, not a firewall.
- **Mallya & Lazebnik (2018)**, "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning", **CVPR 2018**, pp.7765–7773, arXiv:1711.05769. [verified: dblp MallyaL18, CVF open-access] — hard parameter isolation via pruning masks ⇒ zero-forgetting by construction. The cleanest "isolation = firewall" prior; contrast with CHLU is *measured κ²-leak* vs binary mask.
- (optional lineage) **Rusu et al. (2016)**, "Progressive Neural Networks", arXiv:1606.04671 — column-per-task isolation with lateral connections; adjacent, cite only if space.

### Interference *measured* as a kernel / gradient object (closest prior — the novelty boundary)
- **Doan, Bennani, Mazoure, Rabusseau & Alquier (2021)**, "A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix", **AISTATS 2021**, PMLR 130, arXiv:2010.04003. [verified] — **most important neighbour.** Defines task-similarity/forgetting via an **NTK overlap matrix**; projected-gradient (OGD/PCA-OGD) mitigates it. CHLU differs: (i) an **inter-module** kernel measured through training on a physics lattice, not an inter-task NTK overlap on a monolith; (ii) a **coupling-law (∝κ², exact-0 off-graph)**, not a data-similarity overlap; (iii) V3's own report *warns raw NTK cosine ≈0.99 fails to distinguish* — the firewall lives in the wake−sleep-difference basin displacement R, not the raw kernel. Cite as the "measured but on a monolith, as an overlap not a coupling-law" foil.
- **Riemer et al. (2019)**, "Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference" (MER), **ICLR 2019**, arXiv:1810.11910. [verified: dblp RiemerCALRTT19, OpenReview B1gTShAct7] — casts CL as a transfer/interference trade-off via **gradient alignment (inner products)**; interference = negative gradient overlap, to be minimized by meta-learning. Diffuse, optimizer-side, monolithic — not an architectural κ-law.
- **Yu, Kumar, Gupta, Levine, Hausman & Finn (2020)**, "Gradient Surgery for Multi-Task Learning" (PCGrad), **NeurIPS 2020**, arXiv:2001.06782. [verified] — defines interference as **conflicting (negative-cosine) gradients** and projects them away. Multi-task, not modular-memory; interference is a nuisance to surgically remove, not a priced coupling.
- **Ortiz-Jimenez, Favero & Frossard (2023)**, "Task Arithmetic in the Tangent Space", **NeurIPS 2023**, arXiv:2305.12827. [verified] — weight-disentanglement / NTK-linearized task editing; interference ↔ non-disentangled directions. Cite if §4 touches task-arithmetic/NTK.
- Background: **Jacot, Gabriel & Hongler (2018)**, "Neural Tangent Kernel: Convergence and Generalization in Neural Networks", **NeurIPS 2018**, arXiv:1806.07572. [verified: dblp JacotHG18, NeurIPS proceedings] — the NTK object V3's Θ specializes.

### Adjacent modular-scaling papers (novelty-sniff neighbours, cite optionally)
- **Boopathy, Jiang, Yue, Hwang, Iyer & Fiete (2025)**, "Breaking Neural Network Scaling Laws with Modularity", **ICLR 2025**, arXiv:2409.05780. [verified: OpenReview 5Qxx5KpFms] — **modular sample-complexity becomes independent of task dimensionality where nonmodular is exponential.** This is the closest "modularity ⇒ scaling separation" result, but it is a *sample-complexity* (generalization) separation, **not** a measured interference-kernel / O(N)-vs-O(1) *cross-talk* law. Strong neighbour for the scaling framing; differentiate on "we measure cross-unit interference, they bound sample complexity."
- **"Studying Cross-cluster Modularity in Neural Networks"**, arXiv:2502.02470 (2025) — a "clusterability loss" regularizer encouraging non-interference between learned clusters; finds interference *rises* with cluster count. Adjacent (emergent, learned clusters; no coupling-law); cite only if you want a "measured interference rises without a firewall" foil. [title/id verified; not deep-read]

---

## NOVELTY SNIFF — verdict per sub-claim
| CHLU sub-claim | Verdict | Owner / nearest |
|---|---|---|
| "interference between modules exists & hurts" | **CROWDED** | French 1999; Riemer 2019; Yu 2020 |
| "modularity / isolation prevents interference" | **CROWDED** | Jacobs 1991; Shazeer 2017; Mallya 2018 (PackNet) |
| "interference is measurable as a kernel during training" | **CROWDED(cite Doan)** | Doan et al. 2021 (NTK-overlap) — but on a monolith, as data-overlap |
| "modularity gives an *architectural scaling separation*" | **CROWDED(cite Boopathy)** | Boopathy et al. 2025 — but sample-complexity, not cross-talk |
| **"measured cross-module interference kernel with ∝κ² coupling power-law, exact-0 beyond the coupling graph, O(1)-vs-O(N)-in-width, on a physics lattice, persisting through training"** | **CLEAR (unclaimed)** | none found |

**Net verdict: keep the specific claim; it is defensible.** Frame novelty as the *conjunction*: measured-kernel × coupling-law(κ²) × exact-off-graph-zero × O(1)-width × physics-structured-lattice. Do **not** claim to have invented "interference" or "modularity ⇒ isolation" — those are owned. The one-line owner-check the task asked for: **nobody owns "the firewall is measured on a physics-structured lattice with an exact zero beyond the coupling graph and a κ² leak."** That phrasing is clear.

Confidence: **HIGH** on all bib entries (each authors/year/venue/id verified against ≥1 primary index: arXiv/dblp/proceedings/DOI). Jacot arXiv-id 1806.07572 verified via dblp+NeurIPS page (id itself from canonical knowledge, not echoed in snippet — standard & correct). **MEDIUM** on the exhaustiveness of the novelty sniff (focused pass, not a systematic sweep; searched cross-module interference-kernel + coupling-law + modular-scaling neighbourhoods — a dedicated "interference kernel between modules measured during training" prior did **not** surface, but absence-of-evidence caveat applies). McCloskey & Cohen / French are pre/paywalled — DOIs verified, full text not fetched (uncontroversial classics).

---

## LIFT-READY PARAGRAPH (guard-railed — splice into V3-short §4; real citations, no over-claim)

> **Modular vs. monolithic parameter sharing.** Catastrophic interference — the abrupt overwriting of stored function when a distributed network learns new content — has been understood since McCloskey & Cohen [McCloskey1989] and French [French1999] as a consequence of *shared* representations. Two families of remedy exist. The first **prevents** interference by construction: modular architectures and mixtures-of-experts route different content to different parameters [Jacobs1991, Shazeer2017], and parameter-isolation continual learners freeze or mask per-task subnetworks [Kirkpatrick2017, Mallya2018]. The second **measures** interference so it can be minimized during optimization: the NTK-overlap matrix quantifies cross-task forgetting [Doan2021], and gradient-alignment / gradient-surgery methods treat conflicting (negative-inner-product) updates as the interference signal to suppress [Riemer2019, Yu2020]. Our lattice sits at the intersection of the two: interference between units is not merely prevented or globally minimized but **measured as a coupling-resolved kernel through training.** In a shared-potential (monolithic) CLU-Net a single unit's contrastive update displaces every other unit's basin by ≈20% of the intended change, with no spatial structure and growing linearly in width; in the modular lattice the same update displaces a neighbour by ≈2×10⁻⁵ and a non-adjacent unit by *exactly* zero, with the residual nearest-neighbour leak scaling as the square of the coupling strength (∝κ², exactly zero at κ=0). The interference thus obeys an architectural **O(1)-in-width vs. O(N)** separation and an explicit coupling power-law — a *priced*, graph-local firewall rather than the diffuse, data-dependent overlap measured on monolithic networks [Doan2021] or the sample-complexity separation established for modular tasks [Boopathy2025]. *(Scope: 2-dim units, chain topology, MLP potentials, N≤8, learned potentials, laptop-CPU; see App. A.)*

**Guard-rails applied** (per di-bernardo-skim precedent): (i) no claim to have invented interference or modular isolation — those cite the owners; (ii) the numbers are stated as *measured on this lattice* with the in-sentence scope qualifier, not as universal law; (iii) Doan and Boopathy are named as the sharp differentiators (measured-but-monolithic-overlap; scaling-but-sample-complexity), so a reviewer sees exactly what is new; (iv) uses the report's own metric discipline (basin-displacement R, not raw NTK cosine — App. D already carries this). Drafter: swap `[Key]` tokens for the bibkeys below; verify the 20% / 2e-5 / κ² / N numbers against the approved CM-9 wording before shipping (they match `v3-interference-ntk` items 1/3).

---

## BIBTEX-READY REFS
```bibtex
@incollection{McCloskey1989,
  author = {McCloskey, Michael and Cohen, Neal J.},
  title = {Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem},
  booktitle = {Psychology of Learning and Motivation}, volume = {24}, pages = {109--165},
  publisher = {Academic Press}, year = {1989}, doi = {10.1016/S0079-7421(08)60536-8}}

@article{French1999,
  author = {French, Robert M.}, title = {Catastrophic forgetting in connectionist networks},
  journal = {Trends in Cognitive Sciences}, volume = {3}, number = {4}, pages = {128--135},
  year = {1999}, doi = {10.1016/S1364-6613(99)01294-2}}

@article{Jacobs1991,
  author = {Jacobs, Robert A. and Jordan, Michael I. and Nowlan, Steven J. and Hinton, Geoffrey E.},
  title = {Adaptive Mixtures of Local Experts}, journal = {Neural Computation},
  volume = {3}, number = {1}, pages = {79--87}, year = {1991}, doi = {10.1162/neco.1991.3.1.79}}

@inproceedings{Shazeer2017,
  author = {Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
  title = {Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2017},
  note = {arXiv:1701.06538}}

@article{Kirkpatrick2017,
  author = {Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A. and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and Hassabis, Demis and Clopath, Claudia and Kumaran, Dharshan and Hadsell, Raia},
  title = {Overcoming catastrophic forgetting in neural networks},
  journal = {Proceedings of the National Academy of Sciences}, volume = {114}, number = {13},
  pages = {3521--3526}, year = {2017}, doi = {10.1073/pnas.1611835114}, note = {arXiv:1612.00796}}

@inproceedings{Mallya2018,
  author = {Mallya, Arun and Lazebnik, Svetlana},
  title = {PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {7765--7773}, year = {2018}, note = {arXiv:1711.05769}}

@inproceedings{Doan2021,
  author = {Doan, Thang and Bennani, Mehdi Abbana and Mazoure, Bogdan and Rabusseau, Guillaume and Alquier, Pierre},
  title = {A Theoretical Analysis of Catastrophic Forgetting through the {NTK} Overlap Matrix},
  booktitle = {Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series = {PMLR}, volume = {130}, year = {2021}, note = {arXiv:2010.04003}}

@inproceedings{Riemer2019,
  author = {Riemer, Matthew and Cases, Ignacio and Ajemian, Robert and Liu, Miao and Rish, Irina and Tu, Yuhai and Tesauro, Gerald},
  title = {Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2019},
  note = {arXiv:1810.11910}}

@inproceedings{Yu2020,
  author = {Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  title = {Gradient Surgery for Multi-Task Learning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2020},
  note = {arXiv:2001.06782}}

@inproceedings{OrtizJimenez2023,
  author = {Ortiz-Jimenez, Guillermo and Favero, Alessandro and Frossard, Pascal},
  title = {Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2023},
  note = {arXiv:2305.12827}}

@inproceedings{Jacot2018,
  author = {Jacot, Arthur and Gabriel, Franck and Hongler, Cl{\'e}ment},
  title = {Neural Tangent Kernel: Convergence and Generalization in Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2018},
  note = {arXiv:1806.07572}}

@inproceedings{Boopathy2025,
  author = {Boopathy, Akhilan and Jiang, Sunshine and Yue, William and Hwang, Jaedong and Iyer, Abhiram and Fiete, Ila},
  title = {Breaking Neural Network Scaling Laws with Modularity},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2025},
  note = {arXiv:2409.05780}}
```
Optional (cite only if space): `@article{Rusu2016, author={Rusu, Andrei A. and Rabinowitz, Neil C. and Desjardins, Guillaume and Soyer, Hubert and Kirkpatrick, James and Kavukcuoglu, Koray and Pascanu, Razvan and Hadsell, Raia}, title={Progressive Neural Networks}, year={2016}, note={arXiv:1606.04671}}`.

---

## Open questions / follow-ups / risks
- **Single-sourced / could-not-verify:** none load-bearing. McCloskey&Cohen 1989 and French 1999 verified by DOI + secondary indices only (full text not fetched — uncontroversial classics). Jacot arXiv id 1806.07572 not echoed in snippet text (dblp+NeurIPS page confirm the paper; id is canonical).
- **Novelty risk (absence-of-evidence):** the "measured cross-module interference kernel with a κ² coupling-law" claim came back CLEAR, but this was a focused pass. If a referee wants harder assurance, next searches: (i) OpenReview ICLR/NeurIPS 2024–2026 full-text for "interference kernel" + "modular"; (ii) Semantic Scholar citation graph *forward* from Doan 2021 (who has since built an inter-module coupling-law on top of NTK-overlap?); (iii) the recurrent/associative-memory line (Hopfield/modern-Hopfield capacity-under-coupling) for a physics-adjacent κ-scaling of cross-pattern crosstalk — that is the likeliest place a "coupling ⇒ crosstalk power-law" prior would hide.
- **Metric-discipline reminder to the drafter (from `v3-interference-ntk` 1(a)):** cite/report the basin-displacement R, never the raw NTK cosine (≈0.99 for both architectures, uninformative). The lift paragraph already does this; keep it if edited.

## Proposed handover updates (for the Hub)
- **V3-short §4 modular-interference slot: bib gap CLOSED.** 11 verified refs + 1 lift-ready guard-railed paragraph in `.claude/outputs/scout-modular-interference.md`. Novelty verdict: **specific claim CLEAR** (measured cross-module interference kernel × κ²-law × exact-off-graph-zero × O(1)-vs-O(N) × physics-lattice — unclaimed), **neighbourhood CROWDED** (cite Doan 2021 NTK-overlap and Boopathy 2025 modular-scaling as the two sharp differentiators; McCloskey/French/Jacobs/Shazeer/Kirkpatrick/Mallya/Riemer/Yu as background).
- **Sharpest foils to name in §4:** Doan et al. 2021 (interference *measured* but as a data-dependent NTK-overlap on a monolith, not a coupling-law) and Boopathy et al. 2025 (modularity ⇒ *sample-complexity* scaling separation, not a *cross-talk* separation). CHLU is the conjunction neither owns.
- **Claims-matrix:** the §4 "V3 modular-related-work bib" open slot (draft's Proposed-updates line) can be marked resolved; bibkeys above are ready to drop into the .tex.
