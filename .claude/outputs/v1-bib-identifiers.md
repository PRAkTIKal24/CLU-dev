# v1-bib-identifiers — web-scout report

Task + acceptance criterion: produce the Zotero-pasteable identifier block for V1's `refs.bib` (17 works), a coverage table, a residual list, and the cite-or-cut list for the six uncited orphans. **Zero guessed identifiers.**
Status: **done**
**DIAL DECLARATION (echoed): NONE — read-only literature lookup producing one report. No experiment, no configuration change, no paper file edited.**

> ## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — first-10-lines rule, AGENT_PROTOCOL §5)
> 1. ⛔⛔ **`Angelopoulos et al.` is now a 2025 journal article.** `Learn then Test` was published as **Ann. Appl. Stat. 19(2), 2025, DOI `10.1214/24-AOAS1998`** (Crossref direct-resolve, 5 authors match). The paper prints **"(2021)"** at l.130 and l.363. **If the Head pastes the DOI, Zotero emits a 2025 entry and the printed year silently breaks in two places.** I put the **arXiv ID in the block** (year-preserving) and give the DOI as a labelled swap. **Head must rule.**
> 2. ⛔⛔ **`Platt (1999)` has NO DOI, and the DOI that *does* exist is for a differently-titled work.** The MIT Press chapter at the same pages is titled **"Probabilities for SV Machines" (2000)**, `10.7551/mitpress/1113.003.0008`. Pasting it changes both the title and the year. **Residual — hand-entry. Head must rule.**
> 3. ⚠ **`Platt` is a SEVENTH near-orphan the task list did not name.** The surname appears in prose exactly **once**, inside the A.2 flag-provenance **table cell** (l.202, *"Platt-calibrated logit margin"*) — not in a sentence. If the cite pass sweeps prose only, Platt drops with the six.
> 4. ⚠ **`Wales & Doye` title is truncated in the reference list.** The paper prints *"Global Optimization by Basin-Hopping."*; the published title is *"Global Optimization by Basin-Hopping **and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms**"*. Zotero will emit the full title.
> 5. ⚠ **Two reference entries are placeholders, not references:** l.369 `Jawahar, P., Pierini, M. (2026). [CHLU primitive].` and l.379 `Anonymous (2026). [The theory note].` Both print square-bracket placeholder titles today.
> 6. ⚠ **Double-blind artifact (report, do not propagate):** l.35 reads *"**Our** reference memory is the Causal Learning Unit (CLU), introduced as CHLU in **Jawahar & Pierini (2026)**."* The possessive + named self-cite is a de-facto identification — identical to the flag `v2-cite-check.md` §26 raised for V2. Also l.17 `[WORKING TITLE: ...]` and l.18 `[AUTHORS PLACEHOLDER]` are unresolved.

---

## 0. File identity (⚠ no shell — md5 not computable)

⛔ **I have no shell in this session.** My toolset is Read / Grep / Glob / WebSearch / WebFetch / Write only. **I cannot compute an md5 or any file hash.** Declaring that rather than inventing one. Fingerprint of the file as I read it:

| field | value |
|---|---|
| path | `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` |
| read at | **2026-08-27**, single read, full file |
| total lines | **383** |
| l.1 / l.383 | `\documentclass[11pt]{article}` / *(blank; `\end{document}` at l.382)* |
| reference block | `\section*{References}` at **l.360**, `\begin{itemize}` l.362, entries **ll.363–379**, `\end{itemize}` l.380 |
| entries | **17** (one `\item` each, ll.363–379) — matches the task's count exactly |
| ⛔ writes by me | **none** to `NIPSsubmission/` or `papers/`. Only `.claude/outputs/v1-bib-identifiers.md`. |

Retrieval date for every identifier below: **2026-08-27**.

---

## Deliverable 1 — ⭐ THE BLOCK

**15 identifiers, one per line, in §1's order** (works 1–5 carried, then 6–16). The **2 residuals (Platt #12, Anonymous #17) are omitted** — the block is therefore NOT positionally aligned with the 17-item list; use Deliverable 2 to map back.

DOI bare where one exists; arXiv `NNNN.NNNNN` otherwise. **Zero guessed identifiers.**

```
2507.02092
1603.08983
2404.02258
2008.02217
2603.01768
2110.01052
2107.05407
10.1016/0370-2693(87)91197-X
1705.08500
10.1007/BF01645779
10.1201/b10905-6
10.2307/3318418
10.52202/068431-1269
1701.06538
10.1021/jp970984n
```

**Format notes for the Head.**
- DOIs are case-insensitive. `10.1016/0370-2693(87)91197-X` is given in **publisher (upper-case X)** form; Crossref echoes it lower-case. Both resolve. Keep the **literal parentheses** — do not escape them.
- `10.1201/b10905-6` contains a hyphen that is part of the DOI, not a line-break.
- Line 6 (`2110.01052`, Angelopoulos) is a **deliberate deviation from the DOI-preferred rule** — see reconciliation item 1. Swap to `10.1214/24-AOAS1998` **only together with** a `2021 → 2025` edit at l.130 and l.363.
- Line 13 (`10.52202/068431-1269`, Schuster/CALM) is the **NeurIPS-Foundation proceedings DOI**. It emits *year 2022, "Advances in Neural Information Processing Systems 35", pp. 17456–17472* — i.e. **no year drift** and it matches the paper's own "NeurIPS" label, which is why the DOI is safe here where it is not for Angelopoulos. arXiv fallback if the Head prefers uniformity: `2207.07061`.

---

## Deliverable 2 — coverage table

`CARRIED` = from §1's five (title re-confirmed this pass, not re-derived) · `NEW` = resolved by me · every returned title quoted verbatim.

| # | work as the paper cites it | identifier | C/N | source + **returned title, quoted** |
|---|---|---|---|---|
| 1 | Gladstone et al. (2025) | `2507.02092` | **CARRIED**, title confirmed | arXiv API `id_list`: **"Energy-Based Transformers are Scalable Learners and Thinkers"**, 10 authors, first = Alexi Gladstone, 2025-07-02. No journal-ref, no DOI. Crossref bibliographic query returned **no** matching record (see traps). |
| 2 | Graves (2016), ACT | `1603.08983` | **CARRIED**, title confirmed | arXiv API: **"Adaptive Computation Time for Recurrent Neural Networks"**, **single author Alex Graves**, 2016-03-29. ⛔ *This is the failure-mode-2 case the task named: `1410.5401` is Neural Turing Machines, a 3-author paper. The IDs are distinct and both are correct for their own works.* |
| 3 | Raposo et al. (2024), MoD | `2404.02258` | **CARRIED**, title confirmed | arXiv API: **"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"**, Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro, 2024-04-02. ⚠ paper prints the short form *"Mixture-of-Depths"*; Zotero emits the full subtitle. |
| 4 | Ramsauer et al. (2021), ICLR | `2008.02217` | **CARRIED**, title confirmed | arXiv API: **"Hopfield Networks is All You Need"**, 16 authors, first = Hubert Ramsauer, 2020-07-16. No DOI (ICLR mints none). |
| 5 | Jawahar & Pierini (2026) | `2603.01768` | **CARRIED**, title confirmed | arXiv API: **"CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning"**, Jawahar & Pierini, 2026-03-02, comments verbatim **"Accepted as a short paper at ICLR 2026 (AI & PDE)"**. No DOI. |
| 6 | Angelopoulos, Bates et al. (2021), LTT | `2110.01052` ⚠ | **NEW** | arXiv API: **"Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control"**, Angelopoulos, Bates, Candès, Jordan, Lei; v5, published 2021-10-03, updated 2022-09-29; **arXiv record carries NO journal-ref**. ⛔ Crossref direct-resolve of `10.1214/24-AOAS1998` returns **"Learn then test: Calibrating predictive algorithms to achieve risk control"**, *The Annals of Applied Statistics* **19(2), 2025**, same 5 authors ⇒ **the venue record exists and is 2025**. |
| 7 | Banino, Balaguer, Blundell (2021), PonderNet | `2107.05407` | **NEW** | arXiv API: **"PonderNet: Learning to Ponder"**, Banino, Balaguer, Blundell, 2021-07-12, comments **"16 pages, 2 figures, 2 tables, 8th ICML Workshop on Automated Machine Learning (2021)"**. Crossref: **no DOI** (see traps). |
| 8 | Duane, Kennedy, Pendleton, Roweth (1987), Phys. Lett. B | `10.1016/0370-2693(87)91197-X` | **NEW** | Crossref **direct DOI re-resolve**: **"Hybrid Monte Carlo"**, *Physics Letters B*, Simon Duane, A.D. Kennedy, Brian J. Pendleton, Duncan Roweth, **1987, 195(2):216–222**. |
| 9 | Geifman & El-Yaniv (2017), NeurIPS | `1705.08500` | **NEW** | arXiv API: **"Selective Classification for Deep Neural Networks"**, Yonatan Geifman, Ran El-Yaniv, 2017-05-23. **No DOI** — NeurIPS 2017 (= NeurIPS vol. 30) has **no Crossref-registered proceedings volume** (MIT Press covers vols ≤19, the NeurIPS Foundation `10.52202` prefix starts at vol. 35). |
| 10 | Lieb & Robinson (1972), Comm. Math. Phys. | `10.1007/BF01645779` | **NEW** | Crossref **direct DOI re-resolve**: **"The finite group velocity of quantum spin systems"**, *Communications in Mathematical Physics*, Elliott H. Lieb & Derek W. Robinson, **1972, 28(3):251–257**. |
| 11 | Neal (2011), Handbook of MCMC | `10.1201/b10905-6` | **NEW** | Crossref **direct DOI re-resolve**: **"MCMC Using Hamiltonian Dynamics"**, *Handbook of Markov Chain Monte Carlo*, Radford M. Neal, **2011**, Chapman and Hall/CRC, **pp. 113–162**, type `book-chapter`. |
| 12 | Platt (1999) | — | **RESIDUAL (D3-1)** | No DOI on any of three surfaces. See below. |
| 13 | Roberts & Tweedie (1996), Bernoulli | `10.2307/3318418` | **NEW** | Crossref **direct DOI re-resolve**: **"Exponential Convergence of Langevin Distributions and Their Discrete Approximations"**, *Bernoulli*, Gareth O. Roberts & Richard L. Tweedie, **1996, 2(4)**. ⚠ Crossref's `page` field is **"341" only**; **second instrument** (Project Euclid, Bernoulli vol. 2 iss. 4) gives the true range **341–363**. Hand-fix the end page. |
| 14 | Schuster et al. (2022), CALM, NeurIPS | `10.52202/068431-1269` | **NEW** | Crossref **direct DOI re-resolve**: **"Confident Adaptive Language Modeling"**, *Advances in Neural Information Processing Systems 35*, Schuster, Fisch, Gupta, Dehghani, Bahri, Tran, Tay, Metzler, **2022, pp. 17456–17472**, publisher NeurIPS Foundation. Corroborated by arXiv **2207.07061**, comments **"NeurIPS 2022 (selected as Oral)"** ⇒ **two-instrument**. |
| 15 | Shazeer et al. (2017), ICLR | `1701.06538` | **NEW** | arXiv API: **"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"**, Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean, 2017-01-23. **No DOI** (ICLR mints none; Crossref title query returned only unrelated works). |
| 16 | Wales & Doye (1997), J. Phys. Chem. A | `10.1021/jp970984n` | **NEW** | Crossref **direct DOI re-resolve**: **"Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms"**, *The Journal of Physical Chemistry A*, David J. Wales & Jonathan P. K. Doye, **1997, 101(28):5111–5116**. ⚠ title truncated in the paper — reconciliation item 4. |
| 17 | Anonymous (2026), the theory note | — | **RESIDUAL (D3-2)** | ⛔ No identifier exists and none may be invented. Head decision, not a lookup. |

**Tally: 15 in block (5 CARRIED + 10 NEW) · 2 residual · 0 guessed.**

**Preprint-vs-venue disclosure (§5 note).** Only **two** of the 17 have both a preprint and a venue record where the choice moves the year:
- **#6 Angelopoulos** — paper cites the **2021 preprint**; a **2025 AoAS** record now exists. **Year drift if the DOI is used.**
- **#14 Schuster** — paper cites **"(2022), NeurIPS"**; the proceedings DOI is also **2022**. **No drift.** Safe to use the DOI.
All other arXiv-only entries (Gladstone, Graves, Raposo, Ramsauer, Jawahar, Banino, Geifman, Shazeer) have **no venue DOI at all**, so no drift is possible.

---

## Deliverable 3 — the residual list (hand-entry required)

### D3-1 — Platt (1999). ⛔ **NO DOI EXISTS for the work as cited.**

Three independent surfaces, all negative for the cited title:
- **Crossref bibliographic query** for the full title + book + year: returned **five other chapters** of *Advances in Large-Margin Classifiers* (Weston, Mangasarian, Wahba, Evgeniou) and **Platt's chapter under a DIFFERENT TITLE** — never the cited title.
- **Semantic Scholar Graph API**: the record **"Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods", J. Platt, 1999** exists, `externalIds` = **`MAG: 1618905105`, `CorpusId: 56563878` — no DOI, no ArXiv, no DBLP field**.
- **DBLP** title search: **no record of Platt's original** (only Lin, Lin & Weng 2007, *"A note on Platt's probabilistic outputs for support vector machines"*, and an unrelated twin-SVM paper).

Positive control: all three surfaces returned DOIs for other works in the same session, so these negatives are method-valid.

```
author    = John C. Platt
title     = Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods
booktitle = Advances in Large Margin Classifiers
publisher = MIT Press
pages     = 61--74          <- VERIFIED (MIT Press chapter record, same pages)
year      = 1999            <- as the paper cites it; the MIT Press book record says 2000
note      = No DOI.
```
⚠ **Do NOT enter the "10(3)" volume/issue** that circulates with this reference in the wild — *Advances in Large Margin Classifiers* is a book, not a journal; that field is a propagated miscitation. I did not find a registry that carries it.
⚠ The tech-report number commonly given as **MSR-TR-98-14** was **NOT verified this pass** — do not enter it without a check.

⛔ **The near-miss the Head must rule on.** A DOI *does* exist for what is almost certainly the same work: `10.7551/mitpress/1113.003.0008` → Crossref direct-resolve returns **"Probabilities for SV Machines"**, *Advances in Large-Margin Classifiers*, **John C. Platt, 2000**, MIT Press, **pp. 61–74**, `book-chapter`. Same author, same book, same pages — but a **different title and a different year**. Pasting it silently rewrites the entry from *"Platt, J. (1999). Probabilistic Outputs for Support Vector Machines"* to *"Platt, J. C. (2000). Probabilities for SV Machines"*. **I did not substitute it.** Head rules: (a) hand-enter the fields above, or (b) accept the DOI and its title/year.

### D3-2 — Anonymous (2026), the theory note.
⛔ **No identifier exists and none may be invented.** This is a **Head decision, not a lookup.** It is referenced in prose at **l.35** (*"The theoretical proofs for the verifiable certificates are detailed in a companion theoretical note (Anonymous, 2026)"*) and again at **l.50** (*"Complete derivations and machine-precision validations are provided in the companion theory note"*) — i.e. **it is load-bearing, not decorative**, exactly as the task states.
```
author = Anonymous ; year = 2026
title  = [The theory note]        <- currently a square-bracket PLACEHOLDER in the .tex (l.379)
note   = Anonymous companion note, provided in the supplementary material.
```
⛔ The entry **must not resolve to a named preprint** under a double-blind venue. Program precedent (`v2-cite-check.md` §27, `v2-bib-doi-list.md` D3-2) is identical.

---

## Deliverable 4 — ⭐ THE CITE-OR-CUT LIST (the six orphans)

**Method + positive control.** Per-file `Grep -n -o` over `pj_sub.tex` with a surname alternation covering all 17 works. **Positive controls fired**: `Graves|Banino|Schuster|Raposo|Shazeer|Gladstone` → **l.30** (prose), `Jawahar|Anonymous` → **l.35**, `Angelopoulos` → **l.130**, `Ramsauer` → **l.132**. So the negatives are real, not a directory-grep artefact (MEMORY: `.claude/` directory-level Grep is unreliable). Result: **Duane, Kennedy, Pendleton, Roweth, Geifman, El-Yaniv, Lieb, Robinson, Neal, Roberts, Tweedie, Wales, Doye appear ONLY inside the reference list (ll.365–378).** Confirms the task's six exactly. ⚠ Plus **Platt**, whose only prose-side occurrence is a **table cell** (l.202) — reconciliation item 3.

⛔ **I recommend only. The Head rules attach-or-cut.**

| # | orphan | ⭐ proposed citation site — **section + quoted sentence** | why it fits | strength |
|---|---|---|---|---|
| 8 | **Duane, Kennedy, Pendleton & Roweth (1987)** | **App. F, "Derivations for Certified Markov Kernels", l.352 (opening):** *"To secure a stationarity certificate for test-time retries, we derive the four operating constraints for the proposed Markov kernel."* — and, in the same paragraph, the sentence that **is** the HMC construction: *"A retry proposing a sign-symmetrized squeeze and accepting via $\min(1,e^{-\Delta H/T})$ forms a detailed-balance kernel for $e^{-H/T}$."* | A Hamiltonian-dynamics proposal accepted by a Metropolis ratio on $e^{-H/T}$ **is** Hybrid/Hamiltonian Monte Carlo. This is the single strongest attach in the list — the paper is re-deriving Duane et al. without naming it. | **ATTACH — very strong.** Cutting would leave the appendix reinventing HMC uncredited, which a referee will flag. |
| 11 | **Neal (2011)** | **App. F, l.354 (whole sentence):** *"Second, the squeeze family orbit represents a single hyperbola, rendering it non-ergodic. Ergodicity strictly necessitates momentum refreshment, requiring the squeeze to be layered with a Metropolis-Adjusted Langevin Algorithm step to ensure proper mixing across the state space."* | Neal's chapter is the canonical modern treatment of **momentum refreshment and ergodicity** in Hamiltonian MCMC; it is the standard citation for exactly this argument. Pairs naturally with Duane at l.352. | **ATTACH — strong.** Secondary site: §5 l.157 (*"...acts as a Markov Chain Monte Carlo method with a strict stationarity certificate."*). |
| 13 | **Roberts & Tweedie (1996)** | **§5 "Design rules for certified Markov kernels", l.161:** *"Second, because the squeeze family orbit is reducible and non-ergodic, it must be layered onto a Metropolis-Adjusted Langevin Algorithm (MALA) step to ensure adequate mixing."* | R&T 1996 is **the** convergence reference for Langevin diffusions and their discretisations (ULA/MALA) — the paper names MALA in main text with no source. | **ATTACH — strong.** Second live site: App. F l.356 (*"the Langevin noise scale must be calibrated using the discrete fluctuation-dissipation relation..."*) — R&T is the discretisation-bias antecedent there. |
| 9 | **Geifman & El-Yaniv (2017)** | **§4.1 l.130:** *"We apply a Learn-then-Test (LTT; Angelopoulos et al. 2021) wrapper to determine exit thresholds based on a dynamic relaxation ladder."* — append the selective-prediction antecedent next to LTT. | The compute-rationing gate **is** a selective-prediction / reject-option classifier; Geifman & El-Yaniv is its deep-learning origin, and it sits one clause from the risk-control wrapper that already cites Angelopoulos. | **ATTACH — strong.** Alternative site: **Intro l.30**, in the existing family list *"confidence-gated early exits (CALM, Schuster et al. 2022)"* — a one-clause insertion. |
| 16 | **Wales & Doye (1997)** | **§3.1 l.73 (opening):** *"To cure energetic escape failure, we apply a mass-weighted Lorentz squeeze $S^{(M)}_\zeta$."* | The squeeze is a **perturb-then-relax move that escapes a barrier into another basin** — that is the basin-hopping move. The paper already uses the vocabulary: §1 l.32 *"the governor re-absorption and **basin-hop** thresholds are verified to leading order"*, and §3.2 l.85 constructs *"a controlled reach battery across a $K$-**basin** double well"*. | **ATTACH — moderate/strong.** Secondary site: §1 l.32, where the word "basin-hop" already appears bare. |
| 10 | **Lieb & Robinson (1972)** | **§2 "Falsifiable reachability constraints", l.57:** *"The resulting causal box $C_T$ is entirely energy-blind, as injected momentum asymptotically approaches, but cannot exceed, the velocity limit $c/\sqrt{M_i}$."* — or the box definition immediately above it (l.53–56). | The paper's own vocabulary invites it: §1 l.35 *"the learned inertial mass $M_i$, which dictates the **light-cone** bounds"*. L&R is the canonical "emergent finite group velocity ⇒ effective light cone" result. | ⚠ **ATTACH AS ANALOGY ONLY — weakest of the six.** L&R is a **quantum spin-lattice** theorem; V1's causal box is a **classical kinematic** bound derived from the relativistic kinetic cap. Attach with an explicit hedge (*"cf."* / *"the analogous statement for lattice quantum systems"*). ⛔ **Do NOT let the cite read as though L&R derives $C_T$** — a physics referee will catch that, and it would be a mis-attributed theorem of exactly the class `v2-cite-check` reconciliation item 1 caught for V2. **If the Head is unwilling to add the hedge, CUT this one** — it is the only genuine cut candidate. |

**Summary recommendation: attach 5 (Duane, Neal, Roberts, Geifman, Wales), attach-with-hedge-or-cut 1 (Lieb & Robinson).** All six attach to sentences that already exist; none requires new prose beyond a parenthetical. Plus: **convert l.202's bare "Platt-calibrated" into a `\cite`** or Platt drops with them.

---

## Deliverable 5 — ⛔ NEVER-COPY TRAPS (recorded so the next pass does not repeat them)

**T1 — ⛔⛔ Crossref's *query* endpoint returned confident garbage four times this pass.** Live demonstrations, all with plausible-looking DOIs:
| query | Crossref's top hits |
|---|---|
| `PonderNet Learning to Ponder` | **"Ponder"** (Johann Jakob Spreng *Glossarium*, `10.24894/...`), **"Ponder"** (*Ordinary Blessings for the Christmas Season*, `10.2307/jj.1640536.102`), **"Ponder ethics"** (PsycEXTRA, 1959), **"Bruce Ponder"** (*The Lancet*) |
| `Selective Classification for Deep Neural Networks` | 8 unrelated works incl. **"MARBLE CLASSIFICATION USING DEEP NEURAL NETWORKS"** and **"Waste Classification Using Deep Neural Networks"** |
| `Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer` | **"Sparsely Gated Mixture of Experts Neural Network For Linearization of RF Power Amplifiers"** (IEEE T-MTT 2024) |
| `Energy-Based Transformers are Scalable Learners and Thinkers` | **"Children as Language Learners and Thinkers"** (*Language Arts*, 2009) |
**Every one would have produced a wrong `.bib` entry.** Title-matching caught all four. ⭐ **Refinement to the task's failure mode 1:** the **direct-resolve** endpoint `api.crossref.org/works/<DOI>` *does* return an honest failure — my fabricated negative control **`10.9999/chlu-negative-control-2026-99999` returned HTTP 404 Not Found**. It is only the **`?query.*=` endpoint** that always returns a best match. ⇒ **Use direct-resolve to confirm; never trust a query hit unverified.**

**T2 — ⛔⛔ Neal (2011) has a 2026 second-edition twin.** `10.1201/9781003453420-2` is **"MCMC Using Hamiltonian Dynamics", *Handbook of Markov Chain Monte Carlo*, Radford M. Neal, 2026, pp. 47–95**. Same author, same title, same book — **wrong edition, wrong year, wrong pages.** The paper cites **2011** ⇒ `10.1201/b10905-6` (pp. 113–162). This is exactly the trap the task warned about ("do not confuse with Neal's other MCMC works"), in a form the task did not anticipate.

**T3 — ⛔ Lieb & Robinson has a Springer reprint DOI.** `10.1007/978-3-662-10018-9_25` = the same title reprinted in the collection ***Statistical Mechanics*, pp. 425–431**. The paper cites **Comm. Math. Phys.** ⇒ `10.1007/BF01645779` (28(3):251–257). *(Sibling-pass precedent: the bioRxiv-vs-journal DOI catch.)*

**T4 — ⛔ Duane 1986, same first author, same journal, wrong work.** `10.1016/0370-2693(86)90940-8` = **"Acceleration of gauge field dynamics"**, Duane, *Phys. Lett. B* 176:143–148, **1986**. Author+journal matching would land here.

**T5 — ⛔ "Learn then Test" ≠ "Learning to (Learn at Test Time)".** The arXiv title query's **top hit** was `2310.13807`, *"Learning to (Learn at Test Time)"* (Sun et al.) — a completely different work. The correct record was **entry 2**. Also present in the same result set: *"Adaptive Learn-then-Test"* (`2409.15844`, Zecchin/Park/Simeone) and *"Quantile Learn-Then-Test"* (`2407.17358`) — **both are follow-ups by other authors, neither is the cited work.**

**T6 — ⛔ Platt's chapter is titled "Probabilities for SV Machines".** See D3-1. Author + book + pages all match; **the title does not**. And the task's own warning holds independently: `arXiv:2202.02164` (Aslan, Platt & Sheard, PMLR v197) is **a different person** — I did not query on the surname at all and never surfaced it.

**T7 — ⚠ Publisher-vs-Crossref case and page truncation.** Crossref lower-cases the Duane DOI's trailing `X` (harmless, DOIs are case-insensitive) and returns **only the start page (341)** for Roberts & Tweedie. The true range **341–363** came from a **second instrument** (Project Euclid). ⚠ Project Euclid shows the legacy handle **`bj/1178291835`** — **that is NOT a DOI**; do not enter it as one.

**T8 — ℹ HTTPS confirmed working.** All arXiv API calls used **`https://export.arxiv.org/api/query?...`** and returned records. The `http://` 301-to-nothing failure the task warned about did **not** recur.

---

## Confidence & gaps

**Verified this pass, 2026-08-27:**
- **Direct DOI re-resolve** (`api.crossref.org/works/<DOI>`, exact title + authors + year + volume/pages quoted above): Duane, Lieb & Robinson, Neal, Roberts & Tweedie, Wales & Doye, Schuster/CALM, Angelopoulos-AoAS, Platt-MIT-chapter. **8/8 titles matched the intended work.**
- **arXiv API exact-title / `id_list`**: all 8 arXiv IDs (Gladstone, Graves, Raposo, Ramsauer, Jawahar, Angelopoulos, Banino, Geifman, Schuster, Shazeer).
- **Negative control**: fabricated DOI → **HTTP 404**, reported above.
- **Positive control for every negative**: the same Crossref query endpoint returned exact-title records for Duane / Lieb / Roberts / Wales / Neal / CALM in the same session, so "no DOI" for Geifman, PonderNet, Shazeer, Gladstone and Platt is method-valid, not tool failure. The NeurIPS-30 absence was separately confirmed by a `query.container-title` sweep (only vols 14, 19, 35 registered).
- **Two-instrument**: CALM (Crossref DOI + arXiv comments "NeurIPS 2022 (selected as Oral)"); Roberts & Tweedie pages (Crossref + Project Euclid); Platt-has-no-DOI (Crossref + Semantic Scholar `externalIds` + DBLP).

**Single-sourced / not verified:**
- **Neal 2011 editors** (Brooks, Gelman, Jones, Meng) and **chapter number (5)** — **NOT verified this pass**; Crossref's chapter record carries neither. Omit them or check before entering.
- **Angelopoulos AoAS page range** — Crossref returned volume 19, issue 2, **no page field**. Zotero may emit a page-less entry.
- **Platt MSR-TR-98-14 report number** — not verified; do not enter.
- **Gladstone et al. venue** — arXiv record has no comments/journal-ref and Crossref has no record; I cannot rule out a venue acceptance not yet reflected. Preprint label is correct as of today.

**Not attempted:** OpenReview (bot-walled per prior scouts); `web.archive.org` (tool-blocked per prior scouts); paywalled publisher full text (not needed — all identifiers resolved on registries).

**What to check next, if the Head wants belt-and-braces:** (i) confirm the Neal chapter's editors/chapter number against the Chapman & Hall front matter before hand-entering; (ii) once `refs.bib` is generated, **diff the 15 Zotero-emitted titles against the 17 `pj_sub` strings** — inspection already found **two** truncations (Wales & Doye, Mixture-of-Depths) and two placeholders, so a systematic diff will likely find more; (iii) re-check Gladstone for a venue nearer submission.

---

## Bibtex-ready refs (the 10 NEW records)

```bibtex
@article{angelopoulos2021ltt,
  title={Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control},
  author={Angelopoulos, Anastasios N. and Bates, Stephen and Cand{\`e}s, Emmanuel J. and Jordan, Michael I. and Lei, Lihua},
  journal={arXiv preprint arXiv:2110.01052}, year={2021},
  note={PREPRINT FORM, matches the paper's printed year. Published version: Annals of Applied Statistics 19(2), 2025, doi 10.1214/24-AOAS1998 -- switching requires a 2021->2025 edit at pj_sub l.130 and l.363. Retrieved 2026-08-27.}}

@inproceedings{banino2021pondernet,
  title={PonderNet: Learning to Ponder},
  author={Banino, Andrea and Balaguer, Jan and Blundell, Charles},
  booktitle={8th ICML Workshop on Automated Machine Learning}, year={2021},
  note={arXiv:2107.05407. Workshop paper; no DOI. Retrieved 2026-08-27.}}

@article{duane1987hmc,
  title={Hybrid {M}onte {C}arlo},
  author={Duane, Simon and Kennedy, A. D. and Pendleton, Brian J. and Roweth, Duncan},
  journal={Physics Letters B}, volume={195}, number={2}, pages={216--222}, year={1987},
  doi={10.1016/0370-2693(87)91197-X},
  note={NOT 10.1016/0370-2693(86)90940-8 (Duane 1986, ``Acceleration of gauge field dynamics''). Retrieved 2026-08-27.}}

@inproceedings{geifman2017selective,
  title={Selective Classification for Deep Neural Networks},
  author={Geifman, Yonatan and El-Yaniv, Ran},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2017},
  note={arXiv:1705.08500. NeurIPS vol. 30 has no Crossref-registered proceedings DOI. Retrieved 2026-08-27.}}

@article{lieb1972finite,
  title={The finite group velocity of quantum spin systems},
  author={Lieb, Elliott H. and Robinson, Derek W.},
  journal={Communications in Mathematical Physics}, volume={28}, number={3}, pages={251--257}, year={1972},
  doi={10.1007/BF01645779},
  note={NOT 10.1007/978-3-662-10018-9_25 (the Springer ``Statistical Mechanics'' reprint, pp. 425--431). Retrieved 2026-08-27.}}

@incollection{neal2011mcmc,
  title={{MCMC} Using {H}amiltonian Dynamics},
  author={Neal, Radford M.},
  booktitle={Handbook of Markov Chain Monte Carlo},
  publisher={Chapman and Hall/CRC}, pages={113--162}, year={2011},
  doi={10.1201/b10905-6},
  note={FIRST EDITION. NOT 10.1201/9781003453420-2 (2nd ed., 2026, pp. 47--95). Editors/chapter number not verified this pass. Retrieved 2026-08-27.}}

@incollection{platt1999probabilistic,
  title={Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods},
  author={Platt, John C.},
  booktitle={Advances in Large Margin Classifiers},
  publisher={MIT Press}, pages={61--74}, year={1999},
  note={NO DOI (Crossref + Semantic Scholar + DBLP all negative). The MIT Press chapter at the same pages is titled ``Probabilities for SV Machines'' (2000), doi 10.7551/mitpress/1113.003.0008 -- using it changes BOTH title and year. Do NOT enter the spurious ``10(3)'' volume/issue. Do NOT confuse with Aslan, Platt & Sheard, arXiv:2202.02164 (different person). Retrieved 2026-08-27.}}

@article{roberts1996exponential,
  title={Exponential Convergence of {L}angevin Distributions and Their Discrete Approximations},
  author={Roberts, Gareth O. and Tweedie, Richard L.},
  journal={Bernoulli}, volume={2}, number={4}, pages={341--363}, year={1996},
  doi={10.2307/3318418},
  note={JSTOR-registered DOI; Crossref returns only the start page 341 -- end page 363 from Project Euclid. ``bj/1178291835'' is a Euclid handle, NOT a DOI. Retrieved 2026-08-27.}}

@inproceedings{schuster2022calm,
  title={Confident Adaptive Language Modeling},
  author={Schuster, Tal and Fisch, Adam and Gupta, Jai and Dehghani, Mostafa and Bahri, Dara and Tran, Vinh Q. and Tay, Yi and Metzler, Donald},
  booktitle={Advances in Neural Information Processing Systems 35 (NeurIPS)},
  pages={17456--17472}, year={2022},
  doi={10.52202/068431-1269},
  note={arXiv:2207.07061, comments ``NeurIPS 2022 (selected as Oral)''. DOI and arXiv both give 2022 -- no year drift either way. Retrieved 2026-08-27.}}

@inproceedings{shazeer2017outrageously,
  title={Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2017},
  note={arXiv:1701.06538. ICLR mints no DOIs. Retrieved 2026-08-27.}}

@article{wales1997basinhopping,
  title={Global Optimization by Basin-Hopping and the Lowest Energy Structures of {L}ennard-{J}ones Clusters Containing up to 110 Atoms},
  author={Wales, David J. and Doye, Jonathan P. K.},
  journal={The Journal of Physical Chemistry A}, volume={101}, number={28}, pages={5111--5116}, year={1997},
  doi={10.1021/jp970984n},
  note={FULL title; pj_sub prints the truncated ``Global Optimization by Basin-Hopping.'' Retrieved 2026-08-27.}}
```

**Carried five (for completeness; titles re-confirmed, records not re-derived):**
```bibtex
@article{gladstone2025ebt,
  title={Energy-Based Transformers are Scalable Learners and Thinkers},
  author={Gladstone, Alexi and Nanduru, Ganesh and Islam, Md Mofijul and Han, Peixuan and Ha, Hyeonjeong and Chadha, Aman and Du, Yilun and Ji, Heng and Li, Jundong and Iqbal, Tariq},
  journal={arXiv preprint arXiv:2507.02092}, year={2025},
  note={Preprint; no venue, no DOI as of 2026-08-27.}}

@article{graves2016act,
  title={Adaptive Computation Time for Recurrent Neural Networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1603.08983}, year={2016},
  note={SINGLE AUTHOR -- never ``Graves et al.''. NOT arXiv:1410.5401 (Neural Turing Machines). Retrieved 2026-08-27.}}

@article{raposo2024mod,
  title={Mixture-of-Depths: Dynamically allocating compute in transformer-based language models},
  author={Raposo, David and Ritter, Sam and Richards, Blake and Lillicrap, Timothy and Humphreys, Peter Conway and Santoro, Adam},
  journal={arXiv preprint arXiv:2404.02258}, year={2024},
  note={pj_sub prints the short title ``Mixture-of-Depths''. Retrieved 2026-08-27.}}

@inproceedings{ramsauer2021hopfield,
  title={Hopfield Networks is All You Need},
  author={Ramsauer, Hubert and Sch{\"a}fl, Bernhard and Lehner, Johannes and Seidl, Philipp and Widrich, Michael and Adler, Thomas and Gruber, Lukas and Holzleitner, Markus and Pavlovi{\'c}, Milena and Sandve, Geir Kjetil and Greiff, Victor and Kreil, David and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021},
  note={arXiv:2008.02217. 16 authors. Retrieved 2026-08-27.}}

@article{jawahar2026chlu,
  title={CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning},
  author={Jawahar, Pratik and Pierini, Maurizio},
  journal={arXiv preprint arXiv:2603.01768}, year={2026},
  note={arXiv comments verbatim: ``Accepted as a short paper at ICLR 2026 (AI \& PDE)'' -- a workshop short, not main track. DOUBLE-BLIND FLAG: see reconciliation item 6. Retrieved 2026-08-27.}}
```

---

Git footprint: **none** (read-only; no tracked file touched; **no file under `NIPSsubmission/` or `papers/` modified**). Only write: `.claude/outputs/v1-bib-identifiers.md`.

Open questions / follow-ups / risks:
1. **Angelopoulos year drift (2021 preprint vs 2025 AoAS)** — the single highest-consequence decision in this block. Needs an explicit Head ruling *before* the `.bib` is generated.
2. **Platt** — hand-enter the 1999 fields, or accept `10.7551/mitpress/1113.003.0008` and its 2000 / "Probabilities for SV Machines" title?
3. **Lieb & Robinson** — attach with an explicit "cf./analogy" hedge, or cut? It is the only orphan whose attachment carries a mis-attribution risk.
4. **Platt as a seventh drop-risk** (l.202 table cell, no `\cite`).
5. **Two placeholder entries** (`[CHLU primitive]`, `[The theory note]`) and the placeholder title/author block will still be placeholders after the `.bib` lands unless someone owns them.
6. Does the Head want **numeric vs author–year** style ruled here as it was for V2? V1's prose is written in **author-year** form throughout (*"Graves 2016"*, *"Angelopoulos et al. 2021"*), so unlike V2 an author–year style is the *matching* choice — but the `Anonymous (2026)` entry still must not auto-resolve.

## Proposed handover updates (for the Hub)
- **V1 bibliography inputs are complete:** 17 entries · **15 resolvable identifiers** (5 CARRIED title-confirmed, 10 NEW) · **2 residuals** requiring hand entry (Platt 1999 · Anonymous 2026 theory note) · **0 guessed**. Block at `.claude/outputs/v1-bib-identifiers.md` §Deliverable 1.
- **New standing bib facts:** Duane et al. → `10.1016/0370-2693(87)91197-X`, Phys. Lett. B 195(2):216–222 · Lieb & Robinson → `10.1007/BF01645779`, CMP 28(3):251–257 · Neal 2011 → `10.1201/b10905-6`, pp. 113–162 (**a 2026 2nd-edition twin exists: `10.1201/9781003453420-2`, pp. 47–95**) · Roberts & Tweedie → `10.2307/3318418`, Bernoulli 2(4):341–363 · Wales & Doye → `10.1021/jp970984n`, JPCA 101(28):5111–5116 · Schuster/CALM → `10.52202/068431-1269`, NeurIPS 35:17456–17472 · **Angelopoulos LTT is now Ann. Appl. Stat. 19(2), 2025, `10.1214/24-AOAS1998`** · **Platt 1999 has NO DOI on Crossref, Semantic Scholar or DBLP.**
- **Method note worth banking:** Crossref's `?query.*=` endpoint fabricates plausible best-matches (four live examples this pass); its **`/works/<DOI>` direct-resolve endpoint returns an honest HTTP 404** on a fabricated DOI. Verify with direct-resolve, never with a query hit.
- **NeurIPS proceedings DOIs exist only for vols ≥35 (2022+)** via the `10.52202` NeurIPS-Foundation prefix, and for vols ≤19 via MIT Press `10.7551`. **Vol. 30 (2017) is unregistered** — so NeurIPS-2017 works are arXiv-only.
- **Cite-or-cut:** 5 of 6 orphans attach to existing sentences with a parenthetical (sites quoted in Deliverable 4); **Lieb & Robinson is the sole cut candidate** unless hedged as an analogy. **Plus `Platt` needs a `\cite` at l.202** or it drops with them.
- **Double-blind item for the Head/Hub:** `pj_sub.tex` l.35 *"**Our** reference memory ... introduced as CHLU in **Jawahar & Pierini (2026)**"* — the same construction `v2-cite-check.md` §26 flagged for V2, now live in V1.
