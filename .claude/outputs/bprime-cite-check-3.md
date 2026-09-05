# bprime-cite-check-3 — web-scout report
Task + acceptance criterion: verify three items — (1) GDN's presentation of Mamba2 as `S_t = α_t S_{t−1} + v_t k_tᵀ` (double-source it or keep the paraphrase-only restriction), (2) Mitzenmacher 2018 citation facts, (3) Kipf et al. 2019 (SOSD) + the accompanying PVLDB study, incl. pinning the source of the "30–80× larger" / "4 orders-of-magnitude" quotes in draft §5.3 — each with a VERIFIED / CORRECTED / UNVERIFIABLE verdict and ready-to-paste BibTeX.
Status: **done** — 1 VERIFIED (restriction liftable), 1 VERIFIED-with-upgrade, **1 CORRECTED: a live mis-attribution in draft §5.3**.
**Dial declaration (protocol §7):** none — instrument/recon (citation verification). No performance number, no laundering control, nothing falsifiable in the dial sense.

## ⛔ DOWNSTREAM RECONCILIATION LIST (owner needed — protocol §5 corollary, first 10 lines)
1. **r4 writer — §5.3 (l.1029–1034) MIS-ATTRIBUTION, must fix.** The quotes *"30–80× larger than B-trees"* and *"4 orders-of-magnitude more time to build"* are **NOT from SOSD / Kipf et al. 2019**. They are from **Chesetti & Pandey, "Evaluating Learned Indexes for External Memory Joins", ACDA 2025 (SIAM), pp. 101–114, arXiv:2407.00590** — a *later, external-memory* study that *extends* SOSD. **SOSD's own verdict runs the opposite way** (RMI size overhead **3%**, RadixSpline **<1%**, vs **B-tree 16%**; abstract: *"learned models indeed often outperform state-of-the-art implementations, and are therefore a promising direction for future research"*). Printing an anti-learned-index verdict under SOSD's name is the kind of error a DB-side reviewer catches in one click.
2. **r4 writer — same sentence, two further precision fixes if the quotes are kept (now correctly attributed):** (a) the subject is **"The RadixSpline and RMI"**, not "a learned index" — in the *same* paragraph **PGM is 4× *smaller* than the B-tree**; (b) that B-tree was **deliberately sparsified** (*"we only build the B-tree by uniformly sampling every 256th key"*), so the ratio is against a shrunk baseline. Quoting the pair without (a)+(b) is exactly the matched-bytes sloppiness §5 is auditing others for.
3. **r4 writer (⟦CITE2⟧) — GDN restriction LIFTED.** `S_t = α_t S_{t−1} + v_t k_tᵀ` is now **triple-sourced incl. the ICLR 2025 camera-ready**; r4 **may quote** it. Carry GDN's own hedge: §2.1 introduces it as Mamba2 *"represented by the following linear recurrence (up to specific parameterization)"*.
4. **r4 writer / curator — §1 (l.68) cite upgrade.** Mitzenmacher 2018 = **NeurIPS 2018, pp. 462–471**; Kipf et al. 2019 = **non-archival** (NeurIPS 2019 ML-for-Systems workshop; DBLP has *only* the CoRR entry) → cite as arXiv:1911.13014 + workshop. The "accompanying PVLDB study" is **Marcus et al.** (first author differs from Kipf), PVLDB **14(1):1–13**, DOI **10.14778/3421424.3421425**.
5. **Program tooling — standing limitation LIFTED.** PDF text/pages are now readable by a scout: `WebFetch` a PDF → it saves the binary to `…/tool-results/<name>.pdf` and prints the path → `Read` that path with `pages: "1-5"` renders the pages. This closed three of four items this wave (GDN camera-ready, SOSD full text, Mitzenmacher published text). ⚠ I have **no `Bash`**, so `pdftotext` was never available to me — the fetch→Read path is the working substitute. **OpenReview bot-blocked again (6th consecutive wave).**

---

## Answer first
**Item 1 (GDN equation) — VERIFIED, triple-sourced, including the published camera-ready.** The ICLR 2025 camera-ready (`jankautz.com/publications/GatedDeltaNet_ICLR25.pdf`, header *"Published as a conference paper at ICLR 2025"*) states in §1: *"Mamba2 addresses this limitation by introducing a simple gated update rule, **S_t = α_t S_{t−1} + v_t k_tᵀ**, which uniformly decays all key-value associations at each time step by a dynamic ratio, α_t ∈ (0,1)."* Independently confirmed by ar5iv's rendering of the arXiv source and by an IA-Scholar full-text phrase hit. `rival-recon` §1.4 is correct as recorded.
**Item 2 (Mitzenmacher) — VERIFIED.** **Michael Mitzenmacher (2018), "A Model for Learned Bloom Filters, and Optimizing by Sandwiching", NeurIPS 2018 (Advances in NeurIPS 31), pp. 462–471**; arXiv:1901.00902 (the earlier notes are 1802.00884 / 1803.01474). The matched-space discipline the draft attributes to it is in the published text verbatim.
**Item 3 (Kipf/SOSD) — citation facts VERIFIED, but the two quoted numbers are CORRECTED to a different paper.** Kipf, Marcus, van Renen, Stoian, Kemper, Kraska, Neumann (2019), *"SOSD: A Benchmark for Learned Indexes"*, arXiv:1911.13014, NeurIPS 2019 Workshop on ML for Systems — real, but **contains neither quote and reaches the opposite conclusion on size**. The quotes belong to Chesetti & Pandey (ACDA 2025).

---

## 1. GDN's Mamba2 equation — **VERIFIED** (quotable)

### 1.1 Primary source: ICLR 2025 camera-ready, read directly
Fetched `https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf` (455.7 KB; the fetch model refused the binary, the saved file was then read as page images). Every page carries the running header **"Published as a conference paper at ICLR 2025"**. Authors as rendered: **Songlin Yang (MIT CSAIL), Jan Kautz (NVIDIA), Ali Hatamizadeh (NVIDIA)**; code link `https://github.com/NVlabs/GatedDeltaNet`.

- **§1 Introduction (p. 1), verbatim:** *"Mamba2 addresses this limitation by introducing a simple gated update rule, **S_t = α_t S_{t−1} + v_t k_tᵀ**, which uniformly decays all key-value associations at each time step by a dynamic ratio, α_t ∈ (0,1). However, this approach does not account for the varying importance of different key-value associations…"*
- **§2.1 "Mamba2: Linear Attention with Decay" (p. 2), verbatim framing + equation:** *"Here we take Mamba2 (Dao & Gu, 2024a) as an example, which can be represented by the following linear recurrence (**up to specific parameterization**): **S_t = α_t S_{t−1} + v_t k_tᵀ, o_t = S_t q_t** where α_t ∈ (0,1) is a data-dependent scalar-valued decay term that varies with t."* ⚠ **The hedge "up to specific parameterization" is GDN's own** — r4 should carry it, because Mamba-2's actual parameterization (SSD, per-head P×N state) is not literally this scalar-decay linear-attention form; GDN is presenting it in a common notation.
- Adjacent equations, same source, useful if r4 needs the family (p. 2 §2.2, p. 4 Eq. 10):
  - vanilla linear attention: `S_t = S_{t−1} + v_t k_tᵀ ∈ ℝ^{d_v×d_k}`, `o_t = S_t q_t ∈ ℝ^{d_v}`
  - DeltaNet: `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`
  - **gated delta rule (GDN, Eq. 10): `S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`**, *"where the data-dependent gating term α_t ∈ (0,1) controls state decay."*
  - GDN's own reading of the two limits (§3.1): *"it can promptly clear memory by setting α_t → 0, while selectively updating specific content without affecting other information by setting α_t → 1 (effectively switching to the pure delta rule)."*

### 1.2 Independent corroboration (two more)
| source | what it returned |
|---|---|
| `ar5iv.labs.arxiv.org/html/2412.06464` (LaTeX rendering of the arXiv source — **works**, unlike `arxiv.org/html/…` which is 404 for v1 and v2) | §2.2 equation `S_t = α_t S_{t−1} + v_t k_tᵀ`, α_t *"data-dependent scalar-valued decay term"*; **Table 1** rows — Mamba2 `S_t = α_t S_{t−1} + v_t k_tᵀ`; DeltaNet `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`; Gated DeltaNet `S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ` |
| `scholar.archive.org` full-text phrase search `"uniformly decays all key-value associations"` | **1 hit**, GDN, snippet: *"Mamba2 addresses this limitation by introducing a simple gated update rule, S t = α t S t−1 + v t k ⊺ t , which uniformly decays all key-value associations"* |
| `openreview.net/pdf?id=r8H7xhYPwz` | ⛔ **bot-verification wall** (6th consecutive wave) — not used |
| `proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html` | confirms title/authors/venue (ICLR 2025); no page numbers (ICLR has none) |
| `github.com/fla-org/flash-linear-attention` README | lists Mamba2 and Gated DeltaNet entries but **prints no update equations** — dead end as a source for the equation |

- Minor observation, **do not cite**: the camera-ready's author footnote renders as *"∗Equation contribution. Work done during SY's internship at NVIDIA."* — almost certainly a typo for "Equal contribution" **in the paper**; noted only so nobody re-reads it as a finding.
- **Verdict: VERIFIED** (published camera-ready + arXiv rendering + IA Scholar). `rival-recon` §1.4 discharged; the cite-check-2 paraphrase-only restriction can be lifted.

---

## 2. Mitzenmacher 2018 — **VERIFIED** (venue/pages/title, + the matched-space quote)

### 2.1 Citation facts
| fact | source 1 | source 2 | grade |
|---|---|---|---|
| Title *"A Model for Learned Bloom Filters, and Optimizing by Sandwiching"* (**comma** after "Filters" on the PDF title page) | NeurIPS proceedings PDF, read directly | DBLP / NeurIPS abstract page render it **without** the comma: *"A Model for Learned Bloom Filters and Optimizing by Sandwiching"* | **double, with a known title variant** |
| Author **Michael Mitzenmacher**, School of Engineering and Applied Sciences, Harvard University | PDF p.1 | DBLP `conf/nips/Mitzenmacher18` | **double** |
| Venue **NeurIPS 2018** — PDF footer verbatim: *"32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada."* | PDF p.1 footer | `proceedings.neurips.cc/paper/2018/hash/0f49c89d…` (*Advances in Neural Information Processing Systems 31*) | **double** |
| **Pages 462–471** | DBLP (`pages 462-471`) | Semantic Scholar Graph API raw JSON: `"journal": {"pages": "462-471"}` (+ ACM DL entry `10.5555/3326943.3326986`) | **double, shared provenance** (DBLP↔S2/MAG likely correlated; NeurIPS's own page prints no pagination) |
| arXiv **1901.00902** is the version of *this* paper | DBLP `journals/corr/abs-1901-00902` | S2 `externalIds.ArXiv = 1901.00902` | **double** |
| Earlier, **different** notes: **arXiv:1802.00884** *"A Model for Learned Bloom Filters and Related Structures"* (3 Feb 2018, comments: *"5 pages, commentary on the 'Learned Index Structures' paper"*) and **arXiv:1803.01474** *"Optimizing Learned Bloom Filters by Sandwiching"* | arXiv abs pages | paper p.1: *"this work incorporates and extends analysis that appeared in two prior working notes [8, 9]"* | **double** ⇒ **do not cite 1802.00884 as "Mitzenmacher 2018 (NeurIPS)"** |

### 2.2 The claim the draft rests on it — supported, verbatim from the **published** text
- **§3.1, NeurIPS PDF p. 3:** *"In essence, [7] suggests using a pre-filter ahead of the Bloom filter, where the pre-filter comes from a neural network and estimates the probability a key is in the set, allowing the use of a smaller Bloom filter than if one just used a Bloom filter alone. **Performance improves if the size to represent the learned function f and the size of the smaller backup filter for false negatives is smaller than the size of a corresponding Bloom filter with the same false positive rate.**"* (identical sentence returned by `ar5iv/1901.00902` ⇒ preprint and published text agree here.)
- **Abstract, outcome (2):** *"we show how to estimate what size the learning function must obtain in order to obtain improved performance"*.
- ⇒ This is precisely *"compare a learned structure against a classical one at matched space"*: the learned model's own bytes are charged to the learned side. **Mitzenmacher is the stronger of the draft's two §1 cites for that claim** (see §3.3 for why Kipf is weaker).
- **Verdict: VERIFIED.**

---

## 3. Kipf et al. 2019 (SOSD) + PVLDB companion — facts VERIFIED, **quotes CORRECTED**

### 3.1 SOSD citation facts
| fact | source 1 | source 2 | grade |
|---|---|---|---|
| Title *"SOSD: A Benchmark for Learned Indexes"* | arXiv abs 1911.13014 | PDF title page (read directly) | **double** |
| Authors **Andreas Kipf\*, Ryan Marcus\*, Alexander van Renen\*, Mihail Stoian, Alfons Kemper, Tim Kraska, Thomas Neumann** (\* = *"equal contribution"*, PDF footnote); TUM + MIT CSAIL | PDF p.1 | arXiv abs author list (same order) | **double** |
| Venue: **non-archival workshop.** arXiv Comments field verbatim: *"NeurIPS 2019 Workshop on Machine Learning for Systems"*; PDF footer: *"33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada."* | arXiv abs | PDF p.1 footer | **double** |
| **DBLP has only the CoRR entry** (`journals/corr/abs-1911-13014`) — no proceedings record, no pages, no DOI beyond arXiv | DBLP API | (absence) | **single (registry-native)** |
| Submitted **29 Nov 2019**, v1 only; no journal-ref | arXiv abs | PDF sidebar `arXiv:1911.13014v1 [cs.DB] 29 Nov 2019` | **double** |

### 3.2 ⛔ The two quotes in draft §5.3 are **not SOSD's** — CORRECTED
I read the **entire** SOSD paper (5 pages, arXiv PDF, page images). It contains **no "30–80×" and no "4 orders-of-magnitude"** claim, and its size verdict is the **reverse** of the draft's framing:

- **SOSD Table 2, "size overhead" row (64-bit keys):** ART **25%**, **B-tree 16%**, BS 0%, IS 0%, RBS **<1%**, **RMI 3%**, RS **<1%**, TIP 0%. (32-bit row: ART 47%, B-tree 16%, FAST 123%, RMI 3%, RS <1%.) ⇒ in SOSD the learned CDF approximators are **an order of magnitude *smaller* in overhead than the B-tree**, not 30–80× larger.
- **SOSD abstract, verbatim:** *"We also show preliminary results for selected index structures, and find that learned models indeed often outperform state-of-the-art implementations, and are therefore a promising direction for future research."*
- **SOSD §4 Takeaways, verbatim:** *"We have seen that the CDF approximators (RMI, RS) can outperform our baseline implementations."* … *"the optimal search strategy depends on whether a user can afford to manually tune and fit a CDF model, as both RMI and RS require dataset-specific tuning."* … *"if users cannot afford the training time, we recommend using ART or FAST for 32-bit keys and ART or RBS for 64-bit keys."*
- SOSD's only build-time statement (§3 Results): *"RS experiences the highest build times for fitting a fine-grained linear spline to the CDF of the data. However, even without optimizations, the build times of the CDF approximators may be acceptable for many applications."*

**Where the quotes actually come from** — Chesetti & Pandey, §"Index size"/"Index construction time" (arXiv:2407.00590**v2**, p. 12 of the PDF, §6.6 *Evaluating learned indexes for joins on disk*; read directly as page images):
> *"For real-world datasets, PGM indexes (both sampled and full) have the lowest memory footprint among all learned indexes being **4× smaller than the B-tree**. **The RadixSpline and RMI are an order of magnitude (30 − 80×) larger than B-trees.** Both the OSM and Books datasets were not able to completely construct the ALEX index as they ran out of memory. For synthetic datasets, PGM indexes are an order-of-magnitude smaller compared to B-trees on all datasets."*
> *"**Index construction time.** **All learned indexes take at least 4 orders-of-magnitude more time to build than B-trees.** Constructing the PGM index with sampling reduces the duration by roughly two orders-of-magnitude."*

⇒ the draft's two quoted **strings are accurate**; the **attribution is wrong**. Three further caveats a careful reviewer will raise:
1. **Subject mismatch:** "RadixSpline and RMI", not "a learned index" — the same paragraph reports PGM **4× smaller**.
2. **Baseline was sparsified:** same paper, §6.6 setup: *"To make the comparison fair with static learned indexes, the B-tree is built by bulk loading the keys. Furthermore, we only build the B-tree by uniformly sampling every **256th** key from the dataset… This makes the B-tree smaller in size and operate in the similar manner to other learned indexes."* The 30–80× is against that shrunken B-tree.
3. **This paper's own headline is "roughly a tie", not a rout.** Conclusion, verbatim: *"We demonstrate that disk-based learned indexes, designed for handling extremely large datasets requiring external memory, exhibit similar performance to B-tree indexes. Despite their advantages in smaller index sizes and more efficient lookups, as evidenced in in-memory scenarios, these benefits are not as pronounced in external memory due to the dominance of I/O costs."* And its takeaway *"the PGM index with sampling offers the best tradeoff in terms of query latency, construction time and space usage. Using sampling, the join can be sped up 2× compared to B-trees and at the same time using 4× less space, while the index itself takes 10× longer to build."*
4. **This paper is peer-reviewed and its section numbering moved between versions.** Published as **ACDA 2025 (SIAM), pp. 101–114, DOI 10.1137/1.9781611978759.8** (double-sourced: DBLP + OpenAlex). The quotes above are pinned to **arXiv v2** (7 Jul 2024), where they sit in §6.6; in **v3** (23 May 2025) the same content is in **§5.2**. I could not read the SIAM camera-ready (paywalled) — **if r4 quotes, pin the version explicitly** (`arXiv:2407.00590v2`) or paraphrase.

### 3.3 What SOSD *does* support (a usable, honest replacement for §5.3)
SOSD reports a **size-overhead column alongside every latency number** — i.e. it makes index comparisons **space-accounted**; it does not *equalize* bytes. So:
- **Safe §1 claim:** learned-index benchmarking reports the learned structure's own space cost next to its speed, so a learned-vs-classical comparison is space-accounted as a matter of course (Mitzenmacher 2018 makes it a *condition for improvement*; Kipf et al. 2019 makes it a *reported column*).
- **Unsafe §1 claim:** "…compare at *matched space*" — nobody in SOSD equalizes bytes. Mitzenmacher is the only one of the two that conditions the *verdict* on total size.
- **Suggested §5.3 rewrite (r4's call):** *"Learned Bloom filters (Mitzenmacher, NeurIPS 2018) and learned-index benchmarks (Kipf et al., 2019; Marcus et al., PVLDB 2020) report the learned structure's own space and build cost next to its speed, and later SOSD-derived studies push it further — Chesetti & Pandey (ACDA 2025) find RadixSpline and RMI '30–80× larger than B-trees' and that 'all learned indexes take at least 4 orders-of-magnitude more time to build than B-trees'. That is the tone this audit imports. What is absent there is any sequence, any test-time dynamics, and any state."*

### 3.4 The "accompanying PVLDB study"
**Marcus, Kipf, van Renen, Stoian, Misra, Kemper, Neumann, Kraska (2020), "Benchmarking Learned Indexes", PVLDB 14(1):1–13**, DOI **10.14778/3421424.3421425**, arXiv:2006.12804 — double-sourced (DBLP API + the VLDB asset path `vldb.org/pvldb/vol14/p1-marcus.pdf`). ⚠ **First author is Marcus, not Kipf**, and the author list gains **Sanchit Misra** vs. SOSD — "Kipf et al. 2019 and the accompanying PVLDB study" is fine as prose but the bibliography must not collapse them. Its build-time finding (ar5iv, **single-sourced**, not read from the published PDF): *"no learned structure yet provides builds as fast as insert-optimized traditional index structures"* — same direction as Chesetti & Pandey, far weaker in magnitude. **Do not quote the PVLDB numbers without a re-read; I did not verify them from the published text.**

---

## Ready-to-paste BibTeX
```bibtex
@inproceedings{yang2025gateddeltanet,
  title     = {Gated Delta Networks: Improving {Mamba2} with Delta Rule},
  author    = {Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2412.06464. Camera-ready verified 2026-08-01 from
               jankautz.com/publications/GatedDeltaNet_ICLR25.pdf (header "Published as a
               conference paper at ICLR 2025") + ar5iv rendering + IA Scholar phrase hit.
               Quotable: "Mamba2 addresses this limitation by introducing a simple gated
               update rule, S_t = alpha_t S_{t-1} + v_t k_t^T, which uniformly decays all
               key-value associations at each time step by a dynamic ratio, alpha_t in (0,1)."
               Sec 2.1 hedges this as Mamba2 "up to specific parameterization" -- carry that.
               Code: github.com/NVlabs/GatedDeltaNet. OpenReview id r8H7xhYPwz (site bot-blocked).}}

@inproceedings{mitzenmacher2018learnedbloom,
  title     = {A Model for Learned {B}loom Filters, and Optimizing by Sandwiching},
  author    = {Mitzenmacher, Michael},
  booktitle = {Advances in Neural Information Processing Systems 31 (NeurIPS 2018)},
  pages     = {462--471},
  year      = {2018},
  note      = {arXiv:1901.00902; supersedes the working notes arXiv:1802.00884 and
               arXiv:1803.01474 -- do NOT cite those as the NeurIPS paper. Verified
               2026-08-01 (proceedings.neurips.cc PDF read directly; DBLP conf/nips/
               Mitzenmacher18; S2 journal.pages "462-471"; ACM DL 10.5555/3326943.3326986).
               PDF title page has a comma after "Filters"; DBLP/NeurIPS index it without.
               Matched-space sentence (published text, Sec 3.1): "Performance improves if the
               size to represent the learned function f and the size of the smaller backup
               filter for false negatives is smaller than the size of a corresponding Bloom
               filter with the same false positive rate."}}

@misc{kipf2019sosd,
  title        = {{SOSD}: A Benchmark for Learned Indexes},
  author       = {Kipf, Andreas and Marcus, Ryan and van Renen, Alexander and Stoian, Mihail
                  and Kemper, Alfons and Kraska, Tim and Neumann, Thomas},
  year         = {2019},
  eprint       = {1911.13014},
  archivePrefix= {arXiv},
  primaryClass = {cs.DB},
  note         = {NeurIPS 2019 Workshop on Machine Learning for Systems (non-archival; DBLP
                  has only the CoRR entry). Verified 2026-08-01 (arXiv abs Comments field +
                  full PDF read). CAUTION: this paper does NOT contain the "30-80x larger" or
                  "4 orders-of-magnitude" claims; its Table 2 size overheads are RMI 3%,
                  RadixSpline <1% vs B-tree 16%, and its abstract says learned models "often
                  outperform state-of-the-art implementations".}}

@article{marcus2020benchmarking,
  title   = {Benchmarking Learned Indexes},
  author  = {Marcus, Ryan and Kipf, Andreas and van Renen, Alexander and Stoian, Mihail and
             Misra, Sanchit and Kemper, Alfons and Neumann, Thomas and Kraska, Tim},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {14},
  number  = {1},
  pages   = {1--13},
  year    = {2020},
  doi     = {10.14778/3421424.3421425},
  note    = {arXiv:2006.12804. Venue/volume/pages/DOI verified 2026-08-01 (DBLP + vldb.org
             asset path). Body text NOT read from the published PDF -- do not quote its
             numbers without a re-check.}}

@inproceedings{chesetti2025learnedjoins,
  title     = {Evaluating Learned Indexes for External Memory Joins},
  author    = {Chesetti, Yuvaraj and Pandey, Prashant},
  booktitle = {2025 Proceedings of the SIAM Conference on Applied and Computational Discrete
               Algorithms (ACDA)},
  pages     = {101--114},
  year      = {2025},
  publisher = {SIAM},
  doi       = {10.1137/1.9781611978759.8},
  note      = {arXiv:2407.00590 (v1 30 Jun 2024, v2 7 Jul 2024, v3 23 May 2025). THIS is the
               source of "The RadixSpline and RMI are an order of magnitude (30-80x) larger
               than B-trees" and "All learned indexes take at least 4 orders-of-magnitude more
               time to build than B-trees" -- both pinned to arXiv v2 Sec 6.6 (= v3 Sec 5.2),
               read directly 2026-08-01. NOT SOSD. Caveats: same paragraph reports PGM 4x
               SMALLER than the B-tree, and the B-tree baseline is built on every 256th key.
               SIAM camera-ready not read (paywalled) -- quote the arXiv version explicitly.}}
```

---

## How I verified (sources actually fetched, 2026-08-01)
| claim | source 1 | source 2 | source 3 | grade |
|---|---|---|---|---|
| GDN: `S_t = α_t S_{t−1} + v_t k_tᵀ` for Mamba2 | **ICLR 2025 camera-ready PDF** (jankautz.com), §1 p.1 and §2.1 p.2, read as page images | `ar5iv.labs.arxiv.org/html/2412.06464` (§2.2 + Table 1) | `scholar.archive.org` phrase search *"uniformly decays all key-value associations"* (1 hit) | **triple, incl. published** |
| GDN Table 1 rows (Mamba2 / DeltaNet / GDN) | ar5iv Table 1 | camera-ready Eq. 10 + §2.2 (equations match) | — | **double** |
| GDN venue/authors | `proceedings.iclr.cc/…/4904fad…-Abstract-Conference.html` | camera-ready header + `arxiv.org/abs/2412.06464` comments *"ICLR 2025 camera ready"* (cite-check-2) | — | **double** |
| Mitzenmacher venue/title/author | NeurIPS proceedings PDF (footer *"32nd Conference on NeurIPS 2018, Montréal"*) | `proceedings.neurips.cc` abstract page | DBLP `conf/nips/Mitzenmacher18` | **triple** |
| Mitzenmacher **pp. 462–471** | DBLP | S2 Graph API raw JSON `"journal":{"pages":"462-471"}` | ACM DL `10.5555/3326943.3326986` (search-surfaced, not fetched) | **double, correlated provenance** |
| Mitzenmacher matched-space sentence | NeurIPS PDF §3.1 p.3 (read directly) | `ar5iv/1901.00902` — identical sentence | — | **double, preprint == published** |
| SOSD authors/venue/date | arXiv abs (Comments: *"NeurIPS 2019 Workshop on Machine Learning for Systems"*) | full PDF read (5 pp.) | DBLP (CoRR only) | **triple** |
| **SOSD does NOT contain the "30–80×"/"4 orders" claims** | full PDF read, all 5 pages incl. Table 2 and §4 Takeaways | ar5iv summary of the same paper (consistent) | — | **double** |
| SOSD Table 2 size overheads (RMI 3%, RS <1%, B-tree 16%, ART 25/47%) | PDF Table 2, read directly | — | — | **single (primary artifact)** |
| "30–80×" + "4 orders-of-magnitude" wording | **arXiv:2407.00590v2 PDF p.12, read directly** | `arxiv.org/html/2407.00590v2` and `v3` fetches (same content; v3 relocates it to §5.2) | — | **double** |
| Chesetti & Pandey = ACDA 2025, pp. 101–114, DOI 10.1137/1.9781611978759.8 | DBLP API | OpenAlex `works/doi:10.1137/1.9781611978759.8` | — | **double** |
| Marcus et al. PVLDB 14(1):1–13, DOI 10.14778/3421424.3421425 | DBLP API | `vldb.org/pvldb/vol14/p1-marcus.pdf` (asset path encodes the DOI suffix) | — | **double** |

### Tooling record (honest)
| attempted | result |
|---|---|
| `WebFetch` a PDF → `Read` the saved `tool-results/*.pdf` with `pages:` | ✅ **WORKS** — closed GDN camera-ready, SOSD (5 pp.), Mitzenmacher (3 pp.), Chesetti & Pandey (6 pp.). **This lifts cite-check-2's "PDF-only sources are structurally unverifiable" limitation.** |
| `Bash` / `pdftotext` | ⛔ **not in my toolset this session** (task file assumed it); the fetch→Read path substituted |
| `openreview.net/pdf?id=r8H7xhYPwz` | ⛔ bot-verification wall (**6th consecutive wave**) |
| `learned.systems/papers/sosd.pdf` | ⛔ **TLS certificate expired** (used arXiv PDF instead) |
| `api.semanticscholar.org/graph/v1/paper/search` | ✅ once (Mitzenmacher), then **HTTP 429** |
| `prashantpandey.github.io/publication/` | 404 (OpenAlex used instead) |
| `arxiv.org/html/2412.06464v1`,`v2` | 404 both (**ar5iv works** — record this: ar5iv is the fallback when arXiv HTML 404s) |
| fetch-model verbatim reproduction of long paragraphs | ⛔ refuses (quote-length cap) — **read the PDF pages yourself when exact wording is load-bearing** |

## Confidence & gaps
- **High:** GDN equation (published camera-ready read directly); Mitzenmacher venue/title/author and the quoted sentence (published text); SOSD identity + the negative finding that it does **not** contain the two quotes (whole paper read); Chesetti & Pandey wording + ACDA publication record.
- **Medium:** Mitzenmacher's **pp. 462–471** — DBLP and S2 agree but likely share provenance, and NeurIPS's own page prints no pagination. Safe to print (NeurIPS entries are commonly cited without pages anyway).
- **Single-sourced / not verified:** Marcus et al. PVLDB body text (ar5iv only); the SIAM/ACDA camera-ready of Chesetti & Pandey (paywalled — quotes are pinned to arXiv v2); GDN Table 1 as *typeset in the camera-ready* (I read pp. 1–4; Table 1 is later — the equations I did read match ar5iv's table exactly).
- **Still open elsewhere (ledger, not my scope):** MUNKEY's workshop identity + presentation type (OpenReview, now 6 waves blocked); the §5 cites listed as unverified in `bprime-cite-check` (Wang/Shi/Fox 2501.12352, ATLAS 2505.23735, Miras 2504.13173, HOLA 2607.02303, Based/MAD/Zoology/RULER).
- **Next search if the Hub wants it (cheap, now that PDFs are readable):** (a) re-run the **Mamba-2 paper body** check that cite-check-2 had to abandon (`arxiv.org/pdf/2405.21060` → Read) to get a paper-sourced "P × N" statement; (b) the ACDA camera-ready of Chesetti & Pandey if any institutional access exists; (c) the remaining §5 qualitative cites, in one batch.

## Proposed handover updates (for the Hub)
- `bprime-cite-check-3` **done. Item 1 VERIFIED — the r4 paraphrase-only restriction on GDN's Mamba2 equation is LIFTED**: `S_t = α_t S_{t−1} + v_t k_tᵀ` is confirmed in the **ICLR 2025 camera-ready** (§1 and §2.1) plus ar5iv plus IA Scholar. Carry GDN's own hedge *"up to specific parameterization"*.
- **Item 2 VERIFIED:** Mitzenmacher = **NeurIPS 2018, pp. 462–471**, arXiv:1901.00902 (⚠ *not* 1802.00884). Published-text quote available for §1's matched-space claim.
- ⛔ **Item 3 CORRECTED — live error in draft §5.3.** The *"30–80× larger than B-trees"* / *"4 orders-of-magnitude more time to build"* quotes are **Chesetti & Pandey, ACDA 2025 (arXiv:2407.00590v2)**, **not SOSD**. **SOSD reports the opposite on size** (RMI 3%, RS <1% vs B-tree 16% overhead) and concludes learned models *"often outperform state-of-the-art implementations"*. Needs an r4 edit; suggested rewrite in §3.3 of my report. Two additional caveats if the quotes are kept: the subject is RadixSpline/RMI only (PGM is 4× *smaller*), and that paper's B-tree baseline is built on **every 256th key**.
- **Citation hygiene:** SOSD is **non-archival** (workshop; DBLP has only CoRR) — cite arXiv:1911.13014. The "accompanying PVLDB study" is **Marcus** et al., PVLDB **14(1):1–13**, DOI 10.14778/3421424.3421425 — different first author, extra co-author; don't collapse the two entries.
- ⭐ **Tooling upgrade, program-wide:** the cite-check-2 escalation *"this machine has no PDF text extraction"* is **resolved in practice** — `WebFetch` a PDF (it saves the binary and prints the path) → `Read` that path with `pages:`. Four PDF-only sources were read this wave. Add to the scout kit alongside `scholar.archive.org` and **ar5iv-as-fallback-when-arxiv-HTML-404s**. **OpenReview: blocked for the 6th straight wave — recommend the Hub stop scoping OpenReview-dependent items.**
