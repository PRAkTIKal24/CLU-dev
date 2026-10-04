# w8 spoke/novelty-scout — NOVELTY.md

**Dial: novelty.** Read-only web scout. No code, no scored run.

Object under test (from the task file): *multilinear / bilinear factored generation with a
request-indexed tensor head* — latent space partitioned into **wells**, each well a **fixed leading
principal subspace** of the training features; a **request** of factor–value terms indexes a
**tensor-product address book** of real coefficients over each well's directions; the concatenated
read is rendered by a **frozen decoder**; content **unsupervised**, labels enter only at the
interface. Separately: **exact deletion** by covariance downdate + re-eigendecomposition.

---

## VERDICT: ⚠ ADJACENT — but one of our claims, as currently phrased, must be STOPPED.

I am not reporting CLEAR. A referee *will* find TensorFaces, and they will be right to.

**The headline phrase "multilinear/bilinear factored generation with a request-indexed tensor head"
is not ours. It is Tenenbaum & Freeman (2000) and Vasilescu & Terzopoulos (2002), and it has been
re-run in deep form by Vlasic et al. (2005) and Georgopoulos et al. (ICML 2020).** If the paper
claims *that sentence* as the contribution, the claim is dead on arrival. ⛔ **Stop that claim as
stated.** What survives is the *composite object* and, above all, the deletion property — see
"What survives", below.

The full object (all six atoms together) is **not** in the literature I can find: no single work
partitions an unsupervised principal-subspace representation into wells, indexes it with a
tensor-product label address book, renders with a frozen decoder, **and** deletes exactly.
So: ADJACENT, one sentence of difference —

> **Difference in one sentence:** the prior multilinear-generation line builds its factor bases *out
> of the labels* (the data tensor is organised by every factor combination, so the "person mode" and
> "expression mode" bases are label-defined), renders linearly, and never deletes; ours keeps the
> content bases **label-free** (leading principal subspaces of the features, computed unsupervised),
> lets labels touch the model **only** at an address book in front of a **frozen nonlinear decoder**,
> and supports **exact deletion** by downdating the sufficient statistic.

---

## 1. The claim-atoms, and who already owns each

I split the object into six atoms so prior art is matched piece-by-piece rather than by gestalt.

| | A1 well-partition | A2 fixed leading principal subspace, **unsupervised** | A3 request = factor–value terms | A4 **tensor-product** address book | A5 **frozen nonlinear** decoder | A6 **exact deletion** |
|---|---|---|---|---|---|---|
| Tenenbaum & Freeman 2000, bilinear style–content | ✗ | ~ (SVD basis, but label-organised) | ✔ (2 factors) | ✔ (y = Σᵢⱼ Wᵢⱼ aᵢ bⱼ) | ✗ linear | ✗ |
| **TensorFaces**, Vasilescu & Terzopoulos 2002 | ✗ | ~ (mode matrices *are* principal subspaces — but the modes are the labels) | ✔ | ✔ (N-mode SVD core ×ₙ factor vectors) | ✗ linear pixels | ✗ |
| Vlasic et al. 2005, Face Transfer (multilinear 3DMM) | ✗ | ~ | ✔ | ✔ | ✗ linear mesh | ✗ |
| Vasilescu & Kim, **CausalX / M-mode Block SVD** | **✔** (blocks = parts, block-diagonal core) | ~ | ✔ | ✔ | ✗ | ✗ (incremental *add* only) |
| **MLC-VAE**, Georgopoulos et al. ICML 2020 | ✗ | ✗ (learned end-to-end) | ✔ | ✔ (CP/Tucker core × one-hot labels) | ✗ (decoder trained jointly) | ✗ |
| GANSpace, Härkönen et al. 2020 | ~ (layer ranges) | **✔** (PCA of features, unsupervised) | ✗ | ✗ (additive, no table) | **✔** | ✗ |
| TCA, Oldfield et al. BMVC 2021 | ~ | **✔** | ✗ | ~ | **✔** | ✗ |
| PandA, Oldfield et al. ICLR 2023 | **✔** (parts × appearances) | **✔** (unsup. semi-NMF tensor factorisation of frozen feature maps) | ✗ | ~ (parts ⊗ appearances) | **✔** | ✗ |
| AGE, CVPR 2022 | ✗ | ✔ (sparse dictionary, unsup.) | ~ (class mean = one address entry) | ✗ (additive) | **✔** | ✗ |
| Cluster-guided synthesis, CVPR 2022 | ~ (per-layer clusters) | ✗ (clusters, not subspaces) | ~ (combine clusters across layers) | ✗ | **✔** | ✗ |
| Cao & Yang 2015 / Ginart 2019 / Unlearn-ALS 2023 / ridge-head 2026 | — | — | — | — | — | **✔**, on other objects |

**The literature splits cleanly into two halves that nobody has joined.** The *multilinear* half
(rows 1–5) owns A3+A4 and pays for it with a label-built basis and a linear renderer. The
*frozen-generator* half (rows 6–10) owns A2+A5 and has no label-indexed tensor head at all — its
labels, when present, enter as a single additive direction or a cluster, not as a request that
indexes a product table. **No row has A6.**

---

## 2. The closest three, with what each shares and lacks

**(1) TensorFaces — Vasilescu & Terzopoulos, ECCV 2002.** *The dangerous one.*
`D = Z ×₁ U_pixels ×₂ U_views ×₃ U_illums ×₄ U_people` — the mode matrices are leading orthonormal
(principal) subspaces of the flattened data, synthesis for a factor combination is exactly a
tensor-product indexing of a coefficient core, and the paper itself states it *subsumes PCA as the
1-factor case* and bilinear style–content as the 2-factor case (verified against the paper's own
text, not a summary).
*Shares:* the request-indexed tensor head over principal subspaces — i.e. **our A3+A4, and the
spirit of A2**.
*Lacks:* it needs a **complete labelled Cartesian-product tensor** (one image per
person × view × illumination × expression); its bases are **defined by the label modes**, so content
is *not* unsupervised and labels do *not* enter "only at the interface"; there is no partition into
wells each carrying its own principal subspace; rendering is a **linear pixel basis**, not a frozen
nonlinear decoder; and — checked by full-text grep — **no deletion, removal or downdating anywhere**.

**(2) MLC-VAE — Georgopoulos, Chrysos, Pantic, Panagakis, ICML 2020 (arXiv:2009.04075).**
Per-class latent Gaussian whose **mean is a multilinear (CP/Tucker) function of the one-hot label
vectors**, built precisely to **generate unseen attribute combinations**.
*Shares:* A3+A4 and the *purpose* — combinatorial generalisation from a product-structured label
head. This is the closest modern statement of our composition claim.
*Lacks:* verified from the paper text — "**train the model end-to-end**"; the decoder is trained
jointly, the basis is learned not fixed, there is no principal subspace, no well partition, and
**no mention of deletion, unlearning or forgetting** (grep over the full extracted text returns
nothing).

**(3) PandA — Oldfield et al., ICLR 2023 (with GANSpace, NeurIPS 2020, behind it).**
Unsupervised (semi-nonnegative) tensor factorisation of a **frozen** GAN's feature maps into
**parts × appearances**, giving a block/part structure with its own unsupervised bases and no labels.
*Shares:* A1+A2+A5 — the well-like partition, the unsupervised subspace content, the frozen decoder.
GANSpace additionally makes A2 literal (PCA of features) and applies directions **layer-wise**,
which is the nearest published thing to "wells".
*Lacks:* there is **no label request and no address book** — nothing indexes coefficients by
factor–value terms — and there is no deletion.

*(Honourable mentions the paper should cite: **CausalX / M-mode Block SVD** (Vasilescu & Kim) — the
one prior work with a genuine **block** multilinear factorisation, whose object is literally "a
combinatorial choice of part representations", and which even has an *incremental* bottom-up update
rule — but for recognition, without a frozen decoder, and with no deletion; SeFa/InterFaceGAN-style
label-supervised directions in a frozen generator, the additive non-tensor ancestor of our address
book; and "Compositional Generalization via Forced Rendering of Disentangled Latents"
(arXiv:2501.18797) together with Compositional Risk Minimization, which supply the modern theory
statement that latent factorisation can fail to survive the decoder — direct motivation for why our
read stays linear right up to a frozen renderer.)*

---

## 3. Exact unlearning of PCA / count / covariance models — searched separately

**Second claim-stop.** ⛔ **Do not claim, as a contribution, that "downdating a covariance and
re-eigendecomposing yields exactly the refit without those rows."** That is not new; it is a
sufficient-statistic identity with a 25-year numerical literature behind it:

- **Hall, Marshall & Martin**, *Adding and subtracting eigenspaces* (BMVC 1999; Image and Vision
  Computing 20:1009–1016, 2002) — eigenspace **subtraction** without reference to the original data.
  The word "subtracting" is in the title.
- **EVD dualdating** (decremental/incremental PCA) — simultaneous arbitrary add **and delete** on the
  EVD of the covariance, with an optimality theorem under a low-rank-plus-shift condition.
- **Gu & Eisenstat** (SIAM J. Matrix Anal. Appl. 15:1266–1276, 1994) — stable rank-one modification
  of the symmetric eigenproblem; the numerical primitive underneath.
- **Cao & Yang**, *Towards Making Systems Forget with Machine Unlearning* (IEEE S&P 2015) — the
  general result that any statistical-query / **summation-form** learner (count tables, covariances)
  forgets exactly by updating its summations. Covariance-based PCA is a textbook instance.
- **Ginart et al.**, *Making AI Forget You* (NeurIPS 2019) — deletion-efficient **k-means**: the
  canonical "exact deletion of an unsupervised count/centroid model".
- **Xu et al.**, *Netflix and Forget* (arXiv:2302.06676) — **exact** unlearning for **bi-linear**
  matrix-completion recommenders (Unlearn-ALS), proved equivalent to ALS retraining on the remainder.
- **Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models**
  (arXiv:2603.12977) — the closest architectural analogue: frozen backbone + closed-form head, exact
  add/delete by maintaining the feature Gram matrix and feature–label moment, verified to <1e-9
  relative Frobenius error. It states explicitly that it covers **ridge regression only — "no PCA,
  eigendecomposition, or generative models."**

**What is genuinely absent.** Nobody applies exact deletion to a *generative* object of this kind.
The one paper that goes looking — **Approximate Data Deletion in Generative Models**
(arXiv:2206.14439) — opens by saying unsupervised/generative deletion is largely an open problem and
then settles for a **density-ratio approximation**. The generative-unlearning surveys concur that
retrain-equivalence is treated as unreachable there and is replaced by behavioural proxies.
Meanwhile "Machine Unlearning in Low-Dimensional Feature Subspace" (arXiv:2601.22456) does update a
covariance inside a constrained subspace, but is **approximate** by its own statement.

So the survivable form of the deletion claim is not "we can downdate a covariance". It is:
**the whole generative object is a sufficient statistic, so a generative model with a
request-indexed tensor head admits deletion that is exactly the refit — a guarantee the generative
unlearning literature currently calls out of reach and replaces with approximations.**

---

## 4. What survives — the differentiator, stated so the paper can lean on it

Since no prior work is IDENTICAL, the differentiator check ("does the matching work also do exact
deletion on the same object?") resolves the same way for **every** candidate: **no.** Not
TensorFaces, not the bilinear style–content models, not the multilinear 3DMMs, not CausalX, not
MLC-VAE, not GANSpace / TCA / PandA / AGE. **Exact deletion on this object is unclaimed.**

Three things survive as defensible, in descending order of strength:

1. ⭐ **Exact deletion of a compositional generative model** — refit-equal erasure of a *generator*,
   against a literature that has only approximate generative deletion. Strongest claim; lead with it.
2. **The interface split** — content bases computed **unsupervised** (no label-organised data tensor,
   hence no Cartesian-product training requirement, which is precisely the constraint TensorFaces and
   the multilinear 3DMMs cannot shed), labels entering **only** through the address book, in front of
   a **frozen** nonlinear decoder. This is the join of two literatures that have stayed apart.
3. **Wells** — a partition into blocks each holding its own leading principal subspace, read
   concatenatively. Weakest of the three: CausalX has block multilinear factorisation, PandA has
   parts, GANSpace has layer-wise ranges. Claim the use, not the idea.

⛔ **Do not claim**: "multilinear/bilinear factored generation with a request-indexed tensor head"
(TensorFaces 2002; Tenenbaum–Freeman 2000); "tensor-product label conditioning for unseen attribute
combinations" (MLC-VAE, ICML 2020); "covariance downdate then re-eigendecompose equals an exact
refit" (Hall–Marshall–Martin 2002; Cao–Yang 2015).

---

## 5. Honest limits of this scout

- Web scout only: no paywalled venue sweep and no systematic citation-graph crawl of the very large
  forward-citation set of TensorFaces. A dedicated forward-citation pass on TensorFaces and on
  MLC-VAE is the one follow-up that could still turn up a closer match, and is the cheapest
  insurance before submission.
- The task's own named failure mode ("cannot separate compositional-generation claims from ours
  because ours is unusual") partly applies. The *compositional-generation* claim on its own is
  crowded and I do not believe it can be won on its own. The claim that separates cleanly is
  deletion.

---

## Sources

- [TensorFaces — Multilinear Analysis of Image Ensembles (ECCV 2002)](http://alumni.media.mit.edu/~maov/tensorfaces/eccv02_corrected.pdf)
- [Tenenbaum & Freeman — Separating Style and Content with Bilinear Models (Neural Computation 2000)](https://people.csail.mit.edu/billf/publications/Separating_Style_and_Content.pdf)
- [Vlasic et al. — Face Transfer with Multilinear Models (SIGGRAPH 2005)](https://homes.cs.washington.edu/~jovan/papers/vlasic-2005-ftm.pdf)
- [Vasilescu — CausalX: Causal eXplanations and Block Multilinear Factor Analysis (arXiv:2102.12853)](https://arxiv.org/pdf/2102.12853)
- [Georgopoulos et al. — Multilinear Latent Conditioning for Generating Unseen Attribute Combinations (ICML 2020, arXiv:2009.04075)](https://arxiv.org/abs/2009.04075)
- [Härkönen et al. — GANSpace: Discovering Interpretable GAN Controls (NeurIPS 2020)](https://arxiv.org/abs/2004.02546)
- [Oldfield et al. — Tensor Component Analysis for Interpreting the Latent Space of GANs (BMVC 2021)](https://arxiv.org/abs/2111.11736)
- [Oldfield et al. — PandA: Unsupervised Learning of Parts and Appearances in the Feature Maps of GANs (ICLR 2023)](https://arxiv.org/abs/2206.00048)
- [Georgopoulos et al. — Cluster-guided Image Synthesis with Unconditional Models (CVPR 2022)](https://arxiv.org/abs/2112.12911)
- [Attribute Group Editing for Reliable Few-shot Image Generation (CVPR 2022)](https://github.com/unibester/age)
- [Hall, Marshall & Martin — Adding and subtracting eigenspaces (IVC 2002)](https://users.cs.cf.ac.uk/Dave.Marshall/Papers/IVCDec2002-eigenSpace.pdf)
- [EVD Dualdating Based Online Subspace Learning](https://www.researchgate.net/publication/270430439_EVD_Dualdating_Based_Online_Subspace_Learning)
- [Cao & Yang — Towards Making Systems Forget with Machine Unlearning (IEEE S&P 2015)](https://www.semanticscholar.org/paper/Towards-Making-Systems-Forget-with-Machine-Cao-Yang/2f2ade8c4944a96a44e6f70ef403b80b058d1725)
- [Ginart et al. — Making AI Forget You: Data Deletion in Machine Learning (NeurIPS 2019)](https://arxiv.org/abs/1907.05012)
- [Xu et al. — Netflix and Forget: Efficient and Exact Machine Unlearning from Bi-linear Recommendations (arXiv:2302.06676)](https://arxiv.org/abs/2302.06676)
- [Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models (arXiv:2603.12977)](https://arxiv.org/html/2603.12977)
- [Approximate Data Deletion in Generative Models (arXiv:2206.14439)](https://arxiv.org/pdf/2206.14439)
- [Machine Unlearning in Low-Dimensional Feature Subspace (arXiv:2601.22456)](https://arxiv.org/pdf/2601.22456)
- [Compositional Generalization via Forced Rendering of Disentangled Latents (arXiv:2501.18797)](https://arxiv.org/html/2501.18797)

---

# ADDENDUM — 2026-09-20: the forward-citation crawl declared NOT-RUN above

*This section is appended, not a rewrite. The verdict above stands unchanged: **ADJACENT**, with the
two claim-stops as stated. The crawl was the one follow-up §5 named as capable of overturning it. It
did not. Below is what was searched, what was found, and the one place where the crawl **corrects**
§3.*

## A. What was actually crawled

Not a search-engine sweep this time: forward-citation sets pulled from the OpenAlex API and screened
**mechanically over every title + abstract**, so nothing depends on which results a search engine
chose to surface. Screening regexes covered the six claim-atoms (A1 wells · A2 principal subspace ·
A3 factor–value request · A4 tensor-product address book · A5 frozen decoder · A6 exact deletion).

| Direction | Anchors | Unique works screened |
|---|---|---|
| **Generation-side forward crawl** | TensorFaces (ECCV'02) + its two sibling records, Tenenbaum & Freeman (2000), Face Transfer (2005), MLC-VAE (ICML'20, both records), CausalX, PandA (both records), GANSpace (both records) | **3,038** |
| **Orthogonal sweep** (cites none of our anchors) | 21 title/abstract queries over all of OpenAlex | **254** |
| **Mirror crawl** (deletion-side forward) | Cao & Yang'15, Ginart'19, Netflix-and-Forget'23, Approximate Data Deletion in Generative Models'23, Machine Unlearning Survey'23, Hall–Marshall–Martin'02 | **759** |
| **Final targeted sweep** | 11 queries on sufficient-statistic / exact-deletion phrasings | ~100 |
| | **Total** | **≈ 4,050** |

## B. Result: negative. Nothing closer than what §2 already names.

The decisive counts, from the mechanical screen:

- **Deletion language anywhere in the 3,038 generation-side citing works: 10 hits (0.3%).** On
  inspection, seven are false positives ("forget gate" in an LSTM paper, "erase"/anonymisation,
  a regularisation survey). Three are real and all are checked in §C below. **None is exact, and none
  deletes training rows.**
- **Works combining frozen-decoder + tensor-head + label-index language: 0 of 3,038.**
- **Highest atom count reached by any single work: 4 of 6 — CausalX** (A1+A2+A3+A4), exactly as §2
  already positioned it. **Nothing reached 5. Nothing reached 6.**
- **Mirror crawl:** of 759 works citing the deletion anchors, those combining PCA + generative +
  frozen language: **0**. Tensor-or-compositional + generative: **2**, both checked below.
- **Orthogonal sweep:** `"exact unlearning" + "subspace"` → 0. `"exact unlearning" + "frozen"` → 0.
  `"sufficient statistic" + "unlearning"` → 0. `"compositional" + "generative" + "exact deletion"` → 0.
  `"tensor product" + "compositional generation"` → 0. `"unlearning" + "tensor decomposition"` → 0.

**A negative crawl is the point of this addendum: it is what lets the paper lean on claim 1.** The
two literatures identified in §1 do not merely fail to overlap in the works I happened to read —
they fail to overlap across every work that cites either side's canonical papers.

## C. The five works the crawl newly surfaced, and what each takes

None takes a claim. Listed so the paper can cite them pre-emptively rather than be shown them.

1. **Feature Unlearning for Pre-trained GANs and VAEs** — Moon, Cho & Kim, **AAAI 2024**. The nearest
   thing in existence to "deletion on a frozen generator". *Takes:* nothing. It unlearns a **feature**
   (e.g. a hairstyle), not training rows; it is **approximate**, fine-tuning-based, with no retraining
   equivalence; and it **updates the generator**, so the decoder is not frozen. Cite it as the closest
   prior attempt and state the three differences.
2. **Data Redaction from Pre-trained GANs** — Kong & Chaudhuri, **SaTML 2023**. *Takes:* nothing, and
   is actively **useful to us**: it states outright that "redaction is a fundamentally different task
   from data deletion, and data deletion may not always lead to redaction". That sentence draws the
   line our claim 1 sits on the far side of. Cite it when defining what exact deletion means here.
3. **Generative Adversarial Networks Unlearning** — Sun et al., **IEEE TDSC 2025**. Substitution
   mechanism + fake labels; reports "minor degradation after unlearning". *Takes:* nothing —
   approximate by construction.
4. **Bilinear Models of Parts and Appearances in GANs** — Oldfield, Tzelepis, Panagakis, Nicolaou &
   Patras, **TPAMI 2024** (journal PandA). *Takes:* nothing new beyond what §2 credits to PandA. Worth
   noting because it is the **2024** state of the frozen-generator half and is still "entirely
   unsupervised" with **no labels and no address book** — evidence that the missing join is not an
   artefact of my citing an older paper.
5. **Incremental N-Mode SVD for Large-Scale Multilinear Generative Models** — IEEE **TIP 2014**. The
   single work in ~4,050 that bridges eigenspace-updating numerics and multilinear *generative*
   models, and therefore the most dangerous-looking hit in the crawl. *Takes:* nothing — it is
   **incremental only**, deriving how to update an N-mode SVD when **new** data arrives. There is no
   removal operator. (Same posture as CausalX's Incremental M-mode Block SVD: add, never subtract.)

## D. ⚠ CORRECTION to §3 — STOP #2 was stated too strongly

§3 told the Hub not to claim that "downdating a covariance and re-eigendecomposing equals the refit",
citing Hall, Marshall & Martin (1999/2002) among the owners. Having now read that paper's full text
rather than its abstract, **that attribution is wrong and must be narrowed.** Their own words:

> "Subtracting eigenspaces yields a result **approximating** that which would obtain were a subset of
> data used." … "The subtraction operator **tends to instability** as the number of points being
> removed [grows]" … "subtraction is best used to remove a **small fraction** of the data points." …
> "We have not performed any rigorous error analysis and hence any explanations we have … are
> **anecdotal in character**."

The reason is structural, and it is the crux: they subtract **deflated** eigenspaces — "only
'significant' eigenvectors and eigenvalues are retained". Once the tail is discarded you cannot
recover the exact covariance of the remaining rows, and the error compounds under subtraction. They
also report being "**unable to find a solution for block downdating [of SVD], except via EVD**".

**So the classical eigenspace-arithmetic line does not own exactness — it is explicitly approximate,
and its authors say so.** Our route is exact for the opposite reason: it downdates the **full,
untruncated second-moment statistic** and re-eigendecomposes afterwards, so no truncation ever enters
the subtraction.

**Revised STOP #2.** Do not claim the *primitive* — the exactness follows in one line from Cao &
Yang's summation form (2015), since a covariance is a summation, and that is the citation that owns
it. **But do not concede exactness to Hall–Marshall–Martin or to the decremental-PCA line either**
(incl. EVD-dualdating, which is optimal only under a low-rank-plus-shift condition, and "Fast and
more accurate incremental-decremental PCA", 2021, which is an online-learning approximation). A
referee who half-remembers that line may believe exact PCA deletion is solved; **it is not, and the
distinction — downdate the statistic, not the truncated factors — is worth one explicit sentence in
the paper.** That sentence is a small but real piece of what survives, and §3 undersold it.

## E. Verdict after the crawl: unchanged

**⚠ ADJACENT.** No work found is IDENTICAL; the differentiator check resolves the same way for every
candidate old and new — **none does exact deletion on this object**. The ranking in §4 stands, with
claim 1 now resting on a systematic negative rather than on absence of evidence, and claim 3's
deletion sub-point strengthened by §D.

**Residual risk, stated honestly.** OpenAlex coverage of very recent preprints (< ~3 months) and of
non-indexed workshop papers is imperfect, and abstract-level screening cannot catch a match that is
buried in a method section without surfacing in the abstract. I judge the remaining chance of a
five-minute referee find to be low but not zero. It is no longer the cheapest outstanding insurance —
that is now a careful read of the two 2024 works in §C.1 and §C.4 in full.

## Sources added by this addendum

- [Moon, Cho & Kim — Feature Unlearning for Pre-trained GANs and VAEs (AAAI 2024, arXiv:2303.05699)](https://arxiv.org/abs/2303.05699)
- [Kong & Chaudhuri — Data Redaction from Pre-trained GANs (SaTML 2023, arXiv:2206.14389)](https://arxiv.org/abs/2206.14389)
- [Sun et al. — Generative Adversarial Networks Unlearning (IEEE TDSC 2025, arXiv:2308.09881)](https://arxiv.org/abs/2308.09881)
- [Oldfield et al. — Bilinear Models of Parts and Appearances in GANs (TPAMI 2024)](https://openaccess.city.ac.uk/id/eprint/33259/)
- [Incremental N-Mode SVD for Large-Scale Multilinear Generative Models (IEEE TIP 2014)](https://ieeexplore.ieee.org/document/6873262/)
- [Hall, Marshall & Martin — Adding and subtracting eigenspaces (BMVC 1999, full text)](https://bmva-archive.org.uk/bmvc/1999/papers/45.pdf)
- [OpenAlex API](https://api.openalex.org) — citation graph used for the crawl (query set and screening scripts in the run scratchpad)
