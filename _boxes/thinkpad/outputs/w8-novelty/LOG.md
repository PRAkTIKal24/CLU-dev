# w8 spoke/novelty-scout — LOG

Role: web-scout. Read-only. No code, no scored run. Output: `NOVELTY.md` (+ this log).
Task file: `/c/chlu/tasks-w8/w8-spoke-novelty-scout.md`
Worktree: `/c/chlu/worktrees/CHLU-w8-novelty` (no code written — read-only spoke)
Outputs: `/c/chlu/outputs/w8-novelty/`

- **2026-09-20T16:36:32Z** — START_UTC.txt written (no live prior one existed). Task file read.
- **2026-09-20T16:37Z** — Decomposed the object into 6 separable claim-atoms so prior art can be
  matched atom-by-atom rather than gestalt:
  A1 partition of latent/feature space into **wells**;
  A2 each well = **fixed leading principal subspace** of training features (unsupervised, PCA/SVD);
  A3 a **request** = set of factor–value terms;
  A4 request indexes a **tensor-product address book** → real coefficients per well-direction;
  A5 concatenated read rendered by a **frozen decoder**;
  A6 **exact deletion** by covariance downdate + re-eigendecomposition == refit-without-rows.
  Prior work is scored on which atoms it takes. Verdict rule: IDENTICAL only if a single work takes
  A1–A5 as one object; the A6 differentiator is checked separately against that same work.
- **2026-09-20T16:38Z** — Search wave 1 launched (multilinear/TensorFaces/bilinear/tensor-conditioned
  generators; GAN-latent PCA families; exact unlearning of PCA and covariance/count models).
- **2026-09-20T16:47Z** — Wave 1 results (primary sources pulled where possible; PDFs extracted
  locally with `pdftotext`, not trusted to summarisers alone):
  - **MLC-VAE** (Georgopoulos, Chrysos, Pantic, Panagakis, ICML 2020, arXiv:2009.04075) — closest hit
    so far on A3/A4: per-class latent Gaussian whose **mean is a multilinear (CP/Tucker) function of
    one-hot label vectors**, explicitly to generate **unseen attribute combinations**. Verified from
    the paper text: "train the model end-to-end"; encoder/decoder trained jointly; no fixed principal
    basis; no block partition; **no deletion/unlearning anywhere** (grep for unlearn/delet/forget/
    eigen/frozen over full text → only Klys-subspace citation).  ⇒ takes A3+A4, misses A1,A2,A5,A6.
  - **TensorFaces** (Vasilescu & Terzopoulos, ECCV 2002) — N-mode SVD, mode matrices ARE leading
    principal subspaces, synthesis IS tensor-product indexing by factor labels. But: one unified
    tensor, **requires a complete labelled tensor** (every person×view×illum×expression present),
    linear pixel rendering (no frozen nonlinear decoder), no block partition, no deletion.
  - Lead opened: Vasilescu **CausalX / block multilinear factor analysis** (arXiv:2102.12853) —
    "block" could be our wells. Must check.
- **2026-09-20T16:48Z** — Wave 2: CausalX/block-multilinear, PandA, SeFa/InterFaceGAN family,
  and the deletion-side literature (Cao&Yang summation form, EVD-dualdating, Unlearn-ALS).
- **2026-09-20T16:58Z** — Wave 2 results:
  - **CausalX / M-mode Block SVD** (Vasilescu & Kim, KDD'19 wkshp; Vasilescu arXiv:2102.12853) —
    extracted full text. This is the *closest structural* relative on the A1+A2 side: a
    **hierarchical BLOCK multilinear factorization** where each part gets its own multilinear
    factorization, core is block-diagonal, and an object is "a combinatorial choice of part
    representations". Also has an **Incremental** M-mode Block SVD (update when data arrives), but
    grep confirms **no deletion/downdating/unlearning**; goal is recognition/verification, not
    generation through a frozen decoder; still needs the combinatorial (Cartesian-product) training set.
  - Deletion side is a **separate literature with no generative/tensor-head member**: Cao & Yang'15
    (summation form / SQ learning), Ginart et al.'19 (deletion-efficient k-means), Xu et al.'23
    Unlearn-ALS (exact for bilinear matrix completion), arXiv:2603.12977 (exact continual unlearning
    of a **ridge head on a frozen backbone** via Gram + feature-label moment — explicitly states "no
    PCA, eigendecomposition, or generative models"), EVD-dualdating / Hall&Marshall eigenspace
    add-subtract (numerics, not framed as unlearning).
- **2026-09-20T16:59Z** — Wave 3: hunting for a single work that is our whole object — frozen
  generator + unsupervised subspace content + label-indexed multiplicative address book. Targets:
  Cluster-Guided Image Synthesis (Georgopoulos), Attribute Group Editing dictionaries, VSA/HRR
  factor codebooks with a neural decoder, bilinear-model-on-PCA-coefficients.
- **2026-09-20T17:12Z** — Wave 3 results. Confirmed the two-halves picture: the multilinear
  generation line (Tenenbaum-Freeman, TensorFaces, Vlasic 3DMM, CausalX, MLC-VAE) owns the
  request-indexed tensor head but builds its bases out of labels and renders linearly; the
  frozen-generator line (GANSpace, TCA, PandA, AGE, cluster-guided synthesis) owns unsupervised
  subspace content behind a frozen decoder but has no label-indexed product table. Nobody joins
  them; nobody deletes. Also confirmed the generative-deletion gap: Kong et al.
  "Approximate Data Deletion in Generative Models" (arXiv:2206.14439) calls generative/unsupervised
  deletion largely open and settles for a density-ratio approximation; arXiv:2601.22456 is
  approximate by its own statement; arXiv:2603.12977 is exact but ridge-head-only and says so.
- **2026-09-20T17:18Z** — `NOVELTY.md` written. Verdict **ADJACENT**, with TWO explicit claim-stops.

---

# FINAL REPORT (last section of LOG.md)

**Spoke:** w8 spoke/novelty-scout. **Dial:** novelty. **Mode:** read-only web scout.
**Deliverable:** `/c/chlu/outputs/w8-novelty/NOVELTY.md`.

## Verdict: ⚠ ADJACENT — with two claims STOPPED

Not CLEAR. A referee finds TensorFaces in five minutes and is right to.

**STOP #1.** The claim *"multilinear/bilinear factored generation with a request-indexed tensor
head"* is already owned: Tenenbaum & Freeman (Neural Computation 2000) for the bilinear case,
Vasilescu & Terzopoulos TensorFaces (ECCV 2002) for the N-factor case — mode matrices that ARE
leading principal subspaces, synthesis by tensor-product indexing of a coefficient core, and the
paper states outright that it subsumes PCA (1-factor) and bilinear style-content (2-factor).
Restated deep by Vlasic et al. (SIGGRAPH 2005) and by MLC-VAE (Georgopoulos et al., ICML 2020,
CP/Tucker core contracted with one-hot labels, explicitly for UNSEEN attribute combinations).
⛔ The Hub must not claim that sentence.

**STOP #2.** The claim *"downdating a covariance and re-eigendecomposing equals the refit without
those rows"* is a sufficient-statistic identity with 25 years of numerics behind it:
Hall/Marshall/Martin "Adding and subtracting eigenspaces" (BMVC'99, IVC 2002), EVD-dualdating
decremental PCA, Gu & Eisenstat (1994), and Cao & Yang's summation form (IEEE S&P 2015) which covers
count/covariance learners generically. ⛔ Do not claim the primitive.

## Why it is ADJACENT and not IDENTICAL

The six claim-atoms (A1 wells; A2 fixed leading principal subspace, unsupervised; A3 factor-value
request; A4 tensor-product address book; A5 frozen nonlinear decoder; A6 exact deletion) are split
across two literatures that have never been joined. The multilinear line takes A3+A4 and part of A2
but builds its bases FROM the labels, needs a complete Cartesian-product labelled tensor, and renders
linearly. The frozen-generator line (GANSpace, TCA, PandA, AGE) takes A1+A2+A5 but has no label
request and no product table. **No work in either half has A6.**

One-sentence difference: *prior multilinear generation builds its factor bases out of the labels,
renders linearly, and never deletes; ours keeps the content bases label-free, lets labels touch the
model only at an address book in front of a frozen nonlinear decoder, and deletes exactly by
downdating the sufficient statistic.*

## What survives (differentiator check, run against every candidate: none does exact deletion)

1. ⭐ **Exact deletion of a compositional generative model** — refit-equal erasure of a *generator*.
   The only paper that looks (arXiv:2206.14439) calls generative deletion largely open and settles
   for approximation; the exact-unlearning results that exist are for k-means, bilinear matrix
   completion, and ridge heads on frozen backbones (the last explicitly excludes PCA,
   eigendecomposition and generative models). **Lead with this.**
2. **The interface split** — unsupervised content bases (so no Cartesian-product training set, the
   constraint TensorFaces cannot shed) + labels only in the address book + frozen nonlinear decoder.
3. **Wells** — claim the use, not the idea (CausalX has block multilinear factorisation, PandA has
   parts, GANSpace has layer ranges).

## Nearest three, for citation regardless of verdict
TensorFaces (ECCV 2002); MLC-VAE (ICML 2020); PandA (ICLR 2023) with GANSpace (NeurIPS 2020) behind
it. Plus CausalX / M-mode Block SVD as the one genuine block-multilinear ancestor.

## NOT-RUN declarations (a declared NOT-RUN is never a null)
- **No code, no scored run, no numbers produced.** This spoke is read-only by its task file; there is
  therefore also **no `chlu.__file__` resolution check to report** — nothing was imported or executed.
- **NOT RUN: paywalled-venue sweep** (IEEE/ACM/Springer full text beyond open PDFs).
- **NOT RUN: systematic forward-citation crawl** of TensorFaces and of MLC-VAE. This is the single
  follow-up that could still surface a closer match and is the cheapest pre-submission insurance.
- **NOT RUN: any assessment of whether our implementation works.** Novelty only.

## Artefacts
- `/c/chlu/outputs/w8-novelty/START_UTC.txt` — 2026-09-20T16:36:32Z
- `/c/chlu/outputs/w8-novelty/NOVELTY.md` — the verdict, atom table, closest three, deletion section
- `/c/chlu/outputs/w8-novelty/LOG.md` — this file
- `/c/chlu/outputs/w8-novelty/w8-novelty.bundle` — branch bundle (no code changes; worktree clean)

---

# ADDENDUM (appended after the final report — nothing above rewritten)

- **2026-09-20T16:44Z** — Hub asked for the forward-citation crawl that the final report above
  declared NOT-RUN. Semantic Scholar API returned 429 (shared unauthenticated pool); switched to the
  **OpenAlex API**, which is reachable and generous. Decision: pull full forward-citation sets to
  disk and screen **every** title+abstract mechanically with atom regexes, rather than reading
  search-engine summaries — a negative is only worth stating if it is exhaustive over a defined set.
- **2026-09-20T16:45Z** — Generation-side crawl. Anchors resolved and crawled: TensorFaces
  W2113055885 (850 citing) + siblings W2160551639 (381) / W2131691213 (234); Tenenbaum & Freeman
  W2170653751 (876); Face Transfer W2135666716 (581); MLC-VAE W3035589463 (7) + W3084065867 (4);
  CausalX W3131867783 (2); PandA W4281710985 (10) + W4322420602 (2); GANSpace W3014852036 (426) +
  W4287817254 (359). **3,038 unique works** with abstracts → `crawl/citing.index.jsonl`.
- **2026-09-20T16:46Z** — Screen 1. A6 (deletion/unlearning language) fires on **10 of 3,038**;
  frozen + tensor-head + label-index fires on **0 of 3,038**. Checked the screen was not too narrow:
  atom prevalence A4=535, A3=431, A2=266, A5=169, A1=32, COMPO=28 — regexes are live, the joint is
  genuinely empty. Highest atom count reached by any single work = **4 (CausalX, A1A2A3A4)**; nothing
  at 5 or 6. This is exactly where §2 of NOVELTY.md already placed CausalX.
- **2026-09-20T16:47Z** — Ran down all 10 A6 hits. Seven false positives (LSTM "forget gate",
  anonymisation "erase", a GAN-regularisation survey). Three real, all verified against source:
  Feature Unlearning for Pre-trained GANs and VAEs (AAAI'24) — unlearns a **feature**, approximate,
  **fine-tunes the generator**; Data Redaction from Pre-trained GANs (SaTML'23) — states outright
  "redaction is a fundamentally different task from data deletion"; GANs Unlearning (TDSC'25) —
  substitution + fake labels, "minor degradation". **None is exact; none deletes rows.**
- **2026-09-20T16:48Z** — Orthogonal sweep, 21 OpenAlex title/abstract queries designed to catch a
  match citing none of my anchors. **254 new unique works.** Notable zeros:
  `"exact unlearning"+"subspace"` 0 · `"exact unlearning"+"frozen"` 0 ·
  `"sufficient statistic"+"unlearning"` 0 · `"unlearning"+"tensor decomposition"` 0 ·
  `"tensor product"+"compositional generation"` 0 · `"compositional"+"generative"+"exact deletion"` 0.
- **2026-09-20T16:50Z** — **Mirror crawl** (my own idea, not requested): forward citations of the
  *deletion* anchors, screened for the *generation* atoms — the reverse direction, so the negative is
  not an artefact of which side I started from. Cao&Yang (589), Ginart (73), Netflix-and-Forget (2),
  Approximate Data Deletion in Generative Models (3), Unlearning Survey (150), Hall&Marshall (70) →
  **759 unique**. PCA + generative + frozen: **0**. Tensor-or-compositional + generative: **2**.
- **2026-09-20T16:51Z** — Chased the more dangerous of those 2: **"Incremental N-Mode SVD for
  Large-Scale Multilinear Generative Models" (IEEE TIP 2014)** — the only work in ~4,050 bridging
  eigenspace-update numerics and multilinear *generative* models. Verified: **incremental only, no
  removal operator.** Same posture as CausalX's incremental block SVD — add, never subtract.
- **2026-09-20T16:53Z** — ⚠ **Correction found, and it goes in our favour.** Pulled the Hall,
  Marshall & Martin BMVC'99 full text (the IVC'02 mirror 404'd) rather than trusting its abstract.
  Their own words: subtraction "yields a result **approximating**" the refit; "**tends to
  instability** as the number of points being removed" grows; "best used to remove a **small
  fraction**"; "no rigorous error analysis … **anecdotal in character**"; and they were "unable to
  find a solution for block downdating [of SVD], **except via EVD**". Root cause: they subtract
  **deflated** eigenspaces ("only 'significant' eigenvectors and eigenvalues are retained"), so
  truncation error compounds under subtraction. ⇒ **§3's STOP #2 over-attributed exactness to that
  line.** Narrowed it in the addendum: the exactness is Cao & Yang's summation form, not
  Hall–Marshall–Martin, and the eigenspace-arithmetic literature is *explicitly approximate*. The
  distinction — downdate the untruncated statistic, not the truncated factors — is a real, small,
  defensible piece of what survives, and the final report above undersold it.
- **2026-09-20T16:55Z** — Final targeted sweep (11 queries on sufficient-statistic / exact-deletion
  phrasings): returns genomics noise only. Nothing.
- **2026-09-20T16:57Z** — Addendum appended to `NOVELTY.md` as a new dated section. **Verdict above
  NOT rewritten** — §§1–5 stand verbatim; the addendum adds §§A–E and flags the one correction in §D.
- **2026-09-20T16:58Z** — Crawl made auditable: scripts + compact indices (id/anchor/year/title for
  all 4,051 screened works) copied to `crawl/` in outputs. Re-bundled the branch.

## ADDENDUM TO THE FINAL REPORT

**The crawl is a negative, and that is the deliverable.** ~4,050 unique works screened across three
independent directions (forward from the generation anchors, forward from the deletion anchors, and
an orthogonal all-of-OpenAlex sweep). **Nothing closer than what the report above already names.**
No work reaches 5 of 6 atoms; the maximum is CausalX at 4. No work in either citation network
combines exact deletion with a compositional/multilinear generative object.

**Verdict unchanged: ⚠ ADJACENT.** Claim 1 (exact deletion of a compositional generative model) now
rests on a systematic negative rather than on absence of evidence — which is precisely what the Hub
asked the crawl to establish.

**One correction to carry forward:** STOP #2 in the report above is too strong as written. Do not
claim the downdate primitive (Cao & Yang 2015 owns it in one line), **but do not concede exactness to
the eigenspace-arithmetic line either** — Hall–Marshall–Martin is approximate and unstable by its
authors' own statement, because it subtracts truncated factors. See §D of `NOVELTY.md`.

**Still NOT RUN, honestly:** full-text reads of the two 2024 works (Feature Unlearning for Pre-trained
GANs/VAEs; Bilinear Models of Parts and Appearances) — abstract-and-targeted-check only. That is now
the cheapest outstanding insurance, ahead of any further crawling. OpenAlex coverage of preprints
younger than ~3 months and of non-indexed workshop papers remains imperfect, and abstract-level
screening cannot catch a match buried in a method section. Residual risk of a five-minute referee
find: low, not zero.

- **2026-09-20T17:02Z** — Spent the outstanding insurance: full-text reads of the two 2024 works.
  **Feature Unlearning for Pre-trained GANs and VAEs** (AAAI'24) — confirmed takes nothing: no lookup/
  address book (a single mean-difference target vector), generator is **fine-tuned** ("we initialize g_theta
  from the pre-trained f"), unlearns a *feature* not rows ("Unlike a common unlearning task where an
  unlearning target is a subset of the training set, we aim to unlearn a specific feature"), no exactness
  guarantee, no PCA/eigen/covariance, no block partition. **PandA / Bilinear Models of Parts and
  Appearances** (TPAMI'24) — extracted full text locally: deletion-atom grep (unlearn|delet|forget|downdat|
  erasure) returns **0 across the entire paper**; "entirely unsupervised", "without using any labels";
  bases from semi-nonnegative tensor factorization (SVD used only for initialisation), pre-trained GAN.
  ⇒ Neither takes a claim. The last named gap in the crawl is now closed; nothing above is revised.

**FINAL RESIDUAL-RISK JUDGEMENT (one line).** With ~4,050 works screened in three directions, both
named 2024 full texts now read clean, and every candidate failing the deletion check, I judge the risk
that a referee finds prior art taking our surviving claims to be **low — the composite object and
claim 1 are safe to publish — with the residue confined to sub-3-month preprints, non-indexed workshop
papers, and matches buried in method sections that no abstract-level screen can reach.**
