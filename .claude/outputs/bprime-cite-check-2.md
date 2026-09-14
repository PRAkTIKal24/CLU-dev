# bprime-cite-check-2 — web-scout report
Task + acceptance criterion: verify the two remaining single-sourced citation items (Poliak et al. \*SEM 2018 incl. the "6 of 10" figure quoted in draft §5.3; Mamba-2 citation facts + the SSD state-size convention), each with a VERIFIED / CORRECTED / UNVERIFIABLE verdict and ready-to-paste BibTeX.
Status: **done** — **both items VERIFIED**, each double-sourced. Two small CORRECTIONS to wording, one tooling erratum, zero retractions.
**Dial declaration (protocol §7):** none — instrument/recon (citation verification). No performance number, no laundering control, nothing falsifiable in the dial sense.

## ⛔ DOWNSTREAM RECONCILIATION LIST (owner needed — protocol §5 corollary, stated in the first 10 lines)
1. **r4 writer (⟦CITE2⟧):** draft §5.3 l.1036 currently says Poliak et al. *"beat the majority baseline on 6 of 10 NLI datasets."* The published sentence is **"significantly outperforms"** and the count is over the **test-set** column. Recommended print form: *"…train on hypotheses alone and **significantly outperform** the majority-class baseline on **six of the ten** NLI datasets"*. The number **6 of 10 is CORRECT as quoted** — this is a precision edit, not a fix.
2. **r4 writer / curator:** `draft-r3.md` §12 item 6 (*"one remaining single-sourced citation to verify"*) → **DISCHARGE**. Both halves of §5.3's conceded-ancestry sentence (Poliak; Feng et al.) are now verified against published text.
3. **Curator / anyone auto-importing BibTeX:** ⚠ **Semantic Scholar mislabels this paper's venue as "International Workshop on Semantic Evaluation" (SemEval)** (raw JSON below). **\*SEM ≠ SemEval.** Never take S2's venue string for this entry; use the ACL Anthology record (S18-2023).
4. **r4 writer / `bprime-mamba2-arm` reconciliation 4:** the Mamba-2 citation facts are VERIFIED **and upgradeable** — the authoritative published record is **PMLR v235:10041–10071**, not just "ICML 2024". Also: **author order is Dao & Gu for Mamba-2 but Gu & Dao for Mamba-1** — do not let the two cites drift into the same order.
5. **Program tooling note (new, cheap):** `scholar.archive.org` (IA Scholar) full-text phrase search **works and reaches published proceedings PDFs** — it is the tool that closed the Poliak double-source where the ACL PDF fetch failed. Add it to the scout kit alongside the standing OpenReview block. Also: `Read` on a fetched PDF **fails on this machine** (`pdftoppm is not installed` — no poppler), so PDF-only sources remain unreadable.

---

## Answer first
**Item 1 (Poliak) — VERIFIED, double-sourced, including the number.** Anthology ID **S18-2023**, *SEM 2018, New Orleans, **pp. 180–191**, **DOI 10.18653/v1/S18-2023**, five authors as cited. The paper states verbatim: *"Across six of the ten datasets, our hypothesis-only model significantly outperforms the majority-baseline, even outperforming the best reported results on one dataset, recast SPR."* IA Scholar returns this exact sentence from **both** the arXiv record **and** the *SEM proceedings record, so — unlike the Feng et al. case — **the preprint and camera-ready wordings do not diverge here**; the draft may quote either. The count is independently confirmed by the paper's own Table 2 (4 of 10 datasets score *identically* to the majority baseline ⇒ 6 with gains).
**Item 2 (Mamba-2) — VERIFIED, double-sourced.** **Tri Dao & Albert Gu (2024), "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024 (PMLR 235:10041–10071), arXiv:2405.21060.** The SSD state convention **d_head · d_state per head** is confirmed in the official reference implementation in two independent files (`mamba2.py`: `ssm_state = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, …)`; `ssd_minimal.py`: state einsum `…->bchpn`, p = head dim, n = state dim) and by the authors' own companion post (*"a single SSM head has total state size P × N"*). ⚠ The **paper PDF's own text could not be extracted with the tooling on this machine** (arXiv HTML 404, ar5iv fatal conversion error, no poppler) — see §2.4.

---

## 1. Poliak et al. — **VERIFIED** (venue, pages, DOI, authors, and the "6 of 10")

### 1.1 Citation facts (double-sourced)
| fact | source 1 | source 2 | grade |
|---|---|---|---|
| Title *"Hypothesis Only Baselines in Natural Language Inference"* | `aclanthology.org/S18-2023/` | `arxiv.org/abs/1805.01042` | **double** |
| Authors: **Adam Poliak, Jason Naradowsky, Aparajita Haldar, Rachel Rudinger, Benjamin Van Durme** (in order) | ACL Anthology `.bib` | Semantic Scholar Graph API on `DOI:10.18653/v1/S18-2023` (identical order) | **double** (+ Bryn Mawr repo, Edinburgh RE) |
| Venue: *Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics* (**\*SEM 2018**), New Orleans, LA, June 2018, ACL | ACL Anthology | `research.ed.ac.uk` publication record (*"7th Joint Conference on Lexical and Computational Semantics"*, New Orleans, 1 Jun 2018) | **double** |
| **Pages 180–191** | ACL Anthology `.bib` (`pages = "180--191"`) | Edinburgh RE + Bryn Mawr repository record (both "180-191") | **double** |
| **DOI 10.18653/v1/S18-2023**, anthology ID **S18-2023** | ACL Anthology | S2 `externalIds` (`"ACL":"S18-2023"`, `"DOI":"10.18653/v1/S18-2023"`, `"DBLP":"conf/starsem/PoliakNHRD18"`) | **double** |
| arXiv preprint **1805.01042**, 2 May 2018, comments *"Accepted at \*SEM 2018 as long paper"* | arXiv abs | S2 `externalIds.ArXiv` | **double** |
| Editors (for a full ACL-style entry): Malvina Nissim, Jonathan Berant, Alessandro Lenci | ACL Anthology `.bib` | — | single (registry-native; it *is* the record) |

⚠ **DBLP's key `conf/starsem/…` and S2's venue string disagree**: S2 reports `"venue": "International Workshop on Semantic Evaluation"` with alternate name `"SemEval"`. This is an S2 venue-clustering error (\*SEM and SemEval are distinct; SemEval 2018 was a *co-located workshop*, \*SEM 2018 the joint conference). **The ACL Anthology record governs.** Recorded here so nobody "corrects" the draft to SemEval later.

### 1.2 The "6 of 10" figure — **CORRECT as quoted; wording should tighten**
- **Verbatim, §5 Results:** *"Across six of the ten datasets, our hypothesis-only model significantly outperforms the majority-baseline, even outperforming the best reported results on one dataset, recast SPR."*
- **Double-sourcing of that exact sentence:** `scholar.archive.org` (IA Scholar) full-text phrase search on `"Across six of the ten datasets"` returns **2 hits, both this paper — one indexed under the arXiv record, one under *"Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics"*** — i.e. the sentence is present in the **published proceedings text**, not only the preprint. Second, independent rendering: `ar5iv.labs.arxiv.org/html/1805.01042` (LaTeX-source rendering of the arXiv version) returns the identical sentence.
  ⭐ **This closes the specific risk `bprime-cite-check` flagged on Feng et al.** (arXiv wording ≠ ACL wording). Here they match, so §5.3 may quote without a version caveat.
- **Internal consistency check (adversarial, from the paper's own Table 2, dev/test accuracy, hypothesis-only vs. majority):**

| dataset | dev (hyp-only / MAJ) | test (hyp-only / MAJ) | gain? |
|---|---|---|---|
| DPR | 50.21 / 50.21 | 49.95 / 49.95 | no (identical) |
| SPR | 86.21 / 65.27 | **86.57 / 65.44** | ✅ |
| FN+ | 62.43 / 56.79 | **61.11 / 57.48** | ✅ |
| ADD-1 | 75.10 / 75.10 | 85.27 / 85.27 | no (identical) |
| SciTail | 66.56 / 50.38 | **66.56 / 60.04** | ✅ |
| SICK | 56.76 / 56.76 | 56.87 / 56.87 | no (identical) |
| MPE | 40.20 / 40.20 | 42.40 / 42.40 | no (identical) |
| JOCI | 61.64 / 57.74 | **62.61 / 57.26** | ✅ |
| SNLI | 69.17 / 33.82 | **69.00 / 34.28** | ✅ |
| MNLI (matched / mismatched dev) | **55.52 / 35.45**, **55.18 / 35.22** | — (test labels withheld) | ✅ |

⇒ exactly **four** datasets at parity (DPR, ADD-1, SICK, MPE) and **six** with gains (**SPR, FN+, SciTail, JOCI, SNLI, MNLI**). The sentence and the table agree. (Table values transcribed via ar5iv; treat the individual decimals as *ar5iv-sourced, single-source* — the **count**, which is what the draft prints, is double-sourced. ⛔ Do not print these decimals in the paper without a re-check; the draft does not need them.)
- **Third-party corroboration of the SNLI figure** (independent, peer-reviewed): Belinkov, Poliak, Shieber, Van Durme & Rush (2019), *"On Adversarial Removal of Hypothesis-only Bias in Natural Language Inference"* (\*SEM 2019, arXiv:1907.04389), verbatim: *"Out of 10 NLI datasets, Poliak et al. (2018) found that the Stanford Natural Language Inference dataset (SNLI; Bowman et al., 2015) contained the most (or worst) hypothesis-only biases — their hypothesis-only model outperformed the majority baseline by roughly 100% (going from roughly 34% to 69%)."* Confirms **10 datasets** and the SNLI 34→69 jump; it does **not** state the "6 of 10" count, so it corroborates the setup, not the count.
- **Precision note for the r4 writer.** The paper's verb is **"significantly outperforms"** (statistical significance, not just a higher number) — the draft's *"beat"* under-states a stronger claim. And the audit-relevant framing for §5.3 is that this is a **partial-input baseline that succeeds**, which is the direction Feng et al.'s caveat says *is* informative — the two cites are load-bearing in complementary directions and the draft already uses them that way.
- **Verdict: VERIFIED** (all citation facts + the quoted number). Optional wording upgrade only.

---

## 2. Mamba-2 — **VERIFIED**, with one recommended upgrade (add the PMLR locus)

### 2.1 Citation facts (double-sourced)
| fact | source 1 | source 2 | grade |
|---|---|---|---|
| Title *"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"* | `proceedings.mlr.press/v235/dao24a.html` | `arxiv.org/abs/2405.21060` | **double** |
| Authors **Tri Dao, Albert Gu** (this order) | PMLR | arXiv abs | **double** |
| Venue **ICML 2024** (41st ICML), **PMLR volume 235, pages 10041–10071** | PMLR proceedings page (authoritative published record) | arXiv comments field: *ICML 2024* | **double** (pages/volume single-sourced to PMLR, which *is* the record) |
| **arXiv:2405.21060**, cs.LG, submitted **31 May 2024 [v1]** | arXiv abs | PMLR entry links the same preprint | **double** |
| Nickname "Mamba-2" is the paper's own | abstract, verbatim: *"…allows us to design a new architecture (**Mamba-2**) whose core layer is a refinement of Mamba's selective SSM that is 2-8X faster, while continuing to be competitive with Transformers on language modeling."* | — | primary |

⇒ `bprime-mamba2-arm` §"Citation ledger (reconciliation 4)" is **correct as written**; **reconciliation 4 is discharged.** Recommended upgrade: cite the PMLR locus rather than a bare "ICML 2024".
⚠ **Author-order trap:** Mamba-1 = **Gu & Dao** (arXiv:2312.00752, COLM 2024); Mamba-2 = **Dao & Gu**. `rival-recon` §1.4 already has both right; keep it that way in the r4 bibliography.

### 2.2 The SSD state-size convention — **VERIFIED: state per head = `d_head × d_state`**
- **Official implementation, `state-spaces/mamba`, `mamba_ssm/modules/mamba2.py`** (defaults `d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256`), verbatim:
  ```python
  ssm_state = torch.zeros(
      batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
  )
  conv_state = torch.zeros(
      batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
  ).transpose(1, 2)
  ```
  ⇒ **per head: `headdim · d_state` elements**; per layer `nheads · headdim · d_state = d_inner · d_state` (+ the conv window). `_get_states_from_cache` allocates identically.
- **Second, independent file in the same repo — the reference SSD kernel `mamba_ssm/modules/ssd_minimal.py`:**
  ```python
  def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
      """
      Arguments:
          X: (batch, length, n_heads, d_head)
          A: (batch, length, n_heads)
          B: (batch, length, n_heads, d_state)
          C: (batch, length, n_heads, d_state)
      Return:
          Y: (batch, length, n_heads, d_head)
      """
  states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
  ```
  ⇒ the chunk state carries index pattern `b c h **p n**` with `p = d_head`, `n = d_state`; inter-chunk state shape `(batch, chunk, n_heads, d_head, d_state)`. **The outer-product `B ⊗ X` construction is exactly `d_state × d_head` per head.**
- **Third source, authors' own companion post** (Tri Dao, *"State Space Duality (Mamba-2) Part I – The Model"*, tridao.me/blog/2024/mamba2-part1-model/), verbatim: *"a single SSM head has total state size **P × N**, which are each governed by separate scalar recurrences in Mamba-1 but are controlled by a single shared recurrence in Mamba-2."* Same post: *"Compared to Mamba-1, Mamba-2 allows much larger state dimensions (from `N=16` in Mamba-1 to `N=64` to `N=256` or even higher in Mamba-2) while simultaneously being much faster during training"*, and P (head dim) *"= 64 or 128"*.
- ⇒ **`rival-recon` §1.4's pinned record is confirmed**: Mamba-2 per-layer state `= (2·d_model + 2·ngroups·d_state)·d_conv + 2·d_model·d_state` elements at defaults (with `d_inner = expand·d_model = 2·d_model`), i.e. **264·d_model + 1024** at `d_state=128, d_conv=4, expand=2, ngroups=1`. Arithmetic re-checked here: conv term `(2·d_model + 2·1·128)·4 = 8·d_model + 1024`; SSM term `2·d_model·128 = 256·d_model`; total **264·d_model + 1024** ✅.
- ⇒ **The arm's `d_state = head_dim = d ⇒ state = d²` declaration is faithful to the reference convention** (it is the `P = N = d` special case of `P × N`), and the reference **`chunk_size = 256`** default the arm deviates from (rig chunk 16) is confirmed from `mamba2.py`.
- **Verdict: VERIFIED** (double-sourced against official code in two files + the authors' own companion text).

### 2.3 What is **not** claimed here
- I did **not** verify the arm's *physics* (that chunking is an exact re-association, the `A ~ U(1,16)` init, the block ablation) — out of scope; those are the engineer's measurements.
- I did **not** re-verify Gated DeltaNet's *"Mamba2 as `S_t = α_t S_{t−1} + v_t k_tᵀ`"* presentation quoted in `bprime-mamba2-arm` §4.1: **GDN's citation facts re-confirm** (Yang, Kautz & Hatamizadeh, *"Gated Delta Networks: Improving Mamba2 with Delta Rule"*, arXiv:2412.06464, comments field verbatim **"ICLR 2025 camera ready"** — corroborates `bprime-cite-check` §4d), but **arXiv HTML is 404 for both v1 and v2**, so the equation itself stays **single-sourced to `rival-recon`**. ⛔ If r4 prints that equation as a quotation, it needs one more pass; if it paraphrases ("GDN describes Mamba2 as a gated outer-product state update"), it is safe on the abstract alone (*"gating enables rapid memory erasure while the delta rule facilitates targeted updates"* + the title's *"Improving Mamba2 with Delta Rule"*).

### 2.4 Tooling failures encountered (honest record)
| attempted source | result |
|---|---|
| `arxiv.org/html/2405.21060v1` | **404** — no arXiv HTML build for this paper |
| `ar5iv.labs.arxiv.org/html/2405.21060` | **"Fatal error" conversion page** — no content |
| `proceedings.mlr.press/v235/dao24a/dao24a.pdf` | **404** (that asset path is wrong; the abstract page resolved fine) |
| `arxiv.org/pdf/2405.21060v1`, `aclanthology.org/S18-2023.pdf` | fetched as **binary PDF**, not parseable by the fetch tool |
| `Read` on the saved PDF | **`pdftoppm is not installed`** (no poppler on this machine) ⇒ **PDF-only sources are unreadable to this agent** |
| `api.semanticscholar.org/graph/v1/snippet/search` | **HTTP 429** twice (rate-limited) |
| `scispace.com` | **403** |
| `arxiv.org/html/2412.06464v1` and `v2` (GDN) | **404** both |
⇒ Consequence: **the Mamba-2 paper's own body text was never read this session.** The state convention rests on official code (2 files) + the authors' blog. I consider that sufficient for the specific claim (it is an implementation convention, and code is the primary artifact for it), but it is *not* "verified against the paper."

---

## Ready-to-paste BibTeX

```bibtex
@inproceedings{poliak-etal-2018-hypothesis,
    title     = "Hypothesis Only Baselines in Natural Language Inference",
    author    = "Poliak, Adam  and
                 Naradowsky, Jason  and
                 Haldar, Aparajita  and
                 Rudinger, Rachel  and
                 Van Durme, Benjamin",
    editor    = "Nissim, Malvina  and
                 Berant, Jonathan  and
                 Lenci, Alessandro",
    booktitle = "Proceedings of the Seventh Joint Conference on Lexical and
                 Computational Semantics",
    month     = jun,
    year      = "2018",
    address   = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/S18-2023/",
    doi       = "10.18653/v1/S18-2023",
    pages     = "180--191",
    note      = "*SEM 2018. arXiv:1805.01042. Verified 2026-08-01 (ACL Anthology .bib +
                 Semantic Scholar on DOI + Edinburgh/Bryn Mawr records).
                 The quotable sentence -- IDENTICAL in preprint and proceedings text
                 (IA Scholar full-text search returns both) -- is:
                 'Across six of the ten datasets, our hypothesis-only model significantly
                 outperforms the majority-baseline, even outperforming the best reported
                 results on one dataset, recast SPR.'
                 WARNING: Semantic Scholar mislabels the venue as SemEval; *SEM != SemEval."}

@inproceedings{dao2024mamba2,
  title     = {Transformers are {SSM}s: Generalized Models and Efficient Algorithms
               Through Structured State Space Duality},
  author    = {Dao, Tri and Gu, Albert},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning
               (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {10041--10071},
  year      = {2024},
  publisher = {PMLR},
  note      = {Mamba-2. arXiv:2405.21060 (v1, 31 May 2024). Reference implementation:
               github.com/state-spaces/mamba. Verified 2026-08-01 (PMLR proceedings page
               + arXiv). NOTE author order: Mamba-2 = Dao & Gu; Mamba-1 = Gu & Dao.}}

@inproceedings{belinkov2019adversarial,
  title     = {On Adversarial Removal of Hypothesis-only Bias in Natural Language
               Inference},
  author    = {Belinkov, Yonatan and Poliak, Adam and Shieber, Stuart M. and
               Van Durme, Benjamin and Rush, Alexander M.},
  booktitle = {Proceedings of the Eighth Joint Conference on Lexical and Computational
               Semantics (*SEM)},
  year      = {2019},
  note      = {arXiv:1907.04389. Cited here only as third-party corroboration of
               Poliak et al. 2018's setup (10 datasets; SNLI 34%->69%). Venue/pages
               NOT independently verified in this pass -- verify before printing.}}
```
⚠ The Belinkov entry is included **only** because I quoted it as corroboration; its own venue/pages were **not** verified this session. Do not print it without a check.

---

## How I verified (sources actually fetched, 2026-08-01)
| claim | source 1 | source 2 | grade |
|---|---|---|---|
| Poliak title/authors/venue/pages/DOI | `aclanthology.org/S18-2023/` + `.bib` | S2 Graph API on `DOI:10.18653/v1/S18-2023`; `research.ed.ac.uk`; `repository.brynmawr.edu/compsci_pubs/86/` | **quadruple** |
| Poliak arXiv id + "accepted at \*SEM 2018 as long paper" | `arxiv.org/abs/1805.01042` | S2 `externalIds.ArXiv` | **double** |
| **"Across six of the ten datasets…" verbatim** | `ar5iv.labs.arxiv.org/html/1805.01042` (arXiv LaTeX rendering) | **`scholar.archive.org` phrase search — 2 hits, one under the \*SEM proceedings record** | **double, incl. published text** |
| the count is consistent with Table 2 | ar5iv Table 2 (4 parity / 6 gains) | — | internal check |
| SNLI 34→69, "out of 10 NLI datasets" | Belinkov et al. 2019 via `ar5iv/1907.04389` | ar5iv Table 2 (34.28 / 69.00) | **double** |
| Mamba-2 title/authors/venue/volume/pages | `proceedings.mlr.press/v235/dao24a.html` | `arxiv.org/abs/2405.21060` (comments: ICML 2024) | **double** |
| SSD state = `d_head × d_state` per head | `raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba2.py` | `…/mamba_ssm/modules/ssd_minimal.py` (`->bchpn`) **+** `tridao.me/blog/2024/mamba2-part1-model/` (*"total state size P × N"*) | **triple (2 code files + authors' post)** |
| Mamba-2 defaults `d_state=128, headdim=64, d_conv=4, expand=2, ngroups=1, chunk_size=256` | `mamba2.py` constructor | blog (*"N=64 to N=256"*, *"P = 64 or 128"*) — consistent, not identical | **double** |
| Mamba-2 paper body text | ⛔ **UNREADABLE** (HTML 404, ar5iv fatal, PDF unparseable, no poppler) | — | **not verified against the paper** |
| GDN = ICLR 2025, Yang/Kautz/Hatamizadeh, arXiv:2412.06464 | `arxiv.org/abs/2412.06464` (comments: *"ICLR 2025 camera ready"*) | `bprime-cite-check` §4d (proceedings.iclr.cc, NVlabs repo) | **double, across waves** |
| GDN's `S_t = α_t S_{t−1} + v_t k_tᵀ` presentation of Mamba2 | `rival-recon` §1.4 only | ⛔ arXiv HTML 404 (v1 and v2) | **single-sourced — still** |

## Confidence & gaps
- **High confidence:** every citation fact in both items, and the "6 of 10" figure. Poliak is the stronger of the two closures — the phrase is confirmed *in the published proceedings text*, which is exactly the check the Feng et al. version-mismatch made necessary.
- **Residual, minor:** Poliak's Table 2 decimals are ar5iv-sourced only (the draft doesn't use them); the "six" datasets' identity (SPR, FN+, SciTail, JOCI, SNLI, MNLI) is my reconstruction from that table, not a quoted list from the paper.
- **Residual, flagged:** the Mamba-2 paper's own body text is unread (tooling); the state convention is verified from official code + authors' post instead. If a referee-proof "the paper says P×N" is wanted, it needs a machine with PDF text extraction.
- **Still single-sourced elsewhere (not my scope, named for the ledger):** GDN's Mamba2 equation (`rival-recon` §1.4); MUNKEY's workshop identity + presentation type (OpenReview, now **5 consecutive waves** blocked); the remaining §5 cites `bprime-cite-check` §"Not verified" listed (Wang/Shi/Fox 2501.12352, ATLAS 2505.23735, Miras 2504.13173, HOLA 2607.02303, Based/MAD/Zoology/RULER, Mitzenmacher, Kipf) — **Poliak was the last of that list that the draft quotes a number from**, so the exposure is now qualitative-only.
- **Next search if the Hub wants it (cheap):** (a) GDN Eq. for Mamba2 from the ICLR 2025 OpenReview PDF or the FLA repo; (b) Mitzenmacher 2018 + Kipf et al. 2019 exact venues (draft §1 l.68 cites both by year, unverified); (c) whether arXiv:2405.21060 has picked up a v2 since (checked: v1 only as of today).

## Proposed handover updates (for the Hub)
- `bprime-cite-check-2` **done — both items VERIFIED**. **Poliak et al.** = \*SEM 2018, **pp. 180–191**, **DOI 10.18653/v1/S18-2023**, anthology **S18-2023**; the **"6 of 10" figure is correct** and the sentence *"Across six of the ten datasets, our hypothesis-only model significantly outperforms the majority-baseline"* is **identical in preprint and published text** (IA Scholar returns it under both records) — so §5.3 may quote it directly. Recommended wording: *"significantly outperform"*, not *"beat"*. **draft-r3 §12 item 6 → DISCHARGED**; `draft-r3` reconciliation 5 → **CLOSED**.
- **Mamba-2 citation facts CONFIRMED and upgradeable:** **Dao & Gu, ICML 2024, PMLR 235:10041–10071, arXiv:2405.21060** — `bprime-mamba2-arm` **reconciliation 4 → CLOSED**. The **SSD state convention `d_head × d_state` per head is triple-sourced** (`mamba2.py`, `ssd_minimal.py`, the authors' companion post); `rival-recon` §1.4's per-layer formula re-checked arithmetically and holds (**264·d_model + 1024** at defaults).
- ⚠ **New never-copy item for the citation ledger:** **Semantic Scholar labels Poliak et al. 2018 as "International Workshop on Semantic Evaluation" (SemEval)**. It is **\*SEM**, a different venue. Any BibTeX auto-imported from S2 for this entry is wrong.
- ⚠ **Author-order trap for the r4 bibliography:** Mamba-2 = **Dao & Gu**; Mamba-1 = **Gu & Dao**.
- ⛔ **Tooling, escalate:** this machine has **no PDF text extraction** (`Read` on a PDF → `pdftoppm is not installed`; the fetch tool cannot parse PDF binaries). Any citation whose evidence exists only as a PDF is **structurally unverifiable** by a scout here — that blocked the Mamba-2 paper body and the ACL PDF this session. Installing poppler (`brew install poppler`) removes a recurring class of "UNVERIFIABLE" verdicts. **Counterweight found:** `scholar.archive.org` full-text phrase search *does* reach published proceedings text and closed the Poliak double-source — add it to the standard scout kit. OpenReview remains bot-blocked (**5th wave**).
