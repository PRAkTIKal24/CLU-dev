# bprime-cite-check — web-scout report
Task + acceptance criterion: verify/complete the four gating citation items for `draft-r2.md` (GDN-2 BibTeX · Erase-then-Delta BibTeX · the Feng et al. quote · standing Titans/MUNKEY/SDM venue checks), delivering ready-to-paste BibTeX with a per-item VERIFIED / CORRECTED / UNVERIFIABLE verdict.
Status: **done** — 3 CORRECTED/VERIFIED, 1 (MUNKEY workshop name) **still QUARANTINED but materially narrowed; Hub ruling needed.**
**Dial declaration:** none — instrument/recon (citation verification). No performance number, no laundering control, nothing falsifiable in the dial sense.

## ⛔ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — owner needed, stated in the first 10 lines)
1. **r3 writer:** §1 cites GDN-2 with no author (*"Gated DeltaNet-2, 2026"*) → **"Hatamizadeh et al., 2026"**. Author list now complete (below).
2. **r3 writer:** the Feng sentence our internal record quotes verbatim is the **arXiv-v1 abstract**, which is **NOT** the published ACL wording. If r3 quotes, quote the **ACL P19-1554** text (given verbatim below). r2 currently *paraphrases* — that is safe and I recommend keeping it; **draft §12 item 5 can be discharged** either way.
3. **Hub ruling:** MUNKEY's **"(oral)"** descriptor is now contested as well as its workshop name (the one venue-own record says **poster**). Recommend extending the quarantine to the presentation type, or adopting the dual-venue reading. Not blocking r2 (MUNKEY is cited **0 times** in `draft-r2.md`).
4. **Hub/r3 (positioning, not citation):** **Erase-then-Delta (EDA, arXiv:2606.26560)** is cited **0 times** in `draft-r2.md`. Completing its BibTeX only pays if r3 names it; see §2 for why it probably should.
5. **Curator:** `bprime-fb1-recon.md` BibTeX block (`gdn2_2026`, `erasethendelta2026`) carries `note={author list not extracted}` — superseded by this file; and the ERRATA-Bprime E2 "approved citation form" carries "(oral)" (see item 3).

---

## Answer first
All three BibTeX/quote items are closed. **GDN-2 = Ali Hatamizadeh, Yejin Choi, Jan Kautz (NVIDIA), arXiv:2605.22791 v1, 21 May 2026, no venue** (double-sourced). **Erase-then-Delta = 18 authors led by Xiao Li (Qwen Team / Nanjing / Zhejiang), arXiv:2606.26560 v1, 25 Jun 2026, no venue** (double-sourced). **The Feng et al. converse caveat is REAL and now double-sourced in its published form — but our internal record quotes the arXiv wording, which differs sentence-for-sentence from the ACL camera-ready.** Standing checks: **Titans = NeurIPS 2025 main conference, verified twice**; **SDM = no Table-1 ratio anywhere in `draft-r2.md`, confirmed by grep**; **MUNKEY's workshop identity is still not resolvable from a single authoritative record — the disagreement has *changed shape*, not vanished.**

---

## 1. GDN-2 — **CORRECTED** (author list completed, title/year/venue as cited are right)

- **Authors, in order: Ali Hatamizadeh, Yejin Choi, Jan Kautz** (NVIDIA). Source 1: arXiv abs page. Source 2: HuggingFace papers page for the same ID. Corroborating: official code `github.com/NVlabs/GatedDeltaNet-2`.
- **Title verbatim:** *"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"* — matches the draft (§5 and App. table) exactly.
- **Date/version:** [v1] 21 May 2026 17:44:57 UTC. **No journal-ref, no venue** — it is a preprint/technical report. The draft correctly cites it bare as `arXiv:2605.22791` with no venue. ⚠ Do **not** upgrade it to a venue.
- Abstract opening (verbatim, arXiv): *"Linear attention replaces the unbounded cache of softmax attention with a fixed-size recurrent state, reducing sequence mixing to linear time and decoding to constant memory. The hard part is not just what to forget, but how to edit this compressed memory without scrambling existing associations."*
- Its own scope claim (as surfaced in indexing of the abstract): SOTA at 1.3 B / 100 B FineWeb-Edu tokens vs Mamba-2, Gated DeltaNet, KDA, Mamba-3 variants. ⚠ This is **the paper's claim**, not an established fact, and the draft's *"it supersedes GDN"* is **our** characterisation — defensible (same senior authors as GDN: Kautz, Hatamizadeh) but it is a positioning sentence, not a citation fact.
- ⭐ **Fix for r3:** §1 line 54 reads *"Gated DeltaNet-2, 2026"* with no author where its two siblings carry "Yang et al." → **"Gated DeltaNet-2, Hatamizadeh et al., 2026"**.
- **Verdict: CORRECTED.** Nothing the draft says about GDN-2 is wrong; the citation was incomplete, and now is not.

## 2. Erase-then-Delta — **CORRECTED** (and it is **not currently cited** in `draft-r2.md`)

- **Title verbatim:** *"Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention"* (arXiv abs + arXiv HTML v1 title page agree; the HuggingFace record truncates the title to the subtitle — do not take the HF form).
- **Authors, in order (18): Xiao Li, Chengruidong Zhang, Hao Luo, Xi Lin, Zekun Wang, Zihan Qiu, Yunfei Mao, Langshi Chen, Man Yuan, Minmin Sun, Huiqiang Jiang, Siqi Zhang, Rui Men, Wei Hu, Gong Cheng, Bo Zheng, Dayiheng Liu, Jingren Zhou.** Affiliations per the HTML title page: **Qwen Team, Nanjing University, Zhejiang University**. Double-sourced (arXiv abs listing + arXiv HTML v1; HF papers gives the identical 18-name list in the same order).
- **Date:** [v1] 25 Jun 2026 03:12:19 UTC, cs.CL. **No comments/journal-ref ⇒ no venue.**
- Mechanism, from its own text: *"first applies a targeted erase step along a learned erase direction, and then performs the standard delta-style corrective write"*, giving *"the missing degree of freedom needed to suppress stale memory at one address before performing a corrective write at another."* Reported at **2.5 B dense and 25B-A2.8B MoE**, long-context midtraining at 32k, extrapolation to 128k; claims stronger average than **KDA and GDN**; measured erase/write address separation **mean |cos| ≈ 0.105**.
- ⚠ **Finding the r3 writer needs:** `draft-r2.md` contains **zero** occurrences of "Erase-then-Delta", "EDA", "26560" or "KDA" (grep). So this BibTeX is currently unused. **Recommendation (positioning, Hub's call):** the draft's §5 sentence *"we use GDN-2 as the delta-rule reference arm because it supersedes GDN"* is a claim about who the frontier is — and **a second, independent, same-quarter paper (EDA, Qwen) makes the same erase/write-decoupling move and claims to beat GDN and KDA at larger scale.** One clause naming EDA (*"concurrently, …"*) costs nothing and pre-empts a referee who reads it as us picking a convenient frontier. It also strengthens §5's "the reference arm has moved" argument by making it a trend rather than one paper.
- **Verdict: CORRECTED** (entry completed; erratum against `bprime-fb1-recon`'s `note={author list not extracted}`).

## 3. Feng, Wallace & Boyd-Graber — **VERIFIED, with a version catch**

**The paper is real and peer-reviewed:** *Misleading Failures of Partial-input Baselines*, **ACL 2019**, Florence, Italy, **pages 5533–5538**, **DOI 10.18653/v1/P19-1554** (ACL Anthology page; Semantic Scholar API keyed on the DOI returns the identical record). The single-sourcing flag is discharged.

⚠ **But the two versions do not say it the same way, and our internal record quotes the arXiv one.**

- **arXiv:1905.05778 abstract (verbatim, arxiv.org/abs):** *"When a partial-input baseline gets high accuracy, a dataset is cheatable. However, the converse is not necessarily true: the failure of a partial-input baseline does not mean a dataset is free of artifacts."*
  — this is **exactly** the sentence `bprime-fb1-recon.md` §D4 quoted and flagged as *"quoted from a search-surfaced PDF extract — single-sourced, re-verify before printing."* **It is verbatim correct — of the arXiv version.**
- **ACL Anthology camera-ready abstract (verbatim; double-sourced: aclanthology.org/P19-1554/ and the Semantic Scholar Graph API on DOI:10.18653/v1/P19-1554, character-identical):**
  > *"A successful partial-input baseline indicates that the dataset is cheatable. But the converse is not necessarily true: failures of partial-input baselines do not mean the dataset is free of artifacts."*
  (Full published abstract also verified; it ends *"Our work provides a caveat for the use and creation of partial-input baselines for datasets."*)
- ⇒ **If r3 prints a quotation, print the ACL wording with page numbers.** Printing the arXiv wording while citing "ACL 2019, pp. 5533–5538" is the kind of mismatch a careful referee catches.

**Status in the draft as it stands.** `draft-r2.md` does **not** quote Feng verbatim anywhere (§2.4 l.225–228 and §6 L8 l.1007–1010 are our own paraphrase, correctly attributed). That paraphrase — *"a substitute audit that a memory passes does not show the memory is doing real work; only a failed one is informative"* — is a **faithful transfer** of the published converse caveat. ⚠ One scope note worth one clause: **Feng et al.'s claim is about datasets and annotation artifacts, not about memories.** The transfer to "a memory passing its own substitute audit" is *ours by analogy*, and §5.3/§6 L8 should say "by the same logic" rather than implying Feng et al. state it about models/memories. That is the only residual referee handle here.

**Verdict: VERIFIED** (existence, venue, pages, DOI, and both abstract wordings). **Draft §12 item 5 ("one citation is single-sourced") is dischargeable.**

## 4. Standing venue checks

### 4a. Titans — **VERIFIED, main conference. Never "a preprint."**
- **neurips.cc/virtual/2025/poster/119639** — *"Titans: Learning to Memorize at Test Time"*, Ali Behrouz, Peilin Zhong, Vahab Mirrokni, **NeurIPS 2025 Poster**.
- **proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html** — listed under **Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Main Conference Track** (the `-Abstract-Conference` route is the main-track route; workshop papers are not in this collection).
- ⇒ Two independent official records. The draft's §5 *"Titans (Behrouz, Zhong & Mirrokni, NeurIPS 2025)"* and App. J's *"peer-reviewed at NeurIPS 2025 (never 'a preprint')"* are **correct as written**. "Poster" is the presentation format of an accepted main-conference paper — **do not write "poster" as if it were a downgrade, and do not write "preprint"** (claims_matrix §0 never-quote 9 holds).
- arXiv:2501.00663 remains the correct arXiv pointer for the preprint version.

### 4b. MUNKEY — **QUARANTINE STANDS. The disagreement has NOT resolved; it has changed shape (and now touches "(oral)" too).**
New evidence this session:
- **arXiv:2603.15033** — v1 16 Mar 2026, v2 24 Mar 2026, **v3 2 Apr 2026**; **Comments field EMPTY, Journal-ref EMPTY** (re-checked today, unchanged since the C2W3 finding). The paper's own record still names no venue.
- ⭐ **NEW, and it is a venue-own record: the ICLR 2026 Workshop on Recursive Self-Improvement (RSI) accepted-papers page lists it as poster #41** — `recursive-workshop.github.io/papers.html`, sections Oral (4) / Spotlight (21) / **Poster (75, contains #41)** / Short (10). The entry links **`openreview.net/forum?id=gGH3Xp1lHR`** — **the exact forum ID our prior passes were bot-blocked on.** So the submission we could never resolve *is* an RSI submission, and RSI lists it as a **poster**.
- **The authors' own page (`sonialagunac.github.io`) states verbatim: _"In ICLR 2026 Workshop TTU (Oral)"_.** TTU is real and separately verified: **"3rd Workshop on Test-Time Updates (TTU): Putting Updates to the Test!", ICLR 2026** (`ttu-iclr2026.github.io`) — but **its accepted-papers page defers entirely to OpenReview**, which is **still bot-blocked (4th consecutive wave)**, so TTU's own record is unreadable.
- ⇒ **Two live readings, and I cannot separate them:** (i) **dual venue** — oral at TTU *and* poster at RSI (two workshop submissions of the same paper is common and would make both sources true), or (ii) one of the two pages is wrong. **Nothing I can reach adjudicates this**; only TTU's OpenReview list would.
- **Recommendation (Hub ruling required — I do not lift quarantines):** keep the workshop **name** quarantined, and **extend the quarantine to the presentation type**, because the only *venue-own* record says **poster** while "(oral)" rests on the authors' page. Safe form: *"MUNKEY (Laguna et al., arXiv:2603.15033), an ICLR-2026 workshop paper"* — no workshop name, no presentation type. If the Hub prefers maximal information, the defensible long form is: *"an ICLR-2026 workshop paper (listed as a poster by the ICLR 2026 RSI workshop; described by the authors as an oral at the ICLR 2026 TTU workshop)."*
- ⚠ **Not blocking r2:** `draft-r2.md` contains **zero** occurrences of "MUNKEY". This check is preventive (it protects `claims_matrix` §0 never-quote 13 and any pillar-4 sentence r3 might add).
- ⛔ **Tooling escalation, now four waves old:** OpenReview bot-blocks every fetch. MUNKEY's venue is one of ≥3 deliverables stalled behind it. It needs a human/authenticated fetch or it stays permanently unresolvable.

### 4c. SDM — **VERIFIED: the draft quotes NO Table-1 ratio.**
- Grep over `draft-r2.md` for `156`, `168`, `111 %`, `Table 1`, `state/parameter`, `Sparse Delta Memory`: **every SDM mention is qualitative.** §5 says only *"Sparse Delta Memory reports isoFLOP and isoParameter with a state-to-parameter ratio column"* (l.855–856) and §5.4 lists *"a state/parameter ratio column (SDM)"* (l.913). **No numeric SDM ratio appears anywhere.**
- **Appendix J states the quarantine explicitly:** *"⛔ Its Table 1 state/parameter ratios are quarantined — two independent extractions disagree — and none is quoted anywhere in this paper."* (l.1565–1566). ✅ Self-consistent with `claims_matrix` §0 never-quote 11.
- ⚠ Note: the **0.967** state/parameter ratio at l.89 / l.212 / l.1326 is **ours** (our own store's measured deviation), not SDM's — no collision, but a reader skimming could conflate them; one clause (*"our own convention, not SDM's column"*) at l.89 would remove the ambiguity. Optional.
- Bonus re-verification (the draft author-year-cites it): **arXiv:2607.07386 v1, 8 Jul 2026, cs.LG, CC BY 4.0, no venue** — *"Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity"*, **Loïc Cabannes, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Matthijs Douze, Maria Lomeli, Ilze Amanda Auzina, Justin Carpentier, Gabriel Synnaeve, Hervé Jégou** — matches `track2-admissibility`'s 9-name list exactly. §1's *"(Cabannes et al., 2026)"* is correct.

### 4d. Free-riding checks on the two sibling delta-rule cites (not asked; both ✅)
- **DeltaNet** — *"Parallelizing Linear Transformers with the Delta Rule over Sequence Length"*, Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim, **NeurIPS 2024** (proceedings.neurips.cc 2024 `-Abstract-Conference`; neurips.cc/virtual/2024/poster/93040). Draft's *"Yang et al., NeurIPS 2024"* ✅.
- **Gated DeltaNet** — *"Gated Delta Networks: Improving Mamba2 with Delta Rule"*, Songlin Yang, Jan Kautz, Ali Hatamizadeh, **ICLR 2025** (proceedings.iclr.cc; iclr.cc/virtual/2025/poster/28219; NVlabs repo tagged "[ICLR 2025]"; arXiv:2412.06464). Draft's *"Yang et al., ICLR 2025"* ✅.

---

## Ready-to-paste BibTeX (replaces the incomplete entries in `bprime-fb1-recon.md`)

```bibtex
@article{hatamizadeh2026gdn2,
  title   = {Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention},
  author  = {Hatamizadeh, Ali and Choi, Yejin and Kautz, Jan},
  journal = {arXiv preprint arXiv:2605.22791},
  year    = {2026},
  note    = {v1, 21 May 2026; NVIDIA; no venue (technical report);
             code at https://github.com/NVlabs/GatedDeltaNet-2;
             author list verified 2026-08-01 (arXiv abs + HuggingFace papers)}}

@article{li2026erasethendelta,
  title   = {Erase-then-Delta Attention: Decoupling Erase and Write Addresses in
             Delta-Rule Linear Attention},
  author  = {Li, Xiao and Zhang, Chengruidong and Luo, Hao and Lin, Xi and Wang, Zekun and
             Qiu, Zihan and Mao, Yunfei and Chen, Langshi and Yuan, Man and Sun, Minmin and
             Jiang, Huiqiang and Zhang, Siqi and Men, Rui and Hu, Wei and Cheng, Gong and
             Zheng, Bo and Liu, Dayiheng and Zhou, Jingren},
  journal = {arXiv preprint arXiv:2606.26560},
  year    = {2026},
  note    = {v1, 25 Jun 2026; cs.CL; Qwen Team / Nanjing University / Zhejiang University;
             no venue; 18 authors verified 2026-08-01 (arXiv abs + arXiv HTML v1)}}

@inproceedings{feng2019misleading,
  title     = {Misleading Failures of Partial-input Baselines},
  author    = {Feng, Shi and Wallace, Eric and Boyd-Graber, Jordan},
  editor    = {Korhonen, Anna and Traum, David and M{\`a}rquez, Llu{\'\i}s},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for
               Computational Linguistics},
  month     = jul,
  year      = {2019},
  address   = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
  pages     = {5533--5538},
  doi       = {10.18653/v1/P19-1554},
  url       = {https://aclanthology.org/P19-1554/},
  note      = {arXiv:1905.05778. WARNING: the arXiv abstract and the ACL camera-ready
               state the converse caveat in DIFFERENT words -- quote the ACL text}}

@inproceedings{behrouz2025titans,
  title     = {Titans: Learning to Memorize at Test Time},
  author    = {Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS 2025)},
  year      = {2025},
  note      = {Main Conference Track (peer-reviewed; NEVER cite as a preprint).
               arXiv:2501.00663. No official code as of 2026-08-01}}

@article{laguna2026munkey,
  title   = {Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion},
  author  = {Laguna, Sonia and da Silva Gon{\c c}alves, Jorge and Vandenhirtz, Moritz and
             Ryser, Alain and Cannistraci, Irene and Vogt, Julia E.},
  journal = {arXiv preprint arXiv:2603.15033},
  year    = {2026},
  note    = {v3, 2 Apr 2026; arXiv comments AND journal-ref fields EMPTY (re-checked
             2026-08-01). An ICLR-2026 WORKSHOP paper -- workshop identity QUARANTINED:
             the ICLR 2026 RSI workshop's own accepted-papers page lists it as poster #41
             (OpenReview forum gGH3Xp1lHR); the authors' page states "ICLR 2026 Workshop
             TTU (Oral)". Presentation type ALSO quarantined. NOT ICML 2026.
             v3 self-describes as "a memory augmented transformer"}}

@article{cabannes2026sdm,
  title   = {Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity},
  author  = {Cabannes, Lo{\"i}c and Mazar{\'e}, Pierre-Emmanuel and Szilvasy, Gergely and
             Douze, Matthijs and Lomeli, Maria and Auzina, Ilze Amanda and
             Carpentier, Justin and Synnaeve, Gabriel and J{\'e}gou, Herv{\'e}},
  journal = {arXiv preprint arXiv:2607.07386},
  year    = {2026},
  note    = {v1, 8 Jul 2026; Meta FAIR; CC BY 4.0; no venue; official code
             github.com/facebookresearch/sparse-delta-memory (SM 80+ required).
             NEVER quote its Table-1 state/parameter ratios (extractions conflict)}}

@inproceedings{yang2024deltanet,
  title     = {Parallelizing Linear Transformers with the Delta Rule over Sequence Length},
  author    = {Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and Kim, Yoon},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024)},
  year      = {2024}}

@inproceedings{yang2025gdn,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2412.06464; code github.com/NVlabs/GatedDeltaNet}}
```

---

## How I verified (sources actually fetched, 2026-08-01)
| claim | source 1 | source 2 | grade |
|---|---|---|---|
| GDN-2 authors/title/date | `arxiv.org/abs/2605.22791` | `huggingface.co/papers/2605.22791` | **double-sourced** (+ NVlabs repo) |
| GDN-2 has no venue | arXiv abs (no journal-ref) | HF (no venue field) | double-sourced (negative evidence) |
| EDA authors (18) / title | `arxiv.org/abs/2606.26560` | `arxiv.org/html/2606.26560v1` (title page + affiliations) | **double-sourced**; HF corroborates the name list |
| Feng ACL wording | `aclanthology.org/P19-1554/` | Semantic Scholar Graph API on `DOI:10.18653/v1/P19-1554` (character-identical) | **double-sourced** |
| Feng arXiv wording differs | `arxiv.org/abs/1905.05778` | — | single-sourced (primary; it *is* the record) |
| Titans = NeurIPS 2025 main track | `neurips.cc/virtual/2025/poster/119639` | `proceedings.neurips.cc/.../2025/hash/a4ca07aa...-Abstract-Conference.html` | **double-sourced** |
| MUNKEY arXiv comments EMPTY at v3 | `arxiv.org/abs/2603.15033` | (matches C2W3's finding) | double-sourced across waves |
| MUNKEY = RSI poster #41 | `recursive-workshop.github.io/papers.html` (venue-own) | — | **single-sourced** |
| MUNKEY = TTU oral | `sonialagunac.github.io` (authors' own) | TTU workshop exists: `ttu-iclr2026.github.io` | **single-sourced on the claim** |
| MUNKEY venue from OpenReview | ⛔ **BLOCKED** (bot-check, 4th wave) | — | **UNVERIFIABLE by this tool** |
| SDM: no Table-1 ratio in the draft | grep over `draft-r2.md` | App. J's own quarantine sentence | verified in-repo |
| SDM authors | `arxiv.org/abs/2607.07386` | `track2-admissibility.md` §3 list (identical) | double-sourced |
| DeltaNet NeurIPS 2024 / GDN ICLR 2025 | proceedings.neurips.cc / proceedings.iclr.cc | neurips.cc + iclr.cc virtual pages, NVlabs repo tag | double-sourced |

## Findings the r3 writer should act on (compressed)
1. §1 l.54: add "Hatamizadeh et al." to the GDN-2 cite.
2. §12 item 5: **discharge** — Feng is verified (ACL 2019, pp. 5533–5538, DOI 10.18653/v1/P19-1554). If a quotation is added, use the **ACL** wording verbatim (above), not the arXiv wording.
3. §2.4 / §6 L8: keep the paraphrase; add "by the same logic" — Feng's caveat is stated about *datasets*, and the transfer to *memories* is ours.
4. §5 (optional, recommended): one clause naming **EDA (Li et al., arXiv:2606.26560)** as concurrent to GDN-2 — it makes "the delta-rule reference arm has moved" a trend, not a pick.
5. §5 l.89 (optional): mark **0.967** as our own convention so it cannot be read as SDM's column.
6. MUNKEY: if r3 introduces it, use the no-name, no-presentation-type form until the Hub rules.

## Confidence & gaps
- **High confidence:** GDN-2 and EDA BibTeX; Feng's venue/pages/DOI and both abstract wordings; Titans main-track; SDM's absence from the draft's numbers.
- **Unresolved:** MUNKEY's workshop **name and presentation type**. The RSI listing is the strongest single record we have (it is the workshop's own page and it carries the OpenReview ID), but the authors assert TTU/oral, and both can be true. **Only an authenticated OpenReview fetch closes this.**
- **Not searched:** whether GDN-2 or EDA have since been accepted anywhere (checked their arXiv records only; both empty as of today, and neither has a v2). Worth a re-check at camera-ready.
- **Not verified:** the draft's other §5 citations (Wang/Shi/Fox 2501.12352, ATLAS 2505.23735, Miras 2504.13173, HOLA 2607.02303, Based/MAD/Zoology/RULER, Mitzenmacher, Kipf, Poliak). Poliak et al. (*SEM 2018) in particular is the *other* half of the conceded-ancestry sentence and was **not** in my scope — it is currently registry-sourced only. **Suggest a cheap follow-up: verify Poliak et al. \*SEM 2018 (anthology ID + pages + the "6 of 10" number the draft quotes at l.896).** That number is quoted in the draft and I did not check it.

## Proposed handover updates (for the Hub)
- `bprime-cite-check` **done**: GDN-2 = **Hatamizadeh, Choi & Kautz** (arXiv:2605.22791, no venue); Erase-then-Delta = **Li et al., 18 authors, Qwen Team** (arXiv:2606.26560, no venue); **Feng et al. VERIFIED** (ACL 2019, pp. 5533–5538, DOI 10.18653/v1/P19-1554) — ⚠ **the arXiv and ACL wordings of the converse caveat DIFFER; our internal record quotes the arXiv one**; **Titans = NeurIPS 2025 main track, double-sourced**; **SDM: confirmed the draft quotes no Table-1 ratio**.
- ⛔ **MUNKEY quarantine STANDS and now covers "(oral)" as well**: the ICLR-2026 **RSI** workshop's own page lists it as **poster #41** (with the `gGH3Xp1lHR` forum ID); the authors' page says **TTU (oral)**. Dual-venue is a live and innocent explanation. **OpenReview blocked for the 4th consecutive wave — escalate as a tooling item.**
- ⚠ **New never-quote candidate:** *"MUNKEY, an ICLR-2026 workshop paper (oral)"* — the presentation type is contested; the approved form in `ERRATA-Bprime.md` §E2 should be re-issued without "(oral)".
- ⭐ **New positioning input for r3/tier-i framing:** **EDA (Qwen, arXiv:2606.26560)** independently decouples erase from write addresses at 2.5 B / 25B-A2.8B with 128k long-context results — a second frontier claimant beside GDN-2. Neither is cited-with-authors in r2 today.
- **Suggested next scout task (small):** verify **Poliak et al., \*SEM 2018** (anthology ID, pages, and the draft's quoted "6 of 10 NLI datasets") — it is the other half of §5.3's conceded ancestry and is currently registry-sourced only.
