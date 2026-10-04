# VENUE.md — ICLR 2027 main track: what the call requires
Owner: `w17/venue-scout`. Pages fetched 2026-09-22T21:49Z. Snapshots (HTML + extracted text + the official zip) are in
`~/clu/outputs/w17-venue-scout/sources/`.

**Cycle status:** the **ICLR 2027** call is **posted**. Every item below comes from the 2027 pages, not an older cycle.
- CFP: https://iclr.cc/Conferences/2027/CallForPapers
- Author Guide: https://iclr.cc/Conferences/2027/AuthorGuidelines
- AI Policy for Authors: https://iclr.cc/Conferences/2027/AIPolicyForAuthors
- Style files: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

---

## ⚠⚠ 0. Deadline flag — read first
| event | as stated on the page | source |
|---|---|---|
| **Abstract deadline** | "Sep 18, 2026 AOE" | CFP; Author Guide |
| **Full paper deadline** | "Sep 25, 2026 11:59 PM AOE" | CFP; Author Guide |
| Supplementary deadline | "The deadline is the same for the full paper and for the supplementary materials." | Author Guide FAQ |
| Reviews released | "Nov 05" | Author Guide |
| Public discussion, when revisions are allowed | "Nov 05 – Nov 18" | Author Guide |
| Decision | "Dec 16" | Author Guide |
| Camera-ready instructions | "mid-February 2027" | Author Guide |

**Today is 2026-09-22, so the abstract deadline has passed.** The Author Guide says: *"Please keep in mind that the
deadlines are final and we cannot make any accommodations for missing the abstract deadline or paper deadline."* and
*"Abstracts submitted by the abstract submission deadline must be genuine; placeholder or duplicate abstracts will be
removed."* This spoke does not know whether an abstract was registered on OpenReview by 18 Sep. **If none was, this
paper cannot go to the ICLR 2027 main track.** The next realistic option would then be the ICLR 2028 cycle, or another
venue. That decision belongs to the Head, and this spoke did not act on it. See §9.

---

## 1. Page limit
- **Initial submission: 9 pages of main text.** "At the time of submission, the main text should be 9 pages or fewer."
- **Rebuttal and camera-ready: 10 pages.** "During the discussion/rebuttal phase and for the camera ready, the page limit
  will be increased to 10 pages to allow for new results/discussions."
- **Enforcement:** "This limit will be strictly enforced. Papers with main text beyond the page limit will be desk-rejected."
- **Not counted toward the limit:**
  - References: "The list of references does not count towards the page limit, and unlimited additional pages are allowed for the bibliography/references."
  - AI use statement: "This section does not count toward the page limit."
  - Ethics statement: "will not count toward the page limit, but should not be more than 1 page."
  - Reproducibility statement: "is not part of the main text and therefore will not count toward the page limit."
  - Acknowledgements: the FAQ "Is the acknowledgement section included in the page limit?" is answered "No."
- The template's own text agrees: "strict upper limit of **9 pages** for the main text of the initial submission ...
  expanded to **10 pages** for rebuttal/camera ready" (`iclr2027_conference.tex` l.131).

**Working bound vs the venue limit:** the hub works to **≤ 12 pp main text** (COMMON.md). The venue limit is
**9 pp at submission**. As the task file requires, this is reported side by side only. **No spoke is asked to trim.**
Trimming is the Head's later editing pass.

## 2. Appendix and supplementary material
- "Authors may use as many pages of appendices (after the bibliography) as they wish, but reviewers are not required to read the appendix."
- "We encourage authors to submit a single file (paper + supplementary text). Supplementary text should appear after the references. Please mark the supplementary material clearly (e.g. as an appendix)."
- "Either is allowed: you can include the appendices at the end of the main pdf after the references, or you can include it as a separate file for the supplementary materials."
- Code: "Source code associated with a paper can be uploaded as part of the supplementary material ... We encourage all authors to submit code". Anonymous code options from the FAQ: an anonymised .zip as supplementary material; an anonymous repository linked in the paper; or an anonymous link posted to reviewers/ACs after the forums open. Any linked host must "not track visitors".
- This fits the hub's appendix-maximalism: appendix length is unlimited, and the appendix goes **after the bibliography** in the same PDF.

## 3. Template (on disk; draft-v2 compiles against this)
- **Location:** `~/clu/outputs/w17-paper/iclr2027/`
- **Source:** the official ICLR 2027 zip `iclr-2027-style-files.zip`, re-fetched 2026-09-22. **Every file on disk is
  byte-identical to the official zip** (SHA-256 checked). It was **not** re-downloaded into the shared dir; the verified
  copy of the zip is in `w17-venue-scout/sources/style.zip`.

| file | SHA-256 (prefix) |
|---|---|
| `iclr2027_conference.sty` (the style file) | `797deef41724e937` |
| `iclr2027_conference.bst` (bib style) | `2d67552db7ed38cc` |
| `iclr2027_conference.tex` (sample / instructions) | `03e556d6e5593e49` |
| `iclr2027_conference.bib` | `cdd86e7d4c31854d` |
| `math_commands.tex` | `90473c4d0542070d` |
| `natbib.sty` | `88bc70c0e4846193` |
| `fancyhdr.sty` | `b56ec4434b9f4607` |

- **Version:** the `.sty` has no version string. Its header still carries the legacy "Rev ... October 2014" line. It
  identifies itself by its running header, "Under review as a conference paper at ICLR 2027". Treat the version as
  "ICLR 2027 style files, zip as served on 2026-09-22", pinned by the hashes above.
- **Preamble to use:**
  ```latex
  \documentclass{article}
  \usepackage{iclr2027_conference,times}
  \input{math_commands.tex}   % optional
  \usepackage{hyperref}
  \usepackage{url}
  % keep \iclrfinalcopy COMMENTED OUT -> "Anonymous authors / Paper under double-blind review"
  ...
  \bibliography{references}            % the hub's bib (cite-tagger owns references.bib)
  \bibliographystyle{iclr2027_conference}
  ```
  Either put the draft in the same folder as the template files or add that folder to `TEXINPUTS`/`BSTINPUTS`.
- **Compile check (done, batmac, TeX Live 2026 pdfTeX 1.40.29):** `pdflatex → bibtex → pdflatex ×2` on a scratch copy
  of the sample `.tex` gave exit code 0 and "Output written on iclr2027_conference.pdf (7 pages)", with no errors. The
  shared dir was not touched by the build.
- **Formatting rules from the template:** "Do not change any aspects of the formatting parameters in the style files";
  US Letter; pages numbered; "The abstract must be limited to one paragraph."

## 4. Double-blind and self-citation
- "ICLR 2027 is double blind, which means that all submitted papers should be anonymous. Any paper where author identity
  is revealed in either the main text or the supplementary material will be desk rejected."
- "Note that related arxiv papers by the same authors do not break anonymity; if cited, these should be cited in third person."
- FAQ, citing your own prior paper A: "Yes, authors should provide the citations to all related work, including paper A.
  When referring to paper A, please use a 3rd person narrative."
- FAQ, nearly identical arXiv version: "No [it does not violate anonymity], so long as you do not refer to it explicitly."
- Template: author block hidden while `\iclrfinalcopy` stays commented out. "Non-anonymous submissions will be rejected without review."
- **Consequences for this paper** (they match COMMON.md): no names, institution, repo URL or acknowledgements; code only
  through an anonymised zip or a non-tracking anonymous link; self-citations in the third person.
- Transparency note: all submissions, including rejected and withdrawn ones, "will be deanonymized after the notification.
  The submissions and reviews will be released to the public." Withdrawn papers "will be de-anonymized immediately" once
  the deadline has passed.

## 5. Reproducibility statement
- **Status:** recommended, not required ("strongly encouraged").
- **Placement:** "a paragraph-long Reproducibility Statement at the end of the main text (before references)".
- **Page limit:** "not part of the main text and therefore will not count toward the page limit."
- **Content rule:** "should not itself describe details needed for reproducing the results, but rather reference the
  parts of the main paper, appendix, and supplemental materials that will help with reproducibility."
- **Hub choice:** include it (closest to an ICLR main-track result).

## 6. Ethics statement
- **Status:** recommended, and optional when there are no ethics issues. FAQ: "there is no need to include an Ethics Statement for all submissions."
- **Placement:** "(at the end of the main text before references)".
- **Page limit:** "will not count toward the page limit, but should not be more than 1 page."
- All authors must read the ICLR Code of Ethics (https://iclr.cc/public/CodeOfEthics), follow it, and acknowledge it at submission.
- **Hub choice:** a short statement is optional. There are no human subjects and the data are synthetic
  Shapes/maze-style environments, so none is strictly needed.

## 7. LLM / AI-usage policy — ⭐ a REQUIRED section
Author Guide heading: **"Required: AI use statement"**, "(See the AI Policy for Authors for more details.) This section does not count toward the page limit."

AI Policy for Authors, quoted verbatim:
> "LLMs are increasingly finding use in various research workflows, varying in sophistication from formulating proofs to
> formatting tables. We ask that authors explicitly state how they used LLMs in their submission, both in the paper's text
> as well as in the paper submission form."
>
> "The paper submission form will solicit some of the above information. Additionally, we require authors to disclose AI
> use in a mandatory section in their paper (this section will not be counted towards the page limit). The ICLR paper
> template includes a boilerplate disclosure to give authors a sense of the expected detail. You are not required to
> follow this format exactly"
>
> "Ultimately the paper's authors are responsible for the contents of their submissions. Consequently, a substantial
> falsehood, instance of plagiarism, or misrepresentation produced by an LLM would be considered a Code of Ethics
> violation on the part of the paper's authors, and might lead to desk rejection of the paper."

- **Required disclosure** (verbatim list): "Generate synthetic data sets, help develop theoretical models or conceptual
  frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, assist in the
  writing of proofs, propose or refine hypotheses, design or provide feedback on research methodology or experiments,
  implement methods, assist with translation, clean and reformat dataset, support qualitative and thematic data analysis,
  interpret results."
- **Recommended disclosure** (verbatim list): "Formulate questions for surveys or interviews, create or modify scientific
  figures or images, suggest experimental parameters, create or edit software code, creation of artifacts, draft parts of
  a research paper, transcribe recordings of research material, summarize or analyse existing literature, discover
  research topics or identify gaps, brainstorming, sourcing/searching for information, edit a research paper to improve
  readability, identify relevant literature, format references, suggest a structure for a research paper, propose a
  title or keywords for a research paper."
- **Template boilerplate:** `\subsection*{AI use statement}` at the end of the main text, before references ("should not
  be more than 1 page"). The template places the AI use, Ethics and Reproducibility statements in that order.
- The CFP also says: "A typical ICLR paper involves significantly more work than goes into a course project, or than can
  be produced autonomously by any current AI agent. ... If you are not sure your submission represents an appropriate
  amount of work for ICLR—or if you're not completely certain that everything in your paper is correct—please keep
  working rather than submitting now."
- **This matters here.** This hub's drafting, evidence pipeline and code are largely agent-produced. Several categories
  on the **required** list therefore apply: implement methods, design or give feedback on experiments, interpret results,
  and likely propose or refine hypotheses. The recommended "draft parts of a research paper" also applies. The draft
  must carry an honest AI use statement. **The draft needs `\TODO{AI use statement — Head to confirm which required
  categories apply}`.** This spoke does not fill it in, because the facts belong to the authors.

## 8. Citation style (cite-tagger keys off this)
- **natbib, author–year, round brackets.** The `.sty` does `\RequirePackage{natbib}` and
  `\setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}`.
- Template text: "Citations within the text should be based on the `natbib` package and include the authors' last names
  and year (with the "et al." construct for more than two authors). When the authors or the publication are included in
  the sentence, the citation should not be in parenthesis using `\citet{}` ... Otherwise, the citation should be in
  parenthesis using `\citep{}`."
- **Not numeric.** Use `\citet{}` when the citation is part of the sentence and `\citep{}` for parenthetical citations.
  Use bib style `iclr2027_conference.bst`. References are listed alphabetically by author. Do not load natbib a second
  time with conflicting options.
- Citing papers under review at ICLR 2027: "OpenReview provides BibTeX entries that do not list the authors".

## 9. Other rules that can desk-reject (as stated in the Author Guide)
- **Reciprocal reviewing:** "all submissions must have at least one author who is registered to review at least 3
  papers". Authors on 3 or more papers must review at least 6. Eligibility is based on papers "accepted for publication
  by the abstract deadline".
- **Co-authorship caps:** no one may be a co-author on more than 20 papers. "Each author may appear as a co-author on at
  most one paper in which no author is an eligible reciprocal reviewer."
- Every author needs an OpenReview profile before the full-paper deadline.
- Authors are frozen at the abstract deadline: "no authors can be added or removed after the abstract submission deadline".
- **Title and abstract:** these may change before the full-paper deadline, but the paper must stay close to the
  registered abstract: "The revised version shouldn't read like a different paper compared to your original abstract
  submission. If so, the paper will be rejected."

## 10. Summary table for other spokes
| item | answer |
|---|---|
| Main-text limit | **9 pp** at submission; **10 pp** for rebuttal and camera-ready. Strict; desk-reject if exceeded |
| Hub working bound | ≤ 12 pp (reported only; **no trimming in these waves**) |
| Outside the limit | references, appendix (unlimited, after the bibliography), AI use statement, ethics statement, reproducibility statement, acknowledgements |
| Template | `iclr2027_conference.sty` + `.bst`, official 2027 zip, SHA-pinned, in `w17-paper/iclr2027/` |
| Anonymity | fully double-blind; self-citations in the third person; desk-reject if identity is revealed |
| AI use statement | **REQUIRED**, outside the page limit |
| Ethics statement | recommended, ≤ 1 page, outside the limit |
| Reproducibility statement | recommended, one paragraph, outside the limit |
| Citations | natbib author–year, `\citet` / `\citep` |
| Deadlines | abstract **Sep 18, 2026 AOE (PASSED as of 2026-09-22)**; paper **Sep 25, 2026 11:59 PM AOE** |
