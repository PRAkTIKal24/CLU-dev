# v1-cite-and-template — paper-writer report

Task + acceptance criterion: convert every manual author-year string in `pj_sub.tex` to natbib `\cite*`, replace the hand-built reference list with `\bibliography{refs}` (attaching the five orphans, keeping the three cut works out), then port to the NeurIPS 2026 `dblblindworkshop` class — with `OTHER` diff hunks = 0 and prose byte-identical around every conversion.
Status: **done** (Part A and Part B both complete; nothing struck).

⚠ **Reconciliation list with an owner needed (protocol §5 corollary, in the first 10 lines):** **one item** — the `refs.bib` `ć` defect in §"Findings" item 1 (`Pavlovi{\'c}`, one character). It is out of this pass's write scope, it silently misspells an author in the built bibliography, and it currently costs the paper the template's `[T1]{fontenc}` line. **The Hub should assign it to the `refs.bib` owner.** No other downstream site changes because of this pass.

## DIAL DECLARATION (echoed from task)
**Dials touched: NONE.** Instrument/mechanical pass: citations wired, document class changed. No experiment, no configuration change, no measured value altered, no number touched.

## Pin check (task item 1)
- `md5 pj_sub.tex` at start = **`6867e06b56d97aadc52398558e9e4797`**.
- ⭐ This **equals the post-edit md5 reported by the theorist pass** (`v1-derivation-appendix`) ⇒ the Head did **not** edit between the theorist landing and this launch. The pin is confirmed, not merely reported.
- `md5 pj_sub.tex` at end = **`de4559a36af659bada4a56ea05156db7`**.
- Mechanical precondition satisfied: `.claude/outputs/v1-derivation-appendix.md` exists (17,628 B, 21:34).

## What I did
1. Read protocol → Positioning Charter (`philosophy-synthesis.md` §581+, as it reads today, incl. the **C-1 reversal**) → claims matrix header → task file → the theorist report (to inherit its reconciliation list and avoid re-opening its items) → the whole paper → `refs.bib` → `neurips_2026.sty` → the shipped NeurIPS template.
2. Pinned the baseline to scratch; captured the protected-file manifest; built the before-state.
3. Applied **Part A** (13 `CITATION` + 1 `BIBLIOGRAPHY` hunks) and verified it **in isolation** (a scratch copy with bare `natbib`, still `article` class) before the class changed — so a citation failure could never be confused with a template failure.
4. Applied **Part B** (2 `TEMPLATE` hunks), found and fixed the bracket-delimiter defect, found the `ć` defect and measured both variants.
5. Re-applied the whole thing **from the pinned baseline in one consolidated script**, so the logged hunk set is *literally* the shipped diff (and is provable by reversion).
6. Wrote `BUILD-NOTE-R5.md` (deliverable 1) and this report.

## How I verified (commands + observed output)
- **Build:** `tectonic 0.15.0` (XeTeX). ⚠ **No `pdflatex`/`bibtex`/`latexmk` on this machine** — the port is verified under tectonic **only**; I did not pseudo-verify a pdflatex build.
- **Final build:** `^! ` errors **0** · `Citation .* undefined` **0** · `Reference .* undefined` **0** · `Missing character` **0** · bibitems **14**.
  - ⚠ `grep -ci error` on the log returns 1; read in context it is the package name **`infwarerr`** — a false friend, exactly the hazard the task warned about.
- **`OTHER` = 0, proved by reconstruction** (git can't diff; `.claude/**` is gitignored): reverting the 16 logged hunks from the shipped file reproduces the baseline md5 **exactly** (`6867e06b…` == `6867e06b…`). The diff therefore *is* the hunk set: **13 CITATION · 1 BIBLIOGRAPHY · 2 TEMPLATE · 0 OTHER**.
- **Word-level byte-identity:** all **13/13** CITATION hunks pass (delete the `\cite*` macro from AFTER, delete the manual author-year token it replaced from BEFORE, require identity). **0 drifted**; no sentence reflowed.
- **Residual sweep, positive-controlled** (normaliser covers the full natbib set incl. `\citeauthor`/`\citeyearpar`, so no false possessive hits): baseline **10** manual author-year hits → final **1**.
- **Protected files byte-untouched:** `submission.tex` `caef2272…` unchanged; `.claude/papers/v1-short/**` 11-file manifest before ≡ after (diff empty); `refs.bib` `58c75795…` and `neurips_2026.sty` `f447d330…` unchanged.
- **Pages:** before **17 pp** (main text pp. 1–9, App. A begins p. 9) → after **16 pp** (main text **pp. 1–8**, App. A begins p. 8, References p. 15).
- **Anonymity:** author block renders "Anonymous Author(s)"; PDF metadata has no Title/Author/Subject/Keywords and no metadata stream; main-text sweep (pp. 1–8) for `@`, acknowledgments, "our previous/prior work", `github.com`, `Pratik`, `Maurizio`, `CERN`, funding/grant = **0 each**.

## Findings/results

**1. ⛔⛔ `refs.bib` silently misspells an author in the built PDF — and it is why I did not add `[T1]{fontenc}`.**
Switching to a real bibliography made the full author lists print for the first time (the hand-built list said "Ramsauer, H., et al."). Under tectonic/XeTeX the class's Times (`\rmdefault=ptm`) **plus** `[T1]{fontenc}` gives `Missing character: There is no ć ("107) in font ptmr8t!` and the letter is **absent from the PDF** — "Milena Pavlović" prints as "Milena Pavlovi", with **no error**. Measured both ways: **with T1** → 1 missing glyph, "Pavlovi", 15 pp; **without T1** → **0 missing glyphs, "Pavlović" correct**, 16 pp. `Candès`, `Schäfl`, `Günter` are fine either way (the task named the first two; the failure is on a third name it did not). I therefore **omitted the template's `[T1]{fontenc}` line** — the single deviation from the template preamble — because the task's explicit instruction for this hunk was to *verify accented names still compile correctly*, and shipping a PDF with a misspelt author is worse than an invisible font-encoding difference. **Permanent fix is one character in `refs.bib`: `Pavlovi{\'c}`**, after which T1 can be restored (recovering a page and better hyphenation). ⛔ Not made here — `refs.bib` is outside my write scope and is Advisor-verified.

**2. ⚠ natbib's default is SQUARE brackets — caught and fixed.** The first ported build rendered `[Gladstone et al., 2025]`, `[Ramsauer et al., 2021]` and, worst, **"Jawahar and Pierini [2026]"** — the charter-mandated CHLU continuity sentence in brackets, next to manual round parentheses in the same sentence. Fixed with `\PassOptionsToPackage{round}{natbib}`, the mechanism documented in the shipped template's own preamble. **Style declared (A4): author-year, round parens, `plainnat`, natbib loaded by the class (`nonatbib` NOT used).**

**3. The conversion self-corrected two stale strings, exactly as A1 predicted — neither by hand.**
- **`Angelopoulos et al. 2021` → `Angelopoulos et al., 2025`**, and the entry now prints the real record (*Annals of Applied Statistics* 19(2), June 2025, doi 10.1214/24-AOAS1998).
- **`Jawahar, P., Pierini, M. (2026). [CHLU primitive].`** — a truncated placeholder title — now prints the full title + arXiv 2603.01768.
- `Graves` prints **2016** despite the key reading `_2017`; `Neal(2011)` prints from key `brooks_mcmc_2011`; `raposo_mixture--depths_2024`'s double hyphen and the `@misc`/`@article`/`@incollection`/`@inproceedings` split were left untouched. ⛔ No `refs.bib` byte changed.

**4. All five orphans attached and confirmed *printing*** (in `.bbl` **and** in the rendered References text): `duane_hybrid_1987` (App. F, the `min(1,e^{-ΔH/T})` detailed-balance sentence — the paper re-derives HMC without naming it), `brooks_mcmc_2011`=Neal (App. F ergodicity/momentum-refreshment), `roberts_exponential_1996` (**both** sites: §5 design rules at the MALA naming, and App. F's FDT noise scale), `geifman_selective_2017` (§4.1), `wales_global_1997` (§3.1 squeeze intro).
- ⚠ **Placement judgement to ratify — Geifman.** The task said "beside the existing Learn-then-Test citation". Putting it *inside* the `(LTT; …)` parenthetical would read as if Geifman were an LTT source. I attached it instead at the end of the **same sentence**, on the clause it actually supports — "…exit thresholds based on a dynamic relaxation ladder \citep{geifman_selective_2017}." — which is a reject-option/selective classifier. Zero prose bytes changed either way. Say the word and it moves.
- ⛔ Cut entries confirmed absent from the built bibliography (`.bbl` and rendered PDF): `Lieb` 0 · `Robinson` 0 · `Platt` 0 · `Anonymous` 0. The theory-note entry was **not** re-created; **no** "supplementary material" note added anywhere. The `Platt-calibrated` table cell survives as plain prose, uncited and undeleted, per ruling.

**5. ⚠ One manual author-year string survives, deliberately: `(Anonymous, 2026)` in §1.** It cannot be converted (entry cut, must not be re-created) and cannot be deleted (prose deletion is an `OTHER` hunk, forbidden by A5). This follows the task's own **Platt precedent** — a cut entry's prose trace stays as plain prose. **It already has an owner:** item 1 of the theory-note reconciliation list in `v1-derivation-appendix.md`, which proposes replacing the sentence with a pointer to Appendix G. Two further theory-note traces carry no author-year and were likewise untouched (§2 "…provided in the companion theory note."; §2 "The theory note proves that…"). **Net: 0 *convertible* manual author-year strings remain.**

**6. ⚠ The "11 printing" acceptance criterion is arithmetically unreachable; read it as 14.** The three cut works were **never in `refs.bib`** — its 14 entries are the 17 hand-built `\item`s **minus** Lieb, Platt and Anonymous. §A2 of the same task states the verified compile as "**14/14 bibitems**", which is what I measure. Reaching 11 would mean deleting three verified, cited entries, forbidden by A2. **Measured: 14 bibitems, 14 distinct keys cited, 0 uncited entries, 0 undefined citations.** Flagged rather than silently reconciled.

**7. Template port facts confirmed by direct reading, not inherited.** `[dblblindworkshop]` is the correct anonymous option (`sglblindworkshop` sets `\@anonymousfalse`); no short-paper option exists in this class; no checklist added. **`\workshoptitle` left UNSET** (Head has not supplied TTCL's expansion; ⛔ inventing one is forbidden) — and I re-verified *why* that is safe: `\@workshoptitle` is consumed only by `\@trackname`, which is used only inside the `\if@neuripsfinal` branch of `\@noticestring`. The submission footer reads, verbatim from the PDF: *"Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute."* Line numbers now appear (the class loads `lineno` for submissions) — expected, not a defect.

## Charter compliance
- **C-1 (as REVERSED 2026-07-07):** no defensive audit-confession paragraph exists or was added; J&P 2026 is cited for the primitive's introduction only (`\citet{jawahar_chlu_2026}`, one site).
- **Naming:** the continuity sentence is intact and renders *"the Causal Learning Unit (CLU), introduced as CHLU in Jawahar and Pierini (2026)"*. ⚠ **Editorial question:** `plainnat` prints "**and**" where the prose had "**&**". The sentence's content is unchanged and the year is right; hard-coding "&" would re-freeze the string and defeat the self-correcting mechanism this pass exists to install. **Flagged for the Head, not "fixed".**
- **C-2 / C-5 / C-6 / C-7 / C-9 / C-10:** untouched by construction — no claim, scale qualifier, certificate caveat, flag table, negative result or appendix was edited (`OTHER` = 0 proves it).
- **C-8 hermetic:** no sibling short was opened; `.claude/papers/v1-short/**` was touched only by `md5`.
- **Matrix:** no CM-worded claim was added or altered; CM-3-style energy-signal-superiority language was neither introduced nor is present.

## Deliverables
1. `.claude/NIPSsubmission/v1-ttcl/BUILD-NOTE-R5.md` — hunk register, the reconstruction proof, the full word-level check transcript, sweeps, orphan table, build/pages, anonymity, and the "noticed and not touched" list.
2. `pj_sub.tex` — the only file written.
3. Scratch (evidence): `.claude/scratch/v1-cite-and-template/` — `baseline/pj_sub.tex.orig`, `apply_all.py`, `hunks.json`, `verify.py`, `verify_words.py`, `final.diff`, `manifest_{before,after}.txt`, and builds `build_before/` (17 pp), `isoA/` (Part A isolated), `build_afterB/` (T1 variant, 15 pp), `build_final/` (**16 pp, the current source**).

## Git footprint
**None.** All writes are under `.claude/` (gitignored): `pj_sub.tex`, `BUILD-NOTE-R5.md`, this report, and scratch. No branch, no commit, no tracked file touched.

## Open questions / follow-ups / risks
1. **`refs.bib` `ć` fix (`Pavlovi{\'c}`)** — needs an owner. Until then the paper ships without `[T1]{fontenc}`; with the fix, restore the line and regain ~1 page.
2. **Ratify the Geifman placement** (same sentence, end, vs inside the `(LTT; …)` parenthetical).
3. **`&` → "and"** in the charter continuity sentence, from `plainnat`. Head call.
4. **`\workshoptitle`** still needs TTCL's expansion before any camera-ready build.
5. **Not verified under pdflatex** — no pdflatex on this machine. A pdflatex/bibtex build should be run before submission (and would likely handle `ć` correctly under T1, making item 1's fix belt-and-braces rather than load-bearing).
6. **The shipped `pj_sub.pdf` is stale** (still the 20:27 build) — this pass writes `.tex` only. Drop-in preview: `scratch/v1-cite-and-template/build_final/pj_sub.pdf`.
7. The theorist's wording debts (`/M_0`, "kinetic energy 0.72", det = 2.05 "contraction", the four theory-note sites) remain open and were **not** touched — all are prose/number edits barred by this diff contract.

## Proposed handover updates (for the Hub)
- **§1 (V1 paper state):** `pj_sub.tex` is now on the **NeurIPS 2026 `dblblindworkshop`** class with a real BibTeX bibliography: 0 manual author-year strings that can be converted, **15 `\cite*` macros / 14 keys / 14 bibitems / 0 undefined citations / 0 errors / 0 missing glyphs**, **16 pp total, main text pp. 1–8**. Author-year round-paren style (`plainnat` + `\PassOptionsToPackage{round}{natbib}`) is the declared choice. Anonymity clean (Anonymous Author(s), metadata scrubbed). Stale strings self-corrected from the verified `.bib`: **Angelopoulos 2021→2025** and J&P's truncated placeholder title.
- **§7 (paper debts, add one):** **`refs.bib` prints "Pavlovi" — the `ć` is dropped** under Times+T1; the paper currently omits `[T1]{fontenc}` to avoid it. One-character fix, owner needed.
- **§7 (paper debts, carry forward):** the `(Anonymous, 2026)` prose mention is still live in §1 and is the Head's reword (theorist's reconciliation item 1); the theorist's other three wording defects are untouched.
- **§8 (minor):** the task's "11 printing" criterion is an arithmetic slip — the three cut works were never in `refs.bib`; the correct target is **14/14**, as that task's own §A2 states.
