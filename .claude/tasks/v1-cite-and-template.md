# V1 — wire the citations, then port to the NeurIPS 2026 template

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.**

**Agent:** `paper-writer` (Bash-capable in this rig — it must build).
**Writes:** `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` only. **Deliverable #1:** `BUILD-NOTE-R5.md` · **Report:** `.claude/outputs/v1-cite-and-template.md`

⛔⛔ **RUN THIS AFTER THE THEORIST PASS LANDS AND IS VERIFIED.** Both write `pj_sub.tex`; a sibling paper hit exactly this clobber. **Mechanical precondition: `.claude/outputs/v1-derivation-appendix.md` must exist on disk.** If it does not, STOP and report.

⛔ **Pin check.** Compute `md5 pj_sub.tex` at start and report it. ⚠ This file is live-edited and has moved seven times this session; the Head may edit between the theorist landing and your launch, so **the pin is a report item, not an abort condition** — but a diff baseline you cannot name is not a baseline.

---

## PART A — the cite pass

### A1. Convert every manual author-year mention to `\cite`
The paper currently has **0 `\cite` commands** and a **hand-built `\item` reference list**. Convert every in-prose author-year string to `\citep`/`\citet`, and replace the hand-built list with `\bibliographystyle{...}` + `\bibliography{refs}`.

⭐ **This conversion is the mechanism by which any stale year or truncated title self-corrects from the verified `.bib`** — and it only works if the conversion is **complete**. A residual manual string keeps its frozen text forever.

⚠ **The Head has relied on this explicitly:** the prose reads *"(LTT; Angelopoulos et al. 2021)"* while `refs.bib` carries the **2025** Annals of Applied Statistics record. ⇒ **converting that string is what makes the year correct.** ⛔ Do not "fix" the year by hand; convert the string and let BibTeX print it.

### A2. `refs.bib` — already verified, do not re-derive
`.claude/NIPSsubmission/v1-ttcl/refs.bib`, **14 entries**, Advisor-verified and compile-tested under this paper's own preamble (**0 LaTeX errors · 0 undefined citations · 0 BibTeX errors · 14/14 bibitems**). Three defects were already repaired: 10 Zotero `file =` fields carrying an account name (a **double-blind** hazard), one wrong year (`graves_adaptive_2017` had the v2 revision date 2017; corrected to **2016**), and 8 abstracts.

⛔ **Four facts about it you must not "fix":**
1. ⛔⛔ **`brooks_mcmc_2011` is the Neal chapter.** Zotero keyed it to the *book's editors* (Brooks, Gelman, Jones & Meng). **The `author` field is correct — `Neal, Radford M.` — and it renders as `Neal(2011)`.** ⚠ **Grepping for "Neal" will not find the key.** Leave the key alone; key churn is not worth it.
2. ⚠ `graves_adaptive_2017`'s **key still says 2017** while its `year` field is now **2016**. Keys are source-only and never printed. ⛔ Do not "correct" it.
3. ⚠ `raposo_mixture--depths_2024` contains a **double hyphen**. It is the real key. ⛔ Do not normalise it.
4. ⛔ Entries are `@misc` for arXiv-only works and `@article`/`@incollection`/`@inproceedings` for venue-published ones. That distinction is deliberate.

### A3. ⛔⛔ The five orphans MUST be attached, or they vanish silently
**Head ruling: attach.** These five are cited **nowhere** in prose, and today the hand-built list prints them anyway. ⛔ **The moment you switch to `\bibliography{}`, any entry without a `\cite` disappears with no error and nothing in the diff.** Attach each to the sentence named:

| entry | attach at |
|---|---|
| `duane_hybrid_1987` | App. F's HMC construction — the sentence describing a sign-symmetrized squeeze accepted by $\min(1,e^{-\Delta H/T})$ forming a detailed-balance kernel for $e^{-H/T}$. **The paper is re-deriving Hybrid Monte Carlo without naming it.** |
| `brooks_mcmc_2011` (= Neal) | App. F, the ergodicity / momentum-refreshment sentence (*"...non-ergodic. Ergodicity strictly necessitates momentum refreshment..."*) |
| `roberts_exponential_1996` | §5's design rules, where **MALA** is named with no source; second live site in App. F at the FDT noise-scale sentence |
| `geifman_selective_2017` | §4.1, beside the existing Learn-then-Test citation — the gate **is** a selective-prediction / reject-option classifier |
| `wales_global_1997` | §3.1's squeeze introduction — a perturb-then-relax move escaping a barrier into another basin **is** the basin-hopping move; the paper already uses "basin-hop" |

⛔ **Cut by Head ruling — must NOT appear anywhere:** Lieb & Robinson · Platt · the `Anonymous (2026)` theory note. ⚠ **The theory-note entry must not be re-created**, and no *"provided in the supplementary material"* note may be added to any entry.
⚠ **Platt's only prose trace is a table cell reading "Platt-calibrated".** With the entry cut, that string stays as **plain prose** — ⛔ do not create a citation for it and do not delete the cell.

### A4. Citation style
⛔ **Do not choose silently — report what you chose and why.** The template ships `natbib` behaviour by default and declares a `nonatbib` option. ⚠ A sibling pass first ruled *numeric* style mandatory on anonymity grounds and then **withdrew that ruling**: the rule governs **phrasing**, not citation rendering, and author-year is compliant. **Author-year is expected here** since the prose is already author-year, which also keeps the diff checkable.

### A5. ⛔⛔ THE DIFF CONTRACT
**Every changed hunk must carry exactly one label: `CITATION` · `BIBLIOGRAPHY` · `TEMPLATE`. ⛔ A hunk that is none of these is class `OTHER` and `OTHER` must be ZERO.**

⛔ Forbidden however tempting: typo fixes, grammar, capitalisation, rewording, re-ordering, re-wrapping, terminology harmonisation, and any touch to a number, caption, label or heading.
⭐ **A word-level check is mandatory:** each citation hunk prints before/after with the citation portion marked, confirming the surrounding words are **byte-identical**. A conversion that also reflows a sentence is a violation.
⚠ **Build your normaliser for the FULL natbib set** — `\cite`, `\citet`, `\citep`, `\citealp`, **`\citeauthor`, `\citeyearpar`**. A sibling pass reported six false prose-drift hits purely because its regex missed the possessive forms.

---

## PART B — the template port

⚠ **Marked strikeable: if the Head prefers to port personally, skip Part B and report Part A complete.**

**Source: `~/Downloads/Formatting_Instructions_For_NeurIPS_2026/`.** ⭐ **Advisor-verified: the `neurips_2026.sty` already in `v1-ttcl/` is byte-identical to the shipped one (`f447d330…`) — no copy is needed.**

**The port:**
1. `\documentclass[11pt]{article}` → **`\documentclass{article}`** + **`\usepackage[dblblindworkshop]{neurips_2026}`**.
2. Add `\workshoptitle{...}` with TTCL's own expansion. ⚠ **If the Head has not supplied it, leave it unset and say so** — ⭐ **Advisor-verified in the `.sty`: `\workshoptitle` renders ONLY in the `final` (camera-ready) build.** A submission build prints *"Submitted to … NeurIPS 2026"* regardless, so this is **not blocking**. ⛔ Do not invent an expansion.
3. Align the preamble with the template's (`inputenc` utf8, `fontenc` T1, `hyperref`, `url`, `booktabs`, `amsfonts`, `microtype`). ⚠ **`refs.bib` contains accented author names** (`Candès`, `Schäfl`) which compile correctly today — verify they still do.
4. ⛔ **Do not add the paper checklist.** It is a NeurIPS main-track artifact.

**Two Advisor-verified facts, so neither is later mistaken for a defect:**
- ⛔ **There is NO short-paper variant of this class.** The declared options are exactly: `final · nonatbib · preprint · main · position · eandd · creativeai · education · sglblindworkshop · dblblindworkshop · nonanonymous`. Short vs full is a **page-count distinction in the CFP**, not a template option. *(The four `short` hits in the `.sty` are `\abovedisplayshortskip`-class math spacing.)*
- ⛔ **`[dblblindworkshop]` is the correct and only anonymous workshop option** — `sglblindworkshop` sets `\@anonymousfalse`. TTCL is double-blind.

**Anonymity checklist after the port:** author block renders anonymously · PDF metadata scrubbed · ⛔ the only permitted author-surname appearances are **bibliography entries** and the charter-mandated CHLU continuity sentence.

---

## Deliverables

1. **`BUILD-NOTE-R5.md`** — every hunk labelled per A5, with the word-level byte-identity check printed. ⛔ `OTHER` = 0.
2. **A residual sweep, positive-controlled:** manual author-year strings remaining (**expect 0**), `\cite` count, bibitem count, undefined citations.
3. **The orphan check:** all five attached entries confirmed **printing**; the three cut entries confirmed **absent**.
4. **Build:** 0 errors · 0 undefined citations · 0 undefined references. **Report total pages and the main-text page split** before and after the port.
5. **Anything you noticed and did not touch**, listed.

## Acceptance criteria

- ⛔ `OTHER` hunks = **0**; prose byte-identical around every conversion.
- ⛔ `submission.tex` and `.claude/papers/v1-short/**` byte-untouched (md5 manifest printed).
- All 14 `refs.bib` entries minus the 3 cut = **11 printing**; ⛔ **0 undefined citations**.
- `Lieb`, `Platt`, `Anonymous` = 0 in the built bibliography.
- Every negative positive-controlled.

## ⚠ Grep hazards

⛔ `grep` is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long `.tex` lines it either **errors "exceeds complexity limits" and exits 0** — a silent false negative — or **hangs**. Use **`/usr/bin/grep`**; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — sweep per-file. ⚠ zsh does not word-split: quote any glob. ⚠ **Case-sensitivity and false friends have produced four wrong findings in this estate this session** — `matvec` returned 0 while the claim was present spelled out, and a "warning" hit turned out to be the log line reporting *zero* warnings. **Read every hit in context.**

## DIAL DECLARATION
**Dials touched: NONE.** Wires citations and changes a document class. No experiment, no configuration change, no measured value altered.
