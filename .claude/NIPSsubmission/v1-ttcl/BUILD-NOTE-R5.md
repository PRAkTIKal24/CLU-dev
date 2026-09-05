# BUILD-NOTE-R5 — the cite pass + the NeurIPS 2026 template port

**Pass:** `v1-cite-and-template` (paper-writer spoke, 2026-08-27).
**File written:** `pj_sub.tex` **only**. `refs.bib`, `neurips_2026.sty`, `submission.tex`,
`figs/**` and `.claude/papers/v1-short/**` are byte-untouched (manifests in §7).

| | md5 |
|---|---|
| `pj_sub.tex` at start (**the pin**) | `6867e06b56d97aadc52398558e9e4797` |
| `pj_sub.tex` at end | `de4559a36af659bada4a56ea05156db7` |

⭐ The starting pin equals the md5 the theorist pass (`v1-derivation-appendix`) reported as its
post-edit state. **No Head edit intervened between the theorist landing and this pass**; the
scoping pin is confirmed, not merely reported.

**DIAL DECLARATION — Dials touched: NONE.** Citations wired and a document class changed.
No experiment, no configuration change, no measured value altered, no number touched.

---

## 1. THE DIFF CONTRACT (A5) — `OTHER` = **0**

**16 hunks: 13 `CITATION` · 1 `BIBLIOGRAPHY` · 2 `TEMPLATE` · 0 `OTHER`.**

### Proof that `OTHER` = 0 (not an assertion — a reconstruction)
`.claude/**` is gitignored, so `git diff` is unavailable. Instead: the 16 logged hunks were
applied to the pinned baseline by one script (`scratch/v1-cite-and-template/apply_all.py`), each
asserted to match **exactly once**; then a second script reverted all 16 from the shipped file:

```
reverted md5 = 6867e06b56d97aadc52398558e9e4797
baseline pin = 6867e06b56d97aadc52398558e9e4797     MATCH: True
```

Reverting the logged hunks reproduces the pin **byte-exactly** ⇒ the shipped diff *is* the logged
hunk set and nothing else. No typo fix, no grammar, no capitalisation, no rewording, no
re-ordering, no re-wrapping, no terminology harmonisation, and **no touch to any number,
caption, label or heading** entered the file. (`diff -u` confirms: 7 diff regions, 111 lines,
all accounted for below; full text at `scratch/v1-cite-and-template/final.diff`.)

### Hunk register

| # | id | class | site | change |
|---|---|---|---|---|
| 1 | `T1` | TEMPLATE | preamble | `\documentclass[11pt]{article}` → `\documentclass{article}` |
| 2 | `T2` | TEMPLATE | preamble | drop `geometry`; add `\PassOptionsToPackage{round}{natbib}`, `\usepackage[dblblindworkshop]{neurips_2026}`, `inputenc`, `amsfonts`, `url` |
| 3 | `C1a` | CITATION | §1 | `(ACT, Graves 2016; PonderNet, Banino et al. 2021)` → `\citealp` ×2 |
| 4 | `C1b` | CITATION | §1 | `(CALM, Schuster et al. 2022)` → `\citealp` |
| 5 | `C1c` | CITATION | §1 | `(Mixture-of-Depths, Raposo et al. 2024; Mixture-of-Experts, Shazeer et al. 2017)` → `\citealp` ×2 |
| 6 | `C1d` | CITATION | §1 | `(Gladstone et al. 2025)` → `\citep` |
| 7 | `C2` | CITATION | §1 Reference architecture | `Jawahar \& Pierini (2026)` → `\citet` |
| 8 | `C3` | CITATION | §4.1 | `(LTT; Angelopoulos et al. 2021)` → `(LTT; \citealp{angelopoulos_learn_2025})` ⭐ |
| 9 | `C4` | CITATION | §4.1 item 1 | `(Ramsauer et al. 2021)` → `\citep` |
| 10 | `O1-duane` | CITATION | App. F ¶1 | **orphan attached** to the HMC sentence |
| 11 | `O2-neal` | CITATION | App. F ¶2 | **orphan attached** to the ergodicity / momentum-refreshment sentence |
| 12 | `O3a-roberts` | CITATION | §5 design rules | **orphan attached** at the MALA naming |
| 13 | `O3b-roberts` | CITATION | App. F ¶3 | **orphan attached** at the FDT noise-scale sentence |
| 14 | `O4-geifman` | CITATION | §4.1 | **orphan attached** at the exit-threshold / reject-option clause |
| 15 | `O5-wales` | CITATION | §3.1 squeeze intro | **orphan attached** at the basin-hopping move |
| 16 | `B1` | BIBLIOGRAPHY | end | hand-built 17-`\item` list → `\bibliographystyle{plainnat}` + `\bibliography{refs}` |

---

## 2. Word-level byte-identity check (mandatory, A5)

Method: delete the `\cite*` macro from AFTER; delete from BEFORE the manual author-year token(s)
that macro replaced (empty for the six pure attachments); require the remainder to be identical.
Whitespace left behind by macro excision is normalised; **no other normalisation is applied.**

```
[C1a        ] PASS   BEFORE (ACT, Graves 2016; PonderNet, Banino et al. 2021)
                     AFTER  (ACT, \citealp{graves_adaptive_2017}; PonderNet, \citealp{banino_pondernet_2021})
                     surrounding words BEFORE : '(ACT,; PonderNet,)'
                     surrounding words AFTER  : '(ACT,; PonderNet,)'
[C1b        ] PASS   '(CALM,)'                        == '(CALM,)'
[C1c        ] PASS   '(Mixture-of-Depths,; Mixture-of-Experts,)' == same
[C1d        ] PASS   'energy-based verification have' == same
[C2         ] PASS   'introduced as CHLU in.'         == same
[C3         ] PASS   '(LTT;) wrapper'                 == same
[C4         ] PASS   'modern Hopfield memory yields'  == same
[O1-duane   ] PASS   'forms a detailed-balance kernel for $e^{-H/T}$.'                == same
[O2-neal    ] PASS   'step to ensure proper mixing across the state space.'           == same
[O3a-roberts] PASS   '(MALA) step to ensure adequate mixing.'                         == same
[O3b-roberts] PASS   'defined as $\sigma_i^\star=\sqrt{M_{{\rm eff},i}T\gamma(2-\gamma)}$.' == same
[O4-geifman ] PASS   'exit thresholds based on a dynamic relaxation ladder.'          == same
[O5-wales   ] PASS   'we apply a mass-weighted Lorentz squeeze $S^{(M)}_\zeta$.'      == same

CITATION hunks: 13 byte-identical, 0 drifted.   (requirement: 0)
```

No sentence was reflowed. Full transcript: `scratch/v1-cite-and-template/verify_words.py` output.

---

## 3. Residual sweep — positive-controlled

Normaliser built for the **full natbib set** (`\cite \citet \citep \citealp \citealt \citeauthor
\citeyear \citeyearpar \citenum`), so possessive/year-only forms cannot produce false drift hits.
`%`-comment lines excluded. Four author-year patterns (`X et al. YYYY`, `X and/& Y (YYYY)`,
`(X, YYYY)`, `X YYYY`).

| sweep | baseline (**positive control**) | final |
|---|---|---|
| manual author-year strings | **10 hits** (all 10 listed & converted) | **1** |

**The single residual is `(Anonymous, 2026)`** — §1 "Reference architecture", the cut theory
note. ⛔ It was **not** converted (the entry is cut by Head ruling and must not be re-created) and
⛔ **not** deleted (deleting prose is an `OTHER` hunk, forbidden by A5). This follows the task's
own Platt precedent: *a cut entry's prose trace stays as plain prose.* **It is a Head reword, and
it already has an owner** — item 1 of the theory-note reconciliation list in
`.claude/outputs/v1-derivation-appendix.md`, which proposes replacing the whole sentence with a
pointer to Appendix G. Two further theory-note prose traces carry no author-year and were likewise
left untouched (§2 "…provided in the companion theory note."; §2 "The theory note proves that…").

**natbib census (final):** `\cite` 0 · `\citet` 1 · `\citep` 8 · `\citealp` 6 · all other forms 0.
**Total 15 macros, 14 distinct keys.**

**Cut-entry sweep of the source** (positive-controlled): `Lieb` 1→**0**, `Robinson` 1→**0**,
`Platt` 2→**1**, `Anonymous` 2→**1**. The surviving `Platt` is the App. A.2 table cell
"Platt-calibrated" — ⛔ left as plain prose per ruling, cell not deleted, no citation created.
`\item ` 30→13 (the 17 reference items removed; 13 genuine list items untouched).

---

## 4. Orphan check — all five attached and **printing**

| entry | attached at | prints as |
|---|---|---|
| `duane_hybrid_1987` | App. F, the `min(1,e^{-ΔH/T})` detailed-balance sentence (the paper re-derives HMC without naming it) | "Simon Duane, A.D. Kennedy, Brian J. Pendleton, and Duncan Roweth. Hybrid Monte Carlo…" ✓ |
| `brooks_mcmc_2011` (= **Neal**) | App. F, ergodicity / momentum refreshment | "Radford M. Neal. MCMC Using Hamiltonian Dynamics…" ✓ |
| `roberts_exponential_1996` | **two** sites: §5 design rules (MALA named) + App. F FDT noise scale | "Gareth O. Roberts and Richard L. Tweedie…" ✓ |
| `geifman_selective_2017` | §4.1, same sentence as the LTT citation | "Yonatan Geifman and Ran El-Yaniv…" ✓ |
| `wales_global_1997` | §3.1 squeeze introduction | "David J. Wales and Jonathan P. K. Doye…" ✓ |

⛔ **Cut entries confirmed absent from the built bibliography** (`.bbl` *and* rendered PDF text of
the References pages): `Lieb` 0 · `Robinson` 0 · `Platt` 0 · `Anonymous` 0.
⛔ The theory-note entry was **not** re-created, and **no** "provided in the supplementary
material" note was added to any entry.

### ⚠ The "11 printing" acceptance criterion is arithmetically unreachable — read it as 14
The criterion says *"All 14 `refs.bib` entries minus the 3 cut = 11 printing."* The three cut
works were **never in `refs.bib`**: the file's 14 entries are exactly the 17 hand-built `\item`s
**minus** Lieb, Platt and Anonymous (17 − 3 = 14). §A2 of the same task states the verified
compile as **"14/14 bibitems"**, which is what is measured here. Reaching 11 would require
deleting three verified, cited entries — forbidden by A2. **Measured: 14 bibitems, 14 distinct
keys cited, 0 uncited entries, 0 undefined citations.** Flagged, not silently reconciled.

---

## 5. Citation style (A4) — **declared, not chosen silently**

**Author-year, round parentheses, `plainnat`, via natbib (the template's default, `nonatbib`
NOT used).** Rationale: the prose is already author-year, so the diff stays word-checkable and
the charter continuity sentence keeps its shape; the sibling pass's numeric-on-anonymity-grounds
ruling was withdrawn (that rule governs phrasing, not rendering).

⚠ **A real defect found and fixed:** natbib's default delimiter is **square brackets**. The first
ported build rendered `[Gladstone et al., 2025]`, `[Ramsauer et al., 2021]` and — worse —
**"Jawahar and Pierini [2026]"**, i.e. the charter-mandated CHLU continuity sentence in brackets,
sitting beside manual round parentheses in the same sentence. Fixed with
`\PassOptionsToPackage{round}{natbib}` — **the mechanism documented in the shipped template's own
preamble comments**. Verified in the rendered PDF: all citations now round.

### What the conversion self-corrected from the verified `.bib` (⭐ the point of A1)
- **`Angelopoulos et al. 2021` → `Angelopoulos et al., 2025`**, and the reference now prints the
  real record: *The Annals of Applied Statistics* 19(2), June 2025, doi 10.1214/24-AOAS1998.
  ⛔ The year was **not** hand-edited; the string was converted and BibTeX printed it.
- **`Jawahar, P., Pierini, M. (2026). [CHLU primitive].`** — a truncated placeholder title in the
  hand-built list — now prints the full title, arXiv id 2603.01768 and date.
- `Graves` prints **2016** (the corrected `year` field), despite the key still reading `_2017`.
- `Neal(2011)` prints from key `brooks_mcmc_2011`; `raposo_mixture--depths_2024`'s double hyphen
  and the `@misc`/`@article`/`@incollection`/`@inproceedings` split were left exactly as shipped.

---

## 6. Build

**Toolchain: `tectonic 0.15.0` (XeTeX). No `pdflatex`, `bibtex` or `latexmk` on this machine** —
the port is verified under tectonic only and has **not** been pseudo-verified under pdflatex.

| | LaTeX errors | undefined citations | undefined refs | missing glyphs | bibitems |
|---|---|---|---|---|---|
| **final** | **0** | **0** | **0** | **0** | **14** |

⚠ A `grep -ci error` on the log returns 1: the hit is the loaded package **`infwarerr`** — a false
friend, read in context as required. `^! ` count is 0.

### Pages and the main-text split

| build | total | main text | References |
|---|---|---|---|
| **before** (pin; `article` 11pt, hand-built list) | **17 pp** | pp. 1–9, App. A begins p. 9 | p. 16 |
| Part A alone (isolated probe, `article`+natbib) | 18 pp | — | — |
| **after** (shipped: `neurips_2026` `dblblindworkshop`) | **16 pp** | **pp. 1–8, App. A begins p. 8** | p. 15 |

Main text tightened by ~1 page purely from the template's type block (5.5in × 9in vs 1in margins).
Section starts after the port: §1 p2 · §2 p3 · §3 p4 · §4 p6 · §5 p7 · App. A p8 · App. B p10 ·
App. C p11 · App. D/E p12 · App. F/G p13 · References p15.

### Template port specifics
- `[dblblindworkshop]` — the correct and only anonymous workshop option (`sglblindworkshop` sets
  `\@anonymousfalse`). ⛔ No paper checklist added (main-track artifact). ⛔ No short-paper option
  exists in this class; short vs full is a CFP page-count matter.
- **`\workshoptitle` deliberately left UNSET** — the Head has not supplied TTCL's expansion and
  ⛔ inventing one is forbidden. Advisor-verified and **re-confirmed here by direct reading of the
  `.sty`**: `\@workshoptitle` is consumed only by `\@trackname`, and `\@trackname` is used only
  inside the `\if@neuripsfinal` branch of `\@noticestring` — so it renders **only** in a
  camera-ready build. The submission build's page-1 footer reads, verbatim as measured in the
  PDF: *"Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do
  not distribute."* **Not blocking.**
- The `.sty` already in `v1-ttcl/` is byte-identical to the shipped one (`f447d330…`); no copy made.
- Line numbers now appear (the class loads `lineno` for submission builds) — expected, not a defect.

### Anonymity checklist after the port
- Author block renders **"Anonymous Author(s) / Affiliation / Address / email"**; the
  `[AUTHORS PLACEHOLDER]` string is suppressed by `\if@anonymous` and never printed.
- `[WORKING TITLE: …]` placeholder retained per policy.
- **PDF metadata scrubbed:** no Title/Author/Subject/Keywords; `Metadata Stream: no`.
- Main-text sweep (pp. 1–8): `@` 0 · acknowledgments 0 · "our previous/prior work" 0 · `github.com`
  0 · `Pratik` 0 · `Maurizio` 0 · `CERN` 0 · funding/grant 0.
- Author surnames appear **only** in bibliography entries and in the charter-mandated continuity
  sentence, which renders: *"Our reference memory is the Causal Learning Unit (CLU), introduced as
  CHLU in Jawahar and Pierini (2026)."*

---

## 7. Protected files — byte-untouched (manifest)

```
submission.tex          caef2272f9dc96d349b46486563d24ee   (unchanged)
refs.bib                58c75795e1fa8f5e46a74cbc2902e457   (unchanged)
neurips_2026.sty        f447d3302c8719cb27619a074c876b44   (unchanged)
.claude/papers/v1-short/**  11 files, before-manifest ≡ after-manifest (diff empty)
```

⚠ The shipped `pj_sub.pdf` in this directory is **still the stale 2026-08-27 20:27 build** — this
pass writes `pj_sub.tex` only, as scoped. Drop-in preview of the current source:
`.claude/scratch/v1-cite-and-template/build_final/pj_sub.pdf` (16 pp).

---

## 8. Noticed and NOT touched

1. ⛔⛔ **`refs.bib` prints "Milena Pavlovi" — the `ć` is dropped.** Found only because this pass
   switched to a real bibliography (the hand-built list said "Ramsauer, H., et al."). Under
   tectonic/XeTeX the class's Times (`\rmdefault=ptm`) **plus** `[T1]{fontenc}` yields
   `Missing character: There is no ć ("107) in font ptmr8t!` and the letter is **silently absent
   from the PDF** — an author's name misspelled in the bibliography, with no error.
   **Therefore `\usepackage[T1]{fontenc}` was deliberately NOT added** (the one deviation from the
   template preamble; `inputenc`, `url`, `booktabs`, `amsfonts`, `microtype`, `hyperref` all
   present). Measured: with T1 → 1 missing glyph, "Pavlovi", 15 pp; without T1 → **0 missing
   glyphs, "Pavlović" correct, 16 pp**. `Candès`, `Schäfl`, `Günter` render correctly either way.
   **Owner: whoever owns `refs.bib`.** The permanent fix is one character —
   `Pavlovi{\'c}` — after which `[T1]{fontenc}` can be restored (recovering the page and the
   better hyphenation). ⛔ Not made here: `refs.bib` is out of this pass's write scope.
2. `\usepackage[hidelinks]{hyperref}` kept rather than the template's bare `\usepackage{hyperref}`
   — dropping `hidelinks` would change link rendering, i.e. visible output. Flagged, not changed.
3. The `\small` that scoped the hand-built reference list is gone with it; the bibliography now
   sets at normal size. Restoring `\small` is a one-line call if pages get tight.
4. The preamble comment *"Standard-package build; compiles with pdflatex or tectonic"* is now
   partially stale (the build requires `neurips_2026.sty`, which is present in-directory). Comment
   edits are `OTHER`; left byte-identical.
5. Pre-existing typography warnings unchanged and not touched: 1 overfull hbox (1.43 pt) at the
   C.1 grid, 1 overfull (2.78 pt) + several underfull hboxes in the B.1 certificate table.
6. The theorist pass's open wording debts are **untouched and still open** — §3.1's displacement
   law missing `/M_0`, §3.2's "kinetic energy 0.72", det = 2.05 called a "contraction", and the
   four theory-note sites. ⛔ All are numbers/prose and outside this diff contract.
