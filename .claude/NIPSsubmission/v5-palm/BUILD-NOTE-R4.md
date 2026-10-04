# `v5-palm/pj_sub.tex` — BUILD NOTE R4 (the HEAD-APPROVED MINIMAL PASS)

**Deliverable #1 of `pj-minimal-v5`** (Shorts-Advisor charter Addendum 61, 2026-08-22). ⛔ **R3 was rejected by the Head as over-scoped and reverted; this pass executes EXACTLY the Head-approved line-item list and nothing else.** Everything below is measured, with the command named. Nothing is asserted.

**DIAL DECLARATION: none — editorial. Zero new measurements. Zero new numbers.** Every added numeric token traces to an ancestor line (two-way numcheck, §4; orphan list **empty**).

| | |
|---|---|
| **File edited (the only content file touched)** | `pj_sub.tex` |
| **md5 before** | `6c1902f74ee9611d718cc65b9fd1a031` (matches the value named in the task) |
| **md5 after** | `594a6919d773b5e9e82b68af07816d0e` |
| **size** | 302 → 426 lines |
| **Edits applied** | **30**, each an exact-match **single-occurrence** replacement asserted by script (`.claude/scratch/pj-minimal-v5/minimal.py`; every assert passed on first run) |
| **Built with** | `/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex` ×2, run **only inside this folder** |
| **Result** | `Output written on pj_sub.pdf (18 pages, 1743808 bytes)` — **0 errors · 0 undefined references/citations · 0 overfull `\vbox`** |
| **Overfull `\hbox`** | **1**, and it is ⚠ **PRE-EXISTING and NOT ENUMERATED** — see §7. The two tables the task named are **both fixed** (was 3 overfull, now 1). |

---

## 0 — Byte-untouched verification for the rest of the folder

`md5 * figs/*` before vs. after (`folder.PRE.md5` / `folder.POST.md5` in scratch). **The only differing lines are `pj_sub.tex`, `pj_sub.aux`, `pj_sub.log`, `pj_sub.pdf`.** Every other file is byte-identical, including `pj_sub.out`. (`BUILD-NOTE-R4.md`, this file, is the pass's own new deliverable #1.)

| file | md5 | status |
|---|---|---|
| `submission.tex` | `1d0906fe45dc78436880c938ad227332` | **unchanged** (mtime still Aug 21 19:16) |
| `neurips_2025_ml4ps.sty` | `393afa47218eca1dbfdf4c42cb0da759` | unchanged |
| `BUILD-NOTE.md` | `2d9bbc948e175c6bc9fa0ca00b45f9c3` | unchanged |
| `BUILD-NOTE-R3.md` | `67e23b8939e652dac5009f8159552d40` | unchanged |
| `submission.{aux,log,out,pdf}` · `pj_sub_buildcopy.*` | — | unchanged |
| all 11 `figs/*.png` | — | unchanged (read-only; none regenerated) |

---

## 1 — The `§`-symbol rule (Head style rule, global)

**4 pre-existing occurrences converted; 0 remain.** `grep -c '\\S' pj_sub.tex` → **0**; `grep -c '§' pj_sub.tex` → **0**. No `\label` had to be added — all four targets already carried one.

| # | line (pre) | before | after | label existed? |
|---|---|---|---|---|
| 1 | 50 | `optimum lie? (\S\ref{sec:vcurve})` | `optimum lie? (Sec.~\ref{sec:vcurve})` | yes (`sec:vcurve`) |
| 2 | 51 | `localized item? (\S\ref{sec:vault})` | `localized item? (Sec.~\ref{sec:vault})` | yes (`sec:vault`) |
| 3 | 52 | `mathematically leaks? (\S\ref{sec:deletion})` | `mathematically leaks? (Sec.~\ref{sec:deletion})` | yes (`sec:deletion`) |
| 4 | 79 | `discussed in \S\ref{sec:vcurve}` | `discussed in Sec.~\ref{sec:vcurve}` | yes (`sec:vcurve`) |

Every sentence **added** by this pass uses the same form: item 1 spells the Guo reference as **`Guo et al.'s (2020) Sec.~2, Eq.~(1)`**, and all cross-references added inside restored blocks are `Appendix~\ref{...}` / `Figure~\ref{...}` (base forms, no `\S`).

---

## 2 — Part 1: figures. **11/11 shipped, 0 placeholders left**

`grep -c 'framebox\|Placeholder'` → **0**.

| # | figure | home | source of caption | note |
|---|---|---|---|---|
| 1 | `fig1_damping_optimum.png` | **main text**, Sec. 3.1 | caption already in place, **untouched** | referee MUST-restore; only the `\framebox` scaffold + its `% Image fallback…` comment were removed |
| 2 | `fig2_vault.png` | **main text**, Sec. 3.2 vault result | **base verbatim** (base L245) | referee MUST-restore; carries `\label{fig:vault}` |
| 3 | `figB_dlaw.png` | App A | caption already in place, **untouched** | placeholder replaced |
| 4 | `figB_signflip.png` | App A | **base verbatim** (base L150) | base home |
| 5 | `figB_massive_vs_flat.png` | App A | **base verbatim** (base L156) | base home |
| 6 | `figA1_damping_optimum_full.png` | App B (head of section) | **base verbatim** (base L164) | base home |
| 7 | `figC_lambda_coset.png` | App B | caption already in place, **untouched** | placeholder replaced |
| 8 | `figC_register_capacity.png` | App B | **base verbatim** (base L219) | base home |
| 9 | `figC_Tstar.png` | App B | **base verbatim** (base L225) | base home |
| 10 | `fig2_two_instruments.png` | App B | **base verbatim** (base L232) | base home; base carries no `\label`, so none was invented |
| 11 | `figC2_vault_emergent.png` | App C (end) | **base verbatim** (base L297) | base home; base carries no `\label`, so none was invented |

Per the Head's rule, figures 8, 9, 10, 11 ship to their appendix homes **with their base captions even though their claim text is not in the paper** — the caption states the result, which is exactly the banked-information-in-an-appendix case.

### The two truncated tables — **both fixed, 0 content change, `\small` NOT used as the fix**

| table | before | after (R3 bank, pre-fitted) | overfull |
|---|---|---|---|
| App A, `T>0` face of the budget (pre-l.190–200) | `\begin{tabular}{lll}` | `\begin{tabular}{p{0.19\linewidth}p{0.40\linewidth}p{0.31\linewidth}}` + one `\addlinespace` | **177.50pt → 0** |
| App B, four-instrument shape claims (pre-l.224–235) | `\begin{tabular}{lcccc}`, 3 wide rows | **transposed**: `{llcccc}`, quantity × instrument rows, seeds as columns, **plus the `Mean ± sd` column** | **152.59pt → 0** |

⛔ **No `\small` was added anywhere.** File-wide `\small` count went **6 → 5** (three placeholder captions removed; two restored tables carry the `\small` their **base** versions carry). The two fixed tables keep exactly the `\small` they already had before this pass; it was not introduced, and it is not the fix.
✅ **Value-for-value check on the transposed table: all 39 values present, none altered** (1 × `γ_crit` row of 3, + 4 instruments × 3 seeds × 3 quantities = 39), and the `Mean ± sd` column is a restoration, not a recomputation — each entry appears verbatim in base L183.

---

## 3 — Part 2: the 12 approved text items (all executed)

| # | site (post) | what changed | ancestor |
|---|---|---|---|
| 1 | l.80 | the Guo sentence replaced by the base's verbatim form, `§` spelled out: *"Formally, \emph{certified} removal is Guo et al.'s (2020) Sec.~2, Eq.~(1), an $\varepsilon$ condition with an unnumbered $(\varepsilon,\delta)$ relaxation immediately after; exact methods delete by isolation, with cost and capacity formalisms in place (Bourtoule et al., 2021; Ginart et al., 2019; **Sekhari et al., 2021**)."* | base L51 |
| 2 | l.83 | `(Mo, 2026)` → `(arXiv:2605.03338)` | base L53 |
| 3a | l.93 | *"strictly minimized at"* → *"minimized at"* | base L59 |
| 3b | l.189 | *"strictly aligning with"* → *"against"* | base L128 |
| 4 | l.391, l.411 | *"Emprical Result"* → *"Empirical Result"* | base L331 |
| 5 | l.158 | J&P entry loses *"[Reference redacted for double-blind review.]"* → full third-person entry: *"CHLU: The causal Hamiltonian learning unit as a symplectic primitive for deep learning. arXiv:2603.01768 (short paper, ICLR 2026 AI \& PDE workshop)."* | `v2-neurreps/submission.tex` L179 (Advisor-ratified Add. 60) |
| 6 | l.124 | clause appended: *"and we claim no certified $(\varepsilon,\delta)$ unlearning"* | base L79 |
| 7 | l.124 | sentence added verbatim: *"This is a store-level guarantee only --- the frozen encoder and any residue of past writes in a learned landscape are separate channels"* | base L79 |
| 8a | l.124 | *"attribute-based eviction"* → *"priority/attribute-based eviction; recency-based eviction remains intrinsically history-dependent and is excluded"* | base L79 |
| 8b | l.39 (abstract) | scale clause appended: *"on a designed, non-learned 3-dimensional datastore at capacities 8--64"* | base abstract L32 (*"on a designed store of dim 3, capacity 8--64, no learning"*) |
| 9 | l.136 (Limitations) | new bullet — *"\textbf{No Task-Level Payoff:} Stated once: external benchmarks won on their own headline metric = ZERO."* | base L79 (sentence) + base L87 / base L340 (bullet label) |
| 10 | l.185 | `($\ell_\theta/\Delta<0.05$; …)` folded into the parenthesis at the **$3.77\pm0.23\times$** site | base L73 |
| 11 | l.112 | *"The direct first-passage vault reads $86.97\pm2.94\times$ and is boundary-layer biased on the outside arm ($\ell_\theta/\Delta=0.079$), so $107.77\times$ is the quoted number and travels with its estimator's name."* | base L241 (verbatim) |
| 12 | l.95 | *"That low endpoint is the ring-profile probe's resolution on a checkpoint whose Hessian $\mu^2$ is machine zero rather than a spectral mass, so eleven orders is one curve on one instrument."* | base L61 (verbatim) |

**⚠ Two line-number departures from the task text, both reported rather than guessed (§7 items B and C):** item 10's *"l.116 area"* contains no `3.77×` — the rider was applied at the actual `3.77±0.23×` site; item 11's *"first 107.77× site"* was read as the first **body** site (Sec. 3.2), the abstract occurrence being governed by item 8b.

### ⛔ The struck item — **branch executed: ROWS-RESTORE**
The 15 negatives rows **did** restore (20/20, §5). The App-E lead sentence *"every negative result observed during evaluation is documented below"* is therefore **true as written and was NOT softened**, per the task's conditional. If the Head later strikes any row, the softening becomes owed.

---

## 4 — Two-way numcheck (script: `numcheck.py` / `numcheck2.py`)

- **Forward.** 166 added lines → **264 distinct numeric tokens** → **ORPHANS: 0.** Every token traces to `submission.tex` (base), the R3 bank, `pj_sub.tex` PRE, or `v2-neurreps/submission.tex`.
- **Stronger forward test (verbatim, not token-level).** Of the 166 added lines, **153 appear verbatim in an ancestor file.** The **13 hand-composed lines are exactly the 12 approved items + the 4 `§` conversions** (several land on the same source line): abstract scale clause · the three `\item` conversions · the Titans `Sec.~` conversion · the Guo sentence · the Mo cite · the 86.97× sentence · the deletion-scope block · the score-sentence bullet · the 3.77× rider · the *"against"* fix · the R50/TTL sentence. **No other line in this file was composed by hand.**
- **Reverse.** Numeric tokens present PRE and absent POST: **1**, and it is `0.8` from the deleted `\parbox{0.8\textwidth}` placeholder scaffolds. **Zero measured values were lost.**
- **Only one added sentence is a two-ancestor splice** and it is flagged here: the App-D retrieval-geometry sentence (§6(d)) takes `1.146/1.083/0.979/0.874/0.752`, `A:1→0.5→0.2→0.1→0.06`, `±0.20`, `0.75–0.77` from **base App D L312** and `1.52×` / `1.146→0.752` / *"independent of age"* from **base L81**. No number was combined, derived or rounded.

---

## 5 — Part 3: banked information restored to APPENDICES ONLY

**(a) Appendix E — 5 rows → 20/20 rows** (11 in table 1, 9 in table 2; R3-bank arrangement, base-verbatim wording). ⚠ **FLAG FOR THE HEAD:** the row *"The lifecycle's protected-fraction leg is exercised — No: 0 refusals on the stream at the measured operating point. Reported as unexercised."* is restored, but **its host claim (the three-state PROTECTED⇄ACTIVE→TRASH lifecycle, base contribution 4) is not in this paper.** It is currently a negative about a mechanism the reader never meets. **Head to keep or cut** — cutting it is a one-line delete and drops the count to 19/20.

**(b) Tables.** App C gains the **emergent-refrigerator table** (24 field cells, obs/absorb 0.9998±0.0019) and the **confinement / hop-fraction table** (which is where the confinement control numbers live: 5.50/42.97/2.36 % → 0.0000 in the hole, scalar control 0.73/10.20/0.26 %), both R3-bank pre-fitted. App B gains the **instrument-gap table**. The surviving App-B table's **mean ± sd** row is restored as the transposed table's `Mean ± sd` column (§2).

**(c) Appendix D** gains the base **Prior-art-in-full paragraph** verbatim — Snyder (1977) / Andersson & Ottmann (1995) / Micciancio (1997) / Naor & Teague (2001) / Blelloch & Golovin (2007) / **Blelloch, Golovin & Vassilevska (2008)** (the fourth attribution), the explicit **no-priority clause** (*"We claim no priority over order-independent placement and no novelty for the displacement rule or its delete-time repair"*), and the *"'Fix-up cascade' is our name for…"* closer. Its base home **is** Appendix D — the section is titled *Prior Art*.

**(d) Appendix D** (l.382) gains the R50 differentiator + TTL comparator (`1.146→0.752`, `1.52×`, `0.75–0.77`) in the Trilemma paragraph. ⛔ **Not in main text.**

### ⭐ THE HEAD'S REPORT ITEM — appendices ADDED or CHANGED vs the Head's rewrite

| appendix | ADDED? | CHANGED? | what changed |
|---|---|---|---|
| **A** `app:budget` | no | **yes** | table re-set to non-truncating columns (0 values changed) · `figB_dlaw` placeholder → real figure · **+2 figures** (`figB_signflip`, `figB_massive_vs_flat`) · item 3b · item 10 |
| **B** `app:emergent` | no | **yes** | four-instrument table → transposed non-truncating version **incl. the `Mean ± sd` column** · **+1 table** (instrument gap) · `figC_lambda_coset` placeholder → real figure · **+4 figures** (`figA1…`, `figC_register_capacity`, `figC_Tstar`, `fig2_two_instruments`) |
| **C** `app:vault` | no | **yes** | **+2 tables** (emergent refrigerator; confinement / hop fractions) · **+1 figure** (`figC2_vault_emergent`) |
| **D** `app:deletion` | no | **yes** | **+prior-art paragraph** (no-priority clause + 4th B&G attribution) · **+R50/TTL retrieval-geometry sentence** |
| **E** `app:negatives` | no | **yes** | 5 → **20** rows · header typo `Emprical` → `Empirical` |
| — | — | — | ⛔ **NO APPENDIX WAS NEWLY CREATED.** The A–E structure of the Head's rewrite is unchanged; no section was added, removed, renamed or reordered. |

---

## 6 — ⛔ Banked items deliberately NOT restored (base home = main text, or on the task's exclusion list)

Named individually so the Head can see what was left on the shelf: **the seam sentence** · **the Titans one-liner** · **all abstract TTL / conditions restorations beyond item 8b** · **the trilemma dial** (*"dropping amplitude-independent latency is the compute-adaptive-read dial…"*) · **the status headers** (*"(Designed atom store, dim 3, capacity 8–64, no learning ⇒ verification.)"* and the per-appendix verification/evidence headers) · **the anonymization note** · **the `fdt` fine-print block beside Sec. 3.2** · **the `T*` claim text** · **the designed-symmetry Sec. 3.2 block** · **all A2 vocabulary fixes not enumerated** (incl. *"Cross-Instrument Verification"* → *"Evidence"*, *"confirmed"* → *"measured"*, the *"never the vault"* rider on the coupled-bath sentence) · **all intensifier / garble edits**. Also left out, as neither enumerated nor appendix-homed here: the base's main-text confinement contrast percentages, the base's `Measured: "…"` quote block in Sec. 3.3, the Contributions paragraph, and the un-enumerated App-B/C/D residue (model-side bound `9.483×10^15`, collapse paragraph, emergent diffusion-law cells, first-passage confound paragraph, `A_75` law, delete-time churn `2.836`, white-box address-depth AUC `0.91409`).

---

## 7 — Open items for the Hub / Head (nothing here was guessed)

**A. ⚠⚠ A THIRD TABLE IS OVERFULL, IT IS PRE-EXISTING, AND IT WAS NOT TOUCHED.** A rebuild of the **untouched** PRE file shows **three** overfull `\hbox`es: `192–200` (177.50pt, App A — **fixed**), `226–235` (152.59pt, App B — **fixed**), and **`271–280` (49.16pt) — the App-D laundering-control table**, which is *not* one of the two the task enumerates. It is now at post-lines `371–380`, page 16, still 49.16pt (~12%) into the right margin. **It was left alone because the task forbids anything not enumerated, "however beneficial it looks."** The acceptance criterion *"0 overfull table boxes"* is therefore met **for the two enumerated tables and not file-wide**. A **content-free** one-line fix (column spec only — no caption change, no values, no `\small`) is pre-computed and staged, unapplied, at `.claude/scratch/pj-minimal-v5/appD-overfull.patch`. **Head/Hub call.**

**B.** Item 10's *"l.116"* has no `3.77×`; applied at the real site (see §3).
**C.** Item 11's *"first 107.77× site"*: applied at the first **body** site (Sec. 3.2), not the abstract.
**D.** The score sentence now appears **twice** — the new Limitations bullet (item 9) and the App-E row *"A task-level payoff is claimed"* (Part 3a). **This matches the base, which also carries it twice**, but the Head may want the *"Stated once:"* prefix on only one of them.
**E.** Author-token sweep: `Jawahar|Pierini` = **2** occurrences — l.158 (bibliography, item 5) and **l.56 (body)**, the Charter-mandated CLU continuity sentence *"introduced as CHLU by Jawahar \& Pierini (2026)"*. **"Bibliography-only" is therefore not literally achievable** while the naming rule stands; reported rather than resolved.
**F.** Restoring the full J&P entry (item 5) puts real author names into a build whose `\author{}` is blank and whose PDF metadata is scrubbed. Advisor-ratified (Add. 60) and third-person throughout, but it is a **de-anonymization surface** and the Head owns it.
**G.** **Page split reported, not optimized** (§8): 11 → 18 pages.

---

## 8 — Page split (reported, not optimized)

`pdftotext` section walk: **main text pp. 1–6** (Introduction p1 · Related Work p2 · Results p3 · Limitations p5) · **References pp. 7–8** · **App A pp. 8–10** · **App B pp. 10–13** · **App C pp. 13–14** · **App D pp. 14–16** · **App E pp. 16–18**. Total **18 pages** (PRE: 11). The growth is entirely figures (0 → 11 real images) and appendix tables/rows; **no main-text section gained a page from prose** — main text is 6 pages before and after.

---

## 9 — Sweeps (all positive-controlled; instrument verified LIVE)

| sweep | result | positive control |
|---|---|---|
| **A.4 never-quote zero list** (35 patterns incl. **`13.9`**, `≈14×`, `outperform`, `beats`, `wins`, `SOTA`, `.claude`, `PALM`, `Morse`, `Moser`, `cryptographic unlearning privacy`, …) | **0 hits on every pattern** | combined pattern `107.77\|Blelloch\|certified\|unlearning\|recency\|encoder\|lifecycle\|ZERO` → **20 matching lines** ⇒ instrument LIVE |
| **`certified` per-occurrence** | **3 occurrences, 0 affirmative.** l.80 = literature-description form (now *correctly* stated); **l.124 = the DENIAL, restored** (*"we claim no certified $(\varepsilon,\delta)$ unlearning"*); l.156 = the Guo bibliography title | same |
| **honest-scope sentence** | **×1** (`store-level guarantee only` = 1) | same |
| **`§` / `\S`** | `\S` → **0**, literal `§` → **0** | pre-pass count was 4 |
| **figures** | 11 `\includegraphics`, **0** `framebox`/`Placeholder` | — |
| **negatives rows** | **20** (11 + 9) | — |

---

## 10 — Reproduction

```
/usr/bin/python3 .claude/scratch/pj-minimal-v5/minimal.py     # 30 asserted edits
cd .claude/NIPSsubmission/v5-palm
/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex   # ×2
/usr/bin/python3 .claude/scratch/pj-minimal-v5/numcheck.py
/usr/bin/python3 .claude/scratch/pj-minimal-v5/numcheck2.py
```
`pj_sub.PRE.tex` (md5 `6c1902f74ee9611d718cc65b9fd1a031`) is preserved in scratch; `pass.diff` holds the full 14-hunk classified diff (42 lines removed, 153 added).
