# BUILD NOTE — `v5-colleague-edits` (the Head-approved edit set from the colleague's review)

**Agent:** `paper-writer` · **Date:** 2026-08-27
**Object:** `.claude/NIPSsubmission/v5-palm/pj_sub.tex`
**Boot md5:** `c63a57fc910663dfa1e644b9b349ce6f` — **verified on disk before any edit** ✅
**Boot mtime:** `1787760290` (Aug 26 18:04:50 2026) · **mtime re-checked immediately before applying: unchanged** ✅
**Final md5:** `ca56fef3b86a2d5d17314f84e130df3a`
**Decision sheet of record:** `.claude/outputs/v5-colleague-edit-list.md` (all rulings executed as written)
**Scratch build dir:** `/tmp/v5edit/` (all iteration happened there; only the verified result was applied)

---

## 0. Scope compliance, up front

| | |
|---|---|
| Items approved | **13** (T.1 · T.2 · T.4 · T.5 · R.0 · R.1 · R.2 · R.3 · R.3b · R.4 · R.5 · P.1 · P.2 · X) |
| Items executed | **all of the above** |
| Item DECLINED by the Head | **T.3 (`atoms` disambiguation)** — ⛔ **not executed.** `atoms` at l.52 is byte-unchanged and `superposed` is byte-unchanged at l.52 **and** l.107 (verified by grep: both strings survive verbatim) |
| Hunks attributable to no approved item | **ZERO** |
| Files written | `pj_sub.tex` (the only `.tex` touched) + this build note |
| Files verified byte-untouched | `figs/**`, `neurips_2025_ml4ps.sty`, `submission.tex`, `submission.pdf/.aux/.log/.out`, `pj_sub_preEdit_stable.tex`, `pj_sub_buildcopy.*`, `palm_pj_sub.pdf`, the three prior BUILD-NOTEs, and **all of `~/Desktop/V5_PALM_Submission/**`** (incl. `refs.bib`, which lives only there) |
| Live `pj_sub.pdf`/`.aux`/`.log`/`.out` | **NOT overwritten** (task §7: `pj_sub.tex` is the only writable file). ⚠ The live `pj_sub.pdf` is therefore now **stale** relative to `pj_sub.tex`; the verified rebuild is at `/tmp/v5edit/pj_sub.pdf`. The Advisor refreshes the build copy at acceptance. |

**Defects noticed but NOT touched** (out of the approved set → findings list only, §8): five items, incl. one that needs a Head/Advisor ruling (**§8.1**).

---

## 1. Diff contract — every changed line, with its item ID

The complete diff is **18 changed lines + 1 insertion block**. Changed pre-pass lines: `35, 40, 42, 44, 57, 59, 74, 76, 78, 80, 89, 93, 95, 97, 107, 119, 120, 123` and an insert after `134` (`\appendix`). Nothing else in the file differs.

| pre-pass line | item ID(s) | what changed |
|---|---|---|
| 35 | `X` | `explicitly` deleted (abstract) |
| 40 | `X`, `X` | `distinct` deleted · `intrinsically` deleted |
| 42 | **`T.1`** | `after the fact` → `even after the entry is nominally deleted` |
| 44 | **`T.2`** | framing sentence replaced verbatim with the Head's wording |
| 57 | `X` | `precisely` deleted |
| 59 | **`T.4`** | Nomenclature bullet gains its CHLU provenance |
| 74 | **`T.5`** + `X` | §2.1 opens with its thesis; `fundamentally` deleted |
| 76 | `X` | `distinct` deleted |
| 78 | **`R.1`** + **`R.2`** + `X`×3 | paragraph inverted (result first); quantity named; `distinct`/`precisely`/`singular` deleted |
| 80 | **`R.2`** + **`R.3`** + **`R.3b`** + `X` | mode-distinguishing clause; numbers → table; intuition line; `distinct` deleted |
| 89 | **`R.4`** | two argmins → table, pointer added |
| 93 | **`R.5`** + `X` | latch numbers → table; `physically` deleted |
| 95 | **`R.5`** + `X` | `1.0068±0.0219` / `25 cells` → table; `successfully` deleted |
| 97 | **`P.1`** + **`R.5`** + **`R.3b`** + `X`×2 | brake/refrigerator sentence replaced; vault numbers → table; intuition line; `explicitly`/`remarkable` deleted |
| 107 | **`P.2`** + `X` | one clause on what "pure function of its live set" buys; `explicitly` deleted |
| 119 | `X` | `explicitly` deleted |
| 120 | `X` | `intrinsically` deleted |
| 123 | `X` | `explicitly` deleted |
| after 134 | **`R.0`** | new first appendix section + `Table~\ref{tab:numbers}` |

### 1a. Before → after, verbatim, for T.1 / T.2 / T.4 / P.1 / P.2

**T.1 (l.42)** — surrounding words byte-identical, `\citep` byte-identical ✅
- before: `…which leaves measurable residue within the network architecture after the fact \citep{chakraborttii_ghost_2026,wang_memleak_2026}.`
- after:  `…which leaves measurable residue within the network architecture even after the entry is nominally deleted \citep{chakraborttii_ghost_2026,wang_memleak_2026}.`

**T.2 (l.44)** — first sentence only; the following sentence (`By modeling the store as a physical system, …`) is byte-identical ✅
- before: `In this work, we propose and analyze a memory framework where forgetting is an intrinsic dynamical property of the store itself, rather than an external bookkeeping rule.`
- after:  `In this work, we propose and analyze a memory framework where forgetting is a prescribed dynamical property of the store, governed by parameters that can be set, targeted, and analyzed in closed form, rather than an emergent side effect or an external bookkeeping rule.`
- ⛔ **Guard honoured:** the word **"interpretability"** (and any synonym asserting the store is interpretable) does **not** appear anywhere in the file. Verified: `grep -ci "interpret" pj_sub.tex` → **0** (positive control: `grep -c "prescribed" pj_sub.tex` → 1).

**T.4 (l.59)** — the block was **NOT relocated**; the four sub-items are byte-identical ✅
- before: `    \item \textbf{Nomenclature:}`
- after:  `    \item \textbf{Nomenclature borrowing from and building on CHLU~\cite{jawahar_chlu_2026}:}`
- key check: `jawahar_chlu_2026` **is** the only CHLU key in `refs.bib` (line 339) and **resolves** — 0 undefined citations, and the entry prints in the bibliography.
- ⚠ **See §8.1 — a rendering decision the Advisor/Head must make.**

**P.1 (l.97)** — colleague's sentence adopted verbatim; the number moved to the table under R.5 ✅
- before: `A localized spatial hole within this field functions concurrently as a brake and a refrigerator ($T_{\rm local}=1.26\times10^{-4}$ versus $10^{-3}$ externally).`
- after:  `A localized spatial hole within this field acts simultaneously as a brake, increasing dissipation, and a refrigerator, reducing the local effective temperature.`

**P.2 (l.107)** — one clause; the deletion-conditions passage at l.109 is **byte-identical** (verified by diff: l.109 does not appear in the diff at all) ✅
- before: `We evaluate whether the store's intrinsic physical state can be explicitly reduced to a pure function of its live set alone.`
- after:  `We evaluate whether the store's intrinsic physical state can be reduced to a pure function of its live set alone --- that is, whether the layout depends only on which items are currently stored, not on the order or history of the writes that stored them.`

### 1b. T.5 + R.1 + R.2 (the one rewrite, shown in full)

**l.74 (T.5 + X):**
- before: `To understand the macro-dynamics of memory decay, we first analyze damping as a retention dial. We demonstrate that retention half-life is non-monotone in relation to friction. The turning point is fundamentally fixed by the stored direction's spectral mass, meaning the optimal setting can be analytically predicted rather than empirically tuned.`
- after:  `Friction is not a monotone retention knob: retention half-life is non-monotone in relation to friction, so the dial has an optimum rather than a direction. The turning point is fixed by the stored direction's spectral mass, meaning the optimal setting can be analytically predicted rather than empirically tuned. To understand the macro-dynamics of memory decay, we analyze damping as that retention dial.`
- ⚠ **Instruction conflict, resolved and reported.** T.5 names `l.72–74` as its object; R.2's consistency note says *"⛔ Do not edit l.35 or l.74."* I read R.2's prohibition as **scoped to R.2's subject — the naming of the quantity** — because T.5 is separately approved `yes` and cannot be executed without touching l.74. **Mitigation: the phrase `retention half-life` at l.74 is preserved verbatim**, so R.2's consistency purpose is fully served, and **l.35 was not touched by R.2** (its only change is the `X` deletion of `explicitly`, which is on X's own approved line list). If the Advisor intended l.74 to be frozen outright, this hunk is the one to revert; nothing else depends on it.

**l.78 (R.1 + R.2 + X):** the paragraph's final sentence (*"…the latch, the overdamped register, and the underdamped working memory…"*) is **moved to the front** — that is the whole of R.1's inversion — and the trained-`SO(2)` mechanics follow. `half-life` → `retention half-life` (R.2). ⛔ **l.78's own numbers (`−1.006`, `+1.23`–`+1.27`, `μ²≈10⁻¹⁵`, `5 validation seeds`) were NOT moved** — R.3 enumerates only l.80's numbers, and footprint discipline says the un-enumerated numbers stay.
- after: `The latch, the overdamped register, and the underdamped working memory are three specific operational regimes of a unified curve evaluated at two disparate values of $\mu$. On a fully trained, designed $SO(2)$ vacuum, the massive radial mode's retention half-life $n_{1/2}(\gamma)$ forms a V-curve, minimized at $\gamma_{\rm crit}=2\varepsilon\mu_{\rm rad}$ (Fig.~\ref{fig:massiveflat}, App.~\ref{app:budget}). The overdamped branch tracks the $\mu^{-2}$ law until $\varepsilon\mu\approx\gamma/2$, after which it saturates at a mass-independent floor. This yields log-slopes of $-1.006$ and between $+1.23$ and $+1.27$ across the 5 validation seeds (See App.~\ref{app:budget} for results). The flat coset (characterized by a Hessian $\mu^2\approx10^{-15}$) maps to the exact same curve at the limit $\mu\to0$, where $\gamma_{\rm crit}\to0$.`

**R.2's distinguishing clause (l.80)** — built **only** from material the paper already states; ⛔ no physics composed, so the STOP clause never fired:
> `--- read on the near-flat stored direction that holds a written value, not on the stiff radial mode of the previous paragraph, though one law governs both (Fig.~\ref{fig:collapse}) ---`

Sources, line by line: *"near-flat coset direction"* = the Nomenclature sub-item at **l.63** (`Stored direction: A written value occupies a near-flat coset direction`); the radial mode's stiffness and the fact that it is not where a value is stored = **l.78** (the massive radial mode) + **Fig. 1's caption at l.86** (`$\mu^2_{\rm rad}=0.670$--$1.348$` vs `$\mu^2_{\rm soft}=2.0$--$5.4\times10^{-2}$`); *"one law governs both"* = **l.86** (*"Eight measured curves collapse onto one"*) + **l.80** (*"Spanning both architecture families, this curve holds…"*). ⛔ **The two `μ²` numbers were deliberately NOT lifted into the body** — they live in Fig. 1's caption and in the table; putting them in the body would have been new numbers in a paragraph whose job this pass is to *empty* of numbers.

### 1c. R.3b — the descriptive intuition lines (2 of the permitted 3)

⛔ Zero numbers · zero capability, payoff, comparison-to-other-work or forward claim · **no programme/vision content** (Head: the vision belongs to the ICLR long).

| block | line added | why it is descriptive only |
|---|---|---|
| **R.3** (l.80) | `Put plainly: the curve a designed unit follows is the curve a learned one follows, read at a different spectral mass.` | restates the paragraph's own transfer result; no payoff, no capability, no forward claim |
| **R.5** (l.97) | `Put plainly: the vault factor is how much longer a written value is retained inside the hole than outside it.` | states what the quantity **is** (its definition), not what it enables |
| **R.4** (l.89) | **none — deliberately** | the block already ends in plain words (*"the gap between instruments dictates the level, not the fundamental law"*). "Fewer is fine if a block does not need one." |

---

## 2. R.0 — the new table, and its ancestor ledger

**Placement:** a new appendix section **`\section{Values Quoted in the Main Text}` (`app:numbers`), inserted immediately after `\appendix`** ⇒ `Table~\ref{tab:numbers}` is the **first table of the appendix** (it prints as **Table 1**; the pre-existing appendix tables renumber 1–5 → 2–6). ⚠ **Appendix letters all shift by one** (Extended Related Work A→B, budget B→C, emergent C→D, vault D→E, deletion E→F, derivations F→G). Every in-file pointer is a `\ref` (17 of them) — **grep confirms zero hard-coded appendix letters**, and the build reports **0 undefined references**.

**Legend, as required:** the caption *and* the section's lead paragraph both define the taxonomy — **`verification` = designed testbed** (an architecturally designed `SO(2)` potential), **`evidence` = learned system** (an MLP `V_θ` trained on symmetric data). ⭐ This closes the standing finding that *verification*/*evidence* are used in main-text figure captions and were defined nowhere.

### 2a. Ancestor ledger — every cell traces to a pre-pass line

| # | row | value | ancestor line in the pre-pass `pj_sub.tex` |
|---|---|---|---|
| 1 | argmin, one-step Jacobian | `0.902±0.003×γ_crit` | **L80** |
| 2 | log-slope, overdamped | `−1.0020±0.0003` | **L80** |
| 3 | log-slope, underdamped | `+1.116±0.011` | **L80** |
| 4 | μ² range spanned | `[1.7×10⁻¹², 7×10⁻²]` | **L80** |
| 4r | probe-resolution rider text | — | **L80** (verbatim sense: *"the ring-profile probe's resolution … machine zero rather than a spectral mass"*) |
| 5 | argmin, rollout envelope rate (I-R3) | `0.9001±0.0052×γ_crit` | **L89** (also L209ff table, I-R3 row) |
| 6 | argmin, one-step Jacobian (I-J) | `0.9032±0.0027×γ_crit` | **L89** (also L209ff table, I-J row) |
| 6r | *"the instrument gap is a level, not a rate"* | — | **L89** + **L236** (App. D instrument-gap caption) |
| 7 | coset drift | `≤4.9×10⁻¹² rad / 200k steps`, `γ∈[0.002,0.5]` | **L93** (also **L154**, App. C latch paragraph) |
| 8 | `D̂_θ/D_θ^pred` | `1.0068±0.0219`, `25 (γ,T)` cells | **L95** (also **L150**, App. C) |
| 8r | `seed 44`, `Δ=0.5 rad` | — | **L150** (*"utilizing seed 44"*), **L148** (*"Δ=0.5 rad"*) |
| 9 | `T_local` inside vs outside | `1.26×10⁻⁴` vs `10⁻³` | **L97** (also **L101** fig-caption, **L284** App. E) |
| 9r | `γ=0.05`, `γ_φ=0.5`, absorb-only | — | **L282** (App. E) |
| 10 | vault factor, `D̂_θ` estimator | `107.77±4.78×` | **L97** (also **L101**, **L284**) |
| 10r | `3 seeds`, `T=10⁻³`, `Δ=0.5 rad` | — | **L101** (fig:vault caption, verbatim string) |
| 11 | scalar-friction control | `13.28±0.12×` | **L97** (also **L284**) |
| 12 | raw first-passage ratio | `86.97±2.94×`, `ℓ_θ/Δ=0.079` | **L97** (also **L101**) |
| cap | `dim 4, hidden 64, ε=0.05, single-CPU` | — | **L76** and **L86** (fig:collapse caption) |
| cap | `langevin_noise="fdt"`, Newtonian kinetic mode | — | **L86**, **L148** |

⛔ **ZERO new numbers.** Nothing was rounded, re-derived, combined or smoothed; every printed digit is copied from its ancestor.

### 2b. Numeric two-way check (machine-run, both lists printed)

Run: token-level numeric multiset diff of the main text (lines 27–133) before vs after, then membership tests. Script: `/tmp/v5edit/numcheck.py`.

**(i) Numbers that LEFT the main text (37 tokens):**
`0.0003 · 0.002 · 0.0027 · 0.003 · 0.0052 · 0.011 · 0.0219 · 0.12 · 0.5 · 0.9001 · 0.902 · 0.9032 · 1.0020 · 1.0068 · 1.116 · 1.26 · 1.7 · 10 ×5 · 107.77 ×2 · 12 ×2 · 13.28 · 2 · 2.94 · 200 · 25 · 3 · 4 · 4.78 · 4.9 · 7 · 86.97`

**(ii) ORPHAN LIST (left main text, absent from the new table): `[]` — EMPTY ✅**

**(iii) Table numbers with no ancestor in the pre-pass file: `[]` — EMPTY ✅**
(the only non-ancestor numerals in the block are the `p{0.165\linewidth}`-style **column widths** and `\tabcolsep`, which are typesetting, not content — excluded explicitly and listed here for completeness.)

**(iv) Numbers ADDED to the main text: exactly one token, `2026`** — the year inside the new `\cite{jawahar_chlu_2026}` key (T.4). No numeric claim was added anywhere.

---

## 3. The R.0a test — every claim whose number moved, and the rider that stayed with it

⛔ *A claim left in main text without its rider is a blocking failure.* **None found.** Item by item:

| # | main-text claim (line) | number(s) moved | **qualification that REMAINS in main text, beside the claim** |
|---|---|---|---|
| 1 | emergent V-curve reproduces the law (l.80) | argmin, both log-slopes | **designed-vs-emergent distinction intact and strengthened**: *"transfers directly to **emergent** units"*, *"three **emergent MLP** checkpoints"*, *"a learned potential"* (table), *"Spanning **both architecture families**"*; plus *"across all 3 seeds"*, `$T=0$`, and the new R.2 clause naming which mode is measured |
| 2 | the μ² span (l.80) | `[1.7×10⁻¹², 7×10⁻²]` | ⛔ **probe-resolution rider kept verbatim in the body**: *"The low endpoint of that span is the ring-profile probe's resolution on a checkpoint whose Hessian μ² is machine zero rather than a spectral mass, so eleven orders is one curve on one instrument (probe-floor tick marked in Fig. …)"* — and **repeated in the table's Scope column in bold** |
| 3 | shape reproduces on a 2nd instrument (l.89) | both argmins | *"on this **second instrument**"*, the `0.35%` agreement (**kept in body** — not enumerated to move), and *"the gap between instruments dictates the **level, not the fundamental law**"* |
| 4 | `T=0` latch (l.93) | drift bound + `γ∈[0.002,0.5]` | *"a **designed** coset"*, `T=0` stated in-sentence, and the C-5 scope qualifier rewritten as **"across every friction setting we evaluated"** with the numeric range one pointer away in the same parenthesis. ⚠ **This is the one place where the rider itself was a number.** R.0a forbids splitting a number from its rider — so **both moved together** and the body kept a non-numeric scope qualifier rather than an unqualified *"any friction"*. Flagged for the Advisor: if you prefer the explicit range in the body, restore `γ∈[0.002,0.5]` to l.93 (it is then duplicated, not orphaned) |
| 5 | diffusion law verified (l.95) | `1.0068±0.0219`, `25 cells` | *"which we verified over a grid of `(γ,T)` cells **on the designed testbed**"* — the arm stays in-sentence; the single-checkpoint (`seed 44`) fine print is in the table's Scope column |
| 6 | brake + refrigerator (l.97) | `T_local` pair | P.1's replacement sentence is qualitative and needs no numeric rider; *"absorb-only"* stays in the preceding sentence, and the table row carries `γ=0.05, γ_φ=0.5, absorb-only field` |
| 7 | the vault (l.97) | `107.77±4.78×`, `13.28±0.12×`, `86.97±2.94×` | ⛔ **estimator name travels**: *"read from the **$\hat D_\theta$ estimator**"* is in the body sentence; *"quoted against a **uniform-scalar-friction control at matched γ_eff**"* is in the body; *"the direct **first-passage** reading is boundary-layer biased on the outside arm (`ℓ_θ/Δ=0.079`) and **is therefore not the quoted vault**"* is in the body, with `ℓ_θ/Δ=0.079` **kept as a number in the body**; *"on **designed** architecture"* stays. The table then repeats the estimator rider **in bold** on the `107.77` row and the "not the quoted vault" rider on the `86.97` row |
| 8 | deletion conditions + recency exclusion + encoder scope (l.109) | — | ⛔ **untouched, as ordered** — l.109 does not appear in the diff |

### 3a. Standalone-ness: does it improve? (the pass's acceptance test)

**Argued improvement, with the mechanism named in each case:**
1. **P.1** replaces two unexplained metaphors with their physical content (*increasing dissipation* / *reducing the local effective temperature*) — a reader who does not know the physics can now read the vault claim.
2. **P.2** converts *"a pure function of its live set alone"* into an operational statement (*layout depends on which items are stored, not the write order/history*) — the deletion claim becomes checkable by an ML reader.
3. **R.2's clause** fixes a real ambiguity: two different half-lives were quoted in adjacent paragraphs with no statement of which mode each belonged to. A reader can now tell the stored (near-flat coset) direction from the stiff radial one, and that one law covers both.
4. **T.5 + R.1** put the claim before the apparatus in the section that carries the headline figure.
5. **The table** gives every moved number a **single addressable home with its arm and its rider** — the arm taxonomy (`verification`/`evidence`) is now **defined in the paper for the first time**, which retroactively makes six existing figure captions readable.
6. **Every rider stayed in the body** (§3 table). No claim became less qualified.

**Honest counterweights (both listed so the Advisor can weigh them):**
- **(a)** l.93's explicit `γ∈[0.002,0.5]` became *"every friction setting we evaluated"* + a table pointer (item 4 above). Strictly, the body is one hop further from its scope numbers.
- **(b)** Reading the exact magnitudes now requires the appendix. This is intrinsic to the Head's approval of R.3/R.4/R.5 and is the trade the colleague asked for.

---

## 4. Group X ledger — all 26 instances

Sweep scope confirmed against the Advisor's tally: `explicitly`×5 (l.35, 97, 107, 119, 123) · `distinct`×4 (l.40, 76, 78, 80) · `intrinsically`×4 (l.40, 109, 120, 121) · `singular`×4 (l.52, 78, 93, 107) · `strictly`×3 (l.67, 118, 121) · `precisely`×2 (l.57, 78) · `fundamentally`, `physically`, `successfully`, `remarkable` ×1 (l.74, 93, 95, 97) = **26**. ⛔ Figure captions excluded. ⛔ Nouns, verbs, numbers, labels and citations untouched; no sentence re-flowed **by the sweep** (where a line was re-flowed, the re-flow is licensed by R.1/R.3/R.5/P.1 and is labelled as such above).

### 4a. Deleted (18) — deletion changes no meaning

| line | word | after |
|---|---|---|
| 35 | `explicitly` | `governed by defined operational conditions` |
| 40 | `distinct` | `operate under retention/deletion mechanisms:` — the colon-list that follows carries the heterogeneity ⚠ *the closest call in the sweep; flagged, not hidden* |
| 40 | `intrinsically` | `one cannot read the half-life of a stored value off a trained LSTM` |
| 57 | `precisely` | `a novel, designed store` |
| 74 | `fundamentally` | `The turning point is fixed by the stored direction's spectral mass` |
| 76 | `distinct` | `across 5 seeds` |
| 78 | `distinct` | `forms a V-curve` |
| 78 | `precisely` | `tracks the $\mu^{-2}$ law until $\varepsilon\mu\approx\gamma/2$` — ⭐ deletion **removes an ambiguity**: "precisely" could attach to the tracking's fidelity or to the crossover's location; the fidelity is stated quantitatively by the log-slopes in the next sentence |
| 78 | `singular` | `of a unified curve` |
| 80 | `distinct` | `three emergent MLP checkpoints` |
| 93 | `physically` | `introducing friction decelerates this decay` |
| 95 | `successfully` | `which we verified over a grid of $(\gamma,T)$ cells` |
| 97 | `explicitly` | `a position-gated friction field that is absorb-only` |
| 97 | **`remarkable`** | `generates a retention vault factor of $(\gamma_{\rm eff}/\gamma)^2$` — ⛔ **the standing forbidden-class word is gone** |
| 107 | `explicitly` | `can be reduced to a pure function of its live set alone` |
| 119 | `explicitly` | `must be designed into the architecture` (the limitation is unchanged) |
| 120 | `intrinsically` | `cannot distinguish between designed and emergent units.` (the limitation is unchanged, and reads stronger) |
| 123 | `explicitly` | `construct localized temperature fields` |

### 4b. RETAINED (8) — with the load-bearing reason for each

| line | word | reason it was NOT deleted |
|---|---|---|
| 52 | `singular` | ⛔ **deletion is ungrammatical** (`a energy function`) and the word here means *a single* — replacing it is a **noun/adjective substitution, which this sweep is not permitted to make**. ⚠ *Also adjacent to `superposed`, which the DECLINED T.3 protects.* → **finding §8.2** |
| 67 | `strictly` | Advisor-flagged: `$T>0$ **strictly** requires FDT-consistent noise` — **the requirement is real** (an FDT-inconsistent noise scale breaks the thermodynamics) |
| 93 | `singular` | same as l.52 — `a singular entry` means *a single entry*; deletion is ungrammatical → **finding §8.2** |
| 107 | `singular` | same as l.52 → **finding §8.2** |
| 109 | `intrinsically` | **load-bearing C-6 fine print**: *"recency-based eviction remains **intrinsically** history-dependent"* marks the breakage as structural (hence excluded) rather than an implementation artifact (hence fixable). ⛔ **Additionally, l.109 is the protected deletion-conditions passage** (R.0a and P.2 both order it untouched) — so the conservative reading wins twice |
| 118 | `strictly` | Advisor-flagged: **C-5 scale qualifier** (`strictly constrained to a dimension of 4`) |
| 121 | `strictly` | Advisor-flagged: `applies **strictly** at the isolated store-level` — **narrows scope** |
| 121 | `intrinsically` | same reason as l.109: *"recency-based eviction **intrinsically** breaks latent history independence"* — marks the breakage as structural, which is why the paper excludes rather than repairs it. Kept for consistency with l.109 |

---

## 5. Build

Toolchain: `/Library/TeX/texbin/{pdflatex,bibtex}` (not on `PATH`; invoked by absolute path). Built in `/tmp/v5edit/` with `neurips_2026.sty`, `refs.bib` and `figs/` copied from `~/Desktop/V5_PALM_Submission/` (⛔ that folder was read-only in this pass).
Sequence: `pdflatex → bibtex → pdflatex → pdflatex`.

| metric | pre-pass baseline | **after** |
|---|---|---|
| errors | 0 | **0** ✅ |
| undefined citations | 0 | **0** ✅ |
| undefined references | 0 | **0** ✅ |
| overfull boxes | 0 | **0** ✅ |
| total pages | 18 | **19** (the new appendix section + table) |

⚠ **Table typesetting took three iterations to reach 0 overfull boxes** and the fix is worth recording: **inside a `p{}` column, `\linewidth` is the column width**, so `\multirow{n}{0.145\linewidth}{…}` builds a `parbox` ~7× too narrow and floods the log. The shipped form is `\multirow{n}{\linewidth}{…}` with `\setlength{\tabcolsep}{4pt}` local to the table (column widths then sum to 0.82 against the 0.879 available). One label (`Cross-instrument check`) dropped its `Sec.~\ref` pointer and keeps only `App.~\ref{app:emergent}`, because the 2-row span could not hold five lines of label without an overfull `\vbox`.

### 5a. Main-text page count — measured, not assumed

**Method (reproducible, `/tmp/v5edit/pagefrac.py`):** full pages before the overflow page, plus the fraction of page 5's text block consumed above the `References` heading, using `pdftotext -bbox` word boxes and page 4's body box as the reference block.

| | before | after | Δ |
|---|---|---|---|
| main text | **4.28 pp** | **4.32 pp** | **+0.04 pp** (≈ +26 pt ≈ 2.4 lines) |

⚠ The task's stated baseline is **4.30 pp**; my instrument reads the same pre-pass PDF as **4.28 pp**. The 0.02 gap is a **metric-definition difference** (where the block is deemed to end), not a content difference — the same instrument is used for both columns above, so the **Δ is sound**.

⛔ **Nothing was cut to hit a number** (task §6.7). **The net effect is positive, i.e. the main text grew**, and P.1/P.2 were correctly predicted to be the cause: Group R removed ~37 numeric tokens but the replacements (P.1's explanatory sentence, P.2's clause, R.2's distinguishing clause, two R.3b lines, four `Table~\ref` pointers, and the estimator/control/first-passage riders that R.0a requires to stay in the body) together exceed them. **The paper remains over the 4-pp limit and is now +0.04 pp further over.** Pruning is explicitly not this pass's job; the Advisor now has a measured number to scope one with.

---

## 6. Integrity checks run

- ✅ Boot `md5` = `c63a57fc910663dfa1e644b9b349ce6f` verified **before** the first edit.
- ✅ `mtime` checked at boot **and** immediately before applying — `1787760290` both times, unmoved (no concurrent Head edit).
- ✅ All iteration in `/tmp/v5edit/`; the live folder was read-only until the single verified `cp`.
- ✅ Every replacement applied by an **exact-string, count-must-equal-1** script (`/tmp/v5edit/apply_edits.py`) — a fuzzy or duplicated match aborts the pass.
- ✅ Diff inspected line by line; changed-line list matches the approved-item list exactly.
- ✅ **Grep hazard respected:** every negative sweep carries a positive control — `interpret` → 0 (control `prescribed` → 1); hard-coded appendix letters → 0 (control: `App.~\ref` → 17); `after the fact` → 0 (control: `even after the entry` → 1).
- ✅ T.3 DECLINED and l.109 protected, verified: **l.52 is byte-identical** (`md5` of the line, before = after = `634b60d8…`) so `atoms` and its `superposed` clause are untouched; **l.109 is byte-identical** (`e4da6554…`); l.107 changed only by P.2+X and its `superposed into a singular energy function` clause survives verbatim (`grep -c superposed` = **2** before and after).
- ✅ Folder-wide `md5` sweep: only `pj_sub.tex` changed; `figs/**` (11 files), `submission.tex`, `neurips_2025_ml4ps.sty`, `pj_sub_preEdit_stable.tex` and all `pj_sub_buildcopy.*` byte-identical to their pre-pass hashes.

---

## 7. What the Advisor must do next

1. **Rule on §8.1** (`\cite` vs `\citep` in the T.4 bullet) — one character, and it is the only thing in this pass that is visibly wrong on the page.
2. Refresh the build copy `~/Desktop/V5_PALM_Submission/paper.tex` from `pj_sub.tex` (⛔ this pass did **not** touch it) and re-run the build there.
3. Decide whether the stale live `pj_sub.pdf` should be refreshed now or at acceptance.
4. Note the **appendix letter shift** (A→B, …, F→G) before quoting any appendix letter in correspondence.
5. Weigh **§3a(a)** (l.93's `γ` range) — one-line restore if you want the explicit range back in the body.

---

## 8. Findings — defects noticed OUTSIDE the approved set. ⛔ NONE was touched.

### 8.1 ⚠ **DECISION NEEDED — the T.4 citation renders as running text**
The Head's wording was executed **verbatim**, including `\cite`. Under `natbib`+`plainnat`, **`\cite` is textual** (`\citet`-like), so the bullet prints as:
> **Nomenclature borrowing from and building on CHLU Jawahar and Pierini [2026]:**

i.e. `CHLU Jawahar and Pierini [2026]` with no parentheses. The intended reading is almost certainly `CHLU [Jawahar and Pierini, 2026]`, which is **`\citep`** — a one-character change. ⛔ **I did not make it:** the task supplied the literal LaTeX as the Head's wording, and silently switching the citation command is exactly the kind of unenumerated "improvement" the scope rule forbids. **Patch, ready to apply on a ruling:** `\cite{jawahar_chlu_2026}` → `\citep{jawahar_chlu_2026}` on l.59 only.

### 8.2 `singular` is used to mean *"a single"* in three places
l.52 (`a singular energy function`), l.93 (`a singular entry`), l.107 (`a singular energy function`). Group X is a **deletion-only** sweep, and deletion here is ungrammatical, so all three were retained (§4b). The correct repair is a word substitution (`singular` → `single`), which is **outside this pass's licence** — and at l.52/l.107 it sits next to `superposed`, which the DECLINED T.3 protects. **Recommend the Advisor scope it as a separate three-token item.**

### 8.3 ⚠ Two pre-existing numeric mismatches, now adjacent in one table
The pass moved l.80's argmin (`0.902±0.003`) and l.89's Jacobian argmin (`0.9032±0.0027`) into the same table. These are **the same quantity on the same instrument** (emergent argmin, one-step Jacobian, 3 seeds) at two different precisions — and `0.9032` does **not** round to `0.902`. The same holds for the underdamped slope: body l.80 says `+1.116±0.011`, App. D's I-J row says `+1.1182±0.0107`. Both discrepancies **pre-date this pass** (they are in the boot file), and ⛔ **I did not adjust, round or reconcile either** — the rule is that a number I cannot source is a missing-experiment note, not an improvisation. ⚠ **But the table now places them ~6 rows apart, where a referee can see both at once.** Rows are labelled by instrument and result-block so the two readings are at least distinguishable. **This needs an owner:** either (a) one of the two is stale and should be retired to the other's value, or (b) they are different fits and the body should say which. **Recommend the Advisor route it to whoever owns the emergent-arm numbers.**

### 8.4 The abstract invokes the vault factor without its estimator
l.35 reads *"demonstrating a `107.77±4.78×` retention factor on designed architectures"* — the arm rider is present, but the **`D̂_θ` estimator name is not**, while R.0a's rule is that the estimator *"travels with the vault factor **wherever it is invoked**."* The abstract's number was **not** in R.5's move list, so **l.35 was left alone** apart from the approved `X` deletion. ⛔ Not a blocking failure under deliverable 4 (no number moved out of the abstract) — but it is the one remaining site in the paper where the vault factor appears unestimated. **Recommend a scoped one-clause item.**

### 8.5 Two small typographic residues
(a) l.42 ends `…presented in App.~\ref{app:related}` with **no full stop**. (b) The pre-pass l.80 carried a **stray doubled parenthesis**, `(Figure~\ref{fig:lambdacoset}))`. ⛔ (a) was not touched. For (b): the second `)` closed nothing, and the R.3 rewrite of that sentence necessarily re-flowed the parenthetical, so the shipped line reads `(Figure~\ref{fig:lambdacoset})` with a single paren. **This is a consequence of an approved hunk, not an independent edit** — recorded here so the diff reader is not surprised by it.

---

## 9. DIAL DECLARATION (echoed)
**Dials touched: NONE.** No experiment, no config, no registry, no charter. One `.tex` file edited within an enumerated set; one build note; one report.
