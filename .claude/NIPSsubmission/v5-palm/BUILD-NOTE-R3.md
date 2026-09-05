# `v5-palm/pj_sub.tex` — BUILD NOTE R3 (the bounded restoration pass)

**Deliverable #1 of `pj-restore-v5`** (Advisor charter Addendum 59; Head ruling 2026-08-22: *"add back the missing information in `pj_sub` without rewriting the rest… strictly follow the tone and academic brevity plus accuracy that currently exists… add back the actual figures everywhere making sure no available figure is banked unnecessarily… don't worry about page limits."*). Everything below is measured, with the command named; nothing is asserted.

**DIAL DECLARATION:** none — editorial restoration. **Zero new measurements. Zero new numbers.** Every added numeric token is traced to an ancestor line below and by the two-way numcheck in §3.

| | |
|---|---|
| **File edited (the only content file touched)** | `pj_sub.tex` |
| **md5 before** | `6c1902f74ee9611d718cc65b9fd1a031` (matches the value `pj-fidelity-v5-r2` audited) |
| **md5 after** | `d83447ef623345084529e1e4810c3e5c` |
| **size** | 42,790 → 69,045 bytes (302 → 447 lines) |
| **Built with** | `/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex` ×2, run **only inside this folder** |
| **Result** | `Output written on pj_sub.pdf (19 pages, 1759289 bytes)` — **0 errors, 0 undefined references/citations, 0 overfull `\hbox`, 0 overfull `\vbox`** |
| **Edits applied** | **50**, each an exact-match single-occurrence replacement asserted by script (`.claude/scratch/pj-restore-v5/restore{,2,3,4,5}.py`) |

---

## 0 — Byte-untouched verification for the rest of the folder

Only `pj_sub.{tex,aux,log,out,pdf}` have new mtimes (08:25 today). Everything else retains its pre-pass mtime and hash:

| file | md5 | mtime |
|---|---|---|
| `submission.tex` | `1d0906fe45dc78436880c938ad227332` (unchanged) | Aug 21 19:16 |
| `neurips_2025_ml4ps.sty` | `393afa47218eca1dbfdf4c42cb0da759` (unchanged) | Aug 21 19:08 |
| `BUILD-NOTE.md` | `2d9bbc948e175c6bc9fa0ca00b45f9c3` | Aug 21 19:18 |
| `submission.{aux,log,out,pdf}` | — | Aug 21 19:16 |
| `pj_sub_buildcopy.*` (5 files, obsolete lineage per Add.57) | — | Aug 22 01:00/04:58 — **not deleted; still awaiting the Head's word** |
| `figs/*.png` (11) | manifest `a5e964aad6c19fbb06b519c70114efe3` | Aug 21 19:08 |

`pdflatex` was run **only** in this folder. No file outside this folder was written except the spoke report `.claude/outputs/pj-restore-v5.md`.

---

## 1 — Page count and split (from `pj_sub.aux` `\@abspage@last` + `pdftotext`, not estimated)

| block | pages |
|---|---|
| main text (title → §4 Limitations) | **pp. 1–6 (6 pp)** |
| references | pp. 7–8 |
| appendices A–E | pp. 8–19 (A p.8 · B p.9 · C p.14 · D p.15 · E p.17) |
| **total** | **19 pp** |

⛔ **Not fought, per the Head's ruling** ("page limits are a future pass's problem"). For reference the accepted base `submission.pdf` is also 19 pp; the pre-pass `pj_sub.pdf` was 11 pp with no figures and 5 of 20 negatives.

---

## 2 — PART A: figures and tables

### 2.1 Figure placement map — **11 available / 11 shipped / 0 banked / 0 placeholders**

`\includegraphics` 0 → **11**; `\framebox` placeholders 3 → **0**.

| # | file | home in this build | base home (`BUILD-NOTE.md` §7) | caption source | evidences |
|---|---|---|---|---|---|
| 1 | `fig1_damping_optimum.png` | **main text, Fig 1** (p.4) | main text | already base-verbatim in `pj_sub` | the collapsed V-curve (headline) |
| 2 | `fig2_vault.png` | **main text, Fig 2** (p.5) | App C | base L245 **verbatim** | refrigerator + 8× mechanism + $107.77\pm4.78\times$ |
| 3 | `figB_dlaw.png` | App A, Fig 3 | App A | Head's caption kept; C(g) tag fix + A2-14 clause restored | the 25-cell diffusion law |
| 4 | `figB_signflip.png` | App A, Fig 4 | App A | base L150 **verbatim** | the sign flip and $n_{1/2}\propto1/T$ |
| 5 | `figB_massive_vs_flat.png` | App A, Fig 5 | App A | base L156 **verbatim** | the two regimes and the exact latch |
| 6 | `figA1_damping_optimum_full.png` | App B, Fig 6 | App B | base L164 **verbatim** | full-size collapse incl. the crimson probe-floor tick |
| 7 | `figC_lambda_coset.png` | App B, Fig 7 | App B | Head's caption kept | emergent V-curve + the $10^{-3}$ latch failure |
| 8 | `figC_register_capacity.png` | App B, Fig 8 | App B | base L219 **verbatim** | "no continuous coset register", ≈1–1.6 bits |
| 9 | `figC_Tstar.png` | App B, Fig 9 | App B | base L225 **verbatim** | the crossover $T^\star$ — **its claim returns with B6**, so the figure has a live home |
| 10 | `fig2_two_instruments.png` | App B, Fig 10 | App B | base L232 **verbatim** | the V-curve on a second instrument |
| 11 | `figC2_vault_emergent.png` | App C, Fig 11 | App C | base L297 **verbatim** | the vault on an emergent register |

**One placement deviates from the base:** `fig2_vault.png` is promoted App C → **main text**, per task A1 and referee MF-1 ("Fig 1 and the vault contrast are the referee's two MUST-restores there"). Nothing else moved. **No figure is banked, and none needed the "no surviving claim" fallback.**

### 2.2 Tables — 4 → **8** (the base's 8)

| table | status |
|---|---|
| App A — the $T>0$ face of the budget | present, **restructured** (A3) |
| App B — the four-instrument shape claims | present, **restructured + mean±sd row restored + threshold rider added** (A2/A3/B2) |
| App B — the instrument-gap table | **RESTORED** (base L188–201, verbatim; columns re-specified) |
| App C — the emergent refrigerator table | **RESTORED** (base L251–265, verbatim) |
| App C — the confinement / hop-fraction table | **RESTORED** (base L267–284, verbatim; carries the B6 control arms) |
| App D — the laundering-control table | present, **restructured**; caption gains the base's σ_obs admission |
| App E — negatives table 1 | present, **+6 rows** |
| App E — negatives table 2 | **RESTORED** (base L348–364, 9 rows) |

### 2.3 A3 — the truncation fix, with the before→after mechanism

⛔ **No `\small` was added anywhere; no content changed.** The pre-existing five `\small` exceptions are untouched (Add.52 carry-item).

| table | before | after | how |
|---|---|---|---|
| App A, $T>0$ budget (tex l.192–200) | Overfull `\hbox` **177.5 pt** — rendered cut mid-clause at *"temperature erases (−0.956 to −0"* | **0 pt** | column spec `lll` → `p{0.19\linewidth}p{0.40\linewidth}p{0.31\linewidth}` (text wraps instead of overflowing) + `\addlinespace` between the two stacked rows |
| App B, four instruments (tex l.226–235) | Overfull `\hbox` **152.6 pt** | **0 pt** | **transposed and stacked**: rows = quantity (γ_crit / argmin / slope below / slope above) × instrument (I-J/I-R1/I-R2/I-R3), columns = seed 42 / 43 / 44 / mean±sd. Every one of the original 39 values is preserved value-for-value; the base's mean±sd row (12 further values) is restored |
| App D, laundering control (tex l.271–280) | Overfull `\hbox` **49.2 pt** (inherited) | **0 pt** | `lccc` → four `p{}` columns |
| App B, instrument-gap (newly restored) | 49.4 pt on first build (the base's own inherited overfull) | **0 pt** | four `p{}` columns |

**Final log:** `grep "Overfull \hbox" pj_sub.log` → **no matches at all** (not merely none > 10 pt).

---

## 3 — Two-way numeric check (printed in full)

Instrument: `.claude/scratch/pj-restore-v5/numcheck.py` — typography arguments excluded (`p{...}`, `\multirow`, `\includegraphics[...]`, `\vspace`, `\parbox`), then all `\d[\d.,]*` tokens counted, three-way against `pj_sub.PRE.tex`, `submission.tex` and `NIPSsubmission/v2-neurreps/submission.tex`.

| check | result |
|---|---|
| distinct numeric tokens added by this pass | **248** |
| **added tokens with NO ancestor** | ⭐ **NONE (empty)** |
| added tokens ancestored in `submission.tex` | 247 |
| added tokens ancestored elsewhere | **1** — `2603.01768`, the J&P arXiv id (see §4 item (d)) |
| tokens **removed** by this pass | **2**: `1.0068`, `0.0219` — each −1 only, because the deleted Figure-2 *placeholder box text* duplicated them. Post-edit counts: `1.0068` = 3, `0.0219` = 3. **No result number was dropped, rounded, moved or re-scaled.** |
| digits/±/units/seed-counts altered | **0** |

**Reverse direction (base ancestors still not in the file): 132 distinct tokens.** These are base content deliberately outside this bounded pass; the substantive ones, listed so nobody re-discovers them as new:

- $n_{1/2}\ge9.483\times10^{15}$ / $9.400\times10^{-13}$ / $7.309\times10^{-17}$ — the $\mu\to0$ model-side bound paragraph (base L207).
- $\mathrm{MSD/pred}=1.0011\pm0.0215$ (15 cells); the $T=0$ write-attenuation control $0.09576$ vs $0.095238$; $\big||\lambda|_{\max}-1\big|\le2.0\times10^{-15}$ (27 cells) — base L241 residue.
- The OU cross-validation triple $112.58\pm1.09$ / $14.16\pm1.38$ / $8.03\pm0.80$ (base L249; **$8.03\pm0.80$ does appear**, via the restored App-E row).
- $297.8\pm196.8\times$ (median $234.4\times$) and the emergent bounded-cell $\hat D_{\rm ou}/D_{\rm absorb}$ series (base L286).
- The emergent first-passage paragraph's detail (base L290) — $>1379\times$, $35.5\times$, $>1290\times$, $86.7\%$, $93.4\%$ **do** appear via the restored App-E row; the per-seed $|\nabla V|$ / $V(\pi)-V(0)$ detail does not.
- $A_{75}=2.57\sigma_{\rm obs}$ ($0.0780/0.2627/0.7537$ vs $0.0771/0.2570/0.7710$), white-box $0.91409\pm0.0813$, $\mathrm{AUC}(z_{\rm hole})$ $0.576$, $\mathrm{AUC}=0.99985$, $2.836$ moves/delete, the $R_{50}$ five-point series' middle values, retention std $0.274\to0.016$ / $0.398\to0.972$ / $0.846\to0.667$ / $-0.5264$ (base L306/L321/L323).
- Bibliography DOIs and Sekhari's arXiv id (metadata thinning inherited from the Head's rewrite; a one-word Head option, not restored here).

---

## 4 — PART C: the enumerated corrections, before → after

Each is the ONLY class of non-additive edit in this pass.

**(a) `REL-4` — the Guo sentence (l.80), restored verbatim from base L51; Sekhari back in the cite.**
- **Before:** *"Formally, certified removal requires strict $(\varepsilon,\delta)$ relaxations (Guo et al., 2020), or complete isolated deletion with associated cost formalisms (Bourtoule et al., 2021; Ginart et al., 2019)."*
- **After:** *"Formally, \emph{certified} removal is Guo et al.'s (2020) \S2 Eq.~(1), an $\varepsilon$ condition with an unnumbered $(\varepsilon,\delta)$ relaxation immediately after; exact methods delete by isolation, with cost and capacity formalisms in place (Bourtoule et al., 2021; Ginart et al., 2019; Sekhari et al., 2021)."*

**(b) `REL-6` — the author token leaves prose (Add.45(2)/Add.51).**
- **Before:** *"…Lyapunov neutral modes (Mo, 2026) that provide kinematic protection…"* → **After:** *"…Lyapunov neutral modes (arXiv:2605.03338) that provide…"*
- Verified: `\bMo\b` now has **1 occurrence, tex l.167 = the bibliography entry** (expressly permitted); prose/captions/labels/filenames **0**. "Morse"/"Moser" = 0/0, so nothing could be lost to the regex.

**(c) `APA-2` + `R31-1` — the two checkably-false exactness claims (MF-4).**
- **Before (App A):** *"…lies between $0.076$ and $0.096$, **strictly aligning with** the computationally predicted bounds of $0.082$–$0.116$…"* (0.076 < 0.082).
- **After:** *"…**against** the computationally predicted bounds of $0.082$–$0.116$…"* **+ the base's explanation added:** *"The measured minimum sits below the prediction: in collapsed variables $x_{\rm min}\approx0.90$, $27\%$ above the continuum $0.707$, a known discrete-map correction (matched designed control $x_{\rm min}=0.83$–$0.93$, Appendix~\ref{app:emergent})."* (tokens: base L207).
- **Before (§3.1):** *"a distinct V-curve, **strictly minimized at** $\gamma_{\rm crit}$"* → **After:** *"a distinct V-curve, **minimized at** $\gamma_{\rm crit}$"*.

**(d) `JP-1` — the J&P reference entry (MF-3). ⚠ READ THE ANCESTOR NOTE.**
- **Before:** *"Jawahar, P., \& Pierini, M. (2026). \emph{Causal Hamiltonian Learning Units (CHLU)}. [Reference redacted for double-blind review.]"*
- **After:** *"Jawahar, P., \& Pierini, M. (2026). CHLU: The causal Hamiltonian learning unit as a symplectic primitive for deep learning. arXiv:2603.01768 (short paper, ICLR 2026 AI \& PDE workshop)."*
- ⛔ **The named base does NOT contain this ancestor.** `submission.tex` L104 carries the *same* redacted entry as `pj_sub`. Rather than invent a locator (rule 3) or ship a locator-less entry, the replacement is taken **verbatim** from two live ancestors in the program's own accepted lineage: **`NIPSsubmission/v2-neurreps/submission.tex` L179** (the sibling clean base, accepted at Add.52) and **`papers/v5-short/draft.tex` L141** (V5's own pre-submission-build draft). This is the sanctioned C-1(d) posture the referee asks for (full third-person citation, no redaction marker). **Flagged for Advisor/Head ratification — it is the one edit whose ancestor lies outside this folder.**
- Sweep consequence: `redacted` 1 → **0** in `pj_sub.tex` (positive control: still 1 in `submission.tex`).

**(e) `APE-1` — `Emprical Result` → `Empirical Result`** (App E table header).

**(f) Vocabulary accuracy, one word each.**

| id | before | after | source |
|---|---|---|---|
| `R31-3` | *"(a **variance** of $0.35\%$)"* | *"(a **difference** of $0.35\%$)"* | A2-1 / base L69 |
| `APC-4` | *"biased artificially low by **nearly** $55\%$"* | *"biased artificially low by **up to** $55\%$"* | A2-2 / base L249 |
| `APC-3` | *"records **exponential factors** of $8.42\times$…"* | *"records **factors** of $8.42\times$…"* | A2-8 / base L241 |
| `APB-3` | *"The primary cause of this threshold offset **is confirmed**:"* | *"…**is measured**:"* | A2-9 / base L189 |
| `APD-2` | *"enforces an exact **boundary** of $1.540000$"* | *"enforces an exact **minimum spacing** of $1.540000$"* | A2-6 / base L306 |
| `R32-2` | *"generates a **remarkable** retention vault factor"* | *"generates a retention vault factor"* | MF-8 |

**(g) Status tags (C-2).**
- `R31-3`: §3.1 heading *"**Cross-Instrument Verification**"* → *"**Cross-Instrument Evidence**"* (the two-instrument result is an emergent/learned result; the base labels it *evidence*, A2-12). The Head's heading phrase is otherwise kept.
- `APA-4`: Fig 3 caption *"Designed testbed: **validation of verification**."* → *"Designed testbed: **verification**."* (A2-13).
- Supporting addition (`INT-2`): the base's labelling **rule** is restored to the Nomenclature block so the tags are legible — *"Results on designed testbeds are labelled verification of exactness; on learned or emergent potentials, evidence."* (base L45).

**(h) The four App-D meaning-destroying garbles (MF-8), each to the base's plain sentence.**

| id | before | after | base |
|---|---|---|---|
| `APD-2` | *"The architectural implementation dictates 24 exhaustive structural orders explicitly measured at $n=4$, mapped alongside 200 independently randomized sequence orders strictly at $n\in\{8,16,40,64\}$."* | *"We measure 24 exhaustive orders at $n=4$ and 200 random orders at $n\in\{8,16,40,64\}$."* | L306 |
| `APD-3` | *"It must be firmly stated that at peak physical overflow limit, operating totally independently without the waitlist active, structural exactness completely and predictably collapses: analyzing exactly 7 baseline cells receiving 8 strict capacity offers,"* | *"At overflow, without the waitlist, exactness fails: with 7 cells and 8 offers,"* | L306 |
| `APD-4` | *"directly multiplying absolute payloads exponentially by their matched amplitudes completely corrupts the core value index criterion measured strictly at $A=1-\text{tol}/\|a_i\|$ (generating hard death physical amplitudes of exactly $0.90/0.80/0.70/0.30$)."* | *"multiplying payloads by amplitudes kills the value criterion at $A=1-\text{tol}/\|a_i\|$ (measured death amplitudes $0.90/0.80/0.70/0.30$ against the formula's $0.900/0.860/0.767/0.300$) and inverts the payload dependence; and launching the payload coordinate at the address-mapped point breaks the anti-decoration guard, after which a trivial substitute passes $0.672$ of queries with no dynamics at all."* | L323 |
| `APD-5` | *"The physically shipped final store structure completely abandons amplitude-independent baseline address limits, generating a strict operational lifetime correlation factor computed at exactly $r=-0.85$ scaling with $a_i^2$."* | *"The shipped store drops amplitude-independent address hold, which is why effective lifetime correlates $r=-0.85$ with $a_i^2$."* | L323 |

---

## 5 — PART B: the additive restorations, item by item

Every row below is an **insertion**; the Head's surrounding prose is not rewritten, reordered or compressed. Approved wordings, mandatory riders and never-quote-adjacent forms are **diff-identical to the base** (Add.30 boundary).

### B1 — the deletion story's guards

| worklist item | edit id | site | ancestor | status |
|---|---|---|---|---|
| encoder-exclusion scope, §A20.5 form | `INT-2` | §1 Nomenclature (new bullet) | base L45 | ✅ verbatim |
| *"This is a store-level guarantee only — the frozen encoder and any residue of past writes in a learned landscape are separate channels"* | `R33-2` | §3.3 | base L79 | ✅ verbatim (full clause incl. *"without the waitlist, exactness fails at overflow"*) |
| *"the store deleting an item is not the system forgetting it"* | `REL-3` | §2 | base L51 | ✅ verbatim |
| the certified denial — *"we claim no certified $(\varepsilon,\delta)$ unlearning"* | `R33-2` | §3.3 | base L79 | ✅ verbatim |
| the certified denial — *"We make none of those claims"* | `REL-4` | §2 | base L51 | ✅ verbatim (with *"…with the encoder excluded, coining no benchmark and no cost claim"*) |
| three conditions + recency exclusion beside the claim | `R33-2` | §3.3 | base L79 / CM-25(f) | ✅ **approved wording restored verbatim**, incl. AUC $0.5000\pm0.0000$ and byte-equal $1.0000$ |
| "priority/" restored to the eviction clause | `R33-2` | §3.3 | base L79 | ✅ (both in the Head's paraphrase and in the verbatim form) |
| three conditions + store scale in the abstract | `ABS-1` | abstract | base L32 | ✅ verbatim |
| store scale (dim 3, capacity 8–64, no learning) in §3.3 | `R33-1` | §3.3 opener | base L79 + L302 | ✅ |
| the score sentence | `R33-1` | §3.3 | base L79 / L356 | ✅ verbatim; also App E row (B3) — `ZERO` = 2 |
| the flat-table trivial-substitute control | `R33-1` | §3.3 | base L79 | ✅ verbatim |
| the σ_obs modelling-choice admission | `R33-4`, `APD-7` | §3.3 + App D caption | base L81, L309 | ✅ verbatim |
| the $0.672$ anti-decoration substitute control | `APD-4` | App D | base L323 | ✅ verbatim |

### B2 — riders on the retention / vault results

| item | edit id | ancestor | status |
|---|---|---|---|
| $\Delta=0.5$ rad **and $\ell_\theta/\Delta<0.05$ in-line at the 3.77× site** (MF-5) | `R32-1` (§3.2, the number restored with its riders), `APA-1` (App A) | base L73, L128 | ✅ — the paper's thrice-promised rule is now delivered at both sites |
| $86.97\pm2.94\times$ + the estimator-name sentence, **both** 107.77× sites (MF-6b) | `R32-2` (§3.2), `APC-3` (App C) | base L241 | ✅ verbatim |
| the 11-decade probe-floor rider (MF-6a; resolves the internal $10^{-15}$-vs-$1.7\times10^{-12}$ contradiction, N-1) | `R31-2` | base L61 | ✅ verbatim |
| App-B threshold-instrument rider (Tier-1 #7) | `APB-2` | base L174 | ✅ verbatim, in the table caption |
| *"and never the vault"* on 13.88× (A2-10) | `APC-2` | base L239 | ✅ verbatim |
| the 7.942-vs-8.11 reconciliation (SF-8) | `APC-3` | base L241 | ✅ *"…the measured field/scalar ratio being $8.11\pm0.37$ against the $7.942$ predicted for the separation of the two hypotheses above."* |
| *"The direction transfers; the number $8.11\pm0.37$ is a designed quantity."* | `APC-6` | base L288 | ✅ verbatim |
| designed-symmetry precondition **in §3.2** with $T^\star\approx3\times10^{-3}$ and designed $\le1.1\times10^{-15}$ (Tier-1 #6) | `R32-2` | base L75 | ✅ verbatim, whole block |
| the λ_coset values $1.06$–$2.96\times10^{-3}$ | `APB-4` | base L209 | ✅ verbatim |
| "6 cells" on the law-referenced emergent vault | `APC-5` | base L286 | ✅ |
| N108's placement-scope and exact-adversary riders (Tier-1 #8) | `R33-4` | base L81 | ✅ verbatim, both |
| `fdt` + Newtonian fine print beside §3.2's $T>0$ claims (Tier-1 #12) | `R32-1` | base L73 | ✅ verbatim |

### B3 — the negatives estate

`APE-2` (+6 rows, base L337/339/340/341/342/343) and `APE-3` (+9 rows, base L353–362, as the base's second table).

⭐ **Counted, not assumed: the base has 20 negatives rows, not 21.** (`pj-fidelity-v5-r2` §B.2 Tier-1 #2 says "5 of 21"; the measured base count is 20 — one row of the base's second table, *"A tilt sets the soft scale on a learned store"*, is the row `pj_sub` already carried as its row 5.) **5 present + 15 restored = 20 = the base's full estate.**

⇒ **The existing sentence *"every negative result observed during evaluation is documented below"* is now TRUE** (verified by script: 20 rows counted in the two tabulars; the sentence is present).

The four rows that guard surviving claims are all in: the score-sentence row ✅ · *"Threshold instruments are usable below $\gamma_{\rm crit}$"* ✅ · *"Our microscopic explanation of the instrument offset — **Wrong**"* ✅ (pairs with correction C(f)/`APB-3`) · *"The placement algorithm is ours — No: Blelloch \& Golovin (2007) own it outright"* ✅ · plus the amplitude-decay/distinguishability row ✅.

### B4 — prior art and citations

- `APD-1`: **App D's prior-art paragraph restored verbatim** (base L304), carrying the scout-corrected lineage (Snyder '77 canonical · Andersson & Ottmann '95 · Micciancio '97 *oblivious, explicitly not canonical* · Naor–Teague '01 weak/strong HI · Blelloch–Golovin '07 · BGV '08), **the no-priority clause** (*"We claim no priority over order-independent placement and no novelty for the displacement rule or its delete-time repair"*), and **the fourth Blelloch–Golovin attribution site** (*"the algorithm is theirs either way"*). `Blelloch` = 8 occurrences; App D no longer promises prior art it does not contain.
- **Orphaned references: 10 → 0.** Andersson & Ottmann, BGV 2008, Buchbinder & Petrank, Micciancio, Naor & Teague, Snyder (all `APD-1`) · Sekhari (`REL-4`, rides C(a)) · Wang B. et al. (`REL-3`, rides B1) · Rusch et al. and Jude et al. (`REL-5`, with the base's *"we make no biological claim"*, present exactly once).

### B5 — positioning and scope

| item | edit id | ancestor |
|---|---|---|
| the seam sentence closing §2 (SF-7) | `REL-4` | base L51, verbatim |
| the Titans one-liner (SF-5) | `REL-1` | base L49, verbatim |
| the abstract's TTL laundering negative + explicit conditions (SF-3) | `ABS-1` | base L32, verbatim |
| the R₅₀ differentiator, $1.146\to0.752$ · $1.52\times$ · TTL radius $0.75$–$0.77$ (Tier-1 #10) | `R33-4` | base L81, verbatim |
| Limitations: the deletion store's own scale (MF-7) | `LIM-1` | base L85 |
| Limitations: item (iv), the instrument-offset / eleven-decade limitation | `LIM-2` | base L85, verbatim |
| Limitations: *"No task-level payoff is claimed"* | `LIM-2` | base L85 |
| Limitations: the $O(n)$-rebuild/unrun deletion-cost clause | `LIM-2` | base L85 |
| named-next: *"deletion at $10^3$-item stores"* | `LIM-3` | base L85 |
| the anonymization note (Tier-1 #16) | `APE-3` | base L367, verbatim |
| the compute-adaptive-read dial named with the trilemma | `R33-3` | base L79, verbatim |
| App C status header (verification/evidence) | `APC-1` | base L237, verbatim |
| App D status header (verification) | `APD-1` | base L302, verbatim |
| the theory-note handling (SF-4) | `INT-1` | base L39 — *"(Anonymous, 2026, a theory note on which nothing here depends)"* |

### B6 — precision restorations on correct numbers

| item | edit id | before → after | ancestor |
|---|---|---|---|
| A2-3 *equal* $\gamma_{\rm eff}$ on the scalar control | `R32-2` | *"a uniform scalar friction"* → *"a uniform scalar friction **of equal $\gamma_{\rm eff}$**"* | base L75 |
| A2-5 the rollout arm's scope | `R31-3` | *"direct nonlinear rollout"* → *"direct nonlinear rollout **(3 emergent seeds, $\delta=0.05$–$0.5$ rad)**"* | base L69 |
| A2-7 *"a large share of sampled"* | `REL-2` | *"driving **significant** recommendation errors"* → *"driving **a large share of sampled** recommendation errors"* | base L51 |
| A2-11 the amortized-experiment-unrun admission | `APD-6` | *"…$O(n)$ rebuild evaluated entirely per operation."* → *"…per operation, **and the amortized-cost experiment has not been run**."* | base L323 |
| the $T^\star$ crossover claim with its base numbers (gives `figC_Tstar` its home) | `APB-4` | new paragraph: bias-corrected $2.75,3.40$ → $0.94$–$1.25$, $T^\star\approx3\times10^{-3}$ (predicted $2.72$–$3.66\times10^{-3}$), washboard barrier $2.29$–$3.57\times10^{-2}$, and the non-discriminating raw exponents | base L209, verbatim |
| the confinement claim's two control arms (Tier-1 #5) | `R32-2` (sentence) + `APC-6` (table) | *"…falls from **$5.5/43.0/2.4\%$ with no hole** to $0.0000$ inside it, 3/3 seeds, while a **scalar friction of the same $\gamma_{\rm eff}$ still hops ($0.73/10.2/0.26\%$)**."* | base L75, L267–284 |

### Additive insertions beyond the enumerated worklist (permitted; declared here)

Three, each justified and base-verbatim:
1. `INT-2` — the C-2 **status-labelling rule** in Nomenclature (base L45). Needed to make correction (g)'s tags legible.
2. `APA-4` — the Fig-3 caption's **multi-seed disambiguation** (*"the single checkpoint on which the 25-cell grid was run; the five-seed statements of this appendix are the sign-flip and latch entries, not this grid"*, base L144, A2-14). A scope rider, dropped by the rewrite.
3. `APB-4` — the base's **Collapse/Scope** sentences (base L207). These are the in-file ancestor for correction (c)'s discrete-map explanation and for restored limitation (iv)'s $2.7\times$.

---

## 6 — Sweeps (per file, positive-controlled, printed)

**Zero-hit list re-run on `pj_sub.tex` — all 0** (fidelity §A.4 list, plus this pass's own additions):
`13.9` · `≈14×` · `we alone` · `CLU-former` · `0 of 5` · `CSF3` · `prior mismatch` · `P=4` · `compositional family` · `residual protects` · `watch stayed green` · `state-of-the-art` · `SOTA` · `best-in-class` · `benchmark win` · `beats` · `wins` · `outperform` · `deletion-compliant` · `0.272` · **`right-to-be-forgotten`** · **`memory provenance`** · `companion` · `sibling` · `forthcoming` · `in preparation` · `github` · `zenodo` · `huggingface` · `.claude` · `chlu/` · `PALM` · `Morse` · `Moser` · `cryptographic unlearning privacy` · `eradication` · **`Placeholder`** · **`redacted`**.

**Instrument LIVE (positive controls that must fire on `pj_sub.tex`):** `107.77` ×7 · `Blelloch` ×8 · **`encoder` ×4 (was 0 — the round-2 regression is closed)** · `ZERO` ×2 · `includegraphics` ×11 · `trivial substitute` ×2 · `verification` ×10 · `evidence` ×11 · `dim 3` ×4 · `capacity 8--64` ×4 · `priority/attribute` ×2 · `budget >= n` ×2 · `leak = 0` ×2.
**Not vacuously clean:** the same zero-list on `submission.tex` fires `redacted` = 1 (the base's own J&P defect), confirming the instrument detects what it is looking for.

**`certified` — per-occurrence table (n = 3):**

| # | tex line | exact text | verdict |
|---|---|---|---|
| 1 | 82 | *"Formally, \emph{certified} removal is Guo et al.'s (2020) \S2 Eq.~(1), an $\varepsilon$ condition with an unnumbered $(\varepsilon,\delta)$ relaxation immediately after…"* | ✅ **literature-description form (permitted)** — and now *correctly stated* (correction (a)) |
| 2 | 128 | *"…and **we claim no certified $(\varepsilon,\delta)$ unlearning**; without the waitlist, exactness fails at overflow…"* | ✅ **THE DENIAL — restored** (absent for two rounds) |
| 3 | 161 | `Guo, C., … (2020). Certified data removal from machine learning models.` | ✅ reference title |

⇒ **zero affirmative forms; the denial is present.** `unlearning` ×6 — l.46/82/128 are denial/contrast forms, l.156/174/178 are reference titles.

**Author-token rule (Add.45(2)/Add.51):** `\bMo\b` = **1 occurrence, tex l.167, the bibliography entry**. Prose/captions/labels/filenames = 0. "Morse"/"Moser" = 0/0.

**Mandated-once strings:** the honest-scope sentence (*"This is not a deployed, large-scale LLM agent memory, nor a generalized system benchmark."*) = **exactly 1** · *"we make no biological claim"* = 1 · *"introduced as CHLU"* (the CLU continuity sentence) = 1.

---

## 7 — Not executed, or executed with a deviation — stated honestly

1. ⚠ **Correction (d)'s ancestor is outside this folder.** See §4(d). The named base carries the same defect. **This is the one edit the Advisor/Head should ratify by name.** The alternative — deleting the redaction note and shipping a locator-less entry — was rejected as a worse artifact.
2. ⚠ **The restored lifecycle negatives row has no host claim in this paper.** B3 mandates all missing rows, and the completeness sentence is only true with it. But the Head's rewrite cut contribution (4) entirely (`lifecycle` was 0 before this pass; it is now 1, *only* in that App-E row: *"The lifecycle's protected-fraction leg is exercised — No: 0 refusals…"*). **Head decision:** keep it (App E's completeness claim is true, and a negative about an unshipped component is honest) or cut it (then the word "every" needs softening). Not a claims defect either way; flagged so nobody re-discovers it.
3. **Not in the worklist, therefore untouched** (each is a live fidelity/referee finding the Head may still want):
   - **A2-4** — §3.2's *"holds **universally** across all evaluated seed and temperature combinations"* against the base's *"10/10 seed × temperature conditions"*. The 10/10 survives in App A; the main-text quantifier is the Head's.
   - **A2-12 second half** — §3.1's *"across the 5 **validation** seeds"* (base: *"5 seeds, verification"*). Only the *heading* was enumerated in C(g).
   - **The intensifier layer (MF-8)** beyond the enumerated words: *"comprehensively"*, *"definitively"*, *"flawlessly"*, *"match perfectly"*, *"heavily randomized"*, *"rigidly"*, `utilizing` ×11 remain. Removing them is a rewrite, not an addition, and only *"remarkable"* was enumerated. `completely and predictably collapses` did go, inside C(h2).
   - **App C's heading** *"Emergent Arm **Translation**"* (fidelity A.2-C4 ⚠).
   - **Bibliography DOIs / Sekhari's arXiv id** (Add.59: "deliberately NOT in scope").
4. **`pj_sub_buildcopy.*` (5 files) not deleted** — Add.59 keeps that awaiting the Head's word; deleting them is outside this pass.
5. **No re-referee, no page pass.** 19 pp is reported, not fought.
6. **Fidelity-report correction:** the base's negatives estate is **20 rows, not 21** (§5/B3). Everything else in `pj-fidelity-v5-r2`'s inventory reproduced exactly.

*Filed by `pj-restore-v5` (paper-writer), 2026-08-22.*
