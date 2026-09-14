# pj-fidelity-v5-r2 — doc-curator report

**Task + acceptance criterion:** ROUND-2 fidelity audit of the Head's rewritten `NIPSsubmission/v5-palm/pj_sub.tex` against the accepted clean base `submission.tex` — Part A numeric fidelity + ⭐ citation ancestry; Part B claims fidelity + re-adjudication of every round-1 finding (⛔ "certified" per-occurrence), the do-not-cut walk, the claims table, the mechanical inventory.
**Status: done.** (No render step in this task; the PDF pre-exists. Page split recovered exactly from `pj_sub.aux`, not estimated.)

**DIAL DECLARATION (echoed):** none — read-and-report only; no performance claim; no laundering control applies to this report.

> ⛔ **THE EDIT-BAR HELD. Zero `Edit`/`Write` calls were issued against any file in `.claude/NIPSsubmission/`.** Tools used on that folder: `Read`, `Grep`, `Glob` only. My single write is this report. ⚠ **I have no shell, so I cannot print the md5** — the Advisor re-verifies `6c1902f74ee9611d718cc65b9fd1a031` after this pass; byte-identity here rests on tool provenance, not a hash.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — READ FIRST (protocol §5 corollary; each needs a named owner):**
> 1. ⛔⛔ **`encoder` = 0 occurrences in the whole file** (base: 5). The store-level/encoder-excluded scope — the single sentence that stops "store-level deletion" reading as system-level unlearning — has vanished **entirely**, including from §1, which carried it in round 1. **This is a REGRESSION against round 1 and now the file's most serious claims defect.** Owner: Head.
> 2. ⛔⛔ **App E asserts completeness falsely:** *"every negative result observed during evaluation is documented below"* — **5 rows against the base's 21.** Round 1 explicitly credited the old framing for *not* claiming completeness. A false statement about the artifact's own evidence base. Owner: Head.
> 3. ⛔ **The Guo citation is now mis-stated:** *"certified removal requires strict $(\varepsilon,\delta)$ relaxations (Guo et al., 2020)"* — Guo §2 Eq. (1) is an **ε-only** condition; the $(\varepsilon,\delta)$ form is the *unnumbered relaxation after it*. The corrected form the base carries is inverted. Owner: Head; cite-check spoke to confirm.
> 4. ⛔ **`(Mo, 2026)` appears in prose at l.83** — Add.45(2)/Add.51 forbid the author token in body text; the base deliberately cites the same work as *"(arXiv:2605.03338)"*. Round 1 measured 0. New violation. Owner: Head.
> 5. ⛔ **The score sentence, the flat-table trivial substitute, and the σ_obs modelling-choice admission remain absent** (`ZERO` = 0, `trivial` = 0, `flat table` = 0, `modelling choice` = 0). Owner: Head.

---

## 0. Instrument and method

- **Objects:** `pj_sub.tex` (302 lines) vs `submission.tex` (371 lines, the Add.52 clean base). Every numeric token of `pj_sub.tex` walked in file order against the base; base line numbers cited throughout.
- **Sweeps are per-file and positive-controlled** (⚠ standing memory: directory-level Grep over `.claude/` returns false negatives).
  - **Positive control (instrument LIVE):** combined pattern `107.77|Blelloch|certified|unlearning|recency|encoder|lifecycle|ZERO|…` on `pj_sub.tex` ⇒ **17 matching lines**.
  - **Zero-hit control (the instrument is not vacuously clean):** the same zero-list pattern `encoder|ZERO|includegraphics|lifecycle|trivial substitute` run on `submission.tex` returns **19 hits**; on `pj_sub.tex` it returns **0**. The zeros below are real.
- **Page split: exact, from `pj_sub.aux`** (`\@abspage@last{11}`; §4 starts p.5, `\clearpage`, App A starts p.7): **main text pp. 1–5 (5 pp) · References p. 6 (1 p) · Appendices A–E pp. 7–11 (5 pp) · total 11 pp.**
- **Not done:** no shell ⇒ no md5, no re-render, no pixel-level check of figure content (moot — there are no figures, §B.4).
- ⚠ **Housekeeping, not mine to fix:** `pj_sub_buildcopy.{tex,aux,out,pdf,log}` are still on disk. Add.57 declares them obsolete lineage; I created the `.tex` in round 1 and cannot delete it (a write into `NIPSsubmission/`). Flagged for the Advisor/Head.

---

# PART A — numeric fidelity and citation ancestry

## A.1 — Numbers: the headline answer, unchanged from round 1

⭐ **No number in `pj_sub.tex` lacks an ancestor in `submission.tex`.** Every surviving numeric token — value, precision, ±, units, seed counts — is transcribed **exactly**: zero digit changes, zero dropped ±, zero unit changes, zero rounding changes, zero invented quantities. This holds across the rewrite's ~5.2 k words including three full tables reproduced value-for-value.

**Spot ledger of the load-bearing tokens (pj → base ancestor), all verdict `exact`:**

| pj site | token(s) | base ancestor |
|---|---|---|
| abs l.39, §3.1 l.93 | $\gamma_{\rm crit}=2\varepsilon\mu$; ≈11 orders in $\mu^2$; dim 4 / hidden 64 / $\varepsilon=0.05$ / 150 epochs; 5 designed / 3 emergent | L32, L61 |
| §3.1 l.93–95 | $-1.006$, $+1.23$–$+1.27$; Hessian $\mu^2\approx10^{-15}$; argmin $0.902\pm0.003\gamma_{\rm crit}$; $-1.0020\pm0.0003$ / $+1.116\pm0.011$; $\mu^2\in[1.7\times10^{-12},7\times10^{-2}]$ | L61 |
| §3.1 l.110 | $0.9001\pm0.0052$ vs $0.9032\pm0.0027$; $0.35\%$ | L69 |
| Fig 1 caption l.106 | $\mu^2_{\rm rad}=0.670$–$1.348$; $\mu^2_{\rm soft}=2.0$–$5.4\times10^{-2}$; 3/3 seeds | L65 (⭐ caption restored verbatim) |
| §3.2 l.114–118 | drift $\le4.9\times10^{-12}$ rad / 200k, $\gamma\in[0.002,0.5]$; $D_\theta=\varepsilon T(2-\gamma)/(2F^2\gamma)$; $1.0068\pm0.0219$ / 25 cells; $T_{\rm local}=1.26\times10^{-4}$ vs $10^{-3}$; $(\gamma_{\rm eff}/\gamma)^2$; $107.77\pm4.78\times$; $13.28\pm0.12\times$; $0.0000$ | L73, L75 |
| §3.3 l.124–126 | $0.29\times$–$1.71\times$; $0.017$ AUC, $0.983$ vs $1.000$ | L79, L81 |
| App A l.180–188 | $\Delta=0.5$ rad; min $0.9644$ / max $1.0484$; five $d\log D/d\log T$; five $d\log D/d\log\gamma$; $+0.9552\pm0.0422$, 10/10; $3.77\pm0.23\times$ + per-seed $3.55,3.64,3.61,3.89,4.15$; $-0.956$…$-0.979$; $\le1.7\times10^{-14}$ / 22 γ / 5 seeds; 30/30; $142.7$ rad / 20k; $0.076$–$0.096$ vs $0.082$–$0.116$; $\mu^2_{\rm rad}$ five values; floor $27.03$ | L126, L128 |
| App A table l.196–197 | $+0.955\pm0.042$; $3.77\times$; $-0.956$…$-0.979$; $-1.006$; max/min $\le1.013$ | L136–137 |
| App B l.222–241 | $R^2\ge0.9955$ / median $0.9996$; **the full 3×13 instrument table (39 values)**; $0.9216\pm0.0027$, $0.0004$; $0.638/0.644/0.327$; nine argmin values at $\delta=0.05/0.2/0.5$; $+1.4\%$, $10\times$, $1.9\%$, $6.5\times10^{-10}$; $\mu^2=5.449/2.029\times10^{-2}$; $\lambda_{\rm ret}=0.998999499$; $692.5$ vs $693.1$ | L171, L179–181, L189, L203, L205 |
| App C l.257–261 | $13.88\times$; $7.942\times$; $0.9986$/$0.36171\pm0.00129$/$0.23044$/$0.17513$/$0.12600\pm0.00031$/$0.12591$; $7.94\times$; $8.42/22.39/44.31/107.77\pm4.78\times$; $110.25$; $\le1.75\times10^{-12}$; $\mu^2_{\rm adiab}=2.77$–$8.93\times10^{-2}$; $55\%$; $106.1\pm5.0\times$; $23.39\pm10.06$; $[6.5,9.5]$ | L239, L241, L249, L286, L288 |
| App D l.265–282 | 24 orders @ $n=4$; 200 orders @ $n\in\{8,16,40,64\}$; 100 interleavings; $1.540000$; 64 queries; $K=64$, $61/64$, $\sigma=0$, $0.9531$, $\times1.05$, $64/64$, $43/64$, $0.6719$; 7 cells / 8 offers, $\mathrm{AUC}=1.00000\pm0.0000$, $0.0000$; table $1.000/1.000/0.000$, $0.983/1.000/0.017$, $0.559/0.996/0.437$; 8 targets × 3 seeds × 128 worlds × 18 levels; $r=-0.85$; $0.90/0.80/0.70/0.30$; $R_{50}$ $1.135\to0.771$ | L302, L305, L306, L309, L314–316, L321, L323 |
| App E l.293–297 | $1.7$–$4.9\times$; $\approx1$–$1.6$ bits; factor $8.11$; $107.77\pm4.78\times$ vs $13.28\pm0.12\times$; $2\alpha$, $\tau_{\max}=\Gamma/2\alpha$ | L334, L336, L354 |

**⭐ Restored since round 1 (numbers that had been dropped and are now back, all exact):** $\Delta=0.5$ rad · $142.7$ rad / 20k control · floor $27.03$ · $\le1.7\times10^{-14}$ · $110.25$ · $8.11$ · $13.88\times$ / $7.942\times$ · the whole negative-packing block ($61/64$, $\sigma=0$, $0.9531$, $\times1.05$, $43/64$, $0.6719$) · the overflow block · $r=-0.85$ · $0.90/0.80/0.70/0.30$ · $R_{50}$ $1.135\to0.771$ · $\mu^2_{\rm rad}$ per-seed · the left-branch identity · finite-write-amplitude sweep.

**Numbers in the base still dropped from `pj_sub.tex` (omissions, not errors):** $9.483\times10^{15}$ · $\le1.1\times10^{-15}$ latch floor · $T^\star\approx3\times10^{-3}$ · $1-|\lambda_{\rm coset}|=1.06$–$2.96\times10^{-3}$ · $86.97\pm2.94\times$ · $297.8\pm196.8\times$ · $0.9998\pm0.0019$ · $0.2235$ · $0.4586\pm0.1181$ · the whole emergent refrigerator table (24 field cells, 2048 walkers × 40 samples) · the whole confinement table incl. **the scalar control $0.73/10.2/0.26\%$ and the no-hole $5.5/43.0/2.4\%$** · $\mathrm{MSD/pred}=1.0011\pm0.0215$ (15 cells) · the OU cross-validation triple ($112.58\pm1.09$ / $14.16\pm1.38$ / $8.03\pm0.80$) · instrument-gap table ($1.3307/1.7981/1.8204$, CV $2.2$–$3.7\%$, rate ratio $0.995\pm0.003$) · mean±sd row of the App-B table · $\mathrm{AUC}=0.5000\pm0.0000$ / byte-equal $1.0000$ · $\mathrm{AUC}=0.99985$ · $R_{50}$ $1.146\to0.752$ and TTL radius $0.75$–$0.77$ / $1.52\times$ · retention $0.832$ at $A=0.051$ · $2.836$ moves/delete · white-box $0.91409\pm0.0813$ · $\mathrm{AUC}(z_{\rm hole})$ $0.85$ vs $0.576$ · early-window $0.52$–$0.85$ vs $[0.65,1.35]$ · $A_{75}=2.57\sigma_{\rm obs}$ · $0.672$ trivial-substitute pass rate · $0.274\to0.016$, $0.398\to0.972$, $0.846\to0.667$, $-0.5264$ · dim 3 / capacity 8–64 · 9/9.

## A.2 — ⚠ Scope, precision and quantifier mismatches attached to correct numbers (Part A's real defect class)

| # | pj text | base form | issue |
|---|---|---|---|
| A2-1 | l.110 *"a variance of $0.35\%$"* | L69 *"a $0.35\%$ difference"* | statistical mislabel — a difference reported as a variance |
| A2-2 | l.261 *"biased artificially **low by nearly** $55\%$"* | L249 *"biased low by **up to** $55\%$"* | an upper bound restated as an approximate central value |
| A2-3 | l.118 *"compared to a baseline of $13.28\pm0.12\times$ utilizing a **uniform** scalar friction"* | L75 *"for a scalar friction of **equal $\gamma_{\rm eff}$**"* | the control's matching condition — the whole point of the control — replaced by "uniform" |
| A2-4 | l.116 *"holds **universally** across all evaluated seed and temperature combinations"* | L73 *"holds in **10/10** seed × temperature conditions"* | count → universal quantifier (the 10/10 does survive in App A l.184) |
| A2-5 | l.110 *"we re-measured **the entire curve**"* | L69 *"(3 emergent seeds, $\delta=0.05$–$0.5$ rad)"* | the rollout arm's scope is dropped; reads as re-measuring designed+emergent |
| A2-6 | l.265 *"the resulting structural spacing enforces an exact **boundary** of $1.540000$"* | L305 *"**minimum spacing** exactly $1.540000$"* | a minimum relabelled as a boundary (round-1 finding recurs in new wording) |
| A2-7 | l.46 *"outdated retention driving **significant** recommendation errors"* | L51 *"drive **a large share of sampled** recommendation errors"* | "sampled" dropped; "significant" is an unearned significance word (no test) |
| A2-8 | l.259 *"records **exponential** factors of $8.42\times$…"* | L241 *"the $D_\theta$-based vault is $8.42\times$…"* | vault factors described as "exponential factors" |
| A2-9 | l.237 *"The primary cause of this threshold offset **is confirmed**"* | L189 *"Cause of the threshold offset, **measured**"* | "confirmed" over-states the base, and the base's own negative (L361: the micro-explanation is **Wrong**) is not carried — see B.1(A.2-C7) |
| A2-10 | l.257 *"experimental results comprehensively refute the coupled-bath prediction"* | L239 *"the coupled-bath value is a refuted prediction **and never the vault**"* | the protective rider "never the vault" dropped from a sentence that prints $13.88\times$ — adjacent to the standing $\approx14\times$/13.9× never-quote |
| A2-11 | l.282 *"we define no explicit deletion-cost execution statement **because** the … $O(n)$ rebuild"* | L323 *"No deletion-cost statement is made: … $O(n)$ rebuild per operation **and the amortized-cost experiment has not been run**"* | the reason shifts from "unrun" to "architecture"; the unrun-experiment admission is lost |
| A2-12 | l.130 *"across the 5 **validation** seeds"* (l.110 heading *"Cross-Instrument **Verification**"*) | L61 *"5 seeds, **verification**"*; L69 *"Two instruments, one law (**evidence**)"* | ⛔ C-2 status-tag misuse: the two-instrument result is **evidence** (emergent) in the base and is titled *Verification* here |
| A2-13 | Fig 2 caption l.210 *"Designed testbed: **validation of verification**"* | L144 *"Designed testbed: **verification**"* | the two-layer status tag garbled into a non-tag |
| A2-14 | Fig 2 caption l.210 | L144 adds *"the single checkpoint on which the 25-cell grid was run; the five-seed statements of this appendix are the sign-flip and latch entries, not this grid"* | the multi-seed-status disambiguation is dropped; "one checkpoint (seed 44)" survives |

**Intensifier load (new, no ancestor, attached to claims):** l.110 *"reproduced flawlessly"* · l.116 *"holds universally"* · l.118 *"a remarkable retention vault factor"*, *"drops definitively"* · l.126 *"significantly before"* · l.237 *"match perfectly"* · l.257/259 *"comprehensively"*, *"definitively"* · l.267 *"completely and predictably collapses"*, *"heavily randomized"* · l.293 *"measuring exactly $1.7$–$4.9\times$"* (an "exactly" in front of a range). None changes a number; several inflate a verdict.

## A.3 — ⭐ Citation-ancestry check (the new Part-A instrument)

**⭐ THE NEW-CITATION LIST IS EMPTY. Zero new citations.** Every work cited in `pj_sub.tex` has an ancestor in the base's bibliography, and the bibliography is **32 entries = the base's 32, one-for-one**. This **corrects the Advisor's pre-flight observation** (Add.57): Rasmussen/Zep (base L113), Mem0/Chhikara (L100), Chakraborttii (L99), Yang (L121) and Uddin (L118) are all **inherited, not new** — they were dropped by the round-1 condensation and have now been *restored*. Nothing here needs a cite-check spoke on ancestry grounds.

- **Base entries DROPPED while their in-text claim survives: NONE.** (32/32 retained.)
- **Metadata thinned (not errors):** DOIs stripped from Aitken, Blelloch & Golovin 2007, BGV 2008, Hochreiter, Micciancio, Minami & Hidaka, Naor & Teague; Buchbinder's journal-version note dropped; Sekhari's arXiv id dropped. Volume/page/venue data otherwise exact.
- ⛔ **One citation is now MIS-STATED (Part-A-serious, since it is the sentence that defines the term this paper must not claim):**
  - pj l.80: *"Formally, certified removal requires strict $(\varepsilon,\delta)$ relaxations (Guo et al., 2020), or complete isolated deletion with associated cost formalisms (Bourtoule et al., 2021; Ginart et al., 2019)."*
  - base L51: *"Formally, \emph{certified} removal is Guo et al.'s (2020) **§2 Eq.~(1), an $\varepsilon$ condition with an unnumbered $(\varepsilon,\delta)$ relaxation immediately after**; exact methods delete by isolation, with cost and capacity formalisms in place (Bourtoule et al., 2021; Ginart et al., 2019; **Sekhari et al., 2021**)."*
  - Two defects: (i) the corrected Guo form (§2 Eq. (1), **ε-only**) is inverted — pj says certification *requires* $(\varepsilon,\delta)$; (ii) Sekhari is dropped from the citation while remaining in the bibliography.
- ⚠ **10 orphaned bibliography entries** (in the bibliography, cited nowhere in the body; all are cited in the base): Andersson & Ottmann 1995 · Blelloch, Golovin & Vassilevska 2008 · Buchbinder & Petrank 2003 · Jude et al. 2023 · Micciancio 1997 · Naor & Teague 2001 · Rusch et al. 2022 · Sekhari et al. 2021 · Snyder 1977 · **Wang B. et al. 2026 (agentic unlearning)**. Cause: the base's prior-art paragraph (L304) and its §2 companions were cut. Consequence ranked: the App-D **prior-art** paragraph is gone from an appendix still **titled** *"Prior Art and Measured Tables"*, and with it the no-priority clause (B.2-4); Rusch orphans while *"learned recurrent models like LSTMs"* survives with only Hochreiter; Wang B. orphans together with the base's *"the store deleting an item is not the system forgetting it"* sentence (finding #1).

## A.4 — Never-quote sweep (per-file, positive-controlled; instrument LIVE)

**Zero-hit list (all 0 in `pj_sub.tex`):** `13.9` · `≈14×` · `we alone` · `CLU-former` · `0 of 5` · `CSF3` · `prior mismatch` · `P=4` · `compositional family` · `residual protects` · `watch stayed green` · `state-of-the-art` · `SOTA` · `best-in-class` · `benchmark win` · `beats` · `wins` · **`outperform` (⭐ round-1 flag FIXED)** · `deletion-compliant` · `0.272` · `right-to-be-forgotten` · `memory provenance` · `companion` · `sibling` · `forthcoming` · `in preparation` · `github` · `zenodo` · `huggingface` · `.claude` · `chlu/` · `PALM` · `Morse` · `Moser` · `cryptographic unlearning privacy` (round-1 coinage, gone).

**Non-zero, adjudicated:**

| pattern | n | line | disposition |
|---|---|---|---|
| `certified` / `Certified` | **2** | 80, 155 | see the per-occurrence table, B.1(a). l.80 = **literature-description form (permitted)**; l.155 = the Guo bibliography title (permitted). ⭐ Round-1's affirmative occurrence is GONE. |
| `unlearning` | 4 | 80, 150, 168, 172 | l.80 *"rather than a stochastic system-level unlearning heuristic"* = contrast/denial form (permitted); l.150/168/172 = reference titles (permitted). ⭐ The round-1 coinage *"cryptographic unlearning privacy"* is gone; l.126 now reads *"rather than robust cryptographic privacy"* — still a phrase with no exact ancestor (base L81: *"retrieval geometry, not privacy"*), but denial-form. ⚠ minor. |
| `(Mo, 2026)` in prose | **1** | 83 | ⛔ **VIOLATION — Add.45(2) / Add.51.** *"strictly equivariant flows possess Lyapunov neutral modes (Mo, 2026)"*. The rule: the token appears **nowhere** in body text, captions, labels or filenames; the bibliography entry (l.161) keeps its authors and is compliant. The base cites the same work in prose as *"a recent preprint on symmetry-protected Lyapunov neutral modes (arXiv:2605.03338)"* — so there is **no ancestor for the named prose form**. Round 1 measured 0 occurrences. **New regression.** |
| `13.88` | 1 | 257 | permitted form (labelled as the **coupled-bath refuted** prediction, ancestor L239) ⚠ but the base's rider *"and never the vault"* is dropped (A2-10). |
| `CHLU` / `Jawahar` / `Pierini` / `Anonymous` | 2/2/2/2 | 56, 146, 157 | **inherited from the base verbatim** (L39, L93, L104) ⇒ not a fidelity defect. ⚠ Standing note: round 1 recorded pj as *stronger* than the base on de-anonymization because these were cut; the rewrite restores the base's two-step exposure (CHLU continuity + the Anonymous theory note). Not scored as a defect — it is the base's own posture — but the Advisor should know the round-1 improvement is reverted. |

---

# PART B — claims fidelity, the round-1 re-adjudication, and what stands unqualified

## B.1 — Round-1 re-adjudication (FIXED / PARTIALLY FIXED / UNFIXED)

### (a) ⛔⛔ The "certified removal" inversion — **FIXED as to the never-quote; the explicit DENIAL has NOT returned**

**Per-occurrence adjudication of every "certified" in `pj_sub.tex` (n = 2):**

| # | line | exact text | verdict |
|---|---|---|---|
| 1 | 80 | *"Formally, **certified removal** requires strict $(\varepsilon,\delta)$ relaxations (Guo et al., 2020), or complete isolated deletion with associated cost formalisms (Bourtoule et al., 2021; Ginart et al., 2019)."* | ✅ **PERMITTED — literature-description form.** The term is attributed to Guo and is not applied to our mechanism. ⛔ But the sentence **mis-states Guo** (A.3): the base's corrected form is §2 Eq. (1), an ε-only condition with an *unnumbered* $(\varepsilon,\delta)$ relaxation after it. |
| 2 | 155 | `Guo, C., … (2020). Certified data removal from machine learning models.` | ✅ **PERMITTED** — reference title. |

**Does any sentence claim or imply certified removal as OUR property?** The adjacent positioning sentence is l.80: *"**We sit functionally between these approaches**: we offer a **store-level, bit-exact structural guarantee** based on canonical geometric placement (Blelloch \& Golovin, 2007), rather than a stochastic system-level unlearning heuristic."*
- **Ruling: no affirmative claim.** *"store-level, bit-exact structural guarantee"* is materially the base's own wording (L51: *"ours is a store-level bit-exactness statement"*), and *"rather than a stochastic system-level unlearning heuristic"* is a denial-by-contrast. Round 1's ⛔⛔ *"Certified removal is proven strictly at the store level"* is **gone. The campaign's most serious claims error is FIXED.**
- ⚠ **Two residues, flagged not resolved:** (i) *"We sit functionally between these approaches"* places us **on the unlearning spectrum by positioning** where the base places us in a **seam** (*"The two literatures leave a seam… our result sits in that seam, one level down"*) — a positioning claim with a weaker ancestor; (ii) the base's clause **"with the encoder excluded"** is dropped from this exact sentence, and (see below) from the entire file.
- ⛔ **The explicit denial is ABSENT from the whole file.** Base L79: *"we claim no certified $(\varepsilon,\delta)$ unlearning"*; L51: *"**We make none of those claims**."* Neither survives, nor any paraphrase. The scope now rests **entirely on contrast** — i.e. on a reader inferring the denial from *"rather than…"*. **Verdict on round-1 item B-2: PARTIALLY FIXED — inversion removed, denial not restored.**

### (b) Store-level deletion with its three conditions and the recency exclusion — **PARTIALLY FIXED**

- **Present, paraphrased, beside the claim (§3.3 l.124):** *"Under explicit conditions (**a budget sufficient for the cell count, zero baseline leakage, and attribute-based eviction**), the physical layout of the store remains byte-identical to a temporal state that theoretically never encountered the deleted item."* ⇒ `budget >= n_cells` ✓, `leak = 0` ✓, attribute eviction ✓ — ⚠ **"priority/" is dropped** from *"priority/attribute eviction"*.
- **Recency exclusion: PRESENT but relocated** — §5 l.135 only: *"recency-based eviction intrinsically breaks history independence and is thus excluded."* ⛔ **Not beside the claim** in §3.3, and **absent from the abstract**.
- ⛔ **The abstract still names none of them:** l.39 *"store-level deletion is exact. Post-deletion, the store is byte-identical to a state that never held the item, **governed by explicitly defined operational conditions**."* — and, **new in round 2, the abstract has lost even the round-1 phrase "on a designed datastore"**: the abstract's deletion claim now carries **no scale, no store dimension, no learning scope, no eviction regime**.
- ⚠ **Compliance question I do not resolve (Advisor's):** CM-25(f) is an *approved wording*. What §3.3 carries is a **paraphrase** of it, not the verbatim form. Whether a paraphrase discharges an approved-wording obligation is a charter ruling, not a curator's.

### (c) The score sentence and the deletion laundering control — **UNFIXED, both still ABSENT**

- `ZERO` = 0 · `headline metric` = 0 · `external benchmark` = 0 ⇒ ⛔ the standing score sentence (*"External benchmarks won on their own headline metric = ZERO"*, charter §4.1, base L79 and L356) **is absent**, and the base's negatives row that carries it is one of the 16 cut rows.
- `trivial` = 0 · `flat table` = 0 ⇒ ⛔ *"A flat table deletes exactly by construction — the trivial substitute"* (base L79) **is absent**. §3.3's exactness claim has **no trivial-substitute control**. (§5's future-work line *"formally assess the amortized per-delete execution cost against flat datastore architectures"* is a future item, not the control.)
- ⛔ Additionally, App D's second laundering control is gone too: the anti-decoration-guard substitute that *"passes $0.672$ of queries with no dynamics at all"* (base L323).

### (d) The Blelloch–Golovin no-priority clause — **UNFIXED (and the attribution regressed at one site)**

- Clause *"We claim no priority over order-independent placement and no novelty for the displacement rule or its delete-time repair"* / *"'Fix-up cascade' is our name for their repair"* (base L304) — ⛔ **still ABSENT**. So is the negatives row *"The placement algorithm is ours — No: Blelloch & Golovin (2007) own it outright"* (base L341).
- **Attribution sites:** abstract l.39 ✓ · §2 l.80 ✓ · §3.3 l.124 ✓ · ⛔ **App D: ABSENT** (round 1 had 4/4; the rewrite's App D says *"the purely canonical stable-matching rule"* with no attribution, in a section titled *"Prior Art and Measured Tables"* that contains **no prior art**). **Regression: 4/4 → 3/4.**

### Round-1 findings, complete table

| round-1 finding | round-2 verdict | evidence |
|---|---|---|
| **A.2-C1 / B-2** "Certified removal is proven strictly at the store level" | **FIXED** (denial not restored ⇒ item B-2 PARTIALLY) | B.1(a) |
| **A.2-C2 / B-1** three conditions + recency exclusion absent | **PARTIALLY FIXED** | B.1(b) |
| **A.2-C3 / B-6** lifecycle reads as *evaluated*, not shipped | **FIXED BY DELETION** — `lifecycle` = 0, `PROTECTED`/`TRASH`/`toy` = 0. No lifecycle claim stands, so the §0.13 riders are moot. ⚠ Contribution (4) is gone: a Head scope decision, not scored | sweep |
| **A.2-C4** the vault *transferring* rather than its laws | **FIXED** — *"Translating this architecture … yields"* is gone; App C l.261 reads *"the **law-referenced** absolute vault computes … to $106.1\pm5.0\times$ against the fundamental prediction of $110.25\times$"* with the bounded-cell restriction and the OU-estimator substitution stated. ⚠ *"6 cells"* dropped; heading still *"Emergent Arm **Translation**"*; both constituent law-fits ($0.9998\pm0.0019$; $1.016$–$1.103$) and the *"never the vault"* record for $297.8\pm196.8\times$ still absent | l.261 vs L286 |
| **A.2-C5 / B-9** confinement lost its equal-$\gamma_{\rm eff}$ scalar control | ⛔ **UNFIXED — and WORSE.** pj l.118: *"this localized field mathematically confines the register: the state fraction outside the boundary ($|\theta|>1$ rad) drops definitively to $0.0000$ within the hole."* Round 1 at least kept *"up to $43.0\%$"*; the rewrite drops **both** the no-hole baseline ($5.5/43.0/2.4\%$) **and** the scalar control ($0.73/10.2/0.26\%$). *The hole confines* — the one vault result with no designed analogue — now has **no comparison arm at all** | l.118 vs L75 |
| **A.2-C6** "four independent **rollout** instruments" | ⭐ **FIXED** — App B l.218–222 now names *"four specific measurement instruments"*: *I-J*, **the exact one-step Jacobian**; I-R1/I-R2/I-R3 rollouts. The internal contradiction with §3.1's Jacobian number is resolved | l.218–222 vs L171 |
| **A.2-C7** a refuted micro-explanation asserted as the finding | **PARTIALLY FIXED → still ⚠⛔.** pj l.237 *"The primary cause … **is confirmed**"* is **stronger** than the base caption's *"Cause …, **measured**"* (L189), and the base's negatives row (L361: *"**Wrong** … the pre-registered single-slow-mode projection **fails**"*) is **not** among the 5 rows kept. The pre-registration failure it belongs with (early-window $0.52$–$0.85$ vs $[0.65,1.35]$, base L203) is also dropped | l.237 |
| **A.2-C8** new market/system assertions with no ancestor | ⭐ **FIXED** — *"fully reconstructible"*, *"significant data leakage vulnerabilities"*, *"production environments require exact, unrecoverable deletion"* all gone; l.80 now tracks the base (*"soft-deleted vectors in graph ANN databases remain reconstructible from raw index files (Chakraborttii et al., 2026)"*) **with the citation restored**. ⚠ Residue: *"exact byte-identity is **crucial for security**"* vs base *"Byte-identity is the opposite design point, and it matters because…"* — mild, ancestor exists | l.80 vs L51 |
| **A.2-C9** *"total information eradication"* / *"completely isolates the target data"* | ⭐ **FIXED** — both phrases gone (`eradication` = 0). ⚠ But the AUC $0.5000\pm0.0000$ evidence went with them: §3.3 now asserts exactness with **no adversarial statistic quoted at all** | sweep |
| **N1** $n_{1/2}$ quoted without $\Delta$ / $\ell_\theta/\Delta$ against the file's own rule | **PARTIALLY FIXED** — ⭐ `\Delta=0.5` rad now appears (App A header l.180, Fig 2 caption l.210; round 1: 0). ⛔ **No $\ell_\theta/\Delta$ VALUE appears anywhere**: the ratio occurs only as a *rule* (l.70, l.180) and as a mention (l.294). The base attaches $\ell_\theta/\Delta<0.05$ to the $3.77\times$ site (L73). The paper still states a reporting rule twice and satisfies it nowhere | sweep |
| **N2** "three regimes … at different $\mu$ values" | ⭐ **FIXED** — l.93 now reads *"three specific operational regimes of a singular unified curve evaluated at **two disparate values of $\mu$**"*. ⚠ The base's *"— not two laws"* is dropped | l.93 vs L61 |
| **A.4** `outperform` / *"cryptographic unlearning privacy"* | ⭐ **BOTH FIXED** (0 hits each) | sweep |
| **C.1** 23 mechanical defects (8 `&`, 15 `\`) | ⭐ **FIXED** — the file compiles clean, zero repairs (Advisor render, Add.57); confirmed by the presence of a complete `pj_sub.aux` with all labels resolved | aux |

## B.2 — The do-not-cut walk (`v5-referee-v02` list), present/absent, ranked by consequence

### ⛔ Tier 1 — a claim now stands without its mandatory rider

| # | item | status | which claim stands unqualified |
|---|---|---|---|
| **1** | **The substrate/encoder scope** — *"These laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, $\varphi$-bytes ledgered"* (§A20.5); and *"This is a store-level guarantee only — the frozen encoder and any residue of past writes in a learned landscape are separate channels"* (L79); and *"the store deleting an item is not the system forgetting it"* (L51) | ⛔⛔ **ABSENT — `encoder` = 0 in the entire file** (base: 5). **REGRESSION vs round 1**, which found it in §1 bullet 2 and §3.3's opening | **Every deletion claim in the paper**, plus the abstract's *"Removal is purely structural"* and §3.3's title *"Absolute Guarantees"*. This is precisely the V5 drift mode "store-level deletion reading as system-level unlearning", and the only three sentences that blocked it are all gone. Partial substitutes: §1 *"We report a fundamental mechanism … not an end-to-end system performance result"* and l.80's *"rather than a stochastic system-level unlearning heuristic"* — neither names the encoder or the learned landscape |
| **2** | **The negatives estate's honesty framing** | ⛔⛔ **INVERTED.** l.286: *"To ensure complete transparency regarding the empirical limitations of our findings, **every negative result observed during evaluation is documented below**."* **5 rows vs the base's 21** (base L327: *"Every negative below is on the record with its measurement; none is dropped"* — true of 21) | A **false statement about the artifact**. Round 1 explicitly credited the old framing for claiming nothing false. 16 negatives are cut, including four that protect surviving claims: the score-sentence row (L356), *"Threshold instruments are usable below $\gamma_{\rm crit}$"* (L360), *"Our microscopic explanation of the instrument offset"* (L361), *"The placement algorithm is ours"* (L341), and *"Amplitude decay reduces an exact adversary's distinguishability"* (L339) |
| **3** | **The score sentence** | ⛔ **ABSENT** (B.1(c)) | The program's honest-posture anchor and charter §4.1 |
| **4** | **The deletion trivial-substitute control** | ⛔ **ABSENT** (B.1(c)) | §3.3's exactness claim; also App D's $0.672$ substitute |
| **5** | **The equal-$\gamma_{\rm eff}$ scalar control on confinement** | ⛔ **ABSENT, worse than round 1** (B.1 table, C5) | *"the hole confines"* |
| **6** | **The designed-symmetry precondition, beside the claim** | ⚠ **PARTIALLY FIXED.** ⭐ §5 l.133 now **states** it substantively (*"The exact $T=0$ retention latch requires precise continuous coset geometries that must be explicitly designed into the architecture"*), curing round 1's dangling cross-reference; App E row 1 and Fig 3's caption (*"falling short by precisely $\sim10^{-3}$ on all emergent seeds"*) carry the evidence. ⛔ **Still absent from §3.2**, where the vault is claimed; and $T^\star\approx3\times10^{-3}$, $1-|\lambda_{\rm coset}|=1.06$–$2.96\times10^{-3}$ and the designed $\le1.1\times10^{-15}$ are gone | §3.2 in full — the base calls this *"the boundary on everything above"* |
| **7** | **App B's instrument rider** — *"both threshold instruments fail below $\gamma_{\rm crit}$, so no rollout $n_{1/2}$ is ever quoted from a threshold there"* | ⛔ **ABSENT**, while the table needing it survives **in full** (caption l.225 carries only $\delta$/48 γ/3 seeds) | The reader sees slope-below values $+0.0725/+0.0688/+0.0455$ beside $-1.0023/-1.0016/-1.0022$ — four instruments disagreeing by ~20× — with no explanation. The mean±sd row is also still missing |
| **8** | **N108's sentence** — *"the store stops answering before it stops leaking"* | ⚠ **PARAPHRASED, not verbatim** (l.126: *"demonstrating that a physically decaying store intrinsically stops answering external queries significantly before the actual structural data stops leaking"*). ⛔ Both riders still absent: the placement scope (*"placement-dependent, quoted for the controller-placed disk"*, base L321) and the exact-adversary denial (*"decay reduces the effect size, not the AUC, so we claim no reduction in distinguishability per se"*, L81) | The N108 claim itself. Mitigating: the retention $0.832$ / $A=0.051$ number is now absent, so the $|c|$-distribution rider has no orphaned number to guard |
| **9** | **The $\sigma_{\rm obs}$-is-our-own-modelling-choice admission** | ⛔ **ABSENT** (`modelling choice` = 0). The App-D table still prints the $\sigma_{\rm obs}=0.1$ row ($0.559$ vs $0.996$), and the caption is now retitled *"The laundering control **validation** execution"* against the base's *"The laundering control **that fires**"* | The $0.437$ separation reads as a real graded protection result — the exact reading the base's caption exists to prevent. ⚠ Re-labelling a control that **fires** as a "validation" is a claim-direction change in a caption |
| **10** | **The $R_{50}$ differentiator's comparator** — *"where a TTL vector store's lookup radius is a constant hard step at $0.75$–$0.77$, independent of age"* | ⛔ **ABSENT**, and now the differentiator's own numbers ($R_{50}$ $1.146\to0.752$, $1.52\times$) are absent too | §3.3 asserts *"physical decay provides architectural retrieval geometry rather than robust cryptographic privacy"* with **no number and no baseline** — the referee's SF-2 failure mode, on the one differentiator the base says needs no adversary model. (App D's $1.135\to0.771$ is the *gated channel*, a different quantity) |
| **11** | **The $\approx11$-decade instrument note** (MF-1) | ⛔ **ABSENT** — *"That low endpoint is the ring-profile probe's resolution on a checkpoint whose Hessian $\mu^2$ is machine zero rather than a spectral mass, so eleven orders is one curve on one instrument"*; §5 also lacks the base's limitation (iv) | §3.1 states *"Hessian $\mu^2\approx10^{-15}$"* (l.93) and *"$\mu^2\in[1.7\times10^{-12},7\times10^{-2}]$"* (l.95) two sentences apart. **The internal contradiction MF-1 was raised to fix is re-created**, unchanged from round 1 |
| **12** | **`fdt` + Newtonian beside the $T>0$ claim** (C-6) | ⚠ **PARTIALLY FIXED.** ⭐ App A's header is restored **including the legacy sentence** (l.180: *"The legacy reference default structure fundamentally violates these thermodynamic laws"*); §1 bullet and §5 item 3 carry it. ⛔ **Still absent from §3.2** | Every $T>0$ number in §3.2 (the diffusion law, the sign flip, the vault) |
| **13** | **The estimator's name on $107.77\times$** (*"$107.77\times$ is the quoted number and travels with its estimator's name"*; first passage reads $86.97\pm2.94\times$, boundary-layer biased) | ⛔ **ABSENT** at both sites (§3.2, App C) | $107.77\pm4.78\times$ travels bare, as in round 1 |
| **14** | **Scale qualifiers** (C-5) | ⚠ **PARTIAL.** Present: §1 *"laptop-scale compute budget"*, §5 *"dimension of 4, hidden size of 64, … laptop-scale CPU across 5 designed and 3 emergent seeds"*, ⭐ *"No larger-scale LLM performance should be inferred"* (a near-restoration of *"scale is a scope choice"*). ⛔ **Absent: the deletion store's scale entirely** — `dim 3` = 0, `capacity 8–64` = 0, `no learning` = 0, `non-learned` = 0 (round 1 had *"designed, non-learned 3-dimensional datastore"*); and **no scale qualifier of any kind in the abstract** | The abstract's three headline claims; and every deletion claim, which no longer states the store is 3-dimensional, capacity-bounded and unlearned. **REGRESSION vs round 1** |
| **15** | **The corrected Guo form** | ⛔ **PRESENT BUT MIS-STATED** (A.3) | It is the sentence that defines "certified" for the reader; it now defines it wrongly |
| **16** | **The anonymization note** (*"Any supplementary or linked material, including code, is anonymized…"*) | ⛔ **ABSENT** | PALM's code-inclusive anonymization rule has no carrier in the artifact |

### ✅ Tier 2 — items that are PRESENT and compliant

| item | status |
|---|---|
| **The honest scope sentence, exactly once** | ✅ **PRESENT, once** — §1 l.61: *"This is not a deployed, large-scale LLM agent memory, nor a generalized system benchmark."* |
| **The recency exclusion** | ✅ present (§5 l.135) — but not beside the claim (B.1(b)) |
| **The TTL laundering control, correctly paired** | ✅ **IDENTICAL** — $0.983$ vs $1.000$, $0.017$ margin, *"against an exact adversary"* (l.126); table rows exact (l.275–277) |
| **The trilemma** | ⭐ **RESTORED** (l.126, cut in round 1): *"exact value fidelity, amplitude-independent address hold, and amplitude-independent read latency. A system may only optimize two."* ⚠ The **compute-adaptive-read dial** is not named anywhere (a named program dial; base L79/L323 leads with it) |
| **No emergent $\sigma_\theta$ ratio** | ✅ compliant **by absence** (as in round 1; the explanatory control $0.4586\pm0.1181$ is also absent, so no record survives of *why*) |
| **The designed-only contrast scope** | ✅ **PRESENT in substance** — App C l.261 states $23.39\pm10.06$ vs the pre-registered $[6.5,9.5]$, the falsifier firing, and the cause (*"the control arm intrinsically delocalizes while the main field functionally adheres to the absorb limit"*). ⚠ The explicit *"the direction, not the number, transfers"* / *"$8.11\pm0.37$ is a designed quantity"* sentence is absent |
| **The k-regime scope clause** | ✅ **N/A** — the erosion study is not in this condensation, so no unqualified chain-length claim exists (same ruling as round 1). Likewise MF-10's arXiv:2503.21536 mis-citation: absent with its host sentence |
| **"right-to-be-forgotten" / "memory provenance" never claimed** | ✅ **COMPLIANT** — 0 hits each |
| **Two-layer status labelling (C-2)** | ⚠ **PARTIALLY RESTORED** (round 1: 0 uses). Present: Fig 1 caption *"(circles, verification…)/(squares, evidence…)"* ✓ base-verbatim; App A *"providing strict verification"* ✓; App B *"establishing rigorous evidence parameters"* ✓. ⛔ Defective: §3.1's *"Cross-Instrument **Verification**"* mislabels an **evidence** (emergent) result; Fig 2's *"validation of verification"* is not a tag; **App C and App D carry no status header at all** (base: verification/evidence and verification) |

## B.3 — Claims table: surviving substantive claims vs their base form

| # | claim (pj) | base form | ruling |
|---|---|---|---|
| 1 | Abstract: *"a $107.77\pm4.78\times$ retention factor **on designed architectures**"* | L32: *"$107.77\pm4.78\times$ designed and $106.1\pm5.0\times$ … on emergent"* | **NARROWER (safe)** ⭐ MF-13(a) stays fixed |
| 2 | Abstract: *"store-level deletion is exact … byte-identical … governed by explicitly defined operational conditions"* | L32 with the three conditions, the recency exclusion, and dim 3 / capacity 8–64 / no learning | ⛔ **WIDER** — names no condition and now no scale either |
| 3 | Abstract: *"Removal is purely structural"* + §3.3 title *"Absolute Guarantees"* | L79 *"a store-level guarantee only"* + encoder exclusion | ⛔ **WIDER / drift mode "store-level → system-level"**, unguarded because `encoder` = 0 |
| 4 | §1: V-curve, $\gamma_{\rm crit}=2\varepsilon\mu$, $\mu^{-2}$ branch, mass-independent floor, $\mu\to0$ corner | L61 | **IDENTICAL** |
| 5 | §1 Nomenclature Def-2 (inertial $M_i$ / spectral $\mu_k^2$; *"Retention claims always utilize $\mu$"*; $\varepsilon$ *"never utilized as a tilt"*) | L45 | **IDENTICAL** ⭐ (drops only the verification/evidence labelling rule and the substrate sentence) |
| 6 | §2: *"We sit functionally between these approaches: … store-level, bit-exact structural guarantee"* | L51: *"We make none of those claims: ours is a store-level bit-exactness statement **with the encoder excluded**"*; *"The two literatures leave a seam … our result sits in that seam, one level down"* | **CHANGED IN KIND (mild)** — a seam becomes a position between two formal families; the denial and the encoder clause are dropped |
| 7 | §3.1: three regimes of one curve at **two** values of $\mu$ | L61 | **IDENTICAL** ⭐ N2 fixed |
| 8 | §3.1: *"we re-measured **the entire curve** utilizing direct nonlinear rollout"*, headed **Verification** | L69 *"Two instruments, one law (**evidence**)"*, 3 emergent seeds | ⛔ **WIDER + status-tag inversion** (A2-5, A2-12) |
| 9 | §3.2: sign flip, friction preserves / temperature erases | L73 | **IDENTICAL** (⚠ 10/10 → "universally"; `fdt` fine print not beside it) |
| 10 | §3.2: *"this localized field mathematically confines the register … drops definitively to $0.0000$"* | L75, with the no-hole and equal-$\gamma_{\rm eff}$ scalar arms | ⛔ **WIDER** — the control that makes it a mechanism claim is gone |
| 11 | App C: *"the **law-referenced** absolute vault computes … to $106.1\pm5.0\times$"* on bounded cells | L286 | **NARROWER/OK** ⭐ C4 fixed (⚠ "6 cells" dropped; "Translation" heading) |
| 12 | App C: contrast $23.39\pm10.06$, falsifier fired, cause = the control | L288, L357 | **IDENTICAL in substance** |
| 13 | §3.3: TTL flag within $0.017$ AUC of decay against an **exact** adversary | L81 | **IDENTICAL** ⭐ |
| 14 | §3.3: *"physical decay provides architectural retrieval geometry rather than robust cryptographic privacy"* | L81, with $R_{50}$ $1.146\to0.752$ vs the TTL radius $0.75$–$0.77$ | ⚠ **NARROWER in wording, unsupported in evidence** — the comparative claim survives with its comparator and both numbers deleted |
| 15 | App B: *"The primary cause of this threshold offset **is confirmed**"* | L189 *"measured"*, contradicted by L361 (*Wrong*) | ⛔ **WIDER** (A2-9) |
| 16 | App D: *"we implement 100 dedicated write/delete functional interleavings and absolute mid-decay continuous deletes … all byte-equal"* | L305 | **IDENTICAL** |
| 17 | App D: *"we define no explicit deletion-cost execution statement"* | L323 (+ *"the amortized-cost experiment has not been run"*) | **NARROWER (safe)**, ⚠ reason changed (A2-11) |
| 18 | App E: *"every negative result observed during evaluation is documented below"* | L327, over 21 rows | ⛔⛔ **WIDER — false as written** (5 of 21) |
| 19 | §5 Limitations (scale · designed-symmetry · thermodynamic scope · deletion scope · future work) | L85 (i)–(vi) + *"Named next"* | **NARROWER (safe)** ⭐ substantially better than round 1; ⚠ missing: (iv) the instrument-offset/eleven-decade limitation, (vi) *"No task-level payoff is claimed"*, the store's dim-3 scope, and *"deletion at $10^3$-item stores"* from the named-next list |

## B.4 — Mechanical inventory

**Page split (exact, from `pj_sub.aux`):**

| block | pages | contents |
|---|---|---|
| Main text | **1–5 (5 pp)** | §1 p.1–2 · §2 p.2–3 · §3 p.3–4 (§3.1 p.3, §3.2 p.4, §3.3 p.4) · §4 Limitations p.5 |
| References | **6 (1 p)** | 32 entries |
| Appendices A–E | **7–11 (5 pp)** | A p.7–8 · B p.8–9 · C p.9–10 · D p.10 · E p.10–11 |
| **Total** | **11** | matches the Advisor's render (Add.57) |

⚠ **Context, reported not judged (the open track question):** PALM = 4 pp short / 9 pp full, references and supplementary excluded. Main text measures **5 pp** ⇒ over the short-track limit, inside the full-track limit. This is a *lower bound* on the compliant length: every restoration this report recommends adds text.

**Figures: `\includegraphics` = 0 in `pj_sub.tex` vs 11 in `submission.tex`.** The file contains **3 `\framebox` placeholders** literally reading *"Figure 1/2/3 Placeholder"*, each carrying a caption (Fig 1's is the base's caption **verbatim**, ⭐ restoring the B-19 scope labels and flag provenance — but describing data the artifact does not contain). ⚠ **A compiled artifact with three visible "Placeholder" boxes is not submittable as-is** — mechanical, ranked separately from claims.

**Mapping — which base figures carried claims that now stand figure-less:**

| base figure | base site | claim it evidenced | status in pj |
|---|---|---|---|
| `fig1_damping_optimum.png` | main Fig 1 | the V-curve collapse (headline) | **placeholder + full caption** — claim asserted §3.1, no image |
| `figB_dlaw.png` | App A | $D_\theta\propto T$, $\propto(2-\gamma)/\gamma$, ratio over 25 cells | **placeholder + caption** |
| `figC_lambda_coset.png` | App B | emergent V-curve un-collapsed; $||\lambda_{\rm coset}|-1|\sim10^{-3}$ | **placeholder + caption** |
| `figB_signflip.png` | App A | **the sign flip** and the temperature lever | ⛔ **dropped, no trace** — the claim survives (§3.2, App A) |
| `figB_massive_vs_flat.png` | App A | the two regimes; $|\lambda|=1$ latch at every $\gamma$ | ⛔ **dropped** — claim survives (§3.1, App A) |
| `fig2_two_instruments.png` | App B | the V-curve on a second instrument | ⛔ **dropped** — claim survives (§3.1 "Cross-Instrument Verification") |
| `fig2_vault.png` | App C | refrigerator + $8\times$ mechanism + $107.77\times$ (with the "not quoted as the vault" note on $86.97\times$) | ⛔ **dropped** — the paper's headline vault claim is now figure-less **and** table-less |
| `figC2_vault_emergent.png` | App C | emergent refrigerator + $\gamma_{\rm eff}^{-2}$ + confinement + first passage | ⛔ **dropped**, together with **both App-C tables** — the emergent vault claim (l.261) has no figure, no table, and no constituent law-fit numbers |
| `figC_register_capacity.png` | App B | the emergent unit has no coset register; capacity $\approx1$–$1.6$ bits | ⛔ **dropped** — the claim survives as App E's negative row 1 |
| `figA1_damping_optimum_full.png` | App B | full-size Fig 1 incl. **the crimson probe-floor tick at $1.7\times10^{-12}$** | ⛔ **dropped** — and it was the annotation nearest to the missing MF-1 instrument note (Tier-1 #11) |
| `figC_Tstar.png` | App B | the crossover $T^\star$ | ⛔ **dropped** — moot: the $T^\star$ claim is not made |

**Tables:** 3 floats (App A budget · App B four-instrument · App D adversary) + 1 unfloated `tabular` (App E negatives) = **4**, against the base's 8. Dropped with their claims: both App-C tables (emergent refrigerator; confinement/hop fractions), the App-B instrument-gap table, the base's second negatives table.
**Typo the Head owes his own file:** l.291 table header reads **"Emprical Result"**.

---

## Findings summary, ranked by consequence

1. ⛔⛔ **`encoder` = 0.** The store-level/encoder-excluded scope is gone from the entire file (regression vs round 1). Every deletion claim, plus *"Removal is purely structural"* and *"Absolute Guarantees"*, now stands with nothing blocking a system-level reading.
2. ⛔⛔ **App E claims completeness over 5 of 21 negatives** — a false statement about the artifact's own evidence base, and a regression against round 1's framing.
3. ⛔ **The certified-removal denial has not returned.** The inversion is fixed (⭐ the campaign's worst claims error is closed) but scope now rests on contrast alone, and the Guo sentence that defines the term **mis-states it** (ε-only → "requires $(\varepsilon,\delta)$").
4. ⛔ **The confinement claim lost both comparison arms** (no-hole and equal-$\gamma_{\rm eff}$ scalar) — worse than round 1.
5. ⛔ **Deletion scale scope deleted:** no `dim 3`, no `capacity 8–64`, no `no learning`, and the abstract's deletion sentence carries no qualifier at all — regression vs round 1.
6. ⛔ **Score sentence · flat-table trivial substitute · $\sigma_{\rm obs}$ modelling-choice admission · anonymization note · no-priority clause · $107.77\times$ estimator name · MF-1 eleven-decade note · App-B threshold rider — all still ABSENT** (unfixed from round 1), and App D's Blelloch–Golovin attribution has **regressed 4/4 → 3/4** in a section titled *"Prior Art"* that contains none.
7. ⛔ **`(Mo, 2026)` in prose** — a new Add.45/51 violation with no ancestor in the base's prose.
8. ⛔ **$R_{50}$ differentiator now has neither comparator nor numbers**; the N108 sentence remains a paraphrase with both riders missing.
9. ⚠ **C-2 status tags partially restored but misapplied** — an emergent result headed "Verification"; "validation of verification"; App C and App D untagged.
10. ⚠ **14 scope/precision/quantifier mismatches on correct numbers** (A.2) plus a pervasive intensifier load.
11. ⭐ **Credit, earned:** zero numeric errors across the whole rewrite including three tables value-for-value · **zero new citations and a 32/32 bibliography restoration** · `outperform` and the "cryptographic unlearning privacy" coinage gone · the certified inversion, the "four rollout instruments" error, the "total information eradication" claim, the market assertions and the N2 μ-value ambiguity all fixed · the trilemma, the negative packing price, the overflow failure, the legacy-default sentence, $\Delta=0.5$ rad and the §5 designed-symmetry precondition all restored · Fig 1's caption restored verbatim · the file compiles clean.

## Open questions / follow-ups / risks

1. **I did not resolve, and flag:** whether a **paraphrase** of an approved wording (CM-25(f)'s three conditions; N108's sentence) discharges the obligation, or whether the verbatim form is required. That is a charter ruling (Advisor/Head), not a curator's.
2. **A base-internal tension I did not resolve:** `submission.tex` L189 (caption) states the partial-write-amplitude explanation as *"Cause …, measured"*, while L361 (negatives) rules the microscopic explanation *"Wrong"*. `pj_sub.tex` inherits the caption and strengthens it to *"confirmed"* while dropping the negative. Both base sentences are the Advisor's to reconcile; I report the direction of the drift only.
3. **No md5, no render** — no shell in this session. The Advisor's post-pass hash verification is the formal close on the edit-bar.
4. `pj_sub_buildcopy.*` (5 files, my round-1 artifact + its build products) are still in `v5-palm/` and are obsolete lineage per Add.57. I cannot delete them under the zero-writes constraint.
5. **Track question, mechanical input only:** main text = 5 pp at default formatting, before any restoration. I do not rule on the track.
6. **No disagreement between an output and the handover was found to flag.** The base, the round-1 report, Add.54 and Add.57 are mutually consistent on every item used — except Add.57's pre-flight note that the rewrite "adds NEW citations absent from the verified bases", which the measurement **contradicts**: all five named works are in the base (A.3). Reported, not resolved.

## Proposed handover updates (for the Hub)

1. **`pj-fidelity-v5-r2` DONE.** Report at `.claude/outputs/pj-fidelity-v5-r2.md`. Zero writes into `NIPSsubmission/`; edit-bar held (tool-provenance, no hash).
2. ⭐ **Part-A answer, round 2: still numerically clean** — zero digit/±/unit/seed-count errors, zero orphan numbers, across a file that grew 2,684 → ~5.2 k words. **Rider loss, not number drift, remains the failure mode of hand-editing** (the round-1 process negative replicates).
3. ⭐ **Part-A citation ancestry: NEW-citation list is EMPTY; bibliography 32/32 restored; no base entry dropped with a surviving claim.** ⇒ **A cite-check spoke is not needed on ancestry grounds.** It IS needed on **one accuracy ground**: the Guo (ε vs (ε,δ)) mis-statement at l.80. Add.57's pre-flight expectation is corrected by measurement.
4. ⛔ **Five items need a Head decision before this file goes anywhere:** (a) restore the encoder-exclusion scope (0 occurrences); (b) fix App E's completeness claim (5 of 21) or cut the word "every"; (c) restore the confinement controls; (d) restore the deletion store's scale scope and the abstract's qualifiers; (e) correct the Guo form and remove `(Mo, 2026)` from prose.
5. **Registry candidates (I wrote to no registry this pass — audit-only task):** (i) *negatives registry* — nothing newly tried-and-failed; (ii) *`future_work.md`* — no new scientific boundary surfaced; (iii) ⭐ **process negative worth recording, second occurrence:** *a full hand-rewrite fixed 8 of 12 round-1 claims findings while introducing 4 new ones (encoder scope deleted, false completeness, prose author token, deletion scale deleted) — i.e. rewrite rounds do not monotonically converge, and a rider-checklist diff must gate every hand-edit round, not just the first.*
6. **`pj-referee-v5-r2` is unaffected** by this report (blind design); it needs the PDF only.

**Git footprint:** none — no tracked file touched, no branch, no commit. Files created: `.claude/outputs/pj-fidelity-v5-r2.md` (this report). `pj_sub.tex` opened read-only.
