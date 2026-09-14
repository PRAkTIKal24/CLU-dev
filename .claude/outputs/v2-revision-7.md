# v2-revision-7 — paper-writer report (V2 → v0.8)

Task + acceptance criterion: close the referee's five blockers + SF-1…SF-8 + NICE items on V2 v0.7 (page budget · bibliography · μ²-orders collision · hermeticity leaks · stale scout marker), typeset the verified bibliography, print the sweeps. **Status: done, with ONE acceptance criterion NOT met and reported as a finding (page budget — see §A).**
**DIAL DECLARATION (echoed):** none — revision pass; no new measurement; no performance claim; no laundering control applies.

> ## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — Hub decision)
> 1. **The 4-pp limit is not reachable from the sanctioned menu.** Main text measured **5 pp + 4 lines ≈ 5.1 pp**, unchanged from v0.7 (3419 → 3413 words). The demotion menu is worth ≈190 words; this revision's *mandatory* additions (SF-3, SF-8, MF-5 scope clause, U1's qualifier, MF-2's in-prose author-year) cost ≈185. **Closing ≈1.1 pp needs ≈850–1,100 further words (25–32 % of the main text) — a Head-level demotion of a lead result or of the headline figure.** Costed menu in §A. **No lead was demoted here (C-10: no self-pruning).**
> 2. **Two editorial changes beyond the literal commission** (both one-pass revertible), flagged in §D: "published" → "recently posted / reported" for the Mo preprint (5+3 sites); the self-cite possessive removed ("Our reference unit" → "The reference unit here"), the mandatory CLU continuity sentence intact.
> 3. **Cite-check reconciliation item 6 is NOT a draft edit and still has no owner:** `venue-follow-up.md` §3 claim-(a) basis is wrong in the program record (arXiv:2503.21536 = Toledo-Marín, Maiti, Fox & Melko; attributes symmetry breaking to hierarchical feature learning, not CD). The draft no longer relies on it; the *ledger* still says otherwise — curator task.
> 4. **V2↔V5 lockstep re-verification** (referee's missing-experiment item 6) is untouched by this pass and still owed: App K.4 gained an instrument note; if V5 carries the same t-lever/v5-gate text it must move in lockstep.
> 5. **Style-file conversion:** citations are author-year *in prose*; there are no `\cite` macros. One pass converts to `\citep` once the venue style file is chosen (bib entries are ready, verbatim in `v2-cite-check.md`).

## What I did (all edits confined to `.claude/papers/v2-short/{draft.md,draft.tex,CHANGELOG.md}`; zero repo/code edits)

### A. MF-1 — page budget: menu applied in full, target NOT met
Applied exactly the referee's priority order, nothing else:
- **(a) §4 ¶2 → 3 sentences + App L pointer.** ¶2 **280 → 216 words**; §4 overall **525 → 471**. Everything demoted is already verbatim in App L (L.1–L.5). CM-21's approved narrow-claim wording and CM-22(b)'s UnICORNN cession are both *retained in §4* (deliberately: §5's "a published symplectic architecture already holds the ground with a gradient bound (§4)" points there).
- **(b) Abstract 249 → 230 words.** Only the (i)/(ii)/(iii) parentheticals were touched, per the menu: the ≈3.2× misprediction factor, the 1.012–1.029 range and the 1.33×10⁻¹⁵ GMOR figure now live only in §3 (with their fine print). Side benefit: the abstract no longer carries a machine-precision claim without its CM-15 ε/δ fine print.
- **(c) §1's duplicated App-M/App-A pointer sentence compressed.**
- ⛔ §2's fine print (a)–(c) **not** thinned (each ≥1 sentence; (c) grew under SF-3).
- ⛔ No further density hand-tuning (referee item (d)): `\documentclass`/geometry untouched.

**Measurement** (`tectonic`, 10pt/0.9in `article` approximation): main text = **5 pp + 4 lines ≈ 5.1 pp**; references start p. 6 and are excluded per the venue rule; PDF 32 pp total.

**Costed menu for the Hub/Head (measured word counts, current draft):**
| block | words | note |
|---|---|---|
| Abstract | 230 | at the menu's floor already |
| §1 | 373 | contributions list is the bulk |
| §2 (setup + two axes + fine print) | 494 | ⛔ fine print (a)–(c) protected (C-6) |
| §3 intro | 38 | |
| **§3.1 budget verification** | **275** | *verification*-grade (C-2); App M §3.1 carries it in full → **cheapest ≈200-word demotion** |
| §3.2 Mo head-to-head | 294 | headline (C-3) — do not touch |
| Fig. 1 caption | 66 | |
| §3.3 designed-vs-emergent + price | 298 | ≈130 words demotable to App M §3.4/H |
| §3.4 baselines + recipe | 405 | ≈150 words demotable to App C / M §3.3 |
| §4 related work | 471 | ¶1 could go to 3 sentences: ≈120 words |
| §5 discussion | 469 | limitations ¶ compressible ≈150 words |
| **Figure 1** | ≈0.45 pp | moving it to App M frees ≈340 word-equivalents but costs the C-3 headline figure |
| **TOTAL main text** | **3413** | need ≈2,300–2,550 for 4 pp |
Text-only demotions sum to ≈750 words (→ ≈4.5 pp); with the figure moved, ≈1,090 (→ ≈4.0 pp). **Head call.**

### B. MF-2 — bibliography: closed
- New **References** section in both files (after §5, before the appendix block). `.tex`: `\section*{References}` + hanging-indent block, small font, excluded from the page count. **33 entries**, each from `v2-cite-check.md`'s verified primary record; preprints labelled (Mo 2026, Iqbal et al. 2026, Di Bernardo et al. 2025, Graves et al. 2014).
- **Rusch & Mishra 2021a (coRNN/ICLR) vs 2021b (UnICORNN/ICML PMLR 139)** disambiguated at **5 sites** (§4 ×2, App L.5(iii), M§4 ×2). Token count verified identical in md and tex (2021a ×3, 2021b ×4).
- System-name-only citations given authors in prose: **EDEN → Karuvally et al. 2025** (×4), **Titans → Behrouz et al. 2025** (×4); **HLW now carries its year (2006)**.
- Usage-check fixes from the cite-check: Kong exponent scoped to its codings + the unverified "address enters as a bias" softened (L.2); Ramsauer conditioned to single head / identity projections / β=1/√D (L.5(i)); UnICORNN "symplectic-Euler" → "structure-preserving (symplectic)" (3 sites).

### C. MF-3/MF-4/MF-5 — closed
- **MF-3:** one *Instrument note* in **App K.4** naming the Jacobian-derived γ-grid probe (B.8) as the source of the 1.7×10⁻¹² endpoint and cross-pointing §3.3/B.1's Hessian μ² ≤ 2.4×10⁻¹⁵ — the "eleven orders" (one measured curve) and "13–14 orders" (Hessian gap) can no longer be read as contradictory. No number changed.
- **MF-4:** the two named leaks fixed (H.3 "paid-access companions" → "future work beyond this paper"; M§3.4 "lattice-scale companions" → "future work at lattice scale") plus "this program" ×2 → "this paper"/the App-G battery. **The semantic pass found three more the token sweep would miss** (App K "standalone companion", M§1 "companion note … evidence-side companion", App D "A companion falsifiable") and **one internal-process leak** (App C "provisional per the Head"). Post-fix: `companion|sibling|our other|this program|the program` = **0 hits in both files.**
- **MF-5:** wiring fix; **zero "pending scout" strings remain** (5 sites replaced: §3.4, App C preamble, App G N5, M§3.5, M§5). Printed form: (a) sharp instance on a known substrate (F&I 2010/2011; Nijkamp 2020), (b)+(c) novel **with the coverage statement and the "absence over the surfaces listed, not proof of none" clause**, (c) with its **k-regime scope clause** vs Decelle et al. 2021 / Agoritsas et al. 2023, and Toledo-Marín et al. 2025 named as the nearest RBM symmetry-breaking neighbour with its non-CD attribution.
  ⚠ **Deliberate deviation from the cite-check's suggested wording:** it proposed printing that both chain lengths "sit on the same side of the model's mixing time". We do not measure the mixing time, so the draft says the sweep **does not resolve** where either sits, and scopes the finding to "frequency-decisive across the two chain lengths we run". Strictly weaker, strictly supportable.

### D. Reconciliation items U1/U13/U14 (I was named owner)
- **U1 (highest priority):** "at least dim(G/𝓗)" now at **all** Mo-theorem sites (§3.2, App A(iv), M§3.2; §4/M§4 were already correct). The lower-bound *reason* is in M§3.2 (main text carries qualifier + pointer, per the page budget).
- **U13:** landscape distortion → **Fischer & Igel 2010** (3 sites); **2011 retained for the CD-gradient bias bound** (M§4) — both records now in the bibliography.
- **U14:** conformal symplecticity → **McLachlan & Perlmutter (2001)**; HLW 2006 kept only for leapfrog/h<2; **Bhatt, Floyd & Moore (2016)** added in M§4 for conformal-symplectic *schemes*.
- **Beyond commission, flagged:** (1) Mo 2026 is a preprint — "a **published** ML lifetime law" was a checkable factual error, so abstract/§1/contributions/§3.2 now read "recently posted" and "his **reported** median 1.013" (3 sites incl. Fig. 1 caption). Iqbal et al. 2026 carried no "published" wording — unchanged. (2) "**Our** reference unit is the CLU…" → "**The reference unit here** is the CLU (Causal Learning Unit), **introduced as CHLU in Jawahar & Pierini (2026)**" — the mandatory continuity sentence is intact (×2 in each file); this only removes the possessive the cite-check flagged as de-facto self-identification. **Both revertible in one pass if the Hub prefers the old rhetoric.**

### E. SF-1…SF-8, N-1/N-2/N-4 — all closed (locations)
SF-1 abstract splice split ("≈15× better **at short horizons**") · SF-2 "≈1 %" → "≤3 % (median-consistent with its reported 1.013)" · SF-3 §2(c) **and** M§2.1 re-fenced on the *tilt-as-a-designed-lifetime-dial construction*, self-broken MLP named an **observed instance, not a dial** · SF-4 𝓗/H convention line in App M's preamble · SF-5 cross-ref convention now "Appendices **A**–L" + A.5's pointer redirected to M§2.1 row 1 · SF-6 three branch names → bare hashes (B.3/B.6/B.9; base commits and B.9's rebase disclosure kept) · SF-7 **N51 added to App G**, cross-pointed to K.6 · SF-8 one clause in §5 **and** M§5 (T², SO(3)→SO(2); "a direction, on which this paper offers no evidence") · N-1 "(Figure 2 = main-text Figure 1)" + "Headline = Figure 2, displayed in the main text as Figure 1" · N-2 A.2 typicality clause (mixed vacua) · N-4 acknowledgement in the double-blind-safe form.
**Guard respected:** A.5's S1 sentence and its inline sign-off flag are **byte-identical** (diff-verified); the flag survives in both files.

### F. Build
`tectonic -X compile draft.tex`: **0 errors, 0 undefined references, 0 overfull/underfull-critical boxes**; `draft.pdf` **32 pp**, 1.08 MiB, 6 figures. Three pre-existing build defects fixed in passing: the A.6 notation table and the M§2.1 / M§3.3 tables converted from `c` to `p{}` columns (kills all three overfull hboxes, incl. the 331 pt one the referee attributed to a branch token) and `\label{fig:mo_own}` added to the Fig. 2b environment (that reference was undefined in the v0.7 build too — the v0.7 CHANGELOG's "0 undefined references" was wrong).

## How I verified
- Every edit applied through an exact-once replacement harness (`.claude/scratch/v2rev7_edit.py`); any string not matching exactly once aborts before writing. **63 md + 67 tex** exact-once replacements (plus one inserted References block per file), all confirmed.
- **md↔tex sync:** token counts identical in both files for all 18 edited-string classes ("at least" 5/5 · "recently posted" 4/4 · "reported median" 3/3 · Karuvally 4/4 · Behrouz 4/4 · McLachlan 3/3 · "Fischer & Igel 2010" 3/3 · Toledo-Mar 2/2 · Decelle 3/3 · Agoritsas 2/2 · 2021a 3/3 · 2021b 4/4 · "Instrument note" 1/1 · N51 1/1 · "future work beyond this paper" 1/1 · "future work at lattice scale" 1/1 · "tilt-as-a" 2/2 · Bhatt 2/2).
- **Page measurement:** `pdftotext` per page; main text ends 4 lines into p. 6 (identical to v0.7); per-section word counts computed from the canonical markdown (table in §A).
- **Never-quote sweep, per-file, end-to-end, positive-controlled** (both `draft.md` and `draft.tex`):
  **ZERO HITS:** CLU-former · certified · unlearning · exact deletion / "the item is gone" · "exact discrete FDT" · "samples Gibbs" · 0.384 · 16.28 · CAFE · C-MAPSS · N-CMAPSS · HEPA · CAMELS · bpc · S_eff · z_hole · 0.99985 · 54.56 · 306.76 · 300.09 · deltanet · ttt_mlp · MUNKEY · 0.4545 · 13.9 · memory vault · 107.77 · compositional · unaskable · 19× · Guo · Ginart · Sekhari · Track A · waitlist · **"pending scout"/"pending a literature scout"** · **paid-access** · **companion/sibling/"our other"/"this program"/"the program"** · **agent/experiment-engineer** · **"per the Head"**.
  **HITS, context-checked, compliant:** "energy units" ×2 (the refusal statements) · "wormhole" ×1 (App H.3's own κ=0 certificate) · "2.6" ×22/23 (2.6e-2 / 2.6e-6 / 2.6e-13 grid+probe values; never the retracted mass-drift number) · "98.3" ×1 (unescaped-dot false positive on `0.98732513581811`) · "critical damping" ×1 and "V-curve" ×4 (App F.2/K.4 only — **0 in abstract/§1/§3**, Q11 intact).
  **Positive controls fired:** GMOR 25/23 · N46 11/11 · N51 1/1 · "introduced as CHLU" 2/2 · Rusch 8/8 · provenance 11/10.
- **Semantic-hermeticity pass result (the standing rule this draft's own history created): "companion / program / sibling / our other" class = 0 hits after adjudication; 6 hits found and each fixed** (2 referee-named + 3 semantic + 1 process leak), listed in §C.

## Findings / open editorial questions for the Hub / Head
1. **The page budget is a decision, not a drafting problem** (reconciliation item 1). The sanctioned menu is exhausted; the residual is ≈1.1 pp. Recommended, in order: demote **§3.1** (verification-grade under C-2; App M §3.1 has it verbatim) → 2 sentences + pointer; then §4 ¶1 → 3 sentences; then §5's limitations ¶ → a scope box. That is ≈470 words ≈ 0.7 pp and keeps every *evidence*-grade result and the headline figure. If the venue template turns out denser than the current 10pt/0.9in approximation, revisit before cutting further.
2. **Keep or revert the two beyond-commission edits?** (the Mo "published"→"recently posted" relabel; the de-possessive self-cite). The first is a correctness fix I would defend to a referee; the second is a partial anonymity mitigation and the Hub still owns the bigger question of whether the self-citation is anonymised at all.
3. **"optimum" appears once in §3.3** ("the damping-optimum retention law itself still generalizes") — pre-existing, untouched, and arguably inside Q11's demotion rule's spirit (it is a *pointer* to App K, not a headline). Flagged so the next referee pass does not read it as a regression.
4. **App C and App K still print "a pending editorial decision"** about placement. Harmless in a draft, but a submission should not carry it; not in this task's scope, so left with the process-language ("per the Head") removed only.
5. No missing-experiment notes generated: this pass needed no number that does not exist, and invented none.

## Proposed handover updates (for the Hub)
- **V2 is at v0.8.** MF-2/MF-3/MF-4/MF-5 and SF-1…SF-8 + N-1/N-2/N-4 **closed**; **MF-1 applied in full but the 4-pp target is not met (5.1 pp, unchanged)** — the referee's demotion menu is exactly worth this revision's own mandated additions. A Head-level demotion (or the venue template) is the only remaining lever; a costed menu is in this report.
- **Bibliography now exists** (33 verified entries, both files, excluded from the page count) and the Rusch & Mishra collision is resolved; citations remain author-year in prose (no `\cite`) pending the style file.
- **Standing rule confirmed by use:** the hermeticity sweep must be semantic. The token sweep would have passed this draft; the semantic pass found 4 further leaks (App K, M§1, App D, App C's "per the Head").
- **Still owed and unowned:** the `venue-follow-up.md` §3 claim-(a) basis correction (curator), the V2↔V5 lockstep check after the J→K relettering, and the `\citep` conversion at style-file time.
