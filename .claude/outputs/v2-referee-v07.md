# v2-referee-v07 — paper-referee report

Task + acceptance criterion: adversarial re-review of V2 v0.7 (post-Q11 re-centering + Appendix-A primer + stale-delta fold + NeurReps-EA venue class); MUST/SHOULD/NICE triage with citations, simulated NeurReps-EA verdict, three hostile quotes, sweep printed, zero edits outside this report.
Status: done. Zero files edited other than this report.

**DIAL DECLARATION (echoed from task):** none — adversarial review; no performance claim; no laundering control applies.

Basis read in full: AGENT_PROTOCOL · Positioning Charter (C-1…C-10, `philosophy-synthesis.md` §581–603) · `critique_register.md` (G1–G7/V*/M*) · claims_matrix v2.17 rows CM-15/CM-16a-b/CM-17/CM-21/CM-22 + §0 headings · shorts charter Add.8–Add.17 · `v2-colleague-physics-review.md` (Add.9 riders R1–R6, S1, T1–T6) · registry N46/N51/N149-N150 sites · `papers/v2-short/{draft.md (565 ln, full), draft.tex (spot), CHANGELOG.md, draft.pdf (pdfinfo), figs/}`.

---

## Verdict (simulated, NeurReps-EA composite reviewer: symmetry/geometry experts, physics-analogy-overreach reflex)

**WEAK-ACCEPT, trending ACCEPT after must-fixes — but the draft is mechanically NOT SUBMITTABLE today** (main text measured 5 pp against the declared 4-pp limit — verified: §4 spills onto PDF p. 5; and the .tex contains **zero `\cite` commands and no reference list at all**). Meta-review: the Q11 re-centering succeeded — the paper now has one sharp organizing claim (a published ML lifetime law is the *overdamped face* of a constitutive mode-mass budget, verified on trained checkpoints, reproduced on the published law's own estimator), the critical-damping demotion is consistent at all five sites I checked, §2.1's SSB discipline survived the merge into §2 intact, and Appendix A is a genuinely useful, correctly-scoped EA on-ramp with all Add.9 riders discharged. The honesty engineering (verification/evidence labels, retirements in App L, compute-inversion disclosure, N149/N150 fence) is the best in the portfolio. What remains are: one genuine M4-class cross-section number collision (the "13–14 orders" vs "eleven orders" μ² gap), two hermeticity leaks the v0.7 sweep missed ("companions"), a five-week-stale "pending scout" marker on the erosion novelty claims, and the mechanical blockers above. None require new experiments at the EA venue class.

---

## Findings

### MUST-FIX (block submission)

**MF-1 — Main text is 5 pp against a 4-pp venue limit (header line 5; CHANGELOG item 6 self-flags it; verified via `pdfinfo`/`pdftotext`: §4 Related work occupies p. 5).** The Hub owns the close; per-item demotion menu, in my priority order: (a) §4 ¶2 ("Retention guarantees elsewhere…", ~330 words) → 3 sentences + App L pointer; App L already carries it verbatim (~0.4 pp). (b) Abstract is ~280 words; ~30% trims cleanly (the (i)/(ii)/(iii) consequence clauses can lose their parentheticals — numbers survive in §3). (c) §1's final sentence duplicates the draft-status block's App-M/App-A pointers. (d) The real NeurReps style file will change density — do NOT hand-tune further before building in the venue template (current build is a 10pt/0.9in `article` approximation, `draft.tex:1,4`). ⚠ Do not close the page by thinning §2's fine print (a)–(c) below one sentence each — C-6/charter requires that fine print adjacent to its claims.

**MF-2 — No bibliography.** `draft.tex` has no `\thebibliography`, no `\bibitem`, no `\cite` (grep: 0 hits); ~25 works are cited inline author-year with no reference list. Compounding: **"(Rusch & Mishra 2021)" is used for BOTH coRNN (ICLR 2021) and UnICORNN (ICML 2021)** in §4 (draft.md:55, 57) — indistinguishable without 2021a/b entries. Venue excludes references from the 4-pp count, so this costs no budget. A cite-check spoke should build the bib (Kong DOI, HiPPO, Jelassi, EDEN, Titans, Csordás & Schmidhuber, Iqbal et al. 2026, Mo 2026 — several were verified in earlier eras but never typeset).

**MF-3 — A reviewer-constructible cross-section contradiction in the designed-vs-emergent gap (the exact M4/CM-7 class this program's MF-3-of-w7 precedent exists for).** §3.3 (line 47) and M§3.4 (line 497): designed flat μ² ≤ 2.4×10⁻¹⁵ vs emergent 5.1–5.9×10⁻² ⇒ "**13–14 orders of magnitude in μ²**". App K.4 (line 343): "a single damping-optimum curve now spans **eleven orders of magnitude in μ²** (1.7×10⁻¹² designed → 7×10⁻² emergent)". Same symbol μ², same architectures, designed endpoint differing by ~3 orders, no reconciling clause. (K.1's "~12-order" gap and §3.3's "15 orders" are fine — each names a *different instrument*: the flat-mode multiplier and Mo's E_eq.) K.4's 1.7×10⁻¹² is evidently the **Jacobian-derived** μ² floor of the B.8 γ-grid probe, not the Hessian μ² of B.1 — one sentence in K.4 naming the instrument and cross-pointing §3.3's Hessian number closes it. Without it, the hostile quote writes itself ("the same gap is 13–14, ~12, 11 and 15 orders depending on the page").

**MF-4 — C-8 hermeticity leaks: the draft assumes other program shorts exist.** (i) App H.3, line 246: "that is the discriminating experiment for **the paid-access companions**, not this paper." (ii) M§3.4, line 513: "is left to **the lattice-scale companions**." Both name unnamed companion papers (V1/V3 material) — violating C-8 ("no short cites **or assumes** another short") and creating the M2 program-linkage surface under double-blind. Also (iii) "this program's models" (A.6 table, line 109) and "claims this program does not make" (App L preamble, line 355) — replace "program" with "paper"/"the theory note". Note: the v0.7 fold's own sweep declared "C-8 hermetic: no reference of any kind to any other short" (CHANGELOG item 1) — it grepped shorts' names, not companion-language; this is the standing lesson that every negative sweep needs a semantic pass, not just tokens. Fix: "…is future work beyond this paper" / "…is deferred to future work at lattice scale."

**MF-5 — The erosion-novelty "pending scout confirmation (Jul-11 novelty check)" marker is five weeks stale and cannot print (App C preamble line 148; §3.4 line 49; M§3.5 line 523; M§5 line 553).** Either the scout ran (then the draft must cite its outcome — wiring fix) or it never ran (then it is the one genuinely outstanding pre-submission task for THIS paper). The claims are correctly hedged today, but a submission cannot carry "pending a literature scout" in print, and un-hedging without the scout would violate the ship-rule the hedge encodes. **Hub-owned; not editable by drafting alone.**

### SHOULD-FIX

**SF-1 — Abstract splice of two non-cross-comparable experiments (line 13).** "a free param-matched twin fits ≈15× better **out to ≈500 steps**, then diverges past a measured ≈700-step crossing" — the ≈15× is `minus-the-physics` (eval-400, wake+sleep, seeds 42–44, commit b41410f); the ≈500/≈700-step horizon curve is `fit-gap-anatomy` (wake-only, seeds 0–2, commit 9a13455). App H's own boxed caveat (line 235) forbids exactly this merge ("the two tables must never be divided into one another"). §3.3 keeps them in separate sentences; the abstract should too: "fits ≈15× better at short horizons, then…".

**SF-2 — Abstract "matches to ≈1%" (line 13) vs its own disclosed range 1.012–1.029.** 1.029 is 2.9%; "≈1%" is a silent round-down of the worst case. Say "to ≤3% (median-consistent with its published 1.013)" or drop the gloss — the parenthetical numbers already carry the claim.

**SF-3 — §2 fine print (c) contradicts §2's own taxonomy three lines above it (lines 31–33; same tension M§2.1 lines 422–424).** "(c) The explicitly-broken cell is **claimed for designed geometries only**" — yet the taxonomy assigns the pseudo-Goldstone cell to "analytic tilts **and the self-broken MLP**" (a learned potential). The intended fence (N149/N150) is on the *tilt-as-lifetime-dial* claim, not on cell membership (the MLP's self-breaking is an observed instance, budget-priced at +12–15%). Reword (c): "the tilt-as-a-designed-lifetime-dial construction is claimed for designed geometries only; the self-broken MLP is an observed instance, not a dial."

**SF-4 — The 𝓗/H rename is split across the document.** Condensed main text uses $\dim(G/\mathcal H)$ (lines 29, 31, 43, 55); Appendix M and Appendix D use $\dim(G/H)$ (lines 168, 421, 426, 456, 465, 501, 537) while $H=T(p)+V_\theta(q)$ is the Hamiltonian in the same sections — the colleague-review T6 collision, now also an *internal inconsistency* between the condensed text and its "verbatim" long form. Cheapest fix: one line in M's preamble ("in this long form the unbroken subgroup is written H; the condensed text writes 𝓗") or harmonize M/D to 𝓗.

**SF-5 — Appendix A sits outside the stated cross-reference convention.** The convention note (line 71) covers "Appendices B–L" only, yet A cites "§2, taxonomy row 1" (A.5, line 103) — the condensed §2 has no table rows (the table is M§2.1). Extend the convention to A, or point A.5 at "M§2.1, taxonomy row 1" explicitly.

**SF-6 — Internal agent branch names in the provenance tables** (`agent/experiment-engineer/f1-gmor-condensate` B.6 line 136; `agent/experiment-engineer/minus-the-physics` B.3; `agent/experiment-engineer/ssb-shell-atoms` B.9 line 142). Workflow-disclosing and anonymity-adjacent under double-blind; replace with bare commit hashes ("code @ 9bc2cf7") before submission. (Also the known overfull hbox is one of these tokens.)

**SF-7 — N51 is an authority of this revision (CHANGELOG registry line) and its substance is load-bearing (K.6's instrument caveat, K's boxed warning) but it carries no registry tag anywhere in the draft** — App G's negatives list (N4/N5/N46/N149-N150/N6/N12–15/N22/N19) omits it. C-9 wiring: one line in App G ("N51 — raw n₁/₂ exponents are not designed-vs-emergent discriminating; Appendix K.6") or tag K.6.

**SF-8 — Named future work omits symmetry generalization — the first question this venue's audience will ask.** §5/M§5 name dim ≳64, N≥8 units, wider model family, control wrapper, (μ,γ,T) cube, N149/N150 re-test — but not one word on groups beyond abelian SO(2) (torus T², SO(3)→SO(2)), for which the program already holds derived predictions (G7a). The G7 discharge-to-longs ruling governs *evidence*, not the future-work list; Q5/Q9 explicitly permit naming directions. One clause costs zero page budget and pre-empts the G7 reflex at a symmetry venue.

### NICE

**N-1 —** Long-form dangling figure numbers: M§3.2 line 456 "(Figure 2, seed error bars)" — no Figure 2 is displayed anywhere (it is main-text Figure 1; the note at line 469 covers it, but an inline "(= main-text Fig. 1)" at each M-mention is kinder); "Figure 2b" exists with no Figure 2.
**N-2 —** A.2's two-case dichotomy is still non-exhaustive (colleague-review S4: mixed vacua r=0 *and* a ring omitted). Not a mandatory rider; A.5's V=r⁴ example partially covers. One word ("typically") would do.
**N-3 —** A.5's inline sign-off flag (line 103) is correctly present; the Head→colleague relay is already tracked (Add.17) — listed here only as a submission gate, not a new finding.
**N-4 —** "Adapted from a co-author's tutorial note; acknowledgement placement pending" (A preamble) — resolve before print; harmless under double-blind.

### Compliance record (checked item by item)

- **C-1** ✓ no audit confession; App G caveats stated as neutral class theorems, no legacy numbers. **C-2** ✓ verification/evidence labels on every result incl. figure captions. **C-3** ✓ ML-first; Mo leads (Fig. 1 headline). **C-4** ✓ EFT-of-memory in horizon only ("we state them as horizon, not result"). **C-5** ✓ scale qualifiers in-sentence throughout; grep for scope-free "CLUs provide/the lattice scales": 0 hits. **C-6** ✓ (BIBO compact-sublevel-set bracket inline; GMOR precision fine print adjacent; no certificate language to audit — V2 has none). **C-7** ✓ App B is exemplary (B.1–B.9, incl. B.9's disclosed rebase and seed-labelling). **C-8** ⚠ MF-4. **C-9** ✓ App G + SF-7. **C-10** ✓ Appendix M realizes condensation-by-relocation.
- **CM-15** ✓ (F=√M_ch·r* vs Σ=r* nomenclature at §3.1 and J.1; machine-precision claim carried with the mandatory ε/δ fine print, "2.2e-16 relative" correctly forbidden and absent). **CM-16a/b** ✓ split respected — never cited as one claim; K leads with the general face; designed-only register + N46 rider travels (App A scope note (i), §3.3, K.1, App G); Δ=0.5 and ℓ_θ/Δ discipline present (B.8, K.3, K.6). **CM-17** ✓ (App G: Prop-9′ with kinetic-mode qualifier; "never assert a relativistic unit has no equilibrium" stated; fdt+Newtonian flag boxed atop K; no d=1 table quoted). **CM-21** ✓ all four retirements printed (L.1–L.4), HiPPO challenge with the honest no-answer, Kong cited, approved replacement wording used with "actively deletable" omitted and "no deletion claim at all" stated (M§4) — the omission is already flagged to the Hub. **CM-22(b)** ✓ UnICORNN occupancy ceded.
- **Q11 residuals:** 0. No "critical damping"/"V-curve"/"optimum" in abstract/§1/§3; "deliberately not a headline" consistent at all 5 sites (§1, F.2, M§1, M§2, M§3.1); abstract's crossover is the exceptional point, correctly a different object.
- **N149/N150 blast-radius completeness (task item 4):** I hunted for an unfenced tilt/ε/lifetime sentence and found none — §2(c), §5 horizon, M§2.1, M§5, A.7, App G entry (with both reading rules and the Γ/2α ceiling), B.9 all carry the fence; the only residual is SF-3's *wording* tension, not a missing fence. The ε-vs-ϵ disambiguation rule is printed (App G rule (a)).
- **Add.9 Appendix-A rider audit:** boxed scope note discharges R1–R4 in one place ✓; renames s/X/θ₀/𝓗 complete inside A (stray-letter grep: every α/J/ω/H in A is the notation-map's rejected-letter column or 𝓗) ✓; T1–T5 all applied ✓; whitening section A.4 ✓ (Sylvester + rotated null direction); channel-vs-unit ✓ (A.1); S1 corrected form with V=r⁴ counterexample + inline sign-off flag ✓; §2 controlling-definition deference stated twice ✓; A carries no new number and no claim ✓. **Appendix A as this venue's reviewer sees it: value-adding, correctly scoped, not padding** — and it is in the appendix, where its ~1.5 pp are free.
- **Add.10 F3 false friends:** N12–N15 "composition/placement" untouched ✓; "compositional" 0 hits ✓.
- **Q5/Q9/scale:** §5(i) scope-choice + named future work ✓; no C3-era number (CAFE/CAMELS/N-CMAPSS/bpc/margins: 0 hits) ✓; §0.1 score sentence present ✓; §A20.5 substrate sentence present in the paper's own voice ✓ (×2 = main + retained long form, consistent with the fold's 2/2 control).

### Never-quote sweep (per-file, positive-controlled — my own, independent of the fold's)

Files: `draft.md`, `draft.tex` (each swept separately; identical results, as expected for a synced pair).
**ZERO HITS (both files):** CLU-former · certified · unlearning · exact-deletion / "the item is gone" · "exact discrete FDT" · "samples Gibbs" · 0.384 · 16.28 · CAFE · C-MAPSS · N-CMAPSS · HEPA · CAMELS · bpc · organizer swap · S_eff · z_hole · 0.99985 · 54.56 · 5456 · 306.76 · 300.09 · deltanet · ttt_mlp · gdn2 · 1/11 · MUNKEY · 0.4545 · 13.9 · memory vault · 107.77 · compositional · unaskable · 19× · 9–10× · Guo · Ginart · Sekhari · Def. A.5 · "CM-"/"Cor-13" registry tokens in prose · Track A/B · run-3 · waitlist/AUC.
**HITS, context-checked:** "energy units" ×2 = the *refusal* statements ("T is not in energy units" under legacy — compliant); "wormhole" ×1 + "paid-access" ×1 (H.3) and "lattice-scale companions" ×1 (M§3.4) = **MF-4, non-compliant**; "this program" ×2 = SF-6-adjacent wording; "2.6" hits are 2.6e-2 grid bounds / table values, never the retracted mass-drift number ✓; "≈4× longer" appears only inside its own retirement statements (App I ×2, M§3.3 with the caveat in-clause) ✓; "98.3" = unescaped-dot false positive on "0.98732…" (verified).
**Positive controls fired:** GMOR 25 · N46 11 · N149 8 · "introduced as CHLU" 2 · "Mo (2026)" 5 · "provenance" 11.

---

## Missing-experiment list (for the Hub — welcome findings under the quality-first posture)

1. **Erosion-novelty literature scout** (MF-5) — the only item that *blocks* this paper; a scout, not an experiment. Determine whether it already ran (venue-follow-up sub-task 5, ≥Jul-11) and wire the outcome in.
2. **Bibliography + cite-check spoke** (MF-2) — several citations (Kong DOI, Iqbal 2026, Titans, EDEN) have never been primary-source-verified for this draft's usage.
3. **Venue-template build** — true page count in the actual NeurReps style file before any further hand-pruning.
4. **HiPPO timescale-invariance micro-experiment** (answers §4/L.1's self-declared open question): dilate the input timescale on the trained units, measure retention-law covariance vs HiPPO's exact equivariance. Cheap (probe-only), directly defangs hostile quote #2. Task candidate, not a blocker at EA class.
5. **Width-matched per-step compute row** (App I's flagged confound): FLOP/wall for a width-matched CLU — measurement-only, no training; converts a flagged confound into a closed one.
6. **Cross-draft lockstep check V2↔V5** — the shared t-lever/v5-gate material moved from App J to App K in the relettering; V5 v0.1 predates it; the standing "numerically identical, move in lockstep" rule needs a verification pass before either ships (M2: a reviewer reading both must find zero contradictions).
7. *(Venue-risk note, Head-ruled to the longs, recorded only)*: one T² cell would be the cheapest possible beyond-SO(2) evidence if the Head ever re-opens G7 for the shorts; SF-8's future-work clause is the zero-cost version.

## The three sentences a hostile reviewer would quote

1. *"Every number in this paper is dim 4, hidden 64, ≤5 seeds, single-unit, laptop-CPU"* (§5(i)) — quoted as self-indictment of scale, with "extended abstract, scope declared" as the only shield.
2. *"a reviewer is entitled to ask why a learned mass spectrum should be preferred to provable timescale invariance, and we have no measurement that answers it"* (§4) — the paper's own open flank, honestly stated and therefore quotable.
3. *"Designed symmetry protection beats emergent protection by 13–14 orders of magnitude in µ²"* (§3.3) set against *"a single damping-optimum curve now spans eleven orders of magnitude in µ²"* (K.4) — the manufactured-contradiction quote MF-3 exists to prevent.

## Proposed handover updates (for the Hub)

- V2 v0.7: referee verdict **weak-accept→accept at NeurReps-EA after fixes; not mechanically submittable** (5 pp vs 4; no bibliography). 5 MUST / 8 SHOULD / 4 NICE. No new experiment required at the EA class; the one hard blocker outside drafting is the stale erosion-novelty scout (MF-5 — confirm whether it ever ran).
- The v0.7 fold's C-8 sweep produced a false all-clear ("no reference of any kind to any other short") — token grep missed semantic references ("the paid-access/lattice-scale companions", H.3/M§3.4). Suggest a standing rule: hermeticity sweeps include a companion/program-language pass, not just short names.
- MF-3 is a new instance of the M4 cross-section class (Jacobian-μ² vs Hessian-μ² endpoints, K.4 vs §3.3); one reconciling clause closes it — candidate for the same revision spoke that closes the page budget.
- V2↔V5 lockstep re-verification owed after the J→K relettering (item 6 above).

## Flags

- ⛔ **MF-1…MF-5 block submission** (page limit · no bib · μ²-orders collision · hermeticity "companions" leaks · stale novelty-scout marker). All are drafting/wiring-class except MF-5 (Hub decision/scout).
- ✅ Q11 re-centering verified clean at every checked site; Appendix A rider audit passes in full; N149/N150 fold found complete (no missed lifetime/ε sentence); all §0.x sweeps zero-hit except the context-compliant and MF-4 items listed.
- ⚠ A.5 S1 wording still awaits the colleague's sign-off (tracked, Add.17); submission-gated, not a new finding.
