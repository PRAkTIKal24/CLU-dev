# v1-revision-2 — paper-writer report

Task + acceptance criterion: close the v1-referee SHOULD-FIX punch-list F1–F7 on `.claude/papers/v1-short/` (F3 via the LANDED `v1-certificate-payoff` PAYOFF A embed); PDF builds; diff-summary. MF-1 out of scope (Head critical path).
Status: **done** — F1–F7 all closed; F3 closed via the *strong* branch (measured), not the referee's soften-option (b). N-a (Fig 3 → appendix) deferred per task item 8. PDF builds clean.

---

## What I did

Edited in place: `draft.md` (canonical), `draft.tex`, `CHANGELOG.md` (v0.3). Added 3 new figures generated from measured artifacts; no source figure regenerated from a re-run.

| item | closure | where |
|---|---|---|
| **F1** headline occludes hero | **NEW Fig 1 = 2 panels.** (a) reach, arms **vertically offset** (+0.024/+0.008/−0.008/−0.024) so wormhole/router/Newtonian are individually visible, with a **per-arm `det J` receipt column** added to the §3.2 table AND the legend; (b) the latch-payoff scatter (slope-1 transport vs slope-0 erasure) — the one plot where wormhole ≠ router is *visually* representable, exactly as `v1-certificate-payoff` recommended | `fig1_certificate.png`; §3.2 + caption |
| **F2** noise wall in no figure | **NEW Fig 2 = 3 panels**, (c) = **THE NOISE WALL** (gate solid vs Hopfield dotted, σ∈{0,0.3,0.6} × kv{32,64,96}, gate error bars, "fidelity ≈1.0 throughout" annotation) | `fig2_regime_map.png`; §4.3 |
| **F3** "beats the router" definitional | **NEW §3.2.1 "The receipt cashed out"** — measured. Router `det J = 0` *exactly* (autodiff; was "undefined by construction"); erases the latch (std(Q_out)=0, 16/16 bit-identical) where the wormhole transports it (std(Q_out)=std(Q_in)=0.0803). Fine print inline. Also a **new contribution #3** on page 1 | §3.2.1, contribution 3, abstract |
| **F4** §5 MH wall-of-text | Main text → **4 numbered design-rule items**; full derivation → **NEW Appendix F** (F.1–F.6, fully written per C-10) | §5, App F |
| **F5** BIBO caveat not in main text | **New §3.1 paragraph** "BIBO fine print, stated with the wormhole claim (C-6)" — and it is now *measured*, not asserted | §3.1; App B.2 + Fig 4 |
| **F6** 4.8× unscoped | Named **kv16**; decay stated (**1.57× @kv24, 1.14× @kv32**); surviving invariant stated ("never pays full price for less accuracy at any level") | §4.1 pt 2, contribution 4 |
| **F7** internal §-number in figure title | Old PNG retired from the paper. New Fig 1a title = "Reach: who lands — and with which receipt"; caption states **the observed edge is d≈3.2, the bracket, not L=2.5** | Fig 1 caption |
| N-c (referee NICE, cheap) | One framing sentence after the contributions owning the boundary/negative shape ("three of our six contributions are boundaries or negatives; §4 is the paper's honest perimeter") | §1 |
| N-a (task item 8) | **Deferred**, as instructed. Fig 3 stays in main text; caption tagged "*(Candidate for the appendix at the pruning pass.)*" | Fig 3 |

**Structural adds:** contributions 5→6 (split #2 into "stack verified" + "receipt has a measured consequence"); App **B.2** (BIBO battery + Fig 4); App **C.1.b** (latch cloud, all arms + replication); App **A.5** (payoff/BIBO flag-provenance); App **D** +3 payoff fine-print negatives; App **F** (MH kernel).

## How I verified

- **Figures built from measured artifacts only** — `.claude/scratch/v1-revision-2/make_figs.py` reads `.claude/outputs/v1-certificate-payoff/paid_access_metrics.json` (Figs 1, 4) and the `regime-remap-2000ep` §Item-1/Item-1-cont tables (Fig 2). No number recomputed, smoothed, or invented. Script prints its inputs:
  - `detJ: {wormhole_coset_tangent: 1.0, wormhole_across_coset: 1.0, random_shift: 1.0, no_physics_router: 0.0}`
  - `Q_out_std router: 0.0 | wormhole across: 0.08028798550367355 | Q_in_std: 0.08028798550367355`
  - `router Q_out unique: [1.5]` (all 16 identical)
  - `ratio r*(2T)/r*(T) blind: {3.6: 1.959, 4.0: 1.936, 5.0: 1.912}` → matches the report's 1.96/1.94/1.91 ✓
- **PDF build:** `tectonic draft.tex` → `draft.pdf` (822,800 B). **0 undefined references.** Overfull hboxes reduced to 2, both sub-3pt (2.78pt, 1.17pt — invisible); fixed the 3 I introduced/aggravated by `\footnotesize` + column re-width on Table 1.
- Could **not** render PDF pages for visual check (`pdftoppm`/poppler not installed on this machine). Figures were inspected directly as PNGs and iterated: Fig 1a legend/annotation collisions fixed (2 passes), Fig 2c annotation collision fixed, Fig 4 annotation moved off the certified curve.
- **Compliance greps:** zero occurrences of "beats a no-physics router" / "no volume certificate" / "det J undefined"; zero positive-form CM-3 claims (the 2 hits are the mandated disclaimers); placeholders `[WORKING TITLE: …]` / `[AUTHORS PLACEHOLDER]` intact; `inertial mass` vs `spectral mass` both present, no bare "mass"; `CLU … introduced as CHLU in Jawahar & Pierini (2026)` present in both md and tex; hermetic reference list unchanged (no new citations needed).
- Figure-file references in md and tex both resolve to files on disk.

## Findings / results (what the revision now asserts, and on what evidence)

**F3 wired at the honest altitude (CM-7 v1.6 verbatim shape).** §3.2.1 makes exactly the approved statement — *"the wormhole reaches with a receipt the router cannot supply, and the receipt has a measured consequence"* — plus the three qualifications the payoff report demanded, all in main text, none appendix-only:
1. **Volume alone is not the latch receipt** (random shift: `det J=1`, ΔQ std 0.2465, out-spread 0.2793 = 3.48× in-spread). The *matched channel* is what preserves Q.
2. **This is a designed-testbed, oracle-placement, mechanism-level violation — not a learned-system win.** §3.2.1(ii) explicitly restates the §4.2 boundary *inside the payoff paragraph*: "the router is cheaper and more accurate at *choosing* where to go, and it destroys phase-space information when it *goes* there." This defuses hostile-quote #1 without contradicting CM-7's routing verdict.
3. **PAYOFF B carries its narrower attribution** ("the receipt buys BIBO, not the jump; `wormhole_blind ≡ no_physics_router` exactly") and is therefore placed in App B.2 + Fig 4, **not** the headline — per the payoff report's "if V1 embeds only one, embed A."

**C-2 grading held.** All payoff/BIBO results are labeled **verification** (designed analytic testbed), never evidence. App B.2 opens with the grade tag. The word "evidence" remains reserved for §4's trained memories.

**C-6 placements now complete.** The referee's one placement gap (BIBO-coercive-exit caveat appendix-only) is closed *and upgraded from asserted to measured*: b=5.0 has ΔH = 0.0 (free, cheaper than the admissible b=3.0's 2.88), an energy-only sub-level test **admits** it, and it escapes with r*∝T (growth 1.91) while the screened exit saturates (1.000).

**F6's honest answer is a shrinking payoff, and the draft says so.** 4.81±0.44× @kv16 → 1.57±0.07× @kv24 → 1.14±0.06× @kv32. The draft now leads with the invariant that survives all three levels rather than the kv16 headline.

**C-7 cross-section defense strengthened.** New A.5 table + an added sentence explaining *why* App B.2 uses γ=0.02 where §3.2 uses γ=0 ("a bounded arm must be able to settle; at γ=0 a conservative orbit never converges, so 'bounded' would be untestable") — pre-empting the obvious "your γ differs between sections" reviewer path.

### ⚠ Two number discrepancies found between `v1-certificate-payoff.md` and its own artifact JSON (NOT improvised around — flagged, per rule 3)

1. **Router exit charge.** The report's prose says *"all 16 land on `Q = 1.475`"*. `paid_access_metrics.json` says **all 16 land on `Q = 1.5` exactly** (`sorted(set(Q_out)) == [1.5]`; and `mean(Q_in)=1.4621 + dQ_mean 0.0379 = 1.5000`). Both sources agree on the load-bearing facts (`std(Q_out)=0`, `max|Q_out − Q_out[0]| = 0.0`). **I quoted only the load-bearing form** ("all 16 states exit at one common charge, bit-identical") and never printed 1.475 or 1.5. → **Hub: please have the number corrected in the source report** (likely a stale copy from a different cell), or confirm 1.5.
2. **BIBO `r*` table horizon.** The report's `r*` table (102.13 / 103.43 / 104.83) is the **`r_star_2T`** row, though the table header reads `r* = max_t‖q_t‖` with `T = 2000`; the `r_star_T` values are 52.12 / 53.43 / 54.83. The figure axis in the payoff PNG correctly says "over 2T steps". The draft's App B.2 table is **explicitly labeled "$r^\ast$ at $2T$"** and states the T-values in a parenthetical. No number changed. → **Hub: source-report header is ambiguous; the draft is now unambiguous.**

Neither affects any claim.

### Editorial questions for the Hub / Head

1. **Fig 3 in main text.** Task item 8 says defer N-a to the pruning pass, so Fig 3 stays. But V1 is now a **4-figure** short (1, 2, 3, 4) in a 4–5pp budget. Fig 4 is appendix-only, so main text carries 3. Confirm that's acceptable until pruning, or authorize moving Fig 3 → App C.4.b now (its caption already carries the deferral tag, and it is physically located next to C.4.b's table in `draft.md`).
2. **Contribution count grew to 6.** Splitting old-#2 into "stack verified" + "receipt has a measured consequence" is what makes the F3 payoff visible on page 1, but 6 enumerated contributions is long for a workshop short. Alternative: merge back into a 2-sentence #2. My recommendation: keep 6 until the pruning pass — the payoff *is* the answer to the referee's sharpest quote and deserves its own line.
3. **Stale figure files retained** (`paid_access_reach.png`, `fig_regime_map.png`) per C-10 "never delete." Neither is referenced by md or tex any more. Confirm they should stay in the paper dir.
4. **MF-1 unchanged.** `Anonymous (2026)` still placeholder in References; §3's Prop-A2/A6/12 still uncheckable by a reviewer. Not mine to fix; still the submission blocker.
5. **App F length.** The MH-kernel derivation is now ~800 words in an appendix. Per C-10 I expanded rather than pruned (it is V3/ICLR feedstock). Confirm.

### Positioning prose reused (attribution per instructions)

- Related-work / genre-map framing in §1 + §5 "Prior-art honesty": unchanged from v0.2, originally condensed from the internal prior-art sweep (`scout-adaptive-compute-prior-art.md`), as the existing footnote `[^scout]` records.
- §3.2.1's recommended wording and the two fine-print clauses: lifted from `v1-certificate-payoff.md` §"Proposed handover updates" (its suggested §3.2 paragraph) and CM-7 v1.6.
- The "own the shape" framing sentence (§1): from `v1-referee.md` N-c.

## Git footprint

**None.** No tracked code touched. All edits under `.claude/papers/v1-short/` and `.claude/scratch/v1-revision-2/` (both gitignored). No branch, no commits. The payoff code itself lives on `agent/experiment-engineer/v1-certificate-payoff` @ `27f232f` (not mine, not merged) — cited as provenance in App A.5.

Files written: `draft.md`, `draft.tex`, `draft.pdf`, `CHANGELOG.md`, `fig1_certificate.png`, `fig2_regime_map.png`, `fig4_bibo.png`, `.claude/scratch/v1-revision-2/make_figs.py`.

## Open questions / follow-ups / risks

- **Risk (low):** Fig 1a's vertical offsets are a presentational device. The caption says so explicitly ("arms are offset vertically for visibility — wormhole, router and Newtonian control all land at exactly 1.0") and the y-ticks stop at 1.0, but a hostile reviewer could still call it a distorted axis. The alternative (markers only, no offset) was worse — the markers still overlapped. Hub may want a second opinion.
- **Risk (low):** Fig 2c's σ=0 column is the corr=0 capacity-axis row (**n=8**) while σ∈{0.3,0.6} are stress-axis rows (**n=5**); the source has no σ=0 row *within* the stress axis. The caption states this mismatch explicitly. If the Hub prefers, drop the σ=0 anchor point and plot only {0.3, 0.6} — at the cost of the visual "wall."
- **Unresolved:** the referee's MF-1 (theory note live) — untouched, still blocks submission.
- The payoff's **coercive screen is itself an oracle** (analytic x_b). This is now stated three times (§5 horizon bullet, App B.2 item 6, App D payoff-scope bullet). It is the natural next experiment and I flagged it as such rather than letting it read as a shipped capability.

## Proposed handover updates (for the Hub)

- **V1 status: v0.3, F1–F7 CLOSED.** Referee verdict was "weak-accept, contingent on MF-1 + SHOULD-FIX craft pass." The craft pass is done, and F3 was closed on the *strong* branch (measured, not softened), which the referee flagged as "the single highest-leverage addition… the only thing that would lift the paper above weak-accept." Recommend a short re-referee pass on §3.2.1 + Figs 1/2 only.
- **Correction needed in `v1-certificate-payoff.md`:** router exit charge is `Q = 1.5` (JSON), not `1.475` (prose); and its BIBO `r*` table is the 2T row, not T. Neither changes a claim. Suggest the doc-curator patch both, since CM-7's evidence pointer resolves to that report.
- **Claims matrix:** CM-7 v1.6's payoff wording is now instantiated verbatim in V1 §3.2.1 with all three fine-print clauses. No matrix edit needed. Suggest adding a note that the *approved* form of the router's volume statement is "`det J = 0` exactly (measured)" and that **"carries no volume certificate" is now deprecated as an understatement** — V2/V3 must not use the weaker phrasing if they ever touch this.
- **Negatives registry (C-9):** three new payoff fine-print negatives are written into V1 App D but carry **no N-numbers** (I did not invent registry IDs). They are: (a) volume-alone ≠ latch receipt; (b) receipt-buys-BIBO-not-the-jump / router ≡ blind wormhole; (c) payoff scope = designed testbed + oracle screen. Suggest the doc-curator assign IDs (N37–N39?) and back-fill `.claude/negative_results.md`.
- **Pruning-pass docket for V1** (do not act now): Fig 3 → App C.4.b; §5 design-rule list → possibly 3 items; App B.2 items 4–6 are the compression target; contributions 6 → 5.
