# v1-revision-3 — paper-writer report
Task + acceptance criterion: close the three MUST-FIXes (MF-A/B/C) + fold CM-14 that dropped V1-short from weak-accept to borderline (v1-referee-2); nothing that works disturbed; PDF rebuilds clean.
Status: **done.** All four items landed in `draft.md` (canonical) and mirrored in `draft.tex`; PDF rebuilt clean.

**MF-B option taken (task asked me to report this): the PREFERRED option — restate the theorem as a PRICING LAW. No harness run.** The referee's ζ arithmetic is derivable analytically from the paper's *own already-verified* bracket formula (bracket `[L, L+p₀ sinh ζ/M₀]` is `paid-access-experiments` L42/L93, referee-confirmed "exact ✓"), so no new experiment was needed to make the claim data-bounded rather than grid-bounded — I present ζ=2.0105 (d=4.0) and ζ=2.6441 (d=5.0) as analytic predictions from the verified bracket, explicitly labelled "not new measurements." I re-verified the arithmetic independently (`python3`, below). Pricing law is stronger, on-thesis ("paid access"), and matches CM-12's bracket framing.

## What I did (per acceptance item)

### MF-A — decision ≠ transport, disentangled and stated once, sharply
The §3 det-J=0 object was fused with §4.2's learned router. Fix:
- Renamed §3's arm from "(physics-free/no-physics) router" → **"state-replacing map"** everywhere in prose (abstract, contributions 3, §3.2 table + arm-(iv) + Fig 1 caption + line-91 two-panel note, §3.2.1 heading + "The violation" + "The claim", App C.1 note + C.1.b table, App B.2 table + Fig 4 caption, App D Payoff-B fine print). Code arm name `no_physics_router` retained in provenance/appendix tables with a clarifying parenthetical.
- **New sharp lead ¶ in §3.2.1** ("Decision is not transport — state this once, sharply"): §4.2's `router_mlp` is a *learned decision head* that routes *through* the wormhole's own det-J=1 edge (`v1-router-baseline` L17: "routes via the **same** direct wormhole edge") and inherits the receipt; §3's object is an untrained analytic map that *is* the transport and is state-replacing. Verbatim crux: **"A learned gate bolted onto a certified channel is not a counterexample to the receipt; it is a consumer of it. The certificate prices only transport, never the decision."**
- Deleted "reaches the same targets **more cheaply**" from §5 (it imported a §4.2 MQAR FLOPs number onto §3's analytic constant map, for which no cost is measured).
- Rewrote the worst conflation — §3.2.1 "The claim we make" qual (ii): the old text said "the router is cheaper and more accurate at choosing … and it destroys phase-space information when it goes there" (treating the two objects as one). Now: the erased object is the untrained state-replacing baseline; §4.2's router transports through the certified channel and erases nothing; both facts hold because decision ⟂ transport.
- Left the 4 legitimate "physics-free learned router" mentions (§4.2 main text, C.2, App D N24) untouched — those correctly denote §4.2's learned router.

### MF-B — squeeze "collapse" → pricing law (preferred; no run)
- Removed "provably cannot beat the box" / "cannot beat the box" / "collapses past the box" / "provably cannot buy reach" from: abstract (2 places), contribution 2, §2, §3.2 heading, §3.2 table (squeeze row), Fig 1 caption, §3.2 arm-(i), §5.
- New load-bearing statement (§3.2 arm-(i), abstract, §5): the squeeze is a canonical map, not a flow; it grants instantaneous displacement `p₀ sinh ζ/M₀` that *does* exceed the box, bought at energy `≤ e^{2|ζ|}H`, so **reach beyond L is exponentially priced in rapidity**; the wormhole buys **unbounded Δ at a fixed ledger**. Dichotomy: *squeeze reach is exponentially priced in distance; wormhole reach is flat-priced.*
- The table/figure "→0 at d≥4.0" now read "priced out of the swept ζ≤2.0 grid, not 'cannot reach'"; a `$^\dagger$` footnote (md + tex) gives the predicted ζ to reach further, labelled analytic-not-measured.
- §2's "energy cannot buy reach" tightened to "cannot buy *flow* reach (cap bounds the Verlet drift)" so it no longer collides with the squeeze's canonical displacement.

### MF-C — 400-ep budget labelled inline at both collision sites
- **§4.1 point 2:** now owns that the 4.81×/etc. models are trained **400 ep**, a band §4.3 shows is not converged; at convergence (§4.3 C.4.a, 2000 ep) the gate accuracy **matches** full-budget rather than exceeding it → robust payoff is *rationing* (9.9× intra-CLU), the accuracy headroom is a property of an imperfect memory. CM-2's "escalatable memory" claim explicitly re-anchored on the *saving*, not the accuracy gain.
- **Contribution 4:** "landing above always-full" now scoped "(at a 400-epoch budget)" + reconciliation clause.
- **App A.5 cross-section note:** added the *epochs* axis (§4.1 = 400 ep; §4.3 = {500,1000,2000,4000}); **deleted** the false sentence "no two numbers in this paper are drawn from configs a reviewer could read as contradictory" and replaced with an honest pointer to the inline reconciliation.
- No numbers touched — 4.81×/1.57×/1.14× trace exactly (v1-pivot L39–40); only their *meaning* is now scoped.

### CM-14 — Newtonian qualifier folded (scope clause, not retraction)
- **§5 design rule 3** + **App F.2**: the FDT σ* "exact" claim now carries the Newtonian-kinetic-mode scope: (a) post-MH, σ* is a *proposal-tuning* scale (mixing efficiency), not a correctness condition — any σ leaves π invariant, the 0.0995→0.0065 gain is finite-budget shadow bias; (b) in relativistic mode π_p is Maxwell–Jüttner, a fixed-covariance Gaussian kick is not a Gibbs refresh, so MH (not σ*) secures stationarity. Written explicitly as a scope clause because **V1's units are Newtonian throughout** (per task + CM-14 amendment).

## How I verified
- MF-B arithmetic (independent, `python3`): displacement `p₀ sinh ζ/M₀` with L=2.5, p₀=1.2, M₀=4.0 → reach ζ=2.0→3.588 (<3.6, misses d=4.0); ζ to land d=4.0 (reach≥3.6) = **2.0105**, d=5.0 (reach≥4.6) = **2.6441**; energy e^{2ζ} = 54.6 / 55.8 / 198. Matches referee exactly.
- Stale-phrase grep on both files: **zero** remaining "cannot beat / collapses past / more cheaply / learned-router-cannot" in claim positions; remaining "physics-free learned router" occurrences are all §4.2 (correct).
- **PDF build:** `tectonic draft.tex` → `draft.pdf` (842 KB, 2026-07-19 19:48), exit 0, **0 undefined references / 0 citation errors**; only hbox spacing warnings (overfull ≤32.7pt in the wide reach table, underfull in the dense provenance line — cosmetic, pruning-pass territory).

## Findings / notes for the Hub
- **MF-B has a claims-matrix consequence the referee already flagged (CM-12/CM-7):** `paid-access-theory` L29 ("the wormhole is the **only** mechanism that beats the causal box") and L244 ("Squeeze cannot beat relativistic C_T | [proven]") are too strong — the proof covers `Q_T ⊆ C_T(q₀)` for the *flow*, not the squeeze's own instantaneous displacement. The paper is now internally consistent (pricing form), but **CM-12's approved wording should be amended to the pricing form before the F5 note ships the same theorem.** Not mine to edit (matrix is Hub's) — flagging.
- **CM-7 approved wording should say "an absorbing/state-replacing jump," not "the router,"** or V2/V3 inherit the MF-A conflation. Flagging for the matrix maintainer.
- **Scope-honest:** I did NOT run the ζ-grid extension (preferred option was restatement). If the Hub later wants the collapse *measured* where it genuinely occurs (fallback option), it is a minutes-long analytic run (missing-experiment #1 in v1-referee-2) — but the pricing-law restatement makes it optional, not blocking.

## Deliberately NOT done (out of task scope — flagged for Hub/next pass)
Task was scoped to MF-A/B/C + CM-14. These referee items remain **open** and were not in my task:
- **MF-D** — contribution 6 attaches "n=8" to the full 198-job grid (should be "n=8 capacity, n=5 stress, n=3 frontier"). One-parenthesis fix; §4.3/App C.4/Fig 2 already correct, only the contributions bullet is wrong.
- **MF-E** — dangling registry refs N21/N27 (cited, never defined) in §4.2 + App D. Trivial define-or-drop.
- **SHOULD-fixes S1–S13** (notably S4 "measured by autodiff" tautology; S5 Fig-1a offsets above 1.0; S8 length ≈2× over 4–5pp — pruning; S3 KE₀=0.72 cross-config number). All deferred.
- **MF-1** — theory note still `(Anonymous, 2026)`; Head critical path, untouched per task.

## Git footprint
None — all edits are under `.claude/papers/v1-short/` (gitignored per protocol §2): `draft.md`, `draft.tex`, `draft.pdf`, `CHANGELOG.md`. No tracked code touched.

## Proposed handover updates (for the Hub)
- **V1 status: v0.4 — MF-A/B/C closed, CM-14 folded; expected verdict weak-accept (plausibly accept via the pricing law).** MF-B taken as the PREFERRED pricing-law restatement (no run). The three hostile-quote errors the referee listed are all defused: #1 (router conflation) → decision≠transport rewrite; #2 (ζ-grid artifact) → pricing law; #3 (§4.1 under-trained) → owned inline as a C-9 negative.
- **Two matrix edits requested (cross-short, not mine):** (1) CM-12 wording → pricing form (squeeze buys additive reach `p₀ sinh ζ/M₀` at cost `e^{2ζ}`; wormhole buys unbounded Δ at fixed ledger) before the F5 note ships the theorem; (2) CM-7 payoff wording → "absorbing/state-replacing jump," not "router," so V2/V3 don't inherit MF-A.
- **Still open for a follow-up writer pass:** MF-D (n=8 scope), MF-E (N21/N27 dangling), pruning (S8, blocking for layout at 4–5pp), S1–S13 cosmetic. MF-1 = Head critical path.
