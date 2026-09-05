# v1-revision — paper-writer report

Task + acceptance criterion: fold settled CM-8 grid + Thread-9 design rules into V1-short (v0.2). Accept = both slots filled from v1.4 approved wordings; noise wall present as a first-class negative; PDF builds. → v1-referee (w10).
Status: **done.** Both slots filled; noise wall foregrounded in main text (§4.3), figure caption, App C.4.c, and App D (N2b); PDF rebuilds clean (tectonic, 0 undefined refs). Edited `.claude/papers/v1-short/` in place (research-only; no tracked code touched).

## What I did (by task item)

**Item 1 — §4.3 + App C.4: `[pending: regime-remap-complete]` → CM-8 SETTLED.**
- Rewrote §4.3 "Accuracy" paragraph (was "preliminary — marked slot") as **"Accuracy (settled — full grid, 198 jobs)"** with three load-bearing qualifiers travelling together, using the approved CM-8 wordings verbatim:
  1. clean/correlated cues, kv≤64 reverse Hopfield (Δ+0.02, n=8); **ρ=0.9 widening (Δ+0.08…+0.16) labelled explicitly as Hopfield-fragility, not CLU strength** (Hopfield 0.72/0.59; CLU also drops 0.87/0.67).
  2. **epoch-budget wall, not capacity wall** (kv96 reverses @4000 +0.03; kv128 ties +0.004; required epochs scale with kv); **kv32 over-trains** 1.00→0.89 (2000→4000 ep).
  3. **THE NOISE WALL** (dominant negative, foregrounded): eval-noise σ∈{0.3,0.6} → **0/6 close at any capacity, even kv32** (gate 0.36 vs Hopfield 0.71 @σ=0.6/kv32) *despite* fidelity ≈1.0. On-thesis sentence included verbatim: *the gate's rationing works on clean retrieval; noise-robustness is Hopfield's.*
- Tally **6/15 close @2000 ep** stated. **Intra-CLU cost caveat attached to every accuracy sentence** (Item-4 wording): "9–10×" is intra-CLU rationing, never a win over Hopfield, which stays the cheaper AND more cue-noise-robust retriever.
- **App C.4** replaced with the full settled grid: C.4.a paired capacity axis (n=8), C.4.b epoch-scaling frontier (n=3, incl. over-train + non-monotone dip negatives), C.4.c both stress axes (correlation + the noise-wall table).
- Added the honest regime-map **figures**: copied `fig_regime_map.png` + `fig_frontier_clean.png` into the paper dir; embedded as Figure (float, `\ref{fig:regime}`) in draft.tex with an evidence-grade caption; referenced as Figures 2/3 in draft.md.
- Updated Intro contribution #5 and the abstract's closing clause to the settled/regime-specific framing (added "more cue-noise-robust").

**Item 2 — §5: `[optional slot: thread9-mh-kernel]` → certified-Markov-kernel design-rule flourish (CM-14).**
- Replaced the placeholder bullet with a standalone closing subsection **"Test-time retries as a certified Markov kernel — and the receipt even *it* carries."** Physics-heavy / ML4PS register. Content (all CM-14 approved):
  - squeeze-MH is a certified HMC-family kernel **under stated conditions** (det J=1 ⇒ no Jacobian; π-reversible) — conditions stated as C-6 fine print: sign-symmetrized ζ + momentum refresh (squeeze-only reducible/non-ergodic), γ=0 inside certified segments.
  - governed cascade = **Metropolis-within-annealing → colder MAP-seeking measure** (T_eff 1.0→0.61); explicit "**never claim stationarity for the governed composite**" + "a feature for retrieval, honestly not-Gibbs."
  - **certified retry erodes the latch** (D=½s², verified 1.29e-3 vs 1.25e-3; N_erode=(Δ_read/s)²); charge-preservation ≠ position-preservation caveat included. Closing line "**Even the certified retry carries a receipt: coset diffusion.**"
  - **design rules**: γ=0 certified segments + mixture ½MALA(σ*)+½squeeze-MH (σ* FDT-load-bearing, L1 0.0995→0.0065) + proposals projected off the coset tangent (quenches latch, D=0).
  - **CM-3 deflation** stated: per-model calibrated T re-imports the learned gate ⇒ MH's asset is the stationarity certificate + erosion accounting, not parsimony, not (on current evidence) performance.
  - **Labelled theory-complete on toy EBMs; experiments = future work; explicitly "no runs on trained CLU checkpoints are claimed."**
- Extended Prior-art honesty (HMC/MALA/Roberts&Tweedie/Mermin–Wagner lineage); added **Roberts & Tweedie 1996** to references (both md + tex).

**Item 3 — Editorial answers (binding, applied).**
- Q1 (position-paper posture confirmed): banner labels/verification-vs-evidence grades kept unchanged.
- Q2 (keep Hopfield 0.18→0.88 via CM-2): the CM-2 approved wording is retained in §4.1; **added `minus-the-physics` Part B (5 seeds) to the App-A.2 provenance** as the source of the Hopfield-transfer number (md table row + tex App-A note).
- Q3 (EBT prominence confirmed): Gladstone et al. 2025 kept prominent in Intro/§5, flagged `(preprint; venue status unverified)` in refs — unchanged.
- Q5 (anonymize at pruning pass): placeholders `[WORKING TITLE …]` / `[AUTHORS PLACEHOLDER]` left in place.

**Item 4 — CM-12 ADOPTED.** No "pending Head confirm"/"conditional pillar" hedge exists anywhere in the draft (CM-12 already stated as verified/adopted throughout; grep clean). Nothing to remove. PDF rebuilt.

## How I verified
- `grep` both files: **0 leftover** `pending:` / `optional slot` / `marked slot` / `regime-remap-complete` / `preliminary` / `folds here` placeholders.
- "noise wall" present ×3 in each file (main text + caption + App).
- CM-3 forbidden-claim grep: every hit is an explicit **disclaimer** ("not a superior confidence signal", "energy-as-superior-signal is not asserted") — no assertion of energy superiority. CM-3 clean.
- Build: `tectonic draft.tex` — clean multi-pass, **0 undefined references** in final log (fig:reach + fig:regime resolve), `draft.pdf` = 335 KiB (up from 159 KiB; both regime figures embedded). Only non-fatal hbox typography warnings (Table 1 tabular). Toolchain: only `tectonic` present (no pdflatex/xelatex) — this is the same builder v0.1 used.

## Evidence backing each section (provenance)
- §4.3 + App C.4 + Fig 2/3: `regime-remap-2000ep.md` §"w8 completion" (Items 1/2/4, 198 jobs, 0 failures) + Item 3 (Hopfield 1-matvec parity). Flag arm = `train_epochs∈{500,1000,2000,4000}`, commit `63fea62`, JAX 0.9.0, f32, langevin_noise=legacy; n=8 capacity / n=5 stress / n=3 frontier. Every number traces to that report's tables; none rounded/smoothed.
- §5 design-rule ¶: `thread9-mh-kernel.md` (V1-grade position paragraph + Prop-MH2, 4 float64 checks, numpy 2.4.1). Verified constants used as-is: D=1.29e-3 vs 1.25e-3, T_eff 1.0→0.61, L1 0.0995→0.0065.
- §4.1 Hopfield transfer: `minus-the-physics` Part B (CM-2), now in App-A.2 provenance.
- Related-work framing lifted from the draft's existing scout condensation (`[^scout]` footnote) + thread9 prior-art positioning (§"Prior-art positioning").

## Charter/matrix compliance
- CM-8 load-bearing phrasings verbatim; noise wall is a first-class negative in main text (not appendix-only). CM-14 folded with C-6 fine print next to each claim; "not-Gibbs / never claim stationarity" honesty preserved. CM-3 forbidden claim never asserted. CM-2 (0.18→0.88) via approved wording. CM-12 canonical constants unchanged. Cost caveat (intra-CLU) on every accuracy sentence (C-5). Designed-testbed §3 = verification, learned §4 = evidence banners intact (C-2). Hermetic refs (C-8): only added Roberts&Tweedie 1996 (citable). Appendix-maximal (C-10): all negatives/robustness in App C/D, nothing self-pruned.

## Git footprint
None — research-only edits, all under gitignored `.claude/papers/v1-short/`. No tracked code touched, no branch, no commits.

## Open questions / follow-ups / risks (for the Hub / v1-referee w10)
1. **Noise-wall diagnosis is future work, not a cure.** The report's recommended next experiment (noise-aware τ / longer relax / denoising init) is flagged in §4.3 + N2b as future work. If a referee asks "can you fix it?", V1 currently answers "no, and Hopfield owns that axis" — consistent with the position-paper posture, but the Head should confirm this is the intended stance before v1-referee.
2. **Figure count.** V1 now has 3 figures (headline reach + 2-panel regime). For a 4–5pp workshop short this may be tight; the pruning pass may want to move Fig 3 (frontier) to appendix. Flagged, not actioned (C-10: no self-pruning now).
3. **§5 length.** The certified-kernel design-rule paragraph is long (~1 dense paragraph). It reads as the closing flourish per task, but at pruning it is a candidate to compress or promote to a short subsection with the CM-14 numerics tabled. Banked for the pruning pass.
4. **CM-8 header vs content.** `claims_matrix.md` line 1 still reads "v1.3, wave-7"; the task referenced "v1.4" and CM-8 SETTLED / CM-14 content is present in the file. I drafted against the current CM-8/CM-14 content (which matches the task's approved wordings). Non-blocking, but the Hub may want to bump the matrix header version string.

## Proposed handover updates (for the Hub)
- V1-short v0.2 shipped: both w9 slots filled (CM-8 SETTLED §4.3/App C.4 with the noise wall foregrounded; CM-14 §5 design-rule ¶). PDF builds clean. Ready for `v1-referee` (w10).
- Provenance added: `minus-the-physics` Part B now listed in V1 App-A.2 for the CM-2 Hopfield transfer (Q2 satisfied).
- No tracked-code changes; no negatives dropped (N2 upgraded SETTLED, N2b noise wall added).
