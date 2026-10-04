# transfer-docs-sync-w9 — doc-curator report

Task + acceptance criterion: bring all three transfer docs current through **wave-8** — update `HEP_primers.md` in place (SF-3 laws-survive-cure, SF-2 4×-retirement, new §6.5 Thread-9 squeeze-MH kernel, CM-8 SETTLED + noise wall, scout interference neighbourhood, ledger rows/reading list); sweep w8 into `negative_results.md` (noise wall, over-training, fidelity dip, 4×-inversion, latch-diffusion-under-MH, 44%-attribution; re-tier N2 final); write the ledger's **Wave-8 addendum** (⟲). Per-doc edit list + output↔handover contradictions.

Status: **done.** All three docs edited; **no tracked code touched**; only gitignored `.claude/**` files written. **No git footprint.**

Inputs read (in order): AGENT_PROTOCOL · doc-curator def · `transfer-docs-sync-w7.md` (prior curator pass) · handover §10 **WAVE-8 REVIEWED** entry (2026-07-08, source of truth) · all 7 w8 outputs {`regime-remap-2000ep §w8-completion`, `thread9-mh-kernel`, `v2-referee-experiments`, `scout-modular-interference`, `v1-short-draft`, `v2-revision`, `v3-revision`} · `claims_matrix.md` (v1.4 **body** content: CM-8 SETTLED, CM-4/6 amended, CM-12 ADOPTED, CM-14 new) · all three transfer docs in full.

Discipline: every number transcribed with report+section citation (no verdict reinterpretation); Def-2 nomenclature (inertial M / spectral μ); two-layer status tags preserved. Doc state before this pass = through **wave-7** (primer maintenance "wave-7", ledger last addendum Wave-7, negatives N32–N36 / "through wave-7"). This pass brings all three **through wave-8**.

---

## Doc 1 — `HEP_primers.md` (in-place updates + 1 new subsection)
1. **Maintenance header** — added the wave-8 line (7 touched sections listed).
2. **§3.3 (retention law)** — NEW wave-8 blockquote: **SF-3 laws-survive-the-cure** (anchored λ=100, 3000 ep, 3 seeds — GMOR ratio 1.00000±1.5e-12/4.6 decades, retention slope −0.956, floor 27.03, latch μ²≈1e-15, EP slope **0.5165 bit-identical to 150 ep**; r*=0.917) + **SF-2 caveat** (retention in map-steps; per-step CLU Verlet 6.2×LSTM/3.1×LEM wall / 14–15× FLOPs; "≈4× longer" **inverts** under compute-normalization 23.5×/14.6×; lead with the qualitative triad). `v2-referee-experiments` SF-3/SF-2. CM-6/CM-4 amended.
3. **NEW §6.5** — "The retry as a certified Markov kernel: squeeze + Metropolis" (house style: philosophy → math → conditions → governor deflation → latch erosion → 3 design rules → temperature → → CLU). Covers CM-14: squeeze+MH = π-reversible HMC-family kernel (det J=1, L1=0.0095); non-ergodic without momentum refresh; **governor ⇒ Metropolis-within-annealing** (T_eff 1.0→0.61); **certified retry erodes the latch** (D=½s², N_erode=(Δ_read/s)²; charge-preservation ≠ position-preservation); 3 design rules (γ=0 segments / ½MALA(σ*)+½squeeze with FDT-load-bearing σ* / coset-projection); CM-3 honest deflation. Cross-refs §5.2, §6.3, §8.4. Placed §5/§8-adjacent per task. `thread9-mh-kernel`.
4. **§8.3** — NEW wave-8 blockquote: **CM-8 SETTLED** on the full 198-job grid — reversal is clean/correlated-cue **kv≤64 only** (n=8, Δ+0.02; ρ=0.9 widening is Hopfield collapse); **epoch-budget wall not capacity** (kv96@4000 +0.03, kv128 ties); **kv32 over-trains** 1.00→0.89; **THE NOISE WALL** (0/6 under σ∈{0.3,0.6}, gate 0.36 vs Hop 0.71 @fid 1.0); intermediate-epoch fidelity dip; cost+noise-robustness stay Hopfield's; 6/15 close. `regime-remap-2000ep §w8-completion`.
5. **Part IX (§9.2)** — NEW wave-8 blockquote: the interference **related-work neighbourhood** (McCloskey/French; Jacobs/Shazeer; Kirkpatrick/Mallya; **Doan 2021 NTK-overlap** + Riemer/Yu + Jacot; **Boopathy 2025** modular-scaling) with the CLEAR-at-specific-claim / CROWDED-at-neighbourhood verdict and the two sharp foils. `scout-modular-interference`.
6. **§10.3 claim-status ledger** — added 4 rows: CM-8 SETTLED (noise wall), retention/GMOR/EP survive the cure (SF-3), "4×" retired (SF-2/SF-1), squeeze-MH kernel (CM-14).
7. **§10.4 reading list** — added MCMC/test-time-compute-as-sampling (HMC/MALA/annealing/Mermin–Wagner) and interference/continual-learning (Doan/Boopathy/McCloskey…) pointers — pays part of the w7 docs-debt #2.

**Deliberately left unchanged:** Parts I–IV core derivations; §8.4 paid-access (already w7-current: CM-12 ADOPTED; Thread-9 cross-referenced from new §6.5 rather than duplicated); §8.5 reversible (w7-current). No chapter-level rewrites beyond the mandated new §6.5.

## Doc 2 — `negative_results.md` (w8 sweep: N2 accuracy SETTLED + N37–N42; 42 entries total)
- **N2 RE-TIERED → SETTLED (final):** appended a `✅ w8 SETTLED` note (full 198-job grid; reversal clean/corr kv≤64 only, 6/15 close, n=8; epoch-budget wall; two new travelling negatives N37/N38; cost+noise-robustness Hopfield's). **Verdict direction no longer provisional; tier A retained.** Prior provisional notes kept (registry append convention). Summary-index row + open-provenance flag updated.
- **6 new entries** (schema-complete: tried · exact numbers+citation · mechanism · scope · tier · vertical · source):
  - **N37** THE NOISE WALL (0/6 under cue noise despite fidelity 1.0) — **tier A, V1** (THE headline negative)
  - **N38** small memories over-train (kv32 1.00→0.89 @4000 ep) — tier B, V1
  - **N39** non-monotone intermediate-epoch fidelity dip (kv96→0.24, kv128→0.09 @1000 ep) — tier C, V1
  - **N40** "≈4× longer retention" retired (inverts 23.5×/14.6× compute) — **tier A, V2**
  - **N41** certified squeeze-MH retry erodes the latch (D=½s²; charge≠position) — tier B, V1+F5 (design guard, proven)
  - **N42** (process) 44% Mo-λ̂ deviation re-attributed to near-EP — tier C, V2
- Updated: header maintenance line, summary index (N2 + 6 rows), paper-writer appendix-mining notes (per-vertical incl. the noise-wall-travels caveat + N40 lead-triad + N41 MH guard + interference bib CLOSED), cross-thread note (**N41 = 4th energy-not-a-signal deflation instance**), open-provenance flags (N2 resolved, N37/N40 scope caveats, N41 no-trained-runs/backlog + kill criterion).

## Doc 3 — `philosophy-synthesis.md` (ledger — WROTE the Wave-8 addendum; ⟲ chapters untouched)
Appended `# ⟲ Wave-8 addendum (2026-07-09, doc-curator)` with:
- **Chapter deltas:** Ch.4 (CM-8 SETTLED + noise wall dominant negative; Thread-9 CM-14 kernel + two deflations, GO, experiment backlogged); Ch.1 (SF-3 laws-survive-cure / SF-2 4×-retirement / SF-1 Mo-estimator + 44% fix); Ch.5 (modular-interference bib CLOSED — Doan/Boopathy foils); drafts (all three exist + build).
- **Scorecard deltas** (rows 4, 1) + **gap-list updates** (CM-8 gap CLOSED → new noise-wall threat; Thread-9 backlog; bib CLOSED; remaining real-data/referee-passes/reversible-in-trainer/entrance-steering/venue-calendar) + 3 new **scope guards** (CM-8 noise-wall-must-travel, CM-4 map-steps≠compute, CM-14 no-Gibbs-for-governed-composite / latch-erosion).
- **Positioning ripples** (V1 self-portrait sharpens; V2 strengthened by its own referee experiments; V3 flank closed; Anonymous-2026 citation still critical-path) + a **curator doc-hygiene flag** on the matrix version-label lag.
- ⟲ protocol respected: no chapter rewritten; verified the addendum covers all 7 w8 outputs before writing.

---

## How I verified
- Every quantitative statement cross-checked against its source-report section before transcription: CM-8 grid tables (`regime-remap-2000ep §w8` Item 1/2/4 + §3 Hopfield-parity); Thread-9 checks C1–C4 (`thread9-mh-kernel` Appendix N + Parts 1–4); SF-1/2/3 tables (`v2-referee-experiments`); interference bib (`scout-modular-interference` EVIDENCE + novelty table).
- Confirmed `claims_matrix.md` **body** carries v1.4 content (CM-8 SETTLED, CM-4/6 amended, CM-12 ADOPTED, CM-14 new) matching the handover WAVE-8 entry and the outputs before transcribing status tags.
- Confirmed the docs' prior state was through w7 (no w8 curator pass existed) — this pass advances all three to w8.
- No tracked code touched; all writes under `.claude/`. No git footprint.

## Output ↔ handover contradictions found (flagged, NOT resolved — per mandate)
- **None substantive.** The 7 w8 outputs, the handover §10 WAVE-8 review entry, and claims-matrix v1.4 body are mutually consistent on CM-8 SETTLED (incl. the noise wall), CM-14 (Thread-9 kernel + design rules), and the SF-1/2/3 amendments (CM-4/CM-6).
- **One doc-hygiene lag (not a contradiction):** `claims_matrix.md` header still self-labels **"v1.3, wave-7 review (2026-07-07)"** while its body is v1.4 (the task itself names it "claims matrix **v1.4**"). Same version-label lag flagged in the w6 curator report; a Hub matter (I don't edit the matrix). I transcribed from the body.

## Docs-debt / notes for the Hub
1. **w7 docs-debt #2 (reading-list gap) partially PAID:** §10.4 now carries MCMC/HMC/MALA/annealing + interference/continual-learning pointers for the §6.5 and Part IX readers.
2. **Ledger frozen scorecard (⟲ structural):** current state now spans FIVE addenda (w4–w8) plus the frozen w3 scorecard — a reader reconciles six locations. Unchanged from prior flags; a single current-state scorecard would violate "chapters never rewritten" and is a Head/Hub call.
3. **Draft files** (`v1-short`, `v2-short/v0.2`, `v3-short/v0.2`, `f5-note`) live under `.claude/papers/` — paper-writer/referee own them; read only via the w8 reports. No transfer-doc action.

## Proposed handover updates (for the Hub)
- Transfer docs current through **w8**: `HEP_primers.md` (new §6.5 + 6 in-place edits, all w8 evidence folded), `negative_results.md` (N2 accuracy SETTLED + N37–N42; **now 42 entries**), `philosophy-synthesis.md` (Wave-8 addendum written per standing duty).
- **The noise wall (N37) is the new dominant V1 negative** and, per the handover, the narrative threat to settle before ICLR — it is now first-class in all three docs and **must travel with every CM-8 accuracy-reversal claim** the paper-writer drafts.
- **CM-14 discipline for drafters (new):** never claim Gibbs stationarity for the *governed* composite (Metropolis-within-annealing); certified retry **erodes the latch** unless coset-projected (N41); MH's asset is the certificate, not parsimony/performance. The Thread-9 experiment is **backlogged (w9 spec exists, not launched)** — do not draft an MH *performance* claim.
- **Minor:** consider re-labeling `claims_matrix.md` header "v1.4 / wave-8 review" (body already v1.4).
