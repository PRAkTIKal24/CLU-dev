# transfer-docs-sync — doc-curator report
Task + acceptance criterion: bring all transfer docs current through wave-5 — audit+update `HEP_primers.md`, CREATE `negative_results.md` sweeping w1→w5, verify the ledger's w5 addendum. Per-doc edit list + docs-debt list.
Status: **done.** All three docs edited; no code touched; only gitignored `.claude/**` files written.

Inputs read: AGENT_PROTOCOL · handover §10 (w4+w5 review entries = source of truth) · w5 outputs {mass-lr-doctrine-test, minus-the-physics, v2-prefreeze-baselines, f5-arxiv-note} · w4 outputs {v1-hopfield-stress, v1-wormhole-routing, seed-sweeps, sleep-erosion-study} · claims_matrix.md v1.0 · brainstorm Thread 8 · both transfer docs in full.

Discipline: every number transcribed with report+section citation (no reinterpretation of verdicts); Def-2 nomenclature (inertial M / spectral μ); status-tag scheme preserved.

---

## Doc 1 — `HEP_primers.md` (updated in place, dated blockquotes per protocol)
The primer had **no dated updates at all** — authored at w2, never advanced through w3/w4/w5. I did the four task-mandated updates + two genuine factual corrections found on audit. Each marked with `> **Update (wave-N):**`.

1. **Header** — added a "Maintenance" note (authored w2, updated through w5; lists the six touched sections + companion docs).
2. **§1.5 (symplecticity)** — NEW dated block: the **measured "which component buys what" attribution table** from `minus-the-physics.md` Part A (3 seeds, dim 4, S¹, 150 ep, params ±0.05%). Volume conservation → BIBO 1.0-vs-0.33 + protected flat μ² 0.008-vs-0.122 + CD-vacuum survival (r* survives vs 0.0); leapfrog structure → latch (coset drift 0.19 vs twin 1.15); physics **costs raw fit** (twin ~15× better MSE). Explicitly scoped as **contraction-forbidden, cap inactive** (points to §8.4). This is the "best pedagogy the program produced" the task flagged.
3. **§2.3 (spectral mass / Hyp-3)** — NEW dated block: the **CM-5 mass-narrowness sharpening**. 300-ep "~0.08 ceiling" = budget artifact; at 1500 ep ordering inducible (Spearman +0.89, 5/5 seeds), 10× mass-lr → spread 0.52 (6.5×) + MSE 1.85→0.64; magnitude never reached (designed 7–14× better); 100× lr **inverts** ordering; curriculum hurt. Point 5's tag changed `[design hypothesis]` → `[design hypothesis → sharpened at wave-5]`. Doctrine restated: *ordering inducible, magnitude designed.*
4. **§5.2 (FDT sampler)** — CORRECTION: the MNIST-imbalance conjecture (was `[conjectured]`) is **REFUTED** (`generative-studies A`: χ²=0.08, p=1.00, 2/192 flips). Mechanism confirmed exact; imbalance = learned landscape; harmless-because-narrow-M (ties §2.3); conditional-promotion once banding lands. Added the no-positional-equilibrium side-finding.
5. **§7.2 (friction field)** — NEW dated block: **adaptive-K + compact-gate status**. Adaptive-K validated (K=1→8, best rejection 0.861, locus 2/3 vs pilot 2/6); compact gate (leakage ↓200×, oracle-compact exactly 0.0 outside radius → retention ceiling while rejecting; needs placement, adaptive-K supplies it). Plus the **governor+field composition negative** (Pareto-dominated 0.48), the **Cor-13 erase-scope clause** (friction latches, can't erase coset), and the "objectives must not legislate friction" design principle.
6. **NEW §8.4 "The two forbidden cheats and paid access"** — Thread-8 economic frame `[design hypothesis + one measured leg]`: (i) unpaid contraction (volume; the *measured* dominant fit-gap term, broken-vol recovers 2.4×, cap inactive), (ii) unpaid long-range moves (v_max=c/√M, sync∝κ^−½). Priced escape-hatch table (γ_φ / wormholes / squeezes). Intra-unit-wormhole candidate + V1-conditional-4th-pillar gating. w6 `fit-gap-anatomy` falsifiables (a/b/c). Scope discipline: squeeze-as-access ≠ the killed retry claim.
7. **§10.3 claim-status ledger** — MNIST row → **refuted (w3)**; added 5 rows (symplecticity attribution; mass-ordering-inducible; memory-agnostic gate; adaptive-K/compact; structural-triad-absent-in-baselines), each with key number + source.

**Deliberately left unchanged** (in scope of physics but not mandated, to respect the bounded-edit mandate): the core Parts I–IV derivations (still exact/correct); §8.2 lattice pricing (v3-lattice numbers already present and correct); the wormhole-routing *positive* result (added only the mis-route caveat context via §8.4, not a full §8.3 rewrite — see docs-debt).

## Doc 2 — `negative_results.md` (CREATED — did not exist)
Built the registry from scratch, one entry per negative, per the schema (tried · exact numbers+citation · mechanism · scope · tier · vertical · source). **23 entries** sweeping w1→w5, with a summary index table + paper-writer appendix-mining notes + open provenance flags. Head policy C-9 fulfilled: ALL negatives documented, tiered by paper-appendix prominence (A/B/C).
- Verified + completed the task's seed list (all 12 present) and added: governor+field composition (N12), λ-resweep null (N13), compact under-coverage (N15), κ_eff exponent inconclusive (N16), raw-R-not-cross-model (N21, the negative that mandated learned-τ), proven saddle/isoenergetic negatives (N22, already in F5 note §5), wormhole gate caveats (N23).
- Tier-A (named-paper appendix): N1 squeeze retries, N2 Hopfield-dominant, N3 energy≈margin/CM-3, N4 isotropization-NO, N5 sleep-erosion-law, N6 no-task-RMSE, N7 mass-not-emergent, N8 100×-inversion, N10 FDT-attribution-refuted, N17 dead-Lyapunov, N18 non-Gibbs-sampler, N22 proven-negatives.
- Cross-thread note recorded: N1+N7+N10 are the same narrow-M fact three times (the pivot variable).

## Doc 3 — `philosophy-synthesis.md` (ledger — coverage check, minimal supplement)
The Hub's **w5 addendum is faithful and complete** against all four w5 outputs (I verified chapter-by-chapter). Appended one tight **`(curator supplement)`** block for three genuine load-bearing precisions the compressed addendum dropped:
1. the **CM-1 scope qualifier on "physics costs fit"** — contraction-forbidden (not the cap, which was inactive) + short-horizon-only (crossing unmeasured until fit-gap-anatomy). Prevents over-reading the twin's 15× fit edge.
2. the **coRNN honest-weak-baseline caveat** — LSTM/LEM (0.18/0.23 rad) carry the fair CM-4 comparison, not coRNN (0.82 rad).
3. doc cross-reference that the C-9 negatives registry now exists (indexes the ledger's per-chapter negatives for appendix mining).
No chapter rewritten (⟲ protocol respected).

---

## How I verified
- Every quantitative statement cross-checked against its source report section before transcription (e.g. mass-lr table lines 37–48; minus-the-physics Part-A table; sleep-erosion §3.1/3.3; v1-hopfield-stress §Findings; f5-note check numbers).
- Confirmed the MNIST-imbalance conjecture was live in the primer (§5.2 + §10.3, both `[conjectured]`) and refuted in `generative-studies A.2` → correction justified.
- Confirmed the w5 addendum's four-output coverage before deciding the supplement was minimal.
- No tracked code touched; all writes under `.claude/` (HEP_primers, negative_results, philosophy-synthesis). No git footprint.

## Docs-debt list (bigger rewrites than this per-wave mandate allows — flagged for the Hub)
1. **HEP_primers is a full w2 snapshot behind on POSITIVE results too.** I updated the six task-scoped sections + two corrections, but the primer's §8.3 (wormholes: still "[design hypothesis]" though routing was *measured* in w4), §8.2/§5 lattice+V3 (banding now powered 5/5; pricing on trained couplings), and Part IX interference (NTK still "unrun" — correct, but no w4/w5 texture) would benefit from a dedicated primer-refresh pass. Not done here to stay in bounded-edit scope.
2. **§8.3 wormhole routing** deserves its own measured subsection (0.875/0.812 vs 0.50 ceilings; 1-hop beats N−1-hop chain; AUROC 0.95–0.96) rather than only being referenced from the new §8.4. Currently the positive result lives only in the ledger/negatives-caveat, not the primer body.
3. **Erosion (N5) is publication-grade** and appears in the ledger + negatives, but the primer's §1.8 forgetting table and §7 have no erosion entry — a §7.3 "when training erodes the geometry it was given" subsection is warranted once the Head fixes placement (V2 appendix vs standalone short).
4. **Ledger scorecard table (lines 318–326)** is frozen at w3 verdicts by ⟲ design; the deltas live in the w4/w5 addenda. A reader must reconcile three locations. If the Head ever wants a single current-state scorecard, that's a Hub call (would violate "chapters never rewritten" if done in place).

## Gaps / disagreements flagged for the Hub (not resolved — per my mandate)
- **No output↔handover contradictions found** this sweep; the w4/w5 review entries, the outputs, and the claims matrix are mutually consistent.
- **N5 novelty is pending scout confirm** (venue-follow-up #5) — I marked the erosion novelty as unconfirmed in both the registry and did not upgrade its claim status.
- **N4 (V2-isotropization under mass-lr) is deferred/unmeasured** — flagged in the registry; the within-channel-isotropy angle remains open.
- **Tier assignments in the registry follow the current M3 rec** (V2 GO / V1 GO-reframed / V3 conditional). If V3 drops to future-work at a later review, N7–N9/N16 re-tier from "V3 short appendix" → "ICLR appendix." Noted in the registry's open-flags.
- **Config hygiene item** seen repeatedly in w4/w5 reports (duplicate `ExperimentV1WormholeConfig` F811 on main; `exp_s1_gamma_field.py` not forwarding `gate=`) — these are engineer/§7 items, not doc items; recorded here only so they aren't lost.

## Proposed handover updates (for the Hub)
- Transfer docs are current through w5: `HEP_primers.md` (6 sections + ledger updated), `negative_results.md` (created, 23 entries, C-9 satisfied), `philosophy-synthesis.md` (w5 addendum verified + curator supplement).
- Recommend the docs-debt items above be folded into a future doc-curator pass or a dedicated primer-refresh once the wave-6 results (fit-gap-anatomy, v3 interference-NTK, band-selection, router-baseline) land — several will directly retire the "[design hypothesis]" tags in primer Parts VIII–IX.
- The negatives registry is ready for the paper-writer to mine per-vertical appendices (appendix-mining notes are in-file).
