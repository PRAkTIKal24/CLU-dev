# transfer-docs-sync-w8 — doc-curator report
Task + acceptance criterion: bring all three transfer docs current through **wave-7** — update `HEP_primers.md`, sweep w7 into `negative_results.md`, and WRITE the ledger's w7 addendum (⟲ curator duty). Per-doc edit list + output↔handover contradictions.
Status: **done.** All three docs edited; no code touched; only gitignored `.claude/**` files written. No git footprint.

Inputs read (in protocol order): AGENT_PROTOCOL · this task file · handover §10 **2026-07-08 w7 review entry** (source of truth) + the w6/w5 entries for context · my w7 pass (`transfer-docs-sync-w7.md`, incl. its docs-debt list) · the seven w7 outputs {`paid-access-experiments`, `regime-remap-2000ep`, `v3-reversible-o1`, `v2-referee`, `v3-short-draft`, `fix-pack-4`, and CM-8/CM-12/CM-13 rows of `claims_matrix.md` v1.3} · all three transfer docs in full.

Discipline: every number transcribed with report+section citation (no verdict reinterpretation); Def-2 nomenclature (inertial M / spectral μ); two-layer status tags preserved. Verdicts summarized, never re-adjudicated; the M3 portfolio tension is **flagged, not resolved** (per mandate).

---

## Doc 1 — `HEP_primers.md` (7 edits: 4 in-place `> **Update (wave-7):**` blocks + 1 new subsection + heading tag + ledger rows)
1. **Maintenance header** — bumped "through wave-6" → "wave-7"; added the wave-7 line listing the five touched loci (§7.3 anchor flag, §8.3 CM-8 cost correction, §8.4 pillar-4 PASS, new §8.5, §10.3).
2. **§7.3** — NEW wave-7 block: `training.anchor_data_energy_lambda` now a **first-class flag** (default 0.0, bit-compatible), `chlu exp-d --anchor-lambda`, and the exp-d erosion-regime `RuntimeWarning` guard (sleep-on ∧ >300 ep ∧ so2 ∧ no-anchor). Suite 189/1-skip on branch. Source `fix-pack-4`.
3. **§8.3** — NEW wave-7 block: the **CM-8 cost correction** (`regime-remap-2000ep` Item 3). Hopfield hits its ceiling in **~1 matvec** (0.947–0.979, ≤0.003 over 10× iters ⇒ cost floor $O(kv\cdot d)$); the "9–10× savings" is **INTRA-CLU** (gate vs full-budget CLU ≥300 Verlet steps), NOT vs Hopfield. Cost story FINAL / accuracy-reversal grid-pending w8. Placed here because §8.3's w6 router block already carries the "a cheap baseline is cheaper" theme.
4. **§8.4 heading tag** — `[…reach/escape PROVEN (w6); end-to-end experiments = w7]` → `[…end-to-end experiments = PASS all four gate criteria (w7); pillar-4 ADOPTED; oracle-placement scope]`.
5. **§8.4** — NEW wave-7 block (`paid-access-experiments`, 5 seeds, dim 2&4, 5 unit tests): all four gate criteria MET with full numbers — (a) reach crossover (squeeze bounded, wormhole flat, Newtonian-squeeze control reaches d>L, throat/dense-V fails reach; **honest bracket [L, L+p₀sinhζ/M₀]**, L=2.5), (b) certificates (wormhole det J=[1.0]×6 & ledger=[0.0]×6 exact; squeeze det S=1.000, H_ratio≤e^{2ζ} matched-quadratic), (c) latch transported (ΔQ=0.25=pᵀXΔ err 0.0), (d) beats no-physics router + dense-V. Scope: ORACLE placement; learned entrance-steering out-of-scope (the crux at scale). Head 2026-07-08 pillar-4 ADOPTED → V1 ships ML4PS position paper.
6. **NEW §8.5** — "Reversible training: the conservative memory learns for free" (`v3-reversible-o1`, CM-13). House style (philosophy → look-up → math [closed-form leapfrog inverse] → measured → γ boundary → →CLU → status). Numbers: 946× peak-mem @T=1024,N=2; grads ≤2e-6 f32 / ~1e-15 f64; ≈0.9× wall-time (CPU/small-D only); **exact only at γ=0**, horizon 3.3e4@γ=1e-3…1.3e2@γ=0.1 (f64). Not yet in `train_chlu`; position vs checkpointing O(√T).
7. **§10.3 ledger** — replaced the w6 CM-8 provisional row with the **w7-refined** row (cost FINAL / accuracy pending); added 2 rows: paid-access **PILLAR-4 GATE PASSED** (all 4) and reversible **CM-13**.

**Deliberately left unchanged:** Parts I–VI core derivations (still exact); §8.2 pricing (w6-correct); Part IX firewall block (w6-correct); §10.4 reading list. No chapter-level rewrites beyond the mandated new §8.5.

## Doc 2 — `negative_results.md` (N2 cost/accuracy split + 5 new entries N32–N36)
- **N2 re-tiered AGAIN (per task):** summary-row + a new **w7 SPLIT** note — the double-provisional resolves cleanly: **COST dimension FINAL** (Hopfield cheaper at ~1 matvec → its own entry N32), **ACCURACY-reversal still grid-pending w8** (`regime-remap-complete`). Verdict not adjudicated; both axes stated per mandate. Output↔handover agree (handover w7: "decisive on cost … accuracy story pending").
- **5 new entries** (schema-complete):
  - **N32** CLU-gate NOT cheaper than Hopfield (the intra-CLU-savings correction) — **tier A, V1** (`regime-remap-2000ep §3`).
  - **N33** reversible O(1) exact ONLY at γ=0 (γ>0 finite (1−γ)⁻ⁿ horizon) — tier B, V3 (`v3-reversible-o1`).
  - **N34** squeeze reach is a **bracket [L, L+p₀sinhζ/M₀]**, not a knife-edge — tier B, V1 (`paid-access-experiments`).
  - **N35** learned entrance-steering UNSOLVED (oracle-placement only; the pillar-4 engineering crux at scale) — **tier A, V1** (paid-access open-risk 1).
  - **N36** (process) v2-referee caught a genuine §3.4↔App D emergent-lifetime contradiction pre-submission — tier C, V2-process (`v2-referee` MF-3).
- Updated summary index (N2 + 5 rows), maintenance line, paper-writer appendix-mining notes (V1: +N32/N34/N35, pillar-4 ADOPTED; V3: +N33) and open-provenance flags (N2 split, N33/34/35 scope, N36-is-process, M3 portfolio flag added to tier note).

## Doc 3 — `philosophy-synthesis.md` (ledger — WROTE the w7 addendum; ⟲ chapters untouched)
Appended `# ⟲ Wave-7 addendum (2026-07-08, doc-curator)`:
- **Chapter deltas:** **Ch.4** (V1 identity DECIDED — the three cheap-baseline results stack [gate memory-agnostic w5 + router wins w6 + Hopfield-cheaper w7/N32]; pillar-4 PASSES all four criteria [CM-12→PASSED] = the ONE physics-specific certificate-backed win; Head ADOPTS, V1 ships ML4PS position paper; honest thesis "certified mechanisms, not benchmark-topping"; N34 bracket + N35 learned-steering crux). **Ch.5** (reversible-O(1) MEASURED, CM-13 — closes ledger backlog #1; 946×, γ=0-only, "reversible memory = conservative memory"; not in `train_chlu` yet). **Ch.1/7** (v2-referee wired the ≈700 crossing + caught §3.4↔App D contradiction pre-submission [N36]; anchor now first-class flag).
- **Scorecard deltas** (rows 4, 5) + **gap-list** (register FULLY DISCHARGED P1–P20; remaining = real data + regime-grid + drafting + reversible-BPTT-in-trainer + N>8/non-MLP + learned steering) + 3 new drafter scope-guards (CM-8 intra-CLU / CM-12 bracket+oracle / CM-13 γ=0-only).
- **Positioning/M3:** recorded the Head's own "V1 is the weakest short" framing as an **explicit Head-decision flag, NOT adjudicated**; V2 borderline→weak-accept after MUST-FIX; V3 done+builds (bib/figure gaps); C-1 split closed by v2-revision MF-1; Cor-3 resolved by Head.
- ⟲ protocol respected: no chapter rewritten; the frozen scorecard (lines ~318–326) left at its ⟲ state; verified the addendum covers all seven outputs before writing.

---

## How I verified
- Every quantitative statement cross-checked against its source-report section before transcription: reach table + certificates + latch + flag-provenance (`paid-access-experiments §7.1–7.3`, §Findings, flag table); Hopfield-cost table (`regime-remap-2000ep §3`); reversible memory/gradient/horizon tables (`v3-reversible-o1` Items 1–2 + the "V3-short sentence"); fix-pack-4 flags/guard (`fix-pack-4` items 1–4 + test log 189/1); the §3.4↔App D ratios (`v2-referee` MF-3: 1.121/1.132/1.152 vs 1.125/1.288/1.052, +5.2% outside band).
- Confirmed `claims_matrix.md` v1.3 (CM-8 refined, CM-12 PASSED, CM-13 new) matches the handover §10 w7 entry and the outputs before transcribing status tags.
- Confirmed the w7 pass's docs-debt items are unchanged in scope (see below).
- No tracked code touched; all writes under `.claude/`. No git footprint.

## Output ↔ handover contradictions found (flagged, NOT resolved — per mandate)
- **None substantive.** The seven w7 outputs, the handover §10 w7 review entry, and `claims_matrix.md` v1.3 are mutually consistent (CM-8 cost-final/accuracy-pending; CM-12 pillar-4 PASSED + Head-adopted; CM-13 reversible measured).
- **Two doc-hygiene lags (not contradictions; Hub-owned files, I don't edit them):**
  1. `claims_matrix.md` **CM-12** still reads "V1 pillar 4 (**Head confirm to adopt**)"; the handover w7 entry (2026-07-08) records the Head **ADOPTED** pillar-4. Version-status lag only — the matrix should update CM-12's appears-in column to "adopted."
  2. `claims_matrix.md` header still self-labels "v1.3, wave-7 review (**2026-07-07**)"; body carries w7-review content. Date-label lag (carried over from the w6 note).
- **One sanctioned tension (documented, not resolved):** N2's accuracy-reversal vs the legacy "26/26 dominant" wording remains **grid-pending w8**; per the handover this is the intended frozen state (cost final, accuracy small-n), recorded as such in N2/N32. Not a disagreement.

## Docs-debt list (flagged for the Hub)
1. **Transfer docs current through w7** on both positive and negative results. All w5 docs-debt items remain PAID (primer §8.2/8.3/8.4/Part IX/§7.3 from prior waves).
2. **§10.4 "go deeper" reading list** still unchanged since w2 — a reversible-integrators / RevNet / gradient-checkpointing entry would help the §8.5 reader; a Hopfield-associative-memory entry would help the §8.3 regime reader. Low priority (unchanged flag).
3. **Ledger frozen scorecard (lines ~318–326)** remains at w3 verdicts by ⟲ design; current state now lives across FOUR addenda (w4/w5/w6/w7). A reader reconciles five locations. A single current-state scorecard would violate "chapters never rewritten" — a Head/Hub call, unchanged from prior flags.
4. **Matrix CM-12 / header labels** (above) — minor, Hub to refresh at w8 review.

## Proposed handover updates (for the Hub)
- Transfer docs current through w7: `HEP_primers.md` (7 edits incl. new §8.5 reversible-O(1) + §8.4 pillar-4 PASS + §8.3 CM-8 cost correction), `philosophy-synthesis.md` (w7 addendum written by curator per standing duty), `negative_results.md` (N2 cost/accuracy split + N32–N36; now **36 entries**).
- **Two live scope-freezes the paper-writer must respect:** (a) **CM-8 accuracy-reversal** — do NOT draft "CLU matches/beats Hopfield" until w8 `regime-remap-complete`; the *cost* correction (Hopfield cheaper, savings are intra-CLU) IS final and must appear. (b) **CM-12/paid-access** — all results are **oracle-placement**; the squeeze crossover is a **bracket**, not a knife-edge; **learned entrance-steering (N35) is unbuilt** = the pillar-4 future-work flag.
- **Three new drafter scope-guards now in both the ledger addendum and the registry:** CM-8 (intra-CLU savings), CM-12 (bracket + oracle), CM-13 (reversible exact only at γ=0).
- **M3 portfolio tension recorded, not resolved:** the Head's "V1 is the weakest short" framing is flagged in the ledger addendum + registry tier note as an explicit Head-decision item — curator did not adjudicate.
- Minor: refresh `claims_matrix.md` CM-12 status ("Head confirm to adopt" → adopted) and the "2026-07-07" header label.
