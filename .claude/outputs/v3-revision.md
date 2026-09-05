# v3-revision — paper-writer report

**Task + acceptance criterion:** fold reversible-O(1) (CM-13) into §3.5, reconcile C-1 (no audit paragraph), and close/flag the 6 editorial questions from `v3-short-draft.md`; PDF rebuilds. Edit `.claude/papers/v3-short/` in place.
**Status: done.** All 5 task items executed; `draft.md` (canonical) + `draft.tex` synced; PDF rebuilds (tectonic exit 0, cosmetic underfull-hbox warnings only, no errors); CHANGELOG v0.2 line added. No tracked repo files touched — **no git footprint** (all under gitignored `.claude/papers/`).

## What I did (by task item)
1. **§3.5 reversible-O(1) filled from CM-13 verbatim** (was a marked slot). Numbers transcribed from `v3-reversible-o1.md`, never rounded:
   - 946× peak activation-memory reduction (6.3 KB reversible vs 6.00 MB standard @ T=1024, N=2), O(1)-in-T vs O(T), standard linear / reversible flat.
   - gradients match stored-activation BPTT ≤2e-6 rel (float32) / ~1e-15 (float64) at T≤1024.
   - ≈0.9× wall-time **with the CPU/small-D caveat in-sentence (C-5)**; explicit note that memory scaling is structural/generalizes but wall-time parity is regime-specific (up to ~2× on GPU/large potentials).
   - **γ=0-only exactness** as the honest scope: γ>0 amplifies error (1−γ)⁻ⁿ → finite horizon (≈3.3e4 @γ=1e-3, ≈5e2 @γ=0.05, float64; ~100× shorter f32). "Reversibly-trainable memory is conservative memory" — tied to the γ=0-memory/γ>0-forgetting budget framing.
   - **NOT yet in the shipped `train_chlu`** (gradient mechanics on untrained models + final-state loss only) stated in-section and in §5.
   - Labeled **structural measurement on untrained models** (property of integrator + autodiff graph), i.e. not shoe-horned into the verification/evidence binary (both are about physics vs learned-system; this is a systems/structural property). C-6 scope stated next to the claim.
   - Added supporting edits so the paper stays coherent: contribution bullet 5 (reversible), one abstract systems-corollary clause, App A.6 flag-provenance table, appendix Figure 5 asset entry.
   - **RevNet/checkpointing prior-art moved to §4** (expanded: gradient checkpointing O(√T) middle-ground + RevNet/momentum-net O(1) lineage; positions §3.5 as *measuring* the trade on this primitive and names the γ=0-only boundary). Published anchors left as `[·]` "to be finalized from the scout bibliography" (rule 5 — not fabricated).
2. **C-1 reconciliation — CONFIRMED, no audit paragraph (editorial Q1 CLOSED).** Removed the header editorial note that was holding/pointing to canonical audit text; replaced with a one-line statement that per charter C-1 (reversed 2026-07-07) no defensive physics-audit paragraph appears and the corrected-mechanism theory lives in the companion note. V3 never contained an audit paragraph; this only removes the meta-note. (Per task: V2 draft is being fixed to match under v2-revision MF-1, so the cross-short split closes there — no V3 action beyond this.)
3. **Headline figure (editorial Q3) — FLAGGED in-draft to Hub, not dropped.** The Figure-1 asset line now states the S-vs-N *scaling-curve* PNG may not exist standalone, instructs to embed the **bars** fig (`v3-interference-ntk/fig1_interference_bars.png`) meanwhile, and asks the Hub to confirm whether the analyst generates the scaling-curve PNG from `v3-interference-ntk/through_training.json` + item-3 data.
4. **Modular/interference bib (editorial Q5) — LEFT `[·]`, FLAGGED.** The §4 modular-vs-monolith paragraph keeps placeholder anchors; the dedicated `scout-modular-interference` task exists (`.claude/tasks/scout-modular-interference.md`) but its output has **not landed** (`.claude/outputs/scout-modular-interference.md` absent). No citations fabricated (rule 5). Splice-ready once the scout brief lands.
5. **C-2/C-5/C-10 discipline preserved** — scale qualifiers in-sentence, appendix-maximal (nothing pruned; added A.6 + Fig 5), designed=verification / learned=evidence labels intact.

## How I verified
- Build: `cd .claude/papers/v3-short && tectonic draft.tex` → `draft.pdf` (141 KB, exit 0). Only 5 residual underfull-hbox warnings (cosmetic, prose justification), zero overfull, zero errors.
- md↔tex parity: every §3.5/§4/§5/contribution/abstract/App-A.6 edit made in both files with identical numbers.
- Number provenance: §3.5 + A.6 numbers transcribed verbatim from `v3-reversible-o1.md` (Item-1/Item-2 tables, flag-provenance table, "V3-short sentence"). CM-13 wording cross-checked against `claims_matrix.md` §2 row 13 — all load-bearing facts present (946×/940×, ≤2e-6 f32, ≈0.9× wall-time GPU-caveat, γ=0-only, horizons, "not in shipped trainer").

## Compliance self-check
- **CM-13 used verbatim**; γ=0-only exactness + "not in shipped trainer" + CPU/small-D wall-time caveat all in-sentence.
- **CM-3 (forbidden) absent** — V3 has no energy-as-confidence/routing content; reversible section makes no such claim.
- **Canonical constants (matrix §1)** unchanged and consistent (latch, (1−γ)ⁿ, μ⁻², floor, h*, det J, inertial M vs spectral μ).
- **Naming:** CLU-introduced-as-CHLU continuity sentence still present (§1); "reversibly-trainable memory is conservative memory" uses the budget framing, not bare "mass".
- **C-1:** no audit paragraph (charter-compliant). **C-5:** reversible claims carry CPU/small-D/untrained/D≤16 qualifiers in-sentence. **C-6:** γ=0-only + not-in-trainer fine print next to the claim. **C-10:** appendix-maximal (added, pruned nothing).

## Open questions / follow-ups / risks (for the Hub)
- **Editorial Q3 (headline PNG):** need a decision — does the analyst generate the S-vs-N scaling-curve PNG from `v3-interference-ntk/through_training.json` + item-3 data, or does the bars fig ship as the headline? Draft currently instructs "embed bars meanwhile." No `\includegraphics` is wired yet (asset paths only); the referee/engineer pass should point at final PNG paths.
- **Editorial Q5 (modular bib):** `scout-modular-interference` output has not landed; §4 anchors remain `[·]`. This is the one remaining citation-free paragraph — blocker for ship, not for the referee pass.
- **Editorial Q4 (theory-note citation string):** still pending live arXiv id + Head title; §4/References use "Anonymous (2026); the theory note" placeholder (matrix §4 open slot) — V2 and V3 must adopt the identical final string.
- **Editorial Q2 CLOSED** (reversible now landed) and **Q1 CLOSED** (C-1 confirmed no audit paragraph).
- **Editorial Q6 (venue length):** still needs the exact ML4PS page limit for the pruning pass; §3.5 and the §3.1 wormhole-skeleton paragraph remain first pruning candidates if tight (do not prune now, C-10).
- **New-CM-row candidate:** the `v3-interference-ntk` report proposed promoting the firewall to its own CM-x (currently folded under CM-9); §3.2 already targets the approved wording if the Hub promotes it.
- **Reversible = ICLR feedstock:** §3.5 is deliberately written as a *structural* result with the four load-bearing follow-ups enumerated in §5 (wire into `train_chlu`; GPU/HBM measurement at T≳4k where standard BPTT OOMs; checkpointing O(√T) Pareto baseline; mixed-γ segment accounting) — this is the debug/feedstock record for the ICLR systems claim.

## Files touched (all gitignored `.claude/papers/v3-short/`)
- `draft.md` — canonical: header C-1 note, abstract clause, contribution 5, §3.5 (filled), §4 reversible ¶, §5 reversible-scope + horizon, App A.6, Figure-1/Figure-5 asset lines.
- `draft.tex` — same edits synced; rebuilt → `draft.pdf` (exit 0).
- `CHANGELOG.md` — v0.2 line.

## Proposed handover updates (for the Hub)
- **V3 short v0.2 ready for `paper-referee` (w8/w9 `v3-referee`).** §3.5 landed from CM-13; C-1 reconciled (no audit paragraph); Q1/Q2 closed, Q3/Q4/Q5/Q6 flagged with specific asks.
- **Two ship-blockers remain, both external to the writer:** (i) `scout-modular-interference` output to fill §4 `[·]` anchors; (ii) theory-note citation string (live arXiv id). Both are matrix §4 open slots.
- **One asset decision:** headline S-vs-N scaling-curve PNG generation (analyst) vs bars-fig-as-headline.
- **Cross-short C-1 now consistent** once v2-revision MF-1 lands (V2 audit ¶ removal) — V3 side is charter-compliant as of this revision.
