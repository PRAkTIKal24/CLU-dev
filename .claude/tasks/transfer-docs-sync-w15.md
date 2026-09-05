# Task: transfer-docs-sync-w15 — the wave-15 doc pass (w16, curator)

- **Agent:** `doc-curator` · **Output:** `.claude/outputs/transfer-docs-sync-w15.md`
- **Read first:** protocol (**§5 now carries the pre-registration rule + reconciliation-list-owner corollary — both live**) · your standing 4 docs (`HEP_primers.md`, `philosophy-synthesis.md` ⟲ addendum, `negative_results.md`, `future_work.md`) · your own `.claude/outputs/ledger-catchup.md` (w15's predecessor — make sure w15 continues cleanly from it, no re-gap) · the w15 outputs: `xy-1d-control`, `clu-anomaly-scorer`, `voraus-baseline-floors`, `fix-pack-7`, `scout-relativistic-samplers`, `venue-follow-up`, `v1-revision-3`, `v3-revision-4`, `v2-revision-5` (and `v3-pricing-n-scaling` **if it has landed** — check; if still pending, note it and cover it next pass) · `.claude/claims_matrix.md` (v1.9+ with the w15 edits — CM-10/13/5 lockstep, CM-17 novelty-scope; **do not edit the matrix**).
- **Scope:** transfer docs only. Never edit drafts/matrix/`chlu/`.

## 1. `negative_results.md` — new entries + one continuity fix
- **The relativistic-sampler methods note is NOT a standalone paper** (`scout-relativistic-samplers`): the no-go is a known corollary (Monomial-Gamma/Zhang 2017), the thermostat is known math (Barndorff-Nielsen NIG; Dunkel-Hänggi 2009), rel-SGHMC's MH-adjusted main method is exact. Only `d·Θ` is new → F5 appendix. **Record as a scoping negative** (a paper idea we correctly killed) — this is the healthy kind, and it saves a future overclaim.
- **The CLU scorer is below statistical baselines on the --quick smoke** (`clu-anomaly-scorer`: AUROC 0.38–0.51). **NOT a real negative** — it's a smoke, quality is unmeasured until the real CSF run. **Record it as a WATCH item, not a negative**, with the promote-to-negative trigger = the real full-config voraus run also losing. (Don't let a smoke number become a cited "CLU loses.")
- **The two CSF blockers** (`voraus-baseline-floors`): env `--extra eval` gap; voraus is episode-labelled ⇒ AUC-ROC not VUS-PR. Record as fixed-in-`g7b-torus-voraus` process negatives (the "we almost compared cross-protocol" near-miss is worth the registry).
- Continuity: confirm `ledger-catchup` discharged N45 and closed the w11/w12 gap; if anything it flagged is still open, carry it.

## 2. `future_work.md`
- **KT is GO** (`xy-1d-control`): the 2-D KT memory-phase experiment is greenlit and scoped (`kt-2d-csf3`). Update the Thread-10 section from "gated on xy-1d-control" to "1-D validated on real path, 2-D funded." Record the exact ξ-match (1.5–6.8% over 5 T) and the broken-symmetry control result.
- **The exact relativistic thermostat exists** (`fdt_relativistic`, fix-pack-7) — the F2 latent-mass fix is shipped and toy-verified; real-Exp-C validation is `fdt-relativistic-expc` (w16). Update the CM-17/F-6 sections.
- **V5 SHIPS** (venue-follow-up erosion verdict): move V5 "Forgetting" from candidate to funded-short; record the novelty verdict per-claim (b,c NOVEL; a,d cite-substrate) and the ship rules.
- **The real-data bridge exists** (`clu-anomaly-scorer`): update the "no result has touched real data" gap — the *bridge* is built; the *result* is pending (g7b). Precision matters here.

## 3. `HEP_primers.md`
- New/updated: the **XY/KT dictionary validated on the real code path** (the parameter-free ξ match; the memory–vortex correspondence) — the physics-for-ML-experts version, with the corrected dictionary (`ρ_s=J=2κr*²`, `n=dim(G/H)+1`). And the **Gibbs no-go as a known corollary** (additive-Gaussian-kick ⇒ Gaussian-smoothed marginal; the Monomial-Gamma framing) so the primer teaches it as established, not as ours.

## 4. Ledger (`philosophy-synthesis.md`)
Wave-15 ⟲ addendum. Chapter deltas that matter:
- **The scout as overclaim-insurance:** `scout-relativistic-samplers` killed a standalone-paper idea *before* it entered a draft. Pair with the w14 lesson (agents re-deriving caught the Hub's matrix errors). The meta-pattern: **the program's honesty is enforced by dedicated adversarial passes, not by good intentions** — write it as a method.
- **KT GO** = the first time a purely-theoretical thread (Thread-10, born from a colleague's Ising seed) survived contact with the real code path — the "physics as a design library" principle (P1) producing a fundable experiment.
- **The bridge-vs-result distinction:** building the scorer is not touching real data; be precise in the ledger's gap analysis.
- Record the freeze-date move (≤ Aug 17) and the two-flagship status as of w15.

**Acceptance:** 4 docs current through w15, continuing cleanly from `ledger-catchup` (no re-gap); the scoping-negative + the scorer WATCH-item recorded with correct framing (not a false "CLU loses"); KT-GO and V5-ships propagated; the corrected XY dictionary in the primers; ledger w15 addendum. Report a docs-debt list. **If `v3-pricing-n-scaling` is still pending, say so and schedule it for the next pass** rather than guessing its result.
