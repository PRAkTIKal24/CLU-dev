# transfer-docs-sync-w15 — doc-curator report

**Task + acceptance criterion:** 4 transfer docs current through w15, continuing cleanly from `ledger-catchup` (no re-gap); the scoping-negative + the scorer WATCH-item recorded with correct framing (not a false "CLU loses"); KT-GO and V5-ships propagated; corrected XY dictionary validated-on-code in the primers; ledger w15 addendum. Docs-debt list returned.

**Status: done (w15) + RE-RUN folded w16 KT/N10/F2/real-data outputs (Head-directed, 2026-07-19) — see the WAVE-16 RE-RUN section at the bottom.** All four docs current through w15, Every acceptance clause met. Continues cleanly from `ledger-catchup` (which discharged N45 and closed the w11/w12 gap — nothing re-gapped). **Zero output↔handover contradictions found** — every number transcribed agrees with handover §10 (2026-07-19 WAVE-15 review) and the source reports, including the deflations (N59 scout-kill, W1 honest smoke). No code touched; no tracked file touched (all four docs gitignored under `.claude/`).

**Reconciliation-list owner note (protocol §5, first-10-lines rule):** this report contains **no new downstream-reconciliation list I created** — the w15 reconciliation work (v3-revision-4's 3-item CM-10/13/5 list) was already owned and applied *by the writer in-wave*. My remaining owner-actions are Hub-only (matrix untouched; Hub-owned-file sweeps) and listed under Docs-debt below.

**Sources, protocol order:** `AGENT_PROTOCOL.md` (§5 pre-reg + reconciliation-owner rules) → handover §10 WAVE-15 review → `claims_matrix.md` v1.9+ (read, not edited) → my predecessor `ledger-catchup.md` (verified continuity) → the w15 outputs (summarized via handover §10 per the Head's "reports primary, log the index" directive) → the four docs I own. **`v3-pricing-n-scaling` had ALREADY landed** (Hub-recovered at the w15 review) — not pending; folded into the ledger addendum now.

---

## Edits per doc (for Hub diff-review)

### 1. `.claude/negative_results.md` — N59, N60, Watch-item W1 added; N56 updated
- **Masthead "Maintained by":** new **WAVE-15 pass** line (N59 scoping-negative, N60 CSF near-misses, W1 watch-item; continuity confirmed — `ledger-catchup` discharged N45 + closed w11/w12, nothing to carry).
- **Summary index:** N59, N60 rows added; a **⚠ Watch-item W1** callout row (with anchor link to the new section) explicitly flagged NOT-a-negative.
- **N59 (tier B, curator-proposed)** — the relativistic-sampler methods-note is **not a standalone paper**. Full scout verdict transcribed: no-go = known corollary (Monomial-Gamma / Zhang 2017); thermostat = known math (Barndorff-Nielsen NIG 1977/97; Dunkel–Hänggi 2009); rel-SGHMC (Lu 2017) MH-adjusted = **exact** (do NOT claim a bug); only `d·Θ` is new. **Binding on CM-17: cite-don't-claim.** F5-appendix disposition. Provenance caveat: the Monomial-Gamma quote is WebFetch-single-sourced.
- **N60 (tier C, curator-proposed)** — the two CSF-prep near-misses as process negatives fixed-in-`g7b-torus-voraus`: (1) env `--extra eval` gap; (2) voraus episode-labelled ⇒ **AUC-ROC not VUS-PR**. Draft/repro guard included.
- **New "Watch items (not-yet-negatives)" section + W1** — the CLU scorer below baselines on the `--quick` smoke (AUROC 0.38–0.51), recorded as a **WATCH item, explicitly NOT a negative** (CM-3, quality unmeasured). **Promote-to-negative trigger recorded:** the real full-config voraus run (correct metric = AUC-ROC per N60) also losing. Explicit prohibition on citing the smoke as "CLU loses." The section header documents the watch-item mechanism (a C-9 refinement).
- **N56 (Update block):** F2 latent-mass thermostat now **SHIPPED + toy-verified** (`fix-pack-7`, bias `−0.727→+0.0011`); `gibbs_defect_parameter(T)` exposes `d·Θ`; real-Exp-C validation → `fdt-relativistic-expc` (w16). Noted F2 packaging ≠ standalone paper (N59). *(Removed the "new" tag from F2 in the fix ranking — it is cited-prior packaging, only `d·Θ` is ours.)*
- **Paper-writer notes:** N60/W1 folded into the flagship-appendix line (W1 flagged as NOT-an-appendix-negative); N59 into the physics-audit novelty-scope line; the Thread-10 line updated with the w15 1-D-control PASS.
- **Provenance flags:** N59/N60 tiers + W1 framing curator-proposed and justified; WebFetch-single-source caveat; curator did NOT adjudicate scorer quality (transcribed as a non-claim).

### 2. `.claude/future_work.md` — 1-D control / bridge / F2 → SHOWN; V5 → SHIPS; 2-D KT + fdt-relativistic-expc TASKED
- **Thread-10 1-D control entry** → `→ SHOWN (wave-15, xy-1d-control)` with the exact ξ-match (1.5–6.8% over 5 T, residual power 6.7e-34), the broken-symmetry control (`⟨cosΔθ⟩=−0.006` vs `0.446`), all three prereqs behaving, item 3c folded into `kt-2d-csf3`.
- **Thread-10 "big claim" entry** → 2-D KT experiment now **`TASKED(kt-2d-csf3, w16)` — GREENLIT** (was "NOT yet funded"); first real CSF3 physics run, no scorer.
- **Relativistic-register section** → new bullet: the **exact F2 latent-mass thermostat now EXISTS** (shipped + toy-verified, `fix-pack-7`), real-Exp-C validation `TASKED(fdt-relativistic-expc, w16)`; scope/novelty per N59.
- **V5 section header** → **SHIPS** (funded short, `v5-short-draft`): per-claim novelty verdict (b,c NOVEL; a,d cite-substrate), ship rules ("sharp instance, cite the substrate"), leads CM-16b / scopes CM-16a designed-only, 107.77±4.78× vault, freeze ≤ Aug 17.
- **"Real data — THE top gap"** → reframed **bridge-SHOWN / result-TASKED** (precision per the bridge-vs-result distinction): bridge `→ SHOWN (clu-anomaly-scorer)`, result `TASKED(g7b-torus-voraus)`, W1 + N60 cross-referenced.
- **New wave-15 dated footer sweep** (sources; → SHOWN list; status moves; folded negatives; "no genuinely new scientific boundary opened by w15"; C-9).

### 3. `.claude/outputs/HEP_primers.md` — §8.6 validated-on-code; §5.3 no-go-as-established + F2 ships; masthead
- **§8.6 (XY/KT):** new ⭐ **wave-15 Update** — the dictionary is now **validated on the REAL code path**: 1-D control passed (ξ 1.5–6.8%, residual 6.7e-34, broken-symmetry control fails as predicted), **KT is GO**, 2-D funded (`kt-2d-csf3`). Written in house style (concept→numbers→consequence→status tag), teaching it as *the moment a theory thread survived contact with running code (P1)*. The corrected dictionary (`ρ_s=J=2κr*²`, `n=dim(G/H)+1`) was already in-section from w13 — I did not re-derive it, only added the validation.
- **§5.3 (Gibbs no-go):** new **wave-15 Update** — teach the no-go as an **established corollary** (Monomial-Gamma/Zhang 2017), the thermostat as known math (Barndorff-Nielsen NIG; Dunkel–Hänggi 2009), rel-SGHMC MH-adjusted as **exact** (no bug claim); only `d·Θ` is ours; **F2 now SHIPS + toy-verified**; real validation `fdt-relativistic-expc` (w16). Binding: methods-note, not a paper (N59); WebFetch-single-source caveat.
- **Masthead Maintenance line:** wave-15 pass recorded (§8.6 + §5.3), companion-doc status lines updated (future_work swept-through-w15; ledger w15 addendum; negatives current-through-w15 with N59/N60/W1).

### 4. `.claude/outputs/philosophy-synthesis.md` (the ledger) — full ⟲ Wave-15 addendum appended
**⟲ protocol honoured: chapters untouched; a dated Wave-15 addendum appended after Wave-14** (the Hub had NOT written a w15 addendum — I wrote the full one, per protocol point 2). The four task-named deltas landed:
- **Ch. 1/5 — P1 earns its keep:** KT-GO = the first purely-theoretical thread (Thread-10, from a colleague's Ising seed) validated on the real code path → a fundable experiment.
- **Ch. 4 — bridge-vs-result:** the scorer bridge exists but *building it is not touching real data*; gap analysis corrected to bridge-SHOWN / result-PENDING; W1 the honest smoke; N60 the metric blocker.
- **Ch. 6/7 — the scout as overclaim-insurance:** N59 killed a standalone-paper idea *before* a draft; paired with the w14 "agents re-derived and corrected the Hub" pattern into **one method written down** — *the program's honesty is enforced by dedicated adversarial passes, not good intentions* (4 scalps: N59/N56-57/N55/N52). F2 ships.
- **Ch. 2/3 — the shorts:** V5 SHIPS (novelty verdict, ship rules); three referee revisions closed clean; the reconciliation-owner rule worked in-wave (v3-revision-4); `v3-pricing-n-scaling` recovered → MF-2 closed.
- **Scorecard deltas** (rows 1, 4, 6, 7, 2/3), **gap-list** (freeze ≤ Aug 17; two-flagship + four-short portfolio; KT/F2/g7b tasked; scope guards aa–dd), **positioning ripples** (the four-stance method story), **doc-hygiene** (ledger contemporaneous again; v3-pricing NOT pending; zero contradictions; Hub sweep list).

---

## Item-by-item acceptance check
- **Item 1 (negatives — scoping-negative + WATCH framing):** N59 (scoping negative, correct "healthy kill" framing) ✓; W1 recorded as a **WATCH item, NOT a negative**, with the exact promote-trigger (real full-config voraus also losing) ✓; N60 (two CSF blockers as fixed-in-g7b process negatives) ✓; continuity — `ledger-catchup` discharge of N45 + w11/w12 closure confirmed, nothing carried ✓.
- **Item 2 (future_work — KT GO, F2, V5, real-data bridge):** Thread-10 1-D→SHOWN + 2-D funded with exact ξ-match + broken-symmetry control ✓; F2 thermostat shipped-toy-verified + `fdt-relativistic-expc` tasked ✓; V5 SHIPS with per-claim novelty (b,c NOVEL; a,d cite-substrate) + ship rules ✓; real-data gap reframed bridge-built / result-pending (precision) ✓.
- **Item 3 (primers — validated XY dictionary + Gibbs no-go as established):** §8.6 validated-on-code update with parameter-free ξ match + memory–vortex correspondence, corrected dictionary in place ✓; §5.3 Gibbs no-go taught as a known corollary (Monomial-Gamma framing; additive-Gaussian-kick ⇒ Gaussian-smoothed marginal) ✓.
- **Item 4 (ledger w15 addendum):** scout-as-overclaim-insurance written as a method ✓; KT-GO = P1 producing a fundable experiment ✓; bridge-vs-result precision ✓; freeze ≤ Aug 17 + two-flagship status recorded ✓.
- **`v3-pricing-n-scaling`:** NOT pending — had already landed (Hub-recovered); folded into the ledger addendum now (MF-2 closed), not deferred. Acceptance note satisfied ✓.
- **C-9:** nothing deleted anywhere; F2's "new" tag softened to "cited-prior packaging" (a correction, struck not removed); W1 kept as a retained provisional non-negative.

## How I verified
- Cross-checked every transcribed number against handover §10 WAVE-15 review (lines 521–535). Spot-matches all agree: ξ 1.5–6.8% / residual 6.7e-34 / `⟨cosΔθ⟩=−0.006` vs 0.446 (`xy-1d-control`); scorer AUROC 0.38–0.51 (`clu-anomaly-scorer`); N60 both blockers (`voraus-baseline-floors`); F2 bias `−0.727→+0.0011`, `c≳√(dT/m₀)≈28` (`fix-pack-7`); scout citations Zhang-2017/Barndorff-Nielsen/Dunkel–Hänggi-2009/Lu-2017 + MH-adjusted-exact + only-`d·Θ`-new (`scout-relativistic-samplers`); V5 novelty b,c/a,d + Aug-17 freeze + TS4H Aug-19 (`venue-follow-up`); ζ=2.0105 d=4 (`v1-revision-3`); v3-pricing N∈{2,4,8,16}, sync −0.49±0.02, n₁/₂ −0.91±0.03 (recovered `v3-pricing-n-scaling`).
- Confirmed `ledger-catchup` continuity: N45 DISCHARGED + w11/w12 gap closed + ledger runs through w14 before this pass.
- Confirmed the corrected XY dictionary (`ρ_s=J`, `n=dim(G/H)+1`) was already in primer §8.6 from w13 — added validation, did not re-edit the derivation.

## What I deliberately left alone
- **`claims_matrix.md` (v1.9+)** — Hub-owned; already carries CM-10/13/5 lockstep + CM-17 novelty-scope. **Not touched.**
- **`handover_context.md`** — Hub's. Not touched (proposed updates below).
- **Ledger chapters 1–7** — ⟲ protocol. Untouched; all deltas appended.
- **Verdicts, tiers, novelty judgments** — summarized/organized, never reinterpreted. N59/N60 tiers + W1 non-negative framing are **curator-proposed** and flagged as such (Hub may re-tier). Scorer quality, V5 per-claim novelty, and KT-GO were **transcribed**, not adjudicated.
- **Primer §10.3 dictionary ledger** — not extended for w15 (the §8.6/§5.3 updates + masthead cover the wave; adding rows risked scope creep). Flagged as optional below.
- **Hub-owned out-of-scope files** (`brainstorm_log.md`, `research_roadmap.md`) — not swept (docs-debt).

## Docs-debt returned (for the Hub)
1. **Ledger is contemporaneous again** (runs …w13·w14·**w15**; the w11/w12 catch-ups are behind us). No continuity gap.
2. **Three w16 tasks the docs now point at** (all already in the Hub's w16 scope, cross-referenced for consistency): `g7b-torus-voraus` (real-data result; N60 blockers folded), `kt-2d-csf3` (2-D KT; item 3c winding null folded), `fdt-relativistic-expc` (F2 real-Exp-C validation; closes N10 loop on corrected sampler).
3. **Hub-owned-file sweeps still outstanding** (carried from w13/w14, re-flagged + extended): grep `research_roadmap.md`, `brainstorm_log.md`, `claims_matrix.md` for residual `ρ_s ↔ F²` / `n = dim(G/H)` / `13.9`/`13.88` / N57's `d=1`-table-at-`d=784` / R7's angular floor; **and now** for any *"CLU on real data"* phrasing reading the scorer smoke as a result (should be **bridge-built / result-pending**), and any *"voraus VUS-PR"* (should be **AUC-ROC**, N60).
4. **Optional next-pass item:** primer §10.3 could gain w15 dictionary/status rows (KT-GO, V5-SHIPS, F2-shipped) if the Hub wants the ledger-table fully synced; deferred to avoid scope creep this pass.
5. **Provenance caveat to resolve before any citation:** the Monomial-Gamma exact-no-go quote (N59) is WebFetch-single-sourced — verify before it enters F5/a draft.

## Git footprint
**None.** No tracked file created, modified, or deleted. All four edited files are gitignored under `.claude/`. Repo untouched.

## Proposed handover updates (for the Hub)
- **§10 / next-wave block:** *"Transfer docs current through **w15**, contemporaneous (no re-gap from `ledger-catchup`). `negative_results.md`: **N59** (scoping-negative — relativistic-sampler note is not a standalone paper, cite-don't-claim, F5-appendix) + **N60** (two CSF near-misses, fixed-in-g7b) + new **Watch-item W1** (scorer `--quick` smoke AUROC 0.38–0.51 recorded as NOT-a-negative, promote-trigger = real voraus run also losing). `future_work.md`: 1-D XY control / real-data bridge / F2 thermostat **→ SHOWN**; **V5 → SHIPS**; KT-2D + `fdt-relativistic-expc` **TASKED**. `HEP_primers.md`: §8.6 XY dictionary **validated on real code path**, §5.3 Gibbs no-go taught as an **established corollary** + F2-ships. Ledger: full **⟲ Wave-15 addendum** (scout-as-overclaim-insurance written as a method; KT-GO=P1-fundable; bridge-vs-result; freeze ≤ Aug 17; two-flagship + four-short portfolio). Zero output↔handover contradictions."*
- **Standing scope-guards for drafters (verbatim, w15 additions):** (aa) relativistic-sampler note is not a standalone paper — cite Zhang-2017/Barndorff-Nielsen/Dunkel–Hänggi/Lu-2017, claim only `d·Θ`, never "we found a bug"; (bb) the scorer `--quick` smoke is NOT a quality result — never "CLU loses"; the real run at **AUC-ROC (voraus episodes, N60)** is what a real-data claim rests on; (cc) V5 ships "a sharp instance, cite the substrate" (b,c NOVEL / a,d attributed), never "we discovered CD bias"; (dd) KT thesis: stable memory is **topological (winding)**, the universal jump **is** the memory transition, `D_Θ=D_θ/N` is trivial in every dimension.
- **Two Hub actions the curator cannot take:** the Hub-owned-file sweeps (item 3); re-tiering N59/N60 or re-framing W1 if the Hub disagrees (curator-proposed).
- **Process note:** this wave's method story (the funded adversary — scout-to-kill N59, prereg-to-bind N55, attack-to-break N52, re-derive-to-overturn N56/57) is now a four-scalp answer to "how do you know you aren't fooling yourselves?" — recommend it enters the longs' methods section as written in the ledger's w15 positioning block.

---

# WAVE-16 RE-RUN (Head-directed, 2026-07-19) — KT / N10 / F2 / real-data outputs folded

**Trigger:** Head instruction *"re-run the docs sync now so KT/N10/F2/real-data based agent outputs are included per your instructions doc."* The w16 outputs `kt-2d-csf3`, `fdt-relativistic-expc`, `g7b-torus-voraus` had landed in `.claude/outputs/`.

**⚠ Discipline note (the one caveat on this whole re-run):** **there is NO Hub `§10` review entry for w16 yet** — the Hub has not reviewed these outputs. Per protocol my source of truth is normally the Hub review entry; absent it, I applied the Head's standing rule (*reports are the primary source, the running log is the index*), read the three reports in full, **transcribed every number with a report citation, marked each fold "⚠ folded from the report, no Hub §10 entry yet," and reinterpreted no verdict.** The Hub must cross-check at its w16 review. The writer tasks `v5-short-draft`/`iclr-long-skeleton` add no new science and are outside the Head's stated scope — not folded.

## What each report established (transcribed, not adjudicated)
- **`kt-2d-csf3` (KT) — PARTIAL: core physics confirmed on the real path, two exponents laptop-under-resolved.** Kill criterion NOT triggered (`L=8` CLU-Langevin reproduces reduced-XY `ρ_s` to 2.0–6.9% across T_KT); Nelson–Kosterlitz `2/π` jump + `T_KT=0.0898` CLU units to <1%; memory contrast decisive (1-D degrades ∝N^{0.7}, 2-D improves slope +5.0/+3.5); broken-sym null confirmed. **Two quantitative exponents (1-D `τ∝1/N` slope −0.7 not −1; 2-D above-T_KT sign change) are resolution-limited, NOT falsified** → CSF3/A100 `L∈{16,32}` tranche. Verdict: **conditional paper, not a kill.** 🔴 Flags a **factor-2 retraction: `T_KT=0.1786`→`0.0893`** (formula `1.786κr*²` right; value doubled) in Hub-owned files.
- **`fdt-relativistic-expc` (N10 + F2) — done.** Q1: **F2 samples MJ exactly on real d=784 Exp-C** (`Var/(M_eff·T)`=783.5 vs 785, `r_MJ/r_obs`=1.00×, KL‖MJ=0.004≪0.14) — first real-data F2 validation. Q2: the exact sampler **does NOT fix the imbalance, it worsens it** (`f(3589)` 0.609→0.896; Spearman(Hfin,f3589)=+1.000) ⇒ **N10 CLOSED**, landscape not sampler. One prereg sub-prediction (F2≈fdt) falsified & reported.
- **`g7b-torus-voraus` (real-data) — done (CSF launch handed to Head).** Literal joint-angle→so2 torus map built + wired + **runs on real voraus** (both CSF blockers fixed, N60; suite 326 passed; balanced-subset pipeline check finite/in-range/above-chance — NOT the floor). **The real floor still awaits the Head's CSF launch.** P3-control rank-invariant at κ=0.05 (design caveat).

## Edits per doc (w16 fold)
- **`negative_results.md`:** masthead w16 line (with the no-Hub-review flag); **N10 index row + entry → CLOSED** (wave-16 block: F2 samples Gibbs yet worsens imbalance); **N56 F2 block → real-data-validated** (fix-pack-7 Open-2 discharged); **N60 → both blockers RESOLVED** (env `--extra eval`+jax-pin, episode-AUROC regression-locked, suite 326); **W1 → updated** (literal torus map built + runs on real voraus, result still CSF-pending; promote-trigger widened to cover the balanced-subset check too).
- **`future_work.md`:** Thread-10 2-D KT entry → **core `→ SHOWN (wave-16)`, conditional paper** (kill passed, `2/π` jump, T_KT<1%, memory contrast; two exponents CSF-pending) + the T_KT factor-2 reconciliation for the Hub; F2 thermostat entry → **real-Exp-C `→ SHOWN`**; O8/N10 → **CLOSED**; real-data bridge → **literal torus map built + real-voraus-runnable, result CSF-pending**; new wave-16 footer sweep.
- **`HEP_primers.md`:** **§8.6** wave-16 update (2-D KT memory phase MEASURED on the real path; `T_KT=0.0898`; the factor-2 `0.1786→0.0893` numeric correction; conditional-paper verdict); **§5.3** wave-16 update (F2 real-data-validated + N10 CLOSED + the honest prereg-falsification of F2≈fdt + the RGE §2b "drift vs quench depth" correction); **§5.2** pointer updated (refuted→reopened→upheld→**CLOSED w16**); masthead maintenance line (w16 pass, no-Hub-review flag).
- **`philosophy-synthesis.md`:** full **⟲ Wave-16 addendum** (chapters untouched): Ch.1/5 (KT measured on real code — P1 arc completed, conditional paper), Ch.6/7 (**N10 CLOSED on the exact sampler — the four-wave instrument-validity arc completes**; F2 real-data-validated; the 5th prereg scalp), Ch.4 (literal torus map built + runs on real robot data, floor launch-pending); scorecard deltas (rows 1,6,7,4); gap-list; scope guards ee/ff/gg; positioning; doc-hygiene with the **pre-Hub-review caveat** front-and-centre.

## Verified
- Numbers spot-checked against the three reports' own tables (kt-2d §2/§3/§5; fdt-relativistic §1/§2 + VERDICT block; g7b §"balanced smoke"/Findings). Mutually consistent — all at commit `e3c8931`.
- **`T_KT=0.1786` grep-verified ABSENT from all four transfer docs** (they carry only the formula `1.786 κ r*²`); the retraction is a **Hub-owned-file** action, flagged not executed.
- Nothing deleted (C-9): N10's full refuted→reopened→upheld→CLOSED trail retained; W1 kept and updated, not graduated (no full-config run yet).

## Docs-debt added by the w16 fold (for the Hub)
1. **⚠ Cross-check this w16 fold at your review** — it is ahead of your §10 entry. If your review's verdicts differ from the reports I transcribed, they override; flag me nothing, just correct in place (these are gitignored transfer docs).
2. **🔴 Retract `T_KT=0.1786`** (factor-2) in Hub-owned `xy-lattice-theory` §4.4/§7 and `xy-1d-control` — correct value `0.0893` (measured 0.0898). Not in my docs.
3. **Matrix rows for the Hub** (CM-17 / N10): add the **F2 real-data-validation** numbers and the **N10-CLOSED** verdict at your w16 review; also the RGE §2b "drift-invariant / quench-depth-not" correction. I did not touch the matrix.
4. **Two reconciliation lists the reports declared** (owners = Hub): `fdt-relativistic-expc`'s (CM-17 F2 real-data + RGE §2b correction) and `kt-2d-csf3`'s (the T_KT retraction).
5. **KT + real-data are both "measured-core / scale-pending"** — the two flagships each want one CSF launch (KT `L∈{16,32}` tranche; voraus floor). Neither result is citable-as-a-number until those land.
