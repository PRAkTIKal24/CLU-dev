# shorts-toy-retirement-sweep — doc-curator report

Task + acceptance criterion: per-draft map of every passage in V1/V2/V3/V5/B′ that leans on the toy compositional substrate formally demoted by **Add.16 §A44.1** (folded as **matrix §0.14**); quote + location + substrate classification + recommended disposition + citation per finding, positive control on every clean draft, report-only.
Status: **done**

**Dial declaration (protocol §7, echoed before first result):** **Dial — none: instrument/recon/doc sweep.** No performance number is produced, so no laundering control applies. *Falsifies the deliverable:* a passage in any swept draft that rests on the demoted compositional family and is absent from this report. *Does NOT falsify it:* a draft resting on a **designed C1-era testbed** or on a **CL cell** — those are excluded from §A44.1 by the task's own critical distinction and by §0.14's scope statement. ⚠ The task file itself carries **no DIAL DECLARATION block** (see Flags F5).

**Headline (first 10 lines, per protocol §5 reconciliation-owner rule): ZERO §A44.1 findings across all five drafts. No draft edit is owed on the demotion. There is no downstream reconciliation list from this sweep.** Four *adjacent* items are flagged for the Head under **different rulings** (§A14.8 gyms, criterion-4 venue admissibility) — they are explicitly **not** §A44.1 findings and I do not recommend dispositions on rulings outside my remit.

---

## 0. What I swept, and its provenance

| # | file swept | revision swept (from its CHANGELOG) | date |
|---|---|---|---|
| 1 | `.claude/papers/v1-short/draft.md` | v0.4, w15 revision (paper-writer) | 2026-07-19 |
| 2 | `.claude/papers/v2-short/draft.md` | v0.6 (CM-16 split + CM-17 App-F qualifier), wave-15 | 2026-07-19 |
| 3 | `.claude/papers/v3-short/draft.md` | v0.6 (`v3-pricing-n-scaling` fold, w16) | 2026-07-19 |
| 4 | `.claude/papers/v5-short/draft.md` | v0.1 (w16, first draft) | w16 |
| 5 | `.claude/papers/bprime/draft-r5.md` | r5 — the **citation-layer** fold (C2W5) | 2026-08-02 |

⛔ **`draft-r6.md` DOES NOT EXIST at run time.** `Glob .claude/papers/bprime/*` returns exactly `draft-v1, draft-r2, draft-r3, draft-r4, draft-r5, CHANGELOG.md`. **I swept `draft-r5.md`**, per the task's fallback. Findings are forward-portable (they are nulls; the r6 fold folds N276 + N294/N295 + N296, none of which is compositional-family material — Add.8 spoke list).

**Chronological corroboration of the null (context, not evidence in itself).** All four shorts were last revised **2026-07-19 / w16**, i.e. before Campaign 2's compositional cells existed as a substrate; B′ r5 is **2026-08-02** and its C2W5 fold was **citation-layer only** (CHANGELOG line 128). The demoted family's two homes — the C2W5-era `set_code`/`P`-particle occupancy work (matrix §0.9, §A20.2) and the C2W11 compositional wave (matrix §0.14, `d_addr = 4`, `N_a = 32`, `F = 4`, `m = 8`) — both post-date or sit outside every swept revision. Shorts charter §3 says the same thing from the other side: *"Every short pre-dates C1W20+ and ALL of C2."* This is consistent with, and independently corroborated by, the per-file searches below.

## 0.1 Method (task §Method requirements 1–3)

⛔ **No directory-level Grep over `.claude/` was run at any point** (known false-negative failure mode on this machine). Every search below names a single file.

Concept-set searched per file (not a single keyword): `compositional` · `composition` · `P-particle` · `particle` · `set code` / `set_code` · `occupancy` · `exact-set` / `exact set` · `orgdiv` · `organizer` / `organiser` · `tier ii` / `tier-ii` · `d_addr` · `N_a` · `multiset` · `slot` · `bag-of` · `toy` · plus the §A20.2 number-side probes `0.272` / `0.2719` / `0.0504` / `P = 4` / `read protocol`, and the §A14.8-adjacent probes `gym` / `dividend` / `launder` / `substrate` / `deepsets` / `planted` / `placement` / `C2W`. Every hit was **read in context** before classification; no hit was classified from the match line alone.

---

## 1. `v1-short/draft.md` — **CLEAN (0 findings)**

**Substrates this draft actually rests on**, established by reading §3 (designed analytic double-well + `SO(2)` latch sector, oracle channel placement, dim 2/4, 5 seeds — Appendix A.1 flag table), §4.1/§4.3 (**MQAR-style CLU-EBM**, vocab-256, kv ∈ {16,24,32}/{16…128}, `v1-pivot`, `regime-remap-2000ep`, 198 jobs — Appendix A.2/A.4), §4.2 (`v1-router-baseline`, N ≤ 8 lattice), Appendix F (float64 **toy EBMs**, `thread9-mh-kernel`). All are **designed C1-era testbeds or MQAR learned-memory cells** — the venues the task's critical distinction explicitly protects, under their existing scope clauses (§5 "Scope", C-5).

**Every `toy` occurrence adjudicated (3, all non-§A44.1):**
- L465 — *"Toy config: double well $V=\beta(q^2-a^2)^2$…"* (Appendix E, analytic verifications) → **designed** analytic well; `paid-access-theory`.
- L480 — *"Grade: theory-complete on toy EBMs. No runs on trained CLU checkpoints are claimed anywhere in this paper."* (Appendix F head) → **designed** float64 EBM; already carries its own scope sentence.
- L494 — *"…$L_1$ error … $0.0995 \to 0.0065$ … float64, toy EBM."* (Appendix F.2) → same.

**Appendix D (negatives, C-9) read in full** (L443–459): N1, N2, N2b, N3, N24, N30, N31, N23 + three fine-print/scope items. Every entry is MQAR/Hopfield, routing, or the designed certificate battery. **No compositional-family negative is imported.**

**Positive control (mandatory for a zero-finding draft):** on the *same file*, `wormhole` → **46 hits**; the broad concept pattern → hits at L465/480/494/506/514. The file is searchable and the negative is trustworthy.

**Disposition: none. No scope sentence, demotion or cut is owed on §A44.1 grounds.**

## 2. `v2-short/draft.md` — **CLEAN (0 findings)**

**Substrates:** designed `so2_invariant` Experiment-D checkpoints (dim 4, hidden 64, `newtonian_learned`, 150 ep, seeds 42–46) and the **emergent MLP** arm (3 seeds) — Appendix A.1–A.8; the `f1-gmor-condensate` spurion probe; `t-lever-forgetting` / `v5-gate` for the Appendix-J finite-$T$ face. All **designed C1-era / trained-checkpoint** venues.

**Both concept hits adjudicated:**
- L273 — *"**N12/N13/N14/N15 — friction-field composition/placement negatives**…"* (Appendix F). "Composition" here = **friction-field composition** (governor + learned $\gamma$-field), and "placement" = locus placement of a friction field. → designed C1-era $\gamma$-field study, logged out-of-scope for ICLR. **Not** the compositional family, **not** the C2W11 placement-organizer sense of "placement". *(Guard for the Head: the word "placement" collides lexically with §0.14's placement-organizer vocabulary; it is a false friend here.)*
- L409 — *"Controlled against an exact **AR(1)-coset toy**: $n_{1/2}(\text{trained})/n_{1/2}(\text{toy})=1.0020\pm0.0495$ over $20$ cells"* (Appendix J.5) → **designed** analytic control process, the instrument that de-biases $\ell_\theta/\Delta$. Legitimate.

**Positive control:** same file, `GMOR|coset` → **41 hits**; §A20.2 number-probes (`0.272|0.2719|read protocol|occupanc|exact.set|set.code`) → **0**.

**Disposition: none owed on §A44.1.**

## 3. `v3-short/draft.md` — **CLEAN (0 findings)** ⚠ *the highest-risk draft lexically, and the one that survives on the merits*

This is the draft whose title contains **"Composing"** and which uses *composition* ~a dozen times. **Every such use is architectural** — composing $N$ CLU units under one joint Hamiltonian — and none refers to the demoted compositional *family*. Evidence, read in context:
- §1 (L23): *"This is not a new cell but a **composition law**"* — a lattice $H=\sum_i[T_i+V_{\theta,i}]+\sum_{(i,j)\in E}V_c(q_i,q_j)$.
- §3.1 heading (L53): *"The CLU-Net and its **guarantees-by-composition**"* — joint conformal symplecticity, bit-level vanishing-coupling reduction.
- §5 (L217): *"The **composition**, owned"* — the firewall-vs-priced-channel argument.

**Substrates:** (a) `channel_spring` **designed `SO(2)`** lattices with frozen U(1)-preserving coupling, $N\in\{2,4,8,16\}$, 5 seeds, pre-registered (§3.3, Figs 9–10, `v3-pricing-n-scaling`) — explicitly protected by the task's distinction; (b) MLP-potential lattices measured **at initialization** for the interference firewall (`v3-interference-extra`, 8–12 seeds); (c) synthetic two-timescale data for banding (§3.4).

**One hit I chased to the bottom rather than pattern-matching (task §Method 2):** Appendix H's **`block_tied` "deep-sets" arm** (L437, L445) — *"block-tied $V=\sum_i f(q_i)$ (deep-sets) … per-pair $\bar R=\mathbf{0.97}$ … the worst arm"*. **DeepSets** is also the encoder in the program's compositional address-leak material (N136 → N157). **It is not the same object here:** Appendix H's arm is *"a **single** potential $V:\mathbb R^{Nd}\to\mathbb R$ (one module; no lattice, no coupling graph, no Hamiltonian coupling)"* (L429) used as a **parameter-sharing control** in the interference battery, i.e. a permutation-equivariant *potential*, not a set-code task or an occupancy read. → **designed C1-era interference battery.** Classified, not flagged.

**Appendix C ("honest unmeasured list", L326–337) read in full** — trajectory-mediated banding, irrep firewall, block monolith (discharged), dynamical interference half-life, through-training interference at $N>2$, parameter-matched monolith, $\kappa$-sweep at $N=16$, non-MLP potentials/richer graphs, the resolved N16 exponent item, intra-unit wormhole. **No compositional-family item, and no compositional future-work promissory.**

**Positive control:** same file, `firewall|interference` → **57 hits**; §A20.2 number-probes → **0**; `gym` → **0**.

**Disposition: none owed on §A44.1.**

## 4. `v5-short/draft.md` — **CLEAN (0 findings)**

**Substrates:** designed `so2_invariant` checkpoints (5 seeds) + emergent MLP arm (3 seeds), `t-lever-forgetting` §2–4 and `v5-gate` §3.4, `fdt` + `newtonian_learned`, float64, dim 4 (§2 "Trained models", Appendix A). §3.4 is the **owned designed-symmetry-precondition negative** (N46). All designed / trained-checkpoint venues.

**Concept hits adjudicated (4):** L71 *"this is an operator identity on trained weights, **not a toy** (Cor-13 of the theory note)"* — an explicit *disclaimer* of toy status; L73 and L161 *"exact AR(1)-coset toy"* / *"ideal-coset toy $3.61$"* — the same designed AR(1) control instrument as V2 App-J.5; L163 (App C) same family. None is the compositional substrate.

**Positive control:** same file, `budget cube|vault` → **28 hits**; §A20.2 number-probes (`0.272|0.2719|0.0504|P = 4|read protocol|occupanc|exact.set`) → **0**.

**Disposition: none owed on §A44.1.**

## 5. `bprime/draft-r5.md` — **CLEAN of §A44.1 (0 findings)**; one adjacent flag under a *different* ruling

**Substrate:** the C2 audit harness — a synthetic stream of `(address ∈ R^4, payload ∈ R^1)` items written into an atom store `V_θ(q)=α‖q‖²−Σ_j A_j exp(−‖q−c_j‖²/2s_j²)` with staged admission, audited against a **matched-bytes launder table** of `K` live `(key, payload)` rows (§Appendix P.1–P.2, L2592–2615; audited cell `N_at = 192, D = 5, K = 5, d = 4, m = 1`), plus the rival table (TTT-Linear · TTT-MLP · DeltaNet · GDN · GDN2 at $n=9$) and FB4.

**Searched and absent:** `compositional` **0** · `P-particle` **0** · `set code`/`set_code` **0** · `occupancy` **0** · `exact-set` **0** · `orgdiv` **0** · `organizer` **0** (only *"a store organi**s**ed well enough to be safe…"*, the paper's own thesis sentence, L51/1026/1380 — the ordinary-English participle, not the C2W11 *placement organizer*) · `tier ii` **0** · `d_addr` **0** · `N_a` **0** · `0.272`/`0.2719` **0** (the only near-miss is a `deltanet` hyper-parameter row at L2114, `0.2729`, and the recurring rival margin `−0.2732 ± 0.0395` — both rival-table numbers, unrelated) · `read protocol` **0** · `gym` **0**.

**Positive control:** same file, `launder` → **64 hits**; `toy` → hits at L163, 169, 551, 595, 1319, 1325, 1657, 2182 — every one a **two-well / third-party scaling toy** used as a designed comparator (e.g. L2182: *"the free-fall factor imported from a two-well toy **does not transfer**"*), and L163 is the paper's own grading rubric (*"**Verification** — results on designed testbeds: architecturally-specified potentials, two-well toys"*).

**Disposition on §A44.1: none owed.** (Adjacent flag F1 below.)

---

## 6. Per-draft count summary

| draft | §A44.1 findings | scope-sentence | demotion | cut | positive control run (same file) |
|---|---|---|---|---|---|
| v1-short v0.4 | **0** | 0 | 0 | 0 | `wormhole` → 46 hits ✅ |
| v2-short v0.6 | **0** | 0 | 0 | 0 | `GMOR\|coset` → 41 hits ✅ |
| v3-short v0.6 | **0** | 0 | 0 | 0 | `firewall\|interference` → 57 hits ✅ |
| v5-short v0.1 | **0** | 0 | 0 | 0 | `budget cube\|vault` → 28 hits ✅ |
| bprime draft-r5 | **0** | 0 | 0 | 0 | `launder` → 64 hits ✅ |
| **total** | **0** | **0** | **0** | **0** | 5/5 controls fired |

**Drafts CLEAN: all five (V1, V2, V3, V5, B′ r5).** Each carries its positive control in its own section above and in the table.

**Passages I could not classify: NONE.** Every hit resolved to a named source experiment or an explicit in-draft definition. (The one hit that required a definitional read rather than a citation — V3's `block_tied` "deep-sets" arm — is resolved in §3 above from Appendix H's own text, L429; it is *not* left as a guess.)

---

## 7. Adjacent items — FLAGGED, NOT FINDINGS (different rulings; the Head decides, I do not recommend)

These are **out of §A44.1's scope by the task's own critical distinction.** I record them because §A44.1 itself cites the first one as its precedent, and because a Head reading "all five drafts are clean" should know exactly *what* they are clean of.

- **F1 — B′ r5's whole evidence base sits on the C2 audit harness, which §A14.8 demoted as a claim venue.** Add.16 §A44.1 says the compositional family retires *"(the same demotion the gyms received in §A14.8)"*, and §A14.8 item 8 reads: *"**Gyms are HEAVILY DEMOTED**: designed families **retire as claim venues** — FB4 killed 3 of 4 — and remain **only as regression instruments for the collapse modes.** Claim venues are tier-ii/tier-iii tasks."* (registry L1945; charter Add.? §A14.8 item 8, L253). ⚠ **Countervailing authority in the same ledger:** Add.16 **§A45** banks *"the B′ audit-paper evidence base (rivals n=9 · FB4 · C6 525× · theorems · no-daylight at three substrates · both C2W10 tripwires as protocol evidence)"* as **quotable with its registered caveats**. Two rulings point opposite ways for the same artifact. ⛔ **I do not resolve this** (curator discipline: flag, never reinterpret a verdict). Owner: Head/Advisor.
- **F2 — V1's learned-memory pillars (§4.1/§4.3) are MQAR cells, and MQAR appears in the criterion-4 confirmation list.** Matrix §0.14's C2W10 block records criterion 4 at *"SIX confirmations (**MAD/zoology/MQAR** · three of four gym families via FB4 · ELEC2 + Covertype + Poker-hand · the SAM-kNN real-stream family · INSECTS · Metro)"* (registry L3145). That is a **venue-admissibility** ruling about where a *performance* claim may live, not a §A44.1 substrate demotion, and V1's MQAR material is already fenced by CM-2's scope clause and §4.1's own memory-agnostic sentence. Flagged only so the one-by-one pass takes the decision deliberately.
- **F3 — lexical false friends the rewrite pass should not "fix" by reflex.** V3's *composition / composing / composition law* (unit-lattice), V2's *friction-field **composition**/**placement*** negatives (N12–N15), B′'s *"a store organi**s**ed well enough to be safe…"* thesis sentence. A careless find-replace against §0.14's vocabulary (*placement organizer*, *compositional family*) would damage correct text.
- **F4 — V3 §3.4/§5 rest partly on "synthetic two-timescale data".** Synthetic ≠ demoted: I found **no** registry entry marking this cell "never a claim venue" (the registry's synthetic-instrument strikes I located are the `recency` family, L1742–1749, and C2W10's lifecycle stream, L3144 — neither is V3's). ⚠ My registry read was **targeted, not exhaustive** (`claim venue` / `smoke/regression` probes on `negative_results.md` only), so this is stated at the altitude it was verified.

---

## 8. How I verified (commands + observed output)

Per-file only; no directory-level Grep. Representative results, all reproduced above in situ:

- `Glob .claude/papers/**/draft*.md` → 9 files; **no `bprime/draft-r6.md`** (confirmed again by `Glob .claude/papers/bprime/*`).
- Concept-set Grep, `v1-short/draft.md` → hits L180, 184, 465, 480, 494, 506, 514 → all read in context → toy-EBM / designed-well / coset-diffusion.
- Concept-set Grep, `v2-short/draft.md` → hits L273, 409 → both read → friction-field composition; AR(1)-coset toy.
- Concept-set Grep, `v3-short/draft.md` → 20+ hits → abstract/intro/§3.1/§5/App C/App H read → architectural composition throughout.
- Concept-set Grep, `v5-short/draft.md` → hits L71, 73, 161, 163 → read → AR(1)-coset toy + an explicit "not a toy" disclaimer.
- Concept-set + §A20.2 + §A14.8 probes, `bprime/draft-r5.md` → compositional-family vocabulary **0**; `substrate` **1** (L492, an MLP/Hopfield scope sentence); `gym` **0**.
- Authority reads: `claims_matrix.md` §0.14 (L455–487, whole block incl. never-quotes (a)–(p) and the closing demotion sentence at L485); `advisor-head-c2-charter.md` Add.16 §A44 (L628–648) and §A14.8 item 8 (L253); `advisor-head-shorts-charter.md` §4.2 + Add.1 §A20.2 line (L175) and **Addendum 8** (L430–449, Q8 = this spoke).
- Flag-provenance note: **no experiment was run and no number was measured by me**, so no flag-provenance table is owed (protocol §5 applies to quantitative *results*); every number quoted above is transcribed verbatim from the draft or authority file cited beside it, with file + line.

## 9. Open questions / risks

1. **The null is a null about *lexical + contextual* reachability.** I read every hit and the full negatives/unmeasured appendices of all five drafts, plus V1 §3–§5, V2 §4–§5 + App A/F/J, V3 §1/§5/App C/App H, V5 §1–§3.1, B′ §Appendix P + rival tables. I did **not** read all ~5 000 lines of B′ r5 end-to-end. Residual risk of an *uncoined* compositional passage in an unread B′ region is small (the family's vocabulary is absent from the file entirely, on 12 independent probes) but non-zero, and I state it rather than claim exhaustiveness.
2. **r6 lands after me.** If the parallel `bprime-r6-evidence-fold` introduces any C2W11-era material, this sweep does not cover it. Per Add.8 the fold's inputs are N276 / N294 / N295 / N296 — none compositional — so a re-sweep should be cheap, but it is **not** discharged by this report.
3. **F1 is a genuine two-ruling conflict** (§A14.8 vs §A45) that I deliberately left unresolved.

---

## Proposed handover updates (for the Hub)

1. **Q8 (Add.8 §4) is DISCHARGED with a null result:** the toy-retirement sweep found **0 passages** across V1 v0.4, V2 v0.6, V3 v0.6, V5 v0.1 and B′ r5 that lean on the §A44.1-demoted toy compositional substrate. **No draft rewrite is owed on the demotion**, and the one-by-one pass is unblocked on that axis. Positive controls fired on 5/5 files.
2. **Record the reason, not just the result:** the shorts pre-date C2 entirely (last revisions 2026-07-19 / w16) and B′ r5's C2W5 fold was citation-layer only (2026-08-02) — so the demoted family never entered them. Any *future* fold that imports C2W5/C2W11 material into a short re-opens this question; suggest the Hub attach that trigger to the next draft-fold task rather than re-running a sweep.
3. **Two flags need an owner** (I am not one): **F1** — B′ r5's harness sits under §A14.8's gym demotion while §A45 banks B′'s evidence base as quotable (conflicting authorities, unresolved by me, Head/Advisor call); **F2** — V1's MQAR pillars vs criterion-4's six confirmations (venue admissibility, not §A44.1).
4. **Curator transfer-doc consequences: NONE are owed by this spoke, and I wrote none.** Nothing was measured, so `negative_results.md` gains no entry (a documentation null is not a program negative); `future_work.md` gains no entry (no new untested regime was surfaced — this sweep closed a *documentation* question, not a scientific boundary); `philosophy-synthesis.md` and `HEP_primers.md` are untouched. If the Hub wants the null on the permanent record, the natural home is a one-line note in the wave's ledger addendum citing this report, and I recommend the Hub write it rather than the curator self-filing a result about the curator's own sweep.
5. **`draft-r6.md` did not exist at run time** — the Advisor's landing-verification checklist for the r6 fold should not assume this sweep covered it.

## Flags

- **F1** — §A14.8 (gyms demoted as claim venues) vs §A45 (B′ evidence base banked/quotable): same artifact, opposite pull. ⛔ Flagged, not resolved.
- **F2** — V1's MQAR-based §4.1/§4.3 pillars vs criterion-4's six confirmations (matrix §0.14 / registry L3145). Venue-admissibility question, out of §A44.1 scope.
- **F3** — lexical false friends (V3 *composition*, V2 *placement*, B′ *organised*) — do not find-replace against §0.14 vocabulary.
- **F4** — V3's "synthetic two-timescale data" is not, so far as I could verify, registry-marked "never a claim venue"; my registry check was targeted, not exhaustive.
- **F5 — protocol non-conformance in my own task file:** `.claude/tasks/shorts-toy-retirement-sweep.md` opens **without a DIAL DECLARATION block**, which protocol §7 makes binding on every task file from w25 on. I supplied the declaration myself at the head of this report ("none: instrument/recon"). Flagged for the Hub/Advisor as a scoping-hygiene item, not as an objection to the task.
- **F6** — B′ r5 not read end-to-end (see §9.1); the null there rests on 12 independent per-file probes + targeted section reads, and I state that scope rather than claim exhaustiveness.
