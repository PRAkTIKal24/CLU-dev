# doc-curator-w23-sync — doc-curator report
Task + acceptance criterion: fold wave-23 (6/6 reviewed, integrated, `main @ 5e466c0`, 612 green) into the transfer documents — apply the three analyst reconciliation blocks verbatim, register the w23 negatives + positives, split CM-23, add the new forbidden claims and the under-trained-fit caveat, ⟲ ledger addendum, primers, `future_work.md`.
Status: **done** — all six mandatory items discharged. No code touched; no handover edit; no task-file edit; nothing pushed.

> **⚠ FOR THE HUB, FIRST 10 LINES — what needs a decision from you:**
> 1. **Confirm or re-tier N89–N94** (curator-proposed: **A** N89 · **A** N90 · **A** N91 · **B** N92 · **B** N93 · **A** N94; rationale in the registry's provenance flags).
> 2. **Finalize N86's tier.** It was registered **A PROVISIONAL** *because* it was quote-blocked by R19-1. The block is lifted and the conclusion upheld ⇒ the natural action is **A (final)** with the L-1 number quotable. Curator did not decide it.
> 3. **`claims_matrix.md` is now labelled v2.2 (CURATOR-PROPOSED, pending Hub confirmation).** The two binding changes you directed are applied (CM-23(b) split; CM-22 new forbidden claims) plus five new CM-23 approved wordings. Confirm the version bump or amend.
> 4. **There is still no Hub §10 wave-23 entry** — the live handover entry lists the wave as in-flight, 0/6 reviewed. The ledger addendum states this explicitly and is **not** a substitute for the §10 entry.
> 5. **One intra-report scope tension flagged, not resolved** (`phi-read-in`: recon block says CLU-in-φ "decisively beats" Hopfield-in-φ; its own prereg scorecard marks that ◐ — *wrong on MNIST-high-M*, 0.957 vs 0.871). All approved wordings I wrote carry the CIFAR/hard-feature-geometry scope. **You adjudicate the emphasis; I did not reinterpret.**

---

## Files edited (all gitignored under `.claude/`)
| file | nature of change |
|---|---|
| `.claude/negative_results.md` | 6 new entries (N89–N94) + index rows + tier-flag note · 3 reconciliations RESOLVED with analyst wording applied **verbatim** · N63 + N86 Update/supersede blocks · 2 new appendix-mining bullets · 4 new provenance flags · header maintenance note |
| `.claude/future_work.md` | 5 entries marked `→ SHOWN (wave-23, …)` · 1 new section (**⭐ Wave-23 fold**, 7 entries incl. the two the Hub named) · 3 in-place corrections where R19-1/R20-1 were cited as open |
| `.claude/outputs/philosophy-synthesis.md` | **⟲ Wave-23 addendum appended** (⟲ protocol respected — no chapter rewritten): cross-cutting theme table, 6 chapter deltas, 6 scorecard rows, gap-list + superseded-numbers list, doc-hygiene |
| `.claude/outputs/HEP_primers.md` | 3 **new subsections** (§11.11 φ + laundering control · §11.12 the MVC-0 controller · §11.13 test-time compute on a memory read) · dated `Update (wave-23)` blocks in §11.6, §11.7, §11.8, §11.9 · Part-XI header note |
| `.claude/claims_matrix.md` | **v2.2 header (curator-proposed)** · CM-23(b) **SPLIT** · CM-22 **+4 forbidden claims (i)–(l)** · CM-23 **+5 approved wordings (g)–(k)** · CM-3 R19-1/R20-1 resolution block · two row-header counts corrected |

---

## 1. The three reconciliation blocks — applied VERBATIM (mandate 1)

| conflict | where the analyst's block now lives | side effects applied |
|---|---|---|
| **R19-1** | `negative_results.md` § "✅ RECONCILIATIONS — wave-19", quoted verbatim in a blockquote, plus the single-knob crossing table and mechanism | **N86 index row rewritten: UPHELD + UNBLOCKED**, quotable number **0.7705 vs 0.5669 (L-1)**; the staged `[N86 = C18-1]` bullet's quote-block struck through with the resolution; **0.4545/0.4568 retired**; CM-3's "N86/R19-1 stays QUOTE-BLOCKED" struck and replaced; `future_work.md` CAFE-section prerequisite updated |
| **R20-1** | `negative_results.md` § "✅ RECONCILIATIONS — waves 20–22", verbatim + the 5-row evidence table (A/D/B/C/E) | **N63 gains a dated Update block retiring 98.3%**; N63's index row amended; CM-3 + CM-22(l) updated; `future_work.md` engineering-debt line updated (was "⛔ do not re-quote 98.3%") |
| **R19-2** | same section, verbatim + per-seed table + shipped-path number | **NOT promoted to an N-entry** (per the analyst's explicit instruction); only *"trained `q*` is low-rank, PR≈1"* kept; the generalization is registered separately as **N94** |

Section headings changed from "⚠ OPEN RECONCILIATIONS (UNRESOLVED)" to "✅ RECONCILIATIONS (RESOLVED at wave-23)". **R19-3 and R19-4 untouched**, as the analyst scoped (R19-1 issue **(b)** — the Hub's withdrawal of the raw-space gate — is explicitly preserved as separate and unaffected).

## 2. New negatives registered (mandate 2) — **tiers are curator-proposed**

| # | one line | proposed tier | source |
|---|---|---|---|
| **N89** | φ laundering fires on all 4 (dataset×arm) cells, **max CLU margin 0.000**, with frozen-PCA **and** separately-trained-AE φ ⇒ the "no embedding" defence is **closed** | **A** | `phi-read-in` §3 |
| **N90** | the feedforward-NN floor (0.99–1.00) **dominates CLU-gated retry in all 8 cells** (−3.5…−42.2 pp) ⇒ CM-23 splits | **A** | `retry-compute-study` §2/§4 |
| **N91** | controller per-offered on **fixed** geometry **0.081 vs gru 0.57** @K=64 (two-part verdict; both metrics mandatory) | **A** | `controller-mvp` §2 |
| **N92** | ⛔ base √2 (1.44) / `d^1.62` is **not** the capacity exponent; the law is `min(2^d, ~32)` and the ceiling is the **WRITE OPERATOR** | **B** | `dimension-aware-budget` §1/§3 |
| **N93** | PREREG refinements: **P7** sized geometry 0.669 (packing 42.8/64, `d_eff<d`), **P9** gate 0.006 ms but relocate 3.1 ms ≈ **4× a read** | **B** | `controller-mvp` §5/§3(d) |
| **N94** | ⭐ standing caveat: **`<40`-epoch CAFE diagnostics are not properties of the shipped model** (98.3% ballistic; 0.0000 collapse) | **A** | `r19-r20-reconciliations` recon 4 |

Every entry carries the schema (tried · verbatim numbers · mechanism · scope · disposition) and — per the Hub's mandate 3 — **the wave's POSITIVES are embedded inside the entries they must travel with**, not filed separately:
- **N90** carries the ⭐ **mechanism attribution** (kick + ensemble **dead flat in all 8 cells**; +6.6…+76.2 pp; auto-stop ×1.2–1.8; ungated collapse 0.96→0.004; τ=0.99 optimum).
- **N92** carries the ⭐ **pinned law** (base-2 = the designed rate for d≤5 — *"the wall is geometry" VINDICATED there*; d-independent write ceiling ~32; d=8 stall resolved 8→32; H-LEARNING rejected; tax stops widening; 4th mass null; 8474×/3434× masked-write locality).
- **N89** carries the ⭐ **phase-doctrine premise** (CIFAR chance 0.012 → 0.81–0.97; CLU-in-φ ≫ Hopfield-in-φ, **0.973 vs 0.008** at CIFAR M=256, **CIFAR-scoped**) and the ⭐ **retry hook surviving φ** (AUROC **0.845–0.988** ⇒ GREEN).
- **N91** carries the ⭐ **per-admitted 1.000 flat to K=64 (best of five)** and the **sized-geometry 0.669 > gru 0.57**, plus the gate-fires evidence (spacing 1.61 ≥ 1.54; intervention 0.20→0.97) and the permanent+leaky decay demo.

## 3. Claims matrix → **v2.2 (curator-proposed)** (mandate 4)
- **CM-23(b) SPLIT** — verbatim approved sentence now in the row: *"CLU draws a rising, auto-stopping accuracy-vs-compute curve that a saturated feedforward memory structurally cannot draw, while remaining below that saturated floor on masked-pixel retrieval."* Any draft asserting CLU "beats feedforward via test-time compute" is marked **non-compliant**.
- **CM-22 gains (i)–(l):** the designed-store-beats-a-trivial-feature-baseline claim (N89) · the √2 exponent (N92) · a single-number controller verdict (N91) · any `<40`-epoch CAFE diagnostic as a shipped-model property, incl. the retired 98.3%, the retired 0.0000 collapse and the retired 0.4545 "win" (N94). Row header corrected **eight → twelve**.
- **CM-23 gains (g)–(k):** retry mechanism attribution · the pinned capacity law + write-operator attribution · the phase-doctrine premise *with its CIFAR scope and the AE-softmax re-collapse caveat* · the controller two-metric rule · the standing methodological caveat. Row header corrected **five → eleven**, and the w23 frame sentence added.

## 4. Standing methodological caveat (mandate 5)
Registered as **N94** (tier A proposed, instrument-validity class alongside N51/N68), cross-written into: the primers §11.7 guardrail 5 and §11.8 update (c); the ledger's Ch-6/Ch-7 delta and its cross-thread note; CM-22(l) + CM-23(k); and N63's Update block. Wording used everywhere: **"diagnostics computed on `<40`-epoch CAFE fits are not properties of the shipped model — every CAFE diagnostic states its epoch count."** Recorded as *joining*, not replacing, the w17 overflow rule.

## 5. Primers + ledger + future_work (mandate 6)
- **Primers:** §11.11 (φ: concept → the fairness condition → both halves of the measurement → the **laundering control as a binding nomenclature item** → status tag → reading pointers), §11.12 (the controller: the four verbs as a table, the two-metric verdict, costs, what it cannot fix), §11.13 (test-time compute: the construction, the curve, the three controls, the NN floor, the mask/noise split, the RUD-C spec and its scoring rule). All four house-style elements present in each. §11.6/§11.9 updates carry the pinned law and the write-ceiling attribution; §11.7 gains five guardrails + the new must-cites (SQHN, iCaRL, GDumb, EBT); §11.8 gains the revised honest position incl. the cross-cutting shape.
- **Ledger:** ⟲ addendum appended, **no chapter rewritten**. Cross-cutting theme recorded explicitly as a 3-row table + the sentence *"three independent tasks, one pattern; this is the program's honest position, not a coincidence."* Superseded numbers named: 98.3% · 0.4545 · 0.0000 collapse · {4,8,8,32,8} → {4,8,16,32,32,32} · base √2 · CM-23 absolute dominance · "no embedding" defence · single-number rematch verdict.
- **`future_work.md`:** the two Hub-named items are in (**write-ceiling-breaking follow-up**, flagged as the wave's highest-value next experiment with the "not geometry / not parameters" evidence and the masked-write lever; **φ-space retry extension, GREEN**), plus five further boundaries the wave surfaced (the CL entry with its binding replay caveat and mandatory GDumb/ER/iCaRL baselines; the address-space sizing rule + farthest-point sampling; the SQHN/EBT cite-and-distinguish debts; the "has any *continuous* energy store been run on Split-MNIST Class-IL" sweep; multi-seed error bars). Five entries marked `→ SHOWN (wave-23, …)` and kept with their provenance (controller · I/O gap φ-half · retry study · dimension-aware budget · φ-before-benchmark).

## What I deliberately did NOT do
- **Did not promote R19-2 to an N-entry** (analyst's explicit instruction; only the low-rank observation kept).
- **Did not resolve the `phi-read-in` internal emphasis tension**, and did not soften either statement — both are recorded with their scopes (item 5 in the header above).
- **Did not re-tier N67–N88** or touch the Hub's discharge notes; **did not edit** `research_roadmap.md`, the handover, or any task file.
- **Did not update two historical/provenance blocks** that still read "R19-1…R19-4 may not be quoted as settled" (`claims_matrix.md` line ~89, the v2.0 discharge note; `negative_results.md`'s struck w19 candidate block). They are struck-through provenance by design — **flagged here in case the Hub wants an inline pointer added**.
- **Did not invent tiers for the positives** — they live inside the negatives they must travel with, per the file's schema, and in CM-23.

## Gaps / risks flagged for the Hub
1. **Seed exposure is the wave's weakest flank:** φ and retry are **seed 0 only**; the budget-adequacy re-checks pinning **d=4=16 / d=5≥32** are **2-seed** on a write the report itself calls seed-fragile at the 0.9 rung. The controller (5 seeds) is the exception. Registered as a provenance flag; **N92's thinnest numbers are named.**
2. **The Chapter-8 decision is now four waves overdue** — all four w23 headline results filed as "Ch-8 candidate" again. This is now the largest structural debt in the ledger.
3. **`research_roadmap.md` (v0.6) was not in my scope** and still describes the pre-w23 frontier (φ as the named bottleneck; the exponent unpinned; the controller unbuilt). It "governs" while the matrix "mirrors" — **a roadmap pass is owed to keep that relationship true.**
4. **The RUD-C spec and the CL entry design are both paper-ready artifacts sitting only in report files** (`retry-compute-study` §6; `continual-learning-recon` Items 1–3). They are pointed at from `future_work.md` and the primers, but no transfer doc *carries* them — a paper-writer would need the reports. Flagging in case the Hub wants them lifted.
5. **Numbers I could not cross-check against a Hub §10 entry** (there is none for w23): every w23 figure in the four docs is transcribed from the reports, with report+section citations throughout, per Def-2 nomenclature. If the Hub's review notes differ from any report digit, the docs follow the **report**.
