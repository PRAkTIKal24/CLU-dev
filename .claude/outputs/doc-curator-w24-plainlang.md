# doc-curator-w24-plainlang — doc-curator report
Task + acceptance criterion: fold the Head's two plain-language records into `HEP_primers.md` (imagery intact), take `research_roadmap.md` to **v0.7** on the four-dial spine with both corrected wordings, record the **R2 boundary note** where a capacity experiment gets scoped, and run the stale-number hygiene pass.
Status: **done** — items 1–4 discharged. No code touched; **`philosophy-synthesis.md` NOT touched** (owned by the parallel `doc-curator-w24-chapter-8` task); handover and task files untouched; nothing pushed; no git footprint.

> **⚠ FOR THE HUB, FIRST 10 LINES — decisions/checks you own:**
> 1. **Ledger-shaped items I did NOT write** (out of scope by task ⛔, routed to you): (a) the **four-dial spine** and (b) the **R2 exemption** are *claim-rule* changes and belong in the Chapter-8 ⟲ addendum too — please confirm the chapter-8 curator covered them; (c) the **R1–R5 result set** now exists only in `research_roadmap.md` v0.7 + the direction doc — the ledger has no entry for it.
> 2. **`claims_matrix.md` is NOT updated by this pass** and is now the doc that lags: the corrected masked-recall wording (3.5–17.6 pp / 42 pp), the R2 exemption, and the CL filing ruling are all matrix-shaped (CM-23 amendment 2 / a new CM row). **v2.2 remains the live version.** Curator did not edit it — it was not in my task.
> 3. **Provenance confirmed on the corrected wording, one rounding to note:** the mask span **3.5–17.6 pp** and the noise max **42 pp** both reconcile exactly to `retry-compute-study` §2 as transcribed in **N90** (MASK −5.5/−13.3/**−3.5**/**−17.6**; NOISE −7.8/−26.6/−11.3/**−42.2**). *"Widening to 42 pp"* is the **maximum** noise cell (42.2 rounded), not the noise span (which is 7.8–42.2). If a paper sentence needs the span, it must say so.
> 4. **One thing I marked as NOT settled that the direction doc states flatly:** the *"not numerics"* clause of the Head's atoms record. Both primers §11.14 and roadmap v0.7 label it **HYPOTHESIS PENDING `write-ceiling-break`'s scale-invariance ablation** (per your own task text). If the Head intends it as settled, that is a conflict for you to resolve — I did not.
> 5. **Hygiene grep is clean** (details §4): no live "3–13pp", no unqualified "beats Hopfield", no √2/1.44/`d^1.62` exponent, no "98.3% ballistic" anywhere in the doc set except inside explicit ⛔-retirement/forbidden-claim blocks.
> 6. ⚠ **Tooling caveat worth knowing:** directory-level greps under `.claude/` silently returned zero for patterns that do match (verified per-file). **Do not trust a single directory-wide grep as an audit** — I re-ran every hygiene check per-file.

---

## Files edited (all gitignored under `.claude/`)
| file | nature of change |
|---|---|
| `.claude/outputs/HEP_primers.md` | **new §11.14** (the two plain-language records, verbatim, + measured annotations) · **Part-XI header gains a dated wave-24 note** · **§11.6 gains the R2 boundary note** as a dated `Update (wave-24)` block |
| `.claude/research_roadmap.md` | **new authoritative block `v0.7 — CONTROL OVER MEMORY IN TIME`** (~85 lines) inserted above v0.6 · top banner rewritten (v0.6 → superseded-in-part) · v0.6 header re-labelled · **stale-row banner inserted into v0.6's status table** |
| `.claude/future_work.md` | **new section "⭐ Wave-24 fold"** (9 entries) · **4 in-place status updates** marking w24 commissions (`TASKED(...)`) |
| `.claude/outputs/philosophy-synthesis.md` | ⛔ **NOT TOUCHED** (parallel task owns it) |

---

## 1. Item 1 — the plain-language records → primers **§11.14** (⭐)

**New section:** `## 11.14 The one-slide primitive, in the Head's words: atoms, and the two gates **[plain-language records — new wave-24]**`, placed after §11.13 (the last Part-XI subsection) and before the end-of-primer footer.

**Imagery preservation (the point of the item):**
- Both records are reproduced **verbatim inside blockquotes**, attributed *"— Head, `head-advisor-w23-direction.md` §'Plain-language records'"*. **No sentence of the Head's text was reworded, shortened, or technicalized.** The atoms record is one blockquote; the two gates are two blockquotes in one block.
- An explicit instruction sits above them: *"Do not rewrite them into technical prose — the imagery is the deliverable. Everything outside the blockquotes is curator annotation."*
- The one-slide line is set as its own display heading: ***write = dig a valley · read = drop a ball · forget = let it fill in.***
- ⚠ A guard added so the legibility does not leak into claims: *"The imagery describes mechanisms, not wins. Nothing here licenses a performance claim; the masked-recall demotion is unaffected by how well the picture reads."*

**House style honoured (concept → math → CLU connection → status tag → reading pointers):**
- *Concept*: why the section exists (adoption is gated on legibility; the Head's Part-3 ambition quoted).
- *Math*: the atom dictionary `V(q) = Σ_a w_a exp(−‖q−c_a‖²/2σ²)` with the designed-vs-learned write stated as *per-valley closed form* vs *one joint optimization over shared atoms*; the admission rule `min_j d ≥ d_safe = 4.4 s`; confidence = cosine-to-nearest-well after settle, directed boost ×1.5, lock-on-retry.
- *CLU connection*: the four dials, with the sentence **"Every future claim should be a claim about one of those four dials"** and the mapping admission=1 / decay=2 / write locality=3 / confidence gate=4.
- *Status tags* (two, one per record) and *reading pointers* to §11.6, §11.9, §11.11, §11.12, §11.13, the source doc, and roadmap v0.7 — the cross-links the task required.

**The Head's closing enumeration is preserved as a 4-row evidence table** (numbers copied exactly, with source):
| clause | evidence transcribed | tag I gave it |
|---|---|---|
| not the terrain | designed reaches **≥256** (designed line censored at `k_cap` 256) | ✅ settled |
| not parameters | d=6 K=64 **0.855 → 0.809** at **36.9k → 73.7k** params; d=4 K=32 flat **~0.83** across 16× atoms | ✅ settled |
| not dimension | K=64 unwritable at **d=6 AND d=8**, site sep **0.795 vs 0.908** | ✅ settled |
| **not numerics** | — | ⚠ **PENDING MEASUREMENT** (`write-ceiling-break` scale-invariance ablation), explicitly *"argued, not measured — do not quote as settled"* |

Gate annotations, all numbers copied exactly with source: gate cost **0.006 ms** vs relocate **3.1 ms ≈ 4× a 0.70 ms read**; per-admitted **1.000** flat to K=64 (best of five: gru 0.57 · mlp 0.43 · attn 0.34 · learned-CLU 0.16); controller-OFF **1.000 → 0.110 → 0.009 → 0.000**; per-offered **0.081** fixed (packing bound 6.12) vs **0.669** sized (`R = 0.808√K`); ungated retry **0.96 → 0.004 at 9×**; **τ=0.99: 0.961@×1.64** vs no-gate **0.828@×1.81**; saturation **×1.2–1.8**; flagship **0.359@1.0× → 0.961@1.64×**; ranking-vs-acceptance **0.998 vs 0.949**. Sources cited inline (`controller-mvp` 5 seeds; `retry-compute-study` 8 cells seed-0; `dimension-aware-budget`; registries **N90/N91/N92/N93**).

**Part-XI header** gained a dated wave-24 note naming §11.14 as the onboarding entry point and pointing at the reframe + the §11.6 boundary note.

## 2. Item 2 — `research_roadmap.md` → **v0.7** (⭐)

New block `## v0.7 — CONTROL OVER MEMORY IN TIME (authoritative, 2026-07-24, post-w23; Head+Advisor direction is BINDING)`, above v0.6. Contents, in order:

1. **Source-of-authority header** — direction doc Parts 2–4 binding, reconciled with handover §10 (both 2026-07-24 entries); explicit statement of **what v0.7 changes** (spine, capacity law, masked-recall status) vs **what it carries unchanged from v0.6** (the scoring rule, the filing rule, no-retreat, all cautions). *v0.6 is superseded in frontier/spine only, not scrapped.*
2. **⭐ The thesis restated as the organising spine** — the four dials as a table, each with **what it controls** and **the law/evidence it has today** (dial 1 → `d_safe=4.4s`, per-admitted 1.000 / per-offered 0.081↔0.669 · dial 2 → `exp(−leak·t)`, 0.705 = e^−0.35, half-life 1.98, permanent+leaky coexist · dial 3 → the C3 ratio across 4.6 decades, masked-write 8474×/3434×, anchored write 103× · dial 4 → +6.6…+76.2 pp, saturating ×1.2–1.8, controls flat in 8/8, ungated 0.96→0.004). Followed by the binding sentence and the one-slide pitch + ambition (Part 3), pointing at primers §11.14.
3. **Head rulings (6)** — verbatim-scoped:
   - **Ruling 1 (masked recall permanently appendix-only)** carries the **corrected wording as a blockquote**: *"CLU sits within **3.5–17.6 pp** of the NN ceiling on masked recall, **widening to 42 pp under Gaussian noise**, and decisively beats closed-form modern Hopfield at load and noise (**CIFAR-scoped on capacity**)"* — with the ⚠ note that the direction doc's "3–13pp" matches only 3 of 8 cells, and the **MNIST counter-example (0.957 vs 0.871)** plus the surviving noise margin (**0.852 vs 0.250**).
   - **Ruling 2** = the binding scope caveat (*equalling a simple baseline is our best case, because CLU approximates the method that wins it*), with "do not re-litigate" and the mandatory laundering control.
   - **Ruling 3** = **the R2 boundary note** (item 3, below).
   - **Ruling 4** = the CL filing ruling + the success bar, both as directed: *winning replay-free while below replay **is a publishable success**; the replay-handicapped regimes are the **strengthening follow-up, NOT a rescue***.
   - **Ruling 5** = φ-stream (task-1-only primary / generic-frozen as declared upper bound / online later).
   - **Ruling 6** = multi-seed before any paper number + **ambiguity, not destruction** for the retry headroom benchmark.
4. **The pinned laws** — `K_learned(d) = min(2^d, K_ceiling ≈ 32)` with the **write-operator** attribution and the three eliminated rivals; the ⛔ √2/1.44/`d^1.62` prohibition; the **address-space sizing rule**; the **N94** under-trained-diagnostics rule; **CLU does not beat persistence (0.7705 vs 0.5669, L-1)**.
5. **⛔ STRUCK from v0.6** (5 items, the task's "strike anything falsified" mandate): the unpinned exponent / d=8-as-optimizer-failure · **"the true bottleneck is the missing `φ`"** (φ built; laundering fired on all 4 cells, margin 0.000) · "the controller is unbuilt" · v0.6 thread (a) as written (retry dominance retracted, CM-23 split) · v0.6 thread (b) as written (capacity edge is not a lead claim; MQAR structurally unwinnable). v0.6's own struck list restated as still-struck.
6. **⭐ The R1–R5 result set as a table with a `dial` column** — the **four dials ↔ five results mapping** the task asked for, each row carrying *the law it ships with* and *status/what is missing*, plus an explicit **dial-coverage check** (1→R2/R4-sizing · 2→R1/R5 · 3→R4 · 4→R3/R5) and the line *"every result has a dial and a law; nothing on this list is a static-recall claim."*
7. **The pathway** — Phase 2 / 3 (**NMI forks here**) / 4, the assembly split (**ICLR main = primitive + three laws + CL sweep + unlearning demo; appendix = NN-ceiling proximity, negatives registry, RUD-C spec**), the open release with the deliberate *"inefficiencies we leave to you"* section, and the closing discipline paragraph.
8. **📋 Wave-24 task table** (8 rows incl. both curator tasks) with what each gates.
9. **Standing cautions carried** — real Mamba, voraus not re-run, mandatory laundering control, N94, dimension-scaled atom budget, plain-language grounding (+ the retained precision *a frozen surface does not decay on its own*).

**Also:** the top banner now reads *"SUPERSEDED 2026-07-24 by v0.7"* (older banners kept as history, per this file's convention), the v0.6 heading is re-labelled *"SUPERSEDED in frontier + spine by v0.7; its SCORING RULE and filing rule remain authoritative"*, and a ⛔ **stale-row banner** was inserted directly above v0.6's *"Where the program actually stands"* table naming the four superseded rows (exponent · φ-bottleneck · retry signal · candidate threads) and the two that stand (score, cost). **No v0.6/v0.5/v0.4 prose was deleted.**

## 3. Item 3 — the R2 boundary note (recorded twice, on purpose)
Wording used in both places: **the masked-recall demotion does NOT apply to R2's capacity law — R2 is a law about the primitive (how many items are addressable), not a competitive claim against nearest-neighbour, so it is exempt; and its figure must never be framed as beating anything** (Hub-confirmed with the Head).
- **`research_roadmap.md` v0.7, Head ruling 3** — where a wave is scoped.
- **`HEP_primers.md` §11.6, a dated `Update (wave-24)` block immediately after the pinned-law update** — *this is the place an engineer scoping a capacity experiment actually reads*, and it is the only site in the doc set that carries the law itself. It adds the two standing denominators (**N77** capacity-per-parameter is a comparison we lose; exponential capacity is the Hopfield field's own headline ⇒ our novelty is the measured constant and mechanism) and closes with the scoping instruction: *"scope a capacity task as 'how far does the addressable-item law extend, and what breaks it', never as 'CLU beats X at recall.'"*
- Cross-referenced from §11.14's status/reading pointers and from the Part-XI header note.

## 4. Item 4 — pass hygiene

**w24 commissions marked in `future_work.md` (4 in-place status updates, nothing deleted):**
| entry | new status |
|---|---|
| Breaking the write ceiling (w23 fold) | **`TASKED(write-ceiling-break)`** + the two additions beyond the entry's original text (**scale-invariance ablation**; **crowding-aware objective**, because write-loss hits 0 while retrieval fails) |
| φ-space retry extension | **partially absorbed into `TASKED(headroom-retry-benchmark)`** ⚠ *the explicit φ-space composition is not named in that task file, so it stays `OPEN` unless the engineer folds it in* |
| The continual-learning entry | **target RATIFIED, DEFERRED to w25**, blocker commissioned as `TASKED(phi-stream-discipline)`; Head's filing ruling attached |
| Multi-seed error bars on the w23 frontier | **`TASKED(multi-seed-w23)`** with the two tier-A priority results named |

**New section "⭐ Wave-24 fold" (9 entries, schema: what-not-shown · why-it-matters · where-it-lands · status):** R1 *"certified"* against the unlearning literature (`TASKED(unlearning-recon)`, scoping only — with both named risks: what "certified" formally requires, and *deletion-by-construction may already be owned by a kNN datastore*) · R2 at d=16+ (⛔ the exemption + the never-framed-as-beating clause travels with it; designed line censored at 256) · the **lattice sharding question** (`TASKED(lattice-capacity-theory)`; capacity ×N **iff writes are local**, the Head's optimizer-sync concern dissolving, plus the unshown read-cost sub-question) · R3's headroom regime (`TASKED`, ambiguity-not-destruction binding; CL-crowding composition still `OPEN`) · **R4's four replay-handicapped regimes** incl. **privacy-constrained CL where replay is illegal by rule** (`OPEN`, Phase 4, with the "not a rescue" ruling attached) · **R5 `CLULayer`** (`OPEN`, Phase 4, flagged riskiest, w22 sequence-slot was *levelling not winning*) · the **cost-of-strictness curve** as a scientific result in its own right (`TASKED(phi-stream-discipline)`) · **legibility/reimplementability as an adoption hypothesis** + the *"inefficiencies we leave to you"* release-design boundary (`OPEN`) · **differentiability/training-speed debt restated** (`PARKED` under the Head's ruling — *state it, don't fund it*), sharpened by the observation that three of the four dials are hand-coded policies on a designed store.

**Stale-number grep (per-file, doc set = primers · roadmap · future_work · negatives · claims matrix · handover):**
| pattern | result |
|---|---|
| `3–13pp` / `3-13` / "within 3…13" | **0 occurrences anywhere.** The only masked-recall range in the doc set is the corrected **3.5–17.6 pp / 42 pp** (roadmap v0.7). ✅ |
| unqualified "beats Hopfield" | **0 live occurrences.** Every instance is either scoped (**CIFAR-scoped on capacity**, with the MNIST 0.957-vs-0.871 counter-example adjacent) or sits inside a ⛔ forbidden-claim block (primers §11.7 w23 update; CM-23). ✅ |
| `√2` / `1.44` / `d^1.62` as an exponent | **0 live occurrences.** 6 hits, all inside ⛔ "never quote" prohibitions (primers §11.6/§11.7 updates, future_work N92 line, negatives N92, handover don't-quote list). ✅ |
| `98.3%` ballistic | **0 live occurrences.** All hits are retirement/Update blocks (N63 Update, R20-1 resolution, future_work `dt` entry's ✅ RESOLVED line, primers §11.7/§11.8 updates, handover). Replacement figures quoted where relevant: **≈77–80% legacy `dt`, ≈50% shipped `dt=0.125`.** ✅ |
| (extra check) `0.4545` persistence "win" · `0.0000` single-basin collapse | retired-only; quotable number is **0.7705 vs 0.5669 (L-1)**. ✅ |

## What I deliberately did NOT do
- ⛔ **Did not open, read-for-edit, or touch `philosophy-synthesis.md`** — including the ⟲ w24 addendum. File-ownership split preserved exactly as the handover requires.
- **Did not edit `claims_matrix.md`** (not in my task; see Hub flag 2 — it is now the lagging doc).
- **Did not edit `negative_results.md`** — no new negative was produced by this pass; N89–N94 tiers are Hub-confirmed and I quoted them as-is (N92 tier A, N86 tier A FINAL / quote-unblocked).
- **Did not delete or rewrite any v0.6/v0.5/v0.4/v0.2 roadmap prose** — superseded material is banner-marked and struck-listed, per this file's own convention and the ⟲ spirit.
- **Did not reword a single sentence of the Head's two records**, and did not "improve" the parking-attendant or clay imagery.
- **Did not resolve** the *"not numerics"* status question (flag 4) or promote the R1 "mostly framing" assumption to a finding — `unlearning-recon` is testing it.
- **Did not invent numbers**: every figure in the three edited docs is transcribed with report+section or registry citation. Where the direction doc and the Hub disagreed (the "3–13pp"), I used the Hub-confirmed correction **and recorded the disagreement inline** rather than silently replacing it.

## Gaps / risks flagged for the Hub
1. **The doc set now has an asymmetry:** the *claim rule* (four dials) lives in the roadmap and primers, but `claims_matrix.md` — the doc every draft is checked against — still encodes the pre-reframe wordings. **A matrix pass (v2.3) is owed**, minimally: CM-23 amendment 2 (the 3.5–17.6/42 pp wording), a row for the R2 exemption, and the CL filing ruling. Until then a paper-writer checking the matrix will not see the reframe.
2. **`future_work.md` now contains `TASKED(...)` entries for tasks that are in flight, not delivered.** If any w24 task is re-scoped or dropped, four status tags go stale. Cheap to fix at the w24 review; flagging so it is not forgotten (the register's rule is provenance, never deletion).
3. **The RUD-C spec and the CL entry design are still carried only by report files** (`retry-compute-study` §6, `continual-learning-recon` Items 1–3) — this was gap 4 of my w23 report and it is unchanged. v0.7 now *names* both as assembly-appendix material, which raises the cost of them living outside a transfer doc.
4. **Seed exposure is still the weakest flank** and it is now load-bearing on the reframe: three of the four dials' headline numbers are seed-0 (`retry-compute-study`) or 2-seed at the frontier cells (`dimension-aware-budget`); only dial 1's controller numbers are 5-seed. v0.7 states the multi-seed ruling but the dial table quotes the single-seed digits (with sources). `multi-seed-w23` is the fix.
5. **Nothing in my scope disagreed with the handover** — the only tension I found (direction doc "3–13pp" and unqualified Hopfield claim vs the Hub's corrections) was already adjudicated by the Hub, and I applied the Hub's version with the disagreement recorded, not resolved silently.

## Proposed handover updates (for the Hub)
- §10 w24 entry, doc-state line: **`research_roadmap.md` is now v0.7** (four-dial spine; R1–R5 with dial mapping; both corrected wordings; v0.6 superseded in frontier/spine only, scoring rule retained) · **`HEP_primers.md` gains §11.14** (Head's plain-language records, verbatim) **and the R2 boundary note in §11.6** · **`future_work.md` gains the wave-24 fold** (9 entries) and four `TASKED` marks. **`philosophy-synthesis.md` untouched by this pass** (chapter-8 task owns it).
- Add to the standing don't-quote list, if not already there: *"3–13 pp"* as the masked-recall gap (**use 3.5–17.6 pp mask / 42 pp noise max**).
- Consider commissioning **`claims_matrix.md` → v2.3** at the w24 review (flag 1).
