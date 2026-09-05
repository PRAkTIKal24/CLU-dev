# doc-curator-w24-chapter-8 — doc-curator report

Task + acceptance criterion: **Open Chapter 8 of the ledger** ("Control over memory in time: the memory with dials") on the Head-ruled, Hub-specified skeleton; gather w19–w23 `Ch-8 candidate` material with dated pointers and nothing deleted; write the ⟲ w24 reframe addendum; update scorecard/verdict rows.
Status: **done** (one scoping caveat: w24's own experimental outputs do not exist yet — the addendum is explicitly labelled **part 1 of 2**, and part 2 is owed at the wave's review).

**⚠ Reconciliation list for the Hub (first-10-lines rule):** seven items in §"What I think the spec got wrong" below need a Hub decision or a downstream edit — most importantly (a) the `best-of-five` ambiguity, (b) the 2-D-only admission bound, (c) the `2^d` rate-vs-value phrasing, (d) the un-scoped `control validated` tag.

## Files touched
- **`.claude/outputs/philosophy-synthesis.md`** — the only file edited. 1568 → ~1760 lines.
- **Not touched (deliberate):** `HEP_primers.md`, `research_roadmap.md`, `future_work.md` (owned this wave by `doc-curator-w24-plainlang` — the ownership split was preserved); `negative_results.md` (no new negatives exist in w24; every negative Chapter 8 cites is already registered and Hub-tiered: N61, N70, N74, N75, N81, N82, N89–N94); `handover_context.md`, task files, all tracked code. **No git footprint** — `.claude/**` is gitignored, no branch, no commits.

## What I did — every edit, in file order (for diff-review)

| # | location | edit | type |
|---|---|---|---|
| 1 | after the "How the threads map to the papers" table | appended a **dated w24 row block** adding chapter 8 (vehicle: ICLR long, NMI forks Phase 3; primer home Part XI/§§11.9–11.13). Original seven rows untouched | addition |
| 2 | Ch. 1 §7 end | `⟲ w24 pointer` — write modes (a/b/c) + Prop C-N → Ch. 8 §2 (lifetimes dial); *"Ch. 1 remains the home of the physics; Ch. 8 the home of the dial"* | pointer |
| 3 | Ch. 2 §7 end | `⟲ w24 pointer` — Prop F1, fiber channel table, `B_total = K_spatial·B_fiber`, parameter ceiling → Ch. 8 §2; Ch. 2 verdict explicitly unchanged (Prop F1 = its 5th confirmation) | pointer |
| 4 | Ch. 3 §7 end | `⟲ w24 pointer` — vault theorem blocks friction-as-eraser; the *realized* forgetting verb (leaky wells / self-eviction / permanent untouched) → Ch. 8 §5.2; protection verdict stays | pointer |
| 5 | Ch. 4 §7 end | `⟲ w24 pointer` — **DUAL-FILE declaration**: mechanism/certificate/energy-shell semantics stay in Ch. 4; the *dial claim* (compute-adaptive reads) is stated in Ch. 8 §5.2 | pointer |
| 6 | Ch. 5 §7 end | `⟲ w24 pointer` — designed `4·2^d`, repaired packing bound (N70), three regimes, `K_learned = min(2^d,~32)` → Ch. 8 §2/§5.3; lattice thread + the new w24 sharding question stay | pointer |
| 7 | Ch. 7 §7 end | `⟲ w24 cross-reference` — nothing migrated; Ch. 7 still governs on epistemics wording; Ch. 8 §8 carries a chapter-scoped subset, duplication intentional | pointer |
| 8 | between Ch. 7 and `# Scorecard` | ⭐ **NEW `# Chapter 8 — Control over memory in time: the memory with dials`** — full 7-part house rubric + §7 Paper vehicle + §8 Chapter epistemics | **new chapter** |
| 9 | after the Scorecard table | appended a **dated row-8 block** (current-state row; addenda retain per-wave history). Rows 1–7 untouched | addition |
| 10 | end of file | ⭐ **`⟲ Wave-24 addendum, part 1 of 2`** — the reframe as a reframe + its cause, the eight Head rulings, the migration table, chapter deltas, scorecard deltas, gap list, and the doc-hygiene / spec-critique section | addendum |
| 11 | last bullet of the w23 addendum | appended a nested `✅ DISCHARGED at w24` line under the "Chapter-8 is four waves overdue" flag (the flag itself left verbatim) | addition |

## Chapter 8 as built (structure, so the Hub can check it against the spec)

1. **Philosophy** — memory = the shape of the landscape, indexed by launch conditions; ⭐ *its value is control, not lookup*; the four-dial table; the one-slide pitch **with the w22 fine print attached** (a frozen surface does not decay on its own).
2. **Theory** — `(V_θ,𝒜)` / `(S,ρ,τ)` + address roles + two read modes; the six propositions (Prop 1–6 + the w20-promoted Prop 7); **Prop F1**; **Prop C-N**; the three capacity regimes; a **laws-attached-to-dials table** (capacity learned + designed, admission, lifetimes, isolation C3, compute) each with a status tag and a report+section citation.
3. **Expected** — Hub w19 formalization incl. the explicit indexing bet; the Head's iterative/gauge-loose amendment (R19-4, still a proposal under test); the w23 phase doctrine and the consequence it predicted.
4. **Experiment** — all nine w19–w23 harnesses named + the r19/r20 adjudication pass.
5. **Measured — negatives first.** §5.1 the three refutations (indexing N61; triple-refuted lookup N81/N89/N90 with the Head's *structural reading* explicitly labelled **not a proved theorem**; the ~32 write ceiling N92 with N75/N91 as companions). §5.2 the four dials, as a table, every number with its source. §5.3 geometry vindicated for d≤5 **with the R2-boundary "never framed as beating anything"** attached.
6. **Verdict** — the Head-ruled `[split — lookup refuted, indexing refuted, control validated]` verbatim, then the two revisions (w19→w22, w23→w24), then a **curator scope note** (designed store · one geometry family · seed exposure · 0 external benchmarks) that does **not** alter the ruled verdict.
7. **Paper vehicle** — ICLR long / NMI at Phase 3 / appendix contents, plus a **status-honesty flag** that the CL sweep and unlearning demo are targets, not cargo.
8. **Chapter epistemics** — the laundering control (C17-3 → N89, now mandatory), the mechanism controls (kick/ensemble flatness is *why* the retry claim survives where the leaderboard claim died), the under-trained-instrument rule (N94), and the w23 prereg-scorecard catch of a Hub overstatement — with the precise mechanism of the catch (the ◐ mark in `phi-read-in`'s own scorecard; curator read it against the summary; Hub confirmed and made the qualifier binding).

**Nomenclature/discipline checks:** Def-2 respected (inertial `M` vs spectral `μ` distinguished wherever both appear; `μ²≡0`, `μ²(δ)`, `M⁻¹∇V` all correct-sense); two-layer tags used (`[proven]`/`[verified]`/`[evidenced]` on theory rows, program-level tag on the verdict); every quantitative claim carries report + section; **no number was re-derived** — all copied from the reports/registry, except one arithmetic *check* I did myself and labelled as such (that `R = 0.808·√K` inverts `N_pack` to `≈K`).

## What I deliberately left in place
- **The w19–w23 addenda are untouched**, including every `Ch-8 candidate` heading. They are the diff record; Chapter 8 is the canonical home, not the only copy.
- **Chapters 1–7 keep all their text and all their verdicts.** No chapter lost a sentence; nothing was struck. No verdict moved this wave (nothing was measured this wave).
- **The Head's verdict string was recorded verbatim** rather than "corrected" to the ledger's scope convention — flagged instead (item 4 below).
- **Ch. 4 was not emptied of the retry.** The dual-file split is the one place I chose a structure the spec did not specify; the spec lists compute-adaptive reads as a Ch. 8 dial but the mechanism, the certificate and the calibration/allocation pillars are genuinely Chapter 4's thread. Both pointers say which chapter governs which kind of claim. **If the Hub disagrees, the fix is one blockquote.**
- **`negative_results.md` untouched** — Chapter 8's negatives are all already registered and tiered; nothing new was tried this wave. **No new negatives are owed until w24's outputs land.**

## What I think the Hub's structural spec got wrong (the requested section)

1. **`best-of-five` is five METHODS, not five seeds** (`controller-mvp` §2), and the run is 5-seed — the ambiguity is live in the spec *and* in the Hub's §10 entry. Disambiguated inline in Ch. 8 §5.2. **Recommend the same fix wherever the Hub's entry is quoted downstream.**
2. **The admission bound `N_pack = πR²/((√3/2)·d_safe²)` is a 2-D hexagonal-packing count on a disk**, measured on a **dim = 3** store with disk proposals; `R = 0.808·√K` is exactly its inverse. **There is no measured d-dimensional admission bound**, and neither arm attains the ideal (5.2/6.12 fixed; 42.8/64 sized). Presenting it as "the packing/admission law" of the store over-generalizes. Scope line added in Ch. 8 §2.
3. **"capacity doubles per dimension (`2^d`) — exactly the designed rate"** conflates rate with value: designed is `4·2^d`, learned `2^d`, a constant 4× prefactor gap for d≤5. Precision note added in Ch. 8 §2. **This is the sentence most likely to become a wrong figure caption.**
4. **`control validated` carries no scope tag**, while Ch. 1 and Ch. 3 use `[validated-with-scope]` for *better-replicated* results. I did not alter the ruled verdict; I added a scope note. **Hub call whether the row should read `control validated-with-scope`.**
5. **The stated ICLR main text lists the CL sweep and the unlearning demo as contents** — both are R4/R1 *targets*, not results. Flagged in Ch. 8 §7; otherwise the chapter would violate Positioning Charter **C-4** (no promissory notes in lead position) in its own paper-vehicle section.
6. **The direction doc's "3–13 pp" NN-ceiling gap is wrong (3 of 8 cells)** — the Hub already corrected this in §10; Chapter 8 and the addendum use the corrected **3.5–17.6 pp mask / to 42 pp noise** and record the correction as a ruling so the direction doc can never be quoted un-corrected. Same for the CIFAR-scoping of the Hopfield-in-φ margin.
7. **Dial 2's evidence is a hand-driven schedule, not autonomous physics.** The decay law is exact, but it is the controller editing well depth; "forget = let it fill in" implies a passive process the shipped system does not perform. Recorded in §1 and §5.2. **R1 ("certified per-item lifetimes") is the target most exposed by this**, and `unlearning-recon` may sharpen it into a real problem.

## Open questions / risks
- **The w24 part-2 addendum is owed** at the wave's review. All seven w24 tasks land in Chapter 8 (write ceiling → §5.1(3) and §5.3; headroom retry → dial 4; multi-seed → the §6 scope note; lattice theory → §2 capacity; unlearning recon → §7 R1; φ-stream → §5.1(2)'s successor experiments). **Chapter 8's numbers will move; the chapter is written so they can be updated by addendum without rewriting it.**
- **Seed exposure is Chapter 8's weakest flank and is stated as such** — the compute dial is seed 0, the d=4/d=5 walls rest on 2-seed adequacy re-checks of a write the report calls seed-fragile. If `multi-seed-w23` moves the laundering verdict or the kick/ensemble flatness, **§5.1(2) and §5.2's compute row are the two places to edit.**
- **R19-4 ("start arbitrary and restructure") is still unsettled**, and the w24 reframe does not settle it — recorded in the addendum's gap list so it is not silently absorbed into the new doctrine.

## Proposed handover updates (for the Hub)
- **§10 / doc state:** ledger now runs …w22 · w23 · **w24 part 1**; **Chapter 8 is OPEN** (the four-wave structural debt is discharged); the scorecard has **eight** rows (row 8 appended, dated, originals untouched); Chapters 1–5 and 7 carry dated `⟲ w24` pointer blocks; **nothing deleted anywhere.**
- **Owed, new:** the **w24 part-2 ledger addendum** once the seven w24 outputs are reviewed (curator task).
- **Decisions requested (all in the critique section):** the `control validated` scope tag (item 4); whether the Ch. 4 / Ch. 8 dual-file split of the retry is the structure the Hub wants (it is the one unspecified structural choice I made); and whether the "best-of-five" disambiguation and the 2-D scoping of `N_pack` should be propagated into `claims_matrix.md` / the §10 entry (both are outside this task's file ownership).
- **Carried unchanged:** 0 external benchmarks won · multi-seed before any paper number · laundering control mandatory on every performance claim · N94 in force · voraus re-run · real Mamba before any SSM claim.
