# Task: doc-curator-w24-chapter-8 — OPEN CHAPTER 8 of the ledger (the structural decision is made) (w24)

- **Agent:** `doc-curator` · **Output:** `.claude/outputs/doc-curator-w24-chapter-8.md` · **Branch:** none (gitignored `.claude/`)
- **You own `philosophy-synthesis.md` exclusively in this wave.** The parallel task `doc-curator-w24-plainlang.md` owns `HEP_primers.md` / `research_roadmap.md` / `future_work.md` and is **forbidden** from touching the ledger. Do not edit its files either.
- **Read first:** `.claude/outputs/head-advisor-w23-direction.md` (**the reframe — Parts 1, 3, 4**) · `philosophy-synthesis.md` (your own Chapter-8 candidate note and every `Ch-8 candidate` filing from w19 onward) · `.claude/negative_results.md` **N61–N94** · `.claude/claims_matrix.md` **v2.2**

## Why this task exists
You correctly declined to open Chapter 8 four waves running: *"the ledger's chapters are Hub-authored and a new chapter is a structural decision, not a curator one — and the thesis has no named paper to attach to."* **Both blockers are now cleared.** The Head has ruled, and the Head's direction doc supplies both the surviving thesis and the paper vehicle. **The structural decision below is made; you are executing it, not deciding it.**

Five waves of the program's central results (w19–w23) are filed as "Ch-8 candidate" and scattered across Chapters 1–5, which destroys the fact that they are **one result**. This is the largest structural debt in the document.

## ⭐ THE STRUCTURAL DECISION (Head-ruled, Hub-specified — build to this)

**Open `Chapter 8 — Control over memory in time: the memory with dials.`**

The original candidate was titled *"the addressable-memory thesis"* with verdict `[split — physics validated, indexing refuted, controller unbuilt]`. **That title is retired**: it names a claim that was refuted, and the controller is now built. **The refuted indexing claim is recorded INSIDE the chapter as its opening negative — it is part of the story, not the name of the story.**

Populate the standard skeleton:

- **Philosophy.** Memory is not a buffer beside the network; it is the **shape of the network's own energy landscape**, indexed by launch conditions. ⭐ **Its value is not lookup — it is control.** Four dials: **admission · lifetimes · isolation under sequential writes · compute-adaptive reads.** One slide: *write = dig a valley · read = drop a ball · forget = let it fill in.*
- **Theory.** The formal object `(V_θ, 𝒜)`; the six propositions and three capacity regimes; **Prop F1** (mass selects, never stores); **Prop C-N** (permanence is a *designed* flat coset, not a training outcome). The laws now attached to the dials: capacity `K_learned(d) = min(2^d, K_ceiling≈32)` · the packing/admission bound `N_pack = πR²/((√3/2)·d_safe²)` with the sizing rule `R = 0.808·√K` · per-item decay `A·exp(−λt)` · the monotone, **self-limiting** accuracy-vs-compute curve.
- **Expected.** The Hub's formalization + the Head's iterative/gauge-loose amendment; the original addressable-memory/indexing bet; the phase doctrine "learn around a designed core."
- **Experiment.** The w19–w23 harnesses: `primitive-harness` · `sequential-write-interference` · `designed-mechanism-learned-content` · `hopfield-capacity-benchmark` · `gated-write-performance` · `phi-read-in` · `retry-compute-study` · `controller-mvp` · `dimension-aware-budget` (+ the `r19-r20-reconciliations` adjudications).
- **Measured — lead with the negatives, per the Head's framing.**
  1. ⛔ **Indexing/addressing fails** — address-gradient search is dead (N61).
  2. ⛔ **Static lookup is TRIPLE-refuted** — w22 raw pixels, w23 φ-space (**the laundering control fires on all four cells, max CLU margin 0.000 — N89**), w23 retry (the NN floor dominates every cell — N90). Structurally: **settling approximates the nearest-neighbour lookup, and an approximation does not beat its target.** ⚠ Record that last sentence as the Head's *structural reading*, not a proved theorem.
  3. ⛔ **The learned write caps at ~32**, d-independently (N92) — the ceiling belongs to the **write operator**, not the terrain, the parameters, or the dimension.
  4. ⭐ **All four dials measured working:** admission (per-admitted **1.000** flat to K=64, best-of-five; the gate demonstrably fires) · lifetimes (exact `exp(−leak·t)`; permanent + leaky wells coexisting; self-eviction) · isolation (masked writes bit-local, **8474×/3434×**) · compute (a monotone self-limiting curve whose **mechanism survives every control** — kick and ensemble dead flat, N90's positive).
  5. ⭐ **Geometry vindicated where the write can reach:** capacity doubles per dimension (`2^d`) for d≤5 — exactly the designed rate.
- **Verdict.** `[split — lookup refuted, indexing refuted, control validated]`
- **Paper vehicle (the second original blocker, now cleared).** ICLR long = the primitive in the hilly-landscape language + three laws (capacity, lifetime, compute-scaling) + the CL sweep + the unlearning demo. **NMI forks at Phase 3** (same results, laws foregrounded, per the filing rule). Appendix = the NN-ceiling proximity result, the negatives registry material, the RUD-C spec.

## Item 1 — gather, do not delete
Migrate the w19–w23 `Ch-8 candidate` material into Chapter 8 as its **canonical home**. ⛔ **Standing rule applies: never delete prior findings.** Leave a **dated pointer** in each originating chapter (`→ migrated to Ch. 8, w24`) rather than removing text, and strike-with-reason anything genuinely superseded. A reader of Chapters 1–5 must still be able to follow the thread.

## Item 2 — the ⟲ w24 addendum: record the reframe AS a reframe
The doctrine shift from **"learn around a designed core"** (w23 phase doctrine) to **"control over memory in time"** (w24) is a genuine change of thesis and must be recorded as one — **with its cause: the φ laundering result.** Note precisely what survived and what changed: the phase doctrine survives as the **architecture** rule (fixed physics + designed store, learned interfaces); the new thesis is the **claim** rule (every claim is about one of four dials). Also record the Head's **[RULING] masked recall → permanently appendix-only** and the binding scope caveat (*equalling a simple baseline is our best case there, because we approximate the method that wins it*).

## Item 3 — the epistemics the chapter should carry
This chapter is the program's best exhibit for its own method. Fold in, as chapter-level epistemics: **the laundering control** (a pre-registered trivial-store swap that killed the program's central bet on schedule — C17-3 → N89, now mandatory on every performance claim) · **the mechanism controls** (kick/ensemble flatness is what makes the retry claim survive where the leaderboard claim died) · **the under-trained-instrument rule** (N94) · and the w23 event where **a Hub overstatement was caught by a spoke's own pre-registration scorecard** before it reached a draft.

## Acceptance
Chapter 8 opened, populated on the standard skeleton, negatives-first, with the paper vehicle named; the w19–w23 candidate material gathered with dated pointers left behind and nothing deleted; the ⟲ w24 addendum recording the reframe and its cause; the scorecard/verdict rows updated. Report what you migrated, what you left in place, and **anything you think the Hub's structural spec got wrong** — the w23 pass caught a Hub error and that was its most valuable output.
