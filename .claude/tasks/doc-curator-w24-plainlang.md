# Task: doc-curator-w24-plainlang — the plain-language primitive records (+ roadmap v0.7) (w24)

- **Agent:** `doc-curator` · **Output:** `.claude/outputs/doc-curator-w24-plainlang.md` · **Branch:** none (all targets are gitignored under `.claude/`)
- **Read first:** `.claude/outputs/head-advisor-w23-direction.md` (**the source text — Parts 2, 3 and 4 are binding Head direction**) · `.claude/outputs/doc-curator-w23-sync.md` (your own previous pass) · `.claude/research_roadmap.md` **v0.6** (the doc you are updating) · `.claude/claims_matrix.md` **v2.2 (Hub-confirmed)** · `.claude/negative_results.md` **N89–N94 (tiers Hub-confirmed; N92 re-tiered A; N86 now tier A FINAL and quote-unblocked)**

## Why
The Head's ambition (Part 3) rests on the primitive being **explainable on one slide**: *write = dig a valley · read = drop a ball · forget = let it fill in.* Adoption is the goal, and adoption is gated on legibility. The Head has written two plain-language records in the direction doc and asked that they enter the transfer documents. Separately, the **roadmap is now the only stale doc in the set** — it is still v0.6 and describes the pre-w23 frontier.

## Item 1 — ⭐ fold the plain-language records into `HEP_primers.md` (verbatim where possible)
Two records from `head-advisor-w23-direction.md` §"Plain-language records":
1. **Atoms** — the landscape as terrain sculpted from thousands of small Gaussian bumps; a stored item = a valley; the **designed write** places clay by formula one valley at a time; the **learned write** hands all atoms to one optimization and asks it to dig all K valleys **jointly** — *and that joint dig is what caps at ~32: different valleys fight over shared clay, and the write objective declares success (loss 0) while retrieval already fails.* Preserve the Head's closing enumeration: **not the terrain** (designed digs ≥256), **not parameters**, **not dimension**, and — **pending the `write-ceiling-break` scale-invariance ablation** — **not numerics**. ⚠ Mark that last clause as *pending measurement*, not settled.
2. **The two gates** — the **admission gate** (writing: measure distance to every existing valley, relocate or refuse inside the safety distance ≈4.4× valley width; the parking-attendant image; the check is nearly free, hunting for a free spot in a crowded lot is what costs; **this is why retention-per-admitted is a flat 1.000**) and the **confidence gate** (reading: how close the ball stopped to a known centre; dead-centre = accept, no-man's-land = boost and re-roll; **never re-kick a confidently settled ball** — gateless retrying is measurably worse at more cost; and the gate makes compute **self-limiting**, ~1.6× rather than a fixed 9×).
Place these as a new primers section, cross-linked to the existing §11.9 (write operator), §11.12 (MVC-0 controller) and §11.13 (test-time compute / RUD-C). **Keep the Head's imagery intact** — its legibility is the point; do not rewrite it into technical prose.

## Item 2 — ⭐ `research_roadmap.md` → **v0.7** (the one stale doc)
v0.6 describes the pre-w23 frontier. Update it to carry, at minimum:
- **The thesis, restated (Head, Part 1 close):** the store's value is **not static lookup** — which is ceiling-capped by the operation it approximates — but **control over memory in time: admission · lifetimes · isolation under sequential writes · compute-adaptive reads.** ⭐ **Every future claim should be a claim about one of those four dials.** Make this the roadmap's organising spine.
- **[HEAD RULING] Masked recall is permanently appendix-only.** Its role is competence, not victory: *"CLU sits within 3.5–17.6 pp of the NN ceiling on masked recall, widening to 42 pp under Gaussian noise, and decisively beats closed-form modern Hopfield at load and noise (CIFAR-scoped on capacity)."* ⚠ **Use those corrected ranges** — the direction doc's "3–13pp" matches only 3 of 8 measured cells, and the Hopfield margin is CIFAR-scoped (CM-23 amendment 1). Both corrections are Hub-confirmed.
- **[HEAD RULING, binding scope caveat] Masked/static retrieval is a task where equalling a simple baseline is our best case, because CLU *approximates* the nearest-neighbour method that wins it.** Record this so no future wave re-litigates it.
- **The pinned capacity law** `K_learned(d) = min(2^d, K_ceiling≈32)` with the ceiling attributed to the **write operator**; strike anything in v0.6 that the φ verdict, the pinned law, or the R19/R20 resolutions falsified.
- **The R1–R5 result set and the Phase 2/3/4 pathway** (Part 4) as the forward plan, and the **success bar** (Part 2 point 6): *win the replay-free class across a suite; approach — if not equal/beat — replay at matched memory.* ⚠ Record the **Head's filing ruling**: winning replay-free while sitting below replay **is a publishable success**, with the replay-handicapped regimes as the *strengthening follow-up*, **not** a rescue.
- **The four dials ↔ the five results** mapping, so each R has its dial and its law.

## Item 3 — the w24 boundary note (prevents an over-applied ruling)
Record explicitly: **the masked-recall demotion does NOT apply to R2's capacity law.** R2 is a **law about the primitive** (how many items are addressable), not a competitive claim against nearest-neighbour — so it is exempt, **and its figure must never be framed as beating anything.** (Hub-confirmed with the Head.) Put this where an engineer scoping a capacity experiment will see it.

## Item 4 — standard pass hygiene
`future_work.md`: mark the w24 tasks as commissioned. Confirm no stale "3–13pp", no unqualified "beats Hopfield-in-φ", no √2 exponent, and no "98.3% ballistic" anywhere in the doc set.

## ⛔ NOT in scope — DO NOT TOUCH `philosophy-synthesis.md` IN THIS PASS
**The entire ledger — including the ⟲ addendum for this direction ruling — belongs to the parallel task `doc-curator-w24-chapter-8.md`**, which runs at the same time as you and opens Chapter 8. **These two tasks are split precisely so you do not both edit `philosophy-synthesis.md`.** You own `HEP_primers.md`, `research_roadmap.md` and `future_work.md`; that task owns the ledger. If you believe something ledger-shaped is missing, **write it in your report for the Hub to route** — do not edit the file.

## Acceptance
The primers records (imagery intact), roadmap **v0.7** with the four-dial spine and both corrected wordings, the R2 boundary note, the ledger addendum, and a grep-clean report on the stale numbers. Flag anything you believe the Hub got wrong — the w23 pass caught a Hub overstatement and that was the most valuable thing in it.
