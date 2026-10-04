# Task: iter2-advisor-review — adjudicate wave 2, and rule on whether its object is the long's object

- **Agent:** Main Advisor (own thread). **Boot your own charter first** (`.claude/tasks/longs-main-advisor.md`), then this file.
- **Output:** `.claude/outputs/iter2-advisor-ruling.md`
- **Scoped by:** the ICLR-long Hub, 2026-09-18, at the Head's direction.

## DIAL DECLARATION
**Dials touched: NONE.** This thread reads, re-derives from raw artefacts, and rules. It runs no
experiment, spends no scored run, edits no pinned file, and writes no registry.

---

## 1. The one job

**Rule on what wave 2 establishes for the ICLR long — and, first, on whether wave 2's object is the
long's object at all.** Those are one job because the second answer prices the first.

⛔ Do not re-plan the programme. ⛔ Do not draft prose. The Head drives the experiment plan.

---

## 2. What landed — the claims you are adjudicating

Four autonomous hubs ran on the ThinkPad against pin `4125fce7`. **2 of 12 scored runs spent. No bar
moved. No pinned file edited.** Everything is **bridge seed 0**.

| hub | verdict | scored |
|---|---|---|
| generation-table-2 | TIE (Δ = −0.0093 vs `knn_proto`, tie band m = 0.020, at 0.956× bytes) | 1/3 |
| composition-2 | WIN on the pre-registered bar (42 vs 12 tuples, McNemar p = 2.27e-7), TIE on the generalising estimate | 1/3 |
| deletion-2 | banked TIE + two unasked-for real results + one charter condition failed | 0/3 |
| admission | NOT SOLVED — vacuous by algebra, exact reason given | 0/3 |

**Re-derive from, in this order** (all under `.claude/outputs/_thinkpad/scratch/reports/`):
`iter2-compiled.md` → `iter2-methods-and-results.md` (the full record; self-contained) →
`refusal-closure-check.md`. Raw artefacts for every number are under `.claude/outputs/_thinkpad/`
(4,919 files, 137 MB), including all nine hub worktrees.

⚠ **Your standing rule 1 applies with force here.** The wave-2 coordinator states in its own §8 that it
verified the *object definitions, hashes and cache checksums* itself but took **every statistical
quantity** — the CIs, the McNemar p, the bootstrap intervals, the variance decomposition, the churn
ratios, the AUROCs — from the hubs' reports without re-deriving them. Those are the numbers above.

### The three results that are larger than the verdicts
1. **The read verb.** Changing only how the store's coefficients are formed — a real-valued linear
   combination over modes instead of picking one mode per well — moves the shipped store from
   `0.5903 / 9 tuples` to `0.7917 / 27 tuples` at a **byte-identical** ledger. Confirmed at 43/43
   held-out corners and on the scored cell, where the effect is **larger** than on dev (+0.2014 vs
   +0.1192). This is a change to a pinned default, not to the object.
2. **Deletion's rebuild cost.** Remove an arbitrary 8,234 of 32,768 rows, rebuild by the ported
   canonical mechanism, and **18.8 % ± 0.8 % of the store's answers change**, against **1.40 % ± 0.05 %**
   for a byte-matched prototype table — a 13.4× ratio. And it **does not shrink with deletion size**:
   over a 9.66× range of rows removed (852 → 8,234) churn moves 1.11×. Neither was pre-registered;
   neither is by construction.
3. **Admission's bandwidth result.** The gate's zero level set sits at `d/σ ≈ 2.0–2.4`; all real data
   sits at `0.79`. `σ` is the median nearest-neighbour distance **between atoms**, and the atoms **are
   train rows** — so the atoms tile their own manifold at spacing ≈ σ by construction, and any real
   point is ≈ 1 σ from one of them whatever the store holds. **The bandwidth rule and the signal it is
   meant to separate are the same number.** 8× the store closes ~0.09 of the gap while pushing the level
   set further out.

### The three disclosures the hubs made against themselves — do not let these get lost
- **Composition's win does not generalise.** Over 105 held-out-shaped corners in two location-disjoint
  families: Δ = +1.09, CI [−1.52, +3.69] — a tie, and **not for lack of power** (CI ±2.6).
- **Composition's level does not replicate.** An independent spoke barred from the hub's code got
  **25, not 42** (the verdict replicated; the number did not), because the pre-registration pinned ridge
  but left the partition, the k-means++ RNG stream and the Lloyd budget free.
- **The corner is 90.9 % of the variance** in the paired Δ (store seed is 9.1 %). **Both** scored results
  in wave 2 are single 72-item cells. Neither can distinguish "won this cell" from "ties in expectation."

---

## 3. ⭐⭐ THE DIVERGENCE — wave 2 did not run the object the long's claim is about

**Verified by the Hub on disk today, from code, not relayed.** State plainly in your ruling whether you
agree, because everything in §2 is priced by it.

### Two unrelated memory objects, and wave 2 used the second
| | the long's line | wave 2's line |
|---|---|---|
| object | `CluStoreCell` (`chlu/core/blocks.py:1124`) — the full physics CLU as a block's memory: learned `V_theta`, two-phase damped relaxation, derived addressing, admission, per-item lifetimes, masked local write, learned trajectory `psi` read-out | `GenStore` (`chlu/core/gen_store.py`, on `main` only) — a factored quantiser over a **cached, frozen** encoder's channels |
| trained? | **end-to-end**, on streaming text, chunk 64, 28.5 M params, cluster, 3 seeds | ⛔ **nothing is trained.** Encoder, decoder and judge are all frozen and hashed; the store is fitted, then read |
| write | shipped masked objective, 4 differentiably-unrolled **sign-SGD** inner steps (the TTT/Titans re-budget of the shipped 300 Adam steps) | not applicable — the store is fitted offline |
| read | settled point / strided trajectory through `psi` | pick modes per well and concatenate; or a linear combination over them |
| metric | **bits per character**, against trained rivals | an external frozen classifier's readings of 72 rendered images |
| rival | Mamba-2, Gated DeltaNet-2 (built), a param-matched GRU, an echo baseline, and the param+state-byte matched arm | `knn_proto`, a per-value prototype **lookup table** at matched bytes |
| scale | full corpora, cluster, 3 seeds | 72-item held-out cell, CPU-only, 4 threads, **seed 0** |

`gen_store.py` states the split deliberately: forcing the physics store onto a channel partition of a
frozen feature cache *"would have meant inventing anchors, i.e. inventing the answer."* What it reuses is
`multiwell_read.soft_occupancy` and the byte accounting from `FactoredStore.n_bytes`. ⛔ **These are not
the same object and the reports do not claim they are.**

### The divergence that matters most
**The long's open problem is integration — the block routes around the memory.** Wave 2 cannot speak to
it, because wave 2 **has no block**: the read is the only path to the decoder, and `assert_no_bypass`
enforces that structurally — it checks by signature that `read_from_codes` does not take the features as
an argument at all, and numerically that perturbing the features while holding the codes leaves the
output bit-identical. ⇒ **wave 2 engineered the long's failure mode out of existence by construction
rather than solving it.** That is legitimate for a phase-1 generative cell. It is not evidence for the
integration claim.

### What DOES transfer, and what does not — checked, not assumed
- ⭐ **The read verb transfers as a question.** Wave 2 measured that restricting the read to a vertex of
  the simplex, instead of the span, costs most of the performance (nearest mode 0.4884 · best convex
  combination 0.5509 · best **linear** combination **0.8264**, with `knn_proto` at 0.7801 for scale).
  **The long's read is a settled point / trajectory. Is it similarly restricted?** That is a question for
  the Head's plan, not a finding.
- ⭐ **Deletion's churn result bears directly on the matched-bytes forgetting claim.** It prices a cost
  the ported *Retention* method does not: that method guarantees byte-identity with a store that never
  held the item, and this measurement shows behavioural stability is a separate and unmet property.
- ⛔ **Admission's bandwidth finding does NOT mechanically transfer.** `nn_median` appears only in
  `chlu/core/gen_landscape.py` — the generative line. The long's `chlu/core/admission.py` is a different
  mechanism: a **write-time** `spacing_gate` (`min_j d(q_new, q_j) >= d_safe`) plus a C3 fixed-point
  drift bound, non-learned, over the physics landscape. **The shape of the failure is still worth
  checking** — a length scale derived from the spacing of the stored items cannot separate new points
  from them — so: **is `d_safe` derived from atom spacing?** ⚠ Flagged as a question. The Hub did not
  measure it and you should not repeat it as a finding.
- ⛔ **`knn_proto` is not a competitive baseline in the long's sense.** It is an untrained lookup table.
  Beating it at matched bytes is a good control; it is not Mamba-2.

⚠ **One thing the Hub could not settle and is not guessing at:** whether the generative cell is a
formal part of the ICLR long, a parallel phase, or a pivot. `main`'s last ~300 commits are all
generative-cell work; the long's standing summary describes only the streaming line; the oldest skeleton
on disk describes a third thing again (lattice/firewall/voraus). **You own this question, not the Hub.**

---

## 4. What you are asked to rule on — numbered, and only these

1. **Is wave 2's object the long's object?** (§3.) If not wholly, say which of §2's results enter the
   long's evidence base, which are phase-1-only, and which are neither.
2. **What may the composition result claim?** A win on the pre-registered bar, a tie on the generalising
   estimate, and a level that does not replicate are all true simultaneously and all are the hub's own
   disclosures.
3. **Do deletion's two unasked-for results get promoted to pre-registered scored claims?** ⚠ The
   conjunction result **cannot be tested on the scored cell at all** — that cell is entirely
   `shape = cylinder` and the deleted conjunction is a shape.
4. **What happens to the admission target?** 3/3 scored runs unspent, a gate vacuous by algebra at the
   shipped launch point, and four concrete mechanism findings that point at the **read**, not the gate.
5. **Does one hub's numbers warrant an independent check before wave 3?** The coordinator names
   deletion-2 — it is the only one of the four with no replication by an agent outside its own code, and
   it is also the hub that had two agents executing it concurrently in one worktree.
6. **The scored-run budget reading is unruled.** The strict reading (three per target, across every
   attempt and every machine) is in force by default. Two machines pursuing one target cannot see each
   other's count. ⭐ **This one is yours and it is cheap to close.**
7. **Seed 0 everywhere** is the largest single exposure in the corpus. Seeds 1–2 need full-scope caches
   from this Mac; they cannot be produced on the ThinkPad without a re-encode, which is forbidden.

---

## 5. ⛔ What would prove this brief wrong

- If `CluStoreCell` and `GenStore` share a fitted component that the Hub missed, §3's table is wrong and
  more of wave 2 transfers than stated. **Check it; do not take the table on trust.**
- If the generative cell **is** the long's declared evidence line, then §3's framing is backwards and
  wave 2 is on-target, not adjacent.
- If composition's 105-corner tie is itself a single-family artefact, disclosure 1 in §2 weakens.
- If deletion's churn measurement is an artefact of the ported rebuild rather than of canonicalisation,
  result 2 in §2 dies. The hub's rung-1 control (identical live set ⇒ churn exactly 0.0000) is the thing
  to attack first.

## 6. Files

- **You own:** `.claude/outputs/iter2-advisor-ruling.md`. Nothing else.
- ⛔ **Read-only, and do not move or delete:** everything under `.claude/outputs/_thinkpad/`.
- ⛔ **Never cite** `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` — superseded broken-split
  file, quarantined. ⚠ It is manifest-pinned, so do not move or delete it either.
- ⛔ No registry writes. Registry corrections and needed experiments are numbered requests via the Head.

## 7. Precondition — mechanical

All four must exist before you start:
- `.claude/outputs/_thinkpad/scratch/reports/iter2-compiled.md`
- `.claude/outputs/_thinkpad/scratch/reports/iter2-methods-and-results.md`
- `.claude/outputs/_thinkpad/scratch/reports/refusal-closure-check.md`
- `chlu/core/blocks.py` containing `class CluStoreCell`

⚠ `chlu/core/gen_store.py` exists **on `main`, not in this checkout** — the primary working tree is 317
commits behind `main`, sitting on the TTT/D5 branch. Read it with `git show main:chlu/core/gen_store.py`.

## 8. Acceptance

A ruling on all seven items in §4, each with the evidence you re-derived and the artefact it came from;
an explicit agree/disagree on §3 with your reason; and any item you decline to rule on named as declined
with what would let you rule. ⛔ Your charter's format rules bind the summary you give the Head, not
this ruling — this one is for the Hub and the Head together, and it should be technical and precise.
