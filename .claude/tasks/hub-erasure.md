# hub/erasure — TARGET: deleted content is GONE, not gated

**⚠ Supersedes `hub-refuse-delete.md` (kept at `hub-refuse-delete.superseded.md`, ⛔ do not read it).
Re-scoped by the Head, 2026-09-19, on the finding in §1. ⛔ Read §1 before anything else.**

**Boot:** `.claude/autonomous-hub-charter.md` → `.claude/tasks/w3-hub-common.md` → this file. Nothing else.
**Worktree/branch:** `hub/erasure`, off **`base/iter3` = `afd37cf`**.
**Log:** `.claude/outputs/hub-erasure/LOG.md`
**⛔⛔ Scored budget: ZERO. You spend no scored run.** The `deletion` target keeps all 3, banked.

## DIAL DECLARATION
**Dial: whether deletion ERASES or merely GATES.** Laundering control beside every number: the
landscape-removed control must collapse, and every claim carries the byte-matched **gate** baseline of
§2. **Falsified if** a gate bolted onto the prototype table is behaviourally indistinguishable from the
CLU's erasure under an adversary that holds the weights.

---

## 1. ⛔⛔ WHY YOUR TARGET CHANGED — the measurement, not an opinion

**Refusal is dead as a byte-matched claim on this benchmark, and it is dead structurally.**

An **8-byte** value-presence bitmap — one bit per (factor, grid value), 57 bits — plus a free
"is this value bracketed by stored values" check scores, measured on the shipped 512-record query sets:

| row | required | the 8-byte rule |
|---|---|---|
| 1 seen | accept | **1.0000** ✓ |
| 2 unseen-valid | accept | **1.0000** ✓ |
| 5 held-out value | refuse | **1.0000** ✓ |
| 6 invalid request | refuse | **1.0000** ✓ (with a free duplicate check) |

Against a **524,288-byte** budget. ⭐ **And it is not fixable by re-drawing the rows:** the request
space *is* the factor grid — **57 values total** — so any predicate over it tabulates in 8 bytes. Even
the **full joint table over all 480,000 combinations is 59 KB, 11 % of the budget.**
⛔ **Do not attempt to design a refusal row that beats this. There isn't one.**
⚠ Row 7 goes the same way: the bitmap updates on delete, so its change-in-refusal is 1.0, not 0.0.

⇒ **You no longer pursue a refusal win.** Rows 5, 6 and 7's refusal behaviour are **reported as a
negative**, with the 8-byte rule printed beside them, and that negative is a deliverable — it closes a
question that would otherwise consume another wave.

## 2. ⭐⭐ WHAT SURVIVES, AND IT IS YOUR TARGET

**A gate and an erasure are behaviourally identical and physically different. Measure the difference.**

An 8-byte bitmap, a deletion log, a refusal list — all of them **still hold the content** and decline
to serve it. The CLU's store was **refitted with no row carrying the value**, so there is nothing left
to serve. ⭐ **That distinction is what the machine-unlearning literature is actually about**, and no
cheap baseline can fake it, because gating is what those baselines *are*.

**Your target: show that the deleted content is unrecoverable from the CLU's store, and recoverable
from a byte-matched gated table, under the same adversary.**

⭐ **The adversary is the experiment. Build it before you build anything else** — it is the only thing
here that can fail. It holds the weights and the request interface and tries to elicit the deleted
value: direct requests, requests naming the value indirectly, the deleted conjunction's parts, sweeps
over the read's coefficients, and whatever else you devise. ⭐ **Declare the adversary's budget and
move-set in your PREREG before you run it** — an adversary invented after seeing the result proves
nothing.

**The two baselines, both mandatory on every number:**
1. ⭐ **`basis_head` + a gate** — the wave-2 fitted basis + linear head, with a deletion log bolted on,
   at matched bytes. **This is the like-for-like.** It refuses correctly and still holds the content.
2. `knn_proto` + a gate, same construction.

## 3. The second claim — rebuild churn, and it is NOT gate-winnable
A gate has no behaviour to churn, so this one is clean. Wave 2 measured: rebuilding after removing an
arbitrary 8,234 of 32,768 rows changed **18.8 % ± 0.8 %** of the store's answers against **1.40 % ±
0.05 %** for a byte-matched table — a 13.4× ratio — and it did **not** shrink with deletion size over a
9.66× range (852 → 8,234 rows; churn moved 1.11×).

⭐ **Establish it properly, which wave 2 could not:** it kept aggregates only, so the ratio was never
auditable. **Persist per-request predictions.** Use the identical-live-set rebuild (churn exactly
0.0000) as the null, ≥ 3 store seeds, and pre-register before you look.

## 4. The conjunction — pre-register it properly
Wave 2: the CLU forgot `(object_hue = 5 ∧ shape = 2)` while the byte-matched table re-composed it from
parts it still held (2.9 % vs 60.4 % retained), at no differential collateral cost.
⭐ Pre-register **≥ 3 conjunctions**, each with a placebo control, on a **declared non-cylinder cell** —
the wave-2 scored cell is entirely `shape = cylinder` and cannot test a shape conjunction at all.
⚠ **Held weakly to begin with** in wave 2 (0.3672 pre-deletion). Report the pre-deletion level beside
every post-deletion number or the effect is unreadable.
⭐ **This is where the gate baseline is sharpest:** a per-factor bitmap **cannot express** a conjunction
deletion, and a joint table that can also refuses every novel combination — killing rows 3–4. **Measure
both. That gap is the clearest thing you can produce.**

## 5. Known and not a claim
⛔ Condition-1 failure on `scale` and `orientation` is a **registry negative**, not a finding to
re-litigate: both are physical defaults — median `scale` index is the default object size, median
`orientation` index is the null pose — and a generative model cannot delete the value it degrades
*towards* by removing rows. ⛔ Do not pick a physical default as a deletion target. The 4–2 geometry
rule stays a hypothesis on n = 6.

## 6. Fill the scoreboard
All seven rows, both controls, as every hub does. ⭐ Rows 5–7 carry the 8-byte rule's score beside them
and the §1 negative in the row's own note. ⛔ No row-5 or row-6 number is quoted as evidence for the store.

## 7. ⛔ What would prove this hub wrong
- The adversary recovers the deleted content from the CLU's store as readily as from the gated table
  ⇒ **the erasure claim is dead** and the deletion target with it. **That is the result; report it.**
- The gated `basis_head` is indistinguishable from the CLU on every measure you can construct ⇒ same.
- Rebuild churn does not survive persisting per-request predictions ⇒ the wave-2 number was an artefact.
- Deleting a conjunction breaks rows 3–4 for the survivors ⇒ deletion is not surgical and §4 dies.
- ⚠ **Your adversary is weak and you cannot tell.** Build a positive control: an adversary run against a
  store that **never held** the value at all should recover nothing. If it recovers something, your
  adversary is measuring the wrong thing.
