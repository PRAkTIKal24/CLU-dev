# HUB CHARTER — `hub/deletion`

- **Agent:** `autonomous-hub` · **Branch:** `hub/deletion` · **Worktree:** `../CHLU-hub-deletion`
- **Base:** ⭐ **the PIN `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`**
- **Boot:** `.claude/autonomous-hub-charter.md` → `.claude/AGENT_PROTOCOL.md` → `.claude/handover_context.md` §7 → this file. ⛔ Nothing else.
- **Report:** `.claude/outputs/hub-deletion.md` · ⭐ **running log: `.claude/outputs/hub-deletion/LOG.md`, updated as you go**
- **Limits:** ≤ 10 h · ⛔ **≤ 3 scored runs, all reported** · ⭐ CPU default · ⛔ **GPU only for a headline**

---

## ⏸ BANKED (Head, 2026-09-16) — ⛔ DO NOT LAUNCH THIS HUB YET
⭐ **Sequencing ruling: `hub/composition` runs FIRST, then this.** The two are the same question from opposite
directions — ⭐ **composition asks whether unseen combinations can be generated; deletion asks whether one can be
made ungenerable while its neighbours survive.** ⛔ A composition result changes what deletion must prove: the
better composition works, the harder unrecoverability becomes, because the parts remain in the store.
⭐ **The charter below is complete and ready.** ⛔ It waits on `hub/composition` reporting, ⛔ not on more scoping.

## ⭐⭐ YOUR TARGET (RE-CHARTERED by the Head, 2026-09-16)
**Delete a COMBINATION from the store — and show the rest of the space still generates perfectly.**
⭐ **The worked example, in the Head's words:** *"if we decide that red cubes are no longer valid and we delete it,
we can still generate a blue cube or a red cylinder perfectly."*
⛔⛔ **A CONJUNCTION, ⛔ not a factor value.** Deleting *all red* is a row filter any table does. ⭐ **Deleting
`(red ∧ cube)` while `(red ∧ cylinder)` and `(blue ∧ cube)` survive is the compositional claim.**

### ⭐ What you must show — four things, and the THIRD is the hard one
1. **Before:** the target combination **was** generable. ⛔ Without this control, "unrecoverable after" is vacuous.
2. **Overlap survives:** every combination sharing a factor value with the deleted one still generates at full
   quality, read by the external classifier — `(red ∧ cylinder)`, `(blue ∧ cube)`, beyond seed spread.
3. ⭐⭐ **THE DELETED COMBINATION IS UNRECOVERABLE — INCLUDING BY COMPOSITION.** ⛔ Not merely absent from a direct
   read. ⭐ **This is where the target is genuinely hard and ⛔ where it cannot be true by construction.**
4. **Byte-exact provenance:** which atoms went, asserted in code.

### ⛔⛔ THE TENSION THAT MAKES THIS A RESEARCH QUESTION — read it before planning
⭐ **A compositional store's strength is that it composes combinations it never stored.** ⇒ ⛔ **after every
`(red ∧ cube)` atom is gone, the store may STILL compose `red` from `(red ∧ cylinder)` and `cube` from
`(blue ∧ cube)` — and hand you a red cube.**
⇒ ⭐⭐ **Deletion and composition pull against each other, and item 3 is where that fight is decided.**
⛔ If the combination re-composes, **that is a finding and you report it** — *"a compositional memory cannot forget a
conjunction whose parts it still holds"* is a real result, and an honest negative.
⚠ **Note the asymmetry with the rivals:** a lookup table passes item 2 **trivially** — dropping one entry leaves the
others untouched, because it never composed anything. ⇒ ⛔ **item 2 alone is NOT a win.** ⭐ **The claim lives in
item 3 and in item 2 at combinations the rival never stored.**

### ⛔ Rivals get the IDENTICAL deletion
⭐ The predicate is over factor tuples, and **every method stores something row-shaped**, so it applies to all of
them: `knn_proto`'s prototypes, `lace`'s table, the atom-mean lookup. ⛔ **Deleting from our store and not from
theirs is not a comparison.** ⭐ **Atoms are verified-bit-identical train rows and `factor_index.npz` carries every
row's tuple** — so this is a predicate, ⛔ not a new primitive you must invent.

### ⚠ What exists, and what does not
⛔ **`gen_store.delete_store(mode=zero|mean|scramble)` is NOT this.** Its own docstring says it exists to be
measured against — **it is the laundering control.** ⭐ Use it as exactly that: your `store_removed` row.
⚠ `store_lifecycle.py` / `well_lifecycle.py` do scoped deletion on the **streaming** path, which the Head has ruled
out of this pivot. ⭐ **Read them for the pattern; ⛔ the generative path is where you must work.**

## ⭐ §E.4 — THE MECHANISM FACTS YOU INHERIT (measured, ⛔ do not re-derive)
- ⭐⭐ **The launch beats the settled state 0.6852 → 0.2114 — the dynamics are the defect on the generation metric.**
  ⚠ **Your target does not depend on settling being good**, but ⛔ **it does depend on compositions being read
  somehow**, and the read you use must be declared and defensible.
  ⚠ **0.6852 is at 8× bytes; at matched bytes the launch scores 0.5008 and loses to the codebook read.**
- ⭐ **Placement is not the lever** (learned placement nulled at 0.0019 vs a 0.10 bar; the inherited ward grouping
  already beats the random floor by 0.0975). ⭐ **The wells are in the right place** — that helps you: deletion is
  a statement about well contents, not well positions.
- ⭐⭐ **The codebook costs 90.5 % of `scale` and 100 % of `orientation`.** ⚠ **Relevant to you:** if a factor value
  is already largely lost in the codebook, ⛔ **"it is unrecoverable after deletion" is not evidence of deletion.**
  ⭐ **Measure recoverability BEFORE the deletion as your own control.**
- ⚠ **Roughly the bottom third of composition-B aggregate is FREE** — a 2.67-tuple constant output scores 0.3272.
  ⭐ Report distinct tuples beside any aggregate.

## ⛔ Falsifiers
1. **All four conditions above, measured, with rivals given the same deletion.**
2. ⭐⭐ **The pre-deletion control**: the deleted content was **recoverable before** the deletion. ⛔ Without it,
   "unrecoverable after" is vacuous.
3. ⭐ **Other factors preserved** beyond seed spread — ⛔ a deletion that quietly costs `scale` is damage.
4. ⭐ **Byte-exact provenance**, asserted in code.

## ⭐ What would make this hub's answer worthless
- ⛔⛔ **A win that is true by construction.** ⭐ **That belongs in the method section and this programme has ruled
  it never a headline.** If that is what you find, ⭐ **report it in those words** — it is an honest answer.
- ⛔ **No pre-deletion control**, so unrecoverability is unmeasured.
- ⛔ **Rivals not given the same deletion**, so the comparison is against an arm that was never asked to do the task.
- ⛔ **"Absent from a direct read" reported as "unrecoverable from every composition."** They are different claims
  and only the second is the target.
