# PILOT-TTT-RULINGS — the three Hub rulings that ride the `pilot-ttt-nan-and-d5-wiring` merge

> ⭐ **THIS IS THE LIVE HOME OF THESE RULINGS.** They were made by the C2W11 Hub on 2026-08-12 and
> filed into `.claude/handover_context.md`, which was **archived the same day** (Campaign 2 closed,
> charter Add.16). This file re-homes them where the **CSF3 run-3 config author** will read them.
> ⛔ **Nothing here is new, re-scored or re-derived** — it is the archived ruling block, restated
> verbatim in substance, with its measured basis attached. Debts **A2 / A3 / A4** of
> `.claude/tasks/c2-closeout-debts.md` are discharged by this file.
>
> **Authority:** charter `advisor-head-c2-charter.md` Add.15 **§A43.5** (Head-ratified 2026-08-12:
> MERGE, Hub-reviewed, incl. the §A20.4 provenance-guard loosening ratified as such) · Add.16 **§A45**.
> **Evidence base:** `.claude/outputs/pilot-ttt-nan-and-d5-wiring.md` (the spoke's own report).
> **Code:** merged at `main @ 5656728` (5 files, +552/−8; `tests/test_ttt_stability_and_d5_wiring.py`,
> 18 tests). ⛔ **The C2 ledger is append-only-frozen — corrections go in the C3 charter, never there.**

---

## Ruling 1 — `ttt_normalized_write`: **(b) THE DEFAULT STAYS OFF; THE TTT ARM IS SUBMITTED WITH THE FLAG SET**

**⇒ What the run-3 config must do:** ship `ttt_normalized_write = False` as the repo default, and set
it **True on the TTT arm's leg only**, with the artifact recording *why* it is set.

**Decisive reason — the anti-hobbling rule.** The fix makes the **rival substantially STRONGER**
(**2.12** against the shipped arm's best-before-NaN **4.75**), and the anti-hobbling rule has already
**inverted one C2W10 verdict**. The rival gets its strongest admissible form or the swap is worthless.

**Claims-safe on the byte match:** zero params, zero state bytes, `cell_ledger()` asserted identical
⇒ ⭐ **the matched-bytes swap ledger does not move.**

**Why not the alternatives:** **(a)** flipping the default would invalidate **every banked journal**
and force a **6-leg rerun**; **(c)** dropping the arm leaves tier iii with **no two-sided control**,
which **§A16.2(iv) makes mandatory**.

**Two conditions, both binding on the run-3 config author:**
1. The artifact **records why the flag is set** — the **`η·n/d ≥ 2` criterion**, measured **3.47 at
   pilot geometry** vs **2.31 at toy geometry**.
2. **Any** TTT change additionally runs the **33 s pilot-memory-geometry rig** — positive-control
   verified: **it NaNs on shipped code**, so it can report a failure.

## Ruling 2 — the `.eqx` precondition: **RULED MECHANICAL, NOT EYEBALL**

Before **any** D5 re-resume on CSF3: assert that **all five `ckpt_{arm}_seed<N>.eqx` are present in
each leg's `$OUT`**, **before submission**, with **`S4.json` backed up first**.
⛔ **A missing checkpoint costs 16 h silently.** An eyeball check is not compliance.

⚠ `.claude/tasks/c3-csf3-harness.md` §3.2 already instructs the harness spoke to *build* this check
("include the `.eqx` precondition check before any re-resume — that is ruling (2)"); **this file is
the ruling's content** — the five-checkpoint assertion and the `S4.json` backup are the spec.

## Ruling 3 — the toy-bit-identity scope cut: **⛔ A NEVER-QUOTE**

⛔ ***"The toy bit-identity gate protects the scale run"*** is **measured FALSE for the TTT arm** and
may not be written in any live document unscoped.

**Why:** the TTT arm's stability criterion is a function of the **solved geometry** (`n` comes from
the CLU cell's byte ledger), which **the toy does not share** — and the toy sits **astride** the
boundary (**2.08–2.31**, ~**40 % of chunks amplifying**), not safely inside it.

✅ **The scoped form, and the only quotable one:** *the gate certifies **bit-identity of the OFF
path** and **nothing** about the numerical stability of any arm whose criterion depends on solved
geometry.*

---

## ⚠ The §A20.4 guard loosening — ships Head-ratified, with one live caveat

`load_journal` now accepts a key the journal **predates** **only** when the current value is
**provably the field default**; strict otherwise. It repairs a defect that had **already stranded
4 × ~16 h of banked CSF3 A100 training** since C2W6 (the fingerprint retro-invalidated *every*
journal whenever `StreamMemoryConfig` gained a field).

⚠⚠ **Its soundness premise is a repo CONVENTION, not an invariant:** that new levers ship OFF and
bit-identical. **If a future `StreamMemoryConfig` field defaults to changed behaviour, a journal
would be wrongly accepted.** The durable fix — an **`as_flag_table()` on that dataclass** — is
**REGISTERED AND NOT DONE**. The harness task's §3.3 regression test (field added → default accepted,
non-default refused) is the guard until it is.

---

*Filed 2026-08-13 by the C2W11 close-out Hub, discharging `c2-closeout-debts.md` A2/A3/A4 into a live
document. ⛔ No ruling was changed, softened or re-derived in the move.*
