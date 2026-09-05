# c2-closeout-debts — the C2 closing ledger (⛔ OWNER: the **C2W11 Hub**, NOT a C3W1 spoke)

> ## ⛔⛔ READ THIS BEFORE SPAWNING ANYTHING AGAINST THIS FILE
>
> **This is a RELAY CHECKLIST, not a C3W1 task file.** The Head ruled on 2026-08-12 (later in the
> same session that opened C3) that **the C2W11 Hub scopes and executes ALL of C2's closing tasks and
> final toy checks**. Both live governing documents say so explicitly:
>
> - **C3 charter §6.3:** *"C2 close-out debts — ⛔ NOT C3W1's … the C2W11 Hub scopes and executes ALL
>   of C2's closing tasks and final toy checks (relay handed to the Head; list in C2-charter Add.16
>   §A45 as amended). C3W1 only consumes two of its outputs."*
> - **C2 charter Add.16 §A45:** *"C2 close-out debts — ⭐ AMENDED (Head ruling 2026-08-12, later): ALL
>   scoped and executed by the C2W11 Hub, not C3W1."*
> - **`advisor-handover-c3.md` §2** carries the same, under *"Owed by others (tracked, not mine)"*.
>
> **What this file therefore is:** the C2W11 Hub's debts, itemized in one place so the relay is
> concrete rather than a pointer into a 172 KB archived ledger — plus, in §B, the exactly-two items
> **C3W1 consumes**. ⛔ **Do not spawn a C3W1 spoke against §A without an explicit Head override**
> (see the numbered decision the Hub handed up with the C3W1 spawn block). If the Head *does*
> override, §A is spawnable as written — one engineer + one curator, and the ownership notes below
> are already disjoint from `c3-csf3-harness`'s.
>
> Source of truth: `.claude/advisor-head-c2-charter.md` **§A43.4, §A43.5, §A43.6, Add.16 §A45**.
> ⛔ **The C2 ledger is append-only-frozen** — corrections to C2-era claims go in the **C3** charter
> with a dated pointer back, **never edited in place**.

---

## §A. The debts (owner: C2W11 Hub)

| # | debt | source | owner-type | mechanical done-check |
|---|---|---|---|---|
| A1 | ⭐ **THE C2W11 HUB'S FIRST ACT** (Advisor 2026-08-12) — **the pilot-ttt merge**, `pilot-ttt-nan-and-d5-wiring` → `main`, Hub-reviewed, **incl. the §A20.4 provenance-guard loosening ratified as such**. ⚠ **The shared checkout SITS ON the pilot branch** (verified 2026-08-12) — **merge from a worktree, or move the checkout first**; do not merge into a checkout that is standing on the source branch. Zero-conflict vs `c8314a8` by merge-tree, and ratified — this should cost hours of Hub attention, not days | §A43.5 | Hub + engineer | ⭐ **the agreed cross-campaign done-signal:** `git cat-file -e main:tests/test_ttt_stability_and_d5_wiring.py` succeeds (that path was verified ABSENT from `main @ c8314a8`, so its presence in `main`'s tree *is* the merge) **and** `git log --oneline main..agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` is empty |
| A2 | **Ruling: the `ttt_normalized_write` default** — built, measured, currently gated OFF; the decision is **recorded for the run-3 config** | §A43.5 | Hub ruling | the decision is written down somewhere the run-3 config author will read, with its measured basis |
| A3 | **Ruling: the `.eqx` precondition check on CSF3** before any D5 re-resume | §A43.5 | Hub ruling → engineer | ⚠ `c3-csf3-harness` §3.2 builds the check; the **ruling** is still C2W11's to make |
| A4 | **Ruling: the toy-bit-identity scope cut** — *"the toy bit-identity gate protects the scale run"* is **measured false for the TTT arm**; the sentence's scope must be cut wherever it lives | §A43.5 | Hub ruling → curator | the sentence no longer appears unscoped in any live doc |
| A5 | **The post-merge confirming suite** — **1 726 expected pre-pilot-merge** (1 681 + 22 + 23); ⭐ **re-derive by `--collect-only` AFTER the pilot merge**, do not carry 1 726 across it | §A43.4, §A45 | engineer | suite green on the merged HEAD, count reported, **HEAD compared before and after the run** (a green against a stale base is not a green) |
| A6 | **The spoke-B §2 contradiction** — its report and file say *"the COVERAGE half did not fire"*; §1/§1b record it **FIRED and PERSISTING** (0.7546 address-space) | §A42.11, §A43.4 | engineer/curator | the contradiction is resolved in the artifact **and** the report, one way, with the correction dated |
| A7 | **The V2-negative error-bar footnote** — B's ±0.0397/±0.0209/±0.0125 do not reproduce from `stage_v2.json`'s own aggregates (±0.0679/±0.0141/±0.0069); **every mean matches, no verdict moves** — cosmetic, but on the record | §A42.12, §A43.4 | curator | footnote filed beside the numbers |
| A8 | **The C2W11 curator fold** — §A43.6's never-quotes + re-labelings into `claims_matrix.md` / `negative_results.md`: ⛔ any "K5 verdict" · ⛔ the banked **0.2719** ceiling in a C2W11 context (recomputed **0.3449 ± 0.0175**) · ⛔ "the organizer swap failed" / "physics lost the swap" (**"unaskable at `d_addr = 4`"** is the only quotable form) · ⛔ V2a **0.7254** as a calibration *win* · ⛔ "a flat anytime curve ⇒ the store carries nothing" (the disjunction rule) · ⛔ "the store is inert" at this substrate · ⛔ N5's SEEN 0.4766 and N4's `set_code` numbers without the noiseless-key flag · `PREREG-C2W11.md` §5's permuted-payload negative **re-specified for depth-keyed channels** · `PREREG-TierII.md` §3.5's F5 discussion gains **both halves** | §A43.6 | doc-curator | each entry present in the registries, dated |
| A9 | **The K6 slip owner — 4th occurrence.** Name an owner, not just a note | §A45 | Hub | an owner is named on the record |
| A10 | ⭐⭐ **The `d_addr` ceiling probe — RUN FIRST OR IN PARALLEL WITH A1, ⛔ NOT behind the A1 → A5 chain** (Advisor ruling 2026-08-12). §A43.2's first cell: a `scipy` linear-assignment call on the label-free co-occurrence matrix, **no store, minutes, laptop**. It is independent of the merge and costs nothing; C3 consumes it, so it must not sit behind C2 hygiene. Ownership stays C2W11's (it is a toy-side check) | §A43.2, §A45 | engineer | **`.claude/outputs/c2w11/DADDR-CEILING-PROBE.json`** exists |
| A11 | **The toy substrate's demotion note** — smoke/regression-instrument designation written **into the toy's own docs**, where the next reader will hit it | §A45 | doc-curator | the note is in the toy's docs, not only in a charter |
| A12 | **`clu-dev` push after the merges.** ⛔ **`origin` is FROZEN at `40c2f31` — never push it** | §A45, C3 §5 | Hub | `clu-dev` advanced; `origin` unchanged |
| A13 | **The C2W11 §10 close entry** — filed **after** A5, per the standing HEAD-stability rule | §A43.4, §A45 | Hub | the entry exists and quotes the re-derived suite count |

**⛔ NOT Hub tasks:** C2 charter **§A26 items 1/3/4/5/8** remain **Head-ratification housekeeping on
the Advisor's queue** (`advisor-handover-c3.md` §2). Do not sweep them in here.

**Sequencing — ⭐ AMENDED AND RATIFIED (Advisor, 2026-08-12).** **A1 is the C2W11 Hub's first act**,
and **A10 runs first or in parallel with it — never behind the chain.** **A1 → A5 → A13** is then a
hard chain (merge, then re-derive the suite on the merged HEAD by `--collect-only`, then close).
A2/A3/A4 are rulings that ride the A1 review. A6/A7/A8/A11 are curator/engineer work parallel to the
chain. ⚠ **Two campaigns are keying on A1 and A10** (§B) — the C3 harness spoke is held on A1's
done-signal, so A1's latency is C3W1's latency. **Nothing else in §A is on anyone's critical path.**

---

## §B. What C3W1 consumes — exactly two items, and how it verifies them

1. **A1, the pilot-ttt merge — a MECHANICAL precondition, verified not promised.**
   `c3-csf3-harness` blocks on `git cat-file -e main:tests/test_ttt_stability_and_d5_wiring.py`.
   ⚠ **Status 2026-08-12: NOT MET.** `main @ c8314a8`; that path is `fatal: … not in 'main'`; the four
   pilot commits (`2469ba5`, `1ed0902`, `171972d`, `7fcef50`) are unmerged and the shared checkout is
   still on the pilot branch. **The harness spoke must not be spawned until this prints
   `PRECONDITION-MET`** — a gated spoke whose precondition is only a promise spawns in parallel and
   its pre-cell findings go stale.
2. **A10, the `d_addr` ceiling-probe numbers — a NON-blocking in-block-store config input**, at
   **`.claude/outputs/c2w11/DADDR-CEILING-PROBE.json`**. `c3-csf3-harness` is instructed to record
   the number and expose it as the config default **if that file is present**, and otherwise to
   expose `d_addr` as a plain flag, change no default, and list the probe as owed. ⛔ It must not
   invent a value and must not wait. **A10 is pulled forward precisely so the present branch is the
   likely one.**

⛔ **Nothing else in §A gates C3W1.** In particular A5/A8/A11/A12/A13 are C2 hygiene: C3W1 neither
waits on them nor performs them.

---

*Filed by the C3W1 Hub, 2026-08-12, as a relay to the C2W11 Hub. ⛔ Not a C3W1 spawn target absent a
Head override.*
