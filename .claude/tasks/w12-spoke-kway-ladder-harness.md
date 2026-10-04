# w12 spoke/kway-ladder-harness — the k-way composition families, k = 3, 4, 5

- **Agent:** `experiment-engineer`.
- **Output:** `/c/chlu/outputs/w12-kway-ladder-harness/` (LOG.md, FINDINGS.md, **LADDER-READY.json**).
- **Owns:** its output dir + branch `w12/kway-ladder-harness`. ⛔ No other spoke's tree.
- **Scored runs:** ⛔ **NONE.** You build the instrument; `hub/composition-ladder` runs it.

## THE ONE JOB
Extend the value-holdout condition with `kway-*` families for **k = 3, 4, 5**: k factors jointly
pinned to a held-out box (the same widths the n-way builder uses), the other 6−k free.
⭐ **Row 3 is k=2 and row 4 is k=6 — the existing ends.** You are building the middle.

## ACCEPTANCE CRITERION
⭐ **≥ 49 pairwise-disjoint corners per k**, the **same paired-corner estimand and bar as rows 3–4**,
and `LADDER-READY.json` naming each family, its corner count, and its disjointness check.
⛔ **PRE-DECLARE THE BAR BEFORE ANY NUMBER EXISTS**, in a file, dated. Charter A.1.

## ⛔ NON-NEGOTIABLE
- ⭐ **Scale and orientation stay INELIGIBLE for VALUE hold-outs**, but are **eligible inside k-way
  boxes**, exactly as in row 4. ⛔ Do not quietly widen this.
- ⭐ **Build the check that can KILL the ladder before the ladder.** If the corners are not disjoint,
  or a k-family cannot reach 49, the ladder measures nothing — so write that check first and run it.
- ⛔ The k=2 and k=6 ends must reproduce the EXISTING row-3/row-4 corner construction. If your k=2
  disagrees with row 3, **your builder is wrong, not row 3.** Assert it.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- A k for which ≥49 disjoint corners do not exist ⇒ ⭐ **declare that k NOT-RUN with the count you
  reached.** ⛔ Never pad with overlapping corners; the paired estimand assumes disjointness.
- Your k=2 not matching row 3's corners ⇒ stop and report; every ladder point would be off-instrument.
