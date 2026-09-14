# PREREG AMENDMENT 1 — absolute dominance by the tripwire is a DROP
**Ruled by the Head, 2026-09-07.** Amends `PREREG.md` §4 clause (b). Frozen; no spoke may edit it.
Repo HEAD at ruling: `7fcef50` (unchanged since the PREREG freeze).

## Why this exists
Clause (b) as drafted asks only whether the matched-byte exemplar store **closes the gap** between the
longest rung and the 512 rung. Wave 1 produced a case the drafted clause cannot score: on C-MAPSS
FD001 the store does not close a gap — it sits **above** HEPA at the 512 rung itself (0.9777 vs a
reported 0.918). Under the literal drafted rule there is no gap to close, so the clause never fires and
the dataset passes a test it has plainly failed.

## The ruling
**If the tripwire matches or beats HEPA at ANY rung, the dataset is DROP — regardless of the shape of
the context curve.** A dataset on which a same-byte lookup equals the model has not earned a
long-horizon section, however cleanly its curve rises.

## Operational form (the verdict spoke applies exactly this)
Clause (b) is now the conjunction of **(b1)** and **(b2)**; failing either is a DROP.

- **(b1) — gap-closing, as originally drafted.** The exemplar store at the same byte budget does not
  close the longest-rung-minus-512 gap to within **0.02** h-AUROC.
- **(b2) — absolute dominance, new.** At **every** rung, `HEPA_mean − tripwire_mean > 0.02`.
  If at any rung `tripwire_mean >= HEPA_mean − 0.02`, clause (b2) fails and the dataset **DROPs**.

The 0.02 tolerance is not new — it is the same tolerance clause (b1) already carried, reused so the two
sub-clauses cannot disagree about what counts as a tie. **A tie fails.** The store matching the model
is the same finding as the store beating it.

## Which HEPA number this is scored against
**The reproduction, never the README.** `HEPA_mean` means HEPA as shipped, run under our harness, on
our cut points, at seeds 42/123/456 — the run commissioned as the first job of `hepa-lh-sweep`. Wave
1's FD001 comparison used HEPA's `REPRODUCIBILITY.md` value (0.918±0.008) because no reproduction
existed yet; **that comparison is provisional and is superseded the moment the reproduction lands.**
If the reproduction disagrees with the README, the reproduction wins and every clause-(b2) verdict is
recomputed against it.

## Standing of this amendment
It is a **post-hoc change to a pre-registration and is logged as one.** It is admissible because it can
only ever make DROP easier and KEEP harder — it adds a way for our own case to fail and removes none.
Any future amendment that loosens a threshold, or that is proposed after a KEEP verdict is known, does
not have that property and must be refused.
