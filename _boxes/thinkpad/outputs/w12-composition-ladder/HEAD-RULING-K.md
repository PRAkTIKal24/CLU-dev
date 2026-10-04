# HEAD RULING ON K — recorded 2026-09-21T12:38:07Z

**Received from the Head, recorded verbatim in substance and applied immediately.**

> Finish the ladder at K = 5 as declared. Main text carries ONE K only, K\*; the K = 5 ladder
> goes to the appendix labelled as the sweep's endpoint. Never both in main text. The Head will
> re-run the store arm at K\* as a free replication once the crossover hub publishes it — do
> not wait for it.

## What this hub does about it, in four points

1. **The ladder finishes at K = 5, unchanged.** `PREREG.md` §3 already declared K = 5 because
   `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` reads `"K_star": null`,
   `"K_star_reason": "K=5 reference row absent; rule not applicable"`. Nothing about the run
   changes: same partition, same arms, same controls, same estimand, same 3 seeds.

2. ⭐ **PLACEMENT IS NOW FIXED, AND IT IS APPENDIX.** Every artifact this hub ships carries the
   label:

   > **APPENDIX — the K-sweep's ENDPOINT at K = 5.** ⛔ Not main text. The main text carries
   > ONE K, K\*, and never both.

   The label is written into `LADDER.json` (`placement`), printed as the first line of
   `LADDER.md`, set as the figure's own title suffix and repeated in its caption, and stated in
   `FINDINGS.md`. ⛔ A reader cannot pick a number out of this hub without the placement
   travelling with it.

3. ⛔ **THIS HUB DOES NOT WAIT FOR K\*.** The lanes continue exactly as launched. K\* was
   unpublished when this hub started (11:31:33Z) and is unpublished now; if it is published
   before this hub finishes, ⛔ **no row of this hub moves to it** — the ladder would then be a
   K = 5 sweep endpoint reported beside a K\* main-text row produced by someone else, which is
   precisely the "never both in main text" the ruling forbids this hub to create by itself.

4. ⭐ **The K\* replication is the HEAD's, not this hub's.** The Head re-runs the store arm at
   K\* as a free replication once the crossover hub publishes it. What this hub owes that
   re-run is a runnable instrument and a declared object, and both are on the branch:
   `scripts/w12_ladder_leg.py --rung <k> --seed <s>` reproduces any rung, and the single
   design's K is the one place that would move
   (`exp_w8_downdate.W8_PINS["partition"] = "factor (K=5, the incumbent's own inherited
   label-MI wells)"`).

## ⚠ One consequence the Head should have in front of it before the K\* re-run

This hub has already measured, on all three seeds at k = 2, that **`pca_tensor` and the single
design are bit-identical** — per-corner `distinct_valid_novel_tuples` max|diff| = 0 over 60
corners, `n_valid` equal on every corner, at 28.6333 / 31.6500 / 31.3333. The mechanism is
measured, not inferred: the read is a span projection and the two stores span the same per-well
subspace to 1e-8 on the three wide wells (1,929 of 2,048 channels); on the two narrow wells
their ranks differ sharply (15 vs 60, 41 vs 59) and it changes not one of the 2,160 judged
requests.

⇒ **At K = 5 the single design's ~24.4× byte total buys exact deletion, not composition.** If
the K\* re-run is to say anything new about *composition*, the thing that has to move at K\* is
the **span**, not the ledger — and that is a checkable prediction, not a hope: the re-run should
report ‖P_singledesign − P_pca‖ per well at K\*, exactly as this hub reports it at K = 5.
