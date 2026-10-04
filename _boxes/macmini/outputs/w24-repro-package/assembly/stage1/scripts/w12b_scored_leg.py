"""w12b hub/nr6-and-kstar -- THE SCORED LEG (NR6).  GATED.

VERBATIM from `scripts/w12_scored_leg.py` (w12 hub/single-design-crossover, folded at
`e58ab05`) with FIVE changes, each one declared here rather than left to a diff:

  1. The ledger gate looks for THIS hub's row, not the w12 hub's.  Another hub's row
     does not authorise this one -- that is the original module's own rule, applied to
     itself.
  2. Everything is WRITTEN under `/c/chlu/outputs/w12b-nr6-and-kstar/`.  The w12 hub's
     directory is READ (for K* and the frozen selection rule) and never written.
  3. `--K` is an ordinary argument, so run 2/3 can buy the K = 5 TEST number the
     crossover hub declared a NOT-RUN.  K* is still READ, never chosen here.
  4. `chlu.__file__` is asserted inside THIS worktree before any number.
  5. ``--refit-tokens``: `transformer` and `lstm` are re-fitted FROM SCRATCH in this
     process, at the same 131,072-parameter pin and on the same fit pool, and scored
     as two EXTRA arms beside the loaded comparators.  Wave 12 measured that
     `w4_learners.train` cast token ids to float32 and that `eqx.nn.Embedding` then
     refused to index, so the two token families could not be fitted at all before the
     fix folded at `1635e0a` (in this base).  Every number this leg prints for those
     two families therefore POST-DATES the fix, and the refit says so by construction
     rather than by assertion.

ORIGINAL DOCSTRING FOLLOWS.

--- w12 hub/single-design-crossover -- THE SCORED LEG.  GATED.

Rows 3 and 4 on the TEST condition at K*, with the charter-C control gate, the learned
baselines, the table FLOOR and the two ablations, all on one object.

GATES, both mechanical and both refused rather than warned about:
  1. `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` must exist.
  2. A ledger row must already be in `/c/chlu/SCORED-LEDGER.md` for THIS hub and THIS
     run number.  Another hub's row does not authorise this one.

K* is read from `CROSSOVER-val.json` and is NOT chosen here.  The rule that produced it
was frozen at 2026-09-21T11:10:24Z in `SELECTION-RULE.md`.

The metric is the programme's own `distinct_valid_novel_tuples` per corner, the unit is
the corner, and the verdict is `fw.paired_estimand` -- never a single cell.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from w12_crossover_lib import log
def score_corners_full(obj, corners, grid, judge, bitmap, tag):
    """One arm, every corner, keeping accept/valid/tuples so the programme's own
    `fw._per_corner` can produce `distinct_valid_novel_tuples`.

    The loop is `exp_gen_factor_wells.run_arm`'s loop, inlined so this hub's object can
    be driven without registering an arm name in the frozen module.
    """
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    done, t0 = {}, time.time()
    for c in corners:
        vals = fw._request_values_from_rows(grid, c["rows"])
        feats, oor = R.arm_features(obj, vals, grid)
        acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
        tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
        val = fw.judge_valid(vals, tup, grid) & acc
        done[c["corner_id"]] = {"accept": acc, "valid": val,
                                "tuples": np.where(acc[:, None], tup, -1),
                                "n": int(vals.shape[0]), "family": c["family"]}
    log(f"    {tag:38s} {len(done)} corners in {time.time() - t0:.0f}s")
    return done


