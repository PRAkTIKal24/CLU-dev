"""w12 -- THE DELETION ROWS: three scopes, one object, every cost line printed.

(a) VALUE        delete every fit row carrying RED (object_hue index 0)
(b) CONJUNCTION  delete every fit row carrying RED and CUBE (shape index 0)
(c) ITEM         delete a random 1 % of fit rows

and, beside them, the DISPLAY refusal row: the coupling/hole table with a hole punched
at the deleted conjunction.

TWO EXACT MECHANISMS, ON ONE OBJECT, REPORTED SEPARATELY AND NEVER MERGED:
  (i)  data forgetting  = the covariance/count DOWNDATE
  (ii) refusal          = the request count / coupling table WITH HOLES

A compositional store can still compose "red cube" from "red" and "cube" after the
red-cube rows are gone.  THAT IS COMPOSITION WORKING, NOT DELETION FAILING -- the
refusal row is what the hole is for.  Both are reported, separately.

"EXACT" HERE MEANS STORE-SCOPED EXACTNESS: `downdate(delete rows)` equals the store
fitted from scratch without those rows, at the same partition and the same rank.  The
frozen encoder and the frozen decoder are NOT refit.  That is the only definition these
measurements support and it is the one written on every row.

LANGUAGE, binding: never "no changes after deletion".  Post-deletion adaptation is
LIGHT and DATA-FREE -- re-eigendecompose the ledger, carry the address book, no row of
data read -- against baselines that can only delete by retraining from scratch without
the deleted rows.

On `--condition val` every number is a DEV number and is marked `SMOKE_NOT_A_CLAIM`.
On `--condition test` the gate and the ledger row are enforced by `w12_scored_leg`'s
`assert_gates`, which this script calls.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RED_FACTOR, RED_INDEX = 2, 0        # object_hue = 0.0, pinned at exp_gen_factor_wells.py:1348
SHAPE_FACTOR, CUBE_INDEX = 4, 0     # shape = 0.0 = cube
SPHERE_INDEX = 2                    # shape = 2.0 = sphere
ITEM_FRACTION = 0.01                # the declared 1 %
def factor_indices(pool: np.ndarray, cards) -> np.ndarray:
    """row id -> (n, F) factor indices, odometer order (last factor fastest)."""
    c = np.asarray([int(x) for x in cards], np.int64)
    strides = np.ones(c.size, np.int64)
    for i in range(c.size - 2, -1, -1):
        strides[i] = strides[i + 1] * c[i + 1]
    return (np.asarray(pool, np.int64)[:, None] // strides[None, :]) % c[None, :]


def corner_requests_value(corner, grid, factor: int, index: int) -> bool:
    """Does this corner ask for that factor value anywhere in its request set?"""
    from chlu.experiments import exp_gen_factor_wells as fw
    vals = fw._request_values_from_rows(grid, corner["rows"])
    gv = np.asarray(grid.values[factor], np.float64)
    target = float(gv[index])
    col = np.asarray(vals, np.float64)[:, factor]
    ok = ~np.isnan(col)
    return bool((np.isclose(col[ok], target)).any())


