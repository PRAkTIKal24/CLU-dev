"""Fit base/iter4's object under the VAL condition (charter A.2: selection on val only).

⛔ Nothing about the mechanism is chosen here.  This only produces the store + head that
every DEV read -- pinned linear span, table interpolation, lookup, and my valley -- is
then run on, so the ONLY thing that differs between them is the READ.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402

if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "factor_aligned_balanced"
    cond = sys.argv[2] if len(sys.argv) > 2 else "val"
    W.PINS["condition"] = cond
    out = C.OUT / "objects" / cond
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obj = W.build_arm(arm, out, log=print)
    print(f"[fit] {arm} ({cond}) done in {time.time()-t0:.0f}s; "
          f"store {obj['byte_ledger']['deployed_bytes']:,} B + head "
          f"{obj['head_bytes']['head_bytes']:,} B; request_only={obj['request_only']}")
