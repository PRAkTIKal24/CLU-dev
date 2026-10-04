"""Fill ALL SEVEN scoreboard rows with the PINNED runner, on cache generation 2.

⛔ Rows other than 2 are REPORTED, NEVER CLAIMED (w4-hub-common section 3).
This calls `exp_gen_factor_wells.run_arm` verbatim -- the pinned row logic, the pinned
judge, the pinned corners -- with only the feature cache redirected to generation 2.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402

if __name__ == "__main__":
    seed = int(sys.argv[1])
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "factor_aligned_balanced", "knn_proto", "basis_head",
        "landscape_removed", "landscape_shuffled"]
    cap = float(sys.argv[3]) if len(sys.argv) > 3 else 5400.0
    W.PINS["condition"] = "test"
    W.PINS["seed"] = seed
    prov = C.use_cache_generation2((seed,))
    print(f"[rows] seed {seed} cache gen 2 asserted: {prov['encoder_proof']}")
    out = C.OUT / "objects" / f"test_s{seed}"
    for arm in arms:
        t0 = time.time()
        try:
            r = W.run_arm(arm, out, max_seconds=cap, log=print)
            print(f"[rows] seed {seed} {arm}: done in {time.time()-t0:.0f}s -> {r}")
        except Exception as ex:
            print(f"[rows] seed {seed} {arm}: FAILED {type(ex).__name__}: {ex}")
