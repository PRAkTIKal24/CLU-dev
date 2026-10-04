"""Fit the scored-condition objects on CACHE GENERATION 2, one directory per seed."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C
import chlu.experiments.exp_gen_factor_wells as W

if __name__ == "__main__":
    seed = int(sys.argv[1])
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "factor_aligned_balanced", "knn_proto", "basis_head",
        "landscape_removed", "landscape_shuffled"]
    W.PINS["condition"] = "test"
    W.PINS["seed"] = seed
    prov = C.use_cache_generation2((seed,))
    print(f"[fit_test] seed {seed} cache gen 2 asserted: {prov['encoder_proof']}")
    out = C.OUT / "objects" / f"test_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        t0 = time.time()
        o = W.build_arm(arm, out, log=print)
        print(f"[fit_test] seed {seed} {arm}: {time.time()-t0:.0f}s "
              f"store={o['byte_ledger']['deployed_bytes']:,}B")
