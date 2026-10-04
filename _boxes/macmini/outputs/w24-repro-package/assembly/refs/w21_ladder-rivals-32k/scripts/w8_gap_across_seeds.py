"""Is the matched-bytes object vulnerable to the seed-2 failure?  ⛔ DEV, NOT scored.

The spoke's assertion 1 FAILED on test seed 2 for the K=5 object: 32.9 float32 ulp
against a <= 1 ulp criterion, confined to the rank-deficient well, caused by a tiny
eigenvalue GAP rather than by the downdate arithmetic.  My rank cut pins the same wrong
quantity (eigenvalue MAGNITUDE, ``EIG_REL_TOL``), so the matched-bytes object is a
priori exposed to the identical defect and I have only ever built it on val seed 0.

This measures, per seed and per candidate:
  * the fit-time predictor ``eps / min_kept_relative_gap`` -- no deletion, no refit
  * the OBSERVED downdate-vs-refit deviation, so predictor and observation are checked
    against each other rather than the predictor being trusted

⭐ The hypothesis worth testing, stated before the run: the matched-bytes builds keep
only ``A = 7..16`` directions out of wells of 42..52 channels, i.e. only the TOP of each
spectrum, where gaps are large.  The K=5 object's failure was in a well where the cut
reached down to a relative eigenvalue of 7.5e-12.  If that is the mechanism, the fine
partitions should be far safer -- but "should be" is not a measurement.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chlu.experiments.exp_w8_single_design import (           # noqa: E402
    CovContentStore, spectral_gap_report,
)
from w8_matched_bytes_recut import ward_balanced_labels       # noqa: E402
from w8_single_design_dev import red_mask                     # noqa: E402

D = 2048
ULP32 = 1.1920928955078125e-07
PART_TEST = "C:/chlu/outputs/hub-factor-wells/run/partitions.npz"
CANDIDATES = [(40, 7), (44, 12), (48, 16)]


def log(m=""):
    print(m, flush=True)


def main() -> int:
    z = dict(np.load(PART_TEST, allow_pickle=True))
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    pool = np.asarray(z["pool"], np.int64)
    lab5 = np.asarray(z["factor"], np.int64)
    inc = [int((lab5 == k).sum()) for k in range(int(lab5.max()) + 1)]

    from chlu.core.gen_store import channel_correlation
    from chlu.experiments import exp_gen_factor_wells as fw

    rows = []
    for seed in (0, 1, 2):
        log(f"\n############ SEED {seed} ############")
        F = fw.features_mmap(seed)
        Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
        corr = channel_correlation(Xc)
        del Xc
        X = np.asarray(fw.gather(F, pool), np.float32)
        red = red_mask(pool)

        jobs = [(K, A, False) for K, A in CANDIDATES] + [(5, 64, True)]
        for K, A, is_inc_grain in jobs:
            if is_inc_grain:
                # the incumbent's OWN partition and A -- the configuration that failed
                wc = [np.flatnonzero(lab5 == k).astype(np.int64) for k in range(len(inc))]
                tag = f"factor K=5 A=64 (the config that FAILED on seed 2)"
            else:
                lab = ward_balanced_labels(corr, K, dead)
                wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
                tag = f"ward_balanced K={K} A={A}"
            sizes = [int(c.size) for c in wc]

            st = CovContentStore(X, wc, A, D, allow_saturated=is_inc_grain)
            gap = spectral_gap_report(st)
            t0 = time.time()
            st.delete_rows(X[red])
            td = time.time() - t0
            ref = CovContentStore(X[~red], wc, A, D, allow_saturated=is_inc_grain)
            devs = [float(np.max(np.abs(st.modes[k] - ref.modes[k])))
                    for k in range(len(wc))]
            obs = max(devs)
            pred = gap["worst_predicted_dev"]
            ranks_ok = st.deployed_rank == ref.deployed_rank

            verdict = "PASS" if obs <= ULP32 else "FAIL"
            log(f"  {tag}")
            log(f"    wells {min(sizes)}..{max(sizes)}  ranks {min(st.deployed_rank)}.."
                f"{max(st.deployed_rank)}  min kept rel gap "
                f"{gap['min_kept_rel_gap_over_wells']:.3e}")
            log(f"    predicted {pred:.3e} ({pred / ULP32:8.3f} ulp)   observed "
                f"{obs:.3e} ({obs / ULP32:8.3f} ulp)   ranks agree {ranks_ok}   "
                f"{verdict}")
            rows.append({"seed": seed, "config": tag, "K": K, "A": A,
                         "well_sizes": sizes, "deployed_rank": st.deployed_rank,
                         "min_kept_rel_gap": gap["min_kept_rel_gap_over_wells"],
                         "predicted_dev": pred, "predicted_ulp": pred / ULP32,
                         "observed_dev": obs, "observed_ulp": obs / ULP32,
                         "ranks_agree": ranks_ok, "verdict": verdict,
                         "downdate_seconds": td, "per_well_gap": gap["per_well"]})
        del X

    log("\n================ SUMMARY ================")
    log(" seed  config                                    min gap   pred ulp   obs ulp  verdict")
    for r in rows:
        log(f" {r['seed']:4d}  {r['config'][:40]:40s} {r['min_kept_rel_gap']:.2e} "
            f"{r['predicted_ulp']:10.3f} {r['observed_ulp']:9.3f}  {r['verdict']}")
    mb = [r for r in rows if r["K"] != 5]
    log(f"\n matched-bytes candidates: {sum(1 for r in mb if r['verdict'] == 'PASS')}"
        f"/{len(mb)} PASS across 3 seeds")
    log(f" worst observed among them: {max(r['observed_ulp'] for r in mb):.4f} ulp")

    out = Path("C:/chlu/outputs/w8-single-design/dev")
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_across_seeds.json").write_text(json.dumps({
        "NOT_A_SCORED_RUN": True,
        "why": ("the spoke's assertion 1 failed on test seed 2 for the K=5 object at "
                "32.9 float32 ulp, caused by a tiny eigenvalue GAP; my rank cut pins "
                "eigenvalue MAGNITUDE, the same wrong quantity, so the matched-bytes "
                "object is a priori exposed and had only been built on val seed 0"),
        "criterion": "<= 1 float32 ulp at unit scale, the spoke's own",
        "rows": rows,
    }, indent=2), encoding="utf-8")
    log(f"\nwrote {out / 'gap_across_seeds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
