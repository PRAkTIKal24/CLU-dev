"""w12 hub/single-design-crossover -- SECTION 1, THE VAL SWEEP.  FREE, no scored run.

What it answers
---------------
Where does composition break as the partition gets finer?  Label-free correlation
partition (ward, balanced) at K in {5, 8, 10, 12, 16, 20, 24, 28, 32}, rank 64 per well
capped STRICTLY below well width, read by the programme's own product address book, and
scored on the existing 24-corner val read.

Per K it reports, on every row:
  * val composition (mean `valid` over the 24 val corners)
  * rank / well width                       -- the saturation tell-tale
  * the FOUR-PART ledger, summed, never netted
  * exactness: downdate(delete red) == refit-without-red
  * the charter-C controls: landscape_removed / landscape_shuffled / random_atoms

SELECTION IS NOT DONE HERE.  `SELECTION-RULE.md` was written and timestamped at
2026-09-21T11:10:24Z, before this script produced a number.

The val condition is the free selection condition (charter A.2).  Every number this
script writes is marked `SMOKE_NOT_A_CLAIM`.
"""
from __future__ import annotations
import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from w12_crossover_lib import (  # noqa: E402
    COUNTS_BYTES, COUPLING_BYTES, D, REFERENCE_PIN, ShimStore, atoms_for,
    four_part_ledger, load_partition, log, perturb_modes, pinned_basis, qr_basis,
    wells_for,
)
OUT = Path("C:/chlu/outputs/w12-single-design-crossover/val_sweep")
CONTROLS = (
    ("landscape_removed", "zero"),
    ("landscape_shuffled", "value_permute"),
    ("landscape_shuffled_across_wells", "well_permute"),
    ("random_atoms", "random_span"),
)
def score_corners(obj, corners, cond, grid, judge, bitmap, tag, log=log):
    """The 24-corner val read, through the SHARED frozen decoder + judge.

    Mirrors `scripts/w8_score_corners.py` exactly, so the number is produced by the same
    code path as w8's and the two waves are comparable.
    """
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    rows = []
    t0 = time.time()
    for c in corners:
        vals = fw._request_values_from_rows(grid, c["rows"])
        feats, oor = R.arm_features(obj, vals, grid)
        acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
        tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
        valid = fw.judge_valid(vals, tup, grid) & acc
        rows.append({"corner_id": c["corner_id"], "n": int(vals.shape[0]),
                     "accept": float(acc.mean()), "valid": float(valid.mean())})
    v = np.asarray([r["valid"] for r in rows], np.float64)
    log(f"    {tag:34s} valid {v.mean():.4f} +- {v.std(ddof=1):.4f} "
        f"[{v.min():.4f}, {v.max():.4f}]  ({time.time() - t0:.0f}s)")
    return {"valid_mean": float(v.mean()), "valid_sd": float(v.std(ddof=1)),
            "valid_min": float(v.min()), "valid_max": float(v.max()),
            "n_corners": len(rows), "per_corner": rows}


def run_one(K: int, seed: int, condition: str, corr, dead, pool, X, red,
            corners, cond, grid, judge, bitmap, do_controls: bool,
            do_exactness: bool, cache_manifest=None) -> dict:
    from chlu.core.gen_read import TensorAddress, assert_read_request_only
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_w8_single_design import CovContentStore

    wc = wells_for(corr, K, dead)
    sizes = [int(c.size) for c in wc]
    A = atoms_for(sizes)
    log(f"\n===== K={K} seed={seed} cond={condition}  wells "
        f"{min(sizes)}..{max(sizes)}  A={A} =====")

    t0 = time.time()
    st = CovContentStore(X, wc, A, D)
    t_build = time.time() - t0
    basis = pinned_basis(st)
    ranks = [int(b.shape[1]) for b in basis]
    rank_over_width = [round(r / s, 4) for r, s in zip(ranks, sizes)]
    log(f"  built {t_build:.1f}s  deployed_rank {min(st.deployed_rank)}.."
        f"{max(st.deployed_rank)}  span_rank {min(ranks)}..{max(ranks)} (sum {sum(ranks)})")
    log(f"  RANK/WIDTH  min {min(rank_over_width):.4f}  max {max(rank_over_width):.4f}"
        f"  (A/width_min = {A / min(sizes):.4f})")

    # ---- the address book -------------------------------------------------
    t0 = time.time()
    Y, T = R._pool_span_targets(st, pool, basis, log=lambda *_: None)
    ntr = int(0.9 * Y.shape[0])
    cards = [int(c) for c in grid.cardinalities]
    addr = TensorAddress(cards, rank=int(R.READ_PINS["rank"]),
                         ridge=float(R.READ_PINS["ridge"]),
                         seed=int(R.READ_PINS["address_seed"]))
    addr.multiplicative = True
    addr.fit(Y[:ntr], [t[:ntr] for t in T], sweeps=int(R.READ_PINS["sweeps"]),
             log=lambda *_: None, Y_val=Y[ntr:], targets_val=[t[ntr:] for t in T])
    t_addr = time.time() - t0
    abytes = int(addr.byte_ledger()["address_bytes"])
    log(f"  address book fitted {t_addr:.1f}s  {abytes:,} B  "
        f"(90/10 split, {ntr:,} train)")

    led = four_part_ledger(st, abytes)
    log(f"  LEDGER  atoms {led['p1_atom_bytes']:,} | cov+count "
        f"{led['p2_covariance_plus_count_bytes']:,} | address {abytes:,} | "
        f"coupling {led['p4_coupling_hole_table_bytes']:,}")
    log(f"          TOTAL {led['honest_total_bytes']:,} B = "
        f"{led['times_reference_pin']}x the 524,288 B REFERENCE (not a budget here)")

    vals0 = grid.values_of(pool[:64])
    ro = assert_read_request_only(addr, st, vals0, grid.values, basis)

    obj = {"arm": f"cov_tensor_K{K}", "read": "tensor", "kind": "span",
           "store": st, "basis": basis, "address": addr, "protos": None,
           "head_bytes": {"head_bytes": 0}}
    main_score = score_corners(obj, corners, cond, grid, judge, bitmap,
                               f"cov_tensor K={K}")

    rec = {
        "SMOKE_NOT_A_CLAIM": True,
        "condition": condition, "K": K, "seed": seed, "A_requested": A,
        "feature_cache": cache_manifest,
        "well_sizes": sizes, "deployed_rank": [int(r) for r in st.deployed_rank],
        "span_rank": ranks, "rank_over_width": rank_over_width,
        "rank_over_width_min": float(min(rank_over_width)),
        "rank_over_width_max": float(max(rank_over_width)),
        "saturated_wells": [d for d in sizes if A >= d],
        "criterion": "ward_balanced (LABEL-FREE correlation-dendrogram cut)",
        "ledger_four_part": led,
        "address_bytes": abytes,
        "build_seconds": t_build, "address_fit_seconds": t_addr,
        "request_only": {k: str(v) for k, v in (ro or {}).items()},
        "composition_val": main_score,
        "controls": {}, "exactness": None,
    }

    # ---- charter C.2 controls --------------------------------------------
    if do_controls:
        for name, kind in CONTROLS:
            mods = perturb_modes(st.modes, kind, seed)
            sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
            if kind == "zero":
                cobj = dict(obj, store=sh, basis=basis, zeroed=True, arm=name)
            else:
                cobj = dict(obj, store=sh, basis=qr_basis(mods, st.deployed_rank, widths=[b.shape[1] for b in basis]),
                            arm=name)
            sc = score_corners(cobj, corners, cond, grid, judge, bitmap, name)
            rec["controls"][name] = {k: v for k, v in sc.items() if k != "per_corner"}
            rec["controls"][name]["drop_from_clu"] = round(
                main_score["valid_mean"] - sc["valid_mean"], 4)

    # ---- exactness: downdate == refit-without -----------------------------
    if do_exactness:
        st2 = copy.deepcopy(st)
        t0 = time.time()
        receipt = st2.delete_rows(X[red])
        t_dd = time.time() - t0
        t0 = time.time()
        ref = CovContentStore(X[~red], wc, A, D)
        t_refit = time.time() - t0
        devs, sub = [], []
        for k in range(K):
            r = min(st2.deployed_rank[k], ref.deployed_rank[k])
            devs.append(float(np.max(np.abs(st2.modes[k] - ref.modes[k]))))
            if r > 0:
                Ud = np.linalg.qr(np.asarray(st2.modes[k][1:1 + r], np.float64).T)[0]
                Ur = np.linalg.qr(np.asarray(ref.modes[k][1:1 + r], np.float64).T)[0]
                sub.append(float(np.linalg.norm(Ud @ Ud.T - Ur @ Ur.T, 2)))
        aud = st2.basis_change_audit()
        rec["exactness"] = {
            "definition": ("STORE-SCOPED: the frozen encoder and decoder are NOT refit. "
                           "downdate(delete red rows) vs a store fitted from scratch "
                           "without those rows, same partition, same rank."),
            "n_rows_deleted": int(red.sum()),
            "max_abs_mode_dev": float(max(devs)), "per_well_mode_dev": devs,
            "max_subspace_projector_dist": float(max(sub)) if sub else None,
            "ranks_agree": bool(st2.deployed_rank == ref.deployed_rank),
            "downdate_seconds": t_dd, "refit_seconds": t_refit,
            "rows_refitted": 0, "data_passes_during_deletion": 0,
            "items_kept": False, "receipt": {k: str(v) for k, v in receipt.items()},
            "basis_change": aud,
        }
        log(f"  EXACTNESS  downdate {t_dd:.2f}s vs refit {t_refit:.1f}s  "
            f"max|mode dev| {max(devs):.3e}  max subspace dist "
            f"{max(sub) if sub else float('nan'):.3e}  ranks agree "
            f"{st2.deployed_rank == ref.deployed_rank}")
        del st2, ref

    d = OUT / f"{condition}_K{K}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "row.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    with open(d / "arm.pkl", "wb") as fh:
        pickle.dump({"store": st, "address": addr, "basis": basis,
                     "well_channels": wc, "K": K, "A": A, "seed": seed,
                     "condition": condition, "read": "tensor"}, fh)
    log(f"  wrote {d / 'row.json'}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val", choices=("val", "test"))
    ap.add_argument("--K", type=int, action="append", required=True)
    ap.add_argument("--seed", type=int, action="append", required=True)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-exactness", action="store_true")
    a = ap.parse_args(argv)

    from chlu.core.gen_store import channel_correlation
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    from w8_single_design_dev import red_mask

    fw.PINS["condition"] = a.condition
    _z, dead, pool = load_partition(a.condition)
    red = red_mask(pool)
    log(f"condition={a.condition}  pool {pool.size:,}  red rows {int(red.sum()):,}  "
        f"dead channels {int(dead.sum())}")

    all_rows = []
    for seed in a.seed:
        # THE CACHE GENERATION IS ASSERTED, NOT ASSUMED.  `CHLU_GEN_FEATURES_ROOT` is
        # NOT set in this box's ambient environment, so a run that forgets it silently
        # reads the frozen bridge -- GENERATION 1, which w5-common section 3.8 forbids
        # for any number.  This assertion is the reason the mistake cannot survive.
        fw.PINS["seed"] = int(seed)
        man = assert_generation2(int(seed))
        log(f"\n### seed {seed}: cache {man}")
        cond = fw.load_condition()
        grid = cond["grid"]
        judge = fw.Judge(int(seed), log=lambda *_: None)
        bitmap = fw.value_presence_bitmap(
            grid, np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond)))
        corners = (fw.val_corners(cond) if a.condition == "val"
                   else fw.nway_corners(cond) + fw.pair_subcorners(cond))
        log(f"    {len(corners)} {a.condition} corners, judge seed {seed}")
        F = fw.features_mmap(int(seed))
        Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
        corr = channel_correlation(Xc)
        del Xc
        X = np.asarray(fw.gather(F, pool), np.float32)
        for K in a.K:
            all_rows.append(run_one(K, int(seed), a.condition, corr, dead, pool, X, red,
                                    corners, cond, grid, judge, bitmap,
                                    not a.no_controls, not a.no_exactness,
                                    cache_manifest=man))
        del X, corr
    log(f"\n{len(all_rows)} rows written under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
