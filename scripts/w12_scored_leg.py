"""w12 hub/single-design-crossover -- THE SCORED LEG.  GATED.

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
    D, ShimStore, atoms_for, four_part_ledger, load_partition, log, perturb_modes,
    pinned_basis, qr_basis, wells_for,
)
from w12_val_sweep import CONTROLS  # noqa: E402

OUT = Path("C:/chlu/outputs/w12-single-design-crossover")
GATE = Path("C:/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json")
CACHE_GATE = Path("C:/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json")
LEDGER = Path("C:/chlu/SCORED-LEDGER.md")

#: The learned rivals the task file names, in its own order.  Rows 3 and 4 ask FULL
#: six-factor requests, so the `full_only` variants are the right ones
#: (`w4_learners.fit_family`: "rows 3 and 4 - the pair and n-way corners, whose runner
#: builds FULL six-factor requests").
#:
#: THESE ARE THE 524,288 B REFERENCE COLUMN, NOT THE 1x COLUMN, AND THE DIFFERENCE
#: RUNS AGAINST THIS HUB'S OWN INTEREST.  `spoke/deletion-baselines` matched them to
#: `param_target = 131,072` (= 524,288 B / 4) -- its BASELINES-READY.json pins say so.
#: This object's four-part total at K* is SEVERAL TIMES that, so a genuinely
#: 1x-matched rival would carry several times these parameters and would be STRONGER.
#: A win over this set is therefore NOT a 1x result and is never reported as one.
RIVALS = ("mlp", "lstm", "transformer", "cont_mlp")
FLOOR = "table"          # printed as a FLOOR, never as a rival
COMPARATORS_ROOT = "C:/chlu/outputs/w12-deletion-baselines/comparators"


def assert_gates(run_n: int, require_ledger: bool = True) -> dict:
    if not GATE.exists():
        raise SystemExit(
            f"GATE CLOSED: {GATE} does not exist. The scored leg is a DECLARED NOT-RUN "
            "with that reason. No single-design run has been spent.")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if require_ledger:
        txt = LEDGER.read_text(encoding="utf-8") if LEDGER.exists() else ""
        tag = f"w12 hub/single-design-crossover | single-design | {run_n}/3"
        if tag not in txt:
            raise SystemExit(
                f"LEDGER ROW MISSING: {LEDGER} has no row '{tag}'. Charter: the row is "
                "written BEFORE the run. Refusing to spend a scored run.")
    return gate


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


def summarise(rows, family=None) -> dict:
    sel = [r for r in rows if family is None or r["family"] == family]
    if not sel:
        return {"n_corners": 0, "DECLARED_NOT_RUN": f"no corner of family {family!r}"}
    nov = np.array([r["distinct_valid_novel_tuples"] for r in sel], float)
    jac = np.array([r["judge_accuracy"] for r in sel], float)
    return {"n_corners": len(sel),
            "novel_per_corner_mean": float(nov.mean()),
            "novel_per_corner_sd": float(nov.std(ddof=1)) if nov.size > 1 else None,
            "judge_accuracy_mean": float(jac.mean()),
            "n_requests_total": int(sum(r["n_requests"] for r in sel))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-n", type=int, required=True, help="1, 2 or 3")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=None, help="default: K* from CROSSOVER-val.json")
    ap.add_argument("--no-ledger-check", action="store_true",
                    help="DEV ONLY -- never for a scored run")
    a = ap.parse_args(argv)

    gate = assert_gates(a.run_n, require_ledger=not a.no_ledger_check)
    log(f"GATE OPEN: {GATE}\n  {json.dumps(gate, indent=2)[:1500]}")
    log(f"CACHE GATE (5 seeds): {CACHE_GATE} exists = {CACHE_GATE.exists()}"
        + ("" if CACHE_GATE.exists() else "  -> 3 SEEDS, DECLARED"))

    if a.K is None:
        doc = json.loads((OUT / "CROSSOVER-val.json").read_text(encoding="utf-8"))
        Kstar = doc["K_star"]
        if Kstar is None:
            raise SystemExit("K* is None in CROSSOVER-val.json -- the declared rule "
                             "selected nothing. That is the result; the rule is NOT "
                             "moved and no scored run is spent.")
        log(f"K* = {Kstar} (from CROSSOVER-val.json, rule frozen 11:10:24Z)")
    else:
        Kstar = int(a.K)
        log(f"K = {Kstar} (explicit override)")

    from chlu.core.gen_read import TensorAddress, assert_read_request_only
    from chlu.core.gen_store import channel_correlation
    from chlu.eval.rivals import w4_api
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    from chlu.experiments.exp_w8_single_design import CovContentStore

    seed = int(a.seed)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = seed
    man = assert_generation2(seed)
    log(f"cache {man}")

    _z, dead, pool = load_partition("test")
    cond = fw.load_condition()
    grid = cond["grid"]
    judge = fw.Judge(seed, log=lambda *_: None)
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    n4 = fw.nway_corners(cond)
    n3 = fw.pair_subcorners(cond)
    corners = n3 + n4
    log(f"TEST corners: row 3 = {len(n3)} pair sub-corners, row 4 = {len(n4)} n-way")

    F = fw.features_mmap(seed)
    Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
    corr = channel_correlation(Xc)
    del Xc
    X = np.asarray(fw.gather(F, pool), np.float32)

    wc = wells_for(corr, Kstar, dead)
    sizes = [int(c.size) for c in wc]
    A = atoms_for(sizes)
    t0 = time.time()
    st = CovContentStore(X, wc, A, D)
    t_build = time.time() - t0
    basis = pinned_basis(st)
    ranks = [int(b.shape[1]) for b in basis]
    rank_over_width = [round(r / s, 4) for r, s in zip(ranks, sizes)]
    log(f"store K={Kstar} A={A} built {t_build:.1f}s  RANK/WIDTH "
        f"{min(rank_over_width):.4f}..{max(rank_over_width):.4f}")

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
    led = four_part_ledger(st, abytes)
    log(f"address book {t_addr:.1f}s  {abytes:,} B")
    log(f"FOUR-PART LEDGER (summed, never netted): atoms {led['p1_atom_bytes']:,} | "
        f"cov+count {led['p2_covariance_plus_count_bytes']:,} | address {abytes:,} | "
        f"coupling {led['p4_coupling_hole_table_bytes']:,} = "
        f"{led['honest_total_bytes']:,} B  [{led['times_reference_pin']}x the "
        f"524,288 B REFERENCE column]")

    vals0 = grid.values_of(pool[:64])
    ro = assert_read_request_only(addr, st, vals0, grid.values, basis)

    arms = {}
    obj = {"arm": "cov_tensor", "read": "tensor", "kind": "span", "store": st,
           "basis": basis, "address": addr, "protos": None,
           "head_bytes": {"head_bytes": 0}}
    arms["cov_tensor"] = score_corners_full(obj, corners, grid, judge, bitmap,
                                            "cov_tensor (THE ARM)")

    for name, kind in CONTROLS:
        mods = perturb_modes(st.modes, kind, seed)
        sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
        cobj = (dict(obj, store=sh, basis=basis, zeroed=True, arm=name) if kind == "zero"
                else dict(obj, store=sh, basis=qr_basis(mods, st.deployed_rank, widths=[b.shape[1] for b in basis]),
                          arm=name))
        arms[name] = score_corners_full(cobj, corners, grid, judge, bitmap, name)

    rival_meta = {}
    from chlu.eval.rivals.w12_deletion_baselines import load_comparator
    jobs = [(f"{r}__as_trained", "w12 comparator (524,288 B reference column)")
            for r in RIVALS]
    jobs.append((FLOOR, "the prototype TABLE, printed as a FLOOR"))
    for name, role in jobs:
        try:
            rv = (w4_api.load_rival(seed, name) if name == FLOOR
                  else load_comparator(seed, name, root=COMPARATORS_ROOT))
            f0, _ = rv(grid.values_of(pool[:8]))
        except Exception as exc:                       # noqa: BLE001
            rival_meta[name] = {"DECLARED_NOT_RUN": f"{type(exc).__name__}: {exc}",
                                "role": role}
            log(f"    {name:38s} DECLARED NOT-RUN: {exc}")
            continue
        meta = getattr(rv, "meta", None) or getattr(getattr(rv, "read", None),
                                                    "meta", {})
        rival_meta[name] = {
            "param_count": meta.get("param_count"),
            "role": role,
            "matched_to_bytes": 524_288,
            "matched_to_params": 131_072,
            "NOT_1x": ("matched to the RETIRED 524,288 B pin, not to this object's "
                       "own four-part total. A 1x-matched rival would be STRONGER, "
                       "so this comparison FLATTERS the arm and no row against it is "
                       "reported as a 1x result."),
        }
        robj = {"arm": name, "read": "rival", "rival": rv, "rival_partial": None,
                "store": None, "head_bytes": {"head_bytes": 0}}
        arms[name] = score_corners_full(robj, corners, grid, judge, bitmap, name)

    rows = {a_: fw._per_corner(corners, c, grid, seen_mask) for a_, c in arms.items()}
    out = {
        "SCORED": True, "run_n": a.run_n, "condition": "test", "seed": seed,
        "K_star": Kstar, "A_requested": A, "well_sizes": sizes,
        "deployed_rank": [int(r) for r in st.deployed_rank],
        "rank_over_width": rank_over_width,
        "rank_over_width_max": float(max(rank_over_width)),
        "feature_cache": man, "gate": gate,
        "ledger_four_part": led, "address_bytes": abytes,
        "build_seconds": t_build, "address_fit_seconds": t_addr,
        "request_only": {k: str(v) for k, v in (ro or {}).items()},
        "n_corners": {"row3_pair": len(n3), "row4_nway": len(n4)},
        "rival_meta": rival_meta,
        "baseline_matching": {
            "1x_column": "DECLARED NOT-RUN",
            "1x_reason": ("the comparator set on this box is matched to "
                          "param_target = 131,072 (= 524,288 B / 4), declared in "
                          "BASELINES-READY.json pins. No 1x-matched entry point is "
                          "named there, and refitting four families at this "
                          "object's own four-part total was not affordable beside "
                          "the sweep."),
            "reference_column_bytes": 524_288,
            "direction_of_the_error": ("AGAINST this hub: a 1x-matched rival would "
                                       "carry several times these parameters and "
                                       "would be STRONGER, so the available "
                                       "comparison flatters the arm"),
        },
        "per_arm": {}, "paired_vs_cov_tensor": {},
    }
    for a_, rr in rows.items():
        out["per_arm"][a_] = {"row3_pair": summarise(rr, "pair"),
                              "row4_nway": summarise(rr, "nway"),
                              "all": summarise(rr), "per_corner": rr}
    for a_ in rows:
        if a_ == "cov_tensor":
            continue
        out["paired_vs_cov_tensor"][a_] = fw.paired_estimand(
            rows["cov_tensor"], rows[a_], "distinct_valid_novel_tuples", 1.0,
            "cov_tensor", a_)

    d = OUT / "scored" / f"run{a.run_n}_seed{seed}_K{Kstar}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scored.json").write_text(json.dumps(out, indent=2, default=str),
                                   encoding="utf-8")
    with open(d / "arm.pkl", "wb") as fh:
        pickle.dump({"store": st, "address": addr, "basis": basis,
                     "well_channels": wc, "K": Kstar, "A": A, "seed": seed,
                     "condition": "test", "read": "tensor"}, fh)
    log(f"wrote {d / 'scored.json'}")
    for a_ in arms:
        s3 = out["per_arm"][a_]["row3_pair"]
        s4 = out["per_arm"][a_]["row4_nway"]
        log(f"  {a_:38s} row3 {s3.get('novel_per_corner_mean', float('nan')):8.3f}  "
            f"row4 {s4.get('novel_per_corner_mean', float('nan')):8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
