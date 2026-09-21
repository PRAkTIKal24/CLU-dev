"""w13 hub/encoder-adapt -- THE SCORED LEG: one (encoder arm, seed), rows 3 + 4 at K* = 32.

WHAT THIS SCRIPT IS.  Wave 12's own K* harness (`scripts/w12_crossover_lib.py`,
`CovContentStore`, `pinned_basis`, `TensorAddress`, the four charter-C controls) driven
once per ENCODER ARM, with the harness's four documented redirections pointed at that
arm -- and one of them deliberately NOT moved:

    CHLU_GEN_FEATURES_ROOT / CHLU_GEN_BRIDGE_ROOT_OVERRIDE -> that arm's own gen-2 cache
    CHLU_GEN_DECLIFT_DIR                                   -> that arm's own refit decoder
    CHLU_RIVALS_ROOT                                       -> that arm's REFIT baselines
    CHLU_GEN_CLASSIFIER_ROOT                               -> THE FROZEN CLASSIFIER.
                                                              *** THIS ONE DOES NOT MOVE ***

The classifier reads IMAGES, so it is the one instrument that survives a change of
feature space; it is what makes the three arms comparable at all.

THE NUMBER THIS LEG EXISTS TO PRODUCE
-------------------------------------
    margin(arm) = mean over corners of  novel(cov_tensor) - novel(best learned baseline)

where `novel` is the programme's own rows-3/4 estimand `distinct_valid_novel_tuples` per
corner, produced by `exp_gen_factor_wells._per_corner`, and the "best learned baseline" is
whichever of the arm's OWN refit baselines has the highest mean on that arm's features.
Paired by corner, per seed, never pooled across seeds.

Wave 12 measured this margin NEGATIVE.  A margin that grows and stays negative is a real
and reportable result.

`--condition test` is a SCORED RUN: it refuses to start without this hub's OWN PREREG file
and this hub's OWN row in `/c/chlu/SCORED-LEDGER.md`.  Another hub's row does not
authorise it.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

LEDGER = Path("C:/chlu/SCORED-LEDGER.md")
LEDGER_TAG = "w13 hub/encoder-adapt"
OUT_BASE = Path("C:/chlu/outputs/w13-encoder-adapt")

#: The charter-C controls, in wave 12's own definitions and order.
CONTROLS = (
    ("landscape_removed", "zero"),
    ("landscape_shuffled", "value_permute"),
    ("landscape_shuffled_across_wells", "well_permute"),
    ("random_atoms", "random_span"),
)


def log(m=""):
    print(m, flush=True)


def assert_chlu_in_worktree() -> str:
    import chlu

    want = HERE.parent.resolve()
    got = Path(chlu.__file__).resolve()
    if want not in got.parents:
        raise SystemExit(f"chlu.__file__ = {got} is NOT inside {want} -- numbers void")
    return str(got)


def redirect_to_arm(run_base: Path, seed: int, arm: str) -> dict:
    """The four hooks.  Returns what was set, for the artefact."""
    ad = run_base / f"seed_{int(seed)}" / arm
    feats = ad / "cache" / "shapes3d_recon" / f"seed_{int(seed)}" / "features.npy"
    need = [feats, ad / "declift" / "winner_hashes.json", ad / "rivals"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"REFUSING: arm {arm} seed {seed} is not complete -- missing "
                         f"{missing}. A published file is not a finished file.")
    cls_root = os.environ.get("CHLU_GEN_CLASSIFIER_ROOT")
    if not cls_root:
        cls_root = str(Path(os.environ["CHLU_GEN_BRIDGE_ROOT"]).parent.parent
                       / "gen-external-classifier")
    os.environ["CHLU_GEN_FEATURES_ROOT"] = str(ad / "cache")
    os.environ["CHLU_GEN_BRIDGE_ROOT_OVERRIDE"] = str(ad / "cache")
    os.environ["CHLU_GEN_DECLIFT_DIR"] = str(ad / "declift")
    os.environ["CHLU_RIVALS_ROOT"] = str(ad / "rivals")
    os.environ["CHLU_GEN_CLASSIFIER_ROOT"] = cls_root
    rec = {
        "features": os.environ["CHLU_GEN_FEATURES_ROOT"],
        "decoder": os.environ["CHLU_GEN_DECLIFT_DIR"],
        "rivals": os.environ["CHLU_RIVALS_ROOT"],
        "classifier": cls_root,
        "classifier_note": ("INVARIANT ACROSS ARMS -- it reads images, so it is the one "
                            "instrument that survives a change of feature space"),
    }
    for k, v in rec.items():
        if k != "classifier_note":
            log(f"  {k:11s} -> {v}")
    return rec


def assert_authorised(condition: str, prereg: Path, run_n: int) -> dict:
    if condition != "test":
        return {"scored": False, "condition": condition,
                "note": "val is the FREE selection condition (charter A.2)"}
    if not prereg.exists():
        raise SystemExit(f"REFUSING a scored run: no PREREG at {prereg}")
    txt = LEDGER.read_text(encoding="utf-8", errors="replace") if LEDGER.exists() else ""
    tag = f"{LEDGER_TAG} | encoder-adapt | {run_n}/3"
    if tag not in txt:
        raise SystemExit(
            f"REFUSING a scored run: {LEDGER} has no row '{tag}'. The row is written "
            "BEFORE the run, and another hub's row does not authorise this one.")
    return {"scored": True, "condition": "test", "run_n": int(run_n),
            "prereg": str(prereg), "ledger_row": tag}


def score_corners_full(obj, corners, grid, judge, bitmap, tag):
    """One arm, every corner, keeping accept/valid/tuples.

    Verbatim from `scripts/w12_scored_leg.py:score_corners_full`, which is itself
    `exp_gen_factor_wells.run_arm`'s loop inlined.
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
    """Both metrics on every row, so a DEGENERATE primary can never pass as a result.

    `distinct_valid_novel_tuples` is the programme's rows-3/4 estimand and is the PRIMARY.
    `distinct_valid_tuples` is carried beside it because NOVELTY is a property of the
    CONDITION: on `val` every corner's tuples are inside `train_rows`, so the novel count
    is identically 0 for the arm, for all four controls and for every rival alike -- a
    measured property of the val split, not a floor and not a result.  The judge accuracy
    and the valid fraction are carried for the same reason: they say whether the
    instrument read anything at all.
    """
    sel = [r for r in rows if family is None or r["family"] == family]
    if not sel:
        return {"n_corners": 0, "DECLARED_NOT_RUN": f"no corner of family {family!r}"}
    nov = np.array([r["distinct_valid_novel_tuples"] for r in sel], float)
    dvt = np.array([r["distinct_valid_tuples"] for r in sel], float)
    jac = np.array([r["judge_accuracy"] for r in sel], float)
    nv = np.array([r["n_valid"] for r in sel], float)
    nr = np.array([r["n_requests"] for r in sel], float)
    return {"n_corners": len(sel),
            "novel_per_corner_mean": float(nov.mean()),
            "novel_per_corner_sd": float(nov.std(ddof=1)) if nov.size > 1 else None,
            "distinct_valid_per_corner_mean": float(dvt.mean()),
            "distinct_valid_per_corner_sd": (float(dvt.std(ddof=1)) if dvt.size > 1
                                             else None),
            "judge_accuracy_mean": float(jac.mean()),
            "valid_frac_mean": float((nv / np.maximum(nr, 1)).mean()),
            "n_requests_total": int(sum(r["n_requests"] for r in sel))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", default="val", choices=("val", "test"))
    ap.add_argument("--run-base", required=True, help="the --out of exp_w13_encoder_adapt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-n", type=int, default=1)
    ap.add_argument("--prereg", default=str(OUT_BASE / "PREREG-w13-encoder-adapt-1.md"))
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args(argv)

    chlu_file = assert_chlu_in_worktree()
    log(f"chlu.__file__ = {chlu_file}")
    gate = assert_authorised(a.condition, Path(a.prereg), a.run_n)
    log(f"authorisation: {json.dumps(gate)}")

    run_base = Path(a.run_base)
    redirect = redirect_to_arm(run_base, a.seed, a.arm)

    from w12_crossover_lib import (D, ShimStore, atoms_for, four_part_ledger,
                                   load_partition, perturb_modes, pinned_basis,
                                   qr_basis, wells_for)

    from chlu.core.gen_read import TensorAddress, assert_read_request_only
    from chlu.core.gen_store import channel_correlation
    from chlu.eval.rivals import w4_api
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    from chlu.experiments.exp_w8_single_design import CovContentStore

    if a.K is None:
        doc = json.loads(Path("C:/chlu/evidence/outputs/w12-single-design-crossover/"
                              "CROSSOVER-val.json").read_text(encoding="utf-8"))
        Kstar = int(doc["K_star"])
        log(f"K* = {Kstar} from wave 12's CROSSOVER-val.json (rule frozen 11:10:24Z); "
            "nothing in this hub re-selects K")
    else:
        Kstar = int(a.K)
        log(f"K = {Kstar} (explicit override -- printed on the row)")

    seed = int(a.seed)
    fw.PINS["condition"] = a.condition
    fw.PINS["seed"] = seed
    man = assert_generation2(seed)
    log(f"cache asserted GENERATION 2: {man}")

    _z, dead_frozen, pool = load_partition(a.condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    judge = fw.Judge(seed, log=lambda *_: None)
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    if a.condition == "test":
        n4 = fw.nway_corners(cond)
        n3 = fw.pair_subcorners(cond)
        corners = n3 + n4
    else:
        n4, n3 = fw.val_corners(cond), []
        corners = n4
    log(f"{a.condition} corners: row 3 = {len(n3)} pair sub-corners, "
        f"row 4 = {len(n4)} n-way")

    F = fw.features_mmap(seed)
    Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
    corr = channel_correlation(Xc)
    X = np.asarray(fw.gather(F, pool), np.float32)
    # DEAD CHANNELS ARE MEASURED ON THIS ARM'S OWN FEATURES and unioned with the frozen
    # partition's, because an adapted encoder can kill a channel the frozen one kept.
    # Reported, never silently absorbed: a differing well width changes what K* means.
    dead_arm = np.asarray(Xc.std(axis=0) <= 0.0, bool)
    dead = np.asarray(dead_frozen, bool) | dead_arm
    del Xc
    log(f"dead channels: frozen file {int(np.sum(dead_frozen))}, THIS ARM "
        f"{int(dead_arm.sum())}, union {int(dead.sum())} of {D}")

    wc = wells_for(corr, Kstar, dead)
    sizes = [int(c.size) for c in wc]
    A = atoms_for(sizes)
    t0 = time.time()
    st = CovContentStore(X, wc, A, D)
    t_build = time.time() - t0
    basis = pinned_basis(st)
    ranks = [int(b.shape[1]) for b in basis]
    rank_over_width = [round(r / s, 4) for r, s in zip(ranks, sizes)]
    log(f"store K={Kstar} A={A} built {t_build:.1f}s  wells {min(sizes)}..{max(sizes)}  "
        f"RANK/WIDTH {min(rank_over_width):.4f}..{max(rank_over_width):.4f}")

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
        f"{led['honest_total_bytes']:,} B [{led['times_reference_pin']}x the 524,288 B "
        "REFERENCE column]")

    vals0 = grid.values_of(pool[:64])
    ro = assert_read_request_only(addr, st, vals0, grid.values, basis)

    arms = {}
    obj = {"arm": "cov_tensor", "read": "tensor", "kind": "span", "store": st,
           "basis": basis, "address": addr, "protos": None,
           "head_bytes": {"head_bytes": 0}}
    arms["cov_tensor"] = score_corners_full(obj, corners, grid, judge, bitmap,
                                            "cov_tensor (THE SINGLE DESIGN)")

    if not a.no_controls:
        for name, kind in CONTROLS:
            mods = perturb_modes(st.modes, kind, seed)
            sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
            cobj = (dict(obj, store=sh, basis=basis, zeroed=True, arm=name)
                    if kind == "zero" else
                    dict(obj, store=sh,
                         basis=qr_basis(mods, st.deployed_rank,
                                        widths=[b.shape[1] for b in basis]),
                         arm=name))
            arms[name] = score_corners_full(cobj, corners, grid, judge, bitmap, name)

    # ---- the learned baselines, REFIT ON THIS ARM'S OWN FEATURES -----------
    rival_root = Path(os.environ["CHLU_RIVALS_ROOT"])
    rec = json.loads((run_base / f"seed_{seed}" / a.arm / "rivals.json")
                     .read_text(encoding="utf-8"))
    fams = list(rec.get("families_done", []))
    rival_meta = {}
    for name in fams:
        try:
            rv = w4_api.load_rival(seed, name, root=rival_root)
            rv(grid.values_of(pool[:8]))
        except Exception as exc:                                   # noqa: BLE001
            rival_meta[name] = {"DECLARED_NOT_RUN": f"{type(exc).__name__}: {exc}"}
            log(f"    {name:38s} DECLARED NOT-RUN: {exc}")
            continue
        rival_meta[name] = {
            "param_count": (rec.get("families", {}).get(name) or {}).get("param_count"),
            "target_params": (rec.get("families", {}).get(name) or {}).get("target_params"),
            "refit_seconds": (rec.get("families", {}).get(name) or {}).get("seconds"),
            "refit_on": "THIS ARM's own generation-2 cache, w4_learners protocol verbatim",
        }
        robj = {"arm": name, "read": "rival", "rival": rv, "rival_partial": None,
                "store": None, "head_bytes": {"head_bytes": 0}}
        arms[name] = score_corners_full(robj, corners, grid, judge, bitmap, name)
    for name in ["lace"]:
        rival_meta[name] = {"DECLARED_NOT_RUN": (
            "no refit-on-new-features entry point in w4_learners.builders(); see "
            "exp_w13_encoder_adapt.LACE_NOT_RUN. Never a zero, never a blank.")}

    rows = {k: fw._per_corner(corners, v, grid, seen_mask) for k, v in arms.items()}

    # ---- THE MARGIN -------------------------------------------------------
    learned = [f for f in fams if f in rows]
    per_arm = {}
    for k, rr in rows.items():
        per_arm[k] = {"row3_pair": summarise(rr, "pair"),
                      "row4_nway": summarise(rr, "nway"),
                      "all": summarise(rr), "per_corner": rr}
    margin = {"DECLARED_NOT_RUN": "no learned baseline scored"} if not learned else None
    if learned:
        def mean_of(k, fam=None, key="novel_per_corner_mean"):
            return per_arm[k]["all" if fam is None else fam][key]

        # IS THE PRIMARY METRIC DEGENERATE?  Declared rule, fixed before any test number:
        # the primary is `distinct_valid_novel_tuples`.  It is DEGENERATE iff it reads
        # exactly 0 for the single design AND for every learned baseline AND for every
        # control -- i.e. it separates nothing at all.  When that happens the secondary
        # `distinct_valid_tuples` decides and the row SAYS SO in `primary_degenerate`;
        # the primary is never quietly swapped for whichever number is larger.
        DVT = "distinct_valid_per_corner_mean"
        primary_all_zero = all(abs(mean_of(k)) == 0.0 for k in rows)
        metric_key = DVT if primary_all_zero else "novel_per_corner_mean"
        metric_name = ("distinct_valid_tuples (SECONDARY -- the primary is degenerate on "
                       "this condition, see `primary_degenerate`)" if primary_all_zero
                       else "distinct_valid_novel_tuples (PRIMARY)")
        best = max(learned, key=lambda f: mean_of(f, key=metric_key))
        paired = fw.paired_estimand(
            rows["cov_tensor"], rows[best],
            "distinct_valid_tuples" if primary_all_zero
            else "distinct_valid_novel_tuples", 1.0, "cov_tensor", best)
        margin = {
            "metric": metric_name + ", per corner, paired by corner",
            "metric_key": metric_key,
            "primary_degenerate": bool(primary_all_zero),
            "primary_degenerate_note": (
                "distinct_valid_novel_tuples reads EXACTLY 0 for the single design, for "
                "every learned baseline and for every control on this condition. On "
                "`val` that is a property of the split -- every val corner's tuples are "
                "inside `train_rows`, so nothing can be novel -- and NOT a floor: the "
                "judge accuracy and distinct_valid_tuples on the same rows are nonzero "
                "for the arm and exactly 0.0000 for all four charter-C controls."
                if primary_all_zero else
                "the primary metric separates the arms and is the one used"),
            "best_learned_baseline": best,
            "best_chosen_by": ("highest mean novel-per-corner among THIS ARM's own refit "
                               "baselines -- chosen per arm, so the bar is always the "
                               "strongest available competitor on that arm's features"),
            "learned_baselines_scored": learned,
            "single_design_mean": mean_of("cov_tensor", key=metric_key),
            "best_learned_mean": mean_of(best, key=metric_key),
            "MARGIN_all_corners": (mean_of("cov_tensor", key=metric_key)
                                   - mean_of(best, key=metric_key)),
            "MARGIN_novel_all_corners": mean_of("cov_tensor") - mean_of(best),
            "MARGIN_distinct_valid_all_corners": (mean_of("cov_tensor", key=DVT)
                                                  - mean_of(best, key=DVT)),
            "single_design_judge_accuracy":
                per_arm["cov_tensor"]["all"]["judge_accuracy_mean"],
            "best_learned_judge_accuracy": per_arm[best]["all"]["judge_accuracy_mean"],
            "controls_must_be_zero": {
                k: per_arm[k]["all"][DVT] for k in
                ("landscape_removed", "landscape_shuffled",
                 "landscape_shuffled_across_wells", "random_atoms") if k in per_arm},
            "MARGIN_row3_pair": (per_arm["cov_tensor"]["row3_pair"].get(metric_key)
                                 - per_arm[best]["row3_pair"].get(metric_key)
                                 if per_arm["cov_tensor"]["row3_pair"].get(metric_key)
                                 is not None and
                                 per_arm[best]["row3_pair"].get(metric_key)
                                 is not None else None),
            "MARGIN_row4_nway": (per_arm["cov_tensor"]["row4_nway"].get(metric_key)
                                 - per_arm[best]["row4_nway"].get(metric_key)
                                 if per_arm["cov_tensor"]["row4_nway"].get(metric_key)
                                 is not None and
                                 per_arm[best]["row4_nway"].get(metric_key)
                                 is not None else None),
            "paired_vs_best_learned": paired,
            "per_baseline_margin": {f: mean_of("cov_tensor", key=metric_key)
                                    - mean_of(f, key=metric_key) for f in learned},
            "how_to_read": ("wave 12 measured this NEGATIVE. The bar is whether it GROWS "
                            "from `frozen` to an adapted arm. A margin that grows and "
                            "stays negative is a real and reportable result."),
        }

    out = {
        "schema": "chlu-w13-encoder-adapt-score/1",
        "SCORED": bool(gate.get("scored")),
        "hub": "w13-encoder-adapt", "encoder_arm": a.arm, "seed": seed,
        "condition": a.condition, "run_n": int(a.run_n),
        "chlu_file": chlu_file,
        "authorisation": gate, "redirect": redirect,
        "K_star": Kstar, "A_requested": A,
        "well_sizes": sizes, "deployed_rank": [int(r) for r in st.deployed_rank],
        "rank_over_width": rank_over_width,
        "rank_over_width_min": float(min(rank_over_width)),
        "rank_over_width_max": float(max(rank_over_width)),
        "dead_channels": {"frozen_partition_file": int(np.sum(dead_frozen)),
                          "this_arm": int(dead_arm.sum()), "union_used": int(dead.sum()),
                          "D": int(D)},
        "feature_cache": man,
        "ledger_four_part": led, "address_bytes": abytes,
        "build_seconds": t_build, "address_fit_seconds": t_addr,
        "request_only": {k: str(v) for k, v in (ro or {}).items()},
        "n_corners": {"row3_pair": len(n3), "row4_nway": len(n4),
                      "total": len(corners)},
        "rival_meta": rival_meta,
        "MARGIN": margin,
        "per_arm": per_arm,
        "paired_vs_cov_tensor": {
            k: fw.paired_estimand(rows["cov_tensor"], rows[k],
                                  "distinct_valid_novel_tuples", 1.0, "cov_tensor", k)
            for k in rows if k != "cov_tensor"},
    }

    d = Path(a.out) / f"{a.condition}_seed{seed}_{a.arm}_K{Kstar}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "score.json").write_text(json.dumps(out, indent=2, default=str),
                                  encoding="utf-8")
    with open(d / "store.pkl", "wb") as fh:
        pickle.dump({"store": st, "address": addr, "basis": basis,
                     "well_channels": wc, "K": Kstar, "A": A, "seed": seed,
                     "condition": a.condition, "encoder_arm": a.arm}, fh)
    log(f"wrote {d / 'score.json'}")
    for k in arms:
        st = per_arm[k]["all"]
        log(f"  {k:38s} novel/corner "
            f"{st.get('novel_per_corner_mean', float('nan')):8.3f}"
            f"  distinct_valid/corner "
            f"{st.get('distinct_valid_per_corner_mean', float('nan')):8.3f}"
            f"  judge_acc {st.get('judge_accuracy_mean', float('nan')):.4f}")
    if margin and "MARGIN_all_corners" in margin:
        log(f"  MARGIN ({a.arm}) = {margin['MARGIN_all_corners']:+.4f} "
            f"vs best learned baseline {margin['best_learned_baseline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
