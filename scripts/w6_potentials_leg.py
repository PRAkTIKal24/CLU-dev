#!/usr/bin/env python
"""w6 hub/potentials -- THE ONE LAUNCHER.  One leg = one (condition, seed)."""
import argparse, json, os, pickle, re, sys, time
from pathlib import Path
import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="val", choices=("val", "test"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--corners", default="nway", choices=("val", "nway", "pair", "row34"))
    p.add_argument("--max-corners", type=int, default=0)
    p.add_argument("--arms", default="field,landscape_removed,landscape_shuffled")
    p.add_argument("--wells", default="")
    p.add_argument("--aux-corners", type=int, default=0)
    p.add_argument("--control-corners", type=int, default=0)
    p.add_argument("--prereg", default="")
    a = p.parse_args(argv)

    if a.condition == "test":
        if not a.prereg or not Path(a.prereg).exists():
            log(f"REFUSING: condition=test is a SCORED RUN; PREREG {a.prereg!r} missing")
            return 2
        # The ledger row names this hub as "w6 hub/potentials"; accept either spelling so
        # the guard fails on a MISSING row and never on a stylistic one.  (It fired on the
        # real launch for exactly that reason, which is the guard working, and is logged.)
        led = Path("/c/chlu/SCORED-LEDGER.md")
        txt = led.read_text(encoding="utf-8", errors="replace") if led.exists() else ""
        if not re.search(r"w6[ /]hub/potentials|w6/potentials", txt):
            log("REFUSING: no /c/chlu/SCORED-LEDGER.md row for w6 hub/potentials")
            return 2
        log(f"scored run authorised by {a.prereg} + the ledger row")

    import chlu.experiments.exp_gen_read as R
    import chlu.experiments.exp_gen_factor_wells as fw
    import chlu.experiments.exp_w5_compose_field as W5
    from chlu.core.gen_field import field_address
    from chlu.core import w6_potentials as W6
    from chlu.experiments import exp_w6_potentials as EX

    W5.install(); R.set_condition(a.seed, a.condition); R.bind_instrument(log=lambda m: None)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    chk = EX.assert_map_matches_repo_integrator()
    EX.jdump(chk, out / "integrator_check.json")
    if not chk["passed"]:
        log("REFUSING: the rollout map does not reproduce the repo integrator")
        return 3
    log(f"integrator check: max|diff| = {chk['max_abs_diff']:.3e} -- PASSED")

    built = W5.build_field_for_leg(out, log=log)
    field = built["field"]
    cond = fw.load_condition(); grid = cond["grid"]
    log(f"field store sha {field.sha256()[:16]} ledger {built['byte_ledger']['deployed_bytes']:,} B")

    if a.corners == "val":
        cs = fw.val_corners(cond)
    elif a.corners == "nway":
        cs = fw.nway_corners(cond)
    elif a.corners == "pair":
        cs = fw.pair_subcorners(cond)
    else:
        cs = fw.nway_corners(cond) + fw.pair_subcorners(cond)
    if a.max_corners:
        cs = cs[: a.max_corners]
    log(f"{len(cs)} corners ({a.corners})")

    # one representative request per corner: the corner's own factor values
    vals = np.stack([grid.values_of(np.asarray(c["rows"][:1], np.int64))[0] for c in cs])
    addr = field_address(field, vals, grid.values)["mu"]
    log(f"addresses {addr.shape}")

    rng = np.random.default_rng(1234 + a.seed)
    wells = ([int(x) for x in a.wells.split(",") if x != ""] or None)
    want = [s for s in a.arms.split(",") if s]
    pca = None
    if "pca_tensor" in want:
        pc = W5.build_pca_content(out, log=log)
        pca = pc["store"]
    wl = list(wells if wells is not None else range(field.K))
    arms = {k: {kk: vv for kk, vv in
                EX.landscape_variants(field, k, np.random.default_rng(99 + k), pca).items()
                if kk in want}
            for k in wl}
    arms_field = {k: {kk: vv for kk, vv in arms[k].items() if kk == "field"} for k in wl}

    # the trap check, once, on the real atoms
    trap = {}
    for k in range(field.K):
        v = W6.well_view(field, k)
        trap[f"well_{k}"] = W6.sigma_provenance(v["atoms"], v["sigma"])
    EX.jdump(trap, out / "sigma_provenance.json")
    log("trap check (well_sigma vs item spacing): " + ", ".join(
        f"w{k}={trap[f'well_{k}']['well_sigma_over_nn_median']:.3f}" for k in range(field.K)))

    # the Hessian / D14 finite-difference check -- cheap, and it gates the predictions
    flo = EX.instrument_floor_on_exact_quadratic(eta=float(EX.PINS["eta"]))
    EX.jdump(flo, out / "instrument_floor.json")
    log(f"instrument floor on an EXACT quadratic (the CRR is true there by "
        f"construction): median |log10| = {flo['median_abs_log10']:.4f}, "
        f"p90 = {flo['p90_abs_log10']:.4f}, max = {flo['max_abs_log10']:.4f} "
        f"over {flo['n_cells']} cells")

    hfd = EX.hessian_fd_check(field, arms_field, addr[0], EX.PINS)
    EX.jdump(hfd, out / "hessian_fd_check.json")
    log(f"Hessian+D14 finite-difference check: max|resid| = {hfd['max_abs_residual']:.3e}")

    n_aux = int(a.aux_corners)
    n_ctl = int(a.control_corners)
    rows, t0 = [], time.time()
    for i, c in enumerate(cs):
        rows += EX.corner_rows(field, c["corner_id"], addr[i], arms_field, EX.PINS,
                               rng, wells)
        if i < n_aux:
            aux = EX.corner_rows(field, c["corner_id"], addr[i], arms_field,
                                 EX.PINS_AUX, rng, wells)
            for r in aux:
                r["grid"] = "aux"
            rows += aux
        if i < n_ctl:
            ctl = {k: {kk: vv for kk, vv in arms[k].items() if kk != "field"} for k in wl}
            if any(ctl.values()):
                rows += EX.corner_rows(field, c["corner_id"], addr[i], ctl, EX.PINS,
                                       rng, wells)
        if (i + 1) % 5 == 0 or i == len(cs) - 1:
            log(f"  corner {i+1}/{len(cs)}  rows={len(rows)}  {time.time()-t0:.0f}s")

    for r in rows:
        r.setdefault("grid", "main")

    main = [r for r in rows if r["arm"] == "field" and r["grid"] == "main"]
    fit = EX.fit_one_parameter(main)
    sc = EX.score_rows(rows, fit)
    est = EX.corner_estimand([r for r in sc["rows"] if r["grid"] == "main"], arm="field")
    vc = {arm: EX.v_curve_analysis([r for r in rows if r["grid"] == "main"], arm=arm)
          for arm in want}
    d14 = EX.d14_confinement_check([r for r in rows if r["grid"] == "aux"], arm="field")
    res = {"condition": a.condition, "seed": a.seed, "corners": a.corners,
           "n_corners": len(cs), "aux_corners": n_aux,
           "control_corners": n_ctl, "n_rows": len(rows),
           "pins": EX.PINS, "pins_aux": EX.PINS_AUX,
           "fit1_C": fit["C"], "fit1_n": fit["n"],
           "field_sha": field.sha256(), "byte_ledger": built["byte_ledger"],
           "integrator_check": chk, "hessian_fd_check": hfd,
           "instrument_floor": {k: v for k, v in flo.items() if k != "rows"},
           "sigma_provenance": trap,
           "estimand": est, "d14_confinement": d14,
           "n_v_curves": {k: len(v) for k, v in vc.items()},
           "wall_seconds": time.time() - t0}
    EX.jdump(res, out / "results.json")
    with open(out / "rows.pkl", "wb") as fh:
        pickle.dump({"rows": sc["rows"], "all_rows": rows, "v_curves": vc,
                     "meta": res}, fh)
    log(f"wrote {out/'results.json'}  rows={len(rows)}  corners={est['n_corners']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
