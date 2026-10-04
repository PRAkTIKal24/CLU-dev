#!/usr/bin/env python
"""PREREG A3.11 — did the store RE-COMPOSE the deleted value? Measured geometrically.

⛔ The external probe is a 10-way classifier with its own biases, so *"the probe said
hue 5"* is a weaker claim than *"the emitted feature IS hue-5-like"*.  This unit reads
the **feature the store emits** and compares it, in the frozen encoder's own space, to
**real cached features of the same scene** at the deleted value and at its neighbours.

⭐ Exploits the fact that Shapes3D is a COMPLETE grid, so for every targeted request
(a full tuple with `object_hue = 5`) the *same scene with only the hue changed* exists
as a real row.  Verified:  row = ((((fh*10 + wh)*10 + oh)*8 + sc)*4 + sh)*15 + ori
reproduces `factor_labels` on every probe row.

    x_i        the feature the store EMITS for request i   (request-addressed)
    f5_i       the CACHED feature of the real row at oh = 5 (the same scene)
    f4_i, f6_i the CACHED features of the same scene at oh = 4 and oh = 6

Statistic: the fraction of requests where  ||x - f5|| < min(||x - f4||, ||x - f6||).

⛔ NO ENCODE: every comparison row is already in the frozen 480,000-row cache; this
script never constructs an encoder.  It also never touches the decoder or the probe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax  # noqa: E402

from chlu.config import CHLUConfig  # noqa: E402
from chlu.core.gen_bridge import dataset_module, features_for_rows, load_feature_cache  # noqa: E402
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior  # noqa: E402
from chlu.core.gen_dream import AddressBook  # noqa: E402
from chlu.core.gen_store import GenStore  # noqa: E402
from chlu.eval.gen_probe import factor_labels  # noqa: E402
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_unit import eval_row_sets  # noqa: E402

CARDS = [10, 10, 10, 8, 4, 15]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def row_of(t) -> int:
    r = 0
    for c, v in zip(CARDS, t):
        r = r * c + int(v)
    return r


def load_unit(d: Path, feat_shape):
    z = np.load(d / "modes.npz")
    nw = len([k for k in z.files if k.startswith("well_")])
    nf = len([k for k in z.files if k.startswith("cond_")])
    st = GenStore([z[f"chan_{j}"] for j in range(nw)], int(z["well_0"].shape[0]),
                  feat_shape, jax.random.PRNGKey(0),
                  init=[z[f"well_{j}"] for j in range(nw)])
    bk = AddressBook(z["log_prior"], [z[f"cond_{f}"] for f in range(nf)])
    return st, bk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="+", default=["det_B", "plc_f2v5", "del_f2v5"])
    ap.add_argument("--factor", type=int, default=2, help="object_hue")
    ap.add_argument("--value", type=int, default=5)
    ap.add_argument("--neighbours", nargs="+", type=int, default=[4, 6])
    ap.add_argument("--target", default="tgt_f2v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dev", type=int, default=4096)
    ap.add_argument("--n-target", type=int, default=1024)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    runs = Path(args.out)
    config = CHLUConfig()
    bdir = bridge_dir(config, "shapes3d", "recon", args.seed)
    crows, feats, cman = load_feature_cache(bdir, verify=False)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    split, _ = dataset_module("shapes3d").build_split(config.gen_data.data_root)
    grid = split.grid

    targets = json.loads((Path(args.out).parent / "targets.json").read_text())
    sets = eval_row_sets(split, config, args.seed, args.n_dev, targets, args.n_target)
    rows = sets[args.target]
    lab = np.asarray(factor_labels(grid, rows), np.int64)
    f, v = args.factor, args.value
    assert (lab[:, f] == v).all(), "targeted set must all name the deleted value"

    # the SAME SCENE at the deleted value and at each neighbour — real cached rows
    sib = {}
    for h in [v] + list(args.neighbours):
        alt = lab.copy()
        alt[:, f] = h
        r = np.array([row_of(t) for t in alt], np.int64)
        sib[h] = features_for_rows(crows, feats, r).reshape((r.size, -1)).astype(np.float64)
        log(f"cached features for the same scenes at {grid.names[f]}={h}: {sib[h].shape}")

    # ⭐ THE DATA'S OWN GEOMETRY, with no store involved at all.  Reported because it
    # governs how every store-relative number below must be read: the hue circle is
    # NOT uniformly spaced in the encoder's feature space, and the deleted value sits
    # OFF the chord joining its neighbours, so "neighbour average" is a poor model of
    # it and the mid-point is not a neutral origin.
    geom = {}
    nb = list(args.neighbours)
    for a in [v] + nb:
        for b in [v] + nb:
            if a < b:
                geom[f"dist_{a}_{b}"] = float(np.linalg.norm(sib[a] - sib[b], axis=1).mean())
    if len(nb) == 2:
        s1 = geom[f"dist_{min(v, nb[0])}_{max(v, nb[0])}"]
        s2 = geom[f"dist_{min(v, nb[1])}_{max(v, nb[1])}"]
        s3 = geom[f"dist_{min(nb)}_{max(nb)}"]
        geom["triangle_ratio"] = float((s1 + s2) / s3)
        geom["note"] = ("triangle_ratio > 1 means the deleted value sits OFF the chord "
                        "joining its neighbours: the hue manifold is curved, so the "
                        "neighbour mid-point is not a neutral origin.")
    log(f"data geometry (no store): {json.dumps(geom, default=float)}")

    R = lab.astype(np.int64)
    out = {}
    for tag in args.tags:
        d = runs / tag
        if not (d / "build.json").exists():
            log(f"SKIP {tag} — not built")
            continue
        st, bk = load_unit(d, feat_shape)
        codes = np.asarray(compose_codes(request_posterior(bk, R))["code"], np.int64)
        x = np.asarray(read_compose(st, codes), np.float32).reshape((R.shape[0], -1)).astype(np.float64)
        dist = {h: np.linalg.norm(x - sib[h], axis=1) for h in sib}
        dv = dist[v]
        dn = np.minimum.reduce([dist[h] for h in args.neighbours])
        closer = float((dv < dn).mean())
        # ⭐ THE SHARPER, SCALE-FREE STATISTIC.  Raw L2 in a 2048-d feature space is
        # dominated by SCENE content (five other factors), so the hue signal is a small
        # component of it and "closest of three" is blunt.  But f4, f5, f6 are the SAME
        # SCENE differing only in hue, so their differences isolate the hue direction:
        #     u   = f_v - mean(f_neighbours)          the "this value, not its neighbours" axis
        #     s   = <x - mean(f_neighbours), u> / ||u||^2
        # s = 1.0  -> the emitted feature sits exactly at the deleted value's hue offset
        # s = 0.0  -> it sits at the neighbour average, i.e. carries none of that value
        # ⛔ Scale-free and per-scene, so it does not need a chance baseline the way the
        # "closest of three" statistic does (that one's null is 1/3 by symmetry).
        mid = np.mean([sib[h] for h in args.neighbours], axis=0)
        u = sib[v] - mid
        un = np.einsum("ij,ij->i", u, u)
        s = np.einsum("ij,ij->i", x - mid, u) / np.maximum(un, 1e-12)
        # ⚠⚠ THE PROJECTION MUST NEVER BE REPORTED ALONE.  `s` measures MAGNITUDE
        # along a chord-based axis; the probe measures DISCRIMINABILITY.  They come
        # apart when the reconstruction error is largely ORTHOGONAL to the hue axis,
        # which is exactly the case here: perpendicular residual ~10.6 against an
        # along-axis component of ~1.0.  A reader given s = 0.148 alone would conclude
        # "the feature is nearly at the neighbour average"; the residual shows it is
        # not.  ⛔ I drew that wrong inference once; the residual is emitted so the
        # same mistake is not available to a reader.
        along = s[:, None] * u
        perp = (x - mid) - along
        out[tag] = {
            "n": int(R.shape[0]),
            # ⭐ PRIMARY feature-level statistic: parameter-free, needs no origin, and
            # is unaffected by the curvature that makes the mid-point a poor origin.
            "frac_emitted_closer_to_deleted_value_than_either_neighbour": closer,
            "closest_of_three_chance_level": 1.0 / 3.0,
            # secondary, more sensitive, ⛔ never quoted without the residual
            "hue_axis_projection_mean": float(s.mean()),
            "hue_axis_projection_median": float(np.median(s)),
            "hue_axis_projection_frac_above_half": float((s > 0.5).mean()),
            "axis_norm_u": float(np.linalg.norm(u, axis=1).mean()),
            # ⭐ SIGNED, and that is the point. `np.linalg.norm(along)` is mean(|s|·‖u‖)
            # = 1.0337 here, against the signed mean(s·‖u‖) = 0.7835 — a 32 % gap,
            # because 20.7 % of requests have NEGATIVE s (min −0.5638): for one request
            # in five the emitted feature sits on the far side of the neighbour average
            # FROM the deleted value, i.e. displaced AWAY from it along the hue axis.
            # ⛔ Taking |s| hides that and inflates the apparent hue-5 content.
            # ⚠ It also blinds the statistic to the very effect §6 is looking for: if
            # deletion randomises the SIGN of s while leaving its magnitude intact —
            # exactly what is expected when the hue term drops out of the addressing and
            # the residual displacement becomes arbitrary — then |s| barely moves while
            # signed s collapses toward 0. Signed is strictly the more sensitive read.
            "along_axis_component_SIGNED": float(np.mean(s * np.linalg.norm(u, axis=1))),
            "along_axis_component_abs": float(np.linalg.norm(along, axis=1).mean()),
            "projection_negative_fraction": float((s < 0).mean()),
            "projection_min": float(s.min()), "projection_max": float(s.max()),
            "perpendicular_residual": float(np.linalg.norm(perp, axis=1).mean()),
            "mean_dist_to_deleted_value": float(dv.mean()),
            **{f"mean_dist_to_neighbour_{h}": float(dist[h].mean()) for h in args.neighbours},
            "mean_margin_deleted_minus_best_neighbour": float((dv - dn).mean()),
            "n_live": json.loads((d / "build.json").read_text())["n_live"],
        }
        log(f"  [{tag}] closer-to-{grid.names[f]}={v}-than-either-neighbour: {closer:.4f}   "
            f"mean d(v)={dv.mean():.4f} vs best neighbour {dn.mean():.4f}")

    payload = {
        "schema": "chlu-hub-deletion/feature-probe/1",
        "factor": grid.names[f], "deleted_value": int(v),
        "neighbours": [int(h) for h in args.neighbours],
        "target_set": args.target, "arms": out, "data_geometry_no_store": geom,
        "what": ("⭐ geometric, not probe-based: is the feature the store EMITS for a "
                 "request naming the deleted value closer to REAL cached features of "
                 "that value than to its neighbours, in the frozen encoder's own space? "
                 "⛔ No encoder, no decoder, no probe is constructed here."),
        "primary_statistic": ("frac_emitted_closer_to_deleted_value_than_either_neighbour "
                          "- parameter-free, no origin, robust to manifold curvature. "
                          "The projection is SECONDARY and must be read WITH the "
                          "perpendicular residual."),
        "reading": ("det_B is the pre-deletion reference and plc_f2v5 the churn floor. "
                    "A del_f2v5 value near det_B's = the store still emits the deleted "
                    "value's content (re-composition by interpolation, NOT a retained "
                    "route, since A3.8 proves the conditional is exactly uniform). A "
                    "value near chance = the emitted feature is no longer that value's."),
        "wall_s": time.perf_counter() - t0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = Path(args.out).parent / "feature_probe.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    log(f"WROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
