"""Build the PW-1 Shapes3D rendered figures into ONE directory (no timestamps, no build paths
in any written file, so two builds are byte-comparable).

    python -m pw1_render_shapes.build <out_dir>
    bash pw1_render_shapes/build_twice.sh

GATES (any failure stops before a figure is written):
  G1 object hashes    every staged w20 TEST object matches SHA256-w20-test-objects.txt
  G2 reproduction     the seed-0 scan this build draws from reproduced every w20 per-corner
                      score (scan.py); here every drawn request is re-rendered and its six
                      verdicts AND six read-backs must equal the scan's
  G3 same decoder     every panel stamped by the ONE hash-asserted decoder object
                      (w17 same_decoder.assert_same_decoder, against winner_hashes.json)
  G4 positive control (i) render+classify == Judge.read_back on the drawn n-CLU features;
                      (ii) a training-set request (fixed-seed draw): the dataset's own cached
                      encoder feature renders VALID, and the n-CLU renders it VALID
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import common as C
from . import scan as SC
from . import select as SEL

FIG_A = "rc_pw1_shapes_per_model"
FIG_B = "rc_pw1_shapes_baseline_wins"
FNAME = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
COL_SUB = {"mlp": "10\u00d7 training steps", "cont_mlp": "10\u00d7 training steps"}


def _fmt(grid, f, i):
    v = float(grid.values[f][int(i)])
    name = grid.names[f]
    if name == "shape":
        return C.SHAPE_NAMES[int(round(v))]
    if name == "orientation":
        return f"{v:.1f}°".replace("-", "−")
    return f"{v:.2f}" if name == "scale" else f"{v:.1f}"


def _describe(grid, idx) -> dict:
    return {grid.names[f]: {"index": int(idx[f]), "value": float(grid.values[f][int(idx[f])]),
                            "reads": _fmt(grid, f, idx[f])} for f in range(len(idx))}


def _row_label(grid, family, idx, head):
    g = lambda n: _fmt(grid, grid.names.index(n), idx[grid.names.index(n)])  # noqa: E731
    fam = ("held-out combination\n(object hue × shape)" if family == "pair"
           else "held-out combination\n(all six factors)")
    return (f"{head}\n{fam}\n{g('shape')}, object hue {g('object_hue')}\n"
            f"floor {g('floor_hue')}, wall {g('wall_hue')}\n"
            f"scale {g('scale')}, orient. {g('orientation')}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    g1 = C.verify_manifest()                                             # G1
    scan_p = C.OUT / "scan" / "verdicts-seed0.json"
    scan = json.loads(scan_p.read_text(encoding="utf-8"))
    if scan["reproduction"]["mismatches"] != 0:
        raise SystemExit("scan did not reproduce")
    ctx = SC.context(0)
    if ctx["od"].hash != scan["decoder_param_hash"]:
        raise SystemExit("decoder differs from the scan's")
    grid, fw = ctx["grid"], ctx["fw"]
    cols = list(C.ORDER)
    stamps, drawn_log = [], []

    def render_rows(sel_rows):
        imgs, verd, recs = [], [], []
        for s in sel_rows:
            vals, res = SC.read_rows(ctx, [s["dataset_row"]])
            bits = "".join("1" if res[c]["valid"][0] else "0" for c in cols)
            rb = [[int(x) for x in res[c]["tuples"][0]] for c in cols]
            srow = next(r for r in scan["requests"] if r[2] == s["dataset_row"])
            if bits != s["scan_bits"] or rb != srow[6]:                  # G2
                raise SystemExit(f"re-render differs from scan for row {s['dataset_row']}")
            for c in cols:
                stamps.append(res[c]["stamps"][0])
            imgs.append([res[c]["imgs"][0] for c in cols])
            verd.append([bool(res[c]["valid"][0]) for c in cols])
            idx = s["requested_factor_indices"]
            recs.append(dict(s, requested=_describe(grid, idx), judge={
                c: {"valid": bool(res[c]["valid"][0]), "accepted": bool(res[c]["accept"][0]),
                    "readback": _describe(grid, rb[k])} for k, c in enumerate(cols)},
                re_render_equals_scan=True))
        return imgs, verd, recs

    labels = [{"cont_mlp": "Continuous\nMLP", C.TABLE: "Lookup table\n(floor)"}.get(c, C.LABEL[c])
              for c in cols]
    subs = [COL_SUB.get(c, "") for c in cols]
    colours = [C.COLOUR[c] for c in cols]
    floor_col = cols.index(C.TABLE)

    # (a) per-model grid
    selA = SEL.per_model(scan)
    rowsA = [r for r in selA["rows"] if "DECLARED" not in r]
    imgsA, verdA, recA = render_rows(rowsA)
    rlA, hlA, grpA = [], {}, []
    for i, r in enumerate(rowsA):
        col, kind = r["row"].split(":")
        rlA.append(_row_label(grid, r["family"], r["requested_factor_indices"],
                              "success" if kind == "valid" else "failure"))
        hlA[(i, cols.index(col))] = C.COLOUR[col]
    for col in (C.CLU,) + C.BASELINES:
        ix = [i for i, r in enumerate(rowsA) if r["row"].startswith(col + ":")]
        if ix:
            grpA.append((C.LABEL[col], C.COLOUR[col], ix[0], ix[-1]))
    from .figure import draw_grid
    figs = {"per_model": draw_grid(out_dir / FIG_A, labels, subs, colours, rlA, imgsA, verdA,
                                   row_groups=grpA, highlight=hlA, floor_col=floor_col)}

    # (b) baseline succeeds, n-CLU fails
    selB = SEL.baseline_wins(scan)
    rowsB = [r for r in selB["rows"] if "DECLARED" not in r]
    recB = []
    if rowsB:
        imgsB, verdB, recB = render_rows(rowsB)
        rlB, hlB, grpB = [], {}, []
        for i, r in enumerate(rowsB):
            col = r["row"].split(":")[0]
            rlB.append(_row_label(grid, r["family"], r["requested_factor_indices"],
                                  f"{C.LABEL[col]} succeeds,\nn-CLU fails"))
            hlB[(i, cols.index(col))] = C.COLOUR[col]
            hlB[(i, cols.index(C.CLU))] = C.COLOUR[C.CLU]
            grpB.append(("", C.COLOUR[col], i, i))
        figs["baseline_wins"] = draw_grid(out_dir / FIG_B, labels, subs, colours, rlB, imgsB,
                                          verdB, row_groups=grpB, highlight=hlB,
                                          floor_col=floor_col)

    # G4 positive controls
    rows_all = [r["dataset_row"] for r in rowsA + rowsB]
    from chlu.experiments import exp_gen_read as R
    vv = fw._request_values_from_rows(grid, np.asarray(rows_all, np.int64))
    fc, _ = R.arm_features(ctx["objs"][C.CLU], vv, grid)
    pc_i = ctx["od"].positive_control(ctx["judge"], fc)
    rng = np.random.default_rng(SEL.SELECT_SEED)
    train = ctx["cond"]["train_rows"]
    trow = int(train[int(rng.integers(train.size))])
    fman = fw.feature_cache_manifest(0)
    fpath = fw.features_seed_dir(0) / "features.npy"
    fsha = C.sha256(fpath)
    if fsha != fman["features_sha256"]:
        raise SystemExit("feature cache sha256 differs from its manifest")
    Ft = fw.gather(fw.features_mmap(0), np.asarray([trow], np.int64))
    tv = fw._request_values_from_rows(grid, np.asarray([trow], np.int64))
    timg, tst = ctx["od"].render(Ft, arm="__positive_control_dataset_feature__")
    tok, ttup = ctx["od"].verdict(tv, np.ones(1, bool), timg, grid)
    stamps.extend(tst)
    _v, tres = SC.read_rows(ctx, [trow])
    for c in cols:
        stamps.append(tres[c]["stamps"][0])
    pc_ii = {"training_row": trow, "draw": f"default_rng({SEL.SELECT_SEED}).integers(len(train_rows))",
             "requested": _describe(grid, grid.factor_indices_of(np.asarray([trow]))[0]),
             "dataset_feature": {"valid": bool(tok[0]), "readback": _describe(grid, ttup[0])},
             "columns": {c: {"valid": bool(tres[c]["valid"][0]),
                             "readback": _describe(grid, tres[c]["tuples"][0])} for c in cols},
             "feature_cache_sha256": fsha}
    if not (bool(tok[0]) and bool(tres[C.CLU]["valid"][0])):
        (out_dir / "POSITIVE-CONTROL-FAILED.json").write_text(json.dumps(pc_ii, indent=1))
        raise SystemExit("POSITIVE CONTROL FAILED -- see POSITIVE-CONTROL-FAILED.json")
    from w17_rendered_compositions.same_decoder import assert_same_decoder
    g3 = assert_same_decoder(stamps, C.WINNER_HASHES, 0)                   # G3
    g3["against"] = "evidence/outputs/gen-decoder-lift/winner/winner_hashes.json"
    g3["classifier_param_hash"] = ctx["od"].classifier_hash
    g3.pop("decoder_object_id", None)

    others = {}
    for s in (1, 2):
        p = C.OUT / "scan" / f"verdicts-seed{s}.json"
        if p.exists():
            others[str(s)] = SEL.counts(json.loads(p.read_text(encoding="utf-8")))
    req = {
        "schema": "pw1-render-shapes/requests/1", "seed": 0, "K_star": 32,
        "columns": [{"key": c, "label": C.LABEL[c], "colour": C.COLOUR[c],
                     "object": ctx["meta"]["objects"][c].replace("/Users/batmac/clu/", "~/clu/")}
                    for c in cols],
        "carried_fit_seed0": ctx["meta"]["carried"],
        "selection_rule": SEL.__doc__, "select_seed": SEL.SELECT_SEED,
        "counts_seed0": SEL.counts(scan),
        "counts_seeds_1_2_not_drawn": others or "NOT SCANNED",
        "per_model": {"shape_order": selA["shape_order"], "shapes_present": selA["shapes_present"],
                      "declared": [r for r in selA["rows"] if "DECLARED" in r], "rows": recA},
        "baseline_wins": {"shape_order": selB["shape_order"],
                          "declared": [r for r in selB["rows"] if "DECLARED" in r], "rows": recB},
        "gates": {"G1_object_hashes": dict(g1, manifest="~/clu/stage/w20-shapes-data-matched/SHA256-w20-test-objects.txt"),
                  "G2_reproduction": scan["reproduction"] | {"drawn_rerender_equals_scan": True},
                  "G3_same_decoder": g3,
                  "G4_positive_control": {"render_equals_Judge_read_back": pc_i,
                                          "training_set_request": pc_ii}},
        "figures": figs,
    }
    (out_dir / "REQUESTS-pw1.json").write_text(json.dumps(req, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
