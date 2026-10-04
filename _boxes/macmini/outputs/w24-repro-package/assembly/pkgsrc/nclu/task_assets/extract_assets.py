"""extract assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from nclu.experiments import factor_wells as fw
from nclu.paths import run_dir

REQUESTS = run_dir("figures", "render_shapes", "REQUESTS-render.json")
NWAY_ID = "nway-034"


def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def counts2(fi, a, b, cards):
    m = np.zeros((cards[a], cards[b]), np.int64)
    np.add.at(m, (fi[:, a], fi[:, b]), 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    rec = {}
    req = json.loads(REQUESTS.read_text())["requests"][0]
    assert req["set"] == "set1" and req["dataset_row"] == 424182, req
    row = int(req["dataset_row"])
    rec["request"] = {
        "source": str(REQUESTS) + " requests[0]",
        "corner_id": req["corner_id"],
        "dataset_row": row,
        "factor_values": req["factor_values"],
    }
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = tuple((int(c) for c in grid.cardinalities))
    assert np.allclose(grid.values_of(np.array([row]))[0], req["factor_values"])
    req_idx = grid.factor_indices_of(np.array([row]))[0]
    man = fw.feature_cache_manifest(0)
    fpath = fw.features_seed_dir(0) / "features.npy"
    got = sha(fpath)
    assert got == man["features_sha256"], (got, man["features_sha256"])
    F = fw.features_mmap(0)
    feat = fw.gather(F, np.array([row]))
    judge = fw.Judge(0)
    from nclu.eval.gen_probe import decode_features, predict_factors

    Z = feat.reshape((-1,) + tuple(fw.PINS["feat_shape"]))
    img = decode_features(judge.decoder, Z, 1)
    tup = np.asarray(predict_factors(judge.external, img, 1), np.int64)
    assert np.array_equal(tup, judge.read_back(feat, 1)), "render path != Judge.read_back"
    valid = bool(fw.judge_valid(np.array([req["factor_values"]]), tup, grid)[0])
    rec["render"] = {
        "decoder": judge.info,
        "features_sha256": got,
        "feature_generation": man["generation"],
        "seed": 0,
        "read_back_indices": tup[0].tolist(),
        "request_indices": req_idx.tolist(),
        "judge_valid": valid,
        "image_shape": list(img.shape),
        "image_range": [float(img.min()), float(img.max())],
    }
    assert fw.PINS["condition"] == "test"
    pool = fw.fit_pool(cond["train_rows"], fw.PINS["n_pool"], fw.PINS["pool_rng_seed"])
    fi_pool = grid.factor_indices_of(np.sort(pool))
    fi_train = grid.factor_indices_of(cond["train_rows"])
    names = list(grid.names)
    oh, sh = (names.index("object_hue"), names.index("shape"))
    pair_pool = counts2(fi_pool, oh, sh, cards)
    pair_train = counts2(fi_train, oh, sh, cards)
    cells = cond["cells"]["cells"]
    pairs = [c for c in cells if c["meta"]["family"] == "pair"]
    pair_idx = []
    for c in pairs:
        p = c["meta"]["pair"]
        i = int(np.argmin(np.abs(grid.values[oh] - p["object_hue"])))
        j = int(np.argmin(np.abs(grid.values[sh] - p["shape"])))
        assert pair_train[i, j] == 0 and pair_pool[i, j] == 0, c["cell_id"]
        pair_idx.append([i, j, c["cell_id"]])
    vc = cond["cells"]["config"]
    vh_hue = int(np.argmin(np.abs(grid.values[oh] - vc["held_out_hue_value"])))
    vh_shape = int(np.argmin(np.abs(grid.values[sh] - vc["held_out_shape_value"])))
    assert pair_train[vh_hue].sum() == 0 and pair_train[:, vh_shape].sum() == 0
    nway = [c for c in cells if c["cell_id"] == NWAY_ID][0]
    box = {}
    for f, n in enumerate(names):
        box[n] = sorted(
            (int(np.argmin(np.abs(grid.values[f] - float(v)))) for v in nway["values"][n])
        )
    fl, wl, sc, orr = (names.index(n) for n in ("floor_hue", "wall_hue", "scale", "orientation"))
    marg_fw = counts2(fi_pool, fl, wl, cards)
    marg_so = counts2(fi_pool, sc, orr, cards)
    marg_os = counts2(fi_pool, oh, sh, cards)
    in_other = (
        np.isin(fi_pool[:, fl], box["floor_hue"])
        & np.isin(fi_pool[:, wl], box["wall_hue"])
        & np.isin(fi_pool[:, oh], box["object_hue"])
        & np.isin(fi_pool[:, sh], box["shape"])
    )
    slice_so = counts2(fi_pool[in_other], sc, orr, cards)
    corner_mask = np.ones(len(fi_pool), bool)
    for f, n in enumerate(names):
        corner_mask &= np.isin(fi_pool[:, f], box[n])
    assert corner_mask.sum() == 0
    proj = {}
    for a in range(6):
        for b in range(a + 1, 6):
            m = counts2(fi_pool, a, b, cards)
            sub = m[np.ix_(box[names[a]], box[names[b]])]
            proj[f"{names[a]}x{names[b]}"] = int(sub.min())
    assert min(proj.values()) > 0, proj
    rec["fit_pool"] = {
        "n_pool": int(len(pool)),
        "pool_rng_seed": fw.PINS["pool_rng_seed"],
        "condition": fw.PINS["condition"],
        "pair_cells": pair_idx,
        "value_holdout": {"object_hue_index": vh_hue, "shape_index": vh_shape},
        "nway_corner": NWAY_ID,
        "nway_box_indices": box,
        "nway_rows_in_pool": int(corner_mask.sum()),
        "nway_min_count_over_all_15_pairwise_projections_of_the_box": proj,
        "slice_zero_cells_in_box": int(
            (slice_so[np.ix_(box["scale"], box["orientation"])] == 0).sum()
        ),
    }
    rec["grid_values"] = {n: grid.values[f].tolist() for f, n in enumerate(names)}
    np.savez(
        out / "task_assets.npz",
        render=img.astype(np.float32),
        read_back=tup,
        request_indices=req_idx,
        pair_pool=pair_pool,
        marg_fw=marg_fw,
        marg_so=marg_so,
        marg_os=marg_os,
        slice_so=slice_so,
    )
    (out / "ASSETS.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: rec[k] for k in ("render", "fit_pool")}, indent=1, default=str))


if __name__ == "__main__":
    main()
