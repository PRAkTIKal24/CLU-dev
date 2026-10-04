"""w4 hub/unseen-valid — shared plumbing.  Built fresh on the pin + base/iter4.

⛔ Wave-3 artefacts in ../../code are NOT inherited (see LOG.md, wave-4 header).
Everything here rides the FROZEN instrument: the pinned decoder, the hash-asserted
external classifier, the cached bridge features, and the value-holdout condition.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import chlu.experiments.exp_gen_factor_wells as W

OUT = Path("C:/chlu/outputs/hub-unseen-valid/w4")
HUE = "object_hue"

# --------------------------------------------------------------------------
# the corner tiling for row 2 (declared here, used by every number this hub makes)
# --------------------------------------------------------------------------
#: A row-2 corner is (bracket pair) x (floor_hue band, width 2) x (scale half, width 4).
#: 7 x 5 x 2 = 70 possible corners, a DISJOINT and EXHAUSTIVE tiling: every row-2
#: request lands in exactly one.  >= 49 as w4-hub-common section 3 requires.
FLOOR_BAND_W = 2
SCALE_HALF_W = 4


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=W._jsonable)


def load_grid_and_split():
    grid, z = W.build_grid()
    return grid, np.asarray(z["train_rows"], np.int64), np.asarray(z["test_rows"], np.int64)


def build_row2(n_requests: int, seed: int = 0):
    """Regenerate row 2 at ``n_requests`` with the PINNED builder (no edits to it)."""
    import dataclasses as dc

    from chlu.data.gen_splits import GenSplit
    from chlu.data.gen_value_holdout import ValueHoldoutConfig, build_row2_unseen

    grid, train_rows, test_rows = load_grid_and_split()
    cells = json.load(open(W.data_root() / "value_holdout" / "value_cells.json",
                           encoding="utf-8"))
    base = ValueHoldoutConfig(**{k: v for k, v in cells["config"].items()
                                 if k in {f.name for f in dc.fields(ValueHoldoutConfig)}})
    cfg = dc.replace(base, row2_n_requests=int(n_requests), row2_seed=int(seed))
    split = GenSplit(dataset=grid.dataset, condition="value_holdout",
                     variant=str(cells.get("variant", "value_holdout")), grid=grid,
                     train_rows=train_rows, test_rows=test_rows,
                     test_predicate=lambda v: np.zeros(len(v), bool))
    return build_row2_unseen(grid, split, cfg), grid, train_rows


def parse_brackets(rec) -> tuple[float, float]:
    """(lo, hi) stored hue values, read off the builder's own reason string."""
    s = rec["reason"].split("stored values ")[1].split(";")[0]
    lo, hi = s.split(" and ")
    return float(lo), float(hi)


def request_values(records, grid) -> np.ndarray:
    vals = np.zeros((len(records), grid.n_factors), np.float64)
    for i, rec in enumerate(records):
        for t in rec["request"]:
            vals[i, int(t["factor_index"])] = float(t["value"])
    return vals


def bracket_rows(records, grid, train_rows):
    """Per request: the two STORED rows that bracket it, and its bracket hue values.

    These are the ORACLE anchors: real dataset rows whose cached features exist.
    """
    k_hue = grid.index_of(HUE)
    hue_vals = grid.values[k_hue]
    vals = request_values(records, grid)
    n = len(records)
    rows = np.zeros((n, 2), np.int64)
    hues = np.zeros((n, 2), np.float64)
    in_train = np.zeros(grid.n_rows, bool)
    in_train[train_rows] = True
    for i, rec in enumerate(records):
        lo, hi = parse_brackets(rec)
        hues[i] = (lo, hi)
        fi = grid.factor_indices_of(np.array([  # the request's own grid cell, hue aside
            int(grid.rows_of(np.array([[
                int(np.argmin(np.abs(grid.values[f] - vals[i, f])))
                if grid.names[f] != HUE else 0
                for f in range(grid.n_factors)]]))[0])]))[0]
        for j, hv in enumerate((lo, hi)):
            fi_b = fi.copy()
            fi_b[k_hue] = int(np.argmin(np.abs(hue_vals - hv)))
            rows[i, j] = int(grid.rows_of(fi_b[None, :])[0])
    assert in_train[rows].all(), "an oracle anchor row is NOT in the store's fit pool"
    return rows, hues, vals


def corner_ids(records, grid, vals, hues) -> np.ndarray:
    """The declared disjoint tiling: (bracket pair, floor band, scale half)."""
    k_f = grid.index_of("floor_hue")
    k_s = grid.index_of("scale")
    fi_f = np.abs(vals[:, k_f][:, None] - grid.values[k_f][None, :]).argmin(1)
    fi_s = np.abs(vals[:, k_s][:, None] - grid.values[k_s][None, :]).argmin(1)
    pair = np.round(hues[:, 0] / 0.1).astype(int)          # lower bracket grid index
    return np.array([f"p{p}-f{f // FLOOR_BAND_W}-s{s // SCALE_HALF_W}"
                     for p, f, s in zip(pair, fi_f, fi_s)])


# --------------------------------------------------------------------------
# the SECOND JUDGE — the oracle-referenced readout
# --------------------------------------------------------------------------
class SecondJudge:
    """Continuous hue readout, referenced to the two ORACLE anchors of each request.

    Judge 1 (the pinned one) is top-1 argmax over 10 hue classes at spacing 0.1; a
    request at 0.05 has no correct class, so a perfect render and a perfect snap are
    indistinguishable (w4-hub-common section 4.5).  This judge keeps the softmax the
    argmax throws away and turns it into a continuous hue estimate, then expresses
    that estimate in the coordinate the two bracketing STORED values define.
    """

    def __init__(self, seed: int, log=print):
        self.seed = int(seed)
        self.judge = W.Judge(int(seed), log=log)
        grid, _, _ = load_grid_and_split()
        self.k_hue = grid.index_of(HUE)
        self.hue_vals = np.asarray(grid.values[self.k_hue], np.float64)

    def readout(self, feats: np.ndarray, batch: int = 256) -> np.ndarray:
        """Posterior-mean hue of the decoded render.  (N,) float64, in hue units."""
        from chlu.eval.gen_probe import decode_features, factor_probs
        Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(W.PINS["feat_shape"]))
        out = np.empty(Z.shape[0], np.float64)
        for s in range(0, Z.shape[0], batch):
            e = min(s + batch, Z.shape[0])
            imgs = decode_features(self.judge.decoder, Z[s:e], batch)
            p = factor_probs(self.judge.external, imgs, self.k_hue, batch)
            out[s:e] = p @ self.hue_vals
        return out

    def anchors(self, rows: np.ndarray, batch: int = 256) -> np.ndarray:
        """Oracle anchors: the readout of the REAL cached features of ``rows``."""
        F = W.features_mmap(self.seed)
        flat = np.asarray(rows, np.int64).reshape(-1)
        r = self.readout(W.gather(F, flat), batch)
        return r.reshape(np.asarray(rows).shape)


def referenced(readout: np.ndarray, a_lo: np.ndarray, a_hi: np.ndarray,
               h_lo: np.ndarray, h_hi: np.ndarray) -> dict:
    """Express a raw readout in the ORACLE-REFERENCED coordinate.

    ``p`` is the position in the anchor span (0 = the low stored value, 1 = the high
    one); ``hue_hat`` maps that back to hue units.  ⛔ The snap is measured HERE, not
    on the raw readout, or the measurement is of the judge and not of the arm.
    """
    span = a_hi - a_lo
    p = (readout - a_lo) / span
    hue_hat = h_lo + p * (h_hi - h_lo)
    return {"p": p, "hue_hat": hue_hat, "span_readout": span}


# --------------------------------------------------------------------------
# cache generation 2 (w4-hub-common section 2: EVERYTHING scored runs on it)
# --------------------------------------------------------------------------
GEN2_ROOT = Path("C:/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon")


def gen2_ready(seed: int) -> bool:
    p = GEN2_ROOT / f"seed_{int(seed)}" / "feature_cache.json"
    if not p.exists():
        return False
    d = json.load(open(p, encoding="utf-8"))
    return int(d.get("generation", 0)) == 2 and int(d.get("n_rows", 0)) == 480_000


def use_cache_generation2(seeds=(0, 1, 2)) -> dict:
    """Point the feature cache at GENERATION 2, and PROVE the encoder still matches.

    ⛔ Only ``features_mmap`` is redirected.  ``classifier_dir`` and the decoder come
    from their own pinned roots and are untouched, so the judge is the same judge.
    The encoder hash of each gen-2 cache is asserted equal to the hash the pinned
    decoder was lifted against -- otherwise the decoder would be reading a cache its
    encoder never produced, and every number would be silently wrong.
    """
    wh = json.load(open(Path(os.environ["CHLU_GEN_DECLIFT_DIR"]) / "winner_hashes.json",
                        encoding="utf-8"))
    proof = {}
    for s in seeds:
        if not gen2_ready(s):
            raise AssertionError(f"cache generation 2 is not ready for seed {s}")
        d = json.load(open(GEN2_ROOT / f"seed_{s}" / "feature_cache.json", encoding="utf-8"))
        got = str(d["encoder_param_hash"])
        exp = str(wh[f"seed_{s}"]["encoder_param_hash_it_was_fit_against"])
        if got[:len(exp)] != exp[:len(got)] and got != exp:
            raise AssertionError(
                f"seed {s}: gen-2 cache encoder {got[:16]} != the hash the pinned "
                f"decoder was fit against ({exp[:16]})")
        proof[f"seed_{s}"] = {"encoder_param_hash": got, "n_rows": int(d["n_rows"]),
                              "generation": int(d["generation"]), "matches_decoder": True}

    def _mmap(seed: int):
        return np.load(GEN2_ROOT / f"seed_{int(seed)}" / "features.npy", mmap_mode="r")

    W.features_mmap = _mmap
    return {"cache_generation": 2, "root": str(GEN2_ROOT), "encoder_proof": proof}
