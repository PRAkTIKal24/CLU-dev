"""Paths, the column order, and loading of the six columns' objects (seed-pinned).

Columns (w20 rows 3-4 TEST, carried fit per family at seed 0 from
SHAPES-MATCHED.json rows34.P1_convergence.<family>.<seed>.carried):
  n-CLU        K* = 32, 131,072 fit rows: w20 `store_K32_n131072` = the reused T7 object
               (its score file's extra.source: w12-single-design-crossover/scored/run1_seed0_K32/arm.pkl)
  MLP          mlp__tenx            (10x training steps)
  LSTM         lstm__declared       (declared fit)
  Transformer  transformer__declared(declared fit)
  Cont. MLP    cont_mlp__tenx       (10x training steps)
  Lookup table the floor (w20 table_FLOOR; spoke-a-rivals table.rival.pkl)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PW1_RS_OUT", "/Users/batmac/clu/outputs/pw1-render-shapes"))
STAGE = Path(os.environ.get("W17_RC_STAGE", "/Users/batmac/clu/stage"))
W20_STAGE = STAGE / "w20-shapes-data-matched"
W20_MANIFEST = W20_STAGE / "SHA256-w20-test-objects.txt"
EVIDENCE = Path("/Users/batmac/clu/evidence/outputs")
W20_EV = EVIDENCE / "w20-shapes-data-matched"
SHAPES_MATCHED = W20_EV / "SHAPES-MATCHED.json"
WINNER_HASHES = EVIDENCE / "gen-decoder-lift" / "winner" / "winner_hashes.json"

#: the n-CLU object per seed (w20 store_K32_n131072 reuses the T7 objects; integrity_T7_reproduced)
CLU_PKL = {
    0: STAGE / "outputs/w12-single-design-crossover/scored/run1_seed0_K32/arm.pkl",
    1: STAGE / "outputs/w12b-nr6-and-kstar/scored/run3_seed1_K32/arm.pkl",
    2: STAGE / "outputs/w12b-nr6-and-kstar/scored/run3_seed2_K32/arm.pkl",
}
CLU = "n_clu"
TABLE = "table"
FAMILIES = ("mlp", "lstm", "transformer", "cont_mlp")
LABEL = {CLU: "n-CLU", "mlp": "MLP", "lstm": "LSTM", "transformer": "Transformer",
         "cont_mlp": "Continuous MLP", TABLE: "Lookup table (floor)"}
#: the paper's one-name-one-colour set (w21_figures/make_figures.py ARMS / FLOOR)
COLOUR = {CLU: "#2a78d6", "mlp": "#1baf7a", "lstm": "#eda100", "transformer": "#e87ba4",
          "cont_mlp": "#008300", TABLE: "#6b6a66"}
ORDER = (CLU,) + FAMILIES + (TABLE,)
BASELINES = FAMILIES
SHAPE_NAMES = {0: "cube", 1: "cylinder", 2: "sphere", 3: "capsule"}


def carried(seed: int) -> dict:
    """family -> 'tenx' | 'declared', read from SHAPES-MATCHED.json (never assumed)."""
    d = json.loads(SHAPES_MATCHED.read_text(encoding="utf-8"))
    return {f: d["rows34"]["P1_convergence"][f][str(seed)]["carried"] for f in FAMILIES}


def score_file(seed: int, col: str, stages: dict) -> Path:
    d = W20_EV / "scores" / "rows34" / f"seed{seed}"
    if col == CLU:
        return d / "store_K32_n131072.json"
    if col == TABLE:
        return d / "table_FLOOR.json"
    return d / f"{col}__{stages[col]}@test.json"


def setenv():
    for k in ("CHLU_GEN_DECLIFT_DIR", "CHLU_GEN_DATA_ROOT", "CHLU_GEN_BRIDGE_ROOT",
              "CHLU_GEN_FEATURES_ROOT"):
        if k not in os.environ:
            raise SystemExit(f"{k} not set: source pw1_render_shapes/run_env.sh")
    sys.path.insert(0, str(REPO / "scripts"))


def load_columns(seed: int) -> tuple:
    """-> (objs {col: arm-object dict for exp_gen_read.arm_features}, meta)."""
    from chlu.eval.rivals import w4_api
    stages = carried(seed)
    with open(CLU_PKL[seed], "rb") as fh:
        saved = pickle.load(fh)
    if int(saved["K"]) != 32 or int(saved["seed"]) != int(seed):
        raise SystemExit(f"{CLU_PKL[seed]}: K={saved['K']} seed={saved['seed']}, not K*=32 seed={seed}")
    objs = {CLU: {"arm": CLU, "read": "tensor", "kind": "span", "store": saved["store"],
                  "basis": saved["basis"], "address": saved["address"], "protos": None,
                  "head_bytes": {"head_bytes": 0}}}
    ck = W20_STAGE / "rivals" / "test" / "ckpt"
    meta = {"seed": seed, "carried": stages, "objects": {CLU: str(CLU_PKL[seed])}}
    for f in FAMILIES:
        name = f"{f}__{stages[f]}"
        rv = w4_api.load_rival(seed, name, root=ck)
        if int(rv.meta["n_fit_rows"]) != 131072 or rv.meta.get("where") != "test":
            raise SystemExit(f"{name} seed {seed}: n_fit_rows={rv.meta['n_fit_rows']} "
                             f"where={rv.meta.get('where')} -- not the data-matched TEST object")
        objs[f] = {"arm": name, "read": "rival", "rival": rv, "rival_partial": None,
                   "store": None, "head_bytes": {"head_bytes": 0}}
        meta["objects"][f] = str(ck / f"seed_{seed}" / f"{name}.rival.eqx")
    tr = STAGE / "outputs" / "spoke-a-rivals" / "rivals"
    objs[TABLE] = {"arm": TABLE, "read": "rival", "rival": w4_api.load_rival(seed, "table", root=tr),
                   "rival_partial": None, "store": None, "head_bytes": {"head_bytes": 0}}
    meta["objects"][TABLE] = str(tr / f"seed_{seed}" / "table.rival.pkl")
    return objs, meta


def verify_manifest() -> dict:
    """shasum -c of every staged w20 TEST object against the Hub's manifest; refuse on any miss."""
    import hashlib
    bad, n = [], 0
    for line in W20_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, rel = line.split(None, 1)
        rel = rel.lstrip("*").strip()
        p = W20_STAGE / rel
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        n += 1
        if h.hexdigest() != want:
            bad.append(rel)
    if bad:
        raise SystemExit(f"OBJECT HASH MISMATCH on {len(bad)} of {n}: {bad} -- nothing is drawn")
    return {"manifest": str(W20_MANIFEST), "n_files": n, "all_match": True}


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()
