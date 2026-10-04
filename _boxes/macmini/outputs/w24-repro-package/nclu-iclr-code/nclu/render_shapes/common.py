"""common."""

from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
from nclu.paths import run_dir, weights_dir

OUT = run_dir("figures", "render_shapes")
MATCHED_STAGE = run_dir("shapes3d")
MATCHED_EV = run_dir("shapes3d")
SHAPES_MATCHED = MATCHED_EV / "SHAPES-MATCHED.json"
WINNER_HASHES = weights_dir() / "weights.json"
CLU_PKL = {
    s: run_dir("shapes3d", "stores", "test", "seed%d" % s, "K32_n131072", "arm.pkl")
    for s in (0, 1, 2)
}
CLU = "n_clu"
TABLE = "table"
FAMILIES = ("mlp", "lstm", "transformer", "cont_mlp")
LABEL = {
    CLU: "n-CLU",
    "mlp": "MLP",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "cont_mlp": "Continuous MLP",
    TABLE: "Lookup table (floor)",
}
COLOUR = {
    CLU: "#2a78d6",
    "mlp": "#1baf7a",
    "lstm": "#eda100",
    "transformer": "#e87ba4",
    "cont_mlp": "#008300",
    TABLE: "#6b6a66",
}
ORDER = (CLU,) + FAMILIES + (TABLE,)
BASELINES = FAMILIES
SHAPE_NAMES = {0: "cube", 1: "cylinder", 2: "sphere", 3: "capsule"}


def carried(seed: int) -> dict:
    d = json.loads(SHAPES_MATCHED.read_text(encoding="utf-8"))
    return {f: d["rows34"]["P1_convergence"][f][str(seed)]["carried"] for f in FAMILIES}


def setenv():
    sys.path.insert(0, str(REPO.parent / "scripts"))


def load_columns(seed: int) -> tuple:
    """-> (objs {col: arm-object dict for exp_gen_read.arm_features}, meta)."""
    from nclu.eval.baselines import api as baseline_api

    stages = carried(seed)
    with open(CLU_PKL[seed], "rb") as fh:
        saved = pickle.load(fh)
    if int(saved["K"]) != 32 or int(saved["seed"]) != int(seed):
        raise SystemExit(
            f"{CLU_PKL[seed]}: K={saved['K']} seed={saved['seed']}, not K*=32 seed={seed}"
        )
    objs = {
        CLU: {
            "arm": CLU,
            "read": "tensor",
            "kind": "span",
            "store": saved["store"],
            "basis": saved["basis"],
            "address": saved["address"],
            "protos": None,
            "head_bytes": {"head_bytes": 0},
        }
    }
    ck = MATCHED_STAGE / "rivals" / "test" / "ckpt"
    meta = {"seed": seed, "carried": stages, "objects": {CLU: str(CLU_PKL[seed])}}
    for f in FAMILIES:
        name = f"{f}__{stages[f]}"
        rv = baseline_api.load_rival(seed, name, root=ck)
        if int(rv.meta["n_fit_rows"]) != 131072 or rv.meta.get("where") != "test":
            raise SystemExit(
                f"{name} seed {seed}: n_fit_rows={rv.meta['n_fit_rows']} where={rv.meta.get('where')} -- not the data-matched TEST object"
            )
        objs[f] = {
            "arm": name,
            "read": "rival",
            "rival": rv,
            "rival_partial": None,
            "store": None,
            "head_bytes": {"head_bytes": 0},
        }
        meta["objects"][f] = str(ck / f"seed_{seed}" / f"{name}.rival.eqx")
    tr = run_dir("shapes3d", "baseline_suite", "rivals")
    objs[TABLE] = {
        "arm": TABLE,
        "read": "rival",
        "rival": baseline_api.load_rival(seed, "table", root=tr),
        "rival_partial": None,
        "store": None,
        "head_bytes": {"head_bytes": 0},
    }
    meta["objects"][TABLE] = str(tr / f"seed_{seed}" / "table.rival.pkl")
    return (objs, meta)


def verify_manifest() -> dict:
    """The objects are built by this package's own commands; nothing staged is re-verified."""
    return {"n_files": 0, "all_match": None, "objects": "built by this package"}


def sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()
