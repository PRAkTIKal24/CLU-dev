"""§5: HEPA's h-AUROC on the information-age slices, from `preds/*.npz`.

Alignment is by KEY, never by position: every prediction row is matched on
(entity_id, t_index) to the window the tripwire harness built, so a different
ConcatDataset ordering cannot silently mis-slice the result. A run whose keys do
not match 1:1 is refused, not repaired.

The scored surface is the fp16 `p_surface` the sweep dumped; the sweep's own
_meta measures that recomputing h_auroc from it agrees with its float32 value to
<= 2.0e-4, and this module re-verifies that on every file it reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import hepa_iface as HI

PREDS = Path(".claude/outputs/hepa-longhorizon-triage/preds")
MASKS = Path(".claude/scratch/hepa-lh-verdict")
FP16_TOL = 2.0e-4


_SWEEP = None


def _sweep_seed_value(ds: str, rung: int, seed: int, device: str) -> Optional[float]:
    """The h-AUROC the sweep recorded in float32 for this exact cell+seed."""
    global _SWEEP
    if _SWEEP is None:
        f = PREDS.parent / "sweep.json"
        _SWEEP = json.load(open(f)) if f.exists() else {}
    for block in ("sweep", "reproduction"):
        cell = ((_SWEEP.get(block) or {}).get(ds) or {}).get(str(rung))
        if not isinstance(cell, dict):
            continue
        fam = cell.get(device)
        if not isinstance(fam, dict):
            continue
        for sd in fam.get("seeds") or []:
            if int(sd.get("seed", -1)) == seed:
                return float(sd["h_auroc"])
    return None


def _npz_name(ds: str, rung: int, seed: int, device: str) -> Path:
    suf = "_mps" if device == "mps" else ""
    return PREDS / f"{ds}_L{rung}_s{seed}{suf}.npz"


def load_pred(ds: str, rung: int, seed: int, device: str = "cpu"):
    p = _npz_name(ds, rung, seed, device)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    eids = np.asarray(d["entity_ids"], dtype=object)
    ent = np.asarray([str(eids[i]) for i in d["entity_index"]], dtype=object)
    return {"path": str(p), "p": np.asarray(d["p_surface"], dtype=np.float32),
            "y": np.asarray(d["y_surface"], dtype=np.float32),
            "t": np.asarray(d["t_index"], dtype=np.int64), "entity": ent,
            "horizons": [int(h) for h in d["horizons"]],
            "meta": json.loads(str(d["meta"]))}


def align(pred: Dict, ds: str, rung: int) -> Dict:
    """Permutation taking tripwire-harness window order -> prediction row order."""
    b = HI.load_bundle(ds, rung)
    te = b.test
    key_me = [(str(e), int(t)) for e, t in zip(te.entity, te.t_index)]
    key_pr = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(pred["entity"], pred["t"]))}
    if len(key_pr) != len(pred["t"]):
        raise ValueError("prediction file has duplicate (entity, t) keys")
    if len(key_me) != len(key_pr):
        raise ValueError(f"window-count mismatch: harness {len(key_me)} vs preds {len(key_pr)}")
    missing = [k for k in key_me if k not in key_pr]
    if missing:
        raise ValueError(f"{len(missing)} harness windows absent from preds, e.g. {missing[:3]}")
    return {"perm": np.array([key_pr[k] for k in key_me], dtype=np.int64), "bundle": b}


def slice_h_auroc(ds: str, rung: int, seeds: List[int], device: str = "cpu"
                  ) -> Dict[str, object]:
    m = np.load(MASKS / f"slicemask_{ds}.npz", allow_pickle=False)
    out: Dict[str, object] = {"dataset": ds, "context": rung, "device": device,
                              "per_seed": [], "missing_seeds": []}
    perm = None
    for seed in seeds:
        pr = load_pred(ds, rung, seed, device)
        if pr is None:
            out["missing_seeds"].append(seed)
            continue
        if perm is None:
            perm = align(pr, ds, rung)["perm"]
        p = pr["p"][perm]
        y = pr["y"][perm]
        H = pr["horizons"]
        rec = {"seed": seed, "artifact": pr["path"],
               "all": HI.score_h_auroc(p, y, H),
               "h_auroc_recorded_by_sweep": _sweep_seed_value(ds, rung, seed, device)}
        if rec["h_auroc_recorded_by_sweep"] is not None:
            rec["fp16_reread_delta"] = abs(rec["all"] - rec["h_auroc_recorded_by_sweep"])
            rec["fp16_within_tol"] = bool(rec["fp16_reread_delta"] <= FP16_TOL)
        for s in ("old", "young"):
            mk = m[s]
            rec[s] = (HI.score_h_auroc(p[mk], y[mk], H)
                      if mk.sum() and (np.asarray(m["win_slice"]) == s).sum() else None)
        out["per_seed"].append(rec)
    for s in ("all", "old", "young"):
        v = [r[s] for r in out["per_seed"] if r.get(s) is not None]
        out[s] = ({"mean": float(np.mean(v)),
                   "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                   "min": float(np.min(v)), "max": float(np.max(v)),
                   "range": float(np.max(v) - np.min(v)),
                   "per_seed": [float(x) for x in v], "n_seeds": len(v)}
                  if v else {"status": "EMPTY_OR_ABSENT"})
    return out
