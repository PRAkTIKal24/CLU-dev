"""Frontier data loader: re-aggregates .claude/scratch/regime-remap-2000ep/runs/*.json
exactly as analyze.py does, restricted to the ne3 (episodes=3) frontier cells,
seeds {42,43,44}.  No number is recomputed, smoothed or invented."""
import json, glob, os
import numpy as np
from collections import defaultdict

RUNS = "/Users/user/Desktop/CHLU/.claude/scratch/regime-remap-2000ep/runs"
EPOCHS = [500, 1000, 2000, 4000]
FRONTIER = [(128, 32, 256), (256, 64, 256), (384, 96, 256), (384, 128, 512)]

def load():
    recs = [json.load(open(p)) for p in glob.glob(os.path.join(RUNS, "*.json"))]
    G = defaultdict(list)
    for r in recs:
        G[(r["axis"], r["N"], r["kv"], r["stress"], r["epochs"],
           r.get("vocab", 256), r["episodes"])].append(r)
    out = {}
    hops = []
    for (N, kv, v) in FRONTIER:
        fid, fide, gate, gatee = [], [], [], []
        for ep in EPOCHS:
            rs = G[("correlation", N, kv, 0.0, ep, v, 3)]
            assert len(rs) == 3, (N, kv, ep, len(rs))
            f = np.array([x["fidelity"] for x in rs]); g = np.array([x["clu_gate_acc"] for x in rs])
            fid.append(f.mean()); fide.append(f.std())
            gate.append(g.mean()); gatee.append(g.std())
            hops.append(float(np.mean([x["hop_acc"] for x in rs])))
        out[kv] = dict(fid=np.array(fid), fid_err=np.array(fide),
                       gate=np.array(gate), gate_err=np.array(gatee))
    return out, (min(hops), max(hops))
