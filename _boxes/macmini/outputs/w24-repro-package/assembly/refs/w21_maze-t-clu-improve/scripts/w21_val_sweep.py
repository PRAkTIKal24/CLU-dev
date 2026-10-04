"""w21 -- VAL sweep of CLU candidates on the cached frozen instrument (free, no rivals).

    python scripts/w21_val_sweep.py <cache_dir> <out_json> <cands_json>

cands_json: list of {"name":..., "K":2, "n_atoms":6, "ridge":1.0, "design_kw":{...}}.
Each candidate is fitted on the SAME train rows / codec latents as A1 and scored with the
w19 judge (W16.sc, 'exact') on every VAL row class.  3 seeds, always.
"""
import json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import chlu.exp_w15_cmaze_t as T
import chlu.exp_w15_cmaze_t_arms as A
import w16_cmaze_t as W16
from w21_clu import TransitionCluX

K_COLOURS = 4
A1 = {"mean": 0.725, "per_seed": [0.802, 0.569, 0.803]}
A1_EXACT = [0.8019, 0.5689, 0.8034]


def score(c, clu, rows=("nway", "pair", "seen", "value")):
    out = {}
    for nm in rows:
        rs = c["va_split"].get(nm, [])
        if not rs:
            continue
        Wd = (c["Wva"][nm], c["Dva"][nm])
        pred = T.decode_codes(K_COLOURS, A.codec_decode(c["codec"], clu.read(Wd)))
        out[nm] = W16.sc(c["va_insts"], rs, pred)
    return out


def main(cache_dir, out_json, cands):
    caches = []
    for s in range(3):
        with open(os.path.join(cache_dir, "val_seed%d.pkl" % s), "rb") as f:
            caches.append(pickle.load(f))
    grids = [np.arange(c, dtype=np.float64) for c in T.alphabets(K_COLOURS)]
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for cd in cands:
        t0 = time.time()
        per = []
        for s, c in enumerate(caches):
            clu = TransitionCluX(c["Ztr"], c["Wtr"], grids, dyn_train=c["Dtr"], K=cd.get("K", 2),
                                 n_atoms=cd.get("n_atoms", 6), ridge=cd.get("ridge", 1.0),
                                 design_kw=cd.get("design_kw", {}), seed=s)
            r = score(c, clu)
            r["read_floats"] = int(clu.ledger()["read_floats"])
            per.append(r)
        nway = [p["nway"] for p in per]
        mean = float(np.mean(nway))
        keep = bool(mean > A1["mean"] and all(n >= a for n, a in zip(nway, A1_EXACT)))
        rec = {"cand": cd, "per_seed": per, "nway": nway, "nway_mean": mean,
               "pair_mean": float(np.mean([p["pair"] for p in per])),
               "seen_mean": float(np.mean([p["seen"] for p in per])),
               "value_mean": float(np.mean([p["value"] for p in per])),
               "read_floats": per[0]["read_floats"], "gate_pass": keep,
               "seconds": time.time() - t0}
        res[cd["name"]] = rec
        print("%-34s nway %.4f (%s)  pair %.4f seen %.4f value %.4f  floats %6d  %s [%.0fs]"
              % (cd["name"], mean, " / ".join("%.4f" % x for x in nway), rec["pair_mean"],
                 rec["seen_mean"], rec["value_mean"], rec["read_floats"],
                 "GATE-PASS" if keep else "gate-fail", rec["seconds"]), flush=True)
        with open(out_json, "w") as f:
            json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])))
