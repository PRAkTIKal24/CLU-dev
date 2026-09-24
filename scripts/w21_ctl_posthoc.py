"""w21 POST-HOC diagnostic (VAL cache, free): the four charter-C controls for a few CLU
candidates, built exactly as the runner builds them (CluFactory.make_control; across-wells a
DERANGEMENT).  ⛔ Run AFTER the config was frozen; it never changes the frozen config.

    python scripts/w21_ctl_posthoc.py <cache_dir> <out_json> <cfg.json> [<cfg.json> ...]
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import chlu.exp_w15_cmaze_t as T          # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A     # noqa: E402
import w16_cmaze_t as W16                 # noqa: E402
from w21_clu import CluFactory            # noqa: E402

K_COLOURS = 4


def main(cache_dir, out_json, cfg_paths):
    grids = [np.arange(c, dtype=np.float64) for c in T.alphabets(K_COLOURS)]
    caches = [pickle.load(open(os.path.join(cache_dir, "val_seed%d.pkl" % s), "rb")) for s in range(3)]
    res = {}
    for cp in cfg_paths:
        cfg = json.load(open(cp))
        fac = CluFactory(cfg)
        rec = {"cfg": cfg, "per_seed": []}
        for s, c in enumerate(caches):
            rs, W = c["va_split"]["nway"], c["Wva"]["nway"]
            dec = lambda o: T.decode_codes(K_COLOURS, A.codec_decode(c["codec"], o.read(W)))
            clu = fac(c["Ztr"], c["Wtr"], grids, seed=s)
            row = {"CLU": W16.sc(c["va_insts"], rs, dec(clu))}
            for ctl in A.TransitionClu.CONTROLS:
                o = fac.make_control(c["Ztr"], c["Wtr"], grids, 2, 6, 1.0, ctl, s)
                if ctl == "landscape_shuffled_across_wells":
                    assert not np.any(np.asarray(o.derangement) == np.arange(len(o.derangement)))
                    row["derangement"] = o.derangement
                row["control: " + ctl] = W16.sc(c["va_insts"], rs, dec(o))
            rec["per_seed"].append(row)
        keys = [k for k in rec["per_seed"][0] if k != "derangement"]
        rec["mean"] = {k: float(np.mean([r[k] for r in rec["per_seed"]])) for k in keys}
        rec["store_gate_pass"] = all(v <= rec["mean"]["CLU"] - 0.15
                                     for k, v in rec["mean"].items() if k.startswith("control"))
        res[cfg["name"]] = rec
        print("%-24s CLU %.4f | %s | gate %s" % (cfg["name"], rec["mean"]["CLU"], " ".join(
            "%s %.4f" % (k.split(": ")[1][:22], v) for k, v in rec["mean"].items() if k.startswith("control")),
            rec["store_gate_pass"]), flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
