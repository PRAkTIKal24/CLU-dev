"""w21 -- VAL diagnostic: per n-way held-out tuple, where does the CLU fail?"""
import os, pickle, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import chlu.exp_w15_cmaze_t as T
import chlu.exp_w15_cmaze_t_arms as A
from w21_clu import TransitionCluX
K = 4
grids = [np.arange(c, dtype=np.float64) for c in T.alphabets(K)]
for s in range(3):
    c = pickle.load(open(os.path.join(sys.argv[1], "val_seed%d.pkl" % s), "rb"))
    import json
    cfg = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    clu = TransitionCluX(c["Ztr"], c["Wtr"], grids, dyn_train=c["Dtr"], seed=s, K=cfg.get("K", 2),
                         n_atoms=cfg.get("n_atoms", 6), ridge=cfg.get("ridge", 1.0),
                         design_kw=cfg.get("design_kw", {}))
    rs = c["va_split"]["nway"]
    pred = T.decode_codes(K, A.codec_decode(c["codec"], clu.read((c["Wva"]["nway"], c["Dva"]["nway"]))))
    true = np.array([(r.legal, r.disp, r.ev, r.pocket_next) for r in rs])
    Wtr = c["Wtr"].astype(int)
    agg = collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()])
    for r, p, t in zip(rs, pred, true):
        w = r.words; key = (w[0], w[2], w[3])
        a = agg[key]; a[0] += 1; a[1] += int((np.asarray(p) == t).all())
        if not (np.asarray(p) == t).all():
            a[2][(tuple(int(x) for x in t), tuple(int(x) for x in p))] += 1
        a[3][w[4]] += 1
    print("== seed", s, "held-out nway", c["ho"].nway)
    for key, (n, ok, errs, dirs) in sorted(agg.items()):
        ntr = int(((Wtr[:, 0] == key[0]) & (Wtr[:, 2] == key[1]) & (Wtr[:, 3] == key[2])).sum())
        print("  (pocket,door,tint)=%s n=%4d acc=%.3f train_triple_rows=%5d dirs=%s top_err(true->pred)=%s"
              % (key, n, ok / n, ntr, dict(dirs), errs.most_common(2)))
