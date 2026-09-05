"""PART B.3 -- band diagnostics + the REGISTERED exclusions (the INSECTS b=4 lesson,
applied BEFORE the map is used).  A band is excluded if:
  (a) best-strong-arm relative MAE improvement over persistence within the band < 5 %
  (b) band SD(y) < 0.25 x global SD(y)
  (c) fewer than 200 scored pairs
"""
import glob, json, os
import numpy as np

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
p = np.load(os.path.join(OUT, "pairs.npz"))
Y = p["y"].astype(np.float64); X = p["X"].astype(np.float64); N = len(Y)
PERS = X[:, 0]; SN168 = X[:, 25]

preds = {}
for f in glob.glob(os.path.join(OUT, "res", "*.npz")):
    tag = os.path.basename(f)[:-4]
    if tag.endswith("_shuf1"):
        continue
    preds[tag] = np.load(f)["pred"].astype(np.float64)
preds["persistence"] = PERS
preds["seasonal_naive_168"] = SN168

STRONG = [k for k in preds if k.split("_")[0] in ("gbdt", "gru", "mlp", "ridge", "rls")
          and "ff" not in k]
EXEMPLAR = [k for k in preds if k.startswith("knn")]


def mae(pr, m):
    return float(np.abs(pr[m] - Y[m]).mean())


def ent(v, nb=10):
    e = np.quantile(Y, np.linspace(0, 1, nb + 1)); e[0] -= 1e-9; e[-1] += 1e-9
    b = np.clip(np.digitize(v, e[1:-1]), 0, nb - 1)
    h = np.bincount(b, minlength=nb).astype(float); h = h[h > 0] / len(v)
    return float(-(h * np.log2(h)).sum())


gsd = float(Y.std())
out = {}
for mapf in sorted(glob.glob(os.path.join(OUT, "pair_regime_*.npy"))):
    mname = os.path.basename(mapf)[len("pair_regime_"):-4]
    lab = np.load(mapf)
    rows = []
    for c in sorted(set(lab.tolist())):
        m = lab == c
        n = int(m.sum())
        if c < 0:
            rows.append(dict(band="UNASSIGNED", n=n, excluded=True, reasons=["unassigned"]))
            continue
        bs = {k: mae(preds[k], m) for k in STRONG}
        be = {k: mae(preds[k], m) for k in EXEMPLAR}
        bstrong = min(bs, key=bs.get); bexe = min(be, key=be.get)
        pm = mae(PERS, m); sm = mae(SN168, m)
        imp = (pm - bs[bstrong]) / pm
        sd = float(Y[m].std())
        reasons = []
        if imp < 0.05: reasons.append("persistence-trivial (<5% strong-arm gain over persistence)")
        if sd < 0.25 * gsd: reasons.append("degenerate dispersion (SD<0.25 global)")
        if n < 200: reasons.append("n<200")
        rows.append(dict(band=int(c), n=n, sd_y=round(sd, 1), sd_ratio=round(sd / gsd, 3),
                         entropy_bits_10bin=round(ent(Y[m]), 3),
                         mae_persistence=round(pm, 2), mae_seasonal_naive_168=round(sm, 2),
                         best_strong=bstrong, mae_best_strong=round(bs[bstrong], 2),
                         best_exemplar=bexe, mae_best_exemplar=round(be[bexe], 2),
                         rel_gain_strong_over_persistence=round(imp, 4),
                         rel_margin_exemplar_vs_strong=round((be[bexe] - bs[bstrong]) / bs[bstrong], 4),
                         excluded=bool(reasons), reasons=reasons))
    out[mname] = rows

with open(os.path.join(OUT, "bands.json"), "w") as f:
    json.dump(dict(global_sd_y=gsd, strong_arms=sorted(STRONG), exemplar_arms=sorted(EXEMPLAR),
                   exclusion_rule=("(a) rel gain of best strong arm over persistence <5%; "
                                   "(b) band SD(y) < 0.25x global SD(y); (c) n<200 "
                                   "-- REGISTERED IN PREREG.md BEFORE THE MAP WAS BUILT"),
                   bands=out), f, indent=2)
for k, rows in out.items():
    print("=====", k)
    for r in rows:
        if r["band"] == "UNASSIGNED":
            print("  UNASSIGNED n=%d EXCLUDED" % r["n"]); continue
        print("  R%s n=%-6d sd=%6.0f H=%.2f  pers=%7.2f sn168=%7.2f  strong=%7.2f(%s)  "
              "exe=%7.2f(%s)  gain=%+.3f  margin=%+.4f  %s"
              % (r["band"], r["n"], r["sd_y"], r["entropy_bits_10bin"], r["mae_persistence"],
                 r["mae_seasonal_naive_168"], r["mae_best_strong"], r["best_strong"],
                 r["mae_best_exemplar"], r["best_exemplar"],
                 r["rel_gain_strong_over_persistence"], r["rel_margin_exemplar_vs_strong"],
                 "EXCLUDED:" + ";".join(r["reasons"]) if r["excluded"] else "kept"))
