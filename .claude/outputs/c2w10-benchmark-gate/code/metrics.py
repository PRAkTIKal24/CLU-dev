"""Metrics from saved per-instance predictions. Zliobaite et al. 2015, Eqs. 13-14."""
import numpy as np, json, glob, os


def load(tag):
    z = np.load(f"preds/{tag}.npz")
    return z["pred"], z["y"]


def all_metrics(pred, y, window=1000, start=1):
    """start=1: index 0 is excluded for EVERY arm (No-Change's first prediction is undefined)."""
    p_, y_ = pred[start:], y[start:]
    corr = (p_ == y_).astype(np.float64)
    acc = corr.mean()
    # sliding-window prequential curve (Souza convention, width 1000)
    c = np.concatenate([[0.0], np.cumsum(corr)])
    idx = np.arange(window, len(corr) + 1)
    curve = (c[idx] - c[idx - window]) / window
    # kappa: chance agreement from the marginals of predictions and truth
    labs = np.unique(y_)
    p0 = sum((p_ == l).mean() * (y_ == l).mean() for l in labs)
    kappa = (acc - p0) / (1 - p0)
    # kappa_per: persistent classifier on the SAME index range
    p_per = float((y[start:] == y[start - 1:-1]).mean())
    kappa_per = (acc - p_per) / (1 - p_per)
    kplus = float(np.sqrt(max(0.0, kappa) * max(0.0, kappa_per)))
    return dict(acc=float(acc) * 100, acc_window_mean=float(curve.mean()) * 100,
                acc_window_final=float(curve[-1]) * 100, kappa=float(kappa),
                kappa_per=float(kappa_per), kappa_plus=kplus, p_chance=float(p0),
                p_per=p_per, n_scored=int(len(corr)), curve=curve)


def band_acc(pred, y, bands, start=1):
    out = []
    for b in bands:
        a, e = max(b["start"], start), b["end"]
        out.append(float((pred[a:e] == y[a:e]).mean()) * 100)
    return out


if __name__ == "__main__":
    S = json.load(open("structure.json"))
    rows = {}
    for f in sorted(glob.glob("preds/*.json")):
        tag = os.path.basename(f)[:-5]
        meta = json.load(open(f))
        pred, y = load(tag)
        m = all_metrics(pred, y)
        curve = m.pop("curve")
        m["bands"] = band_acc(pred, y, S["bands"])
        rows[tag] = {**meta, **m}
        np.save(f"preds/{tag}_curve.npy", curve)
    json.dump(rows, open("metrics.json", "w"), indent=1)
    hdr = f"{'arm':22s} {'acc':>7s} {'win-mean':>9s} {'kappa':>7s} {'k_per':>7s} {'k+':>7s} {'wall_s':>7s}"
    print(hdr); print("-" * len(hdr))
    for t, r in sorted(rows.items(), key=lambda kv: -kv[1]["acc"]):
        print(f"{t:22s} {r['acc']:7.2f} {r['acc_window_mean']:9.2f} {r['kappa']:7.4f} "
              f"{r['kappa_per']:7.4f} {r['kappa_plus']:7.4f} {r['wall_s']:7.0f}")
