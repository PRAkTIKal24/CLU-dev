"""Tables for deletion-waitlist-stiffness — every number in the report comes from here.

AUCs are DIRECTION-CALIBRATED per example (`max(AUC, 1-AUC)`) and then averaged, exactly
as in `mia-decay-measurement` / `placement-landing`; raw AUCs stay in the JSONs.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cal(vals):
    v = np.asarray(vals, float)
    v = np.maximum(v, 1.0 - v)
    return float(v.mean()), float(v.std())


def partA(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        for k, v in d.items():
            if k == "meta":
                continue
            arm, off, tid = k.split("|")
            r = {"arm": arm, "offers": int(off[3:]), "tid": int(tid[3:]),
                 "waitlist": v["waitlist"], "byte_eq": v["byte_equal_frac"][0],
                 "seated": v["target_seated_frac"],
                 "moves": v["moves_per_delete"], "reads": v["with_reads"]}
            for stat, s in v["columns"]["history"].items():
                m, sd = cal(s["auc_all"])
                r[stat] = (m, sd)
                r[stat + "_tpr1"] = float(np.mean(s["tpr@fpr0.01_all"]))
            for stat, s in v["columns"].get("paired", {}).items():
                r["paired_" + stat] = cal(s["auc_all"])[0]
            rows.append(r)
    rows.sort(key=lambda r: (r["tid"], r["arm"], r["offers"]))
    hdr = ["arm", "offers", "tid", "seated", "byte_eq", "n_live", "hole", "s4", "s5",
           "s1", "s2", "moves"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        cells = [r["arm"], str(r["offers"]), str(r["tid"]), f"{r['seated']:.3f}",
                 f"{r['byte_eq']:.4f}"]
        for s in ("n_live", "hole", "s4", "s5", "s1", "s2"):
            cells.append(f"{r[s][0]:.4f}±{r[s][1]:.4f}" if s in r else "—")
        cells.append(f"{r['moves'][0]:.2f}/{r['moves'][1]:.0f}")
        print("| " + " | ".join(cells) + " |")
    return rows


def r50(rs, rets):
    """Radius at which retention crosses 0.5 (linear interpolation, mia's convention)."""
    rs, rets = np.asarray(rs, float), np.asarray(rets, float)
    for i in range(len(rs) - 1):
        if rets[i] >= 0.5 > rets[i + 1]:
            f = (rets[i] - 0.5) / (rets[i] - rets[i + 1])
            return float(rs[i] + f * (rs[i + 1] - rs[i]))
    return float("nan")


def partB(paths):
    D = {}
    for p in paths:
        d = json.load(open(p))
        for panel in ("panelA", "panelB", "lengths"):
            D.setdefault(panel, {}).update(d.get(panel, {}))
        D["meta"] = d["meta"]
    arms, A_A, A_B, R = [], [], [], []
    for k in D["panelA"]:
        a, lvl = k.split("|")
        arms.append(a) if a not in arms else None
        A_A.append(float(lvl)) if float(lvl) not in A_A else None
    for k in D["panelB"]:
        a, Al, rl = k.split("|")
        A_B.append(float(Al[1:])) if float(Al[1:]) not in A_B else None
        R.append(float(rl[1:])) if float(rl[1:]) not in R else None
    A_A, A_B, R = sorted(A_A, reverse=True), sorted(A_B, reverse=True), sorted(R)

    if D["panelA"]:
        print("\n### Panel A — retention / value error / MIA vs amplitude\n")
        print("| A | " + " | ".join(f"{a} ret | {a} err | {a} s1auc" for a in arms) + " |")
        print("|" + "---|" * (1 + 3 * len(arms)))
        for A in A_A:
            cells = [f"{A:g}"]
            for a in arms:
                v = D["panelA"].get(f"{a}|{A:g}")
                cells += ["—", "—", "—"] if v is None else [
                    f"{v['retention'][0]:.4f}", f"{v['val_err'][0]:.4f}",
                    f"{cal(v['auc_s1_paired_all'])[0]:.4f}"]
            print("| " + " | ".join(cells) + " |")

        print("\n### §5 payload dependence — corr(retention, a_i^2) over the 24 examples\n")
        print("| arm | A | r(ret, a^2) | r(ret, |a|) | ret std | ret min | ret(|a|=1) |")
        print("|---|---|---|---|---|---|---|")
        for a in arms:
            for A in [x for x in A_A if x <= 0.07]:
                v = D["panelA"].get(f"{a}|{A:g}")
                if v is None:
                    continue
                ret = np.asarray(v["retention_all"]); a2 = np.asarray(v["a2_all"])
                ai = np.abs(np.asarray(v["a_i_all"]))
                rr = (float(np.corrcoef(ret, a2)[0, 1]) if ret.std() > 1e-12
                      else float("nan"))
                r1 = (float(np.corrcoef(ret, ai)[0, 1]) if ret.std() > 1e-12
                      else float("nan"))
                m1 = ret[np.isclose(ai, 1.0)]
                print(f"| {a} | {A:g} | {rr:.3f} | {r1:.3f} | {ret.std():.4f} | "
                      f"{ret.min():.4f} | {m1.mean() if len(m1) else float('nan'):.4f} |")

    if D["panelB"]:
        print("\n### Panel B — radius sweep and R50 (the decisive number)\n")
        print("| arm | A | " + " | ".join(f"r={r:g}" for r in R) + " | **R50** |")
        print("|" + "---|" * (3 + len(R)))
        for a in arms:
            for A in A_B:
                rets = [D["panelB"].get(f"{a}|A{A:g}|r{r:g}") for r in R]
                if any(x is None for x in rets):
                    continue
                y = [x["retention"][0] for x in rets]
                print(f"| {a} | {A:g} | " + " | ".join(f"{v:.3f}" for v in y)
                      + f" | **{r50(R, y):.3f}** |")

    if D["lengths"]:
        print("\n### Read-length law (B5)\n")
        print("| arm | read_steps | A | g0+A | val err | median err | retention |")
        print("|---|---|---|---|---|---|---|")
        for k, v in sorted(D["lengths"].items()):
            a, Al = k.split("|")
            A = float(Al[1:]); g0 = D["meta"]["arms"][a][1] if a in D["meta"]["arms"] else 0
            print(f"| {a} | {v['read_steps'][0]:.0f} | {A:g} | {g0 + A:.4f} | "
                  f"{v['val_err'][0]:.4f} | {v['val_err_med'][0]:.4f} | "
                  f"{v['retention'][0]:.4f} |")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    pats = sys.argv[2:] or (["waitlist_mia_*.json"] if which == "A" else ["gate_mia_*.json"])
    paths = sorted(sum([glob.glob(os.path.join(HERE, p)) for p in pats], []))
    paths = [p for p in paths if "quick" not in p]
    print("files:", [os.path.basename(p) for p in paths])
    (partA if which == "A" else partB)(paths)
