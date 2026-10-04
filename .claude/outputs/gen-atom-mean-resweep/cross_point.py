"""⭐ The statistic the PREREG gate does NOT have: cross-DRAW selection stability.

The interleaved-half test compares two halves of the SAME validation draw — same
rows, same `fit` set, same atoms, same prototypes. ⛔ It therefore measures
row-sampling noise WITHIN a draw and is blind to variation ACROSS draws. This
script asks the question the ladder actually needs answered: do two n_val points,
each of which passes the gate, AGREE with each other about the winner?
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "runs"
PTS = [("nval0512_tune0256", 512, 256), ("nval0512_tune0512", 512, 512),
       ("nval1024_tune1024", 1024, 1024), ("nval2048_tune2048", 2048, 2048),
       ("nval4096_tune4096", 4096, 4096), ("nval8192_tune8192", 8192, 8192)]
K = ("k", "metric", "normaliser", "scope")
SEEDS = (0, 1, 2)


def field(lab, which):
    d = {}
    for s in SEEDS:
        p = R / lab / f"seed_{s}" / "atom_manifest.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        for tag, b in sorted((m.get("budgets") or {}).items()):
            d[(s, tag)] = b["tuning"]["stability"][which]
    return d


def fmt(t):
    return "—" if t is None else f"k{t['k']}/{t['metric'][:3]}/{t['normaliser'][:4]}/{t['scope'][:3]}"


def main():
    sel = {lab: field(lab, "selected") for lab, _, _ in PTS}
    cells = sorted(sel["nval8192_tune8192"])

    print("## X1 ⭐ THE SELECTED WINNER AT EVERY n_val — does the ladder converge?\n")
    print("| cell | " + " | ".join(f"{nv}/{tn}" for _, nv, tn in PTS) + " | distinct |")
    print("|---|" + "---|" * (len(PTS) + 1))
    for c in cells:
        row = [sel[lab].get(c) for lab, _, _ in PTS]
        n = len({tuple(r[k] for k in K) for r in row if r})
        print(f"| s{c[0]}/{c[1]} | " + " | ".join(fmt(r) for r in row) + f" | **{n}** |")

    a, b = sel["nval4096_tune4096"], sel["nval8192_tune8192"]
    same = [c for c in cells if all(a[c][k] == b[c][k] for k in K)]
    print(f"\n## X2 ⛔ n_val = 4,096 vs n_val = 8,192 — two draws that BOTH pass the "
          f"gate — pick the SAME winner in only **{len(same)} of {len(cells)}** cells\n")
    print("| cell | n_val 4,096 | n_val 8,192 | |")
    print("|---|---|---|---|")
    for c in cells:
        ok = all(a[c][k] == b[c][k] for k in K)
        print(f"| s{c[0]}/{c[1]} | {fmt(a[c])} | {fmt(b[c])} | "
              f"{'✅ match' if ok else '⛔ **DIFFER**'} |")

    print("\n## X3 ⭐ PER-AXIS IDENTIFICATION — which parts of the tuple are real?\n")
    print("### (a) ACROSS n_val points\n")
    print("| axis | cells stable over ALL 6 points | over the 4 points n_val ≥ 1024 | "
          "verdict |")
    print("|---|---|---|---|")
    for ax in K:
        allp = sum(1 for c in cells if len({sel[l][c][ax] for l, _, _ in PTS}) == 1)
        hi = sum(1 for c in cells if len({sel[l][c][ax] for l, _, _ in PTS[2:]}) == 1)
        v = ("✅ **IDENTIFIED** (unanimous)" if allp == 6 else
             "◐ identified from n_val ≥ 1024" if hi == 6 else
             f"⛔ **NOT identified** ({hi}/6 stable even at n_val ≥ 1024)")
        print(f"| `{ax}` | {allp} / 6 | {hi} / 6 | {v} |")

    print("\n### (b) WITHIN a draw — the interleaved-half test, decomposed per axis\n")
    print("| n_val | `k` | `metric` | `normaliser` | `scope` | whole tuple |")
    print("|---|---|---|---|---|---|")
    for lab, nv, tn in PTS:
        s, h1, h2 = (field(lab, "selected"), field(lab, "argmax_ihalf1"),
                     field(lab, "argmax_ihalf2"))
        ag = field(lab, "ihalves_agree")
        row = [f"{sum(1 for c in cells if not (s[c][ax] == h1[c][ax] == h2[c][ax]))}/6"
               for ax in K]
        print(f"| {nv}/{tn} | " + " | ".join(row)
              + f" | **{sum(1 for c in cells if not ag[c])}/6** |")

    print("\n### (c) every value ever selected, over all 36 cell-points\n")
    for ax in K:
        v = {}
        for lab, _, _ in PTS:
            for c in cells:
                v[sel[lab][c][ax]] = v.get(sel[lab][c][ax], 0) + 1
        print(f"- `{ax}`: {dict(sorted(v.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
