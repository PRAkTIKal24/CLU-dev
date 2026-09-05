"""Emit the markdown tables for the report."""
import collections
import json
from pathlib import Path

O = Path("/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire")


def load(fn):
    p = O / fn
    return [json.loads(l) for l in open(p)] if p.exists() else []


def t_ladder():
    rows = load("arms.jsonl") + load("arms_distfix.jsonl")
    lad = collections.defaultdict(lambda: -9)
    meta = {}
    for r in rows:
        sel = "kmeans" if "kmeans" in r["tag"] else "random"
        key = (r["window"], r["scaling"], r["L"], sel)
        if r["median_nse_447"] > lad[key]:
            lad[key] = r["median_nse_447"]
            meta[key] = r
    print("\n### CAMELS regional ladder (best over k x weight x target)\n")
    print("| window | scaling | L | selection | store bytes | in budget | "
          "sub | best median NSE (447) | best k / target |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(lad):
        r = meta[key]
        print(f"| {key[0]}d | {key[1]} | {key[2]:,} | {key[3]} | "
              f"{r['bytes']:,} | {'yes' if r['in_budget'] else '**NO**'} | "
              f"{r['sub']} | **{lad[key]:.4f}** | k={r['k']} {r['target']} |")


def t_local():
    rows = load("arms_local.jsonl")
    if not rows:
        return
    lad = collections.defaultdict(lambda: -9)
    meta = {}
    for r in rows:
        key = (r["window"], r["scaling"], r["L_per_basin"])
        if r["median_nse_447"] > lad[key]:
            lad[key] = r["median_nse_447"]
            meta[key] = r
    print("\n### CAMELS LOCAL (same-basin) arms — statics dropped\n")
    print("| window | scaling | L/basin | total bytes | x budget | in budget |"
          " sub | best median NSE (447) |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(lad):
        r = meta[key]
        print(f"| {key[0]}d | {key[1]} | {key[2]:,} | {r['bytes_total']:,} | "
              f"{r['budget_multiple']}x | "
              f"{'yes' if r['in_budget'] else '**NO**'} | {r['sub']} | "
              f"**{lad[key]:.4f}** |")


def t_comp():
    print("\n### CAMELS mandatory companion rows\n")
    print("| arm | bytes | in budget | in protocol | median NSE (447) | "
          "mean | n(NSE<=0) |")
    print("|---|---|---|---|---|---|---|")
    for r in load("companions.jsonl"):
        print(f"| {r['arm']} | {r['bytes']:,} | "
              f"{'yes' if r['in_budget'] else 'no'} | "
              f"{'yes' if r['in_protocol'] else '**NO — DIFFERENT TASK**'} | "
              f"**{r['median_nse_447']:.4f}** | {r['mean_nse_447']:.4f} | "
              f"{r['n_le0_447']} |")
    rg = load("arms_ridge.jsonl")
    if rg:
        print("\n### DECLARED POST-HOC (unregistered) classical arms\n")
        print("| arm | window | target | bytes | in budget | median NSE (447) |")
        print("|---|---|---|---|---|---|")
        for r in rg:
            print(f"| {r['arm']} | {r['window']}d | {r['target']} | "
                  f"{r['bytes']:,} | {'yes' if r['in_budget'] else '**NO**'} |"
                  f" **{r['median_nse_447']:.4f}** |")


def t_nc():
    rows = load("ncmapss_arms.jsonl")
    triv = [r for r in rows if r.get("arm") not in ("knn", "traj_similarity")]
    print("\n### N-CMAPSS DS02 — the criterion-2 rows (NOT PUBLISHED "
          "anywhere; supplied here)\n")
    print("| baseline | inputs | state bytes | RMSE [cycles] | s x 1e5 |")
    print("|---|---|---|---|---|")
    for r in triv:
        print(f"| {r['arm']} | {r['inputs']} | {r['bytes']} | "
              f"**{r['rmse']:.3f}** | {r['s_1e5']:.3f} |")
    st = [r for r in rows if r.get("arm") in ("knn", "traj_similarity")]
    lad = collections.defaultdict(lambda: 9e9)
    meta = {}
    for r in st:
        key = (r.get("feats", r.get("rep")), r.get("L", r.get("n_library")))
        if r["rmse"] < lad[key]:
            lad[key] = r["rmse"]
            meta[key] = r
    print("\n### N-CMAPSS DS02 — exemplar-store ladder (best over k)\n")
    print("| representation | L | dim | bytes | in budget | best RMSE | "
          "s x 1e5 | best k |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(lad, key=lambda x: (x[0], x[1])):
        r = meta[key]
        print(f"| {key[0]} ({r.get('scaling', r.get('metric'))}) | {key[1]:,} |"
              f" {r['dim']} | {r['bytes']:,} | "
              f"{'yes' if r['in_budget'] else '**NO**'} | "
              f"**{r['rmse']:.3f}** | {r['s_1e5']:.3f} | {r['k']} |")


if __name__ == "__main__":
    t_ladder()
    t_local()
    t_comp()
    t_nc()
