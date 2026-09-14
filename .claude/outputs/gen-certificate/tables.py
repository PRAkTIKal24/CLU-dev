"""gen-certificate — render the report tables from the shipped manifests.
SCRATCH ANALYSIS SCRIPT (protocol: not production code; lives under .claude/outputs/).
⛔ Reads only; computes no new model quantity."""
import json, sys
from pathlib import Path
import numpy as np

BASE = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-certificate/runs/shapes3d_recon")
SEEDS = [int(s) for s in (sys.argv[1:] or ["0", "1", "2"])]

mans = {}
for s in SEEDS:
    p = BASE / f"seed_{s}" / "cert_manifest.json"
    if p.exists():
        m = json.load(open(p))
        if m.get("arms"):
            mans[s] = m
print(f"seeds with data: {sorted(mans)}")

def ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    return (v.mean(), v.std(), v.size) if v.size else (float("nan"), float("nan"), 0)

ARMS = ["clu", "clu_pair_reach", "reach", "pair", "curv", "energy",
        "distance", "distance_reach", "count"]
KEYS = [("A_vs_C", "A-vs-C"), ("B_vs_C", "B-vs-C"),
        ("AB_vs_C_balanced", "A∪B-vs-C bal"),
        ("B_vs_C[duplicate_factor]", "B-vs-Cdup"),
        ("B_vs_C[out_of_range]", "B-vs-Coor"),
        ("A7_vs_C[duplicate_factor]", "⭐A7-vs-Cdup"),
        ("pooled_UNBALANCED_do_not_quote", "pooled(⛔)")]

for g in ("correlation", "random"):
    have = [s for s in mans if g in mans[s]["arms"]]
    if not have:
        continue
    print(f"\n## grouping = {g}   (seeds {have})")
    hdr = "| arm | " + " | ".join(h for _, h in KEYS) + " | AURC(B∪C) | refuse A | refuse B | refuse C |"
    print(hdr)
    print("|" + "---|" * (len(KEYS) + 5))
    for a in ARMS:
        cells = []
        for k, _ in KEYS:
            m, sd, n = ms([mans[s]["arms"][g]["arms"][a].get(k) for s in have])
            cells.append(f"{m:.4f} ± {sd:.4f}" if n else "—")
        for k in ("aurc",):
            m, sd, n = ms([mans[s]["arms"][g]["arms"][a][k]["B_vs_C"] for s in have])
            cells.append(f"{m:.4f} ± {sd:.4f}")
        for k in ("refusal_rate_A", "refusal_rate_B", "refusal_rate_C"):
            m, sd, n = ms([mans[s]["arms"][g]["arms"][a]["operating_point"][k] for s in have])
            cells.append(f"{m:.3f} ± {sd:.3f}")
        print(f"| `{a}` | " + " | ".join(cells) + " |")
    # per-seed for the headline arms
    for a in ("clu", "clu_pair_reach", "distance", "distance_reach"):
        for k in ("B_vs_C", "B_vs_C[duplicate_factor]", "A7_vs_C[duplicate_factor]"):
            v = [mans[s]["arms"][g]["arms"][a].get(k) for s in have]
            if any(x is not None for x in v):
                print(f"   per-seed {a}.{k}: " +
                      " / ".join("—" if x is None else f"{x:.4f}" for x in v))
    # attribution + audit
    for s in have:
        att = mans[s]["arms"][g]["violated_factor_attribution"]
        aud = mans[s]["arms"][g]["no_invalid_supervision_audit"]
        print(f"   seed {s} attribution: " + ", ".join(
            f"{k}={v['correct']}/{v['n']} ({v['rate']:.3f}, chance {v['chance']:.3f})"
            for k, v in att.items()) +
            f" | audit {aud['n_fits_audited']} fits / {aud['n_ids_audited']} ids / "
            f"{aud['n_invalid_ids_found_in_any_fit']} C hits")
    # margins
    for lab, a1, a2, key in (
        ("P5  clu−distance_reach  (B-vs-Cdup)", "clu", "distance_reach", "B_vs_C[duplicate_factor]"),
        ("A2-P5 clu−distance_reach (A7-vs-Cdup)", "clu", "distance_reach", "A7_vs_C[duplicate_factor]"),
        ("A2-P5' pair−distance_reach (A7-vs-Cdup)", "pair", "distance_reach", "A7_vs_C[duplicate_factor]"),
        ("P4  clu−distance (B-vs-C)", "clu", "distance", "B_vs_C"),
    ):
        d = [mans[s]["arms"][g]["arms"][a1].get(key, float("nan"))
             - mans[s]["arms"][g]["arms"][a2].get(key, float("nan")) for s in have]
        m, sd, n = ms(d)
        if n:
            print(f"   ⭐ {lab}: {m:+.4f} ± {sd:.4f}  per-seed " +
                  " / ".join(f"{x:+.4f}" for x in d))

# grouping contrast
both = [s for s in mans if "correlation" in mans[s]["arms"] and "random" in mans[s]["arms"]]
for key in ("B_vs_C[duplicate_factor]", "A7_vs_C[duplicate_factor]", "B_vs_C"):
    d = [mans[s]["arms"]["correlation"]["arms"]["clu"].get(key, float("nan"))
         - mans[s]["arms"]["random"]["arms"]["clu"].get(key, float("nan")) for s in both]
    m, sd, n = ms(d)
    if n:
        print(f"\nP9 grouping contrast (corr−rand) clu.{key}: {m:+.4f} ± {sd:.4f}  "
              "per-seed " + " / ".join(f"{x:+.4f}" for x in d))

# component means, to show the direction each component moves
for g in ("correlation",):
    have = [s for s in mans if g in mans[s]["arms"]]
    if not have:
        continue
    cs = mans[have[0]]["arms"][g]["component_summary"]
    print(f"\n## component means, seed {have[0]}, {g}")
    print("| set | " + " | ".join(cs["A"].keys()) + " |")
    print("|" + "---|" * (len(cs["A"]) + 1))
    for nm in ("A", "A7", "B", "C"):
        if nm in cs:
            print(f"| {nm} | " + " | ".join(f"{cs[nm][c]['mean']:.4g}" for c in cs["A"]) + " |")
