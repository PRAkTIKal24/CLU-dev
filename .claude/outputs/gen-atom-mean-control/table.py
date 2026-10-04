"""Regenerate every table in .claude/outputs/gen-atom-mean-control.md FROM THE MANIFESTS.

⛔ No constant below is typed by hand; run this after any re-run.
    .venv/bin/python .claude/outputs/gen-atom-mean-control/table.py
"""
import json, math, statistics as st
from pathlib import Path

R = Path(__file__).resolve().parent / "runs" / "shapes3d_recon"
agg = json.loads((R / "atom_results.json").read_text())
mans = [json.loads((R / f"seed_{s}" / "atom_manifest.json").read_text())
        for s in agg["seeds_present"]]
LAW = {64: 0.2783, 512: 0.2783 * (512 / 64) ** (-2 / 6.503)}


def ms(d):
    return f"{d['mean']:.4f} ± {d['std']:.4f}"


print("## 1. The control, at matched bytes (probe + external, 3 seeds)\n")
print("| budget | bytes | atoms/well | byte match | selected (k, metric, norm, scope) | "
      "channel bits | ext in-corner | probe in-corner | distinct ATOM-CONFIGS on B | "
      "distinct factor-tuples on B |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t in sorted(agg["budgets"]):
    r = agg["budgets"][t]
    sel = "; ".join(f"({s['k']},{s['metric'][:3]},{s['normaliser'][:4]},{s['scope']})"
                    for s in r["selected"])
    print(f"| **{t}** | {r['budget_bytes']:,} | {r['n_atoms_per_well']} | "
          f"**{r['byte_match_exact']}** | {sel} | {ms(r['channel_bits'])} | "
          f"**{ms(r['external_in_corner_balacc'])}** | {ms(r['probe_in_corner_balacc'])} | "
          f"**{ms(r['distinct_atom_configurations_B'])}** | "
          f"{ms(r['distinct_factor_tuples_B'])} |")
k = agg["knn_proto_regression"]
print(f"| *knn_proto (regression check, UNBOUNDED channel)* | *475,136* | — | — | — | *inf* | "
      f"*{st.mean([v['external_in_corner_balacc'] for v in k['per_seed'].values()]):.4f}* | "
      f"*{ms(k['probe_in_corner_balacc'])}* | — | "
      f"*{st.mean([v['distinct']['factor_tuples'] for v in k['per_seed'].values()]):.1f}* |")
c = agg["ceiling"]
print(f"| *decoder ceiling (real features)* | — | — | — | — | — | "
      f"*{ms(c['external_in_corner_balacc'])}* | *{ms(c['probe_in_corner_balacc'])}* | — | — |")

print("\n## 2. ⭐ Relative read error on `A` (rows EXCLUDED from the atom pool)\n")
print("| leg | budget A (N=64) | budget B (N=512) | fitted slope d ln ε / d ln N | implied PR |")
print("|---|---|---|---|---|")
for leg in ("centroid_control/k1", "oracle_feature_addressed/k1",
            "oracle_feature_addressed/k_selected", "request_addressed/k1",
            "request_addressed/k_selected"):
    a = agg["budgets"]["A"]["read_error"]["A"][leg]
    b = agg["budgets"]["B"]["read_error"]["A"][leg]
    p = agg["exponent_pooled"][leg]
    print(f"| `{leg}` | {ms(a)} | {ms(b)} | **{p['mean']:+.4f} ± {p['std']:.4f}** | "
          f"{p['implied_pr']['mean']:.2f} |")
print(f"| *centroid law, gen-dynamical-read PREREG §2 (RENDERED, not measured here)* | "
      f"*{LAW[64]:.4f}* | *{LAW[512]:.4f}* | *−0.3076* | *6.50* |")
print("\n| budget | medoid penalty (medoid k=1 / centroid k=1, same N, same Lloyd fit) | "
      "measured centroid ÷ the PREREG law |")
print("|---|---|---|")
for t, n in (("A", 64), ("B", 512)):
    mp = agg["budgets"][t]["medoid_penalty"]["A"]
    cc = agg["budgets"][t]["read_error"]["A"]["centroid_control/k1"]["mean"]
    print(f"| {t} | **×{mp['mean']:.3f} ± {mp['std']:.3f}** | ×{cc / LAW[n]:.3f} |")

print("\n## 3. ⭐ Six-factor retention ladder (probe), ⛔ every row with n_values_realised\n")
for rs in ("A", "B", "MS"):
    print(f"\n**rows = `{rs}`**\n")
    print("| factor | n_values_realised | budget A | budget B | knn_proto (unrestricted) |")
    print("|---|---|---|---|---|")
    lads = {t: agg["budgets"][t]["retention_ladder"][rs] for t in ("A", "B")}
    for nm in lads["A"]["per_factor"]:
        rowA, rowB = lads["A"]["per_factor"][nm], lads["B"]["per_factor"][nm]
        kp = st.mean([m["knn_proto_regression"]["retention_ladder"][rs]["per_factor"][nm]
                      ["accuracy"] for m in mans])
        na = ("**" + str(rowA["n_values_realised"]) + "** ⛔ NOT APPLICABLE"
              if not rowA["applicable"] else str(rowA["n_values_realised"]))
        print(f"| `{nm}` | {na} | {ms(rowA['accuracy'])} | {ms(rowB['accuracy'])} | {kp:.4f} |")
    kagg = [m["knn_proto_regression"]["retention_ladder"][rs]["aggregate"] for m in mans]
    print(f"| **aggregate** | — | **{ms(lads['A']['aggregate'])}** | "
          f"**{ms(lads['B']['aggregate'])}** | {st.mean(kagg):.4f} ± {st.pstdev(kagg):.4f} |")

print("\n## 4. ⚠ The declared WEAK companion (random subsample at the same bytes)\n")
print("| budget | k-medoid stronger on READ ERROR (per seed) | k-medoid stronger on "
      "in-corner (per seed) | subsample in-corner |")
print("|---|---|---|---|")
for t in sorted(agg["budgets"]):
    w = agg["budgets"][t]["weak_companion"]
    print(f"| {t} | **{w['kmedoid_stronger_on_read_error']}** | "
          f"{w['kmedoid_stronger_on_in_corner']} | {ms(w['subsample_in_corner'])} |")

print("\n## 5. ⚠ Row-set diagnostics — `val` is NOT exchangeable with `fit`\n")
print("| row set | n | feature variance | n_values_realised per factor |")
print("|---|---|---|---|")
d = mans[0]["row_set_diagnostics"]
for rsk in ("fit", "val", "A", "B", "MS"):
    v = d[rsk]
    nv = ", ".join(f"{a} {b}" for a, b in v["n_values_realised"].items())
    print(f"| `{rsk}` | {v['n']} | {v['feature_variance']:.4f} | {nv} |")

print("\n## 6. ⛔ knn_proto regression check (acceptance item 4)\n")
print("| seed | measured probe in-corner (full precision) | published | Δ | bit-identical |")
print("|---|---|---|---|---|")
for s, v in sorted(k["per_seed"].items()):
    print(f"| {s} | `{v['probe_in_corner_balacc']!r}` | `{v['published_probe']!r}` | "
          f"**{v['delta']:+.17g}** | **{v['bit_identical']}** |")

print("\n## 7. Selection stability (⚠ the `val` structure, measured)\n")
print("| seed | budget | selected | argmax on val half 1 | half 2 | halves agree | "
      "best−second gap | trial spread |")
print("|---|---|---|---|---|---|---|---|")
for m in mans:
    for t in sorted(m["budgets"]):
        s = m["budgets"][t]["tuning"]["stability"]
        f = lambda q: f"k={q['k']},{q['metric'][:3]},{q['normaliser'][:4]}"
        print(f"| {m['seed']} | {t} | {f(s['selected'])} | {f(s['argmax_half1'])} | "
              f"{f(s['argmax_half2'])} | **{s['halves_agree']}** | "
              f"{s['gap_best_to_second']:.4f} | {s['spread_over_trials']:.4f} |")

print("\n## 8. The persisted atom sets (⭐ gen-dynamical-read must LOAD these)\n")
print("| seed | budget | file | sha256 | atoms/well | scope | deployed bytes |")
print("|---|---|---|---|---|---|---|")
for m in mans:
    for t in sorted(m["budgets"]):
        p = m["budgets"][t]["atoms_persisted"]
        print(f"| {m['seed']} | {t} | `{Path(p['path']).name}` | `{p['sha256'][:24]}…` | "
              f"{p['n_atoms']} | {p['canonical_scope']} | {p['deployed_bytes']:,} |")
