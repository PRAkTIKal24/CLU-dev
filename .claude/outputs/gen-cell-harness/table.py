"""Regenerate every table in `.claude/outputs/gen-cell-harness.md` from the manifests."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "runs/shapes3d_recon"
agg = json.load(open(ROOT / "cell_results.json"))
mans = {}
for s in agg["seeds_present"]:
    mans[s] = json.load(open(ROOT / f"seed_{s}" / "cell_manifest.json"))

def ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "NOT RUN"

def per(v, f="{:.4f}"):
    return " / ".join("NOT RUN" if x is None else f.format(x) for x in v)

ORDER = ["clu", "store_removed", "knn_proto", "lace", "ebm_compose",
         "lace/ld_published", "lace/ld_corrected"]
arms = [a for a in ORDER if a in agg["arms"]] + [a for a in sorted(agg["arms"]) if a not in ORDER]
seeds = agg["seeds_present"]

print("### THE TABLE — in-corner `shape` balanced accuracy on IMAGES (external classifier)\n")
print("| arm | deployed bytes | tuning trials | IMAGE balacc (ext) | per seed | AUROC | TPR / TNR "
      "| distinct vectors of 72 | ⭐ distinct factor-tuples of 72 | probe companion |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in arms:
    r = agg["arms"][a]
    rows = [mans[s]["arms"][a] for s in seeds if a in mans[s]["arms"]]
    b = r["byte_ledger"]
    nb = b.get("deployed_bytes")
    nbs = f"{nb:,}" if nb else "—"
    if b.get("addressing_head_bytes"):
        nbs += f" + {b['addressing_head_bytes']:,} head"
    print(f"| `{a}` | {nbs} | {r['tuning']['trials_realised']} of "
          f"{r['tuning']['budget_available']} | **{ms([x['external']['in_corner']['balanced_accuracy'] for x in rows])}** "
          f"| {per([x['external']['in_corner']['balanced_accuracy'] for x in rows])} "
          f"| {ms([x['external']['in_corner']['auroc'] for x in rows])} "
          f"| {ms([x['external']['in_corner']['tpr'] for x in rows])} / {ms([x['external']['in_corner']['tnr'] for x in rows])} "
          f"| {ms([x['distinct']['B']['feature_vectors'] for x in rows])} "
          f"| **{ms([x['distinct']['B']['factor_tuples'] for x in rows])}** "
          f"| {ms([x['probe_companion']['in_corner']['balanced_accuracy'] for x in rows])} |")
c = agg["ceiling"]
print(f"| *decoder ceiling (real features)* | — | — | *{c['external_in_corner_balacc']['mean']:.4f} "
      f"± {c['external_in_corner_balacc']['std']:.4f}* | "
      f"{per(c['external_in_corner_balacc']['per_seed'])} | — | — | 72 | — | "
      f"*{c['probe_in_corner_balacc']['mean']:.4f} ± {c['probe_in_corner_balacc']['std']:.4f}* |")

print("\n### B-set composition, on IMAGES\n")
print("| arm | B aggregate (ext) | B `shape` (ext) | distinct factor-tuples on MS of ~199 |")
print("|---|---|---|---|")
for a in arms:
    r = agg["arms"][a]
    print(f"| `{a}` | {ms(r['external_B_aggregate']['per_seed'])} | "
          f"{ms(r['external_B_shape']['per_seed'])} | {ms(r['distinct_factor_tuples_MS']['per_seed'])} |")

print("\n### Instrument invariance — does swapping the reader re-order the arms?\n")
inv = agg["instrument_invariance"]
print("| arm | external | probe | Δ |")
print("|---|---|---|---|")
for a in arms:
    v = inv[a]
    print(f"| `{a}` | {v['external']:.4f} | {v['probe']:.4f} | {v['external_minus_probe']:+.4f} |")
print(f"\nranking by external: {inv['_ranking']['by_external']}")
print(f"ranking by probe:    {inv['_ranking']['by_probe']}")
print(f"identical: {inv['_ranking']['identical']}")

print("\n### Regression check vs the published probe numbers\n")
print("| arm | seed | measured (probe) | published | Δ | bit-identical | REGRESSED |")
print("|---|---|---|---|---|---|---|")
for s in seeds:
    rc = mans[s].get("rival_regression_check", {})
    for a in ("clu", "knn_proto", "lace"):
        v = rc.get(a)
        if not v or v.get("NOT_RUN"):
            print(f"| `{a}` | {s} | NOT RUN | | | | |"); continue
        print(f"| `{a}` | {s} | {v['measured']:.4f} | {v['published']:.4f} | {v['delta']:+.4f} "
              f"| {'✅' if v['bit_identical'] else '—'} | {'⛔ YES' if v['regressed'] else 'no'} |")

print("\n### Sampler-noise floor\n")
for s, v in agg["sampler_noise_floor"].items():
    for arm, d in (v or {}).items():
        if "NOT_RUN" in d:
            print(f"- seed {s} `{arm}`: NOT RUN — {d['NOT_RUN']}")
        else:
            print(f"- seed {s} `{arm}`: sd **{d['sd']:.4f}** over {d['repeats']} independent "
                  f"sampler keys; values {[round(x,4) for x in d['values']]}")

print("\n### NOT RUN\n")
for s, v in agg["NOT_RUN"].items():
    print(f"- seed {s}: {v if v else 'nothing — every requested arm completed'}")
print(f"- seeds requested {agg.get('seeds_requested')}, not run {agg.get('seeds_not_run')}")

print("\n### Hash discipline (⛔ asserted per arm per seed)\n")
for a in arms:
    r = agg["arms"][a]
    print(f"- `{a}`: decoder {[h[:12] for h in r['decoder_param_hashes']]} · "
          f"classifier {[h[:12] for h in r['external_classifier_param_hashes']]}")

# --- LACE best-of-family: the fairer never-weaken-a-competitor comparison ------
print("\n### LACE best-of-family (R9.4: *report LACE's best*), probe-vs-probe\n")
A7 = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-arms-a7-wiring/runs/historical_688128")
print("| seed | published best (sampler) | measured best (sampler) | Δ |")
print("|---|---|---|---|")
pub_b, mea_b = [], []
for s in seeds:
    pub = json.load(open(A7 / f"seed_{s}" / "baselines_manifest.json"))["arms"]["lace"]["samplers"]
    pub = {k: v["in_corner_discrimination"]["balanced_accuracy"] for k, v in pub.items()
           if not (k.endswith("_wexact") or k.endswith("_compsel"))}
    pb = max(pub, key=pub.get)
    mea = {}
    for key, name in (("lace", "ode"), ("lace/ld_published", "ld_published"),
                      ("lace/ld_corrected", "ld_corrected")):
        if key in mans[s]["arms"]:
            mea[name] = mans[s]["arms"][key]["probe_companion"]["in_corner"]["balanced_accuracy"]
    mb = max(mea, key=mea.get)
    pub_b.append(pub[pb]); mea_b.append(mea[mb])
    print(f"| {s} | {pub[pb]:.4f} (`{pb}`) | {mea[mb]:.4f} (`{mb}`) | {mea[mb]-pub[pb]:+.4f} |")
print(f"\n**best-of-family: published {np.mean(pub_b):.4f} ± {np.std(pub_b):.4f} · "
      f"measured {np.mean(mea_b):.4f} ± {np.std(mea_b):.4f} · "
      f"Δ {np.mean(mea_b)-np.mean(pub_b):+.4f}**")
