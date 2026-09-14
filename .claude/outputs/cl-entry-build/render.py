"""Render the cl-entry metrics JSON into the report's markdown tables."""
import json
import sys

path = sys.argv[1]
d = json.load(open(path))
cfg = d["config"]
print(f"### {path}  seeds={d['seeds']}  dataset={cfg['dataset']}  phi_dim={cfg['phi_dim']} "
      f"memory_items={cfg['memory_items']}\n")

print("| method | class | ACC | sd | BWT | forgetting | mem items | mem floats | seeds |")
print("|---|---|---|---|---|---|---|---|---|")
for r in d["baseline_table"]:
    print(f"| {r['method']} | {r['class']} | **{r['ACC']:.3f}** | ±{r['ACC_sd']:.3f} | "
          f"{r['BWT']:+.3f} | {r['forgetting']:.3f} | {r['memory_items']} | "
          f"{r['memory_floats']} | {r['n_seeds']} |")

print("\n**verdict**")
for k, v in d["verdict"].items():
    if isinstance(v, float):
        v = round(v, 4)
    print(f"- `{k}`: {v}")

if d.get("baseline_tuning"):
    print("\n**N78 tuning (seed 0 grid → value used for all seeds)**\n")
    print("| method | hyper | grid | ACC per value | chosen |")
    print("|---|---|---|---|---|")
    for t in d["baseline_tuning"]:
        accs = ", ".join(f"{a:.3f}" for a in t["ACC_per_value"])
        print(f"| {t['method']} | {t['hyper']} | {t['grid']} | {accs} | **{t['chosen']}** |")

print("\n**controller / geometry per seed (task1_only)**\n")
print("| seed | task | offered | admitted | refused(spacing) | adm.frac | s | d_safe | live |")
print("|---|---|---|---|---|---|---|---|---|")
for run in d["entry_runs"]:
    if run["regime"] != "task1_only" or run.get("decay_on"):
        continue
    for p in run["per_task"]:
        print(f"| {run['seed']} | {p['task']} | {p['offered']} | {p['admitted']} | "
              f"{p['refused_spacing']} | {p['admitted_fraction']:.3f} | "
              f"{p['well_width_s']:.3f} | {p['d_safe']:.3f} | {p['n_live']} |")
for run in d["entry_runs"]:
    if run["regime"] == "task1_only" and not run.get("decay_on"):
        print(f"\nseed {run['seed']} geometry: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in run['geometry'].items() if k != 'packing_slack_note'})}")

print("\n**per-task final accuracies A[T,i] (task1_only, CLU / kNN-same-keys)**\n")
for run in d["entry_runs"]:
    if run["regime"] != "task1_only" or run.get("decay_on"):
        continue
    print(f"- seed {run['seed']}: clu {[round(x,3) for x in run['metrics_clu']['final_per_task']]} "
          f"| knn {[round(x,3) for x in run['metrics_knn_same_keys']['final_per_task']]}")

for rt in d.get("retry_native", []):
    for c in rt.get("cells", []):
        print(f"\n**retry — {c['label']}, mask_p={c['mask_p']}, n_items={c['n_items']}, "
              f"first-pass {c['first_pass_acc']:.3f}, kNN-in-φ floor {c['knn_phi_floor']:.3f}, "
              f"conf AUROC cos {c['confidence_auroc_cosine']:.3f} / -dist "
              f"{c['confidence_auroc_neg_distance']:.3f}**\n")
        ks = sorted({int(k) for line in c["ladders"].values() for k in line})
        print("| line | " + " | ".join(f"k={k} (acc @ ×compute)" for k in ks) + " |")
        print("|---" * (len(ks) + 1) + "|")
        for name, dd in c["ladders"].items():
            cells = []
            for k in ks:
                v = dd.get(str(k))
                cells.append(f"{v[0]:.3f} @{v[1]:.2f}×" if v else "—")
            print(f"| {name} | " + " | ".join(cells) + " |")
        if c["tau_sweep"]:
            print("\nτ-sweep (gated):")
            for tau, dd in c["tau_sweep"].items():
                best = max(dd.values(), key=lambda v: v[0])
                print(f"- τ={tau}: best {best[0]:.3f} @ {best[1]:.2f}× "
                      f"(k=0 {dd['0'][0]:.3f})")
        if c["by_task_age"]:
            print("\nper task-age (gated):")
            for a, dd in sorted(c["by_task_age"].items()):
                g = dd["gated"]
                best = max(g.values(), key=lambda v: v[0])
                print(f"- age {a} (n={dd['n']}): first {dd['first_pass_acc']:.3f} → "
                      f"best {best[0]:.3f} @ {best[1]:.2f}×, kNN-in-φ {dd['knn_phi']:.3f}")

agg = {}
for rt in d.get("retry_native", []):
    for c in rt.get("cells", []):
        stage = c["label"].split("_seed")[0]
        key = (stage, c["mask_p"])
        e = agg.setdefault(key, {"first": [], "gated": [], "gcomp": [], "kick": [],
                                 "ens": [], "ung": [], "knn": [], "ff": []})
        e["first"].append(c["first_pass_acc"])
        e["knn"].append(c["knn_phi_floor"])
        best = max(c["ladders"]["gated"].values(), key=lambda v: v[0])
        e["gated"].append(best[0])
        e["gcomp"].append(best[1])
        e["kick"].append(max(v[0] for v in c["ladders"]["kick"].values()))
        e["ens"].append(max(v[0] for v in c["ladders"]["ensemble"].values()))
        e["ung"].append(c["ladders"]["ungated"][str(max(int(k) for k in c["ladders"]["ungated"]))][0])
        ffline = c["ladders"].get("feedforward_knn_phi", {})
        e["ff"].append(max(v[0] for v in ffline.values()) if ffline else float("nan"))

if agg:
    import statistics as st
    def ms(v):
        return f"{st.mean(v):.3f}±{(st.pstdev(v) if len(v) > 1 else 0):.3f}"
    print("\n**⭐ retry aggregate over seeds (best-over-ladder)**\n")
    print("| stage | mask p | n seeds | first-pass | gated best | @compute | kick best | "
          "ensemble best | ungated @maxk | kNN-in-φ floor | ff-in-φ (k+1 votes) | gated−kNN |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (stage, p), e in sorted(agg.items()):
        diff = [g - k for g, k in zip(e["gated"], e["knn"])]
        print(f"| {stage} | {p} | {len(e['first'])} | {ms(e['first'])} | {ms(e['gated'])} | "
              f"{ms(e['gcomp'])}× | {ms(e['kick'])} | {ms(e['ens'])} | {ms(e['ung'])} | "
              f"{ms(e['knn'])} | {ms(e['ff'])} | **{ms(diff)}** |")

ret = d.get("retention", {})
for run in ret.get("runs", []):
    print(f"\n**scheduled per-item retention — seed {run['seed']}, "
          f"end-of-stream ACC with decay ON = {run['metrics_clu_with_decay']['ACC']:.3f}**\n")
    law = run["law"]
    print("| cohort | leak | half-life | ticks | live | mean amp (measured) | "
          "predicted exp(−leak·t) | max abs err | retrieval retention |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cohort in ("permanent", "slow", "fast"):
        c = law.get(cohort, {})
        if not c.get("n_points"):
            print(f"| {cohort} | — | — | 0 | — | — | — | — | — |")
            continue
        m = ", ".join(f"{v:.3f}" for v in c["measured_amp"])
        p = ", ".join(f"{v:.3f}" for v in c["predicted_amp_exp_minus_leak_t"])
        r = ", ".join(f"{v:.2f}" for v in c["retrieval_retention"])
        hl = "∞" if c["half_life_ticks"] is None else f"{c['half_life_ticks']:.1f}"
        print(f"| {cohort} | {c['leak']} | {hl} | {c['ticks']} | {c['n_live']} | {m} | "
              f"{p} | {c['max_abs_error']:.2e} | {r} |")
    print(f"\nevictions: {law['evictions']}")
