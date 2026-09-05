"""Re-derive every number in the report from the shipped per-arm metrics JSONs.

Usage:  python render.py            (reads ../../scratch/c2w8-cifar-strong-phi/*/results/)
"""
import json
import math
import os
import sys

SCRATCH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scratch", "c2w8-cifar-strong-phi",
)
ARMS = ["pca", "randconv", "convae", "simclr"]
BANKED_PCA_NULL = 0.149  # w25 cl-entry-build, Split-CIFAR-10 reduced protocol


def _load(arm):
    p = os.path.join(SCRATCH, arm, "results", "exp_cl_entry_cifar10_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, float("nan"), 1
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n), n


def per_seed(res, method):
    return {r["seed"]: r["metrics"] for r in res["rows"] if r["method"] == method}


def arm_summary(arm, res):
    clu = per_seed(res, "clu_entry_task1_only")
    same = per_seed(res, "knn_phi_same_keys_task1_only")
    ring = per_seed(res, "knn_phi_ringbuffer_task1_only")
    seeds = sorted(clu)
    acc = [clu[s]["ACC"] for s in seeds]
    bwt = [clu[s]["BWT"] for s in seeds]
    fgt = [clu[s]["forgetting"] for s in seeds]
    # N8 is PAIRED per seed against the STRONGER of the two launder lines
    marg = [clu[s]["ACC"] - max(same[s]["ACC"], ring[s]["ACC"]) for s in seeds]
    marg_same = [clu[s]["ACC"] - same[s]["ACC"] for s in seeds]
    marg_ring = [clu[s]["ACC"] - ring[s]["ACC"] for s in seeds]
    row = {r["method"]: r for r in res["baseline_table"]}
    return {
        "arm": arm,
        "phi_arm_in_json": res["config"]["phi_arm"],
        "phi_dim": res["config"]["phi_dim"],
        "enc_steps": res["config"].get("enc_steps"),
        "seeds": seeds,
        "ACC": mean_se(acc), "BWT": mean_se(bwt), "forgetting": mean_se(fgt),
        "knn_same": mean_se([same[s]["ACC"] for s in seeds]),
        "knn_ring": mean_se([ring[s]["ACC"] for s in seeds]),
        "knn_ring_bwt": mean_se([ring[s]["BWT"] for s in seeds]),
        "N8_margin": mean_se(marg), "N8_per_seed": dict(zip(seeds, marg)),
        "N8_positive_seeds": sum(1 for m in marg if m > 0),
        "margin_vs_same_keys": mean_se(marg_same),
        "margin_vs_ringbuffer": mean_se(marg_ring),
        "clu_acc_per_seed": dict(zip(seeds, acc)),
        "memory_floats": row["clu_entry_task1_only"]["memory_floats"],
        "phi_param_floats": row["clu_entry_task1_only"]["phi_param_floats"],
        "total_floats": row["clu_entry_task1_only"]["total_floats"],
        "ring_total_floats": row["knn_phi_ringbuffer_task1_only"]["total_floats"],
        "memory_items": row["clu_entry_task1_only"]["memory_items"],
        "verdict": res["verdict"],
        "phi_provenance": res["entry_runs"][0]["phi_provenance"],
        "geometry": res["entry_runs"][0]["geometry"],
        "admitted_fraction_per_task": res["entry_runs"][0][
            "admitted_fraction_per_task"],
    }


def fmt(ms, k=3):
    m, se, n = ms
    if n <= 1:
        return f"{m:.{k}f} (n=1)"
    return f"{m:.{k}f} ± {se:.{k}f}"


def main():
    out = {}
    for arm in ARMS:
        res = _load(arm)
        if res is None:
            print(f"[render] {arm}: NOT PRESENT", file=sys.stderr)
            continue
        out[arm] = arm_summary(arm, res)

    print("\n=== CLU entry per arm (Split-CIFAR-10, reduced protocol, task1_only) ===")
    print(f"{'arm':10s} {'phi_dim':>7s} {'steps':>6s} {'n':>2s} {'CLU ACC':>16s} "
          f"{'kNN same':>16s} {'kNN ring':>16s} {'N8 (paired)':>18s} {'+/n':>5s}")
    for arm, s in out.items():
        print(f"{arm:10s} {s['phi_dim']:>7d} {str(s['enc_steps']):>6s} "
              f"{len(s['seeds']):>2d} {fmt(s['ACC']):>16s} {fmt(s['knn_same']):>16s} "
              f"{fmt(s['knn_ring']):>16s} {fmt(s['N8_margin']):>18s} "
              f"{s['N8_positive_seeds']}/{len(s['seeds']):>2}")

    print("\n=== N8 decomposed (paired, per seed) + BWT ===")
    for arm, s in out.items():
        print(f"{arm:10s} vs same-keys {fmt(s['margin_vs_same_keys'], 4):>20s} | "
              f"vs ring buffer {fmt(s['margin_vs_ringbuffer'], 4):>20s} | "
              f"BWT {fmt(s['BWT']):>16s} | forget {fmt(s['forgetting']):>16s}")
        print(f"{'':10s} CLU per seed {({k: round(v, 4) for k, v in s['clu_acc_per_seed'].items()})}"
              f"  N8 per seed {({k: round(v, 4) for k, v in s['N8_per_seed'].items()})}")

    print("\n=== N7: lift over the banked PCA-phi null ===")
    ref = out.get("pca")
    for arm, s in out.items():
        if arm == "pca":
            continue
        lift_banked = s["ACC"][0] - BANKED_PCA_NULL
        line = f"{arm:10s} vs banked 0.149: {lift_banked:+.3f}"
        if ref:
            d = s["ACC"][0] - ref["ACC"][0]
            se = math.sqrt(
                (0 if math.isnan(s["ACC"][1]) else s["ACC"][1]) ** 2
                + (0 if math.isnan(ref["ACC"][1]) else ref["ACC"][1]) ** 2
            )
            line += f" | vs in-harness pca {ref['ACC'][0]:.3f}: {d:+.3f} ± {se:.3f}"
            line += f"  [{'HIT' if d >= 0.10 else 'MISS'} vs the registered +0.10]"
        print(line)

    print("\n=== Byte ledger (floats; phi params counted on every arm) ===")
    print(f"{'arm':10s} {'items':>6s} {'store':>9s} {'phi params':>11s} "
          f"{'CLU total':>10s} {'ring total':>11s}")
    for arm, s in out.items():
        print(f"{arm:10s} {s['memory_items']:>6d} {s['memory_floats']:>9d} "
              f"{s['phi_param_floats']:>11d} {s['total_floats']:>10d} "
              f"{s['ring_total_floats']:>11d}")

    base = _load("pca")
    if base:
        print("\n=== Baseline table (from the pca reference run; phi-independent) ===")
        for r in base["baseline_table"]:
            print(f"{r['method']:34s} {r['class']:16s} ACC {r['ACC']:.3f} "
                  f"± {r['ACC_sd']:.3f}  BWT {r['BWT']:+.3f}  "
                  f"forget {r['forgetting']:.3f}  mem {r['memory_floats']:>8d} "
                  f"fixed {r.get('fixed_state_floats', 0):>7d} "
                  f"phi {r.get('phi_param_floats', 0):>7d} "
                  f"total {r.get('total_floats', 0):>8d}  n={r['n_seeds']}")

    print("\n=== Geometry / provenance ===")
    for arm, s in out.items():
        g = s["geometry"]
        print(f"{arm:10s} med-NN {g.get('median_nn_addresses', float('nan')):.3f} "
              f"s {g.get('well_width_s', float('nan')):.3f} "
              f"sigma_q {g.get('sigma_q_norm', float('nan')):.3f} "
              f"slack {g.get('packing_slack_corrected', float('nan')):.3f} | "
              f"phi {s['phi_provenance']}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
