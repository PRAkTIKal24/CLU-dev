"""MOVED / UNCHANGED table: published manifests vs the re-score on the repaired split."""
import json, sys, os
from pathlib import Path
import numpy as np

REPO = Path("/Users/user/Desktop/CHLU")
OUT = REPO / ".claude/outputs/gen-val-split-fix"
PUB = REPO / ".claude/outputs"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)
    elif isinstance(d, (str, bool)):
        out[prefix] = d
    return out


def cmp_pair(old, new, keys=None, skip=()):
    fo, fn = flat(old), flat(new)
    rows = []
    for k in sorted(set(fo) | set(fn)):
        if any(s in k for s in skip):
            continue
        if keys is not None and not any(k.endswith(x) or x in k for x in keys):
            continue
        a, b = fo.get(k), fn.get(k)
        if isinstance(a, float) and isinstance(b, float):
            rows.append((k, a, b, b - a, a == b))
        elif a != b:
            rows.append((k, a, b, None, False))
    return rows


def seed_table(name, oldpat, newpat, keys, seeds=(0, 1, 2), skip=()):
    print(f"\n{'='*100}\n### {name}\n{'='*100}")
    agg = {}
    for s in seeds:
        o, n = load(oldpat.format(seed=s)), load(newpat.format(seed=s))
        if o is None or n is None:
            print(f"seed {s}: MISSING (old={o is not None}, new={n is not None})")
            continue
        for k, a, b, d, same in cmp_pair(o, n, keys, skip):
            agg.setdefault(k, {})[s] = (a, b, d, same)
    for k, per in agg.items():
        if all(v[3] for v in per.values()):
            tag = "UNCHANGED (bit-identical)"
            print(f"  {k:<72} {tag}  {list(per.values())[0][0]}")
        else:
            olds = [v[0] for v in per.values()]
            news = [v[1] for v in per.values()]
            if all(isinstance(x, float) for x in olds + news):
                print(f"  {k:<72} MOVED  "
                      f"{np.mean(olds):.4f}±{np.std(olds):.4f} -> "
                      f"{np.mean(news):.4f}±{np.std(news):.4f}  "
                      f"(Δmean {np.mean(news)-np.mean(olds):+.4f}; per-seed "
                      + ", ".join(f"{a:.4f}->{b:.4f}" for a, b, _, _ in per.values()) + ")")
            else:
                print(f"  {k:<72} MOVED  " +
                      ", ".join(f"s{s}: {v[0]!r}->{v[1]!r}" for s, v in per.items()))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    HEAD_KEYS = ["balanced_accuracy", "aggregate", "auroc", "AUROC", "refused_fraction",
                 "val_aggregate", "val_compositional", "selected", "n_entries", "sigma",
                 "form", "shrinkage", "metric", "normaliser", "deployment", ".k",
                 "fired", "is_exact", "mean_abs_diag_minus_one", "realised_bytes",
                 "n_holdout_leaked", "param_hash", "relative", "read_error", "exponent"]
    if which in ("all", "D"):
        seed_table(
            "LEG D — deployed 524,288 B, kNN family (knn / knn_compsel / knn_proto)",
            str(PUB / "gen-arms-a7-wiring/runs/deployed_524288/seed_{seed}/baselines_manifest.json"),
            str(OUT / "runs/deployed_524288/seed_{seed}/baselines_manifest.json"),
            HEAD_KEYS, skip=("trials", "per_request", "row_split"),
        )
    if which in ("all", "H"):
        seed_table(
            "LEG H — historical 688,128 B, kNN + LACE (LACE's published leg + R27.3 fallback)",
            str(PUB / "gen-arms-a7-wiring/runs/historical_688128/seed_{seed}/baselines_manifest.json"),
            str(OUT / "runs/historical_688128/seed_{seed}/baselines_manifest.json"),
            HEAD_KEYS, skip=("trials", "per_request", "row_split"),
        )
    if which in ("all", "C"):
        seed_table(
            "CELL HARNESS — clu / ebm_compose / lace / knn_proto / store_removed",
            str(PUB / "gen-cell-harness/runs/shapes3d_recon/seed_{seed}/cell_manifest.json"),
            str(OUT / "runs/cell/seed_{seed}/cell_manifest.json"),
            HEAD_KEYS, skip=("trials", "per_request", "row_split"),
        )
    if which in ("all", "A"):
        seed_table(
            "ATOM-MEAN CONTROL — both budgets",
            str(PUB / "gen-atom-mean-control/runs/shapes3d_recon/seed_{seed}/atom_manifest.json"),
            str(OUT / "runs/atom_mean/seed_{seed}/atom_manifest.json"),
            HEAD_KEYS, skip=("trials", "per_request", "row_split"),
        )
