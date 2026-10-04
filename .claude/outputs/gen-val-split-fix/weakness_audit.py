"""⭐ Acceptance item 5: is ANY rival weaker after the re-selection?

Rule pre-registered in PREREG.md P5: a rival's 3-seed MEAN falling by more than ONE
OLD seed-sd is the stop-and-report trigger.  ⛔ Every rival statistic, not a chosen few.
"""
import json, numpy as np
from pathlib import Path
R = Path("/Users/user/Desktop/CHLU/.claude/outputs")
N = R / "gen-val-split-fix/runs"

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
    return out

# statistics where HIGHER = stronger for the rival
HIGHER_IS_BETTER = ("balanced_accuracy", "aggregate", "auroc", "AUROC", "factor_tuples",
                    "atom_configurations", "detection_rate", "tpr", "tnr")
SKIP = ("trials", "per_request", "row_split", "ORACLE", "oracle", "val_", "chance",
        "n_values_realised", "decoded_mean", "exact_match", "n_c", "n_control",
        "published", "delta", "tolerance", "byte", "bytes", "n_entries", "bits")

def audit(name, oldpat, newpat, roots):
    print(f"\n=== {name} ===")
    rows = {}
    for s in (0, 1, 2):
        o, n = load(oldpat.format(seed=s)), load(newpat.format(seed=s))
        if o is None or n is None:
            print(f"  seed {s}: MISSING"); return
        for root in roots:
            fo = flat(o.get("arms", {}).get(root) if root in (o.get("arms") or {}) else None, root)
            fn = flat(n.get("arms", {}).get(root) if root in (n.get("arms") or {}) else None, root)
            for k in set(fo) & set(fn):
                if any(x in k for x in SKIP): continue
                if not any(x in k for x in HIGHER_IS_BETTER): continue
                rows.setdefault(k, ([], []))
                rows[k][0].append(fo[k]); rows[k][1].append(fn[k])
    flagged = []
    for k, (a, b) in sorted(rows.items()):
        if len(a) != 3: continue
        a, b = np.array(a), np.array(b)
        d = b.mean() - a.mean()
        if d < -a.std() and abs(d) > 1e-12:
            flagged.append((k, a.mean(), a.std(), b.mean(), b.std(), d))
    print(f"  {len(rows)} rival statistics compared; {len(flagged)} fall by more than one OLD seed-sd")
    for k, am, asd, bm, bsd, d in sorted(flagged, key=lambda r: r[-1]):
        print(f"    ⚠ {k:<78} {am:.4f}±{asd:.4f} -> {bm:.4f}±{bsd:.4f}   Δ {d:+.4f}  (1 old sd = {asd:.4f})")

audit("LEG H (688,128 B) — kNN + LACE",
      str(R/"gen-arms-a7-wiring/runs/historical_688128/seed_{seed}/baselines_manifest.json"),
      str(N/"historical_688128/seed_{seed}/baselines_manifest.json"), ("knn", "lace"))
audit("LEG D (524,288 B) — kNN",
      str(R/"gen-arms-a7-wiring/runs/deployed_524288/seed_{seed}/baselines_manifest.json"),
      str(N/"deployed_524288/seed_{seed}/baselines_manifest.json"), ("knn",))
audit("CELL HARNESS — the rivals only (clu/store_removed are ours)",
      str(R/"gen-cell-harness/runs/shapes3d_recon/seed_{seed}/cell_manifest.json"),
      str(N/"cell/seed_{seed}/seed_{seed}/cell_manifest.json"),
      ("knn_proto", "lace", "lace/ld_published", "lace/ld_corrected", "ebm_compose"))
