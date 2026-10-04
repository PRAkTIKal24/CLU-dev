"""Did the SELECTED configuration move? One row per (family, seed)."""
import json, sys
from pathlib import Path
R = Path("/Users/user/Desktop/CHLU/.claude/outputs")
N = R / "gen-val-split-fix/runs"

def g(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None

def knn_lace(old, new, leg, arms=("knn", "lace")):
    print(f"\n--- {leg} ---")
    n_moved = n_tot = 0
    for s in (0, 1, 2):
        o, w = g(old.format(seed=s)), g(new.format(seed=s))
        if o is None or w is None:
            print(f"  seed {s}: MISSING"); continue
        for a in arms:
            ao, aw = (o["arms"].get(a)), (w["arms"].get(a))
            if not ao or not aw: continue
            ks = ("k","metric","normaliser","deployment") if a == "knn" else ("sigma","form","shrinkage")
            for rule in ("selected", "selected_compositional_rule"):
                so = {k: ao["tuning"][rule][k] for k in ks}
                sw = {k: aw["tuning"][rule][k] for k in ks}
                n_tot += 1; moved = so != sw; n_moved += moved
                print(f"  seed {s} {a:5s} {rule:28s} {'MOVED   ' if moved else 'same    '} "
                      f"{so} -> {sw}")
            if a == "lace":
                go = ao.get("whitening_exactness_gate") or {}
                gw = aw.get("whitening_exactness_gate") or {}
                print(f"  seed {s} lace  R27.3 gate fired {go.get('fired')} -> {gw.get('fired')}; "
                      f"selected_is_exact {go.get('selected_is_exact')} -> {gw.get('selected_is_exact')}; "
                      f"fallback {go.get('reason')!r} -> {gw.get('reason')!r}")
    print(f"  => selection moved in {n_moved}/{n_tot} (arm, seed, rule) cells")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all","D"):
        knn_lace(str(R/"gen-arms-a7-wiring/runs/deployed_524288/seed_{seed}/baselines_manifest.json"),
                 str(N/"deployed_524288/seed_{seed}/baselines_manifest.json"),
                 "LEG D (deployed 524,288 B) — kNN only", arms=("knn",))
    if which in ("all","H"):
        knn_lace(str(R/"gen-arms-a7-wiring/runs/historical_688128/seed_{seed}/baselines_manifest.json"),
                 str(N/"historical_688128/seed_{seed}/baselines_manifest.json"),
                 "LEG H (historical 688,128 B) — kNN + LACE")
