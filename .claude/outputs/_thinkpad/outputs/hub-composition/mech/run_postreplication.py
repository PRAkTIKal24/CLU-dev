"""⭐ ACTING ON THE INDEPENDENT REPLICATION (2026-09-16).  Three repairs it demanded:
  R1 the permuted-residual control over MANY draws (my single draw of 95 was REFUTED)
  R2 the WEAKER reading of "valid" — produced tuple lands in the held-out cell at all —
     which the replication correctly says I declared and then never measured
  R3 the per-factor breakdown of the round-trip's loss
⛔ `val` + frozen B artefacts only. B's renders here are FROZEN Mac outputs, ⛔ not a new run."""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from chlu.core.gen_store import address_codes
from chlu.core.gen_compose import read_compose
from chlu.eval.gen_probe import predict_factors
from mech_driver import build_context, render

ctx = build_context(0)
rows, X, lab = ctx["sets"]["val"]
Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
nrm = np.linalg.norm(Xf, axis=1)
code = address_codes(ctx["store"], Xf)
rt = np.asarray(read_compose(ctx["store"], code), np.float32).reshape(Xf.shape)
e = rt - Xf
out = {}


def nvalid(Z):
    imgs = render(ctx["decoder"], Z.astype(np.float32), ctx["feat_shape"], ctx["eval_batch"])
    p = np.asarray(predict_factors(ctx["external"], imgs, ctx["eval_batch"]), np.int64)
    return int((p == np.asarray(lab, np.int64)).all(axis=1).sum()), p


# --- R1: the permuted control over 25 independent draws --------------------
vals = []
for s in range(25):
    r = np.random.default_rng(90000 + s)
    v, _ = nvalid(Xf + e[r.permutation(Xf.shape[0])])
    vals.append(v)
v = np.array(vals, float)
out["R1_permuted_residual"] = {"draws": 25, "values": [int(x) for x in vals],
                               "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                               "min": int(v.min()), "max": int(v.max()),
                               "my_original_single_draw": 95}
print(f"R1 permuted residual over 25 draws: mean={v.mean():.1f} sd={v.std(ddof=1):.1f} "
      f"range [{int(v.min())}, {int(v.max())}]  (my original single draw was 95)", flush=True)
gv = []
for s in range(25):
    r = np.random.default_rng(70000 + s)
    g = r.standard_normal(Xf.shape).astype(np.float32)
    g *= (np.linalg.norm(e, axis=1, keepdims=True) /
          np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9))
    gv.append(nvalid(Xf + g)[0])
g2 = np.array(gv, float)
out["R1_gaussian"] = {"draws": 25, "mean": float(g2.mean()), "sd": float(g2.std(ddof=1)),
                      "min": int(g2.min()), "max": int(g2.max()), "my_original_single_draw": 393}
print(f"R1 gaussian over 25 draws:          mean={g2.mean():.1f} sd={g2.std(ddof=1):.1f} "
      f"range [{int(g2.min())}, {int(g2.max())}]  (my original single draw was 393)", flush=True)

# --- R3: per-factor breakdown of the round-trip ----------------------------
names = list(ctx["names"])
_, p_rt = nvalid(rt)
_, p_ce = nvalid(Xf)
L = np.asarray(lab, np.int64)
out["R3_per_factor"] = {n: {"roundtrip_acc": float((p_rt[:, j] == L[:, j]).mean()),
                            "ceiling_acc": float((p_ce[:, j] == L[:, j]).mean()),
                            "n_values_realised_in_targets": int(np.unique(L[:, j]).size)}
                        for j, n in enumerate(names)}
print("R3 per-factor (round-trip vs ceiling, with n_values_realised):", flush=True)
for n, d in out["R3_per_factor"].items():
    print(f"   {n:13s} roundtrip={d['roundtrip_acc']:.4f}  ceiling={d['ceiling_acc']:.4f}  "
          f"n_values={d['n_values_realised_in_targets']}", flush=True)

# --- R2: the WEAKER reading of "valid", on the FROZEN B artefacts ----------
ART = ".claude/outputs/gen-read-cell-arm/runs/artefacts/shapes3d_recon"
out["R2_weak_valid_on_frozen_B"] = {}
for seed in (0, 1, 2):
    for safe, pretty in (("dynread", "dynread (A, full)"), ("dynread__B", "dynread/B (full)"),
                         ("dynread__A_request_off", "dynread/A_request_off"),
                         ("dynread__B_request_off", "dynread/B_request_off")):
        p = f"{ART}/seed_{seed}/decoded_s{seed}_{safe}_B.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        P = np.asarray(z["factor_tuples"], np.int64)
        T = np.asarray(z["labels"], np.int64)
        corner = set(map(tuple, np.unique(T, axis=0).tolist()))
        at = np.load(f"{ART}/seed_{seed}/stored_atom_tuples_s{seed}_"
                     f"{'A' if '_A_' in safe or safe == 'dynread' else 'B'}.npz")
        atoms = set(map(tuple, np.asarray(at["atom_factor_tuples"], np.int64).tolist()))
        inc = np.array([tuple(r) in corner for r in P.tolist()], bool)
        nov = np.array([tuple(r) not in atoms for r in P.tolist()], bool)
        strict = int(np.unique(P[(P == T).all(axis=1)], axis=0).shape[0]) if (P == T).all(axis=1).any() else 0
        weak = int(np.unique(P[inc & nov], axis=0).shape[0]) if (inc & nov).any() else 0
        out["R2_weak_valid_on_frozen_B"][f"{pretty}|seed{seed}"] = {
            "STRICT_distinct_valid_novel": strict, "WEAK_distinct_in_cell_and_novel": weak,
            "n_items_landing_in_cell": int(inc.sum()), "n": int(P.shape[0])}
        print(f"R2 {pretty:22s} seed{seed}: STRICT={strict:3d}  WEAK(in-cell & novel)={weak:3d}  "
              f"items landing in the cell={int(inc.sum()):3d}/72", flush=True)
json.dump(out, open(os.path.join(os.environ['OUT'], 'mech', 'postreplication.json'), 'w'), indent=2)
print("done")
