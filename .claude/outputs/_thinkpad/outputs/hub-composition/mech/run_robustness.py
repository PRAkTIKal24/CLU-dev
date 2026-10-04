"""⭐ HOW MUCH OF THIS IS NOISE?  ⛔ Seeds 1-2 are unavailable on this box (cache), so the
spread I CAN honestly report is (a) a bootstrap over the 512 `val` rows and (b) a spread over
the ARM's own seed (atom selection + coupling fit), ⛔ NOT over bridge seeds. Labelled as such.
⛔ `val` only; B never enters."""
import os, sys, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from chlu.core.gen_store import address_codes
from chlu.core.gen_compose import read_compose
from chlu.eval.gen_probe import predict_factors
from chlu.experiments.exp_gen_dynamical_read import build_landscape
from mech_driver import build_context, arm_clu, arm_x0, score, render

ctx = build_context(0)
rows, X, lab = ctx["sets"]["val"]
Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
rng = np.random.default_rng(20260916)
out = {"bootstrap": {}, "arm_seed_spread": {}}


def valid_mask(feats):
    imgs = render(ctx["decoder"], feats, ctx["feat_shape"], ctx["eval_batch"])
    p = predict_factors(ctx["external"], imgs, ctx["eval_batch"])
    return (np.asarray(p, np.int64) == np.asarray(lab, np.int64)).all(axis=1)


def boot(name, mask, B=2000):
    n = mask.size
    s = np.array([mask[rng.integers(0, n, n)].sum() for _ in range(B)], float)
    lo, hi = np.percentile(s, [2.5, 97.5])
    out["bootstrap"][name] = {"observed": int(mask.sum()), "n": int(n),
                              "ci95_low": float(lo), "ci95_high": float(hi),
                              "sd": float(s.std())}
    print(f"{name:34s} observed={int(mask.sum()):4d}/{n}  bootstrap 95% CI "
          f"[{lo:.1f}, {hi:.1f}]  sd={s.std():.2f}", flush=True)


boot("DECODER CEILING", valid_mask(Xf))
code = address_codes(ctx["store"], Xf)
boot("store round-trip (the BOUND)",
     valid_mask(np.asarray(read_compose(ctx["store"], code), np.float32)))
boot("clu (codebook read)", valid_mask(arm_clu(ctx, "val")))

print("\n--- x0 over the ARM's OWN seed (atoms + coupling); ⛔ bridge seed is 0 throughout ---",
      flush=True)
for budget in ("A", "B"):
    vals = []
    for s in (0, 1, 2):
        built = build_landscape(ctx["cfg"], budget, ctx["X_store"], ctx["lab_store"], ctx["wc"],
                                ctx["feat_shape"], ctx["cards"], s, log=lambda *a: None)
        atoms = np.asarray(ctx["lab_store"], np.int64)[np.asarray(built["store"].atom_rows, np.int64)]
        lau = arm_x0(ctx, built["land"], built["book"], "val")
        rep = score(ctx, lau["x0"], lab, atoms)
        vals.append(rep["distinct_valid_novel"])
        print(f"  x0[{budget}] arm_seed={s}: VALID^NOVEL={rep['distinct_valid_novel']:4d}/512 "
              f"(valid={rep['distinct_valid']}, distinct={rep['distinct']})", flush=True)
    out["arm_seed_spread"][budget] = {"values": [int(v) for v in vals],
                                      "mean": float(np.mean(vals)), "sd": float(np.std(vals))}
    print(f"  ⭐ x0[{budget}] over 3 ARM seeds: {np.mean(vals):.2f} ± {np.std(vals):.2f}", flush=True)
json.dump(out, open(os.path.join(os.environ['OUT'], 'mech', 'robustness.json'), 'w'), indent=2)
print("done")
