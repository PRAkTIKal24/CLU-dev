"""hub/placement -- IS THE STORE FIT WHAT DECIDES THE OUTCOME?  A judge-free diagnosis.

MEASURED, AND IT SAYS NO (2026-09-19).  This hub inferred that its seed-1 catastrophe
(-12.755 novel tuples/corner on row 4) must live in the store fit and the head, because the
swap probe's signature was nearly identical on all three seeds.  This module measures that
inference and refutes it: there is no degenerate well anywhere, and the span-captured delta
between the two partitions is constant and POSITIVE on all three seeds (+0.0034 / +0.0017 /
+0.0018) while the deployed outcome swings 14 tuples/corner.

WHAT IT MEASURES, per (seed, arm, well), from the fitted artifacts only:
  * ``span_captured``    the fraction of the well's feature variance that the 64 modes' span
                         captures on the store-fit rows.  This is the CEILING on what any
                         linear read of that well can deliver, so it is the quantity the
                         deployed pipeline actually consumes -- not a proxy for it.
  * ``eff_rank``         participation ratio of the modes' singular values, plus the hard
                         SVD rank.  A well whose modes collapsed has 64 modes and the rank
                         of far fewer.
  * ``mode_spread``      mean pairwise mode distance, normalised by the well's feature
                         scale.  Collapsed modes sit on top of one another.
  * ``head_fro_norm``    an exploded head block is a sign the ridge is fighting an
                         ill-conditioned span.

WHY IT IS WORTH KEEPING EVEN THOUGH ITS ANSWER WAS NO: it is cheap (no judge, no decode),
it rules out the most obvious mechanical explanation for a partition behaving badly, and a
hub that DOES find a degenerate well here has found a real defect in one measurement.
"""

import io, json
import numpy as np
import pickle
from pathlib import Path
from chlu.experiments import exp_gen_factor_wells as W

ARMS = ("place_norm_rank", "factor_aligned_balanced")
DEPLOYED = {0: +1.388, 1: -12.755, 2: +2.163}
ROOT = Path("C:/chlu/outputs/hub-placement/scored-1")
W.PINS["condition"] = "test"

out = {"deployed_row4_delta": DEPLOYED, "prediction": __doc__.split("PREDICTION,")[1].strip(),
       "per_seed": {}}

for s in (0, 1, 2):
    W.PINS["seed"] = s
    cond = W.load_condition()
    pool = W.fit_pool(cond["train_rows"], int(W.PINS["n_pool"]),
                      int(W.PINS["pool_rng_seed"]))
    F = W.features_mmap(s)
    X = W.gather(F, pool[: int(W.PINS["n_store_train"])])
    e = {}
    for arm in ARMS:
        p = ROOT / f"seed{s}" / "arms" / arm / "arm.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as fh:
            o = pickle.load(fh)
        st, heads = o["store"], o["heads"]
        rows = []
        for w in range(int(st.n_wells)):
            ch = np.asarray(st.well_channels[w], np.int64)
            G = np.asarray(st.modes[w], np.float64)          # (64, d_w)
            Z = X[:, ch].astype(np.float64)                  # (n, d_w)
            Zc = Z - Z.mean(axis=0, keepdims=True)
            tot = float((Zc ** 2).sum())
            # the span of the modes (mean-centred the same way the read sees it)
            Gc = G - G.mean(axis=0, keepdims=True)
            U, sv, Vt = np.linalg.svd(Gc, full_matrices=False)
            k = int((sv > sv.max() * 1e-8).sum()) if sv.size and sv.max() > 0 else 0
            B = Vt[:k]                                       # orthonormal basis of the span
            proj = Zc @ B.T @ B
            cap = float((proj ** 2).sum() / tot) if tot > 0 else float("nan")
            pr = float((sv.sum() ** 2) / (sv ** 2).sum()) if (sv ** 2).sum() > 0 else 0.0
            d2 = ((G[:, None, :] - G[None, :, :]) ** 2).sum(-1)
            scale = float(np.sqrt((Zc ** 2).mean())) or 1.0
            spread = float(np.sqrt(d2[np.triu_indices(G.shape[0], 1)]).mean()) / scale
            hb = np.asarray(heads[w], np.float64)
            rows.append({"well": w, "n_channels": int(ch.size),
                         "span_captured": cap, "eff_rank": pr, "svd_rank": k,
                         "mode_spread_norm": spread,
                         "head_fro_norm": float(np.linalg.norm(hb))})
        e[arm] = {"wells": rows,
                  "span_captured_mean": float(np.mean([r["span_captured"] for r in rows])),
                  "span_captured_min": float(np.min([r["span_captured"] for r in rows])),
                  "eff_rank_mean": float(np.mean([r["eff_rank"] for r in rows])),
                  "eff_rank_min": float(np.min([r["eff_rank"] for r in rows])),
                  "recon_relative_mse": o["recon_mse"]["relative_mse"]}
        print("seed %d  %-26s span_captured mean %.4f min %.4f | eff_rank mean %.1f min %.1f | rel_mse %.6f"
              % (s, arm, e[arm]["span_captured_mean"], e[arm]["span_captured_min"],
                 e[arm]["eff_rank_mean"], e[arm]["eff_rank_min"], e[arm]["recon_relative_mse"]))
        for r in rows:
            print("        well %d (%4d ch): span %.4f  eff_rank %5.1f (svd %2d)  spread %.3f  head|F| %.1f"
                  % (r["well"], r["n_channels"], r["span_captured"], r["eff_rank"],
                     r["svd_rank"], r["mode_spread_norm"], r["head_fro_norm"]))
    if len(e) == 2:
        a, b = e[ARMS[0]], e[ARMS[1]]
        e["span_delta_arm_minus_baseline_mean"] = (a["span_captured_mean"]
                                                  - b["span_captured_mean"])
        e["span_delta_arm_minus_baseline_min"] = (a["span_captured_min"]
                                                 - b["span_captured_min"])
        e["deployed_row4_delta"] = DEPLOYED[s]
        print("   -> span delta (arm - baseline): mean %+.4f  min %+.4f   DEPLOYED %+.3f"
              % (e["span_delta_arm_minus_baseline_mean"],
                 e["span_delta_arm_minus_baseline_min"], DEPLOYED[s]))
    out["per_seed"][str(s)] = e
    del X

ps = out["per_seed"]
ks = [k for k in sorted(ps) if "span_delta_arm_minus_baseline_mean" in ps[k]]
if len(ks) >= 3:
    sd = [ps[k]["span_delta_arm_minus_baseline_mean"] for k in ks]
    dp = [ps[k]["deployed_row4_delta"] for k in ks]
    out["pearson_spanmean_vs_deployed"] = float(np.corrcoef(sd, dp)[0, 1])
    out["sign_agreement"] = [bool(np.sign(x) == np.sign(y)) for x, y in zip(sd, dp)]
    print()
    print("span-mean delta per seed: %s" % [round(x, 4) for x in sd])
    print("deployed delta per seed : %s" % dp)
    print("sign agreement: %s   pearson %+.4f"
          % (out["sign_agreement"], out["pearson_spanmean_vs_deployed"]))
W._jdump(out, Path("C:/chlu/outputs/hub-placement/store_fit_diagnosis.json"))
print("-> store_fit_diagnosis.json")
