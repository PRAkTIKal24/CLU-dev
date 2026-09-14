"""Report tables from capture_armA.json artifacts (+ the pass-1 census baseline)."""
import json
import sys
import statistics as st


def rows(path, label):
    d = json.load(open(path))
    arm = d["arms"]["armA_compact"]
    out = []
    for cell, legs, of in zip(arm["cells"], arm["gate"]["legs_by_seed"],
                              arm["own_foreign_by_seed"]):
        ws = cell["census"]["wells"]
        cap = [w["capture_radius"] for w in ws]
        dr = [w["site_drift"] for w in ws]
        lam = [w["lambda_min"] for w in ws]
        out.append(dict(
            label=label, seed=cell["seed"],
            s=cell["flags"]["clu_system_non_defaults"].get("atom_width"),
            spacing=cell["geometry"]["median_nn_task1"],
            n=len(ws),
            cap_pos=sum(c > 0 for c in cap), cap_sig=legs["G_CAP"]["n_capture_ge_sigma_q"],
            cap_med=st.median(cap),
            dec=legs["G_DEC"]["decode"], se=legs["G_DEC"]["margin_in_se"],
            drift_med=legs["G_DRIFT"]["median_site_drift"],
            drift0=sum(x < 0.01 for x in dr),
            lam_bowl=sum(abs(x - 0.1) < 1e-6 for x in lam),
            depth_med=cell["census"]["depth_raw_median"],
            own_med=of["own_median"], for_med=of["foreign_median"],
            own_mean=of["own_mean"], for_mean=of["foreign_mean"],
            f_gt_o=of["n_foreign_exceeds_own"],
            gcap=legs["G_CAP"]["pass"], gdec=legs["G_DEC"]["pass"],
            gdr=legs["G_DRIFT"]["pass"], wall=cell["wall_s"],
            bytes_clu=cell["bytes"]["clu_total_bytes"],
            bytes_knn=cell["bytes"]["knn_launder_bytes"],
            probe=cell["self_probe"],
        ))
    return out


if __name__ == "__main__":
    allr = []
    for spec in sys.argv[1:]:
        path, label = spec.split("=", 1)
        allr += rows(path, label)
    hdr = ("label seed s spacing cap_pos/n cap>=sq cap_med dec(SE) drift_med drift0 "
           "bowl depth own_med for_med f>o G-CAP G-DEC G-DRIFT wall")
    print(hdr)
    for r in allr:
        print(f"{r['label']:>10s} {r['seed']} {r['s']:.4f} {r['spacing']:.4f} "
              f"{r['cap_pos']}/{r['n']} {r['cap_sig']} {r['cap_med']:.3f} "
              f"{r['dec']:.4f}({r['se']:+.2f}) {r['drift_med']:.4f} {r['drift0']} "
              f"{r['lam_bowl']} {r['depth_med']:.3f} {r['own_med']:.4f} {r['for_med']:.5f} "
              f"{r['f_gt_o']} {r['gcap']} {r['gdec']} {r['gdr']} {r['wall']:.0f}s")
