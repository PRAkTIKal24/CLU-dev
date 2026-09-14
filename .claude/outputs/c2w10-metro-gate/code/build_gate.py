import glob, json, os, subprocess
import numpy as np

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
DST = "/Users/user/Desktop/CHLU/.claude/outputs/c2w10-metro-gate"
J = lambda f: json.load(open(os.path.join(OUT, f)))
facts, meta, m1 = J("facts.json"), J("pairs_meta.json"), J("m1.json")
dmaps, bands, nullm, dec = J("driftmaps.json"), J("bands.json"), J("null_meta.json"), J("decimation.json")
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, "res", "*.json"))}

STRONG_PREFIX = ("gbdt", "gru", "mlp", "ridge_batch", "rls")
strong = {k: v for k, v in R.items() if k.startswith(STRONG_PREFIX) and "shuf" not in k
          and k not in ("rls_ff",)}
exe = {k: v for k, v in R.items() if k.startswith("knn") and "shuf" not in k}

# --- registered arithmetic ---
best_strong = min(strong, key=lambda k: strong[k]["mae"])
budget = {"knn_windows_634kib": 5037, "knn_windows_133kb": 1007, "knn_windows_clu_1875kib": 14894}
exe_at = {}
for lbl, L in budget.items():
    cands = {k: v for k, v in exe.items() if f"_{L}_" in k}
    b = min(cands, key=lambda k: cands[k]["mae"])
    exe_at[lbl] = dict(arm=b, **{q: cands[b][q] for q in ("mae", "rmse", "state_bytes")})
best_exe_budgeted = min(("knn_windows_634kib", "knn_windows_133kb"), key=lambda k: exe_at[k]["mae"])
mae_e = exe_at[best_exe_budgeted]["mae"]
mae_s = strong[best_strong]["mae"]
margin_rel = (mae_e - mae_s) / mae_s
cleared = bool(margin_rel > 0.02)

robust = []
for k in sorted(strong, key=lambda k: strong[k]["mae"]):
    mr = (mae_e - strong[k]["mae"]) / strong[k]["mae"]
    robust.append(dict(strong_reference=k, mae=round(strong[k]["mae"], 3),
                       margin_rel=round(mr, 5), fires=bool(mr <= 0.02)))
# also the strongest possible launder (unregistered, k-ladder + unbudgeted) against the best strong
best_exe_any = min(exe, key=lambda k: exe[k]["mae"])

mae_pers, mae_sn = m1["persistence"]["mae"], m1["seasonal_naive_168"]["mae"]
m3_head = dict(
    best_strong=best_strong, mae_best_strong=round(mae_s, 3),
    mae_persistence=round(mae_pers, 3), mae_seasonal_naive_168=round(mae_sn, 3),
    rel_gain_over_persistence=round((mae_pers - mae_s) / mae_pers, 5),
    rel_gain_over_seasonal_naive_168=round((mae_sn - mae_s) / mae_sn, 5),
    rel_gain_of_best_exemplar_over_persistence=round((mae_pers - exe[best_exe_any]["mae"]) / mae_pers, 5),
    rel_gain_of_best_exemplar_over_seasonal_naive_168=round((mae_sn - exe[best_exe_any]["mae"]) / mae_sn, 5))
m1_pass = bool(m3_head["rel_gain_over_persistence"] >= 0.05)

freeze = subprocess.run([os.path.join(OUT, ".venv/bin/python"), "-m", "pip", "freeze"],
                        capture_output=True, text=True)
if freeze.returncode != 0:
    freeze = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True,
                            env={**os.environ, "VIRTUAL_ENV": os.path.join(OUT, ".venv")})
open(os.path.join(DST, "scratch_venv_freeze.txt"), "w").write(freeze.stdout)

shuf = {k: v for k, v in R.items() if k.endswith("_shuf1")}
null_res = []
for k in sorted(shuf):
    base = k[:-6]
    if base in R:
        null_res.append(dict(arm=base, mae_ordered=round(R[base]["mae"], 3),
                             mae_shuffled=round(shuf[k]["mae"], 3),
                             rel_degradation=round((shuf[k]["mae"] - R[base]["mae"]) / R[base]["mae"], 5)))

gate = dict(
    _what=("Criterion-4 tripwire + drift map for Metro Interstate Traffic Volume under the "
           "hidden-clock 24-h-ahead protocol. Produced by results-analyst / c2w10-metro-gate. "
           "MECHANICS instrument only: NO CLU cell of any kind was run."),
    m1_persistence=dict(mae=round(m1["persistence"]["mae"], 4), rmse=round(m1["persistence"]["rmse"], 4)),
    m1_seasonal_naive=dict(
        t_minus_24h=dict(mae=round(m1["seasonal_naive_24"]["mae"], 4),
                         rmse=round(m1["seasonal_naive_24"]["rmse"], 4),
                         note=("DEGENERATE: at a 24-h horizon the period-24 seasonal-naive IS "
                               "last-observed-at-origin. max|diff| from persistence = %g"
                               % m1["_degeneracy_check_abs_diff"])),
        t_minus_168h=dict(mae=round(m1["seasonal_naive_168"]["mae"], 4),
                          rmse=round(m1["seasonal_naive_168"]["rmse"], 4),
                          note="the non-degenerate second naive; supplied because t-24h collapsed"),
        others={k: dict(mae=round(v["mae"], 4), rmse=round(v["rmse"], 4))
                for k, v in m1.items() if not k.startswith("_")}),
    m1_pass=m1_pass,
    m1_pass_rule="registered: (mae_persistence - mae_best_strong)/mae_persistence >= 0.05",
    m2_arms={**exe_at,
             **{k: dict(arm=k, mae=round(v["mae"], 4), rmse=round(v["rmse"], 4),
                        state_bytes=v["state_bytes"]) for k, v in strong.items()}},
    m2_best_strong=dict(arm=best_strong, mae=round(mae_s, 4)),
    m2_best_exemplar_at_budget=dict(budget=best_exe_budgeted, **exe_at[best_exe_budgeted]),
    m2_best_exemplar_any=dict(arm=best_exe_any, mae=round(exe[best_exe_any]["mae"], 4),
                              state_bytes=exe[best_exe_any]["state_bytes"],
                              note="unregistered / unbudgeted upper bound on the launder"),
    m2_margin_rel=round(margin_rel, 6),
    m2_margin_rule="criterion4_cleared_metro = (m2_margin_rel > 0.02)",
    criterion4_cleared_metro=cleared,
    m2_robustness=robust,
    m2_exemplar_ladder={k: dict(mae=round(v["mae"], 4), rmse=round(v["rmse"], 4),
                                state_bytes=v["state_bytes"]) for k, v in
                        sorted(exe.items(), key=lambda x: x[1]["mae"])},
    m3_headroom=m3_head,
    byte_ledger=dict(
        bytes_per_exemplar=132, bytes_per_exemplar_derivation="(32 float32 features + 1 float32 target) x 4 B",
        L_at_665000B=665000 // 132, L_at_133000B=133000 // 132, L_at_CLU_1966080B=1966080 // 132,
        arms={k: v["state_bytes"] for k, v in sorted(R.items(), key=lambda x: x[1]["state_bytes"])
              if "shuf" not in k},
        not_byte_matched=["gbdt/gbdt_recent/gbdt_cat (pickled HistGradientBoostingRegressor); "
                          "these are NOT matched to the 665,000 B exemplar budget",
                          "gru/gru_big (n_params x 4 B, excludes optimizer state and the replay buffer, "
                          "which is NOT counted and would dominate: buf x 132 B)",
                          "mlp (pickled sklearn MLPRegressor)"],
        note=("every arm receives the same 32-D feature vector for free; the ledger counts only "
              "RETAINED state, i.e. the store / model parameters")),
    drift_map=dict(method=dmaps["_method"],
                   annotation_ownership="OURS, NOT THE LITERATURE'S -- Metro has no published drift annotation",
                   maps={k: v for k, v in dmaps.items() if not k.startswith("_")},
                   band_diagnostics=bands["bands"],
                   exclusion_rule=bands["exclusion_rule"],
                   excluded_bands={m: [r["band"] for r in rows if r["excluded"]]
                                   for m, rows in bands["bands"].items()}),
    drift_free_null=dict(**nullm, results=null_res),
    stream=dict(path=facts["stream_csv"]["path"], sha256=facts["stream_csv"]["sha256"],
                source_zip_sha256=facts["source_zip"]["sha256"],
                csv_gz_sha256=facts["stream_csv_gz"]["sha256"],
                n_records=facts["n_raw_records"], n_unique_hours=facts["n_unique_timestamps"],
                n_duplicate_rows=facts["n_duplicate_rows"],
                grid_hours=facts["grid_hours"], n_missing_hours=facts["n_missing_hours"],
                largest_gap_hours=facts["longest_gap_hours"][0],
                pairs_npz_sha256=meta["stream_sha256"], n_scored_pairs=meta["n_pairs"],
                features_used=meta["feature_names"], n_features=32,
                horizon_hours=24, hidden_clock=True,
                canonicalisation=dict(dedupe=facts["dedupe_rule"], gapfill_max_hours=3,
                                      gapfill_method="linear interp on traffic, forward-fill on weather",
                                      target_never_imputed=True)),
    scratch_venv=dict(path=os.path.join(OUT, ".venv"), python="3.12.9",
                      freeze_file=os.path.join(DST, "scratch_venv_freeze.txt"),
                      project_lock_touched=False),
    river_or_lib_versions=dict(river="NOT USED (regression venue; river ships no regression SAM-kNN)",
                               numpy="2.5.2", scikit_learn="1.9.0", scipy="1.18.0",
                               torch="2.13.0", pandas="3.0.5 (installed, unused)"),
    reproduction_gate=dict(
        contract_sha256_csv=facts["stream_csv"]["sha256"],
        contract_sha256_pairs=meta["stream_sha256"],
        rule=("the CLU harness's own loader must reproduce BOTH sha256 values -- the raw UCI CSV and "
              "the built 34,848-pair array (sha256 over X.tobytes()+y.tobytes()+tgt.tobytes(), "
              "float32/float32/int64, C order) -- BEFORE any number in this file is consumed. "
              "Mismatch = hard stop, not a tolerance."),
        builder_scripts=["prep.py", "build.py"]),
    decimation_ladder=dec,
    decimation_hazards=[
        "the 24-h horizon is 24 records at m=1 and 12 at m=2, but 4.8 at m=5 and 2.4 at m=10: "
        "NON-INTEGER. m in {5,10} is NOT USABLE without redefining the horizon and every lag feature.",
        "decimation compresses the drift timeline; any adaptation quantity must be reported "
        "PER-INSTANCE-SINCE-CHANGE, never per-stream-position",
        "Metro is already 0.7x INSECTS in scored pairs (34,848 vs 79,986) and the full arm suite "
        "costs seconds-to-minutes on CPU: decimation is NOT NEEDED. m = 1 is the recommendation."],
    notes=[])
os.makedirs(DST, exist_ok=True)
with open(os.path.join(DST, "METRO-GATE.json"), "w") as f:
    json.dump(gate, f, indent=2)
print("criterion4_cleared_metro =", cleared, " margin_rel =", round(margin_rel, 5))
print("best strong:", best_strong, round(mae_s, 3), "| best exemplar at budget:",
      exe_at[best_exe_budgeted]["arm"], round(mae_e, 3), "| best exemplar any:", best_exe_any,
      round(exe[best_exe_any]["mae"], 3))
print("m1_pass =", m1_pass, m3_head)
