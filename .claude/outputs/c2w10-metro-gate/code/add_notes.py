import glob, json, os
DST = "/Users/user/Desktop/CHLU/.claude/outputs/c2w10-metro-gate"
OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
g = json.load(open(os.path.join(DST, "METRO-GATE.json")))
leaky = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, "res_leaky", "*.json"))}
emb = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, "res", "*.json"))}
delta = []
for k in sorted(set(leaky) & set(emb)):
    if k.endswith("_shuf1"):
        continue
    delta.append(dict(arm=k, mae_leaky_no_embargo=round(leaky[k]["mae"], 3),
                      mae_embargoed_PRIMARY=round(emb[k]["mae"], 3),
                      leak_worth_rel=round((leaky[k]["mae"] - emb[k]["mae"]) / emb[k]["mae"], 5)))
delta.sort(key=lambda r: r["leak_worth_rel"])
g["protocol_correction_24h_label_embargo"] = dict(
    what=("Found by results-analyst mid-run. PREREG.md 1 asserted that plain prequential "
          "test-then-train over pairs ordered by target time is leak-free. IT IS NOT: pair t-1's "
          "label is the traffic volume at target-time j-1, which is 23 h AFTER pair t's forecast "
          "origin j-24. Plain test-then-train therefore hands every continuously-updated learner up "
          "to 23 h of future traffic, and it does so ASYMMETRICALLY -- an immediately-updated k-NN "
          "store gains from it, a GBDT refit every 720 pairs does not. The bias is toward FIRING "
          "criterion 4."),
    fix=("A(t) = index of the last pair whose TARGET time is <= pair t's ORIGIN time "
         "(= searchsorted(tgt, tgt-24, 'right')-1; median t-A(t) = 24, mean 22.86, max 24). Every "
         "store/fit/update is restricted to indices <= A(t). ALL PRIMARY NUMBERS IN THIS FILE ARE "
         "EMBARGOED."),
    delta_table=delta,
    verdict_under_both=("criterion4_cleared_metro = false under BOTH protocols: leaky margin_rel "
                        "= (307.647-335.203)/335.203 is not comparable (leaky strong arms differ); "
                        "computed within-protocol, leaky margin_rel = -0.0946 vs best leaky strong "
                        "(gbdt 339.796), embargoed margin_rel = -0.06154 vs best embargoed strong "
                        "(gbdt_tuned 335.203). The embargo REDUCES the exemplar store's advantage "
                        "by ~3 points of relative MAE and does not change the verdict."))
g["static_holdout_sanity_check"] = dict(
    protocol="single 70/30 chronological split of the 34,848 pairs, no streaming, no refits",
    gbdt_default_mae=273.54, gbdt_tuned_mae=266.33,
    knn_k5_mae=277.32, knn_k10_mae=267.58, knn_k25_mae=265.47,
    persistence_mae=562.28, seasonal_naive_168_mae=326.30,
    reading=("confirms the prequential GBDT is not broken, and that k-NN/GBDT parity on this "
             "feature space is a property of the DATA, not of the streaming harness: on a static "
             "holdout a plain distance-weighted k-NN (265.47) EDGES the tuned GBDT (266.33) by "
             "0.3 % relative. Criterion 4 fires under the static reading too."))
g["observed_divergences_and_failures"] = [
    dict(arm="rls_ff (RLS with forgetting factor 0.999)", observed="MAE 1881.72 / RMSE 52680.9 "
         "(leaky run); re-run at forget=0.999 gave MAE 2979.11 / RMSE 227708.7",
         cause="Sherman-Morrison covariance update is numerically unstable at forget<1 on these "
               "features", disposition="EXCLUDED from every table and from the strong-baseline set; "
               "declared NOT-RUN under the embargoed protocol"),
    dict(arm="gru (first run, pre-fix)", observed="MAE 928.60 / RMSE 57288.6; two predictions of "
         "7.7e6 and 7.4e6 at pair indices 12566 and 4215",
         cause="causal per-feature z-score reached 1e6 on near-constant early features "
               "(snow_1h, holiday) at t<500",
         disposition="fixed by a declared +-10 SD clip applied to EVERY standardised arm; the GRU "
                     "then scored 442.36 (leaky) / 447.47 (embargoed). The pre-fix number is "
                     "reported here and used nowhere."),
]
g["m1_criterion2_caveat"] = dict(
    headroom_over_persistence=g["m3_headroom"]["rel_gain_over_persistence"],
    headroom_over_seasonal_naive_168=g["m3_headroom"]["rel_gain_over_seasonal_naive_168"],
    reading=("M1 PASSES as registered (41.6 % gain over persistence). But the registered rule used "
             "persistence, and persistence at h=24 is NOT the strongest trivial rule on this stream. "
             "The best tuned strong baseline beats the ONE-LINE weekly seasonal-naive by only "
             "2.17 % relative MAE (335.20 vs 342.65). That is the ELEC2 pathology one level up: a "
             "trivial method is essentially at the strong baselines' ceiling. INDEPENDENTLY of "
             "criterion 4, this is a criterion-2 (real headroom) problem for Metro."))
g["notes"] = [
    "MECHANICS instrument only. NO CLU cell of any kind was run. No tracked file was modified.",
    "Registered k for the exemplar arms is 5; the k-ladder {1,3,10,25,50} is an UNREGISTERED "
    "anti-hobbling extension of the launder and it made the launder stronger (k=10 beats k=5 at "
    "every budget >= 5037). The gate consumes the best, per the F3 anti-hobbling rule.",
    "Exemplar arms ran raw AND causally standardised; RAW WINS EVERYWHERE on Metro (the reverse of "
    "INSECTS, where standardisation was worth +4 to +6 points). Cause: 27 of the 32 features are "
    "traffic volumes on one common scale, so the raw Euclidean metric is already well-conditioned "
    "and standardisation only up-weights the 5 weather features.",
    "The exemplar byte-frontier on Metro is MONOTONE IMPROVING in store size (407.91 -> 304.02 MAE "
    "from L=250 to L=34,847), the OPPOSITE of the INSECTS gate's finding (75.56 -> 59.75 % accuracy "
    "over the same span). On INSECTS recency was the hidden regime variable; on Metro the regime "
    "clock is fully encoded inside each pair's own 24-lag window, so old exemplars never go stale.",
    "gbdt_tuned's 3,618,071 B of state is 5.44x the 665,000 B exemplar budget it loses to.",
]
json.dump(g, open(os.path.join(DST, "METRO-GATE.json"), "w"), indent=2)
print("notes added;", len(delta), "arms in the leak delta table")
for r in delta[:6] + delta[-4:]:
    print("  %-22s leaky %8.2f  embargo %8.2f  leak worth %+.2f%%" %
          (r["arm"], r["mae_leaky_no_embargo"], r["mae_embargoed_PRIMARY"], 100 * r["leak_worth_rel"]))
