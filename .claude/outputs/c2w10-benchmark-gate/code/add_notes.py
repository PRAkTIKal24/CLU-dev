import json
OUT="/Users/user/Desktop/CHLU/.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json"
g=json.load(open(OUT))
g["notes"]=[
 "VERDICT: criterion4_cleared = FALSE. The byte-matched exemplar store is within 2.0 points of ARF "
 "on the PRIMARY. Per PREREG-C2W10 §2.0 / Head ruling 1 (2026-08-10) the Metro fallback is "
 "PRE-AUTHORIZED and INSECTS is FILED as a registered admissibility finding "
 "('INSECTS is metric-native at matched bytes'), the fifth confirmation of the criterion-4 theorem.",
 "The verdict is robust to the ARF reference: margin = -1.90 vs our 100-tree ARF (the strongest, "
 "MOA-default reference, pre-registered), -0.50 vs our 10-tree ARF (river default), -0.21 vs Souza's "
 "published 77.13. It fires under all three. Against our strongest ARF the margin sits 0.10 points "
 "inside the 2.0-point threshold - stated explicitly rather than buried.",
 "MECHANISM (reportable in its own right): accuracy DECREASES MONOTONICALLY with exemplar-store size "
 "beyond L~500. kNN_S(std): L=250 -> 75.56, 500 -> 76.03, 1000 -> 75.36, 2000 -> 73.39, 5000 -> 68.15, "
 "14782 -> 59.75. Recency IS the hidden temperature variable: a short window is a per-regime "
 "classifier for free. SAM-kNN wins by DISCOVERING this - at L_max=5000 its self-adjusting STM "
 "averages 945 of a permitted 3000 examples. This is the opposite of the scout's §2.3 family, where "
 "a 5000-example window is 2nd of 6 by average rank.",
 "ARF is in permanent reset on this stream, i.e. it is itself a recency mechanism: at the end of the "
 "pass 78 of 100 trees hold exactly 1 node (616 nodes total); a 10-tree probe every 5000 instances "
 "gives sums 302/188/326/148/232/382/198/374. ARF's edge here is churn, not accumulation.",
 "SAM-kNN(1000) = 77.06 vs the BEST fixed window kNN_S(500) = 76.03: an off-the-shelf PERSISTENT "
 "exemplar store buys +1.03 points over pure recency on this stream. Its LTM is selected as the "
 "predicting memory on 14,149 / 79,986 = 17.7 % of instances at L_max=5000. That +1.03 is the "
 "literature's own persistent-vs-episodic contrast, measured, and it is the bar for V1.",
 "SECOND CONDITION (incremental-abrupt-reoccurring bal.) reproduces the same verdict: our No-Change "
 "42.3779 vs published 42.39; our ARF-100 76.225 +/- 0.037 vs published 74.95; best exemplar "
 "SAM-kNN(1000,std) 74.3802; margin -1.845 -> criterion4_cleared = FALSE there too.",
 "PROVENANCE, verified not assumed: river's mirror is BIT-IDENTICAL to the USP Data Stream Repository "
 "original. sha256 of river's incremental_reoccurring_balanced.csv == sha256 of "
 "'USP DS Repository/INSECTS/INSECTS incremental-reoccurring_balanced.csv' extracted from the "
 "repository archive (f267c0fb...), and likewise for incremental-abrupt (c1cd19d4...). No password "
 "was required for extraction.",
 "BLOCKER FOR PREREG-C2W10 §2 / V3: the 'out-of-control' stream (905,145 instances, 24 classes, the "
 "published drift-free null) is NOT in the current USP DS Repository archive (dated 2024-04-16) and "
 "is NOT shipped by river 0.25.0. The archive's INSECTS folder holds exactly 10 files, all 6-class. "
 "V3 as registered has no data source. DECLARED NOT-RUN, not a null.",
 "CORRECTION to c2w10-benchmark-scout §5: river does NOT ship SAM-kNN. river 0.25.0's neighbors "
 "module is {KNNClassifier, KNNRegressor, LazySearch, SWINN}. The 'one-line baseline' estimate was "
 "wrong; the arms here use a Python-3 port of the AUTHORS' OWN reference implementation "
 "(github.com/vlosing/SAMkNN), validated against their published Weather row: our SAM-kNN 21.70 % "
 "error vs published 21.74, our kNN_S 21.68 vs published 21.53 (both L=5000, k=5, distance weights, "
 "raw features).",
 "NAMING: Souza's 'incremental-abrupt-reoccurring (balanced)' is river's variant string "
 "'incremental_abrupt_balanced' and the USP file 'INSECTS incremental-abrupt_balanced.csv' "
 "(sha256-verified identical). Asking river for 'incremental_abrupt_reoccurring_balanced' raises "
 "ValueError.",
 "LABELS ARE NOT 0-BASED: the class column takes values {2,3,4,5,11,12}, 13,331 instances each. Any "
 "harness assuming 0..5 will mis-load the stream.",
 "SCALING: the frozen file is NOT normalised (feature range 0.0020 to 8014.5; 7 of 33 columns are "
 "O(1e2-1e3)). Every exemplar arm was run raw AND with causal prequential standardisation; "
 "standardisation is worth +4.0 to +6.3 points to the kNN arms and the B2 arithmetic uses the max "
 "(F3 anti-hobbling). ARF is scale-invariant and was run on raw features only.",
 "DECIMATION: all three cycles and both change points survive the whole registered ladder "
 "m in {1,2,5,10}; per-cycle counts are in decimation_ladder. HAZARD: decimation compresses the "
 "drift timeline (a change that took 1000 instances takes 1000/m), so adaptation must be reported "
 "PER-INSTANCE-SINCE-CHANGE, never per-stream-position.",
 "REPRODUCTION GATE: no C2W10 cell may consume any number in this file until the project harness's "
 "own loader reproduces the sha256 in reproduction_gate.contract_files. A mismatch is a hard stop.",
]
json.dump(g,open(OUT,"w"),indent=1)
req=["b1_pass","b1_no_change_acc","b1_arf_acc","b2_arms","b2_margin_pts","criterion4_cleared",
     "b3_kappa_per_arf","byte_ledger","change_points","band_map","streams","river_version","notes"]
print("missing keys:",[k for k in req if k not in g])
for k in ["samknn_5000","knn_s_5000","knn_s_1000","arf"]:
    a=g["b2_arms"][k]; print(k, {f:a.get(f) for f in ["acc","kappa","kappa_per","kappa_plus","state_bytes"]})
print("\nbyte ledger:"); print(json.dumps(g["byte_ledger"],indent=1))
print("\nsecond condition:", json.dumps({k:v for k,v in g["second_condition"].items() if k!="arms"},indent=1))
