import json, os, hashlib, subprocess, math
import numpy as np

OUT = "/Users/user/Desktop/CHLU/.claude/outputs/c2w10-benchmark-gate"
CACHE = "/Users/user/Desktop/CHLU/.claude/data/c2w10-streams"
M = json.load(open("metrics.json"))
S = json.load(open("structure.json"))

PUB = {"nochange": 40.46, "arf": 77.13}
TOL = 2.0
EX_B = 33 * 4 + 1          # 133 B per stored exemplar (33 float32 + 1 label byte)
CLU_B = 32768 * (13 + 2) * 4   # 1_966_080 B, d=12, n_atoms=512*sqrt(2^12)

def agg(tags):
    a = [M[t]["acc"] for t in tags]
    return dict(mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                sem=float(np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0,
                n_seeds=len(a), per_seed=a)

def arm(tag, state_bytes, note=""):
    r = M[tag]
    return dict(acc=round(r["acc"], 4), acc_window_mean=round(r["acc_window_mean"], 4),
                kappa=round(r["kappa"], 5), kappa_per=round(r["kappa_per"], 5),
                kappa_plus=round(r["kappa_plus"], 5), state_bytes=state_bytes,
                wall_s=round(r["wall_s"], 1), note=note)

# ---------------- ARF reference (registered: n_models=100, MOA default) ----------------
arf_tags = ["arf100_s1", "arf100_s2", "arf100_s3"]
ARF = agg(arf_tags)
arf10_tags = sorted(t for t in M if t.startswith("arf10_s"))
ARF10 = agg(arf10_tags)
arf_bytes100 = json.load(open("arfbytes_100_1.json"))["pickle_bytes"] if os.path.exists("arfbytes_100_1.json") else None
arf_bytes10 = json.load(open("arfbytes_10_1.json"))["pickle_bytes"]

# ---------------- B1 ----------------
b1_nc = M["nochange"]["acc"]
b1_arf = ARF["mean"]
b1_pass = abs(b1_nc - PUB["nochange"]) <= TOL and abs(b1_arf - PUB["arf"]) <= TOL

# ---------------- B2 (registered arm set) ----------------
# exemplar arms are run raw+std; per PREREG §2.2 the arithmetic consumes the MAX (anti-hobbling)
def best(prefix):
    cands = [t for t in M if t.startswith(prefix) and not t.startswith("ABR_")]
    return max(cands, key=lambda t: M[t]["acc"])

reg = {"samknn_5000": best("samknn_5000_"), "knn_s_5000": best("knns_5000_"),
       "knn_s_1000": best("knns_1000_")}
b2_arms = {k: arm(v, EX_B * int(v.split("_")[1]), f"best of raw/std = {v.split('_')[-1]}")
           for k, v in reg.items()}
b2_arms["arf"] = dict(acc=round(ARF["mean"], 4), acc_sd=round(ARF["sd"], 4),
                      acc_sem=round(ARF["sem"], 4), n_seeds=ARF["n_seeds"],
                      per_seed=[round(x, 4) for x in ARF["per_seed"]],
                      acc_window_mean=round(float(np.mean([M[t]["acc_window_mean"] for t in arf_tags])), 4),
                      kappa=round(float(np.mean([M[t]["kappa"] for t in arf_tags])), 5),
                      kappa_per=round(float(np.mean([M[t]["kappa_per"] for t in arf_tags])), 5),
                      kappa_plus=round(float(np.mean([M[t]["kappa_plus"] for t in arf_tags])), 5),
                      state_bytes=arf_bytes100,
                      note="river 0.25.0 ARFClassifier n_models=100 (MOA default); state_bytes = "
                           "measured pickle size of the fitted model, seed 1, protocol 5")
b2_arms["no_change"] = arm("nochange", 4, "persistence; state = 1 stored label (int32)")

best_ex_tag = max(reg.values(), key=lambda t: M[t]["acc"])
b2_best = M[best_ex_tag]["acc"]
b2_margin = b2_best - ARF["mean"]
criterion4_cleared = bool(b2_margin < -TOL)     # B2's arithmetic, not judgement

# supplementary (non-registered) exemplar arms, incl. the stronger SAM-kNN(1000)
supp = {t: arm(t, EX_B * int(t.split("_")[1])) for t in M
        if (t.startswith(("samknn_", "knns_")) and not t.startswith("ABR_") and t not in reg.values())}
best_any_tag = max([t for t in M if t.startswith(("samknn_", "knns_")) and not t.startswith("ABR_")],
                   key=lambda t: M[t]["acc"])
margin_any = M[best_any_tag]["acc"] - ARF["mean"]

# ---------------- second condition ----------------
abr_tags = sorted(t for t in M if t.startswith("ABR_arf100"))
COND2 = {}
if abr_tags:
    A2 = agg(abr_tags)
    ex2 = [t for t in M if t.startswith("ABR_") and ("samknn" in t or "knns" in t)]
    b2t = max(ex2, key=lambda t: M[t]["acc"])
    COND2 = dict(stream="incremental_abrupt_balanced (= Souza incremental-abrupt-reoccurring bal.)",
                 published_no_change=42.39, our_no_change=round(M["ABR_nochange"]["acc"], 4),
                 published_arf=74.95, our_arf=round(A2["mean"], 4), arf_sd=round(A2["sd"], 4),
                 arf_per_seed=[round(x, 4) for x in A2["per_seed"]],
                 best_exemplar_arm=b2t, best_exemplar_acc=round(M[b2t]["acc"], 4),
                 margin_pts=round(M[b2t]["acc"] - A2["mean"], 4),
                 criterion4_cleared=bool(M[b2t]["acc"] - A2["mean"] < -TOL),
                 arms={t: arm(t, EX_B * int(t.split("_")[2])) for t in ex2})

# ---------------- byte ledger ----------------
ledger = dict(
    bytes_per_exemplar=EX_B, bytes_per_exemplar_formula="33 features x 4 B (float32) + 1 B label",
    exemplar_L5000_bytes=EX_B * 5000, exemplar_L5000_MiB=round(EX_B * 5000 / 2**20, 5),
    exemplar_L1000_bytes=EX_B * 1000, exemplar_L1000_MiB=round(EX_B * 1000 / 2**20, 5),
    clu_store_bytes=CLU_B, clu_store_MiB=round(CLU_B / 2**20, 5),
    clu_store_formula="n_atoms=512*sqrt(2^12)=32768 ; bytes = n_atoms*(dim+2)*4, dim=addr+payload=13",
    exemplars_at_clu_budget=CLU_B // EX_B, exemplars_at_clu_budget_bytes=(CLU_B // EX_B) * EX_B,
    exemplars_at_clu_budget_remainder_B=CLU_B - (CLU_B // EX_B) * EX_B,
    clu_over_samknn5000_ratio=round(CLU_B / (EX_B * 5000), 4),
    arf10_measured_state_bytes=arf_bytes10, arf100_measured_state_bytes=arf_bytes100,
    arf_state_bytes_method="pickle.dumps(model, protocol=5) of the fitted river ARFClassifier after "
                           "the full 79,986-instance pass, seed 1; an upper bound on a minimal "
                           "serialisation, reported as measured, not as a minimal encoding",
    arf100_over_samknn5000_ratio=round(arf_bytes100 / (EX_B * 5000), 4) if arf_bytes100 else None,
    no_change_state_bytes=4,
    caveat="ARF is NOT byte-matched to the exemplar arms on this stream: it is the LARGER store. "
           "Any 'ARF is the reference' sentence in the wave must carry this.")

# ---------------- streams ----------------
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

F = json.load(open("facts.json"))
streams = []
for fn, nm, ncls, npub, cps in [
        ("incremental_reoccurring_balanced.csv", "INSECTS incremental-reoccurring (balanced)", 6, 79986, [26568, 53364]),
        ("incremental_abrupt_balanced.csv", "INSECTS incremental-abrupt-reoccurring (balanced)", 6, 79986, [26568, 53364]),
        ("incremental_reoccurring_imbalanced.csv", "INSECTS incremental-reoccurring (imbalanced) [STRETCH, not run]", 6, 452044, [150683, 301365])]:
    p = os.path.join(CACHE, fn)
    k = fn[:-4]
    d = dict(name=nm, file=fn, path=p, sha256=sha(p), size_bytes=os.path.getsize(p),
             n_instances=npub, n_features=33, n_classes=ncls, published_change_points=cps)
    if k in F:
        d.update(n_instances=F[k]["n"], class_histogram=F[k]["hist"],
                 no_change_acc=round(F[k]["nochange"], 4))
    streams.append(d)

pkgs = subprocess.run([".venv/bin/python", "-m", "uv", "pip", "freeze"], capture_output=True, text=True)
if pkgs.returncode != 0:
    pkgs = subprocess.run(["uv", "pip", "freeze", "--python", ".venv"], capture_output=True, text=True)
freeze = [l for l in pkgs.stdout.strip().split("\n") if l]

gate = dict(
    _schema="c2w10 BENCHMARK GATE (spoke S1). Produced by results-analyst, 2026-08-10.",
    _authority="criterion4_cleared is computed ARITHMETICALLY from B2 (margin < -2.0). The Hub rules "
               "on the venue; this file does not.",
    b1_pass=bool(b1_pass), b1_no_change_acc=round(b1_nc, 4), b1_arf_acc=round(b1_arf, 4),
    b1_published=dict(no_change=PUB["nochange"], arf=PUB["arf"], source="Souza et al. 2020, DMKD 34(6), Table 5"),
    b1_deltas=dict(no_change=round(b1_nc - PUB["nochange"], 4), arf=round(b1_arf - PUB["arf"], 4), tolerance=TOL),
    b2_arms=b2_arms,
    b2_supplementary_arms=supp,
    b2_best_registered_arm=best_ex_tag, b2_margin_pts=round(b2_margin, 4),
    b2_best_any_arm=best_any_tag, b2_margin_pts_any_arm=round(margin_any, 4),
    criterion4_cleared=criterion4_cleared,
    criterion4_rule="cleared iff best_exemplar_acc - arf_acc < -2.0 points",
    criterion4_robustness=dict(
        vs_our_arf100=round(b2_margin, 4),
        vs_our_arf10=round(b2_best - ARF10["mean"], 4),
        vs_published_arf_77_13=round(b2_best - PUB["arf"], 4),
        vs_our_arf100_best_seed=round(b2_best - max(ARF["per_seed"]), 4),
        vs_our_arf100_worst_seed=round(b2_best - min(ARF["per_seed"]), 4),
        note="the tripwire fires under every ARF reference tried; the margin against our STRONGEST "
             "ARF (100 trees) is the one closest to the threshold, at 0.10 pts inside it"),
    b3_kappa_per_arf=round(float(np.mean([M[t]["kappa_per"] for t in arf_tags])), 5),
    b3_pass=bool(np.mean([M[t]["kappa_per"] for t in arf_tags]) > 0),
    b3_kappa_per_all_arms={t: round(M[t]["kappa_per"], 5) for t in sorted(M)},
    byte_ledger=ledger,
    change_points=S["change_points"], cycles=S["cycles"],
    band_map=dict(_label="OUR CONSTRUCTION, derived from Souza's verbatim schedule text; band-level "
                         "alignment is NOT published (PREREG-C2W10 §2.2).",
                  B=5, bands=S["bands"], revisit_pairing=S["revisit_pairing"]),
    decimation_ladder=S["decimation"],
    second_condition=COND2,
    streams=streams,
    river_version="0.25.0",
    scratch_venv=dict(path=".claude/scratch/c2w10/.venv", python="3.12.9", freeze=freeze),
    reproduction_gate=dict(
        rule="NO C2W10 cell may consume a baseline number from this file until the project harness's "
             "own loader reproduces the sha256 of the stream file it reads. A mismatch is a HARD "
             "STOP, not a tolerance.",
        contract_files={s["file"]: s["sha256"] for s in streams},
        loader_spec="headerless CSV, 79,986 rows x 34 comma-separated columns; columns 0-32 = "
                    "features f1..f33 (float64 as written), column 33 = class label in "
                    "{2,3,4,5,11,12} (integers, NOT 0-based); row order is the stream order; no "
                    "shuffling, no normalisation applied to the frozen file."),
    notes=[],
)
json.dump(gate, open(os.path.join(OUT, "BENCHMARK-GATE.json"), "w"), indent=1)
print(json.dumps({k: gate[k] for k in ["b1_pass", "b1_no_change_acc", "b1_arf_acc", "b1_deltas",
                                       "b2_best_registered_arm", "b2_margin_pts", "b2_best_any_arm",
                                       "b2_margin_pts_any_arm", "criterion4_cleared",
                                       "criterion4_robustness", "b3_kappa_per_arf", "b3_pass"]}, indent=1))

# ---------------- revisit diagnostic (cycle 3 vs cycle 1, bands 0-3 only) ----------------
BS = json.load(open("bandstats.json"))
arms_b = [k for k in BS[0] if k not in ("cycle", "band", "start", "end", "entropy_bits", "majority_pct")]
rev = {}
for a in arms_b:
    c1 = [r[a] for r in BS if r["cycle"] == 1 and r["band"] < 4]
    c3 = [r[a] for r in BS if r["cycle"] == 3 and r["band"] < 4]
    rev[a] = dict(cycle1_bands0_3=round(float(np.mean(c1)), 3), cycle3_bands0_3=round(float(np.mean(c3)), 3),
                  revisit_delta_pts=round(float(np.mean(c3) - np.mean(c1)), 3),
                  per_band_delta=[round(y - x, 3) for x, y in zip(c1, c3)])
g = json.load(open(os.path.join(OUT, "BENCHMARK-GATE.json")))
g["band_stats"] = BS
g["degenerate_band_warning"] = dict(
    finding="band 4 (the terminal band) of EVERY cycle is persistence-trivial and at ceiling for every "
            "arm. Class entropy collapses to 0.84-1.05 bits (vs 2.07-2.37 in bands 0-3), the majority "
            "class is 54-73%, No-Change scores 91.25 / 71.32 / 90.78 %, and every fitted arm scores "
            "97-99%. The longest single-label run in the stream is 3,597 instances (starts at index "
            "76,389, label 5). Contiguous stream regions where the No-Change windowed accuracy exceeds "
            "90% are [23634,26753], [49007,49209], [53269,53561], [77043,79985] - 8.30% of the stream, "
            "and the first and third STRADDLE the published change points.",
    consequence="PREREG-C2W10 §2.2 anchors retention R(b) on 'the last 1000 instances of that band's "
                "first visit' and pairs (c1,b) with (c2,4-b). For b=4 that anchor sits inside the "
                "degenerate zone and is paired against a non-degenerate band. R(4) and A(4) are "
                "UNINTERPRETABLE as registered.",
    recommendation="restrict every retention/adaptation claim to bands 0-3 (12 of 15 bands, ~80% of "
                   "the stream), or re-band on a criterion that excludes the low-entropy tail; report "
                   "the per-band No-Change accuracy in EVERY band-level table (Zliobaite's rule, "
                   "applied per band rather than per stream).")
g["revisit_diagnostic"] = dict(
    definition="mean accuracy over bands 0-3 of cycle 3 minus the same over cycle 1, on the SAME "
               "prequential pass; band 4 excluded as degenerate. A positive value is free retention.",
    arms=rev,
    reading="no existing arm shows a material cycle-3-over-cycle-1 gain on this stream, persistent "
            "exemplar store included. This is the bar the wave's V1 persistent-vs-episodic contrast "
            "has to clear, and it is measured here at ~0.")
json.dump(g, open(os.path.join(OUT, "BENCHMARK-GATE.json"), "w"), indent=1)
print("\nrevisit diagnostic:"); print(json.dumps(rev, indent=1))
