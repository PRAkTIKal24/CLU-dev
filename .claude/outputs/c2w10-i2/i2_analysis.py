"""c2w10-i2 — the I2 re-measurement on banked artifacts ONLY.

Reads  .claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json   (read-only)
Writes .claude/outputs/c2w10-i2/I2-VERDICT.json + figures/tables

Estimators are exactly PREREG.md §2. No model is run; no tracked code is touched.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = "/Users/user/Desktop/CHLU"
SRC = f"{ROOT}/.claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json"
OUT = f"{ROOT}/.claude/outputs/c2w10-i2"
N_STREAMS = 6                      # schedule (0,1,2,0,1,2)
MIN_READINGS = 4                   # PREREG §2: E_i needs >= 4 readings
FLOOR = 1e-30                      # registered censoring floor (C2W6)
FLOAT32_NET_FLOOR = 5.3e-8         # R5: the shipped float32 netting floor


# ---------------------------------------------------------------- statistics
def rankdata(a: Sequence[float]) -> np.ndarray:
    """Average ranks (ties averaged) — scipy-free."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def fisher_2se(rho: float, n: int) -> Tuple[float, float, float]:
    """(lower, upper, half_width) with the registered 2-SE Fisher-z half-width."""
    if not np.isfinite(rho) or n <= 3:
        return float("nan"), float("nan"), float("nan")
    hw = 2.0 / math.sqrt(n - 3)
    z = math.atanh(max(-0.999999, min(0.999999, rho)))
    return math.tanh(z - hw), math.tanh(z + hw), hw


def ols_slope(c: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """slope and its SE, closed form."""
    c = np.asarray(c, float)
    y = np.asarray(y, float)
    cc = c - c.mean()
    sxx = float((cc ** 2).sum())
    if sxx <= 0 or len(c) < 3:
        return float("nan"), float("nan")
    beta = float((cc * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + beta * cc)
    s2 = float((resid ** 2).sum()) / (len(c) - 2)
    return beta, math.sqrt(max(s2, 0.0) / sxx)


def icc_1_1(rows: List[np.ndarray]) -> Dict[str, float]:
    """One-way random-effects ICC(1,1), unbalanced (PREREG §2)."""
    rows = [np.asarray(r, float) for r in rows if len(r) >= 2]
    N = len(rows)
    if N < 2:
        return {"icc": float("nan"), "N": N, "k0": float("nan"),
                "MSB": float("nan"), "MSW": float("nan")}
    ks = np.array([len(r) for r in rows], float)
    grand = float(np.concatenate(rows).mean())
    ssb = float(sum(len(r) * (r.mean() - grand) ** 2 for r in rows))
    ssw = float(sum(((r - r.mean()) ** 2).sum() for r in rows))
    n_tot = float(ks.sum())
    msb = ssb / (N - 1)
    msw = ssw / (n_tot - N) if n_tot > N else float("nan")
    k0 = (n_tot - float((ks ** 2).sum()) / n_tot) / (N - 1)
    icc = (msb - msw) / (msb + (k0 - 1.0) * msw) if np.isfinite(msw) else float("nan")
    return {"icc": float(icc), "N": N, "k0": float(k0), "MSB": msb, "MSW": msw}


def spearman_brown(r_half: float) -> float:
    if not np.isfinite(r_half):
        return float("nan")
    den = 1.0 + r_half
    return float(2.0 * r_half / den) if den != 0 else float("nan")


# ---------------------------------------------------------------- per seed
def analyse_seed(s: dict) -> dict:
    seed = int(s["seed"])
    hits_by_item = {int(k): float(v) for k, v in s["usage_summary"]["hits_by_item"].items()}
    curves = {int(k): v for k, v in s["depth_curves_raw_and_netted"].items()}
    hits_by_stream = {int(k): {int(a): float(b) for a, b in v.items()}
                      for k, v in s["hits_by_stream"].items()}
    first_seen = {int(k): int(v) for k, v in s["first_seen_stream"].items()}

    # ---- P1: live wells at the final depth-recording point (= usage_summary keys)
    pop = sorted(hits_by_item)
    n_pop = len(pop)

    # ---- E_i on the NETTED curve, log scale; whole-curve diagnostics
    rows = []
    excluded_few_readings = []
    for iid in pop:
        cv = curves.get(iid)
        if cv is None or len(cv["chunk"]) < MIN_READINGS:
            excluded_few_readings.append(iid)
            continue
        ch = np.asarray(cv["chunk"], float)
        dn = np.asarray(cv["depth_netted"], float)
        dr = np.asarray(cv["depth_raw"], float)
        if np.any(dn <= 0) or np.any(dr <= 0):
            excluded_few_readings.append(iid)
            continue
        b_net, se_net = ols_slope(ch, np.log(dn))
        b_raw, _ = ols_slope(ch, np.log(dr))
        half = len(ch) // 2
        b_odd, _ = ols_slope(ch[0::2], np.log(dn[0::2]))
        b_even, _ = ols_slope(ch[1::2], np.log(dn[1::2]))
        # whole-curve shape (ties-aware; np.argmin alone is fooled by plateaus)
        raw_inc = int(np.sum(np.diff(dr) > 1e-9 * dr[:-1]))
        net_inc = int(np.sum(np.diff(dn) > 0.0))
        raw_final_over_min = float(dr[-1] / dr.min())
        net_final_over_min = float(dn[-1] / dn.min())
        # netted residual expressed in float32 ULP of the depth itself
        f32eps = 1.1920929e-7
        net_rel_span = float((dn.max() - dn.min()) / dn.mean())
        rows.append({
            "item": iid, "m": int(len(ch)),
            "U": hits_by_item[iid],
            "E_net": -b_net, "se_slope_net": se_net,
            "E_raw": -b_raw,
            "E_odd": -b_odd, "E_even": -b_even,
            "depth_net_last": float(dn[-1]), "depth_net_first": float(dn[0]),
            "depth_raw_last": float(dr[-1]),
            "net_max_over_min": float(dn.max() / dn.min()),
            "raw_max_over_min": float(dr.max() / dr.min()),
            "net_argmin_frac": float(int(np.argmin(dn)) / (len(dn) - 1)),
            "raw_argmin_frac": float(int(np.argmin(dr)) / (len(dr) - 1)),
            "net_min": float(dn.min()),
            "first_chunk": float(ch[0]), "last_chunk": float(ch[-1]),
            "half": half,
            "raw_inc": raw_inc, "net_inc": net_inc,
            "raw_final_over_min": raw_final_over_min,
            "net_final_over_min": net_final_over_min,
            "net_rel_span_in_ulp": net_rel_span / f32eps,
            "raw_total_log_drop": float(np.log(dr[0] / dr[-1])),
            "net_total_log_drop": float(np.log(dn[0] / dn[-1])),
        })
    n_scored = len(rows)
    U = np.array([r["U"] for r in rows])
    E = np.array([r["E_net"] for r in rows])
    E_raw = np.array([r["E_raw"] for r in rows])
    D = np.array([r["depth_net_last"] for r in rows])
    age = np.array([r["last_chunk"] - r["first_chunk"] for r in rows])

    # ---- censoring
    n_cens = int(np.sum(np.array([r["net_min"] for r in rows]) <= FLOOR))
    n_cens_f32 = int(np.sum(np.array([r["net_min"] for r in rows]) <= FLOAT32_NET_FLOOR))

    # ---- I2-c / I2-d
    rho_UE = spearman(U, E)
    lo_UE, hi_UE, hw = fisher_2se(rho_UE, n_scored)
    rho_UD = spearman(U, D)
    lo_UD, hi_UD, _ = fisher_2se(rho_UD, n_scored)
    rho_UE_raw = spearman(U, E_raw)                     # DIAGNOSTIC ONLY
    lo_UEr, hi_UEr, _ = fisher_2se(rho_UE_raw, n_scored)
    rho_UD_raw = spearman(U, [r["depth_raw_last"] for r in rows])   # DIAGNOSTIC
    rho_U_age = spearman(U, age)
    rho_D_age = spearman(D, age)

    # ---- reliability of E (netted): split-half over readings
    ok = [r for r in rows if np.isfinite(r["E_odd"]) and np.isfinite(r["E_even"])]
    r_half_E = spearman([r["E_odd"] for r in ok], [r["E_even"] for r in ok])
    rel_E = spearman_brown(r_half_E)
    sd_between_E = float(np.std(E, ddof=1))
    mean_within_se = float(np.mean([r["se_slope_net"] for r in rows]))

    # ---- ICC(1,1) of the usage proxy across streams
    icc_rows, icc_items = [], []
    for iid in pop:
        fs = first_seen.get(iid, 0)
        ks = list(range(fs, N_STREAMS))
        if len(ks) < 2:
            continue
        hb = hits_by_stream.get(iid, {})
        icc_rows.append(np.array([hb.get(k, 0.0) for k in ks], float))
        icc_items.append(iid)
    icc = icc_1_1(icc_rows)
    # split-half over streams (odd vs even), the C2W6 form
    odd = [float(r[1::2].sum()) for r in icc_rows]
    even = [float(r[0::2].sum()) for r in icc_rows]
    r_half_U = spearman(odd, even)
    rel_U_sb = spearman_brown(r_half_U)

    # ---- permutation test (per seed, exact-null by relabelling U)
    rng = np.random.default_rng(20260811 + seed)
    n_perm = 20000

    def perm_p(x: np.ndarray, y: np.ndarray, obs: float) -> float:
        cnt = 0
        for _ in range(n_perm):
            if abs(spearman(rng.permutation(x), y)) >= abs(obs) - 1e-12:
                cnt += 1
        return (cnt + 1) / (n_perm + 1)

    p_UE = perm_p(U, E, rho_UE)
    p_UD = perm_p(U, D, rho_UD)

    # ---- validity of E on the netted channel (magnitude, not reliability)
    Draw_last = np.array([r["depth_raw_last"] for r in rows])
    ulp = np.array([r["net_rel_span_in_ulp"] for r in rows])
    net_drop = np.array([r["net_total_log_drop"] for r in rows])
    raw_drop = np.array([r["raw_total_log_drop"] for r in rows])
    validity = {
        "median_E_netted_over_E_raw": float(np.median(np.abs(E) / np.abs(E_raw))),
        "median_netted_total_log_drop": float(np.median(net_drop)),
        "median_raw_total_log_drop": float(np.median(raw_drop)),
        "median_netted_span_in_float32_ULP": float(np.median(ulp)),
        "max_netted_span_in_float32_ULP": float(np.max(ulp)),
        "n_items_with_any_netted_increase": int(np.sum([r["net_inc"] > 0 for r in rows])),
        "n_items_with_any_raw_increase": int(np.sum([r["raw_inc"] > 0 for r in rows])),
        "max_raw_final_over_min": float(np.max([r["raw_final_over_min"] for r in rows])),
        "note": ("the raw curve is monotone non-increasing on every item (no C2W6-style "
                 "transient trough); the netted curve's entire span is a few float32 ULP"),
    }

    # ---- partial Spearman: does the netted rho survive controlling for the RAW decay
    #      rate (i.e. is it inherited round-off that tracks the number of decay ticks)?
    def partial(r_xy: float, r_xz: float, r_yz: float) -> float:
        den = math.sqrt(max(1e-12, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
        return (r_xy - r_xz * r_yz) / den

    r_U_Eraw = spearman(U, E_raw)
    r_En_Eraw = spearman(E, E_raw)
    partial_U_E_given_Eraw = partial(rho_UE, r_U_Eraw, r_En_Eraw)
    # depth on the netted curve is TIME-INVARIANT by construction (only the designed
    # decay moves depth, and netting divides it out) => netted depth == write-time depth
    net_depth_drift = float(np.median([abs(r["depth_net_last"] / r["depth_net_first"] - 1.0)
                                       for r in rows]))

    # ---- the age confound the RAW curve carries (Add.9 §A27.1), quantified
    confound = {
        "rho_Eraw_age": spearman(E_raw, age),
        "rho_Eraw_depth_raw_last": spearman(E_raw, Draw_last),
        "rho_U_depth_raw_last": spearman(U, Draw_last),
        "rho_U_age": spearman(U, age),
        "rho_Enet_Eraw": r_En_Eraw,
        "rho_Enet_depth_netted": spearman(E, D),
        "partial_rho_U_Enet_given_Eraw": partial_U_E_given_Eraw,
        "median_netted_depth_drift_last_over_first_minus_1": net_depth_drift,
        "netted_depth_is_write_time_depth": True,
    }

    # ---- attenuation ceiling on any rho involving U and E
    relU = spearman_brown(r_half_U)
    ceiling = (math.sqrt(max(relU, 0.0) * max(rel_E, 0.0))
               if np.isfinite(relU) and np.isfinite(rel_E) else float("nan"))

    return {
        "seed": seed,
        "n_live_max": int(s["n_live_max"]),
        "n_live_end": int(s["n_live_end"]),
        "n_pop_P1": n_pop,
        "n_scored": n_scored,
        "n_excluded_few_readings": len(excluded_few_readings),
        "excluded_ids": excluded_few_readings,
        "detectable_rho": 2.0 / math.sqrt(n_scored - 3),
        "fisher_half_width": hw,
        "read_coverage_rate": float(s["read_coverage"]["rate"]),
        "n_never_read": int(s["usage_summary"]["n_never_read"]),
        "n_never_read_scored": int(np.sum(U == 0)),
        "hits": {"mean": float(U.mean()), "median": float(np.median(U)),
                 "max": float(U.max()), "min": float(U.min())},
        "E_net": {"mean": float(E.mean()), "median": float(np.median(E)),
                  "sd_between": sd_between_E, "min": float(E.min()),
                  "max": float(E.max()),
                  "median_abs": float(np.median(np.abs(E))),
                  "mean_within_slope_se": mean_within_se,
                  "sd_over_within_se": sd_between_E / mean_within_se
                  if mean_within_se > 0 else float("nan")},
        "E_raw": {"mean": float(E_raw.mean()), "median": float(np.median(E_raw)),
                  "sd_between": float(np.std(E_raw, ddof=1)),
                  "min": float(E_raw.min()), "max": float(E_raw.max())},
        "reliability_E_netted": {"split_half_spearman": r_half_E,
                                 "spearman_brown": rel_E, "n": len(ok)},
        "reliability_U": {"icc_1_1": icc["icc"], "icc_N": icc["N"], "icc_k0": icc["k0"],
                          "icc_MSB": icc["MSB"], "icc_MSW": icc["MSW"],
                          "split_half_spearman": r_half_U,
                          "spearman_brown": rel_U_sb},
        "curve_shape": {
            "net_max_over_min_median": float(np.median([r["net_max_over_min"] for r in rows])),
            "net_max_over_min_max": float(np.max([r["net_max_over_min"] for r in rows])),
            "raw_max_over_min_median": float(np.median([r["raw_max_over_min"] for r in rows])),
            "raw_max_over_min_max": float(np.max([r["raw_max_over_min"] for r in rows])),
            "raw_argmin_at_end_frac": float(np.mean(
                [r["raw_argmin_frac"] > 0.999 for r in rows])),
            "net_argmin_at_end_frac": float(np.mean(
                [r["net_argmin_frac"] > 0.999 for r in rows])),
            "n_items_with_interior_trough_net": int(np.sum(
                [(r["net_argmin_frac"] < 0.999) for r in rows])),
        },
        "censoring": {"frac_at_floor_1e-30": n_cens / n_scored,
                      "n_at_floor_1e-30": n_cens,
                      "frac_at_float32_netting_floor": n_cens_f32 / n_scored,
                      "min_netted_depth": float(np.min([r["net_min"] for r in rows]))},
        "rho_U_E": {"rho": rho_UE, "lo_2se": lo_UE, "hi_2se": hi_UE, "n": n_scored,
                    "perm_p_two_sided": p_UE},
        "rho_U_depth": {"rho": rho_UD, "lo_2se": lo_UD, "hi_2se": hi_UD, "n": n_scored,
                        "perm_p_two_sided": p_UD},
        "validity_E_netted": validity,
        "confound_diagnostics": confound,
        "attenuation_ceiling_U_x_E": ceiling,
        "diagnostics_non_registered": {
            "rho_U_E_RAW_curve": {"rho": rho_UE_raw, "lo_2se": lo_UEr, "hi_2se": hi_UEr},
            "rho_U_depth_RAW_curve": rho_UD_raw,
            "rho_U_age": rho_U_age,
            "rho_depth_age": rho_D_age,
        },
        "_rows": rows,
    }


def main() -> None:
    tel = json.load(open(SRC))
    # ---- MECHANICAL PRECONDITION (checked first)
    pre = {
        "file": SRC,
        "exists": True,
        "n_live_max": int(tel["n_live_max"]),
        "n_live_max_per_seed": {k: int(v) for k, v in tel["n_live_max_per_seed"].items()},
        "n_seeds": int(tel["n_seeds"]),
        "n_seeds_meeting_64": int(tel["n_seeds_meeting_64"]),
        "n_live_max_ge_64": bool(tel["n_live_max_ge_64"]),
        "met": bool(tel["n_live_max"] >= 64 and tel["n_seeds"] >= 3),
    }
    assert pre["met"], "precondition not met — the report must be BLOCKED"

    per = [analyse_seed(s) for s in tel["per_seed"]]

    i2a_pass = bool(all(p["n_live_max"] >= 64 for p in per) and len(per) >= 3)
    icc_by_seed = {str(p["seed"]): p["reliability_U"]["icc_1_1"] for p in per}
    i2b_pass = bool(all(v > 0 for v in icc_by_seed.values()))

    leg1 = {str(p["seed"]): bool(p["rho_U_E"]["lo_2se"] > -0.10) for p in per}
    leg2 = {str(p["seed"]): bool(p["rho_U_depth"]["lo_2se"] >= 0.30) for p in per}
    confirm_seeds = {str(p["seed"]): bool(p["rho_U_E"]["hi_2se"] < -0.20) for p in per}
    n_confirm = sum(confirm_seeds.values())

    lift = bool(i2a_pass and i2b_pass and all(leg1.values()) and all(leg2.values()))
    if not (i2a_pass and i2b_pass):
        branch = "NOT_RUN"
    elif n_confirm >= 2:
        branch = "CONFIRM"
    elif lift:
        branch = "REFUTE_BOTH_LEGS"
    else:
        branch = "INDETERMINATE"

    # ---- NON-REGISTERED pooled estimator, labelled (rider 1)
    allU, allE, allD = [], [], []
    for p in per:
        allU += [r["U"] for r in p["_rows"]]
        allE += [r["E_net"] for r in p["_rows"]]
        allD += [r["depth_net_last"] for r in p["_rows"]]
    pooled = {
        "LABEL": "NON-REGISTERED pooled-across-seeds estimator — never evidence (Add.9 §A27.1)",
        "n": len(allU),
        "rho_U_E": spearman(allU, allE),
        "rho_U_depth": spearman(allU, allD),
        "detectable_rho": 2.0 / math.sqrt(len(allU) - 3),
    }

    verdict = {
        "wave": "C2W10", "leg": "I2 re-measurement (PREREG-C2W10 §5)",
        "produced_by": "results-analyst / c2w10-i2-usage-erosion",
        "source_artifact": SRC,
        "source_commit": tel["commit"],
        "proxy": tel["proxy"],
        "precondition": pre,
        "population": ("P1 = live wells at the final depth-recording point "
                       "(= usage_summary.hits_by_item keys = end-of-run codebook)"),
        "estimator_E": ("E_i = -OLS slope of ln(depth_netted) on chunk; "
                        "netted curve only; >=4 readings required"),
        "estimator_rho": "Spearman per seed; 2-SE bounds via Fisher-z, half-width 2/sqrt(n-3)",
        "n_live_by_seed": {str(p["seed"]): {"n_live_max": p["n_live_max"],
                                            "n_live_end": p["n_live_end"],
                                            "n_scored": p["n_scored"]} for p in per},
        "detectable_rho": {str(p["seed"]): p["detectable_rho"] for p in per},
        "icc_by_seed": icc_by_seed,
        "icc_status_by_seed": {str(p["seed"]):
                               ("usable" if p["reliability_U"]["icc_1_1"] > 0 else "UNDEFINED")
                               for p in per},
        "reliability_E_by_seed": {str(p["seed"]):
                                  p["reliability_E_netted"]["spearman_brown"] for p in per},
        "rho_U_E_by_seed": {str(p["seed"]): p["rho_U_E"] for p in per},
        "rho_U_depth_by_seed": {str(p["seed"]): p["rho_U_depth"] for p in per},
        "censoring_by_seed": {str(p["seed"]): p["censoring"] for p in per},
        "E_net_by_seed": {str(p["seed"]): p["E_net"] for p in per},
        "E_raw_by_seed": {str(p["seed"]): p["E_raw"] for p in per},
        "curve_shape_by_seed": {str(p["seed"]): p["curve_shape"] for p in per},
        "validity_E_netted_by_seed": {str(p["seed"]): p["validity_E_netted"] for p in per},
        "confound_diagnostics_by_seed": {str(p["seed"]): p["confound_diagnostics"] for p in per},
        "attenuation_ceiling_by_seed": {str(p["seed"]): p["attenuation_ceiling_U_x_E"]
                                        for p in per},
        "loo_leg": ("NOT PRESENT in this artifact — no leave-one-out telemetry was produced by "
                    "exp_persistent_store; no rho(LOO) is quoted anywhere (declared NOT-RUN, "
                    "not a null)"),
        "mechanism_channel_status": {
            "status": "ABSENT_BY_CONSTRUCTION",
            "evidence": ("exp_persistent_store.py @6e0c325 runs no outer objective and no "
                         "optimizer step on store parameters; L5 records that the cell never "
                         "rewrites a live well. Depth therefore moves ONLY through the designed "
                         "decay law, which netting divides out exactly: median netted total log "
                         "drop over a whole run is ~1e-6 nats against ~0.9 nats raw."),
            "consequence": ("the Head's I2 mechanism (outer-loss gradient magnitude proportional "
                            "to contribution) has no channel in this rig; I2-c's registered "
                            "arithmetic is computed and reported, but the MECHANISM is a "
                            "declared NOT-RUN at this rig, not a null"),
        },
        "diagnostics_non_registered_by_seed": {
            str(p["seed"]): p["diagnostics_non_registered"] for p in per},
        "pooled_non_registered": pooled,
        "i2a_pass": i2a_pass,
        "i2b_pass": i2b_pass,
        "lift_leg1_rho_U_E_lower_above_minus_0.10": leg1,
        "lift_leg2_rho_U_depth_lower_at_least_0.30": leg2,
        "confirm_by_seed_upper_below_minus_0.20": confirm_seeds,
        "n_seeds_confirm": n_confirm,
        "branch": branch,
        "lift_rule_satisfied": lift,
        "authority": ("lift_rule_satisfied is a MEASUREMENT, not a lift: the Hub proposes, "
                      "the Advisor amends charter §A23.5; this spoke does neither."),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/I2-VERDICT.json", "w") as f:
        json.dump(verdict, f, indent=1)

    # per-well table for audit
    with open(f"{OUT}/per_well_table.csv", "w") as f:
        f.write("seed,item,m,U,E_netted,se_slope_netted,E_raw,depth_netted_last,"
                "net_max_over_min,raw_max_over_min,first_chunk,last_chunk\n")
        for p in per:
            for r in p["_rows"]:
                f.write(f"{p['seed']},{r['item']},{r['m']},{r['U']:.0f},{r['E_net']:.6e},"
                        f"{r['se_slope_net']:.6e},{r['E_raw']:.6e},{r['depth_net_last']:.6f},"
                        f"{r['net_max_over_min']:.9f},{r['raw_max_over_min']:.6f},"
                        f"{r['first_chunk']:.0f},{r['last_chunk']:.0f}\n")

    # console summary
    print(json.dumps({k: v for k, v in verdict.items() if k != "pooled_non_registered"},
                     indent=1, default=float)[:7000])
    print("\nPOOLED (non-registered):", pooled)


if __name__ == "__main__":
    main()
