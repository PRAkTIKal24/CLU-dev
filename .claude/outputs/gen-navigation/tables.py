"""⭐ Every table in `.claude/outputs/gen-navigation.md`, rebuilt FROM DISK.

⛔ No model, no decoder, no classifier is loaded here — this reads
`runs/shapes3d_recon/seed_*/records_s*.json` and `summary.json` only, so a table can
never disagree with the artefacts.  Run:  python tables.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "shapes3d_recon"
SEEDS = [0, 1, 2]
ARMS = ["store_snap", "linear", "deleted_mean", "deleted_scramble"]
FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]


def load(seed):
    with open(RUNS / f"seed_{seed}" / f"records_s{seed}.json") as fh:
        d = json.load(fh)
    for r in d["records"]:
        r["_seed"] = seed
    return d["records"], d["selection"]


REC, SEL = {}, {}
for s in SEEDS:
    REC[s], SEL[s] = load(s)


def ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std())) if v.size else (float("nan"), float("nan"))


def per_seed(fn, sel):
    return [fn([r for r in REC[s] if sel(r)]) for s in SEEDS]


def rate(rows, arm, key):
    return float(np.mean([bool(r["arms"][arm][key]) for r in rows])) if rows else np.nan


def mean(rows, arm, key):
    v = [r["arms"][arm][key] for r in rows]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def nav(rows):
    q = [r for r in rows if r["qualifies_monotone"]]
    if not q:
        return np.nan
    return float(np.mean([
        r["arms"]["store_snap"]["TRAVERSES"] and not r["arms"]["linear"]["TRAVERSES"]
        and not r["arms"]["deleted_mean"]["TRAVERSES"] for r in q]))


def nvr(pool, factor=None):
    v = {int(row["n_values_realised"]) for s in SEEDS for row in SEL[s]
         if row["pool"] == pool and (factor is None or row["factor_name"] == factor)}
    return sorted(v)


def fmt(m, sd, p=3):
    return f"{m:.{p}f} ± {sd:.{p}f}" if np.isfinite(m) else "—"


out = []
A = out.append
A("# gen-navigation — tables, rebuilt from disk by `tables.py`\n")
A("⛔ Every row carries **both controls**. ⛔ Every per-factor row carries "
  "`n_values_realised` (R31, fourth strike).\n")

# ---- 1. headline -----------------------------------------------------------
A("\n## 1. ⭐ THE PRE-REGISTERED HEADLINE — `NAV`, and the marginal traverse rates\n")
A("| pool | n qualifying pairs (per seed) | n_values_realised | **NAV** | "
  "store_snap | **linear** (control 1) | deleted_mean (control 2) | deleted_scramble |")
A("|---|---|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID", "ALL"]:
    def sel(r, pool=pool):
        return True if pool == "ALL" else r["pool"] == pool
    q = [len([r for r in REC[s] if sel(r) and r["qualifies_monotone"]]) for s in SEEDS]
    n = nvr("CORNER") + nvr("GRID") if pool == "ALL" else nvr(pool)
    row = [pool, "/".join(str(x) for x in q), ", ".join(str(x) for x in sorted(set(n)))]
    row.append("**" + fmt(*ms(per_seed(nav, sel)), 4) + "**")
    for a in ARMS:
        row.append(fmt(*ms(per_seed(
            lambda rows, a=a: rate([r for r in rows if r["qualifies_monotone"]],
                                   a, "TRAVERSES"), sel)), 4))
    A("| " + " | ".join(row) + " |")

# ---- 2. the clause decomposition ------------------------------------------
A("\n## 2. ⭐⭐ WHY it is zero — the four clauses of `TRAVERSES`, decomposed\n")
A("(fractions over qualifying pairs; a pair must satisfy **all four**)\n")
A("| pool | arm | endpoints_read_ok | monotone | visits_intermediate | "
  "off_target_stability (pre-reg, vs REQUEST) | …≥0.9 | **TRAVERSES** |")
A("|---|---|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    for a in ARMS:
        def sel(r, pool=pool):
            return r["pool"] == pool and r["qualifies_monotone"]
        cells = [pool, a]
        for key in ["endpoints_read_ok", "monotone", "visits_intermediate"]:
            cells.append(fmt(*ms(per_seed(lambda rows, a=a, k=key: rate(rows, a, k), sel))))
        cells.append(fmt(*ms(per_seed(
            lambda rows, a=a: mean(rows, a, "off_target_stability"), sel))))
        cells.append(fmt(*ms(per_seed(
            lambda rows, a=a: float(np.mean([r["arms"][a]["off_target_stability"] >= 0.9
                                             for r in rows])) if rows else np.nan, sel))))
        cells.append("**" + fmt(*ms(per_seed(
            lambda rows, a=a: rate(rows, a, "TRAVERSES"), sel)), 4) + "**")
        A("| " + " | ".join(cells) + " |")

# ---- 3. addendum diagnostics ----------------------------------------------
A("\n## 3. ⭐ ADDENDUM-1 DIAGNOSTICS (⛔ declared post-hoc, ⛔ never the headline)\n")
A("| pool | arm | endpoint_fidelity (6 factors) | off_target_stability (vs REQUEST) | "
  "**off_target_constancy (vs step 0)** | non-target factors that move (of 5) |")
A("|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    for a in ARMS:
        def sel(r, pool=pool):
            return r["pool"] == pool and r["qualifies_monotone"]
        cells = [pool, a]
        cells.append(fmt(*ms(per_seed(lambda rows, a=a: float(np.mean(
            [0.5 * (r["arms"][a]["endpoint_fidelity_a"] + r["arms"][a]["endpoint_fidelity_b"])
             for r in rows])) if rows else np.nan, sel))))
        cells.append(fmt(*ms(per_seed(
            lambda rows, a=a: mean(rows, a, "off_target_stability"), sel))))
        cells.append("**" + fmt(*ms(per_seed(
            lambda rows, a=a: mean(rows, a, "off_target_constancy"), sel))) + "**")
        cells.append(fmt(*ms(per_seed(
            lambda rows, a=a: mean(rows, a, "n_off_target_factors_that_move"), sel)), 2))
        A("| " + " | ".join(cells) + " |")

# ---- 4. by factor ----------------------------------------------------------
A("\n## 4. By factor — ⛔ read every row WITH its `n_values_realised`\n")
A("| pool | factor | n_values_realised | qualifies | pairs/seed | "
  "store_snap TRAVERSES | linear TRAVERSES | store visits_intermediate | "
  "linear visits_intermediate | identical endpoint codes | interior codes off-endpoints |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    for f in FACTORS:
        rows0 = [r for r in REC[0] if r["pool"] == pool and r["factor_name"] == f]
        n = nvr(pool, f)
        if not rows0:
            A(f"| {pool} | {f} | {n[0] if n else '—'} | ⛔ **NOT APPLICABLE** | 0 | "
              "⛔ NOT RUN | ⛔ NOT RUN | ⛔ NOT RUN | ⛔ NOT RUN | — | — |")
            continue
        def sel(r, pool=pool, f=f):
            return r["pool"] == pool and r["factor_name"] == f
        qual = bool(rows0[0]["qualifies_monotone"])
        cells = [pool, f, ", ".join(str(x) for x in n),
                 "yes" if qual else "⚠ switch-only", str(len(rows0))]
        for a in ["store_snap", "linear"]:
            cells.append(fmt(*ms(per_seed(lambda rows, a=a: rate(rows, a, "TRAVERSES"), sel)), 3)
                         if qual else "⛔ n/a")
        for a in ["store_snap", "linear"]:
            cells.append(fmt(*ms(per_seed(
                lambda rows, a=a: rate(rows, a, "visits_intermediate"), sel)), 3))
        cells.append(fmt(*ms(per_seed(lambda rows: float(np.mean(
            [r["endpoint_codes_identical"] for r in rows])) if rows else np.nan, sel)), 3))
        cells.append(fmt(*ms(per_seed(lambda rows: float(np.mean(
            [r["arms"]["store_snap"]["n_interior_codes_off_endpoints"] for r in rows]))
            if rows else np.nan, sel)), 2))
        A("| " + " | ".join(cells) + " |")

# ---- 5. smoothness ---------------------------------------------------------
A("\n## 5. Smoothness — does it move smoothly or jump?\n")
A("| pool | arm | mean step RMSE | jump_ratio (max/mean, 1 = even) | "
  "path_efficiency (1 = direct) | distinct codes on path | zero-length steps |")
A("|---|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    for a in ARMS:
        def sel(r, pool=pool):
            return r["pool"] == pool
        cells = [pool, a]
        for key in ["mean_step_rmse", "jump_ratio", "path_efficiency"]:
            cells.append(fmt(*ms(per_seed(lambda rows, a=a, k=key: float(np.mean(
                [r["arms"][a]["smoothness"][k] for r in rows
                 if np.isfinite(r["arms"][a]["smoothness"][k])]))
                if rows else np.nan, sel)), 3))
        cells.append(fmt(*ms(per_seed(lambda rows, a=a: float(np.mean(
            [r["arms"][a]["n_distinct_codes_on_path"] for r in rows
             if r["arms"][a]["n_distinct_codes_on_path"] is not None]))
            if rows else np.nan, sel)), 2))
        cells.append(fmt(*ms(per_seed(lambda rows, a=a: float(np.mean(
            [r["arms"][a]["smoothness"]["n_zero_steps"] for r in rows]))
            if rows else np.nan, sel)), 2))
        A("| " + " | ".join(cells) + " |")

# ---- 6. switch-only rows ---------------------------------------------------
A("\n## 6. The switch-only rows (⛔ excluded from the headline by the prereg)\n")
A("| pool | factor | n_values_realised | why switch-only | store_snap clean_switch | "
  "linear clean_switch | deleted_mean clean_switch |")
A("|---|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    for f in FACTORS:
        rows0 = [r for r in REC[0] if r["pool"] == pool and r["factor_name"] == f]
        if not rows0 or rows0[0]["qualifies_monotone"]:
            continue
        def sel(r, pool=pool, f=f):
            return r["pool"] == pool and r["factor_name"] == f
        n = nvr(pool, f)
        why = "CATEGORICAL" if f == "shape" else f"only {n[0]} values realised"
        cells = [pool, f, ", ".join(str(x) for x in n), why]
        for a in ["store_snap", "linear", "deleted_mean"]:
            cells.append(fmt(*ms(per_seed(
                lambda rows, a=a: rate(rows, a, "clean_switch"), sel)), 3))
        A("| " + " | ".join(cells) + " |")

# ---- 7. channel ------------------------------------------------------------
with open(RUNS / "seed_0" / "navigate_manifest_s0.json") as fh:
    man = json.load(fh)
A("\n## 7. ⛔ The realised channel width AT EVERY INTERMEDIATE POINT\n")
A("| arm | per-point width | vs R20's 30-bit design constant | informative bits | "
  "path information (2·K·log2 M + log2(S+1)) |")
A("|---|---|---|---|---|")
for a in ARMS:
    ch = man["channel_by_arm"][a]
    A(f"| {a} | {ch['per_point_bits']:.1f} bits | "
      f"{'⛔ EXCEEDS' if ch['exceeds_design_constant'] else '⭐ within'} | "
      f"{ch['informative_bits']:.1f} | {ch['path_information_bits']:.2f} bits |")

# ---- 8. the fairest possible reading for the store (⛔ post-hoc, declared) ---
A("\n## 8. ⭐ SENSITIVITY — the fairest possible reading for the store arm\n")
A("⛔ Post-hoc, declared. The pre-registered `TRAVERSES` conjunction is progressively "
  "relaxed to check the negative is not an artefact of one clause.\n")
A("| pool | arm | (a) endpoints ∧ monotone ∧ visits ∧ stab≥0.9 (**pre-reg**) | "
  "(b) drop the stability clause | (c) (b) AND condition on endpoints reading correctly | "
  "n pairs in (c)/seed |")
A("|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    for a in ["store_snap", "linear", "deleted_scramble"]:
        def sel(r, pool=pool):
            return r["pool"] == pool and r["qualifies_monotone"]

        def relaxed(rows, a=a):
            if not rows:
                return np.nan
            return float(np.mean([
                r["arms"][a]["endpoints_read_ok"] and r["arms"][a]["monotone"]
                and r["arms"][a]["visits_intermediate"] for r in rows]))

        def cond(rows, a=a):
            k = [r for r in rows if r["arms"][a]["endpoints_read_ok"]]
            if not k:
                return np.nan
            return float(np.mean([r["arms"][a]["monotone"]
                                  and r["arms"][a]["visits_intermediate"] for r in k]))

        def ncond(rows, a=a):
            return float(len([r for r in rows if r["arms"][a]["endpoints_read_ok"]]))

        A("| " + " | ".join([
            pool, a,
            fmt(*ms(per_seed(lambda rows, a=a: rate(rows, a, "TRAVERSES"), sel)), 4),
            fmt(*ms(per_seed(relaxed, sel)), 3),
            fmt(*ms(per_seed(cond, sel)), 3),
            fmt(*ms(per_seed(ncond, sel)), 1)]) + " |")

A("\n### 8b. Head-to-head on the conditional set — ⛔ store vs the laundering control\n")
A("| pool | metric | store_snap | linear | Δ (store − linear) |")
A("|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    for label, key in [("visits_intermediate", "visits_intermediate"),
                       ("monotone", "monotone"),
                       ("off_target_constancy", "off_target_constancy")]:
        def sel(r, pool=pool):
            return r["pool"] == pool and r["qualifies_monotone"]
        vals = {}
        for a in ["store_snap", "linear"]:
            def g(rows, a=a, key=key):
                k = [r for r in rows if r["arms"][a]["endpoints_read_ok"]]
                if not k:
                    return np.nan
                return float(np.mean([float(r["arms"][a][key]) for r in k]))
            vals[a] = per_seed(g, sel)
        d = [x - y for x, y in zip(vals["store_snap"], vals["linear"], strict=True)]
        A("| " + " | ".join([pool, label, fmt(*ms(vals["store_snap"])),
                             fmt(*ms(vals["linear"])), fmt(*ms(d))]) + " |")

# ---- 9. per-factor endpoint fidelity: the ceiling, decomposed ---------------
A("\n## 9. ⭐⭐ PER-FACTOR ENDPOINT FIDELITY of the 30-bit read "
  "(read by the EXTERNAL classifier, ⛔ not the probe)\n")
A("How often the decoded endpoint image shows the factor value the request asked "
  "for. ⛔ This is the **ceiling** on any traversal claim, and it is a "
  "cross-instrument replication of R31's mechanism.\n")
A("| pool | factor | n_values_realised | chance | endpoint fidelity (3 seeds) | vs chance |")
A("|---|---|---|---|---|---|")
for pool in ["CORNER", "GRID"]:
    per = []
    for s_ in SEEDS:
        acc = np.zeros(len(FACTORS))
        n = 0
        for r in REC[s_]:
            if r["pool"] != pool:
                continue
            R = np.asarray(r["arms"]["store_snap"]["readings"])
            for step, req in ((0, r["request_a"]), (-1, r["request_b"])):
                acc += (R[step] == np.asarray(req))
                n += 1
        per.append(acc / n)
    per = np.stack(per)
    for j, f in enumerate(FACTORS):
        n = nvr(pool, f)
        if not n:
            A(f"| {pool} | {f} | — | — | ⛔ NOT RUN (no pair) | — |")
            continue
        card = max(n)
        ch = 1.0 / card if card > 1 else float("nan")
        m, sd = float(per[:, j].mean()), float(per[:, j].std())
        if card == 1:
            # ⛔ R31, fourth strike, in reverse: with ONE realised value there is no
            # chance level to be below — the number is a DETECTION rate.
            tag = "⚠ **n/a** — ONE value realised ⇒ a detection rate, not discrimination"
        else:
            tag = ("⛔ **BELOW chance**" if m < ch - 1e-9
                   else ("≈ chance" if abs(m - ch) < 0.05 else "above chance"))
        note = ""
        A(f"| {pool} | {f} | {', '.join(str(x) for x in n)} | {ch:.3f} | "
          f"{m:.3f} ± {sd:.3f} | {tag}{note} |")

# ---- the figure the headline actually needs --------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
pools = ["CORNER", "GRID"]
x = np.arange(len(pools))


def bars(axis, fns, labels, title, ylab, ylim=(0, 1)):
    w = 0.8 / len(fns)
    for i, (fn, lab) in enumerate(zip(fns, labels, strict=True)):
        m, e = [], []
        for pool in pools:
            def sel(r, pool=pool):
                return r["pool"] == pool and r["qualifies_monotone"]
            a, b = ms(per_seed(lambda rows, fn=fn: fn(rows), sel))
            m.append(a)
            e.append(b)
        axis.bar(x + (i - (len(fns) - 1) / 2) * w, m, w, yerr=e, capsize=3, label=lab)
    axis.set_xticks(x)
    axis.set_xticklabels(pools)
    axis.set_ylim(*ylim)
    axis.set_ylabel(ylab)
    axis.set_title(title, fontsize=9)
    axis.legend(fontsize=7)


def mk_trav(a):
    return lambda rows: rate(rows, a, "TRAVERSES")


def mk_cond(a):
    def f(rows):
        k = [r for r in rows if r["arms"][a]["endpoints_read_ok"]]
        if not k:
            return np.nan
        return float(np.mean([r["arms"][a]["monotone"]
                              and r["arms"][a]["visits_intermediate"] for r in k]))
    return f


def mk_mean(a, key):
    return lambda rows: mean(rows, a, key)


ARMS4 = ARMS
bars(ax[0], [mk_trav(a) for a in ARMS4], ARMS4,
     "(1) PRE-REGISTERED TRAVERSES — zero everywhere", "traverse rate")
bars(ax[1], [mk_cond(a) for a in ARMS4], ARMS4,
     "(2) fairest reading: endpoints correct, stability clause dropped\n"
     "-> the LINEAR CONTROL WINS", "monotone & visits intermediate")
bars(ax[2], [mk_mean(a, "off_target_constancy") for a in ARMS4], ARMS4,
     "(3) off-target constancy (post-hoc): the only axis the store leads,\n"
     "by +0.03", "constancy vs step 0")
bars(ax[3], [mk_mean(a, "endpoint_fidelity_a") for a in ARMS4], ARMS4,
     "(4) ENDPOINT FIDELITY — the ceiling on any traversal claim", "6-factor fidelity")
fig.suptitle("gen-navigation — NAV = 0.0000 (3/3 seeds); the store's landscape does "
             "not out-traverse straight-line interpolation", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "navigation_headline.png", dpi=140)
plt.close(fig)

(ROOT / "tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
