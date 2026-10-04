"""gen-atom-mean-resweep — assemble the n_val calibration tables from the manifests.

⛔ Reads only; every number below comes from an ``atom_manifest.json`` on disk.
⭐ ± is POPULATION sd (ddof=0) throughout, matching gen-val-split-fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
#: (label, n_val, tune_n) in sweep order.  ⭐ the anchor is the shipped pair.
POINTS = [
    ("nval0512_tune0256", 512, 256),
    ("nval0512_tune0512", 512, 512),
    ("nval1024_tune1024", 1024, 1024),
    ("nval2048_tune2048", 2048, 2048),
    ("nval4096_tune4096", 4096, 4096),
    ("nval8192_tune8192", 8192, 8192),
]
SEEDS = (0, 1, 2)
KEYS = ("k", "metric", "normaliser", "scope")


def sd(x):
    return float(np.std(np.asarray(x, float))) if len(x) else float("nan")


def cell_rows(label):
    """One row per (seed, budget) cell that is actually on disk."""
    out = []
    for s in SEEDS:
        p = RUNS / label / f"seed_{s}" / "atom_manifest.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        for tag, b in sorted((m.get("budgets") or {}).items()):
            t = b["tuning"]
            st = t["stability"]
            tr = t["trials"]
            full = np.array([x["val_aggregate"] for x in tr])
            i1 = np.array([x["val_aggregate_ihalf1"] for x in tr])
            i2 = np.array([x["val_aggregate_ihalf2"] for x in tr])
            o = np.argsort(full)[::-1]
            a, bb = int(o[0]), int(o[1])
            out.append({
                "seed": s, "budget": tag,
                "n_fit": int(m["rows"]["fit"]), "n_val": int(m["rows"]["val"]),
                "n_tune": int(t.get("n_tune_realised", -1)),
                "ihalves_agree": bool(st["ihalves_agree"]),
                "halves_agree": bool(st["halves_agree"]),
                "gap": float(st["gap_best_to_second"]),
                "spread": float(st["spread_over_trials"]),
                "selected": tuple(st["selected"][k] for k in KEYS),
                "ihalf1": tuple(st["argmax_ihalf1"][k] for k in KEYS),
                "ihalf2": tuple(st["argmax_ihalf2"][k] for k in KEYS),
                # ⭐ the two independent half-estimates of the top-2 contrast
                "c_full": float(full[a] - full[bb]),
                "c_h1": float(i1[a] - i1[bb]), "c_h2": float(i2[a] - i2[bb]),
                "wall_s": float(b.get("wall_clock_s", float("nan"))),
                "traded": int((m.get("row_split", {}).get("fit_val_trade") or {})
                              .get("fit_rows_traded_away", 0)),
            })
    return out


def seed_rows(label):
    """Per-seed statistics: the knn_proto price and the deployable exponent."""
    out = []
    for s in SEEDS:
        p = RUNS / label / f"seed_{s}" / "atom_manifest.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        kp = m.get("knn_proto_regression") or {}
        e = m.get("exponent") or {}
        out.append({
            "seed": s,
            "n_fit": int(m["rows"]["fit"]), "n_val": int(m["rows"]["val"]),
            "knn_proto": float(kp.get("probe_in_corner_balacc", float("nan"))),
            "knn_proto_published": kp.get("published_probe"),
            "knn_proto_delta": kp.get("delta"),
            "bit_identical": kp.get("bit_identical"),
            "exp_req_k1": ((e.get("request_addressed/k1") or {}).get("slope")),
            "exp_req_ksel": ((e.get("request_addressed/k_selected") or {}).get("slope")),
            "exp_oracle_k1": ((e.get("oracle_feature_addressed/k1") or {}).get("slope")),
            "wall_s": float(m.get("wall_clock_s", float("nan"))),
            "NOT_RUN": m.get("NOT_RUN") or {},
        })
    return out


def sigma_c(rows):
    """σ of the top-2 contrast at HALF size, from |c_h1 − c_h2| (E|X−Y| = 2σ/√π).

    ⚠ Returns ``(sigma, se)``. The estimator averages over **6 cells only**, so its
    own relative standard error is ~30 %. ⛔ A point-to-point wobble inside that band
    is NOT evidence that the n_val^(-1/2) law broke, and must not be scored as one.
    """
    d = np.abs([r["c_h1"] - r["c_h2"] for r in rows])
    if not d.size:
        return float("nan"), float("nan")
    k = np.sqrt(np.pi) / 2 / np.sqrt(2)
    sig = float(d.mean() * k)
    se = float(d.std(ddof=1) / np.sqrt(d.size) * k) if d.size > 1 else float("nan")
    return sig, se


def main():
    print("# gen-atom-mean-resweep — assembled tables\n")

    print("## T1 ⭐ DISAGREEMENT vs n_val  (⛔ the headline)\n")
    print("| n_val | tune_n | rows/half | cells | ⭐ INTERLEAVED disagree | contiguous "
          "disagree | gap best→2nd | median gap | trial spread | realised n_fit | "
          "fit traded | σ_c(half) ± se | ⭐ median gap / σ_c | wall (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    t1 = []
    for label, nv, tn in POINTS:
        rows = cell_rows(label)
        if not rows:
            print(f"| {nv} | {tn} | — | **0 / 6** | ⛔ NOT RUN | — | — | — | "
                  "— | — | — | — | — | — |")
            continue
        n = len(rows)
        dis = sum(1 for r in rows if not r["ihalves_agree"])
        disc = sum(1 for r in rows if not r["halves_agree"])
        g = [r["gap"] for r in rows]
        sp = [r["spread"] for r in rows]
        nf = sorted({r["n_fit"] for r in rows})
        tr = sorted({r["traded"] for r in rows})
        w = sum(r["wall_s"] for r in rows)
        sig, se = sigma_c(rows)
        # ⭐ THE RESOLUTION RATIO: how many half-noise units the winner's margin is.
        # ⛔ Identification needs this > ~2. It is the quantity the sweep is buying.
        res = float(np.median(g)) / sig if sig > 0 else float("nan")
        t1.append((nv, tn, n, dis, sig))
        print(f"| {nv} | {tn} | {rows[0]['n_tune'] // 2} | {n} | **{dis} / {n}** | "
              f"{disc} / {n} | {min(g):.4f}–{max(g):.4f} | {np.median(g):.4f} | "
              f"{min(sp):.4f}–{max(sp):.4f} | "
              f"{'/'.join(f'{x:,}' for x in nf)} | {'/'.join(str(x) for x in tr)} | "
              f"{sig:.4f} ± {se:.4f} | **{res:.2f}** | {w:.0f} |")

    print("\n## T2 The six cells, per n_val (⛔ every cell, never a chosen few)\n")
    print("| n_val | cell | agree (interleaved) | selected | ihalf1 | ihalf2 | "
          "gap | c_full | c_h1 | c_h2 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for label, nv, tn in POINTS:
        for r in cell_rows(label):
            f = lambda t: f"k{t[0]}/{t[1][:3]}/{t[2][:4]}/{t[3][:3]}"  # noqa: E731
            print(f"| {nv}/{tn} | s{r['seed']}/{r['budget']} | "
                  f"{'✅' if r['ihalves_agree'] else '❌'} | {f(r['selected'])} | "
                  f"{f(r['ihalf1'])} | {f(r['ihalf2'])} | {r['gap']:.4f} | "
                  f"{r['c_full']:+.4f} | {r['c_h1']:+.4f} | {r['c_h2']:+.4f} |")

    print("\n## T3 ⭐ THE PRICE — knn_proto (no tuning axis) re-fitted on the "
          "surviving fit rows\n")
    print("| n_val | realised n_fit | % of 32,256 | s0 | s1 | s2 | mean ± sd | "
          "Δ vs published | bit-identical |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, nv, tn in POINTS:
        rows = seed_rows(label)
        if not rows:
            continue
        by = {r["seed"]: r for r in rows}
        # ⛔ a point still running must render as a PARTIAL row, never as a
        # silently-shifted one: a mis-aligned column is a wrong number.
        cells = [(f"{by[s]['knn_proto']:.4f}" if s in by else "⛔ not run")
                 for s in SEEDS]
        v = [r["knn_proto"] for r in rows]
        d = [r["knn_proto_delta"] for r in rows if r["knn_proto_delta"] is not None]
        nf = rows[0]["n_fit"]
        part = "" if len(rows) == len(SEEDS) else f" ⚠ {len(rows)}/{len(SEEDS)} seeds"
        print(f"| {nv}/{tn}{part} | {nf:,} | {100 * nf / 32256:.1f} % | "
              + " | ".join(cells)
              + f" | **{np.mean(v):.4f} ± {sd(v):.4f}** | "
              + (f"{np.mean(d):+.4f}" if d else "—") + " | "
              + ("✅" if all(r["bit_identical"] for r in rows) else "⚠ waived") + " |")

    print("\n## T4 ⭐ THE DEPLOYABLE (request-addressed) EXPONENT — ⛔ SUSPENDED, "
          "NON-ADJUDICATING\n")
    print("| n_val | leg | s0 | s1 | s2 | mean ± sd |")
    print("|---|---|---|---|---|---|")
    for label, nv, tn in POINTS:
        rows = seed_rows(label)
        if not rows:
            continue
        by = {r["seed"]: r for r in rows}
        for leg in ("exp_req_k1", "exp_req_ksel", "exp_oracle_k1"):
            v = [r[leg] for r in rows if r[leg] is not None]
            if not v:
                continue
            cells = [(f"{by[s][leg]:.6f}" if s in by and by[s][leg] is not None
                      else "⛔ not run") for s in SEEDS]
            part = "" if len(v) == len(SEEDS) else f" ⚠ {len(v)}/{len(SEEDS)} seeds"
            print(f"| {nv}/{tn}{part} | `{leg}` | " + " | ".join(cells)
                  + f" | **{np.mean(v):.4f} ± {sd(v):.4f}** |")

    print("\n## T5 NOT RUN / wall clock\n")
    for label, nv, tn in POINTS:
        rows = seed_rows(label)
        if not rows:
            print(f"- **{nv}/{tn}: ⛔ NOT RUN** (no manifest on disk)")
            continue
        nr = {r["seed"]: r["NOT_RUN"] for r in rows if r["NOT_RUN"]}
        print(f"- {nv}/{tn}: seeds {[r['seed'] for r in rows]} · wall "
              f"{sum(r['wall_s'] for r in rows) / 60:.1f} min"
              + (f" · ⛔ NOT_RUN {nr}" if nr else ""))


if __name__ == "__main__":
    main()
