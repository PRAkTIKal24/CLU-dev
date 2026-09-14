"""The reported tables: the bit-identity regression, both columns by kind, the
seen/unseen decomposition, and the R27.3 two-leg comparison.

⛔ Every number is read from a manifest or from the persisted per-request scores;
⛔ nothing is inferred from an aggregate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-arms-a7-wiring")
RUNS = ROOT / "runs"
PUB = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-baselines/runs/shapes3d_recon")
SEEDS = (0, 1, 2)


def auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(neg.size, bool), np.ones(pos.size, bool)])
    order = np.argsort(s, kind="mergesort")
    sr, r = s[order], np.empty(s.size, float)
    ranks = np.arange(1.0, s.size + 1)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = ranks[i:j + 1].mean()
        i = j + 1
    return float((r[y].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def load(leg, seed):
    d = RUNS / leg / f"seed_{seed}"
    m = json.loads((d / "baselines_manifest.json").read_text())
    sc = json.loads((d / "refusal_scores.json").read_text())
    return m, sc


def variants(man):
    out = {}
    for v, vv in (man["arms"].get("knn", {}).get("variants", {}) or {}).items():
        out[v] = vv["refusal"]
    for v, vv in (man["arms"].get("lace", {}).get("samplers", {}) or {}).items():
        out[f"lace_{v}"] = vv["refusal"]
    return out


def kindval(ref, kind, control="B"):
    if control == "B":
        d = (ref.get("auroc_c_vs_b_by_kind") or {}).get(kind)
    else:
        d = ((ref.get("a7_controlled") or {}).get("auroc_by_kind_PER_CONTROL")
             or {}).get(control, {}).get(kind)
    return None if not d or "auroc" not in d else d["auroc"]


def ms(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    return (float(a.mean()), float(a.std()), a.tolist()) if a.size else (None, None, [])


def main():
    print("=" * 100)
    print("1. ⭐ BIT-IDENTITY REGRESSION — leg H (historical 688,128 B) vs the PUBLISHED run")
    print("=" * 100)
    worst = 0.0
    rows = []
    for s in SEEDS:
        pub = json.loads((PUB / f"seed_{s}" / "baselines_manifest.json").read_text())
        try:
            man, _ = load("historical_688128", s)
        except FileNotFoundError:
            print(f"  seed {s}: ⛔ NOT RUN")
            continue
        pv, nv = variants(pub), variants(man)
        for v in sorted(set(pv) & set(nv)):
            for kind in ("out_of_range", "duplicate_factor"):
                a, b = kindval(pv[v], kind), kindval(nv[v], kind)
                if a is None or b is None:
                    continue
                d = abs(a - b)
                worst = max(worst, d)
                rows.append((s, v, kind, a, b, d))
            for key in ("auroc_c_vs_b", "auroc_c_vs_a", "auroc_c_vs_a_union_b"):
                d = abs(pv[v][key] - nv[v][key])
                worst = max(worst, d)
        # composition too
        for arm, holder, field in (("knn", "variants", "in_corner_discrimination"),
                                   ("lace", "samplers", "in_corner_discrimination")):
            for v in (pub["arms"].get(arm, {}).get(holder, {}) or {}):
                if v in (man["arms"].get(arm, {}).get(holder, {}) or {}):
                    a = pub["arms"][arm][holder][v][field]["balanced_accuracy"]
                    b = man["arms"][arm][holder][v][field]["balanced_accuracy"]
                    worst = max(worst, abs(a - b))
    for s, v, kind, a, b, d in rows:
        if kind == "out_of_range":
            flag = "✅ bit-identical" if d == 0.0 else f"⛔ DIFFERS by {d:.3e}"
            print(f"  seed {s} {v:<24} {kind:<16} published {a:.16f} -> {b:.16f}  {flag}")
    print(f"\n  ⭐ worst absolute difference over EVERY shared refusal + composition "
          f"number: {worst:.3e}")

    print()
    print("=" * 100)
    print("2. ⭐ REFUSAL BY KIND — CONTAMINATED vs A7-CONTROLLED (⛔ never merged)")
    print("=" * 100)
    for leg in ("historical_688128", "deployed_524288"):
        mans = {}
        for s in SEEDS:
            try:
                mans[s] = load(leg, s)[0]
            except FileNotFoundError:
                pass
        if not mans:
            print(f"\n-- {leg}: ⛔ NOT RUN")
            continue
        print(f"\n-- leg {leg} (byte target "
              f"{mans[min(mans)]['provenance']['byte_target']['total_bytes']} B), "
              f"seeds {sorted(mans)}")
        names = sorted({v for m in mans.values() for v in variants(m)})
        hdr = (f"  {'variant':<26}{'kind':<17}{'CONTAMINATED(vs B)':>20}"
               f"{'A7-CONTROLLED':>18}{'A7D (dedup)':>18}{'seen-valid A':>18}")
        print(hdr)
        for v in names:
            for kind in ("out_of_range", "duplicate_factor"):
                raw, ctl, dd, va = [], [], [], []
                for s, m in mans.items():
                    ref = variants(m).get(v)
                    if ref is None:
                        continue
                    raw.append(kindval(ref, kind))
                    c = ((ref.get("a7_controlled") or {}).get("auroc_by_kind")
                         or {}).get(kind, {})
                    ctl.append(c.get("auroc"))
                    dd.append(kindval(ref, kind, "A7D"))
                    # seen-valid A, from the PERSISTED per-request scores
                    sc = load(leg, s)[1]["scores"]
                    arm = "lace" if v.startswith("lace_") else "knn"
                    key = v[5:] if v.startswith("lace_") else v
                    t = sc.get(arm, {}).get(key)
                    if t:
                        kinds = np.asarray(t["_kinds_C"])
                        C = np.asarray(t["C"]["scores"], float)[kinds == kind]
                        va.append(auroc(np.asarray(t["A"]["scores"], float), C))
                f = lambda x: "  n/a" if x[0] is None else f"{x[0]:.4f}±{x[1]:.4f}"
                print(f"  {v:<26}{kind:<17}{f(ms(raw)):>20}{f(ms(ctl)):>18}"
                      f"{f(ms(dd)):>18}{f(ms(va)):>18}")

    print()
    print("=" * 100)
    print("3. ⭐ R27.3 — the two LACE legs, both reported")
    print("=" * 100)
    for s in SEEDS:
        try:
            man, _ = load("historical_688128", s)
        except FileNotFoundError:
            continue
        arm = man["arms"].get("lace")
        if not arm:
            continue
        g = arm["whitening_exactness_gate"]
        print(f"\n-- seed {s}: selected shrinkage={g['selected_shrinkage']} "
              f"|diag-1|={g['selected_mean_abs_diag_minus_one']:.4f} "
              f"exact={g['selected_is_exact']} fired={g['fired']}")
        print(f"   fallback: {g['fallback_reason']}"
              + ("" if not g["fallback_selected"] else
                 f" -> sigma={g['fallback_selected']['sigma']} "
                 f"form={g['fallback_selected']['form']} "
                 f"shrinkage={g['fallback_selected']['shrinkage']} "
                 f"(val_agg {g['fallback_selected']['val_aggregate']:.4f} vs selected "
                 f"{arm['tuning']['selected']['val_aggregate']:.4f})"))
        for lname, leg in (arm.get("legs") or {}).items():
            print(f"   [{lname:<14}] best comp {leg['best_composition']['sampler']:<20}"
                  f"{leg['best_composition']['in_corner_balacc']:.4f} · best pooled refusal "
                  f"{leg['best_refusal_pooled_CONTAMINATED']['auroc_c_vs_b']:.4f} · "
                  f"best A7-dup "
                  f"{(leg['best_refusal_duplicate_factor_A7_CONTROLLED']['auroc'] or 0):.4f}"
                  f" · oor "
                  + ", ".join(f"{k}={0 if vv is None else vv:.4f}"
                              for k, vv in leg["out_of_range_A7_CLEAN"].items()))

    print()
    print("=" * 100)
    print("4. ⭐ P5 — is the composition cell byte-budget-invariant?")
    print("=" * 100)
    for s in SEEDS:
        try:
            h, _ = load("historical_688128", s)
            d, _ = load("deployed_524288", s)
        except FileNotFoundError:
            print(f"  seed {s}: ⛔ one leg NOT RUN")
            continue
        for v in ("knn", "knn_proto", "knn_compsel"):
            hv = h["arms"]["knn"]["variants"].get(v)
            dv = d["arms"]["knn"]["variants"].get(v)
            if not hv or not dv:
                continue
            a = hv["in_corner_discrimination"]["balanced_accuracy"]
            b = dv["in_corner_discrimination"]["balanced_accuracy"]
            print(f"  seed {s} {v:<14} in-corner balacc 688,128 B {a:.6f} -> "
                  f"524,288 B {b:.6f}   Δ={b - a:+.6f}"
                  + ("   ✅ INVARIANT" if a == b else "   ⚠ moves"))
        for v in ("knn", "knn_proto"):
            hv, dv = h["arms"]["knn"]["variants"].get(v), d["arms"]["knn"]["variants"].get(v)
            if hv and dv:
                for kind in ("out_of_range", "duplicate_factor"):
                    a, b = kindval(hv["refusal"], kind), kindval(dv["refusal"], kind)
                    print(f"  seed {s} {v:<14} refusal {kind:<16} {a:.6f} -> {b:.6f}"
                          f"   Δ={b - a:+.6f}")


def decomposition():
    """⭐ Why an A7-controlled number must never be quoted bare.

    A7 is built from TRAIN rows by ruling, so swapping the control from B to A7
    changes TWO things at once: the term count (6 -> 7) and the seen-ness of the
    negative pool (unseen-valid -> seen-valid).  A7D — the same records with the
    redundant term removed — holds the rows fixed and moves only the term count,
    so the two legs are separable and are reported separately.
    """
    print()
    print("=" * 100)
    print("5. ⭐ THE DECOMPOSITION — `duplicate_factor`, both legs separated")
    print("=" * 100)
    print(f"  {'seed':<6}{'variant':<26}{'B (unseen,6)':>14}{'A7D (seen,6)':>14}"
          f"{'A7 (seen,7)':>13}{'seen/unseen':>13}{'term count':>12}")
    agg = {}
    for s in SEEDS:
        try:
            man, _ = load("historical_688128", s)
        except FileNotFoundError:
            continue
        for v, ref in sorted(variants(man).items()):
            b = kindval(ref, "duplicate_factor")
            dd = kindval(ref, "duplicate_factor", "A7D")
            a7 = ((ref.get("a7_controlled") or {}).get("auroc_by_kind")
                  or {}).get("duplicate_factor", {}).get("auroc")
            if None in (b, dd, a7):
                continue
            print(f"  {s:<6}{v:<26}{b:>14.4f}{dd:>14.4f}{a7:>13.4f}"
                  f"{dd - b:>+13.4f}{a7 - dd:>+12.4f}")
            agg.setdefault(v, []).append((b, dd, a7))
    print("\n  -- mean over the seeds present --")
    for v, rows in sorted(agg.items()):
        a = np.asarray(rows, float)
        print(f"  {'':<6}{v:<26}{a[:, 0].mean():>14.4f}{a[:, 1].mean():>14.4f}"
              f"{a[:, 2].mean():>13.4f}{(a[:, 1] - a[:, 0]).mean():>+13.4f}"
              f"{(a[:, 2] - a[:, 1]).mean():>+12.4f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
    decomposition()
