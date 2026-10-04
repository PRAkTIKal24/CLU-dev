"""PREREG P1/P6 — ⛔ the control on ME.

The anchor (`n_val` 512, `tune_n` 256) is the shipped pair, i.e. EXACTLY the
`gen-val-split-fix` leg-A configuration. If my harness — above all the one-pass
`_trial_scores` — moved anything, it shows up here as a non-zero delta and every
number downstream of it is void.

⛔ Tolerance is exactly zero on the selection-stability numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLAUDE = Path(__file__).resolve().parents[1]
MINE = CLAUDE / "gen-atom-mean-resweep" / "runs" / "nval0512_tune0256"
THEIRS = CLAUDE / "gen-val-split-fix" / "runs" / "atom_mean"
KEYS = ("k", "metric", "normaliser", "scope")
SEEDS = (0, 1, 2)


def load(p):
    return json.load(open(p)) if p.exists() else None


def main():
    ok, checked, missing = True, 0, []
    print("# PREREG P1 — the anchor reproduces gen-val-split-fix leg A\n")
    print("| cell | field | ⭐ mine | split-fix | Δ | bit-identical |")
    print("|---|---|---|---|---|---|")
    exp_mine, exp_theirs = [], []
    for s in SEEDS:
        a = load(MINE / f"seed_{s}" / "atom_manifest.json")
        b = load(THEIRS / f"seed_{s}" / f"seed_{s}" / "atom_manifest.json")
        if a is None or b is None:
            missing.append(s)
            continue
        for field, path in (("n_fit", ("rows", "fit")), ("n_val", ("rows", "val"))):
            va, vb = a["rows"][path[1]], b["rows"][path[1]]
            same = va == vb
            ok &= same
            checked += 1
            print(f"| s{s} | {field} | {va:,} | {vb:,} | {va - vb:+,} | "
                  f"{'✅' if same else '⛔ **MOVED**'} |")
        for tag in sorted(a.get("budgets") or {}):
            if tag not in (b.get("budgets") or {}):
                continue
            sa = a["budgets"][tag]["tuning"]["stability"]
            sb = b["budgets"][tag]["tuning"]["stability"]
            for field in ("gap_best_to_second", "spread_over_trials"):
                va, vb = float(sa[field]), float(sb[field])
                same = va == vb
                ok &= same
                checked += 1
                print(f"| s{s}/{tag} | {field} | {va:.10f} | {vb:.10f} | "
                      f"{va - vb:+.2e} | {'✅' if same else '⛔ **MOVED**'} |")
            for field in ("selected", "argmax_ihalf1", "argmax_ihalf2",
                          "ihalves_agree", "halves_agree"):
                va = (tuple(sa[field][k] for k in KEYS)
                      if isinstance(sa[field], dict) else sa[field])
                vb = (tuple(sb[field][k] for k in KEYS)
                      if isinstance(sb[field], dict) else sb[field])
                same = va == vb
                ok &= same
                checked += 1
                print(f"| s{s}/{tag} | {field} | `{va}` | `{vb}` | — | "
                      f"{'✅' if same else '⛔ **MOVED**'} |")
            # ⭐ and every one of the 24 trials, not only the winner
            ta = a["budgets"][tag]["tuning"]["trials"]
            tb = b["budgets"][tag]["tuning"]["trials"]
            for col in ("val_aggregate", "val_aggregate_half1", "val_aggregate_half2",
                        "val_aggregate_ihalf1", "val_aggregate_ihalf2"):
                xa = np.array([t[col] for t in ta])
                xb = np.array([t[col] for t in tb])
                same = xa.shape == xb.shape and bool((xa == xb).all())
                ok &= same
                checked += 1
                print(f"| s{s}/{tag} | all 24 `{col}` | — | — | "
                      f"max|Δ| {np.abs(xa - xb).max():.2e} | "
                      f"{'✅' if same else '⛔ **MOVED**'} |")
        ea = (a.get("exponent") or {}).get("request_addressed/k1") or {}
        eb = (b.get("exponent") or {}).get("request_addressed/k1") or {}
        if "slope" in ea and "slope" in eb:
            exp_mine.append(float(ea["slope"]))
            exp_theirs.append(float(eb["slope"]))

    # ⛔ INCOMPLETE is not FALSIFIED. A verdict that scores a missing seed as a
    # failure is the same error in the other direction as scoring it as a pass.
    if not ok:
        verdict = ("⛔ **P1 FALSIFIED — my harness MOVED something; everything "
                   "downstream of it is VOID**")
    elif missing:
        verdict = (f"⚠ **P1 HOLDING on the seeds run so far (missing {missing}) — "
                   "⛔ NOT YET CONFIRMED, and ⛔ not a failure either**")
    else:
        verdict = ("✅ **P1 CONFIRMED — the anchor is bit-identical to "
                   "split-fix leg A on all 3 seeds**")
    print(f"\n**{checked} comparisons, {checked} matched; verdict: {verdict}**")

    if exp_mine:
        print("\n# PREREG P6 — the deployable (request-addressed, k=1) exponent\n")
        print("| seed | ⭐ mine (repaired) | split-fix leg A (repaired) | "
              "gen-atom-mean-control (BROKEN, published) |")
        print("|---|---|---|---|")
        broken = [-0.165481, -0.165334, -0.154152]
        for i, s in enumerate(SEEDS[:len(exp_mine)]):
            print(f"| {s} | {exp_mine[i]:.6f} | {exp_theirs[i]:.6f} | {broken[i]:.6f} |")
        print(f"| **mean ± sd** | **{np.mean(exp_mine):.4f} ± {np.std(exp_mine):.4f}** | "
              f"**{np.mean(exp_theirs):.4f} ± {np.std(exp_theirs):.4f}** | "
              f"**{np.mean(broken):.4f} ± {np.std(broken):.4f}** |")
        print("\n⛔ EXPONENT STUDIES ARE SUSPENDED (Head, 2026-09-15). This number buys "
              "no claim and adjudicates nothing.")


if __name__ == "__main__":
    main()
