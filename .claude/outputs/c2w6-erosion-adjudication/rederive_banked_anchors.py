#!/usr/bin/env python
"""Independent re-derivation of every banked anchor PREREG-AntiErosion.md §3 rests on.

Run:  /Users/user/Desktop/CHLU/.venv/bin/python \
        .claude/outputs/c2w6-erosion-adjudication/rederive_banked_anchors.py
(cwd = repo root).  Reads only banked raw JSON; runs no model, trains nothing.
"""
import json
import numpy as np

PSI = ".claude/outputs/psi-payload-residual/"
PROBE = ".claude/outputs/pilot-placement-probe/"


def se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x))


def main():
    # --- untrained run-2-config depth (prereg §3 "0.0727/0.1490/0.2250") ---
    led = json.load(open(PSI + "psires_ledger_records.json"))["records"]
    r1 = sorted([x for x in led if x["cell"] == "run1"], key=lambda y: y["seed"])
    unt = np.array([x["forward"]["0"]["depth_median"] for x in r1])

    # --- trained 200-step arms (prereg §3 off/on anchors) ---
    tr = sorted(json.load(open(PSI + "psires_trained_records.json"))["records"],
                key=lambda y: y["seed"])
    off = np.array([x["arms"]["residual_off"]["depth_median"] for x in tr])
    on = np.array([x["arms"]["residual_on"]["depth_median"] for x in tr])
    bpc_off = np.array([x["arms"]["residual_off"]["bpc_live"] for x in tr])
    bpc_on = np.array([x["arms"]["residual_on"]["bpc_live"] for x in tr])

    print("untrained      ", np.round(unt, 4).tolist(), "(prereg: 0.0727/0.1490/0.2250)")
    print("residual_off   ", np.round(off, 4).tolist(), "(prereg: 0.0011/0.0582/0.0373)")
    print("residual_on    ", np.round(on, 4).tolist(), "(prereg: 0.0021/0.2732/0.1210)")
    print("on/off ratios  ", ["%.1fx" % v for v in on / off], "(psires: 1.9/4.7/3.2)")
    print("paired d(depth)  %+.4f +- %.4f  (%.2f SE)" % ((on - off).mean(), se(on - off),
                                                         (on - off).mean() / se(on - off)))
    lr = np.log(on / off)
    print("paired log-ratio %.3fx [%.2f, %.2f] (%.2f SE)  <-- log scale, same data"
          % (np.exp(lr.mean()), np.exp(lr.mean() - se(lr)), np.exp(lr.mean() + se(lr)),
             lr.mean() / se(lr)))
    print("paired d(bpc)    %+.4f +- %.4f  (psires: +0.0053 +- 0.0030)"
          % ((bpc_on - bpc_off).mean(), se(bpc_on - bpc_off)))
    print("residual-only pooled mean depth %.4f  (prereg P-residual bar '0.132 banked');"
          " seeds meeting it: %d/3" % (on.mean(), int((on >= on.mean()).sum())))
    print("within-arm seed spread: off %.0fx  on %.0fx" % (off.max() / off.min(),
                                                           on.max() / on.min()))

    # --- probe R3 anchor (prereg §3 "0.0288 -> 4.95e-63") ---
    pb = json.load(open(PROBE + "probe_trained_records.json"))["records"]
    base = sorted([x for x in pb if x["cell"] == "baseline"], key=lambda y: y["seed"])
    dep = np.array([x["arms"]["clu_store"]["depth_median"] for x in base])
    print()
    print("probe trained  ", ["%.4g" % v for v in dep])
    print("  arithmetic mean %.4g  <-- the banked '4.95e-63'" % dep.mean())
    print("  geometric  mean %.4g" % np.exp(np.log(dep).mean()))
    print("  spread %.1f orders of magnitude across 3 seeds"
          % (np.log10(dep.max()) - np.log10(dep.min())))


if __name__ == "__main__":
    main()
