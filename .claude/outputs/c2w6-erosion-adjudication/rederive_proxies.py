#!/usr/bin/env python
"""I2 construct-validity + reliability, and the P-residual row re-scored (F2 repair).

G. do the two registered usefulness proxies agree with each other?
H. are they reliable at all (split-half over readings / ICC over LOO checkpoints)?
   -> attenuation ceiling: the largest |rho| the rig CAN observe.
I. is a SLOT a WELL? same-slot cross-reading site displacement vs the between-slot
   spread within a reading.
J. the P-residual interaction row, scored three ways (engineer's pooled-mean bar,
   my paired per-seed repair at the unmatched horizon, and at a matched step count).
K. the engineer's 'pooled (18 wells)' rho column vs the REGISTERED per-seed-then-mean.

Run: /Users/user/Desktop/CHLU/.venv/bin/python \
       .claude/outputs/c2w6-erosion-adjudication/rederive_proxies.py  (cwd=repo root)
"""
import numpy as np

from rederive_i2 import (i2_for, load, se, spearman, wells_of)  # noqa: E402

PSI = ".claude/outputs/psi-payload-residual/"
ARMS = ["p1_off", "w40_p1_off", "resoff_p1_off", "p1_on"]


def main():
    by = load()

    print("=" * 78)
    print("G. DO THE TWO REGISTERED USEFULNESS PROXIES AGREE? rho(sel, loo)")
    print("=" * 78)
    for cell in ARMS:
        v = []
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            o = i2_for(by[(cell, s)])
            v.append(spearman([w["sel"] for w in o["wells"]],
                              [w["loo"] for w in o["wells"]]))
        v = np.array(v)
        print("  %-14s per-seed %s | mean %+.4f +- %.4f (%.2f SE, %d/%d negative)"
              % (cell, np.array2string(v, precision=3), v.mean(), se(v),
                 abs(v.mean()) / se(v), int((v < 0).sum()), v.size))

    print("\n  sign of 'usefulness' itself (loo_delta_bpc > 0 = deleting the well "
          "COSTS bpc = useful):")
    for cell in ARMS:
        pos = tot = 0
        vals = []
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            o = i2_for(by[(cell, s)])
            for w in o["wells"]:
                tot += 1
                pos += int(w["loo"] > 0)
                vals.append(w["loo"])
        print("  %-14s useful wells %d/%d | |loo| median %.2e max %.2e"
              % (cell, pos, tot, np.median(np.abs(vals)), np.max(np.abs(vals))))

    print()
    print("=" * 78)
    print("H. RELIABILITY of each proxy (can rho be nonzero at all?)")
    print("=" * 78)
    print("  H1 read-selection split-half (first half of readings vs second),"
          " Spearman across wells")
    rel_sel = []
    for cell in ARMS:
        v = []
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            ws = [w for w in wells_of(by[(cell, s)]) if w["live"].mean() > 0.5]
            n = len(ws[0]["step"])
            h = n // 2
            a = [w["sel"][:h].mean() for w in ws]
            b = [w["sel"][h:].mean() for w in ws]
            v.append(spearman(a, b))
        v = np.array(v)
        # Spearman-Brown correction from half-length to full-length reliability
        sb = 2 * v / (1 + v)
        print("    %-14s halves %s | mean %+.3f | Spearman-Brown full-test "
              "reliability %+.3f" % (cell, np.array2string(v, precision=3),
                                     v.mean(), np.nanmean(sb)))
        if cell == "p1_off":
            rel_sel = np.nanmean(sb)

    print("\n  H2 leave-one-well-out: variance across the 4 checkpoints (within "
          "well) vs across wells")
    rel_loo = []
    for cell in ARMS:
        iccs = []
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            ws = [w for w in wells_of(by[(cell, s)]) if w["live"].mean() > 0.5]
            M = []
            for w in ws:
                v = w["loo"][np.isfinite(w["loo"])]
                M.append(v)
            k = min(len(v) for v in M)
            M = np.array([v[:k] for v in M])          # (n_wells, k)
            gm = M.mean()
            ms_b = k * ((M.mean(1) - gm) ** 2).sum() / (M.shape[0] - 1)
            ms_w = ((M - M.mean(1, keepdims=True)) ** 2).sum() / (M.shape[0] * (k - 1))
            icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
            iccs.append(icc)
            # reliability of the MEAN of k checkpoints (Spearman-Brown)
        iccs = np.array(iccs)
        rel_k = k * iccs / (1 + (k - 1) * iccs)
        print("    %-14s ICC(1,1) per seed %s | reliability of the k=%d mean %s"
              % (cell, np.array2string(iccs, precision=3), k,
                 np.array2string(rel_k, precision=3)))
        if cell == "p1_off":
            rel_loo = float(np.mean(np.clip(rel_k, 0, 1)))

    print("\n  H3 ATTENUATION CEILING on the primary arm (p1_off):")
    print("     rel(read-selection, full test) = %.3f ; rel(LOO mean of 4) = %.3f"
          % (rel_sel, rel_loo))
    print("     max observable |rho| if rho_true = 1.0 : sel %.3f | loo %.3f"
          % (np.sqrt(max(rel_sel, 0)), np.sqrt(max(rel_loo, 0))))
    print("     (erosion-rate reliability assumed 1.0 — an upper bound, so these"
          " ceilings are optimistic)")

    print()
    print("=" * 78)
    print("I. IS A SLOT A WELL? same-slot drift vs between-slot spread")
    print("=" * 78)
    for cell in ["p1_off", "w40_p1_off"]:
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            ws = [w for w in wells_of(by[(cell, s)]) if w["live"].mean() > 0.5]
            within, between = [], []
            for w in ws:
                m = w["live"] > 0.5
                st = w["site"][m]
                within += list(np.linalg.norm(np.diff(st, axis=0), axis=1))
            n = len(ws[0]["step"])
            for j in range(n):
                pts = np.array([w["site"][j] for w in ws if w["live"][j] > 0.5])
                if len(pts) > 1:
                    d = [np.linalg.norm(pts[a] - pts[b])
                         for a in range(len(pts)) for b in range(a + 1, len(pts))]
                    between += d
            print("  %-12s s%d  within-slot |dsite| median %.3f  |  "
                  "between-slot median %.3f  |  ratio %.2f  (place radius 0.30)"
                  % (cell, s, np.median(within), np.median(between),
                     np.median(within) / np.median(between)))

    print()
    print("=" * 78)
    print("J. THE P-RESIDUAL INTERACTION ROW, scored three ways")
    print("=" * 78)
    import json
    tr = sorted(json.load(open(PSI + "psires_trained_records.json"))["records"],
                key=lambda y: y["seed"])
    banked_on = np.array([x["arms"]["residual_on"]["depth_median"] for x in tr])
    on_fin = np.array([by[("p1_on", s)]["depth_final"] for s in (0, 1, 2)])
    off_fin = np.array([by[("p1_off", s)]["depth_final"] for s in (0, 1, 2)])
    on_200 = np.array([by[("p1_on", s)]["depth_at_200"] for s in (0, 1, 2)])
    print("  banked residual-only (200 steps, w4)   %s  (pooled mean %.4f)"
          % (np.array2string(banked_on, precision=5), banked_on.mean()))
    print("  p1_on final (1000 steps)               %s"
          % np.array2string(on_fin, precision=5))
    print("  p1_on at step 200 (matched step count) %s"
          % np.array2string(on_200, precision=5))
    print("  p1_off final (1000 steps)              %s"
          % np.array2string(off_fin, precision=5))
    print()
    print("  (a) ENGINEER's scoring, per-seed vs the POOLED MEAN 0.1321 : %d/3"
          % int((on_fin >= banked_on.mean()).sum()),
          "seeds", [s for s in range(3) if on_fin[s] >= banked_on.mean()])
    print("  (b) F2 REPAIR, paired per-seed, UNMATCHED horizon (1000 vs 200): %d/3"
          % int((on_fin >= banked_on).sum()),
          "seeds", [s for s in range(3) if on_fin[s] >= banked_on[s]])
    print("  (c) F2 REPAIR, paired per-seed at a MATCHED step count (200 vs 200): "
          "%d/3" % int((on_200 >= banked_on).sum()),
          "seeds", [s for s in range(3) if on_200[s] >= banked_on[s]])
    print("      paired log-ratio (c): %s -> geo %.3fx (%.2f SE)"
          % (np.array2string(on_200 / banked_on, precision=3),
             np.exp(np.log(on_200 / banked_on).mean()),
             np.log(on_200 / banked_on).mean() / se(np.log(on_200 / banked_on))))
    print("  (d) the DISPROOF clause (ON collapses below OFF, within-wave paired):"
          " ON<OFF on %d/3 seeds; ON/OFF %s geo %.3fx"
          % (int((on_fin < off_fin).sum()),
             np.array2string(on_fin / off_fin, precision=3),
             np.exp(np.log(on_fin / off_fin).mean())))

    print()
    print("=" * 78)
    print("K. the engineer's 'pooled (n wells)' rho vs the REGISTERED estimator")
    print("=" * 78)
    for cell in ARMS:
        sel, loo, gr, rate = [], [], [], []
        per_seed_sel, per_seed_gr = [], []
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            o = i2_for(by[(cell, s)])
            sel += [w["sel"] for w in o["wells"]]
            loo += [w["loo"] for w in o["wells"]]
            gr += [w["grad"] for w in o["wells"]]
            rate += [w["rate"] for w in o["wells"]]
            per_seed_sel.append(o["rho_sel"])
            per_seed_gr.append(o["rho_grad"])
        ps = np.array(per_seed_sel)
        pg = np.array([x for x in per_seed_gr if np.isfinite(x)])
        print("  %-14s POOLED(%d wells) sel %+.4f loo %+.4f grad %+.4f  ||  "
              "REGISTERED per-seed-mean sel %+.4f grad %s"
              % (cell, len(sel), spearman(sel, rate), spearman(loo, rate),
                 spearman(gr, rate), ps.mean(),
                 ("%+.4f" % pg.mean()) if pg.size else "nan"))


if __name__ == "__main__":
    main()
