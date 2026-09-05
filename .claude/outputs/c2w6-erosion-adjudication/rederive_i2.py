#!/usr/bin/env python
"""I2 re-adjudication + the designed-decay separation audit, from raw telemetry.

Everything here is recomputed from ``records[*].telemetry[*].wells`` — the
per-reading per-slot series — never from ``records[*].i2``.

Designed decay (read out of the code, not assumed):
  train_cluformer.build_lane_plan l.804-809 -> group_scale[c, s] = exp(-leak)
  for every live slot at every chunk tick, leak = 0.02 (run-2 config).
  blocks.py l.1402 applies it BEFORE chunk c's write, so a slot last written at
  chunk c carries (n_chunks-1-c) further applications by the end of the pass:
      amp   x exp(-0.02 * (15 - c))
      depth x exp(-0.04 * (15 - c))       [depth ~ amp^2]
  Cross-check: 1 - exp(-0.04) = 0.039211 vs the harness's own measured
  median_rel_drop_own = 0.03921 (telemetry[*].interference).

Run: /Users/user/Desktop/CHLU/.venv/bin/python \
       .claude/outputs/c2w6-erosion-adjudication/rederive_i2.py   (cwd=repo root)
"""
import glob
import itertools
import json
import os

import numpy as np

AE = ".claude/outputs/c2w6-anti-erosion/"
LEAK = 0.02
N_CHUNKS = 16
RNG = np.random.default_rng(20260805)


def load():
    recs = []
    for f in sorted(glob.glob(AE + "erosion_*_records.json")):
        if os.sep + "smoke" + os.sep in f:
            continue
        recs += json.load(open(f))["records"]
    return {(r["cell"], int(r["seed"])): r for r in recs}


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, float)
    r[order] = np.arange(1, a.size + 1, dtype=float)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    a, b = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def log_slope(steps, dep):
    s, d = np.asarray(steps, float), np.asarray(dep, float)
    m = np.isfinite(s) & np.isfinite(d) & (d > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(s[m], np.log(d[m]), 1)[0])


def wells_of(rec):
    """Per-slot series: step, depth(raw), depth(decay-netted), sel, grad, loo."""
    rows = [t for t in rec["telemetry"] if t.get("applicable") and t.get("wells")]
    K = len(rows[0]["wells"])
    out = []
    for i in range(K):
        st, dr, dn, sel, gr, loo, lwc, live, site = ([] for _ in range(9))
        for t in rows:
            w = t["wells"][i]
            st.append(int(t["at_step"]))
            live.append(float(w["live"]))
            dr.append(float(w["depth"]))
            c = int(w["last_write_chunk"])
            lwc.append(c)
            # net the designed decay: divide out exp(-2*leak*(N-1-c))
            f = np.exp(-2.0 * LEAK * (N_CHUNKS - 1 - c)) if c >= 0 else np.nan
            dn.append(float(w["depth"]) / f if (c >= 0 and f > 0) else np.nan)
            sel.append(float(w["read_selection"]))
            gr.append(float(w["grad_atoms"]))
            loo.append(float(w["loo_delta_bpc"]))
            site.append(list(w["site_addr"]))
        out.append(dict(slot=i, step=np.array(st), live=np.array(live),
                        depth=np.array(dr), depth_net=np.array(dn),
                        sel=np.array(sel), grad=np.array(gr),
                        loo=np.array(loo), lwc=np.array(lwc),
                        site=np.array(site, float)))
    return out


def i2_for(rec, netted=False):
    ws = wells_of(rec)
    use, allw = [], []
    for w in ws:
        lf = float(w["live"].mean())
        d = w["depth_net"] if netted else w["depth"]
        row = dict(slot=w["slot"], live_frac=lf,
                   rate=-log_slope(w["step"], d),
                   sel=float(np.mean(w["sel"])),
                   grad=float(np.mean(w["grad"])),
                   loo=float(np.mean(w["loo"][np.isfinite(w["loo"])]))
                   if np.isfinite(w["loo"]).any() else np.nan,
                   n_pos=int((d > 0).sum()))
        allw.append(row)
        if lf > 0.5:
            use.append(row)
    r = [w["rate"] for w in use]
    return dict(n_wells=len(use),
                rho_sel=spearman([w["sel"] for w in use], r),
                rho_loo=spearman([w["loo"] for w in use], r),
                rho_grad=spearman([w["grad"] for w in use], r),
                wells=use, all_wells=allw)


def fisher_pool(rhos):
    r = np.asarray([x for x in rhos if np.isfinite(x)], float)
    if r.size < 2:
        return float("nan"), float("nan"), r.size
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(z.mean())), float(np.tanh(z.mean() + se(z))), r.size


def exact_perm_null(n):
    """Exact null distribution of Spearman rho for n untied items."""
    base = np.arange(1, n + 1, dtype=float)
    rb = base - base.mean()
    den = (rb ** 2).sum()
    out = [float((rb * (np.array(p, float) - base.mean())).sum() / den)
           for p in itertools.permutations(base)]
    return np.array(out)


def main():
    by = load()
    print("=" * 78)
    print("A. DESIGNED-DECAY SEPARATION AUDIT (task S3)")
    print("=" * 78)
    # A1 - the decay constant, from the harness's own interference audit
    drops = []
    for k, r in by.items():
        for t in r["telemetry"]:
            v = t.get("interference", {}).get("median_rel_drop_own")
            if v is not None and np.isfinite(v):
                drops.append(float(v))
    print("A1 designed per-chunk depth drop: predicted 1-exp(-2*leak) = %.6f"
          % (1 - np.exp(-2 * LEAK)))
    print("   harness-measured median_rel_drop_own over %d readings: "
          "median %.6f  min %.6f  max %.6f"
          % (len(drops), np.median(drops), np.min(drops), np.max(drops)))
    own_res = [float(t["interference"]["max_abs_own_residual_vs_decay_law"])
               for k, r in by.items() for t in r["telemetry"]
               if t.get("interference", {}).get("n_events", 0) > 0]
    print("   max own-leg residual vs the decay law over all readings: %.3e"
          % max(own_res))

    # A2 - does last_write_chunk drift? (the confound)
    print("\nA2 last_write_chunk drift per slot (the confound the netting removes)")
    for cell in ["p1_off", "w40_p1_off"]:
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            ws = wells_of(by[(cell, s)])
            desc = []
            for w in ws:
                m = w["live"] > 0.5
                if m.sum() < 3:
                    continue
                c = w["lwc"][m]
                desc.append("s%d:[%d..%d]n%d" % (w["slot"], c.min(), c.max(),
                                                 len(np.unique(c))))
            print("   %-12s seed %d  %s" % (cell, s, "  ".join(desc)))

    # A3 - end-to-end recomputation of ONE seed's curve from the per-well series
    print("\nA3 end-to-end curve recomputation from per-well raw series "
          "(p1_off seed 0)")
    r = by[("p1_off", 0)]
    ws = wells_of(r)
    steps = ws[0]["step"]
    med_raw, med_net = [], []
    for j in range(len(steps)):
        live = [w["depth"][j] for w in ws if w["live"][j] > 0.5]
        liven = [w["depth_net"][j] for w in ws if w["live"][j] > 0.5]
        med_raw.append(np.median(live) if live else np.nan)
        med_net.append(np.median(liven) if liven else np.nan)
    med_raw, med_net = np.array(med_raw), np.array(med_net)
    pub = np.array([c["depth_median"] for c in r["curve"]])
    print("   max |median(live wells) - curve.depth_median| = %.3e  (n=%d)"
          % (np.max(np.abs(med_raw - pub)), len(pub)))
    i200 = int(np.argmin(np.abs(steps - 200)))
    print("   RAW    final/200 = %.4f   final/untrained = %.4f"
          % (med_raw[-1] / med_raw[i200], med_raw[-1] / med_raw[0]))
    print("   NETTED final/200 = %.4f   final/untrained = %.4f"
          % (med_net[-1] / med_net[i200], med_net[-1] / med_net[0]))

    # A4 - netted E1/E2 for every cell
    print("\nA4 E1/E2 ratios, RAW vs DESIGNED-DECAY-NETTED, all cells")
    print("   cell            seed  raw f/200  net f/200  raw f/unt  net f/unt")
    net_tab = {}
    for cell in ["p1_off", "p1_on", "w40_p1_off", "w40_p1_on",
                 "resoff_p1_off", "resoff_p1_on"]:
        for s in (0, 1, 2):
            if (cell, s) not in by:
                continue
            ws = wells_of(by[(cell, s)])
            st = ws[0]["step"]
            mr, mn = [], []
            for j in range(len(st)):
                lv = [w["depth"][j] for w in ws if w["live"][j] > 0.5]
                ln_ = [w["depth_net"][j] for w in ws if w["live"][j] > 0.5]
                mr.append(np.median(lv) if lv else np.nan)
                mn.append(np.median(ln_) if ln_ else np.nan)
            mr, mn = np.array(mr), np.array(mn)
            i2i = int(np.argmin(np.abs(st - 200)))
            net_tab.setdefault(cell, []).append(
                (mr[-1] / mr[i2i], mn[-1] / mn[i2i],
                 mr[-1] / mr[0], mn[-1] / mn[0]))
            print("   %-14s s%d   %8.4f   %8.4f   %8.4f   %8.4f"
                  % (cell, s, mr[-1] / mr[i2i], mn[-1] / mn[i2i],
                     mr[-1] / mr[0], mn[-1] / mn[0]))
    print("\n   -> E1/E2 verdicts under netting (rule: OFF <=0.5 on >=2/3; "
          "ON >=0.5 on 3/3)")
    for cell, v in net_tab.items():
        a = np.array(v)
        print("   %-14s raw n<=0.5: %d/3   net n<=0.5: %d/3 | "
              "geo raw %.3f  geo net %.3f"
              % (cell, int((a[:, 0] <= 0.5).sum()), int((a[:, 1] <= 0.5).sum()),
                 np.exp(np.log(a[:, 0]).mean()), np.exp(np.log(a[:, 1]).mean())))

    print()
    print("=" * 78)
    print("B. I2 — rho(usefulness, erosion rate), re-derived")
    print("=" * 78)
    for cell in ["p1_off", "w40_p1_off", "resoff_p1_off", "p1_on"]:
        for netted in (False, True):
            rhos_s, rhos_l, rhos_g, nw = [], [], [], []
            for s in (0, 1, 2):
                if (cell, s) not in by:
                    continue
                o = i2_for(by[(cell, s)], netted=netted)
                rhos_s.append(o["rho_sel"])
                rhos_l.append(o["rho_loo"])
                rhos_g.append(o["rho_grad"])
                nw.append(o["n_wells"])
                if not netted:
                    eng = by[(cell, s)]["i2"]
                    d1 = abs(o["rho_sel"] - eng["rho_read_selection"])
                    d2 = abs(o["rho_loo"] - eng["rho_loo_delta_bpc"])
                    tag = "MATCH" if max(d1, d2) < 1e-12 else "MISMATCH %.2e" % max(d1, d2)
                    print("   %-14s s%d n_wells %d  rho_sel %+.4f  rho_loo %+.4f "
                          " rho_grad %+.4f   [vs harness: %s]"
                          % (cell, s, o["n_wells"], o["rho_sel"], o["rho_loo"],
                             o["rho_grad"], tag))
            lab = "NETTED" if netted else "RAW   "
            for nm, rr in [("sel", rhos_s), ("loo", rhos_l), ("grad", rhos_g)]:
                rr2 = [x for x in rr if np.isfinite(x)]
                if not rr2:
                    print("   %-14s %s rho_%-4s all-nan" % (cell, lab, nm))
                    continue
                fp, fhi, n = fisher_pool(rr2)
                print("   %-14s %s rho_%-4s mean %+.4f +- %.4f (n=%d) | "
                      "Fisher %+.4f | per-seed %s"
                      % (cell, lab, nm, np.mean(rr2),
                         se(rr2) if len(rr2) > 1 else np.nan, len(rr2), fp,
                         np.array2string(np.array(rr2), precision=3)))
        print()

    print("=" * 78)
    print("C. POWER / DISCRIMINABILITY at n_wells = 5-6 (Hub Q2)")
    print("=" * 78)
    for n in (5, 6):
        null = exact_perm_null(n)
        print(" n=%d exact null: sd(rho) = %.4f ; P(|rho|>=0.3) = %.3f ; "
              "P(rho>=0.5) = %.3f ; n_perm %d"
              % (n, null.std(), np.mean(np.abs(null) >= 0.3),
                 np.mean(null >= 0.5), null.size))
    # 3-seed mean-rho sampling distribution, by simulation from a Gaussian copula
    print("\n 3-seed MEAN rho (the registered statistic) — 200k sims, n_wells=6")
    for rho_true in (0.0, 0.3, 0.5, 0.7):
        pear = 2 * np.sin(np.pi * rho_true / 6)
        n, B = 6, 200000
        z1 = RNG.standard_normal((B, 3, n))
        z2 = RNG.standard_normal((B, 3, n))
        y = pear * z1 + np.sqrt(max(1 - pear ** 2, 0)) * z2
        rx = np.argsort(np.argsort(z1, -1), -1) + 1.0
        ry = np.argsort(np.argsort(y, -1), -1) + 1.0
        rx -= rx.mean(-1, keepdims=True)
        ry -= ry.mean(-1, keepdims=True)
        rho = (rx * ry).sum(-1) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))
        m = rho.mean(-1)
        print("   true rho_s=%.1f -> mean-of-3 rho: mean %+.3f sd %.3f | "
              "P(read as NO_STRUCTURE ||m|<0.3) = %.3f | P(m>=+0.5 confirm) "
              "= %.3f | P(m<=-0.3 refute) = %.3f"
              % (rho_true, m.mean(), m.std(), np.mean(np.abs(m) < 0.3),
                 np.mean(m >= 0.5), np.mean(m <= -0.3)))

    print("\n D. stratified exact permutation test on the PRIMARY arm "
          "(p1_off, read-selection), H0: no association within any seed")
    obs, perms = [], []
    for s in (0, 1, 2):
        o = i2_for(by[("p1_off", s)])
        sel = np.array([w["sel"] for w in o["wells"]])
        rt = np.array([w["rate"] for w in o["wells"]])
        obs.append(spearman(sel, rt))
        ps = [spearman(np.array(p), rt) for p in itertools.permutations(sel)]
        perms.append(np.array(ps))
    obs_mean = float(np.mean(obs))
    B = 200000
    draws = np.stack([p[RNG.integers(0, p.size, B)] for p in perms]).mean(0)
    print("   observed mean rho = %+.4f ; null sd = %.4f ; "
          "two-sided p = %.4f ; P(null >= +0.5) = %.4f"
          % (obs_mean, draws.std(), np.mean(np.abs(draws) >= abs(obs_mean)),
             np.mean(draws >= 0.5)))
    print("   95%% CI of the null on the mean-of-3 statistic: [%.3f, %.3f]"
          % (np.quantile(draws, 0.025), np.quantile(draws, 0.975)))

    print("\n E. allocation drift — is a SLOT a WELL? (p1_off, per slot)")
    for s in (0, 1, 2):
        ws = wells_of(by[("p1_off", s)])
        out = []
        for w in ws:
            m = w["live"] > 0.5
            if m.sum() < 3:
                continue
            st = w["site"][m]
            d = np.linalg.norm(np.diff(st, axis=0), axis=1)
            out.append("s%d:med|dsite|=%.3f max=%.3f" %
                       (w["slot"], np.median(d), d.max()))
        print("   seed %d  %s" % (s, "  ".join(out)))

    print("\n F. usefulness-proxy degeneracy (read-selection ties, p1_off)")
    for s in (0, 1, 2):
        o = i2_for(by[("p1_off", s)])
        sel = np.array([w["sel"] for w in o["wells"]])
        loo = np.array([w["loo"] for w in o["wells"]])
        rt = np.array([w["rate"] for w in o["wells"]])
        print("   seed %d  sel %s" % (s, np.array2string(sel, precision=2)))
        print("            loo %s" % np.array2string(loo, precision=5))
        print("            rate %s" % np.array2string(rt, precision=5))
        print("            rho(sel,loo) = %+.3f" % spearman(sel, loo))


if __name__ == "__main__":
    main()
