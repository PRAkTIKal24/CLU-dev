"""Merge the re-measurement with bprime-c6's banked artifacts and build the tables."""
import json
import numpy as np

BANK = '.claude/outputs/bprime-c6/results/exp_route3_thirdparty.json'
BANK_TOP = '.claude/outputs/bprime-c6/results/thirdparty_topup_R042.json'
NEW = ['/tmp/rm_main.json', '/tmp/rm_topup.json']
OUT = '.claude/outputs/bprime-s-remeasure/'

bank = json.load(open(BANK))['cells'] + json.load(open(BANK_TOP))['cells']
new = sum([json.load(open(p))['cells'] for p in NEW], [])
key = lambda c: (round(float(c['ball_radius']), 6), int(c['seed']))
B = {key(c): c for c in bank}
N = {key(c): c for c in new}
assert set(B) == set(N), (set(B) ^ set(N))

rows = []
for k in sorted(B):
    b, n = B[k], N[k]
    e1s = n['E1_subtracted']['median']; e1u = n['E1_unsubtracted']['median']
    e2s = n['E2_subtracted']['median']; e2u = n['E2_unsubtracted']['median']
    rows.append(dict(R=k[0], seed=k[1], adm=b['admissible'],
                     banked_s=b['s_fitted_well'], e1s=e1s, e1u=e1u,
                     r1=e1u / e1s, e2s=e2s, e2u=e2u, r2=e2u / e2s,
                     delta_banked=e1s - b['s_fitted_well'],
                     bitwise=n['E1_matches_shipped_bitwise'],
                     sep_bank=b['sep'], sep_new=n['sep'],
                     d=b['d_third_mean'], lam=b['lambda_min']))

print("### per-cell (18 + 3 top-up)")
hdr = ("| R | seed | adm | banked s | E1 subtracted | E1 UNsub | U/S | "
       "E2 subtracted | E2 UNsub | U/S | Δ(E1sub − banked) |")
print(hdr); print('|' + '---|' * 11)
for r in rows:
    print(f"| {r['R']:.2f} | {r['seed']} | {'Y' if r['adm'] else 'n'} | {r['banked_s']:.6f} | "
          f"{r['e1s']:.6f} | {r['e1u']:.6f} | {r['r1']:.4f} | {r['e2s']:.6f} | {r['e2u']:.6f} | "
          f"{r['r2']:.4f} | {r['delta_banked']:.1e} |")

adm = [r for r in rows if r['adm']]
print(f"\nbitwise-identical to shipped well_fits(): {sum(r['bitwise'] for r in rows)}/{len(rows)}")
print(f"max |Δ(E1sub − banked)| over all cells: {max(abs(r['delta_banked']) for r in rows):.3e}")
print(f"max |sep_new − sep_bank|             : {max(abs(r['sep_new']-r['sep_bank']) for r in rows):.3e}")
print(f"\nE1 U/S over ADMISSIBLE cells (n={len(adm)}): mean {np.mean([r['r1'] for r in adm]):.4f} "
      f"sd {np.std([r['r1'] for r in adm], ddof=1):.4f} range [{min(r['r1'] for r in adm):.4f}, "
      f"{max(r['r1'] for r in adm):.4f}]")
print(f"E2 U/S over ADMISSIBLE cells (n={len(adm)}): mean {np.mean([r['r2'] for r in adm]):.4f} "
      f"sd {np.std([r['r2'] for r in adm], ddof=1):.4f} range [{min(r['r2'] for r in adm):.4f}, "
      f"{max(r['r2'] for r in adm):.4f}]")

# --- per-radius (bprime-c6's own aggregation: MEAN over admissible cells) ----
radii = sorted({r['R'] for r in rows})
per = []
for R in radii:
    g = [r for r in rows if r['R'] == R and r['adm']]
    d = float(np.mean([r['d'] for r in g]))
    m = lambda k: float(np.mean([r[k] for r in g]))
    per.append(dict(R=R, n=len(g), d=d, e1s=m('e1s'), e1u=m('e1u'), e2s=m('e2s'), e2u=m('e2u'),
                    banked=m('banked_s')))
print("\n### per radius (mean over admissible; banked convention)")
print("| R | n | d̄ | s banked=E1sub | d/s (corrected) | E1 UNsub | d/s (UNsub) | E2 sub | d/s (E2) | d/s atom-width 0.30 |")
print('|' + '---|' * 10)
for p in per:
    print(f"| {p['R']:.2f} | {p['n']} | {p['d']:.4f} | {p['e1s']:.4f} | {p['d']/p['e1s']:.3f} | "
          f"{p['e1u']:.4f} | {p['d']/p['e1u']:.3f} | {p['e2s']:.4f} | {p['d']/p['e2s']:.3f} | "
          f"{p['d']/0.30:.3f} |")
print(f"\nsweep-mean s: corrected(E1sub) {np.mean([p['e1s'] for p in per]):.4f} | "
      f"UNsub {np.mean([p['e1u'] for p in per]):.4f} | E2sub {np.mean([p['e2s'] for p in per]):.4f} "
      f"(bprime-c6 §1.1 quoted 0.4006 for the first of these)")

# --- the T5.5 law fit: invariant to the s convention -------------------------
def law(d, k, sigma_q=0.15):
    d = np.asarray(d, float); k = np.asarray(k, float)
    y = np.log(k) - np.log(d / sigma_q); x = d ** 2 - sigma_q ** 2
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef; ssr = float(((y - pred) ** 2).sum()); sst = float(((y - y.mean()) ** 2).sum())
    return dict(slope=float(coef[0]), s_implied=float(np.sqrt(-1 / (2 * coef[0]))),
                r2=1 - ssr / sst, decades=float(np.log10(k.max() / k.min())))

bcells = {key(c): c for c in bank}
gr, c0q, dd = [], [], []
for R in radii:
    g = [bcells[k] for k in bcells if abs(k[0] - R) < 1e-9 and bcells[k]['admissible']]
    dd.append(float(np.mean([c['d_third_mean'] for c in g])))
    gr.append(float(np.mean([c['grad_ratio'] for c in g])))
    c0q.append(float(np.mean([rw['coupling'] for c in g for rw in c['rows']
                              if rw['slot'] == 0 and rw['channel'] == 'q'])))
print("\n### the T5.5 suppression fit (bprime-c6 §1.1), recomputed from banked raw")
print(" gradient-ratio channel:", {k: round(v, 6) for k, v in law(dd, gr).items()})
print(" slot-coupling  channel:", {k: round(v, 6) for k, v in law(dd, c0q).items()})
print(" ⇒ s is the OUTPUT of this fit; no s enters it ⇒ R² and s_implied are")
print("   invariant to the subtraction convention BY CONSTRUCTION.")

# --- what the suppression predicts at the shipped cell under each s ----------
sig = 0.15
d_ship = [p['d'] for p in per if abs(p['R'] - 1.0) < 1e-9][0]
for name, s in [('E1 subtracted (banked, corrected)', per[4]['e1s']),
                ('E1 UNsubtracted', per[4]['e1u']),
                ('E2 subtracted (cat-test estimator)', per[4]['e2s']),
                ('law-fit s_implied', law(dd, gr)['s_implied']),
                ('atom-width proxy 0.30', 0.30)]:
    ds = d_ship / s
    kap = (d_ship / sig) * np.exp(-(d_ship ** 2 - sig ** 2) / (2 * s ** 2))
    print(f"  shipped cell, s={s:.4f} ({name}): d/s={ds:.3f}  exp(-½(d/s)²)={np.exp(-0.5*ds**2):.3e}"
          f"  T5.5 κ={kap:.3e}  (measured κ=0.01534)")

json.dump({"per_cell": rows, "per_radius": per,
           "law_gradient_ratio": law(dd, gr), "law_slot_coupling_q": law(dd, c0q)},
          open(OUT + 'remeasure_summary.json', 'w'), indent=2)
print("\nwrote", OUT + 'remeasure_summary.json')
