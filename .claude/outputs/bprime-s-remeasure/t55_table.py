import json
import numpy as np
S = json.load(open('.claude/outputs/bprime-s-remeasure/remeasure_summary.json'))
bank = json.load(open('.claude/outputs/bprime-c6/results/exp_route3_thirdparty.json'))['cells'] \
     + json.load(open('.claude/outputs/bprime-c6/results/thirdparty_topup_R042.json'))['cells']
sig = 0.15
t55 = lambda d, s: (d / sig) * np.exp(-(d ** 2 - sig ** 2) / (2 * s ** 2))
print("| R | d̄ | measured κ ± 2SE | T5.5 @ s=0.30 (atom width) | T5.5 @ E1 subtracted | "
      "T5.5 @ E1 UNsub | T5.5 @ E2 subtracted | ratio meas/T5.5(E1sub) |")
print('|' + '---|' * 8)
for p in S['per_radius']:
    g = [c for c in bank if abs(c['ball_radius'] - p['R']) < 1e-9 and c['admissible']]
    k = np.array([c['grad_ratio'] for c in g], float)
    se = k.std(ddof=1) / np.sqrt(len(k))
    d = p['d']
    print(f"| {p['R']:.2f} | {d:.4f} | {k.mean():.4e} ± {2*se:.1e} | {t55(d,0.30):.4e} | "
          f"{t55(d,p['e1s']):.4e} | {t55(d,p['e1u']):.4e} | {t55(d,p['e2s']):.4e} | "
          f"{k.mean()/t55(d,p['e1s']):.3f} |")
