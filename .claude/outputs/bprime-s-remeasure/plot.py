import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = json.load(open('/Users/user/Desktop/CHLU/.claude/outputs/bprime-s-remeasure/remeasure_summary.json'))
cells, per = S['per_cell'], S['per_radius']
R = [p['R'] for p in per]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

for k, lab, c, mk in [('e1s', 'E1 $-\\alpha\\|q\\|^2$ (banked = corrected)', 'tab:blue', 'o'),
                      ('e1u', 'E1 UNsubtracted', 'tab:red', 's'),
                      ('e2s', 'E2 (cat-test est.) $-\\alpha\\|q\\|^2$', 'tab:green', '^'),
                      ('e2u', 'E2 UNsubtracted', 'tab:orange', 'v')]:
    ax[0].plot(R, [p[k] for p in per], mk + '-', color=c, label=lab)
    ax[0].scatter([r['R'] for r in cells if r['adm']], [r[k] for r in cells if r['adm']],
                  s=9, color=c, alpha=0.35)
ax[0].axhline(0.318, ls=':', color='k', label='orgdiv-cat-test $s=0.318$ (other store)')
ax[0].axhline(0.30, ls='--', color='gray', label='atom_width proxy 0.30')
ax[0].set_xlabel('ball_radius'); ax[0].set_ylabel('fitted $s$')
ax[0].set_title('$s$ under both conventions (points = admissible cells)')
ax[0].legend(fontsize=7)

for k, lab, c, mk in [('e1s', 'corrected (banked)', 'tab:blue', 'o'),
                      ('e1u', 'UNsubtracted', 'tab:red', 's'),
                      ('e2s', 'E2 corrected', 'tab:green', '^')]:
    ax[1].plot(R, [p['d'] / p[k] for p in per], mk + '-', color=c, label=lab)
ax[1].plot(R, [p['d'] / 0.30 for p in per], 'd--', color='gray', label='atom-width ruler')
ax[1].set_xlabel('ball_radius'); ax[1].set_ylabel('$d/s$')
ax[1].set_title('$d/s$: the correction does NOT move the blue curve')
ax[1].legend(fontsize=8)

r1 = [r['e1u'] / r['e1s'] for r in cells if r['adm']]
r2 = [r['e2u'] / r['e2s'] for r in cells if r['adm']]
ax[2].hist([r1, r2], bins=8, label=[f'E1  mean {np.mean(r1):.3f}', f'E2  mean {np.mean(r2):.3f}'])
ax[2].axvline(1.44, color='k', ls=':', label='N224 1.44$\\times$ (cat-test store)')
ax[2].axvline(1.0, color='gray', ls='--')
ax[2].set_xlabel('inflation factor  $s_{\\rm UNsub}/s_{\\rm sub}$ (18 admissible cells)')
ax[2].set_title('the hazard, measured on THIS rig')
ax[2].legend(fontsize=8)
fig.tight_layout()
out = '/Users/user/Desktop/CHLU/.claude/outputs/bprime-s-remeasure/s_conventions.png'
fig.savefig(out, dpi=140)
print('wrote', out)
