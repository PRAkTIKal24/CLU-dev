"""Re-derive every CMAZE-T number the w22 draft-rulings edit writes, from the per-seed run JSONs."""
import json, os
EV = os.path.expanduser('~/clu/evidence/outputs/')
def J(p): return json.load(open(EV + p))
A1 = ['w19-maze-t-symmetric/test/BOTH_test_seed%d.json' % s for s in (0, 1, 2)] + \
     ['w21-maze-t-seed4/test/BOTH_test_seed%d.json' % s for s in (3, 4, 5)]
REL = ['w21-maze-t-clu-improve/test/BOTH_test_seed%d.json' % s for s in (0, 1, 2)] + \
      ['w21-rel-newseeds/test/BOTH_test_seed%d.json' % s for s in (3, 4, 5)]
TRAINED = ('CLU', 'rival', 'diagnostic', 'control:', 'arm:', 'floor:')
def summ(paths, tag):
    out = []
    for s, p in enumerate(paths):
        n = J(p)['ROWS']['nway']
        clu = n['CLU']
        matched = {k: v for k, v in n.items() if k.startswith('rival:') or k.startswith('rival@10x-steps:')}
        rel = {k: v for k, v in n.items() if '+rel' in k}
        ctrl = {k: v for k, v in n.items() if k.startswith('control:')}
        ceil_arms = {k: v for k, v in n.items() if k != 'n' and k.startswith(TRAINED) and '+rel' not in k}
        bm = max(matched, key=matched.get)
        c = max(ceil_arms.values())
        row = dict(seed=s, n=n['n'], clu=clu, best_matched=(bm, matched[bm]), margin=clu - matched[bm],
                   ceiling=c, ceiling_by=[k for k, v in ceil_arms.items() if v == c],
                   max_ctrl=max(ctrl.items(), key=lambda kv: kv[1]),
                   best_rel=max(rel.items(), key=lambda kv: kv[1]) if rel else None)
        out.append(row)
        print(tag, s, 'n=%d CLU=%.4f best=%s %.4f margin=%+.4f ceil=%.4f by=%s maxctrl=%s %.4f' % (
            row['n'], clu, bm, matched[bm], row['margin'], c, row['ceiling_by'], row['max_ctrl'][0], row['max_ctrl'][1]),
            ('best+rel=%s %.4f clu-rel=%+.4f' % (row['best_rel'][0], row['best_rel'][1], clu - row['best_rel'][1])) if rel else '')
    m = sum(r['margin'] for r in out) / len(out)
    print(tag, 'six-seed mean margin %+.4f  (%+.3f)' % (m, m), ' mean CLU %.4f' % (sum(r['clu'] for r in out) / 6))
    return out
a1 = summ(A1, 'A1 ')
rl = summ(REL, 'REL')
for tag, paths in (('A1 ', A1), ('REL', REL)):
    for s, p in enumerate(paths):
        d = J(p); ro = d.get('ROLLOUT', {})
        print(tag, s, 'rollout', {k: v for k, v in (ro.get('validity', ro) if isinstance(ro, dict) else {}).items() if 'CLU' in k} if isinstance(ro, dict) else ro)
