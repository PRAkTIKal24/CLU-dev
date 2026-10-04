"""Pass-5 curator arithmetic. Reads evidence JSONs only (read-only); writes curator_arithmetic.json.
Every derived number the registry quotes that is NOT printed by a spoke is produced here."""
import json, statistics as st
E = '/Users/batmac/clu/evidence/outputs/'
out = {}
# (1) w19 relabel sigma check: sd(ddof=1) of CLU - R* per seed, from w19 BARS JSONs
rel = {}
for cond, f in [('WORDS-MAIN', 'val/BARS-WORDS-MAIN-val.json'), ('WORDS-FULL', 'val/BARS-WORDS-FULL-val.json'),
                ('OBS', 'val/BARS-OBS-val.json'), ('OBS-X10', 'val/BARS-OBS-X10-val.json'), ('BOTH-TEST', 'test/BARS-BOTH-test.json')]:
    r = json.load(open(E + 'w19-maze-t-symmetric/' + f))['rows']['nway']
    clu = r['CLU_per_seed']
    rs = r['per_seed'][r['R_star']]
    rel[cond] = {'R_star': r['R_star']}
    if rs:
        d = [a - b for a, b in zip(clu, rs)]
        rel[cond].update(paired_diff=d, mean=sum(d) / 3, sd_ddof1=st.stdev(d), ahead=sum(x > 0 for x in d))
out['w19_relabel_check'] = rel
# (2) maze-t-seed4: margin under both estimands, 3 vs 6 seeds
b = json.load(open(E + 'w21-maze-t-seed4/MAZE-T-SEED4.json'))['nway']['BOTH-TEST']
clu = [b[str(s)]['CLU'] for s in range(6)]
riv = {}
for s in range(6):
    for k, v in b[str(s)]['all_arms'].items():
        if k.startswith('rival'):
            riv.setdefault(k, []).append(v)
per_seed_best = [max(riv[k][s] for k in riv) for s in range(6)]
dps = [c - m for c, m in zip(clu, per_seed_best)]
def rstar(n):
    means = {k: sum(v[:n]) / n for k, v in riv.items()}
    k = max(means, key=means.get)
    d = [clu[s] - riv[k][s] for s in range(n)]
    return {'R_star': k, 'R_star_mean': means[k], 'CLU_mean': sum(clu[:n]) / n, 'margin': sum(clu[:n]) / n - means[k],
            'paired_diff': d, 'sd_ddof1_paired_diff': st.stdev(d), 'seeds_ahead': sum(x > 0 for x in d)}
out['seed4_estimands'] = {
    'per_seed_best_margin_mean_6': sum(dps) / 6, 'per_seed_best_margin_mean_3': sum(dps[:3]) / 3,
    'per_seed_best_diff': dps, 'R_star_by_mean_3_seeds': rstar(3), 'R_star_by_mean_6_seeds': rstar(6),
    'note': 'curator arithmetic over MAZE-T-SEED4.json nway.BOTH-TEST.<seed>.all_arms (rival* keys only); no label computed'}
# (3) w20 rival after-minus-before on red^sphere and all scopes
d = json.load(open(E + 'w20-shapes-data-matched/SHAPES-MATCHED.json'))
ps = d['deletion_matched']['per_scope']
out['w20_rival_after_minus_before'] = {sc: {fam: [ps[sc]['rivals'][fam][s]['rival_after_minus_before'] for s in '012']
                                            for fam in ps[sc]['rivals']} for sc in ps}
out['w20_rival_after_minus_before_family_mean_conj'] = {fam: sum(v) / 3 for fam, v in out['w20_rival_after_minus_before']['conjunction_live'].items()}
out['w20_store_max_a_mode_dev'] = max(ps[sc]['store'][s]['a']['max_abs_mode_dev'] for sc in ps for s in '012')
out['w20_store_max_b2_rendered_rel_dev'] = max(ps[sc]['store'][s]['b']['b2_book']['rendered_features_rel_dev_on_survivor_requests'] for sc in ps for s in '012')
out['w20_store_max_b2_abs_W_dev'] = max(ps[sc]['store'][s]['b']['b2_book']['max_abs_W_dev'] for sc in ps for s in '012')
r = d['rows34']['P3_margin']['store_K32_n131072']['per_family']['mlp']
out['w20_rows34_P1_carried_rule_literal'] = {'per_seed': [r[s]['delta'] for s in '012'], 'mean': r['mean_delta'],
    'carried': [r[s]['carried_rival'] for s in '012'],
    'note': 'per_family.mlp.delta uses the P1-carried fit (seed 2 = DECLARED, P1 PASS); the headline uses max(DECLARED,TENX)'}
json.dump(out, open('/Users/batmac/clu/outputs/w21-registry-fold/curator/curator_arithmetic.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:6000])
# (4) staged file count for w20 (the Hub report says 1,482 mirrored)
import os
n = sum(len(fs) for _, _, fs in os.walk(E + 'w20-shapes-data-matched'))
out['w20_staged_file_count_on_batmac'] = n
json.dump(out, open('/Users/batmac/clu/outputs/w21-registry-fold/curator/curator_arithmetic.json', 'w'), indent=1)
print('files', n)
