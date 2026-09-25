"""Pass-6 curator arithmetic. READS evidence only (JSON / git blobs copied read-only); means, differences, counts. No fit, no score."""
import json, math, statistics as st, pathlib
E = pathlib.Path('/Users/batmac/clu/evidence/outputs')
OUT = pathlib.Path(__file__).with_suffix('.json')
F3 = pathlib.Path(__file__).parent / 'src_w22_figures3'
R = {}
# ---------------- 1. rel-newseeds ----------------
rel = json.load(open(E / 'w21-rel-newseeds/REL-NEWSEEDS.json'))
S = rel['nway_summary']; seeds = [str(s) for s in range(6)]
g = lambda k: [S[k][s] for s in seeds]
r = {}
r['REL_CLU'] = g('REL CLU'); r['A1_CLU'] = g('A1 CLU')
r['REL_CLU_mean6'] = st.mean(g('REL CLU')); r['A1_CLU_mean6'] = st.mean(g('A1 CLU'))
r['REL_minus_A1_per_seed'] = [a - b for a, b in zip(g('REL CLU'), g('A1 CLU'))]
r['A1_CLU_sorted_ascending'] = sorted(zip(g('A1 CLU'), range(6)))
r['A1_margin_per_seed'] = [a - b for a, b in zip(g('A1 CLU'), g('A1 best matched rival'))]
r['class_from_maze_t_seed4'] = rel['class_from_maze_t_seed4']
r['CLU_minus_best_rel'] = [a - b for a, b in zip(g('REL CLU'), g('REL best +rel'))]
r['CLU_minus_best_rel_mean6'] = st.mean(r['CLU_minus_best_rel'])
r['CLU_minus_best_matched'] = [a - b for a, b in zip(g('REL CLU'), g('REL best matched rival'))]
r['CLU_minus_best_matched_mean6'] = st.mean(r['CLU_minus_best_matched'])
r['rel_lift'] = [a - b for a, b in zip(g('REL best +rel'), g('REL best matched rival'))]
r['share_recovered'] = [(l / m if abs(m) > 1e-12 else None) for l, m in zip(r['rel_lift'], r['CLU_minus_best_matched'])]
r['CLU_minus_max_control'] = [a - b for a, b in zip(g('REL CLU'), g('REL max control'))]
r['CLU_at_ceiling_all6'] = all(abs(a - b) < 1e-15 for a, b in zip(g('REL CLU'), g('REL ceiling')))
r['ceiling_incl_rel_above_CLU'] = {s: S['REL ceiling incl +rel'][s] - S['REL CLU'][s] for s in seeds if S['REL ceiling incl +rel'][s] > S['REL CLU'][s]}
# per-tuple control equality (zero-fit replay files)
pt = {}
for s in range(6):
    P = json.load(open(E / f'w21-rel-newseeds/replay/BOTH_test_seed{s}_PERTUPLE.json'))['PER_TUPLE_NWAY']
    c = P['CLU']
    pt[s] = {'n_tuples': len(c), 'CLU_exact_rows': sum(v[1] for v in c.values()), 'n_rows': sum(v[0] for v in c.values()),
             'controls': {k: {'tuples_equal_to_CLU': sum(P[k][t] == c[t] for t in c), 'exact_rows': sum(v[1] for v in P[k].values())}
                          for k in P if k.startswith('control')}}
r['pertuple_controls'] = pt
r['seed4_CLU_exact_frac'] = pt[4]['CLU_exact_rows'] / pt[4]['n_rows']
r['seed4_random_atoms_exact_frac'] = pt[4]['controls']['control: random_atoms']['exact_rows'] / pt[4]['n_rows']
flags = rel['rival_nonconvergence_flags']
r['A1_flags_seed5'] = [f for f in flags['A1'] if f['seed'] == 5] if 'A1' in flags else None
r['REL_flags_seeds345'] = [f for f in flags['CLU-REL-A10'] if f['seed'] >= 3]
r['REL_worse80k_seeds345'] = [f for f in r['REL_flags_seeds345'] if any('WORSE' in x for x in f['flags'])]
R['rel_newseeds'] = r
# ---------------- 2. ladder-rivals-32k ----------------
L = json.load(open(E / 'w21-ladder-rivals-32k/LADDER-RIVALS-32K.json'))
x1 = []; lad = {}
for n, rr in L['rungs'].items():
    X = rr['crosscheck']['X1_same_rows_same_steps']
    for s, v in X['per_seed'].items():
        p = v['paired_mine_minus_ref']
        x1.append({'rung': int(n), 'seed': int(s), 'delta': p['delta'], 'ci95': p['ci95'], 'excl0': p['ci95'][0] > 0 or p['ci95'][1] < 0,
                   'steps_mine': v['mine_steps'], 'steps_ref': v['ref_steps'], 'rows': v['mine_n_fit_rows']})
    rep = rr['crosscheck']['X1_diagnostic_ref_recipe_replay']['per_seed']
    m = rr['margins_K32_store_minus_rival']
    lad[n] = {k: {'mean': m[k]['mean_delta']['mean'], 'per_seed': m[k]['mean_delta']['per_seed'],
                  'ci_low': [m[k]['per_seed'][s]['ci95'][0] for s in ('0', '1', '2')]}
              for k in m if k != 'setB_integrity_vs_w20_printed'}
    lad[n]['setB_identical_to_w20'] = m['setB_integrity_vs_w20_printed']['identical']
    lad[n]['x1_replay_corners_differing'] = [rep[s]['n_corners_differing'] for s in ('0', '1', '2')]
ds = [e['delta'] for e in x1]
X1 = {'legs': x1, 'n_legs': len(x1), 'n_ci_excl0': sum(e['excl0'] for e in x1), 'min_delta': min(ds), 'max_delta': max(ds),
      'max_abs_delta': max(abs(d) for d in ds), 'n_positive': sum(d > 0 for d in ds),
      'sign_by_seed': {s: [e['delta'] > 0 for e in x1 if e['seed'] == s] for s in (0, 1, 2)},
      'rms_delta': math.sqrt(st.mean([d * d for d in ds])),
      'single_fit_sd_if_independent_draws = rms/sqrt2': math.sqrt(st.mean([d * d for d in ds])) / math.sqrt(2),
      'mean_abs_delta': st.mean(abs(d) for d in ds)}
R['ladder_X1'] = X1
K = {'A': 'setA_vs_strongest', 'B': 'setB_vs_strongest_declared_or_tenx_RECOMPUTED', 'C': 'setC_vs_strongest_declared_or_tenx',
     'C_sm': 'setC_vs_strongest_stepmatched', 'C_best': 'setC_vs_best_of_every_fit'}
lr = {'per_rung': lad}
for a, k in K.items():
    lr[f'mean_over_rungs_{a}'] = st.mean(lad[n][k]['mean'] for n in lad)
    lr[f'min_ci_low_{a}'] = min(min(lad[n][k]['ci_low']) for n in lad)
    lr[f'min_per_seed_margin_{a}'] = min(min(lad[n][k]['per_seed']) for n in lad)
    lr[f'n_per_seed_margins_below_X1_max_abs_{a}'] = sum(x < X1['max_abs_delta'] for n in lad for x in lad[n][k]['per_seed'])
lr['C_minus_B_per_rung'] = {n: lad[n][K['C']]['mean'] - lad[n][K['B']]['mean'] for n in lad}
lr['Csm_minus_B_per_rung'] = {n: lad[n][K['C_sm']]['mean'] - lad[n][K['B']]['mean'] for n in lad}
R['ladder_margins'] = lr
# w20 headline, for the X1 comparison (read, not recomputed)
H = json.load(open(E / 'w20-shapes-data-matched/SHAPES-MATCHED.json'))['headline']
R['w20_headline_read'] = {k: H[k] for k in H if 'margin' in k}
R['w20_seed2_margin_minus_X1_max_abs'] = H['data_matched_margin_rows34_K32']['per_seed']['2'] - X1['max_abs_delta']
R['X1_max_abs_over_w20_mean'] = X1['max_abs_delta'] / H['data_matched_margin_rows34_K32']['mean']
# ---------------- 3. figures3 (git blobs copied read-only from w22/figures3 @cb5e0ac) ----------------
A = json.load(open(F3 / 'AMBIGUITY.json'))
t = A['A1_w19']['all_arms_in_items_file']['total']; br = A['A1_w19']['all_arms_in_items_file']['by_reason_pair']
Bf = json.load(open(F3 / 'AMBIGUITY-before.json'))
R['figures3'] = {'A1_total': t, 'A1_by_reason': br, 'A1_by_reason_sum': sum(br.values()),
                 'A1_pixel_different_opposite_mark_pairs = opp - pixels': t['opposite_mark_pairs'] - t['same_picture']['pixels'],
                 'A1_state_different = opp - state': t['opposite_mark_pairs'] - t['same_picture']['state'],
                 'A1_state+legal minus state+legal+pocket': t['same_picture']['state+legal'] - t['same_picture']['state+legal+pocket'],
                 'before_file_by_reason': Bf['all_arms_in_items_file']['by_reason_pair'],
                 'REL_total': A['CLU_REL_A10']['all_arms_in_items_file']['total'],
                 'glyph_blocks_equal': sum(b['disc_pixels_equal'] for b in json.load(open(F3 / 'GLYPH-IDENTITY.json'))['blocks']),
                 'glyph_blocks': len(json.load(open(F3 / 'GLYPH-IDENTITY.json'))['blocks'])}
json.dump(R, open(OUT, 'w'), indent=1, default=str)
print(json.dumps({k: (v if k != 'rel_newseeds' else {kk: vv for kk, vv in v.items() if kk != 'pertuple_controls'}) for k, v in R.items() if k != 'ladder_margins'}, indent=1, default=str)[:9000])
print(json.dumps({k: v for k, v in lr.items() if k != 'per_rung'}, indent=1))
