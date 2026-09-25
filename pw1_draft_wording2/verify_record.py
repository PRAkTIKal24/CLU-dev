#!/usr/bin/env python3
"""Verify the five seed-5 numbers of PW-1 item 7 verbatim in the W23 diagnosis record.
Prints value -> file -> path (JSON leaf where one exists, else prose line). Exit 1 if any fails."""
import json, os, sys
R = os.path.expanduser('~/clu/evidence/outputs/w23-seed5-diagnosis/')
ok = True
def out(v, f, p, good):
    global ok; ok &= good
    print('%-6s | %-26s | %-70s | %s' % (v, f, p, 'OK' if good else 'FAIL'))
qc = json.load(open(R + 'Q1-CONTENT.json'))
code = [c for c in qc['codes'] if c['code'] == [1, 3, 0, 2]]
assert len(code) == 1; c = code[0]; i = qc['codes'].index(c)
diag = json.load(open(R + 'W23-SEED5-DIAGNOSIS.json'))['Q1']['failing']
em = dict(c['clu_in_sample_emits_top3'])
out('0.04', 'Q1-CONTENT.json', 'codes[%d] (code [1,3,0,2]) ["clu.pkl"].oracle_disp_margin = %r (rounds to 0.04)' % (i, c['clu.pkl']['oracle_disp_margin']), round(c['clu.pkl']['oracle_disp_margin'], 2) == 0.04)
for t in ('2,0,1,2', '2,0,3,2'):
    m = diag[t]['l2_oracle_margins'][1]
    out('0.04', 'W23-SEED5-DIAGNOSIS.json', 'Q1.failing["%s"].l2_oracle_margins[1] (displacement) = %r' % (t, m), round(m, 2) == 0.04)
F = open(R + 'FINDINGS.md').read().split('\n')
for n, l in enumerate(F, 1):
    if '0.04' in l and ('margin' in l):
        out('0.04', 'FINDINGS.md', 'l.%d verbatim: ...%s...' % (n, l[max(0, l.find('0.04')-50):l.find('0.04')+14].strip()), True)
out('317', 'Q1-CONTENT.json', 'codes[%d].n_train = %r' % (i, c['n_train']), c['n_train'] == 317)
for t in ('2,0,1,2', '2,0,3,2'):
    out('317', 'W23-SEED5-DIAGNOSIS.json', 'Q1.failing["%s"].truth_code_fit_rows = %r' % (t, diag[t]['truth_code_fit_rows']), diag[t]['truth_code_fit_rows'] == 317)
for v, k, name in ((211, '(1, 3, 0, 2)', 'S (truth)'), (63, '(1, 2, 0, 2)', 'E'), (43, '(1, 1, 0, 2)', 'N')):
    out(str(v), 'Q1-CONTENT.json', 'codes[%d].clu_in_sample_emits_top3 %s -> %r  [%s]' % (i, k, em.get(k), name), em.get(k) == v)
out('0.666', 'Q1-CONTENT.json', 'codes[%d].clu_in_sample_exact = %r (211/317 = %.10f)' % (i, c['clu_in_sample_exact'], 211/317), abs(c['clu_in_sample_exact'] - 211/317) < 1e-12)
for n in (30, 71):
    out('prose', 'FINDINGS.md', 'l.%d: %s' % (n, F[n-1].strip()[:150]), ('211' in F[n-1] and '317' in F[n-1]))
out('106', '(arithmetic)', '317 - 211 = %d ; 63 + 43 = %d ; emits sum %d = n_train' % (317-211, 63+43, sum(em.values())), 317-211 == 106 == 63+43 and sum(em.values()) == 317)
print('ALL OK' if ok else 'FAILURE: STOP, install nothing'); sys.exit(0 if ok else 1)
