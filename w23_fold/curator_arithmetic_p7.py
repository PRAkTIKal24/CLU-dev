"""w23 fold, curator arithmetic (CA7): every number the pass-7 registry text derives, read from artefacts. Read-only; fits and
scores nothing. usage: curator_arithmetic_p7.py <out json>"""
import json, re, sys, hashlib
from decimal import Decimal
from pathlib import Path
EV = Path('/Users/batmac/clu/evidence/outputs'); OUT = Path('/Users/batmac/clu/outputs'); REG = OUT / 'w18-seeds-and-tables'
ca = {}
# PoE / LACE budget multiples (GENTABLE-KSTAR.json)
g = json.load(open(EV / 'w15-lace-poe-rows/GENTABLE-KSTAR.json'))['PART_B_rows_3_4']['arms']
poe = g['PoE']['budget_per_seed']; lace = g['LACE']['budget_per_seed']
ca['poe'] = {'x_store_reference_per_seed': {s: poe[s]['x_store_reference'] for s in poe}, 'deployed_bytes': poe['0']['deployed_bytes'],
             'expert_params': poe['0']['expert_params'], 'reference_bytes': 524288,
             'quotient_exact': str(Decimal(poe['0']['deployed_bytes']) / Decimal(524288)),
             'quotient_2dp_half_up': str((Decimal(poe['0']['deployed_bytes']) / Decimal(524288)).quantize(Decimal('0.01'))),
             'quotient_4dp': str((Decimal(poe['0']['deployed_bytes']) / Decimal(524288)).quantize(Decimal('0.0001'))),
             'truncated_2dp': str((Decimal(poe['0']['deployed_bytes']) / Decimal(524288)).quantize(Decimal('0.01'), rounding='ROUND_DOWN'))}
ca['lace'] = {'x_classifier_only': lace['0']['x_store_reference_classifier_only'], 'x_with_whitening': lace['0']['x_store_reference_with_whitening'],
              'bytes_classifier_only': lace['0']['deployed_bytes_classifier_only'],
              'quotient_classifier_only_exact': str(Decimal(lace['0']['deployed_bytes_classifier_only']) / Decimal(524288)),
              'quotient_with_whitening_exact': str(Decimal(lace['0']['deployed_bytes_classifier_plus_whitening']) / Decimal(524288))}
# AMENDMENT-1 (b) rendered-feature deviation, 9 legs (SHAPES-MATCHED.json)
d = json.load(open(EV / 'w20-shapes-data-matched/SHAPES-MATCHED.json'))['deletion_matched']['per_scope']
dev = {f'{sc}/{s}': d[sc]['store'][s]['b']['b2_book']['rendered_features_rel_dev_on_survivor_requests'] for sc in d for s in d[sc]['store']}
ca['rendered_dev'] = {'per_leg': dev, 'min': min(dev.values()), 'argmin': min(dev, key=dev.get), 'max': max(dev.values()), 'argmax': max(dev, key=dev.get),
                      'item_scope': [dev[k] for k in sorted(dev) if k.startswith('item/')], 'n_legs': len(dev)}
# drafts: read only
for f in ('DRAFT-v2.tex', 'DRAFT-v2-style.tex'):
    t = (OUT / 'w17-paper' / f).read_text(encoding='utf-8'); L = t.splitlines()
    ca[f] = {'md5': hashlib.md5(t.encode()).hexdigest(),
             'placeholder_uses': sum(1 for l in L if re.match(r'^[^%]*\\PLACEHOLDER\{', l)),
             'placeholder_use_lines': [i for i, l in enumerate(L, 1) if re.match(r'^[^%]*\\PLACEHOLDER\{', l)],
             'todo_uses': sum(1 for l in L if re.match(r'^[^%]*\\TODO\{', l)),
             'figslot_uses': [re.search(r'\\figslot\{([^}]*)\}', l).group(1) for l in L if re.match(r'^[^%]*\\figslot\{', l)],
             'lines_27.135x': [i for i, l in enumerate(L, 1) if '27.135$\\times$' in l],
             'lines_27.13x': [i for i, l in enumerate(L, 1) if '27.13$\\times$' in l]}
    ca[f]['n_figslot_uses'] = len(ca[f]['figslot_uses'])
# the register's pass-2 manifest: how many of its cells does the build actually read?
m = json.load(open(REG / 'draft_pass3/pass2-manifest.json')); h = json.load(open(REG / 'TABLES-w18.json'))['cells_historical']
ca['pass2_manifest'] = {'cells': len(m['cells']), 'read_by_build_historical': len(h), 'unread': len(m['cells']) - len(h), 'first_historical': sorted(h)[0]}
json.dump(ca, open(sys.argv[1], 'w'), indent=1, ensure_ascii=False)
print(json.dumps(ca, indent=1, ensure_ascii=False)[:3000])
# register pass (w23 fold): counts, with the comma-grouped spellings the registry text uses
C = Path('/Users/batmac/clu/outputs/w23-fold/curator')
gs = json.load(open(C / 'gen_stats.json')); bef = json.load(open(C / 'register_coverage_before.json')); aft = json.load(open(C / 'register_coverage_after.json'))
b = (C / 'install-build.txt').read_text(); m5 = re.search(r'PASS: (\d+) cells', b)
reg = {'cells_after': int(m5.group(1)), 'cells_pass5': 2282, 'cells_added': int(m5.group(1)) - 2282,
       'registered_v2': gs['stats']['V']['registered'], 'registered_style': gs['stats']['S']['registered'],
       'source_cells': gs['n_source_cells'], 'source_files': gs['n_source_files'],
       'tokens_v2': bef['DRAFT-v2.tex']['tokens_under_any_SRC'], 'tokens_style': bef['DRAFT-v2-style.tex']['tokens_under_any_SRC'],
       'covered_before_v2': bef['DRAFT-v2.tex']['covered_by_a_draft_cell'], 'covered_after_v2': aft['DRAFT-v2.tex']['covered_by_a_draft_cell'],
       'covered_before_style': bef['DRAFT-v2-style.tex']['covered_by_a_draft_cell'], 'covered_after_style': aft['DRAFT-v2-style.tex']['covered_by_a_draft_cell']}
reg['uncovered_after_v2'] = reg['tokens_v2'] - reg['covered_after_v2']; reg['uncovered_after_style'] = reg['tokens_style'] - reg['covered_after_style']
reg['w22_lines_v2'] = gs['stats']['V']['w22_lines']; reg['w22_lines_style'] = gs['stats']['S']['w22_lines']
reg['strong_v2'] = gs['stats']['V']['registered_strong_4plus_digits']; reg['short_v2'] = reg['registered_v2'] - reg['strong_v2']
vr = json.load(open(C / 'verify_register7.json')); reg['tables_md_inserted_lines'] = [v['inserted_lines'] for k, v in vr.items() if k.endswith('/TABLES.md')][0]
reg['formatted'] = {k: f"{v:,}" for k, v in reg.items() if isinstance(v, int)}
ca['register'] = reg
json.dump(ca, open(sys.argv[1], 'w'), indent=1, ensure_ascii=False)
print(json.dumps(reg, indent=1))
