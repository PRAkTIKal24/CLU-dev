"""Assemble the shipped weights/ tree (original directory layout, array-only .eqx files,
sanitised manifests, one registry weights/weights.json). Inputs: assembly/weights (decoder,
classifier) and assembly/weights_encoder (encoder run). Tooling only."""
import json, shutil, hashlib, sys
from pathlib import Path
A = Path('/Users/batmac/clu/outputs/w24-repro-package/assembly')
ST = Path('/Users/batmac/clu/stage/outputs')
OUT = Path(sys.argv[1])
if OUT.exists():
    shutil.rmtree(OUT)
reg = {}
old = json.load(open(A / 'weights' / 'weights.json'))
wh = json.load(open(ST / 'gen-decoder-lift/winner/winner_hashes.json'))
(OUT / 'decoder').mkdir(parents=True)
win = {}
for s in (0, 1, 2):
    e = old['decoder']['files'][str(s)]
    f = OUT / 'decoder' / f'decoder_v1_s{s}.eqx'
    shutil.copy(A / 'weights' / e['file'], f)
    reg[str(f.relative_to(OUT))] = dict(cls='GenDecoder', hyperparameters=old['decoder']['hyperparameters'],
                                        param_hash=e['param_hash'], sha256=e['sha256'], bytes=e['bytes'])
    w = wh[f'seed_{s}']
    win[f'seed_{s}'] = {k: w[k] for k in ('param_hash', 'param_count', 'recipe', 'encoder_param_hash_it_was_fit_against')}
(OUT / 'decoder' / 'winner_hashes.json').write_text(json.dumps(win, indent=1) + '\n')
for s in (0, 1, 2):
    e = old['classifier']['files'][str(s)]
    d = OUT / 'classifier' / f'seed_{s}'
    d.mkdir(parents=True)
    f = d / 'external_classifier.eqx'
    shutil.copy(A / 'weights' / e['file'], f)
    reg[str(f.relative_to(OUT))] = dict(cls='ExternalFactorCNN', hyperparameters=old['classifier']['hyperparameters'],
                                        param_hash=e['param_hash'], sha256=e['sha256'], bytes=e['bytes'])
    cm = json.load(open(ST / f'gen-external-classifier/seed_{s}/classifier_manifest.json'))
    keep = {'classifier': {k: cm['classifier'][k] for k in ('param_hash', 'param_count', 'channels', 'kernel', 'hidden', 'steps', 'lr', 'batch')},
            'gate': {'external': {'balanced_accuracy': cm['gate']['external']['balanced_accuracy']}},
            'dataset': cm['dataset'], 'condition': cm['condition']}
    (d / 'classifier_manifest.json').write_text(json.dumps(keep, indent=1) + '\n')
shutil.copy('/Users/batmac/clu/evidence/outputs/gen-external-classifier/classifier_gate.json', OUT / 'classifier' / 'classifier_gate.json')
src = A / 'weights_encoder' / 'shapes3d_recon'
for s in (0, 1, 2):
    d = OUT / 'encoder' / 'shapes3d_recon' / f'seed_{s}'
    d.mkdir(parents=True)
    models = json.load(open(src / f'seed_{s}' / 'models.json'))
    for name, m in models.items():
        f = d / m['file']
        shutil.copy(src / f'seed_{s}' / m['file'], f)
        hp = {k: v for k, v in m.items() if k not in ('cls', 'file', 'param_hash', 'sha256', 'bytes')}
        reg[str(f.relative_to(OUT))] = dict(cls=m['cls'], hyperparameters=hp, param_hash=m['param_hash'], sha256=m['sha256'], bytes=m['bytes'])
    for n in ('bridge_manifest.json', 'encoder_manifest.json', 'feature_cache.json', 'shared_decoder_hash.json'):
        shutil.copy(src / f'seed_{s}' / n, d / n)
(OUT / 'weights.json').write_text(json.dumps({'files': reg}, indent=1) + '\n')
tot = sum(v['bytes'] for v in reg.values())
print(len(reg), 'weight files', tot, 'bytes')
