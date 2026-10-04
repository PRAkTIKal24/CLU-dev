"""Convert the frozen decoder / judge-classifier pickles into array-only equinox leaf
files + a hyperparameter manifest, and prove the round trip by param-hash equality."""
import json, hashlib, sys
from pathlib import Path
import jax, equinox as eqx
from chlu.utils.checkpoints import load_model
from chlu.core.gen_bridge import param_hash, GenDecoder
from chlu.eval.gen_classifier import ExternalFactorCNN

ST = Path('/Users/batmac/clu/stage/outputs')
OUT = Path(sys.argv[1])
DEC = dict(feat_shape=[128, 4, 4], out_shape=[3, 64, 64], channels=[256, 128, 64], groups=8, blocks_per_stage=2)
CLF = dict(in_shape=[3, 64, 64], channels=[32, 64, 128, 128], kernel=5, stride=2, groups=8, hidden=256,
           cardinalities=[10, 10, 10, 8, 4, 15])
man = {"decoder": {"hyperparameters": DEC, "files": {}}, "classifier": {"hyperparameters": CLF, "files": {}}}
wh = json.load(open(ST / 'gen-decoder-lift/winner/winner_hashes.json'))
for s in (0, 1, 2):
    d = load_model(ST / f'gen-decoder-lift/winner/decoder_v1_s{s}.pkl')
    c = load_model(ST / f'gen-external-classifier/seed_{s}/external_classifier.pkl')
    cm = json.load(open(ST / f'gen-external-classifier/seed_{s}/classifier_manifest.json'))
    assert param_hash(d) == wh[f'seed_{s}']['param_hash']
    assert param_hash(c) == cm['classifier']['param_hash']
    for kind, m, hp, cls in (('decoder', d, DEC, GenDecoder), ('classifier', c, CLF, ExternalFactorCNN)):
        f = OUT / f'{kind}_s{s}.eqx'
        eqx.tree_serialise_leaves(f, m)
        skel = cls(**{k: (tuple(v) if k.endswith('shape') else v) for k, v in hp.items()}, key=jax.random.PRNGKey(0))
        back = eqx.tree_deserialise_leaves(f, skel)
        assert param_hash(back) == param_hash(m), (kind, s)
        man[kind]['files'][str(s)] = {"file": f.name, "param_hash": param_hash(m),
                                      "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                      "bytes": f.stat().st_size}
        print(kind, s, 'round-trip param_hash OK', param_hash(m)[:16], f.stat().st_size)
(OUT / 'weights.json').write_text(json.dumps(man, indent=1))
