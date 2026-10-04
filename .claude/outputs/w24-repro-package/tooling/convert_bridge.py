"""Convert the frozen-encoder run (encoder, probe, bridge decoder) per seed to array-only
equinox files + models.json; prove the round trip by parameter-hash equality against the
original pickles and the run's manifests."""
import json, hashlib, sys
from pathlib import Path
import jax, equinox as eqx
from chlu.utils.checkpoints import load_model
from chlu.core.gen_bridge import param_hash, GenEncoder, GenDecoder, FactorProbe

SRC = Path('/Users/batmac/clu/stage/outputs/gen-feature-bridge/runs/shapes3d_recon')
OUT = Path(sys.argv[1])
CARDS = [10, 10, 10, 8, 4, 15]
for s in (0, 1, 2):
    d = SRC / f'seed_{s}'
    o = OUT / 'shapes3d_recon' / f'seed_{s}'
    o.mkdir(parents=True, exist_ok=True)
    bm = json.load(open(d / 'bridge_manifest.json'))
    em = json.load(open(d / 'encoder_manifest.json'))
    enc, prb, dec = (load_model(d / f'{n}.pkl') for n in ('encoder', 'probe', 'decoder'))
    assert param_hash(enc) == bm['encoder_param_hash'] == em['encoder']['param_hash']
    assert param_hash(prb) == bm['probe_param_hash']
    assert param_hash(dec) == bm['decoder_param_hash']
    e = em['encoder']
    hp = {
        'encoder': dict(cls='GenEncoder', in_shape=[3, 64, 64], channels=e['channels'], kernel=e['kernel'],
                        stride=e['stride'], groups=e['groups']),
        'probe': dict(cls='FactorProbe', in_shape=[3, 64, 64],
                      channels=[int(c.out_channels) for c in prb.convs],
                      groups=int(prb.norms[0].groups) if prb.norms[0] is not None else 0, cardinalities=CARDS),
        'decoder': dict(cls='GenDecoder', feat_shape=list(dec.feat_shape), out_shape=list(dec.out_shape),
                        channels=[int(c.out_channels) for c in dec.convs[:-1]],
                        groups=int(next(n for n in dec.norms if n is not None).groups), blocks_per_stage=1),
    }
    man = {}
    for name, m in (('encoder', enc), ('probe', prb), ('decoder', dec)):
        h = dict(hp[name]); cls = {'GenEncoder': GenEncoder, 'FactorProbe': FactorProbe, 'GenDecoder': GenDecoder}[h.pop('cls')]
        f = o / f'{name}.eqx'
        eqx.tree_serialise_leaves(f, m)
        kw = {k: (tuple(v) if k.endswith('shape') else v) for k, v in h.items()}
        back = eqx.tree_deserialise_leaves(f, cls(**kw, key=jax.random.PRNGKey(0)))
        assert param_hash(back) == param_hash(m), (s, name)
        man[name] = dict(hp[name], file=f.name, param_hash=param_hash(m),
                         sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)
        print(s, name, 'round-trip OK', param_hash(m)[:16], f.stat().st_size)
    (o / 'models.json').write_text(json.dumps(man, indent=1) + '\n')
    # manifests without the origin machine's absolute path
    bm.pop('encoder_dir', None)
    (o / 'bridge_manifest.json').write_text(json.dumps(bm, indent=1) + '\n')
    for n in ('encoder_manifest.json', 'feature_cache.json', 'shared_decoder_hash.json'):
        (o / n).write_text((d / n).read_text())
