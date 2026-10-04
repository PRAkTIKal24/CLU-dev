"""Confirm the suite-only failures are caused by jax_enable_x64 leaking from
another test module (several enable it AT MODULE IMPORT, process-globally)."""
import os, sys
sys.path.insert(0, os.environ["CHLU_ROOT"])
import jax
for x64 in (False, True):
    jax.config.update("jax_enable_x64", x64)
    import importlib, chlu.core.gen_bridge as gb; importlib.reload(gb)
    from chlu.config import get_default_config
    IN=(3,16,16)
    ek,dk,sk = jax.random.split(jax.random.PRNGKey(20260914),3)
    enc = gb.GenEncoder(IN,(8,16),4,2,4,ek)
    dbase = gb.make_decoder(enc.feat_shape, IN, gb.resolve_decoder_cfg(get_default_config().gen_bridge), dk)
    dk2 = jax.random.split(jax.random.PRNGKey(20260914),3)[1]
    dbase2 = gb.make_decoder(enc.feat_shape, IN, gb.resolve_decoder_cfg(get_default_config().gen_bridge), dk2)
    print(f"jax_enable_x64={x64!s:5s}  weight dtype={enc.convs[0].weight.dtype}")
    print(f"   enc_hash           = {gb.param_hash(enc)}")
    print(f"      == FROZEN 7dcf753f...  {gb.param_hash(enc)=='7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693'}")
    print(f"      == SUITE-FAIL 094ae9c3...  {gb.param_hash(enc).startswith('094ae9c3941b8fb6')}")
    print(f"   dec_hash_baseline  = {gb.param_hash(dbase2)}")
    print(f"      == FROZEN 457977e4...  {gb.param_hash(dbase2)=='457977e44d7b2a6ac30742fa5632aad8d31cfab67c71c61e24265fb50b38fea3'}")
    print(f"      == SUITE-FAIL a73a12cd...  {gb.param_hash(dbase2).startswith('a73a12cd18a8')}")
    print()
