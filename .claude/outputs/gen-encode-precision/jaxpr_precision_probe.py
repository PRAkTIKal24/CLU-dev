"""Negative control: what `precision` do the REAL conv equations of the shipped
`encode_images` carry?  Run with PYTHONPATH pointing at either tree.

    main   @ dcd667e  ->  [None, None]                       (XLA default => TF32 on Ada)
    branch @ 70f3bf6  ->  [(HIGHEST, HIGHEST), (HIGHEST, HIGHEST)]
"""
import equinox as eqx, jax, numpy as np
import chlu.core.gen_bridge as gb

enc = gb.GenEncoder((3, 16, 16), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
x = np.zeros((2, 3, 16, 16), np.float32)

captured, real = {}, eqx.filter_jit
def capture(fn, *a, **kw):
    captured["fn"] = fn
    return real(fn, *a, **kw)

eqx.filter_jit = capture
try:
    gb.encode_images(enc, x, 2)
finally:
    eqx.filter_jit = real

jaxpr = jax.make_jaxpr(captured["fn"])(x)
print([e.params.get("precision") for e in jaxpr.jaxpr.eqns
       if "conv_general_dilated" in str(e.primitive)])
