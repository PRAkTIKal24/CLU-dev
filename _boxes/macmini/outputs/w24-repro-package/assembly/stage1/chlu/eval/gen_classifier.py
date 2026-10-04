"""⭐ The **external** factor classifier — the instrument that reads IMAGES.

⭐ **Why this exists (R29 pivot item 2).**  Every composition number in phase 1 is
currently read by :class:`~chlu.core.gen_bridge.FactorProbe`, and that probe is
fitted **inside the bridge run, on the same 8,192 train rows the shared decoder is
fitted on, from the same seed-derived key chain**
(``exp_gen_bridge.run_decoder_stage``: ``key, dk, pk = jax.random.split(key, 3)``).
R29's headline table is ``request → read → codes → decoder → image → EXTERNAL
classifier``.  ⛔ **A probe that shares the decoder's lineage cannot be the
instrument that scores the decoder's output.**  This module is that instrument.

**What "external" means here, operationally — four separations, all asserted:**

1. **Rows.**  :func:`classifier_rows` draws a subsample of the train split that is
   **disjoint** from the bridge's fit rows, from ``eval_a``, and from every row in
   the photometric corner the phase evaluates in.  ⛔ Every row it is scored on is
   a row it has never seen in any form.
2. **Renders.**  ⛔ It never sees a decoder render at fit time — and that is
   *verified*, not asserted: :func:`assert_no_render_leak` re-streams a sample of
   the training tensor straight out of the HDF5 file and requires bit-identity.
3. **Architecture.**  Kernel 5 / SiLU / four stages / a hidden projection, against
   the probe's kernel 4 / ReLU / three stages / direct linear heads.
4. **Key chain.**  ``PRNGKey(seed * 104_729 + 20_260_913)``, ⛔ never the bridge's
   ``seed * 10_007 + 20_260_910``.

⛔ **Architecture alone is not independence.**  Independence is only testable where
the two instruments can disagree — on the decoder's own renders — which is what the
experiment module's leg 3 measures and ⛔ what this module deliberately does not
adjudicate.

**The gate** (:func:`gate_verdict`) is pre-registered and stop-and-return: in-corner
``shape`` balanced accuracy on **REAL** held-out-combination images must reach the
floor on **every** seed, because R15 established that the shipped probe reads real
images at **1.0000** and is saturated — an instrument that cannot read real images
in the corner is not an instrument.
"""
from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
from ..core.gen_bridge import FactorHeads
class ExternalFactorCNN(eqx.Module):
    """A per-factor classifier over **raw images**, independent of the decoder.

    ``[conv(k, stride) -> GroupNorm -> SiLU] x len(channels)`` -> flatten ->
    ``Linear(hidden) -> SiLU`` -> one linear head per factor.  Calling it returns a
    **list of per-factor logits**, exactly like
    :class:`~chlu.core.gen_bridge.FactorProbe`, so the two instruments are scored by
    *the same* ``chlu.eval.gen_probe`` functions and a gap between them can never be
    a difference in the scoring code.
    """

    convs: list
    norms: list
    proj: eqx.nn.Linear
    heads: FactorHeads

    def __init__(self, in_shape, channels, kernel, stride, groups, hidden, cardinalities, key):
        keys = jax.random.split(key, len(channels) + 2)
        convs, norms = [], []
        c_in, h, w = (int(x) for x in in_shape)
        pad = max((int(kernel) - int(stride)) // 2, 0)
        for c_out, k in zip(channels, keys[: len(channels)], strict=True):
            convs.append(
                eqx.nn.Conv2d(c_in, int(c_out), int(kernel), stride=int(stride),
                              padding=pad, key=k)
            )
            norms.append(
                eqx.nn.GroupNorm(min(int(groups), int(c_out)), int(c_out))
                if int(groups) > 0
                else None
            )
            c_in = int(c_out)
            h = (h + 2 * pad - int(kernel)) // int(stride) + 1
            w = (w + 2 * pad - int(kernel)) // int(stride) + 1
        self.convs = convs
        self.norms = norms
        self.proj = eqx.nn.Linear(int(c_in * h * w), int(hidden), key=keys[-2])
        self.heads = FactorHeads(int(hidden), cardinalities, keys[-1])

    def __call__(self, x):
        # ⚠ §7.23: a module-scoped ``jax_enable_x64`` fixture elsewhere in the suite
        # makes eqx build float64 weights; cast rather than assume float32.
        x = jnp.asarray(x, self.convs[0].weight.dtype)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.silu(x)
        return self.heads(jax.nn.silu(self.proj(x.reshape(-1))))


