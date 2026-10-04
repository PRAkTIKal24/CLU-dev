"""Synthetic sequence-modelling tasks for the primitive harness (w20).

Two families that complement MQAR (`chlu/data/mqar.py`), chosen because their
*expected winners differ* — a harness whose families all favour the same
primitive measures nothing:

- ``generate_adding``  — the classic long-range integration benchmark
  (Hochreiter & Schmidhuber 1997; the canonical HiPPO/S4 stress test). Two
  positions are marked; the target is the sum of their values. Requires
  lossless propagation over T steps.
- ``generate_parity``  — cumulative XOR. A *state-tracking* task: solvable by a
  finite-state recurrence, provably hard for fixed-depth attention. This is the
  family where recurrence is expected to beat attention.

Both return sequence-shaped targets so they use the identical training loop as
MQAR; ``IGNORE_INDEX``/masking conventions match `mqar.py`.
"""

import jax
import jax.numpy as jnp

IGNORE_INDEX = -100


def generate_adding(key, n_sequences: int, seq_len: int):
    """Adding problem: x = [value, marker]; target = sum of the two marked values.

    Values ~ U(0, 1); exactly two positions carry marker 1 (one drawn from each
    half of the sequence, the standard construction, so the dependency spans
    ~T/2 on average). Target is supervised at the LAST position only.

    Returns dict with:
        inputs:  (n, T, 2) float32
        targets: (n, T, 1) float32 (value at the final position)
        mask:    (n, T) bool — True only at the final position
    """

    def one(k):
        k1, k2, k3 = jax.random.split(k, 3)
        vals = jax.random.uniform(k1, (seq_len,))
        half = seq_len // 2
        i1 = jax.random.randint(k2, (), 0, half)
        i2 = jax.random.randint(k3, (), half, seq_len)
        markers = jnp.zeros(seq_len).at[i1].set(1.0).at[i2].set(1.0)
        total = vals[i1] + vals[i2]
        inputs = jnp.stack([vals, markers], axis=1)
        targets = jnp.zeros((seq_len, 1)).at[seq_len - 1, 0].set(total)
        mask = jnp.zeros(seq_len, dtype=bool).at[seq_len - 1].set(True)
        return inputs, targets, mask

    inputs, targets, mask = jax.vmap(one)(jax.random.split(key, n_sequences))
    return {"inputs": inputs, "targets": targets, "mask": mask}


def generate_parity(key, n_sequences: int, seq_len: int):
    """Cumulative parity: target[t] = XOR of bits[0..t]. Supervised at every step.

    Returns dict with:
        tokens:  (n, T) int32 in {0, 1}
        targets: (n, T) int32 in {0, 1}
        mask:    (n, T) bool — all True
    """
    bits = jax.random.bernoulli(key, 0.5, (n_sequences, seq_len)).astype(jnp.int32)
    targets = jnp.cumsum(bits, axis=1) % 2
    return {
        "tokens": bits,
        "targets": targets.astype(jnp.int32),
        "mask": jnp.ones((n_sequences, seq_len), dtype=bool),
    }
