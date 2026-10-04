"""shapes fast judge."""

from __future__ import annotations
import numpy as np


class FastJudge:
    """A drop-in for :class:`exp_gen_factor_wells.Judge` at ``read_back``."""

    def __init__(self, judge):
        import equinox as eqx
        import jax
        from nclu.eval.gen_probe import model_dtype

        self.judge = judge
        self.info = judge.info
        self.decoder = judge.decoder
        self.external = judge.external
        self._dec_dtype = model_dtype(judge.decoder)
        self._clf_dtype = model_dtype(judge.external)
        self._dec = eqx.filter_jit(jax.vmap(judge.decoder))
        self._clf = eqx.filter_jit(jax.vmap(judge.external))
        self._n_heads = len(judge.external.heads.linears)

    def read_back(self, feats: np.ndarray, batch: int = 256) -> np.ndarray:
        import jax.numpy as jnp
        from nclu.experiments.factor_wells import PINS

        Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(PINS["feat_shape"]))
        out = np.empty((Z.shape[0], 6), np.int64)
        for s in range(0, Z.shape[0], batch):
            e = min(s + batch, Z.shape[0])
            imgs = np.asarray(self._dec(jnp.asarray(Z[s:e], self._dec_dtype)), np.float32)
            logits = self._clf(jnp.asarray(imgs, self._clf_dtype))
            preds = np.stack([np.asarray(jnp.argmax(lg, axis=-1)) for lg in logits], axis=1)
            out[s:e] = np.asarray(preds.astype(np.int32), np.int64)
        return out


def assert_identical(judge, fast: FastJudge, feats: np.ndarray, batch: int = 256) -> dict:
    a = judge.read_back(feats, batch)
    b = fast.read_back(feats, batch)
    if a.shape != b.shape or not np.array_equal(a, b):
        n = int((a != b).sum()) if a.shape == b.shape else -1
        raise AssertionError(
            "FastJudge disagrees with the shipped Judge on %d of %d elements -- the hoisted jit is NOT a no-op and no number may be taken from it"
            % (n, a.size)
        )
    return {
        "check": "FastJudge == exp_gen_factor_wells.Judge.read_back",
        "n_requests": int(np.asarray(feats).shape[0]),
        "n_elements_compared": int(a.size),
        "identical": True,
        "what": "the shipped path rebuilds eqx.filter_jit(jax.vmap(model)) on every call, so it recompiles per call; this path builds it once. Same weights, same dtype, same batching, same argmax.",
    }
