"""CLU as a registered ``cafe-bench`` model (the CAFE integration bridge).

CAFE (Classification / Anomaly / Event-prediction) is an external benchmark
harness whose entire model-side contract is one method::

    class MyModel(BaseModel):
        name = "my_model"
        def encode(self, X: np.ndarray) -> np.ndarray:   # (N, T, C) -> (N, D)
            ...
    register_model("my_model", MyModel)

The harness owns splits, probes and metrics; the model only supplies frozen
embeddings. This module implements that contract for CLU by reusing the
``clu_scorer`` training path (``_SharedCLUFit``: the simplified Hamiltonian
contrastive-divergence objective — wake prediction MSE + denoising-EBM
contrastive energy) and exposing the trained unit's PHYSICS as the embedding.

⚠ CAFE is NOT vendored into this repo. Clone it separately (it is a private
sibling project) and put it on ``PYTHONPATH``; this module imports
``cafe_bench`` lazily so that ``chlu`` never hard-depends on it.

──────────────────────────────────────────────────────────────────────────────
THE FAIRNESS POINT (corrects the scouting assumption — read before tuning)
──────────────────────────────────────────────────────────────────────────────
The HEPA *paper* finetunes its own monotone survival-CDF head. **CAFE's HEPA
wrapper does not.** ``cafe_bench/models/hepa_model.py`` states verbatim: *"The
encoder is frozen; all downstream tasks use the default linear probes defined
in BaseModel."* So the leaderboard's HEPA Event-Prediction number is an
``encode()``-only number scored through the default **CoxPH** probe.

Therefore CLU also ships ``encode()``-only for Event Prediction and does NOT
override ``event_predict``. Overriding it would beat HEPA's number with a
different *head*, not a different *encoder*, and the comparison would be void.
An override belongs in a separately-registered, separately-labelled model.

``anomaly_score`` IS overridden (``clu_valley``), because the default kNN probe
hands the task to CLU's measured weak axis (local outlier detection) and the
valley-aware energy read is the CLU-specific claim. That model is registered
under its own key so the two are never conflated.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.eval.clu_scorer import _SharedCLUFit, rollout_on_data_grid
from chlu.eval.config import CLUCafeEncodeConfig, CLUScorerConfig


def _potential(model):
    """``q -> V(q)`` as a scalar, for either a CHLU or a CLULattice."""
    net = model.potential_net
    return lambda q: jnp.sum(net(q))


def _trend(y: jnp.ndarray) -> jnp.ndarray:
    """Least-squares slope of ``y`` against its own (centered) index.

    The basin-exit signal: a window whose energy / ||grad V|| is *rising* is
    climbing out of its basin, independently of the absolute level (which is
    engine-specific). Scale-free in the index, so it is comparable across
    windows of equal length.
    """
    n = y.shape[0]
    t = jnp.arange(n, dtype=y.dtype)
    tc = t - jnp.mean(t)
    return jnp.sum(tc * (y - jnp.mean(y))) / (jnp.sum(tc * tc) + 1e-12)


def _window_features(model, w, dt, gamma, relax_steps, hz, groups,
                     data_dt=None, substeps=1):
    """Physics feature vector for ONE window ``w`` of shape (L, C).

    Momentum is the finite difference ``(q_{t+1} - q_t) / data_dt`` — the same
    estimator ``clu_scorer`` uses, so energies are directly comparable between
    the CAFE bridge and the anomaly harness.

    Args:
        dt: INTEGRATOR step (``cfg.dt_eff``) for the relax/predict rollouts.
        data_dt: PHYSICAL sampling interval for the finite-difference momentum
            and the predict rollout's horizon. Defaults to ``dt`` — i.e. the
            pre-w20 conflated behaviour — so existing callers are unchanged.
        substeps: integrator steps per data frame (``cfg.substeps``).
    """
    L, C = w.shape
    if data_dt is None:
        data_dt = dt
    V = _potential(model)
    gradV = jax.grad(V)

    q = w[:-1]
    p = (w[1:] - w[:-1]) / data_dt

    H_t = jax.vmap(model.H)(q, p)
    V_t = jax.vmap(V)(q)
    K_t = H_t - V_t
    g_t = jax.vmap(gradV)(q)
    gn_t = jnp.sum(g_t * g_t, axis=-1)

    feats = []
    if "energy" in groups:
        feats += [jnp.mean(H_t), H_t[-1], jnp.std(H_t), _trend(H_t)]
    if "potential" in groups:
        feats += [jnp.mean(V_t), V_t[-1], _trend(V_t)]
    if "kinetic" in groups:
        feats += [jnp.mean(K_t), K_t[-1]]
    if "gradv" in groups:
        feats += [jnp.mean(gn_t), gn_t[-1], _trend(gn_t)]

    # Damped relaxation from the window's FINAL state -> the settled point q*.
    # "Where does the system I am looking at RIGHT NOW come to rest, and how
    # far is that?" — the basin-membership read.
    need_relax = ("relax" in groups) or ("basin_coords" in groups)
    if need_relax:
        traj = model(q[-1], p[-1], relax_steps, dt, gamma)
        q_star = traj[-1, :C]
        g_star = gradV(q_star)
        if "relax" in groups:
            feats += [
                jnp.sum(g_star * g_star),          # relaxation residual
                V(q_star),                          # settled height
                jnp.sum((q_star - q[-1]) ** 2),     # drift to the basin
            ]
    if "predict" in groups:
        pred = rollout_on_data_grid(
            model, w[0], (w[1] - w[0]) / data_dt, hz, dt, substeps, 0.0
        )[:, :C]
        feats += [jnp.mean((pred - w[1 : hz + 1]) ** 2)]

    # ``basin_coords`` alone leaves no scalar features — stack() rejects [].
    if feats:
        out = jnp.stack([jnp.asarray(f, dtype=jnp.float32).reshape(()) for f in feats])
    else:
        out = jnp.zeros((0,), dtype=jnp.float32)
    if "basin_coords" in groups:
        out = jnp.concatenate([out, q_star.astype(jnp.float32)])
    return out


class CLUCafeMixin:
    """CLU's ``encode()`` implementation, independent of ``cafe_bench``.

    This is a MIXIN, not the registered class. :func:`register` composes it
    with ``cafe_bench.models.base.BaseModel`` at import time so that CLU
    inherits the harness's *default* probes verbatim (LogisticRegression /
    kNN / CoxPH). Keeping the dependency inversion this way means ``chlu``
    imports cleanly on a machine with no CAFE checkout, while the registered
    model is still a genuine ``BaseModel`` subclass.

    Training is lazy and happens on the FIRST ``encode`` call. That is safe
    because every default probe in ``BaseModel`` (``classify``,
    ``anomaly_score``, ``event_predict``) encodes the TRAIN split before the
    test split — verified against the harness source. The fitted model is then
    reused, so train and test are embedded by the same frozen CLU.
    """

    name = "clu"

    def __init__(
        self,
        clu_config: CLUScorerConfig | None = None,
        encode_config: CLUCafeEncodeConfig | None = None,
        device: str | None = None,      # accepted + ignored (CAFE CLI passes it)
        checkpoint_path: str | None = None,   # accepted + ignored
        **_kwargs,
    ):
        self.cfg = clu_config or CLUScorerConfig()
        self.enc = encode_config or CLUCafeEncodeConfig()
        self._shared: _SharedCLUFit | None = None
        self._scaler: tuple[np.ndarray, np.ndarray] | None = None
        self._fitted = False
        #: Fraction of rows zero-filled by the last :meth:`encode` call. A
        #: spread of 0 with this above 0 is an overflow, not a basin collapse.
        self.last_nonfinite_fraction = 0.0

    # ── internals ────────────────────────────────────────────────────────
    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"CAFE encode expects (N, T, C); got {X.shape}")
        if self.enc.standardize:
            if self._scaler is None:
                mean = X.mean(axis=(0, 1), keepdims=True)
                std = X.std(axis=(0, 1), keepdims=True) + 1e-8
                self._scaler = (mean, std)
            mean, std = self._scaler
            X = (X - mean) / std
        return X

    def _ensure_fit(self, X: np.ndarray) -> None:
        if self._fitted:
            return
        N, T, C = X.shape
        self._shared = _SharedCLUFit(self.cfg, window_size=T)
        # _SharedCLUFit speaks the anomaly harness's FLATTENED window contract.
        self._shared.ensure_fit(X.reshape(N, T * C))
        self._fitted = True

    def _features(self, X: np.ndarray) -> np.ndarray:
        cfg, enc = self.cfg, self.enc
        model = self._shared.model
        N, T, C = X.shape
        hz = int(min(cfg.predict_horizon, T - 1))
        relax = int(enc.relax_steps if enc.relax_steps is not None else cfg.relax_steps)
        gamma = float(enc.relax_gamma if enc.relax_gamma is not None else cfg.gamma)
        groups = tuple(enc.feature_groups)

        @eqx.filter_jit
        def batch_feats(m, Wb):
            return jax.vmap(
                lambda w: _window_features(
                    m, w, cfg.dt_eff, gamma, relax, hz, groups,
                    data_dt=cfg.data_dt, substeps=cfg.substeps,
                )
            )(Wb)

        out = []
        for i in range(0, N, enc.batch_size):
            Wb = jnp.asarray(X[i : i + enc.batch_size])
            out.append(np.asarray(batch_feats(model, Wb)))
        return np.concatenate(out, axis=0).astype(np.float32)

    # ── CAFE contract ────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray) -> "CLUCafeMixin":
        """Explicitly train the CLU on ``X_train``.

        The harness never calls this — ``encode`` auto-fits on its first call,
        which is the TRAIN split for every default probe. Call it explicitly in
        analysis scripts that encode the test split first: otherwise the lazy
        fit silently trains the CLU (and the standardizer) on TEST data. That
        footgun cost a contaminated diagnostic during this integration; the
        harness path itself is unaffected.
        """
        self._ensure_fit(self._prepare(X_train))
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Map (N, T, C) windows to (N, D) CLU physics embeddings.

        ⚠ The non-finite guard below is NOT cosmetic, and it must be LOUD.
        Zero-filling an overflowed rollout turns a numerical blow-up into a
        perfectly well-formed constant embedding: the cross-sample std becomes
        EXACTLY 0.000 and the downstream probe goes singular — which reads
        exactly like a physical "every window settles onto one point" basin
        collapse. That misreading was published once (a `relax_gamma` above ~2
        makes ``p <- (1 - gamma) * p`` amplify by ``|1 - gamma|`` per step, so
        the rollout overflows within ~50 steps). Always check
        ``last_nonfinite_fraction`` before interpreting a spread of 0.
        """
        Xs = self._prepare(X)
        self._ensure_fit(Xs)
        Z = self._features(Xs)
        bad = ~np.isfinite(Z)
        self.last_nonfinite_fraction = float(bad.any(axis=1).mean())
        if self.last_nonfinite_fraction > 0:
            warnings.warn(
                f"{self.last_nonfinite_fraction:.1%} of CLU embeddings contain "
                "non-finite values and are being zero-filled; the resulting "
                "cross-sample spread is a NUMERICAL artifact, not a basin "
                f"collapse. Check relax_gamma (={self.enc.relax_gamma}): the "
                "dissipative step diverges for gamma > 2.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # ── provenance ───────────────────────────────────────────────────────
    def feature_names(self) -> list[str]:
        """Names of the embedding coordinates, in emission order.

        Must mirror ``_window_features`` exactly — the anomaly arms and any
        coefficient reporting index by name.
        """
        g = self.enc.feature_groups
        names: list[str] = []
        if "energy" in g:
            names += ["energy_mean", "energy_last", "energy_std", "energy_trend"]
        if "potential" in g:
            names += ["V_mean", "V_last", "V_trend"]
        if "kinetic" in g:
            names += ["K_mean", "K_last"]
        if "gradv" in g:
            names += ["gradV_mean", "gradV_last", "gradV_trend"]
        if "relax" in g:
            names += ["relax_residual", "relax_V_star", "relax_drift"]
        if "predict" in g:
            names += ["predict_mse"]
        if "basin_coords" in g:
            n_c = self._shared.n_channels if self._shared is not None else 0
            names += [f"q_star_{i}" for i in range(n_c)]
        return names


class CLUValleyMixin(CLUCafeMixin):
    """CLU with the valley-aware anomaly arm replacing CAFE's kNN probe.

    Registered under its own key (``clu_valley``) so that an anomaly number
    produced with the overridden probe is never confused with a default-probe
    number.
    """

    name = "clu_valley"

    def anomaly_score(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """VALLEY-AWARE anomaly score (overrides CAFE's default kNN probe).

        Scores the SETTLED state rather than the transient: a normal window
        relaxes into the learned basin (small residual, low V(q*)); an
        anomalous one cannot settle. Standardized on the train split so the
        summed arms are commensurate, then combined UNTUNED (equal weight) —
        per CM-3, the honest head-to-head is the result, not a tuned score.
        """
        mode = self.enc.anomaly_mode
        Ztr = self.encode(X_train)
        Zte = self.encode(X_test)
        names = self.feature_names()

        def col(tag: str) -> int:
            if tag not in names:
                raise ValueError(
                    f"anomaly_mode={mode!r} needs feature {tag!r}, which is not "
                    f"in feature_groups={self.enc.feature_groups}"
                )
            return names.index(tag)

        def z(idx: int) -> np.ndarray:
            mu, sd = Ztr[:, idx].mean(), Ztr[:, idx].std() + 1e-8
            return (Zte[:, idx] - mu) / sd

        if mode == "energy":
            return z(col("energy_mean"))
        if mode == "predict":
            return z(col("predict_mse"))
        s = z(col("relax_residual")) + z(col("relax_V_star"))
        if mode == "valley_predict":
            s = s + z(col("predict_mse"))
        return s


def build_model_classes():
    """Compose the mixins with CAFE's ``BaseModel`` into registrable classes.

    Done at call time (not import time) so that importing ``chlu.eval``
    never requires a CAFE checkout on ``PYTHONPATH``.
    """
    from cafe_bench.models.base import BaseModel

    clu = type("CLUCafeModel", (CLUCafeMixin, BaseModel), {})
    clu_valley = type("CLUValleyCafeModel", (CLUValleyMixin, BaseModel), {})
    return clu, clu_valley


def register() -> dict:
    """Register CLU with the CAFE registry (the harness's one-line convention).

    Returns the ``{key: cls}`` mapping that was registered.
    """
    from cafe_bench.registry import register_model

    clu, clu_valley = build_model_classes()
    register_model("clu", clu)
    register_model("clu_valley", clu_valley)
    return {"clu": clu, "clu_valley": clu_valley}
