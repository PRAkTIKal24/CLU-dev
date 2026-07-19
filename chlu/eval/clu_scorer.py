"""CLU anomaly scorer — the CLU -> eval-harness bridge (G7b flagship).

The evaluation harness scores anomalies through
``BaselineScorer.fit(train_windows) / score(windows)`` (see
``chlu.eval.baselines``). This module is the first CLU that implements that
contract, so the torus-CLU voraus-AD flagship can be scored at all.

THE ANOMALY SCORE IS ITSELF THE EXPERIMENT (Head, 2026-07-19). There is no
single "right" score; finding it is a G7b sub-result. The scorer therefore
exposes several score arms that all read ONE trained CLU, so the head-to-head
is a single experiment (``make_clu_scorers`` registers one arm per mode):

  * ``energy``   — mean H(q, p) over a window's states (finite-difference
                   momentum). The EBM reading: anomalies are high-energy
                   states off the learned data basin.
  * ``residual`` — relaxation residual R0: run a short damped rollout from a
                   few window anchor states and measure the residual force
                   ``||grad V||^2`` after settling — "how badly the window
                   fails to settle into a basin".
  * ``predict``  — multi-step CLU-rollout prediction MSE over the window (the
                   conventional TSAD framing; fairest head-to-head vs PCA-recon).
  * ``hybrid``   — a documented, UNTUNED z-score combination of energy+predict
                   (hook only; the final iteration, tried only if it beats both
                   singles — do NOT tune it in the first pass).

CM-3 discipline (binding): NEVER claim energy is a *superior* signal to a
reconstruction baseline. It is a *physically-grounded* one carrying guarantees
the baselines lack; the honest head-to-head is the result.

The single CLU is trained by a simplified Hamiltonian Contrastive Divergence
(§1.4): a wake prediction-MSE term (makes ``predict`` meaningful) plus a
denoising-EBM contrastive energy term that pushes H(data) down and H(noised
data) up (makes ``energy``/``residual`` meaningful). Both arms then read the
same model, so the comparison is apples-to-apples.

Scaler contract: the harness fits a ``StandardScaler`` on the training windows
and scales every window before calling ``fit``/``score`` (see
``harness._fit_scaler``), so this scorer sees already-scaled flattened windows
and does no scaling of its own.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.core.chlu_unit import CHLU
from chlu.core.lattice import build_lattice, torus_edges
from chlu.eval.baselines import BaselineScorer, make_default_baselines
from chlu.eval.config import (
    CLU_DEFAULT_SCORE_MODES,
    CLU_SCORE_MODES,
    CLUScorerConfig,
    EvalConfig,
)


def _infer_channels(width: int, window_size: int) -> int:
    """Recover the channel count C from a flattened window width (size * C)."""
    if width % window_size != 0:
        raise ValueError(
            f"flattened window width {width} is not divisible by window size "
            f"{window_size}; cannot recover the channel count"
        )
    return width // window_size


def _build_model(cfg: CLUScorerConfig, n_channels: int, key):
    """Construct the CLU (single ``CHLU`` or ``CLULattice``) for C channels.

    Lattice hook (G7b prerequisite): when ``cfg.lattice`` is set, tile the C
    channels into units of ``unit_dim`` channels each and forward the
    coupling/topology knobs to ``build_lattice``. The literal joint-angle ->
    so2_invariant-coset mapping is the next task (``g7b-torus-voraus``); this
    only guarantees the config is expressible and yields a model of total
    dim == C.
    """
    if cfg.lattice is None:
        return CHLU(
            dim=n_channels,
            hidden=cfg.hidden,
            rest_mass=cfg.rest_mass,
            c=cfg.c,
            kinetic_mode=cfg.kinetic_mode,
            potential_type=cfg.potential_type,
            tie_channel_mass=cfg.tie_channel_mass,
            key=key,
        )

    lc = cfg.lattice
    if n_channels % lc.unit_dim != 0:
        raise ValueError(
            f"lattice hook: C={n_channels} channels not divisible by "
            f"unit_dim={lc.unit_dim}. The literal channel->coset mapping "
            "(g7b-torus-voraus) must handle non-divisible layouts; this hook "
            "requires an exact tiling."
        )
    n_units = n_channels // lc.unit_dim
    if lc.topology == "torus":
        side = int(round(n_units**0.5))
        if side * side != n_units:
            raise ValueError(
                f"torus topology needs n_units (={n_units}) to be a perfect "
                f"square; got side≈{n_units**0.5:.3f}"
            )
        edges = torus_edges(side)
    else:
        edges = None  # build_lattice defaults to a chain
    return build_lattice(
        key=key,
        unit_dims=[lc.unit_dim] * n_units,
        hidden=cfg.hidden,
        potential_type=lc.potential_type,
        kinetic_mode=cfg.kinetic_mode,
        edges=edges,
        coupling_type=lc.coupling_type,
        kappa_c=lc.kappa_c,
        coupling_dim=lc.coupling_dim,
        rest_mass=cfg.rest_mass,
        c=cfg.c,
        tie_channel_mass=lc.tie_channel_mass,
    )


class _SharedCLUFit:
    """One trained CLU shared by all arms (so the comparison is one experiment).

    The harness fits every scorer on the same training array before scoring
    (``harness._run_cross_unit`` fits all, then scores; ``_run_per_unit_prefix``
    fits+scores each method within a fresh ``factory()`` per unit). This object
    trains lazily on the first ``ensure_fit`` and is reused by later arms.
    """

    def __init__(self, cfg: CLUScorerConfig, window_size: int):
        self.cfg = cfg
        self.window_size = int(window_size)
        self.model = None
        self.n_channels = None
        self._hybrid_stats = None  # (e_mean, e_std, p_mean, p_std)

    # -- reshaping ----------------------------------------------------------
    def _reshape(self, windows: np.ndarray) -> jnp.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        n, width = windows.shape
        c = _infer_channels(width, self.window_size)
        if self.n_channels is None:
            self.n_channels = c
        elif c != self.n_channels:
            raise ValueError(
                f"channel count changed between fit ({self.n_channels}) and "
                f"score ({c}) — datasets must be consistent"
            )
        return jnp.asarray(windows.reshape(n, self.window_size, c))

    # -- momentum estimate --------------------------------------------------
    def _momentum(self, q_now: jnp.ndarray, q_next: jnp.ndarray) -> jnp.ndarray:
        if self.cfg.momentum_init == "zero":
            return jnp.zeros_like(q_now)
        return (q_next - q_now) / self.cfg.dt

    # -- training -----------------------------------------------------------
    def ensure_fit(self, train_windows: np.ndarray) -> None:
        if self.model is not None:
            return
        cfg = self.cfg
        W = self._reshape(train_windows)  # (n, L, C)
        # seeded uniform subsample of training windows (memory/time guard)
        rng = np.random.default_rng(cfg.seed)
        if len(W) > cfg.max_fit_windows:
            idx = np.sort(
                rng.choice(len(W), size=cfg.max_fit_windows, replace=False)
            )
            W = W[idx]
        n, L, C = W.shape
        key = jax.random.PRNGKey(cfg.seed)
        key, mkey = jax.random.split(key)
        model = _build_model(cfg, C, mkey)

        optim = optax.adam(cfg.lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        dt = cfg.dt
        hz = int(min(cfg.predict_horizon, L - 1))
        pw, ew, ereg = cfg.predict_weight, cfg.energy_weight, cfg.energy_reg
        neg = cfg.neg_noise_scale

        def loss_fn(model, batch, nkey):
            # batch: (B, L, C)
            q0 = batch[:, 0, :]
            p0 = self._momentum(q0, batch[:, 1, :])
            # --- wake prediction MSE (dynamics) ---
            def roll(qi, pi):
                return model(qi, pi, hz, dt, 0.0)[:, :C]  # (hz, C) positions

            pred = jax.vmap(roll)(q0, p0)  # (B, hz, C)
            target = batch[:, 1 : hz + 1, :]
            predict_mse = jnp.mean((pred - target) ** 2)
            # --- contrastive energy (denoising EBM) ---
            qs = batch[:, :-1, :].reshape(-1, C)
            ps = ((batch[:, 1:, :] - batch[:, :-1, :]) / dt).reshape(-1, C)
            h_data = jax.vmap(model.H)(qs, ps)
            noise = neg * jax.random.normal(nkey, qs.shape)
            h_neg = jax.vmap(model.H)(qs + noise, ps)
            energy_contrastive = jnp.mean(h_data) - jnp.mean(h_neg)
            reg = ereg * (jnp.mean(h_data**2) + jnp.mean(h_neg**2))
            return pw * predict_mse + ew * energy_contrastive + reg

        @eqx.filter_jit
        def step(model, opt_state, batch, nkey):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, nkey)
            updates, opt_state = optim.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        bs = int(min(cfg.batch_size, n))
        steps_per_epoch = max(1, n // bs)
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            for s in range(steps_per_epoch):
                bidx = perm[s * bs : (s + 1) * bs]
                if len(bidx) == 0:
                    continue
                key, nkey = jax.random.split(key)
                model, opt_state, _ = step(model, opt_state, W[bidx], nkey)

        self.model = model
        # hybrid z-score stats on a capped train subset (untuned combination).
        stat_W = W[: min(512, n)]
        e = self._energy_scores(stat_W)
        p = self._predict_scores(stat_W)
        self._hybrid_stats = (
            float(np.mean(e)),
            float(np.std(e) + 1e-8),
            float(np.mean(p)),
            float(np.std(p) + 1e-8),
        )

    # -- score arms (operate on reshaped (n, L, C) arrays) ------------------
    def _energy_scores(self, W: jnp.ndarray) -> np.ndarray:
        model, dt = self.model, self.cfg.dt
        qs = W[:, :-1, :]
        ps = (W[:, 1:, :] - W[:, :-1, :]) / dt

        def per_window(qw, pw):
            return jnp.mean(jax.vmap(model.H)(qw, pw))

        return np.asarray(jax.vmap(per_window)(qs, ps))

    def _predict_scores(self, W: jnp.ndarray) -> np.ndarray:
        model, cfg = self.model, self.cfg
        n, L, C = W.shape
        hz = int(min(cfg.predict_horizon, L - 1))
        q0 = W[:, 0, :]
        p0 = self._momentum(q0, W[:, 1, :])

        def roll(qi, pi):
            return model(qi, pi, hz, cfg.dt, 0.0)[:, :C]

        pred = jax.vmap(roll)(q0, p0)
        target = W[:, 1 : hz + 1, :]
        return np.asarray(jnp.mean((pred - target) ** 2, axis=(1, 2)))

    def _residual_scores(self, W: jnp.ndarray) -> np.ndarray:
        model, cfg = self.model, self.cfg
        n, L, C = W.shape
        n_anchors = int(min(cfg.residual_anchors, L - 1))
        anchor_idx = jnp.asarray(
            np.linspace(0, L - 2, n_anchors).round().astype(int)
        )
        gamma, dt, relax = cfg.gamma, cfg.dt, cfg.relax_steps
        V = model.potential_net

        def resid_one(q0i, p0i):
            traj = model(q0i, p0i, relax, dt, gamma)  # (relax, 2C)
            q_rel = traj[-1, :C]
            g = jax.grad(lambda q: jnp.sum(V(q)))(q_rel)
            return jnp.sum(g * g)

        def per_window(w):
            q_a = w[anchor_idx]  # (A, C)
            q_next = w[anchor_idx + 1]
            p_a = self._momentum(q_a, q_next)
            return jnp.mean(jax.vmap(resid_one)(q_a, p_a))

        return np.asarray(jax.vmap(per_window)(W))

    def score(self, windows: np.ndarray, mode: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CLU scorer used before fit()")
        W = self._reshape(windows)
        if mode == "energy":
            return self._energy_scores(W)
        if mode == "residual":
            return self._residual_scores(W)
        if mode == "predict":
            return self._predict_scores(W)
        if mode == "hybrid":
            e = self._energy_scores(W)
            p = self._predict_scores(W)
            em, es, pm, ps = self._hybrid_stats
            # UNTUNED equal-weight z-score combination (energy-gated prediction
            # error is the intended shape; kept documented + untuned in pass 1).
            return (e - em) / es + (p - pm) / ps
        raise ValueError(f"unknown score_mode {mode!r}; must be one of {CLU_SCORE_MODES}")


class CHLUScorer(BaselineScorer):
    """A single CLU anomaly-score arm, implementing the harness ABC.

    Several arms share one trained CLU via a ``_SharedCLUFit`` so the whole
    energy-vs-predict comparison is a single experiment. Construct via
    ``make_clu_scorers`` rather than directly for the shared-model wiring.
    """

    def __init__(self, mode: str, shared: _SharedCLUFit, name: str | None = None):
        if mode not in CLU_SCORE_MODES:
            raise ValueError(f"score_mode must be one of {CLU_SCORE_MODES}, got {mode}")
        self.mode = mode
        self._shared = shared
        self.name = name or f"clu_{mode}"

    def fit(self, train_windows: np.ndarray) -> "CHLUScorer":
        self._shared.ensure_fit(train_windows)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        return self._shared.score(windows, self.mode)


def make_clu_scorers(
    config: EvalConfig,
    clu_config: CLUScorerConfig | None = None,
    modes: tuple = CLU_DEFAULT_SCORE_MODES,
) -> dict:
    """Factory: the four mandatory statistical baselines + the CLU arms.

    The harness rejects a factory missing the mandatory baselines
    (``harness.py`` — ``scorer_factory must include ...``); this keeps them and
    *adds* one CLU arm per requested ``mode``. All CLU arms share one trained
    model (``_SharedCLUFit``), so training happens once per ``factory()`` call.

    Args:
        config: the harness ``EvalConfig`` (supplies ``window.size`` and the
            baseline knobs).
        clu_config: CLU scorer configuration (defaults to ``CLUScorerConfig()``).
        modes: which CLU arms to register (default = the two mandatory first-pass
            arms + the residual variant: energy, residual, predict).

    Returns:
        ``dict[name, BaselineScorer]`` — pass a ``lambda: make_clu_scorers(...)``
        as ``evaluate_dataset(scorer_factory=...)``.
    """
    clu_config = clu_config or CLUScorerConfig()
    for m in modes:
        if m not in CLU_SCORE_MODES:
            raise ValueError(f"unknown CLU score mode {m!r}; valid: {CLU_SCORE_MODES}")
    shared = _SharedCLUFit(clu_config, window_size=config.window.size)
    scorers = make_default_baselines(config)
    for m in modes:
        scorers[f"clu_{m}"] = CHLUScorer(m, shared, name=f"clu_{m}")
    return scorers
