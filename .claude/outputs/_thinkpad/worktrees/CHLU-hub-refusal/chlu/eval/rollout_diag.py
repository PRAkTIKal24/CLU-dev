"""The BOUNDED-vs-INFORMATIVE rollout diagnostic.

WHY THIS EXISTS (read before using it as a success metric)
──────────────────────────────────────────────────────────
The claim we want to test about an iterated latent rollout is *graceful
degradation at long horizons*. The trap is that **a rollout which collapses to
a fixed point is perfectly "stable", perfectly bounded, and completely
uninformative** — and on C-MAPSS this is not hypothetical: CLU's learned
potential was measured to be effectively single-basin, with the cross-sample
spread of the settled point reaching **exactly 0.000** at damping budget
``gamma*steps*dt = 64``.

So neither of the two obvious metrics can be trusted alone:

* **boundedness** is satisfied *most strongly* exactly when the rollout is dead;
* **error** can also *improve* on collapse, because collapsing onto the dataset
  mean is a decent constant predictor. A dead rollout can look like a win on
  both.

The discriminator is a third curve — the **cross-sample spread** of the
predicted state. A rollout is *informative* only while different inputs still
map to different predictions.

DEFINITION (adopt this verbatim)
────────────────────────────────
For anchors ``i = 1..N``, channels ``c = 1..C`` and rollout step ``n``:

    S(n)      :=  mean_c  std_i  q_i(n)[c]          (cross-sample spread)
    S_rel(n)  :=  S(n) / S(0)                        (normalized to the input)

    COLLAPSE LENGTH   n*  :=  min { n : S_rel(n) < tau },  tau = 0.01 default

``n*`` is the **honest ceiling on any long-horizon claim**: beyond it the model
emits (numerically) the same state regardless of its input, so any accuracy
reported there is a property of the dataset's marginal, not of the model.

Report ``n*`` together with the **damping budget** ``gamma * n* * dt``, which is
the dimensionless quantity that actually controls settling (see
:meth:`CLUCafeEncodeConfig.relax_budget`) and makes the number comparable
across gamma.

FAILURE MODES OF THE DIAGNOSTIC ITSELF (state these when you quote it)
──────────────────────────────────────────────────────────────────────
1. **Spread is not information.** ``S_rel`` can stay high while the rollout is
   pure noise (e.g. a divergent or chaotic system). It is a NECESSARY, not
   sufficient, condition — always read it next to the error curve and the
   persistence baseline. ``n*`` upper-bounds the useful horizon; it does not
   certify usefulness below ``n*``.
2. **Scale, not shape.** ``S`` is a per-channel std averaged over channels, so a
   rollout that preserves total variance while collapsing onto a
   lower-dimensional manifold is NOT detected. Use a spectral variant
   (participation ratio of the state covariance) if rank collapse is the
   concern.
3. **Anchor dependence.** ``S(0)`` is set by the anchor sample; comparing
   ``S_rel`` across datasets assumes comparable input normalization (true for
   CAFE C-MAPSS, which z-scores per channel on train statistics).
4. **tau is a convention.** 0.01 is a round number chosen to sit far below the
   useful regime and far above float noise; report the curve, not only ``n*``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import numpy as np


@dataclass(frozen=True)
class RolloutDiagConfig:
    """Configuration for :func:`rollout_diagnostic`.

    Attributes:
        horizon: number of rollout steps (for CLU on C-MAPSS one step == one
            engine cycle, the same contract the ``predict`` feature uses).
        gammas: dissipation values to sweep. Include 0.0 (conservative, what the
            prediction feature uses) and the encode-time gamma, since the whole
            point is that they trade informativeness against settling.
        spread_threshold: tau in the collapse-length definition.
        n_anchors: how many windows to roll out (spread is a cross-sample
            statistic, so this is the sample size that determines its noise).
        seed: anchor subsampling seed.
    """

    horizon: int = 256
    gammas: tuple = (0.0, 0.1, 0.5)
    spread_threshold: float = 0.01
    n_anchors: int = 1500
    seed: int = 42

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0.0 < self.spread_threshold < 1.0:
            raise ValueError("spread_threshold must lie in (0, 1)")
        if self.n_anchors < 2:
            raise ValueError("n_anchors must be >= 2 (spread needs a sample)")
        if any(g < 0 for g in self.gammas):
            raise ValueError("gammas must be >= 0")

    def to_json(self) -> str:
        import json

        return json.dumps(asdict(self), sort_keys=True)


def collapse_length(s_rel: np.ndarray, threshold: float = 0.01) -> int | None:
    """First 1-indexed step at which ``S_rel`` drops below ``threshold``.

    Returns ``None`` if the rollout never collapses within the measured
    horizon — which must be reported as "> horizon", never as "no collapse".
    """
    below = np.where(np.asarray(s_rel) < threshold)[0]
    return int(below[0] + 1) if len(below) else None


def rollout_spread(
    model,
    q0,
    p0,
    steps: int,
    dt: float,
    gamma: float,
) -> dict:
    """Roll ``model`` forward from ``(q0, p0)`` and measure spread/boundedness.

    Args:
        model: any CHLU-like unit with ``__call__(q0, p0, steps, dt, gamma)``
            returning a ``(steps, 2*dim)`` trajectory.
        q0, p0: ``(N, C)`` anchor states.
        steps, dt, gamma: rollout length, timestep, dissipation.

    Returns:
        ``q`` (N, steps, C), ``S`` (steps,), ``S_rel`` (steps,), ``max_abs``
        (steps,) and ``S0``.
    """
    q0 = np.asarray(q0)
    c = q0.shape[1]
    traj = jax.vmap(lambda a, b: model(a, b, steps, dt, gamma))(q0, p0)
    q = np.asarray(traj[:, :, :c])
    s = q.std(axis=0).mean(axis=1)
    s0 = float(q0.std(axis=0).mean())
    return {
        "q": q,
        "S": s,
        "S0": s0,
        "S_rel": s / max(s0, 1e-12),
        "max_abs": np.abs(q).max(axis=(0, 2)),
    }


def rollout_diagnostic(
    model,
    q0,
    p0,
    dt: float,
    cfg: RolloutDiagConfig | None = None,
    truth: np.ndarray | None = None,
) -> dict:
    """Full bounded-vs-informative diagnostic, swept over ``cfg.gammas``.

    Args:
        truth: optional ``(N, horizon, C)`` ground-truth future states. When
            given, the error curve and a PERSISTENCE baseline (predict
            ``q(n) = q(0)``) are reported alongside the spread — the error curve
            is what stops a collapsed-but-bounded rollout from reading as a win.

    Returns:
        ``{f"gamma={g}": {...}}`` with ``collapse_length``, ``collapse_budget``
        (``gamma * n* * dt``), the ``S_rel`` curve, ``bounded``, and — if
        ``truth`` was supplied — ``mse`` and ``mse_persistence``.
    """
    cfg = cfg or RolloutDiagConfig()
    q0 = np.asarray(q0)
    out: dict = {}
    for g in cfg.gammas:
        r = rollout_spread(model, q0, p0, cfg.horizon, dt, float(g))
        n_star = collapse_length(r["S_rel"], cfg.spread_threshold)
        entry = {
            "S0": round(r["S0"], 6),
            "collapse_length": n_star,
            "collapse_budget": None if n_star is None else round(g * n_star * dt, 4),
            "S_rel": r["S_rel"].tolist(),
            "max_abs": r["max_abs"].tolist(),
            "bounded": bool(np.isfinite(r["max_abs"]).all()),
        }
        if truth is not None:
            t = np.asarray(truth)
            entry["mse"] = ((r["q"] - t) ** 2).mean(axis=(0, 2)).tolist()
            entry["mse_persistence"] = (
                ((q0[:, None, :] - t) ** 2).mean(axis=(0, 2)).tolist()
            )
        out[f"gamma={g}"] = entry
    return out
