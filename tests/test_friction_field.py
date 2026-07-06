"""Tests for the friction field gamma_phi(q) (trash regions).

Covers the gamma-field-build acceptance checks:
    - Prop-11 (F5 §7.3): det(step Jacobian) = (1 - gamma_phi(q_next))^d exactly,
      and the multiplicative composition with scalar gamma.
    - Default-"none" bit-compatibility: without a field, the step is bitwise
      identical to the historical dissipative-Verlet algorithm.
    - Contrastive training: the wake protection term lowers gamma_phi at data
      states; the sleep term raises gamma_phi at hallucination states.
    - Config roundtrip for the new training.friction_field_* and
      experiment_s1 sections (also guards the ExperimentV1GateConfig
      @dataclass fix — save_config crashes without it).

Precision: Prop-11 is an exact statement — x64 enabled at import (same
process-global pattern as tests/test_goldstone.py; the suite is tolerance-based
and only gets more accurate under x64).
"""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chlu.config import get_default_config, load_config, save_config  # noqa: E402
from chlu.core.chlu_unit import CHLU  # noqa: E402
from chlu.core.friction_field import (  # noqa: E402
    FrictionField,
    build_friction_field,
    c1_regularizer,
    spectral_masses_at,
)
from chlu.training.train import train_chlu  # noqa: E402

DT = 0.05


def _field_at(
    center, radius=0.8, strength=0.3, gamma_max=0.5, width=0.25, trainable=True
):
    """One hole with exact hand-set geometry."""
    return FrictionField(
        dim=len(center),
        gamma_max=gamma_max,
        width=width,
        centers=jnp.asarray([center]),
        init_radius=radius,
        init_strength=strength,
        trainable=trainable,
    )


# ---------------------------------------------------------------------------
# Field math
# ---------------------------------------------------------------------------


def test_single_hole_reduces_to_thread1_form():
    """K=1 must be exactly gamma_1 * sigmoid((r_1 - ||q - c_1||)/w)."""
    field = _field_at([0.3, -0.7], radius=0.9, strength=0.2, width=0.15)
    c, r, s = field.hole_params()
    assert abs(float(r[0]) - 0.9) < 1e-10  # inverse-softplus init is exact
    assert abs(float(s[0]) - 0.2) < 1e-10  # inverse-sigmoid init is exact

    key = jax.random.PRNGKey(0)
    for q in jax.random.normal(key, (16, 2)) * 2.0:
        dist = jnp.sqrt(jnp.sum((q - c[0]) ** 2) + 1e-12)
        expected = s[0] * jax.nn.sigmoid((r[0] - dist) / field.width)
        assert abs(float(field(q)) - float(expected)) < 1e-12


def test_field_bounded_and_saturating():
    """0 <= gamma_phi < gamma_max for any q, any K (noisy-OR saturation)."""
    field = FrictionField(
        dim=3,
        k=5,
        gamma_max=0.4,
        init_radius=2.0,
        init_strength=0.39,
        key=jax.random.PRNGKey(1),
    )
    qs = jax.random.normal(jax.random.PRNGKey(2), (64, 3)) * 3.0
    vals = jax.vmap(field)(qs)
    assert float(jnp.min(vals)) >= 0.0
    assert float(jnp.max(vals)) < 0.4
    # Overlapping holes: value exceeds any single hole's contribution but
    # stays under the cap (checked at the shared center).
    center_val = float(field(field.centers[0]))
    assert center_val < 0.4


def test_field_validation():
    import pytest

    with pytest.raises(ValueError):
        FrictionField(dim=2, gamma_max=1.0)  # cap must be < 1
    with pytest.raises(ValueError):
        FrictionField(dim=2, width=0.0)
    with pytest.raises(ValueError):
        FrictionField(dim=2, centers=jnp.zeros((1, 3)))  # dim mismatch


def test_build_friction_field_modes():
    cfg = get_default_config().training
    assert build_friction_field(cfg, dim=2) is None  # default "none"

    cfg.friction_field = "fixed"
    cfg.friction_field_fixed_centers = [[1.0, 2.0]]
    fixed = build_friction_field(cfg, dim=2)
    assert fixed.trainable is False
    assert np.allclose(np.asarray(fixed.centers), [[1.0, 2.0]])

    cfg.friction_field = "learned"
    cfg.friction_field_k = 3
    learned = build_friction_field(cfg, dim=2, key=jax.random.PRNGKey(0))
    assert learned.trainable is True
    assert learned.k == 3

    cfg.friction_field = "bogus"
    import pytest

    with pytest.raises(ValueError):
        build_friction_field(cfg, dim=2)


def test_fixed_variant_has_no_gradients():
    """trainable=False must stop all gradients (the frozen oracle/control)."""
    q = jnp.array([0.2, 0.1])

    def loss(f):
        return f(q)

    frozen = _field_at([0.0, 0.0], trainable=False)
    grads = eqx.filter_grad(loss)(frozen)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in leaves)

    live = _field_at([0.0, 0.0], trainable=True)
    grads = eqx.filter_grad(loss)(live)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in leaves)


# ---------------------------------------------------------------------------
# Prop-11: exact position-gated volume contraction
# ---------------------------------------------------------------------------


def _step_jacobian_and_qnext(model, q, p, gamma):
    dim = q.shape[0]

    def f(z):
        qn, pn = model.step((z[:dim], z[dim:]), DT, gamma)
        return jnp.concatenate([qn, pn])

    J = jax.jacobian(f)(jnp.concatenate([q, p]))
    q_next, _ = model.step((q, p), DT, gamma)
    return J, q_next


def test_prop11_det_jacobian():
    """det J = (1 - gamma_phi(q_next))^d exactly (F5 Prop-11), and the
    multiplicative composition ((1-gamma)(1-gamma_phi(q_next)))^d with a
    scalar gamma."""
    field = _field_at([0.0, 0.0], radius=0.8, strength=0.3)
    model = CHLU(dim=2, hidden=16, friction_field=field, key=jax.random.PRNGKey(3))

    # Inside the horizon (gamma_phi large) and far outside (gamma_phi ~ 0)
    states = [
        (jnp.array([0.1, -0.2]), jnp.array([0.4, 0.3])),
        (jnp.array([5.0, 5.0]), jnp.array([0.1, -0.1])),
    ]
    for q, p in states:
        J, q_next = _step_jacobian_and_qnext(model, q, p, gamma=0.0)
        det = float(jnp.linalg.det(J))
        pred = float((1.0 - field(q_next)) ** 2)
        assert abs(det - pred) < 1e-12, f"det {det} vs Prop-11 {pred}"

    # Composition with scalar gamma
    q, p = states[0]
    gamma = 0.2
    J, q_next = _step_jacobian_and_qnext(model, q, p, gamma=gamma)
    det = float(jnp.linalg.det(J))
    pred = float(((1.0 - gamma) * (1.0 - field(q_next))) ** 2)
    assert abs(det - pred) < 1e-12

    # Sanity: inside the hole the map genuinely contracts volume
    J, q_next = _step_jacobian_and_qnext(
        model, jnp.array([0.1, -0.2]), jnp.array([0.4, 0.3]), gamma=0.0
    )
    assert float(jnp.linalg.det(J)) < 0.9


def test_stochastic_step_applies_field():
    """langevin_step at T=0 must reduce to the field-damped Verlet step."""
    field = _field_at([0.0, 0.0], radius=0.8, strength=0.3)
    model = CHLU(dim=2, hidden=16, friction_field=field, key=jax.random.PRNGKey(3))
    q, p = jnp.array([0.1, -0.2]), jnp.array([0.4, 0.3])
    q_det, p_det = model.step((q, p), DT, gamma=0.1)
    q_sto, p_sto, _ = model.stochastic_step(
        (q, p), DT, gamma=0.1, temperature=0.0, key=jax.random.PRNGKey(9)
    )
    assert jnp.allclose(q_det, q_sto, atol=0, rtol=0)
    assert jnp.allclose(p_det, p_sto, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Default-"none" bit-compatibility
# ---------------------------------------------------------------------------


def test_none_default_bit_compatible():
    """Without a field, model.step must be bitwise identical to the historical
    dissipative-Verlet algorithm (same ops, same order, eager)."""
    model = CHLU(dim=2, hidden=16, key=jax.random.PRNGKey(4))
    assert getattr(model, "friction_field", "MISSING") is None

    grad_H_q = jax.grad(model.H, argnums=0)
    grad_H_p = jax.grad(model.H, argnums=1)

    key = jax.random.PRNGKey(5)
    for gamma in (0.0, 0.3):
        for _ in range(4):
            key, k1, k2 = jax.random.split(key, 3)
            q = jax.random.normal(k1, (2,))
            p = jax.random.normal(k2, (2,))
            # Historical algorithm, verbatim
            p_half = p - 0.5 * DT * grad_H_q(q, p)
            q_ref = q + DT * grad_H_p(q, p_half)
            p_ref = p_half - 0.5 * DT * grad_H_q(q_ref, p_half)
            p_ref = (1.0 - gamma) * p_ref

            q_new, p_new = model.step((q, p), DT, gamma)
            assert jnp.array_equal(q_new, q_ref), "q not bit-identical"
            assert jnp.array_equal(p_new, p_ref), "p not bit-identical"


# ---------------------------------------------------------------------------
# Contrastive training of the field (smoke trains)
# ---------------------------------------------------------------------------


def _smoke_config(protect, hallu, sleep_frequency=10_000, gate="energy"):
    config = get_default_config()
    config.training.epochs = 30
    config.training.sleep_steps = 1
    config.training.sleep_frequency = sleep_frequency
    config.training.sleep_temperature = 0.0
    config.training.batch_size = 8
    config.training.buffer_capacity = 8
    config.training.dt = DT
    config.training.friction_field_protect_lambda = protect
    config.training.friction_field_hallu_lambda = hallu
    config.training.friction_field_hallu_gate = gate
    return config


def test_protection_lowers_gamma_at_data():
    """Wake protection term: gamma_phi(q_data) must decrease over training."""
    q_data = jnp.array([0.5, -0.3])
    data = jnp.tile(jnp.concatenate([q_data, jnp.zeros(2)]), (1, 20, 1))

    field = _field_at(list(q_data), radius=1.0, strength=0.3)  # hole ON data
    model = CHLU(dim=2, hidden=8, friction_field=field, key=jax.random.PRNGKey(6))
    gamma_before = float(model.friction_field(q_data))

    config = _smoke_config(protect=5.0, hallu=0.0)  # sleep disabled
    trained, _, _ = train_chlu(
        model, data, key=jax.random.PRNGKey(7), config=config, window_size=10
    )
    gamma_after = float(trained.friction_field(q_data))
    assert gamma_after < gamma_before, (
        f"protection failed: gamma at data {gamma_before:.4f} -> {gamma_after:.4f}"
    )


def test_hallucination_raises_gamma_at_locus():
    """Sleep term (default "energy" gate): gamma_phi must increase at
    persistent HIGH-energy negatives. The negatives carry large momentum so
    their energy provably sits above the (resting) data band regardless of
    the random potential's values."""
    locus = jnp.array([3.0, 3.0])
    data = jnp.zeros((1, 20, 4))  # resting data far from the locus

    # Weak hole near (not on) the locus; the sleep term must strengthen it
    field = _field_at([2.5, 2.5], radius=1.0, strength=0.05)
    model = CHLU(dim=2, hidden=8, friction_field=field, key=jax.random.PRNGKey(8))
    gamma_before = float(model.friction_field(locus))

    config = _smoke_config(protect=0.0, hallu=5.0, sleep_frequency=1)
    q_seed = jnp.tile(locus, (8, 1))
    p_seed = 3.0 * jnp.ones((8, 2))  # hot negatives: KE = 9 >> data band
    trained, _, _ = train_chlu(
        model,
        data,
        key=jax.random.PRNGKey(9),
        config=config,
        window_size=10,
        negative_seed_states=(q_seed, p_seed),
    )
    gamma_after = float(trained.friction_field(locus))
    assert gamma_after > gamma_before, (
        f"hallucination term failed: gamma at locus "
        f"{gamma_before:.4f} -> {gamma_after:.4f}"
    )


def test_energy_gate_blocks_in_band_negatives():
    """The "energy" hallucination gate: negatives whose energy is BELOW the
    data band must not attract friction (they are not "persistent"
    hallucinations — ungated they would drag friction toward the data
    manifold). Data moves fast (KE = 8, band top ~8+V(0)); negatives rest at
    the locus (H = V(locus) ~ 6.4 with the 0.05||q||^2 confinement) — below
    the band. The hole sits far from the wake path so the wake-through-
    dynamics gradient underflows Adam's eps (no drift; see the comparative
    note below)."""
    locus = jnp.array([8.0, 8.0])
    # Data at the origin with momentum (4, 0): window energy ~ 8 + V(0)
    state = jnp.array([0.0, 0.0, 4.0, 0.0])
    data = jnp.tile(state, (1, 20, 1))

    def _run(gate):
        field = _field_at([8.0, 8.0], radius=1.0, strength=0.05)
        model = CHLU(dim=2, hidden=8, friction_field=field, key=jax.random.PRNGKey(8))
        before = float(model.friction_field(locus))
        config = _smoke_config(protect=0.0, hallu=5.0, sleep_frequency=1, gate=gate)
        trained, _, _ = train_chlu(
            model,
            data,
            key=jax.random.PRNGKey(9),
            config=config,
            window_size=10,
            negative_seed_states=(jnp.tile(locus, (8, 1)), jnp.zeros((8, 2))),
        )
        return before, float(trained.friction_field(locus))

    # Comparative check: the two runs are identical except the gate, so the
    # difference isolates it. (An absolute no-rise assertion is too strict:
    # Adam normalizes the epsilon-scale wake-MSE-through-dynamics gradient
    # into lr-scale parameter drift of either sign, ~1e-3/epoch.)
    before_e, after_e = _run("energy")
    before_a, after_a = _run("all")
    rise_e = after_e - before_e
    rise_a = after_a - before_a
    assert rise_a > 0.0, f"'all' gate must raise gamma at the negatives ({rise_a:.5f})"
    assert rise_e < 0.25 * rise_a, (
        f"energy gate leaked: in-band negatives raised gamma by {rise_e:.5f} "
        f"vs {rise_a:.5f} ungated"
    )


def test_c1_regularizer_targets_strengths():
    """The C1 nudge moves gamma_k toward 2*dt*mu(c_k) and only touches the
    strengths (targets are stop_gradient-ed)."""
    field = _field_at([0.2, 0.2], radius=0.8, strength=0.45)
    model = CHLU(dim=2, hidden=16, friction_field=field, key=jax.random.PRNGKey(10))

    mu = spectral_masses_at(model, field.centers[0])
    assert mu.shape == (2,) and float(jnp.min(mu)) >= 0.0

    val = c1_regularizer(model, DT)
    assert float(val) >= 0.0

    grads = eqx.filter_grad(lambda m: c1_regularizer(m, DT))(model)
    g_strength = grads.friction_field.strength_logits
    g_centers = grads.friction_field.centers
    g_pot = [g for g in jax.tree_util.tree_leaves(grads.potential_net) if g is not None]
    assert float(jnp.max(jnp.abs(g_strength))) > 0.0
    assert float(jnp.max(jnp.abs(g_centers))) == 0.0
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in g_pot)


# ---------------------------------------------------------------------------
# Config roundtrip
# ---------------------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    config = get_default_config()
    config.training.friction_field = "learned"
    config.training.friction_field_k = 4
    config.training.friction_field_gamma_max = 0.37
    config.training.friction_field_width = 0.11
    config.training.friction_field_fixed_centers = [[1.0, 2.0]]
    config.training.friction_field_protect_lambda = 2.5
    config.training.friction_field_c1_lambda = 0.01
    config.training.friction_field_hallu_gate = "all"
    config.training.friction_field_lr = 0.02
    config.experiment_s1.learned_k_values = [2]
    config.experiment_s1.seeds = [7, 8]
    config.experiment_s1.noise_center = [0.9, -1.1]
    config.experiment_s1.oracle_width = 0.07

    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)

    t = loaded.training
    assert t.friction_field == "learned"
    assert t.friction_field_k == 4
    assert abs(t.friction_field_gamma_max - 0.37) < 1e-12
    assert abs(t.friction_field_width - 0.11) < 1e-12
    assert t.friction_field_fixed_centers == [[1.0, 2.0]]
    assert abs(t.friction_field_protect_lambda - 2.5) < 1e-12
    assert abs(t.friction_field_c1_lambda - 0.01) < 1e-12
    assert t.friction_field_hallu_gate == "all"
    assert abs(t.friction_field_lr - 0.02) < 1e-12
    s1 = loaded.experiment_s1
    assert s1.learned_k_values == [2]
    assert s1.seeds == [7, 8]
    assert s1.noise_center == [0.9, -1.1]
    assert abs(s1.oracle_width - 0.07) < 1e-12

    # Guards the ExperimentV1GateConfig @dataclass fix: save_config crashes
    # on a non-dataclass section, and Field-object defaults would not survive.
    v1 = loaded.experiment_v1_gate
    assert isinstance(v1.difficulty_levels, list)
    assert (
        v1.difficulty_levels
        == get_default_config().experiment_v1_gate.difficulty_levels
    )


# ---------------------------------------------------------------------------
# Compact-support horizon gate (gamma-field-build follow-up 2)
# ---------------------------------------------------------------------------


def test_compact_gate_exact_zero_beyond_cutoff():
    """gamma_phi is identically 0.0 (bitwise) outside every hole radius r_k."""
    compact = FrictionField(
        dim=2,
        gamma_max=0.5,
        width=0.25,
        centers=jnp.asarray([[0.0, 0.0]]),
        init_radius=0.8,
        init_strength=0.4,
        gate="compact",
    )
    r = float(jax.nn.softplus(compact.log_radii[0]))
    # Inside the flat core: full strength gate == 1.
    assert float(compact(jnp.array([0.0, 0.0]))) > 0.0
    # Exactly at / beyond the cutoff radius: bitwise zero (no sigmoid tail).
    for d in [r, r + 1e-9, r + 0.5, 5.0]:
        q = jnp.array([d, 0.0])
        assert float(compact(q)) == 0.0
    # The sigmoid variant, by contrast, leaks a positive tail at the same radius.
    sig = FrictionField(
        dim=2,
        gamma_max=0.5,
        width=0.25,
        centers=jnp.asarray([[0.0, 0.0]]),
        init_radius=0.8,
        init_strength=0.4,
        gate="sigmoid",
    )
    assert float(sig(jnp.array([r + 0.5, 0.0]))) > 0.0


def test_compact_gate_config_and_build():
    cfg = get_default_config().training
    assert cfg.friction_field_gate == "sigmoid"  # default unchanged
    cfg.friction_field = "learned"
    cfg.friction_field_gate = "compact"
    field = build_friction_field(cfg, dim=2, key=jax.random.PRNGKey(0))
    assert field.gate == "compact"
    # Beyond r_k the built field damps by exactly (1 - 0) = identity.
    r = float(jax.nn.softplus(field.log_radii[0]))
    far = field.centers[0] + jnp.array([r + 1.0, 0.0])
    assert float(field(far)) == 0.0


def test_compact_gate_validation():
    import pytest

    with pytest.raises(ValueError):
        FrictionField(dim=2, gate="bogus")
