"""Tests for the primitive harness (w20): the drop-in slot and its fairness guards.

The harness's entire value is in being trustworthy, so the tests here pin the
properties a reviewer would attack: that the slot really is interchangeable,
that every primitive is really causal (a non-causal block would cheat on every
recall number), that the parameter match is real, and that the tuning budget is
equal by construction.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from chlu.config import (
    ExperimentPrimitiveHarnessConfig,
    get_default_config,
    load_config,
    save_config,
)
from chlu.core.blocks import PRIMITIVES, CLUBlock, SequenceModel, make_block
from chlu.data.seq_tasks import generate_adding, generate_parity
from chlu.experiments.exp_primitive_harness import (
    AddingFamily,
    MQARFamily,
    ParityFamily,
    build_model,
    count_params,
    match_width,
    train_one,
)

KEY = jax.random.PRNGKey(0)


# --------------------------------------------------------------------------
# Item 1: the drop-in slot
# --------------------------------------------------------------------------
@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_block_is_shape_preserving(primitive):
    """Every primitive maps (T, d_model) -> (T, d_model). This IS the interface."""
    x = jax.random.normal(KEY, (16, 32))
    y = make_block(primitive, 32, 24, key=KEY)(x)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_block_is_causal(primitive):
    """Perturbing input t must not change any output before t.

    Without this, a recall result is meaningless — the block could read the
    answer from a later position.
    """
    T, t_pert = 12, 8
    x = jax.random.normal(KEY, (T, 16))
    block = make_block(primitive, 16, 12, key=KEY)
    diff = jnp.abs(block(x) - block(x.at[t_pert].add(5.0))).max(axis=1)
    assert float(diff[:t_pert].max()) == 0.0
    # ...and the perturbation must actually do something at t (guards a block
    # that is "causal" only because it ignores its input entirely).
    assert float(diff[t_pert]) > 1e-6


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_model_is_interchangeable(primitive):
    """The same SequenceModel signature accepts every primitive, both I/O modes."""
    discrete = SequenceModel(
        primitive, d_model=16, width=12, out_dim=5, max_len=8,
        vocab_size=9, n_layers=2, key=KEY,
    )
    assert discrete(jnp.arange(8)).shape == (8, 5)

    continuous = SequenceModel(
        primitive, d_model=16, width=12, out_dim=1, max_len=8,
        in_dim=2, n_layers=2, key=KEY,
    )
    assert continuous(jnp.zeros((8, 2))).shape == (8, 1)


def test_model_rejects_ambiguous_io_spec():
    with pytest.raises(ValueError, match="exactly one"):
        SequenceModel("gru", d_model=8, width=4, out_dim=2, max_len=4,
                      vocab_size=5, in_dim=3, key=KEY)


def test_unknown_primitive_raises():
    with pytest.raises(ValueError, match="Unknown primitive"):
        make_block("transformer_xl", 8, 4, key=KEY)


def test_clu_block_uses_the_real_chlu_unit():
    """The CLU block must not fork the physics — it wraps a genuine CHLU."""
    from chlu.core.chlu_unit import CHLU

    block = make_block("clu", 16, 8, key=KEY)
    assert isinstance(block, CLUBlock)
    assert isinstance(block.clu, CHLU)
    assert block.clu.dim == 8
    # Concession 3: the carry is (q, p), so the read is over 2*d_clu.
    assert block.w_out.in_features == 16


def test_clu_block_gamma_is_dissipative_by_default():
    """Concession 2: the SHIPPED default is gamma > 0.

    ⚠ The original justification (w19 clu-retrieval-demo §6: 1.000 at gamma=0.02
    vs 0.813 at gamma=0) was superseded in w20 — that 0.813 is a single-phase
    artifact. The default is retained only so the w20 harness numbers reproduce
    bit-for-bit; gamma is now a swept knob (w21 gamma-read-sweep). This test
    therefore pins REPRODUCIBILITY, not a physical requirement.
    """
    assert ExperimentPrimitiveHarnessConfig().clu_gamma == 0.05
    assert make_block("clu", 16, 8, key=KEY).gamma == 0.05


# --------------------------------------------------------------------------
# Item 2: matched budgets
# --------------------------------------------------------------------------
@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_param_match_within_tolerance(primitive):
    cfg = get_default_config().experiment_primitive_harness
    cfg.target_block_params = 20000
    family = MQARFamily(64, 4, 64)
    width, block_p, total_p, err = match_width(primitive, cfg, family, KEY)
    assert err <= cfg.param_tol, f"{primitive}: {block_p} vs {cfg.target_block_params}"
    assert width >= cfg.width_search_lo
    assert total_p > block_p  # embedding/pos/head are shared and non-empty


def test_shared_scaffolding_is_identical_across_primitives():
    """Everything outside the block must be the same object shape for all.

    This is what makes the parameter match meaningful: only the block differs.
    """
    cfg = get_default_config().experiment_primitive_harness
    family = MQARFamily(32, 2, 32)
    shared = {}
    for primitive in PRIMITIVES:
        model = build_model(primitive, cfg, family, 8, KEY)
        shared[primitive] = (
            count_params(model.embed),
            int(model.pos.size),
            count_params(model.head),
        )
    assert len(set(shared.values())) == 1, shared


# --------------------------------------------------------------------------
# Item 4: equal tuning budget, by construction
# --------------------------------------------------------------------------
def test_lr_grid_is_shared_and_fixed():
    """One grid for all primitives — no CLU-only knob can enter this way."""
    cfg = ExperimentPrimitiveHarnessConfig()
    assert len(cfg.lr_grid) >= 2
    assert cfg.primitives == list(PRIMITIVES)


# --------------------------------------------------------------------------
# Task families
# --------------------------------------------------------------------------
def test_adding_task_targets_are_the_marked_sum():
    d = generate_adding(KEY, 8, 32)
    vals, markers = d["inputs"][..., 0], d["inputs"][..., 1]
    assert jnp.allclose(markers.sum(axis=1), 2.0)  # exactly two marked positions
    expected = (vals * markers).sum(axis=1)
    got = d["targets"][:, -1, 0]
    assert jnp.allclose(got, expected, atol=1e-5)
    assert d["mask"].sum(axis=1).min() == 1  # supervised at the last step only


def test_parity_task_is_cumulative_xor():
    d = generate_parity(KEY, 8, 16)
    expected = jnp.cumsum(d["tokens"], axis=1) % 2
    assert jnp.array_equal(d["targets"], expected)
    assert bool(d["mask"].all())


def test_mqar_family_masks_non_query_positions():
    family = MQARFamily(32, 2, 32)
    tokens, targets, mask = family.batch(KEY, 4)
    assert tokens.shape == (4, 32)
    # Labels are clamped in-range so take_along_axis is safe; the MASK, not the
    # sentinel label, is what excludes non-query positions.
    assert int(targets.min()) >= 0
    assert int(mask.sum()) == 4 * 2


# --------------------------------------------------------------------------
# End-to-end (tiny)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", [MQARFamily(16, 2, 16), AddingFamily(16), ParityFamily(16)])
def test_train_one_runs_and_reports_cost(family):
    cfg = get_default_config().experiment_primitive_harness
    cfg.train_steps, cfg.eval_batch, cfg.batch_size = 3, 8, 4
    out = train_one("clu", cfg, family, width=4, lr=1e-3, seed=0, measure_cost=True)
    assert out["steps_run"] == 3
    assert not out["diverged"]
    # Item 2: compute cost must be *stated*, not implied by the parameter match.
    assert out["wallclock_s_per_step"] > 0
    assert "fwd_flops" in out


def test_gradients_flow_through_the_clu_block():
    """A finite, non-zero gradient must reach the potential net and log_mass.

    The clu-retrieval-demo §6 finding is that friction kills the *address*
    gradient; that must not be confused with the block being untrainable. Here
    the parameters are trained (not the address), and they must receive signal.
    """
    family = ParityFamily(12)
    cfg = get_default_config().experiment_primitive_harness
    model = build_model("clu", cfg, family, 6, KEY)
    x, y, mask = family.batch(KEY, 4)
    grads = eqx.filter_grad(family.loss)(model, x, y, mask)
    g_pot = grads.blocks[0].clu.potential_net.layers[0].weight
    g_mass = grads.blocks[0].clu.log_mass
    assert jnp.all(jnp.isfinite(g_pot)) and float(jnp.abs(g_pot).max()) > 0.0
    assert jnp.all(jnp.isfinite(g_mass)) and float(jnp.abs(g_mass).max()) > 0.0


# --------------------------------------------------------------------------
# Config plumbing (the three-site rule: dataclass, load_config, save_config)
# --------------------------------------------------------------------------
def test_harness_config_round_trips(tmp_path):
    config = get_default_config()
    config.experiment_primitive_harness.d_model = 123
    config.experiment_primitive_harness.lr_grid = [1e-2]
    config.experiment_primitive_harness.clu_gamma = 0.077
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.experiment_primitive_harness.d_model == 123
    assert loaded.experiment_primitive_harness.lr_grid == [1e-2]
    assert loaded.experiment_primitive_harness.clu_gamma == pytest.approx(0.077)


# --------------------------------------------------------------------------
# LR rescue pass (Item 4: the baselines must be real)
# --------------------------------------------------------------------------
def test_rescue_is_monotone_under_winners_curse():
    """The rescue must never LOWER a reported score.

    Regression test for a real bug in this harness: the probe that triggers a
    rescue is a single seed, but the reported number is an n_seeds mean, so a
    lucky probe could replace a good result with a worse average (observed:
    adding_T128/mlp, 0.1825 -> 0.1832 on a lower-is-better metric). The pass
    exists to protect baselines; silently degrading one would invert its purpose.
    """
    from chlu.experiments.exp_primitive_harness import run_lr_rescue

    cfg = get_default_config().experiment_primitive_harness
    cfg.train_steps, cfg.tune_steps, cfg.eval_batch = 2, 2, 8
    cfg.batch_size, cfg.n_seeds = 4, 2
    cfg.lr_grid = [1e-4, 1e-3]
    family = AddingFamily(8)  # lower-is-better metric
    prior = [{
        "primitive": "gru", "family": family.name, "width": 4,
        "best_lr": 1e-3, "metric_mean": -1.0,  # unbeatable: nothing may replace it
        "metric_std": 0.0, "all_diverged": False,
    }]
    out = run_lr_rescue(cfg, {family.name: family}, prior, log=lambda *a: None)
    assert out[0]["rescued"] is False
    assert out[0]["metric_mean"] == -1.0


def test_rescue_adopts_a_genuine_improvement():
    """...but it must still adopt an LR that really is better."""
    from chlu.experiments.exp_primitive_harness import run_lr_rescue

    cfg = get_default_config().experiment_primitive_harness
    cfg.train_steps, cfg.tune_steps, cfg.eval_batch = 2, 2, 8
    cfg.batch_size, cfg.n_seeds = 4, 2
    cfg.lr_grid = [1e-4, 1e-3]
    family = AddingFamily(8)
    prior = [{
        "primitive": "gru", "family": family.name, "width": 4,
        "best_lr": 1e-3, "metric_mean": 1e9,  # absurdly bad: anything beats it
        "metric_std": 0.0, "all_diverged": False,
    }]
    out = run_lr_rescue(cfg, {family.name: family}, prior, log=lambda *a: None)
    assert out[0]["rescued"] is True
    assert out[0]["metric_mean"] < 1e9
    assert out[0]["pre_rescue_metric_mean"] == 1e9


# --------------------------------------------------------------------------
# w21 gamma-read-sweep: CLU-internal read modes and the sweep driver
# --------------------------------------------------------------------------
def test_trajectory_read_is_identity_at_one_step():
    """At clu_steps=1 the fiber has ONE element, so the two read modes coincide.

    This is the load-bearing caveat of the (gamma x read-mode) table: at the
    shipped clu_steps=1 the read-mode axis is *provably* degenerate, so any
    apparent difference there would be a bug (different key consumption, a
    reordered concatenation, ...). Asserted bit-exactly, not approximately.
    """
    x = jax.random.normal(KEY, (10, 16))
    end = make_block("clu", 16, 8, key=KEY, clu_steps=1, clu_read_mode="endpoint")
    traj = make_block("clu", 16, 8, key=KEY, clu_steps=1, clu_read_mode="trajectory")
    assert traj.w_out.in_features == end.w_out.in_features == 16
    assert jnp.array_equal(traj.w_out.weight, end.w_out.weight)
    assert jnp.array_equal(traj(x), end(x))


def test_trajectory_read_widens_the_readout_and_differs_at_multistep():
    """At clu_steps>1 the fiber read consumes the whole intra-token rollout."""
    x = jax.random.normal(KEY, (10, 16))
    end = make_block("clu", 16, 8, key=KEY, clu_steps=4, clu_read_mode="endpoint")
    traj = make_block("clu", 16, 8, key=KEY, clu_steps=4, clu_read_mode="trajectory")
    assert end.w_out.in_features == 2 * 8
    assert traj.w_out.in_features == 4 * 2 * 8  # clu_steps x (q, p)
    assert not jnp.allclose(traj(x), end(x))
    assert jnp.all(jnp.isfinite(traj(x)))


def test_unknown_read_mode_raises():
    with pytest.raises(ValueError, match="read_mode"):
        make_block("clu", 16, 8, key=KEY, clu_read_mode="settled")


def test_trajectory_read_stays_causal():
    """The fiber read must not leak future tokens (it only widens the per-token read)."""
    T, t_pert = 12, 8
    x = jax.random.normal(KEY, (T, 16))
    block = make_block("clu", 16, 12, key=KEY, clu_steps=3, clu_read_mode="trajectory")
    diff = jnp.abs(block(x) - block(x.at[t_pert].add(5.0))).max(axis=1)
    assert float(diff[:t_pert].max()) == 0.0
    assert float(diff[t_pert]) > 1e-6


def test_gamma_zero_block_is_volume_preserving():
    """gamma=0 restores the symplectic map: det J of one Verlet step is 1.

    The whole point of the sweep is that gamma is a knob, so the gamma=0 end of
    the grid must genuinely be the conservative unit and not a near-miss.
    """
    block = make_block("clu", 8, 4, key=KEY, clu_gamma=0.0)
    q, p = jax.random.normal(KEY, (4,)), jax.random.normal(jax.random.PRNGKey(1), (4,))

    def one_step(z):
        qq, pp = block.clu.step((z[:4], z[4:]), block.dt, block.gamma)
        return jnp.concatenate([qq, pp])

    jac = jax.jacobian(one_step)(jnp.concatenate([q, p]))
    assert float(jnp.abs(jnp.linalg.det(jac) - 1.0)) < 1e-5


def test_memory_half_life_matches_the_2ln2_over_gamma_rule():
    """The quoted 'half-life ~ 2 ln2 / gamma tokens' must be what the code does."""
    import numpy as np

    from chlu.experiments.exp_primitive_harness import memory_half_life_tokens

    assert memory_half_life_tokens(0.0) == float("inf")
    for g in (0.001, 0.01, 0.05):
        assert memory_half_life_tokens(g) == pytest.approx(2 * np.log(2) / g, rel=0.05)
    # clu_steps damps clu_steps times per TOKEN, so the half-life shortens.
    assert memory_half_life_tokens(0.05, 4) == pytest.approx(
        memory_half_life_tokens(0.05, 1) / 4, rel=1e-6
    )


def test_sweep_cell_restores_config_and_tags_results(tmp_path):
    """A sweep cell must not leak its overrides into the next cell."""
    from chlu.experiments.exp_primitive_harness import _sweep_cell

    cfg = get_default_config().experiment_primitive_harness
    cfg.train_steps, cfg.tune_steps, cfg.eval_batch = 2, 2, 8
    cfg.batch_size, cfg.n_seeds, cfg.lr_grid = 4, 1, [1e-3]
    cfg.target_block_params = 2000
    family = AddingFamily(8)
    results = []
    _sweep_cell(cfg, family, {"clu_gamma": 0.0, "clu_steps": 2}, results,
                str(tmp_path / "sweep.json"), log=lambda *a: None)
    assert cfg.clu_gamma == 0.05 and cfg.clu_steps == 1  # restored
    assert results[0]["cell"] == {"clu_gamma": 0.0, "clu_steps": 2}
    assert results[0]["half_life_tokens"] == float("inf")
    assert (tmp_path / "sweep.json").exists()


def test_gamma_sweep_grids_round_trip(tmp_path):
    config = get_default_config()
    cfg = config.experiment_primitive_harness
    assert cfg.clu_read_mode == "endpoint"  # shipped behaviour preserved
    assert cfg.clu_gamma_sweep[0] == 0.0 and cfg.clu_gamma in cfg.clu_gamma_sweep
    cfg.clu_gamma_sweep = [0.0, 0.02]
    cfg.clu_read_mode = "trajectory"
    cfg.clu_steps_sweep = [1, 8]
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path).experiment_primitive_harness
    assert loaded.clu_gamma_sweep == [0.0, 0.02]
    assert loaded.clu_read_mode == "trajectory"
    assert loaded.clu_steps_sweep == [1, 8]


def test_gated_write_is_off_by_default_and_bit_identical():
    """EXPLORATORY write mode must not perturb the shipped block by one bit.

    The gate key is folded in from k2 rather than taken from a 4-way split
    precisely so the default block's initialisation is unchanged; a 4-way split
    would silently re-randomise every published w20 CLU cell.
    """
    x = jax.random.normal(KEY, (10, 16))
    default = make_block("clu", 16, 8, key=KEY)
    explicit = make_block("clu", 16, 8, key=KEY, clu_write_mode="linear")
    assert default.write_mode == "linear" and default.w_gate is None
    assert jnp.array_equal(default(x), explicit(x))
    assert ExperimentPrimitiveHarnessConfig().clu_write_mode == "linear"


def test_gated_write_adds_a_multiplicative_input_gate():
    x = jax.random.normal(KEY, (10, 16))
    gated = make_block("clu", 16, 8, key=KEY, clu_write_mode="gated")
    assert gated.w_gate is not None
    assert gated.w_gate.weight.shape == gated.w_in.weight.shape
    # Same w_in as the linear block (only the gate is new), yet a different map.
    assert jnp.array_equal(gated.w_in.weight, make_block("clu", 16, 8, key=KEY).w_in.weight)
    assert not jnp.allclose(gated(x), make_block("clu", 16, 8, key=KEY)(x))
    assert jnp.all(jnp.isfinite(gated(x)))


def test_unknown_write_mode_raises():
    with pytest.raises(ValueError, match="write_mode"):
        make_block("clu", 16, 8, key=KEY, clu_write_mode="conditional")
