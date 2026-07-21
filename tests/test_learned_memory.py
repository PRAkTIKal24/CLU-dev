"""Tests for the LEARNED write -> address -> read loop (exp_learned_memory, w20).

These pin the things that, if they silently broke, would turn a negative result
into a fake positive (or vice versa):

  * the designed part of a ``DesignFreedomPotential`` must stay FIXED during
    training — if the "designed" terms drifted, the design-freedom ladder would
    not be a ladder;
  * the ``designed`` rung must have zero learnable parameters and must still
    reproduce the w19 loop through the NEW two-phase retrieval path (the
    baseline that licenses interpreting the other rungs);
  * the anti-decoration guard (q2 = p2 = 0 at launch) must survive into the new
    query generator;
  * ``trainable_filter`` must return BOOLEANS (equinox rejects an array-valued
    filter spec — the first bug this harness hit);
  * a blank landscape must NOT read above chance on the payload channel.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config, load_config, save_config
from chlu.core.memory_potentials import (
    DESIGN_RUNGS,
    DesignFreedomPotential,
    RBFAtoms,
    designed_payloads,
    ring_sites,
)
from chlu.experiments.exp_learned_memory import (
    build_landscape,
    evaluate_cell,
    item2_design_freedom,
    make_queries,
    model_for,
    write_items,
)
from chlu.training.train_memory import trainable_filter


def _cfg(**kw):
    cfg = get_default_config().experiment_learned_memory
    # Small but real: these tests must run in seconds, not minutes.
    cfg.write_steps = 60
    cfg.write_n_perturb = 8
    cfg.n_query_per_item = 6
    cfg.address_steps = 60
    cfg.read_steps = 60
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# The potential family
# ---------------------------------------------------------------------------


def test_all_rungs_finite_and_differentiable_including_origin():
    pay = designed_payloads(4, seed=0)
    for rung in DESIGN_RUNGS:
        V = DesignFreedomPotential(rung, 3, pay, jax.random.PRNGKey(0))
        gradV = jax.grad(
            lambda x, pot=V: pot(x)
        )  # bind V, don't close over the loop var
        for q in (
            jnp.zeros(3),
            jnp.array([1.0, 0.0, 0.5]),
            jnp.array([1e-9, -1e-9, 0.0]),
        ):
            assert jnp.isfinite(V(q)), f"{rung}: V not finite at {q}"
            g = gradV(q)
            assert jnp.all(jnp.isfinite(g)), f"{rung}: grad V not finite at {q}"


def test_design_freedom_is_monotone_and_designed_rung_has_no_learned_params():
    pay = designed_payloads(4, seed=0)
    freedoms = []
    for rung in DESIGN_RUNGS:
        V = DesignFreedomPotential(rung, 3, pay, jax.random.PRNGKey(0))
        freedoms.append(V.design_freedom)
        if rung == "designed":
            assert V.learned is None
            assert trainable_filter(V) is None
        else:
            assert trainable_filter(V) is not None
    assert freedoms == sorted(freedoms) == list(range(len(DESIGN_RUNGS)))


def test_trainable_filter_is_boolean_valued():
    """equinox requires a boolean filter spec; an array-valued one raises."""
    pay = designed_payloads(3, seed=0)
    V = DesignFreedomPotential("free_mlp", 3, pay, jax.random.PRNGKey(0))
    spec = trainable_filter(V)
    leaves = jax.tree_util.tree_leaves(spec)
    assert leaves, "filter spec has no leaves"
    assert all(isinstance(x, bool) for x in leaves)
    params, static = eqx.partition(V, spec)  # must not raise
    assert jax.tree_util.tree_leaves(params)


def test_training_leaves_the_designed_part_untouched():
    """The ladder is only a ladder if the designed terms do not drift."""
    pay = designed_payloads(4, seed=0)
    cfg = _cfg()
    V = build_landscape("skeleton_residual", cfg, pay, jax.random.PRNGKey(0))
    before = jax.tree_util.tree_leaves(eqx.filter(V.designed, eqx.is_inexact_array))
    V2, hist = write_items(V, cfg, pay, jax.random.PRNGKey(1))
    after = jax.tree_util.tree_leaves(eqx.filter(V2.designed, eqx.is_inexact_array))
    assert len(before) == len(after) and before
    for b, a in zip(before, after, strict=True):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
    # and the learned part DID move
    lb = jax.tree_util.tree_leaves(eqx.filter(V.learned, eqx.is_inexact_array))
    la = jax.tree_util.tree_leaves(eqx.filter(V2.learned, eqx.is_inexact_array))
    assert any(
        not np.allclose(np.asarray(b), np.asarray(a))
        for b, a in zip(lb, la, strict=True)
    )
    # NB: no assertion on hist[-1] < hist[0] here — the write loss is stochastic
    # (fresh perturbations each step) and this rung starts near-converged, so a
    # single-step comparison is noise. Descent is asserted in
    # test_write_loss_decreases_..., which uses a rung that starts far away.
    assert hist


def test_write_loss_decreases_and_makes_targets_lower_than_their_neighbourhood():
    # K=4, NOT K=3: an odd K puts an exact 0.0 in the payload grid, and for that
    # item the target IS its own query point (q2 = a_i = 0), so "target strictly
    # below its query point" is unsatisfiable by construction.
    pay = designed_payloads(4, seed=0)
    cfg = _cfg(write_steps=200)
    V = build_landscape("free_mlp", cfg, pay, jax.random.PRNGKey(0))
    targets = ring_sites(4, f=cfg.f, dim=3, payloads=pay)
    V2, hist = write_items(V, cfg, pay, jax.random.PRNGKey(1))
    assert hist[-1] < hist[0]
    # each written target should sit below the q2=0 query manifold above it
    for z in np.asarray(targets):
        q = jnp.asarray(z)
        query = q.at[2].set(0.0)
        assert float(V2(q)) < float(V2(query)), "target is not below its query point"


def test_rbf_atoms_only_dig_wells():
    """softplus depths => the learned atom term is non-positive everywhere."""
    V = RBFAtoms(3, 8, jax.random.PRNGKey(0), confine=0.0)
    pts = jax.random.normal(jax.random.PRNGKey(1), (32, 3))
    vals = np.asarray(jax.vmap(V)(pts))
    assert np.all(vals <= 1e-6)


# ---------------------------------------------------------------------------
# The loop: guards and the designed baseline
# ---------------------------------------------------------------------------


def test_queries_obey_the_anti_decoration_guard():
    """q2(0) = p2(0) = 0 always, else the read could get the payload from the
    address and every retrieval number becomes decorative."""
    cfg = _cfg()
    Q0, P0, labels = make_queries(jax.random.PRNGKey(0), 4, 5, cfg, dim=3)
    np.testing.assert_allclose(np.asarray(Q0)[:, 2], 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(P0)[:, 2], 0.0, atol=0.0)
    assert len(labels) == 20
    # the address plane IS jittered (otherwise the "perturbed query" is not one)
    assert np.std(np.asarray(Q0)[:, :2]) > 0.0


def test_designed_rung_reproduces_the_w19_loop_through_two_phase_retrieval():
    """The baseline that licenses interpreting every other rung. If this fails,
    the harness is wrong, not the physics (committed in PREREG)."""
    # The settling budget matters: the w19 loop ran 1200 steps and this one
    # splits 400 (relax) + 400 (read). At 200+200 the particle has not settled
    # and strict success reads 0.06 for reasons that are not about the landscape.
    cfg = _cfg(n_query_per_item=8, address_steps=400, read_steps=400)
    pay = designed_payloads(2, seed=0)
    V = build_landscape("designed", cfg, pay, jax.random.PRNGKey(0))
    cell = evaluate_cell(model_for(V), cfg, pay, seed=42)
    assert cell["finite"]
    assert cell["basin_success_rate"] == 1.0
    assert cell["strict_success_rate"] >= 0.9
    assert cell["acc_payload_codebook_read"] >= 0.9
    assert cell["payload_abs_err_mean"] < cfg.payload_tol


def test_blank_landscape_reads_at_chance_on_the_payload_channel():
    """The load-bearing control: nothing stored => nothing retrievable."""
    cfg = _cfg(n_query_per_item=16, address_steps=200, read_steps=200)
    pay = designed_payloads(2, seed=0)
    blank = build_landscape(
        "designed", cfg, jnp.zeros_like(jnp.asarray(pay)), jax.random.PRNGKey(0)
    )
    cell = evaluate_cell(model_for(blank), cfg, pay, seed=42)
    assert cell["acc_payload_codebook_read"] <= cell["chance"] + cfg.blank_margin


def test_two_phase_retrieval_is_not_the_same_as_single_phase():
    """gamma_address and gamma_read are genuinely separate knobs (item 4 would be
    vacuous if the second phase ignored its own gamma)."""
    cfg = _cfg(n_query_per_item=4, address_steps=50, read_steps=200)
    pay = designed_payloads(2, seed=0)
    V = build_landscape("designed", cfg, pay, jax.random.PRNGKey(0))
    m = model_for(V)
    a = evaluate_cell(m, cfg, pay, 0, gamma_address=0.0, gamma_read=0.0)
    b = evaluate_cell(m, cfg, pay, 0, gamma_address=0.0, gamma_read=0.2)
    assert a["payload_abs_err_mean"] != b["payload_abs_err_mean"]


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def test_failing_blank_control_disqualifies_a_rung():
    """A cell whose blank control fails is NOT a measurement, however good the
    written number is — the scoring must enforce that, not just report it."""
    cfg = _cfg()
    fake = {
        "rows": [
            {
                "rung": "free_mlp",
                "design_freedom": 4,
                "n_learned_params": 10,
                "written": {
                    "K": 2,
                    "basin_success_rate": 1.0,
                    "strict_success_rate": 1.0,
                    "acc_payload_codebook_read": 1.0,
                    "payload_abs_err_mean": 0.0,
                },
                "blank": {
                    "acc_payload_codebook_read": 1.0,
                    "read_val_site_spread": 0.4,
                },
                "blank_control_passes": False,
            }
        ],
        "w19_baseline": {},
    }
    cfg.rungs = ["free_mlp"]
    out = item2_design_freedom(fake, cfg)
    assert out["curve"][0]["passes"] is False
    assert out["minimum_viable_design_rung"] is None
    assert out["loop_survives_learning"] is False


def test_learned_memory_config_round_trips(tmp_path):
    """New config groups must be wired into CHLUConfig, load_config AND
    save_config (the w19 three-site lesson)."""
    cfg = get_default_config()
    cfg.experiment_learned_memory.write_steps = 123
    cfg.experiment_learned_memory.rungs = ["designed", "free_mlp"]
    cfg.experiment_learned_memory.gamma_address = 0.077
    p = tmp_path / "config.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.experiment_learned_memory.write_steps == 123
    assert back.experiment_learned_memory.rungs == ["designed", "free_mlp"]
    assert back.experiment_learned_memory.gamma_address == pytest.approx(0.077)
    assert back.experiment_learned_memory.w19_baseline[
        "codebook_read_K8"
    ] == pytest.approx(0.992)
