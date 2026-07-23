"""Tests for the POTENTIAL FUNCTION CLASS sweep (exp_potential_class, w21).

These pin the things that, if they silently broke, would turn a negative result
into a fake positive (or vice versa):

  * the **matched-parameter** contract — an unmatched comparison settles nothing,
    so every arm must sit within ``param_tol`` of ``param_target``;
  * :class:`HopfieldPotential` must be *exactly* the modern-Hopfield energy, i.e.
    its stationarity condition must be one step of attention over the codebook.
    If it drifts from that, the arm stops being the Ramsauer test the task asks
    for and becomes "an MLP wearing a hat";
  * the **local atom write must actually be local** — the atoms of every other
    item must come out of a write *bit-identical*. This is the adversarial check
    the task demands before any ``atoms_local`` interference number is reported,
    and a loss check would not catch a leaky mask (``optax.adamw``'s decoupled
    weight decay shrinks frozen parameters if you mask gradients instead of
    updates);
  * the atom amplitude must be able to *move* under the shared write budget —
    the first version of this class used ``softplus(-8)``, whose 3.4e-4 gradient
    made the write a silent no-op and would have reported "atom dictionaries
    fail" as physics;
  * ``learned_family`` must default to ``"mlp"``, so every w20 result is
    reproduced bit-for-bit by the default path.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config, load_config, save_config
from chlu.core.memory_potentials import (
    AtomDictionaryPotential,
    AttentionPotential,
    DesignFreedomPotential,
    HopfieldPotential,
    atom_write_mask_fn,
    designed_payloads,
    ring_sites,
)
from chlu.core.potentials import PotentialMLP
from chlu.experiments.exp_potential_class import (
    ARM_TABLE,
    _c3_law_at_stored_sites,
    build_arm,
    param_match_table,
    write_arm,
)


def _cfg(**kw):
    cfg = get_default_config().experiment_potential_class
    # Small but real: these tests must run in seconds, not minutes.
    cfg.write_steps = 40
    cfg.local_write_steps = 20
    cfg.write_n_perturb = 8
    cfg.n_query_per_item = 6
    cfg.address_steps = 60
    cfg.read_steps = 60
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# The matched-parameter contract
# ---------------------------------------------------------------------------


def test_every_arm_is_within_tolerance_of_the_parameter_target():
    cfg = get_default_config().experiment_potential_class
    table = param_match_table(cfg, K=4)
    assert table["all_within_tolerance"], table["rows"]
    by_arm = {r["arm"]: r for r in table["rows"]}
    # The target IS the w20 baseline's own size, so the baseline must hit it exactly.
    assert by_arm["mlp"]["n_learned_params"] == cfg.param_target
    assert by_arm["designed"]["n_learned_params"] == 0
    for arm in ("hopfield", "attn", "atoms", "atoms_local"):
        assert by_arm[arm]["rel_dev_from_target"] <= 0.05, (arm, by_arm[arm])


def test_arm_table_covers_the_configured_classes():
    cfg = get_default_config().experiment_potential_class
    for arm in cfg.potential_classes:
        assert arm in ARM_TABLE


# ---------------------------------------------------------------------------
# The transformer arm really is the modern-Hopfield energy
# ---------------------------------------------------------------------------


def test_hopfield_potential_is_exactly_the_modern_hopfield_energy():
    """``grad V = 0`` must read ``q = sum_i softmax(beta <q,k_i>)_i k_i``.

    With ``alpha = 1/2`` (the configured default) the gradient of
    ``-(1/b) lse(b <q,k>) + (1/2)|q|^2`` is ``q - sum_i a_i k_i`` exactly, i.e.
    a gradient step IS one step of attention over the memory codebook. This is
    what makes the arm an in-framework test of attention-as-memory rather than a
    generic high-capacity net.
    """
    key = jax.random.PRNGKey(0)
    V = HopfieldPotential(3, 16, key, beta=4.0, confine=0.5, key_init=0.7)
    q = jnp.array([0.3, -0.7, 0.2])
    g = jax.grad(lambda x: V(x))(q)
    a = jax.nn.softmax(V.beta * (V.keys_ @ q) + V.bias)
    attention_step = a @ V.keys_
    assert np.allclose(np.asarray(g), np.asarray(q - attention_step), atol=1e-5)


def test_hopfield_support_is_exponentially_local_in_beta():
    """The claim the w21 report rests on: attention's parameter support decays
    like ``exp(-beta * gap)``, so a SHARPER attention is a MORE LOCAL memory —
    which is why "attention is more global than an MLP" is not automatic."""
    key = jax.random.PRNGKey(1)
    ks = jnp.eye(3)[:2]  # two orthogonal keys, inner-product gap 1
    q = ks[0]

    def sens(beta):
        V = HopfieldPotential(3, 2, key, beta=beta, confine=0.5)
        V = eqx.tree_at(lambda m: m.keys_, V, replace=ks)
        # d V(q) / d k_1 (the key NOT at q) is -softmax_1 * q
        dk = jax.grad(
            lambda kk: HopfieldPotential.__call__(
                eqx.tree_at(lambda m: m.keys_, V, replace=kk), q
            )
        )(ks)
        return float(jnp.linalg.norm(dk[1]))

    s2, s8 = sens(2.0), sens(8.0)
    assert s8 < s2
    # exp(-beta*gap) scaling: the ratio must be O(exp(-6)), not O(1).
    assert s8 / s2 < 1e-2


def test_attention_potential_is_finite_and_bounded_by_its_values():
    key = jax.random.PRNGKey(2)
    V = AttentionPotential(3, 8, key, beta=1.0, confine=0.05, d_head=4)
    q = jnp.array([1.0, -2.0, 0.5])
    v = float(V(q))
    lo, hi = float(V.values.min()), float(V.values.max())
    assert np.isfinite(v)
    assert lo - 1e-5 <= v - 0.05 * float(jnp.sum(q**2)) <= hi + 1e-5


# ---------------------------------------------------------------------------
# The atom dictionary
# ---------------------------------------------------------------------------


def test_atom_dictionary_starts_flat_and_only_digs_wells():
    key = jax.random.PRNGKey(3)
    V = AtomDictionaryPotential(3, 256, key, depth_init=1e-4, confine=0.0)
    q = jnp.array([0.2, -0.1, 0.4])
    assert float(V(q)) <= 0.0  # atoms can only dig
    assert abs(float(V(q))) < 1e-1  # ...and the landscape starts essentially flat


def test_atom_amplitude_gradient_does_not_vanish_at_the_flat_start():
    """Regression: a ``softplus(-8)`` amplitude has a 3.4e-4 gradient, so a
    600-step Adam write can only reach depth 2e-3 and the write silently
    no-ops (measured strict 0.062). The squared amplitude has an O(1) relative
    gradient at the same depth."""
    key = jax.random.PRNGKey(4)
    V = AtomDictionaryPotential(3, 4, key, depth_init=1e-4)
    V = eqx.tree_at(lambda m: m.centers, V, replace=jnp.zeros((4, 3)))
    g = jax.grad(
        lambda amp: AtomDictionaryPotential.__call__(
            eqx.tree_at(lambda m: m.amp, V, replace=amp), jnp.zeros(3)
        )
    )(V.amp)
    assert float(jnp.abs(g).max()) > 1e-3
    assert float(jax.nn.softplus(jnp.asarray(-8.0))) < 1e-3  # the trap, pinned


def test_atom_groups_partition_the_dictionary_exactly_once():
    key = jax.random.PRNGKey(5)
    V = AtomDictionaryPotential(3, 100, key, n_groups=7)
    masks = np.stack([np.asarray(V.group_rows(g)) for g in range(7)])
    assert masks.sum(axis=0).max() == 1  # no atom in two groups
    assert masks.sum() == 100  # every atom owned
    assert not np.asarray(V.group_rows(99)).any()  # out-of-range selects nothing


# ---------------------------------------------------------------------------
# ⭐ The local write must be local (the task's adversarial check)
# ---------------------------------------------------------------------------


def test_local_atom_write_leaves_other_items_bit_identical():
    """Writing item 0 must not move a single parameter of items 1..K-1.

    Bit-level, not loss-level: this is the check the task requires before any
    ``atoms_local`` interference number may be reported. It also pins the reason
    the mask is applied to the UPDATES rather than the gradients — ``adamw``'s
    decoupled weight decay would still shrink frozen rows under a gradient mask.
    """
    cfg = _cfg(n_atoms=64)
    K = 4
    pay = designed_payloads(K, seed=cfg.payload_seed)
    V0 = build_arm("atoms_local", cfg, pay, jax.random.PRNGKey(0), K=K)
    sites = ring_sites(K, f=cfg.f, dim=3, payloads=pay)
    V1, _, _ = write_arm(
        V0, "atoms_local", cfg, sites[:1], jax.random.PRNGKey(1), item_ids=[0]
    )

    m0 = np.asarray(V0.learned.group_rows(0))
    for name in ("centers", "log_width", "amp"):
        a = np.asarray(getattr(V0.learned, name))
        b = np.asarray(getattr(V1.learned, name))
        moved = np.any((a != b).reshape(a.shape[0], -1), axis=1)
        assert not moved[~m0].any(), f"{name}: frozen atoms moved"
        assert moved[m0].any(), f"{name}: the written block did not move at all"


def test_global_atom_write_moves_every_block(monkeypatch):
    """The control for the test above: with the ordinary GLOBAL write, the same
    substrate does move other items' atoms. Locality is a property of the write
    operator, not of the basis alone."""
    cfg = _cfg(n_atoms=64)
    K = 4
    pay = designed_payloads(K, seed=cfg.payload_seed)
    V0 = build_arm("atoms", cfg, pay, jax.random.PRNGKey(0), K=K)
    sites = ring_sites(K, f=cfg.f, dim=3, payloads=pay)
    V1, _, _ = write_arm(V0, "atoms", cfg, sites[:1], jax.random.PRNGKey(1))
    m0 = np.asarray(V0.learned.group_rows(0))
    a = np.asarray(V0.learned.amp)
    b = np.asarray(V1.learned.amp)
    assert np.any(a[~m0] != b[~m0])


def test_local_write_is_rejected_for_a_family_without_atom_blocks():
    cfg = _cfg()
    pay = designed_payloads(2, seed=cfg.payload_seed)
    V = build_arm("mlp", cfg, pay, jax.random.PRNGKey(0), K=2)
    sites = ring_sites(2, f=cfg.f, dim=3, payloads=pay)
    with pytest.raises(TypeError):
        write_arm(V, "atoms_local", cfg, sites, jax.random.PRNGKey(1))


def test_atom_write_mask_fn_zeroes_exactly_the_masked_rows():
    key = jax.random.PRNGKey(6)
    V = DesignFreedomPotential(
        "free_mlp",
        3,
        jnp.zeros(4),
        key,
        learned_family="atoms",
        n_atoms=10,
        atom_groups=2,
    )
    ones = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), V)
    mask = V.learned.group_rows(0)
    out = atom_write_mask_fn(mask)(ones)
    got = np.asarray(out.learned.amp)
    assert np.allclose(got, np.asarray(mask, dtype=float))


# ---------------------------------------------------------------------------
# The C3 drift law must be evaluated where the item actually IS
# ---------------------------------------------------------------------------


def test_c3_law_is_evaluated_at_the_relaxed_fixed_point_not_the_nominal_site():
    """The read's LAUNCH point is not a stationary point of the landscape.

    Retrieval launches on the query manifold ``q2 = 0`` (the anti-decoration
    guard). Evaluating the C3 Hessian there gives a NEGATIVE ``lambda_min`` even
    for the fully DESIGNED landscape — measured -16.76 at the site whose payload
    is -1, and -4.93 averaged over the three probed sites — which would make the
    C3 bound vacuous for a landscape that in fact retrieves at 1.000. At the
    RELAXED fixed point the same landscape is a clean Morse minimum
    (lambda_min = +1.000), and an unwritten landscape must perturb nothing.
    """
    cfg = _cfg(address_steps=400)
    K = 4
    pay = designed_payloads(K, seed=cfg.payload_seed)
    V = build_arm("designed", cfg, pay, jax.random.PRNGKey(0), K=K)
    out = _c3_law_at_stored_sites(V, V, cfg, K)
    assert out["grad_dV_at_stored_sites"] == 0.0
    assert out["measured_drift"] == 0.0
    assert out["n_non_morse_sites"] == 0, "the designed items must be Morse minima"
    assert out["lambda_min_at_stored_sites"] > 0.0
    # ...and the LAUNCH point is not: this is the trap, pinned.
    launch = np.asarray(ring_sites(K, f=cfg.f, dim=3))[: K - 1]  # q2 = 0
    lam = [
        float(
            np.linalg.eigvalsh(
                np.asarray(jax.hessian(lambda q: V(q))(jnp.asarray(z)))
            ).min()
        )
        for z in launch
    ]
    assert min(lam) < 0.0
    assert float(np.mean(lam)) < 0.0


# ---------------------------------------------------------------------------
# w20 compatibility: the default path must be unchanged
# ---------------------------------------------------------------------------


def test_learned_family_defaults_to_mlp_so_w20_is_reproduced():
    key = jax.random.PRNGKey(7)
    for rung in ("skeleton_residual", "sites_learned_payload", "free_mlp"):
        V = DesignFreedomPotential(rung, 3, jnp.zeros(4), key)
        assert V.learned_family == "mlp"
        assert isinstance(V.learned, PotentialMLP)
    # local_rbf IS a family statement and records its own, whatever is passed.
    V = DesignFreedomPotential("local_rbf", 3, jnp.zeros(4), key, learned_family="attn")
    assert V.learned_family == "rbf_atoms"
    V = DesignFreedomPotential("designed", 3, jnp.zeros(4), key)
    assert V.learned_family == "none"
    assert V.learned is None


def test_unknown_learned_family_is_rejected():
    with pytest.raises(ValueError):
        DesignFreedomPotential(
            "free_mlp", 3, jnp.zeros(2), jax.random.PRNGKey(0), learned_family="nope"
        )


def test_every_arm_builds_and_is_finite_and_differentiable():
    cfg = _cfg(n_atoms=32, hopfield_n_mem=16, attn_n_mem=8)
    pay = designed_payloads(4, seed=cfg.payload_seed)
    for arm in ARM_TABLE:
        V = build_arm(arm, cfg, pay, jax.random.PRNGKey(0), K=4)
        for q in (jnp.zeros(3), jnp.array([1.0, -1.0, 0.5]), jnp.full((3,), 5.0)):
            assert np.isfinite(float(V(q))), arm
            g = jax.grad(lambda x, m=V: m(x))(q)
            assert np.all(np.isfinite(np.asarray(g))), arm


def test_potential_class_config_round_trips(tmp_path):
    cfg = get_default_config()
    g = cfg.experiment_potential_class
    g.potential_classes = ["designed", "atoms_local"]
    g.hopfield_beta_sharp = 12.5
    g.atom_depth_init = 2e-4
    g.class_seeds = [7, 8]
    p = tmp_path / "config.yaml"
    save_config(cfg, p)
    back = load_config(p).experiment_potential_class
    assert back.potential_classes == ["designed", "atoms_local"]
    assert back.hopfield_beta_sharp == pytest.approx(12.5)
    assert back.atom_depth_init == pytest.approx(2e-4)
    assert back.class_seeds == [7, 8]
