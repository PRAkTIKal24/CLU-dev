"""The dynamical read: a landscape over the WRITTEN ATOMS, and its guards.

⭐ The load-bearing tests here are the ones that **fail on purpose**:

- ``test_bypass_assertion_FIRES_on_a_variant_bypassed_at_each_stage`` — the
  request-only guarantee is over **launch + settle + read as ONE path**, and it is
  seen to fire at *each* of the three.  ⛔ A guard never seen to fail is decoration,
  which is precisely what this spoke's task file names as "what would prove this
  spoke wrong".
- ``test_signature_leg_FIRES_on_a_stage_that_takes_features`` — the structural leg
  catches a future edit that adds ``X`` to a stage, rather than a numerical accident.
- ``test_atoms_guard_FIRES_when_the_store_is_built_over_CENTRES`` — reproduces the
  design §0 failure the Advisor's constraint forbids (minima = the ``k``-means
  centres) and shows the guard convicting it.
- ``test_split_is_prefix_stable_so_arm_keys_MUST_come_from_fold_in`` — the measured
  JAX landmine: under ``jax_threefry_partitionable`` ``split(k,2)[1]`` **is**
  ``split(k,3)[1]``, so "add one more arm to the split" silently decorrelates
  nothing.  ⭐ Positive control: the property is asserted, then ``fold_in`` is shown
  not to have it.
- ``test_byte_ledger_matches_PREREG_section_3_exactly`` — the ledger is a
  PRE-REGISTERED number (524,288 / 4,194,304 B; **2.00×** budget at ``r = 16`` /
  ``r = 128``), so it is pinned in a test rather than recomputed in prose.
- ``test_request_off_moves_the_trajectory_on_the_SAME_chain`` — dreaming's exact
  lesson: if request-off gives the same trajectory the chain is settling onto a
  mode, not being steered.

Everything runs on synthetic tensors; the real-data path is exercised by the smoke
reported in ``.claude/outputs/gen-dynamical-read.md``.

⛔ Wells are integer-indexed here as everywhere else.
"""

import jax
import numpy as np
import pytest

from chlu.config import GenDynamicalReadConfig, gen_dynamical_read_config, get_default_config
from chlu.core.gen_landscape import (
    ARMS,
    BYPASS_STAGES,
    AtomLandscape,
    AtomStore,
    SettleCfg,
    arm_key,
    assert_atoms_are_train_rows,
    assert_request_only,
    atom_bandwidths,
    build_atom_store,
    check_no_feature_signature,
    convergence_report,
    coupling_byte_ledger,
    decode_width,
    dynamical_read,
    fit_atom_address_book,
    fit_low_rank_coupling,
    landscape_byte_ledger,
    launch_state,
    pairwise_separation,
    read_settled,
    save_trace,
    select_atom_rows,
    settle_chain,
    settled_state_spread,
    store_byte_ledger,
)

K_WELLS, D_WELL, N_TRAIN, N_ATOMS = 3, 6, 900, 16
CARDS = [3, 4, 2, 2]


@pytest.fixture(scope="module")
def fixture():
    """A tiny factored world: 3 wells x 6 channels, 4 factors, 900 train rows."""
    rng = np.random.default_rng(7)
    D = K_WELLS * D_WELL
    wc = tuple(tuple(range(k * D_WELL, (k + 1) * D_WELL)) for k in range(K_WELLS))
    Y = np.stack([rng.integers(0, c, N_TRAIN) for c in CARDS], 1)
    base = rng.normal(size=tuple(CARDS) + (D,)) * 2.0
    Z = (base[tuple(Y[:, j] for j in range(len(CARDS)))]
         + 0.3 * rng.normal(size=(N_TRAIN, D))).astype(np.float32)
    store = build_atom_store(wc, (D,), Z, N_ATOMS, "kmeanspp", 0)
    sigma = atom_bandwidths(store, 1.0, "nn_median")
    book = fit_atom_address_book(store, Z, Y, CARDS, sigma, 1.0)
    coupling = fit_low_rank_coupling(store, Z, rank=2)
    land = AtomLandscape(store, sigma, coupling, 1.0, 1.0)
    req = np.stack(np.meshgrid(*[np.arange(c) for c in CARDS], indexing="ij"),
                   -1).reshape(-1, len(CARDS))
    return {"Z": Z, "Y": Y, "wc": wc, "D": D, "store": store, "sigma": sigma,
            "book": book, "coupling": coupling, "land": land, "requests": req}


def _cfg(steps=12):
    return SettleCfg(steps=steps, step_size=0.2, sigma_scale_start=4.0,
                     sigma_scale_end=1.0, temp_start=0.0, temp_end=0.0)


# ---------------------------------------------------------------------------
# 1. ⛔ The atoms are stored TRAIN DATA POINTS — never the centres
# ---------------------------------------------------------------------------


def test_atoms_are_verbatim_train_rows(fixture):
    out = assert_atoms_are_train_rows(fixture["store"], fixture["Z"])
    assert out["atoms_are_train_rows"] is True
    assert out["max_abs_delta_vs_train_rows"] == 0.0
    assert out["n_distinct_atom_rows"] == N_ATOMS
    rows = np.asarray(fixture["store"].atom_rows, np.int64)
    for w, a in zip(fixture["store"].well_channels, fixture["store"].atoms, strict=True):
        ref = fixture["Z"][np.ix_(rows, np.asarray(w, np.int64))]
        assert np.array_equal(np.asarray(a), ref)


def test_atoms_guard_FIRES_when_the_store_is_built_over_CENTRES(fixture):
    """⛔ The failure the Advisor's constraint forbids, reproduced deliberately.

    A landscape whose minima *are* the ``k``-means centres inherits the codebook
    loss: a continuous read over discrete minima is still a discrete read.  The
    guard must convict it, or it is decoration.
    """
    store = fixture["store"]
    centres = [np.asarray(a).mean(axis=0, keepdims=True).repeat(N_ATOMS, axis=0)
               for a in store.atoms]
    import equinox as eqx

    faked = eqx.tree_at(lambda t: t.atoms, store,
                        [np.asarray(c, np.float32) for c in centres])
    with pytest.raises(AssertionError, match="NOT STORED TRAIN ROWS|collapsed to a centre"):
        assert_atoms_are_train_rows(faked, fixture["Z"])


@pytest.mark.parametrize("selection", ["random", "kmeanspp", "first"])
def test_every_atom_selection_returns_ROWS_never_centroids(fixture, selection):
    rows = select_atom_rows(fixture["Z"], N_ATOMS, selection, 3)
    assert rows.dtype == np.int64
    assert rows.size == N_ATOMS
    st = AtomStore(fixture["wc"], (fixture["D"],), fixture["Z"], rows, selection)
    assert assert_atoms_are_train_rows(st, fixture["Z"])["max_abs_delta_vs_train_rows"] == 0.0


# ---------------------------------------------------------------------------
# 2. ⭐ The byte ledger — a PRE-REGISTERED number, pinned
# ---------------------------------------------------------------------------


def test_byte_ledger_matches_PREREG_section_3_exactly():
    """``PREREG.md`` §3, verbatim: both budgets cost **2.00x** their budget.

    ⛔ Declared before the run, so a drift in ``d_k`` that quietly moved the rank
    would be a silent renegotiation of the ledger.  ``K = 5``, ``D = 2048``.
    """
    # ⭐ seed 0's ACTUAL wells, read off the shipped store in the smoke run
    well_sizes = [303, 305, 573, 501, 366]
    assert sum(well_sizes) == 2048
    a = landscape_byte_ledger(64, well_sizes, 16, 524_288, "A")
    assert a["store"]["store_bytes"] == 524_288
    assert a["coupling"]["coupling_bytes"] == 524_288
    assert a["total_bytes"] == 1_048_576
    assert a["multiple_of_budget"] == pytest.approx(2.0)
    assert a["d3_rule_binding"] is True
    b = landscape_byte_ledger(512, well_sizes, 128, 4_194_304, "B")
    assert b["store"]["store_bytes"] == 4_194_304
    assert b["coupling"]["coupling_bytes"] == 4_194_304
    assert b["total_bytes"] == 8_388_608
    assert b["multiple_of_budget"] == pytest.approx(2.0)
    assert b["d3_rule_binding"] is True
    # the store is byte-matched to the codebook at N = M
    assert store_byte_ledger(64, well_sizes)["store_bytes"] == 64 * 2048 * 4
    # ⭐ and the coupling identity r*(K-1)*D*4 holds however the wells are CUT, so a
    # per-seed drift in d_k cannot silently renegotiate the pre-registered rank
    for other in ([283, 744, 400, 321, 300], [2044, 1, 1, 1, 1], [410, 410, 410, 409, 409]):
        assert sum(other) == 2048
        assert coupling_byte_ledger(16, other)["coupling_bytes"] == 16 * 4 * 2048 * 4
        assert coupling_byte_ledger(128, other)["coupling_bytes"] == 128 * 4 * 2048 * 4


def test_config_byte_ledger_expectations_agree_with_the_computed_ledger():
    c = GenDynamicalReadConfig()
    well_sizes = [303, 305, 573, 501, 366]
    for label in ("A", "B"):
        spec = c.budget_spec(label)
        exp = c.byte_ledger_expectations(label)
        got = landscape_byte_ledger(spec["n_atoms"], well_sizes, spec["rank"],
                                    spec["budget_bytes"], label)
        assert got["store"]["store_bytes"] == exp["store_bytes"]
        assert got["coupling"]["coupling_bytes"] == exp["coupling_bytes"]
        assert got["total_bytes"] == exp["total_bytes"]


def test_the_declared_channel_is_the_stores_float_width_and_NEVER_30_bits(fixture):
    w = decode_width(fixture["store"])
    assert w["declared_channel_floats"] == fixture["D"]
    assert w["declared_channel_bits"] == fixture["D"] * 32
    assert "knn_proto" in w["parity_with"]
    # ⛔ "30 bits" may not be quietly inherited for a continuous state.
    assert "30" not in str(w["declared_channel_bits"]) or fixture["D"] * 32 != 30


def test_config_block_memoises_and_takes_overrides():
    config = get_default_config()
    c = gen_dynamical_read_config(config)
    assert gen_dynamical_read_config(config) is c
    assert c.dec_recipe == "v1_capacity_schedule"   # ⛔ never "baseline"
    c2 = gen_dynamical_read_config(config, steps=7, request_strength=0.0)
    assert c2 is c and c.steps == 7 and c.request_strength == 0.0


# ---------------------------------------------------------------------------
# 3. ⛔⭐ The bypass guarantee — over launch + settle + read AS ONE PATH
# ---------------------------------------------------------------------------


def test_bypass_assertion_HOLDS_on_the_real_path(fixture):
    rng = np.random.default_rng(1)
    n = fixture["requests"].shape[0]
    A = rng.normal(size=(n, fixture["D"])).astype(np.float32)
    B = rng.normal(size=(n, fixture["D"])).astype(np.float32) * 5.0
    out = assert_request_only(fixture["land"], fixture["book"], fixture["requests"],
                              _cfg(), jax.random.PRNGKey(0), A, B)
    assert out["request_only_over_launch_settle_read"] is True
    assert out["max_abs_delta_per_stage"] == {"launch": 0.0, "settle": 0.0, "read": 0.0}
    assert out["feature_tensor_max_abs_difference"] > 0.0


@pytest.mark.parametrize("stage", [s for s in BYPASS_STAGES if s != "none"])
def test_bypass_assertion_FIRES_on_a_variant_bypassed_at_each_stage(fixture, stage):
    """⛔ A guard that has never been seen to fail is DECORATION.

    The bypass is injected at exactly one stage and the assertion must convict the
    path — including the stages **downstream** of the injection, which is what
    "launch + settle + read as ONE path" means.
    """
    rng = np.random.default_rng(2)
    n = fixture["requests"].shape[0]
    A = rng.normal(size=(n, fixture["D"])).astype(np.float32)
    B = rng.normal(size=(n, fixture["D"])).astype(np.float32) * 5.0
    with pytest.raises(AssertionError, match="REQUEST-ONLY GUARANTEE VIOLATED"):
        assert_request_only(fixture["land"], fixture["book"], fixture["requests"],
                            _cfg(), jax.random.PRNGKey(0), A, B,
                            bypass_stage=stage, bypass_alpha=1e-2)


def test_bypass_assertion_refuses_to_pass_VACUOUSLY_on_identical_tensors(fixture):
    rng = np.random.default_rng(3)
    A = rng.normal(size=(fixture["requests"].shape[0], fixture["D"])).astype(np.float32)
    with pytest.raises(AssertionError, match="IDENTICAL"):
        assert_request_only(fixture["land"], fixture["book"], fixture["requests"],
                            _cfg(2), jax.random.PRNGKey(0), A, A.copy())


def test_signature_leg_FIRES_on_a_stage_that_takes_features():
    """⭐ Structural, not numerical: a future edit adding ``X`` is caught at once."""
    assert check_no_feature_signature([launch_state, settle_chain, read_settled])["signature_ok"]

    def leaky_settle(land, x0, log_w, cfg, key, X):     # noqa: N803
        return None

    with pytest.raises(AssertionError, match="REQUEST-ONLY SIGNATURE CHECK FAILED"):
        check_no_feature_signature([leaky_settle])


def test_soft_read_is_NOT_on_the_dynamical_path():
    """⛔ ``gen_store.soft_read(store, X, tau)`` takes FEATURES and would void the
    guarantee (PREREG §7).  ⭐ Checked on the **AST**, not on the source text: the
    module's docstring names the function precisely in order to forbid it, so a
    substring search would convict the warning instead of the defect."""
    import ast
    import inspect

    import chlu.core.gen_landscape as gl

    tree = ast.parse(inspect.getsource(gl))
    called = {
        (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "soft_read" not in called
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "soft_read" not in imported
    assert not hasattr(gl, "soft_read")


# ---------------------------------------------------------------------------
# 4. ⚠ The measured JAX landmine
# ---------------------------------------------------------------------------


def test_split_is_prefix_stable_so_arm_keys_MUST_come_from_fold_in():
    """⚠ Measured on this box (JAX 0.9.0): ``split(k,2)[1]`` **IS** ``split(k,3)[1]``.

    ⭐ **Positive control first**: the landmine is asserted to exist, so this test
    cannot pass by the property having quietly gone away.  Then ``fold_in``-derived
    arm keys are shown to differ, which is the reason the helper exists.
    """
    k = jax.random.PRNGKey(0)
    if not jax.config.jax_threefry_partitionable:
        pytest.skip("jax_threefry_partitionable is OFF: the landmine does not apply")
    assert np.array_equal(np.asarray(jax.random.split(k, 2)[1]),
                          np.asarray(jax.random.split(k, 3)[1])), (
        "the prefix-stability landmine has changed behaviour — re-audit every "
        "harness that derives arm keys by lengthening a split"
    )
    keys = [np.asarray(arm_key(k, a)) for a in ARMS]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.array_equal(keys[i], keys[j]), f"{ARMS[i]} and {ARMS[j]} share a key"


# ---------------------------------------------------------------------------
# 5. ⭐ The request is an ENERGY TERM, and the two controls are real
# ---------------------------------------------------------------------------


def test_request_off_moves_the_trajectory_on_the_SAME_chain(fixture):
    """⛔ Dreaming's lesson: if request-off gives the same trajectory, the chain is
    settling onto a mode, not being steered."""
    land = fixture["land"]
    key = jax.random.PRNGKey(0)
    full = dynamical_read(land, fixture["book"], fixture["requests"], _cfg(),
                          arm_key(key, "full"))
    off = dynamical_read(land.with_strengths(request_strength=0.0), fixture["book"],
                         fixture["requests"], _cfg(), arm_key(key, "full"))
    # SAME chain: identical launch, identical key — the ONLY difference is beta.
    assert np.array_equal(full["x0"], off["x0"])
    assert float(np.max(np.abs(full["x_settled"] - off["x_settled"]))) > 0.0


def test_coupling_off_moves_the_trajectory(fixture):
    land = fixture["land"]
    key = arm_key(jax.random.PRNGKey(0), "full")
    full = dynamical_read(land, fixture["book"], fixture["requests"], _cfg(), key)
    off = dynamical_read(land.with_strengths(coupling_strength=0.0), fixture["book"],
                         fixture["requests"], _cfg(), key)
    assert np.array_equal(full["x0"], off["x0"])
    assert float(np.max(np.abs(full["x_settled"] - off["x_settled"]))) > 0.0


def test_a_landscape_with_no_coupling_is_SEPARABLE_across_wells(fixture):
    """⭐ Positive control for the coupling: with ``lambda = 0`` the energy is additive,
    so perturbing well 0's coordinates cannot move well 1's gradient."""
    from chlu.core.gen_landscape import total_energy

    import jax.numpy as jnp

    land = fixture["land"].with_strengths(coupling_strength=0.0)
    esum = lambda z: jnp.sum(total_energy(land, z, None))  # noqa: E731
    x = np.asarray(fixture["Z"][:8], np.float32)
    g = np.asarray(jax.grad(esum)(x))
    x2 = x.copy()
    x2[:, list(fixture["wc"][0])] += 1.0
    g2 = np.asarray(jax.grad(esum)(x2))
    cols1 = np.asarray(fixture["wc"][1], np.int64)
    assert np.allclose(g[:, cols1], g2[:, cols1], atol=1e-5)


def test_zero_step_returns_the_launch_unchanged(fixture):
    out = settle_chain(fixture["land"], np.asarray(fixture["Z"][:8], np.float32), None,
                       SettleCfg(steps=0), jax.random.PRNGKey(0))
    assert np.array_equal(out["x"], np.asarray(fixture["Z"][:8], np.float32))
    rep = convergence_report(out["trace"])
    assert rep["zero_step"] is True
    assert "ZERO-STEP CONTROL" in rep["verdict"]
    # ⭐ every key is present, so a dynamics table has no ragged row
    for k in ("frac_still_descending", "accepted_fraction", "displacement_step_1",
              "energy_first", "energy_final"):
        assert k in rep


# ---------------------------------------------------------------------------
# 6. ⛔ The dynamics report is binding (PREREG §6)
# ---------------------------------------------------------------------------


def test_the_trace_logs_energy_acceptance_and_displacement_FROM_STEP_1(fixture):
    steps = 12
    out = dynamical_read(fixture["land"], fixture["book"], fixture["requests"],
                         _cfg(steps), jax.random.PRNGKey(0))
    tr = out["trace"]
    n = fixture["requests"].shape[0]
    assert tr["energy"].shape == (steps + 1, n)        # every step, plus the launch
    assert tr["energy_ref"].shape == (steps + 1, n)
    assert tr["accept_prob"].shape == (steps, n)
    assert tr["accepted"].shape == (steps, n)
    assert tr["displacement"].shape == (steps, n)      # ⭐ FROM STEP 1
    assert np.isfinite(tr["displacement"]).all()
    assert float(tr["displacement"][0].mean()) > 0.0


def test_a_chain_still_descending_is_reported_NOT_CONVERGED_never_a_null(fixture):
    """⭐ PREREG §6, binding.  A short chain has not settled and must SAY SO."""
    out = dynamical_read(fixture["land"], fixture["book"], fixture["requests"],
                         _cfg(8), jax.random.PRNGKey(0))
    rep = convergence_report(out["trace"], window=4, tol=1e-3)
    assert rep["frac_still_descending"] > 0.0
    assert "NOT CONVERGED" in rep["verdict"]


def test_convergence_report_says_settled_on_a_chain_that_has_stopped():
    """Positive control for the other side of the rule: a flat trace is settled."""
    n_steps, n = 50, 4
    flat = np.tile(np.linspace(-10.0, -10.0, n_steps + 1)[:, None], (1, n))
    trace = {"energy_ref": flat, "displacement": np.zeros((n_steps, n)),
             "accept_prob": np.ones((n_steps, n)), "accepted": np.ones((n_steps, n), bool)}
    rep = convergence_report(trace, window=10, tol=1e-3)
    assert rep["frac_still_descending"] == 0.0
    assert "NOT CONVERGED" not in rep["verdict"]


def test_the_energy_that_is_DIFFERENCED_is_the_fixed_bandwidth_one(fixture):
    """⚠ With ``sigma`` annealed, ``energy[0]`` and ``energy[-1]`` are values of two
    different functions.  ``energy_ref`` scores every state at the FIXED reference
    bandwidth, and it is the only one the report differences."""
    out = dynamical_read(fixture["land"], fixture["book"], fixture["requests"],
                         _cfg(10), jax.random.PRNGKey(0))
    tr = out["trace"]
    assert not np.allclose(tr["energy"][0], tr["energy_ref"][0])   # sigma really moved
    rep = convergence_report(tr, window=5)
    assert rep["energy_first"] == pytest.approx(float(tr["energy_ref"][0].mean()))
    assert rep["energy_final"] == pytest.approx(float(tr["energy_ref"][-1].mean()))


def test_save_trace_writes_every_step_to_disk(fixture, tmp_path):
    out = dynamical_read(fixture["land"], fixture["book"], fixture["requests"],
                         _cfg(6), jax.random.PRNGKey(0))
    p = save_trace(out["trace"], tmp_path / "t.npz")
    with np.load(p) as z:
        assert z["energy"].shape[0] == 7
        assert z["displacement"].shape[0] == 6
        assert set(("energy", "energy_ref", "accept_prob", "accepted",
                    "displacement")).issubset(set(z.files))


# ---------------------------------------------------------------------------
# 7. Launch, read and the reported spread
# ---------------------------------------------------------------------------


def test_launch_is_a_function_of_the_request_and_the_store_alone(fixture):
    lau = launch_state(fixture["land"], fixture["book"], fixture["requests"])
    assert lau["x0"].shape == (fixture["requests"].shape[0], fixture["D"])
    # every launched coordinate is an ATOM coordinate of its well
    for k, w in enumerate(fixture["store"].well_channels):
        A = np.asarray(fixture["store"].atoms[k])
        block = lau["x0"][:, np.asarray(w, np.int64)]
        assert np.isin(np.round(block, 5), np.round(A, 5)).all()


def test_prior_atom_launch_ignores_the_request(fixture):
    """⭐ The launch-off control: the same ``x0`` for every request."""
    lau = launch_state(fixture["land"], fixture["book"], fixture["requests"],
                       launch_mode="prior_atom")
    assert np.allclose(lau["x0"], lau["x0"][0][None, :])


def test_atom_mixture_read_lies_in_the_convex_hull_of_the_stored_atoms(fixture):
    x = np.asarray(fixture["Z"][:16], np.float32)
    r = read_settled(fixture["land"], x, "atom_mixture").reshape(16, -1)
    for k, w in enumerate(fixture["store"].well_channels):
        A = np.asarray(fixture["store"].atoms[k])
        cols = np.asarray(w, np.int64)
        assert (r[:, cols] >= A.min(axis=0)[None, :] - 1e-4).all()
        assert (r[:, cols] <= A.max(axis=0)[None, :] + 1e-4).all()


def test_settled_spread_reports_a_participation_ratio_and_refuses_the_distinct_trap(fixture):
    out = dynamical_read(fixture["land"], fixture["book"], fixture["requests"],
                         _cfg(6), jax.random.PRNGKey(0))
    sp = settled_state_spread(out["x_settled"], fixture["store"])
    assert sp["participation_ratio"] > 0.0
    # ⛔ the count of distinct CONTINUOUS states is worthless and the module says so
    assert "NOT falsifier 2" in sp["note"]


def test_arms_tuple_names_both_mandatory_controls():
    assert "request_off" in ARMS and "coupling_off" in ARMS and "zero_step" in ARMS


# ---------------------------------------------------------------------------
# 8. ⛔ The flag-ordering hazard — a MEASURED bug, pinned
# ---------------------------------------------------------------------------


def test_quick_must_not_clobber_an_explicit_budget_or_step_flag(monkeypatch):
    """⛔ MEASURED: ``--quick --budget A B --steps 1000`` ran **budget A only**.

    ``apply_quick`` REPLACES the seed / budget / arm / step lists, so an explicit
    flag applied before it is silently discarded — and the log looks entirely
    normal, which is exactly the class of defect that costs a wave.  ⭐ Quick first,
    explicit flags last.
    """
    from chlu.experiments import exp_gen_dynamical_read as ex

    seen = {}

    def _capture(config=None, out_dir=None, log=None):
        seen["cfg"] = ex._cfg(config)
        return {}

    monkeypatch.setattr(ex, "run_experiment_gen_dynamical_read", _capture)
    ex.main(["--quick", "--budgets", "A", "B", "--steps", "1000",
             "--arms", "full", "coupling_off", "--seeds", "1", "2"])
    c = seen["cfg"]
    assert c.budgets == ["A", "B"], "⛔ --quick clobbered the explicit budget list"
    assert c.steps == 1000, "⛔ --quick clobbered the explicit step count"
    assert c.arms == ["full", "coupling_off"]
    assert c.seeds == [1, 2]
    # and --quick alone really does shrink them (so the hazard is real, not theoretical)
    seen.clear()
    ex.main(["--quick"])
    assert seen["cfg"].budgets == ["A"] and seen["cfg"].steps == 25


# ---------------------------------------------------------------------------
# 9. ⚠ The participation ratio is NOT distinctness — the lesson, pinned
# ---------------------------------------------------------------------------


def test_pairwise_separation_catches_a_collapse_that_PR_calls_healthy():
    """⚠ MEASURED: the full arm's ``PR`` fell 5.31 -> 1.08 and read like a collapse,
    while the mean pairwise separation had RISEN 0.248 -> 0.385 and the minimum had
    risen from **0.0** (9.2 % of launch pairs were the same atom tuple) to 0.012.
    ⭐ ``PR`` counts DIRECTIONS; it does not count distinct states.  Both directions
    of the confusion are pinned here."""
    rng = np.random.default_rng(0)
    # (a) one direction, well separated: PR ~ 1 but every state is distinct
    line = (np.linspace(-1, 1, 40)[:, None] * rng.normal(size=(1, 12))).astype(np.float32)
    sp_line = settled_state_spread(line)
    assert sp_line["participation_ratio"] < 1.5
    assert pairwise_separation(line)["min_pairwise_rms_distance"] > 0.0
    assert pairwise_separation(line)["frac_pairs_collapsed"] < 0.2
    # (b) a genuine collapse: every state identical
    flat = np.repeat(rng.normal(size=(1, 12)).astype(np.float32), 40, axis=0)
    col = pairwise_separation(flat)
    assert col["all_states_identical"] is True
    assert col["frac_pairs_collapsed"] == 1.0


def test_separation_is_reported_beside_the_declared_channel_width(fixture):
    out = dynamical_read(fixture["land"], fixture["book"], fixture["requests"],
                         _cfg(6), jax.random.PRNGKey(0))
    sp = settled_state_spread(out["x_settled"], fixture["store"])
    assert "pairwise" in sp
    assert "NOT falsifier 2" in sp["pairwise"]["note"]
    assert "not the participation ratio" in sp["pairwise"]["note"]
