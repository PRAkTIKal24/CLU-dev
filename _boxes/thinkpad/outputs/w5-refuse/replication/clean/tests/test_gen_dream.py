"""Conditional dreaming: the clamped-well Langevin rollout and its audit trail.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_separability_POSITIVE_CONTROL_lambda_zero_makes_the_clamp_irrelevant`` —
  with the inter-well coupling switched off the energy is additive, so a *different*
  clamp must leave the relaxed wells **bit-identical**.  That is the exact failure
  mode the whole spoke exists to detect ("the sampler is inert; we are decoding the
  clamp"), reproduced deliberately so the contrast that detects it is demonstrably
  able to fire.
- ``test_decoder_guard_refuses_the_superseded_baseline_recipe`` /
  ``..._refuses_a_hash_mismatch`` — the guard against the ``dec_recipe="baseline"``
  trap (the decoder that FAILED the gate) is seen to fire.
- ``test_mala_samples_the_gibbs_measure_on_an_exactly_gaussian_landscape`` — with
  one mode per well the landscape is *exactly* ``|x - m|^2 / 2``, so the stationary
  law is ``N(m, T I)`` in closed form and the sampler's correctness is checkable
  rather than assertable.
- ``test_summary_reads_the_FIXED_tau_energy_never_the_annealed_one`` — pins the
  trap ``gen-store-grouping`` filed for annealed training losses: with ``tau``
  moving, the endpoints of ``trace['energy']`` are values of two different
  functions.

Everything runs on synthetic tensors; the real-data path is exercised by the
harness runs reported in ``.claude/outputs/gen-dream-sampler.md``.

⛔ Wells are integer-indexed here as everywhere else.
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import GenDreamConfig, get_default_config, load_config, save_config
from chlu.core.gen_bridge import param_hash
from chlu.core.gen_dream import (
    ARMS,
    SAMPLERS,
    VARIANTS,
    WINNING_DECODER_RECIPE,
    DreamLandscape,
    address_requests,
    assert_dream_decoder,
    clamp_from_posterior,
    coupling_bytes,
    dream_rollout,
    energy,
    fit_address_book,
    fit_code_coupling,
    init_state,
    mode_distance_scale,
    occupancy_saturation,
    quantise,
    requests_from_labels,
    resolve_rollout_cfg,
    rollout_read,
    save_trace,
    well_energies,
)
from chlu.core.gen_store import GenStore, assert_no_bypass, read_from_codes, store_bytes


class _Cfg:
    """The rollout knobs, as the experiment passes them."""

    sampler = "mala"
    steps = 30
    step_size = 0.05
    tau_start = 1.0
    tau_end = 0.2
    temp_start = 0.02
    temp_end = 0.02
    uld_dt = 0.05
    uld_gamma = 0.1


def _store(k=4, m=8, dk=6, seed=0):
    wc = tuple(tuple(range(j * dk, (j + 1) * dk)) for j in range(k))
    return GenStore(wc, m, (k * dk,), jax.random.PRNGKey(seed))


def _coupled_codes(n=3000, k=4, m=8, seed=0):
    """Wells 0 and 1 are a perfect copy of a latent; wells 2, 3 are independent."""
    rng = np.random.default_rng(seed)
    lat = rng.integers(0, m, size=n)
    return np.stack(
        [lat if j < 2 else rng.integers(0, m, size=n) for j in range(k)], axis=1
    ), lat


# ---------------------------------------------------------------------------
# the coupling field
# ---------------------------------------------------------------------------


def test_coupling_is_symmetric_and_has_no_self_block():
    codes, _ = _coupled_codes()
    J = fit_code_coupling(codes, 8)
    assert J.shape == (4, 4, 8, 8)
    assert np.allclose(J[0, 1], J[1, 0].T)
    assert np.abs(J[2, 2]).max() == 0.0


def test_coupling_is_large_where_the_codes_co_occur_and_small_where_they_do_not():
    codes, _ = _coupled_codes()
    J = fit_code_coupling(codes, 8)
    linked = float(np.mean(np.diag(J[0, 1])))          # wells 0,1 are a copy
    independent = float(np.abs(J[2, 3]).max())          # wells 2,3 are independent
    assert linked > 1.0
    assert independent < 0.5
    assert linked > 2 * independent


def test_coupling_bytes_are_declared_EXTRA_to_the_store_bytes():
    """⚠ ``gen-baselines`` byte-matches to ``store_bytes``; the coupling is a second
    structure and must be declared separately or that match is wrong."""
    st = _store(k=5, m=64, dk=8)
    cb = coupling_bytes(5, 64)
    assert cb["coupling_bytes"] == 10 * 64 * 64 * 4
    assert cb["coupling_bytes"] != store_bytes(st)["store_bytes"]
    # ⛔ but the read channel is NOT widened
    assert cb["bits_per_read_unchanged"] == store_bytes(st)["bits_per_read"]


# ---------------------------------------------------------------------------
# the landscape
# ---------------------------------------------------------------------------


def test_energy_is_exactly_additive_over_wells_when_the_coupling_is_off():
    """⛔ THE trap this module exists to work around: a separable energy makes
    conditional dreaming provably inert."""
    st = _store()
    codes, _ = _coupled_codes()
    land = DreamLandscape(st, fit_code_coupling(codes, 8), 0.0)
    X = np.asarray(read_from_codes(st, codes[:16])).reshape(16, -1)
    e = energy(land, jnp.asarray(X, jnp.float32), jnp.float32(0.5))
    ew, _ = well_energies(land, jnp.asarray(X, jnp.float32), jnp.float32(0.5))
    assert float(np.max(np.abs(np.asarray(e) - np.asarray(ew)))) == 0.0


def test_the_coupling_makes_the_energy_non_separable():
    st = _store()
    codes, _ = _coupled_codes()
    J = fit_code_coupling(codes, 8)
    land = DreamLandscape(st, J, 1.0)
    X = np.asarray(read_from_codes(st, codes[:16])).reshape(16, -1)
    e = np.asarray(energy(land, jnp.asarray(X, jnp.float32), jnp.float32(0.5)))
    ew = np.asarray(well_energies(land, jnp.asarray(X, jnp.float32), jnp.float32(0.5))[0])
    assert float(np.max(np.abs(e - ew))) > 0.0


def test_energy_casts_a_float64_input_to_the_store_dtype(tmp_path):
    """§7.23: six test modules enable x64 at import and sort before this one; the
    remedy is to cast to the module's own parameter dtype, never to pin float32."""
    st = _store()
    land = DreamLandscape(st, None, 0.0)
    X = np.zeros((4, st.dim), np.float64)
    out = energy(land, X, 0.5)
    assert out.dtype == st.modes[0].dtype


# ---------------------------------------------------------------------------
# the addressing head
# ---------------------------------------------------------------------------


def test_address_book_recovers_a_planted_factor_to_code_map():
    codes, lat = _coupled_codes(m=8)
    labels = np.stack([lat, np.zeros_like(lat)], axis=1)   # factor 0 IS the latent
    book = fit_address_book(codes, labels, [8, 1])
    req = np.stack([np.arange(8), np.zeros(8, np.int64)], axis=1)
    clamp = clamp_from_posterior(address_requests(book, req), 0.5)
    # wells 0 and 1 carry the latent exactly, so the request pins them
    assert np.array_equal(clamp["code"][:, 0], np.arange(8))
    assert np.array_equal(clamp["code"][:, 1], np.arange(8))
    assert clamp["clamped"][:, :2].all()
    # wells 2 and 3 are independent of the factor: the request does not address them
    assert not clamp["clamped"][:, 2:].any()


def test_clamp_threshold_is_the_only_thing_that_decides_clamped_vs_relaxed():
    codes, lat = _coupled_codes()
    labels = np.stack([lat, np.zeros_like(lat)], axis=1)
    book = fit_address_book(codes, labels, [8, 1])
    ls = address_requests(book, np.stack([np.arange(8), np.zeros(8, np.int64)], axis=1))
    loose = clamp_from_posterior(ls, 0.0)
    tight = clamp_from_posterior(ls, 1.01)
    assert loose["clamped"].all()
    assert not tight["clamped"].any()
    assert np.array_equal(loose["code"], tight["code"])   # the CODE never moves


def test_partial_requests_drop_exactly_n_factors_and_are_reproducible():
    Y = np.tile(np.arange(6), (20, 1))
    a = requests_from_labels(Y, "partial", 2, 7)
    b = requests_from_labels(Y, "partial", 2, 7)
    c = requests_from_labels(Y, "partial", 2, 8)
    assert ((a < 0).sum(axis=1) == 2).all()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.array_equal(requests_from_labels(Y, "full", 2, 7), Y)


# ---------------------------------------------------------------------------
# ⭐ the clamp, and the rollout's contract with it
# ---------------------------------------------------------------------------


def _setup(lam=1.0, k=4, m=8, threshold=0.5, seed=0):
    st = _store(k=k, m=m)
    codes, lat = _coupled_codes(k=k, m=m)
    J = fit_code_coupling(codes, m)
    labels = np.stack([lat, np.zeros_like(lat)], axis=1)
    book = fit_address_book(codes, labels, [m, 1])
    req = np.stack(
        [np.arange(m) % m, np.zeros(m, np.int64)], axis=1
    )
    clamp = clamp_from_posterior(address_requests(book, req), threshold)
    prior = np.exp(np.asarray(book.log_prior))
    ini = init_state(st, clamp, prior, "centroid", seed)
    return st, DreamLandscape(st, J, lam), clamp, ini, book, prior


def test_zero_step_rollout_decodes_exactly_its_own_clamp():
    """⭐ THE laundering control: 0 steps must return the state untouched, so the
    caller is decoding the clamp and nothing else."""
    st, land, clamp, ini, _, _ = _setup()
    out = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(0), steps=0)
    assert float(np.max(np.abs(out["x"] - ini["x0"]))) == 0.0
    codes, _ = rollout_read(st, out["x"])
    assert np.array_equal(codes, ini["init_code"])
    assert out["trace"]["accept_prob"].shape[0] == 0


@pytest.mark.parametrize("sampler", list(SAMPLERS))
def test_clamped_coordinates_never_move(sampler):
    st, land, clamp, ini, _, _ = _setup()
    out = dream_rollout(
        land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(1), sampler=sampler
    )
    cc = ini["coord_clamped"]
    assert cc.any(), "the fixture must actually clamp something"
    assert float(np.max(np.abs(out["x"][cc] - ini["x0"][cc]))) == 0.0


def test_separability_POSITIVE_CONTROL_lambda_zero_makes_the_clamp_irrelevant():
    """⛔⭐ The failure mode, reproduced on purpose.

    With ``coupling_strength = 0`` the energy is additive over wells, so the
    relaxed wells cannot feel the clamp: two *different* clamps, same key, must
    leave the relaxed coordinates **bit-identical**.  Switching the coupling on
    must break that.  An inertness test that has never been seen to fire is not
    evidence.
    """
    st, _, clamp, ini, book, prior = _setup()
    codes, _ = _coupled_codes()
    J = fit_code_coupling(codes, 8)
    # a second clamp: the same wells clamped, different modes
    other = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in clamp.items()}
    other["code"] = (other["code"] + 3) % 8
    ini2 = init_state(st, other, prior, "centroid", 0)
    free = ~ini["coord_clamped"]
    assert np.array_equal(ini["coord_clamped"], ini2["coord_clamped"])

    for lam, must_match in ((0.0, True), (1.0, False)):
        land = DreamLandscape(st, J, lam)
        a = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(3))
        b = dream_rollout(land, ini2["x0"], ini2["coord_clamped"], _Cfg(), jax.random.PRNGKey(3))
        same = bool(np.array_equal(a["x"][free], b["x"][free]))
        assert same is must_match, (
            f"coupling_strength={lam}: relaxed coordinates "
            f"{'should' if must_match else 'should not'} be independent of the clamp"
        )


def test_clamp_pull_measures_what_the_landscape_does_to_the_clamp():
    """The audit quantity that says whether the landscape acts on the clamp at all.

    ⚠ Measured, not assumed: with ``M > 1`` modes per well a mode is **not** an
    exact stationary point of the mixture free energy (the other modes pull), so
    the pull at ``lambda = 0`` is small but non-zero.  The exact-zero statement
    holds only where the within-well term genuinely has its minimum at the clamp —
    one mode per well — and that case is checked too, because a "pull" that were
    non-zero *there* would mean the masking is wrong.
    """
    # (a) exact: one mode per well, no coupling -> the clamp IS the minimum
    wc = ((0, 1, 2, 3), (4, 5, 6, 7))
    st1 = GenStore(wc, 1, (8,), jax.random.PRNGKey(0))
    land1 = DreamLandscape(st1, None, 0.0)
    x0 = np.asarray(read_from_codes(st1, np.zeros((8, 2), np.int64))).reshape(8, -1)
    cc = np.zeros_like(x0, bool)
    cc[:, :4] = True
    out = dream_rollout(land1, x0, cc, _Cfg(), jax.random.PRNGKey(5))
    assert float(out["trace"]["summary"]["mean_clamp_pull"]) == pytest.approx(0.0, abs=1e-7)

    # (b) the real landscape: switching the coupling ON strictly increases the pull
    st, _, clamp, ini, _, _ = _setup()
    codes, _ = _coupled_codes()
    J = fit_code_coupling(codes, 8)
    pulls = {}
    for lam in (0.0, 1.0):
        land = DreamLandscape(st, J, lam)
        o = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(5))
        pulls[lam] = float(o["trace"]["summary"]["mean_clamp_pull"])
    assert pulls[0.0] >= 0.0
    assert pulls[1.0] > pulls[0.0]


# ---------------------------------------------------------------------------
# ⭐ the sampler
# ---------------------------------------------------------------------------


def test_mala_samples_the_gibbs_measure_on_an_exactly_gaussian_landscape():
    """⭐ With ONE mode per well the landscape is exactly ``|x - m|^2 / 2``
    (``tau`` cancels in the log-sum-exp of a single term), so the stationary law is
    ``N(m, T I)`` in closed form.  The corrected sampler must reproduce its
    variance; this is the one place the sampler's correctness is *checkable*."""
    wc = ((0, 1, 2, 3, 4, 5, 6, 7),)
    st = GenStore(wc, 1, (8,), jax.random.PRNGKey(0))
    land = DreamLandscape(st, None, 0.0)
    temp = 0.25

    class C(_Cfg):
        steps = 400
        step_size = 0.05
        temp_start = temp
        temp_end = temp
        tau_start = 1.0
        tau_end = 1.0

    m = np.asarray(st.modes[0][0])
    x0 = np.tile(m, (256, 1)).astype(np.float32)
    out = dream_rollout(land, x0, np.zeros_like(x0, bool), C(), jax.random.PRNGKey(0))
    var = float(np.var(out["x"] - m[None, :]))
    assert var == pytest.approx(temp, rel=0.25), f"measured {var} vs target {temp}"
    a = float(out["trace"]["summary"]["mean_accept_prob"])
    assert 0.0 <= a <= 1.0


def test_metropolis_is_the_ONLY_difference_between_mala_and_ula():
    """``ula`` is the same proposal with the accept/reject step removed — one bit."""
    st, land, clamp, ini, _, _ = _setup()
    a = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(9), sampler="mala")
    b = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(9), sampler="ula")
    # step 0 sees the identical state and the identical key, so the identical proposal
    assert np.allclose(a["trace"]["accept_prob"][0], b["trace"]["accept_prob"][0])
    assert np.array_equal(a["trace"]["accepted"][0], b["trace"]["accepted"][0])
    assert a["trace"]["summary"]["metropolis_correction_applied"] is True
    assert b["trace"]["summary"]["metropolis_correction_applied"] is False


def test_the_unadjusted_arms_still_log_the_acceptance_they_are_skipping():
    """⭐ The audit trail must let a referee price the missing correction."""
    st, land, clamp, ini, _, _ = _setup()
    for sampler in ("ula", "uld"):
        out = dream_rollout(
            land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(2), sampler=sampler
        )
        a = out["trace"]["accept_prob"]
        assert a.shape == (_Cfg.steps, ini["x0"].shape[0])
        assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
        assert out["trace"]["summary"]["metropolis_correction_applied"] is False


def test_summary_reads_the_FIXED_tau_energy_never_the_annealed_one():
    """⚠ ``tau`` moves during the rollout, so ``trace['energy']``'s endpoints are
    values of two different functions.  Only ``energy_ref`` may be differenced."""
    st, land, clamp, ini, _, _ = _setup()
    out = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(4))
    tr, s = out["trace"], out["trace"]["summary"]
    assert s["energy_first"] == pytest.approx(float(tr["energy_ref"][0].mean()))
    assert s["energy_final"] == pytest.approx(float(tr["energy_ref"][-1].mean()))
    # the annealed trace really is a different quantity here (tau_start != tau_end)
    assert not np.allclose(tr["energy"][0], tr["energy_ref"][0])


def test_the_chain_leaves_its_initialisation():
    """⛔ A chain that never moves would still decode something plausible on seen
    combinations and tell us nothing."""
    st, land, clamp, ini, _, _ = _setup()
    out = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(6))
    s = out["trace"]["summary"]
    assert s["frac_rollouts_that_moved"] > 0.0
    assert not s["energy_is_flat"]


# ---------------------------------------------------------------------------
# ⛔ the read still goes through the store
# ---------------------------------------------------------------------------


def test_the_dream_reaches_the_decoder_through_the_codes_and_nothing_else():
    """⭐ A dream that handed its continuous state to the decoder would carry D
    floats across the encoder->decoder boundary and destroy R20's declared 30-bit
    design constant."""
    st, land, clamp, ini, _, _ = _setup()
    out = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(7))
    codes, read = rollout_read(st, out["x"])
    assert np.array_equal(codes, quantise(st, out["x"]))
    ref = np.asarray(read_from_codes(st, codes)).reshape(read.shape)
    assert float(np.max(np.abs(read - ref))) == 0.0
    assert assert_no_bypass(st, out["x"][:8])["read_is_pure_codebook_gather"]


# ---------------------------------------------------------------------------
# ⛔ the decoder guard — seen to fire
# ---------------------------------------------------------------------------


def _toy_decoder(seed=0):
    return eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(seed))


def test_decoder_guard_passes_the_winner():
    d = _toy_decoder()
    info = assert_dream_decoder(d, param_hash(d), WINNING_DECODER_RECIPE)
    assert info["checked"] and info["decoder_recipe"] == WINNING_DECODER_RECIPE


def test_decoder_guard_refuses_a_hash_mismatch():
    d = _toy_decoder(0)
    with pytest.raises(AssertionError, match="hash mismatch"):
        assert_dream_decoder(d, param_hash(_toy_decoder(1)), WINNING_DECODER_RECIPE)


def test_decoder_guard_refuses_the_superseded_baseline_recipe():
    """⛔ ``chlu/config.py``'s ``dec_recipe`` defaults to ``"baseline"`` — the
    decoder that FAILED the gate.  A run through it looks entirely normal in the
    logs and is void, so the guard must fire on the NAME as well as the hash."""
    d = _toy_decoder()
    with pytest.raises(AssertionError, match="baseline"):
        assert_dream_decoder(d, param_hash(d), "baseline")


# ---------------------------------------------------------------------------
# the audit trail on disk
# ---------------------------------------------------------------------------


def test_every_audited_quantity_is_written_to_disk(tmp_path):
    st, land, clamp, ini, _, _ = _setup()
    out = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(8))
    p = save_trace(out["trace"], tmp_path / "t.npz")
    z = np.load(p)
    for k in ("energy", "energy_ref", "accept_prob", "accepted", "clamp_pull",
              "tau", "temperature", "code_init", "code_final"):
        assert k in z, f"{k} missing from the audit trail"
    assert z["energy"].shape[0] == _Cfg.steps + 1
    assert z["accept_prob"].shape[0] == _Cfg.steps
    assert z["tau"][0] > z["tau"][-1]     # the anneal is on disk, not in prose


# ---------------------------------------------------------------------------
# config / CLI
# ---------------------------------------------------------------------------


def test_gen_dream_config_defaults_are_the_declared_ones():
    g = get_default_config().gen_dream
    assert g.sampler == "mala"                       # ⭐ Metropolis IS applied
    assert g.decoder_source == "winner"
    assert g.expected_decoder_recipe == "v1_capacity_schedule"   # ⛔ never "baseline"
    assert "zero_step" in g.variants                 # ⭐ the control is not optional
    assert "store_removed" in g.arms and "no_coupling" in g.arms
    assert set(g.arms) <= set(ARMS) and set(g.variants) <= set(VARIANTS)
    assert g.step_size is None                       # tuned to an ACCEPTANCE target


def test_gen_dream_config_round_trips_through_yaml(tmp_path):
    c = get_default_config()
    c.gen_dream.steps = 77
    c.gen_dream.coupling_strength = 0.5
    c.gen_dream.arms = ["clu"]
    p = tmp_path / "c.yaml"
    save_config(c, p)
    back = load_config(p)
    assert back.gen_dream.steps == 77
    assert back.gen_dream.coupling_strength == 0.5
    assert back.gen_dream.arms == ["clu"]
    assert isinstance(back.gen_dream, GenDreamConfig)


def test_cli_exposes_the_binding_contract():
    from chlu.experiments.exp_gen_dream import build_arg_parser

    opts = {a.option_strings[0] for a in build_arg_parser()._actions if a.option_strings}
    for flag in ("--dataset", "--pretext", "--stage", "--arm", "--seed", "--out"):
        assert flag in opts
    assert "--condition" in opts   # ⭐ the wide configuration without an edit


def test_module_refuses_a_stage_that_is_not_its_own(capsys):
    from chlu.experiments.exp_gen_dream import run_experiment_gen_dream

    assert run_experiment_gen_dream(stage="encoder") == {}


# ---------------------------------------------------------------------------
# ⭐ tau / temperature calibration — the defect that froze the first run
# ---------------------------------------------------------------------------


def _far_store(k=2, m=8, dk=200, scale=10.0, seed=0):
    """A store whose modes are as far apart as a real 2048-dim feature cache's."""
    rng = np.random.default_rng(seed)
    wc = tuple(tuple(range(j * dk, (j + 1) * dk)) for j in range(k))
    init = [(scale * rng.normal(size=(m, dk))).astype(np.float32) for _ in range(k)]
    return GenStore(wc, m, (k * dk,), jax.random.PRNGKey(seed), init=init)


def test_absolute_tau_SATURATES_the_occupancy_and_the_calibration_fixes_it():
    """⛔⭐ The defect, reproduced on purpose.

    ``tau`` specified in ABSOLUTE units (the store's own *training* schedule) leaves
    ``exp(-d^2/2 tau^2)`` one-hot on a real-scale cache: the coupling term's
    gradient vanishes and the chain sits in an isolated basin.  Measured on the
    first real run as ``moved = 0.05``; here it is measured directly as the top-1
    occupancy mass, which is the audit column that convicts it.
    """
    st = _far_store()
    land = DreamLandscape(st, None, 0.0)
    x = np.asarray(read_from_codes(st, np.zeros((16, st.n_wells), np.int64))).reshape(16, -1)
    x = x + 0.5 * np.random.default_rng(0).normal(size=x.shape)
    saturated = occupancy_saturation(land, x, 1.0)["mean_max_pi"]
    scale = mode_distance_scale(st)["scale"]
    calibrated = occupancy_saturation(land, x, scale)["mean_max_pi"]
    assert saturated > 0.99, "the fixture must actually reproduce the saturation"
    assert calibrated < 0.5
    assert scale > 10.0   # the store's own geometry, not a number near 1


def test_resolve_rollout_cfg_expresses_tau_in_the_stores_own_geometry():
    st = _far_store()
    land = DreamLandscape(st, None, 0.0)
    x = np.asarray(read_from_codes(st, np.zeros((8, st.n_wells), np.int64))).reshape(8, -1)
    cfg = get_default_config().gen_dream
    rc = resolve_rollout_cfg(cfg, st, land, x)
    scale = mode_distance_scale(st)["scale"]
    assert rc.tau_start == pytest.approx(cfg.tau_start * scale)
    assert rc.tau_end == pytest.approx(cfg.tau_end * scale)
    assert rc.calibration["tau_calibration"] == "mode_distance"
    assert rc.calibration["saturation_at_tau_start"]["mean_max_pi"] < 0.99
    # ⛔ "absolute" must reproduce the pre-fix behaviour bit-for-bit
    cfg.tau_calibration = "absolute"
    cfg.temp_calibration = "absolute"
    rc2 = resolve_rollout_cfg(cfg, st, land, x)
    assert rc2.tau_start == pytest.approx(cfg.tau_start)
    assert rc2.temp_start == pytest.approx(cfg.temp_start)


def test_energy_relative_drop_survives_a_near_zero_per_sample_energy():
    """⚠ A clamped well sits exactly on its mode and contributes ~0, so a per-sample
    ``|E_0|`` denominator explodes (measured -149011.5 on one arm of the first run).
    The statistic is a ratio of MEANS."""
    wc = ((0, 1, 2, 3), (4, 5, 6, 7))
    st = GenStore(wc, 1, (8,), jax.random.PRNGKey(0))
    land = DreamLandscape(st, None, 0.0)
    x0 = np.asarray(read_from_codes(st, np.zeros((16, 2), np.int64))).reshape(16, -1)
    out = dream_rollout(land, x0, np.zeros_like(x0, bool), _Cfg(), jax.random.PRNGKey(0))
    s = out["trace"]["summary"]
    assert np.isfinite(s["energy_relative_drop"])
    assert abs(s["energy_relative_drop"]) <= 1.0 + 1e-6
    assert np.isfinite(s["energy_absolute_change"])
    assert float(np.abs(out["trace"]["energy_ref"][0]).mean()) == pytest.approx(0.0, abs=1e-6)


def test_the_saturation_column_is_in_every_rollout_summary():
    st, land, clamp, ini, _, _ = _setup()
    out = dream_rollout(land, ini["x0"], ini["coord_clamped"], _Cfg(), jax.random.PRNGKey(0))
    s = out["trace"]["summary"]
    for k in ("mean_max_pi_first", "mean_max_pi_final", "occupancy_is_saturated"):
        assert k in s
    assert 0.0 <= s["mean_max_pi_first"] <= 1.0


# ---------------------------------------------------------------------------
# ⛔⭐ the ladder contract: the plan's labels index ENCODER CHANNELS
# ---------------------------------------------------------------------------


def test_the_plan_label_vector_is_only_meaningful_on_its_OWN_encoder(tmp_path):
    """⛔⭐ The defect, reproduced on purpose, and the guard that catches it.

    ``store_plan.json``'s ``labels`` vector was written from ONE seed's encoder.
    Every bridge seed trained its own encoder, so channel ``c`` of seed 1 has no
    relation to channel ``c`` of seed 0: taking the plan's labels verbatim on
    another seed is a size-matched **random** partition wearing the clustered
    grouping's name. ``resolve_grouping`` therefore re-derives the grouping per
    seed and ⭐ **asserts bit-identity with the plan on the seed whose encoder
    wrote it** — which is what proves it is the same procedure.
    """
    from chlu.core.gen_store import (
        channel_correlation,
        dead_channels,
        grouping_labels,
        save_store_plan,
        store_bytes,
    )
    from chlu.experiments.exp_gen_dream import resolve_grouping

    rng = np.random.default_rng(0)
    k, per = 4, 8
    lat = rng.normal(size=(2000, k))
    W = np.zeros((k, k * per))
    for j in range(k):
        W[j, j * per : (j + 1) * per] = 1.0
    X = (lat @ W + 0.05 * rng.normal(size=(2000, k * per))).astype(np.float32)

    cfg = get_default_config()
    cfg.gen_store.n_corr = 2000
    cfg.gen_store.grouping_method = "auto"
    dead = dead_channels(X)
    ref = grouping_labels(
        channel_correlation(X), k, "correlation", 0, "auto",
        dead=dead, min_well_frac=cfg.gen_store.min_well_frac,
    )[0]
    st = _store(k=k, m=4, dk=per)
    plan = save_store_plan(
        tmp_path / "store_plan.json", dataset="shapes3d", pretext="recon", k=k,
        n_modes=4, dim=k * per, feat_shape=(k * per,), labels=ref,
        k_choice={"k": k, "rule": "test"}, grouping_method="auto",
        store_bytes_dict=store_bytes(st), extra={"encoder_param_hash": "ENC0"},
    )

    # (a) on the plan's OWN encoder the re-derivation must reproduce it exactly
    lab, info = resolve_grouping(plan, X, cfg, 0, "ENC0", log=lambda *_: None)
    assert info["reproduces_the_plan_bit_identically"] is True
    assert np.array_equal(lab, np.asarray(plan["labels"], np.int64))

    # (b) on a DIFFERENT encoder it re-derives and says so, loudly
    lab2, info2 = resolve_grouping(plan, X, cfg, 1, "ENC1", log=lambda *_: None)
    assert info2["grouping_source"].startswith("per_seed")
    assert info2["reproduces_the_plan_bit_identically"] is None

    # (c) the defect is still reachable, and it carries its own warning
    cfg.gen_dream.grouping_source = "plan"
    lab3, info3 = resolve_grouping(plan, X, cfg, 1, "ENC1", log=lambda *_: None)
    assert np.array_equal(lab3, np.asarray(plan["labels"], np.int64))
    assert "RANDOM" in info3["warning"]

    # ⛔ and K is NEVER re-derived: it comes from the plan either way
    assert len(set(np.asarray(lab).tolist())) == k
