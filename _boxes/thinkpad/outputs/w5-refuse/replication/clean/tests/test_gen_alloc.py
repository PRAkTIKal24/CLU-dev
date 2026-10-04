"""⭐⭐ D6 — per-well codeword allocation at CONSTANT TOTAL BYTES.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_the_byte_gate_raises_on_a_bigger_store`` — the gate is this spoke's own
  falsifier #1 (*"total bytes are not equal ⇒ it is a bigger store, not an
  allocation"*).  A gate that never raises gates nothing, so it is shown raising.
- ``test_equal_distortion_is_WORSE_than_uniform_on_the_weighted_error`` — ⭐ the
  counter-intuitive fact the whole spoke turns on: equalising ``ε_k`` at a fixed
  budget levels the **easy** wells UP.  ⛔ If this ever passed, the report's central
  claim would be wrong.
- ``test_masking_the_posterior_prevents_an_OUT_OF_RANGE_code`` — ``fit_address_book``
  builds a rectangular ``(K, max M_k)`` table with Laplace smoothing, so an
  unmasked ``argmax`` on a short well CAN emit a code the store does not own.  The
  unmasked failure is exhibited first, then the fix.
- ``test_n_modes_RAISES_on_a_non_uniform_store`` — a single ``M`` would be a wrong
  number, not an approximate one, and it would reach a byte ledger silently.
- ``test_uniform_allocation_through_the_per_well_path_is_BIT_IDENTICAL`` — the
  control arm must be the shipped code path, not a re-implementation of it.
- ``test_law_fit_recovers_a_planted_exponent`` — the fit that acceptance 5 reports
  is shown recovering a slope it was given.

Everything runs on synthetic tensors; the real-data path is exercised by the harness
run reported in ``.claude/outputs/gen-codebook-allocation.md``.
"""

import jax
import numpy as np
import pytest

from chlu.config import (
    GenCodebookAllocConfig,
    gen_codebook_alloc_config,
    get_default_config,
    load_config,
    save_config,
)
from chlu.core.gen_alloc import (
    ALLOCATIONS,
    STORE_BYTE_BUDGET,
    allocate_equal_distortion,
    allocate_mse_optimal,
    allocate_uniform,
    allocation_ledger,
    assert_bytes_not_exceeded,
    build_allocation,
    infeasibility_table,
    law_fit,
    mask_posterior_to_allocation,
    mode_usage_per_well,
    onehot_codes_per_well,
    per_well_distortion,
    predicted_distortion,
    random_codes_per_well,
    variance_weighted_distortion,
    well_spectra,
    well_spectrum,
)
from chlu.core.gen_compose import compose_codes
from chlu.core.gen_store import GenStore, address_codes, init_store, store_bytes
from chlu.core.gen_whiten import per_well_capacity

#: ⭐ The REAL per-seed numbers `gen-variance-equalisation` published — the
#: allocations in the report are derived from exactly these.
SHIPPED = {
    0: dict(PR=[3.117, 3.463, 12.545, 10.498, 4.067], d=[303, 305, 573, 501, 366],
            v=[68.90, 86.48, 118.92, 68.63, 113.22]),
    1: dict(PR=[5.619, 4.007, 3.371, 3.856, 11.998], d=[397, 300, 307, 300, 744],
            v=[105.77, 70.10, 82.49, 102.09, 104.25]),
    2: dict(PR=[3.461, 3.852, 8.453, 13.635, 5.639], d=[283, 283, 390, 616, 476],
            v=[81.65, 63.73, 83.09, 91.64, 171.45]),
}


def _channels(sizes):
    out, i = [], 0
    for s in sizes:
        out.append(tuple(range(i, i + int(s))))
        i += int(s)
    return tuple(out)


# ---------------------------------------------------------------------------
# 1. ⛔ THE BYTE GATE — this spoke's own falsifier #1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_uniform_arm_IS_the_shipped_byte_budget(seed):
    """``M·D·4 = 524,288 B`` — the control is the shipped store's budget, per seed."""
    s = SHIPPED[seed]
    m = allocate_uniform(s["d"], 64)
    led = allocation_ledger("uniform", m, s["d"], s["PR"], s["v"], 64)
    assert sum(s["d"]) == 2048
    assert led["store_bytes"] == STORE_BYTE_BUDGET == 524_288
    assert led["bytes_delta"] == 0
    assert led["bits_per_read"] == pytest.approx(30.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", list(ALLOCATIONS))
def test_no_allocation_ever_exceeds_the_uniform_arms_bytes(seed, name):
    """⛔ A store one byte bigger is a BIGGER STORE, not a re-allocation."""
    s = SHIPPED[seed]
    m = build_allocation(name, s["PR"], s["d"], s["v"], 64)
    led = allocation_ledger(name, m, s["d"], s["PR"], s["v"], 64)
    assert led["store_bytes"] <= STORE_BYTE_BUDGET
    # ...and the shortfall is small enough that it is rounding, not a smaller store
    assert led["bytes_shortfall_frac"] < 0.005


def test_the_byte_gate_raises_on_a_bigger_store():
    """⭐ The gate is shown FAILING on purpose — a gate that never raises gates nothing."""
    s = SHIPPED[0]
    good = allocation_ledger("uniform", allocate_uniform(s["d"], 64), s["d"], s["PR"],
                             s["v"], 64)
    fat = allocation_ledger("fat", [65] * 5, s["d"], s["PR"], s["v"], 64)
    assert fat["store_bytes"] > good["store_bytes"]
    with pytest.raises(AssertionError, match="BIGGER STORE"):
        assert_bytes_not_exceeded([good, fat], "uniform")
    # and it passes when every arm is within budget
    ok = assert_bytes_not_exceeded(
        [good, allocation_ledger("ed", allocate_equal_distortion(s["PR"], s["d"], 64),
                                 s["d"], s["PR"], s["v"], 64)], "uniform")
    assert ok["per_arm"]["ed"]["within_budget"] is True


def test_the_byte_gate_demands_its_control_arm():
    s = SHIPPED[0]
    led = allocation_ledger("ed", allocate_equal_distortion(s["PR"], s["d"], 64),
                            s["d"], s["PR"], s["v"], 64)
    with pytest.raises(ValueError, match="control is mandatory"):
        assert_bytes_not_exceeded([led], "uniform")


# ---------------------------------------------------------------------------
# 2. ⭐ THE ALLOCATIONS THEMSELVES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_equal_distortion_actually_equalises_the_predicted_distortion(seed):
    """The rule D6 names does what it says: ``ε_k`` becomes flat across wells."""
    s = SHIPPED[seed]
    m = allocate_equal_distortion(s["PR"], s["d"], 64)
    e = np.array([predicted_distortion(p, mm) for p, mm in zip(s["PR"], m, strict=True)])
    uni = np.array([predicted_distortion(p, 64) for p in s["PR"]])
    assert e.max() - e.min() < 0.06          # flat
    assert uni.max() - uni.min() > 0.30      # ...and the uniform arm is NOT


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_equal_distortion_is_WORSE_than_uniform_on_the_weighted_error(seed):
    """⭐⭐ **The spoke's central arithmetic fact.**

    Equalising distortion at a FIXED budget does not reduce the total error — it
    moves error **into** the cheap wells.  ⛔ If this test ever passed, the report's
    central claim would be wrong.
    """
    s = SHIPPED[seed]
    uni = variance_weighted_distortion(s["PR"], allocate_uniform(s["d"], 64), s["v"])
    ed = variance_weighted_distortion(
        s["PR"], allocate_equal_distortion(s["PR"], s["d"], 64), s["v"])
    assert ed > uni
    assert ed > 1.5 * uni


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_mse_optimal_allocation_is_never_worse_than_uniform(seed):
    """...and ⭐ it is barely BETTER — which is the pre-registered NULL reading."""
    s = SHIPPED[seed]
    uni = variance_weighted_distortion(s["PR"], allocate_uniform(s["d"], 64), s["v"])
    opt = variance_weighted_distortion(
        s["PR"], allocate_mse_optimal(s["PR"], s["d"], s["v"], 64), s["v"])
    assert opt <= uni
    assert (uni - opt) / uni < 0.05, "a >5% headroom would make re-allocation a lever"


def test_equal_distortion_starves_the_low_PR_wells_and_feeds_the_high_PR_one():
    """The mechanism, not just the total: `M_k` is monotone in `PR_k`."""
    s = SHIPPED[2]
    m = np.asarray(allocate_equal_distortion(s["PR"], s["d"], 64))
    order_pr = np.argsort(np.asarray(s["PR"]))
    assert list(np.argsort(m)) == list(order_pr)
    assert m[int(np.argmax(s["PR"]))] > 64
    assert m[int(np.argmin(s["PR"]))] < 64


def test_the_infeasibility_table_says_the_hard_wells_are_UNREACHABLE():
    """⛔ A partial fix is not a fix: at ε = 0.10 the high-PR well needs millions."""
    s = SHIPPED[2]
    tab = infeasibility_table(s["PR"], s["d"], targets=(0.10,))
    row = tab["targets"][0]
    worst = max(row["modes_needed_per_well"])
    assert worst > 1e6
    assert row["feasible_at_budget"] is False
    assert row["vs_uniform64_budget"] > 100.0


def test_build_allocation_refuses_an_unknown_rule():
    s = SHIPPED[0]
    with pytest.raises(ValueError, match="unknown allocation"):
        build_allocation("whatever", s["PR"], s["d"], s["v"], 64)


# ---------------------------------------------------------------------------
# 3. ⭐ GenStore WITH A PER-WELL ALLOCATION
# ---------------------------------------------------------------------------


def _toy_store(ms=(3, 8, 5), d=(4, 6, 5), seed=0):
    rng = np.random.default_rng(seed)
    wc = _channels(d)
    init = [rng.normal(size=(int(m), int(dd))).astype(np.float32)
            for m, dd in zip(ms, d, strict=True)]
    return GenStore(wc, int(ms[0]), (int(sum(d)),), jax.random.PRNGKey(0), init=init)


def test_n_modes_per_well_and_bits_per_read_on_a_non_uniform_store():
    st = _toy_store((3, 8, 5), (4, 6, 5))
    assert st.n_modes_per_well == (3, 8, 5)
    assert st.bits_per_read == pytest.approx(np.log2(3) + np.log2(8) + np.log2(5))


def test_n_modes_RAISES_on_a_non_uniform_store():
    """⛔ A single ``M`` is a WRONG number here, not an approximate one."""
    st = _toy_store((3, 8, 5), (4, 6, 5))
    with pytest.raises(ValueError, match="NON-UNIFORM"):
        _ = st.n_modes


def test_a_uniform_store_keeps_its_old_n_modes_and_bits_EXACTLY():
    """⛔ The generalisation must not move one digit of the shipped uniform path."""
    st = _toy_store((7, 7, 7), (4, 6, 5))
    assert st.n_modes == 7
    assert st.bits_per_read == pytest.approx(3 * np.log2(7))
    sb = store_bytes(st)
    assert sb["n_modes"] == 7 and sb["uniform_allocation"] is True
    assert sb["bytes_formula"] == "n_modes * sum_k d_k * 4 = n_modes * D * 4"


def test_store_bytes_refuses_to_print_a_single_M_when_there_is_none():
    st = _toy_store((3, 8, 5), (4, 6, 5))
    sb = store_bytes(st)
    assert sb["n_modes"] is None
    assert sb["n_modes_per_well"] == [3, 8, 5]
    assert sb["uniform_allocation"] is False
    assert sb["store_bytes"] == (3 * 4 + 8 * 6 + 5 * 5) * 4


def test_store_bytes_is_sum_M_k_d_k_and_matches_the_ledger():
    s = SHIPPED[0]
    m = allocate_equal_distortion(s["PR"], s["d"], 64)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, sum(s["d"]))).astype(np.float32)
    st = init_store(X, _channels(s["d"]), [int(x) for x in m], (sum(s["d"]),),
                    jax.random.PRNGKey(0), seed=0)
    led = allocation_ledger("ed", m, s["d"], s["PR"], s["v"], 64)
    assert store_bytes(st)["store_bytes"] == led["store_bytes"]
    assert list(st.n_modes_per_well) == [int(x) for x in m]


def test_uniform_allocation_through_the_per_well_path_is_BIT_IDENTICAL():
    """⭐ The control arm is the SHIPPED code path, not a re-implementation."""
    rng = np.random.default_rng(3)
    d = (5, 7, 4)
    X = rng.normal(size=(200, sum(d))).astype(np.float32)
    a = init_store(X, _channels(d), 6, (sum(d),), jax.random.PRNGKey(1), seed=11)
    b = init_store(X, _channels(d), [6, 6, 6], (sum(d),), jax.random.PRNGKey(1), seed=11)
    for ma, mb in zip(a.modes, b.modes, strict=True):
        assert np.array_equal(np.asarray(ma), np.asarray(mb))


def test_init_store_refuses_a_mismatched_allocation():
    rng = np.random.default_rng(0)
    d = (5, 7, 4)
    X = rng.normal(size=(50, sum(d))).astype(np.float32)
    with pytest.raises(ValueError, match="3 wells"):
        init_store(X, _channels(d), [4, 4], (sum(d),), jax.random.PRNGKey(0), seed=0)


def test_address_codes_on_an_allocated_store_stay_in_each_wells_range():
    s = SHIPPED[0]
    m = [int(x) for x in allocate_equal_distortion(s["PR"], s["d"], 64)]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(256, sum(s["d"]))).astype(np.float32)
    st = init_store(X, _channels(s["d"]), m, (sum(s["d"]),), jax.random.PRNGKey(0), seed=0)
    codes = address_codes(st, X)
    assert codes.shape == (256, 5)
    assert (codes.max(axis=0) < np.asarray(m)).all()
    assert (codes.min(axis=0) >= 0).all()


# ---------------------------------------------------------------------------
# 4. ⛔ THE PLUMBING THE SHIPPED HELPERS ASSUME IS SCALAR
# ---------------------------------------------------------------------------


def test_onehot_per_well_uses_the_right_width_per_well():
    codes = np.array([[0, 0, 0], [2, 7, 4]], np.int64)
    oh = onehot_codes_per_well(codes, [3, 8, 5])
    assert oh.shape == (2, 16)
    assert oh.sum() == 6
    assert oh[0, 0] == 1.0 and oh[0, 3] == 1.0 and oh[0, 11] == 1.0
    assert oh[1, 2] == 1.0 and oh[1, 10] == 1.0 and oh[1, 15] == 1.0


def test_onehot_per_well_refuses_an_out_of_range_code():
    with pytest.raises(ValueError, match="well 1: codes out of range"):
        onehot_codes_per_well(np.array([[0, 8, 0]], np.int64), [3, 8, 5])


def test_random_codes_per_well_respects_each_wells_width():
    r = random_codes_per_well(500, [3, 8, 5], seed=7)
    assert r.shape == (500, 3)
    assert (r.max(axis=0) < np.array([3, 8, 5])).all()
    # the control must actually USE its width, or it is not a matched control
    assert (r.max(axis=0) == np.array([2, 7, 4])).all()


def test_masking_the_posterior_prevents_an_OUT_OF_RANGE_code():
    """⛔ Shown FAILING first: the smoothed rectangular address book really can
    ``argmax`` onto a mode a short well does not own."""
    ms = [3, 8, 5]
    P = np.full((4, 3, 8), 1e-6, np.float32)
    P[:, 0, 6] = 1.0          # well 0 owns 3 modes — 6 is not one of them
    P[:, 1, 7] = 1.0
    P[:, 2, 4] = 1.0
    bad = compose_codes(P)["code"]
    assert bad[0, 0] == 6 and bad[0, 0] >= ms[0]        # the failure, exhibited
    good = compose_codes(mask_posterior_to_allocation(P, ms))["code"]
    assert (good.max(axis=0) < np.asarray(ms)).all()
    assert good[0, 1] == 7 and good[0, 2] == 4          # the long wells are untouched


def test_masking_renormalises_and_leaves_a_uniform_allocation_alone():
    P = np.full((3, 2, 4), 0.25, np.float32)
    out = mask_posterior_to_allocation(P, [4, 4])
    assert np.allclose(out, P)
    out2 = mask_posterior_to_allocation(P, [2, 4])
    assert np.allclose(out2.sum(axis=-1), 1.0)
    assert np.allclose(out2[:, 0, :2], 0.5) and np.allclose(out2[:, 0, 2:], 0.0)


def test_mask_refuses_a_posterior_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"must be \(N, K=2, M\)"):
        mask_posterior_to_allocation(np.zeros((3, 5, 4), np.float32), [2, 2])


def test_mode_usage_per_well_reports_each_wells_own_M():
    codes = np.array([[0, 0], [1, 0], [2, 0], [2, 1]], np.int64)
    u = mode_usage_per_well(codes, [3, 8])
    assert u["per_well"][0]["used_modes"] == 3 and u["per_well"][0]["n_modes"] == 3
    assert u["per_well"][1]["used_modes"] == 2 and u["per_well"][1]["n_modes"] == 8
    assert u["total_allocated_modes"] == 11
    assert u["per_well"][1]["max_bits"] == pytest.approx(3.0)


def test_per_well_capacity_accepts_a_sequence_and_is_UNCHANGED_for_a_scalar():
    """⛔ The generalisation must not move the shipped scalar result by one digit."""
    names = ["a", "b", "c"]
    cards = [10, 10, 8]
    mi = np.array([[3.0, 0.1, 0.1], [0.1, 3.0, 3.0]])
    H = [3.32, 3.32, 3.0]
    a = per_well_capacity(mi, H, cards, names, 64)
    b = per_well_capacity(mi, H, cards, names, [64, 64])
    assert a["per_well"] == b["per_well"]
    assert a["aggregate_bits_available"] == pytest.approx(b["aggregate_bits_available"])
    assert a["n_modes"] == 64 and b["n_modes"] == 64
    c = per_well_capacity(mi, H, cards, names, [4, 186])
    assert c["n_modes"] is None and c["n_modes_per_well"] == [4, 186]
    assert c["per_well"][0]["bits_available"] == pytest.approx(2.0)
    assert c["aggregate_bits_available"] == pytest.approx(2.0 + np.log2(186))


# ---------------------------------------------------------------------------
# 5. ⭐ THE LAW, AND ITS FIT
# ---------------------------------------------------------------------------


def test_participation_ratio_is_label_free_and_correct_on_a_planted_spectrum():
    rng = np.random.default_rng(0)
    n, d = 20000, 40
    lam = np.zeros(d)
    lam[:4] = 1.0                                  # exactly 4 equal directions
    Z = rng.normal(size=(n, d)) * np.sqrt(lam)
    s = well_spectrum(Z, list(range(d)))
    assert s["participation_ratio"] == pytest.approx(4.0, rel=0.15)
    assert s["d_k"] == d and s["label_free"] is True


def test_well_spectra_walks_the_partition_in_store_order():
    rng = np.random.default_rng(0)
    d = (5, 9, 3)
    X = rng.normal(size=(300, sum(d)))
    sp = well_spectra(X, _channels(d))
    assert [s["d_k"] for s in sp] == list(d)


def test_more_codewords_lower_the_MEASURED_distortion():
    """The law's direction, measured through the real store path.

    ⚠ ``init_store`` is k-means++ **seeding only** (no Lloyd steps), so at ``M = 2``
    on isotropic data the relative distortion is legitimately **above 1** — two
    sampled data points are a worse predictor than the mean.  ⛔ That is a property
    of the seeding, not a defect, and it is asserted rather than hidden: only the
    *direction* and the large-``M`` value are the law's content here.
    """
    rng = np.random.default_rng(5)
    d = (6, 6)
    X = rng.normal(size=(2000, sum(d))).astype(np.float32)
    seen, prev = [], None
    for m in (2, 8, 32, 128):
        st = init_store(X, _channels(d), m, (sum(d),), jax.random.PRNGKey(0), seed=2)
        dd = per_well_distortion(st, X, address_codes(st, X))
        cur = dd["variance_weighted_relative_distortion"]
        assert cur > 0.0
        if prev is not None:
            assert cur < prev
        prev = cur
        seen.append(cur)
    assert seen[0] > 1.0, "k-means++ seeding at M=2 is worse than the mean"
    assert seen[-1] < 0.35


def test_per_well_distortion_is_zero_when_the_codebook_holds_the_data():
    """Two wells, two atoms each, and the atoms ARE the data."""
    atoms = [np.array([[0.0, 0.0], [5.0, 5.0]], np.float32),
             np.array([[1.0], [-1.0]], np.float32)]
    wc = ((0, 1), (2,))
    st = GenStore(wc, 2, (3,), jax.random.PRNGKey(0), init=atoms)
    X = np.array([[0, 0, 1], [5, 5, -1], [0, 0, -1], [5, 5, 1]], np.float32)
    dd = per_well_distortion(st, X, address_codes(st, X))
    assert dd["variance_weighted_relative_distortion"] == pytest.approx(0.0, abs=1e-6)


def test_law_fit_recovers_a_planted_exponent():
    """⭐ The fit acceptance 5 reports is shown recovering a slope it was given."""
    p = np.geomspace(1e-3, 0.9, 40)
    m = 1.3 * p ** 0.8
    f = law_fit(p, m)
    assert f["slope"] == pytest.approx(0.8, abs=0.02)
    assert f["multiplicative_offset"] == pytest.approx(1.3, rel=0.05)
    assert f["pearson_r_log"] == pytest.approx(1.0, abs=1e-6)
    assert f["n_points"] == 40


def test_law_fit_drops_non_finite_points_rather_than_propagating_nan():
    p = np.array([0.1, 0.2, np.nan, 0.0, 0.4])
    m = np.array([0.1, 0.2, 0.3, 0.4, np.inf])
    f = law_fit(p, m)
    assert f["n_points"] == 2


def test_predicted_distortion_is_the_law_verbatim():
    assert predicted_distortion(13.63, 64) == pytest.approx(64.0 ** (-2.0 / 13.63))
    assert predicted_distortion(3.12, 1) == pytest.approx(1.0)
    assert np.isnan(predicted_distortion(0.0, 64))


# ---------------------------------------------------------------------------
# 6. Config block — ⛔ append-only, no CHLUConfig field, round-trips
# ---------------------------------------------------------------------------


def test_config_block_defaults_and_the_decoder_trap():
    c = GenCodebookAllocConfig()
    assert c.arms == ["uniform", "equal_distortion", "mse_optimal"]
    assert c.m_uniform == 64 and c.m_min == 2
    # ⛔ never the recipe that FAILED the gate
    assert c.dec_recipe == "v1_capacity_schedule"
    assert c.assert_bytes_within_budget is True
    assert c.per_arm_random_control is True
    assert c.seeds == [0, 1, 2]


def test_config_accessor_memoises_and_is_not_a_CHLUConfig_field():
    cfg = get_default_config()
    a = gen_codebook_alloc_config(cfg)
    b = gen_codebook_alloc_config(cfg)
    assert a is b
    assert "gen_codebook_alloc" not in {f.name for f in cfg.__dataclass_fields__.values()}


def test_the_new_block_does_not_disturb_config_round_trip(tmp_path):
    cfg = get_default_config()
    gen_codebook_alloc_config(cfg).m_uniform = 32
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    # ⛔ the block is deliberately NOT serialised (no house wiring site); the shipped
    # fields must still round-trip untouched.
    assert back.gen_store.n_modes == cfg.gen_store.n_modes
    assert gen_codebook_alloc_config(back).m_uniform == 64


def test_quick_does_not_shrink_M_or_the_byte_budget():
    """⭐ ``K·log2(M) = 30`` is R20's design constant — the smoke runs the REAL channel."""
    from chlu.experiments.exp_gen_codebook_alloc import apply_quick

    cfg = get_default_config()
    c = gen_codebook_alloc_config(cfg)
    apply_quick(cfg)
    assert c.m_uniform == 64
    assert c.probe_steps == c.quick_probe_steps
    assert cfg.gen_store.train_steps == c.quick_store_train_steps


# ---------------------------------------------------------------------------
# 7. ⛔ THE VERDICT BRANCHES — the bug PREREG_ADDENDUM_1 declares, pinned
# ---------------------------------------------------------------------------


def _summary_stub(d_total, d_distinct, d_scale=0.0, d_orient=0.0,
                  spread=0.05, dspread=1.0, bytes_delta=(-240, -644, -792)):
    """A minimal `summarize` output with exactly the fields `_verdict` reads."""
    def s(m, sd):
        return {"mean": m, "std": sd, "n_seeds": 3, "ddof": 0}

    def arm(tot, dist, sc, orr):
        return {
            "per_factor": {"scale": {"retention_code": s(sc, spread)},
                           "orientation": {"retention_code": s(orr, spread)}},
            "total_retained": s(tot, spread),
            "distinct_request_side": s(dist, dspread),
        }
    return {
        "by_eval_set": {
            "SPREAD": {"uniform": arm(3.678, 0, 0.2245, 0.2425),
                       "x": arm(3.678 + d_total, 0, 0.2245 + d_scale, 0.2425 + d_orient)},
            "CORNER_B": {"uniform": arm(0, 9.0, 0, 0),
                         "x": arm(0, 9.0 + d_distinct, 0, 0)},
        },
        "allocation": {"x": {"bytes_delta_vs_uniform_per_seed": list(bytes_delta)}},
    }


@pytest.mark.parametrize(
    "d_total,d_distinct,d_scale,d_orient,label,passed",
    [
        (+0.50, +0.0, 0.0, 0.0, "PASS", "YES"),
        (-0.50, +0.0, +0.20, +0.20, "TRADE", "NO"),
        # ⭐ the branch the first implementation MISLABELLED as LOSS
        (-0.012, +5.33, 0.0, 0.0, "SPLIT", "NO"),
        (-0.012, +0.50, 0.0, 0.0, "NULL", "NO"),
        (-0.50, -5.00, 0.0, 0.0, "LOSS", "NO"),
    ],
)
def test_every_verdict_branch_is_reachable_and_correctly_named(
    d_total, d_distinct, d_scale, d_orient, label, passed
):
    from chlu.experiments.exp_gen_codebook_alloc import _verdict

    v = _verdict(_summary_stub(d_total, d_distinct, d_scale, d_orient), "x")
    assert v["reading_label"] == label
    assert v["pre_registered_pass"] == passed


def test_a_flat_total_with_a_RISEN_distinct_count_is_NOT_called_a_loss():
    """⛔ The measured `mse_optimal` case: nothing fell, so 'LOSS' is factually false."""
    from chlu.experiments.exp_gen_codebook_alloc import _verdict

    v = _verdict(_summary_stub(-0.012, +5.33), "x")
    assert "LOSS" not in v["reading"]
    assert v["distinct_of_corner_b"]["rose_beyond_spread"] is True
    assert v["distinct_of_corner_b"]["fell_beyond_spread"] is False
    assert v["pre_registered_pass"] == "NO"   # ⛔ still not a pass


def test_a_bigger_store_can_never_pass_however_good_it_looks():
    from chlu.experiments.exp_gen_codebook_alloc import _verdict

    v = _verdict(_summary_stub(+0.50, +5.0, bytes_delta=(+4, 0, 0)), "x")
    assert v["bytes_within_uniform_every_seed"] is False
    assert v["pre_registered_pass"] == "NO"


def test_within_well_M_exponent_recovers_a_planted_exponent():
    """⚠ The POST-HOC instrument (PREREG_ADDENDUM_1 §2), shown recovering a slope."""
    from chlu.experiments.exp_gen_codebook_alloc import within_well_m_exponent

    pts = [{"well": 0, "arm": a, "n_modes": m, "measured_eps": 0.5 * m ** -0.26,
            "participation_ratio": 4.0}
           for a, m in zip(("uniform", "equal_distortion", "mse_optimal"), (4, 64, 160),
                           strict=True)]
    rec = [{"seed": 0, "spectrum": {"per_well": [{"d_k": 10}]}, "law": {"points": pts}}]
    out = within_well_m_exponent(rec)
    assert out["n_points"] == 1
    assert out["per_well"][0]["fitted_exponent"] == pytest.approx(-0.26, abs=1e-6)
    assert out["per_well"][0]["law_exponent"] == pytest.approx(-0.5)
    assert out["status"].startswith("⚠ POST-HOC")
