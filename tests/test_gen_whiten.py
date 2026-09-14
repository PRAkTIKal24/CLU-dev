"""Per-well feature equalisation: the transform, the premise test, the byte ledger.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_premise_test_kills_a_dataset_where_the_signal_is_HIGH_variance`` — the
  Leg-1 premise test is the gate on the whole experiment.  On a synthetic well whose
  discriminative signal sits in the **leading** principal axis it must return
  ``DEAD``; on one where it sits in a **trailing** axis it must return ``HOLDS``.
  A gate that returns ``HOLDS`` on both gates nothing.
- ``test_kmeans_recovers_a_low_variance_factor_ONLY_after_whitening`` — ⭐ **the
  mechanism itself**, end to end and falsifiable: on a block whose class structure
  lives in a low-variance direction, k-means on the raw block is at chance on that
  factor and k-means on the whitened block is not.  ⛔ If this failed, whitening
  would not be doing what the hypothesis says it does.
- ``test_pca_and_zca_whitening_induce_the_SAME_clustering`` — the stated reason
  PCA is used rather than ZCA.
- ``test_whitening_does_not_move_the_partition`` / ``..._is_block_diagonal`` — the
  ruled order (R43.1) enforced structurally, not by discipline.
- ``test_byte_ledger_reproduces_the_hub_arithmetic`` — the 3,600,672 B / 16,384 B
  figures the report is read against, re-derived from the real well sizes.
- ``test_unwhiten_store_is_the_fit_time_only_deployment`` — the claim "0 deployed
  bytes on the request-side path" made into an assertion.

Everything runs on synthetic tensors; the real-data path is exercised by the harness
run reported in ``.claude/outputs/gen-variance-equalisation.md``.
"""

import jax
import numpy as np
import pytest

from chlu.config import (
    GenVarianceEqConfig,
    gen_variance_eq_config,
    get_default_config,
    load_config,
    save_config,
)
from chlu.core.gen_store import GenStore, address_codes
from chlu.core.gen_whiten import (
    STORE_BYTE_BUDGET,
    TRANSFORMS,
    WellTransform,
    assert_deployment_equivalence,
    assert_grouping_is_unwhitened,
    between_class_variance,
    capacity_reading,
    per_well_capacity,
    premise_verdict,
    transform_byte_ledger,
    unwhiten_store,
    well_variance_spectrum,
)

#: ⭐ The REAL seed-0 well sizes the byte ledger in the report is read against.
SHIPPED_WELL_SIZES = [303, 305, 573, 501, 366]
NAMES = ["floor_hue", "wall_hue", "scale", "orientation"]
CARDS = [4, 4, 4, 4]


def _wells(sizes, seed=0):
    """A random (but fixed) partition of ``sum(sizes)`` channels into wells."""
    d = int(sum(sizes))
    perm = np.random.default_rng(seed).permutation(d)
    out, i = [], 0
    for s in sizes:
        out.append(tuple(int(x) for x in np.sort(perm[i : i + s])))
        i += s
    return tuple(out)


def _block(n=4000, d=12, seed=0, signal_rank="low", gap=3.0):
    """A well whose class means sit in a LEADING or a TRAILING principal axis.

    The spectrum is fixed by construction (``sd_j = 1/(j+1)``) and one axis is then
    given a class-mean offset **proportional to that axis's own sd**, so the planted
    signal is the same *fraction* of its axis either way and only its **rank** in the
    spectrum differs.  A random rotation is applied so the analysis has to recover
    the axes rather than being handed them.
    """
    rng = np.random.default_rng(seed)
    sd = 1.0 / (np.arange(d) + 1.0)
    Z = rng.normal(size=(n, d)) * sd[None, :]
    y = rng.integers(0, 4, size=n)
    axis = (d - 1) if signal_rank == "low" else 0
    Z[:, axis] += gap * sd[axis] * y
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return (Z @ Q.T).astype(np.float32), y.astype(np.int64), axis


# ---------------------------------------------------------------------------
# 1. The transform is invertible, block-diagonal and partition-preserving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", TRANSFORMS)
def test_round_trip_is_exact_to_float32(kind):
    wc = _wells([7, 9, 5])
    X = np.random.default_rng(1).normal(size=(400, 21)).astype(np.float32) * 3.0
    tr = WellTransform.fit(X, wc, kind=kind)
    rt = tr.round_trip(X)
    assert rt["max_rel_error"] < 1e-3, rt


def test_identity_arm_is_the_LITERAL_identity():
    """⭐ The control must not silently centre the features: the shipped store path
    feeds raw features to ``init_store``, so the control has to as well."""
    wc = _wells([4, 6])
    X = np.random.default_rng(2).normal(size=(50, 10)).astype(np.float32) + 5.0
    tr = WellTransform.fit(X, wc, kind="identity")
    assert np.array_equal(tr.forward(X), X)
    assert np.array_equal(tr.inverse(X), X)


def test_whitening_is_BLOCK_DIAGONAL_and_never_mixes_two_wells():
    """⛔ The wells partition the coordinates, so the map cannot move information
    across the (already exonerated) partition."""
    wc = _wells([5, 5], seed=3)
    X = np.random.default_rng(3).normal(size=(300, 10)).astype(np.float32)
    tr = WellTransform.fit(X, wc, kind="pca_whiten")
    # perturb one channel of well 0 only; nothing in well 1 may move
    x0 = np.zeros((1, 10), np.float32)
    x1 = x0.copy()
    x1[0, wc[0][0]] = 1.0
    d = np.abs(tr.forward(x1) - tr.forward(x0))[0]
    assert d[list(wc[1])].max() == 0.0
    assert d[list(wc[0])].max() > 0.0


def test_whitening_does_not_move_the_partition():
    """R43.1 made an assertion: a transform fitted on a DIFFERENT grouping is refused."""
    wc = _wells([4, 6], seed=4)
    other = _wells([6, 4], seed=5)
    X = np.random.default_rng(4).normal(size=(100, 10)).astype(np.float32)
    tr = WellTransform.fit(X, wc, kind="pca_whiten")
    assert assert_grouping_is_unwhitened(tr.well_channels, wc)["asserted"]
    with pytest.raises(AssertionError, match="NOT the unwhitened partition"):
        assert_grouping_is_unwhitened(tr.well_channels, other)


def test_pca_whitening_actually_whitens_each_well():
    wc = _wells([6, 8], seed=6)
    rng = np.random.default_rng(6)
    A = rng.normal(size=(14, 14))
    X = (rng.normal(size=(4000, 14)) @ A).astype(np.float32)
    # ⛔ scale_match is a uniform per-well scalar, so unit covariance is the
    # property of the UNMATCHED map; the matched one is that map times alpha.
    Z = WellTransform.fit(X, wc, kind="pca_whiten", scale_match=False).forward(X)
    for w in wc:
        C = np.cov(np.asarray(Z[:, list(w)], np.float64).T)
        assert np.abs(C - np.eye(len(w))).max() < 0.1


# ---------------------------------------------------------------------------
# 2. ⭐ LEG 1 — the premise test must be able to say DEAD
# ---------------------------------------------------------------------------


def test_variance_spectrum_locates_a_planted_signal_at_the_right_rank():
    Zl, yl, axl = _block(signal_rank="low", seed=7)
    Zh, yh, axh = _block(signal_rank="high", seed=7)
    d = Zl.shape[1]
    Y = np.stack([yl, yl, yl, yl], 1)
    lo = well_variance_spectrum(Zl, range(d), Y, NAMES, CARDS)["per_factor"]["scale"]
    Yh = np.stack([yh, yh, yh, yh], 1)
    hi = well_variance_spectrum(Zh, range(d), Yh, NAMES, CARDS)["per_factor"]["scale"]
    assert lo["rank_frac_median"] > hi["rank_frac_median"]
    assert hi["frac_signal_in_top10pct"] > lo["frac_signal_in_top10pct"]
    # the LOW-variance planted signal costs a tiny share of the well's variance
    assert lo["variance_explained"] < hi["variance_explained"]


def test_premise_test_kills_a_dataset_where_the_signal_is_HIGH_variance():
    """⛔⭐ The gate on the whole experiment, shown FAILING before it is trusted.

    Same spectrum, same planted signal strength, ⛔ **only the rank differs**: the
    probe factors go into the trailing axes (premise HOLDS) or into axes 2 and 3,
    right beside the reference factors (premise DEAD).  A gate that returned HOLDS on
    both would gate nothing.
    """
    d, n = 24, 6000

    def _spectrum(sig_rank_for_probe):
        rng = np.random.default_rng(11)
        sd = 1.0 / (np.arange(d) + 1.0)
        Z = rng.normal(size=(n, d)) * sd[None, :]
        y = rng.integers(0, 4, size=(n, 4))
        # the reference factors always sit in the LEADING axes
        Z[:, 0] += 4.0 * sd[0] * y[:, 0]
        Z[:, 1] += 4.0 * sd[1] * y[:, 1]
        ax = (d - 1, d - 2) if sig_rank_for_probe == "low" else (2, 3)
        Z[:, ax[0]] += 4.0 * sd[ax[0]] * y[:, 2]
        Z[:, ax[1]] += 4.0 * sd[ax[1]] * y[:, 3]
        return [well_variance_spectrum(Z.astype(np.float32), range(d), y, NAMES, CARDS)]

    holds = premise_verdict(_spectrum("low"), NAMES)
    dead = premise_verdict(_spectrum("high"), NAMES)
    assert holds["direction_test_passed"] is True, holds
    assert holds["verdict"] in ("HOLDS", "PARTIAL"), holds
    assert dead["verdict"] == "DEAD", dead
    assert dead["stop_after_leg1"] is True


def test_premise_verdict_reports_PARTIAL_rather_than_rounding():
    """⛔ PARTIAL is its own word: the direction test passes, the share test does not."""
    d = 20
    rng = np.random.default_rng(12)
    sd = np.ones(d)  # flat spectrum -> shares are comparable
    Z = rng.normal(size=(1500, d)) * sd[None, :]
    y = rng.integers(0, 4, size=(1500, 4))
    Z[:, 0] += 1.0 * y[:, 0]
    Z[:, 1] += 1.0 * y[:, 1]
    Z[:, d - 1] += 3.0 * y[:, 2]
    Z[:, d - 2] += 3.0 * y[:, 3]
    v = premise_verdict([well_variance_spectrum(Z.astype(np.float32), range(d), y, NAMES, CARDS)],
                        NAMES)
    assert v["verdict"] in ("PARTIAL", "HOLDS", "DEAD")
    assert v["verdict"] == ("PARTIAL" if (v["direction_test_passed"]
                                          and not v["share_test_passed"])
                            else ("HOLDS" if v["direction_test_passed"] else "DEAD"))


def test_a_one_valued_factor_is_stamped_not_emitted_bare():
    """R31's fourth strike, enforced structurally."""
    Z, y, _ = _block(seed=8)
    Y = np.stack([y, y, y, np.zeros_like(y)], 1)
    row = well_variance_spectrum(Z, range(Z.shape[1]), Y, NAMES, CARDS)["per_factor"]
    assert row["orientation"]["degenerate"] is True
    assert "NOT APPLICABLE" in row["orientation"]["status"]
    assert row["scale"]["degenerate"] is False


def test_debias_removes_the_null_signal_from_the_LEADING_axes():
    """⭐⭐ Without the null subtraction, "the signal is in the top PCs" is a property
    of the ESTIMATOR, not of the features — shown, not asserted."""
    from chlu.core.gen_whiten import debias_between_class_variance

    rng = np.random.default_rng(61)
    n, d = 500, 20
    sd = 1.0 / (np.arange(d) + 1.0)
    Z = (rng.normal(size=(n, d)) * sd[None, :]).astype(np.float32)
    y = rng.integers(0, 4, size=(n, 4))          # ⛔ labels carry NOTHING here
    out = well_variance_spectrum(Z, range(d), y, NAMES, CARDS)["per_factor"]["scale"]
    # the RAW statistic puts the phantom signal in the leading axes...
    assert out["rank_frac_median_raw"] < 0.25
    # ...and the raw share is the predicted null bias, (V-1)/N
    assert out["variance_explained_raw"] == pytest.approx(3 / n, rel=0.6)
    # ⭐ the debiased share is an order of magnitude smaller
    assert out["variance_explained"] < 0.35 * out["variance_explained_raw"]
    b = debias_between_class_variance(np.array([1.0, 1.0]), np.array([100.0, 1.0]), 100, 5)
    assert b[0] == 0.0 and b[1] == pytest.approx(0.96)


def test_between_class_variance_is_zero_without_classes():
    T = np.random.default_rng(9).normal(size=(200, 5))
    assert float(between_class_variance(T, np.zeros(200, np.int64)).max()) == 0.0


# ---------------------------------------------------------------------------
# 3. ⭐⭐ THE MECHANISM, end to end and falsifiable
# ---------------------------------------------------------------------------


def _kmeans_codes(Z, m, seed=0, iters=60, n_init=8):
    """Plain Lloyd k-means with restarts — ⛔ deliberately NOT the store's optimiser,
    so these tests measure the *metric*, not the trainer.  ⚠ Restarts matter: a
    single random init lands in the nuisance-direction local optimum often enough to
    make the mechanism test flaky, which would be a property of Lloyd, not of
    whitening."""
    rng = np.random.default_rng(seed)
    best, best_in = None, np.inf
    for _ in range(int(n_init)):
        C = Z[rng.choice(Z.shape[0], m, replace=False)].astype(np.float64)
        lab = np.zeros(Z.shape[0], np.int64)
        for _ in range(iters):
            d2 = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d2.argmin(1)
            for j in range(m):
                if (lab == j).any():
                    C[j] = Z[lab == j].mean(0)
        inertia = float(((Z - C[lab]) ** 2).sum())
        if inertia < best_in:
            best, best_in = lab, inertia
    return best


def test_kmeans_recovers_a_low_variance_factor_ONLY_after_whitening():
    """⭐⭐ The hypothesis itself, end to end and falsifiable.

    A block with one **huge label-free nuisance** direction and one **small
    class-carrying** direction: k-means on the raw block spends its codewords on the
    nuisance and is at chance on the class; on the whitened block it recovers the
    class.  ⛔ If this failed, whitening would not do what the hypothesis says it
    does and the whole Leg 2 would be pointless.
    """
    rng = np.random.default_rng(21)
    n = 1200
    y = rng.integers(0, 2, size=n)
    Z = np.empty((n, 2))
    Z[:, 0] = rng.normal(size=n) * 10.0            # ⛔ the nuisance: 20x the scale
    Z[:, 1] = 0.5 * y + rng.normal(size=n) * 0.005  # ⭐ the class-carrying axis
    Zf = Z.astype(np.float32)
    wc = (tuple(range(2)),)

    def _purity(codes):
        return sum(int(np.bincount(y[codes == cc], minlength=2).max())
                   for cc in np.unique(codes)) / n

    raw = _purity(_kmeans_codes(np.asarray(Zf, np.float64), 2, seed=1))
    wht = _purity(_kmeans_codes(
        np.asarray(WellTransform.fit(Zf, wc, "pca_whiten").forward(Zf), np.float64), 2, seed=1))
    assert raw < 0.60, raw          # ≈ chance (1/2) — the nuisance axis eats the codes
    assert wht > 0.95, wht          # ⭐ the class is recovered
    assert wht - raw > 0.3


def test_scale_matching_is_METRIC_NEUTRAL_and_restores_commensurability():
    """⭐⭐ The matched-compute control, shown to be neutral rather than assumed.

    ``train_store`` anneals a softmax over ``−d²/(2τ²)`` with ``τ`` in **absolute**
    distance units, so an arm whose feature scale differs by ~15× sees a different
    anneal.  The remedy is a **uniform** per-well scalar: Euclidean k-means is
    equivariant under it, so the induced clustering is ⛔ **identical** — it changes
    the training dynamics' units, never the metric.
    """
    wc = _wells([6, 8], seed=41)
    rng = np.random.default_rng(41)
    X = (rng.normal(size=(800, 14)) @ rng.normal(size=(14, 14))).astype(np.float32) * 4.0
    raw = WellTransform.fit(X, wc, "pca_whiten", scale_match=False)
    mat = WellTransform.fit(X, wc, "pca_whiten", scale_match=True)
    Zr, Zm = raw.forward(X), mat.forward(X)
    for k, w in enumerate(wc):
        idx = list(w)
        # the matched block has the RAW block's mean squared norm
        assert float(np.mean((Zm[:, idx].astype(np.float64) ** 2).sum(1))) == pytest.approx(
            float(np.mean((X[:, idx].astype(np.float64) ** 2).sum(1))), rel=1e-4)
        # ⭐ and it is the unmatched block times ONE scalar => same clustering
        a = _kmeans_codes(np.asarray(Zr[:, idx], np.float64), 4, seed=5)
        b = _kmeans_codes(np.asarray(Zm[:, idx], np.float64), 4, seed=5)
        assert np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])
        assert mat.scales[k] != pytest.approx(1.0, abs=1e-6)
    assert mat.round_trip(X)["max_rel_error"] < 1e-3


def test_scale_matching_leaves_the_identity_control_bit_identical():
    """⛔ The control must stay the LITERAL shipped path, matched or not."""
    wc = _wells([4, 6], seed=42)
    X = np.random.default_rng(42).normal(size=(200, 10)).astype(np.float32) * 9.0
    for m in (False, True):
        tr = WellTransform.fit(X, wc, "identity", scale_match=m)
        assert np.array_equal(tr.forward(X), X)
        assert all(abs(a - 1.0) < 1e-12 for a in tr.scales)


def test_pca_and_zca_whitening_induce_the_SAME_clustering():
    """⭐ The stated reason PCA is used and not ZCA: Euclidean k-means is invariant
    under an orthogonal change of basis, so the two induce the same partition."""
    rng = np.random.default_rng(22)
    A = rng.normal(size=(8, 8))
    X = (rng.normal(size=(500, 8)) @ A).astype(np.float32)
    wc = (tuple(range(8)),)
    tr = WellTransform.fit(X, wc, "pca_whiten")
    Zp = np.asarray(tr.forward(X), np.float64)
    V = np.asarray(tr.fwd[0])
    # ZCA = PCA followed by a rotation back: Z_zca = Z_pca @ R with R orthogonal
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    Zz = Zp @ Q
    a = _kmeans_codes(Zp, 4, seed=2)
    b = _kmeans_codes(Zz, 4, seed=2)
    # same partition up to a relabelling: the co-clustering matrices must agree
    assert np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])
    assert V.shape == (8, 8)


# ---------------------------------------------------------------------------
# 4. ⭐ The byte ledger and the fit-time-only deployment
# ---------------------------------------------------------------------------


def test_byte_ledger_reproduces_the_hub_arithmetic():
    """⭐ The two figures the report is read against, re-derived from the REAL seed-0
    well sizes — ⛔ never hand-typed into a table."""
    pca = transform_byte_ledger("pca_whiten", SHIPPED_WELL_SIZES)
    diag = transform_byte_ledger("diag_std", SHIPPED_WELL_SIZES)
    assert pca["sum_d_squared"] == 898_120
    assert pca["deployed_bytes"] == 3_600_672
    assert pca["deployed_bytes"] / STORE_BYTE_BUDGET == pytest.approx(6.87, abs=0.01)
    assert diag["deployed_bytes"] == 16_384
    assert diag["deployed_bytes"] / STORE_BYTE_BUDGET == pytest.approx(0.03125, abs=1e-6)
    assert transform_byte_ledger("identity", SHIPPED_WELL_SIZES)["deployed_bytes"] == 0
    # ⛔ full-D whitening is NOT the ruled arm and must dwarf the per-well one
    assert transform_byte_ledger("pca_whiten", [2048])["deployed_bytes"] > 16_000_000
    for led in (pca, diag):
        assert led["fit_time_only_bytes"] == 0
    # ⭐ the matched-compute scalar is FOLDED IN: it costs no extra bytes
    wc = _wells([5, 7], seed=51)
    X = np.random.default_rng(51).normal(size=(200, 12)).astype(np.float32)
    a = WellTransform.fit(X, wc, "pca_whiten", scale_match=False).byte_ledger()
    b = WellTransform.fit(X, wc, "pca_whiten", scale_match=True).byte_ledger()
    assert a["deployed_bytes"] == b["deployed_bytes"]


def test_unwhiten_store_is_the_fit_time_only_deployment():
    """⭐ "0 deployed bytes on the request-side path" as an assertion, not a claim."""
    wc = _wells([5, 7], seed=31)
    X = np.random.default_rng(31).normal(size=(400, 12)).astype(np.float32) * 2.0
    tr = WellTransform.fit(X, wc, "pca_whiten")
    store = GenStore(wc, 8, (12,), jax.random.PRNGKey(0))
    codes = address_codes(store, tr.forward(X))
    out = assert_deployment_equivalence(store, unwhiten_store(store, tr), tr, codes)
    assert out["max_rel_delta"] < 1e-4, out
    # the byte count of the store itself is untouched by the transform
    assert int(unwhiten_store(store, tr).n_bytes()) == int(store.n_bytes())


def test_unwhiten_store_refuses_a_transform_from_a_different_partition():
    wc = _wells([5, 7], seed=32)
    other = _wells([7, 5], seed=33)
    X = np.random.default_rng(32).normal(size=(100, 12)).astype(np.float32)
    store = GenStore(wc, 4, (12,), jax.random.PRNGKey(0))
    with pytest.raises(AssertionError, match="NOT the unwhitened partition"):
        unwhiten_store(store, WellTransform.fit(X, other, "pca_whiten"))


def test_deployment_equivalence_FAILS_on_a_tampered_store():
    """⛔ The equivalence check must be able to fail, or it proves nothing."""
    import equinox as eqx

    wc = _wells([4, 4], seed=34)
    X = np.random.default_rng(34).normal(size=(200, 8)).astype(np.float32)
    tr = WellTransform.fit(X, wc, "pca_whiten")
    store = GenStore(wc, 6, (8,), jax.random.PRNGKey(1))
    good = unwhiten_store(store, tr)
    bad = eqx.tree_at(lambda t: t.modes, good, [m + 7.0 for m in good.modes])
    codes = address_codes(store, tr.forward(X))
    with pytest.raises(AssertionError, match="0 deployed bytes"):
        assert_deployment_equivalence(store, bad, tr, codes)


# ---------------------------------------------------------------------------
# 5. ⭐⭐ Leg 4 — the per-well capacity bound and its pre-registered reading
# ---------------------------------------------------------------------------


def test_per_well_capacity_flags_two_hues_in_one_well():
    """⛔ ``M = 64`` is already exceeded by two 10-valued hues in one well (100)."""
    names = ["floor_hue", "wall_hue", "scale"]
    cards = [10, 10, 8]
    H = [np.log2(10), np.log2(10), np.log2(8)]
    mi = np.array([[H[0] * 0.9, H[1] * 0.9, 0.0],   # well 0 carries BOTH hues
                   [0.0, 0.0, H[2] * 0.9]])         # well 1 carries scale alone
    cap = per_well_capacity(mi, H, cards, names, 64)
    assert cap["per_well"][0]["load_combinations"] == 100
    assert cap["per_well"][0]["over_capacity"] is True
    assert cap["per_well"][1]["load_combinations"] == 8
    assert cap["per_well"][1]["over_capacity"] is False
    assert cap["n_wells_over_capacity"] == 1
    # ⛔ the aggregate code is SUFFICIENT and may not be written as the cause
    assert cap["aggregate_is_sufficient"] is True
    assert "may NOT be written as the cause" in cap["ruling"]


def test_aggregate_sufficiency_is_the_shipped_arithmetic():
    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    cards = [10, 10, 10, 8, 4, 15]
    H = [float(np.log2(c)) for c in cards]
    cap = per_well_capacity(np.zeros((5, 6)), H, cards, names, 64)
    assert cap["aggregate_bits_available"] == pytest.approx(30.0)
    assert cap["aggregate_bits_needed"] == pytest.approx(np.log2(480_000), abs=1e-6)
    assert cap["aggregate_bits_needed"] == pytest.approx(18.87, abs=0.01)
    assert cap["aggregate_is_sufficient"] is True


def test_capacity_reading_fires_only_on_a_FLAT_total():
    flat = capacity_reading(3.68, 3.71, 0.10)
    moved = capacity_reading(3.68, 4.40, 0.10)
    assert flat["total_is_flat"] is True
    assert "CAPACITY IS BINDING" in flat["reading"]
    assert moved["total_is_flat"] is False
    assert "NOT the flat-total case" in moved["reading"]


# ---------------------------------------------------------------------------
# 6. The config block
# ---------------------------------------------------------------------------


def test_config_block_round_trips_through_its_own_access_pattern(tmp_path):
    cfg = get_default_config()
    c = gen_variance_eq_config(cfg)
    assert isinstance(c, GenVarianceEqConfig)
    assert gen_variance_eq_config(cfg) is c              # memoised
    assert c.arms == ["identity", "pca_whiten", "diag_std"]
    assert c.dec_recipe == "v1_capacity_schedule"        # ⛔ never "baseline"
    assert c.stop_if_premise_dead is True
    p = tmp_path / "config.yaml"
    save_config(cfg, p)
    assert gen_variance_eq_config(load_config(p)).arms == c.arms


def test_quick_does_not_shrink_M_or_disable_the_partition_assertion():
    """⭐ ``K·log2(M) = 30 bits`` is R20's design constant; the smoke exercises it."""
    from chlu.experiments.exp_gen_variance_eq import apply_quick

    cfg = get_default_config()
    m_before = int(cfg.gen_store.n_modes)
    apply_quick(cfg)
    assert int(cfg.gen_store.n_modes) == m_before == 64
    assert gen_variance_eq_config(cfg).probe_steps == 30
