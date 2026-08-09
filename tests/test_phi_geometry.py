"""Tests for the C2W8 pass-3 **φ_dim → addr_dim map** and the geometry it is measured with.

What must never break silently here:

  * **the map exists and is not a truncation.** ``exp_well_lifecycle.PhiAddress``
    forces ``phi_dim = addr_dim`` and truncates; a genuine projection is the whole
    point of pass 3, so :func:`assert_no_truncation` must *fire* when it is handed a
    φ that is wider than the store's address space;
  * ⛔ **Head ruling R2(b): the launder reads the PROJECTED φ.** Asserted here, not
    intended: the kNN launder's keys must be **bit-identical** to the store's own
    addresses, and :func:`launder_audit` must **raise** on the handicap-match case
    (256-dim launder against an 8-dim store);
  * **the map is on the byte ledger of every arm, launder included** (§A4.3), and the
    ledger's φ term is the **same number** on the store arm and on the launder;
  * **the map is neutral on the reference arm** — PCA-of-(PCA-256) fit on the same
    pool reproduces PCA-``d`` — so a strong-φ gain cannot be an artifact of the map;
  * ⭐ **the §4 scale-invariance guard**: the rig normalises addresses by
    ``1 / r95``, so multiplying φ by any constant must move every geometry leg by
    **exactly** 0. If a rescale moves a leg, the leg measures the scale, not the memory;
  * **the GO/NO-GO rule is mechanical** — it fires on a planted improvement and
    refuses on a planted null, both computed, neither argued.

Everything runs on tiny synthetic data — no download, no encoder fitting.
"""

import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments import exp_phi_geometry as pg
from chlu.experiments.exp_well_lifecycle import PhiAddress
from chlu.experiments.phi_encoders import PhiProjection, ProjectedReadIn


# ---------------------------------------------------------------------------
# a tiny stand-in φ: a fixed random linear read-in with a param count
# ---------------------------------------------------------------------------
class _ToyPhi:
    def __init__(self, dim, k, seed=0, scale=1.0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(k, dim)).astype(np.float32)
        self.k = int(k)
        self.scale = float(scale)

    def __call__(self, X):
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None, :]
        return (X @ self.W.T) * self.scale

    def param_floats(self):
        return int(self.W.size)


def _toy_data(n=64, dim=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. the map is a projection, not a truncation
# ---------------------------------------------------------------------------
def test_projection_reduces_and_is_not_a_truncation():
    X = _toy_data()
    phi = _ToyPhi(40, 16)
    proj = PhiProjection(phi(X), 4, form="pca")
    pp = ProjectedReadIn(phi, proj)
    out = np.asarray(pp(X))
    assert out.shape == (len(X), 4)
    # ⛔ a genuine projection mixes ALL 16 input coordinates; truncation would keep 4
    trunc = ProjectedReadIn(phi, PhiProjection(phi(X), 4, form="truncate"))
    assert not np.allclose(out, np.asarray(trunc(X)))
    assert np.allclose(np.asarray(trunc(X)), np.asarray(phi(X))[:, :4], atol=1e-5)
    # every input coordinate is used by the pca map
    assert np.all(np.abs(proj.components).sum(axis=0) > 0)


def test_assert_no_truncation_fires_on_the_shipped_defect():
    X = _toy_data()
    phi = _ToyPhi(40, 16)
    pg.assert_no_truncation(ProjectedReadIn(phi, PhiProjection(phi(X), 8)), 8)
    with pytest.raises(AssertionError, match="TRUNCATE"):
        pg.assert_no_truncation(phi, 8)  # a 16-dim φ against an 8-dim store


def test_projection_refuses_to_invent_dimensions():
    X = _toy_data()
    phi = _ToyPhi(40, 8)
    with pytest.raises(ValueError, match="never invent"):
        PhiProjection(phi(X), 16)


# ---------------------------------------------------------------------------
# 2. ⛔ R2(b): the launder reads the PROJECTED φ — asserted, not intended
# ---------------------------------------------------------------------------
def test_launder_reads_the_projected_phi_bit_identically():
    X = _toy_data(n=24)
    phi = _ToyPhi(40, 16)
    pp = ProjectedReadIn(phi, PhiProjection(phi(X), 6, form="pca"))
    embed = PhiAddress(pp, dim=7, addr_dim=6, scale=0.5)
    rep = pg.launder_audit(embed, X[:12], 6)
    assert rep["launder_key_dim"] == 6 == rep["store_address_dim"]
    assert rep["launder_reads_projected_phi"] and rep["bit_identical_to_store_addresses"]


def test_launder_audit_raises_on_the_handicap_match():
    """A launder reading 16 dims against a 6-dim store is not a launder."""
    X = _toy_data(n=24)
    phi = _ToyPhi(40, 16)
    embed = PhiAddress(phi, dim=17, addr_dim=16, scale=0.5)  # the UNPROJECTED φ
    with pytest.raises(AssertionError, match="handicap match"):
        pg.launder_audit(embed, X[:12], 6)


def test_phi_params_are_on_every_arms_ledger_including_the_launder():
    X = _toy_data()
    phi = _ToyPhi(40, 16)
    proj = PhiProjection(phi(X), 4, form="pca")
    pp = ProjectedReadIn(phi, proj)
    # hand count: mean (16) + components (4×16)
    assert proj.param_floats() == 16 + 4 * 16
    assert pp.param_floats() == phi.param_floats() + proj.param_floats()
    led = pg.byte_ledger(pp, n_launder=16, addr_dim=4, capacity=16)
    assert led["phi_param_floats_total"] == pp.param_floats()
    # ⛔ the SAME φ ⇒ the SAME ledger term on the store arm and on the launder
    assert led["clu_arm_phi_floats"] == led["knn_launder_phi_floats"]
    assert led["encoder_param_floats"] == phi.param_floats()
    assert led["knn_launder_total_floats"] == pp.param_floats() + 16 * 5


def test_unfitted_maps_carry_no_floats_but_report_the_dense_cost():
    X = _toy_data()
    phi = _ToyPhi(40, 16)
    for form in ("gaussian", "truncate"):
        p = PhiProjection(phi(X), 4, form=form, seed=3)
        assert p.param_floats() == 0
        assert p.param_floats_materialised() > 0


# ---------------------------------------------------------------------------
# 3. map neutrality on the reference arm (PCA of PCA is PCA)
# ---------------------------------------------------------------------------
def test_pca_map_is_neutral_on_a_pca_reference():
    """The reference cannot be improved *by the map*, so a strong-φ gain is φ's."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 40)).astype(np.float32)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    direct = Xc @ Vt[:3].T                       # PCA-3 of the data
    wide = Xc @ Vt[:12].T                        # PCA-12 of the data
    mapped = PhiProjection(wide, 3, form="pca")(wide)   # PCA-3 of PCA-12
    # equal up to per-axis sign (SVD sign convention)
    for j in range(3):
        s = np.sign(np.dot(direct[:, j], mapped[:, j]))
        assert np.allclose(direct[:, j], s * mapped[:, j], atol=1e-3)


# ---------------------------------------------------------------------------
# 4. ⭐ the §4 scale-invariance guard: a rescale must move geometry by EXACTLY 0
# ---------------------------------------------------------------------------
def test_geometry_is_invariant_to_rescaling_phi():
    X = _toy_data(n=80)
    base = _ToyPhi(40, 12, seed=2, scale=1.0)
    for c in (0.017, 91.0):
        loud = _ToyPhi(40, 12, seed=2, scale=c)
        rows = []
        for phi in (base, loud):
            pp = ProjectedReadIn(phi, PhiProjection(phi(X), 5, form="pca"))
            sc = pg.address_scale(pp, X)
            emb = PhiAddress(pp, dim=6, addr_dim=5, scale=sc["scale"])
            keys = np.asarray(emb.keys(X[:20]), float)
            rows.append(pg.geometry_row(keys, sigma_q=0.15, d_safe=0.1,
                                        scale=sc["scale"], addr_dim=5, n_keys=20))
        for leg in ("median_nn_spacing", "sigma_q_over_spacing",
                    "spacing_over_uniform_ball", "participation_ratio"):
            assert rows[0][leg] == pytest.approx(rows[1][leg], rel=1e-4), leg


# ---------------------------------------------------------------------------
# 5. the joint dial (d, atom budget) is the SHIPPED law, and d=256 is forbidden
# ---------------------------------------------------------------------------
def test_atom_budget_reproduces_the_shipped_law():
    from chlu.core.clu_system import CluSystemConfig

    for d in (4, 8, 12, 16):
        assert pg.atom_budget(d, capacity=16) == CluSystemConfig(
            addr_dim=d, capacity=16).n_atoms
    assert pg.atom_budget(8) == 8192
    assert pg.atom_budget(16) == 131072
    assert pg.atom_budget(256) > 1e40  # ⛔ naive 256-dim addressing is forbidden


# ---------------------------------------------------------------------------
# 6. rider (b): the admission gate's refusal rate is MEASURED, and can fire
# ---------------------------------------------------------------------------
def test_refusal_simulation_is_two_sided():
    keys = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.01, 0.0]])
    assert pg.refusal_simulation(keys, d_safe=0.5)["refusal_rate"] == 0.25
    assert pg.refusal_simulation(keys, d_safe=1e-6)["refusal_rate"] == 0.0
    assert pg.refusal_simulation(keys, d_safe=10.0)["refusal_rate"] == 0.75


def test_d_safe_pricing_exposes_the_vacuous_gate():
    """A d_safe sized on a DENSE set cannot fire on a SPARSE population."""
    rng = np.random.default_rng(0)
    dense = rng.normal(size=(200, 8))
    sparse = dense[:16]
    p = pg.d_safe_pricing(dense, sparse, 0.88)
    assert p["sizing_set_spacing"] < p["population_spacing"]
    assert p["d_safe_rig_over_population_spacing"] < 0.88
    assert p["d_safe_repriced_over_population_spacing"] == pytest.approx(0.88)
    # ⛔ the rig's pricing is the one that (nearly) cannot fire; re-pricing it on the
    # population it actually gates strictly increases what it refuses
    r_rig = pg.refusal_simulation(sparse, p["d_safe_rig"])["refusal_rate"]
    r_new = pg.refusal_simulation(sparse, p["d_safe_repriced"])["refusal_rate"]
    assert r_rig < 0.1
    assert r_new > r_rig


# ---------------------------------------------------------------------------
# 7. the GO/NO-GO rule is mechanical — it fires, and it refuses
# ---------------------------------------------------------------------------
def _verdict_rows(strong_vals, ref_vals, d=8):
    rows = []
    for s, (a, b) in enumerate(zip(strong_vals, ref_vals, strict=True)):
        for arm, v in (("simclr->pca", a), ("pca@d", b)):
            rows.append({"seed": s, "arm": arm, "addr_dim": d, "n_keys": 16,
                         "sigma_q_over_spacing": float(v)})
    return rows


def test_geometry_verdict_fires_on_a_planted_improvement():
    v = pg.geometry_verdict(_verdict_rows([0.50, 0.52, 0.48], [1.00, 1.02, 0.98]),
                            strong_arm="simclr->pca", ref_arm="pca@d", n_keys=16,
                            se_multiple=2.0, min_seeds_positive=3)
    assert v["geometry_go"] is True
    assert v["by_d"]["8"]["n_seeds_positive"] == 3
    assert v["by_d"]["8"]["ratio_strong_over_reference"] < 1.0


def test_geometry_verdict_refuses_a_planted_null():
    v = pg.geometry_verdict(_verdict_rows([1.00, 0.90, 1.10], [1.00, 1.10, 0.90]),
                            strong_arm="simclr->pca", ref_arm="pca@d", n_keys=16,
                            se_multiple=2.0, min_seeds_positive=3)
    assert v["geometry_go"] is False


def test_geometry_verdict_favours_the_d_with_the_lowest_sigma_over_spacing():
    rows = (_verdict_rows([0.9, 0.9, 0.9], [1.0, 1.0, 1.0], d=8)
            + _verdict_rows([0.4, 0.4, 0.4], [1.0, 1.0, 1.0], d=16))
    v = pg.geometry_verdict(rows, strong_arm="simclr->pca", ref_arm="pca@d",
                            n_keys=16, se_multiple=2.0, min_seeds_positive=3)
    assert v["d_favoured_by_geometry"] == 16


# ---------------------------------------------------------------------------
# 8. the config knob is real and additive
# ---------------------------------------------------------------------------
def test_config_group_exists_with_the_declared_defaults():
    g = get_default_config().experiment_phi_geometry
    assert g.dataset == "cifar10"          # Head ruling R1
    assert list(g.addr_dims) == [8, 12, 16]  # the feasible band under the atom law
    assert g.phi_regime == "task1_only"    # no leakage from unseen tasks
    assert g.query_sigma == 0.15
