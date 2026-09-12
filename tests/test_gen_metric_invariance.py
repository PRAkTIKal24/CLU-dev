"""R26 ruling 2 — the pre-registered configuration-invariance check.

The criterion, verbatim: *"the axis must give the same value for the same features
under different partitions."*  ⭐ The load-bearing tests here are the ones that can
**fail on purpose**, because the whole point of the check is that it is capable of
rejecting something:

- ``test_NEGATIVE_CONTROL_a_class_count_ratio_axis_is_rejected`` — a transfer-ratio
  axis scored with a 6-class numerator against a 15-class denominator must be
  called a FAIL.  ⛔ A check that passes the axis we already know is broken is not
  a check.
- ``test_configuration_invariance_fires_on_a_single_inverted_seed`` — one inverted
  seed fails the leg.  ⛔ It is never averaged away.
- ``test_dci_completeness_is_exactly_invariant_to_well_relabelling`` — the reason
  DCI-C is promoted, asserted as arithmetic and not as a hope.
- ``test_dci_completeness_does_not_read_the_declared_class_count`` — DCI-C's
  normalisers are ``log K`` and the **realised** ``H_f``; handing it the full grid's
  cardinalities on rows that realise fewer classes must change nothing.

⛔ Wells are integer-indexed here as everywhere else.
"""

import numpy as np
import pytest

from chlu.config import GenMetricInvarianceConfig, get_default_config, load_config, save_config
from chlu.core.gen_store import (
    balanced_random_grouping,
    chance_corrected_balanced_accuracy,
    class_count_invariance,
    class_count_restriction,
    configuration_invariance,
    contiguous_grouping,
    dci_scores,
    invariance_verdict,
    mig_score,
    permute_mode_codes,
    permute_well_codes,
    relabelling_invariance,
)

FACTORS = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
CARDS = (10, 10, 10, 8, 4, 15)


def _synthetic_codes(n=4096, k=5, m=64, seed=0):
    """Codes that genuinely carry factor structure: well j reads factor j through a
    deterministic map plus noise, well 4 is pure noise."""
    rng = np.random.default_rng(seed)
    labels = np.stack([rng.integers(0, c, n) for c in CARDS], axis=1).astype(np.int64)
    codes = np.zeros((n, k), np.int64)
    for j in range(k - 1):
        flip = rng.random(n) < 0.15
        codes[:, j] = np.where(flip, rng.integers(0, m, n), labels[:, j] % m)
    codes[:, k - 1] = rng.integers(0, m, n)
    return codes, labels


# ---------------------------------------------------------------------------
# leg 1a — relabelling
# ---------------------------------------------------------------------------


def test_dci_completeness_is_exactly_invariant_to_well_relabelling():
    codes, labels = _synthetic_codes()
    base = dci_scores(codes, labels, CARDS)["completeness"]
    rng = np.random.default_rng(7)
    vals = [base]
    for _ in range(5):
        perm = rng.permutation(codes.shape[1])
        vals.append(dci_scores(permute_well_codes(codes, perm), labels, CARDS)["completeness"])
    leg = relabelling_invariance(vals, tol=1e-12, label="dci_c")
    assert leg["pass"], leg
    # ⚠ MEASURED, and it falsified this spoke's own prediction P4: the agreement is
    # NOT bit-identical.  DCI-C is row-permutation-invariant as ALGEBRA, but the
    # sum order inside ``_mi_bits``/``R.sum`` moves with the well order, so the
    # value wobbles at the last ulp (~5e-17).  ⛔ That is four orders inside the
    # pre-registered 1e-12 tolerance and is float arithmetic, not a class-count
    # dependence — but it is reported, not asserted away.
    assert leg["max_abs_deviation"] < 1e-15, leg["max_abs_deviation"]


def test_dci_completeness_is_exactly_invariant_to_mode_relabelling():
    codes, labels = _synthetic_codes()
    base = dci_scores(codes, labels, CARDS)["completeness"]
    rng = np.random.default_rng(11)
    perms = [rng.permutation(64) for _ in range(codes.shape[1])]
    moved = permute_mode_codes(codes, perms)
    assert not np.array_equal(moved, codes)
    assert dci_scores(moved, labels, CARDS)["completeness"] == pytest.approx(base, abs=1e-15)


def test_mig_is_also_relabelling_invariant():
    codes, labels = _synthetic_codes()
    perm = np.array([3, 0, 4, 1, 2])
    assert mig_score(permute_well_codes(codes, perm), labels, CARDS)["mig"] == pytest.approx(
        mig_score(codes, labels, CARDS)["mig"], abs=1e-12
    )


def test_permute_well_codes_refuses_a_non_permutation():
    codes, _ = _synthetic_codes(n=16)
    with pytest.raises(ValueError):
        permute_well_codes(codes, [0, 0, 1, 2, 3])
    with pytest.raises(ValueError):
        permute_well_codes(codes, [0, 1, 2])


def test_permute_mode_codes_refuses_a_short_permutation():
    codes, _ = _synthetic_codes(n=64)
    with pytest.raises(ValueError):
        permute_mode_codes(codes, [np.arange(4)] * codes.shape[1])


def test_relabelling_invariance_needs_something_to_compare():
    with pytest.raises(ValueError):
        relabelling_invariance([0.5])


def test_relabelling_invariance_rejects_a_moving_value():
    leg = relabelling_invariance([0.5, 0.5, 0.500001], tol=1e-12)
    assert not leg["pass"]
    assert leg["max_abs_deviation"] == pytest.approx(1e-6, rel=1e-3)


# ---------------------------------------------------------------------------
# leg 2 — class count
# ---------------------------------------------------------------------------


def test_dci_completeness_does_not_read_the_declared_class_count():
    """⭐ The structural reason DCI-C survives the disease that killed the leakage
    axis: it normalises by ``log K`` and by the **realised** factor entropy, never
    by a declared number of classes."""
    codes, labels = _synthetic_codes()
    res = class_count_restriction(labels, FACTORS, {"orientation": 6, "scale": 6})
    assert res["realised_classes_restricted"]["orientation"] == 6
    assert res["realised_classes_restricted"]["scale"] == 6
    sub_c, sub_y = codes[res["mask"]], labels[res["mask"]]
    realised = tuple(
        int(np.unique(sub_y[:, j]).size) if n in ("orientation", "scale") else CARDS[j]
        for j, n in enumerate(FACTORS)
    )
    full = dci_scores(sub_c, sub_y, CARDS)
    real = dci_scores(sub_c, sub_y, realised)
    assert full["completeness"] == real["completeness"]
    assert full["informativeness"] == real["informativeness"]


def test_class_count_restriction_keeps_the_lowest_indexed_values():
    _, labels = _synthetic_codes()
    res = class_count_restriction(labels, FACTORS, {"orientation": 6})
    assert res["kept_values"]["orientation"] == [0, 1, 2, 3, 4, 5]
    assert res["n_kept"] < res["n_total"]


def test_class_count_restriction_refuses_an_unknown_factor():
    _, labels = _synthetic_codes(n=32)
    with pytest.raises(KeyError):
        class_count_restriction(labels, FACTORS, {"colour": 3})


def test_NEGATIVE_CONTROL_a_class_count_ratio_axis_is_rejected():
    """⛔ THE TEETH TEST.  A transfer ratio whose numerator realises 6 classes and
    whose denominator realises 15 is the axis that died in ``gen-r0-wide-refit``
    §4.2.  Scored through the identical leg, it must FAIL."""
    rng = np.random.default_rng(3)

    def ratio_axis(n_num, n_den, skill):
        """balanced accuracy improves as the class count falls, at fixed skill."""
        ba = lambda c: 1.0 / c + skill * (1.0 - 1.0 / c)  # noqa: E731
        return 1.0 - ba(n_num) / ba(n_den)

    # correlation is the stronger arm at matched class counts...
    matched = ratio_axis(15, 15, 0.62) - ratio_axis(15, 15, 0.45)
    # ...and the weaker arm gains more from the easier numerator.
    mismatched = ratio_axis(6, 15, 0.62) - ratio_axis(6, 15, 0.45)
    leg = class_count_invariance(matched, mismatched, tol_ratio=0.5, label="leak_raw")
    assert not leg["pass"], leg
    assert leg["sensitivity_S"] > 0.5
    assert rng.random() >= 0.0  # rng kept for provenance of the construction

    verdict = invariance_verdict("leak_raw", [leg])
    assert verdict["verdict"] == "FAIL"
    assert verdict["legs_failed"] == ["2_class_count"]


def test_class_count_invariance_passes_a_genuinely_stable_margin():
    leg = class_count_invariance(0.1745, 0.1620, tol_ratio=0.5, per_cell_signs=[1] * 6)
    assert leg["pass"]
    assert leg["sensitivity_S"] < 0.1


def test_class_count_invariance_counts_inverted_cells():
    leg = class_count_invariance(
        0.10, 0.09, tol_ratio=0.5, per_cell_signs=[1, 1, 1, -1, -1, 1], min_sign_agreement=5
    )
    assert leg["n_cells_sign_agree"] == 4
    assert not leg["pass"]


# ---------------------------------------------------------------------------
# leg 3 — configuration
# ---------------------------------------------------------------------------


def test_configuration_invariance_fires_on_a_sign_inversion():
    """⛔ The measured failure of the old axis: Montero +0.1337, wide −0.0660."""
    leg = configuration_invariance(
        {
            "recomb2element": [0.0764, 0.1264, 0.1983],
            "recomb2element_wide": [-0.0458, -0.0746, -0.0778],
        },
        band=3.0,
        label="leak_raw",
    )
    assert not leg["pass"]
    assert not leg["sign_agreement"]
    assert len(leg["inverted_cells"]) == 3


def test_configuration_invariance_fires_on_a_single_inverted_seed():
    """⛔ One inverted seed is a finding, never noise to average away."""
    leg = configuration_invariance(
        {"a": [0.17, 0.18, 0.19], "b": [0.13, -0.01, 0.14]}, band=3.0
    )
    assert not leg["pass"]
    assert leg["n_cells_sign_agree"] == 5
    assert leg["inverted_cells"][0]["seed_index"] == 1


def test_configuration_invariance_fires_on_an_out_of_band_magnitude():
    leg = configuration_invariance({"a": [0.30, 0.30, 0.30], "b": [0.05, 0.05, 0.05]}, band=3.0)
    assert leg["sign_agreement"]
    assert not leg["band_ok"]
    assert not leg["pass"]


def test_configuration_invariance_accepts_agreeing_signs_inside_the_band():
    leg = configuration_invariance({"a": [0.166, 0.165, 0.193], "b": [0.126, 0.133, 0.130]})
    assert leg["pass"]
    assert leg["n_cells_sign_agree"] == 6
    assert leg["magnitude_ratio"] < 3.0


def test_configuration_invariance_needs_two_configurations():
    with pytest.raises(ValueError):
        configuration_invariance({"a": [0.1, 0.2]})


# ---------------------------------------------------------------------------
# the chance correction, the partition variants, the verdict
# ---------------------------------------------------------------------------


def test_chance_correction_maps_chance_to_zero_at_any_class_count():
    for c in (2, 4, 6, 8, 15):
        assert chance_corrected_balanced_accuracy(1.0 / c, c) == pytest.approx(0.0, abs=1e-12)
        assert chance_corrected_balanced_accuracy(1.0, c) == pytest.approx(1.0, abs=1e-12)


def test_chance_correction_is_undefined_for_one_class():
    assert np.isnan(chance_corrected_balanced_accuracy(1.0, 1))


def test_chance_correction_does_not_remove_the_difficulty_dependence():
    """⚠ The honest limit of R26's companion: the correction removes the chance
    FLOOR, not the fact that a 6-way problem is easier than a 15-way one.  Two
    probes of equal *skill* still score differently once skill is not a pure
    interpolation between chance and 1."""
    # a concrete, monotone difficulty model: per-class recall decays with c.
    ba = {6: 0.70, 15: 0.55}
    cc = {c: chance_corrected_balanced_accuracy(v, c) for c, v in ba.items()}
    assert cc[6] != pytest.approx(cc[15], abs=1e-3)


def test_contiguous_grouping_matches_a_well_size_histogram():
    sizes = [295, 286, 286, 744, 437]
    lab = contiguous_grouping(sizes, d=sum(sizes))
    assert lab.size == sum(sizes)
    assert [int(x) for x in np.bincount(lab)] == sizes
    assert lab[0] == 0 and lab[-1] == len(sizes) - 1
    with pytest.raises(ValueError):
        contiguous_grouping(sizes, d=sum(sizes) + 1)


def test_balanced_random_grouping_is_equal_sized_and_deterministic():
    lab = balanced_random_grouping(2048, 5, seed=0)
    counts = np.bincount(lab, minlength=5)
    assert lab.size == 2048
    assert int(counts.max() - counts.min()) <= 1, counts
    assert np.array_equal(lab, balanced_random_grouping(2048, 5, seed=0))
    assert not np.array_equal(lab, balanced_random_grouping(2048, 5, seed=1))


def test_invariance_verdict_is_an_AND_over_the_legs():
    good = {"leg": "1a_relabelling", "pass": True}
    bad = {"leg": "3_configuration", "pass": False}
    assert invariance_verdict("x", [good, good])["verdict"] == "PASS"
    v = invariance_verdict("x", [good, bad])
    assert v["verdict"] == "FAIL"
    assert v["legs_failed"] == ["3_configuration"]
    with pytest.raises(ValueError):
        invariance_verdict("x", [])


def test_the_config_block_round_trips_and_keeps_its_pre_registered_defaults(tmp_path):
    cfg = get_default_config()
    g = cfg.gen_metric_invariance
    assert isinstance(g, GenMetricInvarianceConfig)
    assert g.relabel_tol == 1e-12
    assert g.class_count_tol_ratio == 0.5
    assert g.configuration_band == 3.0
    assert g.class_budget == {"orientation": 6, "scale": 6}
    assert g.negative_control_axis == "leak_raw"
    assert g.candidate_axis == "dci_c"
    assert g.partitions[:2] == ["correlation", "random"]
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p).gen_metric_invariance
    assert back.class_budget == g.class_budget
    assert back.partitions == g.partitions
    assert back.configurations == g.configurations


def test_class_count_restriction_accepts_an_explicit_value_set():
    """⭐ The form a RATIO axis needs.  A count budget resolved on the FULL set keeps
    the lowest-indexed values; applied to a corner that fixes the HIGHEST-indexed
    ones (Montero: ``scale >= 1.1``, ``orientation > 20``) it keeps **nothing**.  So
    the survivors are read off the numerator instead of chosen."""
    _, labels = _synthetic_codes()
    corner = np.isin(labels[:, 5], [9, 10, 11, 12, 13, 14])
    full_budget = class_count_restriction(labels, FACTORS, {"orientation": 6})
    chosen = full_budget["kept_values"]["orientation"]
    assert chosen == [0, 1, 2, 3, 4, 5]
    # the full set's budget, applied to the corner, empties it
    empty = class_count_restriction(labels[corner], FACTORS, {"orientation": chosen})
    assert empty["n_kept"] == 0
    # the numerator's own realised classes, applied to the full set, do not
    realised = sorted(int(v) for v in np.unique(labels[corner, 5]))
    res = class_count_restriction(labels, FACTORS, {"orientation": realised})
    assert res["kept_values"]["orientation"] == realised
    assert res["realised_classes_restricted"]["orientation"] == len(realised)
    assert 0 < res["n_kept"] < res["n_total"]
