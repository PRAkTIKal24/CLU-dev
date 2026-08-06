"""⭐ **K9 — the re-registered merge criterion must be able to REFUSE.**

Registration: `ERRATA-C2W8-PASS2.md` §2 (filed **before** the predicate was
written). Kill-condition: `PREREG-C2W8-PASS2.md` K9.

**What went wrong in pass 1, in one line.** The shipped criterion
(`well_lifecycle.mergeable_pairs`, which these tests do **not** touch) admitted a
pair iff ``payload_dist <= 0.1`` and ``center_sep <= R_cert``. On the frozen
census ``R_cert`` was **10.31-11.23x the measured key spacing** so the geometric
leg refused **nothing**, and **every** admitted pair's ``payload_dist`` was
**exactly 0.0** — true by construction in a class-incremental stream. Monitor #3
``vacuous_gate`` tripped **3/3 at refusal rate 0.000**. Shipping it would ship
*"collapse to one well per class"* wearing a certificate costume.

⛔ **No merge verb exists and none is tested here.** These tests pin a
**predicate** and, above all, pin that it **can say no on either leg** — the
thing pass 1's criterion could not do.

⚠ The frozen-census numbers below are **transcribed** from
`.claude/outputs/c2w8-well-lifecycle/census.json` (gitignored, so the values are
inlined with their provenance rather than loaded).
"""

import numpy as np
import pytest

from chlu.core.soft_certificate import (
    MERGE_RHO_GEOM,
    MERGE_TAU_PAYLOAD,
    MergeCriterionConfig,
    merge_admissible,
    merge_criterion_report,
    payload_scale_from_pairs,
)

#: (seed, key spacing `geometry.median_nn_task1`, pass-1 `R_cert`, pass-1 admitted
#: count, min `center_sep` among the admitted pairs) — frozen `census.json`.
CENSUS = [
    (0, 0.14071388837267038, 1.5402313814684787, 28, 0.2898529505533079),
    (1, 0.13753963021394877, 1.4177282633637860, 29, 0.2055168033337777),
    (2, 0.14683180691371017, 1.6481952510934572, 29, 0.1701577167759895),
]

#: pass-1's payload leg, verbatim: an ABSOLUTE tolerance (`cfg.payload_tol`).
PASS1_PAYLOAD_TOL = 0.1


def _pass1_admits(center_sep, payload_dist, r_cert, payload_tol=PASS1_PAYLOAD_TOL):
    """Pass 1's criterion, reimplemented here so the two can be raced."""
    return bool(payload_dist <= payload_tol and center_sep <= r_cert)


# ---------------------------------------------------------------------------
# 1. the registered operating point is commensurate with the address resolution
# ---------------------------------------------------------------------------
def test_the_pass1_radius_was_ten_times_the_key_spacing_and_the_new_one_is_one():
    """The retirement, stated in numbers: 10.31-11.23x vs 1.00x."""
    ratios = [r_cert / ks for _, ks, r_cert, _, _ in CENSUS]
    assert min(ratios) > 10.0 and max(ratios) < 11.5, ratios
    for _, ks, _, _, _ in CENSUS:
        v = merge_admissible(0.0, 0.0, key_spacing=ks, payload_scale=1.0)
        assert v["commensurability"] == pytest.approx(MERGE_RHO_GEOM)
        assert v["r_merge"] == pytest.approx(MERGE_RHO_GEOM * ks)


# ---------------------------------------------------------------------------
# 2. ⭐ THE DESIGNED NEGATIVES — one per leg, plus the anti-vacuity clause
# ---------------------------------------------------------------------------
def test_designed_negative_refused_on_geometry_and_pass1_would_have_admitted_it():
    """⭐ N-geom: a pair 2.13x the key spacing apart.

    Pass 1 **admits** it (0.30 <= 1.54 and 0.0 <= 0.1); the re-registered
    criterion **refuses it on the geometry leg**. This negative is what makes the
    two criteria distinguishable rather than a re-labelling.
    """
    ks = CENSUS[0][1]
    v = merge_admissible(0.30, 0.0, key_spacing=ks, payload_scale=1.0)
    assert v["admitted"] is False
    assert v["refused_on"] == ["geometry"]
    assert v["geometry_applicable"] is True and v["geometry_ok"] is False
    assert v["payload_ok"] is True            # the OTHER leg passed: the refusal is the geometry's
    assert 0.30 / ks == pytest.approx(2.132, abs=1e-3)
    assert _pass1_admits(0.30, 0.0, CENSUS[0][2]) is True


def test_designed_negative_refused_on_payload():
    """⭐ N-pay: centres inside one key spacing, payloads far apart on the measured scale."""
    ks = CENSUS[0][1]
    v = merge_admissible(0.05, 0.9, key_spacing=ks, payload_scale=1.0)
    assert v["admitted"] is False
    assert v["refused_on"] == ["payload"]
    assert v["geometry_ok"] is True           # the OTHER leg passed: the refusal is the payload's
    assert v["payload_applicable"] is True and v["payload_ok"] is False
    assert v["payload_tol"] == pytest.approx(MERGE_TAU_PAYLOAD)


def test_the_anti_vacuity_clause_refuses_a_degenerate_payload_channel():
    """⭐ N-degen — **the clause that kills pass 1's vacuity.**

    This is the frozen census's own configuration: ``payload_dist = 0.0``
    everywhere. Under an ABSOLUTE tolerance it passes trivially. Under the
    re-registered criterion the payload *scale* is 0, the channel carries no
    discriminative content, and the leg is INAPPLICABLE ⇒ **refuse**, never a
    silent pass.
    """
    ks = CENSUS[0][1]
    v = merge_admissible(0.05, 0.0, key_spacing=ks, payload_scale=0.0)
    assert v["admitted"] is False
    assert v["refused_on"] == ["payload_degenerate"]
    assert v["payload_applicable"] is False
    assert v["geometry_ok"] is True
    # ... and pass 1 admits exactly this pair.
    assert _pass1_admits(0.05, 0.0, CENSUS[0][2]) is True


def test_an_unmeasured_key_spacing_refuses_rather_than_certifies():
    """A ruler that was not measured certifies nothing (both non-finite and <= 0)."""
    for bad in (float("nan"), 0.0, -1.0):
        v = merge_admissible(1e-6, 0.0, key_spacing=bad, payload_scale=1.0)
        assert v["admitted"] is False
        assert v["refused_on"] == ["geometry_inapplicable"]
        assert v["geometry_applicable"] is False


# ---------------------------------------------------------------------------
# 3. the designed POSITIVE — the criterion is not merely refusing everything
# ---------------------------------------------------------------------------
def test_designed_positive_is_admitted_on_both_legs():
    """P-pos: inside one key spacing AND inside a quarter of the measured payload spread."""
    ks = CENSUS[0][1]
    v = merge_admissible(0.05, 0.10, key_spacing=ks, payload_scale=1.0)
    assert v["admitted"] is True
    assert v["refused_on"] == []
    assert v["geometry_ok"] and v["payload_ok"]


def test_both_legs_are_individually_binding_on_the_same_population():
    """A 4-pair population separates the legs: 1 admitted, 1 geometry, 1 payload, 1 both."""
    ks = CENSUS[0][1]
    pairs = [
        {"center_sep": 0.05, "payload_dist": 0.10},   # admitted
        {"center_sep": 0.30, "payload_dist": 0.10},   # geometry
        {"center_sep": 0.05, "payload_dist": 0.90},   # payload
        {"center_sep": 0.30, "payload_dist": 0.90},   # both
    ]
    rep = merge_criterion_report(pairs, key_spacing=ks, payload_scale=1.0)
    assert rep["n_pairs"] == 4 and rep["n_admitted"] == 1
    assert rep["refusal_rate"] == pytest.approx(0.75)
    assert rep["n_refused_geometry"] == 2
    assert rep["n_refused_payload"] == 2
    assert rep["vacuous_gate_would_trip"] is False


# ---------------------------------------------------------------------------
# 4. the frozen census, re-scored (registration §2.3 R1/R2/R3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,ks,r_cert,n_adm,min_sep", CENSUS)
def test_R1_every_pair_pass1_admitted_is_refused_on_geometry(seed, ks, r_cert, n_adm, min_sep):
    """⭐ R1: 28 / 29 / 29 admitted -> **0**, all refused on the geometry leg.

    Provable from the banked minima alone: the *smallest* admitted separation is
    already 2.06 / 1.49 / 1.16 key spacings, i.e. above ``rho_geom = 1.0``, so
    every admitted pair is above it.
    """
    assert n_adm > 0
    assert min_sep / ks > MERGE_RHO_GEOM
    v = merge_admissible(min_sep, 0.0, key_spacing=ks, payload_scale=1.0)
    assert v["admitted"] is False and v["refused_on"] == ["geometry"]
    assert _pass1_admits(min_sep, 0.0, r_cert) is True


@pytest.mark.parametrize("seed,ks,r_cert,n_adm,min_sep", CENSUS)
def test_R2_the_frozen_census_trips_vacuous_gate_in_the_OPPOSITE_direction(
    seed, ks, r_cert, n_adm, min_sep
):
    """⭐ R2, registered **as expected, not as a failure.**

    The banked population is entirely refused (f = 1.000, not pass 1's 0.000).
    Pass 1 measured a store with ``capture_radius`` 0.000 on 47/48 wells — there
    is nothing there to merge — so the criterion's discriminating power is proven
    by the designed pairs above, not by this census.
    """
    pairs = [{"center_sep": min_sep, "payload_dist": 0.0}] * n_adm
    rep = merge_criterion_report(pairs, key_spacing=ks, payload_scale=1.0)
    assert rep["n_admitted"] == 0
    assert rep["refusal_rate"] == 1.0
    assert rep["vacuous_gate_would_trip"] is True     # f = 1, the opposite end
    assert rep["n_refused_geometry"] == n_adm


@pytest.mark.parametrize("seed,ks,r_cert,n_adm,min_sep", CENSUS)
def test_R3_the_rho_that_would_admit_any_banked_pair_is_registered(
    seed, ks, r_cert, n_adm, min_sep
):
    """⭐ R3: 2.06 / 1.49 / 1.16 — so loosening ``rho_geom`` is visibly a decision."""
    rho_needed = min_sep / ks
    expected = {0: 2.0599, 1: 1.4942, 2: 1.1589}[seed]
    assert rho_needed == pytest.approx(expected, abs=1e-3)
    just_under = merge_admissible(min_sep, 0.0, key_spacing=ks, payload_scale=1.0,
                                  cfg=MergeCriterionConfig(rho_geom=rho_needed * 0.999))
    just_over = merge_admissible(min_sep, 0.0, key_spacing=ks, payload_scale=1.0,
                                 cfg=MergeCriterionConfig(rho_geom=rho_needed * 1.001))
    assert just_under["admitted"] is False and just_over["admitted"] is True


# ---------------------------------------------------------------------------
# 5. the measured payload scale
# ---------------------------------------------------------------------------
def test_payload_scale_is_measured_on_the_whole_population_and_zero_is_degenerate():
    assert payload_scale_from_pairs([0.0] * 28) == 0.0          # the frozen census
    assert payload_scale_from_pairs([]) == 0.0                  # empty => degenerate, not a pass
    assert payload_scale_from_pairs([0.0, 0.0, 1.0, 2.0, 4.0]) == pytest.approx(1.0)
    assert payload_scale_from_pairs([np.nan, 1.0, 3.0]) == pytest.approx(2.0)


def test_a_wholly_degenerate_population_is_refused_end_to_end():
    """The census's own payload column, scored without a hand-supplied scale."""
    ks = CENSUS[0][1]
    pairs = [{"center_sep": 0.05, "payload_dist": 0.0} for _ in range(28)]
    rep = merge_criterion_report(pairs, key_spacing=ks)        # scale measured => 0.0
    assert rep["payload_scale"] == 0.0
    assert rep["n_admitted"] == 0
    assert rep["n_refused_payload_degenerate"] == 28
    assert rep["vacuous_gate_would_trip"] is True


# ---------------------------------------------------------------------------
# 6. ⛔ the boundary this spoke must not cross
# ---------------------------------------------------------------------------
def test_no_merge_verb_exists():
    """⛔ K9 gates any merge verb anywhere. The criterion ships; the verb does not."""
    import chlu.core.soft_certificate as sc
    import chlu.core.well_lifecycle as wl

    for mod in (sc, wl):
        for name in dir(mod):
            assert not name.startswith("merge_wells"), f"{mod.__name__}.{name}"
        assert not hasattr(mod, "merge_pair")
        assert not hasattr(mod, "do_merge")
