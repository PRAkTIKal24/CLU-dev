"""⭐⭐ **G-ADDR** (C2W8 pass 3, charter §A30.1) — the addressability leg, and the
designed controls that prove it can BOTH fail and pass.

*"A gate that cannot fail on the thing that matters."* Three instances in one
wave: pass 1's vacuous ``M`` (geometric leg ~10x the address resolution, refused
nothing) · monitor #3's admission gate (refusal rate 0.000, still open) · the
pass-2 capture gate, which passed **arm B on 3/3 seeds with 16/16 items never
read** because G-CAP/G-DEC/G-DRIFT all measure retrievability *at the sites* and
nothing measured whether a query reaches its site.

⛔ **And its mirror image, guarded here too: a gate that cannot PASS is equally
vacuous.** So this file carries a designed **positive** control alongside the
designed negatives.

| control | construction | asserted |
|---|---|---|
| **C+** | planted store, wells at the keys, cue inside the basins | **G-ADDR PASSES**, ``A1 >= 0.80`` |
| **N1** | **arm B's banked measured legs** (`census_armB.json`, 3 seeds) | verdict **FAILS** |
| **N1'** | planted **narrow** wells: retrievable at their own sites, unreachable from a cue | **FAILS** on A1/A2 |
| **N2** | **planted permutation**: same store, same queries, wrong declared targets | ``A1 == 0``, **FAILS** |
| **S** | **scale-only**: the identical rig with every address quantity x ``a`` | ``|dA1| <= 0.05`` |
| **R3** | Ruling 3's counterfactual: the write objective prefers a displaced minimum | the attractor **MOVES** (``follow >= 0.5``) |

Plus the cross-kernel repair of ``own_foreign_site_depth`` (pass-2 reconciliation
item 1) and the config round-trip of the new additive flags.

Thresholds and predictions were registered in
``.claude/outputs/c2w8p3-gate-addr/PREREG.md`` **before** any cell here ran.
"""

import jax
import numpy as np
import pytest

from chlu.config import get_default_config, load_config, save_config
from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.well_lifecycle import (
    GADDR_A2_MAX,
    LEGAL_RESCALE,
    cue_queries,
    displaced_write_counterfactual,
    flatten_unused_groups,
    gate_addr,
    gate_addr_verdict,
    own_foreign_site_depth,
    plant_item,
    scale_guard,
)

#: The planted rigs live at the census's own address dimension. Below ~6 dims the
#: cue geometry stops being representative: an isotropic jitter is nearly
#: orthogonal to the inter-key direction in high d and is NOT in low d, which is
#: exactly why the census can score above chance at all.
ADDR_DIM = 8
#: The planted rigs' DECLARED cue scale (there is no `phi` pool to measure one
#: from). Reported as a dimensionless ratio by `gate_addr` itself.
PLANTED_SPACING_REF = 0.10


def _planted_system(n=6, *, depth=1.0, width=0.30, radius=0.7, seed=0):
    """``n`` planted wells on orthogonal axes — ground truth by construction."""
    cfg = CluSystemConfig(
        addr_dim=ADDR_DIM, payload_dim=1, capacity=int(n), atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=0.05, read_steps=120, address_steps=60,
        n_query_per_item=4, atom_width=float(width),
    )
    sysm = build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)
    centers = np.zeros((n, ADDR_DIM), dtype=float)
    for i in range(n):
        centers[i, i % ADDR_DIM] = float(radius) * (1.0 if i < ADDR_DIM else -1.0)
    for i in range(n):
        plant_item(sysm, i, centers[i], payload=0.15 * (i - n / 2.0),
                   depth=float(depth), width=float(width), leak=0.0)
    flatten_unused_groups(sysm)
    return sysm, centers


def _addr(sysm, **kw):
    kw.setdefault("spacing", PLANTED_SPACING_REF)
    kw.setdefault("n_query_per_item", 4)
    kw.setdefault("n_dirs", 8)
    kw.setdefault("bisect_steps", 6)
    return gate_addr(sysm, **kw)


# --------------------------------------------------------------------------
# C+ — THE DESIGNED POSITIVE: a gate that cannot pass is as vacuous as one
#      that cannot fail
# --------------------------------------------------------------------------
def test_positive_control_gate_addr_passes_on_a_planted_addressable_store():
    sysm, _ = _planted_system(n=6, depth=1.0, width=0.30)
    g = _addr(sysm, seed=0)
    assert g["A1"]["correct_basin_rate"] >= 0.80, g["A1"]
    assert g["A1"]["pass"] and g["A2"]["pass"] and g["A3"]["A3a_pass"], g
    # A3b has no stream on a planted rig: a DECLARED not-applicable, never a null
    assert g["A3"]["A3b_applicable"] is False
    assert g["A3"]["A3b_stream_margin"] is None
    assert "NOT-APPLICABLE" in g["A3"]["A3b_status"]
    assert g["gate_addr_pass"] is True, g
    # ... and the arm-level verdict agrees on a single-cell rig
    assert gate_addr_verdict([g], min_seeds=1)["gate_addr_pass"] is True


# --------------------------------------------------------------------------
# N2 — THE PLANTED PERMUTATION: identical store, identical queries, wrong
#      declared targets. A leg that cannot tell right from wrong scores the same.
# --------------------------------------------------------------------------
def test_negative_planted_permutation_scores_zero_and_fails():
    sysm, _ = _planted_system(n=6, depth=1.0, width=0.30)
    good = _addr(sysm, seed=0)
    bad = _addr(sysm, seed=0, permute=True)
    assert bad["permuted_targets"] is True
    assert bad["A1"]["correct_basin_rate"] <= 0.02, bad["A1"]
    assert bad["A2"]["never_addressed_frac"] >= 1.0 - 1e-9, bad["A2"]
    assert bad["A3"]["A3a_cue_margin"] <= 0.0, bad["A3"]
    assert bad["gate_addr_pass"] is False
    # ⭐ the two runs differ ONLY in the declared target, so the gap IS the leg
    assert good["gate_addr_pass"] and not bad["gate_addr_pass"]
    # the "some basin" question — the one pass 2 asked — cannot tell them apart
    assert good["A1"]["any_basin_rate"] == pytest.approx(
        bad["A1"]["any_basin_rate"], abs=1e-12)


# --------------------------------------------------------------------------
# N1' — THE ARM-B-CLASS BLIND SPOT, live and cheap: wells that are retrievable
#       at their own sites but that a cue never reaches.
# --------------------------------------------------------------------------
def test_negative_narrow_wells_are_retrievable_but_not_addressable():
    sysm, _ = _planted_system(n=6, depth=1.0, width=0.03)
    g = _addr(sysm, seed=0)
    assert g["A1"]["correct_basin_rate"] < 0.25, g["A1"]
    assert g["gate_addr_pass"] is False, g
    # the wells exist (this is the point): they are simply not reachable
    assert g["A1"]["n_items_with_zero_basin"] <= g["n_items"]


# --------------------------------------------------------------------------
# N1 — ⭐ ARM B'S BANKED CONFIGURATION MUST FAIL G-ADDR.
#      The single most important assertion in this file.
# --------------------------------------------------------------------------
#: Arm B's banked measured legs. Provenance: `.claude/outputs/c2w8p2-emission-head/
#: census_armB.json`, seeds 0/1/2 — `usage.frac_never_read` = 1.0 on every seed
#: and the stream launder margin `mean(read_acc - knn_acc)` = -0.609/-0.609/-0.563
#: (mean -0.594, the number the Advisor quotes at §A29.3). A1 is left at its
#: measured re-score's most GENEROUS possible value (1.0) precisely to show that
#: arm B fails **without** A1 having to help.
ARM_B_BANKED_A3B = (-0.609375, -0.609375, -0.5625)


#: ⭐⭐ **C2W8 CLOSE-OUT, item (iv) + acceptance 3 — N1 IS REWRITTEN, NOT DELETED.**
#: As originally filed, N1 fed ``a2 = 1.0`` from arm B's banked
#: ``usage.frac_never_read``. That is the **launch-point** statistic Advisor
#: erratum 1 (§A31.1) RETIRED: ``covered`` is computed on ``q0`` against the
#: codebook, so it is store-invariant by construction. Arm B's **measured**
#: settle-side G-ADDR legs (`GATE-ADDR-VALIDATED.json::arm_rescores`) are
#: ``A1 = 0.9297 / 0.9844 / 0.8750`` (pass 3/3) and ``A2 = 0.0625 / 0.0 / 0.125``
#: (pass 3/3); it failed **only** on A3, the launder margin — which charter
#: §A33.1 now labels a **DIAGNOSTIC, never a pass condition**.
#: ⇒ Under the repaired instrument arm B's banked configuration **PASSES the two
#: MECHANICS legs.** That is a mechanical consequence of a ratified rule, ⛔ not a
#: re-scoring into a claim and ⛔ not an arm-race adjudication (§A30.1: the race
#: is VOID and stays unadjudicated; arm B is claim-barred regardless).
#: The live designed negatives that make G-ADDR falsifiable are N1' and N2 below
#: — planted, ground-truth, and running the shipped code, not banked arithmetic.
ARM_B_BANKED_A2_SETTLE_SIDE = (0.0625, 0.0, 0.125)
ARM_B_BANKED_A1 = (0.9296875, 0.984375, 0.875)


def _legs(a1, a2, a3a, a3b, n_items=16, n_q=128):
    """A leg dict in `gate_addr`'s shape, for pinning the verdict ARITHMETIC.

    ⭐ Mirrors the shipped rule **as amended by the close-out**: the pass
    condition is the two MECHANICS legs (A1, A2); A3 is measured, reported and
    excluded (§A33.1).
    """
    chance = 1.0 / n_items
    se = float(np.sqrt(chance * (1 - chance) / n_q))
    thr = max(4.0 * chance, chance + 2 * se)
    n_correct = int(round(a1 * n_q))
    n_needed = int(np.ceil(thr * n_q - 1e-9))
    a1_pass = bool(a1 >= thr)
    return {
        "A1": {"correct_basin_rate": a1, "pass": a1_pass,
               "threshold": thr, "chance": chance, "se": se,
               "margin_in_se_vs_threshold": float((a1 - thr) / se),
               "n_correct": n_correct, "n_correct_needed": n_needed,
               "reads_to_flip": (n_correct - n_needed + 1 if a1_pass
                                 else n_needed - n_correct),
               "label": "MECHANICS"},
        "A2": {"never_addressed_frac": a2, "pass": bool(a2 <= GADDR_A2_MAX),
               "label": "MECHANICS"},
        "A3": {"A3a_cue_margin": a3a, "A3a_pass": bool(a3a > 0),
               "A3b_stream_margin": a3b, "A3b_applicable": True,
               "A3b_pass": bool(a3b > 0),
               "label": "DIAGNOSTIC", "in_pass_condition": False,
               "pass": bool(a3a > 0 and a3b > 0)},
        # ⛔ A3 is NOT in the pass condition (close-out acceptance 3)
        "gate_addr_pass": bool(a1 >= thr and a2 <= GADDR_A2_MAX),
    }


def test_arm_b_banked_configuration_fails_only_on_the_diagnostic_leg():
    """⭐ The rewritten N1, stated honestly and in both directions."""
    legs = [_legs(a1=a1, a2=a2, a3a=-0.0625, a3b=m)
            for a1, a2, m in zip(ARM_B_BANKED_A1, ARM_B_BANKED_A2_SETTLE_SIDE,
                                 ARM_B_BANKED_A3B, strict=True)]
    v = gate_addr_verdict(legs, min_seeds=3)
    # the MECHANICS legs pass on the MEASURED (settle-side) numbers ...
    assert v["A1_pass_seeds"] == 3 and v["A2_pass_seeds"] == 3, v
    # ... and A3, the launder margin, fails 3/3 — as a DIAGNOSTIC column
    assert v["A3_pass_seeds"] == 0, v
    assert "DIAGNOSTIC" in v["A3_label"]
    assert v["gate_addr_pass"] is True, v
    # ⛔ the retired launch-side statistic must never be substituted for A2 again
    launch_side_a2 = 1.0   # arm B's banked usage.frac_never_read, 3/3 seeds
    assert launch_side_a2 != ARM_B_BANKED_A2_SETTLE_SIDE[0]


def test_the_launder_margin_is_a_diagnostic_and_cannot_decide_the_verdict():
    """⭐⭐ Acceptance 3 (§A33.1): A3 out of the pass condition, pinned.

    Two leg sets identical except for A3 must produce the **same** verdict.
    """
    good_a3 = [_legs(a1=0.9, a2=0.0, a3a=+1.0, a3b=+1.0) for _ in range(3)]
    bad_a3 = [_legs(a1=0.9, a2=0.0, a3a=-1.0, a3b=-1.0) for _ in range(3)]
    v_good = gate_addr_verdict(good_a3, min_seeds=3)
    v_bad = gate_addr_verdict(bad_a3, min_seeds=3)
    assert v_good["gate_addr_pass"] == v_bad["gate_addr_pass"] is True
    assert v_good["A3_pass_seeds"] == 3 and v_bad["A3_pass_seeds"] == 0
    # ... while a MECHANICS leg still decides it
    assert gate_addr_verdict([_legs(a1=0.0, a2=0.0, a3a=+1.0, a3b=+1.0)] * 3,
                             min_seeds=3)["gate_addr_pass"] is False


def test_arm_b_would_have_passed_the_pass_two_gate_legs():
    """The defect class, stated as a test: arm B's banked G-CAP/G-DEC/G-DRIFT all
    pass 3/3 (`census_armB.json::gate`) — the pass-2 gate could not see
    addressability at all. ⚠ What G-ADDR changed about arm B's verdict was
    re-read by the close-out (§A33.1): its A1/A2 pass and only the DIAGNOSTIC
    launder margin fails."""
    pass2 = {"G_CAP": [True] * 3, "G_DEC": [True] * 3, "G_DRIFT": [True] * 3}
    assert all(all(v) for v in pass2.values())
    # the leg that can still fail arm-B-class stores is A1/A2 — proven live by
    # the planted N1' (narrow wells) above, not by banked arithmetic.
    assert gate_addr_verdict([_legs(a1=0.0, a2=1.0, a3a=+1.0, a3b=+1.0)] * 3,
                             min_seeds=3)["gate_addr_pass"] is False


# --------------------------------------------------------------------------
# S — THE SCALE-INVARIANCE GUARD (PREREG-C2W8-PASS3 §4): if rescaling phi moves
#     G-ADDR, the leg measures the SCALE and does not ship.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("a", [0.8, 1.25])
def test_scale_only_control_does_not_move_gate_addr(a):
    base, _ = _planted_system(n=6, depth=1.0, width=0.30, radius=0.7)
    g0 = _addr(base, seed=0)
    scaled, _ = _planted_system(n=6, depth=1.0, width=0.30 * a,
                                radius=0.7 * a)
    g1 = _addr(scaled, seed=0, spacing=PLANTED_SPACING_REF * a)
    # the DIMENSIONLESS ratios are identical by construction — that is the point
    assert g1["cue_sigma_over_codebook_spacing"] == pytest.approx(
        g0["cue_sigma_over_codebook_spacing"], rel=1e-6)
    d = abs(g1["A1"]["correct_basin_rate"] - g0["A1"]["correct_basin_rate"])
    # ⭐ close-out item (iii): the metric bound is now the DIAGNOSTIC ...
    assert d <= 0.05, (d, g0["A1"], g1["A1"])
    # ... and the VERDICT is the pass condition
    guard = scale_guard([("1.0", g0), (str(a), g1)])
    assert guard["verdict_stable"] is True, guard
    assert guard["scale_guard_pass"] is True, guard


# --------------------------------------------------------------------------
# ⭐⭐ C2W8 CLOSE-OUT item (iii) — THE REPAIRED SCALE GUARD ASSERTS THE VERDICT
# --------------------------------------------------------------------------
#: The exact banked pair the OLD guard passed and should not have
#: (`c2w8p3-capture-strong-phi` §7): a **legal** full-state co-scaled rescale
#: (a = 0.8, payload co-scaled) moved A1 0.24219 -> 0.28906 across the registered
#: 0.25 threshold. |dA1| = 0.04688 <= 0.05 => the metric bound HELD, and the
#: leg's verdict flipped False -> True, the baseline sitting ONE READ below the
#: threshold.
BANKED_SCALE_FLIP = (0.24219, 0.28906)


def test_scale_guard_fails_on_the_banked_metric_bounded_verdict_flip():
    """⛔ The designed negative of the repaired guard: bounded metric, flipped
    verdict, and the guard must FAIL. The old guard passed this exact pair."""
    a1_lo, a1_hi = BANKED_SCALE_FLIP
    lo = _legs(a1=a1_lo, a2=0.375, a3a=-0.078, a3b=-0.031)
    hi = _legs(a1=a1_hi, a2=0.375, a3a=-0.117, a3b=+0.016)
    # the registered threshold is 0.25 at n_items = 16, and the pair straddles it
    assert lo["A1"]["threshold"] == pytest.approx(0.25, abs=1e-9)
    assert lo["A1"]["pass"] is False and hi["A1"]["pass"] is True
    g = scale_guard([("a=1.0", lo), ("a=0.8 (full-state co-scaled)", hi)])
    # the OLD guard's condition still holds — and is reported as a DIAGNOSTIC ...
    assert abs(a1_hi - a1_lo) <= 0.05
    assert g["metric_bounded_DIAGNOSTIC"] is True, g
    # ... while the REPAIRED guard fails, because the verdict moved
    assert g["verdict_stable"] is False, g
    assert g["scale_guard_pass"] is False, g
    assert "A1" in " ".join(g["legs_flipped"]) and "gate_addr_pass" in " ".join(
        g["legs_flipped"]), g


def test_scale_guard_passes_only_when_every_leg_boolean_is_stable():
    stable = [("a=1.0", _legs(a1=0.90, a2=0.0, a3a=+1.0, a3b=+1.0)),
              ("a=0.8", _legs(a1=0.61, a2=0.0, a3a=+1.0, a3b=+1.0))]
    g = scale_guard(stable)
    assert g["verdict_stable"] is True and g["scale_guard_pass"] is True
    # ⛔ a HUGE metric move with a stable verdict passes the guard: the guard is
    # about the verdict, and the metric movement is reported beside it
    assert g["metric_bounded_DIAGNOSTIC"] is False, g
    assert g["by_scale"][0]["abs_delta_A1"] == pytest.approx(0.29, abs=1e-9)
    # one scale is not a guard — declared, never a null
    assert scale_guard(stable[:1])["scale_guard_pass"] is False
    assert "NOT INFORMATIVE" in scale_guard(stable[:1])["status"]


def test_the_legal_rescale_is_full_state_co_scaling_including_the_payload():
    """§A31.6, Head-ratified: address-only rescaling is NOT a symmetry."""
    from chlu.experiments.exp_well_lifecycle import full_state_coscaled_config

    cfg = get_default_config()
    w0 = cfg.experiment_well_lifecycle
    base_addr, base_pay = float(w0.addr_scale_mult), float(w0.payload_scale)
    out = full_state_coscaled_config(cfg, 0.8).experiment_well_lifecycle
    assert out.addr_scale_mult == pytest.approx(base_addr * 0.8)
    # payload = (label - 4.5) / payload_scale, so /0.8 multiplies the PAYLOAD
    assert out.payload_scale == pytest.approx(base_pay / 0.8)
    assert cfg.experiment_well_lifecycle.addr_scale_mult == base_addr  # no mutation
    assert "payload" in LEGAL_RESCALE and "NOT a symmetry" in LEGAL_RESCALE
    with pytest.raises(ValueError):
        full_state_coscaled_config(cfg, 0.0)


# --------------------------------------------------------------------------
# ⭐⭐ C2W8 CLOSE-OUT item (ii) — A1's MARGIN-IN-SE BESIDE THE BOOLEAN
# --------------------------------------------------------------------------
def test_a1_reports_its_margin_and_reads_to_flip_beside_the_boolean():
    """⚠ `randconv` scored 31 / 31 / 29 of 128 against a 32/128 threshold — it
    failed by ONE read on 2 of 3 seeds. A discrete threshold reported without its
    margin turns a tie into a verdict."""
    n_items, n_q = 16, 128
    thr = max(4.0 / n_items, 1.0 / n_items + 2.0 * float(
        np.sqrt((1 / 16) * (15 / 16) / n_q)))
    assert thr == pytest.approx(0.25, abs=1e-9)
    assert int(np.ceil(thr * n_q - 1e-9)) == 32          # the 32/128 threshold
    for n_correct, expect_pass, flip in ((31, False, 1), (29, False, 3),
                                         (32, True, 1), (40, True, 9)):
        lg = _legs(a1=n_correct / n_q, a2=0.0, a3a=0.0, a3b=0.0)
        assert lg["A1"]["pass"] is expect_pass
        assert lg["A1"]["reads_to_flip"] == flip, (n_correct, lg["A1"])
    # and the margin is emitted in SE units against the THRESHOLD, not chance
    lg = _legs(a1=31 / n_q, a2=0.0, a3a=0.0, a3b=0.0)
    assert lg["A1"]["margin_in_se_vs_threshold"] < 0.0
    assert lg["A1"]["margin_in_se_vs_threshold"] == pytest.approx(
        (31 / n_q - thr) / lg["A1"]["se"], rel=1e-9)


def test_gate_addr_emits_the_margin_everywhere_a1_is_emitted():
    sysm, _ = _planted_system(n=6, depth=1.0, width=0.30)
    g = _addr(sysm, seed=0)
    for k in ("margin_in_se", "margin_in_se_vs_chance",
              "margin_in_se_vs_threshold", "n_correct", "n_correct_needed",
              "reads_to_flip", "label"):
        assert k in g["A1"], (k, sorted(g["A1"]))
    assert g["A1"]["label"] == "MECHANICS"
    assert g["A3"]["label"] == "DIAGNOSTIC" and g["A3"]["in_pass_condition"] is False
    # the arm-level emitter carries them too
    v = gate_addr_verdict([g], min_seeds=1)
    assert v["A1_margin_in_se_vs_threshold_by_seed"][0] == pytest.approx(
        g["A1"]["margin_in_se_vs_threshold"])
    assert v["A1_reads_to_flip_by_seed"] == [g["A1"]["reads_to_flip"]]


# --------------------------------------------------------------------------
# R3 — RULING 3's COUNTERFACTUAL (charter §A30.3): the attractor CAN move off the
#      stored key when the write objective demands it. Outcome, not identity.
# --------------------------------------------------------------------------
def test_ruling3_attractor_can_move_off_the_stored_key():
    cfg = CluSystemConfig(
        addr_dim=ADDR_DIM, payload_dim=1, capacity=2, atoms_per_item=16,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=0,
        d_safe_override=0.05, read_steps=200, address_steps=100,
        n_query_per_item=2, write_steps=200,
        atom_width=0.20, atom_site_local_init=True, atom_site_local_radius=0.20,
    )
    sysm = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
    a = np.zeros((ADDR_DIM,), dtype=float)
    a[0] = 0.4
    delta = np.zeros((ADDR_DIM,), dtype=float)
    delta[1] = 0.30  # one declared spacing, orthogonal to the key
    out = displaced_write_counterfactual(sysm, 0, a, 0.2, delta=delta, seed=0)
    assert out["atom_site_local_init"] is True
    # ⛔ follow < 0.5 REVERSES the compliance ruling — the test says so out loud
    assert out["attractor_can_move"] is True, out
    assert out["follow_fraction"] >= 0.5, out


# --------------------------------------------------------------------------
# housekeeping 1 — the CROSS-KERNEL repair of `own_foreign_site_depth`
# --------------------------------------------------------------------------
def _two_group_store(kernel, width=0.10, cutoff=2.5):
    cfg = CluSystemConfig(
        addr_dim=2, payload_dim=1, capacity=2, atoms_per_item=4,
        min_atoms=8, min_atoms_base=4, min_atoms_c=1.0, seed=0,
        d_safe_override=0.05, read_steps=20, address_steps=10,
        n_query_per_item=1, atom_width=float(width),
        atom_kernel=str(kernel), atom_kernel_cutoff=float(cutoff),
    )
    sysm = build_system(cfg, key=jax.random.PRNGKey(0), loud=False)
    plant_item(sysm, 0, np.array([0.5, 0.0]), payload=0.0, depth=1.0,
               width=float(width))
    plant_item(sysm, 1, np.array([-0.5, 0.0]), payload=0.0, depth=1.0,
               width=float(width))
    flatten_unused_groups(sysm)
    return sysm


def _legacy_gaussian_own_foreign(store, slot, site):
    """The pre-pass-3 estimator, verbatim — the thing being corrected."""
    atoms = store.atoms
    A = np.asarray(atoms.amp, dtype=float) ** 2
    s = np.exp(np.asarray(atoms.log_width, dtype=float))
    c = np.asarray(atoms.centers, dtype=float)
    z = np.asarray(site, dtype=float).reshape(1, -1)[:, : c.shape[1]]
    d2 = np.sum((c - z) ** 2, axis=-1)
    w = A * np.exp(-d2 / (2.0 * s**2 + 1e-12))
    m = np.asarray(store.group_rows(int(slot)), dtype=bool)
    return float(np.sum(w[m])), float(np.sum(w[~m]))


def test_own_foreign_matches_the_legacy_gaussian_form():
    """⛔ Continuity invariant: every banked own/foreign number was taken on a
    Gaussian store, so the repair must not move a reported digit.

    The only difference under ``gaussian`` is the store's own epsilon (``1e-9``
    vs the estimator's old ``1e-12``): a **3e-8 relative** change, six orders
    below the 4-dp resolution anything is ever quoted at.
    """
    sysm = _two_group_store("gaussian", width=0.30)
    site = np.zeros((sysm.store.dim,), dtype=float)
    site[0] = 0.5
    new = own_foreign_site_depth(sysm.store, 0, site)
    old = _legacy_gaussian_own_foreign(sysm.store, 0, site)
    assert new[0] == pytest.approx(old[0], rel=1e-6, abs=1e-12)
    assert new[1] == pytest.approx(old[1], rel=1e-6, abs=1e-12)
    # ... and no reported digit moves
    assert round(new[0], 4) == round(old[0], 4)
    assert round(new[1], 4) == round(old[1], 4)


def test_numpy_atom_profile_mirrors_the_shipped_one_for_every_kernel():
    """⛔ The float64 mirror must track `atom_profile` for EVERY declared kernel;
    adding a kernel without mirroring it fails here."""
    from chlu.core.memory_potentials import ATOM_KERNELS, atom_profile
    from chlu.core.well_lifecycle import _atom_profile_np

    d2 = np.linspace(0.0, 1.0, 41) ** 2
    for k in ATOM_KERNELS:
        for s in (0.05, 0.2, 0.5):
            ref = np.asarray(atom_profile(d2, np.full_like(d2, s), k, 2.5),
                             dtype=float)
            got = _atom_profile_np(d2, np.full_like(d2, s), k, 2.5)
            assert np.allclose(got, ref, atol=2e-6), (k, s)
        assert _atom_profile_np(np.zeros(1), np.full(1, 0.2), k, 2.5)[0] == \
            pytest.approx(1.0, abs=1e-9)   # profile(0) = 1 for every kernel


def test_own_foreign_reads_the_compact_kernel_and_the_gaussian_form_over_reads():
    """⭐ THE CROSS-KERNEL TEST (pass-2 reconciliation item 1).

    Under ``wendland`` the atoms have **compact support** ``R = cutoff * s``, so
    a foreign atom beyond ``R`` contributes **exactly 0** — while the hard-coded
    Gaussian estimator credits it with a tail that the landscape does not have.
    """
    w = 0.30   # arm-A-scale width: R = 2.5 s = 0.75 < the 1.0 site separation
    sysm = _two_group_store("wendland", width=w, cutoff=2.5)
    site = np.zeros((sysm.store.dim,), dtype=float)
    site[0] = 0.5  # slot 0's own site; slot 1 sits 1.0 away, R = 0.25
    own_new, foreign_new = own_foreign_site_depth(sysm.store, 0, site)
    own_old, foreign_old = _legacy_gaussian_own_foreign(sysm.store, 0, site)

    assert foreign_new == 0.0                      # compact: exactly zero
    assert foreign_old > 0.0                       # the Gaussian tail is fictitious
    assert own_new == pytest.approx(1.0, abs=1e-6)  # profile(0) = 1 => depth
    # the estimator now agrees with the store's OWN landscape, which is the test
    from chlu.core.memory_potentials import atom_profile
    atoms = sysm.store.atoms
    d2 = np.sum((np.asarray(atoms.centers, dtype=float) - site[None, :]) ** 2, axis=-1)
    ref = float(np.sum((np.asarray(atoms.amp, dtype=float) ** 2)
                       * np.asarray(atom_profile(
                           d2, np.exp(np.asarray(atoms.log_width, dtype=float)),
                           "wendland", 2.5), dtype=float)))
    assert own_new + foreign_new == pytest.approx(ref, rel=1e-5)
    # ⭐ the bias the pass-2 arm A report had to carry as a caveat: the
    # Gaussian-hardcoded form invents a foreign contribution that is a
    # measurable fraction of the own leg where the true one is EXACTLY zero
    assert foreign_old / own_old > 1e-3


def test_cross_kernel_over_read_factor_is_reported_not_hidden():
    """The direction of the pass-2 bias, pinned: the Gaussian form OVER-reads."""
    sysm = _two_group_store("truncated_gaussian", width=0.10, cutoff=2.0)
    site = np.zeros((sysm.store.dim,), dtype=float)
    site[0] = 0.5
    new = own_foreign_site_depth(sysm.store, 0, site)
    old = _legacy_gaussian_own_foreign(sysm.store, 0, site)
    assert old[1] >= new[1]          # foreign over-read (or equal at exact zero)
    assert old[0] + old[1] >= new[0] + new[1]


# --------------------------------------------------------------------------
# the cue set itself
# --------------------------------------------------------------------------
def test_cue_queries_carry_ground_truth_and_scale_with_the_declared_spacing():
    c = np.eye(4)[:, :3] * 0.5
    q_a, t_a, d_a = cue_queries(c, 4, spacing=0.1, kappa_q=1.0, n_per_item=5, seed=3)
    q_b, t_b, d_b = cue_queries(c, 4, spacing=0.2, kappa_q=1.0, n_per_item=5, seed=3)
    assert q_a.shape == (20, 4) and np.array_equal(t_a, d_a)
    # same key stream, twice the declared spacing => exactly twice the offset
    off_a = q_a[:, :3] - c[t_a]
    off_b = q_b[:, :3] - c[t_b]
    assert np.allclose(off_b, 2.0 * off_a, atol=1e-12)
    assert np.allclose(q_a[:, 3], 0.0)  # payload channels stay zero


def test_permutation_changes_only_the_declared_target():
    c = np.eye(4)[:, :3] * 0.5
    q0, t0, d0 = cue_queries(c, 4, spacing=0.1, n_per_item=3, seed=1)
    q1, t1, d1 = cue_queries(c, 4, spacing=0.1, n_per_item=3, seed=1, permute=True)
    assert np.array_equal(q0, q1) and np.array_equal(t0, t1)
    assert not np.array_equal(d0, d1)
    assert np.all(d1 == (t1 + 1) % c.shape[0])


def test_gate_addr_verdict_needs_min_seeds():
    ok = _legs(a1=0.9, a2=0.0, a3a=0.5, a3b=0.1)
    assert gate_addr_verdict([ok], min_seeds=3)["gate_addr_pass"] is False
    assert gate_addr_verdict([ok] * 3, min_seeds=3)["gate_addr_pass"] is True


# --------------------------------------------------------------------------
# the additive config flags round-trip (and default to the pass-1/2 behaviour)
# --------------------------------------------------------------------------
def test_new_well_lifecycle_flags_default_to_the_banked_behaviour(tmp_path):
    cfg = get_default_config()
    w = cfg.experiment_well_lifecycle
    assert w.addr_scale_mult == 1.0        # scale-only control OFF by default
    assert w.run_gate_addr is True
    assert w.gaddr_kappa_q == 1.0
    w.addr_scale_mult = 0.8
    w.gaddr_kappa_q = 1.5
    w.run_gate_addr = False
    p = tmp_path / "config.yaml"
    save_config(cfg, p)
    back = load_config(p).experiment_well_lifecycle
    assert back.addr_scale_mult == 0.8
    assert back.gaddr_kappa_q == 1.5
    assert back.run_gate_addr is False


# --------------------------------------------------------------------------
# a blocking-bug regression found while re-scoring the banked arms
# --------------------------------------------------------------------------
def test_arm_a_store_config_accepts_the_overrides_seam():
    """⚠ At `main @ 1eda6a0` **arm A could not run at all**: `run_census_cell`
    always passes the pass-2 `overrides=` seam and arm A's substituted factory
    did not accept it (`TypeError`). Arm A and arm B were merged independently
    and neither merge re-ran the other."""
    from chlu.experiments.exp_capture_armA import arm_store_config
    from chlu.experiments.exp_well_lifecycle import store_config

    cfg = get_default_config()
    plain = arm_store_config(cfg, 0, 0.12)
    with_over = arm_store_config(cfg, 0, 0.12, overrides={"emission_head": True})
    assert plain.emission_head is False and with_over.emission_head is True
    # and the frozen pass-1 factory itself still takes both call shapes
    assert store_config(cfg, 0, 0.12, overrides=None).addr_dim == \
        store_config(cfg, 0, 0.12).addr_dim
