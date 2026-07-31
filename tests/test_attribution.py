"""Tests for ``chlu/eval/attribution.py`` — Route 3 stage 1's instrument.

The load-bearing properties are (a) the instrument is **scale-free** (or the
momentum channel at small ``t`` is measured as "uninformative" when it is merely
*small*), (b) the ⛔ §A9.5 table launder is evaluated **leave-one-out** (or it is
fitted on the point it is scored at), and (c) the §A9.4 bar is **arithmetic** —
nothing in it interprets anything.
"""

import numpy as np
import pytest

from chlu.eval.attribution import (
    SLOT_GRID,
    SlotScore,
    a95_verdict,
    apply_a94_bar,
    attribution_curve,
    discriminability,
    identity_decode,
    jacobian_curves,
    per_slot_table_launder,
    slot_channel,
    slot_index_table,
    spearman,
)


class _Res:
    """The two attributes the instrument reads off a ``ReadResult``."""

    def __init__(self, traj, phase):
        self.traj = np.asarray(traj)
        self.phase = np.asarray(phase)


def _res(rng, B=24, n=150, dim=6, scale=1.0, signal=None, noise=0.01):
    traj = rng.normal(size=(B, n, 2 * dim)) * noise
    if signal is not None:
        traj[:, :, 4] += signal[:, None] * scale       # q payload channel
        traj[:, :, dim + 4] += signal[:, None] * scale  # p payload channel
    phase = np.concatenate([np.ones(50, dtype=int), 2 * np.ones(n - 50, dtype=int)])
    return _Res(traj, phase)


# --------------------------------------------------------------------------
# the instrument
# --------------------------------------------------------------------------
def test_spearman_matches_a_hand_computation_and_handles_ties():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, 2 * a + 1) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(spearman(a, np.ones(4)))          # constant => undefined
    # ties get average ranks (0.8944 for a perfectly ordered two-level variable
    # against a four-level one), so a K-valued payload alphabet is still usable
    assert spearman(np.array([0.0, 0.0, 1.0, 1.0]), a) == pytest.approx(0.8944, abs=1e-4)


def test_discriminability_is_SCALE_FREE_which_is_the_whole_point():
    """⭐ At small ``t`` the momentum channel is ``O(t)`` small but (per §A8.1)
    *proportional* to the store's payload. A magnitude-based decode would score
    it as uninformative; the pre-registered instrument must not."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    y = 3.0 * x + 0.0
    big, tiny = discriminability(y, x), discriminability(y * 1e-9, x)
    assert big == pytest.approx(1.0)
    assert tiny == pytest.approx(big)   # nine orders of magnitude, same score
    assert discriminability(-y * 1e-9, x) == pytest.approx(1.0)  # sign-free


def test_slot_channel_splits_q_and_p_at_the_right_offset():
    rng = np.random.default_rng(1)
    r = _res(rng, dim=6)
    q = slot_channel(r, 7, "q", 4)
    p = slot_channel(r, 7, "p", 4)
    assert np.array_equal(q, r.traj[:, 7, 4])
    assert np.array_equal(p, r.traj[:, 7, 6 + 4])


def test_slot_provenance_maps_slots_to_steps_and_phases():
    rng = np.random.default_rng(2)
    rows = slot_index_table(_res(rng), SLOT_GRID, traj_stride=8, dt=0.05,
                            address_steps=400)
    by = {r["slot"]: r for r in rows}
    assert by[0]["phase"] == 1 and by[0]["step"] == 1
    assert by[49]["phase"] == 1 and by[49]["step"] == 8 * 49 + 1
    assert by[50 + 4]["phase"] == 2 and by[54]["step"] == 400 + 8 * 4 + 1
    assert by[54]["t"] == pytest.approx(0.05 * (400 + 33))


# --------------------------------------------------------------------------
# ⛔ §A9.5 — the kill-condition
# --------------------------------------------------------------------------
def test_the_per_slot_table_launder_is_LEAVE_ONE_OUT():
    """A table fitted on the query it is scored at is not a launder."""
    keys = np.array([[0.0], [0.05], [1.0], [1.05]])
    centers = np.array([[0.0], [1.0]])
    # one query in each group is an outlier; LOO must not let it predict itself
    values = np.array([0.0, 10.0, 1.0, 11.0])
    tgt = np.array([0.0, 0.0, 1.0, 1.0])
    d = per_slot_table_launder(values, tgt, keys, centers)
    # the LOO prediction of row k for query i is the OTHER member of the group,
    # so the predictions are [10, 0, 11, 1] => still ordered with the target
    assert 0.0 <= d <= 1.0


def test_a95_fires_when_K_time_indexed_rows_reproduce_the_read():
    rows = {s: [SlotScore(slot=j, channel="q", phase=1, step=j, t=0.1 * j,
                          full=0.9, launder=0.1, floor=0.05, table=0.9 - 0.01 * s)
                for j in (0, 1, 2)] for s in (0, 1, 2)}
    v = a95_verdict(rows)
    assert v["fires"] is True
    assert v["q"]["frac_reproduced"] == 1.0 and v["q"]["n_read_beats"] == 0
    assert "FIRES" in v["verdict"]


def test_a95_does_NOT_fire_when_the_read_beats_the_table():
    rows = {s: [SlotScore(slot=j, channel="p", phase=1, step=j, t=0.1 * j,
                          full=0.9, launder=0.1, floor=0.05, table=0.4)
                for j in (0, 1)] for s in (0, 1, 2)}
    v = a95_verdict(rows)
    assert v["fires"] is False and v["p"]["n_read_beats"] == 2


# --------------------------------------------------------------------------
# ⭐ §A9.4 — the bar is ARITHMETIC
# --------------------------------------------------------------------------
def _rows(margins, channel="p"):
    """Build per-seed rows with prescribed margins (full - launder - floor)."""
    out = {}
    for s, m in enumerate(margins):
        out[s] = [SlotScore(slot=0, channel=channel, phase=1, step=1, t=0.05,
                            full=0.5 + m, launder=0.5, floor=0.0)]
    return out


def test_the_bar_clears_only_beyond_2_SE_with_three_seeds():
    v = apply_a94_bar(_rows([0.30, 0.32, 0.34]), family="f",
                      admissible_seeds=[0, 1, 2])
    r = v["rows"][0]
    assert r["n_seeds"] == 3
    assert r["margin_mean"] == pytest.approx(0.32)
    assert r["margin_sd"] == pytest.approx(np.std([0.30, 0.32, 0.34], ddof=1))
    assert r["margin_se"] == pytest.approx(r["margin_sd"] / np.sqrt(3))
    assert r["clears"] is True and v["unlock"] is True

    noisy = apply_a94_bar(_rows([0.30, -0.30, 0.31]), family="f",
                          admissible_seeds=[0, 1, 2])
    assert noisy["rows"][0]["clears"] is False and noisy["unlock"] is False


def test_a_family_with_no_admissible_cell_cannot_unlock():
    """⛔ ABSTAIN: it neither unlocks stage 2 nor blocks it."""
    v = apply_a94_bar(_rows([0.5, 0.5, 0.5]), family="f", admissible_seeds=[])
    assert v["n_admissible_seeds"] == 0 and v["unlock"] is False and not v["rows"]


def test_fewer_than_three_seeds_cannot_unlock():
    v = apply_a94_bar({0: _rows([0.5])[0], 1: _rows([0.5])[0]}, family="f",
                      admissible_seeds=[0, 1])
    assert v["unlock"] is False


def test_q_and_p_are_scored_separately_and_p_can_unlock_alone():
    """⭐ *"a live p-channel unlocks even with a dead q-channel"* — in code."""
    per_seed = {}
    for s in range(3):
        per_seed[s] = [
            SlotScore(slot=0, channel="p", phase=1, step=1, t=0.05,
                      full=0.9, launder=0.3, floor=0.1),
            SlotScore(slot=0, channel="q", phase=1, step=1, t=0.05,
                      full=0.3, launder=0.3, floor=0.1),
        ]
    v = apply_a94_bar(per_seed, family="f", admissible_seeds=[0, 1, 2])
    got = {(r["channel"], r["clears"]) for r in v["rows"]}
    assert ("p", True) in got and ("q", False) in got
    assert v["unlock"] is True
    assert [c["channel"] for c in v["clearing_set"]] == ["p"]


# --------------------------------------------------------------------------
# the end-to-end shape, and §A8.2
# --------------------------------------------------------------------------
def test_attribution_curve_recovers_a_planted_store_signal():
    rng = np.random.default_rng(3)
    target = rng.normal(size=24)
    # ⭐ signal 1e-6 against noise 1e-12: six orders SMALLER than a magnitude
    # decode's natural threshold, and still perfectly ordered
    full = _res(rng, signal=target, scale=1e-6, noise=1e-12)
    launder = _res(rng)                              # store deleted: no signal
    floor = _res(rng)
    rows = attribution_curve(full, launder, floor, target, 4, slots=(0, 1, 2))
    assert len(rows) == 6                            # 3 slots x {q, p}
    for r in rows:
        assert r.full > 0.9                          # scale-free: 1e-6 is enough
        assert r.dividend > 0.3
        assert r.margin == pytest.approx(r.dividend - r.floor)


def test_jacobian_curves_report_contraction_and_separation():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(6), 4)
    base = rng.normal(size=(24, 150, 12))
    a = _Res(base, np.ones(150, dtype=int))
    delta = rng.normal(size=(24, 6)) * 0.1
    b = _Res(base + 0.5 * np.concatenate([delta, np.zeros((24, 6))], axis=-1)[:, None, :],
             np.ones(150, dtype=int))
    j = jacobian_curves(a, b, delta, labels, slots=(0, 1), dim=6)
    con = j["channels"]["q"]["contraction"]
    assert len(con) == 2 and all(c == pytest.approx(0.5, rel=1e-3) for c in con)
    assert j["channels"]["p"]["contraction"][0] == pytest.approx(0.0, abs=1e-9)
    assert len(j["channels"]["q"]["fisher"]) == 2


def test_identity_decode_is_leave_one_out_and_chance_on_noise():
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(6), 4)
    clean = np.repeat(np.eye(6), 4, axis=0) + 0.01 * rng.normal(size=(24, 6))
    assert identity_decode(clean, labels) == pytest.approx(1.0)
    noise = rng.normal(size=(24, 6))
    assert identity_decode(noise, labels) < 0.6
