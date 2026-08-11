"""C2W10 stream sources — the synthetic instrument, decimation, and the
reproduction gate.

⛔ Per §A14.8 the synthetic is a **regression / mechanics instrument and never a
claim venue**; these tests pin the properties the designed negatives rely on
(shared input space, exact revisit schedule, known change points, capacity
pressure, the drift-free control) and the two rules decimation must obey
(structure preserved **with counts**, and the assertion FAILS at an ``m`` that
breaks it).
"""

import json
import os

import numpy as np
import pytest

from chlu.experiments.stream_sources import (
    DECIMATION_LADDER,
    assert_structure_preserved,
    decimate,
    label_of,
    load_frozen_stream,
    make_regime_switcher,
    read_benchmark_gate,
    regime_maps,
    select_decimation,
    sha256_of,
    structure_preserved,
    structure_summary,
)


# ---------------------------------------------------------------- synthetic
def test_the_same_x_region_carries_a_different_y_per_regime():
    """The design in one assertion: shared X, regime-dependent label."""
    W = regime_maps(3, 8, 4, seed=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y0, y1 = label_of(X, W[0]), label_of(X, W[1])
    assert np.mean(y0 != y1) > 0.5


def test_the_regime_variable_is_withheld_from_the_features():
    s = make_regime_switcher(seed=0)
    assert s.X.shape[1] == 8
    # the regime is recorded for evaluation and is not a column of X
    for j in range(s.X.shape[1]):
        assert not np.array_equal(s.X[:, j], s.regime.astype(float))


def test_the_revisit_schedule_is_exact_and_its_change_points_are_known():
    s = make_regime_switcher(n_per_stream=16, seed=1)
    assert s.meta["schedule"] == [0, 1, 2, 0, 1, 2]
    assert s.n_streams == 6
    assert s.meta["n_boundaries"] == 5 >= 3  # k >= 3 for L3
    assert s.change_points == [16, 32, 48, 64, 80]
    summ = structure_summary(s)
    assert summ["regime_sequence"] == [0, 1, 2, 0, 1, 2]
    assert summ["n_revisits"] == 3
    assert summ["counts_per_run"] == [16] * 6


def test_since_change_is_the_only_legal_adaptation_axis():
    s = make_regime_switcher(n_per_stream=8, seed=0)
    assert s.since_change[0] == 0
    assert s.since_change[7] == 7
    assert s.since_change[8] == 0  # a change point resets it
    assert len(s.since_change) == len(s)


def test_the_drift_free_control_has_no_change_points():
    s = make_regime_switcher(n_per_stream=8, seed=0, drift_free=True)
    assert s.change_points == []
    assert len(set(s.regime.tolist())) == 1
    assert structure_summary(s)["n_revisits"] == 0


def test_capacity_pressure_is_available():
    s = make_regime_switcher(n_per_stream=32, seed=0)
    assert len(s) == 192  # >> any well budget this wave runs at


def test_the_synthetic_declares_that_it_is_never_a_claim_venue():
    s = make_regime_switcher(seed=0)
    assert "MECHANICS INSTRUMENT" in s.meta["role"]
    assert "A14.8" in s.meta["role"]


def test_seeds_are_threaded():
    a = make_regime_switcher(seed=0, n_per_stream=8)
    b = make_regime_switcher(seed=0, n_per_stream=8)
    c = make_regime_switcher(seed=1, n_per_stream=8)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.X, c.X)


# --------------------------------------------------------------- decimation
@pytest.mark.parametrize("m", DECIMATION_LADDER)
def test_decimation_preserves_the_structure_with_counts(m):
    """⭐ ASSERTED, not claimed (Head ruling 5, condition 2)."""
    s = make_regime_switcher(n_per_stream=40, seed=0)
    before = structure_summary(s)
    after = structure_summary(decimate(s, m))
    assert_structure_preserved(before, after, m)
    assert after["regime_sequence"] == before["regime_sequence"] == [0, 1, 2, 0, 1, 2]
    assert after["n_change_points"] == 5
    assert after["counts_per_run"] == [int(np.ceil(40 / m))] * 6


def test_the_structure_assertion_fails_at_an_m_that_breaks_a_cycle():
    """The guard is load-bearing: at a decimation coarse enough to empty a
    segment the assertion must FAIL, not warn."""
    s = make_regime_switcher(n_per_stream=4, seed=0)
    before = structure_summary(s)
    after = structure_summary(decimate(s, 10))
    assert not structure_preserved(before, after)
    with pytest.raises(AssertionError, match="does not preserve"):
        assert_structure_preserved(before, after, 10)


def test_decimation_refuses_an_m_off_the_registered_ladder():
    s = make_regime_switcher(n_per_stream=8, seed=0)
    with pytest.raises(ValueError, match="ladder"):
        decimate(s, 3)


def test_decimation_travels_in_the_ledger():
    s = decimate(make_regime_switcher(n_per_stream=8, seed=0), 2)
    assert s.meta["decimation_m"] == 2
    assert s.meta["n_instances_pre_decimation"] == 48


def test_decimation_compresses_the_drift_timeline():
    s = make_regime_switcher(n_per_stream=20, seed=0)
    d = decimate(s, 5)
    assert int(np.max(s.since_change)) == 19
    assert int(np.max(d.since_change)) == 3  # per-instance-since-change, compressed


def test_the_selection_rule_picks_the_smallest_m_meeting_the_target():
    sel = select_decimation(wall_s_at_m1=10_000.0, target_s=7_200.0)
    assert sel["selected_m"] == 2
    assert sel["rows"][0]["meets_target"] is False
    sel2 = select_decimation(wall_s_at_m1=1_000.0, target_s=7_200.0)
    assert sel2["selected_m"] == 1


# --------------------------------------------------- the reproduction gate
def _write_csv(tmp_path, n=12):
    p = os.path.join(str(tmp_path), "frozen.csv")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)
    with open(p, "w") as f:
        f.write("f0,f1,f2,label\n")
        for i in range(n):
            f.write(f"{X[i,0]},{X[i,1]},{X[i,2]},{y[i]}\n")
    return p


def test_the_loader_reproduces_the_recorded_digest(tmp_path):
    p = _write_csv(tmp_path)
    digest = sha256_of(p)
    s = load_frozen_stream(p, digest, change_points=[6])
    assert s.meta["sha256"] == digest
    assert len(s) == 12
    assert s.change_points == [6]
    assert s.n_streams == 2


def test_the_loader_refuses_a_file_whose_digest_does_not_match(tmp_path):
    """Reproduce first, consume second: one frozen file, one sha256, all arms."""
    p = _write_csv(tmp_path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_frozen_stream(p, "0" * 64)


def test_the_loader_never_downloads_a_missing_stream(tmp_path):
    with pytest.raises(FileNotFoundError, match="re-freeze"):
        load_frozen_stream(os.path.join(str(tmp_path), "nope.csv"), None)


def test_a_missing_benchmark_gate_is_a_declared_not_run_not_a_crash(tmp_path):
    assert read_benchmark_gate(os.path.join(str(tmp_path), "absent.json")) is None
    p = os.path.join(str(tmp_path), "BENCHMARK-GATE.json")
    with open(p, "w") as f:
        json.dump({"frozen_file": "x.csv", "sha256": "ab" * 32}, f)
    assert read_benchmark_gate(p)["sha256"] == "ab" * 32
