"""Tests for the spectral band-selection recipe (V3.2 / P10)."""

import math

import jax
import numpy as np
import pytest

from chlu.data.band_selection import (
    estimate_unit_frequencies,
    mismatch_band,
    select_mass_bands,
)
from chlu.data.two_timescale_orbits import generate_two_timescale_orbits


def _default_data(seq_len=256):
    return np.asarray(
        generate_two_timescale_orbits(
            jax.random.PRNGKey(0),
            n_traj=64,
            seq_len=seq_len,
            dt=0.05,
            omegas=(0.5, 2.0),
            masses=(4.0, 0.25),
            radius=1.0,
        )
    )


@pytest.mark.parametrize("method", ["fft", "autocorr"])
def test_estimate_frequencies_recovers_omega(method):
    """Per-unit dominant frequency ~ ground-truth omega [0.5, 2.0]."""
    omega = estimate_unit_frequencies(
        _default_data(), n_units=2, dt=0.05, method=method
    )
    assert omega.shape == (2,)
    # FFT bin resolution on a 256-step / dt=0.05 window is ~0.078 Hz (~0.49
    # rad); the slow unit sits at ~1 cycle, so allow a generous absolute band.
    assert abs(omega[0] - 0.5) < 0.2
    assert abs(omega[1] - 2.0) < 0.2
    assert omega[1] > omega[0]  # fast unit resolved above slow unit


@pytest.mark.parametrize("method", ["fft", "autocorr"])
def test_selector_recovers_oracle_band(method):
    """M_i proportional 1/omega_i^2 recovers the oracle band [4.0, 0.25]."""
    band = select_mass_bands(_default_data(), n_units=2, dt=0.05, method=method)
    assert len(band) == 2
    # geomean-normalized band should be close to the oracle [4.0, 0.25]
    np.testing.assert_allclose(band, [4.0, 0.25], rtol=0.25)
    # geometric mean is pinned to 1.0 (gauge convention)
    assert math.isclose(math.sqrt(band[0] * band[1]), 1.0, rel_tol=1e-6)


def test_selector_four_units_monotone():
    """At N=4 the selector reproduces a monotone 16x-ratio band from data."""
    R, N = 16.0, 4
    masses = [math.sqrt(R) * (1.0 / R) ** (i / (N - 1)) for i in range(N)]
    omegas = [1.0 / math.sqrt(m) for m in masses]
    data = np.asarray(
        generate_two_timescale_orbits(
            jax.random.PRNGKey(1),
            n_traj=64,
            seq_len=512,
            dt=0.05,
            omegas=tuple(omegas),
            masses=tuple(masses),
            radius=1.0,
        )
    )
    band = select_mass_bands(data, n_units=4, dt=0.05, method="fft")
    assert np.all(np.diff(band) < 0)  # strictly descending, like true masses
    np.testing.assert_allclose(band, masses, rtol=0.2)


def test_mismatch_band_same_spread():
    """anti-matched preserves the log-spread; common-mode zeroes it."""
    matched = [4.0, 0.25]
    anti = mismatch_band(matched, "anti")
    assert anti == [0.25, 4.0]
    # same absolute log-spread, inverted ordering
    assert math.isclose(abs(np.diff(np.log(anti))[0]), abs(np.diff(np.log(matched))[0]))
    common = mismatch_band(matched, "shuffle_common")
    assert math.isclose(common[0], common[1])  # zero differential
    assert math.isclose(math.sqrt(common[0] * common[1]), 1.0, rel_tol=1e-6)
