"""Spectral band-selection recipe (V3.2 / P10): choose inertial-mass bands
FROM DATA instead of from an oracle prior.

Motivation (critique V3.2 / CM-5): the banded lattice's win is real, but the
bands were matched to the ground-truth data timescales by hand. This module
is the cheapest defensible answer to "where would the bands come from without
the oracle?" — a spectral estimate of each unit's timescale.

In the two-timescale composite task each unit's timescale is its dominant
oscillation frequency: unit i's position channels orbit at
``omega_i = sqrt(k_i / M_i)``. With a SHARED curvature k (the F5 §5.3 role-3
geometry: M redistributes one shared landscape's timescales), the inertial-mass
band is recoverable up to a global scale from the per-unit dominant frequency:

    M_i  proportional to  1 / omega_i^2      (shared-stiffness assumption)

We estimate ``omega_i`` per unit by a trajectory-averaged periodogram (FFT) or
by the first zero-crossing of the autocorrelation of that unit's position
channels, then map to a mass band normalized to unit geometric mean. A global
mass rescale is a gauge choice at linear order (F5 §5 "honest deflation") — only
the RATIO between units is physical, so the selector targets the ratio, not the
absolute magnitude.
"""

from typing import List, Sequence

import numpy as np


def _unit_q_channels(data: np.ndarray, n_units: int) -> List[np.ndarray]:
    """Split the concatenated [q_1..q_N, p_1..p_N] layout into per-unit
    POSITION blocks. ``data`` is (..., seq_len, 2*D) with D = sum of unit dims;
    here every unit is dim-2 so D = 2*n_units and unit i owns q-channels
    [2i, 2i+1]."""
    data = np.asarray(data)
    total = data.shape[-1]
    if total % 2 != 0:
        raise ValueError(f"state dim {total} is not even (expected [q; p])")
    D = total // 2
    if D % n_units != 0:
        raise ValueError(f"position dim {D} not divisible by n_units {n_units}")
    d_unit = D // n_units
    q = data[..., :D]
    return [q[..., i * d_unit : (i + 1) * d_unit] for i in range(n_units)]


def estimate_unit_frequencies(
    data: np.ndarray,
    n_units: int,
    dt: float,
    method: str = "fft",
) -> np.ndarray:
    """Estimate each unit's dominant ANGULAR frequency omega_i (rad / time).

    Args:
        data: trajectories of shape (n_traj, seq_len, 2*D) or (seq_len, 2*D),
            concatenated [q; p] layout (CLULattice convention).
        n_units: number of units N (each dim-2 here).
        dt: sampling step of ``data``.
        method: "fft" (trajectory-averaged periodogram peak) or "autocorr"
            (first autocorrelation zero-crossing -> quarter period).

    Returns:
        omega: (n_units,) estimated angular frequencies, ascending unit order.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[None]
    blocks = _unit_q_channels(data, n_units)  # each (n_traj, T, d_unit)
    seq_len = data.shape[-2]
    omegas = np.empty(n_units, dtype=np.float64)

    if method == "fft":
        # rfft frequency grid in Hz (cycles / time); drop the DC bin.
        freqs = np.fft.rfftfreq(seq_len, d=dt)
        for i, blk in enumerate(blocks):
            # remove per-trajectory, per-channel mean so DC does not dominate
            blk = blk - blk.mean(axis=1, keepdims=True)
            spec = np.abs(np.fft.rfft(blk, axis=1)) ** 2  # (n_traj, F, d_unit)
            psd = spec.mean(axis=(0, 2))  # average power over traj + channels
            psd[0] = 0.0  # kill residual DC
            peak = int(np.argmax(psd))
            omegas[i] = 2.0 * np.pi * freqs[peak]
    elif method == "autocorr":
        for i, blk in enumerate(blocks):
            blk = blk - blk.mean(axis=1, keepdims=True)
            # normalized autocorrelation averaged over traj + channels
            T = blk.shape[1]
            ac = np.zeros(T)
            for lag in range(T):
                ac[lag] = np.mean(np.sum(blk[:, : T - lag] * blk[:, lag:], axis=1))
            ac = ac / max(ac[0], 1e-12)
            # first zero crossing = quarter period -> omega = pi / (2 * t_zc)
            zc = np.nonzero(ac <= 0.0)[0]
            if zc.size:
                t_zc = zc[0] * dt
                omegas[i] = np.pi / (2.0 * max(t_zc, 1e-9))
            else:
                omegas[i] = np.nan
    else:
        raise ValueError(f"unknown method {method!r} (use 'fft' or 'autocorr')")
    return omegas


def select_mass_bands(
    data: np.ndarray,
    n_units: int,
    dt: float,
    method: str = "fft",
    normalize: str = "geomean",
    ref_scale: float = 1.0,
) -> List[float]:
    """Data-driven inertial-mass band M_i proportional to 1 / omega_i^2.

    Args:
        data, n_units, dt, method: forwarded to ``estimate_unit_frequencies``.
        normalize: "geomean" (band geometric mean == ref_scale; matches the
            F5 gauge convention used by the oracle band [4.0, 0.25], geomean 1),
            "max" (heaviest unit == ref_scale), or "none" (raw 1/omega^2).
        ref_scale: target scale for the chosen normalization.

    Returns:
        mass_scales: list of per-unit inertial-mass scales, ascending unit
        order — a drop-in stand-in for ``ExperimentLatticeConfig.banded_mass_scales``.
    """
    omega = estimate_unit_frequencies(data, n_units, dt, method=method)
    if np.any(~np.isfinite(omega)) or np.any(omega <= 0.0):
        raise ValueError(f"non-positive / non-finite frequency estimate: {omega}")
    raw = 1.0 / (omega**2)
    if normalize == "geomean":
        g = float(np.exp(np.mean(np.log(raw))))
        scaled = raw / g * ref_scale
    elif normalize == "max":
        scaled = raw / float(np.max(raw)) * ref_scale
    elif normalize == "none":
        scaled = raw
    else:
        raise ValueError(f"unknown normalize {normalize!r}")
    return [float(x) for x in scaled]


def mismatch_band(
    matched: Sequence[float],
    kind: str = "anti",
) -> List[float]:
    """Construct a mis-banded control with the SAME log-spread as ``matched``.

    Args:
        matched: the oracle/matched band (ascending unit order).
        kind: "anti" reverses the unit->mass assignment (same spread, inverted
            ordering); "shuffle_common" moves the whole spread into the common
            mode (every unit at the matched geometric mean — a zero-differential
            "orthogonal" band that spends no budget on the timescale axis; only
            distinguishable from uniform by its global scale).

    Returns:
        the mis-banded mass_scales (same length as ``matched``).
    """
    m = [float(x) for x in matched]
    if kind == "anti":
        return list(reversed(m))
    if kind == "shuffle_common":
        g = float(np.exp(np.mean(np.log(np.asarray(m)))))
        return [g for _ in m]
    raise ValueError(f"unknown kind {kind!r} (use 'anti' or 'shuffle_common')")
