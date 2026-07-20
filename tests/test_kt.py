"""KT memory-phase suite tests (Thread-10 packaging; chlu/experiments/kt/).

The contract (task acceptance): promoting the validated ``kt-2d-csf3`` laptop
scripts into the tracked tree must be **packaging, not a rewrite**. So:

  (1) the numerics ROUND-TRIP bit-exactly against the committed laptop results
      — same seeds, same sizing, same numbers (the acceptance gate),
  (2) the settings guards FIRE: float64 + langevin_noise="fdt" +
      newtonian_learned + no governor, so a misconfigured CSF3 run dies loudly
      instead of silently producing garbage (handover §7.22),
  (3) the sweep-grid / array-sharding bookkeeping is consistent,
  (4) the retracted T_KT value "0.1786" is nowhere asserted (the correct value
      is 1.786*kappa*r*^2 = 0.0893 CLU units at kappa=0.05).

The round-trip reference values in (1) are inlined from
``.claude/outputs/kt-2d-csf3/{reduced_xy,kt_winding_msd,kt_clu}.json`` (commit
e3c8931). They are inlined deliberately: ``.claude/**`` is gitignored, so the
test must not depend on it being present.

x64 enabled at import, process-global (same pattern and caveat as
test_goldstone.py / test_lattice.py).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from chlu.config import get_default_config  # noqa: E402
from chlu.experiments.kt import KT_MODES  # noqa: E402
from chlu.experiments.kt.clu_path import (  # noqa: E402
    assert_kt_settings,
    run_winding_msd,
)
from chlu.experiments.kt.reduced_xy import run_cell, run_winding  # noqa: E402
from chlu.experiments.kt.runner import cells  # noqa: E402


# --------------------------------------------------------------- (1) parity --
def test_winding2d_round_trip_exact():
    """2-D winding survival reproduces the laptop D_winding rows exactly."""
    # reduced_xy.json D_winding, TJ=1.30, seed 700, nwalk=24, n_max=6000
    ref = {8: (16.0, 18.833333333333332), 12: (27.0, 28.916666666666668),
           16: (29.0, 31.0)}
    for L, (tau_med, tau_mean) in ref.items():
        got = run_winding(L=L, TJ=1.30, seed=700, nwalk=24, n_max=6000)
        assert got["tau_med"] == tau_med, f"L={L} tau_med drifted"
        assert got["tau_mean"] == pytest.approx(tau_mean, rel=0, abs=0), (
            f"L={L} tau_mean drifted"
        )
        assert got["censored"] == 0


def test_reduced_rho_s_round_trip_exact():
    """Reduced-XY rho_s (section B) reproduces the laptop aggregate exactly."""
    # reduced_xy.json B_rho_s, L=8, TJ=0.60, seeds 100-102, nwalk=4, 1500/4000/5
    rows = [run_cell(8, 0.60, s, 4, 1500, 4000, 5) for s in (100, 101, 102)]
    rf = np.array([r["rho_fluct"] for r in rows])
    assert rf.mean() == pytest.approx(0.8244828012941149, rel=0, abs=1e-15)
    assert rf.std() / np.sqrt(3) == pytest.approx(
        0.00041260235470954687, rel=0, abs=1e-15
    )


@pytest.mark.parametrize(
    "N,ref_rate,ref_msd", [(8, 6.54966062238462e-05, 1.39453125)]
)
def test_winding1d_msd_round_trip_exact(N, ref_rate, ref_msd):
    """The real CLU-path (JAX) 1-D winding MSD reproduces the laptop exactly."""
    got = run_winding_msd(N=N, TJ=1.0, n_chunks=300, CH=100, seed=31)
    assert got["slip_rate_msd"] == pytest.approx(ref_rate, rel=1e-12)
    assert got["msd_final"] == pytest.approx(ref_msd, rel=0, abs=0)
    # default = full-range fit (no diffusive-window cut) -> legacy behaviour
    assert got["msd_fit_max"] is None
    assert got["fit_points"] == 300


def test_msd_fit_window_shortens_the_fit():
    """The diffusive-window cut must actually restrict the fit, and the default
    must remain the (bit-exact) full-range fit.

    Guards the saturation trap: at T/J=1.0 the winding decorrelates in ~1e3
    steps, so a full-range fit measures saturation, not a slip rate.
    """
    full = run_winding_msd(N=8, TJ=1.0, n_chunks=300, CH=100, seed=31)
    cut = run_winding_msd(
        N=8, TJ=1.0, n_chunks=300, CH=100, seed=31, msd_fit_max=0.3
    )
    assert cut["fit_points"] < full["fit_points"]
    assert cut["fit_tmax"] < full["fit_tmax"]
    # saturation biases the full-range rate DOWN
    assert cut["slip_rate_msd"] > full["slip_rate_msd"]


# ------------------------------------------------------- (2) settings guards --
def test_assert_kt_settings_accepts_the_validated_configuration():
    assert_kt_settings("fdt", "newtonian_learned", use_governor=False)


@pytest.mark.parametrize(
    "noise,kinetic,gov,needle",
    [
        ("legacy", "newtonian_learned", False, "fdt"),
        ("fdt", "relativistic", False, "newtonian_learned"),
        ("fdt", "newtonian_learned", True, "governor"),
    ],
)
def test_assert_kt_settings_rejects_misconfiguration(noise, kinetic, gov, needle):
    """A misconfigured CSF3 run must die loudly, not silently make garbage."""
    with pytest.raises(RuntimeError, match=needle):
        assert_kt_settings(noise, kinetic, use_governor=gov)


def test_kt_config_defaults_are_the_validated_physics():
    kt = get_default_config().experiment_kt
    assert kt.langevin_noise == "fdt"  # NOT the repo-default "legacy" (§7.22)
    assert kt.kinetic_mode == "newtonian_learned"
    assert kt.kappa == 0.05 and kt.rstar == 1.0
    # J = 2 kappa r*^2 = 0.10 and T_KT = 1.786 kappa r*^2 = 0.0893 CLU units.
    assert 2 * kt.kappa * kt.rstar**2 == pytest.approx(0.10)
    assert 1.786 * kt.kappa * kt.rstar**2 == pytest.approx(0.0893)
    # the retracted value, guarded explicitly
    assert 1.786 * kt.kappa * kt.rstar**2 != pytest.approx(0.1786)


# ---------------------------------------------------------- (3) sweep grids --
def test_sweep_grids_and_shard_indices():
    kt = get_default_config().experiment_kt
    assert set(KT_MODES) == {
        "winding1d", "winding2d", "bridge", "reduced", "postproc"
    }
    assert cells("winding1d", kt) == [{"N": n} for n in kt.winding1d_n_values]
    grid2d = cells("winding2d", kt)
    assert len(grid2d) == len(kt.winding2d_tj_values) * len(kt.winding2d_l_values)
    # every cell is addressable by exactly one array task id, no duplicates
    assert len({tuple(sorted(c.items())) for c in grid2d}) == len(grid2d)
    assert cells("postproc", kt) == []


def test_run_kt_rejects_out_of_range_task_id(tmp_path):
    from chlu.experiments.kt import run_kt

    config = get_default_config()
    n = len(cells("bridge", config.experiment_kt))
    with pytest.raises(IndexError, match="out of range"):
        run_kt(config, mode="bridge", out_dir=tmp_path, task_id=n)


def test_run_kt_rejects_unknown_mode(tmp_path):
    from chlu.experiments.kt import run_kt

    with pytest.raises(ValueError, match="mode must be one of"):
        run_kt(get_default_config(), mode="nope", out_dir=tmp_path)


# ------------------------------------------------------------ (4) postproc ---
def test_postproc_merges_shards_and_survives_missing_sections(tmp_path):
    """Array shards fold into the canonical file; absent sections are skipped."""
    import json

    from chlu.experiments.kt.postproc import postprocess

    for k, (TJ, L, tau) in enumerate([(1.10, 32, 90.0), (1.10, 64, 85.0)]):
        (tmp_path / f"winding2d_task{k}.json").write_text(
            json.dumps(
                {
                    "meta": {"mode": "winding2d"},
                    "D_winding": [
                        {"L": L, "TJ": TJ, "tau_med": tau, "tau_mean": tau,
                         "censored": 0, "nwalk": 24, "n_max": 6000}
                    ],
                }
            )
        )
    summary = postprocess(tmp_path, make_figures=False, log=lambda *_: None)
    assert (tmp_path / "reduced_xy.json").exists()
    merged = json.loads((tmp_path / "reduced_xy.json").read_text())
    assert len(merged["D_winding"]) == 2
    # slope of a flat-ish tau(L) is small — the L^2 masking dying out
    assert summary["winding_2d"][1.10]["loglog_slope"] == pytest.approx(
        np.log(85.0 / 90.0) / np.log(64 / 32)
    )
    # no B_rho_s section present -> no crash, no jump keys
    assert "crossings_TJ" not in summary
