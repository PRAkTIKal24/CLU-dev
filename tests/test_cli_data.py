"""Smoke tests for the `chlu data` CLI commands.

These call the command functions directly (fast, no subprocess) and assert the
saved `.npz` contents have the expected q/p split shapes. Regression guard for
the broken `generate_figure8_data` / `generate_sine_data` imports (see §7.1).
"""

from types import SimpleNamespace

import numpy as np

from chlu.cli.data_cmd import cmd_data_figure8, cmd_data_sine


def test_cli_data_figure8(tmp_path):
    """`chlu data figure8` runs and saves q/p with matching shapes."""
    out = tmp_path / "f8.npz"
    args = SimpleNamespace(steps=50, dt=None, output=out)

    rc = cmd_data_figure8(args)
    assert rc == 0
    assert out.exists()

    data = np.load(out)
    # figure-8 state is [x, y, vx, vy] -> q=[x,y], p=[vx,vy]
    assert data["q"].shape == (50, 2)
    assert data["p"].shape == (50, 2)
    assert data["trajectory"].shape == (50, 4)
    assert np.all(np.isfinite(data["trajectory"]))


def test_cli_data_sine(tmp_path):
    """`chlu data sine` runs and saves q/p with matching shapes."""
    out = tmp_path / "sine.npz"
    args = SimpleNamespace(n_waves=4, steps=50, dt=None, output=out)

    rc = cmd_data_sine(args)
    assert rc == 0
    assert out.exists()

    data = np.load(out)
    # sine state is [x, dx/dt] -> q=[x], p=[dx/dt]
    assert data["q"].shape == (4, 50, 1)
    assert data["p"].shape == (4, 50, 1)
    assert data["waves"].shape == (4, 50, 2)
    assert np.all(np.isfinite(data["waves"]))
