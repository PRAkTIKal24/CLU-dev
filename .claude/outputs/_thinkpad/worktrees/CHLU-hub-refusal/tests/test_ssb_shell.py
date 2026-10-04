"""Tests for the ⭐ **Route-1 x Route-2 2x2** seam in ``exp_ssb_shell``.

C2W2 declared the 2x2 a NOT-RUN because the shell rig's monkey-patched
``build_system`` **swallowed the write-objective seam** — its signature had no
``write_objective``, so the crossed cell could not even be constructed. These
tests pin the fix: the seam is forwarded verbatim, it is *live* (a mis-spelled
coefficient raises rather than masquerading as an inert term), and the shipped
arm is untouched when it is absent.
"""

import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.experiments.exp_ssb_shell import (
    ARMS,
    TWO_BY_TWO_STORES,
    TWO_BY_TWO_WRITES,
    _2x2_aggregate,
    shell_rig,
)


def _cfg(**kw):
    base = dict(seed=0, capacity=4, address_steps=40, read_steps=60, traj_stride=8,
                atoms_per_item=8, min_atoms=32, min_atoms_base=32, write_steps=5,
                n_query_per_item=2)
    base.update(kw)
    return CluSystemConfig(**base)


@pytest.mark.parametrize("arm_name", ["gauss", "shell", "shell_r0"])
def test_the_shell_rig_forwards_the_write_objective_seam(arm_name):
    """⛔ The C2W2 blocker, in one assertion: the patched builder must accept and
    forward ``write_objective`` or the 2x2 is unconstructable."""
    spec = {"loss_kwargs": {"lambda_path": 0.3, "path_kwargs": {"n_interp": 7}}}
    sink = []
    with shell_rig(ARMS[arm_name], sink=sink):
        import chlu.experiments.exp_memory_gym as gym_exp

        sys_ = gym_exp.build_system(_cfg(), key=None, loud=False,
                                    write_objective=spec)
    assert sink and sys_ is sink[0]
    assert sys_.write_objective["loss_kwargs"]["lambda_path"] == 0.3


def test_the_seam_is_LIVE_through_the_rig_a_bad_spec_raises():
    """The Route-1 D0 freeze property must survive the monkey-patch: a spec key
    that is silently dropped would report as *"the term is inert"*, which is
    exactly the finding a gate must never fabricate."""
    with shell_rig(ARMS["shell"]):
        import chlu.experiments.exp_memory_gym as gym_exp

        with pytest.raises(ValueError, match="unknown write-objective key"):
            gym_exp.build_system(_cfg(), key=None, loud=False,
                                 write_objective={"loss_kwrags":
                                                  {"lambda_path": 0.3}})


def test_absent_write_objective_is_the_shipped_write_bit_for_bit():
    """The 2x2's control column must BE the shipped anchor, not a re-spelling."""
    a = build_system(_cfg(), key=None, loud=False)
    sink = []
    with shell_rig(ARMS["gauss"], sink=sink):
        import chlu.experiments.exp_memory_gym as gym_exp

        b = gym_exp.build_system(_cfg(), key=None, loud=False)
    assert a.write_objective == b.write_objective          # both empty/None
    ca = np.asarray(a.store.V.learned.centers)
    cb = np.asarray(b.store.V.learned.centers)
    assert np.array_equal(ca, cb)                           # bitwise


def test_the_2x2_grid_is_the_declared_one():
    """The grid is pre-registered (``PREREG.md`` §5): {gauss, shell} x
    {endpoint, path@0.3}. ``shell`` has a LEARNED radius — the arm the
    hypothesis is about."""
    assert TWO_BY_TWO_STORES == ("gauss", "shell")
    assert TWO_BY_TWO_WRITES["endpoint"] is None
    assert (TWO_BY_TWO_WRITES["path@0.3"]["loss_kwargs"]["lambda_path"]
            == pytest.approx(0.3))
    assert ARMS["shell"].freeze_radius is False
    assert ARMS["shell_r0"].radius_scale == 0.0


def test_the_2x2_aggregate_uses_sample_sd():
    cells = [{"family": "f", "store": "gauss", "write": "endpoint", "seed": s,
              "dividend": d, "full": 1.0, "radii_mean": None, "admissible": True}
             for s, d in enumerate([-0.10, -0.20, -0.30])]
    agg = _2x2_aggregate(cells)["f/gauss/endpoint"]
    assert agg["dividend_mean"] == pytest.approx(-0.20)
    assert agg["dividend_sd"] == pytest.approx(np.std([-0.1, -0.2, -0.3], ddof=1))
    assert agg["dividend_se"] == pytest.approx(agg["dividend_sd"] / np.sqrt(3))
    assert agg["n_admissible"] == 3
