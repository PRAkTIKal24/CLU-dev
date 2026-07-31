"""Tests for B′'s **two-sided byte ledger** and the structural ledger identity.

The two things under test are the two things a referee will attack:

1. the **learned-initial-state rule** — the init is parameters, only the deviation
   is state — applied identically to the rivals and to the CLU;
2. **T1's ledger identity as integers** (`bprime-rivals` D7 / theorist C3):
   ``full == 4[N_at(D+2) + Kd]`` and ``launder == 4K(d+m)``. A float ratio check
   passes on a store whose leaf structure has silently changed; an integer
   identity does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from chlu.eval.dividend import (
    ByteAccount,
    LedgerIdentityError,
    assert_ledger_identity,
    ledger_identity,
)
from chlu.eval.rivals.ledger import (
    TTT_MINI_BATCH,
    LedgerError,
    TwoSidedLedger,
    assert_identical_phi,
    head_width_for_budget,
    matched_table_rows,
    phi_row,
    table_ledger,
)


# --------------------------------------------------------------------------
# the pre-registered sizing rule
# --------------------------------------------------------------------------
def test_iso_state_head_widths_match_the_prereg_arithmetic():
    """⛔ Filed in `.claude/outputs/bprime-rivals/PREREG.md` §1.4 BEFORE any run.

    Budget = the CLU's banked ``aggregate@base`` 5456 B = 1364 float32.
    """
    assert head_width_for_budget("ttt_linear", 1364) == 29   # 29^2 + 16*29 = 1305
    assert head_width_for_budget("ttt_mlp", 1364) == 12      # 8*12^2 + 16*12 = 1344
    assert head_width_for_budget("delta", 1364) == 36        # 36^2 = 1296
    # and each is the LARGEST that fits: one more must not
    assert 30 ** 2 + 16 * 30 > 1364
    assert 8 * 13 ** 2 + 16 * 13 > 1364
    assert 37 ** 2 > 1364


def test_head_width_rejects_unknown_kind():
    with pytest.raises(ValueError):
        head_width_for_budget("mamba2", 1364)


def test_matched_table_rows_is_a_floor_not_a_round():
    # 1296 floats / (36+36) = exactly 18 rows
    assert matched_table_rows(1296, 36, 36) == 18
    # a budget one float short must NOT buy the row
    assert matched_table_rows(1295, 36, 36) == 17
    assert matched_table_rows(0, 4, 1) == 0


# --------------------------------------------------------------------------
# the ledger object
# --------------------------------------------------------------------------
def test_ledger_breakdowns_must_sum_to_the_totals():
    ok = TwoSidedLedger(arm="x", param_floats=10, state_floats=4,
                        param_breakdown={"a": 6, "b": 4},
                        state_breakdown={"s": 4})
    assert ok.check() is ok
    assert ok.param_bytes == 40 and ok.state_bytes == 16
    assert ok.state_over_param == pytest.approx(0.4)
    bad = TwoSidedLedger(arm="x", param_floats=10, state_floats=4,
                         param_breakdown={"a": 6, "b": 3},
                         state_breakdown={"s": 4})
    with pytest.raises(LedgerError):
        bad.check()


def test_table_ledger_is_byte_exact():
    t = table_ledger(18, 36, 36)
    assert t.state_floats == 18 * 72 == 1296
    assert t.state_bytes == 1296 * 4


# --------------------------------------------------------------------------
# the identical-phi invariant: raises, never warns
# --------------------------------------------------------------------------
def test_identical_phi_raises_on_a_deliberate_mismatch():
    from chlu.core.psi_readout import PhiMismatchError

    q0 = np.arange(12.0).reshape(4, 3)
    keys = np.arange(8.0).reshape(4, 2)
    a = phi_row(q0, keys)
    assert a["phi_bytes"] == 0 and a["phi_params"] == 0
    assert assert_identical_phi({"full": a, "launder": dict(a)}) == a["phi_id"]
    b = phi_row(q0 + 1e-9, keys)
    with pytest.raises(PhiMismatchError):
        assert_identical_phi({"full": a, "rival": b})


# --------------------------------------------------------------------------
# T1's ledger identity, on a real store (D7 / theorist C3)
# --------------------------------------------------------------------------
class _FakeLearned:
    def __init__(self, atoms, dim):
        self.centers = np.zeros((atoms, dim))


class _FakeV:
    def __init__(self, atoms, dim):
        self.learned = _FakeLearned(atoms, dim)


class _FakeStore:
    def __init__(self, atoms, dim, per_atom):
        self.V = _FakeV(atoms, dim)
        self._n = 4 * atoms * per_atom

    def n_bytes(self):
        return self._n


class _FakeSystem:
    def __init__(self, atoms, dim, per_atom):
        self.store = _FakeStore(atoms, dim, per_atom)


def test_ledger_identity_reproduces_the_banked_aggregate_cell():
    """The banked ``aggregate@base`` cell: 5456 B full / 100 B launder / 54.56x."""
    sysm = _FakeSystem(atoms=192, dim=5, per_atom=7)   # D = 4+1+0, D+2 = 7
    keys = np.zeros((5, 4))
    pays = np.zeros((5, 1))
    idn = ledger_identity(sysm, keys, pays)
    assert idn["full_expected"] == 4 * (192 * 7 + 5 * 4) == 5456
    assert idn["launder_expected"] == 4 * 5 * 5 == 100
    assert idn["ratio_corrected"] == pytest.approx(5456 / 100)
    out = assert_ledger_identity(sysm, keys, pays,
                                 account=ByteAccount(full_bytes=5456,
                                                     launder_bytes=100))
    assert out["ok"] is True


def test_ledger_identity_is_the_CORRECTED_law_at_n_spectator_1():
    """⭐ The corrected law ``[A(D+2)+d]/(d+m)`` is exact at ``n_spectator = 1``,
    and ``byte_ratio_law`` now agrees with it.

    **History (C2W3→C2W4, kept so the tripwire's purpose survives its firing).**
    This test was written by ``bprime-rivals`` while the shipped
    ``byte_ratio_law`` still divided by the *store* dim ``D`` where the launder
    row is ``(d+m)``, so it missed every ``n_spectator > 0`` cell by
    **+8.6667** (24 of 28, not 28 of 28 — the published *"verified to 1e-9 in
    all 28 cells"* was wrong, in the **conservative** direction). It was
    deliberately written to assert that live disagreement so that
    ``harness-debt``'s P0 fix would **flip a test rather than pass silently**.
    It did exactly that at the C2W4 integration merge, and the Hub flipped the
    assertion here. ⛔ Do not re-loosen this to an inequality: the two forms
    must now agree at every ``n_spectator``.
    """
    from chlu.experiments.memory_gym import byte_ratio_law

    atoms, k, d, m, spec = 192, 6, 4, 1, 1
    dim = d + m + spec
    sysm = _FakeSystem(atoms=atoms, dim=dim, per_atom=dim + 2)
    idn = ledger_identity(sysm, np.zeros((k, d)), np.zeros((k, m)))
    corrected = (atoms / k * (dim + 2) + d) / (d + m)
    assert idn["ratio_corrected"] == pytest.approx(corrected)
    shipped = byte_ratio_law(atoms / k, addr_dim=d, payload_dim=m, n_spectator=spec)
    assert shipped == pytest.approx(corrected)          # the C2W4 fix, asserted
    # and the pre-fix formula is what it is NOT: D in the denominator, not (d+m)
    pre_fix = (atoms / k) * (dim + 2) / dim + d / dim
    assert corrected - pre_fix == pytest.approx(8.6667, abs=1e-3)


def test_ledger_identity_raises_when_the_store_drifts():
    sysm = _FakeSystem(atoms=192, dim=5, per_atom=8)     # a 4th leaf appeared
    with pytest.raises(LedgerIdentityError):
        assert_ledger_identity(sysm, np.zeros((5, 4)), np.zeros((5, 1)),
                               account=ByteAccount(full_bytes=192 * 8 * 4 + 80,
                                                   launder_bytes=100))


def test_byte_floor_is_2_20_and_rises_to_2_40_at_one_spectator():
    """``bprime-theory`` T1: the floor is the byte price of ONE privately
    deletable atom per item, and it RISES at ``n_spectator = 1``."""
    def floor(d, m, spec):
        D = d + m + spec
        return ((D + 2) + d) / (d + m)

    assert floor(4, 1, 0) == pytest.approx(2.20)
    assert floor(4, 1, 1) == pytest.approx(2.40)


def test_ttt_mini_batch_is_the_papers_own_value():
    assert TTT_MINI_BATCH == 16
