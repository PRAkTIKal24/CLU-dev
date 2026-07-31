"""⭐ **B′'s rival memories** — minimal faithful reimplementations for the audit.

`.claude/tasks/bprime-rivals.md` D1/D2, charter ADDENDUM 3 §A14.2. The audit
paper's question is *"when does test-time dynamics buy anything over a table at
matched bytes?"*, asked **the same way for every family**:

* :mod:`~chlu.eval.rivals.ttt` — TTT-Linear / TTT-MLP (arXiv:2407.04620,
  Eqs. 1/2/4/5, §2.4 ``b=16``, §2.7 learnable ``W_0``/``eta``, LN + residual);
* :mod:`~chlu.eval.rivals.deltanet` — DeltaNet (Eq. 5), Gated DeltaNet (Eq. 6) and
  ⭐ **Gated DeltaNet-2** (arXiv:2605.22791, **Eq. 10** with Eqs. 8/9/11/12), the
  §A14.2 **reference** delta-rule arm;
* :mod:`~chlu.eval.rivals.ledger` — the **two-sided byte ledger** and the
  learned-initial-state rule, applied to the rivals **and to the CLU**;
* :mod:`~chlu.eval.rivals.fit` — one outer loop and one five-arm protocol for all
  of them, because uniformity across families *is* the deliverable.

⛔ **Not built here, declared NOT-RUN with reason** (never as nulls): **Titans**
(no official code, chunk size never given a numeric value, no seeds ⇒ an arm would
be our reconstruction audited against our reconstruction's table) and **Sparse
Delta Memory** (official code needs Torch >= 2.8 / Triton >= 3.4 / SM 80+ and
cannot run on this machine). Both enter B′ as **positioning**.
"""

from chlu.eval.rivals.deltanet import DELTA_VARIANTS, DeltaMemory
from chlu.eval.rivals.fit import (
    DEFAULT_BUDGET_FLOATS,
    LR_GRID,
    RIVALS,
    FitExample,
    fit_best_of_grid,
    fit_rival,
    make_rival,
    rival_arms,
    table_budget,
)
from chlu.eval.rivals.ledger import (
    TTT_MINI_BATCH,
    LedgerError,
    TwoSidedLedger,
    assert_identical_phi,
    clu_two_sided_ledger,
    head_width_for_budget,
    matched_table_rows,
    phi_row,
    table_ledger,
)
from chlu.eval.rivals.ttt import TTTMemory, measured_state_floats

__all__ = [
    "DELTA_VARIANTS", "DeltaMemory", "TTTMemory", "measured_state_floats",
    "TwoSidedLedger", "LedgerError", "matched_table_rows", "table_ledger",
    "head_width_for_budget", "clu_two_sided_ledger", "phi_row",
    "assert_identical_phi", "TTT_MINI_BATCH",
    "RIVALS", "FitExample", "make_rival", "fit_rival", "fit_best_of_grid",
    "rival_arms", "table_budget", "DEFAULT_BUDGET_FLOATS", "LR_GRID",
]
