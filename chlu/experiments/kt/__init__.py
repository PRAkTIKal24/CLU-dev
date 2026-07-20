"""
Kosterlitz-Thouless memory-phase measurement suite (Thread-10).

Promoted verbatim from the ``kt-2d-csf3`` laptop run: the physics in
``reduced_xy`` / ``clu_path`` / ``postproc`` is the *validated* physics that
produced ``.claude/outputs/kt-2d-csf3/*.json`` — this package is packaging, not
a rewrite. Round-trip parity against those JSONs is the acceptance gate
(``tests/test_kt.py``).

What it measures
----------------
An ``L x L`` torus of designed SO(2) CLU registers (``channel_spring(kappa)``,
``MexicanHatPotential``, ``newtonian_learned``, no governor) reduces on its
vacuum ring to the 2-D XY model with ``J = 2 kappa r*^2``. Hence:

* the Nelson-Kosterlitz universal jump ``rho_s/T -> 2/pi`` at ``T_KT``;
* ``T_KT = 1.786 kappa r*^2``, i.e. **0.0893 CLU units at kappa=0.05**
  (= 0.8929 J with J = 0.10). ⚠ NOT "0.1786" — that value, which appears in
  some earlier theory notes, is wrong by a factor 2 (retraction carried by
  ``kt-2d-csf3`` §"reconciliation list"; the measurement lands at 0.0898);
* winding survival ``tau ~ L^(pi rho_s/T - 2)`` in 2-D (memory IMPROVES with
  size below ``T_KT``) versus ``tau ~ 1/N`` in 1-D (memory DEGRADES) — the
  memory contrast that is the ML-facing result.

Modes (see ``runner.run_kt``)
-----------------------------
``winding1d``  1-D CLU-ring winding MSD vs N   (JAX, GPU)  -> soft exponent (b)
``winding2d``  2-D winding survival tau vs L   (numpy, CPU) -> soft exponent (a)
``bridge``     CLU-Langevin vs reduced rho_s   (JAX, GPU)  -> kill criterion
``reduced``    reduced-XY phase diagram B/C/D/F (numpy, CPU)
``postproc``   summary.json + figures

⚠ Settings discipline (handover §7.22): every CLU-path mode asserts float64 and
``langevin_noise="fdt"``. The repo default is ``"legacy"``, under which T is
NOT in energy units and none of this physics holds.
"""

from .runner import KT_MODES, run_kt

__all__ = ["KT_MODES", "run_kt"]
