"""⭐ C2W2 Route 2 — the shell-atom race card + the pseudo-Goldstone tilt dial.

Charter Addendum-1 §A4.1/§A4.2, task `.claude/tasks/ssb-shell-atoms.md`. This is
the **basis**-side half of the C2W2 gate: `traj-write-objective` asks the write
*objective* to put information off the endpoint; this module changes the store's
**basis** so the degeneracy is designed in and only its placement is learned.

Four things live here and nothing else:

1. **The arms** (:data:`ARMS`) — ``gauss`` (control = today's store) ·
   ``shell_r0`` (the blocking r=0 regression gate: must equal ``gauss``) ·
   ``shell`` (learned ``r_j``) · ``shell_fixed`` (**designed** ``r``, learned
   placement — the w20 doctrine arm) · ``shell_tilt_<eps>``.
2. **The rig patch** (:func:`shell_rig`) — ⚠ *declared technique*: the cell is
   run by ``exp_memory_gym.run_cell`` **unmodified**, with two symbols
   monkey-patched inside a context manager: ``exp_memory_gym.build_system`` (to
   swap the store's ``.learned`` subtree for shell atoms) and
   ``clu_system.atom_write_mask_fn`` (to extend C3 locality over the shell's two
   extra leaves). Nothing in `traj-write-objective`'s files is edited, and every
   launder / control / monitor / byte-ledger path is **bit-for-bit the rig Route
   1 uses** — which is the only way the two routes are a race and not two
   experiments.
3. **The dial probe** (:func:`dial_probe`) — ``ε`` is a *static* knob, so it can
   be swept on one written store: ``λ_min(ε)``, the curvature hierarchy, the
   soft-mode participation ratio, and ⭐ the **drift timescale ``τ(ε)``**, the
   lifetime dial. Pre-registered in `.claude/outputs/ssb-shell-atoms/PREREG.md`.
4. **The race card** — emitted through ``chlu.eval.race`` (route ``route2``), the
   shared C2W2 schema.

Run: ``python -m chlu.experiments.exp_ssb_shell [--quick]``. There is **no CLI
hook**: ``chlu/cli/experiment_cmd.py`` is `traj-write-objective`'s this wave, so
this module is invoked directly (task §7).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import chlu.core.clu_system as clu_system_mod
import chlu.experiments.exp_memory_gym as gym_exp
from chlu.core.clu_system import _rollout
from chlu.core.shell_atoms import (
    ShellAtomDictionaryPotential,
    shell_potential_from,
    shell_write_mask_fn,
)
from chlu.experiments.memory_gym import PRIMARY_METRIC

ROUTE = "route2"

#: ⭐ The ε grid, PRE-REGISTERED (PREREG §0): >=3 non-zero values spanning >=2
#: decades, with ``eps = 10`` as the **destructive liveness anchor** (it is
#: ~lambda_massive, so it must visibly perturb the store, per the Head's ruling
#: that "a term that never moves anything at any tested setting hasn't been
#: asked; it's been whispered at"). ``0.0`` is the mandatory zero point.
EPS_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

#: The designed shell radius of the ``shell_fixed`` family (declared, not tuned).
R_DESIGNED = 0.5


@dataclass(frozen=True)
class ShellArm:
    """One arm of the race. ``radius_scale = None`` means "shipped Gaussian"."""

    name: str
    radius_scale: Optional[float] = None
    r_init: float = R_DESIGNED
    tilt_eps: float = 0.0
    freeze_radius: bool = False
    freeze_tilt: bool = False
    tilt_weight: str = "envelope"

    @property
    def is_shell(self) -> bool:
        return self.radius_scale is not None

    def as_flags(self) -> Dict[str, Any]:
        return {"arm": self.name, "radius_scale": self.radius_scale,
                "r_init": self.r_init if self.is_shell else None,
                "tilt_eps": self.tilt_eps, "tilt_weight": self.tilt_weight,
                "freeze_radius": self.freeze_radius,
                "freeze_tilt": self.freeze_tilt, "route": ROUTE}


def _arms() -> Dict[str, ShellArm]:
    out = {
        "gauss": ShellArm("gauss"),
        "shell_r0": ShellArm("shell_r0", radius_scale=0.0),
        "shell": ShellArm("shell", radius_scale=1.0),
        "shell_fixed": ShellArm("shell_fixed", radius_scale=1.0, freeze_radius=True),
    }
    for e in EPS_GRID:
        if e == 0.0:
            continue
        out[f"shell_tilt_{e:g}"] = ShellArm(
            f"shell_tilt_{e:g}", radius_scale=1.0, freeze_radius=True, tilt_eps=float(e)
        )
        # the second, q-independent-weight implementation of the same ruling
        out[f"shell_tiltd_{e:g}"] = ShellArm(
            f"shell_tiltd_{e:g}", radius_scale=1.0, freeze_radius=True,
            tilt_eps=float(e), tilt_weight="depth",
        )
    return out


ARMS: Dict[str, ShellArm] = _arms()

#: The three race families. ``manifold`` is the crux (a shell can express what a
#: point cannot); ``overload`` at the shipped atom budget and ``aggregate`` are
#: mandatory because a store change that helps manifolds and destroys addressing
#: is a loss (C1W27's m=4 measured exactly that kind of side-effect).
FAMILIES = ("manifold", "overload", "aggregate")


# ==========================================================================
# the rig patch
# ==========================================================================
def _install_shell(system, arm: ShellArm, tilt_key):
    """Swap the store's Gaussian atoms for shell atoms, in place."""
    V = shell_potential_from(
        system.store.V,
        radius_scale=float(arm.radius_scale),
        r_init=float(arm.r_init),
        tilt_eps=float(arm.tilt_eps),
        tilt_key=tilt_key,
        tilt_weight=str(arm.tilt_weight),
    )
    system.store = eqx.tree_at(lambda s: s.V, system.store, V)
    return system


@contextlib.contextmanager
def shell_rig(arm: ShellArm, *, sink: Optional[List] = None, tilt_seed: int = 0):
    """Run ``exp_memory_gym.run_cell`` on a shell store, editing nothing.

    Two symbols are patched for the duration:

    * ``exp_memory_gym.build_system`` — every system the cell builds (including
      the **blank/empty-store control**, which must share the arm's architecture
      or the control is not a control) gets the shell ``.learned`` subtree.
    * ``chlu.core.clu_system.atom_write_mask_fn`` — the shipped mask covers three
      leaves; a shell store has five, and an unmasked ``radius_raw`` would let a
      write for item *i* move **every** item's radius (asserted in
      ``tests/test_shell_atoms.py::test_shipped_mask_would_leak_the_radius``).
      :func:`chlu.core.shell_atoms.shell_write_mask_fn` degrades to the shipped
      behaviour on a Gaussian store, so the ``gauss`` arm is unaffected.
    """
    orig_build = gym_exp.build_system
    orig_mask = clu_system_mod.atom_write_mask_fn

    def patched_build(cfg, key=None, phi=None, psi=None, loud=True):
        sys_ = orig_build(cfg, key=key, phi=phi, psi=psi, loud=loud)
        if arm.is_shell:
            _install_shell(sys_, arm, jax.random.PRNGKey(tilt_seed)
                           if arm.tilt_eps else None)
        if sink is not None:
            sink.append(sys_)
        return sys_

    def patched_mask(row_mask):
        return shell_write_mask_fn(row_mask, freeze_radius=arm.freeze_radius,
                                   freeze_tilt=arm.freeze_tilt)

    gym_exp.build_system = patched_build
    clu_system_mod.atom_write_mask_fn = patched_mask
    try:
        yield
    finally:
        gym_exp.build_system = orig_build
        clu_system_mod.atom_write_mask_fn = orig_mask


# ==========================================================================
# spectra, the dial and the lifetime
# ==========================================================================
def _sites(system) -> np.ndarray:
    ids, centers, pays = system.codebook()
    ccfg = system.cfg
    z = np.zeros((len(ids), ccfg.dim), dtype=np.float32)
    z[:, : ccfg.addr_dim] = centers
    z[:, ccfg.addr_dim: ccfg.addr_dim + ccfg.payload_dim] = pays
    return z


def _with_eps(system, eps: float):
    """The same written store with a different **static** tilt strength.

    ``ε`` is a static field, so this rebuilds the module (not the parameters):
    every learned array is carried over unchanged, which is what makes the sweep
    a measurement of the DIAL rather than of a different store.
    """
    atoms = system.store.V.learned
    if not isinstance(atoms, ShellAtomDictionaryPotential):
        return None
    dim = int(atoms.centers.shape[1])
    fresh = ShellAtomDictionaryPotential(
        dim, atoms.n_atoms, jax.random.PRNGKey(0), confine=float(atoms.confine),
        n_groups=int(atoms.n_groups), radius_scale=float(atoms.radius_scale),
        tilt_eps=float(eps), tilt_weight=str(atoms.tilt_weight),
    )
    td = atoms.tilt_dir
    if td is None:
        td = jnp.zeros((int(atoms.n_groups), dim)).at[:, dim - 1].set(1.0)
    out = eqx.tree_at(
        lambda t: [t.centers, t.log_width, t.amp, t.radius_raw],
        fresh,
        replace=[atoms.centers, atoms.log_width, atoms.amp, atoms.radius_raw],
    )
    if eps != 0.0:
        out = eqx.tree_at(lambda t: t.tilt_dir, out, td,
                          is_leaf=lambda x: x is None)
    return eqx.tree_at(lambda s: s.V.learned, system.store, out)


def spectra_at_sites(system, store=None) -> Dict[str, Any]:
    """Full ``Hess V`` spectrum at every recorded site + participation ratios.

    ⚠ The gym's ridge failure was diagnosed exactly this way — participation
    **1.000 on an *unstable* mode** — so the sign of ``lambda_min`` and the
    participation are always reported together, never one without the other.
    """
    st = store if store is not None else system.store
    V = system.model(store=st).potential_net
    ccfg = system.cfg
    j = ccfg.addr_dim + ccfg.payload_dim
    atoms = st.V.learned
    tilt = np.asarray(atoms.tilt_dir) if getattr(atoms, "tilt_dir", None) is not None \
        else None
    lam_min, lam_max, lam_2nd, part_spec, part_tilt, grads = [], [], [], [], [], []
    vac: list = []
    for si, z in enumerate(_sites(system)):
        H = np.asarray(jax.hessian(lambda q: V(q))(jnp.asarray(z)), dtype=np.float64)
        H = 0.5 * (H + H.T)
        w, U = np.linalg.eigh(H)
        lam_min.append(float(w[0]))
        lam_2nd.append(float(w[1]))
        lam_max.append(float(w[-1]))
        if ccfg.n_spectator > 0:
            part_spec.append(float(np.sum(U[j: j + ccfg.n_spectator, 0] ** 2)))
        if tilt is not None and si < tilt.shape[0]:
            u = tilt[si] / (np.linalg.norm(tilt[si]) + 1e-12)
            part_tilt.append(float(np.dot(U[:, 0], u) ** 2))
        grads.append(float(np.linalg.norm(
            np.asarray(jax.grad(lambda q: V(q))(jnp.asarray(z))))))
        if tilt is not None:
            vac.append(float(np.asarray(atoms.tilt_vacuum_residual(jnp.asarray(z)))))
    out = {
        "lambda_min": lam_min, "lambda_2nd": lam_2nd, "lambda_max": lam_max,
        "grad_norm": grads,
        "lambda_min_min": float(np.min(lam_min)) if lam_min else float("nan"),
        "lambda_min_median": float(np.median(lam_min)) if lam_min else float("nan"),
        "lambda_max_median": float(np.median(lam_max)) if lam_max else float("nan"),
        "hierarchy_median": (float(np.median(np.asarray(lam_max)
                                             / np.maximum(np.asarray(lam_min), 1e-12)))
                             if lam_min else float("nan")),
        "grad_norm_max": float(np.max(grads)) if grads else float("nan"),
    }
    if part_spec:
        out["softest_spectator_participation"] = part_spec
        out["spectator_participation_median"] = float(np.median(part_spec))
    if part_tilt:
        out["softest_tilt_participation"] = part_tilt
        out["tilt_participation_median"] = float(np.median(part_tilt))
    if vac:
        # ⭐ the pseudo-Goldstone VACUUM CONDITION: 0 => §A4.2's lambda_soft = eps
        # holds exactly; 1/dim => the site is a random orientation w.r.t. its own
        # atoms and the tilt cannot behave as specified.
        out["tilt_vacuum_residual"] = vac
        out["tilt_vacuum_residual_median"] = float(np.median(vac))
        out["tilt_vacuum_random_baseline"] = 1.0 / float(ccfg.dim)
    return out


def drift_timescale(system, store, *, delta: float = 0.1, steps: int = 4000
                    ) -> Dict[str, float]:
    """⭐ ``τ(ε)``: the on-shell payload's drift timescale under the shipped read.

    Displace the settled site along its **softest** eigenvector, roll out under
    the shipped phase-2 friction (``γ_read``, ``dt``), and fit the decay rate of
    the monotone upper envelope of the displacement's component along that mode.

    PREREG P4: the soft mode is a damped oscillator ``q̈ + Γq̇ + λq = 0`` with
    ``Γ = γ_read/dt = 0.4``, so ``τ = Γ/λ`` (slope **−1** in ``ε``) below
    ``ε* = Γ²/4 = 0.04`` and ``τ = 2/Γ = 5.0`` time units (slope **0**) above it
    — a **knee** the charter's pure ``1/ε`` does not predict.
    """
    ccfg = system.cfg
    V = system.model(store=store).potential_net
    z = _sites(system)
    if z.shape[0] == 0:
        return {"tau_time": float("nan"), "censored": True}
    taus, rates = [], []
    for site in z[: min(4, z.shape[0])]:
        H = np.asarray(jax.hessian(lambda q: V(q))(jnp.asarray(site)), dtype=np.float64)
        w, U = np.linalg.eigh(0.5 * (H + H.T))
        v0 = U[:, 0]
        q0 = jnp.asarray((site + delta * v0)[None, :], dtype=jnp.float32)
        p0 = jnp.zeros_like(q0)
        tr = np.asarray(_rollout(system.model(store=store), q0, p0, int(steps),
                                 ccfg.dt, ccfg.gamma_read))[0]
        q = tr[:, : ccfg.dim]
        c = np.abs((q - q[-1][None, :]) @ v0)
        env = np.maximum.accumulate(c[::-1])[::-1]
        c0 = max(float(env[0]), 1e-12)
        t = np.arange(env.size) * float(ccfg.dt)
        sel = (env <= 0.92 * c0) & (env >= 0.03 * c0) & (env > 1e-9)
        if int(sel.sum()) >= 8:
            sl = np.polyfit(t[sel], np.log(env[sel]), 1)[0]
            rate = float(-sl)
            if rate > 1e-9:
                rates.append(rate)
                taus.append(1.0 / rate)
        elif float(env[-1]) > 0.92 * c0:  # censored: fit the whole window
            sl = np.polyfit(t, np.log(np.maximum(env, 1e-12)), 1)[0]
            if -sl > 1e-9:
                rates.append(float(-sl))
                taus.append(float(1.0 / -sl))
    if not taus:
        return {"tau_time": float("nan"), "censored": True, "n_sites": 0}
    return {"tau_time": float(np.median(taus)),
            "tau_steps": float(np.median(taus) / float(ccfg.dt)),
            "rate": float(np.median(rates)), "n_sites": len(taus),
            "censored": bool(np.median(taus) > steps * float(ccfg.dt) * 0.5)}


def dial_probe(system, eps_grid: Sequence[float] = EPS_GRID,
               *, with_lifetime: bool = True) -> Dict[str, Any]:
    """Sweep ``ε`` on ONE written store — the P-B dial-liveness gate.

    ``ε`` is static, so this varies the dial and nothing else: the learned
    centres/widths/amplitudes/radii are the same arrays at every point. The
    registered observables are ``λ_min`` (P2), the hierarchy and participation
    (P3), and ``τ`` (P4).
    """
    rows = []
    for e in eps_grid:
        st = _with_eps(system, float(e))
        if st is None:
            return {"applicable": False,
                    "reason": "the arm's store is not a shell store"}
        row = {"eps": float(e)}
        row.update(spectra_at_sites(system, store=st))
        if with_lifetime:
            row.update({f"tau_{k}": v for k, v in
                        drift_timescale(system, st).items()})
        rows.append(row)
    lam = np.asarray([r["lambda_min_median"] for r in rows], dtype=float)
    eps = np.asarray([r["eps"] for r in rows], dtype=float)
    out: Dict[str, Any] = {"applicable": True, "grid": list(map(float, eps_grid)),
                           "rows": rows}
    nz = eps > 0
    if int(nz.sum()) >= 2 and np.all(np.isfinite(lam[nz])):
        d = lam[nz] - lam[0]
        ok = d > 0
        if int(ok.sum()) >= 2:
            out["kappa_fit"] = float(np.polyfit(eps[nz][ok], d[ok], 1)[0])
            out["loglog_slope_lambda"] = float(
                np.polyfit(np.log10(eps[nz][ok]), np.log10(d[ok]), 1)[0])
        out["lambda_min_at_eps0"] = float(lam[0])
        out["lambda_span"] = float(np.max(lam) - np.min(lam))
    if with_lifetime:
        tau = np.asarray([r.get("tau_tau_time", np.nan) for r in rows], dtype=float)
        m = nz & np.isfinite(tau)
        if int(m.sum()) >= 2:
            out["loglog_slope_tau_all"] = float(
                np.polyfit(np.log10(eps[m]), np.log10(tau[m]), 1)[0])
        lo = m & (eps <= 0.04)
        hi = m & (eps >= 0.1)
        if int(lo.sum()) >= 2:
            out["loglog_slope_tau_below_knee"] = float(
                np.polyfit(np.log10(eps[lo]), np.log10(tau[lo]), 1)[0])
        if int(hi.sum()) >= 2:
            out["loglog_slope_tau_above_knee"] = float(
                np.polyfit(np.log10(eps[hi]), np.log10(tau[hi]), 1)[0])
    return out


def group_geometry(system) -> Dict[str, Any]:
    """⭐ Does a group's designed degeneracy survive SUPERPOSITION?

    One shell has an exactly degenerate minimum set. A *group* of ``atoms_per_item``
    shells at different centres does not: the sum has no exact symmetry unless the
    centres coincide. This measures how far from coincident they are, both raw and
    depth-weighted, next to the shell radius — so "the designed degeneracy is
    destroyed by superposition" is a number rather than an argument.
    """
    atoms = system.store.V.learned
    c = np.asarray(atoms.centers, dtype=float)
    A = np.asarray(atoms.amp, dtype=float) ** 2
    spread, wspread, conc = [], [], []
    for g in range(int(atoms.n_groups)):
        m = np.asarray(atoms.group_rows(g), dtype=bool)
        if m.sum() < 2:
            continue
        cg, ag = c[m], A[m]
        spread.append(float(np.sqrt(np.mean(np.sum((cg - cg.mean(0)) ** 2, -1)))))
        w = ag / max(ag.sum(), 1e-12)
        mu = (w[:, None] * cg).sum(0)
        wspread.append(float(np.sqrt(np.sum(w * np.sum((cg - mu) ** 2, -1)))))
        conc.append(float(1.0 / max(np.sum(w**2) * m.sum(), 1e-12)))
    out = {"n_atoms_per_group": int(atoms.centers.shape[0] // max(atoms.n_groups, 1))}
    if spread:
        out.update({"centre_spread_median": float(np.median(spread)),
                    "depth_weighted_spread_median": float(np.median(wspread)),
                    "depth_participation_median": float(np.median(conc))})
    if isinstance(atoms, ShellAtomDictionaryPotential):
        r = float(np.median(np.asarray(atoms.radii())))
        out["shell_radius_median"] = r
        if spread:
            out["spread_over_radius"] = float(np.median(wspread) / max(r, 1e-9))
    return out


def emit_race_from_json(path: str, out_path: Optional[str] = None) -> Optional[dict]:
    """Re-emit the race card from a saved metrics JSON.

    Exists because the shared card (``chlu.eval.race``, `traj-write-objective`'s
    D0 freeze) may land AFTER the science has run: the cell records carry
    everything the card needs, so the gate evidence is reconstructed without
    re-running a single write.
    """
    race = _race()
    if race is None:
        return None
    with open(path) as fh:
        blob = json.load(fh)
    cells = [to_race_cell(r, race) for r in blob["records"]
             if not r.get("degenerate") and not r.get("error")]
    verdicts = race.score_card(cells)
    blob["race_cells"] = [c.as_dict() for c in cells]
    blob["verdicts"] = [v.as_dict() for v in verdicts]
    blob["coverage"] = race.coverage_table(verdicts)
    blob["gate"] = race.gate_summary(verdicts)
    blob["verdict_markdown"] = race.verdicts_to_markdown(verdicts)
    blob["race_card_available"] = True
    with open(out_path or path, "w") as fh:
        json.dump(blob, fh, indent=2, default=_json_default)
    race.save_cells(os.path.join(os.path.dirname(path), "race_cells_route2.json"),
                    cells)
    return blob


# ==========================================================================
# one cell
# ==========================================================================
def run_shell_cell(family: str, arm_name: str, seed: int = 0, *,
                   gym_overrides: Optional[dict] = None, quick: bool = False,
                   loud: bool = False, dial: bool = False) -> Dict[str, Any]:
    """One race cell: ``exp_memory_gym.run_cell`` on a shell store + the spectra."""
    arm = ARMS[arm_name]
    sink: List[Any] = []
    t0 = time.time()
    with shell_rig(arm, sink=sink, tilt_seed=seed + 4242):
        rec = gym_exp.run_cell(family, arm="base", seed=seed,
                               gym_overrides=gym_overrides, quick=quick, loud=loud)
    rec["shell_arm"] = arm.as_flags()
    rec["route"] = ROUTE
    rec["wall_cell_s"] = time.time() - t0
    system = sink[0] if sink else None
    if system is not None and not rec.get("degenerate"):
        rec["spectra"] = spectra_at_sites(system)
        rec["group_geometry"] = group_geometry(system)
        atoms = system.store.V.learned
        if isinstance(atoms, ShellAtomDictionaryPotential):
            r = np.asarray(atoms.radii())
            rec["radii"] = {"mean": float(r.mean()), "min": float(r.min()),
                            "max": float(r.max()),
                            "per_group_mean": [
                                float(r[np.asarray(atoms.group_rows(g), bool)].mean())
                                for g in range(atoms.n_groups)]}
            rec["shell_byte_ledger"] = atoms.byte_ledger()
        if dial:
            rec["dial"] = dial_probe(system)
    return rec


# ==========================================================================
# the race card
# ==========================================================================
def _race():
    """Import the shared C2W2 race card (``traj-write-objective``'s D0 freeze).

    Imported lazily and reported honestly if absent: this module can produce all
    of its physics without the card, but it cannot emit into the **gate** until
    the freeze commit is on the branch.
    """
    try:
        import chlu.eval.race as race  # noqa: PLC0415
        return race
    except Exception:  # pragma: no cover - only before the freeze commit lands
        return None


def to_race_cell(rec: Dict[str, Any], race, *, liveness: Optional[dict] = None):
    """Map one cell record onto the shared race-card schema (route ``route2``)."""
    family = rec["family"]
    metric = PRIMARY_METRIC[family]
    sc = rec.get("scores", {})
    div = rec.get("dividend", {})
    ledger = rec.get("byte_ledger", {})
    spec = rec.get("spectra", {})
    audit = rec.get("trivial_substitute_audit", {})
    losses = [x for x in rec.get("write_losses", []) if np.isfinite(x)]
    final_loss = float(losses[-1]) if losses else float("nan")
    lam_min = float(spec.get("lambda_min_min", float("nan")))
    grad_max = float(spec.get("grad_norm_max", float("nan")))
    # ⭐ PRE-REGISTERED admissibility convention (PREREG §P5): a designed flat
    # direction pins ``L_min`` at the ``margin`` floor BY CONSTRUCTION, so a loss
    # plateau is not a valid non-convergence test on the shell arms. Convergence
    # is therefore SPECTRAL. The loss is reported either way.
    #
    # ⚠ DECLARED DEVIATION from PREREG §P5, made before the science run and on
    # the smoke run's evidence: PREREG registered ``lambda_min >= -1e-3 AND
    # ||grad V|| < 0.1``. The smoke run measured ``grad_norm_max = 0.1417`` on the
    # SHIPPED Gaussian control itself, so the gradient leg would have excluded
    # every cell of every arm including the control — "admissibility filtering
    # quietly gutting coverage until the gate cannot fire", the exact failure
    # mode the Head named at scoping. The gate-voting rule is therefore the
    # Head's ruling (i) verbatim (``lambda_min < 0`` => inadmissible), which is
    # also ``race.WriteRecord``'s own semantics; the stricter registered rule is
    # reported alongside as ``prereg_strict_admissible`` so both coverages are
    # visible to the Hub rather than argued.
    converged = bool(np.isfinite(lam_min) and lam_min >= -1e-3)
    strict = bool(converged and np.isfinite(grad_max) and grad_max < 0.1)
    reason = ""
    if not converged:
        reason = (f"spectral: lambda_min_min={lam_min:+.4f} < -1e-3 "
                  f"(grad_norm_max={grad_max:.4f})")
    return race.make_cell(
        ROUTE, rec["shell_arm"]["arm"], family, int(rec["seed"]), metric,
        full=float(sc.get("clu", {}).get(metric, float("nan"))),
        settle_deleted_launder=float(sc.get("settle_deleted", {}).get(metric, np.nan)),
        same_keys_null=float(sc.get("same_keys_null", {}).get(metric, np.nan)),
        blank=float(rec.get("blank", {}).get("family_primary_score", np.nan)),
        plus_zero_byte_substitute=float(audit.get("best_zero_byte", np.nan)),
        bytes={"full": int(ledger.get("full_bytes", 0)),
               "launder": int(ledger.get("launder_bytes", 0)),
               "breakdown": {k: int(v) for k, v in
                             dict(ledger.get("breakdown", {})).items()}},
        phi_id="identity(embedded)", phi_bytes=0,
        write={"steps": int(rec.get("clu_config_non_default", {})
                            .get("write_steps", 300)),
               "final_loss": final_loss, "lambda_min_min": lam_min,
               "converged": converged, "plateaued": False, "reason": reason},
        liveness=(liveness or {}),
        monitors={"trips": rec.get("trips", []),
                  "acq": rec.get("self_probe", {}).get("acq", float("nan")),
                  "decode": rec.get("self_probe", {}).get("decode", float("nan")),
                  "n_live": rec.get("n_live", 0)},
        flags={**rec["shell_arm"], **rec.get("clu_config_non_default", {}),
               **rec.get("gym_config_non_default", {}),
               "prereg_strict_admissible": strict,
               "grad_norm_max": grad_max,
               "dividend_from_gym": div.get("dividend", float("nan")),
               "spectator_participation_median":
                   spec.get("spectator_participation_median", float("nan")),
               "tilt_participation_median":
                   spec.get("tilt_participation_median", float("nan")),
               "hierarchy_median": spec.get("hierarchy_median", float("nan")),
               "radii_mean": rec.get("radii", {}).get("mean", float("nan")),
               "shell_byte_overhead_frac":
                   rec.get("shell_byte_ledger", {}).get("overhead_frac", 0.0)},
        seeds_n=1,
        notes=("Route 2 (designed degeneracy). Admissibility is SPECTRAL by "
               "pre-registration: a designed flat direction pins L_min at the "
               "write objective's own margin floor, so a loss plateau is not a "
               "non-convergence test here."),
    )


# ==========================================================================
# the experiment
# ==========================================================================
DEFAULT_ARMS = ("gauss", "shell_r0", "shell", "shell_fixed",
                "shell_tilt_0.01", "shell_tilt_1")
MANIFOLD_ARMS = ("gauss", "shell_r0", "shell", "shell_fixed",
                 "shell_tilt_0.001", "shell_tilt_0.01", "shell_tilt_0.1",
                 "shell_tilt_1", "shell_tilt_10",
                 "shell_tiltd_0.01", "shell_tiltd_1", "shell_tiltd_10")

#: the arms the ε-dial sweep is run on (one per tilt IMPLEMENTATION, seed 0).
DIAL_ARMS = ("shell_fixed", "shell_tiltd_0.01")


def plan(families: Sequence[str] = FAMILIES, seeds: Sequence[int] = (0, 1, 2)
         ) -> List[tuple]:
    out = []
    for fam in families:
        arms = MANIFOLD_ARMS if fam == "manifold" else DEFAULT_ARMS
        for arm in arms:
            for s in seeds:
                out.append((fam, arm, int(s)))
    return out


def run_experiment_ssb_shell(
    families: Sequence[str] = FAMILIES,
    seeds: Sequence[int] = (0, 1, 2),
    arms: Optional[Sequence[str]] = None,
    save_dir: str = ".claude/outputs/ssb-shell-atoms",
    quick: bool = False,
    dial_family: str = "manifold",
    loud: bool = False,
    low_confine_dial: Optional[float] = 0.005,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    race = _race()
    cells: List[Any] = []
    records: List[Dict[str, Any]] = []
    todo = [(f, a, s) for (f, a, s) in plan(families, seeds)
            if arms is None or a in arms]
    t0 = time.time()
    for i, (fam, arm, seed) in enumerate(todo):
        # the dial probe runs once per (family, tilt arm, seed 0) — it is a sweep
        # on ONE written store, so running it on every cell buys nothing.
        want_dial = (fam == dial_family and arm in DIAL_ARMS and seed == 0)
        print(f"[{i + 1}/{len(todo)}] {ROUTE} {fam}/{arm}@s{seed}"
              f"{' +dial' if want_dial else ''}", flush=True)
        try:
            rec = run_shell_cell(fam, arm, seed, quick=quick, loud=loud,
                                 dial=want_dial)
        except Exception as exc:  # a failed cell is reported, never dropped
            rec = {"cell": f"{fam}/{arm}@s{seed}", "family": fam, "seed": seed,
                   "shell_arm": ARMS[arm].as_flags(), "error": repr(exc),
                   "degenerate": True}
            print(f"   ⛔ ERROR {exc!r}", flush=True)
        records.append(rec)
        if race is not None and not rec.get("degenerate") and not rec.get("error"):
            cells.append(to_race_cell(rec, race))

    # ⭐ EXTRA (declared): the overdamped branch of the lifetime law.
    # PREREG P4 predicts τ ∝ ε^-1 only BELOW the knee ε* = Γ²/4 = 0.04. The smoke
    # run measured λ_soft(ε=0) ≈ 0.08 on the shipped store — already ABOVE the
    # knee, because the confinement floors λ_soft at ~2α = 0.10. So on the
    # shipped α the ε dial can only ever sit on the flat branch, and P4a is
    # untestable there. This cell lowers α to 0.005 (floor 2α = 0.01 < ε*) so the
    # −1 branch is reachable. The PREDICTION is unchanged; only the operating
    # point moves, and this addition is disclosed as post-smoke in the report.
    low_dial = None
    if low_confine_dial and not quick and dial_family in families:
        print(f"[extra] low-confinement dial cell (confine={low_confine_dial})",
              flush=True)
        try:
            rec = run_shell_cell(
                dial_family, "shell_fixed", 0, quick=quick, loud=loud, dial=True,
                gym_overrides={"clu_overrides": {"confine": float(low_confine_dial)}})
            low_dial = {"confine": float(low_confine_dial),
                        "dial": rec.get("dial"), "spectra": rec.get("spectra"),
                        "primary": rec.get("scores", {}).get("clu", {}),
                        "note": ("EXTRA cell, NOT a race arm and NOT in the gate: "
                                 "a diagnostic inside the full system (intervention "
                                 "§8.1) to reach the overdamped branch of P4")}
        except Exception as exc:
            low_dial = {"error": repr(exc)}
            print(f"   ⛔ ERROR {exc!r}", flush=True)

    out: Dict[str, Any] = {
        "route": ROUTE,
        "families": list(families), "seeds": list(seeds),
        "arms": sorted({a for _, a, _ in todo}),
        "eps_grid": list(EPS_GRID), "r_designed": R_DESIGNED,
        "n_cells": len(records), "wall_s": time.time() - t0,
        "records": records,
        "low_confine_dial": low_dial,
        "race_card_available": race is not None,
    }
    if race is not None and cells:
        verdicts = race.score_card(cells)
        out["race_cells"] = [c.as_dict() for c in cells]
        out["verdicts"] = [v.as_dict() for v in verdicts]
        out["coverage"] = race.coverage_table(verdicts)
        out["gate"] = race.gate_summary(verdicts)
        out["verdict_markdown"] = race.verdicts_to_markdown(verdicts)
        race.save_cells(os.path.join(save_dir, "race_cells_route2.json"), cells)
    with open(os.path.join(save_dir, "exp_ssb_shell_metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (jnp.ndarray,)):
        return np.asarray(o).tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--save-dir", default=".claude/outputs/ssb-shell-atoms")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--loud", action="store_true")
    ap.add_argument("--dial-family", default="manifold")
    a = ap.parse_args()
    res = run_experiment_ssb_shell(
        families=a.families, seeds=a.seeds, arms=a.arms, save_dir=a.save_dir,
        quick=a.quick, dial_family=a.dial_family, loud=a.loud)
    print(f"\n{res['n_cells']} cells in {res['wall_s'] / 60:.1f} min "
          f"(race card: {'yes' if res['race_card_available'] else 'NOT AVAILABLE'})")
    if "verdict_markdown" in res:
        print(res["verdict_markdown"])


if __name__ == "__main__":
    main()
