"""⭐ C2W2 Route 2 — **designed degeneracy**: shell atoms + the pseudo-Goldstone tilt dial.

Charter Addendum-1 §A4.1/§A4.2. This module is the *basis*-side attack on the
blocker `memory-gym-v0` §3.5 measured: **there are no flat directions in the
shipped store** (`λ_min = 0.0846–0.1000 ≈ 2α = 0.10` at 14/18 sites, so the
settle drives the "unconstrained" spectator axis to zero), and the obvious fix
fails — a multi-target **ridge write produces a SADDLE, not a valley**
(`λ_min = −0.5946`, spectator participation 1.000), because ``write_loss``
minimises ``V`` at each target *independently* and never constrains the
connecting path.

Route 1 (`traj-write-objective`) attacks that from the objective side. This
module attacks it from the **basis** side: a store element whose degeneracy is
*designed in*, with only its placement learned — the w20 doctrine ("designed
mechanism + learned use beats learned-everything") applied to symmetry.

The shell atom
--------------
Charter §A4.1, verbatim::

    V = α‖q‖²  −  Σ_j A_j · exp( −(‖q − c_j‖ − r_j)² / (2 s_j² + 1e−9) )

with ``A_j = amp_j²`` (the shipped squared parameterisation — see
:class:`~chlu.core.memory_potentials.AtomDictionaryPotential` for *why* it is
squared and not ``softplus``) and a **learnable softplus radius**
``r_j = radius_scale · softplus(radius_raw_j)``.

⛔ **The r = 0 regression gate (charter §A4.1, blocking).** At ``r_j ≡ 0`` the
shell atom must reduce to the shipped Gaussian atom **exactly** — not "to
tolerance". That is a statement about IEEE754 arithmetic, and it is the reason
the shell displacement is evaluated as

    u² = d2 − 2·ρ·r + r²      (ρ = sqrt(d2 + 1e−12),  d2 = ‖q − c‖²)

rather than as the algebraically identical ``(ρ − r)²``: at ``r = 0`` the first
form is ``d2 − 0.0 + 0.0``, which is ``d2`` bit-for-bit, while ``sqrt`` followed
by squaring is **not**. The ``+1e−12`` under the sqrt keeps ``∂ρ/∂q`` finite at
``q = c`` so that the reverse-mode contribution ``−2r·(diff/ρ)`` is an exact
``0.0`` rather than ``0 · NaN``. :mod:`tests.test_shell_atoms` asserts the
identity on ``V``, ``∇V``, ``Hess V`` and on a *written* store.

The tilt dial
-------------
Charter §A4.2 (the pseudo-Goldstone ruling): *designed flat directions carry the
payload and are NEVER zeroed; a small tilt ``ε`` makes ``λ_min = ε > 0``*, which
restores implicit-gradient conditioning, terminates settles, and is itself the
**manifold-payload lifetime dial** (drift timescale ∝ 1/ε). The implemented form
is rank-1, **per group**, envelope-weighted and group-normalised::

    V_tilt(q) = (ε/2) · Σ_j w_j(q) · ( û_{o(j)} · (q − c_j) )² ,
    w_j = g_j / (Σ_{k ∈ group(j)} g_k + 1e−6),   g_j = the atom's own envelope,
    û_g = u_g / ‖u_g‖   (LEARNED, one direction per item group)

Two properties are load-bearing and are the reason for the normalisation:

1. At a site ``z`` on the shell with ``û·(z − c_j) ≈ 0`` every product-rule term
   carries a factor ``û·(q − c_j)`` **except** one, so ``Hess V_tilt(z) = ε ûûᵀ``
   — the constant of proportionality between ``λ_soft`` and ``ε`` is **1 by
   construction**, which is what makes §A4.2's "λ_min = ε" a *testable* claim
   rather than a proportionality with a free constant.
2. The normalisation is **per group**, not global, so the tilt cannot couple one
   item's atoms to another's — the C3 locality of the masked write survives.

⚠ **What this module does NOT claim.** A shell atom is *not* exactly degenerate
inside the shipped confinement. Stationarity on the shell gives a residual
tangential curvature ``λ_tan = 2α‖c‖/ρ`` — the confinement tilts the shell before
any ``ε`` is added. That is derived in ``.claude/outputs/ssb-shell-atoms/PREREG.md``
(P2d) and measured in the report; it is a property of the route, not of the code.

Byte ledger (charter §A4.3 / gym PREREG-B1)
-------------------------------------------
Per atom the shell stores ``dim + 3`` floats (centres, ``log_width``, ``amp``,
``radius_raw``) against the Gaussian's ``dim + 2`` — **+1/(dim+2)**, i.e. +14.3 %
at ``dim = 5`` and +12.5 % at ``dim = 6``. The tilt adds ``dim`` floats **per
group** (rank-1, group-shared), which is < 0.5 % at the shipped atom budgets.
:meth:`ShellAtomDictionaryPotential.byte_ledger` returns the split so the ledger
travels with every arm, the launder's included.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.memory_potentials import AtomDictionaryPotential, _atom_group_owner

#: Added under the sqrt so ``∂‖q−c‖/∂q`` stays finite at ``q = c``. It never
#: enters the ``r = 0`` value path (it only multiplies ``r``), which is what
#: keeps the regression gate bit-exact.
RHO_EPS = 1e-12

#: Floor on the per-group envelope normaliser (a group whose atoms are all far
#: from ``q`` contributes no tilt rather than 0/0).
TILT_NORM_EPS = 1e-6


@lru_cache(maxsize=64)
def _owner_onehot(n_atoms: int, n_groups: int) -> np.ndarray:
    """(n_atoms, n_groups) one-hot of the contiguous block partition.

    Cached because it is a trace-time constant: the partition is fixed by
    ``(n_atoms, n_groups)`` and matches
    :meth:`AtomDictionaryPotential.group_rows` exactly.
    """
    owner = _atom_group_owner(int(n_atoms), int(n_groups))
    out = np.zeros((int(n_atoms), int(n_groups)), dtype=np.float32)
    out[np.arange(int(n_atoms)), owner] = 1.0
    return out


class ShellAtomDictionaryPotential(eqx.Module):
    """The charter §A4.1 shell atom, a drop-in for :class:`AtomDictionaryPotential`.

    It carries the *same three* leaf names (``centers``/``log_width``/``amp``)
    and the same ``n_groups``/``group_rows``/``n_atoms`` surface, because the
    shipped harness reaches into those directly (``LearnedVStore.group_stats``,
    ``scale_group_amplitude``, ``reinit_group``,
    ``memory_potentials.atom_write_mask_fn``, ``train_memory.atom_crowding_penalty``).
    The two additions are ``radius_raw`` (the shell) and ``tilt_dir`` (the dial).

    Args:
        radius_scale: **static** multiplier on ``softplus(radius_raw)``. ``0.0``
            makes ``r_j`` exactly zero and the potential exactly the shipped
            Gaussian atom (the r=0 regression gate); ``1.0`` is the shell.
        r_init: initial radius (``radius_raw`` is set to ``softplus⁻¹(r_init)``).
        tilt_eps: **static** ε, the explicit-symmetry-breaking dial. ``0.0``
            disables the term entirely (not merely multiplies it by zero), so a
            no-tilt arm carries **no tilt bytes and no tilt FLOPs**.
        tilt_key: PRNG key for the learned tilt directions; ``None`` with
            ``tilt_eps > 0`` initialises them on the LAST coordinate axis (the
            spectator axis in the harness's layout), which is a *declared*
            designed init, not a learned result.
    """

    centers: jnp.ndarray  # (n_atoms, dim)
    log_width: jnp.ndarray  # (n_atoms,)
    amp: jnp.ndarray  # (n_atoms,) -> amp**2 -> depth
    radius_raw: jnp.ndarray  # (n_atoms,) -> radius_scale * softplus -> r_j
    tilt_dir: Optional[jnp.ndarray]  # (n_groups, dim) or None
    confine: float = eqx.field(static=True)
    n_groups: int = eqx.field(static=True)
    radius_scale: float = eqx.field(static=True)
    tilt_eps: float = eqx.field(static=True)
    axis_width_scale: Optional[tuple] = eqx.field(static=True, default=None)

    def __init__(
        self,
        dim: int,
        n_atoms: int,
        key: jax.random.PRNGKey,
        init_scale: float = 1.0,
        init_width: float = 0.3,
        confine: float = 0.05,
        depth_init: float = 1e-4,
        n_groups: int = 1,
        group_centers=None,
        local_radius: float = 0.0,
        axis_width_scale=None,
        *,
        radius_scale: float = 1.0,
        r_init: float = 0.5,
        tilt_eps: float = 0.0,
        tilt_key: Optional[jax.random.PRNGKey] = None,
    ):
        base = AtomDictionaryPotential(
            dim,
            n_atoms,
            key,
            init_scale=init_scale,
            init_width=init_width,
            confine=confine,
            depth_init=depth_init,
            n_groups=n_groups,
            group_centers=group_centers,
            local_radius=local_radius,
            axis_width_scale=axis_width_scale,
        )
        self.centers = base.centers
        self.log_width = base.log_width
        self.amp = base.amp
        self.confine = base.confine
        self.n_groups = base.n_groups
        self.axis_width_scale = base.axis_width_scale
        self.radius_scale = float(radius_scale)
        self.tilt_eps = float(tilt_eps)
        raw0 = float(np.log(np.expm1(max(float(r_init), 1e-6))))
        self.radius_raw = jnp.full((int(n_atoms),), raw0, dtype=self.centers.dtype)
        if self.tilt_eps == 0.0:
            self.tilt_dir = None
        else:
            g = max(1, int(n_groups))
            if tilt_key is None:
                d = jnp.zeros((g, int(dim)), dtype=self.centers.dtype)
                self.tilt_dir = d.at[:, int(dim) - 1].set(1.0)
            else:
                self.tilt_dir = jax.random.normal(
                    tilt_key, (g, int(dim)), dtype=self.centers.dtype
                )

    # -- construction from an existing Gaussian store ----------------------
    @classmethod
    def from_gaussian(
        cls,
        atoms: AtomDictionaryPotential,
        *,
        radius_scale: float = 1.0,
        r_init: float = 0.5,
        tilt_eps: float = 0.0,
        tilt_key: Optional[jax.random.PRNGKey] = None,
    ) -> "ShellAtomDictionaryPotential":
        """Wrap an *already constructed* Gaussian atom dictionary.

        This is the constructor the experiment uses, because it makes the
        ``gauss`` and ``shell_*`` arms share their init **bit-for-bit**: the
        centres/widths/amplitudes are the same arrays, so any difference between
        the arms is the shell, never the RNG stream.
        """
        n_atoms, dim = int(atoms.centers.shape[0]), int(atoms.centers.shape[1])
        obj = cls(
            dim,
            n_atoms,
            jax.random.PRNGKey(0),
            confine=float(atoms.confine),
            n_groups=int(atoms.n_groups),
            axis_width_scale=atoms.axis_width_scale,
            radius_scale=radius_scale,
            r_init=r_init,
            tilt_eps=tilt_eps,
            tilt_key=tilt_key,
        )
        return eqx.tree_at(
            lambda t: [t.centers, t.log_width, t.amp],
            obj,
            replace=[atoms.centers, atoms.log_width, atoms.amp],
        )

    # -- the AtomDictionaryPotential surface the harness reaches into ------
    @property
    def n_atoms(self) -> int:
        return int(self.centers.shape[0])

    def group_rows(self, group: int) -> jnp.ndarray:
        """Boolean row mask (n_atoms,) selecting the atoms owned by ``group``."""
        n = self.n_atoms
        mask = np.zeros((n,), dtype=bool)
        if 0 <= group < self.n_groups:
            lo = round(group * n / self.n_groups)
            hi = round((group + 1) * n / self.n_groups)
            mask[lo:hi] = True
        return jnp.asarray(mask)

    # -- the shell ---------------------------------------------------------
    def radii(self) -> jnp.ndarray:
        """``r_j = radius_scale · softplus(radius_raw_j)`` — exactly ``0`` when
        ``radius_scale == 0`` (the regression-gate configuration)."""
        return self.radius_scale * jax.nn.softplus(self.radius_raw)

    def __call__(self, q: jnp.ndarray) -> float:
        s = jnp.exp(self.log_width)
        raw = q[None, :] - self.centers
        diff = raw
        if self.axis_width_scale is not None:
            diff = diff / jnp.asarray(self.axis_width_scale, dtype=diff.dtype)[None, :]
        d2 = jnp.sum(diff**2, axis=-1)
        denom = 2.0 * s**2 + 1e-9
        # ⛔ bit-exact r=0 reduction, in TWO layers.
        #
        # (1) Arithmetic: the shell displacement is written as `d2 − 2ρr + r²`,
        #     NOT as `(ρ − r)²`, so at `r = 0` every op is `d2 − 0.0 + 0.0` and
        #     `V`, `∇V`, `Hess V` come out bit-identical to the shipped Gaussian.
        #     MEASURED, and it holds in eager mode to exactly 0.
        # (2) ⚠ Graph: that is NOT sufficient under `eqx.filter_jit`. XLA fuses
        #     and reassociates the enlarged expression, and the *parameter*
        #     gradient then differs by ~1 ULP (measured: 9.5e-7 in `centers`
        #     after 60 Adam steps, forward pass still bit-identical). Since
        #     `radius_scale` is STATIC, the `r ≡ 0` arm takes the shipped code
        #     path at trace time, so the emitted HLO is identical and the
        #     *written store* is bit-identical too — which is the leg of the
        #     charter's gate that the arithmetic alone cannot carry.
        if self.radius_scale == 0.0:
            u2 = d2
        else:
            r = self.radii()
            rho = jnp.sqrt(d2 + RHO_EPS)
            u2 = d2 - 2.0 * rho * r + r * r
        env = jnp.exp(-u2 / denom)
        depth = self.amp**2
        v = -jnp.sum(depth * env)
        v = v + self.confine * jnp.sum(q**2)
        if self.tilt_eps != 0.0 and self.tilt_dir is not None:
            v = v + self._tilt(raw, env)
        return v

    def _tilt(self, raw: jnp.ndarray, env: jnp.ndarray) -> jnp.ndarray:
        """``(ε/2) Σ_j w_j (û_{o(j)}·(q − c_j))²`` with per-group normalised ``w``."""
        oh = jnp.asarray(_owner_onehot(self.n_atoms, self.n_groups), dtype=raw.dtype)
        u = self.tilt_dir / (
            jnp.linalg.norm(self.tilt_dir, axis=-1, keepdims=True) + 1e-12
        )
        u_per_atom = oh @ u  # (n_atoms, dim)
        proj = jnp.sum(raw * u_per_atom, axis=-1)  # (n_atoms,)
        grp = env @ oh  # (n_groups,) per-group envelope mass
        w = env / (oh @ (grp + TILT_NORM_EPS))
        return 0.5 * self.tilt_eps * jnp.sum(w * proj**2)

    # -- ledger ------------------------------------------------------------
    def byte_ledger(self) -> dict:
        """Float32 byte split: shipped-atom bytes vs the shell/tilt surcharge."""
        n, dim = self.n_atoms, int(self.centers.shape[1])
        gauss = n * (dim + 2) * 4
        shell = n * 4
        tilt = 0 if self.tilt_dir is None else int(np.asarray(self.tilt_dir).size) * 4
        return {
            "gauss_bytes": int(gauss),
            "shell_bytes": int(shell),
            "tilt_bytes": int(tilt),
            "total_bytes": int(gauss + shell + tilt),
            "overhead_frac": float((shell + tilt) / gauss),
        }


def shell_write_mask_fn(row_mask: jnp.ndarray, *, freeze_radius: bool = False,
                        freeze_tilt: bool = False):
    """Update mask for a shell store — the C3-locality guard, extended.

    :func:`chlu.core.memory_potentials.atom_write_mask_fn` masks exactly three
    leaves (``centers``/``log_width``/``amp``). A shell store has two more
    (``radius_raw``, ``tilt_dir``); leaving them unmasked would let a write for
    item ``i`` move **every** item's radius, which is precisely the C3-locality
    the masked write exists to provide. This mask covers all five.

    Args:
        freeze_radius: zero the radius updates entirely — the ``shell_fixed``
            arm (**designed** degeneracy, learned placement: the w20 doctrine).
        freeze_tilt: likewise for the tilt directions.
    """
    m = jnp.asarray(row_mask, dtype=jnp.float32)

    def apply(updates):
        atoms = updates.learned
        where, replace = [], []
        where += [
            lambda u: u.learned.centers,
            lambda u: u.learned.log_width,
            lambda u: u.learned.amp,
        ]
        replace += [
            atoms.centers * m[:, None],
            atoms.log_width * m,
            atoms.amp * m,
        ]
        if getattr(atoms, "radius_raw", None) is not None:
            where.append(lambda u: u.learned.radius_raw)
            replace.append(
                jnp.zeros_like(atoms.radius_raw) if freeze_radius
                else atoms.radius_raw * m
            )
        if getattr(atoms, "tilt_dir", None) is not None:
            g = int(atoms.tilt_dir.shape[0])
            gm = jnp.asarray(
                (np.asarray(_owner_onehot(int(m.shape[0]), g)).T
                 @ np.asarray(row_mask, dtype=np.float32)) > 0.0,
                dtype=jnp.float32,
            )
            where.append(lambda u: u.learned.tilt_dir)
            replace.append(
                jnp.zeros_like(atoms.tilt_dir) if freeze_tilt
                else atoms.tilt_dir * gm[:, None]
            )
        return eqx.tree_at(lambda u: [w(u) for w in where], updates, replace=replace)

    return apply


def shell_potential_from(V: eqx.Module, **kwargs) -> eqx.Module:
    """Swap a ``DesignFreedomPotential``'s ``.learned`` Gaussian atoms for shells.

    This is the ``store_potential_factory`` entry point: it takes the store's
    *already built* potential and returns the same object with a shell
    ``.learned`` subtree carrying **identical** centres/widths/amplitudes, so a
    ``shell_r0`` arm is bit-identical to ``gauss`` by construction rather than by
    coincidence. ``kwargs`` are forwarded to
    :meth:`ShellAtomDictionaryPotential.from_gaussian`.
    """
    atoms = getattr(V, "learned", None)
    if not isinstance(atoms, AtomDictionaryPotential):
        raise TypeError(
            "shell_potential_from expects a potential whose .learned is an "
            f"AtomDictionaryPotential, got {type(atoms).__name__}"
        )
    shell = ShellAtomDictionaryPotential.from_gaussian(atoms, **kwargs)
    return eqx.tree_at(lambda t: t.learned, V, replace=shell)


def shell_hessian_spectrum(V, z, tilt_dir=None, axis: Optional[int] = None
                           ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Eigen-decomposition of ``Hess V`` at ``z`` plus the participation ratios.

    Returns ``(eigenvalues ascending, eigenvectors (columns), info)`` where
    ``info`` carries the softest mode's participation on the **designed shell
    coordinate** (``tilt_dir``) and on a named coordinate ``axis`` (the harness's
    spectator axis). The gym's ridge failure was diagnosed exactly this way —
    participation 1.000 on an *unstable* mode — so both halves are reported.
    """
    H = np.asarray(jax.hessian(lambda q: V(q))(jnp.asarray(z)), dtype=np.float64)
    H = 0.5 * (H + H.T)
    w, U = np.linalg.eigh(H)
    v0 = U[:, 0]
    info = {
        "lambda_min": float(w[0]),
        "lambda_max": float(w[-1]),
        "hierarchy": float(w[-1] / w[0]) if abs(w[0]) > 1e-12 else float("inf"),
    }
    if tilt_dir is not None:
        u = np.asarray(tilt_dir, dtype=np.float64)
        u = u / (np.linalg.norm(u) + 1e-12)
        info["participation_tilt"] = float(np.dot(v0, u) ** 2)
    if axis is not None:
        info["participation_axis"] = float(v0[int(axis)] ** 2)
    return w, U, info


__all__ = [
    "ShellAtomDictionaryPotential",
    "shell_write_mask_fn",
    "shell_potential_from",
    "shell_hessian_spectrum",
    "RHO_EPS",
    "TILT_NORM_EPS",
]
