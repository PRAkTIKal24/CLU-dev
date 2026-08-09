"""⭐ Experiment PHI-GEOMETRY (C2W8 pass 3, wt2) — **build the ``phi_dim -> addr_dim``
map, then MEASURE whether strong ``phi`` actually separates.**

⛔ **DIAL DECLARATION.** Dial: **none — encoder/geometry instrumentation.** No claim
cell, no performance number, no verdict. Laundering control: N/A (nothing is scored) —
**but the byte ledger is mandatory**, because the map's parameters land on the ledger of
**every** arm *including the launder*. Falsifies: nothing this module claims; its NO-GO
**re-labels** the spine, it does not cancel it (Head ruling R4). Every reading here is
labelled **non-promotable** (N94).

**The defect this closes.** ``exp_well_lifecycle.PhiAddress`` forces
``phi_dim = addr_dim`` (``cl.phi_dim = int(w.addr_dim)``) and then TRUNCATES
(``out[:, :addr_dim] = f[:, :addr_dim] * scale``). **There is no projection.** So
"strong ``phi`` at ``addr_dim = 8``" is today either a *weak 8-dim encoder refit at
d = 8* — not the encoder that was built and priced — or *8 of 256 coordinates*, throwing
248 away. Neither is the ``phi`` that measured 0.16080 -> 0.31912 on Split-CIFAR-10.
:class:`chlu.experiments.phi_encoders.PhiProjection` is the map; this module measures
what the map buys.

**What is measured** (Head ruling R1: on **Split-CIFAR-10**, where the encoders were
built and priced — ⛔ the pass-1/2 census geometry is MNIST and is **not** a baseline
here; the weak-``phi`` reference is re-measured *in this run, at matching* ``d``):

per seed, per ``d in {8, 12, 16}``, per arm — **median-NN key spacing**,
**``sigma_q`` / spacing**, **``d_safe`` / spacing**, the achieved **atom budget**, and
the fill-uniformity of the key cloud inside the address ball.

⚠ **The scale-invariance guard (``PREREG-C2W8-PASS3`` §4).** The rig normalises
addresses to unit radius (``scale = 1 / r95``) while ``sigma_q = 0.15`` is absolute, so
with ``n`` items in a unit ``d``-ball the spacing is **essentially geometric** — a
property of ``(n, d)`` and of how uniformly ``phi`` fills the ball. **Strong ``phi``
cannot enlarge the volume; it can only spread items more uniformly inside it.** Every
quantity here is therefore a **dimensionless ratio with the scale stated**, and ``n`` is
declared on every row (it is never implicit).

⛔ **``(d, atom budget)`` is ONE joint dial** (§A4.3): ``n_atoms = round(512 *
sqrt(2)**d)`` costs 8 192 / 32 768 / 131 072 atoms at d = 8/12/16 and **1.7e41 at
d = 256**, so naive 256-dim addressing is *forbidden by the atom law* and the feasible
band is exactly ``{8, 12, 16}``.

The registered mechanical reading (``PREREG-C2W8-PASS3`` §5), computed by
:func:`geometry_verdict`, never argued at review:

    **GO** iff strong ``phi`` improves ``sigma_q`` / spacing over the PCA reference **at
    the same d**, beyond noise, over >= 3 seeds. **NO-GO** otherwise.

Runnable::

    uv run python -m chlu.experiments.exp_phi_geometry --quick
    chlu exp-phi-geometry [--project N] [--seeds 0,1,2] [--quick]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional

import jax
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.experiments.exp_cl_entry import (
    PHI_PRIMARY,
    RingBufferKNN,
    apply_cifar10,
    build_cl_stream,
    build_phi,
)
from chlu.experiments.phi_encoders import PhiProjection, ProjectedReadIn

#: read-only import of the rig's own address embedding — the object the spine uses.
#: ⭐ Composing it with a :class:`ProjectedReadIn` is exactly how the truncation is
#: retired: the projected ``phi`` is ALREADY ``addr_dim``-dimensional, so
#: ``f[:, :addr_dim]`` is the identity. :func:`assert_no_truncation` asserts it.
from chlu.experiments.exp_well_lifecycle import PhiAddress  # noqa: E402

#: the atom law (``CluSystemConfig.n_atoms``): ``round(min_atoms_base * c**d)``
ATOM_BASE = 512
ATOM_C = float(np.sqrt(2.0))


# ---------------------------------------------------------------------------
# the joint dial: (d, atom budget) — priced BEFORE anything is run
# ---------------------------------------------------------------------------
def atom_budget(addr_dim: int, capacity: int = 16, atoms_per_item: int = 32,
                min_atoms: int = 384) -> int:
    """``max(atoms_per_item*K, min_atoms, round(512 * sqrt(2)**d))`` — the shipped law.

    Reproduced here (rather than imported through a built system) so the budget can be
    **priced before** a cell is spent, which Head ruling R3 requires at ``d = 16``.
    """
    geo = round(ATOM_BASE * ATOM_C ** int(addr_dim))
    return int(max(atoms_per_item * int(capacity), int(min_atoms), int(geo)))


def atom_budget_table(dims=(4, 8, 12, 16, 20, 24, 256), **kw) -> List[Dict[str, Any]]:
    """The feasibility table. ⛔ ``d = 256`` is 1.7e41 atoms — forbidden, not expensive."""
    return [{"addr_dim": int(d), "n_atoms": atom_budget(int(d), **kw)} for d in dims]


# ---------------------------------------------------------------------------
# the rig: stream, phi, the declared map, the address scale
# ---------------------------------------------------------------------------
def cl_config(config: CHLUConfig, arm: str, phi_dim: int):
    """The CL-entry config this measurement drives, at ``(arm, phi_dim)``.

    ⚠ The stream itself depends on **none** of ``(arm, phi_dim)`` — only on the
    dataset/protocol knobs — so the stream is bit-identical across arms at a seed and
    every arm comparison here is **paired**. :func:`stream_fingerprint` checks it.
    """
    g = config.experiment_phi_geometry
    c2 = copy.deepcopy(config)
    if g.dataset == "cifar10":
        apply_cifar10(c2)
    cl = c2.experiment_cl_entry
    cl.dataset = g.dataset
    cl.phi_arm = str(arm)
    cl.phi_dim = int(phi_dim)
    cl.enc_steps = int(g.enc_steps)
    cl.n_fit_region = int(g.n_fit_region)
    cl.n_fit_pool = int(g.n_fit_pool)
    cl.phi_regimes = [g.phi_regime]
    return cl


def stream_fingerprint(stream) -> str:
    """SHA-256 (first 16 hex) over the stream arrays — the pairing evidence."""
    import hashlib

    h = hashlib.sha256()
    for k in ("train_X", "train_y", "test_X", "test_y"):
        for a in stream[k]:
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(stream["fit_pool_task1_only"])).tobytes())
    return h.hexdigest()[:16]


def offer_order_keys(stream, n: int, n_offer_per_task: int) -> np.ndarray:
    """The images the census would OFFER, in the order it offers them.

    ⚠ ``n`` is a declared quantity, not an implicit one: with ``n`` items in a unit
    ``d``-ball the spacing is essentially geometric in ``(n, d)``, so a spacing quoted
    without its ``n`` is not comparable to anything.

    ⚠ **The population is CAPPED** at ``n_tasks * n_offer_per_task`` — the census offers
    no more than that — so a requested ``n`` above the cap silently becomes the cap.
    :func:`achieved_n_grid` resolves the request against the cap so a row can never
    claim an ``n`` it did not have.
    """
    out = []
    for t in range(len(stream["train_X"])):
        Xt = np.asarray(stream["train_X"][t], np.float32)
        out.append(Xt[: int(n_offer_per_task)])
        if sum(len(a) for a in out) >= int(n):
            break
    X = np.concatenate(out, axis=0)
    return X[: int(n)]


def achieved_n_grid(requested, n_available: int) -> List[int]:
    """The requested key counts, clipped to what the offer order actually supplies,
    **de-duplicated** — so the same population is never reported twice under two
    different ``n`` labels."""
    return sorted({int(min(int(n), int(n_available))) for n in requested if int(n) >= 3})


def build_arm(config: CHLUConfig, stream, seed: int, arm: str, phi_dim: int,
              addr_dim: int, projection: str, cache: Optional[dict] = None):
    """Build ``phi`` at ``phi_dim`` and the declared ``phi_dim -> addr_dim`` map.

    ⚠ **The map is fitted under the same ``task1_only`` regime as every other ``phi``
    in this programme** — on ``stream["fit_pool_task1_only"]``, which is disjoint from
    every stream item and drawn from task-1 classes only. No leakage from unseen tasks.
    """
    g = config.experiment_phi_geometry
    key = (str(arm), int(phi_dim))
    if cache is not None and key in cache:
        phi, prov = cache[key]
    else:
        cl = cl_config(config, arm, phi_dim)
        phi, prov = build_phi(g.phi_regime or PHI_PRIMARY, stream, cl, seed)
        if cache is not None:
            cache[key] = (phi, prov)
    pool = np.asarray(stream[f"fit_pool_{g.phi_regime}"], np.float32)
    if int(phi_dim) == int(addr_dim) and projection in ("identity", None):
        proj = PhiProjection(np.asarray(phi(pool[:1])), addr_dim, form="identity")
    else:
        proj = PhiProjection(np.asarray(phi(pool), np.float32), int(addr_dim),
                             form=str(projection), seed=int(seed))
    return ProjectedReadIn(phi, proj), prov


def address_scale(phi_proj, fit_pool) -> Dict[str, float]:
    """The rig's own normalisation: ``scale = 1 / r95(||phi(fit_pool)||)``.

    ⭐ This is *why* every leg below is a ratio: the scale is chosen so the address
    cloud has unit 95th-percentile radius, while ``sigma_q`` is an absolute constant.
    A φ rescaled by any constant produces **exactly** the same normalised addresses
    (asserted in ``tests/test_phi_geometry.py``), so an absolute-units leg would be
    measuring the scale rather than the memory.
    """
    F = np.asarray(phi_proj(np.asarray(fit_pool, np.float32)), float)
    r95 = float(np.percentile(np.linalg.norm(F, axis=1), 95.0))
    return {"r95_phi_norm": r95, "scale": float(1.0 / max(r95, 1e-9))}


def assert_no_truncation(phi_proj, addr_dim: int) -> None:
    """⛔ The pass-3 mapping contract: the store must read **all** of the projected φ."""
    k = int(getattr(phi_proj, "k", -1))
    if k != int(addr_dim):
        raise AssertionError(
            f"the φ→addr map emits {k} dims but the store addresses {addr_dim}: "
            f"PhiAddress would TRUNCATE, which is the pass-3 defect, not a mapping"
        )


# ---------------------------------------------------------------------------
# the geometry (the point of this spoke)
# ---------------------------------------------------------------------------
def nn_distances(keys: np.ndarray) -> np.ndarray:
    k = np.asarray(keys, float)
    d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def uniform_ball_spacing(n: int, d: int, reps: int = 64, seed: int = 20260809) -> float:
    """Median NN spacing of ``n`` points drawn **uniformly in the unit d-ball**.

    ⭐ The dimensionless yardstick §4 asks for. ``spacing / uniform_spacing`` says how
    uniformly ``phi`` fills the address ball — which is the ONLY thing a strong ``phi``
    can improve here, since it cannot enlarge a volume the normalisation fixes.
    """
    rng = np.random.default_rng(int(seed) + 1000 * int(d) + int(n))
    vals = []
    for _ in range(int(reps)):
        g = rng.normal(size=(int(n), int(d)))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        r = rng.uniform(size=(int(n), 1)) ** (1.0 / int(d))
        vals.append(float(np.median(nn_distances(g * r))))
    return float(np.median(vals))


def participation_ratio(keys: np.ndarray) -> float:
    """``(sum L)^2 / sum L^2`` of the key covariance — the cloud's *effective* dimension.

    Scale-invariant by construction (a global rescale cancels), so it is a legal
    §4 quantity: it says how many address directions the items actually use.
    """
    K = np.asarray(keys, float)
    if K.shape[0] < 2:
        return float("nan")
    lam = np.linalg.eigvalsh(np.cov(K - K.mean(0), rowvar=False))
    lam = np.clip(lam, 0.0, None)
    s1, s2 = float(lam.sum()), float((lam**2).sum())
    return float(s1 * s1 / max(s2, 1e-30))


def geometry_row(keys: np.ndarray, *, sigma_q: float, d_safe: float,
                 scale: float, addr_dim: int, n_keys: int) -> Dict[str, Any]:
    """One geometry row. ⚠ Every leg is a **ratio**; the scale is carried, not hidden."""
    nn = nn_distances(keys)
    spacing = float(np.median(nn))
    unif = uniform_ball_spacing(int(n_keys), int(addr_dim))
    return {
        "n_keys": int(n_keys),
        "addr_dim": int(addr_dim),
        "n_atoms": atom_budget(int(addr_dim)),
        "scale": float(scale),
        "median_nn_spacing": spacing,
        "min_nn_spacing": float(np.min(nn)),
        "mean_nn_spacing": float(np.mean(nn)),
        "key_r95": float(np.percentile(np.linalg.norm(keys, axis=1), 95.0)),
        # ---- the dimensionless legs (the only ones that may be compared) ----
        "sigma_q_over_spacing": float(sigma_q / max(spacing, 1e-12)),
        "d_safe_over_spacing": float(d_safe / max(spacing, 1e-12)),
        "spacing_over_uniform_ball": float(spacing / max(unif, 1e-12)),
        "uniform_ball_spacing": unif,
        "participation_ratio": participation_ratio(keys),
        "participation_fraction": float(participation_ratio(keys) / max(addr_dim, 1)),
        "sigma_q": float(sigma_q),
        "d_safe": float(d_safe),
    }


# ---------------------------------------------------------------------------
# rider (b) — re-price d_safe so monitor #3 can go quiet HONESTLY
# ---------------------------------------------------------------------------
def refusal_simulation(keys: np.ndarray, d_safe: float) -> Dict[str, Any]:
    """The admission gate's own rule, replayed on the offer order.

    ``Controller._admit_no_relocate``: admit iff ``min_separation(q_new, stored) >=
    d_safe``. ⛔ The rate is **measured, never tuned to a target**.
    """
    K = np.asarray(keys, float)
    admitted: List[np.ndarray] = []
    n_ref = 0
    for q in K:
        if admitted:
            dmin = float(np.min(np.linalg.norm(np.stack(admitted) - q, axis=1)))
        else:
            dmin = float("inf")
        if dmin >= float(d_safe):
            admitted.append(q)
        else:
            n_ref += 1
    n = int(K.shape[0])
    return {
        "n_offered": n,
        "n_admitted": int(len(admitted)),
        "n_refused": int(n_ref),
        "refusal_rate": float(n_ref / max(n, 1)),
        "d_safe": float(d_safe),
    }


def d_safe_pricing(task1_keys: np.ndarray, population_keys: np.ndarray,
                   frac: float) -> Dict[str, Any]:
    """The two pricings of ``d_safe`` — the rig's, and the re-priced one (rider b).

    **The defect this exposes.** The rig sizes ``d_safe = frac * median-NN(task-1 keys)``
    on a *sizing set* of ~200 task-1 images, but the store's *population* is ~16 items
    spread over several tasks. Spacing falls with ``n``, so the sizing spacing is far
    SMALLER than the population spacing ⇒ ``d_safe`` sits well below every pairwise
    distance the gate ever sees and **it cannot fire**. That is monitor #3's refusal
    rate of 0.000 — a gate that is vacuous, not a store that needed nothing refused.
    """
    s_size = float(np.median(nn_distances(task1_keys)))
    s_pop = float(np.median(nn_distances(population_keys)))
    return {
        "d_safe_frac": float(frac),
        "sizing_set_n": int(task1_keys.shape[0]),
        "sizing_set_spacing": s_size,
        "population_n": int(population_keys.shape[0]),
        "population_spacing": s_pop,
        "d_safe_rig": float(frac) * s_size,
        "d_safe_repriced": float(frac) * s_pop,
        "d_safe_rig_over_population_spacing": float(frac) * s_size / max(s_pop, 1e-12),
        "d_safe_repriced_over_population_spacing": float(frac),
    }


# ---------------------------------------------------------------------------
# ⛔ Head ruling R2(b) — the launder reads the PROJECTED phi. Asserted, not intended.
# ---------------------------------------------------------------------------
def launder_audit(embed: PhiAddress, X, addr_dim: int, labels=None) -> Dict[str, Any]:
    """Prove the kNN launder addresses through the **same projected φ** as the store.

    ⛔ A launder reading 256 dims while the store reads 8 is **not a launder — it is a
    handicap match** (fairness invariant §A4.3). This feeds the shipped
    :class:`~chlu.experiments.exp_cl_entry.RingBufferKNN` from ``embed.keys`` — the very
    array the store writes — and asserts **bit-identity**, raising otherwise.
    """
    keys = np.asarray(embed.keys(X), float)
    if keys.shape[1] != int(addr_dim):
        raise AssertionError(
            f"launder would read {keys.shape[1]}-dim keys against an {addr_dim}-dim "
            f"store: that is a handicap match, not a laundering control (§A4.3)"
        )
    y = np.arange(len(keys)) if labels is None else np.asarray(labels, int)
    launder = RingBufferKNN(len(keys))
    for k, lab in zip(keys, y, strict=False):
        launder.offer(k, int(lab))
    lk = np.stack(launder.keys)
    if not np.array_equal(lk, keys):
        raise AssertionError("launder keys are not the store's projected φ keys")
    store_addr = np.asarray(embed(X), float)[:, :addr_dim]
    return {
        "launder_key_dim": int(lk.shape[1]),
        "store_address_dim": int(store_addr.shape[1]),
        "launder_reads_projected_phi": True,
        "bit_identical_to_store_addresses": bool(np.array_equal(lk, store_addr)),
        "n_keys": int(lk.shape[0]),
    }


def byte_ledger(phi_proj: ProjectedReadIn, n_launder: int, addr_dim: int,
                capacity: int) -> Dict[str, Any]:
    """⛔ The map's params on the ledger of **every** arm, launder included (§A4.3)."""
    phi_floats = int(phi_proj.param_floats())
    map_floats = int(phi_proj.projection.param_floats())
    enc_floats = phi_floats - map_floats
    launder_item_floats = int(n_launder * (addr_dim + 1))
    return {
        "encoder_param_floats": enc_floats,
        "projection_param_floats": map_floats,
        "phi_param_floats_total": phi_floats,
        "projection_param_floats_materialised": int(
            phi_proj.projection.param_floats_materialised()
        ),
        "clu_arm_phi_floats": phi_floats,
        "knn_launder_phi_floats": phi_floats,  # ⛔ the SAME φ, so the SAME ledger term
        "knn_launder_item_floats": launder_item_floats,
        "knn_launder_total_floats": phi_floats + launder_item_floats,
        "clu_site_floats_capacity_bound": int(capacity * (addr_dim + 1)),
        "note": "φ params (encoder + map) ride on EVERY arm's ledger, the laundering "
                "control included, because both read through the same φ (§A4.3).",
    }


# ---------------------------------------------------------------------------
# rider (a) — the revived (d, atom-budget) cell: does the store dig AT ALL?
# ---------------------------------------------------------------------------
def depth_probe_cell(config: CHLUConfig, seed: int, addr_dim: int, phi_proj,
                     scale: float, keys_X, d_safe: float,
                     verbose: bool = True) -> Dict[str, Any]:
    """ONE cell (never a sweep): write a few designed sites and read the well depths.

    ⚠ **Declared risk R1:** ``CluSystem``'s learned ``V_theta`` has never been run on
    CIFAR ``phi``, and pass 1 measured it **inert at d >= 16 on MNIST** (fitted depth
    2.1e-9 / 6.8e-10 / 0.000). ⛔ An inert store makes a geometry census vacuous for a
    reason that is **NOT** the GO/NO-GO reason, and the two must never be conflated —
    which is why this runs, and reports, *first*.
    """
    g = config.experiment_phi_geometry
    assert_no_truncation(phi_proj, addr_dim)
    dim = int(addr_dim) + 1
    embed = PhiAddress(phi_proj, dim=dim, addr_dim=int(addr_dim), scale=float(scale))
    n_atoms = atom_budget(int(addr_dim), capacity=int(g.capacity))
    t0 = time.time()
    syscfg = CluSystemConfig(
        addr_dim=int(addr_dim), payload_dim=1, capacity=int(g.capacity),
        budget=int(g.capacity), seed=int(seed), stage_lifetimes=True,
        d_safe_override=float(d_safe), write_steps=int(g.write_steps),
        read_steps=int(g.read_steps), address_steps=int(g.address_steps),
        n_query_per_item=int(g.n_query_per_item), quick=bool(g.quick),
    )
    sysm = build_system(syscfg, key=jax.random.PRNGKey(int(seed)), phi=embed, loud=False)
    keys = np.asarray(embed.keys(keys_X), float)
    n_w = int(min(g.depth_probe_writes, len(keys)))
    n_adm = 0
    for j in range(n_w):
        rep = sysm.write_stream([{
            "item_id": int(j), "address": keys[j], "payload": float((j % 10 - 4.5) / 9.0),
            "permanent": True, "leak": 0.0,
        }])
        n_adm += int(bool(rep.admitted))
    depths, _ = sysm.well_fits()
    depths = np.asarray(depths, float)
    med = float(np.median(depths)) if depths.size else float("nan")
    probe = sysm.self_probe()
    out = {
        "addr_dim": int(addr_dim), "n_atoms": int(n_atoms),
        "n_atoms_from_system": int(sysm.cfg.n_atoms),
        "atom_budget_honoured": bool(int(sysm.cfg.n_atoms) == int(n_atoms)),
        "n_writes": n_w, "n_admitted": n_adm,
        "fitted_depths": [float(x) for x in depths],
        "depth_median": med, "depth_max": float(np.max(depths)) if depths.size else None,
        "inert": bool(not np.isfinite(med) or med < float(g.depth_inert_tol)),
        "inert_tol": float(g.depth_inert_tol),
        "self_probe_strict": float(probe.get("strict", float("nan"))),
        "self_probe_decode": float(probe.get("decode", float("nan"))),
        "wall_s": float(time.time() - t0),
        "d_safe": float(d_safe),
        "role": "rider (a): the revived (d, atom-budget) cell — ONE cell, not a sweep",
    }
    if verbose:
        print(f"  [depth probe] d={addr_dim} n_atoms={n_atoms} writes={n_w} "
              f"depth_median={med:.4g} inert={out['inert']} "
              f"[{out['wall_s']:.0f}s]", flush=True)
    return out


# ---------------------------------------------------------------------------
# the registered GO / NO-GO reading — mechanical, computed, never argued
# ---------------------------------------------------------------------------
def geometry_verdict(rows: List[Dict[str, Any]], *, strong_arm: str, ref_arm: str,
                     n_keys: int, se_multiple: float,
                     min_seeds_positive: int) -> Dict[str, Any]:
    """**GO** iff strong ``phi`` improves ``sigma_q``/spacing over the PCA reference at
    the same ``d``, beyond noise, over >= ``min_seeds_positive`` seeds.

    "Improves" = **lower** ``sigma_q``/spacing (the query jitter is a smaller fraction
    of the distance between neighbouring keys). Paired per seed, because the stream is
    bit-identical across arms at a seed.
    """
    by_d: Dict[int, Dict[str, Any]] = {}
    dims = sorted({int(r["addr_dim"]) for r in rows})
    for d in dims:
        sel = [r for r in rows
               if int(r["addr_dim"]) == d and int(r["n_keys"]) == int(n_keys)]
        strong = {int(r["seed"]): r for r in sel if r["arm"] == strong_arm}
        ref = {int(r["seed"]): r for r in sel if r["arm"] == ref_arm}
        seeds = sorted(set(strong) & set(ref))
        deltas = [float(ref[s]["sigma_q_over_spacing"]
                        - strong[s]["sigma_q_over_spacing"]) for s in seeds]
        arr = np.asarray(deltas, float)
        n = int(arr.size)
        mean = float(arr.mean()) if n else float("nan")
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        n_pos = int(np.sum(arr > 0))
        go = bool(n >= int(min_seeds_positive) and n_pos >= int(min_seeds_positive)
                  and np.isfinite(se) and mean > float(se_multiple) * se)
        by_d[d] = {
            "addr_dim": d, "n_atoms": atom_budget(d), "n_keys": int(n_keys),
            "seeds": seeds, "n_seeds": n,
            "sigma_q_over_spacing_strong": [
                float(strong[s]["sigma_q_over_spacing"]) for s in seeds],
            "sigma_q_over_spacing_reference": [
                float(ref[s]["sigma_q_over_spacing"]) for s in seeds],
            "paired_improvement_mean": mean,
            "paired_improvement_se": se,
            "n_seeds_positive": n_pos,
            "ratio_strong_over_reference": (
                float(np.mean([strong[s]["sigma_q_over_spacing"]
                               / max(ref[s]["sigma_q_over_spacing"], 1e-12)
                               for s in seeds])) if n else float("nan")),
            "go": go,
        }
    means = {d: float(np.mean(v["sigma_q_over_spacing_strong"]))
             if v["sigma_q_over_spacing_strong"] else float("inf")
             for d, v in by_d.items()}
    d_fav = int(min(means, key=lambda k: means[k])) if means else -1
    return {
        "rule": ("GO iff strong φ improves σ_q/spacing (i.e. LOWERS it) over the PCA "
                 "reference at the same d, beyond noise (mean > %.1f·SE) on >= %d "
                 "paired seeds" % (float(se_multiple), int(min_seeds_positive))),
        "n_keys_primary": int(n_keys),
        "strong_arm": strong_arm, "reference_arm": ref_arm,
        "by_d": {str(k): v for k, v in by_d.items()},
        "d_favoured_by_geometry": d_fav,
        "d_favoured_reason": ("argmin over d of the strong arm's mean σ_q/spacing at "
                              "the primary n — pure geometry, so an inert store cannot "
                              "leak into the GO/NO-GO reason (Head ruling R1)"),
        "geometry_go": bool(by_d[d_fav]["go"]) if d_fav in by_d else False,
    }


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def run_seed(config: CHLUConfig, seed: int, data=None,
             verbose: bool = True) -> Dict[str, Any]:
    """One seed: one stream, the encoders, the maps, the geometry, the riders."""
    g = config.experiment_phi_geometry
    t0 = time.time()
    stream = build_cl_stream(cl_config(config, g.phi_arm_reference, 8), seed, data=data)
    fp = stream_fingerprint(stream)
    pool = np.asarray(stream[f"fit_pool_{g.phi_regime}"], np.float32)
    if verbose:
        print(f"[phi-geometry] seed {seed}: stream {fp}, fit pool {len(pool)}",
              flush=True)

    # the ENCODERS are fitted ONCE per seed and reused at every d — the map is what
    # changes with d, which is the whole point of having a map.
    encoders: Dict[str, Any] = {}
    prov: Dict[str, Any] = {}
    for arm in (g.phi_arm_strong, g.phi_arm_control):
        te = time.time()
        cl = cl_config(config, arm, int(g.phi_dim_strong))
        phi, p = build_phi(g.phi_regime or PHI_PRIMARY, stream, cl, seed)
        encoders[arm] = phi
        prov[arm] = {**p, "fit_wall_s": float(time.time() - te)}
        if verbose:
            print(f"  encoder {arm} @phi_dim={g.phi_dim_strong} fitted in "
                  f"{prov[arm]['fit_wall_s']:.0f}s", flush=True)

    rows: List[Dict[str, Any]] = []
    ledgers: List[Dict[str, Any]] = []
    launder_checks: List[Dict[str, Any]] = []
    dsafe_rows: List[Dict[str, Any]] = []
    depth_rows: List[Dict[str, Any]] = []
    n_max = max(int(max(g.n_keys_grid)), int(g.d_safe_sizing_n))
    pop_X = offer_order_keys(stream, n_max, int(g.n_offer_per_task))
    if verbose:
        print(f"  offer-order population: requested up to {n_max}, available "
              f"{len(pop_X)} ⇒ n grid {achieved_n_grid(g.n_keys_grid, len(pop_X))}",
              flush=True)
    task1_X = np.asarray(stream["train_X"][0], np.float32)[: int(g.d_safe_sizing_n)]

    phi_cache: Dict[Any, Any] = {}
    for d in [int(x) for x in g.addr_dims]:
        arms: Dict[str, ProjectedReadIn] = {}
        # ⛔ the weak-φ reference, RE-MEASURED here at MATCHING d (Head ruling R1:
        # pass-1/2 census geometry is MNIST and is not this run's baseline)
        arms[f"{g.phi_arm_reference}@d"], p_ref = build_arm(
            config, stream, seed, g.phi_arm_reference, d, d, "identity",
            cache=phi_cache)
        prov.setdefault(f"{g.phi_arm_reference}@{d}", p_ref)
        # the map-neutrality control: PCA-256 -> d through the SAME map must reproduce
        # the reference (PCA of PCA is PCA), so the map is provably not doing magic
        arms[f"{g.phi_arm_reference}256->{g.projection}"], _ = build_arm(
            config, stream, seed, g.phi_arm_reference, 256, d, g.projection,
            cache=phi_cache)
        # ⭐ the PRIMARY strong arm, and the map controls
        for form in [g.projection] + list(g.projection_controls):
            proj = PhiProjection(
                np.asarray(encoders[g.phi_arm_strong](pool), np.float32), d,
                form=form, seed=seed)
            arms[f"{g.phi_arm_strong}->{form}"] = ProjectedReadIn(
                encoders[g.phi_arm_strong], proj)
        proj_c = PhiProjection(
            np.asarray(encoders[g.phi_arm_control](pool), np.float32), d,
            form=g.projection, seed=seed)
        arms[f"{g.phi_arm_control}->{g.projection}"] = ProjectedReadIn(
            encoders[g.phi_arm_control], proj_c)

        for name, phi_proj in arms.items():
            assert_no_truncation(phi_proj, d)
            sc = address_scale(phi_proj, pool)
            embed = PhiAddress(phi_proj, dim=d + 1, addr_dim=d, scale=sc["scale"])
            t1_keys = np.asarray(embed.keys(task1_X), float)
            pop_keys = np.asarray(embed.keys(pop_X), float)
            pricing = d_safe_pricing(t1_keys, pop_keys[: int(g.n_keys_primary)],
                                     float(g.d_safe_frac))
            for n in achieved_n_grid(g.n_keys_grid, len(pop_keys)):
                k = pop_keys[:n]
                if k.shape[0] < 3:
                    continue
                rows.append({
                    "seed": int(seed), "arm": name, "projection_form": (
                        phi_proj.projection.form),
                    **geometry_row(k, sigma_q=float(g.query_sigma),
                                   d_safe=pricing["d_safe_rig"], scale=sc["scale"],
                                   addr_dim=d, n_keys=int(k.shape[0])),
                    "r95_phi_norm_fitpool": sc["r95_phi_norm"],
                })
            # rider (b): the two pricings and what each one refuses
            pop_primary = pop_keys[: int(g.n_keys_primary)]
            dsafe_rows.append({
                "seed": int(seed), "arm": name, "addr_dim": d, **pricing,
                "refusal_rig": refusal_simulation(pop_primary, pricing["d_safe_rig"]),
                "refusal_repriced": refusal_simulation(
                    pop_primary, pricing["d_safe_repriced"]),
            })
            ledgers.append({
                "seed": int(seed), "arm": name, "addr_dim": d,
                **byte_ledger(phi_proj, int(g.n_keys_primary), d, int(g.capacity)),
            })
            launder_checks.append({
                "seed": int(seed), "arm": name, "addr_dim": d,
                **launder_audit(embed, pop_X[: int(g.n_keys_primary)], d),
            })

        # rider (a) — ONE depth cell per d, on the PRIMARY strong arm, seed 0 only
        if g.depth_probe and int(seed) == int(g.depth_probe_seed) \
                and d in [int(x) for x in g.depth_probe_dims]:
            phi_p = arms[f"{g.phi_arm_strong}->{g.projection}"]
            sc = address_scale(phi_p, pool)
            emb = PhiAddress(phi_p, dim=d + 1, addr_dim=d, scale=sc["scale"])
            ds = d_safe_pricing(np.asarray(emb.keys(task1_X), float),
                                np.asarray(emb.keys(pop_X), float)[
                                    : int(g.n_keys_primary)], float(g.d_safe_frac))
            depth_rows.append({
                "seed": int(seed), "arm": f"{g.phi_arm_strong}->{g.projection}",
                **depth_probe_cell(config, seed, d, phi_p, sc["scale"],
                                   pop_X[: int(g.depth_probe_writes)],
                                   ds["d_safe_rig"], verbose=verbose),
            })

    return {
        "seed": int(seed), "stream_fingerprint": fp,
        "phi_provenance": prov, "geometry_rows": rows, "byte_ledger": ledgers,
        "launder_audit": launder_checks, "d_safe_rider": dsafe_rows,
        "depth_probe": depth_rows, "wall_s": float(time.time() - t0),
    }


def run_experiment_phi_geometry(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "plots",
    seeds: Optional[List[int]] = None,
    quick: bool = False,
    data=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run every seed and write ``PHI-GEOMETRY.json`` with a mechanical ``geometry_go``."""
    config = config or get_default_config()
    if quick:
        apply_quick(config)
    g = config.experiment_phi_geometry
    seeds = [int(s) for s in (seeds if seeds is not None else g.seeds)]

    cells = [run_seed(config, s, data=data, verbose=verbose) for s in seeds]
    rows = [r for c in cells for r in c["geometry_rows"]]
    verdict = geometry_verdict(
        rows, strong_arm=f"{g.phi_arm_strong}->{g.projection}",
        ref_arm=f"{g.phi_arm_reference}@d", n_keys=int(g.n_keys_primary),
        se_multiple=float(g.go_se_multiple),
        min_seeds_positive=int(g.go_min_seeds_positive),
    )
    depth = [d for c in cells for d in c["depth_probe"]]
    inert = {int(d["addr_dim"]): bool(d["inert"]) for d in depth}
    feasible = [d for d in [int(x) for x in g.addr_dims] if not inert.get(d, False)]
    d_geo = int(verdict["d_favoured_by_geometry"])
    results = {
        "experiment": "phi_geometry",
        "wave": "C2W8 pass 3 (wt2)",
        "dial": "none — encoder/geometry instrumentation. No claim cell, no verdict.",
        "promotable": False,
        "why_not_promotable": (
            "N94: instrumentation. Nothing here is scored against a baseline; the "
            "GO/NO-GO re-labels the spine (Head ruling R4) and never blocks it."
        ),
        "substrate": g.dataset,
        "substrate_ruling": (
            "Head ruling R1: the spine runs on Split-CIFAR-10 because that is where the "
            "strong encoders were built and priced. ⛔ pass-1/2 census geometry is MNIST "
            "and is NOT this run's baseline — the PCA reference below is measured HERE, "
            "at matching d, in this run."
        ),
        "seeds": seeds,
        "mapping": {
            "declared": (
                "PhiProjection(form=%r): mean-centre + top-d principal directions of "
                "φ(fit_pool), fitted on the task1_only pool (no leakage from unseen "
                "tasks), frozen thereafter." % g.projection
            ),
            "controls": list(g.projection_controls),
            "truncate_is_the_shipped_defect": True,
            "launder_reads_projected_phi": all(
                bool(a["launder_reads_projected_phi"])
                for c in cells for a in c["launder_audit"]),
            "launder_bit_identical_to_store": all(
                bool(a["bit_identical_to_store_addresses"])
                for c in cells for a in c["launder_audit"]),
        },
        "joint_dial_d_atom_budget": atom_budget_table(capacity=int(g.capacity)),
        "geometry_rows": rows,
        "verdict": verdict,
        "geometry_go": bool(verdict["geometry_go"]),
        "d_favoured_by_geometry": d_geo,
        "depth_probe": depth,
        "store_inert_by_d": {str(k): v for k, v in inert.items()},
        "d_recommended_operational": (
            d_geo if d_geo in feasible else (max(feasible) if feasible else None)),
        "d_recommendation_note": (
            "⛔ the geometry-favoured d and the operationally usable d are reported "
            "SEPARATELY on purpose: an inert store makes a census vacuous for a reason "
            "that is NOT the GO/NO-GO reason (Head ruling R1) and the two must never be "
            "conflated."
        ),
        "byte_ledger": [r for c in cells for r in c["byte_ledger"]],
        "launder_audit": [r for c in cells for r in c["launder_audit"]],
        "d_safe_rider": [r for c in cells for r in c["d_safe_rider"]],
        "phi_provenance": {str(c["seed"]): c["phi_provenance"] for c in cells},
        "stream_fingerprints": {str(c["seed"]): c["stream_fingerprint"] for c in cells},
        "declared_not_runs": [
            "any performance/ACC number — NOT RUN, this is instrumentation",
            "the spine's capture gate (wt3) — NOT RUN here",
            "G-ADDR (wt1's deliverable) — NOT RUN here",
            "convae / any encoder arm beyond simclr+randconv — NOT RUN, declared",
        ],
        "flags": flag_table(config, seeds),
        "wall_s": float(sum(c["wall_s"] for c in cells)),
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "PHI-GEOMETRY.json")
    with open(path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    results["json_path"] = path
    if verbose:
        print(f"\n[phi-geometry] {verdict['rule']}\n"
              f"  d_favoured={d_geo}  geometry_go={results['geometry_go']}\n"
              f"  store inert by d: {results['store_inert_by_d']}\n"
              f"  wrote {path}", flush=True)
    return results


def flag_table(config: CHLUConfig, seeds) -> Dict[str, Any]:
    g = config.experiment_phi_geometry
    return {
        "seeds": [int(s) for s in seeds], "dataset": g.dataset,
        "addr_dims": [int(x) for x in g.addr_dims],
        "phi_arm_strong": g.phi_arm_strong, "phi_arm_control": g.phi_arm_control,
        "phi_arm_reference": g.phi_arm_reference,
        "phi_dim_strong": int(g.phi_dim_strong), "phi_regime": g.phi_regime,
        "enc_steps": int(g.enc_steps), "n_fit_region": int(g.n_fit_region),
        "n_fit_pool": int(g.n_fit_pool), "projection": g.projection,
        "projection_controls": list(g.projection_controls),
        "n_keys_primary": int(g.n_keys_primary),
        "n_keys_grid": [int(x) for x in g.n_keys_grid],
        "query_sigma": float(g.query_sigma), "d_safe_frac": float(g.d_safe_frac),
        "d_safe_sizing_n": int(g.d_safe_sizing_n),
        "depth_probe_writes": int(g.depth_probe_writes),
        "write_steps": int(g.write_steps), "read_steps": int(g.read_steps),
        "address_steps": int(g.address_steps), "capacity": int(g.capacity),
        "quick": bool(g.quick), "promotable": False,
    }


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


def apply_quick(config: CHLUConfig) -> None:
    """Smoke mode: the same code path on a tiny stream. ⛔ Never a measurement."""
    g = config.experiment_phi_geometry
    g.quick = True
    g.seeds = [0]
    g.addr_dims = [4, 6]
    g.phi_dim_strong = 12
    g.enc_steps = 2
    g.n_fit_region = 200
    g.n_fit_pool = 80
    g.n_keys_primary = 8
    g.n_keys_grid = [8, 16]
    g.d_safe_sizing_n = 24
    g.depth_probe_dims = [4]
    g.depth_probe_writes = 2
    g.write_steps = 20
    g.read_steps = 40
    g.address_steps = 20
    g.capacity = 4
    cl = config.experiment_cl_entry
    cl.n_tasks = 3
    cl.n_train_per_task = 40
    cl.n_test_per_task = 20
    cl.enc_channels = [4, 8]
    cl.enc_pool = 2


def main():
    parser = argparse.ArgumentParser(
        description="C2W8 pass 3: the φ→addr map and the address geometry it buys")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="e.g. 0,1,2")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save-dir", type=str, default="plots")
    parser.add_argument("--dims", type=str, default=None, help="addr dims, e.g. 8,12,16")
    parser.add_argument("--no-depth-probe", action="store_true")
    args = parser.parse_args()

    config = get_default_config()
    save_dir = args.save_dir
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        paths = pm.get_paths(args.project)
        save_dir = str(paths["plots"])
        cfg_path = paths["config"] / "config.yaml"
        if cfg_path.exists():
            from chlu.config import load_config

            config = load_config(cfg_path)
    if args.quick:
        apply_quick(config)
    if args.dims:
        config.experiment_phi_geometry.addr_dims = [int(x) for x in args.dims.split(",")]
    if args.no_depth_probe:
        config.experiment_phi_geometry.depth_probe = False
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_experiment_phi_geometry(config, save_dir=save_dir, seeds=seeds)


if __name__ == "__main__":
    main()


__all__ = [
    "atom_budget", "atom_budget_table", "cl_config", "build_arm", "address_scale",
    "offer_order_keys", "achieved_n_grid",
    "assert_no_truncation", "geometry_row", "geometry_verdict", "refusal_simulation",
    "d_safe_pricing", "launder_audit", "byte_ledger", "depth_probe_cell",
    "run_seed", "run_experiment_phi_geometry", "apply_quick",
]
