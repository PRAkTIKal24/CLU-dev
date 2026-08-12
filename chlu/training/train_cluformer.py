"""Tier-iii pilot trainer — the full CLU store as a streaming block's memory.

⛔ **This is a THIRD training path and it is not either of the other two.**
``training/train.py`` is dynamics Wake-Sleep (MSE + Lyapunov, Exp A/B);
``training/train_generative.py`` is pure EBM PCD (Exp C). Neither is used here.
This module trains a **byte-level language model** end to end with plain Adam on
next-token cross-entropy, and the CLU store is a *layer* inside it. The store's
own write objective (:func:`~chlu.training.train_memory.write_loss`) runs as a
differentiably-unrolled **inner** loop at chunk granularity — the Titans/TTT
convention, given identically to the swap controls.

**The two-pass forward (`decision replay`) — the load-bearing design.**
The C2W1 controller's verbs are *discrete* and branch on ``numpy`` values, so it
cannot run inside a traced/differentiated forward. Deleting it is forbidden
(Head ruling 0.1: no full-CLU feature is turned off). Instead:

1. :func:`plan_pass` runs the **real** controller — admission, placement,
   eviction, decay, the confidence gate — on detached latents, layer by layer,
   and emits a :class:`~chlu.core.blocks.WritePlan` per layer;
2. :func:`loss_fn` replays the plan inside a fully differentiable forward.

Every guard fires, every verb is exercised, the monitors see the real trip
states. What is given up is ``d(decision)/d(theta)`` — which is **zero anyway**
(the verbs are discrete), and which T3's corollary identifies as a gradient
attractor rather than a channel worth having.

**What must be identical across arms** (or the system-level swap is not a swap):
embedding, positional table, LayerNorms, the intra-chunk causal conv, ``phi``,
the assimilation projection, the token-wise MLP, the head, the optimiser, the
LR schedule, the **data order**, the seeds, and the **chunk granularity**. Only
``block.cell`` changes. :func:`assert_shared_shell_identical` checks it.
"""

from __future__ import annotations

import atexit
import gc
import json
import math
import multiprocessing as mp
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.core.blocks import (
    MEMORY_CELLS,
    StreamMemoryConfig,
    StreamModel,
    WritePlan,
    blank_plan,
    make_memory_cell,
    round_robin_plan,
    solve_matched_gru,
    solve_matched_ttt,
    store_byte_law,
    swap_ledger,
)
from chlu.core.clu_system import CluSystemConfig, make_controller
from chlu.core.monitors import MonitorContext, default_registry
from chlu.data.enwik8 import bits_per_character


# ==========================================================================
# config (⚠ NOT chlu/config.py — standing read-only to C2 engineers)
# ==========================================================================
@dataclass
class PilotConfig:
    """Every knob of the tier-iii pilot, with the scale it was declared at.

    Override path: a ``cluformer:`` block in ``projects/<name>/config/config.yaml``,
    read through :meth:`from_mapping` (same tolerance as ``chlu.config.load_config``).
    """

    # -- scale ------------------------------------------------------------
    d_model: int = 512
    n_layers: int = 12
    seq_len: int = 1024
    batch: int = 8
    vocab_size: int = 256

    # -- store geometry (the C2W1 known-productive band, scaled) -----------
    addr_dim: int = 8
    payload_dim: int = 4
    capacity: int = 32
    atoms_per_item: int = 256

    # -- optimisation ------------------------------------------------------
    steps: int = 2000
    lr: float = 1e-3
    warmup: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    seed: int = 0

    # -- evaluation ---------------------------------------------------------
    eval_batches: int = 20
    dyneval_lr: float = 1e-4
    dyneval_batches: int = 20

    # -- data ---------------------------------------------------------------
    data_bytes: Optional[int] = None      # None => the full 100 MB stream
    data_root: Optional[str] = None

    # -- arms ----------------------------------------------------------------
    arms: Tuple[str, ...] = ("clu_store", "gru_matched", "ttt_matched", "none", "echo")
    #: ⛔⭐ `pilot-ttt-nan-and-d5-wiring` DEFECT 1 — the ``ttt_matched`` arm's
    #: inner loop is **divergent by construction at the pilot geometry**: its
    #: update is non-expansive only while ``eta ||theta_K z||^2 < 2``, and that
    #: product is fixed by ``n``, which :func:`solve_matched_ttt` reads off the
    #: CLU cell's byte ledger. Measured at init: **3.47 on 100 % of chunks at
    #: pilot** (vs 2.31 on 44 % at toy) => ``||W||`` grows x8e4 inside one
    #: forward pass and the arm went NaN at step 135/4000 on CSF3, losing the
    #: rival column. ``True`` switches :class:`MatchedTTTCell` to the normalized
    #: delta rule (its own docstring's "closed-form step"), which is
    #: non-expansive for any ``eta in (0,2)`` and scale-free.
    #:
    #: ⛔ **Default ``False`` = the shipped arithmetic, bit-for-bit.** This moves
    #: a PUBLISHED rival column, so the flip is a Hub ruling. ⭐ It is a
    #: ``PilotConfig`` field precisely so it stays OUT of the resume fingerprint
    #: while unset: ``as_flag_table`` emits non-default fields only, so the five
    #: in-flight CSF3 legs resume against their banked journals unchanged.
    ttt_normalized_write: bool = False

    # -- the memory slot ------------------------------------------------------
    memory: Dict[str, Any] = field(default_factory=dict)   # -> StreamMemoryConfig
    store: Dict[str, Any] = field(default_factory=dict)    # -> CluSystemConfig extras

    # -- runtime --------------------------------------------------------------
    quick: bool = False
    #: ⭐ Run the REAL C2W1 controller in the plan pass. ``False`` substitutes a
    #: round-robin allocator; permitted ONLY for gradient/wall-clock probes, and
    #: every reported performance number asserts this is ``True``.
    real_controller: bool = True
    #: Monitors are run every ``monitor_every`` optimisation steps (and always at
    #: evaluation). Running all 13 per chunk would dominate the clock.
    monitor_every: int = 100
    #: ⭐ Lanes of the batch are **independent** in the plan pass (each builds its
    #: own controller), so they may run in worker PROCESSES. ``0`` = serial (the
    #: shipped behaviour, bit-identical); ``-1`` = ``min(batch, cpu_count)``.
    #: Layers stay sequential — layer ``l+1``'s latents need layer ``l``'s
    #: decisions — so this is a batch-axis cut only.
    plan_workers: int = 0
    #: Start method for those workers. ⛔ ``"spawn"`` by default and not ``"fork"``:
    #: JAX is **not fork-safe** (the parent has live backend threads), and the
    #: controller's shadow store is an Equinox module. Spawn pays one import per
    #: worker at pool creation and nothing thereafter (the pool is persistent).
    plan_mp_start: str = "spawn"
    #: ⚠ §7.27 in-flight watch: log untrained-vs-trained well depth and the ``q*``
    #: payload spread at ``monitor_every`` cadence into the run artifact. Cheap,
    #: on by default — store destruction must be caught in flight, not post-mortem.
    store_watch: bool = True

    # -- `csf3-memory-fit`: the two levers that live OUTSIDE the model ---------
    #: ⭐ **Gradient accumulation** — the fallback memory lever if remat alone is
    #: short. ``1`` (default) = the shipped single-shot backward, bit-identical.
    #: ``n > 1`` cuts the BATCH axis into ``n`` equal microbatches, backprops
    #: them **sequentially** (``lax.scan``, so XLA cannot overlap them) and sums
    #: the gradients, then takes ONE optimiser step. The *effective* batch is
    #: preserved exactly: the registered ``batch = 8`` is still what one update
    #: sees. ⚠ **Summation-order caveat:** ``mean(mean_i)`` over equal-size
    #: microbatches equals the full-batch mean in exact arithmetic but not
    #: bit-for-bit in float32, so this lever is NOT part of the bit-identity
    #: control (it is validated to ~1e-6 relative instead). Requires
    #: ``batch % accum_steps == 0``.
    accum_steps: int = 1
    #: ⭐ Lanes used by :func:`allocation_liveness`. ``0`` (default) = the whole
    #: batch, as shipped. At pilot scale that probe is a full-batch backward
    #: taken at init, and it is a **liveness anchor, not a paper number** — the
    #: quantity is a gradient norm and a slot-entropy, both defined per lane —
    #: so the scale submission runs it on ``1``. The lane count used is reported
    #: in the returned dict, never implied.
    liveness_lanes: int = 0
    #: ⚠ Lanes used by :func:`gradient_probe`. ``0`` (default) = the whole batch.
    #: Unlike ``liveness_lanes`` this one moves a **published** number (the S2
    #: ``||dL/dphi||`` magnitudes scale with the batch), so it is left OFF by
    #: default and the trajectory-vs-settled-point CONTRAST — which is what S2
    #: claims — is unaffected because both arms use the same lanes.
    probe_lanes: int = 0

    # -- `pilot-checkpoint-resume`: the HOST-memory levers ---------------------
    #: ⭐ **The CSF3 attempt-1 crash site.** ``True`` (default) releases the JAX
    #: compilation caches (:func:`release_host_memory`) at every eval-phase
    #: boundary, so the next phase's XLA compile spike does not stack on the
    #: retained executables of the phase before it. The eval block compiles a
    #: pile of ONE-SHOT programs — the dyn-eval backward, the blank-store read,
    #: five anytime budgets, the gradient probe's four backwards — none of which
    #: is ever re-used, so the only price is a re-trace if a shape recurs.
    #: ⛔ Decision-inert by construction (it drops executables, not values); the
    #: toy bit-identity gate is what licenses saying so.
    eval_cache_hygiene: bool = True
    #: ⭐ Log host RSS (:func:`host_rss`) at every phase boundary, to stdout and
    #: into the PARTIAL artifact. Cheap (two ``/proc`` reads) and on by default:
    #: an ``oom_kill`` leaves no JSON behind, so the log line is the ONLY thing
    #: that can attribute the next crash to a named phase.
    rss_log: bool = True
    #: ⚠ **TEST/OPS HOOK, default 0 = off.** Hard-exit (``os._exit``, i.e. no
    #: finalisers — an ``oom_kill`` has none either) once this many arms have
    #: been banked to the journal. It exists so the resume path can be gated by
    #: an actual interrupted run rather than by inspection, and so the Head can
    #: deliberately split a run across jobs. ⛔ Never part of a reported config;
    #: exempt from the resume flag-equality check for that reason.
    stop_after_arms: int = 0

    @classmethod
    def from_mapping(cls, overrides: Optional[dict] = None) -> "PilotConfig":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in dict(overrides or {}).items() if k in known}
        if "arms" in kw:
            kw["arms"] = tuple(kw["arms"])
        return cls(**kw)

    def as_flag_table(self) -> Dict[str, Any]:
        base = PilotConfig()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(base, f.name)}

    def memory_cfg(self) -> StreamMemoryConfig:
        return StreamMemoryConfig.from_mapping(self.memory)

    def store_cfg(self) -> CluSystemConfig:
        """The **full** C2W1 store config (ruling 0.1: every lever staged ON).

        ⭐ The stage flags are all ``True`` here, unlike the C2W1 defaults, because
        the Head's ruling is that no full-CLU feature is turned off. The soft
        certificate is ON (it is the sharing/interaction precondition) and ``B``
        is declared through ``soft_certificate_kwargs``.
        """
        base = dict(
            addr_dim=self.addr_dim, payload_dim=self.payload_dim,
            capacity=self.capacity, atoms_per_item=self.atoms_per_item,
            seed=self.seed,
            # every lever live
            stage_lifetimes=True, stage_admission=True, stage_capacity_pressure=True,
            stage_deletion=True, stage_basin_interaction=True, stage_retry=True,
            stage_trajectory_read=True,
            soft_certificate=True,
            leak=0.02, retry_max_rounds=1, retry_tau=0.5,
            budget=max(1, int(self.capacity * 3 // 4)),   # real capacity pressure
        )
        base.update(dict(self.store or {}))
        return CluSystemConfig.from_mapping(base)


# ==========================================================================
# arm construction + the matched-swap solve
# ==========================================================================
@dataclass
class ArmSpec:
    """One arm of the system-level swap, with its resolved cell geometry."""

    name: str
    hidden: Optional[int] = None
    ttt_shape: Optional[Tuple[int, int]] = None


def solve_arms(pcfg: PilotConfig, key) -> Tuple[Dict[str, ArmSpec], Dict[str, Any]]:
    """Solve every swap arm's geometry **against the measured CLU cell**.

    ⛔ Returns the ledger too, because the D2 finding is decided here and not at
    reporting time: a GRU's ``params = Theta(h^2)`` against a state of
    ``Theta(h)``, while the CLU store's parameters *are* its state, so matched
    parameters and matched state-bytes cannot both be hit. Both columns are
    computed and published; the TTT-class arm carries the two-sided match.
    """
    from chlu.core.blocks import CluStoreCell

    scfg, mcfg = pcfg.store_cfg(), pcfg.memory_cfg()
    ref = CluStoreCell(scfg, mcfg, key=key)
    led = ref.cell_ledger()
    dim = int(scfg.dim)
    h_params = solve_matched_gru(led["params"], dim)
    kn = solve_matched_ttt(led["params"], led["state_floats"], dim)
    specs = {
        "clu_store": ArmSpec("clu_store"),
        "gru_matched": ArmSpec("gru_matched", hidden=h_params),
        "ttt_matched": ArmSpec("ttt_matched", ttt_shape=kn),
        "none": ArmSpec("none"),
        "echo": ArmSpec("echo"),
    }
    cells = {n: make_memory_cell(n, latent_dim=dim, clu_cfg=scfg, mcfg=mcfg,
                                 hidden=h_params, ttt_shape=kn,
                                 ttt_normalized_write=bool(pcfg.ttt_normalized_write),
                                 key=key)
             for n in MEMORY_CELLS}
    ledger = swap_ledger(cells)
    # ⛔ the OTHER GRU column: matched state-bytes, which is what makes the swap
    # one-sided. Computed by arithmetic (constructing it would be 39 G params).
    hs = int(led["state_floats"])
    ledger["gru_matched_state_ARITHMETIC_ONLY"] = {
        "hidden": hs,
        "params": 3 * hs * hs + (4 * dim + 5) * hs + dim * dim + dim,
        "state_floats": hs,
        "state_bytes": 4 * hs,
        "note": "NOT CONSTRUCTED - see report; this is the two-sided-match "
                "impossibility for a GRU cell, stated as arithmetic.",
    }
    ledger["_byte_law_per_item"] = store_byte_law(
        led["n_atoms"] // int(scfg.capacity), dim, int(scfg.addr_dim),
        int(scfg.payload_dim))
    ledger["_table_rows_at_matched_state"] = int(
        led["state_floats"] // (int(scfg.addr_dim) + int(scfg.payload_dim)))
    ledger["_store_items"] = int(scfg.capacity)
    return specs, ledger


def calibrate_phi_gain(pcfg: PilotConfig, tokens, *, key) -> float:
    """⭐ The declared **anti-collapse initialisation** of the shared ``phi``.

    §A13's design rule: *allocation collapse is a gradient-flow ATTRACTOR — the
    moment a policy reaches the degenerate corner its own gradient dies and it
    cannot leave. Initialise away from it.* A tanh MLP on LayerNormed pooled
    chunk summaries emits addresses with RMS ``~0.1`` at ``d_model = 64``, deep
    inside the admission gate's merge radius, so at gain 1 the store starts life
    refusing nearly every offer — the degenerate corner, reached at step 0.

    The rule applied here is principled and scale-free, not tuned to an outcome:
    **set the gain so the RMS address norm equals the store's declared
    ``ball_radius``** — i.e. phi's addresses fill the address ball the store's
    geometry is defined on. It depends only on the arm-identical shell, so the
    same gain is used by every arm, and it is reported as a flag.
    """
    mcfg = pcfg.memory_cfg()
    scfg = pcfg.store_cfg()
    probe = build_arm("none", pcfg, {"none": ArmSpec("none")}, key=key)
    h = jax.vmap(lambda t: jax.vmap(probe.embed)(t))(jnp.asarray(tokens, jnp.int32))
    h = h + probe.pos[: h.shape[1]][None]
    z = np.asarray(jax.vmap(probe.blocks[0].chunk_latents)(h))
    rms = float(np.sqrt((z[..., : scfg.addr_dim] ** 2).sum(-1).mean()))
    return float(mcfg.phi_gain) * float(scfg.ball_radius) / max(rms, 1e-9)


def calibrate_atom_group_centers(pcfg: PilotConfig, tokens, *, key) -> tuple:
    """⭐ **H1's localization targets: the φ-image of the EARLIEST chunks.**

    `cluformer-pilot` §5.3's placement hypothesis is that a few unrolled write
    steps cannot gather 128 atoms scattered at ``init_scale = 1.0`` into a well
    at the target, so the binding constraint is atom *placement at init*. The fix
    it names is the shipped N98 lever — *"atoms seeded near the φ-image of early
    chunks instead of scattered at scale 1.0"* — which needs one localization
    target per atom group.

    This returns exactly that: lane 0's first ``capacity`` chunk latents from the
    same calibration batch the φ-gain is set on, **address axes only** (N46), as
    a hashable tuple of tuples (``StreamMemoryConfig`` is a static field).

    ⚠ **Declared property: this is a data-dependent INITIALISATION.** It is
    therefore PARAMETERS under the learned-initial-state rule (PREREG-Bprime §4),
    exactly like a GRU's ``h0``, and it changes no byte of the STATE column. It
    also means the **blank-store control must be run with the same localized
    init** — otherwise a self-probe hit bought by the initialisation is scored as
    a retrieval. ⛔ Call ``calibrate_phi_gain`` FIRST and put its gain into
    ``pcfg.memory``, or the targets are in the wrong scale.
    """
    scfg = pcfg.store_cfg()
    probe = build_arm("none", pcfg, {"none": ArmSpec("none")}, key=key)
    h = jax.vmap(lambda t: jax.vmap(probe.embed)(t))(jnp.asarray(tokens, jnp.int32))
    h = h + probe.pos[: h.shape[1]][None]
    z = np.asarray(jax.vmap(probe.blocks[0].chunk_latents)(h))   # (B, n_chunks, dim)
    K, d = int(scfg.capacity), int(scfg.addr_dim)
    flat = z[0, :, :d]
    if flat.shape[0] < K:      # short sequence: wrap the earliest chunks
        reps = int(np.ceil(K / max(flat.shape[0], 1)))
        flat = np.concatenate([flat] * reps, axis=0)
    return tuple(tuple(float(v) for v in row) for row in flat[:K])


def build_arm(name: str, pcfg: PilotConfig, specs: Dict[str, ArmSpec], *, key
              ) -> StreamModel:
    """Build one arm. ⭐ The shell's keys do NOT depend on the arm.

    ``StreamModel`` splits ``key`` into ``n_layers + 3`` and hands slot ``2 + i``
    to layer ``i``; the shell's parameters (embedding, positional, norms, conv,
    phi, assimilation, MLP, head) are therefore **bit-identical across arms at
    the same seed** and only the cell differs. Asserted by
    :func:`assert_shared_shell_identical`.
    """
    scfg, mcfg = pcfg.store_cfg(), pcfg.memory_cfg()
    spec = specs[name]
    ks = jax.random.split(key, pcfg.n_layers + 4)
    cells = [make_memory_cell(name, latent_dim=int(scfg.dim), clu_cfg=scfg,
                              mcfg=mcfg, hidden=spec.hidden,
                              ttt_shape=spec.ttt_shape,
                              ttt_normalized_write=bool(pcfg.ttt_normalized_write),
                              key=ks[i])
             for i in range(pcfg.n_layers)]
    return StreamModel(vocab_size=pcfg.vocab_size, d_model=pcfg.d_model,
                       n_layers=pcfg.n_layers, max_len=pcfg.seq_len, cells=cells,
                       mcfg=mcfg, latent_dim=int(scfg.dim),
                       addr_dim=int(scfg.addr_dim), payload_dim=int(scfg.payload_dim),
                       key=ks[-1])


def shell_of(model: StreamModel):
    """The arm-identical shell — everything except ``blocks[i].cell``."""
    return eqx.tree_at(lambda m: [b.cell for b in m.blocks], model,
                       replace=[None] * len(model.blocks),
                       is_leaf=lambda x: x is None)


def assert_shared_shell_identical(models: Dict[str, StreamModel]) -> Dict[str, int]:
    """⛔ Blocking check: every arm's non-cell parameters are **bit-identical**.

    "Everything except the cell is bit-identical" is the whole basis of the
    system-level swap. It is asserted, not asserted-in-prose.
    """
    names = list(models)
    ref = jax.tree_util.tree_leaves(eqx.filter(shell_of(models[names[0]]),
                                               eqx.is_inexact_array))
    for n in names[1:]:
        cur = jax.tree_util.tree_leaves(eqx.filter(shell_of(models[n]),
                                                   eqx.is_inexact_array))
        if len(cur) != len(ref):
            raise AssertionError(f"arm '{n}' shell has a different structure")
        for i, (a, b) in enumerate(zip(ref, cur, strict=True)):
            if a.shape != b.shape or not bool(jnp.all(a == b)):
                raise AssertionError(
                    f"arm '{n}' shell leaf {i} differs from '{names[0]}' — "
                    "the swap is not a swap")
    shell_params = int(sum(x.size for x in ref))
    return {"shared_shell_params": shell_params,
            "shared_shell_bytes": 4 * shell_params}


# ==========================================================================
# ⭐ the concrete plan pass — the REAL C2W1 controller
# ==========================================================================
@eqx.filter_jit
def _embed_stream(model: StreamModel, tokens: jnp.ndarray):
    h = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tokens)
    return h + model.pos[: h.shape[1]][None]


@eqx.filter_jit
def _block_chunk_latents(blk, h):
    return jax.vmap(blk.chunk_latents)(h)


@eqx.filter_jit
def _block_forward(blk, h, plan):
    return jax.vmap(blk)(h, plan)


def _chunk_latents(model: StreamModel, tokens: jnp.ndarray):
    """Per-layer chunk latents under the *current* plan, computed concretely.

    Yields ``(layer_index, z)`` with ``z`` of shape ``(B, n_chunks, latent)`` and
    accepts the plan for that layer back through ``send``, coroutine-style, so
    layer ``l+1``'s latents are computed with layer ``l``'s real decisions in
    force. That ordering matters: a plan computed on a forward that ignored the
    earlier layers' memories would not be the plan the model runs.

    ⭐ **The three stages are ``filter_jit``-ed (`pilot-placement-probe`).** This
    changes no result — the same functions on the same inputs — but it is the
    single biggest wall-clock lever in the block. `cluformer-pilot` §8.2 reported
    that **77–84 % of the CLU arm's step is "the plan pass"** and attributed it to
    the Python controller; **measured, only ~1.6 % of the plan pass is the
    controller** — the other ~98 % was this forward, executed EAGERLY op-by-op
    while the differentiable pass next to it ran under ``filter_jit``. Jitting it
    is not a vectorisation of the controller and does not touch a single decision.
    """
    h = _embed_stream(model, tokens)
    for blk in model.blocks:
        z = _block_chunk_latents(blk, h)
        plan = yield z
        h = _block_forward(blk, h, plan)


# --------------------------------------------------------------------------
# ⭐ the two picklable summaries that make a lane a WORKER-PROCESS unit
# --------------------------------------------------------------------------
@dataclass
class LaneControllerSummary:
    """Picklable stand-in for the live :class:`CluControllerV0` of one lane.

    The lane call used to hand its **live controller** back through
    ``_stats["controller"]`` (it reaches :class:`MonitorContext`). A live
    controller owns an Equinox shadow store and the monitor registry, so a lane
    cannot cross a process boundary while it returns one — and shipping the
    store's arrays back per lane per layer per step would cost more than the
    parallelism buys. Everything a consumer actually reads off it is a *summary*:
    guard counts, the verb log, the live records, the stop state.

    ⭐ The SERIAL path returns this too, so serial and pooled runs are the same
    object graph and the equivalence test compares like with like.
    """

    guard_counts: Dict[str, int] = field(default_factory=dict)
    n_live: int = 0
    t: int = 0
    budget: int = 0
    stopped: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    log: List[Dict[str, Any]] = field(default_factory=list)

    def guard_fire_counts(self) -> Dict[str, int]:
        """Duck-types the controller method of the same name."""
        return dict(self.guard_counts)

    @classmethod
    def of(cls, ctrl) -> "LaneControllerSummary":
        return cls(
            guard_counts=dict(ctrl.guard_fire_counts()),
            n_live=int(ctrl.allocator.n_live),
            t=int(ctrl.t),
            budget=int(ctrl.budget),
            stopped=(None if ctrl.stopped is None else str(ctrl.stopped)),
            policy=asdict(ctrl.policy),
            records=[{"slot": int(r.slot), "item_id": int(r.item_id),
                      "center": np.asarray(r.center, dtype=float).tolist(),
                      "payload": float(getattr(r, "payload", 0.0)),
                      "permanent": bool(getattr(r, "permanent", False))}
                     for r in ctrl.allocator.records.values()],
            log=[{"verb": str(v.verb), "applied": bool(v.applied),
                  "reason": str(v.reason), "guard": str(v.guard), "t": int(v.t)}
                 for v in ctrl.log],
        )


class _ClassITrips:
    """Picklable read-only stand-in for the monitor registry.

    The controller consults the registry for **one** thing — ``class_i_tripped()``
    before a memory-mutating verb (doctrine §5, consequence 1) — and never writes
    to it. Within a single :func:`plan_pass` nothing calls ``registry.observe``,
    so that list is **constant for the whole pass**; snapshotting it once and
    handing every lane the snapshot is exactly equivalent to sharing the live
    registry, and it is picklable. (Asserted in the test suite against a registry
    carrying a real class-I trip.)
    """

    __slots__ = ("_names",)

    def __init__(self, names: Sequence[str] = ()):
        self._names = tuple(str(n) for n in names)

    def class_i_tripped(self, window: int = 1) -> List[str]:
        _ = window
        return list(self._names)


# --------------------------------------------------------------------------
# ⭐ the lane-parallel controller (C2W5, probe §8.1)
# --------------------------------------------------------------------------
#: Env forced on every worker **before** it imports JAX. ⛔ Without
#: ``JAX_PLATFORMS=cpu`` each of ``plan_workers`` processes would open its own
#: handle on the job's single GPU and pre-allocate against it. The workers do
#: pure controller bookkeeping (numpy admission geometry + a small Equinox
#: shadow store); they never touch the model.
_WORKER_ENV = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
_POOLS: Dict[Tuple[int, str], ProcessPoolExecutor] = {}
_POOL_FAILED: List[str] = []


def _worker_ping(i: int) -> int:
    """Forces a worker to actually start (and pay its import) at pool creation."""
    return int(os.getpid()) + 0 * int(i)


def _lane_worker(args) -> Dict[str, np.ndarray]:
    """Module-level (hence picklable) entry point for one lane."""
    z_lane, scfg, class_i = args
    return _controller_plan_for_lane(z_lane, scfg, _ClassITrips(class_i))


def _resolve_workers(pcfg: "PilotConfig", n_lanes: int) -> int:
    w = int(getattr(pcfg, "plan_workers", 0) or 0)
    if w < 0:
        w = min(int(n_lanes), os.cpu_count() or 1)
    return max(0, min(w, int(n_lanes)))


def _lane_pool(workers: int, start: str) -> ProcessPoolExecutor:
    """One persistent pool per (workers, start-method), created lazily.

    ⚠ Creating it per step would pay the spawn/import cost 4000 times; the whole
    point is that it is paid once. ``_worker_ping`` is submitted eagerly so the
    cost lands here and not inside the first step's plan-pass timing.
    """
    key = (int(workers), str(start))
    ex = _POOLS.get(key)
    if ex is not None:
        return ex
    saved = {k: os.environ.get(k) for k in _WORKER_ENV}
    os.environ.update(_WORKER_ENV)
    try:
        ex = ProcessPoolExecutor(max_workers=int(workers),
                                 mp_context=mp.get_context(str(start)))
        list(ex.map(_worker_ping, range(2 * int(workers))))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _POOLS[key] = ex
    return ex


def shutdown_lane_pools() -> None:
    """Tear every worker pool down (registered at exit; safe to call twice)."""
    for key in list(_POOLS):
        try:
            _POOLS.pop(key).shutdown(wait=False, cancel_futures=True)
        except Exception:  # a dead pool is already down
            pass


atexit.register(shutdown_lane_pools)


# --------------------------------------------------------------------------
# ⭐ host-memory instrumentation + hygiene (`pilot-checkpoint-resume`)
# --------------------------------------------------------------------------
_GB = 1.0 / (1024.0 ** 3)


def _proc_status_kb(pid: Any = "self") -> Dict[str, float]:
    """``VmRSS``/``VmHWM`` in kB from ``/proc/<pid>/status``; ``{}`` off Linux."""
    out: Dict[str, float] = {}
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    # split(":", 1) already drops the colon — k is the full key.
                    # (A k[:-1] here once truncated the keys to "VmRS"/"VmHW" and
                    # crashed every CSF3 job at the first phase boundary; macOS
                    # has no /proc, so only the cluster ever ran this branch.)
                    k, v = line.split(":", 1)
                    out[k] = float(v.strip().split()[0])
    except OSError:
        pass
    return out


def _ps_rss_kb(pid: Any) -> float:
    """Current RSS in kB via ``ps`` — the no-``/proc`` fallback (macOS dev box).

    ⚠ ``resource.getrusage`` only exposes the *peak* (``ru_maxrss``), so without
    this the "did the hygiene pass actually give the memory back?" question is
    unanswerable off Linux — which is where it has to be answered, because the
    cluster is unreachable from an agent machine. One ``ps`` per phase boundary
    (~20 per run) is free; it is never called in a hot loop.
    """
    import subprocess

    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().split()[0])
    except Exception:
        return float("nan")


def host_rss(*, with_children: bool = True) -> Dict[str, float]:
    """Host RSS in **GB**: ``rss_gb`` (now) and ``hwm_gb`` (peak so far).

    ⭐ **This is the number CSF3 attempt 1 was killed on.** Job 18136619 was
    ``oom_kill``ed with ``MaxRSS = 125.6 GB`` against a ``ReqMem`` of 125.7 GB —
    a *host* kill, not a device OOM — after 22 h of training, in the eval block.
    The kernel's own high-water mark is ``VmHWM``, so on Linux (the cluster)
    that is exactly what is read; elsewhere (macOS, the dev laptop) only the
    peak is available, via ``resource.getrusage``, whose ``ru_maxrss`` is
    **bytes on Darwin and kilobytes on Linux**.

    ⚠ ``VmHWM`` never decreases, which is the point: a jump in ``hwm_gb``
    between two phase boundaries attributes the spike to the phase between them,
    while ``rss_gb`` shows whether the hygiene pass actually gave it back.

    ⚠ Only THIS process is measured by those two. The ``plan_workers`` pool's
    children have their own RSS and the cgroup a job is killed against is the
    **sum**, so ``children_rss_gb`` is reported beside them.
    """
    st = _proc_status_kb("self")
    rec: Dict[str, float] = {}
    if st:
        rec["rss_gb"] = st.get("VmRSS", float("nan")) * 1024.0 * _GB
        rec["hwm_gb"] = st.get("VmHWM", float("nan")) * 1024.0 * _GB
    else:
        ru = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rec["rss_gb"] = _ps_rss_kb(os.getpid()) * 1024.0 * _GB
        rec["hwm_gb"] = ru * (1.0 if sys.platform == "darwin" else 1024.0) * _GB
    if with_children:
        kids, n = 0.0, 0
        for ex in list(_POOLS.values()):
            for pid in list(getattr(ex, "_processes", {}) or {}):
                s = _proc_status_kb(pid)
                v = s.get("VmRSS")  # absent for a zombie/vanished worker even on Linux
                if v is None:
                    v = _ps_rss_kb(pid)
                if not math.isnan(v):  # a NaN read must not poison the whole sum
                    kids += v * 1024.0 * _GB
                    n += 1
        rec["children_rss_gb"] = kids
        rec["n_children"] = float(n)
    return rec


def release_host_memory(*, clear_jax: bool = True) -> None:
    """Drop everything droppable between two eval phases.

    ``jax.clear_caches()`` releases the retained compiled executables of the
    phase just finished (and the tracing/lowering caches behind them) so that
    the NEXT phase's compile does not spike on top of them; ``gc.collect()``
    then reaps the Python-side cycles holding the buffers.

    ⛔ **Value-inert.** It drops *executables*, never data: a program that is
    needed again is re-traced and re-compiled, and XLA compilation of the same
    HLO is deterministic. Nothing here touches a weight, a plan or a metric.
    """
    if clear_jax:
        jax.clear_caches()
    gc.collect()


def _plan_lanes(zc: np.ndarray, scfg: CluSystemConfig, class_i: Sequence[str],
                pcfg: "PilotConfig") -> Tuple[List[Dict[str, np.ndarray]], str]:
    """Run every lane's controller — serially, or one process per lane.

    ⭐ **Lanes are independent**: each :func:`_controller_plan_for_lane` builds
    its own controller over its own latents and shares nothing but the (now
    snapshotted) class-I trip list, so the batch axis is embarrassingly parallel.
    **Layers are not** — layer ``l+1``'s latents are computed from layer ``l``'s
    decisions — so this is a batch-axis cut only, and ``ex.map`` preserves lane
    order, which is what keeps the stacked plan lane-for-lane identical.

    A broken pool is a wall-clock failure, never a correctness one: it falls back
    to the serial path and says so, once.
    """
    B = int(zc.shape[0])
    workers = _resolve_workers(pcfg, B)
    if workers > 1 and not _POOL_FAILED:
        try:
            ex = _lane_pool(workers, str(pcfg.plan_mp_start))
            args = [(np.asarray(zc[b]), scfg, tuple(class_i)) for b in range(B)]
            return list(ex.map(_lane_worker, args)), f"pool[{workers}/{pcfg.plan_mp_start}]"
        except Exception as exc:      # BrokenProcessPool, pickling, OS limits...
            _POOL_FAILED.append(repr(exc))
            print(f"[plan_pass] ⚠ lane pool unavailable ({exc!r}); "
                  f"falling back to the serial controller", flush=True)
    stub = _ClassITrips(class_i)
    mode = "serial" if not _POOL_FAILED else "serial (pool failed)"
    return [_controller_plan_for_lane(zc[b], scfg, stub) for b in range(B)], mode


def _controller_plan_for_lane(z_lane: np.ndarray, scfg: CluSystemConfig,
                              registry) -> Dict[str, np.ndarray]:
    """Run the real controller over one lane's chunk latents.

    Exercises the designed verbs in the order :meth:`CluSystem.write_stream`
    does: ``admit`` (with the reach margin when admission is staged) -> the
    allocator's ``place``/``evict`` under budget pressure -> ``decay`` for
    per-item lifetimes. Refusals, evictions and decay factors all land in the
    plan, so the differentiable pass reproduces them exactly.
    """
    n_chunks, dim = z_lane.shape
    K = int(scfg.capacity)
    d, m = int(scfg.addr_dim), int(scfg.payload_dim)
    ctrl = make_controller(scfg, registry)
    slot = np.zeros((n_chunks,), np.int32)
    admitted = np.zeros((n_chunks,), np.float32)
    group_scale = np.ones((n_chunks, K), np.float32)
    reset = np.zeros((n_chunks, K), np.float32)
    sites = np.full((n_chunks, K, dim), 1e3, np.float32)
    live = np.zeros((n_chunks, K), np.float32)
    stats = {"offers": n_chunks, "refused": 0, "evicted": 0, "guards": {}}

    def slot_of(iid):
        for r in ctrl.allocator.records.values():
            if r.item_id == int(iid):
                return int(r.slot)
        return None

    rows: List[dict] = []
    for c in range(n_chunks):
        z = np.asarray(z_lane[c], dtype=float)
        addr, pay = z[:d], z[d: d + m]
        # ⚠ Slots BEFORE the offer. Which slots an admission evicted cannot be
        # recovered from the evicted item's id — the allocator has already
        # removed its record — so the eviction is detected as a set difference.
        # (An earlier version sliced `ctrl.log` by a COUNT of past evictions and
        # looked the victim up by id: it over-counted evictions and then failed
        # to mark ANY reset, so evicted groups were never re-drawn.)
        n0 = len(ctrl.log)
        slots_before = {int(r.slot) for r in ctrl.allocator.records.values()}
        res = ctrl.admit(
            c, addr, float(pay[0]) if m else 0.0, utility=1.0,
            reach_margin=None,
            permanent=False,
            leak=float(scfg.leak) if scfg.stage_lifetimes else 0.0,
        )
        admitted[c] = 1.0 if res.applied else 0.0
        if not res.applied:
            stats["refused"] += 1
        else:
            s = slot_of(c)
            slot[c] = 0 if s is None else s
        slots_after = {int(r.slot) for r in ctrl.allocator.records.values()}
        n_ev = sum(1 for v in ctrl.log[n0:] if v.verb == "evict" and v.applied)
        stats["evicted"] += n_ev
        for ev in (slots_before - slots_after) | (
                {int(slot[c])} if (n_ev and admitted[c]) else set()):
            reset[c, ev] = 1.0
        row = dict(res.detail.get("row", {}))
        row.update({"t": c, "item_id": c, "verb": "admit", "applied": bool(res.applied),
                    "guard": res.guard,
                    "decision": row.get("decision", res.reason)})
        rows.append(row)
        if scfg.stage_lifetimes:
            dv = ctrl.decay(1)
            for iid, f in (dv.detail or {}).get("factors", {}).items():
                s = slot_of(int(iid))
                if s is not None:
                    group_scale[c, s] = float(f)
        for r in ctrl.allocator.records.values():
            s = int(r.slot)
            live[c, s] = 1.0
            a = np.asarray(r.center, dtype=float).reshape(-1)[:d]
            sites[c, s, : a.shape[0]] = a
            if m:
                sites[c, s, d: d + m] = float(getattr(r, "payload", 0.0))
    stats["guards"] = dict(ctrl.guard_fire_counts())
    stats["n_live_end"] = int(ctrl.allocator.n_live)
    stats["rows"] = rows
    # ⭐ a SUMMARY, not the live controller — a lane must be able to cross a
    # process boundary (see :class:`LaneControllerSummary`).
    stats["controller"] = LaneControllerSummary.of(ctrl)
    return {"slot": slot, "admitted": admitted, "group_scale": group_scale,
            "reset": reset, "sites": sites, "live": live,
            "retry": np.zeros((n_chunks,), np.int32), "_stats": stats}


def plan_pass(model: StreamModel, tokens: jnp.ndarray, pcfg: PilotConfig,
              *, registry=None, blank: bool = False) -> Tuple[List[WritePlan], Dict]:
    """Concrete pass 1: the real controller emits one plan per layer.

    ``blank=True`` produces the **blank-store control** (collapse mode #4): the
    store is read exactly as in the live arm but nothing is ever admitted.

    ⭐ With ``pcfg.plan_workers > 1`` the ``batch`` lane-calls of each layer run
    in worker **processes** (:func:`_plan_lanes`) — the only CPU-serial term left
    after the forward was jitted, priced at ≈ 2.6-3.7 s/step at pilot scale.
    **Decision-replay is the spec**, so the pooled path is required to produce a
    lane-for-lane identical :class:`WritePlan`; that is a blocking test, not a
    hope.
    """
    scfg = pcfg.store_cfg()
    B, T = tokens.shape
    n_chunks = T // int(pcfg.memory_cfg().chunk)
    K, dim = int(scfg.capacity), int(scfg.dim)
    registry = registry if registry is not None else default_registry(loud=False)
    # ⭐ ONE snapshot of the class-I trip list for the whole pass. Nothing here
    # calls ``registry.observe``, so it cannot change mid-pass; snapshotting is
    # what lets a lane be a picklable unit of work (see :class:`_ClassITrips`).
    class_i = tuple(registry.class_i_tripped()
                    if hasattr(registry, "class_i_tripped") else ())
    gen = _chunk_latents(model, tokens)
    plans: List[WritePlan] = []
    diag: Dict[str, Any] = {"layers": []}
    z = gen.send(None)
    while True:
        if blank:
            plan = jax.tree_util.tree_map(
                lambda a: jnp.broadcast_to(a, (B,) + a.shape),
                blank_plan(n_chunks, K, dim))
            lstats = {"offers": 0, "refused": 0, "evicted": 0, "n_live_end": 0}
        elif not pcfg.real_controller:
            plan = jax.tree_util.tree_map(
                lambda a: jnp.broadcast_to(a, (B,) + a.shape),
                round_robin_plan(n_chunks, K, dim))
            lstats = {"controller": "round_robin (probe only)"}
        else:
            zc = np.asarray(jax.lax.stop_gradient(z))
            lanes, lane_mode = _plan_lanes(zc, scfg, class_i, pcfg)
            lstats = {k: int(sum(la["_stats"][k] for la in lanes))
                      for k in ("offers", "refused", "evicted", "n_live_end")}
            lstats["lane_mode"] = lane_mode
            gsum: Dict[str, int] = {}
            for la in lanes:
                for gk, gv in la["_stats"]["guards"].items():
                    gsum[gk] = gsum.get(gk, 0) + int(gv)
            lstats["guards"] = gsum
            lstats["rows"] = [r for la in lanes for r in la["_stats"]["rows"]]
            lstats["controllers"] = [la["_stats"]["controller"] for la in lanes]
            plan = WritePlan(**{
                k: jnp.asarray(np.stack([la[k] for la in lanes]))
                for k in ("slot", "admitted", "group_scale", "reset", "sites",
                          "live", "retry")})
        # the confidence gate: a retry round is asked for where the read's
        # settle residual is worst. Concrete by necessity (a discrete verb).
        plans.append(plan)
        diag["layers"].append(lstats)
        try:
            z = gen.send(plan)
        except StopIteration:
            break
    return plans, diag


# ==========================================================================
# loss / training
# ==========================================================================
def _nll(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(lp, targets[..., None].astype(jnp.int32),
                                         axis=-1))


def loss_fn(model: StreamModel, tokens: jnp.ndarray, targets: jnp.ndarray,
            plans, read_mode: Optional[str] = None,
            verlet: Optional[Tuple[int, int]] = None) -> jnp.ndarray:
    """Mean next-token NLL in nats. ``plans`` is one batched plan per layer."""
    logits = jax.vmap(lambda t, *p: model(t, list(p), read_mode, verlet))(tokens, *plans)
    return _nll(logits, targets)


@eqx.filter_jit
def _eval_loss(model, tokens, targets, plans, read_mode, verlet):
    """``loss_fn`` under ``filter_jit`` — ``read_mode``/``verlet`` are Python
    values and become static, so the D5 Verlet-budget override still re-compiles
    per budget exactly as before. Evaluation was running the whole block forward
    EAGERLY (`pilot-placement-probe` §plan-pass); this is the same function on
    the same inputs."""
    return loss_fn(model, tokens, targets, plans, read_mode, verlet)


@eqx.filter_jit
def _train_step(model, opt_state, tokens, targets, plans, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tokens, targets, plans)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


def _microbatch(a, n: int):
    """``(B, ...) -> (n, B//n, ...)`` — the microbatch axis, leading."""
    a = jnp.asarray(a)
    return a.reshape((int(n), a.shape[0] // int(n)) + a.shape[1:])


@eqx.filter_jit
def _accum_grads(model, tokens, targets, plans, n_micro: int):
    """Mean loss + mean gradient over ``n_micro`` SEQUENTIAL microbatches.

    ⭐ `csf3-memory-fit`. The backward's peak activation is set by the
    microbatch, not the batch; the update is still one step on the full
    effective batch. ``lax.scan`` (not a Python loop) so XLA cannot schedule the
    microbatch backwards concurrently and hand the memory back.
    """
    n_micro = int(n_micro)
    tk, tg = _microbatch(tokens, n_micro), _microbatch(targets, n_micro)
    pl = jax.tree_util.tree_map(lambda a: _microbatch(a, n_micro), plans)
    params = eqx.filter(model, eqx.is_inexact_array)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)

    def body(carry, xs):
        g_acc, l_acc = carry
        t, y, p = xs
        loss, g = eqx.filter_value_and_grad(loss_fn)(model, t, y, p)
        g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc,
                                       eqx.filter(g, eqx.is_inexact_array))
        return (g_acc, l_acc + loss), None

    (g_sum, l_sum), _ = jax.lax.scan(body, (zeros, jnp.asarray(0.0)), (tk, tg, pl))
    scale = 1.0 / float(n_micro)
    return (l_sum * scale,
            jax.tree_util.tree_map(lambda a: a * scale, g_sum))


@eqx.filter_jit
def _train_step_accum(model, opt_state, tokens, targets, plans, optimizer,
                      n_micro: int):
    loss, grads = _accum_grads(model, tokens, targets, plans, int(n_micro))
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


def make_optimizer(pcfg: PilotConfig):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=pcfg.lr, warmup_steps=max(1, pcfg.warmup),
        decay_steps=max(2, pcfg.steps), end_value=pcfg.lr * 0.1)
    return optax.chain(optax.clip_by_global_norm(pcfg.grad_clip),
                       optax.adamw(sched, weight_decay=pcfg.weight_decay))


def train_arm(name: str, model: StreamModel, pcfg: PilotConfig, batches,
              *, log_every: int = 25, log: Optional[List] = None,
              monitor_registry=None, monitor_tokens=None,
              monitor_out: Optional[List] = None,
              probe=None, probe_out: Optional[List] = None):
    """Train one arm. ⭐ ``batches`` must be the SAME sequence for every arm.

    ``monitor_registry`` (+ ``monitor_tokens``) runs the 13 monitors every
    ``pcfg.monitor_every`` steps **through one persistent registry**, which is
    what gives monitor #6 (objective divergence) the window of
    ``(write_loss, acq)`` pairs it needs — on a single observation it is
    inapplicable, and an inapplicable monitor is not a passing monitor.

    ⚠ ``pcfg.store_watch`` (default ON) additionally runs
    :func:`store_health_probe` on **fixed** tokens at the same cadence, starting
    with an untrained reading taken before the first update, and returns the
    series as ``store_health``. That is §7.27's watch-item: it lands in the run
    artifact by default, so a cluster job cannot destroy its own store silently.

    ⭐ ``probe`` (`c2w6-anti-erosion`) is an OPTIONAL caller-supplied instrument
    ``probe(model, step, tag) -> dict`` run on the **same schedule and the same
    fixed tokens** as the watch (untrained reading first, then every
    ``monitor_every`` steps and at the last step); its records are appended to
    ``probe_out`` and returned as ``probe_series``. ``None`` (default) is the
    shipped loop, unchanged — this exists so an experiment can carry its own
    per-well telemetry (I2) without forking the trainer.
    """
    optimizer = make_optimizer(pcfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    n_micro = max(1, int(getattr(pcfg, "accum_steps", 1)))
    if n_micro > 1 and int(pcfg.batch) % n_micro != 0:
        raise ValueError(f"accum_steps={n_micro} does not divide batch={pcfg.batch}")
    hist, t0 = [], time.time()
    plan_s = 0.0
    watch: List[Dict[str, Any]] = []
    watch_tokens = None
    watch_on = bool(getattr(pcfg, "store_watch", False))
    base_depth = base_spread = float("nan")
    probes: List[Dict[str, Any]] = probe_out if probe_out is not None else []

    def _probe(step: int, tag: str) -> None:
        """One caller-supplied telemetry reading (`c2w6-anti-erosion`'s I2)."""
        if probe is None:
            return
        rec = probe(model, int(step), str(tag))
        if rec is None:
            return
        rec = dict(rec)
        rec["at_step"], rec["tag"] = int(step), str(tag)
        probes.append(rec)

    def _watch(step: int, tag: str) -> None:
        """One §7.27 reading, printed in flight and kept for the artifact."""
        nonlocal watch_on, base_depth, base_spread
        rec = store_health_probe(model, pcfg, watch_tokens)
        if not rec.get("applicable", False):
            watch_on = False          # not a store arm: nothing to watch
            return
        rec["at_step"] = int(step)
        rec["tag"] = tag
        if tag == "untrained":
            base_depth = float(rec.get("depth_median", float("nan")))
            base_spread = float(rec.get("qstar_payload_spread", float("nan")))
        rec["untrained_depth_median"] = base_depth
        rec["depth_ratio_vs_untrained"] = (
            float(rec.get("depth_median", float("nan")) / base_depth)
            if np.isfinite(base_depth) and base_depth > 0 else float("nan"))
        rec["untrained_qstar_payload_spread"] = base_spread
        rec["spread_ratio_vs_untrained"] = (
            float(rec.get("qstar_payload_spread", float("nan")) / base_spread)
            if np.isfinite(base_spread) and base_spread > 0 else float("nan"))
        watch.append(rec)
        print(f"[watch/{name}] step {step} ({tag}): depth "
              f"{rec.get('depth_median', float('nan')):.4g} "
              f"(x{rec['depth_ratio_vs_untrained']:.3g} vs untrained) | "
              f"q* payload spread {rec.get('qstar_payload_spread', float('nan')):.4g} "
              f"(x{rec['spread_ratio_vs_untrained']:.3g}) | "
              f"n_live {rec.get('n_live')}", flush=True)

    for i, (x, y) in enumerate(batches):
        tk = jnp.asarray(x, dtype=jnp.int32)
        tg = jnp.asarray(y, dtype=jnp.int32)
        if watch_on and watch_tokens is None:
            # ⭐ FIXED tokens for the whole series (the monitors' batch when there
            # is one), and the baseline is taken BEFORE the first update — an
            # "untrained vs trained" ratio measured on different tokens is not a
            # ratio of anything.
            watch_tokens = (jnp.asarray(monitor_tokens, dtype=jnp.int32)
                            if monitor_tokens is not None else tk)
            _watch(0, "untrained")
        if i == 0:
            _probe(0, "untrained")
        tp = time.time()
        plans, _ = plan_pass(model, tk, pcfg)
        plan_s += time.time() - tp
        if n_micro > 1:
            model, opt_state, loss = _train_step_accum(
                model, opt_state, tk, tg, plans, optimizer, n_micro)
        else:
            model, opt_state, loss = _train_step(
                model, opt_state, tk, tg, plans, optimizer)
        hist.append(float(loss))
        if (monitor_registry is not None and monitor_tokens is not None
                and i > 0 and i % max(1, pcfg.monitor_every) == 0):
            mp = monitor_pass(model, pcfg, monitor_tokens,
                              registry=monitor_registry, write_loss_now=float(loss))
            mp["at_step"] = i
            if monitor_out is not None:
                monitor_out.append(mp)
        if watch_on and i > 0 and (i % max(1, pcfg.monitor_every) == 0
                                   or i == pcfg.steps - 1):
            _watch(i, "trained")
        if i > 0 and (i % max(1, pcfg.monitor_every) == 0
                      or i == pcfg.steps - 1):
            _probe(i, "trained")
        if i % log_every == 0 or i == pcfg.steps - 1:
            row = {"arm": name, "step": i, "nll": float(loss),
                   "bpc": bits_per_character(float(loss)),
                   "wall_s": time.time() - t0, "plan_s": plan_s}
            if log is not None:
                log.append(row)
            # ⭐ `csf3-memory-fit` §7 claimed this print existed; it did not (Head
            # checked attempt 1's `.out`/`.err`: no such line anywhere), so the
            # first A100 step time for this model had to be reconstructed from
            # wallclock. It exists now.
            print(f"[train/{name}] step {i}/{pcfg.steps} | nll {row['nll']:.4f} "
                  f"bpc {row['bpc']:.4f} | wall_s {row['wall_s']:.1f} "
                  f"({row['wall_s'] / (i + 1):.2f} s/step) | plan_s {plan_s:.1f} "
                  f"({100.0 * plan_s / max(row['wall_s'], 1e-9):.0f}%)", flush=True)
    return model, {"loss_history": hist, "wall_s": time.time() - t0,
                   "plan_pass_s": plan_s,
                   "plan_pass_frac": plan_s / max(time.time() - t0, 1e-9),
                   "store_health": watch, "probe_series": probes}


def evaluate(model: StreamModel, pcfg: PilotConfig, batches, *, blank: bool = False,
             verlet: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
    """Held-out bpc on a **contiguous, order-preserving** iterator."""
    tot, n = 0.0, 0
    for x, y in batches:
        tk = jnp.asarray(x, dtype=jnp.int32)
        tg = jnp.asarray(y, dtype=jnp.int32)
        plans, _ = plan_pass(model, tk, pcfg, blank=blank)
        tot += float(_eval_loss(model, tk, tg, plans, None, verlet))
        n += 1
    nll = tot / max(n, 1)
    return {"nll": nll, "bpc": bits_per_character(nll), "n_batches": n}


def dynamic_eval(model: StreamModel, pcfg: PilotConfig, batches,
                 lrs: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """⛔ **Dynamic evaluation (Krause et al., ICML 2018) — a MANDATORY column.**

    The published criterion-3 weakness (1.08 bpc on enwik8). It is applied to
    **every arm in the same table**, with the same optimiser and the same LR,
    because it is a *substitute* for exactly what a test-time memory sells: SGD
    on the weights over the test stream. §A9.11 made it mandatory; the
    pre-committed consequence (task D3) is that **if the dividend vanishes once
    dynamic evaluation is in the table, the primary is dead.**

    Protocol: strictly causal — for each batch, score it FIRST with the current
    weights, then take one SGD step on it. No batch is ever scored by weights
    that have seen it. ⭐ **The learning rate is swept and the BEST (lowest bpc)
    is reported per arm**, because a badly-tuned substitute is a weak substitute
    and a weak substitute flatters us. Krause et al. tune it on validation; we
    give every arm the same grid and take its own best.

    ⚠ `csf3-memory-fit`: dynamic evaluation takes the SAME backward as a training
    step, so it inherits ``pcfg.accum_steps`` — otherwise a run that needed the
    microbatch lever to fit would OOM in the one column the job header's
    cut-order forbids cutting.
    """
    grid = list(lrs) if lrs is not None else [pcfg.dyneval_lr, 10 * pcfg.dyneval_lr,
                                              100 * pcfg.dyneval_lr]
    n_micro = max(1, int(getattr(pcfg, "accum_steps", 1)))
    bat = list(batches)
    per_lr = {}
    for lr in grid:
        opt = optax.sgd(float(lr))
        m = model
        st = opt.init(eqx.filter(m, eqx.is_inexact_array))
        tot, n = 0.0, 0
        for x, y in bat:
            tk = jnp.asarray(x, dtype=jnp.int32)
            tg = jnp.asarray(y, dtype=jnp.int32)
            plans, _ = plan_pass(m, tk, pcfg)
            if n_micro > 1:
                loss, grads = _accum_grads(m, tk, tg, plans, n_micro)
            else:
                loss, grads = eqx.filter_value_and_grad(loss_fn)(m, tk, tg, plans)
            tot += float(loss)      # scored BEFORE the update — strictly causal
            n += 1
            upd, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
            m = eqx.apply_updates(m, upd)
        per_lr[float(lr)] = bits_per_character(tot / max(n, 1))
    best_lr = min(per_lr, key=per_lr.get)
    return {"bpc": per_lr[best_lr], "nll": per_lr[best_lr] * math.log(2.0),
            "best_lr": best_lr, "per_lr": per_lr, "n_batches": len(bat)}


# ==========================================================================
# ⭐ S2 — the gradient probe (the single most valuable measurement, task §3)
# ==========================================================================
def gradient_probe(model: StreamModel, pcfg: PilotConfig, tokens, targets,
                   ) -> Dict[str, Any]:
    """``||dL/dphi||`` through the trajectory read vs the settled-point arm's 0.0.

    ⭐ This is T3 tested **in-system** rather than in a probe. ``d q*/d q0 = 0``
    exactly, so a settled-point read sends **no** gradient to its read-in; the
    trajectory read is the only channel **through the read** that does. The two
    arms here are identical models differing ONLY in
    ``StreamMemoryConfig.read_mode``, so the comparison is internal and needs no
    baseline.

    ⚠ It is NOT the only channel to ``phi`` in the block. With
    ``atom_place_radius > 0`` (the shipped run-1/2/3 config) H1b's localized
    placement is a differentiable write-side path to ``phi`` that neither the
    theorem nor the sign write closes (see ``StreamMemoryConfig.write_sign``),
    so the settled-point arm's ``grad_phi`` is a non-zero floor and the ratio
    below UNDERSTATES the read's share. ``erosion_partition = True`` closes it
    (§A22: 27 % of layer-0's ``phi`` gradient).

    ⚠ `csf3-memory-fit`: ``pcfg.probe_lanes > 0`` cuts the batch here too (two
    full backwards, back to back). Default OFF — this one moves a **published**
    magnitude. The CONTRAST S2 claims is lane-count-invariant because both modes
    read the same lanes; ``n_lanes`` is reported so the table can say so.
    """
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    tg = jnp.asarray(targets, dtype=jnp.int32)
    n_lanes = int(getattr(pcfg, "probe_lanes", 0) or 0)
    if 0 < n_lanes < int(tk.shape[0]):
        tk, tg = tk[:n_lanes], tg[:n_lanes]
    out: Dict[str, Any] = {"n_lanes": int(tk.shape[0])}
    plans, _ = plan_pass(model, tk, pcfg)
    for mode in ("trajectory", "settled_point"):
        t0 = time.time()
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tk, tg, plans, mode)
        jax.block_until_ready(grads)
        wall = time.time() - t0
        out[mode] = {
            "loss": float(loss),
            "grad_phi": _norm([b.phi for b in grads.blocks]),
            "grad_psi": _norm([getattr(b.cell, "psi", None) for b in grads.blocks]),
            "grad_store": _norm([getattr(getattr(b.cell, "clu", None),
                                         "potential_net", None)
                                 for b in grads.blocks]),
            "grad_gamma": _norm([getattr(b.cell, "log_gamma_addr", None)
                                 for b in grads.blocks]
                                + [getattr(b.cell, "log_gamma_read", None)
                                   for b in grads.blocks]),
            "grad_mass": _norm([getattr(getattr(b.cell, "clu", None), "log_mass", None)
                                for b in grads.blocks]),
            "grad_embed": _norm([grads.embed]),
            "wall_s": wall,
        }
    # ⭐ Separate the TWO reasons the settled-point arm can read exactly zero:
    # (i) the theorem (d q*/d q0 = 0), and (ii) sign-SGD's zero derivative, which
    # severs the inner loop's d(store state)/d(phi). The plain-SGD write leaves
    # channel (ii) open, so its settled-point number is the theorem's alone.
    # ⚠ "exactly zero" presumes atom_place_radius == 0: H1b's placement is a
    # THIRD channel that neither (i) nor (ii) closes, so at the shipped 0.3 both
    # settled-point numbers are non-zero unless erosion_partition is on.
    if model.blocks[0].cell.mcfg.write_sign:
        import dataclasses as _dc

        alt = _dc.replace(model.blocks[0].cell.mcfg, write_sign=False)
        m_alt = _swap_mcfg(model, alt)
        pl_alt, _ = plan_pass(m_alt, tk, pcfg)
        for mode in ("trajectory", "settled_point"):
            _, ga = eqx.filter_value_and_grad(loss_fn)(m_alt, tk, tg, pl_alt, mode)
            out.setdefault("plain_sgd_write", {})[mode] = {
                "grad_phi": _norm([b.phi for b in ga.blocks]),
                "grad_store": _norm([getattr(getattr(bb.cell, "clu", None),
                                             "potential_net", None)
                                     for bb in ga.blocks]),
            }
        pa = out["plain_sgd_write"]["trajectory"]["grad_phi"]
        pb = out["plain_sgd_write"]["settled_point"]["grad_phi"]
        out["plain_sgd_write"]["ratio_traj_over_point"] = (pa / pb) if pb > 0 else float("inf")
    a = out["trajectory"]["grad_phi"]
    b = out["settled_point"]["grad_phi"]
    out["ratio_traj_over_point"] = (a / b) if b > 0 else float("inf")
    out["wall_ratio_traj_over_point"] = (out["trajectory"]["wall_s"]
                                         / max(out["settled_point"]["wall_s"], 1e-9))
    return out


def _swap_mcfg(model: StreamModel, mcfg) -> StreamModel:
    """Rebuild every store cell around a new (static) ``StreamMemoryConfig``.

    Parameters are carried over unchanged, so the two arms are the SAME model —
    only the static config differs. ``eqx.tree_at`` cannot do this (a static
    field is part of the treedef, not a leaf), hence the manual reconstruction.
    """
    def swap(cell):
        if not hasattr(cell, "mcfg"):
            return cell
        return eqx.tree_at(lambda c: [c.clu, c.psi, c.log_gamma_addr, c.log_gamma_read],
                           _blank_like(cell, mcfg),
                           replace=[cell.clu, cell.psi, cell.log_gamma_addr,
                                    cell.log_gamma_read])
    return eqx.tree_at(lambda m: [b.cell for b in m.blocks], model,
                       replace=[swap(b.cell) for b in model.blocks])


def _blank_like(cell, mcfg):
    from chlu.core.blocks import CluStoreCell

    return CluStoreCell(cell.cfg, mcfg, key=jax.random.PRNGKey(0))


def _norm(mods) -> float:
    tot = 0.0
    for mod in mods:
        if mod is None:
            continue
        for leaf in jax.tree_util.tree_leaves(eqx.filter(mod, eqx.is_inexact_array)):
            tot += float(jnp.sum(jnp.asarray(leaf) ** 2))
    return float(math.sqrt(tot))


def allocation_liveness(model: StreamModel, pcfg: PilotConfig, tokens, targets
                        ) -> Dict[str, Any]:
    """⭐ The anti-attractor anchor (T3 corollary), measured at init.

    Allocation collapse is a **gradient-flow attractor**: at the all-endpoint
    corner the policy's own gradient dies and it cannot leave. The C2W1
    controller's policy is *rule-based*, not logit-parameterised, so there are no
    policy logits to differentiate — the learned quantity that steers allocation
    is ``phi``'s **address head**. This reports its gradient norm beside the
    address-utilisation entropy, which is the liveness anchor the design rule
    actually needs.

    ⚠ `csf3-memory-fit`: ``pcfg.liveness_lanes > 0`` runs it on that many lanes of
    the batch instead of all of them. It is a **full-batch backward taken at
    init** — the crash site of CSF3 run 1 — and both quantities it reports are
    per-lane objects, so the anchor survives the cut. The lane count is returned.
    """
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    tg = jnp.asarray(targets, dtype=jnp.int32)
    n_lanes = int(getattr(pcfg, "liveness_lanes", 0) or 0)
    if 0 < n_lanes < int(tk.shape[0]):
        tk, tg = tk[:n_lanes], tg[:n_lanes]
    plans, _ = plan_pass(model, tk, pcfg)
    _, grads = eqx.filter_value_and_grad(loss_fn)(model, tk, tg, plans)
    ent, occ = [], []
    for p in plans:
        s = np.asarray(p.slot).reshape(-1)
        adm = np.asarray(p.admitted).reshape(-1) > 0.5
        s = s[adm]
        K = int(np.asarray(p.group_scale).shape[-1])
        cnt = np.bincount(s, minlength=K).astype(float)
        pr = cnt / max(cnt.sum(), 1.0)
        nz = pr[pr > 0]
        ent.append(float(-(nz * np.log(nz)).sum() / math.log(max(K, 2))))
        occ.append(float((cnt > 0).mean()))
    return {
        "grad_phi_addr_head": _norm([b.phi for b in grads.blocks]),
        "n_lanes": int(tk.shape[0]),
        "policy_logits": None,
        "policy_logits_note": "the C2W1 controller's policy is RULE-BASED, not "
                              "logit-parameterised: there are no policy logits. "
                              "phi's address head is the learned allocator.",
        "slot_entropy_normalised_per_layer": ent,
        "slot_occupancy_per_layer": occ,
    }


def anytime_curve(model: StreamModel, pcfg: PilotConfig, batches,
                  budgets: Sequence[Tuple[int, int]],
                  *, hygiene: Optional[bool] = None) -> List[Dict[str, Any]]:
    """⭐ D5 — accuracy vs Verlet-steps-per-read, on a model trained at ONE budget.

    ⚠ **SHAPE claim only** (charter §A3). The anytime figure is occupied — DEQs,
    Energy-Based Transformers, Titans-Revisited all own it — so no uniqueness is
    claimed and none may be quoted from this curve. ⚠ The trajectory read costs
    **17.1x** the point read (`trainability-spike` §3) and that price travels
    with every point on it.

    ⭐ `pilot-checkpoint-resume`: ``verlet`` is a *static* argument of
    :func:`_eval_loss`, so every budget compiles its OWN program and **none of
    them is ever re-used** — five one-shot executables stacking inside a single
    phase. With ``hygiene`` (default: ``pcfg.eval_cache_hygiene``) each is
    released before the next is compiled.
    """
    hyg = bool(pcfg.eval_cache_hygiene if hygiene is None else hygiene)
    bat = list(batches)
    out = []
    for b in budgets:
        t0 = time.time()
        r = evaluate(model, pcfg, iter(bat), verlet=tuple(int(x) for x in b))
        out.append({"address_steps": int(b[0]), "read_steps": int(b[1]),
                    "verlet_per_read": int(b[0]) + int(b[1]),
                    "bpc": r["bpc"], "wall_s": time.time() - t0})
        if hyg:
            release_host_memory()
    return out


# ==========================================================================
# monitors — all 13, as reported artifacts
# ==========================================================================
def store_self_probe(cell, state, sites, live, *, payload_tol: float = 0.1
                     ) -> Dict[str, Any]:
    """⭐ The store's **label-free self-probe**: re-read every live item's own site.

    This is what turns "the monitors ran" into "the monitors were exercised": #5
    (addressing), #6 (objective divergence), #9 (lifetimes) and #4 (blank) are
    all *inapplicable* without it, and an inapplicable monitor is not a passing
    monitor.

    For each live item the read is launched at its recorded address on the
    payload-zero manifold and the recovered payload block is decoded to the
    NEAREST STORED payload (N110's honest metric — never an absolute threshold,
    which flatters a store whose payloads happen to be far apart).
    """
    idx = [i for i in range(sites.shape[0]) if live[i] > 0.5]
    d, m = int(cell.cfg.addr_dim), int(cell.cfg.payload_dim)
    if len(idx) == 0:
        return {"acq": float("nan"), "n_probed": 0}
    pays = np.stack([sites[i, d: d + m] for i in idx])
    got, ret = [], []
    for i in idx:
        q = jnp.asarray(sites[i], dtype=jnp.float32)
        r = np.asarray(cell.read(state, q))
        got.append(r[d: d + m])
        D, _s = cell_group_depth(cell, state, i, sites[i, :d])
        ret.append(D)
    got = np.stack(got)
    dist = np.linalg.norm(got[:, None, :] - pays[None, :, :], axis=-1)
    hit = (np.argmin(dist, axis=1) == np.arange(len(idx)))
    return {
        "acq": float(hit.mean()),
        "strict": float((np.linalg.norm(got - pays, axis=-1) <= payload_tol).mean()),
        "chance": 1.0 / max(len(idx), 1),
        "n_probed": len(idx),
        "retention": [float(x) for x in ret],
        "payload_abs": [float(abs(p[0])) for p in pays],
        "decoded": got.tolist(),
    }


def cell_group_depth(cell, state, slot: int, center) -> Tuple[float, float]:
    """``(D_i, s_i)`` of an item's own wells, read off the LIVE atom state.

    Mirrors :meth:`LearnedVStore.group_stats` but against the per-sequence state
    rather than the parameter init (the store is written at inference here, so
    the parameters are not where the item lives).
    """
    mrows = np.asarray(cell.group_matrix[int(slot)], dtype=bool)
    A = np.asarray(state.amp, dtype=float)[mrows] ** 2
    sw = np.exp(np.asarray(state.log_width, dtype=float)[mrows])
    c = np.asarray(state.centers, dtype=float)[mrows]
    z = np.zeros((int(cell.cfg.dim),), dtype=float)
    z[: int(cell.cfg.addr_dim)] = np.asarray(center, dtype=float)[: int(cell.cfg.addr_dim)]
    d2 = np.sum((c - z[None, :]) ** 2, axis=-1)
    w = A * np.exp(-d2 / (2.0 * sw ** 2 + 1e-12))
    D = float(np.sum(w))
    s_eff = float(np.sum(w * sw) / max(np.sum(w), 1e-12)) if D > 0 else float(np.mean(sw))
    return D, s_eff


@eqx.filter_jit
def _cell_write(cell, state, z, plan_c):
    """One chunk-write under ``filter_jit`` — used by the §7.27 watch only.

    ⚠ The watch replays a whole sequence's writes at every observation, and an
    EAGER replay of ``write_inner_steps = 40`` costs ~49 s per reading at pilot
    store geometry on this laptop (measured) — which is not a thing you run every
    25 steps. Same function, same inputs, one compile. ``monitor_pass``'s own
    replay is deliberately left alone (its numbers are published).
    """
    return cell.write(state, z, plan_c)


def store_health_probe(model: StreamModel, pcfg: PilotConfig, tokens, *,
                       layer: int = 0) -> Dict[str, Any]:
    """⚠ **§7.27's in-flight watch: is the outer loop destroying the store?**

    The probe measured 200 outer steps driving the in-block store's well depth
    from ``0.0288`` to ``4.95e-63`` at the shipped config — the unnamed cause of
    the pilot's monitor #9 ``Delta_ret = 7.8e-86``. A 4000-step cluster run that
    only reports this at the end has measured a store that its own optimiser
    deleted, and nobody knew until the job was over. So it is logged **during**
    training (Head ruling 3, 2026-08-01).

    Two numbers, and both are required:

    * ``depth_median`` — the median fitted depth ``D_i`` of the live items' own
      wells (:func:`cell_group_depth`), reported as a ratio to the **untrained**
      reading of the same tokens;
    * ``qstar_payload_spread`` — the BETWEEN-ITEM range of the settled point's
      payload coordinate ``q*[payload]`` (probe §6.2), each item launched at its
      own recorded site on the payload-zero manifold.

    ⛔ Depth alone is not a health signal (§7.26): raising ``write_margin``
    deepens every well *at a shared payload location*, which collapses the
    spread ``0.114 -> 0.054`` and makes the memory strictly worse. A run whose
    depth holds while its spread collapses is failing, and only the pair says so.
    """
    from chlu.core.blocks import CluStoreCell

    blk = model.blocks[layer]
    cell = blk.cell
    if not isinstance(cell, CluStoreCell):
        return {"applicable": False,
                "why": f"layer {layer} cell is {type(cell).__name__}, not the store"}
    t0 = time.time()
    scfg = pcfg.store_cfg()
    d, m = int(scfg.addr_dim), int(scfg.payload_dim)
    # ⭐ ONE lane. The probe's diagnostic replays lane 0 and the lanes are
    # independent (each builds its own controller over its own latents), so
    # planning the other 7 would be pure cost — at pilot geometry this is the
    # difference between an 8-lane plan pass per reading and a 1-lane one.
    tk = jnp.asarray(tokens, dtype=jnp.int32)[:1]
    plans, _ = plan_pass(model, tk, pcfg)
    h = _embed_stream(model, tk)
    for i in range(layer):
        h = _block_forward(model.blocks[i], h, plans[i])
    z = _block_chunk_latents(blk, h)

    pl0 = jax.tree_util.tree_map(lambda a: a[0], plans[layer])
    st = cell.init_state()
    for c in range(int(z.shape[1])):
        pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl0)
        st = _cell_write(cell, st, z[0, c], pc)

    live = np.asarray(pl0.live)[-1]
    sites = np.asarray(pl0.sites)[-1]
    idx = [i for i in range(int(scfg.capacity)) if live[i] > 0.5]
    depths, qs_pay, true_pay = [], [], []
    for i in idx:
        D, _s = cell_group_depth(cell, st, i, sites[i, :d])
        depths.append(float(D))
        qstar = np.asarray(cell.read_diag(st, jnp.asarray(sites[i],
                                                          dtype=jnp.float32))["q_star"])
        qs_pay.append(float(qstar[d]) if m else float("nan"))
        true_pay.append(float(sites[i, d]) if m else float("nan"))
    if not idx:
        return {"applicable": True, "n_live": 0, "wall_s": time.time() - t0,
                "why": "no live item to fit a well on"}
    dep = np.asarray(depths, dtype=float)
    qp = np.asarray(qs_pay, dtype=float)
    return {
        "applicable": True,
        "n_live": len(idx),
        "depth_median": float(np.median(dep)),
        "depth_min": float(np.min(dep)),
        "depth_max": float(np.max(dep)),
        "depth_per_item": [float(x) for x in dep],
        "qstar_payload": [float(x) for x in qp],
        "qstar_payload_spread": (float(np.ptp(qp)) if len(idx) > 1 else float("nan")),
        "payload_true": [float(x) for x in true_pay],
        "payload_true_spread": (float(np.ptp(np.asarray(true_pay, dtype=float)))
                                if len(idx) > 1 else float("nan")),
        "wall_s": time.time() - t0,
    }


def monitor_pass(model: StreamModel, pcfg: PilotConfig, tokens, *, layer: int = 0,
                 registry=None, write_loss_now: Optional[float] = None
                 ) -> Dict[str, Any]:
    """Run the **13-monitor registry** against the running stream.

    ⭐ ``full-clu-harness``'s acceptance criterion is inherited verbatim: the
    system runs the stream **without tripping a silent collapse mode**. "Does not
    collapse", not "wins". Every monitor's trip-state is a reported artifact, so
    the claim is checkable or it is not made.

    ⚠ **``inapplicable`` is reported as ``inapplicable``, never as a pass.** The
    monitors were designed against :class:`~chlu.core.clu_system.CluSystem`,
    whose ``observe`` supplies self-probe, certificate and codebook context. Here
    the store is a *layer of a language model*, so that context is reconstructed
    from the layer's own plan, its live atom state and a real self-probe pass;
    what genuinely cannot be reconstructed is declared, not silently passed.

    Pass ``registry`` to accumulate across calls — monitor #6 (objective
    divergence) needs a window of ``(write_loss, acq)`` pairs and is inapplicable
    on a single observation.
    """
    from chlu.core.blocks import CluStoreCell
    from chlu.core.monitors import saddle_reach_threshold

    blk = model.blocks[layer]
    cell = blk.cell
    if not isinstance(cell, CluStoreCell):
        return {"applicable": False,
                "why": f"layer {layer} cell is {type(cell).__name__}, not the store"}
    scfg = pcfg.store_cfg()
    d = int(scfg.addr_dim)
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    plans, pdiag = plan_pass(model, tk, pcfg)
    plan = plans[layer]
    h = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tk)
    h = h + model.pos[: h.shape[1]][None]
    for i in range(layer):
        h = jax.vmap(model.blocks[i])(h, plans[i])
    z = jax.vmap(blk.chunk_latents)(h)                       # (B, n_chunks, dim)

    # --- replay lane 0 concretely, collecting read diagnostics --------------
    pl0 = jax.tree_util.tree_map(lambda a: a[0], plan)
    st = cell.init_state()
    diags = []
    for c in range(int(z.shape[1])):
        pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl0)
        diags.append(cell.read_diag(st, z[0, c]))
        st = cell.write(st, z[0, c], pc)
    res = np.array([float(x["residual"]) for x in diags])
    g0 = np.array([float(x["grad0"]) for x in diags])
    rho = np.array([float(x["rho_conv"]) for x in diags])
    q_star = np.stack([np.asarray(x["q_star"]) for x in diags])
    q0 = np.asarray(z[0])[:, : int(scfg.dim)]

    live = np.asarray(plan.live[0])[-1]
    sites = np.asarray(plan.sites[0])[-1]
    live_idx = [i for i in range(int(scfg.capacity)) if live[i] > 0.5]
    centers = sites[live_idx][:, :d] if live_idx else np.zeros((0, d))
    sep = _min_sep(centers)

    # --- #1/#2: read diagnostics -------------------------------------------
    reads = {
        "grad_norm_q0": g0, "grad_norm_qstar": res,
        "displacement": np.linalg.norm(q_star[:, :d] - q0[:, :d], axis=-1),
        "rho_conv": rho, "residual": res,
        "corr_q0_qstar": _corr(q0[:, :d].ravel(), q_star[:, :d].ravel()),
    }
    if len(live_idx) >= 2:
        # settle basin vs the arg-min over the store's OWN keys — monitor #2's
        # `D <= U` legs, computed on the same queries the block actually ran.
        reads["assign_settle"] = _assign(q_star[:, :d], centers)
        reads["assign_argmin"] = _assign(q0[:, :d], centers)
        reads["covered"] = np.ones((q0.shape[0],), dtype=bool)

    # --- #5/#6/#9: the self-probe ------------------------------------------
    probe = store_self_probe(cell, st, sites, live, payload_tol=float(scfg.payload_tol))
    if write_loss_now is not None:
        probe["write_loss"] = float(write_loss_now)

    # --- #4: the BLANK control, on the same probe ---------------------------
    blank_probe = store_self_probe(cell, cell.init_state(), sites, live,
                                   payload_tol=float(scfg.payload_tol))
    n_p = max(int(blank_probe.get("n_probed", 0)), 1)
    chance = float(blank_probe.get("chance", float("nan")))
    blank_ctx = {"score": float(blank_probe.get("acq", float("nan"))),
                 "chance": chance,
                 "se": float(np.sqrt(max(chance, 1e-9) * (1 - chance) / n_p)),
                 "metric": "self-probe decode (nearest stored payload)",
                 "representation": "trajectory psi"}

    # --- #8: certificates ----------------------------------------------------
    certs = None
    if len(live_idx) >= 2:
        lam = _lambda_min_at(cell, st, q_star[:, :d])
        pays = np.array([sites[i, d] for i in live_idx])
        gap = float(np.min(np.abs(pays[:, None] - pays[None, :])
                           + np.eye(len(pays)) * 1e9)) if len(pays) > 1 else float("nan")
        certs = {
            "injective": bool(sep > 1e-6),
            "sep_over_sigma_q": float(sep / max(float(scfg.query_sigma), 1e-12)),
            "lambda_min": lam,
            "payload_gap": gap,
            "delta_read_basin_conditioned": float(np.median(
                np.linalg.norm(q_star[:, :d] - q0[:, :d], axis=-1))),
        }

    # --- #11: reach margins --------------------------------------------------
    margins, a_us = [], []
    for i in live_idx:
        D, s_i = cell_group_depth(cell, st, i, sites[i, :d])
        # ⚠ A group whose wells never got dug has D ~ 1e-86 and a depth-weighted
        # width that underflows; `saddle_reach_threshold` then divides by
        # 2*alpha*s^2 == 0.0. An item with no well is not "unreachable", it is
        # ABSENT — excluded from #11 and visible instead in #5/#9.
        if not (D > 1e-12) or not np.isfinite(s_i) or not (s_i > 1e-6):
            continue
        a_u = saddle_reach_threshold(D, s_i, float(scfg.confine),
                                     float(np.linalg.norm(sites[i, :d])))
        a_us.append(float(a_u))
        margins.append(float(a_u - abs(sites[i, d])))

    # --- #12/#3/M14 ----------------------------------------------------------
    lay = pdiag["layers"][layer]
    per_item_depth = np.array([cell_group_depth(cell, st, i, sites[i, :d])[0]
                               for i in live_idx]) if live_idx else np.array([np.nan])
    extras = {
        "sep": sep,
        "kinetic_mode": scfg.kinetic_mode,
        "write_steps": int(blk.mcfg.write_inner_steps),
        "wall_clock_s": 0.0,
        "utilisation": float(len(live_idx) / max(int(scfg.capacity), 1)),
        "fairness": float(_fairness(np.asarray(plan.admitted[0]),
                                    np.asarray(plan.slot[0]), int(scfg.capacity))),
        "c3_ratio": float("nan"),
        "c3_pairs": [],
        "oldest_retention_drop": float(
            1.0 - per_item_depth.min() / max(per_item_depth.max(), 1e-12))
        if np.isfinite(per_item_depth).all() and per_item_depth.size else float("nan"),
        "min_sep_minus_2s": (sep - 2.0 * float(np.exp(np.asarray(st.log_width)).max()))
        if np.isfinite(sep) else float("nan"),
        "reach_margins": margins, "a_U": a_us,
        "canary_guard_counts": lay.get("guards"),
        "knob_reads": None, "knobs_declared": None,
        "knob_tier_a_implemented": False,
    }
    if certs is not None:
        extras["certificates"] = certs

    reg = registry if registry is not None else default_registry(loud=False)
    ctx = MonitorContext(stage="stream", t=int(z.shape[1]), system=None,
                         reads=reads, self_probe=probe, blank=blank_ctx,
                         write_log=lay.get("rows", []),
                         controller=(lay.get("controllers") or [None])[0],
                         extras=extras)
    readings = reg.observe(ctx)
    return {
        "applicable": True,
        "readings": [r.as_dict() for r in readings],
        "n_monitors": len(readings),
        "n_applicable": int(sum(1 for r in readings if getattr(r, "applicable", True))),
        "n_tripped": int(sum(1 for r in readings if getattr(r, "tripped", False))),
        "tripped": [r.name for r in readings if getattr(r, "tripped", False)],
        "inapplicable": [r.name for r in readings
                         if not getattr(r, "applicable", True)],
        "self_probe": {k: v for k, v in probe.items() if k != "decoded"},
        "blank_probe_acq": blank_ctx["score"],
        "certificates": certs,
        "read_residual_median": float(np.median(res)),
        "read_rho_conv_median": float(np.median(rho)),
        "sep": sep,
        "plan": {k: v for k, v in lay.items() if k not in ("controllers", "rows")},
        "maturity_write_steps": int(blk.mcfg.write_inner_steps),
        "maturity_floor": 40,
        "maturity_trips_by_arithmetic": bool(blk.mcfg.write_inner_steps < 40),
    }


def _assign(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    return np.argmin(d, axis=1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _lambda_min_at(cell, state, points: np.ndarray) -> float:
    """Smallest Hessian eigenvalue of ``V_theta`` over the settled points."""
    V = cell._model(state).potential_net
    hess = jax.hessian(lambda q: V(q))
    vals = []
    d = int(cell.cfg.dim)
    for p in np.asarray(points)[:8]:
        q = np.zeros((d,), dtype=np.float32)
        q[: p.shape[0]] = p
        H = np.asarray(hess(jnp.asarray(q)))
        vals.append(float(np.min(np.linalg.eigvalsh(0.5 * (H + H.T)))))
    return float(np.min(vals)) if vals else float("nan")


def _min_sep(centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return float("nan")
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _fairness(admitted, slot, K) -> float:
    s = np.asarray(slot)[np.asarray(admitted) > 0.5]
    if s.size == 0:
        return 0.0
    c = np.bincount(s, minlength=K).astype(float)
    return float(c.min() / max(c.max(), 1e-12))


# ==========================================================================
# artifacts
# ==========================================================================
def save_json(path: str | Path, obj: Any, *, atomic: bool = False) -> Path:
    """Dump ``obj`` as JSON. ``atomic=True`` writes a sibling tmp + ``os.replace``.

    ⚠ The atomic path exists for the crash-resume **journal**: it is rewritten
    after every phase of a 30 h job, so a kill landing mid-``write_text`` would
    otherwise leave a truncated file that is worse than no file at all.
    ``os.replace`` is atomic within a filesystem, which the tmp sibling
    guarantees.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, indent=2, default=_jsonable)
    if not atomic:
        p.write_text(blob)
        return p
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(blob)
    os.replace(tmp, p)
    return p


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (jnp.ndarray,)):
        return np.asarray(x).tolist()
    if hasattr(x, "__dataclass_fields__"):
        return asdict(x)
    return str(x)


__all__ = [
    "PilotConfig", "ArmSpec", "solve_arms", "build_arm", "shell_of",
    "assert_shared_shell_identical", "plan_pass", "loss_fn", "train_arm",
    "evaluate", "dynamic_eval", "gradient_probe", "allocation_liveness",
    "monitor_pass", "make_optimizer", "save_json", "anytime_curve",
    "calibrate_phi_gain", "store_health_probe", "LaneControllerSummary",
    "shutdown_lane_pools", "host_rss", "release_host_memory",
]
_ = Sequence
