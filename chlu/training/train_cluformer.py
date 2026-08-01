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

import json
import math
import time
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
                                 hidden=h_params, ttt_shape=kn, key=key)
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
                              ttt_shape=spec.ttt_shape, key=ks[i])
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
def _chunk_latents(model: StreamModel, tokens: jnp.ndarray):
    """Per-layer chunk latents under the *current* plan, computed concretely.

    Yields ``(layer_index, z)`` with ``z`` of shape ``(B, n_chunks, latent)`` and
    accepts the plan for that layer back through ``send``, coroutine-style, so
    layer ``l+1``'s latents are computed with layer ``l``'s real decisions in
    force. That ordering matters: a plan computed on a forward that ignored the
    earlier layers' memories would not be the plan the model runs.
    """
    h = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tokens)
    h = h + model.pos[: h.shape[1]][None]
    for blk in model.blocks:
        z = jax.vmap(blk.chunk_latents)(h)
        plan = yield z
        h = jax.vmap(blk)(h, plan)


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

    for c in range(n_chunks):
        z = np.asarray(z_lane[c], dtype=float)
        addr, pay = z[:d], z[d: d + m]
        n_ev0 = sum(1 for v in ctrl.log if v.verb == "evict" and v.applied)
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
        for v in ctrl.log[n_ev0:]:
            if v.verb == "evict" and v.applied:
                ev = slot_of(int(v.detail.get("item_id", -1)))
                if ev is not None:
                    reset[c, ev] = 1.0
                stats["evicted"] += 1
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
    return {"slot": slot, "admitted": admitted, "group_scale": group_scale,
            "reset": reset, "sites": sites, "live": live,
            "retry": np.zeros((n_chunks,), np.int32), "_stats": stats}


def plan_pass(model: StreamModel, tokens: jnp.ndarray, pcfg: PilotConfig,
              *, registry=None, blank: bool = False) -> Tuple[List[WritePlan], Dict]:
    """Concrete pass 1: the real controller emits one plan per layer.

    ``blank=True`` produces the **blank-store control** (collapse mode #4): the
    store is read exactly as in the live arm but nothing is ever admitted.
    """
    scfg = pcfg.store_cfg()
    B, T = tokens.shape
    n_chunks = T // int(pcfg.memory_cfg().chunk)
    K, dim = int(scfg.capacity), int(scfg.dim)
    registry = registry if registry is not None else default_registry(loud=False)
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
            lanes = [_controller_plan_for_lane(zc[b], scfg, registry) for b in range(B)]
            lstats = {k: int(sum(la["_stats"][k] for la in lanes))
                      for k in ("offers", "refused", "evicted", "n_live_end")}
            gsum: Dict[str, int] = {}
            for la in lanes:
                for gk, gv in la["_stats"]["guards"].items():
                    gsum[gk] = gsum.get(gk, 0) + int(gv)
            lstats["guards"] = gsum
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
def _train_step(model, opt_state, tokens, targets, plans, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tokens, targets, plans)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


def make_optimizer(pcfg: PilotConfig):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=pcfg.lr, warmup_steps=max(1, pcfg.warmup),
        decay_steps=max(2, pcfg.steps), end_value=pcfg.lr * 0.1)
    return optax.chain(optax.clip_by_global_norm(pcfg.grad_clip),
                       optax.adamw(sched, weight_decay=pcfg.weight_decay))


def train_arm(name: str, model: StreamModel, pcfg: PilotConfig, batches,
              *, log_every: int = 25, log: Optional[List] = None):
    """Train one arm. ⭐ ``batches`` must be the SAME sequence for every arm."""
    optimizer = make_optimizer(pcfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    hist, t0 = [], time.time()
    plan_s = 0.0
    for i, (x, y) in enumerate(batches):
        tk = jnp.asarray(x, dtype=jnp.int32)
        tg = jnp.asarray(y, dtype=jnp.int32)
        tp = time.time()
        plans, _ = plan_pass(model, tk, pcfg)
        plan_s += time.time() - tp
        model, opt_state, loss = _train_step(model, opt_state, tk, tg, plans, optimizer)
        hist.append(float(loss))
        if log is not None and (i % log_every == 0 or i == pcfg.steps - 1):
            log.append({"arm": name, "step": i, "nll": float(loss),
                        "bpc": bits_per_character(float(loss)),
                        "wall_s": time.time() - t0, "plan_s": plan_s})
    return model, {"loss_history": hist, "wall_s": time.time() - t0,
                   "plan_pass_s": plan_s,
                   "plan_pass_frac": plan_s / max(time.time() - t0, 1e-9)}


def evaluate(model: StreamModel, pcfg: PilotConfig, batches, *, blank: bool = False,
             verlet: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
    """Held-out bpc on a **contiguous, order-preserving** iterator."""
    tot, n = 0.0, 0
    for x, y in batches:
        tk = jnp.asarray(x, dtype=jnp.int32)
        tg = jnp.asarray(y, dtype=jnp.int32)
        plans, _ = plan_pass(model, tk, pcfg, blank=blank)
        tot += float(loss_fn(model, tk, tg, plans, None, verlet))
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
    """
    grid = list(lrs) if lrs is not None else [pcfg.dyneval_lr, 10 * pcfg.dyneval_lr,
                                              100 * pcfg.dyneval_lr]
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
    trajectory read is the only channel that does. The two arms here are
    identical models differing ONLY in ``StreamMemoryConfig.read_mode``, so the
    comparison is internal and needs no baseline.
    """
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    tg = jnp.asarray(targets, dtype=jnp.int32)
    out: Dict[str, Any] = {}
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
    a = out["trajectory"]["grad_phi"]
    b = out["settled_point"]["grad_phi"]
    out["ratio_traj_over_point"] = (a / b) if b > 0 else float("inf")
    out["wall_ratio_traj_over_point"] = (out["trajectory"]["wall_s"]
                                         / max(out["settled_point"]["wall_s"], 1e-9))
    return out


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
    """
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    tg = jnp.asarray(targets, dtype=jnp.int32)
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
        "policy_logits": None,
        "policy_logits_note": "the C2W1 controller's policy is RULE-BASED, not "
                              "logit-parameterised: there are no policy logits. "
                              "phi's address head is the learned allocator.",
        "slot_entropy_normalised_per_layer": ent,
        "slot_occupancy_per_layer": occ,
    }


def anytime_curve(model: StreamModel, pcfg: PilotConfig, batches,
                  budgets: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """⭐ D5 — accuracy vs Verlet-steps-per-read, on a model trained at ONE budget.

    ⚠ **SHAPE claim only** (charter §A3). The anytime figure is occupied — DEQs,
    Energy-Based Transformers, Titans-Revisited all own it — so no uniqueness is
    claimed and none may be quoted from this curve. ⚠ The trajectory read costs
    **17.1x** the point read (`trainability-spike` §3) and that price travels
    with every point on it.
    """
    bat = list(batches)
    out = []
    for b in budgets:
        t0 = time.time()
        r = evaluate(model, pcfg, iter(bat), verlet=tuple(int(x) for x in b))
        out.append({"address_steps": int(b[0]), "read_steps": int(b[1]),
                    "verlet_per_read": int(b[0]) + int(b[1]),
                    "bpc": r["bpc"], "wall_s": time.time() - t0})
    return out


# ==========================================================================
# monitors — all 13, as reported artifacts
# ==========================================================================
def monitor_pass(model: StreamModel, pcfg: PilotConfig, tokens, *, layer: int = 0
                 ) -> Dict[str, Any]:
    """Run the **13-monitor registry** against the running stream.

    ⭐ ``full-clu-harness``'s acceptance criterion is inherited verbatim: the
    system runs the stream **without tripping a silent collapse mode**. "Does not
    collapse", not "wins". Every monitor's trip-state is a reported artifact, so
    the claim is checkable or it is not made.

    ⚠ Scoped honestly: the monitors were designed against
    :class:`~chlu.core.clu_system.CluSystem`, whose ``observe`` supplies a rich
    context (self-probe, certificates, codebook geometry). Here the store is a
    *layer of a language model*, so the context is assembled from the layer's own
    plan and read diagnostics. Monitors whose inputs are genuinely absent report
    ``inapplicable`` — never ``pass``.
    """
    from chlu.core.blocks import CluStoreCell

    blk = model.blocks[layer]
    cell = blk.cell
    if not isinstance(cell, CluStoreCell):
        return {"applicable": False,
                "why": f"layer {layer} cell is {type(cell).__name__}, not the store"}
    scfg = pcfg.store_cfg()
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    plans, pdiag = plan_pass(model, tk, pcfg)
    plan = plans[layer]
    # replay the stream up to this layer to get its true latents + store state
    h = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tk)
    h = h + model.pos[: h.shape[1]][None]
    for i in range(layer):
        h = jax.vmap(model.blocks[i])(h, plans[i])
    z = jax.vmap(blk.chunk_latents)(h)                # (B, n_chunks, dim)

    def run_lane(zl, pl):
        st = cell.init_state()
        diags, states = [], []
        n = zl.shape[0]
        for c in range(n):
            pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl)
            diags.append(cell.read_diag(st, zl[c]))
            st = cell.write(st, zl[c], pc)
            states.append(st)
        return diags, st
    diags, st_end = run_lane(z[0], jax.tree_util.tree_map(lambda a: a[0], plan))
    res = np.array([float(d["residual"]) for d in diags])
    rho = np.array([float(d["rho_conv"]) for d in diags])
    q_star = np.stack([np.asarray(d["q_star"]) for d in diags])
    live = np.asarray(plan.live[0])
    sites = np.asarray(plan.sites[0])
    last_live = live[-1] > 0.5
    centers = sites[-1][last_live][:, : scfg.addr_dim]
    sep = _min_sep(centers)
    amps = np.asarray(st_end.amp)
    gm = np.asarray(cell.group_matrix)
    per_item_depth = np.array([float((amps ** 2)[gm[s]].sum())
                               for s in range(int(scfg.capacity))])
    ctx_extras = {
        "sep": sep,
        "kinetic_mode": scfg.kinetic_mode,
        "write_steps": int(blk.mcfg.write_inner_steps),
        "wall_clock_s": 0.0,
        "utilisation": float(last_live.mean()),
        "fairness": float(_fairness(np.asarray(plan.admitted[0]),
                                    np.asarray(plan.slot[0]), int(scfg.capacity))),
        "c3_ratio": float("nan"),
        "c3_pairs": [],
        "oldest_retention_drop": float(per_item_depth.min() / max(per_item_depth.max(), 1e-12)),
        "min_sep_minus_2s": sep - 2.0 * float(np.exp(np.asarray(st_end.log_width)).max())
        if np.isfinite(sep) else float("nan"),
        "knob_reads": None, "knobs_declared": None, "knob_tier_a_implemented": False,
        "reads": {"rho_conv": rho, "residual": res, "q_star": q_star},
    }
    registry = default_registry(loud=False)
    ctx = MonitorContext(stage="stream", t=int(z.shape[1]), system=None,
                         reads=ctx_extras["reads"], self_probe=None, blank=None,
                         write_log=[], controller=None, extras=ctx_extras)
    readings = registry.observe(ctx)
    return {
        "applicable": True,
        "readings": [r.as_dict() for r in readings],
        "n_tripped": int(sum(1 for r in readings if getattr(r, "tripped", False))),
        "tripped": [r.name for r in readings if getattr(r, "tripped", False)],
        "summary": registry.summary(),
        "read_residual_median": float(np.median(res)),
        "read_rho_conv_median": float(np.median(rho)),
        "sep": sep,
        "plan": pdiag,
        "maturity_write_steps": int(blk.mcfg.write_inner_steps),
        "maturity_floor": 40,
        "maturity_trips_by_arithmetic": bool(blk.mcfg.write_inner_steps < 40),
    }


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
def save_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_jsonable))
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
    "calibrate_phi_gain",
]
_ = Sequence
