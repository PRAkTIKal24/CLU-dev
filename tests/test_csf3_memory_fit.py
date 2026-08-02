"""`csf3-memory-fit` — the backward-memory levers, as tests.

CSF3 run 1 died at compile time: XLA wanted **97.82 GiB** for the backward of
``loss_fn`` against an 80 GB A100, and its own auto-remat floored at 76.70 GiB.
There was no explicit ``jax.checkpoint`` anywhere in ``blocks.py`` /
``train_cluformer.py``, so the backward through ``lax.scan(step, ...)`` kept
every within-chunk intermediate — the two-phase Verlet unroll over ``n_atoms``
atoms *and* the differentiated inner writes — alive for all ``n_chunks`` x
``n_layers`` x ``batch``.

This file is the control the fix has to pass: **the levers change the memory
schedule and NOTHING else.** Concretely —

* every new knob ships OFF, so no pre-existing number moves;
* the read is bit-identical under segmented rollout remat;
* the block forward, the held-out bpc and **all 7 ``WritePlan`` fields** are
  bit-identical with ``remat_chunks`` on;
* the *gradient* moves only by declared float32 round-off (``||dg||/||g||`` =
  8.67e-10 measured, gate 1e-8, one float32 ULP = 1.19e-7) — a remat that moved
  it further would be a silent training change, not a memory fix;
* gradient accumulation reproduces the full-batch gradient to float tolerance,
  and is honest about being the ONE lever that is not bitwise (mean-of-means
  re-associates the float sum — declared, not hidden).
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.blocks import (
    CluStoreCell,
    StreamMemoryConfig,
    StreamModel,
    make_memory_cell,
)
from chlu.core.clu_system import CluSystemConfig
from chlu.training.train_cluformer import (
    PilotConfig,
    _accum_grads,
    allocation_liveness,
    gradient_probe,
    loss_fn,
    plan_pass,
)


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.2 / §7.23, N211).

    ⚠ Module scope is load-bearing and must be paired with the restore below:
    other modules enable ``jax_enable_x64`` at import time and a function-scoped
    fixture would run *after* them.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# a tiny but STRUCTURALLY COMPLETE rig: the real store cell, the real block,
# the real controller in the plan pass.
# ---------------------------------------------------------------------------
def _pcfg(**kw):
    base = dict(d_model=16, n_layers=2, seq_len=32, batch=2, vocab_size=16,
                addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=16,
                steps=2, warmup=1, seed=0,
                memory=dict(chunk=8, address_steps=6, read_steps=6, traj_stride=3,
                            psi_hidden=8, write_inner_steps=1, write_n_perturb=4,
                            retry_rounds=1, conv_kernel=3, mlp_mult=2),
                store=dict(min_atoms=64, min_atoms_base=32, budget=3))
    mem = dict(base["memory"])
    mem.update(kw.pop("memory", {}))
    base["memory"] = mem
    base.update(kw)
    return PilotConfig.from_mapping(base)


def _scfg(pcfg):
    return pcfg.store_cfg()


def _model(pcfg, *, key=None) -> StreamModel:
    key = jax.random.PRNGKey(0) if key is None else key
    scfg, mcfg = pcfg.store_cfg(), pcfg.memory_cfg()
    cells = [make_memory_cell("clu_store", latent_dim=int(scfg.dim), clu_cfg=scfg,
                              mcfg=mcfg, key=jax.random.PRNGKey(11))
             for _ in range(pcfg.n_layers)]
    return StreamModel(vocab_size=pcfg.vocab_size, d_model=pcfg.d_model,
                       n_layers=pcfg.n_layers, max_len=pcfg.seq_len, cells=cells,
                       mcfg=mcfg, latent_dim=int(scfg.dim),
                       addr_dim=int(scfg.addr_dim),
                       payload_dim=int(scfg.payload_dim), key=key)


def _tokens(pcfg, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, pcfg.vocab_size, size=(pcfg.batch, pcfg.seq_len))
    y = rng.integers(0, pcfg.vocab_size, size=(pcfg.batch, pcfg.seq_len))
    return jnp.asarray(x, jnp.int32), jnp.asarray(y, jnp.int32)


def _leaves(tree):
    return [np.asarray(x) for x in
            jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))]


# ---------------------------------------------------------------------------
# 1. every lever ships OFF
# ---------------------------------------------------------------------------
def test_every_memory_lever_defaults_to_off():
    """⛔ The regression gate: toy history and every pre-CSF3 artifact must be
    untouched by this branch, so the OLD behaviour has to be the default."""
    m = StreamMemoryConfig()
    assert m.remat_chunks is False
    assert m.remat_read_segments == 0
    p = PilotConfig()
    assert p.accum_steps == 1
    assert p.liveness_lanes == 0
    assert p.probe_lanes == 0


def test_stream_memory_config_stays_hashable_with_the_remat_levers():
    """``StreamMemoryConfig`` is an equinox STATIC field — an unhashable knob
    would break every ``jit`` cache."""
    a = StreamMemoryConfig(remat_chunks=True, remat_read_segments=2)
    assert isinstance(hash(a), int)
    assert a == StreamMemoryConfig(remat_chunks=True, remat_read_segments=2)
    assert a != StreamMemoryConfig()


def test_the_new_flags_show_up_in_the_flag_table():
    """Flag-provenance rule: a lever that does not print is a lever that is not
    reported."""
    assert StreamMemoryConfig(remat_chunks=True).as_flag_table() == {
        "remat_chunks": True}
    assert PilotConfig(accum_steps=2, liveness_lanes=1).as_flag_table() == {
        "accum_steps": 2, "liveness_lanes": 1}


# ---------------------------------------------------------------------------
# 2. rung 2 — the segmented read is the SAME read
# ---------------------------------------------------------------------------
def _cell(pcfg, **mem):
    """The same cell (same key ⇒ bit-identical parameters), different mcfg."""
    return CluStoreCell(_scfg(pcfg), dataclasses.replace(pcfg.memory_cfg(), **mem),
                        key=jax.random.PRNGKey(3))


@pytest.mark.parametrize("n_seg", [2, 3, 6])
def test_the_segmented_verlet_read_is_bit_identical(n_seg):
    """⭐ The segments chain ``(q, p)`` unchanged and re-assemble the strided
    buffer with the global stride phase, so ``psi`` is handed the same array."""
    pcfg = _pcfg()
    base, seg = _cell(pcfg), _cell(pcfg, remat_read_segments=n_seg)
    for x, y in zip(_leaves(base), _leaves(seg), strict=True):
        assert np.array_equal(x, y), "the two cells are not the same parameters"
    st = base.init_state()
    z = jnp.asarray(np.random.default_rng(0).normal(size=(int(_scfg(pcfg).dim),)),
                    jnp.float32)
    a = np.asarray(base.read(st, z))
    b = np.asarray(seg.read(st, z))
    assert np.array_equal(a, b), f"segmented read differs by {np.abs(a - b).max()}"


def test_a_non_dividing_segment_count_falls_back_to_the_shipped_call():
    """Never a silently different trajectory: if the phase's step count is not
    divisible by ``remat_read_segments`` the shipped single call is used."""
    pcfg = _pcfg()          # address_steps = read_steps = 6, so 4 does not divide
    base, bad = _cell(pcfg), _cell(pcfg, remat_read_segments=4)
    st = base.init_state()
    z = jnp.zeros((int(_scfg(pcfg).dim),), jnp.float32) + 0.3
    assert np.array_equal(np.asarray(base.read(st, z)), np.asarray(bad.read(st, z)))


# ---------------------------------------------------------------------------
# 3. rung 1 — the chunk-scan remat: the jit-tripwire control
# ---------------------------------------------------------------------------
def _with(pcfg, **mem):
    return _pcfg(memory={**pcfg.memory, **mem})


def _pair(**mem):
    """``(model_off, model_on)`` — same keys, so the PARAMETERS are bit-identical
    and the only difference between them is the (static) remat schedule."""
    off = _pcfg()
    on = _with(off, **mem)
    m0, m1 = _model(off), _model(on)
    for x, y in zip(_leaves(m0), _leaves(m1), strict=True):
        assert np.array_equal(x, y), "the remat flag moved a parameter"
    return off, m0, m1


def test_the_block_forward_and_the_bpc_are_bit_identical_under_chunk_remat():
    """⭐ The acceptance control: held-out nll (=> bpc) identical, remat on vs off."""
    pcfg, m0, m1 = _pair(remat_chunks=True)
    tk, tg = _tokens(pcfg)
    plans, _ = plan_pass(m0, tk, pcfg)
    l0 = float(loss_fn(m0, tk, tg, plans))
    l1 = float(loss_fn(m1, tk, tg, plans))
    assert l0 == l1, f"nll moved by {abs(l0 - l1):.3e} under remat"


def test_all_seven_write_plan_fields_are_bit_identical_under_chunk_remat():
    """⭐ The controller sees the same latents ⇒ takes the same decisions. The
    plan pass runs the block forward, so remat could in principle move it."""
    pcfg, m0, m1 = _pair(remat_chunks=True, remat_read_segments=2)
    tk, _ = _tokens(pcfg)
    p0, _ = plan_pass(m0, tk, pcfg)
    p1, _ = plan_pass(m1, tk, pcfg)
    assert len(p0) == len(p1) == pcfg.n_layers
    for layer, (a, b) in enumerate(zip(p0, p1, strict=True)):
        for field in ("slot", "admitted", "group_scale", "reset", "sites",
                      "live", "retry"):
            assert np.array_equal(np.asarray(getattr(a, field)),
                                  np.asarray(getattr(b, field))), \
                f"layer {layer} plan field {field!r} moved under remat"


#: ⭐ **The declared tolerance of the remat control** (task dial declaration:
#: "≤ float-ULP tolerance declared if XLA reassociates"). The FORWARD is bitwise
#: — asserted, not toleranced. The BACKWARD is not: a rematted VJP is
#: recomputed, so XLA fuses and re-associates its float32 sums differently.
#: Measured on this rig: ``||dg|| / ||g||`` = **8.67e-10** for ``remat_chunks``,
#: **3.18e-10** for ``remat_read_segments=3``, i.e. ~7e-3 of ONE float32 ULP
#: (eps = 1.19e-7). The gate is set an order above the measurement and two
#: orders below the ULP, so it catches a real change and passes round-off.
GRAD_RELL2_TOL = 1e-8


@pytest.mark.parametrize("mem", [
    dict(remat_chunks=True),
    dict(remat_read_segments=3),
    dict(remat_chunks=True, remat_read_segments=3),
])
def test_the_gradient_moves_only_by_float32_round_off_under_remat(mem):
    """⛔ The one that matters most. A remat that moves the gradient is a
    training change wearing a memory fix's clothes.

    Two separate assertions, deliberately: the **loss is bitwise** (the forward
    is literally the same graph), and the **gradient is within a declared
    round-off band** (the backward is recomputed and re-associated)."""
    pcfg, m0, m1 = _pair(**mem)
    tk, tg = _tokens(pcfg)
    plans, _ = plan_pass(m0, tk, pcfg)
    l0, g0 = eqx.filter_value_and_grad(loss_fn)(m0, tk, tg, plans)
    l1, g1 = eqx.filter_value_and_grad(loss_fn)(m1, tk, tg, plans)
    assert float(l0) == float(l1), "the forward is not bitwise under remat"
    a, b = _leaves(g0), _leaves(g1)
    assert len(a) == len(b)
    num = sum(float(((x.astype(np.float64) - y.astype(np.float64)) ** 2).sum())
              for x, y in zip(a, b, strict=True))
    den = sum(float((x.astype(np.float64) ** 2).sum()) for x in a)
    rel = float(np.sqrt(num / den))
    assert rel <= GRAD_RELL2_TOL, f"||dg||/||g|| = {rel:.3e} > {GRAD_RELL2_TOL:.0e}"


@pytest.mark.parametrize("arm", ["gru_matched", "ttt_matched", "none", "echo"])
def test_every_swap_arm_survives_chunk_remat_bitwise(arm):
    """⭐ ``remat_chunks`` wraps the SHARED ``StreamBlock`` scan, so it applies to
    every arm of the system-level swap, not just the store — including the two
    controls whose cells have a zero-size / constant state. The swap is only a
    swap if the flag is inert for all of them."""
    pcfg = _pcfg()
    scfg = pcfg.store_cfg()
    out = []
    for mem in ({}, {"remat_chunks": True}):
        mcfg = dataclasses.replace(pcfg.memory_cfg(), **mem)
        cells = [make_memory_cell(arm, latent_dim=int(scfg.dim), clu_cfg=scfg,
                                  mcfg=mcfg, hidden=6, ttt_shape=(3, 4),
                                  key=jax.random.PRNGKey(11))
                 for _ in range(pcfg.n_layers)]
        m = StreamModel(vocab_size=pcfg.vocab_size, d_model=pcfg.d_model,
                        n_layers=pcfg.n_layers, max_len=pcfg.seq_len, cells=cells,
                        mcfg=mcfg, latent_dim=int(scfg.dim),
                        addr_dim=int(scfg.addr_dim),
                        payload_dim=int(scfg.payload_dim),
                        key=jax.random.PRNGKey(0))
        tk, tg = _tokens(pcfg)
        plans, _ = plan_pass(m, tk, pcfg)
        out.append(float(loss_fn(m, tk, tg, plans)))
    assert out[0] == out[1], f"{arm}: nll moved by {abs(out[0] - out[1]):.3e}"


# ---------------------------------------------------------------------------
# 4. the microbatch lever — exact in effective batch, float-approximate in sum
# ---------------------------------------------------------------------------
def test_grad_accumulation_reproduces_the_full_batch_gradient():
    """⚠ NOT bitwise, and deliberately so: ``mean(mean_i)`` re-associates the
    float sum. The effective batch is preserved EXACTLY (both see all 4 lanes);
    the arithmetic agrees to float32 tolerance."""
    pcfg = _pcfg(batch=4, accum_steps=2)
    m = _model(pcfg)
    tk, tg = _tokens(pcfg)
    plans, _ = plan_pass(m, tk, pcfg)
    l_full, g_full = eqx.filter_value_and_grad(loss_fn)(m, tk, tg, plans)
    l_acc, g_acc = _accum_grads(m, tk, tg, plans, 2)
    assert float(l_full) == pytest.approx(float(l_acc), rel=1e-5, abs=1e-6)
    for x, y in zip(_leaves(g_full), _leaves(g_acc), strict=True):
        if not x.size:
            continue
        scale = max(float(np.abs(x).max()), 1e-8)
        assert float(np.abs(x - y).max()) <= 3e-5 * scale + 1e-8


def test_dynamic_eval_inherits_accum_steps_without_moving_its_number():
    """⛔ dyn-eval takes the SAME backward as a training step, so it must inherit
    the lever — the job header's cut-order forbids cutting this column, and a
    run that needed the microbatch to fit would otherwise OOM here."""
    from chlu.training.train_cluformer import dynamic_eval

    pcfg = _pcfg(batch=4)
    m = _model(pcfg)
    bat = [_tokens(pcfg, seed=s) for s in (5, 6)]
    a = dynamic_eval(m, pcfg, list(bat), lrs=[1e-4])
    b = dynamic_eval(m, _pcfg(batch=4, accum_steps=2), list(bat), lrs=[1e-4])
    assert a["bpc"] == pytest.approx(b["bpc"], rel=1e-5)


def test_accum_steps_must_divide_the_batch():
    from chlu.training.train_cluformer import train_arm

    pcfg = _pcfg(batch=2, accum_steps=3)
    with pytest.raises(ValueError, match="does not divide batch"):
        train_arm("clu_store", _model(pcfg), pcfg, iter([]))


# ---------------------------------------------------------------------------
# 5. the instruments shrink, and SAY they shrank
# ---------------------------------------------------------------------------
def test_liveness_lanes_cuts_the_probe_batch_and_reports_it():
    pcfg = _pcfg(batch=2, liveness_lanes=1)
    m = _model(pcfg)
    tk, tg = _tokens(pcfg)
    out = allocation_liveness(m, pcfg, tk, tg)
    assert out["n_lanes"] == 1
    assert len(out["slot_entropy_normalised_per_layer"]) == pcfg.n_layers
    full = allocation_liveness(m, _pcfg(batch=2), tk, tg)
    assert full["n_lanes"] == 2


def test_probe_lanes_defaults_to_the_whole_batch():
    """Default OFF: this one moves a PUBLISHED magnitude."""
    pcfg = _pcfg(batch=2)
    m = _model(pcfg)
    tk, tg = _tokens(pcfg)
    out = gradient_probe(m, pcfg, tk, tg)
    assert out["n_lanes"] == 2


# ---------------------------------------------------------------------------
# 6. the config plumbing the resubmission line depends on
# ---------------------------------------------------------------------------
def test_the_submission_flags_survive_the_cli_override_path():
    """``--mem remat_chunks=true --set accum_steps=2 liveness_lanes=1`` is the
    resubmission line; it must reach the objects that read it."""
    from chlu.experiments.exp_cluformer_pilot import _parse_kv, make_config

    parsed = dict(_parse_kv(p) for p in
                  ["remat_chunks=true", "remat_read_segments=2"])
    cfg = make_config("pilot", 0, {"memory": {**{}, **parsed},
                                   "accum_steps": 2, "liveness_lanes": 1})
    assert cfg.memory_cfg().remat_chunks is True
    assert cfg.memory_cfg().remat_read_segments == 2
    assert cfg.accum_steps == 2 and cfg.liveness_lanes == 1


def test_the_store_config_of_the_pilot_is_untouched_by_this_branch():
    """§A20.4: run 2 must be same-config-otherwise. The memory levers live in
    ``StreamMemoryConfig`` / ``PilotConfig`` and touch no store semantics."""
    from chlu.experiments.exp_cluformer_pilot import make_config

    a = make_config("pilot", 0).store_cfg()
    b = make_config("pilot", 0, {"memory": {"remat_chunks": True},
                                 "accum_steps": 2}).store_cfg()
    assert isinstance(a, CluSystemConfig)
    assert a == b
