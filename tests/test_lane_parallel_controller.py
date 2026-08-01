"""The lane-parallel plan pass: it is a SCHEDULING change or it is nothing.

⭐ **Decision-replay is the SPEC.** ``plan_pass`` runs the real, discrete C2W1
controller outside the differentiable forward and the forward *replays* its
:class:`WritePlan`; a wall-clock change to how the lanes are scheduled is
therefore only admissible if the plan is **identical**. These are the tests that
fail if the ``ProcessPoolExecutor`` path ever stops being a pure scheduling
change — bit-identical discrete fields across seeds, identical monitor
summaries, and an explicit check that snapshotting the registry's class-I trip
list reproduces what the live registry decides.

Companion tripwire: ``test_placement_probe.py::
test_jitting_the_plan_pass_changes_no_controller_decision`` (the jit).
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.monitors import MonitorReading, default_registry
from chlu.training.train_cluformer import (
    LaneControllerSummary,
    PilotConfig,
    _ClassITrips,
    _controller_plan_for_lane,
    build_arm,
    monitor_pass,
    plan_pass,
    shutdown_lane_pools,
    solve_arms,
)

#: 2 workers, not 8: the point is the boundary crossing, and every extra worker
#: is another JAX import in a spawned process.
WORKERS = 2
PLAN_FIELDS = ("slot", "admitted", "group_scale", "reset", "sites", "live", "retry")


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.2 / §7.23).

    ⚠ Module scope is load-bearing: other test modules flip x64 on at import, and
    a function-scoped fixture is set up *after* the module-scoped ones, so the
    store cell would be built in float64 and exercised in float32.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def pcfg():
    return PilotConfig(
        d_model=16, n_layers=2, seq_len=32, batch=3, vocab_size=32,
        addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=32,
        steps=2, warmup=1, eval_batches=1, dyneval_batches=1,
        store=dict(min_atoms=64, min_atoms_base=32),
        memory=dict(chunk=8, address_steps=6, read_steps=6, traj_stride=3,
                    psi_hidden=8, write_inner_steps=1, write_n_perturb=4,
                    retry_rounds=1, conv_kernel=3, mlp_mult=2),
    )


@pytest.fixture(scope="module")
def pooled(pcfg):
    """The same config with the lane pool on. Torn down at module exit."""
    import dataclasses

    yield dataclasses.replace(pcfg, plan_workers=WORKERS)
    shutdown_lane_pools()


def _model(pcfg, seed: int):
    specs, _ = solve_arms(pcfg, jax.random.PRNGKey(seed))
    return build_arm("clu_store", pcfg, specs, key=jax.random.PRNGKey(100 + seed))


def _tokens(pcfg, seed: int):
    return jnp.asarray(np.random.default_rng(seed).integers(
        0, pcfg.vocab_size, (pcfg.batch, pcfg.seq_len)), jnp.int32)


# ---------------------------------------------------------------------------
# the blocking equivalence test — the plan is the spec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lane_parallel_plan_is_bit_identical_to_serial(pcfg, pooled, seed):
    """⭐⭐ Every field of every layer's plan, bit-identical, on 3 seeds.

    Not ``allclose``: the lanes run the *same* numpy/Equinox code, only in
    another process, so anything but exact equality is a bug (unlike the jit
    tripwire, where XLA fusion legitimately moves ``sites`` by a float32 ULP).
    """
    model = _model(pcfg, seed)
    tk = _tokens(pcfg, seed)
    ser, dser = plan_pass(model, tk, pcfg)
    par, dpar = plan_pass(model, tk, pooled)
    assert len(ser) == len(par) == pcfg.n_layers
    for lay, (a, b) in enumerate(zip(ser, par, strict=True)):
        for f in PLAN_FIELDS:
            assert np.array_equal(np.asarray(getattr(a, f)),
                                  np.asarray(getattr(b, f))), f"layer {lay} field {f}"
    # ...and the pool was actually used (a silent fallback would pass the above)
    assert dpar["layers"][0]["lane_mode"].startswith("pool[")
    assert dser["layers"][0]["lane_mode"] == "serial"


def test_lane_parallel_monitor_summaries_are_unchanged(pcfg, pooled):
    """⭐ Same guard counts, same offers/refusals/evictions, same verb rows.

    The monitors read the plan diagnostics, not the controller object, and M14
    trips on ``canary_guard_counts``: if the merge of the per-lane summaries lost
    a guard fire, M14's verdict would move. It must not.
    """
    model, tk = _model(pcfg, 0), _tokens(pcfg, 0)
    _, dser = plan_pass(model, tk, pcfg)
    _, dpar = plan_pass(model, tk, pooled)
    for lser, lpar in zip(dser["layers"], dpar["layers"], strict=True):
        for k in ("offers", "refused", "evicted", "n_live_end"):
            assert lser[k] == lpar[k], k
        assert lser["guards"] == lpar["guards"]
        assert len(lser["rows"]) == len(lpar["rows"])
        for r_s, r_p in zip(lser["rows"], lpar["rows"], strict=True):
            assert r_s["decision"] == r_p["decision"]
            assert r_s["applied"] == r_p["applied"]
            assert r_s["guard"] == r_p["guard"]
        cs, cp = lser["controllers"], lpar["controllers"]
        assert [c.guard_fire_counts() for c in cs] == [c.guard_fire_counts() for c in cp]
        assert [c.n_live for c in cs] == [c.n_live for c in cp]
        assert [c.records for c in cs] == [c.records for c in cp]
        assert [c.log for c in cs] == [c.log for c in cp]


def test_monitor_trip_states_are_identical_under_the_pool(pcfg, pooled):
    """⭐ End-to-end: the 13 monitors + M14 give the same readings either way."""
    model, tk = _model(pcfg, 0), _tokens(pcfg, 0)
    a = monitor_pass(model, pcfg, tk, registry=default_registry(loud=False))
    b = monitor_pass(model, pooled, tk, registry=default_registry(loud=False))
    assert a["n_monitors"] == b["n_monitors"]
    assert a["tripped"] == b["tripped"]
    assert a["inapplicable"] == b["inapplicable"]
    assert a["plan"]["guards"] == b["plan"]["guards"]
    for ra, rb in zip(a["readings"], b["readings"], strict=True):
        assert ra["name"] == rb["name"]
        assert ra["tripped"] == rb["tripped"]
        assert ra["applicable"] == rb["applicable"]


# ---------------------------------------------------------------------------
# the two picklability blockers the probe named
# ---------------------------------------------------------------------------
def test_the_lane_summary_is_picklable_and_duck_types_the_controller(pcfg):
    """The lane used to hand back a LIVE ``CluControllerV0`` — the blocker."""
    scfg = pcfg.store_cfg()
    z = np.random.default_rng(1).normal(size=(12, scfg.dim)).astype(np.float32)
    out = _controller_plan_for_lane(z, scfg, default_registry(loud=False))
    summary = out["_stats"]["controller"]
    assert isinstance(summary, LaneControllerSummary)
    again = pickle.loads(pickle.dumps(out))                     # the whole lane
    assert again["_stats"]["controller"].guard_fire_counts() == summary.guard_fire_counts()
    assert summary.guard_fire_counts() == out["_stats"]["guards"]
    assert summary.n_live == out["_stats"]["n_live_end"]


def test_class_i_snapshot_reproduces_the_live_registry(pcfg):
    """⭐ The registry blocker. The controller reads ``class_i_tripped()`` and
    never writes; nothing inside a plan pass calls ``observe``; so a snapshot is
    equivalent. Asserted with a registry that actually **carries a class-I trip**
    — with an empty registry the claim would be vacuous.
    """
    scfg = pcfg.store_cfg()
    z = np.random.default_rng(3).normal(size=(16, scfg.dim)).astype(np.float32)
    reg = default_registry(loud=False)
    reg.readings.append(MonitorReading(          # 'blank' is class I
        name="blank", mode=4, value=1.0, band="n/a", tripped=True))
    assert reg.class_i_tripped() == ["blank"]
    live = _controller_plan_for_lane(z, scfg, reg)
    snap = _controller_plan_for_lane(z, scfg, _ClassITrips(reg.class_i_tripped()))
    for k in PLAN_FIELDS:
        assert np.array_equal(live[k], snap[k]), k
    assert live["_stats"]["guards"] == snap["_stats"]["guards"]
    assert reg.class_i_tripped() == ["blank"]    # the pass did not mutate it
