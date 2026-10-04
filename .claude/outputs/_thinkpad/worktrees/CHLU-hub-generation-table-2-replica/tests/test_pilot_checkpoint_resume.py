"""`pilot-checkpoint-resume` — the crash journal, as tests.

CSF3 attempt 1 (job 18136619) trained ``clu_store`` for 4000 steps over ~22 h and
was then **host-RAM ``oom_kill``ed inside the post-training eval block** — MaxRSS
131 682 856 K ≈ 125.6 GB against a ReqMem of 125.7 GB, i.e. killed *at* the
proportional 12-core gpuA ceiling, with no ``--mem`` that can buy more. Because
``run_pilot`` wrote its JSON only at ``_finish``, the crash cost the entire run.

This file is the control the fix has to pass. The fix is **IO and cache
management only**, so:

* the FINAL artifact gains no key and loses none — the journal's instrumentation
  (``host_rss``, ``_journal``) lives in the PARTIAL, which is additive;
* a run that dies in the eval block and is resumed **reproduces an uninterrupted
  run bitwise** on every non-timing field, *including* monitor #6's window, which
  is why ``monitors_final`` is taken inside the training segment;
* the data stream a resumed arm sees is the stream it would have seen — asserted
  directly, not asserted in prose;
* the cache-hygiene pass is value-inert;
* a journal written under a different config is REFUSED, never silently mixed.
"""

import json

import jax
import numpy as np
import pytest

import chlu.experiments.exp_cluformer_pilot as EXP
from chlu.data import enwik8 as E
from chlu.training.train_cluformer import (
    PilotConfig,
    anytime_curve,
    build_arm,
    host_rss,
    release_host_memory,
    save_json,
    solve_arms,
    train_arm,
)


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.23, N211).

    ⚠ Module scope is load-bearing and must be paired with the restore: other
    test modules enable ``jax_enable_x64`` at import time, and a function-scoped
    fixture would be set up *after* them.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# a network-free stream + a tiny but STRUCTURALLY COMPLETE pilot
# ---------------------------------------------------------------------------
N_FAKE = 40_000


@pytest.fixture(scope="module", autouse=True)
def fake_enwik8(tmp_path_factory):
    """A 40 000-byte fake 'enwik8'. CI never touches a Hutter-Prize mirror."""
    root = tmp_path_factory.mktemp("enwik8")
    (root / "enwik8").write_bytes((np.arange(N_FAKE) % 251).astype(np.uint8).tobytes())
    saved = (E.N_TOTAL, E.N_TRAIN, E.N_VALID, E.N_TEST)
    E.N_TOTAL, E.N_TRAIN, E.N_VALID, E.N_TEST = N_FAKE, 36_000, 2_000, 2_000
    yield root
    E.N_TOTAL, E.N_TRAIN, E.N_VALID, E.N_TEST = saved


def _overrides(root):
    """The real store, the real controller, the real plan pass — just small."""
    return dict(
        d_model=16, n_layers=2, seq_len=16, batch=2, vocab_size=256,
        addr_dim=2, payload_dim=1, capacity=4, atoms_per_item=16,
        steps=2, warmup=1, eval_batches=1, dyneval_batches=1,
        monitor_every=1, data_bytes=N_FAKE, data_root=str(root),
        arms=("clu_store", "none"),
        store=dict(min_atoms=64, min_atoms_base=32),
        memory=dict(chunk=8, address_steps=4, read_steps=4, traj_stride=2,
                    psi_hidden=8, write_inner_steps=1, write_n_perturb=4,
                    retry_rounds=1, conv_kernel=3, mlp_mult=2),
    )


#: Fields that are wall-clock and were never reproducible. Declared in advance
#: (PREREG P1) rather than discovered when a comparison fails.
_TIMING = {"wall_s", "wall_s_total", "plan_s", "plan_pass_s", "plan_pass_frac",
           "wall_ratio_traj_over_point", "t_s", "wall_clock_s", "cost_ms"}


def _diffs(a, b, path="", out=None):
    """Every non-timing leaf on which two records differ, as ``path`` strings."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            if k in _TIMING:
                continue
            _diffs(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _diffs(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (np.isnan(a) and np.isnan(b))):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def _artifact(d):
    return json.loads(next(d.glob("pilot_toy_seed*_S*.json")).read_text())


@pytest.fixture(scope="module")
def reference(tmp_path_factory, fake_enwik8):
    """One clean, uninterrupted run — the thing every other run must match."""
    d = tmp_path_factory.mktemp("ref")
    EXP.run_pilot("toy", seed=0, stage="s3", out_dir=str(d),
                  overrides=_overrides(fake_enwik8), with_d5=True)
    return d, _artifact(d)


# ---------------------------------------------------------------------------
# ⛔ the artifact's content-shape is unchanged
# ---------------------------------------------------------------------------
def test_the_final_artifact_carries_no_journal_key(reference):
    """The PARTIAL is additive; the FINAL is not allowed to grow.

    ``--plot-only``/:func:`aggregate` and the analyst read this file, so the
    instrumentation must not leak into it.
    """
    _d, rec = reference
    for k in EXP._JOURNAL_ONLY_KEYS:
        assert k not in rec, f"the final artifact gained '{k}'"
    # ⭐ `c3-csf3-harness` adds exactly ONE top-level key: `byte_ledger`. It is a
    # deliberate content-shape change, not instrumentation leaking out of the
    # PARTIAL — the tier-iii claim is at matched params AND matched state-bytes,
    # so an arm that cannot state its inference-time state in bytes (incl. φ)
    # must not be reportable, and the ledger is therefore emitted by EVERY run.
    assert list(rec) == [
        "scale", "seed", "stage_requested", "flags", "stages_reached", "not_run",
        "data", "phi_gain_calibrated", "swap_ledger", "byte_ledger", "shell",
        "total_params",
        "monitors_init", "allocation_liveness_init", "gradient_probe_init",
        "train_log", "arms", "swap_table", "wall_s_total"]
    assert list(rec["arms"]["clu_store"]) == [
        "train", "static", "dyneval", "blank_store", "anytime_curve",
        "monitors_during", "monitors_final", "gradient_probe_final",
        "selectors_final", "wall_s"]
    assert list(rec["arms"]["none"]) == ["train", "static", "dyneval", "wall_s"]


def test_the_journal_and_one_checkpoint_per_arm_land_on_disk(reference):
    d, rec = reference
    p = EXP.partial_path(d, "toy", 0)
    assert p.exists(), "no PARTIAL journal was written"
    j = json.loads(p.read_text())
    assert set(j["_journal"]["trained"]) == {"clu_store", "none"}
    assert j["host_rss"] and j["host_rss"][0]["phase"] == "run_pilot/enter"
    for arm in rec["arms"]:
        assert EXP.ckpt_path(d, arm, 0).exists(), arm
    assert not list(d.glob("*.tmp")), "an atomic write left its tmp sibling behind"


def test_plot_only_glob_does_not_pick_up_the_partial(reference):
    """``pilot_toy_seed0_PARTIAL.json`` must not be aggregated as a record."""
    d, _rec = reference
    assert [p.name for p in sorted(d.glob("pilot_toy_seed*_S*.json"))] == [
        "pilot_toy_seed0_S3.json"]


# ---------------------------------------------------------------------------
# ⭐ the resume itself — a crash in the eval block, exactly as CSF3 saw it
# ---------------------------------------------------------------------------
def test_resume_after_an_eval_block_crash_is_bitwise(tmp_path, monkeypatch,
                                                     capsys, reference, fake_enwik8):
    """Kill the run where attempt 1 died, resume, and demand bit-identity.

    ⭐ The kill lands in ``dynamic_eval`` — i.e. *after* 100 % of ``clu_store``'s
    training and after its ``static`` column, which is the situation that cost
    22 h. The resumed run must (a) NOT retrain, (b) lift ``static`` verbatim,
    (c) recompute ``dyneval`` onwards, and (d) agree bitwise with a run that was
    never interrupted.
    """
    ref_dir, ref = reference
    ov = _overrides(fake_enwik8)

    def boom(*a, **k):
        raise RuntimeError("simulated host-RAM oom_kill in the eval block")

    monkeypatch.setattr(EXP, "dynamic_eval", boom)
    with pytest.raises(RuntimeError):
        EXP.run_pilot("toy", seed=0, stage="s3", out_dir=str(tmp_path),
                      overrides=ov, with_d5=True)
    monkeypatch.undo()

    j = json.loads(EXP.partial_path(tmp_path, "toy", 0).read_text())
    assert "clu_store" in j["_journal"]["trained"], "the weights were not banked"
    assert EXP.ckpt_path(tmp_path, "clu_store", 0).exists()
    assert "static" in j["arms"]["clu_store"], "the static column was not banked"
    assert "dyneval" not in j["arms"]["clu_store"]

    capsys.readouterr()
    EXP.run_pilot("toy", seed=0, stage="s3", out_dir=str(tmp_path),
                  overrides=ov, with_d5=True, resume=True)
    log = capsys.readouterr().out
    assert "training SKIPPED" in log, "the resumed run retrained a banked arm"
    assert "phase 'clu_store/static': lifted" in log
    assert "phase 'monitors_init': lifted" in log

    got = _artifact(tmp_path)
    assert _diffs(ref, got) == [], _diffs(ref, got)[:20]
    assert got["arms"]["clu_store"]["monitors_final"]["n_applicable"] == \
        ref["arms"]["clu_store"]["monitors_final"]["n_applicable"], (
            "monitor #6's window did not survive the resume — `monitors_final` "
            "must be taken inside the training segment, while the persistent "
            "registry is still alive")
    assert ref_dir != tmp_path


def test_resume_refuses_a_journal_written_under_a_different_config(
        tmp_path, reference, fake_enwik8):
    """⛔ A resumed leg must be the SAME leg (§A20.4)."""
    d, _ = reference
    j = json.loads(EXP.partial_path(d, "toy", 0).read_text())
    save_json(EXP.partial_path(tmp_path, "toy", 0), j)
    ov = dict(_overrides(fake_enwik8))
    ov["memory"] = dict(ov["memory"], write_inner_steps=3)   # a DIFFERENT leg
    with pytest.raises(SystemExit, match="refusing to resume"):
        EXP.run_pilot("toy", seed=0, stage="s1", out_dir=str(tmp_path),
                      overrides=ov, resume=True)


def test_stop_after_arms_is_exempt_from_the_resume_flag_check():
    """The interrupted run carries it; its resumption does not."""
    a = {"pilot": {"stop_after_arms": 1, "steps": 6}, "memory": {"phi_gain": 3.0},
         "store": {}, "store_dim": 3, "store_n_atoms": 128}
    b = {"pilot": {"steps": 6}, "memory": {"phi_gain": 9.9},
         "store": {}, "store_dim": 3, "store_n_atoms": 128}
    assert EXP._flag_fingerprint(a) == EXP._flag_fingerprint(b)
    c = {**b, "pilot": {"steps": 7}}
    assert EXP._flag_fingerprint(c) != EXP._flag_fingerprint(b)


# ---------------------------------------------------------------------------
# ⭐ the guarantee the resume rests on: the stream is arm-independent
# ---------------------------------------------------------------------------
def test_the_training_stream_carries_nothing_between_arms(fake_enwik8):
    """⭐ Why no fast-forwarding is needed, asserted rather than argued.

    ``_train_batches`` materialises the stream once from ``(seed, steps)`` and
    every arm consumes a FRESH iterator over that same list, so arm *k*'s
    batches are independent of arms ``0..k-1`` — and of whether those arms ran
    in this process at all.
    """
    pcfg = EXP.make_config("toy", 0, _overrides(fake_enwik8))
    tr, _va, _te = EXP._data(pcfg)
    a = EXP._train_batches(tr, pcfg)
    b = EXP._train_batches(tr, pcfg)
    assert len(a) == len(b) == pcfg.steps
    for (xa, ya), (xb, yb) in zip(a, b, strict=True):
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    # two arms, two fresh iterators, same bytes in the same order
    first_arm = [x for x, _ in iter(a)]
    second_arm = [x for x, _ in iter(a)]
    assert all(np.array_equal(p, q)
               for p, q in zip(first_arm, second_arm, strict=True))


def test_checkpoint_round_trip_is_bitwise(fake_enwik8):
    """``build_arm`` from ``seed`` alone is a valid deserialisation template."""
    pcfg = EXP.make_config("toy", 0, _overrides(fake_enwik8))
    specs, _ = solve_arms(pcfg, jax.random.PRNGKey(3))
    m = build_arm("clu_store", pcfg, specs, key=jax.random.PRNGKey(7))
    out = fake_enwik8
    EXP.save_arm_checkpoint(out, "clu_store", 0, m)
    like = build_arm("clu_store", pcfg, specs, key=jax.random.PRNGKey(11))
    back = EXP.load_arm_checkpoint(out, "clu_store", 0, like)
    import equinox as eqx

    la = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(back, eqx.is_inexact_array))
    assert len(la) == len(lb) and la
    for x, y in zip(la, lb, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# ---------------------------------------------------------------------------
# host-memory hygiene + instrumentation
# ---------------------------------------------------------------------------
def test_host_rss_reports_a_finite_peak_and_a_child_column():
    r = host_rss()
    assert np.isfinite(r["hwm_gb"]) and r["hwm_gb"] > 0.0
    assert "children_rss_gb" in r and "n_children" in r
    release_host_memory()          # must not raise, must not change anything
    assert np.isfinite(host_rss()["hwm_gb"])


def test_proc_status_parser_keeps_the_full_vm_key_names(tmp_path, monkeypatch):
    """Regression: the Linux /proc parse path, which macOS never executes.

    A ``k[:-1]`` in the parser once truncated the keys to ``VmRS``/``VmHW`` —
    the parent then printed ``nan`` and the children loop raised ``KeyError``,
    killing every CSF3 resubmission job at the first phase boundary (42 s in).
    This feeds the parser real /proc-format text so the bug class cannot land
    on the cluster untested again.
    """
    import builtins

    import chlu.training.train_cluformer as T

    fake = tmp_path / "status"
    fake.write_text("Name:\tpython\nVmHWM:\t  234567 kB\nVmRSS:\t  123456 kB\n")
    real_open = builtins.open
    monkeypatch.setattr(
        builtins, "open",
        lambda path, *a, **k: real_open(fake, *a, **k)
        if str(path).startswith("/proc/") else real_open(path, *a, **k),
    )
    st = T._proc_status_kb(12345)
    assert st == {"VmRSS": 123456.0, "VmHWM": 234567.0}


def test_host_rss_children_survive_a_keyless_status_and_a_nan_ps(monkeypatch):
    """A zombie/vanished worker (status readable but no ``VmRSS``, ``ps`` NaN)
    must be skipped — never a ``KeyError``, never a NaN-poisoned sum."""
    import chlu.training.train_cluformer as T

    class _FakePool:
        _processes = {999999991: None}

    monkeypatch.setitem(T._POOLS, "_test_fake", _FakePool())
    # self → {} (parent takes the getrusage fallback); child → keyless-but-truthy
    monkeypatch.setattr(T, "_proc_status_kb",
                        lambda pid="self": {} if pid == "self" else {"VmHWM": 1.0})
    monkeypatch.setattr(T, "_ps_rss_kb", lambda pid: float("nan"))
    r = T.host_rss()
    assert r["children_rss_gb"] == 0.0 and r["n_children"] == 0.0
    assert np.isfinite(r["hwm_gb"])


def test_cache_hygiene_is_value_inert_on_the_anytime_curve(fake_enwik8):
    """⛔ ``release_host_memory`` drops executables, never values."""
    pcfg = EXP.make_config("toy", 0, _overrides(fake_enwik8))
    specs, _ = solve_arms(pcfg, jax.random.PRNGKey(3))
    m = build_arm("clu_store", pcfg, specs, key=jax.random.PRNGKey(7))
    _tr, _va, te = EXP._data(pcfg)
    ev = EXP._eval_batches(te, pcfg, 1)
    budgets = [(2, 2), (4, 4)]
    off = anytime_curve(m, pcfg, ev, budgets, hygiene=False)
    on = anytime_curve(m, pcfg, ev, budgets, hygiene=True)
    for a, b in zip(off, on, strict=True):
        assert a["bpc"] == b["bpc"], (a, b)
        assert a["verlet_per_read"] == b["verlet_per_read"]


def test_save_json_atomic_leaves_no_tmp_and_round_trips(tmp_path):
    p = tmp_path / "j.json"
    save_json(p, {"a": [1.0, 2.0]}, atomic=True)
    assert json.loads(p.read_text()) == {"a": [1.0, 2.0]}
    assert not list(tmp_path.glob("*.tmp"))


def test_the_new_host_memory_knobs_default_to_the_fix(capsys):
    """⭐ Unlike `csf3-memory-fit`'s levers these ship ON — the Head directed the
    footprint reduction and §A20.4 reruns all six legs on this code uniformly."""
    p = PilotConfig()
    assert p.eval_cache_hygiene is True
    assert p.rss_log is True
    assert p.stop_after_arms == 0        # ⛔ the test hook is OFF
    assert "stop_after_arms" not in p.as_flag_table()
    _ = capsys


# ---------------------------------------------------------------------------
# the `csf3-memory-fit` §7 erratum: the per-25-steps timing print
# ---------------------------------------------------------------------------
def test_train_arm_prints_wall_and_plan_seconds(fake_enwik8, capsys):
    """`csf3-memory-fit` §7 claimed this line existed; attempt 1's `.out`/`.err`
    carried no such line and the first A100 step time had to be reconstructed
    from wallclock. It exists now."""
    pcfg = EXP.make_config("toy", 0, dict(_overrides(fake_enwik8), steps=2))
    specs, _ = solve_arms(pcfg, jax.random.PRNGKey(3))
    m = build_arm("none", pcfg, specs, key=jax.random.PRNGKey(7))
    tr, _va, _te = EXP._data(pcfg)
    capsys.readouterr()
    train_arm("none", m, pcfg, iter(EXP._train_batches(tr, pcfg)), log_every=1)
    out = capsys.readouterr().out
    assert "[train/none] step 0/2" in out
    assert "wall_s" in out and "plan_s" in out and "s/step" in out
