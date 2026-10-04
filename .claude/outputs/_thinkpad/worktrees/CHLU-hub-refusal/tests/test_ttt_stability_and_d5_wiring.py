"""`pilot-ttt-nan-and-d5-wiring` — the two tier-iii pilot defects, as tests.

The first landed tier-iii artifact (`run1`/seed 0, 22.78 h on an A100) carried two
defects that this file locks:

**DEFECT 1 — the ``ttt_matched`` arm went NaN at step 135/4000** and never
recovered, so ``static``, ``dyneval`` and both ``margin_vs_clu_*`` are NaN: the
published rival column is missing from the leg. The mechanism is *arithmetic and
structural*, not a bad hyperparameter draw. :class:`MatchedTTTCell`'s inner
update ``W <- W (I - eta k k^T) + eta v k^T`` is non-expansive only while
``eta ||k||^2 < 2``, and ``||k||^2 = ||theta_K z||^2 ~ n ||z||^2 / d`` is fixed by
``n``, which :func:`solve_matched_ttt` reads off the **CLU cell's byte ledger**.
⛔ So the two-sided byte match — the thing that makes the swap fair — silently
chooses the rival's inner-loop stability, and nothing checked it. At the pilot
ledger ``(P, S, d) = (168986, 115072, 12) -> (k, n) = (2197, 52)`` the shipped
update amplifies; at the toy ledger ``(8616, 5144, 3) -> (571, 9)`` it does not.
**That is why every toy gate passed.**

**DEFECT 2 — D5 (the anytime curve) never ran on any launch**: ``--d5`` is a CLI
flag and no launch path set it, so a pre-registered deliverable is absent from
the artifact in a way indistinguishable from a deliberate cut. The wiring is one
line; what is worth a test is the *mechanism* that makes it nearly free —
``--d5`` is not a ``PilotConfig`` field, so it cannot move the resume
fingerprint, and a **finished** leg re-resumed with it lifts every banked phase
(including the 219 GB ``dyneval``) and computes only ``anytime_curve``.

**And the constraint both fixes had to clear:** five CSF3 legs hold banked
journals + ``ckpt_{arm}.eqx``. ``load_journal`` refuses a journal whose config
fingerprint differs *by key* — and the ``memory`` group is a full ``asdict``, so
it already did: the four C2W6 fields (``erosion_partition``, ``refresh_*``) made
the real banked journals unresumable on ``main`` before this task started.
"""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chlu.experiments.exp_cluformer_pilot as EXP
from chlu.core.blocks import (
    MatchedTTTCell,
    StreamMemoryConfig,
    make_memory_cell,
    solve_matched_ttt,
)
from chlu.data import enwik8 as E
from chlu.training.train_cluformer import PilotConfig, build_arm, solve_arms

REPO = Path(__file__).resolve().parents[1]

#: The tier-iii pilot's own ledger, straight off the landed artifact's
#: ``swap_ledger`` (``clu_store`` params / state_floats) and ``store_dim``.
PILOT_LEDGER = (168_986, 115_072, 12)
#: The toy ledger, for the contrast that explains why the toy gate passed.
TOY_LEDGER = (8_616, 5_144, 3)


@pytest.fixture(autouse=True, scope="module")
def float32_dynamics():
    """Pin float32 for the WHOLE module (handover §7.23, N211)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# DEFECT 1 — the inner loop's stability region
# ---------------------------------------------------------------------------
def _cell(ledger, normalized):
    p, s, d = ledger
    k, n = solve_matched_ttt(p, s, d)
    return MatchedTTTCell(d, k, n, key=jax.random.PRNGKey(0),
                          normalized_write=normalized), k, n


def _write_chain(cell, z_seq):
    """``||W||`` after each chunk write — the quantity that overflows."""
    st = cell.init_state()
    out = []
    for z in z_seq:
        st = cell.write(st, z, None)
        out.append(float(jnp.linalg.norm(st)))
    return out


def _worst_direction(cell):
    """The unit ``z`` maximising ``||theta_K z||`` — deterministic, no lucky draw.

    A real chunk stream is strongly *correlated* (successive chunks of one text
    through one ``phi``), so the amplification compounds coherently along one
    direction rather than being averaged away. This is that direction.
    """
    _u, _s, vt = np.linalg.svd(np.asarray(cell.theta_K), full_matrices=False)
    return jnp.asarray(vt[0] / np.linalg.norm(vt[0]))


def _criterion(cell, z):
    return float(jax.nn.softplus(cell.log_eta)) * float(
        jnp.sum((cell.theta_K @ z) ** 2))


def test_the_pilot_byte_ledger_is_what_selects_the_ttt_geometry():
    """⛔ The rival's inner-loop stability is a *consequence* of the byte match."""
    assert solve_matched_ttt(*PILOT_LEDGER) == (2197, 52)
    assert solve_matched_ttt(*TOY_LEDGER) == (571, 9)


def test_the_stability_criterion_is_eta_n_over_d_and_pilot_exceeds_2():
    """⭐ The mechanism in closed form, and the reason the toy gate passed.

    For unit-RMS latents ``E||theta_K z||^2 = n ||z||^2 / d``, so the stability
    product is ``eta n / d`` — **a pure function of the solved geometry**:
    ``0.693 * 52/12 = 3.00`` at pilot against ``0.693 * 9/3 = 2.08`` at toy,
    i.e. the pilot sits decisively outside the ``< 2`` region while the toy sits
    astride its boundary. On the REAL ``phi`` latents (measured, not synthesised)
    the separation is wider still — ``3.47`` on **100 %** of pilot chunks vs
    ``2.31`` on 44 % of toy chunks — and the outcome matches: 300 steps at toy
    geometry are clean, the pilot NaNs at step 107.
    ⚠ The toy is *marginal*, not safe. That is the finding: a toy gate cannot
    certify this arm at another scale, because the scale sets the criterion.
    """
    pil, _kp, n_p = _cell(PILOT_LEDGER, False)
    toy, _kt, n_t = _cell(TOY_LEDGER, False)
    eta = float(jax.nn.softplus(pil.log_eta))
    assert eta * n_p / PILOT_LEDGER[2] > 2.0
    assert eta * n_t / TOY_LEDGER[2] < eta * n_p / PILOT_LEDGER[2]
    # and the analytic mean is what the sampler measures
    z = jax.random.normal(jax.random.PRNGKey(1), (512, PILOT_LEDGER[2])) \
        / np.sqrt(PILOT_LEDGER[2])
    got = float(np.mean([_criterion(pil, zz) for zz in z]))
    assert abs(got - eta * n_p / PILOT_LEDGER[2]) < 0.3, got
    assert got > 2.0


def test_the_shipped_write_runs_away_on_a_coherent_stream_at_pilot_geometry():
    """16 writes — one forward pass' worth of chunks — along one direction."""
    cell, _k, _n = _cell(PILOT_LEDGER, False)
    z = _worst_direction(cell)
    assert _criterion(cell, z) > 2.0
    g = _write_chain(cell, [z] * 16)
    assert g[-1] / g[0] > 1e3, g


def test_normalized_write_is_bounded_and_SCALE_FREE_across_both_geometries():
    """⭐ The property that stops a future ledger solve from re-breaking it.

    Under the normalized rule the factor along ``k`` is ``1 - eta`` whatever
    ``n``, ``d`` or ``||z||`` are, so the growth over 16 coherent writes is the
    SAME number at the pilot ledger and at the toy ledger — bounded, and not
    bounded-by-luck.
    """
    growths = []
    for ledger in (PILOT_LEDGER, TOY_LEDGER):
        cell, _k, _n = _cell(ledger, True)
        z = _worst_direction(cell)
        assert _criterion(cell, z) > 2.0     # the SHIPPED rule would run away
        g = _write_chain(cell, [z] * 16)
        assert np.isfinite(g).all() and g[-1] / g[0] < 10.0, (ledger, g)
        growths.append(g[-1] / g[0])
    assert abs(growths[0] - growths[1]) < 1e-3, growths


def test_normalized_write_at_eta_one_is_the_exact_closed_form_step():
    """⭐ The cell's docstring says "one **closed-form** step"; only the
    normalized rule is one — ``eta = 1`` solves ``W k = v`` along ``k`` exactly."""
    d, k, n = 4, 5, 3
    cell = MatchedTTTCell(d, k, n, key=jax.random.PRNGKey(2), normalized_write=True)
    cell = eqx.tree_at(lambda c: c.log_eta, cell,
                       jnp.asarray(float(np.log(np.expm1(1.0)))))   # softplus -> 1
    z = jax.random.normal(jax.random.PRNGKey(3), (d,))
    W = jax.random.normal(jax.random.PRNGKey(4), (k, n))
    W2 = cell.write(W, z, None)
    kk, vv = cell.theta_K @ z, cell.theta_V @ z
    assert np.allclose(np.asarray(W2 @ kk), np.asarray(vv), atol=1e-4)


def test_the_default_write_is_bit_identical_to_the_shipped_arithmetic():
    """⛔ ``normalized_write=False`` must be the published column, unchanged."""
    d, k, n = 4, 5, 3
    cell = MatchedTTTCell(d, k, n, key=jax.random.PRNGKey(2))
    assert cell.normalized_write is False
    z = jax.random.normal(jax.random.PRNGKey(3), (d,))
    W = jax.random.normal(jax.random.PRNGKey(4), (k, n))
    kk, vv = cell.theta_K @ z, cell.theta_V @ z
    want = W - jax.nn.softplus(cell.log_eta) * jnp.outer(W @ kk - vv, kk)
    assert np.array_equal(np.asarray(cell.write(W, z, None)), np.asarray(want))


def test_the_lever_reaches_the_cell_through_the_config_and_only_the_ttt_arm():
    pcfg = PilotConfig(ttt_normalized_write=True, n_layers=2, arms=("ttt_matched",))
    specs, _ = solve_arms(pcfg, jax.random.PRNGKey(0))
    m = build_arm("ttt_matched", pcfg, specs, key=jax.random.PRNGKey(1))
    assert all(b.cell.normalized_write for b in m.blocks)
    off = build_arm("ttt_matched", PilotConfig(n_layers=2), specs,
                    key=jax.random.PRNGKey(1))
    assert not any(b.cell.normalized_write for b in off.blocks)
    # the flag is inert for every other cell type
    gru = make_memory_cell("gru_matched", latent_dim=4, hidden=3,
                           ttt_normalized_write=True, key=jax.random.PRNGKey(0))
    assert not hasattr(gru, "normalized_write")


def test_the_lever_is_static_so_it_costs_no_parameter_and_no_state_byte():
    """⛔ The swap ledger is a published table; the fix may not move a byte."""
    a, _k, _n = _cell(PILOT_LEDGER, False)
    b, _k2, _n2 = _cell(PILOT_LEDGER, True)
    assert a.cell_ledger() == b.cell_ledger()


# ---------------------------------------------------------------------------
# ⛔ THE HARD CONSTRAINT — the five in-flight legs must still resume
# ---------------------------------------------------------------------------
def test_ttt_normalized_write_defaults_off_and_is_absent_from_the_fingerprint():
    """⭐ Why the knob is a ``PilotConfig`` field and not a ``StreamMemoryConfig``
    one: ``as_flag_table()`` emits **non-default keys only**, so at the default
    the flag block — hence the resume fingerprint — is byte-unchanged."""
    p = PilotConfig()
    assert p.ttt_normalized_write is False
    assert "ttt_normalized_write" not in p.as_flag_table()
    pcfg = EXP.make_config("toy", 0)
    keys = set(EXP._flag_dict({"pilot": pcfg.as_flag_table(),
                               "memory": asdict(pcfg.memory_cfg()),
                               "store": pcfg.store_cfg().as_flag_table(),
                               "store_dim": 3, "store_n_atoms": 128}))
    assert not any("ttt_normalized_write" in k for k in keys)


def test_turning_it_on_is_honestly_declared_as_a_different_leg():
    """⛔ Not a loophole: a leg run WITH the fix is a different leg and says so."""
    assert "ttt_normalized_write" in PilotConfig(
        ttt_normalized_write=True).as_flag_table()


def _flags(**mem):
    return {"pilot": {"steps": 6}, "store": {},
            "memory": dict(asdict(StreamMemoryConfig()), phi_gain=3.0, **mem),
            "store_dim": 3, "store_n_atoms": 128}


def _journal_at(tmp_path, flags):
    EXP.save_json(EXP.partial_path(tmp_path, "toy", 0),
                  {"flags": flags, "arms": {}, "_journal": {"trained": {}}})


def test_a_journal_predating_a_memory_field_is_still_accepted(tmp_path, capsys):
    """⛔⭐ THE REGRESSION THIS TASK EXISTS TO PREVENT.

    The ``memory`` group of the fingerprint is a full ``asdict``, so **every
    field added to ``StreamMemoryConfig`` retro-invalidates every journal on
    disk**. It had already happened on ``main``: the four C2W6 fields make the
    real banked CSF3 journals unresumable, i.e. 4 x 16 h of A100 training that
    the checkpoint machinery was built to protect. A key the journal predates,
    sitting at its own field default, is the SAME leg.
    """
    now = _flags()
    old = {**now, "memory": {k: v for k, v in now["memory"].items()
                             if not (k == "erosion_partition"
                                     or k.startswith("refresh_"))}}
    _journal_at(tmp_path, old)
    capsys.readouterr()
    assert EXP.load_journal(tmp_path, "toy", 0, now) is not None
    out = capsys.readouterr().out
    assert "post-dates this journal and is at its default" in out


def test_a_post_dating_field_set_AWAY_from_its_default_is_still_refused(tmp_path):
    """⛔ The hole the reconciliation must not open."""
    now = _flags(erosion_partition=True)
    old = {**now, "memory": {k: v for k, v in now["memory"].items()
                             if k != "erosion_partition"}}
    _journal_at(tmp_path, old)
    with pytest.raises(SystemExit, match="refusing to resume"):
        EXP.load_journal(tmp_path, "toy", 0, now)


def test_a_field_the_journal_has_and_the_code_lacks_is_still_refused(tmp_path):
    """⛔ A deleted field's old behaviour is not reconstructible — stay strict."""
    now = _flags()
    _journal_at(tmp_path, {**now, "memory": dict(now["memory"], gone_field=7)})
    with pytest.raises(SystemExit, match="refusing to resume"):
        EXP.load_journal(tmp_path, "toy", 0, now)


def test_a_genuinely_different_value_of_a_shared_key_is_still_refused(tmp_path):
    now = _flags()
    _journal_at(tmp_path, {**now,
                           "memory": dict(now["memory"], write_inner_steps=3)})
    with pytest.raises(SystemExit, match="refusing to resume"):
        EXP.load_journal(tmp_path, "toy", 0, now)


# ---------------------------------------------------------------------------
# DEFECT 2 — the D5 passthrough, and the ~free re-resume it unlocks
# ---------------------------------------------------------------------------
JOB = REPO / "scripts" / "csf3" / "job_gpu_cluformer.sh"


def test_the_job_script_declares_D5_and_passes_it_through_like_RESUME():
    txt = JOB.read_text()
    assert 'D5="${D5:-0}"' in txt, "no D5 default declaration"
    resume = '[ "$RESUME" = "1" ] && EXTRA="$EXTRA --resume"'
    d5 = '[ "$D5" = "1" ] && EXTRA="$EXTRA --d5"'
    assert resume in txt and d5 in txt
    # the same idiom, name-for-name — not a lookalike
    assert d5 == resume.replace("RESUME", "D5").replace("--resume", "--d5")
    assert "D5='$D5'" in txt, "the echoed provenance line omits D5"
    subprocess.run(["bash", "-n", str(JOB)], check=True)


def test_every_registered_cli_flag_has_a_job_script_passthrough():
    """⛔ N-registry: a pre-registered phase gated behind a flag no launch path
    sets is indistinguishable, in the artifact, from a deliberate cut."""
    txt = JOB.read_text()
    for flag in ("--steps", "--arms", "--mem", "--store", "--set", "--resume",
                 "--d5"):
        assert f'EXTRA="$EXTRA {flag}' in txt, f"{flag} has no passthrough"


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


_TIMING = {"wall_s", "wall_s_total", "plan_s", "plan_pass_s", "plan_pass_frac",
           "wall_ratio_traj_over_point", "t_s", "wall_clock_s", "cost_ms"}


def _diffs(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            if k not in _TIMING:
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


def test_re_resuming_a_FINISHED_leg_with_d5_runs_only_the_anytime_curve(
        tmp_path, capsys, fake_enwik8):
    """⭐⭐ The single most valuable mechanism in this task, asserted end to end.

    The four dead-then-recovered CSF3 legs are *finished*: journal complete, all
    five arms banked, ``ckpt_{arm}.eqx`` on disk. Re-running them with ``D5=1``
    is either **~free** or **22 h/leg**, and which one it is depends on whether
    the resume re-enters ``clu_store``'s eval block. It does:

    * ``--d5`` is a CLI argument, **not** a ``PilotConfig`` field, so it never
      enters ``rec['flags']`` and the journal check cannot see it;
    * every banked phase is lifted by key — including ``dyneval``, the phase
      that demanded **219 GB** of host RAM and killed four legs;
    * ``anytime_curve`` is the one key not in the journal, so it is the one
      phase that runs;
    * training is skipped for every arm (the checkpoints are loaded).

    ⛔ And the artifact must otherwise be bit-identical: D5 adds a key, it does
    not perturb a number.
    """
    ov = _overrides(fake_enwik8)
    EXP.run_pilot("toy", seed=0, stage="s3", out_dir=str(tmp_path),
                  overrides=ov, with_d5=False)
    first = json.loads(next(tmp_path.glob("pilot_toy_seed0_S3.json")).read_text())
    assert "anytime_curve" not in first["arms"]["clu_store"]

    capsys.readouterr()
    EXP.run_pilot("toy", seed=0, stage="s3", out_dir=str(tmp_path),
                  overrides=ov, with_d5=True, resume=True)
    log = capsys.readouterr().out
    second = json.loads(next(tmp_path.glob("pilot_toy_seed0_S3.json")).read_text())

    for arm in ("clu_store", "none"):
        assert f"[resume] arm '{arm}'" in log and "training SKIPPED" in log
        for phase in ("static", "dyneval"):
            assert f"phase '{arm}/{phase}': lifted from the journal" in log
    for phase in ("blank_store", "gradient_probe_final", "selectors_final"):
        assert f"phase 'clu_store/{phase}': lifted from the journal" in log
    # ⭐ the ONE phase that actually executes
    assert "phase 'clu_store/anytime_curve': lifted" not in log
    assert "[rss] clu_store/anytime_curve/enter" in log

    curve = second["arms"]["clu_store"]["anytime_curve"]
    assert len(curve) == 5 and [p["verlet_per_read"] for p in curve] == \
        sorted(p["verlet_per_read"] for p in curve)
    stripped = dict(second)
    stripped["arms"] = {a: {k: v for k, v in r.items() if k != "anytime_curve"}
                        for a, r in second["arms"].items()}
    assert _diffs(first, stripped) == [], _diffs(first, stripped)[:20]


def test_d5_is_not_a_config_field_so_it_cannot_move_the_resume_fingerprint():
    """The mechanical reason the re-resume above is even attemptable."""
    assert not hasattr(PilotConfig(), "d5")
    assert not hasattr(PilotConfig(), "with_d5")
    assert "d5" not in PilotConfig(steps=7).as_flag_table()
