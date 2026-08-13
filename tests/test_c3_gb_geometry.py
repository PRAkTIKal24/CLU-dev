"""⭐ **GEOMETRY G-B**: the full-size store in 3 of 12 layers — pinned by test.

`c3-gb-landing`. G-B was RATIFIED (Head + Advisor, 2026-08-13) after
`c3-rival-ladder-prereg` reported that **no config-only geometry** fits the ruled
≈2 MB ceiling without descending below the w23 atom floor (`capacity` and
`atoms_per_item` are BYTE-INERT at ``addr_dim = 8``; handover 7.38). G-B keeps the
floor intact and uses **fewer cells** instead of **starved cells**.

What is pinned here:

1. ⭐ **the byte arithmetic** — 3 store-bearing layers × 460,288 B =
   **1,380,864 B**, occupancy **0.658×** of 2 MiB, and the CLU/TTT-matched match
   ratio **1.0072×** preserved (the two-sided swap survives the selection);
2. ⛔ **the pilot path is UNDISTURBED** — an unset ``store_layers`` emits nothing
   into ``as_flag_table()``, so it cannot become a SECOND key differing from the
   run-2 journal, which would break run 3's pre-registered-continuation
   exemption. This is the same failure mode the exemption spoke found for its own
   flag (report §2.1) and it is asserted end-to-end against a run-2-shaped
   journal, not merely on the dataclass;
3. ⭐ **the selection is explicit, never a default** — ``None`` means *every
   layer*, and a bad selection raises rather than silently moving the ledger;
4. ⚠ **G-B's declared price**: it keeps the pilot's PER-LAYER cell, so
   ``solve_matched_ttt`` still returns ``(2197, 52)`` and the ``eta*n/d >= 2``
   criterion still fires ⇒ ``ttt_normalized_write=True`` on the TTT arm
   (PILOT-TTT-RULINGS ruling 1 — already ruled, recorded here, not re-decided);
5. ⛔ **run 3's launch blocker B2** — ``job_gpu_cluformer.sh`` carries the
   ``PREREG_CONT`` passthrough, and the command line it emits carries
   ``--prereg-continuation``;
6. ⛔ the ``StateByteBudgetError`` **remedy text names levers that MOVE BYTES**.
"""

from __future__ import annotations

import json
import subprocess

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The RATIFIED G-B numbers. ⛔ These are the ratified arithmetic; if one of them
#: moves, the geometry moved and the decision must be retaken, not the test.
GB_STORE_LAYERS = (2, 6, 10)
GB_N_STORE_LAYERS = 3
GB_CELL_BYTES_PER_LAYER = 460_288
GB_TOTAL_STATE_BYTES = 1_380_864          # 3 * 460,288
GB_TTT_TOTAL_STATE_BYTES = 1_370_928      # 3 * 456,976
CEILING = 2_097_152                       # 2 MiB (INTERIM constant, unchanged)
PILOT_N_LAYERS = 12


def _pilot(**ov):
    import chlu.experiments.exp_cluformer_pilot as EXP

    return EXP.make_config("pilot", 0, ov)


def _small(**ov):
    """A structurally identical but tiny model — the byte cells use ``_pilot``."""
    base = dict(d_model=32, n_layers=6, seq_len=64, batch=2,
                addr_dim=4, payload_dim=2, capacity=4, atoms_per_item=4,
                store=dict(min_atoms_base=4, min_atoms=1),
                memory=dict(chunk=16, address_steps=4, read_steps=4, traj_stride=2,
                            psi_hidden=8, write_inner_steps=2, write_n_perturb=2))
    base.update(ov)
    return _pilot(**base)


def _swap_ledger(pcfg):
    """The per-cell ledger, measured off CONSTRUCTED cells (not a formula)."""
    import jax

    from chlu.training.train_cluformer import solve_arms

    return solve_arms(pcfg, jax.random.PRNGKey(0))[1]


# ==========================================================================
# 1. ⭐ THE RATIFIED BYTE ARITHMETIC
# ==========================================================================
def test_the_pilot_cell_is_460288_B_per_layer_and_12_layers_bust_the_ceiling():
    """The premise of the whole decision: the per-layer cell is unchanged."""
    from chlu.eval.byte_ledger import arm_ledger

    pcfg = _pilot()
    led = _swap_ledger(pcfg)
    assert int(led["clu_store"]["state_bytes"]) == GB_CELL_BYTES_PER_LAYER
    row = arm_ledger("clu_store", pcfg, led, budget=CEILING)
    assert row["n_store_layers"] == PILOT_N_LAYERS == row["n_layers"]
    assert row["total_state_bytes"] == 12 * GB_CELL_BYTES_PER_LAYER == 5_523_456
    assert row["within_budget"] is False
    assert row["occupancy"] == pytest.approx(2.6338, abs=1e-4)


def test_G_B_is_1_380_864_B_and_0_658x_of_the_2_MiB_CEILING():
    """⭐⭐ THE RATIFIED NUMBER. 3 store-bearing layers x 460,288 B."""
    from chlu.eval.byte_ledger import arm_ledger

    pcfg = _pilot(store_layers=GB_STORE_LAYERS)
    led = _swap_ledger(pcfg)
    row = arm_ledger("clu_store", pcfg, led, budget=CEILING)

    assert row["n_layers"] == PILOT_N_LAYERS            # the model is still 12 deep
    assert row["n_store_layers"] == GB_N_STORE_LAYERS   # ...3 of them carry a store
    assert row["store_layer_fraction"] == pytest.approx(0.25)
    assert row["total_state_bytes"] == GB_TOTAL_STATE_BYTES
    assert row["total_state_bytes"] == GB_N_STORE_LAYERS * GB_CELL_BYTES_PER_LAYER
    assert row["occupancy"] == pytest.approx(0.658, abs=5e-4)
    assert row["within_budget"] is True
    assert row["phi_accounted"] is True
    # ⛔ the arithmetic string states the denominator, so a reader of the artifact
    # never has to infer how many layers the total was summed over
    assert "n_store_layers * cell_state_bytes" in row["arithmetic"]
    assert "store-bearing layers [2, 6, 10] of 12" in row["arithmetic"]
    assert row["store_layer_indices"] == [2, 6, 10]


def test_G_B_preserves_the_two_sided_CLU_TTT_MATCH_RATIO():
    """⭐ The swap stays two-sided: both members lose the same 9 layers."""
    from chlu.eval.byte_ledger import arm_ledger

    pcfg = _pilot(store_layers=GB_STORE_LAYERS)
    led = _swap_ledger(pcfg)
    clu = arm_ledger("clu_store", pcfg, led, budget=CEILING)
    ttt = arm_ledger("ttt_matched", pcfg, led, budget=CEILING)

    assert clu["total_state_bytes"] == GB_TOTAL_STATE_BYTES
    assert ttt["total_state_bytes"] == GB_TTT_TOTAL_STATE_BYTES
    ratio = clu["total_state_bytes"] / ttt["total_state_bytes"]
    assert ratio == pytest.approx(1.0072, abs=5e-4)
    # the pre-registered admissible band for the match (PREREG-C3-LADDER §5.1)
    assert 0.99 <= ratio <= 1.01
    assert ttt["within_budget"] and ttt["occupancy"] == pytest.approx(0.654, abs=5e-4)


def test_every_arm_fits_the_ceiling_under_G_B_and_the_null_arms_are_still_zero():
    from chlu.eval.byte_ledger import build_byte_ledger

    pcfg = _pilot(store_layers=GB_STORE_LAYERS)
    led = _swap_ledger(pcfg)
    art = build_byte_ledger(pcfg, led, pcfg.arms, budget=CEILING, enforce=True,
                            verbose=False)
    assert art["over_budget"] == []
    assert art["arms"]["none"]["total_state_bytes"] == 0
    assert art["arms"]["echo"]["total_state_bytes"] == 0
    # ⛔ and the SAME config with the store in every layer still busts it: the
    # selection is what buys the compliance, nothing else moved.
    over = build_byte_ledger(_pilot(), led, pcfg.arms, budget=CEILING,
                             enforce=False, verbose=False)
    assert over["over_budget"] == ["clu_store", "ttt_matched"]


# ==========================================================================
# 2. ⛔ THE PILOT PATH IS UNDISTURBED — run 3 must stay submittable
# ==========================================================================
def test_an_unset_store_layers_emits_NOTHING_into_the_pilot_flag_table():
    """⛔ The stop condition of this task, asserted directly."""
    from chlu.training.train_cluformer import PilotConfig

    pcfg = _pilot()
    assert pcfg.store_layers is None
    assert "store_layers" not in pcfg.as_flag_table()
    assert "store_layers" not in PilotConfig().as_flag_table()
    # ...and when it IS set it is emitted, as a TUPLE (hashable, and equal to the
    # default only when it is the default)
    on = _pilot(store_layers=GB_STORE_LAYERS)
    assert on.as_flag_table()["store_layers"] == GB_STORE_LAYERS


def test_the_run2_run3_flag_block_still_differs_by_EXACTLY_the_ONE_key(tmp_path):
    """⛔⛔ The end-to-end guard: run 3's exemption must still verify.

    The exemption's own report (§2.1) measured that a new ``PilotConfig`` field
    which enters ``as_flag_table()`` becomes a SECOND differing key and breaks the
    identity check the exemption is verified by. ``store_layers`` must not do
    that — asserted against a run-2-shaped journal, through the real verifier.
    """
    import chlu.experiments.exp_cluformer_pilot as EXP

    cfg2 = EXP.make_config("toy", 0)
    flags = EXP.flag_block(cfg2)
    for k in ("erosion_partition", "refresh_amp_ceiling", "refresh_max_gain",
              "refresh_monotonic"):
        flags["memory"].pop(k)                     # as the REAL run-2 journals are
    assert "store_layers" not in flags["pilot"]    # ⭐ the new field is INVISIBLE
    j = EXP.partial_path(tmp_path, "toy", 0)
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(json.dumps({"flags": flags, "arms": {},
                             "_journal": {"trained": {}}}))

    cfg3 = EXP.make_config("toy", 0, {
        "memory": dict(EXP.TOY["memory"], erosion_partition=True)})
    ex = EXP.verify_preregistered_continuation(
        {"journal": str(j), "flag": "memory.erosion_partition",
         "prereg": ".claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md"},
        cfg3)
    assert list(ex.fingerprint_keys_moved) == ["memory.erosion_partition",
                                               "pilot.memory"]


def test_a_G_B_config_IS_refused_as_a_continuation_of_a_pilot_journal(tmp_path):
    """⭐ The other direction, and it is the one that protects the claim: the
    ladder's geometry is NOT run 2's, so it cannot ride run 3's exemption."""
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    flags = EXP.flag_block(EXP.make_config("toy", 0))
    j = EXP.partial_path(tmp_path, "toy", 0)
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(json.dumps({"flags": flags, "arms": {},
                             "_journal": {"trained": {}}}))
    gb = EXP.make_config("toy", 0, {
        "memory": dict(EXP.TOY["memory"], erosion_partition=True),
        "store_layers": (0,)})
    with pytest.raises(ContinuationExemptionError) as e:
        EXP.verify_preregistered_continuation(
            {"journal": str(j), "flag": "memory.erosion_partition",
             "prereg": "P.md"}, gb)
    assert "store_layers" in str(e.value)


def test_a_store_layers_journal_RESUMES_despite_the_tuple_list_round_trip(tmp_path):
    """⚠ JSON has no tuple: the journal stores ``[2, 6, 10]`` and the live config
    holds ``(2, 6, 10)``. The fingerprint compares ``json.dumps`` of both, so they
    match — asserted, because a false refusal here costs a cluster job."""
    import chlu.experiments.exp_cluformer_pilot as EXP

    cfg = EXP.make_config("toy", 0, {"store_layers": (0,)})
    j = EXP.partial_path(tmp_path, "toy", 0)
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(json.dumps({"flags": EXP.flag_block(cfg), "arms": {},
                             "_journal": {"trained": {}}}))
    assert json.loads(j.read_text())["flags"]["pilot"]["store_layers"] == [0]
    assert EXP.load_journal(tmp_path, "toy", 0, EXP.flag_block(cfg)) is not None
    # ...and a DIFFERENT selection is refused, by name
    other = EXP.make_config("toy", 0, {"store_layers": (1,)})
    with pytest.raises(SystemExit) as e:            # the §A20.4 refusal
        EXP.load_journal(tmp_path, "toy", 0, EXP.flag_block(other))
    assert "store_layers" in str(e.value)


def test_ttt_normalized_write_is_NOT_a_memory_field_and_MEM_would_DROP_it():
    """⛔⛔ The trap that breaks RUN3-LAUNCH.md §4 as written (`c3-gb-landing`).

    ``ttt_normalized_write`` is a **PilotConfig** field. Passed through ``--mem``
    it (a) never reaches the config — ``StreamMemoryConfig.from_mapping`` filters
    unknown keys silently — and (b) still lands in the raw ``pilot.memory``
    override dict, so it becomes a SECOND key differing from the run-2 journal and
    the pre-registered-continuation exemption REFUSES the leg. Pinned so nobody
    writes that submission line again.
    """
    import chlu.experiments.exp_cluformer_pilot as EXP

    cfg = EXP.make_config("toy", 0, {
        "memory": dict(EXP.TOY["memory"], ttt_normalized_write=True)})
    assert cfg.ttt_normalized_write is False            # (a) silently dropped
    assert not hasattr(cfg.memory_cfg(), "ttt_normalized_write")
    assert cfg.as_flag_table()["memory"]["ttt_normalized_write"] is True   # (b)


def test_the_default_model_is_STRUCTURALLY_UNCHANGED_by_the_new_field():
    """⛔ Every layer still carries the arm's cell when nothing is selected."""
    import jax

    from chlu.core.blocks import CluStoreCell
    from chlu.training.train_cluformer import ArmSpec, build_arm

    pcfg = _small(n_layers=4)
    m = build_arm("clu_store", pcfg, {"clu_store": ArmSpec("clu_store")},
                  key=jax.random.PRNGKey(0))
    assert m.store_layers == (0, 1, 2, 3)
    assert all(isinstance(b.cell, CluStoreCell) for b in m.blocks)


# ==========================================================================
# 3. ⭐ THE SELECTION ITSELF — explicit, validated, never a silent default
# ==========================================================================
def test_resolve_store_layers_defaults_to_EVERY_layer_and_never_to_a_placement():
    from chlu.core.blocks import parse_store_layers, resolve_store_layers

    assert resolve_store_layers(12, None) == tuple(range(12))   # ⛔ NOT (0, 1, 2)
    assert parse_store_layers(None) is None
    assert resolve_store_layers(12, [10, 2, 6]) == (2, 6, 10)   # sorted
    assert resolve_store_layers(12, "2,6,10") == (2, 6, 10)     # the `--set` form
    assert resolve_store_layers(12, "2 6 10") == (2, 6, 10)
    assert parse_store_layers([10, 2, 6]) == (10, 2, 6)         # syntax only


@pytest.mark.parametrize("bad", [(), [12], [-1], [0, 0], [3, 3, 7], "", [1.5, "x"]])
def test_a_BAD_store_layer_selection_RAISES_rather_than_moving_the_ledger(bad):
    from chlu.core.blocks import resolve_store_layers

    with pytest.raises(ValueError):
        resolve_store_layers(12, bad)


def test_the_MODEL_puts_the_cell_in_exactly_the_selected_layers():
    import jax

    from chlu.core.blocks import CluStoreCell, NullMemoryCell
    from chlu.training.train_cluformer import ArmSpec, build_arm

    pcfg = _small(store_layers=(1, 4))
    m = build_arm("clu_store", pcfg, {"clu_store": ArmSpec("clu_store")},
                  key=jax.random.PRNGKey(0))
    assert m.store_layers == (1, 4)
    kinds = [isinstance(b.cell, CluStoreCell) for b in m.blocks]
    assert kinds == [False, True, False, False, True, False]
    assert all(isinstance(m.blocks[i].cell, NullMemoryCell)
               for i in (0, 2, 3, 5))


def test_the_SHELL_stays_bit_identical_across_arms_under_a_selection():
    """⛔ The swap is only a swap if everything except the cell is identical —
    including across store-layer selections (the shell key is ks[2+i] either
    way)."""
    import jax

    from chlu.training.train_cluformer import (
        assert_shared_shell_identical, build_arm, solve_arms)

    pcfg = _small(store_layers=(1, 4))
    key = jax.random.PRNGKey(0)
    specs, _ = solve_arms(pcfg, key)
    models = {a: build_arm(a, pcfg, specs, key=key)
              for a in ("clu_store", "ttt_matched", "none", "echo")}
    assert_shared_shell_identical(models)           # raises if it is not a swap
    # ⭐ and the shell is the same one the SELECTION-FREE model has
    full = build_arm("clu_store", _small(), specs, key=key)
    assert_shared_shell_identical({"sel": models["clu_store"], "full": full})


def test_the_none_arm_is_BIT_IDENTICAL_with_and_without_a_selection():
    """⭐ The null arm has null cells everywhere anyway, so the selection cannot
    move it — a cheap, sharp check that the selection touches ONLY the slot."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from chlu.training.train_cluformer import ArmSpec, build_arm

    key = jax.random.PRNGKey(0)
    spec = {"none": ArmSpec("none")}
    a = build_arm("none", _small(), spec, key=key)
    b = build_arm("none", _small(store_layers=(1, 4)), spec, key=key)
    la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_inexact_array))
    assert len(la) == len(lb)
    assert all(bool(jnp.all(x == y)) for x, y in zip(la, lb, strict=True))


def test_a_selected_model_RUNS_and_the_plan_pass_skips_the_non_store_layers():
    """⭐ G-B's compute claim, on the host side too: the controller is not run for
    a layer whose null cell would ignore its plan (decision-inert by
    construction)."""
    import jax
    import jax.numpy as jnp

    from chlu.training.train_cluformer import ArmSpec, build_arm, plan_pass

    pcfg = _small(n_layers=4, store_layers=(1,))
    m = build_arm("clu_store", pcfg, {"clu_store": ArmSpec("clu_store")},
                  key=jax.random.PRNGKey(0))
    tokens = jnp.zeros((2, 64), jnp.int32).at[:, ::3].set(65)
    plans, diag = plan_pass(m, tokens, pcfg)
    assert len(plans) == 4
    ran = [ly.get("store_layer", True) for ly in diag["layers"]]
    assert ran == [False, True, False, False]
    assert diag["layers"][1]["offers"] > 0          # the store layer DID plan
    logits = jax.vmap(lambda t, *p: m(t, list(p)))(tokens, *plans)
    assert logits.shape == (2, 64, pcfg.vocab_size)
    assert bool(jnp.all(jnp.isfinite(logits)))


# ==========================================================================
# 4. ⚠ G-B'S DECLARED PRICE — the TTT criterion still fires (already ruled)
# ==========================================================================
def test_G_B_keeps_the_per_layer_cell_so_the_TTT_eta_n_over_d_CRITERION_FIRES():
    """⚠ Recorded, NOT re-decided: PILOT-TTT-RULINGS ruling 1 already says the
    TTT arm is submitted with ``ttt_normalized_write=True``. G-B shrinks the
    NUMBER of cells, not the cell, so ``solve_matched_ttt`` sees the identical
    per-layer ledger and the divergent inner loop returns."""
    import jax

    from chlu.core.blocks import MatchedTTTCell, solve_matched_ttt

    pcfg = _pilot(store_layers=GB_STORE_LAYERS)
    led = _swap_ledger(pcfg)["clu_store"]
    dim = int(pcfg.store_cfg().dim)
    k, n = solve_matched_ttt(int(led["params"]), int(led["state_floats"]), dim)
    assert (k, n) == (2197, 52)                     # the pilot's solve, unchanged
    # eta = softplus(log_eta = 0) = ln 2, and E||theta_K z||^2 = n||z||^2/d
    cell = MatchedTTTCell(dim, k, n, key=jax.random.PRNGKey(0))
    eta = float(jax.nn.softplus(cell.log_eta))
    eta_n_over_d = eta * n / dim
    assert eta_n_over_d == pytest.approx(3.004, abs=1e-3)
    assert eta_n_over_d >= 2.0                      # ⛔ the criterion FIRES
    # ⛔ and the SELECTION-FREE pilot solves identically: G-B changed the number
    # of cells, not the cell, so this is the pilot's price, inherited.
    pled = _swap_ledger(_pilot())["clu_store"]
    assert solve_matched_ttt(int(pled["params"]), int(pled["state_floats"]),
                             dim) == (k, n)


# ==========================================================================
# 5. ⛔ RUN 3's LAUNCH BLOCKER B2 — run 2's script carries the passthrough
# ==========================================================================
def test_run2s_job_script_carries_the_PREREG_CONT_passthrough():
    """⛔ Runs 1-2 used THIS script and run 3 must use it too (a narrowed
    ``--arms`` in the ladder script would be a SECOND differing key)."""
    job = REPO / "scripts" / "csf3" / "job_gpu_cluformer.sh"
    t = job.read_text()
    assert 'PREREG_CONT="${PREREG_CONT:-}"' in t
    assert '[ -n "$PREREG_CONT" ] && EXTRA="$EXTRA --prereg-continuation ' \
           '$PREREG_CONT"' in t
    # RUN3-LAUNCH.md §1's own gate, executed here so it cannot regress
    assert t.count("prereg-continuation") == 1
    assert subprocess.run(["bash", "-n", str(job)]).returncode == 0


def test_the_EMITTED_command_line_carries_the_exemption_and_word_splits():
    """⚠ The zsh no-word-splitting trap (§7.37/7.45), checked not assumed."""
    out = subprocess.run(
        ["bash", "-c", 'EXTRA=""; [ -n "$PREREG_CONT" ] && '
                       'EXTRA="$EXTRA --prereg-continuation $PREREG_CONT"; '
                       'for w in $EXTRA; do echo "$w"; done'],
        capture_output=True, text=True,
        env={"PREREG_CONT": "journal=/tmp/j.json flag=memory.erosion_partition "
                            "prereg=P.md sha256=deadbeef", "PATH": "/usr/bin:/bin"})
    assert out.stdout.split() == ["--prereg-continuation", "journal=/tmp/j.json",
                                  "flag=memory.erosion_partition", "prereg=P.md",
                                  "sha256=deadbeef"]


def test_the_tokens_the_script_emits_are_ACCEPTED_by_the_REAL_CLI(tmp_path):
    """⭐ End-to-end: bash emits the argv, the REAL parser turns it into the spec,
    and a bad journal is refused BY THE EXEMPTION (not by argparse) — i.e. the
    tokens travelled the whole way."""
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    argv = ["--scale", "toy", "--stage", "s1", "--out", str(tmp_path),
            "--quick", "--arms", "none",
            "--prereg-continuation", f"journal={tmp_path}/absent.json",
            "flag=memory.erosion_partition", "prereg=P.md"]
    with pytest.raises((ContinuationExemptionError, SystemExit)) as e:
        EXP.main(argv)
    assert "absent.json" in str(e.value)


# ==========================================================================
# 6. ⛔ THE LEDGER'S REMEDY TEXT — a wrong remedy is worse than none
# ==========================================================================
def test_the_budget_error_names_levers_that_MOVE_BYTES():
    from chlu.eval.byte_ledger import StateByteBudgetError, build_byte_ledger

    pcfg = _pilot()
    led = _swap_ledger(pcfg)
    with pytest.raises(StateByteBudgetError) as e:
        build_byte_ledger(pcfg, led, ("clu_store",), budget=CEILING,
                          enforce=True, verbose=False)
    msg = " ".join(str(e.value).split())            # de-wrapped, one line
    # ⭐ the levers that DO move bytes are named as the remedy
    assert "store_layers" in msg and "min_atoms_base" in msg
    assert "1,380,864 B = 0.658x" in msg            # ...with the ratified target
    # ⛔ and the byte-INERT knobs are named ONLY to warn that they are inert
    assert "NOT with `capacity` or `atoms_per_item`" in msg
    assert "move ZERO BYTES" in msg
    assert "8192" in msg                            # the floor that dominates
    # ⛔ the OLD, WRONG remedy must not survive anywhere in the message
    assert "Shrink the store (capacity" not in msg
