"""Route B — learned well placement, and the guards that decide whether it means anything.

⭐ The load-bearing tests here are the ones that **fail on purpose**, because this spoke's
task file names them as what would prove it wrong:

- ``test_label_free_gate_FIRES_on_the_leaky_variant`` — ⛔ an assertion that has never
  fired is not an assertion.  The label-free gate is shown to **convict** the declared
  weakly-supervised objective, which is the only thing that licenses believing it when it
  acquits the label-free one.
- ``test_byte_ledger_ABORTS_when_placement_would_renegotiate_the_budget`` — learned
  placement must cost what inherited placement costs; the ledger raises rather than
  reports.
- ``test_cache_coverage_gate_names_the_seed_and_the_shortfall_and_DIES`` — the measured
  asymmetry (seed 0 holds 480,000 rows, seeds 1 and 2 hold 9,288) must fail at **startup**,
  ⛔ never on the GPU after the scheduler has allocated it.
- ``test_sharpness_is_UNDEFINED_not_zero_on_a_one_valued_factor`` — Montero's corner pins
  ``shape`` to 1 of 4 and has already produced one wrong headline in this programme.
- ``test_collapse_is_reported_as_collapse_not_as_a_win`` — sharpness can rise by emptying
  the wells; occupancy travels with every sharpness number.
- ``test_resume_is_BITWISE_identical_to_the_uninterrupted_run`` — the gate
  ``pilot-checkpoint-resume`` passed.  ⛔ A resume path that has never survived an actual
  interruption is not a resume path.
- ``test_split_is_prefix_stable_so_arm_keys_MUST_come_from_fold_in`` — the measured JAX
  landmine, re-asserted here because route B derives two learned arms' keys.

Everything runs on synthetic tensors; the real-data path is exercised by the verification
run reported in ``.claude/outputs/gen-wells-route-b.md``.

⛔ Wells are integer-indexed here as everywhere else.
"""

import json

import equinox as eqx
import jax
import numpy as np
import pytest

from chlu.config import GenWellsRouteBConfig, gen_wells_route_b_config, get_default_config
from chlu.core.gen_landscape import arm_key
from chlu.core.gen_store import random_grouping, well_channels_from_labels
from chlu.core.gen_wells import (
    ARMS,
    FLOOR_ARM,
    LABEL_FREE_ARMS,
    CheckpointCfg,
    LabelLeakError,
    PlacementState,
    address_sharpness,
    assert_byte_ledger,
    assert_cache_coverage,
    assert_label_free,
    assert_partition,
    collapse_verdict,
    config_fingerprint,
    dmatched_random_placement,
    harden_to_histogram,
    latest_checkpoint,
    list_checkpoints,
    placement_loss,
    placement_margin,
    save_checkpoint,
    sharpness_table,
    soft_memberships,
    train_placement,
    weak_supervision_term,
)

D, K, M, N, F = 48, 4, 6, 400, 3


def _reference_labels(seed=0):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, K, D)
    lab[:K] = np.arange(K)          # ⛔ no empty wells: that is a degenerate partition
    return np.asarray(lab, np.int64)


def _sizes(lab):
    return [int(x) for x in np.bincount(np.asarray(lab), minlength=K)]


def _state(key=0):
    k = jax.random.PRNGKey(key)
    return PlacementState(0.01 * jax.random.normal(k, (D, K)),
                          0.1 * jax.random.normal(k, (K, M, D)))


def _xy(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(N, D)).astype(np.float32),
            rng.integers(0, 4, (N, F)).astype(np.int64))


# ---------------------------------------------------------------------------
# 1. ⭐⭐ The D-matched floor and the hardening that keeps the arms comparable
# ---------------------------------------------------------------------------


def test_the_floor_is_D_matched_exactly_not_approximately():
    """⛔ A floor whose well sizes differ from the arm's is not a floor: a margin over it
    would be attributable to the size profile rather than to placement."""
    ref = _reference_labels()
    floor = dmatched_random_placement(ref, 1)
    assert _sizes(floor) == _sizes(ref)
    # ...and it really is a different assignment, not a no-op relabelling.
    assert not np.array_equal(floor, ref)


def test_hardening_realises_the_TARGET_histogram_exactly():
    """The learned arm is constrained to the reference histogram so that it, the floor and
    the inherited partition differ in exactly ONE thing: which coordinate sits where."""
    ref = _reference_labels()
    rng = np.random.default_rng(3)
    A = rng.random((D, K))
    A /= A.sum(axis=1, keepdims=True)
    lab = harden_to_histogram(A, _sizes(ref))
    assert _sizes(lab) == _sizes(ref)
    assert_partition(lab, D, K)


def test_hardening_is_deterministic_and_a_pure_function_of_its_inputs():
    rng = np.random.default_rng(4)
    A = rng.random((D, K))
    A /= A.sum(axis=1, keepdims=True)
    sizes = _sizes(_reference_labels())
    assert np.array_equal(harden_to_histogram(A, sizes), harden_to_histogram(A, sizes))


def test_hardening_REFUSES_a_histogram_that_does_not_partition_D():
    """⛔ A drifting histogram silently renegotiates the byte budget."""
    A = np.full((D, K), 1.0 / K)
    with pytest.raises(ValueError, match="partition"):
        harden_to_histogram(A, [1] * K)


def test_an_empty_well_is_REPORTED_never_silently_merged():
    lab = np.zeros(D, np.int64)
    with pytest.raises(AssertionError, match="EMPTY well"):
        assert_partition(lab, D, K)


def test_a_confident_membership_hardens_to_its_own_argmax():
    """Positive control for the size-constrained greedy: when the targets already match the
    argmax counts, the constraint costs nothing and the result IS the argmax."""
    ref = _reference_labels()
    A = np.full((D, K), 0.01)
    A[np.arange(D), ref] = 10.0
    A /= A.sum(axis=1, keepdims=True)
    assert np.array_equal(harden_to_histogram(A, _sizes(ref)), ref)


# ---------------------------------------------------------------------------
# 2. ⛔ The byte ledger — it ABORTS, it does not report
# ---------------------------------------------------------------------------


def test_byte_ledger_is_invariant_to_the_partition_at_fixed_K_and_M():
    """``Σ_k M·d_k·4 = M·D·4`` for ANY partition of D — which is exactly what lets a
    byte-matched placement comparison exist at all."""
    ref = _reference_labels()
    a = assert_byte_ledger(ref, M, M * D * 4, k=K, dim=D, rank=16, arm="ref")
    b = assert_byte_ledger(dmatched_random_placement(ref, 2), M, M * D * 4,
                           k=K, dim=D, rank=16, arm="floor")
    assert a["store_bytes"] == b["store_bytes"] == M * D * 4
    assert a["coupling"]["coupling_bytes"] == b["coupling"]["coupling_bytes"]
    assert a["coupling"]["matches_closed_form"] and b["coupling"]["matches_closed_form"]


def test_byte_ledger_ABORTS_when_placement_would_renegotiate_the_budget():
    """⛔ The named falsifier: learned placement must cost what inherited placement costs."""
    ref = _reference_labels()
    with pytest.raises(AssertionError, match="BYTE LEDGER MISMATCH"):
        assert_byte_ledger(ref, M, M * D * 4 + 4, k=K, dim=D, arm="learned")


def test_byte_ledger_can_be_measured_without_enforcing_but_says_so():
    ref = _reference_labels()
    led = assert_byte_ledger(ref, M, M * D * 4 + 4, k=K, dim=D, arm="x", enforce=False)
    assert led["ok"] is False and led["matches_expected"] is False


# ---------------------------------------------------------------------------
# 3. ⛔ The cache-coverage gate — it must die at STARTUP, naming the seed
# ---------------------------------------------------------------------------


def test_cache_coverage_gate_names_the_seed_and_the_shortfall_and_DIES():
    """⚠ Measured: seed 0 holds 480,000 rows (``scope="full"``), seeds 1 and 2 hold 9,288
    (``scope="used"``).  ⛔ A run on seed 1 that asks for more would discover it on the GPU,
    after the scheduler had already allocated it."""
    cache = np.arange(0, 9288, dtype=np.int64)
    needed = np.arange(0, 32768, dtype=np.int64)
    with pytest.raises(AssertionError) as e:
        assert_cache_coverage(cache, needed, seed=1, scope="used")
    msg = str(e.value)
    assert "SEED 1" in msg and "9288" in msg and "32768" in msg
    assert "scope='full'" in msg          # ⭐ it tells you what to do about it


def test_cache_coverage_PASSES_on_the_full_scope_seed():
    cache = np.arange(0, 480_000, dtype=np.int64)
    rep = assert_cache_coverage(cache, np.arange(0, 32768), seed=0, scope="full")
    assert rep["covered"] and rep["n_shortfall"] == 0
    assert rep["exercised_untested_path"] is False


def test_a_deliberate_reencode_is_reported_as_an_EXERCISED_UNTESTED_PATH():
    """⚠ If the untested path IS taken deliberately, it is said so — ⛔ not reported as a
    normal run."""
    rep = assert_cache_coverage(np.arange(0, 100), np.arange(0, 200), seed=1,
                                scope="used", allow_reencode=True)
    assert rep["exercised_untested_path"] is True
    assert rep["n_shortfall"] == 100


# ---------------------------------------------------------------------------
# 4. ⭐⭐ The label-free gate, and the leaky variant that proves it has teeth
# ---------------------------------------------------------------------------


def _clean(st, x, tau):
    return placement_loss(st, x, tau, lambda_occ=0.1, lambda_fact=1.0)[0]


def _leaky(st, x, y, tau):
    return _clean(st, x, tau) + 1.0 * weak_supervision_term(st, x, y, tau)


def test_the_label_free_objective_is_bit_identical_under_a_label_permutation():
    X, Y = _xy()
    rep = assert_label_free(_clean, _state(), X[:96], Y[:96], tau=1.0,
                            arm="learned_label_free")
    assert rep["verdict"] == "LABEL-FREE"
    assert rep["abs_loss_delta"] == 0.0 and rep["max_abs_grad_delta"] == 0.0
    assert rep["structural_clean"] is True


def test_label_free_gate_FIRES_on_the_leaky_variant():
    """⛔⭐ THE load-bearing test.  An assertion that has never fired is not an assertion:
    the gate must CONVICT the declared weakly-supervised objective."""
    X, Y = _xy()
    rep = assert_label_free(_leaky, _state(), X[:96], Y[:96], tau=1.0,
                            arm="learned_weak_sup", expect_label_free=False)
    assert rep["verdict"] == "LEAKY-AS-DECLARED"
    assert rep["bit_identical_under_label_permutation"] is False
    assert rep["max_abs_grad_delta"] > 0.0


def test_the_gate_RAISES_when_a_label_free_arm_is_actually_leaky():
    X, Y = _xy()
    with pytest.raises(LabelLeakError, match="declared LABEL-FREE but is not"):
        assert_label_free(_leaky, _state(), X[:96], Y[:96], tau=1.0,
                          arm="learned_label_free", expect_label_free=True)


def test_the_gate_RAISES_when_the_leaky_variant_does_not_move_it():
    """⛔ If the check cannot fire on the thing built to make it fire, it proves nothing
    about the label-free arm either — fix the check, do not proceed."""
    X, Y = _xy()
    with pytest.raises(AssertionError, match="CHECK-HAS-NO-TEETH|did not fire"):
        assert_label_free(lambda st, x, y, tau: _clean(st, x, tau),
                          _state(), X[:96], Y[:96], tau=1.0,
                          arm="fake_leaky", expect_label_free=False)


def test_the_structural_leg_catches_a_labels_argument_even_if_it_is_unused():
    """A future edit that adds ``labels`` to the objective is caught by signature, not by
    waiting for a numerical accident."""
    X, Y = _xy()
    with pytest.raises(AssertionError):
        assert_label_free(lambda st, x, labels, tau: _clean(st, x, tau),
                          _state(), X[:96], Y[:96], tau=1.0,
                          arm="learned_label_free", expect_label_free=True)


def test_the_declared_arms_are_pinned_so_neither_can_be_swapped_in_later():
    assert FLOOR_ARM == "random_dmatched" and FLOOR_ARM in ARMS
    assert LABEL_FREE_ARMS == ("learned_label_free",)
    assert "learned_weak_sup" in ARMS and "learned_weak_sup" not in LABEL_FREE_ARMS


# ---------------------------------------------------------------------------
# 5. ⭐⭐ The metric — ``n_values_realised`` on every row, and the degenerate case
# ---------------------------------------------------------------------------


def _codes(seed=0, n=N):
    return np.random.default_rng(seed).integers(0, M, (n, K))


def test_sharpness_is_UNDEFINED_not_zero_on_a_one_valued_factor():
    """⛔⛔ Montero's corner pins ``shape`` to **1 of 4** values and yields a
    confident-looking zero — that has produced a wrong headline in this programme once
    already."""
    _, Y = _xy()
    Y = Y.copy()
    Y[:, 2] = 1                                    # one value realised
    sh = address_sharpness(_codes(), Y, ["a", "b", "pinned"], [4, 4, 4],
                           n_modes_per_well=[M] * K)
    d = sh["per_factor"]["pinned"]
    assert d["degenerate"] is True
    assert d["max_sharpness"] is None and d["mean_sharpness"] is None
    assert d["n_values_realised"] == 1
    for row in d["per_well"]:
        assert row["sharpness"] is None            # ⛔ never 0.0, ⛔ never 1.0


def test_every_per_factor_row_carries_n_values_realised():
    """⛔ A per-factor number without ``n_values_realised`` is void."""
    _, Y = _xy()
    sh = address_sharpness(_codes(), Y, ["a", "b", "c"], [4, 4, 4],
                           n_modes_per_well=[M] * K)
    for d in sh["per_factor"].values():
        assert "n_values_realised" in d
        for row in d["per_well"]:
            assert "n_values_realised" in row and "cardinality" in row


def test_the_markdown_table_prints_undefined_not_a_number_for_a_degenerate_factor():
    _, Y = _xy()
    Y = Y.copy()
    Y[:, 2] = 3
    sh = address_sharpness(_codes(), Y, ["a", "b", "pinned"], [4, 4, 4],
                           n_modes_per_well=[M] * K)
    table = sharpness_table({"arm": sh}, ["a", "b", "pinned"])
    assert "**undefined**" in table
    assert "| `pinned` | 1 |" in table


def test_random_codes_score_near_zero_and_a_perfect_code_scores_one():
    """The two ends of the scale, so a middling number means something.

    ⚠ The perfect-code leg is measured at **negligible smoothing**, and that is the point
    of the leg rather than a convenience.  At the shipped ``smoothing = 1.0`` a perfect
    code scores **0.883**, not 1.0: Laplace smoothing adds ``M·V`` pseudo-counts (here 16)
    against ``N = 400`` real ones, which is a ~4 % dilution of the joint and shows up
    directly in the conditional entropy.  ⛔ That is worth knowing before anyone reads a
    sharpness of 0.88 as "not quite a delta" — at this ``N`` it IS a delta.
    """
    _, Y = _xy()
    rnd = address_sharpness(_codes(), Y, ["a", "b", "c"], [4, 4, 4],
                            n_modes_per_well=[M] * K)
    assert rnd["mean_max_sharpness"] < 0.15
    # a code that IS the factor: one mode per value, in every well.
    perfect = np.repeat(Y[:, [0]], K, axis=1)
    sharp = address_sharpness(perfect, Y, ["a", "b", "c"], [4, 4, 4],
                              n_modes_per_well=[4] * K, smoothing=1e-9)
    assert sharp["per_factor"]["a"]["max_sharpness"] > 0.99
    # ...and the dilution at the shipped smoothing is real, bounded and monotone.
    diluted = address_sharpness(perfect, Y, ["a", "b", "c"], [4, 4, 4],
                                n_modes_per_well=[4] * K, smoothing=1.0)
    assert 0.85 < diluted["per_factor"]["a"]["max_sharpness"] < 0.99


def test_the_marginal_only_baseline_travels_with_every_row():
    """⚠ ``S`` is normalised by ``log V``, not ``H(f)``, so a SKEWED marginal scores above
    zero while the code says nothing.  The baseline that makes that legible is reported."""
    rng = np.random.default_rng(5)
    Y = np.zeros((N, 1), np.int64)
    Y[: N // 10, 0] = 1                            # heavily skewed, 2 values realised
    sh = address_sharpness(rng.integers(0, M, (N, K)), Y, ["skew"], [2],
                           n_modes_per_well=[M] * K)
    d = sh["per_factor"]["skew"]
    row = d["per_well"][0]
    assert row["marginal_only_sharpness"] > 0.3    # the skew alone buys this
    assert abs(row["sharpness_above_marginal"]) < 0.15   # the code buys ~nothing


def test_a_margin_is_the_only_reporting_path_and_it_carries_the_floor():
    """⛔ A sharpness number without its random-placement floor beside it is void."""
    _, Y = _xy()
    arm = address_sharpness(_codes(0), Y, ["a", "b", "c"], [4, 4, 4],
                            n_modes_per_well=[M] * K)
    floor = address_sharpness(_codes(1), Y, ["a", "b", "c"], [4, 4, 4],
                              n_modes_per_well=[M] * K)
    mg = placement_margin(arm, floor, arm="learned_label_free")
    assert mg["floor_arm"] == FLOOR_ARM
    for v in mg["per_factor"].values():
        assert v["floor_max_sharpness"] is not None
        assert v["margin_max"] == pytest.approx(
            v["arm_max_sharpness"] - v["floor_max_sharpness"])


def test_a_degenerate_factor_yields_a_null_margin_not_a_zero_one():
    _, Y = _xy()
    Y = Y.copy()
    Y[:, 1] = 0
    a = address_sharpness(_codes(0), Y, ["a", "pin", "c"], [4, 4, 4],
                          n_modes_per_well=[M] * K)
    f = address_sharpness(_codes(1), Y, ["a", "pin", "c"], [4, 4, 4],
                          n_modes_per_well=[M] * K)
    mg = placement_margin(a, f)
    assert mg["per_factor"]["pin"]["margin_max"] is None


# ---------------------------------------------------------------------------
# 6. ⭐ The collapse guard — occupancy travels with every sharpness number
# ---------------------------------------------------------------------------


def test_collapse_is_reported_as_collapse_not_as_a_win():
    """⭐ Sharpness can rise by COLLAPSING wells: one mode eats the mass and the address
    book looks sharp because there is nothing left to choose between."""
    from chlu.core.gen_store import mode_usage

    healthy = mode_usage(np.random.default_rng(0).integers(0, M, (N, K)), M)
    collapsed = mode_usage(np.zeros((N, K), np.int64), M)
    v = collapse_verdict(collapsed, healthy, frac=0.80)
    assert v["collapsed"] is True
    assert "COLLAPSE" in v["verdict"]
    assert v["arm_mean_used_modes"] == 1.0


def test_occupancy_held_is_not_called_a_collapse():
    from chlu.core.gen_store import mode_usage

    a = mode_usage(np.random.default_rng(0).integers(0, M, (N, K)), M)
    b = mode_usage(np.random.default_rng(1).integers(0, M, (N, K)), M)
    assert collapse_verdict(a, b, frac=0.80)["collapsed"] is False


# ---------------------------------------------------------------------------
# 7. ⛔⛔ Checkpoint / resume — the gate ``pilot-checkpoint-resume`` passed
# ---------------------------------------------------------------------------


_TRAIN = dict(steps=30, batch=64, lr=1e-2, tau_start=1.0, tau_end=0.3,
              lambda_occ=0.1, lambda_fact=1.0, arm="t")


def _fp():
    return config_fingerprint({"steps": _TRAIN["steps"], "lr": _TRAIN["lr"]})


def test_resume_is_BITWISE_identical_to_the_uninterrupted_run(tmp_path):
    """⭐⭐ The whole point.  ⛔ A resume path that has never survived an interruption is
    not a resume path.

    The guarantee comes from the batch for step ``i`` being ``fold_in(base_key, i)`` — a
    pure function of ``(base_key, i)`` and ⛔ **not** of a key carried through the loop.
    """
    X, _ = _xy()
    bk = jax.random.PRNGKey(7)
    full, hist_full, _ = train_placement(_state(), X, base_key=bk, **_TRAIN,
                                         log=lambda *_: None)

    ck = CheckpointCfg(tmp_path / "k", every_seconds=1e9, every_steps=5, keep_last=99)
    train_placement(_state(), X, base_key=bk, **_TRAIN, ckpt=ck, config_fp=_fp(),
                    log=lambda *_: None)
    # simulate the kill: everything after step 15 never happened.
    for s in list_checkpoints(tmp_path / "k", "t"):
        if s > 15:
            (tmp_path / "k" / f"ckpt_t_step{s:08d}.json").unlink()
            (tmp_path / "k" / f"ckpt_t_step{s:08d}.eqx").unlink()
    assert latest_checkpoint(tmp_path / "k", "t") == 15

    got, hist_got, info = train_placement(_state(), X, base_key=bk, **_TRAIN, ckpt=ck,
                                          config_fp=_fp(), resume=True,
                                          log=lambda *_: None)
    assert info["resumed_from"] == 15 and info["steps_run"] == 15

    def leaves(s):
        return [np.asarray(t) for t in
                jax.tree_util.tree_leaves(eqx.filter(s, eqx.is_inexact_array))]

    differing = [i for i, (p, q) in enumerate(zip(leaves(full), leaves(got)))
                 if not np.array_equal(p, q)]
    assert differing == [], f"{len(differing)} leaves differ — the resume is NOT bitwise"
    assert hist_full == hist_got


def test_resume_REFUSES_a_checkpoint_from_a_different_config(tmp_path):
    """⛔ A resumed leg must be the SAME leg."""
    X, _ = _xy()
    ck = CheckpointCfg(tmp_path / "k", every_seconds=1e9, every_steps=10, keep_last=9)
    train_placement(_state(), X, base_key=jax.random.PRNGKey(7), **_TRAIN, ckpt=ck,
                    config_fp=_fp(), log=lambda *_: None)
    with pytest.raises(SystemExit, match="DIFFERENT config"):
        train_placement(_state(), X, base_key=jax.random.PRNGKey(7), **_TRAIN, ckpt=ck,
                        config_fp=config_fingerprint({"steps": 999}), resume=True,
                        log=lambda *_: None)


def test_only_the_last_N_good_checkpoints_are_kept(tmp_path):
    X, _ = _xy()
    ck = CheckpointCfg(tmp_path / "k", every_seconds=1e9, every_steps=5, keep_last=3)
    train_placement(_state(), X, base_key=jax.random.PRNGKey(7), **_TRAIN, ckpt=ck,
                    config_fp=_fp(), log=lambda *_: None)
    steps = list_checkpoints(tmp_path / "k", "t")
    assert len(steps) == 3 and steps == sorted(steps)
    assert steps[-1] == _TRAIN["steps"]


def test_a_TORN_checkpoint_write_is_ignored_and_the_last_good_one_survives(tmp_path):
    """⭐ The ``.json`` is written second and is the commit marker, so a crash *during* a
    checkpoint write cannot corrupt the last good one."""
    X, _ = _xy()
    ck = CheckpointCfg(tmp_path / "k", every_seconds=1e9, every_steps=10, keep_last=9)
    train_placement(_state(), X, base_key=jax.random.PRNGKey(7), **_TRAIN, ckpt=ck,
                    config_fp=_fp(), log=lambda *_: None)
    good = latest_checkpoint(tmp_path / "k", "t")
    (tmp_path / "k" / "ckpt_t_step00000099.eqx").write_bytes(b"torn")   # no .json
    assert 99 not in list_checkpoints(tmp_path / "k", "t")
    assert latest_checkpoint(tmp_path / "k", "t") == good


def test_the_checkpoint_write_is_atomic_and_leaves_no_tmp_behind(tmp_path):
    X, _ = _xy()
    ck = CheckpointCfg(tmp_path / "k", every_seconds=1e9, every_steps=10, keep_last=9)
    train_placement(_state(), X, base_key=jax.random.PRNGKey(7), **_TRAIN, ckpt=ck,
                    config_fp=_fp(), log=lambda *_: None)
    assert list((tmp_path / "k").glob("*.tmp")) == []


def test_the_resume_path_is_INSTRUMENTED(tmp_path, capsys):
    """⭐ It must print what it recovered and from which step."""
    X, _ = _xy()
    ck = CheckpointCfg(tmp_path / "k", every_seconds=1e9, every_steps=10, keep_last=9)
    train_placement(_state(), X, base_key=jax.random.PRNGKey(7), **_TRAIN, ckpt=ck,
                    config_fp=_fp(), log=lambda *_: None)
    lines = []
    train_placement(_state(), X, base_key=jax.random.PRNGKey(7), **_TRAIN, ckpt=ck,
                    config_fp=_fp(), resume=True, log=lines.append)
    joined = "\n".join(str(x) for x in lines)
    assert "[resume]" in joined and "RECOVERED" in joined and "step" in joined


# ---------------------------------------------------------------------------
# 8. ⚠ The measured JAX landmine — two learned arms, two keys
# ---------------------------------------------------------------------------


def test_split_is_prefix_stable_so_arm_keys_MUST_come_from_fold_in():
    """⚠ Under ``jax_threefry_partitionable`` ``split(k,2)[1]`` **is** ``split(k,3)[1]``,
    so "add one more arm to the split" hands two arms the SAME key and reports their
    agreement as a finding.  ⭐ Positive control: assert the property, then show ``fold_in``
    does not have it."""
    k = jax.random.PRNGKey(0)
    assert np.array_equal(np.asarray(jax.random.split(k, 2)[1]),
                          np.asarray(jax.random.split(k, 3)[1]))
    a = arm_key(k, "learned_label_free")
    b = arm_key(k, "learned_weak_sup")
    assert not np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# 9. The objective's own sanity, and the config block
# ---------------------------------------------------------------------------


def test_the_objective_rewards_multimodality_and_penalises_dependence():
    """The two named terms must move in the declared directions, or the objective is
    decoration."""
    X, _ = _xy()
    st = _state()
    _, parts = placement_loss(st, X[:128], 1.0, lambda_occ=0.1, lambda_fact=1.0)
    assert float(parts["mean_occupancy_entropy_nats"]) > 0.0
    assert float(parts["mean_pairwise_mi_nats"]) >= 0.0
    # a collapsed occupancy has LOWER entropy, so a larger L_occ.
    collapsed = PlacementState(st.logits, st.modes * 0.0 + 1e3)
    _, cparts = placement_loss(collapsed, X[:128], 1.0, lambda_occ=0.1, lambda_fact=1.0)
    assert float(cparts["mean_occupancy_entropy_nats"]) <= float(
        parts["mean_occupancy_entropy_nats"]) + 1e-6


def test_soft_memberships_are_row_stochastic():
    """The relaxation is a relaxation OF A PARTITION, not of an arbitrary mask."""
    a = np.asarray(soft_memberships(_state().logits))
    assert a.shape == (D, K)
    assert np.allclose(a.sum(axis=1), 1.0, atol=1e-6)
    assert (a >= 0).all()


def test_training_is_deterministic_given_the_base_key():
    X, _ = _xy()
    a, ha, _ = train_placement(_state(), X, base_key=jax.random.PRNGKey(3), **_TRAIN,
                               log=lambda *_: None)
    b, hb, _ = train_placement(_state(), X, base_key=jax.random.PRNGKey(3), **_TRAIN,
                               log=lambda *_: None)
    assert ha == hb


def test_the_config_block_is_an_ACCESSOR_not_a_CHLUConfig_field():
    """⛔ The merge rule for ``config.py``'s tail is 'take the peer's whole file and
    re-apply the block by its BEGIN/END markers', which only holds if the block has zero
    house wiring sites.  ⚠ A presence check that assumes FIELDS would report a false loss
    here — it has, twice."""
    import dataclasses

    cfg = get_default_config()
    assert "gen_wells_route_b" not in {f.name for f in dataclasses.fields(type(cfg))}
    blk = gen_wells_route_b_config(cfg)
    assert isinstance(blk, GenWellsRouteBConfig)
    assert gen_wells_route_b_config(cfg) is blk            # memoised
    assert gen_wells_route_b_config(cfg, placement_steps=11).placement_steps == 11


def test_the_config_pins_the_winner_decoder_not_the_baseline_that_failed_the_gate():
    """⛔ ``GenBridgeConfig.dec_recipe`` still defaults to ``"baseline"`` — a run through it
    looks entirely normal in the logs and is VOID."""
    c = gen_wells_route_b_config(get_default_config())
    assert c.dec_recipe == "v1_capacity_schedule"
    assert c.assert_dec_recipe is True
    assert get_default_config().gen_bridge.dec_recipe == "baseline"   # the trap, still set


def test_the_floor_arm_is_first_in_the_declared_arm_order():
    """⭐ A floor measured after the thing it is the floor for is a floor that can be
    talked about."""
    assert list(gen_wells_route_b_config(get_default_config()).arms)[0] == FLOOR_ARM


def test_quick_does_not_relax_the_byte_ledger_or_the_label_free_gate():
    """⛔ The ledger is a PRE-REGISTERED number and a smoke that shrinks it proves nothing;
    a label leak is not less of a leak in a smoke."""
    from chlu.experiments.exp_gen_wells_route_b import apply_quick

    cfg = get_default_config()
    c = apply_quick(cfg) and gen_wells_route_b_config(cfg)
    assert c.assert_byte_ledger is True and c.assert_label_free is True
    assert c.placement_steps == c.quick_placement_steps


def test_quick_and_scored_together_are_REFUSED():
    """⛔ A quick number is never reported."""
    from chlu.experiments.exp_gen_wells_route_b import main

    with pytest.raises(SystemExit, match="quick"):
        main(["--quick", "--scored"])


# ---------------------------------------------------------------------------
# 10. ⭐ Two regressions, each of which shipped and was caught by a real run
# ---------------------------------------------------------------------------


def test_the_weak_sup_term_survives_JIT():
    """⛔ REGRESSION.  ``weak_supervision_term`` inferred its one-hot width with
    ``y.max()``, which reads a **traced** array and raises inside a jitted step.

    ⚠ The failure mode is nasty: the un-jitted label-free gate passes happily a few lines
    above, so the log shows the weakly-supervised arm being *validated* and then dying.
    ⭐ Supplying static ``cardinalities`` is what makes it jittable, and this test is the
    thing that would have caught it.
    """
    X, Y = _xy()
    cards = [int(Y[:, f].max()) + 1 for f in range(Y.shape[1])]

    @eqx.filter_jit
    def step(st, xb, yb, tau):
        return weak_supervision_term(st, xb, yb, tau, cardinalities=cards)

    v = float(step(_state(), X[:64], Y[:64], 1.0))
    assert np.isfinite(v)
    # ⛔ and the un-jitted inference path must still agree with the static one.
    # ⚠ ABSOLUTE tolerance, deliberately: on an untrained state this term sits at ~5e-6,
    # where float32 accumulation order (which XLA is free to change under jit) moves the
    # last digits and a RELATIVE tolerance is meaningless.  The claim under test is "the
    # jitted and un-jitted paths compute the same quantity", not "bitwise equality".
    v2 = float(weak_supervision_term(_state(), X[:64], Y[:64], 1.0))
    assert v == pytest.approx(v2, abs=1e-6)


def test_the_placement_logits_actually_MOVE_under_the_objective():
    """⛔⭐ REGRESSION, and the most dangerous one in this spoke.

    The relaxation carries ``K·M·D`` mode floats against ``D·K`` placement logits.  At a
    shared Adam rate the modes absorb the gradient and the placement does not move **at
    all**: the first smoke of this harness hardened with **0 of 2048 coordinates moved** —
    which reads exactly like the pre-registered finding *"placement was never the lever"*
    and was in fact an optimisation artefact.

    ⛔ A null must not be manufacturable by a learning rate.  This test fails if the
    placement is frozen, so the null can only ever be earned.
    """
    X, _ = _xy()
    ref = _reference_labels()
    logits = np.zeros((D, K), np.float32)
    logits[np.arange(D), ref] = 1.0
    st0 = PlacementState(logits, 0.1 * jax.random.normal(jax.random.PRNGKey(2), (K, M, D)))

    moved = {}
    for mult in (1.0, 30.0):
        out, _, info = train_placement(
            st0, X, steps=60, batch=64, lr=3e-3, tau_start=1.0, tau_end=0.25,
            lambda_occ=0.1, lambda_fact=1.0, base_key=jax.random.PRNGKey(5),
            arm="t", logit_lr_mult=mult, log=lambda *_: None)
        lab = harden_to_histogram(np.asarray(soft_memberships(out.logits)), _sizes(ref))
        moved[mult] = int((lab != ref).sum())
        assert info["logit_lr_mult"] == mult

    assert moved[30.0] > 0, (
        "the placement did not move a single coordinate — the relaxation is frozen and "
        "any 'no margin over the floor' result would be an optimisation artefact, not a "
        "finding")
    assert moved[30.0] > moved[1.0]
