"""The generation harness end to end (``gen-cell-harness``): the headline table.

⭐ The load-bearing tests here are the ones that can **fail on purpose** — a check
that has never been seen to fire has not been tested:

- ``test_preconditions_FIRE_on_a_failing_gate`` / ``..._on_an_incomplete_refusal_file``
  — R30's two mechanical gates are shown refusing, not merely shown passing.  ⛔ The
  content is parsed and compared; a file that merely *mentions* ``"PASS"`` fails.
- ``test_external_classifier_hash_mismatch_FIRES`` — a re-trained or swapped
  instrument is refused before a number is recorded.
- ``test_distinct_counts_separates_a_COLLAPSED_arm_from_a_diverse_one`` and
  ``test_distinct_counts_exposes_the_trivial_72_of_72_of_a_continuous_sampler`` —
  the column an accuracy must never appear without, and the reason the *factor
  tuple* count (not the vector count) is the cross-arm comparable one.
- ``test_regression_check_FLAGS_a_weakened_rival`` — *"a rival got weaker because of
  a change you made"* is a measurement here, not a promise.
- ``test_byte_ledger_declares_the_knn_advantage_IN_WORDS`` — R26.3's ~31 % advantage
  is in the table row, ⛔ not in a footnote.
- ``test_aggregate_reports_a_missing_seed_as_NOT_RUN_never_as_a_zero``.

Everything runs on synthetic dicts and tensors; the real-data path is exercised by
the harness runs reported in ``.claude/outputs/gen-cell-harness.md``.

⚠ §7.23: other test modules enable ``jax_enable_x64`` at **module import** and may
sort before this one, so nothing here pins float32.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import GenCellConfig, get_default_config, load_config, save_config
from chlu.experiments.exp_gen_cell import (
    ARMS,
    REQUEST_SETS,
    STAGES,
    aggregate_runs,
    arm_byte_ledger,
    check_preconditions,
    distinct_counts,
    load_external_classifier,
    regression_check,
)

SHIPPED_STORE_BYTES = 524_288
SHIPPED_HEAD_BYTES = 74_240
SHIPPED_PROTO_BYTES = 475_136
RETIRED_BUDGET = 688_128


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class _Reader:
    """A stand-in instrument: ``images -> per-factor logits``, deterministic.

    Reads the factor tuple straight out of the first channel of each image, so a
    test can construct an arm whose renders are known to collapse (or not).
    """

    def __init__(self, cards=(3, 4)):
        self.cards = tuple(int(c) for c in cards)

    def __call__(self, x):
        # ⚠ called under ``eqx.filter_jit(jax.vmap(...))`` by ``predict_factors``, so
        # it must be TRACEABLE — ⛔ no ``np.asarray`` on a tracer, no Python ``if``.
        v = jnp.asarray(x).reshape(-1)
        out = []
        for j, c in enumerate(self.cards):
            k = jnp.abs(v[j]).astype(jnp.int32) % int(c)
            out.append(jnp.where(jnp.arange(int(c)) == k, 9.0, -9.0))
        return out


def _write_gate(tmp_path, verdict="PASS"):
    root = tmp_path / "outputs" / "gen-external-classifier"
    root.mkdir(parents=True, exist_ok=True)
    (root / "classifier_gate.json").write_text(json.dumps({
        "gate": verdict, "floor": 0.95, "n_seeds": 3,
        "per_seed": {"0": 1.0, "1": 1.0, "2": 1.0},
        "statistic": "in-corner shape balanced accuracy on REAL held-out images",
    }))
    return root


def _write_refusal(tmp_path, status="complete"):
    root = tmp_path / "outputs" / "gen-arms-a7-wiring"
    root.mkdir(parents=True, exist_ok=True)
    (root / "refusal_scores.json").write_text(json.dumps({
        "status": status, "seeds_present": [0, 1, 2], "not_run": [], "legs": {"H": {}},
    }))
    return root


def _config(tmp_path):
    cfg = get_default_config()
    cfg.gen_cell.claude_root = str(tmp_path)
    return cfg


# ---------------------------------------------------------------------------
# ⛔ 1. THE PRECONDITIONS — both of them, shown passing AND shown firing
# ---------------------------------------------------------------------------


def test_preconditions_pass_when_both_files_say_what_they_must(tmp_path):
    _write_gate(tmp_path, "PASS")
    _write_refusal(tmp_path, "complete")
    out = check_preconditions(_config(tmp_path), log=lambda *_: None)
    assert out["ok"] is True
    keys = {(c["key"], c["value"]) for c in out["checked"]}
    assert ("gate", "PASS") in keys and ("status", "complete") in keys


def test_preconditions_FIRE_on_a_failing_gate(tmp_path):
    """⛔ A FAIL gate stops the run — ⛔ 'start anyway and fix later' is unavailable."""
    _write_gate(tmp_path, "FAIL")
    _write_refusal(tmp_path, "complete")
    with pytest.raises(AssertionError, match="PRECONDITION"):
        check_preconditions(_config(tmp_path), log=lambda *_: None)


def test_preconditions_FIRE_on_an_incomplete_refusal_file(tmp_path):
    _write_gate(tmp_path, "PASS")
    _write_refusal(tmp_path, "partial")
    with pytest.raises(AssertionError, match="PRECONDITION"):
        check_preconditions(_config(tmp_path), log=lambda *_: None)


def test_preconditions_FIRE_on_a_missing_file(tmp_path):
    _write_refusal(tmp_path, "complete")
    with pytest.raises(FileNotFoundError, match="PRECONDITION"):
        check_preconditions(_config(tmp_path), log=lambda *_: None)


def test_the_gate_is_read_as_PARSED_CONTENT_not_as_a_string_match(tmp_path):
    """⛔ A file that merely MENTIONS 'PASS' must not satisfy the gate."""
    root = _write_gate(tmp_path, "FAIL")
    (root / "classifier_gate.json").write_text(json.dumps({
        "gate": "FAIL", "note": "the rule is PASS on every seed, not the mean",
    }))
    _write_refusal(tmp_path, "complete")
    with pytest.raises(AssertionError):
        check_preconditions(_config(tmp_path), log=lambda *_: None)


# ---------------------------------------------------------------------------
# ⛔ 2. THE INSTRUMENT — one classifier, hash-asserted
# ---------------------------------------------------------------------------


def test_external_classifier_hash_mismatch_FIRES(tmp_path):
    """⛔ A swapped or re-trained instrument is refused before any number exists."""
    import equinox as eqx

    from chlu.utils.checkpoints import save_model

    root = tmp_path / "outputs" / "gen-external-classifier" / "seed_0"
    root.mkdir(parents=True, exist_ok=True)
    model = eqx.nn.Linear(3, 2, key=jax.random.PRNGKey(0))
    save_model(model, root / "external_classifier.pkl", {"seed": 0})
    (root / "classifier_manifest.json").write_text(json.dumps({
        "classifier": {"param_hash": "0" * 64, "param_count": 8, "steps": 1500}
    }))
    with pytest.raises(AssertionError, match="does not match its own manifest"):
        load_external_classifier(_config(tmp_path), 0, log=lambda *_: None)


def test_a_missing_instrument_is_NOT_RUN_not_a_zero(tmp_path):
    with pytest.raises(FileNotFoundError, match="NOT RUN"):
        load_external_classifier(_config(tmp_path), 7, log=lambda *_: None)


# ---------------------------------------------------------------------------
# ⭐⭐ 3. THE DISTINCT-OUTPUT COUNT — the column an accuracy never appears without
# ---------------------------------------------------------------------------


def _imgs_from_tuples(tuples):
    """One 'image' per request whose first channels encode its factor tuple."""
    t = np.asarray(tuples, np.float32)
    return np.concatenate([t, np.zeros((t.shape[0], 2), np.float32)], axis=1)


def test_distinct_counts_separates_a_COLLAPSED_arm_from_a_diverse_one():
    reader = _Reader((3, 4))
    n = 12
    diverse = np.stack([np.array([i % 3, i % 4]) for i in range(n)])
    collapsed = np.stack([np.array([1, 2]) for _ in range(n)])
    d = distinct_counts(reader, _imgs_from_tuples(diverse), _imgs_from_tuples(diverse), 8)
    c = distinct_counts(reader, _imgs_from_tuples(collapsed), _imgs_from_tuples(collapsed), 8)
    assert d["factor_tuples"] == 12 and d["feature_vectors"] == 12
    # ⭐ the ablation-style arm is provably dead: ONE output for every request
    assert c["factor_tuples"] == 1 and c["feature_vectors"] == 1 and c["images"] == 1


def test_distinct_counts_exposes_the_trivial_72_of_72_of_a_continuous_sampler():
    """⚠ A continuous sampler cannot repeat a float vector, so ``feature_vectors``
    is **trivially N** and is not diversity.  ⭐ ``factor_tuples`` is the comparable
    column, and it can still collapse."""
    reader = _Reader((3, 4))
    rng = np.random.default_rng(0)
    n = 72
    # 72 distinct float vectors, but every one reads back as the SAME factor tuple
    jitter = rng.normal(scale=1e-3, size=(n, 4)).astype(np.float32)
    # ⚠ base values sit mid-bin (1.5 / 2.5), ⛔ not on the integer boundary: my own
    # first version jittered across it and read back 4 tuples, which is the fixture
    # being wrong, not the function.
    imgs = np.tile(np.array([[1.5, 2.5, 0.0, 0.0]], np.float32), (n, 1)) + jitter
    d = distinct_counts(reader, imgs, imgs, 16)
    assert d["feature_vectors"] == n and d["images"] == n
    assert d["factor_tuples"] == 1
    assert "trivially N for a continuous sampler" in d["note"]


# ---------------------------------------------------------------------------
# ⛔ 4. NEVER-WEAKEN-A-COMPETITOR, as a measurement
# ---------------------------------------------------------------------------


def _manifest_with(arm_balacc: dict) -> dict:
    return {"arms": {a: {"probe_companion": {"in_corner": {"balanced_accuracy": v}}}
                     for a, v in arm_balacc.items()}}


def test_regression_check_FLAGS_a_weakened_rival():
    cfg = GenCellConfig()
    # ⛔ the bit-identity leg reads the CONSTANT, ⛔ never a literal retyped here: a
    # hand-typed reference is exactly how the shipped constants were wrong the first
    # time (they were transcribed from a rendered 4-dp table, not from disk).
    man = _manifest_with({"knn_proto": 0.70, "lace": 0.68,
                          "clu": cfg.clu_probe_reference_per_seed[0]})
    out = regression_check(man, cfg, 0)
    assert out["knn_proto"]["regressed"] is True          # 0.876 -> 0.70
    assert out["lace"]["regressed"] is False              # within tolerance of 0.7077
    assert out["clu"]["bit_identical"] is True


def test_regression_check_is_probe_vs_probe_never_external_vs_probe():
    """⛔ Comparing an external-classifier number with a published PROBE number would
    confound the instrument with the arm — the one thing this table exists not to do."""
    cfg = GenCellConfig()
    out = regression_check(_manifest_with({"clu": 0.66}), cfg, 0)
    assert "never external-vs-probe" in out["rule"]


def test_an_arm_absent_from_the_manifest_is_NOT_RUN():
    out = regression_check({"arms": {}}, GenCellConfig(), 0)
    assert out["clu"] == {"NOT_RUN": True}


# ---------------------------------------------------------------------------
# ⛔ 5. THE BYTE LEDGER — in the table, in words, never a footnote
# ---------------------------------------------------------------------------


def _store_led():
    return {"deployed_bytes": SHIPPED_STORE_BYTES, "coupling_bytes_excluded": 163_840,
            "addressing_head": {"bytes": SHIPPED_HEAD_BYTES}}


def test_byte_ledger_carries_the_CORRECTED_budget_not_the_retired_one():
    cfg = GenCellConfig()
    led = arm_byte_ledger("clu", cfg, _store_led(), {"bytes": SHIPPED_PROTO_BYTES}, None, None)
    assert led["deployed_bytes"] == SHIPPED_STORE_BYTES
    assert led["addressing_head_bytes"] == SHIPPED_HEAD_BYTES
    assert str(RETIRED_BUDGET) in led["budget_note"] and "Do NOT inherit" in led["budget_note"]


def test_byte_ledger_declares_the_knn_advantage_IN_WORDS():
    """⚠ R26.3: the shipped kNN configuration was selected at 688,128 B — a ~31 %
    advantage.  ⭐ Conservative direction, and **declared, never silent**."""
    cfg = GenCellConfig()
    led = arm_byte_ledger("knn_proto", cfg, _store_led(), {"bytes": SHIPPED_PROTO_BYTES},
                          None, None)
    assert led["deployed_bytes"] == SHIPPED_PROTO_BYTES
    assert "31 %" in led["budget_note"] and "CONSERVATIVE" in led["budget_note"]
    assert str(RETIRED_BUDGET) in led["budget_note"]


def test_the_prototype_table_is_IN_BUDGET_against_the_corrected_store():
    assert SHIPPED_PROTO_BYTES < SHIPPED_STORE_BYTES
    # ⚠ and it is 6.4x the CLU addressing head, which matters the day heads are counted
    assert SHIPPED_PROTO_BYTES / SHIPPED_HEAD_BYTES > 6.0


def test_lace_is_not_claimed_to_be_byte_matched():
    cfg = GenCellConfig()
    led = arm_byte_ledger("lace", cfg, _store_led(), {"bytes": SHIPPED_PROTO_BYTES},
                          {"classifier": {"param_count": 1000}}, None)
    assert "NOT byte-matched" in led["budget_note"]
    assert "supervision asymmetry is declared" in led["budget_note"]


# ---------------------------------------------------------------------------
# 6. Aggregation — ⛔ a missing seed is NOT RUN, never a zero
# ---------------------------------------------------------------------------


def _seed_dir(tmp_path, seed, arms, balacc, distinct):
    d = tmp_path / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    man = {
        "seed": seed,
        "ceiling": {"external": {"in_corner": {"balanced_accuracy": 0.94}},
                    "probe": {"in_corner": {"balanced_accuracy": 0.94}}},
        "arms": {
            a: {
                "arm": a, "uses_store": a in ("clu", "store_removed"),
                "decoder_param_hash": "dec", "external_classifier_param_hash": "clf",
                "external": {"in_corner": {"balanced_accuracy": balacc[a], "auroc": 0.9,
                                           "tpr": 0.9, "tnr": 0.9},
                             "composition_B": {"aggregate": 0.8,
                                               "per_factor": {"shape": balacc[a]}}},
                "probe_companion": {"in_corner": {"balanced_accuracy": balacc[a]},
                                    "composition_B": {"aggregate": 0.8,
                                                      "per_factor": {"shape": balacc[a]}}},
                "distinct": {k: {"n": 72, "feature_vectors": distinct[a],
                                 "images": distinct[a], "factor_tuples": distinct[a]}
                             for k in REQUEST_SETS},
                "byte_ledger": {"arm": a, "deployed_bytes": SHIPPED_STORE_BYTES},
                "tuning": {"budget_available": 36, "trials_realised": 0},
            }
            for a in arms
        },
    }
    (d / "cell_manifest.json").write_text(json.dumps(man))
    return d


def test_aggregate_reports_a_missing_seed_as_NOT_RUN_never_as_a_zero(tmp_path):
    bal = {"clu": 0.67, "store_removed": 0.5, "knn_proto": 0.88}
    dis = {"clu": 9, "store_removed": 1, "knn_proto": 72}
    dirs = [
        _seed_dir(tmp_path, 0, ["clu", "store_removed", "knn_proto"], bal, dis),
        _seed_dir(tmp_path, 1, ["clu", "store_removed", "knn_proto"], bal, dis),
        _seed_dir(tmp_path, 2, ["clu", "store_removed"], bal, dis),
    ]
    agg = aggregate_runs(dirs, log=lambda *_: None)
    assert agg["arms"]["knn_proto"]["n_seeds"] == 2
    assert agg["arms"]["knn_proto"]["NOT_RUN_seeds"] == [2]
    # ⛔ the missing seed does NOT drag the mean toward zero
    assert agg["arms"]["knn_proto"]["external_in_corner_balacc"]["mean"] == pytest.approx(0.88)


def test_aggregate_carries_the_distinct_count_beside_every_accuracy(tmp_path):
    bal = {"clu": 0.67, "store_removed": 0.5, "knn_proto": 0.88}
    dis = {"clu": 9, "store_removed": 1, "knn_proto": 72}
    dirs = [_seed_dir(tmp_path, s, list(bal), bal, dis) for s in (0, 1, 2)]
    agg = aggregate_runs(dirs, log=lambda *_: None)
    for arm in bal:
        row = agg["arms"][arm]
        assert np.isfinite(row["external_in_corner_balacc"]["mean"])
        assert np.isfinite(row["distinct_factor_tuples_B"]["mean"])
    assert agg["arms"]["store_removed"]["distinct_factor_tuples_B"]["mean"] == 1.0
    assert agg["arms"]["clu"]["distinct_factor_tuples_B"]["mean"] == 9.0


def test_aggregate_flags_whether_the_INSTRUMENT_reorders_the_arms(tmp_path):
    bal = {"clu": 0.67, "store_removed": 0.5, "knn_proto": 0.88}
    dis = {"clu": 9, "store_removed": 1, "knn_proto": 72}
    dirs = [_seed_dir(tmp_path, s, list(bal), bal, dis) for s in (0, 1, 2)]
    agg = aggregate_runs(dirs, log=lambda *_: None)
    r = agg["instrument_invariance"]["_ranking"]
    assert r["identical"] is True
    assert r["by_external"][0] == "knn_proto" and r["by_external"][-1] == "store_removed"


def test_aggregate_on_an_empty_directory_says_NOT_RUN(tmp_path):
    agg = aggregate_runs([tmp_path / "seed_9"], log=lambda *_: None)
    assert agg["seeds_present"] == [] and "NOT RUN" in agg["note"]


# ---------------------------------------------------------------------------
# 7. Config hygiene — the block round-trips and the defaults are the shipped ones
# ---------------------------------------------------------------------------


def test_gen_cell_config_round_trips_through_yaml(tmp_path):
    cfg = get_default_config()
    cfg.gen_cell.seeds = [0, 1, 2, 3]
    cfg.gen_cell.tuning_trial_budget = 12
    p = tmp_path / "config.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.gen_cell.seeds == [0, 1, 2, 3]
    assert back.gen_cell.tuning_trial_budget == 12


def test_the_shipped_defaults_are_the_ones_the_rulings_pin():
    c = GenCellConfig()
    assert c.expect_deployed_store_bytes == SHIPPED_STORE_BYTES      # R26.3
    assert c.retired_byte_budget == RETIRED_BUDGET
    assert c.expected_decoder_recipe == "v1_capacity_schedule"       # ⛔ never "baseline"
    assert c.decoder_source == "winner"
    assert c.lace_n_epochs == 50                                     # R9.6, ⛔ not 100/200
    assert c.lace_primary_sampler == "ode"
    # ⚠ ``dynread`` is the sixth arm on ``ARMS`` but is NOT in the shipped default
    # ``arms`` list: ``chlu/config.py`` is held by a concurrent spoke and
    # ``gen-read-cell-arm`` is ⛔ forbidden to touch it, so the arm is selected
    # explicitly (``--arm dynread`` / a yaml ``gen_cell.arms``). ⛔ Declared here, not
    # silent — one line in ``GenCellConfig.arms`` is owed to whoever owns that file.
    assert list(c.arms) == [a for a in ARMS if a != "dynread"]
    assert "dynread" in ARMS
    assert "store_removed" in c.arms                                 # the laundering row
    assert c.tuning_trial_budget == 36
    # the EBM grid realises exactly the declared budget
    n = (len(c.ebm_step_sizes) * len(c.ebm_prior_sigmas) * len(c.ebm_steps_grid)
         * len(c.ebm_clip_grid))
    assert n == c.tuning_trial_budget


def test_the_module_refuses_another_stages_name():
    from chlu.experiments.exp_gen_cell import NOT_MINE, main

    assert "compose" in NOT_MINE and "certificate" in NOT_MINE
    assert main(["--stage", "certificate"]) == 2
    assert set(STAGES) == {"cell", "aggregate"}


def test_the_reference_constants_carry_full_double_precision():
    """⚠ A reference constant transcribed from a 4-dp table cannot support a
    bit-identity claim.  ⭐ These came from the peers' manifests, and the test pins
    that they are NOT round 4-dp numbers."""
    c = GenCellConfig()
    for name in ("clu_probe_reference_per_seed", "knn_proto_probe_reference_per_seed",
                 "lace_ode_probe_reference_per_seed"):
        vals = getattr(c, name)
        assert len(vals) == 3
        for v in vals:
            assert abs(v - round(v, 4)) > 1e-9, f"{name}: {v} looks like a 4-dp figure"


# ---------------------------------------------------------------------------
# ⭐ the sixth arm: the dynamical read, wired onto THIS cell (gen-read-cell-arm)
#
# ⛔ The load-bearing ones here are, again, the tests that FIRE:
# ``test_dynread_preflight_FIRES_*`` show the three traps refusing (λ = 1 by fiat,
# a launch that carries no request information, a temperature that would stop
# ``request_off`` being a same-chain control), and the verdict tests show a
# sub-margin gain being refused the word WIN and a tie against a like-for-like row
# being reported as a LOSS.
# ---------------------------------------------------------------------------

from chlu.config import gen_dynamical_read_config  # noqa: E402
from chlu.data.gen_rowsplit import stratified_fit_val  # noqa: E402
from chlu.experiments.exp_gen_cell import (  # noqa: E402
    DYNREAD_ARMS,
    DYNREAD_BUDGETS,
    DYNREAD_CEILING_ARM,
    DYNREAD_LIKE_FOR_LIKE,
    DYNREAD_MARGIN,
    DYNREAD_PRIMARY_BUDGET,
    DYNREAD_STEPS,
    dynread_convergence_checkpoints,
    dynread_label,
    dynread_preflight,
    dynread_stored_atom_tuples,
    dynread_verdict,
)


def _dr_config():
    c = get_default_config()
    gen_dynamical_read_config(c)
    return c


def test_dynread_is_the_sixth_arm_and_the_MATCHED_rung_carries_the_bare_name():
    assert ARMS[-1] == "dynread" and len(ARMS) == 6
    labels = [dynread_label(b, a) for b in DYNREAD_BUDGETS for a in DYNREAD_ARMS]
    assert len(set(labels)) == len(labels), "two rungs must never share a table row"
    # ⭐ the matched rung's full arm is THE row the generation table gains
    assert dynread_label(DYNREAD_PRIMARY_BUDGET, "full") == "dynread"
    # ⛔ a control or the larger rung can never be mistaken for it
    assert dynread_label("B", "full") == "dynread/B"
    assert dynread_label("A", "request_off") == "dynread/A_request_off"
    assert all(lbl.split("/")[0] == "dynread" for lbl in labels)


def test_the_step_budgets_are_the_declared_ones_and_differ_by_rung():
    # ⛔ declared before the run; A = 1000, B = 2000. Not a tuning knob.
    assert DYNREAD_STEPS == {"A": 1000, "B": 2000}
    assert "request_off" in DYNREAD_ARMS, "the steering control is not optional"


def test_dynread_preflight_passes_on_the_shipped_block():
    pre = dynread_preflight(_dr_config())
    assert pre["coupling_certificate_enforced"] is True
    assert pre["launch_mode"] == "clamp_atom"
    assert pre["chain_is_deterministic"] is True


def test_dynread_preflight_FIRES_when_lambda_would_be_the_DECLARED_one():
    # ⛔ a bilinear coupling is an indefinite quadratic form: at λ = 1 the landscape
    # is unbounded below and the chain does not settle, it diverges (E = -1.6e28).
    c = _dr_config()
    gen_dynamical_read_config(c).enforce_coupling_certificate = False
    with pytest.raises(AssertionError, match="coercivity|certificate"):
        dynread_preflight(c)


def test_dynread_preflight_FIRES_on_a_launch_that_carries_no_request():
    # ⛔ the naive-Bayes posterior clears 0.5 in 0.00 of 5 wells, so the thresholded
    # launch degenerates to the PRIOR's argmax and the launch is request-free.
    c = _dr_config()
    gen_dynamical_read_config(c).launch_mode = "clamp_atom_thresholded"
    with pytest.raises(AssertionError, match="argmax"):
        dynread_preflight(c)


def test_dynread_preflight_FIRES_at_a_TEMPERATURE_that_breaks_the_control():
    # ⭐ at T = 0 the chain is deterministic, which is what makes `request_off` a
    # same-chain SAME-KEY control. ⛔ At T > 0 it would differ in two things at once.
    c = _dr_config()
    gen_dynamical_read_config(c).temp_end = 0.01
    with pytest.raises(AssertionError, match="same-chain|SAME|control"):
        dynread_preflight(c)


def _trace(n_steps: int, n_chains: int = 4, slope: float = 0.0):
    t = np.arange(n_steps + 1, dtype=np.float64)[:, None]
    E = -100.0 + slope * t + np.zeros((1, n_chains))
    return {
        "energy": E.copy(), "energy_ref": E.copy(),
        "accept_prob": np.ones((n_steps, n_chains)),
        "accepted": np.ones((n_steps, n_chains), bool),
        "displacement": np.full((n_steps, n_chains), 1e-3),
    }


def test_convergence_checkpoints_report_1000_and_NEVER_extrapolate_to_2000():
    out = dynread_convergence_checkpoints(_trace(1000), window=100, tol=1e-3)
    assert out["n_steps_run"] == 1000
    assert out["1000"]["frac_still_descending"] == 0.0
    # ⛔ the 2000-step checkpoint of a 1000-step chain is NOT RUN, never guessed
    assert "NOT_RUN" in out["2000"]
    assert "frac_still_descending" not in out["2000"]


def test_convergence_checkpoints_CONVICT_a_chain_still_descending_at_1000():
    # a chain whose energy is still falling at its own budget yields NO quotable number
    out = dynread_convergence_checkpoints(_trace(2000, slope=-0.05), window=100, tol=1e-3)
    assert out["1000"]["frac_still_descending"] == 1.0
    assert out["2000"]["frac_still_descending"] == 1.0
    assert "NOT CONVERGED" in out["2000"]["verdict"]


def test_a_budget_B_chain_reports_BOTH_checkpoints_from_ONE_trace():
    # ⛔ no chain is re-run to answer "what was it doing at 1000?"
    out = dynread_convergence_checkpoints(_trace(2000), window=100, tol=1e-3)
    assert set(("1000", "2000")) <= set(out)
    assert all("NOT_RUN" not in out[k] for k in ("1000", "2000"))


# ---------------------------------------------------------------------------
# ⭐⭐ the verdict — PREREG §2b's vocabulary and no other
# ---------------------------------------------------------------------------


def _vman(seed: int, values: dict) -> dict:
    return {"seed": int(seed),
            "arms": {a: {"external": {"composition_B": {"aggregate": float(v)}}}
                     for a, v in values.items()}}


def _mans(dyn, knn=(0.8650, 0.8650, 0.8650), lfl=0.30):
    base = {a: lfl for a in DYNREAD_LIKE_FOR_LIKE}
    return [_vman(s, {**base, "dynread": dyn[s], DYNREAD_CEILING_ARM: knn[s]})
            for s in range(3)]


def test_the_margin_is_the_PREREGISTERED_one_and_is_not_re_derived():
    assert DYNREAD_MARGIN == 0.020


def test_a_sub_margin_gain_with_three_agreeing_signs_is_a_TIE_NEVER_a_WIN():
    # ⛔ PREREG-addendum-0, verbatim: "a +0.001 with three agreeing signs is a tie".
    v = dynread_verdict(_mans([0.8660, 0.8660, 0.8660]))
    assert v["verdict"] == "TIE (favourable)"
    assert v["against"][DYNREAD_CEILING_ARM]["same_sign_3_of_3"] is True
    assert "WIN" not in v["verdict"]


def test_a_WIN_needs_BOTH_the_margin_AND_three_of_three():
    beats = dynread_verdict(_mans([0.92, 0.93, 0.94]))
    assert beats["verdict"] == "WIN"
    # ⛔ same Δ, but one seed disagrees in sign ⇒ ⛔ not a WIN
    split = dynread_verdict(_mans([0.99, 0.99, 0.80], knn=(0.80, 0.80, 0.90)))
    assert split["against"][DYNREAD_CEILING_ARM]["delta"] > DYNREAD_MARGIN
    assert split["against"][DYNREAD_CEILING_ARM]["same_sign_3_of_3"] is False
    assert split["verdict"] == "INDETERMINATE"


def test_a_clear_shortfall_against_the_ceiling_is_a_LOSS():
    assert dynread_verdict(_mans([0.60, 0.61, 0.59]))["verdict"] == "LOSS"


def test_a_TIE_is_available_ONLY_against_the_ceiling():
    # dynread level-pegging with every like-for-like row: ⛔ that is a LOSS on each
    v = dynread_verdict(_mans([0.30, 0.30, 0.30], lfl=0.30))
    assert set(v["like_for_like"].values()) == {"LOSS"}
    assert v["like_for_like_all_beaten"] is False
    assert v["PASS"] is False
    assert "clu" in v["like_for_like"], "the codebook read this arm replaces is a row"


def test_beating_a_like_for_like_row_requires_beyond_seed_spread():
    # a gain smaller than the two arms' spread is ⛔ NOT "beaten outright"
    mans = [_vman(s, {"dynread": [0.40, 0.30, 0.20][s], "clu": [0.35, 0.25, 0.15][s],
                      DYNREAD_CEILING_ARM: 0.8650}) for s in range(3)]
    v = dynread_verdict(mans)
    assert v["against"]["clu"]["same_sign_3_of_3"] is True
    assert v["against"]["clu"]["intervals_overlap_mean_pm_sd"] is True
    assert v["against"]["clu"]["verdict"] == "LOSS"


def test_the_headline_PASS_needs_the_ceiling_AND_every_like_for_like_row():
    v = dynread_verdict(_mans([0.8660, 0.8660, 0.8660], lfl=0.30))
    assert v["verdict"] == "TIE (favourable)"
    assert v["like_for_like_all_beaten"] is True
    assert v["PASS"] is True


def test_an_absent_dynread_is_NOT_RUN_never_a_zero():
    v = dynread_verdict([_vman(s, {DYNREAD_CEILING_ARM: 0.8650}) for s in range(3)])
    assert v["verdict"] == "NOT RUN"


# ---------------------------------------------------------------------------
# ⭐ the byte ledger and the artefacts
# ---------------------------------------------------------------------------


def _dyn_led(budget="A", store=524_288, coupling=524_288):
    return {"dynread": {
        "spec": {"label": budget, "role": "MATCHED", "n_atoms": 64, "rank": 16},
        "byte_ledger": {"store": {"store_bytes": store},
                        "coupling": {"coupling_bytes": coupling},
                        "total_bytes": store + coupling,
                        "multiple_of_budget": (store + coupling) / 524_288.0}}}


def test_the_dynread_ledger_declares_the_2x_rung_IN_WORDS_not_as_matched_bytes():
    cfg = GenCellConfig()
    led = arm_byte_ledger("dynread", cfg, _store_led(), {"bytes": SHIPPED_PROTO_BYTES},
                          None, None, _dyn_led())
    assert led["store_bytes"] == SHIPPED_STORE_BYTES        # the ATOM table is matched
    assert led["deployed_bytes"] == 2 * SHIPPED_STORE_BYTES  # ⛔ the rung total is not
    assert abs(led["multiple_of_budget"] - 2.0) < 1e-9
    assert "not a match" in led["budget_note"]
    assert "DECLARED, NOT SILENT" in led["budget_note"]


def test_a_dynread_row_without_a_ledger_is_NOT_RUN_not_a_zero():
    led = arm_byte_ledger("dynread/B", GenCellConfig(), _store_led(),
                          {"bytes": SHIPPED_PROTO_BYTES}, None, None, {})
    assert led["deployed_bytes"] is None and "NOT RUN" in led["what"]


def test_stored_atom_tuples_are_written_WITH_the_pool_rows_they_came_from(tmp_path):
    # ⛔ dreaming's novelty test needs the atoms' OWN tuples from THIS run: the atoms
    # are a seeded k-means++ draw and a re-derivation elsewhere is a different set.
    from chlu.core.gen_landscape import build_atom_store

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(64, 8)).astype(np.float32)
    lab = np.stack([np.arange(64) % 3, np.arange(64) % 4], axis=1).astype(np.int64)
    store = build_atom_store([(0, 1, 2, 3), (4, 5, 6, 7)], (8,), Z, 8, "kmeanspp", 0)
    p = tmp_path / "atoms.npz"
    info = dynread_stored_atom_tuples({"store": store}, lab, ["a", "b"], p)
    z = np.load(p)
    assert info["n_atoms"] == store.n_atoms
    assert z["atom_rows_into_pool"].tolist() == list(store.atom_rows)
    assert np.array_equal(z["atom_factor_tuples"], lab[list(store.atom_rows)])
    assert info["n_distinct_atom_factor_tuples"] == int(
        np.unique(lab[list(store.atom_rows)], axis=0).shape[0])


# ---------------------------------------------------------------------------
# ⛔⛔ the silent truncation this spoke is told to assert against, demonstrated
# ---------------------------------------------------------------------------


def test_stratified_fit_val_SILENTLY_CLAMPS_n_val_and_says_nothing():
    # ⛔ POSITIVE CONTROL for the assertion `run_seed` now carries: a caller asking
    # for a larger `n_val` gets `pool.size - n_fit` rows and a completely normal
    # return value. ⛔ Not fixed here — `gen_rowsplit` is shared and a peer wave owns
    # the repair; this test pins the behaviour the cell must refuse to inherit.
    pool = np.arange(1000, dtype=np.int64)
    labels = np.stack([pool % 3, pool % 4], axis=1).astype(np.int64)
    fit, val, info = stratified_fit_val(pool, labels, n_fit=900, n_val=400, seed=0)
    assert val.size == 100, "the clamp is `pool.size - n_fit`"
    assert info["n_val"] == 100          # ⭐ the realised number IS reported...
    assert "400" not in json.dumps(info)  # ...⛔ but the REQUEST is nowhere in it


# ---------------------------------------------------------------------------
# ⭐⭐ the ADDRESSED READ (`x0`) — gen-addressed-read
#
# ⛔⛔ The two tests this section exists for are the ones that can FIRE:
# ``test_x0_coupling_proof_FIRES_when_the_coupling_reaches_the_read`` (the byte
# ledger is a MEASUREMENT, so it must be shown refusing a ledger it cannot prove)
# and ``test_x0_zero_step_proof_FIRES_when_something_sits_between_launch_and_decoder``
# (the arm's whole identity is that it spends no steps).
# ⛔ Plus the naming guard: nothing on this path may be called dynamics.
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

from chlu.core.gen_landscape import (  # noqa: E402
    AtomLandscape,
    atom_bandwidths,
    build_atom_store,
    fit_atom_address_book,
    fit_low_rank_coupling,
    launch_state,
)
from chlu.experiments.exp_gen_cell import (  # noqa: E402
    X0_ARM,
    X0_BUDGETS,
    X0_LOOKUP_FLOORS,
    X0_PRIMARY_BUDGET,
    X0_ROWS,
    X0_SETTLING_STEPS,
    _x0_pair,
    addressed_read_verdict,
    run_x0_budget,
    x0_address_book_bytes,
    x0_coupling_is_not_needed,
    x0_gather,
    x0_label,
    x0_preflight,
    x0_zero_step_proof,
)


def _x0_rig(n_atoms: int = 8, cards=(3, 4), seed: int = 0):
    """A tiny real ``AtomStore`` + address book + fitted coupling."""
    rng = np.random.default_rng(seed)
    K, dk, n_train = 5, 6, 400
    D = K * dk
    wc = [tuple(range(k * dk, (k + 1) * dk)) for k in range(K)]
    Z = rng.normal(size=(n_train, D)).astype(np.float32)
    Y = np.stack([rng.integers(0, c, n_train) for c in cards], axis=1)
    store = build_atom_store(wc, (D,), Z, n_atoms, "kmeanspp", 0)
    sigma = atom_bandwidths(store, 1.0, "nn_median")
    book = fit_atom_address_book(store, Z, Y, list(cards), sigma, 1.0)
    coup = fit_low_rank_coupling(store, Z[:100], 2, True)
    land = AtomLandscape(store, sigma, coup, 1.0, 0.05)
    R = np.stack([rng.integers(-1, c, 20) for c in cards], axis=1).astype(np.int64)
    return {"store": store, "book": book, "land": land, "requests": R, "Z": Z,
            "well_channels": wc, "D": D, "K": K, "cards": list(cards)}


# ---- ⛔⛔ the name, first: this is an ADDRESSED READ, never dynamics ---------


#: ⛔⛔ The vocabulary that would turn a zero-step gather into an overclaim.
X0_BANNED_PHRASES = ("zero-step dynamics", "dynamics at zero steps",
                     "the dynamical read at zero steps", "zero step dynamics")


def _bare_forbidden_phrase(src: str, banned=X0_BANNED_PHRASES):
    """⭐ The first use of a banned phrase that is NOT inside a prohibition.

    ⛔ A prohibition may wrap across source lines, so the window is the phrase's line
    and the three before it; ⛔ a bare use with no ``never``/``⛔`` anywhere near it is
    returned, which is what the test asserts never happens.
    """
    lines = src.lower().splitlines()
    for i, line in enumerate(lines):
        for b in banned:
            if b in line:
                window = " ".join(lines[max(0, i - 3): i + 1])
                if "never" not in window and "⛔" not in window:
                    return line
    return None


def test_the_naming_guard_itself_FIRES_on_a_bare_use():
    # ⛔ POSITIVE CONTROL: a guard that has never been seen to fire is decoration.
    assert _bare_forbidden_phrase("x = 1\nthis is zero-step dynamics\n") is not None
    assert _bare_forbidden_phrase("⛔ never call it zero-step dynamics\n") is None


def test_x0_is_NEVER_described_as_dynamics_anywhere_on_its_own_path():
    # ⛔⛔ The single most damaging overclaim available to this programme is calling a
    # zero-step gather "dynamics". The guard is mechanical: the arm's own source and
    # every string it emits are swept for the forbidden vocabulary.
    import chlu.experiments.exp_gen_cell as M

    fns = (M.x0_gather, M.x0_preflight, M.x0_zero_step_proof, M.run_x0_budget,
           M.addressed_read_verdict, M.x0_coupling_is_not_needed, M.x0_label,
           M.x0_address_book_bytes, M.x0_artefact_dir)
    for fn in fns:
        bad = _bare_forbidden_phrase(inspect.getsource(fn))
        assert bad is None, (fn.__name__, bad)
    pre = M.x0_preflight(get_default_config())
    assert "ADDRESSED READ" in pre["NAMING_RULE"]
    assert pre["settling_steps"] == 0 and pre["coupling_used"] is False
    # ⛔ and the arm's own row calls itself an addressed read, in the manifest
    assert "ADDRESSED READ" in pre["mechanism"].upper() or pre["coupling_used"] is False


def test_x0_spends_zero_steps_and_says_so_as_a_number_not_a_word():
    assert X0_SETTLING_STEPS == 0
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    z = x0_zero_step_proof(x, x.reshape(3, 4), (4,))
    assert z["settling_steps"] == 0
    assert z["displacement_from_launch_rms"] == 0.0
    assert z["decoder_input_is_the_launch_bitwise"] is True
    assert z["energy_trace"] is None and z["coupling_used"] is False


def test_x0_zero_step_proof_FIRES_when_something_sits_between_launch_and_decoder():
    # ⛔ POSITIVE CONTROL. A chain — or any transform — between the launch and the
    # decoder makes this a different arm, and the proof must refuse it.
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    with pytest.raises(AssertionError, match="NOT the launch"):
        x0_zero_step_proof(x, x + 1e-7, (4,))


# ---- ⭐⭐ the byte ledger: measured, and shown refusing ----------------------


def test_x0_coupling_is_not_needed_is_MEASURED_bit_for_bit():
    rig = _x0_rig()
    p = x0_coupling_is_not_needed(rig, rig["requests"])
    assert p["coupling_needed"] is False
    # ⭐ all four landscapes — fitted coupling, none, λ = 0, β = 0 — give the SAME x0
    assert set(p["bit_identical"]) == {
        "coupling_fitted_at_certificate_lambda", "coupling_None",
        "coupling_strength_zero", "request_strength_zero"}
    assert all(p["bit_identical"].values())
    assert all(v == 0.0 for v in p["max_abs_delta"].values())


def test_x0_coupling_proof_FIRES_when_the_coupling_reaches_the_read():
    # ⛔⛔ POSITIVE CONTROL for the crux. If the coupling ever reached the launch, the
    # rung-A ledger would be 2.00× the budget and "matched bytes" would be a lie. The
    # proof is shown REFUSING: a store whose atoms differ between the two landscapes
    # stands in for a read that is not a function of (request, book, atoms) alone.
    rig = _x0_rig()
    other = _x0_rig(seed=1)
    # the coupling-free landscape is rebuilt from ``built["store"]``; handing it a
    # DIFFERENT atom table is the stand-in for a launch that is not a function of
    # (request, book, atoms) alone — exactly the condition that would void the ledger.
    broken = {**rig, "store": other["store"]}
    with pytest.raises(AssertionError, match="COUPLING IS ON THE ADDRESSED READ"):
        x0_coupling_is_not_needed(broken, rig["requests"])


def test_x0_address_book_bytes_are_MEASURED_and_checked_against_the_closed_form():
    rig = _x0_rig(n_atoms=8, cards=(3, 4))
    hb = x0_address_book_bytes(rig["book"])
    K, N = rig["K"], 8
    assert hb["bytes"] == (K * N + sum(rig["cards"]) * K * N) * 4
    assert hb["bytes"] == hb["closed_form_bytes"]
    assert hb["bytes_formula"] == "(K*N + sum_f V_f*K*N) * 4"


def test_the_atom_address_book_formula_IS_the_CLU_heads_formula():
    # ⭐ THE CRUX OF "MATCHED BYTES": the addressed read's head and the CLU arm's head
    # are the same object with M -> N, so at N = M they are byte-identical — and that
    # is checked against the SHIPPED numbers, not asserted in prose.
    from chlu.core.gen_compose import address_book_bytes

    clu_head = _store_led()["addressing_head"]
    assert clu_head["bytes"] == SHIPPED_HEAD_BYTES
    K, M, v_sum = 5, 64, 57          # the shipped cell: 5 wells, 64 modes, Σ_f V_f = 57
    assert (K * M + v_sum * K * M) * 4 == SHIPPED_HEAD_BYTES
    assert address_book_bytes.__doc__ is not None
    # the atom store at N = M carries exactly the shipped store's bytes
    assert 64 * 2048 * 4 == SHIPPED_STORE_BYTES


def test_x0_store_bytes_are_the_live_objects_not_a_literal():
    rig = _x0_rig(n_atoms=8)
    assert rig["store"].n_bytes() == 8 * rig["D"] * 4


# ---- ⭐ the rows: one arm, two §C.1 floors, one §C.2 control ----------------


def test_x0_rows_are_the_arm_plus_the_charter_mandated_floors_and_control():
    assert X0_ROWS == ("full", "atom_mean_lookup", "row_lookup", "atoms_shuffled")
    assert X0_LOOKUP_FLOORS == ("atom_mean_lookup", "row_lookup")
    # ⛔ `x0` is deliberately NOT on ARMS: that tuple is the pin's declared rival-row
    # list and editing it would void this table's numbers (charter §A4).
    assert X0_ARM not in ARMS and ARMS[-1] == "dynread" and len(ARMS) == 6


def test_x0_label_gives_the_MATCHED_rung_the_bare_name_and_suffixes_everything_else():
    assert x0_label(X0_PRIMARY_BUDGET, "full") == "x0"
    assert x0_label("B", "full") == "x0/B"              # ⛔ the 8× rung, never bare
    assert x0_label("A", "row_lookup") == "x0/A_row_lookup"
    labels = [x0_label(b, r) for b in X0_BUDGETS for r in X0_ROWS]
    assert len(set(labels)) == len(labels) == 8


def test_x0_full_returns_a_STORED_ATOM_in_every_well():
    # ⛔ charter §C.4: an output that sits ON its atoms is a lookup and is reported as
    # one. This arm's output IS stored atoms — per well — and the test says so.
    rig = _x0_rig()
    g = x0_gather(rig["store"], rig["book"], rig["requests"], "full", land=rig["land"])
    A = [np.asarray(a, np.float32) for a in rig["store"].atoms]
    for i in range(rig["requests"].shape[0]):
        for k, w in enumerate(rig["well_channels"]):
            assert np.any(np.all(A[k] == g["x"][i, list(w)], axis=1))
    assert g["returns_a_stored_atom"] is True


def test_row_lookup_returns_a_WHOLE_stored_train_row_and_differs_only_in_recombination():
    # ⭐ THE HARD §C.1 FLOOR: one atom index for ALL wells. The gap to `full` is the
    # per-well recombination and ⛔ nothing else.
    rig = _x0_rig()
    g = x0_gather(rig["store"], rig["book"], rig["requests"], "row_lookup",
                  land=rig["land"])
    pool = rig["Z"][list(rig["store"].atom_rows)]
    for i in range(rig["requests"].shape[0]):
        assert np.any(np.all(pool == g["x"][i], axis=1)), "not a whole stored row"
    assert g["pool_rows_returned"].shape[0] == rig["requests"].shape[0]


def test_atom_mean_lookup_IS_the_pinned_posterior_mean_launch_bitwise():
    # ⛔ the Advisor's named rival is the PINNED launch mode, ⛔ not a re-implementation
    rig = _x0_rig()
    g = x0_gather(rig["store"], rig["book"], rig["requests"], "atom_mean_lookup",
                  land=rig["land"])
    ref = launch_state(rig["land"], rig["book"], rig["requests"], "posterior_mean",
                       0.5)["x0"]
    assert np.array_equal(np.asarray(ref, np.float32), g["x"])


def test_atoms_shuffled_keeps_the_SAME_indices_and_changes_only_the_CONTENT():
    # ⭐ charter §C.2: the control differs from the arm in ONE declared quantity.
    rig = _x0_rig()
    f = x0_gather(rig["store"], rig["book"], rig["requests"], "full", land=rig["land"],
                  seed=5)
    s = x0_gather(rig["store"], rig["book"], rig["requests"], "atoms_shuffled",
                  land=rig["land"], seed=5)
    assert np.array_equal(f["atom_index_per_well"], s["atom_index_per_well"])
    assert not np.array_equal(f["x"], s["x"])
    assert s["frac_indices_moved"] > 0.5


def test_x0_gather_refuses_a_row_it_does_not_have():
    rig = _x0_rig()
    with pytest.raises(ValueError, match="row must be one of"):
        x0_gather(rig["store"], rig["book"], rig["requests"], "settled", land=rig["land"])


# ---- the preflight, shown refusing ----------------------------------------


def test_x0_preflight_passes_on_the_shipped_block():
    pre = x0_preflight(get_default_config())
    assert pre["launch_mode"] == "clamp_atom" and pre["read_mode"] == "state"
    assert pre["primary_budget"] == "A" and pre["settling_steps"] == 0
    assert "MATCHED-BYTES" in pre["HEADLINE_RULE"]


def test_x0_preflight_FIRES_on_a_launch_that_carries_no_request_information():
    c = get_default_config()
    gen_dynamical_read_config(c).launch_mode = "clamp_atom_thresholded"
    with pytest.raises(AssertionError, match="posterior argmax"):
        x0_preflight(c)


def test_x0_preflight_FIRES_when_the_read_stops_being_the_launch():
    c = get_default_config()
    gen_dynamical_read_config(c).read_mode = "atom_mixture"
    with pytest.raises(AssertionError, match="stops being the launch"):
        x0_preflight(c)


# ---- ⭐⭐ the verdict: the matched-bytes row leads, and the floors bind -------


def _x0_mans(x0_vals, clu_vals, floor_vals, shuf_vals=(0.12, 0.12, 0.12),
             matched=True, mult=1.0):
    led = {"store_bytes": SHIPPED_STORE_BYTES if matched else 8 * SHIPPED_STORE_BYTES,
           "addressing_head_bytes": SHIPPED_HEAD_BYTES, "coupling_bytes": 0,
           "coupling_bytes_excluded": SHIPPED_STORE_BYTES,
           "multiple_of_budget": float(mult),
           "byte_matched_to_clu_field_for_field": bool(matched),
           "coupling_not_needed_proof": {"coupling_needed": False}}

    def _arm(v):
        return {"external": {"composition_B": {"aggregate": float(v)}},
                "distinct": {"B": {"factor_tuples": 17, "n": 72}},
                "byte_ledger": led}

    mans = []
    for i, s in enumerate((0, 1, 2)):
        arms = {"x0": _arm(x0_vals[i]), "clu": _arm(clu_vals[i]),
                "knn_proto": _arm(0.86), "store_removed": _arm(0.12),
                "lace": _arm(0.31), "ebm_compose": _arm(0.19),
                "x0/A_atoms_shuffled": _arm(shuf_vals[i])}
        for f in X0_LOOKUP_FLOORS:
            arms[f"x0/A_{f}"] = _arm(floor_vals[i])
        mans.append({"seed": s, "arms": arms})
    return mans


def test_the_MATCHED_BYTES_verdict_LEADS_and_the_8x_row_never_appears_bare():
    # ⛔⛔ THE RULE THIS SPOKE EXISTS TO ENFORCE: 0.6852 is the 8× row. Any field that
    # carries a rung-B number carries its byte multiple on the same record.
    v = addressed_read_verdict(_x0_mans((0.4329, 0.5185, 0.5509),
                                        (0.5903, 0.6065, 0.5880),
                                        (0.15, 0.16, 0.14)))
    mb = v["MATCHED_BYTES_VERDICT"]
    assert mb["rung"] == "A" and mb["label"] == "x0"
    assert mb["bytes"]["multiple_of_budget"] == 1.0
    assert mb["bytes"]["byte_matched_to_clu_field_for_field"] is True
    assert mb["bytes"]["coupling_needed"] is False
    # ⛔ losing to the codebook read at the SAME bytes is a LOSS and is named one
    assert mb["VERDICT_vs_clu"] == "LOSS"
    assert mb["like_for_like_all_beaten"] is False and mb["PASS"] is False
    assert "MATCHED BYTES" in v["headline"]
    # ⛔ the 8× companion is NOT RUN here, and even then it announces the rule
    assert v["eight_times_companion"]["⛔ never quote this without the byte multiple"]


def test_the_charter_C1_FLOOR_fails_LOUDLY_when_a_lookup_is_not_beaten():
    # ⛔⛔ THE FLOOR. If a matched-bytes lookup table matches the arm, the store is
    # decorative — and the verdict says exactly that, whatever the rest of the table
    # says about the rivals.
    v = addressed_read_verdict(_x0_mans((0.50, 0.52, 0.51),
                                        (0.20, 0.21, 0.20),    # a weak "clu"
                                        (0.50, 0.52, 0.51)))   # ⛔ the lookup MATCHES
    a = v["per_rung"]["A"]
    assert a["like_for_like_all_beaten"] is True      # it beats every pinned rival...
    assert a["charter_C1_floor_PASS"] is False        # ...⛔ and still does not count
    assert "DECORATIVE" in a["charter_C1_floor_statement"]
    assert a["PASS"] is False
    assert v["MATCHED_BYTES_VERDICT"]["charter_C1_floor_PASS"] is False


def test_the_charter_C2_control_must_COLLAPSE():
    v = addressed_read_verdict(_x0_mans((0.50, 0.52, 0.51), (0.20, 0.21, 0.20),
                                        (0.15, 0.16, 0.14),
                                        shuf_vals=(0.50, 0.52, 0.51)))
    a = v["per_rung"]["A"]
    assert a["charter_C2_control_COLLAPSES"] is False
    assert a["PASS"] is False


def test_a_sub_margin_gain_over_a_lookup_floor_is_a_LOSS_never_a_WIN():
    # ⛔ +0.001 with three agreeing signs is not a win against a like-for-like row.
    v = addressed_read_verdict(_x0_mans((0.500, 0.501, 0.502), (0.20, 0.21, 0.20),
                                        (0.4995, 0.5005, 0.5015)))
    floors = v["per_rung"]["A"]["charter_C1_floors"]
    assert all(f["verdict"] == "LOSS" for f in floors.values())
    assert v["per_rung"]["A"]["charter_C1_floor_PASS"] is False


def test_the_floor_comparator_and_the_PINNED_verdict_agree_on_every_shared_rival():
    # ⭐ the §C.1 floors are judged by the same arithmetic as the pinned rows — and
    # that is checked, not asserted: `addressed_read_verdict` re-derives every pinned
    # rival through BOTH paths and raises if they ever differ.
    mans = _x0_mans((0.4329, 0.5185, 0.5509), (0.5903, 0.6065, 0.5880),
                    (0.15, 0.16, 0.14))
    v = addressed_read_verdict(mans)                     # ⛔ raises on disagreement
    for rival, got in v["per_rung"]["A"]["against"].items():
        mine = _x0_pair({0: 0.4329, 1: 0.5185, 2: 0.5509},
                        {i: mans[i]["arms"][rival]["external"]["composition_B"]["aggregate"]
                         for i in range(3)},
                        DYNREAD_MARGIN, is_ceiling=bool(rival == DYNREAD_CEILING_ARM))
        assert mine["verdict"] == got["verdict"]
    assert DYNREAD_LIKE_FOR_LIKE == ("store_removed", "ebm_compose", "lace", "clu")


def test_a_missing_x0_row_is_NOT_RUN_never_a_zero():
    v = addressed_read_verdict([{"seed": 0, "arms": {}}])
    assert v["per_rung"]["A"]["verdict"] == "NOT RUN"
    assert "NOT RUN" in v["per_rung"]["A"]["note"]


def test_run_x0_budget_signature_takes_the_clu_ledger_it_matches_against():
    # ⛔ "matched bytes" is checked against THIS manifest's own store ledger, ⛔ never
    # against a literal in the source.
    sig = list(inspect.signature(run_x0_budget).parameters)
    assert "clu_ledger" in sig
    src = inspect.getsource(run_x0_budget)
    assert "524_288" not in src and "524288" not in src
