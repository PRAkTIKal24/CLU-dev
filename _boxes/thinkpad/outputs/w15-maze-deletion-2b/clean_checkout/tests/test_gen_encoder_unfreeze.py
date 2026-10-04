"""Unfreezing the encoder with the store in the loop — ⛔ without opening a bypass.

⭐ **The load-bearing tests here are the ones that fail on purpose.**  Unfreezing
is exactly the change that can silently create an encoder → decoder channel,
because the decoder's loss now flows back into the encoder.  So:

- ``test_bypass_check_POSITIVE_CONTROL_fires_on_a_deliberately_bypassed_path`` —
  a path with a real additive feature leak is handed to
  :func:`chlu.core.gen_bridge.assert_no_feature_bypass`, which MUST raise, and
  the **collision leg** must be among the legs that fire.  A check that cannot
  fail proves nothing.
- ``test_collision_criterion_catches_R27s_named_soft_read_trap`` — the criterion
  itself (same codes ⇒ same decoder input) is applied to
  :func:`chlu.core.gen_store.soft_read`, the Hub's own named trap: identical
  arithmetic to the composed read, but it **takes the features**.
- §1 pins the **frozen path** bit-identically against fingerprints recorded on
  the unmodified tree at ``main @ 994ca9b`` *before* the additive block existed.

⛔ Everything runs on synthetic tensors and a toy image shape; the real-data path
is the local smoke reported in ``.claude/outputs/gen-encoder-unfreeze.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import (
    GenEncoderUnfreezeConfig,
    get_default_config,
    load_config,
    save_config,
)
from chlu.core.gen_bridge import (
    CODE_GRADS,
    GenEncoder,
    assert_no_feature_bypass,
    gather_from_codes_traced,
    joint_filter_spec,
    joint_loss_terms,
    make_decoder,
    param_hash,
    quantise_through_store,
    realised_channel_bits,
    resolve_decoder_cfg,
    store_codes_traced,
    train_joint,
)
from chlu.core.gen_store import (
    address_codes,
    assert_no_bypass,
    channel_correlation,
    correlation_grouping,
    init_store,
    read_from_codes,
    soft_read,
    store_bytes,
    well_channels_from_labels,
)

IN_SHAPE = (3, 16, 16)
#: ⭐ Recorded on the UNMODIFIED tree (``main @ 994ca9b``) by
#: ``.claude/scratch/gen-encoder-unfreeze/baseline_fingerprint.py`` before a
#: single line of the unfrozen block was written.  ⛔ If one of these moves, the
#: additive block was not additive and every phase-1 number downstream is void.
FROZEN_FINGERPRINT = {
    "feat_shape": [16, 4, 4],
    "enc_hash": "7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693",
    "dec_hash_v1": "40236b47ba4f53c913b5ac247af50abad16fd742d04c54fdacc3f588a34251df",
    "dec_hash_baseline": "457977e44d7b2a6ac30742fa5632aad8d31cfab67c71c61e24265fb50b38fea3",
    "store_hash": "a2a916f4a054a1d8574cbd87646d356b078d6906cd67cbb848353ac4da18dea1",
    "well_sizes": [8, 23, 225],
    "codes_sum": 84,
    "bits_per_read": 6.0,
    "store_bytes": 4096,
}


# ---------------------------------------------------------------------------
# Fixtures — one toy encoder/store/decoder triple, built the way the harness does
# ---------------------------------------------------------------------------


def _triple(k_wells=3, n_modes=4, n=32, seed=20260914):
    key = jax.random.PRNGKey(seed)
    ek, dk, sk = jax.random.split(key, 3)
    enc = GenEncoder(IN_SHAPE, (8, 16), 4, 2, 4, ek)
    rng = np.random.default_rng(0)
    imgs = rng.random((n,) + IN_SHAPE).astype(np.float32)
    Z = np.asarray(jax.vmap(enc)(jnp.asarray(imgs)))
    Zf = Z.reshape((Z.shape[0], -1))
    labels = correlation_grouping(channel_correlation(Zf), k_wells, method="auto")
    wc = well_channels_from_labels(labels, k_wells)
    store = init_store(Zf, wc, n_modes, enc.feat_shape, sk, seed=0)
    recipe = resolve_decoder_cfg(get_default_config().gen_bridge, "v1_capacity_schedule")
    dec = make_decoder(enc.feat_shape, IN_SHAPE, recipe, dk)
    return enc, store, dec, imgs, Zf


@pytest.fixture(scope="module")
def triple():
    return _triple()


# ---------------------------------------------------------------------------
# §1. ⛔ THE FROZEN PATH IS BIT-IDENTICAL BEFORE AND AFTER THE ADDITIVE BLOCK
# ---------------------------------------------------------------------------


def test_frozen_path_is_bit_identical_to_the_pre_change_fingerprint(triple):
    enc, store, dec, _imgs, Zf = triple
    got = {
        "feat_shape": [int(x) for x in enc.feat_shape],
        "enc_hash": param_hash(enc),
        "dec_hash_v1": param_hash(dec),
        "store_hash": param_hash(store),
        "well_sizes": [len(w) for w in store.well_channels],
        "codes_sum": int(address_codes(store, Zf).sum()),
        "bits_per_read": store.bits_per_read,
        "store_bytes": store_bytes(store)["store_bytes"],
    }
    for k, v in got.items():
        assert v == FROZEN_FINGERPRINT[k], (
            f"{k}: the FROZEN path moved ({v!r} != {FROZEN_FINGERPRINT[k]!r}). "
            "The unfrozen block must be additive; every downstream phase-1 number "
            "rests on the frozen path being unchanged."
        )


def test_the_baseline_decoder_recipe_still_builds_the_same_decoder(triple):
    enc, _store, _dec, _imgs, _Zf = triple
    dk = jax.random.split(jax.random.PRNGKey(20260914), 3)[1]
    base = make_decoder(
        enc.feat_shape, IN_SHAPE, resolve_decoder_cfg(get_default_config().gen_bridge), dk
    )
    assert param_hash(base) == FROZEN_FINGERPRINT["dec_hash_baseline"]


def test_the_shipped_read_still_passes_its_own_no_bypass_check(triple):
    _enc, store, _dec, _imgs, Zf = triple
    rep = assert_no_bypass(store, Zf)
    assert rep["signature_ok"] and rep["read_is_pure_codebook_gather"]
    assert rep["max_abs_delta_vs_independent_gather"] == 0.0


# ---------------------------------------------------------------------------
# §2. ⛔⭐ THE BYPASS CHECK — shown FAILING first, then passing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1e-3, 1e-1, 1.0])
def test_bypass_check_POSITIVE_CONTROL_fires_on_a_deliberately_bypassed_path(
    triple, alpha
):
    """⭐ The check must be able to FAIL.  A leak of ``alpha * features`` into the
    decoder's input is a real encoder → decoder channel of unbounded width."""
    _enc, store, _dec, _imgs, Zf = triple
    with pytest.raises(AssertionError) as exc:
        assert_no_feature_bypass(store, Zf, bypass_alpha=alpha)
    msg = str(exc.value)
    assert "NO-BYPASS CHECK FAILED" in msg
    assert "(2) COLLISION" in msg, (
        "the collision leg — different features, same codes, different decoder "
        "input — is the discriminating one and must be among the legs that fire"
    )
    assert "(4) VALUE" in msg


@pytest.mark.parametrize("code_grad", list(CODE_GRADS))
def test_bypass_check_passes_on_the_real_path(triple, code_grad):
    _enc, store, _dec, _imgs, Zf = triple
    rep = assert_no_feature_bypass(store, Zf, code_grad=code_grad)
    assert rep["max_abs_delta_decoder_input_vs_pure_gather"] == 0.0
    assert rep["max_abs_delta_same_codes_different_features"] == 0.0
    assert rep["max_abs_delta_under_jitter"] == 0.0
    assert rep["collision_test_n_rows"] > 0
    # the collision test is only meaningful if the two feature tensors DIFFER
    assert rep["collision_test_feature_max_abs_difference"] > 0.0


def test_collision_criterion_catches_R27s_named_soft_read_trap(triple):
    """⛔ ``soft_read`` is the Hub's own named trap (R27): the same arithmetic as
    the composed read, but it **takes the features**.  The criterion this spoke
    enforces must catch it."""
    _enc, store, _dec, _imgs, Zf = triple
    codes = address_codes(store, Zf)
    Z_alt = np.asarray(read_from_codes(store, codes)).reshape(Zf.shape)
    assert np.max(np.abs(Z_alt - Zf)) > 0.0  # genuinely different features
    assert np.array_equal(address_codes(store, Z_alt), codes)  # identical codes

    pure_a = np.asarray(read_from_codes(store, codes))
    pure_b = np.asarray(read_from_codes(store, address_codes(store, Z_alt)))
    assert np.max(np.abs(pure_a - pure_b)) == 0.0  # the real read: collides safely

    soft_a = np.asarray(soft_read(store, Zf, tau=1.0))
    soft_b = np.asarray(soft_read(store, Z_alt, tau=1.0))
    assert np.max(np.abs(soft_a - soft_b)) > 0.0, (
        "soft_read would have passed the collision criterion, which would mean the "
        "criterion cannot see a feature-dependent read"
    )


def test_straight_through_changes_the_gradient_but_not_the_value(triple):
    _enc, store, _dec, _imgs, Zf = triple
    st, gather_st, codes_st = quantise_through_store(store, Zf, "straight_through")
    sp, gather_sp, codes_sp = quantise_through_store(store, Zf, "stop")
    assert np.max(np.abs(np.asarray(st) - np.asarray(sp))) == 0.0
    assert np.array_equal(np.asarray(codes_st), np.asarray(codes_sp))

    # the VALUE is identical; the JVP is not — that is the whole point.
    def f(mode):
        return lambda z: quantise_through_store(store, z, mode)[0].sum()

    v = jnp.asarray(np.ones_like(Zf))
    _, d_st = jax.jvp(f("straight_through"), (jnp.asarray(Zf),), (v,))
    _, d_sp = jax.jvp(f("stop"), (jnp.asarray(Zf),), (v,))
    assert float(d_sp) == 0.0
    assert float(d_st) != 0.0


def test_traced_halves_are_bit_identical_to_the_shipped_ones(triple):
    _enc, store, _dec, _imgs, Zf = triple
    codes = np.asarray(store_codes_traced(store, jnp.asarray(Zf)), np.int64)
    assert np.array_equal(codes, address_codes(store, Zf))
    g = np.asarray(gather_from_codes_traced(store, codes))
    r = np.asarray(read_from_codes(store, codes)).reshape(g.shape)
    assert np.max(np.abs(g - r)) == 0.0


def test_realised_channel_width_is_the_declared_design_constant(triple):
    _enc, store, _dec, _imgs, Zf = triple
    rep = realised_channel_bits(store, address_codes(store, Zf))
    assert rep["design_bits"] == store.bits_per_read
    assert rep["design_bits"] == float(store.n_wells * np.log2(store.n_modes))
    assert rep["n_distinct_code_configs"] <= rep["max_code_configs"]
    assert rep["realised_bits_used"] <= rep["design_bits"] + 1e-12


def test_bypass_alpha_is_rejected_by_an_unknown_code_grad(triple):
    _enc, store, _dec, _imgs, Zf = triple
    with pytest.raises(ValueError, match="code_grad"):
        quantise_through_store(store, Zf, "free_for_all")


# ---------------------------------------------------------------------------
# §3. The matched-compute control really is a control
# ---------------------------------------------------------------------------


def test_the_frozen_filter_spec_marks_every_encoder_leaf_false(triple):
    enc, store, dec, _imgs, _Zf = triple
    spec = joint_filter_spec(enc, store, dec, unfreeze=False)
    enc_leaves = jax.tree_util.tree_leaves(spec[0])
    assert enc_leaves and not any(enc_leaves)
    assert any(jax.tree_util.tree_leaves(spec[1]))
    assert any(jax.tree_util.tree_leaves(spec[2]))
    on = joint_filter_spec(enc, store, dec, unfreeze=True)
    assert any(jax.tree_util.tree_leaves(on[0]))


def _smoke_cfg(**over):
    cfg = GenEncoderUnfreezeConfig()
    cfg.joint_steps = 3
    cfg.joint_batch = 8
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_frozen_arm_leaves_the_encoder_bit_identical_and_unfrozen_arm_moves_it(triple):
    enc, store, dec, imgs, _Zf = triple
    before = param_hash(enc)
    key = jax.random.PRNGKey(7)
    e_f, s_f, _d_f, rep_f, _k = train_joint(
        enc, store, dec, imgs, _smoke_cfg(), key, unfreeze=False
    )
    assert param_hash(e_f) == before
    assert rep_f["info"]["encoder_changed"] is False

    e_u, s_u, _d_u, rep_u, _k2 = train_joint(
        enc, store, dec, imgs, _smoke_cfg(), key, unfreeze=True
    )
    assert param_hash(e_u) != before
    assert rep_u["info"]["encoder_changed"] is True
    # ⭐ MATCHED COMPUTE: identical step count, identical batch, identical key.
    assert rep_f["info"]["gradient_steps"] == rep_u["info"]["gradient_steps"]
    assert rep_f["info"]["images_seen"] == rep_u["info"]["images_seen"]
    # the store moves in BOTH arms (it is not the treatment)
    assert param_hash(s_f) != param_hash(store)
    assert param_hash(s_u) != param_hash(store)


def test_the_joint_path_still_passes_the_bypass_check_after_training(triple):
    enc, store, dec, imgs, _Zf = triple
    e, s, _d, _rep, _k = train_joint(
        enc, store, dec, imgs, _smoke_cfg(joint_steps=5), jax.random.PRNGKey(3),
        unfreeze=True,
    )
    Z = np.asarray(jax.vmap(e)(jnp.asarray(imgs)))
    rep = assert_no_feature_bypass(s, Z.reshape((Z.shape[0], -1)))
    assert rep["max_abs_delta_same_codes_different_features"] == 0.0
    assert rep["design_bits"] == s.bits_per_read


def test_joint_loss_terms_are_three_finite_scalars(triple):
    enc, store, dec, imgs, _Zf = triple
    parts = joint_loss_terms((enc, store, dec), jnp.asarray(imgs[:8]), "straight_through")
    assert len(parts) == 3
    for p in parts:
        assert np.isfinite(float(p))
    assert float(parts[0]) > 0.0


# ---------------------------------------------------------------------------
# §4. Config block
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_and_round_trips(tmp_path):
    cfg = get_default_config()
    assert isinstance(cfg.gen_encoder_unfreeze, GenEncoderUnfreezeConfig)
    cfg.gen_encoder_unfreeze.joint_steps = 11
    p = tmp_path / "config.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.gen_encoder_unfreeze.joint_steps == 11


def test_the_decoder_recipe_default_is_the_WINNER_not_the_one_that_failed():
    """⛔ Live trap 1: ``GenBridgeConfig.dec_recipe`` still defaults to
    ``"baseline"`` — the decoder that FAILED the gate.  This block must pin the
    winner explicitly and must never inherit the default."""
    c = GenEncoderUnfreezeConfig()
    assert c.dec_recipe == "v1_capacity_schedule"
    assert c.expected_decoder_recipe == "v1_capacity_schedule"
    assert get_default_config().gen_bridge.dec_recipe == "baseline"  # still, upstream


def test_the_config_cannot_express_a_bypass():
    """⛔ ``bypass_alpha`` is a test hook, not a knob: no config field carries it."""
    fields = set(GenEncoderUnfreezeConfig().__dict__)
    assert not any("bypass" in f for f in fields), sorted(fields)
    assert GenEncoderUnfreezeConfig().code_grad in CODE_GRADS


# ---------------------------------------------------------------------------
# §5. The experiment module — plumbing that must not be able to fail quietly
# ---------------------------------------------------------------------------


def _exp():
    import chlu.experiments.exp_gen_encoder as m

    return m


def test_data_root_fails_loudly_with_no_default(monkeypatch):
    """⛔ R26.7: a silently-defaulted data root is 'declared not-run reported as
    null' in code form.  It must raise, not resolve."""
    m = _exp()
    monkeypatch.delenv("CHLU_GEN_DATA_ROOT", raising=False)
    cfg = get_default_config()
    cfg.gen_data.data_root = None
    with pytest.raises(ValueError, match="no gen-data root"):
        m.resolve_data_root(cfg, "shapes3d")


def test_data_root_refuses_a_directory_with_no_dataset_in_it(tmp_path, monkeypatch):
    m = _exp()
    monkeypatch.setenv("CHLU_GEN_DATA_ROOT", str(tmp_path))
    cfg = get_default_config()
    cfg.gen_data.data_root = None
    with pytest.raises(FileNotFoundError, match="refusing to guess"):
        m.resolve_data_root(cfg, "shapes3d")


def test_the_output_root_is_FRESH_and_never_a_bridge_or_store_path():
    """⛔ R18: a changed encoder voids every downstream number, so this
    configuration sits BESIDE the frozen one and writes nowhere near it."""
    m = _exp()
    p = str(m.out_root(get_default_config(), "shapes3d", "recon"))
    assert "gen-encoder-unfreeze" in p
    for forbidden in (
        "gen-feature-bridge", "gen-store-grouping", "gen-decoder-lift",
        "gen-read-compose", "gen-cell", "gen-certificate", "gen-bridge-wide",
    ):
        assert forbidden not in p


def test_quick_keeps_the_thirty_bit_channel_and_voids_the_number():
    """⭐ ``--quick`` shrinks steps and rows but ⛔ NOT ``M``: ``K·log2(M) = 30``
    is R20's declared design constant and the smoke must exercise it."""
    m = _exp()
    cfg = m.apply_quick(get_default_config())
    c = cfg.gen_encoder_unfreeze
    assert float(c.k_wells) * np.log2(float(c.n_modes)) == 30.0
    assert c.joint_steps < GenEncoderUnfreezeConfig().joint_steps
    assert c.n_train < GenEncoderUnfreezeConfig().n_train


def test_both_arms_are_the_default_and_the_order_is_pinned():
    m = _exp()
    assert tuple(GenEncoderUnfreezeConfig().arms) == m.ARMS == ("frozen", "unfrozen")


def test_the_summary_refuses_to_call_the_budgets_matched_when_they_are_not():
    m = _exp()

    def man(steps, imgs, dB, dMS):
        return {
            "distinct_codes_request_side": {
                "B": {"n_distinct_code_configs": dB, "n_requests": 72,
                      "mean_confidence": 0.5},
                "MS": {"n_distinct_code_configs": dMS, "n_requests": 200,
                       "mean_confidence": 0.5},
            },
            "distinct_codes_feature_side": {
                "B": {"n_distinct_code_configs": 8}, "MS": {"n_distinct_code_configs": 9},
            },
            "compute": {"gradient_steps": steps, "images_seen": imgs,
                        "n_params_optimised": 1, "encoder_changed": False,
                        "train_seconds": 1.0},
            "reconstruction": {"B": {"pixel_mse_through_store": 0.1}},
            "channel": {"design_bits": 30.0},
            "address_book": {"n_fit_rows": 512},
        }

    warm = {"encoder_param_hash": "a", "store_param_hash": "b",
            "decoder_param_hash": "c", "D": 2048, "feat_shape": [128, 4, 4]}
    ok = m.summarise_seed(
        {"frozen": man(60, 960, 7, 10), "unfrozen": man(60, 960, 1, 1)}, warm, 0, "d", "recon"
    )
    assert ok["unfrozen_minus_frozen"]["budgets_matched"] is True
    assert ok["unfrozen_minus_frozen"]["distinct_B"] == -6

    bad = m.summarise_seed(
        {"frozen": man(60, 960, 7, 10), "unfrozen": man(120, 1920, 30, 40)},
        warm, 0, "d", "recon",
    )
    assert bad["unfrozen_minus_frozen"]["budgets_matched"] is False, (
        "an unfrozen arm that got twice the steps must NOT be reported as matched "
        "— that is a compute win, not an encoder win"
    )


def test_a_stale_schema_artefact_is_RE_RUN_not_silently_reused():
    """⛔ The trap this caught for real: ``skip_if_exists`` reused a manifest
    written before a field existed, and the run died inside the aggregate with a
    ``KeyError``.  Resume must compare the SCHEMA, not only the status."""
    m = _exp()
    assert m.ARM_SCHEMA.endswith("/2")
    warm = {"encoder_param_hash": "a", "store_param_hash": "b",
            "decoder_param_hash": "c", "D": 2048, "feat_shape": [128, 4, 4]}
    stale = {
        "distinct_codes_request_side": {
            "B": {"n_distinct_code_configs": 7, "n_requests": 72, "mean_confidence": 0.5},
            "MS": {"n_distinct_code_configs": 9, "n_requests": 200, "mean_confidence": 0.5},
        },
        "distinct_codes_feature_side": {
            "B": {"n_distinct_code_configs": 8}, "MS": {"n_distinct_code_configs": 9},
        },
        "compute": {"gradient_steps": 60, "images_seen": 960, "n_params_optimised": 1,
                    "encoder_changed": False, "train_seconds": 1.0},
        "reconstruction": {"B": {"pixel_mse_through_store": 0.1}},
        "channel": {"design_bits": 30.0},
    }  # ⛔ no "address_book" key — exactly the pre-/2 artefact
    got = m.summarise_seed({"frozen": stale}, warm, 0, "d", "recon")
    assert got["arms"]["frozen"]["address_book_fit_rows"] is None


def test_the_launcher_stage_name_and_the_module_name_agree():
    """⛔ ``job_gpu_gen.sh`` maps ``STAGE=encoder`` to this module by NAME and the
    launcher checks the module exists before spending wallclock."""
    m = _exp()
    assert "encoder" in m.STAGES
    parser = m.build_arg_parser()
    args = parser.parse_args(
        ["--dataset", "shapes3d", "--pretext", "recon", "--stage", "encoder",
         "--arm", "both", "--seed", "0", "--out", "/tmp/x"]
    )
    assert args.stage == "encoder" and args.arm == "both" and args.seed == [0]


def test_the_launchers_arm_whitelist_cannot_express_our_arms_so_clu_means_both():
    """⛔ ``job_gpu_gen.sh`` whitelists ``ARMS`` to ``clu store_removed lace knn``
    and is read-only to this spoke, so ``ARMS="frozen unfrozen"`` is refused at
    submit time.  The alias is explicit here, never a silent fallback."""
    m = _exp()
    assert m.LAUNCHER_ARM_ALIASES == {"clu": "both"}
    parsed = m.build_arg_parser().parse_args(["--arm", "clu"])
    assert m.LAUNCHER_ARM_ALIASES.get(parsed.arm) == "both"
    with pytest.raises(SystemExit):
        m.build_arg_parser().parse_args(["--arm", "lace"])
