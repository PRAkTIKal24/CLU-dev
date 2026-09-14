"""⭐ Candidate reconstruction encoders (``gen-encoder-pretrain``, R40.E).

⛔ **The load-bearing tests here are the ones that can fail on purpose:**

- ``test_the_control_arm_is_the_SHIPPED_encoder_path_bit_for_bit`` — the whole
  spoke is a comparison *against* the current encoder; if the control row is not
  literally ``gen_bridge.train_encoder``, every margin in the report is a
  comparison of two new things.
- ``test_a_control_arm_whose_config_DRIFTED_is_refused`` — positive control for
  the like-for-like guard.
- ``test_a_CORNER_only_pass_does_NOT_clear_the_bar`` — ⚠ trap 2, the exact way
  this spoke was told it could be wrong: the corner is 216 rows realising 3 of 8
  ``scale`` values, the spread grid is the test.
- ``test_an_MLP_only_pass_is_reported_SEPARATELY_and_never_as_PASS`` — the
  pre-registered reading of "present but not linearly readable".
- ``test_the_row_disjointness_assertion_FIRES`` — an assertion never seen to fail
  is not evidence.

Everything runs on synthetic tensors: ⛔ no dataset on disk is required.  The
real-data numbers live in ``.claude/outputs/gen-encoder-pretrain.md``.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import CHLUConfig, GenPretrainConfig, load_config, save_config
from chlu.core.gen_bridge import FactorHeads, GenEncoder, param_hash, train_encoder
from chlu.core.gen_encoders import (
    PRETRAIN_VARIANTS,
    PROBE_KINDS,
    SHIPPED_PATH_VARIANTS,
    VAEEncoder,
    VariantError,
    bar_verdict,
    fit_feature_probe,
    gaussian_kl,
    masked_mse,
    patch_mask,
    realised_values,
    standardize,
    standardize_stats,
    train_variant,
    variant_spec,
)
from chlu.eval.gen_probe import probe_accuracy
from chlu.experiments.exp_gen_pretrain import (
    POOLS,
    aggregate,
    apply_quick,
    assert_rows_disjoint,
    build_arg_parser,
    flag_provenance,
    run_variant_seed,
    select_pretrain_rows,
    write_tables,
)


# ---------------------------------------------------------------------------
# tiny fixtures
# ---------------------------------------------------------------------------


def _tiny_cfg(**over):
    cfg = GenPretrainConfig()
    cfg.enc_channels = [4, 8]
    cfg.hires_channels = [4]
    cfg.dec_channels = [8, 4]
    cfg.enc_kernel = 4
    cfg.enc_stride = 2
    cfg.enc_groups = 2
    cfg.dec_groups = 2
    cfg.enc_steps = 2
    cfg.enc_batch = 4
    cfg.probe_steps = 2
    cfg.probe_batch = 4
    cfg.probe_hidden = 8
    cfg.eval_batch = 4
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _tiny_bridge_cfg(cfg):
    """A ``GenBridgeConfig`` that matches ``cfg`` on every like-for-like field."""
    bc = CHLUConfig().gen_bridge
    bc.enc_channels = list(cfg.enc_channels)
    bc.dec_channels = list(cfg.dec_channels)
    bc.enc_kernel = int(cfg.enc_kernel)
    bc.enc_stride = int(cfg.enc_stride)
    bc.enc_groups = int(cfg.enc_groups)
    bc.dec_groups = int(cfg.dec_groups)
    bc.enc_batch = int(cfg.enc_batch)
    bc.enc_lr = float(cfg.enc_lr)
    bc.enc_steps = int(cfg.enc_steps)
    return bc


def _tiny_images(n=8, c=3, s=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, c, s, s)).astype(np.float32)


def _toy_grid():
    from chlu.data.gen_splits import FactorGrid

    return FactorGrid(
        names=("a", "scale", "orientation"),
        values=(np.arange(2, dtype=float), np.arange(3, dtype=float),
                np.arange(4, dtype=float)),
        dataset="toy",
    )


def _toy_split():
    from chlu.data.gen_splits import GenSplit

    grid = _toy_grid()
    rows = np.arange(grid.n_rows, dtype=np.int64)

    def pred(values):
        v = np.asarray(values)
        return (v[:, 0] >= 1) & (v[:, 1] >= 2) & (v[:, 2] >= 3)

    mask = pred(grid.values_of(rows))
    return GenSplit(dataset="toy", condition="toy", variant="leave1out", grid=grid,
                    train_rows=rows[~mask], test_rows=rows[mask], test_predicate=pred)


# ---------------------------------------------------------------------------
# 1. ⭐ the control row IS the shipped path
# ---------------------------------------------------------------------------


def test_the_control_arm_is_the_SHIPPED_encoder_path_bit_for_bit():
    """⭐ ``recon_control`` must be ``gen_bridge.train_encoder``, not a lookalike.

    ⛔ If this ever fails, every margin in the report is a comparison of two new
    encoders and the laundering control has quietly stopped being a control.
    """
    cfg = _tiny_cfg()
    bc = _tiny_bridge_cfg(cfg)
    imgs = _tiny_images()
    key = jax.random.PRNGKey(3)
    shipped, _, _ = train_encoder(imgs, "recon", bc, key)
    got, _, _, meta = train_variant(imgs, "recon_control", cfg, key, bridge_cfg=bc)
    assert param_hash(got) == param_hash(shipped)
    assert meta["path"] == "gen_bridge.train_encoder"
    assert "recon_control" in SHIPPED_PATH_VARIANTS


def test_ae_plain_is_the_control_objective_from_the_SAME_init():
    """⭐ The plumbing null.  ``ae_plain`` runs the control's objective through this
    module's own loop, so any gap between the two rows is loop nuisance and nothing
    else.  The encoder INIT is identical by construction (the same ``ek``).

    ⚠ **Measured on this machine, and it is an environment fact worth knowing:**
    under ``jax_threefry_partitionable=True`` (the default from JAX 0.9)
    ``jax.random.split(key, n)`` is **prefix-stable** — ``split(k, 2)[1]`` *is*
    ``split(k, 3)[1]`` — so this module's ``split(key, 3)`` draws the *same* batch
    stream as the shipped loop's ``split(key, 2)``, and the two arms come out
    **bit-identical**.  ⛔ Do not assume "I split one extra subkey" decorrelates an
    RNG stream in this environment; it does not.
    """
    cfg = _tiny_cfg()
    bc = _tiny_bridge_cfg(cfg)
    imgs, key = _tiny_images(), jax.random.PRNGKey(11)
    ctl, _, _, ctl_meta = train_variant(imgs, "recon_control", cfg, key, bridge_cfg=bc)
    plain, _, _, plain_meta = train_variant(imgs, "ae_plain", cfg, key, bridge_cfg=bc)
    assert ctl_meta["objective"] == plain_meta["objective"]
    assert plain_meta["path"] == "gen_encoders.train_variant"
    # both paths take their encoder key from ``split(key, 3)[1]``, so the INIT is
    # identical whatever the split implementation does.
    _, ek, _ = jax.random.split(key, 3)
    assert param_hash(GenEncoder((3, 16, 16), tuple(cfg.enc_channels), cfg.enc_kernel,
                                 cfg.enc_stride, cfg.enc_groups, ek)) is not None
    if bool(jax.config.jax_threefry_partitionable):
        # prefix-stable split => the same batch stream => the same trained encoder
        assert param_hash(ctl) == param_hash(plain)


def test_the_control_arm_restores_the_bridge_config_it_borrowed():
    """⛔ A spoke that leaves ``gen_bridge.enc_steps`` mutated poisons the next run."""
    cfg = _tiny_cfg()
    bc = _tiny_bridge_cfg(cfg)
    bc.enc_steps = 5
    train_variant(_tiny_images(), "recon_control_2x", cfg, jax.random.PRNGKey(0), bridge_cfg=bc)
    assert bc.enc_steps == 5


def test_a_control_arm_whose_config_DRIFTED_is_refused():
    """⭐ POSITIVE CONTROL for the like-for-like guard: it must FIRE, and name the field."""
    cfg = _tiny_cfg()
    bc = _tiny_bridge_cfg(cfg)
    bc.enc_channels = [4, 8, 16]  # the bridge quietly grew a stage
    with pytest.raises(VariantError, match="enc_channels"):
        train_variant(_tiny_images(), "recon_control", cfg, jax.random.PRNGKey(0), bridge_cfg=bc)


def test_the_shipped_path_arms_need_the_bridge_config():
    cfg = _tiny_cfg()
    with pytest.raises(VariantError, match="gen_bridge"):
        train_variant(_tiny_images(), "recon_control", cfg, jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# 2. the arms: compute, the laundering floor, and the excluded family
# ---------------------------------------------------------------------------


def test_the_compute_control_runs_exactly_N_times_the_steps():
    """⭐ ``recon_control_2x`` exists so a compute win cannot pass as a representation win."""
    cfg = _tiny_cfg(compute_control_mult=3)
    bc = _tiny_bridge_cfg(cfg)
    _, _, _, base = train_variant(_tiny_images(), "recon_control", cfg,
                                  jax.random.PRNGKey(0), bridge_cfg=bc)
    _, _, _, more = train_variant(_tiny_images(), "recon_control_2x", cfg,
                                  jax.random.PRNGKey(0), bridge_cfg=bc)
    assert more["steps"] == 3 * base["steps"]
    assert base["steps_mult"] == 1 and more["steps_mult"] == 3


def test_the_random_init_floor_trains_for_zero_steps():
    """⭐ The trivial substitute: an untrained conv stack, so it can never win on compute."""
    cfg = _tiny_cfg()
    enc, _, hist, meta = train_variant(_tiny_images(), "random_init", cfg, jax.random.PRNGKey(1))
    assert meta["steps"] == 0 and hist == [] and meta["objective"] == "NONE (untrained)"
    assert meta["is_control"] is True
    key, ek, _ = jax.random.split(jax.random.PRNGKey(1), 3)
    fresh = GenEncoder((3, 16, 16), tuple(cfg.enc_channels), cfg.enc_kernel,
                       cfg.enc_stride, cfg.enc_groups, ek)
    assert param_hash(enc) == param_hash(fresh)


def test_no_augmentation_contrastive_arm_exists():
    """⛔ By ruling (R40.E): it trains invariance to exactly ``scale`` and ``orientation``.

    ⚠ This is a *design* assertion, not a style one — an arm added "for
    completeness" would be optimising away the two factors the spoke measures.
    """
    banned = ("contrastive", "simclr", "byol", "moco", "augment")
    assert not [v for v in PRETRAIN_VARIANTS if any(b in v.lower() for b in banned)]


def test_every_named_variant_resolves_and_only_the_vae_is_the_comparison_arm():
    cfg = _tiny_cfg()
    specs = {v: variant_spec(v, cfg) for v in PRETRAIN_VARIANTS}
    assert [v for v, s in specs.items() if s.is_comparison_arm] == ["vae"]
    assert set(v for v, s in specs.items() if s.is_control) == {
        "recon_control", "random_init", "ae_plain", "recon_control_2x"}
    with pytest.raises(VariantError):
        variant_spec("contrastive", cfg)


def test_ae_hires_keeps_D_inside_the_lace_whitening_budget():
    """⭐ ``D`` is a design constraint (LACE needs a ``D x D`` eigendecomposition)."""
    from chlu.core.gen_bridge import WHITENING_D_LIMIT

    cfg = CHLUConfig().gen_pretrain
    enc = GenEncoder((3, 64, 64), tuple(cfg.hires_channels), cfg.enc_kernel,
                     cfg.enc_stride, cfg.enc_groups, jax.random.PRNGKey(0))
    assert enc.feature_dim == 8192
    assert enc.feature_dim <= WHITENING_D_LIMIT


# ---------------------------------------------------------------------------
# 3. the MAE and VAE objectives
# ---------------------------------------------------------------------------


def test_patch_mask_is_constant_within_a_patch_and_hits_the_ratio():
    m = np.asarray(patch_mask(jax.random.PRNGKey(0), (64, 1, 16, 16), 4, 0.5))
    blocks = m.reshape(64, 1, 4, 4, 4, 4)
    assert np.all(blocks.min(axis=(3, 5)) == blocks.max(axis=(3, 5)))
    assert 0.4 < m.mean() < 0.6  # Bernoulli: the EXPECTED fraction, stated as such


def test_masked_mse_ignores_the_pixels_the_encoder_was_shown():
    x = jnp.zeros((2, 1, 4, 4))
    recon = jnp.ones((2, 1, 4, 4))
    mask = jnp.zeros((2, 1, 4, 4)).at[:, :, :2, :].set(1.0)
    assert float(masked_mse(recon, x, mask, True)) == pytest.approx(1.0)
    assert float(masked_mse(recon, x, mask, False)) == pytest.approx(1.0)
    # a wrong reconstruction of a VISIBLE pixel is free under the MAE loss
    recon2 = recon.at[:, :, 3, :].set(7.0)
    assert float(masked_mse(recon2, x, mask, True)) == pytest.approx(1.0)
    assert float(masked_mse(recon2, x, mask, False)) > 1.0


def test_gaussian_kl_vanishes_at_the_standard_normal():
    z = jnp.zeros((3, 4))
    assert float(gaussian_kl(z, z)) == pytest.approx(0.0, abs=1e-6)
    assert float(gaussian_kl(jnp.ones((3, 4)), z)) > 0.0


def test_the_vae_arm_reads_deterministically_when_frozen():
    """⛔ A frozen encoder that resamples at read time is not frozen."""
    cfg = _tiny_cfg()
    enc, _, _, meta = train_variant(_tiny_images(), "vae", cfg, jax.random.PRNGKey(0))
    assert isinstance(enc, VAEEncoder)
    x = _tiny_images(n=1)[0]
    a, b = np.asarray(enc(x)), np.asarray(enc(x))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, np.asarray(enc.moments(x)[0]), rtol=0, atol=0)
    assert meta["is_comparison_arm"] is True


# ---------------------------------------------------------------------------
# 4. ⭐ the probe and the bar
# ---------------------------------------------------------------------------


def test_the_linear_probe_is_the_shipped_head_family_and_the_shipped_scorer():
    """⭐ Same head family + same scoring code as the rest of the phase."""
    cfg = _tiny_cfg()
    cards = (2, 3, 4)
    feats = np.asarray(np.random.default_rng(0).random((16, 12)), np.float32)
    labels = np.stack([np.arange(16) % c for c in cards], axis=1).astype(np.int32)
    probe, _ = fit_feature_probe(feats, labels, cards, cfg, jax.random.PRNGKey(0), "linear")
    assert probe.hidden is None and isinstance(probe.heads, FactorHeads)
    out = probe(feats[0])
    assert [int(o.shape[0]) for o in out] == list(cards)
    acc = probe_accuracy(probe, feats, labels, ("a", "scale", "orientation"), 8)
    assert set(acc["per_factor"]) == {"a", "scale", "orientation"}
    mlp, _ = fit_feature_probe(feats, labels, cards, cfg, jax.random.PRNGKey(0), "mlp")
    assert mlp.hidden is not None
    with pytest.raises(VariantError):
        fit_feature_probe(feats, labels, cards, cfg, jax.random.PRNGKey(0), "knn")
    assert set(PROBE_KINDS) == {"linear", "mlp"}


def test_standardisation_uses_the_FIT_rows_statistics_only():
    """⛔ Re-fitting the z-score on the eval rows leaks eval statistics into the read."""
    fit = np.asarray([[0.0, 10.0], [2.0, 14.0]], np.float32)
    stats = standardize_stats(fit)
    np.testing.assert_allclose(stats["mean"], [1.0, 12.0])
    ev = np.asarray([[1.0, 12.0], [3.0, 16.0]], np.float32)
    z = standardize(ev, stats)
    np.testing.assert_allclose(z[0], [0.0, 0.0], atol=1e-6)
    assert not np.allclose(z.mean(axis=0), 0.0)  # ⛔ it did NOT re-centre on eval


def test_the_bar_needs_EVERY_factor_not_the_mean_of_them():
    """⭐ ``scale`` 0.99 with ``orientation`` 0.10 is a FAIL, not a 0.55."""
    pf = {"scale": 0.99, "orientation": 0.10, "shape": 1.0}
    v = bar_verdict(pf, ["scale", "orientation"], 0.9)
    assert v["verdict"] == "FAIL"
    assert v["margin"]["orientation"] == pytest.approx(-0.8)
    assert bar_verdict({"scale": 0.91, "orientation": 0.90},
                       ["scale", "orientation"], 0.9)["verdict"] == "PASS"


def test_a_per_factor_number_always_carries_n_values_realised():
    """⚠ Trap 1: a per-factor statistic without ``n_values_realised`` is meaningless."""
    labels = np.array([[0, 5, 1], [0, 6, 1], [0, 7, 2]], np.int32)
    rv = realised_values(labels, ("a", "scale", "orientation"))
    assert rv["a"]["n_values_realised"] == 1 and rv["a"]["chance"] == pytest.approx(1.0)
    assert rv["scale"]["n_values_realised"] == 3 and rv["scale"]["values"] == [5, 6, 7]
    assert rv["orientation"]["majority_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 5. rows: disjointness, and the corner kept OUT of the spread pools
# ---------------------------------------------------------------------------


def test_select_rows_keeps_the_corner_out_of_the_probe_pools():
    split = _toy_split()
    corner = np.asarray([0, 1, 2, 3], np.int64)
    cfg = _tiny_cfg(n_enc_train=4, n_probe_train=4, n_probe_eval=4)
    rows = select_pretrain_rows(split, corner, cfg, seed=0)
    assert not np.intersect1d(rows["probe_train"], corner).size
    assert not np.intersect1d(rows["probe_eval"], corner).size
    assert not np.intersect1d(rows["corner_eval"], rows["enc_train"]).size
    assert assert_rows_disjoint(rows)


def test_the_row_disjointness_assertion_FIRES():
    """⭐ POSITIVE CONTROL: a probe scored on a row it was fitted on must be caught."""
    rows = {
        "enc_train": np.array([0, 1], np.int64),
        "probe_train": np.array([2, 3], np.int64),
        "probe_eval": np.array([3, 4], np.int64),  # ⛔ shares row 3 with probe_train
        "corner_eval": np.array([9], np.int64),
    }
    with pytest.raises(AssertionError, match="probe_train"):
        assert_rows_disjoint(rows)


def test_the_corner_is_dropped_from_include_corner_false():
    split = _toy_split()
    cfg = _tiny_cfg(n_enc_train=4, n_probe_train=4, n_probe_eval=4, include_corner=False)
    rows = select_pretrain_rows(split, np.asarray([0, 1], np.int64), cfg, seed=0)
    assert rows["corner_eval"].size == 0


# ---------------------------------------------------------------------------
# 6. ⭐ the verdict: what a corner-only or MLP-only pass is allowed to say
# ---------------------------------------------------------------------------


def _fake_result(variant, seed, grid_vals, corner_vals, kinds=("linear",)):
    names = ["a", "scale", "orientation"]

    def acc(vals, n):
        return {"n": n, "per_factor": dict(zip(names, vals, strict=True)),
                "aggregate": float(np.mean(vals)), "exact_match": 0.0}

    realised = {f: {"n_values_realised": 3, "chance": 1 / 3, "majority_rate": 0.4,
                    "values": [0, 1, 2]} for f in names}
    probes = {k: {"kind": k, "grid": acc(grid_vals, 2048), "corner": acc(corner_vals, 216),
                  "train": acc(grid_vals, 4096)} for k in kinds}
    return {
        "schema": "chlu-gen-pretrain/1", "variant": variant, "seed": seed,
        "encoder": {"family": "ae", "note": "", "is_control": False,
                    "is_comparison_arm": False, "enc_channels": [4], "dec_channels": [4],
                    "dec_blocks_per_stage": 1, "steps": 10, "steps_mult": 1,
                    "encoder_param_count": 1, "pretext_head_param_count": 1, "D": 8,
                    "feat_shape": [2, 2, 2], "objective": "mse", "loss_final": 0.1},
        "probes": probes,
        "pools": {p: {"n": 10, "realised": realised} for p in POOLS},
        "timing": {"train_seconds": 1.0, "encode_seconds": 0.1},
    }


def test_a_CORNER_only_pass_does_NOT_clear_the_bar():
    """⭐⭐ Trap 2 / "what would prove this spoke wrong": the corner is 216 rows with
    3 of 8 realised ``scale`` values; the SPREAD GRID is the test."""
    config = CHLUConfig()
    results = [_fake_result("ae_deep", s, [1.0, 0.40, 0.35], [1.0, 0.97, 0.96])
               for s in (0, 1, 2)]
    summary = aggregate(results, config)
    assert summary["by_variant"]["ae_deep"]["by_probe"]["linear"]["corner"]["bar"]["verdict"] == "PASS"
    assert summary["by_variant"]["ae_deep"]["by_probe"]["linear"]["grid"]["bar"]["verdict"] == "FAIL"
    assert summary["verdict"] == "FAIL"
    assert summary["variants_clearing_bar_grid_linear"] == []


def test_an_MLP_only_pass_is_reported_SEPARATELY_and_never_as_PASS():
    """⭐ Pre-registered reading: linear FAIL + MLP PASS = *present but not linearly
    readable*, a different finding — ⛔ it does not clear the bar."""
    config = CHLUConfig()
    results = []
    for s in (0, 1, 2):
        r = _fake_result("mae", s, [1.0, 0.5, 0.5], [1.0, 0.5, 0.5], kinds=("linear", "mlp"))
        r["probes"]["mlp"]["grid"]["per_factor"] = {"a": 1.0, "scale": 0.95, "orientation": 0.93}
        results.append(r)
    summary = aggregate(results, config)
    assert summary["verdict"] == "FAIL"
    assert summary["variants_clearing_bar_grid_linear"] == []
    assert summary["variants_clearing_bar_grid_mlp_only"] == ["mae"]
    assert summary["headline_probe"] == "linear"


def test_a_genuine_grid_pass_is_reported_as_PASS():
    config = CHLUConfig()
    results = [_fake_result("ae_hires", s, [1.0, 0.94, 0.91], [1.0, 0.99, 0.98])
               for s in (0, 1, 2)]
    summary = aggregate(results, config)
    assert summary["verdict"] == "PASS"
    assert summary["variants_clearing_bar_grid_linear"] == ["ae_hires"]


def test_tables_carry_n_values_realised_chance_and_the_compute_column(tmp_path):
    config = CHLUConfig()
    results = [_fake_result("ae_deep", s, [1.0, 0.4, 0.3], [1.0, 0.9, 0.9]) for s in (0, 1, 2)]
    text = write_tables(aggregate(results, config), tmp_path / "t.md").read_text()
    assert "n_values_realised" in text and "chance" in text
    assert "pool=grid" in text and "pool=corner" in text
    assert "a compute win is not a representation win" in text


# ---------------------------------------------------------------------------
# 7. end-to-end on synthetic pools (⛔ no dataset on disk)
# ---------------------------------------------------------------------------


def test_one_variant_seed_runs_end_to_end_and_writes_its_own_result(tmp_path):
    cfg = _tiny_cfg(n_enc_train=8, n_probe_train=8, n_probe_eval=8,
                    probe_kinds=["linear", "mlp"], bar_factors=["scale", "orientation"])
    config = CHLUConfig()
    config.gen_pretrain = cfg
    config.gen_bridge = _tiny_bridge_cfg(cfg)
    names = ("a", "scale", "orientation")
    cards = (2, 3, 4)
    pools = {k: _tiny_images(n=8, seed=i) for i, k in
             enumerate(("enc_train", "probe_train", "probe_eval", "corner_eval"))}
    pools["factor_names"] = list(names)
    pools["cardinalities"] = cards
    labels = {k: np.stack([np.arange(8) % c for c in cards], axis=1).astype(np.int32)
              for k in ("enc_train", "probe_train", "probe_eval", "corner_eval")}
    rows = {k: np.arange(8, dtype=np.int64) for k in
            ("enc_train", "probe_train", "probe_eval", "corner_eval")}
    rows["n_corner_dropped_in_enc_train"] = 0
    res = run_variant_seed("ae_plain", 0, pools, labels, rows, config, tmp_path)
    on_disk = json.loads((tmp_path / "result.json").read_text())
    assert on_disk["variant"] == "ae_plain" and on_disk["seed"] == 0
    assert set(res["probes"]) == {"linear", "mlp"}
    for kind in ("linear", "mlp"):
        assert set(res["probes"][kind]) >= {"train", "grid", "corner"}
        assert res["bar"][kind]["grid"]["verdict"] in {"PASS", "FAIL"}
    assert res["pools"]["grid"]["realised"]["scale"]["n_values_realised"] == 3
    assert res["encoder"]["pretext_head_discarded"] is True
    assert res["flags"]["pass_bar"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 8. config + CLI plumbing
# ---------------------------------------------------------------------------


def test_gen_pretrain_config_round_trips_through_yaml(tmp_path):
    config = CHLUConfig()
    config.gen_pretrain.enc_steps = 123
    config.gen_pretrain.variants = ["recon_control", "mae"]
    save_config(config, tmp_path / "c.yaml")
    back = load_config(tmp_path / "c.yaml")
    assert back.gen_pretrain.enc_steps == 123
    assert back.gen_pretrain.variants == ["recon_control", "mae"]


def test_the_defaults_keep_both_controls_and_the_declared_comparison_arm():
    gp = CHLUConfig().gen_pretrain
    assert {"recon_control", "random_init", "recon_control_2x"} <= set(gp.variants)
    assert "vae" in gp.variants
    assert gp.pass_bar == pytest.approx(0.90)
    assert gp.bar_factors == ["scale", "orientation"]
    assert gp.seeds == [0, 1, 2]
    # ⚠ reduced scale is DECLARED: below the bridge's 8192 / 1500, deliberately.
    assert gp.n_enc_train < CHLUConfig().gen_bridge.n_train
    assert gp.enc_steps < CHLUConfig().gen_bridge.enc_steps


def test_quick_is_plumbing_only_and_shrinks_every_axis():
    config = apply_quick(CHLUConfig())
    gp = config.gen_pretrain
    assert gp.enc_steps == gp.quick_enc_steps and gp.seeds == gp.quick_seeds
    assert gp.n_enc_train == gp.quick_n_enc_train
    assert gp.probe_steps == gp.quick_probe_steps


def test_the_cli_refuses_an_unknown_arm():
    p = build_arg_parser()
    assert p.parse_args(["--variants", "mae", "recon_control"]).variants == ["mae", "recon_control"]
    with pytest.raises(SystemExit):
        p.parse_args(["--variants", "contrastive"])


def test_flag_provenance_names_every_training_knob_the_report_must_carry():
    flags = flag_provenance(CHLUConfig(), quick=False)
    for k in ("enc_steps", "enc_lr", "enc_batch", "n_enc_train", "probe_steps",
              "probe_lr", "pass_bar", "bar_factors", "vae_beta", "mae_mask_ratio",
              "compute_control_mult", "seeds", "jax_version"):
        assert k in flags, k


def test_gen_pretrain_config_is_a_dataclass_with_no_mutable_default_shared():
    a, b = GenPretrainConfig(), GenPretrainConfig()
    a.variants.append("mae")
    assert "mae" not in b.variants or b.variants.count("mae") == 1
    assert dataclasses.is_dataclass(GenPretrainConfig)
