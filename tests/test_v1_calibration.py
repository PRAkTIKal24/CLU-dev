"""Smoke tests for the V1-calibration experiment pieces (exp_v1_calibration)."""

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.data.mqar import make_token_embeddings
from chlu.experiments.exp_v1_calibration import (
    _fit_episode_heads,
    _hopfield_confidences,
    _ladder_records,
    _probe_cues,
)


def _tiny_cfg():
    cfg = get_default_config().experiment_v1_gate
    cfg.dt = 0.05
    cfg.relax_steps = 6
    cfg.calib_n_stages = 2
    cfg.calib_stage_steps = 4
    cfg.governor_sensitivity = 0.9
    cfg.calib_probes_per_key = 3
    cfg.calib_cue_noise_scales = [0.1, 0.3]
    cfg.calib_n_impostors = 4
    cfg.vocab_size = 64
    return cfg


def test_probe_cues_composition_and_determinism():
    cfg = _tiny_cfg()
    e = 4
    key = jax.random.PRNGKey(0)
    embeds = make_token_embeddings(jax.random.PRNGKey(1), cfg.vocab_size, e)
    keys_tok = jnp.array([3, 7, 11])
    vals_tok = jnp.array([40, 45, 50])
    q0, true_tok, is_imp = _probe_cues(key, embeds, keys_tok, vals_tok, cfg, e)
    kv, P = 3, cfg.calib_probes_per_key
    T = kv * P + cfg.calib_n_impostors
    assert q0.shape == (T, 2 * e)
    assert true_tok.shape == (T,)
    assert is_imp.shape == (T,) and is_imp.sum() == cfg.calib_n_impostors
    # jittered probes carry their key's bound value; impostors the -1 sentinel
    assert np.all(np.asarray(true_tok[: kv * P]) == np.repeat([40, 45, 50], P))
    assert np.all(np.asarray(true_tok[kv * P :]) == -1)
    # value half of every cue starts at zero
    assert np.allclose(np.asarray(q0[:, e:]), 0.0)
    # impostor cues are exact embeddings of tokens NOT in the dictionary
    imp_cues = np.asarray(q0[kv * P :, :e])
    stored_keys = {3, 7, 11}
    matches = np.isclose(
        imp_cues[:, None, :], np.asarray(embeds)[None, :, :], atol=1e-7
    ).all(axis=2)
    for row in matches:
        toks = np.flatnonzero(row)
        assert len(toks) == 1 and int(toks[0]) not in stored_keys
        assert 1 <= int(toks[0]) < cfg.vocab_size // 2
    # deterministic given the key
    q0b, true_b, imp_b = _probe_cues(key, embeds, keys_tok, vals_tok, cfg, e)
    assert np.allclose(np.asarray(q0), np.asarray(q0b))
    assert np.all(np.asarray(true_tok) == np.asarray(true_b))


def test_ladder_records_shapes_cost_and_clamp():
    cfg = _tiny_cfg()
    e, dim, T = 2, 4, 5
    key = jax.random.PRNGKey(2)
    model = CHLU(dim=dim, hidden=8, kinetic_mode="relativistic", key=key)
    q0 = jax.random.normal(jax.random.fold_in(key, 1), (T, dim)) * 0.3
    p0 = jnp.zeros((T, dim))
    val_embeds = jax.random.normal(jax.random.fold_in(key, 2), (8, e)) * 0.5
    val_tokens = jnp.arange(100, 108)
    true_tok = jnp.array([100, 101, 102, 103, -1])  # last = impostor sentinel

    rec = _ladder_records(
        model, q0, p0, true_tok, val_embeds, val_tokens, e, 0.0, cfg, cd=e
    )
    Ss = cfg.calib_n_stages + 1
    for k in ("R", "margin", "pred", "correct"):
        assert rec[k].shape == (T, Ss)
    assert rec["correct"].dtype == bool
    # impostor sentinel can never be decoded as correct
    assert not rec["correct"][-1].any()
    # cost ladder: base, then + stage_steps per stage
    expected = [cfg.relax_steps]
    for _ in range(cfg.calib_n_stages):
        expected.append(expected[-1] + cfg.calib_stage_steps)
    assert rec["cost"].tolist() == expected


def test_hopfield_confidences_shapes_and_easy_retrieval():
    # e = 8: enough embedding dimensions that random val embeddings do not
    # collide in the nearest-neighbor decode (e = 4 provably collides here;
    # the experiment default is e = 16).
    e = 8
    key = jax.random.PRNGKey(3)
    embeds = make_token_embeddings(key, 64, e, scale=2.0)
    keys_tok = jnp.array([1, 2, 3, 4])
    vals_tok = jnp.array([40, 41, 42, 43])
    stored = jnp.concatenate([embeds[keys_tok], embeds[vals_tok]], axis=1)
    val_tokens = jnp.arange(32, 64)
    val_embeds = embeds[val_tokens]
    q0 = jnp.concatenate([embeds[keys_tok], jnp.zeros((4, e))], axis=1)
    pred, msp, lm = _hopfield_confidences(
        q0, stored, beta=20.0, e=e, val_embeds=val_embeds, val_tokens=val_tokens
    )
    assert pred.shape == msp.shape == lm.shape == (4,)
    # exact-cue retrieval from random Gaussian patterns should be perfect
    assert np.all(pred == np.asarray(vals_tok))
    assert np.all(msp > 0.5) and np.all(msp <= 1.0)
    assert np.all(lm > 0)


def test_fit_episode_heads_returns_all_four():
    cfg = _tiny_cfg()
    rng = np.random.default_rng(0)
    T, Ss = 20, 3
    probe_rec = {
        "R": rng.normal(1.0, 0.5, (T, Ss)),
        "margin": rng.uniform(0, 1, (T, Ss)),
        "correct": rng.uniform(size=(T, Ss)) < 0.7,
    }
    hop_wrong = rng.uniform(size=T) < 0.3
    hop_lm = rng.uniform(0, 5, T)
    heads = _fit_episode_heads(probe_rec, hop_wrong, hop_lm, cfg)
    assert set(heads) == {"r", "margin", "r_margin", "hopfield"}
    pw = heads["r_margin"].p_wrong(R=probe_rec["R"], margin=probe_rec["margin"])
    assert pw.shape == (T, Ss)
    assert np.all((pw >= 0) & (pw <= 1))
    # stage-0-only fit also works
    cfg.calib_fit_all_stages = False
    heads0 = _fit_episode_heads(probe_rec, hop_wrong, hop_lm, cfg)
    assert heads0["r_margin"].n_fit == T
