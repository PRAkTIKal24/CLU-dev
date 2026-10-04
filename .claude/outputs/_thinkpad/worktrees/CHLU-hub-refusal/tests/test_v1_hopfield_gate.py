"""Smoke tests for the V1.1 gate-stack-on-Hopfield experiment (P7/V1.1)."""

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import get_default_config
from chlu.data.mqar import make_token_embeddings
from chlu.experiments.exp_v1_hopfield_gate import (
    HOPFIELD_LADDER,
    _hopfield_energy,
    _hopfield_ladder,
    run_experiment_v1_hopfield_gate,
)


def test_hopfield_energy_shape_and_finite():
    key = jax.random.PRNGKey(0)
    patterns = jax.random.normal(key, (8, 6))
    z = jax.random.normal(jax.random.PRNGKey(1), (5, 6))
    E = _hopfield_energy(z, patterns, beta=10.0)
    assert E.shape == (5,)
    assert np.all(np.isfinite(np.asarray(E)))


def test_hopfield_ladder_retrieves_stored_value():
    """A clean cue retrieves its bound value at the top of the ladder."""
    e = 16  # matches the experiment scale; e=4 has too many collisions to store 4 keys
    embeds = make_token_embeddings(jax.random.PRNGKey(2), 32, e, scale=2.0)
    keys_tok = jnp.arange(0, 4)
    vals_tok = jnp.arange(16, 20)
    stored = jnp.concatenate([embeds[keys_tok], embeds[vals_tok]], axis=1)
    val_tokens = jnp.arange(16, 32)
    val_embeds = embeds[val_tokens]
    # exact cue for key 0 (value half zero)
    q0 = jnp.concatenate([embeds[keys_tok], jnp.zeros((4, e))], axis=1)
    rec = _hopfield_ladder(q0, stored, 20.0, e, val_embeds, val_tokens, vals_tok)
    assert rec["R"].shape == (4, len(HOPFIELD_LADDER))
    assert rec["correct"].all()  # clean cues recall their bound values


def test_experiment_quick_runs_and_reports_metrics():
    cfg = get_default_config()
    out = run_experiment_v1_hopfield_gate(config=cfg, save_dir="/tmp/mp_hop", quick=True)
    s = out["summary"]
    assert "pooled" in s
    assert "auroc_raw_R" in s["pooled"] and "auroc_calibrated" in s["pooled"]
    for blk in s["per_level"].values():
        assert "savings" in blk and "ltt" in blk
