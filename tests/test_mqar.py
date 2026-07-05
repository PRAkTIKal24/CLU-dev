"""Tests for the MQAR data generator (Zoology task semantics)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.data.mqar import (
    IGNORE_INDEX,
    PAD_TOKEN,
    generate_mqar,
    make_token_embeddings,
)


def test_mqar_shapes_and_vocab_split():
    key = jax.random.PRNGKey(0)
    n, N, kv, V = 4, 64, 8, 256
    out = generate_mqar(key, n, N, kv, vocab_size=V)
    assert out["tokens"].shape == (n, N)
    assert out["targets"].shape == (n, N)
    assert out["keys"].shape == (n, kv)
    assert out["values"].shape == (n, kv)
    assert out["query_pos"].shape == (n, kv)
    # keys in [1, V/2), values in [V/2, V)
    assert np.all(np.asarray(out["keys"]) >= 1)
    assert np.all(np.asarray(out["keys"]) < V // 2)
    assert np.all(np.asarray(out["values"]) >= V // 2)
    assert np.all(np.asarray(out["values"]) < V)
    # per-sequence keys and values are distinct (injective dictionary)
    for s in range(n):
        assert len(set(np.asarray(out["keys"][s]).tolist())) == kv
        assert len(set(np.asarray(out["values"][s]).tolist())) == kv


def test_mqar_task_semantics():
    """At every query position the target is the token that followed the
    key's first occurrence; everywhere else the target is IGNORE_INDEX."""
    key = jax.random.PRNGKey(1)
    n, N, kv, V = 3, 128, 16, 512
    out = generate_mqar(key, n, N, kv, vocab_size=V)
    tokens = np.asarray(out["tokens"])
    targets = np.asarray(out["targets"])
    qpos = np.asarray(out["query_pos"])
    for s in range(n):
        qset = set(qpos[s].tolist())
        for i in range(N):
            if i in qset:
                k_tok = tokens[s, i]
                first = int(np.argmax(tokens[s] == k_tok))
                assert first < i, "query precedes first occurrence"
                assert targets[s, i] == tokens[s, first + 1]
            else:
                assert targets[s, i] == IGNORE_INDEX
        # non-query tail slots are PAD
        tail = np.arange(2 * kv, N)
        non_query = [t for t in tail if t not in qset]
        assert np.all(tokens[s, non_query] == PAD_TOKEN)


def test_mqar_deterministic_and_powerlaw():
    key = jax.random.PRNGKey(2)
    a = generate_mqar(key, 2, 64, 4, vocab_size=128)
    b = generate_mqar(key, 2, 64, 4, vocab_size=128)
    assert np.array_equal(np.asarray(a["tokens"]), np.asarray(b["tokens"]))
    c = generate_mqar(
        key,
        2,
        64,
        4,
        vocab_size=128,
        gap_distribution="powerlaw",
        powerlaw_alpha=1.0,
    )
    assert c["tokens"].shape == (2, 64)


def test_mqar_validation_errors():
    key = jax.random.PRNGKey(3)
    with pytest.raises(ValueError):
        generate_mqar(key, 1, 16, 8, vocab_size=256)  # 2*8+8 > 16
    with pytest.raises(ValueError):
        generate_mqar(key, 1, 64, 8, vocab_size=10)  # vocab too small
    with pytest.raises(ValueError):
        generate_mqar(key, 1, 64, 4, vocab_size=64, gap_distribution="bogus")


def test_token_embeddings():
    key = jax.random.PRNGKey(4)
    emb = make_token_embeddings(key, 128, 16, scale=1.0)
    assert emb.shape == (128, 16)
    norms = jnp.linalg.norm(emb, axis=1)
    # norms concentrate around `scale`
    assert 0.5 < float(jnp.mean(norms)) < 1.5
