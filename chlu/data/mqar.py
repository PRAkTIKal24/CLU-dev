"""Multi-Query Associative Recall (MQAR) data generator.

Follows the Zoology task spec (Arora et al. 2023, arXiv:2312.04927):
sequences over a vocabulary C contain key-value pairs; whenever a key token
reoccurs later in the sequence (a "query"), the correct output at that
position is the token that followed the key's *first* occurrence. Difficulty
knobs: sequence length N, number of KV pairs, vocab size, and the key->query
gap distribution (uniform, or power-law as in the "Based" follow-up).

Layout (matches the published task semantics):
    [k_1 v_1 k_2 v_2 ... k_kv v_kv | tail: queries at sampled positions,
     remaining tail slots filled with the PAD token 0]
    target[i] = value bound to tokens[i] if position i is a query, else -100.

Implementation choices (documented deviations, none change task semantics):
- Vocab is split in half: keys are drawn (without replacement, per sequence)
  from [1, vocab_size // 2), values from [vocab_size // 2, vocab_size).
  Token 0 is reserved as PAD. Zoology's main experiments use vocab 8192;
  scale down for small models and document it.
- Values are drawn without replacement so each per-sequence dictionary is
  injective (keeps retrieval-correctness decoding unambiguous).
- Each stored key is queried exactly `num_queries_per_key` times (default 1,
  as in Zoology's multiquery layout where every pair is queried).
"""

import jax
import jax.numpy as jnp

IGNORE_INDEX = -100
PAD_TOKEN = 0


def generate_mqar(
    key: jax.random.PRNGKey,
    n_sequences: int,
    seq_len: int,
    num_kv_pairs: int,
    vocab_size: int = 8192,
    num_queries_per_key: int = 1,
    gap_distribution: str = "uniform",
    powerlaw_alpha: float = 0.01,
) -> dict:
    """
    Generate a batch of MQAR sequences.

    Args:
        key: JAX random key
        n_sequences: Number of sequences to generate
        seq_len: Sequence length N (Zoology sweeps N in {64..512})
        num_kv_pairs: Number of key-value pairs stored per sequence
        vocab_size: Vocabulary size c (Zoology default 8192)
        num_queries_per_key: How many times each key is queried (default 1)
        gap_distribution: "uniform" or "powerlaw" — distribution of query
            positions in the tail. "powerlaw" weights a tail slot at distance
            g from the key block by g^(-powerlaw_alpha) ("Based"-style gaps).
        powerlaw_alpha: Exponent for the power-law gap distribution.

    Returns:
        dict with:
            tokens:    (n_sequences, seq_len) int32 token ids
            targets:   (n_sequences, seq_len) int32, value id at query
                       positions, IGNORE_INDEX (-100) elsewhere
            keys:      (n_sequences, num_kv_pairs) key tokens (first-occurrence order)
            values:    (n_sequences, num_kv_pairs) bound value tokens
            query_pos: (n_sequences, num_kv_pairs * num_queries_per_key)
                       positions of the queries
            query_key_idx: same shape — index into `keys` of each query
    """
    if vocab_size < 2 * num_kv_pairs + 2:
        raise ValueError(
            f"vocab_size={vocab_size} too small for {num_kv_pairs} KV pairs "
            "(need distinct keys and values in each half-vocab)."
        )
    n_queries = num_kv_pairs * num_queries_per_key
    kv_block = 2 * num_kv_pairs
    tail_len = seq_len - kv_block
    if tail_len < n_queries:
        raise ValueError(
            f"seq_len={seq_len} too short: kv block {kv_block} + "
            f"{n_queries} queries do not fit."
        )
    if gap_distribution not in ("uniform", "powerlaw"):
        raise ValueError(f"Unknown gap_distribution: {gap_distribution}")

    key_vocab_lo, key_vocab_hi = 1, vocab_size // 2
    val_vocab_lo, val_vocab_hi = vocab_size // 2, vocab_size

    def make_one(seq_key):
        k1, k2, k3 = jax.random.split(seq_key, 3)

        # Per-sequence injective dictionary: distinct keys, distinct values.
        keys_tok = jax.random.choice(
            k1,
            jnp.arange(key_vocab_lo, key_vocab_hi),
            shape=(num_kv_pairs,),
            replace=False,
        )
        vals_tok = jax.random.choice(
            k2,
            jnp.arange(val_vocab_lo, val_vocab_hi),
            shape=(num_kv_pairs,),
            replace=False,
        )

        # KV block: k_1 v_1 k_2 v_2 ...
        kv_block_tokens = jnp.stack([keys_tok, vals_tok], axis=1).reshape(-1)

        # Which key each query asks about (each key queried num_queries_per_key times)
        query_key_idx = jnp.tile(jnp.arange(num_kv_pairs), num_queries_per_key)

        # Query positions: n_queries distinct slots in the tail [kv_block, seq_len).
        # Gumbel-top-k over per-slot log-weights = sampling without replacement.
        slots = jnp.arange(tail_len)
        if gap_distribution == "powerlaw":
            # Distance of tail slot from the KV block (>= 1)
            log_w = -powerlaw_alpha * jnp.log(slots.astype(jnp.float32) + 1.0)
        else:
            log_w = jnp.zeros(tail_len)
        gumbel = jax.random.gumbel(k3, (tail_len,))
        _, slot_idx = jax.lax.top_k(log_w + gumbel, n_queries)
        query_pos = kv_block + jnp.sort(slot_idx)

        # Assemble tokens: pad tail, scatter query key-tokens at query_pos
        tail = jnp.full((tail_len,), PAD_TOKEN, dtype=keys_tok.dtype)
        tokens = jnp.concatenate([kv_block_tokens, tail])
        tokens = tokens.at[query_pos].set(keys_tok[query_key_idx])

        # Targets: value token at each query position, IGNORE_INDEX elsewhere
        targets = jnp.full((seq_len,), IGNORE_INDEX, dtype=jnp.int32)
        targets = targets.at[query_pos].set(vals_tok[query_key_idx].astype(jnp.int32))

        return tokens, targets, keys_tok, vals_tok, query_pos, query_key_idx

    seq_keys = jax.random.split(key, n_sequences)
    tokens, targets, keys_tok, vals_tok, query_pos, query_key_idx = jax.vmap(make_one)(
        seq_keys
    )

    return {
        "tokens": tokens.astype(jnp.int32),
        "targets": targets,
        "keys": keys_tok.astype(jnp.int32),
        "values": vals_tok.astype(jnp.int32),
        "query_pos": query_pos.astype(jnp.int32),
        "query_key_idx": query_key_idx.astype(jnp.int32),
    }


def make_token_embeddings(
    key: jax.random.PRNGKey,
    vocab_size: int,
    embed_dim: int,
    scale: float = 1.0,
) -> jnp.ndarray:
    """
    Fixed random Gaussian token embeddings for energy-based retrieval.

    Entries ~ N(0, (scale/sqrt(embed_dim))^2), so each embedding has norm
    ~= scale and entries stay well inside [-1, 1] (compatible with the
    generative-PCD training path's pixel clamp).

    Returns:
        (vocab_size, embed_dim) embedding table.
    """
    return jax.random.normal(key, (vocab_size, embed_dim)) * scale / jnp.sqrt(embed_dim)
