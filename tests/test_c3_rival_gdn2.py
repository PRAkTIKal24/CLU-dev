"""Tests for the C3 **Gated DeltaNet-2 arm** (`c3-rival-arms` §B, arXiv:2605.22791).

Five things are guarded here, each because it is a way this arm could silently
stop being an admissible control:

1. **the byte ledger** — the arm reproduces
   ``chlu.eval.byte_ledger.RIVAL_SPECS["gated_deltanet2"]`` **to the byte** at the
   published pin, uses :func:`~chlu.eval.byte_ledger.shrink_to_budget`'s *solved*
   knob (never a hand-solved one), and declares ``dtype_bytes``, φ and the
   short-convolution state it carries **beyond** what ``RIVAL_SPECS`` counts;
2. ⚠⚠ **TRAP 2** — no library default is inherited, and the
   ``flash-linear-attention`` 3x is **re-derived** rather than quoted;
3. **the recurrence** — the ported ``jax.lax.scan`` agrees with the paper's boxed
   Eq. 10 **and** with a transcription of the reference Triton kernel's own
   statement order (the kernel is the shim's ground truth, not the prose);
4. **drop-in-ness** — the pilot's ``loss_fn`` / per-token NLL run on this arm
   unchanged, which is what lets the ladder consume it;
5. ⛔ **zero ladder arms trained** — the ladder rows are arithmetic on a config,
   and the test proves no model is constructed to produce them.
"""

from __future__ import annotations

import numpy as np
import pytest

BUDGET = 2_097_152


# ==========================================================================
# 1. the byte ledger — the only door into the ladder
# ==========================================================================
def test_published_pin_reproduces_rival_specs_to_the_byte():
    from chlu.eval.byte_ledger import RIVAL_SPECS
    from chlu.eval.rivals.gdn2_lm import (assert_reproduces_rival_specs,
                                          gdn2_published_config)

    cfg = gdn2_published_config()
    # the PAPER's own d_k = d_v = 128 survives the scaling to d_model 512
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads) == (24, 512, 4)
    assert cfg.head_k_dim == cfg.head_v_dim == 128
    row = assert_reproduces_rival_specs(cfg)
    assert row["main_recurrent_state_bytes"] == 3_145_728
    assert row["main_recurrent_state_bytes"] == RIVAL_SPECS[
        "gated_deltanet2"].state_bytes()
    assert row["rival_specs_delta_bytes"] == 0


def test_shrink_uses_the_SOLVED_knob_never_a_hand_solved_one():
    """⭐ Task §1.4: ``shrink_to_budget`` owns the solve; we consume it."""
    from chlu.eval.byte_ledger import shrink_to_budget
    from chlu.eval.rivals.gdn2_lm import gdn2_shrunk_config

    cfg, solved = gdn2_shrunk_config(BUDGET)
    ref = shrink_to_budget("gated_deltanet2", BUDGET)
    assert solved["knob"] == "n_heads" == ref["knob"]
    assert (solved["published_value"], solved["shrunk_value"]) == (4, 6)
    assert cfg.n_heads == ref["shrunk_value"] == 6
    # ⚠ the DISCLOSED coincidence: 24*512^2/6*2 B lands on the ceiling exactly.
    assert solved["ledger_ideal_bytes"] == ref["state_bytes_shrunk"] == BUDGET


def test_shrunk_geometry_has_no_integer_realisation_and_lands_UNDER_the_ceiling():
    """⚠⚠ P3, disclosed rather than smoothed over.

    ``RIVAL_SPECS``' formula ties ``d_k = d_v = d_model/H``; at the solved
    ``H = 6`` that tie is ``512/6 = 85.33…``, so the ledger's ideal
    ``2,097,152 B`` is **not realizable**. The floor is what deploys, and it is
    smaller — the shrink direction is preserved.
    """
    from chlu.eval.rivals.gdn2_lm import gdn2_shrunk_config, main_recurrent_state_bytes

    cfg, solved = gdn2_shrunk_config(BUDGET)
    assert solved["integer_geometry_exact"] is False
    assert solved["realized_head_dim"] == cfg.head_k_dim == 85
    assert main_recurrent_state_bytes(cfg) == solved["realized_bytes"] == 2_080_800
    assert solved["realized_minus_ideal_bytes"] == -16_352
    # ⛔ never grows past the ceiling; that is the whole direction of the control
    assert solved["realized_bytes"] < BUDGET
    assert 0.99 < solved["realized_occupancy"] < 1.0


def test_short_conv_state_is_ledgered_SEPARATELY_and_the_total_is_declared():
    """⚠ P4 — a STOP for `c3-gb-landing`, surfaced as three explicit columns.

    The official layer caches ``(conv_state_q, conv_state_k, conv_state_v)``
    beside ``recurrent_state``; ``RIVAL_SPECS`` counts only the paper's **main
    recurrent state**. Both conventions are reported; neither is chosen here.
    """
    from chlu.eval.rivals.gdn2_lm import gdn2_ledger_row, gdn2_shrunk_config

    cfg, _ = gdn2_shrunk_config(BUDGET)
    row = gdn2_ledger_row(cfg)
    assert row["conv_state_bytes"] == 220_320
    assert row["total_state_bytes_as_deployed"] == 2_080_800 + 220_320 == 2_301_120
    # the two conventions genuinely disagree, and the row says so both ways
    assert row["within_budget_main"] is True
    assert row["within_budget_total_as_deployed"] is False
    assert row["occupancy_total_as_deployed"] > 1.0
    # turning the convolution off is the ONLY thing that zeroes it, and it is a
    # declared config flag, not a silent omission
    from dataclasses import replace
    assert gdn2_ledger_row(replace(cfg, use_short_conv=False))["conv_state_bytes"] == 0


def test_ledger_row_declares_dtype_bytes_and_accounts_phi_as_an_explicit_zero():
    from chlu.eval.rivals.gdn2_lm import gdn2_ledger_row, gdn2_published_config

    row = gdn2_ledger_row(gdn2_published_config())
    assert row["dtype_bytes"] == 2                      # bf16 AS DEPLOYED
    assert row["phi_accounted"] is True
    assert row["phi_state_bytes"] == 0 and row["phi_params_bytes"] == 0
    assert "standalone" in row["phi_note"]
    assert "2605.22791" in row["citation"] and "2412.06464" in row["citation"]
    assert "24 * 4 * 128 * 128" in row["arithmetic"]


def test_an_arm_that_cannot_reproduce_rival_specs_FAILS_LOUDLY():
    import jax

    from chlu.eval.rivals.gdn2_lm import (GDN2Config, UnledgeredGDN2Error,
                                          assert_reproduces_rival_specs,
                                          build_gdn2_arm)

    bad = GDN2Config(n_layers=24, d_model=512, n_heads=4, head_dim=256)  # FLA's
    with pytest.raises(UnledgeredGDN2Error, match="cannot enter the ladder"):
        assert_reproduces_rival_specs(bad)
    small = GDN2Config(n_layers=1, d_model=8, n_heads=2)
    with pytest.raises(UnledgeredGDN2Error):
        build_gdn2_arm(small, key=jax.random.PRNGKey(0), check_ledger=True)


# ==========================================================================
# 2. ⚠⚠ TRAP 2 — no library default is inherited
# ==========================================================================
def test_the_fla_3x_trap_is_RE_DERIVED_not_quoted():
    from chlu.eval.rivals.gdn2_lm import fla_trap_check

    t = fla_trap_check()
    assert t["fla_state_floats_per_layer"] == 6 * 256 * 512 == 786_432
    assert t["fla_over_paper"] == 3.0
    # ⭐ and the finding: NVlabs' OWN layer defaults are not the trap
    assert t["official_state_floats_per_layer"] == 262_144
    assert t["official_over_paper"] == 1.0


def test_no_pinned_value_equals_a_flash_linear_attention_default():
    from chlu.eval.rivals.gdn2_lm import gdn2_published_config, gdn2_shrunk_config

    for cfg in (gdn2_published_config(), gdn2_shrunk_config(BUDGET)[0]):
        assert cfg.head_k_dim != 256          # FLA head_dim
        assert cfg.expand_v != 2              # FLA expand_v
    # ⚠ the shrunk arm's n_heads IS 6, which is also FLA's default — and that is
    # a coincidence of the SOLVE, not an inheritance. Pinning the other two is
    # what makes the state 1/3 of FLA's, so assert the consequence.
    cfg, _ = gdn2_shrunk_config(BUDGET)
    assert cfg.n_heads == 6 and cfg.head_k_dim == 85 and cfg.expand_v == 1.0


def test_every_config_field_carries_an_admissible_provenance_string():
    from dataclasses import fields

    from chlu.eval.rivals.gdn2_lm import (GDN2_PROVENANCE, GDN2Config,
                                          PROVENANCE_PREFIXES)

    names = {f.name for f in fields(GDN2Config)}
    assert names <= set(GDN2_PROVENANCE), names - set(GDN2_PROVENANCE)
    for k, v in GDN2_PROVENANCE.items():
        assert v.startswith(PROVENANCE_PREFIXES), (k, v[:40])
    # the rival's OWN state-bearing numbers must be sourced to the rival, never
    # to us: only the weight class we chose may carry HARNESS LEDGER.
    ours = {k for k, v in GDN2_PROVENANCE.items()
            if v.startswith("HARNESS LEDGER:")}
    assert ours == {"n_layers", "d_model", "vocab_size", "dtype_bytes"}, ours
    for k in ("n_heads", "head_dim", "expand_v"):
        assert GDN2_PROVENANCE[k].startswith(("PAPER:", "OFFICIAL IMPLEMENTATION:"))
        assert "2605.22791" in GDN2_PROVENANCE[k] or "NVlabs" in GDN2_PROVENANCE[k]


def test_grouped_value_attention_is_REFUSED_not_silently_ignored():
    from chlu.eval.rivals.gdn2_lm import GDN2Config

    with pytest.raises(NotImplementedError, match="grouped value attention"):
        GDN2Config(n_layers=1, d_model=64, n_heads=2, num_v_heads=4).validate()


# ==========================================================================
# 3. the recurrence — against the paper AND against the reference kernel
# ==========================================================================
def _inputs(T=7, H=2, dk=3, dv=3, seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(size=s).astype(np.float64) for k, s in
            (("q", (T, H, dk)), ("k", (T, H, dk)), ("v", (T, H, dv)))} | {
        "g": -np.abs(rng.normal(size=(T, H, dk))).astype(np.float64),
        "b": rng.uniform(0, 1, size=(T, H, dk)),
        "w": rng.uniform(0, 1, size=(T, H, dv)),
    }


def _paper_eq10(inp, scale):
    """PAPER Eq. 10 boxed form, written as matrices, one head at a time.

    ``S_t = (I - k_t (b_t ⊙ k_t)^T) D_t S_{t-1} + k_t (w_t ⊙ v_t)^T``,
    ``o_t = S_t^T q_t``.
    """
    T, H, dk = inp["k"].shape
    dv = inp["v"].shape[-1]
    out = np.zeros((T, H, dv))
    l2 = lambda x: x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)  # noqa: E731
    q, k = l2(inp["q"]) * scale, l2(inp["k"])
    for h in range(H):
        S = np.zeros((dk, dv))
        for t in range(T):
            D = np.diag(np.exp(inp["g"][t, h]))
            kt = k[t, h]
            e = inp["b"][t, h] * kt
            z = inp["w"][t, h] * inp["v"][t, h]
            S = (np.eye(dk) - np.outer(kt, e)) @ D @ S + np.outer(kt, z)
            out[t, h] = S.T @ q[t, h]
    return out


def _kernel_transcription(inp, scale):
    """``fused_recurrent_gdn2_fwd_kernel``'s own statements, in its own order."""
    T, H, dk = inp["k"].shape
    dv = inp["v"].shape[-1]
    out = np.zeros((T, H, dv))
    for h in range(H):
        b_h = np.zeros((dk, dv))
        for t in range(T):
            b_q, b_k = inp["q"][t, h].copy(), inp["k"][t, h].copy()
            b_q = b_q / np.sqrt((b_q * b_q).sum() + 1e-6)
            b_k = b_k / np.sqrt((b_k * b_k).sum() + 1e-6)
            b_q = b_q * scale
            b_h = b_h * np.exp(inp["g"][t, h])[:, None]
            b_bk = inp["b"][t, h] * b_k
            erase_d = (b_h * b_bk[:, None]).sum(0)
            b_v_new = inp["w"][t, h] * inp["v"][t, h] - erase_d
            b_h = b_h + b_k[:, None] * b_v_new[None, :]
            out[t, h] = (b_h * b_q[:, None]).sum(0)
    return out


def test_ported_recurrence_matches_the_paper_and_the_reference_kernel():
    import jax
    import jax.numpy as jnp

    from chlu.eval.rivals.gdn2_lm import GDN2Config, GDN2Layer

    cfg = GDN2Config(n_layers=1, d_model=6, n_heads=2, head_dim=3)
    layer = GDN2Layer(cfg, key=jax.random.PRNGKey(0))
    inp = _inputs()
    scale = 3.0 ** -0.5
    got = np.asarray(layer._recur(*(jnp.asarray(inp[k], jnp.float32)
                                    for k in ("q", "k", "v", "g", "b", "w"))))
    np.testing.assert_allclose(got, _paper_eq10(inp, scale), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got, _kernel_transcription(inp, scale),
                               rtol=2e-5, atol=2e-5)


def test_read_is_from_the_UPDATED_state_not_the_decayed_one():
    """PAPER §3.1: 'The output is o_t = S_t^T q_t' — read AFTER the write.

    Reading ``S_{t-1}`` instead would make a memory that stores nothing score
    like one that stores everything on the first token.
    """
    import jax
    import jax.numpy as jnp

    from chlu.eval.rivals.gdn2_lm import GDN2Config, GDN2Layer

    cfg = GDN2Config(n_layers=1, d_model=4, n_heads=1, head_dim=4)
    layer = GDN2Layer(cfg, key=jax.random.PRNGKey(1))
    one = {"q": np.ones((1, 1, 4)), "k": np.ones((1, 1, 4)),
           "v": np.full((1, 1, 4), 2.0), "g": np.zeros((1, 1, 4)),
           "b": np.ones((1, 1, 4)), "w": np.ones((1, 1, 4))}
    o = np.asarray(layer._recur(*(jnp.asarray(one[k], jnp.float32)
                                  for k in ("q", "k", "v", "g", "b", "w"))))
    assert np.abs(o).max() > 1e-3, "first-token read is zero => read-before-write"


def test_negative_eigenvalue_variant_scales_ONLY_the_erase_gate():
    """PAPER §3.1 / App. C.1: b -> [0,2]^{d_k}; w stays in [0,1]^{d_v}."""
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from chlu.eval.rivals.gdn2_lm import GDN2Config, GDN2Layer

    cfg = GDN2Config(n_layers=1, d_model=8, n_heads=2, head_dim=4)
    off = GDN2Layer(cfg, key=jax.random.PRNGKey(3))
    on = GDN2Layer(replace(cfg, allow_neg_eigval=True), key=jax.random.PRNGKey(3))
    x = jax.random.normal(jax.random.PRNGKey(4), (5, 8))
    a, b = np.asarray(off(x)), np.asarray(on(x))
    assert not np.allclose(a, b)                       # the gate really moves
    # the two share every parameter: only the erase scale differs
    import equinox as eqx
    la = jax.tree_util.tree_leaves(eqx.filter(off, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(on, eqx.is_inexact_array))
    assert all(bool(jnp.all(u == v)) for u, v in zip(la, lb, strict=True))


# ==========================================================================
# 4. drop-in-ness for the ladder
# ==========================================================================
def test_arm_is_drop_in_for_the_pilots_loss_and_per_token_nll():
    import jax
    import jax.numpy as jnp

    from chlu.eval.rivals.gdn2_lm import GDN2Config, build_gdn2_arm
    from chlu.training.train_cluformer import loss_fn, token_nll

    m = build_gdn2_arm(GDN2Config(n_layers=1, d_model=16, n_heads=2),
                       key=jax.random.PRNGKey(0), check_ledger=False)
    tk = jnp.asarray(np.arange(2 * 12).reshape(2, 12) % 256, jnp.int32)
    tg = jnp.asarray((np.arange(2 * 12).reshape(2, 12) + 1) % 256, jnp.int32)
    ll = float(loss_fn(m, tk, tg, []))
    pt = np.asarray(token_nll(m, tk, tg, []))
    assert np.isfinite(ll) and pt.shape == (2, 12)
    np.testing.assert_allclose(pt.mean(), ll, rtol=1e-5)


def test_measured_params_match_the_config_arithmetic_exactly():
    import jax

    from chlu.eval.rivals.gdn2_lm import GDN2Config, build_gdn2_arm

    for cfg in (GDN2Config(n_layers=2, d_model=32, n_heads=2),
                GDN2Config(n_layers=1, d_model=24, n_heads=3, use_mlp=False),
                GDN2Config(n_layers=1, d_model=30, n_heads=6, use_short_conv=False)):
        m = build_gdn2_arm(cfg, key=jax.random.PRNGKey(0), check_ledger=False)
        row = m.ledger_row()
        assert row["params_measured_matches_arithmetic"], (
            cfg, row["params_measured"], row["params_total"])


def test_the_param_class_conflict_is_REPORTED_not_hidden():
    """⚠ P6: 24 L + the paper's block is ~95 M, i.e. outside 26-47 M."""
    from chlu.eval.rivals.gdn2_lm import GDN2_PARAM_CLASS_NOTE, gdn2_param_class_table

    t = gdn2_param_class_table()
    assert t["with_mlp_24L"] == 95_374_944
    assert t["mlp_free_24L"] == 44_490_336
    assert t["in_class"] == {"with_mlp_24L": False, "mlp_free_24L": True}
    # halving the layers halves the STATE too — the resolutions are not free
    assert t["half_layers_state_bytes"] == 1_572_864
    assert "jointly infeasible" in GDN2_PARAM_CLASS_NOTE
    assert "jointly infeasible" in t["note"]


def test_the_arm_declares_its_sanity_anchor_and_that_it_is_not_a_baseline():
    from chlu.eval.rivals.gdn2_lm import GDN2_ARM

    a = GDN2_ARM["sanity_anchor"]
    assert "never a matched baseline" in a
    assert "15.90" in a                                # the rival-vs-rival number
    assert GDN2_ARM["rival_spec"] == "gated_deltanet2"


# ==========================================================================
# 5. ⛔ zero ladder arms trained
# ==========================================================================
def test_the_ladder_rows_are_ARITHMETIC_and_build_no_model(monkeypatch):
    """⛔ Task §1.7: the interim-budget guard blocks ladder arms, deliberately."""
    import chlu.eval.rivals.gdn2_lm as g
    from chlu.experiments.exp_c3_rival_gdn2 import ladder_ledger

    def _boom(*a, **k):                       # pragma: no cover - must not run
        raise AssertionError("⛔ a ladder arm was CONSTRUCTED")

    monkeypatch.setattr(g, "GDN2LM", _boom)
    monkeypatch.setattr(g, "build_gdn2_arm", _boom)
    led = ladder_ledger()
    assert led["published"]["reproduces_rival_specs"] is True
    assert led["shrunk"]["main_recurrent_state_bytes"] == 2_080_800
    assert led["trained_by_this_spoke"] is False


def test_smoke_leg_trains_checkpoints_resumes_ledgers_and_slices(tmp_path):
    """The §1.6 acceptance path, in miniature, on a synthetic byte stream.

    ⛔ Nothing here is a claim; the assertions are all mechanical (the loss
    moves, the resume is **bit-identical**, the ledger and slice keys exist).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from chlu.data.enwik8 import Enwik8Split
    from chlu.eval.rivals.gdn2_lm import GDN2Config, build_gdn2_arm
    from chlu.experiments.exp_c3_rival_gdn2 import (evaluate, ladder_ledger,
                                                    slices, train)
    from chlu.utils.checkpoints import load_model, save_model

    # ⚠ TRAP 1: the revisit unit is the enclosing WHITESPACE-DELIMITED token, so
    # the stream must have words — a wall of letters degenerates the slice and
    # `assert_non_degenerate` (rightly) refuses it.
    rng = np.random.default_rng(0)
    vocab = [bytes(rng.integers(97, 123, size=int(n), dtype=np.uint8))
             for n in rng.integers(3, 12, size=400)]
    chunks, n = [], 0
    while n < 40_000:
        if n % 4_000 < 12:
            chunks.append(b"<page> ")
            n += 7
        wtok = vocab[int(rng.integers(0, len(vocab)))]
        chunks.append(wtok + b" ")
        n += len(wtok) + 1
    body = np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()
    split = Enwik8Split("valid", body)

    cfg = GDN2Config(n_layers=1, d_model=16, n_heads=2, vocab_size=256)
    m0 = build_gdn2_arm(cfg, key=jax.random.PRNGKey(0), check_ledger=False)
    full = train(m0, split, steps=4, batch=2, seq_len=64, lr=1e-2, seed=0)
    assert full["losses"][-1] < full["losses"][0]      # it trains

    half = train(m0, split, steps=2, batch=2, seq_len=64, lr=1e-2, seed=0)
    save_model(half["model"], tmp_path / "m.pkl")
    save_model(half["opt_state"], tmp_path / "o.pkl")
    rest = train(load_model(tmp_path / "m.pkl"), split, steps=4, batch=2,
                 seq_len=64, lr=1e-2, seed=0, start_step=2,
                 opt_state=load_model(tmp_path / "o.pkl"))
    la = jax.tree_util.tree_leaves(eqx.filter(full["model"], eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(rest["model"], eqx.is_inexact_array))
    assert all(bool(jnp.all(u == v)) for u, v in zip(la, lb, strict=True)), \
        "resume is not bit-identical"

    ev = evaluate(full["model"], split, batch=2, seq_len=64, n_batches=2)
    assert np.isfinite(ev["bpc"]) and ev["n_tokens"] == 2 * 64 * 2
    assert "NOT_A_CLAIM" in ev

    sl = slices(full["model"], split, corpus="enwik8", batch=2, seq_len=64,
                n_batches=2, min_n=1)
    assert sl["bins"] and "controls" in sl
    # not every target position is binned (the unit is the enclosing token), so
    # the scored count is positive and bounded by the tokens actually evaluated
    assert 0 < sl["n_scored"] <= 2 * 64 * 2
    assert any(b["n"] > 0 for b in sl["bins"].values())
    assert sl["controls"]["non_degeneracy"]["passed"] is True

    led = ladder_ledger()
    assert led["published"]["rival_specs_delta_bytes"] == 0
