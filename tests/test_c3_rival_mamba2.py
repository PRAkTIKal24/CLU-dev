"""Tests for the **Mamba-2 tuned rival arm** (`c3-rival-mamba2`, task §A).

The arm is a *control*, so what has to be guarded is not its accuracy but its
honesty. Five things, each because it is a way the control could silently stop
being one:

1. **the byte reproduction** — the pinned config must reproduce
   :data:`chlu.eval.byte_ledger.RIVAL_SPECS`' 6,475,776 B **to the byte**, and
   the shrunk config must be the harness's *solved* knob, never a hand number;
2. **the provenance** — ⛔ TRAP 2: every state-bearing hyperparameter carries a
   ``PAPER:``/``OFFICIAL IMPLEMENTATION:`` string, and (uniquely for Mamba-2)
   **none** of them may claim a paper table, because the appendix was NOT
   OBTAINED;
3. **the recurrence** — the cell computes the official ``step()``: conv roll,
   selective ``(Delta, B, C)``, SSD update, ``D`` skip, gated RMSNorm. Asserted
   against an independent NumPy transcription, not against itself;
4. **the state** — the two-tensor inference cache (``ssm_state`` + ``conv_state``)
   is exactly what the ledger charges, and the learned initial state is
   PARAMETERS (PREREG-Bprime §4.1);
5. **the ladder seam** — the arm enters through ``build_byte_ledger`` with the
   shell bit-identical to every other arm's.
"""

from __future__ import annotations

import numpy as np
import pytest

BUDGET = 2_097_152
PINNED_BYTES_24L = 6_475_776
SHRUNK_BYTES_24L = 2_075_616


# ==========================================================================
# 1. the byte reproduction — the acceptance criterion
# ==========================================================================
def test_the_pinned_config_reproduces_RIVAL_SPECS_to_the_byte():
    from chlu.eval.byte_ledger import RIVAL_SPECS
    from chlu.eval.rivals.c3_mamba2 import PUBLISHED_D_STATE, Mamba2ArmConfig

    pub = Mamba2ArmConfig()._replace(d_state=PUBLISHED_D_STATE)
    assert pub.total_state_bytes() == RIVAL_SPECS["mamba2"].state_bytes()
    assert pub.total_state_bytes() == PINNED_BYTES_24L
    # ...and the two arithmetics agree per-element, not just in the total
    assert (pub.state_elements_per_layer() * pub.n_layers
            == RIVAL_SPECS["mamba2"].elements())


def test_the_shrink_is_the_harness_SOLVE_never_a_hand_number():
    """⭐ Task §1.4: use ``shrink_to_budget``'s value, state it, do not re-solve."""
    from chlu.eval.byte_ledger import shrink_to_budget
    from chlu.eval.rivals.c3_mamba2 import Mamba2ArmConfig, resolve_config

    sol = shrink_to_budget("mamba2", BUDGET)
    cfg = resolve_config({})
    assert cfg.d_state == sol["shrunk_value"] == 39
    assert cfg.total_state_bytes() == sol["state_bytes_shrunk"] == SHRUNK_BYTES_24L
    # shrunk, never grown, and it actually fits
    assert cfg.total_state_bytes() <= BUDGET < Mamba2ArmConfig()._replace(
        d_state=sol["published_value"]).total_state_bytes()
    assert sol["knob"] == "d_state"


def test_the_reference_row_carries_published_shrunk_and_the_solution():
    from chlu.eval.rivals.c3_mamba2 import reference_row

    r = reference_row(BUDGET)
    assert r["published"]["total_state_bytes"] == PINNED_BYTES_24L
    assert r["shrunk"]["total_state_bytes"] == SHRUNK_BYTES_24L
    assert r["published"]["occupancy"] == pytest.approx(3.0879, abs=1e-4)
    assert r["shrunk"]["occupancy"] == pytest.approx(0.98973, abs=1e-5)
    assert r["provenance_kind"] == "OFFICIAL IMPLEMENTATION"
    # ⚠ on the record: the paper's per-size appendix was NOT obtained
    assert r["paper_appendix_obtained"] is False


def test_the_deployed_row_states_the_shells_layer_count_not_the_rivals():
    """⚠ The shell holds 12 cells, the rival's reference geometry has 24, so the
    solved knob leaves the arm at ~half the ceiling. STATED, never re-solved."""
    from chlu.eval.rivals.c3_mamba2 import deployed_row, resolve_config

    cfg = resolve_config({})
    r12 = deployed_row(cfg, 12, BUDGET)
    assert r12["total_state_bytes"] == 1_037_808
    assert r12["occupancy"] == pytest.approx(0.49487, abs=1e-5)
    assert r12["within_budget"] and r12["reference_n_layers"] == 24
    assert deployed_row(cfg, 24, BUDGET)["total_state_bytes"] == SHRUNK_BYTES_24L


# ==========================================================================
# 2. ⛔ TRAP 2 — provenance, and no inherited library default
# ==========================================================================
def test_every_state_bearing_number_carries_a_provenance_string():
    from chlu.eval.rivals.c3_mamba2 import MAMBA2_PROVENANCE, Mamba2ArmConfig

    for k, v in MAMBA2_PROVENANCE.items():
        assert v.startswith(("PAPER:", "OFFICIAL IMPLEMENTATION:")), (k, v[:60])
    # every field of the pinned config is accounted for
    missing = set(Mamba2ArmConfig()._fields) - set(MAMBA2_PROVENANCE)
    assert not missing, missing


def test_mamba2s_provenance_is_CODE_and_never_claims_a_paper_table():
    """⚠ The scout could not obtain arXiv:2405.21060's per-size appendix (the PDF
    would not parse; ar5iv returned front matter only). Every string must say so
    by construction — a ``PAPER:`` prefix here would be a fabricated citation."""
    from chlu.eval.byte_ledger import RIVAL_SPECS
    from chlu.eval.rivals.c3_mamba2 import MAMBA2_PROVENANCE

    assert all(v.startswith("OFFICIAL IMPLEMENTATION:")
               for v in MAMBA2_PROVENANCE.values())
    spec = RIVAL_SPECS["mamba2"]
    assert "NOT OBTAINED" in spec.provenance and "CODE" in spec.provenance


def test_the_pinned_geometry_IS_the_official_implementations_defaults():
    """⛔ No library default is inherited implicitly — but every pinned number
    must still equal the official one, or the arm is not the rival."""
    from chlu.eval.byte_ledger import RIVAL_SPECS
    from chlu.eval.rivals.c3_mamba2 import resolve_config

    cfg = resolve_config({})
    pinned = RIVAL_SPECS["mamba2"].params
    for k in ("d_conv", "expand", "headdim", "ngroups"):
        assert getattr(cfg, k) == pinned[k], k
    # d_state is the ONLY knob that differs, and only because it was solved down
    assert cfg.d_state < pinned["d_state"]
    # the official block's structural defaults are on, not silently dropped
    assert cfg.conv_bias and cfg.rmsnorm and cfg.use_D and not cfg.proj_bias


def test_dtype_is_DECLARED_as_deployed_and_not_normalised():
    """⛔ Head+Advisor: TOTAL bytes AS DEPLOYED. bf16 for the rival, fp32 for our
    store — the asymmetry favours the rival and it stays."""
    import jax

    from chlu.eval.rivals.c3_mamba2 import Mamba2Cell, resolve_config

    cfg = resolve_config({})
    assert cfg.dtype_bytes == 2
    led = Mamba2Cell(cfg, latent_dim=12, key=jax.random.PRNGKey(0)).cell_ledger()
    assert led["dtype_bytes"] == 2
    assert led["state_bytes"] == 2 * led["state_floats"]


def test_an_unknown_or_impossible_override_fails_loudly():
    from chlu.eval.rivals.c3_mamba2 import resolve_config

    with pytest.raises(ValueError, match="unknown mamba2 arm config key"):
        resolve_config({"d_stat": 39})
    with pytest.raises(ValueError, match="not divisible by headdim"):
        resolve_config({"d_model": 100})       # 200 % 64 != 0
    with pytest.raises(ValueError, match="not divisible by\n?\\s*ngroups|ngroups"):
        resolve_config({"ngroups": 3})         # nheads 16 % 3 != 0


# ==========================================================================
# 3. the recurrence — asserted against an independent transcription
# ==========================================================================
def _numpy_step(cell, ssm, conv, z):
    """An independent NumPy transcription of ``mamba_ssm`` ``Mamba2.step()``.

    Written from the official control flow rather than from the module under
    test: in_proj -> split(z, xBC, dt) -> conv roll + silu -> split(x, B, C) ->
    dt = softplus(dt + dt_bias), dA = exp(dt * A), A = -exp(A_log) ->
    h <- h*dA + (dt x) B^T -> y = h C + D x -> gated RMSNorm -> out_proj.
    """
    cfg = cell.cfg
    W = np.asarray(cell.in_proj.weight)
    zx = W @ np.asarray(z, np.float64)
    di, cd, N, G = cfg.d_inner, cfg.conv_dim, cfg.d_state, cfg.ngroups
    H, P = cfg.n_heads, cfg.headdim
    z_gate, xBC, dt_raw = zx[:di], zx[di:di + cd], zx[di + cd:]
    win = np.concatenate([np.asarray(conv, np.float64), xBC[None, :]], 0)
    pre = (win * np.asarray(cell.conv_w, np.float64).T).sum(0) \
        + np.asarray(cell.conv_b, np.float64)
    xbc = pre / (1.0 + np.exp(-pre))                       # silu
    x, B, C = xbc[:di], xbc[di:di + G * N], xbc[di + G * N:]
    dt = np.log1p(np.exp(dt_raw + np.asarray(cell.dt_bias, np.float64)))
    dA = np.exp(dt * (-np.exp(np.asarray(cell.A_log, np.float64))))
    Bh = np.repeat(B.reshape(G, N), H // G, axis=0)
    Ch = np.repeat(C.reshape(G, N), H // G, axis=0)
    h = np.asarray(ssm, np.float64)
    y = np.einsum("hpn,hn->hp", h, Ch) + np.asarray(cell.D, np.float64)[:, None] \
        * x.reshape(H, P)
    g = y.reshape(di) * (z_gate / (1.0 + np.exp(-z_gate)))
    gg = g.reshape(G, di // G)
    gg = gg / np.sqrt((gg * gg).mean(-1, keepdims=True) + 1e-5)
    out = np.asarray(cell.out_proj.weight, np.float64) @ (
        gg.reshape(di) * np.asarray(cell.norm_w, np.float64))
    h_new = h * dA[:, None, None] + (x.reshape(H, P) * dt[:, None])[:, :, None] \
        * Bh[:, None, :]
    return out, h_new, win[1:]


def test_the_cell_computes_the_official_step():
    import jax

    from chlu.eval.rivals.c3_mamba2 import Mamba2Cell, resolve_config

    cfg = resolve_config({"d_model": 64})       # small but structurally identical
    cell = Mamba2Cell(cfg, latent_dim=6, key=jax.random.PRNGKey(3))
    st = cell.init_state()
    ssm, conv = np.asarray(st.ssm), np.asarray(st.conv)
    rng = np.random.default_rng(0)
    for _ in range(5):
        z = rng.normal(size=(6,)).astype(np.float32)
        r = np.asarray(cell.read(st, z))
        st = cell.write(st, z)
        r_ref, ssm, conv = _numpy_step(cell, ssm, conv, z)
        assert np.allclose(r, r_ref, atol=2e-5), np.abs(r - r_ref).max()
        assert np.allclose(np.asarray(st.ssm), ssm, atol=2e-5)
        assert np.allclose(np.asarray(st.conv), conv, atol=2e-6)


def test_the_conv_state_retains_exactly_the_d_conv_minus_1_past_taps():
    import jax
    import jax.numpy as jnp

    from chlu.eval.rivals.c3_mamba2 import Mamba2Cell, resolve_config

    cfg = resolve_config({"d_model": 64})
    cell = Mamba2Cell(cfg, latent_dim=6, key=jax.random.PRNGKey(1))
    st = cell.init_state()
    rng = np.random.default_rng(1)
    zs = [rng.normal(size=(6,)).astype(np.float32) for _ in range(5)]
    for z in zs:
        st = cell.write(st, z)
    # the retained window is the RAW pre-conv xBC of the last d_conv-1 chunks
    di, cd = cfg.d_inner, cfg.conv_dim
    want = np.stack([np.asarray(cell.in_proj(jnp.asarray(z)))[di:di + cd]
                     for z in zs[-(cfg.d_conv - 1):]])
    assert st.conv.shape == (cfg.d_conv - 1, cd)
    assert np.allclose(np.asarray(st.conv), want, atol=1e-6)


def test_the_recurrence_stays_finite_and_bounded_over_a_long_stream():
    """⚠ The lesson of the ttt_matched arm (DEFECT 1): an inner loop that is
    divergent by construction loses the rival column at step 135. Mamba-2's decay
    is ``exp(-dt exp(A_log)) in (0,1)``, so the state is a contraction — asserted,
    not assumed."""
    import jax

    from chlu.eval.rivals.c3_mamba2 import Mamba2Cell, resolve_config

    cell = Mamba2Cell(resolve_config({}), latent_dim=12, key=jax.random.PRNGKey(7))
    st = cell.init_state()
    rng = np.random.default_rng(2)
    norms = []
    for _ in range(200):
        z = (3.0 * rng.normal(size=(12,))).astype(np.float32)
        r = cell.read(st, z)
        st = cell.write(st, z)
        norms.append(float(np.linalg.norm(np.asarray(st.ssm))))
        assert np.isfinite(np.asarray(r)).all()
    assert np.isfinite(norms[-1]) and norms[-1] < 1e3, norms[-1]


# ==========================================================================
# 4. the state — the official two-tensor cache, and the learned-init rule
# ==========================================================================
def test_the_state_is_the_official_two_tensor_inference_cache():
    import jax

    from chlu.eval.rivals.c3_mamba2 import Mamba2Cell, resolve_config

    cfg = resolve_config({})
    cell = Mamba2Cell(cfg, latent_dim=12, key=jax.random.PRNGKey(0))
    st = cell.init_state()
    assert st.ssm.shape == (cfg.n_heads, cfg.headdim, cfg.d_state)
    assert st.conv.shape == (cfg.d_conv - 1, cfg.conv_dim)
    led = cell.cell_ledger()
    # ⛔ the ledger charges EXACTLY what the state pytree holds — no more, no less
    assert led["state_floats"] == st.ssm.size + st.conv.size
    assert led["state_breakdown"] == {"ssm_state": st.ssm.size,
                                      "conv_state": st.conv.size}


def test_the_learned_initial_state_is_PARAMETERS_not_STATE():
    """PREREG-Bprime §4.1, the rule the GRU's ``h0``, TTT's ``W0`` and the CLU's
    ``V0`` are all held to."""
    import equinox as eqx
    import jax

    from chlu.eval.rivals.c3_mamba2 import Mamba2Cell, resolve_config

    cell = Mamba2Cell(resolve_config({}), latent_dim=12, key=jax.random.PRNGKey(0))
    led = cell.cell_ledger()
    total = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(cell, eqx.is_inexact_array)))
    assert led["params"] == total
    assert led["learned_init_is_params"] is True
    # the initial state IS the parameter, bit for bit
    st = cell.init_state()
    assert st.ssm is cell.ssm0 and st.conv is cell.conv0


def test_the_cells_parameters_actually_receive_gradient():
    """A rival whose cell is inert would be a hobbled control, not a control."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from chlu.eval.rivals.c3_mamba2 import Mamba2Cell, resolve_config

    cell = Mamba2Cell(resolve_config({"d_model": 64}), latent_dim=6,
                      key=jax.random.PRNGKey(0))
    zs = jnp.asarray(np.random.default_rng(0).normal(size=(8, 6)), jnp.float32)

    def loss(c):
        def step(s, z):
            return c.write(s, z), c.read(s, z)
        _, rs = jax.lax.scan(step, c.init_state(), zs)
        return jnp.mean(rs ** 2)

    g = eqx.filter_grad(loss)(cell)
    for name in ("in_proj", "out_proj", "conv_w", "A_log", "dt_bias", "ssm0"):
        leaf = getattr(g, name)
        arr = leaf.weight if hasattr(leaf, "weight") else leaf
        assert np.isfinite(np.asarray(arr)).all(), name
        assert float(np.abs(np.asarray(arr)).max()) > 0.0, name


# ==========================================================================
# 5. the ladder seam — the arm enters ONLY through the byte ledger
# ==========================================================================
def test_the_registry_exposes_the_arm_and_refuses_a_duplicate_name():
    from chlu.eval.rivals.c3_registry import (C3RivalArm, c3_rival_names,
                                              is_c3_rival, register_c3_rival)

    assert "mamba2" in c3_rival_names() and is_c3_rival("mamba2")
    assert not is_c3_rival("clu_store")
    with pytest.raises(ValueError, match="already registered"):
        register_c3_rival(C3RivalArm(
            name="mamba2", spec_name="mamba2", resolve=lambda o: o,
            build=lambda c, **k: None, reference_row=dict, deployed_row=dict))


def test_the_arm_enters_the_ladder_through_the_byte_ledger():
    import jax

    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config
    from chlu.training.train_cluformer import solve_arms

    p = make_config("toy", 0)
    p.arms = ("clu_store", "mamba2", "none")
    p.d_model, p.n_layers, p.seq_len, p.batch = 32, 2, 128, 2
    specs, led = solve_arms(p, jax.random.PRNGKey(0))
    assert specs["mamba2"].rival_cfg is not None
    art = build_byte_ledger(p, led, p.arms)
    row = art["arms"]["mamba2"]
    assert row["within_budget"] and row["phi_accounted"]
    assert row["total_state_bytes"] == 2 * led["mamba2"]["state_bytes"]
    # the pinned reference table travels with every artifact
    assert art["rival_reference"]["rivals"]["mamba2"]["state_bytes"] == PINNED_BYTES_24L
    # and the arm's own row states the shell's deployment
    assert led["mamba2"]["deployed"]["deployed_n_layers"] == 2
    assert led["mamba2"]["provenance"] and led["mamba2"]["rival_config"]


def test_swapping_in_the_rival_leaves_the_shell_BIT_IDENTICAL():
    """The system-level swap is a swap: only ``block.cell`` may change."""
    import jax

    from chlu.experiments.exp_cluformer_pilot import make_config
    from chlu.training.train_cluformer import (assert_shared_shell_identical,
                                               build_arm, solve_arms)

    p = make_config("toy", 0)
    p.arms = ("clu_store", "mamba2", "none")
    p.d_model, p.n_layers, p.seq_len, p.batch = 32, 2, 128, 2
    specs, _ = solve_arms(p, jax.random.PRNGKey(0))
    models = {a: build_arm(a, p, specs, key=jax.random.PRNGKey(5)) for a in p.arms}
    shell = assert_shared_shell_identical(models)
    assert shell["shared_shell_params"] > 0


def test_the_arm_runs_the_stream_end_to_end_and_the_loss_is_finite():
    import jax
    import jax.numpy as jnp

    from chlu.experiments.exp_cluformer_pilot import make_config
    from chlu.training.train_cluformer import (build_arm, loss_fn, plan_pass,
                                               solve_arms)

    p = make_config("toy", 0)
    p.arms = ("clu_store", "mamba2")
    p.d_model, p.n_layers, p.seq_len, p.batch = 32, 2, 128, 2
    p.memory = dict(p.memory, chunk=32)
    specs, _ = solve_arms(p, jax.random.PRNGKey(0))
    m = build_arm("mamba2", p, specs, key=jax.random.PRNGKey(5))
    tok = np.random.default_rng(0).integers(0, p.vocab_size, (p.batch, p.seq_len))
    plans, _ = plan_pass(m, tok, p)
    lo = float(loss_fn(m, jnp.asarray(tok), jnp.asarray(tok), plans))
    assert np.isfinite(lo) and lo > 0.0
