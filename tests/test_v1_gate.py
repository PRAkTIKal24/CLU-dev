"""Smoke tests for the V1-gate cascade machinery (exp_v1_gate)."""

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import ExperimentV1GateConfig
from chlu.core.chlu_unit import CHLU
from chlu.core.transforms import effective_mass
from chlu.experiments.exp_v1_gate import _run_cascade, _simulate_tau_policy


def _tiny_cfg():
    cfg = ExperimentV1GateConfig()
    cfg.dt = 0.05
    cfg.relax_steps = 6
    cfg.retry_relax_steps = 4
    cfg.retry_budget = 2
    cfg.zeta_grid = [-0.3, 0.3]
    cfg.zeta_scale_per_retry = 1.5
    cfg.governor_sensitivity = 0.9
    return cfg


def test_cascade_records_shapes_and_cost():
    key = jax.random.PRNGKey(0)
    e, dim, T = 2, 4, 3
    model = CHLU(dim=dim, hidden=8, kinetic_mode="relativistic", key=key)
    m_eff = effective_mass(model)
    kq, kv_, kt = jax.random.split(key, 3)
    q0 = jax.random.normal(kq, (T, dim)) * 0.3
    p0 = jnp.zeros((T, dim))
    val_embeds = jax.random.normal(kv_, (8, e)) * 0.5
    val_tokens = jnp.arange(100, 108)
    true_tok = jnp.array([100, 101, 102])
    cfg = _tiny_cfg()

    for arm, kick in [("mass", None), ("raw", None), ("kick", kt)]:
        rec = _run_cascade(
            model,
            m_eff,
            q0,
            p0,
            true_tok,
            val_embeds,
            val_tokens,
            e,
            floor=0.0,
            cfg=cfg,
            arm=arm,
            select_by="energy",
            kick_key=kick,
        )
        B1 = cfg.retry_budget + 1
        assert rec["R"].shape == (T, B1)
        assert rec["margin"].shape == (T, B1)
        assert rec["correct"].shape == (T, B1)
        assert rec["correct"].dtype == bool
        assert rec["cost"].shape == (B1,)
        # cost: relax-1, then + G*(retry-1) per stage
        G = len(cfg.zeta_grid)
        expected = [cfg.relax_steps - 1]
        for _ in range(cfg.retry_budget):
            expected.append(expected[-1] + G * (cfg.retry_relax_steps - 1))
        assert rec["cost"].tolist() == expected
        # best-so-far residual is non-increasing for energy selection
        assert np.all(np.diff(rec["R"], axis=1) <= 1e-5)
        if arm == "mass":
            assert "scatter" in rec
            assert rec["scatter"]["dq_total"].shape == (T, dim)


def test_simulate_tau_policy_hand_case():
    score = np.array([[5.0, 1.0], [0.1, 0.1]])
    correct = np.array([[False, True], [True, True]])
    cost = np.array([10, 20])
    acc, cst = _simulate_tau_policy(score, correct, cost, [0.5, 10.0])
    # tau=0.5: trial0 never passes -> full budget (correct, 20);
    #          trial1 passes at stage0 (correct, 10)
    assert acc[0] == 1.0 and cst[0] == 15.0
    # tau=10: both exit at stage 0 -> acc 0.5, cost 10
    assert acc[1] == 0.5 and cst[1] == 10.0
