#!/usr/bin/env python
"""L. The structural ceiling of the REGISTERED I2 decision rule, and the price
of a rig that could answer it. Simulation only — no model, no training.

Run: /Users/user/Desktop/CHLU/.venv/bin/python \
       .claude/outputs/c2w6-erosion-adjudication/power_pricing.py
"""
import numpy as np

RNG = np.random.default_rng(20260805)
B = 200000


def sim_mean_rho(rho_s, n, seeds, rel=1.0):
    """Mean-of-`seeds` Spearman rho, n items/seed, proxy reliability `rel`.

    Reliability enters as classical attenuation: the observed usefulness is the
    true one plus independent noise with variance (1-rel)/rel, so the observed
    correlation is rho_true * sqrt(rel).
    """
    pear = 2 * np.sin(np.pi * rho_s / 6)
    z1 = RNG.standard_normal((B, seeds, n))
    y = pear * z1 + np.sqrt(max(1 - pear ** 2, 0)) * RNG.standard_normal((B, seeds, n))
    if rel < 1.0:
        z1 = np.sqrt(rel) * z1 + np.sqrt(1 - rel) * RNG.standard_normal((B, seeds, n))
    rx = np.argsort(np.argsort(z1, -1), -1) + 1.0
    ry = np.argsort(np.argsort(y, -1), -1) + 1.0
    rx -= rx.mean(-1, keepdims=True)
    ry -= ry.mean(-1, keepdims=True)
    rho = (rx * ry).sum(-1) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))
    return rho.mean(-1)


def main():
    print("L1. The registered rule 'CONFIRM iff observed mean rho >= +0.5' —")
    print("    P(confirm) when the TRUE rho is exactly the registered +0.5")
    print("    (rel = read-selection reliability measured on the primary arm)")
    for n, seeds, rel in [(6, 3, 1.0), (6, 3, 0.648), (20, 3, 1.0),
                          (20, 3, 0.648), (100, 3, 1.0), (100, 10, 1.0),
                          (6, 30, 1.0)]:
        m = sim_mean_rho(0.5, n, seeds, rel)
        print("      n_wells=%3d seeds=%2d rel=%.3f -> mean %+.3f sd %.3f | "
              "P(confirm) %.3f | P(read NO_STRUCTURE) %.3f"
              % (n, seeds, rel, m.mean(), m.std(), np.mean(m >= 0.5),
                 np.mean(np.abs(m) < 0.3)))
    print("    => the rule thresholds AT the effect size, so P(confirm) is")
    print("       capped near 0.5 by construction and MORE DATA DOES NOT FIX IT.")

    print("\nL2. False-refutation rate of the registered rule "
          "('REFUTE iff mean rho <= -0.3') under H0 (true rho = 0)")
    for n, seeds in [(6, 3), (20, 3), (100, 3)]:
        m = sim_mean_rho(0.0, n, seeds)
        print("      n_wells=%3d seeds=%2d -> P(false refute) %.3f | "
              "P(false confirm) %.3f" % (n, seeds, np.mean(m <= -0.3),
                                         np.mean(m >= 0.5)))

    print("\nL3. A rule that CAN be answered: 'rho significantly > 0' "
          "(Fisher-z, two-sided 5%, power 80%) at true rho_s = 0.5")
    for rel in (1.0, 0.648):
        z = np.arctanh(0.5 * np.sqrt(rel))
        need = (2.80 / z) ** 2      # = seeds * (n-3)
        print("      rel=%.3f: need seeds*(n_wells-3) >= %.1f  ->  "
              "3 seeds needs n_wells >= %.0f ; 6 wells needs seeds >= %.0f"
              % (rel, need, need / 3 + 3, need / 3))

    print("\nL4. What the CURRENT rig can and cannot separate (primary arm, "
          "read-selection, n=6, 3 seeds, rel=0.648)")
    for t in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7):
        m = sim_mean_rho(t, 6, 3, 0.648)
        lo, hi = np.quantile(m, [0.025, 0.975])
        print("      true rho_s=%.1f -> observed mean rho 95%% range "
              "[%+.3f, %+.3f] (mean %+.3f)" % (t, lo, hi, m.mean()))
    print("    observed on p1_off: -0.2571 -> compatible with every true rho in")
    print("    the rows whose range covers it.")


if __name__ == "__main__":
    main()
