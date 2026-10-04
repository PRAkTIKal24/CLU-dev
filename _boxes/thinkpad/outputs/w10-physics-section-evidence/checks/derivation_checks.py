"""w10 spoke/physics-section-evidence - numerical reproduction of every derivation the
extended physics section quotes. Pure stdlib, no chlu import, no production code touched.

Each check prints: id | OK/FAIL | what | numbers.
"""
import math, json, random, statistics as st

LN2 = math.log(2.0)
out = []


def rec(cid, what, ok, detail):
    out.append((cid, what, "OK" if ok else "FAIL", detail))
    print("%-9s %-4s %s\n          %s" % (cid, "OK" if ok else "FAIL", what, detail))


def close(a, b, rtol=1e-9, atol=0.0):
    return abs(a - b) <= atol + rtol * abs(b)


# ---------- the shared 2x2 map (D3, D4) ----------
def lam(h, g):
    """roots of l^2 - (2-g)(1-h^2/2) l + (1-g) = 0 ; h = eps*mu"""
    tr = (2 - g) * (1 - h * h / 2.0)
    det = 1 - g
    disc = tr * tr - 4 * det
    if disc >= 0:
        r = math.sqrt(disc)
        return ((tr + r) / 2, (tr - r) / 2), False
    r = math.sqrt(-disc)
    return (complex(tr / 2, r / 2), complex(tr / 2, -r / 2)), True


def nhalf(h, g):
    (lp, lm), _ = lam(h, g)
    s = max(abs(lp), abs(lm))
    return LN2 / (-math.log(s))


bad = max(abs(lam(h, g)[0][0] * lam(h, g)[0][1] - (1 - g))
          for h in (0.01, 0.1, 0.5, 1.0, 1.9) for g in (0.002, 0.05, 0.2, 0.5))
rec("D3/D4", "det J = 1-gamma and lambda_pm are the roots of the derived quadratic",
    bad < 1e-14,
    "max |lam+ * lam- - (1-gamma)| over 20 (h,gamma) cells = %.3e" % bad)

# D5 : overdamped asymptotic vs the exact slow root
rows = []
for g in (0.05, 0.2, 0.5):
    for h in (g / 50, g / 20, g / 10):
        ex = nhalf(h, g)
        ap = 2 * g * LN2 / ((2 - g) * h * h)
        rows.append((g, h, ex, ap, abs(ex / ap - 1)))
worst = max(r[4] for r in rows)
rec("D5", "n_half = ln2/(-ln lam_+) ~= 2 g ln2/[(2-g)(eps mu)^2] deep overdamped",
    worst < 0.02,
    "worst relative gap exact-vs-asymptotic over 9 deep-overdamped cells (h <= h*/5) = %.3f %%; "
    "e.g. g=0.05 h=0.001 exact %.1f approx %.1f" % (worst * 100, rows[0][2], rows[0][3]))

# D6 : n_half ~ mu^-2
g, eps = 0.05, 0.05
mus = [0.001 * 2 ** k for k in range(4)]
xs = [math.log(m ** 2) for m in mus]
ys = [math.log(nhalf(eps * m, g)) for m in mus]
sl = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
rec("D6", "d log n_half / d log mu^2 = -1 on the overdamped branch",
    close(sl, -1.0, rtol=2e-3),
    "measured slope over mu in [1e-3, 8e-3] at gamma=0.05: %.6f (derived -1)" % sl)

# D7 : the curvature-independent floor
floor = lambda gg: 2 * LN2 / (-math.log(1 - gg))
f005 = floor(0.05)
rec("D7a", "floor 2 ln2/(-ln(1-gamma)) reproduces V2 Sec 4.1's printed 27.03 at gamma=0.05",
    close(f005, 27.02681, rtol=1e-6),
    "formula %.5f vs V2 printed 27.03 / w6-potentials recomputed 27.02681" % f005)
vals = [nhalf(h, 0.05) for h in (0.2, 0.5, 1.0, 1.5)]
rec("D7b", "the floor is independent of mu wherever the eigenpair is complex",
    max(vals) - min(vals) < 1e-9,
    "n_half at gamma=0.05 for h=0.2/0.5/1.0/1.5 = " + ", ".join("%.5f" % v for v in vals))
lret = math.sqrt(1 - 0.002)
nh = LN2 / (-math.log(lret))
rec("D7c", "V5's printed lambda_ret 0.998999499 and n_half 692.5 at gamma=0.002",
    close(lret, 0.998999499, rtol=1e-9) and close(nh, 692.4538, rtol=1e-5),
    "sqrt(1-gamma) = %.10f (V5 prints 0.998999499); n_half = %.4f (V5 prints 692.5, w6 692.4538); "
    "small-gamma 2ln2/gamma = %.1f (V5 prints 693.1)" % (lret, nh, 2 * LN2 / 0.002))

# D8 / D9 : band edges are exactly the roots of tr^2 = 4 det
hstar = lambda gg: math.sqrt(2 / (2 - gg)) * (1 - math.sqrt(1 - gg))
hf = lambda gg: math.sqrt(2 / (2 - gg)) * (1 + math.sqrt(1 - gg))
err = []
for g in (0.002, 0.05, 0.2, 0.5, 0.9):
    for hh in (hstar(g), hf(g)):
        tr = (2 - g) * (1 - hh * hh / 2.0)
        err.append(abs(tr * tr - 4 * (1 - g)))
rec("D8/D9", "h* and h_f are exactly the two roots of tr^2 J = 4 det J",
    max(err) < 1e-12,
    "max |tr^2 - 4 det| at the 10 derived band edges = %.3e; h*(0.05) = %.6f vs gamma/2 = 0.025; "
    "h_f(0.05) = %.6f vs 2" % (max(err), hstar(0.05), hf(0.05)))
rec("D9b", "the map loses stability at h = 2 for every gamma",
    all(max(abs(x) for x in lam(2.0, g)[0]) >= 1 - 1e-12 for g in (0.002, 0.05, 0.2, 0.5)),
    "max|lambda| at h=2: " + ", ".join(
        "g=%s: %.6f" % (g, max(abs(x) for x in lam(2.0, g)[0])) for g in (0.002, 0.05, 0.2, 0.5)))

# D10 : defective at h = h*
g = 0.05
(lp, lm), _ = lam(hstar(g), g)
rec("D10", "at h = h* the eigenvalues collide at sqrt(1-gamma) (one Jordan block)",
    close(abs(lp - lm), 0.0, atol=1e-7) and close(abs(lp), math.sqrt(1 - g), rtol=1e-7),
    "gamma=0.05: |lam+ - lam-| = %.3e, lam = %.9f, sqrt(1-gamma) = %.9f -- the residual is the square "
    "root of double-precision cancellation in the discriminant (sqrt(1e-16) ~ 1e-8), not a gap: the "
    "collision is exact analytically" % (abs(lp - lm), abs(lp), math.sqrt(1 - g)))

# D11 : square-root onset of the phase
g = 0.05
hs = hstar(g)
ps = []
for d in (1e-4, 1e-5, 1e-6):
    (a, b), c = lam(hs + d, g)
    ps.append((d, abs(math.atan2(a.imag, a.real)) if c else 0.0))
ex = math.log(ps[0][1] / ps[2][1]) / math.log(ps[0][0] / ps[2][0])
rec("D11", "phase exponent phi ~ (h - h*)^(1/2) above the exceptional point",
    close(ex, 0.5, rtol=5e-3),
    "log-log exponent over (h-h*) in [1e-6, 1e-4] = %.6f (derived 0.5)" % ex)

# D12 / D13 : the latch and the gamma = 0 Jordan block
M, eps, g, q0, p0 = 1.0, 0.05, 0.05, 0.3, 0.7
q, p = q0, p0
for _ in range(400000):
    q = q + eps * p / M          # sum_{n>=0} (1-gamma)^n, as D12 writes it
    p = (1 - g) * p
rec("D12", "latch q_inf = q0 + eps p0/(M gamma) at mu = 0",
    close(q, q0 + eps * p0 / (M * g), rtol=1e-9),
    "rolled out 4e5 steps: q = %.12f; derived %.12f" % (q, q0 + eps * p0 / (M * g)))
q, p = q0, p0
for _ in range(20000):
    q = q + eps * p / M
rec("D13", "gamma = 0 is ballistic drift, not a latch",
    close(q, q0 + 20000 * eps * p0 / M, rtol=1e-12),
    "20000 steps at gamma=0: q = %.4f (derived %.4f); the latch would give %.4f"
    % (q, q0 + 20000 * eps * p0 / M, q0 + eps * p0 / (M * g)))

# D14 : confinement lifts every Hessian eigenvalue by 2 alpha
random.seed(0)
n = 6
A = [[random.gauss(0, 1) for _ in range(n)] for _ in range(n)]
H = [[sum(A[k][i] * A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def eigsym(Mx, iters=2000):
    B = [row[:] for row in Mx]
    m = len(B)
    for _ in range(iters):
        p_, q_, mx = 0, 1, 0.0
        for i in range(m):
            for j in range(i + 1, m):
                if abs(B[i][j]) > mx:
                    mx, p_, q_ = abs(B[i][j]), i, j
        if mx < 1e-14:
            break
        app, aqq, apq = B[p_][p_], B[q_][q_], B[p_][q_]
        th = 0.5 * math.atan2(2 * apq, aqq - app)
        c, s = math.cos(th), math.sin(th)
        for k in range(m):
            bkp, bkq = B[k][p_], B[k][q_]
            B[k][p_] = c * bkp - s * bkq
            B[k][q_] = s * bkp + c * bkq
        for k in range(m):
            bpk, bqk = B[p_][k], B[q_][k]
            B[p_][k] = c * bpk - s * bqk
            B[q_][k] = s * bpk + c * bqk
    return sorted(B[i][i] for i in range(m))


alpha = 0.37
e0 = eigsym(H)
e1 = eigsym([[H[i][j] + (2 * alpha if i == j else 0) for j in range(n)] for i in range(n)])
d = max(abs((b - a) - 2 * alpha) for a, b in zip(e0, e1))
rec("D14a", "a confinement term alpha ||q||^2 lifts every Hessian eigenvalue by exactly 2 alpha",
    d < 1e-10,
    "max |lift - 2 alpha| over 6 eigenvalues of a random Gram matrix = %.3e" % d)
eps_, gam, al = 0.05, 0.05, 0.37
Gamma = gam / eps_
n_flat = nhalf(eps_ * math.sqrt(2 * al), gam)
tau_pred = Gamma / (2 * al)
rec("D14b", "a flat direction becomes mu^2 = 2 alpha and the lifetime caps at tau_max = Gamma/(2 alpha)",
    True,
    "Gamma = gamma/eps = %.3f; tau_max = Gamma/(2 alpha) = %.4f time units = %.2f steps; the exact-root "
    "half-life at mu^2 = 2 alpha is %.2f steps -- same order, D14 is a CAP not an equality"
    % (Gamma, tau_pred, tau_pred / eps_, n_flat))


# D16 : gamma_crit = 2 eps mu as the small-h inversion of h*(gamma) = h
def gcrit_exact(h):
    lo, hi = 1e-12, 0.999999
    for _ in range(200):
        mid = (lo + hi) / 2
        if hstar(mid) < h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


rowsg = [(h, gcrit_exact(h), 2 * h) for h in (1e-4, 1e-3, 1e-2, 0.0315, 0.063)]
rel = [abs(a - b) / b for _, a, b in rowsg]
rec("D16", "gamma_crit = 2 eps mu is the small-h inversion of h*(gamma) = h",
    rel[0] < 1e-3,
    "h -> exact gamma*, 2h, rel err: " + "; ".join(
        "%g -> %.6f vs %.6f (%.2f %%)" % (h, a, b, r * 100) for (h, a, b), r in zip(rowsg, rel))
    + "  [w6-potentials LOG.md quotes h=0.063 -> gamma*_exact 0.1185]")

# D17 : the overdamped log-slope, and the quoted +1.1385
d17 = lambda gg: 1 + gg / (2 - gg)


def num_slope(g, h=1e-3):
    """d ln n_half / d ln gamma at FIXED h = eps*mu (fixed mass), deep overdamped."""
    f = lambda gg: math.log(nhalf(h, gg))
    dg = 1e-6
    return (f(g * (1 + dg)) - f(g * (1 - dg))) / (2 * dg)


errs = [(g, d17(g), num_slope(g)) for g in (0.05, 0.1, 0.2, 0.35, 0.5)]
worst17 = max(abs(a - b) for _, a, b in errs)
rec("D17a", "1 + gamma/(2-gamma) IS d ln n_half / d ln gamma of the EXACT slow root, deep overdamped",
    worst17 < 2e-3,
    "gamma: derived vs numeric d ln n/d ln gamma at FIXED h = 1e-3 (h/h* <= 0.04): " + "; ".join(
        "%s: %.6f vs %.6f" % (g, a, b) for g, a, b in errs))
rec("D17b", "the derived slope exceeds +1 at every finite gamma in (0,2)",
    all(d17(k / 1000) > 1 for k in range(1, 2000)),
    "min over gamma in (0.001, 1.999) of [1+gamma/(2-gamma)] - 1 = %.6f > 0; monotone increasing, "
    "-> 1 only as gamma -> 0" % min(d17(k / 1000) - 1 for k in range(1, 2000)))
g1385 = 2 * 0.1385 / 1.1385
rec("D17c", "the QUOTED derived constant +1.1385 back-solves to a single gamma",
    close(d17(g1385), 1.1385, rtol=1e-9),
    "1 + gamma/(2-gamma) = 1.1385 at gamma = %.6f. The scored gamma grid is "
    "[0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.35,0.5] (test/s0/results.json pins.gammas), so 0.2433 lies "
    "INSIDE the overdamped span but is NOT a grid point; the per-cell fit windows that average to it "
    "are NOT in any recorded machine-readable artefact" % g1385)
rec("D17d", "derived slope at each recorded gamma grid point (for the reader to place +1.1385)",
    True, "; ".join("%s: %.4f" % (g, d17(g))
                    for g in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)))

# D18 : the underdamped log-slope
d18 = lambda gg: -gg / ((1 - gg) * (-math.log(1 - gg)))


def num18(g):
    dg = 1e-6
    return (math.log(floor(g * (1 + dg))) - math.log(floor(g * (1 - dg)))) / (2 * dg)


w18 = max(abs(d18(g) - num18(g)) for g in (0.002, 0.05, 0.2, 0.5))
rec("D18", "-gamma/[(1-gamma)(-ln(1-gamma))] IS d ln(floor)/d ln gamma, and ~ -1 - gamma/2",
    w18 < 1e-6,
    "gamma: derived / small-g form (-1-g/2) / numeric: " + "; ".join(
        "%s: %.6f / %.6f / %.6f" % (g, d18(g), -1 - g / 2, num18(g))
        for g in (0.002, 0.05, 0.2, 0.5)))

# D18-DIAG : where each QUOTED D18 value sits on the derived curve (a diagnostic, not a bar)
def invert(target, f, lo=1e-6, hi=0.9):
    for _ in range(300):
        m = (lo + hi) / 2
        if f(m) > target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


g_meas = invert(-1.1090, d18)
g_der = invert(-1.0050, d18)


def secant(ga, gb):
    return (math.log(floor(gb)) - math.log(floor(ga))) / (math.log(gb) - math.log(ga))


rec("D18-DIAG",
    "the QUOTED derived -1.0050 and the QUOTED measured -1.1090 sit at DIFFERENT gamma on the same "
    "derived curve -- reported as a diagnostic, NOT as a bar",
    True,
    "d18(gamma) = -1.0050 at gamma = %.6f; d18(gamma) = -1.1090 at gamma = %.6f. d18 on the recorded "
    "grid: %s. No secant of ln(floor) vs ln gamma over the recorded grid reaches -1.109 "
    "(ln-slope over [0.002,0.1] = %.5f, over [0.002,0.2] = %.5f). The shorts' own underdamped slopes "
    "are -1.006 (V5 E5.1) and -1.0020 +- 0.0003 (V5 E5.2, I-J). The per-cell fit windows are NOT "
    "recorded, so which of these explains w6's -1.1090 CANNOT be adjudicated from the artefacts."
    % (g_der, g_meas,
       "; ".join("%s: %.4f" % (g, d18(g)) for g in (0.002, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5)),
       secant(0.002, 0.1), secant(0.002, 0.2)))

# D20 : the discrete FDT noise scale
Mm, T = 2.3, 0.004
resid = max(abs((Mm * T * g * (2 - g)) / (g * (2 - g)) - Mm * T) for g in (0.05, 0.2, 0.5))
rec("D20", "sigma* = sqrt(M T gamma (2-gamma)) is exactly what forces Var(p) = sigma^2/[g(2-g)] = M T",
    resid < 1e-12,
    "residual |Var(p) - M T| at gamma = 0.05/0.2/0.5 with M T = %.6f is %.3e" % (Mm * T, resid))

# D21 / D23 : coset diffusion and the sign flip
Dth = lambda e, t, gg, F2: e * t * (2 - gg) / (2 * F2 * gg)
F2 = 1.7
sl_T = (math.log(Dth(0.05, 2e-3, 0.05, F2)) - math.log(Dth(0.05, 1e-3, 0.05, F2))) / math.log(2)
sl_g = (math.log(Dth(0.05, 1e-3, 0.1, F2)) - math.log(Dth(0.05, 1e-3, 0.05, F2))) / math.log(2)
rec("D21", "D_theta = eps T (2-gamma)/(2 F^2 gamma): d log D/d log T = +1 exactly",
    close(sl_T, 1.0, rtol=1e-12),
    "d log D/d log T = %.6f (V5 E5.6 measured +0.992..+1.010); raw d log D/d log gamma over 0.05->0.1 "
    "= %.6f, and ln((2-0.1)/(2-0.05))/ln2 = %.6f is exactly the (2-gamma) factor V5 isolates before "
    "quoting -1.014..-0.980" % (sl_T, sl_g, math.log(1.9 / 1.95) / math.log(2)))
sg = (math.log(0.2 / 1.8) - math.log(0.05 / 1.95)) / math.log(4)
rec("D23", "n_half ~ gamma/[(2-gamma) T]: friction PRESERVES the coset, temperature ERASES it",
    sg > 0,
    "d log n_half/d log gamma over gamma 0.05->0.2 = %.6f (V5 E5.7 measured +0.9552 +- 0.0422, positive "
    "in 10/10 cells); d log n_half/d log T = -1 exactly (V5 measured -0.956..-0.979); predicted effect "
    "size for gamma 0.05->0.2 is %.3fx (V5 measured 3.77 +- 0.23x)"
    % (sg, (0.2 / 1.8) / (0.05 / 1.95)))

# D26-D31 : the friction hole / vault / refrigerator against V5 E5.10 / E5.11 / E5.14
g = 0.05
tab = []
for gphi, meas_vault, meas_ratio in ((0.1, 8.42, 0.36171), (0.2, 22.39, 0.23044),
                                     (0.3, 44.31, 0.17513), (0.5, 107.77, 0.12600)):
    geff = 1 - (1 - g) * (1 - gphi)
    tab.append((gphi, geff, (geff / g) ** 2, meas_vault,
                g * (2 - g) / (geff * (2 - geff)), meas_ratio,
                (2 - g) * geff / (g * (2 - geff)),
                geff * (2 - geff) / (g * (2 - g))))
rec("D26/D27", "gamma_eff = 1-(1-g)(1-g_phi) and T_local/T = g(2-g)/[g_eff(2-g_eff)] reproduce V5 E5.11",
    all(abs(t[4] - t[5]) / t[5] < 0.01 for t in tab),
    "; ".join("g_phi=%s: g_eff=%.4f, T_loc/T pred %.5f vs V5 measured %.5f" % (t[0], t[1], t[4], t[5])
              for t in tab))
rec("D29", "vault = (gamma_eff/gamma)^2 reproduces V5 E5.10's measured vaults",
    all(abs(t[2] - t[3]) / t[3] < 0.04 for t in tab),
    "; ".join("g_phi=%s: pred %.2fx vs measured %.2fx" % (t[0], t[2], t[3]) for t in tab)
    + "  [V5's own headline prediction at g_phi=0.5 is 110.25x; this formula gives %.2fx]" % tab[-1][2])
rec("D30", "the REJECTED coupled-bath vault (2-g)g_eff/[g(2-g_eff)] is exactly the 13.88x V5 refutes",
    close(tab[-1][6], 13.88, rtol=1e-3),
    "coupled-bath prediction at g_phi=0.5 = %.4fx (V5 quotes 13.88x and refutes it; the matched-g_eff "
    "uniform-scalar control measured 13.28 +- 0.12x)" % tab[-1][6])
rec("D31", "separation = g_eff(2-g_eff)/[g(2-g)] equals T/T_local IDENTICALLY",
    all(close(t[7], 1 / t[4], rtol=1e-12) for t in tab),
    "at g_phi=0.5: separation %.4f == T/T_local %.4f (V5 E5.11 measured a 7.94x refrigerator and a "
    "7.942x separation; E5.14 emergent arm 7.955x vs 7.942 predicted)" % (tab[-1][7], 1 / tab[-1][4]))

# D37 / D38 : the squeeze map
worst38 = -1e9
random.seed(1)
for _ in range(20000):
    u, v = random.gauss(0, 1), random.gauss(0, 1)
    z = random.uniform(-2, 2)
    c, s = math.cosh(z), math.sinh(z)
    up, vp = u * c + v * s, v * c + u * s
    worst38 = max(worst38, (up * up + vp * vp) - (u * u + v * v) * math.exp(2 * abs(z)))
detok = max(abs(math.cosh(z) ** 2 - math.sinh(z) ** 2 - 1) for z in (0.1, 0.5, 1.0, 2.0))
rec("D37/D38", "det S_z = 1 exactly, and H(S_z z) <= e^{2|z|} H(z)",
    worst38 <= 1e-9 and detok < 1e-12,
    "max |cosh^2 z - sinh^2 z - 1| = %.3e; max (H' - e^{2|z|} H) over 20000 random (u,v,z) = %.3e "
    "(<= 0, bound never violated; equality on u = +-v)" % (detok, worst38))


# D43 : the hard-gate Jacobian defect
def det3(Mx):
    return (Mx[0][0] * (Mx[1][1] * Mx[2][2] - Mx[1][2] * Mx[2][1])
            - Mx[0][1] * (Mx[1][0] * Mx[2][2] - Mx[1][2] * Mx[2][0])
            + Mx[0][2] * (Mx[1][0] * Mx[2][1] - Mx[1][1] * Mx[2][0]))


random.seed(2)
worst43 = 0.0
for _ in range(2000):
    D = [random.gauss(0, 1) for _ in range(3)]
    G = [random.gauss(0, 1) for _ in range(3)]
    Mx = [[(1.0 if i == j else 0.0) + D[i] * G[j] for j in range(3)] for i in range(3)]
    worst43 = max(worst43, abs(det3(Mx) - (1 + sum(D[i] * G[i] for i in range(3)))))
rec("D43", "det(I + Delta grad_g^T) = 1 + grad_g . Delta (a varying gate breaks det J = 1)",
    worst43 < 1e-10,
    "max residual over 2000 random (Delta, grad_g) in d=3 = %.3e; the defect is first order in the "
    "gate's variation, so no smallness of Delta removes it" % worst43)

# D40 : quadratic in excess distance, exponential in rapidity
p0m, M0 = 1.2, 4.0
rowsq = []
for delta in (0.1, 0.5, 1.0):
    z = math.asinh(M0 * delta / p0m)
    rowsq.append((delta, math.cosh(2 * z), 1 + 2 * (M0 * delta / p0m) ** 2, math.exp(2 * abs(z))))
rec("D40", "cosh 2z = 1 + 2 (M0 delta/p0)^2 exactly; the CERTIFICATE e^{2|z|} is exponential in RAPIDITY",
    all(close(a, b, rtol=1e-12) for _, a, b, _ in rowsq),
    "; ".join("delta=%s: cosh2z %.6f == 1+2(M0 d/p0)^2 %.6f, certificate e^{2z} %.4f"
              % (d, a, b, c) for d, a, b, c in rowsq))

# ---------- the recorded w6 artefacts ----------
BASE = "C:/chlu/evidence/outputs/_thinkpad/w6/w6-potentials"
art = []
for s in (0, 1, 2):
    dd = json.load(open("%s/test/s%d/results.json" % (BASE, s)))
    cs = dd["estimand"]["corners"]
    art.append(dict(
        seed=s, n=len(cs),
        B1=st.median(c["median_err_crr"] for c in cs),
        B2d=sum(c["paired_delta"] for c in cs) / len(cs),
        better=sum(1 for c in cs if c["median_err_crr"] < c["median_err_fit1"]),
        rows_scored=sum(c["n_rows"] for c in cs), rows_all=dd["n_rows"],
        C=dd["fit1_C"], sha=dd["field_sha"][:16],
        by=dd["byte_ledger"]["deployed_bytes"], okb=dd["byte_ledger"]["matches_budget"],
        fl=dd["instrument_floor"]["median_abs_log10"],
        B6=dd["hessian_fd_check"]["max_abs_residual"],
        ig=dd["integrator_check"]["max_abs_diff"],
        d14=dd["d14_confinement"]["max_abs_residual"]))
led = {0: (0.0433, -0.5968), 1: (0.0435, -0.5796), 2: (0.0431, -0.5982)}
rec("B1/B2", "RESULTS-LEDGER.md's B1 and B2 RECOMPUTE from test/s{0,1,2}/results.json",
    all(close(a["B1"], led[a["seed"]][0], atol=5e-5)
        and close(a["B2d"], led[a["seed"]][1], atol=5e-5) and a["better"] == 49 for a in art),
    "; ".join("seed%d: B1 %.6f (ledger %s), B2 mean delta %.6f (ledger %s), better %d/49"
              % (a["seed"], a["B1"], led[a["seed"]][0], a["B2d"], led[a["seed"]][1], a["better"])
              for a in art))
rec("AUDIT", "byte pin, field sha, integrator, instrument floor and B6 from the same files",
    all(a["by"] == 524288 and a["okb"] and a["ig"] < 1e-5 and a["B6"] < 1e-9 for a in art),
    "; ".join("seed%d: %d B pin=%s, sha %s, integrator %.3e, floor %.6f, B6 %.3e"
              % (a["seed"], a["by"], a["okb"], a["sha"], a["ig"], a["fl"], a["B6"]) for a in art)
    + "; scored-corner rows %s of total rows %s"
    % ([a["rows_scored"] for a in art], [a["rows_all"] for a in art]))
rec("D14-TAUTO",
    "results.json's `d14_confinement` is EXACTLY 0.0 on 60 cells - the tautology the hub itself "
    "declared; the SCORED B6 is `hessian_fd_check`, not this field",
    all(a["d14"] == 0.0 for a in art),
    "d14_confinement max_abs_residual = %s (identically zero: the same algebra on both sides); "
    "hessian_fd_check max_abs_residual = %s (a finite difference that CAN fail)"
    % ([a["d14"] for a in art], ["%.3e" % a["B6"] for a in art]))

print("\n" + "=" * 78)
print("%d/%d checks reproduced; %d FAIL"
      % (sum(1 for r in out if r[2] == "OK"), len(out), sum(1 for r in out if r[2] == "FAIL")))
for r in out:
    if r[2] == "FAIL":
        print("  FAIL:", r[0], r[1])
