# -*- coding: utf-8 -*-
import io, sys, hashlib

SRC = "/tmp/v5edit/pj_sub_BEFORE.tex"
DST = "/tmp/v5edit/pj_sub.tex"

s = io.open(SRC, encoding="utf-8").read()
assert hashlib.md5(s.encode("utf-8")).hexdigest() == "c63a57fc910663dfa1e644b9b349ce6f", "boot md5 mismatch"

EDITS = []  # (item_id, old, new)

def E(item, old, new):
    EDITS.append((item, old, new))

# ---------------- Group X: l.35 ----------------
E("X l.35 explicitly",
  "governed by explicitly defined operational conditions,",
  "governed by defined operational conditions,")

# ---------------- Group X: l.40 ----------------
E("X l.40 distinct",
  "currently operate under distinct retention/deletion mechanisms:",
  "currently operate under retention/deletion mechanisms:")
E("X l.40 intrinsically",
  "one cannot intrinsically read the half-life of a stored value",
  "one cannot read the half-life of a stored value")

# ---------------- T.1: l.42 ----------------
E("T.1 l.42",
  "within the network architecture after the fact \\citep{chakraborttii_ghost_2026,wang_memleak_2026}",
  "within the network architecture even after the entry is nominally deleted \\citep{chakraborttii_ghost_2026,wang_memleak_2026}")

# ---------------- T.2: l.44 ----------------
E("T.2 l.44",
  "In this work, we propose and analyze a memory framework where forgetting is an intrinsic dynamical property of the store itself, rather than an external bookkeeping rule.",
  "In this work, we propose and analyze a memory framework where forgetting is a prescribed dynamical property of the store, governed by parameters that can be set, targeted, and analyzed in closed form, rather than an emergent side effect or an external bookkeeping rule.")

# ---------------- Group X: l.57 ----------------
E("X l.57 precisely",
  "rely on a novel, precisely designed store.",
  "rely on a novel, designed store.")

# ---------------- T.4: l.59 ----------------
E("T.4 l.59",
  "    \\item \\textbf{Nomenclature:}",
  "    \\item \\textbf{Nomenclature borrowing from and building on CHLU~\\cite{jawahar_chlu_2026}:}")

# ---------------- T.5 (+ X l.74 fundamentally): l.74 ----------------
E("T.5+X l.74",
  "To understand the macro-dynamics of memory decay, we first analyze damping as a retention dial. We demonstrate that retention half-life is non-monotone in relation to friction. The turning point is fundamentally fixed by the stored direction's spectral mass, meaning the optimal setting can be analytically predicted rather than empirically tuned.",
  "Friction is not a monotone retention knob: retention half-life is non-monotone in relation to friction, and there is an optimal friction. The turning point is fixed by the stored direction's spectral mass, meaning the optimal setting can be analytically predicted rather than empirically tuned. To understand the macro-dynamics of memory decay, we analyze damping as that retention dial.")

# ---------------- Group X: l.76 ----------------
E("X l.76 distinct",
  "$V_\\theta$, across 5 distinct seeds)",
  "$V_\\theta$, across 5 seeds)")

# ---------------- R.1 + R.2 + X: l.78 ----------------
E("R.1+R.2+X l.78",
  "On a fully trained, designed $SO(2)$ vacuum, the massive radial mode's half-life $n_{1/2}(\\gamma)$ forms a distinct V-curve, minimized at $\\gamma_{\\rm crit}=2\\varepsilon\\mu_{\\rm rad}$ (Fig.~\\ref{fig:massiveflat}, App.~\\ref{app:budget}). The overdamped branch tracks the $\\mu^{-2}$ law precisely until $\\varepsilon\\mu\\approx\\gamma/2$, after which it saturates at a mass-independent floor. This yields log-slopes of $-1.006$ and between $+1.23$ and $+1.27$ across the 5 validation seeds (See App.~\\ref{app:budget} for results). The flat coset (characterized by a Hessian $\\mu^2\\approx10^{-15}$) maps to the exact same curve at the limit $\\mu\\to0$, where $\\gamma_{\\rm crit}\\to0$. Therefore, the latch, the overdamped register, and the underdamped working memory represent three specific operational regimes of a singular unified curve evaluated at two disparate values of $\\mu$.",
  "The latch, the overdamped register, and the underdamped working memory are three specific operational regimes of a unified curve evaluated at two disparate values of $\\mu$. On a fully trained, designed $SO(2)$ vacuum, the massive radial mode's retention half-life $n_{1/2}(\\gamma)$ forms a V-curve, minimized at $\\gamma_{\\rm crit}=2\\varepsilon\\mu_{\\rm rad}$ (Fig.~\\ref{fig:massiveflat}, App.~\\ref{app:budget}). The overdamped branch tracks the $\\mu^{-2}$ law until $\\varepsilon\\mu\\approx\\gamma/2$, after which it saturates at a mass-independent floor. This yields log-slopes of $-1.006$ and between $+1.23$ and $+1.27$ across the 5 validation seeds (See App.~\\ref{app:budget} for results). The flat coset (characterized by a Hessian $\\mu^2\\approx10^{-15}$) maps to the exact same curve at the limit $\\mu\\to0$, where $\\gamma_{\\rm crit}\\to0$.")

# ---------------- R.2 + R.3 + R.3b + X: l.80 ----------------
E("R.2+R.3+R.3b+X l.80",
  "Evaluating three distinct emergent MLP checkpoints, the coset-tracked $T=0$ half-life traces the identical V-curve framework: an argmin of $0.902\\pm0.003\\times\\gamma_{\\rm crit}$, and respective log-slopes of $-1.0020\\pm0.0003$ and $+1.116\\pm0.011$ across all 3 seeds (Figure~\\ref{fig:lambdacoset})). Spanning both architecture families, this curve holds consistently across $\\mu^2\\in[1.7\\times10^{-12},\\,7\\times10^{-2}]$ (Fig.~\\ref{fig:collapse}). That low endpoint is the ring-profile probe's resolution on a checkpoint whose Hessian $\\mu^2$ is machine zero rather than a spectral mass, so eleven orders is one curve on one instrument (probe-floor tick marked in Fig.~\\ref{fig:collapse-full}).",
  "Evaluating three emergent MLP checkpoints, the coset-tracked $T=0$ retention half-life --- read on the near-flat stored direction that holds a written value, not on the stiff radial mode of the previous paragraph, though one law governs both (Fig.~\\ref{fig:collapse}) --- traces the identical V-curve framework: an argmin in units of $\\gamma_{\\rm crit}$, and respective log-slopes on the two asymptotic branches, across all 3 seeds (Figure~\\ref{fig:lambdacoset}), with the measured values in Table~\\ref{tab:numbers}. Spanning both architecture families, this curve holds consistently across a $\\mu^2$ span of eleven orders (Table~\\ref{tab:numbers}, Fig.~\\ref{fig:collapse}). The low endpoint of that span is the ring-profile probe's resolution on a checkpoint whose Hessian $\\mu^2$ is machine zero rather than a spectral mass, so eleven orders is one curve on one instrument (probe-floor tick marked in Fig.~\\ref{fig:collapse-full}). Put plainly: the curve a designed unit follows is the curve a learned one follows, read at a different spectral mass.")

# ---------------- R.4: l.89 ----------------
E("R.4 l.89",
  "The underlying shape is reproduced well (see Fig.~\\ref{fig:twoinstruments}): the argmin measured at $0.9001\\pm0.0052$ compared to the Jacobian's $0.9032\\pm0.0027$ in units of $\\gamma_{\\rm crit}$ (a variance of $0.35\\%$).",
  "The underlying shape is reproduced well on this second instrument (see Fig.~\\ref{fig:twoinstruments}), the two argmins agreeing to $0.35\\%$ in units of $\\gamma_{\\rm crit}$ (both readings in Table~\\ref{tab:numbers}).")

# ---------------- R.5 + X: l.93 ----------------
E("R.5+X l.93",
  "a designed coset latches indefinitely across any friction threshold (drift $\\le4.9\\times10^{-12}$ rad over $200$k steps, spanning all $\\gamma\\in[0.002,0.5]$)(Fig~\\ref{fig:massiveflat}b). Conversely, at $T>0$, the coset degrades via diffusion, and introducing friction physically decelerates this decay.",
  "a designed coset latches indefinitely across every friction setting we evaluated (Fig~\\ref{fig:massiveflat}b; drift bound and $\\gamma$ range in Table~\\ref{tab:numbers}). Conversely, at $T>0$, the coset degrades via diffusion, and introducing friction decelerates this decay.")

# ---------------- R.5 + X: l.95 ----------------
E("R.5+X l.95",
  "(App.~\\ref{app:deriv:6}), which we successfully verified to a tolerance of $1.0068\\pm0.0219$ across 25 $(\\gamma,T)$ experimental cells.",
  "(App.~\\ref{app:deriv:6}), which we verified over a grid of $(\\gamma,T)$ cells on the designed testbed (Table~\\ref{tab:numbers}).")

# ---------------- P.1 + R.5 + R.3b + X: l.97 ----------------
E("P.1+R.5+R.3b+X l.97",
  "We propose the memory vault: a position-gated friction field that is explicitly absorb-only. A localized spatial hole within this field functions concurrently as a brake and a refrigerator ($T_{\\rm local}=1.26\\times10^{-4}$ versus $10^{-3}$ externally). This mechanism generates a remarkable retention vault factor of $(\\gamma_{\\rm eff}/\\gamma)^2$, yielding $107.77\\pm4.78\\times$ on designed architecture (Fig.~\\ref{fig:vault}), compared to a baseline of $13.28\\pm0.12\\times$ utilizing a uniform scalar friction. The direct first-passage vault reads $86.97\\pm2.94\\times$ and is boundary-layer biased on the outside arm ($\\ell_\\theta/\\Delta=0.079$), so $107.77\\times$ is the quoted number and travels with its estimator's name.",
  "We propose the memory vault: a position-gated friction field that is absorb-only. A localized spatial hole within this field acts simultaneously as a brake, increasing dissipation, and a refrigerator, reducing the local effective temperature. This mechanism generates a retention vault factor of $(\\gamma_{\\rm eff}/\\gamma)^2$ on designed architecture (Fig.~\\ref{fig:vault}), read from the $\\hat D_\\theta$ estimator and quoted against a uniform-scalar-friction control at matched $\\gamma_{\\rm eff}$; the direct first-passage reading is boundary-layer biased on the outside arm ($\\ell_\\theta/\\Delta=0.079$) and is therefore not the quoted vault. All four readings --- the $\\hat D_\\theta$ vault, the scalar control, the first-passage value, and the local temperature --- are in Table~\\ref{tab:numbers}.")

E("R.3b l.97 intuition",
  "drops definitively to $0.0000$ within the hole (see Fig.~\\ref{fig:vault-emergent} and App.~\\ref{app:vault}).",
  "drops definitively to $0.0000$ within the hole (see Fig.~\\ref{fig:vault-emergent} and App.~\\ref{app:vault}). Put plainly: the vault factor is how much longer a written value is retained inside the hole than outside it.")

# ---------------- P.2 + X: l.107 ----------------
E("P.2+X l.107",
  "We evaluate whether the store's intrinsic physical state can be explicitly reduced to a pure function of its live set alone.",
  "We evaluate whether the store's intrinsic physical state can be reduced to a pure function of its live set alone --- that is, whether the layout depends only on which items are currently stored, not on the order or history of the writes that stored them.")

# ---------------- Group X: l.119 / l.120 / l.123 ----------------
E("X l.119 explicitly",
  "coset geometries that must be explicitly designed into the architecture.",
  "coset geometries that must be designed into the architecture.")
E("X l.120 intrinsically",
  "cannot distinguish between designed and emergent units intrinsically.",
  "cannot distinguish between designed and emergent units.")
E("X l.123 explicitly",
  "and construct explicitly localized temperature fields.",
  "and construct localized temperature fields.")

# ---------------- R.0: the new first appendix table ----------------
TABLE = r"""\appendix

\section{Values Quoted in the Main Text}\label{app:numbers}

Sec.~\ref{sec:results} states its results in words; the measured values behind them are collected here, each beside the arm it was measured on and the rider it must be quoted with. \emph{Arm} names the class of the measurement: \textbf{verification} = a \emph{designed testbed} (an architecturally designed $SO(2)$ potential), \textbf{evidence} = a \emph{learned system} (an MLP $V_\theta$ trained on symmetric data). Every value is repeated from the section or appendix named beside it; none is new here.

\begin{table}[!htb]\centering\small
\caption{The measured values behind Sec.~\ref{sec:results}, with the arm each was measured on and the rider each must travel with. \emph{Arm}: \emph{verification} = designed testbed; \emph{evidence} = learned system. All rows are measured on the checkpoints and grids of Sec.~\ref{sec:vcurve} and Apps.~\ref{app:budget}--\ref{app:vault} (dim 4, hidden 64, $\varepsilon=0.05$, single-CPU); every $T>0$ cell uses \texttt{langevin\_noise="fdt"} with a Newtonian kinetic mode.}
\label{tab:numbers}
\begin{tabular}{p{0.15\linewidth}p{0.19\linewidth}p{0.15\linewidth}p{0.10\linewidth}p{0.27\linewidth}}
\toprule
Result & Quantity & Value & Arm & Scope / rider\\
\midrule
\multirow{3}{*}{\parbox{0.15\linewidth}{V-curve, emergent arm (Sec.~\ref{sec:vcurve}; App.~\ref{app:emergent})}}
 & argmin, one-step Jacobian & $0.902\pm0.003\times\gamma_{\rm crit}$ & evidence & 3 emergent MLP seeds, $T=0$; learned potential, not designed\\
 & log-slope, overdamped branch & $-1.0020\pm0.0003$ & evidence & same 3 emergent seeds\\
 & log-slope, underdamped branch & $+1.116\pm0.011$ & evidence & same 3 emergent seeds\\
\addlinespace
Collapse span (Sec.~\ref{sec:vcurve}) & $\mu^2$ range spanned by the one curve & $[1.7\times10^{-12},\,7\times10^{-2}]$ & both arms & \textbf{Probe resolution:} the low endpoint is the ring-profile probe's resolution on a checkpoint whose Hessian $\mu^2$ is machine zero --- \emph{not} a measured spectral mass. Eleven orders is one curve on one instrument\\
\addlinespace
\multirow{2}{*}{\parbox{0.15\linewidth}{Cross-instrument check (Sec.~\ref{sec:vcurve}; App.~\ref{app:emergent})}}
 & argmin, rollout envelope rate (I-R3) & $0.9001\pm0.0052\times\gamma_{\rm crit}$ & evidence & 3 emergent seeds, $\delta=0.05$ rad\\
 & argmin, one-step Jacobian (I-J) & $0.9032\pm0.0027\times\gamma_{\rm crit}$ & evidence & same cells; the instrument gap is a level, not a rate\\
\addlinespace
$T=0$ latch (Sec.~\ref{sec:vault}; App.~\ref{app:budget}) & coset drift & $\le4.9\times10^{-12}$ rad per 200k steps & verification & designed $SO(2)$ coset, all $\gamma\in[0.002,0.5]$; $T=0$ only\\
\addlinespace
Coset diffusion law (Sec.~\ref{sec:vault}; App.~\ref{app:budget}) & $\hat D_\theta/D_\theta^{\rm pred}$ & $1.0068\pm0.0219$ & verification & 25 $(\gamma,T)$ cells on one trained designed checkpoint (seed 44), $\Delta=0.5$ rad\\
\addlinespace
\multirow{4}{*}{\parbox{0.15\linewidth}{Friction-hole vault (Sec.~\ref{sec:vault}; App.~\ref{app:vault})}}
 & $T_{\rm local}$ inside vs.\ outside the hole & $1.26\times10^{-4}$ vs.\ $10^{-3}$ & verification & designed arm, $\gamma=0.05$, $\gamma_\phi=0.5$; absorb-only field\\
 & vault factor, $\hat D_\theta$ estimator & $107.77\pm4.78\times$ & verification & \textbf{the quoted vault, and it travels with this estimator's name}; designed $SO(2)$, 3 seeds, $T=10^{-3}$, $\Delta=0.5$ rad\\
 & scalar-friction control & $13.28\pm0.12\times$ & verification & uniform scalar friction at matched $\gamma_{\rm eff}$, same cells\\
 & raw first-passage ratio & $86.97\pm2.94\times$ & verification & boundary-layer biased on the outside arm ($\ell_\theta/\Delta=0.079$); \emph{not} the quoted vault\\
\bottomrule
\end{tabular}
\end{table}
"""
E("R.0 new appendix table", "\\appendix\n", TABLE)

for item, old, new in EDITS:
    n = s.count(old)
    if n != 1:
        print("FAIL: %s -> %d occurrences" % (item, n)); sys.exit(1)
    s = s.replace(old, new)
    print("ok: %s" % item)

io.open(DST, "w", encoding="utf-8").write(s)
print("written", DST)
