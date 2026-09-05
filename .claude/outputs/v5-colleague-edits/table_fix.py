# -*- coding: utf-8 -*-
import io, re
DST = "/tmp/v5edit/pj_sub.tex"
s = io.open(DST, encoding="utf-8").read()
start = s.index("\\begin{table}[!htb]\\centering\\small\n\\caption{The measured values behind")
end = s.index("\\end{table}\n", start) + len("\\end{table}\n")
NEW = r"""\begin{table}[!htb]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{The measured values behind Sec.~\ref{sec:results}, with the arm each was measured on and the rider each must travel with. \emph{Arm}: \emph{verification} = designed testbed; \emph{evidence} = learned system. All rows come from the checkpoints and grids of Sec.~\ref{sec:vcurve} and Apps.~\ref{app:budget}--\ref{app:vault} (dim 4, hidden 64, $\varepsilon=0.05$, single-CPU); every $T>0$ cell uses \texttt{langevin\_noise="fdt"} with a Newtonian kinetic mode.}
\label{tab:numbers}
\begin{tabular}{p{0.145\linewidth}p{0.155\linewidth}p{0.155\linewidth}p{0.075\linewidth}p{0.30\linewidth}}
\toprule
Result & Quantity & Value & Arm & Scope / rider\\
\midrule
V-curve, emergent arm & argmin, one-step Jacobian & $0.902\pm0.003$ $\times\gamma_{\rm crit}$ & evidence & 3 emergent MLP seeds, $T=0$; a learned potential, not a designed one\\
(Sec.~\ref{sec:vcurve}, & log-slope, over- & $-1.0020$ & evidence & same 3 emergent seeds\\
App.~\ref{app:emergent}) & damped branch & $\pm0.0003$ & & \\
 & log-slope, under- & $+1.116$ & evidence & same 3 emergent seeds\\
 & damped branch & $\pm0.011$ & & \\
\addlinespace
Collapse span (Sec.~\ref{sec:vcurve}) & $\mu^2$ range spanned by the one curve & $[1.7\times10^{-12},$ $7\times10^{-2}]$ & both arms & \textbf{Probe resolution:} the low endpoint is the ring-profile probe's resolution on a checkpoint whose Hessian $\mu^2$ is machine zero --- \emph{not} a measured spectral mass\\
\addlinespace
Cross-instrument check & argmin, rollout envelope rate & $0.9001\pm0.0052$ $\times\gamma_{\rm crit}$ & evidence & 3 emergent seeds, $\delta=0.05$ rad\\
(Sec.~\ref{sec:vcurve}, App.~\ref{app:emergent}) & argmin, one-step Jacobian & $0.9032\pm0.0027$ $\times\gamma_{\rm crit}$ & evidence & same cells; the instrument gap is a level, not a rate\\
\addlinespace
$T=0$ latch (Sec.~\ref{sec:vault}, App.~\ref{app:budget}) & coset drift & $\le4.9\times10^{-12}$ rad / 200k steps & verifi\-cation & designed $SO(2)$ coset, every $\gamma\in[0.002,0.5]$; $T=0$ only\\
\addlinespace
Coset diffusion law (Sec.~\ref{sec:vault}, App.~\ref{app:budget}) & $\hat D_\theta/D_\theta^{\rm pred}$ & $1.0068\pm0.0219$ & verifi\-cation & 25 $(\gamma,T)$ cells on one trained designed checkpoint (seed 44), $\Delta=0.5$ rad\\
\addlinespace
Friction-hole vault & $T_{\rm local}$ inside vs.\ outside & $1.26\times10^{-4}$ vs.\ $10^{-3}$ & verifi\-cation & designed arm, $\gamma=0.05$, $\gamma_\phi=0.5$; absorb-only field\\
(Sec.~\ref{sec:vault}, & vault factor, $\hat D_\theta$ estimator & $107.77\pm4.78\times$ & verifi\-cation & \textbf{the quoted vault; it travels with this estimator's name.} Designed $SO(2)$, 3 seeds, $T=10^{-3}$, $\Delta=0.5$ rad\\
App.~\ref{app:vault}) & scalar-friction control & $13.28\pm0.12\times$ & verifi\-cation & uniform scalar friction at matched $\gamma_{\rm eff}$, same cells\\
 & raw first-passage ratio & $86.97\pm2.94\times$ & verifi\-cation & boundary-layer biased on the outside arm ($\ell_\theta/\Delta=0.079$); \emph{not} the quoted vault\\
\bottomrule
\end{tabular}
\end{table}
"""
s = s[:start] + NEW + s[end:]
io.open(DST,"w",encoding="utf-8").write(s)
print("table replaced")
