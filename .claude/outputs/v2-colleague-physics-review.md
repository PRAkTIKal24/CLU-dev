# v2-colleague-physics-review — physics-theorist report

**Task + acceptance criterion:** rigor-check the colleague's SO(2) SSB primer (`.claude/colleague/main.tex`), reconcile its conventions against the theory estate, produce the complete claims-collision table with mandatory riders, and map integration options — flag, never fix; zero edits outside this report.
**Status: done.** No file other than this report (and my scratch script) was touched. The colleague's file, the drafts, and the registries are untouched.

**DIAL DECLARATION (echoed).** The task file carries no DIAL DECLARATION block (Advisor-commissioned, pre-scoped as review). Echoing per protocol §7: **Dial: none — instrument/recon/theory** (a rigor/reconciliation review; no performance claim is made or measured here). Laundering control: n/a (no performance number). Falsifies: an equation in `main.tex` failing verification would be the headline (none did). Does-not-falsify: n/a.

**Headline:** **the primer's two equations and its central derivation are correct and verified numerically to machine precision** (`∇²V(q*)·Jq* = 0` to `1.3e-16` on a Mexican hat; zero-eigenvector alignment `1.000000000000`). The defects are: **one substantive overstatement** (the closing Wigner–Weyl claim, counterexampled below), **one unscoped claims-bearing sentence** (the "stored without being pulled back" sentence collides with N46/CM-16a designed-only scope and CM-16b's T>0 erasure, and quietly needs γ>0), and a set of notation defects/collisions (`H ∈ G`; coset-space-vs-dimension; group-vs-algebra phrasing; and the symbols α, J, ω, H all collide with load-bearing program conventions).

---

## §1. Correctness audit

### 1.1 Line-by-line verification of every mathematical statement

| # | statement (main.tex line) | verdict | evidence |
|---|---|---|---|
| 1 | Latent transforms `q ↦ R(α)q`, R(α) the SO(2) rotation matrix (l.12) | ✅ correct | standard; check C1 |
| 2 | Eq. (1): `V(R(α)q) = V(q) ∀α` defines SO(2)-invariance (l.13–16) | ✅ correct | definitionally |
| 3 | "symmetric ⇔ radial-only, V ≡ V(r)" (l.17) | ✅ correct (on ℝ², SO(2) orbits = circles) | — |
| 4 | r*=0 ⇒ vacuum invariant, `R(α)q* = q*`, Wigner–Weyl (l.19) | ✅ correct | q*=0 is the unique fixed point of every rotation |
| 5 | r*>0 ⇒ vacuum set `q* = r*(cos ω, sin ω) ∀ω`; `R(α)q* ≠ q*` while Eq. (1) holds (l.19) | ✅ correct | orbit check C2: `max|∇V(e^{αJ}q*)| = 3.6e-15`, V spread on orbit `1.6e-32` |
| 6 | "ground-state only invariant under **H ∈ G** (trivial subgroup here)" (l.19) | ⚠ notation defect T1 (content right: stabilizer of q*≠0 in SO(2) is {e}) | see §1.3 |
| 7 | "coset space G/H **corresponds to the number of** broken generators, **forms** the NG modes" (l.19) | ⚠ notation defects T2 (content: dim(G/H) = #broken generators = 1 here; NG modes are the tangent fluctuations along G/H) | see §1.3 |
| 8 | "Lie algebra corresponds to two-dimensional rotations by an angle α" (l.12) | ⚠ notation defect T3 (the **group** is the rotations; the algebra is span{J}, the infinitesimal ones) | see §1.3 |
| 9 | `R(α) = e^{αJ}`, J the single generator (l.19) | ✅ correct | check C1: `max|expm(αJ) − R(α)| = 2.2e-16` over 5 random α |
| 10 | Tangent to the vacuum orbit is `Jq*`; every `e^{αJ}q*` is another vacuum (l.19) | ✅ correct | d/dα e^{αJ}q*\|₀ = Jq*; check C2 |
| 11 | Eq. (2): differentiating stationarity along the orbit ⇒ `∇²V·Jq* = 0` (l.20–22) | ✅ **correct — the load-bearing equation** | ∇V(e^{αJ}q*)=0 ∀α ⇒ d/dα at α=0 gives ∇²V(q*)Jq* = 0. Check C2/C3: `|∇²V(q*)·Jq*| = 1.3e-16`; zero-eigenvector overlap with Jq*/‖Jq*‖ = `1.000000000000`; radial eigenvalue = `2λf²` exactly (`2.106000000000` at λ=1.3, f=0.9) |
| 12 | "Jq* is a Hessian eigenvector with eigenvalue zero … no curvature along the tangent" (l.23) | ✅ correct (given a G-invariant V and r*>0; note the eigenvalue is of the **bare** Hessian — see T4) | check C2 |
| 13 | "SSB gives massless Goldstone bosons … in ML a neutral direction where information is stored without being pulled back" (l.23) | ⚠ **claims-bearing; correct only under scope (T=0, γ>0, designed potential)** — full collision table §3 | — |
| 14 | "the unbroken realisation **doesn't give** a zero spectral mass, since … Jq* would be trivial **giving no information** about the eigenvalue" (l.23) | ⛔ **substantive S1: the first half is false as an implication; the second half is the correct statement** | check C5: `V = r⁴` is G-invariant with a **unique, G-invariant** minimum at the origin (Wigner–Weyl) and Hessian ≡ 0 ⇒ μ² = 0 with **no** SSB. Unbroken ⇒ the Goldstone argument yields no constraint; it does **not** ⇒ μ² > 0. (Generic case with a nondegenerate minimum: μ² > 0, Schur-degenerate within irreps — V2 §2.1 row 1.) |

**No equation fails verification.** The derivation chain (invariance → orbit of vacua → differentiate stationarity → zero mode along Jq*) is exactly the program's own "classical, tree-level Goldstone theorem" (V2 draft §2.1, second ¶) and is sound.

### 1.2 Substantive issues (complete list — separate from riders and nits)

- **S1 — Wigner–Weyl overclaim (l.23, final sentence).** "the unbroken symmetry realisation doesn't give a zero spectral mass" is false as stated: `V = r⁴` (or any invariant V with a degenerate-Hessian invariant minimum) is unbroken with μ² = 0 (check C5). The sentence's own tail — "giving no information about the eigenvalue" — is the correct claim. A one-word-class fix ("doesn't **force/guarantee** a zero spectral mass") reconciles it, and also aligns with V2 §2.1's WW row, which asserts μ²>0 *generically* (Schur degeneracy at a nondegenerate minimum), not as a theorem. ⛔ Flag only; not fixed.
- **S2 — the final ML sentence is unscoped** ("neutral direction in which information can be stored without being pulled back"). As physics of the *designed, T=0, γ>0* system it is correct and is precisely F5's latch theorem + Prop-kinblind territory; as written it carries no scope and collides with three binding registry objects (full table in §3). Classified substantive because the missing scope changes the claim's truth value on learned stores (N46) and at T>0 (CM-16b).
- **S3 — "stored" silently requires dissipation.** A flat direction alone gives *neutrality*, not a latch: at γ=0 the flat mode is a **marginal drifting integrator** (write never freezes; `q_n = q_0 + nεp_0/m`); any γ>0 converts it into the exact latch `q_∞ = q_0 + εp_0/(mγ)` (F5 `f5-note.tex` §3.1 Thm-latch, l.108–113). The primer's storage claim needs the γ>0 clause or a pointer to V2 §2's budget ("latch (μ=0, symmetry-protected)… γ>0"). Substantive because "flat ⇒ storable" without γ is wrong in the conservative cell of the program's own phase table (`v2-symmetry-deepdive` §1, NG/γ=0 row: "∞ but *unfrozen* (drifts)").
- **S4 — dichotomy of vacua is not exhaustive (l.19, minor-substantive).** "minimum at origin (non-degenerate)" vs "degenerate minima r*>0" omits mixed cases (invariant V can have minima at r=0 *and* on a ring), and "non-degenerate" conflates *unique minimizer* with *non-degenerate Hessian* (C5 separates them: unique minimizer, fully degenerate Hessian). Harmless for the pedagogical flow; flag so no downstream text quotes it as a classification.

### 1.3 Technical / notation defects (Advisor's three candidates verified, plus three more)

- **T1 (Advisor candidate, confirmed):** `H ∈ G` should be `H ⊂ G` (or H ≤ G) — H is a sub**group**, not an element; "subset symmetry group" → "subgroup". Content (H = {e} for SO(2) fully broken) is right.
- **T2 (Advisor candidate, confirmed + extended):** "The coset space G/H which corresponds to the number of broken generators, forms the NG modes" — (a) the *space* is conflated with its *dimension*: dim(G/H) = #broken generators (=1 here); (b) the NG modes are not the coset space; they are the fluctuation directions tangent to it (the fields parametrizing G/H). Standard fix: "one NG mode per broken generator; the NG fields parametrize G/H."
- **T3 (Advisor candidate, confirmed):** "the SO(2) group whose Lie algebra corresponds to two-dimensional rotations by an angle α" — the **group elements** are the rotations R(α); the **Lie algebra** is span{J}, the infinitesimal generators. As written it attributes the finite rotations to the algebra.
- **T4 (new):** "spectral mass" is used for a **bare-Hessian** eigenvalue. The program defines spectral mass on the **mass-whitened** Hessian: `μ_k² = λ_k(M_eff^{−1/2} ∇²V_θ M_eff^{−1/2})` (F5 §2.3 "Two masses", l.74–82; V2 §1 Nomenclature ¶). For the **zero mode** the two agree exactly for any invertible M (Sylvester congruence — F5 Prop-kinblind l.126–133; verified check C4: zero survives anisotropic M to `4.1e-16`, but the null **eigenvector rotates** to `M^{1/2}Jq*`, overlap with bare Jq* = 0.9707 ≠ 1, and the massive eigenvalue moves `2.106 → 0.497`). So the primer's μ²=0 claim transfers; any *quantitative* extension (GMOR, retention) must whiten or it will disagree with every V2 number.
- **T5 (new):** Eq. (2) omits the evaluation point — should read `∇²V(q*)·Jq* = 0`, with the α-derivative evaluated at α=0; and the "stationarity condition" it differentiates (`∇V(R(α)q*) = 0 ∀α`) is named but never displayed.
- **T6 (new):** the primer's `H` (unbroken subgroup) collides with `H(q,p) = T + V_θ` (the Hamiltonian) — the single most-used symbol in the program. V2 already lives with G/H vs H(q,p) in adjacent sections, but F5 writes the subgroup calligraphic (`\mathcal H`, f5-note.tex l.129); the primer should too if integrated.

### 1.4 Numerical sanity check (code + output, per acceptance criterion 2)

Script: `.claude/scratch/v2-colleague-physics-review/check_primer.py` (pure numpy f64 — JAX deliberately avoided per the cold-start note, protocol §4). Reproduced inline:

```python
"""Checks: C1 R(a)=expm(aJ); C2/C3 Mexican-hat vacuum Hessian: zero eigenvalue along Jq*,
positive radial 2*lam*f^2, primer eq.(2) directly; C4 whitened Hessian (program's spectral
mass) under anisotropic M: zero survives (Sylvester), eigenvector rotates; C5 Wigner-Weyl
counterexample V=r^4; C6 forward-compat: linear spurion gives K_theta = delta/r* (GMOR)."""
import numpy as np
rng = np.random.default_rng(20260818)
J = np.array([[0.0, -1.0], [1.0, 0.0]])          # primer's generator (== F5's X for SO(2))

def expm_series(A, terms=60):
    out = np.eye(A.shape[0]); term = np.eye(A.shape[0])
    for k in range(1, terms):
        term = term @ A / k; out = out + term
    return out

lam, f = 1.3, 0.9
def V(q):     return 0.25 * lam * (q @ q - f * f) ** 2
def gradV(q): return lam * (q @ q - f * f) * q
def hessV(q):
    r2 = q @ q
    return lam * ((r2 - f * f) * np.eye(2) + 2.0 * np.outer(q, q))
def hess_fd(Vf, q, h=1e-5):
    d = len(q); H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei = np.zeros(d); ej = np.zeros(d); ei[i] = h; ej[j] = h
            H[i, j] = (Vf(q+ei+ej) - Vf(q+ei-ej) - Vf(q-ei+ej) + Vf(q-ei-ej)) / (4*h*h)
    return 0.5 * (H + H.T)

# C1
worst = 0.0
for a in rng.uniform(-np.pi, np.pi, 5):
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    worst = max(worst, np.abs(expm_series(a * J) - R).max())
# C2/C3
w = rng.uniform(0, 2 * np.pi); qstar = f * np.array([np.cos(w), np.sin(w)])
K = hessV(qstar); evals, evecs = np.linalg.eigh(K)
t = J @ qstar; t = t / np.linalg.norm(t)
# C4
M = np.diag([0.31, 4.7]); Mih = np.diag(1/np.sqrt(np.diag(M))); Mh = np.diag(np.sqrt(np.diag(M)))
W = Mih @ K @ Mih; ew, evw = np.linalg.eigh(W)
v0 = evw[:, np.argmin(np.abs(ew))]; cand = Mh @ (J @ qstar); cand /= np.linalg.norm(cand)
# C5
K4 = hess_fd(lambda q: (q @ q) ** 2, np.zeros(2))
# C6
u = np.array([1.0, 0.0])
for delta in [1e-6, 1e-3, 1e-1]:
    r = f
    for _ in range(80):
        r -= (lam*(r*r - f*f)*r - delta) / (lam*(3*r*r - f*f))
    qd = r * u; tt = J @ qd / np.linalg.norm(J @ qd)
    K_theta = tt @ hessV(qd) @ tt      # ratio K_theta / (delta/r*) printed
```

**Observed output (verbatim, run 2026-08-18):**

```
== C1: R(alpha) = expm(alpha J) ==
max |expm(aJ) - R(a)| over 5 random alpha = 2.220e-16
== C2/C3: Mexican-hat vacuum Hessian ==
vacuum angle w = 1.482836,  |gradV(q*)| = 1.299e-16
analytic-vs-FD Hessian max dev = 2.601e-10
Hessian eigenvalues = [-1.37923926e-16  2.10600000e+00]   (predicted: 0 and 2*lam*f^2 = 2.106000000000)
|K @ (Jq*)| = 1.284e-16   (primer eq. 2)
|<zero-evec, Jq*_normalized>| = 1.000000000000 (should be 1)
max |gradV(e^(aJ)q*)| over orbit = 3.637e-15;  V spread on orbit = 1.602e-32
== C4: whitened (spectral-mass) Hessian, anisotropic M ==
eig(W) = [-4.10904366e-16  4.97054088e-01]   (zero survives; massive: bare 2.106000 -> whitened 0.497054)
|<null(W), M^(1/2)Jq*>| = 1.000000000000 (should be 1);  |<null(W), Jq*>| = 0.970666834633 (NOT 1 -> direction rotates)
== C5: Wigner-Weyl counterexample V = r^4 ==
unique G-invariant minimum at origin; Hessian eigenvalues = [8.e-10 8.e-10]  -> mu^2 = 0 WITHOUT SSB
== C6: forward-compatibility: linear spurion V - delta*(u.q) ==
delta= 1.0e-06: ... ratio = 0.999999999891
delta= 1.0e-03: ... ratio = 1.000000000000
delta= 1.0e-01: r*(d)=0.944177418887,  K_thetatheta=1.059122978369e-01,  delta/r* = 1.059122978369e-01,  ratio = 1.000000000000
```

(C5's `8e-10` is the h²-scale finite-difference floor of an exactly-zero Hessian.)

**Flag-provenance (mandatory).** Repo HEAD `7fcef50` (read-only; no tracked code touched, no checkpoints used). Script deterministic given `np.random.default_rng(20260818)`; main venv `/Users/user/Desktop/CHLU/.venv` python 3.11.13, numpy 2.4.1; float64 throughout; no JAX, no training, no config flags in play (analytic toy only). Command: `.venv/bin/python .claude/scratch/v2-colleague-physics-review/check_primer.py`.

---

## §2. Convention reconciliation (symbol-by-symbol)

| primer symbol / term | primer meaning | F5 note (`papers/f5-note/f5-note.tex`) | `v2-symmetry-deepdive` | V2 draft §2.1/§3 (`papers/v2-short/draft.md`) | verdict |
|---|---|---|---|---|---|
| `q` | 2-D latent position | `q ∈ ℝ^d`, position/content (§2.1 l.43) | same | same; trained units are **dim 4** = channel plane + spectators (§2 l.44) | ✅ compatible; primer's "two-dimensional latent space" is the *channel*, not the unit — say so if integrated |
| `α` | **rotation angle** | ⛔ **confinement coefficient** `α‖q‖²` (l.43); also the program-wide canonical constant in N150's `2α` lifetime ceiling ("α is the ceiling") | not used for angle | `α` not used as an angle anywhere; group parameter is `s` in F5 (`g_s`, l.118) | ⛔ **collision, load-bearing** — `2α` is a headline constant (N150 never-quote). Rename to `s` (F5) or `φ` on integration |
| `J` | SO(2) generator | ⛔ `J = DΦ` = **the map Jacobian** (`J^⊤ΩJ = (1−γ)Ω`, Prop-conf l.87–88); the generator is **`X`** (`Q_X = p^⊤Xq`, l.118–119) | generator written `X_a` (§7) | §2 quotes `J^⊤ΩJ=(1−γ)Ω` with J = Jacobian (l.40) | ⛔ **collision** — same letter for generator (primer) and Jacobian (F5/V2 §2, three lines above where the primer would land). Numerically primer-J ≡ F5-X for SO(2) (verified: `p^⊤Xq = q₁p₂−q₂p₁` gives the same matrix). Rename J → X on integration |
| `R(α)` | rotation matrix `e^{αJ}` | `g_s = e^{sX}` acting linearly, lifted `(q,p)↦(g_s q, g_s^{−⊤}p)` (l.118) | same | §2.1 speaks of the orbit abstractly | ✅ compatible (primer works on configuration space only; fine — the lift matters only for the charge, which the primer doesn't touch) |
| `r*` (`r^⋆`) | radius of degenerate minima | `r_∗` = **vacuum radius = order parameter**, explicitly **not** the decay constant (l.137) | `Σ = r*(δ)` = condensate; `F = √(M_ch)·r*` = decay constant (§2.2) | §3.1: "the orbit radius is the **condensate** Σ=r*, not the decay constant" (l.76) | ✅ compatible as used; ⚠ if extended to GMOR the F-vs-r* nomenclature (CM-15) is mandatory |
| `ω` | vacuum angle in `q* = r*(cos ω, sin ω)` | ⛔ `ω_k = μ_k` = **normal-mode frequency** (§2.3 l.79); also `ω_map` (l.97) | coset angle is `θ`/`ϑ` | coset/stored angle is `θ`; `ω` not used | ⛔ **collision** — ω is frequency everywhere in the estate (and data-frequency in the reference paper's Exp II). Rename ω → θ₀ or ϑ on integration |
| `H` (subgroup) | unbroken subgroup | ⛔ `H(q,p) = T + V_θ` = the Hamiltonian; subgroup is `\mathcal H` (Prop-kinblind l.127) | `G/H` plain (inherited) | `H=T(p)+V_θ(q)` (§1 l.19) *and* `dim(G/H)` (§2.1) — collision already latent in V2 | ⚠ divergence; use `\mathcal G/\mathcal H` (F5 house style) on integration |
| "vacuum expectation value" | name for the minimum | "vacuum radius, i.e. the order parameter" (l.137) | "the VEV / order parameter" (§2.2 table) — VEV used once, as gloss | §3.5: `r*` = "the order parameter" | ✅ compatible; prefer "order parameter r*" as primary term to match §3.5's condensate-melting narrative |
| "spectral mass" (zero) | bare `∇²V` eigenvalue | `μ_k² = λ_k(M_eff^{−1/2}K M_eff^{−1/2})` (§2.3 l.79) | same | §1 Nomenclature ¶ (l.23): spectral mass is the **whitened** eigenvalue | ⚠ divergence T4 — zero-mode equivalent (Sylvester; check C4), massive modes not; quantitative extensions must whiten |
| "Nambu-Goldstone modes", "Wigner-Weyl realisation" | as standard | — | three-realization taxonomy §1/§3 | §2.1 table rows 1–2 verbatim concepts | ✅ compatible, same taxonomy language |
| "massless Goldstone bosons … analogy to particle physics" | pedagogical bridge | F5 abstains from particle language in theorems | deep-dive §2.5: **no loops, no chiral logs, no anomalies** — tree-level only | §2.1 "What we do not import" ¶ (l.62) | ✅ compatible **iff** the primer stays tree-level (it does); the §2.1 scope ¶ must remain adjacent |

**Forward-compatibility with the pseudo-Goldstone / explicit-breaking case (explicit statement, as tasked).** The primer stops at the massless (exact-NG) case. Its *method* is forward-compatible: differentiating the **tilted** stationarity condition gives the pseudo-Goldstone curvature exactly — verified in check C6, where the linear spurion `V − δ(u·q)` yields tangential curvature `K_θθ = δ/r*(δ)` to `1.0000000000` (three δ over five decades), which is GMOR `μ²F² = δΣ` in bare-Hessian form at M = I. **But three things must be added, not merely appended, for the third realization (the home of V2's mode-mass budget):** (i) the whitening convention T4, or the μ² numbers won't match CM-15/§3.1; (ii) the `F = √(M_ch)r* ≠ r* = Σ` nomenclature (CM-15, F5 l.137); (iii) the scope riders of §3 below, because the moment the primer's flat direction acquires a small mass the N149/N150 never-quotes are in range. As shipped, the primer neither supports nor contradicts the pseudo-Goldstone row — it is silent, and V2 §2.1's table is strictly more complete.

---

## §3. Claims-collision table

Claims-bearing sentences identified: **(A)** "in the context of ML, it produces a neutral direction in which information can be stored without being pulled back" (l.23, final); **(B)** the "spectral mass of zero" framing (l.19/l.23); **(C)** "the unbroken symmetry realisation doesn't give a zero spectral mass" (l.23); **(D)** "This act of choosing a vacuum … is called spontaneous symmetry breaking" (l.19).

| sentence × registry object | collision? | mandatory rider (verbatim where registry supplies wording) |
|---|---|---|
| **A × N46 / CM-16a** (designed-only scope for any coset-register/latch claim) | ⛔ **YES** — "information can be stored" with no architecture scope reads as a claim about CLUs *per se*, including learned/emergent potentials. N46: the emergent arm has **no** such direction. | Rider (CM-16a scope column + N46 headline, verbatim): **"CM-16a: designed exact SO(2) ONLY — the emergent arm has no coset register (N46)"**; N46: *"the 'flat' direction is a mid-spectrum massive mode (1.7–4.9× softer than the stiffest mode); any written δ relaxes completely; capacity ≈1–1.6 bits, not a continuum. Cor-13 / CM-16(a) is designed-only."* Per N46's disposition this rider **must travel with every CM-16a citation** — and sentence A, if printed in V2, becomes one. |
| **A × CM-16b** (friction preserves / temperature erases) | ⛔ **YES** — "without being pulled back" is exactly the property that makes the flat direction the place where **T>0 diffusion erases**: no restoring force ⇒ unbiased random walk. The sentence needs a T=0 scope beside it. | Rider (V2 §2.1 l.58, the Coleman/Mermin–Wagner ¶, already-approved wording): *"the infinite half-life is a **T=0** statement about the deterministic damped map. At T>0 the coset coordinate performs a random walk with diffusion constant D_θ = εT(2−γ)/(2F²γ) … a diffusive register, not long-range order."* If any T>0 number is quoted, App J's mandatory flag travels verbatim: *"All results require `langevin_noise="fdt"` … **and a Newtonian kinetic mode**."* |
| **A × F5 Thm-latch (γ-scope; not a registry N-number but binding theory)** | ⚠ **YES (precision)** — a flat direction alone is neutrality, not storage: at γ=0 the mode is a marginal drifting integrator; storage (write freezes, ∞ half-life) requires γ>0. | Rider (F5 §3.1, l.113): *"any γ>0 converts it into an exact latch — a momentum impulse p₀ writes the finite displacement εp₀/(mγ) and the stored value then persists with infinite half-life"* — cite next to sentence A, or amend A to "…stored (once the write is damped to rest, γ>0) without being pulled back (T=0)." |
| **A × N149/N150** (pseudo-Goldstone tilt refuted in sign on a learned store; α is the ceiling) | **No direct collision** — the primer stops at the massless case and never claims a tilt/lifetime dial. ⚠ **Conditional (blast radius):** the primer's natural next sentence ("a small explicit breaking gives a small mass ⇒ long-but-finite lifetime") is exactly the N149-refuted territory on learned stores, and N150 makes *"ε is the manifold-payload lifetime dial, lifetime ∝ 1/ε"* a **never-quote without the 2α coercivity ceiling** (`τ_max = Γ/2α`). | If (and only if) an explicit-breaking extension is written: N149 rider — *"λ = ε holds only in the single-atom geometry it was specified in … the pseudo-Goldstone ruling survives as GEOMETRY; its shipped instantiation does not"* (learned-store tilt moves λ_min **the wrong way**: +0.0994 → −8.28); N150 rider — the `2α` ceiling clause verbatim. |
| **A × CM-17** (relativistic Gibbs no-go; novelty scope: cite, don't claim) | **No collision as written** — sentence A makes no sampling/thermal claim and names no kinetic mode. One-line reason: the primer is configuration-space-only physics; CM-17 lives in the momentum-marginal sampler. ⚠ Conditional: any T>0 wording added near it inherits CM-17's forbidden list (*never "fdt is exact discrete FDT / samples Gibbs" without a kinetic-mode qualifier*). | — |
| **A × Mo (2026), via CM-17's cite-don't-claim discipline + V2 §3.2/§4** | ⚠ **YES (novelty framing)** — "flat direction ⇒ neutral memory direction" in ML is published kinematics (Mo 2026: dim(G/H) zero Lyapunov exponents along the group orbit; V2 §3.3 even names it "Mo's neutrality theorem"). Sentence A must be presented as pedagogy with citation, not as a contribution of this paper. | Rider: cite Mo beside sentence A; and do **not** let A imply symmetry is *necessary* — the program's own sharper result is that the latch is a **modulus** of V_θ ("Equivariance of the full map is therefore *sufficient but not necessary* for a neutral memory direction", V2 §2.1 l.60, with the measured counterexample §3.4/App C). |
| **B ("spectral mass of zero") × CM-15 / V2 §1 nomenclature** | ⚠ **convention, not claim** — primer's zero is bare-Hessian; program's μ² is whitened (T4). Zero-mode transfer is exact (Sylvester / F5 Prop-kinblind; check C4) so no printed number is contradicted. | Rider: on integration, one sentence — "we quote the bare Hessian; the paper's spectral mass whitens by M_eff^{−1/2}, which preserves the zero mode exactly (Sylvester) but rescales massive eigenvalues." No registry citation exists against B; CM-15's F/Σ nomenclature becomes mandatory only if B is extended to GMOR. |
| **C ("unbroken ⇒ no zero spectral mass") × V2 §2.1 taxonomy row 1** | ⛔ **YES (internal consistency)** — V2's WW row says μ²>0 *degenerate within each irrep (Schur)* as the generic case at a nondegenerate minimum; the primer states it as a consequence of the argument. Counterexample in check C5. | Rider: reword to "is not forced to be zero — the orbit argument degenerates (Jq* = 0) and gives no information" (the primer's own second clause). Registry citation: none exists (this is a math error, not a measured claim); pointer = V2 §2.1 table row 1 + check C5 of this report. |
| **D ("act of choosing a vacuum…") × V2 §2.1 SSB-definition ¶** | ⚠ **YES (definitional discipline)** — V2 §2.1 spends a paragraph pre-empting *"there is no SSB in a 0+1-D system"* by defining SSB as *"the minimiser is not G-invariant"*, tree-level only, "no thermodynamic limit, no ħ". The primer's "act of choosing" heuristic, printed without that paragraph adjacent, re-opens the reviewer objection the deep-dive flagged as a condition of adopting the frame at all (`v2-symmetry-deepdive` §3.3 obligation 1 and §3.2: "half-adopting it is worse"). | Rider: wherever the primer lands, V2 §2.1's definition ¶ (l.56) must remain adjacent and controlling; the primer may not be the *only* SSB definition in the document. |

**Summary of mandatory riders if the primer enters V2 in any form: R1** = N46/CM-16a designed-only clause at sentence A; **R2** = T=0 scope + D_θ law pointer (CM-16b/App J) at sentence A; **R3** = γ>0 latch clause (F5 Thm-latch) at sentence A; **R4** = Mo citation + no-necessity framing at sentence A; **R5** = S1 rewording of sentence C; **R6** = §2.1 definition ¶ stays adjacent (sentence D). Conditional (only on extension to explicit breaking): **R7** = N149/N150 never-quotes; **R8** = whitening + F/Σ nomenclature (T4/CM-15).

---

## §4. Integration map (options with full cost printed; no ranking — Head + colleague decide)

**Option 1 — Replace part of §2.1 with the primer.**
- *Riders traveling:* all of R1–R6 land in main text (R1–R4 clustered at sentence A); R7/R8 latent.
- *Duplication:* the primer's derivation duplicates §2.1's second ¶ ("classical, tree-level Goldstone theorem … ∇²V_θ(q*) annihilates every orbit tangent") almost 1:1, but §2.1's version carries the finite-dim/Coleman discipline the primer lacks — replacement **loses** precision unless the two are merged, and the primer covers only 2 of the 3 realizations (no pseudo-Goldstone row, which is the budget's home).
- *Reconciliation required:* α→s/φ, J→X, ω→θ, H→𝓗 (four symbol renames, all collisions load-bearing per §2); S1 reword; whitening sentence (T4); realization table row 3 must be reinstated from the existing §2.1.
- *Venue cost:* neutral length-wise only if the existing ¶s are actually removed — which sacrifices approved, referee-cleared wording (v0.5 passed `v2-referee-3` with §2.1 as-is).

**Option 2 — Pedagogical lead-in feeding §2.1 (short "primer" sub-block before the taxonomy table).**
- *Riders traveling:* R1–R3 can be discharged by one forward-reference sentence ("scope and finite-T fate of this direction: §2.1 below and App J") **if** sentence A is softened to "a candidate neutral direction"; R4 (Mo cite) still mandatory at first occurrence; R5 mandatory; R6 automatic (the §2.1 ¶ survives).
- *Duplication:* deliberate (tutorial-then-precise); acceptable at a NeurReps/ML4PS pedagogy norm but costs ~0.4–0.5 pp in a 4–5 pp short whose main text is at budget — something else moves to an appendix.
- *Reconciliation required:* same four symbol renames; S1 reword; the "two-dimensional latent space" phrase needs the channel-vs-unit clarification (§2 row 1).

**Option 3 — Appendix primer (new appendix, e.g. "K: an SO(2) SSB primer for ML readers").**
- *Riders traveling:* R1–R4 as a single boxed scope note at the appendix head (the App-J mandatory-flag pattern already exists as house style); R5 mandatory; R6 satisfied by construction (main-text §2.1 unchanged).
- *Duplication:* fully tolerated in an appendix; zero disturbance to referee-cleared main text; cross-refs: App K → §2.1 (definition), → App J (T>0 fate), → §3.4 (sufficient-not-necessary).
- *Reconciliation required:* symbol renames still mandatory (an appendix using α, J, ω, H against the paper's own meanings would be reviewer-visible inconsistency); S1 reword still mandatory (it is a false statement wherever printed).
- *Cost:* the draft already carries 10 appendices (A–J) and a standing "placement pending" editorial note on App J; one more appendix touches the same pending Head decision.

**Option 4 — Not in this short (companion/tutorial note, V5, or thesis material).**
- *Riders traveling:* none bind V2; if the primer is published anywhere under the program's name, R5 (S1 falsehood) and R4 (Mo cite) still apply to the standalone text, and sentence A unscoped would still contradict the program's public registry positions (N46 is tier A on the published track).
- *Duplication/reconciliation:* none for V2.
- *Cost:* the colleague's contribution doesn't appear in V2; that trade-off is an authorship/priority decision that is explicitly not mine to weigh.

**Cross-option invariants:** (i) sentence A never prints without R1–R3 in some form; (ii) S1 never prints as-is; (iii) the four symbol renames are required in every in-paper option; (iv) nothing in the primer may become the document's only SSB definition.

---

## Editorial nits (separate list; not substantive, not riders)

1. `\label{symmetry_definig_eq}` — "definig" typo (invisible in PDF; visible to co-authors).
2. `equation \ref{...}` → `Eq.~\eqref{...}` (needs `amsmath`; currently no packages loaded — fine for `equation`, not for `\eqref`).
3. `$\mathrm{SO(2)}$` (l.12) vs `$SO(2)$` (l.19) — inconsistent typesetting of the same group.
4. No `\author{}`; `\date{August 2026}`.
5. British "realisation/realises" vs V2's American "realization" — harmonize on integration.
6. "is of the same potential as $q^\star$" → "has the same potential energy as".
7. "subset symmetry group" → "subgroup" (see T1).
8. Comma splice: "The coset space G/H which corresponds to …, forms the …" — restrictive clause needs restructuring (content fix is T2 anyway).
9. The stationarity condition is referenced but never displayed; a one-line `∇V(q*)=0` would make Eq. (2)'s derivation self-contained (T5).
10. "Considering the minimum of the potential being at the origin" → "If the minimum is at the origin".

---

## Verdict ledger (proven / evidenced / conjectured)

- **Proven:** primer Eqs. (1)–(2) and the orbit/tangent derivation (symbolic + C1–C3 numerics); zero-mode survival under whitening with rotated eigenvector (Sylvester + C4); S1's falsity (C5 counterexample); C6 forward-compatibility of the method to explicit breaking.
- **Evidenced (by the program's registries, cited):** the designed-only scope of any latch claim (N46, 3 emergent seeds); T>0 erasure of the flat direction (CM-16b, 25 cells); tilt-sign refutation on learned stores (N149).
- **Conjectured:** nothing in this report.

## Open questions / follow-ups / risks

1. **Who owns the S1 reword?** I flagged, didn't fix (task ⛔). The fix is one word-class but sits in the colleague's prose — Head/colleague action, not an agent edit.
2. If Option 1/2 is chosen, the reconciled text should be re-run past `paper-referee` — §2.1 as it stands is referee-cleared wording (v0.5→v0.6 CHANGELOG) and any replacement resets that.
3. The primer says "each architecture realises a different memory latch" (l.12, opening) — "each architecture" hints at a planned multi-architecture continuation not present in the 25 lines reviewed. If more colleague material is coming, this review covers only `main.tex` as of today (25 lines, "August 2026").
4. Downstream reconciliation list: **none** — no tracked doc currently quotes the primer; nothing to reconcile until an integration option is chosen (stated here per protocol §5 corollary; no owner needed yet).

## Proposed handover updates (for the Hub)

- **§7/§8 (or the shorts charter ledger):** record that the colleague primer is **mathematically sound at its core** (Eq. 2 verified to 1.3e-16) with one substantive overstatement (WW ⇒ μ²>0; counterexample V=r⁴) and one unscoped claims sentence whose mandatory riders are N46/CM-16a (designed-only), CM-16b (T=0 scope), F5 Thm-latch (γ>0), and a Mo citation. Four symbol collisions (α, J, ω, H) are load-bearing against F5/V2 house conventions and must be renamed in any in-paper integration.
- **Claims-matrix hygiene:** if any integration option prints sentence A, that instance becomes a CM-16a citation site and inherits N46's travel-with rider — worth a one-line note in the CM-16a scope column once placement is decided.

## Flags

- ⛔ **S1** — false implication in the primer's final sentence (unbroken ⇒ μ²>0); counterexampled; must be reworded before any publication use (V2 §2.1 row 1 pointer; check C5).
- ⛔ **Sentence A unscoped** — collides with N46/CM-16a (designed-only) and CM-16b (T>0 erasure); riders R1–R3 mandatory in every in-paper option.
- ⚠ **Symbol collisions α/J/ω/H** — all four load-bearing (N150's `2α`; F5's Jacobian J; F5's ω_k=μ_k; H = Hamiltonian); rename on integration.
- ⚠ **N149/N150 blast radius is conditional, not live** — the primer as shipped makes no tilt/lifetime claim; the never-quotes arm only if an explicit-breaking extension is written.
- ✅ **No equation failed verification**; the report's headline is the audit passing, with scoped defects.
