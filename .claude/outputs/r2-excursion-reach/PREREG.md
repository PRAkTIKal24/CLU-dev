# PREREG — r2-excursion-reach (w26, experiment-engineer)

Written **before** any harness that measures a registered quantity was run.
What preceded it: (i) a JAX warm-up + one **regression** cell on the *shipped* default
config (`learned_global`, d=4, K=16, seed 0, 2048 atoms → strict `0.865234375`), used
only to time the machine and, after the refactor, to prove the refactor is
bit-identical; (ii) the codebook-geometry unit check in §2 below (pure numpy, no
dynamics). Neither measures a registered quantity.

Base: local `main` @ `ff85573`. Branch `agent/experiment-engineer/r2-excursion-reach`,
worktree `../CHLU-r2reach`, **main venv reused** (JAX 0.9.0, equinox 0.13.4).

---

## 0. DIAL DECLARATION (echoed from the task)
- **Dial:** capacity (the R2 law). A law about the primitive; its figure is never
  framed as beating anything (CM-23(m)).
- **Laundering control:** the designed write at matched geometry must keep reaching
  its own wall, `K_designed(4) = 128`, **at every payload format I introduce**.
- **Falsifies:** neither excursion arm moves the wall at ≥3 seeds under a
  budget-adequate atom count **with payload read-noise ON**.
- **Does NOT falsify:** failing to reach the designed `4·2^d`; any comparison to kNN
  or external methods.

## 1. THE FIVE BINDING FAIRNESS CONDITIONS, as checkable items
| # | condition | how it is satisfied, and where it is checked |
|---|---|---|
| 1 | **bits-per-item constant** | Every arm stores one of **K** codewords whose **minimum pairwise separation is exactly `Δ = 2/(K−1)`** — the separation of the shipped `linspace(−1,1,K)` codebook. Same K ⇒ same `log₂K` bits; same Δ at the same per-axis read noise ⇒ same discriminability. **Checked numerically in §2 for every (m,K) I run.** The multi-channel code lowers only `max‖a‖` (the *reach demand*), never Δ. |
| 2 | **byte accounting pinned** | `m` channels cost `m−1` extra latent coordinates ⇒ `n_atoms·(m−1)` extra learned floats. `n_learned_params` and `param_bits_budget` are reported for **every** cell, and a **spectator control** (m channels allocated, code written in channel 0 only, excursion left at 1.0) separates "extra dimensions" from "the code". |
| 3 | **payload read-noise ON** | Two independent switches, both default-off and both ON in every headline Stage-B number: `payload_launch_sigma` (the query's payload channels launch at `N(0,σ)` instead of *exactly* 0 — the store must denoise it) and `payload_obs_sigma` (additive observation noise on the read-out value — the store *cannot* denoise it). Headline: **σ_launch = 0.05, σ_obs = 0.01**, swept over σ_obs ∈ {0, 0.005, 0.01, 0.02}. **The value criterion becomes `pass_metric="decode"`** (nearest-codeword decoding) because an *absolute* error tolerance is structurally blind to the codebook spacing and therefore neither rewards nor punishes a change of excursion — which is exactly the hole w25 fell through. |
| 4 | **baselines get the same format** | `BallRegisterPotential` is generalised to `(K,m)` payloads and `inflate_potential` anneals the designed well too, so the designed arm reads the *same* code through the *same* schedule. |
| 5 | **laundering travels** | The designed arm is re-measured at every format change; `K_designed(4)=128` must hold. |

**Extra control that condition 3 demands and that I am registering as decisive:** the
**w25 `pscale=0.5` arm** (codebook *and* tolerance halved together — the manipulation
that took d4K32 from 0.824 to 1.000 in a noise-free harness) is run **inside the same
noise sweep**. Registered: it is a free lunch at σ_obs = 0 and **dies** as σ_obs rises,
because it halves Δ; the multi-channel arm does not, because it holds Δ fixed. If the
multi-channel arm dies with it, arm (a) is dead too and I will say so.

## 2. The codebook geometry (numpy check, run before any dynamics)
`payload_codebook` places the K codewords on the Δ-spaced integer lattice in `R^m`,
keeping the K smallest-norm points, then applies the `designed_payloads` permutation.
Measured (min separation must equal Δ exactly; max norm is the reach demand):

| m \ K | 16 | 32 | 64 |
|---|---|---|---|
| **Δ (all m)** | 0.13333 | 0.06452 | 0.03175 |
| max‖a‖, m=1 | 1.0000 | 1.0000 | 1.0000 |
| max‖a‖, m=2 | 0.2828 | 0.1881 | 0.1571 |
| max‖a‖, m=4 | 0.1333 | 0.0912 | 0.0550 |

⚠ **The honest cost of arm (a), registered in advance:** an m-dimensional lattice code
has more nearest neighbours (kissing number) than a line, so at the *same* Δ and the
*same* per-axis σ its union-bound decode error is **larger** by that factor (≈2 → ≈8
at m=4). Arm (a) is therefore mildly *penalised*, never flattered, by condition 3. I
keep σ_obs in a range where this term is small (Δ/2σ ≥ 3.2 at K=32, σ_obs = 0.01 ⇒
union error ≤ 0.006) and report the whole sweep.

## 3. Registered predictions

### Stage A — the 2×2 init×width factorial (d=6, K=64, monolithic, 3 seeds, 4096 atoms, m=1, no anneal, no noise, `strict` = shipped tol metric, value-blank on every cell)
| # | quantity | registered value | derivation |
|---|---|---|---|
| A0 | baseline cell (local=F, w=0.30) | **0.85 ± 0.10** | w23 d6K64 = 0.818 at the w23 budget; N98 reports the same cell as its test bed |
| A1 | **main effect of `atom_init_local`** at w=0.30 | **+0.051** (±0.04) | *N98's own measured monolithic value at this exact cell* |
| A2 | **main effect of width** 0.30→0.15 at local=F | **−0.25** (±0.15), sign certain | r2geom measured −0.342 at d4K16 with the same flag |
| A3 | ⭐ **INTERACTION** `Δ(local | w=0.15) − Δ(local | w=0.30)` | **0.00 ± 0.05 — the two levers are ADDITIVE, i.e. TWO effects** | mechanism: the localized init moves atoms along **address** axes only (N46 constraint, by construction), while the width-0.15 collapse is a **payload-axis reach** failure (r2geom §4: corr(strict,\|a_i\|) = −0.887, small-\|a\| items retrieve at 1.000 at *both* widths). An address-side lever cannot repair a payload-side failure. |
| A3′ | **Advisor-2's registered alternative** ("substantially ONE effect") | interaction ≥ **+0.15**, i.e. localization *rescues* the narrow width to ≥0.85 | recorded so the cell adjudicates between us; **P(my side) = 0.70** |
| A4 | best cell of the four | local=T, w=0.30 | — |
| — | ⛔ | **no lever becomes a default in this task** (Head B1.4) | — |

### Arm (b) — the annealed / continuation read (d=4, width 0.30, 8192 atoms = w25's coverage-raised budget, 3 seeds; schedule `s_eff(t)=√(s²+s_extra(t)²)`, `s_extra` linear to 0 over L=4 stages, address+read phases, **equal total Verlet steps** by construction)
| # | registered | value |
|---|---|---|
| B0 | baseline reproduction | K=16 → **0.937 ± 0.02**, K=32 → **0.824 ± 0.02** (r2geom) |
| B1 | ⭐ K=32, `s0=0.30`, amplitude mode, noise OFF, tol metric | **0.93** [0.85, 0.99]; **P(≥0.90) = 0.50** |
| B2 | K=16, same | **≥0.97**, P = 0.60 |
| B3 | **mass mode** (exact Gaussian convolution, depth falls as `(s/s_eff)^dim`) | **≤ baseline + 0.02** — predicted null-or-loss; amplitude mode is the lever |
| B4 | schedule shape | an optimum at `s0 ∈ [0.20, 0.40]`; at `s0 ≥ site_sep/2` (0.355 at K=32) the wells merge in the ADDRESS space and the read degrades |
| B5 | **noise ON** (σ_launch .05, σ_obs .01, decode) | the annealed *gain* persists to within **0.03** of the noise-OFF gain |
| B6 | the wall | `K_learned(4)`: 16 → **32** under the annealed read, **P = 0.45**; → 64, P = 0.15; unmoved, P = 0.40 |
| — | force argument behind B1 | at `s≈0.31`, `r=|a|max=1`: `exp(−r²/2s²)=5.5e−3`; at `s_eff=0.60`: `0.25` ⇒ **~45× more force at the launch manifold** (amplitude mode). Under mass mode the same inflation costs `(s/s_eff)^5 = 0.036`, i.e. the net force gain is ~1.6× — hence B3. |

### Arm (a) — the multi-channel payload (d=4, width 0.30, 8192 atoms, grid code, 3 seeds, **noise ON**, decode metric)
| # | registered | value |
|---|---|---|
| C1 | ⭐ K=32, m=4 vs m=1 | m=1 ≈ **0.82**; m=4 **≥ 0.95**, **P = 0.55** (reach demand falls 11×: max‖a‖ 1.000 → 0.091) |
| C2 | K=32, m=2 | intermediate, **0.88–0.95** (max‖a‖ 0.188) |
| C3 | **spectator control** (m=4 dims, code in channel 0, excursion 1.0) | within **±0.03** of m=1 ⇒ the gain is the *code*, not the extra dimensions/params |
| C4 | **w25 `pscale=0.5` control** | gains ≥0.10 at σ_obs=0 and **loses ≥0.10 at σ_obs=0.02**; the multi-channel arm loses **<0.03** over the same sweep |
| C5 | laundering | `K_designed(4) = 128` at m=1, 2, 4 and under the annealed read |
| C6 | the wall | `K_learned(4)`: 16 → **≥32** at m=4 with noise ON, **P = 0.55** |

### Global falsifier (task-level)
If **neither** arm moves the wall at ≥3 seeds, budget-adequate (N92 re-check at 2×
atoms at every first-fail cell), **with payload read-noise ON**, the R2 excursion
route is closed and I report it as a clean close with a named mechanism.

## 4. Compute priority order (declared before starting; NOT-RUN ≠ null)
1. **Stage A** (d=6 K=64, 12 cells) — cheap, gates a default.
2. **Arm (b) at d=4** — read-only, so ONE write per (K,seed) serves every schedule ×
   noise level. Cheapest decisive evidence in the wave.
3. **Arm (a) at d=4** — needs its own write per format.
4. d=6 for either arm — 1 seed if reached.
5. d=8 — expected **NOT RUN** (w25: ~1340 s/write/seed at d=8/16384 atoms; 4 engineer
   worktrees share 8 cores).
Anything not reached is reported as **NOT RUN**, never as a null.

## 5. Provenance of every number I will report
Flag-provenance table in the report: commit, seeds, `atom_init_local(_mult)`,
`atom_init_width`, `min_atoms_base` (atom budget), `n_payload_channels`,
`payload_code`, `payload_launch_sigma`, `payload_obs_sigma`, `pass_metric`,
`read_anneal_*`, plus the shipped write/read schedule. `langevin_noise` is **N/A**
(deterministic Verlet; no temperature anywhere in this task).
