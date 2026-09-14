# PREREG — r2-geometry-revival (w25)

Written **before** any measurement harness was executed (only a JAX import warm-up and
`git worktree add` had run at write time). Protocol §5 pre-registration rule.

## DIAL DECLARATION (echoed from the task)
- **Dial:** capacity (the R2 law) — a law about the primitive; the figure is never framed
  as beating anything (CM-23(m)).
- **Laundering control:** the designed write at matched geometry must keep reaching its own
  wall. If a lever "works" only by making the learned write more designed → N46 scope
  collapse, not a win.
- **Falsifies:** §5.0 — trained widths < ~0.18, or minsep/width varying >2× across d.
  §5.2 — no movement of the wall at ≥3 seeds under an adequate budget.
- **Does NOT falsify:** failing to reach the designed `4·2^d` (the 4× prefactor gap is
  expected); any comparison to kNN/external methods.

---

## Stage 0 — the trained-width dump

### Quantities I will measure (definitions fixed now)
Written landscape only (global write, `learned_global` arm, w23 flags verbatim).

- **W1 `w_atom`** — the theorist's literal quantity: the trained `exp(log_width)` of the
  atoms that actually form the well at each stored site. "At the stored sites" is
  operationalised as: per site `x*`, rank atoms by their contribution
  `A_j·exp(-|x*-c_j|²/2s_j²)` (`A_j = amp_j²`), keep the atoms supplying the top 90 % of
  the summed contribution, and take the **contribution-weighted median** of `s_j` over
  that set; then the median over the K sites. (Reported also: unweighted all-atom median,
  and the init value 0.30 as reference.)
- **W2 `s_fit`** — the *geometric* well width that actually enters a sep/width ratio:
  least-squares fit of the radial profile of the REAL learned `V` around each site,
  `V(x*+r·u) = V0 + D·(1 − exp(−r²/2s²))`, `r` on a grid, `u` random unit directions,
  2 free params `(D, s)`. Reported with fit R². This is width-of-the-well, not
  width-of-an-atom, and can differ from W1 if the write builds a well out of many
  offset narrow atoms.
- **W3 `sep/width`** — `site_separation(designed_sites(d,K))` divided by W1 and by W2, at
  the last-PASS K and the first-FAIL K of the w23 budget-adequate ladder.

### Registered predictions (theorist's, adopted verbatim; my own bands added)
| # | prediction | source | falsifier |
|---|---|---|---|
| **S0.1** | median effective well width **≥ 0.28** (the write does not narrow the atoms below init 0.30) | theorist §5.0 | median < 0.18 |
| **S0.2** | `minsep(K_wall)/width ∈ [2.4, 3.1]` at **every** d | theorist §5.0 | ratio varies >2× across d |
| **S0.3** | d=8 K=64 (PASS) and d=6 K=64 (FAIL) trained sep/width **straddle** the transition window (d=8 high side ≥ ~2.9, d=6 low side ≤ ~2.7) | task §Stage 0 | both on the same side |
| **S0.4** *(mine)* | W1 will come in **just below** init: I register `w_atom ∈ [0.24, 0.31]`, i.e. I expect a small *downward* drift (weight decay 1e-4 acts on `log_width` too, and a narrower atom lowers the barrier term's cost) but not a collapse. If W1 lands in [0.18, 0.24] the theorist's ≥0.28 is FALSIFIED while the geometric account survives in weakened form — I declare in advance that I will report this as **"S0.1 fails, account survives"**, not as a pass. | mine | — |
| **S0.5** *(mine)* | W2 `s_fit` ≥ W1 (a superposition of positive-amplitude Gaussians of width s cannot be narrower than s — theorist §4.2 pt 2). I register `s_fit/w_atom ∈ [1.0, 2.0]`. A measured `s_fit < w_atom` would mean the fit or my definition is broken, not physics. | mine | — |
| **S0.6** *(mine)* | widths will be **anisotropic in one respect only**: the payload channel `q[d]` is written to a *value*, so I expect no systematic difference in `s_j` between address- and payload-dominant atoms. Registered as a null; a >1.5× split is a finding. | mine | — |

### Amendment A1 (written after the d=2 K=4 smoke cell only — declared)
The script was smoke-tested on the cheapest cell, **d=2 K=4 (last PASS), seed 0**, before
the full set launched; its output (`w_atom=0.2737`, `s_fit=0.2714`, `sep/w=4.23`,
fit R²=0.999) was seen before this amendment. Nothing below is tuned to it, but the
reading rule for S0.2 was ambiguous in the original text and I fix it now, before the
other 11 cells return:

**S0.2 reading rule.** "minsep(K_wall)/width" is evaluated at **both** the last-PASS K and
the **first-FAIL** K of the w23 budget-adequate ladder. The registered band **[2.4, 3.1]**
is tested against the **first-FAIL** ratio (the wall is where it breaks). Secondary and
sharper: there must exist a **single threshold t\*** such that every PASS cell has
sep/width > t\* and every FAIL cell has sep/width < t\*; the account predicts
t\* ∈ [2.4, 3.1]. "Varies >2× across d" is evaluated on the first-FAIL column.

**Stage-0 verdict rule (fixed now):** the geometric account SURVIVES iff
(S0.1 or S0.4-weakened) **and** S0.2 hold. If widths collapse below 0.18, or the sep/width
ratio spans more than 2× across d, I stop after Stage 0 and report N96's operator reading
as standing.

## Stage 1 — the one-flag revival (`atom_init_width` 0.30 → 0.15)

| # | prediction | source |
|---|---|---|
| **S1.1** | `K_learned(4)`: 16 → **64–128** | theorist §5.2 |
| **S1.2** | `K_learned(6)`: 32 → **≥128** | theorist §5.2 |
| **S1.3** | the wall moves by ~`2^d` in general | theorist §5.2 |
| **S1.4** *(mine)* | **I register a materially lower expectation than the theorist.** Halving the width halves the *achievable* separation threshold only if the write can still dig the wells at the new scale, and the w23/w24 evidence says the binding failure is placement (basin ≈ strict), not depth. My registered central estimates: `K_learned(4) = 32` (one rung, not 2–3) and `K_learned(6) = 64` (one rung). I register **P(the wall moves at least one ladder rung at d=4) = 0.6** and **P(the theorist's full `2^d` movement) = 0.2**. |
| **S1.5** *(mine, the noise-floor trade)* | width 0.15 = `query_sigma` exactly. I predict a measurable **robustness cost**: at cells that pass under BOTH widths, strict at width 0.15 will be **lower by 0.01–0.06** than at width 0.30, and the loss will be concentrated in the `basin_ok` term (jittered query lands outside the narrower basin), not in payload error. If capacity rises AND this trade appears, the finding is the capacity↔robustness trade, per task. |
| **S1.6** *(mine, laundering)* | the designed arm at matched geometry stays censored (≥256 at d ≥ 5) and is **unaffected** by `atom_init_width` (it does not read that flag). Any code path where the designed arm's number changes when only `atom_init_width` changes is a bug, and I will report it as such. |

**Budget-adequacy rule (N92, fixed now):** every first-fail cell is re-checked at 2× atoms.
A cell that passes at 2× atoms is reported as **budget-limited (lower bound)**, never as a wall.

**Declared compute deviation, registered in advance:** the theorist's coverage spec
`min_atoms_base × 2^{d/2}` implies a floor `512·2^d` (d=6 → 32768, d=8 → 131072 atoms).
The measured w23 write cost at d=8 K=64 / 16384 atoms is ~1340 s **per write per seed** and
each cell writes twice (written + blank). A d=8 cell at 131072 atoms is therefore ~3 h per
write per seed ⇒ ~18 h for 3 seeds for ONE cell. I register now that I will run Stage 1 in
priority order **d=4 → d=5 → d=6 → d=8**, apply the full coverage raise where affordable,
and **report any dimension I could not reach at the specified budget as NOT RUN**, not as a
null. Partial Stage-1 coverage is declared, never silently interpreted.
