# f5-corrigendum — physics-theorist report

**Task + acceptance criterion:** fix the proven kinetic-isotropy error at all three F5 sites (blindness theorem in place of the false clause), re-scope the over-covering `[proven; verified (g)]` tag, fix the decay-constant naming, state what this unblocks. **Acceptance met.**
**Status:** done. F5 is **arXiv-ready on this axis**; the blocking claim is gone from `f5-note.tex` and the note now carries a strictly stronger result.

---

## What I did

1. **Re-derived the correction independently** (did not take the deep-dive on trust) and proved a slightly stronger form than the one handed to me:
   - `ker W = M^{1/2} ker K`, `rank W = rank K`, **`inertia(W) = inertia(K)`** (Sylvester, not just rank);
   - the latched **physical** `q`-direction is **`ker K` itself — not even rotated** by `M` (new relative to the deep-dive);
   - **global** latch statement: every vacuum point with `p=0` is an *exact fixed point* of the damped-Verlet map for **any** `M` (`∇V=0 ⇒ p_½=p=0 ⇒ q'=q`) — so the latch survives anisotropy non-perturbatively, not just at linear order;
   - **no-secular-drift is elementary**: `H` conserved + `V_θ` coercive ⇒ compact orbit ⇒ `|Q_X| ≤ sup|q||p|`. Boundedness of the charge does not need the ring calculation at all. The ring closed form is then the *exact envelope*.
2. **Verified everything numerically** (three self-contained numpy scripts, §"How I verified").
3. **Ran the F5 check-(g) forensics** the task asked for (item 2) — and found more than expected: the "2.6" is **not reproducible** (see F-2 finding below).
4. **Edited all three F5 sites** + the tag + the naming; rebuilt the PDF; added a CHANGELOG entry.

---

## How I verified

`uv run --no-project --with numpy python .claude/scratch/f5-corrigendum/{verify,verify2,verify3}.py` — numpy 2.0.2, float64, `default_rng(42/7/11)`. Map semantics mirror `chlu/core/integrators.py::velocity_verlet_step` (KDK, then `p←(1−γ)p`). **Repo-read-only; no checkpoints, no training config** (⇒ the flag-provenance table below is a *map-parameter* table; no training flags exist for these claims).

| check | claim | observed |
|---|---|---|
| A2 | flat `μ²` under **diagonal** anisotropic `M` (= the coded `log_mass`) | **`0.0` exactly, bit-level**, 8 random `M` |
| A | flat `μ²` under **general non-diagonal SPD** `M` (cond ≤ 14.7) | `≤ 7.2e-16` (eigensolver round-off); `#zero(W)=2=#zero(K)`; inertia preserved 6/6 |
| A | `ker(W) = M^{1/2} ker(K)` | `‖W·(M^{1/2} ker K)‖ = 1.2e-15` |
| H | flat **physical** dirs = `ker K` (unrotated) | max principal-angle sine `3.3e-16`, 6 random SPD `M` |
| H | latch endpoint `x_∞ = εp₀/(mγ)` on the flat mode | `≤ 2.1e-15` |
| B | latch under anisotropy, `γ=0.05`, 2×10⁴ steps | coset-angle drift **`0.0` exactly** for `M=(1,1)`, `(1,2)`, `(0.31,4.7)` |
| — | `Q̇ = pᵀXM⁻¹p` (= `p₁p₂(M₁⁻¹−M₂⁻¹)` for `SO(2)`) | agrees with the flow derivative to `0.00e+00` |
| D2 | ring closed form `Q = √(2E)·F(ϑ)`, `F²=r²(M₁sin²+M₂cos²)` | amplitude ratio **1.0007 / 1.0025 / 1.0064**; **elliptic** half-period ratio **1.00045 / 0.99935 / 0.99971** (`M`=(1,2),(1,4),(0.5,3)) |
| E2 | boundedness ⇒ no secular drift | `\|H_n−H_0\| ≤ 2.2e-5` ∀n≤5×10⁵; `sup_n\|Q_n\| = 0.582` < compact bound `0.820` |
| C2 | check-(g) forensics, 10⁶ steps | per-decile `sup\|L\|` slope **`+6.1e-4`**; midrange slope `−6.7e-4`/decile = `5.7e-3` of the range ⇒ **envelope stationary** |
| G | naming | `F² = Q/ϑ̇ = M_eff r*²` exactly; `μ²F² = δn²` exactly |

**Flag provenance.** Repo `27f232f` (untouched). Map params: blindness `λ=3, f=1`, `M` random SPD/diagonal; latch `ε=0.05, γ=0.05`; charge `λ=1, r₀=1, ε=0.02, γ=0, q₀=(1,0), p₀=(0.4,0.3)` (F5's literal check-(g) init); ring `λ=400, ε=0.002, γ=0, ϑ̇₀=0.3`. Seeds `42/7/11`. No `chlu` import, no JAX, no checkpoint, no training flags.

---

## Findings

### 1. The correction (proven, and stronger than specified)

**Prop-17 / `prop:kinblind` (kinetic-spurion blindness).** For `V_θ` `G`-invariant with vacuum `q*`, stiffness `K = ∇²V_θ(q*)`, and **any** `M_eff ≻ 0` (need not commute with the group action), `W := M_eff^{−1/2} K M_eff^{−1/2}` satisfies

```
ker W = M_eff^{1/2} ker K,   rank W = rank K,   inertia(W) = inertia(K).
```

Hence **every flat direction keeps `μ² = 0` exactly, for any anisotropy.** The vacuum manifold, the channel count `dim(G/H)`, and the `γ>0` latch are intact; the latched physical direction is `ker K` itself.
*Proof:* `W = CᵀKC`, `C = M_eff^{−1/2}` invertible ⇒ congruence ⇒ Sylvester's law of inertia. Globally, vacuum×`{p=0}` are exact fixed points of the map for any `M`. ∎ **Verdict: proven + verified.**

**What anisotropy actually does.** `Q̇_X = pᵀXM⁻¹p ≠ 0` iff `[M,X] ≠ 0`. But `H` conserved + `V_θ` coercive ⇒ compact orbit ⇒ `|Q_X| ≤ sup|q||p|`: **secular drift is impossible.** On the vacuum orbit the excursion is exactly periodic:
`Q = F²ϑ̇ = √(2E)·F(ϑ) ∈ [√(2E) r*√M_min, √(2E) r*√M_max]`, amplitude `√(2E) r*(√M_max−√M_min)`, **period = half a revolution**.

**Design rule, corrected.** An anisotropic channel **still latches, with infinite half-life**. What it loses is a *conserved write current* (the write gain becomes `ϑ`-dependent). Tie the channel masses for a clean `ϑ`-independent current (and, at `dim(G/H)≥2`, a degenerate pNG multiplet) — **not** to save the register. *To detect kinetic symmetry breaking, measure the charge law: the Hessian and the latch are provably blind to it.* (This is exactly what `v2-full-runs` item 5 measured: `μ²_ang ~ 1e-15`, angular `n₁/₂ = ∞`, write-freeze `= 0`.)

### 2. Check (g)'s "2.6" — worse than "over-covering" (F-2, resolved here)

The tag `[proven; verified (g)]` covered only the *charge* half. Two further problems, both now fixed in the sources:

- **"2.6" is a running-supremum excursion** `max_n |L_n − L_0|/|L_0|`, not a drift rate. As a running supremum it is monotone in the window: **2.62 at 10⁵ steps → 2.96 at 10⁶ steps.** Quoting it as "drift" invites exactly the wrong reading.
- **It is not reproducible.** Two *algebraically identical* spellings of `∇V` — F5's `lam*(q[0]**2+q[1]**2−r0**2)*q` vs `lam*(q@q−r0**2)*q` — give **2.625 vs 2.734** at 10⁵ steps. The off-ring check-(g) orbit (soft ring `λ=1`, large radial momentum) is a **chaotic rosette**, so any point statistic on it is float-order-of-operations dependent. What *is* stable: the envelope `[−0.5, +0.5]`, its stationarity (per-decile `sup|L|` slope `+6.1e-4`), and `sup_n|L_n| = 0.582 < 0.820` (the compact bound).
- The clean, quotable object is the **ring** closed form (verified to 0.07–0.6% in amplitude and 0.03–0.07% in the elliptic half-period). Note the deep-dive's amplitude formula is an **on-orbit** statement; off-orbit only boundedness holds. I stated it that way in the sources.

**Verdict:** the deep-dive's F-2 conjecture ("2.6 is very likely the bounded-oscillation envelope") is **confirmed as to boundedness/non-drift**, and **refined**: for check-(g)'s specific off-ring init it is an excursion of a chaotic orbit, *not* the clean `0.414·√(2E)·r` ring envelope (which requires the near-rigid-ring initialization). Both readings kill "drift".

### 3. Decay-constant naming (7.16)

`F := √(M_eff)·r*` is the object with `Q = F²ϑ̇` on the vacuum manifold and `μ²F² = δn²` (verified exactly). F5's formula `μ² = δn²/(M_eff f²)` was already correct with `f ≡ r*`; only the **name** was misplaced. Now: **`F`** = decay constant, **`r*`** = vacuum radius (= condensate `Σ`, the order parameter).

---

## Every edit (change-log)

All files are under gitignored `.claude/**`.

### `.claude/outputs/formalism-note.md` (internal F5 source)
| line(s) | edit |
|---|---|
| §4.1 header | tag `[proven; verified (g)]` → `[proven; verified (g), (g′)]`; **added a dated Corrigendum blockquote** naming the false corollary and the over-covering tag |
| §4.1 "Kinetic isotropy condition" | retitled *"…a condition on the **current**, not on the **register**"*; **false clause deleted**; requirement re-scoped to "`M₁=M₂` **for `Q_X` to be conserved**"; inline tag re-scoped to the charge half only |
| §4.1 (new) | **Prop-17 (kinetic-spurion blindness)** + proof (Sylvester; global fixed-point argument) |
| §4.1 (new) | "What the kinetic spurion perturbs instead: the Noether current, boundedly" — `Q̇=pᵀXM⁻¹p`, compactness bound, ring closed form, amplitude, half-revolution period |
| §4.1 (new) | corrected **Design consequence** + restated falsifiable ("measure the charge law, not the Hessian/latch") + `[proven; verified (g′)]` |
| §4.1 (new) | ⚠ **Retraction of a number**: "2.6" is a running-supremum excursion; non-reproducible; envelope stationary |
| §3.3(c) | `μ² = δn²/(M_eff f²)` → `…/(M_eff r*²) = δn²/F²`, decay constant `F := √(M_eff) r*` defined; clarified the breaking is **of `V_θ`, not `T`** (Prop-17) |
| §4.3 (HEFT) | "coset metric with decay constant `f`" → "decay constant `F = √(M_eff) r*` (**not** the orbit radius)" |
| §8 glossary | `f` row → `r*` (vacuum radius / condensate `Σ`) **and** `F := √(M_eff) r*` (decay constant) |
| §8 (new) | **Decay-constant naming rule** box (formulae were right, name was wrong; "`f` buys robustness" → "`F² = M_eff r*²` buys robustness") |
| App-N table | row (g) re-scoped ("excursion … *not* a drift rate"); **new row (g′)** with all blindness numbers |
| App-N header | pointer to the new verification scripts |
| §9 provenance block, item 8 | **struck through** (`~~…~~`) + `[SUPERSEDED 2026-07-09 by Prop-17 — do not propagate]` with the corrected statement. *(Historical block marked "retained for provenance only" — per handover §5 discipline I struck rather than rewrote.)* |

### `.claude/outputs/f5-arxiv-note.md` (arXiv-bound markdown)
| site | edit |
|---|---|
| abstract | *"give the kinetic-isotropy (Schur) condition an equivariant memory channel must satisfy"* → *"…a conserved write current requires, showing that this condition binds the **current alone**: by Sylvester's law of inertia a flat direction keeps `μ²=0` under any inertia, so the latch is blind to kinetic anisotropy…"* — **the old clause implicitly carried the error** (it says a memory channel *must* satisfy isotropy; it must not). "All twelve results" → "All results". |
| §0 scope list | added "+ the kinetic-spurion blindness proposition (corrigendum)" |
| **§3.2:139** | **false clause deleted**; +**Proposition 5′** (blindness) with proof; +bounded-current paragraph with ring closed form and the `F := √(M_eff) r*` naming; +corrected design consequence |
| §3.2 verified | check (g) re-scoped; **new `[verified — check (g′)]`** paragraph |
| §8 | one clause: isotropy ⇒ equivariance of the *map*; by Prop 5′ the latch persists without it (prevents an apparent internal contradiction with the new proposition) |
| results table | row (g) re-scoped; **new row (g′)** |
| coverage table | row 5 re-scoped "(on the **current**)"; **new row 5b** (blindness) |

> **Numbering note:** the `.md` uses one hand-maintained counter (Prop 1, Cor 2, Cor 3, Thm 4, Prop 5, Thm 6, …). I numbered the new result **`5′`** to avoid renumbering Thm 6 → Prop 14 and their ~12 in-text cross-references. The `.tex` auto-numbers, so it renders as **Proposition 3** there; every `.tex` reference is a `\ref`, so nothing broke.

### `.claude/papers/f5-note/f5-note.tex` (**the copy that gates the push**)
Same four changes as the `.md`: abstract clause; `\begin{proposition}[kinetic-spurion blindness]\label{prop:kinblind}` + `\begin{proof}`; bounded-current paragraph + design consequence; §8 equivariance clause; `Verified (check g)` re-scoped + new `Verified (check g′)`; verification-table row `(g′)`; provenance appendix now carries an explicit **Corrigendum note** naming the retracted claim and the retracted "2.6", plus the new script paths and the check-count reconciliation ("13 legacy checks (a)–(m) … (g′) is new; 14 table rows").

**Build:** `tectonic -X compile f5-note.tex` → clean, **0 unresolved refs**, 10 → **11 pages**. `f5-note.pdf` refreshed in place. `CHANGELOG.md`: new **v0.4 corrigendum** entry at the top.

---

## What this unblocks

> **F5's arXiv push is unblocked on this axis.** The unhedged false clause at `f5-note.tex:124` is gone; in its place the note carries a proposition that is *strictly stronger* (an exact, one-line-proof structural theorem about the whole class), is verified to machine precision, and — unlike the claim it replaces — **agrees with the program's own measurements** (`v2-full-runs` item 5). The note also stops asserting, in its abstract, that an equivariant memory channel *must* be kinetically isotropic.

---

## Downstream references to reconcile (FLAGGED, not edited — per task scope)

| where | text | verdict | action |
|---|---|---|---|
| **`claims_matrix.md`** | *no CM row inherits the false clause* — checked all rows for `isotrop`/`Schur`/`pseudo-Goldstone`. CM-14 mentions "isotropic squeezes" (unrelated). | ✅ **clean** | none |
| `negative_results.md` **N4** | "Kinetic symmetry-breaking is *invisible to the Hessian and to the latch* and shows up only in the charge law" | ✅ **already correct** — it is the blindness theorem, measured | none |
| `negative_results.md` **N4** | "Unequal-$M$ Noether-charge **drift** = 2.6 ($O(1)$) in the F5 toy (check g)" | ⚠ **wording + retracted number** | → "bounded $O(1)$ charge non-conservation (an excursion, not a drift)"; **drop the 2.6** (non-reproducible, window-dependent) |
| `negative_results.md` **N4** mechanism | "unequal inertial masses **silently break the channel** through the kinetic term" | ⚠ **wording** | they break the *symmetry of `H`* / the *current*; they do **not** break the *channel* (the register survives) |
| **`papers/v2-short/draft.md:107`** | "the Noether-charge **drift** scales linearly with the split" | ⚠ **7.17, already tasked** to `v2-revision-4` item 5 | "bounded **oscillation**, amplitude `√(2E)r(√M_max−√M_min)`, linear in the split at small split". Note the measured `5.4/1.6/0.082e-2` are amplitudes; `A/‖[M,X]‖` is *not* constant (deep-dive §4.1b) |
| `papers/v2-short/draft.md:141,155` | "the kinetic-isotropy (Schur) condition **as the price of an equivariant channel**" | ⚠ **mild** | → "the price of an equivariant **write current**". Not false, but it invites the retracted reading |
| `papers/v2-short/draft.md:227` | N4 summary line | ✅ correct as written | none |
| `tie_channel_mass` (`chlu/experiments/exp_d_goldstone.py` flag) | **no "else pseudo-Goldstone" justification found in code or CM** | ✅ clean | the flag is still useful — it buys a conserved, `ϑ`-independent write current — but its *rationale string* anywhere it appears should not claim it protects the register |
| `future_work.md`, `v2-symmetry-deepdive.md` | reference the error as an open item | ✅ | Hub may mark R3/S6 **closed** |

**No short was edited** (task §Scope discipline). `v2-revision-4` already carries items 5 (App-C oscillation) and the naming fix; the two extra wording flags above (`N4`, "price of an equivariant channel") should be folded into it.

---

## Open questions / risks

1. **`.md` vs `.tex` proposition numbering now differs** (`5′` vs auto `3`). Intentional and documented in-line, but if the Hub prefers, the `.md` can be fully renumbered (Thm 6→7, Props 7–14→8–15, ~12 cross-refs) — I judged the churn/benefit ratio bad for a corrigendum.
2. **The note grew 10 → 11 pages.** If there is a page target for the arXiv submission, the ring-closed-form paragraph is the compressible one (the proposition + proof are not).
3. **I added one clause beyond the three named sites** (§8, "isotropy ⇒ equivariance of the map; the latch persists without it"). Rationale: without it the note asserts the neutral mode is *symmetry-protected* two pages after proving it survives symmetry breaking — an internal contradiction a referee would find. It is a bare corollary of the new proposition; I deliberately did **not** brand it "modulus vs Goldstone" or use it against Mo (that is S5, a Head-gated V2 decision).
4. **Not proven / not attempted:** whether the blindness theorem survives the *relativistic* `T` in the non-linearized regime. `∇_pT ∥ M⁻¹p` holds in all three kinetic modes, so the linearized statement (and hence Prop-17, which is a statement about `∇²V` and `M_eff` at a critical point) goes through verbatim with `M_eff = m₀M`; but the coset metric acquires momentum dependence away from `p=0` (F5 Prop-2), so the *current* statement `F² = Q/ϑ̇` picks up a boost correction. This is deep-dive **O5** and is untouched here.
5. **Concurrency:** `git status` shows uncommitted modifications by another agent in `chlu/config.py`, `chlu/core/chlu_unit.py`, `chlu/core/potentials.py`, `chlu/experiments/exp_d_goldstone.py`, `tests/test_goldstone.py` (almost certainly `f1-gmor-condensate`). **I touched none of them.** If that agent's `LinearSpurionPotential` work lands, F-1's `μ²F² = δΣ` becomes measurable and F5's new `F` naming is exactly what it will report against.

## Git footprint

**None.** No tracked file created, modified, or staged; `HEAD` unchanged at `27f232f`; no branch created (all four edited files are gitignored under `.claude/**`, verified with `git check-ignore -v`). Scratch: `.claude/scratch/f5-corrigendum/{verify.py, verify2.py, verify3.py}`.

---

## Proposed handover updates (for the Hub)

**§7 — resolve two items, add one nuance:**
- **7.15 [PROVEN, BLOCKING] → ✅ RESOLVED (2026-07-09, `f5-corrigendum`).** All three sites corrected; `f5-note.tex` now carries **Prop `prop:kinblind`** (kinetic-spurion blindness: congruence ⇒ Sylvester ⇒ `μ²≡0` for any invertible `M`; latch/vacuum/channel-count exactly intact; latched direction `= ker K`, unrotated) + the bounded-charge corollary. Abstract clause re-scoped. PDF rebuilt (11 pp, clean). **F5's arXiv push is unblocked on this axis.**
- **7.16 [PROVEN] → ✅ RESOLVED.** Decay constant named `F := √(M_eff)·r*` throughout F5; `r*` = vacuum radius / condensate `Σ`. Naming rule box added to F5 §8. Downstream: "`f` buys robustness" → "`F² = M_eff r*²` buys robustness".
- **7.17 [wording] — strengthen and extend.** Beyond "drift → bounded oscillation": F5 check-(g)'s **`2.6` is retracted as a number** (running-supremum excursion; `2.62 → 2.96` from 10⁵→10⁶ steps; **2.625 vs 2.734** across two algebraically identical `∇V` spellings — the off-ring orbit is chaotic). The quotable statements are (i) `|Q| ≤ sup|q||p|` (compactness — *no secular drift, ever*), and (ii) the **on-orbit** closed form `Q = √(2E)F(ϑ)`, amplitude `√(2E)r*(√M_max−√M_min)`, half-revolution period (verified: amplitude 1.0007–1.0064, elliptic period 0.9994–1.0005). **The clean ring envelope requires a near-rigid-ring init; do not attach it to check (g)'s off-ring numbers.** Also flag `negative_results.md` N4 ("drift = 2.6", "silently break the channel") and `v2-short:141/155` ("price of an equivariant channel" → "…write current") into `v2-revision-4`.

**§1 (the physics) — add one line:**
- **Kinetic-spurion blindness.** A flat direction of `V_θ` keeps `μ² = 0` under **any** inertial anisotropy (Sylvester); kinetic symmetry breaking is invisible to the Hessian and to the latch and shows up **only in the Noether current**, as a bounded, non-secular oscillation. Symmetry buys the *write current*; `V_θ`'s flatness buys the *register*. (Registered as F5 Prop-17 / `prop:kinblind`; this is the theory statement behind measured Finding N4.)

**§8 — one closure, one carry:**
- Deep-dive **R3/S6 closed** (the error is fixed; the replacement is stronger). Deep-dive **F-2 answered** here (oscillation-not-drift, plus the reproducibility finding) — `v2-revision-4` item 5 can cite this report rather than re-running the traces, though re-reading V2's raw `Q(t)` at `γ=0` is still worth ~1 h to confirm the *measured* `5.4/1.6/0.082e-2` are amplitudes.
- Carry **O5** (relativistic running decay constant): Prop-17 survives verbatim in relativistic mode (`M_eff = m₀M`); the *current* relation `F² = Q/ϑ̇` does not — it picks up a boost correction. That is now the sharpest open theory question the corrigendum leaves standing.
