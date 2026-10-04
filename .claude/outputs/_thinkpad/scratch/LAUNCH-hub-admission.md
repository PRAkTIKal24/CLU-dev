# ADDENDUM — `hub/admission` ⭐ NEW TARGET. The refusal target, re-posed.
**Read `LAUNCH-COMMON.md` first** (operating structure, gate ruling, budget, R1–R5). Your task is
**DIRECTIVE rev. 3 §7.5**, reproduced here in full. ⛔ There is no `tasks/hub-admission.md` in the bundle —
this addendum is your task file.

- **Slug:** `hub-admission` · **Branch:** `hub/admission` · **already created at `945a3e9` (`base/iter2`)**
- **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-admission` · **already created**
- **Threads: 4.** Sub-agents ≤ 2, inside those 4, **every one on `model: "opus"`**. You run on Opus.
- **12 h true wall clock.** New `START_UTC.txt`. Seed 0 only.

⚠ **You are the fourth concurrent hub.** `generation-table-2`, `composition-2` and `deletion-2` are
mid-run. ⛔ **Do not read their outputs, do not contact them, do not branch from them.**
✅ `hub-refusal`, `hub-refusal-2` and the iteration-1 hubs are **inherited fact** — read freely.

---

# §0. ⛔⛔ THE GATE — registered verbatim from DIRECTIVE rev. 3 §1b property 1. Do not invent another.

> **Derived, not fitted.** The gate per well is the **zero level set of the kernel free energy**:
> `refuse well k iff E_k(x_k) > 0`, where
> `E_k(x) = −σ_k² · log Σ_i exp(−‖x − a_ki‖² / 2σ_k²)`.
> Zero is not a choice: **every stored atom has `E_k(a_ki) ≤ 0`** because its own term contributes 1 to
> the sum, so the sub-level set at 0 is exactly *"within reach of a stored atom"* in the well's own units.
> The **certified sufficient condition** is the bound `d(x_k, nearest atom) ≥ σ_k · √(2 log N)`, beyond
> which `E_k > 0` whatever the atom configuration; it is reported as **the certified fraction of
> refusals**, ⛔ never as a second criterion. Inputs: `σ_k` from `atom_bandwidths` at the shipped rule,
> and `N`. ⛔ No quantile on `val`, no invalid example, no free parameter. Zero-supervision *including
> zero threshold fitting*.

⭐ **Required in your PREREG-1, verbatim:** *"every stored atom has `E_k ≤ 0`; the PREREG verifies this
identity numerically on the live store before any gate number."* Compute `max_i E_k(a_ki)` per well on the
live store and log it. ⚠ **A violation is a bug in your code, not a result** — stop and fix, do not report
it as a finding.

⛔ **`E_k` is the shipped `gen_landscape.store_energy` per well — CALLED, never re-implemented.**
`σ_k` comes from `atom_bandwidths` at the shipped rule (`nn_median`). The gate is evaluated **at the
launch point**; settle dynamics are optional and **off by default** (Head: nice-to-have).

⛔⛔ **STRUCK, and you must not use it:** rev. 2 derived the gate from *"the pinned coercivity certificate
`λ ≈ 0.0485`"*. The coordinator traced it: that `λ` is **`coupling_strength_used`**, a dimensionless
**inter-well coupling budget** (`gen_landscape.py:616`: `C_kk = s_k²/2`, `C_kl = −λ‖W_kl‖₂/2`, coercive
iff `C ⪰ 0`, `λ* = 1/ρ(D^{−1/2} S D^{−1/2})`) — **not a distance**. And the kernel energy is **coercive
everywhere** (`E_k ≥ min_i‖x_k − a_ki‖²/2 − σ_k² log N`), so **there is no coercive edge to be outside
of**. The *Certified Access* screen does not map onto this landscape. ⛔ **No `λ`, no coercive component,
no radius other than the bound above.**

## §0b. ⭐ MEASURED BY THE COORDINATOR BEFORE LAUNCH — two facts that will save you an hour

**(i) The identity holds, and comfortably.** Computed with the shipped primitives on the rung-A store
(5 wells × 64 atoms), `max_i E_k(a_ki)` per well:

| well | N | `σ_k` | `max_i E_k(a_ki)` | certified bound `σ_k√(2 ln N)` |
|---|---|---|---|---|
| 0 | 64 | 4.6343 | **−30.39** | 13.366 |
| 1 | 64 | 5.0714 | **−40.86** | 14.626 |
| 2 | 64 | 9.1827 | **−225.58** | 26.484 |
| 3 | 64 | 6.3054 | **−89.68** | 18.185 |
| 4 | 64 | 5.1006 | **−30.39**† | 14.710 |

⭐ `identity_E_le_0_holds = true`, worst case **−30.39**, nowhere near zero. Artefact:
`_hub_scratch/_gate_identity.out.json`. †worst-over-all-wells is −30.39.
⚠ These are the **refusal hub's rung-A store (K = 5)**, not your K = 32 reference object — the identity is
algebraic and store-independent, but ⛔ **recompute the table on YOUR store** as PREREG-1 requires.

**(ii) ⛔⛔ `gen_landscape.store_energy` SUMS OVER WELLS — it is not your `E_k`.**
`store_energy(land, X)` accumulates `e = e − σ_k² · logsumexp(lk_k)` across every well and returns one
scalar per row. **Your gate is per well.** Use the shipped primitive and apply the reduction yourself:
```python
from chlu.core.gen_landscape import _well_logk
lk = _well_logk(land.store, X, sigma)      # list of (n, N), each = −‖x−a‖²/2σ²
E_k = -(sigma[k] ** 2) * jax.nn.logsumexp(lk[k], axis=-1)   # per well
```
⭐ `_well_logk` returns exactly `−d²/(2σ²)` with **no normalising constant**, so the directive's formula
and the shipped code agree term for term. ⛔ Still not a re-implementation — you are calling the shipped
kernel and doing the per-well reduction the API does not expose.

## §0c. ⚠⚠ THE THING MOST LIKELY TO END THIS RUN IN HOUR ONE — check it in minute five
The zero level set sits at roughly `d ≈ σ_k√(2 ln N)` from the atom cloud — **2.88 σ at N = 64**, 3.53 σ
at N = 512. But `hub/refusal-2` measured that the **linear read sits ~21 atom-spreads from the nearest
atom, for valid and invalid alike.**

⛔ **If "atom-spread" there is commensurate with `σ_k`, then valid launches sit far outside the level set
and the gate refuses everything — §5.7 fires before B is ever touched.** ⚠ The two may not be the same
normaliser (§C.4 uses distance over *atom spread*, which may be the cloud's std rather than the kernel
bandwidth) — **that is exactly what you must establish, and it is cheap.**

⭐ **Do this before anything else in step 2's companion check:** take `val` corners, launch symbol-mode,
and report per well: `d(x_launch, nearest atom)`, `σ_k`, `d/σ_k`, `E_k(x_launch)`, and the **accept rate
at the gate**. One number decides the shape of your whole run:
- accept rate on valid `val` is **high** ⇒ the gate is live; proceed to step 2 proper.
- accept rate is **near zero** ⇒ **§5.7 has fired.** Report the derived number and a fitted number side
  by side, ⛔ **do not fit a threshold**, and say so in hour one rather than hour eleven.
⭐ Either way this is a **complete and valuable answer** — and finding it in minute five is worth more
than finding it at the scored run.

---

# §1. ⚠ ONE CORRECTION THE COORDINATOR MADE, AND WHY — check it before you file PREREG-1
DIRECTIVE §7.5 step 3 mandates a disclosure "verbatim" whose text still reads *"derived from the store's
geometry **and the pinned coercivity certificate** with no free parameter"* — but rev. 3 **struck** the
coercivity certificate from the derivation (§0 above). Registering it verbatim would put a **false
statement about your own derivation** into your PREREG.

⭐ **File this corrected wording, and log the substitution as a dated bullet quoting both versions:**

> *"This target was posed after two refusal hubs analysed invalid examples on a fit-derived synthetic
> analogue, and after two benchmark invalid records were displayed to a predecessor. The admission gate is
> the zero level set of the kernel free energy, derived from the store's own geometry — the per-well
> bandwidth `σ_k` at the shipped rule and the atom count `N` — with no free parameter, so the
> zero-supervision claim is 'derived, not blind', and the paper says so."*

✅ **COUNTERSIGNED by the Head, 2026-09-17.** This is the wording to file. ⛔ Do not adjust it further;
if you think it is still wrong, log it and carry on with this wording.

---

# §2. THE SOURCE TEXT — extracted by the coordinator from the paper. Quote from here.
⚠ The directive says PDF rendering is unavailable. `pdftoppm` is indeed absent, but **`pdftotext` works**
(`/mingw64/bin/pdftotext`); the coordinator extracted all four papers. These are the load-bearing passages
from `F_v1_TTCL_finalSub_2.pdf` — *Certified Access: Test-Time Compute on Conservative Physics-Structured
AI Memory*.

**For control (e), the margin comparison — the paper's own applications section:**
> *"The residual is a valid gate feature but adds at most **+0.024 AUROC** over a standard readout margin
> and is therefore **not the right exit threshold**."*

**And its negative-findings section, which is stronger:**
> *"we confirm that the raw energy signal adds **negligible predictive value** over a standard readout
> margin, establishing that our framework relies on the calibrated rationing mechanism itself rather than
> an inherently superior physics signal."*

⭐⭐ **Read that as scope, not discouragement.** It is exactly why your claim is **properties 1–4**
(derived · localised · live-set-exact · one object two outputs) and ⛔ **not** *"the CLU detects invalid
queries better than kNN"*. If the factored kNN gate ties you on the pair, **that is an anticipated
outcome and the report says so** — it does not make the run a failure.

**The coercive screen, for context only (⛔ not your gate — see §0):**
> *"membership in the coercive component is the mandatory operational clause for BIBO stability"*; on the
> testbed `V = ½kq² − λq⁴` (`k = 1.0`, `λ = 0.02`) the component edge is `x_b = 3.5355`, exits
> `b ∈ {1,2,3,3.6,4,5}` give `ADMIT ADMIT ADMIT REJECT REJECT REJECT`, and the screened channel
> *"holds at r = 0.09 because it **refuses the jump rather than making it safe**"*.
⭐ That last clause is the spirit of admission. ⛔ The mechanism does not transfer; the posture does.

---

# §3. WHAT YOU BUILD, IN ORDER — each step a dated bullet before the next starts
1. **Ground truth about the wells (≤ 1 h).** Confirm `feat_shape` and what the 2048 latent dimensions are
   (feature maps × spatial, or pooled); confirm `well_channels` for K = 32. Measure the **factor-to-well
   map**: per well, per factor, MI over shuffled-label floors (the refusal hub's method, inherited).
   → `FACTOR_WELL_MAP_seed0.json`. ⭐ Without it, exemplar mode and term localisation are undefined and
   you proceed in **symbol and free modes only** — say so if that happens.
2. **Does the affine head cross the level set? (≤ 1 h; falsifier §5.10).** Symbol-mode `out_of_range` at
   severities `w ∈ {2, 3, 4, 7, 13}` via the R8 rule on the one-hot design: measure `E_k(x_launch)` and
   `d/σ_k` per well, plus the per-term residual. **They must rise with severity.** ⛔ If flat, report and
   **stop the symbolic out-of-range leg** before building anything else.
   ⭐ **Companion check, same hour:** where do **valid** `val` launches sit? If `E_k(x_launch) > 0` for most
   valid requests, **the gate refuses everything and §5.7 has fired before B is touched.** Report it then.
3. **PREREG-1 — before any accept/refuse number.** The gate from §0 verbatim; the certified bound reported
   as a fraction; the `max_i E_k(a_ki) ≤ 0` self-check; the term-localisation rule (the requested term with
   the largest residual among uncertified wells); the query-set definitions; the §1 disclosure.
4. **Query sets — all from seed 0's cached features. ⛔ NO RE-ENCODE, ever.**
   - **seen:** `val` tuples, symbol mode.
   - **compositional (B):** held-out valid compositions, symbol mode — ⛔ **held out; touched only in the
     scored run.**
   - **unseen value:** one continuous factor (scale, orientation) launched at an interpolated point between
     adjacent grid values, on `val` corners.
   - **out-of-range:** R8 severities as in step 2, on `val` corners, one factor at a time.
   - **out-of-vocabulary, symbol:** a request naming a value the store never wrote.
   - **out-of-vocabulary, exemplar** (the "black cube" set, **on real images**): rebuild the store from
     `store_train ∖ rows(v)` for one value `v` of each of the six factors (six stores), **from cached
     features**. Queries: real images of `v` through the frozen encoder, contributing only the channels the
     factor-to-well map assigns to that factor; other factors by symbol. Plus **write-back:** full store,
     same images — must be **accepted**. This is live-set control (d).
     ⛔ A shortfall > 0 on any rebuild ⇒ **stop and report the rows** (falsifier §5.6).
   - **A7:** the redundant-term valid control, printed beside B on every row.
   - **duplicate_factor:** interface grammar check only, reported as such. ⛔ No landscape number.
5. **DEV numbers** (val corners, non-held-out) with all six controls. Then, **only if the bar is met on
   DEV**, file **PREREG-2** and spend the one scored run.
6. **The certified-access figure (DEV, unscored).** Walk a launch point outward from a `val` atom along a
   ray; plot external-classifier accuracy of the decoded output vs `d/σ_k`, for the CLU read and for a
   plain decoder of the same latent, with the level set and the kNN quantile drawn. Report **where the
   certificate flips relative to where decode quality collapses.** One figure; ⛔ no claim beyond it.

# §4. MANDATORY CONTROLS — on every row
(a) whole-latent kNN gate, `q = 0.95` · (b) ⭐ the **fair** rival: per-well-factored kNN gate, same `q` ·
(c) single-well (`K = 1`) CLU on the same atoms, which must lose localisation · (d) live-set
hold-out/write-back · (e) physics-free head margin `M` · (f) ⭐ **`accept_A7` beside `accept_B`, every row,
arm and rival, same threshold** (programme-wide rule as of today).
⛔ **NOT the §C.2 atom shuffle** — it tests decoding and cannot certify an interface-facing claim (§1a).

# §5. THE BAR — set now, PREREG carries it, ⛔ never moved
- **Pair:** win-or-tie under R1 vs the **factored** kNN gate on `(refuse_out_of_support, accept_B)`, where
  out-of-support pools out-of-range and out-of-vocabulary; and `accept_A7 ≥ accept_B − 0.05`.
- **Localisation:** term-localisation accuracy on single-factor perturbations **≥ 0.50** (chance 1/6), and
  beats the single-well control beyond seed spread.
- **Live-set:** refuse rate on held-out-value exemplar queries **≥ 0.90** and accept rate on the same
  images after write-back **≥ 0.90**. (Byte-exact provenance is phase 3, once deletion-2's canonical
  rebuild is inherited fact — ⛔ not this iteration.)
- **No tuning:** the gate reported is `E_k > 0`. A fitted alternative may be **reported beside it,
  labelled, never adopted.**

# §6. FALSIFIERS — stop and report on any of these
**§5.6** shortfall > 0 on a held-out-value rebuild · **§5.7** the gate does not calibrate (`accept_B < 0.50`
while the whole-latent kNN gate at `q = 0.95` gives `accept_B ≥ 0.90` on the same rows) — report the derived
number and a fitted number **side by side** and ⛔ **do not fit** · **§5.8** no design advantage: factored
kNN ties the pair **and** localisation does not beat the single-well control ⇒ admission is kNN with extra
steps; say so · **§5.9** the encoder does not separate: exemplar launches from held-out-value images fall
inside the level set within 0.05 of valid images' rate ⇒ exemplar refusal dead, symbol and free continue ·
**§5.10** the affine head's out-of-range does not cross the level set.

# §7. ⛔ ALSO FORBIDDEN
Moving the gate off zero, scaling `σ_k`, or replacing the level set with a quantile · exemplar mode on any
generation number or any read where `assert_no_bypass` must pass (exemplar is its own labelled arm) ·
**synthesising** out-of-vocabulary images (hold out a **real** value instead) · re-encoding any row ·
editing any manifest-pinned file or any pinned bandwidth · deleting anything · quoting analogue numbers as
benchmark numbers · quoting `gen-arms-a7-wiring/refusal_scores.json` (superseded — check the **directory**,
it shares a basename with the good file) · writing *"the CLU refuses what transformers decode as garbage"*
without the kNN-gated baseline in the same figure.

# §8. SCORED BUDGET & REPORT
**≤ 1 scored run, seed 0**, reported as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*. The
scored run is the **B-accept / held-out-value-refuse pair** with all six controls. Everything else is
DEV/val and costs nothing. Seeds 1–2 are unreachable here (caches are `scope="used"`; covering them is a
re-encode). **Report:** the pair with controls · localisation table · live-set table · the figure · the
factor-to-well map · §5.6–§5.10 answered **with numbers** · cost line · elapsed true wall clock ·
solved-truthfully or not-solved.
⭐ ***"The derived gate does not calibrate and here is by how much"* is a complete and valuable answer.**
