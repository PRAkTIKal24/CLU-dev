# doctrine-repairs — the nine owed reconciliations, the monitor specs, and the P1 adjudication

**Campaign 2, wave C2W2. Agent:** physics-theorist. **No worktree, no production-code edits** (roster
constraint — the *code* landings are assigned to `phi-particle-head`; you write the specs it implements
and the ones C2W3 implements). Charter Addendum-1 **§A5/C2W2 task 4**. Cheap: numpy/sympy checks only.

**Read first:** `.claude/AGENT_PROTOCOL.md` (**§5 pre-registration — it binds you: your acceptance is a
prediction**), `.claude/advisor-head-c2-charter.md` **IN FULL incl. ADDENDUM 1 — §A2 (the five binding
interpretive findings; §A2.5 is YOUR pre-registered prediction), §A4.2, §A4.5**,
`.claude/advisor-head-intervention.md` **§5**, and your own C2W1 output
`.claude/outputs/controller-doctrine.md` **in full** plus
`.claude/outputs/trainability-spike-theory.md`, `.claude/outputs/full-clu-harness.md` **§3.4/§3.6**,
`.claude/outputs/memory-gym-v0.md` **§3.3/§3.5/§3.6**.

⛔ **REGISTRY LAG (Head parked the curator pass, 2026-07-30).** `negative_results.md` (N122),
`claims_matrix.md` (v2.5), `research_roadmap.md` (v0.9) and the ledger (⟲ w26) are **two campaigns
behind**: **C1W27's and ALL of C2W1's results are in no registry.** Quote them **only** from
`.claude/outputs/*` and the `[C1W27]`/`[C2W1]` §10 entries. ⚠ **This bites you specifically:** your §2
site list is normally handed to a curator at review, and **there is no curator this wave** — so make the
list self-contained and explicit enough that it survives until C2W3 without you.

---

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — doctrine/theory. No performance claim, no benchmark, no dividend.
- **Laundering control:** n/a, except that **mode #2 IS the laundering control promoted to a runtime
  monitor** and stays the table's most important row.
- **Falsifies:** §4. **Does NOT falsify:** a narrow band; a band that is only measurable; a monitor that
  costs a diagnostic pass; a repair that makes a monitor trip **more** often.

---

## 1. ⭐ THE HEADLINE: the P1 / merge-certificate adjudication, with §A2.5 PRE-REGISTERED

Two waves have surfaced the same unadjudicated contradiction and the doctrine's own falsifier calls a
proven-empty band pair *"the single most valuable output of the wave."*
- Your C2W1 P1 result: **7 of 54** grid configurations satisfy all 13 bands simultaneously while
  remaining non-degenerate (`D > 0`, `ρ_ex` 0.103–0.127, dividend +0.0000…+0.0033) — with your own
  caveat that all 7 sit **at a corner of the searched region** (the grid's lowest `σ_q` = 0.24, the two
  smallest radii) and the grid **does not bracket the lower edge**.
- The harness's structural statement: **permitted basin interaction and the merge certificate
  `2 s_max + κ′σ_q ≤ sep` are mutually exclusive by construction.** S4 could only exist with
  `d_safe_override` deliberately out of band; the gym needed the same override twice (0.58, 0.32).

**⛔ THE PRE-REGISTERED PREDICTION (charter §A2.5, binding — file it in PREREG before you compute):**
> *All 7 feasible P1 configs satisfy the merge certificate — i.e. the 13-band intersection exists only
> in the separable (provably-zero-dividend) regime. If confirmed: the certificate becomes a monitored
> SOFT constraint in dividend-hunting mode.*

Adjudicate it. Required outputs:
1. **Scored, per config**, whether the merge certificate holds — and, if it holds in all 7, whether that
   is a *theorem* on your grid (does D1's `U → 0` + N2's `sep/σ_q ≥ 5.15` force it?) or a *property of
   the grid* (extend the grid downward in `σ_q` and outward in basin overlap until you either bracket
   the lower edge or prove you cannot).
2. **If confirmed** ⇒ deliver **the soft-certificate spec** (§3 below): the certificate as a *monitored
   soft constraint* with a declared violation budget, a reported margin, and an explicit statement of
   what is given up (Prop D1's guarantee is exactly what is being spent — quantify the price, as your
   Q3.3 did for `d_safe`: `λ_min` falls ~6.5× and the implicit gradient rises ~147× before merger).
3. **If refuted** (a config satisfies all 13 bands *and* violates the merge certificate) ⇒ that is a
   bigger result: the intersection reaches into the non-separable regime and the campaign has a target
   region. Report it with its witness, at the same evidence grade you used for the 7/54.

## 2. The nine owed reconciliations — every one needs a written disposition
Filed at the C2W1 review, assigned to you here. For each: **state the corrected form, name every site
that must change (file:line where you can), and say whether the fix is theory, code, or wording.**

| # | item | what it is |
|---|---|---|
| **R1** | **Prop D1's `sep/2` inradius proxy is broken in production and its error is STRUCTURED** | harness `D/U` 1.5–2.5×; gym reaches **7.44×**. It **holds** (0.81–1.04) where `U` is large (queries *between* wells) and **fails 4.4–7.4×** where `U → 0` (queries near centres). Your own §2b guard claimed "up to 4.8 %". Deliver either a corrected cheap proxy with a stated validity domain, or a rule that forbids the proxy outside it. **`sep/2` may not be quoted as a certified inradius anywhere until this lands.** |
| **R-1** | **`k* = 269` governs an ENDPOINT loss, not a trajectory loss — wrong by 3 orders** | for `L(ψ(traj))` the θ-gradient error at k=270 is **0.680 and flat in k** (0.695/0.690/0.685/0.679/0.680 at k=0/50/100/180/270). Give the correct truncation statement for a **whole-window** read-out, or state that none exists and why. Every quoting site must gain *"of `∂q_N/∂θ`"*. |
| **R-2** | **your §7 truncation recipe makes `φ` untrainable — by an exact 0** | tail truncation enters through a `stop_gradient`, so `‖∂L/∂φ‖` is **exactly 0.0** at k = 0…270 and nonzero only at full backprop. **Truncation direction is load-bearing and the theory does not name it.** Name it, and give the recipe that keeps φ trainable. |
| **R-3** | **monitor #10's "dead axis" wording is wrong** | `settled_point_psi` moves **exactly 0.000** noise units at every stride × 3 seeds because that ψ never reads the buffer. Correct diagnosis: **"no shipped ψ consumes the buffer."** Different bug, different fix. Wording fix + who owns each site. |
| **R2** (gym) | **monitor #6 has no dead-band** | 29 of 58 first-ever trips at slopes of **−5.2e-17**. The predicate `slope_loss < −eps` is being landed by `phi-particle-head`; you own the **`eps` derivation** (what scale, and why that scale). |
| **R3** (gym) | ⛔ **#3's validity leg does not predict drift on a learned `V_θ`** | `corr(gate margin, post-write drift)` = **−0.99 … +0.56, sign-unstable** ⇒ the leg fires ~half the time for the wrong reason. Your own row says *"a certificate that does not predict drift is not a certificate."* Either give a predicate that **does** predict drift on a learned store, or retire the leg and say what replaces it. **This one is yours alone and it blocks the code fix.** |
| **R2** (doctrine) | **your γ band's constants do not transfer** | band `γ ∈ [0.05, 0.5]` at N=400 vs the shipped read at `ρ_conv = 4.3e-7`, **within 2.3× of the edge**, and the annealed read **trips** it (3.6e-6). Scope every γ-band statement by harness, and give the harness-invariant form if one exists. |
| **R5** (gym) | **the dedicated #11 reach probe was a no-op** | the over-excursion item was refused by the **merge** gate, not by reach. Specify a reach probe that **also clears `d_safe`** — i.e. the construction, so the next engineer can build it. |
| **I-14 / #9** | **the "uncleanable by any verb" scope** | #9 trips everywhere except R1/R3 and is pre-declared uncleanable; C1W27's option-(d) gated stiffness measured it **payload-independent at every amplitude (N119)** but ships OFF and **C2 must not build it**. State precisely what C2 may claim about lifetimes until it is on. |

## 3. The monitor-repair specs (what `phi-particle-head` lands now, what C2W3 lands)
Produce a **diff against your own C2W1 13-row table**, row by row: *confirmed / sharpened / replaced*,
with provenance. Rows known to move: **#1** (I-3, landing now) · **#9** (I-4, landing now) · **#6**
(dead-band, landing now — you supply `eps`) · **#2** (I-6 `U < 0.01` inapplicability) · **#10** (tier (a)
access counter, landing now; and the R-3 re-wording) · **#7** (I-7 whole-trajectory + `kinetic_mode`) ·
**#3** (validity leg — **blocked on you**) · **#8** (the `sep/2` scoping — **blocked on you**).
Plus the **soft-certificate spec** (§1.2) for C2W3's factored store, where charter §A4.5 makes basin
interaction and shared wells the *design*: *"semantic placement with basin interaction permitted (soft
certificate)."*

## 4. Falsifiers (register them in PREREG before you compute)
- ⛔ **§A2.5 refuted** — a config satisfies all 13 bands and violates the merge certificate. Report as a
  headline; it re-prices the campaign upward, not downward.
- ⛔ **A mode with no runtime-computable invariant** after the repairs (your C2W1 falsifier (a)).
- ⛔ **Two productive bands provably disjoint** *after* the soft-certificate relaxation ⇒ staged
  activation is impossible as specified ⇒ **Head ruling required**, escalate immediately.
- **Does NOT falsify:** a repair that widens a band; a proxy retired without a replacement (say so);
  a reconciliation whose only honest disposition is *"the statement must be scoped, not fixed."*

## 5. Scope discipline
- ⚠ **You are theory. You edit no tracked code.** Scripts live in `.claude/scratch/doctrine-repairs/`,
  results JSONs beside them, and every number must be reproducible from that directory.
- **Single-seed theory checks are fine and must be labelled** — *"every number here is a THEORY check,
  none is a paper number"* (your own C2W1 convention; keep it).
- ⛔ **"Certificate/certified" keeps the scope you gave it**: the geometric margin condition only, never
  the machine-unlearning sense. And note the standing **unruled** item: the program-wide "certified"
  ruling is **NOT RULED** (C1 left it open) — do not treat it as settled either way.
- ⛔ Never quote: `sep/2` as a certified inradius · `k*` without its object · monitor #2's `D` as a
  progress signal · the `2α|c|` product claim as exact (C1W27: **approximate** — a 2.27× α cut needs
  only a 1.64× `|c|` cut) · `D_fit` values uncorrected for α-contamination (≈0.37α inflation: w26's
  0.910/0.459 are ≈0.89/0.44 as well depths) · "11.3 orders"/`2.2e-12` as the contraction (your own R-1:
  it is an FD floor; the law is `(1−γ)^{N/2}`).

## 6. Compute
**Priority P2** — no worktree, no GPU, no engineer slot. Launch immediately, in parallel with everything.
Estimated ~3–5 h. Your §3 specs are **blocking inputs to C2W3**, and R3(gym) is blocking a code fix in
**this** wave — deliver R3 and the #6 `eps` **early** and say so in your report's first 10 lines.

## 7. Output
`.claude/outputs/doctrine-repairs.md`, protocol §5 format: PREREG scorecard (§A2.5 scored explicitly),
the reconciliation dispositions as a table, the 13-row diff, the soft-certificate spec, and the
**site-by-site list of what must change and who owns it** — in the **first 10 lines**, per the protocol
§5 corollary, because that list becomes the curator's and the next engineer's task. Report to the Hub.
