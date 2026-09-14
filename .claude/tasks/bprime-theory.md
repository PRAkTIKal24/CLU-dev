# bprime-theory — the audit paper's theorem set (+ the `allocate` spec review)

**Campaign 2, wave C2W3. Agent:** physics-theorist. **NO WORKTREE, NO PRODUCTION CODE** (the roster bars
you from it; you flag what an engineer must change and the Hub routes it). Launch **immediately** —
you gate on nothing.
Charter **ADDENDUM 2 §A11 task 5**.

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **in full, especially
ADDENDUM 2 §A7 (the gate verdict), §A8 (the four interpretive findings), §A9 (the twelve rulings — §A9.6
is your T5) and §A11**; `.claude/advisor-head-intervention.md` **§5, §6, §8**; the **live `2026-07-31`
`[C2W2]` §10 entry** in `.claude/handover_context.md` (**reconciliation #9 is yours, jointly with the
curator**); `.claude/outputs/track2-admissibility/PREREG-Bprime.md` **§7 (what B′ reuses — your theorems
are what license that reuse)**; `.claude/outputs/doctrine-repairs.md` (**your own C2W2 report — R1, the
two new blockers, and SC-1…SC-7**); `.claude/outputs/trainability-spike.md` **and**
`.claude/outputs/trainability-spike-theory.md` (`∂q*/∂q₀ = 0`); `.claude/outputs/phi-particle-head.md`
**§2 (the mass-gauge dissolution — the other half of your T3)**; `.claude/outputs/memory-gym-v0.md`
(the byte-floor law's 28 cells); `.claude/outputs/full-clu-harness.md` (Prop D2a's first confirmation).

⛔ **REGISTRY LAG — THREE WAVES (C1W27 · C2W1 · C2W2).** Quote results **only** from
`.claude/outputs/*` and the §10 review entries. `doc-curator-c1w27-c2w1-sync` (the unparked 3-wave pass)
runs in parallel with you; **coordinate through your reports, not by editing each other's files** — the
curator files registry entries, you write the theorems the entries cite. ⚠ **Reconciliations 9 and 11
are jointly owned with them and they are waiting on your corrected wording** (their Pass-C items 19 and
21) — put that wording in your report early rather than at the end.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none — theory.** No dial, no leaderboard, no dividend. You are writing the
  audit paper's **theorem set**: the statements that make B′'s banked evidence *reusable* rather than
  re-measurable, and the caveats that keep its protocol honest.
- **Laundering control:** n/a for a derivation — **but every proposition you state carries its
  DOMAIN OF VALIDITY, and a proposition without one is not filed.** C2W2 taught this twice at cost:
  `sep/2` was quoted as a certified inradius outside its validity domain, and `λ_min > 0` was treated
  as certifying a nonempty basin when a genuine minimum (`λ_min = +0.910`) has a measured capture
  radius of **0.000**.
- **Falsifies:** §4. **Does NOT falsify:** a theorem coming out **narrower** than the campaign hoped
  (that is the point of proving it) · a caveat that costs us a claim we liked.

---

## 0. Why this task exists
B′ is an **audit paper**, and an audit paper's load-bearing content is not its runs — it is the set of
statements that say *why the protocol is valid, what it can and cannot conclude, and which of the
program's 29 waves of evidence can be cited rather than re-run*. `PREREG-Bprime.md` §7 lists banked
evidence B′ **must not re-measure**; your theorems are what make that list legitimate instead of lazy.
Two of the three theorems are already measured to high precision — your job is to state them properly,
with assumptions and domains, and to say exactly what would break them.

## 1. Deliverables

### T1 — ⭐ The byte-floor theorem
> **`ratio = 1.4 · atoms_per_item + 0.8`**, verified to **1e-9** in **all 28** C2W1 cells; architectural
> floor **2.20×**; measured minimum **2.28×**; minimum ratio anywhere in C2W2 **17.11×**.

State it as a theorem of the **per-item atom-group** store: assumptions, derivation, and the exact
structural reason it holds. Then answer the question the campaign actually needs answered:

- ⭐ **What would a shared / factored substrate have to do to break the floor?** Charter §A2.3 is
  binding here: *"per-item atom groups structurally exclude compression (dividend candidate (a));
  private per-item parameters cannot superpose; matched bytes is unreachable (≥2.20×, 0 violations /
  28 cells) **because of** the C3 masking, not despite it."* Give the sharpest statement you can of
  what sharing must buy, and at what rate, for the ratio to reach 1.
- ⚠ **Couple it to §A9.9 (standing).** Any future shared-substrate work measures deletion as a **curve**
  — exactness preserved on the private-atom fraction, measured degradation on the shared fraction —
  because **byte-exact deletion is never spent silently.** Byte-exact deletion (**AUC 0.5000 ± 0.0000**,
  byte-equal **3072/3072**) is a *consequence* of per-item atom groups: **the same property that
  excludes compression.** ⭐ **These are the same trade.** Formalise it — the deletion-vs-sharing
  frontier — so a future wave cannot spend one to buy the other without seeing the exchange rate.
  ⛔ The shell atom *raises* the architectural floor by `1/(dim+2)` (**52.00 → 58.40×** at `dim = 6`,
  **+12.5 %** on the atom term) — a basis change is not a route to matched bytes.

### T2 — ⭐ Prop D2a, stated properly
> **A settle is arg-min** over the stored centres, given: separable wells (enforced spacing), an
> endpoint-trained landscape, and a settled-point-only read. Three independent confirmations; the
> shipped anchor's dividend is **exactly 0.0000 with D = 0**.

State the hypotheses **individually** and say which one, dropped, breaks the conclusion — that is the
proposition's whole value to the paper, because it converts a negative result into a **map of where a
dividend could structurally live**. Connect it to §A2.1 (the point-estimator diagnosis): *the write
loss constrains only isolated settled endpoints; placement enforces separability; the read returns a
point; the in-between regions were never asked to carry information, so every read that touches them
loses.* ⚠ **Monitor #2's `D` is the dividend's VARIANCE, not its magnitude** (the `D = 0.931` cell has
dividend **−0.875**) — it bounds where a dividend could live and is **never** a progress signal. Say so
in the proposition's own text so it cannot be misread downstream.

### T3 — ⭐⭐ THE ONE STATEMENT (§A11's own framing: *"as one statement"*)
> **`∂q*/∂q₀ = 0` + the mass-gauge dissolution ⇒ A SETTLED-POINT READ IS UNTRAINABLE END-TO-END, IN
> BOTH DIRECTIONS.**

This is the audit paper's structural theorem and the strongest thing 29 waves produced. The two halves:
- **Read-in direction (C2W1):** a settled-point read sends **zero** gradient to its read-in.
  `‖∂L/∂φ‖` = **0.0** (implicit) / **2.654e-9** (unroll) / **6.421e-3** (trajectory) — ratio **2.42e6**.
  The implicit zero is exact, not small.
- **Particle direction (C2W2, §A2.2 SUPPORTED):** the implicit-settle mass gradient is **exactly 0.0
  BITWISE, 3/3 seeds**; a trajectory read sends **1.74e-3 / 1.17e-2 / 1.88e-3** to a per-query mass and
  **2.4e-2 / 4.1e-2 / 3.3e-2** to a per-query friction ⇒ ratios **1.7e5–2.9e5** (mass), **2.6e5–4.9e5**
  (friction), cross-checked against **float64** finite differences. **Friction is the ~14× stronger
  channel.** The point arm's zero is **structural — Prop Q1.1: `∇V` contains neither `M` nor `γ`** —
  not lucky.

Prove them as one statement, with the shared mechanism named. ⭐ **Then discharge the cheap owed
confirmation (§A2.2, still outstanding): does this retro-explain w19's learned-addressing death
(0/18, 4.2 % — gradient search for an address returning ≈chance) and φ's whole history?** It has been
owed since Addendum 1 and it is one paragraph plus a check.
⚠ **Prop F1's mass-gauge is dissolved only under trajectory reads** — the endpoint is `M`-independent,
the trajectory is not. State the scope precisely; "mass as selector" is live, and it is live *there*.

### T4 — Protocol caveats, as numbered propositions with domains
These are the audit paper's honesty section and they are load-bearing. Each gets a statement, a domain,
and the measurement that forced it:
1. ⛔ **Below `s/sep ≈ 0.15` the basin boundary is INERTIAL, not a static watershed — no static proxy is
   valid there.**
2. ⛔ **`λ_min > 0` does NOT certify a nonempty basin** — a genuine minimum at `λ_min = +0.910` measured
   a capture radius of **0.000**.
3. ⛔ **`sep/2` is NOT a certified inradius.** `D ≤ U` is a theorem **under a certified ball**, and the
   gym computed `U` from `sep/2` on stores whose sites **were not minima** — all 7 cells with `D/U > 4`
   have `λ_min ∈ [−1.199, −0.372]`. **"Prop D1 is violated (1.5–7.44×)" is RETIRED.** The corrected
   inradius is **14.55×** more accurate **inside its stated domain**, and the domain is
   `s/sep ∈ [0.15, 0.30]` — state it every time.
4. ⛔ **`k*` governs `∂q_N/∂θ` ONLY where fixed-point sensitivity dominates the transient** (ratio 64× ⇒
   holds; **27 396× ⇒ error 0.448, flat in `k`**), and in a K-item store the far-well parameters are
   **exactly the interference gradients**. Never quote `k*` without that qualifier.
5. ⚠ **`ε` is not the manifold-payload lifetime dial.** The shipped confinement floors the soft mode at
   **2α**, so **`τ_max = Γ/2α`** — **`α` is the ceiling, and lowering `α` breaks the write.** Every
   manifold-lifetime claim carries the **2α coercivity coupling**. §A4.2's tilt instantiation is
   **REFUTED on a learned store** (monotonically *reduces* `λ_min`, **+0.0994 → −8.28**, two independent
   implementations, every family) because **a designed degeneracy does not survive superposition**
   (written-site vacuum residual **0.140–0.343** vs a random-orientation baseline of **0.167** — at or
   **worse** than random). ⭐ **The pseudo-Goldstone ruling survives as GEOMETRY; its shipped
   instantiation does not.** `λ = ε` holds only in the single-atom geometry it was specified in
   (unit-tested to 5 %). Draw that line precisely — it is a charter amendment and it needs to be exact.
6. ⚠ **Reconciliation #9 (yours, with the curator): doctrine I-7's gauge is `newtonian_learned`-ONLY,
   not "Newtonian".** Under `newtonian_identity` the orbit is **not a gauge orbit at all** (residual
   **0.2505** vs **2.52e-7**). Restate I-7 with its scope and hand the curator the corrected wording.
7. ⚠ Your own two errata from C2W2's S1–S10 site list: the **γ-band row mixes tolerances**, and the
   **`sep/2` 4.8 % figure needs its validity domain.** Close them here.

### T5 — ⭐ REVIEW THE `allocate` SPEC **BEFORE** STAGE 2 BUILDS IT (§A9.6)
`route3-stage2` is scoped and conditional; if §A9.4 unlocks it, it implements `allocate` v0. **Your
review is a prerequisite input to it** and its task file points at this deliverable by name. Answer:
- Is the action space (**simplex/discrete over endpoint dims · `(q, p)` slot pairs · particle
  attributes**) **byte-ledger-conserving by construction**, or only by convention? ⛔ *"Allocation must
  never be a hidden capacity increase"* is a declared collapse mode — can it be made **structurally**
  impossible rather than monitored?
- ⭐ **Is "the launder receives the same allocation budget" a fair comparison, formally?** A table given
  the same freedom to spread its bytes across slots — is that the right null, or is there a stronger
  one? This is the single question that decides whether stage 2's dividend means anything.
- Does `allocate` stay inside §3.2 (**designed action space, learned policy**), or does any part of the
  *mechanism* become learned? §A8.3 rules it conformant; check the instantiation, not the intent —
  w20's lesson (free learning erases design) and C2W2's weaker, worse sibling (**design the objective
  cannot see is neither erased nor used** — learned shell radius moved **0.500 → 0.501** in 300 steps).
- What is the **weakest** inter-slot coupling that a **per-slot table launder provably cannot express**?
  ⭐ §A9.5 says the headline claim must *require* such coupling — **name the candidates.** This is the
  most valuable paragraph you can write for stage 2, because without it there is no claim to make.

## 2. Method
Rigorous notes plus **small numerical sanity checks** (jax / sympy / numpy) to confirm or refute — your
standing practice, and it has caught real errors. Scratch under `.claude/scratch/bprime-theory/` or
`.claude/outputs/bprime-theory/`. ⛔ **No production code.** Where a theorem implies a code change,
write the change as a numbered request for the Hub to route to an engineer — do not edit.

## 3. PREREG
Not applicable in the usual form (you measure nothing new). ⭐ **But where you run a numerical check to
confirm or refute a derived quantity, write the predicted value and its derivation BEFORE you run it**,
in the note itself. The protocol's own justification applies to you directly: *a pre-registered
prediction that survives is evidence; one that fails is a finding; an un-pre-registered agreement is
neither* — and it was a theorist-adjacent prediction (the "≈13.9× memory vault") that propagated as fact
for two waves before `v5-gate` rejected it by a factor **8.11**.

## 4. Falsifiers
- ⛔ **T3 does not hold as one statement** — the two zeros have different mechanisms and cannot be
  unified ⇒ the audit paper loses its structural theorem and gets two weaker ones. Say so.
- ⛔ **The byte-floor theorem is not a theorem** — the 1e-9 agreement across 28 cells is a coincidence of
  the sizing convention rather than a structural bound ⇒ **`PREREG-Bprime.md` §7's reuse licence is
  void** and `bprime-rivals` must re-measure. **Report the same day**; it is the heaviest task's
  foundation.
- ⛔ **`allocate` cannot be made byte-ledger-conserving** ⇒ tell the Hub **before** stage 2 spends a
  worktree on it.
- ⛔ **No inter-slot coupling exists that a per-slot table cannot express** ⇒ **§A9.5's kill-condition is
  unsatisfiable in principle and Route 3 has no headline claim available to it.** This is decision-grade
  and it is cheaper to learn from you than from a build.
- **Does NOT falsify:** a theorem that is narrower than hoped · a caveat that costs a claim · finding
  that the pseudo-Goldstone ruling survives only as geometry (that is already the charter's position).

## 5. ⛔ Never-quote (inherited — you are also the person most likely to be quoted BY others)
**"Prop D1 is violated (1.5–7.44×)"** (retired) · **`sep/2`** as a certified inradius, and the corrected
proxy outside `s/sep ∈ [0.15, 0.30]` · **`λ_min > 0`** as certifying a nonempty basin (measured **0.000**
at `λ_min = +0.910`) · **`k*`** without *"of `∂q_N/∂θ`, and only where the fixed-point sensitivity
dominates the transient"* · the ridge saddle **`λ_min = −0.5946`** as a multi-seed result (**seed 0**;
3-seed mean **+0.177 ± 0.469**) · **`ε` as "the manifold-payload lifetime dial ∝ 1/ε"** without the
**2α** coercivity ceiling · the recency family's **`0.3019 ± 0.0679`** as a null (scoring-domain
**defect**) · monitor #6's **"58 trips"** without *"pre-repair"* (**27**; artefact count **31 of 58**) ·
any **`AttentionPsi`** trajectory number (it leaks) · **Titans as "a preprint"** (NeurIPS 2025) · any
**SDM Table 1 state/param ratio** · **"MAD `compression` is the admissible synthetic"** · **"principled
forgetting"** as a novelty phrase · **"we alone delete"** (the MUNKEY narrowing: MIA-AUROC → 0.5 by
design, but **not exact** — gap to retraining **0.56 ± 0.21**, and a ViT classifier, not a sequence
memory) · any C2W3 cell as a **byte-matched** dividend (min ratio anywhere **17.11×**).

## 6. Output
`.claude/outputs/bprime-theory.md`, protocol §5 format, with:
- **T1–T5 as numbered propositions**, each with **assumptions, domain of validity, and the measurement
  that supports or bounds it**;
- ⭐ **T5's answer to *"name the inter-slot couplings a per-slot table cannot express"* in the first 10
  lines** — `route3-stage2` reads that line before it writes a design;
- a **numbered list of code changes your theorems imply**, for the Hub to route (you do not edit code);
- your reconciliation list in the **first 10 lines** (protocol §5 corollary — #9 and your own two
  errata are already on it, and the curator is waiting for the corrected wording);
- ⛔ **declared NOT-DERIVED items, never presented as settled.**
