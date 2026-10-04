# deletion-waitlist-stiffness — the P2 waitlist and option (d), behind a flag

**Agent:** experiment-engineer. **Worktree MANDATORY** (§3.2). Base local `main @ 082d095` (post-w26).
**Campaign tag: [C1W27].** Two independent w26 follow-ups, ruled by the Head 2026-07-29 (queue item
4 = YES; the "also in scope" list = option (d) behind a flag + the mia re-run **including `R₅₀`**).
Read `.claude/outputs/placement-landing.md` (§ recommendations 1–2),
`.claude/outputs/readout-channel-theory.md` §3.2–3.3, `.claude/outputs/mia-decay-measurement.md`
§3(a)/(b)/§5, and `.claude/outputs/carried-remeasurements.md` (the allocator-leak curve) first.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** two — **isolation/deletion** (Part A) and **lifetimes** (Part B).
- **Laundering control:**
  - Part A: the trivial substitute is a **flat datastore row-delete**, which is exact *by
    construction*. The claim is only that the physical store **matches** it at overflow. ⛔ Never
    write a sentence in which the store beats a dict on a deletion-exactness axis.
  - Part B: **N108 stands** — against an *exact* adversary a physical `exp(−leak·t)` amplitude and a
    **boolean TTL flag are equally detectable** (1.000 vs 1.000 white-box; 0.983 vs 1.000 query).
    The only differentiator that needs no adversary caveat is the **retrieval geometry** (`R₅₀`
    contracts 1.146 → 0.752 vs a TTL's constant step at ≈0.77). **That is what must be
    re-measured under the gate**, and it is the whole reason `R₅₀` is named in this task.
- **Falsifies the claim:**
  - Part A: on the un-inflated 7-cell / 8-offer geometry the waitlist does **not** take
    `canon_native` `AUC(n_live)` 1.000 → 0.500 with byte-equality restored, at 3 seeds.
  - Part B: the `R₅₀` contraction **disappears** under the gate at `g₀ = amp_floor` **and** the
    graded `g₀ ≪ amp_floor` + long-read variant does not restore it ⇒ the lifetimes dial has lost
    its only adversary-caveat-free differentiator, and you report exactly that.
- **Does NOT falsify:** losing to a dict or a TTL flag on any AUC axis (standing
  exactness/indistinguishability result, N108); a step-shaped retention curve at `g₀ = amp_floor`
  (that is the trilemma's stated price, not a defect).

## Part A — the P2 waitlist (~20 lines; queue item 4, Head: YES)
`placement-landing` proved PGCP **below capacity** (`AUC(z_hole)` 0.99985 → **0.5000 ± 0.0000**,
IN-after-delete byte-equal 3072/3072) but ⛔ **fails at overflow** (`AUC(n_live)` = 1.000,
`AUC(s4)` = 0.914, byte-equal fraction 0.000) because a background item refused in the IN world does
not counterfactually return when the target is deleted. The fix, as specced by the analyst: **keep
refused records in a side dict; re-run admission by priority on any delete.**

**Acceptance:** on the same paired-world harness (3 seeds × 8 targets × 128 paired worlds), the
un-inflated 7-cell / 8-offer overflow geometry gives `AUC(n_live)` **1.000 → 0.500**, `AUC(s4)` →
0.5, byte-equal fraction **0.000 → 1.000**, with the below-capacity result unchanged.

⚠ **State it at a load, and quote the curve.** `carried-remeasurements` showed the allocator leak is
a **curve, not a constant** — AUC **0.6715 / 0.9165 / 0.9961 / 0.99985** at **2 / 4 / 6 / 8** offers.
Report the waitlist's effect **across that whole load sweep**, not at one point. Then supply, in your
report, the **exact replacement scope sentence**: `placement-landing`'s *"exact below capacity or
under set-function eviction"* becomes *"exact"* **only if** the sweep is flat at 0.5 — and if it is
flat only up to some load, the sentence names that load. The Hub will relay your sentence to
`r1-positioning-pass`; do not edit any doc yourself.

## Part B — option (d), gated-stiffness payload channel, behind a flag
The theorist **refuted both** mia-D3 fixes (a) and (b) and recommends **(d)**:
```
V_pay = 0.5·κ·G(x)·(y − ā(x))²,   G(x) = g₀ + Σ_i m_i A_i e_i        (stiffness: FLOORED)
                                   ā(x) = Σ_i m_i A_i a_i e_i / (ε + Σ_i m_i A_i e_i)   (UNFLOORED)
```
⚠ **The normaliser must NOT be floored.** The theorist's first implementation floored it
(`ā = A a/(g₀+A)`), destroyed the value at small `A`, measured **worse than baseline** and refuted
their own P2.7. Do not repeat it. Reference implementation: `GatedStore2` in
`.claude/scratch/readout-channel-theory/q2_round2.py` (a theorist's toy, not shipped code).

**Implement in `AtomStorePotential` behind a `payload_gate` flag, `g₀` exposed as a config knob.**
⛔ **The flag defaults OFF and no shipped default changes** (B1.4 precedent) — you are measuring.

**Re-run the `mia-decay` harness §1 / §3(b) / §5 on it, including `R₅₀`**, which was **NOT
re-measured under any fix** and is the reason this task exists. Report both configurations:
- `g₀ = amp_floor` (theorist's measured 1.000 payload-independent retention) — expect a **step**
  retention curve, i.e. the TTL shape. Does `R₅₀` still contract? That is the decisive number.
- `g₀ ≪ amp_floor` **+ 4× read steps** — graded *and* payload-independent to `A = 0.06`. Report the
  read-length requirement `τ_y = η/(κ(g₀+A))` as measured. ⭐ This is **the compute-adaptive-read
  dial** (the trilemma's third corner), not a defect — but report it as a measurement, not a pitch.
Also report the payload dependence you are trying to remove (`r ≈ −0.846` with `a_i²` on the
baseline) before and after, at 3 seeds.

## File ownership (standing practice — w26's split produced zero conflicts)
**You own:** `chlu/core/placement.py`, `chlu/core/controller.py`, the **`AtomStorePotential` class
only** in `chlu/core/memory_potentials.py` (from its `class AtomStorePotential` line to EOF), the
store/controller config fields in `chlu/config.py`, and `.claude/scratch/mia-decay-measurement/`.
⛔ **Do NOT touch** `chlu/experiments/exp_designed_mechanism.py` or the
`ExperimentDesignedMechanismConfig` block in `chlu/config.py` (owned by `r2-d-sweep-close` this
wave), nor any class **above** `AtomStorePotential` in `memory_potentials.py`.

## Compute
Far lighter than `r2-d-sweep-close`, which has priority on the machine. **Cap yourself at ≤3
concurrent background jobs**; 8 cores total, no fan control on this machine (w26 hit load 575).
`PPID=1` on a background job means the harness detached it, **not** that it died.

## Deliverable
PREREG first (`.claude/outputs/deletion-waitlist-stiffness/PREREG.md`) — the two acceptance tests as
checkable items, your registered prediction for whether `R₅₀` survives the gate at `g₀ = amp_floor`,
and the load sweep you will run for Part A. Report at
`.claude/outputs/deletion-waitlist-stiffness.md`, standard format, PREREG scorecard, reconciliation
list in the first 10 lines, **plus the exact replacement scope sentence for Part A**. Full
`pytest tests/` green, `ruff` clean, atomic commits on
`agent/experiment-engineer/deletion-waitlist-stiffness`. **Do not push.**

⛔ **Do-not-quote, carried:** "certified" · "unlearning" · "deletion-compliant" · unqualified "exact
deletion" · "our fix-up cascade" as a possessive (it is Blelloch–Golovin's) · "0.99985" without its
load · any claim that decay reduces distinguishability *per se* · any claim that eviction removes the
item (under the shipped placement it does not) · **quote the curve, not the endpoint.**
