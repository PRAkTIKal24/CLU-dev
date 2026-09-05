# route3-stage1-plus-2x2 — the store-attribution curve (Route 3 stage 1) + the Route-1 × Route-2 2×2

**Campaign 2, wave C2W3. Agent:** experiment-engineer.
**Worktree MANDATORY** — you hold **Route 3's worktree, slot 3 of 3** (charter §A9.3: cap ≤3 staggered;
B′ holds first claim on 2, Route 3 holds 1, and **stage 1 shares the 2×2's worktree**). Base local
`main` (`6ff4c1d` at scoping). Branch `agent/experiment-engineer/route3-stage1-plus-2x2`.
Charter **ADDENDUM 2 §A11 task 3**, implementing **§A10 stage 1**, **§A9.4** (the unlock bar, applied
mechanically by you), **§A9.5** (the anti-lookup kill-condition, built now) and **§A9.1** (the 2×2 is
funded; the read-length re-price is **replaced** by the store-attribution curve).

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **in full, especially
ADDENDUM 2 §A8.1–§A8.2 (the interpretive finding your whole task rests on), §A9.4, §A9.5, §A10 and
§A11**; `.claude/advisor-head-intervention.md` **§5, §6, §8 (§8.2 is instantiated as your kill-condition)**;
the **live `2026-07-31` `[C2W2]` §10 entry** in `.claude/handover_context.md` (**the NOT-RUN list — the
2×2 is on it, and both C2W2 engineers independently flagged it as the wave's most informative unrun
cell**; reconciliation **#1 is yours**); `.claude/outputs/traj-write-objective.md` (Route 1: the
objective, the seam, the race card, **§6 the AttentionPsi leak you are quarantining**);
`.claude/outputs/ssb-shell-atoms.md` (Route 2: the shell rig, the r=0 gate, the tilt refutation);
`.claude/outputs/phi-particle-head.md` **§2 (the mass/friction gradient ratios — friction is the ~14×
channel)**; `.claude/outputs/trainability-spike.md` **§3.1 (`∂q*/∂q₀ = 0` exactly)**.

⛔ **REGISTRY LAG — THREE WAVES (C1W27 · C2W1 · C2W2).** Quote results **only** from
`.claude/outputs/*` and the §10 review entries — never from `negative_results.md` (N122),
`claims_matrix.md` (v2.5), `research_roadmap.md` (v0.9), the ledger (⟲ w26) or the primers, and **never
infer from a registry's silence that a result does not exist.**

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **pillar 1 (expressive latents), measurement stage.** ⛔ **You claim no dividend
  and no win.** D1–D3 are a *measurement* that decides whether Route 3 stage 2 gets built at all; D4 is
  a declared NOT-RUN being closed. Any performance reading you take is reported as an audit cell, with
  its launder, never as a headline.
- **Laundering control:** three of them, and they are the task. **(a)** the per-slot **settle-deleted
  launder** (the attribution curve *is* full − launder, per slot); **(b)** the **§A9.5 per-slot
  matched-bytes TABLE launder** — mandatory, and it is a *kill-condition*, not a control; **(c)** the
  **trajectory launder** on any ψ that can see the address block (it fired in C2W2 and caught a real
  leak — 0/54 cells on Route 1, but it is the reason the leak elsewhere was found).
  Identical φ + φ-bytes on every arm, ledgered, enforced in code (`PhiMismatchError`).
- **Falsifies:** §6. **Does NOT falsify:** a dead q-channel with a live p-channel (**§A9.4 scores them
  separately, and a live p-channel unlocks stage 2 on its own**) · a flat attribution curve on a family
  whose store was never admissibly written (that is an admissibility exclusion, reported with its
  reason, not a null).

---

## 0. Why this task exists — the finding it is built on, stated exactly (§A8.1)
C2W2's Route 1 asked the write to put information into the trajectory and it was a **monotone cost**:
dividend **−0.0278 (endpoint-only control) → −0.1111 → −0.1806 → −0.1944 → −0.2639** across a
three-decade `λ_traj` grid, `path` worse than `traj`, and the two together **−0.4722**. The engineer's
own diagnosis: the write loss is exactly **`L_endpoint + 0.55·λ`**, a constant the optimiser never
reduces, because a **60-step** write rollout only reaches **0.27–0.50** of a payload the read climbs
over **400** steps.

The Advisor's ruling on that caveat is the reason you exist, and it is **not** "buy a longer rollout":

> **The read-length requirement was an artifact of the payload convention, not intrinsic.** Payload
> channels pinned to 0 at launch must *climb*, so information lives late. **If the latent is the visited
> state itself, the store acts from the first step**: acceleration is `−M⁻¹∇V` and **`∇V` IS the store**
> — and **momentum `p_t ≈ −∫∇V dt` at small `t` is almost pure store**, while **position at small `t` is
> almost pure query** (exactly the D6/`AttentionPsi` leak mode). ⇒ **the design-deciding quantity is
> store-attribution over time, not rollout length.**

⛔ **THE READ-LENGTH ROLLOUT (400–1200 steps) IS EXPLICITLY REPLACED AND YOU DO NOT RUN IT** (§A7,
§A9.1: *"the read-length re-price is replaced by the store-attribution curve — length was the wrong
variable to buy"*). If you find yourself sizing a 20–60× write cost, you have taken a wrong turn.

## 1. Deliverables

### D1 — ⭐ THE STORE-ATTRIBUTION CURVE (§A10 stage 1; the measurement the whole track turns on)
On the **existing merged rig** (no new store, no new objective — that is stage 2's job), for each
family that admits an admissibly-written store:

> **per-slot dividend vs `t`** = (full read at slot `t`) − (**settle-deleted launder** at slot `t`),
> with **q-slots and p-slots scored SEPARATELY**, **≥3 seeds**, **laundered per slot**, against the
> **launch-noise floor**.

- **Slots:** strided points of the write-shaped trajectory, `t` spanning **small `t` to the settled
  point**, and the small-`t` end is the interesting one — that is where §A8.1 predicts `p` is almost
  pure store and `q` is almost pure query. Declare the stride grid in your PREREG.
- **The launch-noise floor** is the reference the §A9.4 bar is measured against: perturb the launch
  within its own cloud and measure the slot's discriminability under that noise alone. A slot that
  clears its own launch noise is carrying something the launch did not put there.
- ⭐ **§A8.2 — measure the flow-map Jacobian structure while you are in there, it is cheap and it is
  the mechanism:** *"encoder controls whether trajectories diverge or coincide"* = supervising the flow
  map's Jacobian — **contractive within an item's launch cloud** (noise robustness) and **separated
  across items** (capacity). Both are measurable per slot, under launder. Report them as two curves
  beside the attribution curve. They are what stage 2's objective would have to shape.

### D2 — ⭐ APPLY §A9.4 MECHANICALLY AND REPORT THE STAGE-2 UNLOCK VERDICT
The bar is **pre-registered by the Head** and you apply it arithmetically — you do not interpret it:

> **Stage 2 unlocks iff the per-slot store-attributable discriminability (full − settle-deleted launder,
> per slot `t`) clears the launch-noise floor BEYOND 2 SE, at ≥3 seeds, on ≥1 family, at ANY `t` —
> q-slots and p-slots scored separately (a live p-channel unlocks even with a dead q-channel).**

Emit `unlock = true|false` with the full arithmetic: per family × per channel (q/p) × per slot, the
margin, the SE (`sd/√3`, sample sd `ddof=1`), and the clearing set. ⛔ **Admissible-cell coverage per
family is a first-class reported number**, at the top, before any verdict — the C2W2 standing rule
exists because *"what must not happen is admissibility filtering quietly gutting coverage."*
⛔ **A family with zero admissibly-written cells ABSTAINS** — it neither unlocks stage 2 nor blocks it —
and it gets **one bounded budget escalation** before it may be called "not testable this wave"
(the C2W2 ruling (i) counterweight; it rescued `overload` and it stopped `aggregate`/`manifold` voting
off an unwritten store).

⚠ **`overload` is quoted only at `load1x_shipped` (478×)** — at the gym's base atom budget it went
**0/18 admissible including the Gaussian control** (reconciliation 6, Hub-accepted).

### D3 — ⭐ BUILD THE §A9.5 PER-SLOT TABLE LAUNDER **NOW**, in stage 1
Intervention §8.2 instantiated, and it is a **kill-condition, not a control**:

> **A per-slot matched-bytes table launder is mandatory. If it reproduces the slotted read, Route 3 has
> degenerated into K time-indexed lookup tables and FAILS REGARDLESS OF DIVIDEND. The headline claim
> must require inter-slot / dynamical coupling that no per-slot table can express.**

Build it in stage 1 — **before** anything is built on top of it — so the kill-condition is measurable
from the very first cell rather than retrofitted onto a result someone has grown attached to. Report,
per slot: the table launder's score, the slotted read's score, and the **signed margin**. If K
independent time-indexed tables reproduce the read, say so in your first 10 lines.

### D4 — ⭐ THE ROUTE-1 × ROUTE-2 2×2 (the funded declared NOT-RUN, §A9.1)
C2W2's most informative unrun cell, flagged independently by **both** engineers, and now cheap: both
branches are merged, the seam takes `write_objective=…` (`CluSystem(write_objective=...)` /
`build_system(write_objective=...)`, normalised by `normalize_write_objective`), and the shell rig is
**orthogonal** to it.

| | Gaussian atoms | shell atoms |
|---|---|---|
| **endpoint write** (control) | the shipped anchor — must reproduce `memory-gym-v0` **digit-for-digit** | Route 2 alone |
| **trajectory/path write** | Route 1 alone | ⭐ **the unrun cell** |

- **3 seeds**, full race-card cells (`chlu/eval/race.py` — **frozen schema**, emit into it, do not
  change it), **dividend + launder + substitute audit + two-sided byte ledger on every cell**.
- ⛔ **The two regression gates are BLOCKING and they are bit-identity, not tolerance:**
  **(a)** at `λ_traj = λ_path = 0` the written `V` is bit-identical to `main`'s (Route 1's
  coefficient-zero gate — both terms sit behind a Python-level `>0` branch, so at λ=0 not one extra op
  is traced); **(b)** at `r = 0` the shell atom reduces to the Gaussian **bit-identically** in f32
  **and** f64, including at `q == c` (Route 2's r=0 gate — the Hub re-verified 12/12 cells last wave).
  If either fails, everything downstream is uninterpretable — stop and report.
- ⭐ **The hypothesis worth stating before you run it:** Route 1's `λ_path` term is *exactly* the signal
  the shell radius is currently blind to (learned `r` moved **0.500 → 0.501** in 300 steps — **not**
  w20's "learning erases design" but something weaker and worse: **the design was ignored, because the
  objective could not see it**). The 2×2 asks whether an objective that *can* see a path makes the
  designed degeneracy visible. Pre-register your prediction.
- ⚠ **The byte surcharge is real:** the shell raises the architectural floor by `1/(dim+2)` — **52.00 →
  58.40×** at `dim = 6`, **+12.5 % on the atom term**. It goes in every shell cell's ledger.
- ⚠ **`ε` (tilt) is refuted as a mechanism on a learned store** (§A4.2 struck): tilt **monotonically
  reduces** `λ_min` (**+0.0994 → −8.28**), on two independent implementations and every family, because
  **a designed degeneracy does not survive superposition** (a written site's vacuum residual
  **0.140–0.343** vs a *random-orientation* baseline of **0.167**). Do not re-litigate it; run the 2×2
  at the **shell arms that survived**, and if you run a tilt arm at all it is `ε` small and declared.

### D5 — ⭐ RIDER: QUARANTINE `AttentionPsi` (§A11 harness-debt rider · reconciliation 1)
C2W2's spike R-4 closed and **fired**: `AttentionPsi` reads `q0_only` at **0.35–0.45** against a bar of
**0.19**, **at every stride**, with a blank store at **0.37–0.47** ⇒ **the pooled-vs-attention hypothesis
is CONFIRMED and no attention-ψ trajectory number is quotable store-relative.**

- In `chlu/core/psi_readout.py`: `AttentionPsi` must **refuse** to produce a store-relative reading
  unless the leak probe passes (`q0_only` below the registered bar on a blank store). **Raise, not
  warn** — the C2W2 `PhiMismatchError` precedent: an invariant enforced in prose is not enforced.
- Put the leak evidence **in the docstring**, with the numbers and the stride-independence, so the next
  person to reach for it reads why before they reach.
- ⛔ **Default-off, additive, bit-identical shipped-behaviour regression test** where the quarantine is
  disabled. Nothing else in `psi_readout.py` changes.
- ⚠ **This does not bar `bprime-fb4-gate`'s `attention` arm** — that is a *table reader* over the
  launder's own (key, payload) rows, it never sees a trajectory, and it is a different object. Note the
  distinction in your report so no one conflates them at the wave review.

⛔ **`store_write_mask_factory` (reconciliation 5) is NOT yours** — it lands in `bprime-fb4-gate`'s
branch (it owns `clu_system.py` this wave). **Route 3 stage 2 gates on that merge.** If stage 1 unlocks
stage 2 before it lands, tell the Hub.

## 2. Compute
Stage 1 is **hours** and it shares the 2×2's rig by design (§A10: *"measurement, hours, shares the 2×2
rig"*). The 2×2 is 4 arms × families × 3 seeds on a rig that already exists. ⚠ **JAX cold-start ~20
min** — keep the session warm, `--quick` for smoke runs, reuse the main venv
(`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in the worktree); if you must
`uv sync`, use `--frozen` and report the resolved JAX version in your flag-provenance table.

## 3. PREREG (`.claude/outputs/route3-stage1-plus-2x2/PREREG.md`, before ANY measured run)
Mandatory — your acceptance criteria are measured margins and a mechanical bar. It must contain:
1. **The §A9.4 bar, verbatim**, and the exact arithmetic you will apply (2 SE, ≥3 seeds, per channel).
2. **The stride/slot grid**, declared before you look at anything.
3. Your predicted **shape** of the attribution curve for q and for p separately. (The Advisor's
   prediction, on the record, is that **`p` at small `t` is almost pure store and `q` at small `t` is
   almost pure query**. Confirm it, refute it, or find it is neither — all three are results.)
4. Your predicted 2×2 outcome, with the derivation.
5. Predicted per-slot table-launder margins — i.e. **do you expect §A9.5 to kill it?**

## 4. Compute-discipline reminder that has cost this program before
⛔ **Declare NOT-RUNs; never report them as nulls.** ⛔ **Multi-seed before any paper number.**
⛔ **Quote the curve, not the endpoint.** ⛔ **`D` (monitor #2) is the dividend's VARIANCE, not its
magnitude** — the `D = 0.931` cell has dividend **−0.875**. It bounds where a dividend could live and it
is **never** a progress signal.

## 5. Falsifiers
- ⛔ **No slot on any family clears §A9.4 in either channel** ⇒ ⭐ **Route 3 is DEAD at this weight
  class**, stage 2 does not unlock, and **B′ absorbs the attribution curve as protocol evidence**
  (§A10, verbatim). This is a clean, cheap, decisive outcome and it is not a failure of the task.
- ⛔ **The §A9.5 per-slot table launder reproduces the slotted read** ⇒ **Route 3 has degenerated into
  K time-indexed lookup tables and FAILS REGARDLESS OF DIVIDEND** (intervention §8.2). Report it in your
  first 10 lines; it overrides any positive attribution result.
- ⛔ **Either regression gate (coefficient-zero, r=0) fails bit-identity** ⇒ the 2×2 is uninterpretable.
- ⛔ **The trajectory launder fires** on any ψ you use ⇒ no ψ number from that arm is quotable.
- **Does NOT falsify:** a dead q-channel with a live p-channel (**that unlocks stage 2 by design**) · the
  2×2 coming out ≤0 (the gate already fired; the 2×2 is a declared re-price, not a second gate) · a
  family abstaining after its one bounded escalation.

## 6. File ownership (the zero-conflict standing practice — 9 branches / 3 waves / 0 conflicts)
**Yours, exclusively this wave:**
`chlu/core/psi_readout.py` · `chlu/eval/attribution.py` (**new**) ·
`chlu/experiments/exp_route3_attribution.py` (**new**) · `chlu/experiments/exp_traj_write.py` ·
`chlu/experiments/exp_ssb_shell.py` · `chlu/core/shell_atoms.py` ·
`tests/test_{psi_readout,attribution,route3_attribution,traj_write_objective,ssb_shell}.py`

**Read-only to you:** `chlu/config.py` (**standing read-only to all C2 engineers**) ·
`chlu/core/clu_system.py`, `chlu/core/{monitors,admission,soft_certificate}.py`,
`chlu/eval/fb4_gate.py` (`bprime-fb4-gate` — ⛔ **you consume the `write_objective` seam, you do not edit
it**; if stage 1 needs a change in `clu_system.py`, **STOP and report to the Hub**) ·
`chlu/eval/rivals/**`, `chlu/eval/dividend.py`, `chlu/experiments/exp_bprime_rivals.py`
(`bprime-rivals`; you **import** from `dividend.py` freely, you do not edit it) ·
`chlu/eval/race.py` (**frozen schema** — emit into it, do not change it) ·
`chlu/training/train_memory.py` (**stage 2's, if it unlocks — leave it alone in stage 1**).

⛔ **If your work requires editing a file you do not own, STOP and report to the Hub.**

## 7. ⛔ Never-quote (inherited — violating these sends the report back)
any **`AttentionPsi`** trajectory number (**it leaks** — that is your own D5) · monitor #6's
**"58 trips"** without *"pre-repair"* (**27**; the artefact count is **31 of 58**, not 29) · the ridge
saddle **`λ_min = −0.5946`** as a multi-seed result (**seed 0**; 3-seed mean **+0.177 ± 0.469**) · the
recency family's **`0.3019 ± 0.0679`** as a null (scoring-domain **defect**; post-fix **−0.0028 ±
0.0619**) · **"Prop D1 is violated (1.5–7.44×)"** (retired — `U` was computed against a certificate that
did not exist) · **`sep/2`** as a certified inradius, and the corrected proxy outside `s/sep ∈ [0.15,
0.30]` · **`λ_min > 0`** as certifying a nonempty basin (measured basin **0.000** at `λ_min = +0.910`) ·
**`k*`** without *"of `∂q_N/∂θ`, and only where the fixed-point sensitivity dominates the transient"* ·
**`ε` as "the manifold-payload lifetime dial ∝ 1/ε"** without the `2α` coercivity ceiling
(`τ_max = Γ/2α`; **`α` is the ceiling and lowering it breaks the write**) · **Titans as "a preprint"**
(NeurIPS 2025) · any **SDM Table 1 state/param ratio** · **"MAD `compression` is the admissible
synthetic"** · **"principled forgetting"** as a novelty phrase · any C2W3 cell as a **byte-matched**
dividend (min ratio anywhere **17.11×**; the shell *raises* the floor by `1/(dim+2)`).

## 8. Output
`.claude/outputs/route3-stage1-plus-2x2.md`, protocol §5 format, with:
- ⭐ **`unlock = true|false` in the first 10 lines**, with the clearing set (family × channel × slot) and
  the margins that produced it — the Hub spawns or does not spawn `route3-stage2` off this line;
- ⭐ **the §A9.5 per-slot table-launder verdict in the first 10 lines** — it overrides the unlock;
- **admissible-cell coverage per family, first-class, before any verdict**;
- the attribution curve (q and p, separately) and the two Jacobian curves, as **curves**, not endpoints;
- the 2×2 race card with both bit-identity gates' results;
- the flag-provenance table (commit, seeds, every non-default flag — mandatory);
- your reconciliation list in the **first 10 lines** if you produce one (protocol §5 corollary);
- ⛔ **declared NOT-RUNs, never reported as nulls.**

⚠ **Report D2's unlock verdict to the Hub as an INTERIM the moment you have it** — `route3-stage2` is
scoped, written and waiting on exactly that line.
