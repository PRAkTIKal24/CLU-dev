# trainability-spike — end-to-end gradients query→φ→settle→ψ→loss, + the learned-ψ trajectory-read pilot

**Campaign 2, wave C2W1. Agent:** experiment-engineer. **Worktree MANDATORY.**
Base local `main`, **branched off `full-clu-harness`'s API-FREEZE commit** (see §Dependency).
Branch `agent/experiment-engineer/trainability-spike`. Charter §6.4, engineer half.
Companion: `.claude/tasks/trainability-spike-theory.md` (theorist half — **read its output before you
choose a tolerance or a truncation depth**; it is deriving both).

Read first: `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c2-charter.md`
(**§2.4 placeholder policy — implicit/DEQ gradients, truncated backprop, plain Adam on φ/ψ/controller
are all Head-approved; §4 pillar 1**), `.claude/advisor-head-intervention.md`
(**§3.2 and §3.7 — trajectory-as-latent and a learned ψ are the two things NEVER tested in 26 waves**),
`.claude/tasks/full-clu-harness.md` (the API you build on), and **`.claude/outputs/full-clu-harness.md`
(the harness AS LANDED — it flags two things directly at you)**.

---

## ⛔⛔ C2W1 AMENDMENT (Hub, 2026-07-30) — READ BEFORE PLANNING PART B
`full-clu-harness` has landed and **merged to `main @ 4160cf7`** — branch off `main`, not off its branch;
the API-FREEZE surface is in the tree (`clu_system.py`, `monitors.py`, `clu_controller.py`,
`eval/dividend.py`). `read()` returns the strided trajectory + `q*`, and the settle exposes
`fixed_point_residual()` for your implicit attachment, exactly as specified. **But:**

**⛔ THE TRAJECTORY AXIS IS MEASURED SEMANTICALLY DEAD AT THE SHIPPED CONFIG.** Monitor #10 tripped on it:
`traj_stride` moves the read by **2.97e-4** noise units and `gamma_read` by **8.9e-4** — both **far below
the 3σ bar** (Hub re-derived both from `exp_clu_system_metrics.json`, digit-for-digit). And the harness's
S6 "+trajectory ψ" stage produced decode **0.4792 — bit-identical to S5 without it**.

**This is collapse mode #10 (degenerate axes / silent knobs), and it is aimed straight at your headline
deliverable.** If you run the stride sweep and the point-vs-trajectory ablation as originally written,
you will sweep a dead axis, get a flat result, and — under this task's own "does NOT falsify" clause —
file it as *"pillar 1's first honest datum."* **It would be nothing of the kind: it would be an
instrument artifact reported as a scientific null on the program's highest-novelty pillar.**
⚠ Note the confound in the other direction too: S6/R4 ran at the **collapsed S4 geometry**
(`sep/σ_q = 3.07`, decode 0.479). A trajectory read cannot show value where addressing has already
failed. **Do not conclude anything from S6.**

⇒ **A new Stage 0 is inserted below and it GATES Part B.** Do not run the ablation until it passes.

## ⭐ STAGE 0 (NEW, blocking) — establish that the trajectory channel carries ANY signal
Before the ablation, prove the axis is alive, at the **healthy** geometry (S0/S3 or the R1/R3
annealed-read configuration, `sep/σ_q = 6.83` — **not** S4's 3.07):
1. **Information probe, no learning.** Does the strided trajectory contain information the settled point
   does not? Cheapest honest test: can *any* decoder (linear probe / kNN on trajectory features)
   recover something about the query, the item, or the competing-well structure that the same decoder
   cannot recover from `q*` alone? This costs a probe, not a training run.
2. **Find the live regime, or prove there isn't one.** If the axis is dead at the shipped read, sweep the
   parameters that *should* wake it — read length, `γ_read`, and **the ambiguity of the query** (charter
   §2.1(c): *a trajectory passing near competing wells encodes a distribution over answers; a settled
   point cannot* — so the signal should appear under **query ambiguity**, and be genuinely absent when a
   query falls cleanly into one basin). **A trajectory read has no reason to help an unambiguous query.**
3. **Gate.** If Stage 0 finds a regime where the trajectory is demonstrably informative → run Part B
   **there**, and report the regime as part of the result. If Stage 0 finds **no** such regime anywhere
   in the swept range → **STOP, do not run the ablation, and report that as the finding**: *"pillar 1's
   trajectory channel is not measurably live on this harness at v0, here is the range swept."*
   ⭐ **That is a real, publishable, honest negative and it is worth more than a flat ablation over a
   dead knob.** It is also a monitor-#10 result, which is exactly what the monitor exists to produce.
4. Report the axis-liveness numbers (movement in noise units, with the 3σ bar) **next to** every ablation
   number you later quote. No ablation number travels without its axis-liveness evidence.

**⚠ Also flagged directly at you by the harness (its PREREG refutation #4):** the blank control stayed at
chance under a tail-mean ψ only because that ψ reads the **payload channel alone**. **The moment your
learned ψ can see the address block, the `q₀ = φ(x)` leak becomes live and the TRAJECTORY LAUNDER IS
MANDATORY** — it is implemented in `chlu/eval/dividend.py` and **has never been run**; running it is your
task. A learned ψ that sees the address is exactly the N68 configuration (blanks scored 0.992–1.000).

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form)
- **Dial (C2 form):** **pillar 1 — expressive latents (point / trajectory / manifold)**, plus the
  trainability infrastructure that every later C2 wave depends on. ⚠ Charter §4: this pillar has
  **zero direct evidence** — trajectory-as-latent has never been run. The point-vs-trajectory ablation
  you run here is the program's **first honest datum** on its highest-novelty claim.
- **Laundering control:** ⭐ **the point-vs-trajectory ablation is itself an internal launder** — the
  settled-point read *is* the "trajectory deleted" substitute, on the same harness, same bytes, same φ.
  Report it that way. Additionally: **implicit gradients are checked against truncated unroll**
  (the substitute-gradient control), and any accuracy number carries the harness-native
  blank/same-keys controls (`chlu/eval/dividend.py`).
- **Falsifies:** (i) implicit gradients do **not** match truncated-unroll gradients to the tolerance the
  theorist registers, on a controlled toy where both are computable; **or** (ii) wall-clock per training
  step exceeds your **declared usability budget by >10×** (declare the budget in PREREG — "usable
  wall-clock" is meaningless undeclared).
- **⭐ Does NOT falsify:** the **point-vs-trajectory ablation coming out flat or negative — PROVIDED
  STAGE 0 PASSED FIRST.** With a live axis, a flat ablation is a *result* about pillar 1, the first the
  program has ever had, and it is reported as the headline either way. Do not tune until it comes out
  positive; that is how a laundering gets manufactured. ⛔ **Without Stage 0, a flat ablation is NOT a
  result and must not be filed as one** — it is monitor #10 firing, and the honest report is
  "axis dead, range swept," not "trajectories carry no information."
  Also does not falsify: needing a ridge term `(H + λI)⁻¹` (standard DEQ practice) · a v0 ψ that is
  weaker than a handcrafted read (say so and quote the curve).

---

## Part A — implicit/DEQ gradients through the settle
**A settle is a fixed point; differentiate through the equilibrium, not the unroll** (charter §2.4).
1. Implement `chlu/core/implicit_grad.py`: a custom-VJP settle whose backward pass solves the implicit
   system rather than unrolling. Use the theorist's stated invertibility condition **for the discrete
   dissipative Verlet map** (not for a gradient flow — the constants differ) and its `(γ, dt, M)`
   prefactors. Ridge/damping is permitted and **must be reported as a flag**, never silently enabled.
2. **Gradcheck** vs (a) truncated unroll at the theorist's registered depth and (b) finite differences,
   on a controlled toy with a known answer. Register the tolerance **before** running. ⚠ Prior art you
   are matching: the C1 controller spec verified `∂q*/∂θ = −H⁻¹∂_θ∇V` against finite differences to
   **1e-5 (atom amplitude) / 1e-7 (confinement α) / 3.1e-7 (mid-range atom at write distance 1.2)** —
   your implementation should reach the same order or you have a bug, not a discovery.
3. **Conditioning telemetry.** Log `λ_min` (or the ridge actually needed) per step. ⭐ The theorist is
   testing whether **reach failure (collapse mode #11) and implicit-gradient ill-conditioning are the
   same object**. If they are, wire `monitors.py`'s reach monitor as your trainer health check —
   **by consuming the monitor, not by editing `monitors.py`** (not yours).
4. **Truncated backprop over trajectory reads.** The trajectory read is *not* at a fixed point, so the
   implicit theorem does not apply — use truncation at the theorist's depth. ⚠ The gradient through the
   unroll dies geometrically (`∂R_γ/∂q₀ = 2.2e-12` after 3000 damped steps): **do not spend compute on
   unroll depth that is numerically worthless.**
5. **Plain Adam on φ / ψ / controller parameters** (charter §2.4 — approved, standard machinery). No
   bespoke optimizer. **The trainer is replaceable infrastructure and the paper says so as a strength**;
   do not invent here.

## Part B — the learned-ψ trajectory-read pilot ⭐
The read-out has **never** been learned, and the trajectory has **never** been read (intervention §3.2,
§3.7). `full-clu-harness`'s `read()` returns the strided trajectory *and* `q*` — that hook exists
precisely so this is a configuration change, not a rewrite.
1. `chlu/core/psi_readout.py`: a learned `ψ` over **strided trajectory points**, with **DeepSets** and
   **attention pooling** variants (charter §6.4 names both). Both must accept the settled-point-only
   input as a degenerate case — that is what makes the ablation internal and fair.
2. **The point-vs-trajectory ablation — the deliverable. ⛔ GATED ON STAGE 0 PASSING.** Same ψ family,
   same parameter count, same bytes, same φ, same seeds: settled-point-only vs strided-trajectory input.
   ⭐ **Matched parameters and matched bytes are non-negotiable** — a trajectory read that wins by being
   bigger is not a result. Report the **stride sweep as a curve** (standing rule: quote the curve, not
   the endpoint), **in the regime Stage 0 identified as live**, and state that regime in every quotation.
   ⚠ Run it at the **healthy** geometry (`sep/σ_q ≈ 6.83`), never at S4's 3.07 — the harness showed a
   trajectory ψ cannot demonstrate value where addressing has already collapsed.
3. **What the trajectory could carry that a point cannot** (charter §2.1(c)): order, ranked
   alternatives, ambiguity — *a trajectory passing near competing wells encodes a distribution over
   answers; a settled point cannot.* If `memory-gym-v0`'s trajectory family (opening (c)) has landed by
   the time you get here, score on it too; if not, an internal probe is acceptable for v0 — say which.

## Acceptance criterion (charter §6.4)
**Gradients flow end-to-end `query → φ → settle → ψ → loss` at usable wall-clock, and the
point-vs-trajectory ablation runs.** Both halves required. "Usable" is the budget you declared in PREREG.

## Dependency & sequencing
✅ **CLEARED — `full-clu-harness` is merged to `main @ 4160cf7`. Base local `main`; the whole API surface
is in the tree.** `trainability-spike-theory` has also landed
(`.claude/outputs/trainability-spike-theory.md`) — **take your gradcheck tolerance and truncation depth
from it**, and register them in PREREG before running. If the API is missing a hook you need, **request
it from the Hub** — do not edit `clu_system.py` or `monitors.py` yourself.
**Order: Part A (gradcheck, no harness needed) → Stage 0 (blocking axis-liveness gate) → Part B.**

## Compute
⚠ **≤3 engineer worktrees across BOTH campaigns**; the C1W27 Hub is running in parallel. Do not launch
background sweeps without telling the Hub. JAX cold-start ~20+ min — budget it, keep the session warm,
`--quick` for smokes, and do not mistake a slow import for a hang. Declare the compute order in PREREG;
anything unreached is **NOT RUN**, never a null. **Priority if compute is short: A1–A2 (gradcheck) →
B1–B2 (the ablation) → A3–A4 → B3.** The ablation outranks polish — it is the pillar's first datum.

## File ownership
**You own (all NEW files):** `chlu/core/implicit_grad.py` · `chlu/core/psi_readout.py` ·
`tests/test_implicit_grad.py` · `tests/test_psi_readout.py` · a pilot runner under
`chlu/experiments/exp_trajectory_read.py`.

⛔ **READ-ONLY — do not edit:** `chlu/core/clu_system.py`, `chlu/core/monitors.py`,
`chlu/core/clu_controller.py`, `chlu/eval/dividend.py`, `chlu/experiments/exp_clu_system.py`
(`full-clu-harness`, now merged) · `chlu/experiments/memory_gym.py`, `exp_memory_gym.py`
(`memory-gym-v0`) · ⚠ **`chlu/cli/experiment_cmd.py`** — the one shared file `full-clu-harness` had to
edit (+53 lines), and **`memory-gym-v0` holds sole ownership of it this wave**. **Do not add a CLI hook.**
Run your pilot by module invocation exactly as the harness ran its own:
`PYTHONPATH=. python -u -m chlu.experiments.exp_trajectory_read --seed N`.
⛔ **READ-ONLY — C1W27 territory, do not edit even by one line:** `chlu/core/memory_potentials.py` ·
`chlu/core/controller.py` and `AtomStorePotential.evict` · `chlu/core/placement.py` ·
`chlu/core/admission.py` · `chlu/core/integrators.py` ⚠ (**the shipped Verlet step — if implicit-grad
needs an integrator change, STOP and report to the Hub; do not touch it, and do not fork a copy of it
into your own file without saying so loudly**) · `chlu/experiments/exp_designed_mechanism.py` ·
`exp_cl_entry.py` · `cl_baselines.py` · `exp_phi_stream.py` · the mia harness · ⚠ **`chlu/config.py`**
(C1W27 owns two blocks in it this wave). **This repo is config-driven and the reflex is to add your
config class there — do not.** ψ / implicit-grad config objects live in your own modules.
**Consume by import; wrap; never edit.**

## Deliverable
`PREREG.md` at `.claude/outputs/trainability-spike/PREREG.md` **before any measured run** — the
gradcheck tolerance (from the theorist), the truncation depth (from the theorist), your **declared
wall-clock usability budget**, your predicted sign and magnitude for the point-vs-trajectory ablation
(⭐ commit to a number — *a pre-registered prediction that survives is evidence; one that fails is a
finding; an un-pre-registered agreement is neither*), and the compute order.
Report at `.claude/outputs/trainability-spike.md`, protocol §5 format, flag-provenance table on every
quantitative result (incl. ridge λ, stride, truncation depth, seeds), PREREG scorecard, reconciliation
list in the **first 10 lines**. Full `uv run pytest -q` green, `ruff` clean, atomic
`[experiment-engineer]`-prefixed commits. **Do not push. Do not merge.**

⛔ **Do-not-quote, carried:** any `tol`-metric number at m>1 · `K_learned` at `pscale ≠ 1` without the
payload-noise condition · "the write operator is the ceiling" · width-lock-as-cause · the √2 / `d^1.62`
exponent · N46's coset register as anything but **designed-only** · "certified".
**Standing:** laundering on every performance claim · **multi-seed before any paper number** ·
quote the curve, not the endpoint · N94 epoch discipline (every diagnostic states its epoch count) ·
`git -C <worktree>` always.
</content>
