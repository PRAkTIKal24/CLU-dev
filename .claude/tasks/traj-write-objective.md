# traj-write-objective — Route 1: ask the write to put information in the trajectory, not just the endpoint

**Campaign 2, wave C2W2. Agent:** experiment-engineer. **Worktree MANDATORY.**
Base local `main` (`233fd9e` at scoping). Branch `agent/experiment-engineer/traj-write-objective`.
Charter Addendum-1 **§A5/C2W2 task 1**, implementing **§A2.1 (the point-estimator diagnosis)** and
**§A2.2 (trajectory read = trainability precondition)**. Bound by intervention **§8** and **§5**.

**Read first:** `.claude/AGENT_PROTOCOL.md` (incl. §5 pre-registration, §7 dial declaration),
`.claude/advisor-head-c2-charter.md` **IN FULL including ADDENDUM 1**,
`.claude/advisor-head-intervention.md` (**§5 the 13 modes · §6 the five criteria · §8 the prohibitions**),
and the C2W1 evidence you build directly on: `.claude/outputs/trainability-spike.md` (**§4 Stage 0, §4.3
the blank-store probe control, §9.2 — this task IS its recommended next experiment**),
`.claude/outputs/memory-gym-v0.md` (**§3.1 the byte floor, §3.2 the per-family dividends, §3.5 the ridge
saddle, §8.1 the fit-budget caveat**), `.claude/outputs/full-clu-harness.md` (**§3.5, §3.6**).

⛔ **REGISTRY LAG (Head parked the curator pass, 2026-07-30).** `negative_results.md` (N122),
`claims_matrix.md` (v2.5), `research_roadmap.md` (v0.9), the ledger (⟲ w26) and the primers are **two
campaigns behind**: **C1W27's and ALL of C2W1's results are in no registry.** Quote them **only** from
`.claude/outputs/*` and the `[C1W27]`/`[C2W1]` §10 entries — never from the matrix/registry/roadmap, and
never read a registry's silence as evidence that a result does not exist.

**You are the wave's spine.** Two other engineer tasks branch off your first commit, and the C2W2 gate
is evaluated on the race card you build. Land D0 fast and tell the Hub.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo this before your first result
- **Dial / pillar:** **pillar 1 — expressive latents.** The dividend axis is the only KPI:
  **dividend ≡ (full CLU) − (its own settle-deleted launder)**, same store, same φ, same read budget.
- **Laundering control:** four, all mandatory on every cell — (i) **settle-deleted launder** (the
  dividend's own denominator), (ii) **same-keys null** (same keys, permuted payloads), (iii)
  **blank/empty-store** control, (iv) ⭐ **the trajectory launder** (`full` / `q0_only` / `endpoints` /
  `blank_store`) on **every ψ that can see the address block — mandatory, no exceptions** (charter
  §A5/C2W2.1). Plus the family's strongest **+0 B substitute** (the audit that went 0-for-4 at v0).
- **Falsifies the claim:** see §6. **Does NOT falsify:** a ≤0 dividend *provided the write-term liveness
  check in §4 passed* — that outcome is the gate's own ≤0 branch and is a result, not a failure;
  17× ψ wall-clock (§A4.4 accepted it); `matched=False` bytes (architectural, ≥2.20×, gym PREREG-B1).

---

## 0. What you inherit, in four sentences
C2W1 measured the dividend at **≈0 or negative on every family** (best `aggregate/tight` **+0.008 ±
0.035**; the shipped anchor **exactly 0.0000 with `D = 0`**), the trajectory channel **not measurably
live** (0 qualifying cells of 588 × 3 seeds, including at `γ_read = 0`), and the substitute audit
**0-for-4**. The Advisor's ratified diagnosis (§A2.1) is that **the write loss constrains only isolated
settled endpoints**, so every read that touches the in-between regions loses by construction — *the
charter's actual hypothesis has never been tested*. The one structural positive is
**`∂q*/∂q₀ = 0` exactly** (‖∂L/∂φ‖ 0.0 implicit / 2.654e-9 unroll / **6.421e-3 trajectory**), i.e. the
trajectory read is the **only trainable read**. Your job is to ask the write objective, explicitly, to
put information into the trajectory and the path — and then measure honestly whether it buys anything.

## 1. ⚖ THE C2W2 GATE (binding decision rule — reproduced VERBATIM from charter §A5)

> **⚖ C2W2 GATE (binding decision rule):** if writes that *explicitly ask* trajectories/paths to carry
> information still give dividend ≤ 0 on every family (multi-seed, substitute-audited, both routes)
> ⇒ **B′ activates** and C2W3 re-scopes toward the audit paper. If any family clears 0 beyond 2 SE ⇒
> C2W3 proceeds as planned. Either way the direction is re-priced, not closed.

**Your race card is half the gate's evidence** (`ssb-shell-atoms` is the other half). **Head-ratified
rulings on how the gate is *applied*** (2026-07-30), all binding on you — each adopted **with a
counterweight, so the gate stays both fireable and ungameable**:

**(i) Write convergence.** A cell may only **vote** if its write **converged**: report `final_write_loss`
and `λ_min` at every recorded site; a cell with `λ_min < 0` or a plateaued loss is reported, labelled
`gate_admissible=false`, and does **not** cast a ≤0 vote (gym §8.1 measured exactly this confound —
letting unconverged cells vote would fire B′ on noise). ⭐ **Counterweight, and it is on you:**
- **Every excluded cell is reported with its exclusion reason.** Silent filtering is forbidden.
- **A family with zero admissible cells gets ONE bounded budget escalation** — the D5 longer-write arm,
  already scoped — before it may be called "not testable this wave."
- **If it still has no admissible cells after escalation, the family ABSTAINS:** it neither blocks B′
  nor supports "proceed." Say so explicitly.
- ⛔ **`admissible-cell coverage per family` is a FIRST-CLASS REPORTED NUMBER**, at the top of your
  results, not buried in a JSON field. *What must not happen is admissibility filtering quietly gutting
  coverage until B′ can never fire* — the Head named this failure mode at scoping and you are the one
  positioned to cause it or prevent it.

**(ii) Term liveness.** A cell may only vote as *"asked and did not deliver"* if the objective term was
**live**; a term inert at **every** registered coefficient **does** vote, one inert at a single setting
does not. ⭐ **Counterweight — the coefficient grid must SPAN to where the term is live:**
- The grid is **declared before any run** (in this file's §2/D1 shape, and pinned with numbers in your
  PREREG), never chosen after seeing a result.
- **It must include at least one coefficient at which the term VISIBLY PERTURBS THE WRITE — even
  destructively** (e.g. a setting that measurably damages endpoint recall, or moves `λ_min`, or moves
  the write loss). Report that cell as the grid's liveness anchor.
- ⛔ *"A term that never moves anything at any tested setting hasn't been asked; it's been whispered
  at"* (Head, verbatim). Without the perturbing anchor, an inert-everywhere result is **not** a
  legitimate ≤0 vote and must be reported as an under-powered grid instead.
- Minimum grid shape: **≥3 non-zero coefficients spanning ≥2 decades**, plus the perturbing anchor
  (which may be the top of the range) and the mandatory zero point (the regression gate, D1).

**(iii) The "weak proceed" caveat** (Head ruling, now **charter §A6**, pre-registered before
adjudication): a family that clears 2 SE **but loses to its +0 B substitute** is a **weak proceed** —
C2W3 sequences as planned but cannot take that family as its sole headline basis, and diagnosing or
beating the substitute deficit becomes a mandatory C2W3 first-order item. ⇒ **Record the +0 B substitute
margin per family beside the dividend** (`plus_zero_byte_substitute` is already in the race-card schema —
also emit the signed margin), so the Hub can compute the grade at review instead of arguing it.

## 2. Deliverables

### D0 — ⭐ RACE+SEAM FREEZE (your FIRST commit; nothing else in it; report the hash to the Hub immediately)
Public surface only, no science, so the other two engineers are unblocked:
- **`chlu/eval/race.py` (new)** — the **race card**: the cell schema every C2W2 arm emits, and the
  scorer. One JSON record per cell containing at minimum:
  `{route, arm, family, seed, metric_name, full, settle_deleted_launder, same_keys_null, blank,
    plus_zero_byte_substitute, dividend, dividend_se, trajectory_launder{full,q0_only,endpoints,blank_store},
    bytes{full,launder,ratio,matched}, phi_id, phi_bytes, write{steps,final_loss,lambda_min_min,converged},
    liveness{...}, monitors{...}, gate_admissible, seeds_n, flags{...}}`.
  Both routes emit this schema; nothing else is comparable across branches.
- **Two seams in `chlu/core/clu_system.py`** (you own this file this wave):
  (a) a **write-objective passthrough** so a caller can hand `train_memory_landscape` an objective spec
      without editing `CluSystemConfig` semantics; (b) a **store-potential factory hook** —
      `store_potential_factory: str | None` resolved by import path — so `ssb-shell-atoms` can register
      the shell-atom family **without editing a single line you own**. Default `None` ⇒ shipped
      behaviour **bit-identical** (assert it in a test).
- ⚠ The freeze commit must leave the full suite green (880 at scoping) and shipped behaviour unchanged.

### D1 — Route 1: the write objective (`chlu/training/train_memory.py`, additive)
Two new terms, each behind its own coefficient, **both defaulting to 0.0**:
- **`λ_traj` — a trajectory-information term.** The write is asked to make the *path* from the launch
  manifold to the item discriminative, not only the endpoint. Design it yourself; declare the exact
  form in PREREG before running. Reasonable candidates (pick one primary, name the others as NOT-RUN):
  a margin on a strided-trajectory read-out, a contrastive term between the true item's path and the
  nearest competitor's, or an explicit mutual-information surrogate on the strided buffer.
- **`λ_path` — the path / equal-depth term** (the gym's named blocker, §3.5): for multi-target writes
  the connecting path must be constrained to **equal depth / zero gradient along the tangent**. The gym
  measured that today a multi-target ridge write produces a **saddle** (`λ_min = −0.5946`, spectator
  participation 1.000), not a valley, *because `write_loss` minimises `V` at each target independently*.
  Your acceptance on this term is spectral, not accuracy-based: **`λ_min` at the ridge item must move
  from −0.595 to ≥ 0** with the tangent direction the softest mode.
- ⛔ **COEFFICIENT-ZERO REGRESSION GATE (blocking, mirrors `ssb-shell-atoms`'s r=0 gate):** at
  `λ_traj = λ_path = 0` the written `V` must be **bit-identical** to `main`'s. Ship a test. If it is not
  bit-identical, stop and fix before any science cell runs.

### D2 — ψ, trained on the store the new objective produced
Use the C2W1 ψ family **unmodified** (`chlu/core/psi_readout.py` is `phi-particle-head`'s this wave —
read-only to you). Both arms at **identical parameter count** (`matched_pair` asserts it). The point-ψ
arm is the internal launder ("trajectory deleted"). **Every ψ number carries its trajectory launder.**

### D3 — The race card run (the gate's evidence)
- **Families (primary):** `overload` at the **shipped atom budget** (the 478× anchor — the only cell in
  the whole gym where the store actually works), `aggregate`, `manifold`. `recency` enters **only after
  D4 clears it**. Declare any family you drop and why; a dropped family is a NOT-RUN, never a null.
- **Arms:** `endpoint_write` (the control = today's objective) · `traj_write` · `path_write` ·
  `traj+path`. Coefficient grid pre-registered per §1(ii): **≥3 non-zero values spanning ≥2 decades,
  plus the perturbing liveness anchor, plus the zero point** (the D1 regression gate).
- **Seeds: {0,1,2} on every cell that votes in the gate.** Sample sd (`ddof=1`), `SE = sd/√3`, and
  report `dividend ± SE`; "clears 0 beyond 2 SE" is the gate's arithmetic, applied by the Hub.
- Every cell: 4 launders + the +0 B substitute + the two-sided byte ledger (`matched` flag) + the
  13 monitors + `gate_admissible`.

### D4 — Rider A (charter §A5): the recency-family diagnostic — **diagnose before concluding**
The gym measured CLU **0.3019 ± 0.0679** vs its own **blank store 0.3065**, both **below the 2-way
chance of 0.5**. That smells like a harness defect, not a null. Find out which. If it is a defect, fix
it and **re-baseline the `endpoint_write` control on the fixed family** so both arms see the same
family; report the pre-fix numbers as the defect's evidence. If it is not a defect, say so with the
evidence and the family stays out of the gate.

### D5 — Rider B (charter §A5): the longer-write-budget arm — **pillar (a) was never actually tested**
Every sub-shipped-budget gym cell ran the same **300** write steps as the shipped one, and the gym
measured `final write loss` **0.20–0.24** and `λ_min` **−0.21…−1.20** there: *"beyond-capacity
compression does not present as graceful read degradation; it presents as write failure."* Run the
declared longer-write arm (e.g. 900/1800 steps, pre-registered) at ≥2 sub-shipped atom budgets and
report the **curve** `decode / λ_min / final_loss` vs write steps. This separates *"the family has no
dividend"* from *"the store was never written"* — and it is the only honest test of opening (a) at this
weight class.

### D6 — Rider C: close spike R-4 (one flag, cheap)
`AttentionPsi` is implemented and never run. The DeepSets ψ read `q0_only` at **0.129 vs chance 0.125**
and refuted both the harness's and the spike's leak predictions; the untested hypothesis is that a
**pooled** read-out dilutes `q₀` to 1 of 150 points while an **attention** ψ can select it. Run
`--family attention` through the trajectory launder and settle it. A leak here would invalidate
trajectory-ψ numbers, so run it **before** D3's headline cells, not after.

## 3. Compute
**Priority P0** (with `ssb-shell-atoms`). Budget: **≤6 h wall-clock** of measured runs; **hard stop and
report at 10 h**. Smoke everything with `--quick` first; JAX cold start here is ~20 min (protocol §4).
⚠ **The real compute risk is the trajectory term inside the write loop** (a K-step rollout per write
step multiplies V-evals by K). Pre-register the rollout length and stride; if the trajectory-term write
exceeds **10×** the endpoint write's per-item wall-clock, report it and reduce the rollout, declaring
the reduction. The trajectory ψ costs **17.1×** the point ψ per training step (§A4.4 accepted this) —
fit ψ on **precomputed** reads wherever the arm allows, and say when you did.

## 4. ⛔ Two blocking preconditions (they exist because C2W1's Stage-0 gate saved the program once)
- **(P-A) Write-term liveness — the Stage-0 analogue, and it BLOCKS D3's verdict.** Before any race
  cell is scored, show that the trajectory-written store's trajectory carries **more decodable
  information than the endpoint-written store's**, against a **capacity-matched** baseline
  (`endpoints = [q₀, q_addr, q*, p*]`, never `q0_only`) **and** against the **blank-store probe
  control** (spike §4.3: 31–63 % of the only replicating v0 effect was reproduced by a store with
  nothing in it). Register the bar and the direction in PREREG. **If the term is inert at every
  registered coefficient, that is the finding and it votes in the gate; if it is inert at one setting
  only, sweep before concluding.** ⚠ And it only votes if the grid carried the **perturbing anchor** of
  §1(ii) — otherwise the correct report is *"under-powered grid,"* not *"the term does nothing."*
- **(P-B) Write convergence.** Every gate-voting cell reports `final_write_loss` and per-site `λ_min`;
  `gate_admissible=false` if the write did not converge. Do not silently average an unwritten store
  into a family verdict. **Excluded cells are reported with their reason; a family with zero admissible
  cells gets the D5 escalation once, then ABSTAINS; admissible-cell coverage per family is reported at
  the top of your results** (§1(i)).

## 5. PREREG (`.claude/outputs/traj-write-objective/PREREG.md`, before ANY measured run)
Register, with numbers and their derivation: the exact form of both objective terms and their
coefficient grids · the liveness bar and its baseline · the predicted `λ_min` at the ridge item under
`λ_path` (point estimate + range; today it is **−0.5946**) · the predicted per-family dividend (point +
range) for each arm · the predicted trajectory-launder values (`q0_only` at chance 0.125; blank at
chance — the C2W1 measurement was 0.129/0.148, so a prediction of a leak needs a reason) · the recency
diagnostic's competing hypotheses · the longer-write curve's predicted knee · the wall-clock budget and
its falsifier. **A prediction that survives is evidence; one that fails is a finding.**

## 6. Falsifiers
- ⛔ **The task's hard falsifier:** the coefficient-zero regression gate fails (the "off" objective is
  not bit-identical) ⇒ every downstream number is uninterpretable; stop and report.
- ⛔ **The trajectory launder fires** (a ψ scores above the blank/`q0_only` bar) ⇒ the ψ is reading
  `φ(x)`, not the store; **no ψ number in the report is quotable** until it is re-run store-relative.
- ◐ **Liveness fails at every registered coefficient** ⇒ the write cannot be made to put information in
  the trajectory *at this weight class*; report as the headline and it votes ≤0 in the gate.
- **Does NOT falsify:** dividend ≤ 0 with liveness passed (that is the gate's own branch and a result);
  losing to a classical method on a metric-native protocol (standing theorem, not news); `matched=False`
  bytes; the 17.1× ψ cost; the ridge term helping `λ_min` but not the read (that is informative).

## 7. File ownership (the zero-conflict standing practice — six branches, two waves, zero conflicts)
**YOURS (sole owner this wave):** `chlu/core/clu_system.py` · `chlu/training/train_memory.py` ·
`chlu/eval/dividend.py` (**append-only, below the frozen C2W1 surface**) · `chlu/eval/race.py` (new) ·
`chlu/experiments/memory_gym.py` · `chlu/experiments/exp_memory_gym.py` ·
`chlu/experiments/exp_traj_write.py` (new) · `chlu/cli/experiment_cmd.py` (**the one shared file —
sole ownership is yours; the other two engineers run by module invocation**) ·
`tests/test_memory_gym.py`, `tests/test_clu_system.py`, and your own new test modules.
**READ-ONLY — import, never edit; STOP and report if you need a change:**
`chlu/core/psi_readout.py`, `chlu/core/monitors.py`, `chlu/core/implicit_grad.py` (all
`phi-particle-head`'s) · `chlu/core/shell_atoms.py` (`ssb-shell-atoms`'s) ·
`chlu/experiments/exp_trajectory_read.py` (C2W1's; you may import `differentiable_read`) ·
`chlu/config.py` (**C2 config objects live in the C2-owned module that uses them**) ·
`chlu/core/{memory_potentials,controller,placement,admission,integrators}.py`.
⭐ **C1 files may now be edited** (C1 is retired, zero worktrees) **but ADDITIVELY ONLY, behind
default-off flags, with a bit-identical-shipped-behaviour regression test** — the C1W27 `payload_gate`
precedent. `train_memory.py` is exactly this case.

## 8. ⛔ Never-quote (inherited; violating these sends the report back)
Any cell as a **byte-matched** dividend (`ratio ≥ 2.20×` is architectural — gym PREREG-B1) · monitor
#2's `D` as a progress signal (the largest `D` = 0.931 had the **worst** dividend, −0.875) ·
"the trajectory axis knob is inert" (the correct statement is **"nothing consumes the buffer"**) ·
`k* ≈ 269` without *"of `∂q_N/∂θ`"* (for a whole-window ψ the error is **0.680 and flat in k**) ·
`sep/2` as a certified inradius (it fails by up to **7.44×** where `U → 0`) · "the harness's leak goes
live once ψ sees the address" (**refuted**) · "certified"/"exact deletion" unqualified · the endpoint
of any curve without the curve.

## 9. Output
`.claude/outputs/traj-write-objective.md` in protocol §5 format, with the flag-provenance table, the
PREREG scorecard, the reconciliation list in the **first 10 lines** if you have one, and the race-card
JSON + figures under `.claude/outputs/traj-write-objective/`. **Report to the Hub, not the Head.**
Branch left unmerged for Hub review; never push `origin`.
