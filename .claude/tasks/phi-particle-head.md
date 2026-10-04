# phi-particle-head — the read-in that parametrizes the particle (+ the owed monitor repairs)

**Campaign 2, wave C2W2. Agent:** experiment-engineer (**small** — the charter sizes it that way).
**Worktree MANDATORY.** Base local `main` (`233fd9e` at scoping) — **you do NOT gate on the FREEZE
commit**, launch immediately. Branch `agent/experiment-engineer/phi-particle-head`.
Charter Addendum-1 **§A5/C2W2 task 3**, implementing the binding design ruling **§A4.3 (φ policy,
including the `(d, atom-budget)` joint dial)** and — per Hub ruling — the **already-specified monitor
repairs** the theorist cannot land (see D4; the roster bars `physics-theorist` from production code).

**Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c2-charter.md` **incl. ADDENDUM 1
§A4.3 (your spec, verbatim) and §A2.2 (mass becomes live under trajectory reads — you test it)**,
`.claude/advisor-head-intervention.md` **§5** (the 13 modes) and **§8**,
`.claude/outputs/trainability-spike.md` **§3.1 (`∂L/∂φ` = 0 exactly) and §3.3**,
`.claude/outputs/controller-doctrine.md` **§7 (implementation requests I-1…I-12 — D4 is I-3/I-4/I-6/I-8/
I-9)**, `.claude/outputs/memory-gym-v0.md` **R2/R3** (the two monitor defects the gym measured).

⛔ **REGISTRY LAG (Head parked the curator pass, 2026-07-30).** `negative_results.md` (N122),
`claims_matrix.md` (v2.5), `research_roadmap.md` (v0.9), the ledger (⟲ w26) and the primers are **two
campaigns behind**: **C1W27's and ALL of C2W1's results are in no registry.** Quote them **only** from
`.claude/outputs/*` and the `[C1W27]`/`[C2W1]` §10 entries. ⚠ **This bites you specifically:** your D4
monitor repairs **re-score C2W1's trip counts**, and there is no curator to file the corrected rows this
wave — so report the before/after counts yourself, in your own output file, as a named deliverable.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none — instrument + infrastructure.** No performance claim, no leaderboard entry,
  no dividend claimed from this task. Everything you land is consumed by C2W2's race and by C2W3.
- **Laundering control:** built, and one is *measured*: the **trajectory launder** on any ψ/φ pair that
  can see the address block. **The identical-φ invariant is yours to enforce in code:** charter §A4.3 —
  *"identical φ for CLU / baselines / launder; φ params in the byte ledger, all arms."*
- **Falsifies:** §5. **Does NOT falsify:** a strong φ costing bytes (declare them) · gradients through
  the particle parameters being small (only *exactly zero* is the interesting outcome — see D3).

---

## 0. Why this task exists
The read-in has never been trained through the store and never will be through a settled point:
**`∂q*/∂q₀ = 0` exactly**, measured as `‖∂L/∂φ‖` = **0.0** (implicit) / **2.654e-9** (unroll) /
**6.421e-3** (trajectory), ratio **2.42e6**. So φ is only trainable through the trajectory channel, and
today's φ is a frozen PCA/identity — *weak φ is a measured failure mode* (the CIFAR null; `full_pca`
came in **under** the 5-dimensional `q0_only` baseline in the Stage-0 sweep, i.e. the registered feature
was under-powered). C2W3's factored store and C2W5's Track-2 entry both need a real read-in and a
particle that φ can actually parametrize. You build it and you gradcheck it — you do **not** chase a
dividend with it.

## 1. Deliverables

### D1 — Strong-φ plumbing (charter §A4.3)
A `φ` interface that admits **standard strong encoders** (small CNN/ResNet, RNN/transformer,
SSL-pretrained where the weight class allows) alongside the existing identity/PCA, with:
- ⭐ **the fairness invariant enforced in code, not in prose:** the *same* φ instance serves CLU, its
  launders and any baseline; a mismatch must **raise**, not warn. This is the invariant every C2W2
  number depends on.
- **φ parameters in the byte ledger on every arm** (`phi_id`, `phi_bytes` are fields in the C2W2 race
  card — `chlu/eval/race.py`, read-only to you; emit them).
- At least one strong encoder wired end-to-end and **smoke-run**; you are not required to train it to a
  benchmark number this wave, and you should not.

### D2 — The particle-parametrizing head (charter §A4.3, verbatim)
> *"φ's output head widens to parametrize the particle — launch `q₀` + per-particle mass + friction."*

Head emits `(q₀, log_mass, friction)` per query. Constraints: mass positive (softplus, the repo's
convention), friction inside the doctrine's band (monitor #1's productive band is **harness-specific** —
doctrine R2 measured the shipped read at `ρ_conv = 4.3e-7`, within **2.3×** of the trip edge; clamp to a
declared range and report where you sit). Ship default-off so shipped behaviour is unchanged.

### D3 — ⭐ The gradcheck, and the one real experiment in this task: **does the mass gauge dissolve?**
Charter §A2.2 states a consequence of the trajectory read that has never been tested:
> *"Prop F1's mass-gauge dissolves under trajectory reads (the endpoint is M-independent; the trajectory
> is not) — 'mass as selector' becomes live for the first time."*

This is cheap and decisive. Measure, on the real store, with a matched-parameter ψ:
`‖∂L/∂log_mass‖` and `‖∂L/∂friction‖` for **(a)** a settled-point ψ and **(b)** a trajectory ψ.
- **Pre-register the prediction:** the point arm is **exactly 0** by Prop Q1.1 (`Fix(T_θ) = {(q,0):
  ∇V = 0}` contains no `M`), the trajectory arm is nonzero; register the ratio's point estimate and its
  floor, and register a finite-difference cross-check tolerance (the spike hit **5.11e-10** implicit-vs-FD
  and **2.42e6** on the φ ratio — those are your calibration anchors).
- ⛔ **If the trajectory arm's mass gradient is also numerically zero, §A2.2 is REFUTED** and *"mass as
  selector"* stays dead. That is a finding of charter-amendment weight; report it as the headline.
- Also report the **full gradcheck**: `query → φ → settle → ψ → loss` with the new head live, implicit
  vs truncated-unroll agreement (spike's registered bars: implicit-vs-FD ≤ 1e-5, truncation `= ρ^k`
  ±5 %), and the wall-clock per step against the 30 s budget (spike measured 0.500 s point / 8.550 s
  trajectory).
- ⚠ **Carry spike R-2 into your design:** tail truncation enters through a `stop_gradient`, so
  `‖∂L/∂φ‖` is **exactly 0.0** at k = 0…270 and nonzero only at full backprop. **Truncation direction is
  load-bearing.** If your gradcheck uses truncation, say which direction, or you will measure a zero and
  mis-diagnose it.

### D4 — Rider (Hub ruling): land the monitor repairs that are ALREADY SPECIFIED
`chlu/core/monitors.py` is yours this wave. Land only what is already specified by a landed C2W1
deliverable — anything needing new theory belongs to `doctrine-repairs` and waits:
| repair | source | what to land |
|---|---|---|
| **#1** | doctrine **I-3**, R3 | trip on `ρ_conv = med‖∇V(q*)‖/‖∇V(q₀)‖` and `δ = med‖q*−q₀‖/sep`; **demote `corr(q*,q₀)` to a reported diagnostic** (healthy 0.973–0.978 vs unconverged 0.993–1.000 — the threshold was in the wrong decade) |
| **#9** | doctrine **I-4**, R4 | trip on the **effect size** `Δ_ret` (band ≤ 0.10), `corr` reported as direction only (`|corr|` exceeded 0.30 at *every* excursion tested — a monitor that is always on gets disabled) |
| **#6** | gym **R2** | a **dead-band**: `slope_loss < −eps and slope_acq <= 0`, `eps ~ 1e-9 × scale`. **29 of its 58 first-ever trips fired at slopes of −5.2e-17.** Re-score the C2W1 gym artifact with the fixed predicate and report how many of the 58 survive |
| **#2** | doctrine **I-6** | report `U` and `ρ_ex = D/U`; mark **INAPPLICABLE (not tripped)** when `U < 0.01` (verify it is already so; if yes, say so and move on) |
| **#10 tier (a)** | doctrine **I-8** | the **O(1) access-counting config proxy** — fail at startup on any declared-but-never-read field. C2W1 declared `knob_tier_a_implemented: false` rather than silently passing; close it. It catches N19/N20/N58 |
| **#7** | doctrine **I-7** | compare the **whole trajectory**, not the endpoint, and parameterise by `kinetic_mode`; the endpoint comparison passes **vacuously** once both runs settle (9.1e-2 → 3.6e-3 by doubling N alone) |
⛔ **NOT yours (theory owes them first):** #3's validity leg (gym R3: `corr` is **−0.99…+0.56 and
sign-unstable** on a learned `V_θ` — the fix needs a new predicate, not a threshold), the `sep/2`
inradius scoping, and the soft-certificate spec. Those are `doctrine-repairs`'.

#### ⛔ D4 FREEZE RULE (Head ruling, 2026-07-30 — binding, and it protects you)
**The rider is frozen at the six repairs in the table above.** If `doctrine-repairs` unblocks #3's
validity leg or the `sep/2` scoping **mid-wave**, those **queue for C2W3** — they are **not** added to
your task in flight, and you should decline them if offered. *Rationale, verbatim from the Head: "a
rider that grows while its host runs is how a 'small' task eats a worktree slot."* You are the small
task; stay small.

#### ⭐ D4 ACCEPTANCE (Head ruling — this converts "repaired" from an assertion into an artifact)
**Re-run the C2W1 shipped anchor and diff the monitor trip-states, before vs after your repairs:**
- **Every changed trip must map to a named repair** in the table above, one-to-one, stated in your report.
- **Everything else must be bit-identical.** An unexplained trip-state change is a regression, not a
  repair — stop and report it rather than rationalising it.
- The anchor is the C2W1 configuration whose artefacts are on disk (`overload/load1x_shipped`, the 478×
  cell; and the harness's `S0`/`R3` records) — diff against `.claude/outputs/{memory-gym-v0,
  full-clu-harness}/*.json`, not against a fresh baseline you generate.
- Report the before/after counts explicitly, including **how many of monitor #6's 58 first-ever trips
  survive the dead-band** (29 of them fired at slopes of −5.2e-17). There is no curator this wave, so
  this table is the only place the corrected rows will exist until C2W3 — make it citable.
⚠ **Monitors are guards, never losses.** No monitor quantity may enter any objective — that is exactly
how modes #6 and #8 are *caused* (doctrine §6).

## 2. Compute
**Priority P1.** Small: budget **≤3 h** of measured runs, hard stop and report at 6 h. Launch
immediately alongside `traj-write-objective`; you hold the second worktree until `ssb-shell-atoms`
needs the third, and the cap is **≤3 total across the machine** (w26 thermal incident, 8 cores).

## 3. The `(d, atom-budget)` joint dial — declare it, do not sweep it
Charter §A4.3 is binding and is a **declaration** requirement for this wave, not an experiment:
> *"φ's address-block dimension `d` is a capacity lever, not a free choice — banked law
> `K_learned(d) = min(2^d, write ceiling)`; but atom budget must co-scale (`min_atoms ∝ c^d`), bytes/well
> grow ∝ d (capacity-per-byte is the honest metric), and reach tightens as `σ√d` — so `(d, atom budget)`
> is a single declared joint dial in the byte ledger, launder included."*
⇒ Make `(d, n_atoms)` a **single named dial** in the config surface you touch, with the co-scaling rule
written down and asserted, and make the byte ledger print both. **Do not run a d-sweep this wave** (that
is C1W27's closed result and C2W3's business). ⛔ Never quote the d=8 designed wall as bracketed — it is
**lower-bounded** (K=2048 was never run).

## 4. PREREG (`.claude/outputs/phi-particle-head/PREREG.md`, before ANY measured run)
Register: the mass/friction gradient-ratio prediction (point + floor + the exact-zero prediction for the
point arm) and its FD cross-check tolerance · the gradcheck bars you inherit · the wall-clock budget and
falsifier · the predicted number of surviving #6 trips after the dead-band (of 58) · the byte cost of
each φ variant.

## 5. Falsifiers
- ⛔ **The trajectory arm's mass/friction gradient is numerically zero too** ⇒ §A2.2 refuted, "mass as
  selector" stays dead. Headline it.
- ⛔ **The identical-φ invariant cannot be enforced** (some arm structurally needs a different φ) ⇒ every
  C2W2 dividend inherits a confound; stop and report **before** the race cells run — this one is urgent.
- ⛔ Any monitor repair changes a **C2W1 trip verdict** in a direction that invalidates a reported C2W1
  result ⇒ report it as a reconciliation with the affected sites named (that is a finding, and the
  curator needs it).
- **Does NOT falsify:** a strong φ being slow or heavy (declare it); the head being default-off; small
  but nonzero particle gradients.

## 6. File ownership
**YOURS (sole owner):** `chlu/core/psi_readout.py` · `chlu/core/monitors.py` ·
`chlu/core/implicit_grad.py` · `chlu/experiments/exp_phi_particle.py` (new) ·
`tests/test_monitors.py`, `tests/test_psi_readout.py`, `tests/test_implicit_grad.py` + new test modules.
**READ-ONLY — import, never edit; STOP and report:** `chlu/core/clu_system.py`,
`chlu/training/train_memory.py`, `chlu/eval/{dividend,race}.py`, `chlu/experiments/{memory_gym,
exp_memory_gym,exp_traj_write}.py`, `chlu/cli/experiment_cmd.py` (**`traj-write-objective`'s — no CLI
hook for you; run by module invocation**) · `chlu/core/shell_atoms.py` (`ssb-shell-atoms`'s) ·
`chlu/config.py` · `chlu/experiments/exp_trajectory_read.py` (import only) · all C1 files.
⚠ **You own `psi_readout.py`; `traj-write-objective` uses your ψ classes UNMODIFIED.** If Route 1 needs
a ψ change it must come to the Hub, not to your file — protect the `matched_pair` bit-identity assertion.

## 7. ⛔ Never-quote (inherited)
`k* ≈ 269` without *"of `∂q_N/∂θ`"* · "the trajectory axis knob is inert" (say **"nothing consumes the
buffer"**) · `sep/2` as a certified inradius · monitor #2's `D` as a progress signal · "the leak goes
live once ψ sees the address" (refuted: `q0_only` **0.129** vs chance 0.125) · "~32, d-independent" ·
`K_designed(4) = 128` at m=1 · any cell as a byte-matched dividend.

## 8. Output
`.claude/outputs/phi-particle-head.md`, protocol §5 format with flag-provenance table, PREREG scorecard,
first-10-lines reconciliation list if any. Artefacts under `.claude/outputs/phi-particle-head/`.
Report to the Hub. Branch left unmerged; never push `origin`.
