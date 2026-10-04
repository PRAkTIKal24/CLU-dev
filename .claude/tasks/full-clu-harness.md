# full-clu-harness — the synthesis harness: learned `V_θ` store · staged levers · 13 loud monitors · controller v0

**Campaign 2, wave C2W1. Agent:** experiment-engineer. **Worktree MANDATORY** (protocol §3.2).
Base local `main`. Branch `agent/experiment-engineer/full-clu-harness`.
**⭐ THE WAVE'S SPINE.** Charter §6.1. Two other C2W1 tasks (`memory-gym-v0`, `trainability-spike`)
build directly on the API you land — see §API FREEZE, which is your **first commit**, not your last.

Read first: `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-intervention.md`
(**§4 what "full CLU" means · §5 the 13 modes · §8 prohibitions — all binding**),
`.claude/advisor-head-c2-charter.md` (§2.1, §3, §6.1).

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form)
- **Dial (C2 form):** none — **instrument**. This task builds the harness; it makes **no performance
  claim** and enters no leaderboard. Its acceptance is *"does not collapse,"* not *"wins."*
- **Laundering control:** implemented, not claimed. You must build the **dividend switch** —
  `(full CLU) − (its own settle-deleted / matched-bytes launder)` on the same harness, same bytes,
  same φ (charter §2.1) — and report it as an instrument reading. ⭐ **A dividend of ≈0 or negative at
  v0 is the expected, honest starting line and falsifies nothing.** Also build the **same-keys null**
  and the **blank/empty-store control**; both are harness-native, not optional add-ons.
- **Falsifies the acceptance criterion:** the full system cannot complete a nontrivial write/read
  stream without at least one monitor tripping, **and** the trip cannot be cleared by returning the
  responsible lever to its known productive band. (Monitors that trip *and are cleared* are not
  failures — they are the ablation table the reviewers will demand, charter §3.1.)
- **⛔ THE HARD FALSIFIER — the one that matters most.** If the *only* configuration in which no
  monitor trips is the degenerate one — explicit per-item arrays, engineered well separation,
  settled-point-only read — then you have re-derived w26's degenerate configuration and
  **intervention §8.2 has fired**. That is a **finding**, and you report it as the headline, loudly.
  Do **not** quietly settle there to make the acceptance criterion pass. Moving *toward* the lookup
  configuration to obtain a clean number is forbidden.
- **Does NOT falsify:** a ≤0 dividend at v0 · losing to any baseline (no baseline is run here) ·
  slow wall-clock at v0 · monitors #13 and #7 not being runtime trips (see the table).

---

## What "full CLU" means here (intervention §4 — the config that must be simultaneously live)
- **Write:** learned `V_θ` holding the items (**not explicit arrays** — this is the post-w26
  unclamping, and it is the point of the task), derived addressing, admission policy, per-item
  lifetimes, local/masked write, **permitted basin interaction**.
- **Read:** learned `φ` in, two-phase relaxation, mass as selector, **trajectory *and* settled point
  available to the read-out**, confidence-gated retry. (Learned `ψ` and the trajectory read are
  `trainability-spike`'s to build — you expose the **hook**, per §API FREEZE, and ship a handcrafted
  ψ as the v0 default.)
- **Structure:** the causal limit as a real constraint. Flat directions / trash regions / wormholes are
  **fixes-as-ammunition** (charter §3.3) — wire the hooks, deploy nothing pre-emptively.
- **Control:** controller v0 over the designed verb set {admit, place, evict, decay, route, retry, stop}.

**Staged activation, not big-bang (charter §3.1).** Every lever starts in its known productive band —
26 waves are the map, and the bands are in the monitor table below. Free levers **one at a time inside
the running full system**. Isolation is permitted only as a diagnostic *within* the full system, never
as the experiment (§8.1). Record the activation order you used; it is a reported artifact.

---

## ⭐ API FREEZE — your first commit, before any science
Two other engineers branch off this. Land, as commit #1 on your branch, the **public surface only**
(signatures + docstrings + `NotImplementedError` bodies are fine), then tell the Hub it is frozen:

- `chlu/core/clu_system.py` — `CluSystem` (assemble from a store, a φ, a ψ, a controller, a monitor
  registry); `CluSystemConfig`; `write_stream(...)`, `read(...)`, `consolidate(...)`.
  ⭐ **`read()` must return the trajectory, not only the settled point** — a strided trajectory buffer
  plus `q*`, so a trajectory read-out is a *configuration*, not a rewrite. This single decision is what
  makes pillar 1 (expressive latents) testable at all; it has never existed in 26 waves.
- `chlu/core/monitors.py` — `Monitor` protocol (`observe(ctx) -> MonitorReading`), a `MonitorRegistry`,
  and the 13 concrete monitors. Every reading carries `{name, value, band, tripped, cost_ms}`.
- `chlu/core/clu_controller.py` — `CluControllerV0` with the seven designed verbs, each a method with a
  designed guard it may not violate and a free parameter a policy may set (charter §3.2).
- `chlu/eval/dividend.py` — `dividend(full, launder)` + the three harness-native controls
  (settle-deleted launder · same-keys null · blank/empty store), with the byte-accounting function.
  ⚠ Co-owned with `memory-gym-v0`: **you land the signatures, they land the gym-side callers.**

**Hook for `trainability-spike`:** `CluSystem` takes `psi: Callable[[Trajectory, State], Array]`, and
the settle exposes the fixed-point residual so an implicit/DEQ gradient can attach. Ship a handcrafted
ψ (settled-point linear read) as the v0 default; do not implement a learned ψ yourself.

---

## ⭐ THE 13 MONITORS — Hub's PROVISIONAL table (implement this now)
`controller-doctrine` (theorist, this wave) is deriving the authoritative version and will **confirm /
sharpen / replace** each row. Implement this table now so you are not blocked; expect a diff at review.
**Every monitor fails LOUDLY** — a trip is logged, timestamped, attributed to a lever, and reported.
**Monitors are guards, never losses.** Do not add any monitor quantity to an objective (that is exactly
how #6 and #8 happen).

| # | mode | provisional invariant + trip predicate | band / source |
|---|---|---|---|
| 1 | overdamping → "the last observation" | `corr(q*, q_launch)` over the batch; **trip > 0.90** | C17-3 measured 0.97; N85 (γ won by turning the physics off) |
| 2 | settle → arg-min | **the same-keys launder run inline**: trip if settle-deleted read ≥ full read on the task metric | w26: launder beat CLU 6/6. ⭐ **this monitor IS the dividend** |
| 3 | vacuous gate | gate fire-rate over the stream; **trip if refusals ∈ {0, all}**, or if `min_separation` never crosses `d_safe` | N74 (spacing 1.414 vs d_safe 1.10); N91 — the *address space* was binding, not the controller |
| 4 | blank controls passing | blank/empty-store read on every reported metric; **trip if blank ≥ chance + 3σ**, or if blank hits ceiling | N68 (blanks 0.992–1.000); w26 `tol` vacuous at m>1 (blank 1.0000) |
| 5 | learned addressing dies | address-acquisition rate = fraction of queries landing in the intended basin (`basin` vs `strict`); **trip at ≈ chance** | w19 0/18 (4.2%). ⭐ w26: the annealed-read gain **is** address acquisition (`readonly` = baseline to 4 dp) |
| 6 | objective/goal divergence | rolling corr(write-loss ↓, retrieval-acc ↑); **trip if write loss → 0 while retrieval flat/declining** | w25/w26 |
| 7 | mass stores nothing | **a gauge TEST, not a runtime monitor**: perturb `(M, V, p₀)` along the gauge orbit, assert read invariance | Prop F1 ×3, verified 6.2e-16. Runs in `pytest`, not the stream |
| 8 | learning erases design | the C1–C5 / N1–N4 certificates (injectivity · margin ≥ κσ · sub-barrier `E(a_i) < h_i` · payload separation) re-checked each consolidation; **trip on any certificate falling below its admission threshold** | w20; `clu-controller-spec` §2 (κ≈5 ⇒ 99%, Gaussian-tail law within 0.021) |
| 9 | payload-dependent lifetimes | `corr(retention, \|a_i\|)` per consolidation; **trip if \|corr\| > 0.30** | w25 r=−0.85; N108. ⚠ the fix (option (d) gated stiffness) is **C1W27's** — C2 monitors only, does not build |
| 10 | degenerate axes / silent knobs | **knob-liveness sweep at harness start**: every declared dial must move ≥1 reported metric by > noise; **a dead axis fails at startup** | N19/N58; read-mode axis dead at `clu_steps=1` |
| 11 | reach failure | ⭐ **the saddle criterion on `L=√(\|c\|²+a²)`** evaluated per item at write time; trip on items predicted unreachable | `readout-channel-theory`: verified 31/32 on the trained shipped `V`, **zero free parameters**. Reach is logarithmically un-buyable (κ 4→5 = 55× depth) |
| 12 | starve-and-overwrite | per-item atom allocation + fidelity of **earlier** items after each later write; **trip if drop exceeds the C3 first-order bound** `‖∇δV(q*)‖/λ_min` | w26; C3 bound matched at median ratio 1.0002 |
| 13 | under-trained artefacts | **a provenance field, not a trip** — every diagnostic carries its epoch count; annotate below the maturity threshold | N94 (standing caveat: `<40`-epoch diagnostics are not properties of the shipped model) |

For each monitor also report its **false-trip mode** (the benign situation that fires it) — an
un-characterised monitor gets disabled by the next engineer, and then it is not a guard.

---

## Acceptance criterion (charter §6.1, verbatim in intent)
**The full system runs a nontrivial write/read stream without tripping a single silent-collapse mode.**
"Nontrivial" = a streamed sequence of writes and interleaved reads with real capacity pressure and at
least one deletion demand, on a learned `V_θ` store, with the levers in §4 simultaneously live.
**Every monitor's trip-state is a reported artifact** — including the ones that never fired (a monitor
that never fires on any configuration you ran is *untested*, and must be labelled as such, not as green).

## Compute (declare, do not silently drop)
⚠ **≤3 engineer worktrees across BOTH campaigns** (w26 thermal incident: 4 worktrees on 8 cores drove
load to 575 and inflated wall-clock 3–6×; macOS on Apple Silicon exposes **no manual fan control**).
The C1W27 Hub is running in parallel. **Do not launch background sweeps without telling the Hub.**
JAX cold-start here is ~20+ min — budget it, keep the session warm, use `--quick` for smokes, and do
not mistake a slow import for a hang. Declare your compute order in `PREREG.md` and report anything you
could not reach as **NOT RUN**, never as a null.
**⚠ Liveness note for whoever checks on you:** `PPID=1` on background jobs means the harness detached
them, **not** that the agent died (w26 precedent — verify by output mtimes + worktree state).

## File ownership (zero-conflict discipline — standing practice since w26)
**You own (all NEW files):** `chlu/core/clu_system.py` · `chlu/core/monitors.py` ·
`chlu/core/clu_controller.py` · `chlu/experiments/exp_clu_system.py` · `chlu/eval/dividend.py`
(signatures; `memory-gym-v0` lands the gym-side callers) · `tests/test_clu_system.py` ·
`tests/test_monitors.py` · `tests/test_clu_controller.py`.

⛔ **READ-ONLY — C1W27 territory, do not edit even by one line:**
`chlu/core/memory_potentials.py` (C1W27's d-sweep at m=4 owns the atom-init/width knobs and
`AtomStorePotential`; option (d) gated-stiffness also lands here) · `chlu/core/controller.py` and
`AtomStorePotential.evict` (C1W27's P2 waitlist) · `chlu/core/placement.py` · `chlu/core/admission.py` ·
`chlu/experiments/exp_designed_mechanism.py` · `exp_cl_entry.py` · `cl_baselines.py` · `exp_phi_stream.py` ·
`exp_write_ceiling.py` · `exp_sharded_store.py` · the mia harness ·
⚠ **`chlu/config.py`** — C1W27 owns **two separate blocks** in this file this wave
(`ExperimentDesignedMechanismConfig` incl. `pass_metric`, and the store/controller config fields).
**This repo is config-driven and the reflex is to add your config class here — do not.**
`CluSystemConfig` and every other C2 config object lives in the C2-owned module that uses it
(`clu_system.py`). If C2 genuinely needs a CLI-registered config, **STOP and report to the Hub** — a
`config.py` edit is the single most likely source of a cross-campaign merge conflict this wave.
**Consume them by import, wrap them, subclass them — never edit them.** If you believe a change to a
read-only file is unavoidable, **STOP and report to the Hub**; do not guess and do not work around it by
copying the file. This split is what produced zero merge conflicts across four engineer branches in w26.

⛔ **Also do not touch:** `chlu/core/psi_readout.py`, `chlu/core/implicit_grad.py` (`trainability-spike`)
· `chlu/experiments/memory_gym.py`, `exp_memory_gym.py` (`memory-gym-v0`).

## Deliverable
`PREREG.md` at `.claude/outputs/full-clu-harness/PREREG.md` **before you run anything measured** —
your lever-activation order, your predicted trip-set (which monitors you expect to fire, in which
order), your predicted v0 dividend sign, and your compute order. *A pre-registered prediction that
survives is evidence; one that fails is a finding. An un-pre-registered agreement is neither.*
Report at `.claude/outputs/full-clu-harness.md`, protocol §5 format, **flag-provenance table on every
quantitative result**, PREREG scorecard, reconciliation list in the **first 10 lines**.
Full `uv run pytest -q` green, `ruff` clean, atomic `[experiment-engineer]`-prefixed commits.
**Do not push. Do not merge. Leave the branch for Hub review.**

⛔ **Do-not-quote, carried:** any `tol`-metric number at m>1 · `K_learned` at `pscale ≠ 1` without the
payload-noise condition · "the write operator is the ceiling" · width-lock-as-cause · "~32,
d-independent" as settled · the √2 / `d^1.62` exponent · "capacity multiplies by sharding" ·
"0.99985" without its load · "certified"/"unlearning"/"deletion-compliant"/unqualified "exact deletion".
**Standing:** laundering control on every performance claim · multi-seed before any paper number ·
quote the curve, not the endpoint · `git -C <worktree>` always.
</content>
