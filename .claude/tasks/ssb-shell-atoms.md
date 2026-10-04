# ssb-shell-atoms — Route 2: designed degeneracy (shell atoms + the pseudo-Goldstone tilt dial)

**Campaign 2, wave C2W2. Agent:** experiment-engineer. **Worktree MANDATORY.**
Base = **`traj-write-objective`'s RACE+SEAM FREEZE commit** (§Dependency). Branch
`agent/experiment-engineer/ssb-shell-atoms`. Charter Addendum-1 **§A5/C2W2 task 2**, implementing the
binding design rulings **§A4.1 (SSB — two routes, raced)** and **§A4.2 (never exactly flat — the
pseudo-Goldstone ruling)**. Bound by intervention **§8** and **§5**.

**Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c2-charter.md` **IN FULL incl.
ADDENDUM 1 (§A4.1, §A4.2 are your specification, verbatim)**, `.claude/advisor-head-intervention.md`
(**§3.4 the V2 physics — never tested; §5; §8**), `.claude/tasks/traj-write-objective.md` (the race card
you emit into), and `.claude/outputs/memory-gym-v0.md` **§3.5** (your blocker, measured) +
`.claude/outputs/trainability-spike-theory.md` **§Q2 (project-and-transport, the Ward identity) and
§Q2.4 (the pseudo-flat band table)** — the theory half of your task is already banked; do not re-derive it.

⛔ **REGISTRY LAG (Head parked the curator pass, 2026-07-30).** `negative_results.md` (N122),
`claims_matrix.md` (v2.5), `research_roadmap.md` (v0.9), the ledger (⟲ w26) and the primers are **two
campaigns behind**: **C1W27's and ALL of C2W1's results are in no registry.** Quote them **only** from
`.claude/outputs/*` and the `[C1W27]`/`[C2W1]` §10 entries — never from the matrix/registry/roadmap, and
never read a registry's silence as evidence that a result does not exist.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **pillar 1 (manifold-valued latents) via pillar-(d) storage.** KPI = the
  **dividend** on the shared race card; no accuracy number without its dividend, no dividend without
  its bytes.
- **Laundering control:** the same four as Route 1 — settle-deleted launder · same-keys null · blank
  store · **trajectory launder on every ψ that can see the address block (mandatory)** — plus the
  family's **+0 B substitute**. ⚠ For the manifold family the +0 B substitute (`echo`) scored **1.0000**
  against CLU's **−0.180** at v0: you are racing a substitute that is currently perfect, and the
  honest framing is *"can the store express a manifold at all"*, not *"can it beat echo"*.
- **Falsifies:** §6. **Does NOT falsify:** a ≤0 dividend with the r=0 gate passed and the tilt dial
  demonstrably live; `matched=False` bytes; needing a nonzero tilt (that is the ruling, not a defect).

---

## 0. What you inherit
There are **no flat directions** in the shipped store: at **14 of 18 sites** `λ_min = 0.0846–0.1000
≈ 2α = 0.10` — the "unconstrained" spectator axis carries **exactly the confinement's curvature**, so
the settle drives it to zero (`manifold_r2 = −0.180 ± 0.171`). And the obvious fix fails: a multi-target
**ridge write produces a SADDLE, not a valley** — `λ_min = −0.5946` with spectator participation
**1.000**, because `write_loss` minimises `V` at each target independently and never constrains the
connecting path. Route 1 attacks that from the **objective** side (a path/equal-depth term). **You
attack it from the BASIS side**: build a store element whose degeneracy is *designed in*, then learn
where to put it — the w20 doctrine ("designed mechanism + learned use beats learned-everything")
applied to symmetry.

## 1. ⚖ THE C2W2 GATE (binding decision rule — VERBATIM from charter §A5)

> **⚖ C2W2 GATE (binding decision rule):** if writes that *explicitly ask* trajectories/paths to carry
> information still give dividend ≤ 0 on every family (multi-seed, substitute-audited, both routes)
> ⇒ **B′ activates** and C2W3 re-scopes toward the audit paper. If any family clears 0 beyond 2 SE ⇒
> C2W3 proceeds as planned. Either way the direction is re-priced, not closed.

**You are the "both routes" clause.** Same race card, same schema, same seeds, same φ, or the race is
not a race. The **Head-ratified rulings on gate application** (2026-07-30) apply to your cells
identically — read `traj-write-objective.md` §1, which is the canonical statement. In your terms:
- **(i) Write convergence.** A cell votes only if the write converged (`final_write_loss`, per-site
  `λ_min`). **Excluded cells are reported with their reason**; a family with **zero** admissible cells
  gets **one** bounded budget escalation (a declared longer write) and then **ABSTAINS** — it neither
  blocks B′ nor supports "proceed". ⛔ **Report admissible-cell coverage per family as a first-class
  number at the top of your results.** Admissibility filtering must never quietly gut coverage until the
  gate cannot fire.
- **(ii) Dial liveness — your `ε` grid must SPAN to where the tilt is live.** Declared before any run;
  **≥3 values spanning ≥2 decades**, and it **must include at least one `ε` that VISIBLY PERTURBS the
  store — even destructively** (an `ε ≈ λ_massive` that measurably costs decode or collapses the
  hierarchy is a perfectly good anchor, and is the natural top of your range). ⛔ *"A term that never
  moves anything at any tested setting hasn't been asked; it's been whispered at"* (Head, verbatim).
  Without that anchor an inert-everywhere tilt is **not** a legitimate ≤0 vote — it is an under-powered
  grid, and you report it as one. The **r=0 point is the mandatory zero** and is the regression gate.
- **(iii) The "weak proceed" caveat** (Head ruling, now **charter §A6**, pre-registered before
  adjudication): a family clearing 2 SE **but losing to its +0 B substitute** is a **weak proceed** —
  C2W3 proceeds but cannot take it as the sole headline basis, and diagnosing/beating the substitute
  deficit becomes a mandatory C2W3 item. ⇒ **emit the signed +0 B substitute margin per family.**
  ⚠ This will very likely bite the `manifold` family, whose `echo` substitute sits at **1.0000** by
  construction — so state the substitute margin there plainly rather than letting a positive dividend
  read as a clean win.

## 2. Deliverables

### D1 — The shell atom (`chlu/core/shell_atoms.py`, new module — no edits to `memory_potentials.py`)
Per charter §A4.1, verbatim:

    V_j = −A_j · exp( −(‖q − c_j‖ − r_j)² / (2 s_j²) )

with **learnable `softplus` radius `r_j`**. Registered into the store through the
`store_potential_factory` import-path hook that `traj-write-objective` lands in the FREEZE commit —
**you edit nothing you do not own.**
- ⛔ **THE r=0 REGRESSION GATE IS MANDATORY AND BLOCKING** (charter §A4.1: *"r=0 ⇒ exact Gaussian
  reduction — a mandatory regression gate"*). At `r_j = 0` the shell atom must reduce to the shipped
  Gaussian atom **exactly**: assert bit-identical `V`, `∇V`, `Hess V` and a bit-identical written store
  against `main`'s atom family, as a **test**, not as a claim in prose. **If this gate does not pass,
  no science cell runs and you report that.**
- The store must remain a **learned `V_θ`** with no per-item arrays in the read path (intervention
  §8.2). Report the byte ledger: shells add `r_j` per atom — **declare the extra bytes and put them in
  the ledger on every arm**, including the launder's.

### D2 — The tilt dial (charter §A4.2, the pseudo-Goldstone ruling — this is the load-bearing half)
An optional **learned low-rank tilt** as the *explicit-symmetry-breaking* dial, with a scalar strength
`ε`. The ruling is binding and it is a design constraint, not a preference:
> **Designed flat directions carry the payload (position-on-shell) and are NEVER zeroed. Spurious flat
> axes may be projected out, participation-ratio-gated. A small tilt `ε` makes `λ_min = ε > 0`.**

Three things `ε` must be shown to do, each pre-registered with a number:
1. **Conditioning:** `λ_min(Hess V)` at a written site tracks `ε` (predict the constant of
   proportionality). This is what restores implicit-gradient conditioning and **terminates settles** —
   at exact flatness the settle does not terminate along the orbit (theory §Q1.1b/§Q2).
2. **The curvature hierarchy:** `λ_soft ≈ ε ≪ λ_massive`. Report the spectrum, not one eigenvalue, and
   report the **participation ratio** of the soft eigenvector on the designed shell coordinate — the
   gym's ridge failure was diagnosed exactly this way (participation 1.000 on an *unstable* mode).
   ⚠ Theory §Q2.4 says the intermediate band is the dangerous one (`cond(H)` 7.69 → 9.6e7 as `b` → 0)
   and its recommendation is *"be exactly flat architecturally, or comfortably massive"*; the charter
   overrules the "exactly flat" half **for the payload direction** and asks for `ε > 0`. Your job is to
   **measure where that trade sits on a real store** — the first data on it.
3. ⭐ **The lifetime dial:** `ε` is the **manifold-payload lifetime dial** — drift/quantisation
   timescale **∝ 1/ε**. Measure the exponent (pre-register it; the charter predicts −1). This is the
   only place in C2W2 where a *new capability claim* could originate, and it is per-atom settable.

### D3 — The race card (the gate's other half)
Same families, same 3 seeds `{0,1,2}`, same metrics, same schema as Route 1 (`chlu/eval/race.py`,
read-only to you — emit into it, do not edit it).
- **Arms:** `gauss` (control = today's store) · `shell_r0` (**the regression gate, must equal `gauss`**)
  · `shell` (learned `r_j`, no tilt) · `shell+tilt(ε)` over a pre-registered `ε` grid (≥3 values,
  spanning the hierarchy from `ε ≪ λ_massive` to `ε ≈ λ_massive`).
- **Families:** `manifold` is your crux (it is where a shell can express something a point cannot) —
  but you must also run `overload` at the **shipped atom budget** and `aggregate`, because a store
  change that helps manifolds and destroys addressing is a loss, and C2W1 measured exactly that kind of
  side-effect once already (C1W27's m=4: a read-side change that widened wells 2.3–2.7× and collapsed
  the address at d=6).
- Every cell: 4 launders + the +0 B substitute + two-sided byte ledger + the 13 monitors +
  `gate_admissible` (write converged, dial live).

### D4 — The combined cell (the 2×2), conditional and declared
Route 1 × Route 2 (shell atoms **under** the trajectory/path write objective) is the interaction the
gate would most like to see. **If** `traj-write-objective`'s write-objective commit has landed on `main`
before your final race run, rebase and run the `manifold` and `overload` 2×2 cells at 3 seeds. **If it
has not, declare the 2×2 a NOT-RUN with the reason** — the Hub runs it at integration. ⛔ Never report a
NOT-RUN as a null.

## 3. Compute
**Priority P0** (with `traj-write-objective`), but you **start on the FREEZE commit, not before** — take
the third worktree slot when it lands. Budget: **≤6 h** of measured runs, **hard stop and report at
10 h**. Smoke with `--quick` first. If the shell atom's `∇V`/`Hess V` cost more than **2×** the Gaussian
atom's per V-eval, report it in the ledger — it travels with every claim, like the ψ's 17.1×.

## 4. ⛔ Blocking preconditions
- **(P-A) The r=0 gate** (D1). Blocking, no exceptions.
- **(P-B) Dial liveness.** Before any `ε` claim: sweep `ε` and show a **declared observable moves by
  > 3× its noise** (monitor #10's own bar). A dial that nothing responds to is a dead axis, and C2W1
  spent a whole task discovering that the correct diagnosis of a flat sweep is usually *"nothing
  consumes it"*, not *"the knob is inert"*.
- **(P-C) Write convergence** on every gate-voting cell (`final_write_loss`, per-site `λ_min`).

## 5. PREREG (`.claude/outputs/ssb-shell-atoms/PREREG.md`, before ANY measured run)
Register with numbers and derivations: the r=0 identity you will assert and its tolerance (bit-identical
is the target — say so) · `λ_min(ε)` point estimate + range · the predicted curvature hierarchy and
soft-mode participation ratio · the **drift-timescale exponent in `1/ε`** (charter predicts −1; register
your own) · per-family dividend point + range per arm · the byte overhead of `r_j` · the predicted
effect on `overload`/`aggregate` (a store change is allowed to cost something — say how much in advance).

## 6. Falsifiers
- ⛔ **r=0 does not reduce to the Gaussian** ⇒ implementation error; everything downstream is invalid.
- ⛔ **The tilt does not produce `λ_min ≈ ε > 0`** at any registered `ε` ⇒ §A4.2's mechanism is refuted
  on a real store; that is a **major finding** and must be reported as the headline, not buried.
- ⛔ **Shell atoms destroy addressing** (`overload`/`aggregate` decode falls outside your registered
  range) ⇒ the basis change is not free and the race is decided against Route 2 on those families —
  report both halves.
- **Does NOT falsify:** a ≤0 manifold dividend when the +0 B `echo` substitute is at 1.0000 by
  construction (say so, and quote the `r2` and `λ_min` evidence for whether the manifold is *expressed*
  at all — that is the scientific content here) · needing `ε > 0` · a byte overhead, declared.

## 7. File ownership
**YOURS (sole owner):** `chlu/core/shell_atoms.py` (new) · `chlu/experiments/exp_ssb_shell.py` (new) ·
`tests/test_shell_atoms.py` (new) + your own new test modules.
**READ-ONLY — import, never edit; STOP and report if you need a change:** `chlu/core/clu_system.py`,
`chlu/training/train_memory.py`, `chlu/eval/{dividend,race}.py`, `chlu/experiments/{memory_gym,
exp_memory_gym,exp_traj_write}.py`, `chlu/cli/experiment_cmd.py` (**all `traj-write-objective`'s — you
run by module invocation, no CLI hook**) · `chlu/core/{psi_readout,monitors,implicit_grad}.py`
(`phi-particle-head`'s) · `chlu/core/memory_potentials.py` · `chlu/config.py` · every other C1 file.
Your registration into the store goes through the **factory hook** in the FREEZE commit. If the hook is
insufficient, **STOP and report** — do not edit around it.

## 8. ⛔ Never-quote (inherited)
Any cell as a byte-matched dividend · monitor #2's `D` as a progress signal · `sep/2` as a certified
inradius · "certified" in any deletion sense · the shipped `b>0` ring register as evidence for the
manifold pillar (**theory R-2: it is 2.32× soft, inside N46's *emergent* band — it is not a flat
direction**) · "graceful degradation above capacity" as our discovery (Clark, PRE 2026) · any endpoint
without its curve.

## 9. Output
`.claude/outputs/ssb-shell-atoms.md`, protocol §5 format: flag-provenance table, PREREG scorecard,
reconciliation list in the **first 10 lines** if any, race-card JSON + spectra/figures under
`.claude/outputs/ssb-shell-atoms/`. Report to the Hub. Branch left unmerged; never push `origin`.
