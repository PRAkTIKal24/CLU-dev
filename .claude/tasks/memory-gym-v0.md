# memory-gym-v0 — Track 1: the internal memory gym, launder-native, dividend as the sole KPI

**Campaign 2, wave C2W1. Agent:** experiment-engineer (analyst hand-off at review).
**Worktree MANDATORY.** Base local `main`, **branched off `full-clu-harness`'s API-FREEZE commit**
(see §Dependency). Branch `agent/experiment-engineer/memory-gym-v0`. Charter §6.2, implementing §2.2
Track 1 and §2.1.

Read first: `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c2-charter.md`
(**§2.1 the dividend and its four structural openings · §2.2 Track 1**),
`.claude/advisor-head-intervention.md` (**§6 — the five admissibility criteria, all binding; §8.3/§8.4
prohibitions**), `.claude/tasks/full-clu-harness.md` (the API you build on), and **`.claude/outputs/full-clu-harness.md`
(the harness AS LANDED — it hands you three decisions and one hard problem)**.

---

## ⭐ C2W1 AMENDMENT (Hub, 2026-07-30) — the harness has landed; read this before PREREG
`full-clu-harness` is **merged to `main @ 4160cf7`** — branch off `main`, not off its branch. All three
harness-native controls are built and fire on every stage; the blank control **passed everywhere**
(0.125–0.167 against bars 0.249–0.328, monitor #4 never tripped). The v0 dividend came in at
**−0.094 (S0) / −0.083 (S3) / 0.000 (best config)** — the charter's expected honest starting line,
landing 0.006 from the harness's own pre-registered point estimate.

**⭐ RULING on the inherited semantics (harness recon R4, Hub confirms — this is now binding on you):**
`same_keys_null` = **same keys, permuted payloads** (address structure kept, content destroyed);
`settle_deleted_launder` = **arg-min over the store's own keys, returning the true payload**. These are
two different controls answering two different questions — *"is the address structure doing the work?"*
and *"is the settle doing anything the arg-min isn't?"* — and **both travel with every gym number**.
The signatures in `chlu/eval/dividend.py` are **frozen**; you land the gym-side callers only.

**⛔ THE HARD PROBLEM YOU INHERIT — BYTES ARE NOT MATCHED, BY 359×.** The harness measured
`V_θ` **57 344 B** + codebook 128 B = **57 472 B** against the launder's **160 B** — `matched=False`,
ratio **359.2×** at K=8 (478.7× at K=6). **At v0 the launder wins on bytes AND on accuracy.** A dividend
computed across a 359× byte gap is not a dividend; it is a subsidy, and this task's own rule says *same
bytes, same φ, or the number is meaningless.*
⇒ **Fixing this axis is now a first-class deliverable of the gym, not a caveat.** The harness names the
two routes and **your opening (a) IS the second one**:
- **shrink `V_θ`** toward the launder's footprint, or
- ⭐ **load the store far beyond the launder's table** — i.e. **beyond-capacity compression, opening (a)
  below, at ~3× overload**. This is exactly the regime where a table must evict verbatim and 57 KB of
  `V_θ` stops being an extravagance and starts being the point.
**Report the byte ledger and `matched` flag on every cell.** A cell with `matched=False` may be reported,
clearly labelled, but **may not be quoted as a dividend**.

**⚠ RISK THE HARNESS FLAGGED AT YOU (its follow-up #7):** its clean band is narrow —
`sep/σ_q` **6.83 → 3.07** is the difference between decode **0.92 and 0.48**. **If your gym's query law
is noisier than `σ_q = 0.15`, expect the collapsed picture, not the clean one.** Declare your `σ_q` in
PREREG and report `sep/σ_q` per cell; if a family only works at an implausibly quiet query law, that is
a finding about the family.

**Also yours:** `shared_metric_launder` is implemented in `chlu/eval/dividend.py` and **has never been
run** — it needs a fitted shared metric, which is gym territory (doctrine I-12). Monitors **#3 (vacuous
gate), #4 (blank), #6 (objective divergence) and #11 (reach) are UNTESTED** — they never fired on any
configuration the harness ran, and #6 is *inapplicable* there because it needs **≥4 consolidation
windows**. A gym stream is long enough to exercise them; **exercising a monitor for the first time is a
reportable result**, and a monitor that never fires anywhere is untested, not green.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form)
- **Dial (C2 form):** the **dynamics dividend** is the *only* KPI —
  **dividend ≡ (full CLU) − (its own settle-deleted / matched-bytes launder)**, same harness, same
  bytes, same φ. No accuracy number is reported without its dividend. No dividend is reported without
  its bytes.
- **Laundering control:** **harness-native from day 1, not bolted on.** Three controls run on *every*
  gym cell, automatically, or the cell does not report: (i) the **settle-deleted launder** (the store
  read with the dynamics removed — this is the dividend's own denominator), (ii) the **same-keys null**
  (identical keys, classical read — this is the control that beat CLU 6/6 in w26), (iii) the
  **blank/empty-store** control (N68: blanks scored 0.992–1.000; w26: a vacuous metric let a blank
  score 1.0000).
- **Falsifies — the gym, not the CLU.** If the same-keys launder ≥ CLU on **every** gym task at v0
  **and** each of those tasks admits a classical provable ceiling (§Metric-native audit), then the gym
  is **metric-native in disguise** and must be redesigned. That is a falsifier of *this instrument*, and
  it is the one you are most likely to hit. Pre-register which task families you expect to survive it.
- **⭐ Does NOT falsify:** **a dividend of ≈0 or negative at v0.** Charter §6.2 states the expectation
  explicitly: *"expected ≈ 0 or negative at v0 — that is the honest starting line, report it."*
  Reporting a negative v0 dividend is a **successful** outcome for this task.
- **⚠ A POSITIVE dividend at v0 is suspicious, not a win.** Twenty-six waves say the trivial substitute
  usually wins. Any positive cell goes through all three controls **plus** a seed re-run before it is
  written down, and it is reported as *unexplained-pending-controls*, never as a result.
  Performance-first means **explanation-later, never rigor-later** (charter §3.4).

---

## Design constraint that governs everything below
Intervention §6 criterion 4 is a **theorem about our situation, confirmed four times**: *if the query
lives in the same metric space as the stored keys, a classical method is the provable ceiling.*
⛔ **A gym task that fails criterion 4 is worthless as development currency** — it will report a
negative dividend forever and teach us nothing. **Every task family you build must carry a written
metric-native argument**: state, for that family, what the best classical method *is* and why it is not
a ceiling. If you cannot write that argument, do not build the task.

Charter §2.1 names the four places the dividend can *structurally* be positive — a table cannot follow.
**Build one gym family per opening**, so the gym spans the hypothesis space rather than sampling it:

**(a) Beyond-capacity compression.** Item-bytes ≫ budget; the table must evict verbatim, a learned `V_θ`
superposes and degrades gracefully. Test at ~3× overload (charter §2.2). Metric: graded degradation, not
hit/miss — **quote the curve, not the endpoint** (standing rule). Classical ceiling argument: a
budget-matched table *provably* cannot hold the items; the question is whether graceful degradation
beats verbatim eviction on the task metric.

**(b) Non-metric-native queries.** Aggregate-over-stored-items · relational queries · answers computed
*between* basins. These are the queries whose answer is **not any stored item**. Classical ceiling
argument: a lookup returns a stored key; the answer here is a function of several.

**(c) Trajectory information.** Order, ranked alternatives, ambiguity — *a trajectory passing near
competing wells encodes a distribution over answers; a settled point cannot.* ⭐ **This is the
highest-novelty pillar and has ZERO direct evidence in 26 waves** (intervention §3.2 — every experiment
ever run used settled points only). Build at least one family whose target is a **ranking or a
distribution**, not a single item. Requires `read()` to return the trajectory — it does
(`full-clu-harness` API freeze). ⚠ The learned trajectory read-out is `trainability-spike`'s;
you provide the **task**, and it is legitimate for v0 to score it with a handcrafted ψ and report that
as the v0 limitation.

**(d) Manifold-valued memories via flat directions.** A flat direction stores a *manifold* of settled
states, which no lookup table can express (w25 saw a low-rank settled ray, PR≈1, and filed it as a minor
observation). Lowest-maturity opening — **a scoped, honest stub with its blocker named is an acceptable
v0 deliverable here**; a fabricated task is not.

**Cross-cutting stream properties every family carries** (charter §2.2): capacity pressure ·
interference · **deletion demands** · **regime re-identification on revisit** · recall under ~3× overload.

---

## The byte accounting (the part that decides whether any number means anything)
`chlu/eval/dividend.py` gets its gym-side callers from you (signatures are `full-clu-harness`'s).
Publish, per cell, an explicit byte ledger for **both sides of the dividend**: atoms × dims × dtype +
codebook + controller state, versus the launder's keys + payloads. ⚠ **Same bytes, same φ, or the
dividend is meaningless.** The w26 frontier is the precedent: at matched bytes CLU beat the whole replay
field 6/6, *and* the matched-bytes launder was never beaten 6/6 — both facts came out of one honest
ledger. Also carry w26's correction: **"24.5× fewer floats" is now 19.1×** — never quote the old figure.

## Acceptance criterion (charter §6.2)
The gym runs **end-to-end on the harness from `full-clu-harness`**, and the **baseline dividend is
measured** on every family, with all three controls firing automatically and the byte ledger published.
Report the dividend per family with seeds and spread. **Multi-seed before any number leaves this task.**

## Dependency & sequencing
✅ **CLEARED — `full-clu-harness` is merged to `main @ 4160cf7`. Base local `main`; the whole API surface
is in the tree.** If the API is missing something you need, **request it from the Hub — do not add it to
`clu_system.py` or `monitors.py` yourself** (they are not yours). Start with the **task-family design +
the metric-native arguments + the byte-ledger spec** — they need no code and belong in your PREREG.

## File ownership
**You own (all NEW files):** `chlu/experiments/memory_gym.py` (the task generators) ·
`chlu/experiments/exp_memory_gym.py` (the runner) · `tests/test_memory_gym.py` · the **gym-side callers**
in `chlu/eval/dividend.py` (append only — **the signatures `full-clu-harness` landed are frozen**) ·
⚠ **`chlu/cli/experiment_cmd.py` — you hold SOLE ownership of this shared file this wave** (it is the
one file `full-clu-harness` had to edit, +53 lines; `trainability-spike` is forbidden from touching it).
Keep your edit purely additive — one parser block + one command function, the harness's own pattern.

⛔ **READ-ONLY — do not edit:** `chlu/core/clu_system.py`, `chlu/core/monitors.py`,
`chlu/core/clu_controller.py`, `chlu/experiments/exp_clu_system.py` (`full-clu-harness`) ·
`chlu/core/psi_readout.py`, `chlu/core/implicit_grad.py` (`trainability-spike`).
⛔ **READ-ONLY — C1W27 territory, do not edit even by one line:** `chlu/core/memory_potentials.py` ·
`chlu/core/controller.py` and `AtomStorePotential.evict` · `chlu/core/placement.py` ·
`chlu/core/admission.py` · `chlu/experiments/exp_designed_mechanism.py` · `exp_cl_entry.py` ·
`cl_baselines.py` · `exp_phi_stream.py` · the mia harness · ⚠ **`chlu/config.py`** (C1W27 owns two
blocks in it this wave). **This repo is config-driven and the reflex is to add your config class there —
do not.** Gym config objects live in `chlu/experiments/memory_gym.py`. **Consume by import; never edit.**
If you think an edit is unavoidable, **STOP and report to the Hub.**

## Compute
⚠ **≤3 engineer worktrees across BOTH campaigns**; the C1W27 Hub is running in parallel. Do not launch
background sweeps without telling the Hub. JAX cold-start ~20+ min — use `--quick` for smokes. Declare
the compute order in PREREG; anything unreached is **NOT RUN**, never a null. This is a **v0** — breadth
across the four openings beats depth on one.

## Deliverable
`PREREG.md` at `.claude/outputs/memory-gym-v0/PREREG.md` **before any cell runs** — the task families,
**the metric-native argument for each**, the predicted sign of the v0 dividend per family (⭐ predicting
"positive" here needs a stated mechanism, and the charter's own expectation is ≈0-or-negative), and the
compute order. Report at `.claude/outputs/memory-gym-v0.md`, protocol §5 format, flag-provenance table
on every quantitative result, PREREG scorecard, reconciliation list in the **first 10 lines**.
Full `uv run pytest -q` green, `ruff` clean, atomic `[experiment-engineer]`-prefixed commits.
**Do not push. Do not merge.**

⛔ **Do-not-quote, carried:** "24.5× fewer floats" (now **19.1×**) · any `tol`-metric number at m>1 ·
"the gate was cleared" without the fit budget · "0.99985" without its load · unqualified "exact
deletion" / "certified" / "unlearning" / "deletion-compliant" · "capacity multiplies by sharding" ·
"the write operator is the ceiling" · the √2 / `d^1.62` exponent.
**Standing:** ⭐ **Track 1 is development currency and is NEVER the paper's primary claim** (charter
§2.2) — nothing measured here is quotable as a headline · laundering on every performance claim ·
multi-seed before any paper number · quote the curve, not the endpoint · N94 epoch discipline.
</content>
