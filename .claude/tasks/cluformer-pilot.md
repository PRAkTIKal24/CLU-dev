# cluformer-pilot — ⚠ PLACEHOLDER NAME — the tier-iii pilot: the FULL C2W1 CLU as a streaming block's memory, on real data, on CSF3

**Campaign 2, wave C2W4. Agent:** experiment-engineer. **The wave's second-heaviest task, and under the
Head's ruling it is a FULL-SYSTEM build, not a light block experiment.**
**Local worktree MANDATORY** (for code) — takes a slot **freed by `bprime-c6` or `harness-debt`**.
**Compute runs on CSF3** (Head ruling, §0.2).
Base local `main` @ the Hub-named commit **after `bprime-rivals` merges**.
Branch `agent/experiment-engineer/cluformer-pilot`.
Charter **ADDENDUM 3 §A15 task 3**, implementing **§A14.3** (the real-data leg, conditional-IN) and
**§A13 tier iii**.

> ## ✅ RELEASED — 2026-07-31, at the C2W4 review. Gates 1 and 2 are CLOSED.
> **Gate 1 — the §A14.3 checkpoint: PASSES.** `bprime-rivals` landed its audit columns on `aggregate`
> (5 arms, full column set, 3 seeds, coverage 0.69–1.00) and the protocol holds (FB1/FB2/FB3 all do not
> fire in the form that would re-scope). **Base: `main @ 21a6dc4`** (three merges, zero conflicts, 1136
> tests, read-only compliance 3/3).
> **Gate 2 — ✅ THE TWO HEAD DECISIONS ARE CLOSED** (§0 below): **the C2W1 full store** and **CSF3**.
> They are **binding, not defaults**. Echo both in your first report line.
> **Gate 3 — your own PREREG (§4) is the ONLY gate left.** ⛔ **The directional falsifier is
> pre-registered BEFORE any run** — not before the *final* run, before the **first** one. §A15.3 is
> explicit and this is the one instruction in this file with no discretion in it.
>
> ### ⭐ THREE C2W4 RESULTS THAT LAND ON YOUR DESIGN — read them before you write the prereg
> 1. ⭐⭐ **`orgdiv-prereg` Theorem O1: under a settled-point read the image of `x ↦ q*` is EXACTLY the
>    set of minima of `V_θ`, so an `N_min`-row table reproduces the read FOR EVERY READER** (measured:
>    **2 distinct settled points from 4000 queries**). ⇒ **§2 D1's "a settled-point-read block is not a
>    valid tier-iii arm" is no longer a design rule inherited from T3 — it is a theorem with a
>    measurement.** Your trajectory-read training path is mandatory *twice over*.
> 2. ⭐ **`bprime-c6` measured the shipped store's actual geometry: `sep = 1.346`, fitted `s = 0.40`
>    (two independent estimates 0.7 % apart) ⇒ `d/s ≈ 4.3` — already inside the designed-gate regime.**
>    ⛔ **`bprime-theory` T5.5's `d/s ≈ 1.9` for "the rig we run" is WRONG by 45–52×** (it read the
>    admission *gate* `d_safe_override = 0.58` as the achieved spacing). **Use `s = 0.40` and the
>    achieved separation; name your ruler in every `d/s` statement.**
> 3. ⚠ **`orgdiv-prereg` Theorem O2: at the designed gate `d_safe = 4.4 s` the settled-point
>    organization is exactly nearest-centroid VQ (`D = 0.0000`).** That is **tier ii's** constraint, not
>    yours — you are not measuring an organization dividend — **but it tells you what the store's
>    default operating point does and does not express**, and it is why (1) matters so much for you.
>
> ⛔⛔ **"CLU-former" IS A PLACEHOLDER NAME (Head ruling §A13/§A14.6). It must NEVER be baked into any
> draft, figure, table, shipped filename, or sentence a referee could read.** A real name comes later.
> Use it in this task file and in your report; nowhere else.

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **ADDENDUM 3 IN FULL —
§A13 is your specification** · **§A14.3 · §A14.6 · §A15.3**; **§2.2's compute constraint** (*"150–1200
Verlet steps per read decides feasibility at sequence scale"* — **it is now yours, and §0.3 is how you
face it**); **§6 (the C2W1 harness scope) and §A4.3/§A4.4 (φ policy, ψ cost)**; **ADDENDUM 2 §A9.11**;
`.claude/advisor-head-intervention.md` **§4 (what "full CLU" means — the enumerated lever list is now
your build spec), §5 (the 13 collapse modes), §6 (the five criteria), §8 (prohibitions)**;
⭐ `.claude/outputs/full-clu-harness.md` **in full — this is the system you are embedding, and its
acceptance criterion (*"does not collapse", not "wins"*) is the one you inherit**;
`.claude/outputs/controller-doctrine.md` (**the 13 monitors → invariants → verbs spec**);
`.claude/outputs/trainability-spike*.md` (**implicit/DEQ gradients, the trajectory-read pilot, and the
17.1× ψ cost — the measured basis of §0.3**); `.claude/outputs/bprime-theory.md` **T3 (why the training
path below is not optional) and T1 (the byte ledger you must match against)**;
`.claude/outputs/track2-admissibility/PREREG-Bprime.md` §4–§5 and §8;
`.claude/outputs/rival-recon.md` §F2; `.claude/outputs/csf3-runbook.md`,
`.claude/outputs/csf3-download-race-and-sbatch.md`, `.claude/sample_script_csf3.sh` (**your compute
route**); `chlu/core/blocks.py` (**the block shell**) and `chlu/core/clu_system.py` (**the store**);
the **`2026-07-31 (later still ×2)` `[C2W4]`** and **`(later still)` `[C2W3]`** §10 entries.

⭐ **REGISTRY STATUS — CURRENT.** *"Quote outputs/§10 only"* is **LIFTED**; `claims_matrix.md` §0 has the
dated never-quote list. Three live errata: byte law **24/28** (corrected `[A(D+2)+d]/(d+m)`, floor
**2.40×** at `n_spec=1`) · **MUNKEY = ICLR-2026 workshop (oral), name QUARANTINED** · **monitor #6's
"27 post-repair" PROVISIONAL** until `harness-debt` lands (it will have, before you spawn — use their
landed count).

---

## 0. ✅ THE HEAD'S RULINGS — CLOSED AT SCOPING, BINDING

### 0.1 ⭐⭐ WHICH CLU: **the C2W1 FULL STORE. No full-CLU feature is turned off.**
> **Head, verbatim (2026-07-31):** *"we use the C2W1 full store version. we don't want to turn off any
> full clu features for upcoming waves."*

**This is a standing direction, not a one-task choice** — it governs upcoming waves, so do not treat any
lever as optional scaffolding you can stub "for the pilot".

**What goes in the block, therefore, is the full CLU as intervention §4 enumerates it:**
- **Write:** learned `V_θ` holding the items (**not arrays**), derived addressing, **admission policy**,
  **per-item lifetimes**, local/masked write, **permitted basin interaction** (under the soft
  certificate, §A9.8 — SC-1…SC-7 landed default-off in C2W3; **you turn them on and declare `B`**).
- **Read:** learned `φ` in, ⭐ **learned `ψ` out**, two-phase relaxation, **mass/friction as selectors**,
  **trajectory *and* settled point available to `ψ`**, confidence-gated retry, wormhole hops where reach
  fails.
- **Control:** the controller with its designed verb set (admit, place, evict, decay, route, retry,
  stop) and **all 13 anti-collapse monitors live as LOUD runtime guards** — ⭐ **`full-clu-harness`'s
  acceptance criterion is inherited verbatim: the system runs the stream WITHOUT TRIPPING A SILENT
  COLLAPSE MODE. "Does not collapse", not "wins". Every monitor's trip-state is a reported artifact.**
- ⛔ **The block-form `CLUBlock` (`chlu/core/blocks.py`) is NOT the memory.** It is a driven-Hamiltonian
  recurrence with no store, no admission, no placement — **it is the w20/w21 object and it is ruled
  out.** You may reuse the *block shell* around it (embedding, norms, residual, decoder) but the memory
  slot holds the C2W1 store. **Declare this distinction explicitly in your report** — a referee reading
  "the CLU" must be told which one, and this time the answer is the strong one.

⭐ **Consequence, stated so it is not discovered late: this is a research build, and it may not complete
inside C2W4.** That is accepted and it is why §3 defines **staged acceptance** — a truthful partial that
lands the block, the training path and the swap control is a real wave deliverable; a rushed 26–47 M
number without them is not. **A truthful partial beats a confident wrong one.**

### 0.2 ✅ WHERE THE COMPUTE RUNS: **CSF3.**
> **Head (2026-07-31):** *"yes we will go to csf3."*

- **Local:** code, unit tests, gradcheck, smoke runs at toy scale, the whole `--quick` path.
- **CSF3:** every 26–47 M run. Follow `.claude/outputs/csf3-runbook.md` and
  `.claude/sample_script_csf3.sh`; ⚠ **`csf3-download-race-and-sbatch.md` documents a real download race
  — read it before you fetch enwik8/WT-103, and stage the data once, deterministically.**
- ⚠ **Pin the environment and report it.** `uv sync --frozen`; ⭐ **report the resolved JAX version in
  your flag-provenance table on BOTH machines** — the w6 lesson is that a fresh sync in a new location
  resolved a newer JAX and flipped a bit-level test. **If local and CSF3 resolve different JAX versions,
  say so before you report a number.**
- ⚠ **The ≤3-worktree cap is about LOCAL cores** (w26 thermal incident, 575/8). CSF3 jobs do not consume
  it; your local worktree still does. **Do not run 26–47 M anything locally.**
- ⛔ **Declare a wall-clock and a CSF3 allocation budget BEFORE the first submitted job, and report
  against it.** §A9.11 pre-committed that no compute is spent on this leg without a decision; the
  decision is now made, the budget is not. If a job will not finish inside it, **stop and report.**

### 0.3 ⭐ The compute constraint is now yours, and it is faced, not hidden (charter §2.2)
*"150–1200 Verlet steps per read decides feasibility at sequence scale"* — and under ruling 0.1 you also
carry the **trajectory read at 17.1× the point read** (§A4.4, accepted for the first paper; the price
travels with every trajectory claim). **Naïvely, that is a full settle per token. It will not run.**
**The two charter-sanctioned mitigations are MANDATORY, not optional:**
1. ⭐ **Memory operations at CHUNK granularity** — *"as Titans-class memories do — fair"* (§2.2 says so
   explicitly, so it is not a concession you have to argue for). Declare the chunk size, and ⛔ **give
   the swap control's cell the same chunk convention** or the comparison is not matched.
2. **Compute-adaptivity as a feature** — the accuracy-vs-Verlet-steps curve (§D5). ⚠ **Shape claim
   only** (§A3): the anytime figure is **occupied** (DEQs, EBTs, Titans-Revisited). Claim no monopoly.
⚠ Deferred-and-not-funded ψ optimisations, so you do not reinvent them: stride/tail windows · chunk
amortisation · **distilling trajectory-ψ → point-ψ where `D = 0`** · Noether-charge / action-integral
accumulators (O(1), physics-native). **Post-paper by standing ruling — note if one becomes necessary,
do not build it.**

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **TIER iii — the full CLU inside a block, on real streams.** ⭐ **Its control is the
  SYSTEM-LEVEL SWAP:** replace the CLU *inside the block* with a matched-state GRU/TTT-class cell,
  **matched params AND state-bytes, everything else bit-identical.** ⛔ **The settle-deleted /
  matched-bytes launder is TIER i's control, NOT yours** (§A13 — *controls move UP a level, they never
  relax*). Reporting a tier-i launder here is a category error; reporting **no** swap control is a
  protocol violation.
- ⛔ **Emergence claims live here and ONLY here — and only with the pre-committed scale falsifier**
  (§2 D4). Everywhere else they are forbidden.
- ⛔ **You do not claim tier ii** (the organization dividend / cat test — `orgdiv-prereg` pre-registers
  it this wave, C2W5 runs it).
- **Laundering control:** the **system-level swap** (mandatory, every claim) · **dynamic evaluation as a
  mandatory substitute column** (D3) · blank/leak controls · **identical embedding + identical φ, with
  φ-bytes ledgered on every arm**.
- **Falsifies:** §5, and your own pre-registered directional falsifier, which is the binding one.
- **"Unexplained win permitted, uncontrolled win never."**

## 1. Why this task exists, in five sentences
§A13 states the mis-specification plainly: **the settle-deleted launder tests whether inference-time
dynamics beat a table *given the organization* — both arms inherit the same placement — and it has never
tested whether physics-trained organization beats non-physics-trained organization.** FB4 then proved
our designed gym families **measure our own constructions back at us** (three of four saturated at ≤4 B),
and `track2-admissibility` proved **zero synthetic Track-2 candidates survive the substitute audit**.
⭐ **The Head's verbatim intent:** *"we need to get to real data tests and that's where the meat of our
paper will lie — enough manufactured gym tests that isolate a single effect."* **Attention itself is a
data-dependent table at inference; its value was only ever demonstrated in composition, inside a
block** — so the full system, in a block, on real streams, against a system-level swap, is the honest
venue, and it is the first time all 26+ waves' levers are simultaneously live on a real task.

## 2. Deliverables

### D1 — the block, chained, on a real stream, with the full store as its memory
Embedding + **the C2W1 full CLU** + control + assimilation/decoder + norms + residual (§A13's own
enumeration), chained to depth, on **enwik8 or WikiText-103**. `chlu/core/blocks.py::SequenceModel` +
`make_block` are the **shell** — extend them; keep every non-CLU block's construction **bit-identical**
so D2's swap is genuinely a swap.
⭐ **§A13's design rules are BINDING, and each is a measured lesson, not a style note:**
- ⛔⛔ **`∂q*/∂q₀ = 0` EXACTLY ⇒ the block trains end-to-end ONLY through trajectory reads / implicit
  gradients — and under ruling 0.1 this is the load-bearing constraint of the whole build.** A
  settled-point read sends **zero** gradient to its read-in (`‖∂L/∂φ‖`: **0.0** implicit · **2.654e-9**
  unroll · **6.421e-3** trajectory — ratio **2.42e6**). ⭐ **Implicit gradients reach the STORE
  parameters but NOT `φ`, `q₀`, `M` or `γ`** — those are transients and `Fix(T_θ)` contains none of them
  (theorist T3). **So a learned φ feeding the store is trainable only through the trajectory channel.
  There is no third option.** ⛔ **A settled-point-read block is not a valid tier-iii arm.**
- ⛔ **γ / M are trainable selectors only through that same channel** — the mass gradient at an implicit
  settle is **exactly 0.0, bitwise, 3/3 seeds**; under a trajectory read the ratios are **1.7–2.9e5**
  (mass) and **2.6–4.9e5** (friction). ⭐ **Friction is the ~14× stronger channel — if you make one a
  selector, make it γ.**
- ⛔ **Allocation collapse is a gradient-flow ATTRACTOR, not merely a monitored outcome** (theorist T3
  corollary): the moment a policy reaches the all-endpoint corner **its own gradient dies and it cannot
  leave.** **Initialise away from it** and report `‖∂L/∂(policy logits)‖` at init beside your liveness
  anchor.
- ⚠ **The 2α coercivity floor** — `τ_max = Γ/2α`; **α is the ceiling and lowering it breaks the write**.
  Every lifetime/manifold claim carries the coupling. ⛔ **`ε` is NOT the lifetime dial** (refuted on a
  learned store, §A4.2 struck).
- ⚠ **The soft certificate is the sharing/interaction precondition** — turn SC-1…SC-7 on, **declare
  `B`**, and use `bprime-c6`'s **re-located** value (C2W3's edge was set with a broken `sep/2` ruler).
- ⚠ **`AttentionPsi` RAISES** (`AttentionPsiLeakError`) — it leaks (`q0_only` 0.35–0.45 vs a bar of
  0.19). **Do not route around the quarantine**; build your ψ on a channel that passes the trajectory
  launder.
⭐ **Report every one of the 13 monitors' trip-states as an artifact** — that is `full-clu-harness`'s
inherited contract and it is what makes a "does not collapse" claim checkable.

### D2 — ⭐⭐ The system-level swap control (mandatory; without it there is no result)
Replace **the CLU inside the block** with a **matched-state GRU** and, if the budget allows, a
**matched-state TTT-class cell** — **matched parameters AND matched state-bytes**, same embedding, same
depth, same norms, same residual, same optimiser, same data order, same seeds, **same chunk granularity
(§0.3)**. **Everything except the cell is bit-identical.**
- ⭐ **Two-sided byte ledger on every arm**, with `PREREG-Bprime.md` §4's conventions and the
  **learned-initial-state rule**: **an initialisation is PARAMETERS; only the per-sequence deviation is
  STATE. Both declared.** Apply it to the store's `V_θ` init too — we score ourselves by the same rule.
- ⚠ **The matching is HARDER for the full store than for a block recurrence, and you must state the
  convention you chose before you measure.** The store's per-item cost is
  **`[A(D+2)+d]/(d+m)`** relative to a table row (corrected law; floor **2.20×**, **2.40×** at
  `n_spec=1`) — matched *state-bytes* against a GRU's `d_hidden` floats therefore buys the CLU **fewer
  items**, and that is a real, reportable property, not a confound to hide. ⛔ **If matched params and
  matched state-bytes cannot both be hit without an arbitrary modelling choice, that is FB2's shape at
  tier iii — report it the same day (§5).**
- **≥3 seeds. Multi-seed before any paper number, no exceptions.**

### D3 — ⛔ Dynamic evaluation as a MANDATORY substitute column
**Krause et al., ICML 2018 — the published criterion-3 weakness, 1.08 bpc.** §A9.11 made it mandatory and
§A14.3 carried it. ⛔ **Pre-registered consequence, already committed and not renegotiable at reporting
time: if the dividend vanishes once dynamic evaluation is in the table, the primary is dead.**
⚠ Plus the family's other substitutes: the **+0 B / trivial reader** in whatever form the task admits,
and a **blank/leak control** (collapse mode #4 — a 1e-4 address leak made classification perfect at N68;
blanks read 0.992–1.000). ⭐ **The audit's lesson transfers: the answer must not be recoverable by
something that costs nothing.**

### D4 — the ⭐ PRE-COMMITTED SCALE FALSIFIER (Head-owned, §A13)
Register, **before any run**, **what the block must show DIRECTIONALLY at 26–47 M** for tier iii to be
alive: a threshold, a sign, a monotone trend — with a tolerance and a seed count. ⭐ **"Needs >500 M to
see" is a FEASIBILITY FINDING and it is the Head's to rule — it is not a result and it is not a null.**
⛔ **A falsifier you cannot fail is not one.**

### D5 — the anytime shape curve (secondary; only if D1–D3 land)
Accuracy vs Verlet-steps-per-read. ⚠ **SHAPE claim, not uniqueness** (§A3 — the figure is occupied).
⚠ The trajectory read costs **17.1×** the point read and that price travels with every trajectory claim.

## 3. ⭐ STAGED ACCEPTANCE — how this task succeeds even if 26–47 M does not finish
Ruling 0.1 makes this a full-system build; **the wave may end mid-build and that is an anticipated
outcome, not a failure.** Land the stages in order and report which you reached:
- **S1 — the block exists and does not collapse.** Full store in the block, all 13 monitors live and
  reported, runs a real stream at toy scale without a silent collapse mode. *(This is
  `full-clu-harness`'s bar, inherited.)*
- **S2 — the training path is real.** Gradients flow end-to-end `token → φ → store → trajectory ψ →
  loss` at usable wall-clock, **gradchecked**, with `‖∂L/∂φ‖` measured and compared against the
  settled-point arm's **0.0**. ⭐ **S2 is the single most valuable thing this task can produce** — it is
  T3 tested in-system rather than in a probe, and it is a first.
- **S3 — the swap control is defined and matched.** Both matching conventions hit, ledgers published,
  the GRU arm trains on the same data order and seeds.
- **S4 — the CSF3 run at 26–47 M**, with D3's substitute column and D4's falsifier adjudicated.
⛔ **Report the stage you reached and declare the rest NOT-RUN with reasons. Never present an unreached
stage as a null.** ⚠ If S1 or S2 fails, that is a **first-order finding about the full system on real
data** and it goes to the Head — it is exactly the kind of result the program exists to produce.

## 4. PREREG (`.claude/outputs/cluformer-pilot/PREREG.md`) — ⛔ BEFORE ANY RUN. NO EXCEPTIONS.
1. ⭐ **The directional falsifier** (D4) — threshold, sign, tolerance, seed count, **and how it was
   derived**.
2. **The scale falsifier** — and what result makes *"needs >500 M"* the honest conclusion.
3. **Predicted swap margins** (CLU vs matched-state GRU vs matched-state TTT cell), signed, per arm,
   **derived not guessed**.
4. **The predicted effect of dynamic evaluation on each arm**, including the pre-committed *"if the
   dividend vanishes with dynamic evaluation in the table, the primary is dead."*
5. **The byte ledger you predict** per arm at matched params, learned-initial-state rule applied, **plus
   the item count the store gets at matched state-bytes** (D2's real consequence).
6. **The chunk granularity** (§0.3) and the per-read Verlet-step budget, both arms.
7. **Your wall-clock + CSF3 allocation budget**, and **what you cut first** if you exceed it.
⚠ A pre-registered prediction that survives is evidence; one that fails is a **finding**; an
un-pre-registered agreement is **neither**.

## 5. Falsifiers (beyond your own)
- ⛔ **"The swap is not a swap."** Matched params and matched state-bytes cannot both be hit without an
  arbitrary modelling choice ⇒ **tier iii has no control this wave.** Report the same day; do not
  proceed on a one-sided match.
- ⛔ **"The block does not train through the mandated path."** Gradients do not flow end-to-end through
  the trajectory read at usable wall-clock ⇒ **T3 biting in-system**, a first-order finding, not a
  setback. Report it with the measured `‖∂L/∂φ‖`.
- ⛔ **"A collapse mode fires and cannot be held."** A lever cannot be kept in its productive band inside
  a real stream ⇒ **name the mode, its monitor, and the verb that failed to restore it.** That is
  intervention §5's whole thesis being tested, and a loud failure is the deliverable.
- ⛔ **"The CLU arm is not competitive and the swap explains it."** ⭐ **Pre-commit to saying so**, in the
  FB3 spirit — a matched-state GRU winning is **a result of the pilot**, and the honest one. Do not
  re-frame.
- **Does NOT falsify:** a hard task · a slow arm · needing more scale (D4's feasibility finding,
  Head-owned) · the CLU losing to attention (**attention is not the swap control**) · not reaching S4.

## 6. What you must NOT do
- ⛔ **No tier-i launder as your control**; ⛔ **no tier-ii claim**; ⛔ **no "CLU-former" anywhere
  draft-adjacent.**
- ⛔ **No turning off a full-CLU feature to make a number legible** — that is the Head's ruling *and*
  intervention §8.2 (*moving toward the degenerate configuration to obtain a clean number is
  forbidden*). **Isolation is permitted only as a diagnostic INSIDE the full-system run, never as the
  experiment** (§8.1). If a lever must be staged, **stage it in its known productive band and say so**;
  do not delete it.
- ⛔ **No gym family as a claim venue** (§A14.8 — heavily demoted to regression instruments). Use them to
  check you did not break a monitor; **not as evidence.**
- ⛔ **No benchmark failing intervention §6 criterion 4 (metric-native).** MAD/zoology/MQAR are
  inadmissible as a Track-2 primary (Arora ICML'24 Thm 3.1, Ω(N)-bit state). `chlu/data/mqar.py` exists
  — **it is not your venue.**
- ⛔ **No primary claim on an axis where the competition is absent by construction** (§8.3).
- ⛔ **No compute — local or CSF3 — before the Hub releases you.**

## 7. File ownership (11 branches / 4 waves / 0 conflicts — keep it)
**Yours, exclusively, from your release** (the Hub confirms the transfers at release, after
`harness-debt` and `bprime-c6` have merged):
`chlu/core/blocks.py` · `chlu/core/psi_readout.py` (**the trajectory ψ — no other C2W4 task owns it;
⛔ do not un-quarantine `AttentionPsi`**) · `chlu/data/enwik8.py` / `chlu/data/wikitext.py` (**new**) ·
`chlu/experiments/exp_cluformer_pilot.py` (**new**) · a **new trainer module you own** under
`chlu/training/` · `tests/test_blocks.py`, `tests/test_cluformer_pilot.py`, `tests/test_data_enwik8.py`.

⚠ **THE STORE IS USED, NOT EDITED.** `chlu/core/{clu_system,admission,placement,memory_potentials,
integrators,controller,monitors,soft_certificate}.py` are the C2W1 system — **compose them through their
public API from your own block module.** ⛔ **If the build genuinely requires editing one, STOP and
report to the Hub with the exact hunk and why** — under ruling 0.1 that is a plausible and legitimate
need, and it is a Hub routing decision, not yours to take unilaterally. ⭐ **The `store_write_mask_factory`
and `store_potential_factory` seams (landed C2W3) exist precisely so a new consumer composes instead of
editing — use them first.**

**Read-only regardless:** `chlu/config.py` (**standing read-only to all C2 engineers**) ·
`chlu/eval/race.py` (**frozen schema**) · `chlu/eval/dividend.py`, `chlu/eval/rivals/`
(**`bprime-rivals`**) · `chlu/eval/attribution.py`, `chlu/cli/experiment_cmd.py` (**`bprime-c6`**) ·
`chlu/experiments/memory_gym.py` (**`harness-debt`**).
⚠ **`chlu/core/blocks.py` is shared history — `CLUBlock`'s 3-way key split is load-bearing** (a 4-way
split silently re-randomises every published w20 CLU cell). **Add; do not re-key.**

## 8. ⛔ Never-quote (full dated list in `claims_matrix.md` §0)
⭐ **"CLU-former"** anywhere draft-adjacent · any **tier-i or tier-ii** claim from your evidence ·
**Titans as "a preprint"** (NeurIPS 2025) · any **SDM Table 1 state/param ratio** · **MUNKEY as "ICML
2026"** or with a named workshop · **"MAD `compression` is the admissible synthetic"** (dead by
arithmetic) · **"verified to 1e-9 in all 28 cells"** (24/28) · **"principled forgetting"** as a novelty
phrase · **"we alone delete"** · the anytime curve as a **uniqueness** claim (shape only) · any
**`AttentionPsi`** trajectory number · **monitor #6's counts** unless you quote `harness-debt`'s landed
post-repair number · **`ε` as the manifold-payload lifetime dial ∝ 1/ε** without the `2α` ceiling
(`τ_max = Γ/2α`) · the ridge saddle **`λ_min = −0.5946`** as multi-seed (seed 0; 3-seed mean
**+0.177 ± 0.469**) · **`sep/2`** as a certified inradius · **`λ_min > 0`** as certifying a nonempty
basin (**0.000** at `λ_min = +0.910`) · **"Prop D1 is violated"** (retired) · **`k*`** without *"of
`∂q_N/∂θ`, and only where the fixed-point sensitivity dominates the transient"* · any C2W3-or-later gym
cell as a **byte-matched** dividend (**17.11×** min).

## 9. Output — `.claude/outputs/cluformer-pilot.md`, protocol §5 format
- ⭐ **First screen:** the two Head rulings (§0) echoed · **which stage of §3 you reached** · the
  **directional falsifier verbatim as pre-registered** and its verdict;
- ⭐ **the swap table**: CLU vs matched-state GRU vs matched-state TTT cell — same block, matched params
  **and** state-bytes, ≥3 seeds, **with the dynamic-evaluation substitute column IN the same table, not
  a footnote**, and the **item count the store gets at matched state-bytes**;
- ⭐ **all 13 monitors' trip-states**, as artifacts — the "does not collapse" claim is checkable or it is
  not made;
- the **two-sided byte ledger** per arm, learned-initial-state rule applied to `V_θ` too;
- ⭐ **`‖∂L/∂φ‖` through the trajectory read vs the settled-point arm's 0.0** — the S2 result;
- the **chunk granularity and per-read Verlet budget**, both arms;
- the **scale falsifier's verdict**, including *"needs >500 M"* as a **feasibility finding** flagged to
  the Head if that is the honest answer — never reported as a null;
- the **PREREG scorecard** and the **flag-provenance table** (commit, seeds, every non-default flag,
  **the resolved JAX version on local AND CSF3** — mandatory);
- **wall-clock and CSF3 allocation spent vs budgeted**;
- **your reconciliation list in the FIRST 10 LINES** if you produce one;
- ⛔ **declared NOT-RUNs, never reported as nulls** — including every §3 stage you did not reach.
- **Git footprint**: branch, commits, files, tests. Before removing the worktree, **verify from the MAIN
  repo that your branch ref shows your commits** (`git -C /Users/user/Desktop/CHLU log --oneline
  main..agent/experiment-engineer/cluformer-pilot`) — the w4 lesson cost 8 commits. ⛔ **Never push
  `origin`**; do not push at all.
