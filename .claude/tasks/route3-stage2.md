# route3-stage2 — the slotted write objective v0 + `allocate` v0 + γ-as-selector

> ## ⛔⛔ CONDITIONAL — DO NOT LAUNCH THIS TASK UNTIL THE HUB POSTS THE §A9.4 UNLOCK
> **Charter ADDENDUM 2 §A9.2: *"Route 3 runs as a staged second track inside C2W3. Stage 2 unlocks only
> on the stage-1 bar (§A9.4)."*** This file is **scoped and ready** and is **NOT spawned at wave
> scoping**. It launches iff `route3-stage1-plus-2x2` reports:
> 1. ✅ **`unlock = true`** under §A9.4 — *the per-slot store-attributable discriminability clears the
>    launch-noise floor beyond **2 SE**, at **≥3 seeds**, on **≥1 family**, at **any `t`**, with q-slots
>    and p-slots scored separately (**a live p-channel unlocks even with a dead q-channel**)*; **AND**
> 2. ✅ the **§A9.5 per-slot table launder did NOT reproduce the slotted read** (if it did, Route 3 has
>    degenerated into K time-indexed lookup tables and **fails regardless of dividend** — no stage 2).
>
> **The unlock ARITHMETIC is mechanical** (§A11: *"both are mechanical or Head-reserved"*) — the Hub
> applies it to stage 1's report and needs no further ruling to reach the verdict. ⛔ **But the LAUNCH
> is the Head's, always** (Head ruling, 2026-07-31, C2W3 decision queue Q4: *"launching is always my
> job"*). The Hub computes `unlock`, checks the prerequisites below, and hands the Head a spawn line;
> **the Head spawns you.** ⚠ **Two prerequisites the Hub checks before it writes that spawn line:**
> `bprime-fb4-gate`'s **`store_write_mask_factory`** (reconciliation 5) must be **merged to `main`** —
> without it a new store family cannot supply its own update mask and an unmasked leaf breaks C3
> locality — and a **worktree slot must be free** (you inherit Route 3's slot, shared with stage 1; cap
> is **≤3 staggered**, §A9.3).
>
> ⛔ **If stage 1 reports `unlock = false`: Route 3 is DEAD at this weight class, this task is never
> spawned, and B′ absorbs the attribution curve as protocol evidence** (§A10, verbatim). That is a
> clean pre-registered outcome, not a failure.

**Campaign 2, wave C2W3. Agent:** experiment-engineer. **Worktree MANDATORY** (Route 3's, slot 3 of 3).
Base: the Hub-named commit at release (`main` **after** `bprime-fb4-gate` and `route3-stage1-plus-2x2`
have merged). Branch `agent/experiment-engineer/route3-stage2`.
Charter **ADDENDUM 2 §A11 task 4**, implementing **§A10 stage 2**, **§A9.6** (`allocate` v0),
**§A9.7** (γ-as-selector), **§A8.4** (the write objective is open to redesign, under three invariants)
and **§A9.5** (the kill-condition, on every cell).

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **in full, especially
ADDENDUM 2 §A8.1–§A8.4, §A9.4–§A9.7, §A10 and §A11**; `.claude/advisor-head-intervention.md` **§5 (the
13 modes — three of your collapse modes are named there), §6, §8 (§8.1 and §8.2 both bind you)**;
`.claude/outputs/route3-stage1-plus-2x2.md` **in full — it is your entire empirical brief**;
`.claude/outputs/bprime-theory.md` **§T5 (the theorist's review of the `allocate` spec — read it BEFORE
you implement `allocate`, it exists to stop you building an un-ledgerable verb)**;
`.claude/outputs/phi-particle-head.md` **§2 and §3.4** (the friction/mass ratios and the monitor-#1 hot
band); `.claude/outputs/traj-write-objective.md` (the objective you are permitted to redesign, and the
liveness machinery you must reuse);
the **live `2026-07-31` `[C2W2]` §10 entry** in `.claude/handover_context.md`.

⛔ **REGISTRY LAG.** Quote results **only** from `.claude/outputs/*` and the §10 review entries — unless
the curator's C2W3 pass has landed by the time you launch, in which case the Hub tells you so
explicitly. Never infer from a registry's silence that a result does not exist.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **pillar 1 (expressive latents) — charter §2.1 candidates (c) trajectory
  information + (d) manifold-valued memories, made concrete**, composed with §A2.2 (γ/M-indexed
  trajectories select among slot streams within one well). This is the first C2 task that may **claim**
  a dividend rather than only measure one — and it may claim it **only** with all four controls green.
- **Laundering control:** four, on **every** cell, no exceptions: **dividend** · **settle-deleted /
  matched-bytes launder** · **+0 B substitute audit** (signed margin) · ⛔ **the §A9.5 per-slot
  matched-bytes TABLE launder** (kill-condition). Plus the **trajectory launder** on any ψ that can see
  the address block, **identical φ + φ-bytes on every arm** (ledgered, enforced in code), and the
  **same-keys null**. ⛔ **`allocate`'s launder receives the SAME allocation budget** (§A9.6) — a launder
  that is denied the verb under test is not a launder.
- **Falsifies:** §5. **Does NOT falsify:** allocation collapse to all-endpoint (**monitored, not
  forbidden** — §A9.6: *"it is D2a from a new angle"*, and it is a finding) · a dividend that is
  positive but loses to its +0 B substitute (that is the §A6 **weak-proceed** pattern and it is
  reported as such, not upgraded).

---

## 0. The hypothesis you are testing, stated exactly (§A10)
> **Distinct pieces of an item's information can be stored at distinct strided points of the
> write-shaped trajectory — slots carry `(q_t, p_t)` pairs — with the settled point carrying the rest;
> the write objective shapes the flow map so that slot contents are store-attributable, contractive
> within the launch cloud, and separated across items.**

Stage 1 measured whether the *shipped* rig already leaks store information into slots. Stage 2 asks the
write to **put it there on purpose**, and gives the read a **designed verb** for deciding where it goes.

## 1. Deliverables

### D1 — ⭐ The slotted write objective v0 (open redesign, under three BINDING invariants)
Head ruling **§A8.4**: *"Route 2's failures are objective-visibility and geometry, not optimizer
quality — the write objective is declared **open to redesign**."* You are the first C2 task allowed to
redesign it. The licence comes with three invariants, and they are not negotiable:

1. ⛔ **Coefficient-zero bit-identity.** At every new coefficient = 0, the written `V` is **bit-identical**
   to the base commit's. Route 1's precedent: both terms sit behind a Python-level `>0` branch so at
   λ=0 **not one extra op is traced**. This is a **blocking test**, and bit-identity means bit-identity
   in f32 **and** f64 — not "within tolerance".
2. ⛔ **A liveness anchor, pre-registered and SPANNING.** The Head's counterweight, verbatim:
   *"A term that never moves anything at any tested setting hasn't been asked; it's been whispered at."*
   Minimum shape: **≥3 non-zero values spanning ≥2 decades + a perturbing anchor (a coefficient at
   which the term visibly perturbs the write, even destructively) + the mandatory zero point.**
   Without the anchor, an inert-everywhere result is an **under-powered grid**, not a ≤0 vote.
   ⚠ C2W2's Route 1 failed its liveness bar at **every** coefficient (write loss exactly
   `L_endpoint + 0.55λ`, a constant) — **that is the failure mode your objective must be designed to
   avoid, and the diagnosis is that the payload only reached 0.27–0.50 of its value inside the write
   rollout.** §A8.1's answer — *the latent is the visited state itself; `∇V` **is** the store; `p_t` at
   small `t` is almost pure store* — is the design direction, not a longer rollout.
3. ⛔ **Per-slot laundering.** Every slot the objective writes into carries its own settle-deleted
   launder **and** its §A9.5 table launder, from the first cell.

### D2 — ⭐ `allocate` v0 (§A9.6, full v0)
The **designed routing mechanism** — and §A8.3 rules explicitly that this is **doctrine-conformant, not
a w20 defiance**: *"the verb set is the designed action space and was always meant to grow; only the
policy is learned"* (§3.2 unchanged). The existing designed verbs are admit · place · evict · decay ·
route · retry · stop; `allocate` joins them.

- **What it routes over:** the **simplex/discrete choice over endpoint dims · `(q, p)` slot pairs ·
  particle attributes.**
- ⛔ **Byte-ledger-conserving.** Allocation must **never** be a hidden capacity increase. The ledger is
  computed before and after and the difference is reported; **`ledger drift` is a declared collapse
  mode** and it is monitored.
- ⛔ **The launder receives the same allocation budget** (§A9.6, verbatim). A table given the same
  freedom to spread its bytes across slots is the honest comparison.
- **A minimal learned policy, trained through the trajectory read.** ⭐ It *can* be trained now and it
  could not before: `∂q*/∂q₀ = 0` exactly means a settled-point read sends **zero** gradient to its
  read-in, and the trajectory channel is the only one that carries any (`‖∂L/∂φ‖`: **0.0** implicit /
  **2.654e-9** unroll / **6.421e-3** trajectory — ratio **2.42e6**). **Designed verbs, learned policy** —
  the program's own strongest cross-wave finding, applied at the controller level. Do not relearn w20's
  lesson up here.
- ⛔ **Three DECLARED collapse modes, monitored and reported whether or not they fire** (§A9.6):
  | mode | what it looks like | disposition |
  |---|---|---|
  | **allocation collapse** | everything allocated to the endpoint = the status quo | **monitored, NOT forbidden** — *"it is D2a from a new angle"*, and it is a legitimate finding |
  | **leak-by-allocation** | early-`t` q-slots store the *query*, not the store | **guarded by the attribution curve** — §A8.1: position at small `t` is almost pure query, and this is exactly the D6/`AttentionPsi` leak mode |
  | **ledger drift** | allocation smuggles in capacity | **blocking** — a dividend bought with hidden bytes is not a dividend |

### D3 — γ-as-selector, ONE pre-registered cell family (§A9.7)
> **Same-well, friction-indexed trajectories as distinct latents. Friction FIRST (the 14× channel),
> mass second. Band bounded below by monitor #1.**

- §A2.2 is **SUPPORTED** and this is what it bought: a trajectory read sends **1.74e-3 / 1.17e-2 /
  1.88e-3** to a per-query **mass** and **2.4e-2 / 4.1e-2 / 3.3e-2** to a per-query **friction**, against
  a settled-point arm at **8.7e-9 / 4.0e-8 / 1.1e-8** (unroll) and **exactly 0.0, bitwise** (implicit) ⇒
  ratios **1.7e5–2.9e5** (mass) and **2.6e5–4.9e5** (friction), 3/3 seeds. The point arm's zero is
  **structural** (Prop Q1.1: `∇V` contains neither `M` nor `γ`), not lucky.
- ⭐ **Friction is the ~14× stronger channel — take friction.** Mass is second and is one family at most.
- ⚠ ⛔ **The declared friction band's low end is monitor-#1 HOT: `γ ≤ 0.03` trips it on the S0 store**
  (overdamping → *"the last observation"*, `corr(q*, q_last) → 0.97`, C17-3). **The band is bounded
  below by monitor #1** (§A9.7) — a γ-indexed result obtained inside a monitor-#1 trip is an
  overdamping artefact, not a selector.
- **ONE pre-registered cell family.** Do not sweep γ across the gym; declare the family and the band in
  your PREREG and run it.

### D4 — every cell, every column (no exceptions)
**dividend + launder + substitute audit + the §A9.5 per-slot table launder**, ≥3 seeds, admissible-cell
coverage per family reported **first-class and before any verdict**, signed **+0 B substitute margin**
per family, two-sided byte ledger on every arm, identical φ + φ-bytes on every arm.

## 2. ⛔ Deferred, with pointers — do NOT build these (§A10)
- **Distribution-over-trajectory reads** (charter §2.1 candidate (c) verbatim — a trajectory passing
  near competing wells encodes a *distribution* over answers). The **Noether-charge / action-integral
  accumulators are its O(1) online form** and are **post-paper by standing ruling** (§A4.4).
- **Mass-as-selector beyond one family.**
- ⛔ **Any attention-ψ number, until the `AttentionPsi` leak quarantine has landed** (reconciliation 1,
  `route3-stage1-plus-2x2` D5). `q0_only` **0.35–0.45** vs a bar of **0.19**, at every stride, blank
  store **0.37–0.47**. Use `DeepSetsPsi` or a pooled read.
- **ψ cost optimisation.** The **17.1×** trajectory-ψ surcharge is **accepted for the first paper** — it
  is the price of the only trainable read and it travels with every trajectory claim. Stride/tail
  windows, chunk amortisation, distilling trajectory-ψ → point-ψ where `D = 0`, and CLU-native ψ
  candidates are **deferred and not funded now** (§A4.4).

## 3. PREREG (`.claude/outputs/route3-stage2/PREREG.md`, before ANY measured run)
Mandatory. It must contain:
1. The **slot grid, the objective's coefficient grid (≥3 non-zero values, ≥2 decades, plus the
   perturbing anchor, plus the mandatory zero point)**, and the γ band with its monitor-#1 lower bound —
   all declared before a single measured run.
2. `allocate`'s action space, its byte-ledger conservation argument, and the launder's matched
   allocation budget.
3. Predicted values for: the per-slot dividend, the +0 B substitute margin, and ⭐ **the §A9.5 table
   launder's margin** — i.e. **state in advance whether you expect the kill-condition to fire.**
4. Which of the three declared collapse modes you predict, and at what setting.

## 4. Compute
Sized against whatever the wave has left when you are released. ⛔ **No real-data leg** — §A9.11 is
Head-reserved and decided at the C2W3→C2W4 boundary with **no compute pre-committed**. ⚠ JAX cold-start
~20 min; keep the session warm; reuse the main venv; `uv sync --frozen` only if you must, and report the
resolved JAX version in your flag-provenance table.

## 5. Falsifiers
- ⛔ **The §A9.5 per-slot table launder reproduces the slotted read** ⇒ **Route 3 has degenerated into K
  time-indexed lookup tables and FAILS REGARDLESS OF DIVIDEND** (intervention §8.2). ⭐ **The headline
  claim must require inter-slot / dynamical coupling that no per-slot table can express** — if you
  cannot name that coupling and measure it, there is no claim. First 10 lines.
- ⛔ **Coefficient-zero bit-identity fails** ⇒ everything downstream is uninterpretable.
- ⛔ **The objective is inert at every registered coefficient including the perturbing anchor** ⇒ a
  repeat of Route 1's failure mode; report it as such, with the mechanism, not as a null.
- ⛔ **Ledger drift** — allocation increases capacity ⇒ blocking; the dividend is void.
- ⛔ **`allocate`'s launder cannot be given the same allocation budget** ⇒ the comparison is invalid;
  stop and report rather than shipping an unmatched control.
- **Does NOT falsify:** allocation collapse to all-endpoint (monitored, not forbidden — it is D2a from a
  new angle) · a dead mass channel with a live friction channel (friction is the 14× channel and was
  ruled first for exactly this reason) · a dividend ≤ 0 (the C2W2 gate has already fired; B′ is the
  paper either way, and **the C2W3→C2W4 boundary re-applies the dividend question to anything Route 3
  produces — re-priced, never closed**).

## 6. File ownership (assigned at scoping; the Hub re-confirms at release)
**Yours, exclusively, at release:**
`chlu/core/allocate.py` (**new**) · `chlu/core/slotted_store.py` (**new**, if the objective needs one —
it registers through `store_potential_factory` **and** `store_write_mask_factory`, so it edits **nothing
it does not own**) · `chlu/training/train_memory.py` (**additive, behind default-off flags, with a
bit-identical-shipped-behaviour regression test** — the C1W27 `payload_gate` precedent) ·
`chlu/experiments/exp_route3_stage2.py` (**new**) · `tests/test_{allocate,slotted_store,route3_stage2}.py`

**Read-only to you:** `chlu/config.py` (**standing read-only to all C2 engineers**) ·
`chlu/core/clu_system.py` (⛔ **you register through the two factory hooks; you do not edit it. If you
believe you must, STOP and report to the Hub**) · `chlu/core/{monitors,admission,soft_certificate,
psi_readout,shell_atoms}.py` · `chlu/eval/{race,dividend,attribution,fb4_gate}.py` · `chlu/eval/rivals/**`.

⛔ **If your work requires editing a file you do not own, STOP and report to the Hub.** Nine branches
over three waves have produced zero conflicts by this rule.

## 7. ⛔ Never-quote (inherited)
The full C2W2 never-quote list applies (it is reproduced in `route3-stage1-plus-2x2.md` §7 — read it
there). The ones most likely to bite **you** specifically:
- any **`AttentionPsi`** trajectory number (**it leaks**);
- **`ε` as "the manifold-payload lifetime dial ∝ 1/ε"** without the `2α` coercivity ceiling
  (`τ_max = Γ/2α`; **`α` is the ceiling and lowering it breaks the write**) — §A4.2's tilt instantiation
  is **REFUTED on a learned store** (tilt monotonically *reduces* `λ_min`, +0.0994 → **−8.28**, two
  independent implementations, every family) because **a designed degeneracy does not survive
  superposition**;
- **`λ_min > 0`** as certifying a nonempty basin (measured capture radius **0.000** at `λ_min = +0.910`);
- any C2W3 cell as a **byte-matched** dividend (min ratio anywhere **17.11×**);
- monitor #6's **"58 trips"** without *"pre-repair"* (**27**; artefact count **31 of 58**).

## 8. Output
`.claude/outputs/route3-stage2.md`, protocol §5 format, with:
- ⭐ **the §A9.5 kill-condition verdict in the first 10 lines** — it overrides every other result;
- the named **inter-slot / dynamical coupling** the headline claim requires, and its measurement;
- **admissible-cell coverage per family, first-class, before any verdict**;
- the three declared collapse modes' status, whether or not they fired;
- the byte ledger before/after `allocate`, with the drift;
- the flag-provenance table (commit, seeds, every non-default flag — mandatory);
- your reconciliation list in the **first 10 lines** if you produce one (protocol §5 corollary);
- ⛔ **declared NOT-RUNs, never reported as nulls.**
