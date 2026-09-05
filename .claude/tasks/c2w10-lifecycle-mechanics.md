# c2w10-lifecycle-mechanics — the THREE-STATE lifecycle, kill-conditions built first

**Campaign 2, C2W10 ("The persistent store"). Agent:** experiment-engineer. **ONE worktree.**
Branch **`agent/experiment-engineer/c2w10-lifecycle-mechanics`** from **`main @ 9e0bb25`**, worktree
`../CHLU-c2w10`.
⚠ **Name the base explicitly — the shared checkout sits on a live spoke's branch
(`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`), so `git worktree add` without a commit-ish
would branch off the wrong tip:**
`git worktree add ../CHLU-c2w10 -b agent/experiment-engineer/c2w10-lifecycle-mechanics 9e0bb25`
⚠ **Never edit the shared checkout.** Reuse the main venv per protocol §4
(`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in the worktree) — do **not**
`uv sync` in the worktree (the w6 JAX-version drift lesson).
Writes `.claude/outputs/c2w10-lifecycle-mechanics.md` + artifacts to `.claude/outputs/c2w10-lifecycle/`.
**Budget:** ≈ 2 days. **This spoke is the wave's spine and it is a MECHANICS build — no VALUE cell,
no performance claim, no verdict.**
⚠ **WORKTREE PRIORITY (Head ruling 4, 2026-08-10):** **C2W11 has FIRST CLAIM** on the engineer slot
once its own gate (`.claude/outputs/c2w8-close/GATE-HARDENING-DONE.json`) clears; C2W10 sequences
behind it (the §A9.3 precedent). If the Head hands you this task while C2W11 holds the slot, **say so
and wait** — do not open a fourth worktree. Nothing else in this wave idles in that window: the
benchmark tripwire, the loader + sha256 freeze and the drift map are zero-worktree items already in
flight.

**Binding documents, read first, in this order:**
- `.claude/outputs/c2w10-persistent-store/PREREG-C2W10.md` **§1, §4 and §8 IN FULL** — your legs are
  L1–L7 and they are already registered with their designed negatives. You implement them; you do not
  re-register them. Anything you cannot land is `false` **with its reason**.
- charter **ADDENDUM 12 §A34.3** (the three-state lifecycle, verbatim), **§A20.6** (P1/P2/P3, I1, I2,
  and the Head addition: wells never useful over k streams → `γ_φ(q)`), **Add.9 §A27.1** (the netting
  build requirement and the −34 % drift), **§A23.2** (`refresh_monotonic`'s home is this wave),
  **§A28.3** (the three corrected lifecycle mechanics: erosion drives depth to zero; depth ≠
  usefulness; the optimizer's erosion is churn, not curation), **§A33.1** (MECHANICS/VALUE labels).
- `.claude/outputs/c2w8-well-lifecycle.md` **§3, §4, §7** — what is already built (the census
  instrument, the `U` telemetry, K2's trash region) and the four reconciliations R1–R5 you inherit.

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** **lifetimes + admission**, as a full-system component build. ⛔ No paper number, no VALUE
  number, no tier-ii/tier-iii verdict, no full-CLU verdict.
- **Laundering control:** none required — **and that is the point** (§A33.1: a launder margin on a
  component gate is a DIAGNOSTIC, never a pass condition). Any margin you report is a diagnostic
  column, labelled as one.
- **Falsifies:** any designed negative that cannot fail ⇒ that leg does not ship.
- **Does NOT falsify:** an empty trash population (that is K-C: the verb is reported UNEXERCISED, not
  broken, and not working either).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ `M` never without its criterion (K9). ⛔ N94.

---

## Build order is part of the acceptance: KILL-CONDITIONS FIRST

**Standing doctrine (§A12): build the kill-condition before the thing it can kill.** Commit the
designed negatives — as failing tests against stubs, then passing against the implementation —
**before** the verbs are wired into the rig, and **report your commit order**. C2W8 killed a
stage-2 build for hours of measurement by doing exactly this; it is why the wave has a lifecycle to
build at all.

## The seven legs (all MECHANICS; label every leg you touch)

The three states are **PROTECTED ⇄ ACTIVE → TRASH**. PROTECTED = no decay (`leak = 0`, the existing
permanent flag in `MemoryController`); ACTIVE = designed decay; TRASH = routed to `γ_φ(q)`.
⛔ **Demotion is PROTECTED → ACTIVE, NEVER to trash.** Trash is the never-useful/spurious route only.

**L1 PROMOTION** — ACTIVE → PROTECTED on **sustained** usage: trailing-window `read_hits ≥ h_hi`
sustained over `≥ d_dwell` chunks (hysteresis). Designed negatives: a single burst reaching `h_hi`
does **not** promote; a well below `h_lo` never promotes.

**L2 DEMOTION ⭐ (the rich-get-richer negative — mandatory, and it is the leg the Head named)** —
PROTECTED → ACTIVE within `d_demote` chunks of usage falling below `h_lo`; the demoted well's depth
then follows the designed decay law within the L6 netting tolerance. Designed negative: **a planted
early-popular-then-abandoned well MUST demote** (high stream-1 usage, zero thereafter ⇒ ACTIVE by
`d_demote`) and **must NOT be trashed by demotion**.

**L3 TRASH (the §A20.6 Head addition; the trash region's first experimental use ON)** — a well
**never useful since first appearance over `k` stream boundaries** routes to `γ_φ(q)` via
`CluSystem.trash_route`. "Useful" is the registered proxy: **item-id-keyed `read_hits`**, aggregated
per stream. ⛔ **Depth never enters the usefulness criterion** (§A28.3(ii): depth ≠ usefulness).
Three designed negatives: (a) useful-in-stream-1-only ⇒ trashed at `k` (the intended positive);
(b) **useful in every stream ⇒ NEVER trashed**; (c) **the censoring guard — a well admitted in the
last stream (age < `k` boundaries) is NEVER trashed**; never-useful-YET ≠ never-useful.
⚠ Carry K2's two implementation facts: OFF means *no field attached at all* (an empty field is not
bit-identical, because the integrator composes `1 − (1−γ)(1−γ_φ)`), and the default gate is the
**compact smoothstep**, not a sigmoid (only the compact gate is exactly zero beyond `r_k`).
`trash_bytes = K·(dim+2)·4` goes on the byte ledger of every cell.

**L4 PROTECTED FRACTION** — bound the protected fraction at `f_max` (Hub default **0.25** of budget;
you may re-derive it, and if you do, **declare the derivation**). Breaching it **refuses further
promotion** and trips a **new named monitor row `protected_saturation`**. Designed negative: forcing
every item's usage high must **trip the monitor and refuse**, never silently protect everything.
⚠ Anti-collapse doctrine: the monitor fails **loudly** at runtime; it is not a loss term.

**L5 I1 REFRESH-MONOTONICITY** — a write into an existing well never reduces its depth (netted);
rewrites refresh/deepen up to budget. Designed negative: **with the guard OFF, a planted destructive
rewrite must reduce depth.**
⚠ **Ownership note that shapes the implementation:** `refresh_monotonic` already exists as a memory
flag in `chlu/core/blocks.py:683` (C2W6, ships OFF, pytest-pinned) — and **`blocks.py` is FROZEN CSF3
territory you may not touch**. Implement the store-level guard in **your own module** on the
`CluSystem`/controller write path.

**L5-b I1 CROSS-IMPLEMENTATION VALIDATION (MECHANICS, mandatory — Head correction, 2026-08-10).**
⛔ **Do NOT state that the `blocks.py` flag is unexercised — it is not, and the evidence is yours to
INHERIT rather than re-derive.** C2W6 ran a **`p1_on_i1_on`** cell through `exp_anti_erosion.py` and
measured I1 directly. ⛔⛔ **[ERRATUM, 2026-08-11 — Add.14 §A40(a), Advisor-owned relay error, corrected
in place and append-only: the struck numbers are the `p1_off` GUARD-OFF BASELINE, not the I1 arm.
✅ THE I1 ARM IS `44 / 62 / 59` REWRITE EVENTS WITH `6 / 0 / 0` PRE-GUARD VIOLATIONS. Charter §A22 is
CORRECT AS PRINTED. ⭐ This spoke's own R1/R2 caught it, and its recomputation gave §A23.2's "zero
post-guard violations" evidence for the first time (`n_rewrite_violations_post_guard` is `null` in all
three C2W6 files; recomputed `0/0/0`).]** ~~rewrite events 27 / 40 / 70 per run, violation rates
0.593 / 0.050 / 0.043, mean 0.228 ± 0.182 — inside the parent prereg's registered 10–40 % band — with
zero post-guard violations.~~ ⇒ **Validate your store-level guard against those block-level numbers** on the same
designed rewrite events, with a declared tolerance; **a divergence beyond it FAILS the leg.** The two
implementations may not silently diverge.
⚠ **One reconciliation you own BEFORE writing that test:** Add.7 §A22 reports the destructive-rewrite
row as OFF `[0.593, 0.050, 0.043]` → ON `[0.027, 0.0, 0.0]` under **P1**, while §A23.2's "0 post-guard
violations" is the **I1** leg. **Confirm from the raw C2W6 artifact which arm each number belongs to**
and state it, before the equivalence test is pinned to either. ⛔ `exp_anti_erosion.py` is **imported
read-only** — that file stays untouched (the C2W8 precedent).
Exercising the `blocks.py` flag *at scale* rides the pilot/CSF3 spoke and is **not needed** to justify
this wave's guard.

**L6 NETTING (BUILD REQUIREMENT, not an option — Add.9 §A27.1)** — **every** depth curve you emit is
raw **AND** netted; the netting replays the per-item decay log with the exponent's `last_write_chunk`
drift. Un-netted curves overstated recovery by up to **34 %** on C2W6's seed 0 and by 14–20 % on
C2W8's census. ⭐ `chlu/core/well_lifecycle.designed_decay_factors` already does this — **import it
read-only**, do not reimplement it and do not edit that file (it belongs to a concurrent spoke).
Pytest: netted ≡ raw **bitwise** at `leak = 0`; netted > raw for `leak > 0, Δt > 0`; a well with no
writes nets to analytic `exp(−leak·Δt)` to 1e-9.

**L7 OFF BIT-IDENTITY** — `persistent_store=False` with every lifecycle verb OFF is **bit-identical
and parameter-count-identical** to current `main` behaviour (the K2 pattern, and the existing γ_φ OFF
regressions stay green).

## The substrate you build it on

**The synthetic regime-switcher** (this wave's second stream, and the designed negatives' home):
`R` hidden regimes over a shared input space (same X region → different y per regime — the INSECTS
design in miniature), an exact scripted revisit schedule with known change points, capacity pressure
(`R` > well budget), `k ≥ 3` stream boundaries so L3's criterion is computable, and a **drift-free
control condition**. ⛔ Per §A14.8 the synthetic is a **regression/mechanics instrument and NEVER a
claim venue** — say so in the report.

**Also run the legs once on the real stream** (`.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json`
names the frozen file + sha256) **if that file exists when you get there**; if it does not, run the
synthetic only and declare the real-stream legs a NOT-RUN with the reason. ⛔ Do not re-download or
re-freeze the stream yourself — one frozen file, one sha256, all arms.

**Decimation (Head ruling 5, 2026-08-10 — ACCEPTED; ⛔ truncation REFUSED, it would delete the third
cycle, i.e. the revisit, i.e. the benchmark).** The pricing probe selects the smallest `m` from the
registered ladder **`m ∈ {1, 2, 5, 10}`** that meets the wall-clock target; you **report the selection
evidence and the Hub files `m` into `PREREG-C2W10.md` §9 BEFORE any claim cell runs.** Four binding
conditions: `m` declared in the prereg before any result · **structure preservation ASSERTED IN A
PYTEST, not claimed** (all three cycles and both change points present at the chosen `m`, **with
counts**, and the test fails at an `m` that breaks either) · **one identical decimated stream for
every arm and baseline** (one frozen file, one sha256) · **decimation in the ledger, travelling with
every number.** ⚠ Decimation **compresses the drift timeline** — any adaptation-like quantity you emit
is **per-instance-since-change, never per-stream-position.**
⭐ **The reproduction gate is binding on you too:** your loader must **reproduce the sha256** recorded
in `BENCHMARK-GATE.json` before any number derived from the scratch-venv baselines is consumed.
Reproduce first, compare second.

**Rig facts carried (§A34.10, all pytest-pinned):** placing write (`atom_site_local_init`) · co-scaled
Wendland widths · **d = 12 operational, d = 16 a declared NOT-RUN** (measured inert, 131 072 atoms,
`exp(−r²/2s²) ≈ 5e-6`) · `erosion_partition=True` (P1) · address block from a **cheap unfitted
projection**, not a task-strong encoder (the §A31.4 inversion) · φ params on every byte ledger.
**Price before you commit to an operating point** (the C2W8 practice): a timing probe fixes chunk size
`C`, write inner steps and read steps; target **≤ 2 h/seed** for this spoke's cells. Any promotable
reading uses **write inner steps ≥ 40** (N94); below it, label the cell non-promotable with its reason.

## Deliverables (exact paths — later spokes gate on these strings verbatim)
1. `.claude/outputs/c2w10-lifecycle/LIFECYCLE-MECHANICS-DONE.json` — per-leg boolean table **L1…L7**,
   each with its designed-negative result, plus `lifecycle_mechanics_done` computed **mechanically** as
   the AND over all legs. ⛔ Anything not landed is `false` with its reason, never omitted, never
   quietly true.
2. `.claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json` — per-item `hits_by_stream`,
   `first_seen_stream`, per-item **raw AND netted** depth curves, per-seed `n_live` at each measurement
   point, `n_seeds`, and `n_live_max`. ⭐ **The I2 analyst spoke gates on this file and on
   `n_live_max ≥ 64` — if your operating point cannot reach 64 live wells within budget, say so
   explicitly in the file and in your report; that is a NOT-RUN, not a null.**
3. Byte ledger per cell (store, φ, codebook, `trash_bytes`), monitor trip states named (#9/#12,
   `vacuous_gate`, and the new `protected_saturation`), and **K-C's verdict**: if L3's target
   population is empty at the measured operating point, report the verb **UNEXERCISED** — not working,
   not broken.

## FILE OWNERSHIP (declared)
**You own (new):** `chlu/core/store_lifecycle.py` · `chlu/experiments/exp_persistent_store.py` ·
`chlu/experiments/stream_sources.py` (the frozen-stream loader + the synthetic generator) ·
`tests/test_store_lifecycle.py` · `tests/test_persistent_store.py` · `tests/test_stream_sources.py`.
**You own (edit):** `chlu/core/controller.py` (the promotion/demotion hooks + the protected-fraction
monitor; the LRU/staleness semantics stay pytest-pinned and unchanged) ·
`chlu/experiments/usage_telemetry.py` (cross-stream aggregation: `hits_by_stream`, `first_seen_stream`)
· `chlu/config.py` (**additive only** — a new group; touch no existing default) ·
`chlu/cli/experiment_cmd.py` (**additive only** — one new command).
**Additive-only, flagged:** `chlu/core/monitors.py` — **append the `protected_saturation` row only**;
if the registry does not support an additive row, implement the monitor inside `store_lifecycle.py`
and say so. A concurrent wave may also be appending here; keep the hunk minimal and contiguous.
⛔ **DO NOT TOUCH — concurrent `c2w8-close-gate-hardening` territory:** `chlu/core/well_lifecycle.py`
(**import read-only** — `designed_decay_factors`) · `chlu/core/clu_system.py` (**call `trash_route` /
`gamma_phi`, never edit**; if a signature change seems necessary, wrap it in your module and report
it) · `chlu/core/soft_certificate.py` · `tests/test_well_lifecycle.py` · `tests/test_gate_addr.py` ·
`tests/test_cifar_strong_phi.py`.
⛔ **DO NOT TOUCH — frozen CSF3 / live pilot spoke territory:** `scripts/csf3/` ·
`chlu/training/train_cluformer.py` · `chlu/core/blocks.py` · `chlu/experiments/exp_cluformer_pilot.py`.
⛔ **DO NOT TOUCH — concurrent C2W11 (compositional wave) territory:** `chlu/core/factored_store.py` ·
`chlu/core/multiplicity_read.py` · `chlu/core/multiwell_read.py` · `chlu/core/psi_readout.py`
(**import read-only if you need a read head**) · `chlu/core/null_arms.py` ·
`chlu/experiments/exp_cat_test.py` · `chlu/experiments/exp_tierii_*.py` ·
`chlu/experiments/exp_null_arms.py`.
⛔ Also read-only: `chlu/core/admission.py`, `chlu/core/placement.py`, `chlu/experiments/exp_anti_erosion.py`
(its residual instrument is **imported**, that file untouched — the C2W8 precedent).

## Acceptance (mechanical)
1. `LIFECYCLE-MECHANICS-DONE.json` and `USAGE-TELEMETRY.json` exist at the exact paths above.
2. **Every designed negative in L1–L5 is pytest-asserted and green, and each one is shown able to
   FAIL** (a guard that cannot fail is not a guard — the defect class caught twice in C2W8).
3. **Commit order reported**, showing the kill-conditions landed before the verbs.
4. L6: every depth curve emitted **raw AND netted**, with the three netting assertions green.
5. L7: OFF bit-identical **and** parameter-count-identical; the existing γ_φ OFF regressions still green.
6. Full suite green **on your branch**, with count arithmetic stated **and the checkout named**
   (⚠ counts are comparable only within one checkout; the base at `9e0bb25` is **1555 selected** in a
   fresh worktree; `tests/test_download_concurrency.py` is the network-hitting pair).
7. `ruff check` clean on every file you touched.
8. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs, never nulls.
⛔ You do NOT build merge verbs (K9 must be re-registered first — C2W8's `M` was vacuous), prune-by-
depth verbs, wormholes, the anytime curve, or any VALUE cell. ⛔ Never push `origin`; the Hub integrates.
