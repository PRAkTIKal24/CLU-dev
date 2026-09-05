# c2w10-value-surface — retention + adaptation at matched bytes, persistent vs episodic

**Campaign 2, C2W10 ("The persistent store"). Agent:** experiment-engineer. **ONE worktree — the SAME
one, reused after `c2w10-lifecycle-mechanics` merges.**
Branch **`agent/experiment-engineer/c2w10-value-surface`** from **`main @ <the Hub names the
integrated base when the mechanics spoke merges>`**, worktree `../CHLU-c2w10v`.
⚠ ⛔ **The Hub fills that base commit before this file is handed over. If you find the placeholder
still unfilled, return `BLOCKED` and say so** — an unfilled base is a Hub error that has been made
once already in this program and it was caught by the spoke, correctly.
Writes `.claude/outputs/c2w10-value-surface.md` + artifacts to `.claude/outputs/c2w10-value/`.
**Budget:** ≈ 2 days.

## ⛔⛔ MECHANICAL PRECONDITIONS — check ALL of these FIRST and refuse if any fails
1. **`.claude/outputs/c2w10-lifecycle/LIFECYCLE-MECHANICS-DONE.json`** exists with
   `lifecycle_mechanics_done == true`.
2. **`.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json`** exists with `b1_pass == true` and a
   `criterion4_cleared` boolean present.
3. **`.claude/outputs/c2w10-persistent-store/PREREG-C2W10.md` contains a dated line beginning
   `AMENDMENT — VALUE VENUE:`** naming the stream this spoke runs its VALUE cells on. ⛔ The venue is
   the Hub's ruling on `criterion4_cleared`; **you do not choose it and you do not infer it.**
**Any precondition failing ⇒ report `BLOCKED`, name the file and the failing field, stop.**

**Binding documents, read first, in this order:**
- `.claude/outputs/c2w10-persistent-store/PREREG-C2W10.md` **§1, §2, §6 IN FULL** — the rig, the
  benchmark, the GO/NO-GO precondition, legs V1–V3, the two-sided byte ledger, seeds and pairing.
- charter **§A33.1** (MECHANICS/VALUE — **VALUE legs exist only at tier level with the tier's own
  control**), **§A13 tier iii** (the system-level swap), **Add.4 §A16.2(iv) = FB2** (a GRU can never be
  matched on params AND state-bytes, Θ(h²) vs Θ(h); **a TTT-class cell can, to ±0.1 %, and is the
  two-sided control for all tier-iii work**), **§A18.1** (never-quote list), **§A31.4** (task features
  ≠ address features), **§A32.1** (⛔ no chasing daylight on a metric-native cue protocol).
- `.claude/outputs/c2w10-benchmark-scout.md` **§4, §5** — the retention/adaptation protocol and the
  published byte budgets; **§2.4** — κ_per/κ⁺ and Žliobaitė Prop. 8.

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** **lifetimes + admission** — retention of revisited regimes and adaptation after change
  points, at matched state-bytes.
- **Tier control (mandatory on every VALUE number):** the **system-level swap** — a matched-params AND
  matched-state-bytes **TTT-class cell** in the same harness, plus the benchmark-native byte-matched
  arms (SAM-kNN, kNN_S, ARF, No-Change). The persistent/episodic toggle is the **ablation**, not the
  control.
- **Falsifies:** persistent − episodic retention margin ≤ 0 at 2 SE; **or** byte-matched SAM-kNN ≥ the
  CLU store on revisit-recovery (the scout's registered criterion-4 tripwire).
- **Does NOT falsify:** losing to a per-regime oracle; losing on the drift-free `out-of-control` null
  (a **win** there falsifies the instrument); losing on a stream where the persistent-vs-episodic
  contrast is absent (the metric-native ceiling theorem, not news).
- ⛔ Depth is not feature importance unless the Hub has posted the I2 lift (§A23.5). ⛔ N94 on every
  promotable reading. ⛔ "CLU-former" is a placeholder and appears nowhere.

---

## 0. TWO READINGS, AND THEY CARRY DIFFERENT WEIGHT (Head ruling 2, 2026-08-10)
The swap control is valid **by its properties, not its file**: the whole memory is replaced and the
matching is **two-sided (params AND state-bytes)**, and it is **TTT-class, never GRU** (FB2: Θ(h²) vs
Θ(h) ⇒ a GRU can never be two-sided matched; a TTT-class cell matches to ±0.1 %). On that definition
**this harness's swap IS a valid tier-iii control** — there is no downgrade to "labelled pilot".
What is scoped is the **claim**:
- ⭐⭐ **the ±persistence contrast is INTERNAL** (same harness, same scale, both arms) and is **this
  wave's PRIMARY evidence** — clean, and promotable as a within-harness result;
- ⚠ **the absolute competitiveness-against-rivals reading is SCALE-SCOPED and explicitly
  NON-PROMOTABLE to the CSF3 tier-iii claim.** That sentence travels **beside every absolute number**,
  in every table you emit and in `VALUE-SURFACE.json` as a field, not only in prose.

## 1. GO / NO-GO — computed and filed BEFORE any VALUE cell
The pass-3 lesson is that a null is worth reporting only if it is **attributable**. Compute, on the
wave's rig, and write the result **before** running a VALUE cell:
1. **Store non-inert** at d = 12: median depth after the first writes > 0.1 on 3/3 seeds.
2. **Addressing lives:** the per-feature **G-ADDR MECHANICS instrument** (§A34.8 — permanently barred
   from VALUE duty; use it as a precondition, never as a leg) reports correct-basin rate above its
   designed floor on the stream-1 items.
3. **L6 netting green** on this rig (no erosion/flattening/recovery statement is legal without it).
4. Preconditions above satisfied.
**NO-GO ⇒ the VALUE cells are a declared NOT-RUN naming the failing leg.** ⛔ Never a null.
⚠ If you quote any **census** number, `.claude/outputs/c2w8-close/GATE-HARDENING-DONE.json` must exist
with `gate_hardening_done == true` (charter §A32.3: no future census number is quotable until the gate
hardening lands). **Simplest compliant path: quote no census number.**

## 2. The three legs
**V1 RETENTION (VALUE).** `R(b)` per PREREG §2.2 — accuracy over the first 1 000 instances after
re-entering band `b`, minus accuracy over the last 1 000 of that band's first visit — for:
persistent CLU · **its own episodic toggle** · the **system-level swap** (matched-both TTT-class cell)
· **SAM-kNN** (persistent exemplar) · **kNN_S** (episodic exemplar) · **ARF** · **No-Change**.
**V2 ADAPTATION (VALUE).** `A(b)` = instances to reach 90 % of the band's asymptotic accuracy after
entry, ⛔⛔ **measured PER-INSTANCE-SINCE-CHANGE, never per-stream-position** — decimation compresses
the drift timeline (a change that took 1 000 instances takes `1 000/m`), so a position-indexed number
**silently inflates apparent adaptation speed**. Internal comparisons survive either form; **any
literature-facing sentence does not**, and only the per-instance-since-change form may leave here.
⚠ The decimation factor `m` is fixed in `PREREG-C2W10.md` §9 **before** you run a claim cell; it
travels in the ledger with every number, and the same decimated stream serves **every** arm. ⛔ **V1 and V2 are reported as a PAIR, always** — a retention gain with an adaptation cost
beyond 2 SE is a stability-plasticity trade and is reported as the trade, not as a win.
**V3 DRIFT-FREE NULL (control).** On `out-of-control` (published: *"this dataset must be drift-free"*),
persistent ≈ episodic within 2 SE. ⚠ Its label set is **24 classes, not 6** — compare only the
persistent−episodic **contrast** across streams, never absolute accuracies.

**Metrics on every table:** prequential accuracy (window 1 000, Souza convention) **+ κ_per + κ⁺**,
with the **No-Change persistence baseline present in every table**.
⚠ **Žliobaitė Prop. 8, registered:** under temporal dependence, *false* drift alarms can RAISE
accuracy. ⇒ **report the controller's event counts (admissions / evictions / promotions / demotions /
trash routings) beside every accuracy number**, so an alarm-frequency artifact is visible.

## 3. The byte ledger — two-sided, both budget points, and neither side picks its favourite
- CLU store at d = 12: `n_atoms = 512·√2^12 = 32 768`, `bytes = n_atoms × (dim+2) × 4` with
  `dim = addr+payload = 13` ⇒ **1 966 080 B ≈ 1.875 MiB** (verify from the code, do not assume).
- Exemplar arms run at **BOTH** the published budget (**L_max = 5 000 ⇒ 665 000 B ≈ 0.634 MiB**) **and**
  the CLU's own byte count (**≈ 14 782 exemplars** — compute it exactly). ⚠ **Anti-hobbling, the F3
  rule:** the byte-matched exemplar arm is the *stronger* baseline and it runs at full strength.
- If the CLU cannot be shrunk to 0.634 MiB at d = 12 (the `min_atoms ∝ √2^d` floor), state that as
  **NOT-REACHABLE with its arithmetic** and label the column an explicit **byte-frontier column, never
  a dividend family** (the §A14.2 pattern).
- **φ params, codebook, and `trash_bytes = K·(dim+2)·4` on every arm's ledger, launders included.**
- The TTT-class swap arm is matched on **params AND state-bytes** (±0.1 % is achievable per FB2) —
  report both matches numerically. A GRU arm, if run at all, is **one-sided by construction** and
  labelled as such.

## 4. Seeds, pairing, and what may be quoted
Arms **paired on the bit-identical stream**. **≥ 5 seeds for every VALUE number**; ≥ 3 for diagnostics.
Per **A17.4**, any leg whose control carries learned-init variance uses paired or multi-init controls
or `n ≥ 9`. ⛔ **No rescue verdict at n = 3** (§A18.1). Quote the curve, not the endpoint.
⛔ Reuse the frozen stream file named in `BENCHMARK-GATE.json` (path + sha256) — one file, all arms.
⛔ Baselines: reuse the benchmark-gate spoke's scratch-venv `river` arms against the same frozen file
and consume their per-instance prediction logs. **Do NOT add `river` to `pyproject.toml`/`uv.lock`.**

## FILE OWNERSHIP (declared)
**You own (edit, all previously created by `c2w10-lifecycle-mechanics` and now merged):**
`chlu/experiments/exp_persistent_store.py` · `chlu/experiments/stream_sources.py` ·
`chlu/core/store_lifecycle.py` · `tests/test_persistent_store.py` · `tests/test_store_lifecycle.py`.
**You own (new):** `chlu/experiments/exp_persistent_value.py` (the VALUE rig + the swap arm) ·
`tests/test_persistent_value.py` · `chlu/config.py` (**additive only**) ·
`chlu/cli/experiment_cmd.py` (**additive only**).
⛔ **DO NOT TOUCH — frozen CSF3 / live pilot territory:** `scripts/csf3/` ·
`chlu/training/train_cluformer.py` · `chlu/core/blocks.py` · `chlu/experiments/exp_cluformer_pilot.py`.
⭐ The TTT-class cell: **import the banked implementation read-only** (`exp_bprime_rivals` /
`exp_cluformer_pilot` lineage) or reimplement the minimal faithful cell **inside your own module**.
⛔ **DO NOT TOUCH — C2W11 territory:** `factored_store.py` · `multiplicity_read.py` ·
`multiwell_read.py` · `psi_readout.py` (import read-only) · `null_arms.py` · `exp_cat_test.py` ·
`exp_tierii_*.py` · `exp_null_arms.py`.
⛔ **DO NOT TOUCH — `c2w8-close-gate-hardening` territory if that spoke has not yet merged:**
`well_lifecycle.py` · `clu_system.py` · `soft_certificate.py` · `tests/test_well_lifecycle.py` ·
`tests/test_gate_addr.py` · `tests/test_cifar_strong_phi.py`. The Hub states its merge status when it
hands you the base commit.

## Acceptance (mechanical)
1. All three preconditions checked and quoted **before** anything else; GO/NO-GO computed and filed
   **before** any VALUE cell, in `.claude/outputs/c2w10-value/GO-NOGO.json`.
2. `.claude/outputs/c2w10-value/VALUE-SURFACE.json` with per-arm, per-band, per-seed `R(b)` and `A(b)`,
   the metric triple (accuracy / κ_per / κ⁺), the controller event counts, and the **two-sided byte
   ledger** — every arm, launders included.
3. V1 and V2 reported as a pair; V3 present; No-Change present in **every** table.
4. ≥ 5 seeds on every VALUE number, paired on the bit-identical stream; the pairing stated.
5. Full suite green on your branch with count arithmetic stated **and the checkout named**.
6. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs, never nulls.
⛔ You do NOT declare a tier-ii verdict, a full-CLU verdict, or any scale claim. ⛔ You do not chase
daylight (§A32.1). ⛔ A clean null under a satisfied GO precondition is a **reportable finding** and is
written up as one — it is not a failure to be tuned away. ⛔ Never push `origin`; the Hub integrates.
