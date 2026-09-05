# c2w8p3-capture-strong-phi — THE SPINE: the completed capture gate at strong φ

**Campaign 2, wave C2W8 PASS 3. Agent:** experiment-engineer.
**Worktree 3 of ≤3.** Branch **`c2w8p3-capture-strong-phi`** from **`main @ 18b4205`** — the
integrated base carrying BOTH preconditions' merges (wt1's G-ADDR + wt2's projection). Create it as:
`git worktree add ../CHLU-c2w8p3c -b c2w8p3-capture-strong-phi 18b4205`.
⚠⚠ **HUB ERRATUM (2026-08-09): the first issue of this task file left this field as an unfilled
placeholder** (`main @ <the Hub will name the integrated base>`), which was an **independent second
blocker** the spoke correctly caught and reported alongside the precondition race. **It is now a real
commit. The Hub owns that error.**
Writes `.claude/outputs/c2w8p3-capture-strong-phi.md` + artifacts to
`.claude/outputs/c2w8p3-capture-strong-phi/`. **Budget:** ≈ 1 day build + ≈ 8–12 h measured.
Price every cell before running; **cut seeds before cutting a cell, and declare the cut.**

## ⛔⛔ MECHANICAL PRECONDITION — BOTH FILES MUST EXIST, OR RETURN `BLOCKED`

```
.claude/outputs/c2w8p3-gate-addr/GATE-ADDR-VALIDATED.json    with  gate_addr_validated == true
.claude/outputs/c2w8p3-phi-geometry/results/PHI-GEOMETRY.json   (geometry_go either value — see below)
```
⛔ **If `GATE-ADDR-VALIDATED.json` is absent or `gate_addr_validated != true`, STOP and return
`BLOCKED`.** Charter §A30.1 is a **binding order**: nothing is scored before the gate can fail on
addressability. ⚠ **HUB ERRATUM: the first issue named this file at the directory root; wt2 wrote it under
`results/`. The path above is the CORRECTED, verified-on-disk one** — a mechanical precondition must
name the path the spoke actually writes, or the gate fires on a file that exists.
⛔ **If `results/PHI-GEOMETRY.json` is absent, STOP and return `BLOCKED`** — without it a null
here is confounded and unattributable.
⭐ **`geometry_go == false` does NOT block you** — it **RE-LABELS** your result: you are then
measuring the physics on a substrate *known in advance* not to have fixed separability, and you say
so in your §1. **Carry that label on every number.**
*(A gated spoke gets a mechanical file precondition, never a prose gate — standing doctrine.)*

**Binding documents:** `.claude/outputs/c2w8-well-lifecycle/ERRATA-C2W8-PASS3.md` **§1 — the four Head rulings (R1 CIFAR-10 substrate · R2 projection in scope + launder uses it · R3 the d rule · R4 NO-GO re-labels). READ BEFORE THE PREREG.** · `PREREG-C2W8-PASS3.md` — **§6 (the spine, yours), §4 (scale guard), §8
predictions Q4/Q5/Q6, §9 NOT-RUNs** · charter **ADDENDUM 10 §A30.2 (your mandate) + §A29.6 (the D2a
warning)** · both precondition files · `.claude/outputs/c2w8-cifar-strong-phi.md` + `results/` ·
pass 1 `census.json` · `c2w8p2-compact-atoms.md` §7.1 (the lever attribution) · intervention §5/§8.

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** none as a new claim — a **component build** measuring whether the physics adds anything
  once the encoder is not the bottleneck. ⛔ **No paper number. No tier-ii verdict** (no organizer
  swap exists here). ⛔ **No full-CLU verdict** (§A28.4). ⛔ **No arm-race adjudication.**
- **Laundering control:** the **same-keys kNN-in-φ launder in the SAME projected φ**, on every cell,
  with the byte ledger beside it (φ params + projection params on **every** arm including the
  launder). **State matched-items vs matched-bytes on every quotation**; the **1 253×** caveat travels.
- **Falsifies:** nothing is "falsified" here — **both branches are pre-registered as reportable**
  (below). What would be non-compliant is *tuning away* branch (b).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94 epoch discipline on every cell.

---

## ✅ BOTH PRECONDITIONS ARE NOW SATISFIED (Hub-verified on disk, 2026-08-09) — verify them yourself, then proceed

- **`GATE-ADDR-VALIDATED.json`: `gate_addr_validated = true`**, all 9 checks — incl. **arm B's banked
  config FAILS G-ADDR**, the **planted permutation fails**, three **scale-control** checks hold, and
  ⭐ **`R3_attractor_can_move_off_the_key = true` ⇒ the `atom_site_local_init` compliance ruling
  STANDS** (no reversal, no escalation).
- **`results/PHI-GEOMETRY.json`: `geometry_go = true`** — strong φ improves σ_q/spacing over the PCA
  reference at matched `d`, beyond 2 SE, **3/3 paired seeds**. The projection is built and
  **`launder_reads_projected_phi = true`** is asserted in code (R2(b) discharged).

## ⛔⛔ THE `d` IS RULED BY MEASUREMENT, AND IT IS **12**, NOT THE GEOMETRY-FAVOURED 16

wt2 reports `d_favoured_by_geometry = 16` **but `store_inert_by_d[16] = true`** ⇒
**`d_recommended_operational = 12`**. ⭐ **Run at `d = 12` (32 768 atoms), and state `d` and its atom
budget together as ONE joint dial.** ⛔ **Do not run the spine at d = 16**: an inert store makes a
census vacuous for a reason that is **NOT** the gate's reason, and the two must never be conflated
(Head ruling R1). ⭐ This also **upgrades the revived (d, atom-budget) rider from a scratch probe to
a censused measurement** — d=16 inertness is now measured at a fully honoured atom budget.

## ⛔ SUBSTRATE, DIMENSION AND SEMANTICS — FIXED BY HEAD RULING (ERRATA §1)

- **R1 — the spine runs on SPLIT-CIFAR-10** (`dataset = "cifar10"`), where the encoders were built and
  priced. ⛔⛔ **Pass-1/pass-2 census numbers are MNIST and are NOT your baseline.** Your weak-φ
  comparison is an **INTERNAL PCA-φ CIFAR-10 census arm at the same `d`, in this same run**. ⛔ **No
  pass-3 number may be compared to a pass-1/2 census number** — that would be cross-dataset AND
  cross-encoder AND cross-checkout at once.
- **R2 — the projection is wt2's, and the launder reads the PROJECTED φ.** Verify that assertion holds
  in the code you inherit; if the launder reads 256-dim φ anywhere, **stop and report it**.
- **R3 — RESOLVED BY MEASUREMENT: run at `d = 12`** (see the block above; the geometry-favoured 16 is
  inert). State `d` and its atom budget as ONE joint dial.
- **R4 — `geometry_go == false` RE-LABELS you, it does not block you.** Carry the label on every
  number. ⛔ **`gate_addr_validated == true` remains a HARD block.**
- ⚠ **DECLARED RISK: `CluSystem` has never been run on CIFAR φ** and was **inert at `d ≥ 16` on
  MNIST**. **Measure well depth FIRST; report an inert store in your first 10 lines.** An inert store
  makes the census vacuous for a reason that is **not** the gate's reason.

## The one question

**Does the physics add anything once the encoder is not the bottleneck?**

Re-run the **frozen census + the COMPLETED gate (G-CAP · G-DEC · G-DRIFT · G-ADDR)** with **arm A's
co-scaled-width store** on a **strong-φ rig**, reusing `c2w8-cifar-strong-phi`'s built-and-priced
encoders: **simclr primary** (`enc_steps = 8000`), **randconv** the cheap control arm, `task1_only`
regime, pass-1 provenance discipline unchanged.

⭐ **The store arm is "CO-SCALED WIDTH", not "compact"** (§A29.4(i)): a plain Gaussian at the same
co-scaled width also cleared the pass-2 gate 3/3 and **decoded better**. **Kernel form is a DECLARED
SECONDARY AXIS**, reported, never the headline.

## ⛔⛔ BOTH BRANCHES ARE REGISTERED AS REPORTABLE, BEFORE YOUR FIRST CELL

- **(a) DAYLIGHT** — measurable separation opens between the settle and its **own same-keys launder**
  once both can address ⇒ the **first candidate physics signal on this substrate**. (Prior **Q5 = 0.15**.)
- **(b) NO DAYLIGHT** — the CIFAR spoke's **±0.0007** result reproduced on the census rig ⇒ **the
  tier-i thesis measured at the CL substrate. This is a REPORTABLE FINDING, not a failure to be tuned
  away.** (Prior **Q6 = 0.70** — ⭐ **the Hub expects branch (b), and registering that now is what
  stops it being tuned away later.**)

⛔ **Neither branch is a tier-ii verdict.** ⛔ **No paper number is produced.**

## ⚠⚠ THE D2a WARNING TRAVELS (§A29.6, standing — read before choosing any objective)

**G-DRIFT → 0 means the settled point approaches a deterministic function of the stored key = D2a =
TABLE-EXPRESSIBLE** — the configuration intervention **§8.2 prohibits**, and **exactly what the CIFAR
arm already measured at strong φ** (settle = same-keys kNN to **±0.0007**).
⛔ **No leg, objective, or tuning choice may treat drift → 0 as a target.**
⛔ **G-DRIFT is reported as a TWO-SIDED diagnostic.** ⚠ **The pass-2 gate as written REWARDS the
degenerate configuration** — if your best-scoring cell is also your lowest-drift cell, **say so
prominently**; that co-occurrence is the D2a signature, not a success.

## The build

1. **Verify both preconditions and echo their key fields** (`gate_addr_validated`, `geometry_go`, the
   declared mapping, the chosen `d` + atom budget as ONE joint dial) **before any cell.**
2. **The cells:** frozen census + completed gate, ≥ 3 seeds, simclr primary + randconv control, with a
   **PCA-φ reference arm at the same `d` in the same harness** — ⛔ the weak-φ comparison must be
   **internal to this run**, not lifted from pass 1/2 (different checkout, different rig; a
   cross-run comparison is exactly the error this wave keeps catching).
3. **Every geometric quantity as a DIMENSIONLESS RATIO with the scale stated** (prereg §4); carry the
   **scale-only control** wt1 built if it is available to you.
4. **Byte ledger on every arm including the launder**, with φ params **and** projection params
   counted, `(d, atom budget)` as one declared joint dial.
5. **Report G-ADDR's A1/A2/A3 as the headline legs** — A1 (correct-basin) is the leg the wave was
   missing; pass 2's arm A sat at **58/62/62 of 64 reads landing in no basin**, digit-identical to
   pass 1, while every capture metric moved.

## FILE OWNERSHIP (declared)

**You own:** `chlu/experiments/exp_capture_strong_phi.py` (**new**) ·
`tests/test_capture_strong_phi.py` (**new**) · `chlu/config.py` (**additive only**).
⛔ **READ-ONLY / DO NOT MODIFY:** `chlu/core/well_lifecycle.py` + `exp_well_lifecycle.py` (**wt1's
G-ADDR — the instrument must be identical across arms or the measurement is not a measurement**) ·
`chlu/experiments/{exp_phi_read_in,phi_encoders}.py` (**wt2's mapping**) ·
`chlu/core/{emission_head,memory_potentials,clu_system}.py` · **the live
`pilot-ttt-nan-and-d5-wiring` spoke's territory: `scripts/csf3/`, `train_cluformer.py`, `blocks.py`,
`exp_cluformer_pilot.py`** · the C2W6/C2W7 files.
⚠ Work **in your worktree**, never the shared checkout.

## Acceptance (mechanical)
1. Both preconditions verified and echoed; `geometry_go == false` carried as a **label on every
   number** if applicable.
2. Completed gate (**all four legs**) reported per seed, ≥ 3 seeds, simclr + randconv + **an internal
   PCA-φ reference arm at the same d**.
3. The branch — (a) or (b) — stated plainly, **with its pre-registered status quoted**, and **branch
   (b) reported as a finding, never as a shortfall**.
4. Byte ledger on every arm incl. launder, with projection params counted; joint dial declared.
5. D2a diagnostic reported two-sided; any best-score/lowest-drift co-occurrence flagged prominently.
6. Full suite green; count arithmetic stated **with the checkout named**.
7. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs, never nulls.

⛔ You do NOT build merge/prune/restoration or any §2.7 claim cell (still deferred — no population,
monitor #3 defect open). ⛔ You do NOT adjudicate the arm race. ⛔ Never push `origin`.
