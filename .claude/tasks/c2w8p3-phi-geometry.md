# c2w8p3-phi-geometry — build the φ_dim → addr_dim map, then MEASURE whether strong φ actually separates

**Campaign 2, wave C2W8 PASS 3. Agent:** experiment-engineer.
**Worktree 2 of ≤3.** Branch **`c2w8p3-phi-geometry`** from `main @ 1eda6a0`, worktree
`../CHLU-c2w8p3b`. ⚠ **Name the base explicitly** — the shared checkout is on another spoke's branch:
`git worktree add ../CHLU-c2w8p3b -b c2w8p3-phi-geometry main`.
Writes `.claude/outputs/c2w8p3-phi-geometry.md` + artifacts to `.claude/outputs/c2w8p3-phi-geometry/`.
**Budget:** ≈ 0.75 day. **Spawns in parallel with wt1** (zero file overlap, declared below).
⭐ **Cheap and decisive: it decides whether the spine's substrate can separate at all, BEFORE the
spine spends a cell.**

**Binding documents:** `.claude/outputs/c2w8-well-lifecycle/ERRATA-C2W8-PASS3.md` **§1 — the four Head rulings (R1 CIFAR-10 substrate · R2 projection in scope + launder uses it · R3 the d rule · R4 NO-GO re-labels). READ BEFORE THE PREREG.** · `.claude/outputs/c2w8-well-lifecycle/PREREG-C2W8-PASS3.md` — **§3 (the mapping
defect, yours), §4 (scale-invariance guard), §5 (your deliverable + its registered GO/NO-GO reading),
§8 prediction Q3** · charter **ADDENDUM 10 §A29.5 + the §A30 standing block** · **§A4.3** (`(d, atom
budget)` is ONE joint dial; identical φ for CLU / baselines / launder) ·
`.claude/outputs/c2w8-cifar-strong-phi.md` + its `results/` (the built, priced encoders) ·
`ERRATA-C2W8.md` §3 (the banked reach arithmetic).

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — **encoder/geometry instrumentation.** ⛔ No claim cell, no performance number, no
  verdict of any kind.
- **Laundering control:** N/A (no performance claim) — **but the byte ledger is mandatory**, because
  your projection's parameters land on **every** arm's ledger **including the launder**.
- **Falsifies:** nothing this pass claims; your NO-GO **re-labels** the spine, it does not cancel it.
- ⛔ N94: any reading here is labelled non-promotable.

---

## ⛔⛔ THE DEFECT YOU ARE FIXING — the mapping does not exist (Hub finding)

`exp_well_lifecycle.PhiAddress` **forces `phi_dim = addr_dim`** (`cl.phi_dim = int(w.addr_dim)`) and
then **TRUNCATES**: `out[:, :addr_dim] = f[:, :addr_dim] * scale`. **There is no projection.**

⇒ "strong φ at `addr_dim = 8`" would today be **either** a *weak 8-dim simclr* (refit at d=8 — not
the encoder that was built and priced) **or** *8 of 256 coordinates* (discarding 248 dims).
⛔ **Neither is the strong φ that measured 0.161 → 0.319.** The spine cannot run honestly until this
is built.

**Feasibility is hard-bounded by the atom law** `min_atoms = round(512·√2^d)`:

| d | 8 | 12 | 16 | 20 | 24 | 256 |
|---|---|---|---|---|---|---|
| atoms | 8 192 | 32 768 | **131 072** | 524 288 | 2 097 152 | **1.7e41 — IMPOSSIBLE** |

⇒ **naive d = 256 addressing is forbidden**; the feasible band is **d ∈ {8, 12, 16}**.

## 0 — ⛔ THE SUBSTRATE IS FIXED BY HEAD RULING R1: **SPLIT-CIFAR-10**

Your geometry is measured on **CIFAR-10 φ**, not MNIST — the spine runs there because that is where
the encoders were **built and priced** (simclr `enc_steps = 8000`, 0.16080 → 0.31912). ⇒ set
`dataset = "cifar10"` and measure **the PCA-φ reference at matching `d` on CIFAR-10 too**.
⛔ **Pass-1/pass-2 census geometry is MNIST and is NOT your baseline** — your PCA reference must be
measured by you, on this substrate, in this run.

## 1 — Build and DECLARE the map

⭐ **HEAD RULING R2: this is IN SCOPE and you BUILD it.** A genuine **projection or read-in head**
`φ(256) → addr_dim`. Its form is yours; **declare it**. ⛔ **R2(b): the launder reads the PROJECTED φ —
ASSERT IT IN CODE, do not merely intend it.**
- ⛔ **Its parameters go on the byte ledger of EVERY arm INCLUDING THE LAUNDER.**
- ⛔⛔ **The launder must use the SAME projected φ, never the 256-dim φ.** A launder reading 256 dims
  while the store reads 8 is not a launder — it is a handicap match. (Fairness invariant §A4.3:
  identical φ for CLU, baselines and launder.)
- **`(d, atom budget)` is ONE declared joint dial** — state both together in every table.
- ⚠ If the projection is **fitted**, it must be fitted under the **`task1_only`** regime like every
  other φ in this programme (no leakage from unseen tasks), and you must say what it was fitted on.

## 2 — MEASURE the geometry (the point of this spoke), ≥ 3 seeds, per candidate d

⚠ **DECLARED RISK (R1):** `CluSystem`'s learned `V_θ` has **never** been run on CIFAR φ, and pass 1
measured it **inert at `d ≥ 16` on MNIST**. **Measure well depth FIRST and report an inert store in
your first 10 lines** — an inert store makes the measurement vacuous for a reason that is **not**
your GO/NO-GO reason, and the two must never be conflated.

On the projected φ, report per seed and per `d ∈ {8, 12, 16}`:
**median-NN key spacing** · ⭐ **σ_q / spacing** · **`d_safe` / spacing** · achieved atom budget.
**Against the PCA reference at the same `d`** — the comparison is strong-φ vs PCA-φ **at equal d**,
never strong-φ-at-one-d vs PCA-at-another.

⭐ **Why this is the whole question (prereg §4):** the rig normalises addresses to unit radius
(`scale = 1/r95`) while **σ_q = 0.15 is absolute**. With `n` items in a unit `d`-ball the spacing is
**essentially geometric** — a property of `(n, d)` and of how uniformly φ fills the ball. **Strong φ
cannot enlarge the volume; it can only spread items more uniformly inside it.** That is exactly what
you are measuring.
⚠ **Report every quantity as a DIMENSIONLESS RATIO with the scale stated** — an absolute-units leg
can be moved by rescaling φ with zero information gain.

## 3 — The registered GO / NO-GO reading (mechanical, computed, not argued)

> **GO** iff strong φ improves **σ_q / spacing** over the PCA reference **at the same `d`**, beyond
> noise, ≥ 3 seeds. **NO-GO** otherwise. (Registered prior **Q3 = 0.55**.)

⭐ **HEAD RULING R4: a NO-GO RE-LABELS the spine, it does NOT block it** — and ⛔ **the FILE ITSELF is
still a hard precondition; only its verdict is soft.** ⭐ **HEAD RULING R3: the spine takes the `d`
your measurement favours** — so measure all three of d ∈ {8, 12, 16} and **state which you favour and
why**; the rider (a) cell runs at **d = 16**.

⛔ **A NO-GO does NOT cancel the spine — it RE-LABELS it.** The spine then measures the physics at a
substrate *known in advance* not to have fixed separability, and its null becomes **attributable
rather than confounded**. ⭐ **Without this file, a pass-3 null would reproduce pass 2's null for a
different reason and be misattributed** — which is the error this whole wave keeps catching late.

## 4 — Two riders

**(a) The REVIVED (d, atom-budget) rider — ONE CELL, NOT A SWEEP.** *Was the banked `d ≥ 16`
inertness measured with the co-scaling honoured?* The Hub established from banked evidence that it
**was** (`ERRATA-C2W8.md` §3's `n_atoms` column matches `round(512·√2^d)` exactly at 4/8/12/16), so
the binding constraint is **REACH, not capacity** — but that came from a **scratch probe of 3
designed-site writes**, not a censused cell. **Run ONE confirming cell at `d = 16`** (Head ruling R3), price it before running, and if it will not fit, **say so and
report the arithmetic instead of a truncated run.**

**(b) Re-price `d_safe` so monitor #3 can go quiet HONESTLY.** Pass 1/2: refusal rate **0.000** at
`d_safe ≈ 0.12` vs spacing ≈ 0.14 — the admission gate **refused nothing**. At strong φ, re-price it
so a quiet monitor #3 means *"nothing needed refusing"* and not *"the gate cannot fire"*. ⛔ **Report
the refusal rate and the achieved `d_safe`/spacing ratio; do not tune the rate to a target.**

## 4c — HOUSEKEEPING RIDER (small, yours because it is CIFAR-φ territory)

⚠ **`tests/test_cifar_strong_phi.py:66-72` carries a STALE 6-line comment** saying the `"cnn"`
backbone cannot be exercised because of an x64 dtype bug. **That bug is FIXED** (pass-2 wt3 closed it
with a function-scoped fixture and x64-off bit-identity). ⛔ **The `backbone = "mlp"` choice may
stay** (it is a cost decision) — **but the stated REASON must change**, or the next reader inherits a
false constraint. Correct the comment; do not change the test's behaviour.

## 5 — Deliverable (the spine's second mechanical precondition)

> **`.claude/outputs/c2w8p3-phi-geometry/PHI-GEOMETRY.json`**

Carrying: the declared mapping + its param count and bytes · per-seed per-d spacing, σ_q/spacing,
d_safe/spacing, atom budget · the PCA reference at matching d · the rider (a) cell or its priced
refusal · the rider (b) refusal rate · and a top-level **`geometry_go`** boolean computed
**mechanically** by §3's rule.

## FILE OWNERSHIP (declared)

**You own:** `chlu/experiments/exp_phi_read_in.py` · `chlu/experiments/phi_encoders.py` ·
`chlu/experiments/exp_cl_entry.py` (**read-mostly**; additive only if the stream needs a projected-φ
hook) · `chlu/experiments/exp_phi_geometry.py` (**new**) · `tests/test_phi_geometry.py` (**new**) · `tests/test_cifar_strong_phi.py` (**comment only, rider 4c**) ·
`chlu/config.py` (**additive only** — ⚠ wt1 also appends there; keep to a delimited block, the Hub
resolves adjacency at merge, and **that adjacency conflict has now fired twice — expect it**).
⛔ **DO NOT TOUCH:** `chlu/core/well_lifecycle.py` + `chlu/experiments/exp_well_lifecycle.py`
(**wt1's this pass**) · `chlu/core/{emission_head,memory_potentials,clu_system}.py` · **the live
`pilot-ttt-nan-and-d5-wiring` spoke's territory: `scripts/csf3/`, `train_cluformer.py`, `blocks.py`,
`exp_cluformer_pilot.py`** · the C2W6/C2W7 files.
⚠ Work **in your worktree**, never the shared checkout.

## Acceptance (mechanical)
1. `PHI-GEOMETRY.json` exists with a **mechanically computed** `geometry_go`.
2. The mapping is **built, declared, and its params are on the byte ledger of every arm including the
   launder**; the launder demonstrably uses the **projected** φ (assert it).
3. All geometry reported as **dimensionless ratios with the scale stated**, ≥ 3 seeds, PCA reference
   at **matching d**.
4. Rider (a) — one cell or a priced, declared refusal. Rider (b) — refusal rate + `d_safe`/spacing.
5. Full suite green; count arithmetic stated **with the checkout named**.
6. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs, never nulls.

⛔ Never push `origin`; the Hub integrates and pushes `clu-dev`.
