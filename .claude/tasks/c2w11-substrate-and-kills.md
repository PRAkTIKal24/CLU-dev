# c2w11-substrate-and-kills — the repaired substrate, the feature-factored family, and every kill-condition FIRST

**Campaign 2, wave 11 (THE COMPOSITIONAL WAVE). Agent:** experiment-engineer. **ONE worktree (wt1).**
Branch **`c2w11-substrate-and-kills`** from ⭐ **`main @ 2e1cdb2` OR LATER — ⛔ NEVER pass 3's
`9e0bb25`** (see the precondition block: four C2W8-close repairs bind your measurements, and
inheriting the pass-3 versions silently scores the wrong instrument).
⚠ **The shared checkout sits on a LIVE spoke's branch** (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`,
4 commits ahead of `main`). Create with
`git worktree add ../CHLU-c2w11a -b c2w11-substrate-and-kills 2e1cdb2`. ⚠ Work **in your worktree**,
never the shared checkout, and run every command with **cwd = the worktree** (`python -m` puts cwd on
`sys.path[0]`; a launch from the main checkout dies with `No module named …`).
Writes `.claude/outputs/c2w11-substrate-and-kills.md` + artifacts to
`.claude/outputs/c2w11-substrate-and-kills/`.
**Budget:** ≈ 2 days. ⭐ **Your whole purpose is to be able to KILL THIS WAVE CHEAPLY, before the
organizer and null spokes are funded at all.** A clean kill is a full acceptance.

---

## ✅ MECHANICAL PRECONDITION — **ALREADY SATISFIED, HUB-VERIFIED ON DISK 2026-08-10**

> **`.claude/outputs/c2w8-close/GATE-HARDENING-DONE.json` → `gate_hardening_done = true`, ALL TWELVE
> items true; branch merged at `main @ 2e1cdb2`.**

The Hub verified the artifact's **content**, not merely its existence (this program has been bitten by
a gate firing on a file that exists). **Re-verify it yourself in one command and quote the per-item
table into your §1** — then proceed. If your read disagrees with this account, **say so and stop.**

## ⛔⛔ THE FOUR C2W8-CLOSE REPAIRS THAT BIND YOUR MEASUREMENTS — read `.claude/tasks/c2w8-close-gate-hardening.md` and the commits below BEFORE you set any config

**(a) A3 is DIAGNOSTIC BY CONSTRUCTION**, removed from the pass condition **in census code**
(`f80c17d`). §A33.1 is now enforced by the instrument, not only by doctrine. **G-DRIFT is now
TWO-SIDED**: `drift → 0` **FAILS** (D2a / table-expressible) instead of scoring perfectly; the floor is
`0.01 × measured codebook spacing` — **a fraction of a measured quantity, never a bare constant.**

⭐⭐ **(b) `d_safe` AND THE G-ADDR CUE ARE NOW SIZED ON THE STORE POPULATION** (`b01d474`), **and
`70b11ae` declares the arm-facing consequence — READ IT BEFORE YOU SET THE ADMISSION CONFIG. The Hub
reads it as a live trap for this spine:**
> every arm store-config factory recovers the spacing as **`d_safe / d_safe_frac`**, so **an arm's
> CO-SCALED ATOM WIDTH now co-scales to the STORE population's spacing too.** The store spacing is
> **~3× the sizing set's** (0.445 vs 0.141, MNIST).
⇒ ⛔⛔ **`atom_width_frac_spacing = 1.5` NO LONGER MEANS WHAT IT MEANT WHEN IT WAS SELECTED. DO NOT
INHERIT IT.** **RE-SELECT the width against the store population and DECLARE your selection**, with the
selection protocol registered in your `PREREG.md` before the sweep. The refuse-at-unselected-width
guard (repair (d)) then makes drift impossible. ⚠ `d_safe_population = "sizing"` reproduces banked
cells **bit-exactly** and is for **reproduction only, never a claim cell** — label any such run a
reproduction.

⭐ **(c) `covered` / `n_never_read` are SPLIT** (`dfa7f43`). ⛔ **Use `settle_covered` and the
settle-side telemetry for EVERY addressability statement.** `launch_covered` is **store-invariant by
construction** — it is what produced the Advisor's own **retracted** erratum (§A31.1: "58/62/62
unassigned, digit-identical" was vacuous) — and is retained only because monitor `settle_argmin` needs
the launch-side U for Prop D1.

**(d) The census REFUSES a non-selected width**, loudly. Also landed: **A1 emits
`margin_in_se_vs_threshold`, `n_correct_needed` and `reads_to_flip`** beside every boolean, and the
**scale guard now asserts VERDICT STABILITY** (not bounded metric movement) under **full-state
co-scaling, address AND payload** — address-only rescaling is **not a symmetry**; the payload channel
is absolute.

**Binding documents, read first, in this order:**
1. `.claude/outputs/c2w11/PREREG-C2W11.md` **IN FULL** — you implement its **§4 (K0–K7-CAP)**, **§6
   (M1, M2, M4, M5, M6)** and **§7 (the coverage half of the C2W9 trigger)**.
2. charter **ADDENDUM 12 §A33–§A34 IN FULL**, especially **§A33.1** (MECHANICS/VALUE), **§A34.1**
   (feature-factored launches; binding is the READ's job), **§A34.8** (G-ADDR is MECHANICS-only),
   **§A34.10** (the carried substrate).
3. `.claude/outputs/orgdiv-prereg/PREREG-TierII.md` **§1, §2, §3.0, §5, §7** — the ledger, the family
   construction, the pre-conditions, the sharing arithmetic, the operating point.
4. `.claude/outputs/orgdiv-cat-test.md` **IN FULL** — this is your vehicle's ancestor and its
   **registered deviations D1–D5 and its reconciliation list are your inheritance.**
5. `.claude/outputs/orgdiv-null-arms.md` **§3, §4, §12** — the mechanism, the decodability ceiling,
   and the K0 proposal you are now registering.
6. `.claude/outputs/c2w8p2-compact-atoms.md` §2–§3 and `.claude/outputs/c2w8p3-capture-strong-phi.md`
   §2–§3 — the placing write, co-scaled widths, and what "not inert ≠ addressable" looks like.

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **TIER ii substrate + kill-conditions.** ⛔ **Every leg you run is MECHANICS.**
  ⛔ No VALUE leg, no `OD`/`OD_min`, no organizer swap, no paper number, no tier-ii verdict, no
  full-CLU verdict. **You do not score the physics arm's value; you decide whether it is worth
  scoring.**
- **Laundering control:** ⛔ **launder margins are DIAGNOSTIC and can never fail one of your legs**
  (§A33.1). Report the **launch-only launder**, the **settle-deleted launder** (inherited tier-i
  diagnostic) and the byte ledger beside every reading, all labelled DIAGNOSTIC.
- **Falsifies:** K0 below its bar, or K1 unsatisfiable at any affordable `a`, or K2/K3/K4 failing —
  **any of these stops the wave here.** A leg that cannot fail its designed negative does not ship.
- **Does NOT falsify:** losing to a table on SEEN queries (Thm O1/D2a); a dividend ≈ 0 on the
  inherited tier-i launder (that is CM-27(b) by design at tier ii).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94 discipline on every reading.
- ⛔ **Wells are never named semantically.** Copy `PREREG-TierII.md` §2.6's sentence verbatim into
  your report and obey it in code comments and figure captions too.

---

## WHAT YOU BUILD

### (1) The repaired substrate on the factored store
Extend `chlu/core/factored_store.py` **additively** with the C2W8 repairs, each behind a flag whose
OFF path is **bit-identical and parameter-count-identical** (the K6/K2-fingerprint pattern):
- ⭐ **the PLACING write** — atoms are **placed**, not dragged. Banked basis: a 300-step gradient
  write drags atoms across the ball and displaced atoms become everyone else's background
  (foreign > own on **45/48**); a write that *places* gives foreign > own on **0/48**. C2W5's cat test
  used the gradient write, and its `S_eff` concentration and its −0.109 occupancy dividend are the
  symptoms you are testing against.
- ⭐ **co-scaled widths, RE-SELECTED** — co-scaled to **each seed's own measured spacing**, never
  hardcoded. ⛔⛔ **DO NOT inherit `atom_width_frac_spacing = 1.5`** — per repair (b) above, the
  spacing that value co-scales against **has changed by ~3×**, so the banked number is no longer a
  selected value. **Sweep, select against the STORE population, and declare the selection in your
  `PREREG.md` before the sweep runs.** The census then refuses any other width.
- **`atom_site_local_init`** — carried; compliance is settled (R3 `attractor_can_move_off_the_key =
  true`, follow-fraction ≈1.008 at δ=0.30).
- **the effective-`s` estimator** — fit `A e^{−r²/2s²}` to each written well's radial profile,
  ⛔ **with `α‖q‖²` subtracted analytically** (1.44× inflation otherwise). **The operating point is set
  on MEASURED `s`**, and every `d/s` you report is the measured one.

### (2) ⭐⭐ FEATURE-FACTORED LAUNCHES (§A34.1 — the wave's structural change)
**One particle per semantic feature channel of φ.** `k` is **structured by the encoder's
decomposition, not free**. ⛔ **You build NO binding structure** — binding is the READ + ψ's job and
belongs to the organizer spoke. You build: the channel decomposition of φ, the per-channel launch
head, and the launch geometry instrumentation. **The address block is its own head** (§A31.4: task
features ≠ address features; cheap conv-class address geometry is a legitimate default and the
task-strong encoder was measured **address-worst** beyond 2 SE).

### (3) The family
`N_a` shared feature wells · `F`-subsets · payload `v_j ∈ R^m` **existing only in the store** ·
`y(x) = Σ_{j∈A(x)} v_j` (registered primary). ⚠ **Inherit C2W5's forced deviations and re-verify them
rather than assuming them:** `m ≥ 8` (the K2 payload half passes on **0.5 %** of queries at the
registered `m = 1`; the sweep was 1/2/4/6/8/12 → 0.005/0.119/0.802/0.987/1.000/1.000) · unit-norm
`v_j` on `S^{m−1}` · `atom_payload_init_radius = 1.0` (a **designed mechanism costing 0 parameters**;
without it every well relaxed to the ORIGIN at `λ_min = 2α` exactly).

### (4) ⛔⛔ THE KILL-CONDITIONS — build them FIRST, run them FIRST, in the PREREG's order
Implement **K0, K1, K2, K3, K4, K5, K6, K7-CAP, K8** exactly as specified in `PREREG-C2W11.md` §4.
**Run order is binding: K0 → K7-CAP/K6 → K1 → K2 → K3 → K4 → K5 → K8.**

> ⭐⭐ **K0 AND M6 ARE THE CHEAPEST KILL SIGNAL IN THE WAVE. RUN THEM FIRST AND REPORT THEM BEFORE
> ANYTHING ELSE** — K0 needs no store and costs seconds; M6 needs only a written store and the launch
> geometry. **Report both in your first screen, before the rest of the K-table.**
> ⭐ **And if they do not move, that is a RESULT, not a wasted wave:** if the structural caps are
> unmoved despite **three measured substrate changes** (placing write · co-scaled widths ·
> feature-factored launches), that is the **FIFTH convergent datum on write-side organization — this
> time with the substrate repairs CONTROLLED FOR.** Write it up as the finding it is.

⭐ **K8 — the `K < N_a` STRUCTURAL CELL (new, Amendment 1). YOU CONSTRUCT AND FREEZE IT; spoke B runs
the confirmatory V1 score on it.** Build the headline family a second time at **`K < N_a`** (e.g.
`N_a = 32, F = 4, K = 24`), assert the rule-4 split exists at that size, and **assert the SP-1 design
matrix is RANK-DEFICIENT there** — i.e. the linear-code probe **provably cannot** recover `v` (verified
at C2W5's `K = 12 < N_a = 16` fixture: it reproduces `y` **without** recovering the payloads).
**Freeze it into the interfaces JSON as `k8_structural_split`.** ⛔ **One cell, headline configuration
only — not across the grid.** Rationale to carry: *a measured guard tells you the leak is small at
this operating point; a structural impossibility tells you it cannot happen*, and **ψ-does-the-work is
this wave's most likely false-positive mode**.

Note specifically:
- **K0** is the leg C2W5 never had: at `P = 4` designed offsets, `≥ F` distinct wells were reachable
  on **5.0 %** of queries and exact-set occupancy was **0.0000 / 2 560** — *a cap that existed before
  any store was written.* Report the **full distribution** of distinct wells reachable, not just the
  mean, and report it **per feature channel**.
- **K4 is re-specified and this matters:** run all four leak controls against the **FULL trained read
  path including ψ at full capacity and the novelty head**, with the store blanked. ⚠ You do not own
  ψ — so run K4 in two forms: (a) the store-only form you can build now, **blocking**; and (b) emit a
  **frozen K4 harness** the organizer spoke must re-run at full ψ, with the assertion baked in.
  **Name that obligation in `FROZEN-INTERFACES-C2W11.json`.**
- **K6 — you own it, and it closes a FIVE-SESSION slip.** One line, computed **before any reader is
  fitted**: the fraction of queries whose asserted set is already exactly right. Report it beside
  every fitted-reader score. Reference fractions: **2/2560 · 3/1280 · 0/2560** (C2W5's cells) vs
  **~18 %** (C2W7's).
- **K7-CAP** — assert every reader's params `< N_a·m`. Banked measurements: `sum_linear` 104 ·
  `well_table` 72 · `knn` 0 · `mlp` 92, against a bound of 256 at `(N_a, m) = (32, 8)`. ⭐ **And run
  the SP-1 probe itself** (OLS on the true indicator, `N_a·m` dof, fitted on SEEN): C5 measured
  **1.0000** exact-set with `‖v̂−v‖∞ = 4.25e-15` **on a blank store**. Report it as a **declared
  out-of-class diagnostic**, never as an arm and never as a K4 leg.
- ⛔ **A zero-parameter reader is a MANDATORY member of the class** (§A26.3). ⚠ Banked: the identity
  reader was **strictly worse** than the fitted one at C2W5's cell (0.0000 vs 0.00078) — it is
  **added to** a class, never substituted for it.

### (5) The MECHANICS legs you own: M1 (=K0) · M2 (=K1) · M4 (sharing/refresh) · M5 (anti-collapse) · M6 (DIAGNOSTIC)
Each with the **designed negative from `PREREG-C2W11.md` §6, pytest-asserted**.
- **M5's instrument is direct wells-visited `W/N_a`, TWO-SIDED** (§A26.4). ⛔ **The `S_eff ∈ [8,16]`
  band is RETIRED** — its lower half is unreachable by construction (`S_eff = K·F/W` with `W ≤ N_a`).
  "COLLAPSED" is reserved for **concentration**; under-usage is labelled under-usage. A run outside
  the declared band is reported **COLLAPSED, not null** (C5's 15/10/14 of 32 wells ever occupied is
  the reference concentration failure).
- **M6 is a DIAGNOSTIC and cannot fail a gate** — but it is the wave's most informative single
  reading. Report **occupancy precision of the raw launch geometry vs after the settle, with sign and
  2 SE**, and the **distinct-wells-occupied pair**. Banked C5: **0.4061 → 0.2967 (dividend −0.1094)**
  and **2.20 → 1.70**. ⛔ **Occupancy precision is scored against the BLANK STORE / raw launch
  geometry, never against `F/N_a`** (C5 reconciliation 4: the store was above chance 0.297 vs 0.125
  *and simultaneously below its own launder* 0.406).

### (6) ⭐ THE C2W9 COVERAGE TRIGGER (you own the coverage half — §7 of the prereg)
Instrument, per query and per feature channel: **is the needed feature well inside the union of the
`k` launch diamonds?** ⛔ **If coverage failure fires at your registered threshold, WRITE
`.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md`** with the measured signature (mode, fraction
of queries, per-feature breakdown, reach radii, seeds). ⛔ **If it does not fire, do NOT create the
file** and say so explicitly. Register the threshold **before** the run.

---

## ⭐⭐ DELIVERABLE — the mechanical gate the other two spokes wait on

> **`.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json`**

⚠ **Write EXACTLY that path.** (C2W8 pass 3's standing lesson: *a mechanical precondition must name
the path the spoke ACTUALLY writes.*) It must carry:
1. `kills_all_passed` — computed **mechanically** as the AND over K0–K7-CAP, each with its own
   boolean, measured value and bar. ⛔ Anything not landed is `false` with its reason.
2. The **frozen family**: `(N_a, F, K, m, a, d)`, the measured `s` and `d/s`, `tol`, `chance`, the
   depth-heterogeneity ratio, `γ_address`/`γ_read`, the read budget in steps.
3. The **frozen launch protocol**: the feature-channel decomposition, `k`, the per-channel launch
   rule, and the **launch keys**, so the null arms are **bit-identical** on launches (assert it).
4. The **frozen φ**: instance, `phi_bytes`, and a byte-comparison hash. Identical on every arm
   (`PhiMismatchError` precedent).
5. The **frozen reader class**: the 4 readers **plus the zero-parameter member**, their measured
   param counts, the fitting protocol and the seen-validation split rule (⚠ the split must inherit
   the family's **own** rule-4 held-out rule).
6. The **byte ledger** template: store / φ / projection / reader params / state, per arm, two-sided.
7. `k4_full_psi_obligation` — the frozen K4 harness the organizer spoke must re-run at full ψ.
8. `coverage_trigger_fired: true|false`.
9. ⭐ `k8_structural_split` — the frozen `K < N_a` family, its rule-4 verification, and the asserted
   SP-1 rank-deficiency, so spoke B can score V1 on it without re-deriving anything.
10. `selected_atom_width` + the selection protocol + the **store-population spacing it was selected
    against** — so no downstream reader can attribute a number to the wrong width (repair (b)).
11. ⭐⭐ `v3_budget_grid` — **the frozen read-compute budget points (≥ 6) for the anytime curve**, on
    the shared ledger (particles-evolved × Verlet steps). **Spokes B and C BOTH score V3 on this exact
    grid and they run concurrently, so only you can freeze it.** ⛔ A mismatched axis **voids VALUE leg
    iii**; this field is the single point of coordination between two spokes that never talk.
12. `store_population_spacing` per seed, and the `σ_q / store_spacing` ratio — the quantity the K0
    prior was re-derived against (measured 0.19–0.37 on the repaired sizing, vs the ~1.07 the retired
    sizing-set number implied).

⛔ **Anything the other spokes need and cannot re-derive from this file is a defect in this file.**
⚠ **Do not repeat `FROZEN-interfaces.md`'s C2W5 failure:** its matched-capacity ledger row and its
reader param counts were **both wrong** (`384·14 = 21 504 B` at `a=12` against the `a=32` cell that
actually ran; `well_table = 16`/`mlp = 88` against the shipped 72/92). **Emit every ledger number
from the code that computes it, never from a doc.**

## FILE OWNERSHIP (declared)
**You own:** `chlu/core/factored_store.py` (**additive**) · **new** `chlu/core/feature_launch.py` ·
**new** `chlu/experiments/exp_c2w11_substrate.py` · **new** `tests/test_c2w11_substrate.py` ·
`tests/test_factored_store.py` (**additive**) · `chlu/cli/experiment_cmd.py`.
⭐ **You own `experiment_cmd.py` EXCLUSIVELY for this wave and you land ALL THREE C2W11 subcommand
stubs in your first commit** (`exp-c2w11-substrate`, `exp-c2w11-organizer`, `exp-c2w11-nulls`),
additively, before any existing line. **The other two C2W11 spokes touch that file not at all.** This
is the wave's conflict-elimination measure — do not skip it.
⛔ **DO NOT TOUCH — the live CSF3 pilot spoke's territory:** `scripts/csf3/` ·
`chlu/training/train_cluformer.py` · `chlu/core/blocks.py` · `chlu/experiments/exp_cluformer_pilot.py`.
⛔ **DO NOT TOUCH — C2W8-close's territory (READ-ONLY all wave):** `chlu/core/well_lifecycle.py` ·
`chlu/experiments/exp_well_lifecycle.py` · `chlu/core/clu_system.py` · `chlu/core/soft_certificate.py` ·
`tests/test_gate_addr.py` · `tests/test_well_lifecycle.py` · `tests/test_cifar_strong_phi.py`.
⛔ **DO NOT TOUCH:** `chlu/config.py` (**zero C2W11 spoke touches it** — configs live next to their
code, the `CatTestConfig`/`SoftCertificateConfig` precedent) · ⭐ **`chlu/core/monitors.py`** —
**IMPORT READ-ONLY.** M15 (marginal well-usage) already exists there and you consume it for M5.
⛔ **Any NEW monitor row you need goes INSIDE your own module, not appended to the registry** — the
concurrent **C2W10 `c2w10-lifecycle-mechanics`** spoke is appending its `protected_saturation` row
there, and two waves appending to one registry is the `config.py` adjacency hazard wearing new
clothes. Say in your report which route you took · `chlu/core/null_arms.py` ·
`chlu/experiments/exp_null_arms.py` (spoke C's) · `chlu/core/psi_readout.py` (spoke B's) ·
`chlu/core/emission_head.py` · `chlu/experiments/exp_capture_strong_phi.py` (banked arms).
⚠ Anything you need changed outside your ownership is a **rider handed back to the Hub**, never an
edit.

## Acceptance (mechanical)
1. `FROZEN-INTERFACES-C2W11.json` exists at exactly that path with a mechanically computed
   `kills_all_passed` and all eight blocks above.
2. **Every kill-condition K0–K7-CAP stated with its bar and its measured value, multi-seed**, in a
   first-screen table. ⚠ **A vacuous pass is reported as vacuous** (C5's K3/K4 "passed" only because
   every number in the cell was ≈ 0 — say so where it can't be missed).
3. **Designed negatives pytest-asserted for M1, M2, M4, M5** (§6 of the prereg).
4. The harness **refuses** to run at a non-selected width, and the refusal is pytest-asserted.
5. Every `d/s` is on **measured** `s` with `α‖q‖²` subtracted; the estimator's `R²` reported.
6. Coverage trigger: the file written **or** its non-firing stated explicitly.
7. Full suite green on your branch, **count arithmetic stated with the checkout named** (⚠ counts are
   comparable only within one checkout; state the base's selected count measured in **your** worktree).
8. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs, never nulls; registered
   deviations argued, never silent.
9. A `PREREG.md` under `.claude/outputs/c2w11-substrate-and-kills/`, filed **before your first cell**,
   with your own numeric predictions for K0–K7-CAP and M1–M6, and a scorecard against it at the end.

⛔ You do NOT build ψ, the novelty head, the organization loss, or any null arm.
⛔ You do NOT compute `OD`, `OD_min`, or any VALUE leg. ⛔ You do NOT adjudicate anything.
⛔ Never push `origin`; the Hub integrates. `clu-dev` only, and only the Hub pushes it.
