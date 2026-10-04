# hub-composition — autonomous-hub report
**Task + acceptance criterion:** a compositional-generation win — held-out combinations rendered
correctly and read by the external classifier, beating every rival incl. `knn_proto`, at
**≥ 60 / 72 distinct valid novel tuples**, with dreaming inside the target.
**Status:** ⭐ **done — target NOT MET, and measured to be unreachable through the pinned store.**
⛔ Reported as a well-controlled failure that names the obstacle with a number, which the charter states
is the expected outcome when the science does not land.

**DIAL DECLARATION (protocol §7)** — *compositional generation, end to end, scored on IMAGES.*
- **Dial:** compute-adaptive reads / composition through the store.
- **Laundering control:** `store_removed` on every row (⭐ PASSES: 1 distinct tuple, balacc 0.5000);
  plus an oracle-selection control and a landscape-removed control on the addressed read.
- **Falsifies:** failing to reach 60/72 valid novel tuples, or losing to `knn_proto`.
- **Does NOT falsify:** losing to an unbounded-channel arithmetic baseline (metric-native-ceiling
  theorem), or absolute levels below the decoder ceiling.

## What I did
- Read the boot chain, applied LAUNCH-COMMON's three corrections, branched from `HEAD` (⛔ not the
  non-existent `ae907466`), quoted the override, logged as I went.
- Cleared two environment blockers **without mutating anything shared** (h5py to a private `--target`;
  Shapes3D to a private gen-data root with the bundled files hardlinked in).
- Found and **reported rather than crossed** the hard constraint: seeds 1–2 cannot run here.
- Built and validated a VALID∧NOVEL instrument; re-derived the frozen Mac table; ran 16 non-scored
  mechanism experiments on `val`/`MS`/frozen artefacts; pre-registered the one scored run I would spend;
  had the headline numbers independently replicated by a separate agent.

## How I verified
All commands, outputs and raw numbers are in `hub-composition/LOG.md` (dated bullets, written as I went)
and `hub-composition/REPORT.md`. Key verifications: my driver reproduces the frozen harness's `clu` row
on MS exactly (distinct 32, n 198); my instrument reproduces the harness's `distinct` and `exact_match`
on **24/24** frozen rows; an independent agent re-derived 5 of 7 headline numbers **exactly**, 1 within
draw noise, and **REFUTED 1**, which I re-ran over 25 draws and corrected.

## Findings (the substance)
1. ⛔ **Falsifier 1 NOT MET** — CLU arm **0.00** vs `knn_proto` **14.33 ± 4.78** distinct valid novel
   tuples of 72. **Falsifier 2 NOT MET** — bar is 60; decoder ceiling is **67.67 ± 2.49**.
   ⭐ Both settled from **frozen artefacts with 0 scored runs**, via the identity
   `distinct_valid_novel = 72 × exact_match` on B (all 72 targets distinct; all 72 absent from every one
   of the 479,928 train rows).
2. ⭐ **Falsifier 3: dreaming is NOT DEAD but INERT.** At the pre-registered T = 0.5, β = 0 yields 463
   distinct / 447 novel / **4 valid novel** of 512 — ⛔ but the zero-step launch yields the same, and
   β = 0 ≡ β = 1 to four decimals in settled separation at every grid temperature.
3. ⭐ **Falsifier 4 MET** — `store_removed` collapses to 1 tuple, balacc exactly 0.5000.
4. ⭐⭐ **The obstacle, isolated:** not composition (a surrogate held-out cell has the *same* rel-err as
   seen data), not addressing (an oracle buys +2 of 512), not the metric (my predicted whitening remedy
   **failed**) — it is **self-alignment of the quantiser's residual**: at one fixed rel-err of 0.3486,
   `x − e` scores **162** and `x + e` scores **10**.
5. ⭐⭐ **The requirement:** `rel-err = e^1.346 · bytes^(−0.1651)` (R² = 0.9960 over a 128× span) ⇒ the bar
   needs **≈ 77 GB, ~147,000×** the deployed 524,288 B store; merely matching `knn_proto` needs **162×**.
   Even an exponent twice as favourable as any measured leaves the bar at **383×**.
6. ⭐ Useful side results: the **partition** is a real lever at constant bytes (K 5→64 lifted untrained
   round-trip 6→47 of 512); `exp_gen_cell` has **never run the pinned `x0` addressed read**.

## Git footprint
Branch `hub/composition` at `8e36bc9` (tree `345244bb…` = `pin_tree`). ⭐ **ZERO commits, zero tracked
files changed.** ⛔ I did not edit the frozen scoring harness, and the one change that would have let
`exp_gen_cell` run here is raised as a question, not made.
Bundle: `hub-composition/hub-composition.bundle`, 3,174,623 B, whole-branch, "records a complete
history", sha256 `9c6895072e2f7295cdc475565198ae2853e5ea88fd5634269add1d060a04ffd4`.

## ⛔ Open questions for the Head — BOTH STILL UNANSWERED
1. ⛔ **How many of the 3 scored runs for `composition` remain across all machines?** Asked at boot.
   ⭐ Moot in practice: I spent **0** and recommend **0** be spent here.
2. ⛔ **May `will_extend=(shortfall > 0)` be wired at `exp_gen_cell.run_seed`'s re-encode-gate call?**
   Without it `exp_gen_cell` cannot run on this box **at all**, even on seed 0 where the shortfall is
   provably 0. ⛔ It is a frozen-harness edit, so I did not make it.
3. ⛔ **Seeds 1 and 2 are unrunnable here** (32,395 / 32,437 of 33,039 rows uncached). Either ship
   `scope="full"` caches for them, or run `PREREG-hub-composition-1.md` on the Mac at 3 seeds.
4. ⚠ **The WEAK reading of "valid" is unmeasurable for the rivals** — `exp_gen_cell` persists per-render
   tuples only for `dynread`. PREREG Amendment 1 asks the scored run to persist them for every arm.

## Proposed handover updates (for the Hub)
- **Composition, generation table:** the CLU arm's distinct-valid-novel count on B is **0**, `knn_proto`'s
  is **14.33 ± 4.78**, the decoder ceiling is **67.67 ± 2.49**. The ≥ 60/72 target is equivalent to
  `exact_match ≥ 0.8333` on B and is **~147,000× of store bytes away**. Recommend the target be retired
  or re-scoped against the measured capacity law.
- **New declared property:** on B, `distinct_valid_novel = 72 × exact_match` (all 72 targets distinct and
  absent from training) — so the headline column needs no new run.
- **New declared property:** novelty does **not** bind on this store (24/24 frozen rows had
  `distinct_novel == distinct`); "N novel tuples" here means "N distinct tuples" and should say so.
- **Mechanism:** the quantiser's residual is self-aligned; re-weighting the metric cannot help. Any
  future read whose output is a stored object inherits this.
- **Owner needed:** `exp_gen_cell.run_seed`'s unscoped re-encode gate blocks every non-Mac machine.
