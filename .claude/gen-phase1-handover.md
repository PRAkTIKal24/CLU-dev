# GEN PHASE 1 — HUB HANDOVER. Rewritten 2026-09-15. ⛔ Read this, then stop and wait for the Head.

**Boot, in this order and NOTHING else:**
1. `.claude/iclr-long-hub.md` — your role. Short by design.
2. **this file** — the whole live state.
3. `.claude/outputs/gen-dynamical-read/PREREG.md` — **the document you are here to get signed.**
4. `.claude/outputs/design-notes/dynamical-read.md` — the mechanism argument, **only when you need §0.5's law.**

⛔ **DO NOT read `.claude/tasks/gen-clu-phase1-rulings.md` end to end. It is 2,100+ lines (R1→R48) and will eat your
context.** ⭐ **Everything still binding is restated below.** Open it only to look up ONE numbered ruling when a
decision needs it. ⛔ No campaign charter, no old handover — different programme.
**The one exception, for you and every spoke: `.claude/handover_context.md` §7 Known Issues only.**

---

## 1. Where things stand
- **`main @ 3eb9eb0`** in the worktree `/Users/user/Desktop/CHLU-merge`. ⚠ The primary checkout
  `~/Desktop/CHLU` is on a merged side branch — it is where `.claude/` lives, so **write notes there, merge in
  `CHLU-merge`**. ⛔ **`.claude/` exists ONLY in the primary checkout** — a heredoc run from the worktree fails silently.
- **Nothing in flight.** Waves 6–10 all merged and reviewed.
- **Backups:** ⚠ `clu-dev` is **~20 commits stale**; last internal-state snapshot 2026-09-14. Use
  `.claude/fork_push_runbook.sh` ⛔ **NOT as written** — see §5.
- **Cluster:** smoke is ⭐ **GREEN** (`.claude/outputs/gen-cluster-smoke/smoke_result.json`). JAX initialises the A100.

## 2. ⭐ THE ONE THING TO DO NEXT
**Get `PREREG.md` (at REVISION 2) corrected and signed, then the Head launches the scored dynamical-read run.**
**Sign-off order (Head ruling): split fix ✅ done → corrected PREREG → Head + Advisor signature → scored run.**
⛔ **The Hub never launches.** Write the task file, hand the Head one copy-able spawn block, one line per spoke.

### ⛔ Two things block the PREREG being finalised — both need Head/Advisor, not you
1. **`knn_proto` got WEAKER after the split repair** and the spoke correctly refused to clear it (§3 item 2).
   **A Hub/Head ruling is owed on whether that is a correction or a hobbling.** ⭐ Hub's read: `knn_proto` has **no
   tuning axis**, so this is a `fit`-row rebalancing, not a selection change — the old number was inflated by an
   unrepresentative `fit`. ⛔ But it re-bases the headline table's ceiling row, so it is not the Hub's call alone.
2. **The dSprites companion FAILED its pre-registered rule by 0.0011** (§3 item 3). ⛔ It may **not** be quietly
   re-labelled a pass. **Decide: re-scope the mean-level effect as a NEW pre-registered question, or drop it.**

## 3. ⭐ THE LIVE RESULTS — everything a decision might need
**The mechanism story, settled over waves 6–9:**
- **The encoder is CLOSED, three ways:** it carries `scale` 0.935 / `orientation` 0.991; unfreezing it at matched
  compute makes things *worse*; ⭐ **an UNTRAINED random-init encoder scores the same** (0.9361 / 0.9865) for 1,246×
  less compute. ⇒ ⛔ **"≥0.9 on seen data" does not measure the pretext.** ⭐ **Every future encoder bar is a MARGIN
  OVER THE D-MATCHED RANDOM-INIT FLOOR, never an absolute accuracy.** ⚠ The encoder question **re-opens on real data** —
  probe saturation is a property of clean synthetic 64×64 images.
- **The codebook is the culprit:** the partition is exonerated (it costs 9.5 % of `scale`, **0 %** of `orientation`);
  **k-means takes 90.5 % / 100 %.**
- **Why it compresses at all:** well features have **effective dimension `PR` = 3.12 … 13.63**, not 283–744.
- ⭐ **The exponent is ≈ −0.25 … −0.28 and only weakly `PR`-dependent** (slope `+0.0702` in `log PR`; runs
  −0.287 → −0.200 over `PR` 3 → 13). ⛔ **The law's `−2/PR` RATE is refuted — it overpredicts by ~3×.** The law
  survives as a **LEVEL** (`r = 0.905`). ⛔ **NEVER quote "6.5 million codewords" or the `−2/PR` rate.**
- **Whitening: premise true, repair self-defeating.** `scale`/`orientation` *do* sit in low-variance directions, and
  equalising them destroyed the code (total retention **3.678 → 1.172**). ⭐ **k-means' reliance on high variance IS
  the compression.**
- **D6 closed:** uniform `M = 64` is already at the byte-constrained optimum to within 1 %.

**The bars the dynamical read must clear (measured, not derived):**
| | budget A (N=64) | budget B (N=512) |
|---|---|---|
| ⭐ **atom-mean lookup, best `k` — the row that decides the claim** | **0.3567** | **0.1851** |
| required margin over it | **5.6×** | **11.4×** |
⭐ **Storing atoms instead of centres buys a WORSE level (×1.43) and a slightly better slope — they nearly cancel.**
⇒ ⛔ **a landscape beating the centre-codebook may be winning by STORAGE, not DYNAMICS. The atom-mean row separates them.**
⭐ **Deployable (request-addressed) is the HEADLINE query protocol** — its exponent is **−0.162**, because the R8
embedding's own error flattens the curve 41 %. **Oracle is a companion; ⛔ quoting the oracle is forbidden.**

**From the split repair (wave 10):**
- ⭐⭐ **The CLU arm did NOT move — bit-identical, 3/3 seeds — and every rival DID.** ⇒ ⛔ the repair cannot have
  manufactured a win for us; it re-priced the comparators only.
- ⚠ **The repair removes a BIAS but does NOT buy IDENTIFICATION.** On an exchangeable halving the two halves still
  pick different winners in **5 of 6** cells (best-to-second gaps 0.0007–0.0130 against a trial spread 0.068–0.111).
  ⇒ ⛔ **a tuned `gen` configuration may not be quoted as *identified* — only as *unbiased*.** `tune_n`/`n_val` is the
  real fix and is **unowned**.
- ⛔ **Refusal's quotable status was SUSPENDED pending the rival re-score.** ⭐ The re-score is done — **the Hub owes a
  ruling on whether it is restored.**

## 4. ⛔ BINDING WORDING — check every draft against this
- ⛔ **No draft mentions "30 bits"** for the dynamical read. Its channel is the store's **full width in floats**
  (`knn_proto` parity). The 30-bit figure was never the binding constraint (3,598 distinct codes on 4,096 rows).
- ⛔ **Composition is RELEASED as a claim** — a prototype table beat us; that is a finding about the task.
- ⛔ **Store load-bearing is BY CONSTRUCTION** ⇒ method section, never a headline.
- ⛔ **Capability 1 (composing unseen-valid) is never novel on its own** — LACE owned it in 2021.
- **Refusal (when restored):** *"one dynamics-derived criterion covers BOTH kinds with zero invalid supervision,
  where each rival covers one"*; `out_of_range` vs kNN is a **TIE**.
- ⛔ **Navigation is NOT RUN**, not "wrong idea" — the read could not reproduce its endpoints.
- ⭐ **The scaling law is a headline mechanism result:** effective dimension, the level's parameter-free
  confirmation, and **the whitening prediction that held**.
- ⛔ **λ is `≈ 0.054` as certified** — never `1.0` (that was dreaming's value, inherited by accident).

## 5. ⛔ TRAPS THAT HAVE EACH COST THIS PROGRAMME TIME
1. **`dec_recipe` defaults to `"baseline"`** — the decoder that FAILED the gate. Winner `v1_capacity_schedule`.
   ⭐ Every spoke selects it explicitly and asserts the hash. A baseline run **looks normal in the logs and is void.**
2. ⭐ **pytest from a worktree silently SKIPS every real-data test** — `.claude/` is only in the primary checkout.
   **Always `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`, and report passed/skipped/failed, never
   passed alone.**
3. ⛔ **The full suite takes the worktree EXCLUSIVELY** — no merge, no checkout, until it reports. **The Hub voided a
   94-minute run by merging mid-run.** Record HEAD before AND after. ⚠ Pipe the FULL output to a file; a `tail` throws
   away the failure detail. `.pytest_cache/v/cache/lastfailed` recovers test names without a re-run.
4. ⚠ **Full-suite state: 2,517 passed / 2 failed @ `81ba81b`.** Both failures are **§7.23's x64 ordering hazard**, not
   a regression — six modules enable `jax_enable_x64` at import and never restore it, and `test_friction_field` sorts
   first. **The frozen path is intact** (`feat_shape` matched; only `enc_hash` moved). ⛔ **The Hub's local fix made it
   WORSE (4 failures) and was reverted** — `triple` is module-scoped and the function-scoped remedy is insufficient.
   **Do not patch it locally again; it belongs in the infrastructure batch.**
5. ⛔ **`chlu/config.py` same-anchor merges:** ⛔ never hunk-by-hunk, ⛔ never a blind "keep both" (git has split a
   conflict **inside a docstring**). ⭐ **Take the whole file, re-apply the peer's block by its `BEGIN`/`END` markers,
   assert 0 deletions, then verify BEHAVIOUR** — instantiate every block and round-trip save/load.
   ⚠ Some blocks are `CHLUConfig` **fields**, others are reached by **accessor functions** — a presence check that
   assumes fields reports a false loss. It did, twice.
6. ⚠ **JAX `split(key, n)` is PREFIX-STABLE here** — `split(k,2)[1]` **IS** `split(k,3)[1]`. ⛔ Derive arm keys with
   `fold_in`, or two arms are bit-identical rather than decorrelated.
7. ⚠ **A per-factor statistic is meaningless without `n_values_realised`** — Montero's corner pins `shape` to ONE
   value and yields a confident-looking zero. This produced a wrong headline once.
8. ⛔ **A spoke may not be retired on a number that has no laundering control.** The Hub retired `gen-encoder-pretrain`
   on an absolute probe accuracy; it ran anyway and produced the control that invalidated the retirement.
9. ⛔ **`fork_push_runbook.sh` is NOT run as written:** it pushes the frozen `clu` (⛔ skip that line — **`clu-dev`
   only**), its on-main guard is unusable (main is checked out in `CHLU-merge`; use
   `git merge-base --is-ancestor HEAD main`), and its `"${BIG[@]}"` dies on **bash 3.2** with an empty array.
   ⛔ **Never `git checkout internal-state` in a working repo — it once wiped the brain.**
10. **CSF3 needs 2FA — no agent reaches it, ever.** Spokes emit submit lines and report cluster items **NOT RUN**.
    ⭐ `GEN_GIT_COMMIT` must be exported **on the login node**, not the laptop. ⛔ Never `-p serial` (the script
    hardcodes `-G 1 -c 8`). ⛔ `git pull` on the cluster serves a stale fork — **`push_repo.sh` is the only path.**

## 6. Standing debts — none blocking
- ⛔⛔ **`stratified_fit_val` SILENTLY CLAMPS `n_val` to `pool.size − n_fit` (= 512 at the shipped `n_fit`).**
  ⛔ A caller asking for more **gets 512 rows and an entirely normal log.** ⭐ Closed inside `exp_gen_atom_mean`
  (`gen-atom-mean-resweep`); ⛔ **`exp_gen_baselines` and `exp_gen_cell` call the same function and are NOT fixed.**
  **Owner: Hub — a scoped repair, ⛔ never a drive-by: it changes what every tuning caller realises.**
- ⭐⭐ **IDENTIFICATION, MEASURED AND BISECTED (`gen-atom-mean-resweep`, 18/18 runs):** `n_val = 4,096` reaches the
  pre-registered **1 of 6** within-draw (from 5 of 6 at 512), `n_fit` 28,672 (88.9 %), and ⭐ **`knn_proto` did not
  degrade** (0.877–0.883 vs a 0.8817 reference). ⛔⛔ **BUT the within-draw gate is NOT SUFFICIENT: `n_val` 4,096 and
  8,192 — both passing it — pick a different winner in 3 of 6 cells.** ⇒ ⭐ **PREREG §5b.0 now has three clauses**
  (within-draw · **across-draw** · per-rung classification). ⭐ **`scope`/`normaliser`-like levers are unanimous
  36/36 even at `n_val`=512 and are quotable today; `k`/`metric`-like levers are NOT identified at ANY affordable
  `n_val` and more rows will not rescue them.**
- ⭐ **Last stale number cleared:** deployable (request-addressed, k=1) exponent **−0.162 ± 0.005 (broken split) →
  −0.1578 ± 0.0062 (repaired)**, Δ +0.0039, inside one seed-sd. ⛔ Exponent work remains SUSPENDED to rebuttal.
- ⛔⛔ **`train_store` RECOMPILES EVERY STEP — a 100× cost — and the obvious fix is NOT numerically neutral.**
  `tau` is a Python float into an `eqx.filter_jit`-ed `step` (`gen_store.py` ~L788), so `filter_jit` treats it as
  **static** and each annealed value retraces. ⭐ **Measured: 1200 steps = 316.89 s → 3.12 s (≈100×).**
  ⛔⛔ **BUT THE NUMBERS MOVE AND THE FIX WAS REVERTED 2026-09-15.** Same seed, same data, 1200 steps:
  | variant | wall | lossN | param sha256 |
  |---|---|---|---|
  | ⭐ **shipped (static `tau`)** | 316.89 s | **14.6261959076** | `21d284a71ca25cca2513` |
  | `jnp.asarray(tau)` | 3.12 s | 14.5891962051 | `8559463ed70f06d93029` |
  | `jnp.asarray(2.0*tau**2)` precomputed in float64 | 3.38 s | 14.5987091064 | `64f7061b126f47b9daba` |
  ⭐ `loss0` is **identical** in all three ⇒ accumulated **precision/fusion drift**, ⛔ not a logic error. Pinning the
  float64 denominator got closer but ⛔ **did not restore bit-identity**: with `tau` static XLA constant-folds and
  fuses differently, and a traced operand cannot reproduce that.
  ⇒ ⛔ **This is NOT a free speedup.** Taking it silently re-bases every store anyone retrains, and shipped store
  hashes are **asserted** (`store_s0.pkl` `3e3424a1ea40017e…`). ⭐ **If wanted, it is a DECLARED re-baseline —
  regenerate the stores, update every asserted hash, re-score what depends on them — a scoped spoke, ⛔ never a
  drive-by cleanup.** ⚠ Route B fixed the identical bug **in its own module**, where nothing was frozen, and there
  it was a pure win (seconds/step → ~0.05 s/step).
- ⛔⛔ **STANDING (Head, 2026-09-15): EVERY CLUSTER RUN CHECKPOINTS AND RESUMES.** An OOM / node failure /
  wallclock kill at hour 70 of 96 must resume, ⛔ never restart. ⚠ **Measured: `job_gpu_single.sh`,
  `job_gpu_array_seeds.sh`, `job_gpu_eval.sh` have ZERO resume logic**; `exp_cluformer_pilot.py` is the only
  experiment that resumes. ⭐ **Reuse `pilot-checkpoint-resume`'s pattern** (bitwise-verified, both paths);
  ⛔ acceptance is a **`kill -9` mid-run + bit-identity check**, never a resume path that was merely written.
  ⭐ Checkpoint to `~/scratch` — ⛔ never `$TMPDIR` (wiped at job end), ⛔ never `$HOME` (quota).
- ⚠ **No path stages gen artefacts to CSF3.** `push_repo.sh:42` excludes `.claude/`; `sync_project.sh` only moves
  `projects/<name>/`. Owner: `gen-cluster-stage-artefacts`. ⛔ The green cluster smoke ran `SMOKE=1` — no science.
- **Infrastructure batch, 5 items, unscoped, one spoke:** §7.23 `conftest.py` x64 remedy · the launcher's wrong
  submit block · three `fork_push_runbook.sh` bugs · `push_repo.sh` shipping 180 MB of regenerable caches ·
  22 task files hardcoding `/Users/user/Desktop/CHLU` (should be `$CHLU_ROOT`).
- **`clu-dev` backup** ~20 commits stale. **Cluster unfreeze run** needs bridge/store artefacts staged onto scratch
  (`push_repo.sh` excludes `.claude/`). **`tune_n`/`n_val`** — the real fix for identification — unowned.
- **Claims matrix** owes: the scaling law as a headline result, and D1's replacement of the "unaddressed wells"
  sentence with the scaling-law justification.
- ⏸ **Multi-device working is PARKED post-ICLR** — the analysis is in the ledger; Zed SSH remoting is the Head's route.

## 7. How this Hub works
Task file → hand the Head **one copy-able spawn block, one line per spoke** → the Head launches → review against the
task → merge `--no-ff` with a `[hub]` prefix → targeted tests on the **merged tip** → record. ⛔ **The Hub never
launches.** **Cap 3 code worktrees.**
⭐ Every task file carries: dial declaration · the one job · files owned · a **mechanical** precondition (a file that
must exist, content-checked) · acceptance · and *"what would prove this spoke wrong"*.
⭐⭐ **The spokes that found the most were the ones told to build the control that could kill their own result.**
**Three of the last five spokes invalidated something the Hub had written. That is the system working — scope for it.**
