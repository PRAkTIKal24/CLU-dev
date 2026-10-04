# AUTONOMOUS HUB CHARTER — CONFIRMED BY THE HEAD 2026-09-16 (binding)

One hub = one broad question, one branch/worktree, ≤10 h wall clock, may spawn sub-agents, CPU by default.
The hub MUST go through the CLU and MAY modify it, subject to §B. The Head launches every hub.

## A. Pre-registration under autonomy
1. Iterate freely on MECHANISM. Every SCORED run (any number on the held-out sets A/B/C) is pre-registered first in
   `PREREG-<hub>-<n>.md` with its bar, its like-for-like rows and its expected direction, BEFORE it runs.
2. Selection on `val` only. ⛔ At most **3 scored runs per hub** against the held-out sets. All 3 are reported; the
   best is never reported alone. The scored-run count travels with every quoted number ("scored 2/3").
3. A hub may not change a bar it has already seen a result against. Amendments are appended, dated, never edited.
4. **Frozen assets, read-only for hubs:** the scoring harness (external classifier, repaired splits, rival rows,
   byte-ledger and bypass assertions) is pinned at one commit on `main`. A hub that edits them voids its numbers.

   ### ⭐⭐ THE PIN — `4125fce` (`main`, 2026-09-16, tree clean)
   ⚠ **RE-PINNED 2026-09-16 from `ae907466`. ⛔ NO HUB HAD STARTED**, so no number's comparability is affected —
   the pin is prospective, and the wave had not begun. ⛔ **It does NOT move again once the first hub launches.**
   **Why it moved:** the old pin carried a top-level `import resource` in `chlu/training/train_cluformer.py:42`
   (Unix-only) which **took the whole `chlu` CLI down on Windows**, including `chlu data shapes3d --download` —
   the command the second machine needs because the raw dataset is deliberately never bundled. ⭐ Re-pinning was
   cheaper and cleaner than teaching the bundler to overlay `chlu/**` files past the pin.
   ⚠ **Harness delta vs `ae907466`:** `exp_gen_cell.py` **+734 lines** — `gen-addressed-read` adding the `x0` arm.
   ⭐ **Additive: `gen-addressed-read`'s SAME-CELL PROOF reproduced all five rival rows BIT-IDENTICALLY.** ⛔ No
   rival's behaviour changed. Every other harness file is byte-identical to the old pin.
   ### (superseded) `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`
   ⛔ **Every hub worktree branches from THIS commit.** ⛔ Not from `agent/experiment-engineer/gen-dynamical-read`,
   ⛔ not from a peer hub's branch, ⛔ not from a moving `main`.
   **Verified present at the pin:**
   | component | site |
   |---|---|
   | external classifier (gated, hash-asserted) | `exp_gen_cell.load_external_classifier` |
   | repaired stratified split | `chlu/data/gen_rowsplit.stratified_fit_val` |
   | rival rows | `exp_gen_cell.ARMS` = `clu, store_removed, knn_proto, lace, ebm_compose, dynread` |
   | byte-ledger assertion | `chlu/core/gen_landscape.landscape_byte_ledger` |
   | bypass assertion | `chlu/core/gen_store.assert_no_bypass` |
   | re-encode gate, scoped to the hazard | `exp_gen_store._assert_cache_reencodes(will_extend=…)` |
   | encoder precision forced to true float32 | `chlu/core/gen_bridge.ENCODE_PRECISION` |

   ### ⚠ DECLARED PROPERTIES OF THE PIN — known, deliberate, ⛔ NOT silently repaired inside it
   1. ⛔ **`GenCellConfig.arms` defaults to FIVE arms; the module constant `ARMS` has SIX.** `dynread` must be
      selected explicitly (`--arm dynread`). ⭐ Left as-is on purpose: adding an expensive, *losing* arm to every
      default cell run imposes compute nobody asked for.
   2. ⛔ **`GenCellConfig.lace_selection_source` still points at `gen-arms-a7-wiring/runs/historical_688128`** —
      the **broken-split** LACE selection. ⭐ Re-pointing it would move a rival's numbers and needs its own scoped
      re-score. ⛔ A hub does not touch it.
   3. ⚠ **CPU/GPU feature agreement is ~53 ULP on ~1 % of elements**, not bit-exact, after the precision fix
      (20.85 % → 0.96 %). ⭐ The re-encode gate binds only when a run will EXTEND the cache. ⛔ A hub that extends
      the cache on a GPU is enforced at full strength.
   4. ⭐ **The `lace` composition-B row on the generation table is `0.3117 ± 0.0153`** (CELL harness,
      `/arms/lace/external/composition_B/aggregate`). ⛔ **`0.3241 ± 0.0165` is the BASELINES instrument at the
      688,128 B leg and is NOT comparable on this table.** ⭐ Both are real; they are different instruments.
5. **Independent replication before any number is spoken:** a separate spoke re-runs the hub's winning configuration
   from its PREREG file on a clean checkout of the hub's branch. Unreplicated = not a result.
6. Every hub ends with: PREREG files · a results ledger (every scored run) · a one-page report · a cost line.

## B. The core philosophy — what a hub may NOT change
1. The store is a LANDSCAPE: content is written as atoms that shape an energy over the latent space; wells are its
   basins. Write, read, decay, delete are operations on that landscape.
2. Addressed, not indexed: a read is reached by address (request/energy), never by row id.
3. Request-only reads: no feature tensor enters launch, settle or read (the bypass assertion must pass).
4. Deletion and lifetimes stay exact and are operations on atoms.
5. Bytes are matched and declared; the ledger assertion must pass.
6. Label-free representation: labels ground the words of a request at the interface only (the address book);
   never in the landscape, the read or the write.
Everything else may change: kernel, coupling, partition, codebook, dynamics, temperature, decoder interface, encoder.

## C. The floor — no manufactured positives
1. Every arm must beat the MATCHED-BYTES LOOKUP (atom-mean / kNN on the same atoms) beyond seed spread on the
   headline metric. Otherwise the store is decorative and the arm is reported as such.
2. Every arm carries a LANDSCAPE-REMOVED control (atoms zeroed or shuffled) that must collapse.
3. A TIE with the lookup counts only if the same object also delivers, on the same artefacts, at least one of:
   refusal with accept-unseen-valid · deletion that changes the output exactly · dreaming that yields valid NOVEL
   tuples. That is the "tie is the win narrative" clause, made mechanical.
4. Any arm whose settled state sits on its nearest atom (distance → 0) is a lookup table and is reported as one.

## D. GPU policy
CPU by default. GPU only for a run whose best case is a HEADLINE number. No GPU for supporting claims.

### ⭐ AMENDMENT 1 (Head, 2026-09-16) — compute placement is PER MACHINE
⛔ **§D's "GCloud CPU is your default" was written for the 16 GB shared laptop**, where a hub's CPU work
competed with Hub/Advisor/Head work. ⭐ **On a DEDICATED hub machine, local CPU is free by definition.**
**Binding, by machine:**
| machine | CPU work | GPU work |
|---|---|---|
| ⭐ **Windows hub box (64 GB, dedicated)** | ⭐ **LOCAL — run it there** | ⭐ **GCloud** |
| the 16 GB shared laptop | ⭐ **GCloud** (⛔ not local — it blocks the Head) | ⭐ GCloud |
⭐ **The sub-agent concurrency cap is a property of the MACHINE, not of the charter.** The profile's "cap 3"
is the 16 GB laptop's limit. ⭐ **On the 64 GB box it may be raised** — ⛔ but a hub still confirms none of its
sub-agents is still running before it finishes.
⛔ **UNCHANGED: GPUs sparingly, and ⛔ never for a run whose best case is a supporting claim.**
⛔ **UNCHANGED: CSF3 is unreachable by any hub (2FA) — every cluster leg is `NOT RUN`, emitted as submit lines.**

## E. Questions and logging (Head, 2026-09-16)
1. A hub is given a TARGET, not a sub-question: "win or tie the generation table", "a compositional-generation win",
   "a refusal win", "a deletion win". How to get there is the hub's to find, inside §A–§D.
2. **Running log, mandatory:** every hub keeps `.claude/outputs/hub-<target>/LOG.md` in its worktree and appends
   simple dated bullets THROUGHOUT — what it tried, what it measured (with the number and its control), what it
   means, what it does next. A reader must be able to open that one file and know exactly what each result was and
   what it means. No prose, no retrospective rewriting; corrections are new bullets.
3. Final report = the last section of the same LOG.md: the pre-registered runs (all of them), the best replicable
   configuration, the cost line, and one line per §C floor check.
4. Every hub is told up front: learned well placement returned a null (inherited grouping already beats the random
   floor 50× over the learned arm); the codebook loses 90.5 % of scale and 100 % of orientation; the density-over-atoms
   descent regresses to seen combinations (launch point 0.685 → settled 0.211). Do not rediscover these.
