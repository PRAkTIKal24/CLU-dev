# GEN PHASE 1 — HUB HANDOVER. Written 2026-09-13. Read this, then stop and wait for the Head.

**Boot:** `.claude/iclr-long-hub.md` (your role) → **this file** → `.claude/tasks/gen-clu-phase1-rulings.md`
(**every binding ruling, R1→R29; the only ledger you need**). ⛔ Read no campaign charter or handover — they are a
different programme. The one exception, for spokes and for you: **`.claude/handover_context.md` §7 Known Issues only.**

---

## 1. Where things stand
- **`main @ 5fee3f4`** (worktree `/Users/user/Desktop/CHLU-merge` is on `main`; the primary checkout `~/Desktop/CHLU`
  sits on a stale merged branch — ⚠ never run `push_repo.sh` from it).
- **Merged and done:** data+splits · CSF3 launcher · h5py extra · C-set sizing · feature bridge · decoder lift ·
  wide split · store grouping · R0 wide re-fit · dream sampler · certificate · baselines · A7 · read-compose ·
  metric invariance · launcher fixes.
- **Nothing in flight.** `gen-arms-a7-wiring` is **scoped but NOT launched**.
- **Backups:** `clu-dev` at `3faef27` (code) + `internal-state @ 97f259e`. ⛔ `clu` frozen at `b0257bc` — never push it,
  never push `origin`. Runbook pushes both, so ⛔ do not run it as written; push `clu-dev` only.

## 2. ⭐ THE PIVOT — read R29 in full before scoping anything
Phase 1 is a **mechanism study**. **Composition is RELEASED as a claim** — kill rule (ii) fired as pre-registered.
New order: **(1) the commitment collapse [PRIMARY] · (2) the generation harness end to end [headline table] ·
(3) encoder unfreeze, cluster · (4) navigation · (5) ladder = appendix, background, must not consume the wave.**
⛔ **Process: the Head gets a worklist (spoke · cost · falsifier · artifact path) BEFORE anything launches.**

## 3. The results that survive
| finding | number |
|---|---|
| ⭐ **A bypass is INEXPRESSIBLE** — the read takes no features; only channel is 30 bits | delete-store Δ **+0.4752** vs a 0.01 null |
| ⭐ **Refusal barrier holds where naive geometry collapses** (A7-controlled) | **0.9146** vs control **0.5630**, 9.5× spread |
| Attribution names the culprit factor | **0.797** dup / **1.000** oor, chance 0.167 |
| Zero-invalid-supervision audit | **0 C-ids in 15 fits / 78,848 ids** |
| Composition IS the store's | `read − store_removed` **+0.1759**, 16.4× spread |
| ⛔ …and still loses to a table of means | `CLU-wide − knn_proto` **−0.1400**, 0/3 |
| ⭐⭐ **COMMITMENT COLLAPSE** — 72 requests → **9.0 ± 0.8** distinct codes (seen 495/512) | the primary target now |
| DCI-C qualified as ladder axis; negative control failed first | 6/6 signs, ratio 1.34 |
| Decoder discriminates held-out shapes | **0.9894 ± 0.0021** (11 cells) |

**Reference rows for every table:** `knn_proto` **0.8884** · store ceiling **0.6866** · decoder ceiling **0.9414**.

## 4. ⛔ Live traps — each has already cost time
1. **`dec_recipe` defaults to `"baseline"`** — the decoder that FAILED the gate. Winner is `v1_capacity_schedule`.
   ⭐ Every spoke must select it explicitly and **assert the decoder parameter hash**. A baseline run looks normal in the logs and is void.
2. **`include_coupling_bytes` defaults to `True`** — the retired 688,128 B budget. Correct is **524,288 B**.
3. **`soft_read(store, X, …)` takes FEATURES** — using it voids the no-bypass guarantee. Mixture weights must come
   from the **request**.
4. **CSF3 needs 2FA. No agent can reach it, ever.** Spokes emit submit lines and report cluster items **NOT RUN**.
   ⛔ **The plumbing smoke has NEVER run** and still gates the pre-registered cell.
5. **`config.py` same-anchor merges:** take the peer's whole file and re-apply yours; assert 0 deletions.
   ⛔ Hunk-by-hunk "keep both" has broken syntax twice.
6. **JAX cold start ~20 min**; a plain `import chlu` will time out a 2-minute tool call.
7. **16 GB machine.** One feature cache is 1.97 GB float16 / 3.93 GB float32. ⛔ Never run a full suite beside a spoke.
8. **Detached spokes keep running but fail their LAST write** — require per-unit incremental writes and skip-if-exists.

## 5. Verification policy (changed 2026-09-13, keep it)
Spokes run the full suite on their own branch. The Hub runs **targeted tests per merge** (~40 s) and **ONE full suite
per wave on the merged tip, when the machine is quiet**. Last full green: **2166 passed** @ `a0034d4`.
⚠ `5fee3f4` has had targeted tests only — **a full suite is owed at wave close.**

## 6. Open, needing the Head
- The worklist for pivot items 1–3 (below in the Hub's outgoing message; re-derive if lost).
- ⛔ **`gen-arms-a7-wiring` scoped, not launched** — the `duplicate_factor` half of refusal, LACE's whitening fallback
  on 2 non-exact seeds, and the `include_coupling_bytes` default. **Nothing in refusal is quotable until it lands.**
- The cluster smoke (2FA, Head-only).
- Disk-only rider from R29 Q2: **do the 9 codes coincide with SEEN configurations?** One line either way.

## 7. How this Hub works
Write a task file → hand the Head **one copy-able spawn block, one line per spoke** → the Head launches → review
against the task → merge `--no-ff` with a `[hub]` prefix → record in the rulings ledger. ⛔ **The Hub never launches.**
⭐ Every task file carries: dial declaration · the one job · files owned · a **mechanical** precondition (a file that
must exist) · acceptance · and *"what would prove this spoke wrong"*. **Cap 4 concurrent spokes.**
⭐ The spokes that found the most were the ones told to build the control that could kill their own result.
