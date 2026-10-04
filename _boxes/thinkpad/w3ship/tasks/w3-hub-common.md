# WAVE 3 — COMMON RULES FOR ALL FOUR HUBS (read after the charter, before your own target file)

**Binding on `hub/factor-wells`, `hub/unseen-valid`, `hub/refuse-delete`, `hub/store-content`.**
Written by the ICLR-long Hub, 2026-09-18, from the Head's rulings of the same day.

## 0. Boot, in this order and nothing else
1. `.claude/autonomous-hub-charter.md` — binding, unchanged.
2. **this file.**
3. your own target file, `.claude/tasks/hub-<target>.md`.
⛔ Do not read the campaign ledgers, the wave-2 reports, or `handover_context.md`. Everything from wave 2
you need is in §4 below. ⛔ Do not re-derive it.

## 1. ⭐⭐ THE NEW LIKE-FOR-LIKE — this is the change from wave 2, and it binds every row
**The control is the FITTED BASIS + LINEAR HEAD** — the wave-2 object: a factored dictionary over frozen
encoder features plus a request-conditioned linear regression head, at **matched bytes**.
`knn_proto` stays as the **second** row.

⛔⛔ **A CLU mechanism that ties basis+head is decoration and is reported as such, in those words.**

**Why this is the bar now:** wave 2 established that on that object **the head does the work and the
store is a compression basis** — medoid atoms tie fitted atoms exactly, and the pair-interaction block,
not the store, is what beats `knn_proto`. ⇒ Beating the prototype table no longer demonstrates anything
about the landscape. Beating basis+head does.

## 2. The scoreboard — fill every row, you are judged on yours
Seven rows (the Head's taxonomy), **every arm and both controls scored on every row**:
**1** seen · **2** unseen-valid · **3** compositional 2-way · **4** compositional n-way ·
**5** compositional refusal · **6** invalid-request refusal · **7** deletion.
Your target file names **your** rows. ⭐ You still fill all seven — that is what makes the four hubs
comparable, and it is cheap once the harness is wired. Rows outside your target are reported, never
claimed.
Refusal rows report the **pair** `(reject-invalid, accept-unseen-valid)`; ⛔ a refuser that refuses
everything scores **0**. The external classifier is the judge throughout.

## 3. ⛔ The estimand — single cells no longer decide anything
Wave 2 measured that **the corner explains 90.9 % of the paired variance** and the store seed 9.1 %.
Both of wave 2's scored results were single 72-item cells, and neither could distinguish "won this cell"
from "ties in expectation."
⇒ **Your bar is the MULTI-CORNER estimand: ≥ 49 corners, paired, with the margin declared in advance.**
⛔ A single held-out cell may not decide a target. Report the cell beside the estimand if you want; the
estimand is the verdict.
⭐ **Pin the store fit completely** in your PREREG — partition, RNG stream, Lloyd budget, ridge, every
one. Wave 2's composition hub pinned ridge and left the rest free; its headline level then failed to
replicate (42 vs 25) while its verdict held. ⛔ Do not repeat that.

## 4. ⭐ KNOWN FACTS — ⛔ do not rediscover any of these
1. **Learned well placement returned a null.** Inherited correlation grouping already beats the random
   floor 50× over the learned arm. ⛔ Do not start from a learned rotation.
2. **The codebook loses 90.5 % of `scale` and 100 % of `orientation`.**
3. **Density-over-atoms descent regresses to seen combinations** (launch 0.685 → settled 0.211).
4. ⭐ **The read verb is the largest measured effect in the programme.** Changing only the read from
   argmax (one vertex per well) to a real-valued **linear** combination over modes — the signed span —
   moves the shipped store from `0.5903 / 9 tuples` to `0.7917 / 27` at a **byte-identical** ledger,
   confirmed at 43/43 corners. On the same measurement: nearest mode 0.4884 · best **convex**
   combination 0.5509 · best **linear** combination **0.8264**. **The modes already span the answer.**
5. ⭐ **The store is used as a SIGNED BASIS, not a mixture** — head coefficients are ~47 % negative,
   L1 ≈ 8.5 per well, and 0.00 % of wells form a probability vector. ⛔ Do not impose a simplex.
6. ⭐ **Above ~2× the byte budget the store is not the binding constraint — the addressing head is.**
   Doubling the store from 2× to 4× buys +0.08 tuples. ⛔ More bytes is not the lever.
7. ⛔ **Dreaming is dead, with a mechanism:** every energy-driven state lies **inside** the mode hull and
   the object lives **outside** it. Not fixed by temperature, τ over three decades, more steps or a
   softer read — all measured. ⛔ Do not revisit.
8. ⛔ **A length scale derived from the spacing of the stored items cannot separate new points from
   them.** The admission gate died on exactly this: `σ` was the median nearest-neighbour distance
   between atoms, the atoms **are** train rows, so they tile their own manifold at spacing ≈ `σ` and any
   real point sits ≈ 1 `σ` from one — whatever the store holds. The gate fired at 2.0–2.4 `σ`; all real
   data sat at 0.79. 8× the store closed 0.09 of the gap. **Binding design constraint: any threshold you
   introduce must not derive its length scale from the stored items' spacing.**
9. ⛔ **`store_energy` is reproduced to 1.2e-07 by the per-well reduction** and the identity
   `max_i E_k(a_ki) ≤ 0` holds with 30 units to spare. The energy machinery is correct. ⛔ Do not re-audit it.
10. ⛔ **Atoms per well beyond the well's channel dimension buy no new directions** — three of five wells
    at the large rung are at the maximum rank their channel count allows. "8× atoms" is not 8× capacity.

## 5. Scored runs — the ledger is written BEFORE the run
⛔ **Three scored runs per TARGET**, across every machine and every attempt, counted from 2026-09-16.
**Current count: `generation-table` 1/3 · `composition` 1/3 · `deletion` 0/3 · `admission` lapsed.**
Your target file names which target's budget you draw on and how many remain.
⭐ **Append your row to `SCORED-LEDGER.md` in the shared hub root BEFORE the run starts.** A hub that
cannot write the ledger cannot spend. Every quoted number carries its count ("scored 2/3").
⛔ All scored runs are reported, including the ones that lost. The best is never reported alone.
⛔ Seed 0 only this wave. Seeds 1–2 are not produced now (Head). **No number reaches the paper below 3
seeds** — so a scored run this wave buys a verdict, not a paper number. Spend accordingly.

## 6. Process — the wave-2 faults, made mechanical
1. ⛔⛔ **One launcher per hub. Refuse to start if a live `START_UTC.txt` exists in your output
   directory.** Two agents executed one hub concurrently in wave 2, sharing a worktree and interleaving
   one log. Write `START_UTC.txt` as your first action; a log two writers append to is not a record.
2. **`LOG.md` in dated bullets, throughout** — what you tried, what you measured with its control, what
   it means, what you do next. ⛔ No retrospective rewriting; corrections are new bullets. Your final
   report is the last section of that same file.
3. ⛔ **A hub must not spawn sub-agents for long-running work.** Replication spokes (§A.5) are the
   exception and are short. Confirm none of your sub-agents is still running before you finish.
4. ⚠ **Cap memory and per-corner wall clock on every launch.** A wave-2 sweep grew to 50.3 GB resident
   and drove the shared box to 0.4 GB free. **Make every run resumable** — the kill then costs one
   corner, not the run.
5. ⛔ **A return needs BOTH a report AND a `git bundle`.** `.claude/**` is gitignored, so a bundle
   carries no outputs and `--return-pack` carries no code. ⚠ `git status` does **not** show gitignored
   work — a worktree holding an entire unsaved outputs tree reports clean. **Use a directory listing.**
6. ⚠ `rsync` to the hub box **requires** `--rsync-path='C:\msys64\usr\bin\rsync.exe'`. Without it rsync
   transfers nothing and exits clean.

## 7. Floors — charter §C, unchanged and still binding
Every arm beats the matched-bytes lookup beyond seed spread, or the store is decorative and is reported
as such · every arm carries a landscape-removed control that must collapse · any arm whose settled state
sits on its nearest atom is a lookup table and is reported as one.
⛔⛔ **§C.3's "a tie is the win narrative" clause is DEAD IN ALL THREE LEGS — amended 2026-09-19.**
Dreaming died in wave 2 with a mechanism (fact 7). **Refusal and deletion-as-refusal died on
measurement:** an **8-byte** value-presence bitmap plus a free bracket check accepts rows 1 and 2 and
refuses rows 5 and 6, all at 1.0000, against a 524,288-byte budget — and it is structural, because the
request space *is* the 57-value factor grid and the full joint table over all 480,000 combinations is
only 59 KB. Row 7 goes the same way: the bitmap updates on delete.
⇒ ⭐⭐ **A TIE ON GENERATION CAN NO LONGER BECOME A HEADLINE. Wave 3 wins rows 3 and 4 outright, or the
paper has no primary claim.** That is the whole wave's situation and every hub should know it.
⭐ What survives is **erasure, not refusal** — that deleted content is *gone* rather than *gated* — and
it is `hub/erasure`'s target, at zero scored runs. ⛔ No hub quotes a row-5 or row-6 number as evidence
for the store.

## 8. ⭐ COMPUTE PLACEMENT — GPU is available, and it is conservative by ruling (Head, 2026-09-18)
**Local CPU on the hub box is the default and is free by definition — it is a dedicated 64 GB machine.**
**GPU is GCloud, it is available, and it is for HEADLINE results only.**

⛔ **Not for:** smoke tests · dry runs · selection sweeps · corner screens · controls · ablations ·
anything already measured to run on CPU. ⭐ **If it works on CPU, it runs on CPU.**

**Before requesting a GPU, write these three lines into `LOG.md`** — if you cannot, you do not have a
GPU case:
1. which scoreboard row the run decides, and that it is a pre-registered scored run or its direct precursor;
2. the **measured** CPU wall clock that makes it infeasible — ⛔ an estimate is not a reason;
3. the cost line.
⭐ **In doubt, CPU.** A GPU leg that turns out to have been avoidable is reported as an error against
yourself, the way wave 2's hubs reported their own budget breaches.

### ⛔⛔ The two hazards that make a careless GPU leg worse than no leg
1. ⛔⛔ **Never mix CPU-produced and GPU-produced quantities inside ONE paired comparison.** CPU and GPU
   features agree to ~53 ULP on ~1 % of elements — **not bit-exactly**. Every comparison must be
   internally same-device. ⚠ A split arm-vs-control is void and the void is invisible in the logs.
2. ⛔⛔ **A run that EXTENDS the feature cache on a GPU is enforced at full strength by the re-encode
   gate.** Check cache coverage and record `shortfall` **first**; if it is non-zero, you are extending.
   ⭐ The gate binds on the hazard, not on every run — so a GPU run that only *reads* the cache is fine.
