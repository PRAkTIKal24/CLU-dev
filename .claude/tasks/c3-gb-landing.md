# c3-gb-landing — land the ratified G-B geometry, unblock run 3's launch, and pin G-B in the prereg

**Campaign 3, wave 1. Agent:** experiment-engineer. **ONE worktree (wt1).** ⛔ **Small and on the critical path** — it blocks **run 3's submission** and **every ladder job**. ~1 day.
Branch **`agent/experiment-engineer/c3-gb-landing`** off **`agent/experiment-engineer/c3-run3-budget-exemption`** (accepted, unmerged — you extend it so the Hub merges **once**, with your §2 fix inside).
⚠ Shared checkout sits on the pilot branch; **take a worktree**. Reuse the main venv (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`).
Writes `.claude/outputs/c3-gb-landing.md`.

**Binding documents:** `.claude/outputs/c3-rival-ladder-prereg.md` **§F1–F3 IN FULL** (the geometry finding and the three options) · `.claude/outputs/c3-rival-ladder-prereg/PREREG-C3-LADDER.md` (**you amend it**) · `.claude/outputs/c3-run3-budget-exemption.md` **§6** · `.claude/outputs/c3-run3-launch/RUN3-LAUNCH.md` **§B2** (the blocker you clear) · `.claude/outputs/c2w11/PILOT-TTT-RULINGS.md` ruling 1.

---

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** **none — instrument/plumbing + prereg amendment.** ⛔ No claim, no ladder arm trained, no bpc.
- **Falsifies:** a `store_layers` selection that changes the byte arithmetic away from the ratified **1,380,864 B**; run 3 still refused after §2; the prereg amendments not surviving a re-read by someone who was not here.
- **Does NOT falsify:** G-B being architecturally unusual. **It is ratified** (Head + Advisor 2026-08-13); you implement it, you do not re-litigate it.

## 1. ⭐ `store_layers` on `StreamModel` — the ratified G-B geometry

**G-B (RATIFIED): full-size 8192-atom stores in 3 of 12 layers** ⇒ **1,380,864 B = 0.658×** of the 2 MiB ceiling. Add a `store_layers` selection to `chlu/core/blocks.py` so which layers carry a CLU cell is a **config value**.

- **Pin the byte arithmetic in a test**: 3 store-bearing layers × 460,288 B = **1,380,864 B**, occupancy **0.658×**, and the CLU/TTT-matched match ratio preserved.
- **Which 3 of 12** is a **design decision, not a default** (§3) — the config must make the choice explicit and the ledger must record it. ⛔ No silent `[0,1,2]`.
- ⚠ **Do not disturb the pilot path.** Run 3 is a continuation of run 2 at the pilot geometry; if `store_layers` defaults in a way that changes `as_flag_table()` for the pilot config, **run 3's exemption breaks** (a second differing key). **Assert this with a test** — the same failure mode the exemption spoke found for its own flag.
- ⚠ G-B keeps the pilot's per-layer cell ⇒ `solve_matched_ttt` returns `(2197, 52)` and **`η·n/d = 3.004 ≥ 2`** ⇒ **`ttt_normalized_write=True` on the TTT arm** is required and is **already ruled** (PILOT-TTT-RULINGS 1). Record the criterion firing in the artifact; ⛔ do not treat it as a new decision.

## 2. ⛔⛔ CLEAR RUN 3's LAUNCH BLOCKER — two lines, and run 3 is submittable

`scripts/csf3/job_gpu_cluformer.sh` (**run 2's script — the one run 3 must use**) has **no `PREREG_CONT` passthrough**; the exemption spoke added it only to `job_gpu_c3_seeds.sh`. ⛔ Run 3 **cannot** be routed through the ladder script instead: it narrows `--arms` per task, `arms` is a `PilotConfig` field, so that is **a second differing key** and the exemption refuses it.

Mirror `job_gpu_c3_seeds.sh:84` and `:169` into `job_gpu_cluformer.sh`:
```bash
PREREG_CONT="${PREREG_CONT:-}"    # run 3's pre-registered continuation
```
```bash
[ -n "$PREREG_CONT" ] && EXTRA="$EXTRA --prereg-continuation $PREREG_CONT"
```
**Verify as the launch package does:** `grep -c 'prereg-continuation' scripts/csf3/job_gpu_cluformer.sh` prints **1**, and add a test asserting the emitted command line carries it (the sibling script already has that test pattern). ⭐ **Then re-read `RUN3-LAUNCH.md` §1–§4 and confirm every line works against your branch** — the operator runs exactly what is written.

## 3. Amend `PREREG-C3-LADDER.md` — three additions, all binding

⛔ Amend in place, dated, marking what changed; do not rewrite the accepted document.

1. ⭐ **The layer placement is a DESIGN DECISION, argued.** *"The CLU as the memory of a streaming block"* becomes *"in 3 of 12 blocks"* — state **which** layers, **why those**, and the alternatives considered. ⛔ **Never a byte-fitting default**: a reader must not be able to discover it as a side-effect of the ceiling. Note the precedent (hybrid/periodic placement) and pre-register what you expect placement to buy or cost.
2. ⭐ **PHASE 1 IS DECLARED, and phase 1 is not the claim.** This ladder = **CLU + TTT swap + dyn-eval arm + slices**. The **six pinned rivals are NOT trained here** (phase 2, funded in parallel). ⛔ **State explicitly that charter §2's tier-iii primary claim WAITS for phase 2 and that no phase-1 result may be quoted as it.** Someone will otherwise quote a phase-1 table as the claim.
3. ⭐⭐ **PRE-REGISTER THE STORE-LIVENESS DIAGNOSTIC on phase 1's FIRST rungs** (Head ruling). At smoke scale the store's read is **inert in the loss** — bpc **bit-identical to 6 d.p.** across a 16× atom range while cell state moved 30,208 → 460,288 B (G4). That is the **C2 flat-curve disjunction**: *carries nothing* **OR** *cannot be addressed* — ⛔ and it must be **separated at real scale, at the first rungs, not deferred to the end.** Design an early leg that distinguishes the two (C2W11's oracle-addressed read is the banked instrument that separated them before — reuse the idea, not the toy code), with **numeric falsifiers and a kill condition**: what result says the store is inert at scale, and what we do then.

## 4. Also fix, while you own the file

⛔ **`byte_ledger.py`'s `StateByteBudgetError` remedy text is WRONG** — it tells the operator to shrink `capacity`/`atoms_per_item`, which move **zero bytes** (F1.1: `n_atoms` is a `max` including the w23 floor `512·√2⁸ = 8192`, which the pilot ties exactly). Use the prereg spoke's replacement wording from its **§Proposed handover updates**, naming the levers that **do** move bytes (`store_layers`, `dim`, `min_atoms_base`). A wrong remedy is worse than none — it sends the operator to a knob that silently does nothing.

## 5. Ownership, stops, acceptance

**Yours:** `chlu/core/blocks.py` (the `store_layers` selection **only**) · `chlu/eval/byte_ledger.py` · `scripts/csf3/job_gpu_cluformer.sh` · `PREREG-C3-LADDER.md` · tests. ⛔ **NOT yours:** the exemption mechanism's verification logic (you inherit it; if it needs changing, **STOP and report**) · run 3's config · anything in `chlu/eval/rivals` (the three concurrent rival spokes) · the corpora/registry surface.
⚠ **Three rival spokes may run concurrently.** Your files and theirs are disjoint by construction; verify by diffing your branch against your base before you finish.

**Stops:** `store_layers` cannot be added without changing the pilot config's flag table → **STOP and report** (it would break run 3) · the ratified 1,380,864 B does not reproduce → **STOP**, the geometry is ratified and a mismatch means the arithmetic moved.

**Acceptance (one line):** `store_layers` lands with the 1,380,864 B / 0.658× arithmetic pinned by test and the pilot flag table provably unchanged; `grep -c 'prereg-continuation' scripts/csf3/job_gpu_cluformer.sh` = 1 with an emitted-command-line test; `RUN3-LAUNCH.md` §1–§4 verified against your branch; `PREREG-C3-LADDER.md` carries all three §3 amendments incl. the pre-registered store-liveness diagnostic with falsifiers and a kill condition; the ledger's remedy text is correct; full suite green with counts against a **named, re-verified HEAD**; branch ref verified from the main repo before the worktree is removed.

**Report:** protocol §5 + flag-provenance + dial declaration + git footprint.
