# bprime-rivals-f3 — the full-F3 rival tuning pass: remove the referee's best attack before the draft freezes

**Campaign 2, wave C2W4 (rider, Head-funded at the C2W4 review, 2026-08-01). Agent:**
experiment-engineer. **Small — hours, existing rig, existing code.**
**Worktree MANDATORY** — you take the slot freed by `bprime-rivals`.
Base local `main` @ **`21a6dc4`**. Branch `agent/experiment-engineer/bprime-rivals-f3`.
Worktree: `git worktree add ../CHLU-f3 -b agent/experiment-engineer/bprime-rivals-f3`.

> ## ⭐ WHY YOU EXIST — the Head's ruling, in its own terms
> The C2W4 audit tuned its rival arms on a **reduced grid** (Adam, 400 steps,
> `lr ∈ {1e-3, 3.16e-3, 1e-2}`; TTT also `b ∈ {1,16}`) — declared honestly by the engineer as *"a
> budget choice, not presented as F3 compliance."* ⚠ **The audit's finding is "rivals lose to their own
> byte-matched tables," and under-tuning a rival produces exactly that finding — the bias runs TOWARD
> our headline.** N78 (rescued baselines) is a standing program value. **Head ruling (2026-08-01,
> verbatim in substance): do the full tuning pass on the arms that showed signs of life — and if proper
> tuning changes any outcome, the paper's claim changes with it. That commitment is the whole point.**
> ⛔ **You are therefore NOT running to defend the C2W4 numbers. You are running to find out whether
> they survive. Either answer is the deliverable.**

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/outputs/bprime-rivals.md` **in full — you are
extending that run, §9's flag-provenance table is your baseline configuration and §4 (the P5/raw-table
finding) is the result your grid must not silently blur**; `.claude/outputs/bprime-rivals/PREREG.md`
(the registered bands you re-test); `.claude/outputs/rival-recon.md` **§F3 and standing rule 5 — the
grid you implement, verbatim**; `.claude/tasks/bprime-rivals.md` §§1–3 (family set, banked numbers,
conventions — all inherited, none re-litigated); the **`2026-07-31 (later still ×3)` `[C2W4]` §10
review entry** (the rescue gate, the Head decision queue item 1 you are discharging).

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none — TIER-i instrument hardening.** No new claim; you are re-scoring an
  existing audit under the program's own standing tuning rule.
- **Laundering control:** unchanged and inherited — the full C2W4 audit column set per cell. ⛔ **You
  change ONLY the tuning grid. Same harness, same seeds, same iso-state budget, same φ, same fit/eval
  split (F2a: the outer parameters never see the eval stream), same rescue gate.** One variable moves.
- **Falsifies:** nothing of yours — **your job is to give the C2W4 numbers the chance to be falsified.**
  Pre-registered consequence (Head): **any outcome that changes, changes the paper.**

## 1. The grid (rival-recon F3, verbatim — this is the whole task)
Per arm, per cell: **`lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`** (6×2),
best-of-grid **selected on the fit split only**; TTT arms additionally `b ∈ {1, 16}` (as in C2W4).
⭐ **Plus F3's sanity gate:** an arm below its literature range is **not rescued and no margin against
it is quotable** — on the gym there is no literature range, so the operational form is the **rescue
gate** (within 2 SE of its own blank ⇒ NOT RESCUED), unchanged from C2W4.
- ⚠ **Steps:** C2W4 used 400 outer steps and separately verified 5× budget (2000) does not help on the
  frontier. For THIS pass, run the grid at 400 **and** re-check the grid's best config at 2000 on
  `aggregate` — so "more steps would have rescued it" is closed with a measurement, not an assertion.
- **Arms — mandatory:** the three RESCUED arms (`ttt_linear`, `gdn`, `gdn2`), per the Head's ruling.
- **Arms — Hub rider (cheap, same rig; the Head may strike):** the same grid on `ttt_mlp` and
  `deltanet`. ⚠ Reason: their **NOT-RESCUED status is itself a tuning-sensitive outcome** — the
  referee's "you hobbled the competition" attack applies *most* strongly to an arm we declared dead on
  three learning rates. If the full grid rescues either, its full column set enters the audit table and
  the FB2/FB3 adjudications are re-run with it in.
- **Family:** `aggregate@base` only (the sole dividend family). ⛔ The byte-frontier column was declared
  NON-INFORMATIVE (0/5 rescued, under-training excluded at 5× budget) — **do not re-run it** unless the
  grid rescues an arm at `aggregate`, in which case re-check that arm's frontier row once, labelled.
- **Seeds:** 0, 1, 2 — identical to C2W4. Multi-seed before any paper number.

## 2. PREREG (`.claude/outputs/bprime-rivals-f3/PREREG.md`) — before ANY grid run
Register, with derivations:
1. **The C2W4 outcomes as priors**, per arm: rescue status · dividend vs own arg-min table · signed
   +0 B margin · raw-table margin. (They are the incumbent numbers; you are testing them.)
2. ⭐ **What counts as "an outcome changed"** — pre-commit the thresholds so adjudication is mechanical:
   a rescue status flips · R5's sign flips on any arm · any raw-table margin crosses 0 by > 2 SE · the
   R5 count (3 of 5 ≤ 0) changes · the P5-vs-raw gap (0.203–1.208) collapses below 2 SE of 0 on any arm.
3. **Your predicted per-arm deltas** from widening the grid (derived from the C2W4 fit-split losses,
   not guessed) — including the honest possibility that the low-lr half of the grid rescues nothing
   because the 31× fit→eval generalisation gap is geometric, not optimisation-limited.

## 3. Output — `.claude/outputs/bprime-rivals-f3.md`, protocol §5 format
- ⭐ **First screen: the before/after table** — every C2W4 audit number beside its full-F3 counterpart,
  per arm, with a CHANGED/UNCHANGED verdict against the §2.2 pre-registered thresholds, and one bolded
  sentence: **which paper claims change, if any.** `bprime-draft`'s number-freeze consumes this table
  directly.
- The chosen config per arm (lr, wd, b, steps) and the full grid's fit-split surface (so "best-of-grid"
  is auditable);
- the 2000-step re-check result;
- rescue-gate verdicts, first-class, including the two rider arms if run;
- PREREG scorecard · flag-provenance table (commit, seeds, every non-default flag, resolved JAX version)
  · reconciliation list in the first 10 lines if any · **declared NOT-RUNs, never nulls**.
- ⛔ **Never push `origin`.** Verify your branch ref from the main repo before removing the worktree.

## 4. File ownership
**Yours (inherited from the freed `bprime-rivals` slot):** `chlu/eval/rivals/` ·
`chlu/experiments/exp_bprime_rivals.py` · `tests/test_bprime_rivals.py`, `tests/test_rivals_ledger.py` ·
`chlu/eval/dividend.py` (**append-only**). **Everything else read-only**; `chlu/config.py` and
`chlu/eval/race.py` standing-frozen. `cluformer-pilot` (blocks.py, data/, training/) and the curator are
concurrent — no shared files. ⛔ **STOP and report if you need a file you do not own.**

## 5. Never-quote (delta from the full §0 list — these are yours specifically)
Any margin against a NOT-RESCUED arm · the frontier column as a dividend or a rival comparison ·
"verified to 1e-9 in all 28 cells" (24/28; corrected law) · any tier-ii/iii claim · **and now: any
C2W4 rival number in a draft without this pass's before/after verdict beside it** (the Head's
commitment, operationalised).
