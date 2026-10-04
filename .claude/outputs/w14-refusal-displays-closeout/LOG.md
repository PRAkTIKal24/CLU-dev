# w14 spoke/refusal-displays-closeout — LOG

⚠ **All timestamps are UTC** (`date -u`). The box's local clock is **UTC+2**, so the wall clock in the
terminal reads two hours ahead of every time below.
⚠⚠ **DISCLOSED: only the two boundary timestamps were actually captured.** `START_UTC.txt` and
`END_UTC.txt` were taken from `date -u`. ⛔ **No per-step clock was read**, so the steps below are
recorded **in order, without invented clock values.** (A first draft of this file carried per-step
times that had not been measured; they were removed rather than kept.)

**Agent:** `doc-curator`. **Branch:** `w14/refusal-displays-closeout`. **Base:** `base/iter8 = 6699665`.
**Worktree:** `C:\chlu\worktrees\CHLU-w14-refusal-displays-closeout`.
**Task file:** `/c/chlu/tasks/w14-spoke-refusal-displays-closeout.md`.
**Window:** **2026-09-22T06:14:40Z → 2026-09-22T06:20:58Z** (the whole pass is file reads and writing;
⛔ **nothing was executed that takes wall clock**).
**Scored runs: ⛔ NONE. Compute: ⛔ NONE. GPU: ⛔ NONE. Background jobs / PIDs launched: ⛔ NONE.
Box-wide kill: ⛔ never issued.** ⭐ No other agent's session, worktree, output directory or process
was touched, listed or signalled. ⛔ `$CHLU_CLAUDE_ROOT` was NOT opened at any point.

---

## Step 1 — START
`START_UTC.txt` written **first**: **2026-09-22T06:14:40Z**. Branch confirmed
`w14/refusal-displays-closeout`; worktree clean.
⭐ `chlu.__file__ = C:\chlu\worktrees\CHLU-w14-refusal-displays-closeout\chlu\__init__.py` — asserted
in this worktree before anything was written.

## Step 2 — task file read FIRST, standing-rules line included
Evidence base `/c/chlu/evidence/outputs/`. Write only to the output dir + worktree. Bundle own branch,
one ref. ⭐ **ZERO compute, ZERO scored runs.**

## Step 3 — programme state read (read-only)
- `outputs/w9-advisor-report.md` (16,248 B) — including **AMENDMENT 1** and **AMENDMENT 2** (the
  CMAZE-1 → **CMAZE-2** relabel, which this closeout carries).
- `outputs/w10-advisor-report.md` (9,236 B) · `outputs/w12-advisor-report.md` (10,881 B).
- `outputs/advisor-wave-plan-w12-w14.md:56` — ⭐ **the commission**: *"maze refusal displays dropped or
  declared uninterpretable (the refuser problem)"* is a **PRECONDITION of `hub/read-verb`**.

## Step 4 — ⭐ md5 OF THE EVIDENCE PAIR, RECORDED AT ARRIVAL (⛔ read-only, never written)

| file | md5 | lines |
|---|---|---|
| `/c/chlu/evidence/negative_results.md` | `c3597411aa834304be27110dab115673` | 3,794 |
| `/c/chlu/evidence/claims_matrix.md` | `39be4b5f1d8f29c5058c5a0bef935243` | 643 |
| `/c/chlu/evidence/outputs/w13-registry-fold/negative_results.md` | `835fad9331b77f8fe6f6b5136e9d4ef5` | 5,732 |
| `/c/chlu/evidence/outputs/w13-registry-fold/claims_matrix.md` | `8d8faa2fcbc92c79d97324acde744778` | 1,511 |

⛔ **The live `/c/chlu/evidence/` pair is STALE** — its md5s are exactly the two `w13
spoke/registry-fold` recorded as the live values at its own close, and at 3,794 lines it predates the
`w12r` repair and the 21 `N-I8-*` entries. ⭐ **BASE FOR THIS PASS: `outputs/w13-registry-fold/`**
(5,732 / 1,511 lines) — the same choice `w13` made, for the same reason.

## Step 5 — ⭐ THE FALSIFIER THE TASK FILE GAVE ME, RUN BEFORE ANY WRITING
> *"The registry already records it ⇒ verify and say so; do not duplicate the entry."*

**It fires, PARTLY. Verified in the base pair:**
- `negative_results.md:5120` — **`N323`** holds the MEASUREMENT in full.
- `negative_results.md:4662` — **`N-I8-19`** holds the CLU DISPLAY rows and the novelty control.
- `claims_matrix.md:1136` — **`CM-R12-3`** is the SCOPE row (*"not yet informative"*).

⛔ **NOT there, and this is the whole delta:** the **Head's retirement ruling**; a **BAR** to revive
(`N323` gives two *options*, not a bar); the statement that the **Shapes3D hole is untouched**; the
**per-seed** CLU rows and the `n` behind `0.8182`.
⇒ ⭐ **NO duplicate entry. One new entry (`N337`) that RULES, pointing at `N323` for the measurement.**

## Step 6 — every number re-read at its own artefact (⛔ nothing paraphrased from a report)
- `outputs/_thinkpad/w9/w9-harness/LOG.md:222-224` — separation table. ✅ matches `N323`.
- `outputs/_thinkpad/w9/w9-harness/LOG.md:228-236` — ridge-probe bands. ✅ matches `N323`.
- `outputs/_thinkpad/w9/w9-harness/refusal_separability.json` — ⭐ **all six per-run AUCs extracted**,
  so the band `0.48–0.55` is replaced by its endpoints, **`n_test = 200`** per run is printed, and the
  count above 0.50 (**3 of 6**) is stated rather than rounded away.
- `outputs/_thinkpad/w9/w9-harness/HARNESS-READY.json` → `notes.CAVEAT_refuser` (the caveat is in the
  machine-readable file, not only in prose) and `param_counts.refuser` = **59,982**.
- `outputs/_thinkpad/w9/w9-maze/LOG.md:543-565` — the CLU DISPLAY table.
- `outputs/_thinkpad/w9/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json` → `display.refusal.*` —
  ⭐ **the three scored runs read directly.**
- `outputs/w12-single-design-crossover/FINDINGS.md:294-300,455-462,464-466,482` — the Shapes3D hole
  and its declared NOT-RUN comparator.

## Step 7 — ⛔⛔ A DEFECT FOUND IN THE TASK FILE'S OWN NUMBERS, AND IN THE REGISTRY'S
The task file quotes *"false refusal **0.7979 / 0.8560 / 0.8182**"* as three seeds. ⛔ **It is not.**
The three scored runs give **0.797945 / 0.855960 / 0.800633**; **`0.8182` is the 3-SEED MEAN**, and it
is what `w9-maze/LOG.md:554`'s §8 table prints — as are the other three values of that table.
⭐ **This is the ONE arithmetic operation this pass performed** — four means of three quoted values
each — **declared here, marked `⊙-computed` in place in `N337` Part C, and re-run mechanically** in
`repro/VERIFY.txt` §6, where all four reproduce the printed table to four decimals.

## Step 8 — the delta written, as COPIES
`repro/build.py` assembles both files from the base plus four source fragments:
- `negative_results.md` ← base **+** the `⊙` WAVE-14 CLOSEOUT block (`⊙.0`–`⊙.3`, entry **`N337`**)
  **+** two **additive in-place pointer notes** (under `N323`, under `N-I8-19`).
- `claims_matrix.md` ← base **+ `§I8.8`** (`CM-W14-1`, `CM-W14-2`) **+ `§I8.9`** (6 open flags), title
  bumped to **v2.21 (PROPOSED)** with the prior title preserved verbatim as its tail.
⛔ Nothing renumbered. ⛔ No existing line edited or deleted. ⭐ Fold symbol **`⊙` (U+2299)** chosen
because it occurs **0** times in the base (`✦` w10, `⌖` w12r, `⊛` w13, `✯`/`✰` C2W10/C2W11 are taken).

## Step 9 — additivity verified MECHANICALLY, not asserted (`repro/verify.py` → `repro/VERIFY.txt`)
1. `difflib.SequenceMatcher` — `negative_results.md` **5,733 equal / 241 insert / 0 replace / 0
   delete**; `claims_matrix.md` **1,511 equal / 113 insert / 1 replace (line 1 only) / 0 delete**.
2. `claims_matrix.md` lines 2–1,512 **byte-identical**; new title `endswith` prior title **True**
   (17,371 → 19,153 chars).
3. `N` entry headings **349 → 350**, added `['N337']`, removed `[]`, **0 duplicates**; `CM` rows
   **31 → 33**, added `['CM-W14-1','CM-W14-2']`, removed `[]`, **0 duplicates**.
4. `N323`, `N-I8-19`, `CM-R12-3`, `N-I8-6`, `N332`, `N334` — **all present, each exactly once.**
   ⛔ **Nothing deleted.**
5. Evidence md5 — **byte-identical to arrival**, all four files.
6. The declared arithmetic re-run from the artefacts: all four means **match the printed table to 4
   dp**; ridge AUC **min 0.4822 / max 0.5518**, `n_test` set `[200]`, **3 of 6 runs above 0.50**;
   budget scalar **0.3840–0.4589**, **0 of 6** at or above 0.50.

## Step 10 — `CLOSEOUT.md` and `FINDINGS.md` written
`CLOSEOUT.md` 19,532 B / 282 lines · `FINDINGS.md` 210 lines · `negative_results.md` 5,732 →
**5,973** lines · `claims_matrix.md` 1,511 → **1,624** lines · `repro/` 7 files.

## Step 11 — END
`END_UTC.txt` written **last**: **2026-09-22T06:20:58Z**. Then: mirror into
`w14_refusal_displays_closeout/` in the worktree, commit, and
`git bundle create .../w14-refusal-displays-closeout.bundle w14/refusal-displays-closeout` —
⛔ **that branch only, never `--all`** — verified, with `list-heads` showing **exactly one ref**.

---

## Step 12 — the bundle, and the ordering problem stated rather than fudged

⚠⚠ **A bundle cannot contain its own hash.** A first cut was taken at `fad34d6`, then the record
commit was made, which moved the tip — so that cut was **no longer the branch**. ⛔ Rather than ship a
bundle that is one commit behind, or record a hash that is not the shipped file's, the order is fixed:

1. this LOG entry is committed **first**, describing the procedure and naming no hash it cannot know;
2. the bundle is cut **last**, at the tip this commit creates, with ⛔ **that branch only, never
   `--all`**;
3. the bundle's own **size, md5 and `list-heads` output** are recorded in
   `/c/chlu/outputs/w14-refusal-displays-closeout/LOG.md` and in `FINDINGS.md` — **outside the
   branch**, which is the only place they can be true.

⭐ **Therefore the in-branch copy of this file ends here, and the output-directory copy carries the
bundle's own numbers below.** ⛔ That divergence is deliberate and is stated in both copies.

```
git bundle create /c/chlu/outputs/w14-refusal-displays-closeout/w14-refusal-displays-closeout.bundle                   w14/refusal-displays-closeout
git bundle verify    <bundle>
git bundle list-heads <bundle>   # must print EXACTLY ONE ref
```

⭐ **FINAL EVIDENCE md5 RE-CHECK — byte-identical to arrival, all four files:**
`negative_results.md` **c3597411aa834304be27110dab115673** ·
`claims_matrix.md` **39be4b5f1d8f29c5058c5a0bef935243** ·
`outputs/w13-registry-fold/negative_results.md` **835fad9331b77f8fe6f6b5136e9d4ef5** ·
`outputs/w13-registry-fold/claims_matrix.md` **8d8faa2fcbc92c79d97324acde744778**.
⛔ **`/c/chlu/evidence/` was never written.**

---

## Step 13 — THE SHIPPED BUNDLE (output-directory copy only; ⛔ deliberately not in the branch)

**Branch tip / bundled commit:** `697a3cb41f769d58a50f295dbc13b7182da6f37a`
(`w14/refusal-displays-closeout`), three commits on `base/iter8 = 6699665`:
`fad34d6` (the closeout) → `f775387` (the first bundle record) → `697a3cb` (the ordering fix).

| check | result |
|---|---|
| `git bundle create … w14/refusal-displays-closeout` | ⛔ **that branch only; `--all` never used** |
| `git bundle verify` | ⭐ **"is okay"** · *"The bundle records a complete history"* · hash algorithm **sha1** |
| `git bundle list-heads` | ⭐ **`697a3cb41f769d58a50f295dbc13b7182da6f37a  refs/heads/w14/refusal-displays-closeout`** |
| ref count (`list-heads \| wc -l`) | ⭐ **exactly 1** |
| bundle size | **7,175,194 B** |
| bundle md5 | **684f85166c004121d388d049a37e4525** |
| worktree `git status` after the final commit | ⭐ **clean** |
