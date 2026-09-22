# w14 spoke/refusal-displays-closeout — FINDINGS (report to the Hub)

**Agent:** `doc-curator`. **Branch:** `w14/refusal-displays-closeout`. **Base:** `base/iter8 = 6699665`.
**Window:** `START_UTC.txt` **2026-09-22T06:14:40Z** → `END_UTC.txt`. ⚠ **All times UTC** (box local
clock is UTC+2).
**Scored runs: ⛔ NONE. Compute: ⛔ NONE. GPU: ⛔ NONE. Background jobs / PIDs launched: ⛔ NONE.
Box-wide kill: ⛔ never issued.** ⭐ No other agent's session, worktree, output directory or process
was touched, listed or signalled. ⛔ `$CHLU_CLAUDE_ROOT` was NOT opened at any point.
⭐ `chlu.__file__ = C:\chlu\worktrees\CHLU-w14-refusal-displays-closeout\chlu\__init__.py`, asserted in
this worktree before anything was written.
⛔ **`/c/chlu/evidence/` was NOT written.** md5 re-checked at close, byte-identical to arrival:
`negative_results.md` **c3597411aa834304be27110dab115673** ·
`claims_matrix.md` **39be4b5f1d8f29c5058c5a0bef935243** ·
`outputs/w13-registry-fold/negative_results.md` **835fad9331b77f8fe6f6b5136e9d4ef5** ·
`outputs/w13-registry-fold/claims_matrix.md` **8d8faa2fcbc92c79d97324acde744778**.
⛔ **The curator re-derived no statistic**, with **ONE declared exception**: four column means, marked
**⊙-computed** in place, each from three values quoted with their own paths and each re-checked
mechanically in `repro/VERIFY.txt` §6.

---

## 0. WHAT SHIPPED

| file | what it is | size |
|---|---|---|
| `CLOSEOUT.md` | ⭐ **the deliverable** — the ruling, the measurement that forces it, the revival bar, the Shapes3D non-generalisation, the not-runs | 19,532 B / 282 lines |
| `negative_results.md` | the registry **+ the `⊙` WAVE-14 CLOSEOUT block**: entry **`N337`**, sections `⊙.0`–`⊙.3`, and **two** additive in-place pointer notes (under `N323` and `N-I8-19`) | 5,732 → **5,973 lines** |
| `claims_matrix.md` | the matrix **+ `§I8.8`** (`CM-W14-1`, `CM-W14-2`) **+ `§I8.9`** (6 open flags), title bumped to **v2.21 (PROPOSED)** | 1,511 → **1,624 lines** |
| `repro/` | `build.py` (assembles the copies), `verify.py`, `VERIFY.txt` (the run), and the four source fragments | — |
| `LOG.md`, `FINDINGS.md`, `START_UTC.txt`, `END_UTC.txt` | the working record and the clock | — |

⭐ **BASE:** `/c/chlu/evidence/outputs/w13-registry-fold/` — the newest pair (**5,732 / 1,511** lines).
⛔ **The live `/c/chlu/evidence/` pair is STALE** (**3,794 / 643** lines, and its md5s are exactly the
two `w13` recorded as the live values at its own close); basing on it would have dropped the `w12r`
repair, the 21 `N-I8-*` entries **and** the whole wave-12 fold.

---

## 1. ⭐ THE FALSIFIER, RUN FIRST — IT FIRES, PARTLY, AND THE PARTIAL IS THE WHOLE DELTA

> *"The registry already records it ⇒ verify and say so; do not duplicate the entry."*

⭐ **Verified in the base, and NOT duplicated:** the measurement lives at `negative_results.md:5120`
(**`N323`** — separation table, ridge-probe table, cause, scope) and at `:4662` (**`N-I8-19`** — the
CLU DISPLAY rows and the novelty control); the scope row is `claims_matrix.md:1136` (**`CM-R12-3`**).

⛔ **FOUR THINGS ARE NOT THERE, and they are exactly what shipped:**
1. ⛔ **The Head's RETIREMENT ruling** (2026-09-21, `outputs/advisor-wave-plan-w12-w14.md:56`) and the
   fact that it is a **precondition of `hub/read-verb`**, not an editorial preference.
2. ⛔ **A BAR to revive the display.** `N323` gives *"what would change it"* as two **options**.
   ⭐ An option is not a bar; `N337` Part D states one.
3. ⛔ **That the Shapes3D refusal-via-the-HOLE is untouched by this.** Nothing in the base stops a
   reader generalising the maze's failure to the store's refusal mechanism.
4. ⛔ **The per-seed CLU rows, and the `n` behind `0.8182`.** See §3.

⇒ ⭐ **One new entry, `N337`, which RULES and points at `N323` for the measurement. Two additive
in-place pointer notes. Two matrix rows. ⛔ Nothing renumbered, nothing edited, nothing deleted.**

---

## 2. ⭐⭐ THE NUMBER THAT RETIRES THE DISPLAY, now with its endpoints and its `n`

`outputs/_thinkpad/w9/w9-harness/refusal_separability.json` — **all six runs, `n_test = 200` each**:

| run | possible vs **all** impossible | budget | legality | budget scalar alone |
|---|---|---|---|---|
| `cmaze1/seed0` | 0.4834 | 0.5254 | 0.7489 | 0.4469 |
| `cmaze1/seed1` | 0.5125 | 0.4352 | 0.8474 | 0.4589 |
| `cmaze1/seed2` | 0.5354 | 0.5645 | 0.6662 | 0.4421 |
| `cmaze1-easy/seed0` | **0.5518** ← max | 0.6256 | 0.5666 | 0.4150 |
| `cmaze1-easy/seed1` | **0.4822** ← min | 0.4900 | 0.5354 | 0.4449 |
| `cmaze1-easy/seed2` | 0.4862 | 0.5509 | 0.7289 | **0.3840** ← min |

⭐ **`N323`'s band `0.48–0.55` is exactly right.** ⚠ **Stated exactly and not rounded in our favour: 3
of the 6 point estimates are above 0.50 and 3 below**, all at `n_test = 200`; `N323` reads the spread
as **chance**. ⭐⭐ **The budget scalar reaches 0.50 on none of the six runs** ⇒ ⛔ no shortcut.
⭐ **The `n` is new to the record** — the registry carried the band without it.

Separation (`outputs/_thinkpad/w9/w9-harness/LOG.md:222-224`): **+0.047 [0.000, 0.115]** (`cmaze1`),
**+0.020 [−0.005, 0.050]** (`cmaze1-easy`) — ⛔ **neither interval excludes 0**; class weighting moves
only the operating point. Refuser size **59,982 params**
(`HARNESS-READY.json` → `param_counts.refuser`).

⇒ ⭐⭐ **Deciding solvability-within-budget is close to solving the maze — a property of the TASK.**
⛔ Not a wiring bug, not a defect of the model, not a defect of CMAZE-1 or CMAZE-2.

---

## 3. ⛔⛔ A DEFECT FOUND AGAINST THE TASK FILE **AND** THE REGISTRY — `0.8182` IS A MEAN

`outputs/_thinkpad/w9/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json` → `display.refusal.*`:

| condition | seed 0 | seed 1 | seed 2 | ⊙-computed mean | printed at `w9-maze/LOG.md:548-554` |
|---|---|---|---|---|---|
| `budget_infeasible` | 0.971429 | 0.969231 | 0.939655 | **0.960105** | 0.9601 |
| `walled` | 0.960573 | 0.984634 | 0.995221 | **0.980143** | 0.9801 |
| `mt_unseen` | 0.972077 | 0.979401 | 0.982919 | **0.978132** | 0.9781 |
| `grounded_possible` | 0.797945 | 0.855960 | 0.800633 | **0.818179** | 0.8182 |

⛔ **All four printed values are 3-SEED MEANS, and `N-I8-19`, `CM-R12-3` and
`outputs/w9-advisor-report.md` each print them bare, with no `n`.**
⛔ **`0.8182` is NOT seed 2 — seed 2 is `0.800633`.**
⚠ **The task file commissioning this spoke writes *"0.7979 / 0.8560 / 0.8182"* as three seeds.** The
three seeds are **0.7979 / 0.8560 / 0.8006**. ⭐ **Corrected in the copies, not inherited.**
⭐ **All four means reproduce the printed table to four decimals** — which is how the "these are means"
claim is **established rather than asserted** (`repro/VERIFY.txt` §6).

⭐⭐ **AND THE NOVELTY CONTROL COMES OUT STRONGER, not weaker: it fires on EVERY seed**, which the mean
alone could not show. `mt_unseen` is **POSSIBLE** and is refused at **0.9721 / 0.9794 / 0.9829**,
inside the IMPOSSIBLE conditions' own per-seed range on all three (`budget_infeasible` 0.9397–0.9714,
`walled` 0.9606–0.9952) ⇒ ⛔ **novelty detection, not wall detection**, with a false-refusal floor of
**0.7979** against a **pre-registered 0.20**.

---

## 4. ⭐⭐ THE BAR TO REVIVE, AND THE TRAP BESIDE IT

| # | condition | today |
|---|---|---|
| 1 | ⭐⭐ **BINDING** — probe of the **same observation** at `t = 0` must clear **AUC 0.50 on every seed × tier run**, `n_test` printed, **declared before any DISPLAY row is scored** | ⛔ **FAILS** — 3 of 6 runs below 0.50; spread read as chance |
| 2 | separation interval must **exclude 0** | ⛔ **FAILS** |
| 3 | `mt_unseen` must separate from `walled`; false refusal **< 0.20** | ⛔ **FAILS** |
| 4 | ⛔⛔ **TRAP** — making impossibility **perceptually detectable** is choosing the answer | ⚠ the legality probe already hits **0.8474** *because extra tints change the image*; w9 said *"anyone who takes it should declare it loudly"* |

⭐ **1 alone re-opens the comparator. 1+2+3 re-opens the display row.**
⚠ **It is a CURATOR DRAFT of a bar, not a ratified pre-registration** — ⛔ only the Head makes it
binding (`§I8.9` item 5).

---

## 5. ⭐⭐ THE SHAPES3D HOLE IS UNTOUCHED — stated explicitly, with its honest other half

⛔⛔ **Do not generalise.** The maze display is a **learned** refusal score; the Shapes3D refusal is a
**structural hole** in the request count / coupling table.

- `N-I8-6`: the hole refuses `red ∧ cylinder` at **1.0000** for **ONE byte**, **0** bytes outside the
  permitted range, **no refit**, at **EXACTLY 0.0000** composition cost, **101** other never-held
  conjunctions untouched, **zero seed spread** (`outputs/w5-advisor-report.md` §2;
  `outputs/w4-advisor-report.md` §4).
- `N332` (w12): **two** holes, `red ∧ cube` and `red sphere`, **1 byte each, 2 bytes total, 0 data rows
  read, no refit**, index space **asserted** equal to `gen_field`'s (`layout_equals_gen_field: true`,
  **1,322 pair slots, 60 indices checked**); ⭐⭐ **`red ∧ cube` is refusable although no red cube was
  ever written** (`outputs/w12-single-design-crossover/FINDINGS.md:294-300,455-462`).

⚠⚠ **The honest other half, so *"untouched"* is not read as *"complete"*:** the Shapes3D display row's
**comparator is a DECLARED NOT-RUN** (`N334`, `CM-W12-9`) — the refuse-token transformer exists in
`BASELINES-READY.json`; the wall clock ran out (`FINDINGS.md:464-466,482`).

⭐⭐ **The distinction that decides both rows:** ***the Shapes3D gap is a BUDGET not-run, which a wave
can close by spending; the maze's is a TASK property, which spending does not close.***

---

## 6. ⭐⭐ ADDITIVITY, MECHANICALLY VERIFIED — `repro/VERIFY.txt`

- `negative_results.md`: `difflib.SequenceMatcher` over base vs new — **5,733 lines `equal`, 241
  `insert`, ⭐ ZERO `replace`, ZERO `delete`** (3 equal blocks, 3 insert blocks). **Not one existing
  line was edited or removed.**
- `claims_matrix.md`: **lines 2–1,512 byte-identical** (list comparison **True**); **line 1 is the only
  changed line** and the **entire prior title is preserved verbatim as its tail**
  (`str.endswith` **True**, 17,371 → 19,153 chars); **113 lines appended.**
- **Numbering:** `N` entry headings **349 → 350**, added `['N337']`, removed `[]`, **0 duplicates**.
  `CM` rows **31 → 33**, added `['CM-W14-1','CM-W14-2']`, removed `[]`, **0 duplicates**.
  ⛔ **Nothing renumbered ⇒ no redirect owed.** Fold symbol `⊙` (U+2299): **0** occurrences in the base,
  so it collides with no earlier fold (`✦` w10, `⌖` w12r, `⊛` w13, `✯`/`✰` C2W10/C2W11).
- **The retired rows are present, each exactly once:** `N323`, `N-I8-19`, `CM-R12-3`, `N-I8-6`, `N332`,
  `N334`. ⛔ **Nothing was deleted.**
- **Evidence md5 re-checked at close:** byte-identical to arrival on all four files.

---

## 7. ⛔ DECLARED NOT-RUN

| not run | reason |
|---|---|
| ⛔ **any scored run, any compute, any GPU** | ⭐ the task file forbids it. Nothing ran but file reads, four means, and one `import chlu` path assertion. |
| ⛔ **re-running the w9 refuser at a larger budget** | the **revival bar** itself; belongs to whichever wave takes the target. ⛔ `maze` is **0/3, exhausted in w9**. |
| ⛔ **the ridge probe at a larger `n_test`** | a new measurement. The band is quoted at the `n_test = 200` it was measured at, and the `n` is now printed. |
| ⛔ **scoring the Shapes3D refuse-token comparator** | ⛔ not this spoke's — `N334`, held for after submission by Head ruling. |
| ⛔ **fixing the CMAZE-2 request layer** | ⛔ not this spoke's — `tasks/w14-spoke-request-layer-fix.md` owns it. ⭐ Seen as a **name in a directory listing only**, never opened; that spoke's worktree, output dir and processes never touched. |
| ⛔ **amending `outputs/w9-advisor-report.md`** for the 3-seed-mean defect | ⛔ read-only evidence and not this spoke's file. Raised for the Hub (`§I8.9` item 2). |
| ⛔ **editing `CM-R12-3`, the task file, or any existing line** | additive-only discipline (`§I8.9` items 1, 3). |
| ⛔ **writing `/c/chlu/evidence/`** | read-only by standing rule; md5 re-checked. |
| ⛔ **confirming a version, promoting a tier, renumbering** | ⭐ the curator never self-confirms. `N337` **CURATOR-PROPOSED A**; `v2.21` **PROPOSED**; `v2.20` and below stay PROPOSED for their own Hubs. |

---

## 8. ⚠ OPEN FLAGS FOR THE HUB (⛔ raised, deliberately not resolved)

1. ⚠ **`CM-R12-3` keeps *"not yet informative"*.** Pointer-only, or a later fold adds a one-line
   in-place note there? ⛔ Not the curator's call.
2. ⛔⛔ **`outputs/w9-advisor-report.md` prints `0.8182` and three companions as bare values when they
   are 3-seed means.** The copies are corrected; the report is not this spoke's file. **Does it take
   an amendment?**
3. ⚠ **The commissioning task file carries the same defect.** Recorded, not inherited.
4. ⚠ **`N337` tier CURATOR-PROPOSED A; `v2.21` PROPOSED.**
5. ⚠ **The revival bar is a curator draft of a bar.** ⛔ Only the Head can make it binding — and it
   should be made binding **before** anyone spends on a maze refuser again.
6. ⚠ **`N-I8-21`(b)** touches this pass (the refuser is an instrument reporting on itself), ⭐ but it
   was caught **by its own spoke, against its own interest** — the healthy form. ⛔ Not filed as a
   seventh instance; the promotion question stays the Head's.

---

## 9. ⭐ WHAT THIS UNBLOCKS

⭐⭐ **`hub/read-verb`'s precondition 2 is DISCHARGED** — *"maze refusal displays dropped or declared
uninterpretable"* (`outputs/advisor-wave-plan-w12-w14.md:56`). ⛔ **Precondition 1 — the CMAZE-2
request layer fixed so routes need more than one key (rule B exercised) — is NOT this spoke's and is
NOT discharged here**; `tasks/w14-spoke-request-layer-fix.md` owns it.
