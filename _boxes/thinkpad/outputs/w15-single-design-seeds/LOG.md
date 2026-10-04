# w15 hub/single-design-seeds — LOG

Branch `w15/single-design-seeds`, worktree `C:/chlu/worktrees/CHLU-w15-single-design-seeds`,
base `base/iter8 = 148f3b7`. Output dir `C:/chlu/outputs/w15-single-design-seeds/`.

---

## 21:26Z — START

`START_UTC.txt` written: 2026-09-22T21:26:36Z.

### Environment — the assert, and a path-format defect in `env_w3.sh`

`source /c/chlu/env_w3.sh` alone FAILS on this box:

```
ModuleNotFoundError: No module named 'equinox'
```

Cause, measured: `env_w3.sh` builds
`PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"` where `_H` is written in MSYS form
(`/c/Users/PratikJawahar/...`). The interpreter on this box is a **native Windows**
python (`C:\Users\sshkey.FORGIS_14\AppData\Local\Programs\Python\Python311\python.exe`),
which cannot resolve a `/c/...` path, so the wave-2 site-packages entry is silently
inert and jax/equinox/chlu's dependencies are invisible. `$CHLU_ROOT` survived only
because the harness had already set the cwd in Windows form.

⭐ **Fix, recorded as a choice:** `outputs/w15-single-design-seeds/env.sh` sources
`env_w3.sh` **unchanged** and then re-exports the **same two entries** through
`cygpath -w`. No package set changes; the site-packages directory is byte-identical.
This is a path-format repair, not an environment change. ⛔ No venv was built, no
`uv sync` was run.

Asserted after the fix:

```
OK C:\chlu\worktrees\CHLU-w15-single-design-seeds\chlu\__init__.py
jax 0.9.0   numpy 2.4.1   equinox 0.13.4
```

`chlu.__file__` resolves INSIDE the worktree. Pinned env confirmed.

---

## 21:35Z — THE FIRST QUESTION, ANSWERED BY READING THE ARTEFACTS

The task file orders: *"Check first whether w12b already produced these seeds. If seeds 1
and 2 at K\*=32 on TEST already exist as scored artefacts, DO NOT RE-SPEND the runs."*

**They exist.** Established by reading, not by assuming:

| artefact | what it holds |
|---|---|
| `C:\chlu\outputs\w12-single-design-crossover\scored\run1_seed0_K32\scored.json` | seed 0, K=32, `condition: test`, `SCORED: true` |
| `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run3_seed1_K32\scored.json` | seed 1, K=32, `condition: test`, `SCORED: true`, `run_n: 3` |
| `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run3_seed2_K32\scored.json` | seed 2, K=32, `condition: test`, `SCORED: true`, `run_n: 3` |

Each carries `per_arm` for `cov_tensor` + the **four charter-C controls** + the rival
column + the `table` floor, `paired_vs_cov_tensor`, and the **four-part byte ledger**.
`C:\chlu\outputs\w12b-nr6-and-kstar\NR6-T7.md` already prints `K32: n = 3 seeds [0, 1, 2]`.

**And the runs are already CHARGED.** `/c/chlu/SCORED-LEDGER.md` lines 43 and 44:

```
2026-09-21T21:33:51Z | w12b hub/nr6-and-kstar | single-design | 2/3 | ... NR6 / THE MISSING K = 5 TEST NUMBER
2026-09-21T21:33:51Z | w12b hub/nr6-and-kstar | single-design | 3/3 | ... NR6 / T7 AT MORE THAN ONE SEED ... adds seeds 1 and 2 at K* = 32 on TEST
```

⇒ ⛔ **`single-design` is EXHAUSTED at 3/3.** The two runs this task file offered me are
not merely unnecessary — they were spent 24 hours ago on exactly this content, by the
hub that produced the artefacts. **DECLARED NOT-NEEDED and NOT AVAILABLE.**

⭐ **Decision (open choice, resolved toward the ICLR main-track result):** I spend zero
scored runs and write a ledger **NOTE**, not a run row, exactly as the task file
instructs. My effort goes entirely to the deletion rows.

---

## 21:40Z — WHAT IS ACTUALLY MISSING, and it is the deletion rows

Directory listing of the deletion artefacts:

```
w12-single-design-crossover/deletion/test_seed0_K32/deletion.json   <- seed 0 only
w12-single-design-crossover/deletion/val_seed0_K5/deletion.json     <- val, SMOKE
w12-single-design-crossover/deletion/probe_test_seed0_K32.json
```

⇒ **RESULT 3 of the PREREG (the deletion rows) exists at seed 0 and NOWHERE ELSE.**
w12b charged runs 2/3 and 3/3 and produced RESULT 1 (T7) only. The deletion rows at
seeds 1 and 2 are the **unproduced RESULT 3 of runs already charged**, on the same
already-fitted arms (`arm.pkl` is on disk for both seeds), the same K, the same
condition, the same PREREG. Completing them is not a new spend and I do not claim one.

Also missing at EVERY seed, including 0: **the rival deletion column.** Seed 0's
`deletion.json` has `display_refusal.comparator_status = "DECLARED NOT-RUN"` and no
rival arm at all. The task file names it: *"Rival deletes only by retrain-without, cost
printed. Also print rival as-trained."* That is genuinely new work and I do it at all
three seeds.

### The comparator checkpoints are on disk for all three seeds

`C:\chlu\outputs\w12-deletion-baselines\comparators\seed_{0,1,2}\` each hold
`{mlp,lstm,transformer,cont_mlp}__as_trained` and `__retrain_without__{value-object_hue-0,
conj-object_hue0-shape2,item-0.01}` (`.rival.eqx` + `.rival.json`), plus the
`unlearn_*` families. `BASELINES-READY.json` names the entry point
(`chlu.eval.rivals.w12_deletion_baselines.load_comparator`) and carries each one's
**cost line** (`seconds`, `data_rows_read`, `bytes_rewritten`, `passes_over_data`,
`gradient_row_visits`).

⚠ **Measured asymmetry, printed on every rival row:** the comparators were fitted at the
BUILD scale `n_fit_rows_used = 16,384` (`BASELINES-READY.json` → `per_seed.*.build_scale`)
while the store is fitted on the full **131,072**-row pool. This runs AGAINST this hub
in the same direction as the already-declared 1× problem: a rival fitted on 8× more rows
would be stronger. Declared, not netted, and printed.

⚠ **The `item` scope deletes DIFFERENT rows in the arm and in the comparator, at every
seed including 0, and this is structural, not a seed-offset bug.** The arm's mask indexes
the partition `pool` (`w12_deletion_rows.py`, `default_rng(20_260_921 + seed)` over
131,072 rows); the comparator's indexes its own `fit_rows` (`partition_fit_rows`,
`default_rng(20260921)` over 16,384 rows). Two different index spaces. ⭐ **Choice:** I
keep w12's mask byte-for-byte — the task file says PURE REPLICATION, no design changes —
and print the mismatch as a caveat on the item row rather than silently re-pinning it.
Both are "a random 1 % of the fit pool"; neither is the other's rows.

### The conjunction pin
⛔ `red ∧ cube` deletes **0** rows — cube is held out of the split entirely. Run and
reported with its count of 0, never as a passing row. The live pin is
`object_hue = 0.0 ∧ shape = 2.0`, named **"red sphere"** everywhere, adopted from
`BASELINES-READY.json` → `pins.conjunction_deletion` so arm and comparator delete the
same rows.

---

## 21:45Z — MACHINE STATE, and the parallelism choice

`nproc` = **4**. Four other `python.exe` processes are already resident (other agents'
work — ⛔ nothing of theirs is touched). Seed 0's deletion run took ~150 s per
109-corner scoring pass on an uncontended box.

⭐ **Choice:** two processes (seed 1, seed 2) at `OMP_NUM_THREADS=2` each, PIDs captured
to `PIDS.txt`, polled to completion inside this turn. The script writes its JSON after
every completed unit so a truncation still leaves usable artefacts.

---

## 22:00Z — THREE JOBS LAUNCHED, AND A FIRST FAILURE THAT WAS MINE TO FIX

### Launch 1 (21:37Z) died at once, and it produced no number

```
AssertionError: seed 1: feature cache is GENERATION 1, not 2.
```

`env_w3.sh` unsets every `CHLU_*` on entry and `CHLU_GEN_FEATURES_ROOT` is **not**
ambient on this box — `scripts/w12_launch.sh` says so in its own comment: *"a run that
forgets it silently reads the frozen bridge — cache GENERATION 1, which w5-common
section 3.8 forbids for any number, and which at bridge seeds 1 and 2 holds only 9,288
rows against the pool's 131,072 indices."* Added to `env.sh` at the same value w12 and
w12b use: `C:/chlu/outputs/spoke-a-rivals/cache_generation2`. The assertion fired
**before** any number, which is what it is for.

### Launch 2 (21:39Z), three jobs, PIDs in `PIDS.txt`

| job | what | windows pid |
|---|---|---|
| seed 1 | full: arm downdate legs + rival column | 34344 |
| seed 2 | full: arm downdate legs + rival column | 35120 |
| seed 0 | rivals only, arm legs carried from w12's `deletion.json` | 15400 |

⭐ **Choice:** three processes at `OMP_NUM_THREADS=2`. Box load was measured at 34 %
before the third launch, so the third job takes headroom rather than another agent's
compute. ⛔ Only these PIDs are ever stopped by this hub.

### ⭐ THE INTEGRITY CHECK PASSED AT BOTH SEEDS, EXACTLY

```
seed 1:  BEFORE reproduces the scored row: max dev 0.0
seed 2:  BEFORE reproduces the scored row: max dev 0.0
```

The before-deletion arm is re-scored here from the **same `arm.pkl`** and reproduces the
already-charged T7 per-corner row **bit for bit** at both seeds. Without this the
deletion legs would not be measured against the published row.

### ⛔ A DEFECT OF w12's HARNESS, FOUND AT SEED 1 AND FIXED BEFORE ANY NUMBER

Seed 1 crashed in `chlu/core/gen_read.py:133`:

```
ValueError: matmul: ... (size 43 is different from 44)
```

**Cause, measured.** `w12_deletion_rows.py` carries the address book onto the
**width-matched** post-deletion basis (`match_basis_width`) and then reads the store
through the **raw** one. Where the two widths agree those are the same array — and at
**seed 0 they always agreed**, `basis_width_adjustments` is `[]` on every scope, which
is why w12 never saw this. At **seed 1** they do not: after the 11,044 red rows go, the
mean's perpendicular part in one well falls below `pinned_basis`'s 1e-4 admission
threshold, so the raw post basis is **43** columns wide against the book's **44**.

⭐ **Fix, recorded as a choice:** the book is read through the basis it was **carried
onto** (`U_post`). The padding column is exactly zero, so the book's coefficient for a
direction the store no longer holds is projected onto nothing — which is what the
deletion *means*, and it is what `match_basis_width`'s own docstring already says the
padding is for. The adjustment is recorded on the row in `basis_width_adjustments`, not
done silently. ⛔ At seed 0 the two arrays are identical, so **this changes no published
number**; it is the same class of defect as AMENDMENT 1 §A1.4 and is handled the same way.

Seeds 1 and 2 relaunched at 22:18Z with the fix (win pids 28880, 12260). Seed 2 was
stopped with `kill` on **its own pid**; ⛔ no box-wide kill was run at any point.

### Numbers already on disk from the arm legs

```
seed 1  SCOPE VALUE   11,044 rows   adaptation 2.0202 s   2,074,368 B rewritten   0 data rows read
        exactness max|mode dev| 4.821e-08   max subspace dist 2.367e-08   ranks agree True
        refit-without 20.3 s, read 120,028 rows
seed 2  SCOPE VALUE   11,044 rows   adaptation 2.489 s    2,078,464 B rewritten   0 data rows read
        exactness max|mode dev| 2.011e-08   max subspace dist 3.408e-08   ranks agree True
        refit-without 19.5 s, read 120,028 rows
```

⚠ **The sphere caveat is BIGGER at seeds 1 and 2 than at seed 0 and it is reported, not
buried.** `landscape_removed`'s judged-sphere excess reads **+0.6709** at both seeds 1
and 2 against **−0.3291** at seed 0: at these seeds a zeroed store degrades *towards*
sphere rather than towards cube. That is precisely the hazard
`gen_value_holdout.py:299` named. ⛔ It does not touch the control gate (a degraded arm
still scores 0 valid novel tuples, because `judge_valid` requires the *whole* requested
tuple), but it does mean the red-sphere conjunction row must be read with its caveat
column, and the caveat column is measured per seed rather than carried from seed 0.

---

## 22:26Z — THE ARM LEGS ARE DONE AT BOTH SEEDS, AND THEY REPLICATE SEED 0

`deletion/test_seed1_K32/deletion.json`, `deletion/test_seed2_K32/deletion.json`:

| seed | scope | rows deleted | survivors | novel/corner before → after | paired | store-scoped exactness | width adj. |
|---|---|---|---|---|---|---|---|
| 1 | value | 11,044 | 89 | 50.528 → 50.528 | +0.0000 | 4.821e-08 | **1** |
| 1 | conjunction (red ∧ cube) | **0** | 109 | 47.404 → 47.404 | +0.0000 | 0.000e+00 | 0 |
| 1 | conjunction_live (red sphere) | 5,511 | 109 | 47.404 → 47.404 | +0.0000 | 1.074e-08 | 0 |
| 1 | item (1 %) | 1,311 | 109 | 47.404 → 47.404 | +0.0000 | 1.486e-08 | 0 |
| 2 | value | 11,044 | 89 | 52.607 → 52.607 | +0.0000 | 2.011e-08 | 0 |
| 2 | conjunction (red ∧ cube) | **0** | 109 | 49.119 → 49.119 | +0.0000 | 0.000e+00 | 0 |
| 2 | conjunction_live (red sphere) | 5,511 | 109 | 49.119 → 49.119 | +0.0000 | 6.253e-08 | 0 |
| 2 | item (1 %) | 1,311 | 109 | 49.119 → 49.119 | +0.0000 | 3.162e-08 | 0 |

⭐ The one `basis_width_adjustments` entry in the whole table is seed 1 / value — exactly
the well the crash diagnosed. It is recorded on the row.

## ⚠ 22:30Z — THE RESULT THAT DOES NOT FLATTER THIS HUB, AND IT REPLICATES AT ALL THREE SEEDS

Deleting **every** red row leaves the **red corners** able to compose red:

| seed | value scope, the 20 DELETED corners | novel/corner before → after | judge accuracy before → after |
|---|---|---|---|
| 0 | 20 | 32.250 → 32.250 | 0.8958 → 0.8958 |
| 1 | 20 | 33.500 → 33.500 | 0.9306 → 0.9306 |
| 2 | 20 | 33.600 → **33.550** | 0.9333 → **0.9319** |

⭐ This is the PREREG's §5 point measured at n = 3 rather than asserted: **data forgetting
and capability removal are two different things on this object.** The downdate is
store-scoped exact (the store after deletion equals the store fitted without those rows,
1e-8), but the *capability* to render red survives, because red is still reachable from
the surviving factor structure through the address book. ⛔ Removing the capability is
the **hole's** job, and the hole is a DISPLAY row, not a claim.

⛔ **Stated plainly rather than buried:** if a reader's question is *"after deletion, can
the system still produce the deleted content?"*, the answer on this object is **yes**,
at all three seeds, and the deletion claim here is **store-scoped exactness plus a
data-free adaptation cost**, not capability erasure.

---

## 23:31Z — ALL THREE JOBS FINISHED CLEAN

```
wrote C:\chlu\outputs\w15-single-design-seeds\deletion\test_seed1_K32\deletion.json
wrote C:\chlu\outputs\w15-single-design-seeds\deletion\test_seed2_K32\deletion.json
wrote C:\chlu\outputs\w15-single-design-seeds\deletion\test_seed0_K32\rivals.json
```

No job was left detached. Seed 0 finished at 22:52Z, seeds 1 and 2 at 23:31Z.

## 23:32Z — THE ACCEPTANCE ARTEFACT

`scripts/w15_t7_3seeds.py` → `T7-3SEEDS.json`; `scripts/w15_findings_tables.py` →
`T7-3SEEDS.md`. Both read JSON on disk and compute nothing of their own except the
seed-wise mean/sd and the rival's own paired before/after.

```
n = 3 seeds [0, 1, 2]
ALL FOUR CONTROLS COLLAPSE ON EVERY SEED: True
cov_tensor row3 32.350 ± 0.815  [31.600, 32.233, 33.217]
cov_tensor row4 67.177 ± 1.320  [66.959, 65.980, 68.592]
```

⭐ **One improvement made to the rival column after the first render, and it is free.**
The rival's as-trained figure was initially the all-109-corner number while its
retrain-without figure was on the scope's own survivors — an 89-corner against a
109-corner quantity on the `value` row, which is not a before/after on the same corners.
The per-corner rows were already in the scored artefacts, so the as-trained figure is now
restricted to each scope's **own survivor corners** and the rival's **own**
after-minus-before is printed beside ours. No re-scoring; `_survivor_as_trained` in
`scripts/w15_t7_3seeds.py`.

## 23:40Z — EVERY NUMBER IN `FINDINGS.md` VERIFIED AGAINST THE JSON

`scripts/w15_verify_findings.py` walks `T7-3SEEDS.json` and asserts each literal appears
in `FINDINGS.md`, and re-asserts the three properties the claim rests on:

```
MISMATCHES: 0
```

(every control exactly 0.0000 · every deletion paired delta exactly 0.0000 · every rank
agrees with the refit-without store · every rival leg beats the 1.0 margin)

It caught one transcription error on the way (`46.083 → 21.97` doubled into the rival
seconds cell at `item`/seed 2/`transformer`), which is what it is for.

---

## CHOICES MADE WHERE THE BRIEF LEFT ONE OPEN — the full list

1. **Spend 0 scored runs.** The task file said to check first; the check said the runs
   were already charged by w12b on exactly this content and the target is exhausted at
   3/3. A ledger **NOTE**, not a run row. (§1 of FINDINGS.)
2. **Charge nothing to the `deletion` target either.** Its 1 remaining run is w5/w6's
   FIELD-STORE line — a different object. The deletion rows on *this* object have always
   been RESULT 3 of the `single-design` PREREG and are charged there.
3. **Keep w12's `item` mask byte-for-byte** (`default_rng(20_260_921 + seed)`) even
   though the comparator's mask lives in a different index space, because the task file
   says PURE REPLICATION. The mismatch is printed on the row instead.
4. **Read the post-deletion store through the width-matched basis.** The only coherent
   option once the widths disagree, identical to w12 wherever they agree, and recorded on
   the row.
5. **Run the rival column at seed 0 too**, not only at seeds 1 and 2 — it was absent at
   every seed, and a rival column at 2 of 3 seeds would not print at n = 3.
6. **Do NOT score the `unlearn_neggrad` / `unlearn_finetune` comparators.** The task
   file's rule is *"rival deletes only by retrain-without"*; scoring them would have added
   24 more 109-corner passes and is not what the row asks for. DECLARED NOT-RUN with the
   path where they sit.
7. **Restrict the rival's as-trained figure to each scope's own survivors** so the rival
   column is a genuine before/after. Free, from rows already on disk.
8. **Report the capability-survival result as a headline limitation** (§6 of FINDINGS)
   rather than as a footnote to the composition row. It is the question an ICLR reader
   asks first about a deletion claim, and the honest answer is not the flattering one.

---

## 23:40Z — CLOSE-OUT

All three artefacts complete and verified by reading them back:

```
deletion/test_seed1_K32/deletion.json  DONE=True  scopes 4  rival scopes 4
deletion/test_seed2_K32/deletion.json  DONE=True  scopes 4  rival scopes 4
deletion/test_seed0_K32/rivals.json    DONE=True  scopes 4  rival scopes 4
```

⚠ Seeds 1 and 2 were still resident 8–10 minutes after their final write, burning CPU
and holding ~10 GB each in interpreter/XLA teardown on a 4-core box shared with other
agents. This hub stopped **those two PIDs only**, each re-verified by its command line
naming `w15_deletion_rows` immediately before the stop. ⛔ **No box-wide kill was run at
any point.** The four other `w15_*` python processes on the box belong to
`w15-cmaze-o-harness` and `w15-cmaze-t-harness` and were confirmed running and
**untouched**.

Branch committed whole (`fda8ead`), bundled to
`w15-single-design-seeds.bundle` and `git bundle verify`d: *"is okay … records a
complete history."* Outputs verified by directory listing, since `git status` cannot see
them.

**Ledger:** the NOTE (21:31:47Z, before any number) and the OUTCOME note (23:37:00Z) are
both in `/c/chlu/SCORED-LEDGER.md`. ⭐ **Scored runs spent by this hub: 0.**
`single-design` remains **EXHAUSTED at 3/3** and no balance moved.
