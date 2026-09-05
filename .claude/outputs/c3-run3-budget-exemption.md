# c3-run3-budget-exemption — experiment-engineer report

**Task + acceptance criterion:** build the narrow, auditable **pre-registered-continuation
exemption** so run 3 (= run 2 + `erosion_partition=True`, geometry unchanged) trains with a full,
**stamped** byte ledger; every break-attempt refused; the interim-budget ladder guard live; the
no-dtype-normalisation ruling in code; suite green at a named HEAD.

**Status: done.** ⚠ **One deliberate design deviation from the task's literal wording, forced by a
measured fact** (§2.1) — the declaration is a **run argument, not a `PilotConfig` field**. As a
field it would itself be a *second* differing key and would **break the very identity check it is
verified by**. Everything else is as specified.

> ⚠ **RECONCILIATION LIST — needs an owner (protocol §5 corollary, in the first 10 lines).**
> 1. ⭐ **Run 3's launch line must carry `PREREG_CONT=...` on EVERY submission *and every
>    re-resume*** (§6). Without it the budget refuses the leg. Passthrough added to
>    `job_gpu_c3_seeds.sh`; **the Hub/Head writes the actual submission line.**
> 2. ⚠ **`BUDGET_IS_INTERIM=True` / `BUDGET_CEILING_PREREG=None` are live and BLOCK rival-ladder
>    arms.** A concurrent spoke (`.claude/scratch/c3-rival-ladder-prereg/`) appears to be filing
>    that prereg. When it lands, **one edit** flips both plus the digit (§4.2). Until then the
>    ladder cannot train — **by ruling, deliberately.**
> 3. 🔍 **Pre-existing, not mine:** `rival_reference_table()` raises `LedgerError` for any budget
>    below ≈197 kB (mamba2 cannot shrink that far), so the ledger cannot be built at a very small
>    budget. Harmless at every real budget; noted because it cost me a demo route (§5.2).

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial:** **none — instrument/plumbing.** ⛔ No claim, no number a paper could quote. Every
  number below is either config arithmetic, a property of the instrument, or a toy-scale execution
  trace.
- **Laundering control:** n/a — but the **inverse obligation** applies: this mechanism *weakens a
  guard*, so the deliverable is the **set of things it still refuses** (§3, and the table in §7).
- **Falsifies the task:** any config that is not exactly run-2-plus-the-registered-flag being
  accepted; the exemption suppressing the ledger rather than annotating it; the unledgered-arm
  check being reachable through this path. **All three were attacked in tests and none fired.**
- **Does NOT falsify:** the pilot geometry still being over the interim budget. **It is — measured
  2.6338× below — and that is deliberate and not this task's problem.**

**Pre-registration:** ⛔ not required, none filed — the criterion is an instrument, not a measured
ratio/exponent/slope/law.

---

## 1. What was built

| piece | where | what it does |
|---|---|---|
| `ContinuationExemption` | `chlu/eval/byte_ledger.py` | frozen, **all fields mandatory** record of a *verified* continuation. ⛔ No default-constructed instance exists ⇒ no forged one. |
| `verify_preregistered_continuation()` | `chlu/experiments/exp_cluformer_pilot.py` | **the only constructor.** Delegates identity to `load_journal`. |
| `flag_block()` | same | `run_pilot`'s flag dict **extracted verbatim**, so the check and the run cannot drift on what the config identity *is*. |
| `build_byte_ledger(..., exemption=)` | `byte_ledger.py` | **annotates, never suppresses** (§3.2). Refuses a plain dict: the *type* is the proof the check ran. |
| `format_ledger_summary()` | `byte_ledger.py` | the ledger printed **every** run, more loudly when exempt. |
| `assert_ladder_arms_admissible()` | `byte_ledger.py` | ⛔ the **interim-ceiling guard** (§4.2). |
| `--prereg-continuation` + `PREREG_CONT=` | CLI + `scripts/csf3/job_gpu_c3_seeds.sh` | the launch path (§6). |

### 1.1 The identity check is DELEGATED, not reimplemented (task §1.1)

The question asked of the existing machinery is a **counterfactual**:

> *"Put the ONE registered knob back to the journal's setting — is this still the SAME LEG?"*

`load_journal` answers. If it accepts, everything except that knob is identical; if it refuses, it
refuses **by name**, and its message is propagated verbatim into the refusal. ⭐ This inherits the
§A20.4 loosening for free — which run 3 **needs**, because the real run-2 journals predate
`erosion_partition` entirely (measured, §5.1). ⛔ No second comparison exists to drift.

### 1.2 ⚠⚠ MEASURED TRAP — one knob moves TWO fingerprint keys

I first wrote the counterfactual at the *fingerprint-key* level. Measured against the **real** run-2
journal, that is wrong:

```
run 3 vs csf3_outs/run2/pilot_pilot_seed0_PARTIAL.json, differing keys:
  ['memory.erosion_partition', 'pilot.memory']        (+3 the journal predates, forgiven)
```

`flags["pilot"]` is `as_flag_table()`, and `memory` **is a `PilotConfig` field**, so the memory
override dict appears in the fingerprint **twice** — resolved (`memory.*`) and raw (`pilot.memory`).
Reverting one key leaves the other differing ⇒ the leg this exists for would have been **refused**.

⇒ **The revert is at the CONFIG KNOB** (`_revert_knob`), which regenerates both consistently. The
unit stays **one registered flag**; its derived consequences travel with it and are **stamped**
(`fingerprint_keys_moved`). ⛔ Failure direction is safe by construction: a wrong revert can only
cause a **false refusal**, never a false acceptance — `load_journal` still has the last word.

---

## 2. The deviation, and why it is not a softening

### 2.1 ⛔ The declaration is a RUN ARGUMENT, not a `PilotConfig` field

The task says "add `preregistered_continuation` to the run config". I built it that way first, then
measured why it cannot be:

1. a `PilotConfig` field enters `as_flag_table()` when set ⇒ `pilot.preregistered_continuation`
   becomes **a second key differing** from the run-2 journal ⇒ **the exemption fails its own
   identity check**;
2. run 3's journal would then differ from run 2's by **two** tokens, violating
   `PREREG-LeakAblation` §4: *"Run 3 changes exactly one token — any other diff between the two
   submissions invalidates the attribution."*

The repo already has the right pattern and this file states it: `--d5`/`--slices` are **CLI
arguments and not config fields, precisely so they cannot move the resume fingerprint.** The
exemption now follows it: `run_pilot(..., prereg_continuation=None)` + `--prereg-continuation`.
⛔ **Nothing about the mechanism is weakened** — it is still one declaration, per run, verified
identically, and it is **recorded in the artifact** (in `byte_ledger.preregistered_continuation`,
which is where an auditor reads it). **Verified mechanically** (test:
`test_the_exemption_is_a_RUN_ARGUMENT_and_NOT_a_PilotConfig_field`) and demonstrated: two **real**
trained legs differ by exactly

```
['memory.erosion_partition', 'pilot.memory']   memory diff: {'erosion_partition': (False, True)}
```

⚠ The alternative — adding the key to `_RESUME_FLAG_EXEMPT` — would have meant **editing the
fingerprint internals**, which the task forbids without a STOP. This design needs no such edit.
`PilotConfig` carries a comment at the old site saying all of this, so the next reader does not
"fix" it back.

---

## 3. What it refuses — the actual deliverable (task §2)

`tests/test_c3_run3_budget_exemption.py`, **38 cases, all green**, plus the same refusals executed
against the **real banked run-2 journal** (§5.1).

### 3.1 Break-attempts (every one from the task's list, plus five more)

| attempt | outcome | evidence |
|---|---|---|
| exemption + **exactly** the registered flag | **ACCEPTED**, ledger printed + stamped | §5, §6 |
| exemption + **a second key** (`refresh_monotonic`) | **REFUSED**, both keys named | test + real journal |
| exemption + **a different single key** | **REFUSED** ("does NOT differ", lists what does) | test + real journal |
| exemption + **geometry "shrunk to fit"** (`atoms_per_item` halved) | **REFUSED**, `store.atoms_per_item: journal=256 now=64` | test + real journal |
| **missing** journal | REFUSED | test |
| **corrupt** journal (not JSON) | REFUSED | test |
| journal whose **own fingerprint is invalid** (3 shapes) | REFUSED | test |
| **sha256 pin** mismatched | REFUSED | test |
| a **final `S3.json`** passed as a journal | REFUSED | test |
| flag = **list / glob / wildcard / unqualified / derived (`store_dim`) / `pilot.memory` / `pilot.store` / non-field** | REFUSED (8 param cases) | test |
| **extra spec key** (`also_allow=...`) | REFUSED ("unknown key") | test |
| malformed spec (string / empty / no `prereg`) | REFUSED | test |
| **forged dict** posing as an exemption | REFUSED (`ContinuationExemptionError`, "VERIFIED") | test |
| **no** exemption + over budget | **still REFUSED** (original behaviour intact) | test + §6 leg A |

### 3.2 It annotates, never suppresses

With the exemption in force and arms over budget, the artifact still carries **true** bytes:

```
enforced: true      budget_exempted: true      over_budget: ['clu_store','ttt_matched']
clu_store    2,509,696 B  occupancy 1.19672  within_budget=false  phi_accounted=true
```

⛔ Not zeroed, not omitted, not relabelled "within budget", and `enforced` stays `true` because the
check **ran**. Both stdout and the artifact carry the stamp: journal path, **sha256**, registered
flag `old → new`, the keys it moved, what the exemption does **not** exempt.

### 3.3 What the exemption cannot reach

Ordering is load-bearing and tested: per-arm ledgering (⇒ `UnledgeredArmError`) and φ accounting run
**before** the exemption is looked at; the ladder guard runs **before** the exemption is even built
(asserted on `run_pilot`'s source order) and takes no exemption argument; `assert_shared_shell_identical`
is untouched and downstream.

---

## 4. The two rulings, recorded in code (task §3)

### 4.1 ⭐ NO dtype normalisation
`DTYPE_NORMALISATION_RULING` ("**NONE** … total state bytes **AS DEPLOYED** … ⛔ this is **NOT a
bug** … do not 'fix' this by normalising widths"), quoted into the **module docstring**, into
`BUDGET_PROVENANCE`, and into **every artifact** (`byte_ledger.dtype_normalisation`) and every
printed header (`dtype normalisation: NONE (as deployed)`). Tested.

### 4.2 ⚠ The ceiling digit is INTERIM — and it blocks the ladder
`MATCHED_STATE_BYTE_BUDGET` → **`INTERIM_MATCHED_STATE_BYTE_BUDGET`** (old name kept as an
**alias**, not a second literal, so one edit still moves the repo). It is the name that now appears
**at the point of use** (`PilotConfig.state_byte_budget`). `BUDGET_IS_INTERIM=True`,
`BUDGET_CEILING_PREREG=None`, both stamped into every artifact.
`assert_ladder_arms_admissible(arms)` refuses any arm in `RIVAL_SPECS` while interim:

```
⛔ rival-ladder arm(s) may not train while the matched-state-byte ceiling is INTERIM:
    ladder arms requested: ['mamba2', 'ttt_linear']
    MISSING PREREG: the rival-ladder pre-registration, which sets the ceiling digit …
```

⭐ **Inert for run 3** (`ladder_arms_requested: []` in both demo artifacts) — the guard blocks what
the ruling blocks and nothing else.

---

## 5. Verification on the REAL run-2 journals (no training)

`.claude/outputs/c3-run3-budget-exemption/pilot-geometry-exemption.json`

### 5.1 All three real banked run-2 legs verify

`csf3_outs/run2/pilot_pilot_seed{0,1,2}_PARTIAL.json`, run-3 config reconstructed from each
journal's own flag table + `erosion_partition=True`:

| seed | sha256 (12) | old → new | keys moved | predated-at-default (forgiven) |
|---|---|---|---|---|
| 0 | `43e6598ddcdf` | `<absent> → True` | `memory.erosion_partition`, `pilot.memory` | the 3 other C2W6 fields |
| 1 | (in artifact) | `<absent> → True` | same | same |
| 2 | `dd803b279a46` | `<absent> → True` | same | same |

⭐ **`<absent>` is the real case**: the banked journals predate the C2W6 fields entirely, and it is
`load_journal`'s repaired §A20.4 rule that makes the leg resumable at all.

### 5.2 The ledger at the **actual** run-3 geometry (PILOT, unchanged)

```
[byte-ledger] budget 2,097,152 B (⚠ INTERIM …) | enforced=True | dtype normalisation: NONE (as deployed)
   clu_store     5,523,456 B  occupancy 2.63380  within=False  phi=True
   ttt_matched   5,483,712 B  occupancy 2.61485  within=False  phi=True
   gru_matched      10,992 B  occupancy 0.00524  within=True   phi=True
   none / echo           0 B  occupancy 0.00000  within=True   phi=True
   ⭐ PRE-REGISTERED CONTINUATION — the BUDGET check (and only it) is exempt for this leg
      over budget anyway: ['clu_store','ttt_matched']  ⛔ TRUE bytes, unmodified.
```

⇒ **run 3 is unblocked at its own geometry, with the 2.63× on the record rather than erased.**
(Identical to `c3-csf3-harness` §5.1 — this task changed no arithmetic.)

---

## 6. End-to-end: it actually trains (toy scale, ⛔ never a claim venue)

**(A) Two REAL legs, exemption verified against a REAL journal produced by a REAL run**
(`demo.sh`; `--arms clu_store none --set steps=4 …`):

| leg | what ran | wall | result |
|---|---|---|---|
| "run 2" | full S1+S2+S3, shield OFF | **350 s** | journal + `S3.json` |
| "run 3" | same **+ `--mem erosion_partition=true`** + `--prereg-continuation` | **219 s** | trained; ledger **stamped** (`budget_exempted: true`) |

Their two flag blocks differ by **exactly** `['memory.erosion_partition','pilot.memory']`.
⚠ At toy geometry the budget does not bite (occupancy 0.0196), so this leg proves *verification +
stamp + train*, not the over-budget branch. That is (B).

**(B) The over-budget branch, through a training run** (`demo3.sh`, `payload_dim=300` ⇒ 2.51 MB vs
the shipped 2 MiB — **geometry**, not a shrunken budget flag; identical command lines but for the
exemption):

| leg | outcome |
|---|---|
| **A — no exemption** | **REFUSED, exit 1**, before training: `clu_store: 2,509,696 B = 1.20x …` |
| **B — with the exemption** | **exit 0**: S1+S2+S3, both arms trained, 2 `.eqx`, `S3.json` with `over_budget: ['clu_store','ttt_matched']`, true bytes, full stamp |

⚠ **Disclosed:** (B)'s run-2 journal is **synthetic** — `flag_block()` output (byte-identical in
shape to a real leg's, C2W6 keys popped as the real ones have them). It has to be: **an over-budget
leg cannot be produced by current code** — the guard refuses it — which is precisely why the only
such artifacts that exist are the *pre-guard* real run-2 journals. §5 uses those real ones.
⚠ Honest failures on the way: the first (B) attempt was **SIGKILLed at 3.27 GB RSS** in
`allocation_liveness` (laptop memory, batch 4 × seq 512 at `payload_dim=300`) — rerun at
`batch=1 seq_len=128 liveness_lanes=1`; and an earlier attempt at `state_byte_budget=10000` died in
`rival_reference_table` (reconciliation item 3).

**Run 3's launch line (for the Hub — I did not submit anything):**

```
PREREG_CONT="journal=<RUN2_OUT>/pilot_pilot_seed${SEED}_PARTIAL.json \
             flag=memory.erosion_partition \
             prereg=.claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md" \
MEM="erosion_partition=true …(run 2's MEM verbatim)"  sbatch scripts/csf3/job_gpu_c3_seeds.sh
```

⛔ It must be present on **every** re-resume too, or the budget refuses the leg.
⭐ Optional and recommended: add `sha256=<digest>` to pin the journal's bytes (seed 0 =
`43e6598ddcdf9ecdd8b2fd9aec089905c9c2bd85eb44bc1d7db1c3d088dcc2b7`).

---

## 7. "What this still refuses" — the table for the Advisor

| someone wants to… | the mechanism… |
|---|---|
| skip the budget on a fresh run | **refuses** — no journal ⇒ no exemption |
| skip it by pointing at any old journal | **refuses** unless the config is identical to it but one knob |
| **shrink the store** to fit and keep the prereg | **refuses** — geometry is part of the identity |
| register **two** flags | **refuses** — one `group.key`, no list/glob/dict; a second is a new prereg **and a code change** |
| register a **derived** quantity (`store_dim`) or a whole override dict | **refuses** |
| point at a journal that has been edited | **refuses** if `sha256=` is pinned; the digest is stamped either way |
| point at a corrupt / fingerprint-invalid / non-journal file | **refuses** |
| use it to hide an **unledgered** arm | **refuses** — that guard runs first and is unreachable |
| use it to hide **φ** or the shared-shell identity | **cannot** — computed before, unmodified, still asserted |
| use it to train a **rival-ladder** arm on the interim ceiling | **refuses** — separate guard, no exemption argument |
| use it to make the over-budget number **go away** | **cannot** — `over_budget`, true bytes and occupancy are in the artifact and on stdout, with the exemption stamped beside them |
| **forge** an exemption in a config file | **cannot** — `build_byte_ledger` takes the frozen type, not a dict |

---

## 8. How I verified

| check | result |
|---|---|
| New tests `tests/test_c3_run3_budget_exemption.py` | ✅ **38 passed** |
| `tests/test_c3_csf3_harness.py` + `test_pilot_checkpoint_resume.py` + `test_ttt_stability_and_d5_wiring.py` | ✅ **70 passed** (662.9 s) |
| **Full suite** | see §8.1 |
| `ruff check chlu/ tests/` | ✅ All checks passed |
| `bash -n scripts/csf3/job_gpu_c3_seeds.sh` | ✅ clean; word-splitting of `PREREG_CONT` asserted |
| Real-journal verification (3 seeds) + 3 real-journal refusals | ✅ §5 |
| Two real trained legs, one-token diff | ✅ §6(A) |
| Over-budget refuse/accept pair through `run_pilot` | ✅ §6(B) |

### 8.1 Full suite

```
$ PYTHONPATH=/Users/user/Desktop/CHLU-wt1 .venv/bin/python -m pytest -q -p no:cacheprovider --no-cov
1819 passed, 29 warnings in 2410.46s (0:40:10)
HEAD_BEFORE=a656746  MAIN_BEFORE=0644c48  DIRTY=[]
HEAD_AFTER =a656746  MAIN_AFTER =0644c48  DIRTY=[]
```

✅ **1819 passed / 0 failed**, against **HEAD `a656746`**, **re-verified**: HEAD, `main` and a clean
working tree checked on **both** sides of the run (a green against a moving base is not a green).

**Arithmetic checked, not assumed:** `c3-csf3-harness` measured **1781** at `f98f939`, my base.
**1781 + 38 = 1819** — exactly my new test file and nothing else. ⛔ No existing test changed.

⚠ **`main` moved while I worked** — the Hub merged `c3-csf3-harness` at `0644c48` (a merge commit
whose **tree is byte-identical to `f98f939`**: `git diff f98f939 0644c48` is empty). So my base is
now an ancestor of `main`, `main..<branch>` is exactly my 4 commits, and merging is conflict-free
by construction. ⛔ I did **not** rebase: my named base has not moved, and rewriting hashes would
have detached the green above from the commit it was measured at.

---

## 9. Flag-provenance table

**Commit `a656746`** (branch tip), **JAX 0.9.0** (main venv reused via `PYTHONPATH=<worktree>
/Users/user/Desktop/CHLU/.venv/bin/python`; ⛔ **no `uv sync` in the worktree**), CPU/float32, macOS.

| | (A) real-journal verification | (B) demo legs (§6A) | (C) over-budget pair (§6B) | (D) tests |
|---|---|---|---|---|
| seed | 0, 1, 2 | 0 | 0 | 0 |
| scale | **pilot** (config arithmetic only, no training) | toy | toy | toy |
| journal | **REAL** `csf3_outs/run2/pilot_pilot_seed{0,1,2}_PARTIAL.json` | real, from leg 1 | ⚠ **synthetic** (`flag_block`) | synthetic (fixture) |
| non-default `PilotConfig` | reconstructed from each journal's own flag table | `steps=4, warmup=1, eval_batches=1, dyneval_batches=1, data_bytes=300000, monitor_every=2, arms=(clu_store,none)` | same **+ `payload_dim=300, batch=1, seq_len=128, liveness_lanes=1`** | toy defaults |
| non-default `memory` | run 2's verbatim **+ `erosion_partition=True`** | TOY **+ `erosion_partition=True`** | TOY **+ `erosion_partition=True`** | TOY ± the case |
| budget | `2,097,152 B` (**INTERIM**), `enforce=True` | same | same | same |
| exemption | `flag=memory.erosion_partition`, `prereg=PREREG-LeakAblation.md` | same | same (leg B only) | same |
| `ttt_normalized_write` | **False** (shipped default) | **False** | **False** | **False** |
| corpus | n/a | `enwik8`, real | `enwik8`, real | n/a |

⛔ **No number here is a claim.** The toy bpc values are execution evidence only and appear nowhere
in this report's tables.

---

## 10. Git footprint

**Branch `agent/experiment-engineer/c3-run3-budget-exemption`**, off
**`agent/experiment-engineer/c3-csf3-harness` @ `f98f939`** (that branch was **not** merged to
`main` when I started — verified: `main..c3-csf3-harness` = 8 commits — so I branched off it as the
task's primary instruction says). Worktree `../CHLU-wt1`. Not pushed, no PR.
⚠ Per §3.5 I did **not** rebase onto `origin/main` (stale). Rebase onto my named base: **no-op**
(base unmoved at `f98f939`).

| commit | files | note |
|---|---|---|
| `45dbdc0` | `chlu/eval/byte_ledger.py` | the two rulings, the ladder guard, `ContinuationExemption`, annotate-never-suppress |
| `e8c921f` | `chlu/experiments/exp_cluformer_pilot.py`, `chlu/training/train_cluformer.py` | the verifier, `flag_block`, wiring, CLI |
| `16ab741` | **new** `tests/test_c3_run3_budget_exemption.py` | 37 anti-loophole cases |
| `a656746` | `scripts/csf3/job_gpu_c3_seeds.sh`, `tests/…` (+1 test) | the `PREREG_CONT` passthrough |

**Shared-file hunks, exactly:**

- `chlu/training/train_cluformer.py` (**+17/−7**, two hunks): the import and the default of
  `state_byte_budget` renamed to `INTERIM_MATCHED_STATE_BYTE_BUDGET` (**same integer**, alias kept),
  and a comment block where a `preregistered_continuation` field would have gone saying why it is
  not one. ⛔ **No behaviour change**; `PilotConfig().state_byte_budget` is still `2_097_152`.
- `chlu/experiments/exp_cluformer_pilot.py`: **additive** except two hunks — `rec["flags"]`'s inline
  dict replaced by `flag_block(pcfg)` (**identical content and key order**), and `run_pilot` gained
  one keyword argument (default `None` ⇒ previous behaviour).
- `scripts/csf3/job_gpu_c3_seeds.sh`: **+3 lines** (`PREREG_CONT` default, the passthrough, the echo)
  — ⚠ **outside the task's named ownership list**; justified by §7.33 (a pre-registered leg behind a
  flag no launch path sets is indistinguishable from a cut) and it is the same branch's own script.
  ⛔ `job_gpu_cluformer.sh` untouched.
- ⛔ **Not touched:** `chlu/core/blocks.py`, `load_journal`/`_flag_dict`/`_flag_defaults`/
  `_RESUME_FLAG_EXEMPT` (**called, never modified**), `PilotConfig.addr_dim`'s default (**stays 8**),
  the tripwire/Track-B surface, `chlu/config.py`, the CLI command module.

**Worktree-ref verification (protocol §3.2, the lost-8-commits precedent) — from the MAIN repo,
BEFORE removal:**

```
$ git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/c3-run3-budget-exemption
a656746  16ab741  e8c921f  45dbdc0        (4 commits)
$ git -C /Users/user/Desktop/CHLU diff --stat main..agent/experiment-engineer/c3-run3-budget-exemption
 5 files changed, 1158 insertions(+), 28 deletions(-)
$ git diff --stat f98f939 main | wc -l          # my base's tree == main's tree
0
```

✅ All 4 commits visible on the shared ref from the main repo; `../CHLU-wt1` removed afterwards and
the ref re-checked. ⚠ Note `../CHLU-wt2` (`agent/experiment-engineer/c3-rival-ladder-prereg`) is a
**concurrent** worktree — I never touched it, and it is the spoke that will retire
`BUDGET_IS_INTERIM` (reconciliation item 2).

Artifacts (all under `.claude/`): `.claude/outputs/c3-run3-budget-exemption/` —
`pilot-geometry-exemption.json`, `demo/` (both real legs' `S3.json`, the over-budget pair's artifact
and both logs, the three demo scripts).

---

## Proposed handover updates (for the Hub)

**§3 (CLI & config) — new surface**

- **`--prereg-continuation journal=… flag=… prereg=… [sha256=…]`** (and `run_pilot(...,
  prereg_continuation=)`, and `PREREG_CONT=` in `job_gpu_c3_seeds.sh`). ⛔ **A CLI argument, NOT a
  `PilotConfig` field** — like `--d5`/`--slices` — because a field would be a *second* differing key
  and would break its own identity check (report §2.1). Exempts the **state-byte budget check only**.
- **`chlu.eval.byte_ledger.MATCHED_STATE_BYTE_BUDGET` → `INTERIM_MATCHED_STATE_BYTE_BUDGET`**
  (old name is an alias; same 2,097,152). `BUDGET_IS_INTERIM=True`, `BUDGET_CEILING_PREREG=None`,
  `DTYPE_NORMALISATION_RULING` — all three now in every artifact.
- Byte-ledger artifacts gained: `budget_is_interim`, `budget_ceiling_prereg`, `dtype_normalisation`,
  `budget_exempted`, `preregistered_continuation`, `ladder_guard`. ⛔ **No new TOP-LEVEL artifact
  key** — the final artifact's shape pin in `test_pilot_checkpoint_resume.py` is untouched.

**§7 — entries**

- **7.34 [RESOLVED for run 3, NOT for the ladder]** The 2.63× bust no longer blocks run 3: it is a
  **pre-registered continuation** of run 2 and rides a narrow, verified, stamped exemption
  (`c3-run3-budget-exemption`). ⛔ The refusal itself **stands unchanged for everything else**, and
  the 2.63× is **printed and archived**, not erased. The **counting conventions** (per-layer vs
  total; fp32 vs bf16) are now **ruled**: total, and **no dtype normalisation, as deployed**.
- **7.38 [NEW, standing] The ceiling digit is INTERIM and the ladder is BLOCKED until its prereg
  lands.** `assert_ladder_arms_admissible` refuses any `RIVAL_SPECS` arm while
  `BUDGET_IS_INTERIM=True`. Filing the rival-ladder prereg = **one edit** (digit +
  `BUDGET_CEILING_PREREG` + the flag). ⛔ The pilot geometry is **not** presumed to be C3 geometry.
- **7.39 [NEW, measured] One config knob moves TWO fingerprint keys.** `memory.*` overrides appear
  in the resume fingerprint **twice** (resolved `memory.k` **and** raw `pilot.memory`), because
  `memory` is itself a `PilotConfig` field. Any future "identical except key K" logic must revert
  the **knob**, not the key. Measured on the real run-2 journal.
- **7.40 [NEW, ops] Run 3 must carry `PREREG_CONT` on every submission AND every re-resume**,
  otherwise the state-byte budget refuses the leg. Passthrough is wired and tested; the submission
  line is the Head's.
- **7.41 [NEW, pre-existing quirk] `rival_reference_table()` raises below a ≈197 kB budget**
  (mamba2's shrink solver cannot reach it). Irrelevant at real budgets; it blocks only artificial
  small-budget tests.
