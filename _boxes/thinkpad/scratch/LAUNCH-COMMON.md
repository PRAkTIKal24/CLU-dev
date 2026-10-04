# LAUNCH COMMON — all FOUR autonomous hubs on FORGIS_14 (Windows, native) · iteration 2
**Read this, then your own `LAUNCH-hub-<slug>.md`, then boot per your task file.**

You are an **autonomous hub**, not a spoke. Your authority is the bundle:
`.claude/autonomous-hub-charter.md` → `.claude/AGENT_PROTOCOL.md` → `.claude/handover_context.md` §7 →
`.claude/tasks/hub-<slug>.md`. ⛔ Nothing else from the campaign archive.
**This file adds only what the bundle cannot know: facts about this machine, and three corrections.**

---

# §-1. THE OPERATING STRUCTURE — who you are, who you report to, what model you run on
| tier | model | role |
|---|---|---|
| **Advisor** | Fable | ideation only, **once per full pass**; writes `DIRECTIVE.md`; never runs anything |
| **Coordinator** ("the hub") | Opus | stages, monitors, compiles your reports into a decision brief, makes small decisions |
| **You** — execution hub | **Opus** | the science, for 12 h; reports; return packs |

⚠ The bundle's charter calls you an "autonomous hub". In this structure you are the **execution tier**
and you report to the **coordinator**, never to the advisor. ⛔ **Never contact or wait on the advisor.**
If you hit something that needs re-ideation (a §5 falsifier, a structural NOT-SOLVED, a bar you want to
change), **write it in your LOG and your report and carry on with what remains legitimate** — the
coordinator escalates, not you.

⭐ **You run on Opus. Every sub-agent you spawn runs on Opus: pass `model: "opus"` on every `Agent(…)`
call, without exception.** Fable is reserved for the advisor pass and is not yours to invoke.

**Small decisions are yours.** Environment fixes, your own code, tooling, batching, which lever to try
next — decide and log. **These are NOT yours:** any bar, bound, gate, criterion, pinned file, or the
object definition shared with peers. Those you log as a question and leave alone.

---

# §-0. ⚠ THIS IS A RELAUNCH — an earlier start was stopped. Read before you touch anything.
All four hubs were launched once already today on `base/iter2` and **stopped by the Head** a few minutes
in, to restructure. **Nothing they wrote has been deleted (§0.1) and nothing will be.** What exists:

- ⭐⭐ **`hub/generation-table-2` completed the §5.1 falsifier before it was stopped, and the verdict is
  CLEAR.** `.claude/outputs/hub-generation-table/reencode_magnitude.json` and
  `REENCODE-VERDICT-iter2.md`: in the `|cached| ≥ 2^-4` band **max ULP = exactly 1.000, zero violators**;
  the 51-ULP figure is entirely sub-`2^-4` post-ReLU dust. **Scored work may proceed on this box (seed 0,
  no extension).** ⛔ Do NOT re-run the falsifier; it is an inherited fact. ⚠ Read its two caveats:
  an *enforced* gate still aborts here (51 > 1), and the fraction bound has **31 elements of headroom** —
  it is not a second reassurance.
- `hub/deletion-2` ran its smoke-scale determinism check: builds `smoke_A` and `smoke_B` from the same live
  set hash **identical** (`labels_sha256 a1c53517…`). Its full-scale `det_A` build was killed mid-run when
  the thread stopped — `runs/` may hold a **partial** artefact; treat it as untrusted, rebuild.
- Each stopped hub left `iter2/dev/*` scratch, a `LOG` and (except refusal) a `START_UTC.txt` under its
  outputs dir, and **untracked new files in its worktree** (listed in your addendum). ✅ You may read all
  of it as inherited fact. ⛔ Do not delete it. ⛔ Do not *resume* it as if it were yours — **start
  fresh: write a NEW `START_UTC.txt` (overwrite is allowed for this one file only, since the old stamp is
  not a scientific artefact) and open a new LOG section headed `RELAUNCH`.** Adopt an inherited file only
  by an explicit dated bullet saying so.
- ⚠ One hub's log recorded that its *perceived* elapsed time ran **~6× fast** in the first minutes. That
  is why §1b says `date -u` and `START_UTC.txt`, never recollection.



---

# §0. THE OVERRIDE — the verifier is red, and you MAY launch anyway
`WORKSPACE-READY.json` reads `"status": "NOT-READY"` and `verify_hub_workspace.sh` prints
*"⛔ Do not launch a hub."* **A human has overridden this in writing. Quote it in your log:**

> **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and
> `hub/generation-table` on v2 despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.**
> The failure is in the test's `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()` calls
> `read_bytes().decode()`, which does **not** fold universal newlines, while real `Path.read_text()` opens
> in text mode and does. On Windows the fixture is written by a real `write_text()`, which expands
> `\n → \r\n`, so the mock counts two characters per newline and `chars` exceeds `len(text)` by exactly the
> newline count. **Nothing to do with encoding.** It passes on macOS only because macOS does not expand
> newlines. The mock is fixed on the Mac and ships in the next cut; the UTF-8 behaviour it guards is
> unaffected and still enforced by the shipped code's explicit `encoding="utf-8"`.

⭐ **Everything that inspects the bundle is green**: 146/146 checksums, 146/146 artefacts opened through
their real loaders, 0 missing, 0 unopenable, import-under-root, root tree == pin tree, write path
writable, `anchors_checked` non-zero on all four anchored docs.
⛔ **This override covers exactly that one test. It is not a licence to wave through any other red check.**
A failing check remains information. ⛔ `SKIP_LOAD=1` / `SKIP_TEST=1` are **not** a pass.

---

# §1. ⛔⛔ THREE CORRECTIONS TO YOUR TASK FILE — it was written against v1

### 1a. ⛔ Your base is `base/iter2`, NOT what your task file says
Your task file says `Base: the PIN ae907466…`. That commit does not exist here, and neither does
`4125fce7`. **Your worktree and branch have already been created for you** from **`base/iter2`
(`945a3e9`)**, which is the pin tree `345244bb…` plus exactly two Head-authorised additions:
the five reference-read files from the predecessor hub (940 insertions, 0 modified), and the §2.2(b)
gate patch (12 insertions, 2 deletions, in `exp_gen_cell.py` only).
⛔ `ULP_BOUND`, `FRAC_BOUND` and `_assert_cache_reencodes` itself are **untouched** — `exp_gen_store.py`
has a zero-line diff from the pin. Verify that yourself if you like.

```
worktree : /c/Users/PratikJawahar/chlu/CHLU-hub-<slug>-2      (already created)
branch   : hub/<slug>-2                                        (already created, at 945a3e9)
```
⚠ The path carries a `-2` suffix because the predecessor hubs' worktrees still occupy the un-suffixed
paths and ⛔ §0.1 forbids deleting them. ⛔ Do not branch from `8e36bc9`, from a peer's branch, or from
an old `hub/*` branch.

### 1b. Your wall-clock limit is **12 h**, not the 10 h your task file states
**Head ruling, 2026-09-16 — this supersedes the task file's `≤ 10 h`:**
**Work until the task is solved TRUTHFULLY, or 12 hours of TRUE WALL-CLOCK time, whichever comes first.**
```bash
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$CHLU_HUB_OUTPUTS/hub-<slug>/START_UTC.txt"
```
⭐ Measure elapsed against **that file and the real clock**. ⛔ Not model-perceived time, not summed tool
time, not time-minus-waiting — your sense of elapsed time is unreliable over a long run.

⛔⛔ **"Truthfully" is the operative word.** A task is NOT solved because the clock is nearly up, because a
partial result looks presentable, or because a threshold was moved until something passed.
⭐ **Stopping at 12 h with "not solved — here is exactly how far I got, and what blocked me" is a SUCCESS,
and is the expected outcome if the science does not land.** Manufacturing a result to fill the window is
the one unrecoverable failure. A deadline is precisely the pressure that produces it — name it to
yourself when you feel it.

### 1c. ⭐ YOUR SCORED-RUN BUDGET — now stated
**Three per target, across every attempt and every machine. ZERO spent so far on any target, on any
machine** — all three overnight hubs spent 0/3, and the Mac has spent 0 on these four targets.

⛔ **This iteration: AT MOST ONE scored run, SEED 0 ONLY**, and only after (i) your PREREG names the exact
object, bar and controls, and (ii) your DEV number clears your bar on non-held-out corners with the §C
controls passing. Report a seed-0 number as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.
After this iteration: 2 remain per target, of which the 3-seed final costs 1.

⚠ **Seeds 1 and 2 cannot be scored on this box at all this iteration** — their caches are
`scope="used"` (9,288 rows) and covering the cell would require encoding ~32,200 new rows. ⛔ That is a
re-encode and it is forbidden. The Mac is building full-scope caches in parallel; not your path.

---

# §2. THIS MACHINE
**Workspace (`$CHLU_ROOT`): `/c/Users/PratikJawahar/chlu/chlu-bundle-v2`** — pin `4125fce7…`
⛔ An older **v1** bundle sits at `…/chlu/chlu-hub-bundle` (pin `ae907466`). **It is not yours. Do not
read from it, do not write to it.**

**Open your shell exactly like this — every line matters:**
```bash
cd /c/Users/PratikJawahar/chlu/chlu-bundle-v2
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null   # a stale CHLU_* silently overrides the root
. ./env_hub.sh
env | grep ^CHLU_          # every path MUST contain chlu-bundle-v2
export PYTHONUTF8=1
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
```
- ⚠ **`PYTHONUTF8=1` is load-bearing.** Without it your FIRST encode dies at `gen_bridge.py:915` with
  `UnicodeEncodeError: 'charmap' codec can't encode character '\u2b50'` — stdout is cp1252 here. That is
  on the **encode path**, not a diagnostic corner.
- ⚠ **Bare `python` / `python3` are the Microsoft Store stub.** Use `$CHLU_ROOT/.venv/Scripts/python.exe`
  (Python **3.11.16**, jax/jaxlib 0.9.0, numpy 2.4.1, CPU). `uv` exists only at
  `C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\Scripts\uv.exe`.
- **16 cores / 16 logical · 63 GB RAM · 833 GB free · Intel Arc Pro GPU ⇒ NO CUDA, ever.**
- Sleep and hibernate are **disabled on AC** for this campaign. The machine will not suspend under you.
- ⭐ **AMENDMENT 1 (binding): CPU jobs run LOCALLY here. GPU jobs run on GCloud.** Supersedes the charter's
  "GCloud CPU is your default, always", which was written for a 16 GB shared laptop.
  **GPUs sparingly — ⛔ never for a run whose best case is a supporting claim.**
- ⛔ **CSF3 is unreachable by any hub (2FA).** Emit the submit lines, report **NOT RUN**.
- **gcloud** works (account `pratik.jawahar@forgis.com`, project `forgisprova`). Job submission is
  authorised. ⛔ Nothing that spends money without saying so first.

---

# §3. ⚠ YOU ARE ONE OF **FOUR** HUBS SHARING 16 CORES — 4 threads each, no exceptions
All four of you run concurrently. **Set this on every unit you launch:**
```bash
export OMP_NUM_THREADS=4  MKL_NUM_THREADS=4  OPENBLAS_NUM_THREADS=4
```
⛔ 4 hubs × 4 threads = 16. ⛔ **No hub exceeds its 4.** **Sub-agent cap: 2 per hub, inside those 4
threads.** ⭐ **Every sub-agent runs on Opus: `model: "opus"` on every `Agent(…)` call.** An unbound numpy/sklearn oversubscribes its share and runs **slower**, not faster.
⭐ **Confirm no sub-agent of yours is still running before you finish.**

⭐ **Collision safety, by construction:** you write only `$CHLU_HUB_OUTPUTS/<your-slug>/**`, your
worktree is yours alone, and `--return-ingest` diverts rather than overwrites. Peers cannot corrupt you.
⛔ **Do not read a CONCURRENT peer's outputs, and do not branch from a peer's branch.**
✅ **Reading the three PREDECESSOR hubs' reports and outputs is allowed and expected** —
`.claude/outputs/hub-refusal/`, `hub-composition.md`, `hub-generation-table.md` and their `LOG.md`
files are **inherited facts** now, not peer work.

---

# §4. PARITY — settled, with its bound stated
Measured here against the Mac reference via `parity.py`
(`18aa8f649137891219b01eb9c33c23484cb49eebec9a9714adb843e43bc3bb16`):

| | Mac (Darwin arm64) | This box (Windows AMD64) | delta |
|---|---|---|---|
| `sum(float64)` | 51849.2300147639 | 51849.2302889892 | **0.07 float32 ULP** |
| `max` | 4.8279623985 | 4.8279609680 | **3.00 float32 ULP** |
| `SHA256` | `8821b833…` | `9e2e3ee9…` | differs |

⭐ **Not bit-identical, and that is expected and accepted.** The charter's declared property #3 already
states CPU/GPU agreement is *"~53 ULP on ~1 % of elements, not bit-exact"*. Three ULP is well inside a bar
the programme has already ruled acceptable. ⛔ **Do not chase bit-identity** — it is unachievable across
ISAs and it is not the question. The question is whether a **reported number** moves.

⛔⛔ **THE HARD CONSTRAINT: never mix Mac-produced and Windows-produced artefacts INSIDE ONE COMPARISON.**
The rival rows are frozen and Mac-produced. Any rebuild of an input that feeds a scored comparison breaks
comparability with them, even if your own arithmetic is perfectly self-consistent.
⭐ **THE GATE RULING (supersedes the old "STOP and report" blanket).** The re-encode gate stays at
**1 ULP and is never loosened** — it simply **never binds on a scored run, because no scored run here may
extend the cache**. The gate exists to catch *writing new rows through an encoder that does not reproduce
the cache*; a run that encodes nothing new cannot trigger that hazard. `base/iter2` gives `run_seed` the
coverage-scoping the pin already ships at `exp_gen_wells_route_b:382`. Seed 0's cache is `scope="full"`
(480,000 rows = the whole split) and provably cannot be extended, **so seed 0 scores here.**

⛔ **A shortfall > 0 still means STOP and report the shortfall. It never means encode.** The moment any run
needs a row that is not in a Mac-written cache, the gate binds at full strength, it will fail, and that is
correct. ⛔ If you find yourself wanting to raise `ULP_BOUND`, widen `FRAC_BOUND`, or pass
`will_extend=False` with a shortfall > 0 — that is the exact loosening this ruling forbids, and it is
your signal to stop.

---

# §5. STANDING RULES
- ⛔ **YOU NEVER DELETE** (§0.1). Leave everything you write in place.
- ⭐ Write **only** to `$CHLU_HUB_OUTPUTS/hub-<slug>/**` (§0.3).
- ⭐ **Running log `…/hub-<slug>/LOG.md`, updated AS YOU GO — ⛔ never written at the end.** If you die at
  hour 9, the log is what survives.
- ⭐ **Structure work as resumable units with a ledger**, like `scripts/csf3/job_cpu_gen.sh`: *"a kill costs
  ONE UNIT, never the run."* ⚠ **Compute is LOCAL here** — unlike GCP, if your process dies the work dies
  with it. Resumability is your only protection.
- ⛔ Never edit a manifest-pinned file. ⛔ Never weaken a check to make it pass.
- **PRE-REGISTRATION before any scored run.** Every scored run goes in the results ledger.
- ⛔ `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` is the **SUPERSEDED broken-split** file —
  ⛔ **never quote it.** ⚠ It shares a basename with the good `gen-val-split-fix/refusal_scores.json`, and
  that is exactly how a wrong number reaches a paper. It is manifest-pinned: never move or delete it.
- ⚠ `GenCellConfig.lace_selection_source` still points at the broken-split LACE selection. Declared in the
  charter, ⛔ not yours to fix.

# §6. FINISHING
PREREG files · a results ledger of every scored run · a one-page report · a cost line. Then
`--return-pack hub-<slug>` (proven working here) and `git bundle create` **whole-branch, ⛔ never a
range** — a thin bundle's prerequisite is this repo's synthetic base commit, which the Mac has never seen.
Take the ~3 MB.
⭐ State your elapsed true wall-clock time and whether the task was **solved truthfully** or **not solved**.
Both are acceptable. Only a manufactured answer is not.


---

# §7. HEAD RULINGS, 2026-09-17 — binding on all four hubs
- **R1 — Tie definition (charter §A / the ceiling rule).** *tie* := the paired 95 % CI on Δ(arm − rival)
  includes 0, **or** |Δ| ≤ m = 0.020. *win* := Δ > 0 with CI excluding 0. *loss* := Δ < −m with CI
  excluding 0. Set **before** any scored run against it (§A.3 satisfied). ⭐ **A tie is banked and
  reported; you then keep pushing for a win with your remaining time.**
- **R2 — Composition's bar is retired.** ≥ 60/72 distinct valid novel tuples is replaced by: *distinct
  valid novel tuples on B at matched bytes, win-or-tie vs `knn_proto` under R1*, plus the measured
  capacity law and the decoder ceiling reported beside it.
- **R3 — Cross-device provenance.** Every reported number carries the machine that produced it. Headline
  comparisons (arm vs rivals) are **same-device by construction**: the scored cell re-runs every arm,
  rivals included, on this box from Mac-encoded features. Mac-published rows are cited as **prior rows
  with provenance**, ⛔ never mixed *inside* a paired comparison. ⭐ *"Too strict costs time we do not
  have"* — do not chase bit-identity, do not block on it.
- **R4 — The PCA-affine dictionary is NOT adopted as a store.** Reported as a **control only**. Reason: a
  principal direction cannot be decoded or deleted, and the deletion target needs atoms.
- **R5 — The trajectory-encoder idea is DEFERRED.** Measured reason: the pairwise request model reaches
  0.9734 / 68 of 72 on the *existing* latent, so latent richness is not the binding constraint here.

# §7b. PROGRAMME-WIDE REPORTING RULE — adopted 2026-09-17, binding on every hub
⭐ **Every refusal or admission row — arm or rival, DEV or scored — prints `accept_A7` beside `accept_B`
at the same threshold.** No exceptions.
**Why:** three of five published rivals falsely refuse ~60 % of the valid A7 redundant-term control, and
the published table does not show it. A refusal number without its A7 companion overstates the rival and
understates the arm, and a reviewer will ask.
⛔ A row without `accept_A7` beside `accept_B` is not reportable.

# §8. THE GOAL, IN ONE LINE (Head)
ICLR-main-track results for the CLU in **generation, composition, refusal and deletion** — as fast as
possible, ~7 days for results including today, scientific and honest.
⛔⛔ **What we strictly do not want is a glorified lookup table.** The Hamiltonian settle dynamics are
*nice to have*; **the store, the memory modes, exact deletion and request-only addressing are the core.**
⭐ You MAY modify the CLU for your task as long as that core is intact (charter §B).
