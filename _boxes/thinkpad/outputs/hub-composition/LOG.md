# hub/composition — RUNNING LOG
**Agent:** autonomous-hub · **Branch:** `hub/composition` · **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-composition`
**START_UTC:** `2026-09-16T19:10:40Z` (see `START_UTC.txt`) · **Budget:** 12 h TRUE wall clock (Head ruling, supersedes task file's 10 h)
**Core budget:** `OMP/MKL/OPENBLAS_NUM_THREADS=5`

> Rule for this file: dated bullets appended AS I GO. Corrections are NEW bullets, never edits.

---

## 2026-09-16T19:10Z — boot

- Read, in order: `LAUNCH-COMMON.md`, `LAUNCH-hub-composition.md`, `.claude/autonomous-hub-charter.md`,
  `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` §7, `.claude/tasks/hub-composition.md`. Nothing else
  from the campaign archive.
- **Override quoted verbatim, as required by COMMON §0:**
  > **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and `hub/generation-table`
  > on v2 despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.** The failure is in the
  > test's `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()` calls `read_bytes().decode()`,
  > which does **not** fold universal newlines, while real `Path.read_text()` opens in text mode and does. On
  > Windows the fixture is written by a real `write_text()`, which expands `\n -> \r\n`, so the mock counts two
  > characters per newline and `chars` exceeds `len(text)` by exactly the newline count. **Nothing to do with
  > encoding.** It passes on macOS only because macOS does not expand newlines. The mock is fixed on the Mac and
  > ships in the next cut; the UTF-8 behaviour it guards is unaffected and still enforced by the shipped code's
  > explicit `encoding="utf-8"`.
  - I treat this as covering **exactly that one test**. Any other red check is information, not something to wave
    through. `SKIP_LOAD=1` / `SKIP_TEST=1` are not a pass.
- **Base commit correction applied (COMMON §1a).** Task file's `Base: ae907466…` does not exist in this repo.
  Verified: `git log --oneline --all` -> exactly one commit `8e36bc9 [bundle] the CHLU tree at pin 4125fce7…`.
  `git rev-parse HEAD^{tree}` -> `345244bbd40a29a7842357342e32e528890e0670` = the bundle's `pin_tree`.
  Branched from `HEAD`:
  `git worktree add /c/Users/PratikJawahar/chlu/CHLU-hub-composition -b hub/composition HEAD` -> OK.
- Peer worktree `CHLU-hub-refusal` (branch `hub/refusal`) is present and live. I do not read its outputs and do
  not branch from it.
- **Correction to the addendum, for the record:** the addendum says `.claude/outputs/hub-composition/` already
  holds `.workspace-write-probe.json`. It does **not** — the only write probe on disk is
  `.claude/outputs/hub-refusal/.workspace-write-probe.json`. My `mkdir -p` created/kept the dir; nothing was
  deleted. There is no prior run to resume, as the addendum says.
- **OPEN QUESTION TO THE OPERATOR, asked now and repeated at every checkpoint (COMMON §1c):**
  **How many of the 3 scored runs for target `composition` remain unspent across all machines?** This box cannot
  see the Mac's count. ⛔ I will not perform a scored run until a human answers. Reading, planning,
  PRE-REGISTRATION and dry runs on non-held-out data continue meanwhile and are unrestricted.

## 2026-09-16T19:12Z — environment: two real blockers found and cleared, nothing shared mutated

- `exp_gen_cell --help` runs in **2.6 s** on this box. ⭐ The protocol's "JAX cold start ~20+ min" is a
  **macOS fact and does not hold here**. Budget accordingly.
- First `--quick` probe died at `chlu/data/shapes3d.py:112`: `ImportError: … requires h5py`. Then
  `3dshapes.h5` is absent — as COMMON says, the raw dataset is deliberately never bundled.
- ⭐ **Preconditions PASS before either blocker**, which is the important part:
  - `classifier_gate.json` `gate=PASS` (floor 0.95, 3 seeds)
  - `gen-arms-a7-wiring/refusal_scores.json` `status=complete` (`not_run=[]`)
  - decoder: `winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b…` (hash + recipe asserted)
  - external classifier `c8741ac417d88ebf…`, 978,873 params
- **Fix, deliberately scoped so NOTHING SHARED IS MUTATED** (§0.3, and the peer hub is live in the same
  venv):
  - h5py installed to a **private** target, not the shared venv:
    `uv pip install --python <venv> --target $OUT/pylibs --no-deps h5py` -> `h5py==3.16.0`.
    `--no-deps` on purpose: numpy stays **2.4.1** (verified after install). `PYTHONPATH=$CHLU_ROOT:$OUT/pylibs`
    and `chlu.__file__` still resolves under `$CHLU_ROOT` (the import trap `env_hub.sh` warns about).
  - dataset downloaded into a **private** gen-data root `$OUT/data/gen/shapes3d`, with the four bundled
    files (`factor_index.npz`, `query_sets.json`, `c_invalid.json`, `a7_redundant.json`) **HARDLINKED**
    in (link count 2, zero extra bytes, byte-identical by construction). `CHLU_GEN_DATA_ROOT` points there.
  - ⛔ I did **not** write into `.claude/data/gen/shapes3d/`. Two other hubs are live; a concurrent partial
    write to a shared 268 MB file is exactly the kind of corruption that is invisible until it is fatal.
- **Download verified against the code's own pin:** 267,573,662 B,
  `sha256 = 0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785` — equal to `shapes3d.H5_SHA256`.
  Licence Apache-2.0 (google-deepmind/3d-shapes). ⭐ Free, public bucket; **no spend**.
- **Split verified end to end:** `build_split()` -> labels `(480000, 6)`, **test = 72**, train = 479,928.
  ⭐ 72 is the number T4 says it must be; it was not reconciled by adjusting anything.
- ⚠ **NOTE FOR THE OPERATOR / PEER HUBS:** the canonical `.claude/data/gen/shapes3d/3dshapes.h5` is still
  absent, and any hub that needs raw Shapes3D will hit the same two blockers. My verified copy is at
  `$OUT/data/gen/shapes3d/3dshapes.h5`. It can be hardlinked into place for free with
  `ln <mine> .claude/data/gen/shapes3d/3dshapes.h5` — ⛔ I have not done it, because that is a write
  outside my scope.

## 2026-09-16T19:14Z — ⛔⛔ BLOCKER FOUND ON THE FIRST `--quick` UNIT. The addendum's exact hazard, measured.

- The `--quick` timing probe did not reach a timing number. It **aborted at the re-encode gate**, at
  `exp_gen_store.py:497`, via `run_seed`'s call at `exp_gen_cell.py:1780`:
  ```
  AssertionError: re-encoding 64 cached rows did not reproduce the cache to within one float16 ULP:
  max |delta| = 0.00195312 = 51.000 ULP of the cache dtype (bound 1);
  1280/131072 elements (0.977%) differ.
  {'n_checked': 64, 'max_ulp_delta': 51.0, 'frac_differing': 0.009765625, 'cache_dtype': 'float16',
   'ulp_bound': 1.0, 'frac_bound': 0.01, 'enforced': True, 'enforced_reason': ''}
  ```
- ⭐ **This is NOT a new defect. It is the charter's own declared property #3, measured:**
  *"CPU/GPU feature agreement is ~53 ULP on ~1 % of elements, not bit-exact."* I measure
  **51.0 ULP on 0.977 % of elements** — Mac-written cache vs Windows encoder. COMMON §4 predicts this
  number and rules it acceptable. ⛔ I am not chasing bit-identity; it is unachievable across ISAs.
- ⚠ Note the shape of the failure: `frac_differing` 0.977 % is **under** its 1 % bound; only the ULP
  bound is exceeded. Both bounds are declared in the gate and neither can be widened by any parameter.

### The measurement that decides the hub — cache coverage per seed
`$OUT/probes/cache_coverage.py` (reads only, encodes nothing). How many rows the cell needs that its
seed's frozen cache does not hold:

| seed | cache scope | cache rows | rows the cell needs | ⭐ **MISSING** | store_train missing | fit missing |
|---|---|---|---|---|---|---|
| **0** | `full` | 480,000 | 33,038 | ⭐ **0** | 0 | 0 |
| **1** | `used` | 9,288 | 33,039 | ⛔ **32,395** | 32,199 | 31,698 |
| **2** | `used` | 9,288 | 33,039 | ⛔ **32,437** | 32,241 | 31,742 |

⭐ **Seed 0's cache is FULL: the cell run provably CANNOT extend it.** Nothing is re-encoded; every
feature is read from the frozen Mac-written array.
⛔ **Seeds 1 and 2 would have to re-encode ~98 % of their inputs** through a Windows encoder that
demonstrably does not reproduce the Mac cache (51 ULP). ⛔ That is the HARD CONSTRAINT in COMMON §4 and
the addendum's explicit STOP condition, word for word.

### ⛔ BLOCKER 1 — SEEDS 1 AND 2 CANNOT RUN ON THIS MACHINE. Reported, not worked around.
The addendum: *"If a line of attack requires EXTENDING the feature cache, ⛔ STOP and report it as a
blocker rather than proceeding. That is a decision for the humans, not for you."* ⭐ I am stopping on
seeds 1 and 2 and reporting. ⛔ I will not re-encode, and ⛔ I will not run them to "see what happens".
**Consequence, stated plainly: a 3-seed table cannot be produced on this box.** Seed spread — which the
charter requires for every claim — is not available to me from a fresh cell run.

### ⚠ BLOCKER 2 (needs a human ruling) — the gate's `will_extend` scoping is NOT WIRED at the cell
- The charter's §A.4 component list and declared property #3 both state the gate *"binds only when a run
  will EXTEND the cache"*, and `_assert_cache_reencodes`' own docstring builds `will_extend=False` for
  precisely this case: *"A run that encodes nothing new cannot trigger that hazard … Such a run was still
  being refused — on a capability it never exercises."* The pattern it prescribes is the caller's own
  coverage evidence (`cov["n_shortfall"] == 0`), as `exp_gen_wells_route_b` does.
- ⛔ **But `exp_gen_cell.run_seed:1780` passes no `will_extend`**, so the strict default applies. So
  **even seed 0 — where the shortfall is provably 0 — is refused.**
- ⛔ `exp_gen_cell.py` is part of the frozen scoring harness (§A.4: *"A hub that edits them voids its
  numbers"*). ⛔ **So I am NOT editing it on my own authority**, even though the one-line wiring would be
  faithful to the declared design and would leave seeds 1/2 aborting at FULL strength (shortfall > 0).
- ⛔ I will not monkey-patch, will not pre-seed `_REENCODE_VERIFIED`, and will not set `SKIP_*`. Those
  are circumvention, not a pass.
- ⭐ **QUESTION 2 TO THE OPERATOR:** may I wire `will_extend=(shortfall > 0)` with its mandatory `reason`
  at `exp_gen_cell.run_seed`, on my branch, so seed 0 can run? It changes no computed number (seed 0's
  features come from the cache either way) and it makes seeds 1/2 fail *more* explicitly, not less.

### ⭐ What I am doing MEANWHILE — the line of attack that needs neither answer
`gen-read-cell-arm/runs/` holds a **complete Mac-produced cell run** (3 seeds) plus 207 MB of persisted
artefacts, including — decisively —
`artefacts/shapes3d_recon/seed_*/decoded_s*_dynread*_{B,MS}.npz` (images, **the external classifier's
factor tuple per decoded image**, features, requests, labels) and
`artefacts/shapes3d_recon/seed_*/stored_atom_tuples_s*_{A,B}.npz` (**the novelty reference**).
⭐ These are frozen, Mac-produced, and already scored by THE instrument. Computing VALID-and-NOVEL
distinct-tuple counts from them is pure **rendering of frozen artefacts** — exactly what the addendum
asks for, with ⛔ no rebuild, ⛔ no re-encode, ⛔ no scored run, and no gate involved. That is next.

## 2026-09-16T19:20Z — ⭐ the frozen Mac cell table, read off disk. ZERO compute, ZERO gate, ZERO scored runs.

`gen-read-cell-arm/runs/cell/seed_{0,1,2}/seed_*/cell_manifest.json` is a **complete 3-seed cell run**,
Mac-produced, scored by THE hash-asserted external classifier. Saved to `$OUT/probes/frozen_table.json`.
⭐ Reading it is not a scored run: it spends nothing and I did not produce it.

| arm | external in-corner balacc | composition-B aggregate | ⭐ **exact_match / 72** | distinct factor-tuples / 72 |
|---|---|---|---|---|
| ⭐ **DECODER CEILING** (real features) | 0.9357 ± 0.0195 | 0.9892 ± 0.0060 | ⭐ **67.67 ± 2.49** (67/71/65) | — |
| `knn_proto` | 0.9165 ± 0.0250 | 0.8650 ± 0.0122 | **14.33 ± 4.78** | 56.00 ± 3.74 |
| `clu` | 0.6568 ± 0.0226 | 0.5949 ± 0.0082 | ⛔ **0.00** | 9.00 ± 0.82 |
| `lace` (ode) | 0.6278 ± 0.0289 | 0.3117 ± 0.0153 | ⛔ **0.00** | 71.67 ± 0.47 |
| `lace/ld_corrected` | 0.6448 ± 0.0115 | 0.2816 ± 0.0180 | ⛔ **0.00** | 72.00 ± 0.00 |
| `lace/ld_published` | 0.5708 ± 0.0479 | 0.2554 ± 0.0022 | ⛔ **0.00** | 55.33 ± 4.50 |
| `dynread` (A, full) | 0.5585 ± 0.0500 | 0.2083 ± 0.0514 | ⛔ **0.00** | 13.33 ± 1.25 |
| `dynread/B` (full) | 0.5174 ± 0.0154 | 0.2114 ± 0.1033 | ⛔ **0.00** | 16.00 ± 1.41 |
| `ebm_compose` | 0.5477 ± 0.0216 | 0.1937 ± 0.0182 | ⛔ **0.00** | 72.00 ± 0.00 |
| ⭐ `dynread/A_request_off` | 0.5000 ± 0.0000 | 0.0463 ± 0.0346 | 0.00 | **1.00 ± 0.00** |
| ⭐ `dynread/B_request_off` | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00 | **1.00 ± 0.00** |
| ⭐ `store_removed` (laundering control) | 0.5000 ± 0.0000 | 0.1204 ± 0.0131 | 0.00 | **1.00 ± 0.00** |

⭐ Cross-checks against the numbers my task file gave me, all reproduced from disk: `clu` 9.00 ± 0.82 ✓ ·
`knn_proto` 56.00 ± 3.74 ✓ · `lace` 71.67 ✓ · dynread 13.3 (A) / 16.0 (B) ✓ ·
`lace` composition-B `0.3117 ± 0.0153` ✓ (declared property #4 — ⛔ **not** the BASELINES instrument's
`0.3241 ± 0.0165`, which is a different instrument and not on this table).
⭐ `store_removed` collapses to **1.00 distinct tuple** and balacc exactly 0.5000 — §C.2 laundering control
**passes**; the table is not laundering.

### ⭐⭐ THE NUMBER MY TASK FILE DID NOT GIVE ME, AND IT CHANGES THE PROBLEM
**`exact_match` — the external classifier reading ALL SIX factors as the REQUESTED tuple — is ZERO of 72
for every arm except `knn_proto`.** `knn_proto` gets **14.33 ± 4.78**. The **decoder ceiling itself is
67.67 ± 2.49**.
- ⛔ `clu` renders 9 distinct tuples, and **not one of them is the tuple that was asked for.**
- ⛔ `lace` renders 71.67 distinct tuples — near-perfect *diversity* — and **not one is correct.**
  ⚠ This is §E.4's free-aggregate trap in its sharpest form: a 0.3117 aggregate with 71.67 distinct
  tuples and **0/72 correct**. Diversity and correctness are close to orthogonal on this table.

### ⛔ ON THE DEFINITION OF "VALID" — pinned from the code, declared BEFORE I measure, ⛔ not negotiable
My target says *"valid per the external classifier, and not the tuple of any stored atom"*, and falsifier 1
says *"held-out combinations rendered CORRECTLY"*. In this codebase **validity is a property of a
REQUEST**, not of a tuple: `gen_splits.C_INVALID_SCHEMA` defines invalid as `duplicate_factor` /
`out_of_range` — structurally unsatisfiable *requests*. A classifier's arg-max tuple is **always** a point
of the exhaustive Shapes3D grid, so "structurally valid tuple" is **vacuously true for every render** and
cannot be what the bar means. Therefore:
- ⭐ **VALID(i) := the external classifier reads render `i` as the tuple that was REQUESTED** — all six
  factors — i.e. `probe_accuracy`'s own per-item `exact_match`. This is falsifier 1's "rendered
  CORRECTLY, read by the external classifier", verbatim, and it is the **strict** reading.
- ⭐ **NOVEL(t) := `t` is not the factor tuple of any stored atom** (`stored_atom_tuples_s*_{A,B}.npz`).
- ⭐ **The headline count := |{ distinct t : t is produced, VALID and NOVEL }|.**
- ⚠ I will also report the weaker readings **beside** it, labelled, so nobody has to take my word that I
  did not pick the reachable one. ⛔ The headline is the strict one regardless of which is kinder.
- ⚠ Declared now, before I have measured any novelty number, precisely so it cannot be tuned afterwards.

### ⭐ What this does to the difficulty, stated honestly before I plan
Under the strict reading the bar **≥ 60 / 72** sits at **89 % of the decoder ceiling (67.67)** and at
**4.2× `knn_proto` (14.33)**, on a criterion where **every store-based arm currently scores exactly 0**.
⭐ My task file already said this: *"a ~4× improvement on our best, on a criterion `knn_proto` itself does
not meet."* The measurement agrees with it. ⛔ I am recording this before choosing a line of attack, so
that a later failure is read as the expected outcome of a hard bar and ⛔ not as a surprise.

## 2026-09-16T19:25Z — the VALID∧NOVEL instrument, built, VALIDATED against the harness, and run on frozen artefacts

- `$OUT/instrument/valid_novel.py` — **one function for every arm and rung**, so no arm can be counted by
  a different rule than another. Definitions are the ones declared at 19:20Z, ⛔ before any novelty number
  existed.
- ⭐ **VALIDATION — it reproduces the frozen harness's own columns EXACTLY on all 24 rows**
  (4 dynread rungs × 2 request sets × 3 seeds): `distinct` agrees on 24/24, and `exact_match` agrees to
  <1e-9 on the 12 B rows. ⛔ It is not a re-spelling of the metric; it is the same arithmetic, checked.
- `$OUT/probes/measure_frozen_novelty.py` reads only. ⛔ No encode, no decode, no classifier call — the
  classifier's tuples were persisted by the run that produced them. ⛔ Not a scored run.

| row (per seed 0/1/2) | distinct | distinct **novel** | distinct **valid** | ⭐ **VALID∧NOVEL** |
|---|---|---|---|---|
| `dynread` (A, full) on B | 12 / 15 / 13 | 12 / 15 / 13 | 0 / 0 / 0 | ⛔ **0 / 0 / 0** |
| `dynread/B` (full) on B | 15 / 18 / 15 | 15 / 18 / 15 | 0 / 0 / 0 | ⛔ **0 / 0 / 0** |
| `dynread/A_request_off` on B | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 | ⛔ **0 / 0 / 0** |
| `dynread/B_request_off` on B | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 | ⛔ **0 / 0 / 0** |

### ⭐⭐ FINDING 1 — NOVELTY IS FREE ON THIS STORE; **VALIDITY IS THE ENTIRE BAR**
`distinct_novel == distinct` on **24 of 24 rows**. ⛔ Not one produced tuple, on any rung, on any seed,
on either request set, coincided with a stored atom's tuple. ⭐ **The novelty clause of my target is
therefore not what is stopping anybody** — and it is close to *vacuous* on this store, for a structural
reason worth stating: rung A stores **64 atoms**, so novelty asks a 6-tuple over a 480,000-point grid to
miss 64 named points. It does, essentially always.
- ⛔ **This is a finding, ⛔ not a licence to redefine "novel".** The definition stays exactly as the
  charter fixes it ("not a stored atom's tuple"); I am reporting that, as fixed, it does not bind here.
  ⭐ It also means any future report of "N novel tuples" on this store is reporting **N distinct tuples**
  and should say so.
- ⭐ Corollary: `distinct_valid_novel == distinct_valid == 0` for every dynread rung. **The gap to the
  bar of 60 is entirely a CORRECTNESS gap.**

### ⭐ FINDING 2 — the `request_off` control is a **1-tuple collapse**, on 3/3 seeds and BOTH rungs
`dynread/{A,B}_request_off` render **exactly 1 distinct tuple of 72**, on every seed. ⭐ That is §E.4's
"dreaming is measured dead at zero temperature", reproduced here from the artefacts rather than quoted:
at β = 0 with `temp = 0.0` the chain has exactly one destination. ⛔ I am **not** yet calling dreaming
DEAD — that verdict is owed at the pre-registered finite-temperature grid, not at `temp = 0.0`, and
naming it dead now would be the excuse the charter forbids.

## 2026-09-16T19:35Z — the mechanism driver, and the unit cost this box actually has

- `$OUT/mech/mech_driver.py`. ⛔ **SELECTION ONLY; B NEVER ENTERS.** Charter §A.1 ("iterate freely on
  MECHANISM") and §A.2 ("selection on `val` only"). ⛔ It is not the scoring harness and nothing it
  prints is a scored number — the scored table is `exp_gen_cell` and only `exp_gen_cell`.
- ⛔⛔ **The hard constraint is enforced MECHANICALLY here, not promised.** `assert_cache_covers()` raises
  unless every row the driver will touch is already in the frozen Mac cache; features come from
  `gen_bridge.features_for_rows` — the harness's own gather **minus the extension branch**, so the driver
  is bit-identical to the harness on cached rows and **structurally cannot encode anything**. Measured
  shortfall: `store_train 0 · val 0 · MS 0`.
- ⭐ **VALIDATED AGAINST THE FROZEN RUN.** `clu` on **MS** (⛔ matched-SEEN rows, not held-out B):
  mine `distinct = 32, n = 198`; the frozen Mac manifest `distinct = 32, n = 198`. **MATCH.** The whole
  chain — store checkpoint, grouping, address book, codes, frozen decoder, hash-asserted classifier — is
  the harness's. The grouping re-derivation also logged *"reproduces the plan BIT-IDENTICALLY"*.

### ⭐⭐ THE UNIT COST — measured here, ⛔ not extrapolated from the Mac (addendum's timing trap)
| step | this box (5 threads) | the Mac anchors I was given |
|---|---|---|
| `chlu … exp_gen_cell --help` (import) | **2.6 s** | "~20+ min" (protocol §4) |
| full context: split, cache, store, grouping, address book | **5.2 s** | — |
| one arm scored end to end on 198 rows | **~2 s** | — |
| `build_landscape` budget **A** (64 atoms/well) | **17 s** | — |
| `build_landscape` budget **B** (512 atoms/well) | **151 s** | — |
| `x0` read + decode + classify, 512 rows | **0.2 s** | — |
⭐ **The `exp_gen_baselines` ~0.85 h/seed and `exp_gen_atom_mean` ~4 h anchors do NOT transfer.** The
expensive parts of my grid are seconds to minutes, not hours. ⛔ The addendum's timing trap does not bite;
compute is **not** my binding constraint. The re-encode gate is.

## 2026-09-16T19:42Z — ⭐⭐ FINDING 3: THE BOTTLENECK IS NOT COMPOSITION. IT IS READ FIDELITY ON **SEEN** DATA.

All on **`val`** — 512 rows the store was FITTED on, i.e. **combinations the model has seen**. ⛔ B never
entered. Same decoder, same hash-asserted external classifier, same `valid_novel` instrument.

| row (seed 0) | n | distinct | ⭐ **VALID (exact match)** | VALID∧NOVEL | exact-match frac |
|---|---|---|---|---|---|
| ⭐ **DECODER CEILING** (real cached features) | 512 | 512 | ⭐ **491** | — | **0.9590** |
| `clu` (codebook read, 524,288 B) | 512 | 490 | ⛔ **8** | 8 | 0.0156 |
| `x0` rung **A** (addressed read, 64 atoms/well, MATCHED bytes) | 512 | 463 | ⛔ **5** | 4 | 0.0098 |
| `x0` rung **B** (addressed read, 512 atoms/well, **8× bytes**) | 512 | 502 | **28** | 21 | 0.0547 |

and on **MS** (198 matched-seen rows): ceiling **165/198** (0.8333) · `clu` **1** · `x0`[A] **0** ·
`x0`[B] **16** (0.0808).

- ⭐ Consistent with §E.4, which I was told not to re-derive and did not: at MATCHED bytes the addressed
  read **loses** to the codebook read (5 vs 8), and at 8× bytes it **wins** (28 vs 8). Same ordering as
  §E.4's 0.5008-loses / 0.6852-wins, on a different statistic.
- ⛔⛔ **The finding: on data the store has SEEN, the decoder renders 96 % of combinations correctly and
  the best available read renders 5.5 % — at EIGHT TIMES the byte budget.** The gap is a factor of
  **17.5×**, and **none of it is a composition gap**: there is nothing to generalise on `val`.
- ⇒ **My target's bar of ≥ 60/72 (83 %) on HELD-OUT B is asking for 83 % exact match from a read that
  achieves 5.5 % on SEEN data.** ⛔ Composition is not the limiting factor. **Read fidelity is.**
- ⚠ Stated now, before I test any remedy, so that it cannot later be presented as a post-hoc explanation
  of a failure.

### ⭐ The next experiment is chosen to SPLIT that obstacle in two, cleanly
`x0` picks, per well, the atom the ADDRESS BOOK scores highest. Its failure has exactly two possible
causes and they demand opposite responses:
1. **REPRESENTATION** — no per-well choice of stored atoms can reconstruct the target at all; or
2. **ADDRESSING** — a good choice exists and the address book does not find it.
⭐ **The oracle-selection control separates them at a stroke:** replace the address book's argmax with the
atom that MINIMISES distance to the TRUE feature on that well's channels — same atoms, same bytes, same
decoder, ⛔ an oracle that is not a candidate arm (it reads features, so it is barred by §B.3 from ever
being one) and exists only to bound what addressing could ever buy.
- If the oracle is **also low** → the atom basis cannot represent the target; ⛔ no address book can help,
  and that is an impossibility result with a number on it.
- If the oracle is **high** → the basis is sufficient and the ADDRESS BOOK is the lever, which is in scope.

## 2026-09-16T19:50Z — ⭐⭐ FINDING 4: THE OBSTACLE IS **REPRESENTATION**, NOT ADDRESSING. With numbers.

**The oracle-selection control** (`$OUT/mech/run_oracle.py`): per well, the atom minimising distance to
the TRUE feature on that well's channels. ⛔ Not an arm and it can never be one — it reads features, so
§B.3 bars it forever. Same atoms, same bytes, same decoder, same instrument. `val` / `MS` only.

| rung | row | ORACLE valid | ORACLE rel-err | `x0` (the arm) valid | `x0` rel-err | index agreement |
|---|---|---|---|---|---|---|
| A (64 atoms/well, matched bytes) | val (512) | **7** (0.0137) | 0.4316 | **5** (0.0098) | 0.4432 | 0.7531 |
| A | MS (198) | **1** | 0.4783 | **0** | 0.4866 | 0.7687 |
| B (512 atoms/well, 8× bytes) | val (512) | **29** (0.0566) | 0.3122 | **28** (0.0547) | 0.3325 | 0.6699 |
| B | MS (198) | **17** | 0.3786 | **16** | 0.3964 | 0.6990 |

⛔⛔ **PERFECT ADDRESSING BUYS +2 ITEMS OF 512 AT RUNG A AND +1 OF 512 AT RUNG B.** The address book is
already within 0.4 percentage points of an oracle that is allowed to cheat.
⇒ **The address book is not the lever, and no better address book can be.** The atom basis cannot
represent the target: even with oracle selection the relative reconstruction error is **43 % (A) / 31 %
(B)**. ⭐ This closes off the entire "improve the addressing" family of ideas with a measurement instead
of an argument.

## 2026-09-16T19:53Z — ⭐⭐⭐ FINDING 5, THE GOVERNING BOUND: **THE STORE CANNOT ROUND-TRIP ITS OWN TRAINING DATA.**

`$OUT/mech/run_roundtrip.py`: hand the deployed 524,288 B store the **TRUE feature**, let it address
**itself** (`gen_store.address_codes`), read back (`gen_compose.read_compose`), decode, classify.
⛔ Not an arm — it takes features. ⭐ It is a **BOUND**: whatever a request-only read addresses *to*, it
must read *out of this store*, so ⛔ **no request-only read can beat this number.**

| row | store round-trip VALID | exact | rel-err | distinct codes | DECODER CEILING |
|---|---|---|---|---|---|
| `val` (512, **SEEN**) | ⛔ **10 / 512** | **0.0195** | 0.3486 | 501 / 512 | ⭐ **491 / 512** (0.9590) |
| `MS` (198) | ⛔ **1 / 198** | 0.0051 | 0.3840 | 37 / 198 | ⭐ **165 / 198** (0.8333) |

- ⚠ It is **not** an addressing collapse: 501 of 512 codes are distinct. The store separates these rows
  fine. It simply **cannot reproduce them**.
- ⭐⭐ **THE HEADLINE OF THIS HUB, and it is an upper bound, not an estimate:**
  **any request-only read through the pinned store is capped at ≈ 2 % exact match on SEEN data, against
  a 96 % decoder ceiling. My target's bar is 83 % on HELD-OUT data. That is ~43× above the cap.**
- ⛔ Composition, addressing, dynamics, temperature and novelty are **all downstream of this** and none of
  them can move it. ⭐ The target as written is not reachable through this store, and this is the number
  that says so.

### ⭐ Where the bits actually go — the sharp form of the diagnosis
The deployed store is **64 modes × 2048 dims × 4 B = 524,288 B**, read as **K = 5 wells × log₂ 64 =
30 bits** per read. ⚠ **30 bits is MORE than the 18.87 bits needed to name any one of the 480,000 grid
points.** ⛔ So the bottleneck is **not raw channel capacity** — it is that the quantiser spends its 30
bits on feature-space geometry rather than on factor identity. ⭐ That is §E.4's inherited fact
("the codebook costs 90.5 % of `scale` and 100 % of `orientation`") arriving from a completely different
direction, and it tells me exactly where a remedy would have to act.

### ⭐ The one remedy the charter explicitly sanctions — and the arithmetic that makes it interesting
§B: *"Everything else may change: kernel, coupling, **partition**, codebook, dynamics, temperature,
decoder interface, encoder."* The **partition** is the lever, and at **constant store bytes**:
- store bytes `= M · Σ_k d_k · 4 = M · 2048 · 4` — ⭐ **independent of K**, because `Σ_k d_k = D` always.
- bits per read `= K · log₂ M` — ⭐ **linear in K**.
⇒ at the SAME 524,288 B, K = 5 → 30 bits, K = 64 → 384 bits, K = 256 → 1,536 bits.
⚠ ⛔ **But the addressing head is NOT free and I will not pretend it is:**
`(K·M + Σ_f V_f·K·M)·4 = 232·K·M` B (verified: K=5, M=64 → 74,240 B = the shipped
`addressing_head_bytes_reference`). At K = 2048 it is 30.4 MB — **58× the store**. ⛔ Growing the head
and calling the arm "matched bytes" is precisely the accounting trick that voids a result.
⭐ So the honest budget is **TOTAL = store + head = M·(8192 + 232·K) = 598,528 B**, and K trades against
M inside it: K=32 → M=38 (168 bits) · K=128 → M≈15 (509 bits) · K=512 → M≈4.7 (1,143 bits).
⭐ **Next: measure the round-trip fidelity of that family at constant TOTAL bytes.** It is the only route
to the target that the charter permits, and it attacks the measured obstacle directly rather than
the ones already ruled out.

## 2026-09-16T19:26:15Z — ⛔ CORRECTION BULLET: my own timestamps above were ESTIMATED, not read.

- ⛔ **The headings `19:35Z`, `19:42Z`, `19:50Z`, `19:53Z` above are wrong.** I wrote them from my sense
  of elapsed time instead of reading the clock. ⭐ Per the charter's "corrections are new bullets, never
  edits", they stay on the page and this bullet corrects them.
- **The real clock, read just now:** `START = 2026-09-16T19:10:40Z`, `NOW = 2026-09-16T19:26:15Z` ⇒
  ⭐ **0.26 h elapsed (15 min), 11.74 h remaining.** Everything logged above — boot, both blockers, the
  cache-coverage measurement, the instrument and its validation, the frozen table, the driver, the x0
  probe, the oracle control and the round-trip bound — happened inside **fifteen minutes**, not the ~45
  those headings imply.
- ⭐ This is exactly the failure LAUNCH-COMMON §1b names: *"⛔ Not model-perceived time … your sense of
  elapsed time is unreliable over a long run."* ⭐ **Standing correction: from here on every timestamp in
  this log is the output of `date -u`, never an estimate.** Elapsed is computed against `START_UTC.txt`.
- ⚠ The *ordering* and the *content* of the bullets above are unaffected; only their clock labels were.

## 2026-09-16T19:31:25Z — ⭐ FINDING 6: THE PARTITION IS A REAL LEVER — **8× on round-trip fidelity at CONSTANT DECLARED BYTES**

`$OUT/mech/run_partition_sweep.py`, k-means++ **init only** (untrained), seed 0, `val`/`MS`.
⛔ Grouping is **label-free** (`grouping_labels(corr, K, "correlation", …)` — channel correlation only),
so §B.6 is untouched: no label enters the landscape, the read or the write.
⛔ Budget held at **TOTAL = store + addressing head = 524,288 + 74,240 = 598,528 B**, with
`M(K) = ⌊598,528 / (8,192 + 232·K)⌋`. ⛔ Every row below is **at or under** the shipped total.

| K | M | bits/read | store B | head B | total B | ⭐ `val` VALID/512 | exact | rel-err | `MS` VALID |
|---|---|---|---|---|---|---|---|---|---|
| **5** (shipped) | 64 | 30.0 | 524,288 | 74,240 | **598,528** | 6 | 0.0117 | 0.4366 | 1 |
| 16 | 50 | 90.3 | 409,600 | 185,600 | 595,200 | 9 | 0.0176 | 0.4161 | 3 |
| 32 | 38 | 167.9 | 311,296 | 282,112 | 593,408 | 35 | 0.0684 | 0.4049 | 4 |
| ⭐ **64** | 25 | 297.2 | 204,800 | 371,200 | 576,000 | ⭐ **47** | ⭐ **0.0918** | 0.3916 | ⭐ **20** |
| 128 | 15 | 500.1 | 122,880 | 445,440 | 568,320 | 1 | 0.0020 | 0.5127 | 1 |
| 256 | 8 | 768.0 | 65,536 | 475,136 | 540,672 | 5 | 0.0098 | 0.5201 | 3 |
| 512 | 4 | 1,024.0 | 32,768 | 475,136 | 507,904 | 17 | 0.0332 | 0.5057 | 11 |

- ⭐ **K = 64 / M = 25 gives 47/512 against the shipped partition's 6/512 — ~8×, at 22,528 B UNDER the
  shipped total budget.** ⛔ Nothing was bought with bytes; only the partition changed.
- ⭐ The curve is **non-monotone in bits**, and that is the interesting part: 1,024 bits/read (K=512)
  scores *worse* than 297 bits (K=64). ⛔ Bit rate is not the currency. Past K≈64, `M` falls below what a
  well needs to cover its own channel block and rel-err **rises** (0.39 → 0.52). The optimum is the
  trade, not either end.
- ⚠ **The honest caveat, stated with the result:** the head is 232·K·M B and grows with K, so this is a
  *reallocation* inside a fixed total, ⛔ not a free lunch. At K=64 the store shrinks to 204,800 B and the
  head grows to 371,200 B. ⛔ Anyone quoting "store bytes" alone would be quoting a 2.6× SMALLER store and
  that would be the accounting trick I refuse to make.
- ⚠ These are **untrained** k-means++ inits. The shipped trained store scores 10/512; its untrained init
  scores 6/512, so training is worth ~1.7× here. A trained sweep over K ∈ {5,24,32,48,64,80,96,128} is
  running now.

### ⛔ What this does NOT do — said before the trained numbers land, so it cannot be spun afterwards
Even at **9.2 %**, the round-trip bound is **~10× below the 95.9 % decoder ceiling** and **~9× below my
target's 83 % bar**. ⭐ An 8× improvement on a quantity that needs ~50× is a **mechanism finding, not a
win**, and I will report it as exactly that.

## 2026-09-16T19:33Z — ⭐⭐⭐ FINDING 7: AN IDENTITY THAT ANSWERS FALSIFIERS 1 AND 2 **WITH ZERO SCORED RUNS**

⭐ **On the held-out set B, for EVERY arm:**
```
distinct_valid_novel  =  distinct_valid  =  72 × exact_match
```
It is an identity, not an estimate, and it rests on two facts I **measured** rather than assumed:

1. ⭐ **All 72 B target tuples are pairwise DISTINCT.** (`n_targets_distinct = 72`, verified on all 12
   frozen B rows.) ⇒ a VALID render reads as its own target, and two valid renders are never the same
   tuple, so *count of valid items = count of distinct valid tuples*.
2. ⭐ **All 72 B target tuples are NOVEL — under the strictest reference available, not the kindest.**
   `$OUT/probes/novelty_reference.py`, all 3 seeds:
   | novelty reference | size | B tuples that are NOVEL |
   |---|---|---|
   | the stored ATOMS (my target's wording) | 64 | **72 / 72** |
   | the whole `store_train` pool | 32,768 | **72 / 72** |
   | ⭐ **every train row in the split** | **479,928** | ⭐ **72 / 72** |
   ⛔ This is Montero's recombination-to-element hold-out doing exactly what it says: those 72
   combinations occur **nowhere** in training. ⇒ the novel clause removes nothing on B, **for any arm,
   under any of the three references.** ⛔ I checked the strictest one precisely so this cannot be read
   as novelty being made easy.

### ⇒ THE HEADLINE COLUMN OF THE FROZEN TABLE, DERIVED — ⛔ no new run, no scored run spent
| arm | ⭐ **distinct VALID NOVEL tuples / 72** (mean ± sd over 3 seeds) |
|---|---|
| ⭐ **DECODER CEILING** | **67.67 ± 2.49** |
| ⭐ **MY BAR** | **≥ 60** |
| `knn_proto` (the rival ceiling) | **14.33 ± 4.78** |
| `clu` | ⛔ **0.00 ± 0.00** |
| `lace` / `lace/ld_corrected` / `lace/ld_published` | ⛔ **0.00** each |
| `ebm_compose` | ⛔ **0.00** |
| `dynread` A / `dynread/B` | ⛔ **0.00** each |
| `dynread/A_request_off`, `dynread/B_request_off`, `store_removed` | ⛔ **0.00** each |

### ⛔⛔ FALSIFIERS 1 AND 2 ARE ANSWERED, AND BOTH ARE **NOT MET**
- **Falsifier 1** — *beat every rival incl. `knn_proto` on the compositional factor*: the CLU arm scores
  **0**, `knn_proto` scores **14.33**. ⛔ **NOT MET.** ⛔ Not close, and not a seed-spread question:
  the gap is 14.33 against a control whose sd is 0.00.
- **Falsifier 2** — *≥ 60/72 distinct valid novel tuples*: our best is **0**; the best **anything** on
  this table reaches is `knn_proto`'s **14.33**; even the **decoder ceiling** is **67.67**, so the bar
  sits at **89 % of the physical maximum**. ⛔ **NOT MET.**
- ⭐ My task file said *"a ~4× improvement on our best, on a criterion `knn_proto` itself does not meet"*.
  ⛔ The measurement is harsher than that: on THIS criterion our best is not 4× short, it is **zero**.
- ⭐ **And the bar restates exactly**: "≥ 60/72 distinct valid novel tuples" **is** "`exact_match` ≥ 0.8333
  on B". ⛔ One number, no room to interpret.

⭐ **Why this matters procedurally:** the two falsifiers that define this hub's target were resolvable
from **frozen, Mac-produced artefacts**. ⛔ **I have spent 0 of my 3 scored runs and, for these two
questions, need none.** The scored-run budget is still entirely unspent and the operator's answer to
QUESTION 1 is still outstanding but is no longer on this hub's critical path.

## 2026-09-16T19:44:19Z — ⭐⭐ TARGET 3 ANSWERED: **DREAMING IS NOT DEAD — IT IS INERT.** The full pre-registered grid.

`$OUT/mech/run_temp_grid.py`, budget A, seed 0, **512 `val` chains**, 1000 settling steps, ~45–55 s/unit,
8 units. ⛔ Grid `temp ∈ {0.0, 0.005, 0.05, 0.5}` **exactly as pre-registered** — ⛔ not widened, before
or after seeing a result. β = `request_strength`: **0.0 is DREAMING**, 1.0 is the steered control on the
**same chain**. ⛔ `val` only; B never entered.

| β | T | SETTLED distinct | novel | valid | ⭐ **VALID∧NOVEL** | the LAUNCH, same chain (distinct / valid) |
|---|---|---|---|---|---|---|
| **0** (dream) | **0.0** | ⛔ **1** | 1 | 0 | ⛔ **0** | 463 / 5 |
| **0** | **0.005** | 23 | 23 | 0 | ⛔ **0** | 463 / 5 |
| **0** | **0.05** | 86 | 86 | 0 | ⛔ **0** | 463 / 5 |
| **0** | **0.5** | ⭐ **463** | ⭐ **447** | 5 | ⭐ **4** | 463 / 5 |
| 1 (steered) | 0.0 | 1 | 1 | 0 | 0 | 463 / 5 |
| 1 | 0.005 | 23 | 23 | 0 | 0 | 463 / 5 |
| 1 | 0.05 | 94 | 94 | 0 | 0 | 463 / 5 |
| 1 | 0.5 | 463 | 447 | 5 | 4 | 463 / 5 |

### ⭐ The verdict, in the words the charter demands — ⛔ and it is NOT a null
**Dreaming is NOT DEAD.** At the **declared temperature T = 0.5**, β = 0 yields **463 distinct tuples of
512, 447 of them novel, and 4 distinct VALID NOVEL tuples**. ⭐ Falsifier 3 asked for *"valid novel tuples
at some declared temperature, or DEAD"* — the answer is **valid novel tuples at T = 0.5**, named.

### ⭐ The zero-temperature collapse is CONFIRMED, and its cause is confirmed with it
1 → 23 → 86 → 463 distinct as T rises. ⭐ §E.4's diagnosis — *"pure zero-temperature descent has no
reason to produce more than one mode"* — is exactly right, and temperature restores diversity
monotonically over the pre-registered grid. ⛔ I did not need to widen the grid to see it.

### ⛔⛔ BUT THE TWO CONTROLS KILL THE HEADLINE, AND THEY ARE THE POINT
1. ⛔⛔ **β DOES NOTHING.** β = 0 and β = 1 agree **exactly** at T = 0.0 (1 = 1), T = 0.005 (23 = 23) and
   T = 0.5 (463 = 463, 4 = 4), and differ by 86 vs 94 at T = 0.05. ⭐ `exp_gen_cell`'s own docstring
   states what this means: *"if it scores the same, the arm is settling onto a mode rather than being
   steered, and that is the result."* ⇒ **the request ENERGY is not steering the chain at any
   temperature on the grid.**
2. ⛔⛔ **At T = 0.5 the settled state is INDISTINGUISHABLE FROM THE LAUNCH.** The zero-step launch
   already gives **463 distinct / 5 valid**; after 1000 settling steps at T = 0.5 it is **463 distinct /
   5 valid**. ⭐ The dynamics contribute **nothing** — the chain is hot enough not to settle anywhere.
⇒ ⭐ **The honest verdict: dreaming is NOT dead, but it is INERT.** At T ≤ 0.05 the dynamics *destroy*
the launch's diversity (463 → 1/23/86); at T = 0.5 they merely *fail to destroy* it. ⛔ At no temperature
on the pre-registered grid do the dynamics ADD anything the launch did not already have, and ⛔ at no
temperature does the request energy change the answer.
- ⚠ **Reported this way deliberately.** "Dreaming yields 4 valid novel tuples at T = 0.5" is *true* and
  would be a **dishonest headline**: the zero-step launch yields the same, so the number is the launch's,
  not dreaming's. ⭐ The §E.4 inheritance — *"the launch beats the settled state"* — reappears here as
  *"the launch IS the settled state, once you stop the dynamics destroying it."*
- ⛔ This also re-confirms §E.4's *"⛔ More steps makes it worse"* from a new direction: 1000 steps at
  T = 0 cost 462 of 463 distinct tuples.

## 2026-09-16T19:50:16Z — ⭐⭐⭐ FINDING 8: THE STORE'S ERROR IS NOT TOO BIG. IT POINTS THE WRONG WAY.

`$OUT/mech/run_capacity.py`, `val` (512), seed 0. Two diagnostics (⛔ both touch true features, so ⛔
neither is or could be an arm). ⛔ B never entered.

**(a) How much error can the decoder actually absorb?** Isotropic Gaussian error of relative norm `r`:

| rel-err | 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | **0.30** | **0.35** | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| exact_match | 0.9590 | 0.9551 | 0.9512 | 0.9219 | 0.9004 | 0.8477 | **0.7734** | **0.7559** | 0.6738 | 0.5605 |
| valid / 512 | 491 | 489 | 487 | 472 | 461 | 434 | **396** | **387** | 345 | 287 |

⭐ **The decoder is extremely robust: at rel-err 0.20 it still renders 90 % of combinations correctly,
and at 0.35 it renders 76 %.**

### ⛔⛔ AND THERE IS THE CONTRADICTION THAT EXPLAINS EVERYTHING
| error source | rel-err | exact_match | valid / 512 |
|---|---|---|---|
| isotropic Gaussian noise | **0.3500** | ⭐ **0.7559** | **387** |
| ⛔ **the store's own round-trip** | **0.3486** | ⛔ **0.0195** | ⛔ **10** |

⭐⭐ **AT THE SAME RELATIVE ERROR MAGNITUDE, RANDOM NOISE COSTS 20 POINTS AND THE STORE COSTS 94.**
⇒ ⛔ **The store's error is not too LARGE. It is STRUCTURED, and structured in the worst possible
direction.** Per unit of norm, the quantiser's residual is ~**39× more damaging** than random error.
⭐ This is the single most useful thing I have measured, because it says the problem is **not** capacity
and **not** bytes — both of which I had been about to spend effort on.

**(b) PCA truncation** — the *variance-optimal* linear compressor, fitted label-free on train rows:

| r | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| rel-err | 0.4170 | 0.3586 | 0.3069 | 0.2698 | 0.2413 | 0.2147 | 0.1852 | 0.1406 |
| exact_match | 0.0078 | 0.0508 | 0.2168 | 0.3438 | 0.3848 | 0.4551 | 0.5254 | 0.6738 |

⭐ **PCA is ALSO far worse than noise at equal rel-err** (r=32: rel-err 0.3069 → 0.2168, vs noise at
0.30 → 0.7734). ⚠ Even keeping **1024 of 2048** components — rel-err 0.1406 — only reaches 0.6738, which
isotropic noise achieves at rel-err **0.40**.

### ⭐⭐ THE UNIFIED DIAGNOSIS — and it explains a fact I was handed and told not to re-derive
**The factor information the classifier reads lives in the LOW-VARIANCE directions of the feature
distribution.** Every *variance-optimal* compressor — vector quantisation, k-means, PCA — spends its
budget on high-variance directions and discards precisely the directions that carry factor identity.
Isotropic noise does not preferentially destroy them, which is why it is so much cheaper per unit norm.
- ⭐ This **derives**, from a completely independent direction, §E.4's inherited fact that *"the codebook
  costs 90.5 % of `scale` and 100 % of `orientation`"*: ⛔ `scale` and `orientation` are exactly the
  factors whose feature-space signature is low-variance. I was told not to re-derive it; I did not set
  out to, and it fell out.
- ⭐ It also explains why my partition sweep helped (8×) but plateaued: more, smaller wells re-allocate
  the *same* variance-driven objective; they do not change what it optimises.

### ⭐ THE REMEDY THIS POINTS AT — cheap, label-free, byte-honest, and testable now
If the damage is that the quantiser's metric is wrong, then **change the metric, not the budget**:
quantise in a **per-channel standardised (diagonally whitened)** space, so low-variance channels are not
sacrificed to high-variance ones.
- ⭐ **Label-free** — a per-channel σ over train features; ⛔ no label enters the write (§B.6 intact).
- ⭐ **Byte-honest** — a diagonal whitening is **D × 4 = 8,192 B**, 1.4 % of the 598,528 B budget. ⛔ A
  full D × D whitening would be 16.8 MB and I am **not** proposing that; it would be 28× the budget.
- ⭐ §B explicitly permits the **kernel** and the **codebook** to change. This changes the kernel the
  codebook is fitted under, and nothing else.
**Predicted before running:** rel-err will get WORSE in the un-whitened norm and exact_match will get
BETTER. ⛔ If exact_match does not rise, the diagnosis above is wrong and I will say so.

## 2026-09-16T20:00:18Z — ⛔ FINDING 9: MY PREDICTED REMEDY **FAILED**. Per-channel whitening does not help.

`$OUT/mech/run_whiten.py` — quantise in a per-channel standardised metric, σ fitted label-free on train
features (σ ratio max/min = **6012×**, so there is plenty of per-channel spread to exploit).

| K | M | metric | total B | `val` VALID/512 | exact | rel-err | `MS` VALID |
|---|---|---|---|---|---|---|---|
| 5 | 64 | plain | 598,528 | 6 | 0.0117 | 0.4366 | 1 |
| 5 | 64 | ⭐ whitened | 606,720 ⚠ over | **6** | 0.0117 | 0.4586 | 0 |
| 32 | 38 | plain | 593,408 | 35 | 0.0684 | 0.4049 | 4 |
| 32 | 38 | ⭐ whitened | 601,600 ⚠ over | ⛔ **17** | 0.0332 | 0.4248 | 1 |
| 64 | 25 | plain | 576,000 | 47 | 0.0918 | 0.3916 | 20 |
| 64 | 25 | ⭐ whitened | 584,192 OK | ⛔ **39** | 0.0762 | 0.4192 | 5 |

⛔ **I pre-declared at 19:50Z: *"Predicted … exact_match will get BETTER. ⛔ If exact_match does not rise,
the diagnosis above is wrong and I will say so."*** It did not rise. At K=5 it is unchanged; at K=32 it
**halves**; at K=64 it falls 47 → 39. ⭐ **Saying so: the low-variance-CHANNEL half of my diagnosis is
wrong.** ⚠ (Two of the whitened rows are also 1.4 % **over** budget once the 8,192 B σ vector is charged
honestly — ⛔ recorded rather than dropped, though it is not why they lost.)

## 2026-09-16T20:01:23Z — ⭐⭐⭐ FINDING 10: THE REAL CAUSE. A CONTROL LADDER AT ONE FIXED REL-ERROR.

`$OUT/mech/run_residual.py`. Let `e_i = roundtrip_i − x_i`. ⭐ **Every row below has rel-err = 0.3486,
exactly** — only the error's *structure* differs.

| perturbation of the true feature | rel-err | exact | ⭐ VALID / 512 |
|---|---|---|---|
| none (decoder ceiling) | 0.0000 | 0.9590 | **491** |
| `x + g`, isotropic Gaussian at matched norm | 0.3486 | 0.7676 | **393** |
| `x − e`, the residual **negated** | 0.3486 | 0.3164 | **162** |
| `x + e_π`, the residual **permuted across items** | 0.3486 | 0.1855 | **95** |
| ⛔ `x + e`, **the store's actual output** | 0.3486 | 0.0195 | ⛔ **10** |

⭐⭐ **The damage decomposes cleanly into two multiplicative factors:**
1. **The residual SUBSPACE** — where quantisation error lives in feature space — costs **393 → 95**
   (**4.1×**), and that survives permutation, so it is a property of the space, not of the item.
2. ⭐⭐ **SELF-ALIGNMENT costs another 95 → 10 (9.5×)**, and this is the dominant term. Attaching each
   residual to **its own** item is what destroys it.
- ⭐ `x − e` (162) beats `x + e` (10) by **16×** at identical magnitude. Moving *away* from the codeword
  is far less harmful than moving *toward* it. ⛔ That is not a subtle effect; it is the whole effect.

### ⭐⭐ WHY THIS IS AN IMPOSSIBILITY ARGUMENT, NOT A TUNING PROBLEM — and why whitening had to fail
A nearest-codeword read's error is, **by construction**, the vector from the item to its codeword: it is
*precisely the component of the item that the store does not contain*. The classifier reads factor
identity from precisely that component. ⭐ So the error of any read whose output is a stored object — a
codeword, an atom, a convex combination of them — is **always** self-aligned, and self-alignment is the
9.5× term.
- ⇒ ⛔ **Re-weighting the metric cannot help**, because self-alignment is invariant to the metric: change
  the metric and the nearest codeword changes, but the residual still points from the item to it. ⭐ That
  is exactly why Finding 9 failed, and it retro-explains it rather than excusing it.
- ⇒ ⛔ It also explains why the **oracle** bought only +2/512 (Finding 4) and why the **partition** sweep
  plateaued (Finding 6): neither removes self-alignment.
- ⇒ ⭐ The **only** remaining lever is to make the residual **smaller** — i.e. tile feature space more
  finely — which is a pure **capacity/bytes** question. Measuring that requirement is running now.

## 2026-09-16T20:22:09Z — ⭐⭐⭐ FINDING 11, THE HUB'S HEADLINE: **THE BAR COSTS ~147,000× THE BYTE BUDGET.**

`$OUT/mech/run_requirement.py`. Two measurements, combined into one requirement.

**(a) How small must the residual be?** Scale the store's *own* residual: `x + α·e`. ⭐ This keeps the
residual's damaging self-aligned structure (Finding 10), so it is ⛔ **not** a heuristic conversion
through the isotropic-noise curve.

| α | 1.0 | 0.5 | 0.3 | 0.25 | 0.2 | **0.15** | 0.1 | 0.05 | 0 |
|---|---|---|---|---|---|---|---|---|---|
| rel-err | 0.3486 | 0.1743 | 0.1046 | 0.0872 | 0.0697 | **0.0523** | 0.0349 | 0.0174 | 0 |
| exact | 0.0195 | — | — | 0.7285 | 0.7969 | **0.8711** | 0.9160 | 0.9473 | 0.9590 |

⭐ **α\* = 0.1754 for the bar (exact = 60/72 = 0.8333) ⇒ the residual must shrink 5.7×, to rel-err
0.0612.** (For merely matching `knn_proto`'s 14.33/72 = 0.1991: α\* = 0.5406, rel-err 0.1884.)

**(b) How does the residual shrink with bytes?** Same partition (K = 5), M from 16 to 2048 — ⭐ a
**128× span** of real measurements, ⛔ not a model:

| M | 16 | 32 | 64 (shipped) | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| store B | 131,072 | 262,144 | **524,288** | 1,048,576 | 2,097,152 | 4,194,304 | 8,388,608 | 16,777,216 |
| rel-err | 0.5428 | 0.4816 | **0.4366** | 0.3978 | 0.3517 | 0.3157 | 0.2758 | 0.2394 |
| valid/512 | 0 | 4 | **6** | 14 | 19 | 26 | 56 | 73 |

⭐ **Power law, R² = 0.9960 over 128×:  rel-err = e^1.346 · bytes^(−0.1651).** The seven per-doubling
local exponents lie in **[−0.2042, −0.1344]** — no trend, so the fit is not hiding a bend.

### ⇒ THE REQUIREMENT, WITH ITS SENSITIVITY — ⛔ and I flag the extrapolation rather than hide it
| exponent used | reach **the bar (60/72)** | merely **match `knn_proto`** |
|---|---|---|
| slowest measured decay (−0.1344) | 1,172 GB — **2,234,595×** | 272 MB — **520×** |
| ⭐ **global fit (−0.1651)** | ⭐ **77.3 GB — 147,427×** | ⭐ **85 MB — 162×** |
| fastest measured decay (−0.2042) | 7.9 GB — **15,097×** | 32 MB — **61×** |
| ⛔ twice as favourable as ANY measured value | 0.20 GB — **383×** | 6.7 MB — **13×** |

- ⚠ **The honest caveat:** reaching the bar extrapolates ~4,600× beyond the largest store I measured.
  ⭐ **But the conclusion does not depend on the extrapolation.** Even granting an exponent **twice as
  favourable as any of the seven measured**, the bar still needs **383×** the byte budget. And the
  **162×** needed merely to *match* `knn_proto` is only ~5× beyond the measured range — that one is
  solid.
- ⭐⭐ **The comparison that settles it:** storing **all 480,000 training rows verbatim** at 2048 float32
  is **3.93 GB**. The vector quantiser needs **19.6× MORE than that** to reach the bar. ⛔ At this
  dimensionality the codebook is ~20× *less* byte-efficient than a verbatim lookup table — which is
  §C.4's *"any arm whose settled state sits on its nearest atom is a lookup table and is reported as
  one"* arriving as an **economic** fact rather than a behavioural one.
- ⭐ Cause is textbook and now measured here: VQ distortion falls as `M^(−1/d)` and the wells have
  `d_k ∈ [303, 573]`. An exponent of −0.165 in *bytes* is what a ~400-dimensional quantiser gives. ⛔ No
  partition, address book, temperature, coupling or dynamics changes that exponent — only `d` does, and
  `d` is set by the encoder's 2048-channel feature, which is upstream of everything this hub may touch.

## 2026-09-16T20:32:14Z — ⭐ FINDING 12: THE SPREAD. ⛔ And the honest statement of what spread I do NOT have.

`$OUT/mech/run_robustness.py`. ⛔⛔ **I cannot report BRIDGE-SEED spread on this box.** Seeds 1 and 2
are cache-blocked (Finding at 19:14Z), so **every number in this log is SEED 0**. ⭐ That is a real
limitation of this hub's output and I am not dressing it up. What I *can* report is two other spreads,
each labelled for what it is:

**(a) Bootstrap over the 512 `val` rows** (2,000 resamples) — sampling noise in the count itself:
| row | observed / 512 | 95 % CI | sd |
|---|---|---|---|
| ⭐ DECODER CEILING | **491** | **[482, 499]** | 4.45 |
| store round-trip (the BOUND) | **10** | [4, 16] | 3.12 |
| `clu` (codebook read) | **8** | [3, 14] | 2.80 |

⚠ **`clu` (8) and the store round-trip (10) are NOT distinguishable** — their CIs overlap almost
entirely. ⭐ That is itself the point: **the codebook read is already AT its store's round-trip bound.**
There is no addressing loss left to recover — consistent with Finding 4's oracle result arrived at from
the codebook side instead of the atom side.
⭐ And the separation that matters is not in doubt: **[482, 499] vs [3, 16]** do not come close to touching.

**(b) Spread over the ARM's own seed** (atom selection + coupling fit re-drawn; ⛔ bridge seed still 0):
| rung | arm seeds 0 / 1 / 2 | mean ± sd |
|---|---|---|
| `x0` A (matched bytes) | 4 / 5 / 2 | **3.67 ± 1.25** |
| `x0` B (8× bytes) | 21 / 21 / 17 | **19.67 ± 1.89** |
⭐ The A-vs-B ordering is stable across all three arm seeds and far outside the spread. ⭐ Every one of
these is 25–130× below the ceiling's 491.

## 2026-09-16T20:32Z — trained partition sweep: partial, and a caveat I must publish against myself
Two of eight units done (it is slow: ~915 s and ~2,400 s per unit, still running, resumable):
- `K=5, M=64` **trained**: `val` VALID = **4** / 512 (rel-err 0.3480).
⚠ ⛔ **That is LOWER than the same partition's untrained k-means++ init (6) and lower than the frozen
`gen-read-compose` store's (10).** ⭐ So **my own store training is not better than the shipped one**,
and every comparison I drew from the *init* sweep (Finding 6) would need re-checking after training
before anyone builds on it. ⛔ I am flagging this against my own most favourable result rather than
letting the init numbers stand unqualified.
⭐ It does not touch Findings 7, 10 or 11: those use the **frozen shipped store** and a measured
bytes↔rel-err power law, neither of which depends on my training.

## 2026-09-16T20:46:38Z — ⭐⭐ INDEPENDENT REPLICATION (charter §A.5) — and it **REFUTED ONE OF MY NUMBERS**

A separate agent re-derived my headline numbers **with its own code**, barred from reading
`mech/`, `instrument/` or `probes/`. Its report: `$OUT/replication/REPLICATION.md`.

| claim | mine | replication | verdict |
|---|---|---|---|
| frozen exact-match, all 12 rows | ceiling 67.67 ± 2.49, `knn_proto` 14.33 ± 4.78, **all ten other arms exactly 0** | identical, "0.0 in every seed, **not rounded**" | ⭐ **CONFIRMED** |
| 72 B targets distinct / none in train | 72 / 72 / 0 | 72 / 72 / 0 | ⭐ **CONFIRMED** |
| decoder ceiling on `val` | 491 / 512 | **491 / 512** | ⭐ **CONFIRMED exactly** |
| store round-trip bound | 10 / 512, rel-err 0.3486 | **10 / 512**, **0.34860397** | ⭐ **CONFIRMED exactly** |
| residual **negated** (`x − e`) | 162 | **162** | ⭐ **CONFIRMED exactly** |
| residual **permuted** | **95** | 117.3 ± 8.0 over 15 draws, min 104 | ⛔ **REFUTED** |
| isotropic Gaussian | 393 | 384.4 ± 9.5 over 15 draws | ⚠ within draw noise, **over-precise as stated** |

### ⛔ I RE-RAN THE TWO DISPUTED CONTROLS OVER 25 DRAWS. **THE REPLICATION IS RIGHT AND I WAS WRONG.**
- **permuted residual: 115.6 ± 7.7, range [102, 134]** over 25 independent permutations. ⛔ **My single
  draw of 95 lies BELOW the entire range.** ⭐ **Correction: the permuted-residual control is ≈ 116, not
  95**, and quoting a single draw as though it were a point estimate was a real error on my part.
- **isotropic Gaussian: 382.4 ± 11.1, range [359, 406]** over 25 draws. ⭐ Correction: **≈ 382, not 393.**
- ⭐ **Finding 10's ladder, corrected and now with spreads — the conclusion is UNCHANGED:**
  | perturbation (rel-err ≈ 0.3486) | valid / 512 |
  |---|---|
  | isotropic Gaussian, matched norm | **382.4 ± 11.1** |
  | residual negated (`x − e`) | **162** (deterministic) |
  | residual permuted across items | **115.6 ± 7.7** |
  | ⛔ the store's actual output (`x + e`) | ⛔ **10** (deterministic) |
  The two factors become **3.3×** (subspace) and **11.6×** (self-alignment) instead of 4.1× and 9.5×.
  ⭐ Self-alignment is, if anything, an even larger share of the damage than I first reported.
- ⚠ Also accepted: my `±` used **population** sd (ddof=0), matching `cell_results.json`'s own convention.
  Sample sd at n = 3 is **1.225× larger**: ceiling **67.67 ± 3.06**, `knn_proto` **14.33 ± 5.86**.

### ⭐ THE REPLICATION'S BEST OBJECTION, AND I HAVE NOW MEASURED IT
It objected — correctly — that at 19:20Z I declared I would report the **weaker reading of "valid"**
("the produced tuple lands in the held-out cell at all", rather than "equals the request") and then
never did, and it named `knn_proto` as the case where the two could diverge sharply.
⭐ **Measured, on the frozen B artefacts, for every rung and seed where the tuples exist:**

| row | STRICT (valid = equals the request) | ⭐ **WEAK (lands in the cell at all, and novel)** | items landing in the cell |
|---|---|---|---|
| `dynread` A / `dynread/B` / both `request_off`, seeds 0,1,2 — **12 rows** | **0** | ⭐ **0** | ⛔ **0 / 72, every row** |

⭐ **The two readings AGREE at zero.** ⛔ Not one of 864 dynread renders lands anywhere in the held-out
cell. ⭐ So the objection does not change any verdict I drew — but it was right that I owed the number.
- ⚠ **It still stands for `knn_proto`, `clu`, `lace` and `ebm_compose`:** `exp_gen_cell._record` persists
  per-render tuples **only for `dynread`**, so I cannot compute their weak count without a run. ⭐ The
  replication's indirect evidence is worth taking seriously — `knn_proto` gets four factors at 1.000 and
  `shape` at 0.986, so its weak count could be far above its strict 14.33. ⛔ **Recorded as an OPEN
  QUESTION, and `PREREG-hub-composition-1.md` is amended to persist tuples for every arm.**

### ⭐ ANOTHER REPLICATION OBJECTION, MEASURED: "10/512 hides a selective loss". It does. Here it is.
Per-factor accuracy of the store round-trip vs the decoder ceiling, on `val`, **with
`n_values_realised`** (⛔ the column whose absence produced a wrong headline in this programme once):

| factor | n values realised | round-trip | ceiling | ⭐ fraction of the ceiling LOST |
|---|---|---|---|---|
| `floor_hue` | 10 | 0.9980 | 1.0000 | 0.2 % |
| `wall_hue` | 10 | 0.9824 | 1.0000 | 1.8 % |
| `object_hue` | 10 | 0.6914 | 1.0000 | 30.9 % |
| `shape` | 4 | 0.6094 | 0.9883 | 38.3 % |
| ⛔ **`scale`** | 8 | ⛔ **0.2383** | 0.9727 | ⛔ **75.5 %** |
| ⛔ **`orientation`** | 15 | ⛔ **0.1738** | 0.9980 | ⛔ **82.6 %** |

⭐⭐ **This is §E.4's inherited *"the codebook costs 90.5 % of `scale` and 100 % of `orientation`"*,
measured independently on a different instrument and a different row set.** ⛔ I did not set out to
re-derive it and was told not to; it fell out of the replication's objection, and it is the same two
factors, in the same order, by the same margin. ⭐ Hue survives; geometry does not.

### ⚠ Three further objections I ACCEPT and record without having closed them
1. ⛔ **The round-trip bound is IN-SAMPLE for the store**: all 512 `val` rows are inside `store_train`.
   ⭐ The direction is conservative — an in-sample store does *better*, so 10/512 is a genuine upper
   bound — but the headline "491 vs 10" does pit a near-held-out decoder number against an in-sample
   store number, and that asymmetry should be stated wherever it is quoted. It now is.
2. ⚠ **The isotropic Gaussian is a weak control in D = 2048** (a random direction is nearly orthogonal to
   whatever the decoder reads). ⭐ Accepted — the load-bearing comparison is `x − e` (**162**) vs `x + e`
   (**10**), a **16×** gap at identical magnitude and *identical residual subspace*, and my α dose-response
   (Finding 11a), ⛔ not the Gaussian row.
3. ⚠ **The three controls are additive; the round-trip is replacive.** ⭐ The α sweep is the bridge, and
   the replication ran its own and got the same shape.

## 2026-09-16T21:01:39Z — ⭐ FINDING 13: THE SEPARATION DIAGNOSTIC CONFIRMS "INERT" ON A CONTINUOUS QUANTITY

⚠ My first temperature-grid run recorded `sep=nan` — I read `pairwise_separation`'s output with the wrong
key names. ⛔ Fixed and the whole grid re-run (8 units, ~40 s each) rather than left broken.
Mean pairwise RMS distance between settled states, ⛔ which the charter is explicit is **NOT** falsifier 2
(that is distinct decoded factor tuples) — it is the mechanism diagnostic beside it.

| β | T | settled separation | the LAUNCH's separation | distinct tuples |
|---|---|---|---|---|
| 0 | 0.0 | ⛔ **0.0000** | 0.6504 | 1 |
| 0 | 0.005 | 0.1036 | 0.6504 | 23 |
| 0 | 0.05 | 0.3252 | 0.6504 | 86 |
| 0 | **0.5** | ⭐ **0.6508** | 0.6504 | 463 |
| 1 | 0.0 | 0.0000 | 0.6504 | 1 |
| 1 | 0.005 | 0.1035 | 0.6504 | 23 |
| 1 | 0.05 | 0.3246 | 0.6504 | 94 |
| 1 | **0.5** | ⭐ **0.6509** | 0.6504 | 463 |

- ⭐⭐ **At T = 0.5 the settled ensemble is EXACTLY as spread as the launch ensemble** (0.6508 vs 0.6504,
  a 0.06 % difference). ⇒ the earlier reading — *"the dynamics merely fail to destroy the launch's
  diversity"* — is now confirmed on a **continuous** quantity, ⛔ not only on a discrete tuple count that
  might have saturated.
- ⭐⭐ **β = 0 and β = 1 agree to FOUR DECIMAL PLACES at every temperature** (0.1036/0.1035,
  0.3252/0.3246, 0.6508/0.6509). ⛔ The request energy does not move the chain **at all** — this is far
  stronger than the tuple counts, which could only ever show ties.
- ⭐ At T = 0 the separation is **0.0000**: §E.4's *"mean pairwise separation ≈ 0, identical final energy
  from different launches"* reproduced here from 512 chains.
⇒ ⭐ **The verdict stands and is now better evidenced: dreaming is NOT DEAD (4 distinct valid novel
tuples at the declared T = 0.5), but the dynamics and the request energy are both INERT.**

## 2026-09-16T21:01:39Z — ⭐ FINDING 13: THE SEPARATION DIAGNOSTIC CONFIRMS "INERT", ON A CONTINUOUS QUANTITY

⚠ My first temperature-grid run printed `sep=nan` — I read `pairwise_separation`'s output under the wrong
key names. ⛔ Fixed, and the whole grid **re-run** (8 units, ~40 s each) rather than left broken.
Mean pairwise RMS distance between settled states. ⛔ The charter is explicit that this is **NOT**
falsifier 2 (that is distinct decoded factor tuples); it is the mechanism diagnostic beside it.

| β | T | settled separation | the LAUNCH's separation | distinct tuples |
|---|---|---|---|---|
| 0 | 0.0 | ⛔ **0.000000** | 0.650419 | 1 |
| 0 | 0.005 | 0.1036 | 0.6504 | 23 |
| 0 | 0.05 | 0.3252 | 0.6504 | 86 |
| 0 | **0.5** | ⭐ **0.6508** | 0.6504 | 463 |
| 1 | 0.0 | 0.0000 | 0.6504 | 1 |
| 1 | 0.005 | 0.1035 | 0.6504 | 23 |
| 1 | 0.05 | 0.3246 | 0.6504 | 94 |
| 1 | **0.5** | ⭐ **0.6509** | 0.6504 | 463 |

- ⭐ At **T = 0 the separation is EXACTLY 0.000000** over 512 chains (min pairwise distance 0, one distinct
  tuple). ⭐ §E.4's *"mean pairwise separation ≈ 0, identical final energy from different launches"*,
  reproduced here rather than quoted.
- ⭐⭐ **At T = 0.5 the settled ensemble is EXACTLY as spread as the launch ensemble** — 0.6508 vs
  0.650419, a **0.06 %** difference. ⇒ *"at T = 0.5 the dynamics merely fail to destroy the launch's
  diversity"* is now confirmed on a **continuous** quantity, ⛔ not only on a discrete count that could
  have saturated at 463.
- ⭐⭐ **β = 0 and β = 1 agree to four decimal places at every temperature** (0.1036/0.1035,
  0.3252/0.3246, 0.6508/0.6509). ⛔ The request energy does not move the chain **at all**. ⭐ This is much
  stronger than the tuple counts, which could only ever have shown a tie.
⇒ ⭐ **Verdict unchanged and better evidenced: dreaming is NOT DEAD — 4 distinct valid novel tuples at the
declared T = 0.5 — but the dynamics and the request energy are both INERT.**

## 2026-09-16T21:13:28Z — ⭐⭐ FINDING 14: THE BOUND TRANSFERS TO UNSEEN COMBINATIONS. **COMPOSITION ADDS NO EXTRA PENALTY.**

⛔ Every number above is on rows the store has SEEN, yet the target is about UNSEEN combinations — the
one real gap in the argument. ⛔ I cannot use B to close it. So I built a **SURROGATE held-out cell**:
the **opposite corner** of the factor grid (floor/wall/object hue ∈ {0,1}, scale ∈ {0,1,2},
shape = cube, orientation ∈ {0,1,2}) — ⭐ **exactly 72 rows, 72 distinct tuples**, the same shape and
size as Montero's B. ⛔ Drawn from `split.train_rows` only; ⛔ B never entered, not even as labels.
Every row whose *tuple* lay in the surrogate cell was removed from the store fit (32,768 → 32,762), a
store was fitted from scratch on the remainder, and its address book fitted on the same rows.

| row set | n | DECODER CEILING | store ROUND-TRIP | rel-err | REQUEST-READ (`clu`-style) |
|---|---|---|---|---|---|
| **SEEN** (fitted on) | 512 | 490 (0.9570) | **9** (0.0176) | **0.3524** | **6** (0.0117) |
| ⭐ **SURROGATE** (never seen) | 72 | 47 (0.6528) | **0** (0.0000) | ⭐ **0.3432** | **0** (0.0000) |

- ⭐⭐ **The quantisation error is the SAME SIZE on combinations the store has never seen: rel-err 0.3432
  vs 0.3524.** If anything it is marginally *lower*. ⇒ ⛔ **the store's failure is not a generalisation
  failure.** It tiles the unseen corner exactly as well (or as badly) as the seen one.
- ⭐ The valid counts are consistent with **one rate, not two**: at the seen rate of 0.0176, the expected
  count on 72 surrogate rows is **1.27**, and 0 is an ordinary draw from that (P(0) ≈ 0.28). ⛔ There is
  **no measurable additional composition penalty** on top of the fidelity bound.
- ⇒ ⭐⭐ **Finding 11's byte requirement, measured on seen data, transfers to held-out combinations.**
  That was the last gap in the argument and it is now closed with a control rather than an assumption.
- ⚠ **Two honest caveats.** (i) Only **6** pool rows carried a surrogate tuple, so the store's *feature
  neighbourhood* is barely changed — but that is precisely Montero's recombination-to-element setting,
  where every factor VALUE is seen and only the COMBINATION is not. (ii) ⚠ My surrogate corner is
  **harder for the decoder** than Montero's: ceiling **0.6528** here vs **0.94** on B. ⭐ So the
  surrogate is a *conservative* stand-in for the ceiling, and a *fair* one for the rel-err comparison,
  which is the quantity the finding rests on.

## 2026-09-16T21:21:31Z — ⭐ FINDING 6, CAVEAT CLOSED: the partition gain SURVIVES TRAINING

The trained partition sweep is slow (915 s at K=5, 2,593 s at K=24, and rising with K), so it is still
running and I report it partial rather than waiting. ⭐ **But the two completed units already answer the
question I raised against my own Finding 6:**

| K | M | total B | ⭐ **trained** `val` VALID/512 | untrained (init) | rel-err (trained) |
|---|---|---|---|---|---|
| **5** (shipped partition) | 64 | 598,528 | **4** | 6 | 0.3480 |
| **24** | 43 | 591,680 | ⭐ **21** | (9 at K=16 / 35 at K=32) | 0.3173 |

- ⭐ **5.3× from the partition alone, after training, at 6,848 B UNDER the shipped total budget.** ⛔ The
  20:32Z caveat — *"my own store training is not better than the shipped one, so the init-sweep numbers
  need re-checking"* — is now answered: the *ordering* is what survives, and it survives.
- ⚠ **The caveat's other half still stands and I am not withdrawing it:** my trained K=5 store scores
  **4**/512 while the shipped `gen-read-compose` store scores **10**/512 on the same rows. ⛔ **My store
  training is worse than the shipped one**, so ⛔ the *absolute* numbers in my partition sweep are not
  comparable with the shipped store's, only with each other. ⭐ The 5.3× is a within-sweep ratio and that
  is the only claim I make from it.
- ⭐ ⛔ **And it changes nothing about the verdict.** Finding 11's requirement is ~147,000×; a 5.3× lever
  does not reach a bar that is five orders of magnitude away. ⭐ It is a **mechanism finding worth
  handing on**, ⛔ not a route to the target, and I am recording it as exactly that.

## 2026-09-16T22:12Z — housekeeping, stated because it involved moving things

- ⛔ **Stopped my own trained partition sweep at 3 of 8 units, deliberately.** It had grown to a **33 GB**
  working set on a 63 GB box with **two peer hubs live**. ⭐ My core budget is written down (5 threads);
  my memory share is not, but taking half the machine's RAM for a fourth point on a curve that Finding 11
  has already made moot is not a fair share. ⛔ Stopping a process is not deleting: all three completed
  units (`K5_M64.json`, `K24_M43.json`, `K32_M38.json`) are on disk and the script is resumable.
  **Trained trend, final: K=5 → 4 · K=24 → 21 · K=32 → 31** (of 512) — monotone, caveat closed.
- ⭐ **Confirmed NOTHING of mine is still running** (COMMON §3): no `python.exe` with `hub-composition`
  in its command line. ⚠ Two other python processes are alive and are **not mine** — ⛔ I did not touch
  them.
- ⚠ **Moved `data/` (267 MB) and `pylibs/` (8 MB) OUT of `$OUT`** to
  `/c/Users/PratikJawahar/chlu/hub-composition-env/`. They are **environment, not results** — a public
  dataset and a pip `--target` dir — and `--return-pack` has no exclude, so leaving them would have sent
  **275 MB** home instead of 3.7 MB. ⛔ Nothing deleted; both moved intact.
  - ⭐ **Verified after the move:** `3dshapes.h5` sha256 is still
    `0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785` = the code's own pin, and
    `run_roundtrip.py` re-run end to end under the new paths reproduces **10/512, rel-err 0.3486,
    ceiling 491/512** exactly. `hubenv.sh` updated so the whole thing is reproducible as it stands.
  - ⛔ I still did **not** write into `.claude/data/gen/shapes3d/`. I offered the operator the one-line
    hardlink at 19:12Z and I am not reversing that on my own authority just because it would be
    convenient now.
- ⭐ `git bundle create … hub/composition` (whole branch, ⛔ **not** a range): **3,174,623 B**,
  `git bundle verify` → *"The bundle records a complete history"*, sha256
  `9c6895072e2f7295cdc475565198ae2853e5ea88fd5634269add1d060a04ffd4`.
- ⭐ Branch state: **0 commits, 0 tracked files changed.** ⛔ The frozen scoring harness is untouched.

---

# ⭐ FINAL SECTION — the report (charter §E.3)

**ELAPSED: 3.043 h of a 12 h budget** (`START 2026-09-16T19:10:40Z` → `END 2026-09-16T22:13:14Z`,
measured against `START_UTC.txt` and `date -u`; see `ELAPSED.txt`).
**SCORED RUNS: 0 of 3.**

## ⛔ THE TASK IS **NOT SOLVED**. ⭐ And that is the truthful answer, not a shortfall I am dressing up.
The target asked for **≥ 60 / 72 distinct valid novel tuples**, beating `knn_proto`, with dreaming inside
it. Measured:
- our best is **0.00**, `knn_proto` is **14.33 ± 4.78**, the **decoder ceiling** is **67.67 ± 2.49**;
- and the bar is **~147,000× of store bytes** away — a requirement derived from a power law fitted over a
  **128× measured span** at **R² = 0.9960**, which survives an exponent twice as favourable as any of the
  seven local values I measured (**383×** even then).
⛔ I did not move a bar, did not widen the pre-registered temperature grid, did not soften "valid" or
"novel" — I measured the weaker reading of "valid" too and it **agreed at zero** — and did not spend a
scored run to make a number appear. ⭐ I stopped at **3 h of 12** because the science was finished,
controlled and independently replicated; ⛔ filling the remaining 9 h would have been manufacturing.

## The one thing I would want read if nothing else
⭐⭐ **The obstacle is not composition, addressing, dynamics, temperature, novelty or the metric. It is
that a nearest-codeword read's error is, by construction, precisely the part of the item the store does
not contain — and that is exactly what the classifier reads factor identity from.** At one fixed
rel-err of 0.3486: isotropic noise leaves **382 ± 11** of 512 correct, the same residual *negated* leaves
**162**, *permuted across items* leaves **116 ± 8**, and the store's own output leaves **10**.
⛔ No addressing, partition or temperature touches that. Only a finer tiling does, and the tiling law is
measured.

## Deliverables (charter §A.6)
| item | where |
|---|---|
| PREREG files | `PREREG-hub-composition-1.md` (+ Amendment 1) |
| results ledger of every scored run | `RESULTS-LEDGER.md` — ⭐ **zero**, with both reasons |
| one-page report | `REPORT.md`, and `../hub-composition.md` in protocol §5 format |
| cost line | `REPORT.md` §6 — ⛔ **zero spend, zero GPU, CSF3 NOT RUN** |
| running log | this file, appended throughout, ⛔ never written at the end |
| independent replication | `replication/REPLICATION.md` — 5/7 exact, 1 within noise, ⛔ **1 REFUTED** |
| branch bundle | `hub-composition.bundle` (3,174,623 B, whole branch) |
| elapsed | `ELAPSED.txt` |

## §C floor checks, one line each
- **§C.1 matched-bytes lookup floor:** ⛔ **FAILS for the CLU arm** — `clu` (8/512) sits inside the
  bootstrap CI of its own store's round-trip (10/512, CI [4, 16]); there is no addressing gain to have.
- **§C.2 landscape-removed control:** ⭐ **PASSES** — `store_removed` → 1 distinct tuple, balacc 0.5000.
- **§C.3 "tie is the win" clause:** ⛔ not claimed, and not available: dreaming is inert and the
  valid-novel count is 0.
- **§C.4 settled state on its nearest atom ⇒ a lookup table:** ⭐ **reported as such**, and given its
  economic form — the quantiser needs **19.6×** more bytes than storing all 480,000 train rows verbatim.

## 2026-09-16T22:13:28Z — return pack
`scripts/bundle_hub_workspace.sh --return-pack hub-composition` → **103 files, 3,714,535 B**
→ `.claude/outputs/hub-composition.return.tar.gz` (3,316,014 B on disk),
sha256 `12d5f254885ae9013e8c8ace020947ac9204bb8f02ef0a32052683aa71c416b7`.
Apple-double check: 0 members. Ingest on the primary machine with
`scripts/bundle_hub_workspace.sh --return-ingest <tar>`.
⚠ ⛔ **This tar was written AFTER the final section above, so it contains the log as of 22:13:28Z.** If
anything is appended to `LOG.md` later, the pack must be rebuilt with `--force` or the returned log will
be stale — noting it here because a silently stale return is exactly how a result goes missing.

## 2026-09-16T22:14Z — ⛔ CORRECTION: the return-tar sha256 quoted above is SUPERSEDED
- ⛔ The sha `12d5f254…` in the 22:13:28Z bullet is the **first** pack, which was built one bullet before
  the log finished and was therefore already stale. ⭐ Rebuilt with `--force`; the packed `LOG.md` was
  then verified **byte-identical to disk** with `cmp`.
- ⚠ **A tar cannot contain its own hash**, so ⛔ do not trust any return-tar sha256 printed *inside* this
  log. ⭐ **The authoritative value is the one `--return-pack` prints at pack time**, and it is repeated
  in this hub's closing message to the operator. The **branch bundle's** sha is stable and inside the
  tar, so that one *can* be trusted here:
  `hub-composition.bundle` = `9c6895072e2f7295cdc475565198ae2853e5ea88fd5634269add1d060a04ffd4`.
