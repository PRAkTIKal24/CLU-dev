# hub/composition-2 — RUNNING LOG (iteration 2)
**Agent** `autonomous-hub` · **branch** `hub/composition-2` · **worktree** `/c/Users/PratikJawahar/chlu/CHLU-hub-composition-2`
**Base** `945a3e9` = `base/iter2` (pin tree `345244bb…` + the five reference-read files + the §2.2(b) gate patch)
**Slug** `hub-composition` · **this iteration writes ONLY under** `.claude/outputs/hub-composition/iter2/**`
⛔ The predecessor's files in `.claude/outputs/hub-composition/` are INHERITED FACT. Nothing there is
deleted, moved or overwritten (§0.1, addendum).

**START_UTC** see `START_UTC.txt` — `2026-09-17T10:49:30Z` · **budget 12 h true wall clock** ⇒ hard stop
`2026-09-17T22:49:30Z`. Elapsed is measured against that file and `date -u`, ⛔ never model-perceived time.
**Threads: 4** (`OMP/MKL/OPENBLAS_NUM_THREADS=4`) — four hubs share 16 cores. **Sub-agents ≤ 2, inside the 4.**

---

## 2026-09-17T10:49:30Z — BOOT

### The override, quoted as §0 of LAUNCH-COMMON requires
> **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and
> `hub/generation-table` on v2 despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.**
> The failure is in the test's `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()` calls
> `read_bytes().decode()`, which does **not** fold universal newlines, while real `Path.read_text()` opens
> in text mode and does. On Windows the fixture is written by a real `write_text()`, which expands
> `\n → \r\n`, so the mock counts two characters per newline and `chars` exceeds `len(text)` by exactly the
> newline count. **Nothing to do with encoding.** It passes on macOS only because macOS does not expand
> newlines. The mock is fixed on the Mac and ships in the next cut; the UTF-8 behaviour it guards is
> unaffected and still enforced by the shipped code's explicit `encoding="utf-8"`.

⛔ This override covers exactly that one test. ⛔ It is not a licence to wave through any other red check.
⛔ `SKIP_LOAD=1` / `SKIP_TEST=1` are not a pass.

- Boot order followed: `LAUNCH-COMMON.md` → `LAUNCH-hub-composition-2.md` → `autonomous-hub-charter.md` →
  `AGENT_PROTOCOL.md` → `handover_context.md` §7 → `.claude/tasks/hub-composition.md` (as amended by R2).
- Env verified: `CHLU_ROOT=…/chlu-bundle-v2`, every `CHLU_*` path contains `chlu-bundle-v2`;
  `chlu.__file__` resolves under the root (import trap clear); Python 3.11.16, jax 0.9.0, numpy 2.4.1;
  `PYTHONUTF8=1` set. Env script: `iter2/hubenv2.sh`.
- Worktree checked: `hub/composition-2` at `945a3e9`, tree clean, parent `8e36bc9` = the pin tree.

### My bar, restated under R2 — I do NOT work to the retired one
⛔ RETIRED: "≥ 60/72 distinct valid novel tuples".
⭐ **LIVE BAR:** distinct valid novel tuples on **B** at matched bytes, **win-or-tie vs `knn_proto`
under R1**, with the capacity law `rel-err = e^1.346 · bytes^−0.1651` (R² 0.9960) and the decoder ceiling
**67.67 ± 2.49** reported beside it.
R1: *tie* := paired 95 % CI on Δ(arm − rival) includes 0, **or** |Δ| ≤ m = 0.020. *win* := Δ > 0, CI
excludes 0. A tie is banked, then I keep pushing for a win.

### Inherited facts I will NOT re-derive (§E.4 + predecessor + addendum)
- Identity on B: `distinct_valid_novel = 72 × exact_match` (all 72 targets distinct, none in any train row).
  ⇒ the headline column needs **no new instrument**. The predecessor's VALID∧NOVEL instrument reproduced
  24/24 frozen harness rows and is **reused, not rebuilt**.
- Nulls, inherited, ⛔ not rediscovered: learned placement; codebook loses 90.5 % scale / 100 % orientation;
  density-over-atoms descent 0.685 → 0.211; dreaming inert **on the argmax read**; the argmax read
  discards the span.
- ⭐⭐ **The predecessor's headline "0 tuples at 147,000× bytes" is a law about the ARGMAX QUANTISER's
  self-aligned residual, NOT about the store.** The same store under the **LINEAR** read is reported at
  **35–50 tuples at the shipped bytes**. That reframing is the most important thing I inherit and it is
  where I start. ⛔ The argmax read is not my object.
- ⛔ Not the PCA-affine dictionary (R4). ⛔ Not the trajectory encoder (R5).

### Budget facts
- Scored runs: **0/3 spent on this target, on any machine.** This iteration: **at most ONE, seed 0 only**,
  and only after a PREREG naming object/bar/controls AND a DEV number that clears the bar on non-held-out
  corners with §C controls passing.
- ⭐ It may cost **nothing**: if my object is identical to `hub/generation-table-2`'s, my readout comes from
  ITS scored cell (per-render tuples are persisted for every arm). I spend my 1 **only** if my object
  genuinely differs, and I say why in the ledger **before** running.
- Gate: seed 0's cache is `scope="full"` (480,000 rows) and cannot be extended ⇒ seed 0 scores here.
  ⛔ Shortfall > 0 ⇒ STOP and report. ⛔ Never encode, never raise `ULP_BOUND`/`FRAC_BOUND`.

### Plan (first three units)
- **U1** — verify the object: read the five reference-read files on `base/iter2`, confirm `clu_wide_vector`
  is the LINEAR read, confirm K=32 / M=26 / rank-16 pair head / 469,504 B and the byte ledger.
- **U2** — reproduce the "35–50 tuples at shipped bytes" claim myself on `val`/dev corners. ⛔ B untouched.
- **U3** — decide object identity vs the generation-table reference object; write PREREG.
