# SPOKE I — Do numbers computed on THIS box match the Mac's? ⭐⭐ The most important open question.
**Core budget: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`.** Read-only w.r.t. the bundle.

Read `_hub_scratch/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`/c/Users/PratikJawahar/chlu/_hub_scratch/reports/I-numerical-divergence.md`**

## Why you exist
The full suite has now run on Windows: **117 failed, 2602 passed, 25 skipped, 42 errors in 9m23s.** Most
failures are harmless platform noise. **Three are not**, and they are all the same shape — a recorded
fingerprint no longer reproduces:

```
tests/test_gen_encoder_unfreeze.py::test_frozen_path_is_bit_identical_to_the_pre_change_fingerprint
E  AssertionError: enc_hash: the FROZEN path moved
   ('094ae9c3941b8fb6…' != '7dcf753fe48ce80f…').
   The unfrozen block must be additive; every downstream phase-1 number rests on the
   frozen path being unchanged.

tests/test_gen_encoder_unfreeze.py::test_the_baseline_decoder_recipe_still_builds_the_same_decoder
E  AssertionError: assert 'a73a12cd18a8…' == '457977e44d7b…'

tests/test_gen_wide_split.py::test_montero_split_is_unchanged_by_the_wide_condition
E  AssertionError: assert 'd27f7af57197…' == '5f1fce25c40c…'
```
and, in the same neighbourhood, two numeric assertions that are **not** float noise:
```
E  assert np.float64(0.8250428728431519) == 0.8244828012941149 ± 1.0e-15     <- differs at 1 part in 1.5e3
E  assert -1.1415797196434596e-08 >= 0.0                                     <- should be non-negative
```
Five more in `tests/test_blocks.py` assert arms "share a **bit-identical** shell" and also fail.

⭐ **If numbers computed here do not match numbers computed on the Mac, then every artefact this machine
produces is incomparable with the existing corpus — and a hub could run for ten hours and produce results
that quietly cannot be pooled.** That is the question you exist to answer.

## ⛔ What you must not do
- ⛔ **Do not "fix" any test, fingerprint or source file.** `tests/` and `chlu/` are pinned content. Every
  finding is reported, never patched.
- ⛔ **No scored runs. No hub. No number that could be quoted as a result.** (Budget: three per *target*,
  across every attempt and machine — see `CONTEXT.md`.)
- ⛔ Do not re-run the whole suite. The log is already at
  `_hub_scratch/reports/_F-suite.log` (444 KB) — **read it, it has the full tracebacks.**

## ⭐ TWO HYPOTHESES ARE ALREADY ELIMINATED. Read this before you plan anything.

**H3 — stale fingerprints — ELIMINATED.** The Mac ran those same test files: **29 passed, 1 skipped.**
They are green there and red here. The fingerprints are correct; this machine disagrees with them.

**H2 — library version drift — ELIMINATED for jax/jaxlib/numpy.** The Mac is on **jax 0.9.0, jaxlib
0.9.0, numpy 2.4.1** — byte-for-byte the same three versions as this venv.

⚠⚠ **BUT ONE VERSION VARIABLE IS *NOT* ELIMINATED, AND IT IS YOUR FIRST EXPERIMENT.**
The Mac runs **Python 3.11.13 / Darwin arm64**. This box runs **Python 3.12.10 / Windows AMD64**.
Two things differ at once — interpreter *and* platform — so "it must be architecture" is an
assumption, not a result. ⭐ **Separate them.** `uv.lock` contains both
`jaxlib-0.9.0-cp311-cp311-win_amd64.whl` and the cp312 wheel, so a Python **3.11** venv is buildable
here:

```bash
uv venv --python 3.11 /c/clu-venv/py311-probe
UV_PROJECT_ENVIRONMENT=/c/clu-venv/py311-probe uv sync --extra gen --group test --frozen
# then run the three fingerprint tests in it
```
- Still red on 3.11/Windows ⇒ **architecture/OS** is the cause. H1 confirmed.
- Green on 3.11/Windows ⇒ **the interpreter minor version** is the cause — a far cheaper fix
  (pin the hub to 3.11) and a completely different conclusion.

⛔ Build that venv **outside** the bundle (`/c/clu-venv/py311-probe`) via `UV_PROJECT_ENVIRONMENT`.
⛔ Do not disturb the working `.venv/` — the verifier depends on it.

## The remaining hypotheses
**H1 — Platform/architecture divergence.** x86-64 (AVX-512, Intel Core Ultra 9) vs Apple Silicon (NEON),
different BLAS, different XLA lowering, different reduction orders. ⇒ *Numbers here genuinely differ.*
**H1b — Interpreter minor version.** Python 3.12.10 here vs 3.11.13 there. See the box above — **test
this first, it is one venv and it is decisive either way.**

⭐ You cannot test the Mac. Your deliverable is evidence that lets the Mac settle anything you cannot.

## What to establish

### 1. Is *loading* bit-exact, even if *computing* is not? ⭐ This decides whether a hub can run here at all.
The bundle ships **frozen pickles** (`encoder.pkl`, `decoder.pkl`, `decoder_v1_s*.pkl`,
`external_classifier.pkl`, `store_s*.pkl`) and **precomputed `features.npy`**, and all 57 sha256s match.
`hub/composition` *loads* those and renders — it does not rebuild them from a recipe. The failing tests
rebuild *from a recipe*. **Those are different operations and the distinction may be the whole answer.**

Establish, with evidence:
- Do the frozen artefacts **load** to bit-identical arrays? (sha256 the raw bytes of a few loaded
  parameter leaves, e.g. `jax.tree_util.tree_leaves` → `np.asarray(...).tobytes()`.)
- Does a **forward pass** through a loaded frozen decoder/classifier reproduce any value recorded in the
  shipped manifests (`bridge_manifest.json`, `store_manifest.json`, `classifier_manifest.json`,
  `winner_hashes.json`, `classifier_gate.json`)? ⭐ **`winner_hashes.json` and `classifier_gate.json` are
  hash/gate records — if either reproduces here, that is strong evidence the *inference* path is stable
  even if the *construction* path is not.** This is the single most valuable check in your brief.

### 2. Pin down the smallest reproducer
For `test_frozen_path_is_bit_identical_to_the_pre_change_fingerprint`, read the test and find what
`enc_hash` actually hashes. Reduce to the smallest snippet that produces a hash, and report the snippet
plus the hash it yields here. The Mac can then run the same snippet and answer H1-vs-H3 immediately.

### 3. Record the exact numerical environment
`jax.__version__`, `jaxlib.__version__`, `numpy.__version__`, `jax.devices()`, `jax.default_backend()`,
`jax.config.jax_enable_x64`, `jax.config.jax_default_matmul_precision`, and any `XLA_FLAGS` /
`JAX_*` env vars. ⚠ **Default matmul precision is the classic cross-platform divergence source** — check
whether the code sets it anywhere (`grep -rn "precision" chlu/core chlu/experiments | head -40`).
Also record `numpy.show_config()` BLAS backend.

### 4. Characterise the size of the divergence
Is it last-bit (1 ULP, benign — reduction-order noise) or structural (the `0.82504` vs `0.82448` case is
**5.6e-4 relative**, which is ~3.7e12 ULPs in float64 and cannot be explained by summation order)?
⭐ **Those two answers have completely different consequences — do not average them into "small".** Report
each failing numeric assertion with its absolute and relative delta.

### 5. Do the same three tests fail *deterministically* here?
Run each of the three fingerprint tests **twice** (`-p no:randomly --no-cov`). If a hash differs between
two runs *on this same machine*, the problem is non-determinism (unseeded RNG, dict ordering, thread
count), not platform — **a completely different and more serious diagnosis.** ⚠ Thread count changes
reduction order: try once at `OMP_NUM_THREADS=4` and once at `1` and say whether the hash moves. If it
does, that alone explains cross-machine divergence and is the finding.

## Also confirm this separately — it is a hard blocker already established
**The entire `chlu` CLI is unimportable on Windows.** `chlu/cli/__init__.py` → `experiment_cmd.py` →
`exp_well_lifecycle` → `exp_anti_erosion` → `exp_placement_probe` → `exp_cluformer_pilot` →
`chlu/training/train_cluformer.py:42: import resource` (Unix-only). So `import chlu.cli`, `import
chlu.chlu` and every `chlu …` console command fail. **`import chlu` itself is fine, and all of
`chlu.experiments.exp_gen_*`, `chlu.core.gen_*`, `chlu.data.gen_splits`, `chlu.data.shapes3d` import
cleanly** — the library works, only the CLI entry point is dead.

⭐ **Already re-confirmed by the hub, do not re-litigate:** `uv run` does **not** help — it uses the same
venv and fails identically (`uv run chlu --version` → `ModuleNotFoundError: No module named 'resource'`).
⭐ **But `uv run python -m chlu.experiments.exp_gen_cell --help` WORKS**, and prints the full surface:
`--dataset --pretext --stage {cell,aggregate,all,encoder,decoder,store,dream,certificate,baselines,compose}
--seed --arm --out --quick --config`. So the generative line is drivable today, bypassing `chlu.cli`.

Answer the practical question: **can the generative work be driven
via `python -m chlu.experiments.exp_gen_cell …` without ever touching `chlu.cli`?** Check whether those
modules' `__main__` paths are self-sufficient. ⚠ Note `chlu data shapes3d --download` is a `chlu.cli`
command, so **the documented dataset-download route is currently dead on this box** — report what the
working alternative is, if any.

## Report shape
Lead with a direct answer to one question: ⭐ **can this machine produce numbers that are comparable with
the Mac's — yes, no, or not yet determined?** If "not yet", say exactly what the Mac must run to settle
it, as a copy-pasteable snippet. Then §1's load-vs-compute finding, the reproducer, the environment
table, the divergence magnitudes, the determinism result, and the CLI blocker.

⭐ Before you finish, confirm no sub-agent of yours is still running.
