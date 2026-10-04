# SPOKE A — Python / uv / JAX toolchain bring-up  **(the only agent that installs things)**

Read `.temp_cl/CONTEXT.md` first.
Report to: **`.temp_cl/reports/A-toolchain.md`**

## Your job
Make `import chlu` and `import jax` work on this machine, CPU-only, and prove it with a real test run.
You are the critical path. Agents B, C and D are read-only and running alongside you.

## Your write surface — nothing outside this
- System-level installs (Python, `uv`, packages). Fine.
- The virtualenv (see "Where the venv goes" below).
- `.temp_cl/reports/A-toolchain.md` and scratch under `.temp_cl/`.
- ⛔ **Zero tracked files.** In particular do not "fix" `pyproject.toml` or regenerate `uv.lock`.
  If `uv sync` wants to change `uv.lock`, use `uv sync --frozen` / `--locked` and **report the conflict**
  rather than letting the lock be rewritten.

## Constraints from `pyproject.toml` (already read for you — do not change it)
- `requires-python = ">=3.10, <3.13"` → **install Python 3.12** (3.13 is out of range).
- Core deps: numpy, scipy, **jax, jaxlib**, equinox, optax, diffrax, matplotlib, tqdm, rich, art,
  scikit-learn, pyyaml.
- Extras exist: `eval` (pandas/pyarrow/pyreadr), `cafe` (lifelines), `gen` (h5py), `cuda` (**Linux-only
  marker — never install**).
- Dependency groups: `dev`, `test` (pytest, pytest-cov, pytest-xdist).
- `[tool.pytest.ini_options] addopts = "--cov=chlu --cov-report=xml --cov-report=term"` — every pytest run
  is a coverage run by default. Use `-p no:cacheprovider --no-cov` when you just want a fast signal.

## Where the venv goes — decide and report, do not agonise
The repo is inside OneDrive with Files-On-Demand. A JAX venv is multiple GB and OneDrive will try to sync
it. **Default to putting the venv OUTSIDE OneDrive:**
```bash
export UV_PROJECT_ENVIRONMENT="/c/clu-venv/CLU-dev"
```
(or the PowerShell equivalent `$env:UV_PROJECT_ENVIRONMENT`). If you do this, **say so loudly in your
report and give the exact activation line** the human will need, because it is non-obvious and every later
session must set it. If you find a concrete reason an out-of-tree venv breaks something, use the in-repo
`.venv/` instead (it is gitignored) and explain why.

## Steps
1. Install Python 3.12 (winget / python.org installer / `uv python install 3.12` — your call; say which).
   Verify it is a **real** interpreter, not the Store stub: `python -c "import sys; print(sys.executable)"`.
2. Install `uv`. Verify `uv --version`.
3. `uv sync --extra gen --group test` (or `pip install -e .` if `uv` proves unworkable — say which and why).
   ⛔ Do not pass `--extra cuda`.
4. Prove JAX is CPU and healthy:
   ```
   python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__); print(jax.devices()); import jax.numpy as jnp; print(jnp.ones((3,3)).sum())"
   ```
   Record the exact `jax.devices()` output and any platform warnings verbatim.
5. Prove the package imports: `python -c "import chlu; print(chlu.__file__)"`.
6. Run a **targeted** test, not the whole suite:
   `python -m pytest tests/ -x -q --no-cov -k "not slow" --maxfail=5` — and if that is too broad, pick 2–3
   small test files and run those. **Report the exact selection you ran, pass/fail counts, and the full
   text of the first 3 failures if any.** Do not chase failures into the source — report them.
7. **Unblock `gcloud`**, which is installed-but-broken because it cannot find Python. Set
   `CLOUDSDK_PYTHON` to your new interpreter and check:
   `gcloud --version`, `gcloud auth list`, `gcloud config list`.
   Report the active account and project. ⛔ Do not run anything that costs money, creates resources, or
   changes cloud state. Read-only gcloud calls only.
8. Report how you made each env var persist (user-level `setx` / PowerShell profile / `~/.bashrc`) — or
   that you did not, and it is session-only.

## Specifically watch for and report
- **MSYS path mangling**: Git Bash may rewrite `/c/Users/...` into Windows form when handing arguments to
  native `.exe` binaries, so a POSIX path exported in bash can reach Python unusable. The hub explicitly
  wants to know whether this bites. If you see it, show the exact before/after.
- Any dependency that has no Windows wheel and tries to build from source.
- Whether `uv sync` respected `uv.lock` or wanted to rewrite it.
- Wall-clock time for the sync, and install size.

## Do not
- Do not install WSL. (That is a decision for the human; Spoke B evaluates it.)
- Do not run the full 117-file test suite to completion if it is slow — a targeted green signal is the ask.
- Do not attempt to fix any test failure.
