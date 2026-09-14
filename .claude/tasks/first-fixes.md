# Task: first-fixes — bug fixes & paper-discrepancy plumbing

- **Agent:** `experiment-engineer`
- **Status:** ✅ **GREENLIT (2026-07-04)** — launch when spawned.
- **Base branch:** `main` · **Work branch:** `agent/experiment-engineer/first-fixes` · **Output:** `.claude/outputs/first-fixes.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` (§7 is the source of these items).
- **Scope guard (added 2026-07-04):** the unit is being renamed CHLU→CLU in papers/docs, but the **code rename is NOT part of this task** — do not touch naming; a dedicated rename task follows after this branch merges. Also: a `results-analyst` may be running concurrently in this repo (read-only + `.claude/` writes); your `uv run` package rebuilds are fine, just don't be surprised by a warm venv.

**Overall acceptance:** the repo builds, `uv run pytest -q` is green (with new smoke tests), the two broken CLI commands work, version reporting is consistent, and the sleep-buffer persistence is configurable — each on atomic, tagged commits. Sections A–C are to implement; Section D is flag-only (no behavior change).

---

## A. Safe bug fixes (implement)

**A1 — `chlu data figure8` and `chlu data sine` are broken.**
`chlu/cli/data_cmd.py:55,83` import `generate_figure8_data` / `generate_sine_data`, which don't exist; the real functions are `generate_figure8` (returns `(T,4)` `[x,y,vx,vy]`) and `generate_sine_waves` (returns `(n,steps,2)` `[x,dx/dt]`). Note the command bodies unpack `q, p = ...` and pass shapes the real generators don't return — so this is **not a pure rename**; reconcile the command to the actual generator output (split state into q/p yourself, or save the raw array to the `--output` `.npz`). Add a smoke test that runs both commands to a temp file and asserts the saved shapes. Verify: `uv run chlu data figure8 --steps 50 --output /tmp/f8.npz` and `... data sine --n-waves 4 --steps 50 --output /tmp/sine.npz`.

**A2 — version strings disagree three ways.** `chlu/__init__.py` = `0.1.0`, `chlu.py --version` = `0.2.3`, `pyproject.toml` = `0.2.4`. Make one source of truth: set `chlu.__version__` to match `pyproject` (or read it via `importlib.metadata.version("chlu")`), and have `--version` print `chlu.__version__`. Verify `uv run chlu --version`.

## B. Consistency / hygiene (implement, minimal)

**B1 — empty `results/`.** Experiments emit plots only; `results/` is always empty though `CLI_GUIDE` advertised `.npz` metrics. Add a small, uniform metrics-save step to each `run_experiment_*` that writes a `results/<exp>_metrics.npz` (e.g. loss history, final energy/target_energy, per-noise MSE for B). Keep it lightweight; don't restructure the experiments. (Coordinate: `results-analyst` will consume these.)

## C. Paper-discrepancy plumbing — sleep-buffer persistence (implement; Head chose "do both & compare")

`training/train.py`'s sleep phase samples from a buffer that is `initialize_random`'d and **never updated** (`# not implemented in-place`), so Exp A/B run *CD with fresh random negatives*, **not** the persistent PCD of paper Algorithm 1 (`ReplayBuffer.add`). `training/train_generative.py` already persists. **Make it configurable, don't just flip it:**
- Add `training.persistent_sleep_buffer: bool` to `config.py` (**default `False` = preserve current A/B behavior**).
- When `True`, persist evolved sleep states back into the buffer at their sampled indices (mirror `train_generative.py`'s `buffer.update`), threading indices/keys correctly through the jitted `sleep_step`.
- Add a smoke test asserting the buffer contents change across epochs iff the flag is on.
- **Do not run the A/B comparison** — that's a later `results-analyst` task; you only build the switch.

## D. Discrepancies to FLAG, not change (report in output, no behavior change)

- **D1** Relativistic mode is not the default for Exp A (`newtonian_identity`) or B (`newtonian_learned`); the paper's Exp II "velocity saturates at c" needs `relativistic` (used in the `finalA` paper run). Document; do not change defaults.
- **D2** Confinement term α present in `PotentialMLP` (`0.05·‖q‖²`) but omitted in `DeepPotentialMLP`/`ConvPotential`. Note; don't change.
- **D3** Lyapunov regularizer may be near-degenerate on a symplectic Jacobian (σ pairs as (σ,1/σ) ⇒ mean(log σ)≈0). Hand to `physics-theorist` for a numerical probe — not a fix here.

---

**Reporting:** write `.claude/outputs/first-fixes.md` per protocol §5 — what changed, commands+observed output for each verification, git footprint (branch + commit hashes + files), and `## Proposed handover updates` for the §7 items you resolved.
