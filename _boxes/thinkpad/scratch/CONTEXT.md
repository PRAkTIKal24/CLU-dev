# SHARED CONTEXT — round 3. Read this first, then your own brief.

You are a **spoke agent**. A hub agent scoped your task. You have no prior context. This file plus
`_hub_scratch/<your-brief>.md` is all of it.

## The workspace
```
/c/Users/PratikJawahar/chlu/chlu-bundle-v2           <- $CHLU_ROOT  ⭐ CURRENT WORKSPACE (pin 4125fce7)
/c/Users/PratikJawahar/chlu/chlu-hub-bundle          <- v1 (pin ae907466). ⛔ FALLBACK ONLY: do not work
                                                        in it, do not modify it. Still `status: ready`.
```
Scratch and reports live **beside** it, never inside it: `/c/Users/PratikJawahar/chlu/_hub_scratch/`.

⚠ There is an **older, unrelated git checkout** at `C:\Users\PratikJawahar\OneDrive - Forgis\Desktop\CLU-dev`.
It is **NOT** the workspace. Do not work there, do not fix it, do not commit to it.

The bundle is a self-contained hub workspace for **CLU/CHLU**, an addressable-memory-store research
programme preparing an ICLR submission. It holds `chlu/ scripts/ tests/` from
`git archive ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`, a 2.0 GB `.claude/` of charters and **frozen
scientific artefacts**, `env_hub.sh`, `BUNDLE_MANIFEST.json` (57 sha256-pinned files) and a real `.git/`.

## ⭐ STATUS: the workspace is READY — that work is DONE, do not redo it
`WORKSPACE-READY.json` reads `"status": "ready"`, exit 0, earned with no `SKIP_LOAD` and no `SKIP_TEST`:
57/57 checksums, 57/57 artefacts opened through their real loaders, `chlu` imported from the bundle,
root tree == pin `12831fe1…`, hub write-probe green, `tests/test_hub_workspace.py` → 7 passed, 19 skipped.

**Native Windows works. WSL2 is NOT needed** for the generative line. Two earlier fears proved false:
MSYS2 *does* translate POSIX env-var paths for native Windows processes (`/c/Users/…` arrives as
`C:/Users/…`, colon→semicolon included), and Python 3.12.10 was already installed.

### How to open a working shell — use this verbatim
```bash
export CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-hub-bundle
cd "$CHLU_ROOT"
source ./env_hub.sh
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export PYTHONUTF8=1
export VERIFY_PYTHON="$CHLU_ROOT/.venv/Scripts/python.exe"
```
`PYTHONUTF8=1` and `CLOUDSDK_PYTHON` are already persisted at user scope in the registry. The `PATH` line
is **not** optional: without it `python3` resolves to the base 3.12.10, which has **no jax**.

## Known environment facts — established, do not re-derive
- **`PYTHONUTF8=1` is load-bearing.** This machine's locale encoding is cp1252; the bundle is UTF-8 and
  full of `⛔ ⭐ —`. Bare `Path.read_text()` raises `UnicodeDecodeError` on byte-perfect files without it.
- **`os.access(p, os.X_OK)` is vacuous on Windows** — returns `True` for any readable file, even at mode
  `0o100666`. Any check built on it proves nothing here.
- **`git status` shows 36 permanently "modified" files.** Mode-only (`100755 => 100644`), zero bytes
  differ, caused by `core.filemode=true` written on macOS meeting a filesystem with no exec bit.
- **`core.autocrlf` is pinned `false` repo-locally.** Note `C:\Program Files\Git\etc\gitconfig` sets it
  **true** system-wide; only the user's global override and now the repo-local value hold it off. If it
  ever flips, all 34 `.sh` files get CRLF and die with `bad interpreter`.
- **`chlu/training/train_cluformer.py:42` does a module-scope `import resource`** — Unix-only. The
  cluformer / C3 / streaming line is **dead on native Windows**. It is not on the `import chlu` path.
- **`rsync` is absent and that is fine** — every `rsync` in the tree is `gcloud storage rsync`
  (gcloud's own subcommand) or a printed suggestion. No code path calls the binary.
- ⚠ **CROSS-MACHINE PARITY: NOT YET ESTABLISHED AS COMPARABLE — and the bar is NOT bit-identity.**
  ⛔⛔ **HEAD'S RULING: no hub starts on this box until parity is settled.**
  ⭐ **The test is "does any number we would REPORT move", not "does every float match."** Chasing
  bit-identity across platforms is a known dead end — a different ISA reorders accumulation and a
  float16 cache amplifies it. Scope every check to the actual hazard.
  **The agreed measurement:** run ONE scored cell on both boxes and compare the **composition-B
  aggregate** and the **distinct-tuple count** — *not tensors*. Stable to well inside seed spread
  ⇒ Windows numbers enter the corpus. Moves ⇒ they do not.
  ⭐ Spoke I has already established the groundwork: **37/37 recorded hashes reproduce here**,
  including the 1.9 GB `features.npy` sha256; the LOAD-and-RENDER path is bit-exact. The only genuine
  residual is a **rebuilt k-means++ `store_hash`**, which is over-specified — it moves under pure
  re-batching *on one machine*, while `well_sizes`, `codes_sum` and the byte ledger stay invariant.
- ⛔⛔ **THE HARD CONSTRAINT, whatever parity turns out to be: a Windows hub must NEVER mix
  Mac-produced and Windows-produced artefacts inside ONE comparison.** The rival rows are frozen and
  Mac-produced, so any hub that **rebuilds an input feeding a scored comparison** breaks comparability
  with them — even if its own arithmetic is perfectly self-consistent. ⇒ **Anything needing a
  re-encode is ruled out here.** `hub/refusal` is safest (it scores off existing traces);
  `hub/composition` and `hub/deletion` are the ones most likely to want to rebuild.
- ⭐ **Nothing already banked is in doubt.** The whole corpus was Mac-produced and is internally
  consistent. There is no retroactive effect.
- **Python version is ELIMINATED as a variable — measured, not assumed.** Spoke I built a 3.11.16
  venv on this box: **every line of the reproducer is byte-identical to 3.12.10**, including
  `store_hash`. Pinning the hub to 3.11 would change nothing. (The Head prefers matching the Mac's
  3.11.13 anyway, to remove a free variable — fine, but it is not a fix.)
- ⭐ **THE `chlu` CLI WORKS AGAIN AT THE v2 PIN.** `import chlu.cli`, `import chlu.chlu` and
  `python -m chlu.chlu --help` (~60 commands) all succeed; `import resource` is now lazy inside a
  non-Linux fallback branch (`train_cluformer.py:829`). ⛔ **The old "CLI is dead" workaround is
  OBSOLETE at v2.** It still applies to v1.
- ⭐ **`anchors_checked` IS NON-ZERO ON v2** — `2 / 4 / 4 / 1` for the four docs with registered
  anchors (8 others legitimately have none). The `.as_posix()` fix works on Windows; the
  charter-truncation guard is live again.
- ⚠ **v2 is `status: NOT-READY` for exactly one reason, and it is not the bundle.**
  `test_hub_workspace.py::test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale` fails
  because its `_WindowsFlavourPath` **mock** uses `read_bytes().decode()`, skipping the
  universal-newline translation real `Path.read_text` performs — it counts 11 CR bytes as 11 extra
  characters. The test's own message blames cp1252 and is **wrong** (cp1252 cannot decode 0x90 at
  all, and the delta would be +6, not +11). 146/146 checksums, 146/146 loaders, import-under-root,
  pin tree and the write probe are all **green**. ⛔ Do not patch the test — it is manifest-pinned.
- ⚠ **`DOC_ANCHORS` still requires the OLD pin string `ae907466…`.** v2's charter passes only because
  it retains a `### (superseded) ae907466…` line, so the guard whose purpose is *"a charter that does
  not carry it cannot pin a hub"* is asserting a **superseded** pin, not v2's real `4125fce7`.
- **Two Windows bugs in the pinned verifier are being fixed upstream by the humans** — `_load_doc()`
  keys `DOC_ANCHORS` off a backslash path so check #3 silently inspects **zero anchors**, and bare
  `read_text()` needs `encoding="utf-8"`. ⛔ **Do not fix them here** — the files are sha256-pinned.

## ⭐ THE OPERATING STRUCTURE (Head, 2026-09-17)
**Advisor** (Fable, once per full pass, ideation only → `DIRECTIVE.md`) · **Coordinator / hub** (Opus,
stages, monitors, compiles, small decisions autonomously, escalates only re-ideation triggers) ·
**Execution hubs / spokes** (Opus, 12 h, the science). ⭐ Hub and spokes ALWAYS on Opus; every sub-agent
`model: "opus"`. See `COORDINATOR-LOOP.md` and `ADVISOR-ROLE.md`.
⭐ **§5.1 falsifier CLEAR on this box** (measured 2026-09-17): `|cached| ≥ 2^-4` band max ULP = 1.000, zero
violators; seed-0 no-extension scored runs are sound. Enforced gate still aborts (51 > 1); fraction bound
has 31 elements of headroom.

## ⭐ RULINGS FROM THE HEAD — these are decisions, not suggestions
- ⭐ **AMENDMENT 1 (compute placement per machine) — BINDING NOW.** **CPU jobs run LOCALLY on this
  box. GPU jobs run on GCloud.** The charter's "GCloud CPU is your
  default, always" was written for a 16 GB shared laptop where hub CPU work blocked the Head. On this
  dedicated 16-core / 63 GB machine, local CPU is free. **This supersedes the charter on that point.**
- ⭐ **RUN-DURATION POLICY (Head): an autonomous agent works until it has solved its assigned task
  TRUTHFULLY, or until 12 HOURS OF TRUE WALL-CLOCK TIME — whichever comes first.**
  ⛔ "Truthfully" is the operative word: a task is NOT solved because the time is nearly up, because a
  partial result looks presentable, or because a check was lowered until it passed. ⭐ **Stopping at 12 h
  with an honest "not solved, here is exactly how far I got and what blocked me" is a SUCCESS.**
  Manufacturing a result to fill the window is the one unrecoverable failure.
  ⚠ **True wall clock** = elapsed real time from launch, measured against a recorded UTC start stamp.
  ⛔ Not model-perceived time, not summed tool time, not time-minus-waiting. Record the start stamp
  first and check it against the clock, because an agent's sense of elapsed time is unreliable.
  ⚠ This 12 h budget does **NOT** relax the charter's §A.2 cap of **three scored runs per target across
  every attempt and every machine**. More hours do not buy more scored runs.
- ⭐ **`hub/refusal` IS AUTHORISED to start on this box** (Head, on v2). All 92 of its required files are
  present natively in v2. ⚠ It must not be launched until the machine pre-flight is done — see
  `PREFLIGHT-12H.md`. ⛔ Other hubs remain unauthorised.
- **GPUs sparingly — never for a run whose best case is a supporting claim.** UNCHANGED.
- **CSF3 is unreachable by any hub (2FA).** Emit the submit lines, report **NOT RUN**. UNCHANGED.
- **The sub-agent cap of 3 may be raised on this box** (it was that laptop's thermal limit). ⭐ But you
  **must confirm no sub-agent is still running before you finish.**
- ⛔⛔ **SCORED-RUN BUDGET — assume the STRICTER reading.** Charter §A.2 caps scored runs at three
  *per hub*. If a relaunched hub got a fresh three, the cap would mean nothing and relaunching would
  become laundering. ⭐ **Treat the budget as THREE PER TARGET, across every attempt and every machine.**
  The Advisor has yet to rule; until they do, the strict reading binds. ⚠ Two machines pursuing one
  target **cannot see each other's count** — a scored run you spend is one the real hub cannot have,
  and it cannot know you spent it.
- **There are FOUR hub targets, not three.** The fourth is **`hub/generation-table`** — target 1, the
  headline. ⚠ **It is not in this bundle**, which is why nothing here mentions it. Do not infer it is
  out of scope; it is merely absent.
- **AppleDouble cleanup is APPROVED** — scope and conditions in Spoke E's brief.
- ⚠ **`hub/refusal`'s missing artefacts: the supplement has ARRIVED and is checksum-verified** at
  `/c/Users/PratikJawahar/chlu/hub-refusal.supplement.tar.gz`
  (`4cf00c6262f1f422ab3bdf423198e5a5375b36bda23eafe455ad29499dbfba55`, 216,089,526 bytes, 93 members,
  zero AppleDouble). **Spoke H applies it — artefacts only, docs skipped.** ⛔ No other spoke may apply,
  extract or move it. Until H reports success, `hub/refusal` must NOT be started.
  ⭐ Its charter's claim that it is "scored off artefacts that already exist" was false only *while the
  artefacts were missing*. Once H lands them the claim is true again — it needs no correction.
- ⚠ **The supplement's `autonomous-hub-charter.md` (7656 B) is NOT the bundle's (6610 B).** The diff is
  **Amendment 1**, above. ⛔ It is still skipped by H, purely because `BUNDLE_MANIFEST.json` pins the
  6610-byte version and the supplement ships no updated manifest — installing it would drop the
  workspace out of `ready`. **Amendment 1 binds you now regardless of which file is on disk.**
  A second, small supplement carrying the fixed verifier, an updated manifest and a
  `--supplement-apply` flag is being built upstream.
- ⛔⛔ **`.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` is QUARANTINED — never quote it.** It is
  the SUPERSEDED broken-split file. ⚠ **"Quarantined" means never cite it as evidence. It IS one of the 57
  sha256-pinned manifest files — do NOT move, rename or delete it.**

## Rules that bind every spoke agent
- ⛔ **Never edit, move or delete any file listed in `BUNDLE_MANIFEST.json`.** They are sha256-pinned and
  several are frozen scientific assets other machines' results already depend on.
- ⛔ **Never weaken, skip, `--force` or edit a script to make a check pass.** `SKIP_LOAD=1` / `SKIP_TEST=1`
  may isolate a failure; the script itself says they are **⛔ NOT a pass**.
- ⛔ **Do not modify `chlu/`, `tests/` or `.claude/` content.** If code looks wrong it is almost certainly
  the environment — **report it, do not patch it.**
- ⛔ **No `git commit`, `git push`, `git reset --hard`, `git checkout --`** unless your brief names it.
- ⛔ **No CUDA/GPU installs. No GPU runs. No cloud spend.**
- ⭐ **Respect your core budget.** Other spokes are running in parallel on the same 16 cores. JAX's CPU
  backend and BLAS will each try to take all of them; pin `OMP_NUM_THREADS`/`MKL_NUM_THREADS` as your
  brief specifies. An unbound numpy/sklearn oversubscribes and runs **slower**, not faster.
- ⭐ **Capture FULL output for anything that fails.** A `tail` throws away the part that matters.

## Deliverable
Write your report to the exact path your brief names, under `_hub_scratch/reports/`. Markdown.
**What I did → What works → What is broken (full output) → What I could not resolve honestly →
Numbered open questions.** The hub reads only your report file.
⭐ "This machine cannot do X because Y" is a complete and valuable answer. Never manufacture a green tick.
