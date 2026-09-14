# TWO-MACHINE PLAN — Mac + ThinkPad (WSL2)

**Ratified by the Head 2026-09-12. Supersedes the options in `two-machine-setup.md` (that file is now history).**
⛔ **Execution is GATED.** Phase 1 may start now. Phases 2–4 wait for: (a) the current wave's spokes to finish, (b) the Head's lead bringing `clu` and `clu-dev` up to date, (c) the Head's explicit go.

---

## 1. The decided architecture

**Both machines hold a full working copy. Either can take either role. Push-pull on every switch.**

Asymmetric by *preference*, symmetric by *capability*:

| | Mac (M1 Pro, 16 GB) | ThinkPad (Ultra 9 285H, 64 GB, WSL2) |
|---|---|---|
| role | lean UI · **CSF3 gateway** | **local heavy compute** |
| GlobalProtect VPN | ✅ only here | ❌ never |
| CSF3 submit/rsync | ✅ only here | ❌ never |
| local sweeps / suites | light | ✅ default |
| working copy | ✅ full | ✅ full |

Threads are **disconnected and non-overlapping** across machines (Head's ruling) — so the two copies rarely diverge in the same file, and push-pull is sufficient.

### Sync mechanism
- **Code** → `clu-dev/main`, ordinary git. Already works.
- **Brain** → a new **orphan branch `brain` on `clu-dev`**, whose root is the *contents* of `.claude/`, cloned into `.claude/` as a **nested repo** (its own `.git`; the parent's `.gitignore` already hides `.claude/**`, so the parent never sees it).
- **`internal-state`** snapshots stay as-is, as a second backup. Not retired.

```
CHLU/            <- clu-dev, branch main    (code)
├── .git/
└── .claude/     <- clu-dev, branch brain   (own .git)
    └── .gitignore   scratch/ data/ *.npy *.npz *.pkl *.eqx .DS_Store
```

`clu` (public-at-acceptance) never needs a gitignore toggle. Its tree stays permanently clean. The leak guard in `push_repo.sh` is never routed around.

### Brain exclusion rule (ratified)
`scratch/` · `data/` · `*.npy` · `*.npz` · `*.pkl` · `*.eqx` · `.DS_Store`
Takes the brain from **8.5 GB → ~275 MB**. Without it the first push fails outright: GitHub hard-rejects files >100 MB and `gen-bridge-wide` holds four `.npy` at 1.8 GB each.

### Network & access
- **Tailscale on both machines** — the transport for SSH and any remote control. On the ThinkPad install it **inside WSL2**, so WSL2 gets its own tailnet address and no Windows→WSL port-proxy is needed.
- **GlobalProtect only on the Mac.** Since CSF3 work never leaves the Mac, GlobalProtect and Tailscale only ever coexist there. ⚠ Still verify: full-tunnel GlobalProtect can capture all traffic and break Tailscale on the Mac while connected. If it does → split-tunnel, or raise the VPN only for the CSF3 rsync window.
- **Remote control:** `ssh` + `tmux` is the primary and is *strictly better* for agent work (see §2). GUI remote control as a rarely-needed extra: RDP over Tailscale if Windows is **Pro**; RustDesk if **Home**. Mac side: enable Screen Sharing, reachable over Tailscale.

---

## 2. Known hazard: Zed ACP threads die on disconnect

The ACP agent **does** run on the remote host in a Zed SSH project, but Zed interrupts a running thread when the client disconnects rather than detaching it (open upstream bug, no fix planned). Zed also does not discover a remote-installed Claude Code and installs its own.

Consequences, in order of importance:
1. **Long unattended work should run under `tmux` on the host, not in a Zed ACP thread.** A `tmux`-hosted session survives lid-close and wifi loss; a Zed ACP thread does not. This is the real reason to want `ssh`+`tmux`, independent of remote control.
2. Spokes must keep launching compute as **detached background jobs** (already the practice) — a dropped connection then loses the *thread*, not the *run*.
3. `handover_context.md` discipline already covers rebuilding a lost thread. No change needed.
4. Because both machines hold full copies, a flaky remote session is never a work stoppage — fall back to working locally.

---

## 3. Porting audit — what breaks on WSL2

### ✅ Clean
- **Zero hardcoded `/Users/...` paths in shipped code** (`chlu/`, `tests/`, `pyproject.toml`). Verified.
- The only two platform branches are both correct: `sys.platform == "darwin"` for `ru_maxrss` units (`train_cluformer.py:698`), and the Linux-gated `cuda` extra.
- Zed settings and keybindings already sync via the Head's account.

### ⛔ `AGENT_PROTOCOL.md` §4 "Environment facts (this machine)" is Mac-specific and will mislead every spoke on the ThinkPad

| line | why it breaks | fix |
|---|---|---|
| `PYTHONPATH=<wt> /Users/user/Desktop/CHLU/.venv/bin/python` — the **preferred** worktree pattern | hardcoded Mac path; every worktree spoke uses it | parameterize as `$CHLU_ROOT/.venv/bin/python`, define `CHLU_ROOT` per machine |
| "JAX cold-start is pathologically slow here (~20+ min)" | an M1 characteristic; x86 Linux will differ | **re-measure on the ThinkPad**, record both numbers per machine. A wrong budget also means a real hang gets excused as a slow import |
| "No `timeout` binary on this macOS" | **false on WSL2** — coreutils has it | make it a per-machine note; spokes on Linux should use it |
| `uv sync` guidance | ⚠ **new trap:** the `cuda` extra is Linux-gated, so on WSL2 `uv sync --extra cuda` will actually resolve `jax[cuda12]` — ~3 GB of CUDA wheels on a machine with **no NVIDIA GPU** (Intel Arc iGPU), and JAX may fail at import trying to init CUDA | document **plain `uv sync` only on the ThinkPad; never `--extra cuda`** |
| expected suite count | extras resolve differently per platform (the known `importorskip("pandas")` −7) | re-baseline the suite count on the ThinkPad once, record it, so N−k is not misread as a regression |

### ⛔ `AGENT_PROTOCOL.md` §3 git discipline is incomplete — it predates the fork topology
- It knows only `origin`. **`clu` and `clu-dev` are never mentioned**, though `clu-dev` is the real daily base. The topology lives in `csf3-operations.md`, `repo_fork_plan.md` and `fork_push_runbook.sh` — none of which a spoke is told to read. A Windows hub would be blind to it.
- Per the Head: the ThinkPad gets **only `clu-dev`** as a remote, named `clu-dev`. No `origin`, no `clu`. Releases stay Mac-only.
- Therefore §3.5's "`origin/main` is stale" warning is *inapplicable* on the ThinkPad (no `origin` there) while the rebase-onto-local-`main` rule still holds. Must be reworded so it is true on both machines.
- **New rule needed — the nested brain repo.** `.claude/` becoming its own repo is a protocol change every agent must know:
  - the parent's `git status` will **never** show brain changes;
  - brain edits need their own commit/push cycle inside `.claude/`;
  - ⛔ **forbid `git clean -xfd` (and any `-x`) in the parent** — it would delete the nested repo and the brain with it. Add to §3.6 Forbidden.
- Commit trailer in §3.4 (`Co-Authored-By: Claude <noreply@anthropic.com>`) matches what is actually in use. No change.

### Datasets
`.claude/data/` (11 GB: `c2w10-metro`, `c2w10-streams`, `c3-camels`, `c3-ncmapss`, `gen`, `hepa`) is **not ported** — lead hubs re-fetch ad hoc. Shipped fetchers exist (`chlu/cli/data_cmd.py`, `chlu/data/industrial/*`, `chlu/data/enwik8.py`).
⚠ Gap: the CAMELS fetch lives in `.claude/scratch/c3-trackb-tripwire/fetch_camels.sh`, and scratch is not ported. Acceptable — CAMELS is C3/CSF3 work, which stays on the Mac. Note it rather than fix it.

### Not ported
Scratch (9 GB), the seven venvs, the agent worktrees (`CHLU-merge`, `CHLU-wt2`, `CHLU-gen-cert`, `CHLU-gc-base`). All regenerable, all local scaffolding.

---

## 4. Skills (Head ruling 2026-09-12)

**User-level copies are canonical, verbatim, with NO CLU specifics.** The project-level `hub-handover` is
CHLU-specialized and **bloated — the Head never wants to use it.**

Actions:
- ⛔ **Delete `.claude/skills/hub-handover`** (the CHLU-specific variant) and the empty `.claude/skills/hub-handover 2`.
- Seed `.claude/skills/` with exact copies of all five `~/.claude/skills/` entries: `create-spokes`, `full-sitrep`,
  `hub-handover`, `kill-stale-proc`, `quick-sitrep`. Do not re-specialize them.
- Symlink `~/.claude/skills/<name>` -> the brain copy on each machine, so they version and sync with the brain
  and reach the ThinkPad automatically. Content stays exactly the user-level content; the brain is only the
  transport and the backup. WSL symlinks behave like Linux -- identical on both machines.
- ⚠ Consequence, accepted by the Head: the Hub resume command word becomes the generic
  *"Act as my lead Hub agent. Read `.claude/handover_context.md` and continue from the latest Running Log entry."*
  The CHLU-specific wording is retired along with the bloated skill.

## 5. Phases

### Phase 1 — prep (⭐ can start NOW, touches no repo, no remote, no running wave)
1. ThinkPad: `wsl --install` (Ubuntu). Confirm WSL2, not WSL1.
2. Inside WSL2: `git`, `uv`, `tmux`, `build-essential`, `curl`.
3. `git config --global core.autocrlf false` ⚠ **CRLF silently breaks every bash script the moment it reaches CSF3.** And `core.longpaths true`.
4. Tailscale on both; **inside WSL2** on the ThinkPad. Verify ping both directions.
5. New SSH key **inside WSL2**; add to GitHub for `clu-dev` access.
6. `sshd` inside WSL2. Windows power settings: stay awake on lid-close while plugged in.
7. Zed on Windows; confirm account sync carries settings + keybindings.
8. ⭐ **De-risk the remote path on a throwaway repo first** — Zed SSH from the Mac into WSL2 on a scratch project, run one ACP thread, pull the network, see what actually happens. Do this *before* betting CHLU on it.
9. Confirm Windows edition (Pro vs Home) → picks RDP vs RustDesk.
10. Mac: check whether GlobalProtect breaks Tailscale while connected.

### Phase 2 — gated on the Head's go (wave done + `clu`/`clu-dev` current)
1. Head's lead brings `clu` and `clu-dev` up to date. `main` is 5 commits unpushed; `clu` 39 behind.
2. Re-verify all branch positions (they move daily).
3. Create the `brain` orphan branch on `clu-dev` with the §1 exclusion rule; push.
4. Convert the Mac's `.claude/` in place — `git init` + remote + fetch + `reset --mixed` (does **not** touch files). ⚠ Never plain-checkout the brain branch over a populated `.claude/`.
5. Skills: seed `.claude/skills/` from `~/.claude/skills/` (user-level wins), delete the duplicate, symlink back.
6. Verify from a clean angle that the Mac's brain is intact and the parent repo still sees no `.claude` changes.

### Phase 3 — ThinkPad bring-up
1. Clone `clu-dev/main` into WSL2 (`~/CHLU`), remote named `clu-dev` **only**.
2. Clone `clu-dev` branch `brain` into `~/CHLU/.claude`.
3. Symlink skills into `~/.claude/skills/`.
4. `uv sync` — **plain, never `--extra cuda`**.
5. Re-measure: JAX cold-start time, suite count, `timeout` availability. Record as the ThinkPad's §4 numbers.
6. Run the full suite. Compare to the Mac baseline; explain any delta before proceeding.

### Phase 4 — protocol update (the Head's #9: stop the Windows hub inheriting stale practice)
Rewrite `AGENT_PROTOCOL.md` §3 and §4 per the §3 findings above:
- §4 becomes **per-machine**: a Mac block and a ThinkPad block, `CHLU_ROOT` parameterized, both cold-start numbers, both suite counts, `timeout` availability per machine, the `--extra cuda` prohibition.
- §3 gains the three-remote topology (and which remotes exist on which machine), the reworded stale-`origin` rule, the nested-brain-repo rules, and `git clean -x` in Forbidden.
- Add the switching discipline: **leaving a machine, commit and push both repos; arriving, pull both.**

---

## 6. Open risks
1. **GlobalProtect vs Tailscale on the Mac** — untested. Phase 1 item 10.
2. **Zed remote SSH quality** — known-rough. Phase 1 item 8 de-risks it; both machines holding full copies means it is never a work stoppage.
3. **First brain push size** — verify the exclusion rule bites *before* pushing, or the 1.8 GB `.npy` files bounce off GitHub's 100 MB limit.
4. **Windows sleeping mid-run** — power settings must be right, or the "stationary server" naps during an unattended sweep.
5. **Divergent brain edits** — low, since threads are non-overlapping, but markdown conflicts are the one thing that can bite. Push before switching, always.
