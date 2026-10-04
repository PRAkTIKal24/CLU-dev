# SPOKE H — Apply the refusal supplement, artefacts only

**Agent:** Spoke H · **Workspace:** `C:\Users\PratikJawahar\chlu\chlu-hub-bundle`
**Ran:** 2026-09-16, ~17:20–17:55 UTC · **Sub-agents launched:** none (see §8)

---

## ⭐ HEADLINE — the two questions, answered first

| Question | Answer |
|---|---|
| **Is `hub/refusal` unblocked on this machine?** | ⭐ **YES.** Every artefact its charter names is present, byte-perfect against `SUPPLEMENT_MANIFEST.json`, and **opens through `numpy.load` / `json.load` on this box**. 78/78 `.npz` opened, all arrays materialised, all finite. |
| **Is the workspace still `status: ready`?** | ⭐ **YES.** `scripts/verify_hub_workspace.sh` exit **0**, `status=ready`, `n_checksum_mismatch: 0`, `n_missing: 0`, `n_load_failed: 0`, `skip_load: false`, targeted test **7 passed, 19 skipped**. |

**Nothing was broken.** The 57 sha256-pinned files are byte-identical before and after (`diff` empty).
The root tree still equals the pin `12831fe1be17d04da027dcbb6e45b4dff6f1ef95`.
⛔ None of the three `.md` docs were extracted — the charter on disk is still the pinned 6610 B / `41b6fcbe…`.

---

## 1. What I did

1. **Prerequisite gate.** `git fsck` → **exit 0, zero bytes of output**. Clean repo. Root tree
   (`172fa35d…^{tree}`) = `12831fe1be17d04da027dcbb6e45b4dff6f1ef95` = pin. `WORKSPACE-READY.json` read
   `status: ready` going in.
2. **Before-snapshot.** sha256 + size of all 57 `BUNDLE_MANIFEST.json` files →
   `$TMP/h_before.sha` (57 lines, 0 missing). Cross-checked against the manifest's own hashes:
   **0 mismatches of 57** at baseline.
3. **Destination check.** `.claude/outputs/` held exactly the six slugs plus `hub-composition/`, as the
   brief predicted. **Neither `gen-read-cell-arm/` nor `gen-val-split-fix/` existed.** Nothing to merge
   over — proceeded.
4. **Re-verified the tarball myself** (did not take it on trust):
   `sha256 = 4cf00c6262f1f422ab3bdf423198e5a5375b36bda23eafe455ad29499dbfba55`, **216,089,526 bytes** —
   both match. `tar tzf` → **93 members, 93 regular files, 0 directory entries, 0 `._*`, 0 `.DS_Store`.**
   Top level: `AGENT_PROTOCOL.md`, `autonomous-hub-charter.md`, `handover_context.md`,
   `SUPPLEMENT_MANIFEST.json`, `outputs/{gen-read-cell-arm,gen-val-split-fix}`.
5. **Extracted the supplement manifest to a temp dir outside the bundle**
   (`$TMP/h_supp_manifest/`). It self-declares `"member_root": ".claude"` — independent confirmation
   that `.claude/` is the correct destination — `n_files: 92`, `schema:
   chlu-hub-workspace-supplement/1`, built on `batMac.local` from `/Users/user/Desktop/CHLU`.
6. **Extracted exactly two subtrees**, verbatim per the brief:
   ```
   tar xzf /c/Users/PratikJawahar/chlu/hub-refusal.supplement.tar.gz \
       -C .claude/ outputs/gen-read-cell-arm outputs/gen-val-split-fix
   ```
   exit **0**.
7. Verified all 89, proved the 57 untouched, re-ran the verifier, copied the fresh ready-state to
   `_hub_scratch/reports/_H-workspace-ready.json`, then load-checked the artefacts.

### ⚠ On the `SCHILY.fflags` warnings — expected, not an error
GNU tar emitted **exactly 92** lines of `tar: Ignoring unknown extended header keyword 'SCHILY.fflags'`
and **nothing else** on stderr (I grepped for non-SCHILY lines: none). This is macOS bsdtar metadata GNU
tar does not model. It appears 92 times rather than 89 because tar scans every member's header while
seeking the two requested subtrees. **Harmless. Not a failure signal.**

---

## 2. The 89-file verification — ⭐ PASS 89/89

Checked **bytes and sha256** for every `outputs/…` entry in `SUPPLEMENT_MANIFEST.json`.

| Subtree | Files in manifest | Byte+sha256 PASS | FAIL |
|---|---|---|---|
| `outputs/gen-read-cell-arm` | 88 | **88** | 0 |
| `outputs/gen-val-split-fix` | 1 | **1** | 0 |
| **Total** | **89** | **⭐ 89** | **⛔ 0** |

**Total bytes verified: 219,687,872.** **Zero mismatches. Zero missing. No file to name.**

Set equality was checked in both directions, not just one:
- files on disk in the two trees: **89** (`find` count)
- **extra on disk, not in manifest: NONE**
- **in manifest, missing from disk: NONE**

So the extraction is exact: nothing short, nothing stray.

---

## 3. Proof the pinned set is untouched

| Check | Result |
|---|---|
| `diff $TMP/h_before.sha $TMP/h_after.sha` | ⭐ **EMPTY** — all 57 byte-identical, size and sha256 |
| Root tree after | `12831fe1be17d04da027dcbb6e45b4dff6f1ef95` — **equals pin** |
| `git fsck` after | **exit 0, zero output** |
| `git status --porcelain` | **0 lines** (`.claude/**` is ignored at `.gitignore:22`; git never saw the 220 MB) |

**The three out-of-scope docs, confirmed unchanged on disk:**

| doc | bytes on disk | sha256 (first 16) | verdict |
|---|---|---|---|
| `.claude/AGENT_PROTOCOL.md` | 9802 | `abf4992f828b1aef` | pinned value, untouched |
| `.claude/autonomous-hub-charter.md` | **6610** | **`41b6fcbec0b2bc44`** | ⭐ **still the pinned 6610 B version** |
| `.claude/handover_context.md` | 1023731 | `5620b61032793425` | pinned value, untouched |

For the record, I confirmed the supplement's competing charter without installing it — extracted to a
temp dir only: **7656 bytes, `bc1a0eff6d176b2e`**. It remains in the tarball and **off the workspace**.
`BUNDLE_MANIFEST.json` was not opened for writing at any point.

---

## 4. Verifier re-run — `status: ready`

`bash scripts/verify_hub_workspace.sh`, run through the CONTEXT.md shell recipe verbatim
(`env_hub.sh`, venv on `PATH`, `PYTHONUTF8=1`, `VERIFY_PYTHON` set). **Exit 0.**

```
[verify] import: chlu=C:\Users\PratikJawahar\chlu\chlu-hub-bundle\chlu\__init__.py under_root=True jax=0.9.0
[verify] git   : root_tree=12831fe1be17d04da027dcbb6e45b4dff6f1ef95 matches_pin=True head=5c8bfec
[verify] files=57 missing=0 mismatched=0 unopenable=0 import_ok=True
[verify] status=ready
[verify] === ⭐ WORKSPACE READY: checksummed, every artefact OPENED through its real loader,
[verify]     chlu imported FROM THE BUNDLE, the hub's output path written, targeted test green. ===
```

| field | value |
|---|---|
| `status` | **ready** |
| `n_checksum_mismatch` | **0** |
| `n_missing` / `n_load_failed` | 0 / 0 |
| `loaders_all_opened` | true |
| `skip_load` | **false** (earned, not skipped) |
| `import_under_root` | true |
| `hub_output_path_writable` | true |
| `targeted_test` | **7 passed, 19 skipped in 11.24s** |
| `verified_at` | 2026-09-16T17:44:14Z |

⚠ **`anchors_checked: 0` on all seven anchored docs** — as the brief anticipated, this is the known
upstream `_load_doc()` backslash-key bug. **It was 0 before I ran and it is 0 now. I changed nothing about
it and did not attempt to fix it** (those files are sha256-pinned).

Fresh ready-state copied to **`_hub_scratch/reports/_H-workspace-ready.json`** (42,722 bytes).

⭐ **Decision worth flagging:** I deliberately ran the verifier **without** pinning `OMP_NUM_THREADS` /
`MKL_NUM_THREADS`. This step is a pass/fail acceptance test against a previously recorded green run;
thread count can change float reduction order, and I did not want to introduce a numeric confound into
the one check whose job is to prove I broke nothing. It is a short, mostly-I/O job. If the hub would
rather I had been a better core citizen, the run is trivially repeatable.

---

## 5. ⭐ The question that actually matters — is the charter's premise now true?

`.claude/tasks/hub-refusal.md` claims the target is *"scored off artefacts that already exist — the energy
traces, convergence reports, settled states and decoded tuples persisted by `gen-read-cell-arm`."*
**That claim is now true on this machine.** Every named artefact is present **and openable**:

### Inventory — complete and symmetric across 3 seeds

`gen-read-cell-arm/runs/` : 78 `.npz` + 5 `.json` + 3 `.log` + `progress.txt` + `run_scored.out` = 88 files.

- **energy traces** — 24 files, `energy_traces_s{0,1,2}_{A,B}_{B,MS}_{full,request_off}.npz`
- **settled states** — 24 files, same 3×2×2×2 grid
- **decoded tuples** — 24 files, `decoded_s{N}_dynread*_{B,MS}.npz`
- **stored atom tuples** — 6 files, `stored_atom_tuples_s{N}_{A,B}.npz`
- ⭐ **`runs/cell_results.json`** — 51,804 B, present
- ⭐ **three `cell/seed_N/seed_N/cell_manifest.json`** — 159,147 / 160,106 / 159,882 B, **all three present
  at the doubled-seed path exactly as the brief described**
- `runs/dynread_seed{0,1,2}.log`, `launch_diagnostic.json`, `progress.txt`, `run_scored.out`

⭐ **`gen-val-split-fix/refusal_scores.json` — 2,899,315 bytes. Exactly the size the brief specified.**

### Load-check — ⛔ opening it is the test, and it passes

I opened **all 78 `.npz`**, not a sample: every array materialised (not just headers) and checked finite.
**78/78 OK. Zero load failures. Zero non-finite values.** Representative shapes/dtypes:

```
energy_traces_s0_A_MS_full.npz        energy       (1001,198) f32   energy_ref   (1001,198) f32
                                      accept_prob  (1000,198) f32   accepted     (1000,198) bool
                                      displacement (1000,198) f32   sigma_scale  (1000,)    f32
                                      temperature  (1000,)    f32
settled_states_s1_B_B_full.npz        x_settled    (72,2048)  f32   x0           (72,2048)  f32
                                      read         (72,128,4,4) f32 requests     (72,6)     i16
decoded_s2_dynread_MS.npz             images       (199,3,64,64) f32 factor_tuples (199,6)  i64
                                      features     (199,2048) f32   requests     (199,6)    i16
                                      labels       (199,6)    i64   factor_names (6,)       <U11
stored_atom_tuples_s0_A.npz           atom_rows_into_pool (64,) i64  atom_factor_tuples (64,6) i64
                                      distinct_atom_factor_tuples (64,6) i64  factor_names (6,) <U11
```

⭐ The energy traces carry a **1001-step energy curve alongside `accepted` / `displacement`** — i.e. the
settled energy **and** the convergence evidence the charter's threshold rule requires are both in the
artefact, per-request. The `72 → 199` row-count shift between `settled_states` and `decoded` is the
`request_off` collapse the charter's §E.4 describes; I note the structure and **compute nothing from it**.

All JSON parsed cleanly under `PYTHONUTF8=1`:
- `cell_results.json` → 12 keys, including `dynread_convergence`, `dynread_verdict`, `same_cell_proof`,
  `honesty`, `NOT_RUN`.
- `cell_manifest.json` (seed_0) → 33 keys, including `dynread_preflight`, `no_bypass`, `byte_ledger_store`.
- `refusal_scores.json` → 9 keys. ⭐ **Structurally complete, exactly as the charter promised:**
  `status: "complete"`, `not_run: []`, **both byte legs** (`historical_688128`, `deployed_524288`) × **3
  seeds each** (`seed_0/1/2` under both), controls declared for `duplicate_factor` (a7_redundant) and
  `out_of_range` (base), with `A7D` explicitly flagged *"DECLARED SENSITIVITY — not a control"*.

⛔ **I scored nothing, ran no hub, and produced no number that could be quoted as a result.** Everything
above is shape, dtype, key, byte-count and presence.

### ⛔ The two same-named files are distinguishable — no ambiguity left

| | superseded (quarantined) | ⭐ the new one |
|---|---|---|
| path | `gen-arms-a7-wiring/refusal_scores.json` | `gen-val-split-fix/refusal_scores.json` |
| bytes | 2,660,729 | **2,899,315** |
| sha256 | `4d9f87b439cc2b6e…` | — |
| `schema` | — | `gen-val-split-fix/refusal-scores/1` |

The superseded file is **still in place, byte-identical, not moved and not deleted** — it appears with the
same hash in both `h_before.sha` and `h_after.sha`. ⭐ **The `schema` string is the cheapest safe
discriminator: the good file self-identifies as `gen-val-split-fix/…`.** Any consumer should assert on it.

---

## 6. What is broken

**Nothing I did broke anything, and nothing blocks `hub/refusal`.** Two findings worth the hub's attention:

### ⚠ 6.1 — Five Mac-absolute `source:` pointers do not resolve here (provenance only, NOT a blocker)
The shipped JSONs record provenance as absolute paths under `/Users/user/Desktop/CHLU/`. I resolved all
**101 distinct** such references against the bundle root: **96/101 exist locally at the identical relative
path** (decoder pickles, external classifiers, feature bridges, the read-compose stores, every one of the
78 npz). Five do not:

| absent reference | where | assessment |
|---|---|---|
| `gen-store-grouping/runs/shapes3d_recon/store_bytes.json` | `refusal_scores.json` `source_store` | ⭐ **Not needed.** The value it sourced is **inlined right beside the pointer** (`store_bytes: 524288`, `bits_per_read: 30.0`, `bytes_formula`). |
| `…/store_plan.json (K=5, n_modes=64)` | `refusal_scores.json` `source_coupling` | ⚠ **Not a real path** — a path plus a human parenthetical. The bare `store_plan.json` **does exist** (17,695 B). My regex flagged it; discount it. |
| `gen-val-split-fix/runs/historical_688128/seed_{0,1,2}/baselines_manifest.json` | `cell_manifest.json` ×3, `source` of the baseline hyperparameter block | ⭐ **Not needed.** The selected hyperparameters, `tuning_trials_inherited: 36` and `selection_criterion` are inlined next to the pointer. |

⭐ **These are `source:` / `source_store:` / `source_coupling:` breadcrumbs recording where a value came
from, with the value already recorded next to them. Nothing needs to be dereferenced to score.** The
supplement genuinely ships only one file under `gen-val-split-fix/` (`refusal_scores.json`) — that is what
`SUPPLEMENT_MANIFEST.json` declares, and it matched 89/89, so this is **by design, not a truncated
transfer**. ⚠ But a hub that naively `open()`s a `source` field will crash, and it will crash on a
**Mac path on a Windows box**, which reads like an environment bug when it is not one. Flagging so the
next agent does not lose an hour to it.

### ⚠ 6.2 — `git status` is now clean; CONTEXT.md's "36 permanently modified files" is stale
CONTEXT.md warns that `git status` shows 36 mode-only modifications. **It no longer does** —
`git status --porcelain` returns **0 lines**, because `core.filemode` is now **`false`** repo-locally.
**I did not set this** (I ran no `git config`); a prior spoke evidently did, which is a sensible fix.
`core.autocrlf` is still correctly pinned `false` both globally and repo-locally. Reporting it only so the
hub updates its mental model — ⭐ **a clean `git status` here is the new correct state, not a sign that
something swallowed a change.**

**No full failure output to reproduce: nothing failed.** The only stderr produced anywhere in this spoke
was the 92 expected `SCHILY.fflags` lines.

---

## 7. What I could not resolve honestly

- **I cannot confirm the artefacts' *numbers* are the Mac's numbers.** I confirmed **bytes** (89/89
  sha256) and **loadability** (78/78). ⭐ Per CONTEXT.md's unresolved cross-platform discrepancy, that is
  exactly the right scope — and note it lands on the *favourable* side of the distinction CONTEXT.md
  draws: these are **frozen artefacts loaded and read**, not a recipe rebuilt locally. Scoring them is a
  read-and-render operation. **But Spoke I has not reported, so I am not asserting that the numbers a hub
  derives from these files on this box would match the Mac's.** I only assert the inputs are byte-identical
  to what the Mac shipped.
- **I did not verify the artefacts open through their *project* loaders**, only through `numpy.load` /
  `json.load`. The 89 new files are not in `BUNDLE_MANIFEST.json`, so the verifier's real-loader pass does
  not cover them — it still checks only the pinned 57. For raw `.npz`/`.json` this gap is narrow, but it
  is a gap and I am naming it rather than implying coverage I did not achieve.
- **Spoke E's completion — confirmed only indirectly.** `_hub_scratch/reports/E-cleanup-gitpreflight.md`
  was already on disk (timestamped 19:26 local, before my 19:40 extraction), which is good evidence E
  finished first, but I did not read it and I have no live channel to that agent. What I verified directly
  is the *condition* the brief actually cares about: `git fsck` clean, tree matches pin, baseline 57/57
  checksums good, `status: ready` before I started. I applied onto a demonstrably clean repo. ⚠ If any
  spoke touches `.claude/` after 17:44 UTC, my "untouched 57" proof is a snapshot, not a forward guarantee.

---

## 8. Housekeeping

⭐ **No sub-agent is running.** I launched **zero** sub-agents — this spoke was entirely sequential shell
and Python in the foreground. Nothing of mine is outstanding.

### Every path I created

**Inside the bundle (the deliverable — 89 files, 210 MB):**
```
.claude/outputs/gen-read-cell-arm/            88 files, 207 MB
.claude/outputs/gen-val-split-fix/             1 file,  2.8 MB
```
**Outside the bundle:**
```
_hub_scratch/reports/H-supplement-apply.md          (this report)
_hub_scratch/reports/_H-workspace-ready.json        (42,722 B)
```
**Throwaway scratch in `%TEMP%` (`C:\Users\PratikJawahar\AppData\Local\Temp`), safe to delete:**
`h_before.sha`, `h_after.sha`, `h_members.txt`, `h_ondisk.txt`, `h_extract.err`, `h_verify.log`,
`h_fsck.txt`, `h_fsck2.txt`, `h_snap.py`, `h_verify_supp.py`, `h_load.py`, `h_loadall.py`,
`h_pathmap.py`, `h_supp_manifest/SUPPLEMENT_MANIFEST.json`, `h_charter_peek/autonomous-hub-charter.md`.

**Rewritten by the verifier itself, not by me** (listed for completeness — these are its normal outputs,
and I confirmed by mtime sweep that *nothing else* in the bundle was touched):
`WORKSPACE-READY.json`, `coverage.xml`, `.coverage`, `.workspace-verify-test.log`,
`.claude/outputs/hub-composition/.workspace-write-probe.json`, plus `__pycache__/` and `.pytest_cache/`.

⛔ I did not run `git commit`, `git push`, `git reset`, or `git checkout --`. I did not touch
`BUNDLE_MANIFEST.json`, `chlu/`, `tests/`, or any pinned file. No GPU, no cloud spend, no scored run.

---

## 9. Numbered open questions

1. ⭐ **Should the 89 new artefacts be added to `BUNDLE_MANIFEST.json`?** They are currently **unpinned**:
   a later spoke could corrupt one and the verifier would still say `ready`, because it checks only 57.
   The supplement ships no updated manifest and I was forbidden to write one. ⚠ **`hub/refusal` is about to
   be scored off 220 MB of unpinned input.** I recommend the upstream "second, small supplement" carry a
   **`BUNDLE_MANIFEST.json` extended to 146 files**, not just the fixed verifier. Until then,
   `SUPPLEMENT_MANIFEST.json` is the only integrity record and it lives **outside** the workspace.
2. **Where should `SUPPLEMENT_MANIFEST.json` live?** I deliberately kept it in `%TEMP%` per the brief
   (⛔ "not into the bundle"). That makes my 89/89 result unreproducible once temp is cleared. Should a
   copy be parked in `_hub_scratch/reports/` so the hub can re-verify independently? I did not do this
   because the brief did not authorise it, but it seems cheap and useful.
3. **Is the `gen-val-split-fix/runs/…/baselines_manifest.json` omission intentional?** The supplement ships
   `gen-val-split-fix/` as a **single file**. Three `cell_manifest.json` `source` fields point into a
   `runs/` subtree that does not exist in the tarball at all. I judged them provenance-only (§6.1) — ⚠ **the
   Mac should confirm** that no scoring path dereferences them, because I can only see that the values are
   inlined, not that every consumer prefers the inline copy.
4. **Does any `hub/refusal` code path `open()` a `source` / `source_store` field?** If so it will die on
   `/Users/user/Desktop/CHLU/…` here. ⭐ A one-line `MAC_ROOT → $CHLU_ROOT` rewrite at load time would make
   **96 of 101** references resolve. Worth a pre-flight check before the hub burns time.
5. **Is the charter's "scored off artefacts that already exist" line now safe to leave uncorrected?**
   My evidence says **yes** — CONTEXT.md predicted this and the prediction held. ⭐ Recording it explicitly
   so the Head can close that item rather than leave it open.
6. **Should the verifier be re-run with `OMP_NUM_THREADS` pinned?** I ran it unpinned on purpose (§4).
   If the hub wants a core-budget-respecting datapoint, or wants to know whether thread count perturbs the
   7 passing tests, say so — it is a cheap repeat and might even be a **free clue for Spoke I's
   cross-platform investigation**.
7. **`hub/refusal` is unblocked — but the scored-run budget still binds.** ⛔ Under the strict
   **three-per-target-across-all-machines** reading, and given the charter's own ⛔ *"You should not need a
   new scored generation run to answer this"*, the hub should be able to answer at **zero** scored runs off
   these artefacts. ⚠ Nothing in this workspace can see what the Mac has already spent. **Someone with
   sight of both machines should confirm the count before `hub/refusal` starts.**
