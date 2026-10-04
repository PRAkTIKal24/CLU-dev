# CONSOLIDATION — every hub's work, staged for the Mac to pull
**Written by the coordinator (Opus) on FORGIS_14, 2026-09-18.**
**Source of everything below: `C:\Users\PratikJawahar\chlu\` — ⛔ nothing there was moved, renamed or
deleted. This is a copy-only operation. Every source path is intact.**

> ⭐ **Read §1 first if you only read one section.** The scientific record is **18.7 MB**. The other
> **9.9 GB** is recomputable cache. Pull them separately.

---

# 0. WHAT IS NOW AT `C:\chlu`

| dir | files | size | what it is |
|---|---|---|---|
| `worktrees\` | 3,887 | 93.9 MB | ⭐ **NEW** — all 9 hub worktrees, file-copied (no git) |
| `outputs\` | 1,776 | 9,910.8 MB | ⭐ **NEW** — every `hub-*` output tree |
| `scratch\` | 79 | 1.9 MB | ⭐ **NEW** — coordinator scratch: DIRECTIVE, all LAUNCH addenda, all reports |
| `bundle-meta\` | 5 | 0.2 MB | ⭐ **NEW** — `BUNDLE_MANIFEST.json`, `env_hub.sh`, `WORKSPACE-READY.json`, etc. |
| `returns\` | 10 | 253 MB | pre-existing, untouched — the bundles/return packs already staged |
| **TOTAL** | **5,757** | **10.02 GB** | |

---

# 1. ⭐⭐ THE MOST IMPORTANT NUMBER IN THIS FILE

Across all six `hub-*` output trees:

| class | files | size |
|---|---|---|
| ⭐ **evidence** — `.md .json .jsonl .py .log .sh .txt .png .tsv` | **1,101** | ⭐ **18.7 MB** |
| `.npz` — DEV caches and intermediate arrays | 661 | ~9.4 GB |
| `.tar.gz` / `.bundle` nested inside outputs | 13 | ~0.5 GB (after the exclusions in §5) |

⇒ **Every PREREG, LOG, REPORT, ledger, scorer script and result JSON from every hub fits in 18.7 MB.**
⭐ **Suggested pull order for the Mac:** `scratch\` + `bundle-meta\` + `worktrees\` + the evidence
extensions under `outputs\` first (≈ 115 MB total, minutes over ssh), then the `.npz` bulk only if
someone actually needs to re-derive an intermediate.

An rsync filter that gets the whole record without the bulk:
```
rsync -av --exclude='*.npz' --exclude='*.npy' sshkey@FORGIS_14:/c/chlu/ ./chlu-from-windows/
```

---

# 2. INVENTORY — what existed at the source, before any copying

## 2.1 Top level of `C:\Users\PratikJawahar\chlu`
| entry | kind | notes |
|---|---|---|
| `chlu-bundle-v2\` | the live workspace (`$CHLU_ROOT`) | ⭐ **holds `.claude\outputs\` — where ALL hub work actually landed** |
| `chlu-hub-bundle\` | v1 fallback bundle | ⛔ excluded per instruction. See §5.1 — I copied it by accident and removed the copy |
| `CHLU-hub-admission\` … ×9 | git worktrees | ⭐ all copied |
| `hub-composition-env\` | a data/lib staging dir | see §4.3 |
| `_hub_scratch\` | coordinator scratch | ⭐ copied to `C:\chlu\scratch\` |
| 9 × `.tar.gz` / `.bundle` at root | return artefacts | already in `C:\chlu\returns\` |

## 2.2 ⚠ NO WORKTREE HAS ITS OWN `.claude\outputs` — and that is expected, not a loss
I checked every worktree with a directory listing. **None contains a `.claude\` directory at all.**
They are git worktrees of one repository; `.claude\` is gitignored and exists only in the main checkout.
Every hub was pointed at `$CHLU_ROOT/.claude/outputs/<slug>/`, which resolved to
`chlu-bundle-v2\.claude\outputs\` **for all of them**. ⇒ ⭐ **There is exactly ONE outputs root. Nothing
had to be merged, and no two hubs wrote to same-named directories under different roots.**

⚠ **One exception, and it is real work that lives outside the outputs tree:**
`CHLU-hub-admission\iter2\hub-admission\` holds a **complete second copy** of admission's deliverables —
LOG.md (56,243 B), REPORT-hub-admission.md (39,450 B), PREREG, 14 result JSONs, the PNG figure, and a
**`scorer\` directory of 22 Python files** (admission_lib, u0c…u10, verify_core/fairness/launch/liveset).
⭐ It is copied intact as part of the worktree copy. It is near-identical to `outputs\hub-admission\`.

## 2.3 The second outputs root, checked and found empty of hub work
`chlu-hub-bundle\.claude\outputs\` (the v1 bundle) contains only the frozen `gen-*` instrument artefacts
and **one** hub entry: `hub-composition\.workspace-write-probe.json` — **154 bytes, a write probe.**
⛔ **No hub work is stranded in v1.**

## 2.4 Source inventory of `chlu-bundle-v2\.claude\outputs` (before copying)
| entry | files | MB | modified |
|---|---|---|---|
| `hub-admission\` | 54 | 1.3 | 09-17 17:19 |
| `hub-composition\` | 617 | 6,064.1 | 09-17 13:03 |
| `hub-deletion\` | 310 | 124.6 | 09-17 19:29 |
| `hub-generation-table\` | 678 | 10,800.6 | 09-17 21:32 |
| `hub-refusal\` | 122 | 75.0 | 09-17 13:49 |
| `hub-refusal-2\` | 24 | 0.2 | 09-17 12:54 |
| `hub-composition.md` · `hub-deletion.md` · `hub-generation-table.md` · `hub-generation-table-2.md` | 4 | 0.06 | ⭐ **the canonical reports live HERE, at the root, not inside the hub dirs** |
| `hub-composition.return.tar.gz` | 1 | 3.2 | |
| `gen-*\` × 8 | 128 | 2,209.2 | ⛔ **not copied** — the frozen Mac-produced instrument (§5.3) |

---

# 3. WHAT WAS COPIED — source → destination, verified

## 3.1 Worktrees → `C:\chlu\worktrees\`
Excluded: `.git`, `__pycache__`, `.pytest_cache`, `.venv`, `node_modules`.
⭐ **Every count below was verified by re-walking source and destination after the copy. All 9 match exactly.**

| worktree | files | MB | verify |
|---|---|---|---|
| `CHLU-hub-admission` | 473 | 11.3 | ⭐ OK |
| `CHLU-hub-composition` | 420 | 10.0 | ⭐ OK |
| `CHLU-hub-composition-2` | 427 | 12.3 | ⭐ OK |
| `CHLU-hub-deletion-2` | 433 | 10.1 | ⭐ OK |
| `CHLU-hub-generation-table` | 427 | 12.3 | ⭐ OK |
| `CHLU-hub-generation-table-2` | 432 | 12.4 | ⭐ OK |
| `CHLU-hub-generation-table-2-replica` | 430 | 10.0 | ⭐ OK |
| `CHLU-hub-refusal` | 420 | 10.0 | ⭐ OK |
| `CHLU-hub-refusal-2` | 425 | 10.0 | ⭐ OK |
| **total** | **3,887** | **98.5** | |

## 3.2 Outputs → `C:\chlu\outputs\`
Excluded: `__pycache__`, `.pytest_cache`, and the two large tarballs in §5.2.
⭐ **All six verified by re-walk. All match.**

| hub | files | MB | verify |
|---|---|---|---|
| `hub-admission` | 51 | 1.3 | ⭐ OK |
| `hub-composition` | 602 | 6,358.4 | ⭐ OK |
| `hub-deletion` | 310 | 130.6 | ⭐ OK |
| `hub-generation-table` | 666 | 3,819.5 | ⭐ OK |
| `hub-refusal` | 117 | 78.6 | ⭐ OK |
| `hub-refusal-2` | 21 | 0.2 | ⭐ OK |
| **total** | **1,767** | **10,388.6** | |

Plus at `C:\chlu\outputs\` root: the four canonical report `.md` files and `hub-composition.return.tar.gz`.

## 3.3 Also copied
- `_hub_scratch\` → **`C:\chlu\scratch\`** (79 files, 1.9 MB). ⭐ **Not in your numbered list, and I judged
  it essential** — it holds `DIRECTIVE.md`, `CONTEXT.md`, `COORDINATOR-LOOP.md`, `ADVISOR-ROLE.md`,
  `LAUNCH-COMMON.md`, every per-hub `LAUNCH-*.md` addendum, `PREFLIGHT-12H.md`, and `reports\` (23 files),
  which includes both advisor-facing documents in §7. Without it the launch conditions for every hub are lost.
- `chlu-bundle-v2` root metadata → **`C:\chlu\bundle-meta\`**: `BUNDLE_MANIFEST.json` (the 57/146 sha256
  pins), `WORKSPACE-READY.json`, `env_hub.sh`, `README-HUB-BUNDLE.md`, `.gitignore`.
- **`C:\chlu\outputs\_recovered-from-return-tarballs\`** — the only files unique to the two excluded
  tarballs (§5.2): two `RETURN_MANIFEST.json` (136 KB, 39 KB), a duplicate `hub-generation-table.md`,
  and a superseded draft report (§5.2).

---

# 4. PER-HUB COMPLETENESS — code · outputs · PREREG · LOG · report

⭐ **All paths below are relative to `C:\chlu\outputs\<hub>\` unless stated.**

| hub | code? | outputs? | PREREG | LOG.md | REPORT | ledger | incident |
|---|---|---|---|---|---|---|---|
| **admission** | ⭐ yes — 21 `.py` (scorer) + worktree + ⭐ 2nd copy in worktree `iter2\` | ⭐ 51 files | ⭐ `PREREG-hub-admission-1.md` | ⭐ 56 KB | ⭐ `REPORT-hub-admission.md` 39 KB | ⭐ `RESULTS_LEDGER.md` | — |
| **composition (v1 + v2)** | ⭐ yes — 65 `.py` | ⭐ 602 files | ⭐ ×3 incl. `iter2r/PREREG-object.json` | ⭐ ×3 (root, `iter2/`, `iter2r/`) | ⭐ `REPORT.md` + `iter2r/REPORT.md` | ⭐ ×3 | — |
| **deletion-2** | ⚠ only 3 `.py` in outputs; the rest is in the worktree | ⭐ 310 files | ⭐ ×2 + a review doc | ⭐ yes | ⭐ `REPORT-hub-deletion.md` + ⭐ the co-agent's preserved skeleton | ⭐ yes | ⭐⭐ **`INCIDENT-two-agents-clu-dev-1e.md`** |
| **generation-table (v1 + v2)** | ⭐ yes — 144 `.py` | ⭐ 666 files | ⭐ ×5 | ⭐ `LOG.md` + `LOG-iter2.md` (100 KB) | ⚠ **at `outputs\` ROOT** as `hub-generation-table.md` / `hub-generation-table-2.md` | ⭐ ×40 | — |
| **refusal (v1 + v2)** | ⭐ yes — 19 `.py` | ⭐ 117 files | ⭐ ×2 | ⭐ yes | ⭐ `REPORT-hub-refusal.md` + `REPORT-hub-refusal-2.md` | ⭐ `runs/LEDGER.jsonl` | — |
| **refusal-2 (dir)** | ⭐ 11 `.py` | ⚠ 21 files only | ⭐ `PREREG-hub-refusal-2-1.md` | ⭐ yes | ⚠ **none here** — it lives in `hub-refusal\REPORT-hub-refusal-2.md` | ⭐ yes | — |

## 4.1 ⚠ Two "missing report" entries that are NOT missing
- **generation-table's reports are at the `outputs\` root**, as `hub-generation-table.md` (20,190 B,
  iteration 1) and `hub-generation-table-2.md` (19,602 B, iteration 2). ⭐ Both copied. The hub dir itself
  contains only a nested copy under `return_stage_2\`.
- **refusal-2's report is in the `hub-refusal\` directory**, not `hub-refusal-2\`. ⭐ Copied.

## 4.2 ⚠ The replica worktree
`CHLU-hub-generation-table-2-replica` (430 files, 10.0 MB) holds **only the repository tree** — no
hub-specific outputs, no PREREG, no log, and nothing under any `.claude` path. ⭐ **Copied anyway.**
⚠ **I could not determine from a file listing alone whether its checked-out branch differs from the
main generation-table-2 worktree** — that is a git question and you told me not to use git to find work.
**Flagging it rather than guessing.**

## 4.3 `hub-composition-env\` — ⛔ NOT copied, and why
| file | size | verdict |
|---|---|---|
| `data\gen\shapes3d\3dshapes.h5` | 255.2 MB | ⛔ the public Shapes3D dataset — downloadable |
| `data\gen\shapes3d\factor_index.npz` | 3.7 MB | ⛔ derived index |
| `a7_redundant.json`, `c_invalid.json`, `query_sets.json` | 0.2 MB | ⭐ **already present** in `bundle-meta`'s source tree and in the v2 bundle at `.claude\data\gen\shapes3d\` |
| `pylibs\` | 7.8 MB, 121 files | ⛔ vendored Python libraries |
⇒ **environment staging, not hub work.** ⚠ Say the word and I will copy it; it is 263 MB.

---

# 5. WHAT I DID NOT COPY — every item, with the reason

## 5.1 ⚠ A MISTAKE I MADE AND CORRECTED — you should know about it
My first worktree glob was `CHLU-hub-*`. **Windows path matching is case-insensitive, so it also matched
`chlu-hub-bundle`** — the v1 fallback you explicitly excluded — and copied 570 files / 2.2 GB of it into
`C:\chlu\worktrees\chlu-hub-bundle\`.
⭐ **I deleted that destination copy only.** I verified the source first: `C:\Users\PratikJawahar\chlu\
chlu-hub-bundle` still exists with **13,709 files**, untouched. ⛔ **Nothing at the source was affected.**

## 5.2 ⭐⭐ TWO 3.5 GB TARBALLS INSIDE `hub-generation-table\` — excluded, and I proved they are duplicates
| file | size |
|---|---|
| `hub-generation-table-2.return.SUPERSEDED-1924Z.tar.gz` | 3,589.4 MB |
| `hub-generation-table-2.FULL-3.7GB.return.tar.gz` | 3,568.5 MB |

**I listed both in full (`tar -tzvf`) and diffed the paths against the live tree they sit inside:**

| tarball | files | uncompressed | paths NOT in the live tree |
|---|---|---|---|
| FULL-3.7GB | 652 | 3,752 MB | ⭐ **3** — `RETURN_MANIFEST.json` (136 KB), `hub-generation-table.md` (identical to the root copy), one nested `.tar.gz` |
| SUPERSEDED-1924Z | 208 | 3,768 MB | ⭐ **2** — `RETURN_MANIFEST.json` (39 KB), `REPORT-hub-generation-table-2.md` (18,622 B) |

⭐ **I extracted all of those unique small files** to `outputs\_recovered-from-return-tarballs\`.
⭐ **And I diffed the recovered report against the live one:** `REPORT-hub-generation-table-2.md`
(18,622 B, 21:22Z) is an **earlier draft** of `hub-generation-table-2.md` (19,602 B, 21:31Z). The diff is
one added block (`atoms_are_atoms.py`: NMF atoms have 0.00 % entries outside the post-ReLU data orthant
vs 47.12 % for the `pca_affine` control, plus its two caveats). ⇒ ⭐ **the live file is a strict superset;
nothing was lost.** The draft is kept anyway.

⇒ ⭐ **7.16 GB excluded, zero unique evidence lost.** ⛔ **Both source files remain in place, untouched.**

## 5.3 `gen-*\` frozen instrument artefacts — ⛔ not copied (2,209 MB)
`gen-feature-bridge`, `gen-decoder-lift`, `gen-external-classifier`, `gen-read-cell-arm`,
`gen-store-grouping`, `gen-read-compose`, `gen-arms-a7-wiring`, `gen-val-split-fix`.
⭐ **These are Mac-produced and the Mac already has them.** The 1.83 GiB `features.npy` still carries the
Mac-recorded `sha256 d6728f01d102b0d0301198dc617f40bfb8ddb6c5b4a12b21a1ec08e1d974f2e2` — I re-verified it
on this box today and it matches byte for byte. ⚠ Tell me if you want them and I will copy.

## 5.4 Excluded per your instruction
`chlu-bundle-v2.tar.gz` (1.41 GB) · `chlu-hub-bundle.tar.gz` (1.20 GB) · `chlu-hub-bundle\` (2.2 GB) ·
`hub-refusal.supplement.tar.gz` (216 MB). ⛔ All still at the source.

## 5.5 `.git` directories — excluded per your instruction. ⚠ **See §6.1, this has a consequence.**

---

# 6. ⛔⛔ WHAT I COULD NOT COVER — read this before you consider the job done

## 6.1 ⛔⛔ `hub/deletion-2` HAS NO GIT BUNDLE ANYWHERE ON THIS MACHINE
I searched the entire tree for `*.bundle` and `*.gitbundle`. **Ten exist. None is deletion's.**

| bundle | location |
|---|---|
| `hub-admission.bundle` · `hub-refusal.bundle` · `hub-refusal-2.bundle` | `C:\chlu\returns\` ⭐ |
| `hub-composition.bundle` · `hub-composition-2.gitbundle` | inside `outputs\hub-composition\` ⭐ copied |
| `hub-generation-table.branch.bundle` · `hub-generation-table-2.branch.bundle` (+3 nested duplicates) | inside `outputs\hub-generation-table\` ⭐ copied |
| ⛔ **`hub-deletion-2.bundle`** | ⛔ **DOES NOT EXIST** |

⚠ **`REPORT-hub-deletion.md` claims it was produced** — *"Code bundle: `hub-deletion-2.bundle`, whole
branch `hub/deletion-2`, 3.2 MB, verified 'records a complete history' (12 commits, 2,493 insertions)."*
**It is not on disk.** Nor is the `hub-deletion.return.tar.gz` (311 files, 30 MB) that the same report
names.

⇒ ⛔⛔ **`hub/deletion-2`'s 12 commits exist in exactly one place: `chlu-bundle-v2\.git`, which you told
me to exclude.** The **working tree** is safe (`worktrees\CHLU-hub-deletion-2`, 433 files) and its
**outputs** are safe (310 files incl. PREREG, LOG, report, incident note) — but the commit history is not.
⭐ **This is the one gap where the single point of failure still exists.**
⚠ **You told me we are deliberately not using git, so I have not run `git bundle create`.** It is a
read-only, additive, ~3 MB operation. **Say the word and I will produce bundles for deletion-2 — and for
any other branch you want — into `C:\chlu\returns\`.**

## 6.2 ⚠ Things that looked like work and I was unsure about
1. **`CHLU-hub-generation-table-2-replica`** — §4.2. Copied, but I cannot tell you what distinguishes it
   without git.
2. **`outputs\hub-generation-table\return_stage\` and `return_stage_2\`** — nested, partially duplicated
   staging trees (`return_stage_2` re-nests a whole `.claude\outputs\hub-generation-table\`). ⭐ **Copied
   whole, not de-duplicated** — de-duplicating is a judgement call about which copy is canonical, and
   that is yours.
3. **`outputs\hub-composition\iter2\` vs `iter2r\`** — two iteration-2 directories. `iter2r` ("relaunch")
   is the one the compiled report treats as authoritative; `iter2` is the stopped first attempt. ⭐ Both
   copied. ⛔ Neither merged.
4. **`outputs\hub-deletion\REPORT-skeleton-clu-dev-89-1127Z.md`** — the second concurrent agent's
   in-progress report skeleton from the incident. ⭐ Preserved, copied, ⛔ not merged with the real report.
5. **`_hub_scratch\round1-superseded\`** (4 files) and **`I-probe\`** (15 files) — superseded spoke
   material. ⭐ Copied inside `scratch\`.

## 6.3 ⚠ What I did not verify
- ⛔ **I did not checksum-compare source against destination file-by-file.** I verified **file counts and
  total bytes** per tree, and all 15 trees matched exactly. That catches truncation and omission; it would
  not catch silent per-file corruption.
- ⛔ **I did not open the 9.4 GB of `.npz`** to confirm they are what their names say.
- ⛔ **I did not test that the `sshkey` account can actually read `C:\chlu`.** You state it can. ⚠ Worth
  one `ls` from the Mac before you consider this done.

---

# 7. ⭐ THE TWO DOCUMENTS FOR THE ADVISORS — where they now are

Both are inside `C:\chlu\scratch\reports\`:

| file | size | what it is |
|---|---|---|
| ⭐⭐ **`iter2-methods-and-results.md`** | **62 KB** | **THE ONE TO READ.** Written 2026-09-18 for the Mac advisor **and** the FORGIS_14 advisor. Self-contained — ⛔ needs no filesystem access. Covers: the shared frozen instrument (encoder, the two decoders, the judge, the cache with hashes); ⭐ **the Mac-vs-Windows representation question answered with measurements** (§1.6); ⭐ **the four different CLU objects the four hubs actually ran, side by side** (§2.7); exactly what each hub ran and measured (§3); cross-cutting facts (§4); the scored-run ledger (§5); limits (§6); ⭐ **the nine open decisions** (§7); and ⭐ **what I verified myself vs took on trust** (§8). |
| `iter2-compiled.md` | 7 KB | the short scoreboard — the four verdicts and the process incident |
| `refusal-closure-check.md` | 3 KB | the measurement that retired `hub/refusal` |

⚠ **Three corrections in `iter2-methods-and-results.md` that supersede earlier documents**, so neither
advisor works from the stale version:
1. ⭐ **There is no Mac-vs-Windows representation difference.** In this codebase the encoder's output **is**
   the penultimate feature map, and the 1.83 GiB cache is the Mac's own bytes — sha256 re-verified here.
2. ⭐ **"The CLU" is four different objects across the four hubs** (K = 64 / 32 / 5 / 5; two partition
   functions; two read verbs). §2.7.
3. ⚠ **A correction to `refusal-closure-check.md`'s rank finding** — three of five rung-B wells are at
   the maximum attainable rank, not deficient. §4.1.

⭐ Also in `C:\chlu\scratch\`: `DIRECTIVE.md` (rev. 3), `CONTEXT.md`, `COORDINATOR-LOOP.md`,
`ADVISOR-ROLE.md`, `HANDOFF-coordinator.md`, `LAUNCH-COMMON.md` and all seven `LAUNCH-hub-*.md` addenda.

---

# 8. VERIFICATION SUMMARY

| check | result |
|---|---|
| worktree file counts, source vs destination | ⭐ **9 of 9 match exactly** |
| outputs file counts, source vs destination | ⭐ **6 of 6 match exactly** |
| total bytes per tree, source vs destination | ⭐ **15 of 15 match** |
| anything moved or deleted at the source | ⭐ **nothing** |
| destination collisions requiring a suffix | ⭐ **none** — `C:\chlu` held only `returns\`, which I did not touch |
| my own erroneous copy | ⚠ one, removed, source verified intact (§5.1) |
| free space after | 742 GB |
