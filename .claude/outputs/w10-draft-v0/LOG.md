# w10 spoke/draft-v0 — LOG

START_UTC: 2026-09-21T08:49:19Z (see START_UTC.txt)
Branch: w10/draft-v0   Worktree: C:/chlu/worktrees/CHLU-w10-draft-v0
Output dir (sole write target): /c/chlu/outputs/w10-draft-v0/

## Gate check (mechanical precondition)
Task requires all three to exist before starting.

| Required | Found at | Status |
|---|---|---|
| `/c/chlu/outputs/w10-seeds-and-tables/TABLES.json` | same path, 263,860 B, mtime 2026-09-21 10:34 | PRESENT |
| `/c/chlu/outputs/w10-physics-section/EVIDENCE.md` | `/c/chlu/outputs/w10-physics-section-evidence/EVIDENCE.md`, 64,732 B, mtime 10:35 | PRESENT (dir name differs from task text by the `-evidence` suffix; that spoke's own dir is `w10-physics-section-evidence/`, it carries START_UTC/LOG/bundle, so this is the named artefact) |
| updated negatives registry | `/c/chlu/outputs/w10-registry-fold/negative_results.md` 1,442,067 B + `claims_matrix.md` 442,589 B + `FINDINGS.md`, mtime 10:31-10:33 | PRESENT |

GATE: PASSED. Proceeding.

Note: `/c/chlu/outputs/w10-referee/` is empty (that spoke had not landed at the time of this
check). Not a precondition for this spoke; recorded as a DECLARED NOT-AVAILABLE input.

## Work log

- **08:49Z** Gate checked (table above). START_UTC written first.
- **08:52Z** Read `/c/chlu/outputs/w10-seeds-and-tables/FINDINGS.md` (13,810 B) and `TABLES.md`
  (40,881 B, 456 lines) in full — T1/T2/T3/T4, the 5 declared NOT-RUNs NR1–NR5, corrections C1–C4,
  figures fig1–fig5, and the 27-entry source register.
- **08:55Z** Read `/c/chlu/outputs/w10-registry-fold/FINDINGS.md` (11,951 B) — the 21 `N-I8-*`
  registry entries and 13 `CM-I8-*` matrix rows, the "9/9 seeds" phrasing correction (§3), and the
  Hall–Marshall–Martin correction that gives ground back (§4).
- **08:58Z** Read BOTH `/c/chlu/evidence/outputs/_thinkpad/w8/w8-novelty/NOVELTY.md` (25,843 B) AND
  its `LOG.md` (18,284 B), as the task requires. The LOG's 16:53Z entry and the NOVELTY addendum §D
  are the correction that gives ground back; the file alone reads too pessimistic.
- **09:02Z** Read `/c/chlu/outputs/w10-physics-section-evidence/EVIDENCE.md` (64,732 B) §0–§7.
- **09:10Z** Traced the composition RESULT the tables spoke did **not** table. `TABLES.md` covers
  w8/w9 artefacts only; the programme's 3-seed, held-out-TEST composition win is
  **w5 `hub/compose-field`** (`/c/chlu/outputs/w5-compose-field/LOG.md:196,247`;
  `/c/chlu/evidence/outputs/w5-advisor-report.md` §1). Verified it is TEST / 3 seeds / 524,288 B
  tolerance 0 / pre-registered bars, so it is a main-text RESULT and I place it as one. ⚠ This is
  NOT in tension with the tables spoke's "no Shapes3D composition scoreboard at 3 seeds" — that
  refers to the **incumbent generation-cell instrument** (NR1: K=64, A=1,216, `eval_b`, validity
  rate), a different instrument from compose-field's distinct-valid-novel-tuples-per-corner.
- **09:14Z** Traced the "one object that composes AND forgets" leg to
  `/c/chlu/outputs/w8-single-design/LOG.md:553-620` (REVISED final report) — the 7.48× / 0.991× pair.
- **09:16Z** Traced plan-space 1.0000 vs latent-space 0.13 to `/c/chlu/outputs/w9-maze/LOG.md:288-346`
  (10:55Z entry). Per-seed latent 0.1649 / 0.0950 / 0.1449 (mean 0.1349).
- **09:20Z** Writing `DRAFT-v0.md`.
- **09:40Z** `DRAFT-v0.md` written (11 sections + §§B–H registers). Verified every artefact path it
  cites resolves on disk: the 5 figures under `w10-seeds-and-tables/figures/`, the 3
  `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`,
  `_thinkpad/w6/w6-potentials/test/s0/results.json`, `w8-downdate/DOWNDATE-READY-test.json`, and
  `w9-maze/LOG.md:288` (the 10:55Z mechanism entry — line number confirmed by grep).
- **09:44Z** `FINDINGS.md` written for the Hub.
- **09:46Z** Deliverables copied into the worktree under `projects/w10-draft-v0/` (additive only);
  committing branch `w10/draft-v0`, then bundling **that branch only**.

## Declared NOT-RUN / NOT-DONE for this spoke
| item | status | reason |
|---|---|---|
| Scored runs | ⛔ NOT RUN | this spoke is a writer; no model, no number produced |
| Prose (abstract, title, body sentences) | ⛔ NOT WRITTEN | ⭐ by mandate — the Head writes the prose |
| Figures 6–9 (composition RESULT, compose-XOR-forget, CRR, plan-vs-latent) | ⛔ NOT BUILT | no artefact exists for them; declared as gaps F1–F4 with their source JSONs named, not left blank |
| Per-cell D17/D18 slope rows | ⛔ CANNOT BE RUN from the artefacts | never written to a machine-readable file (gap F6) |
| `/c/chlu/outputs/w10-referee/` | ⛔ NOT AVAILABLE | directory empty at 08:50Z; recorded, not substituted |
| `$CHLU_CLAUDE_ROOT` | ⛔ NOT READ | standing rule: frozen, stale snapshot |
| Background jobs / PIDs / box-wide kill | ⛔ NONE | no long-running process launched at any point |
- **09:54Z** Committed `84402e1` on `w10/draft-v0` (4 files under `projects/w10-draft-v0/`; the repo
  `.gitignore` ignores `projects/*/`, so `git add -f` was required — same convention the sibling
  spokes used). Worktree clean after commit.
- **09:55Z** `git bundle create /c/chlu/outputs/w10-draft-v0/w10-draft-v0.bundle w10/draft-v0` —
  **that branch only, never `--all`**. `git bundle verify`: *"is okay"*, *"records a complete
  history"*, and `bundle list-heads` shows **exactly one ref**: `84402e1… refs/heads/w10/draft-v0`.
  Clone-tested into the scratchpad: all four deliverables present and **byte-identical after CRLF
  normalisation** (the checkout applies Git's autocrlf; contents match exactly). Test clone removed.
- **09:56Z** `END_UTC.txt` written. ⛔ No background job was ever launched, none is running, and no
  box-wide kill was issued at any point in this session.
