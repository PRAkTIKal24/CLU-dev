# SPOKE H — Apply the refusal supplement, artefacts only
## ⛔⛔ DO NOT LAUNCH THIS WITHOUT EXPLICIT HUMAN APPROVAL. It is gated on a decision.

Read `_hub_scratch/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`/c/Users/PratikJawahar/chlu/_hub_scratch/reports/H-supplement-apply.md`**
Prerequisite: **Spoke E must have finished and `git fsck` must pass.** Apply onto a clean repo, not a
contaminated one.

## The situation — established, do not re-derive
`hub-refusal.supplement.tar.gz` is downloaded and **checksum-verified** at
`/c/Users/PratikJawahar/chlu/hub-refusal.supplement.tar.gz`
(`sha256 = 4cf00c6262f1f422ab3bdf423198e5a5375b36bda23eafe455ad29499dbfba55`, 216,089,526 bytes).
It is hygienically clean: **93 members, zero `._*` AppleDouble, zero `.DS_Store`.**

**The documented apply route does not exist on this machine.** `scripts/bundle_hub_workspace.sh` has no
`--supplement-apply` flag — the string "supplement" appears **zero** times in `scripts/` or `tests/` — and
the tarball ships no `scripts/` directory, so it cannot bootstrap its own applier. The parser's catch-all
would print `unknown flag` and exit 1.

**The tar's members are rooted at `.claude/`'s CONTENTS, without the `.claude/` prefix.** Top-level
members are `AGENT_PROTOCOL.md`, `autonomous-hub-charter.md`, `handover_context.md`,
`SUPPLEMENT_MANIFEST.json`, `outputs/…`. ⚠ **Extracting to the workspace root would scatter `outputs/`
into the wrong place.** The correct destination is `$CHLU_ROOT/.claude/`.

### ⛔⛔ THE THREE DOCS ARE OUT OF SCOPE. This is the whole reason this spoke is gated.
| doc | in bundle | in supplement | manifest pins |
|---|---|---|---|
| `AGENT_PROTOCOL.md` | `abf4992f…` 9802 B | `abf4992f…` 9802 B | `abf4992f…` — **identical, no-op** |
| `autonomous-hub-charter.md` | `41b6fcbe…` **6610 B** | `bc1a0eff…` **7656 B** | `41b6fcbe…` — **DIFFERS** |
| `handover_context.md` | `5620b610…` 1023731 B | `5620b610…` 1023731 B | `5620b610…` — **identical, no-op` |

The supplement carries **no updated `BUNDLE_MANIFEST.json`**. Overwriting the charter would therefore
replace a sha256-pinned file with content the manifest does not know, producing a checksum **MISMATCH**
and dropping the workspace from `status: ready` to `NOT-READY`.

⛔ **You extract the two `outputs/` trees and NOTHING ELSE.** ⛔ Do not extract any of the three `.md`
files — not even the two that are byte-identical. ⛔ Do not touch `BUNDLE_MANIFEST.json`.

## Procedure
1. **Snapshot before.** sha256 of all 57 `BUNDLE_MANIFEST.json` files → `/tmp/h_before.sha`. Record the
   root tree (`12831fe1…`) and the current `WORKSPACE-READY.json` `status`.
2. **Confirm the destinations do not already exist.** `.claude/outputs/` currently holds six slugs plus
   `hub-composition/` (the verifier's write-probe). ⛔ **Neither `gen-read-cell-arm/` nor
   `gen-val-split-fix/` may already exist.** If either does, **stop and report** — something applied this
   already and you must not merge over it.
3. **Extract exactly two subtrees**, into `.claude/`:
   ```bash
   cd "$CHLU_ROOT"
   tar xzf /c/Users/PratikJawahar/chlu/hub-refusal.supplement.tar.gz \
       -C .claude/ outputs/gen-read-cell-arm outputs/gen-val-split-fix
   ```
   ⚠ GNU tar will emit `Ignoring unknown extended header keyword 'SCHILY.fflags'` for every member —
   macOS bsdtar metadata, **harmless**. Do not let it look like an error in your report; say it is expected.
4. **Verify every extracted file against `SUPPLEMENT_MANIFEST.json`.** Extract that manifest to a temp
   location (⛔ not into the bundle) and check **bytes and sha256 for all 89 `outputs/…` entries**. Report
   a pass/fail count and name every mismatch. ⭐ This is the real acceptance test — a byte-perfect file its
   loader rejects is still a dead run, so if any `.npz` fails, say so loudly.
5. **Prove the pinned set is untouched.** Re-checksum the 57 → `/tmp/h_after.sha`, `diff` against
   `/tmp/h_before.sha` — **must be empty**. Re-check the root tree equals the pin.
6. **Re-run the verifier**: `bash scripts/verify_hub_workspace.sh`. It must still be `status: ready` with
   `n_checksum_mismatch: 0`. ⭐ If it is not, **you have broken something — report it as the headline.**
   Note `anchors_checked` will still be `0` on the four anchored docs; that is the known upstream Windows
   bug and **not** something this spoke changes or should try to fix.
7. Copy the fresh `WORKSPACE-READY.json` to `_hub_scratch/reports/_H-workspace-ready.json`.

## Then answer the question that actually matters
`hub/refusal`'s charter claims it is "scored off artefacts that already exist". **Check whether that is now
true.** Read `.claude/tasks/hub-refusal.md`, then confirm the artefacts it names are present and openable:
- `.claude/outputs/gen-read-cell-arm/runs/` — energy traces, settled states, decoded npz, `cell_results.json`,
  three `cell/seed_N/seed_N/cell_manifest.json`
- `.claude/outputs/gen-val-split-fix/refusal_scores.json` (2,899,315 bytes)

Open two or three `.npz` files with `numpy.load` and report shapes/dtypes — **opening it is the test.**
⛔ Do **not** start scoring, do **not** run the hub, do **not** produce any number that could be quoted as
a result. You are confirming the inputs exist and load. Nothing more.

⛔ `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` is the **SUPERSEDED broken-split** file. Never
quote it. It is sha256-pinned — never move or delete it. ⚠ Do not confuse it with the new
`gen-val-split-fix/refusal_scores.json`; they have the same basename and that is exactly how a wrong number
gets into a paper.

## Report shape
Lead with: **is `hub/refusal` unblocked on this machine, yes or no**, and **is the workspace still
`status: ready`**. Then the 89-file verification table, the untouched-57 proof, and the artefact
load-check. End with numbered open questions.

⭐ Before you finish, confirm no sub-agent of yours is still running, and list every path you created.
