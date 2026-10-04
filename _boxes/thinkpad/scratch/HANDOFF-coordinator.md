# HANDOFF — you are the new COORDINATOR (lead hub agent). Read this first, then act.
**Written 2026-09-18 by the outgoing coordinator, whose thread filled. Model: `opus[1m]`.**

## Who you are
The **coordinator** in a three-tier structure. ⛔ You do not run experiments and you do not ideate a new
pass. Read **`_hub_scratch/COORDINATOR-LOOP.md`** — it is your operating procedure — and
**`_hub_scratch/CONTEXT.md`** for the accumulated machine facts and standing rulings.

| tier | model | role |
|---|---|---|
| Advisor | **Fable** | ideation **once per pass**; writes `DIRECTIVE.md`; ⛔ never touches the filesystem |
| **You** — coordinator | **Opus** | stage · monitor · verify · compile · small decisions · escalate triggers |
| Execution hubs | **Opus** | the science, 12 h each |

⭐ **The advisor is fed a condensed text block PASTED INTO ITS PROMPT** — ⛔ never a file path, never
"the reports are there if you want to check". Verification and compilation are **yours**, done before it
is invoked. See `ADVISOR-ROLE.md`.

## Where things stand — iteration 2 is COMPLETE
**⭐ Read `_hub_scratch/reports/iter2-compiled.md` — that is the full picture, already verified.**
One-line scoreboard: `generation-table-2` **TIED** (scored 1/3) · `composition-2` **WIN with three
caveats** (1/3) · `deletion-2` **banked TIE + 2 unasked-for real results + 1 charter condition failed**
(0/3) · `admission` **NOT SOLVED, dead on arrival for an exact structural reason** (0/3).
**2 of 12 scored runs spent. No bar moved, no pinned file edited, nothing deleted.**

⚠ `hub-generation-table.md` and `hub-composition/REPORT.md` at the top level are **iteration-1**
documents. Iteration 2 is `hub-generation-table/LOG-iter2.md` and `hub-composition/iter2r/REPORT.md`.

## The workspace
- `$CHLU_ROOT` = `/c/Users/PratikJawahar/chlu/chlu-bundle-v2` (pin `4125fce7`, tree `345244bb`).
  ⛔ v1 at `…/chlu-hub-bundle` is a fallback — do not work in it.
- **`base/iter2` = `945a3e9`** — the pin tree + five reference-read files + the Head-authorised
  `run_seed` gate patch. All iteration-2 branches came off it. ⛔ `ULP_BOUND`/`FRAC_BOUND` untouched.
- Worktrees: `CHLU-hub-{generation-table-2, composition-2, refusal-2, deletion-2, admission}` plus
  iteration-1 ones. ⛔ **Never delete a worktree or an output dir** (§0.1).
- Open a shell **exactly** as `LAUNCH-COMMON.md §2` says: `unset` stale `CHLU_*`, source `env_hub.sh`,
  `export PYTHONUTF8=1`, prepend `$CHLU_ROOT/.venv/Scripts` to PATH.
  ⚠ `PYTHONUTF8=1` is load-bearing — without it the first encode dies at `gen_bridge.py:915`.
  ⚠ Bare `python`/`python3` are the MS Store stub. `uv` lives only in `…\Python312\Scripts`.
  ⚠ Git Bash `/tmp` ≠ native-Python `/tmp`. Write scratch to real paths under `_hub_scratch/`.
  ⚠ Writing files from Python: open in **binary** (`"wb"`) or pass `newline="\n"` — text mode
  silently converts the whole file to CRLF. This has nearly landed twice.
  ⚠ Heredocs with apostrophes break the Bash tool; write a `.py` file and run it instead.

## ⛔ WHAT NEEDS A DECISION — do not decide these yourself
1. ⛔⛔ **Two autonomous agents ran `hub/deletion-2` concurrently**, sharing one worktree and one output
   dir, interleaving `LOG.md`. Incident at `hub-deletion/INCIDENT-two-agents-clu-dev-1e.md`. The hub
   handled it correctly and carried on. **This is a coordination failure above hub level and needs a fix
   before iteration 3.** Surface it to the human early.
2. **Iteration 3 needs an advisor pass.** Four programme-level results are in — that is a re-ideation
   trigger. Compile the brief (it is largely `iter2-compiled.md`, tightened to ≤ 5 KB) and hand it over.
3. **The composition "win" needs adjudication.** It is a WIN on the pre-registered bar (Δ = +30.00,
   p = 2.27e-7, independently replicated) **and** a TIE on the generalising estimate (Δ = +1.09,
   CI [−1.52, +3.69] over 105 corners), **and** its B level does not replicate (42 vs 25). All three are
   the hub's own disclosures. What the paper may claim is a Head/Advisor call.
4. **Admission's structural finding may generalise beyond admission:** `σ_k = nn_median` between atoms,
   atoms are train rows, so the manifold sits ≈ 1 `σ_k` from an atom **by construction** and any
   `σ_k`-scaled radius is pinned outside the data. That is a statement about the bandwidth rule, not
   about admission — it may bear on the read, the landscape and the earlier bandwidth defect.

## Unfinished work you inherit
- **`_hub_scratch/reports/refusal-closure.md` was never written** (DIRECTIVE rev.3 §2.4). Inputs exist:
  both refusal reports, `RIVAL_ROWS.json`, the two-record disclosure in `PREREG-hub-refusal-2.md` §0.1,
  and `reports/refusal-closure-check.md` (done: `max_abs(P_full − P_shuffled) = 0.0` exactly, both rungs;
  plus a new finding — **rung B is rank-deficient in 4 of 5 wells**, 236/305/479/366 of 512).
- **DIRECTIVE §6 upstream items** for the Mac are listed and unactioned (charter amendments, the
  `tasks/hub-refusal.md` retirement, the reporting rule, the bandwidth knob, full-scope seed 1–2 caches).
- ⭐ `pdftotext` **works** at `/mingw64/bin/pdftotext` (only `pdftoppm` is missing). All four NeurIPS
  papers are extracted to `/tmp/papers/*.txt` in Git-Bash space — re-extract if you need them.

## Standing rules you enforce
⛔ never delete · ⛔ never edit a manifest-pinned file · ⛔ never loosen a check to pass it · ⛔ never
re-encode (a shortfall > 0 means **stop and report**) · ⛔ never mix Mac- and Windows-produced quantities
inside one paired comparison · ⭐ true wall clock from `START_UTC.txt`, never perceived time · ⭐ an honest
"not solved" is a success; a manufactured result is the one unrecoverable failure.
⭐ **Verify headline claims yourself before compiling** — re-run cheap checks, and **flag in the brief
anything you could not verify** so the advisor treats the flag as the fact.

## Machine
FORGIS_14 · Windows 11 · 16 cores / 63 GB · Intel Arc (⛔ no CUDA) · pagefile **98304 MB** · sleep and
hibernate **disabled on AC** · ~735 GB free. `gcloud` works (project `forgisprova`). CSF3 unreachable (2FA).
⚠ When a hub thread dies it can **orphan live subprocesses** — this happened twice. Check
`Get-Process python*` and their CPU delta before assuming work stopped, and before assuming it continued.
