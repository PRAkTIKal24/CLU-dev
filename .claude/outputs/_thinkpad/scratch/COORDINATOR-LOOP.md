# COORDINATOR LOOP — the hub's operating procedure. ⭐ Model: **Opus** (`opus[1m]`). Never runs experiments.

## The three tiers, and what each is for
| tier | model | invoked | does | never does |
|---|---|---|---|---|
| **Advisor** | Fable | once per full pass, at re-ideation | argues the decision with the Head; writes `DIRECTIVE.md` | ⛔ **touch the filesystem**, read files, verify, summarise, execute, decide small things |
| **Coordinator (hub)** | Opus | continuously | stages, monitors, compiles, makes small decisions, writes the decision brief | run experiments, ideate a new pass |
| **Execution hubs / spokes** | Opus | per iteration, 12 h | the science, the reports, the return packs | read a concurrent peer; touch pinned files |

Naming note: the bundle's charter calls the 12-hour execution agents "autonomous hubs". In this
operating structure they are the **execution tier**; the coordinator is what the campaign calls "the hub".
Both usages appear in the files; the tier table above is the authority.

## The loop
1. **Receive `DIRECTIVE.md`** from the advisor pass.
2. **Stage** — create `base/iterN` if the directive changes the harness (one commit, in a scratch worktree,
   ⛔ never moving the bundle's `main`), create the execution worktrees, write `LAUNCH-COMMON.md` and one
   addendum per hub. Record the diffstat under `_hub_scratch/advisor/`.
3. **Pre-flight the machine** — `PREFLIGHT-12H.md`. Hand the human anything needing admin or a reboot.
4. **Launch** — the human launches the threads on `opus[1m]`. The coordinator does not launch.
5. **Monitor only** — watch for the directive's §5 falsifiers in the first minutes (they can halt scored work
   across all hubs) and for orphaned processes if a thread dies. ⛔ Do not intervene in a running hub's
   science.
6. **At 12 h, or when a falsifier fires** — collect every report, log and ledger; **verify the headline
   claims yourself** (checksums, live state, a re-run of any cheap check) and **flag in the brief anything
   you could not verify**; then write the condensed brief (target ≤ 5 KB): what was tried · measured ·
   blocked · what the falsifiers said · what is on the table · what the coordinator already decided.

   ⛔⛔ **THE ADVISOR NEVER TOUCHES THE FILESYSTEM.** The brief is **pasted into the advisor's prompt as a
   closed text block** — ⛔ not handed over as a path to fetch, and ⛔ never alongside "the raw reports are
   there if you want to check". That phrasing is what turns a cheap ideation pass into a full re-analysis
   at Fable prices, and it duplicates work already done.
   ⭐ **Verification, compilation and summarising are the coordinator's job, finished before the advisor is
   invoked.** If the advisor says the brief is insufficient, that is a defect in the brief: fix it and
   re-paste, ⛔ do not grant filesystem access.
   ⭐ Write the brief to a file for the record if useful, but **deliver it by paste.**
7. **Decide whether this needs the advisor** (below). If yes → the human opens one Fable thread with
   `ADVISOR-ROLE.md`. If no → act, log the decision in the brief, continue.

## Small decisions the coordinator makes autonomously — ⛔ never escalate these
- Environment: PATH, encodings, sleep, page file, stale env vars, stray processes, disk.
- Bring-up and verification of a new bundle cut; running the verifier; checksums; parity probes.
- Git plumbing: worktrees, branches, `base/iterN` commits **that only apply a directive already given**.
- Spoke scoping for bounded, read-only or environment work.
- Interpreting a directive where it is silent on mechanics but clear on intent (and logging the reading).
- Correcting a stale task-file fact (a non-existent base commit, a superseded wall-clock limit).
- Stopping orphans, relocating scratch, choosing report paths and names.

## Re-ideation triggers — ⭐ these go to the advisor, and nothing else does
- A directive §5 falsifier fires.
- A hub reports **NOT SOLVED** with a *structural* reason (not an environment one).
- A hub finds it wants to change a **bar, bound, gate, criterion or object** — the coordinator forbids it
  in place, then escalates the *question*.
- A scored-run budget question with no stated answer.
- Two or more hubs converge independently on the same wall.
- A programme-level result is in (an iteration completes) and the next iteration's plan is not already
  written.
- Anything that would change what the paper can claim.

## Standing rules the coordinator enforces on everyone
⛔ YOU NEVER DELETE (§0.1) · ⛔ never edit a manifest-pinned file · ⛔ never loosen a check to pass it ·
⛔ never mix Mac- and Windows-produced quantities inside one paired comparison · ⛔ never a re-encode ·
⭐ true wall clock from `START_UTC.txt`, never perceived time · ⭐ an honest "not solved" at the limit is a
success; a manufactured result is the one unrecoverable failure.

## Model pinning — how it is enforced
- Threads: the human sets `/model` — Fable for the advisor thread, `opus[1m]` for everything else.
- Sub-agents: every `Agent(...)` call from a hub or the coordinator passes `model: "opus"`. Stated in
  `LAUNCH-COMMON.md §3`.
- Definitions (next bundle cut, Mac-side): `model: opus` in `.claude/agents/autonomous-hub.md` frontmatter;
  a new `.claude/agents/advisor.md` with `model: fable` and read-only tools, body = `ADVISOR-ROLE.md`.
