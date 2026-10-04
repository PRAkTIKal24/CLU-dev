# hepa-lh-audit — do these datasets even HAVE history older than 512 steps?

**ICLR long. Agent: results-analyst. CPU only, local. NO worktree, NO branch, NO commit** — you touch
zero tracked files. Everything you write lives under `.claude/`.
Runs **concurrently with `hepa-lh-tripwire`**; your ownership lists are disjoint (§5).
**Budget: ~1.5 days.** Data acquisition is the risk here, not compute.

## ⛔ READ FIRST, IN FULL, BEFORE ANYTHING
1. `.claude/outputs/hepa-longhorizon-triage/PREREG.md` — **frozen. You may not edit it.** It names the
   protocol, the seeds, the rungs, the kill rule, and a blind per-dataset prediction you are about to
   test. Do not read the predictions as guidance — your job is to measure, and a result that
   contradicts §5 is the most valuable thing you can produce.
2. `.claude/outputs/hepa-benchmark-scout.md` §3, §4, §10 — the loader semantics and the source URLs.
3. `.claude/AGENT_PROTOCOL.md` §2 and §5.

## 1 · THE ONE JOB
For the **nine in-scope datasets** — C-MAPSS FD001/2/3/4, PSM, MBA, GECCO, BATADAL, ETTm1, Weather,
Beijing-AQ (that is 11 rows; C-MAPSS counts as four) — acquire the data, then answer one question per
dataset with numbers: **how much entity history actually exists, and how old is the information that
decides an event?**

⛔ **SMAP, TEP and VIX are out of scope.** Do not download them, do not audit them, do not mention
them in your results. VIX fits its label threshold on the test split; SMAP and TEP ship event
definitions their own authors call placeholder.

## 2 · ACCEPTANCE CRITERION (one, mechanical)
**`.claude/outputs/hepa-longhorizon-triage/audit.json` exists and contains, for each of the 11 dataset
rows, EITHER the four measurements of §3 plus a criterion-1 verdict, OR an explicit
`{"status":"NOT_ACQUIRED","reason":"..."}` record.**
A dataset you could not obtain is **NOT_ACQUIRED with the reason**. It is never a zero, never a null,
never "failed". Partial acquisition is fine and expected; silent omission is not.

## 3 · THE FOUR MEASUREMENTS (per dataset)
Use **HEPA's own loader** for splits, channel selection, normalisation and label construction — clone
the repo (§6) and import it. Do not reimplement a loader; a reimplementation measures your code, not
theirs. Where HEPA's loader defines entities (engine id, station, channel-group, or "one continuous
series"), use its definition and **write down which it is** — for several of these datasets the whole
answer turns on whether the series is one entity or many.

1. **Per-entity series length.** n_entities, and the min/median/p90/max length in native steps.
2. **Event-history coverage.** Fraction of labelled events whose *own entity's* history preceding the
   event exceeds **512 / 2048 / 8192** steps. Three fractions.
3. **Time-to-previous-event** distribution (median, p90) in native steps, per entity.
4. **Time-to-last-similar-regime.** Define "similar regime" ONCE, mechanically, and state the
   definition in your report: nearest prior window (same entity) whose standardised channel-mean
   vector is within the 10th percentile of all pairwise distances for that dataset. If a dataset makes
   this ill-defined, say so and report items 1-3 only — do not invent a substitute silently.

Also compute and record **`max_available_context`** per dataset: the largest context rung that is
actually reachable given entity lengths and the split boundaries. The sweep spoke consumes this as a
hard ceiling, so a wrong value here silently corrupts wave 2. State how you derived it.

## 4 · THE KILL YOU CAN FIRE ALONE
**Criterion 1 of the pre-registered kill rule: a dataset whose entities rarely exceed 512 steps cannot
carry a long-history claim — mark it DROP on the audit alone.**
Make "rarely" mechanical and declare it before you compute it: **DROP if <10% of labelled events have
>512 steps of own-entity history.** Emit `verdict: "DROP_ON_AUDIT" | "PASSES_AUDIT"` per dataset.
This is the whole point of running you first — every DROP you fire saves a GPU sweep we then never buy.

The PREREG predicts all four C-MAPSS rows DROP here. **Check it, do not assume it.** If C-MAPSS
engines do exceed 512 cycles, that is a headline finding and the prediction was wrong — report it
plainly in those words.

## 5 · FILES YOU OWN (nothing else)
- `.claude/data/hepa/**` — all staged data. **Never committed, never backed up.**
- `.claude/outputs/hepa-lh-audit.md` — your report.
- `.claude/outputs/hepa-longhorizon-triage/audit.json` — the artifact wave 2 consumes.
- `.claude/outputs/hepa-longhorizon-triage/PROVENANCE.md` — see §6.
- `.claude/scratch/hepa-lh-audit/**` — scratch.
⛔ You do NOT own and must not write: `PREREG.md`, anything under
`.claude/outputs/hepa-longhorizon-triage/code/` (that is the tripwire spoke's), any tracked file.

## 6 · DATA STAGING RULES (these are program rules, not suggestions)
Datasets live in `.claude/data/hepa/<name>/` and are **never committed and never backed up**. They are
re-downloadable, so **the provenance record IS the backup.** For every dataset write into
`PROVENANCE.md`: source URL, date fetched, licence (or **"NOT STATED"** — several genuinely are),
file names, byte sizes, and a **sha256 per file**. A dataset without a provenance row does not exist.
Total is well under 1 GB across all nine — this is a link-rot and registration-wall problem, not a
disk problem. GECCO (challenge site) and BATADAL are the two most likely to resist; if one does, that
is a NOT_ACQUIRED row and you move on rather than burning the budget on it.

**Clone HEPA to `.claude/hepa-src/`** (`github.com/Forgis-Labs/HEPA`). It is **CC BY-NC-SA 4.0 —
non-commercial and share-alike**. It must never enter a tracked file; share-alike would propagate into
our repo. Record the clone's commit hash in `PROVENANCE.md` — every number downstream is relative to it.

## 7 · SEQUENCING OBLIGATION (the other spoke is blocked on you)
**Stage C-MAPSS FD001 FIRST** (smallest, most reliably available), and the moment its loader returns
data, write the marker file **`.claude/data/hepa/cmapss/READY`**. The tripwire spoke is waiting on
exactly that path to validate its harness. Do this before auditing anything.

## 8 · WHAT WOULD PROVE THE IDEA WRONG
The idea under test is that this suite contains datasets needing >512 steps of history. **You falsify
it by finding that all eleven rows DROP on criterion 1** — that every entity in the suite is short, or
every event's precursor is recent. That is a clean, publishable negative and it kills the long-horizon
section before we spend a GPU-hour on it. Report it as the result, not as a disappointment.

## 9 · REPORT
`.claude/outputs/hepa-lh-audit.md`, per AGENT_PROTOCOL §5. One table, one row per dataset: n_entities,
length quantiles, the three coverage fractions, time-to-previous-event, time-to-similar-regime,
max_available_context, verdict. Then: which PREREG §5 predictions your audit already settles, and the
NOT_ACQUIRED list with reasons. Every number traces to a file under your owned paths — quote the path.
⛔ No CLU arm, no CLU number. Under ~2 pages plus the table.
