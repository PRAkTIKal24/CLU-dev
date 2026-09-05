# pilot-placement-probe — does localized placement wake the in-block store? (+ two B′ riders first)

**Campaign 2, wave C2W5. Agent:** experiment-engineer. **Small. Worktree 2 of ≤3, spawns
immediately.** Branch `pilot-placement-probe`, scoped worktree. Writes
`.claude/outputs/pilot-placement-probe.md` + `.claude/outputs/pilot-placement-probe/*`. Charter
**ADDENDUM 4 §A19 task 5**, per ruling **A18.4**.

⭐ **STATUS RULE (A18.4, verbatim intent):** this probe **informs the CSF3 submitted config; it
NEVER gates the CSF3 commitment** — the scale run happens regardless (Head-submitted; *"a good
real-data score at scale is NON-NEGOTIABLE"*; nulls re-price the route, they do not close it). The
probe exists so the submitted config is the best-informed one.

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/outputs/cluformer-pilot.md` **in full — §5
(the inert-store mechanism: live = blank at float32; 16× write budget buys zero acquisition; depth
saturates at 0.045 vs shipped 0.46–0.80; the three candidate fixes §5.3)**; charter **§A16.2** (the
pilot's verified findings) · **§A17.1** (the three fixes are tier-ii organizer tooling) · **§A18.4**;
`scripts/csf3/job_gpu_cluformer.sh` (the job script your recommendation block targets); the
`[C2W4-CLOSE]` §10 entry. ⚠ **"CLU-former" is a PLACEHOLDER NAME — never bake it into any artifact;
say "the tier-iii block".**

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** none — **instrument/diagnostic** (tier-iii mechanism isolation). ⛔ Nothing
  from this probe is a paper number; it is config evidence for the CSF3 run and design evidence for
  tier ii.
- **Control:** the pilot's own arms reused — **live vs blank vs memory-deleted**, paired seeds; the
  in-block acquisition self-probe (chance = the reference).
- **Falsifies (the probe's own hypotheses):** H1 (localized init) fails if `atom_local_radius` at
  its designed band leaves acquisition at chance AND live = blank at float32. H2 (trajectory write
  term) same bar. Both failing = the placement hypothesis is refuted at toy scale — a FINDING that
  re-prices (not closes) the route, per A18.4.
- **Does NOT falsify:** anything about tier ii or the scale run; toy-scale GRU superiority
  (pre-registered, already measured); monitor #13/N94 non-promotability is inherited, stated on
  every reading (4 inner write steps vs floor 40 — quote it beside every number).

---

## 0. FIRST ACT — the two B′ riders (~15 min total, BEFORE the probe; they feed `draft-r3`)
Run these from the main repo read-only (outputs are gitignored artifacts; no code changes needed —
if a script edit IS needed, it goes in your worktree):
1. **The n=9 full-column re-aggregation** (F3 follow-up 2, ~10 min): pool the null/launder/ledger
   columns to uniform n=9 from the existing `{run400,seeds3to8,repro_c2w4*}/
   exp_bprime_rivals_metrics.json` artifacts → `.claude/outputs/pilot-placement-probe/
   n9_full_columns.json` + a paper-ready table. This retires draft App. I.1c's "un-aggregated
   columns" caveat and the mixed-n labelling.
2. **The labelled deltanet byte-frontier row** (F3 follow-up 4, ~5 min): the frontier row for the
   newly-rescued deltanet, explicitly labelled **byte-frontier column, never a dividend family**
   (§A14.2) → same output dir.
Ping the Hub when both land (the `bprime-draft-r3` writer is gated on them + the referee report).

## 1. The probe (in the pilot's toy rig, 0.16 M scale, hours not days)
Test the pilot's §5.3 hypotheses **in the registered order**:
1. **H1 — localized atom init** (`atom_local_radius` at its designed band): atoms seeded near the
   φ-image of early chunks instead of scattered at scale 1.0. The pilot's mechanism says few
   unrolled steps cannot gather scattered atoms — localization removes the gathering problem.
2. **H2 — the trajectory write term** (second, only if H1 alone is insufficient or for the
   interaction row): the C2W2 machinery as tier-ii organizer tooling (§A14.1).
   (The ψ payload residual is third-priority; build only if H1+H2 both fail and budget remains.)
**Success signal (pre-registered here):** the store stops being inert — (a) in-block acquisition
self-probe off chance (> chance + 2·SE, 3 paired seeds), (b) live ≠ blank at float32 resolution,
(c) well depth leaves the 0.045 saturation toward the shipped 0.46–0.80 band. Report all three per
arm; partial wakings (depth moves, acquisition doesn't) are mechanistically informative — report
the pattern, don't binarize.
**Discipline:** ≥ 3 paired seeds per cell; the φ-gain calibration (RMS address norm = ball radius)
is the declared anti-collapse init; γ statements read-budget-scoped; declared NOT-RUNs never nulls.

## 2. The CSF3 recommendation block (the deliverable the Head submits with)
Your report ends with a **recommended config block for `scripts/csf3/job_gpu_cluformer.sh`**:
- atom init scheme + `atom_local_radius` (from H1/H2 evidence), write budget (N94's floor 40 inner
  steps — the toy 4-step reading is non-promotable; say what the scale run should use), read budget
  + γ scoping (chunk-granularity ρ_conv caveat), φ-gain calibration, monitor set + `monitor_every`
  (monitor #6 needs a longer window or smaller `monitor_every` to be applicable in-block).
- ⚠ **The plan-pass problem, priced:** 77–84 % of CLU wall-clock is the Python plan pass — the GPU
  will idle behind it at pilot scale. Quantify the expected GPU idle fraction for your recommended
  config. **If a vectorisation of the plan pass is cheap (≤ half a day), do it in your worktree and
  measure the speedup; if not, price it for the Head** (it then needs its own spoke before the GPU
  run wastes A100-hours).

## 3. Acceptance
Riders delivered (§0) · H1 (and H2 if reached) measured against the pre-registered success signal,
3 paired seeds, monitor #13 caveat on every reading · the CSF3 recommendation block complete ·
tests green on your branch · declared NOT-RUNs listed.

**File ownership:** you own the probe's config/experiment files + any plan-pass vectorisation in
`controller.py`-adjacent code you declare, + your tests. ⛔ Do NOT touch the factored-store family
files, `monitors.py` (`orgdiv-cat-test`'s), or anything `orgdiv-null-arms` declares. Declare your
exact list first. **Git:** branch + scoped worktree; never push `origin`; `clu-dev` only.
Report → Hub, spawn nothing.
