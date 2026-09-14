# hepa-lh-verdict — price every rung against a same-byte lookup, then apply the kill rule

**ICLR long, wave 2. Agent: experiment-engineer. CPU only, local.** NO worktree, NO branch, NO commit.
Runs **concurrently with `hepa-lh-sweep`**; ownership disjoint (§5). **Budget: ~1.5 days.**

## ⛔ READ FIRST, IN FULL
`.claude/outputs/hepa-longhorizon-triage/PREREG.md` (frozen) · `.claude/outputs/hepa-lh-tripwire.md`
**in full** — you inherit that harness and its byte ledger · `.claude/outputs/hepa-lh-audit.md`
Tables A/B · `.claude/AGENT_PROTOCOL.md` §2, §5.

## 1 · THE ONE JOB
Decide **KEEP / DROP / HOLD** for every dataset, by (a) running the matched-byte tripwire at **every
rung the sweep runs**, and (b) applying the pre-registered kill rule to the sweep's curves.

## 2 · ACCEPTANCE CRITERION (one, mechanical)
**`.claude/outputs/hepa-longhorizon-triage/verdicts.json` exists** with one record per dataset row
carrying: verdict, the four measurements, the tripwire number at each rung, the information-age slice,
the PREREG §5 prediction and whether it held, and the artifact paths behind each number.

## 3 · MECHANICAL PRECONDITIONS — check on disk, do not judge whether they matter
Two gates, and **they gate different parts of your job**:

> **GATE A — `.claude/outputs/hepa-longhorizon-triage/PREREG_AMENDMENT.md` must exist.**
> It carries the Head's ruling on how kill-rule clause (b) reads. **Nothing in §6 may run without it.**
> If absent: do §4 (which is ungated), then report `PARTIAL` and hand back. Do not guess the reading.

> **GATE B — `.claude/outputs/hepa-longhorizon-triage/sweep.json` must exist**, and
> `preds/<dataset>_L<rung>_s<seed>.npz` for the rows you slice.
> Gates **§5 and §6 only**. If absent when §4 is done: report `PARTIAL`, hand over the rung table.

**§4 is ungated — start it immediately.** It needs only the staged data and the wave-1 harness.

## 4 · TRIPWIRES AT EVERY RUNG (ungated — start here)
Run the wave-1 harness at every (dataset, rung) the sweep covers: ETTm1 {512, 2048, 8192},
Beijing-AQ {512, 2048}, PSM {100, 512, 2048, 8192}, Weather {512, 2048}. Plus **GECCO {512, 2048,
8192}** and **MBA {512}** — those two are out of the sweep but their tripwire numbers are cheap and
they are the two rows where wave 1 already found a no-model competitor worth pinning down.
- **Recompute the byte budget at every rung** — it grows with L and the store must grow with it. A
  store fixed at the 512 size makes the 8192 rung a strawman.
- Report all three budget readings; `residual_stream` stays primary unless the Hub re-rules.
- All three trivial baselines every time (prevalence, seasonal, last-event-age), each reported even
  when weak. A suppressed weak baseline is a suppressed result.
- **Honour the deterministic-store rule**: where the pool is fully retained, report one number labelled
  `deterministic` — never a fabricated ±0.00 across three seeds. MBA is deterministic at every rung.

## 5 · THE INFORMATION-AGE SLICE (gated on B)
For each swept (dataset, rung): split ft_test events by whether the **decisive precursor** — the audit's
time-to-last-similar-regime, falling back to time-to-previous-event where the regime match is CENSORED —
is **older than 512 steps** or younger. Report h-AUROC on each slice separately, 3 seeds.
State the n in each slice beside every number. Where a slice has **fewer than 10 events**, report it as
`UNDERPOWERED (n=...)` and do not draw a conclusion from it. Weather (7 test events total) cannot
produce two usable slices; say so rather than splitting it into two meaningless halves.
**This is the sharpest test in the whole task**: if long context helps at all, it must help *here*,
on the old-precursor slice. A dataset whose curve rises but whose gain sits entirely in the
young-precursor slice has not shown a memory effect — it has shown a capacity effect, and you should
say exactly that.

## 6 · APPLY THE KILL RULE (gated on A and B)
Per PREREG §4, using the ruling in `PREREG_AMENDMENT.md` for clause (b):
- (a) longest-rung h-AUROC beats the 512 rung by **more than the seed spread** (max − min at 512).
- (b) per the amendment.
KEEP only if both. Otherwise DROP — or HOLD if the data audit was the only failure.
Emit the verdict per row **including the rows the sweep never ran**, carrying wave 1's reason forward:
FD001–4 `DROP_ON_AUDIT`; MBA, BATADAL `DROP` (no rung above the reference exists); GECCO `HOLD`
(1 ft_test event — clause (a) is not measurable, and that is a power failure, not a negative result).

## 7 · FILES YOU OWN
`.claude/outputs/hepa-longhorizon-triage/{verdicts.json, rungs/**}` · `.claude/outputs/hepa-lh-verdict.md`
· `.claude/scratch/hepa-lh-verdict/**` · **`code/**` is yours in wave 2** (the sweep spoke has it
read-only) — extend the harness, do not rewrite it, and keep the acceptance run bit-identical.
⛔ Not yours: `PREREG.md`, `PREREG_AMENDMENT.md`, `audit.json`, `sweep.json`, `preds/**` (read-only),
any tracked file. `.claude/data/hepa/**` read-only.

## 8 · WHAT WOULD PROVE THE IDEA WRONG
Wave 1 found a same-byte kNN beating HEPA on FD001 and a 24-bin daily-phase table matching the best
published baseline on GECCO. **You falsify the whole long-horizon premise by finding that the tripwire
tracks or beats HEPA at every rung on every dataset** — that would mean the suite's margins are not
evidence of learned representation and the context question is moot. Report that in your first line if
it happens. ⛔ Do not soften it, and do not go looking for a rung where HEPA wins to lead with instead.

## 9 · REPORT
`.claude/outputs/hepa-lh-verdict.md`, protocol §5. Rung table, slice table, verdict table, and for each
PREREG §5 prediction whether it held — including the ones that did not. ⛔ **No CLU arm, no CLU number.**
Under ~2 pages plus tables.
