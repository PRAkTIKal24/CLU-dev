# pilot-ttt-nan-and-d5-wiring — experiment-engineer

**Spawn now (Head launches).** No worktree needed if none are free; this is a small, surgical task.
Two defects found by the Hub's integrity pass on the FIRST landed tier-iii artifact
(`.claude/outputs/cluformer-pilot/csf3_outs/pilot_pilot_seed0_S4.json`, run 1 / seed 0).
Read §10's `2026-08-06` block first — it has the full integrity-pass numbers.

⛔ **THE HARD CONSTRAINT, READ FIRST.** Five more legs are running or queued on CSF3 **against the
current code**, and four of them hold banked resume journals + `ckpt_{arm}.eqx`. Your change must
NOT invalidate them: `load_journal` refuses a journal whose config fingerprint differs **by key**.
So (a) do not rename/remove any existing config key, and (b) any new knob must sit outside the
journal's compared fingerprint — **demonstrate resume-accept in a test**. If a fix cannot meet
this, STOP AND REPORT with the options; do not spend the banked training.

## Defect 1 — the `ttt_matched` arm goes NaN at step 135/4000 (BLOCKING the rival column)
`arms.ttt_matched.train.loss_history` reads 5.708 … 4.511 at step 134 then **NaN from step 135
onward**, never recovering ⇒ `static`, `dyneval`, and both `margin_vs_clu_*` are NaN. The other
four arms (`clu_store`, `gru_matched`, `none`, `echo`) are clean end-to-end — no NaN anywhere —
so this is TTT-specific, not a global instability.
1. **Diagnose, don't paper over.** Reproduce at the smallest scale that shows it (toy first;
   escalate shape toward pilot only as needed). Name the mechanism: LR too high for the pilot
   shape? inner-loop update divergence? a missing normalisation/clip that the toy shape hid?
   fp32 overflow in the TTT inner state? Report the evidence for the mechanism you name.
2. **Fix at the mechanism**, with the cheapest decision-inert lever. ⛔ **The TTT arm is a
   published rival column** — if the only fix changes TTT's *configuration* (LR, clip, init),
   that is a claims-relevant change: report the proposed value, the evidence, and **STOP for a
   Hub ruling** before making it the default. A pure *bug* fix (a genuine numerical error in
   the arm's implementation) you may land, with the bit-identity gate below.
3. ⚠ Note for scoping: the toy-scale runs never showed this, so the toy gate alone will not
   protect the rerun. State explicitly what evidence would convince us the fix holds at pilot
   shape without burning another 22 h leg.

## Defect 2 — D5 (`anytime_curve`) never ran on ANY launch: the flag is not wired
`exp_cluformer_pilot.py` gates the anytime curve behind `--d5` (`action="store_true"`), and
`scripts/csf3/job_gpu_cluformer.sh` builds `EXTRA` from `STEPS/ARMS/MEM/STORE/SET/RESUME` **only**
— there is no `D5` passthrough. So `with_d5=False` on every attempt, and no `anytime_curve` key
exists in the landed record or its phase list. D5 is a **pre-registered deliverable** and the
"NOTHING IS CUT" ruling (§A18.4/A19) was violated by plumbing, not by decision.
1. Wire it: `[ "$D5" = "1" ] && EXTRA="$EXTRA --d5"` (match the existing `RESUME` idiom exactly),
   documented in the script header beside the other exported vars.
2. ⚠ **Cost check, and report it BEFORE we resubmit:** D5 evaluates 5 read-budget settings; the
   eval block is where the host-RAM OOM lives (`dyneval` alone demanded ≥107 GB — §10 2026-08-05).
   Measure/project D5's own host footprint at pilot shape and say whether it fits the `-G 2 -c 24`
   (~251 GB) envelope **on top of** the phases that already run. If it does not, say so plainly —
   the Head then rules on `-G 2` + D5-in-a-separate-resume-pass vs anything else.
3. ⭐ Because `RESUME=1` skips completed arms, D5 can likely be obtained by **re-resuming the
   finished legs with `D5=1`** rather than retraining — confirm whether the resume path will
   re-enter the `clu_store` eval block and run only the missing phase, or whether the journal
   marks the arm done and skips it. **This is the difference between ~free and 22 h/leg; it is
   the single most valuable thing in this task.** Report the mechanism either way.

## Gates (both mandatory)
- **Toy bit-identity:** old-vs-new `run_pilot` at toy S4 with `D5=0` and the TTT fix inert ⇒
  **0 differing leaves** in the final JSON. (The D5 wiring must be a no-op when unset.)
- **Resume-accept test:** a journal written pre-change is accepted post-change (the hard
  constraint above), demonstrated in a test, not by assertion.
- Suite green (expect 1363 + yours), ruff green, scoped branch, report to
  `.claude/outputs/pilot-ttt-nan-and-d5-wiring.md` with proposed §7/§10 updates.

## Acceptance
TTT's NaN mechanism named with evidence + fixed (or stopped-and-reported if the fix is
claims-relevant) · D5 wired with the `RESUME` idiom + its host-memory cost projected against the
251 GB envelope · **the "can D5 be obtained by re-resuming a finished leg?" question answered
mechanically** · both gates green · resume compatibility demonstrated.
