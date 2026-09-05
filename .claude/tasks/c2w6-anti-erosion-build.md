# c2w6-anti-erosion-build — P1 stop-gradient partition · I1 refresh monotonicity · I2 telemetry

**Campaign 2, wave C2W6 ("Protect the memory"). Agent:** experiment-engineer.
**Worktree 1 of ≤3.** Branch **`c2w6-anti-erosion`** (the wave branch, from `main @ 104ca19`),
worktree `../CHLU-c2w6`. Writes `.claude/outputs/c2w6-anti-erosion.md` + artifacts to
`.claude/outputs/c2w6-anti-erosion/`. Budget ≈ 1 day build + ≈ 6–9 h of measured runs (price the
w40 cell before running it; cut seeds there before cutting the cell, declared).

**Binding documents, read first:** charter **§A20.6 in full** (your specification) + **§A21 C2W6
row** (your gate) · `.claude/outputs/c2w6-anti-erosion/PREREG-AntiErosion.md` (**the wave prereg —
your kill-conditions K1–K5 and predictions E1–E3/I1/I2 are already registered; you implement them,
you do not re-derive them**; anything you need to add goes in a dated addendum filed BEFORE the
cells it governs) · `.claude/outputs/psi-payload-residual.md` §5 + §11 (the R3 evidence + the
residual-resists-R3 finding) · `.claude/outputs/pilot-placement-probe.md` §7 + §10 row 10 ·
the `[C2W5-HANDOVER-2]` §10 entry (CSF3 state; you touch none of it).

**The mechanism you are killing (N223):** `CluStoreCell.write` is differentiably unrolled inside
the outer step, so the outer LM loss reaches φ and the initial-atom leaves THROUGH the write; under
a net-cost store the optimizer teaches the writer to stop writing (depth → 1e-63 at 200 toy
steps). Forgetting must happen through designed channels (decay law, eviction, trash region) —
never as an optimizer side effect.

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** lifetimes/isolation (training-time protection of written content) — tier-iii
  component build. ⛔ No paper number; CSF3 run-3 config evidence + the wave's ablation rows.
- **Laundering control:** live vs blank vs memory-deleted, paired seeds, on BOTH arms; **K4** (the
  relocation detector) is the tier-appropriate control and is mandatory on the gate verdict.
- **Falsifies / does not falsify:** per the prereg §3 (E1–E3, I1, I2, the P-residual interaction).
  A partition that fails is a re-price toward P3/P2, never a silent ship.
- ⚠ Monitor #13/N94 travels with every w4 reading; the w40 cell is the undemoted confirmation.
- ⛔ Depth is not quotable as feature importance until I2 reports (charter §A21, active).

## The build (kill-conditions FIRST — K1/K2 are tests that exist before any science cell runs)

1. **P1 — `erosion_partition` flag** (ships OFF; OFF = bit-identical AND parameter-count-identical,
   the psires precedent). Semantics per §A20.6: the outer objective gets a **stop-gradient at the
   write boundary** — written content and the write machinery's depth-determining parameters
   (initial-atom `amp`/`log_width`/`centers` leaves; any trainable write-path parameter) are owned
   by the write/decay/evict channels only. Implementation freedom is yours (stop_gradient on the
   write's returned `StoreState`, an `eqx.partition` freeze of the atom-init leaves from the outer
   optimizer, or both) — **the acceptance is K1's bitwise-zero leaf audit, not the mechanism.**
   ⚠ The read-side path (loss → ψ → read → γ/M, and φ's QUERY gradient) must survive — you are
   severing the write channel, not the read channel; a gradient probe distinguishing the two is
   part of K1's evidence.
2. **I1 — refresh-on-rewrite monotonicity.** First MEASURE (guard OFF): instrument rewrite events
   (admitted write into an occupied slot) with fitted depth before/after — the I1-a event rate.
   Then the designed guard (flag, ships OFF): a rewrite never reduces the well's fitted depth —
   refresh/deepen up to budget, **K5-capped** (amp ceiling from the decay law; byte ledger
   unchanged; a violation-free write stays bit-identical — I1-b). This is the interference channel
   (#9/#12), adjacent to and measured separately from optimizer erosion.
3. **I2 — usage-vs-erosion telemetry** (instrument, always on in the harness, decision-inert to
   the model): per-well, per monitor window — fitted depth · last-write step · read-selection
   count (plan lanes) · an outer-loss gradient-magnitude proxy into the well's atoms · at 3–4
   checkpoints a leave-one-well-out probe-batch bpc (the loss-contribution measurement). Artifact:
   the per-well time series the analyst adjudicates ρ(usefulness, erosion rate) on.
4. **The erosion-curve harness:** run-2 config ± P1, 3 paired seeds, **1000 outer steps**, depth
   sampled every 25; wells tagged by last-write step (fresh-write depth vs post-write decay
   separable — designed decay must not be scored as erosion). Cells: `p1_off` · `p1_on` ·
   `p1_on_i1_on` (the shipped candidate) · **`w40` pair** (± P1 at `write_inner_steps=40`, shorter
   horizon — 400 steps — priced first, seeds cut before the cell) · diagnostic rider: the
   residual-off pair at w4 only (labelled DIAGNOSTIC — the §A20.6 "erosion intrinsic vs symptom"
   2×2 corner; never a claim cell).
5. **Gate adjudication, mechanical:** an `aggregate()` that applies prereg §4 verbatim and prints
   the run-3 verdict (`EARNS_SLOT` / `FAILS_K3` / `FAILS_FLATTEN` / `K4_RELOCATED`), plus the
   run-3 flag block (run-2's exact config + only your flags, via `--mem/--store/--set`; zero
   module edits) — emitted EITHER WAY, labelled with the verdict; the Advisor decides promotion.

## File ownership (declared, zero-conflict discipline — C2W7 is live in wt2 concurrently)
- **YOURS:** `chlu/training/train_cluformer.py` (the gradient paths / partition plumbing / outer
  step) · `chlu/core/blocks.py` (`CluStoreCell` + `StreamMemoryConfig` hunks, additive) ·
  `chlu/experiments/exp_anti_erosion.py` (new) · `tests/test_anti_erosion.py` (new) ·
  `scripts/csf3/job_gpu_cluformer.sh` ONLY if a new env passthrough is needed (flag-path
  precedent).
- ⛔ **NOT YOURS:** the factored-store / read-fix / launch-head files (**C2W7's, wt2**) ·
  `chlu/core/psi_readout.py` (AttentionPsi quarantine) · `chlu/core/monitors.py` (C2W7 adds the
  launch-collapse row; if I1 wants a monitor row, put the check in your experiment + blocks and
  flag the monitor-registry row for the Hub) · `chlu/config.py` · `chlu/core/{clu_system,
  admission,placement,memory_potentials,controller,clu_controller,soft_certificate,
  implicit_grad}.py` · `chlu/eval/**` · anything CSF3-in-flight (the running jobs, their OUT
  dirs, `outputs/cluformer-pilot/**`).

## Acceptance + hygiene
K1/K2 green as TESTS before any science cell · erosion curves delivered per prereg §1 · E1–E3,
I1-a/b, the P-residual interaction adjudicated against the registered bands, mechanically · I2
telemetry artifact complete (the analyst adjudicates ρ; you report the raw series + your own ρ
computation labelled provisional) · gate verdict + run-3 flag block emitted · new tests green +
the affected modules' regression suites (`test_blocks`, `test_cluformer_pilot`,
`test_placement_probe`, `test_psi_residual`) · ruff green · full-suite run before merge is the
Hub's gate, not your claim · report ends with proposed §7/§10/N-registry updates · reconciliation
list in the first 10 lines if anything upstream is contradicted · ⛔ never push `origin`; do not
push `clu-dev` (the Hub pushes after merge) · worktree left for Hub review.
