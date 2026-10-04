# Task: r2-geometry-revival — adjudicate the ceiling's mechanism, then the one-flag revival (w25)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/r2-geometry-revival.md` · **Branch:** `agent/experiment-engineer/r2-geometry-revival`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (§3.2 worktree · **§7 dial declaration**) · `.claude/outputs/lattice-capacity-theory.md` **§4.2 + §5.0 + §5.2 (the spec of this task — build to it)** · `.claude/outputs/write-ceiling-break.md` (the levers that failed; the arms you do NOT re-run) · `.claude/outputs/multi-seed-w23.md` §Item-3 update (**the d=8 crack**) · `negative_results.md` **N92/N96 (mechanism CONTESTED)**

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** capacity (the R2 law) — a **law about the primitive**, exempt from the masked-recall demotion, and its figure is **never framed as beating anything** (CM-23(m)).
- **Laundering control:** the designed write at matched geometry (it must keep reaching its own wall; if a lever "works" only by making the learned write more designed, that is N46 scope collapse, not a win).
- **Falsifies (per stage):** §5.0 — trained widths < ~0.18 or the minsep/width ratio varying >2× across d kills the geometric account and N96's operator reading stands. §5.2 — no movement of the wall at ≥3 seeds under an adequate budget kills the width route.
- **Does NOT falsify:** the ceiling failing to reach the designed `4·2^d` value (the 4× prefactor gap is expected); any comparison to kNN or external methods (out of scope for a primitive law).

## Why — the Head ruling on N96, and the new crack
"Dead via the write operator" was **premature**: the decisive check was never run, and the Hub's post-hoc dim run has now measured a crack a d-independent ceiling cannot explain — **d=8 K=64 PASSES at 2× atoms (0.9067±0.0068, 3 seeds) while d=6 K=64 fails firmly (0.818)**. The width-lock account *predicts* that split (sep/width 3.03 vs 2.65 against the measured 2.4–3.0 transition window). Two staged experiments settle it; the first is ~free.

## Stage 0 (≤1 hour, run FIRST, report immediately) — the §5.0 width dump
Measure the **trained `log_width` distribution at the stored sites** on w23 `dimension-aware-budget` cells (re-run one cell per d dumping widths if no checkpoint survives — cheap). **Pre-registered predictions (the theorist's, adopt verbatim):** median effective well width **≥ 0.28** (the write does not narrow the atoms below init 0.30), and **minsep(K_wall)/width ∈ [2.4, 3.1] at every d**. Include the new d=8-K=64-pass and d=6-K=64-fail cells: the account predicts their trained sep/width straddle the transition window.
- **If FALSIFIED** → the geometric account is dead; N96's operator reading stands; **STOP after Stage 0**, report, and hand the mechanism question to the theorist. The task is then complete and *valuable* — the contested flag on N92/N96 resolves.
- **If it SURVIVES** → Stage 1.

## Stage 1 — the one-flag revival (§5.2)
`atom_init_width: 0.30 → 0.15` (matching the designed width, which sits at the query-noise floor `σ_q ≈ 0.15`), with the atom budget raised for coverage (`min_atoms_base ×2^{d/2}` per the theorist's spec). Run the `K_learned(d)` discriminator at d ∈ {4, 5, 6, 8} on the w23 harness, **≥3 seeds**, N92 budget-adequacy (2×-atom re-check at every first-fail — a stall under an inadequate budget is not a wall).
**Pre-registered predictions (theorist's):** `K_learned(4)` rises **16 → 64–128**; `K_learned(6)` rises **32 → ≥128** — i.e. the "ceiling" moves by ~`2^d`. Falsifier: no movement ⇒ the ceiling is an operator limit after all; report plainly, contested flag resolves the other way.
⚠ **Watch the noise floor:** width 0.15 = σ_q. If capacity rises but noise robustness collapses (basins no longer contain the jittered query), report the capacity↔robustness trade as the finding — that is the honest law, not a failure of the task.

## Stage 2 (conditional on Stage 1 moving the wall) — the d-sweep for the R2 figure
Extend to d ∈ {2…10} at the surviving width; produce the log-linear `K_learned(d)` figure with the designed line, per-point adequacy, ≥3 seeds. **The figure is a law about the primitive; its caption never says "beats."** Precision rule (Hub ruling on the ch-8 critique): designed = `4·2^d`, learned = `2^d` — a 4× prefactor gap; never write "exactly the designed rate" without it.

## Acceptance
PREREG before Stage 0 (adopt the theorist's numbers as the registered predictions; add your own bands). Staged reporting: Stage-0 verdict lands even if it kills the task. Tests green for any config knob added (all sites + `save_config`); `ruff` clean; worktree; echo the DIAL DECLARATION in the report.

## ⚠ Standing traps
- Do NOT re-run write-ceiling-break's failed levers; this task tests the *width/geometry* axis those levers never touched.
- CM-22(j): never quote √2/`d^1.62`. N96: never write "the ceiling is the write operator" as settled — this task is what settles it.
- If Stage 1 works only with formula-placed centers or hand-set per-item widths, that is a designed write in disguise (N46) — declare the fairness category of every knob.
