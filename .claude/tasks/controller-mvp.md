# Task: controller-mvp — the minimum viable hand-coded controller on a designed store (w23)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/controller-mvp.md` · **Branch:** `agent/experiment-engineer/controller-mvp`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (§3.2 worktrees — 4 parallel engineer tasks this wave) · `.claude/outputs/clu-controller-spec.md` (the theorist's conditions + three decision rules + what a controller provably cannot fix — build to THIS spec) · `.claude/outputs/sequential-write-interference.md` (N74: the gate cannot fire on the w20 geometry — spacing 1.4142 vs d_safe 1.10; on a DESIGNED store the gate is decisive, retention 1.000 vs 0.16 at a capacity price ≈ the packing bound) · `.claude/outputs/address-space-dimension-scaling.md` (the repaired packing law — sets the geometry so the gate CAN fire)
- **⭐ Purpose:** every verb in the primitive vision — *decide, add, trash, evict* — is a controller verb and none is built (roadmap v0.6). Continual learning (the Head's chosen HG1 family) is *precisely* a write-policy problem. Hand-coded is explicitly OK (Head ruling, w20: differentiability is not a first-paper requirement — the debt is stated, not hidden).

## Item 1 — build the MVP (hand-coded, designed store)
Per `clu-controller-spec`'s three decision rules: **admission** (novelty test against existing wells using the d_safe/packing geometry — N74's lesson: choose the address geometry so admission is *decidable*, spacing vs d_safe reported), **placement** (derived address — write where the write-operator's locality holds), **eviction/decay** (a budget policy: when full, evict by staleness or scheduled decay — the per-item retention machinery from w22 gives you permanent + leaky wells in one store). Config-driven; flags documented; no learning anywhere in the loop.

## Item 2 — the N75 rematch
Re-run the `sequential-write-interference` sequential-parametric-write benchmark (K up to 64, same seeds/protocol) with the controller ON vs OFF: retention-vs-K for **CLU+controller vs the w21 gru/mlp/attn lines** (reuse their numbers; re-run only if the protocol demands it). w21 verdict was CLU WORST of four (0.16 at K=64, learned-everything). The question: does designed-store + controller move CLU from worst toward best, and at what admitted-capacity price (report admitted/K against the packing bound, as N74 did: 6.0±0.9/16 ≈ 6.1 predicted)?

## Item 3 — the honest accounting
The controller *refuses* items the baselines accept — retention-per-admitted-item vs retention-per-offered-item are different metrics and both go in the table (the abstention-vs-accuracy trade is the field's oldest trick; state it before a referee does). Also state plainly what the controller cannot fix (per the theorist's no-go list) and the cost of the admission test per write.

## Acceptance
The retention-vs-K table (controller ON/OFF, four-primitive comparison, both per-admitted and per-offered), the admitted-fraction-vs-packing-bound check, geometry so the gate demonstrably CAN fire (spacing/d_safe reported), config flags documented, pre-registered expectations stated before running. Tests green. ⚠ If controller-ON still loses to the GRU on per-offered retention, that is the headline — report it plainly; it scopes the CL entry design (`continual-learning-recon`) before we bet a benchmark on it.
