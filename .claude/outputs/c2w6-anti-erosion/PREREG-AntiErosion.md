# PREREG — C2W6 "Protect the memory" (anti-erosion: P1 · I1 · I2)

**Filed 2026-08-05 by the [C2W6] Hub, BEFORE any harness cell has run.** Binding scope: charter
ADDENDUM 6 §A21 C2W6 row; specification §A20.6. Convention: this file is never edited after the
first cell runs — corrections and extensions go in dated ADDENDA below the line, filed before the
cells they govern (the `PREREG-Bprime.md` discipline).

**The mechanism under attack (N223, measured):** the write is differentiably unrolled inside the
outer step (`CluStoreCell.write`, `chlu/core/blocks.py` ~l.1256–1351), so the outer LM loss
backpropagates through the write into φ and the store's initial-atom leaves; under a net-cost store
the optimizer uses that channel to teach the writer to stop writing (fresh-write depth → ~1e-63
after 200 outer toy steps at the shipped config — probe §7 R3; pre-fix ψ). The ψ payload residual
measurably RESISTS it (paired depth 1.9×/4.7×/3.2× on 3/3 seeds, psires §5) — i.e. the one channel
P1 severs is also the one that gave wells "a reason to exist." That tension is what this wave
measures, not assumes.

## 1. The rig (design rule, binding)

FULL block from the CSF3 runs with **exactly one capability toggled**: the CSF3 **run-2 config**
(`atom_place_radius=0.3`, `write_margin=0.6`, `psi_payload_residual=True`,
`psi_residual_source=q_star`, all stage flags TRUE) on local toy enwik8 (0.16 M: `d_model 64`,
`n_layers 2`, `seq_len 512`, `batch 4`, `chunk 32`), **± `erosion_partition` (P1)**. 3 paired
seeds (0/1/2) on every claim cell. Main curve at w4 (⚠ monitor #13/N94 demotes it — declared) plus
one **w40 confirmation cell** at N94's floor, shorter horizon. No isolated-arm studies; the
residual-off pair is permitted only as a labelled diagnostic rider inside the full-system rig
(intervention §8.1).

**Erosion curve** = per-well fitted depth (the pilot's convention: median fitted well depth at the
item's own site on the launch manifold) vs outer step, sampled every `monitor_every=25`, each well
tagged with its last-write step so fresh-write depth and post-write decay are separable. Horizon:
**1000 outer steps** (5× the banked 200-step horizon).

## 2. Kill-conditions (built FIRST — they are the harness's first tests, before any science cell)

- **K1 — partition integrity (build gate):** with P1 ON, `‖∂L_outer/∂θ‖` is **exactly 0.0
  bitwise** for every declared depth-determining leaf (initial-atom `amp`/`log_width`/`centers`
  leaves; any trainable write-path parameter), measured by a gradient probe, not asserted. Any
  leak ⇒ the arm is invalid; no science cell runs until K1 is green.
- **K2 — bit-identity OFF (build gate):** `erosion_partition=False` ⇒ outputs AND gradients
  bit-identical to the shipped block (the coefficient-zero precedent). Fails ⇒ nothing is valid.
- **K3 — bpc harm (gate leg, from the charter):** paired bpc(P1 ON − OFF) worse beyond 2 SE
  (3 seeds) ⇒ the component FAILS the CSF3 run-3 gate regardless of how well depth is protected.
- **K4 — relocation detector (§A20.6 caveat, mandatory):** if P1 protects depth but the store
  stays useless (live−blank at float32 AND memory-deleted margin unchanged), the collapse has
  relocated into the block's other weights — the gate is NOT earned on depth alone, and the arm's
  verdict is "protected-but-useless," quoted with the system-level-swap caveat.
- **K5 — I1 is not a hidden capacity increase:** the refresh mechanism must be budget-capped
  (amp ceiling from the decay law; byte ledger unchanged on every arm). A depth floor that lets
  rewrites deepen without bound is a ledger violation, not a fix.

## 3. Registered numeric predictions (before any cell runs)

Banked anchors: untrained run-2-config depth per seed 0.0727/0.1490/0.2250; after 200 outer steps
residual-off 0.0011/0.0582/0.0373, residual-on 0.0021/0.2732/0.1210 (psires §5); shipped-config
collapse 0.0288 → 4.95e-63 (probe §7).

- **E1 (erosion continues without P1):** partition-OFF depth at step 1000 ≤ **0.3×** its own
  step-200 value on ≥2/3 seeds (band 0.02×–0.5×). The residual slows R3; it does not stop it.
- **E2 (P1 flattens the curve):** partition-ON depth at step 1000 ≥ **0.7×** the post-write depth
  net of designed decay (band [0.5, 1.05]), 3/3 seeds; the residual slope beyond the designed
  decay law is consistent with 0.
- **E3 (bpc, the gate's second leg):** paired Δbpc(ON − OFF) within **±0.01** at 1000 toy steps
  (both arms 4.55–4.65; token count barely past unigram — paired margins only).
- **I1-a (baseline rewrite audit, guard OFF):** rewrites into an occupied well REDUCE its fitted
  depth in **10–40 %** of rewrite events (band 2–60 %; the interference channel, collapse modes
  #9/#12 — distinct from optimizer erosion and measured separately from it).
- **I1-b (guard ON):** depth-reduction events = **exactly 0** by construction; when no violation
  would occur the write is bit-identical to the unguarded write.
- **I2 (the Head's hypothesis, registered direction):** on the partition-OFF arm, Spearman
  **ρ(well usefulness, erosion rate) ≥ +0.5** (most-useful wells erode fastest; usefulness = read-
  selection frequency and loss-contribution proxy, both reported). Refutation branch registered:
  ρ ≤ −0.3 means the optimizer preferentially prunes USELESS wells and the quotation caveat on
  depth-as-importance can be lifted early; |ρ| < 0.3 is "no usage structure," caveat stays.
- **P-residual interaction (the §A20.6 tension, registered):** severing the outer→write channel
  (P1 ON) removes the residual's depth-protection mechanism, so partition-ON depth protection must
  come from the partition itself; predicted: partition-ON final depth ≥ residual-only final depth
  (0.132 banked) on ≥2/3 seeds. If partition-ON depth COLLAPSES below the partition-OFF arm, P1 is
  disproved as specified (protecting the write while starving it of its one useful gradient) and
  P3's coefficient form gets priced — a finding, not a patch.

## 4. The CSF3 run-3 gate (charter verbatim, operationalized)

The component earns the run-3 config slot **iff** the erosion curve flattens (E2 met: ≥0.5× band,
3/3 seeds, while the OFF arm decays per E1) **with bpc not worse beyond 2 SE** (K3 green),
multi-seed, K4 not fired. Anything less: the component is re-priced (P3 or P2/C2W8), never
silently shipped.

## 5. Standing caveats carried

⛔ Every w4 number is monitor-#13-demoted (N94); the w40 cell is the undemoted confirmation.
⛔ Depth is NOT quotable as feature importance until I2 reports (charter §A21, active now).
⛔ No toy number is ever a pilot-scale number. ⛔ γ statements are read-budget-scoped.
⛔ "CLU-former" is a placeholder. ⛔ Tier-appropriate control: the system-level swap detects
relocation (K4); the settle-deleted launder is an inherited diagnostic only.
