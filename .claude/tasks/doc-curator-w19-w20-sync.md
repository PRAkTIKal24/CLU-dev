# Task: doc-curator-w19-w20-sync — process the w19 backlog and record the vision reframing (w20)

- **Agent:** `doc-curator` · **Output:** per your own protocol (edits in place) + a summary at `.claude/outputs/doc-curator-w19-w20-sync.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/handover_context.md` Running Log entries dated **2026-07-21** (all of them — the vision block, the reframing block, and the WAVE-19 review) · `.claude/outputs/clu-retrieval-demo.md` · `.claude/outputs/clu-latent-io-audit.md` · `.claude/outputs/clu-memory-architecture.md` · `.claude/outputs/scout-dynamical-memory-priorart.md`

## Why
w19 was reviewed and integrated but its documentation was **Hub-recorded, not curator-processed**. The transfer docs are the program's debugging record and must not lag the evidence base.

## Item 1 — negatives registry (`.claude/negative_results.md`)
Process the **"PENDING REGISTRATION — WAVE-19"** block: candidates **N61–N65**, the WATCH items, and the reconciliation-owed list. Assign final numbers, write them to protocol, and clear the pending block.
⚠ **Three reconciliations are genuinely unresolved — register them as open, do not resolve them yourself:**
- **The two persistence gates DISAGREE** — the audit measures CLU *winning* at n=1 (0.4545 vs 0.5673) and losing from n=5; the earlier "loses at n=1" does not reproduce. **Neither may be quoted until protocols are reconciled.** *(Note the Hub separately withdrew the raw-space gate as testing a claim we do not make.)*
- **"Single-basin collapse" is REINSTATED** by the audit in a form that survives the overflow retraction — flag for re-verification before re-promotion.
- **N7 is WRONG on the CAFE path** — `log_mass` does move (+3.56), but as common mode; the *spectrum* is inert. **Amend N7 rather than deleting it**, and record that the Head's conclusion (no timescale hierarchy) is CONFIRMED while both proposed mechanisms were wrong.

## Item 2 — claims matrix (`.claude/claims_matrix.md`)
Discharge the **"WAVE-19 MATRIX ACTIONS OWED"** block — 8 items, including the two new CM candidates, the **CM-5/N7 mechanism amendment**, the **K-dependent D1 partition** (`η_m/(η_m + K·η_k)`; K=2 → 1/21 matched to 13 digits — **the old `1/11` is wrong for multi-item settings**), the **three-regime capacity rule**, and the positioning claims to retire. Bump the header version.

⛔ **Retire these claims wherever they appear — all four are refuted, and the first three by external prior art:**
1. **"attention has no retention analogue"** — HiPPO-LegS has a principled non-decay guarantee (`O(tL/√N)`, gradient `Θ(1/t)` polynomial). A reviewer will know.
2. **"retrieval is a rollout, not a weighted sum" as our novelty** — Kong, Brewer & Lai, *Nature Communications* 2024, published location-addressable retrieval of dynamical attractors. **Must be cited.**
3. **continuity as our NTM/DNC escape hatch** — canonical NTM **and** DNC were already fully soft, continuous and end-to-end differentiable. The precedent transfers as a **warning**; 2 of 3 diagnosed DNC failure modes bite CLU's address scheme.
4. **the "8-item ceiling" as CLU's capacity** — it is a 2-D-ring artifact (`K_max ≈ 0.2·2π/σ_θ`); the packing bound is `(1+2R/w)^d`, exponential in `d`. Both w19 agents independently warned against quoting it.

## Item 3 — record the vision reframing
The 2026-07-21 entries contain the program's largest reframing to date and it must reach the transfer docs:
- **CLU is a latent information carrier, NOT a model of the observed data's dynamics.** The Hub's "primitive–task mismatch" hypothesis is **REJECTED and withdrawn**; never diagnose "prior mismatch" again.
- **CLU is a general AI primitive** — peer to MLP/GRU/Mamba/attention/DeepSVDD, not a special case of one; special cases can later be built *on* it.
- **The architectural gap:** read-in `φ` is ~identity, read-out `ψ` is handcrafted, so all representational burden falls on `V_θ` alone. This single gap predicts most of our negatives.
- **The missing piece is the CONTROLLER** — every verb in the vision (decide/triage/add/create/trash/select/combine) is a controller verb and none is built.
- **Address learning by gradient descent is DEAD**; the redirection is derived addresses + write-side restructuring + retry-as-capture.
⚠ **Also record the standing tension rather than smoothing it:** "start arbitrary and restructure" is in direct tension with **N46 / N7 / CM-5 / CM-16a** and with **D1 / D3**. The working resolution — loss-driven restructuring **plus** designed structural scaffolding, since **T3 proves a regularizer can restore diversity but never choose assignment** — is a *proposal under test*, not a settled result. Label it as such.

## Item 4 — HEP primers
Add what a new agent now needs: the three capacity regimes, Props 2/4/5/6 in accessible form, and the write/address/retrieve loop as actually demonstrated (designed, zero learning) — **with the designed-vs-emergent boundary stated explicitly at every point.** That boundary is the single most misquotable thing in the program.

## Item 5 — the KT tranche, closed as shelved
Record for the negatives registry: the CSF3 `winding2d` run **completed on CPU** (CUDA error 303, JAX fell back), returned **1 usable seed of 3** (all three array tasks overwrote the same `reduced_xy.json`), and scored against PREREG as **P2a CONFIRMED** (slope +0.253 med / +0.132 mean vs predicted < +0.5), **P2b FAILED** (+0.077 / +0.046 vs predicted < 0 — the sign change is *not* resolved), **P2c UNSCORABLE** (286/288 and 243/288 walkers censored at the 20000 cap). `T_KT` measured 0.898 J vs 0.893 predicted. **Head has SHELVED the KT line.** Register that **no soft exponent was obtained** on either arm — (b) was already HELD under the §7.23 estimator ruling.

## Acceptance
All four docs current, the pending blocks cleared, the four retired claims struck everywhere they appear, and a summary of what changed. ⚠ **Where evidence conflicts, record the conflict — do not adjudicate it.** Adjudication is the Hub's job and premature resolution destroys the debugging record.
