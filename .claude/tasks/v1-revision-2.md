# Task: v1-revision-2 — close the V1 referee craft punch-list (w12; the F3 payoff has LANDED)

- **Agent:** `paper-writer` · **Output:** report to `.claude/outputs/v1-revision-2.md`; edit `.claude/papers/v1-short/` in place (CHANGELOG v0.3).
- **Read first:** protocol · `.claude/outputs/v1-referee.md` (verdict weak-accept; F1–F7 + MF-1) · **`.claude/outputs/v1-certificate-payoff.md` — LANDED: F3 is now MEASURED. Wire PAYOFF A (the strong embed): the det-J=0 router ERASES the latch (std(Q_out)=0) where the wormhole TRANSPORTS it by exact pᵀXΔ (std=std(Q_in)=0.0803). Its panel A figure is the natural F1 headline replacement (visibly different where landing-rate can't be).** Carry the honest fine print (matrix CM-7 updated): volume-alone≠latch-receipt (a det-J=1 random shift also scrambles Q); a FREE ΔH=0 exit still escapes ⇒ coercive-component membership, not the energy ledger, is the BIBO clause (PAYOFF B, with its "receipt buys it not the jump; router≡blind-wormhole" caveat). · claims matrix **v1.6**.
- **Context:** V1 is weak-accept; the fixes are craft/framing, all cheap. MF-1 (theory note live) is a Head/critical-path item, NOT this task.

## Items
1. **F1 (headline figure occludes its hero):** the wormhole/router/Newtonian all land at 1.0 and overlap, hiding the wormhole. Add a second panel or per-arm annotation carrying the **receipt** that distinguishes the arms (det J / ledger / latch-shift per arm) — the distinction is the certificate, not the landing rate. If `v1-certificate-payoff` landed, its latch/BIBO figure is the natural second panel.
2. **F2 (noise wall in no figure):** add a noise-σ panel (gate vs Hopfield across σ∈{0,0.3,0.6}) to Fig 2 — makes the foregrounded dominant negative visible (honesty asset). Data in `regime-remap-2000ep` §w8 eval-noise table.
3. **F3 (definitional "beats the router"):** IF `v1-certificate-payoff` delivered a measured guarantee-violation → wire it in §3.2 (router erases latch / blows up BIBO vs wormhole preserves), converting definitional→measured. ELSE soften to "reaches with a receipt the router cannot supply" + state the receipt's downstream value is argued (App B), not measured here.
4. **F4:** break the §5 MH-kernel wall-of-text into 3–4 design-rule sentences in main text + move the derivation (T_eff annealing, D=½s² erosion, FDT σ*) to an appendix.
5. **F5 (C-6):** add the BIBO-coercive-exit caveat to §3.1 main text next to the wormhole claim ("det J=1 certifies volume, not boundedness — the exit must lie in a coercive sub-level set or BIBO can fail, App B").
6. **F6:** name the kv (kv16) at which the 4.8× allocation payoff holds; state whether it persists at kv24/32 (numbers in `v1-pivot`).
7. **F7:** relabel Fig 1 title (drop the internal "§7.1"; note the observed edge is d≈3.2, the bracket, not L=2.5).
8. (N-a, defer) Fig 3 → appendix at the pruning pass, not now.

**Acceptance:** F1–F7 closed (F3 per whichever branch); PDF builds; diff-summary. Lifts V1 from "reads weaker than its honesty warrants" to clean weak-accept.
