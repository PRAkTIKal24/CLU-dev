# Task: v3-revision — fold reversible-O(1) + reconcile C-1 + fill the flagged gaps in the V3 short (w8)

- **Agent:** `paper-writer` · **Output:** report to `.claude/outputs/v3-revision.md`; edit `.claude/papers/v3-short/` in place.
- **Read first:** protocol · `.claude/outputs/v3-short-draft.md` (its 6 open editorial questions — this task answers them) · Charter (C-1 REVERSED) · `.claude/claims_matrix.md` (CM-13 = reversible-O(1) now measured) · `.claude/outputs/v3-reversible-o1.md` (the numbers for §3.5).

## Items
1. **§3.5 reversible-O(1) — fill the marked slot** with CM-13 verbatim: 946× peak-mem reduction (6.3 KB vs 6.00 MB @ T=1024, N=2), O(1)-vs-O(T), gradients ≤2e-6 rel (f32), ≈0.9× wall-time **(CPU/small-D caveat in-sentence, C-5)**; the **γ=0-only exactness** as the honest scope (γ>0 horizon ≈5e2 @γ=0.05; "reversibly-trainable memory = conservative memory" — ties to the budget framing); note it is NOT yet in the shipped trainer (gradient mechanics only). Move the RevNet/checkpointing prior-art to §4.
2. **C-1 reconciliation (editorial Q1) — CONFIRMED: no audit paragraph.** V3 already omits it (correct). No action beyond removing the header editorial note; the V2 draft is being fixed to match (v2-revision MF-1), so the cross-short split closes.
3. **Headline figure (editorial Q3):** the O(N)-vs-O(1) *scaling curve* (S vs N) may not exist as a standalone PNG — the asset is the *bars* fig. Flag to the Hub whether the analyst must generate the scaling-curve PNG from `v3-interference-ntk/through_training.json` + item-3 data; embed the bars fig meanwhile.
4. **Modular/interference related-work bib (editorial Q5):** the §4 modular-vs-monolithic paragraph has NO citations (no scout covered catastrophic-interference/MoE prior art). **DO NOT fabricate** (rule 5). Leave `[·]` placeholders + flag to the Hub for a scout micro-pass (folded into venue-follow-up or a dedicated `scout-modular-interference`).
5. Apply the C-2/C-5 discipline the draft already follows; keep appendix-maximal (C-10).

**Acceptance:** PDF rebuilds; §3.5 filled from CM-13; editorial Q1 closed; Q3/Q5 flagged with specific asks (not silently dropped). Then → `paper-referee` (w8/w9 `v3-referee`).
