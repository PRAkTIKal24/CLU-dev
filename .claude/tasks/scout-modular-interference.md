# Task: scout-modular-interference — bib + novelty sniff for V3's modular-vs-monolithic related work (w8, small)

- **Agent:** `web-scout` · **Output:** `.claude/outputs/scout-modular-interference.md`
- **Read first:** protocol · `.claude/outputs/v3-short-draft.md` editorial Q5 (the gap: §4's modular/interference paragraph has `[·]` placeholders — the writer correctly refused to fabricate citations) · `v3-interference-ntk.md` §1 (what we actually measured: cross-unit CD-update basin displacement R; modular ≈2e-5 vs shared-V_θ ≈0.20; ∝κ²; O(1)-vs-O(N)).
- **Why:** the V3 short cannot ship a citation-free related-work paragraph. Small, focused pass — NOT a deep-research sweep.

## Items
1. **Citation-ready bib entries** (verified: authors/year/venue/arXiv id) for: catastrophic interference (McCloskey & Cohen lineage; French's review); modular NNs / mixture-of-experts (Jacobs et al. 1991 → sparse-MoE era); parameter-isolation continual learning (EWC, PackNet class); NTK-based interference/task-arithmetic analyses (whatever actually exists — verify before listing); any "interference kernel between modules measured during training" prior work.
2. **Novelty sniff (the claim we'd like to keep):** is anyone measuring a **cross-module interference kernel with a coupling-strength power law (∝κ²) and an O(1)-vs-O(N) architectural separation** in trained networks? Verdict: CLEAR / CROWDED(cite) / TAKEN(cite). Our defensible line is presumably "the firewall is *measured* on a physics-structured lattice with an exact zero beyond the coupling graph" — check nobody owns that phrasing-level claim.
3. **One lift-ready paragraph** (draft prose, guard-railed like di-bernardo-skim's) the v3-revision writer can splice into §4 with real citations.

Return a cited, verified brief; primary sources only (arXiv/DOI links per entry). Half-thread scale — do not expand scope.
