# bprime-draft-r5-cite-fold — fold both cite-checks into r4 → produce draft-r5

**Campaign 2, wave C2W5. Agent:** paper-writer. **Small. NO WORKTREE, no branch** (papers live under
gitignored `.claude/papers/`). Input: `.claude/papers/bprime/draft-r4.md` (2621 lines, the r4 fold is
COMPLETE and accepted). Output: `.claude/papers/bprime/draft-r5.md` + `.claude/outputs/bprime-draft-r5-cite-fold.md`.
Sources of record: `.claude/outputs/bprime-cite-check-2.md` and `.claude/outputs/bprime-cite-check-3.md`
(both Hub-reviewed and ACCEPTED — every verdict below is theirs, not yours to re-derive).

## The fold items (all citation-layer; no measured number moves)

1. ⛔ **§5.3 MIS-ATTRIBUTION, MUST FIX (cite-check-3 recon 1–2).** The quotes *"30–80× larger than
   B-trees"* and *"4 orders-of-magnitude more time to build"* are **NOT SOSD/Kipf 2019** — they are
   **Chesetti & Pandey, ACDA 2025 (SIAM), pp. 101–114, arXiv:2407.00590v2 §6.6**. SOSD's own verdict
   is the REVERSE on size (RMI 3 %, RS <1 % overhead vs B-tree 16 %; *"learned models indeed often
   outperform state-of-the-art implementations"*). Use cite-check-3 §3.3's suggested rewrite as the
   base. If the quotes are kept: (a) subject is **RadixSpline and RMI** (PGM is 4× *smaller*),
   (b) the B-tree baseline was **sparsified** (every 256th key) — both caveats must ride along.
   Pin the version (`arXiv:2407.00590v2`) — the SIAM camera-ready was not read.
2. **Remove all 8 ⟦CITE2⟧ markers** — the verification they gated is done:
   - **Poliak:** VERIFIED incl. "6 of 10"; preprint ≡ published text, quote freely. ⚠ Precision edit:
     the paper's verb is **"significantly outperforms"**, not "beat" (both §5.3-style sites).
   - **Mamba-2:** VERIFIED — cite the PMLR locus: **Dao & Gu, ICML 2024, PMLR 235:10041–10071,
     arXiv:2405.21060**. ⚠ Author-order trap: Mamba-2 = Dao & Gu; Mamba-1 = Gu & Dao.
3. ⭐ **GDN equation restriction LIFTED (cite-check-3 item 1).** `S_t = α_t S_{t−1} + v_t k_tᵀ` is
   triple-sourced incl. the ICLR 2025 camera-ready — r5 **may quote** it. ⛔ Must carry GDN's own
   hedge: they present Mamba2 in this form *"up to specific parameterization"*.
4. **§1 (l.≈68) cite upgrades (cite-check-3 recon 4):** Mitzenmacher = NeurIPS 2018, pp. 462–471,
   arXiv:1901.00902 (⛔ never cite 1802.00884 as the NeurIPS paper). Kipf/SOSD = **non-archival**
   (NeurIPS 2019 ML-for-Systems workshop; cite arXiv:1911.13014). The "accompanying PVLDB study" =
   **Marcus** et al., PVLDB 14(1):1–13, DOI 10.14778/3421424.3421425 — different first author; do
   NOT collapse the two bibliography entries. ⚠ Marcus et al. body text is UNVERIFIED — no number
   quotes from it. Prefer cite-check-3 §3.3's safe/unsafe §1 claim split (Mitzenmacher conditions the
   *verdict* on size; SOSD makes space a *reported column* — "matched space" is only Mitzenmacher).
5. **Bibliography:** paste the five verified BibTeX entries from cite-check-3 + the two from
   cite-check-2. ⚠ The Belinkov corroboration entry is UNVERIFIED — include only if actually cited,
   with its own caveat note. ⛔ Never-copy: Semantic Scholar labels Poliak's venue "SemEval" —
   **\*SEM ≠ SemEval**; the ACL Anthology record governs.
6. ⭐ **Run the claims_matrix §0.9 / CM-30 sweep over the full draft** (the matrix is now
   **v2.11 HUB-CONFIRMED**; §0.9 postdates r4's own sweep). Two items to check by name:
   (a) **the §A20.2 form** — the refuted object is *"the P-particle occupancy read protocol at
   P = 4"*, NEVER *"the compositional family"* (the Hub hands you this form explicitly per the
   curator's rec 5); (b) **`null*` = 0.00117 is an ORACLE-SELECTED upper bound over 584 configs,
   never quotable as "the best null arm scored…"** (selected-config scores are 0.0000–0.00039).
   Print the sweep results as r4 did (§4 pattern).
7. **CHANGELOG line** per A18.5 (r-convention), per-item disposition. Everything else in r4 is
   frozen — this is a citation-layer revision; if you find yourself moving a measured number, stop
   and flag the Hub instead.

Report → Hub with the sweep printed and any residual single-sourced items named.
