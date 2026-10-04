# w12 spoke/positioning-scout — LOG

- START_UTC: see START_UTC.txt (2026-09-21T11:09:27Z)
- Task file: /c/chlu/tasks/w12-spoke-positioning-scout.md (read in full)
- Output dir: /c/chlu/outputs/w12-positioning-scout/ (task body path; the standing-rules header
  in the task file contains a typo'd variant `w12-w12-spoke-positioning-scout` — using the
  body/prompt path, which is the one the Hub named.)
- Worktree/branch: C:/chlu/worktrees/CHLU-w12-positioning-scout, branch w12/positioning-scout
- Scored runs: NONE (task says none). No chlu import needed -> chlu.__file__ assertion is a
  DECLARED NOT-RUN: this spoke runs no code and reports no numeric result from the package.

## Mechanical check — WEB ACCESS: PASS
- WebFetch https://arxiv.org/abs/2008.02217 -> title "Hopfield Networks is All You Need",
  abstract first sentence "We introduce a modern Hopfield network with continuous states and a
  corresponding update rule." => web reachable. NO-WEB-ACCESS.txt NOT written.

## Evidence base read (read-only, no writes outside my dir)
- /c/chlu/evidence/outputs/w9-advisor-report.md (239 l) + w10-advisor-report.md (139 l) — the two
  current-state reports, as instructed. Older advisor reports NOT read as state.
- /c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md §0, §1 — binding novelty wording + my own
  commission line.
- /c/chlu/evidence/claims_matrix.md lines 540/572/579/582/636/640 (grep hits: CRR, half-life, CM-21
  FORBIDDEN POSITIONING CLAIMS, CM-16a/b, CM-18).
- /c/chlu/evidence/outputs/w10-physics-section-evidence/EVIDENCE.md §0-§2.8 (the CRR spine) and
  BENCHMARK-STATEMENT.md §1-§3 head.
- /c/chlu/evidence/outputs/w10-draft-v0/DRAFT-v0.md rows 8.2/8.12/K19/K20/K21 (grep).
- $CHLU_CLAUDE_ROOT: NOT read (frozen/stale, per standing rules).

Key numbers I will cite as OUR side of the brief (all with file paths):
- CRR B1 median |log10 n_meas/n_pred| = 0.0433/0.0435/0.0431, factor 1.105, bar 0.10, 3 seeds
  (w10-physics-section-evidence/EVIDENCE.md §2.3, sourced to _thinkpad/w6/w6-potentials/RESULTS-LEDGER.md)
- B2 mean paired delta vs sighted fitted rival -0.5968/-0.5796/-0.5982, 49/49 corners all 3 seeds (ibid)
- B4 MISS x3: 0.3362/0.3301/0.3520 vs factor-2 bar (ibid §2.6)
- D17 slope measured +1.1368/+1.1367/+1.1368 vs derived +1.1385 (ibid §2.4)
- h/h* bucket table: 0.0008 median error at h/h*<0.1; 0.1482 in the 0.5-2 bucket (ibid §2.5)
- instrument floor median 0.0349 on an exact quadratic (ibid §2.2)
- the dynamics COST composition: row 4 paired delta -39.959/-46.918/-39.061 (ibid §1)
- charter C.1 "DOES NOT APPLY ... this hub claims no row against any rival"
  (BENCHMARK-STATEMENT.md §2 N4, sourced to _thinkpad/w6/w6-potentials/LOG.md §7)

## Web work (all fetched this session, 2026-09-21)
Reachable: arxiv.org (abs + ar5iv full text), iclr.cc, colmweb.org, general web search.
⛔ NOT reachable: openreview.net — every route returns
   {"name":"ChallengeRequiredError", "status":403} (api2.openreview.net, api.openreview.net,
   openreview.net/forum, openreview.net/pdf, with and without a browser User-Agent).
   => ALL PRIMARY REVIEW TEXT IS A DECLARED NOT-RUN. No review is quoted from OpenReview below.
   The motivation-vs-claim section is therefore built on ICLR's OWN published reviewer guide
   (iclr.cc), which is primary and authoritative, not on individual reviews.
Also rate-limited: api.semanticscholar.org (HTTP 429) -> venue lookups done from arxiv comments
   fields and the COLM accepted-paper list instead.

Primary artefacts downloaded to scratchpad (ar5iv full text -> .txt via perl tag-strip):
  1806.07366 Neural ODEs · 2111.00396 S4 · 2008.02217 Ramsauer modern Hopfield ·
  2302.07253 Energy Transformer · 2312.00752 Mamba · plus iclr2026_reviewerguide.txt,
  iclr2026_acguide.txt. Abstracts pulled verbatim from arxiv <blockquote class="abstract">.

## Deliverable written
POSITIONING.md — sections:
 0 the answer up front · 1 positioning sentences (6 papers, verbatim + non-claims) ·
 2 the MINIMUM dynamics evidence per paper (+2.6 the two closest analogues to our case) ·
 3 motivation vs claim from the ICLR 2026 Reviewer Guide (four key questions; the SOTA sentence;
   the worked reject example incl. "Things to improve ... that did not impact the score") ·
 4 the five discharge routes R-A..R-E and which we hold ·
 5 THE DELIVERABLE: the outlook section's shape, a 6-sentence drafted paragraph, the forbidden
   list bound to CM-21 + the wave-plan §0 rules, ladder placement, and the one experiment that
   would upgrade it · 6 what would prove this spoke wrong (answered: the stated falsifier is FALSE;
   the real finding is that we have no comparative object) · 7 limits + NR-1..NR-9 · 8 sources.

Verdict for the Hub: position the latent-dynamics material as an R-E phenomenon result in the
Edge-of-Stability grammar at ladder rung 3, NOT as a primitive. ICLR's own guide says SOTA is not
required; what is required is support at the level of your own verb.
