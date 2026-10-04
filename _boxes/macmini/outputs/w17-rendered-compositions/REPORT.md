# w17 rendered-compositions: REPORT (resume after staging)

## Verdict: DONE. All three panel sets and the deletion panel are RENDERED (PDF+PNG). One sub-item is DECLARED.
Output is in /Users/batmac/clu/outputs/w17-paper/figures/rendered/. See README.md there for the captions and LaTeX.

## Guarantees shown in code (SAME-DECODER.json, REQUESTS.json)
- Same frozen decoder: every panel is stamped, and all of them carry ONE hash, ONE object and seed 0. The hash was checked against the
  evidence-base winner_hashes.json (seed_0 01ef91f7...). The staged manifest is byte-identical to the evidence copy.
- Positive control: render+classify == Judge.read_back, element for element.
- Reproduction: each rendered corner reproduces the scored.json n_valid for every arm. The full scan re-rendered all
  test corners at seeds 0/1/2 and reproduced every per-corner n_valid in the three T7 scored.json files, with 0 mismatches.
- Deletion: features.npy sha256 == its manifest (ef6359a4..., gen 2). The downdated CLU reproduces deletion.json's
  after-counts on the set-1 corners.

## Results
- Set 1: CLU valid, every rival and the floor invalid (2-way pair-001/f4-hh, 6-factor nway-034).
- Set 2: DECLARED. No held-out request at seed 0, 1 or 2 is valid for all six arms. Every 5-of-6 request fails only the
  continuous MLP. The figure shows that nearest pattern and keeps the continuous MLP's X visible. The title says so.
- Set 3: every arm fails, the CLU included (2-way ×4 plus one n-way request found by scan).
- Deletion: on the pre-fixed extreme red-sphere row the CLU is valid before AND after the downdate. Every rival is invalid as trained
  AND retrained without, and the judge puts their miss on scale/orientation. The caption says the panel shows the operation,
  not forgetting. The lookup floor is excluded because it has no retrain-without comparator.

## Notes / deviations
- The data was staged at $W17_RC_STAGE/outputs/data/gen, not at $W17_RC_STAGE/data/gen as the resume note said.
- The comparator .rival.json manifests were not staged. They are joined from the evidence base by symlink (_assembled_comparators/).
- The co-validity counts behind the set-2 declaration come from this spoke's own re-render (COVALID-SCAN.json), not from
  the evidence base. They are kept out of every caption as numbers.

## Did NOT run
- Seeds 1/2 are not rendered as figures (they were only scanned). No w15/w16 artefact was read. No rates were computed for the draft.
