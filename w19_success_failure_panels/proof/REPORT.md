# w19 success-failure-panels: REPORT

## Verdict: T7 DONE (rendered, hash-asserted, rebuild byte-identical). CMAZE-T ENGINE-ONLY; its arm columns are DECLARED NOT BUILDABLE.
The figures are in `~/clu/outputs/w17-paper/figures/rendered/` (`*w19*`, PDF + PNG). Read `README-w19.md` there.

## What was built
- **T7 per-arm success/failure** (seed 0, K\*=32): one ✓ row and one ✗ row each for the CLU, MLP, LSTM, Transformer and Continuous MLP. All six columns are rendered on every row.
- **T7 rival-succeeds / CLU-fails.** **Such requests EXIST at every seed**: any rival, 30 / 24 / 30 of 5,688 held-out requests at seeds 0 / 1 / 2 (CLU valid on 5,177 / 5,167 / 5,354). MLP, LSTM and Transformer are shown at seed 0. The continuous MLP has exactly one such request (seed 1), shown in its own seed-1 figure.
- **CMAZE-T**: three engine-only figures (families, door-then-tint, deletion on colour 1). Each shows $s_t$ + action and the true $s_{t+1}$ with legal/illegal. The arm columns are hatched.
- Set 1 and set 3 (w17) are untouched.

## Guarantees
- Same decoder: w17's OneDecoder is asserted per seed (seed 0: 78 panels, seed 1: 6), with hashes equal to winner_hashes.json. The positive control is identical. Each selected request, re-rendered alone, equals the scan.
- Scan: 3 seeds × 109 corners × 5,688 requests. Every per-corner per-arm n_valid reproduces scored.json (0 mismatches).
- CMAZE-T re-derivation equals A1_test_seed{0,1,2}.json: bank meta, hold-outs, and scored row counts 761/742/686.
- Rebuild: two processes, 15 files, SHA-256 identical (`SHA256-w19-build{1,2}.txt`).

## Findings worth stating
1. Rival-succeeds / CLU-fails is **not empty**: 24–30 requests per seed, mostly MLP/LSTM, rarely the Transformer. The continuous MLP has one request across all three seeds. The lookup floor beats the CLU on 26–67 requests per seed.
2. The CMAZE-T family `rule_a` (wrong key, untinted closed door) has **0 / 0 / 1** TEST n-way rows at seeds 0 / 1 / 2. The five-family n-way row is effectively four families.

## DECLARED NOT-DONE
- **CMAZE-T arm outputs** (CLU and each rival's generated $s_{t+1}$; one success and one failure per arm; the post-deletion arms). w16 persisted no codec, arm or rival object and no per-transition prediction. Re-training would be a new scored TEST run with a new codec. `NEEDS-ARTEFACTS.json` names what a persistence re-run must write. Whether to run it is a Hub/ledger decision.
- No .tex was edited. No scored run was made. The CMAZE-T figures are w16 material → `\PLACEHOLDER` slots under AMENDMENT-1.
