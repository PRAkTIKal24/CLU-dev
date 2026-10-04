# pw1-render-shapes LOG
- 2026-09-25T10:56:43Z START. Read COMMON.md, render-shapes.md, PLAN.md §1 item 9.
- 2026-09-25T10:57:03Z hash-check w20 TEST objects vs manifest: 116/116 OK (HASHCHECK-w20-objects.txt)
- 2026-09-25T11:00:46Z scan seed 0 started (pid 16358). Choices: n-CLU = w20 store_K32_n131072 (reused T7 run1_seed0_K32/arm.pkl, no manifest hash exists for it -> gated by per-corner reproduction of the w20 score file); split = stage/outputs/data/gen value_holdout (the T7/w20 split); palette = paper's existing colours (validator not run: no node).
- 2026-09-25T11:06:47Z scan seed 0 DONE: 109 corners, 5688 requests, 2616 per-corner comparisons (n_requests/n_accepted/n_valid/distinct_valid_novel_tuples x 6 columns) vs w20 score files, 0 mismatches. Seeds 1-2 scan started (counts only) pid 17176.
- 2026-09-25T11:19:20Z scans seeds 1,2 DONE: each 109 corners, 5688 requests, 2616 comparisons, 0 mismatches. Trial builds looked at (headers collided -> wrapped; (b) group labels removed). Starting build_twice.
- 2026-09-25T11:20:46Z build_twice #1 byte-identical, but (b) row head collided with group bar -> head split on two lines; rebuilding twice (build #1 moved to trial/superseded-*).
- 2026-09-25T11:24:32Z final build_twice BYTE-IDENTICAL (6 files); CAPTIONS.tex written + 19/19 numbers verified + banned-word scan 0 + standalone compile clean (0.9 linewidth for (a)); FINDINGS.md written; branch committed (9549f26), bundled and verified.
- 2026-09-25T11:24:35Z END.
