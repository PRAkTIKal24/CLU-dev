# pw1-render-shapes — plan item 9, the Shapes3D rendered success/failure panels
Session `pw1-render-shapes` · worktree `~/clu/worktrees/CHLU-pw1-render-shapes` · branch `pw1/render-shapes` (from `w19/success-failure-panels`) · out `~/clu/outputs/pw1-render-shapes/`
Read `COMMON.md`, then PLAN.md §1 item 9 in full. Load `dataviz` first. ⛔ **You do NOT edit either draft and you do NOT install.** A later spoke places your figures.

## Inputs
- **w20 data-matched objects, TEST leg**, copied from forgis-14 by the Hub and verified per file (116 files): `~/clu/stage/w20-shapes-data-matched/{stores/test,rivals/test}` with the manifest `SHA256-w20-test-objects.txt`. Run `shasum -a 256 -c` against the manifest BEFORE loading anything. The original run's record (LOG, SHAPES-MATCHED.json, FINDINGS) is at `~/clu/evidence/outputs/w20-shapes-data-matched/`.
- Carried fit per family: MLP and continuous MLP at `$10\times$` steps; LSTM and Transformer at the declared fit. Read which checkpoint that is from SHAPES-MATCHED.json and the w20 record, and print the mapping.
- n-CLU: K*=32, TEST leg, seed 0.
- Frozen decoder: `$CHLU_GEN_DECLIFT_DIR`. Hash-pinned judge classifier: `$CHLU_GEN_CLASSIFIER_ROOT`. Dataset: `~/clu/data/gen/shapes3d/` at the runbook pin. The w19 pipeline and its record are in your worktree and in `~/clu/outputs/w19-success-failure-panels/`. The feature cache here is `seed_0` only; if you need seeds 1–2, write `NEEDS-ARTEFACTS.json`, and do not substitute.

## The one job
(a) The per-model grid: for n-CLU, MLP, LSTM, Transformer and Continuous MLP, one held-out request each model renders validly and one it renders invalidly. Draw every model's image on both requests through the ONE frozen decoder, the judge's tick or cross on every panel, and the lookup table as the floor column.
(b) Nice-to-have: "baseline succeeds, n-CLU fails" rows where they exist. Say how many qualify, and whether all n-CLU wins are capsules.
**Selection:** a fixed-seed draw among qualifying requests under a stated DIVERSITY objective: maximise distinct shapes across rows (capsule must not dominate) and spread object, floor and wall hues. Print the rule and the seed, and never choose by appearance. Print the judge's six read-back values for every drawn request.
**Gates:** a same-decoder assertion, a positive control (a training-set request that renders validly), the object-hash check, two builds byte-identical, and PROVENANCE with hashes. Zero fits.

## Deliverables in your out dir
- `build/a`, `build/b`: the figure PDF+PNG files (stems `rc_pw1_shapes_per_model`, and `rc_pw1_shapes_baseline_wins` if (b) exists), plus `SHA256-build{1,2}.txt` and `PROVENANCE-render.json`.
- `REQUESTS-pw1.json`: every qualifying count, the rule, the seed, the drawn requests, and the six read-backs per image.
- `CAPTIONS.tex`: a main-text caption for each figure, stating the fixed-seed draw and the diversity rule, using n-CLU / baseline naming, "capsule", "held-out combination", and no banned words. Also write the one-paragraph text that introduces the figures in the Shapes results section.
- In FINDINGS: which existing draft slots these replace (the w19 `rc_w19_t7_*` figures), by label.
**What would prove this wrong:** any object whose hash does not match the manifest, a decoder mismatch, a positive control that renders invalid, or two builds that differ. If any of these happens, STOP drawing, report it, and still deliver everything else.
