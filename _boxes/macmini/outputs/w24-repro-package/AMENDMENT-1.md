# AMENDMENT 1 — the Hub has filled most of NEEDS-ARTEFACTS.json (2026-09-26 ~22:55Z)

## Now available (checksums verified end to end)
1. **The frozen-encoder "bridge", all three seeds:** `/Users/batmac/clu/stage/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_{0,1,2}/`
   - `encoder.pkl` (1.7 MB), `probe.pkl` (2.5 MB), `decoder.pkl` (1.0 MB), `bridge_manifest.json`, `encoder_manifest.json`, `feature_cache.json`, `feature_rows.npy`, `shared_decoder_hash.json`
   - seeds 1 and 2 also have their generation-1 `features.npy` (38 MB). Seed 0's generation-1 `features.npy` (1.97 GB) was NOT staged; it can be rebuilt from the encoder.
2. **Branches `w15/crossover-seeds`, `w15/coverage-control`, `w15/lace-poe-rows`, `w15/bar-d-score`** are now in your clone.

## Still not available
- Generation-2 feature caches for seeds 1 and 2 (1.97 GB each). The Hub will not stage them: the package rebuilds them from pixels plus the encoder (`--stage cache2`). The proof stays at seed 0.

## What to do
- **Encoder, probe and decoder weights are all under 50 MB, so they ship** under `weights/` by the Head's rule. Convert them the way you converted the decoder and classifier, and assert the param hash against the manifests. Report sizes.
- **Fold in the w15 code** that the map needs: the three-seed crossover aggregator and figure, the coverage-control leg, and the K* generation-cell row. **`Tab. bard-gpu`:** the paper prints it, so it is on the map. Ship its code if it can be anonymised. Smoke it on CPU at toy size only; do not run a GPU. Document the hardware it needs. If you leave it out, say why in the report.
- **The entry points that exited at a missing file must now run their smokes.**
- **The final proof must cover the FINAL package.**
  - Re-scan, including the positive control.
  - Run a fresh-venv smoke of every entry point.
  - Full reproductions need NOT re-run if no file they execute changed. Prove that with a sha256 diff of those files between proof B's copy and the final package. If any executed file changed, re-run that reproduction.
- **Keep in the report:** the cross-machine finding. Gradient-trained baselines differ from the original records; the n-CLU, controls, blind arms, floors and planners are bit-exact. Your diagnostic run is the evidence. Do not soften it.
