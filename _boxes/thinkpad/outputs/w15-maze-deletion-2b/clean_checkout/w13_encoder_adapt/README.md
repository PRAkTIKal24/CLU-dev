# w13 hub/encoder-adapt — the artefacts this branch carries

⭐ **Read `FINDINGS.md` first.** `LOG.md` is the running record; `CROSS-ARM.md` is the
machine-built table; `CROSS-ARM.json` is the same thing with every field.

⛔ **This directory holds the SMALL artefacts only.** The 480,000-row generation-2 feature
caches, the encoder / store / decoder checkpoints and the fitted rival checkpoints live
under `/c/chlu/outputs/w13-encoder-adapt/` (18 GB) and are deliberately NOT in the bundle.
Every JSON here names the file it came from.

| path | what it is |
|---|---|
| `FINDINGS.md` | the report to the Hub |
| `LOG.md` | the running log, START to END |
| `CROSS-ARM.{json,md}` | the cross-arm table, built by `scripts/w13_report.py` |
| `PREREG-w13-encoder-adapt-1.md` (+ 2 AMENDMENTS, each with its own timestamp file) | written and timestamped before any number |
| `score/test_seed{S}_{arm}_K32/score.json` | ⭐ the 9 SCORED legs, TEST condition, 109 corners |
| `valscore/val_seed0_frozen_K32/score.json` | the FREE val floor check that licensed them |
| `run/seed_{S}/warm_start.json` | the shared warm start, one per seed |
| `run/seed_{S}/{arm}/arm.json` | the arm manifest: pins, compute, displacement, hashes |
| `run/seed_{S}/{arm}/probe.json` | the factor probe as a MARGIN over random init |
| `run/seed_{S}/{arm}/rivals.json` | the five learned baselines refit on that arm's features |
| `run/seed_{S}/{arm}/cache.json` | that arm's generation-2 cache manifest (sha256 per block) |
| `killproof/` | the kill -9 resumability proof, run before any spend |
| `MY_PIDS.txt` | every PID this hub launched. ⛔ No box-wide kill was ever issued. |

## The code, on this branch

| file | what it is |
|---|---|
| `chlu/experiments/exp_w13_encoder_adapt.py` | the three arms, the adapter, the anchor penalty, the stages |
| `scripts/w13_score_leg.py` | wave 12's K\* = 32 harness driven per ENCODER ARM |
| `scripts/w13_killproof.sh` | the kill -9 proof |
| `scripts/w13_arm_pipeline.sh`, `scripts/w13_launch.sh` | the lane runners |
| `scripts/w13_report.py` | the cross-arm table (re-derives nothing) |
| `chlu/eval/rivals/w4_learners.py` | ⚠ carries this wave's DECLARED REPAIR — see `FINDINGS.md` §9 |
