# `w15_cmaze_o_harness` — CMAZE-O, the observation instrument

**Spoke:** `w15/cmaze-o-harness`. **Scored runs: 0.** Output dir mirrored here; the live
copy is `/c/chlu/outputs/w15-cmaze-o-harness/`.

**Wave 16 is gated on `HARNESS-READY.json`. It reads `"status": "ready"`.**

## What this is

CMAZE-O is the world model's **observation model** over CMAZE-1. A request is **nine
words** — layout id × agent colour × key colour × door colour × tinted-floor colour ×
agent cell × key cell × door cell × goal cell — and the answer is **the rendered grid
state**. The judge is **exact placement of every object AND legality of the state under
the engine's rules**, with **no tolerance**: `tolerance_used` is `null` on every row.

`w8_envscout.maze_colour` supplies the engine, the solver and the legality decision and is
**imported unmodified** (sha256 in `HARNESS-READY.json → provenance.engine`).

## Files

| file | what it is |
|---|---|
| `chlu/exp_w15_cmaze_o_env.py` | the layout bank, the nine-word vocabulary, the state, the legality suite, the item tensor, the state decoder and **the judge** |
| `chlu/exp_w15_cmaze_o_splits.py` | VALUE / 2-way / n-way hold-outs and the disjointness proof |
| `chlu/exp_w15_cmaze_o_encoder.py` | the ONE shared frozen autoencoder; its loss is shaped by the judge's argmax |
| `chlu/exp_w15_cmaze_o_arms.py` | CLU single design, charter-C controls, matched-bytes lookup floor, four matched-parameter rivals, five BLIND columns |
| `chlu/exp_w15_cmaze_o_pipeline.py` | one seed end to end |
| `chlu/exp_w15_cmaze_o_deletion.py` | value and conjunction deletion, store-scoped exactness, costs |
| `chlu/exp_w15_cmaze_o_verify.py` | **the independent checker** — imports no part of the judge path |
| `chlu/exp_w15_cmaze_o_readiness.py` | assembles and CHECKS `HARNESS-READY.json` |
| `scripts/w15_cmaze_o_{env.sh,probes,valsmoke,span,report,finalise}` | the drivers |

## Read this first

`FINDINGS.md` §0 and §6. The short version: the instrument is sound and the CLU separates
from every rival on 3/3 seeds, but **the charter-C controls do not collapse**, so this is
an **INTERFACE claim, not a STORE claim**. The atoms are 0.76 % of the deployed read; the
address book is 99.24 %. The pair-block ablation is the next measurement and is
**DECLARED NOT-RUN**.
