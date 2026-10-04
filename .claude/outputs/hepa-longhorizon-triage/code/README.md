# hepa-lh-tripwire — the matched-byte tripwire

The cheap reference that decides whether a long-context h-AUROC gain is a gain of
**mechanism** or a gain of **bytes**. Built and shipped **before** the sweep it judges.

## Files
| file | what it is |
|---|---|
| `tw_config.py` | every constant: HEPA arch numbers, protocol strides, seeds, descriptor list, seasonal periods. **No magic numbers live anywhere else.** |
| `byte_ledger.py` | HEPA's per-example state-byte arithmetic at context L, and store sizing. Auditable; the arithmetic strings are emitted into the JSON. |
| `features.py` | the four window descriptors + the RevIN-parity / right-alignment conventions. |
| `arms.py` | the matched-byte kNN exemplar store and the three trivial references. |
| `hepa_iface.py` | **the only module that touches HEPA.** Loader, splits, cut points, label surface, h-AUROC — all imported from `.claude/hepa-src`, never reimplemented. |
| `tripwire.py` | CLI entry point; writes the result JSON. |
| `PREREG_tripwire.md` | spoke-level pre-registration, written before any arm was scored. |

## Environment (the harness needs torch; the CHLU venv has none)
```bash
uv venv .claude/scratch/hepa-lh-tripwire/venv --python 3.11
VIRTUAL_ENV=.claude/scratch/hepa-lh-tripwire/venv uv pip install torch numpy pandas scikit-learn
```
Torch is needed only because `hepa/evaluation/surface.py` and `hepa/training/finetune.py`
are torch modules; no model is trained anywhere in this harness. Resolved versions are
written into every result JSON under `provenance`.

## THE CALL SIGNATURE WAVE 2 USES AT AN ARBITRARY RUNG
```bash
cd /Users/user/Desktop/CHLU
PYTHONPATH=.claude/outputs/hepa-longhorizon-triage/code \
.claude/scratch/hepa-lh-tripwire/venv/bin/python \
  .claude/outputs/hepa-longhorizon-triage/code/tripwire.py \
  --dataset  <FD001|FD002|FD003|FD004|PSM|MBA|GECCO|BATADAL|ETTm1|Weather|BeijingAQ> \
  --context  <rung, e.g. 512 | 2048 | 8192> \
  --seeds    42 123 456 \
  --out      <path>.json
```
Optional flags: `--budgets token_only residual_stream kv_cache` (default all three),
`--descriptors chan_stats last_patch patch_means raw_window` (default all four),
`--no-diagnostic` (drop the PREREG-non-compliant surface-payload store).
Env overrides: `HEPA_SRC`, `HEPA_DATA_DIR`, `TRIPWIRE_OUT`.

As a library:
```python
import tripwire
res = tripwire.run("FD001", 512, [42,123,456], ["residual_stream"], ["chan_stats"])
res["tripwire"]["tripwire_h_auroc"]     # the one number PREREG §4(b) consumes
```

## The number the kill rule consumes
`result["tripwire"]["tripwire_h_auroc"]` = max(best PREREG-compliant exemplar store,
best trivial baseline) on **ft_test**, with descriptor and k selected on **ft_val**.
PREREG §4(b) is then: the rung's HEPA gain survives only if it exceeds this by > 0.02.

## Non-negotiables baked in
* **Budget recomputed at every rung** (`byte_ledger.state_bytes(L)`), so the store grows with L.
  A store frozen at the 512 size would make the 8192 rung a strawman.
* **Scalar-concat horizon conditioning only** (PREREG §2.4): the store carries a scalar
  time-to-event and answers `p(t,dt) = weighted frac(neighbour tte <= dt)`. The K-vector
  surface store is available but is flagged `prereg_compliant: false` and is excluded
  from the tripwire number.
* **Degenerate horizons** follow the *shipped code* (drop only exactly-0/exactly-1),
  not the paper's 0.001/0.999 (scout D4).
* **No fabricated spread.** If the store fits the budget without subsampling the seed is
  inert; the JSON says `"regime": "deterministic"` and reports one number.
* **`--dataset`/`--context` combinations that cannot hold one exemplar** emit
  `{"status": "NOT_RUN", "reason": ...}`, never a zero.
