# AMENDMENT 5 to `PREREG.md` §2 — the seed gate, re-evaluated per seed

**Written 2026-09-21T12:28:44Z** (`AMENDMENT-5.timestamp.txt`), after
`/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` appeared. That file is stamped
`"produced_utc": "2026-09-21T12:27:01Z"` — **56 minutes after this hub started** (11:31:33Z),
so the task file's own start-time clause had already placed this hub on the 3-seed branch. It
is re-examined here anyway, per seed, rather than resting on that timing.

## The per-seed gate, applied

> A seed is usable only if its `CACHE-READY.json` entry has a **non-null `array_shape`** and
> **59 blocks**.

| seed | `array_shape` | `n_blocks` | `len(block_sha256)` | `scope` | gate |
|---|---|---|---|---|---|
| **3** | `[480000, 128, 4, 4]` | **59** | **59** | `full` | ⭐ **PASSES** |
| **4** | *absent* | *absent* (0) | — | — | ⛔ **FAILS** — `"status": "NOT_RUN"`, `"reason": "generation-2 cache for this seed was not completed"` |

Seed 3's entry also carries `generation: 2`, `n_rows: 480000`, `dtype: float16`,
`features_sha256 79ae328ca1432233a93d91f7922c79e060c8bd5eb12c199badfc50a07409f058`,
`shape_dtype_matches_seeds_0_2: true`, and a determinism check reading
*"reproducible: the same rows encode to the same bytes"*. ⭐ **The cache for seed 3 is real,
complete and gated.** Seed 4 is a declared NOT-RUN by its own producer.

## ⛔ But seed 3 cannot be scored, for a reason that is not the cache

The cache is only one of the four things a scored row needs. The other three are **absent for
seed 3 on this box**, measured, not assumed:

| what | where it must be | seeds present |
|---|---|---|
| the frozen decoder | `$CHLU_GEN_DECLIFT_DIR/decoder_v1_s<seed>.pkl` | **0, 1, 2** only; `winner_hashes.json` has exactly `['seed_0','seed_1','seed_2']` |
| the hash-asserted external classifier | `gen-external-classifier/seed_<seed>/` | **0, 1, 2** only |
| the fitted rival set | `spoke-a-rivals/rivals/seed_<seed>/` | **0, 1, 2** only |

`exp_gen_factor_wells.Judge` **is** the frozen decoder plus the hash-asserted classifier, so
it simply refuses:

```
Judge(3) REFUSES: FileNotFoundError Model file not found:
  ...\gen-decoder-lift\winner\decoder_v1_s3.pkl
```

⇒ **No arm of this hub can be scored at seed 3**, whatever the feature cache says. And even if
a decoder appeared, the five learned arms would be a NOT-RUN at seed 3 for want of fitted
rivals, which would put a 4-seed store row beside a 3-seed learned row in the same
comparison — exactly the mixing the task forbids.

`CACHE-READY.json` says as much about itself: seed 3's `bridge` field is
`{"NOT_RUN": "the decoder/probe/ceiling stage of exp_gen_bridge had not finished when this
file was written..."}`, and its status is `CACHE_READY_BRIDGE_PENDING`, not `READY`.

## ⭐ THE COUNT USED, DECLARED

> **n_seeds = 3. Seeds 0, 1, 2.** Every row, every table, every figure panel of this hub.

- **Seed 3: DECLARED NOT-RUN** — its generation-2 cache passes the gate, but the frozen
  decoder, the external classifier and the fitted rivals it would be scored through do not
  exist for it on this box (`decoder_v1_s3.pkl` absent; `Judge(3)` raises).
- **Seed 4: DECLARED NOT-RUN** — its `CACHE-READY.json` entry is `status: NOT_RUN`,
  `array_shape` absent, 0 blocks. It fails the gate.
- ⛔ Neither is reported as a zero, a blank, or silence.
- ⛔ No 3-seed row and 4-seed row is mixed anywhere, and `n` is printed on every plot and
  every table.

`PREREG.md` §2's *number* is unchanged; what changes is that its justification is now the
per-seed gate above plus a measured downstream refusal, rather than the absence of a file.
