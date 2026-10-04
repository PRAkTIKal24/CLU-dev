# w4-spoke-a — the rival set, and cache generation 2

Everything a wave-4+ hub needs from this spoke, and the rules that come with it.
**Start with `RIVALS-READY.json`.** It is rewritten every time an artifact lands,
so it is never stale and never ahead of the evidence.

## The rules, before any number

1. **GENERATION 2 ONLY.** Never quote a number computed on the generation-2 cache
   beside one computed on generation 1, in either direction. Generation 1 and
   generation 2 disagree on ~0.5 % of feature elements by one float16 ULP with an
   identical encoder hash, on all three seeds — see `gen1_vs_gen2.json`. That is
   the signature of a cache not produced entirely on one device.
2. **Generation 1 is untouched.** It is still on disk and still hashed. This
   spoke only ever read it.
3. **Know which population your number is on.** `population.json` records, per
   statistic, what it was measured on. This spoke's scored sets are the Montero
   **combination** hold-out and are retained-values-only (measured: 0 rows
   carrying a value absent from the fit pool). They are **not** the value-holdout
   population. The only numbers here about a value the arm never held are the
   deletion comparator's.
4. **A dash is never a win for us.** `row_eligibility.json` says in advance which
   rival can enter which row, with a named footnote on every dash, and says when
   the limit is the *method's* rather than this spoke's.
5. **An accuracy is never read without its distinct-output count.** Every rival
   row carries `distinct_factor_tuples` on B out of 72.

## Files

| file | what it is |
|---|---|
| `RIVALS-READY.json` | the readiness file. Per-rival booleans, seeds, hashes, parameter counts, population, the callable API. |
| `cache_generation2/shapes3d_recon/seed_*/` | the generation-2 caches. Full scope, 480,000 rows, float16, CPU-only, one device. Each is also a drop-in bridge directory. |
| `determinism_check.json` | the falsifier: one shard re-encoded twice, sha256 compared. |
| `gen1_vs_gen2.json` | how far generation 2 moved from generation 1, per seed. |
| `population.json` | which population every statistic is on, measured. |
| `row_eligibility.json` | which rival can enter which row, and the footnote when it cannot. |
| `rivals/seed_*/` | per-rival scores, and the fitted rivals as loadable checkpoints. |
| `rescore_generation2/` | every existing arm re-scored on generation 2. |
| `rescore_gen1_vs_gen2.json` | whether re-scoring moved a published arm by more than its seed spread. |
| `LOG.md` | dated bullets throughout, and the final report as its last section. |

## Running a rival through your own corners

The rivals are not only scores on this spoke's request sets. Each fitted rival is
saved and exposed as a request-to-features callable, with the contract
`exp_gen_factor_wells.arm_features` already uses:

```python
from chlu.eval.rivals.w4_api import load_rival, RivalRefusal, available

available(seed)                  # what is loadable, and why anything missing is missing
r = load_rival(seed, "table")    # table | mlp | cont_mlp | transformer | lstm
feats, out_of_range = r(values)  # values (N, F) float, np.nan = "not requested"
                                 # feats (N, D) float32, out_of_range (N,) bool
```

`r.names` gives the factor order — do not hardcode it. `out_of_range` is True for
a row in which any **requested** value falls outside that factor's realised grid
range; an unrequested factor never sets it.

**Refusals are deliberate and are not gaps.** `lace` and `poe_ebm` raise
`RivalRefusal`: both are sampler draws, not functions of the request, so a
"function of the request" would be a fiction. Off-grid values are refused by
every family whose input is a class index — snapping an off-grid request onto the
nearest grid value would score the rival on a request it never received. Only
`table` and `cont_mlp` accept off-grid values.

If you need LACE or the PoE-EBM on your own corners, ask this spoke to score
them; do not reconstruct them yourself.

## Rebuilding any of it

```
python -m chlu.experiments.exp_w4_rivals --stage determinism            # the falsifier, first
python -m chlu.experiments.exp_w4_rivals --stage cache2   --seeds 0 1 2 # resumable, CPU-only
python -m chlu.experiments.exp_w4_rivals --stage compare  --seeds 0 1 2
python -m chlu.experiments.exp_w4_rivals --stage table learners deletion lace poe --seeds 0
python -m chlu.experiments.exp_w4_rivals --stage bridgify rescore --seeds 0 1 2
```

Every stage is resumable at artifact granularity and skips what is already on
disk unless given `--force`. The cache build is resumable block by block and
takes `--max-seconds` as a soft per-launch cap. `JAX_PLATFORMS=cpu` is pinned in
the module; the generation-2 builder refuses to write a byte unless JAX sees
exactly one device and it is a CPU.
