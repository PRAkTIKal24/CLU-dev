# ⭐ §5.1 FALSIFIER VERDICT — **CLEAR. SCORED WORK MAY PROCEED ON THIS BOX.**
**Produced by `hub/generation-table-2`, FORGIS_14 (Windows 11 AMD64), 2026-09-17T10:5x Z.**
**Artefact: `.claude/outputs/hub-generation-table/reencode_magnitude.json`**
**Script: `.claude/outputs/hub-generation-table/iter2/scripts/falsifier_reencode_magnitude.py`**

> ⭐ **This file exists for the other three hubs.** The §5.1 stop condition did NOT fire.

## The measurement
Seed 0 cache (`shapes3d_recon/seed_0`, `n_rows=480000`, `scope="full"`, `D=2048`, `dtype=float16`),
`crows[:64]`, re-encoded through the frozen `encoder.pkl` (param hash
`98d326028316ae2cd6b960b26a922cd47fe8a2c9e73199fb1ba2927864a29c0d`, **matches the bridge manifest**),
arithmetic byte-for-byte the same as `_assert_cache_reencodes` (float16 round-trip, `np.spacing` at the
cache dtype).

| quantity | value |
|---|---|
| elements checked | 131,072 (64 rows x 2048) |
| differing elements | 1,280 — **0.9766 %** |
| `max_abs_delta` | 1.953125e-03 |
| `max_ulp_delta` (all) | **51.0** |
| argmax-ULP element: `\|cached\|` | **1.7583e-04** (row 24, channel 1185) |
| argmax-ULP element: delta | 6.0797e-06 |
| argmax-ULP element: local `np.spacing` | 1.1921e-07 (**the float16 subnormal gap**) |

### ⭐⭐ Split by magnitude — the whole question
| band | elements | differing | **max ULP** | max abs delta |
|---|---|---|---|---|
| ⭐ **`\|cached\| >= 2^-4`** | 63,655 | 575 | ⭐ **1.000** | 1.953e-03 |
| `\|cached\| < 2^-4` | 67,417 | 705 | **51.0** | 3.052e-05 |

⭐ **In the `>= 2^-4` band the maximum ULP delta is EXACTLY 1.000, and the mean over differing elements is
0.9991 — every differing element there moved by exactly one float16 rounding step and not one moved
further.** That is a rounding-boundary crossing, precisely the thing the gate's docstring says it must not
convict.

⛔ **Violators of the stop rule (`\|cached\| >= 2^-4` AND `ulp_delta > 2`): ZERO.**

⭐ The 51-ULP figure is entirely an artefact of ULP counting below `2^-4`: at `|cached| = 1.76e-04`,
`np.spacing(float16)` has collapsed to the float16 **subnormal gap** `5.96e-08`... in fact to
`1.1921e-07`, and the *absolute* move is `6.08e-06` — six millionths, post-ReLU dust. The prediction in
the launch file is confirmed by measurement, not assumed.

## ⚠⚠ TWO THINGS EVERY HUB MUST TAKE FROM THIS — do not read the CLEAR verdict as "the gate passes here"
1. ⛔⛔ **An ENFORCED `_assert_cache_reencodes` on this box ABORTS.** `max_ulp_delta = 51.0 > ULP_BOUND = 1.0`.
   `would_have_failed` is **true**, verdict key `ulp`. ⭐ The only reason any scored run happens here is
   COMMON §4's ruling that the gate **does not bind when the run extends nothing** (`n_shortfall == 0`).
   ⛔ **The moment your run needs one uncached row, the gate binds, it fails, and you STOP and report the
   shortfall. You do not encode.** That is not a loophole to route around; it is the ruling working.
2. ⚠ **The FRACTION bound has almost no headroom.** `frac_differing = 0.9766 %` against `FRAC_BOUND = 1.0 %`
   — a margin of 0.023 pp, i.e. **31 elements of 131,072**. ⛔ Nobody should read "the fraction bound passes"
   as robust on this hardware. It is not a second, independent reassurance; it is very nearly red too.
   (It is also not *evidence of a different computation* — the 1 % bound was derived ~25x above a ~4e-4
   expectation, and what we measure sits at the derivation's own order of magnitude, driven by the
   sub-`2^-4` dust band where float16 has essentially no precision left.)

## What this licenses, exactly
⭐ **A no-extension run on this box is sound**: seed 0's cache is `scope="full"` (480,000 rows = the whole
split) and provably cannot be extended, and the arithmetic divergence that exists is confined to elements
whose absolute magnitude is <= 3.05e-05 — it cannot move a reported number.
⛔ **It licenses nothing about seeds 1 and 2** (`scope="used"`), nothing about a GPU, and nothing about
raising `ULP_BOUND`, widening `FRAC_BOUND`, or passing `will_extend=False` with a shortfall > 0.
