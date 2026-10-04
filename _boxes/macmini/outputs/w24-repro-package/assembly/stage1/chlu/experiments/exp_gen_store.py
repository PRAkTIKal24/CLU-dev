"""Experiment: group the cached channels into wells, learn the modes, and **prove
the read cannot bypass the store** (R0 of the factorisation ladder).

⭐ DIAL DECLARATION — *isolation*.
- **Laundering control:** ⭐ random grouping at the same ``K`` and the same well-size
  histogram, run as a **full arm on every number**, never a footnote.
- **Falsifies:** random grouping matches correlation clustering within seed spread;
  **or** deleting the store leaves the read output unchanged.
- **Does NOT falsify:** a low absolute read accuracy that is low for *both*
  groupings — that is a hard dataset or a weak feature cache, and the *contrast*
  is what is under test.

⛔ **The failure this exists to catch.**  The programme's standing result is that the
block routed around the memory: deleting the entire store changed the loss by under
0.001 bits.  So here **no encoder->decoder bypass exists by construction**
(:func:`chlu.core.gen_store.read_from_codes` does not take the features as an
argument) *and* the deletion is **measured**:
:func:`chlu.core.gen_store.delete_store` empties the store and the read is re-run.
⛔ A near-zero delta is a **failed task**, reported as such, not repaired and re-run.

⚠ **Read the compositional term separately.**  ``gen-feature-bridge`` showed the
aggregate is dominated by four factors sitting at ~1.0 that carry no compositional
information, and that the ``scale`` deficit is a *photometric* rendering artefact of
the Montero corner.  Every table below therefore carries ``shape`` as its own
column, and the leakage transfer is normalised against the **matched-seen** control
(R16), never against random-A.

⛔ **Not this spoke's job:** no B arm comparison, no certificate, no C-as-queries, no
kill-rule verdict.  ``eval_b`` is touched **only** by the per-well leakage probes,
which never go near the decoder.

⛔ **Standing convention:** wells are never named semantically.  Integer indices only.

Runnable:
    chlu exp-gen-store [--dataset shapes3d] [--pretext recon] [--quick]
    uv run python -m chlu.experiments.exp_gen_store \\
        --dataset shapes3d --pretext recon --stage store --arm clu --seed 0 --out DIR
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from ..config import CHLUConfig
from ..core.gen_bridge import encode_images, features_for_rows, stream_images
from ..experiments.exp_gen_bridge import select_rows  # noqa: E402
def bridge_root(config: CHLUConfig) -> Path:
    """Directory of the frozen encoder run (NCLU_ENCODER_ROOT; not shipped, see README)."""
    r = config.gen_store.bridge_root or os.environ.get("NCLU_ENCODER_ROOT")
    if r:
        return Path(r).expanduser()
    from ..paths import ROOT
    return ROOT / "weights" / "encoder"


def bridge_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    d = bridge_root(config) / f"{dataset}_{pretext}" / f"seed_{seed}"
    if not (d / "bridge_manifest.json").exists():
        raise FileNotFoundError(
            f"no frozen bridge at {d} — this spoke consumes the FROZEN cache and "
            "⛔ never rebuilds it (set gen_store.bridge_root or $CHLU_GEN_BRIDGE_ROOT)"
        )
    return d


def select_store_rows(split, config: CHLUConfig, seed: int) -> dict:
    """Rows for the store, reusing the bridge's own ``eval_a``/``eval_b`` verbatim.

    ⛔ ``eval_a`` is excluded from the store's training pool, so the read metric is
    measured on rows no stage of this pipeline ever fitted.  The held-out rows are
    excluded structurally (the pool is ``split.train_rows``) *and* re-asserted.
    """
    bridge_rows = select_rows(split, config.gen_bridge, int(seed))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), bridge_rows["eval_a"])
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_910)
    n = min(int(config.gen_store.n_store_train), pool.size)
    return {
        "store_train": np.sort(rng.permutation(pool)[:n]),
        "eval_a": bridge_rows["eval_a"],
        "eval_b": bridge_rows["eval_b"],
        "bridge_train": bridge_rows["train"],
    }


def matched_seen_rows(dataset: str, split, values: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """⭐ R16's permanent companion: the photometrically MATCHED SEEN control.

    Same hue corner as the Montero held-out cell, ``shape != cylinder`` (so the
    combinations are seen), minus every row used for fitting.  ⛔ Normalising the
    per-well transfer against random-A instead would score the corner's rendering
    artefact as leakage — ``gen-feature-bridge`` measured that the ``scale`` deficit
    on B is statistically indistinguishable from the same deficit on matched-seen
    data.
    """
    if dataset != "shapes3d":
        raise NotImplementedError(
            "the matched-seen control is defined by the Shapes3D Montero corner; "
            f"⛔ do not invent one for {dataset!r} — return to the Hub"
        )
    v = np.asarray(values)
    corner = (
        (v[:, 0] >= 0.8) & (v[:, 1] >= 0.8) & (v[:, 2] >= 0.8)
        & (v[:, 3] >= 1.1) & (v[:, 5] > 20)
    )
    matched = np.where(corner & (v[:, 4] != 1.0))[0]
    matched = np.setdiff1d(matched, np.asarray(exclude, np.int64))
    if np.intersect1d(matched, split.test_rows).size:
        raise AssertionError("the matched-seen control overlaps the held-out split")
    return matched


_REENCODE_VERIFIED: set[tuple] = set()
def _cache_identity(cache_rows, cman: dict) -> tuple:
    """A stable key for "this cache, written by this encoder"."""
    return (
        str(cman.get("encoder_param_hash", "")),
        str(cman.get("dtype", "")),
        int(cman.get("D", 0)),
        int(cman.get("n_rows", -1)),
        int(np.asarray(cache_rows).size),
    )


def gather_features(
    dataset: str,
    config: CHLUConfig,
    encoder,
    cache_rows: np.ndarray,
    features,
    cman: dict,
    rows: np.ndarray,
    log: Callable = print,
) -> np.ndarray:
    """Cached features for ``rows``, **extending coverage through the FROZEN encoder**
    for rows the cache does not hold.

    ⚠ Measured constraint, not a choice: ``gen-feature-bridge`` wrote seed 0 at
    ``cache_scope="full"`` (480,000 rows) but seeds 1 and 2 at ``"used"`` (9,288
    rows = its own train + eval_A + eval_B).  This spoke needs more train rows than
    that, and it needs the matched-seen control rows, which are in **no** ``used``
    cache.

    ⛔ This is **not** rebuilding the bridge.  The encoder is loaded from the
    bridge's own checkpoint and its parameter hash is asserted against the cache
    manifest, so an on-the-fly row is produced by *the same frozen map* that wrote
    the cache.  Newly encoded rows are quantised to the cache's own dtype so a
    cached row and an encoded row are numerically indistinguishable — and that is
    **verified**, not asserted: :func:`_assert_cache_reencodes` re-encodes a sample
    of genuinely cached rows and requires agreement to within **one ULP of the cache
    dtype**, plus a differing fraction under 1%.  ⛔ Bit-identity is unachievable across
    hardware — a GPU accumulates convolutions in a different order and a single element
    can cross a float16 rounding boundary (measured on an L40S, job 20603452).
    """
    rows = np.asarray(rows, np.int64)
    d = int(cman["D"])
    out = np.zeros((rows.size, d), np.float32)
    pos = np.searchsorted(cache_rows, rows)
    safe = np.minimum(pos, cache_rows.size - 1)
    hit = (pos < cache_rows.size) & (cache_rows[safe] == rows)
    if hit.any():
        out[hit] = features_for_rows(cache_rows, features, rows[hit]).reshape((-1, d))
    n_miss = int((~hit).sum())
    if n_miss:
        # ⭐⭐ THE HAZARD, CAUGHT AT THE POINT OF EXTENSION.  Rows encoded here are
        # returned interleaved with rows READ FROM THE CACHE; if this encoder does not
        # reproduce the cache, the two populations are a different computation and
        # ⛔ nothing downstream would ever notice.
        # ⛔ The preflight may legitimately have run UNENFORCED — a run that the
        # coverage gate said was fully covered.  That is exactly why the check is
        # repeated HERE, at FULL STRENGTH, with no parameter that can ask for less.
        # ⚠ Charged once per (cache, encoder) per process: the check is deterministic,
        # and `_REENCODE_VERIFIED` is only ever populated by an ENFORCED pass.
        if _cache_identity(cache_rows, cman) not in _REENCODE_VERIFIED:
            log("    ⭐ cache extension requested — re-encode check runs ENFORCED here")
            _assert_cache_reencodes(
                dataset, config, encoder, cache_rows, features, cman, log=log,
                reason="gather_features is about to EXTEND the cache (point of extension)",
            )
        log(f"    extending cache coverage: {n_miss} row(s) encoded through the frozen encoder")
        missing = rows[~hit]
        slot = np.flatnonzero(~hit)
        # ⚠ chunked: 32k Shapes3D images as float32 is ~1.6 GB resident, and the
        # h5 block scan is re-walked per chunk, so this trades a second pass for
        # not holding the whole split in RAM.
        chunk = int(config.gen_store.encode_chunk)
        for i in range(0, missing.size, chunk):
            sel = slice(i, min(i + chunk, missing.size))
            imgs = stream_images(
                dataset, missing[sel], config.gen_data.data_root,
                int(config.gen_bridge.cache_block),
            )
            f = encode_images(encoder, imgs, int(config.gen_bridge.eval_batch))
            out[slot[sel]] = (
                f.astype(np.dtype(cman["dtype"])).astype(np.float32).reshape((-1, d))
            )
            del imgs, f
    return out


def _assert_cache_reencodes(dataset, config, encoder, cache_rows, features, cman, n=64,
                            log=print, *, will_extend: bool = True, reason: str = ""):
    """⭐ Verify that re-encoding CACHED rows reproduces the cache **to within one
    representable step of the format the cache is stored in**.

    Without this, "extending the cache through the frozen encoder" is a claim.  With
    it, it is a measurement.

    ⛔ **Why not bit-identity** (the defect this replaced — cluster job 20603452,
    L40S, 2026-09-15).  The cache is written as ``cman["dtype"]`` (float16 in every
    shipped manifest) and read back through float32, so a GPU that accumulates the
    encoder's convolutions in a different order than the CPU that wrote the cache can
    land a float32 output the other side of a float16 rounding boundary.  One element
    of 64 x 2048 moved by ``0.00390625`` — **exactly 2^-8, one float16 ULP for values
    in [4, 8)** — and the old ``if max_abs != 0.0: raise`` convicted an encoder whose
    parameter hash had matched in the same preflight moments earlier.  Bit-identity
    across hardware is unachievable, and it is not what "the same encoder" means.

    ⭐ **The tolerance is DERIVED FROM THE CACHE DTYPE, never picked.**  Each element is
    scaled by :func:`numpy.spacing` evaluated on *its own cached value at the cache
    dtype*, so the bound is ~0.00098 near 1.0 and ~0.0078 near 8.0.  ⛔ A single global
    ULP would be simultaneously too tight at the bottom of the range and too loose at
    the top.  ``np.spacing(|v|)`` is the gap *above* ``|v|``; the gap below is never
    larger (it halves only at a binade boundary), so a genuine one-step move always
    scores <= 1 ULP and is never convicted, and nothing larger is ever excused.

    ⚠ **Two bounds, both required — one ULP per element is not enough on its own.**  A
    run in which *most* elements moved by 1 ULP is a different computation even though
    no single element exceeds the per-element bound, so the differing **fraction** is
    bounded too.

        *Derivation of the 1% fraction bound.*  A float32 accumulation reorder
        perturbs the encoder output by relative ~1e-7.  float16 has relative precision
        2^-11 ~ 4.9e-4, so an element re-rounds only if its float32 value sits within
        ~1e-7 of a rounding midpoint: an expected flip fraction of ~2 x 1e-7 / 4.9e-4
        ~ 4e-4.  The cluster incident measured 1 / 131,072 = 7.6e-6.  ``1e-2`` sits
        ~25x above the derivation and ~1300x above the observation, while a genuinely
        different encoder moves essentially every element (measured at 100.0% of
        elements and ~1e2-1e4 ULP by ``tests/test_gen_store.py``'s positive control),
        so the bound is ~50x below the failure mode it exists to catch.

    ⭐ The statistics are computed and logged **always, pass or fail**, and returned so
    they reach the run manifest instead of living only in an exception string:
    ``max_abs_delta`` · ``max_ulp_delta`` · ``n_differing`` · ``frac_differing`` ·
    ``n_checked``.  ⛔ Backward compatible: ``n_checked`` still counts **rows** (two
    callers log it as such); every other key is new.

    ⭐ On a real mismatch the message states what was **measured**, in ULP.  It does not
    say "this is NOT the encoder" — the parameter hash is asserted separately and by
    the time this runs it has already matched; the reader draws the conclusion.

    ⭐⭐ **``will_extend`` — what the gate BINDS on, and why it is not a tolerance.**
    The hazard this gate exists to prevent is a run that **writes new rows** into the
    cache's feature space through an encoder that does not reproduce the rows already
    there (:func:`gather_features`' extension path).  A run that encodes **nothing new**
    cannot trigger that hazard: it reads cached rows and returns them, and a
    re-encode disagreement changes not one number it computes.  Such a run was still
    being refused — on a capability it never exercises.

    ⛔ **The bounds are NOT relaxed and there is no parameter that can relax them.**
    ``will_extend`` decides only whether the (unchanged) bounds **bind**:

    * ``will_extend=True`` — **the default, and today's behaviour exactly**.  A caller
      that passes nothing gets the abort.  ⛔ There is no silent loosening: the
      relaxation must be *asked for*, per call site, in the open.
    * ``will_extend=False`` — the caller has *evidence in hand* that this run cannot
      extend the cache (in ``exp_gen_wells_route_b``, the coverage gate's own
      ``cov["n_shortfall"] == 0``, two lines above the call — ⛔ never re-derived here,
      so the two can never disagree).  The statistics are still computed and still
      logged, the returned dict carries ``enforced: False`` plus the caller's
      ``reason``, and ⭐ ``would_have_failed`` / ``would_have_failed_why`` record
      whether the check *would* have aborted — so a reader can never mistake a skipped
      check for a passed one.  ⛔ A ``reason`` is **mandatory** here; omitting it is a
      :class:`ValueError`, not a quiet skip.

    ⭐ Scoping the preflight does **not** open a hole behind it.  An unenforced pass
    deliberately does **not** register in ``_REENCODE_VERIFIED``, so if such a run
    reaches :func:`gather_features` with an uncached row after all, the check re-runs
    there at full strength and aborts.  ⛔ Preflight is not the only place the hazard
    lives, and it is not the only place it is caught.
    """
    # ⭐ Declared here, in force for every caller, recorded in the returned dict so the
    # manifest carries the bound that was applied.  ⛔ Deliberately NOT keyword
    # arguments: there is no per-caller, per-device way to widen or disable EITHER
    # bound — ``will_extend`` governs whether they BIND, never how wide they are.
    ULP_BOUND = 1.0
    FRAC_BOUND = 1e-2

    if not will_extend and not str(reason).strip():
        raise ValueError(
            "⛔ _assert_cache_reencodes(will_extend=False) requires a `reason`. An "
            "unenforced check that leaves no trace is indistinguishable in the log and "
            "in the manifest from a check that passed — which is how this comes back as "
            "a mystery. Name the evidence that this run cannot extend the cache."
        )

    take = np.asarray(cache_rows[: int(n)], np.int64)
    cached = features_for_rows(cache_rows, features, take).reshape((take.size, -1))
    cache_dtype = np.dtype(cman["dtype"])
    imgs = stream_images(dataset, take, config.gen_data.data_root, int(config.gen_bridge.cache_block))
    fresh = (
        encode_images(encoder, imgs, int(config.gen_bridge.eval_batch))
        .astype(cache_dtype)
        .astype(np.float32)
        .reshape((take.size, -1))
    )

    # ⛔ A NaN defeats every comparison below silently (`nan > 0` is False), which would
    # make the gate unable to fail.  Count it before the arithmetic and convict on it.
    n_nonfinite = int(np.count_nonzero(~(np.isfinite(cached) & np.isfinite(fresh))))

    delta = np.abs(cached.astype(np.float64) - fresh.astype(np.float64))
    # one representable step of the CACHE's own format, at THIS element's magnitude
    ulp = np.spacing(np.abs(cached).astype(cache_dtype)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ulp_delta = np.where(delta > 0.0, delta / ulp, 0.0)

    n_elem = int(cached.size)
    n_diff = int(np.count_nonzero(delta > 0.0))
    stats = {
        "n_checked": int(take.size),  # rows — unchanged meaning, two callers log it
        "n_elements": n_elem,
        "max_abs_delta": float(np.max(delta)) if n_elem else 0.0,
        "max_ulp_delta": float(np.max(ulp_delta)) if n_elem else 0.0,
        "n_differing": n_diff,
        "frac_differing": (n_diff / n_elem) if n_elem else 0.0,
        "n_nonfinite": n_nonfinite,
        "cache_dtype": str(cache_dtype),
        "ulp_bound": float(ULP_BOUND),
        "frac_bound": float(FRAC_BOUND),
        # ⭐ whether the bounds BOUND on this call, and on whose word
        "enforced": bool(will_extend),
        "enforced_reason": str(reason),
    }
    log(
        f"  ⭐ cache re-encode check: {stats['n_checked']} rows, "
        f"{n_diff}/{n_elem} elements differ ({stats['frac_differing']:.3%}), "
        f"max |delta| = {stats['max_abs_delta']:.6g} = {stats['max_ulp_delta']:.3f} "
        f"{cache_dtype} ULP (bounds: {ULP_BOUND:g} ULP, {FRAC_BOUND:.1%} of elements)"
    )

    log(
        f"  {'⭐ ENFORCED' if will_extend else '⚠ NOT ENFORCED'} "
        f"(enforced={str(bool(will_extend)).lower()}): "
        f"{reason or 'no caller opinion — the strict default applies'}"
    )

    # ⭐ Every conviction is computed the same way whether or not it will be raised, so
    # the record of an unenforced run says exactly what an enforced one would have done.
    verdicts: list[tuple[str, str]] = []
    if n_nonfinite:
        verdicts.append((
            "nonfinite",
            f"re-encode check: {n_nonfinite} of {n_elem} element(s) are NaN or Inf in "
            f"the cache or in the re-encoded features, so the comparison is undefined "
            f"— {stats}"
        ))
    if stats["max_ulp_delta"] > ULP_BOUND:
        verdicts.append((
            "ulp",
            f"re-encoding {stats['n_checked']} cached rows did not reproduce the cache "
            f"to within one {cache_dtype} ULP: max |delta| = "
            f"{stats['max_abs_delta']:.6g} = {stats['max_ulp_delta']:.3f} ULP of the "
            f"cache dtype (bound {ULP_BOUND:g}); {n_diff}/{n_elem} elements "
            f"({stats['frac_differing']:.3%}) differ.  ⭐ That is further than a "
            f"rounding step of the stored format, so the loaded encoder does not "
            f"reproduce these cached values — {stats}"
        ))
    if stats["frac_differing"] > FRAC_BOUND:
        verdicts.append((
            "fraction",
            f"re-encoding {stats['n_checked']} cached rows stayed within one "
            f"{cache_dtype} ULP (max {stats['max_ulp_delta']:.3f} ULP) but "
            f"{n_diff}/{n_elem} elements ({stats['frac_differing']:.3%}) moved, over "
            f"the {FRAC_BOUND:.3%} bound: a rounding-boundary crossing touches a "
            f"handful of elements, a population this size is a different computation "
            f"— {stats}"
        ))

    # ⭐ The record of what the bounds SAID, independent of whether they bound.  ⚠ The
    # verdict messages above stringify `stats` as it was when they were built, so they
    # do not echo these two keys — they are for the returned dict and the manifest,
    # where a reader compares a skipped check against a passed one.
    stats["would_have_failed"] = bool(verdicts)
    stats["would_have_failed_why"] = [k for k, _ in verdicts]

    if will_extend:
        if verdicts:
            raise AssertionError(verdicts[0][1])
        # ⭐ Only an ENFORCED pass discharges the extension-point check in this process.
        _REENCODE_VERIFIED.add(_cache_identity(cache_rows, cman))
        return stats

    if verdicts:
        log(
            f"  ⛔⛔ THE RE-ENCODE CHECK WOULD HAVE ABORTED THIS RUN "
            f"({', '.join(k for k, _ in verdicts)}) and was NOT enforced, because: "
            f"{reason}. ⭐ The run encodes nothing new, so no inconsistent feature can "
            f"be written — but the encoder does NOT reproduce this cache on this "
            f"machine, and any run that extends the cache here WILL abort.\n"
            f"      {verdicts[0][1]}"
        )
    else:
        log("  ⭐ (the check was not enforced, and it would have PASSED anyway)")
    return stats


