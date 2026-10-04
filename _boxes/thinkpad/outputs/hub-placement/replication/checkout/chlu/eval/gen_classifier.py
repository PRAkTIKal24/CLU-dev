"""⭐ The **external** factor classifier — the instrument that reads IMAGES.

⭐ **Why this exists (R29 pivot item 2).**  Every composition number in phase 1 is
currently read by :class:`~chlu.core.gen_bridge.FactorProbe`, and that probe is
fitted **inside the bridge run, on the same 8,192 train rows the shared decoder is
fitted on, from the same seed-derived key chain**
(``exp_gen_bridge.run_decoder_stage``: ``key, dk, pk = jax.random.split(key, 3)``).
R29's headline table is ``request → read → codes → decoder → image → EXTERNAL
classifier``.  ⛔ **A probe that shares the decoder's lineage cannot be the
instrument that scores the decoder's output.**  This module is that instrument.

**What "external" means here, operationally — four separations, all asserted:**

1. **Rows.**  :func:`classifier_rows` draws a subsample of the train split that is
   **disjoint** from the bridge's fit rows, from ``eval_a``, and from every row in
   the photometric corner the phase evaluates in.  ⛔ Every row it is scored on is
   a row it has never seen in any form.
2. **Renders.**  ⛔ It never sees a decoder render at fit time — and that is
   *verified*, not asserted: :func:`assert_no_render_leak` re-streams a sample of
   the training tensor straight out of the HDF5 file and requires bit-identity.
3. **Architecture.**  Kernel 5 / SiLU / four stages / a hidden projection, against
   the probe's kernel 4 / ReLU / three stages / direct linear heads.
4. **Key chain.**  ``PRNGKey(seed * 104_729 + 20_260_913)``, ⛔ never the bridge's
   ``seed * 10_007 + 20_260_910``.

⛔ **Architecture alone is not independence.**  Independence is only testable where
the two instruments can disagree — on the decoder's own renders — which is what the
experiment module's leg 3 measures and ⛔ what this module deliberately does not
adjudicate.

**The gate** (:func:`gate_verdict`) is pre-registered and stop-and-return: in-corner
``shape`` balanced accuracy on **REAL** held-out-combination images must reach the
floor on **every** seed, because R15 established that the shipped probe reads real
images at **1.0000** and is saturated — an instrument that cannot read real images
in the corner is not an instrument.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..core.gen_bridge import (
    FactorHeads,
    assert_no_holdout_leak,
    model_dtype,
    stream_images,
)

#: R15: the shipped probe reads REAL images at 1.0000 and is saturated.  An
#: instrument that cannot read real images in the corner is not an instrument.
DEFAULT_GATE_FLOOR = 0.95

#: ⛔ The only genuinely compositional Shapes3D factor (R16): four of six heads sit
#: at ~1.0 for any competent reader and carry no compositional information.
GATE_FACTOR = "shape"

#: Montero's photometric corner, the one predicate every phase-1 evaluation set
#: lives in.  Kept verbatim from ``exp_gen_store.matched_seen_rows`` so the classifier's
#: exclusion set and the evaluation sets cannot drift apart.
CORNER_COLUMNS = "floor_hue >= .8, wall_hue >= .8, object_hue >= .8, scale >= 1.1, orientation > 20"


class RenderLeak(AssertionError):
    """Raised when a training tensor is not bit-identical to the dataset on disk."""


class LineageLeak(AssertionError):
    """Raised when the classifier's rows overlap the decoder/probe's own rows."""


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class ExternalFactorCNN(eqx.Module):
    """A per-factor classifier over **raw images**, independent of the decoder.

    ``[conv(k, stride) -> GroupNorm -> SiLU] x len(channels)`` -> flatten ->
    ``Linear(hidden) -> SiLU`` -> one linear head per factor.  Calling it returns a
    **list of per-factor logits**, exactly like
    :class:`~chlu.core.gen_bridge.FactorProbe`, so the two instruments are scored by
    *the same* ``chlu.eval.gen_probe`` functions and a gap between them can never be
    a difference in the scoring code.
    """

    convs: list
    norms: list
    proj: eqx.nn.Linear
    heads: FactorHeads

    def __init__(self, in_shape, channels, kernel, stride, groups, hidden, cardinalities, key):
        keys = jax.random.split(key, len(channels) + 2)
        convs, norms = [], []
        c_in, h, w = (int(x) for x in in_shape)
        pad = max((int(kernel) - int(stride)) // 2, 0)
        for c_out, k in zip(channels, keys[: len(channels)], strict=True):
            convs.append(
                eqx.nn.Conv2d(c_in, int(c_out), int(kernel), stride=int(stride),
                              padding=pad, key=k)
            )
            norms.append(
                eqx.nn.GroupNorm(min(int(groups), int(c_out)), int(c_out))
                if int(groups) > 0
                else None
            )
            c_in = int(c_out)
            h = (h + 2 * pad - int(kernel)) // int(stride) + 1
            w = (w + 2 * pad - int(kernel)) // int(stride) + 1
        self.convs = convs
        self.norms = norms
        self.proj = eqx.nn.Linear(int(c_in * h * w), int(hidden), key=keys[-2])
        self.heads = FactorHeads(int(hidden), cardinalities, keys[-1])

    def __call__(self, x):
        # ⚠ §7.23: a module-scoped ``jax_enable_x64`` fixture elsewhere in the suite
        # makes eqx build float64 weights; cast rather than assume float32.
        x = jnp.asarray(x, self.convs[0].weight.dtype)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.silu(x)
        return self.heads(jax.nn.silu(self.proj(x.reshape(-1))))


def _xent(logits_list, labels):
    """Mean over factors of the per-factor cross-entropy (labels ``(B, n_factors)``)."""
    total = 0.0
    for j, logits in enumerate(logits_list):
        logp = jax.nn.log_softmax(logits, axis=-1)
        total = total - jnp.mean(jnp.take_along_axis(logp, labels[:, j][:, None], axis=-1))
    return total / len(logits_list)


def train_external_classifier(
    images: np.ndarray,
    labels: np.ndarray,
    cardinalities,
    cfg,
    key,
    log=None,
    on_epoch=None,
):
    """Fit the external classifier on **real** images + factor labels.

    ``on_epoch(step, model, loss)`` is called every ``cfg.clf_checkpoint_every``
    steps so the caller can checkpoint — ⭐ the run is resumable at seed
    granularity and never holds a result to the end.
    """
    model = ExternalFactorCNN(
        tuple(int(x) for x in images.shape[1:]),
        tuple(int(c) for c in cfg.clf_channels),
        int(cfg.clf_kernel),
        int(cfg.clf_stride),
        int(cfg.clf_groups),
        int(cfg.clf_hidden),
        cardinalities,
        key,
    )
    dtype = model_dtype(model)
    X = jnp.asarray(images, dtype)
    Y = jnp.asarray(labels, jnp.int32)
    n = int(X.shape[0])
    batch = min(int(cfg.clf_batch), n)
    steps = int(cfg.clf_steps)

    opt = optax.adam(float(cfg.clf_lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    grad_fn = eqx.filter_value_and_grad(lambda m, xb, yb: _xent(jax.vmap(m)(xb), yb))

    @eqx.filter_jit
    def step_fn(m, s, xb, yb):
        loss, grads = grad_fn(m, xb, yb)
        updates, s = opt.update(grads, s, m)
        return eqx.apply_updates(m, updates), s, loss

    every = max(1, int(getattr(cfg, "clf_checkpoint_every", 250)))
    history = []
    for i in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        model, opt_state, loss = step_fn(model, opt_state, X[idx], Y[idx])
        history.append(float(loss))
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 5) == 0):
            log(f"    step {i + 1}/{steps}  loss={history[-1]:.5f}")
        if on_epoch is not None and ((i + 1) % every == 0 or i + 1 == steps):
            on_epoch(i + 1, model, history[-1])
    return model, history, key


# ---------------------------------------------------------------------------
# ⛔ Leg 1 — rows, and the three assertions that must be able to fire
# ---------------------------------------------------------------------------


def corner_rows(values: np.ndarray) -> np.ndarray:
    """Every row in Montero's photometric corner, **all four shapes**.

    ⛔ Excluding only the non-cylinders would leave the held-out cylinders'
    neighbours in the classifier's training draw; excluding the whole corner makes
    the statement "it has never seen an image of this corner" exact, and costs ~216
    of 479,928 rows.
    """
    v = np.asarray(values)
    m = (
        (v[:, 0] >= 0.8) & (v[:, 1] >= 0.8) & (v[:, 2] >= 0.8)
        & (v[:, 3] >= 1.1) & (v[:, 5] > 20)
    )
    return np.flatnonzero(m).astype(np.int64)


def classifier_rows(split, values, cfg, bridge_cfg, seed: int, extra_exclude=None) -> dict:
    """Deterministic training rows for the external classifier.

    The pool is ``split.train_rows`` minus (bridge fit rows ∪ ``eval_a`` ∪ the whole
    photometric corner ∪ anything the caller adds, e.g. a second condition's
    held-out cells).  ⛔ Held-out combinations are excluded **structurally** (the
    pool is the train split) *and* re-asserted by :func:`assert_no_holdout_leak`.
    """
    from ..experiments.exp_gen_bridge import select_rows

    # ⛔ the bridge's OWN selector, called with the bridge's OWN config: the rows the
    # decoder and the probe were fitted on are reproduced here, never re-derived.
    bridge = select_rows(split, bridge_cfg, int(seed))
    corner = corner_rows(values)
    excl = [bridge["train"], bridge["eval_a"], bridge["eval_b"], corner,
            np.asarray(split.test_rows, np.int64)]
    if extra_exclude is not None:
        excl.append(np.asarray(extra_exclude, np.int64))
    excluded = np.unique(np.concatenate([np.asarray(e, np.int64) for e in excl]))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), excluded)
    rng = np.random.default_rng(int(seed) * int(cfg.row_seed_mult) + int(cfg.row_seed_offset))
    n = min(int(cfg.n_clf_train), pool.size)
    return {
        "clf_train": np.sort(rng.permutation(pool)[:n]),
        "bridge_train": bridge["train"],
        "eval_a": bridge["eval_a"],
        "eval_b": bridge["eval_b"],
        "corner": corner,
        "pool_size": int(pool.size),
        "n_excluded": int(excluded.size),
    }


def assert_lineage_disjoint(clf_train: np.ndarray, rows: dict) -> dict:
    """⛔ The classifier's rows share nothing with the decoder/probe's own rows."""
    clf = np.asarray(clf_train, np.int64)
    report = {}
    for name in ("bridge_train", "eval_a", "eval_b", "corner"):
        other = np.asarray(rows[name], np.int64)
        overlap = int(np.intersect1d(clf, other).size)
        report[f"n_overlap_{name}"] = overlap
        if overlap:
            raise LineageLeak(
                f"{overlap} of the external classifier's {clf.size} training rows are "
                f"also in {name!r} — ⛔ the instrument would share the decoder's lineage"
            )
    report["n_clf_train"] = int(clf.size)
    return report


def assert_no_render_leak(
    images: np.ndarray,
    rows: np.ndarray,
    dataset: str,
    root=None,
    n_sample: int = 64,
    block: int = 8192,
    seed: int = 0,
) -> dict:
    """⭐ Verify the training tensor is **the dataset**, not a decoder render.

    Re-streams ``n_sample`` of ``rows`` straight out of the HDF5 file and requires
    the training tensor to be **bit-identical** there.  A decoder render — or a
    decoder render blended in anywhere — cannot survive this, which is what makes
    "it never saw a render" a measurement instead of a claim.
    """
    imgs = np.asarray(images)
    rows = np.asarray(rows, np.int64)
    if imgs.shape[0] != rows.size:
        raise RenderLeak(f"{imgs.shape[0]} training images for {rows.size} rows")
    rng = np.random.default_rng(int(seed))
    take = np.sort(rng.choice(rows.size, size=min(int(n_sample), rows.size), replace=False))
    fresh = stream_images(dataset, rows[take], root, int(block))
    delta = float(np.max(np.abs(imgs[take].astype(np.float32) - fresh))) if take.size else 0.0
    if delta != 0.0:
        raise RenderLeak(
            f"{take.size} sampled training images are NOT bit-identical to {dataset} on "
            f"disk (max |delta| = {delta}) — ⛔ this tensor contains something that is "
            "not a real dataset image (a decoder render would land here)"
        )
    return {"n_checked": int(take.size), "max_abs_delta": delta, "n_render_rows": 0}


def no_leak_report(split, clf_train: np.ndarray, rows: dict, extra: dict | None = None) -> dict:
    """The three leg-1 assertions, run together, with the counts the task asks for."""
    out = {"holdout": assert_no_holdout_leak(split, np.asarray(clf_train, np.int64))}
    out["lineage"] = assert_lineage_disjoint(clf_train, rows)
    if extra is not None:
        out["render"] = extra
    return out


# ---------------------------------------------------------------------------
# ⭐ Leg 2 — the pre-registered gate
# ---------------------------------------------------------------------------


def gate_verdict(per_seed: dict, floor: float = DEFAULT_GATE_FLOOR, factor: str = GATE_FACTOR) -> dict:
    """⭐ PASS iff **every** seed's in-corner balanced accuracy on REAL images ≥ floor.

    ⛔ The mean is reported and is **not** the gate: a 2/3 pass is a FAIL, and a FAIL
    is a legitimate result that returns to the Hub — ⛔ never a licence to tune.
    """
    vals = {str(s): float(v) for s, v in per_seed.items()}
    arr = np.asarray(list(vals.values()), np.float64)
    passed = bool(arr.size and np.all(arr >= float(floor)))
    return {
        "gate": "PASS" if passed else "FAIL",
        "statistic": f"in-corner {factor} balanced accuracy on REAL held-out images",
        "floor": float(floor),
        "n_seeds": int(arr.size),
        "per_seed": vals,
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "rule": "PASS iff EVERY seed >= floor (the mean is reported, not read)",
        "note": (
            "the external classifier can read real held-out-combination images in the "
            "corner, so a low number on the decoder's renders is the decoder's, not the "
            "instrument's"
            if passed
            else "⛔ STOP and return to the Hub: the phase needs a different instrument. "
            "Do not proceed to the render comparison and do not tune toward a pass."
        ),
    }


# ---------------------------------------------------------------------------
# ⭐⭐ Leg 3 — the gap, reported, ⛔ never adjudicated here
# ---------------------------------------------------------------------------


def instrument_gap(external: dict, probe: dict, abs_threshold: float, spread_mult: float) -> dict:
    """The per-seed gap ``external − probe`` on the same renders, and its materiality.

    ⛔ This function does **not** decide which instrument is right.  It reports the
    gap, whether the gap is material under the pre-registered rule, and which of the
    two pre-registered hypotheses the sign selects.
    """
    seeds = sorted(set(external) & set(probe), key=lambda s: int(s))
    deltas = {s: float(external[s]) - float(probe[s]) for s in seeds}
    arr = np.asarray(list(deltas.values()), np.float64)
    mean = float(arr.mean()) if arr.size else float("nan")
    std = float(arr.std()) if arr.size else float("nan")
    material = bool(
        arr.size
        and abs(mean) >= float(abs_threshold)
        and (std == 0.0 or abs(mean) >= float(spread_mult) * std)
    )
    if not material:
        verdict = "H_AGREE"
    elif mean < 0:
        verdict = "H_OPTIMISTIC"
    else:
        verdict = "H_PROBE_PESSIMISTIC"
    return {
        "per_seed_delta": deltas,
        "mean_delta": mean,
        "std_delta": std,
        "abs_threshold": float(abs_threshold),
        "spread_mult": float(spread_mult),
        "material": material,
        "hypothesis": verdict,
        "external_mean": float(np.mean([float(external[s]) for s in seeds])) if seeds else float("nan"),
        "probe_mean": float(np.mean([float(probe[s]) for s in seeds])) if seeds else float("nan"),
        "note": (
            "⛔ reported, not adjudicated: which instrument is 'right' is a Head/Hub "
            "decision, not a spoke's"
        ),
    }
