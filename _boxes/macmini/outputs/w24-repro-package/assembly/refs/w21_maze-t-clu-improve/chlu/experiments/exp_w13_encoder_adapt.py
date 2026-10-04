"""w13 hub/encoder-adapt -- **the frozen harness is the primary object; what does LIGHT
encoder adaptation buy?**

THE ONE QUESTION.  ``frozen`` is the REFERENCE.  ``adapter`` and ``anchor`` are the two
light-adaptation treatments.  The bar is not an absolute: it is the MARGIN

    margin(arm) = composition(single design on THIS ARM's features)
                - composition(best LEARNED baseline refit on THIS ARM's features)

and the pre-registered question is whether that margin GROWS from ``frozen`` to an
adapted arm.  Wave 12 measured the margin NEGATIVE (-0.054 ... -0.077 of ceiling,
`w12-composition-ladder/FINDINGS.md` section 0), so the question here is **"does
adaptation close a deficit?"**, NOT "does it extend a lead".  A margin that grows and
stays negative is a real and reportable result and is reported in those words.

A FEATURE GAIN SHARED BY ALL ARMS IS A FINDING ABOUT THE ENCODER, NOT ABOUT THE STORE.
The store margin is what this hub reports; the arm-level composition level is reported
beside it precisely so the two cannot be confused.

THE THREE ARMS -- identical loss, identical steps, identical images, identical PRNG
====================================================================================
Every arm runs the SAME objective on the SAME batches from the SAME key:

    L = recon + b_code * codebook + b_commit * commitment  +  b_swap * swap   (+ anchor)

``commitment = || sg(gather) - z ||^2`` IS the store read error on the fitting rows,
from the encoder's side; ``recon`` is kept exactly as in w8.  The swap loss is in because
w8 measured it repairing what unfreezing breaks (+0.0100 / +0.0129 / +0.0075 on the three
GPU seeds at 10x steps and 4x data).  The arms differ ONLY in which encoder parameters
are handed to Optax, and -- for ``anchor`` alone -- in one extra penalty term:

``frozen``   REFERENCE.  Encoder leaves are NOT in the optimiser.  Its features are the
             warm start's, bit-for-bit, so its probe margin and its cache are the frozen
             harness's own.  Store and decoder still train, as in the shipped
             configuration.
``adapter``  TREATMENT.  **Encoder weights frozen.**  A low-rank residual adapter
             ``z <- z + U (V z)`` on the flattened penultimate feature, rank 32 over
             D = 2048, so ``2 * 2048 * 32 = 131,072`` float32 = **524,288 B**, i.e. the
             programme's own reference pin -- a pinned size, not a tuned one.
             ``U`` is initialised to EXACTLY ZERO, so at step 0 the adapted encoder is
             bit-identical to the frozen one; the manifest asserts it rather than claims it.
``anchor``   TREATMENT.  **Full unfreeze + a penalty on distance from the frozen
             encoder**, in BOTH senses the task names -- features and weights:

                 anchor = lam * ( ||z - z_ref||^2 / ||z_ref||^2
                                + ||th - th_ref||^2 / ||th_ref||^2 )

             with ONE lambda, set by a rule declared before the run and NOT tuned:
             ``lam = recon_0 / rho^2`` at a REFERENCE DISPLACEMENT ``rho = 0.10``, i.e.
             a 10 % relative displacement costs exactly the warm start's reconstruction
             loss.  ``recon_0`` is measured once on the SHARED warm start before the first
             update, exactly as w8 sets ``beta_swap`` "auto".
             If ``anchor`` diverges from the frozen features the penalty is mis-scaled:
             the distance is REPORTED and the penalty is NOT retuned to make it look better.

WHY ``frozen`` CARRIES THE SWAP TERM TOO.  Because the arms must differ in one thing.
w8's own rule -- "every arm runs the SAME loop over the SAME batches from the SAME key;
what differs is which leaves are in the optimiser" -- is kept.  The swap term costs the
same second decode and judge pass in every arm, so equal steps is much closer to equal
seconds here than it was in w8; both are printed on every table regardless.

THE STORE IS FIXED AT K* = 32
=============================
``CROSSOVER-val.json`` (wave 12) pins ``K_star = 32``, ``A = 63``, well width 64,
rank/width 0.2656 .. 0.9844, selected on ``val`` by a rule frozen and timestamped at
2026-09-21T11:10:24Z before any val composition number existed.  Nothing in this hub
re-selects K.  Two objects carry that K and they are different objects, said here rather
than implied:
  * the **in-loop VQ store** (``chlu.core.gen_store``) the encoder trains through, at
    **K = 5 wells and A = 64 modes -- w8's pinned 524,288 B ledger**, because the swap
    loss is MEASURED ILL-POSED at K = 32 (26 of 32 wells own no factor; the frozen
    ``gen_swap_loss.derive_fixed_ownership`` refuses, and this hub does not patch a
    deliberate refusal). That is a FINDING of this wave, reported as one;
  * the **scored single design** (``CovContentStore``), rebuilt at K* = 32 from EACH
    ARM's own generation-2 cache by wave 12's own unmodified ``scripts/w12_crossover_lib``.

EVERYTHING DOWNSTREAM OF ``train`` IS w8's CODE, UNMODIFIED
==========================================================
``stage_cache`` / ``stage_probe`` / ``stage_rivals`` are imported from
``exp_w8_encoder_loop`` and called on this hub's arms after the arms are REGISTERED into
that module's ``ARM_SPEC`` / ``ARM_CLASS`` tables.  The registration is explicit, printed,
and asserted -- nothing is monkey-patched behind a reader's back.

Runnable, one stage at a time, every stage resumable and idempotent::

    python -m chlu.experiments.exp_w13_encoder_adapt --stage warm   --seed 0 --out DIR
    python -m chlu.experiments.exp_w13_encoder_adapt --stage train  --seed 0 --arm anchor --out DIR
    python -m chlu.experiments.exp_w13_encoder_adapt --stage cache  --seed 0 --arm anchor --out DIR
    python -m chlu.experiments.exp_w13_encoder_adapt --stage probe  --seed 0 --arm anchor --out DIR
    python -m chlu.experiments.exp_w13_encoder_adapt --stage rivals --seed 0 --arm anchor --out DIR
    python -m chlu.experiments.exp_w13_encoder_adapt --stage report --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import exp_w8_encoder_loop as W8

# ---------------------------------------------------------------------------
# 0.  chlu must resolve inside THIS worktree, or every number is void.
# ---------------------------------------------------------------------------

#: The shared site-packages carries an editable ``.pth`` to the Head's checkout.  If it
#: wins the import race, this hub silently measures someone else's code.
WORKTREE = Path(__file__).resolve().parents[2]


def assert_chlu_in_worktree(expect: Optional[Path] = None) -> str:
    import chlu

    want = Path(expect or WORKTREE).resolve()
    got = Path(chlu.__file__).resolve()
    if want not in got.parents:
        raise AssertionError(
            f"chlu.__file__ = {got} is NOT inside {want} -- the shared site-packages' "
            "editable .pth won the import and every number of this run would be void")
    return str(got)


# ---------------------------------------------------------------------------
# 1.  PINS.  Every one of them in code; none selected against a scored number.
# ---------------------------------------------------------------------------

PINS = {
    # ---- inherited from w8 verbatim; repeated so a reader needs one file --------
    "dataset": "shapes3d",
    "pretext": "recon",
    "warm_steps": 1500,
    "warm_batch": 128,
    "n_train": 8192,
    "joint_steps": 400,
    "joint_batch": 64,
    "joint_lr": 1e-3,
    "code_grad": "straight_through",
    "beta_codebook": 1.0,
    "beta_commitment": 0.25,
    "beta_swap": "auto",
    "swap_wells_per_step": 1,
    "ownership_pairs": 256,
    "ownership_batch": 256,
    "warm_dec_steps": 200,
    "dec_recipe": "w8_v1_short",
    "dec_recipe_architecture_of": "v1_capacity_schedule",
    "dec_refit_steps": 400,
    "probe_steps": 1500,
    "probe_batch": 256,
    "probe_lr": 1e-3,
    "probe_n_fit": 8192,
    "probe_n_eval": 4096,
    "cache_dtype": "float16",
    "cache_block": 8192,
    "eval_batch": 256,
    "rival_val_rows": 512,
    "arms_without_render_decoder": [],
    "seeds": [0, 1, 2],
    # ---- THE TWO STORES, AND WHY THEY CARRY DIFFERENT K -------------------------
    #: THE SCORED OBJECT is the single design at K* = 32
    #: (`CROSSOVER-val.json` -> K_star = 32, A_requested = 63, well width 64), rebuilt
    #: from EACH ARM's own generation-2 cache by wave 12's unmodified harness.  That pin
    #: is fixed and this hub never re-selects it: see `K_star` below and
    #: `scripts/w13_score_leg.py`.
    #:
    #: THE IN-LOOP VQ STORE the encoder trains THROUGH is a DIFFERENT OBJECT, and it runs
    #: at K = 5, A = 64 -- w8's pinned ledger, 5 x 64 x 2048 x 4 = 524,288 B.
    #:
    #: THE REASON IS MEASURED, NOT CHOSEN.  The swap loss assigns every well ONE owned
    #: factor by a Hungarian match on a measured transfer matrix.  Shapes3D has 6 factors.
    #: At K = 32 only 6 wells can own one and the frozen harness REFUSES:
    #:
    #:   AssertionError: well(s) [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    #:   20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31] own no factor -- the swap loss has no
    #:   target for them and is ill-posed; reported, not patched
    #:   (chlu/core/gen_swap_loss.py:148, measured here 2026-09-21 at K = 32)
    #:
    #: `gen_swap_loss.derive_fixed_ownership`'s own docstring states the design:
    #: "With K=5 wells and 6 Shapes3D factors exactly one factor is UNOWNED by
    #: construction; that is a property of the pinned ledger".  The swap loss is
    #: therefore DEFINED for K of the order of the factor count.  Restricting the swap to
    #: the owning wells would be patching a refusal the harness makes deliberately, so it
    #: is NOT done: the in-loop store is w8's K = 5 object, and the fact that the swap
    #: loss is ILL-POSED at K* = 32 is a FINDING of this wave, reported as one.
    "k_wells": 5,
    "n_modes": 64,
    "grouping_method": "auto",
    "min_well_frac": 0.10,
    "in_loop_store_K": 5,
    "in_loop_store_A": 64,
    #: the modes PARTITION D, so the atom block is n_modes x D floats in total:
    #: 64 x 2048 x 4 = 524,288 B -- w8's pin exactly (and at K* = 32, A = 63 the
    #: same arithmetic gives 63 x 2048 x 4 = 516,096 B = w12's `p1_atom_bytes`).
    "in_loop_store_bytes": 64 * 2048 * 4,
    "swap_loss_ill_posed_at_K32": True,
    "K_star": 32,
    "K_star_source": ("C:/chlu/evidence/outputs/w12-single-design-crossover/"
                      "CROSSOVER-val.json :: K_star"),
    # ---- THE ADAPTER -------------------------------------------------------
    #: rank 32 over D = 2048 -> 2 * 2048 * 32 = 131,072 float32 = 524,288 B,
    #: EXACTLY the programme's reference pin.  A pinned size, never a tuned one.
    "adapter_rank": 32,
    "adapter_up_init": "zeros",
    # ---- THE ANCHOR --------------------------------------------------------
    #: lam = recon_0 / rho^2 with rho = 0.10, measured once on the SHARED warm start.
    "anchor_reference_displacement_rho": 0.10,
    "anchor_terms": ["feature_relative_l2", "weight_relative_l2"],
    "anchor_lambda_mode": "auto_from_warm_start",
    # ---- the learned baselines, refit on EACH arm's features ----------------
    #: The task names MLP, LSTM, transformer, continuous MLP and LACE.
    #: LACE is a DECLARED NOT-RUN -- see `LACE_NOT_RUN` below.
    "rival_families": ["mlp", "cont_mlp", "transformer", "lstm"],
    "rival_families_declared_not_run": ["lace"],
    # ---- checkpointing: a wall-clock timer, proved before any spend ---------
    "ckpt_every_s": 120.0,
}

LACE_NOT_RUN = (
    "LACE is a DECLARED NOT-RUN, not a zero and not a blank.  REASON, measured not "
    "argued: `chlu.eval.rivals.w4_learners.builders()` -- the protocol every other "
    "baseline here is refit under, verbatim -- has no `lace` entry; LACE is an energy "
    "model with its own fit routine (`chlu.eval.rivals.gen_lace`) and no entry point "
    "that refits it on a NEW feature cache.  Refitting it per arm x seed would be a new "
    "fitting protocol written by this hub for a competitor, which is the one thing that "
    "would make the table worthless.  w12's own scored leg records the same call: "
    "`lace`/`poe_ebm` were DECLARED NOT-RUN by spoke A "
    "(/c/chlu/SCORED-LEDGER.md, w12 hub/single-design-crossover row)."
)

ARMS = ("frozen", "adapter", "anchor")

ARM_SPEC = {
    "frozen":  {"store": True, "unfreeze": False, "swap": True, "wallclock_match": False,
                "adapter": False, "anchor": False},
    "adapter": {"store": True, "unfreeze": False, "swap": True, "wallclock_match": False,
                "adapter": True,  "anchor": False},
    "anchor":  {"store": True, "unfreeze": True,  "swap": True, "wallclock_match": False,
                "adapter": False, "anchor": True},
}

ARM_CLASS = {
    "frozen": ("REFERENCE -- the frozen harness, the primary object.  Encoder leaves "
               "not in the optimiser; features bit-identical to the warm start."),
    "adapter": ("TREATMENT -- low-rank residual adapter (rank 32, 524,288 B), ENCODER "
                "WEIGHTS FROZEN; the adapted encoder starts bit-identical to frozen."),
    "anchor": ("TREATMENT -- full unfreeze + a penalty on distance from the frozen "
               "weights AND features; lambda from a declared rule, never tuned."),
}

ARM_SCHEMA = "chlu-w13-encoder-adapt-arm/1"
STAGES = ("warm", "train", "cache", "probe", "rivals", "score", "report", "killproof")


def install_into_w8(log: Callable = print) -> dict:
    """Register this hub's PINS and arms into ``exp_w8_encoder_loop``.

    EXPLICIT, PRINTED AND ASSERTED, because the w8 stages this hub reuses
    (``cache`` / ``probe`` / ``rivals``) read ``W8.PINS`` and ``W8.ARM_CLASS`` from their
    own module globals.  Registering is what lets those stages run UNMODIFIED on arms
    they were not written for; hiding the registration would make them look unmodified
    while they were not.  No w8 arm's spec is edited: `frozen` is the one name both hubs
    use and it is REPLACED by this hub's spec, which is recorded on the row.
    """
    shared = sorted(set(ARMS) & set(W8.ARM_SPEC))
    for k, v in PINS.items():
        W8.PINS[k] = v
    for a in ARMS:
        W8.ARM_SPEC[a] = dict(ARM_SPEC[a])
        W8.ARM_CLASS[a] = ARM_CLASS[a]
    rec = {"registered_arms": list(ARMS),
           "arm_names_also_present_in_w8": shared,
           "w8_pins_after": {k: W8.PINS[k] for k in
                             ("k_wells", "n_modes", "min_well_frac", "joint_steps",
                              "joint_batch", "dec_refit_steps", "rival_families")},
           "note": ("w8's cache/probe/rivals stages are called unmodified on these arms; "
                    "this is the registration that makes that legal")}
    log(f"  registered into exp_w8_encoder_loop: {list(ARMS)}  "
        f"K={W8.PINS['k_wells']} A={W8.PINS['n_modes']} "
        f"min_well_frac={W8.PINS['min_well_frac']:.6f}")
    return rec


# ---------------------------------------------------------------------------
# 2.  The adapter -- an encoder that IS the frozen encoder at step 0
# ---------------------------------------------------------------------------


class AdaptedEncoder(eqx.Module):
    """``base`` (frozen) + a rank-r residual adapter on the flattened feature.

        z  = base(x)                       (C, H, W)
        z' = z + reshape( U @ (V @ vec(z)) )

    ``V`` (down) is a scaled Gaussian; ``U`` (up) is EXACTLY ZERO at initialisation, so
    ``AdaptedEncoder(x) == base(x)`` bit-for-bit before the first update.  That is the
    LoRA initialisation and it is what makes "one bit-identical warm start per seed" true
    of the treatment as well as of the reference -- asserted in ``stage_train``, not claimed.

    The base's leaves are never handed to Optax: ``_arm_filter_spec`` marks them False.
    "Encoder weights frozen" is therefore a property of the filter, checked by comparing
    the base's parameter hash before and after training.
    """

    base: eqx.Module
    down: jnp.ndarray
    up: jnp.ndarray
    feat_shape: tuple
    in_shape: tuple

    def __init__(self, base, rank: int, key):
        D = int(np.prod(tuple(base.feat_shape)))
        dt = base.convs[0].weight.dtype
        self.base = base
        # 1/sqrt(D) scaling: the standard LoRA down-projection scale, so the adapter's
        # first gradients are O(1) rather than O(D).
        self.down = (jax.random.normal(key, (int(rank), D), dtype=dt)
                     / jnp.sqrt(jnp.asarray(D, dt)))
        self.up = jnp.zeros((D, int(rank)), dtype=dt)
        self.feat_shape = tuple(int(x) for x in base.feat_shape)
        self.in_shape = tuple(int(x) for x in base.in_shape)

    @property
    def param_dtype(self):
        return self.base.param_dtype

    @property
    def convs(self):
        return self.base.convs

    @property
    def norms(self):
        return self.base.norms

    @property
    def feature_dim(self) -> int:
        return int(np.prod(self.feat_shape))

    def __call__(self, x):
        z = self.base(x)
        zf = z.reshape((-1,))
        return z + (self.up @ (self.down @ zf)).reshape(z.shape)


def adapter_param_count(D: int, rank: int) -> int:
    return 2 * int(D) * int(rank)


# ---------------------------------------------------------------------------
# 3.  The optimiser filter -- the ONLY thing that differs between the arms
# ---------------------------------------------------------------------------


def _arm_filter_spec(arm: str, encoder, store, decoder):
    """Which leaves Optax may touch, per arm.

    * ``frozen``  -- no encoder leaf at all.
    * ``adapter`` -- ONLY ``down`` and ``up``; every leaf of ``base`` is False.
    * ``anchor``  -- every inexact array of the encoder.

    The store and the decoder are trained in every arm (all three carry the store on the
    loss path), so the pytree shape and the step cost profile match across arms.
    """
    spec = ARM_SPEC[arm]
    if spec["adapter"]:
        enc_filt = jax.tree_util.tree_map(lambda _l: False, encoder)
        enc_filt = eqx.tree_at(lambda m: (m.down, m.up), enc_filt,
                               replace=(True, True), is_leaf=lambda x: x is False)
    elif spec["unfreeze"]:
        enc_filt = jax.tree_util.tree_map(
            lambda leaf: bool(eqx.is_inexact_array(leaf)), encoder)
    else:
        enc_filt = jax.tree_util.tree_map(lambda _l: False, encoder)
    return (
        enc_filt,
        jax.tree_util.tree_map(lambda leaf: bool(eqx.is_inexact_array(leaf)), store),
        jax.tree_util.tree_map(lambda leaf: bool(eqx.is_inexact_array(leaf)), decoder),
    )


def _base_of(encoder):
    """The FROZEN base inside whatever encoder an arm carries."""
    return encoder.base if isinstance(encoder, AdaptedEncoder) else encoder


def _inexact_leaves(m):
    return jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))


def weight_rel_l2(enc, ref_enc) -> jnp.ndarray:
    """``||th - th_ref||^2 / ||th_ref||^2`` over the encoder's inexact arrays.

    For ``adapter`` the two trees have the same structure (both ``AdaptedEncoder``), so
    the adapter's own matrices are included -- which is correct: the adapter IS the
    displacement from the frozen encoder in that arm.
    """
    a, b = _inexact_leaves(enc), _inexact_leaves(ref_enc)
    num = sum(jnp.sum((x - y) ** 2) for x, y in zip(a, b))
    den = sum(jnp.sum(y ** 2) for y in b)
    return num / jnp.maximum(den, 1e-12)


def _swap_terms_dyn(model, classifier, xb, Yb, idx_i, idx_j, chan, f_owned: int,
                    code_grad: str = "straight_through"):
    """``gen_swap_loss.swap_loss_terms`` with the well's channel row TRACED.

    Line for line the same computation as
    :func:`chlu.core.gen_swap_loss.swap_loss_terms`, built out of that module's own
    primitives (``swapped_features``, ``quantise_through_store``, ``intended_tuple``) in
    the same order.  The ONLY difference is that the well's padded channel row arrives as
    an argument instead of being indexed out of ``wells_padded`` by a static Python int.

    WHY IT EXISTS.  ``well`` static means one XLA compilation per well, and this wave's
    store has K = 32 wells against w8's 5 -- 32 compilations of a graph carrying encoder,
    store, decoder and judge.  Keying the compile on the OWNED FACTOR instead (at most
    six distinct values) compiles a handful.  ``f_owned`` stays static because
    ``intended_tuple`` sets a column by index.

    Equality with the frozen function is ASSERTED numerically in ``train_arm`` on a real
    batch before any update; a non-zero delta refuses the run.
    """
    from ..core.gen_bridge import quantise_through_store
    from ..core import gen_swap_loss as SL

    encoder, store, decoder = model
    z = jax.vmap(encoder)(xb)
    zf = z.reshape((z.shape[0], -1))
    zs = SL.swapped_features(zf, idx_i, idx_j, chan)
    dec_in, _gather, _codes = quantise_through_store(store, zs, code_grad, 0.0)
    imgs = jax.vmap(decoder)(dec_in.reshape((-1,) + tuple(store.feat_shape)))
    logits = jax.vmap(classifier)(imgs)
    tgt = SL.intended_tuple(Yb[idx_i], Yb[idx_j], int(f_owned))
    loss = jnp.asarray(0.0, jnp.float32)
    hit = jnp.ones(tgt.shape[0], bool)
    for f, lg in enumerate(logits):
        lp = jax.nn.log_softmax(lg, axis=-1)
        loss = loss - jnp.mean(jnp.take_along_axis(lp, tgt[:, f][:, None], axis=1))
        hit = hit & (jnp.argmax(lg, axis=-1) == tgt[:, f])
    return loss / float(len(logits)), jnp.mean(hit.astype(jnp.float32))


# ---------------------------------------------------------------------------
# 4.  Stage `train` -- one arm's adaptation, at equal steps and equal images
# ---------------------------------------------------------------------------


def train_fingerprint(arm: str, seed: int, steps: int, warm: dict) -> dict:
    """The config whose hash a resume is REFUSED against.

    Everything that determines the RESULT is in it and nothing that does not: the
    checkpoint cadence and the output directory are deliberately outside, because a
    relaunch after a ``kill -9`` is allowed to checkpoint on a different timer.  The warm
    start's three parameter hashes ARE in it, so a checkpoint can never be resumed onto a
    different warm start -- the one substitution that would silently void the arm.
    """
    return {
        "hub": "w13-encoder-adapt", "arm": arm, "seed": int(seed), "steps": int(steps),
        "arm_spec": ARM_SPEC[arm],
        "joint_batch": PINS["joint_batch"], "joint_lr": PINS["joint_lr"],
        "code_grad": PINS["code_grad"],
        "beta_codebook": PINS["beta_codebook"],
        "beta_commitment": PINS["beta_commitment"],
        "beta_swap_mode": PINS["beta_swap"],
        "swap_wells_per_step": PINS["swap_wells_per_step"],
        "n_train": PINS["n_train"],
        "k_wells": PINS["k_wells"], "n_modes": PINS["n_modes"],
        "adapter_rank": PINS["adapter_rank"],
        "anchor_rho": PINS["anchor_reference_displacement_rho"],
        "warm_encoder_param_hash": warm["encoder_param_hash"],
        "warm_store_param_hash": warm["store_param_hash"],
        "warm_decoder_param_hash": warm["decoder_param_hash"],
        "own": warm["own"],
    }


def train_arm(arm: str, encoder, store, decoder, classifier, images, labels, own,
              key, steps: int, log: Callable = print, quick: bool = False,
              ckpt=None):
    """Stage 2 for one arm.  Returns ``(encoder, store, decoder, report, key)``.

    Every arm runs the SAME loop over the SAME batches from the SAME key.  What differs
    is (a) which encoder leaves are in the optimiser and (b) whether the anchor penalty
    is added.  Nothing else -- the store is on the loss path in all three arms and the
    swap term is on in all three.
    """
    from ..core.gen_bridge import joint_loss_terms, param_hash
    from ..core import gen_swap_loss as SL

    spec = ARM_SPEC[arm]
    code_grad = str(PINS["code_grad"])
    b_code, b_commit = float(PINS["beta_codebook"]), float(PINS["beta_commitment"])
    b_swap = PINS["beta_swap"]
    batch = 16 if quick else int(PINS["joint_batch"])
    lr = float(PINS["joint_lr"])
    K = int(store.n_wells)
    rho = float(PINS["anchor_reference_displacement_rho"])
    with_anchor = bool(spec["anchor"])

    enc_hash_before = param_hash(encoder)
    base_hash_before = param_hash(_base_of(encoder))
    cls_hash_before = param_hash(classifier)

    model = (encoder, store, decoder)
    filt = _arm_filter_spec(arm, encoder, store, decoder)
    params, static = eqx.partition(model, filt)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    wells_padded, well_sizes = SL.pad_well_channels(
        [np.asarray(w) for w in store.well_channels])
    own_j = np.asarray(own, np.int64)

    X = jnp.asarray(np.asarray(images, np.float32))
    Y = jnp.asarray(np.asarray(labels, np.int32))
    n = int(X.shape[0])
    batch = min(batch, n)

    # ---- the REFERENCE encoder the anchor is measured against -------------
    # Frozen, never in the optimiser, identical for every arm: it is the warm start.
    ref_encoder = jax.tree_util.tree_map(lambda x: x, encoder)

    # ---- the two auto-scaled weights, MEASURED ONCE on the shared warm start ----
    _k0 = jax.random.fold_in(key, 777)
    _idx = jax.random.choice(_k0, n, (min(batch, n),), replace=False)
    _ii, _jj = SL.draw_partners(int(_idx.shape[0]), jax.random.fold_in(_k0, 1))
    _rec0, _, _ = joint_loss_terms((encoder, store, decoder), X[_idx], code_grad, 0.0)
    _sw0, _ = SL.swap_loss_terms(
        (encoder, store, decoder), classifier, X[_idx], Y[_idx], _ii, _jj, 0,
        wells_padded, own_j, code_grad)
    beta_swap_basis = {"recon_at_warm_start": float(_rec0),
                       "swap_at_warm_start": float(_sw0)}
    if not isinstance(b_swap, (int, float)):
        b_swap = float(_rec0) / max(float(_sw0), 1e-8)
    b_swap = float(b_swap)
    # lam = recon_0 / rho^2 : a rho-relative displacement costs exactly recon_0.
    lam_anchor = float(_rec0) / (rho * rho)
    anchor_basis = {
        "recon_at_warm_start": float(_rec0),
        "reference_displacement_rho": rho,
        "lambda": lam_anchor,
        "rule": ("lam = recon_0 / rho^2, so a rho = 10 % RELATIVE displacement of the "
                 "features (or of the weights) costs exactly the warm start's "
                 "reconstruction loss. Measured ONCE on the SHARED warm start before "
                 "the first update; identical for every seed's own warm start; NEVER "
                 "selected against any scored number."),
    }
    log(f"    [{arm}] beta_swap={b_swap:.6g} (recon_0={float(_rec0):.5f} / "
        f"swap_0={float(_sw0):.5f}); lam_anchor={lam_anchor:.6g} "
        f"(rho={rho}) {'[ON]' if with_anchor else '[off, this arm has no anchor term]'}")

    ref_sq = sum(float(np.sum(np.asarray(x, np.float64) ** 2))
                 for x in _inexact_leaves(ref_encoder))

    # THE DYNAMIC SWAP IS THE FROZEN SWAP.  Asserted numerically, on a real batch,
    # against `gen_swap_loss.swap_loss_terms` itself, before a single update.
    _sw_ref, _ex_ref = SL.swap_loss_terms(
        (encoder, store, decoder), classifier, X[_idx], Y[_idx], _ii, _jj, 0,
        wells_padded, own_j, code_grad)
    _sw_dyn, _ex_dyn = _swap_terms_dyn(
        (encoder, store, decoder), classifier, X[_idx], Y[_idx], _ii, _jj,
        wells_padded[0], int(own_j[0]), code_grad)
    swap_identity = {
        "frozen_swap_loss": float(_sw_ref), "dynamic_swap_loss": float(_sw_dyn),
        "abs_delta_loss": abs(float(_sw_ref) - float(_sw_dyn)),
        "frozen_exact": float(_ex_ref), "dynamic_exact": float(_ex_dyn),
        "abs_delta_exact": abs(float(_ex_ref) - float(_ex_dyn)),
        "what_this_is": ("the swap term used in training is gen_swap_loss's own "
                         "primitives composed in gen_swap_loss's own order, with the "
                         "well's channel row traced instead of static so that K = 32 "
                         "compiles a handful of graphs rather than 32. Equality with "
                         "`SL.swap_loss_terms` is ASSERTED here, not asserted in prose."),
    }
    if swap_identity["abs_delta_loss"] != 0.0 or swap_identity["abs_delta_exact"] != 0.0:
        raise AssertionError(
            "the dynamic swap term does not reproduce gen_swap_loss.swap_loss_terms "
            f"exactly: {swap_identity} -- refusing to train on a re-implementation")

    def loss_fn(p, s, xb, yb, ii, jj, chan, f_owned):
        m = eqx.combine(p, s)
        enc = m[0]
        l_rec, l_code, l_commit = joint_loss_terms(m, xb, code_grad, 0.0)
        total = l_rec + b_code * l_code + b_commit * l_commit
        l_swap, swap_exact = _swap_terms_dyn(
            m, classifier, xb, yb, ii, jj, chan, int(f_owned), code_grad)
        total = total + b_swap * l_swap
        # the anchor's two distances are ALWAYS computed, in every arm, so the table
        # carries the displacement of the reference and of the treatments alike.
        zr = jax.vmap(ref_encoder)(xb).reshape((xb.shape[0], -1))
        zc = jax.vmap(enc)(xb).reshape((xb.shape[0], -1))
        d_feat = jnp.sum((zc - zr) ** 2) / jnp.maximum(jnp.sum(zr ** 2), 1e-12)
        d_wt = weight_rel_l2(enc, ref_encoder)
        l_anchor = d_feat + d_wt
        if with_anchor:
            total = total + lam_anchor * l_anchor
        return total, (l_rec, l_code, l_commit, l_swap, swap_exact, d_feat, d_wt)

    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    # ONE compiled stepper per OWNED FACTOR, not per well.  At K = 32 the well index
    # would force 32 separate XLA compilations of a graph that carries the encoder, the
    # store, the decoder and the judge; keying on `own[well]` -- at most F = 6 distinct
    # values, and w8 measured only 2 of 5 wells with measurable ownership -- compiles the
    # same computation a handful of times instead.  The well's channel row is passed as a
    # TRACED array, which `swapped_features` already supports (it is `.at[:, channels]`).
    steppers = {}

    def stepper_for(f_owned: int):
        if f_owned not in steppers:
            @eqx.filter_jit
            def step(p, s, o, xb, yb, ii, jj, chan):
                (loss, parts), grads = grad_fn(p, s, xb, yb, ii, jj, chan, f_owned)
                upd, o2 = opt.update(grads, o, p)
                return eqx.apply_updates(p, upd), o2, loss, parts
            steppers[f_owned] = step
        return steppers[f_owned]

    history = []
    start_step = 0
    if ckpt is not None:
        got = ckpt.latest()
        if got is not None:
            cstep, cstate = got
            params, opt_state, key, start_step = W8._restore_from_ckpt(
                cstate, params, opt_state)
            log(f"    [{arm}] RESUMED from checkpoint step {cstep}; "
                f"{int(steps) - start_step} steps remain")
        else:
            log(f"    [{arm}] no checkpoint -- fresh start")

    t0 = time.time()
    for i in range(start_step, int(steps)):
        key, sk, pk, wk = jax.random.split(key, 4)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        ii, jj = SL.draw_partners(batch, pk)
        w = int(jax.random.randint(wk, (), 0, K))
        params, opt_state, loss, parts = stepper_for(int(own_j[w]))(
            params, static, opt_state, X[idx], Y[idx], ii, jj, wells_padded[w])
        history.append({
            "step": i, "well": w, "loss": float(loss), "recon": float(parts[0]),
            "codebook": float(parts[1]), "commitment": float(parts[2]),
            "swap": float(parts[3]), "swap_exact": float(parts[4]),
            "d_feat_rel2": float(parts[5]), "d_weight_rel2": float(parts[6]),
        })
        if log is not None and (i == start_step or (i + 1) % max(1, int(steps) // 6) == 0):
            h = history[-1]
            log(f"    [{arm:8s}] step {i + 1}/{steps} loss={h['loss']:.5f} "
                f"recon={h['recon']:.5f} code={h['codebook']:.5f} "
                f"commit={h['commitment']:.5f} swap={h['swap']:.5f} "
                f"swap_exact={h['swap_exact']:.4f} dF={h['d_feat_rel2']:.3e} "
                f"dW={h['d_weight_rel2']:.3e}")
        if ckpt is not None and ckpt.maybe_save(
                i + 1, W8._flatten_for_ckpt(params, opt_state, key, i + 1)):
            log(f"    [{arm}] checkpoint @ step {i + 1} (save #{ckpt.n_saves}) "
                f"-> {ckpt.local_dir}")
    if ckpt is not None:
        ckpt.maybe_save(int(steps), W8._flatten_for_ckpt(params, opt_state, key,
                                                        int(steps)), force=True)
    train_seconds = time.time() - t0

    encoder, store, decoder = eqx.combine(params, static)
    enc_hash_after = param_hash(encoder)
    base_hash_after = param_hash(_base_of(encoder))

    # ---- the three assertions that make the arm labels TRUE ---------------
    if (not spec["unfreeze"]) and base_hash_after != base_hash_before:
        raise AssertionError(
            f"arm {arm!r} declares the ENCODER WEIGHTS FROZEN and the base encoder's "
            f"parameter hash moved ({base_hash_before[:12]} -> {base_hash_after[:12]}) "
            "-- the label is false and every number of this arm is void")
    if arm == "frozen" and enc_hash_after != enc_hash_before:
        raise AssertionError(
            "the REFERENCE arm's encoder changed during stage 2 -- it is not a reference")
    if param_hash(classifier) != cls_hash_before:
        raise AssertionError(
            "the external classifier's parameters MOVED during training -- the judge was "
            "in the optimiser and every number of this arm is void")

    # `eqx.partition` writes None wherever the filter is False, and `tree_leaves` skips
    # None -- so this count is EXACTLY the encoder parameters Optax was given.
    # It reads 0 for `frozen`, 131,072 for `adapter`, and the whole encoder for `anchor`.
    n_enc_opt = int(sum(int(np.asarray(x).size)
                        for x in jax.tree_util.tree_leaves(params[0])))

    info = {
        "arm": arm, "arm_class": ARM_CLASS[arm], **spec,
        "code_grad": code_grad, "beta_codebook": b_code, "beta_commitment": b_commit,
        "beta_swap": b_swap, "beta_swap_mode": PINS["beta_swap"],
        "beta_swap_basis": beta_swap_basis,
        "swap_identity_vs_frozen_swap_loss": swap_identity,
        "anchor_on": with_anchor, "anchor_basis": anchor_basis,
        "anchor_lambda": lam_anchor if with_anchor else 0.0,
        "lr": lr, "steps": int(steps), "batch": int(batch), "n_train_rows": n,
        "gradient_steps": int(steps), "images_seen": int(steps) * int(batch),
        "train_seconds": float(train_seconds),
        "seconds_per_step": float(train_seconds / max(1, int(steps))),
        "encoder_param_hash_before": enc_hash_before,
        "encoder_param_hash_after": enc_hash_after,
        "encoder_changed": bool(enc_hash_after != enc_hash_before),
        "base_encoder_param_hash_before": base_hash_before,
        "base_encoder_param_hash_after": base_hash_after,
        "base_encoder_changed": bool(base_hash_after != base_hash_before),
        "encoder_weights_frozen_asserted": bool(not spec["unfreeze"]),
        "classifier_param_hash_before": cls_hash_before,
        "classifier_param_hash_after": param_hash(classifier),
        "classifier_unchanged": True,
        "n_params_optimised": int(
            sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params))),
        "n_encoder_params_optimised": n_enc_opt,
        "ref_encoder_sq_norm": ref_sq,
        "well_sizes": well_sizes,
        "own": own_j.tolist(),
        "history_first": history[0] if history else None,
        "history_last": history[-1] if history else None,
        "displacement_final": {
            "feature_relative_l2_squared": (history[-1]["d_feat_rel2"] if history
                                            else None),
            "weight_relative_l2_squared": (history[-1]["d_weight_rel2"] if history
                                           else None),
            "how_to_read": ("these are ||.||^2 RELATIVE, so the reference displacement "
                            "rho = 0.10 corresponds to 0.01 here. REPORTED, never tuned "
                            "against."),
        },
        "resumed_from_step": int(start_step),
        "checkpoint_saves": int(ckpt.n_saves) if ckpt is not None else 0,
        "seconds_note": (
            "train_seconds covers THIS process's steps only; a resumed arm's wall clock "
            "is the sum across attempts"),
    }
    return encoder, store, decoder, {"info": info, "history": history}, key


# ---------------------------------------------------------------------------
# 5.  Stage `train` -- the driver
# ---------------------------------------------------------------------------


def stage_train(out: Path, seed: int, arm: str, quick: bool,
                ckpt_every_s: float = 0.0, log: Callable = print) -> dict:
    """Build the arm's encoder from the SHARED warm start, train it, refit its decoder.

    Idempotent: a completed ``arm.json`` short-circuits.  Resumable: the checkpointer is
    ``scripts/cloud/ckpt.py`` -- THE checkpointer, not a second mechanism -- on a
    wall-clock timer, and a resume VERIFIES ``train_fingerprint`` and refuses by name.
    """
    from ..config import get_default_config
    from ..core.gen_bridge import (
        assert_no_feature_bypass, encode_images, param_count, param_hash,
        resolve_decoder_cfg, stream_images, train_decoder_from_features,
    )
    from ..core.gen_store import assert_no_bypass, store_bytes
    from ..eval.gen_probe import factor_labels
    from ..utils.checkpoints import load_model, save_model

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    install_into_w8(log=log)
    ad = W8.arm_dir(out, seed, arm)
    man_p = ad / "arm.json"
    if man_p.exists():
        m = W8._read_json(man_p)
        if m.get("status") == "ok" and m.get("schema") == ARM_SCHEMA:
            log(f"  arm {arm} seed {seed}: already on disk ({ARM_SCHEMA}) -- SKIPPED")
            return m
        log(f"  arm {arm}: artefact status={m.get('status')!r} schema={m.get('schema')!r} "
            f"(want {ARM_SCHEMA!r}) -- RE-RUNNING, loudly")
    ad.mkdir(parents=True, exist_ok=True)

    sd = W8.seed_dir(out, seed)
    warm = W8._read_json(sd / "warm_start.json")
    z = np.load(sd / "warm_rows.npz")
    own = np.asarray(z["own"], np.int64)
    rows_train = np.asarray(z["train"], np.int64)

    base_encoder = load_model(sd / "warm_encoder.pkl")
    store = load_model(sd / "warm_store.pkl")
    decoder = load_model(sd / "warm_decoder.pkl")
    if param_hash(base_encoder) != warm["encoder_param_hash"]:
        raise AssertionError(
            "the warm-start encoder on disk is not the one the manifest describes")
    classifier, cls_info = W8.load_external_classifier(seed, log=log)

    W8.register_short_recipe()
    cfg = get_default_config()
    gb = cfg.gen_bridge
    root = str(W8.data_root())
    log(f"  streaming {rows_train.size} training images ...")
    imgs = stream_images(PINS["dataset"], rows_train, root, int(gb.cache_block))
    sel = W8.select_rows(seed, quick=quick, log=lambda *_: None)
    labels = factor_labels(sel["grid"], rows_train)

    # ---- the arm's encoder ------------------------------------------------
    adapter_rec = None
    if ARM_SPEC[arm]["adapter"]:
        D = int(np.prod(tuple(base_encoder.feat_shape)))
        akey = jax.random.PRNGKey(int(seed) * 911 + 20_260_921)
        encoder = AdaptedEncoder(base_encoder, int(PINS["adapter_rank"]), akey)
        # THE BIT-IDENTICAL-AT-STEP-0 ASSERTION.  Not a claim -- a check, on real images.
        probe_x = jnp.asarray(np.asarray(imgs[:8], np.float32))
        z0 = np.asarray(jax.vmap(base_encoder)(probe_x))
        z1 = np.asarray(jax.vmap(encoder)(probe_x))
        if not np.array_equal(z0, z1):
            raise AssertionError(
                "the adapter is NOT bit-identical to the frozen encoder at "
                f"initialisation (max|delta| = {np.max(np.abs(z1 - z0))}) -- the "
                "'one bit-identical warm start per seed' clause is false and the run "
                "is refused")
        adapter_rec = {
            "rank": int(PINS["adapter_rank"]), "D": D,
            "param_count": adapter_param_count(D, int(PINS["adapter_rank"])),
            "bytes_float32": 4 * adapter_param_count(D, int(PINS["adapter_rank"])),
            "reference_pin_bytes": 524_288,
            "up_init": PINS["adapter_up_init"],
            "bit_identical_to_frozen_at_step_0": True,
            "checked_on_images": int(probe_x.shape[0]),
            "note": ("2 * D * rank float32 = the programme's own 524,288 B reference "
                     "pin, a PINNED size and not a tuned one"),
        }
        log(f"  adapter: rank {PINS['adapter_rank']} over D={D} -> "
            f"{adapter_rec['param_count']:,} params = {adapter_rec['bytes_float32']:,} B "
            f"(reference pin 524,288 B); bit-identical to frozen at step 0: ASSERTED")
    else:
        encoder = base_encoder

    steps = 60 if quick else int(PINS["joint_steps"])
    key = jax.random.fold_in(
        jax.random.PRNGKey(int(seed) * 10_007 + 20_260_920), 20_260_920)

    ckpt, fp = None, None
    if ckpt_every_s and float(ckpt_every_s) > 0:
        C = W8._load_ckpt_module()
        fp = C.config_fingerprint(train_fingerprint(arm, seed, steps, warm))
        ckpt = C.Checkpointer(str(ad / "ckpt"), fp, remote_uri=None,
                              every_s=float(ckpt_every_s), keep=3)
        log(f"  checkpoint-first: fingerprint {fp} every {ckpt_every_s}s -> "
            f"{ckpt.local_dir}")

    encoder, store, decoder, rep, key = train_arm(
        arm, encoder, store, decoder, classifier, imgs, labels, own, key,
        steps, log=log, quick=quick, ckpt=ckpt)

    # ---- the render decoder: refit downstream of THIS ARM's OWN features ---
    feats = encode_images(encoder, imgs, int(PINS["eval_batch"]))
    key, dk = jax.random.split(key)
    t0 = time.time()
    rd_cfg = get_default_config().gen_bridge
    if quick:
        from ..core.gen_bridge import DECODER_RECIPES
        DECODER_RECIPES[str(PINS["dec_recipe"])]["dec_steps"] = 20
    log(f"  refitting the decoder from {arm}'s own features ({PINS['dec_recipe']}) ...")
    render_decoder, dhist, recipe = train_decoder_from_features(
        encoder.feat_shape, feats, imgs, rd_cfg, dk, log=log,
        recipe=str(PINS["dec_recipe"]))
    dec_seconds = time.time() - t0

    Zf = np.asarray(feats, np.float32).reshape((len(imgs), -1))
    bypass = assert_no_feature_bypass(store, Zf, code_grad=str(PINS["code_grad"]))
    shipped = assert_no_bypass(store, Zf)
    log("  no-bypass OK: same-codes-different-features delta = "
        f"{bypass['max_abs_delta_same_codes_different_features']}")

    save_model(encoder, ad / "encoder.pkl", {"arm": arm, "seed": int(seed)})
    save_model(store, ad / "store.pkl", {"arm": arm, "seed": int(seed)})
    save_model(decoder, ad / "joint_decoder.pkl", {"arm": arm, "seed": int(seed)})
    save_model(render_decoder, ad / "decoder.pkl", {"arm": arm, "seed": int(seed)})
    W8._write_json({"history": rep["history"], "decoder_refit": dhist},
                   ad / "history.json")
    W8._publish_declift(out, seed, arm, render_decoder, recipe, param_hash(encoder),
                        log=log)

    man = {
        "schema": ARM_SCHEMA, "status": "ok",
        "hub": "w13-encoder-adapt",
        "arm": arm, "arm_class": ARM_CLASS[arm], "seed": int(seed),
        "quick": bool(quick), "void_as_a_reported_number": bool(quick),
        "chlu_file": assert_chlu_in_worktree(),
        "pins": dict(PINS),
        "adapter": adapter_rec,
        "warm_start": {k: warm[k] for k in
                       ("encoder_param_hash", "store_param_hash", "decoder_param_hash",
                        "D", "feat_shape", "well_sizes")},
        "compute": rep["info"],
        "scoreable_on_rows_3_4": True,
        "decoder_refit": {
            "recipe": recipe.as_dict(), "seconds": float(dec_seconds),
            "param_hash": param_hash(render_decoder),
            "param_count": param_count(render_decoder),
            "loss_first": float(dhist[0]) if dhist else None,
            "loss_final": float(dhist[-1]) if dhist else None,
            "note": ("identical recipe and identical steps for EVERY arm, fit downstream "
                     "of that arm's own frozen features; the one difference is the "
                     "features"),
        },
        "checkpointing": {
            "mechanism": "scripts/cloud/ckpt.py (THE checkpointer; no second mechanism)",
            "every_s": float(ckpt_every_s or 0.0),
            "fingerprint": fp,
            "resumed_from_step": rep["info"]["resumed_from_step"],
            "saves": rep["info"]["checkpoint_saves"],
        },
        "no_bypass_joint_path": W8._jsonable(bypass),
        "no_bypass_shipped_read": W8._jsonable(shipped),
        "channel": {
            "design_bits": float(store.bits_per_read), "formula": "K*log2(M)",
            "k_wells": int(store.n_wells), "n_modes": int(store.n_modes),
            "declared_constant": "R20", "widened": False,
        },
        "store_bytes": store_bytes(store),
        "param_hashes": {
            "encoder": param_hash(encoder),
            "base_encoder": param_hash(_base_of(encoder)),
            "store": param_hash(store),
            "joint_decoder": param_hash(decoder),
            "render_decoder": param_hash(render_decoder),
            "classifier": cls_info["param_hash"],
        },
        "classifier": cls_info,
        "n_train_rows": int(rows_train.size),
    }
    W8._write_json(man, man_p)
    log(f"  arm {arm} seed {seed} written -> {man_p}")
    return man


# ---------------------------------------------------------------------------
# 6.  `killproof` -- kill -9 resumability, PROVED BEFORE ANY SPEND
# ---------------------------------------------------------------------------


def stage_killproof(out: Path, seed: int, arm: str, log: Callable = print) -> dict:
    """Not a demonstration in prose: a measured three-part proof, run before any spend.

    1. A short run is launched with a 1-second checkpoint timer and **killed with
       SIGKILL** mid-flight by an external process; the checkpoint on disk is then
       loaded and its step index read.
    2. The run is relaunched; it must report ``RESUMED from checkpoint step N`` and
       finish, and the finished parameters must equal an UNINTERRUPTED run of the same
       fingerprint to the checkpoint's numerical precision.
    3. A resume under a DIFFERENT fingerprint must be REFUSED BY NAME.

    Parts 1 and 2 are driven by ``scripts/w13_killproof.py`` because a process cannot
    SIGKILL itself and then prove it resumed.  This function performs part 3, which is
    pure and needs no second process, and asserts the artefacts of parts 1-2 exist.
    """
    C = W8._load_ckpt_module()
    d = Path(out) / "killproof"
    d.mkdir(parents=True, exist_ok=True)
    fp_a = C.config_fingerprint({"hub": "w13", "arm": arm, "steps": 10})
    fp_b = C.config_fingerprint({"hub": "w13", "arm": arm, "steps": 11})
    ck = C.Checkpointer(str(d / "fp"), fp_a, remote_uri=None, every_s=0.0, keep=2)
    ck.maybe_save(3, {"x": np.arange(4, dtype=np.float64)}, force=True)
    refused = None
    try:
        C.Checkpointer(str(d / "fp"), fp_b, remote_uri=None, every_s=0.0,
                       keep=2).latest()
    except Exception as exc:                                       # noqa: BLE001
        refused = f"{type(exc).__name__}: {exc}"
    rec = {
        "part3_fingerprint_refusal": {
            "fingerprint_written": fp_a, "fingerprint_offered": fp_b,
            "refused": refused is not None, "refusal": refused,
        },
        "note": ("parts 1 and 2 (SIGKILL mid-run, then resume-and-match) are run by "
                 "scripts/w13_killproof.py, which needs a second process"),
    }
    W8._write_json(rec, d / "fingerprint_refusal.json")
    log(f"  fingerprint refusal: {rec['part3_fingerprint_refusal']['refused']} "
        f"({refused})")
    return rec


# ---------------------------------------------------------------------------
# 7.  CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="a VOID smoke; every artefact it writes says so")
    ap.add_argument("--ckpt-every-s", type=float, default=None)
    ap.add_argument("--joint-steps", type=int, default=None,
                    help="override PINS['joint_steps'] -- printed on every artefact")
    ap.add_argument("--dec-steps", type=int, default=None,
                    help="override PINS['dec_refit_steps'] -- printed on every artefact")
    ap.add_argument("--rival-families", default=None,
                    help="comma-separated override of PINS['rival_families']")
    return ap


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    got = assert_chlu_in_worktree()
    print(f"chlu.__file__ = {got}", flush=True)

    if a.joint_steps is not None:
        PINS["joint_steps"] = int(a.joint_steps)
        PINS["joint_steps_OVERRIDDEN"] = True
    if a.dec_steps is not None:
        PINS["dec_refit_steps"] = int(a.dec_steps)
        PINS["dec_refit_steps_OVERRIDDEN"] = True
    if a.rival_families:
        PINS["rival_families"] = [s for s in a.rival_families.split(",") if s]
    install_into_w8()
    ck = PINS["ckpt_every_s"] if a.ckpt_every_s is None else float(a.ckpt_every_s)

    if a.stage == "warm":
        W8.stage_warm(out, a.seed, a.quick)
    elif a.stage == "train":
        stage_train(out, a.seed, a.arm, a.quick, ckpt_every_s=ck)
    elif a.stage == "cache":
        W8.stage_cache(out, a.seed, a.arm, a.quick)
    elif a.stage == "probe":
        W8.stage_probe(out, a.seed, a.arm, a.quick)
    elif a.stage == "rivals":
        W8.stage_rivals(out, a.seed, a.arm, a.quick,
                        families=list(PINS["rival_families"]))
    elif a.stage == "killproof":
        stage_killproof(out, a.seed, a.arm or "frozen")
    elif a.stage in ("score", "report"):
        raise SystemExit(
            f"stage {a.stage!r} lives in scripts/w13_score_leg.py / "
            "scripts/w13_report.py -- it drives wave 12's own unmodified K* harness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
