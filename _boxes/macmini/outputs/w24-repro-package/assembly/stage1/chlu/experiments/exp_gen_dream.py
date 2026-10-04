"""Experiment: **conditional dreaming** by clamped-well Langevin, with an audit
trail that can convict the sampler.

⭐ DIAL DECLARATION — *isolation*: does clamping the wells a request addresses
actually steer what comes out?
- **Laundering control:** ⭐ the **zero-step rollout** — decode the clamp directly,
  no Langevin, same clamp, same seeds.  ⛔ It runs on every reported number.
- **Falsifies:** zero-step matches the full rollout ⇒ the sampler is inert and we
  are decoding the clamp.  Also falsifying: flat energy traces, or a chain that
  never leaves its initialisation.
- **Does NOT falsify:** blurry or low-fidelity samples in absolute terms — the
  frozen decoder's measured ceiling bounds that, and it is read as a ratio (R3.2),
  never absolutely.

⭐ **The acceptance criterion is the CONTRAST, not the accuracy.**  A clamped-well
rollout that simply decodes its own clamp produces respectable accuracy on seen
combinations while performing no inference at all.

⛔ **A-CELL SMOKE ONLY.**  Seen combinations, reported as a fraction of the
bridge's ``ceiling_A`` **and** of the store ceiling (R23.5, two ceilings on every
table).  ⛔ No B, no C, no kill-rule verdict — the cell belongs to
``gen-cell-run``, after the prereg is filed.

⛔ **THE DECODER TRAP, CHECKED FIRST.**  ``chlu/config.py``'s ``dec_recipe``
defaults to ``"baseline"`` — the **superseded** decoder that FAILED the gate
(0.7191 balanced accuracy, 0/3 seeds).  The winner is ``v1_capacity_schedule``
(0.9414, 3/3).  This experiment loads ``gen-decoder-lift``'s winner checkpoint and
**asserts its parameter hash and recipe name** before any number is recorded; an
arm decoded through the baseline looks entirely normal in the logs and is void.

Runnable:
    chlu exp-gen-dream [--dataset shapes3d] [--pretext recon] [--quick]
    uv run python -m chlu.experiments.exp_gen_dream \\
        --dataset shapes3d --pretext recon --stage dream --arm clu --seed 0 --out DIR
"""
from __future__ import annotations
from typing import Callable
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from ..config import CHLUConfig
from ..core.gen_store import channel_correlation, dead_channels, grouping_labels
def resolve_grouping(plan, X_store, config: CHLUConfig, seed: int,
                     encoder_hash: str, log: Callable = print) -> tuple:
    """⛔⭐ The grouping, resolved per seed — and why it CANNOT be the plan's vector.

    ``store_plan.json``'s ``labels`` vector indexes **encoder channels**, and the
    plan was written from the ``k_choice_seed``'s encoder alone (``i == 0`` in
    ``exp_gen_store``).  Every bridge seed trained its **own** encoder, so channel
    ``c`` of seed 1 has no relation to channel ``c`` of seed 0: applying the plan's
    labels to another seed is a size-matched **random** partition wearing the
    clustered grouping's name.  ⛔ ``gen-store-grouping`` itself did not do this --
    it re-derived the grouping per seed -- but the plan's "load this VERBATIM"
    wording invites exactly that mistake.

    What IS pinned by the plan is what ``assert_plan_matched`` actually guards:
    ``K``, ``n_modes``, ``D`` and ``store_bytes``.  The plan's own note says only
    ``labels`` may change.  So the grouping is re-derived here by the *same*
    procedure, on this seed's own train activations, at the plan's ``K``.

    ⭐ And the re-derivation is **verified, not asserted**: on the seed whose
    encoder hash matches the plan's, the labels must come back **bit-identical to
    the plan's**, which proves this is the same procedure that wrote it.
    """
    cfg = config.gen_dream
    K = int(plan["K"])
    if str(cfg.grouping_source) == "plan":
        labels = np.asarray(plan["labels"], np.int64)
        return labels, {
            "grouping_source": "plan (VERBATIM)",
            "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
            "warning": (
                "⛔ the plan's labels index the k_choice seed's encoder channels; "
                "on any other seed this is a size-matched RANDOM partition"
            ),
        }
    if str(cfg.grouping_source) != "per_seed":
        raise ValueError(
            f"grouping_source must be 'per_seed' or 'plan', got {cfg.grouping_source!r}"
        )
    gs = config.gen_store
    Xc = np.asarray(X_store, np.float32)[: int(gs.n_corr)]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    labels, info = grouping_labels(
        corr, K, "correlation", int(seed), gs.grouping_method,
        dead=dead, min_well_frac=float(gs.min_well_frac),
    )
    out = {
        "grouping_source": "per_seed (re-derived at the plan's K)",
        "n_corr_rows": int(Xc.shape[0]),
        "n_dead_channels": int(dead.sum()),
        "grouping_info": info,
        "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
        "plan_encoder_param_hash": plan.get("encoder_param_hash"),
        "this_encoder_param_hash": str(encoder_hash),
    }
    if str(encoder_hash) == str(plan.get("encoder_param_hash")):
        same = bool(np.array_equal(labels, np.asarray(plan["labels"], np.int64)))
        out["reproduces_the_plan_bit_identically"] = same
        if not same:
            raise AssertionError(
                "the per-seed grouping does NOT reproduce store_plan.json's labels "
                "on the seed whose encoder wrote it — ⛔ this is not the same "
                "procedure and the ladder comparison would be void"
            )
        log("  ⭐ grouping re-derivation reproduces the plan BIT-IDENTICALLY on its own seed")
    else:
        out["reproduces_the_plan_bit_identically"] = None
        log(
            "  ⚠ this seed's encoder differs from the plan's "
            f"({str(encoder_hash)[:12]}… vs {str(plan.get('encoder_param_hash'))[:12]}…), "
            "so the grouping is re-derived on its own channels — ⛔ the plan's label "
            "vector would be a size-matched RANDOM partition here"
        )
    return labels, out


