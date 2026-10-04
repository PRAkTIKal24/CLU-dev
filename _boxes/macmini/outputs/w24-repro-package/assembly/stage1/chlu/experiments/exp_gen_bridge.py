"""Experiment: the frozen encoder/decoder **bridge** and the ceiling that gates the phase.

⭐ DIAL DECLARATION — *none: instrument, plus one gating measurement.*  There is
no store, no well, no arm and no CLU number in this file.  Nothing it produces is
evidence for the primitive.

What it builds, once, for every downstream arm to share:

1. a small conv **encoder** fit on the Montero *train split only* with a declared
   pretext (``recon`` = the mandatory primary, R1; ``cls`` = the diagnostic arm),
   then frozen and hashed;
2. the **feature cache** of penultimate maps, written once with a checksum;
3. ⭐ **THE decoder** — one renderer, fit on real ``(feature, image)`` pairs from
   the train split, frozen, hashed, and shared by every arm forever.  ⛔ No arm
   substitutes its own; ``chlu.core.gen_bridge.assert_decoder_hash`` is the check;
4. the **probe** — a per-factor classifier on real images (R1's measurement
   exemption), whose accuracy on *real* held-out images is reported so the
   instrument is auditable before anything is built on it;
5. **the gate**: ``ceiling_A`` / ``ceiling_B`` from decoding real cached features,
   and the verdict ``PROCEED`` iff ``ceiling_B >= 0.8 * ceiling_A`` (R3.1).
   ⛔ A ``STOP`` is a legitimate result; it is not a licence to tune the decoder.

The run is **arm-independent by construction**: every draw is threaded from
``seed`` alone, so the decoder frozen at a seed is bit-identical whichever
``--arm`` the launcher passes.  That is asserted in ``tests/test_gen_bridge.py``.

Runnable:
    chlu exp-gen-bridge [--dataset shapes3d] [--pretext recon] [--quick]
    uv run python -m chlu.experiments.exp_gen_bridge \\
        --dataset shapes3d --pretext recon --stage encoder --arm clu --seed 0 --out DIR

⛔ The CSF3 launcher (``scripts/csf3/job_gpu_gen.sh``) defaults ``STAGE=encoder``
to the module ``chlu.experiments.exp_gen_encoder``; submit the bridge stages with
``PAYLOAD_MODULE=chlu.experiments.exp_gen_bridge`` (the launcher's own override).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from ..config import CHLUConfig
DEFAULT_CONDITION = "recomb2element"
def condition_of(config: CHLUConfig) -> str:
    """The hold-out condition this run is fitted against (``gen_data.condition``)."""
    return str(getattr(config.gen_data, "condition", DEFAULT_CONDITION))


def select_rows(split, cfg, seed: int) -> dict:
    """Deterministic, arm-independent row selection.

    ``train`` fits everything; ``eval_a`` is drawn from the SAME (seen) split but
    is held back from every fitting stage, so ceiling_A is measured on novel
    *images* of seen *combinations* — exactly matched to B (novel images, unseen
    combinations).  Every held-out row is in ``eval_b``, never subsampled.
    """
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(split.train_rows)
    n_train = min(int(cfg.n_train), perm.size)
    n_a = min(int(cfg.n_eval_a), perm.size - n_train)
    return {
        "train": np.sort(perm[:n_train]),
        "eval_a": np.sort(perm[n_train : n_train + n_a]),
        "eval_b": np.asarray(split.test_rows, np.int64),
    }


