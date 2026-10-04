"""exp gen bridge."""

from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import numpy as np
from nclu.config import NCLUConfig

DEFAULT_CONDITION = "recomb2element"


def condition_of(config: NCLUConfig) -> str:
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
