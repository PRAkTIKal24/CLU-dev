"""Experiment: the **two competitor arms** — byte-matched kNN, and LACE.

⭐ DIAL DECLARATION — *none: competitor arms / instrument.*
- **Laundering control:** N/A — ⛔ **this module IS the control.** No CLU number is
  produced here and ⛔ no kill-rule verdict is rendered (``gen-cell-run`` renders
  verdicts against a filed prereg).
- **Falsifies:** nothing, in the usual direction.  ⭐ **The falsifier inverts:** the
  kNN arm is the comparator for kill rules (ii) on B **and** (iv) on C, so a *weak*
  kNN makes both pass trivially and lets a dead mechanism through.  A hobbled
  baseline here does not cost us a fair fight, **it costs us the truth, in the
  direction that flatters us.**  If the kNN looks bad, the first hypothesis is that
  we hobbled it.
- **Does NOT falsify:** a rival that **beats** the CLU.  That is the control
  working, and it is reported unsoftened.

⛔ **THE TRAP, checked first and not last.**  ``GenBridgeConfig.dec_recipe``
defaults to ``"baseline"`` — the decoder that **FAILED** the gate (balanced
accuracy 0.7191, 0/3 seeds clear).  The winner is ``v1_capacity_schedule``
(0.9414, 3/3).  A run decoded through the baseline **looks entirely normal in the
logs and is void**, so this module selects the winner explicitly
(``gen_baselines.decoder_source``) and **asserts its parameter hash** against
``gen-decoder-lift``'s ``winner_hashes.json`` before recording any number.

**Metrics, as ruled (R20 / R23) — ⛔ none invented here.**
- **Composition:** ⭐ **balanced accuracy** of the compositional term *within the
  corner* — positives = the arm's renders of the held-out requests, negatives =
  its renders of the **matched-seen** requests (same colour corner, a seen shape).
  AUROC is a mandatory companion and is **never** the gate statistic.  Aggregate
  and per-factor probe accuracy are reported alongside; ⚠ ``shape`` is the only
  compositional factor and is always broken out — four of six factors sit at ~1.0.
- **Refusal:** ⭐ the primary is **AUROC(B unseen-valid vs C impossible)**.
  C-vs-A and C-vs-(A∪B) are reported alongside.  Thresholds come from **valid data
  only**.
- ⭐ **Two ceilings on every table:** the **decoder ceiling** (re-measured in-run
  from real cached features) **and** the **STORE ceiling** (0.6866, R22/R23,
  quoted with provenance).  ⛔ A number quoted against only one of the two is
  incomplete.

⛔ **Not this module's job:** the CLU arm, the addressing head, the store, the
certificate, any verdict.  A/B/C are single-sourced and ⛔ never regenerated.

Runnable::

    chlu exp-gen-baselines [--dataset shapes3d] [--pretext recon] [--quick]
    python -m chlu.experiments.exp_gen_baselines \\
        --dataset shapes3d --pretext recon --stage baselines --arm all --seed 0 --out DIR
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from ..eval.gen_probe import decode_features, probe_accuracy
def score_composition(decoder, probe, feats, labels, names, feat_shape, batch) -> dict:
    """Decode features through THE frozen decoder, read them with THE frozen probe."""
    Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(feat_shape))
    imgs = decode_features(decoder, Z, int(batch))
    out = probe_accuracy(probe, imgs, labels, names, int(batch))
    out["decoded_mean"] = float(imgs.mean())
    return out


def _slice_dense(req, n: int):
    from ..core.gen_request import DenseRequests

    n = int(n)
    return DenseRequests(
        req.cls_targets[:n], req.val_targets[:n], req.extra_factor[:n],
        req.extra_cls[:n], req.extra_val[:n], req.n_factors,
    )


