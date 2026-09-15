"""⭐⭐ **THE GENERATION HARNESS, END TO END, SCORED ON IMAGES** (R29 pivot item 2).

One path::

    request -> launch read -> codes -> THE frozen decoder -> image
            -> THE external classifier

five arms — ``clu`` · ``store_removed`` · ``knn_proto`` · ``lace`` · ``ebm_compose`` —
three seeds, and **one table**.

⭐ **DIAL DECLARATION** — *composition, end to end, scored on IMAGES.*

- **Laundering control:** ⭐ the **store-removed** arm, on **every row**.  ⛔ It is not a
  footnote; it is the row that says whether any number in the table came from the store.
- **Falsifies:** the table itself.  ⛔ **Composition was RELEASED as a claim in R29 — kill
  rule (ii) already fired.**  Nothing here tries to win it back; this module measures,
  honestly, where the mechanism sits against three rivals on a metric nobody in this phase
  has used yet.
- **Does NOT falsify:** losing to an unbounded-channel arithmetic baseline (that is the
  metric-native-ceiling theorem and R27.1's already-rendered verdict), or absolute levels
  below the decoder ceiling.

⛔ **Three honesty rules this module is built around.**

1. ⛔ **Capability 1 — composing unseen-but-valid combinations — is NEVER novel on its own.**
   LACE owned it in 2021 (R4, R14.6).
2. ⛔ **Store load-bearing is BY CONSTRUCTION** — a method-section fact, ⛔ never a headline
   (R29).
3. ⭐ **An accuracy is never reported without its DISTINCT-OUTPUT COUNT.**  On B the 30-bit
   read renders **9 ± 1** distinct vectors of 72 while ``knn_proto`` renders 72/72; an
   accuracy column alone hides the single most important measured fact about this mechanism
   (R29 Q2, R30, R31).

⛔ **Live traps this module defends against, each with an assertion rather than a comment.**

- :attr:`~chlu.config.GenBridgeConfig.dec_recipe` still defaults to ``"baseline"`` — the
  decoder that **FAILED** the gate.  The winner ``v1_capacity_schedule`` is selected
  explicitly and its **parameter hash is asserted per arm per seed** (R9.2); a baseline run
  looks entirely normal in the logs and is **void**.
- Every arm is read by the **same** external classifier, hash-asserted, loaded from
  ``gen-external-classifier``'s gated checkpoint.  ⛔ Nothing is re-fitted here.
- ``gen_store.soft_read`` takes **FEATURES**; using it on the CLU path would void the
  structural no-bypass guarantee.  ``gen_compose.assert_request_only`` runs before any
  number is recorded.
- The deployed store is **524,288 B** (R26.3), ⛔ not the retired 688,128 B.  ⚠ The shipped
  ``knn_proto`` configuration was *selected* at the retired budget — a **~31 % advantage** —
  which is the **conservative** direction and is **declared in the table**.

Runnable::

    python -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon \\
        --stage cell --seed 0 --out DIR
    python -m chlu.experiments.exp_gen_cell --stage aggregate --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import CHLUConfig, get_default_config, gen_row_split_config  # noqa: E402
from ..data.gen_rowsplit import (  # noqa: E402
    assert_row_split_balanced,
    row_split_balance,
    split_fit_val,
)
from ..core.gen_bridge import load_feature_cache, param_hash  # noqa: E402
from ..core.gen_compose import (  # noqa: E402
    COMPOSITIONAL_FACTOR,
    address_book_bytes,
    assert_request_only,
    compose_codes,
    deployed_byte_ledger,
    read_compose,
    request_posterior,
    store_removed_read,
)
from ..core.gen_dream import (  # noqa: E402
    fit_address_book,
    requests_from_labels,
)
from ..core.gen_request import (  # noqa: E402
    FactorPrototypes,
    dense_request_arrays,
    requests_from_rows,
)
from ..core.gen_store import (  # noqa: E402
    address_codes,
    assert_no_bypass,
    init_store,
    load_store_plan,
    train_store,
    well_channels_from_labels,
)
from ..eval.gen_probe import (  # noqa: E402
    decode_features,
    discrimination_metrics,
    factor_labels,
    factor_probs,
    predict_factors,
    probe_accuracy,
)
from ..eval.rivals import gen_lace as lace  # noqa: E402
from ..experiments.exp_gen_bridge import condition_of  # noqa: E402
from ..experiments.exp_gen_compose import (  # noqa: E402
    _ms,
    _write_json,
    resolve_decoder,
)
from ..experiments.exp_gen_dream import resolve_grouping  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

def _cell_root(config: CHLUConfig) -> Path:
    """``<claude>`` for THIS spoke.

    ⚠ ⛔ Deliberately **not** ``exp_gen_compose._claude_root``: that one reads
    ``gen_compose.claude_root``, so a ``gen_cell.claude_root`` override would be
    silently dead — and a config knob that does nothing is how a test stops testing
    anything (found by ``test_preconditions_FIRE_on_a_failing_gate``, which did not
    fire until this existed).
    """
    r = config.gen_cell.claude_root or os.environ.get("CHLU_CLAUDE_ROOT")
    if r:
        return Path(r).expanduser()
    return Path(__file__).resolve().parents[2] / ".claude"


STAGES = ("cell", "aggregate")
#: ⛔ Kept so the launcher's other ``--stage`` values do not silently no-op here.
NOT_MINE = ("encoder", "decoder", "store", "dream", "certificate", "baselines", "compose")
#: ⭐ ``store_removed`` is second on purpose: the laundering control sits beside the
#: mechanism in every table this module writes, ⛔ never in a footnote.
ARMS = ("clu", "store_removed", "knn_proto", "lace", "ebm_compose")
#: The request sets the gate statistic needs.  ``A`` is an optional companion.
REQUEST_SETS = ("B", "MS")


# ---------------------------------------------------------------------------
# 0. ⛔ THE PRECONDITIONS — mechanical, content-checked, before any other work
# ---------------------------------------------------------------------------


def check_preconditions(config: CHLUConfig, log: Callable = print) -> dict:
    """⛔ Both gate files, **content**-checked.  A failure here stops the run.

    R30's sequencing: ``gen-cell-harness`` runs only when
    ``gen-external-classifier``'s ``classifier_gate.json`` reads ``"gate": "PASS"``
    **and** ``gen-arms-a7-wiring``'s ``refusal_scores.json`` reads
    ``"status": "complete"``.  ⛔ "Start anyway and fix later" is not available.
    """
    cfg = config.gen_cell
    out = {"checked": [], "ok": True}
    croot = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (
        _cell_root(config) / "outputs" / "gen-external-classifier"
    )
    gate_path = croot / str(cfg.classifier_gate_file)
    if bool(cfg.require_classifier_gate):
        if not gate_path.exists():
            raise FileNotFoundError(
                f"⛔ PRECONDITION: no classifier gate at {gate_path}. The external "
                "instrument is ungated, so nothing here is scorable. Report blocked."
            )
        with open(gate_path) as fh:
            gate = json.load(fh)
        verdict = str(gate.get("gate"))
        if verdict != "PASS":
            raise AssertionError(
                f"⛔ PRECONDITION: {gate_path} reads gate={verdict!r}, not 'PASS'. "
                "⛔ Do nothing else and report blocked."
            )
        out["checked"].append({
            "file": str(gate_path), "key": "gate", "value": verdict,
            "per_seed": gate.get("per_seed"), "floor": gate.get("floor"),
            "statistic": gate.get("statistic"),
        })
        log(f"  ⭐ PRECONDITION 1/2: {gate_path.name} gate={verdict} "
            f"(floor {gate.get('floor')}, {gate.get('n_seeds')} seeds)")

    if cfg.refusal_scores_file and bool(cfg.require_refusal_complete):
        rpath = _cell_root(config) / "outputs" / str(cfg.refusal_scores_file)
        if not rpath.exists():
            raise FileNotFoundError(
                f"⛔ PRECONDITION: no refusal scores at {rpath}. The rivals have not "
                "been re-scored under A7, so ⛔ none of their numbers may be quoted "
                "beside ours (R27.4). Report blocked."
            )
        # ⚠ 2.7 MB: read the head only, and assert on the parsed value — a grep for
        # the string would pass on a file that merely mentions it.
        with open(rpath) as fh:
            rs = json.load(fh)
        status = str(rs.get("status"))
        if status != "complete":
            raise AssertionError(
                f"⛔ PRECONDITION: {rpath} reads status={status!r}, not 'complete'."
            )
        out["checked"].append({
            "file": str(rpath), "key": "status", "value": status,
            "seeds_present": rs.get("seeds_present"), "not_run": rs.get("not_run"),
            "legs": list((rs.get("legs") or {}).keys()) if isinstance(rs.get("legs"), dict)
            else rs.get("legs"),
        })
        log(f"  ⭐ PRECONDITION 2/2: {rpath.name} status={status} "
            f"(not_run={rs.get('not_run')})")
    return out


# ---------------------------------------------------------------------------
# 1. ⛔ THE INSTRUMENT — loaded, hash-asserted, and the SAME for every arm
# ---------------------------------------------------------------------------


def load_external_classifier(config: CHLUConfig, seed: int, log: Callable = print):
    """⭐ ``gen-external-classifier``'s gated instrument for this seed.

    ⛔ Never re-fitted here.  The parameter hash is asserted against the seed's own
    manifest, so an arm scored by a different (or re-trained) reader cannot slip in.
    """
    cfg = config.gen_cell
    root = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (
        _cell_root(config) / "outputs" / "gen-external-classifier"
    )
    ckpt = root / cfg.classifier_checkpoint_pattern.format(seed=int(seed))
    man_p = root / cfg.classifier_manifest_pattern.format(seed=int(seed))
    if not ckpt.exists():
        raise FileNotFoundError(
            f"⛔ no external classifier at {ckpt} — this seed's instrument does not "
            "exist and its arms are reported NOT RUN, ⛔ never as zeros"
        )
    with open(man_p) as fh:
        man = json.load(fh)
    model = load_model(ckpt)
    h = param_hash(model)
    expect = str((man.get("classifier") or {}).get("param_hash"))
    if expect and h != expect:
        raise AssertionError(
            f"⛔ external classifier hash {h[:16]}… does not match its own manifest "
            f"{expect[:16]}… — the instrument on disk is not the one that was gated"
        )
    info = {
        "path": str(ckpt), "param_hash": h,
        "param_count": (man.get("classifier") or {}).get("param_count"),
        "steps": (man.get("classifier") or {}).get("steps"),
        "gate_balacc_on_real": ((man.get("gate") or {}).get("external") or {}).get(
            "balanced_accuracy"
        ),
        "note": ("⭐ ONE instrument for every arm, loaded and hash-asserted; ⛔ not "
                 "fitted here and ⛔ not of the decoder's lineage"),
    }
    log(f"  ⭐ external classifier {h[:16]}… ({info['param_count']} params) — "
        "THE instrument, identical across every arm in this seed")
    return model, info


# ---------------------------------------------------------------------------
# 2. Scoring — ⛔ the SAME functions read every arm and both instruments
# ---------------------------------------------------------------------------


def render(decoder, feats: np.ndarray, feat_shape, batch: int) -> np.ndarray:
    """``(N, D) arm output -> (N, C, H, W) images`` through THE frozen decoder."""
    Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(feat_shape))
    return decode_features(decoder, Z, int(batch))


def score_images(model, images, labels, names, batch: int) -> dict:
    out = probe_accuracy(model, images, labels, names, int(batch))
    out["compositional_factor"] = COMPOSITIONAL_FACTOR
    return out


def in_corner(model, img_pos, img_neg, lab_pos, shape_idx: int, batch: int) -> dict:
    """⭐ The gate statistic (R20/R23): DISCRIMINATION within the corner, on IMAGES.

    Positives = the arm's renders of the 72 held-out requests; negatives = its renders
    of the matched-seen requests.  Balanced accuracy is the statistic, AUROC the
    mandatory companion, ⛔ chance is 0.5 and TPR is *detection*, never the gate.
    """
    pp = factor_probs(model, img_pos, int(shape_idx), int(batch))
    pn = factor_probs(model, img_neg, int(shape_idx), int(batch))
    targets = np.unique(np.asarray(lab_pos, np.int64)[:, int(shape_idx)])
    if targets.size != 1:
        raise AssertionError(
            f"the held-out cell realises {targets.size} {COMPOSITIONAL_FACTOR} classes; "
            "the in-corner binary read assumes exactly one (R20)"
        )
    return discrimination_metrics(pp, pn, int(targets[0]))


def distinct_counts(model, feats: np.ndarray, images: np.ndarray, batch: int) -> dict:
    """⭐⭐ The column an accuracy is never reported without.

    Three counts, because they answer three different questions and ⛔ only the third
    is comparable across a discrete-code arm and a continuous sampler:

    * ``feature_vectors`` — distinct arm outputs.  ⚠ For a stochastic sampler this is
      **trivially N** (a continuous draw cannot repeat a float vector) and must not be
      read as diversity.
    * ``images`` — distinct renders.  Same caveat.
    * ⭐ ``factor_tuples`` — distinct factor readings **by the instrument**.  This is
      the comparable one: it counts distinct *content*, and it is the column on which
      the 30-bit read's commitment collapse is visible.
    """
    F = np.asarray(feats, np.float32).reshape((int(feats.shape[0]), -1))
    G = np.asarray(images, np.float32).reshape((int(images.shape[0]), -1))
    preds = predict_factors(model, images, int(batch))
    return {
        "n": int(F.shape[0]),
        "feature_vectors": int(np.unique(F, axis=0).shape[0]),
        "images": int(np.unique(G, axis=0).shape[0]),
        "factor_tuples": int(np.unique(np.asarray(preds, np.int64), axis=0).shape[0]),
        "note": ("⚠ feature_vectors/images are trivially N for a continuous sampler; "
                 "⭐ factor_tuples is the cross-arm comparable diversity count"),
    }


# ---------------------------------------------------------------------------
# 3. The LACE arm — ⛔ its configuration is INHERITED, never re-selected
# ---------------------------------------------------------------------------


def lace_selection(config: CHLUConfig, seed: int, log: Callable = print) -> dict:
    """⭐ Read ``gen-arms-a7-wiring``'s SELECTED LACE configuration from disk.

    ⛔ **Why inherited rather than re-tuned.**  R27.3 is a rule about never weakening a
    competitor *through our own code*.  Re-running the 36-trial selection on a fresh key
    stream could land on a different configuration and move LACE's number for a reason
    that has nothing to do with the instrument this table is about.  The inherited leg is
    the **pre-registered** one — which is the **stronger on composition on both gated
    seeds** (R33 erratum), so quoting it is correct and required, and LACE's published
    composition row (``ode`` 0.6797 ± 0.0207) stands.

    ⚠ **Residue carried, unresolved:** on seeds 0 and 2 that leg ran with the prior-score
    omission unlicensed, and R27.3's second route (the general form **with** the prior
    score) is **NOT RUN** — it needs a new drift term in ``sample_ode``.  ⛔ Not made here.
    """
    cfg = config.gen_cell
    if not cfg.lace_selection_source:
        return {"inherited": False, "reason": "no selection source configured"}
    base = _cell_root(config) / "outputs" / str(cfg.lace_selection_source)
    path = base / cfg.lace_selection_pattern.format(seed=int(seed))
    with open(path) as fh:
        man = json.load(fh)
    arm = (man.get("arms") or {}).get("lace")
    if arm is None:
        raise AssertionError(f"⛔ {path} carries no LACE arm to inherit from")
    sel = dict(arm["tuning"]["selected"])
    gate = arm.get("whitening_exactness_gate") or {}
    out = {
        "inherited": True,
        "source": str(path),
        "selected": {k: sel[k] for k in ("sigma", "form", "shrinkage")},
        "selection_criterion": arm["tuning"].get("selection_criterion"),
        "tuning_trials_inherited": int(arm["tuning"].get("budget", 0)),
        "whitening_gate_fired": bool(gate.get("fired")),
        "whitening_selected_is_exact": bool(gate.get("selected_is_exact")),
        "published_in_corner_balacc": {
            k: v["in_corner_discrimination"]["balanced_accuracy"]
            for k, v in (arm.get("samplers") or {}).items()
        },
        "R27_3": ("⭐ both legs were run by gen-arms-a7-wiring and the STRONGER is quoted "
                  "per axis; on COMPOSITION the pre-registered leg is stronger on both "
                  "gated seeds, so it is the one inherited here. ⛔ The label 'non-exact' "
                  "is not a reason to swap a number — R27.3 is a rule about the NUMBER."),
        "NOT_RUN": ("R27.3's second route — the general form WITH the prior score in the "
                    "ODE drift — needs a sampler change nobody has made. ⛔ NOT RUN."),
    }
    log(f"  ⭐ [lace] inheriting the SELECTED configuration from disk: "
        f"sigma={out['selected']['sigma']} form={out['selected']['form']} "
        f"shrinkage={out['selected']['shrinkage']} "
        f"(gate fired={out['whitening_gate_fired']}, "
        f"{out['tuning_trials_inherited']} trials inherited, ⛔ not re-selected)")
    return out


def run_lace_arm(ctx: dict, config: CHLUConfig, seed: int, sel: dict,
                 log: Callable = print) -> dict:
    """⭐ LACE on the inherited configuration: whiten, fit at the shipped 50 epochs, sample.

    ⛔ Constants are ``chlu.eval.rivals.gen_lace``'s and are not touched (R9.6: shipped
    ``n_epochs=50``, the staircase decay never fires, ``dis_temp`` is not swept, OR is
    non-commutative, hard 10-operand limit).  ⛔ ``DO_NOT_USE_AS_A_BAR``: LACE's published
    0.94–0.98 is StyleGAN at 1024² on their splits.
    """
    cfg = config.gen_cell
    t0 = time.time()
    s = sel["selected"]
    shrink = str(s["shrinkage"])
    # ⛔ ``fit_whitening`` takes the STRINGS "none"/"ledoit_wolf" verbatim and an
    # explicit intensity otherwise — identical to ``exp_gen_baselines._parse_shrinkage``.
    # ⚠ An earlier version mapped "none" to Python ``None`` and LACE was reported
    # NOT RUN on seed 1 with a ``TypeError``; ⛔ do not "improve" this again.
    shrink_v = shrink if shrink in ("none", "ledoit_wolf") else float(shrink)
    wh = lace.fit_whitening(ctx["X_fit"], shrink_v)
    check = lace.whitening_check(wh, ctx["X_fit"])
    log(f"  [lace] whitening shrinkage={shrink}: |diag-1|="
        f"{check['mean_abs_diag_minus_one']:.4g} |offdiag|={check['mean_abs_offdiag']:.4g}")

    key = jax.random.PRNGKey(int(seed) * 10_009 + 20_260_913)
    key, fk = jax.random.split(key)
    model, fit_info, key = lace.train_classifier(
        wh.forward(ctx["X_fit"]), ctx["lab_fit"], ctx["cont_fit"], ctx["cardinalities"], fk,
        n_epochs=int(cfg.lace_n_epochs), batch=int(cfg.lace_batch), lr=float(cfg.lace_lr),
        warmup=int(cfg.lace_warmup_iters), log=None,
    )
    clf_acc = lace.classifier_accuracy(model, wh.forward(ctx["X_val"]), ctx["lab_val"])
    log(f"  [lace] classifier: {fit_info['steps']} steps, loss {fit_info['loss_first']:.4f} "
        f"-> {fit_info['loss_final']:.4f}; its own val per-factor mean {clf_acc['mean']:.4f}")

    out = {"samplers": {}, "whitening": {**wh.meta, "check": check},
           "classifier": {**fit_info, "own_val_accuracy": clf_acc}}
    for sname in list(cfg.lace_samplers):
        spec = lace.SAMPLERS[sname]
        feats = {}
        for rs in REQUEST_SETS:
            key, rk = jax.random.split(key)
            req = ctx[f"req_{rs}"]
            if spec["kind"] == "ode":
                U = lace.sample_ode(model, req, rk, form=s["form"], sigma=float(s["sigma"]),
                                    batch=int(cfg.eval_batch))
            else:
                U = lace.sample_ld(model, req, rk, std=float(spec["std"]),
                                   lr=lace.LACE_SGLD_LR, steps=int(cfg.lace_ld_steps),
                                   form=s["form"], sigma=float(s["sigma"]),
                                   batch=int(cfg.eval_batch))
            feats[rs] = np.asarray(wh.inverse(U), np.float32)
        out["samplers"][sname] = feats
    out["wall_clock_s"] = float(time.time() - t0)
    out["_model"], out["_wh"], out["_key"] = model, wh, key
    return out


def lace_resample(lace_state: dict, ctx: dict, config: CHLUConfig, sel: dict,
                  sampler: str, key) -> dict:
    """⭐ One more independent draw at a FIXED (seed, configuration) — the noise floor."""
    cfg = config.gen_cell
    s = sel["selected"]
    spec = lace.SAMPLERS[sampler]
    model, wh = lace_state["_model"], lace_state["_wh"]
    feats = {}
    for rs in REQUEST_SETS:
        key, rk = jax.random.split(key)
        req = ctx[f"req_{rs}"]
        if spec["kind"] == "ode":
            U = lace.sample_ode(model, req, rk, form=s["form"], sigma=float(s["sigma"]),
                                batch=int(cfg.eval_batch))
        else:
            U = lace.sample_ld(model, req, rk, std=float(spec["std"]), lr=lace.LACE_SGLD_LR,
                               steps=int(cfg.lace_ld_steps), form=s["form"],
                               sigma=float(s["sigma"]), batch=int(cfg.eval_batch))
        feats[rs] = np.asarray(wh.inverse(U), np.float32)
    return feats


# ---------------------------------------------------------------------------
# 4. ⭐ The ENERGY-BASED COMPOSITION arm — Du et al. 2020, ⛔ not LACE
# ---------------------------------------------------------------------------


class FactorExpert(eqx.Module):
    """One **independent** energy expert over a single factor.

    ⭐ The structural contrast with LACE, in three points, so the two arms are never
    read as the same method:

    1. **Independent experts.**  LACE has ONE shared ``DenseEmbedder`` trunk with
       per-factor heads; here each factor gets its own network, which is what makes
       ``E(x | c₁ ∧ c₂) = E(x | c₁) + E(x | c₂)`` a *product of experts* rather than a
       decomposition of one joint model.
    2. **No whitening.**  Sampling happens in the standardised feature space, and the
       Gaussian data prior's score enters the sampler **explicitly** — it is not
       licensed away by an exactness identity (which is LACE's R9.1 dependency).
    3. **Langevin in the data space**, from noise, as in the original.
    """

    layers: list
    head: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, depth: int, n_classes: int, key):
        keys = jax.random.split(key, int(depth) + 1)
        dims = [int(dim)] + [int(hidden)] * int(depth)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(int(depth))
        ]
        self.head = eqx.nn.Linear(dims[-1], int(n_classes), key=keys[-1])

    def __call__(self, u):
        x = jnp.asarray(u, self.layers[0].weight.dtype).reshape(-1)
        for lin in self.layers:
            x = jax.nn.silu(lin(x))
        return self.head(x)


def train_expert(U, y, n_classes: int, cfg, key, log: Callable = print):
    """Cross-entropy on the fit rows — the SAME dense-label supervision LACE gets (R9.3)."""
    key, mk = jax.random.split(key)
    model = FactorExpert(int(U.shape[1]), int(cfg.ebm_hidden), int(cfg.ebm_depth),
                         int(n_classes), mk)
    opt = optax.adam(float(cfg.ebm_lr))
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m, st, ub, yb):
        def loss_fn(mm):
            logits = jax.vmap(mm)(ub)
            return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(yb.shape[0]), yb])

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        upd, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, loss

    n, b = int(U.shape[0]), int(cfg.ebm_batch)
    Uj, yj = jnp.asarray(U, jnp.float32), jnp.asarray(y, jnp.int32)
    first = last = None
    for _ in range(int(cfg.ebm_epochs)):
        key, pk = jax.random.split(key)
        perm = jax.random.permutation(pk, n)
        for i in range(0, n - b + 1, b):
            sl = perm[i : i + b]
            model, state, loss = step(model, state, Uj[sl], yj[sl])
            last = float(loss)
            if first is None:
                first = last
    return model, {"loss_first": first, "loss_final": last,
                   "epochs": int(cfg.ebm_epochs)}, key


def ebm_sample(experts, targets: np.ndarray, key, step_size: float, prior_sigma: float,
               n_steps: int, batch: int = 256) -> np.ndarray:
    """SGLD on ``E(u) = Σⱼ −log pⱼ(vⱼ | u) + ‖u‖²/(2σ²)``, from a noise init.

    ⭐ Langevin-consistent noise ``√(2·lr)`` — ⛔ ⚠ *not* LACE's published ``1e-2``, which
    `gen-lace-spec` F2 derived to be ~200× colder than its own target.  Choosing the
    correct temperature for the rival is the never-weaken-a-competitor direction.
    """
    D = int(experts[0].layers[0].weight.shape[1])
    tg = jnp.asarray(np.asarray(targets, np.int32))
    lr, s2 = float(step_size), float(prior_sigma) ** 2
    std = float(np.sqrt(2.0 * lr))

    def energy(U, T):
        e = 0.5 * jnp.sum(U * U, axis=1) / s2
        for j, ex in enumerate(experts):
            logits = jax.vmap(ex)(U)
            e = e - jax.nn.log_softmax(logits)[jnp.arange(U.shape[0]), T[:, j]]
        return jnp.sum(e)

    grad_fn = eqx.filter_jit(jax.grad(energy))
    n = int(targets.shape[0])
    out = np.empty((n, D), np.float32)
    for i in range(0, n, int(batch)):
        sl = slice(i, min(i + int(batch), n))
        key, sk = jax.random.split(key)
        U = jax.random.normal(sk, (sl.stop - sl.start, D), jnp.float32) * float(prior_sigma)
        Tb = tg[sl]
        for _ in range(int(n_steps)):
            key, nk = jax.random.split(key)
            U = U - lr * grad_fn(U, Tb) + std * jax.random.normal(nk, U.shape, jnp.float32)
        out[sl] = np.asarray(U, np.float32)
    return out


def run_ebm_arm(ctx: dict, config: CHLUConfig, seed: int, log: Callable = print) -> dict:
    """⭐ Train the experts, tune on VALIDATION only, sample the reported sets.

    ⛔ The selection rule is the one both other rivals get — validation aggregate probe
    accuracy, ⛔ C never enters — so no arm is selected by a kinder rule than another.
    """
    cfg = config.gen_cell
    t0 = time.time()
    mu = np.mean(ctx["X_fit"], axis=0)
    sd = np.maximum(np.std(ctx["X_fit"], axis=0), 1e-3)
    std_fit = ((ctx["X_fit"] - mu) / sd).astype(np.float32)
    std_val = ((ctx["X_val"] - mu) / sd).astype(np.float32)

    key = jax.random.PRNGKey(int(seed) * 10_010 + 20_260_913)
    experts, fit_infos = [], []
    for j, c in enumerate(ctx["cardinalities"]):
        m, info, key = train_expert(std_fit, ctx["lab_fit"][:, j], int(c), cfg, key, log)
        experts.append(m)
        fit_infos.append({"factor": ctx["names"][j], **info})
    own = []
    for j, ex in enumerate(experts):
        pred = np.argmax(np.asarray(jax.vmap(ex)(jnp.asarray(std_val))), axis=1)
        own.append(float((pred == ctx["lab_val"][:, j]).mean()))
    log(f"  [ebm] {len(experts)} independent experts trained; their OWN val accuracy "
        f"per factor {[round(a, 4) for a in own]} (mean {float(np.mean(own)):.4f})")

    n_tune = int(min(int(cfg.ebm_tune_n), ctx["lab_val"].shape[0]))
    tune_T = ctx["lab_val"][:n_tune]
    grid = [
        {"step_size": float(lr), "prior_sigma": float(ps), "n_steps": int(ns), "clip": bool(cl)}
        for lr in list(cfg.ebm_step_sizes)
        for ps in list(cfg.ebm_prior_sigmas)
        for ns in list(cfg.ebm_steps_grid)
        for cl in list(cfg.ebm_clip_grid)
    ]
    log(f"  [ebm] tuning budget: {len(grid)} trials "
        f"(⭐ the SAME budget every arm is entitled to: {cfg.tuning_trial_budget})")
    trials = []
    for t_i, tr in enumerate(grid):
        key, sk = jax.random.split(key)
        U = ebm_sample(experts, tune_T, sk, tr["step_size"], tr["prior_sigma"],
                       tr["n_steps"], int(cfg.eval_batch))
        X = U * sd + mu
        if tr["clip"]:
            X = np.maximum(X, 0.0)
        comp = score_images(
            ctx["probe"],
            render(ctx["decoder"], X, ctx["feat_shape"], int(cfg.eval_batch)),
            tune_T, ctx["names"], int(cfg.eval_batch),
        )
        trials.append({**tr, "val_aggregate": comp["aggregate"],
                       "val_compositional": comp["per_factor"][COMPOSITIONAL_FACTOR]})
        if (t_i + 1) % 6 == 0 or t_i == 0:
            log(f"  [ebm] trial {t_i + 1}/{len(grid)} {tr}: val agg={comp['aggregate']:.4f}")
    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    log(f"  ⭐ [ebm] SELECTED (validation aggregate, ⛔ C never enters): {best}")

    feats = {}
    for rs in REQUEST_SETS:
        key, sk = jax.random.split(key)
        U = ebm_sample(experts, ctx[f"labT_{rs}"], sk, best["step_size"],
                       best["prior_sigma"], best["n_steps"], int(cfg.eval_batch))
        X = U * sd + mu
        if best["clip"]:
            X = np.maximum(X, 0.0)
        feats[rs] = np.asarray(X, np.float32)
    n_params = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(experts)))
    return {
        "features": feats,
        "experts": {"n": len(experts), "param_count": n_params,
                    "deployed_bytes": int(n_params * 4),
                    "hidden": int(cfg.ebm_hidden), "depth": int(cfg.ebm_depth),
                    "fit": fit_infos, "own_val_accuracy_per_factor": own,
                    "own_val_accuracy_mean": float(np.mean(own))},
        "tuning": {"budget": len(grid), "selected": best, "trials": trials,
                   "selection_criterion":
                       "validation aggregate probe accuracy (⛔ C never enters)"},
        "standardiser": {"mode": "diagonal, fitted on the fit rows",
                         "clip_nonnegative_selected": bool(best["clip"]),
                         "note": ("⭐ post-ReLU features are non-negative by "
                                  "construction; BOTH clip and no-clip are in the grid "
                                  "and validation chooses — ⛔ never us")},
        "method": ("Du, Li & Mordatch 2020 — compositional visual generation with "
                   "energy-based models: independent per-factor experts, AND = sum of "
                   "energies, SGLD in the data space with the prior score explicit. "
                   "⛔ NOT LACE (one trunk, whitened coordinates, prior score omitted "
                   "under an exactness identity)."),
        "wall_clock_s": float(time.time() - t0),
        "_state": (experts, mu, sd, best),
    }


def ebm_resample(state, ctx: dict, cfg, key) -> dict:
    experts, mu, sd, best = state
    feats = {}
    for rs in REQUEST_SETS:
        key, sk = jax.random.split(key)
        U = ebm_sample(experts, ctx[f"labT_{rs}"], sk, best["step_size"],
                       best["prior_sigma"], best["n_steps"], int(cfg.eval_batch))
        X = U * sd + mu
        if best["clip"]:
            X = np.maximum(X, 0.0)
        feats[rs] = np.asarray(X, np.float32)
    return feats


# ---------------------------------------------------------------------------
# 5. ⭐ THE BYTE LEDGER — a table COLUMN, never a footnote (R26.3)
# ---------------------------------------------------------------------------


def arm_byte_ledger(arm: str, cfg, store_led: dict, proto_led: dict,
                    lace_state: Optional[dict], ebm: Optional[dict]) -> dict:
    """Deployed bytes per arm, with the ~31 % kNN advantage **declared in words**."""
    base = {"arm": arm, "budget_bytes": int(cfg.expect_deployed_store_bytes or 0)}
    if arm in ("clu", "store_removed"):
        return {**base,
                "deployed_bytes": int(store_led["deployed_bytes"]),
                "addressing_head_bytes": int(store_led["addressing_head"]["bytes"]),
                "what": "the store's modes (n_modes·D·4); the addressing head is "
                        "reported BESIDE by store_bytes.json's own convention",
                "coupling_bytes_EXCLUDED": int(store_led.get("coupling_bytes_excluded", 0)),
                "budget_note": ("⭐ 524,288 B — R26.3's corrected budget; the 163,840 B "
                                "inter-well coupling left the deployed store when "
                                "dreaming left the critical path. ⛔ Do NOT inherit "
                                f"{int(cfg.retired_byte_budget)}.")}
    if arm == "knn_proto":
        return {**base,
                "deployed_bytes": int(proto_led["bytes"]),
                "addressing_head_bytes": 0,
                "what": "the per-factor prototype table (R8's declared embedding), "
                        "decoded directly — an UNBOUNDED real-valued channel",
                "budget_note": ("⚠ **DECLARED, NOT SILENT:** the shipped knn_proto "
                                f"CONFIGURATION was selected at the retired "
                                f"{int(cfg.retired_byte_budget)} B budget — a ~31 % byte "
                                "advantage it should not have (R26.3). ⭐ That is the "
                                "CONSERVATIVE direction: an over-resourced competitor "
                                "that still wins is a stronger result for it, and a "
                                "competitor we beat despite it would be a stronger "
                                "result for us. ⭐ Measured byte-INVARIANT by "
                                "gen-arms-a7-wiring (Δ = 0.000000, 3/3 seeds).")}
    if arm == "lace":
        n = (lace_state or {}).get("classifier", {}).get("param_count")
        return {**base,
                "deployed_bytes": int(n * 4) if n else None,
                "what": "the LACE attribute classifier's parameters + the whitening "
                        "transform; ⛔ LACE has no enumerable store to byte-match "
                        "against, so this is reported for scale, ⛔ not as a match",
                "budget_note": "⚠ LACE is NOT byte-matched and is not claimed to be. "
                               "It gets dense factor labels our store never sees (R9.3) "
                               "— the supervision asymmetry is declared, not hidden."}
    if arm == "ebm_compose":
        return {**base,
                "deployed_bytes": int((ebm or {}).get("experts", {}).get("deployed_bytes", 0))
                or None,
                "what": "the independent per-factor energy experts' parameters",
                "budget_note": "⚠ not byte-matched; same declared supervision asymmetry "
                               "as LACE."}
    return base


# ---------------------------------------------------------------------------
# 6. One seed — ⭐ every (arm, seed) is written the moment it completes
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick number is NEVER reported."""
    g = config.gen_cell
    g.seeds = list(g.quick_seeds)
    g.arms = list(g.quick_arms)
    g.n_fit = int(g.quick_n_fit)
    g.n_val = int(g.quick_n_val)
    g.lace_n_epochs = int(g.quick_lace_n_epochs)
    g.lace_ld_steps = int(g.quick_lace_ld_steps)
    g.ebm_epochs = int(g.quick_ebm_epochs)
    g.ebm_step_sizes = list(g.quick_ebm_step_sizes)
    g.ebm_prior_sigmas = list(g.quick_ebm_prior_sigmas)
    g.ebm_steps_grid = list(g.quick_ebm_steps_grid)
    g.ebm_clip_grid = list(g.quick_ebm_clip_grid)
    g.ebm_tune_n = int(g.quick_ebm_tune_n)
    g.sampler_noise_repeats = int(g.quick_sampler_noise_repeats)
    g.verify_cache_checksum = bool(g.quick_verify_cache_checksum)
    return config


def store_plan_path(config: CHLUConfig, dataset: str, pretext: str) -> Path:
    p = config.gen_cell.store_plan
    if p:
        return Path(p).expanduser()
    return (_cell_root(config) / "outputs" / "gen-store-grouping" / "runs"
            / f"{dataset}_{pretext}" / "store_plan.json")


def resolve_store(config: CHLUConfig, dataset: str, pretext: str, seed: int, X_store,
                  wc, plan, feat_shape, log: Callable = print):
    """⭐ Load ``gen-read-compose``'s fitted store so the ONLY thing that changed
    between its table and this one is the **instrument**.

    ⛔ A silently re-trained store would make the two tables incomparable and would be
    exactly the failure this spoke is told to avoid: *an arm's number moves and you
    cannot name what moved it*.
    """
    cfg = config.gen_cell
    root = Path(cfg.store_checkpoint_root).expanduser() if cfg.store_checkpoint_root else (
        _cell_root(config) / "outputs" / "gen-read-compose"
    )
    path = root / cfg.store_checkpoint_pattern.format(
        dataset=dataset, pretext=pretext, seed=int(seed)
    )
    if path.exists():
        store = load_model(path)
        prov = {"source": "gen-read-compose checkpoint", "path": str(path),
                "trained_here": False,
                "note": "⭐ the CLU arm's probe-space number must reproduce "
                        "gen-read-compose's row bit-identically"}
        log(f"  ⭐ store loaded from {path} (⛔ not re-trained)")
    elif bool(cfg.allow_store_training):
        key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_911)
        key, ik, tk = jax.random.split(key, 3)
        store = init_store(X_store, wc, int(plan["n_modes"]), feat_shape, ik, seed=int(seed))
        log(f"  ⚠ no checkpoint at {path} — training the modes "
            f"({config.gen_store.train_steps} steps); ⛔ this store is NOT the one "
            "gen-read-compose measured and the regression check will say so")
        store, hist, key = train_store(store, X_store, config.gen_store, tk, log=log)
        prov = {"source": "trained here", "path": str(path), "trained_here": True,
                "loss_final": float(hist[-1]) if hist else None}
    else:
        raise FileNotFoundError(f"⛔ no store checkpoint at {path} and training disabled")
    return store, prov


def run_seed(dataset: str, pretext: str, seed: int, out_dir: Path, config: CHLUConfig,
             arms: Optional[list] = None, log: Callable = print) -> dict:
    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    cfg = config.gen_cell
    sdir = Path(out_dir) / f"seed_{int(seed)}"
    sdir.mkdir(parents=True, exist_ok=True)
    man_path = sdir / "cell_manifest.json"
    manifest = {}
    if man_path.exists():
        with open(man_path) as fh:
            manifest = json.load(fh)
    condition = condition_of(config)
    bdir = bridge_dir(config, dataset, pretext, seed)
    tlog(f"seed {seed} · condition={condition} · frozen bridge {bdir}")

    # ---- ⛔ the two guards, FIRST: the decoder and the instrument -------------
    # ⛔ ``resolve_decoder`` resolves its own default winner root through
    # ``gen_compose.claude_root``; forward ours so the two can never disagree.
    config.gen_compose.claude_root = str(_cell_root(config))
    config.gen_compose.decoder_source = str(cfg.decoder_source)
    config.gen_compose.winner_root = cfg.winner_root
    config.gen_compose.winner_decoder_pattern = str(cfg.winner_decoder_pattern)
    config.gen_compose.expected_decoder_recipe = str(cfg.expected_decoder_recipe)
    decoder, dec_info, bman = resolve_decoder(config, bdir, seed, log=tlog)
    external, clf_info = load_external_classifier(config, seed, log=tlog)
    probe = load_model(bdir / "probe.pkl")
    if param_hash(probe) != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match the bridge manifest hash")
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")

    # ---- rows: B, its matched-seen negatives, and the rivals' fit/val --------
    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    shape_idx = names.index(COMPOSITIONAL_FACTOR)
    rows = select_store_rows(split, config, seed)
    pool = np.asarray(rows["store_train"], np.int64)
    # ⛔ THE SAME stratified cut ``exp_gen_baselines`` makes, from the same block, on
    # the same (seed, n_fit, n_val) — so the rivals' fit rows here are bit-identical to
    # the rows they published from.  ⛔ Never the head/tail of a sorted array again.
    rs = gen_row_split_config(config)
    fit_rows, val_rows, split_info = split_fit_val(
        pool, split.grid.factor_indices_of(pool), int(cfg.n_fit), int(cfg.n_val),
        int(seed), mode=str(rs.mode), tiebreak_jitter=float(rs.tiebreak_jitter),
        seed_mult=int(rs.seed_mult), seed_offset=int(rs.seed_offset), log=tlog,
    )
    n_fit, n_val = int(fit_rows.size), int(val_rows.size)
    row_balance = row_split_balance(
        split.grid.factor_indices_of(fit_rows), split.grid.factor_indices_of(val_rows),
        list(split.grid.names), list(split.grid.cardinalities),
    )
    if bool(rs.assert_balanced) and str(rs.mode) != "head_tail":
        row_balance = assert_row_split_balanced(
            row_balance, require_all_values=bool(rs.require_all_values),
            max_count_ratio=float(rs.max_count_ratio),
        )
    row_split = {**split_info, "balance": row_balance}
    tlog("  ⭐ row split: mode=" + str(row_split["mode"]) + " — val realises "
         + " · ".join(f"{k} {v['val_n_values_realised']}/{v['cardinality']}"
                      for k, v in row_balance["factors"].items())
         + f" · max per-value count ratio {row_balance['max_val_count_ratio']:.3f}")
    row_sets = {
        "B": rows["eval_b"],
        "MS": matched_seen_rows(dataset, split, values, rows["store_train"]),
        "fit": fit_rows,
        "val": val_rows,
    }
    if int(cfg.n_eval_a) > 0:
        row_sets["A"] = rows["eval_a"][: int(cfg.n_eval_a)]
    tlog(f"rows: store_train={pool.size} B={row_sets['B'].size} MS={row_sets['MS'].size} "
         f"fit={n_fit} val={n_val}")

    # ---- the frozen cache ---------------------------------------------------
    crows, feats_cache, cman = load_feature_cache(
        bdir, verify=bool(cfg.verify_cache_checksum)
    )
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    config.gen_store.encode_chunk = int(cfg.encode_chunk)
    reenc = _assert_cache_reencodes(dataset, config, encoder, crows, feats_cache, cman,
                                    log=tlog)
    tlog(f"cache {cman['n_rows']} rows (scope={cman['scope']}), D={cman['D']}; "
         f"re-encode check bit-identical on {reenc.get('n_checked')} rows")

    def gather(r):
        return gather_features(dataset, config, encoder, crows, feats_cache, cman, r,
                               log=tlog)

    X_store = gather(pool)
    X = {k: gather(v) for k, v in row_sets.items()}
    lab = {k: factor_labels(grid, v) for k, v in row_sets.items()}
    lab_store = factor_labels(grid, pool)

    # ---- the store: pinned plan, per-seed grouping, ⭐ loaded not re-trained --
    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    K, D = int(plan["K"]), int(np.prod(feat_shape))
    config.gen_dream.grouping_source = str(cfg.grouping_source)
    labels_vec, ginfo = resolve_grouping(plan, X_store, config, seed,
                                         bman["encoder_param_hash"], log=tlog)
    wc = well_channels_from_labels(labels_vec, K)
    store, store_prov = resolve_store(config, dataset, pretext, seed, X_store, wc, plan,
                                      feat_shape, log=tlog)

    codes_train = address_codes(store, np.asarray(X_store, np.float32).reshape(
        (X_store.shape[0], -1)))
    book = fit_address_book(codes_train, lab_store, cards, float(cfg.address_smoothing))
    store_led = deployed_byte_ledger(store, book)
    if (cfg.expect_deployed_store_bytes is not None
            and int(store_led["deployed_bytes"]) != int(cfg.expect_deployed_store_bytes)):
        raise AssertionError(
            f"deployed store is {store_led['deployed_bytes']} B, expected "
            f"{cfg.expect_deployed_store_bytes} B (R26.3)"
        )
    tlog(f"⭐ byte ledger: store {store_led['deployed_bytes']} B + addressing head "
         f"{store_led['addressing_head']['bytes']} B (⛔ coupling "
         f"{store_led.get('coupling_bytes_excluded')} B EXCLUDED; ⛔ not "
         f"{int(cfg.retired_byte_budget)})")

    # ---- ⛔ the structural guarantees, BEFORE any number ---------------------
    bypass = assert_no_bypass(store, X["B"][:64], rng_seed=int(seed))
    req_arr = {k: requests_from_labels(lab[k], "full", 0, int(seed)) for k in REQUEST_SETS}
    post = {k: request_posterior(book, req_arr[k]) for k in REQUEST_SETS}
    codes = {k: compose_codes(post[k]) for k in REQUEST_SETS}
    req_only = assert_request_only(store, book, req_arr["B"], decoy_features=X["B"])
    tlog(f"⭐ no-bypass: pure codebook gather, {bypass['bits_per_read']:.0f} bits/read "
         "(R20's declared design constant) · ⭐ request-only path asserted on all four "
         "legs (⛔ no feature argument on ANY CLU call; gen_store.soft_read NOT imported)")

    # ---- R8's declared request embedding, fitted on the FIT rows only --------
    protos = FactorPrototypes.fit(grid, X["fit"], row_sets["fit"])
    req_dicts = {k: requests_from_rows(grid, row_sets[k]) for k in REQUEST_SETS}
    proto_led = protos.byte_ledger()

    # ---- the rivals' shared context ----------------------------------------
    cont_cols = []
    for k in range(grid.n_factors):
        vals_k = np.asarray(grid.values[k], np.float64)
        span = max(float(vals_k[-1] - vals_k[0]), 1e-12)
        cont_cols.append((vals_k[lab["fit"][:, k]] - vals_k[0]) / span)
    ctx = {
        "names": names, "cardinalities": cards, "feat_shape": feat_shape,
        "decoder": decoder, "probe": probe, "external": external,
        "X_fit": X["fit"], "X_val": X["val"],
        "lab_fit": lab["fit"], "lab_val": lab["val"],
        "cont_fit": np.stack(cont_cols, axis=1).astype(np.float32),
        "req_B": dense_request_arrays(grid, req_dicts["B"]),
        "req_MS": dense_request_arrays(grid, req_dicts["MS"]),
        "labT_B": lab["B"], "labT_MS": lab["MS"],
    }

    # ---- ⭐ THE DECODER CEILING (real features), the row every arm is read against
    ceil_imgs = {k: render(decoder, X[k], feat_shape, cfg.eval_batch)
                 for k in REQUEST_SETS}
    ceiling = {
        "external": {
            "in_corner": in_corner(external, ceil_imgs["B"], ceil_imgs["MS"], lab["B"],
                                   shape_idx, cfg.eval_batch),
            "composition_B": score_images(external, ceil_imgs["B"], lab["B"], names,
                                          cfg.eval_batch),
        },
        "probe": {
            "in_corner": in_corner(probe, ceil_imgs["B"], ceil_imgs["MS"], lab["B"],
                                   shape_idx, cfg.eval_batch),
            "composition_B": score_images(probe, ceil_imgs["B"], lab["B"], names,
                                          cfg.eval_batch),
        },
        "note": ("⭐ ONE render array read by BOTH instruments, so a gap between them "
                 "can never be a difference in what they looked at"),
    }
    del ceil_imgs
    tlog(f"⭐ DECODER ceiling on IMAGES: external in-corner balacc="
         f"{ceiling['external']['in_corner']['balanced_accuracy']:.4f} · probe="
         f"{ceiling['probe']['in_corner']['balanced_accuracy']:.4f}")

    manifest.update({
        "schema": "chlu-gen-cell/1",
        "dataset": dataset, "pretext": pretext, "condition": condition, "seed": int(seed),
        "path": ("request → launch read → codes → THE frozen decoder → image → "
                 "THE external classifier"),
        "bridge_dir": str(bdir),
        "decoder": dec_info,
        "external_classifier": clf_info,
        "preconditions": manifest.get("preconditions"),
        "provenance": {
            "encoder_param_hash": bman["encoder_param_hash"],
            "probe_param_hash": bman["probe_param_hash"],
            "feature_cache_sha256": cman["features_sha256"],
            "feature_cache_scope": cman["scope"],
            "cache_reencode_check": reenc,
            "store": store_prov,
            "store_plan_path": str(store_plan_path(config, dataset, pretext)),
        },
        "K": K, "D": D, "n_modes": int(plan["n_modes"]), "grouping": ginfo,
        "byte_ledger_store": store_led,
        "byte_ledger_prototypes": proto_led,
        "no_bypass": bypass,
        "request_only_path": req_only,
        "address_book_bytes": address_book_bytes(book),
        "rows": {k: int(v.size) for k, v in row_sets.items()},
        "row_split": row_split,
        "request_addressing": {
            "n_distinct_code_configs": {k: codes[k]["n_distinct_code_configs"]
                                        for k in REQUEST_SETS},
            "mean_confidence": {k: codes[k]["mean_confidence"] for k in REQUEST_SETS},
        },
        "ceiling": ceiling,
        "tuning_trial_budget": int(cfg.tuning_trial_budget),
        "arms": manifest.get("arms", {}),
        "NOT_RUN": manifest.get("NOT_RUN", {}),
        "NO_KILL_RULE_VERDICT": ("⛔ arm numbers only. Kill rule (ii) fired in R27/R29 "
                                 "and is ⛔ NOT reopened by a better metric."),
    })
    _write_json(manifest, man_path)

    # ---- the arms -----------------------------------------------------------
    want = [a for a in (arms or cfg.arms)]
    lace_state, ebm_state, sel = None, None, None

    def _record(arm: str, feats: dict, extra: dict, tuning_trials: int) -> None:
        imgs = {k: render(decoder, feats[k], feat_shape, cfg.eval_batch)
                for k in REQUEST_SETS}
        row = {
            "arm": arm,
            "uses_store": bool(arm in ("clu", "store_removed")),
            # ⛔ asserted PER ARM PER SEED, not once per run (R9.2)
            "decoder_param_hash": dec_info["decoder_param_hash"],
            "decoder_recipe": dec_info["decoder_recipe"],
            "external_classifier_param_hash": clf_info["param_hash"],
            "external": {
                "in_corner": in_corner(external, imgs["B"], imgs["MS"], lab["B"],
                                       shape_idx, cfg.eval_batch),
                "composition_B": score_images(external, imgs["B"], lab["B"], names,
                                              cfg.eval_batch),
            },
            "probe_companion": {
                "in_corner": in_corner(probe, imgs["B"], imgs["MS"], lab["B"], shape_idx,
                                       cfg.eval_batch),
                "composition_B": score_images(probe, imgs["B"], lab["B"], names,
                                              cfg.eval_batch),
            },
            "distinct": {k: distinct_counts(external, feats[k], imgs[k], cfg.eval_batch)
                         for k in REQUEST_SETS},
            "byte_ledger": arm_byte_ledger(arm, cfg, store_led, proto_led, lace_state,
                                           ebm_state),
            "tuning": {"budget_available": int(cfg.tuning_trial_budget),
                       "trials_realised": int(tuning_trials)},
            **extra,
        }
        if dec_info["decoder_param_hash"] != manifest["decoder"]["decoder_param_hash"]:
            raise AssertionError("⛔ the decoder moved between arms")
        if clf_info["param_hash"] != manifest["external_classifier"]["param_hash"]:
            raise AssertionError("⛔ the external classifier moved between arms")
        manifest["arms"][arm] = row
        # ⛔ a stale NOT_RUN entry from an earlier attempt must not outlive the result
        manifest.get("NOT_RUN", {}).pop(arm, None)
        manifest.get("NOT_RUN", {}).pop(arm.split("/")[0], None)
        _write_json(manifest, man_path)   # ⭐ written the moment the arm completes
        e = row["external"]["in_corner"]
        d = row["distinct"]["B"]
        tlog(f"  ⭐ [{arm}] IMAGE-space in-corner {COMPOSITIONAL_FACTOR} balacc="
             f"{e['balanced_accuracy']:.4f} (AUROC {e['auroc']:.4f}, TPR {e['tpr']:.3f} / "
             f"TNR {e['tnr']:.3f}) · distinct on B: {d['feature_vectors']} vectors / "
             f"{d['factor_tuples']} factor-tuples of {d['n']} · probe companion "
             f"{row['probe_companion']['in_corner']['balanced_accuracy']:.4f}")

    for arm in want:
        if arm in manifest["arms"]:
            tlog(f"  [{arm}] already on disk for this seed — skipped")
            continue
        try:
            if arm == "clu":
                feats = {k: np.asarray(read_compose(store, codes[k]["code"]), np.float32)
                         for k in REQUEST_SETS}
                _record(arm, feats, {
                    "channel_bits": float(bypass["bits_per_read"]),
                    "mechanism": ("R26.3 — the k-particle launch READ over the requested "
                                  "modes, then decode. ⛔ No landscape, no clamp."),
                }, 0)
            elif arm == "store_removed":
                feats = {k: np.asarray(store_removed_read(
                    store, codes[k]["code"], str(cfg.store_removed_delete_mode), int(seed)
                ), np.float32) for k in REQUEST_SETS}
                _record(arm, feats, {
                    "role": ("⭐ THE LAUNDERING CONTROL, on every row: the identical read "
                             "path with the store's content ablated. ⛔ Not a footnote."),
                }, 0)
            elif arm == "knn_proto":
                feats = {k: np.asarray(protos.embed_many(req_dicts[k]), np.float32)
                         for k in REQUEST_SETS}
                _record(arm, feats, {
                    "channel_bits": None,
                    "mechanism": ("R8's declared per-factor prototype embedding, decoded "
                                  "DIRECTLY — the whole D-float vector reaches the "
                                  "decoder (an UNBOUNDED channel, ⛔ never compared in "
                                  "the same column as 30 bits)."),
                }, 0)
            elif arm == "lace":
                sel = lace_selection(config, seed, log=tlog)
                lace_state = run_lace_arm(ctx, config, seed, sel, log=tlog)
                for sname, feats in lace_state["samplers"].items():
                    _record(f"lace/{sname}" if sname != cfg.lace_primary_sampler else "lace",
                            feats, {
                        "sampler": sname,
                        "selection": sel,
                        "whitening": lace_state["whitening"],
                        "classifier": lace_state["classifier"],
                        "primary": bool(sname == cfg.lace_primary_sampler),
                        "DO_NOT_USE_AS_A_BAR": {
                            "published_unseen_combination": [0.94, 0.98],
                            "note": ("StyleGAN at 1024² on their splits and labels (R9.8). "
                                     "⛔ Only this re-implementation is comparable."),
                        },
                    }, int(sel.get("tuning_trials_inherited", 0)))
            elif arm == "ebm_compose":
                ebm_state = run_ebm_arm(ctx, config, seed, log=tlog)
                _record(arm, ebm_state["features"], {
                    "experts": ebm_state["experts"],
                    "tuning_detail": ebm_state["tuning"],
                    "standardiser": ebm_state["standardiser"],
                    "method": ebm_state["method"],
                }, int(ebm_state["tuning"]["budget"]))
            else:
                raise ValueError(f"unknown arm {arm!r} (have {ARMS})")
        except Exception as exc:  # ⛔ NOT RUN, never a zero and never a silent gap
            manifest["NOT_RUN"][arm] = {"reason": f"{type(exc).__name__}: {exc}"}
            _write_json(manifest, man_path)
            tlog(f"  ⛔ [{arm}] NOT RUN — {type(exc).__name__}: {exc}")

    # ---- ⭐ the sampler-noise floor: how much of the spread is the sampler? --
    noise = {}
    for arm in list(cfg.sampler_noise_arms):
        if int(seed) != int(cfg.sampler_noise_seed):
            continue
        vals = []
        for r in range(int(cfg.sampler_noise_repeats)):
            k = jax.random.PRNGKey(20_260_913 + 991 * (r + 1) + int(seed))
            try:
                if arm == "lace" and lace_state is not None and sel is not None:
                    feats = lace_resample(lace_state, ctx, config, sel,
                                          str(cfg.lace_primary_sampler), k)
                elif arm == "ebm_compose" and ebm_state is not None:
                    feats = ebm_resample(ebm_state["_state"], ctx, cfg, k)
                else:
                    continue
            except Exception as exc:
                noise[arm] = {"NOT_RUN": f"{type(exc).__name__}: {exc}"}
                break
            imgs = {q: render(decoder, feats[q], feat_shape, cfg.eval_batch)
                    for q in REQUEST_SETS}
            vals.append(in_corner(external, imgs["B"], imgs["MS"], lab["B"], shape_idx,
                                  cfg.eval_batch)["balanced_accuracy"])
        if vals:
            noise[arm] = {
                "repeats": len(vals), "values": [float(v) for v in vals],
                "mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                "what": ("⭐ independent SAMPLER keys at a FIXED (seed, configuration). "
                         "⛔ Any arm difference smaller than this sd is sampler noise, "
                         "not an arm difference — 72 items cannot carry a confident "
                         "claim without this number."),
            }
            tlog(f"  ⭐ sampler-noise floor [{arm}]: sd={noise[arm]['sd']:.4f} over "
                 f"{len(vals)} independent keys (values "
                 f"{[round(v, 4) for v in vals]})")
    manifest["sampler_noise_floor"] = noise

    # ---- ⭐ the rival regression check: did anything get WEAKER? ------------
    manifest["rival_regression_check"] = regression_check(manifest, cfg, int(seed))
    for k, v in manifest["rival_regression_check"].items():
        if isinstance(v, dict) and v.get("regressed"):
            tlog(f"  ⛔ REGRESSION on {k}: measured {v['measured']:.4f} vs published "
                 f"{v['published']:.4f} (Δ {v['delta']:+.4f}) — reported, ⛔ not absorbed")
    manifest["wall_clock_s"] = float(time.time() - t0)
    _write_json(manifest, man_path)
    tlog(f"seed {seed} done in {manifest['wall_clock_s']:.0f}s")
    return manifest


def regression_check(manifest: dict, cfg, seed: int) -> dict:
    """⛔ *"A rival got weaker because of a change you made"* — checked, not assumed.

    Each arm's **probe-companion** number is compared with the number that arm published
    on the same statistic.  ⛔ The comparison is probe-vs-probe: comparing an
    external-classifier number with a published probe number would confound the
    instrument with the arm, which is the one thing this table exists not to do.
    """
    refs = {
        "clu": list(cfg.clu_probe_reference_per_seed),
        "knn_proto": list(cfg.knn_proto_probe_reference_per_seed),
        "lace": list(cfg.lace_ode_probe_reference_per_seed),
    }
    out = {"rule": ("probe-companion vs the arm's own published probe number, same "
                    "statistic, same seed; ⛔ never external-vs-probe"),
           "tolerance": float(cfg.rival_regression_tolerance)}
    for arm, ref in refs.items():
        row = manifest.get("arms", {}).get(arm)
        if row is None or seed >= len(ref):
            out[arm] = {"NOT_RUN": True}
            continue
        m = float(row["probe_companion"]["in_corner"]["balanced_accuracy"])
        p = float(ref[seed])
        out[arm] = {
            "measured": m, "published": p, "delta": m - p,
            "bit_identical": bool(m == p),
            "regressed": bool(m < p - float(cfg.rival_regression_tolerance)),
        }
    return out


# ---------------------------------------------------------------------------
# 7. Aggregate — ⭐ ≥3 seeds, spread on every number, NOT RUN never a zero
# ---------------------------------------------------------------------------


def aggregate_runs(run_dirs, log: Callable = print) -> dict:
    mans = []
    for d in run_dirs:
        p = Path(d) / "cell_manifest.json"
        if p.exists():
            with open(p) as fh:
                mans.append(json.load(fh))
    if not mans:
        return {"seeds_present": [], "note": "⛔ nothing on disk — every row is NOT RUN"}
    seeds = [int(m["seed"]) for m in mans]
    arms = sorted({a for m in mans for a in m.get("arms", {})})
    table = {}
    for arm in arms:
        rows = [m["arms"][arm] for m in mans if arm in m.get("arms", {})]
        present = [int(m["seed"]) for m in mans if arm in m.get("arms", {})]
        table[arm] = {
            "seeds": present,
            "n_seeds": len(present),
            "external_in_corner_balacc": _ms(
                [r["external"]["in_corner"]["balanced_accuracy"] for r in rows]),
            "external_in_corner_auroc": _ms(
                [r["external"]["in_corner"]["auroc"] for r in rows]),
            "external_tpr": _ms([r["external"]["in_corner"]["tpr"] for r in rows]),
            "external_tnr": _ms([r["external"]["in_corner"]["tnr"] for r in rows]),
            "external_B_aggregate": _ms(
                [r["external"]["composition_B"]["aggregate"] for r in rows]),
            "external_B_shape": _ms(
                [r["external"]["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR]
                 for r in rows]),
            "probe_in_corner_balacc": _ms(
                [r["probe_companion"]["in_corner"]["balanced_accuracy"] for r in rows]),
            "distinct_feature_vectors_B": _ms(
                [r["distinct"]["B"]["feature_vectors"] for r in rows]),
            "distinct_factor_tuples_B": _ms(
                [r["distinct"]["B"]["factor_tuples"] for r in rows]),
            "distinct_factor_tuples_MS": _ms(
                [r["distinct"]["MS"]["factor_tuples"] for r in rows]),
            "n_B": int(rows[0]["distinct"]["B"]["n"]),
            "byte_ledger": rows[0]["byte_ledger"],
            "tuning": rows[0]["tuning"],
            "uses_store": bool(rows[0]["uses_store"]),
            "decoder_param_hashes": sorted({r["decoder_param_hash"] for r in rows}),
            "external_classifier_param_hashes": sorted(
                {r["external_classifier_param_hash"] for r in rows}),
        }
        missing = sorted(set(seeds) - set(present))
        if missing:
            table[arm]["NOT_RUN_seeds"] = missing
    out = {
        "schema": "chlu-gen-cell-aggregate/1",
        "seeds_present": sorted(seeds),
        "arms": table,
        "ceiling": {
            "external_in_corner_balacc": _ms(
                [m["ceiling"]["external"]["in_corner"]["balanced_accuracy"] for m in mans]),
            "probe_in_corner_balacc": _ms(
                [m["ceiling"]["probe"]["in_corner"]["balanced_accuracy"] for m in mans]),
        },
        "sampler_noise_floor": {m["seed"]: m.get("sampler_noise_floor") for m in mans},
        "rival_regression_check": {m["seed"]: m.get("rival_regression_check")
                                   for m in mans},
        "NOT_RUN": {m["seed"]: m.get("NOT_RUN", {}) for m in mans},
        "instrument_invariance": {},
        "honesty": [
            "⛔ Composing unseen-but-valid combinations is NOT novel on its own — LACE "
            "owned it in 2021 (R4, R14.6).",
            "⛔ Store load-bearing is BY CONSTRUCTION: a method-section fact, ⛔ never a "
            "headline (R29).",
            "⛔ Kill rule (ii) fired in R27/R29 and is NOT reopened by a better metric.",
            "⭐ Every accuracy above ships with its distinct-output count; an accuracy "
            "alone hides the measured commitment collapse.",
        ],
    }
    for arm, row in table.items():
        e, p = row["external_in_corner_balacc"], row["probe_in_corner_balacc"]
        out["instrument_invariance"][arm] = {
            "external_minus_probe": float(e["mean"] - p["mean"]),
            "external": e["mean"], "probe": p["mean"],
        }
    # ⭐ the ordering check: does swapping the instrument re-order the arms?
    def _order(key):
        return [a for a, _ in sorted(
            ((a, r[key]["mean"]) for a, r in table.items() if np.isfinite(r[key]["mean"])),
            key=lambda kv: -kv[1])]

    oe, op = _order("external_in_corner_balacc"), _order("probe_in_corner_balacc")
    out["instrument_invariance"]["_ranking"] = {
        "by_external": oe, "by_probe": op, "identical": bool(oe == op),
        "what": ("⭐ if the two orderings agree, the table's verdict is a property of "
                 "the ARMS, ⛔ not of the instrument"),
    }
    log(f"⭐ aggregate: {len(seeds)} seeds, {len(arms)} arms; instrument ordering "
        f"{'IDENTICAL' if oe == op else '⛔ DIFFERS'}")
    return out


# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------


def _plot(agg: dict, path: Path) -> None:
    arms = [a for a in ARMS if a in agg["arms"]] + [
        a for a in sorted(agg["arms"]) if a not in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(arms))
    ext = [agg["arms"][a]["external_in_corner_balacc"]["mean"] for a in arms]
    exs = [agg["arms"][a]["external_in_corner_balacc"]["std"] for a in arms]
    prb = [agg["arms"][a]["probe_in_corner_balacc"]["mean"] for a in arms]
    axes[0].bar(x - 0.2, ext, 0.4, yerr=exs, label="external classifier (IMAGES)",
                color="#2b6cb0", capsize=3)
    axes[0].bar(x + 0.2, prb, 0.4, label="probe (companion)", color="#a0aec0")
    axes[0].axhline(0.5, ls="--", c="k", lw=1, label="chance")
    axes[0].axhline(agg["ceiling"]["external_in_corner_balacc"]["mean"], ls=":", c="g",
                    lw=1.2, label="decoder ceiling (external)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(arms, rotation=20, ha="right")
    axes[0].set_ylabel("in-corner `shape` balanced accuracy")
    axes[0].set_title("the headline: composition, scored on IMAGES")
    axes[0].legend(fontsize=7)
    n_b = max(agg["arms"][a]["n_B"] for a in arms)
    dt = [agg["arms"][a]["distinct_factor_tuples_B"]["mean"] for a in arms]
    dts = [agg["arms"][a]["distinct_factor_tuples_B"]["std"] for a in arms]
    axes[1].bar(x, dt, 0.55, yerr=dts, color="#c05621", capsize=3)
    axes[1].axhline(n_b, ls="--", c="k", lw=1, label=f"all {n_b} requests distinct")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(arms, rotation=20, ha="right")
    axes[1].set_ylabel("distinct factor-tuples read off B")
    axes[1].set_title("the column an accuracy must never appear without")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Entry points
# ---------------------------------------------------------------------------


def run_experiment_gen_cell(config: Optional[CHLUConfig] = None, quick: bool = False,
                            out_dir: Optional[str] = None, stage: str = "cell",
                            seeds: Optional[list] = None, arms: Optional[list] = None,
                            log: Callable = print) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    cfg = config.gen_cell
    dataset, pretext = str(cfg.dataset), str(cfg.pretext)
    out = Path(out_dir) if out_dir else (
        _cell_root(config) / "outputs" / "gen-cell-harness" / "runs"
        / f"{dataset}_{pretext}")
    out.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in (seeds if seeds is not None else cfg.seeds)]

    if stage in ("cell", "all"):
        pre = check_preconditions(config, log=log)
        for s in seed_list:
            man = run_seed(dataset, pretext, s, out, config, arms=arms, log=log)
            man["preconditions"] = pre
            _write_json(man, out / f"seed_{s}" / "cell_manifest.json")
    if stage in ("aggregate", "all"):
        # ⛔ The aggregate is over EVERY seed on disk, not only the seeds this
        # invocation asked for. ⚠ Otherwise a repair run (`--seed 1 --arm lace`)
        # silently overwrites a 3-seed table with a 1-seed one — which it did, once,
        # and was caught only because the file was read straight afterwards.
        on_disk = sorted(int(d.name.split("_")[1]) for d in out.glob("seed_*")
                         if d.is_dir() and (d / "cell_manifest.json").exists())
        agg = aggregate_runs([out / f"seed_{s}" for s in sorted(set(seed_list) | set(on_disk))],
                             log=log)
        agg["seeds_requested"] = sorted(set(seed_list) | set(on_disk))
        agg["seeds_requested_this_invocation"] = seed_list
        agg["seeds_not_run"] = sorted(set(seed_list) - set(agg.get("seeds_present", [])))
        _write_json(agg, out / "cell_results.json")
        if cfg.make_plots and agg.get("arms"):
            try:
                _plot(agg, out / "cell_table.png")
            except Exception as exc:  # a plot never loses a result
                log(f"  ⚠ plot failed (results are already on disk): {exc}")
        return agg
    return {"stage": stage, "out": str(out)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None)
    p.add_argument("--stage", default="cell", choices=list(STAGES) + ["all"] + list(NOT_MINE))
    p.add_argument("--seed", type=int, action="append", default=None)
    p.add_argument("--arm", action="append", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--config", default=None)
    return p


def _config_from_args(args) -> CHLUConfig:
    if args.config:
        from ..config import load_config

        config = load_config(Path(args.config))
    else:
        config = get_default_config()
    if args.dataset:
        config.gen_cell.dataset = str(args.dataset)
    if args.pretext:
        config.gen_cell.pretext = str(args.pretext)
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.stage in NOT_MINE:
        print(f"⛔ stage={args.stage!r} belongs to another module — this one runs "
              f"{STAGES}. Refusing rather than writing cell artefacts into another "
              "stage's provenance.")
        return 2
    config = _config_from_args(args)
    out = args.out or os.environ.get("CHLU_GEN_CELL_OUT")
    t0 = time.time()
    print(f"=== gen-cell-harness: {config.gen_cell.dataset}/{config.gen_cell.pretext} "
          f"stage={args.stage} quick={args.quick} ===")
    run_experiment_gen_cell(config, quick=bool(args.quick), out_dir=out, stage=args.stage,
                            seeds=args.seed, arms=args.arm)
    print(f"=== done in {time.time() - t0:.0f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
