"""⭐⭐ **THE GENERATION HARNESS, END TO END, SCORED ON IMAGES** (R29 pivot item 2).

One path::

    request -> launch read -> codes -> THE frozen decoder -> image
            -> THE external classifier

six arms — ``clu`` · ``store_removed`` · ``knn_proto`` · ``lace`` · ``ebm_compose`` ·
``dynread`` — three seeds, and **one table**.

⭐ **``dynread`` is WIRING, not building** (`gen-read-cell-arm`, 2026-09-15).  The
dynamical read (``request -> posterior-argmax launch -> settle over stored ATOMS ->
decode the settled continuous state``) is built by :mod:`chlu.core.gen_landscape` and
assembled by :mod:`chlu.experiments.exp_gen_dynamical_read`; this module **calls**
both and scores the result **on the cell already here** — same store rows, same
stratified split, same frozen decoder, same hash-asserted external classifier.  ⛔ An
arm scored on a cell of its own is not on this table, and that is the whole point.
⭐ Its ``request_off`` control (``β = 0``, SAME launch, SAME chain) runs beside every
rung: the launch is request-addressed either way, so the control isolates the request
**energy** — and if it scores the same, the arm is settling onto a mode rather than
being steered, **and that is the result**.

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
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import (  # noqa: E402
    CHLUConfig,
    gen_dynamical_read_config,
    gen_row_split_config,
    get_default_config,
)
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
from ..core.gen_landscape import (  # noqa: E402
    AtomLandscape,
    arm_key,
    atom_posterior,
    convergence_report,
    launch_state,
    read_settled,
    save_trace,
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
# ⭐ The dynamical read is CALLED, never re-implemented: ``build_landscape`` and
# ``run_arm`` are ``gen-dynamical-read``'s own, so the atoms, the bandwidths, the
# address book, the coercivity certificate and the settling loop on this table are
# bit-for-bit the ones that spoke built.  ⛔ A second implementation here would make
# the row incomparable with everything that spoke measured.
from ..experiments.exp_gen_dynamical_read import (  # noqa: E402
    build_landscape as dynread_build_landscape,
    run_arm as dynread_run_arm,
)
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
ARMS = ("clu", "store_removed", "knn_proto", "lace", "ebm_compose", "dynread")
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
# 4b. ⭐ THE DYNAMICAL READ as a sixth arm — ⛔ WIRING, not building
# ---------------------------------------------------------------------------

#: ⭐ The two PRE-REGISTERED budgets (``gen-dynamical-read`` PREREG §3).  ``A`` is the
#: MATCHED rung — 64 atoms/well = 524,288 B, **exactly** the codebook's budget — so it
#: is the rung that carries the arm's bare name on the generation table.  ``B`` is the
#: declared-larger rung (512 atoms/well) and is reported beside it, never instead.
DYNREAD_BUDGETS = ("A", "B")
DYNREAD_PRIMARY_BUDGET = "A"
#: ⛔ **Step budgets, declared here and not read from**
#: :attr:`~chlu.config.GenDynamicalReadConfig.steps`, which is a single scalar: this
#: table fixes a different budget per rung (A = 1000, B = 2000).  ⛔ Not a tuning
#: knob — a chain still descending at **its own** step budget is NOT CONVERGED and
#: yields no quotable number, and the number reported is the one at the budget
#: declared before the run.
DYNREAD_STEPS = {"A": 1000, "B": 2000}
#: ⭐ The step counts the still-descending fraction is reported at for EVERY rung, so
#: a budget-B chain's 1000-step state is on the record beside its 2000-step one and
#: "it converged eventually" can never be asserted without the earlier number.
DYNREAD_CONVERGENCE_CHECKPOINTS = (1000, 2000)
#: ⭐ ``full`` and its MANDATORY ``request_off`` control, ⛔ and nothing else.  This
#: spoke is deliberately lean: ``coupling_off``, ``request_off_launch`` and
#: ``zero_step`` belong to ``gen-dynamical-read`` and are ⛔ NOT re-run here.
#: ``request_off`` is not optional — the launch is request-addressed in BOTH arms, so
#: it is the only thing that isolates the request **energy**, which is the entire
#: steering claim (PREREG §4; dreaming's clamp produced a gain its control took whole).
DYNREAD_ARMS = ("full", "request_off")
#: ⛔ The launch this table's rows are scored from.  ⭐ The posterior **argmax** —
#: never the thresholded variant, which degenerates to the prior's argmax because the
#: naive-Bayes posterior clears 0.5 in **0.00 of 5** wells and would carry no request
#: information at all.
DYNREAD_LAUNCH_MODE = "clamp_atom"


def dynread_label(budget: str, inner_arm: str) -> str:
    """``(budget, inner arm) -> the label the table row carries``.

    ⭐ The matched rung's ``full`` arm is plain ``dynread`` — it is THE row the
    generation table gains.  Everything else is suffixed, on the ``lace/<sampler>``
    precedent, so a reader can never mistake a control or the larger rung for it.
    """
    if str(budget) == DYNREAD_PRIMARY_BUDGET and str(inner_arm) == "full":
        return "dynread"
    suffix = "" if str(inner_arm) == "full" else f"_{inner_arm}"
    return f"dynread/{budget}{suffix}"


def dynread_artefact_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    """⭐ Where the traces, settled states and decoded images land.

    ⛔ Under ``.claude/`` and therefore untracked, and ⛔ deliberately **not** under
    ``gen-cell-harness/runs``: these are this spoke's artefacts, and the refusal and
    dreaming spokes are told to look for them here.
    """
    r = os.environ.get("CHLU_GEN_CELL_ARTEFACTS")
    root = Path(r).expanduser() if r else (
        _cell_root(config) / "outputs" / "gen-read-cell-arm" / "runs")
    return root / f"{dataset}_{pretext}" / f"seed_{int(seed)}"


def dynread_convergence_checkpoints(
    trace: dict, window: int, tol: float,
    checkpoints=DYNREAD_CONVERGENCE_CHECKPOINTS,
) -> dict:
    """⭐ The still-descending fraction at **1000 and 2000 steps**, from ONE trace.

    ⛔ A chain is not re-run to answer this: the energy trace is logged every step, so
    the earlier checkpoint is a slice.  ⛔ A checkpoint beyond the chain's own budget
    is reported NOT RUN and is ⛔ never extrapolated.
    """
    E = np.asarray(trace["energy_ref"])
    n_steps = int(E.shape[0]) - 1
    per_step = ("accept_prob", "accepted", "displacement")
    per_state = ("energy", "energy_ref")
    out = {"n_steps_run": n_steps, "window": int(window), "tol": float(tol)}
    for cp in checkpoints:
        cp = int(cp)
        if cp > n_steps:
            out[str(cp)] = {
                "NOT_RUN": (f"⛔ the chain ran {n_steps} steps; the {cp}-step "
                            "checkpoint is NOT RUN and ⛔ is never extrapolated")}
            continue
        sub = {k: np.asarray(trace[k])[: cp + 1] for k in per_state}
        sub.update({k: np.asarray(trace[k])[:cp] for k in per_step})
        r = convergence_report(sub, int(window), float(tol))
        out[str(cp)] = {
            "frac_still_descending": r["frac_still_descending"],
            "n_still_descending": r["n_still_descending"],
            "n_chains": r["n_chains"],
            "energy_final": r["energy_final"],
            "displacement_final_step": r["displacement_final_step"],
            "verdict": r["verdict"],
        }
    return out


def dynread_preflight(config: CHLUConfig) -> dict:
    """⛔ The four standing traps for this arm, as ASSERTIONS rather than comments.

    1. ``λ`` comes from the **coercivity certificate**, ⛔ never the declared 1.0: a
       bilinear coupling is an indefinite quadratic form and an unbounded-below
       landscape does not settle, it diverges (measured ``E = −1.6e28``).
    2. The launch is the posterior **argmax**.
    3. ``T = 0`` at both ends — ⭐ the settling chain is therefore **deterministic**
       (the proposal is a pure gradient step and MH degenerates to *accept iff the
       energy fell*), which is what makes ``request_off`` a genuinely SAME-CHAIN,
       SAME-KEY control rather than a differently-seeded run.
    4. The atoms are stored train rows, asserted bit-for-bit downstream by
       :func:`~chlu.core.gen_landscape.assert_atoms_are_train_rows`.
    """
    c = gen_dynamical_read_config(config)
    if not bool(c.enforce_coupling_certificate):
        raise AssertionError(
            "⛔ enforce_coupling_certificate is OFF: λ would be the declared "
            f"{float(c.coupling_strength):g} rather than the coercivity bound, and an "
            "unbounded-below landscape does not settle — it diverges. Refusing.")
    if str(c.launch_mode) != DYNREAD_LAUNCH_MODE:
        raise AssertionError(
            f"⛔ launch_mode={c.launch_mode!r}; this table's rows are scored from the "
            f"posterior argmax ({DYNREAD_LAUNCH_MODE!r}). ⛔ The thresholded variant "
            "degenerates to the prior's argmax (0.00 of 5 wells clear 0.5) and would "
            "carry no request information at all.")
    if float(c.temp_start) != 0.0 or float(c.temp_end) != 0.0:
        raise AssertionError(
            f"⛔ temp_start={c.temp_start}, temp_end={c.temp_end}: at T > 0 the chain "
            "is stochastic and `request_off` stops being the SAME-CHAIN control the "
            "steering claim rests on. Refusing rather than reporting a control that "
            "differs from its arm in two things at once.")
    return {
        "coupling_certificate_enforced": True,
        "coupling_safety_frac": float(c.coupling_safety_frac),
        "declared_coupling_strength": float(c.coupling_strength),
        "launch_mode": str(c.launch_mode),
        "clamp_threshold": float(c.clamp_threshold),
        "read_mode": str(c.read_mode),
        "temperature": {"start": float(c.temp_start), "end": float(c.temp_end)},
        "chain_is_deterministic": True,
        "same_key_control": ("⭐ T = 0 at both ends ⇒ the proposal is a pure gradient "
                             "step and the MH accept is `de < 0`: the PRNG key is "
                             "INERT, so `request_off` is a same-chain same-key control "
                             "by construction, ⛔ not by assumption."),
        "step_budgets": dict(DYNREAD_STEPS),
        "budgets": list(DYNREAD_BUDGETS),
        "inner_arms": list(DYNREAD_ARMS),
        "NOT_RUN": ("⛔ coupling_off, request_off_launch and zero_step are "
                    "gen-dynamical-read's controls and are NOT re-run on this table "
                    "(this spoke is scoped LEAN); ⛔ their absence is declared, not "
                    "silent."),
    }


def dynread_stored_atom_tuples(built: dict, lab_store: np.ndarray, names, path: Path) -> dict:
    """⛔⭐ **The stored atoms' factor tuples, on disk — dreaming's other half.**

    Dreaming's novelty test is *"a valid tuple that is not a stored atom's tuple"*.
    That needs the decoded tuples **and** the stored atoms' tuples, ⛔ both from THIS
    run: the atoms are chosen by a seeded k-means++ over this seed's pool, so a
    re-derivation elsewhere is a different set of atoms and a different answer.
    """
    rows = np.asarray(built["store"].atom_rows, np.int64)
    tuples = np.asarray(lab_store, np.int64)[rows]
    uniq = np.unique(tuples, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, atom_rows_into_pool=rows, atom_factor_tuples=tuples,
                        distinct_atom_factor_tuples=uniq,
                        factor_names=np.asarray([str(x) for x in names]))
    return {"path": str(path), "n_atoms": int(rows.size),
            "n_distinct_atom_factor_tuples": int(uniq.shape[0]),
            "what": ("⭐ the stored atoms' factor tuples AND the pool row indices they "
                     "came from — the reference set dreaming's novelty test subtracts")}


def run_dynread_budget(
    config: CHLUConfig, budget: str, X_store, lab_store: np.ndarray, well_channels,
    feat_shape, names, cards, seed: int, req_arr: dict, art_dir: Path,
    record: Callable, already: Sequence = (), log: Callable = print,
) -> dict:
    """⭐ One rung of the dynamical read, scored **on this cell**, artefacts kept.

    ⛔ Nothing here re-implements the read.  The landscape is
    ``gen-dynamical-read``'s :func:`build_landscape` over THIS cell's store rows and
    THIS cell's well partition; the chain is its :func:`run_arm`.  What this function
    owns is (a) the per-rung step budget, (b) the artefacts, and (c) handing the
    settled read to the **same** ``_record`` every other arm goes through.

    ⚠ ``key`` derivation is by :func:`~chlu.core.gen_landscape.arm_key`
    (``fold_in``), ⛔ never by lengthening a ``split``: with
    ``jax_threefry_partitionable`` ``split(k, 2)[1]`` **is** ``split(k, 3)[1]``, so a
    lengthened split would hand two arms the same key and their agreement would be
    reported as a finding.
    """
    c = gen_dynamical_read_config(config)
    c.steps = int(DYNREAD_STEPS[str(budget)])
    t0 = time.time()
    wanted = [dynread_label(budget, a) for a in DYNREAD_ARMS]
    if all(w in set(already) for w in wanted):
        # ⛔ Do not rebuild the landscape to skip it: the k-means++ over 32,768 pool
        # rows and the low-rank coupling fit are the expensive half of a rung.
        log(f"  [dynread/{budget}] every rung row is already on disk — skipped")
        return {"budget": str(budget), "rows": [], "skipped": True}
    log(f"  ⭐ [dynread] budget {budget}: {c.steps} settling steps "
        f"(⛔ declared before the run, not tuned)")
    built = dynread_build_landscape(config, str(budget), X_store, lab_store,
                                    well_channels, feat_shape, cards, int(seed), log=log)
    cert = built["coupling_certificate"]
    log(f"    [dynread/{budget}] λ used = {cert['coupling_strength_used']:.4g} "
        f"(declared {cert['declared_coupling_strength']:g}, λ* = "
        f"{cert['max_safe_coupling_strength']:.4g}, capped="
        f"{bool(cert['certificate_bound_the_strength'])}) — ⛔ never 1.0 by fiat")
    atoms_ref = dynread_stored_atom_tuples(
        built, lab_store, names,
        art_dir / f"stored_atom_tuples_s{int(seed)}_{budget}.npz")
    spec = built["spec"]
    key0 = jax.random.PRNGKey(int(c.row_seed_offset) + int(seed))
    bkey = arm_key(key0, f"budget_{budget}")
    rows_done = []
    for inner in DYNREAD_ARMS:
        label = dynread_label(budget, inner)
        if label in already:
            log(f"    [{label}] already on disk for this seed — skipped")
            continue
        feats, per_set, arte = {}, {}, {}
        for rs in REQUEST_SETS:
            rec, states = dynread_run_arm(config, built, inner, req_arr[rs],
                                          arm_key(bkey, rs), log=log)
            feats[rs] = np.asarray(states["read"], np.float32)
            # ⭐ WRITTEN THE MOMENT THE CHAIN FINISHES.  ⛔ Refusal is scored off the
            # energy trace `(final energy, still-descending)` and dreaming off the
            # decoded tuples; losing either costs a full re-run of the scored cell.
            tp = art_dir / f"energy_traces_s{int(seed)}_{budget}_{rs}_{inner}.npz"
            save_trace(states["trace"], tp)
            sp = art_dir / f"settled_states_s{int(seed)}_{budget}_{rs}_{inner}.npz"
            np.savez_compressed(
                sp, x_settled=np.asarray(states["x_settled"], np.float32),
                x0=np.asarray(states["x0"], np.float32),
                read=np.asarray(states["read"], np.float32),
                requests=np.asarray(req_arr[rs], np.int16))
            conv_cp = dynread_convergence_checkpoints(
                states["trace"], int(c.descent_window), float(c.descent_tol))
            per_set[rs] = {
                "n_requests": int(np.asarray(req_arr[rs]).shape[0]),
                "launch": rec["launch"],
                "dynamics": rec["dynamics"],
                "convergence_at": conv_cp,
                "stability": rec["stability"],
                "displacement_from_launch": rec["displacement_from_launch"],
                "settled_spread": rec["settled_spread"],
                "separation": rec["separation"],
                "wall_clock_s": rec["wall_clock_s"],
            }
            arte[rs] = {"energy_traces": str(tp), "settled_states": str(sp)}
            # ⚠ ``·`` not ``/``: "dynread/B" is a RUNG label, and a request-set
            # suffix spelled with a slash would read as one.
            log(f"    ⭐ [{label} · {rs}] E {rec['dynamics']['energy_first']:.3f} -> "
                f"{rec['dynamics']['energy_final']:.3f} · still_descending="
                f"{rec['dynamics']['frac_still_descending']} · "
                f"{rec['dynamics']['verdict']}")
        b = per_set["B"]
        quotable = bool(
            (not b["stability"]["diverged"])
            and float(b["dynamics"]["frac_still_descending"]) == 0.0)
        extra = {
            "dynread": {
                "budget": str(budget), "role": spec["role"],
                "n_atoms_per_well": int(spec["n_atoms"]), "coupling_rank": int(spec["rank"]),
                "inner_arm": str(inner),
                "request_strength": float(0.0 if inner == "request_off"
                                          else c.request_strength),
                "steps": int(c.steps),
                "step_size": float(c.step_size),
                "sigma_anneal": [float(c.sigma_scale_start), float(c.sigma_scale_end)],
                "read_mode": str(c.read_mode),
                "launch_mode": str(c.launch_mode),
                "coupling_certificate": cert,
                "byte_ledger": built["byte_ledger"],
                "byte_ledger_matches_prereg": built["byte_ledger_matches_prereg"],
                "atoms_are_train_rows": built["atoms_are_train_rows"],
                "declared_channel": built["declared_channel"],
                "n_coupling_fit_rows": built["n_coupling_fit_rows"],
                "request_sets": per_set,
                "artefacts": {**arte, "stored_atom_tuples": atoms_ref},
                "QUOTABLE": quotable,
                "quotable_rule": ("⛔ a number is quotable only if its B chains are "
                                  "bounded AND no chain is still descending at the "
                                  "rung's own step budget"),
            },
            "mechanism": (
                "⭐ request → posterior-ARGMAX launch over the stored ATOMS → settle "
                f"{c.steps} steps on E_store + β·E_request − λ·E_couple → decode the "
                "SETTLED CONTINUOUS STATE. ⛔ No codebook gather; the output is not "
                "constrained to lie on a stored point."
                if inner == "full" else
                "⭐ THE STEERING CONTROL: identical launch, identical chain, identical "
                "(inert) key, β = 0. ⛔ The request reaches the state ONLY through the "
                "launch. ⭐ If this scores the same as the full arm, the gain is not "
                "steering — and that is the result, not something to tune around."),
            "channel_bits": None,
        }
        record(label, feats, extra)
        rows_done.append(label)
    return {
        "budget": str(budget), "spec": spec, "rows": rows_done,
        "coupling_certificate": cert,
        "byte_ledger": built["byte_ledger"],
        "stored_atom_tuples": atoms_ref,
        "wall_clock_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# 4b. ⭐⭐ THE ADDRESSED READ (``x0``) — ⛔⛔ AND THE NAME IT MUST NEVER BE GIVEN
# ---------------------------------------------------------------------------
#
# ⭐ ``x0`` is **request -> address-book posterior ARGMAX over the written atoms ->
# decode**.  It is an **ADDRESSED READ**.
#
# ⛔⛔ It is **NEVER** described as dynamics, as settling, as a landscape read, or as
# "the dynamical read at zero steps".  **It spends no steps, it uses no coupling, and
# it has no energy trace.**  ``settle_chain`` does not appear on its path;
# :func:`x0_zero_step_proof` proves that as arithmetic (the decoder's input IS the
# launch, displacement exactly 0.0) rather than asserting it in a docstring, and
# ``test_x0_path_calls_no_chain`` pins it at the source level.
#
# ⭐⭐ **Why this arm exists.**  ``gen-read-cell-arm`` measured, as a WIRING
# DIAGNOSTIC, that the launch its chain starts from scores 0.5008 (rung A) and 0.6852
# (rung B) while the settled state scores 0.2083 / 0.2114.  ⛔ Those were diagnostics,
# not table rows: no byte ledger, no controls, no floor.  This section promotes the
# launch to a scored arm **with its own measured ledger and the charter §C floors**, so
# the question "does addressing ALONE carry the generation table?" has an answer that a
# referee can check.
#
# ⛔⛔ **THE ONE SENTENCE THAT MAY NOT BE MIS-SAID.**  0.6852 is the **8x-BYTES** row.
# The MATCHED-BYTES row is rung A, and at matched bytes this arm is measured against
# ``clu``'s 0.5949 at the SAME 524,288 B.  ⛔ Quoting 0.6852 against 0.5949 without the
# byte multiple attached on the same line is forbidden (PREREG §7.1).

#: ⭐ The arm's name on the table.  ⛔ Deliberately NOT appended to :data:`ARMS`: that
#: tuple is the PIN's declared rival-row list (charter §A4 / the pin table) and editing
#: it would void this spoke's numbers.  ``x0`` is selected explicitly (``--arm x0``),
#: exactly as ``dynread`` is.
X0_ARM = "x0"
#: ⭐ The two byte points, inherited from ``gen-dynamical-read``'s PREREG §3 so the rows
#: are comparable with the dynamical read's at the same atom counts.  ⭐ ``A`` is the
#: **MATCHED-BYTES** rung and carries the arm's bare name; ``B`` is **8x** and is ⛔
#: never quoted as matched.
X0_BUDGETS = ("A", "B")
X0_PRIMARY_BUDGET = "A"
#: ⛔ The launch this arm IS.  The posterior **argmax** — ⛔ never the thresholded
#: variant, which degenerates to the prior's argmax (0.00 of 5 wells clear 0.5).
X0_LAUNCH_MODE = "clamp_atom"
#: ⭐ The atom-mean lookup is spelled as a PINNED launch mode, ⛔ not re-implemented.
X0_LOOKUP_LAUNCH_MODE = "posterior_mean"
#: ⭐ The read this arm's output is decoded through — the SAME one every landscape row
#: on this table uses.  At ``"state"`` it is a reshape, so the decoder's input is the
#: launch itself, bit-for-bit.
X0_READ_MODE = "state"
#: ⭐ One arm (``full``) and **three floors/controls the charter makes mandatory**:
#:
#: ``atom_mean_lookup``  §C.1, **the Advisor's named rival** — the posterior-weighted
#:                       MEAN of the same atoms, at **exactly** the same bytes.
#: ``row_lookup``        §C.1, the *harder* floor — the single best-scoring **whole
#:                       stored train row** (ONE atom index shared by all wells).  It
#:                       removes the per-well recombination and ⛔ nothing else, so the
#:                       gap to ``full`` **is** what composing across wells buys.
#: ``atoms_shuffled``    §C.2, the landscape-removed control — the SAME indices, an
#:                       atom table permuted independently per well.  If this does not
#:                       collapse, the output does not depend on what was written.
#:
#: ⛔ None of the three is a second arm; each is a row the charter refuses to score
#: ``full`` without.
X0_ROWS = ("full", "atom_mean_lookup", "row_lookup", "atoms_shuffled")
#: ⭐ The floors §C.1 binds the arm against.  ⛔ Losing to EITHER means the store is
#: decorative and the arm is reported as such.
X0_LOOKUP_FLOORS = ("atom_mean_lookup", "row_lookup")
#: ⛔ Steps spent. Zero. It is not a budget, not a default and not a knob — it is what
#: the arm IS, and :func:`x0_zero_step_proof` measures it rather than declaring it.
X0_SETTLING_STEPS = 0


def x0_label(budget: str, row: str) -> str:
    """``(rung, row) -> the label the table carries``.

    ⭐ The MATCHED rung's ``full`` row is plain ``x0`` — the one row the generation
    table gains.  Everything else is suffixed on the ``lace/<sampler>`` precedent so a
    reader can never mistake a floor, a control, or the **8x** rung for it.
    """
    if str(budget) == X0_PRIMARY_BUDGET and str(row) == "full":
        return X0_ARM
    suffix = "" if str(row) == "full" else f"_{row}"
    return f"{X0_ARM}/{budget}{suffix}"


def x0_address_book_bytes(book) -> dict:
    """⭐ The atom address book's bytes, **measured on the live object**.

    ⛔ Not computed from the formula and then believed: the leaves are summed, and the
    closed form ``(K*N + sum_f V_f*K*N)*4`` is checked **against** the measurement.
    That closed form is ``gen_compose.address_book_bytes``' formula with ``M -> N``, so
    at ``N = M`` this head is byte-identical to the CLU arm's, which is what makes the
    rung-A comparison a match rather than a resemblance.
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(book, eqx.is_inexact_array))
    measured = int(sum(int(np.asarray(x).size) for x in leaves) * 4)
    K, N = int(book.n_wells), int(book.n_atoms)
    v_sum = int(sum(int(np.asarray(lc).shape[0]) for lc in book.log_cond))
    closed = int((K * N + v_sum * K * N) * 4)
    if measured != closed:
        raise AssertionError(
            f"⛔ the atom address book measures {measured} B but the closed form "
            f"(K*N + sum_f V_f*K*N)*4 = {closed} B — the ledger and the object "
            "disagree and ⛔ no byte claim may be made from either.")
    return {"bytes": measured, "closed_form_bytes": closed, "n_wells": K,
            "n_atoms": N, "sum_cardinalities": v_sum,
            "bytes_formula": "(K*N + sum_f V_f*K*N) * 4", "dtype": "float32",
            "role": ("⭐ the ADDRESSED READ's request-addressing head — the same "
                     "role, the same formula and (at N = M) the same SIZE as the CLU "
                     "arm's; ⛔ excluded from store_bytes by the same convention, so "
                     "the two arms are compared field for field")}


def x0_gather(store, book, requests: np.ndarray, row: str, land=None,
              clamp_threshold: float = 0.5, seed: int = 0) -> dict:
    """``requests -> the decoder's input``, for one of :data:`X0_ROWS`.

    ⭐ ``full`` and ``atom_mean_lookup`` are :func:`~chlu.core.gen_landscape.launch_state`
    **called**, ⛔ not re-implemented — the pinned argmax and the pinned posterior-mean.
    ``row_lookup`` and ``atoms_shuffled`` are derived from the pinned
    :func:`~chlu.core.gen_landscape.atom_posterior` and differ from ``full`` in **one**
    declared quantity each:

    * ``row_lookup``     — one atom index for ALL wells (``S.sum(axis=1).argmax()``)
      instead of one per well.  ⭐ The atoms share a row index across wells by
      construction (``atoms[k] = Z[rows][:, cols_k]``), so this returns a **genuine
      stored train row**, whole.
    * ``atoms_shuffled`` — the SAME per-well indices ``full`` uses, read out of an atom
      table permuted independently per well.  ⛔ Same bytes, same book, same indices;
      only the CONTENT the index points at changes.
    """
    if row not in X0_ROWS:
        raise ValueError(f"row must be one of {X0_ROWS}, got {row!r}")
    R = np.asarray(requests, np.int64)
    if row in ("full", "atom_mean_lookup"):
        mode = X0_LAUNCH_MODE if row == "full" else X0_LOOKUP_LAUNCH_MODE
        lau = launch_state(land, book, R, mode, float(clamp_threshold))
        idx = np.asarray(lau["atom_index"], np.int64)
        return {"x": np.asarray(lau["x0"], np.float32), "launch_mode": mode,
                "mean_confidence": float(lau["mean_confidence"]),
                "mean_n_clamped": float(lau["mean_n_clamped"]),
                "atom_index_per_well": idx,
                "n_distinct_index_tuples": int(np.unique(idx, axis=0).shape[0]),
                "returns_a_stored_atom": bool(row == "full"),
                "what": ("⭐ the posterior-ARGMAX atom in every well, concatenated — "
                         "the ADDRESSED READ" if row == "full" else
                         "⭐ §C.1 FLOOR: the posterior-weighted MEAN of the same "
                         "atoms, at exactly the same bytes")}
    S = atom_posterior(book, R)                       # (n, K, N) log-scores
    x = np.zeros((int(R.shape[0]), int(store.dim)), np.float32)
    if row == "row_lookup":
        # ⛔ the per-well normalisers are constant in the atom index, so summing the
        # UNNORMALISED per-well log-scores is the joint argmax over whole rows.
        idx = np.asarray(S, np.float64).sum(axis=1).argmax(axis=-1)   # (n,)
        for k, w in enumerate(store.well_channels):
            x[:, np.asarray(w, np.int64)] = np.asarray(store.atoms[k], np.float32)[idx]
        rows_into_pool = np.asarray(store.atom_rows, np.int64)[idx]
        return {"x": x, "launch_mode": "row_lookup",
                "atom_index_shared": idx.astype(np.int64),
                "pool_rows_returned": rows_into_pool,
                "n_distinct_rows_returned": int(np.unique(idx).size),
                "returns_a_stored_atom": True,
                "what": ("⭐ §C.1 FLOOR, the harder one: the single best-scoring WHOLE "
                         "stored train row. ⛔ One index for all wells — it differs "
                         "from `full` in the per-well recombination and NOTHING else")}
    idx = np.asarray(S, np.float64).argmax(axis=-1)                   # (n, K)
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_916)
    n_distinct = int(np.unique(idx, axis=0).shape[0])
    perms = []
    for k, w in enumerate(store.well_channels):
        A = np.asarray(store.atoms[k], np.float32)
        p = rng.permutation(A.shape[0])
        perms.append(p)
        x[:, np.asarray(w, np.int64)] = A[p][idx[:, k]]
    return {"x": x, "launch_mode": "atoms_shuffled",
            "atom_index_per_well": idx.astype(np.int64),
            "permutation_seed": int(int(seed) * 7919 + 20_260_916),
            "n_distinct_index_tuples": n_distinct,
            "n_wells_permuted": len(perms),
            "frac_indices_moved": float(np.mean([float((p != np.arange(p.size)).mean())
                                                 for p in perms])),
            "returns_a_stored_atom": True,
            "what": ("⭐ §C.2 LANDSCAPE-REMOVED CONTROL: the SAME indices, an atom "
                     "table permuted independently per well. ⛔ If this does not "
                     "collapse, the output does not depend on what was written")}


def x0_coupling_is_not_needed(built: dict, requests: np.ndarray,
                              clamp_threshold: float = 0.5) -> dict:
    """⛔⭐ **THE CRUX, MEASURED — PREREG §4 assertion A3.**

    The rung-A ledger is ``clu``'s **only if the addressed read does not need the
    low-rank coupling**.  ⛔ That is not assumed here and it is not read off
    :func:`~chlu.core.gen_landscape.launch_state`'s source.  ``x0`` is recomputed on
    four landscapes that differ **only** in the coupling and the two strengths — the
    fitted coupling at the certificate's ``lambda``, ``coupling=None``, ``lambda = 0``
    and ``beta = 0`` — and all four must be **bit-identical** (``np.array_equal``).

    ⛔ If any differ, the coupling IS on the read path, ``P2`` is FALSE, the rung-A
    ledger is 2.00x the budget, **"matched bytes" is withdrawn** and the run aborts
    rather than publishing a match it does not have.
    """
    land = built["land"]
    store, book = built["store"], built["book"]
    R = np.asarray(requests, np.int64)
    ref = np.asarray(
        launch_state(land, book, R, X0_LAUNCH_MODE, float(clamp_threshold))["x0"],
        np.float32)
    variants = {
        "coupling_fitted_at_certificate_lambda": land,
        "coupling_None": AtomLandscape(store, np.asarray(land.sigma), None,
                                       float(land.request_strength),
                                       float(land.coupling_strength)),
        "coupling_strength_zero": land.with_strengths(coupling_strength=0.0),
        "request_strength_zero": land.with_strengths(request_strength=0.0),
    }
    same, maxdiff = {}, {}
    for name, lv in variants.items():
        got = np.asarray(
            launch_state(lv, book, R, X0_LAUNCH_MODE, float(clamp_threshold))["x0"],
            np.float32)
        same[name] = bool(np.array_equal(got, ref))
        maxdiff[name] = float(np.max(np.abs(got - ref))) if got.shape == ref.shape else float("inf")
    if not all(same.values()):
        raise AssertionError(
            "⛔⛔ THE COUPLING IS ON THE ADDRESSED READ'S PATH: x0 is NOT bit-identical "
            f"across {list(variants)} (bit-identical: {same}, max |delta|: {maxdiff}). "
            "⇒ PREREG §5 P2 is FALSE, the rung-A ledger is 2.00x the budget and "
            "'matched bytes' is WITHDRAWN. Refusing to score a match that is not one.")
    return {
        "bit_identical": same, "max_abs_delta": maxdiff,
        "n_requests": int(R.shape[0]),
        "coupling_needed": False,
        "verdict": ("⭐⭐ MEASURED, NOT ASSUMED: x0 is bit-identical with the fitted "
                    "coupling, with NO coupling, at lambda = 0 and at beta = 0. ⇒ the "
                    "addressed read is a function of (request, address book, atoms) "
                    "ONLY, the coupling is NOT deployed, and the arm's ledger is "
                    "STORE + ADDRESSING HEAD — ⭐ half the dynamical read's."),
    }


def x0_zero_step_proof(x: np.ndarray, read: np.ndarray, feat_shape) -> dict:
    """⛔⭐ **"It spends zero steps" as ARITHMETIC** — PREREG §4 assertion A4.

    At ``read_mode="state"`` the read is a reshape, so the decoder's input **is** the
    launch.  This measures that: the displacement from launch to decoder input is
    **exactly** 0.0, bit-for-bit.  ⛔ A row that cannot show this has a chain on its
    path and is not this arm.
    """
    flat = np.asarray(read, np.float32).reshape((int(np.asarray(read).shape[0]), -1))
    xa = np.asarray(x, np.float32)
    if not np.array_equal(flat, xa):
        raise AssertionError(
            "⛔ the decoder's input is NOT the launch, bit-for-bit — something is on "
            f"this arm's path between them (max |delta| = {np.max(np.abs(flat - xa))})")
    return {
        "settling_steps": int(X0_SETTLING_STEPS),
        "displacement_from_launch_rms": 0.0,
        "displacement_from_launch_max": 0.0,
        "decoder_input_is_the_launch_bitwise": True,
        "energy_trace": None,
        "coupling_used": False,
        "read_mode": str(X0_READ_MODE),
        "feat_shape": [int(v) for v in feat_shape],
        "what": ("⭐ MEASURED: the decoder's input is the launch itself, bit-for-bit. "
                 "⛔ No chain, no steps, no energy trace, no coupling. ⛔⛔ This arm is "
                 "an ADDRESSED READ and is NEVER called dynamics, settling, a "
                 "landscape read, or 'the dynamical read at zero steps'."),
    }


def x0_preflight(config: CHLUConfig) -> dict:
    """⛔ The traps for an ADDRESSED read, as assertions rather than comments."""
    c = gen_dynamical_read_config(config)
    if str(c.launch_mode) != X0_LAUNCH_MODE:
        raise AssertionError(
            f"⛔ launch_mode={c.launch_mode!r}; this arm IS the posterior argmax "
            f"({X0_LAUNCH_MODE!r}). ⛔ The thresholded variant degenerates to the "
            "prior's argmax and would carry no request information at all.")
    if str(c.read_mode) != X0_READ_MODE:
        raise AssertionError(
            f"⛔ read_mode={c.read_mode!r}; at anything but {X0_READ_MODE!r} the read "
            "is not a reshape and the decoder's input stops being the launch, which "
            "is the one thing this arm's zero-step claim rests on.")
    return {
        "arm": X0_ARM,
        "mechanism": ("request -> address-book posterior ARGMAX over the written "
                      "atoms -> decode"),
        "launch_mode": str(c.launch_mode),
        "clamp_threshold": float(c.clamp_threshold),
        "read_mode": str(c.read_mode),
        "settling_steps": int(X0_SETTLING_STEPS),
        "coupling_used": False,
        "budgets": list(X0_BUDGETS),
        "primary_budget": X0_PRIMARY_BUDGET,
        "rows": list(X0_ROWS),
        "lookup_floors": list(X0_LOOKUP_FLOORS),
        "NAMING_RULE": ("⛔⛔ This arm is an ADDRESSED READ. It is NEVER described as "
                        "dynamics, as settling, as a landscape read, or as 'the "
                        "dynamical read at zero steps'. It spends ZERO steps, uses NO "
                        "coupling and has NO energy trace."),
        "HEADLINE_RULE": ("⛔⛔ The headline is the MATCHED-BYTES rung (A). The rung-B "
                          "row is 8x the bytes and is quoted ONLY with its byte "
                          "multiple attached on the same line."),
        "NOT_RUN": ("⛔ no ladder, no tuning rungs, no settling, no temperature, no "
                    "second mechanism — declared, not silent."),
    }


def run_x0_budget(
    config: CHLUConfig, budget: str, X_store, lab_store: np.ndarray, well_channels,
    feat_shape, names, cards, seed: int, req_arr: dict, art_dir: Path,
    record: Callable, clu_ledger: dict, already: Sequence = (), log: Callable = print,
) -> dict:
    """⭐ One byte point of the addressed read, scored **on this cell**.

    ⛔ Nothing here re-implements the read.  The store, the bandwidths and the address
    book are ``gen-dynamical-read``'s :func:`build_landscape` over THIS cell's store
    rows and THIS cell's well partition — so the atoms are bit-for-bit the ones the
    ``dynread`` rows launched from — and the gather is
    :func:`~chlu.core.gen_landscape.launch_state`.  What this function owns is the
    **measured byte ledger**, the **floors**, and handing each row to the **same**
    ``_record`` every other arm on this table goes through.
    """
    c = gen_dynamical_read_config(config)
    t0 = time.time()
    wanted = [x0_label(budget, r) for r in X0_ROWS]
    if all(w in set(already) for w in wanted):
        log(f"  [x0/{budget}] every row is already on disk — skipped")
        return {"budget": str(budget), "rows": [], "skipped": True}
    log(f"  ⭐ [x0] byte point {budget}: the ADDRESSED READ — ⛔ ZERO settling steps, "
        "⛔ no coupling, ⛔ no energy trace")
    # ⚠ ``build_landscape`` logs its own ledger, which INCLUDES the coupling and reads
    # "2.00x budget".  ⛔ That is the DYNAMICAL read's ledger.  The correction below is
    # logged immediately after so no reader of this log can inherit the wrong number.
    built = dynread_build_landscape(config, str(budget), X_store, lab_store,
                                    well_channels, feat_shape, cards, int(seed), log=log)
    spec, store, book = built["spec"], built["store"], built["book"]

    # ---- ⭐⭐ THE BYTE LEDGER — measured on the live objects, PREREG §4 A1-A3 ----
    coupling_proof = x0_coupling_is_not_needed(built, req_arr["B"],
                                               float(c.clamp_threshold))
    head = x0_address_book_bytes(book)
    store_bytes = int(store.n_bytes())
    if store_bytes != int(built["byte_ledger"]["store"]["store_bytes"]):
        raise AssertionError(
            f"⛔ the live store measures {store_bytes} B, the declared ledger says "
            f"{built['byte_ledger']['store']['store_bytes']} B")
    if store_bytes != int(spec["budget_bytes"]):
        raise AssertionError(
            f"⛔ the live store measures {store_bytes} B and rung {budget}'s declared "
            f"budget is {spec['budget_bytes']} B — the ledger and the object disagree")
    matched = bool(int(store_bytes) == int(clu_ledger["deployed_bytes"])
                   and int(head["bytes"]) == int(clu_ledger["addressing_head"]["bytes"]))
    if str(budget) == X0_PRIMARY_BUDGET and not matched:
        raise AssertionError(
            f"⛔⛔ rung {budget} is THE MATCHED-BYTES rung and it does not match: store "
            f"{store_bytes} vs clu {clu_ledger['deployed_bytes']}, head {head['bytes']} "
            f"vs clu {clu_ledger['addressing_head']['bytes']}. ⛔ Refusing to publish a "
            "matched-bytes verdict without matched bytes.")
    ledger = {
        "arm": X0_ARM, "rung": str(spec["label"]), "rung_role": str(spec["role"]),
        "budget_bytes": int(clu_ledger["deployed_bytes"]),
        "deployed_bytes": int(store_bytes),
        "store_bytes": int(store_bytes),
        "addressing_head_bytes": int(head["bytes"]),
        "addressing_head": head,
        "coupling_bytes": 0,
        "coupling_bytes_excluded": int(built["byte_ledger"]["coupling"]["coupling_bytes"]),
        "coupling_excluded_reason": coupling_proof["verdict"],
        "coupling_not_needed_proof": coupling_proof,
        "n_atoms_per_well": int(spec["n_atoms"]),
        "multiple_of_budget": float(store_bytes) / float(clu_ledger["deployed_bytes"]),
        "byte_matched_to_clu_field_for_field": matched,
        "clu_reference": {"store_bytes": int(clu_ledger["deployed_bytes"]),
                          "addressing_head_bytes":
                              int(clu_ledger["addressing_head"]["bytes"]),
                          "source": "⭐ read from THIS manifest's own store ledger, "
                                    "⛔ not from a literal in this file"},
        "bytes_formula": "N * D * 4 (store) ; (K*N + sum_f V_f*K*N)*4 (addressing head)",
        "what": ("the stored ATOMS (genuine train rows) plus the request-addressing "
                 "head; the read is one stored atom per well, concatenated, at the "
                 "store's FULL FLOAT WIDTH"),
        "budget_note": (
            "⭐⭐ MATCHED, AND MEASURED: the atom table is 524,288 B and the addressing "
            "head 74,240 B — ⭐ EXACTLY the CLU arm's two numbers, field for field, at "
            "N = M = 64. ⛔ The low-rank coupling is NOT on this read's path (proved "
            "bit-for-bit above) so its bytes are NOT deployed: the dynamical read "
            "carried 2.00x the budget at this rung, the addressed read carries 1.00x."
            if matched else
            "⛔ DECLARED LARGER, NEVER MATCHED: this rung's atom table is "
            f"{float(store_bytes) / float(clu_ledger['deployed_bytes']):.2f}x the "
            "524,288 B budget. ⛔ Its number is quoted ONLY with that multiple "
            "attached on the same line."),
    }
    log(f"    ⭐⭐ [x0/{budget}] MEASURED LEDGER: store {store_bytes:,} B + addressing "
        f"head {head['bytes']:,} B = {ledger['multiple_of_budget']:.2f}x the "
        f"{clu_ledger['deployed_bytes']:,} B budget · coupling "
        f"{ledger['coupling_bytes_excluded']:,} B ⛔ NOT DEPLOYED (bit-identical x0 "
        f"without it) · byte-matched to clu field-for-field: {matched}")

    rows_done = []
    for row in X0_ROWS:
        label = x0_label(budget, row)
        if label in already:
            log(f"    [{label}] already on disk for this seed — skipped")
            continue
        feats, per_set, arte = {}, {}, {}
        for rs in REQUEST_SETS:
            g = x0_gather(store, book, req_arr[rs], row, land=built["land"],
                          clamp_threshold=float(c.clamp_threshold), seed=int(seed))
            read = read_settled(built["land"], g["x"], X0_READ_MODE)
            zero = x0_zero_step_proof(g["x"], read, feat_shape)
            feats[rs] = np.asarray(read, np.float32)
            sp = art_dir / f"x0_states_s{int(seed)}_{budget}_{rs}_{row}.npz"
            sp.parent.mkdir(parents=True, exist_ok=True)
            # ⛔ arrays go to the npz, ⛔ never into the manifest as a length: a count
            # standing where an index array was is how a manifest stops meaning
            # anything.  The manifest keeps the SCALARS only.
            np.savez_compressed(
                sp, x0=np.asarray(g["x"], np.float32),
                requests=np.asarray(req_arr[rs], np.int16),
                **{k: np.asarray(v) for k, v in g.items()
                   if k != "x" and isinstance(v, np.ndarray)})
            per_set[rs] = {"n_requests": int(np.asarray(req_arr[rs]).shape[0]),
                           "zero_step_proof": zero,
                           "arrays_on_disk": sorted(
                               k for k, v in g.items()
                               if k != "x" and isinstance(v, np.ndarray)),
                           **{k: v for k, v in g.items()
                              if k != "x" and not isinstance(v, np.ndarray)}}
            arte[rs] = {"x0_states": str(sp)}
        extra = {
            "addressed_read": {
                "budget": str(budget), "role": spec["role"],
                "row": str(row),
                "is_the_arm": bool(row == "full"),
                "role_on_the_table": (
                    "⭐ THE ARM" if row == "full" else
                    "⭐ charter §C.1 MATCHED-BYTES LOOKUP FLOOR" if row in X0_LOOKUP_FLOORS
                    else "⭐ charter §C.2 LANDSCAPE-REMOVED CONTROL"),
                "n_atoms_per_well": int(spec["n_atoms"]),
                "settling_steps": int(X0_SETTLING_STEPS),
                "coupling_used": False,
                "energy_trace": None,
                "byte_ledger": ledger,
                "atoms_are_train_rows": built["atoms_are_train_rows"],
                "declared_channel": built["declared_channel"],
                "request_sets": per_set,
                "artefacts": arte,
                "QUOTABLE": True,
                "quotable_rule": ("⭐ an addressed read has no chain, so there is no "
                                  "still-descending fraction and no convergence "
                                  "caveat: the number IS the read. ⛔ What bounds it "
                                  "instead is the BYTE MULTIPLE in its ledger."),
            },
            # ⛔ `arm_byte_ledger` is a PINNED byte-ledger assertion and this spoke may
            # not edit it; the measured ledger is supplied here instead, where
            # `_record`'s `**extra` places it over the default.
            "byte_ledger": ledger,
            "mechanism": (
                "⭐ request -> address-book posterior ARGMAX over the written atoms -> "
                "decode. ⛔⛔ AN ADDRESSED READ: zero settling steps, no coupling, no "
                "energy trace. ⛔ NEVER 'dynamics', 'settling' or 'the dynamical read "
                "at zero steps'."
                if row == "full" else
                "⭐ charter §C.1 FLOOR — the posterior-weighted MEAN of the same atoms "
                "at exactly the same bytes. ⛔ If the arm does not beat this, the "
                "store is decorative."
                if row == "atom_mean_lookup" else
                "⭐ charter §C.1 FLOOR (the harder one) — the single best-scoring WHOLE "
                "stored train row, ONE index for all wells. ⛔ It differs from the arm "
                "in the per-well recombination and NOTHING else."
                if row == "row_lookup" else
                "⭐ charter §C.2 CONTROL — the SAME indices into an atom table permuted "
                "independently per well. ⛔ If this does not collapse, the output does "
                "not depend on what was written."),
            "channel_bits": None,
        }
        record(label, feats, extra)
        rows_done.append(label)
    return {"budget": str(budget), "spec": spec, "rows": rows_done,
            "byte_ledger": ledger, "coupling_not_needed": coupling_proof,
            "wall_clock_s": float(time.time() - t0)}


# ---------------------------------------------------------------------------
# 5. ⭐ THE BYTE LEDGER — a table COLUMN, never a footnote (R26.3)
# ---------------------------------------------------------------------------


def arm_byte_ledger(arm: str, cfg, store_led: dict, proto_led: dict,
                    lace_state: Optional[dict], ebm: Optional[dict],
                    dynread_ledgers: Optional[dict] = None) -> dict:
    """Deployed bytes per arm, with the ~31 % kNN advantage **declared in words**."""
    base = {"arm": arm, "budget_bytes": int(cfg.expect_deployed_store_bytes or 0)}
    if arm.split("/")[0] == "dynread":
        led = (dynread_ledgers or {}).get(arm)
        if led is None:
            return {**base, "deployed_bytes": None,
                    "what": "⛔ NOT RUN — no landscape ledger was recorded for this row"}
        spec, ledger = led["spec"], led["byte_ledger"]
        return {**base,
                "deployed_bytes": int(ledger["total_bytes"]),
                "store_bytes": int(ledger["store"]["store_bytes"]),
                "coupling_bytes": int(ledger["coupling"]["coupling_bytes"]),
                "addressing_head_bytes": 0,
                "rung": spec["label"], "rung_role": spec["role"],
                "n_atoms_per_well": int(spec["n_atoms"]), "coupling_rank": int(spec["rank"]),
                "multiple_of_budget": float(ledger["multiple_of_budget"]),
                "what": ("the stored ATOMS (N·D·4 — genuine train rows, ⛔ not fitted "
                         "centres) plus the low-rank inter-well coupling; the read is "
                         "the settled continuous state at the store's FULL FLOAT WIDTH"),
                "budget_note": (
                    "⚠ **DECLARED, NOT SILENT:** at rung A the ATOM TABLE is byte-matched "
                    f"to the codebook ({int(ledger['store']['store_bytes']):,} B = the "
                    "524,288 B store) but the coupling is a SECOND table of the same size, "
                    f"so the rung's total is {float(ledger['multiple_of_budget']):.2f}x the "
                    "budget — PREREG §3's declared D3 allowance, ⛔ not a match. ⭐ Rung B "
                    "is declared LARGER and is never quoted as matched-bytes.")}
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
    # ⛔⛔ A MEASURED SILENT TRUNCATION IN THE SHARED PATH (2026-09-15).
    # ``stratified_fit_val`` CLAMPS ``n_val`` to ``pool.size − n_fit`` — 512 at the
    # shipped ``n_fit`` — so a caller asking for a larger ``n_val`` gets 512 rows and a
    # **completely normal log**.  ⛔ The repair belongs to whoever owns
    # ``chlu/data/gen_rowsplit.py``; what is binding HERE is that the REALISED number
    # is asserted against the requested one and recorded, so this cell can never be
    # silently truncated and a future caller cannot inherit the trap unseen.
    # ⛔ The shipped ``n_val = 512`` is NOT raised in this spoke — that is the ladder's
    # business, not the scored cell's.
    n_val_requested = int(cfg.n_val)
    if n_val != n_val_requested:
        raise AssertionError(
            f"⛔ SILENT TRUNCATION: n_val was requested as {n_val_requested} and the "
            f"split realised {n_val} (pool {pool.size} − n_fit {n_fit} = "
            f"{max(pool.size - int(cfg.n_fit), 0)}). `stratified_fit_val` clamps and "
            "logs normally. ⛔ Refusing to score a cell whose validation set is not "
            "the one that was asked for.")
    row_balance = row_split_balance(
        split.grid.factor_indices_of(fit_rows), split.grid.factor_indices_of(val_rows),
        list(split.grid.names), list(split.grid.cardinalities),
    )
    if bool(rs.assert_balanced) and str(rs.mode) != "head_tail":
        row_balance = assert_row_split_balanced(
            row_balance, require_all_values=bool(rs.require_all_values),
            max_count_ratio=float(rs.max_count_ratio),
        )
    row_split = {**split_info, "balance": row_balance,
                 "n_val_requested": n_val_requested, "n_val_realised": n_val,
                 "n_val_clamp_headroom": int(max(pool.size - int(cfg.n_fit), 0)),
                 "n_val_truncation_asserted": True}
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
         f"re-encode check within 1 ULP of the cache dtype on {reenc.get('n_checked')} rows")

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
    dynread_ledgers: dict = {}

    def _record(arm: str, feats: dict, extra: dict, tuning_trials: int,
                persist_to: Optional[Path] = None) -> None:
        imgs = {k: render(decoder, feats[k], feat_shape, cfg.eval_batch)
                for k in REQUEST_SETS}
        # ⭐ ARTEFACTS, not results.  ⛔ Refusal (target 2) and dreaming (target 4) are
        # scored off what this run produces and throws away unless it is kept: the
        # DECODED IMAGES and the external classifier's factor tuple per decoded image.
        # ⛔ Losing them costs a full re-run of the scored cell.
        if persist_to is not None:
            persist_to.mkdir(parents=True, exist_ok=True)
            safe = arm.replace("/", "__")
            for k in REQUEST_SETS:
                dp = persist_to / f"decoded_s{int(seed)}_{safe}_{k}.npz"
                np.savez_compressed(
                    dp,
                    images=np.asarray(imgs[k], np.float32),
                    factor_tuples=np.asarray(
                        predict_factors(external, imgs[k], int(cfg.eval_batch)), np.int64),
                    features=np.asarray(feats[k], np.float32).reshape(
                        (int(np.asarray(feats[k]).shape[0]), -1)),
                    requests=np.asarray(req_arr[k], np.int16),
                    labels=np.asarray(lab[k], np.int64),
                    factor_names=np.asarray([str(x) for x in names]))
                extra.setdefault("dynread", {}).setdefault("artefacts", {}).setdefault(
                    k, {})["decoded"] = str(dp)
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
                                           ebm_state, dynread_ledgers),
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
        if arm == "dynread":
            # ⭐ ``dynread`` is FOUR rows (2 rungs x {full, request_off}), so the
            # resume check is per ROW, ⛔ not per arm name: a run that died after the
            # matched rung must not be told the whole arm is done.
            wanted_rows = {dynread_label(b, i)
                           for b in DYNREAD_BUDGETS for i in DYNREAD_ARMS}
            if wanted_rows <= set(manifest["arms"]):
                tlog("  [dynread] every rung already on disk for this seed — skipped")
                continue
        elif arm == X0_ARM:
            # ⭐ ``x0`` is EIGHT rows (2 byte points x {the arm, two §C.1 lookup
            # floors, the §C.2 control}), so the resume check is per ROW, ⛔ not per
            # arm name.
            wanted_rows = {x0_label(b, r) for b in X0_BUDGETS for r in X0_ROWS}
            if wanted_rows <= set(manifest["arms"]):
                tlog("  [x0] every row is already on disk for this seed — skipped")
                continue
        elif arm == "kstar":
            # ⭐ w15/lace-poe-rows: the K* single design is FIVE rows (the arm + four
            # charter-C controls), so the resume check is per ROW.
            from . import exp_gen_cell_kstar as _ks
            if {_ks.kstar_label(r) for r in _ks.KSTAR_ROWS} <= set(manifest["arms"]):
                tlog("  [kstar] every row is already on disk for this seed — skipped")
                continue
        elif arm in manifest["arms"]:
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
            elif arm == "dynread":
                pre = dynread_preflight(config)
                manifest["dynread_preflight"] = pre
                art = dynread_artefact_dir(config, dataset, pretext, int(seed))
                art.mkdir(parents=True, exist_ok=True)
                tlog(f"  ⭐ [dynread] artefacts -> {art}  (⛔ they are not results; "
                     "refusal and dreaming are scored FROM them and must never have "
                     "to re-run this cell)")

                def _rec_dynread(label: str, feats: dict, extra: dict,
                                 _art: Path = art) -> None:
                    d = extra["dynread"]
                    # ⛔ the ledger must exist BEFORE _record, which reads it
                    dynread_ledgers[label] = {
                        "spec": {"label": d["budget"], "role": d["role"],
                                 "n_atoms": d["n_atoms_per_well"],
                                 "rank": d["coupling_rank"]},
                        "byte_ledger": d["byte_ledger"]}
                    _record(label, feats, extra, 0, persist_to=_art)

                snapshot = set(manifest["arms"])
                runs = {}
                for budget in DYNREAD_BUDGETS:
                    runs[budget] = run_dynread_budget(
                        config, budget, X_store, lab_store, wc, feat_shape, names,
                        cards, int(seed), req_arr, art, _rec_dynread,
                        already=snapshot, log=tlog)
                manifest["dynread"] = {"preflight": pre, "artefact_dir": str(art),
                                       "budgets": runs}
                _write_json(manifest, man_path)
            elif arm == X0_ARM:
                # ⭐⭐ THE ADDRESSED READ.  ⛔⛔ Zero settling steps, no coupling, no
                # energy trace — and ⛔ never described as dynamics (§4b).
                pre = x0_preflight(config)
                manifest["x0_preflight"] = pre
                art = Path(x0_artefact_dir(config, dataset, pretext, int(seed)))
                art.mkdir(parents=True, exist_ok=True)
                tlog(f"  ⭐ [x0] artefacts -> {art}")

                def _rec_x0(label: str, feats: dict, extra: dict) -> None:
                    _record(label, feats, extra, 0)

                snapshot = set(manifest["arms"])
                x0_runs = {}
                for budget in X0_BUDGETS:
                    x0_runs[budget] = run_x0_budget(
                        config, budget, X_store, lab_store, wc, feat_shape, names,
                        cards, int(seed), req_arr, art, _rec_x0, store_led,
                        already=snapshot, log=tlog)
                manifest["x0"] = {"preflight": pre, "artefact_dir": str(art),
                                  "budgets": x0_runs}
                _write_json(manifest, man_path)
            elif arm == "kstar":
                # ⭐ w15/lace-poe-rows — THE K* SINGLE DESIGN (``exp_gen_cell_kstar``).
                # ⛔ Deliberately NOT on ARMS (the pin's rival-row list), as x0/dynread.
                from . import exp_gen_cell_kstar as _ks
                built = _ks.build(X_store, lab_store, cards, int(seed), log=tlog)
                kobjs = _ks.objects(built, int(seed))
                vals = {k: _ks.request_values(grid.values, lab[k])
                        for k in list(REQUEST_SETS) + ["val"]}
                # ⭐ VAL FIRST: the gate is evaluated on the cell's val rows and written
                # to the manifest BEFORE any B/MS number of this arm is recorded.
                val_scores = {}
                for r, o in kobjs.items():
                    fv, _ = _ks.read(built, o, vals["val"], grid.values)
                    val_scores[r] = float(score_images(
                        external, render(decoder, fv, feat_shape, cfg.eval_batch),
                        lab["val"], names, cfg.eval_batch)["aggregate"])
                gate = _ks.val_gate(val_scores)
                manifest["kstar_val"] = {"gate": gate, "info": built["info"],
                                         "ledger": built["ledger"]}
                _write_json(manifest, man_path)
                tlog(f"  ⭐ [kstar] VAL GATE: arm {gate['arm_val_aggregate']:.4f} · "
                     f"controls {gate['controls_val_aggregate']} · PASS={gate['PASS']}")
                if os.environ.get("CHLU_KSTAR_VAL_ONLY") == "1":
                    raise RuntimeError("val-only invocation (CHLU_KSTAR_VAL_ONLY=1): the "
                                       "gate is on disk, ⛔ no table row is recorded")
                if not gate["PASS"]:
                    raise AssertionError(f"⛔ the K* arm FAILED its val bar: {gate}")
                snapshot = set(manifest["arms"])
                for r in _ks.KSTAR_ROWS:
                    label = _ks.kstar_label(r)
                    if label in snapshot:
                        continue
                    feats, per_set = {}, {}
                    for rs_ in REQUEST_SETS:
                        f_, oor_ = _ks.read(built, kobjs[r], vals[rs_], grid.values)
                        feats[rs_] = f_
                        per_set[rs_] = {"n": int(f_.shape[0]),
                                        "n_out_of_range": int(np.sum(oor_))}
                    _record(label, feats, _ks.row_extra(built, r, per_set), 0)
            else:
                raise ValueError(
                    f"unknown arm {arm!r} (have {ARMS} + {X0_ARM!r}; ⛔ {X0_ARM!r} is "
                    "deliberately NOT on ARMS — that tuple is the pin's declared "
                    "rival-row list and editing it would void this table's numbers)")
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
    # ⭐⭐ the generation-table verdict, in PREREG §2b's words and no others
    out["dynread_verdict"] = dynread_verdict(mans)
    # ⭐⭐ the ADDRESSED READ's verdict — the SAME rule, plus the charter §C.1 floors
    out["x0_verdict"] = addressed_read_verdict(mans)
    out["dynread_convergence"] = {
        int(m["seed"]): {
            a: {"B": (((r.get("dynread") or {}).get("request_sets") or {})
                      .get("B", {}).get("convergence_at")),
                "QUOTABLE": (r.get("dynread") or {}).get("QUOTABLE")}
            for a, r in m.get("arms", {}).items() if a.split("/")[0] == "dynread"
        } for m in mans}
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
    if out["dynread_verdict"].get("headline"):
        log(out["dynread_verdict"]["headline"])
    return out


# ---------------------------------------------------------------------------
# 7b. ⭐⭐ THE VERDICT — PREREG §2b's words, ⛔ and no other vocabulary
# ---------------------------------------------------------------------------

#: ⛔ ``gen-dynamical-read`` PREREG §2b's equivalence margin, HARDENED by
#: ``PREREG-addendum-0`` (Advisor, 2026-09-15).  Fixed **before** the run as the sum of
#: the two arms' published 3-seed sds (0.0122 + 0.0082 = 0.0204, rounded down to
#: 0.020).  ⛔ ``m`` is not re-derived after the run, and ⛔ nothing here re-reads it
#: from the measured spread.
DYNREAD_MARGIN = 0.020
#: ⛔⛔ **A TIE IS AVAILABLE ONLY AGAINST THE CEILING.**
DYNREAD_CEILING_ARM = "knn_proto"
#: ⭐ Every one of these is a like-for-like row and must be **BEATEN OUTRIGHT, beyond
#: seed spread**.  ⛔ A tie against any of them is a LOSS and is reported as one.
#: ⚠ ``clu`` is the codebook read this arm REPLACES; beating everything except it
#: would mean the read improved nothing.
DYNREAD_LIKE_FOR_LIKE = ("store_removed", "ebm_compose", "lace", "clu")
#: ⛔ The headline metric for target 1.  ⛔ ``in-corner balacc`` is reported beside it
#: and is **not substitutable for it after the fact**.
DYNREAD_HEADLINE = "external composition-B aggregate"


def _paired_by_seed(mans: list, arm: str) -> dict:
    return {int(m["seed"]): float(m["arms"][arm]["external"]["composition_B"]["aggregate"])
            for m in mans if arm in m.get("arms", {})}


def dynread_verdict(mans: list, arm: str = "dynread",
                    margin: float = DYNREAD_MARGIN) -> dict:
    """⭐⭐ The generation-table verdict, computed — ⛔ never written by hand.

    ⛔ **The vocabulary is PREREG §2b's and there is no other.**  WIN · TIE
    (favourable) · TIE · LOSS · INDETERMINATE against the ceiling; BEATEN OUTRIGHT or
    LOSS against every like-for-like row.  ⛔ A ``+0.001`` with three agreeing signs is
    a **TIE**, never a WIN — that is what ``PREREG-addendum-0`` hardened.

    ⛔ ``Δ`` is the mean **paired, per-seed** difference on the SAME seeds, so a rival
    missing a seed can never flatter or damage the comparison through the mean.
    "Beyond seed spread" is ``mean ± std`` disjointness **and** 3/3 agreeing signs.
    """
    ours = _paired_by_seed(mans, arm)
    out = {
        "arm": arm,
        "metric": DYNREAD_HEADLINE,
        "margin_m": float(margin),
        "margin_provenance": ("⛔ PREREG §2b, fixed BEFORE the run as 0.0122 + 0.0082 "
                              "= 0.0204 rounded down; ⛔ not re-derived after it"),
        "rule": ("⭐ PREREG §2b as hardened by PREREG-addendum-0. ⛔ A TIE is available "
                 f"ONLY against the ceiling ({DYNREAD_CEILING_ARM}); every like-for-like "
                 "row must be BEATEN OUTRIGHT, beyond seed spread, and a tie against one "
                 "of those is reported as a LOSS."),
        "ours": {"per_seed": ours, "mean": float(np.mean(list(ours.values())))
                 if ours else float("nan"),
                 "std": float(np.std(list(ours.values()))) if ours else float("nan")},
        "against": {},
    }
    if not ours:
        out["verdict"] = "NOT RUN"
        out["note"] = f"⛔ {arm!r} is on no seed's manifest — ⛔ reported NOT RUN, never as a zero"
        return out
    for rival in (DYNREAD_CEILING_ARM,) + tuple(DYNREAD_LIKE_FOR_LIKE):
        theirs = _paired_by_seed(mans, rival)
        shared = sorted(set(ours) & set(theirs))
        if not shared:
            out["against"][rival] = {"verdict": "NOT RUN",
                                     "why": "⛔ no seed carries both arms"}
            continue
        d = [ours[s] - theirs[s] for s in shared]
        delta = float(np.mean(d))
        a_m, a_s = float(np.mean([ours[s] for s in shared])), float(np.std([ours[s] for s in shared]))
        b_m, b_s = float(np.mean([theirs[s] for s in shared])), float(np.std([theirs[s] for s in shared]))
        all_pos = bool(all(x > 0 for x in d))
        all_neg = bool(all(x < 0 for x in d))
        same_sign = bool(all_pos or all_neg)
        overlap = bool(not (a_m - a_s > b_m + b_s or b_m - b_s > a_m + a_s))
        is_ceiling = bool(rival == DYNREAD_CEILING_ARM)
        if is_ceiling:
            if delta > margin and all_pos:
                v = "WIN"
            elif 0.0 < delta <= margin and all_pos:
                v = "TIE (favourable)"
            elif abs(delta) <= margin and overlap:
                v = "TIE"
            elif delta < -margin:
                v = "LOSS"
            else:
                v = "INDETERMINATE"
        else:
            v = ("BEATEN OUTRIGHT" if (delta > 0.0 and all_pos and not overlap)
                 else "LOSS")
        out["against"][rival] = {
            "role": "ceiling (⭐ the only row a TIE is available against)" if is_ceiling
                    else "like-for-like (⛔ must be BEATEN OUTRIGHT; a tie here is a LOSS)",
            "seeds": shared,
            "ours_mean": a_m, "ours_std": a_s,
            "theirs_mean": b_m, "theirs_std": b_s,
            "delta": delta,
            "paired_per_seed": {int(s): float(ours[s] - theirs[s]) for s in shared},
            "same_sign_3_of_3": same_sign,
            "n_seeds_same_sign": int(max(sum(1 for x in d if x > 0),
                                         sum(1 for x in d if x < 0))),
            "intervals_overlap_mean_pm_sd": overlap,
            "verdict": v,
        }
    ceil = out["against"].get(DYNREAD_CEILING_ARM, {})
    lfl = {k: out["against"][k]["verdict"] for k in DYNREAD_LIKE_FOR_LIKE
           if k in out["against"]}
    not_beaten = [k for k, v in lfl.items() if v != "BEATEN OUTRIGHT"]
    out["verdict"] = ceil.get("verdict", "NOT RUN")
    out["like_for_like"] = lfl
    out["like_for_like_all_beaten"] = bool(not not_beaten)
    out["PASS"] = bool(
        (not not_beaten)
        and out["verdict"] in ("WIN", "TIE (favourable)", "TIE"))
    out["headline"] = (
        f"⭐ {arm} vs {DYNREAD_CEILING_ARM}: Δ = {ceil.get('delta', float('nan')):+.4f} "
        f"(m = {margin:.3f}) ⇒ **{out['verdict']}**; like-for-like "
        + (" · ".join(f"{k}: {v}" for k, v in lfl.items()) or "NOT RUN")
        + ("" if not not_beaten else
           f"  ⛔ NOT BEATEN OUTRIGHT: {', '.join(not_beaten)} — reported as LOSS(es)"))
    return out


def x0_artefact_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    """⭐ Where this arm's launches land — under ``.claude/`` and therefore untracked."""
    r = os.environ.get("CHLU_GEN_X0_ARTEFACTS")
    root = Path(r).expanduser() if r else (
        _cell_root(config) / "outputs" / "gen-addressed-read" / "runs")
    return root / f"{dataset}_{pretext}" / f"seed_{int(seed)}"


def _x0_pair(ours: dict, theirs: dict, margin: float, is_ceiling: bool = False) -> dict:
    """⭐ ONE paired comparison, in PREREG §2b's vocabulary and no other.

    ⛔ This is :func:`dynread_verdict`'s rule, spelled once so the charter §C.1 floors
    — which are NOT on the pin's :data:`DYNREAD_LIKE_FOR_LIKE` tuple and ⛔ may not be
    added to it — are judged by **exactly** the same arithmetic.  The agreement is not
    left to inspection: :func:`addressed_read_verdict` re-derives every pinned rival
    through *both* paths and ⛔ aborts if they ever disagree.
    """
    shared = sorted(set(ours) & set(theirs))
    if not shared:
        return {"verdict": "NOT RUN", "why": "⛔ no seed carries both rows"}
    d = [ours[s] - theirs[s] for s in shared]
    delta = float(np.mean(d))
    a_m, a_s = float(np.mean([ours[s] for s in shared])), float(np.std([ours[s] for s in shared]))
    b_m, b_s = float(np.mean([theirs[s] for s in shared])), float(np.std([theirs[s] for s in shared]))
    all_pos = bool(all(x > 0 for x in d))
    same_sign = bool(all_pos or all(x < 0 for x in d))
    overlap = bool(not (a_m - a_s > b_m + b_s or b_m - b_s > a_m + a_s))
    if is_ceiling:
        if delta > margin and all_pos:
            v = "WIN"
        elif 0.0 < delta <= margin and all_pos:
            v = "TIE (favourable)"
        elif abs(delta) <= margin and overlap:
            v = "TIE"
        elif delta < -margin:
            v = "LOSS"
        else:
            v = "INDETERMINATE"
    else:
        v = ("BEATEN OUTRIGHT" if (delta > 0.0 and all_pos and not overlap) else "LOSS")
    return {"seeds": shared, "ours_mean": a_m, "ours_std": a_s,
            "theirs_mean": b_m, "theirs_std": b_s, "delta": delta,
            "paired_per_seed": {int(s): float(ours[s] - theirs[s]) for s in shared},
            "same_sign_3_of_3": same_sign,
            "intervals_overlap_mean_pm_sd": overlap, "verdict": v}


def addressed_read_verdict(mans: list, margin: float = DYNREAD_MARGIN) -> dict:
    """⭐⭐ **THE ADDRESSED READ'S VERDICT — the matched-bytes row first, always.**

    ⛔⛔ The headline of this arm is the **MATCHED-BYTES** rung (A, 524,288 B, the byte
    ledger ``clu`` deploys field for field).  The rung-B row is **8x the bytes** and is
    emitted only inside ``eight_times_companion``, with ``byte_multiple`` on the same
    record — ⛔ there is no field in this output that carries the 8x number without it.

    Three things are computed per rung:

    1. the pinned :func:`dynread_verdict` against ``knn_proto`` (ceiling) and the four
       like-for-like rows — ⭐ including ``clu``, **the codebook read at the same bytes**;
    2. ⭐ the **charter §C.1 floors** — the matched-bytes lookups.  ⛔ An arm that does
       not beat BOTH is one whose store is decorative, and it is reported as such
       whatever the rest of the table says;
    3. the **charter §C.2 control** — the permuted atom table, which must collapse.
    """
    out = {
        "arm": X0_ARM,
        "metric": DYNREAD_HEADLINE,
        "margin_m": float(margin),
        "what_the_arm_is": ("⭐ request → address-book posterior ARGMAX over the "
                            "written atoms → decode. ⛔⛔ AN ADDRESSED READ: zero "
                            "settling steps, no coupling, no energy trace. ⛔ NEVER "
                            "'dynamics', 'settling', or 'the dynamical read at zero "
                            "steps'."),
        "reporting_rule": ("⛔⛔ THE HEADLINE IS THE MATCHED-BYTES RUNG. The 8x row "
                           "never appears without its byte multiple on the same line."),
        "per_rung": {},
    }
    for budget in X0_BUDGETS:
        label = x0_label(budget, "full")
        ours = _paired_by_seed(mans, label)
        rung = {"label": label, "byte_point": str(budget)}
        led = next((m["arms"][label]["byte_ledger"]
                    for m in mans if label in m.get("arms", {})), None)
        if led:
            rung["bytes"] = {
                "store_bytes": int(led.get("store_bytes", 0)),
                "addressing_head_bytes": int(led.get("addressing_head_bytes", 0)),
                "coupling_bytes": int(led.get("coupling_bytes", 0)),
                "coupling_bytes_excluded": int(led.get("coupling_bytes_excluded", 0)),
                "multiple_of_budget": float(led.get("multiple_of_budget", float("nan"))),
                "byte_matched_to_clu_field_for_field":
                    bool(led.get("byte_matched_to_clu_field_for_field", False)),
                "coupling_needed": bool(
                    (led.get("coupling_not_needed_proof") or {}).get("coupling_needed",
                                                                     True)),
            }
        if not ours:
            rung["verdict"] = "NOT RUN"
            rung["note"] = f"⛔ {label!r} is on no seed's manifest — ⛔ NOT RUN, never a zero"
            out["per_rung"][budget] = rung
            continue
        rung["ours"] = {"per_seed": ours, "mean": float(np.mean(list(ours.values()))),
                        "std": float(np.std(list(ours.values())))}
        rung["distinct_factor_tuples_B"] = _ms(
            [m["arms"][label]["distinct"]["B"]["factor_tuples"]
             for m in mans if label in m.get("arms", {})])
        pinned = dynread_verdict(mans, arm=label, margin=margin)
        rung["against"] = pinned["against"]
        rung["verdict_vs_ceiling"] = pinned["verdict"]
        rung["like_for_like"] = pinned["like_for_like"]
        rung["like_for_like_all_beaten"] = pinned["like_for_like_all_beaten"]
        # ⛔ the two paths must agree, or neither is trusted
        for rival, got in pinned["against"].items():
            mine = _x0_pair(ours, _paired_by_seed(mans, rival), margin,
                            is_ceiling=bool(rival == DYNREAD_CEILING_ARM))
            if str(mine["verdict"]) != str(got["verdict"]):
                raise AssertionError(
                    f"⛔ the floor comparator and the pinned `dynread_verdict` disagree "
                    f"on {rival!r}: {mine['verdict']!r} vs {got['verdict']!r}. ⛔ No "
                    "verdict is written from two rules that differ.")
        # ---- ⭐ charter §C.1 — THE MATCHED-BYTES LOOKUP FLOORS -----------------
        floors = {}
        for f in X0_LOOKUP_FLOORS:
            flabel = x0_label(budget, f)
            floors[flabel] = {
                "role": ("⭐ charter §C.1 matched-bytes lookup — ⛔ the arm must beat "
                         "it beyond seed spread or the store is DECORATIVE"),
                "their_per_seed": _paired_by_seed(mans, flabel),
                **_x0_pair(ours, _paired_by_seed(mans, flabel), margin),
            }
        rung["charter_C1_floors"] = floors
        judged = [v["verdict"] for v in floors.values() if v["verdict"] != "NOT RUN"]
        rung["charter_C1_floor_PASS"] = bool(
            judged and all(v == "BEATEN OUTRIGHT" for v in judged))
        rung["charter_C1_floor_statement"] = (
            "⭐ the arm BEATS every matched-bytes lookup outright — the store is "
            "load-bearing on this metric"
            if rung["charter_C1_floor_PASS"] else
            "⛔⛔ THE ARM DOES NOT BEAT A MATCHED-BYTES LOOKUP TABLE ⇒ the store is "
            "DECORATIVE and this row does NOT count as a positive (charter §C.1), "
            "whatever the rest of the table says")
        # ---- ⭐ charter §C.2 — the landscape-removed control -------------------
        clabel = x0_label(budget, "atoms_shuffled")
        ctrl = _paired_by_seed(mans, clabel)
        rung["charter_C2_control"] = {
            "label": clabel,
            "role": ("⭐ the SAME indices into an atom table permuted independently "
                     "per well — it must COLLAPSE"),
            "their_per_seed": ctrl,
            **_x0_pair(ours, ctrl, margin),
        }
        rung["charter_C2_control_COLLAPSES"] = bool(
            rung["charter_C2_control"].get("verdict") == "BEATEN OUTRIGHT")
        # ---- the rung's own PASS ---------------------------------------------
        rung["PASS"] = bool(
            pinned["like_for_like_all_beaten"]
            and rung["charter_C1_floor_PASS"]
            and rung["charter_C2_control_COLLAPSES"]
            and pinned["verdict"] in ("WIN", "TIE (favourable)", "TIE"))
        out["per_rung"][budget] = rung

    a = out["per_rung"].get(X0_PRIMARY_BUDGET, {})
    clu_row = (a.get("against") or {}).get("clu", {})
    out["MATCHED_BYTES_VERDICT"] = {
        "⛔ read this field first": ("⭐⭐ THIS IS THE HEADLINE. The arm's number at the "
                                     "bytes the codebook read deploys."),
        "rung": X0_PRIMARY_BUDGET,
        "label": a.get("label"),
        "bytes": a.get("bytes"),
        "ours": a.get("ours"),
        "distinct_factor_tuples_B": a.get("distinct_factor_tuples_B"),
        "vs_clu_the_codebook_read_at_the_same_bytes": clu_row,
        "VERDICT_vs_clu": clu_row.get("verdict", "NOT RUN"),
        "VERDICT_vs_ceiling_knn_proto": a.get("verdict_vs_ceiling", "NOT RUN"),
        "like_for_like": a.get("like_for_like"),
        "like_for_like_all_beaten": a.get("like_for_like_all_beaten"),
        "charter_C1_floor_PASS": a.get("charter_C1_floor_PASS"),
        "charter_C1_floor_statement": a.get("charter_C1_floor_statement"),
        "charter_C2_control_COLLAPSES": a.get("charter_C2_control_COLLAPSES"),
        "PASS": a.get("PASS"),
    }
    b = out["per_rung"].get("B", {})
    bm = ((b.get("bytes") or {}).get("multiple_of_budget"))
    out["eight_times_companion"] = {
        "⛔ never quote this without the byte multiple": True,
        "label": b.get("label"),
        "byte_multiple": bm,
        "one_line": (
            f"{b.get('label')} = {(b.get('ours') or {}).get('mean', float('nan')):.4f} "
            f"± {(b.get('ours') or {}).get('std', float('nan')):.4f} "
            f"**at {bm:.2f}× the bytes** (⛔ NOT byte-matched; the matched-bytes row is "
            f"`{a.get('label')}`)" if bm else "⛔ NOT RUN"),
        "vs_clu_at_8x_bytes": (b.get("against") or {}).get("clu", {}),
        "charter_C1_floor_PASS": b.get("charter_C1_floor_PASS"),
        "PASS": b.get("PASS"),
    }
    out["headline"] = (
        f"⭐⭐ MATCHED BYTES ({a.get('bytes', {}).get('store_bytes', '?')} B store + "
        f"{a.get('bytes', {}).get('addressing_head_bytes', '?')} B head = "
        f"{(a.get('bytes') or {}).get('multiple_of_budget', float('nan')):.2f}× budget): "
        f"`{a.get('label')}` = {(a.get('ours') or {}).get('mean', float('nan')):.4f} ± "
        f"{(a.get('ours') or {}).get('std', float('nan')):.4f} vs `clu` "
        f"{clu_row.get('theirs_mean', float('nan')):.4f} ± "
        f"{clu_row.get('theirs_std', float('nan')):.4f} ⇒ **{clu_row.get('verdict', 'NOT RUN')}** "
        f"(Δ = {clu_row.get('delta', float('nan')):+.4f}, m = {margin:.3f}) · "
        f"§C.1 floor {'PASS' if a.get('charter_C1_floor_PASS') else '⛔ FAIL'} · "
        f"ceiling {a.get('verdict_vs_ceiling', 'NOT RUN')} · "
        f"PASS: {'YES' if a.get('PASS') else 'NO'}")
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
