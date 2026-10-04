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

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, get_default_config, gen_row_split_config  # noqa: E402
from ..core.gen_bridge import (  # noqa: E402
    PRETEXTS,
    assert_decoder_hash,
    assert_no_holdout_leak,
    dataset_module,
    load_feature_cache,
    param_hash,
)
from ..core.gen_request import (  # noqa: E402
    FactorPrototypes,
    control_map_of,
    deduplicate_terms,
    dense_request_arrays,
    load_c_requests,
    load_query_set_registry,
    load_registered_control_requests,
    requests_from_rows,
)
from ..data.gen_rowsplit import (  # noqa: E402
    assert_row_split_balanced,
    row_split_balance,
    split_fit_val,
)
from ..data.gen_splits import default_gen_data_root  # noqa: E402
from ..eval.gen_probe import (  # noqa: E402
    binary_auroc,
    decode_features,
    discrimination_metrics,
    factor_labels,
    factor_probs,
    multiclass_balanced_accuracy,
    probe_accuracy,
)
from ..eval.rivals import gen_knn as knn  # noqa: E402
from ..eval.rivals import gen_lace as lace  # noqa: E402
from ..experiments.exp_gen_bridge import artifacts_tag, condition_of  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

STAGES = ("baselines", "aggregate")
#: Kept so the launcher's other ``--stage`` values do not silently no-op here.
NOT_MINE = ("encoder", "decoder", "store", "cell")
ARMS = ("knn", "lace", "all")
#: ⭐ The only genuinely compositional factor (R16); always its own column.
COMPOSITIONAL_FACTOR = "shape"
#: Request sets, in the order every table uses them.
REQUEST_SETS = ("A", "B", "MS", "C")
#: ⭐ The VALID CONTROL sets the C kinds declare in ``query_sets.json`` (A7).  ⛔ They
#: are scored for **refusal only** — a control is never decoded, never probed and
#: never enters composition, so adding one cannot move a composition number.
#: ⛔ Kept OUT of :data:`REQUEST_SETS` deliberately: that tuple drives the decode /
#: probe path and the LACE sampler's PRNG stream, and re-ordering it would move
#: every historical number.
CONTROL_SETS = ("A7", "A7D")
#: The registry name each entry of :data:`CONTROL_SETS` instantiates.  ⛔ ``A7D`` is
#: deliberately absent: it is a DECLARED SENSITIVITY (the A7 records with the
#: redundant term removed), ⛔ never a control a kind may be scored against.
CONTROL_SET_NAMES = {"A7": "a7_redundant"}
#: The sensitivity set and the control it perturbs.
CONTROL_DEDUP_OF = {"A7D": "A7"}
#: ⭐ The registry's ``base`` set — "one term per factor, the plain valid request" —
#: is instantiated by the arms' own **B** pool (R14.1's primary negative set), ⛔ not
#: by a second copy built here.  ``out_of_range`` declares ``base``, which is why it
#: is A7-clean as shipped and must reproduce bit-identically.
CONTROL_BASE_SET = "B"


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick run's numbers are NEVER reported."""
    g = config.gen_baselines
    g.seeds = list(g.quick_seeds)
    g.n_fit = g.quick_n_fit
    g.n_val = g.quick_n_val
    g.lace_n_epochs = g.quick_lace_n_epochs
    g.lace_tune_epochs = g.quick_lace_tune_epochs
    g.lace_tune_n = g.quick_lace_tune_n
    g.lace_tune_ld_steps = g.quick_lace_tune_ld_steps
    g.lace_ld_steps = g.quick_lace_ld_steps
    g.knn_kmeans_iters = g.quick_knn_kmeans_iters
    g.knn_ks = list(g.quick_knn_ks)
    g.lace_sigmas = list(g.quick_lace_sigmas)
    g.lace_shrinkages = list(g.quick_lace_shrinkages)
    config.gen_bridge.n_eval_a = g.quick_n_eval_a
    return config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def decoder_lift_dir(config: CHLUConfig) -> Path:
    """Where ``gen-decoder-lift`` wrote the WINNING decoder (⛔ not the baseline)."""
    d = config.gen_baselines.decoder_lift_dir or os.environ.get("CHLU_GEN_DECLIFT_DIR")
    if d:
        return Path(d).expanduser()
    return _repo_root() / ".claude" / "outputs" / "gen-decoder-lift" / "winner"


def store_runs_root(config: CHLUConfig) -> Path:
    d = config.gen_baselines.store_runs_root or os.environ.get("CHLU_GEN_STORE_RUNS")
    if d:
        return Path(d).expanduser()
    return _repo_root() / ".claude" / "outputs" / "gen-store-grouping" / "runs"


def resolve_store_bytes(config: CHLUConfig, dataset: str, pretext: str, log: Callable) -> dict:
    """⭐ The byte-match target: **store + inter-well coupling** — ⛔ never hand-solved.

    ⭐ **The landscape is two structures, not one** (Head ruling, 2026-09-12).  The
    modes are ``store_bytes``; ``gen_dream.fit_code_coupling`` additionally learns an
    inter-well PMI field ``J[k, l][m, m']`` which is **extra** to it —
    ``gen_dream.coupling_bytes``'s own docstring says a byte-matched baseline *"must
    be told about it"* or *"the byte match is wrong"*.

    ⛔ Matching to ``store_bytes`` alone under-resources the comparator by 24 % at the
    shipped cell, and the kNN arm is the comparator for **two** kill rules — (ii) on B
    and (iv) on C — so a weak one hands **both** a false pass.  That is the inversion
    this spoke exists to avoid.

    ⭐ Both components are derived mechanically: ``store_bytes`` from
    ``store_bytes.json``, the coupling from ``K(K-1)/2 * M^2 * 4`` using
    ``store_plan.json``'s **own** ``K`` and ``n_modes``.  ⛔ Neither is hand-typed.
    ⭐ The read channel is untouched at ``K*log2(M) = 30`` bits (R20's design
    constant); this is a **capacity** correction, not a channel one.

    ⛔ **SUPERSEDED BY R26 RULING 3 (2026-09-13), and this is now the default.**
    Conditional dreaming is dropped as the composition mechanism, so the coupling
    field leaves the deployed store and the target reverts **688,128 -> 524,288 B**.
    :attr:`GenBaselinesConfig.include_coupling_bytes` therefore defaults to ``False``
    and the realised total is **asserted** against
    :attr:`GenBaselinesConfig.expect_byte_target_bytes` — a budget that silently
    reverts is the defect this assertion exists to catch.  ⭐ The flag survives so
    the historical 688,128 B run reproduces bit-identically.
    """
    g = config.gen_baselines
    runs = store_runs_root(config) / f"{dataset}_{pretext}"
    p = runs / "store_bytes.json"
    if g.store_bytes is not None:
        store_b, src, meta = int(g.store_bytes), "pinned by gen_baselines.store_bytes", None
    else:
        if not p.exists():
            raise FileNotFoundError(
                f"no machine-readable store byte count at {p}. ⛔ The kNN arm cannot be "
                "byte-matched without it and an unmatched kNN arm is worthless — stop "
                "and report (set gen_baselines.store_bytes or $CHLU_GEN_STORE_RUNS)."
            )
        with open(p) as fh:
            meta = json.load(fh)
        store_b, src = int(meta["store_bytes"]), str(p)

    coupling_b, c_src, k_wells, n_modes = 0, "excluded (include_coupling_bytes=False)", None, None
    if bool(g.include_coupling_bytes):
        if g.coupling_bytes is not None:
            coupling_b, c_src = int(g.coupling_bytes), "pinned by gen_baselines.coupling_bytes"
        else:
            plan = runs / "store_plan.json"
            if plan.exists():
                with open(plan) as fh:
                    pl = json.load(fh)
                k_wells, n_modes = int(pl["K"]), int(pl["n_modes"])
                c_src = f"{plan} (K={k_wells}, n_modes={n_modes})"
            elif meta is not None:
                k_wells, n_modes = int(meta["n_wells"]), int(meta["n_modes"])
                c_src = f"{p} (K={k_wells}, n_modes={n_modes})"
            else:
                raise FileNotFoundError(
                    f"cannot derive the coupling bytes: neither {plan} nor a store "
                    "manifest is readable, and gen_baselines.coupling_bytes is unset. "
                    "⛔ Guessing it would silently under- or over-resource the kNN arm."
                )
            coupling_b = (k_wells * (k_wells - 1) // 2) * n_modes * n_modes * 4

    total = store_b + coupling_b
    expect = g.expect_byte_target_bytes
    if expect is not None and int(total) != int(expect):
        raise AssertionError(
            f"⛔ realised byte-match target {total} B != the expected "
            f"{int(expect)} B (store {store_b} + coupling {coupling_b}). R26 ruling 3 "
            "took the inter-well coupling out of the deployed store: the target is "
            "524,288 B. Set gen_baselines.expect_byte_target_bytes (or --byte-budget "
            "historical) to reproduce the superseded 688,128 B run DELIBERATELY — "
            "⛔ never by inheriting a stale default."
        )
    log(
        f"  ⭐ byte-match target: {total} B = store {store_b} + coupling {coupling_b} "
        f"(K(K-1)/2 * M^2 * 4). ⛔ The read channel is UNCHANGED at K*log2(M) bits."
        + (f"  [asserted == {int(expect)} B]" if expect is not None
           else "  ⚠ assertion DISABLED (expect_byte_target_bytes=None)")
    )
    if coupling_b:
        log("  ⚠ include_coupling_bytes=True reproduces the SUPERSEDED (R26.3) budget: "
            "the kNN arm carries a declared byte ADVANTAGE and is ⛔ NOT byte-matched.")
    return {
        "store_bytes": store_b,
        "coupling_bytes": coupling_b,
        "total_bytes": total,
        "expected_total_bytes": (int(expect) if expect is not None else None),
        "assertion": ("realised == expect_byte_target_bytes" if expect is not None
                      else "⚠ DISABLED — expect_byte_target_bytes is None"),
        "superseded_by_R26_ruling_3": bool(coupling_b),
        "K": k_wells,
        "n_modes": n_modes,
        "coupling_bytes_formula": "K(K-1)/2 * M^2 * 4",
        "source_store": src,
        "source_coupling": c_src,
        "store_meta": meta,
        "rationale": (
            "⛔ R26 ruling 3 (2026-09-13) SUPERSEDES the 2026-09-12 Head ruling that set "
            "this target to store + inter-well PMI coupling. Conditional dreaming is "
            "dropped as the composition mechanism, so gen_dream's coupling field is "
            "neither fitted nor deployed and the target reverts 688,128 -> 524,288 B "
            "(= store_bytes). The historical 3-seed run at 688,128 B gave the kNN arm a "
            "declared ~31 % byte ADVANTAGE — conservative in direction, ⛔ but it is "
            "NOT byte-matched and must never be quoted as such. ⭐ bits_per_read is "
            "untouched at K*log2(M) either way (R20)."
        ),
    }


def _extract_per_request_scores(arms: dict) -> dict:
    """POP the per-request tables out of the arm dicts into their own payload.

    ⭐ They are persisted (acceptance item: a future re-score must be a spreadsheet),
    but they are **bulk** — keeping them inside ``baselines_manifest.json`` would
    multiply it by an order of magnitude and every reader of the manifest pays for
    a table only the re-score needs.
    """
    out: dict = {}
    for arm_name, arm in (arms or {}).items():
        holders = {}
        holders.update(arm.get("variants", {}) or {})
        holders.update(arm.get("samplers", {}) or {})
        for vname, v in holders.items():
            table = (v.get("refusal") or {}).pop("per_request_scores", None)
            if table is not None:
                out.setdefault(arm_name, {})[vname] = table
    return out


def _write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# ⭐ THE shared frozen decoder — hash-asserted before any number is recorded
# ---------------------------------------------------------------------------


def load_shared_decoder(config: CHLUConfig, bdir: Path, bman: dict, seed: int, log: Callable):
    """Load THE decoder every arm renders through, and **assert its identity**.

    ⛔ ``decoder_source="bridge"`` loads the **SUPERSEDED** decoder that failed the
    gate; it exists only so a deliberate historical reproduction is possible and
    it shouts in the log when selected.
    """
    g = config.gen_baselines
    src = str(g.decoder_source)
    if src == "bridge":
        dec = load_model(bdir / "decoder.pkl")
        if g.assert_decoder_hash:
            assert_decoder_hash(dec, bman["decoder_param_hash"])
        log(
            "  ⛔⛔ decoder_source='bridge': this is the SUPERSEDED decoder that FAILED "
            "the gate (0.7191 balanced accuracy, 0/3 seeds). Every number below is "
            "historical reproduction, NOT a phase-1 result."
        )
        return dec, {
            "decoder_source": "bridge",
            "recipe": "baseline",
            "param_hash": bman["decoder_param_hash"],
            "superseded": True,
        }
    if src != "decoder_lift_v1":
        raise ValueError(
            f"unknown decoder_source {src!r} (have 'decoder_lift_v1', 'bridge')"
        )
    wdir = decoder_lift_dir(config)
    hashes_path = wdir / "winner_hashes.json"
    if not hashes_path.exists():
        raise FileNotFoundError(
            f"no gen-decoder-lift winner manifest at {hashes_path}. ⛔ Refusing to fall "
            "back to the bridge decoder: that is the recipe that FAILED the gate and a "
            "run through it looks entirely normal in the logs."
        )
    with open(hashes_path) as fh:
        wh = json.load(fh)[f"seed_{int(seed)}"]
    dec = load_model(wdir / f"decoder_v1_s{int(seed)}.pkl")
    got = param_hash(dec)
    if g.assert_decoder_hash:
        if got != wh["param_hash"]:
            raise AssertionError(
                f"V1 decoder parameter hash {got} != {wh['param_hash']} — this is NOT "
                "gen-decoder-lift's winner; every arm number would be void"
            )
        if wh["encoder_param_hash_it_was_fit_against"] != bman["encoder_param_hash"]:
            raise AssertionError(
                "the V1 decoder was fit against a different encoder than this seed's "
                "bridge encoder — the feature space does not match"
            )
    log(
        f"  ⭐ decoder: {wh['recipe']['name']} (gen-decoder-lift winner) "
        f"param_hash={got[:16]}… params={wh['param_count']} — asserted"
    )
    return dec, {
        "decoder_source": "decoder_lift_v1",
        "recipe": wh["recipe"]["name"],
        "param_hash": got,
        "param_count": int(wh["param_count"]),
        "encoder_param_hash_it_was_fit_against": wh["encoder_param_hash_it_was_fit_against"],
        "superseded": False,
    }


# ---------------------------------------------------------------------------
# Rows and requests
# ---------------------------------------------------------------------------


def select_baseline_rows(split, config: CHLUConfig, seed: int, log: Callable = None) -> dict:
    """Fit / validation / A / B rows.

    ⭐ ``fit`` and ``val`` **partition exactly** the row set ``gen-store-grouping``
    fitted its store on (``select_store_rows``), so the matched-seen negatives are
    excluded by the same rows and the realised negative count reproduces R22's
    (198 / 199 / 199).  ⛔ ``val`` is disjoint from ``fit``, from ``eval_a`` (A), from
    ``eval_b`` (B) and from the negatives, and C never enters tuning at all.

    ⛔ **The partition is STRATIFIED** (:mod:`chlu.data.gen_rowsplit`), ⛔ no longer the
    head and tail of a *sorted* array.  A sorted tail is a contiguous corner of the
    factor grid: it realised **1** ``floor_hue`` value against ``fit``'s 10, and the two
    halves of it chose different winning trials in 5 of 6 cells, so every configuration
    selected on it was unidentified.  ⭐ ``pool`` itself is untouched — only how it is
    cut.  The realised balance is asserted and recorded.
    """
    g = config.gen_baselines
    rs = gen_row_split_config(config)
    rows = select_store_rows(split, config, int(seed))
    pool = np.asarray(rows["store_train"], np.int64)
    labels = split.grid.factor_indices_of(pool)
    fit, val, info = split_fit_val(
        pool, labels, int(g.n_fit), int(g.n_val), int(seed),
        mode=str(rs.mode), tiebreak_jitter=float(rs.tiebreak_jitter),
        seed_mult=int(rs.seed_mult), seed_offset=int(rs.seed_offset), log=log,
    )
    balance = row_split_balance(
        split.grid.factor_indices_of(fit), split.grid.factor_indices_of(val),
        list(split.grid.names), list(split.grid.cardinalities),
    )
    if bool(rs.assert_balanced) and str(rs.mode) != "head_tail":
        balance = assert_row_split_balanced(
            balance, require_all_values=bool(rs.require_all_values),
            max_count_ratio=float(rs.max_count_ratio),
        )
    out = dict(rows)
    out["fit"] = fit
    out["val"] = val
    out["row_split"] = {**info, "balance": balance}
    return out


def _corner_mask(values: np.ndarray) -> np.ndarray:
    """Montero's colour/scale/orientation corner (the same predicate ``matched_seen_rows`` uses)."""
    v = np.asarray(values)
    return (
        (v[:, 0] >= 0.8) & (v[:, 1] >= 0.8) & (v[:, 2] >= 0.8)
        & (v[:, 3] >= 1.1) & (v[:, 5] > 20)
    )


# ---------------------------------------------------------------------------
# Scoring helpers (⛔ every metric comes from chlu.eval.gen_probe)
# ---------------------------------------------------------------------------


def score_composition(decoder, probe, feats, labels, names, feat_shape, batch) -> dict:
    """Decode features through THE frozen decoder, read them with THE frozen probe."""
    Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(feat_shape))
    imgs = decode_features(decoder, Z, int(batch))
    out = probe_accuracy(probe, imgs, labels, names, int(batch))
    out["decoded_mean"] = float(imgs.mean())
    return out


def corner_discrimination(
    decoder, probe, feats_pos, feats_neg, labels_pos, labels_neg,
    shape_idx: int, n_shape_classes: int, feat_shape, batch: int
) -> dict:
    """⭐ What kill rule (i) reads (R20/R23): DISCRIMINATION within the corner.

    Positives = renders of the held-out requests; negatives = renders of the
    matched-seen requests (same corner, a seen shape).  Balanced accuracy is the
    statistic; AUROC is the mandatory companion and ⛔ never the gate.
    ⛔ Chance is 0.5 and these numbers are **not** comparable in level to a raw
    multi-class ``shape`` accuracy.
    """
    Zp = np.asarray(feats_pos, np.float32).reshape((-1,) + tuple(feat_shape))
    Zn = np.asarray(feats_neg, np.float32).reshape((-1,) + tuple(feat_shape))
    ip = decode_features(decoder, Zp, int(batch))
    inn = decode_features(decoder, Zn, int(batch))
    pp = factor_probs(probe, ip, shape_idx, int(batch))
    pn = factor_probs(probe, inn, shape_idx, int(batch))
    targets = np.unique(np.asarray(labels_pos, np.int64)[:, shape_idx])
    out = {"targets_realised": [int(t) for t in targets]}
    if targets.size == 1:
        out.update(discrimination_metrics(pp, pn, int(targets[0])))
    else:
        # ⭐ Multi-class held-out cells (the widened configuration): the binary
        # in-corner read is moot, so the macro-recall read is the like-for-like one.
        pred = np.concatenate([np.argmax(pp, 1), np.argmax(pn, 1)])
        true = np.concatenate(
            [np.asarray(labels_pos)[:, shape_idx], np.asarray(labels_neg)[:, shape_idx]]
        )
        mc = multiclass_balanced_accuracy(true, pred, int(n_shape_classes))
        out.update(
            {"balanced_accuracy": mc["balanced_accuracy"], "auroc": float("nan"),
             "n_pos": int(pp.shape[0]), "n_neg": int(pn.shape[0]),
             "multiclass": mc, "sufficient": True,
             "note": "multi-class held-out cell: macro-recall, ⛔ not the binary in-corner read"}
        )
    return out


def _oracle_aurocs(s_a, s_b, s_c) -> dict:
    """⭐ ORACLE-ONLY per-trial refusal AUROCs.

    ⛔ These use C and are therefore **never** the selection criterion; they exist
    so the family's upper bound is available on **both** metrics — including
    ``C-vs-(A u B)``, which is the metric kill rule (iv) actually names.

    ⚠ Why an oracle bound is the honest comparator here, stated once.  A refusal
    configuration cannot be selected from valid data alone: no valid-only
    statistic measures *detection power*.  Giving the kNN a surrogate invalid set
    would fix that — and would also **break the supervision-class match** that
    makes kill rule (iv) meaningful (R7: kNN's threshold is "the same supervision
    class as us: no labels, no dynamics, **no invalid examples**").  So the arm
    ships a deployable, composition-selected number *and* an explicitly-labelled
    oracle bound, and ⛔ does not invent a third thing.
    """
    a, b, c = (np.asarray(x, np.float64) for x in (s_a, s_b, s_c))

    def au(neg, pos):
        return binary_auroc(
            np.concatenate([neg, pos]),
            np.concatenate([np.zeros(neg.size, bool), np.ones(pos.size, bool)]),
        )

    return {
        "auroc_c_vs_b_ORACLE_ONLY": au(b, c),
        "auroc_c_vs_a_ORACLE_ONLY": au(a, c),
        "auroc_c_vs_a_union_b_ORACLE_ONLY": au(np.concatenate([a, b]), c),
    }


def _auroc_by_kind(B: np.ndarray, C: np.ndarray, c_kinds) -> dict:
    """AUROC(B vs C) restricted to each C kind, plus each kind's count."""
    if c_kinds is None:
        return {}
    kinds = np.asarray([str(k) for k in c_kinds])
    if kinds.size != C.size:
        return {"error": f"{kinds.size} kinds for {C.size} C scores"}
    out = {}
    for kind in sorted(set(kinds.tolist())):
        sel = kinds == kind
        s = np.concatenate([B, C[sel]])
        y = np.concatenate([np.zeros(B.size, bool), np.ones(int(sel.sum()), bool)])
        out[kind] = {"auroc": binary_auroc(s, y), "n_c": int(sel.sum())}
    return out


def _auroc_controlled_by_kind(scores: dict, c_kinds, control_map) -> dict:
    """⭐ **A7-CONTROLLED AUROC, per kind** — each kind scored against the valid
    control the on-disk registry declares for it (R26.1 standing rule, R27.4).

    ⛔ This does **not** replace :func:`_auroc_by_kind`; it ships **beside** it, so
    the contaminated and controlled columns are always readable side by side and
    ⛔ never merged into one.  The contamination is structural, not subtle: every
    ``duplicate_factor`` record carries ``n_factors + 1`` terms against the plain
    pool's ``n_factors``, so a bare term counter scores **1.0000** on that half and
    four ordering features score 0.9167 (``gen-c-set-a7`` §1).  ``a7_redundant``
    erases both; ``out_of_range`` declares ``base`` and is therefore **unchanged by
    construction** — which is the regression check that this wiring did not move a
    rival.
    """
    if c_kinds is None or not control_map:
        return {}
    kinds = np.asarray([str(k) for k in c_kinds])
    C = np.asarray(scores["C"], np.float64)
    if kinds.size != C.size:
        return {"error": f"{kinds.size} kinds for {C.size} C scores"}
    out = {}
    for kind in sorted(set(kinds.tolist())):
        cname = control_map.get(kind)
        if cname is None:
            out[kind] = {"error": ("⛔ no control declared for this kind in "
                                   "query_sets.json — it would be scored uncontrolled")}
            continue
        set_key = CONTROL_BASE_SET if cname == "base" else None
        if set_key is None:
            set_key = next((k for k, v in CONTROL_SET_NAMES.items() if v == cname), None)
        if set_key is None or set_key not in scores:
            out[kind] = {"error": f"control set {cname!r} was not scored by this arm"}
            continue
        neg = np.asarray(scores[set_key], np.float64)
        sel = kinds == kind
        pos = C[sel]
        y = np.concatenate([np.zeros(neg.size, bool), np.ones(pos.size, bool)])
        out[kind] = {
            "auroc": binary_auroc(np.concatenate([neg, pos]), y),
            "control": cname,
            "control_set": set_key,
            "n_c": int(sel.sum()),
            "n_control": int(neg.size),
        }
    return out


def _pooled_controlled(by_kind: dict) -> dict:
    """⛔ COMPANION ONLY (R27.4): the ``n_c``-weighted mean of the per-kind numbers.

    ⛔ It is **not** an AUROC over a pooled score vector, because the two kinds have
    **different controls** — pooling them would put a 7-term and a 6-term negative
    pool in one ranking and re-introduce exactly the giveaway A7 removes.  ⭐ With
    equal ``n_c`` (36/36 at the shipped cell) it coincides with ½(dup + oor), which
    is also what the contaminated pooled AUROC equals to machine precision.
    """
    rows = [v for v in by_kind.values() if "auroc" in v]
    if not rows:
        return {}
    w = np.asarray([float(v["n_c"]) for v in rows])
    a = np.asarray([float(v["auroc"]) for v in rows])
    return {
        "value": float((w * a).sum() / w.sum()),
        "definition": ("n_c-weighted mean of the per-kind CONTROLLED AUROCs — ⛔ a "
                       "COMPANION, never the reported number (R27.4: always by kind)"),
        "kinds": {k: float(v["auroc"]) for k, v in by_kind.items() if "auroc" in v},
    }


def refusal_metrics(scores: dict, valid_quantile: float, c_kinds=None,
                    control_map=None, request_ids=None) -> dict:
    """⭐ Primary = AUROC(B vs C).  C-vs-A and C-vs-(A∪B) alongside (R14.1).

    Positives are always **C** (impossible); a higher refusal score must mean
    "more likely refused", so an AUROC of 0.5 is chance and < 0.5 means the arm
    has the sign backwards.  The threshold comes from **A only** — valid, seen
    data — ⛔ never from B, never from C.

    ⭐ ``control_map`` (``{C kind: registered control set}``, read from disk) adds
    the **A7-controlled** column beside the contaminated one; ``request_ids`` adds
    the **per-request** score table, so the next re-score is a spreadsheet rather
    than a 3-seed re-run.
    """
    A, B, C = (np.asarray(scores[k], np.float64) for k in ("A", "B", "C"))
    MS = np.asarray(scores.get("MS", np.zeros(0)), np.float64)

    def au(neg, pos):
        s = np.concatenate([neg, pos])
        y = np.concatenate([np.zeros(neg.size, bool), np.ones(pos.size, bool)])
        return binary_auroc(s, y)

    thr = knn.threshold_from_valid(A, float(valid_quantile))
    controls = {k: np.asarray(scores[k], np.float64) for k in CONTROL_SETS if k in scores}
    by_kind_controlled = _auroc_controlled_by_kind(scores, c_kinds, control_map)
    # ⭐ The full cross-table: every C kind against EVERY available valid pool, so the
    # contaminated column, the declared control and the dedup sensitivity are read off
    # one object and ⛔ no reader has to trust that they were computed the same way.
    by_kind_per_control = {
        set_key: _auroc_by_kind(np.asarray(scores[set_key], np.float64), C, c_kinds)
        for set_key in (("B",) + CONTROL_SETS) if set_key in scores
    }
    out = {
        "auroc_c_vs_b": au(B, C),  # ⭐ PRIMARY (kill rules (iii) and (iv) bind here)
        "auroc_c_vs_a": au(A, C),
        "auroc_c_vs_a_union_b": au(np.concatenate([A, B]), C),
        "threshold": {
            "value": thr,
            "quantile": float(valid_quantile),
            "set": "A only (valid, seen) — ⛔ no C item and no held-out item participates",
        },
        "refused_fraction": {
            "A": knn.refusal_rates(A, thr)["refused_fraction"],
            "B": knn.refusal_rates(B, thr)["refused_fraction"],
            "MS": knn.refusal_rates(MS, thr)["refused_fraction"] if MS.size else float("nan"),
            "C": knn.refusal_rates(C, thr)["refused_fraction"],
            # ⚠ A7 is VALID: refusing it is a FALSE refusal, exactly as for B.
            **{k: knn.refusal_rates(v, thr)["refused_fraction"]
               for k, v in controls.items() if v.size},
        },
        "score_summary": {
            k: {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size)}
            for k, v in ((("A", A), ("B", B), ("MS", MS), ("C", C))
                         + tuple(controls.items())) if v.size
        },
        # ⭐ Free diagnostic the Hub will want: the two C kinds are very different
        # problems.  ``out_of_range`` walks off the grid (a geometry question);
        # ``duplicate_factor`` stays *on* the manifold between two incompatible
        # values (a consistency question).  An arm can be excellent at one and at
        # chance on the other, and the pooled AUROC hides exactly that.
        "auroc_c_vs_b_by_kind": _auroc_by_kind(B, C, c_kinds),
        # ⭐ A7 (R26.1 / R27.4): the SAME kinds, each against the control its
        # registry declares. ⛔ Reported beside the line above, never merged into it.
        "a7_controlled": {
            "auroc_by_kind": by_kind_controlled,
            "pooled_COMPANION": _pooled_controlled(by_kind_controlled),
            "control_map": dict(control_map or {}),
            "auroc_by_kind_PER_CONTROL": by_kind_per_control,
            "note": ("⭐ per-kind controls read from query_sets.json on disk. "
                     "out_of_range's control is `base` (the ordinary valid pool) ⇒ it "
                     "is A7-clean as shipped and MUST reproduce bit-identically; only "
                     "duplicate_factor moves."),
        },
    }
    if controls:
        out["auroc_c_vs_control_POOLED_COMPANION"] = {
            k: au(v, C) for k, v in controls.items() if v.size
        }
    if request_ids:
        # ⭐ PERSISTED PER REQUEST. ⛔ The absence of exactly this table is what made
        # the first A7 re-score impossible from disk (gen-baselines §0): with only
        # score_summary aggregates saved, a re-score needs a 3-seed re-run.
        out["per_request_scores"] = {
            k: {"ids": [str(i) for i in request_ids[k]],
                "scores": [float(x) for x in np.asarray(scores[k], np.float64)]}
            for k in scores
            if k in request_ids and len(request_ids[k]) == np.asarray(scores[k]).size
        }
        out["per_request_scores"]["_kinds_C"] = [str(k) for k in (c_kinds or [])]
    return out


def _a7_log(ref: dict) -> str:
    """One compact ``by kind: contaminated -> controlled`` string for the run log."""
    by = (ref.get("a7_controlled") or {}).get("auroc_by_kind") or {}
    raw = ref.get("auroc_c_vs_b_by_kind") or {}
    if not by:
        return ""
    bits = []
    for kind in sorted(by):
        c = by[kind]
        if "auroc" not in c:
            continue
        r = raw.get(kind, {}).get("auroc")
        bits.append(f"{kind}: {r:.4f}(vs B) -> {c['auroc']:.4f}(vs {c['control']})"
                    if r is not None else f"{kind}: {c['auroc']:.4f}(vs {c['control']})")
    return ("\n      ⭐ A7 " + " · ".join(bits)) if bits else ""


def _mean_std(vals) -> dict:
    a = np.asarray([float(v) for v in vals], np.float64)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)),
            "values": [float(x) for x in a], "n_seeds": int(a.size)}


# ---------------------------------------------------------------------------
# The kNN arm
# ---------------------------------------------------------------------------


def run_knn_arm(ctx: dict, config: CHLUConfig, seed: int, log: Callable) -> dict:
    """⭐ Byte-matched kNN + its prototype-direct relative, both in-budget.

    The tuning budget is ``|ks| x |metrics| x |normalisers| x |deployments|`` and is
    reported so it can be compared with every other arm's (Head's constraint).
    Selection is on the **validation** split's aggregate probe accuracy only;
    ⛔ C never enters tuning.  The oracle-selected ``max AUROC(B vs C)`` over all
    trials is reported **alongside**, explicitly labelled, so a CLU that beats the
    selected kNN but loses to the oracle kNN cannot be flattered by the selection
    rule.
    """
    g = config.gen_baselines
    budget = int(ctx["byte_target"]["total_bytes"])
    store_only = int(ctx["byte_target"]["store_bytes"])
    Xfit = ctx["X_fit"]
    t0 = time.time()

    stores, normalisers = {}, {}
    for dep in list(g.knn_deployments):
        stores[dep] = knn.build_store(
            Xfit, budget, dep, int(seed), method=str(g.knn_store_method),
            iters=int(g.knn_kmeans_iters),
        )
        led = stores[dep].info["byte_ledger"]
        log(
            f"  [knn] store {dep}: {led['n_entries']} entries, {led['realised_bytes']} B "
            f"of {budget} B (slack {led['slack_bytes']})"
        )
    for nm in list(g.knn_normalisers):
        normalisers[nm] = knn.fit_normaliser(Xfit, nm)

    grid = knn.tuning_grid(g.knn_ks, g.knn_metrics, g.knn_normalisers, g.knn_deployments)
    log(f"  [knn] tuning budget: {len(grid)} trials")

    def read_and_dist(trial, Q_raw):
        store = stores[trial["deployment"]]
        norm = normalisers[trial["normaliser"]]
        Sv = store.values()
        idx, d = knn.knn_search(norm(Sv), norm(Q_raw), int(trial["k"]), trial["metric"])
        return knn.knn_read(Sv, idx), knn.refusal_scores(d, str(g.knn_refusal_reduce))

    trials = []
    for t_i, trial in enumerate(grid):
        r_val, _ = read_and_dist(trial, ctx["E_val"])
        comp = score_composition(
            ctx["decoder"], ctx["probe"], r_val, ctx["lab_val"], ctx["names"],
            ctx["feat_shape"], g.eval_batch,
        )
        _, s_a = read_and_dist(trial, ctx["E_A"])
        _, s_b = read_and_dist(trial, ctx["E_B"])
        _, s_c = read_and_dist(trial, ctx["E_C"])
        # ⛔ Reported for the ORACLE bound only; ⛔ never the selection criterion.
        orc = _oracle_aurocs(s_a, s_b, s_c)
        au = orc["auroc_c_vs_b_ORACLE_ONLY"]
        n_ent = stores[trial["deployment"]].n_entries
        trials.append({
            **trial,
            "n_entries": int(n_ent),
            "bits_per_read": knn.bits_per_read(n_ent, int(trial["k"])),
            "val_aggregate": comp["aggregate"],
            "val_compositional": comp["per_factor"][COMPOSITIONAL_FACTOR],
            **orc,
        })
        log(
            f"  [knn] trial {t_i + 1}/{len(grid)} k={trial['k']} {trial['metric']} "
            f"{trial['normaliser']} {trial['deployment']}: val agg={comp['aggregate']:.4f} "
            f"(AUROC(B vs C)={au:.4f}, oracle-only)"
        )

    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    # ⭐ A SECOND, equally valid-only selection rule, declared (PREREG amendment A3).
    # The pre-registered rule selects on the validation AGGREGATE — but ⚠ four of
    # the six factors sit at ~1.0 and carry no compositional information, so the
    # aggregate is *not* the quantity the reported metric reads.  Selecting on the
    # validation ``shape`` accuracy is the same supervision class (valid rows only,
    # ⛔ no C, ⛔ no held-out row) and is aligned with what is scored.  Both are
    # reported; ⛔ neither is dropped because of how it came out.
    best_comp = trials[int(np.argmax([t["val_compositional"] for t in trials]))]
    oracle_i = int(np.argmax([t["auroc_c_vs_b_ORACLE_ONLY"] for t in trials]))
    log(
        f"  ⭐ [knn] SELECTED (pre-registered, validation aggregate): k={best['k']} "
        f"{best['metric']} {best['normaliser']} {best['deployment']} — "
        f"val agg {best['val_aggregate']:.4f} val shape {best['val_compositional']:.4f}"
    )
    log(
        f"  ⭐ [knn] SELECTED (declared 2nd rule, validation `shape`): k={best_comp['k']} "
        f"{best_comp['metric']} {best_comp['normaliser']} {best_comp['deployment']} — "
        f"val agg {best_comp['val_aggregate']:.4f} val shape {best_comp['val_compositional']:.4f}"
    )

    reads, scores = {}, {}
    for name in REQUEST_SETS:
        reads[name], scores[name] = read_and_dist(best, ctx[f"E_{name}"])
    # ⭐ A7: the control is scored by the SAME rule (distance to the same store) and
    # ⛔ never decoded — a valid control has no composition number and needs none.
    for name in CONTROL_SETS:
        if f"E_{name}" in ctx:
            _, scores[name] = read_and_dist(best, ctx[f"E_{name}"])
    same_sel = all(best[k] == best_comp[k] for k in ("k", "metric", "normaliser", "deployment"))
    reads_c, scores_c = ({}, {}) if same_sel else ({}, {})
    if not same_sel:
        for name in REQUEST_SETS:
            reads_c[name], scores_c[name] = read_and_dist(best_comp, ctx[f"E_{name}"])
        for name in CONTROL_SETS:
            if f"E_{name}" in ctx:
                _, scores_c[name] = read_and_dist(best_comp, ctx[f"E_{name}"])

    # ⭐ Declared sensitivity, ⛔ NOT a 37th tuning trial: the distance reduction
    # over the k neighbours is a free choice the specification does not fix, so
    # both are reported and ``min`` is the declared headline.
    reduce_sensitivity = {}
    for red in ("min", "mean"):
        store = stores[best["deployment"]]
        norm = normalisers[best["normaliser"]]
        Sv = store.values()
        sc = {}
        for name in REQUEST_SETS:
            _, d = knn.knn_search(norm(Sv), norm(ctx[f"E_{name}"]), int(best["k"]), best["metric"])
            sc[name] = knn.refusal_scores(d, red)
        reduce_sensitivity[red] = refusal_metrics(
            sc, float(g.refusal_valid_quantile), ctx["c_kinds"]
        )["auroc_c_vs_b"]
    log(f"  [knn] refusal-reduction sensitivity (declared, not tuned): "
        f"min={reduce_sensitivity['min']:.4f} mean={reduce_sensitivity['mean']:.4f}")
    # ⭐ MEASURED, not assumed: under ``min`` reduction the refusal score is the
    # distance to the FIRST neighbour, so it does not depend on ``k`` at all. The
    # 36 trials therefore contain only |metric| x |normaliser| x |deployment|
    # DISTINCT refusal configurations, and ``k`` is free on that axis. Recorded so
    # nobody reads 36 independent refusal trials off this table.
    by_conf = {}
    for t in trials:
        by_conf.setdefault(
            (t["metric"], t["normaliser"], t["deployment"]), set()
        ).add(round(t["auroc_c_vs_b_ORACLE_ONLY"], 12))
    refusal_axis = {
        "n_distinct_refusal_configurations": len(by_conf),
        "refusal_score_is_invariant_in_k": bool(all(len(v) == 1 for v in by_conf.values())),
        "note": ("⭐ measured: with reduce='min' the refusal score is the distance to "
                 "the first neighbour, so k does not enter it"),
    }
    proto_reads = {name: knn.prototype_direct_read(ctx[f"E_{name}"]) for name in REQUEST_SETS}

    out = {
        "arm": "knn",
        "tuning": {
            "budget": len(grid),
            "axes": {"k": list(g.knn_ks), "metric": list(g.knn_metrics),
                     "normaliser": list(g.knn_normalisers),
                     "deployment": list(g.knn_deployments)},
            "selection_criterion": "validation aggregate probe accuracy (⛔ C never enters)",
            "selected": best,
            "selected_compositional_rule": best_comp,
            "selection_rules_coincide": bool(same_sel),
            "trials": trials,
            "ORACLE_UPPER_BOUND_auroc_c_vs_b": {
                "value": trials[oracle_i]["auroc_c_vs_b_ORACLE_ONLY"],
                # ⭐ kill rule (iv) names C-vs-(A u B); it is carried here too.
                "auroc_c_vs_a_union_b_at_that_trial":
                    trials[oracle_i]["auroc_c_vs_a_union_b_ORACLE_ONLY"],
                "max_auroc_c_vs_a_union_b_over_all_trials": float(max(
                    t["auroc_c_vs_a_union_b_ORACLE_ONLY"] for t in trials)),
                "trial": {k: trials[oracle_i][k] for k in
                          ("k", "metric", "normaliser", "deployment")},
                "note": ("⭐ ORACLE-SELECTED over all trials using C — an explicitly "
                         "labelled UPPER BOUND, ⛔ not an achievable arm result. It "
                         "ships so the selection rule cannot flatter the CLU."),
            },
        },
        "byte_ledger": {
            dep: stores[dep].info["byte_ledger"] for dep in list(g.knn_deployments)
        },
        # ⭐ The superseded R19 figure, kept as a declared row so the correction is
        # visible rather than silent: what the arm WOULD have got had it matched
        # `store_bytes` alone (24 % fewer bytes, hence 24 % fewer entries).
        "byte_ledger_store_only_SUPERSEDED": {
            dep: knn.byte_ledger(
                knn.entries_for_budget(store_only, int(Xfit.shape[1]), dep),
                int(Xfit.shape[1]), dep, store_only,
            )
            for dep in list(g.knn_deployments)
        },
        "prototype_table_ledger": ctx["protos"].byte_ledger(),
        "store_bytes_target": budget,
        "byte_target": ctx["byte_target"],
        "refusal_reduction_sensitivity": reduce_sensitivity,
        "refusal_axis": refusal_axis,
        "variants": {},
        "wall_clock_s": None,
    }

    variant_specs = [
        ("knn", reads, scores, {"bits_per_read": best["bits_per_read"],
                                "selection": "PRE-REGISTERED (validation aggregate)",
                                "config": {k: best[k] for k in
                                           ("k", "metric", "normaliser", "deployment")},
                                "note": "k * log2(n_entries); the request embedding NEVER reaches the decoder"}),
    ]
    if not same_sel:
        variant_specs.append(
            ("knn_compsel", reads_c, scores_c,
             {"bits_per_read": best_comp["bits_per_read"],
              "selection": "DECLARED 2nd rule (validation `shape`) — same supervision class",
              "config": {k: best_comp[k] for k in ("k", "metric", "normaliser", "deployment")},
              "note": "k * log2(n_entries); the request embedding NEVER reaches the decoder"})
        )
    variant_specs.append(
        ("knn_proto", proto_reads, scores,
         {"bits_per_read": float("inf"),
          "selection": "N/A — the prototype read depends on NO tuning axis",
          "note": ("⭐ the byte-matched PROTOTYPE arm: the composed prototype is decoded "
                   "directly, so the whole D-float vector reaches the decoder. In-budget "
                   "(475,136 B < 524,288 B) and reported as its own arm ⛔ never in the "
                   "same column as knn's bits/read.")})
    )
    for vname, vreads, vscores, chan in variant_specs:
        comp_a = score_composition(ctx["decoder"], ctx["probe"], vreads["A"], ctx["lab_A"],
                                   ctx["names"], ctx["feat_shape"], g.eval_batch)
        comp_b = score_composition(ctx["decoder"], ctx["probe"], vreads["B"], ctx["lab_B"],
                                   ctx["names"], ctx["feat_shape"], g.eval_batch)
        disc = corner_discrimination(
            ctx["decoder"], ctx["probe"], vreads["B"], vreads["MS"], ctx["lab_B"], ctx["lab_MS"],
            ctx["shape_idx"], ctx["n_shape_classes"], ctx["feat_shape"], g.eval_batch,
        )
        out["variants"][vname] = {
            "channel": chan,
            "composition_A": comp_a,
            "composition_B": comp_b,
            "in_corner_discrimination": disc,
            # ⭐ Both kNN-family members refuse by the SAME rule (distance to the
            # byte-matched store) — knn_proto's contribution is composition, not a
            # second refusal criterion. Stated so the column is not read twice.
            "refusal": refusal_metrics(vscores, float(g.refusal_valid_quantile),
                                       ctx["c_kinds"], ctx.get("control_map"),
                                       ctx.get("request_ids")),
        }
        log(
            f"  ⭐ [{vname}] in-corner {COMPOSITIONAL_FACTOR} balacc="
            f"{disc['balanced_accuracy']:.4f} (AUROC {disc['auroc']:.4f}) · "
            f"B aggregate={comp_b['aggregate']:.4f} · "
            f"refusal AUROC(B vs C)={out['variants'][vname]['refusal']['auroc_c_vs_b']:.4f}"
            + _a7_log(out["variants"][vname]["refusal"])
        )
    out["wall_clock_s"] = float(time.time() - t0)
    return out


# ---------------------------------------------------------------------------
# The LACE arm
# ---------------------------------------------------------------------------


def _parse_shrinkage(s):
    """``"none"`` / ``"ledoit_wolf"`` pass through; anything else is an intensity."""
    return s if s in ("none", "ledoit_wolf") else float(s)


def run_lace_arm(ctx: dict, config: CHLUConfig, seed: int, log: Callable) -> dict:
    """⭐ LACE per R9: mandatory whitening, both Langevin temperatures, every
    concession declared.

    ⛔ Nothing is vendored (NVIDIA non-commercial licence); every constant carries
    a ``PAPER:`` / ``OFFICIAL IMPLEMENTATION:`` / ``DERIVED:`` tag at its definition in
    ``chlu.eval.rivals.gen_lace``.
    """
    g = config.gen_baselines
    t0 = time.time()
    key = jax.random.PRNGKey(int(seed) * 10_009 + 20_260_911)
    Xfit = ctx["X_fit"]
    cards = ctx["cardinalities"]

    # ---- whitening (⛔ mandatory; a bare 1/2||x||^2 on raw features is refused) --
    whitenings, wchecks = {}, {}
    for sh in list(g.lace_shrinkages):
        w = lace.fit_whitening(Xfit, _parse_shrinkage(sh))
        whitenings[sh] = w
        # ⛔ the whole fit set, NOT a head slice — `whitening_check` subsamples at random
        wchecks[sh] = lace.whitening_check(w, Xfit)
        # ⭐ 0.10, not 0.05, and the reason is structural rather than tolerant: the
        # cache has 71 DEAD (constant post-ReLU) coordinates, so 71 of 2048
        # eigendirections carry no variance and cannot be whitened by anything.
        # They contribute an irreducible 71/2048 = 0.0347 to this statistic before
        # any real violation. MEASURED on the shipped cache: "none" 0.0591,
        # ledoit_wolf 0.1436, 1e-3 0.2322 -- the threshold separates them correctly.
        # ⭐ Now config-backed (``lace_whitening_exact_tol``) because R27.3 makes this
        # threshold a GATE on the configuration, not just a printed diagnostic.
        exact = float(wchecks[sh]["mean_abs_diag_minus_one"]) <= float(g.lace_whitening_exact_tol)
        wchecks[sh]["dead_direction_floor"] = float(
            w.meta["n_eigenvalues_floored"] / max(w.meta["D"], 1)
        )
        wchecks[sh]["prior_identity_near_exact"] = bool(exact)
        log(
            f"  [lace] whitening shrinkage={sh}: lambda={w.meta['shrinkage_intensity']:.4g} "
            f"eig[{w.meta['eigenvalue_min']:.3g}, {w.meta['eigenvalue_max']:.3g}] "
            f"|diag-1|={wchecks[sh]['mean_abs_diag_minus_one']:.4g} "
            f"|offdiag|={wchecks[sh]['mean_abs_offdiag']:.4g}"
            + ("" if exact else "  ⚠ p0(u)=N(0,I) is NOT near-exact at this intensity — "
                                "the ODE drift's licence to omit the prior score is weakened")
        )

    # ---- normalised continuous targets for the fit rows -----------------------
    val_targets_fit = ctx["cont_fit"]
    lab_fit = ctx["lab_fit"]

    # ---- 36 tuning trials on a SHORT schedule --------------------------------
    grid = lace.tuning_grid(g.lace_sigmas, g.lace_forms, [str(s) for s in g.lace_shrinkages])
    log(f"  [lace] tuning budget: {len(grid)} trials (short schedule, {g.lace_tune_epochs} epochs)")
    tune_models = {}
    for sh in list(g.lace_shrinkages):
        key, tk = jax.random.split(key)
        U = whitenings[sh].forward(Xfit)
        m, info, key = lace.train_classifier(
            U, lab_fit, val_targets_fit, cards, tk,
            n_epochs=int(g.lace_tune_epochs), batch=int(g.lace_batch), lr=float(g.lace_lr),
            warmup=int(g.lace_warmup_iters), log=None,
        )
        tune_models[sh] = m
        log(f"  [lace] tuning classifier (shrinkage={sh}): {info['steps']} steps, "
            f"loss {info['loss_first']:.4f} -> {info['loss_final']:.4f}")

    n_tune = min(int(g.lace_tune_n), ctx["req_val"].n)
    req_tune = _slice_dense(ctx["req_val"], n_tune)
    lab_tune = ctx["lab_val"][:n_tune]

    trials = []
    for t_i, trial in enumerate(grid):
        m = tune_models[trial["shrinkage"]]
        key, sk = jax.random.split(key)
        U = lace.sample_ld(
            m, req_tune, sk, std=lace.LACE_SGLD_STD_CORRECTED, lr=lace.LACE_SGLD_LR,
            steps=int(g.lace_tune_ld_steps), form=trial["form"], sigma=float(trial["sigma"]),
            batch=int(g.eval_batch),
        )
        X = whitenings[trial["shrinkage"]].inverse(U)
        comp = score_composition(ctx["decoder"], ctx["probe"], X, lab_tune, ctx["names"],
                                 ctx["feat_shape"], g.eval_batch)
        e_b = lace.energy_of(m, whitenings[trial["shrinkage"]].forward(ctx["X_B"]),
                             ctx["req_B"], trial["form"], float(trial["sigma"]), g.eval_batch)
        # ⚠ C has no image, so its energy is read at a SAMPLE, not at a cached feature.
        key, ck = jax.random.split(key)
        Uc = lace.sample_ld(m, ctx["req_C"], ck, std=lace.LACE_SGLD_STD_CORRECTED,
                            lr=lace.LACE_SGLD_LR, steps=int(g.lace_tune_ld_steps),
                            form=trial["form"], sigma=float(trial["sigma"]),
                            batch=int(g.eval_batch))
        e_c = lace.energy_of(m, Uc, ctx["req_C"], trial["form"], float(trial["sigma"]),
                             g.eval_batch)
        e_a = lace.energy_of(m, whitenings[trial["shrinkage"]].forward(ctx["X_A"]),
                             ctx["req_A"], trial["form"], float(trial["sigma"]), g.eval_batch)
        orc = _oracle_aurocs(e_a, e_b, e_c)
        au = orc["auroc_c_vs_b_ORACLE_ONLY"]
        trials.append({**trial, "val_aggregate": comp["aggregate"],
                       "val_compositional": comp["per_factor"][COMPOSITIONAL_FACTOR],
                       **orc})
        log(
            f"  [lace] trial {t_i + 1}/{len(grid)} sigma={trial['sigma']} {trial['form']} "
            f"shrink={trial['shrinkage']}: val agg={comp['aggregate']:.4f} "
            f"(AUROC(B vs C)={au:.4f}, oracle-only)"
        )

    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    # ⭐ The same declared second rule the kNN arm gets (PREREG amendment A3), so
    # the two arms stay symmetric: ⛔ a selection rule applied to one competitor
    # and not the other is a thumb on the scale.
    best_comp = trials[int(np.argmax([t["val_compositional"] for t in trials]))]
    same_sel = all(best[k] == best_comp[k] for k in ("sigma", "form", "shrinkage"))
    oracle_i = int(np.argmax([t["auroc_c_vs_b_ORACLE_ONLY"] for t in trials]))
    log(
        f"  ⭐ [lace] SELECTED (pre-registered, validation aggregate): sigma={best['sigma']} "
        f"form={best['form']} shrinkage={best['shrinkage']} — val agg {best['val_aggregate']:.4f}"
    )
    log(
        f"  ⭐ [lace] SELECTED (declared 2nd rule, validation `shape`): sigma={best_comp['sigma']} "
        f"form={best_comp['form']} shrinkage={best_comp['shrinkage']} — "
        f"val shape {best_comp['val_compositional']:.4f}"
        + ("  (identical to the pre-registered choice)" if same_sel else "")
    )

    # ---- ⭐ R27.3: the whitening-exactness GATE on the SELECTED configuration ----
    # "A rival weakened by our own selection procedure is our fault." If the
    # pre-registered rule lands on a non-exact whitening, p0(u) = N(0, I) fails and
    # the ODE drift's licence to omit the prior score is not ours to grant — so the
    # NEAREST EXACT configuration in the same grid is run too, and ⛔ BOTH are
    # reported. ⛔ The pre-registered leg is never dropped and its RNG stream is
    # untouched (the fallback draws from its own key), so it stays bit-identical.
    def _is_exact(sh) -> bool:
        return bool(wchecks[str(sh)]["prior_identity_near_exact"])

    wexact_sel, wexact_reason = None, None
    gate_fired = not _is_exact(best["shrinkage"])
    if gate_fired and bool(g.lace_whitening_exactness_fallback):
        exact_trials = [t for t in trials if _is_exact(t["shrinkage"])]
        if not exact_trials:
            wexact_reason = ("⛔ NO exact whitening exists anywhere in the grid — the "
                             "fallback cannot be run and the gate is reported unrepaired")
        else:
            near = [t for t in exact_trials
                    if t["sigma"] == best["sigma"] and t["form"] == best["form"]]
            pool = near or exact_trials
            cand = pool[int(np.argmax([t["val_aggregate"] for t in pool]))]
            already = [c[1] for c in ([("", best)] + ([("", best_comp)] if not same_sel else []))]
            if any(all(cand[k] == a[k] for k in ("sigma", "form", "shrinkage"))
                   for a in already):
                wexact_reason = ("the nearest exact configuration is already being run "
                                 "as another leg — ⛔ not run twice")
            else:
                wexact_sel = cand
                wexact_reason = ("nearest exact in the grid: same sigma/form, shrinkage "
                                 "swapped" if near else
                                 "no exact trial at this sigma/form; best exact by the "
                                 "same validation-aggregate rule")
    if gate_fired:
        log(f"  ⚠ [lace] R27.3 GATE FIRED: selected shrinkage={best['shrinkage']} has "
            f"|diag-1|={wchecks[str(best['shrinkage'])]['mean_abs_diag_minus_one']:.4g} "
            f"> {float(g.lace_whitening_exact_tol)} ⇒ p0(u)=N(0,I) is NOT near-exact. "
            f"Fallback: {wexact_reason}")
    else:
        log(f"  ⭐ [lace] R27.3 gate PASSES: selected shrinkage={best['shrinkage']} is "
            "whitening-exact; ⛔ no fallback needed")

    # ---- retrain the SELECTED configuration at the shipped 50 epochs ----------
    wh = whitenings[best["shrinkage"]]
    key, fk = jax.random.split(key)
    model, fit_info, key = lace.train_classifier(
        wh.forward(Xfit), lab_fit, val_targets_fit, cards, fk,
        n_epochs=int(g.lace_n_epochs), batch=int(g.lace_batch), lr=float(g.lace_lr),
        warmup=int(g.lace_warmup_iters), log=log,
    )
    clf_acc = lace.classifier_accuracy(model, wh.forward(ctx["X_val"]), ctx["lab_val"])
    log(f"  [lace] classifier (its own supervision): val per-factor mean {clf_acc['mean']:.4f}")

    # ---- every sampler; ⭐ LACE's best is what gets reported (R9.4) -----------
    configs = [("", best, model, wh)]
    fit_info_compsel = None
    if not same_sel:
        wh_c = whitenings[best_comp["shrinkage"]]
        if best_comp["shrinkage"] == best["shrinkage"]:
            model_c = model  # same whitening ⇒ the same classifier, no refit
        else:
            key, fk2 = jax.random.split(key)
            model_c, fit_info_compsel, key = lace.train_classifier(
                wh_c.forward(Xfit), lab_fit, val_targets_fit, cards, fk2,
                n_epochs=int(g.lace_n_epochs), batch=int(g.lace_batch), lr=float(g.lace_lr),
                warmup=int(g.lace_warmup_iters), log=None,
            )
        configs.append(("_compsel", best_comp, model_c, wh_c))
        log(f"  [lace] 2nd-rule config needs its own whitening/classifier: "
            f"shrinkage={best_comp['shrinkage']}"
            if best_comp["shrinkage"] != best["shrinkage"] else
            "  [lace] 2nd-rule config shares the pre-registered whitening ⇒ no refit")

    # ⭐ R27.3's fallback leg, on its OWN key stream (⛔ the pre-registered leg's
    # numbers must not move because we added a leg beside it).
    fit_info_wexact = None
    if wexact_sel is not None:
        wh_w = whitenings[wexact_sel["shrinkage"]]
        fb_key = jax.random.PRNGKey(int(seed) * 10_009 + 20_260_911 + 7_717)
        fb_key, wk = jax.random.split(fb_key)
        model_w, fit_info_wexact, _ = lace.train_classifier(
            wh_w.forward(Xfit), lab_fit, val_targets_fit, cards, wk,
            n_epochs=int(g.lace_n_epochs), batch=int(g.lace_batch), lr=float(g.lace_lr),
            warmup=int(g.lace_warmup_iters), log=None,
        )
        configs.append(("_wexact", wexact_sel, model_w, wh_w))
        log(f"  ⭐ [lace] R27.3 FALLBACK leg: sigma={wexact_sel['sigma']} "
            f"form={wexact_sel['form']} shrinkage={wexact_sel['shrinkage']} "
            f"(|diag-1|="
            f"{wchecks[str(wexact_sel['shrinkage'])]['mean_abs_diag_minus_one']:.4g}) — "
            "⛔ reported BESIDE the pre-registered leg, never instead of it")

    samplers = {}
    # ⭐ A7's sampling draws from an INDEPENDENT, deterministic stream: folding it into
    # ``key`` would re-order every subsequent split and silently move every historical
    # LACE number. ⛔ The whole point of the control is that nothing else moves.
    a7_key = jax.random.PRNGKey(int(seed) * 10_009 + 20_260_911 + 4_242)
    a7_draw = 0
    for suffix, sel, model_s, wh_s in configs:
        for sname_base in list(g.lace_samplers):
            sname = sname_base + suffix
            spec = lace.SAMPLERS[sname_base]
            key, sk = jax.random.split(key)
            U, E = {}, {}
            for rs in REQUEST_SETS:
                key, rk = jax.random.split(key)
                req = ctx[f"req_{rs}"]
                if spec["kind"] == "ode":
                    U[rs] = lace.sample_ode(model_s, req, rk, form=sel["form"],
                                            sigma=float(sel["sigma"]), batch=int(g.eval_batch))
                else:
                    U[rs] = lace.sample_ld(
                        model_s, req, rk, std=float(spec["std"]), lr=lace.LACE_SGLD_LR,
                        steps=int(g.lace_ld_steps), form=sel["form"],
                        sigma=float(sel["sigma"]), batch=int(g.eval_batch),
                    )
                E[rs] = lace.energy_of(model_s, U[rs], req, sel["form"], float(sel["sigma"]),
                                       g.eval_batch)
            for rs in CONTROL_SETS:
                req = ctx.get(f"req_{rs}")
                if req is None:
                    continue
                ck = jax.random.fold_in(a7_key, a7_draw)
                a7_draw += 1
                if spec["kind"] == "ode":
                    Uc = lace.sample_ode(model_s, req, ck, form=sel["form"],
                                         sigma=float(sel["sigma"]), batch=int(g.eval_batch))
                else:
                    Uc = lace.sample_ld(
                        model_s, req, ck, std=float(spec["std"]), lr=lace.LACE_SGLD_LR,
                        steps=int(g.lace_ld_steps), form=sel["form"],
                        sigma=float(sel["sigma"]), batch=int(g.eval_batch),
                    )
                # ⚠ Like C, a control request is scored at a SAMPLE. A7 records are
                # satisfiable, so a cached feature exists for their row — but reading it
                # would score the control by a different rule than C, which is the one
                # asymmetry that would make the controlled AUROC meaningless.
                E[rs] = lace.energy_of(model_s, Uc, req, sel["form"], float(sel["sigma"]),
                                       g.eval_batch)
            X = {rs: wh_s.inverse(U[rs]) for rs in REQUEST_SETS}
            comp_a = score_composition(ctx["decoder"], ctx["probe"], X["A"], ctx["lab_A"],
                                       ctx["names"], ctx["feat_shape"], g.eval_batch)
            comp_b = score_composition(ctx["decoder"], ctx["probe"], X["B"], ctx["lab_B"],
                                       ctx["names"], ctx["feat_shape"], g.eval_batch)
            disc = corner_discrimination(
                ctx["decoder"], ctx["probe"], X["B"], X["MS"], ctx["lab_B"], ctx["lab_MS"],
                ctx["shape_idx"], ctx["n_shape_classes"], ctx["feat_shape"], g.eval_batch,
            )
            ref = refusal_metrics({k: E[k] for k in E},
                                  float(g.refusal_valid_quantile), ctx["c_kinds"],
                                  ctx.get("control_map"), ctx.get("request_ids"))
            samplers[sname] = {
                "spec": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in spec.items()},
                "composition_A": comp_a, "composition_B": comp_b,
                "in_corner_discrimination": disc, "refusal": ref,
                "sample_norm_mean": float(np.mean(np.linalg.norm(U["B"], axis=1))),
            }
            log(
                f"  ⭐ [lace/{sname}] in-corner {COMPOSITIONAL_FACTOR} balacc="
                f"{disc['balanced_accuracy']:.4f} · B aggregate={comp_b['aggregate']:.4f} · "
                f"refusal AUROC(B vs C)={ref['auroc_c_vs_b']:.4f}"
                + _a7_log(ref)
            )

    best_sampler_comp = max(
        samplers, key=lambda s: samplers[s]["in_corner_discrimination"]["balanced_accuracy"]
    )
    best_ref = max(samplers, key=lambda s: samplers[s]["refusal"]["auroc_c_vs_b"])

    def _leg(suffix: str) -> dict:
        """Best-of-leg on each axis, so the two R27.3 legs are comparable at a glance."""
        names = [n for n in samplers
                 if (n.endswith(suffix) if suffix else
                     not (n.endswith("_compsel") or n.endswith("_wexact")))]
        if not names:
            return {}
        bc = max(names, key=lambda n:
                 samplers[n]["in_corner_discrimination"]["balanced_accuracy"])
        br = max(names, key=lambda n: samplers[n]["refusal"]["auroc_c_vs_b"])

        def _ctrl(n, kind):
            k = (samplers[n]["refusal"].get("a7_controlled") or {}).get("auroc_by_kind") or {}
            return k.get(kind, {}).get("auroc")
        bd = max(names, key=lambda n: (_ctrl(n, "duplicate_factor") or -1.0))
        return {
            "samplers": sorted(names),
            "best_composition": {"sampler": bc,
                                 "in_corner_balacc":
                                 samplers[bc]["in_corner_discrimination"]["balanced_accuracy"]},
            "best_refusal_pooled_CONTAMINATED": {
                "sampler": br, "auroc_c_vs_b": samplers[br]["refusal"]["auroc_c_vs_b"]},
            "best_refusal_duplicate_factor_A7_CONTROLLED": {
                "sampler": bd, "auroc": _ctrl(bd, "duplicate_factor")},
            "out_of_range_A7_CLEAN": {
                n: _ctrl(n, "out_of_range") for n in sorted(names)},
        }

    legs = {"pre_registered": _leg(""), "compsel": _leg("_compsel"),
            "wexact_R27_3": _leg("_wexact")}
    legs = {k: v for k, v in legs.items() if v}

    out = {
        "arm": "lace",
        "licence": "NVIDIA non-commercial (NVlabs/LACE@dcfd4a5) — ⛔ nothing vendored; "
                   "clean re-implementation from the pinned spec",
        "DO_NOT_USE_AS_A_BAR": {
            "cifar10_fid": 6.63, "cifar10_acc": 0.972, "ffhq_glasses_fid": 20.93,
            "unseen_combination": [0.94, 0.98],
            "note": "StyleGAN up to 1024^2, their splits, their labels. ⛔ NOT a target.",
        },
        "whitening": {
            sh: {**whitenings[sh].meta, "check": wchecks[sh]} for sh in list(g.lace_shrinkages)
        },
        "concessions_declared": [
            "dense factor labels over the train split (our store gets none) — R9.3",
            "a validation split for tuning the OR/NOT constants — R9.3; ⚠ every A/B/C "
            "request is a pure conjunction, so OR and NOT are never exercised by a "
            "reported number",
            "both Langevin temperatures run and LACE's best reported — R9.4",
        ],
        "classifier": {**fit_info, "own_val_accuracy": clf_acc},
        "classifier_compsel": fit_info_compsel,
        "classifier_wexact": fit_info_wexact,
        # ⭐ R27.3, in one readable block: did the gate fire, what was run because it
        # did, and which leg is stronger on each axis. ⛔ "The stronger" is quoted by
        # the report; ⛔ neither leg is dropped here.
        "whitening_exactness_gate": {
            "tolerance": float(g.lace_whitening_exact_tol),
            "statistic": "mean |diag(cov(u)) - 1| on the fit rows",
            "selected_shrinkage": str(best["shrinkage"]),
            "selected_mean_abs_diag_minus_one":
                float(wchecks[str(best["shrinkage"])]["mean_abs_diag_minus_one"]),
            "selected_is_exact": bool(_is_exact(best["shrinkage"])),
            "fired": bool(gate_fired),
            "fallback_enabled": bool(g.lace_whitening_exactness_fallback),
            "fallback_selected": wexact_sel,
            "fallback_reason": wexact_reason,
            "per_shrinkage": {
                sh: {"mean_abs_diag_minus_one":
                     float(wchecks[sh]["mean_abs_diag_minus_one"]),
                     "prior_identity_near_exact":
                     bool(wchecks[sh]["prior_identity_near_exact"])}
                for sh in list(g.lace_shrinkages)
            },
            "ruling": ("R27.3 — a rival weakened by our own selection procedure is our "
                       "fault. A non-exact whitening means p0(u)=N(0,I) fails, and that "
                       "identity is the ONLY thing licensing the ODE drift to omit the "
                       "prior score (R9.1). ⛔ Both legs are reported; the stronger is "
                       "quoted; ⛔ never-weaken-a-competitor applies to our own code."),
        },
        "tuning": {
            "budget": len(grid),
            "axes": {"sigma": list(g.lace_sigmas), "form": list(g.lace_forms),
                     "shrinkage": [str(s) for s in g.lace_shrinkages]},
            "selection_criterion": "validation aggregate probe accuracy (⛔ C never enters)",
            "selected": best,
            "selected_compositional_rule": best_comp,
            "selection_rules_coincide": bool(same_sel),
            "trials": trials,
            "ORACLE_UPPER_BOUND_auroc_c_vs_b": {
                "value": trials[oracle_i]["auroc_c_vs_b_ORACLE_ONLY"],
                # ⭐ kill rule (iv) names C-vs-(A u B); it is carried here too.
                "auroc_c_vs_a_union_b_at_that_trial":
                    trials[oracle_i]["auroc_c_vs_a_union_b_ORACLE_ONLY"],
                "max_auroc_c_vs_a_union_b_over_all_trials": float(max(
                    t["auroc_c_vs_a_union_b_ORACLE_ONLY"] for t in trials)),
                "trial": {k: trials[oracle_i][k] for k in ("sigma", "form", "shrinkage")},
                "note": "⭐ ORACLE-SELECTED over all trials using C — an UPPER BOUND, ⛔ not achievable.",
            },
        },
        "samplers": samplers,
        "legs": legs,
        "best_sampler_composition": best_sampler_comp,
        "best_sampler_refusal": best_ref,
        "wall_clock_s": float(time.time() - t0),
    }
    if bool(g.lace_tune_or_not):
        out["or_not_calibration"] = calibrate_or_not(model, ctx, g, best, log)
    return out


def _slice_dense(req, n: int):
    from ..core.gen_request import DenseRequests

    n = int(n)
    return DenseRequests(
        req.cls_targets[:n], req.val_targets[:n], req.extra_factor[:n],
        req.extra_cls[:n], req.extra_val[:n], req.n_factors,
    )


def calibrate_or_not(model, ctx: dict, g, best: dict, log: Callable) -> dict:
    """R9.3's granted concession, discharged honestly.

    ⭐ **Declared:** every A/B/C request in this protocol is a pure conjunction, so
    OR and NOT are **never exercised by a reported number**.  Freezing ``cof`` and
    the ``0.1`` in ``α`` at the published 20/0.1 would nonetheless be the easiest way
    to hobble OR/NOT if a later cell used them, so the constants are calibrated
    here on the **validation** split, with their own budget, and ⛔ are **not**
    folded into the arm's 36 trials.

    The calibration objective is a proxy no reported number depends on: the OR of
    two validation requests should score a **lower** composed energy than the
    worse of its two operands, and NOT should invert the ordering.
    """
    n = min(64, ctx["req_val"].n)
    req = _slice_dense(ctx["req_val"], n)
    U = ctx["U_val_for_or_not"]
    if U is None or U.shape[0] < n:
        return {"skipped": "no whitened validation features available"}
    e = lace.energy_of(model, U[:n], req, best["form"], float(best["sigma"]), int(g.eval_batch))
    e_shift = np.roll(e, 1)
    rows = []
    for cof in list(g.lace_or_cofs):
        for num in list(g.lace_not_nums):
            a1, a2 = -e, -e_shift
            or_neg = np.log(np.exp(a1) * float(cof) + np.exp(a2))
            alpha = np.clip(float(num) / np.maximum(np.abs(a2), 1e-12), 0.0, 1.0)
            not_neg = -alpha * a2
            rows.append({
                "cof": float(cof), "not_num": float(num),
                "or_beats_worse_operand_fraction": float(np.mean(-or_neg <= np.maximum(e, e_shift))),
                "not_inverts_fraction": float(np.mean(np.sign(not_neg) != np.sign(a2))),
            })
    best_row = max(rows, key=lambda r: (r["or_beats_worse_operand_fraction"], r["not_inverts_fraction"]))
    log(f"  [lace] OR/NOT calibration (⛔ not in the 36): cof={best_row['cof']} "
        f"not_num={best_row['not_num']}")
    return {
        "budget": len(rows),
        "published_constants": {"cof": lace.LACE_OR_COF, "not_num": lace.LACE_NOT_NUM},
        "selected": best_row,
        "grid": rows,
        "note": ("⭐ DECLARED and SEPARATE: no A/B/C request uses OR or NOT, so these "
                 "constants enter no reported number. OR is non-commutative and there "
                 "is a hard 10-operand limit (gen-lace-spec F5)."),
    }


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def run_seed(dataset: str, pretext: str, seed: int, out_dir: Path, config: CHLUConfig,
             arm: str = "all", log: Callable = print) -> dict:
    g = config.gen_baselines
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_seed = time.time()
    condition = condition_of(config)
    log(f"== seed {seed} · {dataset}/{pretext}/{condition} · out {out_dir}")

    byte_target = resolve_store_bytes(config, dataset, pretext, log)

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    shape_idx = names.index(COMPOSITIONAL_FACTOR)
    rows = select_baseline_rows(split, config, seed, log=log)
    _rsp = rows["row_split"]
    _bal = _rsp["balance"]
    log(f"  ⭐ row split: mode={_rsp['mode']} "
        f"(rule: {_rsp.get('rule')}) — val realises "
        + " · ".join(f"{k} {v['val_n_values_realised']}/{v['cardinality']}"
                     for k, v in _bal["factors"].items())
        + f" · max per-value count ratio {_bal['max_val_count_ratio']:.3f}")

    leak = assert_no_holdout_leak(split, np.concatenate([rows["fit"], rows["val"]]))
    log(f"  ⭐ no-leak: {leak['n_checked']} fit/val rows checked, "
        f"{leak['n_holdout_leaked']} leaked (of {leak['n_holdout_total']} held-out)")

    bdir = bridge_dir(config, dataset, pretext, seed)
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    probe = load_model(bdir / "probe.pkl")
    if param_hash(encoder) != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    if param_hash(probe) != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match the bridge manifest hash")
    decoder, dec_prov = load_shared_decoder(config, bdir, bman, seed, log)

    crows, feats, cman = load_feature_cache(bdir, verify=bool(g.verify_cache_checksum))
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    if cman["encoder_param_hash"] != bman["encoder_param_hash"]:
        raise AssertionError("the feature cache was written by a different encoder")
    log(f"  cache: {cman['n_rows']} rows (scope={cman['scope']}), D={cman['D']}, {cman['dtype']}")
    cfg_for_gather = config
    cfg_for_gather.gen_store.encode_chunk = int(g.encode_chunk)
    reenc = _assert_cache_reencodes(dataset, cfg_for_gather, encoder, crows, feats, cman, log=log)

    ms_rows = matched_seen_rows(dataset, split, values, rows["store_train"])
    log(f"  rows: fit={rows['fit'].size} val={rows['val'].size} A={rows['eval_a'].size} "
        f"B={rows['eval_b'].size} matched_seen={ms_rows.size}")

    def gather(r):
        return gather_features(dataset, cfg_for_gather, encoder, crows, feats, cman, r, log=log)

    X = {"fit": gather(rows["fit"]), "val": gather(rows["val"]),
         "A": gather(rows["eval_a"]), "B": gather(rows["eval_b"]), "MS": gather(ms_rows)}
    lab = {"fit": factor_labels(grid, rows["fit"]), "val": factor_labels(grid, rows["val"]),
           "A": factor_labels(grid, rows["eval_a"]), "B": factor_labels(grid, rows["eval_b"]),
           "MS": factor_labels(grid, ms_rows)}

    # ---- ⭐ ceiling 1 of 2: the DECODER ceiling, re-measured from REAL features --
    ceil_comp = score_composition(decoder, probe, X["A"], lab["A"], names, feat_shape, g.eval_batch)
    ceil_disc = corner_discrimination(
        decoder, probe, X["B"], X["MS"], lab["B"], lab["MS"], shape_idx,
        int(cards[shape_idx]), feat_shape, g.eval_batch,
    )
    log(
        f"  ⭐ DECODER ceiling (real features): A aggregate={ceil_comp['aggregate']:.4f} · "
        f"in-corner {COMPOSITIONAL_FACTOR} balacc={ceil_disc['balanced_accuracy']:.4f} "
        f"(AUROC {ceil_disc['auroc']:.4f})"
    )
    log(f"  ⭐ STORE ceiling (R22/R23, quoted): {g.store_ceiling_balacc:.4f} balanced accuracy")

    # ---- requests (⛔ A/B single-sourced from the split, C read from disk) ------
    req_rows = {"A": rows["eval_a"], "B": rows["eval_b"], "MS": ms_rows, "val": rows["val"]}
    requests = {k: requests_from_rows(grid, v) for k, v in req_rows.items()}
    root = Path(config.gen_data.data_root) if config.gen_data.data_root else default_gen_data_root()
    c_path = root / dataset / "c_invalid.json"
    requests["C"] = load_c_requests(c_path)
    log(f"  C set: {len(requests['C'])} records read VERBATIM from {c_path} (⛔ never regenerated)")

    # ---- ⭐ A7: the per-kind VALID CONTROLS, discovered from the on-disk registry ----
    control_map, control_prov = {}, {"scored": False}
    if bool(g.score_registered_controls):
        reg_path = root / dataset / str(g.query_set_registry_file)
        if not reg_path.exists():
            msg = (f"⛔ no query-set registry at {reg_path}: the C kinds' structural "
                   "controls are undeclared, so C would be scored against the "
                   "uncontrolled valid pool and a term counter would score 1.0000 on "
                   "the duplicate_factor half (R26.1). Emit it with `chlu data "
                   "<dataset>` (gen_data.emit_a7), or set "
                   "gen_baselines.require_registered_controls=False to run the "
                   "CONTAMINATED column only — ⛔ and say so in the report.")
            if bool(g.require_registered_controls):
                raise FileNotFoundError(msg)
            log("  ⚠ " + msg)
        else:
            registry = load_query_set_registry(reg_path)
            control_map = control_map_of(registry)
            loaded = load_registered_control_requests(
                root / dataset, registry, sorted(set(control_map.values()))
            )
            for set_key, reg_name in CONTROL_SET_NAMES.items():
                if reg_name in loaded:
                    requests[set_key] = loaded[reg_name]
            # ⭐ the declared dedup sensitivity (see ``deduplicate_terms``)
            if bool(g.control_dedup_sensitivity):
                for d_key, src in CONTROL_DEDUP_OF.items():
                    if src in requests:
                        requests[d_key] = deduplicate_terms(requests[src])
            held = set(int(r) for r in np.asarray(split.test_rows, np.int64).tolist())
            spent = {k: sum(1 for r in requests[k] if r.get("row") in held)
                     for k in CONTROL_SET_NAMES if k in requests}
            if any(v for v in spent.values()):
                raise AssertionError(
                    f"⛔ a valid control spends held-out rows ({spent}) — the control "
                    "would leak B into the negative pool it controls"
                )
            control_prov = {
                "scored": True,
                "registry": str(reg_path),
                "control_map": control_map,
                "sets": {
                    set_key: {
                        "registered_as": CONTROL_SET_NAMES[set_key],
                        "file": str(root / dataset
                                    / registry["sets"][CONTROL_SET_NAMES[set_key]]["file"]),
                        "sha256_registered":
                            registry["sets"][CONTROL_SET_NAMES[set_key]].get("sha256"),
                        "n_records": len(requests[set_key]),
                        "n_terms": len(requests[set_key][0]["request"]),
                        "held_out_rows_spent": spent.get(set_key, 0),
                    }
                    for set_key in CONTROL_SET_NAMES if set_key in requests
                },
                "dedup_sensitivity_sets": {
                    d: {"derived_from": src, "n_records": len(requests[d]),
                        "n_terms": len(requests[d][0]["request"]),
                        "role": ("⭐ DECLARED SENSITIVITY — isolates R8's double-counted "
                                 "redundant term; ⛔ never a registered control")}
                    for d, src in CONTROL_DEDUP_OF.items() if d in requests
                },
                "base_is_instantiated_by": CONTROL_BASE_SET,
                "note": ("⭐ out_of_range declares `base` (the ordinary valid pool) so it "
                         "is A7-CLEAN AS SHIPPED and must reproduce bit-identically; ⛔ "
                         "only the duplicate_factor half and the pooled companion move."),
            }
            log(f"  ⭐ A7 controls: {control_map} — "
                + ", ".join(f"{k}={len(requests[k])} records, "
                            f"{len(requests[k][0]['request'])} terms"
                            for k in CONTROL_SETS if k in requests))

    protos = FactorPrototypes.fit(grid, X["fit"], rows["fit"])
    offgrid = protos.offgrid_report(requests["C"])
    if offgrid["n_offgrid_terms"] == 0:
        raise AssertionError(
            "⛔ not one C term is off-grid — the out_of_range half of C has been "
            "clipped somewhere and the arm would be scoring satisfiable requests"
        )
    log(
        f"  ⭐ R8 embedding: {protos.n_prototypes} prototypes ({protos.n_bytes} B); C has "
        f"{offgrid['n_offgrid_terms']}/{offgrid['n_terms']} off-grid terms, max "
        f"{offgrid['max_distance_outside_range']:.4g} outside the range"
    )

    dense = {k: dense_request_arrays(grid, v) for k, v in requests.items()}
    # LACE normalises its continuous attributes; an off-grid request value then
    # lands outside [0, 1] in proportion to how far off-grid it is (declared in
    # ``chlu.core.gen_request.normalise_value`` and used identically there).
    cont_cols = []
    for k in range(grid.n_factors):
        vals_k = np.asarray(grid.values[k], np.float64)
        span = max(float(vals_k[-1] - vals_k[0]), 1e-12)
        cont_cols.append((vals_k[lab["fit"][:, k]] - vals_k[0]) / span)
    cont_fit = np.stack(cont_cols, axis=1).astype(np.float32)

    ctx = {
        "names": names, "cardinalities": cards, "feat_shape": feat_shape,
        "shape_idx": shape_idx, "n_shape_classes": int(cards[shape_idx]),
        "decoder": decoder, "probe": probe, "protos": protos,
        "byte_target": byte_target,
        "X_fit": X["fit"], "X_val": X["val"], "X_A": X["A"], "X_B": X["B"], "X_MS": X["MS"],
        "lab_fit": lab["fit"], "lab_val": lab["val"], "lab_A": lab["A"],
        "lab_B": lab["B"], "lab_MS": lab["MS"],
        "cont_fit": cont_fit,
        "E_A": protos.embed_many(requests["A"]), "E_B": protos.embed_many(requests["B"]),
        "E_MS": protos.embed_many(requests["MS"]), "E_C": protos.embed_many(requests["C"]),
        "E_val": protos.embed_many(requests["val"]),
        "req_A": dense["A"], "req_B": dense["B"], "req_MS": dense["MS"],
        "req_C": dense["C"], "req_val": dense["val"],
        "c_kinds": [r.get("kind") for r in requests["C"]],
        "control_map": control_map,
        # ⭐ ids travel with the scores so the persisted per-request table is joinable
        # back to the request records without re-deriving an order.
        "request_ids": {k: [str(r["id"]) for r in requests[k]]
                        for k in requests if k != "val"},
        "U_val_for_or_not": None,
    }
    for set_key in CONTROL_SETS:
        if set_key in requests:
            ctx[f"E_{set_key}"] = protos.embed_many(requests[set_key])
            ctx[f"req_{set_key}"] = dense[set_key]

    arms = {}
    want = ("knn", "lace") if arm in ("all", None) else (arm,)
    if "knn" in want and "knn" in list(g.arms):
        arms["knn"] = run_knn_arm(ctx, config, seed, log)
        _write_json({"partial": True, "arms": arms}, out_dir / "baselines_partial.json")
    if "lace" in want and "lace" in list(g.arms):
        # whitened validation features, for the declared OR/NOT calibration only
        try:
            ctx["U_val_for_or_not"] = lace.fit_whitening(
                X["fit"], _parse_shrinkage(str(g.lace_shrinkages[0]))
            ).forward(X["val"])
        except Exception:  # pragma: no cover - calibration is a side diagnostic
            ctx["U_val_for_or_not"] = None
        arms["lace"] = run_lace_arm(ctx, config, seed, log)

    manifest = {
        "schema": "chlu-gen-baselines/1",
        "dataset": dataset, "pretext": pretext, "condition": condition, "seed": int(seed),
        "bridge_dir": str(bdir),
        "provenance": {
            "encoder_param_hash": bman["encoder_param_hash"],
            "probe_param_hash": bman["probe_param_hash"],
            "decoder": dec_prov,
            "feature_cache_sha256": cman["features_sha256"],
            "feature_cache_scope": cman["scope"],
            "feature_cache_n_rows": int(cman["n_rows"]),
            "cache_reencode_check": reenc,
            "c_invalid_path": str(c_path),
            "byte_target": byte_target,
            # ⭐ A7 (R26.1/R27.4): which valid pool controls which C kind, read from
            # the registry on disk — ⛔ never a list hardcoded in this experiment.
            "registered_controls": control_prov,
        },
        "no_leak": leak,
        "rows": {k: int(np.asarray(v).size) for k, v in
                 {**req_rows, "fit": rows["fit"]}.items()},
        # ⭐ How fit/val were cut, and the realised per-factor balance of the cut.
        # ⛔ A tuned configuration whose manifest cannot show this is not identified.
        "row_split": rows["row_split"],
        "request_embedding": {
            "rule": ("R8 — composed per-factor prototypes: "
                     "x_hat = mu + sum_terms (mu_{k,v} - mu); off-grid values are "
                     "LINEARLY EXTRAPOLATED in value space, ⛔ never clipped onto the grid"),
            "prototype_ledger": protos.byte_ledger(),
            "offgrid_report": offgrid,
            "n_rows_fit": protos.n_rows_fit,
        },
        "ceilings": {
            "decoder_ceiling": {
                "composition_A": ceil_comp, "in_corner_discrimination": ceil_disc,
                "note": "re-measured in-run from REAL cached features through THE frozen decoder",
            },
            "store_ceiling_balanced_accuracy": float(g.store_ceiling_balacc),
            "store_ceiling_note": ("R22/R23: the in-corner store read, 0.6866 vs the 0.9414 "
                                   "real-feature ceiling. ⛔ A number quoted against only one "
                                   "of the two ceilings is incomplete."),
        },
        "arms": arms,
        "wall_clock_s": float(time.time() - t_seed),
        "NO_KILL_RULE_VERDICT": ("⛔ This spoke reports arm numbers only; gen-cell-run "
                                 "renders verdicts against a filed prereg."),
    }
    # ⭐ The per-request refusal scores leave the manifest and land in their own file:
    # ⛔ their ABSENCE is what made the first A7 re-score impossible (gen-baselines §0).
    # Written BEFORE the manifest, so a manifest on disk always has its scores beside it.
    if bool(g.persist_per_request_refusal_scores):
        per_request = _extract_per_request_scores(arms)
        _write_json(
            {
                "schema": "chlu-gen-baselines-refusal-scores/1",
                "dataset": dataset, "pretext": pretext, "condition": condition,
                "seed": int(seed),
                "byte_target": byte_target,
                "control_map": control_map,
                "c_invalid_path": str(c_path),
                "registered_controls": control_prov,
                "sets": {k: len(v) for k, v in requests.items()},
                "scores": per_request,
                "note": ("⭐ one row per REQUEST per arm-variant. A re-score against a "
                         "new control is a join on `ids`, ⛔ not a re-run."),
            },
            out_dir / "refusal_scores.json",
        )
        log(f"  ⭐ per-request refusal scores persisted: "
            f"{sum(len(v) for v in per_request.values())} variants -> "
            f"{out_dir / 'refusal_scores.json'}")
    _write_json(manifest, out_dir / "baselines_manifest.json")
    (out_dir / "baselines_partial.json").unlink(missing_ok=True)
    if g.make_plots:
        try:
            _plot_seed(manifest, out_dir / "baselines.png")
        except Exception as exc:  # pragma: no cover - a plot never fails a run
            log(f"  ⚠ plot skipped: {exc}")
    log(f"== seed {seed} done in {manifest['wall_clock_s']:.0f} s")
    return manifest


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def _refusal_row(r: dict) -> dict:
    """⭐ R27.4: BY KIND always, both columns, pooled as a companion only.

    ⛔ The contaminated (``*_vs_B``) and A7-controlled columns are separate keys and
    are never averaged together — a pooled-only row hides that neither rival covers
    both kinds, which is the differentiator the cell exists to show.
    """
    raw = r.get("auroc_c_vs_b_by_kind") or {}
    ctrl = (r.get("a7_controlled") or {}).get("auroc_by_kind") or {}
    row = {
        "auroc_c_vs_b": r["auroc_c_vs_b"],  # ⛔ POOLED COMPANION, contaminated on dup
        "auroc_c_vs_a": r["auroc_c_vs_a"],
        "auroc_c_vs_a_union_b": r["auroc_c_vs_a_union_b"],
    }
    for kind, v in raw.items():
        if "auroc" in v:
            row[f"CONTAMINATED_{kind}"] = v["auroc"]
    for kind, v in ctrl.items():
        if "auroc" in v:
            row[f"A7_{kind}"] = v["auroc"]
    pooled = (r.get("a7_controlled") or {}).get("pooled_COMPANION") or {}
    if "value" in pooled:
        row["A7_pooled_COMPANION"] = pooled["value"]
    per_ctrl = (r.get("a7_controlled") or {}).get("auroc_by_kind_PER_CONTROL") or {}
    for kind, v in (per_ctrl.get("A7D") or {}).items():
        if isinstance(v, dict) and "auroc" in v:
            row[f"A7D_SENSITIVITY_{kind}"] = v["auroc"]
    return row


def _arm_rows(man: dict) -> dict:
    """Flatten one seed's manifest to ``{label: {metric: value}}``."""
    out = {}
    knn_arm = man["arms"].get("knn")
    if knn_arm:
        for vname, v in knn_arm["variants"].items():
            out[vname] = {
                "in_corner_balacc": v["in_corner_discrimination"]["balanced_accuracy"],
                "in_corner_auroc": v["in_corner_discrimination"]["auroc"],
                "B_aggregate": v["composition_B"]["aggregate"],
                "B_compositional": v["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR],
                "A_aggregate": v["composition_A"]["aggregate"],
                **_refusal_row(v["refusal"]),
            }
        ob = knn_arm["tuning"]["ORACLE_UPPER_BOUND_auroc_c_vs_b"]
        out["knn"]["ORACLE_auroc_c_vs_b"] = ob["value"]
        out["knn"]["ORACLE_auroc_c_vs_a_union_b"] = ob["max_auroc_c_vs_a_union_b_over_all_trials"]
    lace_arm = man["arms"].get("lace")
    if lace_arm:
        for sname, s in lace_arm["samplers"].items():
            out[f"lace_{sname}"] = {
                "in_corner_balacc": s["in_corner_discrimination"]["balanced_accuracy"],
                "in_corner_auroc": s["in_corner_discrimination"]["auroc"],
                "B_aggregate": s["composition_B"]["aggregate"],
                "B_compositional": s["composition_B"]["per_factor"][COMPOSITIONAL_FACTOR],
                "A_aggregate": s["composition_A"]["aggregate"],
                **_refusal_row(s["refusal"]),
            }
        ob = lace_arm["tuning"]["ORACLE_UPPER_BOUND_auroc_c_vs_b"]
        lbl = f"lace_{lace_arm['best_sampler_composition']}"
        out[lbl]["ORACLE_auroc_c_vs_b"] = ob["value"]
        out[lbl]["ORACLE_auroc_c_vs_a_union_b"] = ob["max_auroc_c_vs_a_union_b_over_all_trials"]
    out["_ceiling_decoder"] = {
        "in_corner_balacc": man["ceilings"]["decoder_ceiling"]["in_corner_discrimination"]["balanced_accuracy"],
        "in_corner_auroc": man["ceilings"]["decoder_ceiling"]["in_corner_discrimination"]["auroc"],
        "A_aggregate": man["ceilings"]["decoder_ceiling"]["composition_A"]["aggregate"],
    }
    return out


def aggregate_runs(run_dirs, log: Callable = print) -> dict:
    mans = []
    for d in run_dirs:
        p = Path(d) / "baselines_manifest.json"
        if not p.exists():
            log(f"  ⚠ {p} missing — reported as NOT RUN, ⛔ never as a zero")
            continue
        with open(p) as fh:
            mans.append(json.load(fh))
    if not mans:
        raise ValueError("no per-seed runs to aggregate")
    per_seed = [_arm_rows(m) for m in mans]
    labels = sorted({k for r in per_seed for k in r})
    summary = {}
    for label in labels:
        keys = sorted({k for r in per_seed if label in r for k in r[label]})
        summary[label] = {
            k: _mean_std([r[label][k] for r in per_seed if label in r and k in r[label]])
            for k in keys
        }
    return {
        "schema": "chlu-gen-baselines-aggregate/1",
        "dataset": mans[0]["dataset"], "pretext": mans[0]["pretext"],
        "condition": mans[0].get("condition"),
        "seeds": [m["seed"] for m in mans],
        "n_seeds": len(mans),
        "decoder": mans[0]["provenance"]["decoder"],
        "store_ceiling_balanced_accuracy": mans[0]["ceilings"]["store_ceiling_balanced_accuracy"],
        "tuning_budget": {
            a: mans[0]["arms"][a]["tuning"]["budget"] for a in mans[0]["arms"]
        },
        "byte_target": mans[0]["provenance"].get("byte_target"),
        "registered_controls": mans[0]["provenance"].get("registered_controls"),
        "byte_ledger": mans[0]["arms"].get("knn", {}).get("byte_ledger"),
        "prototype_table_ledger": mans[0]["arms"].get("knn", {}).get("prototype_table_ledger"),
        "summary": summary,
        "NO_KILL_RULE_VERDICT": mans[0]["NO_KILL_RULE_VERDICT"],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_seed(man: dict, path: Path) -> None:
    rows = _arm_rows(man)
    labels = [k for k in rows if not k.startswith("_")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ceil = rows["_ceiling_decoder"]["in_corner_balacc"]
    store_ceil = man["ceilings"]["store_ceiling_balanced_accuracy"]
    vals = [rows[k].get("in_corner_balacc", np.nan) for k in labels]
    axes[0].bar(range(len(labels)), vals, color="tab:blue")
    axes[0].axhline(ceil, ls="--", c="k", label=f"decoder ceiling {ceil:.3f}")
    axes[0].axhline(store_ceil, ls=":", c="tab:red", label=f"STORE ceiling {store_ceil:.3f}")
    axes[0].axhline(0.5, ls="-", c="grey", lw=0.8, label="chance 0.5")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("in-corner shape balanced accuracy")
    axes[0].set_title(f"composition (seed {man['seed']}) — BOTH ceilings")
    axes[0].legend(fontsize=6)
    vals = [rows[k].get("auroc_c_vs_b", np.nan) for k in labels]
    axes[1].bar(range(len(labels)), vals, color="tab:green")
    axes[1].axhline(0.5, ls="-", c="grey", lw=0.8, label="chance 0.5")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[1].set_ylabel("AUROC(B vs C)")
    axes[1].set_title("refusal — PRIMARY metric")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_experiment_gen_baselines(
    config: Optional[CHLUConfig] = None, dataset: Optional[str] = None,
    pretext: Optional[str] = None, out_dir: str | Path = "results/gen_baselines",
    seeds=None, arm: str = "all", quick: bool = False, log: Callable = print,
) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    g = config.gen_baselines
    dataset = dataset or g.dataset
    pretext = pretext or g.pretext
    out = Path(out_dir) / artifacts_tag(dataset, pretext, condition_of(config))
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = []
    for seed in (seeds if seeds is not None else list(g.seeds)):
        d = out / f"seed_{int(seed)}"
        run_seed(dataset, pretext, int(seed), d, config, arm=arm, log=log)
        run_dirs.append(d)
        # ⛔ Written after EVERY seed: a crash at hour 6 must not lose hours 1-5.
        try:
            _write_json(aggregate_runs(run_dirs, log), out / "baselines_results.json")
        except Exception as exc:  # pragma: no cover
            log(f"  ⚠ incremental aggregate skipped: {exc}")
    return aggregate_runs(run_dirs, log)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chlu.experiments.exp_gen_baselines",
        description="the two competitor arms: byte-matched kNN, and LACE",
    )
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument("--stage", default="baselines", choices=list(STAGES) + list(NOT_MINE))
    p.add_argument("--arm", default="all", choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="artifacts directory")
    p.add_argument("--condition", default=None,
                   help="hold-out condition (default: gen_data.condition). ⛔ A non-default "
                        "condition is a SEPARATE full configuration; never pooled.")
    p.add_argument("--quick", action="store_true", help="plumbing smoke; ⛔ NOT a result")
    p.add_argument("--bridge-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--decoder-source", default=None, choices=["decoder_lift_v1", "bridge"])
    p.add_argument("--store-bytes", type=int, default=None)
    p.add_argument(
        "--byte-budget", default="deployed", choices=["deployed", "historical"],
        help="⭐ `deployed` (default) = 524,288 B, the store alone, per R26 ruling 3. "
             "⛔ `historical` = 524,288 + 163,840 coupling = 688,128 B: the SUPERSEDED "
             "budget the 2026-09-12 run used, kept only so that run reproduces "
             "bit-identically. It hands the kNN arm a ~31 %% byte advantage.",
    )
    p.add_argument(
        "--expect-byte-target", type=int, default=None,
        help="assert the realised byte-match target equals this (⛔ -1 disables the "
             "assertion). Default: gen_baselines.expect_byte_target_bytes.",
    )
    p.add_argument("--n-fit", type=int, default=None)
    p.add_argument("--runs", nargs="+", default=None, help="per-seed dirs for --stage aggregate")
    p.add_argument("--no-plots", action="store_true")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    if args.quick:
        config = apply_quick(config)
    g = config.gen_baselines
    if args.data_root:
        config.gen_data.data_root = args.data_root
    if args.condition:
        config.gen_data.condition = args.condition
    if args.bridge_root:
        g.bridge_root = args.bridge_root
        config.gen_store.bridge_root = args.bridge_root
    if args.decoder_source:
        g.decoder_source = args.decoder_source
    if args.store_bytes is not None:
        g.store_bytes = int(args.store_bytes)
    if getattr(args, "byte_budget", "deployed") == "historical":
        # ⛔ Deliberate reproduction of the SUPERSEDED target, never an inherited default.
        g.include_coupling_bytes = True
        g.expect_byte_target_bytes = None
    if getattr(args, "expect_byte_target", None) is not None:
        g.expect_byte_target_bytes = (None if int(args.expect_byte_target) < 0
                                      else int(args.expect_byte_target))
    if args.n_fit is not None:
        g.n_fit = int(args.n_fit)
    if args.no_plots:
        g.make_plots = False
    config.gen_store.bridge_root = g.bridge_root or config.gen_store.bridge_root
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.stage in NOT_MINE:
        print(
            f"[gen-baselines] stage={args.stage} is not this module's "
            "(encoder/decoder -> exp_gen_bridge, store -> exp_gen_store, cell -> "
            "exp_gen_cell). Refusing to no-op.",
            file=sys.stderr,
        )
        return 2
    config = _config_from_args(args)
    g = config.gen_baselines
    dataset = args.dataset or g.dataset
    pretext = args.pretext or g.pretext
    tag = artifacts_tag(dataset, pretext, condition_of(config))
    out = Path(args.out or (Path("results") / "gen_baselines" / tag))
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[gen-baselines] dataset={dataset} pretext={pretext} condition={condition_of(config)} "
        f"stage={args.stage} arm={args.arm} seed={args.seed} out={out}"
    )
    print(
        "[gen-baselines] ⛔ no CLU number and no kill-rule verdict is produced here; "
        "⭐ these arms are built to WIN if they can."
    )
    if args.stage == "baselines":
        run_seed(dataset, pretext, int(args.seed), out, config, arm=args.arm)
    if args.stage == "aggregate":
        if not args.runs:
            print("[gen-baselines] --stage aggregate needs --runs", file=sys.stderr)
            return 2
        _write_json(aggregate_runs(args.runs), out / "baselines_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
